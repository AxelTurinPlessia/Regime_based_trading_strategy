from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd

from regime_hmm import build_market_hmm_features, fit_oos_hmm
from regime_rules import build_regime_path_from_probability
from strategy_primitives import mean_reversion_returns_with_exit, momentum_sma_crossover


TRADING_DAYS_PER_YEAR = 252.0
StrategyName = Literal["momentum", "mean_reversion"]


@dataclass(frozen=True)
class RegimeStrategyConfig:
    split_date: str | pd.Timestamp
    regime_probability_threshold: float = 0.90
    regime_min_days: int | None = None
    transaction_cost_bps: float = 20.0
    realized_vol_window: int = 21
    momentum_fast_window: int = 20
    momentum_slow_window: int = 50
    mean_reversion_window: int = 5
    mean_reversion_theta: float = 1.0
    use_log_returns: bool = False
    liquidate_on_regime_change: bool = True


def apply_low_vol_sentiment_overlay(
    base_position: pd.Series,
    sentiment_signal: pd.Series | None = None,
) -> pd.Series:
    """
    Placeholder hook for future sentiment integration.

    Current behavior intentionally leaves the low-volatility position unchanged.
    Keeping the hook here lets the notebook call a stable API now and swap the
    internal implementation later once the sentiment rule is agreed.
    """
    frames = [pd.Series(base_position).rename("Base_Low_Vol_Position")]
    if sentiment_signal is not None:
        frames.append(pd.Series(sentiment_signal).rename("Sentiment_Signal"))
    aligned = pd.concat(frames, axis=1)
    return aligned["Base_Low_Vol_Position"].rename("Low_Vol_Position")


def build_strategy_leg_table(
    traded_asset_price: pd.Series,
    momentum_fast_window: int = 20,
    momentum_slow_window: int = 50,
    mean_reversion_window: int = 5,
    mean_reversion_theta: float = 1.0,
    use_log_returns: bool = False,
    sentiment_signal: pd.Series | None = None,
) -> pd.DataFrame:
    momentum_leg = momentum_sma_crossover(
        traded_asset_price,
        fast_window=momentum_fast_window,
        slow_window=momentum_slow_window,
        use_log_returns=use_log_returns,
    )
    mean_reversion_leg = mean_reversion_returns_with_exit(
        traded_asset_price,
        window=mean_reversion_window,
        theta=mean_reversion_theta,
        use_log_returns=use_log_returns,
    )

    low_vol_position = apply_low_vol_sentiment_overlay(momentum_leg["position"], sentiment_signal=sentiment_signal)

    strategy_legs = pd.concat(
        [
            momentum_leg["price"].rename("Traded_Asset_Price"),
            momentum_leg["r"].rename("Asset_Return"),
            momentum_leg["position"].rename("Momentum_Position"),
            momentum_leg["strategy_ret"].rename("Momentum_Strategy_Return"),
            low_vol_position,
            mean_reversion_leg["position"].rename("Mean_Reversion_Position"),
            mean_reversion_leg["strategy_ret"].rename("Mean_Reversion_Strategy_Return"),
        ],
        axis=1,
    ).sort_index()

    strategy_legs["High_Vol_Position"] = strategy_legs["Mean_Reversion_Position"]
    return strategy_legs


def run_strategy_for_universe(
    price_panel: pd.DataFrame,
    strategy_name: StrategyName,
    momentum_fast_window: int = 20,
    momentum_slow_window: int = 50,
    mean_reversion_window: int = 5,
    mean_reversion_theta: float = 1.0,
    use_log_returns: bool = False,
) -> dict[str, pd.DataFrame]:
    if price_panel.empty:
        raise ValueError("price_panel cannot be empty.")

    position_frames: list[pd.Series] = []
    asset_return_frames: list[pd.Series] = []
    strategy_return_frames: list[pd.Series] = []

    for ticker in price_panel.columns:
        asset_price = pd.Series(price_panel[ticker]).dropna()
        if asset_price.empty:
            continue

        if strategy_name == "momentum":
            strategy_output = momentum_sma_crossover(
                asset_price,
                fast_window=momentum_fast_window,
                slow_window=momentum_slow_window,
                use_log_returns=use_log_returns,
            )
        else:
            strategy_output = mean_reversion_returns_with_exit(
                asset_price,
                window=mean_reversion_window,
                theta=mean_reversion_theta,
                use_log_returns=use_log_returns,
            )

        position_frames.append(strategy_output["position"].rename(ticker))
        asset_return_frames.append(strategy_output["r"].rename(ticker))
        strategy_return_frames.append(strategy_output["strategy_ret"].rename(ticker))

    if not position_frames:
        raise ValueError("No asset-level strategy outputs were generated for the requested universe.")

    positions = pd.concat(position_frames, axis=1).sort_index()
    asset_returns = pd.concat(asset_return_frames, axis=1).sort_index()
    strategy_returns = pd.concat(strategy_return_frames, axis=1).sort_index()
    weights = positions.div(positions.notna().sum(axis=1).replace(0, np.nan), axis=0)

    summary = pd.Series(
        {
            "Strategy_Name": strategy_name,
            "Assets": len(positions.columns),
            "Start_Date": positions.index.min().date(),
            "End_Date": positions.index.max().date(),
        }
    )

    return {
        "positions": positions,
        "weights": weights,
        "asset_returns": asset_returns,
        "strategy_returns": strategy_returns,
        "summary": summary,
    }


def build_equal_weight_universe_weights(
    asset_returns: pd.DataFrame,
    exposure: float = 1.0,
) -> pd.DataFrame:
    if asset_returns.empty:
        raise ValueError("asset_returns cannot be empty.")

    availability = asset_returns.notna().astype(float)
    asset_counts = availability.sum(axis=1).replace(0.0, np.nan)
    return availability.div(asset_counts, axis=0).mul(exposure).fillna(0.0)


def build_regime_decision_table(
    hmm_forecasts: pd.DataFrame,
    threshold: float = 0.90,
    min_days: int | None = None,
) -> pd.DataFrame:
    if "Next_Day_Forecast_Prob_High_Vol" not in hmm_forecasts.columns:
        raise KeyError("HMM forecast table must contain 'Next_Day_Forecast_Prob_High_Vol'.")

    decision_prob_high_vol = hmm_forecasts["Next_Day_Forecast_Prob_High_Vol"].rename("Decision_Prob_High_Vol")
    decision_regime = build_regime_path_from_probability(
        decision_prob_high_vol,
        threshold=threshold,
        min_days=min_days,
    ).rename("Decision_Regime")

    decision_table = pd.concat(
        [
            decision_prob_high_vol,
            (1.0 - decision_prob_high_vol).rename("Decision_Prob_Low_Vol"),
            decision_regime,
        ],
        axis=1,
    )
    decision_table["Decision_Confidence"] = decision_table[
        ["Decision_Prob_Low_Vol", "Decision_Prob_High_Vol"]
    ].max(axis=1)
    decision_table["Decision_Regime_Change"] = decision_table["Decision_Regime"].ne(
        decision_table["Decision_Regime"].shift()
    )
    if not decision_table.empty:
        decision_table.iloc[0, decision_table.columns.get_loc("Decision_Regime_Change")] = False
    return decision_table


def build_vix_regime_decision_table(
    vix_close: pd.Series,
    lookback_days: int = 25,
    threshold_multiplier: float = 1.10,
) -> pd.DataFrame:
    if lookback_days <= 0:
        raise ValueError("lookback_days must be positive.")
    if threshold_multiplier <= 0.0:
        raise ValueError("threshold_multiplier must be positive.")

    vix = pd.Series(vix_close).dropna().astype(float).sort_index().rename("VIX_Close")
    past_average = vix.shift(1).rolling(window=lookback_days, min_periods=lookback_days).mean().rename(
        "VIX_Past_25D_Average"
    )
    high_vol_threshold = (past_average * threshold_multiplier).rename("VIX_High_Vol_Threshold")
    decision_high_vol_flag = vix.ge(high_vol_threshold).rename("Decision_High_Vol_Flag")
    decision_regime = pd.Series(
        np.where(decision_high_vol_flag, "high_vol", "low_vol"),
        index=vix.index,
        name="Decision_Regime",
    )

    decision_table = pd.concat(
        [
            vix,
            past_average,
            high_vol_threshold,
            (vix / past_average).rename("VIX_Ratio_To_Past_Average"),
            decision_high_vol_flag,
            decision_regime,
        ],
        axis=1,
    ).dropna(subset=["VIX_Past_25D_Average", "VIX_High_Vol_Threshold"])

    decision_table["Decision_Regime_Change"] = decision_table["Decision_Regime"].ne(
        decision_table["Decision_Regime"].shift()
    )
    if not decision_table.empty:
        decision_table.iloc[0, decision_table.columns.get_loc("Decision_Regime_Change")] = False
    return decision_table


def combine_disjoint_price_panels(
    first_panel: pd.DataFrame,
    second_panel: pd.DataFrame,
) -> pd.DataFrame:
    overlapping_columns = sorted(set(first_panel.columns) & set(second_panel.columns))
    if overlapping_columns:
        preview = ", ".join(overlapping_columns[:10])
        raise ValueError(
            "Expected disjoint ticker sets when combining the two universes. "
            f"Found overlapping tickers: {preview}."
        )
    return pd.concat([first_panel, second_panel], axis=1).sort_index()


def build_buy_hold_strategy_for_universe(price_panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if price_panel.empty:
        raise ValueError("price_panel cannot be empty.")

    asset_returns = price_panel.sort_index().pct_change(fill_method=None)
    positions = asset_returns.notna().astype(float)
    weights = build_equal_weight_universe_weights(asset_returns, exposure=1.0)
    strategy_returns = (weights.shift(1).fillna(0.0) * asset_returns.fillna(0.0)).sum(axis=1).rename("Buy_Hold_Return")

    summary = pd.Series(
        {
            "Strategy_Name": "buy_hold",
            "Assets": len(price_panel.columns),
            "Start_Date": price_panel.index.min().date(),
            "End_Date": price_panel.index.max().date(),
        }
    )

    return {
        "positions": positions,
        "weights": weights,
        "asset_returns": asset_returns,
        "strategy_returns": strategy_returns,
        "summary": summary,
    }


def backtest_regime_switched_strategy(
    regime_decisions: pd.DataFrame,
    strategy_legs: pd.DataFrame,
    transaction_cost_bps: float = 20.0,
) -> pd.DataFrame:
    aligned = pd.concat([regime_decisions, strategy_legs], axis=1).sort_index()
    aligned["Live_Regime"] = aligned["Decision_Regime"].shift(1)
    aligned["Live_Momentum_Position"] = aligned["Low_Vol_Position"].shift(1)
    aligned["Live_Mean_Reversion_Position"] = aligned["High_Vol_Position"].shift(1)
    aligned = aligned.dropna(
        subset=[
            "Asset_Return",
            "Live_Regime",
            "Live_Momentum_Position",
            "Live_Mean_Reversion_Position",
        ]
    ).copy()

    aligned["Active_Strategy"] = np.where(
        aligned["Live_Regime"].eq("high_vol"),
        "mean_reversion",
        "momentum",
    )
    aligned["Combined_Position"] = np.where(
        aligned["Live_Regime"].eq("high_vol"),
        aligned["Live_Mean_Reversion_Position"],
        aligned["Live_Momentum_Position"],
    )
    aligned["Gross_Strategy_Return"] = aligned["Combined_Position"] * aligned["Asset_Return"]
    aligned["Previous_Combined_Position"] = aligned["Combined_Position"].shift().fillna(0.0)
    aligned["Turnover"] = (aligned["Combined_Position"] - aligned["Previous_Combined_Position"]).abs()
    aligned["Transaction_Cost"] = aligned["Turnover"] * transaction_cost_bps / 10000.0
    aligned["Net_Strategy_Return"] = aligned["Gross_Strategy_Return"] - aligned["Transaction_Cost"]
    aligned["Buy_Hold_Return"] = aligned["Asset_Return"]

    aligned["Regime_Change_Executed"] = aligned["Live_Regime"].ne(aligned["Live_Regime"].shift())
    if not aligned.empty:
        aligned.iloc[0, aligned.columns.get_loc("Regime_Change_Executed")] = False

    aligned["Cum_Buy_Hold"] = (1.0 + aligned["Buy_Hold_Return"].fillna(0.0)).cumprod()
    aligned["Cum_Gross_Strategy"] = (1.0 + aligned["Gross_Strategy_Return"].fillna(0.0)).cumprod()
    aligned["Cum_Net_Strategy"] = (1.0 + aligned["Net_Strategy_Return"].fillna(0.0)).cumprod()
    return aligned


def backtest_cross_sectional_regime_strategy(
    regime_decisions: pd.DataFrame,
    low_vol_weights: pd.DataFrame,
    low_vol_asset_returns: pd.DataFrame,
    high_vol_weights: pd.DataFrame,
    high_vol_asset_returns: pd.DataFrame,
    transaction_cost_bps: float = 20.0,
    liquidate_on_regime_change: bool = True,
    low_vol_strategy_label: str = "momentum_small_caps",
    high_vol_strategy_label: str = "mean_reversion_large_caps",
) -> pd.DataFrame:
    if regime_decisions.empty:
        raise ValueError("regime_decisions cannot be empty.")

    all_columns = sorted(
        set(low_vol_weights.columns)
        | set(low_vol_asset_returns.columns)
        | set(high_vol_weights.columns)
        | set(high_vol_asset_returns.columns)
    )
    all_index = regime_decisions.index

    low_weights = low_vol_weights.reindex(index=all_index, columns=all_columns)
    high_weights = high_vol_weights.reindex(index=all_index, columns=all_columns)
    low_returns = low_vol_asset_returns.reindex(index=all_index, columns=all_columns)
    high_returns = high_vol_asset_returns.reindex(index=all_index, columns=all_columns)

    live_regime = regime_decisions["Decision_Regime"].shift(1).rename("Live_Regime")
    live_low_weights = low_weights.shift(1)
    live_high_weights = high_weights.shift(1)

    combined_weights = pd.DataFrame(0.0, index=all_index, columns=all_columns)
    low_mask = live_regime.eq("low_vol")
    high_mask = live_regime.eq("high_vol")
    combined_weights.loc[low_mask] = live_low_weights.loc[low_mask].fillna(0.0)
    combined_weights.loc[high_mask] = live_high_weights.loc[high_mask].fillna(0.0)

    union_asset_returns = low_returns.combine_first(high_returns).fillna(0.0)
    gross_strategy_return = (combined_weights * union_asset_returns).sum(axis=1).rename("Gross_Strategy_Return")

    previous_combined_weights = combined_weights.shift().fillna(0.0)
    turnover = (combined_weights - previous_combined_weights).abs().sum(axis=1).rename("Turnover")
    regime_change_executed = live_regime.ne(live_regime.shift()).fillna(False).rename("Regime_Change_Executed")

    if liquidate_on_regime_change:
        liquidate_then_reopen_turnover = (
            previous_combined_weights.abs().sum(axis=1) + combined_weights.abs().sum(axis=1)
        )
        turnover = turnover.where(~regime_change_executed, liquidate_then_reopen_turnover).rename("Turnover")

    if not turnover.empty:
        turnover.iloc[0] = combined_weights.iloc[0].abs().sum()
    transaction_cost = (turnover * transaction_cost_bps / 10000.0).rename("Transaction_Cost")
    net_strategy_return = (gross_strategy_return - transaction_cost).rename("Net_Strategy_Return")

    low_leg_return = (live_low_weights.fillna(0.0) * low_returns.fillna(0.0)).sum(axis=1).rename("Low_Vol_Return")
    high_leg_return = (live_high_weights.fillna(0.0) * high_returns.fillna(0.0)).sum(axis=1).rename(
        "High_Vol_Return"
    )
    buy_hold_all_stocks = union_asset_returns.mean(axis=1).rename("Buy_Hold_All_Stocks")
    buy_hold_small_caps = low_returns.mean(axis=1).rename("Buy_Hold_Small_Caps")
    buy_hold_large_caps = high_returns.mean(axis=1).rename("Buy_Hold_Large_Caps")
    active_strategy = pd.Series(
        np.where(live_regime.eq("high_vol"), high_vol_strategy_label, low_vol_strategy_label),
        index=all_index,
        name="Active_Strategy",
    )

    backtest = pd.concat(
        [
            regime_decisions,
            live_regime,
            active_strategy,
            gross_strategy_return,
            transaction_cost,
            net_strategy_return,
            turnover,
            regime_change_executed,
            low_leg_return,
            high_leg_return,
            buy_hold_all_stocks,
            buy_hold_small_caps,
            buy_hold_large_caps,
        ],
        axis=1,
    ).dropna(subset=["Live_Regime", "Gross_Strategy_Return", "Net_Strategy_Return"])

    backtest["Gross_Exposure"] = combined_weights.loc[backtest.index].abs().sum(axis=1)
    backtest["Net_Exposure"] = combined_weights.loc[backtest.index].sum(axis=1)
    backtest["Cum_Gross_Strategy"] = (1.0 + backtest["Gross_Strategy_Return"].fillna(0.0)).cumprod()
    backtest["Cum_Net_Strategy"] = (1.0 + backtest["Net_Strategy_Return"].fillna(0.0)).cumprod()
    backtest["Cum_Buy_Hold_All_Stocks"] = (1.0 + backtest["Buy_Hold_All_Stocks"].fillna(0.0)).cumprod()
    backtest["Cum_Buy_Hold_Small_Caps"] = (1.0 + backtest["Buy_Hold_Small_Caps"].fillna(0.0)).cumprod()
    backtest["Cum_Buy_Hold_Large_Caps"] = (1.0 + backtest["Buy_Hold_Large_Caps"].fillna(0.0)).cumprod()
    return backtest


def backtest_regime_timed_directional_benchmark(
    regime_decisions: pd.DataFrame,
    low_vol_asset_returns: pd.DataFrame,
    high_vol_asset_returns: pd.DataFrame,
    transaction_cost_bps: float = 20.0,
    liquidate_on_regime_change: bool = True,
) -> pd.DataFrame:
    low_vol_weights = build_equal_weight_universe_weights(low_vol_asset_returns, exposure=1.0)
    high_vol_weights = build_equal_weight_universe_weights(high_vol_asset_returns, exposure=-1.0)

    backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=low_vol_weights,
        low_vol_asset_returns=low_vol_asset_returns,
        high_vol_weights=high_vol_weights,
        high_vol_asset_returns=high_vol_asset_returns,
        transaction_cost_bps=transaction_cost_bps,
        liquidate_on_regime_change=liquidate_on_regime_change,
        low_vol_strategy_label="buy_hold_small_caps",
        high_vol_strategy_label="short_large_caps",
    )
    backtest["Benchmark_Name"] = "long_small_caps_low_vol_short_large_caps_high_vol"
    return backtest


def build_equal_weight_buy_hold_return(
    price_panel: pd.DataFrame,
    name: str,
) -> pd.Series:
    if price_panel.empty:
        raise ValueError("price_panel cannot be empty.")
    return price_panel.sort_index().pct_change().mean(axis=1).rename(name)


def build_equal_weight_universe_price_index(
    price_panel: pd.DataFrame,
    name: str,
    base_value: float = 1.0,
) -> pd.Series:
    universe_return = build_equal_weight_buy_hold_return(price_panel, name=f"{name}_Return")
    return (1.0 + universe_return.fillna(0.0)).cumprod().mul(base_value).rename(name)


def summarize_return_stream(
    return_series: pd.Series,
    cumulative_curve: pd.Series,
    strategy_name: str,
    turnover: pd.Series | None = None,
    transaction_cost: pd.Series | None = None,
) -> pd.Series:
    cleaned_returns = pd.Series(return_series).dropna()
    cleaned_curve = pd.Series(cumulative_curve).dropna()
    if cleaned_returns.empty or cleaned_curve.empty:
        raise ValueError("Performance inputs cannot be empty.")

    return pd.Series(
        {
            "Strategy": strategy_name,
            "Start_Date": cleaned_returns.index.min().date(),
            "End_Date": cleaned_returns.index.max().date(),
            "Observations": len(cleaned_returns),
            "Total_Return": cleaned_curve.iloc[-1] - 1.0,
            "Annualized_Return": annualized_return(cleaned_returns),
            "Annualized_Vol": annualized_volatility(cleaned_returns),
            "Sharpe": sharpe_ratio(cleaned_returns),
            "Max_Drawdown": max_drawdown(cleaned_curve),
            "Average_Daily_Turnover": pd.Series(turnover).dropna().mean() if turnover is not None else np.nan,
            "Total_Transaction_Cost": pd.Series(transaction_cost).dropna().sum() if transaction_cost is not None else 0.0,
        }
    )


def run_regime_gated_strategy_comparison(
    low_vol_price_panel: pd.DataFrame,
    high_vol_price_panel: pd.DataFrame,
    regime_decisions: pd.DataFrame,
    transaction_cost_bps: float = 10.0,
    momentum_fast_window: int = 50,
    momentum_slow_window: int = 200,
    mean_reversion_window: int = 5,
    mean_reversion_theta: float = 2.0,
    use_log_returns: bool = False,
    liquidate_on_regime_change: bool = True,
) -> dict[str, Any]:
    small_cap_basket = build_equal_weight_universe_price_index(low_vol_price_panel, name="Small_Caps").to_frame()
    large_cap_basket = build_equal_weight_universe_price_index(high_vol_price_panel, name="Large_Caps").to_frame()

    momentum_strategy = run_strategy_for_universe(
        price_panel=small_cap_basket,
        strategy_name="momentum",
        momentum_fast_window=momentum_fast_window,
        momentum_slow_window=momentum_slow_window,
        mean_reversion_window=mean_reversion_window,
        mean_reversion_theta=mean_reversion_theta,
        use_log_returns=use_log_returns,
    )
    zero_large_cap_weights = pd.DataFrame(
        0.0,
        index=large_cap_basket.index,
        columns=large_cap_basket.columns,
    )

    mean_reversion_strategy = run_strategy_for_universe(
        price_panel=large_cap_basket,
        strategy_name="mean_reversion",
        momentum_fast_window=momentum_fast_window,
        momentum_slow_window=momentum_slow_window,
        mean_reversion_window=mean_reversion_window,
        mean_reversion_theta=mean_reversion_theta,
        use_log_returns=use_log_returns,
    )
    zero_small_cap_weights = pd.DataFrame(
        0.0,
        index=small_cap_basket.index,
        columns=small_cap_basket.columns,
    )

    momentum_backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=momentum_strategy["weights"],
        low_vol_asset_returns=momentum_strategy["asset_returns"],
        high_vol_weights=zero_large_cap_weights,
        high_vol_asset_returns=mean_reversion_strategy["asset_returns"],
        transaction_cost_bps=transaction_cost_bps,
        liquidate_on_regime_change=liquidate_on_regime_change,
        low_vol_strategy_label="momentum_small_cap_basket",
        high_vol_strategy_label="flat_high_vol",
    )
    mean_reversion_backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=zero_small_cap_weights,
        low_vol_asset_returns=momentum_strategy["asset_returns"],
        high_vol_weights=mean_reversion_strategy["weights"],
        high_vol_asset_returns=mean_reversion_strategy["asset_returns"],
        transaction_cost_bps=transaction_cost_bps,
        liquidate_on_regime_change=liquidate_on_regime_change,
        low_vol_strategy_label="flat_low_vol",
        high_vol_strategy_label="mean_reversion_large_cap_basket",
    )

    comparison_index = momentum_backtest.index.union(mean_reversion_backtest.index)
    buy_hold_small_caps = build_equal_weight_buy_hold_return(
        low_vol_price_panel,
        name="Buy_Hold_Small_Caps",
    ).reindex(comparison_index)
    buy_hold_large_caps = build_equal_weight_buy_hold_return(
        high_vol_price_panel,
        name="Buy_Hold_Large_Caps",
    ).reindex(comparison_index)

    comparison_curves = pd.concat(
        [
            momentum_backtest["Cum_Net_Strategy"].rename("Cum_Momentum_Low_Vol_Only"),
            mean_reversion_backtest["Cum_Net_Strategy"].rename("Cum_Mean_Reversion_High_Vol_Only"),
            (1.0 + buy_hold_small_caps.fillna(0.0)).cumprod().rename("Cum_Buy_Hold_Small_Caps"),
            (1.0 + buy_hold_large_caps.fillna(0.0)).cumprod().rename("Cum_Buy_Hold_Large_Caps"),
        ],
        axis=1,
    ).dropna(how="all")

    summary = pd.DataFrame(
        [
            summarize_return_stream(
                momentum_backtest["Net_Strategy_Return"],
                momentum_backtest["Cum_Net_Strategy"],
                strategy_name="Momentum_Small_Cap_Basket_Low_Vol_Only",
                turnover=momentum_backtest["Turnover"],
                transaction_cost=momentum_backtest["Transaction_Cost"],
            ),
            summarize_return_stream(
                mean_reversion_backtest["Net_Strategy_Return"],
                mean_reversion_backtest["Cum_Net_Strategy"],
                strategy_name="Mean_Reversion_Large_Cap_Basket_High_Vol_Only",
                turnover=mean_reversion_backtest["Turnover"],
                transaction_cost=mean_reversion_backtest["Transaction_Cost"],
            ),
            summarize_return_stream(
                buy_hold_small_caps,
                comparison_curves["Cum_Buy_Hold_Small_Caps"],
                strategy_name="Buy_Hold_Small_Caps",
            ),
            summarize_return_stream(
                buy_hold_large_caps,
                comparison_curves["Cum_Buy_Hold_Large_Caps"],
                strategy_name="Buy_Hold_Large_Caps",
            ),
        ]
    ).set_index("Strategy")

    return {
        "config": {
            "transaction_cost_bps": transaction_cost_bps,
            "momentum_fast_window": momentum_fast_window,
            "momentum_slow_window": momentum_slow_window,
            "mean_reversion_window": mean_reversion_window,
            "mean_reversion_theta": mean_reversion_theta,
            "use_log_returns": use_log_returns,
            "liquidate_on_regime_change": liquidate_on_regime_change,
            "comparison_scope": "momentum_small_cap_basket_and_mean_reversion_large_cap_basket",
        },
        "momentum_strategy": momentum_strategy,
        "mean_reversion_strategy": mean_reversion_strategy,
        "momentum_backtest": momentum_backtest,
        "mean_reversion_backtest": mean_reversion_backtest,
        "buy_hold_small_caps": buy_hold_small_caps,
        "buy_hold_large_caps": buy_hold_large_caps,
        "comparison_curves": comparison_curves,
        "summary": summary,
    }


def summarize_transaction_cost_scenario(
    backtest: pd.DataFrame,
    transaction_cost_bps: float,
) -> pd.Series:
    net_strategy_return = backtest["Gross_Strategy_Return"] - backtest["Turnover"] * transaction_cost_bps / 10000.0
    cum_net_strategy = (1.0 + net_strategy_return.fillna(0.0)).cumprod()

    return pd.Series(
        {
            "Transaction_Cost_Bps": transaction_cost_bps,
            "Net_Total_Return": cum_net_strategy.iloc[-1] - 1.0,
            "Net_Annualized_Return": annualized_return(net_strategy_return),
            "Net_Annualized_Vol": annualized_volatility(net_strategy_return),
            "Net_Sharpe": sharpe_ratio(net_strategy_return),
            "Net_Max_Drawdown": max_drawdown(cum_net_strategy),
            "Profitable_On_Total_Return": bool(cum_net_strategy.iloc[-1] > 1.0),
            "Profitable_On_Annualized_Return": bool(annualized_return(net_strategy_return) > 0.0),
        }
    )


def build_transaction_cost_sweep(
    backtest: pd.DataFrame,
    transaction_cost_grid_bps: list[float] | np.ndarray | pd.Index,
) -> pd.DataFrame:
    scenarios = [
        summarize_transaction_cost_scenario(backtest, transaction_cost_bps=float(cost_bps))
        for cost_bps in transaction_cost_grid_bps
    ]
    return pd.DataFrame(scenarios)


def estimate_break_even_transaction_cost(
    backtest: pd.DataFrame,
    profitability_metric: Literal["annualized_return", "total_return"] = "annualized_return",
    upper_bound_bps: float = 1000.0,
    tolerance_bps: float = 0.01,
    max_iter: int = 60,
) -> float:
    def objective(cost_bps: float) -> float:
        summary = summarize_transaction_cost_scenario(backtest, transaction_cost_bps=cost_bps)
        if profitability_metric == "total_return":
            return float(summary["Net_Total_Return"])
        return float(summary["Net_Annualized_Return"])

    lower = 0.0
    upper = float(upper_bound_bps)
    lower_value = objective(lower)
    if lower_value <= 0.0:
        return 0.0

    upper_value = objective(upper)
    while upper_value > 0.0 and upper < 100000.0:
        upper *= 2.0
        upper_value = objective(upper)

    if upper_value > 0.0:
        return np.nan

    for _ in range(max_iter):
        midpoint = 0.5 * (lower + upper)
        midpoint_value = objective(midpoint)
        if midpoint_value > 0.0:
            lower = midpoint
        else:
            upper = midpoint
        if upper - lower <= tolerance_bps:
            break

    return upper


def annualized_return(return_series: pd.Series) -> float:
    cleaned = pd.Series(return_series).dropna()
    if cleaned.empty:
        return np.nan
    cumulative_return = (1.0 + cleaned).prod()
    return float(cumulative_return ** (TRADING_DAYS_PER_YEAR / len(cleaned)) - 1.0)


def annualized_volatility(return_series: pd.Series) -> float:
    cleaned = pd.Series(return_series).dropna()
    if cleaned.empty:
        return np.nan
    return float(cleaned.std(ddof=0) * np.sqrt(TRADING_DAYS_PER_YEAR))


def sharpe_ratio(return_series: pd.Series) -> float:
    ann_vol = annualized_volatility(return_series)
    if ann_vol == 0 or np.isnan(ann_vol):
        return np.nan
    return annualized_return(return_series) / ann_vol


def max_drawdown(cumulative_curve: pd.Series) -> float:
    cleaned = pd.Series(cumulative_curve).dropna()
    if cleaned.empty:
        return np.nan
    running_peak = cleaned.cummax()
    drawdown = cleaned / running_peak - 1.0
    return float(drawdown.min())


def summarize_strategy_backtest(backtest: pd.DataFrame) -> pd.Series:
    if backtest.empty:
        raise ValueError("Backtest output is empty.")

    return pd.Series(
        {
            "Start_Date": backtest.index.min().date(),
            "End_Date": backtest.index.max().date(),
            "Observations": len(backtest),
            "High_Vol_Time_Share_Pct": 100.0 * backtest["Live_Regime"].eq("high_vol").mean(),
            "Executed_Regime_Changes": int(backtest["Regime_Change_Executed"].sum()),
            "Average_Daily_Turnover": backtest["Turnover"].mean(),
            "Total_Transaction_Cost": backtest["Transaction_Cost"].sum(),
            "Gross_Total_Return": backtest["Cum_Gross_Strategy"].iloc[-1] - 1.0,
            "Net_Total_Return": backtest["Cum_Net_Strategy"].iloc[-1] - 1.0,
            "Buy_Hold_Total_Return": backtest["Cum_Buy_Hold"].iloc[-1] - 1.0,
            "Gross_Annualized_Return": annualized_return(backtest["Gross_Strategy_Return"]),
            "Net_Annualized_Return": annualized_return(backtest["Net_Strategy_Return"]),
            "Buy_Hold_Annualized_Return": annualized_return(backtest["Buy_Hold_Return"]),
            "Gross_Annualized_Vol": annualized_volatility(backtest["Gross_Strategy_Return"]),
            "Net_Annualized_Vol": annualized_volatility(backtest["Net_Strategy_Return"]),
            "Buy_Hold_Annualized_Vol": annualized_volatility(backtest["Buy_Hold_Return"]),
            "Gross_Sharpe": sharpe_ratio(backtest["Gross_Strategy_Return"]),
            "Net_Sharpe": sharpe_ratio(backtest["Net_Strategy_Return"]),
            "Buy_Hold_Sharpe": sharpe_ratio(backtest["Buy_Hold_Return"]),
            "Gross_Max_Drawdown": max_drawdown(backtest["Cum_Gross_Strategy"]),
            "Net_Max_Drawdown": max_drawdown(backtest["Cum_Net_Strategy"]),
            "Buy_Hold_Max_Drawdown": max_drawdown(backtest["Cum_Buy_Hold"]),
        }
    )


def run_regime_switching_strategy(
    traded_asset_price: pd.Series,
    market_portfolio_return: pd.Series,
    vix_close: pd.Series,
    market_volume_change: pd.Series | None = None,
    config: RegimeStrategyConfig | None = None,
    sentiment_signal: pd.Series | None = None,
) -> dict[str, Any]:
    resolved_config = config or RegimeStrategyConfig(split_date="2010-01-01")
    feature_table = build_market_hmm_features(
        portfolio_return=market_portfolio_return,
        vix_close=vix_close,
        volume_change=market_volume_change,
        realized_vol_window=resolved_config.realized_vol_window,
    )
    hmm_result = fit_oos_hmm(
        feature_table=feature_table,
        split_date=resolved_config.split_date,
        volatility_columns=["Abs_Return", "Realized_Volatility", "VIX_Close"],
    )

    regime_decisions = build_regime_decision_table(
        hmm_result["forecasts"],
        threshold=resolved_config.regime_probability_threshold,
        min_days=resolved_config.regime_min_days,
    )
    strategy_legs = build_strategy_leg_table(
        traded_asset_price=traded_asset_price,
        momentum_fast_window=resolved_config.momentum_fast_window,
        momentum_slow_window=resolved_config.momentum_slow_window,
        mean_reversion_window=resolved_config.mean_reversion_window,
        mean_reversion_theta=resolved_config.mean_reversion_theta,
        use_log_returns=resolved_config.use_log_returns,
        sentiment_signal=sentiment_signal,
    )
    backtest = backtest_regime_switched_strategy(
        regime_decisions=regime_decisions,
        strategy_legs=strategy_legs,
        transaction_cost_bps=resolved_config.transaction_cost_bps,
    )
    summary = summarize_strategy_backtest(backtest)

    return {
        "config": asdict(resolved_config),
        "feature_table": feature_table,
        "hmm": hmm_result,
        "regime_decisions": regime_decisions,
        "strategy_legs": strategy_legs,
        "backtest": backtest,
        "summary": summary,
    }


def summarize_cross_sectional_backtest(backtest: pd.DataFrame) -> pd.Series:
    if backtest.empty:
        raise ValueError("Backtest output is empty.")

    summary = {
        "Start_Date": backtest.index.min().date(),
        "End_Date": backtest.index.max().date(),
        "Observations": len(backtest),
        "High_Vol_Time_Share_Pct": 100.0 * backtest["Live_Regime"].eq("high_vol").mean(),
        "Executed_Regime_Changes": int(backtest["Regime_Change_Executed"].sum()),
        "Average_Daily_Turnover": backtest["Turnover"].mean(),
        "Total_Transaction_Cost": backtest["Transaction_Cost"].sum(),
        "Gross_Total_Return": backtest["Cum_Gross_Strategy"].iloc[-1] - 1.0,
        "Net_Total_Return": backtest["Cum_Net_Strategy"].iloc[-1] - 1.0,
        "Small_Cap_Buy_Hold_Total_Return": backtest["Cum_Buy_Hold_Small_Caps"].iloc[-1] - 1.0,
        "Large_Cap_Buy_Hold_Total_Return": backtest["Cum_Buy_Hold_Large_Caps"].iloc[-1] - 1.0,
        "Gross_Annualized_Return": annualized_return(backtest["Gross_Strategy_Return"]),
        "Net_Annualized_Return": annualized_return(backtest["Net_Strategy_Return"]),
        "Gross_Annualized_Vol": annualized_volatility(backtest["Gross_Strategy_Return"]),
        "Net_Annualized_Vol": annualized_volatility(backtest["Net_Strategy_Return"]),
        "Gross_Sharpe": sharpe_ratio(backtest["Gross_Strategy_Return"]),
        "Net_Sharpe": sharpe_ratio(backtest["Net_Strategy_Return"]),
        "Net_Max_Drawdown": max_drawdown(backtest["Cum_Net_Strategy"]),
        "Average_Gross_Exposure": backtest["Gross_Exposure"].mean(),
        "Average_Net_Exposure": backtest["Net_Exposure"].mean(),
    }
    if "Cum_Buy_Hold_All_Stocks" in backtest.columns:
        summary["All_Stock_Buy_Hold_Total_Return"] = backtest["Cum_Buy_Hold_All_Stocks"].iloc[-1] - 1.0
    return pd.Series(summary)


def run_vix_threshold_cross_sectional_strategy(
    small_cap_price_panel: pd.DataFrame,
    large_cap_price_panel: pd.DataFrame,
    vix_close: pd.Series,
    config: RegimeStrategyConfig | None = None,
    vix_lookback_days: int = 25,
    vix_threshold_multiplier: float = 1.10,
) -> dict[str, Any]:
    resolved_config = config or RegimeStrategyConfig(split_date="2010-01-01")
    all_stock_price_panel = combine_disjoint_price_panels(small_cap_price_panel, large_cap_price_panel)

    regime_decisions = build_vix_regime_decision_table(
        vix_close=vix_close,
        lookback_days=vix_lookback_days,
        threshold_multiplier=vix_threshold_multiplier,
    )
    first_asset_date = all_stock_price_panel.dropna(how="all").index.min()
    if first_asset_date is None:
        raise ValueError("The combined stock universe does not contain any usable price history.")
    regime_decisions = regime_decisions.loc[regime_decisions.index >= first_asset_date].copy()

    low_vol_strategy = run_strategy_for_universe(
        price_panel=all_stock_price_panel,
        strategy_name="momentum",
        momentum_fast_window=resolved_config.momentum_fast_window,
        momentum_slow_window=resolved_config.momentum_slow_window,
        mean_reversion_window=resolved_config.mean_reversion_window,
        mean_reversion_theta=resolved_config.mean_reversion_theta,
        use_log_returns=resolved_config.use_log_returns,
    )
    high_vol_strategy = run_strategy_for_universe(
        price_panel=all_stock_price_panel,
        strategy_name="mean_reversion",
        momentum_fast_window=resolved_config.momentum_fast_window,
        momentum_slow_window=resolved_config.momentum_slow_window,
        mean_reversion_window=resolved_config.mean_reversion_window,
        mean_reversion_theta=resolved_config.mean_reversion_theta,
        use_log_returns=resolved_config.use_log_returns,
    )

    backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=low_vol_strategy["weights"],
        low_vol_asset_returns=low_vol_strategy["asset_returns"],
        high_vol_weights=high_vol_strategy["weights"],
        high_vol_asset_returns=high_vol_strategy["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
        low_vol_strategy_label="momentum_all_stocks",
        high_vol_strategy_label="mean_reversion_all_stocks",
    )
    summary = summarize_cross_sectional_backtest(backtest)
    cost_sweep = build_transaction_cost_sweep(backtest, transaction_cost_grid_bps=np.arange(0.0, 205.0, 5.0))
    break_even_costs = pd.Series(
        {
            "Break_Even_Cost_Bps_Annualized_Return": estimate_break_even_transaction_cost(
                backtest,
                profitability_metric="annualized_return",
            ),
            "Break_Even_Cost_Bps_Total_Return": estimate_break_even_transaction_cost(
                backtest,
                profitability_metric="total_return",
            ),
        }
    )
    regime_rule_summary = pd.Series(
        {
            "Rule_Name": "VIX threshold regime rule",
            "VIX_Lookback_Days": vix_lookback_days,
            "VIX_Threshold_Multiplier": vix_threshold_multiplier,
            "VIX_Threshold_Pct_Above_Average": 100.0 * (vix_threshold_multiplier - 1.0),
            "Start_Date": regime_decisions.index.min().date(),
            "End_Date": regime_decisions.index.max().date(),
            "Observations": len(regime_decisions),
            "High_Vol_Time_Share_Pct": 100.0 * regime_decisions["Decision_Regime"].eq("high_vol").mean(),
        }
    )

    return {
        "config": {
            **asdict(resolved_config),
            "vix_lookback_days": vix_lookback_days,
            "vix_threshold_multiplier": vix_threshold_multiplier,
        },
        "regime_rule_summary": regime_rule_summary,
        "regime_decisions": regime_decisions,
        "low_vol_strategy": low_vol_strategy,
        "high_vol_strategy": high_vol_strategy,
        "backtest": backtest,
        "summary": summary,
        "transaction_cost_sweep": cost_sweep,
        "break_even_costs": break_even_costs,
    }


def run_cross_sectional_regime_switching_strategy(
    low_vol_price_panel: pd.DataFrame,
    high_vol_price_panel: pd.DataFrame,
    market_portfolio_return: pd.Series,
    vix_close: pd.Series,
    market_volume_change: pd.Series | None = None,
    config: RegimeStrategyConfig | None = None,
) -> dict[str, Any]:
    resolved_config = config or RegimeStrategyConfig(split_date="2010-01-01")

    feature_table = build_market_hmm_features(
        portfolio_return=market_portfolio_return,
        vix_close=vix_close,
        volume_change=market_volume_change,
        realized_vol_window=resolved_config.realized_vol_window,
    )
    hmm_result = fit_oos_hmm(
        feature_table=feature_table,
        split_date=resolved_config.split_date,
        volatility_columns=["Abs_Return", "Realized_Volatility", "VIX_Close"],
    )
    regime_decisions = build_regime_decision_table(
        hmm_result["forecasts"],
        threshold=resolved_config.regime_probability_threshold,
        min_days=resolved_config.regime_min_days,
    )

    low_vol_strategy = run_strategy_for_universe(
        price_panel=low_vol_price_panel,
        strategy_name="momentum",
        momentum_fast_window=resolved_config.momentum_fast_window,
        momentum_slow_window=resolved_config.momentum_slow_window,
        mean_reversion_window=resolved_config.mean_reversion_window,
        mean_reversion_theta=resolved_config.mean_reversion_theta,
        use_log_returns=resolved_config.use_log_returns,
    )
    high_vol_strategy = run_strategy_for_universe(
        price_panel=high_vol_price_panel,
        strategy_name="mean_reversion",
        momentum_fast_window=resolved_config.momentum_fast_window,
        momentum_slow_window=resolved_config.momentum_slow_window,
        mean_reversion_window=resolved_config.mean_reversion_window,
        mean_reversion_theta=resolved_config.mean_reversion_theta,
        use_log_returns=resolved_config.use_log_returns,
    )

    backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=low_vol_strategy["weights"],
        low_vol_asset_returns=low_vol_strategy["asset_returns"],
        high_vol_weights=high_vol_strategy["weights"],
        high_vol_asset_returns=high_vol_strategy["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
    )
    regime_directional_benchmark = backtest_regime_timed_directional_benchmark(
        regime_decisions=regime_decisions,
        low_vol_asset_returns=low_vol_strategy["asset_returns"],
        high_vol_asset_returns=high_vol_strategy["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
    )
    summary = summarize_cross_sectional_backtest(backtest)
    regime_directional_benchmark_summary = summarize_cross_sectional_backtest(regime_directional_benchmark)
    cost_sweep = build_transaction_cost_sweep(backtest, transaction_cost_grid_bps=np.arange(0.0, 205.0, 5.0))
    break_even_costs = pd.Series(
        {
            "Break_Even_Cost_Bps_Annualized_Return": estimate_break_even_transaction_cost(
                backtest,
                profitability_metric="annualized_return",
            ),
            "Break_Even_Cost_Bps_Total_Return": estimate_break_even_transaction_cost(
                backtest,
                profitability_metric="total_return",
            ),
        }
    )

    return {
        "config": asdict(resolved_config),
        "feature_table": feature_table,
        "hmm": hmm_result,
        "regime_decisions": regime_decisions,
        "low_vol_strategy": low_vol_strategy,
        "high_vol_strategy": high_vol_strategy,
        "backtest": backtest,
        "summary": summary,
        "regime_directional_benchmark": {
            "backtest": regime_directional_benchmark,
            "summary": regime_directional_benchmark_summary,
        },
        "transaction_cost_sweep": cost_sweep,
        "break_even_costs": break_even_costs,
    }
