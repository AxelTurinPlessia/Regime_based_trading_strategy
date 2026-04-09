from __future__ import annotations

from dataclasses import asdict, replace
from itertools import product
from typing import Any

import numpy as np
import pandas as pd

from regime_strategy import (
    RegimeStrategyConfig,
    build_vix_regime_decision_table,
    build_transaction_cost_sweep,
    build_buy_hold_strategy_for_universe,
    combine_disjoint_price_panels,
    estimate_break_even_transaction_cost,
    run_strategy_for_universe,
    backtest_cross_sectional_regime_strategy,
    summarize_cross_sectional_backtest,
)


def apply_rebalance_schedule(
    data: pd.Series | pd.DataFrame,
    rebalance_rule: str = "W-FRI",
) -> pd.Series | pd.DataFrame:
    if not isinstance(data.index, pd.DatetimeIndex):
        raise TypeError("Rebalance scheduling requires a DatetimeIndex.")

    sorted_data = data.sort_index()
    rebalanced = sorted_data.resample(rebalance_rule).last()
    scheduled = rebalanced.reindex(sorted_data.index).ffill()

    if isinstance(sorted_data, pd.Series):
        scheduled.name = sorted_data.name
    else:
        scheduled.columns = sorted_data.columns
    return scheduled


def apply_rebalance_schedule_to_strategy_output(
    strategy_output: dict[str, Any],
    rebalance_rule: str = "W-FRI",
) -> dict[str, Any]:
    positions = apply_rebalance_schedule(strategy_output["positions"], rebalance_rule=rebalance_rule)
    weights = apply_rebalance_schedule(strategy_output["weights"], rebalance_rule=rebalance_rule)
    asset_returns = strategy_output["asset_returns"].sort_index()
    strategy_return_name = getattr(strategy_output.get("strategy_returns"), "name", None) or "Strategy_Return"
    strategy_returns = (weights.shift(1).fillna(0.0) * asset_returns.fillna(0.0)).sum(axis=1).rename(
        strategy_return_name
    )

    summary = pd.Series(strategy_output["summary"]).copy()
    summary["Rebalance_Rule"] = rebalance_rule
    summary["Signal_Frequency"] = "weekly"

    rebalanced_output = dict(strategy_output)
    rebalanced_output["positions"] = positions
    rebalanced_output["weights"] = weights
    rebalanced_output["strategy_returns"] = strategy_returns
    rebalanced_output["summary"] = summary
    return rebalanced_output


def invert_strategy_output(
    strategy_output: dict[str, Any],
    strategy_name: str | None = None,
) -> dict[str, Any]:
    inverted_output = dict(strategy_output)
    inverted_output["positions"] = -strategy_output["positions"]
    inverted_output["weights"] = -strategy_output["weights"]
    strategy_return_name = getattr(strategy_output.get("strategy_returns"), "name", None) or "Strategy_Return"
    inverted_output["strategy_returns"] = (-strategy_output["strategy_returns"]).rename(strategy_return_name)

    summary = pd.Series(strategy_output["summary"]).copy()
    if strategy_name is not None:
        summary["Strategy_Name"] = strategy_name
    inverted_output["summary"] = summary
    return inverted_output


def slice_dated_data(
    data: pd.Series | pd.DataFrame,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.Series | pd.DataFrame:
    sliced = data.sort_index()
    if start_date is not None:
        sliced = sliced.loc[sliced.index >= pd.Timestamp(start_date)]
    if end_date is not None:
        sliced = sliced.loc[sliced.index <= pd.Timestamp(end_date)]
    return sliced.copy()


def build_weekly_vix_regime_decisions(
    vix_close: pd.Series,
    asset_price_panel: pd.DataFrame,
    lookback_days: int,
    threshold_multiplier: float,
    rebalance_rule: str = "W-FRI",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> pd.DataFrame:
    if asset_price_panel.empty:
        raise ValueError("asset_price_panel cannot be empty.")

    regime_decisions = build_vix_regime_decision_table(
        vix_close=vix_close,
        lookback_days=lookback_days,
        threshold_multiplier=threshold_multiplier,
    )
    regime_decisions = slice_dated_data(regime_decisions, start_date=start_date, end_date=end_date)

    first_asset_date = asset_price_panel.dropna(how="all").index.min()
    if first_asset_date is None:
        raise ValueError("The asset universe does not contain any usable price history.")
    regime_decisions = regime_decisions.loc[regime_decisions.index >= first_asset_date].copy()
    if regime_decisions.empty:
        raise ValueError("No overlapping VIX regime history was found for the requested window.")

    regime_decisions["Decision_High_Vol_Flag"] = apply_rebalance_schedule(
        regime_decisions["Decision_High_Vol_Flag"],
        rebalance_rule=rebalance_rule,
    ).astype("boolean")
    regime_decisions["Decision_Regime"] = apply_rebalance_schedule(
        regime_decisions["Decision_Regime"],
        rebalance_rule=rebalance_rule,
    )
    regime_decisions["Decision_Regime_Change"] = regime_decisions["Decision_Regime"].ne(
        regime_decisions["Decision_Regime"].shift()
    )
    regime_decisions["Decision_Regime_Change"] = regime_decisions["Decision_Regime_Change"].where(
        regime_decisions["Decision_Regime"].notna(),
        False,
    )
    if not regime_decisions.empty:
        regime_decisions.iloc[0, regime_decisions.columns.get_loc("Decision_Regime_Change")] = False
    return regime_decisions


def build_parameter_search_grid(
    parameter_grid: dict[str, list[Any]] | None = None,
) -> dict[str, list[Any]]:
    default_grid = {
        "vix_lookback_days": [10, 20, 25],
        "vix_threshold_multiplier": [1.10, 1.15, 1.20],
        "mean_reversion_window": [3, 5, 10],
        "mean_reversion_theta": [0.5, 0.75, 1.0],
    }
    resolved_grid = default_grid if parameter_grid is None else parameter_grid
    required_keys = set(default_grid)
    missing = sorted(required_keys - set(resolved_grid))
    if missing:
        raise KeyError(f"Parameter grid is missing required keys: {', '.join(missing)}.")

    normalized_grid: dict[str, list[Any]] = {}
    for key in default_grid:
        values = list(resolved_grid[key])
        if not values:
            raise ValueError(f"Parameter grid entry '{key}' cannot be empty.")
        normalized_grid[key] = values
    return normalized_grid


def score_summary_metric(summary: pd.Series, metric_name: str) -> float:
    metric_value = pd.Series(summary).get(metric_name)
    if metric_value is None or pd.isna(metric_value):
        return float("-inf")
    return float(metric_value)


def build_cross_sectional_momentum_strategy(
    price_panel: pd.DataFrame,
    formation_window: int = 252,
    skip_window: int = 21,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
) -> dict[str, pd.DataFrame]:
    if price_panel.empty:
        raise ValueError("price_panel cannot be empty.")
    if formation_window <= skip_window:
        raise ValueError("formation_window must be larger than skip_window.")
    if not (0.0 < long_quantile <= 0.5):
        raise ValueError("long_quantile must be in (0, 0.5].")
    if not (0.0 < short_quantile <= 0.5):
        raise ValueError("short_quantile must be in (0, 0.5].")

    prices = price_panel.sort_index().astype(float)
    asset_returns = prices.pct_change(fill_method=None)

    # Standard 12-1 style cross-sectional momentum:
    # rank on trailing return from t-formation_window to t-skip_window.
    score = prices.shift(skip_window).div(prices.shift(formation_window)) - 1.0
    score = score.rename_axis(index=prices.index.name, columns=prices.columns.name)

    positions = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)

    for date, row in score.iterrows():
        valid_scores = row.dropna()
        if len(valid_scores) < 2:
            continue

        long_count = max(1, int(np.floor(len(valid_scores) * long_quantile)))
        long_count = min(long_count, len(valid_scores) - 1)
        long_names = valid_scores.nlargest(long_count).index

        remaining_scores = valid_scores.drop(index=long_names, errors="ignore")
        if remaining_scores.empty:
            continue

        short_count = max(1, int(np.floor(len(valid_scores) * short_quantile)))
        short_count = min(short_count, len(remaining_scores))
        short_names = remaining_scores.nsmallest(short_count).index

        positions.loc[date, long_names] = 1.0
        positions.loc[date, short_names] = -1.0
        weights.loc[date, long_names] = 1.0 / len(long_names)
        weights.loc[date, short_names] = -1.0 / len(short_names)

    strategy_returns = (weights.shift(1).fillna(0.0) * asset_returns.fillna(0.0)).sum(axis=1).rename(
        "Cross_Sectional_Momentum_Return"
    )

    summary = pd.Series(
        {
            "Strategy_Name": "cross_sectional_momentum",
            "Assets": len(prices.columns),
            "Start_Date": prices.index.min().date(),
            "End_Date": prices.index.max().date(),
            "Formation_Window": formation_window,
            "Skip_Window": skip_window,
            "Long_Quantile": long_quantile,
            "Short_Quantile": short_quantile,
        }
    )

    return {
        "positions": positions,
        "weights": weights,
        "asset_returns": asset_returns,
        "strategy_returns": strategy_returns,
        "scores": score,
        "summary": summary,
    }


def run_vix_threshold_cross_sectional_momentum_experiment(
    small_cap_price_panel: pd.DataFrame,
    large_cap_price_panel: pd.DataFrame,
    vix_close: pd.Series,
    config: RegimeStrategyConfig | None = None,
    vix_lookback_days: int = 25,
    vix_threshold_multiplier: float = 1.10,
    long_quantile: float = 0.2,
    short_quantile: float = 0.2,
    rebalance_rule: str = "W-FRI",
) -> dict[str, Any]:
    resolved_config = config or RegimeStrategyConfig(split_date="1990-01-01")
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
    regime_decisions["Decision_High_Vol_Flag"] = apply_rebalance_schedule(
        regime_decisions["Decision_High_Vol_Flag"],
        rebalance_rule=rebalance_rule,
    ).astype("boolean")
    regime_decisions["Decision_Regime"] = apply_rebalance_schedule(
        regime_decisions["Decision_Regime"],
        rebalance_rule=rebalance_rule,
    )
    regime_decisions["Decision_Regime_Change"] = regime_decisions["Decision_Regime"].ne(
        regime_decisions["Decision_Regime"].shift()
    )
    regime_decisions["Decision_Regime_Change"] = regime_decisions["Decision_Regime_Change"].where(
        regime_decisions["Decision_Regime"].notna(),
        False,
    )
    if not regime_decisions.empty:
        regime_decisions.iloc[0, regime_decisions.columns.get_loc("Decision_Regime_Change")] = False

    low_vol_strategy = build_cross_sectional_momentum_strategy(
        price_panel=all_stock_price_panel,
        formation_window=resolved_config.momentum_slow_window,
        skip_window=resolved_config.momentum_fast_window,
        long_quantile=long_quantile,
        short_quantile=short_quantile,
    )
    low_vol_strategy = apply_rebalance_schedule_to_strategy_output(
        low_vol_strategy,
        rebalance_rule=rebalance_rule,
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
    high_vol_strategy = apply_rebalance_schedule_to_strategy_output(
        high_vol_strategy,
        rebalance_rule=rebalance_rule,
    )

    backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=low_vol_strategy["weights"],
        low_vol_asset_returns=low_vol_strategy["asset_returns"],
        high_vol_weights=high_vol_strategy["weights"],
        high_vol_asset_returns=high_vol_strategy["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
        low_vol_strategy_label="cross_sectional_momentum_all_stocks",
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
    valid_regimes = regime_decisions["Decision_Regime"].dropna()
    regime_rule_summary = pd.Series(
        {
            "Rule_Name": "VIX threshold regime rule",
            "VIX_Lookback_Days": vix_lookback_days,
            "VIX_Threshold_Multiplier": vix_threshold_multiplier,
            "VIX_Threshold_Pct_Above_Average": 100.0 * (vix_threshold_multiplier - 1.0),
            "Rebalance_Rule": rebalance_rule,
            "Start_Date": valid_regimes.index.min().date(),
            "End_Date": valid_regimes.index.max().date(),
            "Observations": len(valid_regimes),
            "High_Vol_Time_Share_Pct": 100.0 * valid_regimes.eq("high_vol").mean(),
        }
    )

    return {
        "config": {
            **asdict(resolved_config),
            "vix_lookback_days": vix_lookback_days,
            "vix_threshold_multiplier": vix_threshold_multiplier,
            "low_vol_strategy_name": "cross_sectional_momentum_all_stocks",
            "cross_sectional_long_quantile": long_quantile,
            "cross_sectional_short_quantile": short_quantile,
            "rebalance_rule": rebalance_rule,
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


def run_vix_threshold_weekly_buy_hold_experiment(
    small_cap_price_panel: pd.DataFrame,
    large_cap_price_panel: pd.DataFrame,
    vix_close: pd.Series,
    config: RegimeStrategyConfig | None = None,
    vix_lookback_days: int = 25,
    vix_threshold_multiplier: float = 1.10,
    rebalance_rule: str = "W-FRI",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    resolved_config = config or RegimeStrategyConfig(split_date="1990-01-01")
    all_stock_price_panel = combine_disjoint_price_panels(small_cap_price_panel, large_cap_price_panel)
    all_stock_price_panel = slice_dated_data(all_stock_price_panel, start_date=start_date, end_date=end_date)
    if all_stock_price_panel.dropna(how="all").empty:
        raise ValueError("No stock price history is available for the requested window.")

    regime_decisions = build_weekly_vix_regime_decisions(
        vix_close=vix_close,
        asset_price_panel=all_stock_price_panel,
        lookback_days=vix_lookback_days,
        threshold_multiplier=vix_threshold_multiplier,
        rebalance_rule=rebalance_rule,
        start_date=start_date,
        end_date=end_date,
    )

    low_vol_strategy = build_buy_hold_strategy_for_universe(all_stock_price_panel)
    low_vol_strategy = apply_rebalance_schedule_to_strategy_output(
        low_vol_strategy,
        rebalance_rule=rebalance_rule,
    )
    all_stocks_high_vol_short = invert_strategy_output(
        low_vol_strategy,
        strategy_name="short_all_stocks",
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
    high_vol_strategy = apply_rebalance_schedule_to_strategy_output(
        high_vol_strategy,
        rebalance_rule=rebalance_rule,
    )

    backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=low_vol_strategy["weights"],
        low_vol_asset_returns=low_vol_strategy["asset_returns"],
        high_vol_weights=high_vol_strategy["weights"],
        high_vol_asset_returns=high_vol_strategy["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
        low_vol_strategy_label="buy_hold_all_stocks",
        high_vol_strategy_label="mean_reversion_all_stocks",
    )
    summary = summarize_cross_sectional_backtest(backtest)

    all_stocks_long_short_benchmark_backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=low_vol_strategy["weights"],
        low_vol_asset_returns=low_vol_strategy["asset_returns"],
        high_vol_weights=all_stocks_high_vol_short["weights"],
        high_vol_asset_returns=all_stocks_high_vol_short["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
        low_vol_strategy_label="buy_hold_all_stocks",
        high_vol_strategy_label="short_all_stocks",
    )
    all_stocks_long_short_benchmark_summary = summarize_cross_sectional_backtest(
        all_stocks_long_short_benchmark_backtest
    )

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
    valid_regimes = regime_decisions["Decision_Regime"].dropna()
    regime_rule_summary = pd.Series(
        {
            "Rule_Name": "VIX threshold regime rule",
            "VIX_Lookback_Days": vix_lookback_days,
            "VIX_Threshold_Multiplier": vix_threshold_multiplier,
            "VIX_Threshold_Pct_Above_Average": 100.0 * (vix_threshold_multiplier - 1.0),
            "Rebalance_Rule": rebalance_rule,
            "Start_Date": valid_regimes.index.min().date(),
            "End_Date": valid_regimes.index.max().date(),
            "Observations": len(valid_regimes),
            "High_Vol_Time_Share_Pct": 100.0 * valid_regimes.eq("high_vol").mean(),
        }
    )

    return {
        "config": {
            **asdict(resolved_config),
            "vix_lookback_days": vix_lookback_days,
            "vix_threshold_multiplier": vix_threshold_multiplier,
            "low_vol_strategy_name": "buy_hold_all_stocks",
            "rebalance_rule": rebalance_rule,
            "start_date": None if start_date is None else str(pd.Timestamp(start_date).date()),
            "end_date": None if end_date is None else str(pd.Timestamp(end_date).date()),
        },
        "regime_rule_summary": regime_rule_summary,
        "regime_decisions": regime_decisions,
        "low_vol_strategy": low_vol_strategy,
        "high_vol_strategy": high_vol_strategy,
        "backtest": backtest,
        "summary": summary,
        "all_stocks_long_short_benchmark_backtest": all_stocks_long_short_benchmark_backtest,
        "all_stocks_long_short_benchmark_summary": all_stocks_long_short_benchmark_summary,
        "transaction_cost_sweep": cost_sweep,
        "break_even_costs": break_even_costs,
    }


def run_vix_threshold_weekly_momentum_experiment(
    small_cap_price_panel: pd.DataFrame,
    large_cap_price_panel: pd.DataFrame,
    vix_close: pd.Series,
    config: RegimeStrategyConfig | None = None,
    vix_lookback_days: int = 25,
    vix_threshold_multiplier: float = 1.10,
    rebalance_rule: str = "W-FRI",
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    resolved_config = config or RegimeStrategyConfig(split_date="1990-01-01")
    all_stock_price_panel = combine_disjoint_price_panels(small_cap_price_panel, large_cap_price_panel)
    all_stock_price_panel = slice_dated_data(all_stock_price_panel, start_date=start_date, end_date=end_date)
    if all_stock_price_panel.dropna(how="all").empty:
        raise ValueError("No stock price history is available for the requested window.")

    regime_decisions = build_weekly_vix_regime_decisions(
        vix_close=vix_close,
        asset_price_panel=all_stock_price_panel,
        lookback_days=vix_lookback_days,
        threshold_multiplier=vix_threshold_multiplier,
        rebalance_rule=rebalance_rule,
        start_date=start_date,
        end_date=end_date,
    )

    low_vol_strategy = run_strategy_for_universe(
        price_panel=all_stock_price_panel,
        strategy_name="momentum",
        momentum_fast_window=resolved_config.momentum_fast_window,
        momentum_slow_window=resolved_config.momentum_slow_window,
        mean_reversion_window=resolved_config.mean_reversion_window,
        mean_reversion_theta=resolved_config.mean_reversion_theta,
        use_log_returns=resolved_config.use_log_returns,
    )
    low_vol_strategy = apply_rebalance_schedule_to_strategy_output(
        low_vol_strategy,
        rebalance_rule=rebalance_rule,
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
    high_vol_strategy = apply_rebalance_schedule_to_strategy_output(
        high_vol_strategy,
        rebalance_rule=rebalance_rule,
    )

    buy_hold_all_stocks = build_buy_hold_strategy_for_universe(all_stock_price_panel)
    buy_hold_all_stocks = apply_rebalance_schedule_to_strategy_output(
        buy_hold_all_stocks,
        rebalance_rule=rebalance_rule,
    )
    short_all_stocks = invert_strategy_output(
        buy_hold_all_stocks,
        strategy_name="short_all_stocks",
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

    all_stocks_long_short_benchmark_backtest = backtest_cross_sectional_regime_strategy(
        regime_decisions=regime_decisions,
        low_vol_weights=buy_hold_all_stocks["weights"],
        low_vol_asset_returns=buy_hold_all_stocks["asset_returns"],
        high_vol_weights=short_all_stocks["weights"],
        high_vol_asset_returns=short_all_stocks["asset_returns"],
        transaction_cost_bps=resolved_config.transaction_cost_bps,
        liquidate_on_regime_change=resolved_config.liquidate_on_regime_change,
        low_vol_strategy_label="buy_hold_all_stocks",
        high_vol_strategy_label="short_all_stocks",
    )
    all_stocks_long_short_benchmark_summary = summarize_cross_sectional_backtest(
        all_stocks_long_short_benchmark_backtest
    )

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
    valid_regimes = regime_decisions["Decision_Regime"].dropna()
    regime_rule_summary = pd.Series(
        {
            "Rule_Name": "VIX threshold regime rule",
            "VIX_Lookback_Days": vix_lookback_days,
            "VIX_Threshold_Multiplier": vix_threshold_multiplier,
            "VIX_Threshold_Pct_Above_Average": 100.0 * (vix_threshold_multiplier - 1.0),
            "Rebalance_Rule": rebalance_rule,
            "Start_Date": valid_regimes.index.min().date(),
            "End_Date": valid_regimes.index.max().date(),
            "Observations": len(valid_regimes),
            "High_Vol_Time_Share_Pct": 100.0 * valid_regimes.eq("high_vol").mean(),
        }
    )

    return {
        "config": {
            **asdict(resolved_config),
            "vix_lookback_days": vix_lookback_days,
            "vix_threshold_multiplier": vix_threshold_multiplier,
            "low_vol_strategy_name": "momentum_all_stocks",
            "rebalance_rule": rebalance_rule,
            "start_date": None if start_date is None else str(pd.Timestamp(start_date).date()),
            "end_date": None if end_date is None else str(pd.Timestamp(end_date).date()),
        },
        "regime_rule_summary": regime_rule_summary,
        "regime_decisions": regime_decisions,
        "low_vol_strategy": low_vol_strategy,
        "high_vol_strategy": high_vol_strategy,
        "backtest": backtest,
        "summary": summary,
        "all_stocks_long_short_benchmark_backtest": all_stocks_long_short_benchmark_backtest,
        "all_stocks_long_short_benchmark_summary": all_stocks_long_short_benchmark_summary,
        "transaction_cost_sweep": cost_sweep,
        "break_even_costs": break_even_costs,
    }


def run_train_test_optimized_vix_strategy(
    small_cap_price_panel: pd.DataFrame,
    large_cap_price_panel: pd.DataFrame,
    vix_close: pd.Series,
    base_config: RegimeStrategyConfig | None = None,
    train_start_date: str | pd.Timestamp = "1990-01-01",
    split_date: str | pd.Timestamp = "2010-01-01",
    test_end_date: str | pd.Timestamp | None = "2025-12-31",
    parameter_grid: dict[str, list[Any]] | None = None,
    rebalance_rule: str = "W-FRI",
    optimization_metric: str = "Net_Sharpe",
) -> dict[str, Any]:
    resolved_base_config = base_config or RegimeStrategyConfig(split_date=split_date)
    train_start_ts = pd.Timestamp(train_start_date)
    split_ts = pd.Timestamp(split_date)
    train_end_ts = split_ts - pd.Timedelta(days=1)
    test_end_ts = None if test_end_date is None else pd.Timestamp(test_end_date)
    combined_universe = combine_disjoint_price_panels(small_cap_price_panel, large_cap_price_panel)
    available_universe = combined_universe.dropna(how="all")
    if available_universe.empty:
        raise ValueError("The stock universe is empty after combining the large-cap and small-cap panels.")

    available_start = available_universe.index.min()
    available_end = available_universe.index.max()

    if train_end_ts < train_start_ts:
        raise ValueError("split_date must be after train_start_date.")
    if test_end_ts is not None and test_end_ts < split_ts:
        raise ValueError("test_end_date must be on or after split_date.")
    effective_train_start_ts = max(train_start_ts, available_start)
    effective_test_end_ts = available_end if test_end_ts is None else min(test_end_ts, available_end)

    if effective_train_start_ts > train_end_ts:
        raise ValueError(
            "The requested training window does not overlap enough stock-universe history. "
            f"Requested train_start_date={train_start_ts.date()} and split_date={split_ts.date()}, "
            f"but the available stock history starts on {available_start.date()}."
        )
    if effective_test_end_ts < split_ts:
        raise ValueError(
            "The stock-universe history ends before the requested test window begins. "
            f"Requested split_date={split_ts.date()}, but the latest stock date is {available_end.date()}."
        )

    resolved_grid = build_parameter_search_grid(parameter_grid)
    candidate_rows: list[dict[str, Any]] = []
    best_score = float("-inf")
    best_summary_value = float("-inf")
    best_params: dict[str, Any] | None = None

    for candidate_values in product(
        resolved_grid["vix_lookback_days"],
        resolved_grid["vix_threshold_multiplier"],
        resolved_grid["mean_reversion_window"],
        resolved_grid["mean_reversion_theta"],
    ):
        (
            vix_lookback_days,
            vix_threshold_multiplier,
            mean_reversion_window,
            mean_reversion_theta,
        ) = candidate_values

        candidate_config = replace(
            resolved_base_config,
            split_date=split_ts,
            mean_reversion_window=int(mean_reversion_window),
            mean_reversion_theta=float(mean_reversion_theta),
        )

        try:
            train_result = run_vix_threshold_weekly_buy_hold_experiment(
                small_cap_price_panel=small_cap_price_panel,
                large_cap_price_panel=large_cap_price_panel,
                vix_close=vix_close,
                config=candidate_config,
                vix_lookback_days=int(vix_lookback_days),
                vix_threshold_multiplier=float(vix_threshold_multiplier),
                rebalance_rule=rebalance_rule,
                start_date=effective_train_start_ts,
                end_date=train_end_ts,
            )
            train_summary = train_result["summary"]
            score = score_summary_metric(train_summary, optimization_metric)
            summary_value = score_summary_metric(train_summary, "Net_Annualized_Return")
            candidate_rows.append(
                {
                    "VIX_Lookback_Days": int(vix_lookback_days),
                    "VIX_Threshold_Multiplier": float(vix_threshold_multiplier),
                    "Mean_Reversion_Window": int(mean_reversion_window),
                    "Mean_Reversion_Theta": float(mean_reversion_theta),
                    "Optimization_Metric": optimization_metric,
                    "Optimization_Score": score,
                    "Train_Net_Sharpe": train_summary.get("Net_Sharpe"),
                    "Train_Net_Annualized_Return": train_summary.get("Net_Annualized_Return"),
                    "Train_Net_Total_Return": train_summary.get("Net_Total_Return"),
                    "Train_Net_Max_Drawdown": train_summary.get("Net_Max_Drawdown"),
                }
            )
            if score > best_score or (score == best_score and summary_value > best_summary_value):
                best_score = score
                best_summary_value = summary_value
                best_params = {
                    "vix_lookback_days": int(vix_lookback_days),
                    "vix_threshold_multiplier": float(vix_threshold_multiplier),
                    "mean_reversion_window": int(mean_reversion_window),
                    "mean_reversion_theta": float(mean_reversion_theta),
                }
        except ValueError as error:
            candidate_rows.append(
                {
                    "VIX_Lookback_Days": int(vix_lookback_days),
                    "VIX_Threshold_Multiplier": float(vix_threshold_multiplier),
                    "Mean_Reversion_Window": int(mean_reversion_window),
                    "Mean_Reversion_Theta": float(mean_reversion_theta),
                    "Optimization_Metric": optimization_metric,
                    "Optimization_Score": float("-inf"),
                    "Train_Net_Sharpe": np.nan,
                    "Train_Net_Annualized_Return": np.nan,
                    "Train_Net_Total_Return": np.nan,
                    "Train_Net_Max_Drawdown": np.nan,
                    "Error": str(error),
                }
            )

    if best_params is None:
        raise ValueError(
            "No valid parameter combination produced a training backtest. "
            f"Available stock-universe history spans {available_start.date()} to {available_end.date()}. "
            "This usually means the requested train/test window does not overlap enough history or every candidate failed warm-up requirements."
        )

    best_config = replace(
        resolved_base_config,
        split_date=split_ts,
        mean_reversion_window=best_params["mean_reversion_window"],
        mean_reversion_theta=best_params["mean_reversion_theta"],
    )

    train_result = run_vix_threshold_weekly_buy_hold_experiment(
        small_cap_price_panel=small_cap_price_panel,
        large_cap_price_panel=large_cap_price_panel,
        vix_close=vix_close,
        config=best_config,
        vix_lookback_days=best_params["vix_lookback_days"],
        vix_threshold_multiplier=best_params["vix_threshold_multiplier"],
        rebalance_rule=rebalance_rule,
        start_date=effective_train_start_ts,
        end_date=train_end_ts,
    )
    test_result = run_vix_threshold_weekly_buy_hold_experiment(
        small_cap_price_panel=small_cap_price_panel,
        large_cap_price_panel=large_cap_price_panel,
        vix_close=vix_close,
        config=best_config,
        vix_lookback_days=best_params["vix_lookback_days"],
        vix_threshold_multiplier=best_params["vix_threshold_multiplier"],
        rebalance_rule=rebalance_rule,
        start_date=split_ts,
        end_date=effective_test_end_ts,
    )

    optimization_table = pd.DataFrame(candidate_rows).sort_values(
        by=["Optimization_Score", "Train_Net_Annualized_Return"],
        ascending=[False, False],
        na_position="last",
    )
    selected_parameters = pd.Series(
        {
            "Optimization_Metric": optimization_metric,
            "Requested_Train_Start_Date": train_start_ts.date(),
            "Train_Start_Date": effective_train_start_ts.date(),
            "Train_End_Date": train_end_ts.date(),
            "Test_Start_Date": split_ts.date(),
            "Requested_Test_End_Date": None if test_end_ts is None else test_end_ts.date(),
            "Test_End_Date": effective_test_end_ts.date(),
            "VIX_Lookback_Days": best_params["vix_lookback_days"],
            "VIX_Threshold_Multiplier": best_params["vix_threshold_multiplier"],
            "Mean_Reversion_Window": best_params["mean_reversion_window"],
            "Mean_Reversion_Theta": best_params["mean_reversion_theta"],
            "Train_Optimization_Score": best_score,
            "Train_Net_Sharpe": train_result["summary"].get("Net_Sharpe"),
            "Train_Net_Annualized_Return": train_result["summary"].get("Net_Annualized_Return"),
            "Train_Net_Total_Return": train_result["summary"].get("Net_Total_Return"),
            "Test_Net_Sharpe": test_result["summary"].get("Net_Sharpe"),
            "Test_Net_Annualized_Return": test_result["summary"].get("Net_Annualized_Return"),
            "Test_Net_Total_Return": test_result["summary"].get("Net_Total_Return"),
        }
    )

    return {
        "config": {
            **asdict(best_config),
            "requested_train_start_date": str(train_start_ts.date()),
            "train_start_date": str(effective_train_start_ts.date()),
            "split_date": str(split_ts.date()),
            "requested_test_end_date": None if test_end_ts is None else str(test_end_ts.date()),
            "test_end_date": str(effective_test_end_ts.date()),
            "rebalance_rule": rebalance_rule,
            "optimization_metric": optimization_metric,
            "parameter_grid": resolved_grid,
        },
        "selected_parameters": selected_parameters,
        "optimization_table": optimization_table,
        "train_result": train_result,
        "train_summary": train_result["summary"],
        "test_result": test_result,
        "test_summary": test_result["summary"],
        "regime_rule_summary": test_result["regime_rule_summary"],
        "regime_decisions": test_result["regime_decisions"],
        "low_vol_strategy": test_result["low_vol_strategy"],
        "high_vol_strategy": test_result["high_vol_strategy"],
        "backtest": test_result["backtest"],
        "summary": test_result["summary"],
        "all_stocks_long_short_benchmark_backtest": test_result["all_stocks_long_short_benchmark_backtest"],
        "all_stocks_long_short_benchmark_summary": test_result["all_stocks_long_short_benchmark_summary"],
        "transaction_cost_sweep": test_result["transaction_cost_sweep"],
        "break_even_costs": test_result["break_even_costs"],
    }
