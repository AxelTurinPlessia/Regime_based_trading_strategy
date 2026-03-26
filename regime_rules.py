from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def build_probability_confirmed_regime(
    prob_high_vol: pd.Series,
    threshold: float = 0.90,
) -> pd.Series:
    confirmed_regime: list[str] = []
    current_regime = "high_vol" if prob_high_vol.iloc[0] >= 0.5 else "low_vol"

    for probability in prob_high_vol:
        if probability >= threshold:
            current_regime = "high_vol"
        elif probability <= 1 - threshold:
            current_regime = "low_vol"
        confirmed_regime.append(current_regime)

    return pd.Series(confirmed_regime, index=prob_high_vol.index, name="Prob_Confirmed_Regime")


def apply_min_consecutive_days(
    candidate_regime: pd.Series,
    min_days: int = 2,
) -> pd.Series:
    filtered_regime: list[str] = []
    current_regime = candidate_regime.iloc[0]
    pending_regime: str | None = None
    pending_streak = 0

    for regime in candidate_regime:
        if regime == current_regime:
            pending_regime = None
            pending_streak = 0
            filtered_regime.append(current_regime)
            continue

        if regime == pending_regime:
            pending_streak += 1
        else:
            pending_regime = regime
            pending_streak = 1

        if pending_streak >= min_days:
            current_regime = pending_regime
            pending_regime = None
            pending_streak = 0

        filtered_regime.append(current_regime)

    return pd.Series(filtered_regime, index=candidate_regime.index, name="Min_Duration_Regime")


def build_regime_path_from_probability(
    prob_high_vol: pd.Series,
    threshold: float = 0.90,
    min_days: int | None = None,
) -> pd.Series:
    candidate_regime = build_probability_confirmed_regime(prob_high_vol, threshold=threshold)
    if min_days is None:
        return candidate_regime
    return apply_min_consecutive_days(candidate_regime, min_days=min_days)


def bucket_run_length(trading_days: int) -> str:
    if trading_days <= 1:
        return "1 day"
    if trading_days == 2:
        return "2 days"
    if trading_days <= 5:
        return "3-5 days"
    if trading_days <= 10:
        return "6-10 days"
    if trading_days <= 20:
        return "11-20 days"
    return "21+ days"


def get_high_vol_intervals(high_vol_indicator: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    high_vol_indicator = high_vol_indicator.astype(bool)
    intervals: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    interval_start: pd.Timestamp | None = None

    for timestamp, is_high_vol in high_vol_indicator.items():
        if is_high_vol and interval_start is None:
            interval_start = timestamp
        elif not is_high_vol and interval_start is not None:
            intervals.append((interval_start, timestamp))
            interval_start = None

    if interval_start is not None:
        intervals.append((interval_start, high_vol_indicator.index[-1]))

    return intervals


def build_regime_runs(regime_path: pd.Series, scenario_name: str = "Regime Path") -> pd.DataFrame:
    switch_flags = regime_path.ne(regime_path.shift())
    switch_flags.iloc[0] = False

    run_start_flags = switch_flags.copy()
    run_start_flags.iloc[0] = True
    run_id = run_start_flags.cumsum()

    grouped = pd.DataFrame(
        {
            "Regime": regime_path,
            "Run_ID": run_id,
        }
    ).groupby("Run_ID")

    regime_runs = grouped.apply(
        lambda frame: pd.Series(
            {
                "Scenario": scenario_name,
                "Regime": frame["Regime"].iloc[0],
                "Start_Date": frame.index[0],
                "End_Date": frame.index[-1],
                "Trading_Days": len(frame),
                "Calendar_Days": (frame.index[-1] - frame.index[0]).days + 1,
            }
        ),
        include_groups=False,
    ).reset_index(drop=True)
    regime_runs["Length_Bucket"] = regime_runs["Trading_Days"].map(bucket_run_length)
    return regime_runs


def summarize_regime_path(regime_path: pd.Series, scenario_name: str = "Regime Path") -> pd.Series:
    regime_runs = build_regime_runs(regime_path, scenario_name=scenario_name)
    high_vol_runs = regime_runs.loc[regime_runs["Regime"] == "high_vol", "Trading_Days"]
    low_vol_runs = regime_runs.loc[regime_runs["Regime"] == "low_vol", "Trading_Days"]
    sample_years = len(regime_path) / 252.0
    regime_changes = int(regime_path.ne(regime_path.shift()).iloc[1:].sum())

    return pd.Series(
        {
            "Scenario": scenario_name,
            "Regime_Changes": regime_changes,
            "Changes_Per_Year": regime_changes / sample_years if sample_years else np.nan,
            "High_Vol_Time_Share_Pct": 100 * regime_path.eq("high_vol").mean(),
            "Average_High_Vol_Run_Days": high_vol_runs.mean() if not high_vol_runs.empty else np.nan,
            "Median_High_Vol_Run_Days": high_vol_runs.median() if not high_vol_runs.empty else np.nan,
            "Average_Low_Vol_Run_Days": low_vol_runs.mean() if not low_vol_runs.empty else np.nan,
            "Median_Low_Vol_Run_Days": low_vol_runs.median() if not low_vol_runs.empty else np.nan,
            "Max_High_Vol_Run_Days": high_vol_runs.max() if not high_vol_runs.empty else np.nan,
            "Max_Low_Vol_Run_Days": low_vol_runs.max() if not low_vol_runs.empty else np.nan,
            "One_Day_High_Vol_Run_Share_Pct": 100 * (high_vol_runs.eq(1).mean() if not high_vol_runs.empty else np.nan),
            "Liquidate_to_Cash_Annual_Turnover_Pct": (100 * regime_changes / sample_years) if sample_years else np.nan,
            "Full_Switch_Annual_Gross_Turnover_Pct": (200 * regime_changes / sample_years) if sample_years else np.nan,
        }
    )


def combine_regime_strategy_returns(
    regime_path: pd.Series,
    low_vol_returns: pd.Series,
    high_vol_returns: pd.Series,
    transaction_cost_bps: float = 0.0,
    signal_lag_days: int = 0,
    full_switch_cost: bool = True,
) -> pd.DataFrame:
    aligned = pd.concat(
        [
            regime_path.rename("Regime"),
            low_vol_returns.rename("Low_Vol_Return"),
            high_vol_returns.rename("High_Vol_Return"),
        ],
        axis=1,
    ).dropna()

    if signal_lag_days:
        aligned["Regime"] = aligned["Regime"].shift(signal_lag_days)
        aligned = aligned.dropna()

    aligned["Gross_Strategy_Return"] = np.where(
        aligned["Regime"].eq("high_vol"),
        aligned["High_Vol_Return"],
        aligned["Low_Vol_Return"],
    )
    aligned["Regime_Change"] = aligned["Regime"].ne(aligned["Regime"].shift()).fillna(False)
    aligned.iloc[0, aligned.columns.get_loc("Regime_Change")] = False
    turnover_units = 2.0 if full_switch_cost else 1.0
    aligned["Transaction_Cost"] = aligned["Regime_Change"].astype(float) * turnover_units * transaction_cost_bps / 10000.0
    aligned["Net_Strategy_Return"] = aligned["Gross_Strategy_Return"] - aligned["Transaction_Cost"]
    aligned["Cum_Gross_Strategy"] = (1 + aligned["Gross_Strategy_Return"].fillna(0)).cumprod()
    aligned["Cum_Net_Strategy"] = (1 + aligned["Net_Strategy_Return"].fillna(0)).cumprod()
    return aligned


def blend_strategy_returns_from_probability(
    prob_high_vol: pd.Series,
    low_vol_returns: pd.Series,
    high_vol_returns: pd.Series,
) -> pd.DataFrame:
    aligned = pd.concat(
        [
            prob_high_vol.rename("Prob_High_Vol"),
            low_vol_returns.rename("Low_Vol_Return"),
            high_vol_returns.rename("High_Vol_Return"),
        ],
        axis=1,
    ).dropna()
    aligned["Prob_Low_Vol"] = 1 - aligned["Prob_High_Vol"]
    aligned["Blended_Return"] = (
        aligned["Prob_High_Vol"] * aligned["High_Vol_Return"]
        + aligned["Prob_Low_Vol"] * aligned["Low_Vol_Return"]
    )
    aligned["Cum_Blended_Strategy"] = (1 + aligned["Blended_Return"].fillna(0)).cumprod()
    return aligned

