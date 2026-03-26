from __future__ import annotations

import numpy as np
import pandas as pd


def mean_reversion_returns_with_exit(
    price: pd.Series,
    window: int = 5,
    theta: float = 1.0,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    df = pd.DataFrame({"price": pd.Series(price)}).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()

    if use_log_returns:
        df["r"] = np.log(df["price"] / df["price"].shift(1))
    else:
        df["r"] = df["price"].pct_change(fill_method=None)

    df["mu"] = df["r"].rolling(window).mean()
    df["sigma"] = df["r"].rolling(window).std()
    df["sigma"] = df["sigma"].replace(0, np.nan)
    df["z"] = (df["r"] - df["mu"]) / df["sigma"]

    position: list[int] = []
    current_pos = 0
    for z_score in df["z"]:
        if pd.isna(z_score):
            position.append(0)
            continue

        if current_pos == 0:
            if z_score < -theta:
                current_pos = 1
            elif z_score > theta:
                current_pos = -1
        elif current_pos == 1:
            if z_score >= 0:
                current_pos = 0
        elif current_pos == -1:
            if z_score <= 0:
                current_pos = 0

        position.append(current_pos)

    df["position"] = position
    df["strategy_ret"] = df["position"].shift(1) * df["r"]
    df["cum_asset"] = (1 + df["r"].fillna(0)).cumprod()
    df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    return df


def momentum_sma_crossover(
    price: pd.Series,
    fast_window: int = 20,
    slow_window: int = 50,
    use_log_returns: bool = False,
) -> pd.DataFrame:
    df = pd.DataFrame({"price": pd.Series(price)}).copy()
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()

    if use_log_returns:
        df["r"] = np.log(df["price"] / df["price"].shift(1))
    else:
        df["r"] = df["price"].pct_change(fill_method=None)

    df["fast_ma"] = df["price"].rolling(window=fast_window).mean()
    df["slow_ma"] = df["price"].rolling(window=slow_window).mean()

    conditions = [
        df["fast_ma"] > df["slow_ma"],
        df["fast_ma"] < df["slow_ma"],
    ]
    choices = [1, -1]
    df["position"] = np.select(conditions, choices, default=0)
    df.loc[df["slow_ma"].isna(), "position"] = 0

    df["strategy_ret"] = df["position"].shift(1) * df["r"]
    df["cum_asset"] = (1 + df["r"].fillna(0)).cumprod()
    df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()
    return df

