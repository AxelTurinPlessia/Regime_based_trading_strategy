import numpy as np
import pandas as pd

def mean_reversion_returns_with_exit(price, window=5, theta=1.0, use_log_returns=False):
    df = pd.DataFrame({'price': pd.Series(price)}).copy()

    # Ensure numeric
    df['price'] = pd.to_numeric(df['price'], errors='coerce')
    df = df.dropna(subset=['price'])

    if use_log_returns:
        df['r'] = np.log(df['price'] / df['price'].shift(1))
    else:
        df['r'] = df['price'].pct_change(fill_method=None)

    df['mu'] = df['r'].rolling(window).mean()
    df['sigma'] = df['r'].rolling(window).std()
    df['z'] = (df['r'] - df['mu']) / df['sigma']

    position = []
    current_pos = 0

    for z in df['z']:
        if pd.isna(z):
            position.append(0)
            continue

        if current_pos == 0:
            if z < -theta:
                current_pos = 1
            elif z > theta:
                current_pos = -1
        elif current_pos == 1:
            if z >= 0:
                current_pos = 0
        elif current_pos == -1:
            if z <= 0:
                current_pos = 0

        position.append(current_pos)

    df['position'] = position
    df['strategy_ret'] = df['position'].shift(1) * df['r']
    df['cum_asset'] = (1 + df['r'].fillna(0)).cumprod()
    df['cum_strategy'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

    return df


# ===== LOAD CSV SAFELY =====
df = pd.read_csv("TSLA_10y.csv", header=0)

print("Raw columns:", df.columns.tolist())
print(df.head(5))

# Remove extra non-date rows
first_col = df.columns[0]
df[first_col] = pd.to_datetime(df[first_col], errors="coerce")
df = df[df[first_col].notna()].copy()

# Rename and index
df = df.rename(columns={first_col: "date"})
df = df.set_index("date")

# Standardize names
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Convert everything numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Keep valid prices only
df = df.dropna(subset=["adj_close"])

print("\nCleaned columns:", df.columns.tolist())
print(df.head())
print(df.dtypes)

price = df["adj_close"]

out = mean_reversion_returns_with_exit(price, window=5, theta=1.0)

if out.empty:
    raise ValueError("Strategy output is empty after cleaning.")

strategy_ret = out["strategy_ret"].dropna()

if len(strategy_ret) > 1 and strategy_ret.std() != 0:
    sharpe = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
else:
    sharpe = np.nan

max_drawdown = (out["cum_strategy"] / out["cum_strategy"].cummax() - 1).min()

print(out[["price", "r", "mu", "sigma", "z", "position", "strategy_ret"]].tail())

print("\nPerformance summary:")
print(f"Final cumulative asset return:    {out['cum_asset'].iloc[-1]:.4f}")
print(f"Final cumulative strategy return: {out['cum_strategy'].iloc[-1]:.4f}")
print(f"Sharpe ratio (annualized):        {sharpe:.4f}")
print(f"Max drawdown:                     {max_drawdown:.4f}")