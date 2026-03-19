import numpy as np
import pandas as pd
import yfinance as yf

# =========================
# DOWNLOAD DATA
# =========================
data = yf.download("TSLA", period="10y", interval="1d", auto_adjust=False)

# Flatten MultiIndex columns if needed
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

# Reset index so Date becomes a column
data = data.reset_index()

# Standardize column names
data.columns = [str(c).strip().lower().replace(" ", "_") for c in data.columns]

print("Downloaded columns:", data.columns.tolist())
print(data.head())

# Save clean CSV
data.to_csv("TSLA_10y.csv", index=False)


# =========================
# STRATEGY FUNCTION
# =========================
def mean_reversion_returns_with_exit(price, window=5, theta=1.0, use_log_returns=False):
    df = pd.DataFrame({"price": pd.Series(price)}).copy()

    # Ensure numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()

    # Compute returns
    if use_log_returns:
        df["r"] = np.log(df["price"] / df["price"].shift(1))
    else:
        df["r"] = df["price"].pct_change(fill_method=None)

    # Rolling mean and std of returns
    df["mu"] = df["r"].rolling(window).mean()
    df["sigma"] = df["r"].rolling(window).std()

    # Avoid division by zero
    df["sigma"] = df["sigma"].replace(0, np.nan)

    # Z-score
    df["z"] = (df["r"] - df["mu"]) / df["sigma"]

    # Trading rule
    position = []
    current_pos = 0

    for z in df["z"]:
        if pd.isna(z):
            position.append(0)
            continue

        if current_pos == 0:
            if z < -theta:
                current_pos = 1      # go long
            elif z > theta:
                current_pos = -1     # go short
        elif current_pos == 1:
            if z >= 0:
                current_pos = 0      # exit long
        elif current_pos == -1:
            if z <= 0:
                current_pos = 0      # exit short

        position.append(current_pos)

    df["position"] = position

    # Strategy return uses previous day's position
    df["strategy_ret"] = df["position"].shift(1) * df["r"]

    # Cumulative returns
    df["cum_asset"] = (1 + df["r"].fillna(0)).cumprod()
    df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    return df


# =========================
# LOAD CLEAN CSV
# =========================
df = pd.read_csv("TSLA_10y.csv")

print("\nRaw columns from CSV:", df.columns.tolist())
print(df.head())

# Convert date column
df["date"] = pd.to_datetime(df["date"], errors="coerce")
df = df.dropna(subset=["date"]).copy()
df = df.set_index("date")

# Standardize names again just in case
df.columns = [str(c).strip().lower().replace(" ", "_") for c in df.columns]

# Convert numeric columns
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Use adj_close if available, otherwise close
price_col = "adj_close" if "adj_close" in df.columns else "close"

# Keep only valid prices
df = df.dropna(subset=[price_col]).copy()

print("\nCleaned columns:", df.columns.tolist())
print(df.head())
print(df.dtypes)

price = df[price_col]

# =========================
# RUN STRATEGY
# =========================
out = mean_reversion_returns_with_exit(price, window=5, theta=1.0)

if out.empty:
    raise ValueError("Strategy output is empty after cleaning.")

strategy_ret = out["strategy_ret"].dropna()

if len(strategy_ret) > 1 and strategy_ret.std() != 0:
    sharpe = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
else:
    sharpe = np.nan

max_drawdown = (out["cum_strategy"] / out["cum_strategy"].cummax() - 1).min()

print("\nLast rows of strategy output:")
print(out[["price", "r", "mu", "sigma", "z", "position", "strategy_ret"]].tail())

print("\nPerformance summary:")
print(f"Price column used:                {price_col}")
print(f"Final cumulative asset return:    {out['cum_asset'].iloc[-1]:.4f}")
print(f"Final cumulative strategy return: {out['cum_strategy'].iloc[-1]:.4f}")
print(f"Sharpe ratio (annualized):        {sharpe:.4f}")
print(f"Max drawdown:                     {max_drawdown:.4f}")