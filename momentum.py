import numpy as np
import pandas as pd
import yfinance as yf

# Get data from yf
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
data.to_csv("TSLA_10y_momentum.csv", index=False)


# Implementation of the momentum strategy (MA crossover style)
def momentum_sma_crossover(price, fast_window=20, slow_window=50, use_log_returns=False):
    """
    Momentum Strategy based on Simple Moving Average (SMA) Crossover.
    Goes long (1) when Fast MA > Slow MA.
    Goes short (-1) when Fast MA < Slow MA.
    """
    df = pd.DataFrame({"price": pd.Series(price)}).copy()

    # Ensure numeric
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["price"]).copy()

    # Compute daily returns
    if use_log_returns:
        df["r"] = np.log(df["price"] / df["price"].shift(1))
    else:
        df["r"] = df["price"].pct_change(fill_method=None)

    # Calculate Fast and Slow Moving Averages
    df["fast_ma"] = df["price"].rolling(window=fast_window).mean()
    df["slow_ma"] = df["price"].rolling(window=slow_window).mean()

    # Determine Position (1 for Long, -1 for Short, 0 for Neutral/Startup)
    # np.where evaluates the condition and applies the first outcome if True, second if False
    conditions = [
        df["fast_ma"] > df["slow_ma"],
        df["fast_ma"] < df["slow_ma"]
    ]
    choices = [1, -1]
    
    # Use 0 as default before the slow_window is fully calculated
    df["position"] = np.select(conditions, choices, default=0)
    
    # If moving averages are NaN, enforce 0 position
    df.loc[df["slow_ma"].isna(), "position"] = 0

    # Strategy return uses previous day's position to avoid look-ahead bias
    df["strategy_ret"] = df["position"].shift(1) * df["r"]

    # Cumulative returns
    df["cum_asset"] = (1 + df["r"].fillna(0)).cumprod()
    df["cum_strategy"] = (1 + df["strategy_ret"].fillna(0)).cumprod()

    return df


# Load csv
df = pd.read_csv("TSLA_10y_momentum.csv")

print("\nRaw columns from CSV:", df.columns.tolist())

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

price = df[price_col]

# Run strategy
# Try 20-day and 50-day SMA as it is a common short-to-medium term momentum pair
out = momentum_sma_crossover(price, fast_window=20, slow_window=50)

if out.empty:
    raise ValueError("Strategy output is empty after cleaning.")

strategy_ret = out["strategy_ret"].dropna()

if len(strategy_ret) > 1 and strategy_ret.std() != 0:
    # Annualized Sharpe ratio for daily data (sqrt of 252 trading days)
    sharpe = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
else:
    sharpe = np.nan

max_drawdown = (out["cum_strategy"] / out["cum_strategy"].cummax() - 1).min()

print("\nLast rows of strategy output:")
print(out[["price", "fast_ma", "slow_ma", "position", "strategy_ret", "cum_strategy"]].tail())

print("\nPerformance summary:")
print(f"Price column used:                {price_col}")
print(f"Final cumulative asset return:    {out['cum_asset'].iloc[-1]:.4f}")
print(f"Final cumulative strategy return: {out['cum_strategy'].iloc[-1]:.4f}")
print(f"Sharpe ratio (annualized):        {sharpe:.4f}")
print(f"Max drawdown:                     {max_drawdown:.4f}")