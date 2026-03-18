import numpy as np
import pandas as pd


def mean_reversion_returns_with_exit(price, window=5, theta=1.0, use_log_returns=False):
    df = pd.DataFrame({'price': pd.Series(price)}).copy()

    # Compute returns
    if use_log_returns:
        df['r'] = np.log(df['price'] / df['price'].shift(1))
    else:
        df['r'] = df['price'].pct_change()

    # Rolling statistics on returns
    df['mu'] = df['r'].rolling(window).mean()
    df['sigma'] = df['r'].rolling(window).std()

    # Z-score
    df['z'] = (df['r'] - df['mu']) / df['sigma']

    # Mean-reversion position logic
    position = []
    current_pos = 0

    for z in df['z']:
        if pd.isna(z):
            position.append(0)
            continue

        # Entry rules
        if current_pos == 0:
            if z < -theta:
                current_pos = 1      # go long
            elif z > theta:
                current_pos = -1     # go short

        # Exit rules
        elif current_pos == 1:
            if z >= 0:
                current_pos = 0
        elif current_pos == -1:
            if z <= 0:
                current_pos = 0

        position.append(current_pos)

    df['position'] = position

    # Strategy return: today's position applied to next period's return
    df['strategy_ret'] = df['position'].shift(1) * df['r']

    # Cumulative returns
    df['cum_asset'] = (1 + df['r'].fillna(0)).cumprod()
    df['cum_strategy'] = (1 + df['strategy_ret'].fillna(0)).cumprod()

    return df


df = pd.read_csv("TSLA_10y.csv")

# Standardize column names
df.columns = [col.lower().replace(" ", "_") for col in df.columns]

# Convert date
df['date'] = pd.to_datetime(df['date'])
df.set_index('date', inplace=True)

# Use adjusted close (best practice)
price = df['adj_close']

# Run the strategy
df = mean_reversion_returns_with_exit(
    price=price,
    window=5,
    theta=1.0,
    use_log_returns=False
)

# Display results
print(df[['price', 'r', 'mu', 'sigma', 'z', 'position', 'strategy_ret']])

# Basic performance statistics
strategy_ret = df['strategy_ret'].dropna()

if len(strategy_ret) > 1 and strategy_ret.std() != 0:
    sharpe = np.sqrt(252) * strategy_ret.mean() / strategy_ret.std()
else:
    sharpe = np.nan

max_drawdown = (df['cum_strategy'] / df['cum_strategy'].cummax() - 1).min()

print("\nPerformance summary:")
print(f"Final cumulative asset return:    {df['cum_asset'].iloc[-1]:.4f}")
print(f"Final cumulative strategy return: {df['cum_strategy'].iloc[-1]:.4f}")
print(f"Sharpe ratio (annualized):        {sharpe:.4f}")
print(f"Max drawdown:                     {max_drawdown:.4f}")