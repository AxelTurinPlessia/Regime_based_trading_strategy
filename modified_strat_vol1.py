"""
Global Macro Regime Strategy (The "Holy Grail" Pairing)

Risk-On (Low Volatility): Cross-Sectional Equity Momentum (Top 20 Stocks)
Risk-Off (High Volatility): Cross-Asset Trend Following (TLT, GLD, or Cash)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import time
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

from project_dataset import load_regime_strategy_data

# Set pandas display options
pd.options.display.float_format = "{:,.6f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ==========================================
# SAFE HAVEN DATA FETCHER
# ==========================================

def fetch_safe_havens(start_date="1990-01-01", data_dir=Path("data")):
    """Downloads TLT (Bonds) and GLD (Gold) with a retry mechanism."""
    tickers = ["TLT", "GLD"]
    prices = {}
    print("\nVerifying Safe Haven assets (TLT, GLD)...")
    
    for ticker in tickers:
        file_path = data_dir / f"{ticker}_macro.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            prices[ticker] = df["Close"]
            continue
            
        # Download if not cached
        for attempt in range(3):
            try:
                print(f"  Downloading {ticker} (Attempt {attempt+1}/3)...")
                df = yf.download(ticker, start=start_date, auto_adjust=False, progress=False)
                if not df.empty:
                    if isinstance(df.columns, pd.MultiIndex):
                        df.columns = df.columns.get_level_values(0)
                    df.index.name = "Date"
                    out_df = pd.DataFrame({"Close": df["Close"]})
                    out_df.to_csv(file_path)
                    prices[ticker] = out_df["Close"]
                    break
            except Exception as e:
                print(f"    Failed: {e}")
            time.sleep(2)
            
    if not prices:
        raise ValueError("Failed to load Safe Haven assets.")
    return pd.DataFrame(prices)


# ==========================================
# STRATEGY SIGNAL GENERATORS
# ==========================================

def cross_sectional_momentum(prices_df, lookback_days=126, top_n=20):
    """
    RISK-ON ENGINE: Ranks all equities by their 6-month (126-day) return.
    Allocates equal weight to the Top N strongest stocks.
    """
    # Calculate lookback returns
    returns = prices_df.pct_change(periods=lookback_days, fill_method=None)
    
    # Rank cross-sectionally (1 is worst, N is best)
    ranks = returns.rank(axis=1, ascending=False)
    
    # Select Top N
    target_weights = (ranks <= top_n).astype(float)
    
    # Normalize weights so they sum to 1.0 daily
    row_sums = target_weights.sum(axis=1).replace(0, 1) # Avoid div by zero
    target_weights = target_weights.div(row_sums, axis=0)
    
    return target_weights.fillna(0)


def cross_asset_trend_following(macro_prices_df, trend_window=200):
    """
    RISK-OFF ENGINE: Evaluates TLT and GLD against their 200-day SMA.
    If in an uptrend, equal weight them. 
    If none are in an uptrend, weights are 0 (Capital defaults to Cash).
    """
    sma = macro_prices_df.rolling(trend_window).mean()
    is_uptrend = (macro_prices_df > sma).astype(float)
    
    # Normalize weights (e.g., if both are up, 0.5 each. If one is up, 1.0. If none, 0.0)
    row_sums = is_uptrend.sum(axis=1).replace(0, 1)
    target_weights = is_uptrend.div(row_sums, axis=0)
    
    return target_weights.fillna(0)


# ==========================================
# ORCHESTRATOR & PORTFOLIO NETTING
# ==========================================

def run_macro_regime_strategy(equity_prices, macro_prices, hmm_prob, tc_eq_bps, tc_macro_bps):
    """Blends the Equity and Macro strategies using continuous HMM probabilities."""
    
    # 1. Generate Strategy Targets
    w_eq_raw = cross_sectional_momentum(equity_prices, lookback_days=126, top_n=20)
    w_sh_raw = cross_asset_trend_following(macro_prices, trend_window=200)
    
    # 2. Align Data Indices
    common_idx = hmm_prob.index.intersection(equity_prices.index).intersection(macro_prices.index)
    prob_high = hmm_prob.loc[common_idx].fillna(method='ffill')
    prob_low = 1.0 - prob_high
    
    # 3. Apply Continuous Regime Blending
    # If 20% High Vol -> 80% Capital to Equities, 20% to Safe Havens
    w_eq = w_eq_raw.loc[common_idx].multiply(prob_low, axis=0)
    w_sh = w_sh_raw.loc[common_idx].multiply(prob_high, axis=0)
    
    # 4. Calculate Daily Returns
    r_eq = equity_prices.pct_change(fill_method=None).loc[common_idx]
    r_sh = macro_prices.pct_change(fill_method=None).loc[common_idx]
    
    # Gross returns (shifted by 1 to avoid look-ahead bias)
    gross_eq_ret = (w_eq.shift(1) * r_eq).sum(axis=1)
    gross_sh_ret = (w_sh.shift(1) * r_sh).sum(axis=1)
    
    # 5. Measure Turnover & Apply Costs
    # We measure absolute change in weights to calculate realistic friction
    turnover_eq = w_eq.diff().abs().sum(axis=1)
    turnover_sh = w_sh.diff().abs().sum(axis=1)
    
    daily_tc = (turnover_eq * (tc_eq_bps / 10000)) + (turnover_sh * (tc_macro_bps / 10000))
    net_ret = gross_eq_ret + gross_sh_ret - daily_tc.fillna(0)
    
    # 6. Package Results
    res = pd.DataFrame({
        'Prob_High_Vol': prob_high,
        'Gross_Eq_Ret': gross_eq_ret,
        'Gross_SH_Ret': gross_sh_ret,
        'Daily_TC': daily_tc,
        'Turnover': turnover_eq + turnover_sh,
        'Net_Strategy_Return': net_ret,
        'Cum_Net_Strategy': (1 + net_ret.fillna(0)).cumprod()
    }, index=common_idx)
    
    cum_ret = res['Cum_Net_Strategy'].iloc[-1]
    ann_ret = cum_ret ** (252 / len(res)) - 1
    ann_vol = res['Net_Strategy_Return'].std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol != 0 else 0
    max_dd = (res['Cum_Net_Strategy'] / res['Cum_Net_Strategy'].cummax() - 1).min()
    
    summary = pd.Series({
        "Net_Total_Return": cum_ret - 1,
        "Net_Annualized_Return": ann_ret,
        "Net_Annualized_Vol": ann_vol,
        "Net_Sharpe": sharpe,
        "Net_Max_Drawdown": max_dd,
        "Avg_Annual_Turnover": res['Turnover'].mean() * 252
    })
    return res, summary


# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================

if __name__ == "__main__":
    
    data_dir = Path("data")
    test_start_date = "2010-01-01" 
    test_end_date = "2025-12-31"
    
    # Transaction cost assumptions (Equities cost more than highly liquid macro ETFs)
    TC_BPS_EQUITY = 15.0
    TC_BPS_MACRO = 5.0
    
    print("Loading data...")
    strategy_data = load_regime_strategy_data(
        data_dir=data_dir,
        dividend_policy="no_dividends",
        market_return_mode="average",
        universe_price_source="folders",
        start_date=test_start_date,
        end_date=test_end_date,
    )
    
    # 1. Combine Equities into a master universe
    all_equities = pd.concat([strategy_data.large_cap_prices, strategy_data.small_cap_prices], axis=1)
    all_equities = all_equities.loc[:, ~all_equities.columns.duplicated()] # Remove duplicate columns if any
    
    # 2. Fetch Macro Safe Havens
    macro_prices = fetch_safe_havens(start_date=test_start_date, data_dir=data_dir)
    macro_prices = macro_prices.loc[test_start_date:test_end_date]
    
    # 3. Load HMM Forecasts
    hmm_path = data_dir / "portfolio_hmm_oos_2010_forecasts.csv"
    if not hmm_path.exists():
        raise FileNotFoundError(f"Missing HMM forecasts at {hmm_path}")
    hmm_df = pd.read_csv(hmm_path, index_col="Date", parse_dates=True)
    hmm_prob_high_vol = hmm_df["Next_Day_Forecast_Prob_High_Vol"].loc[test_start_date:test_end_date]
    
    print(f"\nMaster Equity Universe: {all_equities.shape[1]} stocks")
    print(f"Safe Haven Universe: {macro_prices.shape[1]} assets (TLT, GLD)")
    
    # 4. Execute Macro Strategy
    print("\nExecuting Global Macro Blended Strategy...")
    backtest_df, performance_summary = run_macro_regime_strategy(
        all_equities,
        macro_prices,
        hmm_prob_high_vol,
        tc_eq_bps=TC_BPS_EQUITY,
        tc_macro_bps=TC_BPS_MACRO
    )
    
    print("\n--- PERFORMANCE SUMMARY (2010-2025) ---")
    print(performance_summary.to_string())
    
    # 5. Visualization
    print("\nGenerating performance plots in browser...")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=backtest_df.index, y=backtest_df["Cum_Net_Strategy"], 
                   mode="lines", name="Global Macro Regime Strategy", line=dict(color="#2ca02c", width=2.5)),
        secondary_y=False
    )
    
    # Benchmark against SPY
    mkt_bnh = (1 + strategy_data.market_return.loc[backtest_df.index].fillna(0)).cumprod()
    fig.add_trace(
        go.Scatter(x=mkt_bnh.index, y=mkt_bnh, 
                   mode="lines", name="Market Buy & Hold", line=dict(color="#7f7f7f", width=1.5, dash="dash")),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=backtest_df.index, y=backtest_df["Prob_High_Vol"], 
                   mode="lines", name="HMM Prob High Vol (Alloc to Safe Havens)", 
                   line=dict(color="rgba(214, 39, 40, 0)"), fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.15)"),
        secondary_y=True
    )
    
    fig.update_layout(
        title="OOS Performance & Capital Allocation",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_yaxes(title_text="Growth of $1", secondary_y=False)
    fig.update_yaxes(title_text="Safe Haven Allocation %", secondary_y=True, range=[0, 1], tickformat=".0%")
    
    fig.show()
    print("Run complete.")