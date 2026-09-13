"""
Regime-Switched Trading Strategy: The "Carry & Crash"

Risk-On (Low Volatility): Dividend Carry (Proxy: Top 20 Minimum Volatility Stocks).
Risk-Off (High Volatility): Crisis Carry (Gold and Short-term Treasuries).
Orchestrator: Blends Yield-collection with Flight-to-Safety.
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
# MACRO DATA FETCHER (BIL & GLD)
# ==========================================

def fetch_carry_crash_havens(start_date="2000-01-01", data_dir=Path("data")):
    """Downloads BIL (Cash Proxy) and GLD (Crisis Hedge) with retry mechanism."""
    tickers = ["BIL", "GLD"]
    prices = {}
    print("\nVerifying Carry & Crash assets (BIL, GLD)...")
    
    for ticker in tickers:
        file_path = data_dir / f"{ticker}_cc.csv"
        if file_path.exists():
            df = pd.read_csv(file_path, index_col="Date", parse_dates=True)
            prices[ticker] = df["Close"]
            continue
            
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
        raise ValueError("Failed to load Carry & Crash safe-havens.")
    return pd.DataFrame(prices)

# ==========================================
# STRATEGY SIGNAL GENERATORS
# ==========================================

def risk_on_dividend_carry_proxy(prices_df, vol_window=252, top_n=20):
    """
    RISK-ON ENGINE: Dividend/Quality Carry.
    Uses 1-year (252-day) minimum volatility as a proxy for stable dividend payers.
    """
    daily_rets = prices_df.pct_change(fill_method=None)
    ann_vol = daily_rets.rolling(vol_window).std() * np.sqrt(252)
    
    # Rank ascending (1 is lowest vol / highest 'quality')
    ranks = ann_vol.rank(axis=1, ascending=True)
    target_weights = (ranks <= top_n).astype(float)
    
    row_sums = target_weights.sum(axis=1).replace(0, 1)
    return target_weights.div(row_sums, axis=0).fillna(0)


def risk_off_crisis_carry(macro_prices_df, trend_window=50):
    """
    RISK-OFF ENGINE: Cash and Gold.
    Allocates to Gold (GLD) if trending, otherwise holds BIL (T-Bills/Cash).
    """
    weights = pd.DataFrame(0.0, index=macro_prices_df.index, columns=macro_prices_df.columns)
    
    gld_sma = macro_prices_df["GLD"].rolling(trend_window).mean()
    
    # If Gold is in uptrend, 50% Gold / 50% Cash. Else 100% Cash.
    gld_up = macro_prices_df["GLD"] > gld_sma
    
    weights.loc[gld_up, "GLD"] = 0.5
    weights.loc[gld_up, "BIL"] = 0.5
    weights.loc[~gld_up, "BIL"] = 1.0
    
    return weights.fillna(0)

# ==========================================
# ORCHESTRATOR & PORTFOLIO NETTING
# ==========================================

def run_carry_crash_strategy(equity_prices, macro_prices, hmm_prob, tc_eq_bps, tc_macro_bps):
    """Blends Equity Carry with Crisis Havens based on HMM Probability."""
    
    # 1. Generate Targets
    w_eq_raw = risk_on_dividend_carry_proxy(equity_prices, top_n=20)
    w_ma_raw = risk_off_crisis_carry(macro_prices, trend_window=50)
    
    # 2. Alignment
    common_idx = hmm_prob.index.intersection(equity_prices.index).intersection(macro_prices.index)
    prob_high = hmm_prob.loc[common_idx].fillna(method='ffill')
    prob_low = 1.0 - prob_high
    
    # 3. Continuous Blending
    # Shift to track separate asset classes
    w_eq = w_eq_raw.loc[common_idx].multiply(prob_low, axis=0)
    w_ma = w_ma_raw.loc[common_idx].multiply(prob_high, axis=0)
    
    # 4. Returns
    r_eq = equity_prices.pct_change(fill_method=None).loc[common_idx]
    r_ma = macro_prices.pct_change(fill_method=None).loc[common_idx]
    
    gross_eq_ret = (w_eq.shift(1) * r_eq).sum(axis=1)
    gross_ma_ret = (w_ma.shift(1) * r_ma).sum(axis=1)
    
    # 5. Turnover & Costs
    turnover_eq = w_eq.diff().abs().sum(axis=1)
    turnover_ma = w_ma.diff().abs().sum(axis=1)
    
    daily_tc = (turnover_eq * (tc_eq_bps / 10000)) + (turnover_ma * (tc_macro_bps / 10000))
    net_ret = gross_eq_ret + gross_ma_ret - daily_tc.fillna(0)
    
    # 6. Package Results
    res = pd.DataFrame({
        'Prob_High_Vol': prob_high,
        'Gross_Ret': gross_eq_ret + gross_ma_ret,
        'Daily_TC': daily_tc,
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
        "Avg_Annual_Turnover": (turnover_eq + turnover_ma).mean() * 252
    })
    return res, summary


# ==========================================
# MAIN EXECUTION SCRIPT
# ==========================================

if __name__ == "__main__":
    
    data_dir = Path("data")
    test_start_date, test_end_date = "2010-01-01", "2025-12-31"
    
    # Costs: Equities 15bps, Macro ETFs (very liquid) 5bps
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
    
    # 1. Macro Assets
    macro_prices = fetch_carry_crash_havens(start_date=test_start_date, data_dir=data_dir)
    macro_prices = macro_prices.loc[test_start_date:test_end_date]
    
    # 2. Equities
    all_equities = pd.concat([strategy_data.large_cap_prices, strategy_data.small_cap_prices], axis=1)
    all_equities = all_equities.loc[:, ~all_equities.columns.duplicated()] 
    
    # 3. HMM Forecasts
    hmm_path = data_dir / "portfolio_hmm_oos_2010_forecasts.csv"
    hmm_df = pd.read_csv(hmm_path, index_col="Date", parse_dates=True)
    hmm_prob_high_vol = hmm_df["Next_Day_Forecast_Prob_High_Vol"].loc[test_start_date:test_end_date]
    
    # 4. Execute Carry & Crash
    print("\nExecuting Carry & Crash Strategy...")
    backtest_df, performance_summary = run_carry_crash_strategy(
        all_equities,
        macro_prices,
        hmm_prob_high_vol,
        tc_eq_bps=TC_BPS_EQUITY,
        tc_macro_bps=TC_BPS_MACRO
    )
    
    print("\n--- PERFORMANCE SUMMARY (2010-2025) ---")
    print(performance_summary.to_string())
    
    # 5. Visualization
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=backtest_df.index, y=backtest_df["Cum_Net_Strategy"], 
                   mode="lines", name="Carry & Crash Strategy", line=dict(color="#2ca02c", width=2.5)),
        secondary_y=False
    )
    
    mkt_bnh = (1 + strategy_data.market_return.loc[backtest_df.index].fillna(0)).cumprod()
    fig.add_trace(
        go.Scatter(x=mkt_bnh.index, y=mkt_bnh, 
                   mode="lines", name="Market Buy & Hold", line=dict(color="#7f7f7f", width=1.5, dash="dash")),
        secondary_y=False
    )
    
    fig.add_trace(
        go.Scatter(x=backtest_df.index, y=backtest_df["Prob_High_Vol"], 
                   mode="lines", name="Safe-Haven Allocation (Cash/Gold)", 
                   line=dict(color="rgba(214, 39, 40, 0)"), fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.15)"),
        secondary_y=True
    )
    
    fig.update_layout(
        title="OOS Performance: Carry & Crash",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_yaxes(title_text="Growth of $1", secondary_y=False)
    fig.update_yaxes(title_text="Safe-Haven Allocation %", secondary_y=True, range=[0, 1], tickformat=".0%")
    
    fig.show()