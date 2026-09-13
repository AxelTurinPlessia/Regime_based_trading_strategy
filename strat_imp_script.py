"""
Regime-Switched Trading Strategy (Continuous Blending)

This script acts as the orchestration layer for the integrated trading strategy.
It dynamically allocates capital between Volatility-Scaled Momentum (Small Caps) 
and Trend-Filtered Mean Reversion (Large Caps) based on the continuous 
probability output of a Gaussian HMM.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from pathlib import Path
from plotly.subplots import make_subplots

# Local imports based on your project structure
from project_dataset import load_regime_strategy_data
from regime_strategy import RegimeStrategyConfig

# Set pandas display options for terminal output
pd.options.display.float_format = "{:,.6f}".format
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

# ==========================================
# STRATEGY SIGNAL GENERATORS
# ==========================================

def panel_vol_scaled_momentum(prices_df, fast_window=21, slow_window=252, vol_window=20, target_vol=0.15, max_lev=2.0):
    """Generates inverse-volatility scaled momentum signals for a panel of stocks."""
    r = prices_df.pct_change(fill_method=None)
    fast_ma = prices_df.rolling(fast_window).mean()
    slow_ma = prices_df.rolling(slow_window).mean()
    
    raw_signal = pd.DataFrame(
        np.select([fast_ma > slow_ma, fast_ma < slow_ma], [1, -1], default=0), 
        index=prices_df.index, columns=prices_df.columns
    )
    raw_signal[slow_ma.isna()] = 0
    
    rolling_vol = r.rolling(vol_window).std() * np.sqrt(252)
    vol_scalar = (target_vol / rolling_vol.clip(lower=0.01)).clip(upper=max_lev)
    return raw_signal * vol_scalar.fillna(0)


def panel_trend_filtered_mr(prices_df, market_index, window=5, theta=1.0, trend_window=200):
    """Generates macro-trend-filtered mean reversion signals for a panel of stocks."""
    r = prices_df.pct_change(fill_method=None)
    mu = r.rolling(window).mean()
    sigma = r.rolling(window).std().replace(0, np.nan)
    z = (r - mu) / sigma
    
    macro_sma = market_index.rolling(trend_window).mean()
    is_uptrend = market_index > macro_sma
    
    positions = pd.DataFrame(0.0, index=prices_df.index, columns=prices_df.columns)
    for col in prices_df.columns:
        z_col, up_col = z[col].to_numpy(), is_uptrend.to_numpy()
        pos_arr = np.zeros(len(z_col))
        current_pos = 0
        for i in range(len(z_col)):
            z_val, is_up = z_col[i], up_col[i]
            if pd.isna(z_val): continue
            
            if current_pos == 0:
                if z_val < -theta and is_up: current_pos = 1
                elif z_val > theta and not is_up: current_pos = -1
            elif current_pos == 1 and z_val >= 0: current_pos = 0
            elif current_pos == -1 and z_val <= 0: current_pos = 0
            pos_arr[i] = current_pos
        positions[col] = pos_arr
    return positions


# ==========================================
# ORCHESTRATOR & PORTFOLIO NETTING
# ==========================================

def run_blended_regime_strategy(small_prices, large_prices, hmm_prob, mkt_ret, tc_small_bps, tc_large_bps, cfg):
    """Blends strategies continuously using HMM probabilities and applies netted transaction costs."""
    mkt_idx = (1 + mkt_ret.fillna(0)).cumprod()
    mom_pos = panel_vol_scaled_momentum(small_prices, cfg.momentum_fast_window, cfg.momentum_slow_window)
    mr_pos = panel_trend_filtered_mr(large_prices, mkt_idx, cfg.mean_reversion_window, cfg.mean_reversion_theta)
    
    r_small = small_prices.pct_change(fill_method=None)
    r_large = large_prices.pct_change(fill_method=None)
    
    common_idx = hmm_prob.index.intersection(small_prices.index)
    prob_high = hmm_prob.loc[common_idx].fillna(method='ffill')
    prob_low = 1.0 - prob_high
    
    n_small, n_large = len(small_prices.columns), len(large_prices.columns)
    w_small = mom_pos.loc[common_idx].multiply(prob_low, axis=0) / n_small
    w_large = mr_pos.loc[common_idx].multiply(prob_high, axis=0) / n_large
    
    turnover_small = w_small.diff().abs().sum(axis=1)
    turnover_large = w_large.diff().abs().sum(axis=1)
    
    daily_tc = (turnover_small * (tc_small_bps / 10000)) + (turnover_large * (tc_large_bps / 10000))
    
    gross_ret = (w_small.shift(1) * r_small.loc[common_idx]).sum(axis=1) + \
                (w_large.shift(1) * r_large.loc[common_idx]).sum(axis=1)
    net_ret = gross_ret - daily_tc.fillna(0)
    
    res = pd.DataFrame({
        'Prob_High_Vol': prob_high,
        'Gross_Ret': gross_ret,
        'Daily_TC': daily_tc,
        'Turnover': turnover_small + turnover_large,
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
    
    # 1. Configuration setup
    data_dir = Path("data")
    dividend_policy = "no_dividends"
    universe_price_source = "folders"
    market_return_mode = "average"
    
    test_start_date = "2010-01-01"
    test_end_date = "2025-12-31"
    
    config = RegimeStrategyConfig(
        split_date=test_start_date,
        momentum_fast_window=21,
        momentum_slow_window=252,
        mean_reversion_window=5,
        mean_reversion_theta=0.75,
        use_log_returns=False,
        regime_probability_threshold=0.50, # Bypassed by continuous blending
        liquidate_on_regime_change=False,
    )
    
    BASE_TC_BPS_SMALL = 30.0
    BASE_TC_BPS_LARGE = 10.0
    
    print("Loading data...")
    strategy_data = load_regime_strategy_data(
        data_dir=data_dir,
        dividend_policy=dividend_policy,
        market_return_mode=market_return_mode,
        universe_price_source=universe_price_source,
        start_date=test_start_date,
        end_date=test_end_date,
    )
    
    hmm_forecasts_path = data_dir / "portfolio_hmm_oos_2010_forecasts.csv"
    if not hmm_forecasts_path.exists():
        raise FileNotFoundError(f"Missing HMM forecasts at {hmm_forecasts_path}")
    
    hmm_df = pd.read_csv(hmm_forecasts_path, index_col="Date", parse_dates=True)
    hmm_prob_high_vol = hmm_df["Next_Day_Forecast_Prob_High_Vol"].loc[test_start_date:test_end_date]
    
    print(f"Small-cap panel shape: {strategy_data.small_cap_prices.shape}")
    print(f"Large-cap panel shape: {strategy_data.large_cap_prices.shape}")
    print(f"HMM Forecasts loaded: {len(hmm_prob_high_vol)} days")
    
    # 2. Execute Strategy
    print("\nExecuting blended strategy...")
    backtest_df, performance_summary = run_blended_regime_strategy(
        strategy_data.small_cap_prices,
        strategy_data.large_cap_prices,
        hmm_prob_high_vol,
        strategy_data.market_return,
        tc_small_bps=BASE_TC_BPS_SMALL,
        tc_large_bps=BASE_TC_BPS_LARGE,
        cfg=config
    )
    
    print("\n--- OUT-OF-SAMPLE PERFORMANCE (2010-2025) ---")
    print(performance_summary.to_string())
    
    # 3. Visualization
    print("\nGenerating performance plots in browser...")
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    
    fig.add_trace(
        go.Scatter(x=backtest_df.index, y=backtest_df["Cum_Net_Strategy"], 
                   mode="lines", name="Blended Regime Strategy", line=dict(color="#2ca02c", width=2.5)),
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
                   mode="lines", name="HMM Prob High Vol (Alloc to Large Cap MR)", 
                   line=dict(color="rgba(214, 39, 40, 0)"), fill="tozeroy", fillcolor="rgba(214, 39, 40, 0.15)"),
        secondary_y=True
    )
    
    fig.update_layout(
        title="OOS Performance & Continuous Capital Allocation",
        xaxis_title="Date",
        template="plotly_white",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0.0),
    )
    fig.update_yaxes(title_text="Growth of $1", secondary_y=False)
    fig.update_yaxes(title_text="High Vol Allocation %", secondary_y=True, range=[0, 1], tickformat=".0%")
    
    fig.show()
    
    # 4. Transaction Cost Sweep
    print("Running transaction cost sweep...")
    cost_multipliers = np.linspace(0.0, 5.0, 21)
    sweep_results = []
    
    for mult in cost_multipliers:
        tc_s = BASE_TC_BPS_SMALL * mult
        tc_l = BASE_TC_BPS_LARGE * mult
        _, smry = run_blended_regime_strategy(
            strategy_data.small_cap_prices, strategy_data.large_cap_prices,
            hmm_prob_high_vol, strategy_data.market_return,
            tc_s, tc_l, config
        )
        sweep_results.append({
            "Multiplier": mult,
            "Small_Cap_Bps": tc_s,
            "Large_Cap_Bps": tc_l,
            "Net_Total_Return": smry["Net_Total_Return"],
            "Net_Annualized_Return": smry["Net_Annualized_Return"]
        })
    
    cost_sweep = pd.DataFrame(sweep_results)
    
    try:
        break_even_idx = cost_sweep[cost_sweep["Net_Annualized_Return"] <= 0].index[0]
        break_even_small_bps = cost_sweep.loc[break_even_idx, "Small_Cap_Bps"]
    except IndexError:
        break_even_small_bps = None
    
    cost_fig = go.Figure()
    cost_fig.add_trace(go.Scatter(x=cost_sweep["Small_Cap_Bps"], y=cost_sweep["Net_Total_Return"], mode="lines+markers", name="Net total return", line=dict(color="#1f77b4", width=2)))
    cost_fig.add_trace(go.Scatter(x=cost_sweep["Small_Cap_Bps"], y=cost_sweep["Net_Annualized_Return"], mode="lines+markers", name="Net annualized return", line=dict(color="#ff7f0e", width=2)))
    cost_fig.add_hline(y=0.0, line_dash="dash", line_color="black")
    
    if break_even_small_bps:
        cost_fig.add_vline(x=break_even_small_bps, line_dash="dot", line_color="#d62728", 
                           annotation_text="Break-Even", annotation_position="top right")
    
    cost_fig.update_layout(
        title="Profitability vs Transaction Cost (X-Axis tracks Small Cap Bps, Large Cap is 1/3x)",
        xaxis_title="Small Cap Transaction Cost (bps)",
        yaxis_title="Return",
        template="plotly_white",
        hovermode="x unified",
    )
    cost_fig.show()
    print("Run complete.")