"""
Regime-Switched Trading Strategy: The "All-Equity Beta Slider"  --  v2
======================================================================

    Risk-On  (low volatility):  long the top N momentum names across the whole universe.
    Risk-Off (high volatility): long the top N minimum-volatility names from the large caps.
    Orchestrator:               blend the two sleeves on the HMM probability,
                                w_t = pi_t * w_def + (1 - pi_t) * w_mom.

OUTPUTS
-------
    figures/fig_holdout_ranking.png        PAPER FIGURE, holdout ranking
    figures/fig_equity_curve.png           PAPER FIGURE, growth of $1 vs both benchmarks
    figures/fig_degradation.png            PAPER FIGURE, tuning vs holdout Sharpe
    figures/fig1_directional_switch.html   exploratory equity curve, generation one
    figures/fig2_beta_slider.html          exploratory equity curve, generation two
    output/summary_tuning.csv              all strategies, tuning window
    output/summary_holdout.csv             all strategies, holdout window 
    output/paper_table1.tex                LaTeX Table 1, holdout window
    output/tuning_grid.csv                 every parameter combination tried, ranked
    output/decomposition_ladder.csv        mechanism effect vs sleeve effect vs signal value
    output/diagnostics.txt                 overlap, betas, placebo, configuration

"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from project_dataset import load_regime_strategy_data
from regime_strategy import run_strategy_for_universe

pd.options.display.float_format = "{:,.6f}".format
warnings.filterwarnings("ignore", category=FutureWarning)

TRADING_DAYS = 252


# ==========================================
# CONFIGURATION
# ==========================================

@dataclass
class Config:
    data_dir: Path = Path("data")
    fig_dir: Path = Path("figures")
    out_dir: Path = Path("output")

    # Windows. Signals are built from `buffer_start` so the rolling windows are warm on the
    # first evaluation day; results are sliced afterwards.
    buffer_start: str = "2009-01-01"
    test_start: str = "2010-01-01"
    tune_end: str = "2016-12-31"      # everything after this is HOLDOUT. Do not tune on it.
    test_end: str = "2025-12-31"
    warmup_signals: bool = True

    # Sleeve parameters. NOT tuned: they come from the literature and from the paper, and
    # tuning them on 41 stocks would be curve fitting with a straight face.
    mom_lookback: int = 126           # 6-month formation window
    vol_window: int = 63              # 3-month realised volatility
    top_n: int = 20

    # Generation-one sleeve parameters, passed through to regime_strategy.
    gen1_momentum_fast_window: int = 20
    gen1_momentum_slow_window: int = 50
    gen1_mean_reversion_window: int = 5
    gen1_mean_reversion_theta: float = 1.0
    gen1_exposure_convention: str = "active"   # "universe" reproduces the original bug

    # Orchestration
    discrete_threshold: float = 0.90
    liquidate_on_switch: bool = True

    # --- Tuned parameters. These values are OVERWRITTEN by the tuning pass. ---
    rebalance: str = "W"
    no_trade_band: float = 0.0025
    prob_halflife: float | None = 10.0
    signal_strength: float = 1.0

    # Tuning grid. Deliberately small: 36 combinations on seven years of daily data is
    # already generous, and every extra axis buys another chance to fit noise.
    run_tuning: bool = True
    grid_rebalance: tuple = ("D", "W", "M")
    grid_no_trade_band: tuple = (0.0, 0.0025, 0.005)
    grid_prob_halflife: tuple = (None, 10.0)
    grid_signal_strength: tuple = (0.75, 1.0)

    # Costs, one-way basis points per unit of notional traded.
    tc_bps_large: float = 15.0
    tc_bps_small: float = 25.0
    charge_drift: bool = True

    # Risk-free rate: a T-bill ETF price series; daily returns are used as rf.
    risk_free_file: str = "BIL_cc.csv"
    rf_annual: float | None = None    # fallback constant if the file is unusable

    # Controls
    placebo_shift_days: int = 400
    cost_grid: tuple = (0.0, 5.0, 10.0, 15.0, 25.0, 40.0, 60.0, 100.0)

    dividend_policy: str = "no_dividends"

    def __post_init__(self) -> None:
        self.fig_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir.mkdir(parents=True, exist_ok=True)

    @property
    def signal_start(self) -> str:
        return self.buffer_start if self.warmup_signals else self.test_start


# ==========================================
# SLEEVES
# ==========================================

def risk_on_high_momentum(all_prices: pd.DataFrame, lookback_days: int = 126,
                          top_n: int = 20) -> pd.DataFrame:
    """
    RISK-ON ENGINE. Rank every name on its trailing `lookback_days` price return and
    equal-weight the top N. The loader supplies prices without dividends, so this is a price
    return, not a total return.
    """
    lookback_return = all_prices.pct_change(periods=lookback_days, fill_method=None)
    ranks = lookback_return.rank(axis=1, ascending=False)
    target = (ranks <= top_n).astype(float)
    row_sums = target.sum(axis=1)
    return target.div(row_sums.where(row_sums > 0, np.nan), axis=0).fillna(0.0)


def risk_off_minimum_volatility(large_cap_prices: pd.DataFrame, vol_window: int = 63,
                                top_n: int = 20) -> pd.DataFrame:
    """
    RISK-OFF ENGINE. Rank the large-cap panel on realised volatility of daily returns and
    equal-weight the N most stable names. Diagonal of the covariance matrix only.
    """
    daily_returns = large_cap_prices.pct_change(fill_method=None)
    rolling_vol = daily_returns.rolling(vol_window).std()
    ranks = rolling_vol.rank(axis=1, ascending=True)
    target = (ranks <= top_n).astype(float)
    row_sums = target.sum(axis=1)
    return target.div(row_sums.where(row_sums > 0, np.nan), axis=0).fillna(0.0)


def build_generation_one_weights(small_cap_prices: pd.DataFrame,
                                 large_cap_prices: pd.DataFrame,
                                 all_columns: pd.Index,
                                 cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generation-one sleeves, built by calling the production code in regime_strategy so this
    script cannot drift away from the strategy that was actually run. The persistence sleeve
    is a moving-average crossover that can hold SHORT positions; the paper's description of
    it as a "buy-and-hold" was wrong.
    """
    persistence = run_strategy_for_universe(
        price_panel=small_cap_prices,
        strategy_name="momentum",
        momentum_fast_window=cfg.gen1_momentum_fast_window,
        momentum_slow_window=cfg.gen1_momentum_slow_window,
        exposure_convention=cfg.gen1_exposure_convention,
    )
    reversion = run_strategy_for_universe(
        price_panel=large_cap_prices,
        strategy_name="mean_reversion",
        mean_reversion_window=cfg.gen1_mean_reversion_window,
        mean_reversion_theta=cfg.gen1_mean_reversion_theta,
        exposure_convention=cfg.gen1_exposure_convention,
    )
    w_persistence = persistence["weights"].reindex(columns=all_columns).fillna(0.0)
    w_reversion = reversion["weights"].reindex(columns=all_columns).fillna(0.0)
    return w_persistence, w_reversion


# ==========================================
# SIGNAL TRANSFORMS
# ==========================================

def transform_probability(prob_high: pd.Series, halflife: float | None = None,
                          signal_strength: float = 1.0) -> pd.Series:
    """
    Smooth and temper the regime probability.

    `halflife` applies a backward-looking EWMA, damping day-to-day flicker in the filter and
    so reducing turnover. It uses no future information.

    `signal_strength` shrinks the signal toward a static 50/50 blend:
        pi_eff = 0.5 + lambda * (pi - 0.5)
    lambda = 0 removes the regime signal entirely; lambda = 1 leaves it untouched. If the
    tuner selects a low lambda, that is direct evidence the signal adds little, and it should
    be reported rather than quietly absorbed.
    """
    transformed = prob_high.copy()
    if halflife:
        transformed = transformed.ewm(halflife=halflife, adjust=False).mean()
    transformed = 0.5 + signal_strength * (transformed - 0.5)
    return transformed.clip(0.0, 1.0)


def make_placebo_probability(prob_high: pd.Series, shift_days: int) -> pd.Series:
    """
    Circularly shift the probability series. Timing information is destroyed; the marginal
    distribution and persistence structure are preserved. A strategy that does as well on the
    placebo as on the real signal is not using the signal.
    """
    values = np.roll(prob_high.to_numpy(), shift_days)
    return pd.Series(values, index=prob_high.index, name="Placebo_Prob_High_Vol")


# ==========================================
# ORCHESTRATORS
# ==========================================

def orchestrate_blend(w_mom: pd.DataFrame, w_def: pd.DataFrame,
                      prob_high: pd.Series) -> pd.DataFrame:
    """Continuous Beta Slider: w_t = pi_t * w_def + (1 - pi_t) * w_mom."""
    return w_mom.multiply(1.0 - prob_high, axis=0) + w_def.multiply(prob_high, axis=0)


def orchestrate_switch(w_mom: pd.DataFrame, w_def: pd.DataFrame, prob_high: pd.Series,
                       threshold: float = 0.90,
                       liquidate_on_switch: bool = True) -> tuple[pd.DataFrame, pd.Series]:
    """
    Generation-one discrete switch with hysteresis, matching
    regime_rules.build_probability_confirmed_regime. `liquidate_on_switch` reproduces the
    mandatory full exit on a regime change.
    """
    state = pd.Series(0, index=prob_high.index, dtype=int)
    current = 1 if prob_high.iloc[0] >= 0.5 else 0
    for timestamp, probability in prob_high.items():
        if probability >= threshold:
            current = 1
        elif probability <= 1.0 - threshold:
            current = 0
        state.loc[timestamp] = current

    weights = w_def.mul(state, axis=0) + w_mom.mul(1 - state, axis=0)
    if liquidate_on_switch:
        switched = state.diff().fillna(0) != 0
        weights.loc[switched.values, :] = 0.0
    return weights, state


def apply_rebalance_frequency(weights: pd.DataFrame, freq: str = "D") -> pd.DataFrame:
    """
    Hold target weights constant between rebalance dates. "D" is a no-op.

    Dropping this in v1 is what produced 2,904% annual turnover: the engine reset the target
    to exact equal weights every day and, with drift charged, paid for it every day.
    """
    if freq.upper() == "D":
        return weights

    alias = {"D": "D", "W": "W", "M": "M", "Q": "Q", "Y": "A", "A": "A"}.get(
        freq.upper(), freq.upper())
    periods = weights.index.to_period(alias)
    marks = pd.Series(weights.index, index=periods).groupby(level=0).max().to_numpy()

    held = weights.copy()
    held.loc[~weights.index.isin(marks), :] = np.nan

    # Seed from the first row that actually holds something, so a warm-up row of zeros is
    # never latched in as the standing target.
    invested_rows = np.flatnonzero(weights.abs().sum(axis=1).to_numpy() > 0)
    if len(invested_rows):
        first = invested_rows[0]
        held.iloc[first] = weights.iloc[first]
    return held.ffill().fillna(0.0)


# ==========================================
# BACKTEST ENGINE
# ==========================================

def run_backtest(target_weights: pd.DataFrame, asset_returns: pd.DataFrame,
                 cost_bps_per_name: pd.Series, charge_drift: bool = True,
                 no_trade_band: float = 0.0) -> pd.DataFrame:
    """
    Weights held on date t-1 earn the return on date t, so no signal ever trades the bar that
    generated it.

    Turnover is measured against DRIFTED weights: between rebalances, realised returns move
    the book away from its target and restoring it is a real trade. Setting
    charge_drift=False reproduces the original target-to-target convention, which understates
    cost.

    `no_trade_band` is the key friction control. A name trades only if its target differs
    from its drifted weight by more than the band; otherwise the drifted weight is kept. This
    removes the cost of continually restoring exact equal weights without pretending that
    cost does not exist.
    """
    columns = target_weights.columns
    targets = target_weights.fillna(0.0).to_numpy(dtype=float)
    returns = (asset_returns.reindex(index=target_weights.index, columns=columns)
               .fillna(0.0).to_numpy(dtype=float))
    costs = (cost_bps_per_name.reindex(columns).fillna(cost_bps_per_name.median())
             .to_numpy(dtype=float) / 1e4)

    n_days = targets.shape[0]
    gross = np.zeros(n_days)
    cost = np.zeros(n_days)
    turnover = np.zeros(n_days)
    invested = np.zeros(n_days)

    held = np.zeros(targets.shape[1])
    for t in range(n_days):
        if t == 0:
            new_weights = targets[0]
            trade = np.abs(new_weights)
        else:
            portfolio_return = float(held @ returns[t])
            gross[t] = portfolio_return
            if charge_drift:
                drifted = held * (1.0 + returns[t]) / (1.0 + portfolio_return)
            else:
                drifted = held
            if no_trade_band > 0.0:
                move = np.abs(targets[t] - drifted) > no_trade_band
                new_weights = np.where(move, targets[t], drifted)
            else:
                new_weights = targets[t]
            trade = np.abs(new_weights - drifted)
        held = new_weights
        turnover[t] = trade.sum()
        cost[t] = float(costs @ trade)
        invested[t] = held.sum()

    net = gross - cost
    return pd.DataFrame(
        {
            "Gross_Return": gross,
            "Daily_TC": cost,
            "Turnover": turnover,
            "Net_Strategy_Return": net,
            "Cum_Net_Strategy": (1.0 + net).cumprod(),
            "Cum_Gross_Strategy": (1.0 + gross).cumprod(),
            "Invested": invested,
        },
        index=target_weights.index,
    )


# ==========================================
# RISK-FREE RATE
# ==========================================

def load_risk_free_returns(cfg: Config, index: pd.DatetimeIndex) -> tuple[pd.Series, bool]:
    """
    Daily risk-free returns from a T-bill ETF price series. Returns (series, is_real).

    Handles the common column layouts defensively. If nothing usable is found the function
    falls back to a constant (or zero) and the caller relabels the risk metric accordingly.
    """
    path = cfg.data_dir / cfg.risk_free_file
    if path.exists():
        try:
            table = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
            column = next((c for c in ("Adj Close", "Close", "close", "PX_LAST", "Price")
                           if c in table.columns), None)
            if column is None:
                numeric = table.select_dtypes(include=[np.number])
                column = numeric.columns[0] if len(numeric.columns) else None
            if column is not None:
                series = pd.to_numeric(table[column], errors="coerce").dropna()
                if series.median() > 1.5:
                    rf = series.pct_change(fill_method=None)          # price level
                elif series.median() > 0.0005:
                    rf = series / TRADING_DAYS                        # annualised rate
                else:
                    rf = series                                       # already daily
                rf = rf.reindex(index).ffill().fillna(0.0)
                print(f"  risk-free: {path.name} column '{column}', "
                      f"{rf.mean() * TRADING_DAYS:.2%} annualised over the sample")
                return rf.rename("RF"), True
        except Exception as exc:
            print(f"  [warn] could not read {path}: {exc}")

    constant = (cfg.rf_annual / TRADING_DAYS) if cfg.rf_annual else 0.0
    print(f"  risk-free: no usable file, using constant {constant * TRADING_DAYS:.2%}")
    return pd.Series(constant, index=index, name="RF"), bool(cfg.rf_annual)


# ==========================================
# PERFORMANCE METRICS
# ==========================================

def drawdown_series(cumulative: pd.Series) -> pd.Series:
    return cumulative / cumulative.cummax() - 1.0


def annualised_return(returns: pd.Series) -> float:
    cleaned = pd.Series(returns).fillna(0.0)
    if cleaned.empty:
        return np.nan
    return float((1.0 + cleaned).prod() ** (TRADING_DAYS / len(cleaned)) - 1.0)


def performance_summary(net_returns: pd.Series, rf_daily: pd.Series | float = 0.0,
                        label: str = "") -> pd.Series:
    returns = pd.Series(net_returns).fillna(0.0)
    if isinstance(rf_daily, pd.Series):
        rf = rf_daily.reindex(returns.index).fillna(0.0)
    else:
        rf = pd.Series(rf_daily, index=returns.index)

    cumulative = float((1.0 + returns).prod())
    ann_return = cumulative ** (TRADING_DAYS / len(returns)) - 1.0
    ann_vol = float(returns.std() * np.sqrt(TRADING_DAYS))

    excess = returns - rf
    ann_excess = float(excess.mean() * TRADING_DAYS)
    sharpe = ann_excess / ann_vol if ann_vol else np.nan

    downside = returns[returns < 0]
    downside_dev = float(downside.std() * np.sqrt(TRADING_DAYS)) if len(downside) else np.nan
    sortino = ann_excess / downside_dev if downside_dev else np.nan

    drawdown = drawdown_series((1.0 + returns).cumprod())
    max_dd = float(drawdown.min())

    return pd.Series(
        {
            "Terminal_Wealth": cumulative,
            "Ann_Return": ann_return,
            "Ann_Volatility": ann_vol,
            "Sharpe": sharpe,
            "Sortino": sortino,
            "Calmar": ann_return / abs(max_dd) if max_dd else np.nan,
            "Max_Drawdown": max_dd,
            "Skewness": float(returns.skew()),
            "Excess_Kurtosis": float(returns.kurtosis()),
            "Hit_Rate": float((returns > 0).mean()),
        },
        name=label or "strategy",
    )


# ==========================================
# MEASURED CLAIMS
# ==========================================

def sleeve_overlap(w_mom: pd.DataFrame, w_def: pd.DataFrame) -> pd.DataFrame:
    """
    The netting channel. A name held by both sleeves has blended weight
    pi/N + (1-pi)/N = 1/N regardless of pi, so it generates no turnover as the regime moves.
    """
    both = ((w_mom > 0) & (w_def > 0)).sum(axis=1)
    n_mom = (w_mom > 0).sum(axis=1)
    return pd.DataFrame({
        "n_overlap": both,
        "n_mom": n_mom,
        "n_def": (w_def > 0).sum(axis=1),
        "overlap_frac": both / n_mom.replace(0, np.nan),
    })


def sleeve_beta(sleeve_weights: pd.DataFrame, asset_returns: pd.DataFrame,
                market_returns: pd.Series, prob_high: pd.Series) -> dict:
    """Full-sample, state-conditional and downside betas of a sleeve held on its own."""
    sleeve = (sleeve_weights.shift(1) * asset_returns.reindex_like(sleeve_weights)).sum(axis=1)
    market = market_returns.reindex(sleeve.index).fillna(0.0)
    invested = sleeve_weights.sum(axis=1).shift(1) > 0
    sleeve, market = sleeve[invested], market[invested]

    def beta(y: pd.Series, x: pd.Series) -> float:
        if len(y) < 30 or x.var() == 0:
            return np.nan
        return float(np.cov(y, x)[0, 1] / x.var())

    high = prob_high.reindex(sleeve.index).fillna(0.0) > 0.5
    worst = market <= market.quantile(0.10)
    return {
        "beta_full_sample": beta(sleeve, market),
        "beta_high_vol_state": beta(sleeve[high], market[high]),
        "beta_low_vol_state": beta(sleeve[~high], market[~high]),
        "beta_worst_market_decile": beta(sleeve[worst], market[worst]),
        "ann_return": annualised_return(sleeve),
        "ann_volatility": float(sleeve.std() * np.sqrt(TRADING_DAYS)),
    }


# ==========================================
# EXPLORATORY FIGURES (plotly, not used by the paper)
# ==========================================

def plot_strategy_vs_market(strategy_curve: pd.Series, market_curve: pd.Series,
                            large_cap_curve: pd.Series, prob_high: pd.Series,
                            title: str, strategy_label: str,
                            shade_label: str, shade_axis_title: str,
                            split_date, path: Path) -> None:
    """
    Growth of $1 for a strategy against BOTH size-matched passive benchmarks, with the regime
    probability shaded and the holdout boundary marked.

    `market_curve` must be built from the traded universe, NOT from the cached market series.
    See F3 in the module docstring.
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        print(f"  [skip] plotly is not installed, cannot write {path}")
        return

    figure = make_subplots(specs=[[{"secondary_y": True}]])
    figure.add_trace(
        go.Scatter(x=prob_high.index, y=prob_high.values, mode="lines", name=shade_label,
                   line=dict(color="rgba(231, 76, 60, 0)", width=0), fill="tozeroy",
                   fillcolor="rgba(231, 76, 60, 0.18)", hoverinfo="skip"),
        secondary_y=True)
    figure.add_trace(
        go.Scatter(x=strategy_curve.index, y=strategy_curve.values, mode="lines",
                   name=strategy_label, line=dict(color="#2ca02c", width=2.0)),
        secondary_y=False)
    figure.add_trace(
        go.Scatter(x=market_curve.index, y=market_curve.values, mode="lines",
                   name="Buy & Hold (traded universe, 41)",
                   line=dict(color="#7f7f7f", width=1.6, dash="dash")),
        secondary_y=False)
    figure.add_trace(
        go.Scatter(x=large_cap_curve.index, y=large_cap_curve.values, mode="lines",
                   name="Buy & Hold (large caps only, 29)",
                   line=dict(color="#444444", width=1.4, dash="dot")),
        secondary_y=False)
    if split_date is not None:
        # add_vline() with a Timestamp routes through plotly's axis-spanning annotation
        # helper, which averages the x endpoints and blows up on datetime axes
        # ("Addition/subtraction of integers ... with Timestamp is no longer supported").
        # Drawing the shape and the label separately avoids that code path entirely.
        figure.add_shape(type="line", x0=split_date, x1=split_date, xref="x",
                         y0=0, y1=1, yref="paper",
                         line=dict(color="#333333", width=1, dash="dot"))
        figure.add_annotation(x=split_date, xref="x", y=1.0, yref="paper",
                              text="holdout begins", showarrow=False,
                              xanchor="left", yanchor="bottom", font=dict(size=10))
    figure.update_yaxes(title_text="Growth of $1", secondary_y=False)
    figure.update_yaxes(title_text=shade_axis_title, range=[0, 1], tickformat=".0%",
                        showgrid=False, secondary_y=True)
    figure.update_layout(title=title, xaxis_title="Date", template="plotly_white",
                         hovermode="x unified",
                         legend=dict(orientation="h", yanchor="bottom", y=1.02,
                                     xanchor="left", x=0.0))
    figure.write_html(str(path))
    print(f"  wrote {path}")


# ==========================================
# PAPER FIGURES
# ==========================================

# (summary key, display label, category) for the ranking figure. Order is the order plotted.
PAPER_FIGURE_ROWS = [
    ("Original (Discrete Switch)",       "Original\n(Discrete Switch)", "gen1"),
    ("Final (Beta Slider)",              "Beta Slider",                 "active"),
    ("CONTROL: static 50/50 blend",      "Static 50/50\n(no signal)",   "control"),
    ("CONTROL: placebo signal",          "Placebo\nsignal",             "control"),
    ("Momentum sleeve only",             "Momentum\nonly",              "sleeve"),
    ("Min-vol sleeve only",              "Min-vol\nonly",               "sleeve"),
    ("Buy & Hold (traded universe, 41)", "B&H traded\nuniverse (41)",   "bench"),
    ("Buy & Hold (large caps only, 29)", "B&H large\ncaps (29)",        "bench"),
]

# Architecture One is excluded from the degradation scatter: its Sharpe sits near -0.9 in
# both windows and would compress the interesting range to nothing. The paper's wording
# ("largest degradation among those plotted") is written to match this exclusion.
DEGRADATION_ROWS = [
    ("Final (Beta Slider)",              "Beta Slider",      "o"),
    ("CONTROL: static 50/50 blend",      "Static 50/50",     "s"),
    ("CONTROL: placebo signal",          "Placebo",          "^"),
    ("Momentum sleeve only",             "Momentum only",    "v"),
    ("Min-vol sleeve only",              "Min-vol only",     "D"),
    ("Buy & Hold (traded universe, 41)", "B&H traded (41)",  "P"),
    ("Buy & Hold (large caps only, 29)", "B&H large (29)",   "X"),
]

CATEGORY_COLOURS = {"gen1": "#b02418", "active": "#2ca02c", "control": "#9467bd",
                    "sleeve": "#1f77b4", "bench": "#7f7f7f"}

REFERENCE_ROW = "Buy & Hold (large caps only, 29)"


def plot_paper_figures(summaries: dict, cfg: Config) -> None:
    """
    Regenerate the two PNG figures the LaTeX paper includes:

        figures/fig_holdout_ranking.png   annualised return and Sharpe, holdout window
        figures/fig_degradation.png       tuning Sharpe against holdout Sharpe

    Both are built from the computed `summaries` rather than from transcribed numbers, so a
    rerun updates the paper figures automatically.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib is not installed, cannot write the paper figures")
        return

    hold, tune = summaries["holdout"], summaries["tuning"]
    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.titlesize": 10, "axes.titleweight": "bold",
                         "legend.frameon": False})

    first_holdout_year = int(cfg.tune_end[:4]) + 1
    last_year = cfg.test_end[:4]

    # ---- Figure 1: holdout ranking ----
    rows = [r for r in PAPER_FIGURE_ROWS if r[0] in hold.columns]
    if not rows or REFERENCE_ROW not in hold.columns:
        print("  [skip] no matching columns for the ranking figure")
        return
    labels = [r[1] for r in rows]
    colours = [CATEGORY_COLOURS[r[2]] for r in rows]
    ann = [float(hold.loc["Ann_Return", r[0]]) * 100 for r in rows]
    shp = [float(hold.loc["Sharpe", r[0]]) for r in rows]
    ref_ann = float(hold.loc["Ann_Return", REFERENCE_ROW]) * 100
    ref_shp = float(hold.loc["Sharpe", REFERENCE_ROW])

    figure, axes = plt.subplots(2, 1, figsize=(9.2, 6.4), sharex=True)
    positions = np.arange(len(rows))
    for axis, values, axis_label, reference in (
        (axes[0], ann, "Annualised return (%)", ref_ann),
        (axes[1], shp, "Sharpe ratio", ref_shp),
    ):
        axis.bar(positions, values, 0.66, color=colours, alpha=0.9)
        axis.axhline(0, color="black", lw=0.8)
        axis.axhline(reference, color="#333333", ls="--", lw=1.1,
                     label="passive large-cap benchmark")
        axis.set_ylabel(axis_label)
        for position, value in zip(positions, values):
            axis.annotate(f"{value:.2f}" if abs(value) < 3 else f"{value:.1f}",
                          (position, value), textcoords="offset points",
                          xytext=(0, 4 if value >= 0 else -12), ha="center", fontsize=8)
        axis.legend(loc="lower right")
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(labels, fontsize=8)
    figure.suptitle(f"Holdout ranking in the window of {first_holdout_year}-{last_year}",
                    fontsize=11, fontweight="bold")
    figure.tight_layout()
    figure.savefig(cfg.fig_dir / "fig_holdout_ranking.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {cfg.fig_dir / 'fig_holdout_ranking.png'}")

    # ---- Figure 2: in-sample against out-of-sample ----
    points = [r for r in DEGRADATION_ROWS
              if r[0] in hold.columns and r[0] in tune.columns]
    if not points:
        print("  [skip] no matching columns for the degradation figure")
        return
    figure, axis = plt.subplots(figsize=(6.6, 5.4))
    for key, label, marker in points:
        x, y = float(tune.loc["Sharpe", key]), float(hold.loc["Sharpe", key])
        colour = ("#2ca02c" if "Slider" in label
                  else "#7f7f7f" if label.startswith("B&H") else "#1f77b4")
        axis.scatter(x, y, marker=marker, s=95, color=colour, zorder=3, label=label)
    values = ([float(tune.loc["Sharpe", k]) for k, _, _ in points]
              + [float(hold.loc["Sharpe", k]) for k, _, _ in points])
    lo, hi = min(values) - 0.08, max(values) + 0.08
    axis.plot([lo, hi], [lo, hi], ls=":", color="#666666", lw=1.2, label="no degradation")
    axis.set_xlim(lo, hi)
    axis.set_ylim(lo, hi)
    axis.set_xlabel(f"Sharpe, tuning window {cfg.test_start[:4]}-{cfg.tune_end[:4]}")
    axis.set_ylabel(f"Sharpe, holdout window {first_holdout_year}-{last_year}")
    axis.set_title("Out of sample degradation")
    axis.legend(fontsize=8, loc="upper left", ncol=2)
    figure.tight_layout()
    figure.savefig(cfg.fig_dir / "fig_degradation.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {cfg.fig_dir / 'fig_degradation.png'}")


def plot_equity_curve_figure(strategy_curve, market_curve, large_cap_curve,
                             prob_high, split_date, cfg) -> None:
    """
    PAPER FIGURE. Growth of $1 for the Beta Slider against both size-matched passive
    benchmarks, with the regime probability shaded on a secondary axis and the tuning/holdout
    boundary marked.

    Note that the tuning window occupies the left-hand portion of this chart. Only the
    segment to the right of the dashed boundary is out of sample.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [skip] matplotlib is not installed, cannot write the equity-curve figure")
        return

    plt.rcParams.update({"font.size": 9, "axes.grid": True, "grid.alpha": 0.25,
                         "axes.spines.top": False, "legend.frameon": False})

    figure, axis = plt.subplots(figsize=(9.2, 5.0))

    shade = axis.twinx()
    shade.fill_between(prob_high.index, 0.0, prob_high.values,
                       color="#e74c3c", alpha=0.14, linewidth=0, zorder=1)
    shade.set_ylim(0, 1)
    shade.set_ylabel("Allocation to defensive sleeve")
    shade.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    shade.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    shade.grid(False)

    axis.set_zorder(shade.get_zorder() + 1)
    axis.patch.set_visible(False)
    axis.plot(large_cap_curve.index, large_cap_curve.values, color="#444444",
              lw=1.4, ls=":", label="Buy & hold, large caps (29)", zorder=3)
    axis.plot(market_curve.index, market_curve.values, color="#7f7f7f",
              lw=1.6, ls="--", label="Buy & hold, traded universe (41)", zorder=3)
    axis.plot(strategy_curve.index, strategy_curve.values, color="#2ca02c",
              lw=2.0, label="All-Equity Beta Slider", zorder=4)
    axis.set_yscale("log")
    axis.set_ylabel("Growth of $1 (log scale)")
    axis.set_xlabel("Date")

    if split_date is not None:
        axis.axvline(split_date, color="#333333", lw=1.0, ls="-.", zorder=2)
        top = axis.get_ylim()[1]
        axis.annotate("tuning", (split_date, top), xytext=(-6, -12),
                      textcoords="offset points", ha="right", va="top",
                      fontsize=8, color="#333333")
        axis.annotate("holdout", (split_date, top), xytext=(6, -12),
                      textcoords="offset points", ha="left", va="top",
                      fontsize=8, color="#333333", fontweight="bold")

    axis.legend(loc="upper left", fontsize=8)
    axis.set_title("Performance in the timeframe of 2010-2025",
                   fontsize=10, fontweight="bold")
    figure.tight_layout()
    figure.savefig(cfg.fig_dir / "fig_equity_curve.png", dpi=200, bbox_inches="tight")
    plt.close(figure)
    print(f"  wrote {cfg.fig_dir / 'fig_equity_curve.png'}")


# ==========================================
# REPORTING
# ==========================================

def write_paper_table1(summary: pd.DataFrame, turnovers: dict, columns: list[str],
                       window_label: str, path: Path) -> None:
    def pct(value) -> str:
        return "---" if pd.isna(value) else f"{value * 100:.1f}\\%"

    def num(value) -> str:
        return "---" if pd.isna(value) else f"{value:.2f}"

    columns = [c for c in columns if c in summary.columns]
    lines = [
        r"\begin{table}[H]", r"\centering",
        rf"\caption{{Comparative performance, {window_label}, net of trading costs.}}",
        r"\label{tab:main}",
        r"\begin{tabular}{@{}l" + "c" * len(columns) + r"@{}}", r"\toprule",
        r"Metric & " + " & ".join(c.replace("&", r"\&") for c in columns) + r" \\",
        r"\midrule",
    ]
    for label, key, formatter in [
        ("Terminal wealth on \\$1", "Terminal_Wealth", num),
        ("Annualised return", "Ann_Return", pct),
        ("Annualised volatility", "Ann_Volatility", pct),
        ("Sharpe ratio", "Sharpe", num),
        ("Sortino ratio", "Sortino", num),
        ("Calmar ratio", "Calmar", num),
        ("Max drawdown", "Max_Drawdown", pct),
    ]:
        lines.append(f"{label} & "
                     + " & ".join(formatter(summary.loc[key, c]) for c in columns) + r" \\")
    lines.append("Avg.\\ annual turnover & "
                 + " & ".join(pct(turnovers.get(c, np.nan)) for c in columns) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    Path(path).write_text("\n".join(lines))
    print(f"  wrote {path}")


def write_diagnostics(payload: dict, path: Path) -> None:
    with open(path, "w") as handle:
        for section, content in payload.items():
            handle.write(f"\n{'=' * 76}\n{section}\n{'=' * 76}\n")
            if isinstance(content, (pd.DataFrame, pd.Series)):
                handle.write(content.to_string() + "\n")
            elif isinstance(content, dict):
                for key, value in content.items():
                    handle.write(f"  {str(key):<46} {value}\n")
            else:
                handle.write(f"{content}\n")
    print(f"  wrote {path}")


# ==========================================
# MAIN
# ==========================================

def main(cfg: Config) -> dict:
    print("Loading data...")
    strategy_data = load_regime_strategy_data(
        data_dir=cfg.data_dir, dividend_policy=cfg.dividend_policy,
        market_return_mode="average", universe_price_source="folders",
        start_date=cfg.signal_start, end_date=cfg.test_end)

    large = strategy_data.large_cap_prices
    small = strategy_data.small_cap_prices
    all_equities = pd.concat([large, small], axis=1)
    all_equities = all_equities.loc[:, ~all_equities.columns.duplicated()]
    n_universe = all_equities.shape[1]
    print(f"  large caps: {large.shape[1]}   small caps: {small.shape[1]}   "
          f"universe: {n_universe}")
    if cfg.top_n > 0.6 * n_universe:
        print(f"  [WARNING] top_n={cfg.top_n} out of {n_universe} names is not a selection, "
              "it is near-total inclusion. Cross-sectional ranking has little room to work.")

    hmm_path = cfg.data_dir / "portfolio_hmm_oos_2010_forecasts.csv"
    if not hmm_path.exists():
        raise FileNotFoundError(f"Missing HMM forecasts at {hmm_path}. Run HMM.ipynb first.")
    prob_full = pd.read_csv(hmm_path, index_col="Date",
                            parse_dates=True)["Next_Day_Forecast_Prob_High_Vol"]

    cost_bps = pd.Series(cfg.tc_bps_large, index=all_equities.columns, dtype=float)
    small_only = [c for c in small.columns if c not in set(large.columns)]
    cost_bps.loc[small_only] = cfg.tc_bps_small

    # ---- Sleeves built on buffered history, then sliced ----
    print("Building sleeves...")
    w_mom_full = risk_on_high_momentum(all_equities, cfg.mom_lookback, cfg.top_n)
    w_def_full = (risk_off_minimum_volatility(large, cfg.vol_window, cfg.top_n)
                  .reindex(columns=all_equities.columns, fill_value=0.0))
    w_per_full, w_rev_full = build_generation_one_weights(
        small, large, all_equities.columns, cfg)

    index = all_equities.loc[cfg.test_start:cfg.test_end].index.intersection(prob_full.index)
    prob_raw = prob_full.reindex(index).ffill().clip(0.0, 1.0)
    w_mom = w_mom_full.reindex(index).fillna(0.0)
    w_def = w_def_full.reindex(index).fillna(0.0)
    w_per = w_per_full.reindex(index).fillna(0.0)
    w_rev = w_rev_full.reindex(index).fillna(0.0)
    asset_returns = all_equities.pct_change(fill_method=None).reindex(index)
    market = strategy_data.market_return.reindex(index).fillna(0.0)

    # Size-matched control for the defensive sleeve, which draws only from the large caps.
    large_cap_benchmark_return = (
        large.pct_change(fill_method=None).mean(axis=1).reindex(index).fillna(0.0)
        .rename("Large_Cap_Equal_Weight")
    )
    # `market` comes from the cached portfolio_returns_equal_weight.csv, whose columns are
    # cross-sectional means of LOG returns blended 50/50 large/small. Compounded as simple
    # returns it understates the SAME 41 securities by roughly 7.9 percentage points a year,
    # so it is not a like-for-like benchmark and must not be plotted or tabulated as one.
    # This is an equal weight of the panel actually traded.
    traded_benchmark_return = (
        all_equities.pct_change(fill_method=None).mean(axis=1).reindex(index).fillna(0.0)
        .rename("Traded_Universe_Equal_Weight")
    )
    implied_small = (
        small.pct_change(fill_method=None).mean(axis=1).reindex(index).fillna(0.0)
        .rename("Small_Cap_Equal_Weight")
    )
    print(f"  benchmark check: traded universe ({n_universe}) "
          f"{annualised_return(traded_benchmark_return):.2%}, "
          f"large caps ({large.shape[1]}) {annualised_return(large_cap_benchmark_return):.2%}, "
          f"small caps ({small.shape[1]}) {annualised_return(implied_small):.2%}, "
          f"cached market series {annualised_return(market):.2%}")

    rf_daily, rf_is_real = load_risk_free_returns(cfg, index)
    if not rf_is_real:
        print("  [note] 'Sharpe' below is computed against rf = 0, so it is a return/vol "
              "ratio. Label it that way in the paper.")

    split = pd.Timestamp(cfg.tune_end)
    tune_index = index[index <= split]
    hold_index = index[index > split]
    print(f"  tuning window : {tune_index.min().date()} to {tune_index.max().date()} "
          f"({len(tune_index)} days)")
    print(f"  HOLDOUT window: {hold_index.min().date()} to {hold_index.max().date()} "
          f"({len(hold_index)} days)")

    def evaluate(weights: pd.DataFrame, window, rebalance: str, band: float) -> pd.DataFrame:
        scheduled = apply_rebalance_frequency(weights.reindex(window), rebalance)
        return run_backtest(scheduled, asset_returns.reindex(window), cost_bps,
                            cfg.charge_drift, band)

    # ---- Tuning pass, on the tuning window ONLY ----
    if cfg.run_tuning:
        print("\nTuning on the 2010-2016 window (the holdout is never touched here)...")
        rows = []
        combinations = list(itertools.product(
            cfg.grid_rebalance, cfg.grid_no_trade_band,
            cfg.grid_prob_halflife, cfg.grid_signal_strength))
        for rebalance, band, halflife, strength in combinations:
            prob = transform_probability(prob_raw, halflife, strength)
            result = evaluate(orchestrate_blend(w_mom, w_def, prob),
                              tune_index, rebalance, band)
            stats = performance_summary(result["Net_Strategy_Return"], rf_daily)
            rows.append({
                "rebalance": rebalance, "no_trade_band": band,
                "prob_halflife": halflife, "signal_strength": strength,
                "Sharpe": stats["Sharpe"], "Ann_Return": stats["Ann_Return"],
                "Max_Drawdown": stats["Max_Drawdown"],
                "Turnover": float(result["Turnover"].mean() * TRADING_DAYS),
            })
        grid = pd.DataFrame(rows).sort_values("Sharpe", ascending=False).reset_index(drop=True)
        best = grid.iloc[0]
        cfg.rebalance = str(best["rebalance"])
        cfg.no_trade_band = float(best["no_trade_band"])
        cfg.prob_halflife = (None if pd.isna(best["prob_halflife"])
                             else float(best["prob_halflife"]))
        cfg.signal_strength = float(best["signal_strength"])
        print(f"  {len(combinations)} combinations tried; Sharpe spread across the grid "
              f"{grid['Sharpe'].min():.2f} to {grid['Sharpe'].max():.2f}")
        print(f"  selected: rebalance={cfg.rebalance}  band={cfg.no_trade_band}  "
              f"halflife={cfg.prob_halflife}  lambda={cfg.signal_strength}")
        if cfg.signal_strength <= 0.5:
            print("  [note] a low lambda means the tuner preferred to shrink the regime "
                  "signal toward a static blend. That is a result; report it.")
    else:
        grid = pd.DataFrame()

    # ---- Build every book with the selected parameters ----
    prob = transform_probability(prob_raw, cfg.prob_halflife, cfg.signal_strength)
    prob_placebo = transform_probability(
        make_placebo_probability(prob_raw, cfg.placebo_shift_days),
        cfg.prob_halflife, cfg.signal_strength)
    static_half = pd.Series(0.5, index=index)

    w_gen1_liq, gen1_state = orchestrate_switch(
        w_per, w_rev, prob, cfg.discrete_threshold, True)
    w_gen1_noliq, _ = orchestrate_switch(
        w_per, w_rev, prob, cfg.discrete_threshold, False)

    books = {
        "Original (Discrete Switch)": w_gen1_liq,
        "Original sleeves, no liquidation": w_gen1_noliq,
        "Original sleeves, continuous blend": orchestrate_blend(w_per, w_rev, prob),
        "Final (Beta Slider)": orchestrate_blend(w_mom, w_def, prob),
        "CONTROL: static 50/50 blend": orchestrate_blend(w_mom, w_def, static_half),
        "CONTROL: placebo signal": orchestrate_blend(w_mom, w_def, prob_placebo),
        "Momentum sleeve only": w_mom,
        "Min-vol sleeve only": w_def,
    }

    summaries, turnovers, results = {}, {}, {}
    for window_name, window in (("tuning", tune_index), ("holdout", hold_index)):
        window_summaries, window_turnovers, window_results = {}, {}, {}
        print(f"\n--- {window_name.upper()} window ---")
        for name, weights in books.items():
            result = evaluate(weights, window, cfg.rebalance, cfg.no_trade_band)
            window_results[name] = result
            window_summaries[name] = performance_summary(
                result["Net_Strategy_Return"], rf_daily, name)
            window_turnovers[name] = float(result["Turnover"].mean() * TRADING_DAYS)
            print(f"  {name:<36} ann {window_summaries[name]['Ann_Return']:>8.2%}   "
                  f"Sharpe {window_summaries[name]['Sharpe']:>5.2f}   "
                  f"turnover {window_turnovers[name]:>7.1%}")

        # CRITICAL CONTROLS. The min-vol sleeve draws only from the large-cap panel, so
        # comparing it with an all-41 benchmark confounds the low-volatility effect with a
        # plain large-cap-versus-small-cap tilt. The large-cap benchmark holds the size
        # exposure fixed. The cached series is reported only so the gap between it and the
        # traded universe stays visible; it is not a valid benchmark.
        for benchmark, series in (
            ("Buy & Hold (traded universe, 41)", traded_benchmark_return),
            ("Buy & Hold (large caps only, 29)", large_cap_benchmark_return),
            ("Buy & Hold (cached market series)", market),
        ):
            window_summaries[benchmark] = performance_summary(
                series.reindex(window), rf_daily, benchmark)
            window_turnovers[benchmark] = 0.0
            print(f"  {benchmark:<36} "
                  f"ann {window_summaries[benchmark]['Ann_Return']:>8.2%}   "
                  f"Sharpe {window_summaries[benchmark]['Sharpe']:>5.2f}   turnover    0.0%")

        summaries[window_name] = pd.DataFrame(window_summaries)
        turnovers[window_name] = window_turnovers
        results[window_name] = window_results

    # ---- Decomposition ladder and signal-value tests, on the holdout ----
    hold = results["holdout"]
    ladder_keys = list(books.keys())[:4]
    ann = {name: annualised_return(hold[name]["Net_Strategy_Return"]) for name in books}
    decomposition = pd.Series({
        "(a) original, switch + liquidation": ann[ladder_keys[0]],
        "(b) original sleeves, no liquidation": ann[ladder_keys[1]],
        "(c) original sleeves, continuous blend": ann[ladder_keys[2]],
        "(d) Beta Slider (new sleeves, blend)": ann[ladder_keys[3]],
        "a -> b: dropping the liquidation rule": ann[ladder_keys[1]] - ann[ladder_keys[0]],
        "b -> c: continuous blending, sleeves fixed": ann[ladder_keys[2]] - ann[ladder_keys[1]],
        "c -> d: changing the sleeves, mechanism fixed": ann[ladder_keys[3]] - ann[ladder_keys[2]],
        "total gap (d - a)": ann[ladder_keys[3]] - ann[ladder_keys[0]],
        "SIGNAL VALUE: Beta Slider minus static 50/50":
            ann["Final (Beta Slider)"] - ann["CONTROL: static 50/50 blend"],
        "SIGNAL VALUE: Beta Slider minus placebo":
            ann["Final (Beta Slider)"] - ann["CONTROL: placebo signal"],
        "regime switches (original)": int((gen1_state.diff().fillna(0) != 0).sum()),
    })
    print("\nDoes the regime signal earn its place? (holdout, annualised)")
    print("  Beta Slider minus static 50/50 blend: "
          f"{decomposition['SIGNAL VALUE: Beta Slider minus static 50/50']:+.2%}")
    print("  Beta Slider minus placebo signal:     "
          f"{decomposition['SIGNAL VALUE: Beta Slider minus placebo']:+.2%}")

    # ---- Measured claims ----
    overlap = sleeve_overlap(w_mom, w_def)
    betas = {
        "Min-vol sleeve": sleeve_beta(w_def, asset_returns,
                                      traded_benchmark_return, prob),
        "Momentum sleeve": sleeve_beta(w_mom, asset_returns,
                                       traded_benchmark_return, prob),
        "Beta Slider": sleeve_beta(books["Final (Beta Slider)"], asset_returns,
                                   traded_benchmark_return, prob),
    }
    print(f"\n  mean overlapping names: {overlap['n_overlap'].mean():.1f} of {cfg.top_n}")
    print("  min-vol sleeve full-sample beta: "
          f"{betas['Min-vol sleeve']['beta_full_sample']:.2f}")

    # ---- Cost sweep on the holdout ----
    cost_rows = {}
    for one_way in cfg.cost_grid:
        scaled = pd.Series(one_way, index=all_equities.columns, dtype=float)
        scaled.loc[small_only] = one_way * (cfg.tc_bps_small / max(cfg.tc_bps_large, 1e-9))
        cost_rows[one_way] = {}
        for name in ("Final (Beta Slider)", "CONTROL: static 50/50 blend",
                     "Min-vol sleeve only"):
            scheduled = apply_rebalance_frequency(
                books[name].reindex(hold_index), cfg.rebalance)
            cost_rows[one_way][name] = annualised_return(
                run_backtest(scheduled, asset_returns.reindex(hold_index), scaled,
                             cfg.charge_drift, cfg.no_trade_band)["Net_Strategy_Return"])
    cost_sweep = pd.DataFrame(cost_rows).T
    cost_sweep.index.name = "one_way_bps"

    # ---- Figures ----
    print("\nGenerating figures...")
    plot_paper_figures(summaries, cfg)

    # Exploratory equity curves over the full window, with the holdout boundary marked.
    # Both benchmarks come from the traded panels; see F3 in the module docstring.
    market_curve = (1.0 + traded_benchmark_return.fillna(0.0)).cumprod()
    large_cap_curve = (1.0 + large_cap_benchmark_return.fillna(0.0)).cumprod()
    for name, filename, strategy_label, shade_label, shade_axis in (
        ("Original (Discrete Switch)", "fig1_directional_switch.html",
         "Directional Switching Strategy",
         "HMM Prob High Vol (Alloc to Large Cap MR)", "High Vol Allocation %"),
        ("Final (Beta Slider)", "fig2_beta_slider.html",
         "All-Equity Beta Slider Strategy",
         "Allocation to Defensive Low-Vol (Large Caps)", "Defensive Allocation %"),
    ):
        curve = evaluate(books[name], index, cfg.rebalance,
                         cfg.no_trade_band)["Cum_Net_Strategy"]
        plot_strategy_vs_market(
            curve, market_curve, large_cap_curve, prob,
            f"OOS Performance: {strategy_label}", strategy_label,
            shade_label, shade_axis, split, cfg.fig_dir / filename)
        if name == "Final (Beta Slider)":
            plot_equity_curve_figure(curve, market_curve, large_cap_curve,
                                     prob, split, cfg)

    # ---- Reports ----
    print("\nWriting reports...")
    summaries["tuning"].to_csv(cfg.out_dir / "summary_tuning.csv")
    summaries["holdout"].to_csv(cfg.out_dir / "summary_holdout.csv")
    cost_sweep.to_csv(cfg.out_dir / "cost_sensitivity.csv")
    decomposition.to_csv(cfg.out_dir / "decomposition_ladder.csv")
    if not grid.empty:
        grid.to_csv(cfg.out_dir / "tuning_grid.csv", index=False)
    write_paper_table1(
        summaries["holdout"], turnovers["holdout"],
        columns=["Original (Discrete Switch)", "Final (Beta Slider)",
                 "CONTROL: static 50/50 blend", "Min-vol sleeve only",
                 "Buy & Hold (traded universe, 41)",
                 "Buy & Hold (large caps only, 29)"],
        window_label=f"{hold_index.min().date()} to {hold_index.max().date()}",
        path=cfg.out_dir / "paper_table1.tex")
    write_diagnostics({
        "CONFIGURATION": {k: getattr(cfg, k) for k in vars(cfg)},
        "RISK-FREE": ("real T-bill series" if rf_is_real
                      else "rf = 0; 'Sharpe' is a return/vol ratio"),
        "TUNING GRID (ranked, tuning window only)": grid,
        "SUMMARY - TUNING WINDOW": summaries["tuning"],
        "SUMMARY - HOLDOUT WINDOW (report this one)": summaries["holdout"],
        "AVG ANNUAL TURNOVER - HOLDOUT":
            {k: f"{v:.1%}" for k, v in turnovers["holdout"].items()},
        "DECOMPOSITION LADDER AND SIGNAL VALUE (holdout)": decomposition,
        "SLEEVE OVERLAP": {
            "mean overlapping names": round(float(overlap["n_overlap"].mean()), 2),
            "median": float(overlap["n_overlap"].median()),
            "max": int(overlap["n_overlap"].max()),
            "mean overlap fraction": f"{overlap['overlap_frac'].mean():.1%}",
        },
        "SLEEVE BETAS (against the traded universe)": pd.DataFrame(betas),
        "COST SENSITIVITY (holdout, net annualised)": cost_sweep,
    }, cfg.out_dir / "diagnostics.txt")

    print("\n--- HOLDOUT HEADLINE (this is what the paper should report) ---")
    print(summaries["holdout"].loc[
        ["Terminal_Wealth", "Ann_Return", "Ann_Volatility", "Sharpe", "Max_Drawdown"]
    ].to_string())
    print("\nRun complete.")
    return {"summaries": summaries, "turnovers": turnovers,
            "decomposition": decomposition, "grid": grid}


if __name__ == "__main__":
    main(Config())