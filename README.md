# Exploration of Volatility Regime Based Trading Strategies

Code accompanying *An Inquiry Into Volatility Regime Based Trading Strategies* (Peisz, Saugy,
Turin-Plessia, 2026).

We estimate a two-state Gaussian hidden Markov model of equity volatility regimes and test
whether conditioning portfolio construction on its output improves performance. Two
architectures are built on the same signal: a discrete switch between a trend sleeve and a
short-horizon reversal sleeve, and a continuous slider that mixes a cross-sectional momentum
sleeve with a minimum-volatility sleeve in proportion to the regime probability.

**Headline result.** The continuous architecture returns 16.74% annualised on the 2017--2025
holdout against −15.33% for the discrete one, and turns over roughly a third as much. It does
not, however, separate from a static 50/50 blend of its own sleeves (16.98%) or from a placebo
signal with its timing destroyed (16.93%), and no variant beats a passive equal weight of the
same securities (18.48% for the traded universe, 21.65% for the large-cap panel). The regime
signal does not measurably contribute.

---

## Repository layout

| File | Purpose |
| --- | --- |
| `HMM.ipynb` | Builds the feature table, estimates the regime model, writes the one-step-ahead forecasts every strategy consumes |
| `all_equity_beta_slider.py` | Main experiment. Both architectures, tuning and holdout evaluation, all controls, every table and figure in the paper |
| `regime_hmm.py` | HMM feature construction and estimation helpers |
| `regime_strategy.py` | Cross-sectional sleeve construction and the generation-one backtest |
| `regime_rules.py` | Regime path rules: probability thresholds, hysteresis, minimum duration, run statistics |
| `strategy_primitives.py` | Single-asset momentum and mean-reversion rules |
| `project_dataset.py` | Loads price panels, benchmarks and regime inputs into a single object |
| `dataset_contracts.py` | Dataset schema and validation |
| `dataset_maker.py` | Downloads the price panels from source |
| `data_structurer.py` | Splits wide price panels into per-ticker files |
| `momentum.py`, `mean_rev.py` | Standalone single-asset demos, not used by the paper |

---

## Getting started

```bash
pip install -r requirements.txt
```

Then, in order:

**1. Build the price panels.** `data/` is not included in this repository (see *Data* below).

```bash
python dataset_maker.py      # downloads per-ticker files into data/large_caps and data/small_caps
```

**2. Estimate the regime model.** Run `HMM.ipynb` end to end. It writes
`data/portfolio_hmm_oos_2010_forecasts.csv`, which contains the
`Next_Day_Forecast_Prob_High_Vol` column the strategies trade on. The model is fitted once on
1993-02-01 to 2009-12-31 and then run purely as a filter; it is never re-estimated.

**3. Run the experiment.**

```bash
python all_equity_beta_slider.py
```

Takes a couple of minutes. Everything the paper reports is regenerated from this single run.

---

## Outputs

```
figures/fig_holdout_ranking.png    paper figure: holdout ranking, all variants and controls
figures/fig_equity_curve.png       paper figure: growth of $1 against both passive benchmarks
figures/fig_degradation.png        paper figure: tuning Sharpe against holdout Sharpe
figures/fig1_directional_switch.html   exploratory equity curve, architecture one
figures/fig2_beta_slider.html          exploratory equity curve, architecture two
output/summary_holdout.csv         all strategies, holdout window -- this is what the paper reports
output/summary_tuning.csv          all strategies, tuning window
output/paper_table1.tex            LaTeX source for the main results table
output/tuning_grid.csv             every parameter combination tried, ranked
output/decomposition_ladder.csv    mechanism effect vs sleeve effect vs signal value
output/cost_sensitivity.csv        net return across a one-way cost grid
output/diagnostics.txt             sleeve overlap, sleeve betas, configuration, all of the above
```

The three PNGs are the figures the paper includes. The HTML files are exploratory.

---

## Evaluation protocol

**Pre-committed split.** 2010-01-04 to 2016-12-30 is the tuning window; **2017-01-03 to
2025-12-31 is a holdout** that is not used for parameter selection. Both windows are reported
so that in-sample to out-of-sample degradation is visible. Only the holdout is a result.

**A small, declared grid.** Four friction parameters are fitted on the tuning window:
rebalancing frequency, a per-security no-trade band, an EWMA half-life applied to the regime
probability, and a tempering coefficient λ where `pi_eff = 0.5 + lambda * (pi - 0.5)`. Sleeve
parameters are fixed from the literature and never tuned. λ is worth watching: at 0 the
strategy is a static blend with no regime information, at 1 it uses the raw signal.

**Three controls.** A static 50/50 blend isolates the value of the signal from the value of
holding the sleeves. A placebo, formed by circularly shifting the probability series, destroys
the timing while preserving its distribution. Size-matched passive benchmarks prevent a size
tilt from being read as strategy performance.

**Frictions charged against drifted weights**, so the trades needed to restore a target book
are paid for rather than assumed away.

If you extend this work, please do not tune against the holdout numbers. Every pass converts
an out-of-sample result into an in-sample one, and a reader cannot see how many passes there
were.

---

## Configuration

Everything is in the `Config` dataclass at the top of `all_equity_beta_slider.py`. The
parameters most worth changing:

| Parameter | Default | Notes |
| --- | --- | --- |
| `tune_end` | `2016-12-31` | Everything after this is holdout |
| `top_n` | `20` | Securities per sleeve. Warns if this is a large share of the universe |
| `mom_lookback` | `126` | Momentum formation window, trading days |
| `vol_window` | `63` | Realised volatility window, trading days |
| `run_tuning` | `True` | Set `False` to use the hard-coded parameters instead of refitting |
| `tc_bps_large` / `tc_bps_small` | `15` / `25` | One-way costs in basis points |
| `gen1_exposure_convention` | `"active"` | `"universe"` reproduces a historical sizing bug; see below |

---

## Known issues and caveats

**The universe is small and survivorship-selected.** 29 large caps and 12 small caps, both
lists filtered on market capitalisation as retrieved at construction time. Securities are in
the large-cap panel because they are large caps *now*, which was not knowable at the start of
the sample. Return levels are inflated for every series, active and passive alike, so only the
*ordering* of results is informative. Selecting 20 of 41 is also near-total inclusion rather
than selection, which leaves cross-sectional ranking very little to do.

**`portfolio_returns_equal_weight.csv` holds LOG returns.** Its columns are cross-sectional
means of log returns, not portfolio returns. This is a good input to a volatility filter and
the wrong input to a compounding benchmark: compounded as simple returns it understates the
same 41 securities by roughly 7.9 percentage points a year. Use it for regime estimation only
and build benchmarks from the price panels. `project_dataset.infer_market_return_series`
carries a docstring explaining why `np.expm1()` is not a fix either.

**Volume is a placeholder.** `data_structurer.py` writes a constant `Volume = 0.0` into the
per-ticker files, so no liquidity or capacity analysis is possible from them. The SPY volume
feature used by the HMM comes from a separate, genuine source and is unaffected.

**The small-cap constituent source is unreliable.** `dataset_maker.get_russell2000_tickers`
scrapes a Wikipedia page that does not publish constituents. It now raises rather than
silently returning a handful of tickers, but a real holdings file should be supplied before
any conclusion is generalised.

**`gen1_exposure_convention`.** `run_strategy_for_universe` historically divided positions by
the number of securities with *data* rather than the number holding a *position*, which left
selective sleeves nearly uninvested. `"active"` is the corrected behaviour and the default
here; `"universe"` reproduces the original for comparison.

**`strategy_implementation.ipynb` does not currently run.** It imports
`run_train_test_optimized_vix_strategy` from a `regime_strategy_experiments` module that is not
in the repository. That notebook is a side experiment and is not used by the paper.

---

## Data

The price panels are not redistributed here. `dataset_maker.py` rebuilds them from source;
expect the download to take a while and some tickers to fail, which the script reports.

The risk-free series is daily returns on a 1--3 month Treasury bill ETF, read from
`data/BIL_cc.csv`. Without it the script falls back to a zero risk-free rate and relabels the
metric accordingly, since a return-to-volatility ratio is not a Sharpe ratio.

---

## Citation

```
Peisz, B., Saugy, G., and Turin-Plessia, A. (2026).
An Inquiry Into Volatility Regime Based Trading Strategies.
```
