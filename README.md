# montecarlo-lab

Monte Carlo lab on BTC and gold daily data: how much do the conclusions
move when you stop assuming Gaussian returns?

Two separate questions, kept separate on purpose:

- **Risk (real-world measure)**: simulate correlated paths with the
  historical drift, then ask about losses - VaR/ES with confidence
  intervals. The drift here is the historical mean, not a pricing input.
- **Option pricing (risk-neutral measure)**: Monte Carlo a European
  option against the Black-Scholes closed form, with the drift set to
  the risk-free rate and declared as an assumption. Never mixed with the
  historical drift from the risk side.

## Data

- BTC-USD spot daily, Coinbase Exchange candles (public, no key).
- Gold daily, Yahoo `GC=F` front-month futures as the XAUUSD proxy (the
  basis vs spot is noise at the daily scale - documented so nobody
  mistakes it for spot ticks).
- `data/fetch.py` extends the CSVs incrementally.

## Checks before simulating anything

`mc/diagnostics.py` runs the assumption checks first: on ~2,850 aligned
days, both assets have fat tails (excess kurtosis ~11 BTC / ~8 gold,
Jarque-Bera rejects normality at any level), returns are basically
uncorrelated with their own past, and squared returns cluster heavily
(volatility regimes). BTC-gold daily return correlation is ~0.02 - the
"correlated" simulation mostly has to get the tails right, the linear
co-movement is near zero.

## Simulation

`mc/simulate.py` compares three innovation models on the same estimated
drift/vol/correlation:

- `gaussian`: correlated GBM, the assumption under test.
- `student-t`: multivariate-t shocks with df fitted from the kurtosis
  (~4.5 for BTC), keeps fat tails but assumes them symmetric.
- `bootstrap`: stationary block resampling of historical return pairs -
  keeps tails, asymmetry and empirical co-movement, cannot leave the
  sample.

At a 21-day horizon the three terminal distributions sit almost on top of
each other (averaging washes the daily tails out); the differences show
in the extreme quantiles, where the bootstrap 1% is the fattest. That is
exactly what the VaR/ES comparison quantifies next.

## Risk: VaR/ES with confidence intervals

`mc/risk.py` prices the tail question: 95%/99% VaR and ES for BTC and a
50/50 daily-rebalanced BTC-gold portfolio at 1 and 21 days, per model,
each with a bootstrap CI on the MC estimate (that CI is simulation noise
only; model risk is the difference between columns). The 1-day story:
gaussian ES99 5.5% vs bootstrap 8.2% - assuming normality undercounts
the daily tail by a third. At 21 days the gap compresses (12.2-12.6%
VaR95 everywhere) because averaging washes the tails out. Full table in
`report/risk.json`.

## Options: MC vs Black-Scholes + variance reduction

`mc/options.py` prices an ATM 30-day BTC call (risk-neutral drift = the
`--rate` assumption, default 4%, never the historical drift) against the
Black-Scholes closed form the same assumptions give, then measures how
much antithetic sampling and a control variate on the discounted
terminal price cut the MC error. On this contract antithetic alone
changes nothing; the control variate cuts the standard error ~2.3x
(~5x less variance) - the convergence plot shows it cleanly.

## Roadmap

Done: data, diagnostics, correlated simulation, VaR/ES with CIs,
European option MC vs BS with variance reduction and convergence plots.
Maybe later: path-dependent payoffs (Asian/barrier) where no closed
form exists, multivariate t-copula calibration, GARCH-filtered bootstrap
for the vol clustering the diagnostics show.
