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

## Roadmap

VaR/ES with confidence intervals per model, European option MC vs
Black-Scholes, variance reduction (antithetic, control variates),
convergence plots.
