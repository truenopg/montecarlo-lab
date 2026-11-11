"""Correlated path simulation, three innovation models.

All three share the same estimated drift/vol/correlation from the aligned
daily log returns; they differ only in how the shocks are drawn:

- gaussian: classic correlated GBM (the assumption under test).
- student-t: multivariate t innovations (Gaussian mixture representation)
  with degrees of freedom fitted from the pooled excess kurtosis - keeps
  the fat tails the QQ plot shows, at the cost of assuming them symmetric.
- bootstrap: stationary-block resampling of the historical return pairs
  (mean block ~10 days), which keeps tails, asymmetry and the empirical
  cross-asset co-movement, but cannot generate paths outside the sample.

Drift is the historical mean return: this is the real-world (physical)
measure for risk questions, NOT a pricing measure. Option pricing in this
repo sets drift to the risk-free rate explicitly, as an assumption, in
mc/options.py.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
rng = np.random.default_rng(7)


def load_pair() -> pd.DataFrame:
    def closes(name):
        d = pd.read_csv(ROOT / "data" / name, parse_dates=["ts"])
        # yahoo sometimes emits two bars for one date; keep the later one
        return d.groupby(d["ts"].dt.date)["close"].last()
    px = pd.DataFrame({"btc": closes("btc_1d.csv"), "gold": closes("xauusd_1d.csv")}).dropna()
    return px


def t_df_from_kurtosis(r: pd.Series) -> float:
    k = stats.kurtosis(r)  # excess; t has 6/(df-4)
    return float(np.clip(4 + 6 / max(k, 1e-9), 4.1, 100.0))


def simulate(rets: pd.DataFrame, horizon: int, n_paths: int, method: str) -> np.ndarray:
    """Returns (n_paths, horizon, 2) log-return paths."""
    mu = rets.mean().to_numpy()
    cov = rets.cov().to_numpy()
    L = np.linalg.cholesky(cov)

    if method == "gaussian":
        z = rng.standard_normal((n_paths, horizon, 2))
        shocks = z @ L.T
    elif method == "student-t":
        df = t_df_from_kurtosis(rets["btc"])  # BTC dominates the tails
        z = rng.standard_normal((n_paths, horizon, 2))
        w = np.sqrt(df / rng.chisquare(df, (n_paths, horizon, 1)))
        shocks = (z * w) @ L.T * np.sqrt((df - 2) / df)  # keep var = cov
    elif method == "bootstrap":
        r = (rets - mu).to_numpy()
        n = len(r)
        starts = rng.integers(0, n, (n_paths, horizon))
        keep = rng.random((n_paths, horizon)) > 1 / 10  # geometric blocks, mean 10
        idx = np.empty((n_paths, horizon), dtype=int)
        idx[:, 0] = starts[:, 0]
        for t in range(1, horizon):
            idx[:, t] = np.where(keep[:, t], (idx[:, t - 1] + 1) % n, starts[:, t])
        shocks = r[idx]
    else:
        raise ValueError(method)

    return mu + shocks


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=21)
    ap.add_argument("--paths", type=int, default=50000)
    args = ap.parse_args()

    px = load_pair()
    rets = np.log(px).diff().dropna()
    s0 = px.iloc[-1].to_numpy()
    print(f"{len(rets)} aligned days, last close BTC {s0[0]:.0f} / gold {s0[1]:.0f}")
    print(f"fitted t degrees of freedom (BTC): {t_df_from_kurtosis(rets['btc']):.1f}")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    term = {}
    for method in ("gaussian", "student-t", "bootstrap"):
        paths = simulate(rets, args.horizon, args.paths, method)
        term[method] = paths.sum(axis=1)  # cumulative log return over horizon
        for j, name in enumerate(("btc", "gold")):
            ax = axes[j]
            ax.hist(term[method][:, j], bins=200, density=True, histtype="step",
                    label=method)
    for j, name in enumerate(("btc", "gold")):
        axes[j].set_title(f"{name}: {args.horizon}-day log return, {args.paths} paths")
        axes[j].legend()
    fig.tight_layout()
    fig.savefig(FIG / "terminal_distributions.png", dpi=110)

    # fan chart for BTC under each model: percentile lines over steps
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"gaussian": "tab:blue", "student-t": "tab:orange", "bootstrap": "tab:green"}
    for method, c in colors.items():
        paths = simulate(rets, args.horizon, min(args.paths, 20000), method)
        cum = np.cumsum(paths[:, :, 0], axis=1)
        lo, med, hi = np.percentile(cum, [2.5, 50, 97.5], axis=0)
        x = np.arange(1, args.horizon + 1)
        ax.plot(x, s0[0] * np.exp(med), color=c, lw=1.2, label=method)
        ax.plot(x, s0[0] * np.exp(lo), color=c, lw=0.9, ls="--")
        ax.plot(x, s0[0] * np.exp(hi), color=c, lw=0.9, ls="--")
    ax.set_title(f"BTC {args.horizon}-day simulation: median (solid) + 95% band (dashed)")
    ax.set_ylabel("BTC USD")
    ax.set_xlabel("days ahead")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "path_fan_btc.png", dpi=110)

    print("\nterminal 21-day return quantiles (BTC | gold):")
    for method, t in term.items():
        q = np.percentile(t, [1, 5, 50, 95, 99], axis=0)
        print(f"  {method:10s} 1%: {q[0, 0]:+.1%} | {q[0, 1]:+.1%}   "
              f"5%: {q[1, 0]:+.1%} | {q[1, 1]:+.1%}   median: {q[2, 0]:+.1%} | {q[2, 1]:+.1%}")
    print(f"\nfigures -> {FIG}")


if __name__ == "__main__":
    main()
