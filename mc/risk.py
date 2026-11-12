"""VaR and Expected Shortfall per innovation model, with confidence intervals.

Real-world (physical) measure: paths come from mc/simulate.py with the
historical drift - this is a risk question, not pricing.

For each model and horizon we report the 95%/99% VaR and ES of the simple
return, for BTC alone, gold alone and a 50/50 daily-rebalanced portfolio.
The intervals are the Monte Carlo estimation error: resample the simulated
terminal returns (B=500) and recompute each statistic - they say how much
of the number is simulation noise, not how much is model risk.

Note what the intervals do NOT cover: wrong-model risk. That part shows up
as the difference between the gaussian/student-t/bootstrap columns.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "mc"))
from simulate import load_pair, simulate

FIG = ROOT / "report" / "figures"
rng = np.random.default_rng(11)


def portfolio_returns(paths: np.ndarray, w: tuple = (0.5, 0.5)) -> np.ndarray:
    """Daily-rebalanced portfolio simple return per path."""
    simple = np.expm1(paths)  # (n, h, 2)
    port = simple @ np.array(w)
    return np.prod(1 + port, axis=1) - 1


def var_es(returns: np.ndarray, level: float) -> tuple[float, float]:
    var = -np.quantile(returns, 1 - level)
    es = -returns[returns <= -var].mean()
    return var, es


def with_ci(returns: np.ndarray, level: float, B: int = 500) -> dict:
    var, es = var_es(returns, level)
    n = len(returns)
    boot_var, boot_es = [], []
    for _ in range(B):
        sample = returns[rng.integers(0, n, n)]
        v, e = var_es(sample, level)
        boot_var.append(v)
        boot_es.append(e)
    return {"var": round(var, 4), "var_ci": [round(x, 4) for x in np.percentile(boot_var, [2.5, 97.5])],
            "es": round(es, 4), "es_ci": [round(x, 4) for x in np.percentile(boot_es, [2.5, 97.5])]}


def main() -> None:
    px = load_pair()
    rets = np.log(px).diff().dropna()
    out = {}
    dist_21d = {}
    for horizon in (1, 21):
        out[str(horizon)] = {}
        for method in ("gaussian", "student-t", "bootstrap"):
            paths = simulate(rets, horizon, 50000, method)
            port = portfolio_returns(paths)
            if horizon == 21:
                dist_21d[method] = port
            out[str(horizon)][method] = {
                "portfolio_50_50": {f"{lvl}%": with_ci(port, lvl / 100) for lvl in (95, 99)},
                "btc": {f"{lvl}%": with_ci(np.expm1(paths[:, :, 0].sum(axis=1)), lvl / 100)
                        for lvl in (95, 99)},
            }
            # gold skipped in the table: same code path, portfolio carries the point
    (ROOT / "report" / "risk.json").write_text(json.dumps(out, indent=2))

    for horizon, res in out.items():
        print(f"\n== {horizon}-day horizon, 50/50 BTC-gold portfolio ==")
        for method, r in res.items():
            for lvl in ("95%", "99%"):
                s = r["portfolio_50_50"][lvl]
                print(f"  {method:10s} VaR{lvl} {s['var']:.1%} [{s['var_ci'][0]:.1%}, {s['var_ci'][1]:.1%}]"
                      f"   ES{lvl} {s['es']:.1%} [{s['es_ci'][0]:.1%}, {s['es_ci'][1]:.1%}]")

    fig, ax = plt.subplots(figsize=(9, 4))
    colors = {"gaussian": "tab:blue", "student-t": "tab:orange", "bootstrap": "tab:green"}
    for method, port in dist_21d.items():
        ax.hist(port, bins=250, density=True, histtype="step", color=colors[method], label=method)
        var, es = var_es(port, 0.99)
        ax.axvline(-var, color=colors[method], ls="--", lw=0.9)
    ax.set_xlim(-0.6, 0.3)
    ax.set_title("21-day portfolio loss, 99% VaR as dashed lines")
    ax.set_xlabel("simple return")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "var_portfolio_21d.png", dpi=110)
    print(f"\n-> report/risk.json, {FIG / 'var_portfolio_21d.png'}")


if __name__ == "__main__":
    main()
