"""European call: Monte Carlo vs Black-Scholes, plus variance reduction.

RISK-NEUTRAL pricing, kept apart from the historical-drift risk sims:
the drift is the risk-free rate passed on the command line, declared as
an assumption (--rate, default 4%). BTC has no yield. Vol is estimated
from the same daily history. Nothing here says the GBM is a good price
model - the point is to measure MC error against the closed form the
same assumptions give, and to show how much antithetic sampling and a
control variate cut that error.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
FIG = ROOT / "report" / "figures"
rng = np.random.default_rng(13)


def bs_call(S: float, K: float, T: float, r: float, sigma: float) -> float:
    d1 = (np.log(S / K) + (r + sigma ** 2 / 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * stats.norm.cdf(d1) - K * np.exp(-r * T) * stats.norm.cdf(d2)


def mc_call(S: float, K: float, T: float, r: float, sigma: float, n: int,
            antithetic: bool, control: bool):
    z = rng.standard_normal(n // 2 if antithetic else n)
    if antithetic:
        z = np.concatenate([z, -z])
    st = S * np.exp((r - sigma ** 2 / 2) * T + sigma * np.sqrt(T) * z)
    payoff = np.exp(-r * T) * np.maximum(st - K, 0)
    if control:
        disc_s = np.exp(-r * T) * st  # known expectation S under RN measure
        beta = np.cov(payoff, disc_s)[0, 1] / disc_s.var()
        payoff = payoff - beta * (disc_s - S)
    return payoff.mean(), payoff.std(ddof=1) / np.sqrt(len(payoff)), len(payoff)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rate", type=float, default=0.04,
                    help="risk-free rate assumption, NOT estimated (default 0.04)")
    ap.add_argument("--days", type=int, default=30)
    args = ap.parse_args()

    import sys
    sys.path.insert(0, str(ROOT / "mc"))
    from simulate import load_pair
    px = load_pair()
    rets = np.log(px["btc"]).diff().dropna()
    S = float(px["btc"].iloc[-1])
    K = round(S, -3)
    T = args.days / 365
    sigma = float(rets.std() * np.sqrt(365))
    price = bs_call(S, K, T, args.rate, sigma)
    print(f"BTC spot {S:.0f}, ATM strike {K}, T {args.days}d, sigma {sigma:.1%}, "
          f"r {args.rate:.1%} (assumption)")
    print(f"Black-Scholes: {price:.2f}\n")

    for n in (10_000, 200_000):
        for name, anti, ctrl in (("plain", False, False),
                                 ("antithetic", True, False),
                                 ("antithetic+control", True, True)):
            est, se, m = mc_call(S, K, T, args.rate, sigma, n, anti, ctrl)
            print(f"  N={m:>7,} {name:19s} {est:8.2f} +- {se:5.2f}  (err {est - price:+.2f})")
        print()

    ns = np.unique(np.logspace(2, 5.3, 25).astype(int))
    curves = {}
    for name, anti, ctrl in (("plain", False, False),
                             ("antithetic", True, False),
                             ("antithetic+control", True, True)):
        ses = [mc_call(S, K, T, args.rate, sigma, n, anti, ctrl)[1] for n in ns]
        curves[name] = ses
    fig, ax = plt.subplots(figsize=(8, 4))
    for (name, ses), c in zip(curves.items(), ("tab:blue", "tab:orange", "tab:green")):
        ax.plot(ns, ses, marker=".", ms=4, label=name, color=c)
    ax.plot(ns, curves["plain"][0] * np.sqrt(ns[0] / ns), color="grey", ls="--",
            lw=0.8, label="1/sqrt(N)")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_title("MC standard error vs paths, ATM BTC call")
    ax.set_xlabel("paths")
    ax.set_ylabel("stderr of price estimate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "option_convergence.png", dpi=110)

    plain_se = curves["plain"][-1]
    ctrl_se = curves["antithetic+control"][-1]
    print(f"variance reduction at {ns[-1]:,} paths: stderr {plain_se:.3f} -> {ctrl_se:.4f} "
          f"({(plain_se / ctrl_se) ** 2:.0f}x less variance)")
    print(f"-> {FIG / 'option_convergence.png'}")


if __name__ == "__main__":
    main()
