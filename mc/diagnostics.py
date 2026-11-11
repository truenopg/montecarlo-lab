"""Do the two assets actually look like the models assume?

Runs the checks before any simulation claims anything: fat tails vs normal,
autocorrelation of returns and squared returns, and the BTC-gold return
correlation (which the correlated simulator must reproduce). Prints stats,
writes QQ plots to report/figures/.
"""
from __future__ import annotations

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


def load_pair() -> pd.DataFrame:
    def closes(name):
        d = pd.read_csv(ROOT / "data" / name, parse_dates=["ts"])
        # yahoo sometimes emits two bars for one date; keep the later one
        return d.groupby(d["ts"].dt.date)["close"].last()
    px = pd.DataFrame({"btc": closes("btc_1d.csv"), "gold": closes("xauusd_1d.csv")}).dropna()
    return np.log(px).diff().dropna()


def acf(x: np.ndarray, lags: int) -> np.ndarray:
    x = x - x.mean()
    return np.array([np.corrcoef(x[:-k], x[k:])[0, 1] if k else 1.0 for k in range(lags + 1)])


def main() -> None:
    rets = load_pair()
    print(f"aligned daily log returns: {len(rets)} days, "
          f"{rets.index.min()} -> {rets.index.max()}")
    print(f"return correlation BTC-gold: {rets.corr().iloc[0, 1]:.3f}")

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, r) in zip(axes, rets.items()):
        mu, sd = r.mean(), r.std()
        jb = stats.jarque_bera(r)
        lb_r = stats.chi2.sf(acf_ljung(r.values, 10), 10)
        lb_r2 = stats.chi2.sf(acf_ljung(r.values ** 2, 10), 10)
        print(f"\n{name}: ann vol {sd * np.sqrt(365 if name == 'btc' else 252):.1%}, "
              f"skew {stats.skew(r):.2f}, excess kurtosis {stats.kurtosis(r):.2f}")
        print(f"  Jarque-Bera p={jb.pvalue:.2e} (normal rejected if tiny)")
        print(f"  Ljung-Box(10) returns p={lb_r:.3f}, squared returns p={lb_r2:.2e}")
        stats.probplot(r, dist="norm", plot=ax)
        ax.set_title(f"{name} log returns vs normal")
    fig.tight_layout()
    fig.savefig(FIG / "qq_returns.png", dpi=110)

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for col, (name, r) in enumerate(rets.items()):
        axes[0, col].plot(acf(r.values, 30), marker=".", ms=4)
        axes[0, col].set_title(f"{name}: acf of returns")
        axes[1, col].plot(acf((r - r.mean()) ** 2, 30), marker=".", ms=4)
        axes[1, col].set_title(f"{name}: acf of squared returns (vol clustering)")
        for a in (axes[0, col], axes[1, col]):
            a.axhline(2 / np.sqrt(len(r)), color="grey", ls="--", lw=0.8)
            a.axhline(-2 / np.sqrt(len(r)), color="grey", ls="--", lw=0.8)
    fig.tight_layout()
    fig.savefig(FIG / "autocorr.png", dpi=110)
    print(f"\nfigures -> {FIG}")


def acf_ljung(x: np.ndarray, lags: int) -> float:
    n = len(x)
    a = acf(x, lags)[1:]
    return n * (n + 2) * np.sum(a ** 2 / (n - np.arange(1, lags + 1)))


if __name__ == "__main__":
    main()
