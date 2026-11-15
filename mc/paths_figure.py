"""The classic Monte Carlo picture: individual simulated price paths.

One panel per asset (BTC, gold) and innovation model, ~250 faint paths
each, same 21-day horizon and starting prices as the risk sims. Paths
share the historical drift - these are possible futures under each shock
model, not a forecast of where the price will go.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from simulate import load_pair, simulate, FIG


def main() -> None:
    px = load_pair()
    rets = np.log(px).diff().dropna()
    s0 = px.iloc[-1].to_numpy()
    last_date = px.index[-1]
    horizon, n_show = 21, 250

    fig, axes = plt.subplots(2, 3, figsize=(12, 6), sharex=True, sharey="row")
    for col, method in enumerate(("gaussian", "student-t", "bootstrap")):
        paths = simulate(rets, horizon, n_show, method)
        for row, (name, s) in enumerate((("BTC", s0[0]), ("gold", s0[1]))):
            prices = s * np.exp(np.cumsum(paths[:, :, row], axis=1))
            prices = np.column_stack([np.full(n_show, s), prices])
            ax = axes[row, col]
            ax.plot(prices.T, lw=0.4, alpha=0.35, color="tab:blue" if row == 0 else "tab:olive")
            ax.axhline(s, color="black", lw=0.6, ls=":")
            if row == 0:
                ax.set_title(method)
            if col == 0:
                ax.set_ylabel(f"{name} (USD)")
            if row == 1:
                ax.set_xlabel("days ahead")
    fig.suptitle(f"{n_show} simulated paths per model, from the last aligned close "
                 f"({last_date}: BTC {s0[0]:,.0f} USD / gold {s0[1]:,.0f} USD)")
    fig.tight_layout()
    fig.savefig(FIG / "simulated_paths.png", dpi=110)
    print(f"-> {FIG / 'simulated_paths.png'}")


if __name__ == "__main__":
    main()
