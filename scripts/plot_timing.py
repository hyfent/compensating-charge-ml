#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "auxiliary" / "timing_raw.csv"
OUT = ROOT / "figures"
LABELS = {"Baseline": "Conventional ML", "IC model": "CC-ML", "Ewald-QEq": "Ewald-QEq"}
COLORS = {"Baseline": "#0072B2", "IC model": "#D55E00", "Ewald-QEq": "#4D4D4D"}
MARKERS = {"Baseline": "o", "IC model": "s", "Ewald-QEq": "^"}


def main() -> None:
    raw = pd.read_csv(DATA)
    means = raw.groupby(["method", "atoms"], as_index=False)["seconds"].mean()
    figure, axes = plt.subplots(2, 1, figsize=(6.2, 6.8), constrained_layout=True)
    for method in ("Baseline", "IC model", "Ewald-QEq"):
        rows = means[means["method"] == method].sort_values("atoms")
        axes[0].plot(rows["atoms"], rows["seconds"], marker=MARKERS[method], color=COLORS[method], label=LABELS[method])
    for method in ("Baseline", "IC model"):
        rows = means[means["method"] == method].sort_values("atoms")
        axes[1].plot(rows["atoms"], 1000.0 * rows["seconds"], marker=MARKERS[method], color=COLORS[method], label=LABELS[method])
    axes[0].set(xlabel="Number of atoms", ylabel="Wall time per structure (s)")
    axes[1].set(xlabel="Number of atoms", ylabel="Wall time per structure (ms)")
    for letter, axis in zip(("a", "b"), axes):
        axis.text(-0.10, 1.02, f"({letter})", transform=axis.transAxes, fontweight="bold")
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "computational_efficiency.pdf", bbox_inches="tight")
    figure.savefig(OUT / "computational_efficiency.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
