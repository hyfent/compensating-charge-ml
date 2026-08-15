#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "auxiliary" / "madelung_convergence.csv"
OUT = ROOT / "figures"


def main() -> None:
    frame = pd.read_csv(DATA)
    radius = frame["cutoff_radius_over_r0"].to_numpy()
    direct = frame["direct_madelung_sum"].to_numpy()
    neutralized = frame["neutralized_madelung_sum"].to_numpy()
    reference = float(frame["reference"].iloc[0])
    delta = frame["delta_q"].to_numpy()
    near_neutral = np.isclose(np.abs(delta), np.min(np.abs(delta)))

    plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "pdf.fonttype": 42})
    figure, axes = plt.subplots(2, 1, figsize=(6.2, 6.8), constrained_layout=True)
    axes[0].plot(radius, direct, color="#D55E00", lw=0.9)
    axes[0].scatter(radius[near_neutral], direct[near_neutral], s=14, color="black", zorder=3)
    axes[0].axhline(reference, color="black", ls="--", lw=1.0)
    axes[0].set(xlabel=r"Cutoff radius $R_c/r_0$", ylabel="Madelung constant $M$")
    axes[0].text(-0.11, 1.02, "(a)", transform=axes[0].transAxes, fontweight="bold")

    axes[1].plot(radius, neutralized, color="#0072B2", lw=1.0)
    axes[1].axhline(reference, color="black", ls="--", lw=1.0)
    axes[1].set(xlabel=r"Cutoff radius $R_c/r_0$", ylabel="Madelung constant $M$")
    axes[1].text(-0.11, 1.02, "(b)", transform=axes[1].transAxes, fontweight="bold")
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)

    OUT.mkdir(parents=True, exist_ok=True)
    figure.savefig(OUT / "madelung_convergence.pdf", bbox_inches="tight")
    figure.savefig(OUT / "madelung_convergence.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
