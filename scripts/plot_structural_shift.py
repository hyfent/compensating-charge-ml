#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "auxiliary" / "structural_shift_distances.csv"
OUT = ROOT / "figures"


def main() -> None:
    frame = pd.read_csv(DATA)
    plt.rcParams.update({"font.family": "sans-serif", "font.size": 9, "pdf.fonttype": 42})
    rng = np.random.default_rng(20260720)
    for material in ("NaCl", "CsPbI3"):
        figure, axes = plt.subplots(2, 1, figsize=(6.2, 7.0), constrained_layout=True)
        for axis, task, letter in zip(axes, ("deformation", "temperature"), ("a", "b")):
            subset = frame[(frame["material"] == material) & (frame["task"] == task)]
            conditions = list(dict.fromkeys(subset["condition"]))
            colors = plt.get_cmap("cividis")(np.linspace(0.12, 0.88, len(conditions)))
            axis.axvspan(-0.5, 1.5, color="#EAF1F8", alpha=0.72)
            axis.axvspan(1.5, len(conditions) - 0.5, color="#FBF4DE", alpha=0.72)
            axis.axvline(1.5, color="#777777", lw=0.9, ls=(0, (3, 2)))
            for index, (condition, color) in enumerate(zip(conditions, colors)):
                rows = subset[subset["condition"] == condition]
                values = rows["distance"].to_numpy()
                role = rows["role"].iloc[0]
                marker = "o" if role == "low" else "s"
                axis.scatter(
                    index + rng.uniform(-0.16, 0.16, len(values)), values, s=16,
                    marker=marker, facecolor=color if role == "low" else "none",
                    edgecolor=color, linewidth=0.6, alpha=0.50,
                )
                q25, median, q75 = np.quantile(values, (0.25, 0.50, 0.75))
                axis.vlines(index, q25, q75, color="#202020", lw=2.6)
                axis.plot(index, median, marker="_", ms=12, mew=1.6, color="#202020")
            axis.set_yscale("log")
            axis.set_xticks(range(len(conditions)))
            axis.set_xticklabels(conditions)
            axis.set_ylabel("Descriptor-space distance")
            axis.text(-0.09, 1.02, f"({letter})", transform=axis.transAxes, fontweight="bold")
            axis.spines["top"].set_visible(False)
            axis.spines["right"].set_visible(False)
        OUT.mkdir(parents=True, exist_ok=True)
        stem = "nacl" if material == "NaCl" else "cspbi3"
        figure.savefig(OUT / f"{stem}_structural_shift.pdf", bbox_inches="tight")
        figure.savefig(OUT / f"{stem}_structural_shift.png", dpi=300, bbox_inches="tight")


if __name__ == "__main__":
    main()
