#!/usr/bin/env python3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, TwoSlopeNorm
import numpy as np
import pandas as pd

from analysis import DATA, path_average, summarize


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
NETWORKS = ("small", "medium", "large")
COLORS = {"baseline": "#0072B2", "ic": "#D55E00"}
LABELS = {"baseline": "Conventional ML", "ic": "CC-ML"}
MARKERS = {"baseline": "o", "ic": "s"}


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 10,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
        }
    )


def draw_curve(axis, subset: pd.DataFrame) -> None:
    for mode in ("baseline", "ic"):
        grouped = subset[subset["mode"] == mode].groupby("train_size")["mae_atom"].agg(["mean", "std"])
        axis.errorbar(
            grouped.index,
            grouped["mean"],
            yerr=grouped["std"],
            color=COLORS[mode],
            marker=MARKERS[mode],
            lw=1.4,
            ms=4,
            capsize=2,
            label=LABELS[mode],
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)


def plot_test_curves(material: str) -> None:
    raw = pd.read_csv(DATA / f"{material}_test_metrics.csv")
    epochs = sorted(raw["total_epochs"].unique())
    figure, axes = plt.subplots(len(epochs), 3, figsize=(8.0, 7.1), sharex=True, constrained_layout=True)
    for row, epoch in enumerate(epochs):
        for column, network in enumerate(NETWORKS):
            axis = axes[row, column]
            draw_curve(axis, raw[(raw["network"] == network) & (raw["total_epochs"] == epoch)])
            if row == 0:
                axis.set_title(network.capitalize())
            if column == 0:
                axis.set_ylabel(f"{epoch} epochs\nMAE (e)")
            if row == len(epochs) - 1:
                axis.set_xlabel("Training structures")
    axes[0, 0].legend(frameon=False)
    figure.savefig(OUT / f"{material}_test_mae.pdf", bbox_inches="tight")
    figure.savefig(OUT / f"{material}_test_mae.png", dpi=300, bbox_inches="tight")


def plot_low_to_high_curves(material: str) -> None:
    raw = path_average(pd.read_csv(DATA / f"{material}_low_to_high_metrics.csv"))
    tasks = ("deformation", "temperature")
    figure, axes = plt.subplots(2, 3, figsize=(8.0, 5.0), constrained_layout=True)
    for row, task in enumerate(tasks):
        for column, network in enumerate(NETWORKS):
            axis = axes[row, column]
            subset = raw[
                (raw["task"] == task)
                & (raw["network"] == network)
                & (raw["total_epochs"] == 600)
            ]
            draw_curve(axis, subset)
            if row == 0:
                axis.set_title(network.capitalize())
            if column == 0:
                axis.set_ylabel(f"{task.capitalize()}\nMAE (e)")
            if row == 1:
                axis.set_xlabel("Training structures")
    axes[0, 0].legend(frameon=False)
    figure.savefig(OUT / f"{material}_low_to_high_mae.pdf", bbox_inches="tight")
    figure.savefig(OUT / f"{material}_low_to_high_mae.png", dpi=300, bbox_inches="tight")


def matrix_values(frame: pd.DataFrame, value: str):
    epochs = sorted(frame["total_epochs"].unique())
    sizes = sorted(frame["train_size"].unique())
    rows = [(network, epoch) for network in NETWORKS for epoch in epochs]
    matrix = np.full((len(rows), len(sizes)), np.nan)
    for i, (network, epoch) in enumerate(rows):
        for j, size in enumerate(sizes):
            selected = frame[
                (frame["network"] == network)
                & (frame["total_epochs"] == epoch)
                & (frame["train_size"] == size)
            ]
            if len(selected):
                matrix[i, j] = float(selected[value].iloc[0])
    return matrix, sizes, rows


def draw_matrix(axis, frame: pd.DataFrame, value: str, colorbar_label: str, gain: bool):
    matrix, sizes, rows = matrix_values(frame, value)
    if gain:
        bound = max(20.0, float(np.nanpercentile(np.abs(matrix), 95)))
        norm = TwoSlopeNorm(vmin=-bound, vcenter=0.0, vmax=bound)
        cmap = "RdBu_r"
    else:
        norm = Normalize(vmin=0.0, vmax=max(100.0, float(np.nanpercentile(matrix, 95))))
        cmap = "viridis_r"
    image = axis.imshow(matrix, aspect="auto", cmap=cmap, norm=norm)
    axis.set_xticks(range(len(sizes)))
    axis.set_xticklabels(sizes)
    axis.set_yticks(range(len(rows)))
    axis.set_yticklabels([f"{network[0].upper()} {epoch}" for network, epoch in rows])
    axis.set_xlabel("Training structures")
    axis.set_ylabel("Network and epochs")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value_here = matrix[i, j]
            if np.isfinite(value_here):
                axis.text(j, i, f"{value_here:.0f}", ha="center", va="center", fontsize=6.5)
    colorbar = axis.figure.colorbar(image, ax=axis, pad=0.02)
    colorbar.set_label(colorbar_label)


def plot_matrices(material: str, kind: str) -> None:
    _, cells, effective = summarize(material, kind)
    low_to_high = kind == "low_to_high"
    if not low_to_high:
        for frame, value, label, suffix, gain in (
            (cells, "gain_percent", r"$I_{CC}$ (%)", "gain", True),
            (effective, "relative_data_percent", "Relative data requirement (%)", "effective_sample", False),
        ):
            figure, axis = plt.subplots(figsize=(7.3, 4.5), constrained_layout=True)
            draw_matrix(axis, frame, value, label, gain)
            figure.savefig(OUT / f"{material}_test_{suffix}.pdf", bbox_inches="tight")
            figure.savefig(OUT / f"{material}_test_{suffix}.png", dpi=300, bbox_inches="tight")
        return

    for frame, value, label, suffix, gain in (
        (cells, "gain_percent", r"$I_{CC}$ (%)", "gain", True),
        (effective, "relative_data_percent", "Relative data requirement (%)", "effective_sample", False),
    ):
        figure, axes = plt.subplots(2, 1, figsize=(7.3, 7.2), constrained_layout=True)
        for axis, task, letter in zip(axes, ("deformation", "temperature"), ("a", "b")):
            draw_matrix(axis, frame[frame["task"] == task], value, label, gain)
            axis.text(-0.09, 1.02, f"({letter})", transform=axis.transAxes, fontweight="bold")
        figure.savefig(OUT / f"{material}_low_to_high_{suffix}.pdf", bbox_inches="tight")
        figure.savefig(OUT / f"{material}_low_to_high_{suffix}.png", dpi=300, bbox_inches="tight")


def main() -> None:
    configure()
    OUT.mkdir(parents=True, exist_ok=True)
    for material in ("nacl", "cspbi3"):
        plot_test_curves(material)
        plot_low_to_high_curves(material)
        plot_matrices(material, "test")
        plot_matrices(material, "low_to_high")


if __name__ == "__main__":
    main()
