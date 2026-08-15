#!/usr/bin/env python3
import argparse
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "metrics"


def strict_pair(frame: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    conventional = frame.loc[frame["mode"].eq("baseline"), keys + ["mae_atom"]].rename(
        columns={"mae_atom": "mae_conventional"}
    )
    ccml = frame.loc[frame["mode"].eq("ic"), keys + ["mae_atom"]].rename(
        columns={"mae_atom": "mae_ccml"}
    )
    paired = conventional.merge(ccml, on=keys, how="outer", indicator=True, validate="one_to_one")
    if not paired["_merge"].eq("both").all():
        raise ValueError("Incomplete conventional-ML/CC-ML pairing")
    paired = paired.drop(columns="_merge")
    paired["gain_percent"] = 100.0 * (
        paired["mae_conventional"] - paired["mae_ccml"]
    ) / paired["mae_conventional"]
    return paired


def path_average(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["task", "network", "train_size", "total_epochs", "seed", "mode"]
    return frame.groupby(keys, as_index=False)["mae_atom"].mean()


def cell_average(paired: pd.DataFrame, low_to_high: bool) -> pd.DataFrame:
    keys = ["network", "total_epochs", "train_size"]
    if low_to_high:
        keys = ["task"] + keys
    cells = paired.groupby(keys, as_index=False)[["mae_conventional", "mae_ccml"]].mean()
    cells["gain_percent"] = 100.0 * (
        cells["mae_conventional"] - cells["mae_ccml"]
    ) / cells["mae_conventional"]
    return cells


def effective_sample(cells: pd.DataFrame, low_to_high: bool) -> pd.DataFrame:
    group_keys = ["network", "total_epochs"]
    if low_to_high:
        group_keys = ["task"] + group_keys
    rows = []
    for group, subset in cells.groupby(group_keys, sort=False):
        subset = subset.sort_values("train_size")
        sizes = subset["train_size"].to_numpy(dtype=int)
        envelope = np.minimum.accumulate(subset["mae_conventional"].to_numpy(dtype=float))
        group_values = group if isinstance(group, tuple) else (group,)
        metadata = dict(zip(group_keys, group_values))
        for item in subset.itertuples(index=False):
            reached = np.flatnonzero(envelope <= item.mae_ccml)
            if len(reached):
                reference_size = int(sizes[reached[0]])
                censored = False
            else:
                reference_size = int(sizes[-1])
                censored = True
            rows.append(
                {
                    **metadata,
                    "train_size": int(item.train_size),
                    "reference_size": reference_size,
                    "relative_data_percent": 100.0 * int(item.train_size) / reference_size,
                    "upper_bound": censored,
                }
            )
    return pd.DataFrame(rows)


def summarize(material: str, kind: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = DATA / f"{material}_{kind}_metrics.csv"
    raw = pd.read_csv(path)
    low_to_high = kind == "low_to_high"
    if low_to_high:
        raw = path_average(raw)
        pair_keys = ["task", "network", "train_size", "total_epochs", "seed"]
    else:
        pair_keys = ["network", "train_size", "total_epochs", "seed"]
    paired = strict_pair(raw, pair_keys)
    cells = cell_average(paired, low_to_high)
    effective = effective_sample(cells, low_to_high)
    return paired, cells, effective


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "derived")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summaries = []
    for material in ("nacl", "cspbi3"):
        for kind in ("test", "low_to_high"):
            paired, cells, effective = summarize(material, kind)
            paired.to_csv(args.output / f"{material}_{kind}_paired.csv", index=False)
            cells.to_csv(args.output / f"{material}_{kind}_cells.csv", index=False)
            effective.to_csv(args.output / f"{material}_{kind}_effective_sample.csv", index=False)
            summaries.append(
                {
                    "material": material,
                    "analysis": kind,
                    "paired_settings": len(paired),
                    "positive_settings": int((paired["gain_percent"] > 0).sum()),
                    "positive_percent": 100.0 * float((paired["gain_percent"] > 0).mean()),
                    "mean_pairwise_gain_percent": float(paired["gain_percent"].mean()),
                    "matrix_cells": len(cells),
                    "positive_cells": int((cells["gain_percent"] > 0).sum()),
                }
            )
    pd.DataFrame(summaries).to_csv(args.output / "summary.csv", index=False)
    print(pd.DataFrame(summaries).to_string(index=False))


if __name__ == "__main__":
    main()
