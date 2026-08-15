#!/usr/bin/env python3
import json
from pathlib import Path

import pandas as pd

from analysis import summarize
from data_io import read_fit_data


ROOT = Path(__file__).resolve().parents[1]


def require(path: Path) -> Path:
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def main() -> None:
    expected_rows = {
        "nacl_test_metrics.csv": 432,
        "cspbi3_test_metrics.csv": 432,
        "nacl_low_to_high_metrics.csv": 1296,
        "cspbi3_low_to_high_metrics.csv": 1296,
    }
    for name, expected in expected_rows.items():
        path = require(ROOT / "data" / "metrics" / name)
        rows = len(pd.read_csv(path))
        if rows != expected:
            raise ValueError(f"{name}: expected {expected} rows, found {rows}")
    auxiliary = {
        "madelung_convergence.csv": 3002,
        "timing_raw.csv": 180,
        "structural_shift_distances.csv": 1600,
    }
    for name, expected in auxiliary.items():
        rows = len(pd.read_csv(require(ROOT / "data" / "auxiliary" / name)))
        if rows != expected:
            raise ValueError(f"{name}: expected {expected} rows, found {rows}")
    for material in ("nacl", "cspbi3"):
        for kind in ("test", "low_to_high"):
            summarize(material, kind)
    config = json.loads(require(ROOT / "config" / "model_config.json").read_text())
    if config["feedback_rounds"] != 3 or config["cutoff_radius_angstrom"] != 6.0:
        raise ValueError("Unexpected model configuration")
    examples = {
        "nacl_12_structures.fit.data": (12, 216),
        "cspbi3_12_structures.fit.data": (12, 320),
    }
    for name, (expected_structures, expected_atoms) in examples.items():
        structures = read_fit_data(require(ROOT / "data" / "example" / name))
        if len(structures) != expected_structures:
            raise ValueError(f"{name}: expected {expected_structures} structures")
        if any(len(structure.symbols) != expected_atoms for structure in structures):
            raise ValueError(f"{name}: unexpected atom count")
        if any(abs(structure.charges.sum()) > 1e-6 for structure in structures):
            raise ValueError(f"{name}: total charge check failed")
    for script in ("qeq_label.py", "train_example.py", "predict.py", "data_io.py"):
        require(ROOT / "scripts" / script)
    print("Release validation passed.")


if __name__ == "__main__":
    main()
