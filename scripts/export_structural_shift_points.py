#!/usr/bin/env python3
import argparse
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

import numpy as np
import pandas as pd


def write_projected_distances(prepare_panel, fit_projection, output: Path) -> None:
    rows = []
    for material in ("NaCl", "CsPbI3"):
        for task in ("Deformation", "Temperature"):
            projected, _, retained = fit_projection(prepare_panel(material, task))
            for item in projected:
                label = item["label"].replace("$", "").replace("\\lambda", "lambda")
                for index, value in enumerate(np.asarray(item["distance"], dtype=float)):
                    rows.append(
                        {
                            "material": material,
                            "task": task.lower(),
                            "condition": label,
                            "role": item["role"],
                            "structure_index": index,
                            "distance": value,
                            "retained_pcs_95pct": retained,
                        }
                    )
    pd.DataFrame(rows).to_csv(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=Path,
        required=True,
        help="Original plot_structural_shift_pca.py with access to trajectory files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "data"
        / "auxiliary"
        / "structural_shift_distances.csv",
    )
    args = parser.parse_args()
    specification = spec_from_file_location("structural_shift_source", str(args.source.resolve()))
    module = module_from_spec(specification)
    specification.loader.exec_module(module)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_projected_distances(module.prepare_panel, module.fit_projection, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
