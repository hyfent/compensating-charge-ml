#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccml_core import ChargeMLP, ccml_predict, neighbor_list, radial_descriptor, species_one_hot, total_charge_projection
from data_io import read_fit_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--artifact", required=True)
    args = parser.parse_args()
    artifact = Path(args.artifact)
    meta = json.loads((artifact / "metadata.json").read_text(encoding="utf-8"))
    cfg, species = meta["model_config"], meta["species"]
    model = ChargeMLP(meta["input_size"], meta["hidden_layers"])
    model.load_state_dict(torch.load(artifact / "model.pt", map_location="cpu"))
    model.eval()
    scales = np.load(artifact / "scalers.npz")
    structures = read_fit_data(args.data)
    all_errors = []
    for structure in structures:
        geometry = radial_descriptor(
            structure.positions, structure.cell, structure.symbols, species,
            cfg["cutoff_radius_angstrom"], cfg["radial_eta"], cfg["radial_shift_angstrom"]
        )
        neighbors = neighbor_list(
            structure.positions, structure.cell, cfg["cutoff_radius_angstrom"]
        )

        def predictor(features, round_index):
            mean, scale = scales["mean_{}".format(round_index)], scales["scale_{}".format(round_index)]
            with torch.no_grad():
                x = torch.as_tensor((features - mean) / scale, dtype=torch.float32)
                return model(x).cpu().numpy()

        if meta["mode"] == "ccml":
            predicted = ccml_predict(
                predictor, geometry, structure.symbols, species, neighbors,
                cfg["formal_charges"], meta["feedback_rounds"], cfg["total_charge"]
            )
        else:
            features = np.column_stack([geometry, species_one_hot(structure.symbols, species)])
            predicted = total_charge_projection(predictor(features, 0), cfg["total_charge"])
        all_errors.append(predicted - structure.charges)
    errors = np.concatenate(all_errors)
    print(json.dumps({
        "structures": len(structures),
        "atoms": int(errors.size),
        "mae": float(np.mean(np.abs(errors))),
        "rmse": float(np.sqrt(np.mean(errors ** 2))),
    }, indent=2))


if __name__ == "__main__":
    main()
