#!/usr/bin/env python3
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ccml_core import (
    ChargeMLP,
    compensating_charge,
    make_input,
    neighbor_list,
    radial_descriptor,
    species_one_hot,
    total_charge_projection,
)
from data_io import read_fit_data


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def prepare(structures, species, cfg):
    geometries, neighborhoods = [], []
    for structure in structures:
        geometries.append(
            radial_descriptor(
                structure.positions,
                structure.cell,
                structure.symbols,
                species,
                cfg["cutoff_radius_angstrom"],
                cfg["radial_eta"],
                cfg["radial_shift_angstrom"],
            )
        )
        neighborhoods.append(
            neighbor_list(structure.positions, structure.cell, cfg["cutoff_radius_angstrom"])
        )
    return geometries, neighborhoods


def features_for(structures, geometries, neighborhoods, species, current, mode):
    blocks = []
    for structure, geometry, neighbors, charges in zip(
        structures, geometries, neighborhoods, current
    ):
        if mode == "ccml":
            delta = compensating_charge(charges, neighbors)
            blocks.append(make_input(geometry, delta, structure.symbols, species))
        else:
            blocks.append(np.column_stack([geometry, species_one_hot(structure.symbols, species)]))
    return blocks


def concatenate_indices(blocks, structures, indices):
    x = np.concatenate([blocks[index] for index in indices], axis=0)
    y = np.concatenate([structures[index].charges for index in indices], axis=0)
    return x, y


def predict_blocks(model, blocks, mean, scale, structures, total_charge):
    outputs = []
    model.eval()
    with torch.no_grad():
        for block, structure in zip(blocks, structures):
            values = torch.as_tensor((block - mean) / scale, dtype=torch.float32)
            raw = model(values).cpu().numpy()
            outputs.append(total_charge_projection(raw, total_charge))
    return outputs


def metrics(predictions, structures, indices):
    errors = np.concatenate(
        [predictions[index] - structures[index].charges for index in indices], axis=0
    )
    return {"mae": float(np.mean(np.abs(errors))), "rmse": float(np.sqrt(np.mean(errors ** 2)))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--material", required=True)
    parser.add_argument("--config", default="config/model_config.json")
    parser.add_argument("--mode", choices=("conventional", "ccml"), default="ccml")
    parser.add_argument("--network", choices=("small", "medium", "large"), default="small")
    parser.add_argument("--epochs", type=int, default=30, help="Total optimization budget")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="example_outputs/ccml")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    cfg, material = load_json(args.config), load_json(args.material)
    species = list(material["types"].keys())
    structures = read_fit_data(args.data)
    if len(structures) < 6:
        raise ValueError("At least six structures are required for the example split")
    order = np.random.permutation(len(structures))
    n_test = max(1, len(structures) // 6)
    n_val = max(1, len(structures) // 6)
    test_indices = order[:n_test].tolist()
    val_indices = order[n_test:n_test + n_val].tolist()
    train_indices = order[n_test + n_val:].tolist()

    geometries, neighborhoods = prepare(structures, species, cfg)
    formal = cfg["formal_charges"]
    current = [np.asarray([formal[s] for s in st.symbols], dtype=float) for st in structures]
    rounds = int(cfg["feedback_rounds"]) if args.mode == "ccml" else 1
    base_epochs, remainder = divmod(args.epochs, rounds)
    round_epochs = [base_epochs + (index < remainder) for index in range(rounds)]
    model, scalers = None, []

    for round_index, epochs in enumerate(round_epochs):
        blocks = features_for(
            structures, geometries, neighborhoods, species, current, args.mode
        )
        train_x, train_y = concatenate_indices(blocks, structures, train_indices)
        mean = train_x.mean(axis=0)
        scale = train_x.std(axis=0)
        scale[scale < 1e-12] = 1.0
        scalers.append((mean, scale))
        if model is None:
            model = ChargeMLP(train_x.shape[1], cfg["networks"][args.network])
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=float(cfg["learning_rate"]),
            weight_decay=float(cfg["weight_decay"]),
            eps=float(cfg["adamw_epsilon"]),
        )
        x_tensor = torch.as_tensor((train_x - mean) / scale, dtype=torch.float32)
        y_tensor = torch.as_tensor(train_y, dtype=torch.float32)
        model.train()
        for _ in range(epochs):
            optimizer.zero_grad()
            loss = torch.mean((model(x_tensor) - y_tensor) ** 2)
            loss.backward()
            optimizer.step()
        current = predict_blocks(
            model, blocks, mean, scale, structures, float(cfg["total_charge"])
        )

    result = {
        "train": metrics(current, structures, train_indices),
        "validation": metrics(current, structures, val_indices),
        "test": metrics(current, structures, test_indices),
    }
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output / "model.pt")
    np.savez(
        output / "scalers.npz",
        **{key: value for i, (mean, scale) in enumerate(scalers)
           for key, value in (("mean_{}".format(i), mean), ("scale_{}".format(i), scale))}
    )
    metadata = {
        "mode": args.mode,
        "network": args.network,
        "hidden_layers": cfg["networks"][args.network],
        "input_size": int(scalers[0][0].size),
        "species": species,
        "feedback_rounds": rounds,
        "total_epochs": args.epochs,
        "round_epochs": round_epochs,
        "seed": args.seed,
        "split": {"train": train_indices, "validation": val_indices, "test": test_indices},
        "metrics": result,
        "model_config": cfg,
    }
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
