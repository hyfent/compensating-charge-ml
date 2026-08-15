#!/usr/bin/env python3
import argparse
import json
import math
from itertools import product
from pathlib import Path

import numpy as np

from data_io import Structure, read_fit_data, write_fit_data


def load_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def lattice_indices(limit, include_zero=True):
    values = np.asarray(list(product(range(-limit, limit + 1), repeat=3)), dtype=int)
    if not include_zero:
        values = values[np.any(values != 0, axis=1)]
    return values


def ewald_matrix(positions, cell, alpha, real_cutoff, reciprocal_limit):
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    n_atoms = len(positions)
    volume = abs(float(np.linalg.det(cell)))
    if volume <= 0.0:
        raise ValueError("Cell volume must be positive")

    shortest = min(np.linalg.norm(cell, axis=1))
    real_limit = int(math.ceil(float(real_cutoff) / shortest)) + 1
    translations = lattice_indices(real_limit) @ cell
    displacement = positions[None, :, :] - positions[:, None, :]
    interaction = np.zeros((n_atoms, n_atoms), dtype=float)
    for translation in translations:
        vectors = displacement + translation
        distance = np.linalg.norm(vectors, axis=-1)
        mask = (distance <= real_cutoff) & (distance > 1e-12)
        screened = np.asarray([math.erfc(alpha * value) for value in distance[mask]])
        interaction[mask] += screened / distance[mask]

    reciprocal_basis = 2.0 * np.pi * np.linalg.inv(cell).T
    k_vectors = lattice_indices(int(reciprocal_limit), include_zero=False) @ reciprocal_basis
    k_squared = np.einsum("ij,ij->i", k_vectors, k_vectors)
    weights = np.exp(-k_squared / (4.0 * alpha ** 2)) / k_squared
    phases = positions @ k_vectors.T
    factor = np.sqrt(weights)
    cosine = np.cos(phases) * factor
    sine = np.sin(phases) * factor
    interaction += (4.0 * np.pi / volume) * (cosine @ cosine.T + sine @ sine.T)
    interaction[np.diag_indices(n_atoms)] -= 2.0 * alpha / math.sqrt(np.pi)
    return interaction


def solve_structure(structure, qeq, total_charge):
    coulomb = ewald_matrix(
        structure.positions,
        structure.cell,
        float(qeq["alpha"]),
        float(qeq["cut"]),
        int(qeq["gmax"]),
    )
    symbols = structure.symbols
    chi = np.asarray([qeq["chi"][symbol] for symbol in symbols], dtype=float)
    hardness = np.asarray([qeq["eta"][symbol] for symbol in symbols], dtype=float)
    factor = float(qeq.get("ke", 1.0))
    if qeq.get("use_epsr", False):
        factor /= float(qeq["eps_r"])
    matrix = factor * coulomb
    matrix[np.diag_indices(len(symbols))] += hardness
    kkt = np.zeros((len(symbols) + 1, len(symbols) + 1), dtype=float)
    rhs = np.zeros(len(symbols) + 1, dtype=float)
    kkt[:-1, :-1] = matrix
    kkt[:-1, -1] = 1.0
    kkt[-1, :-1] = 1.0
    rhs[:-1] = -chi
    rhs[-1] = float(total_charge)
    return np.linalg.solve(kkt, rhs)[:-1]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--qeq", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    qeq = load_json(args.qeq)
    structures = read_fit_data(args.data, limit=args.limit)
    labeled = []
    for index, structure in enumerate(structures):
        charges = solve_structure(structure, qeq, qeq.get("qtot", 0.0))
        labeled.append(
            Structure(
                structure.comment + " relabeled=compact_Ewald-QEq",
                structure.cell,
                structure.symbols,
                structure.positions,
                charges,
            )
        )
        print("labeled structure {}: sum(q)={:.3e}".format(index, charges.sum()))
    write_fit_data(labeled, args.output)


if __name__ == "__main__":
    main()
