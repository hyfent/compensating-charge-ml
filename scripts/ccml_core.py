#!/usr/bin/env python3
from typing import Callable, List, Mapping, Sequence

import numpy as np

try:
    import torch
    from torch import nn
except ImportError:
    torch = None
    nn = None


def minimum_image_displacements(positions: np.ndarray, cell: np.ndarray) -> np.ndarray:
    positions = np.asarray(positions, dtype=float)
    cell = np.asarray(cell, dtype=float)
    inverse = np.linalg.inv(cell)
    displacement = positions[None, :, :] - positions[:, None, :]
    fractional = displacement @ inverse
    fractional -= np.round(fractional)
    return fractional @ cell


def neighbor_list(positions: np.ndarray, cell: np.ndarray, cutoff: float) -> List[np.ndarray]:
    displacement = minimum_image_displacements(positions, cell)
    distance = np.linalg.norm(displacement, axis=-1)
    return [np.flatnonzero(distance[i] < cutoff) for i in range(len(distance))]


def compensating_charge(charges: np.ndarray, neighborhoods: Sequence[np.ndarray]) -> np.ndarray:
    charges = np.asarray(charges, dtype=float)
    return np.asarray([-charges[index].sum() for index in neighborhoods], dtype=float)


def total_charge_projection(charges: np.ndarray, total_charge: float = 0.0) -> np.ndarray:
    charges = np.asarray(charges, dtype=float)
    return charges + (float(total_charge) - charges.sum()) / len(charges)


def radial_descriptor(
    positions: np.ndarray,
    cell: np.ndarray,
    symbols: Sequence[str],
    species: Sequence[str],
    cutoff: float = 6.0,
    etas: Sequence[float] = (0.05, 0.5, 4.0),
    shifts: Sequence[float] = (0.0, 1.5, 4.0),
) -> np.ndarray:
    displacement = minimum_image_displacements(positions, cell)
    distance = np.linalg.norm(displacement, axis=-1)
    np.fill_diagonal(distance, np.inf)
    inside = distance < cutoff
    cutoff_value = np.zeros_like(distance)
    cutoff_value[inside] = 0.5 * (np.cos(np.pi * distance[inside] / cutoff) + 1.0)
    symbols = np.asarray(symbols)
    columns = []
    for neighbor_species in species:
        species_mask = symbols == neighbor_species
        for eta in etas:
            for shift in shifts:
                values = np.zeros_like(distance)
                values[inside] = (
                    np.exp(-float(eta) * (distance[inside] - float(shift)) ** 2)
                    * cutoff_value[inside]
                )
                columns.append(values[:, species_mask].sum(axis=1))
    return np.column_stack(columns)


def species_one_hot(symbols: Sequence[str], species: Sequence[str]) -> np.ndarray:
    lookup = {symbol: index for index, symbol in enumerate(species)}
    indices = np.asarray([lookup[symbol] for symbol in symbols], dtype=int)
    return np.eye(len(species), dtype=float)[indices]


def make_input(
    geometry: np.ndarray,
    delta_q: np.ndarray,
    symbols: Sequence[str],
    species: Sequence[str],
) -> np.ndarray:
    return np.column_stack(
        [np.asarray(geometry, dtype=float), np.asarray(delta_q, dtype=float), species_one_hot(symbols, species)]
    )


if nn is not None:
    class ChargeMLP(nn.Module):
        def __init__(self, input_size: int, hidden_layers: Sequence[int]):
            super().__init__()
            layers = []
            current = int(input_size)
            for width in hidden_layers:
                layers.extend((nn.Linear(current, int(width)), nn.SiLU()))
                current = int(width)
            layers.append(nn.Linear(current, 1))
            self.network = nn.Sequential(*layers)

        def forward(self, values):
            return self.network(values).squeeze(-1)


def ccml_predict(
    predictor: Callable[[np.ndarray, int], np.ndarray],
    geometry: np.ndarray,
    symbols: Sequence[str],
    species: Sequence[str],
    neighborhoods: Sequence[np.ndarray],
    formal_charges: Mapping[str, float],
    rounds: int = 3,
    total_charge: float = 0.0,
) -> np.ndarray:
    current = np.asarray([formal_charges[symbol] for symbol in symbols], dtype=float)
    for round_index in range(int(rounds)):
        delta_q = compensating_charge(current, neighborhoods)
        features = make_input(geometry, delta_q, symbols, species)
        raw = np.asarray(predictor(features, round_index), dtype=float).reshape(-1)
        current = total_charge_projection(raw, total_charge)
    return current
