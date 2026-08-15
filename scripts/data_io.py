#!/usr/bin/env python3
from pathlib import Path

import numpy as np


class Structure:
    def __init__(self, comment, cell, symbols, positions, charges):
        self.comment = str(comment)
        self.cell = np.asarray(cell, dtype=float).reshape(3, 3)
        self.symbols = list(symbols)
        self.positions = np.asarray(positions, dtype=float).reshape(-1, 3)
        self.charges = np.asarray(charges, dtype=float).reshape(-1)
        if not (len(self.symbols) == len(self.positions) == len(self.charges)):
            raise ValueError("Inconsistent atom counts in structure")


def read_fit_data(path, limit=None):
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    structures = []
    cursor = 0
    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1
        if cursor >= len(lines):
            break
        comment = ""
        if lines[cursor].lstrip().startswith("#"):
            comment = lines[cursor].strip()[1:].strip()
            cursor += 1
        n_atoms = int(lines[cursor].strip())
        cursor += 1
        cell = np.fromstring(lines[cursor], sep=" ", dtype=float)
        cursor += 1
        if cell.size != 9:
            raise ValueError("Expected 9 cell entries near line {}".format(cursor))
        symbols, positions, charges = [], [], []
        for _ in range(n_atoms):
            fields = lines[cursor].split()
            cursor += 1
            if len(fields) < 5:
                raise ValueError("Incomplete atom record near line {}".format(cursor))
            symbols.append(fields[0])
            positions.append([float(value) for value in fields[1:4]])
            charges.append(float(fields[-1]))
        structures.append(Structure(comment, cell, symbols, positions, charges))
        if limit is not None and len(structures) >= int(limit):
            break
    return structures


def write_fit_data(structures, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for index, structure in enumerate(structures):
            comment = structure.comment or "structure_{:05d}".format(index)
            handle.write("# {}\n".format(comment))
            handle.write("{}\n".format(len(structure.symbols)))
            handle.write("  " + "  ".join("{:.10f}".format(v) for v in structure.cell.ravel()) + "\n")
            for symbol, xyz, charge in zip(
                structure.symbols, structure.positions, structure.charges
            ):
                handle.write(
                    "{:<2s}  {: .10f}  {: .10f}  {: .10f}  "
                    "0.00000000  0.00000000  0.00000000  {: .10f}\n".format(
                        symbol, xyz[0], xyz[1], xyz[2], charge
                    )
                )


def subset_fit_data(source, destination, count=12):
    structures = read_fit_data(source, limit=count)
    if len(structures) != int(count):
        raise ValueError("Requested {} structures but found {}".format(count, len(structures)))
    write_fit_data(structures, destination)
