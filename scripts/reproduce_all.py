#!/usr/bin/env python3
import subprocess
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent


def run(name: str) -> None:
    subprocess.run([sys.executable, str(HERE / name)], check=True)


if __name__ == "__main__":
    for script in (
        "validate_release.py",
        "analysis.py",
        "plot_madelung.py",
        "plot_performance.py",
        "plot_structural_shift.py",
        "plot_timing.py",
    ):
        run(script)
    print("All compact-release analyses completed.")
