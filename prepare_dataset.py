#!/usr/bin/env python3
"""Prepare the California Housing CSV using scikit-learn's documented loader."""
from __future__ import annotations

from pathlib import Path

from sklearn.datasets import fetch_california_housing

EXPECTED_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude", "MedHouseVal",
]
EXPECTED_SHAPE = (20_640, 9)


def main() -> int:
    project_dir = Path(__file__).resolve().parent
    output_path = project_dir / "data" / "california_housing.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dataset = fetch_california_housing(as_frame=True)
    frame = dataset.frame
    if tuple(frame.shape) != EXPECTED_SHAPE:
        raise ValueError(f"Expected dataset shape {EXPECTED_SHAPE}, got {tuple(frame.shape)}")
    if list(frame.columns) != EXPECTED_COLUMNS:
        raise ValueError(f"Unexpected dataset columns: {list(frame.columns)}")

    temporary_path = output_path.with_suffix(".csv.tmp")
    try:
        frame.to_csv(temporary_path, index=False)
        temporary_path.replace(output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    print(f"Prepared {len(frame):,} rows x {len(frame.columns)} columns.")
    print(f"CSV written to: {output_path}")
    print("The analysis script reads this file without modifying it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
