"""Independent tests for the California Housing analysis contract."""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import housing_analysis as analysis  # noqa: E402

EXPECTED_COLUMNS = [
    "MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population",
    "AveOccup", "Latitude", "Longitude", "MedHouseVal",
]
DATA_PATH = ROOT / "data" / "california_housing.csv"


def sample_row(**overrides: float) -> dict[str, float]:
    row = {
        "MedInc": 3.5,
        "HouseAge": 20.0,
        "AveRooms": 5.0,
        "AveBedrms": 1.0,
        "Population": 100.0,
        "AveOccup": 2.5,
        "Latitude": 37.0,
        "Longitude": -120.0,
        "MedHouseVal": 2.0,
    }
    row.update(overrides)
    return row


class TestSchemaContract(unittest.TestCase):
    def test_expected_columns_are_explicit_and_ordered(self) -> None:
        self.assertEqual(analysis.EXPECTED_COLUMNS, EXPECTED_COLUMNS)
        self.assertEqual(len(analysis.EXPECTED_COLUMNS), 9)


class TestRangeChecks(unittest.TestCase):
    def test_valid_sample_has_no_range_violations(self) -> None:
        frame = pd.DataFrame([sample_row()], columns=EXPECTED_COLUMNS)
        checks = analysis.create_range_checks(frame)
        self.assertEqual(int(checks["observed_violations"].sum()), 0)
        self.assertTrue((checks["status"] == "PASS").all())

    def test_invalid_sample_is_flagged_by_each_applicable_rule(self) -> None:
        invalid = sample_row(
            MedInc=0.0,
            HouseAge=53.5,
            AveRooms=0.0,
            AveBedrms=0.0,
            Population=-1.0,
            AveOccup=0.0,
            Latitude=91.0,
            Longitude=-181.0,
            MedHouseVal=-1.0,
        )
        checks = analysis.create_range_checks(
            pd.DataFrame([sample_row(), invalid], columns=EXPECTED_COLUMNS)
        ).set_index("check")
        expected_violations = {
            "MedInc positive": 1,
            "HouseAge within plausible observed range": 1,
            "AveRooms positive": 1,
            "AveBedrms positive": 1,
            "Population non-negative": 1,
            "AveOccup positive": 1,
            "Latitude geographic bounds": 1,
            "Longitude geographic bounds": 1,
            "Target non-negative": 1,
            "HouseAge integer-valued": 1,
            "Population integer-valued": 0,
        }
        for name, count in expected_violations.items():
            with self.subTest(check=name):
                self.assertEqual(int(checks.loc[name, "observed_violations"]), count)
        self.assertEqual(checks.loc["HouseAge integer-valued", "status"], "REVIEW")


@unittest.skipUnless(DATA_PATH.is_file(), "Run `python prepare_dataset.py` to create the local source CSV.")
class TestPreparedDataset(unittest.TestCase):
    def test_dataset_matches_documented_contract(self) -> None:
        frame = pd.read_csv(DATA_PATH)
        self.assertEqual(frame.shape, (20_640, 9))
        self.assertEqual(list(frame.columns), EXPECTED_COLUMNS)
        self.assertEqual(int(frame.isna().sum().sum()), 0)
        self.assertTrue(np.isfinite(frame.to_numpy(dtype=float)).all())


if __name__ == "__main__":
    unittest.main(verbosity=2)
