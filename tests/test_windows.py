import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crypto_generative.data.windows import REQUIRED_RETURN_COLUMNS, TemporalWindowBuilder


def return_row(index, valid=True):
    return {
        "open_time_utc": f"2020-01-{1 + index // 4:02d}T{(index % 4) * 6:02d}:00:00Z",
        "btc_log_return": str(index / 100) if valid else "",
        "eth_log_return": str(-index / 100) if valid else "",
        "returns_valid": "1" if valid else "0",
        "invalid_reason": "" if valid else "synthetic_gap",
    }


def write_returns(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_RETURN_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class TemporalWindowBuilderTest(unittest.TestCase):
    def test_builds_condition_and_target_with_correct_boundaries(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "returns.csv"
            write_returns(path, [return_row(index) for index in range(6)])

            dataset = TemporalWindowBuilder(condition_steps=2, target_steps=2).build(path)

        self.assertEqual(dataset.condition_returns.shape, (3, 2, 2))
        self.assertEqual(dataset.target_returns.shape, (3, 2, 2))
        np.testing.assert_allclose(dataset.condition_returns[0, :, 0], [0.0, 0.01])
        np.testing.assert_allclose(dataset.target_returns[0, :, 0], [0.02, 0.03])
        self.assertEqual(dataset.index_rows[0]["condition_end_utc"], "2020-01-01T06:00:00Z")
        self.assertEqual(dataset.index_rows[0]["target_start_utc"], "2020-01-01T12:00:00Z")

    def test_rejects_every_window_that_touches_invalid_return(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "returns.csv"
            rows = [return_row(index, valid=index != 4) for index in range(10)]
            write_returns(path, rows)

            dataset = TemporalWindowBuilder(condition_steps=2, target_steps=2).build(path)

        np.testing.assert_array_equal(dataset.start_indices, [0, 5, 6])
        self.assertEqual(dataset.audit.candidate_windows, 7)
        self.assertEqual(dataset.audit.rejected_windows, 4)
        self.assertEqual(dataset.audit.valid_windows, 3)

    def test_stride_reduces_candidate_windows(self):
        with TemporaryDirectory() as directory:
            path = Path(directory) / "returns.csv"
            write_returns(path, [return_row(index) for index in range(10)])

            dataset = TemporalWindowBuilder(
                condition_steps=2,
                target_steps=2,
                stride_steps=2,
            ).build(path)

        np.testing.assert_array_equal(dataset.start_indices, [0, 2, 4, 6])
        self.assertEqual(dataset.audit.candidate_windows, 4)


if __name__ == "__main__":
    unittest.main()
