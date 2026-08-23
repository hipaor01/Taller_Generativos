import csv
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crypto_generative.data.condition import ConditionFeatureBuilder, INDEX_COLUMNS


PANEL_COLUMNS = (
    "open_time_utc",
    "btc_high",
    "btc_low",
    "btc_close",
    "btc_volume",
    "eth_high",
    "eth_low",
    "eth_close",
    "eth_volume",
    "is_complete",
)


def timestamp(index):
    value = datetime(2020, 1, 1, tzinfo=timezone.utc) + timedelta(hours=6 * index)
    return value.isoformat().replace("+00:00", "Z")


def write_fixture(root, complete=True):
    windows_path = root / "windows.npz"
    index_path = root / "index.csv"
    panel_path = root / "panel.csv"
    condition = np.asarray(
        [[[0.01, 0.02], [0.02, 0.04], [-0.01, -0.02], [0.03, 0.06]]],
        dtype=np.float64,
    )
    target = np.zeros((1, 2, 2), dtype=np.float64)
    with windows_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            condition_returns=condition,
            target_returns=target,
            start_indices=np.asarray([0], dtype=np.int64),
            assets=np.asarray(["BTC", "ETH"]),
        )
    with index_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": 0,
                "condition_start_utc": timestamp(0),
                "condition_end_utc": timestamp(3),
                "target_start_utc": timestamp(4),
                "target_end_utc": timestamp(5),
            }
        )
    with panel_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=PANEL_COLUMNS)
        writer.writeheader()
        for index in range(6):
            writer.writerow(
                {
                    "open_time_utc": timestamp(index),
                    "btc_high": 110 + index,
                    "btc_low": 100 + index,
                    "btc_close": 105 + index,
                    "btc_volume": 10 + index,
                    "eth_high": 55 + index,
                    "eth_low": 50 + index,
                    "eth_close": 52 + index,
                    "eth_volume": 20 + index,
                    "is_complete": "0" if index == 1 and not complete else "1",
                }
            )
    return windows_path, index_path, panel_path


class ConditionFeatureBuilderTest(unittest.TestCase):
    def test_builds_finite_summary_in_declared_order(self):
        with TemporaryDirectory() as directory:
            paths = write_fixture(Path(directory))
            dataset = ConditionFeatureBuilder(
                volume_recent_steps=2,
                correlation_steps=3,
                annualization_days=1,
            ).build(*paths)

        self.assertEqual(dataset.features.shape, (1, 14))
        self.assertEqual(len(dataset.feature_names), 14)
        self.assertAlmostEqual(dataset.features[0, 0], 0.05)
        self.assertAlmostEqual(dataset.features[0, 6], 0.10)
        self.assertAlmostEqual(dataset.features[0, 12], 1.0)
        self.assertTrue(np.isfinite(dataset.features).all())
        self.assertEqual(dataset.audit.non_finite_values, 0)

    def test_rejects_incomplete_condition_panel(self):
        with TemporaryDirectory() as directory:
            paths = write_fixture(Path(directory), complete=False)
            with self.assertRaisesRegex(ValueError, "contiene velas incompletas"):
                ConditionFeatureBuilder(
                    volume_recent_steps=2,
                    correlation_steps=3,
                ).build(*paths)

    def test_builds_latest_condition_without_requiring_a_future_target(self):
        with TemporaryDirectory() as directory:
            _, _, panel_path = write_fixture(Path(directory))
            snapshot = ConditionFeatureBuilder(
                volume_recent_steps=2,
                correlation_steps=3,
                annualization_days=1,
            ).build_latest(panel_path, condition_steps=4)

        self.assertEqual(snapshot.features.shape, (14,))
        self.assertEqual(snapshot.condition_returns.shape, (4, 2))
        self.assertEqual(snapshot.condition_start_utc, timestamp(2))
        self.assertEqual(snapshot.condition_end_utc, timestamp(5))
        self.assertEqual(snapshot.forecast_start_utc, timestamp(6))
        np.testing.assert_allclose(snapshot.initial_prices, [110.0, 57.0])


if __name__ == "__main__":
    unittest.main()
