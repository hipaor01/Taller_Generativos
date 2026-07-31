import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from crypto_generative.data.splits import (
    REQUIRED_INDEX_COLUMNS,
    PurgedTemporalSplitBuilder,
    TemporalSplitConfig,
)


def row(sample_id, condition_start, target_start, target_end):
    return {
        "sample_id": sample_id,
        "condition_start_utc": condition_start,
        "condition_end_utc": target_start,
        "target_start_utc": target_start,
        "target_end_utc": target_end,
    }


def write_index(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=REQUIRED_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


class PurgedTemporalSplitBuilderTest(unittest.TestCase):
    config = TemporalSplitConfig(
        train_end_exclusive_utc="2020-01-03T00:00:00Z",
        validation_start_utc="2020-01-05T00:00:00Z",
        validation_end_exclusive_utc="2020-01-07T00:00:00Z",
        test_start_utc="2020-01-09T00:00:00Z",
        minimum_purge_days=2,
    )

    def test_assigns_all_blocks_and_preserves_raw_separation(self):
        rows = [
            row(0, "2019-12-30T00:00:00Z", "2020-01-01T00:00:00Z", "2020-01-02T00:00:00Z"),
            row(1, "2020-01-01T00:00:00Z", "2020-01-03T00:00:00Z", "2020-01-04T00:00:00Z"),
            row(2, "2020-01-03T00:00:00Z", "2020-01-05T00:00:00Z", "2020-01-06T00:00:00Z"),
            row(3, "2020-01-05T00:00:00Z", "2020-01-07T00:00:00Z", "2020-01-08T00:00:00Z"),
            row(4, "2020-01-07T00:00:00Z", "2020-01-09T00:00:00Z", "2020-01-10T00:00:00Z"),
        ]
        with TemporaryDirectory() as directory:
            path = Path(directory) / "index.csv"
            write_index(path, rows)
            split = PurgedTemporalSplitBuilder(self.config).build(path)

        self.assertEqual(split.train_ids.tolist(), [0])
        self.assertEqual(split.purge_train_validation_ids.tolist(), [1])
        self.assertEqual(split.validation_ids.tolist(), [2])
        self.assertEqual(split.purge_validation_test_ids.tolist(), [3])
        self.assertEqual(split.test_ids.tolist(), [4])
        self.assertFalse(split.audit.raw_intervals_overlap)

    def test_rejects_purge_shorter_than_configured_minimum(self):
        config = TemporalSplitConfig(
            train_end_exclusive_utc="2020-01-03T00:00:00Z",
            validation_start_utc="2020-01-04T00:00:00Z",
            validation_end_exclusive_utc="2020-01-07T00:00:00Z",
            test_start_utc="2020-01-09T00:00:00Z",
            minimum_purge_days=2,
        )
        with self.assertRaisesRegex(ValueError, "inferior al minimo"):
            PurgedTemporalSplitBuilder(config)


if __name__ == "__main__":
    unittest.main()
