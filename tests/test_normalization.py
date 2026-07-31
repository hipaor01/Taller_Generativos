import csv
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from crypto_generative.data.normalization import TrainingOnlyNormalizer, ZScoreScaler


def write_fixture(root):
    windows_path = root / "windows.npz"
    conditions_path = root / "conditions.npz"
    split_path = root / "split.npz"
    returns_path = root / "returns.csv"

    raw = np.asarray([[1, 10], [2, 20], [3, 30], [4, 40], [5, 50], [6, 60]], dtype=float)
    condition = np.stack([raw[start : start + 2] for start in range(4)])
    target = np.stack([raw[start + 2 : start + 3] for start in range(4)])
    with windows_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            condition_returns=condition,
            target_returns=target,
            start_indices=np.arange(4, dtype=np.int64),
            assets=np.asarray(["BTC", "ETH"]),
        )
    features = np.asarray([[1, 10], [3, 20], [100, 30], [200, 40]], dtype=float)
    with conditions_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            condition_features=features,
            feature_names=np.asarray(["feature_a", "feature_b"]),
            sample_ids=np.arange(4, dtype=np.int64),
        )
    with split_path.open("wb") as handle:
        np.savez_compressed(
            handle,
            train_sample_ids=np.asarray([0, 1]),
            validation_sample_ids=np.asarray([2]),
            test_sample_ids=np.asarray([3]),
            purge_train_validation_sample_ids=np.asarray([], dtype=np.int64),
            purge_validation_test_sample_ids=np.asarray([], dtype=np.int64),
        )
    with returns_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("btc_log_return", "eth_log_return", "returns_valid"),
        )
        writer.writeheader()
        for btc, eth in raw:
            writer.writerow(
                {
                    "btc_log_return": btc,
                    "eth_log_return": eth,
                    "returns_valid": "1",
                }
            )
    return windows_path, conditions_path, split_path, returns_path


class TrainingOnlyNormalizerTest(unittest.TestCase):
    def test_fits_only_train_and_transforms_validation_test_without_refit(self):
        with TemporaryDirectory() as directory:
            paths = write_fixture(Path(directory))
            dataset = TrainingOnlyNormalizer().build(*paths)

        np.testing.assert_allclose(dataset.condition_parameters.mean, [2, 15])
        np.testing.assert_allclose(dataset.condition_parameters.scale, [1, 5])
        np.testing.assert_allclose(dataset.condition_features[:2].mean(axis=0), [0, 0])
        self.assertGreater(dataset.condition_features[2, 0], 90)
        np.testing.assert_allclose(dataset.return_parameters.mean, [2.5, 25])
        self.assertEqual(dataset.audit.unique_training_return_rows, 4)
        self.assertEqual(dataset.audit.normalized_non_finite_values, 0)

    def test_zero_variance_column_uses_unit_scale(self):
        values = np.asarray([[1.0, 2.0], [1.0, 4.0]])
        parameters = ZScoreScaler.fit(values)
        transformed = ZScoreScaler.transform(values, parameters)

        self.assertEqual(parameters.zero_variance_columns, [0])
        self.assertEqual(parameters.scale[0], 1.0)
        np.testing.assert_allclose(transformed[:, 0], [0, 0])


if __name__ == "__main__":
    unittest.main()
