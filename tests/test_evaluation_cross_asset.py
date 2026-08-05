import unittest

import numpy as np

from crypto_generative.evaluation import (
    CrossAssetDependenceConfig,
    TrajectoryEvaluator,
)


class TrajectoryEvaluatorCrossAssetTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = TrajectoryEvaluator()
        self.config = CrossAssetDependenceConfig(
            rolling_window=5,
            stress_quantile=0.80,
            joint_drop_quantile=0.10,
            lower_tail_quantile=0.10,
        )
        rng = np.random.default_rng(42)
        common = rng.normal(scale=0.02, size=(100, 30))
        first = common + rng.normal(scale=0.003, size=common.shape)
        second = 0.8 * common + rng.normal(scale=0.003, size=common.shape)
        self.reference = np.stack((first, second), axis=2)

    def test_identical_batches_have_zero_cross_asset_errors(self):
        result = self.evaluator.evaluate_cross_asset_dependence(
            self.reference,
            self.reference.copy(),
            self.config,
        )

        self.assertAlmostEqual(
            result.contemporaneous_correlation_absolute_error,
            0.0,
        )
        self.assertAlmostEqual(result.rolling_correlation_wasserstein_1, 0.0)
        self.assertAlmostEqual(result.calm_correlation_absolute_error, 0.0)
        self.assertAlmostEqual(result.stress_correlation_absolute_error, 0.0)
        self.assertAlmostEqual(result.joint_drop_probability_absolute_error, 0.0)
        self.assertAlmostEqual(result.lower_tail_dependence_absolute_error, 0.0)

    def test_breaking_pairing_is_detected(self):
        rng = np.random.default_rng(7)
        independent_candidate = self.reference.copy()
        independent_candidate[:, :, 1] = rng.permutation(
            independent_candidate[:, :, 1].ravel()
        ).reshape(independent_candidate.shape[:2])

        result = self.evaluator.evaluate_cross_asset_dependence(
            self.reference,
            independent_candidate,
            self.config,
        )

        self.assertGreater(result.contemporaneous_correlation_absolute_error, 0.5)
        self.assertGreater(result.rolling_correlation_wasserstein_1, 0.5)
        self.assertGreater(result.lower_tail_dependence_absolute_error, 0.3)

    def test_joint_drop_thresholds_are_fitted_only_on_reference(self):
        downward_shift = self.reference - 0.20
        result = self.evaluator.evaluate_cross_asset_dependence(
            self.reference,
            downward_shift,
            self.config,
        )

        self.assertGreater(
            result.candidate.joint_drop_probability,
            result.reference.joint_drop_probability,
        )
        self.assertAlmostEqual(result.lower_tail_dependence_absolute_error, 0.0)
        self.assertEqual(set(result.joint_drop_thresholds), {"BTC", "ETH"})

    def test_rolling_correlation_never_crosses_path_boundaries(self):
        paths = np.empty((2, 8, 2), dtype=float)
        increasing = np.arange(8, dtype=float)
        paths[0, :, 0] = increasing
        paths[0, :, 1] = increasing
        paths[1, :, 0] = increasing
        paths[1, :, 1] = -increasing
        config = CrossAssetDependenceConfig(rolling_window=4)

        result = self.evaluator.evaluate_cross_asset_dependence(paths, paths, config)

        self.assertAlmostEqual(result.reference.rolling_correlation.mean, 0.0)
        self.assertAlmostEqual(result.reference.rolling_correlation.valid_fraction, 1.0)

    def test_degenerate_candidate_is_reported_without_crashing(self):
        candidate = np.zeros_like(self.reference)
        result = self.evaluator.evaluate_cross_asset_dependence(
            self.reference,
            candidate,
            self.config,
        )

        self.assertIsNone(result.candidate.contemporaneous_correlation)
        self.assertEqual(result.candidate.rolling_correlation.valid_fraction, 0.0)
        self.assertIsNone(result.rolling_correlation_wasserstein_1)
        self.assertIsNone(result.contemporaneous_correlation_absolute_error)

    def test_records_are_ready_for_tabular_consumption(self):
        result = self.evaluator.evaluate_cross_asset_dependence(
            self.reference,
            self.reference,
            self.config,
        )
        record = result.to_records()[0]

        self.assertEqual(record["asset_pair"], "BTC-ETH")
        self.assertIn("reference_stress_correlation", record)
        self.assertIn("candidate_lower_tail_dependence", record)

    def test_rejects_wrong_asset_count_or_window(self):
        three_asset_paths = np.concatenate(
            (self.reference, self.reference[:, :, :1]),
            axis=2,
        )
        three_asset_evaluator = TrajectoryEvaluator(assets=("A", "B", "C"))
        with self.assertRaisesRegex(ValueError, "exactamente 2"):
            three_asset_evaluator.evaluate_cross_asset_dependence(
                three_asset_paths,
                three_asset_paths,
            )

        with self.assertRaisesRegex(ValueError, "rolling_window"):
            self.evaluator.evaluate_cross_asset_dependence(
                self.reference,
                self.reference,
                CrossAssetDependenceConfig(rolling_window=31),
            )


if __name__ == "__main__":
    unittest.main()
