import unittest

import numpy as np

from crypto_generative.evaluation import (
    TemporalDependenceConfig,
    TrajectoryEvaluator,
)


class TrajectoryEvaluatorTemporalTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = TrajectoryEvaluator()
        self.config = TemporalDependenceConfig(
            max_lag=5,
            volatility_window=5,
            high_volatility_quantile=0.80,
            extreme_quantile=0.90,
            extreme_clustering_window=2,
        )
        rng = np.random.default_rng(42)
        innovations = rng.normal(scale=0.01, size=(80, 40, 2))
        self.reference = innovations.copy()
        for step in range(1, self.reference.shape[1]):
            self.reference[:, step] += 0.75 * self.reference[:, step - 1]

    def test_identical_batches_have_zero_temporal_errors(self):
        result = self.evaluator.evaluate_temporal_dependence(
            self.reference,
            self.reference.copy(),
            self.config,
        )

        for asset in ("BTC", "ETH"):
            evaluation = result.by_asset[asset]
            self.assertAlmostEqual(evaluation.return_acf_rmse, 0.0)
            self.assertAlmostEqual(evaluation.absolute_return_acf_rmse, 0.0)
            self.assertAlmostEqual(evaluation.squared_return_acf_rmse, 0.0)
            self.assertAlmostEqual(
                evaluation.volatility_persistence_absolute_error,
                0.0,
            )
            self.assertAlmostEqual(
                evaluation.mean_high_volatility_run_length_absolute_error,
                0.0,
            )

    def test_time_shuffle_destroys_autocorrelation_without_changing_marginals(self):
        rng = np.random.default_rng(7)
        shuffled = self.reference.copy()
        for path in shuffled:
            rng.shuffle(path, axis=0)

        result = self.evaluator.evaluate_temporal_dependence(
            self.reference,
            shuffled,
            self.config,
        )

        for asset in ("BTC", "ETH"):
            evaluation = result.by_asset[asset]
            self.assertGreater(evaluation.return_acf_rmse, 0.1)
            self.assertGreater(evaluation.volatility_persistence_absolute_error, 0)

    def test_thresholds_are_fitted_only_on_reference(self):
        amplified = self.reference * 10
        result = self.evaluator.evaluate_temporal_dependence(
            self.reference,
            amplified,
            self.config,
        )

        for asset in ("BTC", "ETH"):
            evaluation = result.by_asset[asset]
            self.assertGreater(
                evaluation.candidate.high_volatility_frequency,
                evaluation.reference.high_volatility_frequency,
            )

    def test_acf_never_joins_two_paths(self):
        paths = np.ones((2, 8, 2), dtype=float)
        paths[1] = -1
        config = TemporalDependenceConfig(
            max_lag=2,
            volatility_window=2,
            extreme_clustering_window=1,
        )

        result = self.evaluator.evaluate_temporal_dependence(paths, paths, config)

        np.testing.assert_allclose(result.by_asset["BTC"].reference.return_acf, [1, 1])

    def test_records_and_config_are_serializable(self):
        result = self.evaluator.evaluate_temporal_dependence(
            self.reference,
            self.reference,
            self.config,
        )

        records = result.to_records()
        self.assertEqual([record["asset"] for record in records], ["BTC", "ETH"])
        self.assertIn("absolute_return_acf_rmse", records[0])
        self.assertEqual(result.to_dict()["config"]["max_lag"], 5)

    def test_rejects_config_incompatible_with_horizon(self):
        invalid_config = TemporalDependenceConfig(max_lag=40)

        with self.assertRaisesRegex(ValueError, "max_lag"):
            self.evaluator.evaluate_temporal_dependence(
                self.reference,
                self.reference,
                invalid_config,
            )


if __name__ == "__main__":
    unittest.main()
