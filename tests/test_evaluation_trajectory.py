import unittest

import numpy as np

from crypto_generative.evaluation import (
    TrajectoryEvaluator,
    TrajectoryMetricsConfig,
)


class TrajectoryEvaluatorPathMetricsTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = TrajectoryEvaluator()
        self.config = TrajectoryMetricsConfig(periods_per_year=4)
        wealth = np.asarray([1.10, 0.88, 0.968])
        btc_returns = np.diff(np.log(np.concatenate(([1.0], wealth))))
        eth_returns = btc_returns * 0.5
        self.reference = np.stack((btc_returns, eth_returns), axis=1)[None, :, :]

    def test_known_path_metrics_are_calculated_exactly(self):
        result = self.evaluator.evaluate_trajectories(
            self.reference,
            self.reference,
            self.config,
        )
        metrics = result.by_asset["BTC"].metrics

        self.assertAlmostEqual(metrics["final_cumulative_return"].reference.mean, -0.032)
        self.assertAlmostEqual(metrics["maximum_drawdown"].reference.mean, 0.20)
        self.assertAlmostEqual(
            metrics["maximum_drawdown_duration_steps"].reference.mean,
            2.0,
        )
        self.assertAlmostEqual(
            metrics["intrahorizon_maximum_return"].reference.mean,
            0.10,
        )
        self.assertAlmostEqual(
            metrics["intrahorizon_minimum_return"].reference.mean,
            -0.12,
        )
        self.assertAlmostEqual(
            metrics["time_to_minimum_value_steps"].reference.mean,
            2.0,
        )

    def test_identical_batches_have_zero_distances(self):
        repeated = np.repeat(self.reference, 10, axis=0)
        result = self.evaluator.evaluate_trajectories(
            repeated,
            repeated.copy(),
            self.config,
        )

        for asset in ("BTC", "ETH"):
            for evaluation in result.by_asset[asset].metrics.values():
                self.assertAlmostEqual(evaluation.wasserstein_1, 0.0)

    def test_more_volatile_candidate_is_detected(self):
        rng = np.random.default_rng(42)
        reference = rng.normal(scale=0.01, size=(100, 30, 2))
        candidate = reference * 3
        result = self.evaluator.evaluate_trajectories(reference, candidate, self.config)

        for asset in ("BTC", "ETH"):
            metrics = result.by_asset[asset].metrics
            self.assertGreater(metrics["realized_volatility"].wasserstein_1, 0)
            self.assertGreater(metrics["maximum_drawdown"].wasserstein_1, 0)

    def test_paths_are_never_concatenated(self):
        first_path = np.full((4, 2), np.log(1.01))
        second_path = np.full((4, 2), np.log(0.99))
        paths = np.stack((first_path, second_path))
        result = self.evaluator.evaluate_trajectories(paths, paths, self.config)
        final_return = result.by_asset["BTC"].metrics["final_cumulative_return"]

        expected = np.mean([1.01**4 - 1, 0.99**4 - 1])
        self.assertAlmostEqual(final_return.reference.mean, expected)

    def test_drawdown_duration_resets_after_recovery(self):
        wealth = np.asarray([1.10, 0.88, 1.20])
        returns = np.diff(np.log(np.concatenate(([1.0], wealth))))
        paths = np.repeat(returns[None, :, None], 2, axis=2)
        result = self.evaluator.evaluate_trajectories(paths, paths, self.config)

        duration = result.by_asset["BTC"].metrics[
            "maximum_drawdown_duration_steps"
        ]
        self.assertAlmostEqual(duration.reference.mean, 1.0)

    def test_records_are_long_and_tabular(self):
        result = self.evaluator.evaluate_trajectories(
            self.reference,
            self.reference,
            self.config,
        )
        records = result.to_records()

        self.assertEqual(len(records), 14)
        self.assertEqual({record["asset"] for record in records}, {"BTC", "ETH"})
        self.assertIn("reference_q01", records[0])
        self.assertIn("candidate_q99", records[0])

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "periods_per_year"):
            self.evaluator.evaluate_trajectories(
                self.reference,
                self.reference,
                TrajectoryMetricsConfig(periods_per_year=0),
            )


if __name__ == "__main__":
    unittest.main()
