import unittest

import numpy as np

from crypto_generative.evaluation import TrajectoryEvaluator


class TrajectoryEvaluatorMarginalTest(unittest.TestCase):
    def setUp(self):
        self.reference = np.asarray(
            [
                [[-0.02, -0.04], [0.00, 0.01], [0.02, 0.03]],
                [[-0.01, -0.02], [0.01, 0.02], [0.03, 0.05]],
            ],
            dtype=float,
        )
        self.evaluator = TrajectoryEvaluator()

    def test_identical_batches_have_zero_wasserstein(self):
        result = self.evaluator.evaluate_marginals(self.reference, self.reference.copy())

        for asset in ("BTC", "ETH"):
            evaluation = result.by_asset[asset]
            self.assertAlmostEqual(evaluation.wasserstein_1, 0.0)
            self.assertAlmostEqual(evaluation.normalized_wasserstein_1, 0.0)
            self.assertEqual(
                evaluation.reference.quantiles,
                evaluation.candidate.quantiles,
            )

    def test_constant_shift_is_detected_in_original_units(self):
        shifted = self.reference + 0.01
        result = self.evaluator.evaluate_marginals(self.reference, shifted)

        for asset in ("BTC", "ETH"):
            self.assertAlmostEqual(result.by_asset[asset].wasserstein_1, 0.01)
            self.assertGreater(result.by_asset[asset].normalized_wasserstein_1, 0)

    def test_extreme_threshold_is_always_fitted_on_reference(self):
        candidate = self.reference * 10
        result = self.evaluator.evaluate_marginals(self.reference, candidate)

        for asset in ("BTC", "ETH"):
            evaluation = result.by_asset[asset]
            self.assertGreater(
                evaluation.candidate.extreme_frequency,
                evaluation.reference.extreme_frequency,
            )

    def test_records_are_ready_for_tabular_consumption(self):
        records = self.evaluator.evaluate_marginals(
            self.reference,
            self.reference,
        ).to_records()

        self.assertEqual([record["asset"] for record in records], ["BTC", "ETH"])
        self.assertIn("reference_q01", records[0])
        self.assertIn("candidate_q99", records[0])

    def test_rejects_invalid_or_incompatible_batches(self):
        with self.assertRaisesRegex(ValueError, "forma"):
            self.evaluator.evaluate_marginals(self.reference[0], self.reference)
        with self.assertRaisesRegex(ValueError, "horizonte"):
            self.evaluator.evaluate_marginals(self.reference, self.reference[:, :-1])

        non_finite = self.reference.copy()
        non_finite[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "no finitos"):
            self.evaluator.evaluate_marginals(self.reference, non_finite)

    def test_zero_reference_variance_has_no_normalized_distance(self):
        reference = np.zeros((2, 3, 2))
        candidate = np.ones((2, 3, 2))

        result = self.evaluator.evaluate_marginals(reference, candidate)

        self.assertIsNone(result.by_asset["BTC"].normalized_wasserstein_1)
        self.assertGreater(result.by_asset["BTC"].wasserstein_1, 0)


if __name__ == "__main__":
    unittest.main()
