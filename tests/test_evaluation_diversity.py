import unittest

import numpy as np

from crypto_generative.evaluation import (
    DiversityMemorizationConfig,
    TrajectoryEvaluator,
)


class TrajectoryEvaluatorDiversityTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = TrajectoryEvaluator()
        self.config = DiversityMemorizationConfig(
            max_paths_per_set=100,
            projection_dimensions=8,
            neighbor_candidates=4,
            discriminator_repetitions=3,
            discriminator_iterations=200,
            random_state=17,
        )
        rng = np.random.default_rng(42)
        self.reference = rng.normal(scale=0.01, size=(80, 20, 2))

    def test_identical_candidate_has_full_coverage_and_training_matches(self):
        result = self.evaluator.evaluate_diversity_and_memorization(
            self.reference,
            self.reference.copy(),
            training_paths=self.reference,
            config=self.config,
        )

        self.assertAlmostEqual(result.candidate_unique_fraction, 1.0)
        self.assertAlmostEqual(result.exact_training_match_fraction, 1.0)
        self.assertAlmostEqual(result.reference_coverage_fraction, 1.0)
        self.assertAlmostEqual(result.regime_total_variation_distance, 0.0)
        self.assertGreater(result.discriminator_accuracy_mean, 0.35)
        self.assertLess(result.discriminator_accuracy_mean, 0.65)

    def test_collapsed_candidate_is_detected_as_redundant(self):
        collapsed = np.repeat(self.reference[:1], len(self.reference), axis=0)
        result = self.evaluator.evaluate_diversity_and_memorization(
            self.reference,
            collapsed,
            training_paths=self.reference,
            config=self.config,
        )

        self.assertAlmostEqual(result.candidate_unique_fraction, 1 / len(collapsed))
        self.assertGreater(result.candidate_redundant_fraction, 0.98)
        self.assertAlmostEqual(result.candidate_near_duplicate_fraction, 1.0)
        self.assertAlmostEqual(
            result.candidate_nearest_neighbor_distance.mean,
            0.0,
        )

    def test_shifted_candidate_has_low_coverage_and_is_discriminable(self):
        shifted = self.reference + 0.10
        result = self.evaluator.evaluate_diversity_and_memorization(
            self.reference,
            shifted,
            training_paths=self.reference,
            config=self.config,
        )

        self.assertAlmostEqual(result.exact_training_match_fraction, 0.0)
        self.assertAlmostEqual(result.reference_coverage_fraction, 0.0)
        self.assertGreater(result.discriminator_accuracy_mean, 0.95)

    def test_four_dimensional_candidate_is_flattened(self):
        candidate = self.reference[:10].reshape(5, 2, 20, 2)
        result = self.evaluator.evaluate_diversity_and_memorization(
            self.reference,
            candidate,
            training_paths=self.reference,
            config=self.config,
        )

        self.assertEqual(result.total_candidate_paths, 10)
        self.assertEqual(result.evaluated_candidate_paths, 10)
        self.assertAlmostEqual(result.exact_training_match_fraction, 1.0)

    def test_training_is_optional_but_memorization_fields_are_empty(self):
        result = self.evaluator.evaluate_diversity_and_memorization(
            self.reference,
            self.reference,
            config=self.config,
        )

        self.assertIsNone(result.evaluated_training_paths)
        self.assertIsNone(result.exact_training_match_fraction)
        self.assertIsNone(result.near_memorization_fraction)
        self.assertIsNone(result.candidate_to_training_distance)

    def test_records_are_ready_for_tabular_consumption(self):
        result = self.evaluator.evaluate_diversity_and_memorization(
            self.reference,
            self.reference,
            training_paths=self.reference,
            config=self.config,
        )
        record = result.to_records()[0]

        self.assertIn("candidate_neighbor_q50", record)
        self.assertIn("candidate_to_training_q01", record)
        self.assertIn("reference_regime_high", record)

    def test_rejects_invalid_configuration(self):
        with self.assertRaisesRegex(ValueError, "projection_dimensions"):
            self.evaluator.evaluate_diversity_and_memorization(
                self.reference,
                self.reference,
                config=DiversityMemorizationConfig(projection_dimensions=1),
            )


if __name__ == "__main__":
    unittest.main()
