import unittest

import numpy as np

from crypto_generative.evaluation import RiskMetricsConfig, TrajectoryEvaluator


def paths_from_losses(losses):
    terminal_log_return = np.log(1.0 - np.asarray(losses, dtype=float))
    returns = terminal_log_return[:, None, None]
    return np.repeat(returns, 2, axis=2)


class TrajectoryEvaluatorRiskTest(unittest.TestCase):
    def setUp(self):
        self.evaluator = TrajectoryEvaluator()
        self.config = RiskMetricsConfig(
            confidence_levels=(0.95, 0.99),
            es_stability_repetitions=20,
            es_stability_sample_size=100,
            random_state=7,
        )
        self.losses = np.linspace(-0.20, 0.40, 1_000)
        self.reference = paths_from_losses(self.losses)

    def test_identical_unconditional_forecast_has_exact_var_and_es(self):
        result = self.evaluator.evaluate_risk(
            self.reference,
            self.reference.copy(),
            self.config,
        )

        self.assertEqual(result.forecast_mode, "unconditional")
        for target in ("BTC", "ETH", "portfolio_60_40"):
            for level in result.by_target[target].levels.values():
                self.assertAlmostEqual(level.unconditional_var_absolute_error, 0.0)
                self.assertAlmostEqual(level.unconditional_es_absolute_error, 0.0)
                self.assertLessEqual(level.coverage_absolute_error, 0.001)

    def test_underestimated_risk_produces_too_many_exceptions(self):
        safer_candidate = paths_from_losses(self.losses - 0.15)
        result = self.evaluator.evaluate_risk(
            self.reference,
            safer_candidate,
            self.config,
        )
        level = result.by_target["BTC"].levels["0.95"]

        self.assertGreater(level.exception_rate, level.expected_exception_rate)
        self.assertGreater(level.coverage_absolute_error, 0.10)

    def test_portfolio_uses_buy_and_hold_weights_without_rebalancing(self):
        btc_wealth = 1.10
        eth_wealth = 0.90
        path = np.log(np.asarray([[[btc_wealth, eth_wealth]]]))
        result = self.evaluator.evaluate_risk(path, path, self.config)

        portfolio = result.by_target["portfolio_60_40"]
        self.assertAlmostEqual(portfolio.reference_losses.mean, -0.02)
        self.assertEqual(portfolio.portfolio_weights, {"BTC": 0.6, "ETH": 0.4})

    def test_conditional_draws_use_one_var_per_reference_condition(self):
        reference_losses = np.asarray([0.05, 0.15, 0.25, 0.35])
        reference = paths_from_losses(reference_losses)
        candidate_loss_draws = np.stack(
            [
                np.linspace(loss - 0.10, loss + 0.10, 200)
                for loss in reference_losses
            ]
        )
        candidate_log_returns = np.log(1.0 - candidate_loss_draws)
        candidate = np.repeat(
            candidate_log_returns[:, :, None, None],
            2,
            axis=3,
        )
        result = self.evaluator.evaluate_risk(reference, candidate, self.config)

        self.assertEqual(result.forecast_mode, "conditional")
        level = result.by_target["BTC"].levels["0.95"]
        self.assertEqual(level.exception_count, 0)
        self.assertAlmostEqual(level.mean_forecast_var, 0.29, places=2)

    def test_es_stability_is_reproducible(self):
        first = self.evaluator.evaluate_risk(
            self.reference,
            self.reference,
            self.config,
        )
        second = self.evaluator.evaluate_risk(
            self.reference,
            self.reference,
            self.config,
        )

        first_value = first.by_target["BTC"].levels[
            "0.99"
        ].candidate_es_stability_standard_deviation
        second_value = second.by_target["BTC"].levels[
            "0.99"
        ].candidate_es_stability_standard_deviation
        self.assertEqual(first_value, second_value)

    def test_records_cover_every_target_and_confidence_level(self):
        result = self.evaluator.evaluate_risk(
            self.reference,
            self.reference,
            self.config,
        )
        records = result.to_records()

        self.assertEqual(len(records), 6)
        self.assertEqual(
            {record["target"] for record in records},
            {"BTC", "ETH", "portfolio_60_40"},
        )
        self.assertIn("coverage_absolute_error", records[0])

    def test_rejects_invalid_weights_and_unaligned_conditional_draws(self):
        with self.assertRaisesRegex(ValueError, "sumar 1"):
            self.evaluator.evaluate_risk(
                self.reference,
                self.reference,
                RiskMetricsConfig(portfolio_weights=(0.5, 0.4)),
            )

        candidate = np.repeat(self.reference[:10, None], 2, axis=1)
        with self.assertRaisesRegex(ValueError, "condiciones reales"):
            self.evaluator.evaluate_risk(self.reference, candidate, self.config)


if __name__ == "__main__":
    unittest.main()
