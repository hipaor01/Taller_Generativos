import unittest
import tempfile
from pathlib import Path

import numpy as np

from crypto_generative.data import FrozenPathBatch, ProjectScenarioLoader
from crypto_generative.portfolio import (
    BuyAndHoldPortfolio,
    PortfolioConfig,
    PortfolioScenarioAccumulator,
    PortfolioStressApplication,
    ScenarioCategory,
    StressScenarioSet,
    build_joint_shock_path,
    default_prefixed_scenarios,
)


class BuyAndHoldPortfolioTest(unittest.TestCase):
    def setUp(self):
        self.portfolio = BuyAndHoldPortfolio(
            PortfolioConfig(initial_value=100_000.0, weights=(0.60, 0.40))
        )

    def test_revalues_without_rebalancing_and_computes_path_losses(self):
        paths = np.asarray(
            [
                [
                    [np.log(1.20), 0.0],
                    [np.log(0.50), 0.0],
                ]
            ]
        )
        result = self.portfolio.revalue(paths)

        np.testing.assert_allclose(result.values[0], [100_000, 112_000, 76_000])
        self.assertAlmostEqual(result.final_loss_amounts[0], 24_000.0)
        self.assertAlmostEqual(result.maximum_loss_amounts[0], 24_000.0)
        self.assertAlmostEqual(result.maximum_drawdowns[0], 1.0 - 76_000 / 112_000)

    def test_four_dimensional_conditional_paths_are_flattened(self):
        paths = np.zeros((3, 4, 5, 2), dtype=float)
        result = self.portfolio.revalue(paths)
        self.assertEqual(result.values.shape, (12, 6))
        np.testing.assert_allclose(result.final_values, 100_000.0)

    def test_stress_summary_reports_amounts_var_es_and_worst_labels(self):
        losses = np.asarray([0.10, 0.20, 0.30, 0.40])
        terminal = np.log(1.0 - losses)
        paths = np.repeat(terminal[:, None, None], 2, axis=2)
        scenarios = StressScenarioSet(
            name="test",
            category=ScenarioCategory.HISTORICAL,
            log_returns=paths,
            labels=("a", "b", "c", "d"),
        )
        summary = self.portfolio.summarize(scenarios)

        self.assertEqual(summary.worst_final_loss_label, "d")
        self.assertAlmostEqual(summary.final_loss_amount.maximum, 40_000.0)
        self.assertGreaterEqual(
            summary.risk["0.95"].expected_shortfall_fraction,
            summary.risk["0.95"].value_at_risk_fraction,
        )

    def test_selects_historical_paths_by_requested_criterion(self):
        paths = np.zeros((3, 2, 2), dtype=float)
        paths[0, -1] = np.log([0.90, 0.90])
        paths[1, -1] = np.log([0.50, 0.50])
        paths[2, -1] = np.log([0.75, 0.75])
        indices = self.portfolio.select_stress_paths(
            paths, 2, criterion="maximum_drawdown"
        )
        np.testing.assert_array_equal(indices, [1, 2])

    def test_application_rejects_duplicate_names(self):
        scenario = StressScenarioSet(
            name="duplicate",
            category=ScenarioCategory.PREFIXED,
            log_returns=np.zeros((1, 2, 2)),
        )
        with self.assertRaisesRegex(ValueError, "únicos"):
            PortfolioStressApplication(self.portfolio).run([scenario, scenario])

    def test_streaming_accumulator_matches_regular_summary(self):
        rng = np.random.default_rng(9)
        paths = rng.normal(0.0, 0.01, size=(17, 8, 2))
        scenario = StressScenarioSet(
            name="generated",
            category=ScenarioCategory.GENERATIVE,
            log_returns=paths,
        )
        expected = self.portfolio.summarize(scenario)
        accumulator = PortfolioScenarioAccumulator(
            self.portfolio,
            name="generated",
            category=ScenarioCategory.GENERATIVE,
            scenario_count=len(paths),
            horizon_steps=paths.shape[1],
        )
        accumulator.add(paths[:6])
        accumulator.add(paths[6:])

        self.assertEqual(accumulator.finalize().to_dict(), expected.to_dict())

    def test_prefixed_crash_and_recovery_have_expected_path_behavior(self):
        crash = build_joint_shock_path(
            (-0.30, -0.40), horizon_steps=120, shock_steps=20
        )
        recovered = build_joint_shock_path(
            (-0.30, -0.40),
            horizon_steps=120,
            shock_steps=20,
            recovery_fraction=0.50,
        )
        crash_values = self.portfolio.revalue(crash[None]).values[0]
        recovered_values = self.portfolio.revalue(recovered[None]).values[0]

        self.assertAlmostEqual(crash_values[-1], 66_000.0)
        self.assertGreater(recovered_values[-1], crash_values[-1])
        self.assertAlmostEqual(recovered_values.min(), 66_000.0)
        self.assertEqual(len(default_prefixed_scenarios()), 3)


class ProjectScenarioLoaderTest(unittest.TestCase):
    def test_loads_normalized_conditions_in_requested_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            np.savez_compressed(
                root / "normalized.npz",
                sample_ids=np.asarray([10, 11, 12]),
                condition_features=np.asarray(
                    [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]
                ),
            )
            loader = ProjectScenarioLoader(
                root / "normalized.npz",
                root / "split.npz",
                root / "index.csv",
                root / "panel.csv",
            )
            selected = loader.load_normalized_conditions(np.asarray([12, 10]))

        np.testing.assert_array_equal(selected, [[5.0, 6.0], [1.0, 2.0]])

    def test_loads_shared_conditional_artifact_and_checks_reference(self):
        real = np.zeros((2, 4, 2), dtype=float)
        conditional = np.ones((2, 3, 4, 2), dtype=float) * 0.01
        reference = FrozenPathBatch(
            split="test",
            sample_ids=np.asarray([10, 11]),
            log_returns=real,
            labels=("first", "second"),
            assets=("BTC", "ETH"),
        )
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = root / "scenarios.npz"
            np.savez_compressed(
                artifact,
                generated_conditional_returns=conditional,
                real_returns=real,
                assets=np.asarray(["BTC", "ETH"]),
            )
            loader = ProjectScenarioLoader(
                root / "normalized.npz",
                root / "split.npz",
                root / "index.csv",
                root / "panel.csv",
            )
            scenarios = loader.load_generated_scenarios(
                "model", artifact, expected_reference=reference
            )

        self.assertEqual(scenarios.category, ScenarioCategory.GENERATIVE)
        self.assertEqual(scenarios.log_returns.shape, (6, 4, 2))


if __name__ == "__main__":
    unittest.main()
