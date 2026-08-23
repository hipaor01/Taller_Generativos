import unittest

from crypto_generative.comparison import (
    DownstreamComparisonBuilder,
    FinalComparisonBuilder,
)


def fake_metadata(score, coverage):
    asset_metric = {
        "BTC": {"normalized_wasserstein_1": score},
        "ETH": {"normalized_wasserstein_1": score},
    }
    temporal = {
        asset: {
            "return_acf_rmse": score,
            "absolute_return_acf_rmse": score,
            "squared_return_acf_rmse": score,
        }
        for asset in ("BTC", "ETH")
    }
    trajectory = {
        asset: {
            metric: {"normalized_wasserstein_1": score}
            for metric in (
                "final_cumulative_return",
                "realized_volatility",
                "maximum_drawdown",
            )
        }
        for asset in ("BTC", "ETH")
    }
    level = {
        "coverage_absolute_error": score,
        "unconditional_var_absolute_error": score,
    }
    return {
        "evaluation": {
            "marginal": {"assets": ["BTC", "ETH"], "by_asset": asset_metric},
            "temporal": {"by_asset": temporal},
            "trajectory": {"by_asset": trajectory},
            "cross_asset": {
                "errors": {
                    "contemporaneous_correlation_absolute_error": score,
                    "stress_correlation_absolute_error": score,
                    "lower_tail_dependence_absolute_error": score,
                }
            },
            "risk": {
                "by_target": {
                    "portfolio_60_40": {
                        "levels": {"0.95": level, "0.99": level}
                    }
                }
            },
            "diversity_and_memorization": {
                "total_candidate_paths": 10,
                "candidate_unique_fraction": 1.0,
                "near_memorization_fraction": score,
                "reference_coverage_fraction": coverage,
                "regime_total_variation_distance": score,
                "discriminator_accuracy_mean": 0.5 + score,
            },
        }
    }


class FinalComparisonBuilderTests(unittest.TestCase):
    def test_extracts_quality_and_ranks_without_global_score(self):
        builder = FinalComparisonBuilder(
            {
                "better_errors": fake_metadata(0.1, 0.8),
                "better_coverage": fake_metadata(0.2, 0.9),
            }
        )
        quality = builder.quality_records()
        rankings = builder.ranking_records(quality)

        self.assertEqual(len(quality), 2)
        self.assertNotIn("global_score", quality[0])
        by_metric = {row["metric"]: row for row in rankings}
        self.assertEqual(
            by_metric["marginal_normalized_w1_mean"]["winner"],
            "better_errors",
        )
        self.assertEqual(
            by_metric["reference_coverage_fraction"]["winner"],
            "better_coverage",
        )


class DownstreamComparisonBuilderTests(unittest.TestCase):
    @staticmethod
    def rows():
        rows = []
        for model in DownstreamComparisonBuilder.MODEL_ORDER:
            for ratio in (0.0, 0.25, 0.5, 1.0):
                rows.append(
                    {
                        "model": model,
                        "dataset": "real_only" if ratio == 0 else f"ratio_{ratio}",
                        "additional_synthetic_ratio": ratio,
                        "validation_mae": 0.1 + abs(ratio - 0.25),
                        "test_mae": 0.2 - 0.05 * ratio,
                        "test_r2": -0.5 + ratio,
                    }
                )
        return rows

    def test_selects_ratio_with_validation_and_reports_test_only_descriptively(self):
        records = DownstreamComparisonBuilder(self.rows()).selection_records()

        self.assertEqual(len(records), 4)
        self.assertTrue(
            all(row["validation_selected_ratio"] == 0.25 for row in records)
        )
        self.assertTrue(
            all(row["test_descriptive_best_dataset"] == "ratio_1.0" for row in records)
        )

    def test_rejects_incomplete_ratio_grid(self):
        with self.assertRaisesRegex(ValueError, "debe contener ratios"):
            DownstreamComparisonBuilder(self.rows()[:-1])


if __name__ == "__main__":
    unittest.main()
