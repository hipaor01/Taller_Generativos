import tempfile
import unittest
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from crypto_generative.generation import (
    MassiveConditionalScenarioGenerator,
    MassiveGenerationConfig,
)


@dataclass(frozen=True)
class FakeConfig:
    trajectory_length: int = 4
    n_assets: int = 2
    condition_dim: int = 3


class FakeSampler:
    config = FakeConfig()

    def sample(self, n, cond, *, seed=None):
        rng = np.random.default_rng(seed)
        return rng.normal(size=(n, 4, 2)).astype(np.float32)


class MassiveGenerationTests(unittest.TestCase):
    def test_generates_memmap_artifact_and_exact_summary(self):
        service = MassiveConditionalScenarioGenerator(
            return_mean=np.asarray([0.001, -0.001]),
            return_scale=np.asarray([0.01, 0.02]),
        )
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "scenarios.npy"
            result = service.generate(
                "fake",
                FakeSampler(),
                np.zeros((1, 3), dtype=np.float32),
                output,
                config=MassiveGenerationConfig(
                    scenario_count=11,
                    batch_size=4,
                    seed=7,
                ),
            )
            stored = np.load(output, mmap_mode="r")

            self.assertEqual(stored.shape, (11, 4, 2))
            self.assertEqual(stored.dtype, np.float32)
            self.assertTrue(np.isfinite(stored).all())
            self.assertEqual(result.summary.scenario_count, 11)
            self.assertEqual(result.summary.horizon_steps, 4)
            self.assertFalse(output.with_suffix(".npy.tmp").exists())


if __name__ == "__main__":
    unittest.main()
