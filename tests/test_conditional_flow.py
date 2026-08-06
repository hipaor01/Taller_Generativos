import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from crypto_generative.models import (
    ConditionalFlowConfig,
    ConditionalFlowGenerator,
    ConditionalRealNVP,
)


class ConditionalFlowTests(unittest.TestCase):
    def setUp(self):
        self.config = ConditionalFlowConfig(
            trajectory_length=4,
            n_assets=2,
            condition_dim=3,
            n_coupling_layers=4,
            hidden_dim=16,
            n_hidden_layers=1,
            batch_size=8,
            max_epochs=2,
            patience=2,
            seed=7,
        )

    def test_forward_inverse_are_consistent(self):
        model = ConditionalRealNVP(self.config)
        x = torch.randn(5, self.config.data_dim)
        c = torch.randn(5, self.config.condition_dim)
        z, forward_log_det = model.transform_to_base(x, c)
        recovered, inverse_log_det = model.transform_from_base(z, c)
        self.assertTrue(torch.allclose(x, recovered, atol=1e-5, rtol=1e-5))
        self.assertTrue(
            torch.allclose(forward_log_det, -inverse_log_det, atol=1e-5, rtol=1e-5)
        )

    def test_wrapper_fit_sample_and_checkpoint(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(40, 4, 2)).astype(np.float32)
        c = rng.normal(size=(40, 3)).astype(np.float32)
        generator = ConditionalFlowGenerator(self.config, device="cpu")
        history = generator.fit(
            x[:30], c[:30], X_validation=x[30:], cond_validation=c[30:], verbose=False
        )
        self.assertGreaterEqual(len(history.train_nll), 1)
        samples = generator.sample(6, c[0])
        self.assertEqual(samples.shape, (6, 4, 2))
        self.assertTrue(np.isfinite(samples).all())
        scores = generator.log_prob(x[30:], c[30:])
        self.assertEqual(scores.shape, (10,))

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "flow.pt"
            generator.save(path)
            loaded = ConditionalFlowGenerator.load(path, device="cpu")
            loaded_scores = loaded.log_prob(x[30:], c[30:])
            np.testing.assert_allclose(scores, loaded_scores, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
