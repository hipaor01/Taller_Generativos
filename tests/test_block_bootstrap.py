import unittest

import numpy as np

from crypto_generative.models import BlockBootstrapConfig, MultivariateBlockBootstrap


class MultivariateBlockBootstrapTest(unittest.TestCase):
    def test_blocks_never_cross_segments_and_assets_remain_paired(self):
        returns = np.asarray(
            [
                [1.0, 10.0],
                [2.0, 20.0],
                [3.0, 30.0],
                [100.0, 1000.0],
                [200.0, 2000.0],
                [300.0, 3000.0],
            ]
        )
        segments = np.asarray([0, 0, 0, 1, 1, 1])
        model = MultivariateBlockBootstrap(
            BlockBootstrapConfig(block_length=2, horizon_steps=4, random_state=7)
        ).fit(returns, segments)

        self.assertEqual(model.blocks.shape, (4, 2, 2))
        np.testing.assert_allclose(model.blocks[:, :, 1], model.blocks[:, :, 0] * 10)
        forbidden = np.asarray([[3.0, 30.0], [100.0, 1000.0]])
        self.assertFalse(np.any(np.all(model.blocks == forbidden, axis=(1, 2))))

    def test_sampling_is_reproducible_and_uses_common_shape(self):
        returns = np.column_stack((np.arange(10), np.arange(10) * 2.0))
        segments = np.zeros(10, dtype=int)
        model = MultivariateBlockBootstrap(
            BlockBootstrapConfig(block_length=3, horizon_steps=5, random_state=11)
        ).fit(returns, segments)

        first = model.sample(8, cond=None)
        second = model.sample(8, cond=np.ones(2))
        self.assertEqual(first.shape, (8, 5, 2))
        np.testing.assert_array_equal(first, second)

    def test_rejects_invalid_inputs(self):
        model = MultivariateBlockBootstrap(BlockBootstrapConfig(block_length=3))
        with self.assertRaisesRegex(ValueError, "alineados"):
            model.fit(np.ones((4, 2)), np.ones(3))
        with self.assertRaisesRegex(RuntimeError, "fit"):
            model.sample(2)


if __name__ == "__main__":
    unittest.main()

