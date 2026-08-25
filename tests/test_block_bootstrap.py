import unittest

import numpy as np

from crypto_generative.models import (
    BlockBootstrapConfig,
    ConditionalBlockBootstrapConfig,
    ConditionalMultivariateBlockBootstrap,
    MultivariateBlockBootstrap,
    frozen_conditional_bootstrap_config,
)


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


class ConditionalMultivariateBlockBootstrapTest(unittest.TestCase):
    def test_frozen_configuration_keeps_validation_selection(self):
        config = frozen_conditional_bootstrap_config(
            random_state=99, horizon_steps=24
        )
        self.assertEqual(config.block_length, 12)
        self.assertEqual(config.n_neighbors, 128)
        self.assertEqual(config.horizon_steps, 24)
        self.assertEqual(config.random_state, 99)

    def test_blocks_are_aligned_with_their_initial_condition(self):
        paths = np.arange(3 * 5 * 2, dtype=float).reshape(3, 5, 2)
        conditions = np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]])
        model = ConditionalMultivariateBlockBootstrap(
            ConditionalBlockBootstrapConfig(block_length=2, n_neighbors=2)
        ).fit(paths, conditions)

        np.testing.assert_array_equal(model.blocks, paths[:, :2])
        np.testing.assert_array_equal(model.training_conditions, conditions)

    def test_condition_selects_the_nearest_joint_block(self):
        paths = np.zeros((2, 3, 2), dtype=float)
        paths[0] = np.asarray([[-1.0, -10.0]] * 3)
        paths[1] = np.asarray([[2.0, 20.0]] * 3)
        conditions = np.asarray([[0.0, 0.0], [10.0, 10.0]])
        model = ConditionalMultivariateBlockBootstrap(
            ConditionalBlockBootstrapConfig(
                block_length=2,
                horizon_steps=5,
                n_neighbors=1,
                random_state=7,
            )
        ).fit(paths, conditions)

        low = model.sample(4, np.asarray([0.1, 0.1]))
        high = model.sample(4, np.asarray([9.9, 9.9]))

        self.assertEqual(low.shape, (4, 5, 2))
        np.testing.assert_array_equal(low[..., 1], low[..., 0] * 10)
        np.testing.assert_array_equal(low, np.asarray([[[-1.0, -10.0]] * 5] * 4))
        np.testing.assert_array_equal(high, np.asarray([[[2.0, 20.0]] * 5] * 4))

    def test_sampling_is_reproducible_with_one_condition_per_path(self):
        paths = np.arange(8 * 4 * 2, dtype=float).reshape(8, 4, 2)
        conditions = np.arange(16, dtype=float).reshape(8, 2)
        model = ConditionalMultivariateBlockBootstrap(
            block_length=2,
            n_neighbors=3,
            random_state=11,
        ).fit(paths, conditions)
        query = conditions[[0, 2, 4, 6]]

        first = model.sample(4, query, horizon_steps=3)
        second = model.sample(n_scenarios=4, cond=query, horizon_steps=3)
        np.testing.assert_array_equal(first, second)

    def test_rejects_missing_or_misaligned_condition(self):
        paths = np.ones((4, 3, 2))
        conditions = np.ones((4, 2))
        model = ConditionalMultivariateBlockBootstrap(block_length=2)
        with self.assertRaisesRegex(ValueError, "alineados"):
            model.fit(paths, conditions[:3])
        model.fit(paths, conditions)
        with self.assertRaisesRegex(ValueError, "obligatorio"):
            model.sample(2)
        with self.assertRaisesRegex(ValueError, "exactamente n"):
            model.sample(2, np.ones((3, 2)))


if __name__ == "__main__":
    unittest.main()
