import unittest
from numpy.random import randn, uniform, choice

from TT_decomposition import *
from helper_functions import *

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestTensorTrain(unittest.TestCase):

    def test_inner_dims(self, max_mps_len: int = 6, max_outer_dim: int = 10):
        for i in range(2, max_mps_len):

            outer_dims = list(choice(range(2, max_outer_dim), size=i, replace=True))
            inner_dim = int((min(outer_dims) + 2) / 2)

            inner_dims = list(np.full(i-1, inner_dim))

            mps_elements = [jnp.array(uniform(size=(1, outer_dims[0], inner_dim)))]

            for j in range(2, i):
                mps_elements.append(jnp.array(uniform(size=(inner_dim, outer_dims[j-1], inner_dim))))

            mps_elements.append(jnp.array(uniform(size=(inner_dim, outer_dims[-1], 1))))

            init_mps = MPS(mps_elements)
            tensor = get_tensor_from_MPS(init_mps)
            res_mps = TT_decomposition(jnp.reshape(tensor, outer_dims), inner_dims)

            for k in range(len(outer_dims) - 1):
                self.assertEqual(res_mps.components[k].shape[2], inner_dim)

    def test_decompose_with_known_rank(self, max_mps_len: int = 6, outer_dim: int = 8, inner_dim: int = 3):
        for i in range(2, max_mps_len):
            mps_elements = [jnp.array(randn(1, outer_dim, inner_dim))]

            for j in range(2, i):
                mps_elements.append(jnp.array(randn(inner_dim, outer_dim, inner_dim)))

            mps_elements.append(jnp.array(randn(inner_dim, outer_dim, 1)))
            init_mps = MPS(mps_elements)

            inner_dims = list(np.full(i - 1, inner_dim))
            outer_dims = list(np.full(i, outer_dim))

            tensor = get_tensor_from_MPS(init_mps)
            res_mps = TT_decomposition(jnp.reshape(tensor, outer_dims), inner_dims)

            init_norm = get_norm_from_MPS(init_mps)
            res_norm = get_norm_from_MPS(res_mps)

            self.assertTrue(np.allclose(init_norm / res_norm, 1, 1e-6))


unittest.main(argv=[''], verbosity=1, exit=True)
