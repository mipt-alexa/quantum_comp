from TT_cross import TT_cross
from classes import MPS
from helper_functions import get_tensor_from_MPS

import unittest
import jax.numpy as jnp
import jax
import numpy as np

import copy


jax.config.update("jax_enable_x64", True)


class MyTensor:
    def __init__(self, out_dims, inner_dims):
        self.out_dims = out_dims

        self.in_dims = copy.copy(inner_dims)
        self.in_dims.insert(0, 1)
        self.in_dims.append(1)

        components = []

        for i in range(len(self.out_dims)):
            components.append(jnp.array(np.random.randn(self.in_dims[i], self.out_dims[i], self.in_dims[i + 1]), dtype=jnp.float64))

        self.mps = MPS(components)

    def get_element(self, ind):

        element = self.mps.components[0][:, ind[0], :]
        for i in range(1, len(self.out_dims)):
            element = jnp.tensordot(element, self.mps.components[i][:, ind[i], :], 1)
        return float(element)


def hilbert_tensor(ind):
    return 1/(jnp.sum(ind) + len(ind))


class TestTTCrossSanity(unittest.TestCase):
    def test_sanity(self):
        for size in (10, 20, 30):
            out_dims = [3] * size
            in_dims = [2] * (size - 1)

            T = MyTensor(out_dims, in_dims)
            new_mps = TT_cross(T.get_element, out_dims, in_dims, tol=1e-1)

            self.assertLess((T.mps.norm() -new_mps.norm()) / T.mps.norm(), 1e-6)

    # def test_hilbert_tensor(self):
    #     a = 20
    #     out_dims = [a] * 4
    #     tensor = jnp.empty(out_dims)
    #     for i0 in range(a):
    #         for i1 in range(a):
    #             for i2 in range(a):
    #                 for i3 in range(a):
    #                     tensor = tensor.at[i0, i1, i2, i3].set(hilbert_tensor(jnp.array([i0, i1, i2, i3])))
    #
    #     for in_dim in range(2, a//2):
    #         in_dims = [in_dim] * 3
    #         mps = TT_cross(hilbert_tensor, out_dims, in_dims)
    #
    #         delta = tensor - jnp.reshape(get_tensor_from_MPS(mps), out_dims)
    #         print(in_dim, jnp.sqrt(jnp.tensordot(delta, delta, 4)))


unittest.main(argv=[''], verbosity=1, exit=True)
