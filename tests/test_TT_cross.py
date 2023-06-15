from TT_cross import TT_cross
from classes import MPS

import unittest
import jax.numpy as jnp
import jax
import numpy as np
import timeit

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

    def get_element(self, ind_arr):

        values = []
        for j in range(len(ind_arr)):
            element = self.mps.components[0][:, ind_arr[j][0], :]
            for i in range(1, len(self.out_dims)):
                element = jnp.tensordot(element, self.mps.components[i][:, ind_arr[j][i], :], 1)
            values.append(float(element))
        return jnp.array(values)


class TestTTCrossSanity(unittest.TestCase):
    def test_sanity(self):
        for size in range(10, 111, 10):
            out_dims = [4] * size
            in_dims = [2] * (size - 1)

            t_0 = timeit.default_timer()
            T = MyTensor(out_dims, in_dims)
            t_1 = timeit.default_timer()
            new_mps, update_time, linalg_operation_time, set_time, maxvol_time, reshape_time, ind_update_time = TT_cross(T.get_element, out_dims, in_dims, tol=1e-3, max_iter=6)
            t_2 = timeit.default_timer()

            print((T.mps.norm() - new_mps.norm()) / T.mps.norm())

            print("err =", np.sqrt(T.mps.norm()**2 + new_mps.norm()**2 - 2*T.mps.dot(new_mps)))

            print("for mps_len", size, "\nTensor initialization", round(t_1 - t_0, 3), "\nTT cross performing", round(t_2 - t_1, 3))

            print(update_time, linalg_operation_time, set_time, maxvol_time, reshape_time, ind_update_time)
            self.assertLess((T.mps.norm() - new_mps.norm()) / T.mps.norm(), 1e-5)

            delta = 0
            for j in range(100):
                rand_multiind = jnp.array(np.random.choice(range(out_dims[0]), size, replace=True))

                element0 = T.mps.components[0][:, rand_multiind[0], :]
                for i in range(1, size):
                    element0 = jnp.tensordot(element0, T.mps.components[i][:, rand_multiind[i], :], 1)

                element1 = new_mps.components[0][:, rand_multiind[0], :]
                for i in range(1, size):
                    element1 = jnp.tensordot(element1, new_mps.components[i][:, rand_multiind[i], :], 1)

                delta += abs(float(element0) - float(element1)) / float(element0)
            print("rel elem err", delta/100)


unittest.main(argv=[''], verbosity=1, exit=True)
