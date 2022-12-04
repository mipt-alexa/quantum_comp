from classes import *
import unittest
import numpy as np


class Test(unittest.TestCase):
    def test_dot(self):
        max_dim = 6  # maximum dimension of any index
        length = 6  # length of MPS

        outer_dims = np.random.randint(1, max_dim, length)  # visible dimensions of each tensor
        inner_dims = np.random.randint(1, max_dim, length - 1)  # bond dimensions of tensor A

        comp = [np.random.randn(1, outer_dims[0], inner_dims[0])]  # filling the first components of mps
        for i in range(1, length - 1):
            comp.append(np.random.randn(inner_dims[i - 1], outer_dims[i], inner_dims[i]))
        comp.append(
            np.random.randn(inner_dims[-1], outer_dims[-1], 1))  # the last components of mps

        mps = MPS(comp)
        mps.left_canonical()

        validation = []

        for i in range(length - 1):
            shape = mps.components[i].shape
            result = jnp.tensordot(mps.components[i], mps.components[i], [[0, 1], [0, 1]])
            expected = np.diag(np.ones(shape[0]*shape[1]))
            validation.append(np.allclose(result, expected, atol=1e-06))

        self.assertTrue(np.all(validation))
        validation.clear()
        mps.right_canonical()

        for i in range(1, length):
            shape = mps.components[i].shape
            result = jnp.tensordot(mps.components[i], mps.components[i], [[1, 2], [1, 2]])
            expected = np.diag(np.ones(shape[1]*shape[2]))
            validation.append(np.allclose(result, expected, atol=1e-06))

        self.assertTrue(np.all(validation))


unittest.main(argv=[''], verbosity=2, exit=True)
