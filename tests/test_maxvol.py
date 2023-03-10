import unittest

import numpy as np
from maxvol import *
import jax.random
from jax.scipy.linalg import lu

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestMaxVol(unittest.TestCase):
    def test_dominant(self, n: int = 100, r: int = 10, tol: float = 1e-3):
        seed = 0
        for tol in np.logspace(-4, -1, num=4):
            print("maxvol tolerance ", tol)
            for _ in range(20):
                key = jax.random.PRNGKey(seed)
                seed += 1
                matrix = jax.random.uniform(key, shape=(n, r))
                rows = maxvol(matrix, tol)
                res_submatrix = matrix[rows]

                P, _, _ = lu(matrix)

                product = jnp.tensordot(jnp.tensordot(jnp.transpose(P), matrix, 1), jnp.linalg.inv(res_submatrix), 1)
                i, j = arg_absmax(product)

                self.assertLess(abs(product[i][j]), 1 + tol)


unittest.main(argv=[''], verbosity=1, exit=True)
