import unittest
import jax
from jax.numpy.linalg import det
from TT_decomposition import *
import itertools
from scipy.special import binom


import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestTensorTrain(unittest.TestCase):

    def test_row_column_algorithm(self, n: int = 8, r: int = 3, alg_tol: float = 1e-4):

        for _ in range(10):
            key = jax.random.PRNGKey(42)
            A = jax.random.normal(key, shape=(n, n))

            I, J = row_column_alternating(A, r, alg_tol)
            res_volume = abs(det(A[I][:, J]))

            num_err = 0
            num_trials = binom(n, r)**2

            for row_comb in itertools.combinations(range(n), r):
                for column_comb in itertools.combinations(range(n), r):
                    volume = abs(det(A[jnp.array(row_comb)][:, jnp.array(column_comb)]))

                    if volume > 1.05*res_volume:
                        num_err += 1

            print(num_err / num_trials)
            self.assertLess(num_err / num_trials, 0.05)


unittest.main(argv=[''], verbosity=1, exit=True)
