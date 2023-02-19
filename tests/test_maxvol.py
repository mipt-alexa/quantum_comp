import numpy as np
import unittest
from maxvol import *
import itertools
import timeit

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestMaxVol(unittest.TestCase):

    def test_advanced_naive_maxvol(self, tol=1e-3):
        for _ in range(50):
            matrix = jnp.array(np.random.randn(100, 10))

            naive_vol = abs(jnp.linalg.det(matrix[naive_maxvol(matrix)]))
            adv_vol = abs(jnp.linalg.det(matrix[maxvol(matrix)]))

            self.assertTrue(np.allclose(naive_vol, adv_vol, 0.2*naive_vol))

    def test_max_volume(self, tol: float = 1e-4):
        for n, r in [(6, 2), (10, 4), (15, 5), (20, 3)]:
            matrix = jnp.array(np.random.randn(n, r))

            res_submatrix = matrix[maxvol(matrix, tol)]
            res_volume = abs(jnp.linalg.det(res_submatrix))

            true_max_vol = 0
            # generate all indexes combinations
            for ind_comb in itertools.combinations(range(n), r):
                curr_vol = abs(jnp.linalg.det(matrix[jnp.array(ind_comb)]))

                if curr_vol > true_max_vol:
                    true_max_vol = curr_vol

            if res_volume < 0.6*true_max_vol:
                raise Exception("Low accuracy")

            if res_volume < true_max_vol / r**2:
                raise Exception("Error estimation do not satisfied")


def test_execution_time(matrix_shapes: list) -> None:
    for shape in matrix_shapes:
        matrix = jnp.array(np.random.randn(shape[0], shape[1]))

        t_0 = timeit.default_timer()
        naive_maxvol(matrix)
        t_1 = timeit.default_timer()
        maxvol(matrix)
        t_2 = timeit.default_timer()

        print("naive:", round(t_1 - t_0, 3), "advanced:", round(t_2 - t_1, 3),
              "sec for matrix", shape)


test_execution_time([[100, 10], [1000, 100], [1000, 10], [10000, 10], [10000, 100]])
unittest.main(argv=[''], verbosity=1, exit=True)
