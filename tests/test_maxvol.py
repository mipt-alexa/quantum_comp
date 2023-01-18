import numpy as np
import unittest
from maxvol import *
import itertools
import timeit

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class TestMaxVol(unittest.TestCase):
    def test_max_volume(self):

        for n, r in [(4, 2), (10, 4), (15, 5), (40, 3)]:
            matrix = np.random.randn(n, r)

            res_submatrix = maxvol_SWM(matrix, 1e-4)
            res_volume = abs(jnp.linalg.det(res_submatrix))

            # generate all indexes combinations
            for ind_comb in itertools.combinations(range(n), r):
                if abs(jnp.linalg.det(matrix[list(ind_comb)])) - res_volume > 1e-4:

                    result_ind = []
                    for i in range(len(matrix)):
                        if matrix[i] in res_submatrix:
                            result_ind.append(i)
                    result_ind.sort()

                    print(res_volume, jnp.linalg.det(matrix[list(ind_comb)]))
                    print(result_ind, ind_comb)

                    raise Exception("Wrong solution")


def test_execution_time(matrix_shapes: list) -> None:
    for shape in matrix_shapes:
        matrix = np.random.randn(shape[0], shape[1])

        t_0 = timeit.default_timer()
        naive_maxvol(matrix)
        t_1 = timeit.default_timer()
        maxvol_SWM(matrix)
        t_2 = timeit.default_timer()

        print("naive:", round(t_1 - t_0, 3), "SWM:", round(t_2 - t_1, 3), "sec for matrix", shape)


test_execution_time([[100, 10], [1000, 100], [1000, 10], [10000, 100], [10000, 1000], [20000, 1000]])
unittest.main(argv=[''], verbosity=1, exit=True)
