from helper_functions import *
import unittest
import numpy as np
import copy

MPS_LEN = 6  # number of components of MPS
MAX_DIM = 6  # maximum dimension of any mps component index


class TestTrunc(unittest.TestCase):
    def test_norm_alteration(self):
        mps = create_mps(MPS_LEN, MAX_DIM)
        init_mps = copy.deepcopy(mps)

        init_norm = get_norm_from_MPS(mps)

        mps.left_canonical()
        trunc_val = np.random.uniform(0, init_norm / 2)
        mps.left_svd_trunc(trunc_val)

        final_norm = get_norm_from_MPS(mps)
        print(init_norm, trunc_val, final_norm)

        self.assertLessEqual(final_norm, init_norm)

        init_tensor = get_tensor_from_MPS(init_mps)
        final_tensor = get_tensor_from_MPS(mps)
        diff = init_tensor - final_tensor

        self.assertLessEqual(jnp.tensordot(final_tensor, diff, mps.len + 2) - \
                             jnp.tensordot(init_tensor, diff, mps.len + 2), trunc_val ** 2)

    def test_sum_errors(self):
        mps = create_mps(MPS_LEN, MAX_DIM)
        init_norm = get_norm_from_MPS(mps)
        init_mps = copy.deepcopy(mps)

        mps.left_canonical()
        trunc_val = np.random.uniform(0, init_norm / 2)
        mps.left_svd_trunc(trunc_val)

        total_error_sq = 0
        for i in range(mps.len - 1):
            matrix_full = unfold_matrix_from_mps(init_mps, i)
            matrix_trunc = unfold_matrix_from_mps(mps, i)

            total_error_sq += jnp.linalg.norm(matrix_full - matrix_trunc) ** 2

        diff = get_tensor_from_MPS(init_mps) - get_tensor_from_MPS(mps)
        trunc_error = np.sqrt(jnp.tensordot(diff, diff, mps.len + 2))

        self.assertLessEqual(trunc_error, np.sqrt(total_error_sq))


unittest.main(argv=[''], verbosity=1, exit=True)
