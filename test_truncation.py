from helper_functions import *
import unittest
import numpy as np

MPS_LEN = 6  # number of components of MPS
MAX_DIM = 6  # maximum dimension of any mps component index


class TestTrunc(unittest.TestCase):
    def test_norm_conservation(self):
        mps = create_mps(MPS_LEN, MAX_DIM)

        init_norm = get_norm_from_MPS(mps)

        mps.left_canonical()
        trunc_val = np.random.uniform(0, init_norm / 2)
        mps.left_svd_trunc(trunc_val)

        final_norm = get_norm_from_MPS(mps)
        print(init_norm, trunc_val, final_norm)

        self.assertLess(final_norm, init_norm)
        self.assertLess(init_norm - trunc_val, final_norm)


unittest.main(argv=[''], verbosity=1, exit=True)
