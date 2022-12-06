from classes import *
import unittest
import numpy as np

MPS_LEN = 6  # number of components of MPS
MAX_DIM = 6  # maximum dimension of any mps component index


def get_norm_from_MPS(x: MPS) -> float:
    result = x.components[0]
    for i in range(1, x.len):
        result = jnp.tensordot(result, x.components[i], 1)
    return np.sqrt(jnp.tensordot(result, result, x.len + 2))


def create_mps(length: int) -> MPS:
    outer_dims = np.random.randint(2, MAX_DIM, length)  # visible dimensions
    inner_dims = np.random.randint(2, MAX_DIM, length - 1)  # bond dimensions

    comp = [np.random.randn(1, outer_dims[0], inner_dims[0])]  # filling the first components of mps
    for i in range(1, length - 1):
        comp.append(np.random.randn(inner_dims[i - 1], outer_dims[i], inner_dims[i]))
    comp.append(
        np.random.randn(inner_dims[-1], outer_dims[-1], 1))  # the last components of mps

    mps = MPS(comp)
    return mps


class TestTrunc(unittest.TestCase):
    def test_norm_conservation(self):
        mps = create_mps(MPS_LEN)

        init_norm = get_norm_from_MPS(mps)

        mps.left_canonical()
        trunc_val = np.random.uniform(0, init_norm / 2)
        mps.left_svd_trunc(trunc_val)

        final_norm = get_norm_from_MPS(mps)
        print(init_norm, trunc_val, final_norm)

        self.assertLess(final_norm, init_norm)
        self.assertLess(init_norm - trunc_val, final_norm)


unittest.main(argv=[''], verbosity=1, exit=True)
