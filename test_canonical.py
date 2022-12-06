from classes import *
import unittest
import numpy as np

MPS_LEN = 6  # number of components of MPS
MAX_DIM = 6  # maximum dimension of any mps component index


def get_tensor_from_MPS(x: MPS) -> jnp.ndarray:
    result = x.components[0]
    for i in range(1, x.len):
        result = jnp.tensordot(result, x.components[i], 1)
    return result


def create_mps(length: int) -> MPS:
    outer_dims = np.random.randint(1, MAX_DIM, length)  # visible dimensions
    inner_dims = np.random.randint(1, MAX_DIM, length - 1)  # bond dimensions

    comp = [np.random.randn(1, outer_dims[0], inner_dims[0])]  # filling the first components of mps
    for i in range(1, length - 1):
        comp.append(np.random.randn(inner_dims[i - 1], outer_dims[i], inner_dims[i]))
    comp.append(
        np.random.randn(inner_dims[-1], outer_dims[-1], 1))  # the last components of mps

    mps = MPS(comp)
    return mps


class TestCanonical(unittest.TestCase):
    def test_hermitian_adjoint(self):
        mps = create_mps(MPS_LEN)

        validation = []
        mps.left_canonical()

        for i in range(MPS_LEN - 1):
            shape = mps.components[i].shape
            result = jnp.tensordot(mps.components[i], mps.components[i], [[0, 1], [0, 1]])
            expected = np.diag(np.ones(shape[2]))
            validation.append(np.allclose(result, expected, atol=1e-06))

        self.assertTrue(np.all(validation))
        validation.clear()
        mps.right_canonical()

        for i in range(1, MPS_LEN):
            shape = mps.components[i].shape
            result = jnp.tensordot(mps.components[i], mps.components[i], [[1, 2], [1, 2]])
            expected = np.diag(np.ones(shape[0]))
            validation.append(np.allclose(result, expected, atol=1e-06))

        self.assertTrue(np.all(validation))

    def test_invariable_norm(self):
        mps = create_mps(MPS_LEN)
        expected = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)

        mps.right_canonical()
        result = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)
        self.assertTrue(np.allclose(expected, result, rtol=1e-5))

        mps.left_canonical()
        result = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)
        self.assertTrue(np.allclose(expected, result, rtol=1e-5))

    def test_canonical_norm_property(self):
        mps = create_mps(MPS_LEN)
        a = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)

        mps.left_canonical()
        b = jnp.tensordot(mps.components[-1], mps.components[-1], 3)
        self.assertTrue(np.allclose(a, b, rtol=1e-5))

        mps.right_canonical()
        c = jnp.tensordot(mps.components[0], mps.components[0], 3)
        self.assertTrue(np.allclose(a, c, rtol=1e-5))


unittest.main(argv=[''], verbosity=2, exit=True)
