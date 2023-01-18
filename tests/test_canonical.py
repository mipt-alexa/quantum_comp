from helper_functions import *
import unittest
import numpy as np

MPS_LEN = 6  # number of components of MPS
MAX_DIM = 6  # maximum dimension of any mps component index


class TestCanonical(unittest.TestCase):
    def test_unitary(self):
        mps = create_mps(MPS_LEN, MAX_DIM)

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
        mps = create_mps(MPS_LEN, MAX_DIM)
        expected = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)

        mps.right_canonical()
        result = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)
        self.assertTrue(np.allclose(expected, result, rtol=1e-5))

        mps.left_canonical()
        result = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)
        self.assertTrue(np.allclose(expected, result, rtol=1e-5))

    def test_canonical_form_norm_property(self):
        mps = create_mps(MPS_LEN, MAX_DIM)
        a = jnp.tensordot(get_tensor_from_MPS(mps), get_tensor_from_MPS(mps), MPS_LEN + 2)

        mps.left_canonical()
        b = jnp.tensordot(mps.components[-1], mps.components[-1], 3)
        self.assertTrue(np.allclose(a, b, rtol=1e-5))

        mps.right_canonical()
        c = jnp.tensordot(mps.components[0], mps.components[0], 3)
        self.assertTrue(np.allclose(a, c, rtol=1e-5))

    def test_unitary_after_truncation(self):
        mps = create_mps(MPS_LEN, MAX_DIM)

        mps.left_canonical()

        trunc_val = np.random.uniform(0, np.sqrt(mps.dot(mps)) / 2)
        mps.left_svd_trunc(trunc_val)

        validation = []
        mps.left_canonical()

        for i in range(MPS_LEN - 1):
            shape = mps.components[i].shape
            result = jnp.tensordot(mps.components[i], mps.components[i], [[0, 1], [0, 1]])
            expected = np.diag(np.ones(shape[2]))
            validation.append(np.allclose(result, expected, atol=1e-06))

        self.assertTrue(np.all(validation))


unittest.main(argv=[''], verbosity=1, exit=True)
