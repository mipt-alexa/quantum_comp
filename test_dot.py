from classes import *
import unittest

import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_tensor_from_MPS(x: MPS) -> jnp.ndarray:
    result = x.components[0]
    for i in range(1, x.len):
        result = jnp.tensordot(result, x.components[i], 1)
    return result


class Test(unittest.TestCase):
    def test_dot(self):
        self.max_dim = 6  # maximum dimension of any index

        for length in range(2, 7):
            self.outer_dims = np.random.randint(2, self.max_dim, size=length)  # visible dimensions of each tensor
            self.inner_dims_a = np.random.randint(2, self.max_dim, length-1)  # bond dimensions of tensor A
            self.inner_dims_b = np.random.randint(2, self.max_dim, length-1)

            print(self.outer_dims, self.inner_dims_a, self.inner_dims_b)

            self.comp_a = [np.random.randn(1, self.outer_dims[0], self.inner_dims_a[0])]  # filling the first components of mps
            self.comp_b = [np.random.randn(1, self.outer_dims[0], self.inner_dims_b[0])]

            for i in range(1, length - 1):
                self.comp_a.append(np.random.randn(self.inner_dims_a[i - 1], self.outer_dims[i], self.inner_dims_a[i]))
                self.comp_b.append(np.random.randn(self.inner_dims_b[i - 1], self.outer_dims[i], self.inner_dims_b[i]))

            self.comp_a.append(
                np.random.randn(self.inner_dims_a[-1], self.outer_dims[-1], 1))  # the last components of mps
            self.comp_b.append(np.random.randn(self.inner_dims_b[-1], self.outer_dims[-1], 1))

            self.a = MPS(self.comp_a)
            self.b = MPS(self.comp_b)

            result = self.a.dot(self.b)
            expected = jnp.tensordot(get_tensor_from_MPS(self.a), get_tensor_from_MPS(self.b), length+2)

            self.assertTrue(np.allclose(result, expected, rtol=1e-5))


unittest.main(argv=[''], verbosity=1, exit=False)
