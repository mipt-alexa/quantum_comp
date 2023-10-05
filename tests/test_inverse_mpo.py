from typing import Tuple, List

import jax
import unittest
import jax.numpy as jnp
import numpy as np
import sys

from QFT import QFT
from classes import MPS
from helper_functions import get_tensor_from_MPS

jax.config.update("jax_enable_x64", True)


def inverse_qft(qft, mps):
    components = []

    n = mps.len
    for i in range(0, n):
        state_shape = mps.components[i].shape
        oper_i = jnp.transpose(qft.components[n - 1 - i], (2, 3, 0, 1))
        oper_shape = oper_i.shape

        subresult = jnp.tensordot(jnp.conj(oper_i), mps.components[i], [[3], [1]])
        components.append(jnp.reshape(jnp.transpose(subresult, (0, 3, 1, 2, 4)),
                                      (oper_shape[0] * state_shape[0], state_shape[1], oper_shape[2] * state_shape[2])))
    return MPS(components)


class TestInverseMPO(unittest.TestCase):

    def test_inverse_QFT(self):

        for n in range(2, 6):

            def random_state(order=n):
                components = [jnp.array(np.random.uniform(size=(1, 2, 2)))]
                for _ in range(1, order - 1):
                    components.append(jnp.array(np.random.uniform(size=(2, 2, 2))))
                components.append(jnp.array(np.random.uniform(size=(2, 2, 1))))

                return MPS(components)

            state = random_state()

            qft = QFT(n, 2 ** n)

            transform = qft.process(state)

            out_state = inverse_qft(qft, transform)

            self.assertTrue(jnp.allclose(get_tensor_from_MPS(out_state), get_tensor_from_MPS(state), rtol=1e-7))


if __name__ == '__main__':
    unittest.main(argv=[''], verbosity=1, exit=True)
