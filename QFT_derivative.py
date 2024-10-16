from typing import Callable

import numpy as np

import jax.numpy as jnp
from TT_cross import TT_cross
from helper_functions import get_tensor_from_MPS
import matplotlib.pyplot as plt
from classes import MPS, MPO
from QFT import QFT
from tests.test_QFT import reverse_operation


def generate_sequences(n):
    if n <= 0:
        return [[]]

    sequences = []
    for seq in generate_sequences(n - 1):
        sequences.append(seq + [0])
        sequences.append(seq + [1])

    return sequences


def my_func(ind: jnp.ndarray) -> jnp.ndarray:
    # sin(2 pi x**2)
    n = ind.shape[1]
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)
    return jnp.sin(2 * np.pi * jnp.power(jnp.tensordot(ind, POWERS, 1), 2))


class LinearMultiply:
    def __init__(self, mps: MPS):
        self.state = mps
        self.reversed_ord = jnp.array(range(mps.len - 1, -1, -1))

    def product(self, ind: jnp.ndarray) -> jnp.ndarray:
        values = []
        for j in range(len(ind)):
            ind = ind[self.reversed_ord]      # reverse the order as a final QFT step

            element = self.state.components[0][:, ind[j][0], :]
            for i in range(1, self.state.len):
                element = jnp.tensordot(element, self.state.components[i][:, ind[j][i], :], 1)
            values.append(float(element) * jnp.dot(ind[j], POWERS))
        return jnp.array(values)


n = 8
N = 2 ** n

mps = TT_cross(my_func, [2] * n, [2] * (n - 1))
qft = QFT(n, 4)

transform = qft.process(mps)

lm = LinearMultiply(transform)
output = TT_cross(lm.product, [2] * n, [2] * (n - 1))

derivative = qft.inverse_process(output)

out_state = jnp.transpose(jnp.reshape(get_tensor_from_MPS(derivative), [2] * N), range(-n, -1, -1))

binary_strings = jnp.array(generate_sequences(n))

