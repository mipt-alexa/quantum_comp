from typing import Callable, List

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


def sin(ind: jnp.ndarray) -> jnp.ndarray:
    if len(ind.shape) == 1:
        n = ind.shape
    else:
        n = ind.shape[1]
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)
    return jnp.sin(2* np.pi * jnp.power(jnp.tensordot(ind, POWERS, 1), 2)) * \
           jnp.sin(4 * np.pi * jnp.tensordot(ind, POWERS, 1))


def cos(ind: jnp.array) -> jnp.array:
    if len(ind.shape) == 1:
        n = ind.shape
    else:
        n = ind.shape[1]
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)
    return jnp.cos(2 * np.pi * jnp.tensordot(ind, POWERS, 1)) + \
           jnp.exp(- jnp.abs(jnp.tensordot(ind, POWERS, 1)))


def const(ind):
    return jnp.ones(ind.shape[0])


def exp(ind):
    n = ind.shape[1]
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)
    return jnp.exp(- jnp.abs(jnp.tensordot(ind, POWERS, 1)))


def sin_f(x):
    return np.sin(3*2*np.pi*x) + 2 * np.sin(5*2*np.pi*x)


def sin_squared(ind):
    n = ind.shape[1]
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)
    return jnp.sin(2 * np.pi * jnp.power(jnp.tensordot(ind, POWERS, 1), 2))


def x(ind: jnp.ndarray) -> jnp.ndarray:
    if len(ind.shape) == 1:
        n = ind.shape
    else:
        n = ind.shape[1]
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)
    return jnp.tensordot(ind, POWERS, 1)


def get_element(mps: MPS, ind_arr: jnp.ndarray) -> List[float]:
    """
    This function extracts the elements of an MPS by index with respect to reversed entries.
    """
    reversed_ord = jnp.array(range(mps.len - 1, -1, -1))
    ind_arr = ind_arr[:, reversed_ord]

    values = []
    for j in range(len(ind_arr)):

        element = mps.components[0][:, ind_arr[j][0], :]
        for i in range(1, mps.len):
            element = jnp.tensordot(element, mps.components[i][:, ind_arr[j][i], :], 1)
        values.append(element[0][0])
    return values


def plot_ft(func: Callable, n: int, qft_bond_dim: int):
    N = 2 ** n
    POWERS = jnp.array([jnp.power(2., -k) for k in range(1, n + 1)], dtype=jnp.float64)

    mps = TT_cross(func, [2] * n, [2] * (n - 1))

    qft = QFT(n, qft_bond_dim)
    output = qft.process(mps)

    # output = jnp.reshape(get_tensor_from_MPS(output), [2] * n)
    # output = jnp.transpose(output, range(n - 1, -1, -1))
    # output = jnp.tensordot(reverse_operation(n), output, 1)

    binary_strings = jnp.array(generate_sequences(n))
    y = np.array(get_element(output, binary_strings))

    plt.stem(range(N), jnp.real(y), linefmt="r--", markerfmt="ro", label="qft real")
    plt.stem(range(N), jnp.imag(y), linefmt="b--", markerfmt="bo", label="qft imag")

    y_1 = np.fft.ifft(func(binary_strings), norm="ortho")
    print(abs(y - y_1))

    plt.stem(range(N), np.real(y_1), linefmt="red", markerfmt=" ", label="fft real")
    plt.stem(range(N), np.imag(y_1), linefmt="blue", markerfmt=" ", label="fft imag")
    plt.xlim(0, 10)
    plt.legend()
    plt.show()


if __name__ == '__main__':
    plot_ft(sin, 6, 10)
