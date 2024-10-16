import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from QFT import QFT
from classes import MPS
from helper_functions import get_tensor_from_MPS
from QFT_examples import get_element, generate_sequences


def exp_linear(N, x, L):
    components = [jnp.reshape(jnp.array([1, jnp.exp(2*jnp.pi*complex(0,1) *x / L* jnp.power(2., i))]),
                              (1, 2, 1)) for i in range(N-1, -1, -1)]

    return MPS(components)


def interpolation(n, L, f):
    N = 2**n
    x0 = jnp.linspace(0, (N-1)/N, 100)
    y = np.empty_like(x0)

    qft = QFT(n, N)
    binary_strings = jnp.array(generate_sequences(n))

    for i in tqdm(range(len(x0))):
        mps = exp_linear(n, x0[i], L)

        result = qft.process(mps)

        result_1d = jnp.array(get_element(result, binary_strings))
        result_1d = jnp.abs(result_1d / jnp.sqrt(N))

        result_new = result_1d
        result_new = result_new[::-1]
        result_new = np.concatenate(([result_new[-1]], result_new[:-1]))

        i_left = int(x0[i] * N)

        signs_right = [(-1) ** (i - i_left - 1) for i in range(i_left + 1, N)]
        signs_left = [(-1) ** (i) for i in range(0, i_left + 1)]

        if i_left == N-1:
            signs = signs_left
        else:
            signs = np.concatenate((signs_left[::-1], signs_right))

        result_new = np.multiply(signs, result_new)
        # result_new = jnp.concatenate((result_1d[0:1], result_1d[1:][::-1]))
        y[i] = jnp.dot(f, result_new)

    return x0, y


def theory_interpolation(n, L, f):
    N = 2**n
    x0 = jnp.linspace(0, (N-1)/N, 100)
    y = np.empty_like(x0)

    for i in range(len(x0)):
        expected = jnp.array([1 / N * jnp.sin(jnp.pi * (x0[i] * N / L - k)) / jnp.sin(jnp.pi * (x0[i] / L - k / N)) for k in range(0, N)])
        y[i] = jnp.dot(f, expected)

    return x0, y


x = 0.97
L = 1
n = 4
N = 2**n

mps = exp_linear(n, x, L)

qft = QFT(n, N)

result = qft.process(mps)

binary_strings = jnp.array(generate_sequences(n))
result_1d = jnp.array(get_element(result, binary_strings))
result_1d = jnp.abs(result_1d / jnp.sqrt(N))

expected = jnp.array([1/N *jnp.sin(jnp.pi*(x*N/L - k)) / jnp.sin(jnp.pi*(x/L - k/N)) for k in range(0, N)])

result_new = result_1d
result_new = result_new[::-1]
result_new = np.concatenate(([result_new[-1]], result_new[:-1]))

result_conj = [(result_1d[i] + np.conj(result_1d[N-i-1])) /2 for i in range(N)]

i_left = int(x*N)
signs_right = [(-1)**(i - i_left-1) for i in range(i_left +1, N)]
signs_left = [(-1)**(i) for i in range(0, i_left + 1)]
signs = np.concatenate((signs_left[::-1], signs_right))

result_new = np.multiply(signs, result_new)

fig, ax = plt.subplots(2, 1)

ax[0].scatter(range(N), result_new, label='qft')
ax[0].scatter(range(N), expected, marker='x', color='red', label='theoretical')
# ax[0].scatter(range(N), result_conj, marker='o')

ax[0].legend()

x_k = jnp.linspace(0, (N-1)/N, N)
f = jnp.sin(4*np.pi*x_k)*jnp.sin(6*np.pi*x_k)

x0, y0 = interpolation(n, L, f)
x1, y1 = theory_interpolation(n, L, f)

ax[1].scatter(x_k, f)
ax[1].plot(x0[:-1], y0[:-1], label='using qft')
ax[1].plot(x1, y1, label='theoretical')
ax[1].legend()

fig.show()

