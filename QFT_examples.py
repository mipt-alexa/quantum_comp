import numpy as np

from QFT_4_qubit import QFT_4_qubit
import jax.numpy as jnp
from TT_cross import TT_cross
from helper_functions import get_tensor_from_MPS
import matplotlib.pyplot as plt
from classes import MPS

N = 4
POWERS = jnp.array([jnp.power(2., -n) for n in range(1, N+1)], dtype=jnp.float64)


def generate_sequences(n):
    if n <= 0:
        return [[]]

    sequences = []
    for seq in generate_sequences(n - 1):
        sequences.append(seq + [0])
        sequences.append(seq + [1])

    return sequences


def sin(ind: jnp.ndarray) -> jnp.ndarray:
    return jnp.sin(np.pi*jnp.dot(ind, jnp.tile(POWERS, 1)))


def get_element(mps: MPS, ind_arr: jnp.ndarray):

    values = []
    for j in range(len(ind_arr)):
        element = mps.components[0][:, ind_arr[j][0], :]
        for i in range(1, mps.len):
            element = jnp.tensordot(element, mps.components[i][:, ind_arr[j][i], :], 1)
        values.append(float(np.real(element)))
    return values

#
# def expected_fourier(x):
#     np.exp(2*np.pi*)


qft = QFT_4_qubit()

out_dims = [2] * N
in_dims = [2] * (N - 1)
sin_mps = TT_cross(sin, out_dims, in_dims)
qft.process(sin_mps)


binary_strings = jnp.array(generate_sequences(N))

y = np.array(np.real(get_element(sin_mps, binary_strings)))
x = np.array(jnp.dot(binary_strings, POWERS))
print(x)
y_1 = np.fft.fft(np.sin(np.pi*x))

x_0 = np.linspace(0, 1 - 1/16, 16)
print(x_0)
y_2 = np.fft.fft(np.sin(np.pi*x_0))

print(x, y)
plt.plot(x, y)
plt.plot(x, y_1)
plt.plot(x_0, y_2)
plt.show()
