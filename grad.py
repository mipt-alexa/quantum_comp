import jax
import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np

from QFT import QFT
from classes import MPS
from TT_cross import TT_cross
from QFT_examples import get_element, generate_sequences
from helper_functions import get_tensor_from_MPS
from tests.test_QFT import perm_reverse_tensor

n = 4
N = 2 ** n

bin_seq = jnp.array(generate_sequences(n))
powers = jnp.array([jnp.power(2., -i - 1) for i in range(n)])
j = complex(0, 1)


def func(indx):
    ind = jnp.dot(indx, powers)
    return jnp.sin(4 * jnp.pi * ind) + jnp.cos(6 * jnp.pi*ind)


f_mps = TT_cross(func, [2] * n, [2] * (n - 1), tol=1e-3, maxvol_tol=1e-2)
f_vec = jnp.ravel((get_tensor_from_MPS(f_mps)))

phi = QFT(n, n).process(f_mps)

x0 = jnp.dot(bin_seq, powers)
x = np.linspace(0, x0[-1], 100)

f_interp = np.empty_like(x)


def interpolation(x, fft_mps=phi):
    interp_mps = MPS([jnp.array([[[1], [jnp.exp(-2 * jnp.pi * x * j * 2 ** k)]]]) for k in range(0, n)])
    interp_mps.components[-1] = jnp.array([[[1], [jnp.exp(2 * jnp.pi * j * N / 2 * x)]]])

    return jnp.real(fft_mps.dot(interp_mps)) / jnp.sqrt(N)


for i in range(len(x)):
    f_interp[i] = interpolation(x[i])

grad = jax.grad(interpolation, argnums=0)

derivative = [grad(x) for x in x]

fig,ax = plt.subplots(2, 1, sharex=True)
ax[0].scatter(x0, f_vec)
ax[0].plot(x, f_interp)
ax[1].plot(x, derivative)
ax[0].set_title('Interpolation')
ax[1].set_title('Derivative')
plt.show()
