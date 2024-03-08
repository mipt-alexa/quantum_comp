import jax.numpy as jnp

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

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
    return jnp.sin(2*jnp.pi*ind) + jnp.cos(4*jnp.pi*ind)


f_mps = TT_cross(func, [2] * n, [2] * (n - 1))
f_vec = jnp.ravel((get_tensor_from_MPS(f_mps)))

x0 = jnp.dot(bin_seq, powers)

phi = QFT(n, n).process(f_mps)

phi_vec = jnp.ravel(jnp.transpose(get_tensor_from_MPS(phi), perm_reverse_tensor(n)))
true_transform = jnp.fft.fft(f_vec, norm='ortho')


x = np.linspace(0, x0[-1], 100)

theor_interp = np.empty_like(x, dtype=complex)
f_interp = np.empty_like(x)

for i in range(len(x)):
    interp_mps = MPS([jnp.array([[[1], [jnp.exp(-2 * jnp.pi * x[i] * j * 2 ** k)]]]) for k in range(0, n)])
    interp_mps.components[-1] = jnp.array([[[1], [jnp.exp(2 * jnp.pi * j * N/2 * x[i])]]])

    mycoef_vec = jnp.ravel(jnp.transpose(get_tensor_from_MPS(interp_mps), perm_reverse_tensor(n)))

    f_interp[i] = jnp.real(phi.dot(interp_mps)) / jnp.sqrt(N)

    theor_coef = np.empty_like(phi_vec)

    theor_coef[0] = 1
    theor_coef[N // 2] = np.cos(jnp.pi / 1 * N * x[i])
    for k in range(1, N // 2):
        theor_coef[k] = jnp.exp(2 * jnp.pi * j * x[i] / 1 * k)
        theor_coef[N - k] = jnp.exp(-2 * jnp.pi * j * x[i] / 1 * k)

    theor_interp[i] = jnp.dot(true_transform, theor_coef) / jnp.sqrt(N)



f_vec = jnp.ravel(get_tensor_from_MPS(f_mps))

plt.scatter(x0, f_vec, color='blue')
plt.plot(x, f_interp.real, label='qft interp', color='red')
plt.legend()
plt.show()
