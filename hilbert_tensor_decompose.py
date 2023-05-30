import jax.numpy as jnp
import timeit
import numpy as np
import jax.config
from jax import jit

from TT_cross import TT_cross


jax.config.update("jax_enable_x64", True)


def hilbert_tensor(ind_arr):
    if len(ind_arr.shape) == 1:
        return jnp.array(1 / (jnp.sum(ind_arr) + len(ind_arr)), dtype=jnp.float64)

    return jnp.array(1 / (len(ind_arr[0]) + jnp.sum(ind_arr, axis=1)), dtype=jnp.float64)


def main():
    mps_len = 50
    out_dims = [30] * mps_len

    len_ = 12
    deltas = np.empty(len_ - 2)
    update_time = np.empty(len_ - 2)
    linalg_time = np.empty(len_ - 2)
    set_time = np.empty(len_ - 2)
    maxvol_time = np.empty(len_ - 2)
    reshape_time = np.empty(len_ - 2)
    ind_update_time = np.empty(len_ - 2)

    hilbert_tensor_jit = jit(hilbert_tensor)

    for in_dim in range(2, len_):
        in_dims = [in_dim]*(mps_len - 1)

        t_0 = timeit.default_timer()
        mps, update_time[in_dim - 2], linalg_time[in_dim - 2], set_time[in_dim - 2], maxvol_time[in_dim - 2], reshape_time[in_dim - 2], ind_update_time[in_dim - 2] =\
            TT_cross(hilbert_tensor_jit, out_dims, in_dims, tol=1e-2, max_iter=6)

        print("in_dim =", in_dim, "time =", timeit.default_timer() - t_0, "norm=", mps.norm())

        delta = 0

        for j in range(100):
            rand_multiind = jnp.array(np.random.choice(range(out_dims[0]), mps_len, replace=True))

            element = mps.components[0][:, rand_multiind[0], :]
            for i in range(1, mps_len):
                element = jnp.tensordot(element, mps.components[i][:, rand_multiind[i], :], 1)

            delta += abs(jnp.float64(element) - jnp.float64(hilbert_tensor(rand_multiind))) / jnp.float64(hilbert_tensor(rand_multiind))
        deltas[in_dim - 2] = delta / 100

    print(np.column_stack((deltas, update_time, linalg_time, set_time, maxvol_time, reshape_time, ind_update_time)))


if __name__ == "__main__":
    main()
