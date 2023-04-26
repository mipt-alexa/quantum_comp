import jax.numpy as jnp
import timeit
import numpy as np
import jax.config

from TT_cross import TT_cross


jax.config.update("jax_enable_x64", True)


def hilbert_tensor(ind_arr):
    if len(ind_arr.shape) == 1:
        return 1 / (jnp.sum(ind_arr) + len(ind_arr))

    return 1 / (len(ind_arr[0]) + jnp.sum(ind_arr, axis=1))


mps_len = 40
out_dims = [20] * mps_len

for in_dim in range(2, out_dims[0]//2):
    in_dims = [in_dim]*(mps_len - 1)
    print(in_dims)
    t_0 = timeit.default_timer()
    mps = TT_cross(hilbert_tensor, out_dims, in_dims)

    print("in_dim =", in_dim, "time =", timeit.default_timer() - t_0)

    delta = 0
    for j in range(100):

        rand_multiind = jnp.array(np.random.choice(range(out_dims[0]), mps_len, replace=True))

        element = mps.components[0][:, rand_multiind[0], :]
        for i in range(1, mps_len):
            element = jnp.tensordot(element, mps.components[i][:, rand_multiind[i], :], 1)

        # print(float(element), float(hilbert_tensor([rand_multiind])))

        delta += abs(float(element) - float(hilbert_tensor(rand_multiind)))

    print("delta", delta)
