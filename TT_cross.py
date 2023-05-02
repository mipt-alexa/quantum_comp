import copy
import timeit
import jax.config

from maxvol import maxvol
from helper_functions import *

from typing import List, Callable


jax.config.update("jax_enable_x64", True)


def rand_mps_comp(out_dims: List[int], in_dims: List[int]) -> List[jnp.array]:
    """
    Generates MPS of given shape with normally distributed elements
    """
    components = []

    for i in range(len(out_dims)):
        components.append(jnp.array(np.random.randn(in_dims[i], out_dims[i], in_dims[i + 1]), dtype=jnp.float64))

    return components


def TT_cross(tensor: Callable[[jnp.ndarray], jnp.ndarray], out_dims: List[int], inner_dims: List[int], max_iter: int = 100,
             tol: float = 1e-2, maxvol_tol: float = 1e-2) -> MPS:
    mps_len = len(out_dims)

    if mps_len != len(inner_dims) + 1:
        print(len(out_dims), len(inner_dims))
        raise Exception("Number of inner dims does not match the tensor dimension")

    in_dims = copy.copy(inner_dims)
    in_dims.insert(0, 1)
    in_dims.append(1)

    # initialization of left- anf right-nested indices sequences
    I, J = [], []
    for i in range(mps_len - 1):
        I.append(jnp.full((in_dims[i + 1], i + 1), 0, dtype=jnp.int32))
        J.append(jnp.full((in_dims[i + 1], mps_len - 1 - i), 0, dtype=jnp.int32))

    iter = 0

    v_prev = rand_mps_comp(out_dims, in_dims)
    v = rand_mps_comp(out_dims, in_dims)

    update_time = 0
    linalg_operation_time = 0
    tensor_return_time = 0

    while iter < max_iter:# and (MPS(v) - MPS(v_prev)).norm() > tol * MPS(v).norm():
        print("iter", iter)
        v_prev = v

        for k in range(0, mps_len):

            t_0 = timeit.default_timer()

            if iter > 0:

                if k == 0:
                    middle = jnp.repeat(jnp.array(range(out_dims[k])), in_dims[k + 1])
                    right = jnp.tile(J[k], [out_dims[k], 1])

                    indices = jnp.hstack((jnp.reshape(middle, (out_dims[k]*in_dims[k + 1], 1)), right))

                elif k == mps_len - 1:

                    middle = jnp.tile(jnp.array(range(out_dims[k])), in_dims[k])
                    left = jnp.repeat(I[k - 1], out_dims[k], axis=0)

                    indices = jnp.hstack((left, jnp.reshape(middle, (in_dims[k]*out_dims[k]*in_dims[k + 1], 1))))

                else:
                    right = jnp.tile(J[k], [in_dims[k] * out_dims[k], 1])
                    middle = jnp.tile(jnp.repeat(jnp.array(range(out_dims[k])), in_dims[k + 1]), in_dims[k])
                    left = jnp.repeat(I[k - 1], out_dims[k] * in_dims[k + 1], axis=0)

                    indices = jnp.hstack((left, jnp.reshape(middle, (in_dims[k]*out_dims[k]*in_dims[k + 1], 1)), right))

                t_3 = timeit.default_timer()

                v[k] = jnp.reshape(tensor(indices), (in_dims[k], out_dims[k], in_dims[k + 1]))

                tensor_return_time += timeit.default_timer() - t_3

            t_1 = timeit.default_timer()
            update_time += t_1 - t_0

            if k < mps_len - 1:

                Q, R = jnp.linalg.qr(jnp.reshape(v[k], (in_dims[k] * out_dims[k], in_dims[k + 1])))
                # print("Q R", Q.shape, R.shape)
                new_inds = maxvol(Q, tol=maxvol_tol)
                Q_hat = Q[new_inds]
                v[k] = jnp.reshape(jnp.tensordot(Q, jnp.linalg.inv(Q_hat), 1),
                                   (in_dims[k], out_dims[k], in_dims[k + 1]))
                if k == mps_len - 2:
                    v[k + 1] = jnp.tensordot(Q_hat, jnp.tensordot(R, v[k + 1], 1), 1)

                for i in range(len(new_inds)):
                    alpha = new_inds[i] // out_dims[k]
                    j_k = new_inds[i] % out_dims[k]

                    if k == 0:
                        I[k] = I[k].at[i, :].set(jnp.array([j_k]))
                    else:
                        I[k] = I[k].at[i, :].set(jnp.concatenate([I[k - 1][alpha], jnp.array([j_k])]))

            t_2 = timeit.default_timer()
            linalg_operation_time += t_2 - t_1
        iter += 1
        for k in range(mps_len - 1, -1, -1):

            t_0 = timeit.default_timer()

            if iter > 0:

                if k == 0:
                    middle = jnp.repeat(jnp.array(range(out_dims[k])), in_dims[k + 1])
                    right = jnp.tile(J[k], [out_dims[k], 1])

                    indices = jnp.hstack((jnp.reshape(middle, (out_dims[k] * in_dims[k + 1], 1)), right))

                elif k == mps_len - 1:

                    middle = jnp.tile(jnp.array(range(out_dims[k])), in_dims[k])
                    left = jnp.repeat(I[k - 1], out_dims[k], axis=0)

                    indices = jnp.hstack((left, jnp.reshape(middle, (in_dims[k] * out_dims[k] * in_dims[k + 1], 1))))

                else:
                    right = jnp.tile(J[k], [in_dims[k] * out_dims[k], 1])
                    middle = jnp.tile(jnp.repeat(jnp.array(range(out_dims[k])), in_dims[k + 1]), in_dims[k])
                    left = jnp.repeat(I[k - 1], out_dims[k] * in_dims[k + 1], axis=0)

                    indices = jnp.hstack((left, jnp.reshape(middle, (in_dims[k] * out_dims[k] * in_dims[k + 1], 1)), right))

                t_3 = timeit.default_timer()

                v[k] = jnp.reshape(tensor(indices), (in_dims[k], out_dims[k], in_dims[k + 1]))

                tensor_return_time += timeit.default_timer() - t_3

            t_1 = timeit.default_timer()
            update_time += t_1 - t_0

            if k > 0:
                Q, R = jnp.linalg.qr(jnp.transpose(jnp.reshape(v[k], (in_dims[k], out_dims[k] * in_dims[k + 1]))))
                new_inds = maxvol(Q, tol=maxvol_tol)
                Q_hat = Q[new_inds]
                v[k] = jnp.reshape(jnp.tensordot(jnp.transpose(jnp.linalg.inv(Q_hat)), jnp.transpose(Q), 1),
                                   (in_dims[k], out_dims[k], in_dims[k + 1]))
                if k == 1:
                    v[k - 1] = jnp.tensordot(v[k - 1], jnp.tensordot(jnp.transpose(R), jnp.transpose(Q_hat), 1), 1)

                for i in range(len(new_inds)):
                    alpha = new_inds[i] % in_dims[k + 1]
                    j_k = new_inds[i] // in_dims[k + 1]

                    if k == mps_len - 1:
                        J[k - 1] = J[k - 1].at[i, :].set(jnp.array([j_k]))
                    else:
                        J[k - 1] = J[k - 1].at[i, :].set(jnp.concatenate([jnp.array([j_k]), J[k][alpha]]))

            t_2 = timeit.default_timer()
            linalg_operation_time += t_2 - t_1
        print("v norm", MPS(v).norm())
        iter += 1

    # print("update time", update_time)
    # # print("tensor return time", tensor_return_time)
    # print("linalg operations", linalg_operation_time)

    return MPS(v), update_time, linalg_operation_time
