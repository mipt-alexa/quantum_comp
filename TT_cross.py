import copy
import time
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


def component_indices(I_k, J_k, shape):
    right = jnp.tile(J_k, [shape[0] * shape[1], 1])
    middle = jnp.tile(jnp.repeat(jnp.array(range(shape[1])), shape[2]), shape[0])
    left = jnp.repeat(I_k, shape[1] * shape[2], axis=0)

    return jnp.hstack((left, jnp.reshape(middle, (shape[0] * shape[1] * shape[2], 1)), right))


# def update_left_nested_seq(new_inds, out_dims_k, I_k):
#     new_ind_seq = []
#
#     for i in range(len(new_inds)):
#         alpha = new_inds[i] // out_dims_k
#         j_k = new_inds[i] % out_dims_k
#         new_ind_seq.append(jnp.concatenate([I_k[alpha], jnp.array([j_k])]))
#     return jnp.array(new_ind_seq)


def TT_cross(tensor: Callable[[jnp.ndarray], jnp.ndarray], out_dims: List[int], inner_dims: List[int],
             max_iter: int = 100, min_iter: int = 1,
             tol: float = 1e-3, maxvol_tol: float = 1e-1) -> MPS:
    mps_len = len(out_dims)

    if mps_len != len(inner_dims) + 1:
        print(len(out_dims), len(inner_dims))
        raise Exception("Number of inner dims does not match the tensor dimension")

    in_dims = copy.copy(inner_dims)
    in_dims.insert(0, 1)
    in_dims.append(1)

    if in_dims[1] > out_dims[0]:
        in_dims[1] = out_dims[0]
    if in_dims[-2] > out_dims[-1]:
        in_dims[-2] = out_dims[-1]

    shapes = [()] * mps_len
    for k in range(mps_len):
        if in_dims[k] * out_dims[k] < in_dims[k + 1]:
            in_dims[k+1] = in_dims[k] * out_dims[k]

    for k in range(mps_len - 1, -1, -1):
        if in_dims[k] > out_dims[k] * in_dims[k+1]:
            in_dims[k] = out_dims[k] * in_dims[k+1]

    for k in range(mps_len):
        shapes[k] = (in_dims[k], out_dims[k], in_dims[k + 1])

    # initialization of left- anf right-nested indices sequences, respectively
    I, J = [], []

    I.append(jnp.array([[]], dtype=int))
    for i in range(mps_len - 1):
        I.append(jnp.full((in_dims[i + 1], i + 1), 0, dtype=jnp.int32))
        J.append(jnp.full((in_dims[i + 1], mps_len - 1 - i), 0, dtype=jnp.int32))
    J.append(jnp.array([[]], dtype=int))

    comp_indices_jit = jax.jit(component_indices, static_argnums=2)

    iter = 0

    v_prev = rand_mps_comp(out_dims, in_dims)
    v = rand_mps_comp(out_dims, in_dims)

    update_time = 0
    linalg_operation_time = 0
    tensor_return_time = 0
    set_time = 0
    maxvol_time = 0
    reshape_time = 0
    ind_update_time = 0

    while iter < min_iter or (MPS(v) - MPS(v_prev)).norm() > tol * MPS(v).norm():
        print("iter", iter)
        v_prev = v

        for k in range(0, mps_len):

            t_0 = timeit.default_timer()

            if iter > 0:
                indices = comp_indices_jit(I[k], J[k], shapes[k])

                t_3 = timeit.default_timer()

                v[k] = jnp.reshape(tensor(indices), shapes[k])

                tensor_return_time += timeit.default_timer() - t_3

            t_1 = timeit.default_timer()
            update_time += t_1 - t_0

            if k < mps_len - 1:

                Q, R = jnp.linalg.qr(jnp.reshape(v[k], (shapes[k][0] * shapes[k][1], shapes[k][2])))

                t_3 = timeit.default_timer()

                new_inds = maxvol(Q, tol=maxvol_tol)

                t_4 = timeit.default_timer()
                maxvol_time += t_4 - t_3

                Q_hat = Q[new_inds]

                v[k] = jnp.reshape(jnp.tensordot(Q, jnp.linalg.inv(Q_hat), 1), shapes[k])

                t_5 = timeit.default_timer()
                reshape_time += t_5 - t_4

                if k == mps_len - 2:
                    v[k + 1] = jnp.tensordot(Q_hat, jnp.tensordot(R, v[k + 1], 1), 1)

                new_ind_seq = []
                t_7 = timeit.default_timer()

                for i in range(len(new_inds)):
                    alpha = new_inds[i] // out_dims[k]
                    j_k = new_inds[i] % out_dims[k]
                    new_ind_seq.append(jnp.concatenate([I[k][alpha], jnp.array([j_k])]))

                I[k + 1] = jnp.array(new_ind_seq)
                set_time += timeit.default_timer() - t_7

                ind_update_time += timeit.default_timer() - t_5

            t_2 = timeit.default_timer()
            linalg_operation_time += t_2 - t_1
        iter += 1
        for k in range(mps_len - 1, -1, -1):

            t_0 = timeit.default_timer()

            if iter > 0:
                indices = comp_indices_jit(I[k], J[k], shapes[k])

                t_3 = timeit.default_timer()

                v[k] = jnp.reshape(tensor(indices), (in_dims[k], out_dims[k], in_dims[k + 1]))
                tensor_return_time += timeit.default_timer() - t_3

            t_1 = timeit.default_timer()
            update_time += t_1 - t_0

            if k > 0:
                Q, R = jnp.linalg.qr(jnp.transpose(jnp.reshape(v[k], (shapes[k][0], shapes[k][1] * shapes[k][2]))))

                t_3 = timeit.default_timer()

                new_inds = maxvol(Q, tol=maxvol_tol)

                t_4 = timeit.default_timer()
                maxvol_time += t_4 - t_3

                Q_hat = Q[new_inds]
                v[k] = jnp.reshape(jnp.tensordot(jnp.transpose(jnp.linalg.inv(Q_hat)), jnp.transpose(Q), 1),
                                   shapes[k])

                t_5 = timeit.default_timer()
                reshape_time += t_5 - t_4

                if k == 1:
                    v[k - 1] = jnp.tensordot(v[k - 1], jnp.tensordot(jnp.transpose(R), jnp.transpose(Q_hat), 1), 1)

                t_7 = timeit.default_timer()

                new_ind_seq = []
                for i in range(len(new_inds)):
                    alpha = new_inds[i] % in_dims[k + 1]
                    j_k = new_inds[i] // in_dims[k + 1]
                    new_ind_seq.append(jnp.concatenate([jnp.array([j_k]), J[k][alpha]]))

                J[k - 1] = jnp.array(new_ind_seq)
                set_time += timeit.default_timer() - t_7

                ind_update_time += timeit.default_timer() - t_5

            t_2 = timeit.default_timer()
            linalg_operation_time += t_2 - t_1

        iter += 1

    # print("update time", update_time)
    # # print("tensor return time", tensor_return_time)
    # print("linalg operations", linalg_operation_time)

    return MPS(v)  # , update_time, linalg_operation_time, set_time, maxvol_time, reshape_time, ind_update_time
