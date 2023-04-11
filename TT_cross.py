from jax.lax import full
import copy

from maxvol import maxvol
from helper_functions import *

from typing import List, Callable


def rand_mps_comp(out_dims: List[int], in_dims: List[int]) -> List[jnp.array]:
    """
    Generates MPS of given shape with normally distributed elements
    """
    components = []

    for i in range(len(out_dims)):
        components.append(jnp.array(np.random.randn(in_dims[i], out_dims[i], in_dims[i + 1])))

    return components


def TT_cross(tensor: Callable[[jnp.ndarray], float], out_dims: List[int], inner_dims: List[int], max_iter: int = 100,
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
        I.append(full((in_dims[i + 1], i + 1), 0))
        J.append(full((in_dims[i + 1], mps_len - 1 - i), 0))

    iter = 0

    v_prev = rand_mps_comp(out_dims, in_dims)
    v = rand_mps_comp(out_dims, in_dims)

    while iter < max_iter and (MPS(v) - MPS(v_prev)).norm() > tol * MPS(v).norm():
        iter += 1
        print("iter", iter)
        v_prev = v

        for k in range(0, mps_len):

            if iter > 0:
                for i in range(in_dims[k]):
                    for j in range(in_dims[k + 1]):
                        for l in range(out_dims[k]):
                            if k == 0:
                                multiind = jnp.concatenate([jnp.array([l]), J[k][j]])
                            elif k == mps_len - 1:
                                multiind = jnp.concatenate([I[k - 1][i], jnp.array([l])])
                            else:
                                multiind = jnp.concatenate([I[k - 1][i], jnp.array([l]), J[k][j]])

                            value = tensor(multiind)
                            v[k] = v[k].at[i, l, j].set(value)

            if k < mps_len - 1:

                Q, R = jnp.linalg.qr(jnp.reshape(v[k], (in_dims[k] * out_dims[k], in_dims[k + 1])))
                # print("Q R", Q.shape, R.shape)
                new_inds = maxvol(Q, tol=maxvol_tol)
                Q_hat = Q[new_inds]
                v[k] = jnp.reshape(jnp.tensordot(Q, jnp.linalg.inv(Q_hat), 1),
                                   (in_dims[k], out_dims[k], in_dims[k + 1]))
                v[k + 1] = jnp.tensordot(Q_hat, jnp.tensordot(R, v[k + 1], 1), 1)

                for i in range(len(new_inds)):
                    alpha = new_inds[i] // out_dims[k]
                    j_k = new_inds[i] % out_dims[k]

                    if k == 0:
                        I[k] = I[k].at[i, :].set(jnp.array([j_k]))
                    else:
                        I[k] = I[k].at[i, :].set(jnp.concatenate([I[k - 1][alpha], jnp.array([j_k])]))
        iter += 1

        for k in range(mps_len - 1, -1, -1):

            if iter > 0:
                for i in range(in_dims[k]):
                    for j in range(in_dims[k + 1]):
                        for l in range(out_dims[k]):

                            if k == 0:
                                multiind = jnp.concatenate([jnp.array([l]), J[k][j]])
                            elif k == mps_len - 1:
                                multiind = jnp.concatenate([I[k - 1][i], jnp.array([l])])
                            else:
                                multiind = jnp.concatenate([I[k - 1][i], jnp.array([l]), J[k][j]])
                            value = tensor(multiind)
                            v[k] = v[k].at[i, l, j].set(value)

            if k > 0:
                Q, R = jnp.linalg.qr(jnp.transpose(jnp.reshape(v[k], (in_dims[k], out_dims[k] * in_dims[k + 1]))))
                new_inds = maxvol(Q, tol=maxvol_tol)
                Q_hat = Q[new_inds]
                v[k] = jnp.reshape(jnp.tensordot(jnp.transpose(jnp.linalg.inv(Q_hat)), jnp.transpose(Q), 1),
                                   (in_dims[k], out_dims[k], in_dims[k + 1]))
                v[k - 1] = jnp.tensordot(v[k - 1], jnp.tensordot(jnp.transpose(R), jnp.transpose(Q_hat), 1), 1)

                for i in range(len(new_inds)):
                    alpha = new_inds[i] % in_dims[k + 1]
                    j_k = new_inds[i] // in_dims[k + 1]

                    if k == mps_len - 1:
                        J[k - 1] = J[k - 1].at[i, :].set(jnp.array([j_k]))
                    else:
                        J[k - 1] = J[k - 1].at[i, :].set(jnp.concatenate([jnp.array([j_k]), J[k][alpha]]))

    return MPS(v)
