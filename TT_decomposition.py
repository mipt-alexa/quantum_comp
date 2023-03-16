import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import det
from functools import reduce
import copy

from classes import MPS
from maxvol import maxvol

from typing import Tuple, List


def row_column_alternating(matrix: jnp.ndarray, r: int, tol: float = 1e-3, maxvol_tol: float = 1e-3) \
        -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    This function performs row-column alternating algorithm for matrices.
    Args:
        matrix: 2-dimensional array
        r: size of target submatrix of maximum volume
        tol: tolerance, the stopping criterion for iterations
        maxvol_tol: tolerance for maxvol algorithm
    Returns:
        (I, J): tuple of row and column indices as jax.numpy arrays
    """

    if len(matrix.shape) != 2:
        raise Exception("Tensor should be 2-dimensional")

    if r > min(matrix.shape):
        raise Exception("Submatrix size exceed one of matrix dimensions")

    J = jnp.array(np.random.choice(range(matrix.shape[1]), size=r, replace=False))
    I = maxvol(matrix[:, J], maxvol_tol)

    curr_det = 0.5 * abs(det(matrix[I][:, J]))

    while abs(det(matrix[I][:, J])) / curr_det > 1. + tol:
        curr_det = abs(det(matrix[I][:, J]))

        J = maxvol(jnp.transpose(matrix[I]), maxvol_tol)
        I = maxvol(matrix[:, J], maxvol_tol)

    return I, J


def TT_decomposition(tensor: jnp.ndarray, inner_dims: List[int], rc_tol: float = 1e-3, maxvol_tol: float = 1e-3) -> MPS:
    """
    This function produces the MPS approximation of a tensor using pseudo-skeleton decomposition
    Args:
        tensor: tensor to be decomposed
        inner_dims: list of desired inner dimensions of the tensor train
        rc_tol: tolerance for row-column alternating algorithm
        maxvol_tol: tolerance for maxvol algorithm
    Returns:
        TT approximation as MPS object
    """

    out_dims = list(tensor.shape)
    mps_len = len(out_dims)

    if mps_len != len(inner_dims) + 1:
        raise Exception("Number of inner dims does not match the tensor")

    in_dims = inner_dims
    in_dims.insert(0, 1)
    in_dims.append(1)

    M = copy.deepcopy(tensor)
    mps_tensors = []

    fold_dim = int(reduce((lambda x, y: x * y), out_dims))

    for i in range(mps_len - 1):
        fold_dim = int(fold_dim / out_dims[i])
        M = jnp.reshape(M, (in_dims[i] * out_dims[i], fold_dim))

        I, J = row_column_alternating(M, in_dims[i + 1], tol=rc_tol, maxvol_tol=maxvol_tol)
        C = M[:, J]

        Q, _ = jnp.linalg.qr(C)
        Q_hat = Q[maxvol(Q)]
        G = jnp.tensordot(Q, jnp.linalg.inv(Q_hat), 1)

        mps_tensors.append(jnp.reshape(G, (in_dims[i], out_dims[i], in_dims[i + 1])))
        M = M[I, :]

    mps_tensors.append(jnp.reshape(M, (in_dims[-2], out_dims[-1], in_dims[-1])))

    return MPS(mps_tensors)
