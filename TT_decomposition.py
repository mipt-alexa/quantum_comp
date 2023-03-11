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

    J = jnp.array(np.random.choice(range(matrix.shape[1]), size=r, replace=False))
    I = maxvol(matrix[:, J], maxvol_tol)

    curr_det = 0.5 * abs(det(matrix[I][:, J]))

    while abs(det(matrix[I][:, J])) / curr_det > 1. + tol:
        curr_det = abs(det(matrix[I][:, J]))

        J = maxvol(jnp.transpose(matrix[I]), maxvol_tol)
        I = maxvol(matrix[:, J], maxvol_tol)

    return I, J


def TT_decomposition(tensor: jnp.ndarray, inner_dims: List[int], rc_tol:float = 1e-3, maxvol_tol:float = 1e-3) -> MPS:
    """
    This function produces the MPS approximation of a tensor using pseudo-skeleton decomposition
    Args:
        tensor: tensor to be decomposed
        inner_dim: list of desired inner dimensions of the tensor train
        rc_tol: tolerance for row-column alternating algorithm
        maxvol_tol: tolerance for maxvol algorithm
    Returns:
        TT approximation as MPS object
    """

    out_dims = list(tensor.shape)
    dim_num = len(out_dims)

    if dim_num != len(inner_dims) + 1:
        raise Exception("Number of inner dims does not match the tensor")

    in_dims = inner_dims
    in_dims.insert(0, 1)
    in_dims.append(1)
    print(in_dims)

    M = copy.deepcopy(tensor)
    mps_tensors = []

    fold_dim = int(reduce((lambda x, y: x * y), out_dims))

    for i in range(dim_num-1):
        fold_dim = int(fold_dim / out_dims[i])
        print(fold_dim)
        M = jnp.reshape(M, (in_dims[i]*out_dims[i], fold_dim))
        print("M", M.shape)

        I, J = row_column_alternating(M, in_dims[i+1], tol=rc_tol, maxvol_tol=maxvol_tol)
        C = M[:, J]
        A_hat = M[I][:, J]
        R = M[I, :]
        print("C", C.shape, "A", A_hat.shape, "R", R.shape)

        mps_tensors.append(jnp.reshape(C, (in_dims[i], out_dims[i], in_dims[i+1])))
        M = jnp.tensordot(jnp.linalg.inv(A_hat), R, 1)

    mps_tensors.append(jnp.reshape(M, (in_dims[-2], out_dims[-1], in_dims[-1])))

    return MPS(mps_tensors)


def TT_decomposition_basic_Q(tensor: jnp.ndarray, inner_dim: List[int]) -> MPS:
    """
    This function performs the tensor pseudo-skeleton decomposition YET of dimensionality 2
    Args:
        tensor: tensor to be decomposed
        inner_dim: list of desired inner dimensions of the tensor train
    Returns:
        TT approximation as MPS object
    """
    shape = list(tensor.shape)
    dim = len(shape)

    mps_tensors = []

    A_1 = tensor.reshape(shape[0], shape[1])
    I, J = row_column_alternating(A_1, inner_dim[0])

    C = A_1[:, J]

    Q, T = jnp.linalg.qr(C)
    Q_hat = Q[maxvol(Q)]

    G = jnp.tensordot(Q, jnp.linalg.inv(Q_hat), 1)
    G = jnp.reshape(G, ([1] + list(G.shape)))
    mps_tensors.append(G)

    R = A_1[I]
    R = jnp.reshape(R, (list(R.shape) + [1]))
    mps_tensors.append(R)

    return MPS(mps_tensors)
