import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import det

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


def TT_decomposition(tensor: jnp.ndarray, inner_dim: List[int]) -> MPS:
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
