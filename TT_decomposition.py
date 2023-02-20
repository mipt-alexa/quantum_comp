import jax.numpy as jnp
import numpy as np
from jax.numpy.linalg import det
from functools import reduce

from classes import MPS
from maxvol import *

from typing import Tuple

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


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


# def TT_decomposition(tensor: jnp.ndarray) -> MPS:
#     """
#     This function performs
#     Args:
#         tensor:
#     Returns:
#
#     """
#     print(tensor.shape)
#     shape = list(tensor.shape)
#     dim = len(shape)
#
#     ind_1 = reduce(lambda x, y: x * y, shape[1:])
#
#     A_1 = tensor.reshape(shape[0], ind_1)
#     print(A_1)
#     _, J = row_column_alternating(A_1, 2)
#
#     C = A_1[:, J]
#
#     Q, T = jnp.linalg.qr(C)
#     I = maxvol(Q)
#     Q_hat = Q[I]
#
#     G_1 = jnp.tensordot(Q, jnp.linalg.inv(Q_hat), 1)
#
