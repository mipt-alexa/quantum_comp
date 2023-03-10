import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, lu
import copy

from typing import Tuple


def arg_absmax(arr: jnp.ndarray) -> Tuple[int, int]:
    """
    This function returns the indices of the absolute maximum among 2d-array entries
    """
    amax = jnp.argmax(arr)
    amin = jnp.argmin(arr)
    if abs(jnp.ravel(arr)[amax]) > abs(jnp.ravel(arr)[amin]):
        i, j = amax // arr.shape[1], amax % arr.shape[1]
    else:
        i, j = amin // arr.shape[1], amin % arr.shape[1]
    return i, j


def maxvol(matrix: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
    """
    This function implements the maxvol algorithm by using the PLU decomposition
    on the first step and Sherman-Woodbury-Morrison formula
    for matrix inverse on the following steps

    Args:
        matrix: 2-dimensional array
        tol: tolerance, the stopping criterion in iterations

    Returns:
        The list of row indices forming the quasi- maximum volume submatrix.
    """

    A = copy.deepcopy(matrix)

    if A.shape[0] < A.shape[1]:
        raise Exception("Condition on matrix dimensions is not satisfied")

    n, r = A.shape
    steps = 0

    P, L, U = lu(A)

    Q = solve_triangular(jnp.transpose(U), jnp.transpose(A), lower=True)
    B = jnp.transpose(solve_triangular(jnp.transpose(L[0:r]), Q, lower=False))

    row_inds = jnp.tensordot(jnp.transpose(P), jnp.array(range(n)), 1)
    row_inds = jnp.asarray(row_inds, dtype=int)
    B = B[row_inds]

    i, j = arg_absmax(B)

    while not abs(B[i][j]) < 1. + tol:

        buff = row_inds[j]
        row_inds = row_inds.at[j].set(row_inds[i])
        row_inds = row_inds.at[i].set(buff)

        e_i, e_j, e_j_T = jnp.zeros(n), jnp.zeros(n), jnp.zeros(r)
        e_i = e_i.at[i].set(1)
        e_j = e_j.at[j].set(1)
        e_j_T = e_j_T.at[j].set(1)

        B -= 1 / B[i, j] * jnp.tensordot(B[:, j] - e_j + e_i, B[i, :] - e_j_T, 0)

        i, j = arg_absmax(B)

        steps += 1
        if steps == jnp.power(n, r + 1):
            raise Exception("computation limit exceeded")

    return row_inds[0:r]


def naive_maxvol(matrix: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
    """
    This function implements the maxvol algorithm in a naive way,
    using only jnp.linalg module functions for matrix operations

    Args:
        matrix: 2-dimensional array
        tol: tolerance, the stopping criterion in iterations

    Returns:
        The list of row indices forming the quasi- maximum volume submatrix.
    """

    A = copy.deepcopy(matrix)

    if A.shape[0] < A.shape[1]:
        raise Exception("Condition on Matrix dimensions is not satisfied")

    n, r = A.shape
    steps = 0

    row_inds = jnp.array(range(n))

    B = jnp.tensordot(A, jnp.linalg.inv(A[0:r]), 1)
    i, j = arg_absmax(B)

    while not abs(B[i][j]) < 1. + tol:

        buff = row_inds[j]
        row_inds = row_inds.at[j].set(row_inds[i])
        row_inds = row_inds.at[i].set(buff)

        B = jnp.tensordot(A[row_inds], jnp.linalg.inv(A[row_inds][0:r]), 1)
        i, j = arg_absmax(B)

        steps += 1
        if steps == jnp.power(n, r + 1):
            raise Exception("computation limit exceed")

    return row_inds[0:r]
