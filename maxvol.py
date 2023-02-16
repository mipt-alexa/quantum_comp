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
    This function implements the maxvol algorithm by using PLU decomposition
    on the first step and Sherman-Woodbury-Morrison formula
    for matrix inverse on the following steps

    Args:
        matrix: 2-dimensional array
        tol: tolerance, the stopping criterion in iterations

    Returns:
        The maximum volume submatrix.
    """

    A = copy.deepcopy(matrix)

    if A.shape[0] < A.shape[1]:
        raise Exception("Condition on matrix dimensions is not satisfied")

    n, r = A.shape
    steps = 0

    P, L, U = lu(A)

    Q = solve_triangular(jnp.transpose(U), jnp.transpose(A), lower=True)
    B = jnp.transpose(solve_triangular(jnp.transpose(L[0:r]), Q, lower=False))

    i, j = arg_absmax(B)

    A = jnp.tensordot(jnp.transpose(P), A, 1)

    while not abs(B[i][j]) < 1. + tol:
        buff = A[i, :]
        A = A.at[i, :].set(A[j, :])
        A = A.at[j, :].set(buff)

        e_j_T = jnp.zeros(r)
        e_j_T = e_j_T.at[j].set(1)

        B -= 1 / B[i][j] * jnp.tensordot(B[:, j], B[i, :] - e_j_T, 0)

        i, j = arg_absmax(B)

        steps += 1
        if steps == jnp.power(n, r + 1):
            raise Exception("computation limit exceeded")

    return A[0:r]


def naive_maxvol(matrix: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
    """
    This function implements the maxvol algorithm in a naive way,
    using only jnp.linalg module functions for matrix inverse

    Args:
        matrix: 2-dimensional array
        tol: tolerance, the stopping criterion in iterations

    Returns:
        The maximum volume submatrix.
    """

    A = copy.deepcopy(matrix)

    if A.shape[0] < A.shape[1]:
        raise Exception("Condition on Matrix dimensions is not satisfied")

    n, r = A.shape
    steps = 0

    submatrix = A[0:r]
    B = jnp.tensordot(A, jnp.linalg.inv(submatrix), 1)

    i, j = arg_absmax(B)

    while not abs(B[i][j]) < 1. + tol:

        buff = A[i, :]
        A = A.at[i, :].set(A[j, :])
        A = A.at[j, :].set(buff)

        submatrix = A[0:r]
        B = jnp.tensordot(A, jnp.linalg.inv(submatrix), 1)
        i, j = arg_absmax(B)

        steps += 1
        if steps == jnp.power(n, r + 1):
            raise Exception("computation limit exceed")

    return submatrix
