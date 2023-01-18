import jax.numpy as jnp
import copy


def arg_absmax(arr: jnp.ndarray) -> tuple:
    """
    This function returns the absolute maximum among 2d-array entries
    """
    amax = jnp.argmax(arr)
    amin = jnp.argmin(arr)
    if jnp.ravel(arr)[amax] > -jnp.ravel(arr)[amin]:
        i, j = amax // arr.shape[1], amax % arr.shape[1]
    else:
        i, j = amin // arr.shape[1], amin % arr.shape[1]
    return i, j


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

        A[[j, i], ] = A[[i, j], ]

        submatrix = A[0:r]
        B = jnp.tensordot(A, jnp.linalg.inv(submatrix), 1)
        i, j = arg_absmax(B)

        steps += 1
        if steps == jnp.power(n, r + 1):
            raise Exception("computation limit exceed")

    return submatrix


def maxvol_SWM(matrix: jnp.ndarray, tol: float = 1e-3) -> jnp.ndarray:
    """
    This function implements the maxvol algorithm in a somewhat efficient way
    by using Sherman-Woodbury-Morrison formula for matrix inverse

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
        A[[j, i], ] = A[[i, j], ]

        submatrix = A[0:r]

        e_i, e_j, e_j_T = jnp.zeros(n), jnp.zeros(n), jnp.zeros(r)
        e_i = e_i.at[i].set(1)
        e_j = e_j.at[j].set(1)
        e_j_T = e_j_T.at[j].set(1)

        B -= 1 / B[i][j] * jnp.tensordot(B[:, j] - e_j + e_i, B[i, :] - e_j_T, 0)

        i, j = arg_absmax(B)

        steps += 1
        if steps == jnp.power(n, r + 1):
            raise Exception("computation limit exceed")

    return submatrix
