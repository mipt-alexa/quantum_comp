import jax.numpy as jnp
from jax.scipy.linalg import solve_triangular, lu
import jax
import copy


def construct_B(U, matrix, L):
    Q = solve_triangular(jnp.transpose(U), jnp.transpose(matrix), lower=True)
    return jnp.transpose(solve_triangular(jnp.transpose(L[0:matrix.shape[1]]), Q, lower=False))


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

    if matrix.shape[0] < matrix.shape[1]:
        raise Exception("Condition on matrix dimensions is not satisfied")

    n, r = matrix.shape
    steps = 0

    lu_jit = jax.jit(lu)
    P, L, U = lu_jit(matrix)

    # Q = solve_triangular(jnp.transpose(U), jnp.transpose(matrix), lower=True)
    # B = jnp.transpose(solve_triangular(jnp.transpose(L[0:r]), Q, lower=False))

    construct_B_jit = jax.jit(construct_B)
    B = construct_B_jit(U, matrix, L)

    row_inds = jnp.tensordot(jnp.transpose(P), jnp.array(range(n)), 1)
    row_inds = jnp.asarray(row_inds, dtype=int)
    B = B[row_inds]

    i, j = jnp.unravel_index(jnp.argmax(jnp.abs(B)), B.shape) # 2d-indices of absolute maximum

    e = jnp.identity(n)
    e_r = jnp.identity(r)

    while not abs(B[i][j]) < 1. + tol:

        buff = row_inds[j]
        row_inds = row_inds.at[j].set(row_inds[i])
        row_inds = row_inds.at[i].set(buff)

        B -= 1 / B[i, j] * jnp.tensordot(B[:, j] - e[j] + e[i], B[i, :] - e_r[j], 0)

        i, j = jnp.unravel_index(jnp.argmax(jnp.abs(B)), B.shape)

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
