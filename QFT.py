from typing import Tuple

from classes import MPO
import jax.numpy as jnp
import numpy as np
import jax

jax.config.update("jax_enable_x64", True)


def Phase_ord_4(theta: float, bond_dim: int, out_dim: int) -> jnp.ndarray:
    """
    This function returns Phase gate as 4-dimensional tensor
    Args:
        theta: the phase
        bond_dim: the size of bond dimensions (indices 0 and 2)
        out_dim: the size of outer dimensions (indices 1 and 3)
    Returns:
        tensor of shape (bond_dim, out_dim, bond_dim, out_dim)
    """
    t = np.full((bond_dim, out_dim, bond_dim, out_dim), 0, dtype=complex)
    for i in range(out_dim):
        for j in range(min(bond_dim, bond_dim)):
            t[j, i, j, i] = np.exp(complex(0, theta * i * j))
    return jnp.array(t, dtype=jnp.complex128)


def Phase_ord_3(theta: float, bond_1_dim: int, out_dim: int) -> jnp.ndarray:
    """
    This function returns order-3 Phase gate as 4-dimensional tensor
    with index 2 being trivial, for ex. shaped (2, 2, 1, 2)
    Args:
        theta: the phase
        bond_1_dim: first bond dimension (index 0)
        out_dim: outer dimension (indices 1 and 3)
    Returns:
        tensor of shape (bond_1_dim, out_dim, 1, out_dim)
    """
    t = np.full((bond_1_dim, out_dim, 1, out_dim), 0, dtype=complex)

    for out_ind in range(out_dim):
        for bond_ind_1 in range(bond_1_dim):
            t[bond_ind_1, out_ind, 0, out_ind] = np.exp(complex(0, theta * bond_ind_1 * out_ind))
    return jnp.array(t, dtype=jnp.complex128)


def SVD(matrix: jnp.ndarray, bond_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    This function performs SVD decomposition with truncation, maintaining the given number of singular values. The
    shape of S matrix is (bond_dim, bond_dim) even in the case when bond_dim exceeds the number of non-zero singular
    values.
    Args:
        matrix: 2d tensor to be decomposed
        bond_dim: number of singular values left after truncation
    Returns:
        matrices U, S(singular values on the diagonal), V
    """
    u, s, v = jnp.linalg.svd(matrix, full_matrices=False)

    s = s[:bond_dim]
    s_matrix = np.zeros((bond_dim, bond_dim))
    np.fill_diagonal(s_matrix, s)

    if u.shape[1] < bond_dim:
        u = jnp.hstack((u, jnp.zeros((u.shape[0], bond_dim - u.shape[1]))))
    else:
        u = u[:, :bond_dim]
    if v.shape[0] < bond_dim:
        v = jnp.vstack((v, jnp.zeros((bond_dim - v.shape[0], v.shape[1]))))
    else:
        v = v[:bond_dim]
    return u, jnp.array(s_matrix), v


def QFT(order: int, bond_dim: int) -> MPO:
    """
    ! This function produces QFT WITHOUT the reversal operation !

    This function computes the Quantum Fourier Transform Operator for a given number of qubits.
    During computational steps, all horizontal indices are assumed to be of size 2, vertical -- of bond_dim size
    (with exception to the trivial indices for the first and last MPO components)
    Args:
        order: number of qubits
        bond_dim: bond dimension
    Returns:
        QFT operator as MPO object
    """
    H = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]], dtype=jnp.complex128)

    if order == 1:
        return MPO([jnp.reshape(H, (1, 2, 1, 2))])

    # identity tensor of shape (2, bond_dim, 2) for manipulations with Hadamard gate
    id_tensor = np.zeros((2, bond_dim, 2))
    id_tensor[tuple([np.arange(2)] * 3)] = 1
    id_tensor = jnp.array(id_tensor, dtype=jnp.complex128)

    components = [jnp.array([]) for _ in range(order)]

    # initialize the components array as tensors from the left column of the diagram
    components[0] = jnp.reshape(jnp.tensordot(H, id_tensor, 1), (1, 2, bond_dim, 2))
    for i in range(1, order - 1):
        components[i] = Phase_ord_4(np.pi / np.power(2, i), bond_dim, 2)
    components[-1] = Phase_ord_3(np.pi / np.power(2, order - 1), bond_dim, 2)

    if order == 2:
        components[-1] = jnp.tensordot(components[-1], H, 1)

    # variables for storing temporary elements
    S_U_tensor = jnp.array([])
    S_V_tensor = jnp.array([])

    # algorithm
    for i in range(1, order - 1):
        for j in range(order - 1, i - 1, -1):
            # for each row convolve current left tensor with its right neighbour

            if i == j:
                # case of Hadamard gate
                subresult = jnp.tensordot(jnp.tensordot(components[j], H, 1), id_tensor, 1)
                subresult = jnp.tensordot(subresult, S_U_tensor, [[2, 3], [0, 1]])
                components[j] = jnp.transpose(subresult, (0, 1, 3, 2))

            else:
                # case of Phase tensor
                if j < order - 1:
                    rhs = Phase_ord_4(np.pi / np.power(2, j - i), bond_dim, 2)
                else:
                    rhs = Phase_ord_3(np.pi / np.power(2, j - i), bond_dim, 2)
                    if i == order - 2:
                        # case of the last convolution in the last row
                        rhs = jnp.tensordot(rhs, H, 1)

                subresult = jnp.tensordot(components[j], rhs, [[3], [1]])

                if j == order - 1:
                    # there is no S_U tensor for convolution in the last row, only phase tensors

                    subresult = jnp.transpose(subresult, (0, 3, 2, 1, 4, 5))
                    subresult_1 = jnp.reshape(subresult, (bond_dim * bond_dim, 2 * 2))

                    u, s_matrix, v = SVD(subresult_1, bond_dim)
                    S_U_tensor = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (bond_dim, bond_dim, bond_dim))
                    components[j] = jnp.transpose(jnp.reshape(v, (bond_dim, 2, 2, 1)), (0, 1, 3, 2))

                else:
                    subresult = jnp.tensordot(subresult, S_U_tensor, [[2, 4], [0, 1]])

                    subresult = jnp.transpose(subresult, (0, 2, 1, 3, 4))
                    subresult_2 = jnp.reshape(subresult, (bond_dim * bond_dim, 2 * 2 * bond_dim))
                    u, s_matrix, v = SVD(subresult_2, bond_dim)

                    S_U_tensor = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (bond_dim, bond_dim, bond_dim))
                    components[j] = jnp.transpose(jnp.reshape(v, (bond_dim, 2, 2, bond_dim)), (0, 1, 3, 2))

        # then proceed with moving the orthogonality center down the diagram
        for k in range(i, order - 1):
            if k == i:
                # there is no SV tensor on the first step
                subresult = jnp.reshape(jnp.transpose(components[k], (0, 1, 3, 2)), (2 * bond_dim * 2, bond_dim))
            else:
                subresult = jnp.tensordot(S_V_tensor, components[k], [1, 0])
                subresult = jnp.reshape(jnp.transpose(subresult, (0, 1, 3, 2)), (bond_dim * 2 * 2, bond_dim))

            u, s_matrix, v = SVD(subresult, bond_dim)

            components[k] = jnp.transpose(jnp.reshape(u, (bond_dim, 2, 2, bond_dim)), (0, 1, 3, 2))
            S_V_tensor = jnp.reshape(jnp.tensordot(s_matrix, v, 1), (bond_dim, bond_dim))

        components[-1] = jnp.tensordot(S_V_tensor, components[-1], [1, 0])

    return MPO(components)

#
# def QFT_4_qubit(bond_dim: int) -> MPO:
#     """
#     This function produces Quantum Fourier Transform operator for 4 qubits as MPO object.
#     Args:
#         bond_dim: upper bound on bond(inner) dimensions of tensor network
#     Returns:
#         QFT as MPO
#     """
#
#     H = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]])
#
#     P_pi_8 = Phase_ord_3(np.pi / 8, 2, 2)
#     P_pi_4_lower = Phase_ord_3(np.pi / 4, 2, 2)
#     subresult_1 = jnp.reshape(jnp.tensordot(P_pi_8, P_pi_4_lower, [3, 1]), (2, 2, 2, 2))
#
#     # step 1
#
#     a = jnp.reshape(subresult_1, (4, 4))
#
#     u, s, v = jnp.linalg.svd(a, full_matrices=False)
#     s = s[:bond_dim]
#     s_matrix = np.zeros((bond_dim, bond_dim))
#     np.fill_diagonal(s_matrix, s)
#
#     u = u[:, :bond_dim]
#     v = v[:bond_dim]
#
#     S_U_4 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (2, 2, bond_dim))
#     V_4 = jnp.reshape(v, (bond_dim, 2, 1, 2))
#
#     # step 1.2
#
#     P_pi_4 = Phase_ord_4(np.pi / 4, 2, 2)
#     P_pi_2 = Phase_ord_4(np.pi / 2, 2, 2)
#     subresult_2 = jnp.tensordot(P_pi_4, P_pi_2, [[3], [1]])
#     subresult_2 = jnp.tensordot(subresult_2, S_U_4, [[2, 3], [0, 1]])
#
#     # step 1.3
#
#     subresult_3 = jnp.reshape(subresult_2, (4, 4 * bond_dim))
#
#     u, s, v = jnp.linalg.svd(subresult_3, full_matrices=False)
#     s = s[:bond_dim]
#     s_matrix = np.zeros((bond_dim, bond_dim))
#     np.fill_diagonal(s_matrix, s)
#
#     u = u[:, :bond_dim]
#     v = v[:bond_dim]
#
#     S_U_3 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (2, bond_dim, 2))
#     V_3 = jnp.reshape(v, (bond_dim, 2, bond_dim, 2))
#
#     # step 1.4
#
#     id_tensor = np.zeros((2,) * 3)
#     id_tensor[tuple([np.arange(2)] * 3)] = 1
#     id_tensor = jnp.array(id_tensor)
#
#     U_1 = jnp.reshape(jnp.tensordot(H, id_tensor, 1), (1, 2, 2, 2))
#
#     subresult = jnp.tensordot(jnp.tensordot(P_pi_2, H, 1), id_tensor, 1)
#
#     T_2 = jnp.tensordot(subresult, S_U_3, [[2, 3], [0, 2]])
#
#     # step 2
#
#     subresult = jnp.reshape(T_2, (2 * 2 * 2, bond_dim))
#
#     u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
#     s = s[:bond_dim]
#     s_matrix = np.zeros((len(s), len(s)))
#     np.fill_diagonal(s_matrix, s)
#
#     u = u[:, :len(s)]
#     v = v[:len(s)]
#
#     U_2 = jnp.reshape(u, (2, 2, bond_dim, 2))
#
#     S_V_2 = jnp.reshape(jnp.tensordot(s_matrix, v, 1), (bond_dim, bond_dim))
#
#     # step 2.1
#
#     subresult = jnp.reshape(jnp.tensordot(S_V_2, V_3, [1, 0]), (bond_dim * 2 * 2, bond_dim))
#
#     u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
#     s = s[:bond_dim]
#     s_matrix = np.zeros((bond_dim, bond_dim))
#     np.fill_diagonal(s_matrix, s)
#
#     u = u[:, :bond_dim]
#     v = v[:bond_dim]
#
#     U_3 = jnp.reshape(u, (bond_dim, 2, bond_dim, 2))
#
#     T_4 = jnp.tensordot(jnp.tensordot(s_matrix, v, 1), V_4, [1, 0])
#
#     # step 3
#
#     P_pi_2_lower = Phase_ord_3(np.pi / 2, 2, 2)
#     subresult = jnp.tensordot(jnp.tensordot(T_4, P_pi_2_lower, [3, 1]), H, 1)
#     subresult = jnp.reshape(subresult, (bond_dim * 2, 2 * 2))
#
#     u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
#     s = s[:bond_dim]
#     s_matrix = np.zeros((bond_dim, bond_dim))
#     np.fill_diagonal(s_matrix, s)
#
#     u = u[:, :bond_dim]
#     v = v[:bond_dim]
#
#     U_S_4 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (bond_dim, bond_dim, 2))
#     V_4 = jnp.reshape(v, (bond_dim, 2, 1, 2))
#
#     subresult = jnp.tensordot(jnp.tensordot(U_3, H, 1), id_tensor, 1)
#
#     T_3 = jnp.reshape(jnp.tensordot(subresult, U_S_4, [[2, 3], [0, 2]]), (bond_dim, 2, bond_dim, 2))
#
#     return MPO([U_1, U_2, T_3, V_4])
