from typing import Tuple, Any, Union

from jax.core import Tracer
from jaxlib.xla_extension import DeviceArrayBase

from numpy import ndarray

from classes import MPO
import jax.numpy as jnp
import numpy as np


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
    return jnp.array(t, dtype=jnp.complex64)


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
    return jnp.array(t, dtype=jnp.complex64)


def SVD(matrix: jnp.ndarray, bond_dim: int) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
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

    u = u[:, :bond_dim]
    v = v[:bond_dim]
    return u, jnp.array(s_matrix), v


def QFT(order: int, bond_dim: int) -> MPO:
    """
    ! This function produces QFT WITHOUT the reversal operation !

    During computational steps, all horizontal indices are assumed to be of size 2, vertical -- of bond_dim size
    (with exception to the trivial indices for the first and last MPO components)
    Args:
        order: number of qubits
        bond_dim: bond dimension
    Returns:
        QFT operator as MPO object
    """
    H = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]])

    if order == 1:
        return MPO([jnp.reshape(H, (1, 2, 1, 2))])

    id_tensor = np.zeros((2, bond_dim, 2))
    id_tensor[tuple([np.arange(2)] * 3)] = 1
    id_tensor = jnp.array(id_tensor)

    components = [jnp.array([]) for _ in range(order)]

    # initialize the components with tensors from left column of the diagram
    components[0] = jnp.reshape(jnp.tensordot(H, id_tensor, 1), (1, 2, bond_dim, 2))
    for i in range(1, order - 1):
        components[i] = Phase_ord_4(np.pi / np.power(2, i), bond_dim, 2)
    components[-1] = Phase_ord_3(np.pi / np.power(2, order - 1), bond_dim, 2)

    S_U_tensor = jnp.array([])
    S_V_tensor = jnp.array([])

    # algorithm
    for i in range(1, order - 1):
        for j in range(order - 1, i - 1, -1):
            # for each row convolve current left tensor with its right neighbour

            if i == j:
                # case of Hadamard gate
                subresult = jnp.tensordot(jnp.tensordot(components[j], H, 1), id_tensor, 1)
                components[j] = jnp.tensordot(subresult, S_U_tensor, [[2, 3], [0, 1]])

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

                    subresult_1 = jnp.reshape(subresult, (bond_dim * bond_dim, 2 * 2))

                    u, s_matrix, v = SVD(subresult_1, bond_dim)

                    S_U_tensor = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (bond_dim, bond_dim, bond_dim))
                    components[j] = jnp.reshape(v, (bond_dim, 2, 1, 2))

                else:
                    subresult = jnp.tensordot(subresult, S_U_tensor, [[2, 3], [0, 1]])

                    subresult_2 = jnp.reshape(subresult, (bond_dim * bond_dim, 2 * 2 * bond_dim))
                    u, s_matrix, v = SVD(subresult_2, bond_dim)

                    S_U_tensor = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (bond_dim, bond_dim, bond_dim))
                    components[j] = jnp.reshape(v, (bond_dim, 2, bond_dim, 2))

        # then proceed with moving the orthogonality center down the diagram
        for k in range(i, order - 1):
            if k == 1:
                subresult = jnp.reshape(components[k], (2 * bond_dim * 2, bond_dim))
            else:
                subresult = jnp.reshape(jnp.tensordot(S_V_tensor, components[k], [1, 0]), (bond_dim * 2 * 2, bond_dim))

            u, s_matrix, v = SVD(subresult, bond_dim)

            components[k] = jnp.reshape(u, (bond_dim, 2, bond_dim, 2))
            S_V_tensor = jnp.reshape(jnp.tensordot(s_matrix, v, 1), (bond_dim, bond_dim))

        components[-1] = jnp.tensordot(S_V_tensor, components[-1], [1, 0])

    return MPO(components)


def QFT_4_qubit(bond_dim: int) -> MPO:
    """
    This function produces Quantum Fourier Transform operator for 4 qubits as MPO object.
    Args:
        bond_dim: upper bound on bond(inner) dimensions of tensor network
    Returns:
        QFT as MPO
    """

    H = 1 / jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]])

    P_pi_8 = Phase_ord_3(np.pi / 8, 2, 2)
    P_pi_4_lower = Phase_ord_3(np.pi / 4, 2, 2)
    subresult_1 = jnp.reshape(jnp.tensordot(P_pi_8, P_pi_4_lower, [3, 1]), (2, 2, 2, 2))

    # step 1

    a = jnp.reshape(subresult_1, (4, 4))

    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    s = s[:bond_dim]
    s_matrix = np.zeros((bond_dim, bond_dim))
    np.fill_diagonal(s_matrix, s)

    u = u[:, :bond_dim]
    v = v[:bond_dim]

    S_U_4 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (2, 2, bond_dim))
    V_4 = jnp.reshape(v, (bond_dim, 2, 1, 2))

    # step 1.2

    P_pi_4 = Phase_ord_4(np.pi / 4, 2, 2)
    P_pi_2 = Phase_ord_4(np.pi / 2, 2, 2)
    subresult_2 = jnp.tensordot(P_pi_4, P_pi_2, [[3], [1]])
    subresult_2 = jnp.tensordot(subresult_2, S_U_4, [[2, 3], [0, 1]])

    # step 1.3

    subresult_3 = jnp.reshape(subresult_2, (4, 4 * bond_dim))

    u, s, v = jnp.linalg.svd(subresult_3, full_matrices=False)
    s = s[:bond_dim]
    s_matrix = np.zeros((bond_dim, bond_dim))
    np.fill_diagonal(s_matrix, s)

    u = u[:, :bond_dim]
    v = v[:bond_dim]

    S_U_3 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (2, bond_dim, 2))
    V_3 = jnp.reshape(v, (bond_dim, 2, bond_dim, 2))

    # step 1.4

    id_tensor = np.zeros((2,) * 3)
    id_tensor[tuple([np.arange(2)] * 3)] = 1
    id_tensor = jnp.array(id_tensor)

    U_1 = jnp.reshape(jnp.tensordot(H, id_tensor, 1), (1, 2, 2, 2))

    subresult = jnp.tensordot(jnp.tensordot(P_pi_2, H, 1), id_tensor, 1)

    T_2 = jnp.tensordot(subresult, S_U_3, [[2, 3], [0, 2]])

    # step 2

    subresult = jnp.reshape(T_2, (2 * 2 * 2, bond_dim))

    u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
    s = s[:bond_dim]
    s_matrix = np.zeros((len(s), len(s)))
    np.fill_diagonal(s_matrix, s)

    u = u[:, :len(s)]
    v = v[:len(s)]

    U_2 = jnp.reshape(u, (2, 2, bond_dim, 2))

    S_V_2 = jnp.reshape(jnp.tensordot(s_matrix, v, 1), (bond_dim, bond_dim))

    # step 2.1

    subresult = jnp.reshape(jnp.tensordot(S_V_2, V_3, [1, 0]), (bond_dim * 2 * 2, bond_dim))

    u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
    s = s[:bond_dim]
    s_matrix = np.zeros((bond_dim, bond_dim))
    np.fill_diagonal(s_matrix, s)

    u = u[:, :bond_dim]
    v = v[:bond_dim]

    U_3 = jnp.reshape(u, (bond_dim, 2, bond_dim, 2))

    T_4 = jnp.tensordot(jnp.tensordot(s_matrix, v, 1), V_4, [1, 0])

    # step 3

    P_pi_2_lower = Phase_ord_3(np.pi / 2, 2, 2)
    subresult = jnp.tensordot(jnp.tensordot(T_4, P_pi_2_lower, [3, 1]), H, 1)
    subresult = jnp.reshape(subresult, (bond_dim * 2, 2 * 2))

    u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
    s = s[:bond_dim]
    s_matrix = np.zeros((bond_dim, bond_dim))
    np.fill_diagonal(s_matrix, s)

    u = u[:, :bond_dim]
    v = v[:bond_dim]

    U_S_4 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (bond_dim, bond_dim, 2))
    V_4 = jnp.reshape(v, (bond_dim, 2, 1, 2))

    subresult = jnp.tensordot(jnp.tensordot(U_3, H, 1), id_tensor, 1)

    T_3 = jnp.reshape(jnp.tensordot(subresult, U_S_4, [[2, 3], [0, 2]]), (bond_dim, 2, bond_dim, 2))

    return MPO([U_1, U_2, T_3, V_4])


def main():
    # check isometry

    theta = 1.34
    P_plus = Phase_ord_4(theta, 4, 2)
    P_minus = Phase_ord_4(-theta, 4, 2)
    # print(jnp.tensordot(P_plus, P_minus, [[3, 2, 1], [3, 0, 1]]))
    # print(jnp.tensordot(P_minus, P_plus, [[3, 2, 1], [3, 0, 1]]))

    # check for isometry property
    # qft = QFT_4_qubit(3)
    # comp = qft.components
    # print(repr(qft))
    # print(jnp.tensordot(comp[0], comp[0], [[1, 0, 3], [1, 0, 3]]))
    # print(jnp.tensordot(jnp.conj(comp[1]), comp[1], [[1, 0, 3], [1, 0, 3]]))
    # print(jnp.tensordot(comp[3], jnp.conj(comp[3]), [[1, 2, 3], [1, 2, 3]]))

    my_qft = QFT(10, 2)
    print(repr(my_qft))


if __name__ == "__main__":
    main()
