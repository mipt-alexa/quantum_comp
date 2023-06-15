from classes import MPO
import jax.numpy as jnp
import numpy as np


def P_tensor(theta: float, bond_1_dim: int, bond_2_dim: int, out_dim: int):
    """
    This function returns Phase gate as 4-dimentional tensor
    Args:
        theta: the phase
        bond_1_dim: first bond dimension (index 0)
        bond_2_dim: second bond dimension (index 2)
        out_dim: outer dimension (indices 1 and 3)
    """
    t = np.full((bond_1_dim, out_dim, bond_2_dim, out_dim), 0, dtype=complex)
    for i in range(out_dim):
        for j in range(min(bond_1_dim, bond_2_dim)):
            t[j, i, j, i] = np.exp(complex(0, theta * i * j))
    return jnp.array(t)


def QFT_4_qubit(bond_dim: int) -> MPO:
    """
    This function produces Quantum Fourier Transform operator for 4 qubits as MPO object.
    Args:
        bond_dim: upper bound on bond(inner) dimensions of tensor network
    Returns:
        QFT as MPO
    """

    H = 1/jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]])

    P_pi_8 = P_tensor(np.pi / 8, 2, 1, 2)
    P_pi_4_lower = P_tensor(np.pi / 4, 2, 1, 2)
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

    P_pi_4 = P_tensor(np.pi/4, 2, 2, 2)
    P_pi_2 = P_tensor(np.pi/2, 2, 2, 2)
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

    P_pi_2_lower = P_tensor(np.pi/2, 2, 1, 2)
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
    P_plus = P_tensor(theta, 2, 2, 2)
    P_minus = P_tensor(-theta, 2, 2, 2)
    # print(jnp.tensordot(P_plus, P_minus, [[3, 2, 1], [3, 0, 1]]))
    # print(jnp.tensordot(P_minus, P_plus, [[3, 2, 1], [3, 0, 1]]))

    # check for isometry property
    qft = QFT_4_qubit(3)
    comp = qft.components
    print(jnp.tensordot(comp[0], comp[0], [[1, 0, 3], [1, 0, 3]]))
    print(jnp.tensordot(jnp.conj(comp[1]), comp[1], [[1, 0, 3], [1, 0, 3]]))
    print(jnp.tensordot(comp[3], jnp.conj(comp[3]), [[1, 2, 3], [1, 2, 3]]))


if __name__ == "__main__":
    main()
