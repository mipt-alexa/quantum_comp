from classes import MPO
from TT_cross import TT_cross
import jax.numpy as jnp
import numpy as np


def P_tensor(theta: float, bond_1_dim: int, bond_2_dim: int, out_dim: int):
    t = np.full((bond_1_dim, out_dim, bond_2_dim, out_dim), 0, dtype=complex)
    for i in range(out_dim):
        for j in range(min(bond_1_dim, bond_2_dim)):
            t[j, i, j, i] = np.exp(complex(0, theta * i * j))
    return jnp.array(t)

# check isometry

theta = 1.34
P_plus = P_tensor(theta, 2, 2, 2)
P_minus = P_tensor(-theta, 2, 2, 2)
# print(jnp.tensordot(P_plus, P_minus, [[3, 2, 1], [3, 0, 1]]))
# print(jnp.tensordot(P_minus, P_plus, [[3, 2, 1], [3, 0, 1]]))
#


def QFT_4_qubit() -> MPO:

    H = 1/jnp.sqrt(2) * jnp.array([[1, 1], [1, -1]])

    P_pi_8 = P_tensor(np.pi / 8, 2, 1, 2)
    P_pi_4_lower = P_tensor(np.pi / 4, 2, 1, 2)
    subresult_1 = jnp.reshape(jnp.tensordot(P_pi_8, P_pi_4_lower, [3, 1]), (2, 2, 2, 2))

    # step 1

    a = jnp.reshape(subresult_1, (4, 4))

    u, s, v = jnp.linalg.svd(a, full_matrices=False)
    s_matrix = np.zeros((u.shape[1], v.shape[0]))
    np.fill_diagonal(s_matrix, s)

    S_U_4 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (2, 2, s_matrix.shape[1]))
    V_4 = jnp.reshape(v, (s_matrix.shape[1], 2, 1, 2))

    # step 1.2

    P_pi_4 = P_tensor(np.pi/4, 2, 2, 2)
    P_pi_2 = P_tensor(np.pi/2, 2, 2, 2)
    subresult_2 = jnp.tensordot(P_pi_4, P_pi_2, [[3], [1]])
    subresult_2 = jnp.tensordot(subresult_2, S_U_4, [[2, 3], [0, 1]])

    # step 1.3

    subresult_3 = jnp.reshape(subresult_2, (4, 16))

    u, s, v = jnp.linalg.svd(subresult_3, full_matrices=False)
    s_matrix = np.zeros((u.shape[1], v.shape[0]))
    np.fill_diagonal(s_matrix, s)

    S_U_3 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (2, s_matrix.shape[1], 2))
    V_3 = jnp.reshape(v, (s_matrix.shape[1], 2, 4, 2))

    # step 1.4

    id_tensor = np.zeros((2,) * 3)
    id_tensor[tuple([np.arange(2)] * 3)] = 1
    id_tensor = jnp.array(id_tensor)

    U_1 = jnp.reshape(jnp.tensordot(H, id_tensor, 1), (1, 2, 2, 2))

    subresult = jnp.tensordot(jnp.tensordot(P_pi_2, H, 1), id_tensor, 1)

    T_2 = jnp.tensordot(subresult, S_U_3, [[2, 3], [0, 2]])

    # step 2

    subresult = jnp.reshape(T_2, (8, 4))

    u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
    s_matrix = np.zeros((u.shape[1], v.shape[0]))
    np.fill_diagonal(s_matrix, s)

    U_2 = jnp.reshape(u, (2, 2, s_matrix.shape[0], 2))

    S_V_2 = jnp.reshape(jnp.tensordot(s_matrix, v, 1), (s_matrix.shape[0], 4))

    # step 2.1

    subresult = jnp.reshape(jnp.tensordot(S_V_2, V_3, [1, 0]), (U_2.shape[2] * 2 * 2, V_4.shape[0]))

    u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
    s_matrix = np.zeros((u.shape[1], v.shape[0]))
    np.fill_diagonal(s_matrix, s)

    U_3 = jnp.reshape(u, (U_2.shape[2], 2, s_matrix.shape[0], 2))

    T_4 = jnp.tensordot(jnp.tensordot(s_matrix, v, 1), V_4, [1, 0])

    # step 3

    P_pi_2_lower = P_tensor(np.pi/2, 2, 1, 2)
    subresult = jnp.tensordot(jnp.tensordot(T_4, P_pi_2_lower, [3, 1]), H, 1)
    subresult = jnp.reshape(subresult, (U_3.shape[2] * 2, 2 * 2))

    u, s, v = jnp.linalg.svd(subresult, full_matrices=False)
    s_matrix = np.zeros((u.shape[1], v.shape[0]))
    np.fill_diagonal(s_matrix, s)

    U_S_4 = jnp.reshape(jnp.tensordot(u, s_matrix, 1), (U_3.shape[2], s_matrix.shape[1], 2))
    V_4 = jnp.reshape(v, (s_matrix.shape[1], 2, 1, 2))

    subresult = jnp.tensordot(jnp.tensordot(U_3, H, 1), id_tensor, 1)

    T_3 = jnp.tensordot(subresult, U_S_4, [[2, 3], [0, 2]])

    return MPO([U_1, U_2, T_3, V_4])


# check for isometry property
qft = QFT_4_qubit()
comp = qft.components
print(jnp.tensordot(comp[0], comp[0], [[1, 0, 3], [1, 0, 3]]))
print(jnp.tensordot(jnp.conj(comp[1]), comp[1], [[1, 0, 3], [1, 0, 3]]))
print(jnp.tensordot(comp[3], jnp.conj(comp[3]), [[1, 2, 3], [1, 2, 3]]))


