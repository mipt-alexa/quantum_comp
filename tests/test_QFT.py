from typing import Tuple

import jax
import unittest
import jax.numpy as jnp
import numpy as np

from QFT import QFT

jax.config.update("jax_enable_x64", True)


def reverse_operation(order: int):
    """
    This function computes the matrix form of bit-reverse operator, that is the final stage of QFT.
    Args:
        order: number of qubits
    Returns:
        matrix of size (2**order, 2**order)
    """
    N = np.power(2, order)
    rev_matrx = []
    format_str = '{0:0' + str(order) + 'b}'
    for i in range(0, order//2):
        rev_step = np.zeros((N, N), dtype=int)

        for j in range(N):
            s = list(format_str.format(j))
            s[i], s[-i - 1] = s[-i - 1], s[i]
            swapped_j = int(''.join(s), 2)
            rev_step[j][swapped_j] = 1

        rev_matrx.append(rev_step)

    total = rev_matrx[0]
    for r in rev_matrx[1:]:
        total = np.tensordot(total, r, 1)
    return total


def perm(n: int) -> Tuple[int, ...]:
    """
    This function returns the permutation of indices for converting the MPO into the matrix form.
    Arguments:
         n: number of qubits
    Returns:
        permutation as a tuple of size n
    """
    permutation = np.empty(2*(n+1), dtype=int)
    permutation[0] = 0
    for j in range(1, n + 1):
        permutation[j] = 2*j - 1
        permutation[n + j] = 2*j
    permutation[-1] = 2*n + 1
    return tuple(permutation)


class TestMaxVol(unittest.TestCase):
    def test_isometry(self, max_ord=9, bond_dim=2):

        for n in range(2, max_ord):
            QFT_n = QFT(n, bond_dim)

            expected = jnp.identity(bond_dim)
            for i in range(n - 1):
                result = jnp.tensordot(jnp.conj(QFT_n.components[i]), QFT_n.components[i], [[1, 0, 3], [1, 0, 3]])
                self.assertTrue(jnp.allclose(result, expected, 1e-6))

    def test_transform(self, max_ord = 7):

        for n in range(2, max_ord):
            N = 2**n

            qft = QFT(n, 2**(n-1))
            matrix_qft = qft.components[0]
            for i in range(1, n):
                matrix_qft = jnp.tensordot(matrix_qft, qft.components[i], [2*i, 0])

            result = jnp.tensordot(jnp.reshape(jnp.transpose(matrix_qft, perm(n)), (N, N)), reverse_operation(n), 1)
            result = np.sqrt(N) * result

            expected = np.empty((N, N), dtype=complex)
            theta = 2 * np.pi / N
            for i in range(N):
                for j in range(N):
                    expected[i][j] = np.exp(complex(0, 1) * theta * i * j)

            print(n)
            # print(n, round(result - expected, 6))
            self.assertTrue(jnp.allclose(result, expected, 1e-6))


unittest.main(argv=[''], verbosity=1, exit=True)
