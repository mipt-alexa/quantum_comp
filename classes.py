import jax.numpy as jnp
import jax.config
import numpy as np
import numpy.linalg as la

from typing import List

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
jax.config.update("jax_enable_x64", True)


def truncate(arr: jnp.ndarray, tol: float) -> jnp.ndarray:
    """ 
    This function performs truncation of the n smallest non-zero values in an array.

    Args:
        arr: the 1-d array of non-negative numbers, which is sorted in descending order
        tol: allowed value of truncating L2 norm

    Returns:
        A truncated array.
    """
    init_norm = la.norm(arr)

    if init_norm < tol:
        raise Exception("Truncation value exceed initial norm")

    ind = len(arr) - 1
    tail_sq_norm = 0

    while ind > 0:
        if np.sum(np.power(arr[ind:],2)) > tol*(2*init_norm - tol):
            break
        else:
            ind -= 1

    return np.delete(arr, np.s_[ind + 1:])


class MPS:
    def __init__(self, tensors: List[jnp.ndarray]):
        """
        Args:
            tensors: The list of tensors in which the MPS is decomposed. Each tensor is expected to have 3 indices,
                     for the first and the last ones corresponding dimensions are expected to be equal to 1.
        """
        self.components = tensors
        self.len = len(tensors)

    def __repr__(self):
        """
        This special method represents the MPS object by the shape and elements of its components.
        """
        rep = 'MPS( '
        for i in range(self.len):
            rep += "component " + str(i) + " of size " + str(self.components[i].shape) + '\n'
            # + str(self.components[i])
        return rep + ')'

    def __sub__(self, other: "MPS") -> "MPS":
        """Produce element-wise difference between 2 MPS objects """
        if self.len != other.len:
            raise Exception("MPS lengths are not equal")

        for i in range(self.len):
            if self.components[i].shape != other.components[i].shape:
                raise Exception("MPS components' sizes by index " + str(i) + "are not equal")

        tensors = []
        for i in range(self.len):
            tensors.append(self.components[i] - other.components[i])

        return MPS(tensors)

    def dot(self, rhs) -> jnp.float64:
        """
        This function computes scalar product of two MPS.

        Args:
            rhs (MPS): The second multiplier.

        Returns:
            The scalar product value.

        """
        if self.len != rhs.len:
            raise Exception("Ranks of multipliers do not match")

        result = jnp.tensordot(self.components[0], rhs.components[0], [[1], [1]])
        for i in range(1, self.len):
            if i == 1:
                result = jnp.tensordot(result, self.components[i], [[1], [0]])
            else:
                result = jnp.tensordot(result, self.components[i], [[2], [0]])

            result = jnp.tensordot(result, rhs.components[i], [[2, 3], [0, 1]])
        return jnp.reshape(result, 1)[0]

    def norm(self) -> jnp.float64:
        """
        This function computes L2 norm of MPS
        Returns:
            L2 norm of MPS
        """
        return jnp.sqrt(self.dot(self))

    def left_canonical(self) -> None:
        """
        This function performs decomposition of MPS to the left canonical form in place.
        """
        for i in range(self.len - 1):
            shape = self.components[i].shape
            a = jnp.reshape(self.components[i], (shape[0] * shape[1], shape[2]))

            u, s, v = jnp.linalg.svd(a, full_matrices=False)
            s_matrix = np.zeros((u.shape[1], v.shape[0]))
            np.fill_diagonal(s_matrix, s)

            u = jnp.reshape(u, (shape[0], shape[1], u.shape[1]))
            rhs = jnp.tensordot(s_matrix, v, 1)
            self.components[i] = u
            self.components[i + 1] = jnp.tensordot(rhs, self.components[i + 1], 1)

    def right_canonical(self) -> None:
        """
        This function performs decomposition of MPS to the right canonical form in place.
        """
        for i in range(self.len - 1, 0, -1):
            shape = self.components[i].shape
            a = jnp.reshape(self.components[i], (shape[0], shape[1] * shape[2]))

            u, s, v = jnp.linalg.svd(a, full_matrices=False)
            s_matrix = np.zeros((u.shape[1], v.shape[0]))
            np.fill_diagonal(s_matrix, s)

            v = jnp.reshape(v, (v.shape[0], shape[1], shape[2]))
            lhs = jnp.tensordot(u, s_matrix, 1)

            self.components[i] = v
            self.components[i - 1] = jnp.tensordot(self.components[i - 1], lhs, 1)

    def left_svd_trunc(self, epsilon: float) -> None:
        """
        This method performs svd-truncation of n non-zero singular values of MPS in left canonical form in place.

        Args:
            epsilon: truncation value of L1 MPS norm
        """
        a = self.components[self.len - 1]

        shape = a.shape
        u, s, v = jnp.linalg.svd(jnp.reshape(a, (shape[0], shape[1])), full_matrices=False)

        s = truncate(s, epsilon)
        s_matrix = np.zeros((len(s), len(s)))
        np.fill_diagonal(s_matrix, s)

        u = u[:, :len(s)]
        v = v[:len(s)]

        a = jnp.tensordot(u, jnp.tensordot(s_matrix, v, 1), 1)
        self.components[self.len - 1] = jnp.reshape(a, shape)


class MPO:
    def __init__(self, tensors: List[jnp.ndarray]):
        """
        Args:
            tensors: The list of tensors in which the MPO is decomposed. Each tensor is expected to have 4 indices,
                     for the first and the last ones corresponding dimensions are expected to be equal to 1.
        """
        self.components = tensors
        self.len = len(tensors)

    def __repr__(self):
        """
        This special method represents the MPO object by the shape and elements of its components.
        """
        rep = 'MPO( '
        for i in range(self.len):
            rep += "component " + str(i) + " of size " + str(self.components[i].shape) + '\n' + str(
                self.components[i]) + '\n'
        return rep + ')'

    def process(self, state: MPS) -> MPS:
        """
        This special method implements the action of operator onto the MPS.

        Args:
            state: The MPS to which the operator is applied.
        """
        if self.len != state.len:
            raise Exception("Ranks of the MPO and MPS do not match")

        components = []

        for i in range(0, self.len):
            state_shape = state.components[i].shape
            oper_shape = self.components[i].shape
            components.append(jnp.reshape(jnp.tensordot(self.components[i], state.components[i], [[1], [1]]),
                                        (oper_shape[0] * state_shape[0], state_shape[1], oper_shape[2] * state_shape[2])))
        return MPS(components)
