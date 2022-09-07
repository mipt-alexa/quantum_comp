import jax.numpy as jnp
import numpy as np
from typing import List

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


class MPS:
    def __init__(self, tensors: List[jnp.ndarray]):
        """
        Args:
            tensors: The list of tensors in which the MPS is decomposed. Each tensor is expected to have 3 indexes,
                     for the first and the last ones corresponding dimensions are expected to be 1.
        """
        self.components = tensors
        self.len = len(tensors)

    def __repr__(self):
        """
        This special method represents the MPS object by the shape and
        elements of its components.

        """
        rep = 'MPS( '
        for i in range(self.len):
            rep += "component " + str(i) + " of size " + str(self.components[i].shape) + '\n'
                   # + str(self.components[i])
        return rep + ')'

    def dot(self, rhs):
        """
        This function computes scalar product of two MPS.

        Args:
            rhs (MPS): The second multiplier.

        Returns:
            jaxlib.xla_extension.DeviceArray: The return value.

        """
        if self.len != rhs.len:
            raise Exception("Ranks do not match")

        result = jnp.tensordot(self.components[0], rhs.components[0], [[1], [1]])
        for i in range(1, self.len):
            if i == 1:
                result = jnp.tensordot(result, self.components[i], [[1], [0]])
            else:
                result = jnp.tensordot(result, self.components[i], [[2], [0]])

            result = jnp.tensordot(result, rhs.components[i], [[2, 3], [0, 1]])
        return result

    def left_canonical(self):
        """
        This function performs decomposition of MPS to the left canonical form in place.
        """
        for i in range(self.len-1):
            shape = self.components[i].shape
            a = jnp.reshape(self.components[i], (shape[0]*shape[1], shape[2]))

            u, s, v = jnp.linalg.svd(a)
            s_matrix = np.zeros((u.shape[0], v.shape[0]))
            np.fill_diagonal(s_matrix, s)

            u = jnp.reshape(u, (shape[0], shape[1], u.shape[1]))
            rhs = jnp.tensordot(s_matrix, v, 1)
            self.components[i] = u
            self.components[i+1] = jnp.tensordot(rhs, self.components[i+1], 1)

    def right_canonical(self):
        """
        This function performs decomposition of MPS to the right canonical form in place.
        """
        for i in range(self.len-1, 0, -1):
            print(i)
            shape = self.components[i].shape
            a = jnp.reshape(self.components[i], (shape[0], shape[1]*shape[2]))

            u, s, v = jnp.linalg.svd(a)
            s_matrix = np.zeros((u.shape[0], v.shape[0]))
            np.fill_diagonal(s_matrix, s)

            v = jnp.reshape(v, (v.shape[0], shape[1], shape[2]))
            lhs = jnp.tensordot(u, s_matrix, 1)
            self.components[i] = v
            self.components[i-1] = jnp.tensordot(self.components[i-1], lhs, 1)


class MPO:
    def __init__(self, tensors: List[jnp.ndarray]):
        """
        Args:
            tensors: The list of tensors in which the MPO is decomposed. Each tensor is expected to have 4 indexes,
                     for the first and the last ones corresponding dimensions are expected to be 1.
        """
        self.components = tensors
        self.len = len(tensors)

    def __repr__(self):
        """
        This special method represents the MPO object by the shape and
        elements of its components.

        """
        rep = 'MPO( '
        for i in range(self.len):
            rep += "component " + str(i) + " of size " + str(self.components[i].shape) + '\n' + str(
                self.components[i]) + '\n'
        return rep + ')'

    def __call__(self, state: MPS):
        """ This special method implements the action of operator onto the MPS in place.

        Args:
            state: The MPS to which the operator is applied.

        """
        if self.len != state.len:
            raise Exception("Ranks of the MPO and MPS do not match")

        for i in range(0, self.len):
            state.components[i] = jnp.tensordot(self.components[i], state.components[i], [[1], [1]])
            shape = state.components[i].shape
            state.components[i] = jnp.reshape(state.components[i], (shape[0]*shape[3], shape[2], shape[1]*shape[4]))
