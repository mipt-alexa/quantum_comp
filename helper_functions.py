from classes import MPS
import numpy as np
import jax.numpy as jnp


def get_tensor_from_MPS(x: MPS) -> jnp.ndarray:
    """
    This function produces the tensor form of an MPS with shape (1, outer_dimensions, 1).
    The cause for first and last dimensions to be 1 is consistency with MPS object structure.
    """
    result = x.components[0]
    for i in range(1, x.len):
        result = jnp.tensordot(result, x.components[i], 1)
    return result


def get_norm_from_MPS(x: MPS) -> float:
    result = x.components[0]
    for i in range(1, x.len):
        result = jnp.tensordot(result, x.components[i], 1)
    return np.sqrt(jnp.tensordot(result, result, x.len + 2))


def create_mps(length: int, max_dim: int) -> MPS:
    """
    This function creates MPS of given length with dimensions not exceeding the given maximum dimension,
    and entries of tensors being random variables having the standard normal distribution.
    """
    outer_dims = np.random.randint(1, max_dim, length)  # visible dimensions
    inner_dims = np.random.randint(1, max_dim, length - 1)  # bond dimensions

    comp = [np.random.randn(1, outer_dims[0], inner_dims[0])]  # filling the first components of mps
    for i in range(1, length - 1):
        comp.append(np.random.randn(inner_dims[i - 1], outer_dims[i], inner_dims[i]))
    comp.append(
        np.random.randn(inner_dims[-1], outer_dims[-1], 1))  # the last components of mps

    mps = MPS(comp)
    return mps


def unfold_matrix_from_mps(mps: MPS, part_index: int) -> jnp.ndarray:
    """
    This function performs conversion of MPS into tensor and then an unfolding 2-D matrix,
    where part_index parameter specifies which of the tensor indices would be merged
    A_i = A(0, ..., part_index; part_index + 1, ... )
    """

    if part_index >= mps.len - 1:
        raise Exception("Partition index exceed the MPS length")

    tensor = mps.components[0]
    shape_2d = [mps.components[0].shape[1], 1]
    for i in range(1, mps.len):
        if i <= part_index:
            shape_2d[0] *= mps.components[i].shape[1]
        else:
            shape_2d[1] *= mps.components[i].shape[1]

        tensor = jnp.tensordot(tensor, mps.components[i], 1)

    return jnp.reshape(tensor, shape_2d)
