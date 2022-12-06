from classes import *


def get_tensor_from_MPS(x: MPS) -> jnp.ndarray:
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
    outer_dims = np.random.randint(1, max_dim, length)  # visible dimensions
    inner_dims = np.random.randint(1, max_dim, length - 1)  # bond dimensions

    comp = [np.random.randn(1, outer_dims[0], inner_dims[0])]  # filling the first components of mps
    for i in range(1, length - 1):
        comp.append(np.random.randn(inner_dims[i - 1], outer_dims[i], inner_dims[i]))
    comp.append(
        np.random.randn(inner_dims[-1], outer_dims[-1], 1))  # the last components of mps

    mps = MPS(comp)
    return mps
