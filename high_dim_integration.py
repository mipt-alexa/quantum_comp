from TT_cross import TT_cross
import jax.numpy as jnp
import numpy as np

N = 11
D = 50
POWERS = jnp.array([jnp.power(2., -n) for n in range(1, N+1)], dtype=jnp.float64)


def sin(ind: jnp.array) -> jnp.array:
    return jnp.sin(jnp.dot(ind, jnp.tile(POWERS, D)))


def grid(ind: jnp.array) -> jnp.array:
    # return jnp.full(len(ind), jnp.power(2., -N*D), dtype=jnp.float64)
    return jnp.full(len(ind), 1., dtype=jnp.float64)

def main():
    out_dims = [2] * N*D
    bond_dims = [2] * (N*D - 1)
    w_bond_dims = [1] * (N*D - 1)
    sin_mps = TT_cross(sin, out_dims, bond_dims, max_iter=10)
    weight_mps = TT_cross(grid, out_dims, w_bond_dims, max_iter=10)

    a1 = sin_mps.dot(weight_mps) * jnp.power(2., -N*D)
    a2 = np.imag(np.power((np.exp(complex(0, 1)) - 1) / complex(0, 1), D))
    print(a1, a2, a1/a2)


if __name__ == "__main__":
    main()
