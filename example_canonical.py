from classes import *
from jax import random


def get_norm_from_MPS(x: MPS) -> float:
    result = x.components[0]
    for i in range(1, x.len):
        result = jnp.tensordot(result, x.components[i], 1)

    return jnp.tensordot(result, result, x.len + 2)


keys = random.split(random.PRNGKey(48012), num=6)

inner_dim_mps = 3
outer_dim = 4

first = random.normal(keys[0], (1, outer_dim, inner_dim_mps))
middle = random.normal(keys[1], (inner_dim_mps, outer_dim, inner_dim_mps))
last = random.normal(keys[2], (inner_dim_mps, outer_dim, 1))

s = MPS([first, middle, last])

print(get_norm_from_MPS(s))

s.right_canonical()
print(repr(s))
s.left_canonical()
print(repr(s))

print(get_norm_from_MPS(s))
s.left_svd_trunc(1)
print(get_norm_from_MPS(s))
