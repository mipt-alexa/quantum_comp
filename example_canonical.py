from classes import *
import jaxlib
from jax import random


keys = random.split(random.PRNGKey(48012), num=6)

inner_dim_mps = 3
outer_dim = 4

first = random.normal(keys[0], (1, outer_dim, inner_dim_mps))
middle = random.normal(keys[1], (inner_dim_mps, outer_dim, inner_dim_mps))
last = random.normal(keys[2], (inner_dim_mps, outer_dim, 1))

s = MPS([first, middle, middle, middle, middle, last])

s.right_canonical()
print(repr(s))
s.left_canonical()
print(repr(s))

s.left_svd_trunc(1)
print(repr(s))

# A = np.zeros((4, 5))
# A[0][0] = 1
# A[0][4] = 2
# A[1][2] = 3
# A[3][1] = 2
# print(A)
#
# u, s, v = jnp.linalg.svd(A)
# print(u)
# print(jnp.tensordot(u, np.transpose(u), 1))
