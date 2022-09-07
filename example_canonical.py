from classes import *

from jax import random

keys = random.split(random.PRNGKey(42), num=6)

inner_dim_mps = 3
outer_dim = 2

first_mps = random.randint(keys[0], (1, outer_dim, inner_dim_mps), minval=0, maxval=4)
middle_mps = random.randint(keys[1], (inner_dim_mps, outer_dim, inner_dim_mps), minval=0, maxval=4)
last_mps = random.randint(keys[2], (inner_dim_mps, outer_dim, 1), minval=0, maxval=4)

s = MPS([first_mps, middle_mps, middle_mps, last_mps])

s.right_canonical()
print(repr(s))
s.left_canonical()
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
