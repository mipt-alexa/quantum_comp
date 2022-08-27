from classes import *
from jax import random

keys = random.split(random.PRNGKey(42), num=6)

inner_dim_mps = 3
outer_dim = 2
inner_dim_mpo = 2

first_mps = random.randint(keys[0], (1, outer_dim, inner_dim_mps), minval=0, maxval=4)
middle_mps = random.randint(keys[1], (inner_dim_mps, outer_dim, inner_dim_mps), minval=0, maxval=4)
last_mps = random.randint(keys[2], (inner_dim_mps, outer_dim, 1), minval=0, maxval=4)

first_mpo = random.randint(keys[3], (1, outer_dim, inner_dim_mpo, outer_dim), minval=0, maxval=4)
middle_mpo = random.randint(keys[4], (inner_dim_mpo, outer_dim, inner_dim_mpo, outer_dim ), minval=0, maxval=4)
last_mpo = random.randint(keys[5], (inner_dim_mpo, outer_dim, 1, outer_dim), minval=0, maxval=4)

s = MPS([first_mps, middle_mps, last_mps])
o = MPO([first_mpo, middle_mpo, last_mpo])

o(s)
print(repr(s))
