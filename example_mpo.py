from classes import *
from jax import random


k1, k2 = random.split(random.PRNGKey(42))

A = random.randint(k1, (2, 2, 2), minval=0, maxval=4)
B = random.randint(k2, (2, 2, 2, 2), minval=0, maxval=4)
C = random.randint(k2, (2, 2), 0, 4)

s = MPS([C, C])
o = MPO([A, A])

# print(repr(s))
# print(repr(o))
s(o)
print(repr(s))

