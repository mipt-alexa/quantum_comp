import numpy as np
from classes import *
from jax import random


a = jnp.array([[0, 1],
              [1, 2]])
b = random.randint(random.PRNGKey(423), (2, 2, 2), minval=0, maxval=4)
c = jnp.array([[3, 1],
              [1, -1]])
m = MPS([a, b, c])
print(m.shape[1][1])
#
# """A small test just for me:"""
#
# e = jnp.array([[1., 0],
#                [2, 1]])
# m = MPS([e, e])
# print(repr(m))
# print(type(m.dot(m)))
