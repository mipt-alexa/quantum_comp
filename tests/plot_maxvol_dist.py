from maxvol import maxvol
import matplotlib.pyplot as plt
import itertools
import jax.numpy as jnp
import numpy as np

import os

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


def get_distribution(n: int = 30, r: int = 4, trials: int = 50):
    data = np.empty(trials)
    for k in range(trials):
        print(k)
        matrix = jnp.array(np.random.randn(n, r))

        max_vol = 0
        # generate all indexes combinations
        for ind_comb in itertools.combinations(range(n), r):
            curr_vol = abs(jnp.linalg.det(matrix[jnp.array(ind_comb)]))

            if curr_vol > max_vol:
                max_vol = curr_vol

        data[k] = abs(jnp.linalg.det(matrix[maxvol(matrix)])) / max_vol

    plt.hist(data, bins=30, range=(0, 1))
    plt.title("Maxvol for n = " + str(n) + ", r = " + str(r))
    plt.show()


get_distribution()

