import numpy as np
from scipy.optimize import newton
from finite_diff.finite_diff import Lagrange
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})


def g(x):
    return (
        -(1 / 2) * (1 - x) * np.log(1 - x)
        - (1 / 2) * (1 + x) * np.log(1 + x)
        + (1 / 2) * np.log(26 / 25)
        + (1 / 5) * np.arctan(5)
    )


roots = [newton(g, -0.75), newton(g, 0.75)]
print(f"Roots: {roots}")


x = np.linspace(-1, 1, 251)

plt.plot(x, g(x), c="k")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$\\phi(x) - \\phi(\\pm i/5)$", fontsize=16)
plt.grid()
plt.show()

n_vals = range(10, 16)

colour_grid = np.linspace(1, 0, len(n_vals))
cmap = plt.get_cmap("viridis")

y = np.linspace(-1, 1, 1001)

for n, c in zip(n_vals, colour_grid):
    x = np.linspace(-1, 1, n + 1)
    unif = Lagrange(x)

    v0 = 1 / (1 + 25 * x**2)
    v = np.array(
        [np.sum([unif(k, i) * v0[i] for i in range(n + 1)]) for k in y]
    )

    plt.plot(y, v, c=cmap(c))

plt.plot(roots[0], 0, "o", c="red")
plt.plot(roots[1], 0, "o", c="red")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$f_N(x)$", fontsize=16)
plt.grid()
plt.show()
