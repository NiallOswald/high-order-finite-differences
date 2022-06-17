"""Demonstrate Runge's phenomenon."""

from finite_diff.finite_diff import Interpolation, Lagrange
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

n = 10
q = 10

non_unif = Interpolation(n, q, max_iter=10)
x = np.linspace(-1, 1, n + 1)
unif = Lagrange(x)

y = np.linspace(-1, 1, 201)

u = 1 / (1 + 25 * non_unif.inter.x**2)
u = np.array([np.dot(non_unif(k), u) for k in y])

v0 = 1 / (1 + 25 * x**2)
v = np.array([np.sum([unif(k, i) * v0[i] for i in range(n + 1)]) for k in y])

i = np.array([np.sum([unif(k, i) * v0[i] for i in range(n + 1)]) for k in x])

# plt.plot(y, u, label="Hermanns-Hernández Spacing")
plt.plot(y, v, "k--", label="Lagrange Interpolation")
plt.plot(y, 1 / (1 + 25 * y**2), "k:", label="Runge Function")
plt.plot(x, i, "k.", label="Interpolation Points", markersize=10)
plt.xlabel("$x$", fontsize=12)
plt.ylabel("$f(x)$", fontsize=12)
plt.legend()
plt.show()
