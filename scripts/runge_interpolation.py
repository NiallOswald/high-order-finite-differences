"""Demonstrate Runge's phenomenon."""

from finite_diff.finite_diff import Interpolation, Lagrange
import numpy as np
import matplotlib.pyplot as plt

n = 10
q = 6

non_unif = Interpolation(n, q, max_iter=910)
x = np.linspace(-1, 1, n + 1)
unif = Lagrange(x)

y = np.linspace(-1, 1, 201)

u = 1 / (1 + 25 * non_unif.inter.x**2)
u = np.array([np.dot(non_unif(k), u) for k in y])

v = 1 / (1 + 25 * x**2)
v = np.array([np.sum([unif(k, i) * v[i] for i in range(n + 1)]) for k in y])

plt.plot(y, u)
plt.plot(y, v)
plt.plot(y, 1 / (1 + 25 * y**2), "--")
plt.xlabel("$x$")
plt.ylabel("$f(x)$")
plt.show()
