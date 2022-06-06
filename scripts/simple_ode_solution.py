"""Solve a simple ODE using the high-order finite difference method."""

from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

# N = 200, q = 12
n = 200
q = 12

inter = Interpolation(n, q, boundary=(0, 1))
x = inter.inter.x

A = np.concatenate(
    [[np.eye(n + 1)[0, :]], np.array([inter.nderivative(u, 1) for u in x[1:]])]
)
b = np.concatenate([[0], x[1:] ** 2])

plt.plot(x, np.dot(np.linalg.inv(A), b))
plt.plot(x, (x**3) / 3)
plt.show()
