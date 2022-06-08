"""Plot convergence to the true solution as n and q varies."""

from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

n_vals = [20, 40, 80, 160, 320, 640]
q_vals = [1, 2, 4, 8, 12, 16, 20]
eps = 1e-3

# True solution
u = (
    lambda x: (2 * eps - 1) / (1 - np.exp(-1 / eps)) * (1 - np.exp(-x / eps))
    + x**2
    + (1 - 2 * eps) * x
)
u_true = np.vectorize(u)

# Convergence over n, q fixed
x_n = []
u_n = []

colour_grid = np.linspace(1, 0, len(n_vals))
cmap = plt.get_cmap("viridis")

for n, i in zip(n_vals, colour_grid):
    inter = Interpolation(n, q=4, boundary=(0, 1), max_iter=500)
    x = inter.inter.x
    x_n.append(x)

    b = 1 + 2 * x
    b[0] = 0
    b[n] = 1

    A = np.array([inter.nderivative(u, 1) for u in x])
    B = np.array([inter.nderivative(u, 2) for u in x])

    L = eps * B + A
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]
    u = np.linalg.solve(L, b)
    u_n.append(u)

    plt.plot(x, u, c=cmap(i), label=f"$n = {n}$")

plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.legend()
plt.show()

# Error plots
errors = [abs(u_n[i] - u_true(x_n[i])) for i in range(len(n_vals))]
for i, j in zip(range(len(n_vals)), colour_grid):
    plt.plot(x_n[i], errors[i], c=cmap(j), label=f"$n = {n_vals[i]}$")
plt.xlabel("$x$")
plt.ylabel("Error: $|u(x) - u_i(x)|$")
plt.yscale("log")
plt.legend()
plt.show()
