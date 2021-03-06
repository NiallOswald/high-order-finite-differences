"""Plot convergence to the true solution as n and q varies."""

from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

n_vals = [20, 30, 40, 50, 60, 70, 80, 90, 100]
q_vals = [2, 3, 4, 5, 6, 7, 8, 9, 10]

cmap = plt.get_cmap("viridis")

# True solution
u = (
    lambda x, eps: (2 * eps - 1)
    / (1 - np.exp(-1 / eps))
    * (1 - np.exp(-x / eps))
    + x**2
    + (1 - 2 * eps) * x
)
u_true = np.vectorize(u)

# Convergence over n, q fixed
x_n = []
u_n = []

eps = 1e-3
q = 4

colour_grid_n = np.linspace(0.9, 0, len(n_vals))

for n, i in zip(n_vals, colour_grid_n):
    inter = Interpolation(n, q, boundary=(0, 1), max_iter=5000)
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

    plt.plot(x, u, c=cmap(i), label=f"$N = {n}$")

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$u(x)$", fontsize=16)
plt.legend()
plt.show()

# Error plots
errors = [abs(u_n[i] - u_true(x_n[i], eps)) for i in range(len(n_vals))]
for i, j in zip(range(len(n_vals)), colour_grid_n):
    plt.plot(
        x_n[i][1:-1], errors[i][1:-1], c=cmap(j), label=f"$N = {n_vals[i]}$"
    )
plt.xlabel("$x$", fontsize=16)
plt.ylabel("Error: $|u(x) - u_i(x)|$", fontsize=16)
plt.yscale("log")
plt.legend()
plt.show()

# Convergence over q, n fixed
x_q = []
u_q = []

eps = 1e-2
n = 250

colour_grid_q = np.linspace(0.9, 0, len(q_vals))

for q, i in zip(q_vals, colour_grid_q):
    inter = Interpolation(n, q, boundary=(0, 1), max_iter=5000)
    x = inter.inter.x
    x_q.append(x)

    b = 1 + 2 * x
    b[0] = 0
    b[n] = 1

    A = np.array([inter.nderivative(u, 1) for u in x])
    B = np.array([inter.nderivative(u, 2) for u in x])

    L = eps * B + A
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]
    u = np.linalg.solve(L, b)
    u_q.append(u)

    plt.plot(x, u, c=cmap(i), label=f"$q = {q}$")

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$u(x)$", fontsize=16)
plt.legend()
plt.show()

# Error plots
errors = [abs(u_q[i] - u_true(x_q[i], eps)) for i in range(len(q_vals))]
for i, j in zip(range(len(q_vals)), colour_grid_q):
    plt.plot(
        x_q[i][1:-1], errors[i][1:-1], c=cmap(j), label=f"$q = {q_vals[i]}$"
    )
plt.xlabel("$x$", fontsize=16)
plt.ylabel("Error: $|u(x) - u_i(x)|$", fontsize=16)
plt.legend()
plt.show()

for i, j in zip(range(len(q_vals)), colour_grid_q):
    plt.plot(
        x_q[i][1:-1], errors[i][1:-1], c=cmap(j), label=f"$q = {q_vals[i]}$"
    )
plt.xlabel("$x$", fontsize=16)
plt.ylabel("Error: $|u(x) - u_i(x)|$", fontsize=16)
plt.yscale("log")
plt.legend()
plt.show()
