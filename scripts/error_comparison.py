"""Plot the errors for different grid spacing schemes."""

from finite_diff.finite_diff import Interpolation, Stencil
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

n = 50
q = 12
eps = 1e-1

# Non-uniform spacing
inter = Interpolation(n, q, boundary=(0, 1), max_iter=5000)
x1 = inter.inter.x

b = 1 + 2 * x1
b[0] = 0
b[n] = 1

print("Computing derivatives...")
A = np.array([inter.nderivative(u, 1) for u in x1])
print("First derivatives complete!")
B = np.array([inter.nderivative(u, 2) for u in x1])
print("Second derivatives complete!")

L = eps * B + A
L[0, :] = np.eye(n + 1)[0, :]
L[n, :] = np.eye(n + 1)[n, :]
u1 = np.linalg.solve(L, b)

plt.plot(x1, u1, "k:", label="Hermanns-Hernández Spacing")

# Uniform spacing
x2 = np.linspace(0, 1, n + 1)
inter = Stencil(x2, q)

b = 1 + 2 * x2
b[0] = 0
b[n] = 1

print("Computing derivatives...")
A = np.array([inter.nderivative(u, i, 1) for i, u in enumerate(x2)])
print("First derivatives complete!")
B = np.array([inter.nderivative(u, i, 2) for i, u in enumerate(x2)])
print("Second derivatives complete!")

L = eps * B + A
L[0, :] = np.eye(n + 1)[0, :]
L[n, :] = np.eye(n + 1)[n, :]
u2 = np.linalg.solve(L, b)

plt.plot(x2, u2, "k--", label="Uniform Spacing")

# True solution
x3 = np.linspace(0, 1, n + 1)
u = (
    lambda x: (2 * eps - 1) / (1 - np.exp(-1 / eps)) * (1 - np.exp(-x / eps))
    + x**2
    + (1 - 2 * eps) * x
)
u = np.vectorize(u)

plt.plot(x3, u(x3), "k-", label="True Solution")
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.legend()
plt.show()

# Error comparison
plt.plot(x1, abs(u1 - u(x1)), "k:", label="Hermanns-Hernández Spacing")
plt.plot(x2, abs(u2 - u(x2)), "k--", label="Uniform Spacing")
plt.xlabel("$x$", fontsize=14)
plt.ylabel("Error: $|u(x) - u_i(x)|$", fontsize=14)
plt.yscale("log")
plt.legend()
plt.show()
