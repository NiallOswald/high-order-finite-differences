"""Plot the errors for different grid spacing schemes."""

from finite_diff.finite_diff import Interpolation, Stencil
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

n = 100
q = 12
eps = 1e-2

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

# Grid Stretching
c = 8
erf = sp.erf(np.sqrt(c))

y = np.linspace(0, 1, n + 1)
x3 = np.array([1 - (sp.erf(np.sqrt(c) * (1 - y_val))) / erf for y_val in y])

inter = Stencil(y, q)

exp = np.exp(c * (1 - y) ** 2)

f = 3 - (2 * (sp.erf(np.sqrt(c) * (1 - y)))) / erf
f[0] = 0
f[n] = 1

print("Computing derivatives...")
A = np.array([inter.nderivative(u, i, 1) for i, u in enumerate(y)])
print("First derivatives complete!")
B = np.array([inter.nderivative(u, i, 2) for i, u in enumerate(y)])
print("Second derivatives complete!")

a0 = (np.sqrt(np.pi) / 2) * (erf / np.sqrt(c)) * exp
a1 = 1 - (eps * np.sqrt(c * np.pi) * erf) * (1 - y) * exp
a = a0 * a1

b = eps * (np.pi / 4) * ((erf**2) / c) * exp**2

L = (B.T * b).T + (A.T * a).T
L[0, :] = np.eye(n + 1)[0, :]
L[n, :] = np.eye(n + 1)[n, :]
u3 = np.linalg.solve(L, f)

plt.plot(x3, u3, "k-.", label="Grid Stretching")

# True solution
x4 = np.linspace(0, 1, n + 1)
u = (
    lambda x: (2 * eps - 1) / (1 - np.exp(-1 / eps)) * (1 - np.exp(-x / eps))
    + x**2
    + (1 - 2 * eps) * x
)
u = np.vectorize(u)

plt.plot(x4, u(x4), "k-", label="True Solution")
plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.legend()
plt.show()

# Error comparison
plt.plot(
    x1[1:-1], abs(u1 - u(x1))[1:-1], "k:", label="Hermanns-Hernández Spacing"
)
plt.plot(x2[1:-1], abs(u2 - u(x2))[1:-1], "k--", label="Uniform Spacing")
plt.plot(x3[1:-1], abs(u3 - u(x3))[1:-1], "k-.", label="Grid Stretching")
plt.xlabel("$x$", fontsize=14)
plt.ylabel("Error: $|u(x) - u_i(x)|$", fontsize=14)
plt.yscale("log")
plt.legend()
plt.show()
