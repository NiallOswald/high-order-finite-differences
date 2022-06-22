"""Plot the solution to the ODE problem using a uniform grid by Moe Okawara."""

from finite_diff.finite_diff import Stencil
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

FIXED_EPSILON = True  # Generate plots for a fixed epsilon

n = 100
q = 10

c = 8
erf = sp.erf(np.sqrt(c))

y = np.linspace(0, 1, n + 1)
x = np.array([1 - (sp.erf(np.sqrt(c) * (1 - y_val))) / erf for y_val in y])

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

print("Plotting...")
k = 10
eps = [10 ** (-i) for i in np.sqrt(np.linspace(1, 3**2, k))]
colour_grid = np.linspace(0.9, 0, k)

cmap = plt.get_cmap("viridis")

for e, i in zip(reversed(eps), colour_grid):
    a0 = (np.sqrt(np.pi) / 2) * (erf / np.sqrt(c)) * exp
    a1 = 1 - (e * np.sqrt(c * np.pi) * erf) * (1 - y) * exp
    a = a0 * a1

    b = e * (np.pi / 4) * ((erf**2) / c) * exp**2

    L = (B.T * b).T + (A.T * a).T
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]

    u = np.linalg.solve(L, f)

    plt.plot(x, u, c=cmap(i))

plt.xlabel("$x$", fontsize=14)
plt.ylabel("$u(x)$", fontsize=14)
plt.show()


# Plots for specific epsilon
e1 = 1e-3
e2 = 1e-5

# Solution plot
if FIXED_EPSILON:
    print(f"Plotting for eps = {e1}...")

    a0 = (np.sqrt(np.pi) / 2) * (erf / np.sqrt(c)) * exp
    a1 = 1 - (e1 * np.sqrt(c * np.pi) * erf) * (1 - y) * exp
    a = a0 * a1

    b = e1 * (np.pi / 4) * ((erf**2) / c) * exp**2

    L1 = (B.T * b).T + (A.T * a).T
    L1[0, :] = np.eye(n + 1)[0, :]
    L1[n, :] = np.eye(n + 1)[n, :]

    u1 = np.linalg.solve(L1, f)

    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$u(x)$", fontsize=16)
    plt.plot(x, u1, c="black")
    plt.show()

    print(f"Plotting for eps = {e2}...")

    a0 = (np.sqrt(np.pi) / 2) * (erf / np.sqrt(c)) * exp
    a1 = 1 - (e2 * np.sqrt(c * np.pi) * erf) * (1 - y) * exp
    a = a0 * a1

    b = e2 * (np.pi / 4) * ((erf**2) / c) * exp**2

    L2 = (B.T * b).T + (A.T * a).T
    L2[0, :] = np.eye(n + 1)[0, :]
    L2[n, :] = np.eye(n + 1)[n, :]

    u2 = np.linalg.solve(L2, f)

    plt.xlabel("$x$", fontsize=16)
    plt.ylabel("$u(x)$", fontsize=16)
    plt.plot(x, u2, c="black")
    plt.show()
