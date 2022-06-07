"""Plot the solution to the ODE problem for various values of epsilon."""

from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

EIGENVALUES = False  # Plot the spectrum of the operators
FIXED_EPSILON = False  # Generate plots for a fixed epsilon

n = 1000
q = 4

inter = Interpolation(n, q, boundary=(0, 1), max_iter=40)
x = inter.inter.x
print(x)

b = 1 + 2 * x
b[0] = 0
b[n] = 1

print("Computing derivatives...")
A = np.array([inter.nderivative(u, 1) for u in x])
print("First derivatives complete!")
B = np.array([inter.nderivative(u, 2) for u in x])
print("Second derivatives complete!")

print("Plotting...")
k = 10
eps = [10 ** (-i) for i in np.sqrt(np.linspace(1, 3**2, k))]
colour_grid = np.linspace(1, 0, k)

cmap = plt.get_cmap("viridis")

for e, i in zip(reversed(eps), colour_grid):
    L = e * B + A
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]

    u = np.linalg.solve(L, b)

    plt.plot(x, u, c=cmap(i))

plt.xlabel("$x$")
plt.ylabel("$u(x)$")
# plt.xlim([-0.01, 0.11])
# plt.ylim([-1.35, -0.55])
plt.show()

# Plot for specific epsilon
e1 = 1e-4
e2 = 1e-5

# Solution plot
if FIXED_EPSILON:
    print(f"Plotting for eps = {e1}...")

    L1 = e1 * B + A
    L1[0, :] = np.eye(n + 1)[0, :]
    L1[n, :] = np.eye(n + 1)[n, :]
    u = np.linalg.solve(L1, b)

    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.plot(x, u, c="black")
    plt.show()

    print(f"Plotting for eps = {e2}...")

    L2 = e2 * B + A
    L2[0, :] = np.eye(n + 1)[0, :]
    L2[n, :] = np.eye(n + 1)[n, :]
    u = np.linalg.solve(L2, b)

    plt.xlabel("$x$")
    plt.ylabel("$u(x)$")
    plt.plot(x, u, c="black")
    plt.show()

# Eigenvalues plot
if EIGENVALUES:
    print(f"Plotting for eps = {e1}...")

    print("Computing eigenvalues...")
    eigs = np.linalg.eigvals(e1 * B + A)
    print("Eigenvalues complete!")

    plt.plot(eigs.real, eigs.imag, ".", c="k")
    plt.xlabel("$Re(\\lambda)$")
    plt.ylabel("$Im(\\lambda)$")
    plt.show()

    print(f"Plotting for eps = {e2}...")

    print("Computing eigenvalues...")
    eigs = np.linalg.eigvals(e2 * B + A)
    print("Eigenvalues complete!")

    plt.plot(eigs.real, eigs.imag, ".", c="k")
    plt.xlabel("$Re(\\lambda)$")
    plt.ylabel("$Im(\\lambda)$")
    plt.show()
