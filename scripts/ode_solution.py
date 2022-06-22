"""Plot the solution to the ODE problem for various values of epsilon."""

from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

EIGENVALUES = False  # Plot the spectrum of the operators
FIXED_EPSILON = True  # Generate plots for a fixed epsilon

n = 100
q = 10

inter = Interpolation(n, q, boundary=(0, 1), max_iter=5000)
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
colour_grid = np.linspace(0.9, 0, k)

cmap = plt.get_cmap("viridis")

for e, i in zip(reversed(eps), colour_grid):
    L = e * B + A
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]

    u = np.linalg.solve(L, b)

    plt.plot(x, u, c=cmap(i))

plt.xlabel("$x$", fontsize=16)
plt.ylabel("$u(x)$", fontsize=16)
plt.show()

# Plot for specific epsilon
e1 = 1e-2
e2 = 1e-3

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
