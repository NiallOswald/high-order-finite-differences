from finite_diff.finite_diff import Stencil
import numpy as np
import matplotlib.pyplot as plt

EIGENVALUES = False
FIXED_EPSILON = False

n = 1000
q = 4

x = np.linspace(0, 1, n + 1)
inter = Stencil(x, q)

b = 1 + 2 * x
b[0] = 0
b[n] = 1


def get_s(i):
    if i == -1:  # Bit of a bodge, should fix
        return n - q
    if i < q // 2:
        return 0
    elif i >= q // 2 and i < n - q // 2:
        return i - q // 2
    else:
        return n - q


def spacing(s, u):
    return np.concatenate([np.zeros(s), u, np.zeros(n - q - s)])


print("Computing derivatives...")
A = np.array(
    [spacing(get_s(i), inter[i].nderivative(u, 1)) for i, u in enumerate(x)]
)
print("First derivatives complete!")
B = np.array(
    [spacing(get_s(i), inter[i].nderivative(u, 2)) for i, u in enumerate(x)]
)
print("Second derivatives complete!")

print("Plotting...")
k = 10
eps = [10 ** (-i) for i in np.sqrt(np.linspace(1, 3**2, k))]
colour_grid = np.linspace(1, -1, k)

cmap = plt.get_cmap("viridis")

for e, i in zip(eps, colour_grid):
    L = e * B + A
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]

    u = np.linalg.solve(L, b)

    plt.plot(x, u, c=cmap(i))

plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.show()


# Plots for specific epsilon
e1 = 1e-3
e2 = 1e-4

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
