from finite_diff.finite_diff import Stencil
import numpy as np
import matplotlib.pyplot as plt

n = 500
q = 12

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
eps = np.linspace(1e-4, 1e-3, k)
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

# Plot for specific epsilon
print("Plotting for eps = 1e-4...")
e = 1e-4

# Solution plot
L = e * B + A
L[0, :] = np.eye(n + 1)[0, :]
L[n, :] = np.eye(n + 1)[n, :]
u = np.linalg.solve(L, b)

plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.plot(x, u, c="black")
plt.show()

# Eigenvalues plot
print("Computing eigenvalues...")
eigs = np.linalg.eigvals(e * B + A)
print("Eigenvalues complete!")

plt.plot(eigs.real, eigs.imag, ".", c="k")
plt.xlabel("$Re(\\lambda)$")
plt.ylabel("$Im(\\lambda)$")
plt.show()

print("Plotting for eps = 1e-3...")
e = 1e-3

# Solution plot
L = e * B + A
L[0, :] = np.eye(n + 1)[0, :]
L[n, :] = np.eye(n + 1)[n, :]
u = np.linalg.solve(L, b)

plt.xlabel("$x$")
plt.ylabel("$u(x)$")
plt.plot(x, u, c="black")
plt.show()

# Eigenvalues plot
print("Computing eigenvalues...")
eigs = np.linalg.eigvals(e * B + A)
print("Eigenvalues complete!")

plt.plot(eigs.real, eigs.imag, ".", c="k")
plt.xlabel("$Re(\\lambda)$")
plt.ylabel("$Im(\\lambda)$")
plt.show()
