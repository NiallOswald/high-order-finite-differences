from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

n = 500
q = 12

inter = Interpolation(n, q, boundary=(0, 1), max_iter=2500)
x = inter.inter.x
print(x)
plt.plot(x, ".")
plt.show()

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
eps = np.linspace(1e-3, 1e-4, k)
colour_grid = np.linspace(-1, 1, k)

cmap = plt.get_cmap("viridis")

for e, i in zip(eps, colour_grid):
    L = e * B + A
    L[0, :] = np.eye(n + 1)[0, :]
    L[n, :] = np.eye(n + 1)[n, :]

    u = np.linalg.solve(L, b)

    plt.plot(x, u, c=cmap(i))

plt.show()
