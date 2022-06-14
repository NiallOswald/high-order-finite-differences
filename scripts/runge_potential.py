import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

n = 250

x = np.linspace(-1, 1, n + 1)
y = np.linspace(-1.5, 1.5, n + 1)

X, Y = np.meshgrid(x, y)


def f(x, y):
    z = x + y * 1j
    return -(1 / 2) * np.real(
        (1 - z) * np.log(np.abs(1 - z)) - (-1 - z) * np.log(np.abs(-1 - z))
    )


Z = f(X, Y)

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")

ax.contour3D(X, Y, Z, 50, cmap="viridis")
# ax.plot_wireframe(X, Y, Z, color="black")

offset = 0.05

# ax.scatter(0, 1 / 5, f(0, 1 / 5), c="red", marker="o")
# ax.scatter(0, -1 / 5, f(0, -1 / 5), c="red", marker="o")

ax.set_xlabel("$\\Re(z)$")
ax.set_ylabel("$\\Im(z)$")
ax.set_zlabel("$\\phi(z)$")

plt.show()
