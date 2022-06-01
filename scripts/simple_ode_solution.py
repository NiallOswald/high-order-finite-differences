from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

# N = 201, q = 12
n = 201
q = 12

inter = Interpolation(n, q)
x = inter.inter.x

A = np.concatenate(
    [[np.eye(n + 1)[0, :]], np.array([inter.nderivative(u, 1) for u in x[1:]])]
)
b = np.concatenate([[-1 / 3], x[1:] ** 2])

plt.plot(x, np.dot(np.linalg.inv(A), b))
plt.plot(x, (x**3) / 3)
plt.show()
