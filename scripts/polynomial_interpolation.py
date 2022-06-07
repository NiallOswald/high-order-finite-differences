"""Interpolate a polynomial using non-uniform spacing."""

from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

a = 0
b = 1
boundary = (a, b)
x = np.linspace(a, b, 2001)

p = Interpolation(12, 12, boundary=boundary)
u = p.inter.x**4


def f(x):
    return p.nderivative(x, 2)


plt.plot(
    x, np.dot(np.vectorize(f, signature="()->(n)")(x), u)
)  # Using excluded=['k'] might tidy things
plt.show()
