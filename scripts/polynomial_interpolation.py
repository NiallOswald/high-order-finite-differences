from finite_diff.finite_diff import Interpolation
import numpy as np
import matplotlib.pyplot as plt

p = Interpolation(12, 12)
u = p.inter.x**4

x = np.linspace(-1, 1, 2001)


def f(x):
    return p.nderivative(x, 2)


plt.plot(
    x, np.dot(np.vectorize(f, signature="()->(n)")(x), u)
)  # Using excluded=['k'] might tidy things
plt.show()
