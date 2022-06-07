"""Script for tuning the step size used in Interpolation."""

from finite_diff.finite_diff import Interpolation, PolyFactor
import numpy as np
import matplotlib.pyplot as plt

n = 500
iters = [1000]  # , 1000, 1500, 2000, 2500]

colour_grid = np.linspace(1, 0, len(iters))
cmap = plt.get_cmap("viridis")

for k, c in zip(iters, colour_grid):
    p = Interpolation(n, 4, max_iter=k)
    x = p.inter.x
    extrema = x[1:-1]

    factors = PolyFactor(p.endpoints[1:-1], p.q - 1)

    errors = np.concatenate(
        [
            [abs(abs(factors(-1, 0)) - abs(factors(extrema[0], 0)))],
            [
                abs(
                    abs(factors(extrema[i], i))
                    - abs(factors(extrema[i + 1], i + 1))
                )
                for i in range(len(extrema) - 1)
            ],
            [abs(abs(factors(extrema[-1], -1)) - abs(factors(1, -1)))],
        ]
    )

    maxima = np.concatenate(
        [
            [abs(factors(-1, 0))],
            [abs(factors(extrema[i], i)) for i in range(len(extrema))],
            [abs(factors(1, -1))],
        ]
    )

    # print(errors)
    print(maxima)

    plt.plot(np.concatenate([[-1], extrema, [1]]), maxima, c=cmap(c))

plt.show()
