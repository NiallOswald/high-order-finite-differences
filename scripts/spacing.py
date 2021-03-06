"""Script for tuning the step size used in Interpolation."""

from finite_diff.finite_diff import Interpolation, PolyFactor
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

n = 100
iters = [100, 500, 1000, 2500, 5000]  # , 10000, 25000]

colour_grid = np.linspace(0.9, 0, len(iters))
cmap = plt.get_cmap("viridis")

for k, c in zip(iters, colour_grid):
    p = Interpolation(n, 10, max_iter=k)
    x = p.inter.x
    extrema = x[1:-1]

    assert p.r_indicator < k * 0.9

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
    # print(maxima)

    plt.plot(
        np.concatenate([[-1], extrema, [1]]),
        maxima,
        c=cmap(c),
        label=f"k = {k}",
    )

plt.xlabel("$x_i$", fontsize=14)
plt.ylabel("$|\\pi_i(x_i)|$", fontsize=14)
plt.legend()
plt.show()
