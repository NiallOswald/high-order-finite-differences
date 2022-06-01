"""High Order Finite Differences based on M Hermanns and J A Hernández."""

import numpy as np
from sympy.utilities.iterables import multiset_permutations


class ConvergenceError(RuntimeError):
    """Raised when the algorithm fails to converge."""

    pass


class Lagrange:
    """A Base class for Lagrange and PiFactor."""

    def __init__(self, x):
        self.x = x
        self.n = len(x)
        self.slicers = {}

    def __call__(self, y, i):
        index_slice = np.ones(self.n, dtype=bool)
        index_slice[i] = False
        return np.prod(y - self.x[index_slice]) / self._denominator(i)

    def nderivative(self, y, i, k):
        if (i, k) not in self.slicers:
            self._build_slicer(i, k)
        slicer = self.slicers[(i, k)]

        u = np.tile(self.x, (len(slicer), 1))
        return np.sum(
            np.prod(
                y - u[slicer].reshape((len(slicer), self.n - k - 1)), axis=1
            )
        ) / self._denominator(i)

    def _denominator(self, i):
        index_slice = np.ones(self.n, dtype=bool)
        index_slice[i] = False
        return np.prod(self.x[i] - self.x[index_slice])

    def _build_slicer(self, i, k):
        slicer = np.array(
            list(
                multiset_permutations(
                    list(range(2, k + 2)) + [1] * (self.n - k - 1)
                )
            )
        )
        slicer[slicer > 1] = 0
        slicer = np.insert(slicer, i, 0, axis=1)
        self.slicers[(i, k)] = slicer.astype(bool)


class Interpolant:
    """An interpolant for a set of points x, q points, and offset s."""

    def __init__(self, x, q, s):
        self.x = x
        self.q = q
        self.s = s
        self.inter = Lagrange(self.x[self.s : self.s + self.q + 1])

    def __call__(self, y):
        return np.array([self.inter(y, i) for i in range(self.q + 1)])

    def nderivative(self, y, k):
        """Return the kth derivative of the interpolant at y."""
        return np.array(
            [self.inter.nderivative(y, i, k) for i in range(self.q + 1)]
        )


class Stencil:
    """A stencil of unbounded interpolants."""

    def __init__(self, x, q):
        self.x = x
        self.q = q
        self._build_stencil()

    def _build_stencil(self):
        n = len(self.x) - 1
        s = 0
        self.stencil = []
        for i in range(n + 1):
            if i > self.q // 2 and s < n - self.q:
                s += 1

            self.stencil.append(Interpolant(self.x, self.q, s))

    def __len__(self):
        return len(self.stencil)

    def __getitem__(self, i):
        return self.stencil[i]

    def __iter__(self):
        return iter(self.stencil)


class PolyFactor:
    def __init__(self, x, q):
        self.x = x
        self.q = q
        self.n = len(x) - 1

        self.slicers = {}

    def __call__(self, y, i):
        s = self._find_s(i)
        return np.prod([y - self.x[s + k] for k in range(self.q + 1)])

    def derivative(self, y, i):
        return self.nderivative(y, i, 1)

    def second_derivative(self, y, i):
        return self.nderivative(y, i, 2)

    def nderivative(self, y, i, k):
        s = self._find_s(i)

        if k not in self.slicers:
            self._build_slicer(k)
        slicer = self.slicers[k]

        u = np.tile(self.x[s : s + self.q + 1], (len(slicer), 1))
        return np.sum(
            np.prod(
                y - u[slicer].reshape((len(slicer), self.q - k + 1)), axis=1
            )
        )

    def __len__(self):
        return self.n + 1

    def _build_slicer(self, k):
        slicer = np.array(
            list(
                multiset_permutations(
                    list(range(2, k + 2)) + [1] * (self.q - k + 1)
                )
            )
        )
        slicer[slicer > 1] = 0
        self.slicers[k] = slicer.astype(bool)

    def _find_s(self, i):
        if i == -1:  # Bit of a bodge, should fix
            return self.n - self.q

        if i < self.q // 2:
            return 0
        elif i >= self.q // 2 and i < self.n - self.q // 2:
            return i - self.q // 2
        else:
            return self.n - self.q


class Interpolation:
    """A final interpolation of the unknown function at x."""

    def __init__(self, n, q, tol=1e-6, max_iter=2500):
        self.n = n
        self.q = q
        self.endpoints, self.inter = self._find_endpoints(tol, max_iter)

    def __getitem__(self, i):
        return self.inter[i]

    def _find_endpoints(self, tol, max_iter):
        endpoints = np.linspace(-1, 1, self.n + 2)
        factors = PolyFactor(endpoints[1:-1], self.q - 1)
        extrema = self._find_extrema(endpoints[1:], factors)

        k = 1

        while self._extrema_diff(factors, extrema) > tol:
            if k % 100 == 0:
                print(f"Iteration {k}:")
                print(self._extrema_diff(factors, extrema))
                print(endpoints)

            h = k

            for i in range(len(extrema) - 1):
                if i == len(factors) - 2:
                    continue
                if abs(factors(extrema[i], i)) > abs(
                    factors(extrema[i + 1], i + 1)
                ):
                    endpoints[i + 2] -= (endpoints[i + 2] - extrema[i]) / (
                        2 * h
                    )
                else:
                    endpoints[i + 2] += (extrema[i + 1] - endpoints[i + 2]) / (
                        2 * h
                    )

            if abs(factors(-1, 0)) > abs(factors(extrema[0], 0)):
                endpoints[1] -= (endpoints[1] + 1) / (2 * h)
            else:
                endpoints[1] += (extrema[0] - endpoints[1]) / (2 * h)

            if abs(factors(1, -1)) < abs(factors(extrema[-1], -1)):
                endpoints[-2] -= (1 - endpoints[-2]) / (2 * h)
            else:
                endpoints[-2] += (endpoints[-1] - endpoints[-2]) / (2 * h)

            factors = PolyFactor(endpoints[1:-1], self.q - 1)
            extrema = self._find_extrema(endpoints[1:], factors)
            k += 1

            if k > max_iter:
                raise RuntimeError("Endpoints do not converge.")

        print("Interpolation complete!")

        return endpoints, Stencil(np.concatenate(([-1], extrema, [1])), self.q)

    def _find_extrema(self, endpoints, poly_factors):
        return np.array(
            [
                newton_raphson(
                    (endpoints[i] + endpoints[i + 1]) / 2,
                    poly_factors.derivative,
                    poly_factors.second_derivative,
                    i=i,
                )
                for i in range(len(endpoints) - 2)
            ]
        )

    def _extrema_diff(self, factors, extrema):
        """Find the sum of the adjacent extrema differences."""
        return (
            sum(
                abs(
                    abs(factors(extrema[i], i))
                    - abs(factors(extrema[i + 1], i + 1))
                )
                for i in range(len(extrema) - 1)
            )
            + abs(abs(factors(-1, 0)) - abs(factors(extrema[0], 0)))
            + abs(abs(factors(extrema[-1], -1)) - abs(factors(1, -1)))
        )

    def _find_domain(self, y):
        if y < self.endpoints[0] or y > self.endpoints[-1]:
            raise ValueError("Outside of interpolation domain.")

        for i, e in enumerate(self.endpoints[1:]):
            if y < e:
                break

        return i

    def _spacing(self, p, u):
        return np.concatenate(
            [np.zeros(p.s), u, np.zeros(self.n - self.q - p.s)]
        )

    def __call__(self, y):
        p = self.inter[self._find_domain(y)]
        return self._spacing(p, p(y))

    def nderivative(self, y, k):
        p = self.inter[self._find_domain(y)]
        return self._spacing(p, p.nderivative(y, k))


def newton_raphson(x, f, df, tol=1e-8, **kwargs):
    """Find the root of f(x) = 0 using Newton-Raphson."""
    x0 = x
    while True:
        x1 = x0 - f(x0, **kwargs) / df(x0, **kwargs)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
