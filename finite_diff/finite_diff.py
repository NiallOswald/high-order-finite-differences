"""High Order Finite Differences based on M Hermanns and J A Hernández."""

import numpy as np
from sympy.utilities.iterables import multiset_permutations


class ConvergenceError(RuntimeError):
    """Raised when the algorithm fails to converge."""

    pass


class Lagrange:
    """A general Lagrange interpolation polynomial."""

    def __init__(self, x):
        self.x = x
        self.n = len(x)
        self.slicers = {}

    def __call__(self, y, i):
        """Evaluate the Lagrange polynomial about x[i] at y."""
        index_slice = np.ones(self.n, dtype=bool)
        index_slice[i] = False
        return np.prod(y - self.x[index_slice]) / self._denominator(i)

    def nderivative(self, y, i, k):
        """Evaluate the value of the kth derivative about x[i] at y."""
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
    """An interpolant for a set of points x, q points, and an offset s."""

    def __init__(self, x, q, s):
        self.x = x
        self.q = q
        self.s = s
        self.inter = Lagrange(self.x[self.s : self.s + self.q + 1])

    def __call__(self, y):
        """Evaluate the interpolant at y."""
        return np.array([self.inter(y, i) for i in range(self.q + 1)])

    def nderivative(self, y, k):
        """Evaluate the kth derivative of the interpolant at y."""
        return np.array(
            [self.inter.nderivative(y, i, k) for i in range(self.q + 1)]
        )


class Stencil:
    """A stencil of unbounded interpolants."""

    def __init__(self, x, q):
        self.x = x
        self.q = q
        self.n = len(x) - 1

    def __len__(self):
        return self.n + 1

    def __getitem__(self, i):
        if i > self.n:
            raise IndexError("Index out of bounds.")
        return Interpolant(self.x, self.q, self._find_s(i))

    def __call__(self, y, i):
        """Evaluate the stencil at y."""
        p = self[i]
        return self._spacing(p, p(y))

    def nderivative(self, y, i, k):
        """Evaluate the kth derivative of the stencil at y."""
        p = self[i]
        return self._spacing(p, p.nderivative(y, k))

    def _find_s(self, i):
        if i == -1:  # Bit of a bodge, should fix
            return self.n - self.q

        if i < self.q // 2:
            return 0
        elif i >= self.q // 2 and i < self.n - self.q // 2:
            return i - self.q // 2
        else:
            return self.n - self.q

    def _spacing(self, p, u):
        return np.concatenate(
            [np.zeros(p.s), u, np.zeros(self.n - self.q - p.s)]
        )


class PolyFactor:
    """Polynomial factors appearing in the error terms."""

    def __init__(self, x, q):
        self.x = x
        self.q = q
        self.n = len(x) - 1

        self.slicers = {}

    def __call__(self, y, i):
        """Evaluate the ith polynomial factor at y."""
        s = self._find_s(i)
        return np.prod([y - self.x[s + k] for k in range(self.q + 1)])

    def derivative(self, y, i):
        """Evaluate the derivative of the ith polynomial factor at y."""
        return self.nderivative(y, i, 1)

    def second_derivative(self, y, i):
        """Evaluate the second derivative of the ith polynomial factor at y."""
        return self.nderivative(y, i, 2)

    def nderivative(self, y, i, k):
        """Evaluate the kth derivative of the ith polynomial factor at y."""
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
        elif i < self.q // 2:
            return 0
        elif i >= self.q // 2 and i < self.n - self.q // 2:
            return i - self.q // 2
        elif i <= self.n:
            return self.n - self.q
        else:
            raise IndexError("Index out of bounds.")


class Interpolation:
    """A final interpolation of the unknown function at x."""

    def __init__(self, n, q, boundary=(-1, 1), tol=None, max_iter=50):
        self.n = n
        self.q = q
        self.a = boundary[0]
        self.b = boundary[1]

        self.l_indicator = 0
        self.r_indicator = 0

        self.endpoints, self.inter = self._find_endpoints(tol, max_iter)

    def __getitem__(self, i):
        return self.inter[i]

    def __call__(self, y):
        """Evaluate the final interpolation at y."""
        p = self.inter[self._find_domain(y)]
        return self._spacing(p, p(y))

    def nderivative(self, y, k):
        """Evaluate the kth derivative of the final interpolation at y."""
        p = self.inter[self._find_domain(y)]
        return self._spacing(p, p.nderivative(y, k))

    def _find_endpoints(self, tol, max_iter):
        endpoints = np.linspace(self.a, self.b, self.n + 2)
        factors = PolyFactor(endpoints[1:-1], self.q - 1)
        extrema = self._find_extrema(factors, endpoints[1:-1])

        if tol is None:

            def _bound_sense(vals, k):
                return k < max_iter

        else:

            def _bound_sense(vals, k):
                return max(vals) - min(vals) > tol

        k = 1

        def step(k, error, max_diff):
            return 1 / (2 * k - 1)

        while _bound_sense(self._extrema_diff(factors, extrema), k):
            diff = self._extrema_diff(factors, extrema)
            max_diff = max(abs(diff))

            if k % 100 == 0:
                print(f"Iteration {k}:")
                vals = self._extrema_vals(factors, extrema)
                print(vals)
                print(f"Error: {max(vals) - min(vals)}")

            for i in range(1, len(extrema)):
                error = diff[i]
                if error > 0:
                    endpoints[i + 1] -= (
                        endpoints[i + 1] - extrema[i - 1]
                    ) * step(k, error, max_diff)
                else:
                    endpoints[i + 1] += (extrema[i] - endpoints[i + 1]) * step(
                        k, error, max_diff
                    )

            error = diff[0]
            if error > 0:
                endpoints[1] -= (endpoints[1] - self.a) * step(
                    k, error, max_diff
                )
                self.l_indicator -= 1
            else:
                endpoints[1] += (extrema[0] - endpoints[1]) * step(
                    k, error, max_diff
                )
                self.l_indicator += 1

            error = diff[-1]
            if error > 0:
                endpoints[-2] -= (endpoints[-2] - extrema[-1]) * step(
                    k, error, max_diff
                )
                self.r_indicator -= 1
            else:
                endpoints[-2] += (self.b - endpoints[-2]) * step(
                    k, error, max_diff
                )
                self.r_indicator += 1

            factors = PolyFactor(endpoints[1:-1], self.q - 1)
            extrema = self._find_extrema(factors, endpoints[1:-1])

            if k > max_iter:
                raise RuntimeError("Endpoints do not converge.")

            k += 1

        print("Interpolation complete!")
        vals = self._extrema_vals(factors, extrema)
        print(f"Error after {k} iterations: {max(vals) - min(vals)}")
        print(f"Left indicator: {self.l_indicator}")
        print(f"Right indicator: {self.r_indicator}")

        return endpoints, Stencil(
            np.concatenate(([self.a], extrema, [self.b])), self.q
        )

    def _find_extrema(self, factors, endpoints):
        return np.array(
            [
                newton_raphson(
                    (endpoints[i] + endpoints[i + 1]) / 2,
                    factors.derivative,
                    factors.second_derivative,
                    i=i,
                )
                for i in range(len(endpoints) - 1)
            ]
        )

    def _extrema_diff(self, factors, extrema):
        vals = self._extrema_vals(factors, extrema)
        return vals[:-1] - vals[1:]

    def _extrema_vals(self, factors, extrema):
        return abs(
            np.concatenate(
                [
                    [factors(self.a, 0)],
                    [factors(extrema[i], i) for i in range(len(extrema))],
                    [factors(self.b, -1)],
                ]
            )
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


def newton_raphson(x, f, df, tol=1e-8, max_iter=1e4, **kwargs):
    """Find the root of f(x) = 0 using Newton-Raphson."""
    x0 = x
    k = 0
    while k < max_iter:
        x1 = x0 - f(x0, **kwargs) / df(x0, **kwargs)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
        k += 1

    raise RuntimeError("Newton-Raphson did not converge.")
