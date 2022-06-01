"""High Order Finite Differences based on M Hermanns and J A Hernández."""

import numpy as np


class ConvergenceError(RuntimeError):
    """Raised when the algorithm fails to converge."""

    pass


class BaseLagrange:
    """A Base class for Lagrange and PiFactor."""

    def __init__(self, x):
        self.x = x
        self.n = len(x)

    def __call__(self, y, i):
        index_slice = np.ones(self.n, dtype=bool)
        index_slice[i] = False
        return np.prod(y - self.x[index_slice])

    def nderivative(self, y, i, k):
        counters = np.concatenate(([i], np.zeros(k, dtype=int)), dtype=int)
        total = 0
        index = 1

        while index:
            while counters[index] < self.n:
                if counters[index] in counters[:index]:
                    counters[index] += 1
                    continue

                if index < k:
                    index += 1
                else:
                    index_slice = np.ones(self.n, dtype=bool)
                    index_slice[counters] = False
                    total += np.prod(y - self.x[index_slice])
                    counters[index] += 1

            counters[index] = 0
            index -= 1
            counters[index] += 1

        return total


class Lagrange(BaseLagrange):
    """A general Lagrange polynomial for a set of points x."""

    def __call__(self, y, i):
        return super()(y, i) / self._denominator(i)

    def nderivative(self, y, i, k):
        return super().nderivative(y, i, k) / self._denominator(i)

    def _denominator(self, i):
        index_slice = np.ones(self.n, dtype=bool)
        index_slice[i] = False
        return np.prod(self.x[i] - self.x[index_slice])


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


class PiFactor:
    def __init__(self, x, q, s):
        self.x = x
        self.q = q
        self.s = s

    def __call__(self, y):
        return np.prod([y - self.x[self.s + k] for k in range(self.q + 1)])

    def derivative(self, y):
        """Return the derivative of the pi factor at y."""
        return np.sum(
            [
                np.prod(
                    [
                        y - self.x[self.s + i]
                        for i in range(self.q + 1)
                        if i != k
                    ]
                )
                for k in range(self.q + 1)
            ]
        )

    def second_derivative(self, y):
        """Return the second derivative of the pi factor at y."""
        return 2 * np.sum(
            [
                np.sum(
                    [
                        np.prod(
                            [
                                y - self.x[self.s + i]
                                for i in range(self.q + 1)
                                if i != j and i != k
                            ]
                        )
                        for j in range(k + 1, self.q + 1)
                    ]
                )
                for k in range(self.q + 1)
            ]
        )


class PolyFactor:
    """Factors appearing in the error of the interpolant."""

    def __init__(self, x, q):
        self.x = x
        self.q = q
        self._build_factors()

    def _build_factors(self):
        n = len(self.x) - 1
        s = 0
        self.poly_factors = []
        for i in range(n + 1):
            if i > self.q // 2 and s < n - self.q:
                s += 1

            self.poly_factors.append(PiFactor(self.x, self.q, s))

    def __len__(self):
        return len(self.poly_factors)

    def __getitem__(self, i):
        return self.poly_factors[i]

    def __iter__(self):
        return iter(self.poly_factors)


class Interpolation:
    """A final interpolation of the unknown function at x."""

    def __init__(self, n, q, tol=1e-5, max_iter=2500):
        self.n = n
        self.q = q
        self.endpoints, self.inter = self._find_endpoints(tol, max_iter)

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

            for i, (f1, f2, e1, e2) in self._generate_iter(factors, extrema):
                if i == len(factors) - 2:
                    continue
                if abs(f1(e1)) > abs(f2(e2)):
                    endpoints[i + 2] -= (endpoints[i + 2] - e1) / (2 * h)
                else:
                    endpoints[i + 2] += (e2 - endpoints[i + 2]) / (2 * h)

            if abs(factors[0](-1)) > abs(factors[0](extrema[0])):
                endpoints[1] -= (endpoints[1] + 1) / (2 * h)
            else:
                endpoints[1] += (extrema[0] - endpoints[1]) / (2 * h)

            if abs(factors[-1](1)) < abs(factors[-1](extrema[-1])):
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
                    poly_factors[i].derivative,
                    poly_factors[i].second_derivative,
                )
                for i in range(len(endpoints) - 2)
            ]
        )

    def _generate_iter(self, factors, extrema):
        return enumerate(
            zip(
                factors.poly_factors[:-1],
                factors.poly_factors[1:],
                extrema[:-1],
                extrema[1:],
            )
        )

    def _extrema_diff(self, factors, extrema):
        """Find the sum of the adjacent extrema differences."""
        iterator = self._generate_iter(factors, extrema)

        return (
            sum(
                abs(abs(f1(e1)) - abs(f2(e2)))
                for i, (f1, f2, e1, e2) in iterator
            )
            + abs(abs(factors[0](-1)) - abs(factors[0](extrema[0])))
            + abs(abs(factors[-1](extrema[-1])) - abs(factors[-1](1)))
        )

    def __getitem__(self, i):
        return self.inter[i]

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


def newton_raphson(x, f, df, tol=1e-8):
    """Find the root of f(x) = 0 using Newton-Raphson."""
    x0 = x
    while True:
        x1 = x0 - f(x0) / df(x0)
        if abs(x1 - x0) < tol:
            return x1
        x0 = x1
