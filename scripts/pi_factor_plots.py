from finite_diff.finite_diff import PolyFactor, Interpolation
import numpy as np
import matplotlib.pyplot as plt

# Uniform Grid N = 10 fig 2. a)
y = np.linspace(-1, 1, 11)
x = [np.linspace(s, e, 101) for s, e in zip(y[:-1], y[1:])]

p = PolyFactor(y, 10)
u = [abs(np.vectorize(f)(z)) for f, z in zip(p, x)]

for a, b in zip(x, u):
    plt.plot(a, b, color="black")
plt.yscale("log")
plt.ylim(1e-5, 1e-2)
plt.show()

# Proposed Grid N = 10, q = 6 fig 3. a)
p = Interpolation(10, 6)
a = p.inter.x
f = PolyFactor(a, 6)
y = p.endpoints

x = [np.linspace(s, e, 101) for s, e in zip(y[:-1], y[1:])]
u = [abs(np.vectorize(g)(z)) for g, z in zip(f, x)]

for w, z in zip(x, u):
    plt.plot(w, z, color="black")
plt.yscale("log")
plt.ylim(1e-5, 1e-2)
plt.show()

# Grid Spacings fig 3. b)
delta = [a[i + 1] - a[i] for i in range(len(a) - 1)]
plt.plot(delta, "-", color="black")
plt.plot(delta, "o", color="black")

delta = 2 / 10 * np.ones(10)
plt.plot(delta, "--", color="black")
plt.plot(delta, "o", color="black")

# Chebyshev points here

plt.show()

# Proposed Grid N = 30, q = 6 fig 4. a)
p = Interpolation(30, 6, tol=1e-8)
a = p.inter.x
f = PolyFactor(a, 6)
y = p.endpoints

x = [np.linspace(s, e, 101) for s, e in zip(y[:-1], y[1:])]
u = [abs(np.vectorize(g)(z)) for g, z in zip(f, x)]

for w, z in zip(x, u):
    plt.plot(w, z, color="black")
plt.yscale("log")
plt.ylim(1e-9, 1e-6)
plt.show()

# Proposed Grid Spacing N = 30, q = 6 fig 4. b)
delta = [a[i + 1] - a[i] for i in range(len(a) - 1)]
plt.plot(delta, color="black")
plt.plot(delta, "o", color="black")
plt.show()

# Proposed Grid Spacing fig 5. a)
# n = 50
# for i in range(1, 5):
#     q = 10 * i
#     p = Interpolation(n, q, tol=1e-11)
#     a = p.inter.x
#
#     delta = [a[i + 1] - a[i] for i in range(len(a) - 1)]
#     plt.plot(delta)

# plt.show()

# requires higher efficiency to plot :(
