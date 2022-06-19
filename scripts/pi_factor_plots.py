from finite_diff.finite_diff import PolyFactor, Interpolation
import numpy as np
import matplotlib.pyplot as plt

plt.rcParams.update({"font.size": 11})

# Uniform Grid N = 10 fig 2. a)
y = np.linspace(-1, 1, 11)
x = [np.linspace(s, e, 101) for s, e in zip(y[:-1], y[1:])]

p = PolyFactor(y, 10)
u = [abs(np.vectorize(p, excluded=["i"])(x[i], i)) for i in range(len(x))]

for a, b in zip(x, u):
    plt.plot(a, b, color="black")
plt.yscale("log")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$|\\pi(x)|$", fontsize=16)
plt.ylim(1e-5, 1e-2)
plt.show()

# Chebshev Roots N = 10 fig 2. b)
y = 2 * np.arange(1, 11) - 1
y = -np.cos(y * np.pi / (2 * 10))
# y = np.concatenate([[-1], y, [1]])

x = [np.linspace(s, e, 1001) for s, e in zip(y[:-1], y[1:])]

p = PolyFactor(y, 9)
u = [abs(np.vectorize(p, excluded=["i"])(x[i], i)) for i in range(len(x))]

for a, b in zip(x, u):
    plt.plot(a, b, color="black")
plt.yscale("log")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$|\\pi(x)|$", fontsize=16)
plt.ylim(1e-5, 1e-2)
plt.show()

# Hermanns-Hernández Grid N = 10, q = 6 fig 3. a)
f = Interpolation(10, 6, max_iter=5000)
a = f.inter.x
p = PolyFactor(a, 6)
y = f.endpoints

x = [np.linspace(s, e, 101) for s, e in zip(y[:-1], y[1:])]
u = [abs(np.vectorize(p, excluded=["i"])(x[i], i)) for i in range(len(x))]

for w, z in zip(x, u):
    plt.plot(w, z, color="black")
plt.yscale("log")
plt.xlabel("$x$", fontsize=16)
plt.ylabel("$|\\pi(x)|$", fontsize=16)
plt.ylim(1e-5, 1e-2)
plt.show()

raise Exception("Stop")

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
p = Interpolation(30, 6, max_iter=5000)
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
