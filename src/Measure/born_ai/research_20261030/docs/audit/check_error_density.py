import numpy as np
from math import lgamma, exp, log


def jdist(a, beta):
    t = exp(-2.0 * beta)
    q = t * t
    lw = np.array([2 * (lgamma(a + 1) - lgamma(j + 1) - lgamma(a - j + 1)) + j * log(q) for j in range(a + 1)])
    p = np.exp(lw - lw.max())
    return t, p / p.sum()


def bessel_mean(z, jmax=400):
    lw = np.array([2 * j * log(z) - 2 * lgamma(j + 1) for j in range(jmax)])
    p = np.exp(lw - lw.max())
    p /= p.sum()
    return float((np.arange(jmax) * p).sum()), float(p[0])


print('Bessel limit (a=inf, z fixed): <j> vs z-1/4 and z-1/2')
for z in (2.0, 5.0, 10.0, 20.0, 50.0):
    mj, p0 = bessel_mean(z)
    print(z, f'<j>={mj:.4f} z-1/4={z-0.25:.4f} z-1/2={z-0.5:.4f} P0={p0:.4f}')

print('finite a=L^2: error density <j>/n vs t and t/(1+t)')
for L in (6, 18, 36):
    a = L * L
    for beta in (0.5, 1.0, 1.5, 2.0, 3.0):
        t, p = jdist(a, beta)
        mj = float((np.arange(a + 1) * p).sum())
        print(L, beta, f'z={a*t:.3f} <j>={mj:.3f} j/n={mj/a:.5f} t={t:.5f} t/(1+t)={t/(1+t):.5f} (t-j/n)/(j/n)={(t-mj/a)/(mj/a):+.3f} P0={p[0]:.4f}')

print('beta needed for P(j=0)>=0.9 and 0.5, compared with ln L')
for L in (6, 12, 18, 36):
    a = L * L
    res = []
    for target in (0.9, 0.5):
        lo, hi = 0.0, 8.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            if jdist(a, mid)[1][0] < target:
                lo = mid
            else:
                hi = mid
        res.append(hi)
    print(L, f'ln L={log(L):.3f} beta(P0=0.9)={res[0]:.3f} beta(P0=0.5)={res[1]:.3f}')
