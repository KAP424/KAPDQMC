import math
import numpy as np


def head(s):
    print()
    print('=== ' + s + ' ===')


def logw(a, beta):
    w = [0.0]
    for j in range(1, a + 1):
        w.append(w[-1] + 2 * math.log((a - j + 1) / j) - 4 * beta)
    return np.array(w)


def bessel_I(n, x):
    s, k = 0.0, 0
    term = (x / 2) ** n / math.factorial(n)
    while True:
        s += term
        k += 1
        term *= (x / 2) ** 2 / (k * (k + n))
        if k > 5 and term < 1e-17 * s:
            return s


head('(6) j distribution: P1/P0 = z^2 and <j>')
for L, beta in ((6, 3.0), (12, 1.0), (18, 1.0), (18, 2.0), (36, 1.0)):
    a = L * L
    t = math.exp(-2 * beta)
    z = a * t
    w = logw(a, beta)
    p = np.exp(w - w.max())
    p /= p.sum()
    mean = float((p * np.arange(a + 1)).sum())
    r = bessel_I(1, 2 * z) / bessel_I(0, 2 * z)
    print('L=%d beta=%.1f z=%.3f P1/P0=%.6g z^2=%.6g' % (L, beta, z, p[1] / p[0], z * z))
    print('   <j>=%.4f  z*I1/I0=%.4f  z-1/4=%.4f  z-1/2=%.4f  | <j>/a=%.4f  t=%.4f  t/(1+t)=%.4f'
          % (mean, z * r, z - 0.25, z - 0.5, mean / a, t, t / (1 + t)))
for z in (5.0, 10.0, 43.85, 100.0):
    r = bessel_I(1, 2 * z) / bessel_I(0, 2 * z)
    print('Bessel limit z=%.2f: z*I1/I0=%.5f  z-1/4=%.5f  z-1/2=%.5f' % (z, z * r, z - 0.25, z - 0.5))

head('kappa_a/t: where it equals 3, and its peak')
lo, hi = 0.05, 5.0
for _ in range(60):
    mid = 0.5 * (lo + hi)
    if bessel_I(1, 2 * mid) / bessel_I(0, 2 * mid) < 0.5:
        lo = mid
    else:
        hi = mid
print('a->inf: I1(2z)/I0(2z)=1/2 at z=%.4f' % lo)


def logZ(a, lq, tab):
    lw = tab[a] + np.arange(a + 1) * lq
    m = lw.max()
    return m + math.log(np.exp(lw - m).sum())


zs = np.logspace(-2, 1.7, 300)
for L in (6, 12, 18, 24, 36, 60):
    a = L * L
    lg = np.array([math.lgamma(i + 1) for i in range(a + 2)])
    tabs = {}
    for aa in (a, a - 1):
        k = np.arange(aa + 1)
        tabs[aa] = 2 * (lg[aa] - lg[k] - lg[aa - k])
    ys = []
    for z in zs:
        t = z / a
        lq = 2 * math.log(t)
        kap = 1 - (1 - t) ** 2 * math.exp(logZ(a - 1, lq, tabs) - logZ(a, lq, tabs))
        ys.append(kap / t)
    ys = np.array(ys)
    if ys.max() < 3.0:
        print('L=%d: kappa/t never reaches 3, max=%.4f' % (L, ys.max()))
        continue
    i3 = int(np.argmax(ys >= 3.0))
    zc = math.exp(np.interp(3.0, ys[i3 - 1:i3 + 1], np.log(zs[i3 - 1:i3 + 1])))
    print('L=%d: kappa/t(z=0.01)=%.4f  midpoint(=3) z=%.4f  peak z=%.3f peak value=%.4f'
          % (L, ys[0], zc, zs[int(np.argmax(ys))], ys.max()))
