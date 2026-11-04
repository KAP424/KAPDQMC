from decimal import Decimal, getcontext
import math
getcontext().prec = 60

def Zsum(a, q, jmax=60):
    s = Decimal(0); term_c = Decimal(1)   # C(a,j)
    c = Decimal(1)
    qq = Decimal(1)
    for j in range(0, min(a, jmax) + 1):
        if j > 0:
            c = c * (a - j + 1) / j
            qq = qq * q
        s += c * c * qq
    return s

def kappa_over_t(a, z):
    t = Decimal(z) / Decimal(a)
    q = t * t
    Za = Zsum(a, q)
    Zm = Zsum(a - 1, q)
    kap = 1 - (1 - t) ** 2 * Zm / Za
    return kap / t

def I(n, x):
    x = Decimal(x); h = x / 2; h2 = h * h
    term = Decimal(1) if n == 0 else h
    s = term; k = 0
    while True:
        k += 1
        term = term * h2 / (Decimal(k) * Decimal(k + (0 if n == 0 else 1)))
        s += term
        if term < Decimal("1e-50") * s: break
    return s

def limit(z):
    return 2 * (1 + I(1, 2 * Decimal(z)) / I(0, 2 * Decimal(z)))

def bisect(f, lo, hi, tol=Decimal("1e-14")):
    flo = f(lo)
    for _ in range(80):
        mid = (lo + hi) / 2
        if (f(mid) > 0) == (flo > 0):
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

zlim = bisect(lambda z: limit(z) - 3, Decimal("0.2"), Decimal("1.5"))
print("limit z_half (a->inf): %.6f" % float(zlim))
zs = []
for L in (6, 12, 18, 24, 30, 36):
    a = L * L
    zh = bisect(lambda z: kappa_over_t(a, z) - 3, Decimal("0.2"), Decimal("1.5"))
    zs.append(float(zh))
    print("L=%2d a=%4d z_half=%.5f   (z_half - z_inf)*L^2 = %.4f" % (L, a, float(zh), (float(zh) - float(zlim)) * L * L))
print("z_half list rounded to 3 dp:", [round(v, 3) for v in zs])
print("limits: small z -> %.6f ; large z (z=40) -> %.6f" % (float(limit(Decimal("1e-8"))), float(limit(Decimal(40)))))
print()
for L, z in ((36, 3.2124628209596007), (36, 3.21), (36, 3.2125), (30, 0.08899231719073675), (30, 1.4278856209340558)):
    a = L * L
    kt = kappa_over_t(a, Decimal(repr(z))); li = limit(Decimal(repr(z)))
    print("L=%d z=%.4f  kappa/t=%.6f  limit=%.6f  rel.err=%.3e" % (L, z, float(kt), float(li), float((li - kt) / li)))
print("\nrel.err * a at z=3.21 for several a (should be constant if ~ 1/a):")
for a in (36, 144, 324, 576, 900, 1296, 5184, 20736):
    kt = kappa_over_t(a, Decimal("3.21")); li = limit(Decimal("3.21"))
    e = (li - kt) / li
    print("  a=%6d rel.err=%.4e   rel.err*a=%.4f" % (a, float(e), float(e * a)))
