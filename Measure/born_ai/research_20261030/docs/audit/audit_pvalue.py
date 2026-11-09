import math

def chi2_sf_even(x, k):
    """Survival Q(k/2, x/2) exactly for even k, via the finite sum."""
    m = k // 2
    s = 0.0
    term = 1.0
    for i in range(m):
        if i > 0:
            term *= (x / 2) / i
        s += term
    return math.exp(-x / 2) * s

def sigma_2sided(p):
    # invert the normal two-sided tail by bisection, pure python
    lo, hi = 0.0, 10.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > p:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2

print("chi2/dof vs a constant, N=5 -> dof=4")
print(" %8s %6s %8s %10s %8s" % ("z", "chi2/dof", "chi2", "p", "sigma"))
for zt, cd in ((0.089, 0.91), (0.357, 2.25), (0.803, 1.22), (1.428, 3.25), (3.213, 0.40)):
    chi2 = cd * 4
    p = chi2_sf_even(chi2, 4)
    print(" %8.3f %6.2f %8.2f %10.4f %8.2f" % (zt, cd, chi2, p, sigma_2sided(p)))

print()
print("SUMMARY: 'two z's depart from a constant at 2.2-3.3 sigma'")
print("  max single |pull| observed: z=0.357 -> 2.09, z=1.428 -> 2.32")
