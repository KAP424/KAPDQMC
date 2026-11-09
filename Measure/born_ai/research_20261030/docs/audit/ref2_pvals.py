"""chi2 p-values (no scipy) for numbers quoted in the referee report."""
import math
def gammainc_upper_reg(a, x):
    if x <= 0: return 1.0
    if x < a + 1:   # series for P
        ap = a; s = 1.0 / a; d = s
        for _ in range(2000):
            ap += 1; d *= x / ap; s += d
            if abs(d) < abs(s) * 1e-15: break
        return 1.0 - s * math.exp(-x + a * math.log(x) - math.lgamma(a))
    b = x + 1 - a; c = 1e300; d = 1 / b; h = d
    for i in range(1, 2000):
        an = -i * (i - a); b += 2; d = an * d + b; d = 1 / (d if abs(d) > 1e-300 else 1e-300)
        c = b + an / c; c = c if abs(c) > 1e-300 else 1e-300; de = d * c; h *= de
        if abs(de - 1) < 1e-15: break
    return math.exp(-x + a * math.log(x) - math.lgamma(a)) * h
def pchi(chi, dof): return gammainc_upper_reg(dof / 2.0, chi / 2.0)
def nsigma(p):   # two-sided
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > p: lo = mid
        else: hi = mid
    return lo
cases = [("matched-z rows const (all, z<=10.55)", 35.24, 25), ("row z=1.428 const", 13.01, 4), ("z=5.93 prop-Dgs", 33.28, 2), ("z=10.55 prop-Dgs", 14.76, 2),
         ("z=5.93 Dgs-shift", 85.03, 2), ("z=10.55 Dgs-shift", 23.30, 2), ("single-scale shift-only L12-36 K=8", 42.3, 17), ("  same, K=6", 46.0, 19),
         ("  shift-only L12-30", 12.2, 11), ("  shift+amp L12-36 K=8", 24.6, 13), ("  shift+amp L12-36 K=6", 32.6, 15), ("  pure z collapse", 224.2, 21),
         ("  window h<0.97 shift-only K=6", 29.6, 9), ("  window h<0.97 shift+amp K=6", 4.3, 5), ("slope=0.3 b_gs vs 0.6 (dchi2 7.35, 1dof)", 7.35, 1), ("flat vs additive (dchi2 20.7, 1dof)", 20.7, 1),
         ("blind z<=3.2 (8.82/6)", 8.82, 6), ("blind train L<=12 test 18-36 (31.5/16)", 31.5, 16)]
for name, chi, dof in cases:
    p = pchi(chi, dof); print("  %-46s chi2=%7.2f dof=%3d  p=%.2e  (%.1f sigma)" % (name, chi, dof, p, nsigma(p)))
