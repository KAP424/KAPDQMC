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
def nsigma(p):
    lo, hi = 0.0, 40.0
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > p: lo = mid
        else: hi = mid
    return lo
