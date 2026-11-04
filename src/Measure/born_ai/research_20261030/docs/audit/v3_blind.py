import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
rows = [dict(L=k[0], beta=k[1], z=e["z"][-1], mean=e["mean"], sem=e["sem"], n=e["n"], src=e["files"][0][0]) for k, e in pts.items()]

def blind(rows, tol=1e-3):
    ps = []
    out = []
    for r in sorted([r for r in rows if r["L"] > 24], key=lambda r: (r["L"], r["z"])):
        ref = [q for q in rows if q["L"] <= 24 and abs(q["z"] - r["z"]) < tol]
        if not ref:
            continue
        w = np.array([1 / q["sem"] ** 2 for q in ref]); m = np.array([q["mean"] for q in ref])
        pred = (w * m).sum() / w.sum(); sp = 1 / np.sqrt(w.sum())
        p = (pred - r["mean"]) / np.hypot(sp, r["sem"])
        ps.append(p); out.append((r["L"], r["z"], pred, r["mean"], r["sem"], p, len(ref)))
    return np.array(ps), out

# (a) exactly as the summary describes: matched_z + largez only  (gapfill excluded)
rows_a = [r for r in rows if r["src"] != "data_gapfill"]
ps, out = blind(rows_a)
print("(a) no gapfill: npred", len(ps), "rms %.3f max|pull| %.3f chi2/dof %.3f" % (np.sqrt(np.mean(ps**2)), np.abs(ps).max(), np.sum(ps**2)/len(ps)))
for o in out: print("   L=%d z=%.4f pred=%.4f meas=%.4f sem=%.4f pull=%+.3f nref=%d" % o)
# (b) all data
ps2, out2 = blind(rows)
print("(b) incl gapfill: npred", len(ps2), "rms %.3f max|pull| %.3f chi2/dof %.3f" % (np.sqrt(np.mean(ps2**2)), np.abs(ps2).max(), np.sum(ps2**2)/len(ps2)))
for o in out2[-6:]: print("   L=%d z=%.4f pred=%.4f meas=%.4f sem=%.4f pull=%+.3f nref=%d" % o)

# fixed-z rows chi2/dof against a constant (all L at same z), (a) without gapfill
print()
zl = sorted({round(r["z"], 3) for r in rows_a})
groups = {}
for r in rows_a:
    for zt in list(groups):
        if abs(zt - r["z"]) < 1e-3:
            groups[zt].append(r); break
    else:
        groups[r["z"]] = [r]
from math import erf, sqrt
def chi2_sf(x, k):
    # survival function via series for the regularized upper gamma, k integer dof
    # use numerical integration of chi2 pdf via math (simple)
    import math
    # P(chi2_k > x) = Q(k/2, x/2); for integer k use finite sums
    if k % 2 == 0:
        s = 0.0; term = 1.0
        for i in range(k // 2):
            if i > 0: term *= (x / 2) / i
            s += term
        return math.exp(-x / 2) * s
    else:
        # odd k: Q = erfc(sqrt(x/2)) + exp(-x/2)*sum
        import math
        s = 0.0
        term = math.sqrt(x / 2) / math.gamma(1.5) * 0 + 0
        q = math.erfc(math.sqrt(x / 2))
        t = math.exp(-x / 2) / math.sqrt(math.pi) * math.sqrt(2 * x) / 1.0  # placeholder
        # direct numeric integration fallback
        N = 200000; hi = max(x * 6, 60)
        xs = np.linspace(x, hi, N)
        pdf = xs ** (k / 2 - 1) * np.exp(-xs / 2) / (2 ** (k / 2) * math.gamma(k / 2))
        return float(np.trapz(pdf, xs)) if hasattr(np, "trapz") else float(np.trapezoid(pdf, xs))
def nsig_two_sided(p):
    # inverse normal two-sided: find s such that erfc(s/sqrt2)=p
    lo, hi = 0.0, 40.0
    import math
    for _ in range(200):
        mid = (lo + hi) / 2
        if math.erfc(mid / math.sqrt(2)) > p: lo = mid
        else: hi = mid
    return lo
print("fixed-z rows (no gapfill), chi2/dof against a constant:")
for zt in sorted(groups):
    g = groups[zt]
    if len(g) < 3: 
        continue
    w = np.array([1 / q["sem"] ** 2 for q in g]); m = np.array([q["mean"] for q in g])
    mu = (w * m).sum() / w.sum(); chi = (w * (m - mu) ** 2).sum(); dof = len(g) - 1
    pv = chi2_sf(chi, dof)
    print("  z=%.3f  Ls=%s  chi2=%.2f dof=%d chi2/dof=%.2f  p=%.4f  ~%.2f sigma(two-sided)" % (zt, [q["L"] for q in sorted(g, key=lambda q: q["L"])], chi, dof, chi/dof, pv, nsig_two_sided(pv)))
