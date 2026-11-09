"""Referee (I)-d2: independent re-implementation of contour exponents p(h) + common-L-set variants."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
rng = np.random.default_rng(777)
byL = {}
for p in pts.values(): byL.setdefault(p["L"], []).append(p)
for L in byL: byL[L].sort(key=lambda p: p["z"])
levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def crossing(zs, hs, lev, maxw=1.25):
    lz = np.log(zs)
    for i in range(len(hs) - 1):
        if hs[i] < lev <= hs[i + 1]:
            w = lz[i + 1] - lz[i]
            if w > maxw: return None
            return lz[i] + (lev - hs[i]) * w / (hs[i + 1] - hs[i])
    return None
def table(means=None):
    out = {}
    for L, ps in byL.items():
        zs = np.array([p["z"] for p in ps]); hs = np.array([(means[(p["L"], p["beta"])] if means else p["mean"]) / p["Dgs"] for p in ps])
        for h in levels:
            x = crossing(zs, hs, h)
            if x is not None: out[(h, L)] = x
    return out
T = table()
print("valid crossings (L: z_h) :")
for h in levels:
    print("  h=%.1f  " % h + "  ".join("L%d:%.2f" % (L, np.exp(T[(h, L)])) for L in sorted(byL) if (h, L) in T))
def pfit(T, h, Lset=None):
    Ls = [L for L in sorted(byL) if (h, L) in T and (Lset is None or L in Lset)]
    if len(Ls) < 3: return None, Ls
    return np.polyfit(np.log(Ls), [T[(h, L)] for L in Ls], 1)[0], Ls
NB = 1500
boots = []
for _ in range(NB):
    m = {k: p["cols"]["D"][rng.integers(0, p["n"], p["n"])].mean() for k, p in pts.items()}
    boots.append(table(m))
def report(Lset, label):
    print("\n%s" % label)
    P0 = {}
    for h in levels:
        p0, Ls = pfit(T, h, Lset)
        if p0 is None: print("   h=%.1f  Ls=%s  -> not enough" % (h, Ls)); continue
        pb = [pfit(Tb, h, Lset)[0] for Tb in boots]; pb = np.array([x for x in pb if x is not None])
        P0[h] = (p0, pb.std(), Ls, len(pb))
        print("   h=%.1f  Ls=%-22s p=%.3f +- %.3f (nboot ok %d)" % (h, Ls, p0, pb.std(), len(pb)))
    return P0
report(None, "A) summary-style (all valid L, per threshold)")
report({18, 24, 30, 36}, "B) common L-set {18,24,30,36}")
report({12, 18, 24, 30}, "C) common L-set {12,18,24,30}")
report({24, 30, 36}, "D) L>=24 only (3 sizes)")
# paired difference with bootstrap on common set B
def pair(h1, h2, Lset):
    d0 = pfit(T, h2, Lset)[0] - pfit(T, h1, Lset)[0]
    db = []
    for Tb in boots:
        a, b = pfit(Tb, h1, Lset)[0], pfit(Tb, h2, Lset)[0]
        if a is not None and b is not None: db.append(b - a)
    return d0, np.std(db), len(db)
for (h1, h2, Ls) in ((0.5, 0.9, {18, 24, 30, 36}), (0.4, 0.8, {18, 24, 30, 36}), (0.5, 0.8, {18, 24, 30, 36}), (0.6, 0.9, {18, 24, 30, 36}), (0.5, 0.9, {12, 18, 24, 30}), (0.5, 0.8, {24, 30, 36})):
    d0, e, n = pair(h1, h2, Ls)
    print("p(%.1f)-p(%.1f) on L=%s: %+.3f +- %.3f  (%.1f sigma; nboot %d)" % (h2, h1, sorted(Ls), d0, e, d0 / e if e > 0 else np.nan, n))
