"""Referee (I)-d5: sensitivity of p(h) to interpolation scheme and to bracket-width cut."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
byL = {}
for p in pts.values(): byL.setdefault(p["L"], []).append(p)
for L in byL: byL[L].sort(key=lambda p: p["z"])
levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
def crossing(zs, hs, lev, scheme, maxw):
    lz = np.log(zs)
    for i in range(len(hs) - 1):
        if hs[i] < lev <= hs[i + 1]:
            w = lz[i + 1] - lz[i]
            if w > maxw: return None
            if scheme == "lin-h": t = (lev - hs[i]) / (hs[i + 1] - hs[i])
            elif scheme == "log-h": t = (np.log(lev) - np.log(hs[i])) / (np.log(hs[i + 1]) - np.log(hs[i]))
            elif scheme == "logit-h": 
                lg = lambda x: np.log(x / (1.15 - x)); t = (lg(lev) - lg(hs[i])) / (lg(hs[i + 1]) - lg(hs[i]))
            return lz[i] + t * w
    return None
def pexp(scheme, maxw, Lset=None):
    out = {}
    for h in levels:
        Ls, ys = [], []
        for L, ps in sorted(byL.items()):
            if Lset and L not in Lset: continue
            zs = np.array([p["z"] for p in ps]); hs = np.array([p["mean"] / p["Dgs"] for p in ps])
            x = crossing(zs, hs, h, scheme, maxw)
            if x is not None: Ls.append(L); ys.append(x)
        out[h] = np.polyfit(np.log(Ls), ys, 1)[0] if len(Ls) >= 3 else np.nan
    return out
for Lset in (None, {18, 24, 30, 36}):
    print("L-set:", "all valid" if Lset is None else sorted(Lset))
    for scheme in ("lin-h", "log-h", "logit-h"):
        for maxw in (1.25, 1.6):
            r = pexp(scheme, maxw, Lset); print("   %-8s maxw=%.2f  p(h)= %s" % (scheme, maxw, "  ".join("%.2f" % r[h] for h in levels)))
