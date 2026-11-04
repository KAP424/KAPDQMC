import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import crossover_structure as cs
pts, base = cs.load_points(); byL = cs.per_L(pts)
use = {0.3: [6, 12, 18, 24, 30], 0.4: [6, 12, 18, 24, 30], 0.5: [6, 12, 18, 24, 30]}
rng = np.random.default_rng(2718)
NB = 3000
P = {h: np.empty(NB) for h in use}
def p_of(bm, h):
    y = []
    for L in use[h]:
        zs = np.array([q["z"] for q in byL[L]]); hs = np.array([bm[(q["L"], q["beta"])] / q["Dgs"] for q in byL[L]])
        z, _ = cs.crossing(zs, hs, h)
        if z is None: return np.nan
        y.append(np.log(z))
    return np.polyfit(np.log(use[h]), y, 1)[0]
m0 = {(p["L"], p["beta"]): p["mean"] for p in pts}
p0 = {h: p_of(m0, h) for h in use}
for b in range(NB):
    bm = {(p["L"], p["beta"]): p["D"][rng.integers(0, p["n"], p["n"])].mean() for p in pts}
    for h in use: P[h][b] = p_of(bm, h)
print("L<=30 only: point p:", {h: round(p0[h], 3) for h in use}, " sd:", {h: round(float(np.nanstd(P[h], ddof=1)), 3) for h in use}, " finite:", {h: int(np.isfinite(P[h]).sum()) for h in use})
for h1, h2 in ((0.4, 0.5), (0.3, 0.4), (0.3, 0.5)):
    d = P[h2] - P[h1]; d = d[np.isfinite(d)]
    print("p(%.1f)-p(%.1f) (L<=30): point %.4f boot sd %.4f -> %.2f sigma" % (h2, h1, p0[h2] - p0[h1], d.std(ddof=1), (p0[h2] - p0[h1]) / d.std(ddof=1)))
