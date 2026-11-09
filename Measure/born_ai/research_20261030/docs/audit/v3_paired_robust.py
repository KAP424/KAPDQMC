import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import crossover_structure as cs
pts, base = cs.load_points()
byL = cs.per_L(pts)
levels = [0.3, 0.4, 0.5, 0.8]
use = {0.3: [6, 12, 18, 24, 30], 0.4: [6, 12, 18, 24, 30, 36], 0.5: [6, 12, 18, 24, 30, 36], 0.8: [6, 18, 24, 30, 36]}
def cross_robust(zs, hs, level):
    lz = np.log(zs)
    for i in range(len(hs) - 1):
        if hs[i] < level <= hs[i + 1]:
            return np.exp(lz[i] + (level - hs[i]) * (lz[i + 1] - lz[i]) / (hs[i + 1] - hs[i]))
    # fallback: first point already above level -> extrapolate through the first two points
    if hs[0] >= level:
        return np.exp(lz[0] + (level - hs[0]) * (lz[1] - lz[0]) / (hs[1] - hs[0]))
    return np.nan
rng = np.random.default_rng(31415)
NB = 3000
P = {h: np.empty(NB) for h in levels}
for b in range(NB):
    bm = {(p["L"], p["beta"]): p["D"][rng.integers(0, p["n"], p["n"])].mean() for p in pts}
    for h in levels:
        y = []
        for L in use[h]:
            zs = np.array([q["z"] for q in byL[L]]); hs = np.array([bm[(q["L"], q["beta"])] / q["Dgs"] for q in byL[L]])
            y.append(np.log(cross_robust(zs, hs, h)))
        P[h][b] = np.polyfit(np.log(use[h]), y, 1)[0] if np.all(np.isfinite(y)) else np.nan
p0 = {}
for h in levels:
    y = []
    for L in use[h]:
        zs = np.array([q["z"] for q in byL[L]]); hs = np.array([q["mean"] / q["Dgs"] for q in byL[L]])
        y.append(np.log(cross_robust(zs, hs, h)))
    p0[h] = np.polyfit(np.log(use[h]), y, 1)[0]
print("finite:", {h: int(np.isfinite(P[h]).sum()) for h in levels})
print("sd(p):", {h: round(float(np.nanstd(P[h], ddof=1)), 3) for h in levels})
for h1, h2 in ((0.4, 0.5), (0.5, 0.8), (0.3, 0.5)):
    d = (P[h2] - P[h1]); d = d[np.isfinite(d)]
    print("p(%.1f)-p(%.1f): point %.4f ; boot mean %.4f ; sd %.4f -> point/sd %.2f ; n=%d" % (h2, h1, p0[h2] - p0[h1], d.mean(), d.std(ddof=1), (p0[h2] - p0[h1]) / d.std(ddof=1), d.size))
