import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import crossover_structure as cs
pts, base = cs.load_points()
byL = cs.per_L(pts)
levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
# well-bracketed sizes, as in cs.main
use = {}
for h in levels:
    Ls = []
    for L in sorted(byL):
        zs = np.array([p["z"] for p in byL[L]]); hs = np.array([p["mean"] / p["Dgs"] for p in byL[L]])
        z, w = cs.crossing(zs, hs, h)
        if z is not None and w <= cs.MAX_BRACKET: Ls.append(L)
    use[h] = Ls
# reproduce the project's RNG stream (cs.RNG fresh at import; main() builds boot_means first)
boot_means = []
for _ in range(cs.NBOOT):
    bm = {}
    for p in pts:
        idx = cs.RNG.integers(0, p["n"], p["n"])
        bm[(p["L"], p["beta"])] = p["D"][idx].mean()
    boot_means.append(bm)
P = {h: [] for h in levels}      # per-replicate slopes (nan if failed)
for bm in boot_means:
    for h in levels:
        Ls = use[h]; yy = []
        for L in Ls:
            zs = np.array([q["z"] for q in byL[L]]); hs = np.array([bm[(q["L"], q["beta"])] / q["Dgs"] for q in byL[L]])
            z, _ = cs.crossing(zs, hs, h)
            yy.append(np.log(z) if z else np.nan)
        P[h].append(np.polyfit(np.log(Ls), yy, 1)[0] if np.all(np.isfinite(yy)) else np.nan)
P = {h: np.array(v) for h, v in P.items()}
print("project-stream bootstrap sd(p):", {h: round(float(np.nanstd(P[h])), 3) for h in levels}, "(finite counts:", {h: int(np.isfinite(P[h]).sum()) for h in levels}, ")")
p0 = {}
for h in levels:
    Ls = use[h]; y = []
    for L in Ls:
        zs = np.array([q["z"] for q in byL[L]]); hs = np.array([q["mean"] / q["Dgs"] for q in byL[L]])
        y.append(np.log(cs.crossing(zs, hs, h)[0]))
    p0[h] = np.polyfit(np.log(Ls), y, 1)[0]
def pair(h1, h2):
    d = P[h2] - P[h1]; ok = np.isfinite(d); d = d[ok]
    pt = p0[h2] - p0[h1]
    return pt, d.mean(), d.std(ddof=1), d.std(ddof=0), d.size
for h1, h2 in ((0.4, 0.5), (0.5, 0.8)):
    pt, bm, sd1, sd0, n = pair(h1, h2)
    print("p(%.1f)-p(%.1f): point %.4f  boot-mean %.4f  sd %.4f (n=%d)  -> point/sd = %.2f sigma ; bootmean/sd = %.2f sigma" % (h2, h1, pt, bm, sd1, n, pt / sd1, bm / sd1))
