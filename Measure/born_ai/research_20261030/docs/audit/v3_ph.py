import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
base = baseline()
Dgs = {int(k): v["D"] for k, v in base.items()}
levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
keys = sorted(pts, key=lambda k: (k[0], pts[k]["z"][-1]))
byL = {}
for k in keys:
    byL.setdefault(k[0], []).append(k)

def crossing(zs, hs, level):
    lz = np.log(zs)
    for i in range(len(hs) - 1):
        if hs[i] < level <= hs[i + 1]:
            w = lz[i + 1] - lz[i]
            return float(np.exp(lz[i] + (level - hs[i]) * w / (hs[i + 1] - hs[i]))), float(w)
    return None, None

def hvec(L, means):
    return np.array([means[k] / Dgs[L] for k in byL[L]])
zv = {L: np.array([pts[k]["z"][-1] for k in byL[L]]) for L in byL}
m0 = {k: pts[k]["mean"] for k in keys}
print("Level crossings z_h(L) (original data), '*' = bracket wider than ln z = 1.25")
use = {}
for h in levels:
    row = []; Ls = []
    for L in sorted(byL):
        z, w = crossing(zv[L], hvec(L, m0), h)
        if z is None:
            row.append("L%d:  --" % L); continue
        ok = w <= 1.25
        row.append("L%d:%6.2f%s" % (L, z, "" if ok else "*"))
        if ok: Ls.append(L)
    use[h] = Ls
    print("  h=%.1f " % h + "  ".join(row))
def pfit(h, means):
    Ls = use[h]
    y = []
    for L in Ls:
        z, _ = crossing(zv[L], hvec(L, means), h)
        if z is None: return np.nan
        y.append(np.log(z))
    return np.polyfit(np.log(Ls), y, 1)[0]
p0 = {h: pfit(h, m0) for h in levels}
print("p(h):", {h: round(v, 3) for h, v in p0.items()})
print("sizes used:", {h: use[h] for h in levels})

NB = 4000
rng = np.random.default_rng(777)
boot = np.empty((NB, len(levels)))
Dsamp = {k: pts[k]["D"] for k in keys}
for b in range(NB):
    mb = {k: Dsamp[k][rng.integers(0, Dsamp[k].size, Dsamp[k].size)].mean() for k in keys}
    boot[b] = [pfit(h, mb) for h in levels]
np.save(AUD + "/v3_boot_p.npy", boot)
nfin = np.isfinite(boot).sum(axis=0)
print("finite replicates per h:", dict(zip(levels, nfin.tolist())), "of", NB)
print("bootstrap sd of p(h):", {h: round(float(np.nanstd(boot[:, i])), 3) for i, h in enumerate(levels)})
print("bootstrap mean of p(h):", {h: round(float(np.nanmean(boot[:, i])), 3) for i, h in enumerate(levels)})
ix = {h: i for i, h in enumerate(levels)}
def paired(h1, h2):
    d = boot[:, ix[h2]] - boot[:, ix[h1]]
    d = d[np.isfinite(d)]
    pt = p0[h2] - p0[h1]
    return pt, d.mean(), d.std(ddof=1), pt / d.std(ddof=1), d.size
for h1, h2 in ((0.4, 0.5), (0.5, 0.8), (0.3, 0.9), (0.3, 0.5), (0.5, 0.7)):
    pt, bm, sd, sg, n = paired(h1, h2)
    print("  p(%.1f)-p(%.1f): point %.4f, boot mean %.4f, sd %.4f -> %.2f sigma (n=%d)" % (h2, h1, pt, bm, sd, sg, n))
