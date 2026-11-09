"""Referee (I)-a2: common L-drift at fixed z, sensitivity to dropping L=6 / using only z>=1.4 rows / bootstrap p-value."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
groups = []
for p in sorted(pts.values(), key=lambda p: p["z"]):
    for g in groups:
        if abs(g[0]["z"] - p["z"]) / p["z"] < 3e-3: g.append(p); break
    else: groups.append([p])
groups = [sorted(g, key=lambda q: q["L"]) for g in groups if len(g) >= 3 and g[0]["z"] < 20]
def prof(gam, rows, L0=18.0):
    t = 0.0
    for L, y, s in rows:
        m = 1 + gam * np.log(L / L0); w = 1 / s ** 2; f = (w * m * y).sum() / (w * m * m).sum(); t += (w * (y - f * m) ** 2).sum()
    return t
def scan(rows, label):
    gs = np.linspace(-0.3, 0.3, 1201); ch = np.array([prof(g, rows) for g in gs]); i = ch.argmin()
    band = gs[ch < ch[i] + 1]; npts = sum(len(r[0]) for r in rows)
    print("  %-42s best g=%+.3f [1sig %+.3f..%+.3f]  dchi2(g=0)=%.2f (%.1f sig)  chi2(g=0)=%.1f/%d" % (
        label, gs[i], band.min(), band.max(), prof(0, rows) - ch[i], np.sqrt(max(prof(0, rows) - ch[i], 0)), prof(0, rows), npts - len(rows)))
def mk(minL=0, zmin=0, zmax=99, only=None):
    rows = []
    for g in groups:
        gg = [q for q in g if q["L"] >= minL]
        if len(gg) >= 2 and zmin <= g[0]["z"] <= zmax:
            rows.append((np.array([q["L"] for q in gg], float), np.array([q["mean"] for q in gg]), np.array([q["sem"] for q in gg])))
    return rows
print("D(L,z) = f(z)*(1+g ln(L/18)) at the matched-z rows (z=0.089..10.55)")
scan(mk(), "all rows, all L")
scan(mk(minL=12), "drop L=6")
scan(mk(minL=18), "L>=18 only")
scan(mk(zmin=1.0), "rows z>=1.4")
scan(mk(zmin=3.0), "rows z>=3.2 (z=3.21,5.93,10.55)")
scan(mk(zmin=3.0, minL=12), "rows z>=3.2, drop L=6")
scan(mk(zmax=1.0), "rows z<=0.8 (theory: exact L-independence)")
# parametric bootstrap p-value of the drift under the null g=0 (f_row fitted)
rng = np.random.default_rng(8); rows = mk(); obs = prof(0, rows) - min(prof(g, rows) for g in np.linspace(-0.3, 0.3, 241))
cnt = 0; nb = 1500
for _ in range(nb):
    rr = []
    for L, y, s in rows:
        w = 1 / s ** 2; f = (w * y).sum() / w.sum(); rr.append((L, f + rng.normal(0, s), s))
    d = prof(0, rr) - min(prof(g, rr) for g in np.linspace(-0.3, 0.3, 241)); cnt += d >= obs
print("  Delta chi2 observed %.2f; MC p-value under g=0: %.3f" % (obs, cnt / nb))
