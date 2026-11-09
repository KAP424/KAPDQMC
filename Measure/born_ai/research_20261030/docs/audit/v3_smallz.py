import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
base = baseline()

# ---------- D_bar/z^2 table ----------
zrows = [0.089, 0.357, 0.803, 1.428, 3.213]
print("D_bar/z^2 (per-file z) +- sem/z^2")
for zt in zrows:
    out = []
    for L in (6, 12, 18, 24, 30, 36):
        for (LL, b), e in pts.items():
            if LL == L and abs(e["z"][-1] - zt) / zt < 3e-3:
                z = e["z"][-1]
                out.append("L%d: %.4f+-%.4f" % (L, e["mean"] / z**2, e["sem"] / z**2))
    print("  z=%.3f  " % zt + "  ".join(out))
print("  (3-dec rounding as in the summary table:)")
for zt in zrows:
    out = []
    for L in (6, 12, 18, 24, 30, 36):
        for (LL, b), e in pts.items():
            if LL == L and abs(e["z"][-1] - zt) / zt < 3e-3:
                z = e["z"][-1]
                out.append("%.3f" % (e["mean"] / z**2))
    print("  z=%.3f  " % zt + " ".join(out))

# ---------- j>=1 counts at z = 0.089 ----------
print("\nz=0.089: n, #(j>=1), fraction of samples with j>=1, share of sum(D) from j>=1, share of sum(DV) from j>=1")
for L in (6, 12, 18, 24, 30):
    for (LL, b), e in pts.items():
        if LL == L and abs(e["z"][-1] - 0.089) < 1e-3:
            j = e["j"]; D = e["D"]; DV = e["DV"]
            m = j >= 1
            print("  L=%2d n=%4d  #j>=1=%3d (%.2f%%)  share_D(j>=1)=%.1f%%  share_DV=%.1f%%  mean j=%.4f" % (
                L, e["n"], m.sum(), 100 * m.mean(), 100 * D[m].sum() / D.sum(), 100 * DV[m].sum() / DV.sum(), j.mean()))
print("\nj>=1 share of sum(D) at other z (for comparison)")
for zt in (0.357, 0.803, 1.428, 3.213):
    out = []
    for L in (6, 12, 18, 24, 30):
        for (LL, b), e in pts.items():
            if LL == L and abs(e["z"][-1] - zt) / zt < 3e-3:
                m = e["j"] >= 1
                out.append("L%d:%.0f%%" % (L, 100 * e["D"][m].sum() / e["D"].sum()))
    print("  z=%.3f " % zt, " ".join(out))

# ---------- per-region ratios at z = 0.089 ----------
vol = {"SA": 8 / 9, "SB": 56 / 81, "SC": 8 / 9, "D": 16 / 81}
print("\nper-region ratio at z=0.089, inverse-variance pooled over L (as in small_z_mechanism.py), and alternatives")
sel = {}
for (LL, b), e in pts.items():
    if abs(e["z"][-1] - 0.089) / 0.089 < 0.01:
        sel[LL] = e
print("  L used:", sorted(sel))
for c in ("SA", "SB", "SC", "D"):
    ms, es = [], []
    for L in sorted(sel):
        e = sel[L]; z = e["z"][-1]
        x = e[c] / z**2
        ms.append(x.mean()); es.append(x.std(ddof=1) / np.sqrt(x.size))
    ms, es = np.array(ms), np.array(es)
    w = 1 / es**2
    mu, se = (w * ms).sum() / w.sum(), 1 / np.sqrt(w.sum())
    # alternative: pool all samples, plain mean
    allx = np.concatenate([sel[L][c] / sel[L]["z"][-1] ** 2 for L in sorted(sel)])
    print("  %s: invvar-pooled %.4f +- %.4f ; theory %.4f ; ratio %.3f +- %.3f   | plain-pooled mean %.4f +- %.4f ratio %.3f +- %.3f" % (
        c, mu, se, vol[c], mu / vol[c], se / vol[c], allx.mean(), allx.std(ddof=1) / np.sqrt(allx.size), allx.mean() / vol[c], allx.std(ddof=1) / np.sqrt(allx.size) / vol[c]))
# per-L ratios
print("  per-L (ratio to leading order):")
for c in ("SA", "SB", "SC", "D"):
    print("   ", c, " ".join("L%d:%.3f+-%.3f" % (L, (sel[L][c] / sel[L]["z"][-1] ** 2).mean() / vol[c],
                                              (sel[L][c] / sel[L]["z"][-1] ** 2).std(ddof=1) / np.sqrt(sel[L]["n"]) / vol[c]) for L in sorted(sel)))
