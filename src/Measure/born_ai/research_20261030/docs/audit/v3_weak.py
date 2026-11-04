import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
base = baseline()
Dgs = {int(k): v["D"] for k, v in base.items()}

def P(L, b):
    return pts[(L, round(b, 4))]

def fixed(b):
    sel = [(L, e) for (L, bb), e in pts.items() if abs(bb - b) < 2e-3]
    return sorted(sel, key=lambda t: t[0])

print("== fixed-beta table: h, Delta, errors ==")
res = {}
for b in (0.5, 1.0, 1.5, 2.0):
    sel = fixed(b)
    print("beta=%.1f  L: " % b + " ".join("%d" % L for L, _ in sel))
    print("   h  : " + " ".join("%.4f" % (e["mean"] / Dgs[L]) for L, e in sel), "  | sem/Dgs: " + " ".join("%.4f" % (e["sem"] / Dgs[L]) for L, e in sel))
    big = [(L, e) for L, e in sel if L >= 24]
    y = np.array([e["mean"] for L, e in big]); s = np.array([e["sem"] for L, e in big]); g = np.array([Dgs[L] for L, e in big])
    x = np.log([L for L, e in big])
    w = 1 / s**2
    dl = g - y
    add = (w * dl).sum() / w.sum(); sadd = 1 / np.sqrt(w.sum())
    chi_add = (w * (dl - add) ** 2).sum()
    k = (w * dl * g).sum() / (w * g * g).sum(); sk = 1 / np.sqrt((w * g * g).sum())
    chi_mul = (w * (dl - k * g) ** 2).sum()
    # slope
    X = np.column_stack([x, np.ones_like(x)]); W = np.diag(w)
    cov = np.linalg.inv(X.T @ W @ X); c = cov @ X.T @ W @ y
    cgs = np.polyfit(x, g, 1)[0]
    ratio, sratio = c[0] / cgs, np.sqrt(cov[0, 0]) / cgs
    print("   Delta(L>=24 const) = %+.4f +- %.4f ; chi2 add %.2f, mult k=%.3f+-%.3f chi2 %.2f ; slope %.4f+-%.4f; gs-slope %.4f; ratio %.3f +- %.3f"
          % (add, sadd, chi_add, k, sk, chi_mul, c[0], np.sqrt(cov[0, 0]), cgs, ratio, sratio))
    res[b] = (ratio, sratio, add, sadd)
print()
rs = np.array([res[b][0] for b in (0.5, 1.0, 1.5)]); ss = np.array([res[b][1] for b in (0.5, 1.0, 1.5)])
w = 1 / ss**2
m = (w * rs).sum() / w.sum(); sm = 1 / np.sqrt(w.sum())
print("combined slope ratio beta<=1.5 (invvar of ratios): %.3f +- %.3f ; (1-m)/sm = %.2f sigma" % (m, sm, (1 - m) / sm))
# alt: combine slopes absolute then divide (same gs slope)
rs4 = np.array([res[b][0] for b in (0.5, 1.0, 1.5, 2.0)]); ss4 = np.array([res[b][1] for b in (0.5, 1.0, 1.5, 2.0)])
w4 = 1 / ss4**2
m4 = (w4 * rs4).sum() / w4.sum(); sm4 = 1 / np.sqrt(w4.sum())
print("combined slope ratio beta<=2   : %.3f +- %.3f ; (1-m)/sm = %.2f sigma" % (m4, sm4, (1 - m4) / sm4))
for b in (0.5, 1.0, 1.5, 2.0):
    print("   beta=%.1f ratio %.3f +- %.3f : %.2f sigma from 1;  %.2f sigma from 0.6; %.2f sigma from 0.75" % (b, res[b][0], res[b][1], (1 - res[b][0]) / res[b][1], (res[b][0] - 0.6) / res[b][1], (res[b][0] - 0.75) / res[b][1]))
print("   combined(<=1.5): from 0.6 %.2f sigma ; from 0.75 %.2f sigma" % ((m - 0.6) / sm, (m - 0.75) / sm))
print("   combined(<=2.0): from 0.6 %.2f sigma ; from 0.75 %.2f sigma" % ((m4 - 0.6) / sm4, (m4 - 0.75) / sm4))

print("\n== beta<=1 excess over ground state, L=12..36 ==")
sel = [((L, b), e) for (L, b), e in sorted(pts.items()) if b <= 1.0 + 1e-9 and L >= 12]
print("n points:", len(sel), [k for k, e in sel])
d = np.array([e["mean"] - Dgs[L] for (L, b), e in sel]); s = np.array([e["sem"] for k, e in sel])
w = 1 / s**2
mu = (w * d).sum() / w.sum(); se = 1 / np.sqrt(w.sum())
chi = (w * (d - mu) ** 2).sum()
print("  excess = %+.4f +- %.4f  (%.2f sigma), chi2/dof = %.2f/%d = %.3f" % (mu, se, mu / se, chi, len(d) - 1, chi / (len(d) - 1)))
h1 = np.array([e["mean"] / Dgs[L] - 1 for (L, b), e in sel]); sh = np.array([e["sem"] / Dgs[L] for (L, b), e in sel])
wh = 1 / sh**2
print("  weighted mean of (h-1) = %+.4f +- %.4f ; unweighted mean (h-1) = %+.4f" % ((wh * h1).sum() / wh.sum(), 1 / np.sqrt(wh.sum()), h1.mean()))
big = [i for i, (k, e) in enumerate(sel) if k[0] >= 30]
w2 = w[big]; mu2 = (w2 * d[big]).sum() / w2.sum(); se2 = 1 / np.sqrt(w2.sum())
print("  L>=30 only (%d pts): %+.4f +- %.4f" % (len(big), mu2, se2))
# also incl. L=24 beta=1.2877? (beta>1) -> no
# excluding L=12
sel6 = [i for i, (k, e) in enumerate(sel) if k[0] >= 18]
w3 = w[sel6]; mu3 = (w3 * d[sel6]).sum() / w3.sum(); print("  L>=18: %+.4f +- %.4f" % (mu3, 1 / np.sqrt(w3.sum())))
# sensitivity: unpooled L=18 beta=1 (two runs as separate points)
print("\n== unpooled L=18 beta=1.0 runs ==")
a1 = read_csv(R + "/data_matched_z/L18_beta1.0000.csv")[1]; a2 = read_csv(R + "/data_matched_largez/L18_beta1.0000.csv")[1]
hd = read_csv(R + "/data_matched_z/L18_beta1.0000.csv")[0]; iD = hd.index("D")
for nm, a in (("matched_z", a1), ("largez", a2)):
    D = a[:, iD]; print("  %s: n=%d D=%.5f +- %.5f  h=%.4f  h-1=%+.3f" % (nm, D.size, D.mean(), D.std(ddof=1) / np.sqrt(D.size), D.mean() / Dgs[18], D.mean() / Dgs[18] - 1))
D1, D2 = a1[:, iD], a2[:, iD]
dd = (D1.mean() - D2.mean()) / np.hypot(D1.std(ddof=1) / np.sqrt(D1.size), D2.std(ddof=1) / np.sqrt(D2.size))
print("  difference between the two runs: %.2f sigma" % dd)

print("\n== z = 43.85: L=12/18/24 pairs ==")
tri = [P(12, 0.5945), P(18, 1.0), P(24, 1.2877)]
for i in range(3):
    for j in range(i + 1, 3):
        a, b = tri[i], tri[j]
        print("  pair (%d,%d): diff %.4f , sigma %.2f" % (i, j, b["mean"] - a["mean"], (b["mean"] - a["mean"]) / np.hypot(a["sem"], b["sem"])))
# with largez run only for L=18
D2 = a2[:, iD]
for i, (nm, mean, sem) in enumerate([("L12", tri[0]["mean"], tri[0]["sem"]), ("L24", tri[2]["mean"], tri[2]["sem"])]):
    print("  using only largez L=18 run vs %s: %.2f sigma" % (nm, (D2.mean() - mean) / np.hypot(D2.std(ddof=1) / np.sqrt(D2.size), sem)) if i == 0 else "  using only largez L=18 run vs %s: %.2f sigma" % (nm, (mean - D2.mean()) / np.hypot(D2.std(ddof=1) / np.sqrt(D2.size), sem)))
print("  D_bar:", ["%.4f" % t["mean"] for t in tri], " h:", ["%.3f" % (t["mean"] / Dgs[L]) for t, L in zip(tri, (12, 18, 24))])
print("  betas:", [0.5945, 1.0, 1.2877])
