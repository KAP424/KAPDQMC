import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all(); base = baseline(); Dgs = {int(k): v["D"] for k, v in base.items()}
for b in (0.5, 1.0, 1.5, 2.0):
    sel = sorted([(L, e) for (L, bb), e in pts.items() if abs(bb - b) < 2e-3 and L >= 24])
    y = np.array([e["mean"] for L, e in sel]); s = np.array([e["sem"] for L, e in sel]); g = np.array([Dgs[L] for L, e in sel]); x = np.log([L for L, e in sel])
    w = 1 / s**2; dl = g - y
    add = (w * dl).sum() / w.sum(); sadd = 1 / np.sqrt(w.sum())
    chi_add = (w * (dl - add)**2).sum()
    k = (w * dl * g).sum() / (w * g * g).sum()
    chi_mul = (w * (dl - k * g)**2).sum()
    X = np.column_stack([x, np.ones_like(x)]); W = np.diag(w); cov = np.linalg.inv(X.T @ W @ X); c = cov @ X.T @ W @ y
    cgs = np.polyfit(x, g, 1)[0]
    print("beta=%.1f  Delta=%+.5f +- %.5f  chi2_add=%.4f chi2_mul=%.4f  slope=%.5f +- %.5f  gs=%.5f ratio=%.4f +- %.4f" % (b, add, sadd, chi_add, chi_mul, c[0], np.sqrt(cov[0, 0]), cgs, c[0] / cgs, np.sqrt(cov[0, 0]) / cgs))
# the 9-point numbers with more digits
sel = [((L, b), e) for (L, b), e in sorted(pts.items()) if b <= 1.0 + 1e-9 and L >= 12]
d = np.array([e["mean"] - Dgs[L] for (L, b), e in sel]); s = np.array([e["sem"] for k, e in sel]); w = 1 / s**2
mu = (w * d).sum() / w.sum(); se = 1 / np.sqrt(w.sum()); chi = (w * (d - mu)**2).sum()
print("excess %.5f +- %.5f  = %.3f sigma; chi2 %.4f dof 8 -> %.4f" % (mu, se, mu / se, chi, chi / 8))
big = [i for i, (k, e) in enumerate(sel) if k[0] >= 30]
print("L>=30: %.5f +- %.5f" % ((w[big] * d[big]).sum() / w[big].sum(), 1 / np.sqrt(w[big].sum())))
# per-L-group view of the 9 points: excess in L=12-24 vs L>=30
small = [i for i, (k, e) in enumerate(sel) if k[0] < 30]
print("L<=24 (%d pts): %.5f +- %.5f" % (len(small), (w[small] * d[small]).sum() / w[small].sum(), 1 / np.sqrt(w[small].sum())))
# combined slope ratio more digits
res = []
for b in (0.5, 1.0, 1.5):
    sel2 = sorted([(L, e) for (L, bb), e in pts.items() if abs(bb - b) < 2e-3 and L >= 24])
    y = np.array([e["mean"] for L, e in sel2]); s2 = np.array([e["sem"] for L, e in sel2]); x = np.log([L for L, e in sel2]); w2 = 1 / s2**2
    X = np.column_stack([x, np.ones_like(x)]); W = np.diag(w2); cov = np.linalg.inv(X.T @ W @ X); c = cov @ X.T @ W @ y
    res.append((c[0], np.sqrt(cov[0, 0])))
g24 = np.polyfit(np.log([24, 30, 36]), [Dgs[24], Dgs[30], Dgs[36]], 1)[0]
r = np.array([a for a, _ in res]) / g24; sr = np.array([s for _, s in res]) / g24
w = 1 / sr**2; print("combined ratio %.4f +- %.4f ; sigma from 1: %.3f" % ((w * r).sum() / w.sum(), 1 / np.sqrt(w.sum()), (1 - (w * r).sum() / w.sum()) * np.sqrt(w.sum())))
