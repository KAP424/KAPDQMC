"""Referee (I)-b2: recompute Delta(beta) (L>=24), additive vs multiplicative chi2, and the drop-L18 / drop-L12 excess."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
def fixed(beta, Lmin=24): return sorted([p for p in pts.values() if abs(p["beta"] - beta) < 2e-3 and p["L"] >= Lmin], key=lambda p: p["L"])
for b in (0.5, 1.0, 1.5, 2.0):
    ps = fixed(b); dl = np.array([p["Dgs"] - p["mean"] for p in ps]); s = np.array([p["sem"] for p in ps]); gs = np.array([p["Dgs"] for p in ps])
    m, e = c.wmean(dl, s); chi_a = (((dl - m) / s) ** 2).sum()
    k = np.sum(dl * gs / s ** 2) / np.sum(gs ** 2 / s ** 2); chi_m = (((dl - k * gs) / s) ** 2).sum()
    # trend of Delta with ln L
    x = np.log([p["L"] for p in ps]); X = np.column_stack([np.ones_like(x), x]); W = 1 / s ** 2; cov = np.linalg.inv(X.T @ (X * W[:, None])); bb = cov @ (X.T @ (W * dl))
    print("beta=%.1f: Delta=%+.4f +- %.4f  add chi2=%.2f  mult(k=%.4f+-%.4f) chi2=%.2f | Delta(L) = %s | dDelta/dlnL = %+.3f +- %.3f -> Delta(36)-Delta(24) = %+.3f +- %.3f" % (
        b, m, e, chi_a, k, 1 / np.sqrt(np.sum(gs ** 2 / s ** 2)), chi_m, np.round(dl, 4), bb[1], np.sqrt(cov[1, 1]), bb[1] * np.log(1.5), np.sqrt(cov[1, 1]) * np.log(1.5)))
sel = sorted([p for p in pts.values() if p["beta"] <= 1.0 + 1e-9 and p["L"] >= 12], key=lambda p: p["L"])
def st(ps, lab):
    m, e = c.wmean([p["mean"] - p["Dgs"] for p in ps], [p["sem"] for p in ps]); print("  %-28s %+.4f +- %.4f (%.1f sig, n=%d)" % (lab, m, e, m / e, len(ps)))
st(sel, "all"); st([p for p in sel if p["L"] not in (18,)], "drop both L=18 points"); st([p for p in sel if p["L"] not in (12, 18)], "drop L=12 and L=18 (=L>=24)")
st([p for p in sel if p["L"] not in (12, 36)], "drop L=12, L=36"); st([p for p in sel if p["L"] <= 24], "L<=24 only")
