"""Referee (I)-c: robustness of the 'D_bar > D_gs' excess at beta<=1, and beta=1.5 plateau power."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np, ref2_common as c
pts, base = c.load()
sel = [p for p in pts.values() if p["beta"] <= 1.0 + 1e-9 and p["L"] >= 12]
sel.sort(key=lambda p: (p["L"], p["beta"]))
print("beta<=1 points (L>=12): ", len(sel))
for p in sel:
    print("  L=%2d beta=%.4f z=%7.2f n=%4d D-Dgs=%+.4f +- %.4f  pull=%+.2f  h-1=%+.4f" % (
        p["L"], p["beta"], p["z"], p["n"], p["mean"] - p["Dgs"], p["sem"], (p["mean"] - p["Dgs"]) / p["sem"], p["mean"] / p["Dgs"] - 1))
def stat(ps, label):
    d = [p["mean"] - p["Dgs"] for p in ps]; s = [p["sem"] for p in ps]
    m, e = c.wmean(d, s); chi = sum(((x - m) / y) ** 2 for x, y in zip(d, s))
    # relative version h-1
    hh = [p["mean"] / p["Dgs"] - 1 for p in ps]; hs = [p["sem"] / p["Dgs"] for p in ps]
    mh, eh = c.wmean(hh, hs)
    print("  %-34s n=%d  <D-Dgs>=%+.4f +- %.4f (%.1f sig) chi2/dof=%.2f | <h-1>=%+.4f +- %.4f (%.1f sig)" % (
        label, len(ps), m, e, m / e, chi / max(len(ps) - 1, 1), mh, eh, mh / eh))
print("\nSubsets:")
stat(sel, "all (L=12-36)")
stat([p for p in sel if p["L"] != 12], "drop L=12")
stat([p for p in sel if p["L"] >= 24], "L>=24 only")
stat([p for p in sel if p["L"] >= 30], "L>=30 only")
stat([p for p in sel if p["L"] in (12, 18)], "L=12,18 only")
stat([p for p in sel if p["L"] == 18], "L=18 only")
stat([p for p in sel if p["L"] == 12], "L=12 only")
stat([p for p in sel if p["beta"] == 0.5 and p["L"] >= 18], "beta=0.5, L>=18")
stat([p for p in sel if p["beta"] == 1.0 and p["L"] >= 18], "beta=1.0, L>=18")
# drop single points (jackknife on points)
print("\nLeave-one-point-out (all 9):")
for i, p in enumerate(sel):
    ps = sel[:i] + sel[i + 1:]
    d = [q["mean"] - q["Dgs"] for q in ps]; s = [q["sem"] for q in ps]
    m, e = c.wmean(d, s); print("   drop L=%d b=%.2f: %+.4f +- %.4f (%.1f sig)" % (p["L"], p["beta"], m, e, m / e))
# L-trend of the excess: fit (D-Dgs) = a + b ln(L/12) and (D-Dgs)=a*exp(-L/L0) not identifiable; use linear in 1/L
L = np.array([p["L"] for p in sel], float); y = np.array([p["mean"] - p["Dgs"] for p in sel]); s = np.array([p["sem"] for p in sel])
for name, x in (("ln L", np.log(L)), ("1/L", 1 / L)):
    X = np.column_stack([np.ones_like(x), x]); W = 1 / s ** 2
    cov = np.linalg.inv(X.T @ (X * W[:, None])); b = cov @ (X.T @ (W * y))
    print("  fit excess = a + b*%s: a=%+.4f+-%.4f b=%+.4f+-%.4f chi2=%.2f (dof %d)  pred excess at L=36: %+.4f, at L=100: %+.4f" % (
        name, b[0], np.sqrt(cov[0, 0]), b[1], np.sqrt(cov[1, 1]), (W * (y - X @ b) ** 2).sum(), len(y) - 2,
        b[0] + b[1] * (np.log(36) if name == "ln L" else 1 / 36), b[0] + b[1] * (np.log(100) if name == "ln L" else 1 / 100)))
# Same set but only beta<=1 pairs pooled per L: how does excess evolve with L?
print("\nPer-L pooled excess (beta<=1):")
for Lv in (12, 18, 24, 30, 36):
    ps = [p for p in sel if p["L"] == Lv]
    d = [p["mean"] - p["Dgs"] for p in ps]; s_ = [p["sem"] for p in ps]; m, e = c.wmean(d, s_)
    print("   L=%d: %+.4f +- %.4f (%.1f sig)  [h-1 = %+.4f]" % (Lv, m, e, m / e, np.mean([p['mean'] / p['Dgs'] - 1 for p in ps])))
