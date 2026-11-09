"""Referee (I)-d4: which L / points drive the single-scale rejection? leave-one-L-out + residual table."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np
import ref2_shift_lib as lib

K = 8
print("Leave-one-L-out of the single-scale (shift-only) fit, window h in [0.25,1.15], L in 12..36, K=%d" % K)
base = lib.Prob(0.25, 1.15, 12, K)
for drop in (None, 12, 18, 24, 30, 36):
    pr = lib.Prob(0.25, 1.15, 12, K)
    if drop is not None:
        keep = pr.L != drop
        for a in ("L", "lz", "y", "s"): setattr(pr, a, getattr(pr, a)[keep])
        pr.N = int(keep.sum()); pr.Ls = sorted(set(pr.L)); pr.kn = np.linspace(pr.lz.min() - 0.2, pr.lz.max() + 0.2, K)
    c1, sh, _ = pr.run(False, 5)
    d1 = pr.N - K - (len(pr.Ls) - 1)
    print("  drop L=%-4s N=%2d chi2=%5.1f dof=%2d  chi2/dof=%.2f  (p-value ~ %s)" % (drop, pr.N, c1, d1, c1 / max(d1, 1), "n/a"))
# residual table of the full fit
pr = lib.Prob(0.25, 1.15, 12, K)
c1, sh, _ = pr.run(False, 6)
u = pr.lz - np.array([sh[l] for l in pr.L]); B = lib.hat(u, pr.kn); W = 1 / pr.s ** 2
A = B.T @ (B * W[:, None]) + 1e-8 * np.eye(K); b = np.linalg.solve(A, B.T @ (W * pr.y)); res = (pr.y - B @ b) / pr.s
print("\nResiduals of the best shift-only fit (L, z, h, pull):  total chi2 %.1f" % c1)
for i in np.argsort(-np.abs(res))[:10]:
    print("   L=%2d z=%8.2f h=%.3f  pull=%+.2f" % (pr.L[i], np.exp(pr.lz[i]), pr.y[i], res[i]))
print("  chi2 share by L:", {int(l): round(float((res[pr.L == l] ** 2).sum()), 1) for l in pr.Ls})
print("  shifts ln z*(L):", {int(l): round(sh[l], 2) for l in pr.Ls}, "  -> slope of shift vs ln L (L=12..36): %.2f" % np.polyfit(np.log(pr.Ls), [sh[l] for l in pr.Ls], 1)[0])
# amplitude + shift: what are a_L?
c2, sh2, am = pr.run(True, 4)
print("\nshift+amp fit chi2 %.1f; amplitudes a_L: %s; shifts %s" % (c2, {int(l): round(am[l], 3) for l in pr.Ls}, {int(l): round(sh2[l], 2) for l in pr.Ls}))
