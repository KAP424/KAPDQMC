"""Referee (I)-d7: point-jackknife of the L=36 points in the single-scale (free shift per L) fit; K=8, window h in [0.25,1.15], L=12..36."""
import sys; sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
import numpy as np
import ref2_shift_lib as lib
import ref2_stats as pv
K = 8
def fit_without(drop_idx):
    pr = lib.Prob(0.25, 1.15, 12, K)
    keep = np.ones(pr.N, bool)
    for i in drop_idx: keep[i] = False
    for a in ("L", "lz", "y", "s"): setattr(pr, a, getattr(pr, a)[keep])
    pr.N = int(keep.sum()); pr.Ls = sorted(set(pr.L)); pr.kn = np.linspace(pr.lz.min() - 0.2, pr.lz.max() + 0.2, K)
    c, sh, _ = pr.run(False, 4); dof = pr.N - K - (len(pr.Ls) - 1)
    return c, dof
pr0 = lib.Prob(0.25, 1.15, 12, K)
i36 = [i for i in range(pr0.N) if pr0.L[i] == 36]
c, dof = fit_without([]); print("all: chi2=%.1f dof=%d p=%.1e" % (c, dof, pv.pchi(c, dof)))
for i in i36:
    c, dof = fit_without([i]); print("  drop L=36 z=%7.2f h=%.3f : chi2=%.1f dof=%d p=%.1e" % (np.exp(pr0.lz[i]), pr0.y[i], c, dof, pv.pchi(c, dof)))
worst = i36[int(np.argmax([abs(0) for i in i36]))]
# drop the two largest-residual L=36 points (z=23.7 and z=64.5)
sel = [i for i in i36 if abs(np.exp(pr0.lz[i]) - 23.74) < 0.1 or abs(np.exp(pr0.lz[i]) - 64.52) < 0.1]
c, dof = fit_without(sel); print("  drop L=36 z=23.7 AND z=64.5: chi2=%.1f dof=%d p=%.1e" % (c, dof, pv.pchi(c, dof)))
