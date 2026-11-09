"""Referee (II)-b: hypothesis <N_R> = r/2 on honeycomb regions A,B,C with the staggered mu=1e-8 convention,
and the closed-form A(L) = 16/81 + 4 D_V0/L^4 constants (no sampling)."""
import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import annealed_gaussian as ag
print("diag(G) per sublattice, <N_R> - r/2, and C_R = r(M-r)+4V0 combination")
for L in (3, 6, 9, 12, 15, 18):
    G0 = ag.ground_projector(L, mu=1e-8); M = G0.shape[0]
    d = np.diag(G0); dA, dB = d[0::2], d[1::2]
    masks = ag.region_masks(L)
    out = []
    C = {}; V = {}
    for k in "ABC":
        m = masks[k]; r = int(m.sum()); NR = float(d[m].sum()); V[k] = float(np.sum(G0[np.ix_(m, ~m)] ** 2))
        out.append("%s: r=%d <N_R>-r/2=%+.2e" % (k, r, NR - r / 2)); C[k] = r * (M - r) + 4 * V[k]
    CD = (2 * C["A"] + C["C"]) / 3 - C["B"]; DV0 = (2 * V["A"] + V["C"]) / 3 - V["B"]
    print("  L=%2d  diagA in [%.6f,%.6f] diagB in [%.6f,%.6f] | %s | A(L)=%.6f  4DV0/L^4=%.2e  tr G0=%.6f (n=%d)" % (
        L, dA.min(), dA.max(), dB.min(), dB.max(), "; ".join(out), CD / L ** 4, 4 * DV0 / L ** 4, d.sum(), L * L))
# is G0 a projector with M=2n?
G0 = ag.ground_projector(6, mu=1e-8); print("  L=6: ||G0^2-G0||=%.1e, symmetric err %.1e" % (np.abs(G0 @ G0 - G0).max(), np.abs(G0 - G0.T).max()))
# zero-mode structure: A-block rank deficiency of Q (exact structural nodes: any S with more than n-n0 A-sites)
for L in (3, 6, 9, 12):
    Gx = ag.ground_projector(L, mu=1e-8); ev, Vv = np.linalg.eigh(Gx); Q = Vv[:, ev > 0.5]
    QA = Q[0::2]; sv = np.linalg.svd(QA, compute_uv=False)
    print("  L=%2d: n=%d, A-block of Q has rank %d (smallest sv %.1e, #sv<1e-6: %d)" % (L, Q.shape[1], int((sv > 1e-6).sum()), sv[-1], int((sv < 1e-6).sum())))
