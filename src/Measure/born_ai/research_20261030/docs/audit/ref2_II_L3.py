"""Referee (II)-e: independent exact L=3 evaluation of lim D_bar/q at small t by full enumeration of C(18,9) outputs
(direct Born weights det(Q^T D^2 Q), direct post-measurement states; no Thouless expansion, no node bookkeeping)."""
import sys, itertools, time
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import annealed_gaussian as ag
L = 3
G0 = ag.ground_projector(L, mu=1e-8); ev, Vv = np.linalg.eigh(G0); Q = Vv[:, ev > 0.5]; M, n = Q.shape
masks = ag.region_masks(L); idx = {k: np.flatnonzero(masks[k]) for k in "ABC"}
def S2(Gr):
    nu = np.clip(np.linalg.eigvalsh(Gr), 0, 1); return float(-np.sum(np.log1p(-2 * nu * (1 - nu))))
def run(t):
    tot_p = 0.0; tot_D = 0.0; q = t * t; dvals = []
    subs = list(itertools.combinations(range(M), n))
    for E in subs:
        d = np.ones(M); d[list(E)] = t
        Phi = Q * d[:, None]; Gm = Phi.T @ Phi; p = np.linalg.det(Gm)
        # orthonormalise: G = Phi Gm^{-1} Phi^T
        G = Phi @ np.linalg.solve(Gm, Phi.T)
        s = {k: S2(G[np.ix_(idx[k], idx[k])]) for k in "ABC"}
        D = (2 * s["A"] + s["C"]) / 3 - s["B"]
        tot_p += p; tot_D += p * D
    return tot_D / tot_p, tot_p
t0 = time.time()
res = {}
for t in (4e-3, 2e-3, 1e-3):
    Dm, Z = run(t); res[t] = Dm / (t * t); print("t=%.0e: D_bar/q = %.8f   (Z=%.8f)   [%.0fs]" % (t, res[t], Z, time.time() - t0))
# Richardson in q = t^2
a, b = res[2e-3], res[1e-3]; q1, q2 = (2e-3) ** 2, (1e-3) ** 2
print("linear extrapolation q->0: %.8f ; summary claims 14.6377513890 (A(3)=0.18071 -> *81 = %.6f)" % ((b * q1 - a * q2) / (q1 - q2), 0.18071 * 81))
