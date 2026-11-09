"""Referee (II)-c3: exact-zero cluster scan at L=15, 18 (few samples) -- is anything below 1e-9?"""
import sys, time
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import annealed_gaussian as ag
rng = np.random.default_rng(4242)
def dpp(Q):
    V = Q.copy(); M, n = V.shape; F = np.empty(n, int)
    for k in range(n):
        p = np.einsum("ij,ij->i", V, V); p = np.clip(p, 0, None); i = rng.choice(M, p=p / p.sum()); F[k] = i
        r = V[i].copy(); al = -np.copysign(np.linalg.norm(r), r[0]); r[0] -= al; r /= np.linalg.norm(r)
        V = (V - 2.0 * np.outer(V @ r, r))[:, 1:]
    return np.sort(F)
for L, ns in ((15, 150), (18, 80)):
    G0 = ag.ground_projector(L, mu=1e-8); ev, Vv = np.linalg.eigh(G0); Q = Vv[:, ev > 0.5]; M, n = Q.shape
    t0 = time.time(); tot = 0; z9 = 0; z6 = 0; z3 = 0; mn = 1e9; conds = []
    for s in range(ns):
        S = dpp(Q); E = np.setdiff1d(np.arange(M), S); QS = Q[S]; X = np.linalg.solve(QS.T, Q[E].T).T; a = np.abs(X)
        tot += a.size; z9 += int((a < 1e-9).sum()); z6 += int((a < 1e-6).sum()); z3 += int((a < 1e-3).sum()); mn = min(mn, a.min()); conds.append(np.linalg.cond(QS))
    print("L=%d ns=%d entries=%.2e [%.0fs]: min|X|=%.1e ; frac<1e-9: %.1e ; <1e-6: %.2e ; <1e-3: %.2e ; median cond %.0f max %.0f" % (L, ns, tot, time.time() - t0, mn, z9 / tot, z6 / tot, z3 / tot, np.median(conds), max(conds)))
