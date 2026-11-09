"""Referee (II)-c2: statistics of the exact-zero cluster (|X|<1e-9) vs L, with batch errors and per-sample multiplicity."""
import sys, time
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import annealed_gaussian as ag
rng = np.random.default_rng(99)
def dpp(Q):
    V = Q.copy(); M, n = V.shape; F = np.empty(n, int)
    for k in range(n):
        p = np.einsum("ij,ij->i", V, V); p = np.clip(p, 0, None); i = rng.choice(M, p=p / p.sum()); F[k] = i
        r = V[i].copy(); al = -np.copysign(np.linalg.norm(r), r[0]); r[0] -= al; r /= np.linalg.norm(r)
        V = (V - 2.0 * np.outer(V @ r, r))[:, 1:]
    return np.sort(F)
for L, ns, nb in ((6, 30000, 15), (9, 4000, 8), (12, 800, 8)):
    G0 = ag.ground_projector(L, mu=1e-8); ev, Vv = np.linalg.eigh(G0); Q = Vv[:, ev > 0.5]; M, n = Q.shape
    z9 = np.zeros(ns, int); z6 = np.zeros(ns, int); t0 = time.time()
    for s in range(ns):
        S = dpp(Q); E = np.setdiff1d(np.arange(M), S); X = np.linalg.solve(Q[S].T, Q[E].T).T; a = np.abs(X)
        z9[s] = int((a < 1e-9).sum()); z6[s] = int((a < 1e-6).sum())
    f9 = z9 / n ** 2; f6 = z6 / n ** 2; B = np.array_split(np.arange(ns), nb)
    bm = np.array([f9[b].mean() for b in B]); bm6 = np.array([f6[b].mean() for b in B])
    print("L=%2d ns=%d [%.0fs]: exact-zero fraction (<1e-9) = %.2e +- %.2e (batch) | (<1e-6) = %.2e +- %.2e | samples with >=1 zero: %.4f%% | max zeros in one sample %d, share of all zeros from top-1%% samples %.2f" % (
        L, ns, time.time() - t0, f9.mean(), bm.std(ddof=1) / np.sqrt(nb), f6.mean(), bm6.std(ddof=1) / np.sqrt(nb),
        100 * (z9 > 0).mean(), z9.max(), np.sort(z9)[::-1][: max(1, ns // 100)].sum() / max(z9.sum(), 1)))
