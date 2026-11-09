"""Referee (II)-c: is 'fraction of |X|<1e-6 in DPP samples' a valid upper bound on exact-zero (node) weight?
Look for a separated cluster of exact zeros in log10|X| (X = Q_E Q_S^{-1}).  Node weight = E[#exact zeros in X]/n^2 exactly
(each exact zero is a rank-deficient swap-neighbour of the DPP sample)."""
import sys, time
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030")
import numpy as np
import annealed_gaussian as ag
rng = np.random.default_rng(2026)

def dpp(Q):
    V = Q.copy(); M, n = V.shape; F = np.empty(n, int)
    for k in range(n):
        p = np.einsum("ij,ij->i", V, V); p = np.clip(p, 0, None); i = rng.choice(M, p=p / p.sum()); F[k] = i
        r = V[i].copy(); al = -np.copysign(np.linalg.norm(r), r[0]); r[0] -= al; r /= np.linalg.norm(r)
        V = (V - 2.0 * np.outer(V @ r, r))[:, 1:]
    return np.sort(F)

def run(L, ns):
    G0 = ag.ground_projector(L, mu=1e-8); ev, Vv = np.linalg.eigh(G0); Q = Vv[:, ev > 0.5]; M, n = Q.shape
    edges = np.arange(-20, 5.01, 1.0); cnt = np.zeros(len(edges) - 1); tot = 0; conds = []
    zeros_rel = 0; t0 = time.time()
    for s in range(ns):
        S = dpp(Q); E = np.setdiff1d(np.arange(M), S)
        QS = Q[S]; X = np.linalg.solve(QS.T, Q[E].T).T
        conds.append(np.linalg.cond(QS))
        a = np.abs(X).ravel(); tot += a.size
        cnt += np.histogram(np.log10(a + 1e-300), bins=edges)[0]
    return edges, cnt, tot, np.array(conds), time.time() - t0

for L, ns in ((3, 4000), (6, 1500), (9, 400)):
    edges, cnt, tot, cond, sec = run(L, ns)
    n = L * L
    print("\nL=%d  samples=%d entries=%d  (n^2=%d)  median cond(Q_S)=%.1e max=%.1e  [%.0fs]" % (L, ns, tot, n * n, np.median(cond), cond.max(), sec))
    print("  decade of |X|   fraction    ratio to next-higher decade (flat-density continuum would give ~0.1)")
    for k in range(len(cnt)):
        lo = edges[k]
        if lo > 1.5: break
        if cnt[k] == 0 and lo < -17: continue
        ratio = cnt[k] / cnt[k + 1] if k + 1 < len(cnt) and cnt[k + 1] > 0 else float("nan")
        print("   1e%+03d..1e%+03d  %10.3e   %8.3f" % (lo, lo + 1, cnt[k] / tot, ratio))
    low = cnt[edges[:-1] < -6].sum() / tot; print("  fraction |X|<1e-6: %.2e ; <1e-9: %.2e ; <1e-12: %.2e" % (low, cnt[edges[:-1] < -9].sum() / tot, cnt[edges[:-1] < -12].sum() / tot))
