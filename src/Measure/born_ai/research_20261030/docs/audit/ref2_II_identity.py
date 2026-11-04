"""Referee (II)-a: independent check of  sum_S [ (Q_E adj(Q_S))_{ec}^2 * 2*1_sep(e,c) ]  =  r(M-r)+4 V0(R).
Independent route: build every swapped determinant det(Q_{S-c+e}) directly (Cramer), no SVD / no inverse.
Also test the general formula 2nr - 4r<N_R> + 4V0 + 4<N_R>^2 for NON particle-hole-symmetric projectors."""
import itertools, numpy as np
rng = np.random.default_rng(12345)

def adj_form_total(Q, R):
    M, n = Q.shape; tot = 0.0; tot_dpp = 0.0
    Rf = R.astype(float)
    detS = {}
    for S in itertools.combinations(range(M), n):
        detS[S] = np.linalg.det(Q[list(S)])
    for S, d0 in detS.items():
        Sset = set(S); E = [i for i in range(M) if i not in Sset]
        for c in S:
            for e in E:
                Sp = tuple(sorted((Sset - {c}) | {e}))
                w = detS[Sp] ** 2             # = (Q_E adj(Q_S))_{ec}^2 by Cramer
                sep = (R[e] != R[c])
                tot += 2.0 * w * sep
    return tot, sum(d ** 2 for d in detS.values())

def check(M, ph=True, region_frac=0.4):
    n = M // 2
    if ph:
        U, _ = np.linalg.qr(rng.normal(size=(n, n)))
        Sm = np.zeros((M, M)); Sm[:n, n:] = U; Sm[n:, :n] = U.T
        G = (np.eye(M) + Sm) / 2
        # random site relabelling so regions are generic
        perm = rng.permutation(M); G = G[np.ix_(perm, perm)]
    else:
        A = rng.normal(size=(M, n)); Qr, _ = np.linalg.qr(A); G = Qr @ Qr.T
    ev, V = np.linalg.eigh(G); Q = V[:, ev > 0.5]
    R = rng.random(M) < region_frac
    r = int(R.sum()); NR = float(np.trace(G[np.ix_(R, R)]))
    V0 = float(np.sum(G[np.ix_(R, ~R)] ** 2))
    tot, wsum = adj_form_total(Q, R)
    closed_ph = r * (M - r) + 4 * V0
    closed_gen = 2 * n * r - 4 * r * NR + 4 * V0 + 4 * NR ** 2
    return tot, closed_ph, closed_gen, NR, r, wsum

worst_ph = 0.0
print("PH-symmetric random states  (M sites, n=M/2 particles), exact enumeration of all C(M,n) outputs")
for M in (8, 10, 12):
    for t in range(4):
        tot, cph, cgen, NR, r, wsum = check(M, True, region_frac=[0.25, 0.4, 0.5, 0.6][t])
        worst_ph = max(worst_ph, abs(tot - cph))
        print("  M=%2d r=%d <N_R>-r/2=%+.1e  sum_S=%.12f  r(M-r)+4V0=%.12f  diff=%+.1e  | general formula diff=%+.1e | sum det^2=%.12f" % (
            M, r, NR - r / 2, tot, cph, tot - cph, tot - cgen, wsum))
print("worst |diff| PH-symmetric: %.2e" % worst_ph)
print("\nNon-PH-symmetric random projectors (<N_R> != r/2): specific closed form r(M-r)+4V0 should FAIL, general formula should hold")
for M in (8, 10, 12):
    tot, cph, cgen, NR, r, wsum = check(M, False)
    print("  M=%2d r=%d <N_R>-r/2=%+.3f  sum_S=%.10f  r(M-r)+4V0=%.10f (diff %+.3f)  general=%.10f (diff %+.1e)" % (M, r, NR - r / 2, tot, cph, tot - cph, cgen, tot - cgen))
