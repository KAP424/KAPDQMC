import sys, itertools
import numpy as np
sys.path.insert(0, 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030')
import annealed_gaussian as ag


def orbitals(L):
    G = ag.ground_projector(L, mu=1e-8)
    ev, V = np.linalg.eigh(G)
    n = L * L
    print('L', L, 'eig gap', ev[-n - 1], ev[-n], 'diagA,B', G[0, 0], G[1, 1], '1/2-1/L^2', 0.5 - 1.0 / n)
    return G, V[:, -n:]


def pair_sum(Q, subs, e, c):
    M, n = Q.shape
    hc = (subs == c).any(axis=1)
    he = (subs == e).any(axis=1)
    sel = subs[hc & ~he].astype(int)
    N = len(sel)
    QF = Q[sel]
    det = np.linalg.det(QF)
    pos = (sel == c).argmax(axis=1)
    QR = QF.copy()
    QR[np.arange(N), pos, :] = Q[e][None, :]
    adj = np.linalg.det(QR)
    s_adj = float(np.sum(adj ** 2))
    ok = np.abs(det) > 1e-8
    k = int(ok.sum())
    b = np.broadcast_to(Q[e], (k, n))[..., None]
    x = np.linalg.solve(np.transpose(QF[ok], (0, 2, 1)), b)[..., 0]
    X = x[np.arange(k), pos[ok]]
    s_dir = float(np.sum(det[ok] ** 2 * X ** 2)) + float(np.sum(adj[~ok] ** 2))
    return s_adj, s_dir, int((~ok).sum()), N


def exhaustive_L3():
    G, Q = orbitals(3)
    M, n = Q.shape
    subs = np.array(list(itertools.combinations(range(M), n)), dtype=np.int16)
    print('subsets', len(subs))
    pairs = [(0, 1), (1, 0), (0, 6), (1, 7), (0, 17), (4, 9), (2, 3), (3, 12), (5, 16), (8, 15)]
    print('e c sub(e) sub(c) | sum_adj sum_direct n_singular N | Wick Gee(1-Gcc)+Gec^2 | 1/4+Gec^2 | rel.err of 1/4 form')
    for e, c in pairs:
        s_adj, s_dir, ns, N = pair_sum(Q, subs, e, c)
        w = G[e, e] * (1 - G[c, c]) + G[e, c] ** 2
        q4 = 0.25 + G[e, c] ** 2
        print(e, c, 'AB'[e % 2], 'AB'[c % 2], f'| {s_adj:.10f} {s_dir:.10f} {ns} {N} | {w:.10f} | {q4:.10f} | {(q4 - s_adj) / s_adj:+.4f}')


def aggregate(L):
    G = ag.ground_projector(L, mu=1e-8)
    M = G.shape[0]
    d = np.diag(G)
    m = ag.region_masks(L)
    P = d[:, None] * (1.0 - d)[None, :] + G ** 2
    for key in 'ABC':
        R = m[key]
        r = int(R.sum())
        cross = R[:, None] ^ R[None, :]
        exact = P[cross].sum()
        quarter = 0.25 * cross.sum() + (G[cross] ** 2).sum()
        V0 = d[R].sum() - np.sum(G[np.ix_(R, R)] ** 2)
        print(f'L={L} R={key} r={r} exact={exact:.8f} quarter={quarter:.8f} diff={exact-quarter:+.2e} r(M-r)/2+2V0={r*(M-r)/2+2*V0:.8f} <N_R>-r/2={d[R].sum()-r/2:+.1e}')


def corner_terms():
    print('L, D_V0, A(L)=16/81+4 D_V0/L^4, corner/volume')
    xs, ys = [], []
    for L in (3, 6, 9, 12, 18, 24, 30, 36):
        G = ag.ground_projector(L, mu=1e-8)
        m = ag.region_masks(L)
        dg = np.diag(G)
        v = {}
        for key in 'ABC':
            R = m[key]
            v[key] = dg[R].sum() - np.sum(G[np.ix_(R, R)] ** 2)
        DV = (2 * v['A'] + v['C']) / 3 - v['B']
        A = 16.0 / 81 + 4 * DV / L ** 4
        print(L, f'DV0={DV:.6f} A={A:.6f} corner/volume={4*DV/L**4/(16.0/81):.3e} lnL/L^4={np.log(L)/L**4:.3e}')
        if L >= 12:
            xs.append(np.log(L))
            ys.append(DV)
    print('D_V0 vs ln L (L>=12) slope, intercept:', np.polyfit(xs, ys, 1))


if __name__ == '__main__':
    exhaustive_L3()
    for L in (3, 6, 9):
        aggregate(L)
    print('pairwise error of the 1/4 form: Gee(1-Gcc)-1/4 = -d+d^2 (e in A, c in B), +d+d^2 (e in B, c in A), -d^2 (same sublattice), d=1/L^2')
    for L in (3, 6, 12, 36):
        d = 1.0 / L ** 2
        print(L, f'{-d + d*d:+.5f} {d + d*d:+.5f} {-d*d:+.2e}', 'rel to 1/4:', f'{4*(-d+d*d):+.4f} {4*(d+d*d):+.4f}')
    corner_terms()
