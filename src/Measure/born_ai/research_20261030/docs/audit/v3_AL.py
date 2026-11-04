import sys, time, json
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *

def honeycomb_T(L):
    """AB hopping block: A(x,y) -- B(x,y), B(x+1,y), B(x,y-1); returns T (L^2 x L^2), index = x*L+y"""
    n = L * L
    T = np.zeros((n, n))
    for x in range(L):
        for y in range(L):
            a = x * L + y
            for (xx, yy) in ((x, y), ((x + 1) % L, y), (x, (y - 1) % L)):
                T[a, xx * L + yy] += 1.0
    return T

def ground_G(L, mu=1e-8):
    """G0 = P_occ, interleaved ordering site = 2*(x*L+y)+sigma. H = [[mu, -T],[-T^T, -mu]]"""
    n = L * L
    T = honeycomb_T(L)
    H = np.zeros((2 * n, 2 * n))
    H[:n, :n] = mu * np.eye(n)
    H[n:, n:] = -mu * np.eye(n)
    H[:n, n:] = -T
    H[n:, :n] = -T.T
    ev, V = np.linalg.eigh(H)
    Q = V[:, :n]               # lowest n states filled
    G = Q @ Q.T
    # reorder to interleaved ordering: index (cell c, sigma) -> 2c+sigma
    perm = np.empty(2 * n, dtype=int)
    for c in range(n):
        perm[2 * c] = c
        perm[2 * c + 1] = n + c
    G = G[np.ix_(perm, perm)]
    return G, ev

def masks(L):
    w = L // 3
    x = np.arange(L)
    X, Y = np.meshgrid(x, x, indexing="ij")
    rep = lambda m: np.repeat(m.ravel(), 2)
    return dict(A=rep(X < w), B=rep((X < w) & (Y < 2 * w)), C=rep(Y < w))

def renyi2(Gsub):
    nu = np.clip(np.linalg.eigvalsh(0.5 * (Gsub + Gsub.T)), 0, 1)
    return float(-np.sum(np.log(nu**2 + (1 - nu) ** 2)))

def V0(G, m):
    return float(np.sum(G[np.ix_(m, ~m)] ** 2))

out = {}
base = baseline()
print("L   D_V0       A(L)=16/81+4DV0/L^4    D_gs(computed)  D_gs(baseline)  nodeG-check(trace)")
for L in (3, 6, 9, 12, 18, 24, 30, 36):
    t0 = time.time()
    G, ev = ground_G(L)
    m = masks(L)
    DV0 = (2 * V0(G, m["A"]) + V0(G, m["C"])) / 3 - V0(G, m["B"])
    A = 16 / 81 + 4 * DV0 / L**4
    if L <= 24 or L in (30, 36):
        S = {k: renyi2(G[np.ix_(m[k], m[k])]) for k in "ABC"}
        Dg = (2 * S["A"] + S["C"]) / 3 - S["B"]
    else:
        Dg = float("nan")
    bg = base.get(str(L), {}).get("D", float("nan"))
    out[L] = dict(DV0=DV0, A=A, Dgs=Dg)
    print("%2d  %.6f   %.6f (%.7f)      %.8f      %.8f   tr=%.6f  [%.0fs]" % (L, DV0, A, A, Dg, bg, np.trace(G), time.time() - t0), flush=True)
json.dump(out, open(AUD + "/v3_AL.json", "w"), indent=1)
print("16/81 = %.7f" % (16 / 81))
