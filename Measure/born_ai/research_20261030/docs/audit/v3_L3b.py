import sys, itertools, time
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
exec(open(AUD + "/v3_AL.py").read().split("out = {}")[0].split("from v3_common import *")[1])   # honeycomb_T, ground_G, masks, renyi2, V0

def ground_Q_svd(L, mu=1e-8):
    n = L * L
    T = honeycomb_T(L)
    U, s, Vt = np.linalg.svd(T)
    s = np.where(np.abs(s) < 1e-12, 0.0, s)
    eps = np.hypot(s, mu)
    cA = s / np.sqrt(2 * eps * (eps + mu))
    cB = np.sqrt((eps + mu) / (2 * eps))
    Q = np.zeros((2 * n, n))
    Qa = U * cA[None, :]            # A block (n x n)
    Qb = Vt.T * cB[None, :]
    Q[0::2, :] = Qa; Q[1::2, :] = Qb      # interleaved: site 2c -> A, 2c+1 -> B
    return Q

L = 3; n = 9; N = 18
Q = ground_Q_svd(L)
print("orthonormality err", np.abs(Q.T @ Q - np.eye(n)).max())
G0 = Q @ Q.T
m = masks(L); masks_f = {k: m[k].astype(float) for k in "ABC"}
acc_full = dict(A=0.0, B=0.0, C=0.0); acc_node = dict(A=0.0, B=0.0, C=0.0)
for S in itertools.combinations(range(N), n):
    S = np.array(S); E = np.setdiff1d(np.arange(N), S)
    U, s, Vt = np.linalg.svd(Q[S])
    if s[-1] > 1e-6:
        w = np.prod(s**2)
        X = Q[E] @ np.linalg.inv(Q[S]); W = X**2
        for k in "ABC":
            re = masks_f[k][E]; ce = masks_f[k][S]
            acc_full[k] += w * 2.0 * (re @ W @ (1 - ce) + (1 - re) @ W @ ce)
    elif s[-2] > 1e-6:
        v = Vt[-1]
        Phi = np.zeros((N, n)); Phi[S, : n - 1] = U[:, : n - 1]; Phi[E, n - 1] = Q[E] @ v
        Phi, _ = np.linalg.qr(Phi); Gn = Phi @ Phi.T
        WS = np.prod(s[: n - 1] ** 2)
        for k in "ABC":
            acc_node[k] += WS * renyi2(Gn[np.ix_(m[k], m[k])])
comb = lambda d: (2 * d["A"] + d["C"]) / 3 - d["B"]
tot = comb(acc_full) + comb(acc_node)
print("C_D full %.10f + node %.10f = %.10f ; claimed 14.6377513890 ; diff %.1e" % (comb(acc_full), comb(acc_node), tot, tot - 14.6377513890106))
print("A(3) = %.7f" % (tot / 81))
