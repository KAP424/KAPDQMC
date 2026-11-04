import sys, itertools, time
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
exec(open(AUD + "/v3_AL.py").read().split("out = {}")[0].split("from v3_common import *")[1])   # reuse honeycomb_T, ground_G, masks, renyi2, V0

L = 3
n = L * L; N = 2 * n
G0, ev = ground_G(L)
# orbitals for the occupied space (interleaved ordering): use eigvecs of G0 with eigenvalue ~1
w_, V_ = np.linalg.eigh(G0)
Q = V_[:, w_ > 0.5]
assert Q.shape == (N, n)
m = masks(L)
masks_f = {k: m[k].astype(float) for k in "ABC"}
acc_full = dict(A=0.0, B=0.0, C=0.0); acc_node = dict(A=0.0, B=0.0, C=0.0)
wfull = 0.0; wnode = 0.0; nnode = 0; nlow = 0; nfull = 0
t0 = time.time()
for S in itertools.combinations(range(N), n):
    S = np.array(S)
    E = np.setdiff1d(np.arange(N), S)
    U, s, Vt = np.linalg.svd(Q[S])
    if s[-1] > 1e-6:
        nfull += 1
        w = np.prod(s**2)
        X = Q[E] @ np.linalg.inv(Q[S])        # E x S
        W = X**2
        for k in "ABC":
            re = masks_f[k][E]; ce = masks_f[k][S]
            acc_full[k] += w * 2.0 * (re @ W @ (1 - ce) + (1 - re) @ W @ ce)
        wfull += w
    elif s[-2] > 1e-6:
        nnode += 1
        keep = s > 1e-6
        v = Vt[-1]                             # null vector of Q_S
        Phi = np.zeros((N, n))
        Phi[S, : n - 1] = U[:, : n - 1]
        Phi[E, n - 1] = Q[E] @ v
        Phi, _ = np.linalg.qr(Phi)
        Gn = Phi @ Phi.T
        WS = np.prod(s[: n - 1] ** 2)
        wnode += WS
        for k in "ABC":
            idx = m[k]
            acc_node[k] += WS * renyi2(Gn[np.ix_(idx, idx)])
    else:
        nlow += 1
comb = lambda d: (2 * d["A"] + d["C"]) / 3 - d["B"]
print("outputs: full-rank %d, rank n-1 (node) %d, rank<=n-2 %d ; total %d" % (nfull, nnode, nlow, nfull + nnode + nlow))
print("sum of full-rank weights det^2 = %.10f ; node weight sum W_S = %.6f ; node weight / n^2 = %.4f%%" % (wfull, wnode, 100 * wnode / n**2))
print("C_D full = %.10f, node = %.10f, total = %.10f" % (comb(acc_full), comb(acc_node), comb(acc_full) + comb(acc_node)))
print("A(3) = total/81 = %.6f   (bright formula 16/81+4DV0/81 = ?)" % ((comb(acc_full) + comb(acc_node)) / 81))
DV0 = (2 * V0(G0, m["A"]) + V0(G0, m["C"])) / 3 - V0(G0, m["B"])
print("bright formula C_D = 16 + 4 D_V0(3) = %.10f -> /81 = %.6f" % (16 + 4 * DV0, (16 + 4 * DV0) / 81))
print("elapsed %.0fs" % (time.time() - t0))
