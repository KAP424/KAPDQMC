"""Referee (II)-a2: check (Q_E adj(Q_S))_{ec}^2 == det(Q_{S-c+e})^2 with adj built from explicit cofactors, incl. a singular Q_S."""
import itertools, numpy as np
rng = np.random.default_rng(7)
def adj(A):
    n = A.shape[0]; C = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            C[i, j] = (-1) ** (i + j) * np.linalg.det(np.delete(np.delete(A, i, 0), j, 1))
    return C.T
M, n = 10, 5
A = rng.normal(size=(M, n)); A[1] = A[0]                 # sites 0 and 1 identical -> any S containing both is singular
Q, _ = np.linalg.qr(A)                                   # rows 0,1 stay equal up to the orthogonal transform: Q = A R^{-1}
worst = 0.0; nsing = 0; nchk = 0
for S in itertools.combinations(range(M), n):
    S = list(S); E = [i for i in range(M) if i not in S]
    QS = Q[S]; dS = np.linalg.det(QS); sing = abs(dS) < 1e-12
    nsing += sing; B = Q[E] @ adj(QS)
    for a, e in enumerate(E):
        for k, c in enumerate(S):
            T = QS.copy(); T[k] = Q[e]; worst = max(worst, abs(B[a, k] ** 2 - np.linalg.det(T) ** 2)); nchk += 1
print("checked %d (S,e,c) triples, %d singular outputs S; max | (Q_E adj Q_S)_{ec}^2 - det(Q_{S-c+e})^2 | = %.2e" % (nchk, nsing, worst))
