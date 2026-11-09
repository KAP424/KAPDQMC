import itertools
import numpy as np
rng = np.random.default_rng(99)
worst = 0.0
worst_nonph = 0.0
for n in (4, 5, 6):
    N = 2 * n
    for trial in range(3):
        U, _ = np.linalg.qr(rng.normal(size=(n, n)))
        Sm = np.zeros((N, N)); Sm[:n, n:] = U; Sm[n:, :n] = U.T
        G = (np.eye(N) + Sm) / 2                 # particle-hole symmetric projector: G_ii = 1/2
        w, V = np.linalg.eigh(G)
        Q = V[:, w > 0.5]
        R = rng.random(N) < 0.4
        r = int(R.sum())
        closed = r * (N - r) + 4 * np.sum(G[np.ix_(R, ~R)] ** 2)
        tot = 0.0
        for S in itertools.combinations(range(N), n):
            S = np.array(S); E = np.setdiff1d(np.arange(N), S)
            QS = Q[S]
            det = np.linalg.det(QS)
            X = Q[E] @ np.linalg.inv(QS)
            W = X ** 2
            re = R[E].astype(float); ce = R[S].astype(float)
            tot += det ** 2 * 2 * (re @ W @ (1 - ce) + (1 - re) @ W @ ce)
        worst = max(worst, abs(tot - closed))
        # control: a generic (non-PH-symmetric) projector should violate the identity
        A_ = rng.normal(size=(N, n)); Qg, _ = np.linalg.qr(A_)
        Gg = Qg @ Qg.T
        closed_g = r * (N - r) + 4 * np.sum(Gg[np.ix_(R, ~R)] ** 2)
        tot_g = 0.0
        for S in itertools.combinations(range(N), n):
            S = np.array(S); E = np.setdiff1d(np.arange(N), S)
            QS = Qg[S]; det = np.linalg.det(QS); X = Qg[E] @ np.linalg.inv(QS); W = X ** 2
            re = R[E].astype(float); ce = R[S].astype(float)
            tot_g += det ** 2 * 2 * (re @ W @ (1 - ce) + (1 - re) @ W @ ce)
        worst_nonph = max(worst_nonph, abs(tot_g - closed_g))
print("independent identity check on random PH-symmetric states (N=8,10,12): max |sum - closed| = %.2e" % worst)
print("control (generic projector, not PH symmetric): max deviation %.2e" % worst_nonph)
