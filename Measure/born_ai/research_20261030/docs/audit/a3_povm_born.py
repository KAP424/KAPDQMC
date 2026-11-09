import itertools, math
import numpy as np


def run(a, beta, seed):
    rng = np.random.default_rng(seed)
    N = 2 * a
    Q, _ = np.linalg.qr(rng.normal(size=(N, a)))
    t = math.exp(-2 * beta)
    q = t * t
    Z = sum(math.comb(a, j) ** 2 * q ** j for j in range(a + 1))
    Zm = sum(math.comb(a - 1, j) ** 2 * q ** j for j in range(a))
    subs = list(itertools.combinations(range(N), a))
    PF = {F: np.linalg.det(Q[list(F), :]) ** 2 for F in subs}
    G0 = Q @ Q.T
    tot = 0.0
    pdev = 0.0
    meanG = np.zeros((N, N))
    for E in subs:
        w = np.ones(N)
        w[list(E)] = q
        p = np.linalg.det(Q.T @ (w[:, None] * Q)) / Z
        pipe = sum(PF[F] * q ** len(set(E) & set(F)) for F in subs) / Z
        pdev = max(pdev, abs(p - pipe))
        tot += p
        Qt = np.sqrt(w)[:, None] * Q
        G = Qt @ np.linalg.inv(Qt.T @ Qt) @ Qt.T
        meanG += p * G
    fock = [sum(q ** len(set(E) & set(F)) for E in subs) for F in subs]
    kap = 1 - (1 - t) ** 2 * Zm / Z
    off = ~np.eye(N, dtype=bool)
    print('a=%d beta=%.2f t=%.4f q=%.5f Z_a=%.6f' % (a, beta, t, q, Z))
    print('  sum_F P(F)=%.12f  sum_E p(E)=%.12f' % (sum(PF.values()), tot))
    print('  POVM: Fock-diagonal sum_E q^|E&F| min/max = %.10f / %.10f (Z_a)' % (min(fock), max(fock)))
    print('  max|p_det(E) - p_pipeline(E)| = %.2e' % pdev)
    print('  mean channel: max|offdiag(meanG - kappa G0)| = %.2e  max|diag(meanG - G0)| = %.2e  kappa=%.6f'
          % (np.abs((meanG - kap * G0)[off]).max(), np.abs(np.diag(meanG - G0)).max(), kap))


for a, beta in ((3, 0.3), (3, 1.0), (4, 0.5), (4, 1.5)):
    run(a, beta, 7 + a)
