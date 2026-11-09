'''Free-fermion one-hop tail moment at large n.

xxz_tail.py gets rho_n = w4 / n^4 by exact enumeration, which stops at n = 7.  For the
free chain (Delta = 0) the Slater structure gives X = Q_E Q_S^-1 directly, and S is a
projection DPP, so w2 = E_S[sum X^2] = n^2 and w4 = E_S[sum X^4] can be estimated by
sampling up to n = 63.  This tests whether rho_n saturates (collapse on z possible at this
order) or keeps growing with n.

The estimator is heavy tailed, so the script also prints the median and the share of the
largest single sample in the total.  When that share is large the mean is not a reliable
number and the quoted SEM understates the real uncertainty.

Usage: D:/app/conda/python.exe xxz_tail_ff.py [nsamp]
'''

import sys
import time

import numpy as np

import onedim_born as ob

SIZES = (5, 7, 9, 15, 21, 33, 45, 63)
EXACT = {5: 0.38016, 7: 0.61513}   # xxz_tail.py, Delta = 0, exact enumeration


def moments(L, nsamp, seed):
    rng = np.random.default_rng(seed)
    Q, _ = ob.chain_state(L)
    N, n = Q.shape
    m2 = np.empty(nsamp)
    m4 = np.empty(nsamp)
    bad = 0
    for k in range(nsamp):
        S = np.sort(ob.sample_dpp(rng, Q))
        keep = np.ones(N, dtype=bool)
        keep[S] = False
        E = np.flatnonzero(keep)
        try:
            X = Q[E, :] @ np.linalg.inv(Q[S, :])
        except np.linalg.LinAlgError:
            bad += 1
            m2[k] = m4[k] = np.nan
            continue
        X2 = X * X
        m2[k] = X2.sum()
        m4[k] = (X2 * X2).sum()
    return n, m2, m4, bad


def main():
    nsamp = int(sys.argv[1]) if len(sys.argv) > 1 else 8000
    print('%4s %7s %17s %19s %9s %10s %9s' % (
        'n', 'nsamp', 'w2/n^2 +- sem', 'rho=w4/n^4 +- sem', 'median', 'max share', 'exact'))
    for L in SIZES:
        t0 = time.time()
        n, m2, m4, bad = moments(L, nsamp, 20261103 + L)
        ok = np.isfinite(m4)
        a2 = m2[ok] / n ** 2
        a4 = m4[ok] / n ** 4
        sem2 = a2.std(ddof=1) / np.sqrt(a2.size)
        sem4 = a4.std(ddof=1) / np.sqrt(a4.size)
        share = a4.max() / a4.sum()
        ex = '%.5f' % EXACT[L] if L in EXACT else '-'
        print('%4d %7d %8.4f+-%.4f %9.4f+-%.4f %9.4f %10.3f %9s  bad=%d [%.0fs]' % (
            n, a4.size, a2.mean(), sem2, a4.mean(), sem4, np.median(a4), share, ex,
            bad, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
