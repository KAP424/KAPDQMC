'''Bounded diagnostic of where the first-order expansion holds, free chain, up to n = 63.

xxz_tail_ff.py tried to follow rho_n = w4 / n^4 to large n by sampling and failed its own
checks: w4 is dominated by rare near-singular outputs (one sample carries up to 92% of the
sum), so the sample mean is neither stable nor reliable.  This script uses a bounded
quantity instead.  For S drawn from the projection DPP (weight det(Q_S)^2, the Born weight
at j = 0) it records r = t * ||X||_2 with X = Q_E Q_S^-1 and t = z / n, and reports the
probability that r exceeds a threshold.  Each indicator lies in [0, 1], so the error is an
ordinary binomial one and there is no heavy tail.

The first-order law S2 ~ z^2 needs r << 1, so P(r > 0.3) at fixed z is the weight that sits
outside the perturbative regime.  If it grows with n at fixed z, then z alone is not the
expansion parameter.  n = 9, z = 1 must reproduce the exact enumeration quoted in the 1D
report (0.990, 0.180, 0.009 for thresholds 0.3, 1, 3).

Usage: D:/app/conda/python.exe ff_nonpert.py [nsamp]
'''

import sys
import time

import numpy as np

import onedim_born as ob

SIZES = (5, 7, 9, 15, 21, 33, 45, 63)
ZS = (0.3, 1.0, 3.0)
THR = (0.3, 1.0, 3.0)
REF9 = {0.3: 0.990, 1.0: 0.180, 3.0: 0.009}


def opnorms(L, nsamp, seed):
    rng = np.random.default_rng(seed)
    Q, _ = ob.chain_state(L)
    N, n = Q.shape
    out = np.empty(nsamp)
    for k in range(nsamp):
        S = np.sort(ob.sample_dpp(rng, Q))
        keep = np.ones(N, dtype=bool)
        keep[S] = False
        E = np.flatnonzero(keep)
        try:
            X = Q[E, :] @ np.linalg.inv(Q[S, :])
            out[k] = np.linalg.norm(X, 2)
        except np.linalg.LinAlgError:
            out[k] = np.inf
    return n, out


def main():
    nsamp = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    print('r = t ||X||_2 with t = z / n; P(r > thr) +- binomial sem')
    print('%4s %5s %8s  %-15s %-15s %-15s' % (
        'n', 'z', 'median r', 'P(r>0.3)', 'P(r>1)', 'P(r>3)'))
    for L in SIZES:
        t0 = time.time()
        n, x = opnorms(L, nsamp, 20261104 + L)
        for z in ZS:
            r = z / n * x
            cells = []
            for th in THR:
                p = float((r > th).mean())
                cells.append('%.3f+-%.3f' % (p, np.sqrt(p * (1 - p) / r.size)))
            print('%4d %5.1f %8.3f  %-15s %-15s %-15s' % (
                n, z, float(np.median(r)), cells[0], cells[1], cells[2]), flush=True)
        if L == 9:
            print('     reference n=9, z=1 (exact enumeration): ' +
                  ', '.join('%.3f' % REF9[th] for th in THR))
        print('     [%d samples, %.0fs]' % (nsamp, time.time() - t0), flush=True)


if __name__ == '__main__':
    main()
