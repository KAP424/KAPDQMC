'''Share of the first-order weight that lies in the perturbative region (free chain).

ff_nonpert.py found that the bulk of r = t ||X||_2 does not grow with n at fixed z (median at
z = 1: 0.62 for n = 5, 0.43 for n = 63).  That contradicts the explanation in RESULTS_1D.md,
which says t ||X|| grows slowly with L.  Yet S2bar / z^2 at fixed z does fall with L, so
something else carries the drift.  This script tests the bounded form of the heavy-tail idea.

Exact first-order identity (small-z theorem, half ring): with S drawn from the projection DPP,
    E_S[ Y ] = n^2 + 4 V0,   Y = 2 * sum over (e, c) separated by the cut of X_ec^2,
X = Q_E Q_S^-1.  The first-order law S2 ~ t^2 Y fails where r = t ||X||_2 is not small, because
S2 saturates there.  Hence
    share(theta) = E_S[ Y * 1(r <= theta) ] / (n^2 + 4 V0)
is the fraction of the first-order weight that sits in the perturbative region.  Unlike w4 it is
bounded sample by sample (Y <= 2 n ||X||_2^2 <= 2 n (theta n / z)^2), so its mean is reliable.
If the drift of S2bar / z^2 with n came from the heavy tail, share would fall with n at fixed z.

Check: n = 9, z = 1 against the exact enumeration in RESULTS_1D.md (mechanism.py), where the
layers r > 0.3, 1, 3 carry 1.019, 0.725, 0.252 of the total 1.02015, i.e. share = 0.001, 0.289, 0.753.

Usage: D:/app/conda/python.exe ff_pert_share.py [nsamp]
'''

import json
import os
import sys
import time

import numpy as np

import onedim_born as ob

HERE = os.path.dirname(os.path.abspath(__file__))
SIZES = (9, 15, 21, 27, 33, 45, 63)
ZS = (0.1, 0.3, 1.0)
THR = (0.3, 1.0, 3.0)
REF9 = {0.3: 0.001, 1.0: 0.289, 3.0: 0.753}


def measured():
    with open(os.path.join(HERE, 'data_matched_big.json')) as fh:
        rows = [r for r in json.load(fh) if 'S2_half' in r]
    out = {}
    for r in rows:
        for z in ZS:
            if abs(r['z'] - z) < 1e-3:
                out[(r['L'], z)] = r['S2_half'] / z ** 2
    return out


def samples(L, nsamp, seed):
    rng = np.random.default_rng(seed)
    Q, _ = ob.chain_state(L)
    N, n = Q.shape
    region = ob.half_ring_region(N)
    inR = np.zeros(N, dtype=bool)
    inR[region] = True
    V0 = ob.ground_state_number_variance(Q, region)
    xn = np.empty(nsamp)
    Y = np.empty(nsamp)
    for k in range(nsamp):
        S = np.sort(ob.sample_dpp(rng, Q))
        keep = np.ones(N, dtype=bool)
        keep[S] = False
        E = np.flatnonzero(keep)
        try:
            X = Q[E, :] @ np.linalg.inv(Q[S, :])
        except np.linalg.LinAlgError:
            xn[k] = np.inf
            Y[k] = np.inf
            continue
        sel = inR[E][:, None] != inR[S][None, :]
        xn[k] = np.linalg.norm(X, 2)
        Y[k] = 2.0 * np.sum(X[sel] ** 2)
    return n, V0, xn, Y


def main():
    nsamp = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    meas = measured()
    print('share(theta) = E[Y 1(t||X||_2 <= theta)] / (n^2 + 4 V0), t = z / n; +- sem')
    print('last column: measured S2bar/z^2 over its small-z limit 1 + 4 V0 / n^2 (data_matched_big.json)')
    print('%4s %5s  %-14s %-14s %-14s  %-9s' % ('n', 'z', 'share(0.3)', 'share(1)', 'share(3)', 'measured'))
    for L in SIZES:
        t0 = time.time()
        n, V0, xn, Y = samples(L, nsamp, 20261105 + L)
        ok = np.isfinite(Y)
        xn, Y = xn[ok], Y[ok]
        tot = n ** 2 + 4.0 * V0
        small = 1.0 + 4.0 * V0 / n ** 2
        for z in ZS:
            r = z / n * xn
            cells = []
            for th in THR:
                v = Y * (r <= th) / tot
                cells.append('%.3f+-%.3f' % (v.mean(), v.std(ddof=1) / np.sqrt(v.size)))
            m = meas.get((L, z))
            mtxt = '%.3f' % (m / small) if m is not None else '-'
            print('%4d %5.1f  %-14s %-14s %-14s  %-9s' % (n, z, cells[0], cells[1], cells[2], mtxt), flush=True)
        if L == 9:
            print('     exact enumeration n=9, z=1: ' + ', '.join('%.3f' % REF9[th] for th in THR))
        print('     [%d samples, %.0fs, V0=%.5f]' % (ok.sum(), time.time() - t0, V0), flush=True)


if __name__ == '__main__':
    main()
