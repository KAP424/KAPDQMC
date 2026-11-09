'''Figure for the bounded perturbative-weight diagnostic (free chain, linear axes).

Reruns the sampling of ff_pert_share.py with the same seeds and keeps the numbers, so the figure
and the printed table cannot disagree.  Panels:
  (a) z = 1: the typical r = t*||X|| and P(r > 1) both FALL with n.
  (b) the share of the exact first-order weight E[Y] = n^2 + 4 V0 that sits at r > 1 RISES with n:
      the weight is carried by ever rarer near-singular outputs.
  (c) measured S2bar/z^2 over its small-z limit FALLS with n, in step with (b).
The 1D measured points have relative errors below about 1 percent, smaller than the markers.

Usage: D:/app/conda/python.exe ff_pert_share_figure.py [nsamp]
'''

import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import ff_pert_share as P

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')


def mean_sem(v):
    return float(v.mean()), float(v.std(ddof=1) / np.sqrt(v.size))


def main():
    nsamp = int(sys.argv[1]) if len(sys.argv) > 1 else 20000
    meas = P.measured()
    rows = []
    for L in P.SIZES:
        n, V0, xn, Y = P.samples(L, nsamp, 20261105 + L)
        ok = np.isfinite(Y)
        xn, Y = xn[ok], Y[ok]
        tot = n ** 2 + 4.0 * V0
        small = 1.0 + 4.0 * V0 / n ** 2
        for z in P.ZS:
            r = z / n * xn
            tail, tail_e = mean_sem(Y * (r > 1.0) / tot)
            m = meas.get((L, z))
            rows.append(dict(n=n, z=z, median_r=float(np.median(r)),
                             p_gt1=float((r > 1.0).mean()), tail1=tail, tail1_sem=tail_e,
                             measured=(m / small if m is not None else None)))
        print('n=%d done' % n, flush=True)
    with open(os.path.join(HERE, 'ff_pert_share.json'), 'w') as fh:
        json.dump(rows, fh, indent=1)

    cm = plt.get_cmap('viridis')
    zcol = {0.1: cm(0.1), 0.3: cm(0.5), 1.0: cm(0.85)}
    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.4))

    ax = axes[0]
    sel = [r for r in rows if r['z'] == 1.0]
    ns = [r['n'] for r in sel]
    ax.plot(ns, [r['median_r'] for r in sel], 'o-', color=cm(0.1), label='median of r')
    ax.plot(ns, [r['p_gt1'] for r in sel], 's--', color=cm(0.8), label='P(r > 1)')
    ax.set_ylabel('r = t*||X||,  at z = 1')
    ax.set_ylim(0, 0.8)
    ax.set_title('(a) the typical r falls with n', fontsize=10)

    ax = axes[1]
    for z in P.ZS:
        sel = [r for r in rows if r['z'] == z]
        ax.errorbar([r['n'] for r in sel], [r['tail1'] for r in sel],
                    yerr=[r['tail1_sem'] for r in sel], marker='o', ms=4, lw=1.0,
                    capsize=2, color=zcol[z], label='z = %g' % z)
    ax.set_ylabel('share of first-order weight at r > 1')
    ax.set_ylim(0, 1.0)
    ax.set_title('(b) the weight moves to rarer events', fontsize=10)

    ax = axes[2]
    for z in P.ZS:
        sel = [r for r in rows if r['z'] == z and r['measured'] is not None]
        ax.plot([r['n'] for r in sel], [r['measured'] for r in sel], 'o-', ms=4, lw=1.0,
                color=zcol[z], label='z = %g' % z)
    ax.set_ylabel('measured S2bar/z^2 over its small-z limit')
    ax.set_ylim(0, 1.05)
    ax.set_title('(c) the measured entropy falls in step', fontsize=10)

    for ax in axes:
        ax.set_xlabel('n (particles)')
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'ff_pert_share.png'), dpi=200)
    plt.close(fig)
    print('figure -> ' + os.path.join(FIG, 'ff_pert_share.png'))


if __name__ == '__main__':
    main()
