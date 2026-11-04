'''Two conditions for the collapse of the corner entropy on z, drawn on linear axes.

Condition 1 (exact): the record-error count j has P_n(j) = C(n,j)^2 q^j / Z_n(q), which at
fixed z = n t tends to z^(2j) / [(j!)^2 I0(2z)] with O(1/n) corrections.  This holds for any
n-particle initial state, so it says nothing about free fermions in particular.

Condition 2 (the one that can fail): the entropy given j, D2bar | j, must also depend on z
only.  The curves below show it does so approximately for j = 0 and j = 1 in 2D, with a drift
in L that accounts for the roughly -4% per e-fold in L found for the unconditioned mean.

Every axis is linear.  Strata with fewer than MIN_N samples are not drawn.

Usage: D:/app/conda/python.exe pj_collapse_figure.py
'''

import glob
import math
import os
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from crossover_structure import read_csv

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
ZMAX = 3.4          # matched-z rows only; larger z lives on beta-anchored rows
MIN_N = 20


def p_inf(z, j):
    '''n -> infinity limit of P(j) at fixed z.'''
    i0 = sum(z ** (2 * k) / math.factorial(k) ** 2 for k in range(80))
    return z ** (2 * j) / math.factorial(j) ** 2 / i0


def load():
    pool = defaultdict(list)
    for f in glob.glob(os.path.join(HERE, 'data_matched_z', '*.csv')):
        a = read_csv(f)
        if a is None:
            continue
        pool[(int(a['L'][0]), round(float(a['beta'][0]), 4))].append(a)
    out = defaultdict(list)
    for (L, beta), arrs in pool.items():
        z = float(arrs[0]['z'][0])
        if z > ZMAX:
            continue
        j = np.concatenate([np.asarray(x['j']) for x in arrs]).astype(int)
        D = np.concatenate([np.asarray(x['D']) for x in arrs])
        out[L].append((z, j, D))
    return {L: sorted(v, key=lambda r: r[0]) for L, v in sorted(out.items())}


def main():
    data = load()
    Ls = sorted(data)
    cm = plt.get_cmap('viridis')
    col = {L: cm(0.05 + 0.85 * i / max(len(Ls) - 1, 1)) for i, L in enumerate(Ls)}
    zz = np.linspace(0.0, ZMAX, 300)
    fig, axes = plt.subplots(2, 2, figsize=(11.0, 8.2))

    for c, jj in enumerate((0, 1)):
        ax = axes[0][c]
        ax.plot(zz, [p_inf(z, jj) for z in zz], 'k-', lw=1.2,
                label=r'$n\to\infty$:  $z^{2j}/[(j!)^2 I_0(2z)]$')
        for L in Ls:
            z = np.array([r[0] for r in data[L]])
            p = np.array([(r[1] == jj).mean() for r in data[L]])
            n = np.array([r[1].size for r in data[L]])
            ax.errorbar(z, p, yerr=np.sqrt(p * (1 - p) / n), marker='o', ms=4, lw=0,
                        elinewidth=1.0, capsize=1.5, color=col[L], label='L=%d' % L)
        ax.set_xlim(0, ZMAX)
        ax.set_ylim(0, 1.02 if jj == 0 else 0.55)
        ax.set_xlabel(r'$z=L^2e^{-2\beta}$')
        ax.set_ylabel(r'$P(j=%d)$' % jj)
        ax.set_title('(%s) record-error count, j=%d: collapses on z' % ('ab'[c], jj), fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, frameon=False, ncol=2)

    for c, jj in enumerate((0, 1)):
        ax = axes[1][c]
        for L in Ls:
            zs, m, e = [], [], []
            for z, j, D in data[L]:
                sel = j == jj
                if sel.sum() < MIN_N:
                    continue
                zs.append(z)
                m.append(D[sel].mean())
                e.append(D[sel].std(ddof=1) / np.sqrt(sel.sum()))
            if zs:
                ax.errorbar(zs, m, yerr=e, marker='o', ms=4, lw=1.0, capsize=1.5,
                            color=col[L], label='L=%d' % L)
        ax.set_xlim(0, ZMAX)
        ax.set_ylim(bottom=0)
        ax.set_xlabel(r'$z=L^2e^{-2\beta}$')
        ax.set_ylabel(r'$\overline{D_2}\,|\,j=%d$' % jj)
        ax.set_title('(%s) entropy given j=%d: L-drift survives' % ('cd'[c], jj), fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='upper left')

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'pj_collapse.png'), dpi=200)
    plt.close(fig)
    print('figure -> ' + os.path.join(FIG, 'pj_collapse.png'))


if __name__ == '__main__':
    main()
