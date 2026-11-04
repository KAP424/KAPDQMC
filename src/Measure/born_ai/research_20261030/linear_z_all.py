'''All measured points versus z on linear axes, plus the L-spread at each matched z.

(a) every 2D point (z up to 477), (b) the z<=12 part of the same set, (c) 2D: each L divided by the
inverse-variance mean over L at the same z, (d) 1D: the same ratio for the matched-z scan.
Panels (c),(d) read the quality of the collapse directly: perfect collapse is every point on 1.

Usage: D:/app/conda/python.exe linear_z_all.py
'''

import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
ONED = os.path.join(HERE, '..', 'research_20261101_1d')
FIG = os.path.join(HERE, 'figures')
sys.path.insert(0, HERE)
import headline_numbers as H


def ratio_panel(ax, groups, col, order, xmax, title):
    seen = set()
    n = 0
    for z, sel in groups:
        if z > xmax or len(sel) < 2:
            continue
        m = np.array([s[1] for s in sel])
        e = np.array([s[2] for s in sel])
        w = 1.0 / e ** 2
        mu = (w * m).sum() / w.sum()
        for L, mean, sem in sel:
            lab = 'L=%d' % L if L not in seen else None
            seen.add(L)
            ax.errorbar(z + (order.index(L) - (len(order) - 1) / 2) * 0.012 * xmax, mean / mu,
                        yerr=sem / mu, marker='o', ms=4, lw=0, elinewidth=1.0, capsize=1.5,
                        color=col[L], label=lab)
            n += 1
    ax.axhline(1.0, color='k', ls='--', lw=0.9)
    ax.set_xlim(0, xmax * 1.03)
    ax.set_xlabel('z = n exp(-2 beta)')
    ax.set_ylabel('value / (mean over L at the same z)')
    ax.set_title(title + ' (%d points)' % n, fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')


def main():
    pts = H.load_pooled()
    Ls = sorted({p['L'] for p in pts})
    cm = plt.get_cmap('viridis')
    col = {L: cm(0.05 + 0.85 * i / (len(Ls) - 1)) for i, L in enumerate(Ls)}
    fig, axes = plt.subplots(2, 2, figsize=(11.8, 8.8))

    for ax, xmax, name in ((axes[0][0], 500.0, '(a) 2D, all'), (axes[0][1], 12.0, '(b) 2D, z<=12')):
        k = 0
        for L in Ls:
            s = sorted([p for p in pts if p['L'] == L and p['z'] <= xmax], key=lambda p: p['z'])
            if not s:
                continue
            k += len(s)
            ax.errorbar([p['z'] for p in s], [p['mean'] for p in s], yerr=[p['sem'] for p in s],
                        marker='o', ms=4, lw=0.7, capsize=1.5, color=col[L], label='L=%d' % L)
        ax.set_xlim(0, xmax)
        ax.set_ylim(bottom=0)
        ax.set_xlabel('z = n exp(-2 beta)')
        ax.set_ylabel('D2bar (2D corner combination)')
        ax.set_title('%s: %d of %d points drawn' % (name, k, len(pts)), fontsize=10)
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7, frameon=False, ncol=2, loc='lower right')

    g2 = [(z, [(p['L'], p['mean'], p['sem']) for p in sel])
          for z, sel in H.group_by_z(pts).items()]
    ratio_panel(axes[1][0], g2, col, Ls, 11.0, '(c) 2D collapse quality, matched z')

    with open(os.path.join(ONED, 'data_matched_big.json')) as fh:
        rows = [r for r in json.load(fh) if 'S2_half' in r]
    by = defaultdict(list)
    for r in rows:
        by[round(r['z'], 3)].append((r['L'], r['S2_half'], r['S2_half_sem']))
    L1 = sorted({r['L'] for r in rows})
    col1 = {L: cm(0.05 + 0.85 * i / (len(L1) - 1)) for i, L in enumerate(L1)}
    g1 = sorted(by.items())
    ratio_panel(axes[1][1], g1, col1, L1, 11.0, '(d) 1D collapse quality, matched z')
    axes[1][0].set_ylim(0.8, 1.2)
    axes[1][1].set_ylim(0.8, 1.2)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'linear_z_all.png'), dpi=200)
    plt.close(fig)
    print('2D points total', len(pts), ' z>12:', sum(p['z'] > 12 for p in pts))
    print('figure ->', os.path.join(FIG, 'linear_z_all.png'))


if __name__ == '__main__':
    main()
