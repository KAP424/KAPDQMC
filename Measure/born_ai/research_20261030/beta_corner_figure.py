'''Corner contribution versus beta, with beta as the legend.

2D: D2bar, the corner combination.  1D: S2bar of the half ring, whose ln L term is the
1D counterpart of the corner log (it also carries the area term, so its absolute value is
not comparable to the 2D panel; compare shapes and slopes, not heights).

Panels (a),(c): the measured points of one L connected in beta order.  Panels (b),(d): the
same data re-cut at fixed beta, so each legend entry is one beta -- these are the panels
where beta is the legend.

Usage: D:/app/conda/python.exe beta_corner_figure.py
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

# beta rows that exist over the whole L range, so the fixed-beta curves are curves
B2 = (0.5, 1.0, 1.5, 2.0)
B1 = (0.25, 0.5, 1.0, 1.5, 2.0, 3.0)


def load2d():
    pts = H.load_pooled()
    return pts


def load1d():
    with open(os.path.join(ONED, 'data_fixed_big.json')) as fh:
        return [r for r in json.load(fh) if 'S2_half' in r]


def main():
    p2 = load2d()
    d1 = load1d()
    cm = plt.get_cmap('viridis')
    fig, axes = plt.subplots(2, 2, figsize=(11.6, 8.6))

    # ---------------- (a) 2D: D2bar versus beta, one curve per L
    ax = axes[0][0]
    byL = defaultdict(list)
    for p in p2:
        byL[p['L']].append((p['beta'], p['mean'], p['sem'], p['z']))
    Ls = sorted(byL)
    for i, L in enumerate(Ls):
        a = np.array(sorted(byL[L]))
        ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], marker='o', ms=4, lw=1.1,
                    capsize=1.5, color=cm(0.05 + 0.85 * i / max(len(Ls) - 1, 1)),
                    label='L=%d' % L)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\overline{D_2}$  (2D corner combination)')
    ax.set_title(r'(a) 2D: angle contribution versus $\beta$', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    # ---------------- (b) 2D: beta as the legend, x = L
    ax = axes[0][1]
    for i, b in enumerate(B2):
        sel = sorted([p for p in p2 if abs(p['beta'] - b) < 2e-3], key=lambda p: p['L'])
        if not sel:
            continue
        ax.errorbar([p['L'] for p in sel], [p['mean'] for p in sel],
                    yerr=[p['sem'] for p in sel], marker='o', ms=5, lw=1.3, capsize=2,
                    color=cm(0.08 + 0.25 * i), label=r'$\beta=%g$' % b)
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\overline{D_2}$')
    ax.set_title(r'(b) 2D: $\beta$ as the legend', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    # ---------------- (c) 1D: S2bar versus beta, one curve per L
    ax = axes[1][0]
    byL1 = defaultdict(list)
    for r in d1:
        byL1[r['L']].append((r['beta'], r['S2_half'], r['S2_half_sem']))
    Ls1 = sorted(byL1)
    for i, L in enumerate(Ls1):
        a = np.array(sorted(byL1[L]))
        ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], marker='o', ms=4, lw=1.1,
                    capsize=1.5, color=cm(0.05 + 0.85 * i / max(len(Ls1) - 1, 1)),
                    label='L=%d' % L)
    gs = sorted({(r['L'], r['S2_half_gs']) for r in d1})
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$\bar S_2$  (1D half ring)')
    ax.set_title(r'(c) 1D: entropy versus $\beta$', fontsize=10)
    ax.set_xlim(0.18, 3.15)
    ax.grid(alpha=0.25)
    # The ground state is a different value for every L, so it cannot be one line
    # here; panel (d) plots it against L instead.  A row of ticks plus a caption at
    # the left edge collided with the small-beta ends of the curves.
    ax.legend(fontsize=7.5, frameon=False, ncol=2, loc='upper right')

    # ---------------- (d) 1D: beta as the legend, x = L
    ax = axes[1][1]
    for i, b in enumerate(B1):
        sel = sorted([r for r in d1 if abs(r['beta'] - b) < 2e-3], key=lambda r: r['L'])
        if not sel:
            continue
        ax.errorbar([r['L'] for r in sel], [r['S2_half'] for r in sel],
                    yerr=[r['S2_half_sem'] for r in sel], marker='o', ms=4.5, lw=1.2,
                    capsize=1.5, color=cm(0.05 + 0.85 * i / (len(B1) - 1)),
                    label=r'$\beta=%g$' % b)
    ax.plot([g[0] for g in gs], [g[1] for g in gs], 'k-', lw=1.7, label='ground state')
    ax.set_xscale('log')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\bar S_2$')
    ax.set_title(r'(d) 1D: $\beta$ as the legend', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=7.5, frameon=False, ncol=2)

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'beta_corner.png'), dpi=200)
    plt.close(fig)
    print('figure ->', os.path.join(FIG, 'beta_corner.png'))


if __name__ == '__main__':
    main()
