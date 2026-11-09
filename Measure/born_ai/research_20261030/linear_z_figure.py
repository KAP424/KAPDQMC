'''Corner contribution versus z on linear axes, every measured point, plus the collapse quality.

(a) 2D: all 48 points, z = L^2 exp(-2 beta) from 0.089 to 477.   Nothing is cut here.
(b) 2D zoom, z <= 12, i.e. the window where the curves are claimed to overlap.
(c) 1D: all points of the matched-z and fixed-beta scans, z = L exp(-2 beta).
(d) 1D zoom, z <= 3.2.
(e) 2D collapse quality: each point divided by the inverse-variance mean over L at the same z.
(f) 1D collapse quality, same construction.  Perfect collapse is every point on 1; the vertical
    spread in (e),(f) IS the deviation from collapse, so these panels are where to judge it.

Panels with 'all' draw everything; there is no x-range filter except in the two zoom panels, whose
titles state how many points they show of how many available.

Usage: D:/app/conda/python.exe linear_z_figure.py
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

CM = plt.get_cmap('viridis')


def series_2d():
    out = defaultdict(list)
    for p in H.load_pooled():
        out[p['L']].append((p['z'], p['mean'], p['sem']))
    return {L: np.array(sorted(v)) for L, v in sorted(out.items())}


def series_1d():
    # matched-z has 5-7 z values per L; the fixed-beta scan adds the intermediate
    # z values, so the curve is not a few straight segments.  Same quantity, same L.
    rows = []
    for name in ('data_matched_big.json', 'data_fixed_big.json'):
        with open(os.path.join(ONED, name)) as fh:
            rows += [r for r in json.load(fh) if 'S2_half' in r]
    out, seen = defaultdict(list), set()
    for r in rows:
        key = (r['L'], round(r['z'], 5))
        if key in seen:
            continue
        seen.add(key)
        out[r['L']].append((r['z'], r['S2_half'], r['S2_half_sem']))
    return {L: np.array(sorted(v)) for L, v in sorted(out.items())}


def curve_panel(ax, data, xmax, coef, ylabel, title, theory):
    Ls = sorted(data)
    col = {L: CM(0.05 + 0.85 * i / max(len(Ls) - 1, 1)) for i, L in enumerate(Ls)}
    drawn = 0
    for L in Ls:
        a = data[L]
        if xmax is not None:
            a = a[a[:, 0] <= xmax]
        if len(a) == 0:
            continue
        drawn += len(a)
        ax.errorbar(a[:, 0], a[:, 1], yerr=a[:, 2], marker='o', ms=3.8, lw=0.9,
                    capsize=1.3, color=col[L], label='L=%d' % L)
    total = sum(len(v) for v in data.values())
    # The y range comes from the DATA, not from the theory curve: c z^2 reaches 80 on
    # the 2D full range, which squashed every measurement onto the x axis.  The theory
    # line is drawn only where it stays inside the frame.
    allv = np.concatenate([v[:, 1] + v[:, 2] for v in data.values()])
    ymax = float(allv.max()) * 1.12
    if xmax is not None:
        ax.set_xlim(0, xmax)
        zlo, zhi = 0.0, xmax
    else:
        zmax = max(v[:, 0].max() for v in data.values())
        ax.set_xlim(0, zmax * 1.02)
        zlo, zhi = 0.0, zmax
    zz = np.linspace(zlo, zhi, 600)
    yy = coef * zz ** 2
    keep = yy <= ymax
    ax.plot(zz[keep], yy[keep], 'k--', lw=1.1, label=theory)
    ax.set_ylim(0, ymax)
    label = ('%s: all %d points' % (title, drawn) if xmax is None
             else '%s: %d of %d points' % (title, drawn, total))
    ax.set_title(label, fontsize=10)
    ax.set_xlabel(r'$z=n\,e^{-2\beta}$')
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='lower right')


def ratio_panel(ax, groups, order, xmax, ylabel, title):
    col = {L: CM(0.05 + 0.85 * i / max(len(order) - 1, 1)) for i, L in enumerate(order)}
    seen, n = set(), 0
    for z, sel in groups:
        if z > xmax or len(sel) < 2:
            continue
        m = np.array([s[1] for s in sel])
        e = np.array([s[2] for s in sel])
        w = 1.0 / e ** 2
        mu = (w * m).sum() / w.sum()
        for L, mean, sem in sel:
            off = (order.index(L) - (len(order) - 1) / 2) * 0.012 * xmax
            ax.errorbar(z + off, mean / mu, yerr=sem / mu, marker='o', ms=4, lw=0,
                        elinewidth=1.0, capsize=1.5, color=col[L],
                        label='L=%d' % L if L not in seen else None)
            seen.add(L)
            n += 1
    ax.axhline(1.0, color='k', ls='--', lw=0.9)
    ax.set_xlim(0, xmax * 1.03)
    ax.set_ylim(0.8, 1.2)
    ax.set_xlabel(r'$z=n\,e^{-2\beta}$')
    ax.set_ylabel(ylabel)
    ax.set_title('%s (%d points)' % (title, n), fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=6.5, frameon=False, ncol=2, loc='lower right')


def main():
    d2, d1 = series_2d(), series_1d()
    lab2 = r'$\overline{D_2}$  (2D corner combination)'
    lab1 = r'$\overline{S_2}$  (1D half ring)'
    rat = r'value / (mean over $L$ at the same $z$)'

    fig, axes = plt.subplots(2, 3, figsize=(16.4, 8.8))
    curve_panel(axes[0][0], d2, None, 16 / 81, lab2, '(a) 2D, linear axis, full range', r'$(16/81)z^2$')
    curve_panel(axes[0][1], d2, 12.0, 16 / 81, lab2, r'(b) 2D zoom, $z\leq12$', r'$(16/81)z^2$')
    curve_panel(axes[0][2], d1, None, 1.0, lab1, '(c) 1D, linear axis, full range', r'$z^2$')
    curve_panel(axes[1][0], d1, 3.2, 1.0, lab1, r'(d) 1D zoom, $z\leq3.2$', r'$z^2$')

    pts = H.load_pooled()
    g2 = [(z, [(p['L'], p['mean'], p['sem']) for p in sel])
          for z, sel in H.group_by_z(pts).items()]
    ratio_panel(axes[1][1], g2, sorted({p['L'] for p in pts}), 11.0, rat,
                '(e) 2D collapse quality')

    with open(os.path.join(ONED, 'data_matched_big.json')) as fh:
        rows = [r for r in json.load(fh) if 'S2_half' in r]
    by = defaultdict(list)
    for r in rows:
        by[round(r['z'], 3)].append((r['L'], r['S2_half'], r['S2_half_sem']))
    ratio_panel(axes[1][2], sorted(by.items()), sorted({r['L'] for r in rows}), 11.0, rat,
                '(f) 1D collapse quality')

    fig.tight_layout()
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, 'linear_z.png'), dpi=180)
    fig.savefig(os.path.join(FIG, 'linear_z.pdf'))
    plt.close(fig)
    print('2D points', sum(len(v) for v in d2.values()), '| 1D points', sum(len(v) for v in d1.values()))
    print('figure ->', os.path.join(FIG, 'linear_z.png'))


if __name__ == '__main__':
    main()
