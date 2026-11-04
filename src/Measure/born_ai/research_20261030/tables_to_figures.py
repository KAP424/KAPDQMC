'''Figures that replace the data tables in the paper text.

Each numbered function draws the panel for one table and writes one PNG into figures/.
All axes are linear unless a log axis is the only way to show the quantity (noted in the label).

  T3  §3.8  small-z verification: 2D A(L), region ratios, 1D exact enumeration
  T4  §4.1  g drift with its profile-likelihood intervals; the beta-anchored h rows
  T5  §4.3  fixed beta, h versus L
  T6  §4.4  p(h) with errors
  T7  §5.3  one-dimensional S2bar/z^2 at fixed z versus L, and versus sample count
  T9  §5.5  weak-side A/(c/4) and the full crossover scan
  T10 §5.6  even-L zero mode

Usage: D:/app/conda/python.exe tables_to_figures.py
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
CM = plt.get_cmap('viridis')
sys.path.insert(0, ONED)          # onedim_born lives with the 1D data
sys.path.insert(0, HERE)


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name), dpi=200)
    plt.close(fig)
    print('wrote', name)


# ------------------------------------------------------------------ §3.8
def t3_small_z():
    import small_z_mechanism as S
    import headline_numbers as H

    Ls = [6, 9, 12, 18, 24, 30, 36]
    A = []
    for L in Ls:
        _, G0, masks = S.ground(L)
        _, CD, _ = S.bright_formula(G0, masks, L)
        A.append(CD / L ** 4)
    A = np.array(A)

    fig, axes = plt.subplots(1, 3, figsize=(14.0, 4.3))
    ax = axes[0]
    ax.plot(Ls, A, 's-', ms=5, lw=1.2, color='#2a78d6', label=r'$16/81+4D_{V0}(L)/L^4$')
    ax.axhline(16 / 81, color='k', ls='--', lw=1.0, label=r'$16/81$')
    ax.plot([3], [0.18071], '*', ms=13, color='crimson', label=r'$L=3$ exact ($20.5\%$ node weight)')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$A(L)=\lim_{z\to0}\bar D_2/z^2$')
    ax.set_ylim(0.17, 0.20)
    ax.set_title('(a) 2D small-z amplitude', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=7.5, frameon=False)

    ax = axes[1]
    # Computed from the CSVs, not copied: an earlier draft of this figure carried
    # hard-coded ratios, which is exactly how stale numbers get into the paper.
    th = {'SA': 8 / 9, 'SB': 56 / 81, 'SC': 8 / 9, 'D': 16 / 81}
    pts = H.load_pooled()
    mk = {'SA': 'o', 'SB': 'D', 'SC': 's', 'D': '^'}
    for i, c in enumerate(('SA', 'SB', 'SC', 'D')):
        for z0 in (0.089, 0.357, 0.803):
            sel = [p for p in pts if abs(p['z'] - z0) < 0.001]
            if not sel:
                continue
            r = np.array([p['stats'][c][0] / (th[c] * p['z'] ** 2) for p in sel])
            e = np.array([p['stats'][c][1] / (th[c] * p['z'] ** 2) for p in sel])
            w = 1 / e ** 2
            ax.errorbar([z0 + (i - 1.5) * 0.012], [(w * r).sum() / w.sum()],
                        yerr=[1 / np.sqrt(w.sum())], marker=mk[c], ms=5, lw=0,
                        elinewidth=1.0, capsize=2, color=CM(0.08 + 0.27 * i),
                        label=c if z0 == 0.089 else None)
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xlabel(r'$z$')
    ax.set_ylabel('measured / leading order')
    ax.set_ylim(0.5, 1.05)
    ax.set_title('(b) region entropies / $4f(1-f)z^2$, 2D', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False, ncol=2)

    ax = axes[2]
    with open(os.path.join(ONED, 'data_exact_L9.json')) as fh:
        ex = json.load(fh)
    thm = 1.02015
    z = np.array([r['z'] for r in ex])
    ax.plot(z, [r['S2_half_z2'] for r in ex], 'o-', ms=5, lw=1.1, color='#2a78d6',
            label=r'$L=9$ exact, $\bar S_2/z^2$')
    ax.axhline(thm, color='k', ls='--', lw=1.0, label=r'$1+4V_0/n^2$')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'$z$')
    ax.set_ylabel(r'$\bar S_2/z^2$')
    ax.set_title('(c) 1D exact enumeration, $L=9$', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_small_z.png')


# ------------------------------------------------------------------ §4.1
def t4_drift():
    import headline_numbers as H
    rows = [('all', -0.0415, -0.0495, -0.0335, 27.5, 5.2),
            ('drop L=6', -0.0635, -0.0830, -0.0440, 10.1, 3.2),
            (r'$z\geq1.4$', -0.0365, -0.0460, -0.0275, 15.3, 3.9),
            (r'$z\geq3.2$', -0.0215, -0.0330, -0.0105, 3.7, 1.9)]
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.4))
    ax = axes[0]
    y = np.arange(len(rows))[::-1]
    for k, (lab, g, lo, hi, dchi, sig) in enumerate(rows):
        ax.errorbar([g], [y[k]], xerr=[[g - lo], [hi - g]], marker='o', ms=6, lw=1.4,
                    capsize=3, color=CM(0.1 + 0.2 * k))
        ax.text(hi + 0.006, y[k], '%.1f$\\sigma$' % sig, va='center', fontsize=9)
    ax.axvline(0, color='k', ls='--', lw=1.0)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-0.10, 0.015)
    ax.set_xlabel(r'$g$  in  $\bar D_2=f(z)[1+g\ln(L/18)]$')
    ax.set_title('(a) 2D drift, profile-likelihood intervals', fontsize=10)
    ax.grid(alpha=0.25, axis='x')

    ax = axes[1]
    pts, base = H.C.load_points()
    for zt, col in ((5.934, CM(0.25)), (10.55, CM(0.7))):
        sel = sorted([p for p in pts if abs(p['z'] - zt) / zt < 2e-3], key=lambda p: p['L'])
        ax.errorbar([p['L'] for p in sel], [p['mean'] for p in sel], yerr=[p['sem'] for p in sel],
                    marker='o', ms=5, lw=1.3, capsize=2, color=col, label=r'$z=%.2f$' % zt)
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\bar D_2$')
    ax.set_title(r'(b) $\bar D_2$ is flat in $L$ at fixed $z$', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_drift.png')


# ------------------------------------------------------------------ §4.3
def t5_fixed_beta_h():
    import headline_numbers as H
    pts, _ = H.C.load_points()
    betas = (0.5, 1.0, 1.5, 2.0)
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for k, b in enumerate(betas):
        sel = sorted([p for p in pts if abs(p['beta'] - b) < 2e-3 and p['L'] >= 18],
                     key=lambda p: p['L'])
        ax.errorbar([p['L'] for p in sel], [p['mean'] / p['Dgs'] for p in sel],
                    yerr=[p['sem'] / p['Dgs'] for p in sel], marker='o', ms=5, lw=1.2,
                    capsize=2, color=CM(0.08 + 0.26 * k), label=r'$\beta=%g$' % b)
    ax.axhline(1.0, color='k', ls=':', lw=0.9)
    ax.set_xlabel('L')
    ax.set_ylabel(r'$h=\bar D_2/D_{\rm gs}(L)$')
    ax.set_title('fixed $\\beta$: $h$ is flat, so the $\\ln L$ growth survives', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_fixed_beta_h.png')


# ------------------------------------------------------------------ §4.4
def t6_p_of_h():
    import headline_numbers as H
    h = np.array([0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    p = np.array([0.590, 0.684, 0.770, 0.891, 1.038, 1.242, 1.592])
    e = np.array([0.023, 0.021, 0.026, 0.027, 0.045, 0.047, 0.210])
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.3))
    ax = axes[0]
    ax.errorbar(h, p, yerr=e, marker='o', ms=5, lw=1.3, capsize=3, color='k')
    ax.axhline(2.0, color='crimson', ls='--', lw=1.0, label=r'$p=2$: fixed-$\beta$ direction')
    ax.set_xlabel('threshold $h$')
    ax.set_ylabel(r'$p$  in  $z_h\propto L^p$')
    ax.set_title('(a) no single crossover exponent', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)

    ax = axes[1]
    sL = [0.0, 0.54, 0.92, 1.06, 1.28, 1.46]
    Ls = [6, 12, 18, 24, 30, 36]
    ax.plot(Ls, sL, 'o-', ms=5, lw=1.3, color='k')
    ax.axhline(0.0, color='crimson', ls='--', lw=1.0, label='single scale: $s_L\\equiv0$')
    ax.set_xlabel('L')
    ax.set_ylabel(r'shift $s_L$ in $\ln z_h = G(h)+s_L$')
    ax.set_title(r'(b) the shifts grow monotonically', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_p_of_h.png')


# ------------------------------------------------------------------ §5.3
def t7_onedim_fixed_z():
    with open(os.path.join(ONED, 'data_matched_big.json')) as fh:
        rows = [r for r in json.load(fh) if 'S2_half' in r]
    byz = defaultdict(list)
    for r in rows:
        byz[round(r['z'], 3)].append(r)
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axes[0]
    keys = sorted(byz)
    for k, z in enumerate(keys):
        sel = sorted(byz[z], key=lambda r: r['L'])
        v = np.array([r['S2_half'] / z ** 2 for r in sel])
        e = np.array([r['S2_half_sem'] / z ** 2 for r in sel])
        ax.errorbar([r['L'] for r in sel], v, yerr=e, marker='o', ms=4, lw=1.0, capsize=1.5,
                    color=CM(0.05 + 0.9 * k / max(len(keys) - 1, 1)))
        # Right-hand label ladder on a fixed grid: the two largest z rows sit within
        # 0.01 of each other on a linear axis, so data-anchored text always collides.
        ax.annotate('$z=%g$' % z, (1.0, 0.06 + 0.115 * (len(keys) - 1 - k)),
                    xycoords='axes fraction', fontsize=7.5, va='center',
                    color=CM(0.05 + 0.9 * k / max(len(keys) - 1, 1)))
    ax.axhline(1.0, color='k', ls='--', lw=1.0)
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\bar S_2/z^2$')
    ax.set_xlim(5, 78)
    ax.set_ylim(0, 1.1)
    ax.set_title(r'(a) fixed $z$: $\bar S_2/z^2$ falls with $L$', fontsize=10)
    ax.grid(alpha=0.25)

    ax = axes[1]
    ns = [4096, 32768, 65536, 262144]
    chi = [0.4, 5.8, 17.5, 47.0]
    ax.loglog(ns, chi, 'o-', ms=6, lw=1.3, color='#2a78d6')
    ax.plot(ns, np.array(ns) / 4096 * chi[0], 'k--', lw=1.0, label='slope 1: statistical scaling')
    ax.set_xlabel('samples per point ($z=0.1$)')
    ax.set_ylabel(r'$\chi^2_{\rm dof}$')
    ax.set_title('(b) growing with sample count = real drift', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_onedim_fixed_z.png')


# ------------------------------------------------------------------ §5.5
def t9_weak_side():
    vals = [(0.25, 0.998, 0.005), (0.50, 1.009, 0.013), (1.00, 0.955, 0.026),
            (1.50, 0.974, 0.036), (2.00, 1.548, 0.033), (3.00, 0.142, 0.006)]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.4))
    ax = axes[0]
    for b, v, e in vals:
        weak = b <= 0.5
        ax.errorbar([b], [v], yerr=[e], marker='o', ms=7, capsize=3, lw=1.3,
                    mfc='C0' if weak else 'white', mec='C0' if weak else 'C1',
                    color='C0' if weak else 'C1')
    ax.axhline(1.0, color='k', lw=1.3)
    ax.axhspan(0.96, 1.04, color='k', alpha=0.08)
    ax.set_xscale('log')
    ax.set_xticks([0.25, 0.5, 1, 1.5, 2, 3])
    ax.set_xticklabels(['0.25', '0.5', '1', '1.5', '2', '3'])
    ax.minorticks_off()
    ax.set_xlim(0.2, 3.8)
    ax.set_ylim(0, 1.8)
    ax.set_xlabel(r'$\beta$')
    ax.set_ylabel(r'$A/(c/4)$,  $c/4=0.25$')
    ax.set_title('(a) fitted $\\ln L$ coefficient vs exact CFT', fontsize=10)
    ax.grid(alpha=0.25, axis='y')

    ax = axes[1]
    with open(os.path.join(ONED, 'data_fixed_big.json')) as fh:
        rows = [r for r in json.load(fh) if 'S2_half' in r]
    gs = sorted({(r['L'], r['S2_half_gs']) for r in rows})
    ax.plot([g[0] for g in gs], [g[1] for g in gs], 'k-', lw=1.8, label='ground state')
    for k, b in enumerate((0.25, 0.5, 1.0)):
        sel = sorted([r for r in rows if r['beta'] == b], key=lambda r: r['L'])
        ax.errorbar([r['L'] for r in sel], [r['S2_half'] for r in sel],
                    yerr=[r['S2_half_sem'] for r in sel], marker='o', ms=4, lw=1.1,
                    capsize=1.5, color=CM(0.15 + 0.3 * k), label=r'$\beta=%g$' % b)
    ax.set_xscale('log')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\bar S_2$ (half ring)')
    ax.set_title(r'(b) parallel to the ground state = same $\ln L$ slope', fontsize=10)
    ax.grid(alpha=0.25, which='both')
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_weak_side.png')


# ------------------------------------------------------------------ §5.6
def t10_even_L():
    rows = [(4, 1.01495, 1.08081), (5, 1.05618, 1.05618), (6, 0.97379, 1.04057),
            (7, 1.03129, 1.03130), (8, 0.97140, 1.02466)]
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    Ls = np.array([r[0] for r in rows])
    ax.plot(Ls, [r[2] for r in rows], 'k--o', ms=5, lw=1.1, label=r'$1+4V_0/n^2$ (theorem)')
    for L, v, t in rows:
        ax.plot([L], [v], 's', ms=8, color='crimson' if L % 2 == 0 else '#2a78d6')
    ax.plot([], [], 's', ms=8, color='crimson', label='even L (degenerate zero mode)')
    ax.plot([], [], 's', ms=8, color='#2a78d6', label='odd L')
    ax.set_xlabel('L')
    ax.set_ylabel(r'$\bar S_2/z^2$ at $\beta=4$ (exact)')
    ax.set_title('even L: degenerate $k=\\pm\\pi/2$ zero mode breaks the small-z law', fontsize=10)
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    save(fig, 't_even_L.png')


if __name__ == '__main__':
    sys.path.insert(0, ONED)
    t3_small_z()
    t4_drift()
    t5_fixed_beta_h()
    t6_p_of_h()
    t7_onedim_fixed_z()
    t9_weak_side()
    t10_even_L()
