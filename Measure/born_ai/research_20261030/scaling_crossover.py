'''Scaling picture of the crossover between the log law and the volume-fraction law.

Same Born-averaged data D2_bar(L, beta), shown four ways:
 (a) vs ln L, linear axes: the log law is a straight line
 (b) vs L, log-log axes: the strong-measurement branch is a straight line of slope 4
 (c) local exponent s = d ln D2_bar / d ln L vs L at fixed beta
 (d) the same exponent vs z = L^2 exp(-2 beta); the black band is the universal
     strong-side curve s(z) = 2 d ln f / d ln z from the L-pooled matched-z points.

Usage: D:/app/conda/python.exe scaling_crossover.py
'''
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import crossover_structure as C

HERE = os.path.dirname(os.path.abspath(__file__))
FIG = os.path.join(HERE, 'figures')
BETAS = (0.5, 1.0, 1.5, 2.0, 3.0)
CMAP = plt.get_cmap('viridis')


def M(s):
    # mathtext helper: the character @ stands for a backslash (keeps the file free of escapes)
    return s.replace('@', chr(92))


def col(b):
    return CMAP(BETAS.index(b) / 4.5)


def pooled(pts, zmax=11.0):
    groups = {}
    for p in pts:
        if p['z'] > zmax:
            continue
        k = next((g for g in groups if abs(g - p['z']) / g < 2e-3), p['z'])
        groups.setdefault(k, []).append(p)
    zs = sorted(g for g in groups if len({q['L'] for q in groups[g]}) >= 2)
    f, e = [], []
    for g in zs:
        w = np.array([1 / q['sem'] ** 2 for q in groups[g]])
        m = np.array([q['mean'] for q in groups[g]])
        f.append(np.sum(w * m) / np.sum(w))
        e.append(1 / np.sqrt(np.sum(w)))
    return np.array(zs), np.array(f), np.array(e)


def local(sel):
    out = []
    for p, q in zip(sel[:-1], sel[1:]):
        dl = np.log(q['L'] / p['L'])
        s = np.log(q['mean'] / p['mean']) / dl
        e = np.hypot(p['sem'] / p['mean'], q['sem'] / q['mean']) / dl
        out.append((np.sqrt(p['L'] * q['L']), np.sqrt(p['z'] * q['z']), s, e))
    return out


def main():
    pts, base = C.load_points()
    Lg = np.array(sorted(int(k) for k in base))
    Dg = np.array([base[str(k)]['D'] for k in Lg])
    sg_L = np.sqrt(Lg[:-1] * Lg[1:])
    sg = np.diff(np.log(Dg)) / np.diff(np.log(Lg))
    fixed = {b: sorted([p for p in pts if abs(p['beta'] - b) < 2e-3], key=lambda p: p['L'])
             for b in BETAS}
    zs, f, fe = pooled(pts)
    zmid = np.sqrt(zs[:-1] * zs[1:])
    su = 2 * np.diff(np.log(f)) / np.diff(np.log(zs))
    se = 2 * np.hypot(fe[:-1] / f[:-1], fe[1:] / f[1:]) / np.diff(np.log(zs))

    fig, ax = plt.subplots(2, 2, figsize=(12.0, 9.4))
    gs_label = M('ground state $D_{@mathrm{gs}}(L)$')
    ylab = M('$@overline{D_2}$')

    a = ax[0, 0]
    a.plot(np.log(Lg), Dg, 'k-', lw=1.6, label=gs_label)
    for b in BETAS:
        s = fixed[b]
        a.errorbar(np.log([p['L'] for p in s]), [p['mean'] for p in s], yerr=[p['sem'] for p in s],
                   marker='o', ms=4, lw=1.1, capsize=2, color=col(b), label=M('$@beta=%g$') % b)
    a.set_xlabel('ln L')
    a.set_ylabel(ylab)
    a.set_title('(a) vs ln L: the log law is a straight line')
    a.legend(fontsize=7.5, frameon=False, loc='upper left')

    a = ax[0, 1]
    a.plot(Lg, Dg, 'k-', lw=1.6, label=gs_label)
    for b in BETAS:
        s = fixed[b]
        a.errorbar([p['L'] for p in s], [p['mean'] for p in s], yerr=[p['sem'] for p in s],
                   marker='o', ms=4, lw=1.1, capsize=2, color=col(b), label=M('$@beta=%g$') % b)
    Lr = np.linspace(5.0, 13.0, 20)
    a.plot(Lr, 16 / 81 * Lr ** 4 * np.exp(-12.0), '--', color=col(3.0), lw=1.4,
           label=M('$(16/81)@,z^2$ at $@beta=3$ ($@propto L^4$)'))
    a.set_xscale('log')
    a.set_yscale('log')
    a.set_ylim(8e-4, 1.3)
    a.set_xlabel('L')
    a.set_ylabel(ylab)
    a.set_title('(b) log-log: strong measurement is a line of slope 4')
    a.legend(fontsize=7.5, frameon=False, loc='lower right')

    a = ax[1, 0]
    a.plot(sg_L, sg, 'k-', lw=1.6, label=M('ground state $d@ln D_{@mathrm{gs}}/d@ln L$'))
    print('local exponent s at fixed beta  (L_mid, z_mid, s, err)')
    for b in BETAS:
        r = local(fixed[b])
        a.errorbar([x[0] for x in r], [x[2] for x in r], yerr=[x[3] for x in r], marker='o', ms=4,
                   lw=1.1, capsize=2, color=col(b), label=M('$@beta=%g$') % b)
        for x in r:
            print('  beta=%.1f  L=%5.1f  z=%8.3f  s=%5.2f +- %4.2f' % (b, x[0], x[1], x[2], x[3]))
    a.axhline(4.0, color='crimson', ls='--', lw=1.1)
    a.text(7.5, 4.18, M('s = 4: volume-fraction law ($z^2@propto L^4$)'), color='crimson', fontsize=8)
    a.set_xscale('log')
    a.set_ylim(-0.6, 5.2)
    a.set_xlabel('L (geometric mean of the two sizes)')
    a.set_ylabel(M('local exponent $s=d@ln@overline{D_2}/d@ln L$'))
    a.set_title(M('(c) fixed $@beta$: the exponent falls from 4 toward the ground-state value'))
    a.legend(fontsize=7.5, frameon=False, loc='upper right', ncol=2)

    a = ax[1, 1]
    sel = (sg_L >= 6) & (sg_L <= 36)
    a.axvspan(0.05, 0.3, color='crimson', alpha=0.08)
    a.axhspan(sg[sel].min(), sg[sel].max(), color='0.6', alpha=0.25, label='ground state, L=6 to 36')
    a.fill_between(zmid, su - se, su + se, color='k', alpha=0.2)
    a.plot(zmid, su, 'k-o', ms=3.5, lw=1.4,
           label=M('universal strong side $2@,d@ln f/d@ln z$'))
    for b in BETAS:
        r = local(fixed[b])
        a.errorbar([x[1] for x in r], [x[2] for x in r], yerr=[x[3] for x in r], marker='o', ms=4,
                   lw=0.8, capsize=2, color=col(b), label=M('$@beta=%g$') % b)
    a.axhline(4.0, color='crimson', ls='--', lw=1.1)
    a.text(0.055, 4.25,'volume-fraction\nlaw (z < 0.3)', fontsize=8, color='crimson')
    a.text(14.0, 1.0, 'ln L regime: s inside the ground-state band', fontsize=8.5)
    a.set_xscale('log')
    a.set_xlim(0.05, 300)
    a.set_ylim(-0.6, 5.2)
    a.set_xlabel(M('$z=L^2e^{-2@beta}$ (geometric mean of the two points)'))
    a.set_ylabel(M('local exponent $s=d@ln@overline{D_2}/d@ln L$'))
    a.set_title('(d) the same exponent against z')
    a.legend(fontsize=7, frameon=False, loc='upper right', ncol=2)

    fig.tight_layout()
    fig.savefig(os.path.join(FIG, 'scaling_crossover.png'), dpi=200)
    fig.savefig(os.path.join(FIG, 'scaling_crossover.pdf'))
    plt.close(fig)
    print('universal strong-side s(z) = 2 dln f/dln z:')
    for z_, s_, e_ in zip(zmid, su, se):
        print('  z=%7.3f  s=%5.2f +- %4.2f' % (z_, s_, e_))
    print('ground-state s(L): ' + ', '.join('L%.0f:%.2f' % (l, v) for l, v in zip(sg_L, sg) if 6 <= l <= 60))
    print('figure ->', os.path.join(FIG, 'scaling_crossover.png'))


if __name__ == '__main__':
    main()
