import sys, glob, os, math
sys.dont_write_bytecode = True
sys.path.insert(0, 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030')
import numpy as np
import crossover_structure as C
import summary_figure as S

HERE = 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030'
VOL = {'SA': 8 / 9, 'SB': 56 / 81, 'SC': 8 / 9, 'D': 16 / 81}


def head(s):
    print()
    print('=== ' + s + ' ===')


def rd(f):
    a = np.genfromtxt(f, delimiter=',', names=True)
    return {k: np.atleast_1d(a[k]) for k in a.dtype.names}


def files(d):
    return sorted(glob.glob(os.path.join(HERE, d, '*.csv')))


def inventory(pts):
    head('1.7 inventory')
    tot = 0
    for d in ('data_matched_z', 'data_matched_largez', 'data_gapfill', 'data_beta0'):
        fs = files(d)
        n = sum(rd(f)['D'].size for f in fs)
        if d != 'data_beta0':
            tot += n
        print(d, 'files', len(fs), 'rows', n)
    print('total rows in 3 dirs', tot, ' distinct points', len(pts), ' n range',
          min(p['n'] for p in pts), max(p['n'] for p in pts))
    for L in sorted(set(p['L'] for p in pts)):
        sel = sorted([p for p in pts if p['L'] == L], key=lambda p: p['z'])
        print('L=%d: ' % L + ' '.join('%.3g/b%.3g/n%d' % (p['z'], p['beta'], p['n']) for p in sel))
    print('points pooled from >1 file:', [(p['L'], p['beta'], p['nfiles']) for p in pts if p['nfiles'] > 1])


def dup():
    head('L=18 beta=1 duplicate runs')
    v = []
    for d in ('data_matched_z', 'data_matched_largez'):
        for f in files(d):
            r = rd(f)
            if int(r['L'][0]) == 18 and abs(float(r['beta'][0]) - 1.0) < 1e-3:
                D = r['D']
                v.append((D.mean(), D.std(ddof=1) / math.sqrt(D.size)))
                print(d, os.path.basename(f), D.size, v[-1])
    if len(v) == 2:
        print('diff sigma', (v[0][0] - v[1][0]) / math.hypot(v[0][1], v[1][1]))


def beta0(base):
    head('beta=0 normalisation')
    for L in (6, 12, 18, 24, 30, 36):
        dev, n = 0.0, 0
        for f in files('data_beta0'):
            r = rd(f)
            if int(r['L'][0]) == L:
                dev = max(dev, float(np.abs(r['D'] - base[str(L)]['D']).max()))
                n += r['D'].size
        print('L=%d n=%d baseline=%.6f max dev=%.2e' % (L, n, base[str(L)]['D'], dev))
    print('baseline L keys:', sorted(int(k) for k in base))


def fig41(pts):
    head('4.1(a) z=43.85 and 4.1(c) h(L)')
    for p in sorted([q for q in pts if abs(q['z'] - 43.85) / 43.85 < 3e-3], key=lambda q: q['L']):
        print('z=%.3f L=%d beta=%.4f Dbar=%.4f+-%.4f h=%.3f' % (p['z'], p['L'], p['beta'], p['mean'], p['sem'], p['mean'] / p['Dgs']))
    for b in (0.5, 1.0, 1.5, 2.0, 3.0):
        sel = sorted([p for p in pts if abs(p['beta'] - b) < 2e-3], key=lambda p: p['L'])
        print('beta=%.1f ' % b + ' '.join('L%d:%.3f' % (p['L'], p['mean'] / p['Dgs']) for p in sel))
    head('4.1(b) z=0.089 ratios')
    acc = {}
    for f in files('data_matched_z'):
        r = rd(f)
        z = float(r['z'][0])
        if abs(z - 0.089) / 0.089 < 0.01:
            acc[int(r['L'][0])] = {c: r[c] / z ** 2 / VOL[c] for c in VOL}
    for L in sorted(acc):
        print('L=%d n=%d ' % (L, acc[L]['D'].size) + ' '.join('%s=%.3f' % (c, acc[L][c].mean()) for c in VOL))
    allv = {c: np.concatenate([acc[L][c] for L in acc]) for c in VOL}
    print('pooled: ' + ' '.join('%s=%.3f+-%.3f' % (c, allv[c].mean(), allv[c].std(ddof=1) / math.sqrt(allv[c].size)) for c in VOL))


def jdecomp():
    head('4.5 j decomposition (beta=3, data_matched_z)')
    for L in (6, 12, 18):
        D, J, z, beta = [], [], 0.0, 0.0
        for f in files('data_matched_z'):
            r = rd(f)
            if int(r['L'][0]) == L and abs(float(r['beta'][0]) - 3.0) < 0.01:
                D.append(r['D']); J.append(r['j']); z = float(r['z'][0]); beta = float(r['beta'][0])
        D = np.concatenate(D)
        jc = np.minimum(np.concatenate(J).astype(int), 3)
        a = L * L
        lw = np.array([2 * (math.lgamma(a + 1) - math.lgamma(k + 1) - math.lgamma(a - k + 1)) - 4 * beta * k for k in range(a + 1)])
        w = np.exp(lw - lw.max())
        w /= w.sum()
        pe = np.array([w[0], w[1], w[2], w[3:].sum()])
        cnt = np.array([(jc == c).sum() for c in range(4)])
        sig = np.sqrt(D.size * pe * (1 - pe))
        share = np.array([D[jc == c].sum() for c in range(4)]) / D.sum()
        print('L=%d z=%.4f beta=%.4f n=%d counts=%s expected=%s pull=%s' % (L, z, beta, D.size, cnt, np.round(D.size * pe, 1), np.round((cnt - D.size * pe) / sig, 2)))
        print('   share=%s' % np.round(share, 3))


def pofh(pts):
    head('4.1(d) p(h) and z_h crossings')
    ph = S.p_of_h(pts, nboot=300)
    print({h: (round(v[0], 3), round(v[1], 3)) for h, v in ph.items()})
    byL = C.per_L(pts)
    for h in (0.3, 0.5, 0.7, 0.9):
        row = []
        for L in sorted(byL):
            zs = np.array([p['z'] for p in byL[L]])
            hs = np.array([p['mean'] / p['Dgs'] for p in byL[L]])
            z, w = C.crossing(zs, hs, h)
            row.append('L%d:--' % L if z is None else 'L%d:%.2f%s' % (L, z, '' if w <= C.MAX_BRACKET else '*'))
        print('h=%.1f ' % h + ' '.join(row))


def safe(fn, *args):
    try:
        fn(*args)
    except Exception as e:
        print('ERROR in', fn.__name__, repr(e))


pts, base = C.load_points()
for fn, args in ((inventory, (pts,)), (dup, ()), (beta0, (base,)), (fig41, (pts,)), (jdecomp, ()), (pofh, (pts,))):
    safe(fn, *args)
