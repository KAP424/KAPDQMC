import sys, glob, os, math
sys.dont_write_bytecode = True
H = 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030'
sys.path.insert(0, H)
import numpy as np
import crossover_structure as C
import bessel_check as B

rng = np.random.default_rng(5)


def head(s):
    print()
    print('=== ' + s + ' ===')


def jstats(a, beta):
    lw = np.zeros(a + 1)
    for j in range(1, a + 1):
        lw[j] = lw[j - 1] + 2 * math.log((a - j + 1) / j) - 4 * beta
    p = np.exp(lw - lw.max())
    p /= p.sum()
    k = np.arange(a + 1)
    m = float((p * k).sum())
    v = float((p * (k - m) ** 2).sum())
    return m, v


head('A. pooled-per-point test of the sampled j distribution against the exact law')
pool = {}
for d in ('data_matched_z', 'data_matched_largez', 'data_gapfill'):
    for f in sorted(glob.glob(os.path.join(H, d, '*.csv'))):
        r = np.genfromtxt(f, delimiter=',', names=True)
        if r.size == 0:
            continue
        key = (int(r['L'][0]), round(float(r['beta'][0]), 4))
        pool.setdefault(key, []).append(np.atleast_1d(r['j']))
pulls = []
for (L, b), arrs in sorted(pool.items()):
    j = np.concatenate(arrs)
    m, v = jstats(L * L, b)
    pulls.append(((j.mean() - m) / math.sqrt(v / j.size), L, b, j.size))
pv = np.array([p[0] for p in pulls])
print('points', len(pv), ' |pull|>2:', int((abs(pv) > 2).sum()), ' >3:', int((abs(pv) > 3).sum()), ' rms %.2f' % float(np.sqrt((pv ** 2).mean())))
for p in sorted(pulls, key=lambda q: -abs(q[0]))[:4]:
    print('  worst pull %.2f at L=%d beta=%.4f n=%d' % p)

head('B. entropy z_* (h=0.5): bessel_check.py data vs pooled crossover_structure data')
rows, ks, vs = B.load_rows()
print('bessel_check rows', len(rows), ' rows with L=18 beta=1:', sum(1 for r in rows if r['L'] == 18 and abs(r['beta'] - 1) < 1e-3), ' anchor rows:', [(r['L'], r['mean']) for r in rows if r['src'].startswith('old')])
zb = B.zstar_per_L(rows)
pts, base = C.load_points()
byL = C.per_L(pts)
for L in sorted(zb):
    zs = np.array([p['z'] for p in byL[L]])
    hs = np.array([p['mean'] / p['Dgs'] for p in byL[L]])
    zc, w = C.crossing(zs, hs, 0.5)
    if zc is None:
        continue
    print('  L=%d  bessel_check z*=%.3f   crossover_structure z_0.5=%.3f  (bracket width %.2f)' % (L, zb[L], zc, w))
cur = [p for p in pts if p['L'] == 36 and abs(p['beta'] - 3.0) < 1e-3]
print('current L=36 beta=3: Dbar=%.4f +- %.4f ; hard-coded anchor in bessel_check: 0.24214 +- 0.00284' % (cur[0]['mean'], cur[0]['sem']))

head('C. p(h) on the common sizes L=18..36 (well-bracketed crossings only)')
levels = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
boot = []
for _ in range(300):
    boot.append({(p['L'], p['beta']): p['D'][rng.integers(0, p['n'], p['n'])].mean() for p in pts})


def lnz(L, h, bm=None):
    zs = np.array([q['z'] for q in byL[L]])
    hs = np.array([(q['mean'] if bm is None else bm[(q['L'], q['beta'])]) / q['Dgs'] for q in byL[L]])
    z, w = C.crossing(zs, hs, h)
    return (None if z is None else math.log(z)), w


res = {}
for h in levels:
    Ls = [L for L in (18, 24, 30, 36) if lnz(L, h)[0] is not None and lnz(L, h)[1] <= C.MAX_BRACKET]
    if len(Ls) < 3:
        print('h=%.1f: sizes %s -> no fit' % (h, Ls))
        continue
    x = np.log(Ls)
    p0 = np.polyfit(x, [lnz(L, h)[0] for L in Ls], 1)[0]
    ps = []
    for bm in boot:
        y = [lnz(L, h, bm)[0] for L in Ls]
        if all(v is not None for v in y):
            ps.append(np.polyfit(x, y, 1)[0])
    res[h] = (p0, float(np.std(ps)))
    print('h=%.1f sizes %s  p=%.3f +- %.3f' % (h, Ls, p0, res[h][1]))
hk = sorted(res)
for h in hk[:-1]:
    d = res[hk[-1]][0] - res[h][0]
    print('  p(%.1f)-p(%.1f) = %.3f  (%.1f sigma)' % (hk[-1], h, d, d / math.hypot(res[h][1], res[hk[-1]][1])))

head('D. L=3 node outputs: count fraction vs weight')
print('count fraction 14067/48620 = %.3f ; weight 16.6387/81 = %.4f' % (14067 / 48620, 16.6386954348408 / 81))
