import sys, math
sys.path.insert(0, 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030')
import numpy as np
import annealed_gaussian as ag


def head(s):
    print()
    print('=== ' + s + ' ===')


def ang(u, v):
    return math.degrees(math.acos(u @ v / (np.linalg.norm(u) * np.linalg.norm(v))))


head('(1) geometry from the hopping rule: a1=d2-d1, a2=d1-d3')
d1 = np.array([0.0, 1.0])
d2 = np.array([math.cos(math.radians(-30)), math.sin(math.radians(-30))])
d3 = np.array([math.cos(math.radians(210)), math.sin(math.radians(210))])
a1 = d2 - d1
a2 = d1 - d3
print('angles between bond vectors:', ang(d1, d2), ang(d2, d3), ang(d1, d3))
print('angle(a1,a2)=', ang(a1, a2), ' |a1|,|a2|=', np.linalg.norm(a1), np.linalg.norm(a2))
print('B corner angles:', ang(a1, a2), ang(-a1, a2), ang(-a1, -a2), ang(a1, -a2))
for L in (6, 12):
    K = ag.dense_hopping(L)
    pos = np.zeros((2 * L * L, 2))
    for x in range(L):
        for y in range(L):
            i = 2 * (x * L + y)
            pos[i] = x * a1 + y * a2
            pos[i + 1] = pos[i] + d1
    lens = []
    for u, v in zip(*np.nonzero(np.triu(K))):
        lens.append(min(np.linalg.norm(pos[v] - pos[u] + n1 * L * a1 + n2 * L * a2)
                        for n1 in (-1, 0, 1) for n2 in (-1, 0, 1)))
    print('L=%d bonds=%d (3L^2=%d) degrees=%s bond length min/max=%.6f/%.6f'
          % (L, len(lens), 3 * L * L, sorted(set(int(v) for v in K.sum(0))), min(lens), max(lens)))

head('(3) bonds cut by each region boundary, split by bond direction')
for L in (6, 12, 18):
    m = ag.region_masks(L)
    w = m['w']
    X, Y = np.meshgrid(np.arange(L), np.arange(L), indexing='ij')
    cell = {'A': X < w, 'B': (X < w) & (Y < 2 * w), 'C': Y < w}
    cnt = {}
    for k, R in cell.items():
        assert np.array_equal(np.repeat(R.ravel(), 2), m[k])
        nx = sum(int(R[x, y] != R[(x + 1) % L, y]) for x in range(L) for y in range(L))
        ny = sum(int(R[x, y] != R[x, (y - 1) % L]) for x in range(L) for y in range(L))
        cnt[k] = (nx, ny)
    comb = [(2 * cnt['A'][i] + cnt['C'][i]) / 3 - cnt['B'][i] for i in (0, 1)]
    print('L=%d w=%d cut bonds (x-type,y-type):' % (L, w), cnt, ' (2A+C)/3-B per direction:', comb)

head('(2) diag(G0) and region occupations, mu=1e-8')
for L in (6, 12):
    G0 = ag.ground_projector(L, mu=1e-8)
    d = np.diag(G0)
    dA = d[0::2]
    dB = d[1::2]
    print('L=%d Tr=%.12f (L^2=%d) max|G^2-G|=%.2e' % (L, d.sum(), L * L, np.abs(G0 @ G0 - G0).max()))
    print('  A: min %.12f max %.12f vs 1/2-1/L^2=%.12f' % (dA.min(), dA.max(), 0.5 - 1 / L ** 2))
    print('  B: min %.12f max %.12f vs 1/2+1/L^2=%.12f' % (dB.min(), dB.max(), 0.5 + 1 / L ** 2))
    m = ag.region_masks(L)
    print('  <N_R>-|R|/2:', {k: float(d[m[k]].sum() - m[k].sum() / 2) for k in 'ABC'})

head('(4) beta / t / z table')
for beta in (0.5, 1.0, 1.5, 2.0, 3.0):
    t = math.exp(-2 * beta)
    print('beta=%.1f t=%.5f z(L=6,18,36)=%.4f %.3f %.2f' % (beta, t, 36 * t, 324 * t, 1296 * t))
print('z(L=12,18,24; beta=1)=%.3f %.3f %.3f' % (144 * math.exp(-2), 324 * math.exp(-2), 576 * math.exp(-2)))
print('beta for z=0.089 at L=6,30:', 0.5 * math.log(36 / 0.089), 0.5 * math.log(900 / 0.089))
