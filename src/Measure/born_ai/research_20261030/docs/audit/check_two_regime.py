import sys
import numpy as np
sys.path.insert(0, 'D:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030')
import crossover_structure as C
import two_regime_check as T

pts, base = C.load_points()
dgs = {int(k): v['D'] for k, v in base.items()}
LS = [6, 12, 18, 24, 30, 36]


def contour(L, h):
    rows = sorted((p['z'], p['mean'] / dgs[L]) for p in pts if int(p['L']) == L)
    for (z1, h1), (z2, h2) in zip(rows[:-1], rows[1:]):
        if h1 < h <= h2 and np.log(z2 / z1) <= 1.25:
            return float(np.exp(np.log(z1) + (h - h1) / (h2 - h1) * np.log(z2 / z1)))
    return None


def pred(zs, f, L, h, g=0.0):
    y = h * dgs[L] / (1.0 + g * np.log(L / 18.0))
    if y < f[0] or y > f[-1]:
        return float('nan')
    return float(np.exp(np.interp(np.log(y), np.log(f), np.log(zs))))


zs, f = T.pooled_f(pts)
print('Part 1: leave-one-L-out residual ln(z_h measured / z_h predicted), f pooled without that L')
for h in (0.3, 0.4, 0.5, 0.6):
    res, Ls = [], []
    for L in LS:
        zm = contour(L, h)
        if zm is None:
            continue
        zl, fl = T.pooled_f([p for p in pts if int(p['L']) != L])
        pl = pred(zl, fl, L, h)
        if np.isnan(pl):
            continue
        res.append(np.log(zm / pl))
        Ls.append(L)
    sl = np.polyfit(np.log(Ls), res, 1)[0] if len(Ls) > 2 else float('nan')
    print(h, 'L', Ls, 'resid', np.round(res, 3), 'slope vs lnL (extra p beyond collapse):', round(float(sl), 3))

print()
print('Part 2: predicted p(h) if f_L(z) = f(z)(1+g ln(L/18)); measured from MEASURED_LOW')
for h, (pm, Ls) in T.MEASURED_LOW.items():
    row = []
    for g in (0.0, -0.034, -0.048, -0.020, 0.001):
        zp = [pred(zs, f, L, h, g) for L in Ls]
        row.append(round(float(np.polyfit(np.log(Ls), np.log(zp), 1)[0]), 3))
    print(h, 'g=0,-0.034,-0.048,-0.020,+0.001 ->', row, 'measured', pm)

print()
print('Part 3: high-h prediction')
(b1, d1), (b2, d2) = sorted(T.DELTA.items())
Ls = [18, 24, 30, 36]
sD = np.polyfit(np.log(Ls), np.log([dgs[L] for L in Ls]), 1)[0]


def p_high(e1, e2):
    kk = np.log(e2 / e1) / (b2 - b1)
    return 2.0 - 2.0 * sD / kk


print('dlnDgs/dlnL (18-36)', round(float(sD), 3), 'p_pred (independent of h):', round(float(p_high(d1, d2)), 3))
for dd1 in (d1 - 0.011, d1, d1 + 0.011):
    print('Delta(1.5)=', round(dd1, 4), [round(float(p_high(dd1, dd2)), 3) for dd2 in (d2 - 0.009, d2, d2 + 0.009)], '(Delta(2.0) at -1sig, central, +1sig)')
k = np.log(d2 / d1) / (b2 - b1)
for h, zm in T.MEASURED_HIGH.items():
    bm = [0.5 * np.log(L * L / z) for L, z in zip(Ls, zm)]
    bp = [b1 + np.log((1 - h) * dgs[L] / d1) / k for L in Ls]
    p24 = np.polyfit(np.log(Ls[1:]), np.log(zm[1:]), 1)[0]
    print(h, 'beta_h measured', np.round(bm, 3), 'predicted', np.round(bp, 3), 'p(L=24-36 only) =', round(float(p24), 3))
