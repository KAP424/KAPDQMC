import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
from math import lgamma, log, exp
pts = load_all()
def jdist(a, beta):
    lw = np.array([2*(lgamma(a+1)-lgamma(j+1)-lgamma(a-j+1)) - 4*beta*j for j in range(a+1)])
    w = np.exp(lw - lw.max()); w /= w.sum()
    return w
bad = []
rows = []
for (L, b), e in sorted(pts.items()):
    a = L*L
    w = jdist(a, b)
    js = np.arange(a+1)
    mj = (w*js).sum(); vj = (w*js**2).sum() - mj**2
    n = e["n"]
    obs = e["j"].mean()
    zs = (obs - mj)/np.sqrt(vj/n)
    # for small z: also count of j>=1
    p1 = 1 - w[0]
    n1 = (e["j"] >= 1).sum()
    zz = (n1 - n*p1)/np.sqrt(n*p1*(1-p1))
    rows.append((L, b, e["z"][-1], n, mj, obs, zs, n*p1, n1, zz))
print("L beta z n  <j>exp <j>obs pull | expected #(j>=1) observed pull")
for r in rows:
    flag = " <<<" if abs(r[6]) > 2.5 or (r[7] < 200 and abs(r[9]) > 2.5) else ""
    print("L=%2d b=%.4f z=%8.3f n=%4d  <j>exp=%9.4f obs=%9.4f pull=%+5.2f | E#(j>=1)=%7.1f obs=%5d pull=%+5.2f%s" % (r + (flag,)))
pulls = np.array([r[6] for r in rows])
print("rms pull of <j> over 48 points: %.3f ; mean %.3f" % (np.sqrt((pulls**2).mean()), pulls.mean()))
# <j> vs z claims
print("\nratio <j>/z (exact finite a) at selected points:")
for r in rows:
    if r[0] in (6, 36):
        print("  L=%d z=%.3f <j>exp=%.4f  <j>/z=%.3f  <j>/z^2=%.3f" % (r[0], r[2], r[4], r[4]/r[2], r[4]/r[2]**2))
