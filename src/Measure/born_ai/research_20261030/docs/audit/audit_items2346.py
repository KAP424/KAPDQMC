import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DZ = os.path.join(ROOT, "data_matched_z")
DL = os.path.join(ROOT, "data_matched_largez")


def load_dir(d):
    out = {}
    for f in sorted(glob.glob(os.path.join(d, "*.csv"))):
        arr = np.genfromtxt(f, delimiter=",", names=True)
        if arr.size == 0:
            continue
        L = int(arr["L"][0]); beta = float(arr["beta"][0]); z = float(arr["z"][0])
        out[(L, beta)] = dict(f=os.path.basename(f), L=L, beta=beta, z=z, a=arr)
    return out


matched = load_dir(DZ); largez = load_dir(DL)
baseline = json.load(open(os.path.join(ROOT, "baseline_D.json")))
ff = json.load(open(os.path.join(ROOT, "final_fit.json")))

def stat(a, key="D"):
    v = a[key]; n = len(v); m = v.mean()
    se = v.std(ddof=1)/np.sqrt(n) if n > 1 else 0.0
    return m, se, n

# ---------------------------------------------------------------- ITEM 2
print("=" * 78)
print("ITEM 2: A(z) = D/z^2 table (z=0.089, 0.357, 0.803, 3.21; L=6..30)")
print("=" * 78)
print("SUMMARY: 0.089: 0.181 0.248 0.178 0.224 0.124   spread 2.0x")
print("SUMMARY: 0.357: 0.168 0.151 0.140 0.136 0.219   spread 1.6x")
print("SUMMARY: 0.803: 0.121 0.114 0.107 0.117 0.113   spread 1.13x")
print("SUMMARY: 3.21 : 0.024 0.024 0.024 0.025 0.024   spread 1.06x")
print()

def pick(z_target, Ls=(6, 12, 18, 24, 30)):
    """For a target z, find the matching (L,beta) entry and return D/z^2."""
    vals = {}
    for L in Ls:
        best = None
        for d in (matched, largez):
            for (LL, beta), v in d.items():
                if LL != L:
                    continue
                if abs(v["z"] - z_target) < 0.01:
                    if best is None or abs(v["z"] - z_target) < abs(best["z"] - z_target):
                        best = v
        if best is not None:
            D, sD, n = stat(best["a"])
            vals[L] = (D / best["z"] ** 2, D, sD, best["z"], n, best["f"])
    return vals

for zt in (0.089, 0.357, 0.803, 3.21):
    v = pick(zt)
    s = "  z=%.3f: " % zt
    for L in (6, 12, 18, 24, 30):
        s += ("%6.3f " % v[L][0]) if L in v else "   -   "
    ratios = [v[L][0] for L in v]
    print(s + "  spread=%.3fx  (max/min)" % (max(ratios) / min(ratios)))
    for L in (6, 12, 18, 24, 30):
        if L in v:
            print("      L=%2d z_used=%.4f D=%.6f sd=%.6f n=%d  A=%.5f  %s"
                  % (L, v[L][3], v[L][1], v[L][2], v[L][4], v[L][0], v[L][5]))

# also the "A=0.1808 at L=6 z=0.089" claim and A_exact
v = pick(0.089)
print("\n  claim: L=6, z=0.089 -> D/z^2 = 0.1808 ; compute = %.5f" % v[6][0])
print("  claim: A_exact = 0.1807 (from 14.63775 * q, q=z^2/81): 14.63775/81 = %.7f"
      % (14.63775 / 81))

# ---------------------------------------------------------------- ITEM 3
print()
print("=" * 78)
print("ITEM 3: c2 model dependence (D_gs from baseline_D.json, L=6..60 step 3)")
print("=" * 78)
Ls = sorted(int(k) for k in baseline)
Dg = np.array([baseline[str(L)]["D"] for L in Ls], float)
Ls = np.array(Ls, float)
print("L values:", Ls.astype(int))

models = {
    "pure log":      lambda L: [np.log(L)],
    "+a/L":          lambda L: [np.log(L), 1 / L],
    "+a/L^{1/2}":    lambda L: [np.log(L), 1 / np.sqrt(L)],
    "+a/L^{3/2}":    lambda L: [np.log(L), 1 / L ** 1.5],
}
print("\nSUMMARY:            L>=24      L>=6")
claim = {"pure log": (0.1750, 0.1963), "+a/L": (0.1554, 0.1558),
         "+a/L^{1/2}": (0.1358, 0.1147), "+a/L^{3/2}": (0.1620, 0.1693)}
for name, fn in models.items():
    line = "%-14s" % name
    for lim in (24, 6):
        m = Ls >= lim
        cols = fn(Ls[m])
        A = np.column_stack(cols + [np.ones(m.sum())])
        coef, *_ = np.linalg.lstsq(A, Dg[m], rcond=None)
        resid = Dg[m] - A @ coef
        rms = np.sqrt(np.mean(resid ** 2))
        nc = 1  # count of log terms
        dof = m.sum() - A.shape[1]
        line += "  c=%8.4f rms=%.2e dof=%d" % (coef[0], rms, dof)
    print(line)
print("  (claimed rms 2.4e-6 for +a/L, L>=24)")

# detail for +a/L L>=24
m = Ls >= 24
A = np.column_stack([np.log(Ls[m]), 1 / Ls[m], np.ones(m.sum())])
coef, *_ = np.linalg.lstsq(A, Dg[m], rcond=None)
resid = Dg[m] - A @ coef
print("\n  +a/L L>=24: coef = c2=%.7f  a=%.5f  d=%.5f  rms=%.3e"
      % (coef[0], coef[1], coef[2], np.sqrt(np.mean(resid ** 2))))
print("  SUMMARY ancillary: 'b = -0.74 is the 1/L coef; constant is +0.133'")
print("  -> a=%.5f  d=%.5f" % (coef[1], coef[2]))

# ---------------------------------------------------------------- ITEM 4
print()
print("=" * 78)
print("ITEM 4: z*(L) by D/D_gs = 1/2 crossing; power law; drift")
print("=" * 78)

def crossing(L, target):
    """log-z linear interpolation of D(z) at fixed L until D = target."""
    pts = []
    for d in (matched, largez):
        for (LL, beta), v in d.items():
            if LL != L:
                continue
            D, sD, n = stat(v["a"])
            pts.append((v["z"], D, v["f"]))
    pts.sort()
    for i in range(len(pts) - 1):
        z0, d0, _ = pts[i]; z1, d1, _ = pts[i + 1]
        if (d0 - target) * (d1 - target) <= 0 and d0 != d1:
            f = (target - d0) / (d1 - d0)
            return float(np.exp(np.log(z0) + f * (np.log(z1) - np.log(z0)))), (z0, z1)
    return None, None

zs = {}
for L in (6, 12, 18, 24, 30, 36):
    tgt = baseline[str(L)]["D"] / 2
    zc, br = crossing(L, tgt)
    zs[L] = zc
    print("  L=%2d target=%.6f  z*=%.5f   (bracketed by z=%.4f,%.4f)  final_fit=%s"
          % (L, tgt, zc, br[0], br[1], ff["zstar"][str(L)]))

Lv = np.array(sorted(zs), float)
zv = np.array([zs[L] for L in sorted(zs)])
x = np.log(Lv); y = np.log(zv)
n = len(x)
sx = x.sum(); sy = y.sum()
sxx = (x * x).sum(); sxy = (x * y).sum()
slope = (n * sxy - sx * sy) / (n * sxx - sx * sx)
inter = (sy - slope * sx) / n
resid = y - (slope * x + inter)
sse = (resid ** 2).sum()
s2 = sse / (n - 2)
se_slope = np.sqrt(s2 * n / (n * sxx - sx * sx))
print("\n  power law fit: z* = %.5f L^{%.4f +- %.4f}   rms(ln) = %.4f"
      % (np.exp(inter), slope, se_slope, np.sqrt(sse / n)))
print("  SUMMARY: z* = 0.348 L^{0.799 +- 0.046}")
print("  p=1 discrepancy: %.2f sigma" % ((1 - slope) / se_slope))
print("  SUMMARY: p=1 is 4.3 sigma away")

# linear z* = a L (through origin)
a_lin = (Lv * zv).sum() / (Lv * Lv).sum()
r = zv - a_lin * Lv
print("\n  linear z*=a L (through origin): a=%.6f  rms=%.4f"
      % (a_lin, np.sqrt((r ** 2).mean())))
print("  final_fit zstar_linear_coef = %.6f" % ff["zstar_linear_coef"])
print("  SUMMARY: power-law rms 0.056, linear rms 0.388")
# also full linear with intercept
A = np.column_stack([Lv, np.ones_like(Lv)])
c2, *_ = np.linalg.lstsq(A, zv, rcond=None)
print("  linear z*=aL+b (with intercept): a=%.5f b=%.4f rms=%.4f"
      % (c2[0], c2[1], np.sqrt(((zv - A @ c2) ** 2).mean())))

print("\n  z*/L^2 drift:")
for L in (6, 12, 18, 24, 30, 36):
    print("    L=%2d  z*/L^2 = %.5f   z*/L = %.4f" % (L, zs[L] / L ** 2, zs[L] / L))
print("    drift = %.2fx  (%.4f -> %.4f)"
      % (zs[6] / 36 / (zs[36] / 1296), zs[6] / 36, zs[36] / 1296))
print("  SUMMARY: 8.3x (0.0386 -> 0.0046)")
print("  SUMMARY z*/L sequence: 0.231, 0.228, 0.205, 0.168, 0.180, 0.167")

# ---------------------------------------------------------------- ITEM 6
print()
print("=" * 78)
print("ITEM 6: L*(beta) = (c e^{2b})^{1/(2-p)}, c=0.348, p=0.799")
print("=" * 78)
c0 = np.exp(inter)
for b in (1, 2, 3):
    print("  beta=%d: L* = %.2f   (p=1: %.2f)" % (b, (c0 * np.exp(2 * b)) ** (1 / (2 - slope)),
                                                 (a_lin * np.exp(2 * b))))
print("  SUMMARY: 2.2 / 11.6 / 61.6 ; p=1 gives 71.8")
alt = (c0 * np.exp(6)) ** (1 / (2 - slope))
print("  p=1 vs fit at beta=3: %.2f vs %.2f  -> %.1f%% larger"
      % (a_lin * np.exp(6), alt, 100 * (a_lin * np.exp(6) / alt - 1)))
