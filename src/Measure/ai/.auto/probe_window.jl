"""Read-only robustness check on the saved ladder.

Refit c_lnL(D_A) with the same three-parameter model (ln L, 1, 1/L) for several
lower cutoffs, to see how much the corner coefficient depends on the window.
Pure Python (no numpy) Gaussian elimination for the 3x3 normal equations.
"""
import csv
import math
from collections import defaultdict

rows = defaultdict(list)
with open("data/strong_scaling_observations.csv") as fh:
    for r in csv.DictReader(fh):
        rows[r["pattern"]].append((int(r["L"]), float(r["D_A"]), float(r["A_A"])))


def solve3(A, b):
    M = [row[:] + [bb] for row, bb in zip(A, b)]
    for i in range(3):
        p = max(range(i, 3), key=lambda k: abs(M[k][i]))
        M[i], M[p] = M[p], M[i]
        for k in range(i + 1, 3):
            f = M[k][i] / M[i][i]
            for j in range(i, 4):
                M[k][j] -= f * M[i][j]
    x = [0.0] * 3
    for i in (2, 1, 0):
        x[i] = (M[i][3] - sum(M[i][j] * x[j] for j in range(i + 1, 3))) / M[i][i]
    return x


def fit(pts, lmin):
    keep = [(L, y) for L, y in pts if L >= lmin]
    if len(keep) < 4:
        return None
    cols = [[math.log(L), 1.0, 1.0 / L] for L, _ in keep]
    A = [[sum(c[i] * c[j] for c in cols) for j in range(3)] for i in range(3)]
    b = [sum(c[i] * y for c, (_, y) in zip(cols, keep)) for i in range(3)]
    coef = solve3(A, b)
    res = [y - sum(c[i] * coef[i] for i in range(3)) for c, (_, y) in zip(cols, keep)]
    rms = math.sqrt(sum(v * v for v in res) / len(keep))
    return coef[0], rms, len(keep)


print(f"{'pattern':<16}{'c(36)':>9}{'c(54)':>9}{'c(72)':>9}{'c(108)':>9}{'spread%':>9}")
for name in sorted(rows):
    pts = [(L, D) for L, D, _ in rows[name]]
    out = []
    for lmin in (36, 54, 72, 108):
        f = fit(pts, lmin)
        out.append(f"{f[0]:9.4f}" if f else f"{'-':>9}")
    vals = [fit(pts, l)[0] for l in (36, 54, 72, 108) if fit(pts, l)]
    spread = 100 * (max(vals) - min(vals)) / abs(vals[0]) if len(vals) > 1 else 0.0
    print(f"{name:<16}{''.join(out)}{spread:9.2f}")

print()
print("area-law density A_A/L at the largest three L (is it settling?)")
for name in sorted(rows):
    pts = sorted(rows[name])
    tail = pts[-3:]
    print(f"  {name:<16}" + "  ".join(f"L={L}:{A / L:.4f}" for L, _, A in tail))
