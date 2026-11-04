"""Build the verification figures for REPORT.md.

Everything is derived from files that survived the consolidation, so the figures
double as proof that the data the report cites is still present:

  figures_report/fig1_strong_measurement_scaling.png
      (a) A_A/L vs L          - the strong-measurement area law
      (b) D_A vs ln L + fits  - the logarithmic corner term
      (c) c(beta) e^{4 beta}  - convergence onto the corner coefficient
  figures_report/fig2_consolidation.png
      measured file/dir counts per run (from .auto/log.jsonl) and the number of
      files removed per batch (from .auto/removed_manifest.csv)

Run:  python .auto/make_report_figures.py
"""
import csv
import json
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "figures_report")
os.makedirs(OUT, exist_ok=True)

# frozen finite-beta corner coefficients (RESULTS.md, "63 组主拟合", L=36-216,
# three-parameter ln L + const + 1/L model), used only as an independent check.
FROZEN = {
    "one_B_2": [(1.00, 0.06742882), (1.25, 0.03642392), (1.50, 0.01627426),
                (1.75, 0.006505095), (2.00, 0.002472039)],
    "diagonal_B_3": [(1.00, 0.08403503), (1.25, 0.04639887), (1.50, 0.02070907),
                     (1.75, 0.008218309), (2.00, 0.003108257)],
}
COLORS = {"uniform_A": "#1f77b4", "one_B_3": "#2ca02c",
          "one_B_2": "#d62728", "diagonal_B_3": "#9467bd"}


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


def corner_fit(points, lmin=36.0):
    """c in  D_A = c ln L + b + d/L  over L >= lmin."""
    keep = [(L, y) for L, y in points if L >= lmin]
    if len(keep) < 4:
        return None
    cols = [[math.log(L), 1.0, 1.0 / L] for L, _ in keep]
    A = [[sum(c[i] * c[j] for c in cols) for j in range(3)] for i in range(3)]
    b = [sum(c[i] * y for c, (_, y) in zip(cols, keep)) for i in range(3)]
    coef = solve3(A, b)
    res = [y - sum(c[i] * coef[i] for i in range(3)) for c, (_, y) in zip(cols, keep)]
    rms = math.sqrt(sum(v * v for v in res) / len(keep))
    return coef, rms, keep


# ------------------------------------------------------------------ load data
rows = defaultdict(list)
with open(os.path.join(ROOT, "data", "strong_scaling_observations.csv")) as fh:
    for r in csv.DictReader(fh):
        rows[r["pattern"]].append((int(r["L"]), float(r["A_A"]), float(r["A_C"]),
                                   float(r["A_B"]), float(r["D_A"])))
for k in rows:
    rows[k].sort()

fig = plt.figure(figsize=(13.5, 4.2))
gs = fig.add_gridspec(1, 3, wspace=0.30)

# ---- (a) area law ----------------------------------------------------------
ax = fig.add_subplot(gs[0, 0])
for name in ("uniform_A", "one_B_3", "one_B_2", "diagonal_B_3"):
    pts = rows.get(name)
    if not pts:
        continue
    L = [p[0] for p in pts]
    ax.plot(L, [p[1] / p[0] for p in pts], "o-", ms=3.2, lw=1.2,
            color=COLORS[name], label=name)
ax.set_xlabel("$L$")
ax.set_ylabel("$A_A / L$")
ax.set_yscale("log")
ax.set_title("(a) area law: $A_R = aL + c_D\\ln L + b$")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25, lw=0.5, which="both")

# ---- (b) corner logarithm --------------------------------------------------
ax = fig.add_subplot(gs[0, 1])
for name in ("uniform_A", "one_B_3", "one_B_2", "diagonal_B_3"):
    pts = rows.get(name)
    if not pts:
        continue
    fit = corner_fit([(p[0], p[4]) for p in pts])
    L = [p[0] for p in pts]
    ax.plot([math.log(x) for x in L], [p[4] for p in pts], "o", ms=3.0,
            color=COLORS[name])
    if fit:
        coef = fit[0]
        xs = [math.log(x) for x in L]
        ys = [coef[0] * x + coef[1] + coef[2] * math.exp(-x) for x in xs]
        ax.plot(xs, ys, "-", lw=1.1, color=COLORS[name],
                label=f"{name}: $c_D$={coef[0]:.3f}")
ax.set_xlabel("$\\ln L$")
ax.set_ylabel("$D_A = (2A_A + A_C)/3 - A_B$")
ax.set_yscale("log")
ax.set_title("(b) corner term is logarithmic, $c_D$ pattern dependent")
ax.legend(fontsize=7.5, loc="upper left")
ax.grid(alpha=0.25, lw=0.5, which="both")

# ---- (c) tail convergence --------------------------------------------------
ax = fig.add_subplot(gs[0, 2])
for name in ("one_B_2", "diagonal_B_3"):
    pts = rows.get(name)
    if not pts or name not in FROZEN:
        continue
    fit = corner_fit([(p[0], p[4]) for p in pts])
    if not fit:
        continue
    cD = fit[0][0]
    betas = [b for b, _ in FROZEN[name]]
    vals = [c * math.exp(4 * b) for b, c in FROZEN[name]]
    ax.plot(betas, vals, "o-", ms=4, color=COLORS[name],
            label=f"{name}: frozen $c(\\beta)e^{{4\\beta}}$")
    ax.axhline(cD, ls="--", lw=1.1, color=COLORS[name], alpha=0.85)
    ax.annotate(f"$c_D$={cD:.2f}", xy=(betas[-1], cD), xytext=(-4, 5),
                textcoords="offset points", fontsize=8, color=COLORS[name],
                ha="right")
ax.set_xlabel("$\\beta$")
ax.set_ylabel("$c(\\beta)\\,e^{4\\beta}$")
ax.set_title("(c) finite-$\\beta$ corner coefficient $\\to c_D$")
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(alpha=0.25, lw=0.5)

fig.suptitle("Strong-measurement entanglement scaling:  "
             "$S_2(R;\\beta) = A_R\\,e^{-4\\beta} + O(e^{-8\\beta})$", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.93])
p1 = os.path.join(OUT, "fig1_strong_measurement_scaling.png")
fig.savefig(p1, dpi=160)
plt.close(fig)

# ---- (d) consolidation trajectory -----------------------------------------
runs, files, dirs = [], [], []
ledger = os.path.join(ROOT, ".auto", "log.jsonl")
if os.path.isfile(ledger):
    with open(ledger, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            m = rec.get("metrics") or {}
            # the consolidation segment is the only one that records dir_count;
            # its primary metric (file_count) lives in rec["metric"]
            if "dir_count" not in m or rec.get("metric") is None:
                continue
            runs.append(rec.get("experiment") or f"#{rec.get('run')}")
            files.append(int(rec["metric"]))
            dirs.append(int(m.get("dir_count", 0)))

batches = defaultdict(int)
batch_bytes = defaultdict(int)
manifest = os.path.join(ROOT, ".auto", "removed_manifest.csv")
if os.path.isfile(manifest):
    with open(manifest, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            batches[r["batch"]] += 1
            batch_bytes[r["batch"]] += int(r["bytes"])

fig2, (axa, axb) = plt.subplots(1, 2, figsize=(11.0, 3.9))
if runs:
    xs = range(len(runs))
    axa.plot(xs, files, "o-", color="#1f77b4", label="files")
    axa.set_xticks(list(xs))
    axa.set_xticklabels(["baseline", "batch A", "batch B", "batch C"][:len(runs)],
                        fontsize=8)
    axa.set_ylabel("files", color="#1f77b4")
    axa.tick_params(axis="y", labelcolor="#1f77b4")
    ax2 = axa.twinx()
    ax2.plot(xs, dirs, "s--", color="#d62728", label="directories")
    ax2.set_ylabel("directories", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    for x, f_, d_ in zip(xs, files, dirs):
        axa.annotate(f"{f_}", (x, f_), textcoords="offset points", xytext=(0, 7),
                     ha="center", fontsize=8, color="#1f77b4")
        ax2.annotate(f"{d_}", (x, d_), textcoords="offset points", xytext=(0, -13),
                     ha="center", fontsize=8, color="#d62728")
    axa.set_title("(d) measured tree size after each run")
    axa.grid(alpha=0.25, lw=0.5)

labels = {"A": "A: caches, vendored,\nduplicates, provenance",
          "B": "B: run logs, manifests,\nsource snapshots",
          "C": "C: unreferenced\nfigure orphans"}
if batches:
    keys = [k for k in ("A", "B", "C") if k in batches]
    axb.bar([labels[k] for k in keys], [batches[k] for k in keys],
            color=["#4c78a8", "#f58518", "#54a24b"])
    for i, k in enumerate(keys):
        axb.annotate(f"{batches[k]} files\n{batch_bytes[k]/2**20:.1f} MB",
                     (i, batches[k]), textcoords="offset points", xytext=(0, 3),
                     ha="center", fontsize=8)
    axb.set_ylabel("files archived + removed")
    axb.set_title("(e) removed per batch (all recoverable)")
    axb.tick_params(axis="x", labelsize=7.5)
    axb.grid(alpha=0.25, lw=0.5, axis="y")

fig2.tight_layout()
p2 = os.path.join(OUT, "fig2_consolidation.png")
fig2.savefig(p2, dpi=160)
plt.close(fig2)

print("wrote", p1)
print("wrote", p2)
for name in ("uniform_A", "one_B_3", "one_B_2", "diagonal_B_3"):
    pts = rows.get(name)
    if not pts:
        continue
    fit = corner_fit([(p[0], p[4]) for p in pts])
    if fit:
        print(f"  {name:<15} c_D = {fit[0][0]:.5f}  rms = {fit[1]:.2e}  n = {len(fit[2])}")
