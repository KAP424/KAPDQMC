import csv
rows = list(csv.DictReader(open("d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/tables/mult_scaling_scan.csv")))
print("rows", len(rows))
nb = [int(r["n_buckets"]) for r in rows]
ch = [float(r["median_chi2_per_dof"]) for r in rows]
import collections
print("n_buckets counts", sorted(collections.Counter(nb).items()))
valid = [(r, float(r["median_chi2_per_dof"])) for r in rows if int(r["n_buckets"]) >= 2]
print("rows with n_buckets>=2:", len(valid), " min chi2/dof among them:", min(c for r, c in valid))
valid3 = [(r, float(r["median_chi2_per_dof"])) for r in rows if int(r["n_buckets"]) >= 3]
print("rows with n_buckets>=3:", len(valid3), " min:", min(c for r, c in valid3))
print("overall min chi2/dof:", min(ch))
best = sorted(valid, key=lambda t: t[1])[:5]
for r, c in best: print("  ", r)
import numpy as np
al = sorted({float(r["alpha"]) for r in rows}); ga = sorted({float(r["gamma"]) for r in rows})
print("alpha values", al); print("gamma values", ga, "grid size", len(al)*len(ga))
