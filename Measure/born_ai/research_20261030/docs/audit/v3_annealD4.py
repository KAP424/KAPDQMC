import sys, csv
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
exec(open(AUD + "/v3_AL.py").read().split("out = {}")[0].split("from v3_common import *")[1])
exec(open(AUD + "/v3_annealD.py").read().split("G0s = {}")[0].split("pts = load_all()")[1])   # kappa_a
rows = list(csv.DictReader(open("d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/analysis/born_scaling/production/means.csv")))
G0s = {}
res = []
for r in rows:
    L = int(r["L"]); b = float(r["beta"])
    if b == 0.0: continue
    if L not in G0s: G0s[L] = ground_G(L)[0]
    G0 = G0s[L]; n = G0.shape[0]; m = masks(L)
    for name, kap in (("exact", kappa_a(L * L, b)), ("sech2", 1 / np.cosh(b) ** 2)):
        Gb = 0.5 * np.eye(n) + kap * (G0 - 0.5 * np.eye(n))
        S = {k: renyi2(Gb[np.ix_(m[k], m[k])]) for k in "ABC"}
        Dann = (2 * S["A"] + S["C"]) / 3 - S["B"]
        res.append((name, L, b, kap, Dann, float(r["D_mean"]), Dann / float(r["D_mean"])))
for name in ("exact", "sech2"):
    sub = sorted([x for x in res if x[0] == name], key=lambda x: x[6])
    print(name, "min %.2f (L=%d,beta=%.2f)  max %.2f (L=%d,beta=%.2f)  n=%d" % (sub[0][6], sub[0][1], sub[0][2], sub[-1][6], sub[-1][1], sub[-1][2], len(sub)))
    print("   lowest 5:", [(x[1], x[2], round(x[6], 2)) for x in sub[:5]])
    print("   highest 5:", [(x[1], x[2], round(x[6], 1)) for x in sub[-5:]])
