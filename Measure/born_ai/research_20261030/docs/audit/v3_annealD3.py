import sys
sys.dont_write_bytecode = True
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
exec(open(AUD + "/v3_AL.py").read().split("out = {}")[0].split("from v3_common import *")[1])
pts = load_all()
G0s = {}
rows = []
for (L, b), e in sorted(pts.items()):
    if L not in G0s: G0s[L] = ground_G(L)[0]
    G0 = G0s[L]; n = G0.shape[0]; m = masks(L)
    out = {}
    for name, kap in (("sech2", 1 / np.cosh(b) ** 2),):
        Gb = 0.5 * np.eye(n) + kap * (G0 - 0.5 * np.eye(n))
        S = {k: renyi2(Gb[np.ix_(m[k], m[k])]) for k in "ABC"}
        Dann = (2 * S["A"] + S["C"]) / 3 - S["B"]
        rows.append((L, b, e["z"][-1], kap, Dann, e["mean"], Dann / e["mean"]))
rows.sort(key=lambda r: r[6])
print("kappa = sech^2(beta):  min ratio %.2f  (L=%d beta=%.4f) ; max ratio %.2f (L=%d beta=%.4f)" % (rows[0][6], rows[0][0], rows[0][1], rows[-1][6], rows[-1][0], rows[-1][1]))
# beta=3 subset
sub = [r for r in rows if abs(r[1] - 3.0) < 2e-3]
print("beta=3 only (exact kappa not used; sech2):", [(r[0], round(r[6], 1)) for r in sorted(sub)])
# Look for 14.7 and 3373 among any ratios
for r in rows:
    if abs(r[6] - 14.7) < 0.3 or abs(r[6] - 3373) < 40: print("  match?", r)
