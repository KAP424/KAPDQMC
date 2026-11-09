import sys, json
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
pts = load_all()
sel = []
for (L, b), e in sorted(pts.items()):
    if abs(b - 3.0) < 2e-3:
        sel.append((L, e))
print("beta=3 points: L =", [L for L, e in sel], "z =", [e["z"][-1] for L, e in sel])
Ls = [L for L, e in sel]
D = {L: e["mean"] for L, e in sel}
DV = {L: e["DV"].mean() for L, e in sel}
print("D:", {L: round(v, 5) for L, v in D.items()})
print("DV:", {L: round(v, 5) for L, v in DV.items()})
print("interval   dlnD/dlnL   dlnDV/dlnL")
for a, b in ((6, 12), (12, 18), (18, 24), (24, 36)):
    print("  %d->%d   %+.2f   %+.2f" % (a, b, np.log(D[b] / D[a]) / np.log(b / a), np.log(DV[b] / DV[a]) / np.log(b / a)))
# D/DV ratios
print("\nD/DV ratio per point (pooled means):")
for (L, b), e in sorted(pts.items(), key=lambda kv: kv[1]["z"][-1]):
    z = e["z"][-1]
    if abs(z - 0.089) < 1e-3 or abs(z - 0.357) < 1e-3 or abs(z - 0.803) < 2e-3 or abs(z - 3.213) < 2e-3 or abs(z-1.428) < 2e-3:
        print("  z=%.3f L=%d  D/DV(mean ratio)=%.3f  ; mean(D/DV) per sample not used" % (z, L, e["mean"] / e["DV"].mean()))
# pooled over L at fixed z
for zt in (0.089, 0.357, 0.803, 1.428, 3.213):
    num = den = 0.0
    r = []
    for (L, b), e in pts.items():
        if abs(e["z"][-1] - zt) / zt < 3e-3 and L <= 36:
            r.append((L, e["mean"], e["DV"].mean()))
    print("  z=%.3f: ratios by L:" % zt, " ".join("L%d:%.2f" % (L, d / v) for L, d, v in sorted(r)), " | sum D/sum DV = %.3f" % (sum(d for _, d, _ in r) / sum(v for _, _, v in r)))
# ground state D/D_V0
AL = json.load(open(AUD + "/v3_AL.json"))
print("\nground state D_gs/D_V0:")
base = baseline()
for L in ("6", "12", "18", "24", "30", "36"):
    print("  L=%s: D_gs=%.5f D_V0=%.5f ratio=%.3f" % (L, base[L]["D"], AL[L]["DV0"], base[L]["D"] / AL[L]["DV0"]))
