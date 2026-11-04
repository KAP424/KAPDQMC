import sys
sys.path.insert(0, "d:/JuliaDQMC/code/KAPDQMC/src/Measure/born_ai/research_20261030/docs/audit")
from v3_common import *
base = baseline()
pts = load_all()
rows = []
for (L, b), e in sorted(pts.items(), key=lambda kv: (kv[0][0], kv[1]["z"][-1])):
    z = e["z"][-1]
    rows.append((L, b, z, e["n"], e["mean"], e["sem"], base[str(L)]["D"], e["mean"] / base[str(L)]["D"]))
with open(AUD + "/v3_all_points.txt", "w") as fh:
    fh.write("L beta z n Dbar sem Dgs h\n")
    for r in rows:
        fh.write("%d %.4f %.4f %d %.6f %.6f %.6f %.4f\n" % r)
for r in sorted(rows, key=lambda r: (r[2], r[0])):
    print("L=%2d beta=%.4f z=%9.4f n=%5d D=%.5f +- %.5f  Dgs=%.4f h=%.3f" % r)
