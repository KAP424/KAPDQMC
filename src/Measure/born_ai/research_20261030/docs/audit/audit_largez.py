import os, json, glob
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
base = json.load(open(os.path.join(ROOT, "baseline_D.json")))
ks = np.array(sorted(int(k) for k in base))
vs = np.array([base[str(k)]["D"] for k in ks])

rows = []
for d in ("data_matched_z", "data_matched_largez"):
    for f in sorted(glob.glob(os.path.join(ROOT, d, "*.csv"))):
        a = np.genfromtxt(f, delimiter=",", names=True)
        if a.size == 0:
            continue
        D = a["D"]; L = int(a["L"][0])
        rows.append(dict(L=L, beta=float(a["beta"][0]), z=float(a["z"][0]),
                         mean=float(D.mean()),
                         sem=float(D.std(ddof=1)/np.sqrt(D.size)),
                         Dgs=float(np.interp(np.log(L), np.log(ks), vs)),
                         src=d + "/" + os.path.basename(f)))

print("=" * 78)
print("CLAIM 2.2(ii): 'z >~ 40 => D/D_gs = 1.01-1.06'  and '1-3% positive bias'")
print("=" * 78)
sel = sorted([r for r in rows if r["z"] >= 40], key=lambda r: (r["z"], r["L"]))
print(" %3s %10s %9s %8s %8s %8s" % ("L", "z", "D", "sem", "h", "h-1(sig)"))
hs = []
for r in sel:
    h = r["mean"] / r["Dgs"]
    hs.append(h)
    print(" %3d %10.3f %9.5f %8.5f %8.4f %8.1f%%  (%5.2f sigma from 1)  %s"
          % (r["L"], r["z"], r["mean"], r["sem"], h, 100 * (h - 1),
             100 * (h - 1) / (100 * r["sem"] / r["Dgs"]), r["src"]))
print("\n  h range over z>=40: %.4f .. %.4f   (h-1 = %+.1f%% .. %+.1f%%)"
      % (min(hs), max(hs), 100 * (min(hs) - 1), 100 * (max(hs) - 1)))
print("  SUMMARY says 1.01-1.06 (and elsewhere '1-3% positive bias')")

print()
print("=" * 78)
print("CLAIM: 'z >= 40 proportional to 1' -- how many rows are below 1.00?")
print("=" * 78)
below = [r for r in sel if r["mean"] / r["Dgs"] < 1.0]
print("  rows with h<1:", [(r["L"], round(r["z"], 1), round(r["mean"] / r["Dgs"], 4)) for r in below])
print("  rows with |h-1| > 0.03:",
      [(r["L"], round(r["z"], 1), round(r["mean"] / r["Dgs"], 4)) for r in sel
       if abs(r["mean"] / r["Dgs"] - 1) > 0.03])

print()
print("=" * 78)
print("ITEM 10 detail: the L=18 'sequence' mixes two files at the SAME z")
print("=" * 78)
for r in rows:
    if r["L"] == 18 and abs(r["z"] - 43.85) < 0.01:
        print("  %s  z=%.4f  D=%.5f+-%.5f  h=%.4f" %
              (r["src"], r["z"], r["mean"], r["sem"], r["mean"] / r["Dgs"]))
for r in rows:
    if r["L"] == 18 and r["z"] > 100:
        print("  %s  z=%.4f  D=%.5f+-%.5f  h=%.4f" %
              (r["src"], r["z"], r["mean"], r["sem"], r["mean"] / r["Dgs"]))

# significance of the claimed dip at L=18
m1 = [r for r in rows if r["L"] == 18 and abs(r["z"] - 43.8486) < 0.01][0]
m2 = [r for r in rows if r["L"] == 18 and abs(r["z"] - 43.85) < 0.001 and "largez" in r["src"]][0]
m3 = [r for r in rows if r["L"] == 18 and r["z"] > 100][0]
print("\n  L=18: h(43.85,largez)=%.4f  ->  h(119.19)=%.4f   drop = %.4f +- %.4f = %.2f sigma"
      % (m2["mean"] / m2["Dgs"], m3["mean"] / m3["Dgs"],
         (m2["mean"] / m2["Dgs"] - m3["mean"] / m3["Dgs"]),
         np.hypot(m2["sem"], m3["sem"]) / m2["Dgs"],
         (m2["mean"] / m2["Dgs"] - m3["mean"] / m3["Dgs"]) / (np.hypot(m2["sem"], m3["sem"]) / m2["Dgs"])))
print("  L=18: h(43.85,matched)=%.4f -> h(119.19)=%.4f   drop = %.4f +- %.4f = %.2f sigma"
      % (m1["mean"] / m1["Dgs"], m3["mean"] / m3["Dgs"],
         (m1["mean"] / m1["Dgs"] - m3["mean"] / m3["Dgs"]),
         np.hypot(m1["sem"], m3["sem"]) / m1["Dgs"],
         (m1["mean"] / m1["Dgs"] - m3["mean"] / m3["Dgs"]) / (np.hypot(m1["sem"], m3["sem"]) / m1["Dgs"])))
