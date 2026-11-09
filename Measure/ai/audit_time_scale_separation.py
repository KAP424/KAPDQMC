"""Independent finite-time scale-separation diagnostics; no equilibrium target.

Only reads existing observations and writes analysis_time_scale/. No simulation,
selection of a preferred coefficient, or modification of the primary analysis.
"""
from pathlib import Path
import csv
import hashlib
import json
import math
from collections import defaultdict
from datetime import datetime, timezone

import numpy as np

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "analysis_time_scale"
SOURCES = [ROOT / "analysis_ref_calibration/calibration_points.csv",
           ROOT / "data/random_half3_beta1_observations.csv"]
WINDOWS = [(2, 4), (2, 5), (2, 6), (3, 6), (3, 8), (4, 8),
           (4, 10), (6, 12), (8, 16), (10, 20), (12, 24), (16, 32)]
PAIRS = [(54, 72), (90, 108), (108, 144), (144, 180), (180, 216)]
MODELS = {"log": (), "log_inv1": (1,), "log_inv2": (2,),
          "log_inv1_inv2": (1, 2), "log_inv1_inv2_inv3": (1, 2, 3)}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def ratio(value, denominator):
    return float(value / denominator) if abs(denominator) >= 1e-10 else None


def fit(times, values, powers):
    if len(times) < 4 + len(powers):
        return None
    x = np.asarray(times, dtype=float)
    y = np.asarray(values, dtype=float)
    design = np.column_stack([np.log(x), np.ones(len(x))] + [x**(-p) for p in powers])
    scales = np.linalg.norm(design, axis=0)
    normalized = design / scales
    solved, _, rank, singular = np.linalg.lstsq(normalized, y, rcond=None)
    if rank != design.shape[1]:
        return None
    coefficients = solved / scales
    residual = y - design @ coefficients
    log_span = abs(coefficients[0]) * math.log(max(x) / min(x))
    correction_span = sum(abs(coefficients[j+2]) * (min(x)**(-p)-max(x)**(-p))
                          for j, p in enumerate(powers))
    return dict(c=float(coefficients[0]), b=float(coefficients[1]),
                **{f"inv{p}": float(coefficients[j+2]) for j, p in enumerate(powers)},
                n=len(x), parameters=design.shape[1], residual_dof=len(x)-design.shape[1],
                rmse=float(np.sqrt(np.mean(residual**2))),
                condition=float(singular[0]/singular[-1]),
                correction_span_over_log_span=ratio(correction_span, log_span))


def main():
    observed = {}
    provenance = []
    duplicate_max = 0.
    for path in SOURCES:
        rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
        provenance.append(dict(path=str(path.relative_to(ROOT)), sha256=sha(path), rows=len(rows)))
        for row in rows:
            theta = float(row["theta"])
            beta = float(row["beta"])
            if not math.isfinite(theta) or beta not in (0., .5, 1., 2.):
                continue
            key = row["pattern"], beta, int(row["L"]), theta
            value = float(row["D"])
            if key in observed:
                difference = abs(observed[key]-value)
                duplicate_max = max(duplicate_max, difference)
                assert difference < 1e-10, key
            else:
                observed[key] = value
    series = defaultdict(dict)
    for (pattern, beta, size, time), value in observed.items():
        series[pattern, beta, size][time] = value
    coverage = []
    for (pattern, beta, size), data in sorted(series.items()):
        lo, hi = min(data), max(data)
        coverage.append(dict(pattern=pattern, beta=beta, L=size, n=len(data),
                             theta_min=lo, theta_max=hi, theta_over_L_min=lo/size,
                             theta_over_L_max=hi/size,
                             theta_over_region_width_max=3*hi/size,
                             grid=",".join(f"{t:g}" for t in sorted(data))))

    comparisons, unavailable = [], []
    groups = sorted({(p, b) for p, b, _ in series})
    for pattern, beta in groups:
        for small, large in PAIRS:
            a, b = series.get((pattern, beta, small), {}), series.get((pattern, beta, large), {})
            if not a or not b:
                continue
            for lo, hi in WINDOWS:
                common = sorted(t for t in a.keys() & b.keys() if lo <= t <= hi)
                meta = dict(pattern=pattern, beta=beta, L_small=small, L_large=large,
                            window=f"{lo}-{hi}", theta_min=lo, theta_max=hi,
                            theta_over_L_small_min=lo/small, theta_over_L_small_max=hi/small,
                            theta_over_L_large_min=lo/large, theta_over_L_large_max=hi/large,
                            theta_over_region_width_large_max=3*hi/large,
                            grid=",".join(f"{t:g}" for t in common))
                if not common or common[0] != lo or common[-1] != hi:
                    unavailable.append(dict(**meta, reason="both_measured_endpoints_required"))
                    continue
                fits = {}
                for model, powers in MODELS.items():
                    fa, fb = fit(common, [a[t] for t in common], powers), fit(common, [b[t] for t in common], powers)
                    if fa is None or fb is None:
                        unavailable.append(dict(**meta, model=model, reason="at_least_parameters_plus_two_points_required"))
                        continue
                    fits[model] = fb
                    denominator = abs(fb["c"])
                    curve_diff = max(abs(a[t]-b[t]) for t in common)
                    comparisons.append(dict(**meta, model=model, **fb, c_small=fa["c"],
                                            size_coefficient_drift=ratio(abs(fa["c"]-fb["c"]), denominator),
                                            size_sign_change=fa["c"]*fb["c"] <= 0,
                                            max_abs_D_difference=curve_diff,
                                            D_difference_over_log_span=ratio(curve_diff, denominator*math.log(hi/lo))))
                for row in comparisons:
                    if all(row[k] == meta[k] for k in ("pattern", "beta", "L_small", "L_large", "window")):
                        base = fits.get("log_inv1")
                        if base:
                            row["difference_from_three_parameter"] = ratio(abs(row["c"]-base["c"]), abs(base["c"]))

    indexed = {(r["pattern"],r["beta"],r["L_small"],r["L_large"],r["window"],r["model"]): r for r in comparisons}
    summary = []
    for pattern, beta in groups:
        primary = indexed.get((pattern,beta,90,108,"2-6","log_inv1"))
        if not primary:
            continue
        row = {k:primary[k] for k in ("pattern","beta","c","size_coefficient_drift",
                                     "theta_over_L_large_min","theta_over_L_large_max",
                                     "correction_span_over_log_span","D_difference_over_log_span")}
        row["interpretation"] = "finite_window_diagnostic_not_certified_scale_separation"
        for label, window, model in (("upper_cutoff","2-5","log_inv1"),
                                      ("lower_cutoff","3-6","log_inv1"),
                                      ("four_parameter","2-6","log_inv1_inv2"),
                                      ("later_window","4-8","log_inv1")):
            other = indexed.get((pattern,beta,90,108,window,model))
            row[label+"_drift"] = ratio(abs(other["c"]-primary["c"]),abs(primary["c"])) if other else None
        for window in ("4-10","6-12"):
            later = indexed.get((pattern,beta,108,144,window,"log_inv1"))
            four = indexed.get((pattern,beta,108,144,window,"log_inv1_inv2"))
            if later:
                prefix = "late_"+window.replace("-","_")
                row[prefix+"_c144"] = later["c"]
                row[prefix+"_size_drift"] = later["size_coefficient_drift"]
                row[prefix+"_four_parameter_drift"] = ratio(abs(four["c"]-later["c"]),abs(later["c"])) if four else None
        summary.append(row)

    OUT.mkdir(exist_ok=True)
    write_csv("time_coverage.csv", coverage)
    write_csv("common_grid_diagnostics.csv", comparisons)
    write_csv("unavailable_windows.csv", unavailable)
    write_csv("all_patterns_summary.csv", summary)
    receipt = dict(generated_utc=datetime.now(timezone.utc).isoformat(), script_sha256=sha(Path(__file__)),
                   sources=provenance, finite_observations=len(observed), pattern_beta_groups=len(groups),
                   duplicate_max_abs_difference=duplicate_max, fit_comparisons=len(comparisons),
                   windows=WINDOWS, models=MODELS, source_files_modified=False,
                   classification="No asymptotic pass list is constructed; no equilibrium coefficient is read or used.",
                   geometric_ratio="theta/L is only a dimensional screening variable; region width is L/3; velocity factors are not fitted.",
                   denominator="Each size drift uses that window/model's own absolute large-size time coefficient; floor 1e-10 gives undefined, never epsilon-regularized.")
    (OUT/"summary.json").write_text(json.dumps(receipt,indent=2),encoding="utf-8")
    print(json.dumps({k:receipt[k] for k in ("finite_observations","pattern_beta_groups","fit_comparisons")},indent=2))
    print("beta=0: primary size %, four-parameter %, later-window %, late6-12 size %")
    for row in summary:
        if row["beta"] == 0:
            print(row["pattern"],*[round(100*row[k],3) if row.get(k) is not None else None for k in
                  ("size_coefficient_drift","four_parameter_drift","later_window_drift","late_6_12_size_drift")])


if __name__ == "__main__":
    main()
