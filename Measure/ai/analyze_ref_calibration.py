"""Ref-calibration sensitivity analysis; no new states and no target-selected fits.

The largest measured size L108 is the main time estimate. An arithmetic L90/108
mean is a separately labelled Ref-style comparison, not a thermodynamic limit.
All time models use a fixed, common measured grid. The protocol records that
these rules were revised after seeing prior and some new observations.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "analysis_ref_calibration"
PATTERNS = ["uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3",
            "diagonal_B_3", "special_matched_a", "special_matched_b",
            "full_empty_cluster3", "random_half_3", "general_matched_a",
            "general_matched_b"]
SOURCES = [HERE / "data/ground_observations_baseline.csv",
           HERE / "data/ref_calibration_observations.csv",
           HERE / "data/ref_calibration_equilibrium.csv",
           HERE / "data/ref_calibration_late.csv"]
MODELS = {"log": (False, False), "log_plus_inv1": (True, False),
          "log_plus_inv2": (False, True), "log_plus_inv1_inv2": (True, True)}
TIMES = [2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8]
WINDOWS = {"2-4": [2, 4], "2-5": [2, 5], "2-6": [2, 6],
           "3-6": [3, 6], "4-8": [4, 8]}
EQ_WINDOWS = {"18-90": [18, 90], "36-90": [36, 90],
              "18-144": [18, 144], "36-144": [36, 144]}
LATE_PATTERNS = ["uniform_A", "one_B_2", "diagonal_B_3", "special_matched_a",
                 "special_matched_b", "full_empty_cluster3", "random_half_3",
                 "general_matched_a", "general_matched_b"]
LATE_WINDOWS = {"4-10": list(range(4, 11)), "6-12": list(range(6, 13))}
ORDER_WINDOWS = {"3-8": [3, 8], "4-8": [4, 8]}
ORDER_MODELS = ("log_plus_inv1_inv2", "log_plus_inv1_inv2_inv3")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


def number(value):
    return float(value) if value not in (None, "") else None


def relative(numerator, denominator, floor=1e-10):
    if denominator is None or abs(denominator) < floor:
        return None
    return float(numerator / abs(denominator))


def write_csv(path, rows, empty_fields=()):
    fields = list(dict.fromkeys(k for row in rows for k in row)) or list(empty_fields)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def init_protocol(path):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    snapshot = []
    for source in SOURCES:
        if source.exists():
            data = source.read_bytes()
            snapshot.append(dict(path=str(source.relative_to(HERE)), sha256=sha(data),
                                 bytes=len(data), newline_count=data.count(b"\n")))
        else:
            snapshot.append(dict(path=str(source.relative_to(HERE)), missing=True))
    protocol = dict(
        schema_version=1, created_utc=utc(),
        status="Method revision after review of prior data and some new data; not blind preregistration.",
        pattern_order=PATTERNS, supported_betas=[0, 0.5, 1, 2],
        source_snapshot_at_rule_recording=snapshot,
        time_sizes=[90, 108], time_grid=TIMES, time_windows=WINDOWS,
        primary_time_window="2-6", primary_time_model="log_plus_inv1",
        primary_size=108, comparison_size=90,
        size_average="Arithmetic mean of measured D90 and D108 at each theta; Ref-style diagnostic only. Do not call it a thermodynamic plateau without size convergence.",
        models={"log": "c*ln(x)+b", "log_plus_inv1": "c*ln(x)+b+d/x",
                "log_plus_inv2": "c*ln(x)+b+e/x^2",
                "log_plus_inv1_inv2": "c*ln(x)+b+d/x+e/x^2"},
        minimum_points="parameter_count + 2", interpolation=False,
        equilibrium_windows=EQ_WINDOWS, equilibrium_primary="18-90/log_plus_inv1",
        equilibrium_extended_main="18-144/log_plus_inv1",
        equilibrium_extension_required_sizes=[108, 126, 144],
        late_beta0_diagnostic=dict(patterns=LATE_PATTERNS, sizes=[108, 144], windows=LATE_WINDOWS,
            added_grid={"108": [9, 10, 11, 12], "144": list(range(4, 13))},
            reason="Eight patterns with early-window time diagnostics failing, plus uniform_A control; expansion chosen using size/window/model sensitivity, not proximity to equilibrium.",
            relative_denominator="abs(c at L144 in the same late window, log_plus_inv1)",
            role="Separate diagnostic only; does not replace the common L108 theta2-6 main series or its stability label."),
        diagnostics=dict(
            denominator="abs(c at L108, theta 2-6, log_plus_inv1) for every stability metric",
            denominator_floor=1e-10,
            size_coefficient_difference_max=0.02,
            upper_cut_window="2-5", upper_cut_relative_difference_max=0.05,
            lower_cut_window="3-6", lower_cut_relative_difference_max=0.05,
            four_parameter_relative_difference_max=0.05,
            sign_rule="Any zero or sign change versus the main coefficient among L90 main, L108 2-5, L108 3-6 or L108 four-parameter 2-6 fails the stability diagnostic.",
            D_size_comparison="Report unshifted pointwise absolute/relative D90-D108 and maximum differences; no new D threshold or automatic plateau declaration.",
            meaning="Current finite-window coefficient stability diagnostic, not an equivalence theorem, asymptotic proof, confidence interval, or pure-corner classification.",
            equilibrium_target_used=False),
        reporting=["Retain all patterns and failed or missing diagnostics.",
                   "Compare pure log and corrected estimators on exactly the same observations.",
                   "No error bars: these are deterministic calculations, not statistical replicates.",
                   "Equilibrium coefficients are a post-analysis comparison, never a target for window selection.",
                   "A small raw D difference need not imply a small fitted coefficient difference.",
                   "Whole-source hashes identify each executed snapshot; final coverage applies only to explicitly required beta values."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return protocol


def add_selection_protocol(protocol, path):
    """Append the explicitly requested posthoc rule without rewriting its history."""
    if "controlled_window_selection" in protocol:
        return protocol
    snapshot=[]
    for source in SOURCES:
        if source.exists():
            data=source.read_bytes()
            snapshot.append(dict(path=str(source.relative_to(HERE)), sha256=sha(data), bytes=len(data)))
        else:
            snapshot.append(dict(path=str(source.relative_to(HERE)), missing=True))
    protocol["controlled_window_selection"]=dict(
        added_utc=utc(), status="Posthoc method revision after viewing early-window results; not independent blind validation.",
        sources_at_revision=snapshot, candidate_windows=list(WINDOWS),
        primary_model="log_plus_inv1", sizes=[90,108],
        denominator="abs(c108 of the individual candidate window under log_plus_inv1)",
        denominator_floor=1e-10, size_relative_difference_max=.02,
        nested_four_parameter_relative_difference_max=.05,
        delete_first_relative_difference_max=.05, delete_last_relative_difference_max=.05,
        trim_rule="At L108 delete exactly the first or the last measured canonical-grid point and refit log_plus_inv1; each trimmed fit must have at least five points.",
        nested_model_rule="The four-parameter log_plus_inv1_inv2 fit must have at least six points.",
        sign_rule="c90, c108 four-parameter, and both L108 trimmed fits must have the same nonzero sign as c108 main.",
        selection_order="Among qualified candidates choose largest theta_min, then largest theta_max. No equilibrium coefficient is used.",
        late_windows_included=False, empty_result="No stable candidate; never fill from a failed fit.",
        interpretation="Finite-scale controlled candidate, not an asymptotic coefficient proof or statistical confidence interval.")
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return protocol


def add_order_protocol(protocol, path):
    if "model_order_diagnostic" in protocol:
        return protocol
    protocol["controlled_window_selection"]["interpretation"] = "Local-window diagnostic only; not a final reliable corner-coefficient list. Endpoint deletion and local model agreement do not establish an asymptotic regime. Not a statistical confidence interval."
    protocol["model_order_diagnostic"] = dict(added_utc=utc(),
        status="Additional posthoc diagnostic requested after examining locally stable but drifting corrected estimates.",
        patterns=PATTERNS, betas=[0,.5,1,2], sizes=[90,108], windows=ORDER_WINDOWS,
        models={"log_plus_inv1_inv2":"c*ln(theta)+b+d/theta+e/theta^2",
                "log_plus_inv1_inv2_inv3":"c*ln(theta)+b+d/theta+e/theta^2+f/theta^3"},
        minimum_points="parameter_count + 2; five-parameter model requires at least seven observations",
        size_drift_denominator="abs(c108 in the same window and same model)",
        window_drift_denominator="abs(c108 in window4-8 for the same model)",
        order_drift_denominator="abs(c108 in the same window under the four-parameter model)",
        denominator_floor=1e-10,
        interpretation="Report every combination, including strong size pollution, sign changes and poor conditioning. No selection or qualification is based on this diagnostic.")
    path.write_text(json.dumps(protocol,indent=2,ensure_ascii=False)+"\n",encoding="utf-8")
    return protocol


def read_sources(paths, betas, stage):
    seen, receipts, duplicates = {}, [], []
    identity_error = 0.0
    for path in paths:
        if not path.exists():
            receipts.append(dict(path=str(path.relative_to(HERE)), missing=True))
            continue
        data = path.read_bytes()
        snapshot = data
        truncated = bool(data and not data.endswith(b"\n"))
        if truncated:
            if stage == "final":
                raise ValueError(f"Incomplete trailing row: {path}")
            data = data[:data.rfind(b"\n") + 1]
        raw_rows = list(csv.DictReader(io.StringIO(data.decode("utf-8-sig"))))
        selected = 0
        for index, raw in enumerate(raw_rows, 2):
            pattern, beta = raw["pattern"], float(raw["beta"])
            if pattern not in PATTERNS or beta not in betas:
                continue
            theta, size = float(raw["theta"]), int(raw["L"])
            if math.isinf(theta):
                if not 18 <= size <= 144:
                    continue
            elif not 0 < theta <= 16:
                continue
            selected += 1
            ground = str(raw.get("premeasurement_ground", raw.get("ground", "true"))).lower()
            excess = float(raw.get("premeasurement_excess") or raw.get("premeasurement_limit_excess") or raw.get("accessible_excess") or 0)
            if ground not in ("true", "1") or abs(excess) >= 1e-8:
                raise ValueError(f"Non-ground-reachable input: {path}:{index}")
            row = dict(pattern=pattern, family=raw.get("family", ""), m=int(raw.get("m", 0)),
                       beta=beta, L=size, theta=theta,
                       **{field: float(raw[field]) for field in ("D", "S_A", "S_C", "S_B")},
                       source_file=str(path.relative_to(HERE)), source_row=index,
                       underlying_source_file=raw.get("source_file", ""),
                       premeasurement_excess=excess)
            if not all(math.isfinite(row[field]) for field in ("D", "S_A", "S_C", "S_B")):
                raise ValueError(f"Nonfinite entropy: {path}:{index}")
            err = abs(row["D"] - (2*row["S_A"] + row["S_C"])/3 + row["S_B"])
            identity_error = max(identity_error, err)
            if err > 1e-9:
                raise ValueError(f"D identity failed: {path}:{index}")
            key = (pattern, beta, size, theta)
            if key in seen:
                previous = seen[key]
                error = max(abs(previous[field] - row[field]) for field in ("D", "S_A", "S_C", "S_B"))
                if error > 1e-8:
                    raise ValueError(f"Conflicting observations {key}: {error}")
                duplicates.append(dict(pattern=pattern, beta=beta, L=size, theta=theta,
                    previous_source=previous["source_file"], later_source=row["source_file"],
                    max_absolute_difference=error))
                # Later direct calculations take precedence; duplicates are never replicates.
            seen[key] = row
        receipts.append(dict(path=str(path.relative_to(HERE)), sha256=sha(snapshot),
                             bytes=len(snapshot), parsed_rows=len(raw_rows), selected_rows=selected,
                             trailing_row_trimmed=truncated))
    return seen, receipts, duplicates, identity_error


def fit(x, y, model):
    x, y = np.asarray(x, float), np.asarray(y, float)
    third = model == "log_plus_inv1_inv2_inv3"
    first, second = (True, True) if third else MODELS[model]
    columns, names = [np.log(x), np.ones(len(x))], ["c", "b"]
    if first:
        columns.append(1/x); names.append("d")
    if second:
        columns.append(1/x**2); names.append("e")
    if third:
        columns.append(1/x**3); names.append("f")
    if len(x) < len(columns) + 2:
        return None
    matrix = np.column_stack(columns)
    norm = np.linalg.norm(matrix, axis=0)
    coefficient, _, rank, _ = np.linalg.lstsq(matrix/norm, y, rcond=None)
    if rank != len(columns):
        raise ValueError("Rank-deficient regression")
    coefficient /= norm
    residual = y-matrix@coefficient
    result = dict(model=model, c=None, b=None, d=None, e=None)
    result.update(dict(zip(names, map(float, coefficient))))
    result.update(n=len(x), parameter_count=len(columns),
        grid=",".join(f"{v:g}" for v in x), x_min=float(min(x)), x_max=float(max(x)),
        rmse=float(np.sqrt(np.mean(residual**2))), max_abs_residual=float(max(abs(residual))),
        condition=float(np.linalg.cond(matrix/norm)))
    log_span=abs(result["c"])*math.log(max(x)/min(x))
    result["inverse1_to_log_span"] = relative(abs(result["d"] or 0)*(1/min(x)-1/max(x)), log_span)
    result["inverse2_to_log_span"] = relative(abs(result["e"] or 0)*(1/min(x)**2-1/max(x)**2), log_span)
    if third:
        result["inverse3_to_log_span"] = relative(abs(result["f"])*(1/min(x)**3-1/max(x)**3), log_span)
    return result


def make_analysis(observations, protocol, betas):
    fits, missing, differences, slopes = [], [], [], []
    coverage = []
    index = {}
    for pattern in PATTERNS:
        for beta in betas:
            tables = {size: {r["theta"]: r for r in observations.values()
                            if r["pattern"] == pattern and r["beta"] == beta and r["L"] == size and math.isfinite(r["theta"])}
                      for size in (90, 108)}
            for size in (90, 108):
                absent = [time for time in TIMES if time not in tables[size]]
                coverage.append(dict(domain="time", pattern=pattern, beta=beta, L=size,
                                     required_grid=",".join(map(str, TIMES)), missing_grid=",".join(map(str, absent)),
                                     complete=not absent))
            for time in TIMES:
                if any(time not in tables[size] for size in (90, 108)):
                    continue
                small, large = tables[90][time]["D"], tables[108][time]["D"]
                differences.append(dict(pattern=pattern, beta=beta, theta=time, D_90=small, D_108=large,
                    D_average=(small+large)/2, D_difference=large-small,
                    absolute_D_difference=abs(large-small), relative_D_difference=relative(abs(large-small), large)))
            for label, bounds in WINDOWS.items():
                grid = [time for time in TIMES if bounds[0] <= time <= bounds[1]]
                absent = [(size, time) for size in (90, 108) for time in grid if time not in tables[size]]
                if absent:
                    missing.append(dict(domain="time", pattern=pattern, beta=beta, window=label,
                                        model="all", reason="missing_common_grid", missing_points=json.dumps(absent)))
                    continue
                values = {size: [tables[size][time]["D"] for time in grid] for size in (90, 108)}
                values["average90_108"] = np.mean([values[90], values[108]], axis=0)
                for j in range(len(grid)-1):
                    lo, hi = grid[j:j+2]
                    local = {size: float((values[size][j+1]-values[size][j])/math.log(hi/lo)) for size in values}
                    slopes.append(dict(pattern=pattern, beta=beta, window=label, theta_low=lo, theta_high=hi,
                        theta_mid=math.sqrt(lo*hi), slope_90=local[90], slope_108=local[108],
                        slope_average=local["average90_108"], size_difference=local[108]-local[90]))
                for size, y in values.items():
                    for model in MODELS:
                        result = fit(grid, y, model)
                        if result is None:
                            missing.append(dict(domain="time", pattern=pattern, beta=beta, window=label,
                                                model=model, L=size, reason="n_less_than_parameters_plus_2", missing_points=""))
                            continue
                        row = dict(domain="time", pattern=pattern, beta=beta, L=size, window=label, **result)
                        fits.append(row)
                        index[("time", pattern, beta, size, label, model)] = row
            eq = {r["L"]: r for r in observations.values()
                  if r["pattern"] == pattern and r["beta"] == beta and math.isinf(r["theta"])}
            for size in protocol["equilibrium_extension_required_sizes"]:
                coverage.append(dict(domain="equilibrium_extension", pattern=pattern, beta=beta, L=size,
                                     required_grid=str(size), missing_grid="" if size in eq else str(size), complete=size in eq))
            for label, bounds in EQ_WINDOWS.items():
                grid = sorted(size for size in eq if bounds[0] <= size <= bounds[1])
                if not grid or grid[0] != bounds[0] or grid[-1] != bounds[1]:
                    missing.append(dict(domain="equilibrium", pattern=pattern, beta=beta, window=label,
                                        model="all", reason="missing_range_endpoint", missing_points=""))
                    continue
                if bounds[1] == 144 and any(size not in eq for size in protocol["equilibrium_extension_required_sizes"]):
                    missing.append(dict(domain="equilibrium", pattern=pattern, beta=beta, window=label,
                                        model="all", reason="incomplete_extension_grid", missing_points=""))
                    continue
                for model in MODELS:
                    result = fit(grid, [eq[size]["D"] for size in grid], model)
                    if result is None:
                        continue
                    row = dict(domain="equilibrium", pattern=pattern, beta=beta, L="", window=label, **result)
                    fits.append(row)
                    index[("equilibrium", pattern, beta, "", label, model)] = row
    comparisons, window_diagnostics = [], []
    floor = protocol["diagnostics"]["denominator_floor"]
    def get(pattern, beta, size=108, window="2-6", model="log_plus_inv1", domain="time"):
        return index.get((domain, pattern, beta, size, window, model))
    def coeff(result):
        return result["c"] if result else None
    for pattern in PATTERNS:
        for beta in betas:
            main = get(pattern, beta)
            c = coeff(main)
            denominator = abs(c) if c is not None and abs(c) >= floor else None
            small, average = get(pattern, beta, 90), get(pattern, beta, "average90_108")
            upper, lower = get(pattern, beta, window="2-5"), get(pattern, beta, window="3-6")
            four = get(pattern, beta, model="log_plus_inv1_inv2")
            pure, pure_average = get(pattern, beta, model="log"), get(pattern, beta, "average90_108", model="log")
            square = get(pattern, beta, model="log_plus_inv2")
            eq = get(pattern, beta, "", "18-90", domain="equilibrium")
            extended = get(pattern, beta, "", "18-144", domain="equilibrium")
            def drift(other):
                return relative(abs(other["c"]-c), denominator, floor) if other and denominator is not None else None
            ds, du, dl, df = map(drift, (small, upper, lower, four))
            thresholds = protocol["diagnostics"]
            gates = dict(size_gate=ds is not None and ds <= thresholds["size_coefficient_difference_max"],
                         upper_window_gate=du is not None and du <= thresholds["upper_cut_relative_difference_max"],
                         lower_window_gate=dl is not None and dl <= thresholds["lower_cut_relative_difference_max"],
                         four_parameter_gate=df is not None and df <= thresholds["four_parameter_relative_difference_max"])
            complete = all(result is not None for result in (main, small, upper, lower, four))
            sign = complete and denominator is not None and all(result["c"]*c > 0 for result in (small, upper, lower, four))
            gates["sign_gate"] = bool(sign)
            stable = complete and denominator is not None and all(gates.values())
            reasons = []
            if not complete:
                reasons.append("missing_required_fit_or_common_grid")
            elif denominator is None:
                reasons.append("main_coefficient_below_relative_diagnostic_floor")
            else:
                reasons.extend(name for name, passed in gates.items() if not passed)
            point_diffs = [r for r in differences if r["pattern"] == pattern and r["beta"] == beta and 2 <= r["theta"] <= 6]
            record = dict(pattern=pattern, beta=beta, main_L=108, comparison_L=90,
                primary_window="2-6", primary_model="log_plus_inv1", c_main108=c, c_90=coeff(small),
                c_average=coeff(average), c_purelog108=coeff(pure), c_purelog_average=coeff(pure_average),
                c_inverse_square108=coeff(square), c_four108=coeff(four),
                c_upper_window108=coeff(upper), c_lower_window108=coeff(lower),
                diagnostic_denominator=denominator, size_drift=ds, neighbor_window_drift=du,
                lower_window_drift=dl, four_parameter_drift=df,
                max_absolute_D_difference=max((r["absolute_D_difference"] for r in point_diffs), default=None),
                max_relative_D_difference=max((r["relative_D_difference"] for r in point_diffs if r["relative_D_difference"] is not None), default=None),
                main_common_grid_complete=bool(main), diagnostic_complete=complete,
                **gates, stable_diagnostic=bool(stable), diagnostic_status="stable" if stable else "unstable" if complete else "incomplete",
                missing_reason=";".join(reasons),
                c_eq_old=coeff(eq), c_eq_extended=coeff(extended),
                eq_relative_difference=relative(c-eq["c"], eq["c"], floor) if main and eq else None,
                eq_extended_relative_difference=relative(c-extended["c"], extended["c"], floor) if main and extended else None)
            comparisons.append(record)
            for label in WINDOWS:
                primary = get(pattern, beta, window=label)
                other_size = get(pattern, beta, 90, label)
                if not primary or not other_size:
                    window_diagnostics.append(dict(pattern=pattern, beta=beta, window=label, status="missing_common_grid"))
                    continue
                for model in MODELS:
                    result = get(pattern, beta, window=label, model=model)
                    small_result = get(pattern, beta, 90, label, model)
                    avg_result = get(pattern, beta, "average90_108", label, model)
                    if not result:
                        window_diagnostics.append(dict(pattern=pattern, beta=beta, window=label, model=model,
                                                       status="n_less_than_parameters_plus_2"))
                        continue
                    window_diagnostics.append(dict(pattern=pattern, beta=beta, window=label, model=model, status="available",
                        c_90=other_size["c"] if model == "log_plus_inv1" else small_result["c"],
                        c_108=result["c"], c_average=avg_result["c"], grid=result["grid"], n=result["n"],
                        diagnostic_denominator=denominator,
                        size_drift=relative(abs(result["c"]-small_result["c"]), denominator, floor) if denominator is not None else None,
                        difference_from_same_window_main=relative(abs(result["c"]-primary["c"]), denominator, floor) if denominator is not None else None,
                        difference_from_primary_window_main=relative(abs(result["c"]-c), denominator, floor) if denominator is not None else None,
                        sign_changed_vs_main=(result["c"]*c <= 0) if denominator is not None else None,
                        rmse=result["rmse"], condition=result["condition"]))
    return dict(fits=fits, comparison=comparisons, size_differences=differences,
                window_diagnostics=window_diagnostics, local_slopes=slopes,
                coverage=coverage, missing_fits=missing)


def analyze_late(observations, protocol):
    """Separate beta-zero size/window/model diagnostics; no replacement main fit."""
    settings = protocol["late_beta0_diagnostic"]
    fits, diagnostics, differences, coverage = [], [], [], []
    floor = protocol["diagnostics"]["denominator_floor"]
    for pattern in settings["patterns"]:
        tables = {size: {r["theta"]: r for r in observations.values()
                        if r["pattern"] == pattern and r["beta"] == 0 and r["L"] == size and math.isfinite(r["theta"])}
                  for size in (108, 144)}
        for size in (108, 144):
            required = sorted(set(t for grid in settings["windows"].values() for t in grid))
            absent = [time for time in required if time not in tables[size]]
            coverage.append(dict(domain="time_late", pattern=pattern, beta=0, L=size,
                                 required_grid=",".join(map(str, required)), missing_grid=",".join(map(str, absent)),
                                 complete=not absent))
        window_fits = {}
        for label, grid in settings["windows"].items():
            absent = [(size, time) for size in (108, 144) for time in grid if time not in tables[size]]
            if absent:
                diagnostics.append(dict(pattern=pattern, beta=0, window=label, status="missing_common_grid",
                                        missing_points=json.dumps(absent)))
                continue
            values = {size: [tables[size][time]["D"] for time in grid] for size in (108, 144)}
            values["average108_144"] = np.mean([values[108], values[144]], axis=0)
            for j, time in enumerate(grid):
                small, large = values[108][j], values[144][j]
                differences.append(dict(pattern=pattern, beta=0, window=label, theta=time,
                    D_108=small, D_144=large, D_average=(small+large)/2, D_difference=large-small,
                    absolute_D_difference=abs(large-small), relative_D_difference=relative(abs(large-small), large, floor)))
            for size, y in values.items():
                for model in MODELS:
                    result = fit(grid, y, model)
                    if result is None:
                        raise ValueError("Registered late diagnostic lacks sufficient data")
                    result = dict(domain="time_late", pattern=pattern, beta=0, L=size, window=label, **result)
                    fits.append(result)
                    window_fits[(label, size, model)] = result
            primary = window_fits[(label, 144, "log_plus_inv1")]
            denominator = abs(primary["c"]) if abs(primary["c"]) >= floor else None
            for model in MODELS:
                small = window_fits[(label, 108, model)]
                large = window_fits[(label, 144, model)]
                avg = window_fits[(label, "average108_144", model)]
                selected = [r for r in differences if r["pattern"] == pattern and r["window"] == label]
                diagnostics.append(dict(pattern=pattern, beta=0, window=label, model=model, status="available",
                    L_small=108, L_large=144, grid=large["grid"], n=large["n"],
                    c_108=small["c"], c_144=large["c"], c_average=avg["c"], diagnostic_denominator=denominator,
                    size_drift=relative(abs(large["c"]-small["c"]), denominator, floor) if denominator is not None else None,
                    model_difference=relative(abs(large["c"]-primary["c"]), denominator, floor) if denominator is not None else None,
                    sign_changed_vs_same_window_main=large["c"]*primary["c"] <= 0,
                    max_absolute_D_difference=max(r["absolute_D_difference"] for r in selected),
                    max_relative_D_difference=max((r["relative_D_difference"] for r in selected if r["relative_D_difference"] is not None), default=None),
                    rmse=large["rmse"], condition=large["condition"],
                    inverse1_to_log_span=large["inverse1_to_log_span"], inverse2_to_log_span=large["inverse2_to_log_span"],
                    classification="diagnostic_only_no_replacement_of_main_label"))
        first = window_fits.get(("4-10", 144, "log_plus_inv1"))
        second = window_fits.get(("6-12", 144, "log_plus_inv1"))
        for row in diagnostics:
            if row["pattern"] == pattern and row["status"] == "available":
                row["c_144_window_difference_4_10_vs_6_12"] = first["c"]-second["c"] if first and second else None
                row["window_difference_relative_to_6_12"] = relative(abs(first["c"]-second["c"]), second["c"], floor) if first and second else None
    return dict(fits=fits, late_diagnostics=diagnostics, late_size_differences=differences, coverage=coverage)


def controlled_selection(observations, analysis, protocol, betas):
    """Fixed-size, target-independent posthoc selection from the five early windows."""
    rules=protocol["controlled_window_selection"]
    floor=rules["denominator_floor"]
    index={(r["domain"],r["pattern"],r["beta"],r["L"],r["window"],r["model"]):r for r in analysis["fits"]}
    candidates, selected, trim_fits=[],[],[]
    for pattern in PATTERNS:
        for beta in betas:
            table={r["theta"]:r for r in observations.values() if r["pattern"]==pattern and r["beta"]==beta and r["L"]==108 and math.isfinite(r["theta"])}
            group=[]
            for window in rules["candidate_windows"]:
                lo,hi=WINDOWS[window]
                grid=[time for time in TIMES if lo<=time<=hi]
                primary=index.get(("time",pattern,beta,108,window,"log_plus_inv1"))
                small=index.get(("time",pattern,beta,90,window,"log_plus_inv1"))
                four=index.get(("time",pattern,beta,108,window,"log_plus_inv1_inv2"))
                trims={}
                if primary:
                    for label,trim_grid in (("delete_first",grid[1:]),("delete_last",grid[:-1])):
                        result=fit(trim_grid,[table[time]["D"] for time in trim_grid],"log_plus_inv1")
                        if result:
                            result=dict(domain="time_trim",pattern=pattern,beta=beta,L=108,
                                        window=window,trim=label,**result)
                            trim_fits.append(result)
                            trims[label]=result
                c=primary["c"] if primary else None
                denominator=abs(c) if c is not None and abs(c)>=floor else None
                def drift(other):
                    return relative(abs(other["c"]-c),denominator,floor) if other and denominator is not None else None
                ds,df,dl,dr=(drift(other) for other in (small,four,trims.get("delete_first"),trims.get("delete_last")))
                gates=dict(
                    size_gate=ds is not None and ds<=rules["size_relative_difference_max"],
                    four_parameter_gate=df is not None and df<=rules["nested_four_parameter_relative_difference_max"],
                    delete_first_gate=dl is not None and dl<=rules["delete_first_relative_difference_max"],
                    delete_last_gate=dr is not None and dr<=rules["delete_last_relative_difference_max"])
                complete=all(value is not None for value in (primary,small,four,trims.get("delete_first"),trims.get("delete_last")))
                gates["sign_gate"]=bool(complete and denominator is not None and all(other["c"]*c>0 for other in (small,four,trims["delete_first"],trims["delete_last"])))
                qualified=complete and denominator is not None and all(gates.values())
                reasons=[]
                if not primary or not small:
                    reasons.append("missing_common_grid")
                else:
                    if denominator is None:
                        reasons.append("coefficient_below_relative_diagnostic_floor")
                    if not four:
                        reasons.append("four_parameter_check_unavailable_n_less_than_6")
                    if len(trims)!=2:
                        reasons.append("trim_check_unavailable_n_less_than_5")
                    for name,passed in gates.items():
                        if not passed:
                            reasons.append(name)
                point_diffs=[r for r in analysis["size_differences"] if r["pattern"]==pattern and r["beta"]==beta and lo<=r["theta"]<=hi]
                row=dict(pattern=pattern,beta=beta,window=window,theta_min=lo,theta_max=hi,
                    L=108,comparison_L=90,model="log_plus_inv1",grid=",".join(f"{time:g}" for time in grid),n=len(grid),
                    c=c,c_90=small["c"] if small else None,c_four108=four["c"] if four else None,
                    c_delete_first=trims.get("delete_first",{}).get("c"),c_delete_last=trims.get("delete_last",{}).get("c"),
                    diagnostic_denominator=denominator,size_drift=ds,four_parameter_drift=df,
                    delete_first_drift=dl,delete_last_drift=dr,checks_available=complete,
                    max_absolute_D_difference=max((r["absolute_D_difference"] for r in point_diffs),default=None),
                    max_relative_D_difference=max((r["relative_D_difference"] for r in point_diffs if r["relative_D_difference"] is not None),default=None),
                    **gates,qualified=bool(qualified),selected=False,rejection_reasons=";".join(reasons))
                candidates.append(row);group.append(row)
            accepted=sorted((r for r in group if r["qualified"]),key=lambda r:(r["theta_min"],r["theta_max"]),reverse=True)
            best=accepted[0] if accepted else None
            if best:
                best["selected"]=True
            eq=index.get(("equilibrium",pattern,beta,"","18-90","log_plus_inv1"))
            extended=index.get(("equilibrium",pattern,beta,"","18-144","log_plus_inv1"))
            selected.append(dict(pattern=pattern,beta=beta,status="qualified_finite_scale_candidate" if best else "no_stable_candidate",
                selection_is_posthoc=True,window=best["window"] if best else "",c=best["c"] if best else None,
                interpretation="local_window_diagnostic_only_not_final_corner_coefficient",
                L=108,model="log_plus_inv1",qualified_candidate_count=len(accepted),
                qualified_windows=",".join(r["window"] for r in accepted),
                c_90=best["c_90"] if best else None,c_four108=best["c_four108"] if best else None,
                c_delete_first=best["c_delete_first"] if best else None,c_delete_last=best["c_delete_last"] if best else None,
                size_drift=best["size_drift"] if best else None,
                four_parameter_drift=best["four_parameter_drift"] if best else None,
                delete_first_drift=best["delete_first_drift"] if best else None,
                delete_last_drift=best["delete_last_drift"] if best else None,
                max_absolute_D_difference=best["max_absolute_D_difference"] if best else None,
                max_relative_D_difference=best["max_relative_D_difference"] if best else None,
                c_eq_old=eq["c"] if eq else None,c_eq_extended=extended["c"] if extended else None,
                eq_relative_difference=relative(best["c"]-eq["c"],eq["c"],floor) if best and eq else None,
                eq_extended_relative_difference=relative(best["c"]-extended["c"],extended["c"],floor) if best and extended else None,
                failure_summary=" | ".join(r["window"]+":"+r["rejection_reasons"] for r in group if not r["qualified"])))
    for row in selected:
        aliases={"selected_window":"window", "selected_c":"c", "selected_model":"model", "selected_L":"L",
                 "selected_size_drift":"size_drift", "selected_model_drift":"four_parameter_drift",
                 "selected_lower_trim_drift":"delete_first_drift", "selected_upper_trim_drift":"delete_last_drift"}
        row.update({alias:row[key] for alias,key in aliases.items()})
    return dict(candidates=candidates,selection=selected,selection_fits=trim_fits)


def order_diagnostic(observations, protocol, betas):
    rules=protocol["model_order_diagnostic"]
    floor=rules["denominator_floor"]
    fits, diagnostics=[],[]
    for pattern in PATTERNS:
        for beta in betas:
            tables={size:{r["theta"]:r["D"] for r in observations.values()
                          if r["pattern"]==pattern and r["beta"]==beta and r["L"]==size and math.isfinite(r["theta"])}
                    for size in (90,108)}
            index={}
            for window,(lo,hi) in rules["windows"].items():
                grid=[time for time in TIMES if lo<=time<=hi]
                absent=[(size,time) for size in (90,108) for time in grid if time not in tables[size]]
                for model in rules["models"]:
                    if absent:
                        diagnostics.append(dict(pattern=pattern,beta=beta,window=window,model=model,
                                                status="missing_common_grid",missing_points=json.dumps(absent)))
                        continue
                    for size in (90,108):
                        result=fit(grid,[tables[size][time] for time in grid],model)
                        if result is None:
                            raise ValueError("Order diagnostic lacks parameter_count+2 points")
                        result=dict(domain="time_order",pattern=pattern,beta=beta,L=size,window=window,**result)
                        fits.append(result);index[(window,size,model)]=result
            for window in rules["windows"]:
                for model in rules["models"]:
                    small=index.get((window,90,model));large=index.get((window,108,model))
                    if not large:
                        continue
                    first=index.get(("3-8",108,model));second=index.get(("4-8",108,model))
                    four=index[(window,108,"log_plus_inv1_inv2")]
                    five=index[(window,108,"log_plus_inv1_inv2_inv3")]
                    grid=[float(value) for value in large["grid"].split(",")]
                    differences=[abs(tables[108][time]-tables[90][time]) for time in grid]
                    diagnostics.append(dict(pattern=pattern,beta=beta,window=window,model=model,status="available",
                        L_small=90,L_large=108,n=large["n"],grid=large["grid"],c_90=small["c"],c_108=large["c"],
                        size_drift=relative(abs(large["c"]-small["c"]),large["c"],floor),
                        size_sign_changed=large["c"]*small["c"]<=0,
                        c_window3_8=first["c"] if first else None,c_window4_8=second["c"] if second else None,
                        window_drift=relative(abs(first["c"]-second["c"]),second["c"],floor) if first and second else None,
                        window_sign_changed=(first["c"]*second["c"]<=0) if first and second else None,
                        c_four108=four["c"],c_five108=five["c"],
                        order_drift=relative(abs(five["c"]-four["c"]),four["c"],floor),
                        order_sign_changed=five["c"]*four["c"]<=0,
                        max_absolute_D_difference=max(differences),rmse=large["rmse"],condition=large["condition"],
                        inverse1_to_log_span=large["inverse1_to_log_span"],inverse2_to_log_span=large["inverse2_to_log_span"],
                        inverse3_to_log_span=large.get("inverse3_to_log_span"),
                        interpretation="model_order_diagnostic_only_no_pass_label_or_eq_target"))
    return dict(order_fits=fits,order_diagnostics=diagnostics)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    parser.add_argument("--betas", default="0,0.5,1,2")
    parser.add_argument("--require-betas", default="0,0.5,1,2", help="Final mode requires the full L90/108 finite-time grids for these betas.")
    parser.add_argument("--require-equilibrium-betas", default="0", help="Final mode requires the L108/126/144 equilibrium extension only for these betas.")
    parser.add_argument("--require-late", action="store_true", help="Require completion of the separate beta-zero L108/144 diagnostic grid.")
    parser.add_argument("--init-protocol", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    protocol_path=OUT/"protocol.json"
    protocol=init_protocol(protocol_path)
    protocol=add_selection_protocol(protocol,protocol_path)
    protocol=add_order_protocol(protocol,protocol_path)
    if args.init_protocol:
        print(json.dumps(dict(protocol=str(protocol_path), sha256=sha(protocol_path.read_bytes()))))
        return
    betas=[float(x) for x in args.betas.split(",")]
    required=[float(x) for x in args.require_betas.split(",") if x]
    required_eq=[float(x) for x in args.require_equilibrium_betas.split(",") if x]
    if not (set(required)|set(required_eq)).issubset(betas):
        raise ValueError("Required beta outside selected beta set")
    observations, receipts, duplicates, identity_error=read_sources(SOURCES, betas, args.stage)
    analysis=make_analysis(observations, protocol, betas)
    late=analyze_late(observations, protocol) if 0 in betas else dict(fits=[], late_diagnostics=[], late_size_differences=[], coverage=[])
    analysis["fits"].extend(late.pop("fits"))
    analysis["coverage"].extend(late.pop("coverage"))
    analysis.update(late)
    analysis.update(controlled_selection(observations,analysis,protocol,betas))
    analysis.update(order_diagnostic(observations,protocol,betas))
    required_missing=[r for r in analysis["coverage"] if not r["complete"] and
                      ((r["domain"] == "time" and r["beta"] in required) or
                       (r["domain"] == "equilibrium_extension" and r["beta"] in required_eq) or
                       (r["domain"] == "time_late" and args.require_late))]
    if args.stage == "final" and required_missing:
        raise ValueError(f"Final required grid incomplete: {json.dumps(required_missing[:10])}")
    points=sorted(observations.values(), key=lambda r:(PATTERNS.index(r["pattern"]), r["beta"], r["L"], r["theta"]))
    tables={"points":points, **analysis, "duplicates":duplicates}
    # JSON uses a string for the equilibrium marker to remain standards-compliant.
    json_points=[dict(r, theta="Inf" if math.isinf(r["theta"]) else r["theta"]) for r in points]
    current_hashes={str(path.relative_to(HERE)):sha(path.read_bytes()) for path in SOURCES if path.exists()}
    changed=[receipt["path"] for receipt in receipts if "sha256" in receipt and current_hashes.get(receipt["path"]) != receipt["sha256"]]
    if args.stage == "final" and changed:
        raise ValueError(f"Sources changed during final analysis: {changed}")
    for name, rows in tables.items():
        write_csv(OUT/f"calibration_{name}.csv", rows, ("pattern", "beta"))
    write_csv(OUT/"selection.csv", analysis["selection"], ("pattern","beta"))
    counts={f"{beta:g}":dict(total=sum(r["beta"] == beta for r in analysis["comparison"]),
                            complete=sum(r["beta"] == beta and r["diagnostic_complete"] for r in analysis["comparison"]),
                            stable=sum(r["beta"] == beta and r["stable_diagnostic"] for r in analysis["comparison"]),
                            controlled_candidates=sum(r["beta"] == beta and r["status"] == "qualified_finite_scale_candidate" for r in analysis["selection"])) for beta in betas}
    summary=dict(generated_utc=utc(), stage=args.stage, required_betas=required,
        required_equilibrium_betas=required_eq, require_late=args.require_late, analyzed_betas=betas,
        protocol_sha256=sha(protocol_path.read_bytes()), script_sha256=sha(Path(__file__).read_bytes()),
        sources=receipts, sources_changed_during_execution=changed, unique_selected_observations=len(points),
        duplicate_observations=len(duplicates), max_D_identity_error=identity_error,
        required_grid_complete=not required_missing, required_grid_missing=required_missing,
        counts_by_beta=counts, table_rows={name:len(rows) for name,rows in tables.items()},
        interpretation="Finite-window candidates. Stable labels are size/window/model diagnostics independent of the equilibrium target; they are not confidence intervals or proof of asymptotic corner equality.")
    (OUT/"calibration_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False)+"\n", encoding="utf-8")
    bundle=dict(protocol=protocol, receipt=summary, points=json_points,
                **analysis)
    (OUT/"plot_data.json").write_text(json.dumps(bundle, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")
    print(json.dumps(dict(stage=args.stage, required_grid_complete=not required_missing,
                         counts_by_beta=counts, table_rows=summary["table_rows"], changed_sources=changed)))


if __name__ == "__main__":
    main()
