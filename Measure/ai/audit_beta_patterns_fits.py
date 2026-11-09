"""Independent 80-digit audit for the additional-pattern dense-beta experiment.

No simulation or primary analysis routine is imported. Least squares is solved
with mpmath QR from raw CSV decimal strings. Pattern is part of every audit key;
the two strip entropies are kept distinct throughout the source checks.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal as D
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import sys

AI = Path(__file__).resolve().parent
OUT = AI / "analysis_beta_patterns"
DEPENDENCIES = OUT / "audit_dependencies"
if DEPENDENCIES.exists():
    sys.path.insert(0, str(DEPENDENCIES))
try:
    import mpmath as mp
except ImportError as error:
    raise SystemExit("Install audit dependency with: python -m pip install --target "
                     "src/Measure/ai/analysis_beta_patterns/audit_dependencies mpmath==1.3.0") from error

# Reuse only independently authored audit logic, never the main analysis/fit.
import audit_uniformA_dense_fits as audit_logic

PRECISION = 80
MODEL_COLUMNS = audit_logic.MODEL_COLUMNS
sha = audit_logic.sha256
canonical = audit_logic.canonical
read_csv = audit_logic.read_csv
write_csv = audit_logic.write_csv
csv_rows = audit_logic.csv_rows
assert_close = audit_logic.assert_close


def fit_mpmath(grid, values, model):
    """Independent rectangular QR solve, unlike the primary numpy SVD fit."""
    columns = MODEL_COLUMNS[model]
    if len(grid) != len(values) or len(grid) < len(columns) + 2:
        raise ValueError("Insufficient actual observations for parameters plus two")
    with mp.workdps(PRECISION):
        rows = []
        for value in grid:
            x = mp.mpf(str(value))
            if x <= 0:
                raise ValueError("Nonpositive logarithmic argument")
            terms = {"a": x, "c": mp.log(x), "b": mp.mpf(1),
                     "d": 1/x, "e": 1/x**2, "f": 1/x**3}
            rows.append([terms[name] for name in columns])
        y = [mp.mpf(str(value)) for value in values]
        coefficients, _ = mp.qr_solve(mp.matrix(rows), mp.matrix(y))
        residual = [value-mp.fsum(row[j]*coefficients[j] for j in range(len(columns)))
                    for row, value in zip(rows, y)]
        result = {name: D(mp.nstr(coefficients[j], PRECISION)) for j, name in enumerate(columns)}
        result.update(rmse=D(mp.nstr(mp.sqrt(mp.fsum(x*x for x in residual)/len(y)), PRECISION)),
                      max_abs_residual=D(mp.nstr(max(abs(x) for x in residual), PRECISION)), n=len(y))
        return result


# The inherited independent diagnostic engine also calls this for third-size fits.
audit_logic.fit_decimal = fit_mpmath


def dependency_receipt():
    package = Path(mp.__file__).resolve().parent
    entries = [(str(path.relative_to(package)).replace("\\", "/"), sha(path.read_bytes()))
               for path in sorted(package.rglob("*.py"))]
    return dict(name="mpmath", version=mp.__version__, precision_digits=PRECISION,
                solver="mpmath.qr_solve; rectangular high-precision QR, unweighted",
                source_tree_sha256=sha(json.dumps(entries, separators=(",", ":")).encode()),
                independent_logic_sha256=sha(Path(audit_logic.__file__).read_bytes()))


def pattern_metadata(plan):
    metadata = {}
    for name in plan["patterns"]:
        bits = plan["pattern_bits"][name]
        m = math.isqrt(len(bits)//2)
        if len(bits) != 2*m*m or set(bits)-{"0", "1"} or bits.count("1") != m*m:
            raise ValueError(f"Invalid declared half-filled pattern: {name}")
        special = all(bits[i:i+2] in ("10", "01") for i in range(0, len(bits), 2))
        metadata[name] = dict(pattern=name, occupied_bits=bits, m=m,
                              family="special" if special else "general")
    return metadata


def load_sources(summary, errors, stage):
    metadata = pattern_metadata(summary["plan"])
    betas = {canonical(b) for b in summary["betas"]}
    merged, receipts = {}, []
    for source in summary["sources"]:
        origin = Path(source["path"])
        if source.get("missing"):
            if stage == "final" and origin.exists():
                errors.append(f"Previously missing source now exists: {origin}")
            receipts.append(dict(path=str(origin), missing=True))
            continue
        snapshot = Path(source["parsed_snapshot"])
        raw = snapshot.read_bytes()
        if sha(raw) != source["parsed_sha256"]:
            errors.append(f"Source snapshot hash mismatch: {snapshot}")
        if stage == "final" and (sha(origin.read_bytes()) != source["sha256"] or source.get("ignored_incomplete_tail_bytes", 0)):
            errors.append(f"Final original source changed or has incomplete tail: {origin}")
        rows = csv_rows(raw)
        if len(rows) != source["rows"]:
            errors.append(f"Source row count mismatch: {origin}")
        receipts.append(dict(path=str(origin), source_sha256=source["sha256"],
                             snapshot_sha256=sha(raw), rows=len(rows)))
        for line, row in enumerate(rows, 2):
            name = row.get("pattern")
            if name not in metadata:
                continue
            beta, theta = canonical(row["beta"]), canonical(row["theta"])
            if beta not in betas or theta.is_nan() or theta <= 0:
                continue
            size = int(row["L"])
            meta = metadata[name]
            if size % (3*meta["m"]):
                errors.append(f"Incompatible geometry: {origin}:{line}")
            for field in ("occupied_bits", "m", "family"):
                if row.get(field) and str(row[field]) != str(meta[field]):
                    errors.append(f"Pattern metadata mismatch: {origin}:{line}/{field}")
            for field in ("phase_x", "phase_y"):
                if row.get(field) and float(row[field]) != 0:
                    errors.append(f"Translated protocol row: {origin}:{line}/{field}")
            for field in ("x0", "y0"):
                if row.get(field) and float(row[field]) != 1:
                    errors.append(f"Different region origin: {origin}:{line}/{field}")
            ground = row.get("premeasurement_ground", row.get("ground", ""))
            if ground and ground.lower() != "true":
                errors.append(f"Row is not premeasurement-ground accessible: {origin}:{line}")
            excess = next((row[field] for field in ("premeasurement_excess", "premeasurement_limit_excess", "accessible_excess")
                           if row.get(field)), None)
            if excess is not None and abs(D(excess)) > D("1e-8"):
                errors.append(f"Premeasurement energy excess: {origin}:{line}")
            values = {field: D(row[field]) for field in ("S_A", "S_C", "S_B", "D")}
            if not all(value.is_finite() for value in values.values()):
                errors.append(f"Nonfinite entropy: {origin}:{line}")
            if abs((2*values["S_A"]+values["S_C"])/3-values["S_B"]-values["D"]) > D("1e-9"):
                errors.append(f"D identity mismatch: {origin}:{line}")
            key = name, beta, size, theta
            if key in merged and max(abs(merged[key][field]-value) for field, value in values.items()) > D("1e-9"):
                errors.append(f"Conflicting duplicate observation: {key}")
            merged[key] = {**meta, **values, "source_file": str(origin), "source_row": line}
    return merged, metadata, receipts


def group_summary(summary, pattern):
    """The primary mathematical protocol is shared; grids remain pattern-local."""
    view = dict(summary)
    view["primary"] = summary["primary"][pattern]
    view["arguments"] = {**summary["arguments"],
                         "eq_max": summary["plan"]["equilibrium_primary"]["window"][1],
                         "theta_min": summary["plan"]["theta_min_screen"],
                         "max_theta_over_L": summary["plan"]["theta_over_L_max_screen"]}
    return view


def expected_group_fits(merged, metadata, summary):
    groups, expected = {}, {}
    for pattern in metadata:
        groups[pattern] = {(beta, size, theta): row for (name, beta, size, theta), row in merged.items() if name == pattern}
        fits = audit_logic.expected_fits(groups[pattern], group_summary(summary, pattern))
        expected.update({(pattern, *key): value for key, value in fits.items()})
    return groups, expected


def verify_metadata(row, metadata, errors, identity):
    for field in ("pattern", "m", "family"):
        if str(row.get(field)) != str(metadata[field]):
            errors.append(f"Output metadata mismatch {identity}/{field}")


def audit_fits(merged, metadata, summary, errors):
    groups, expected = expected_group_fits(merged, metadata, summary)
    local_expected_groups = {name: {key[1:]: value for key, value in expected.items() if key[0] == name}
                             for name in metadata}
    actual, hashes = {}, {}
    for domain, filename in (("equilibrium", "equilibrium_fits.csv"),
                             ("time_individual", "time_individual_fits.csv"),
                             ("time_pair", "time_pair_fits.csv")):
        raw = (OUT/filename).read_bytes()
        hashes[filename] = sha(raw)
        for row in csv_rows(raw):
            key = (row["pattern"], *audit_logic.fit_key(domain, row))
            if key in actual:
                errors.append(f"Duplicate fit key: {key}")
            actual[key] = row
    if expected.keys() != actual.keys():
        errors.append(f"Fit coverage differs: missing={sorted(expected.keys()-actual.keys())}; extra={sorted(actual.keys()-expected.keys())}")
    coefficients, comparisons = {}, []
    maxima = dict(c=D(0), parameter=D(0), residual=D(0), metric=D(0))
    aliases = {"d": "inv1", "e": "inv2", "f": "inv3"}
    for key, data in expected.items():
        pattern, domain = key[:2]
        fit = fit_mpmath(data["grid"], data["values"], key[-1])
        coefficients[key] = fit
        small = None
        if domain == "time_pair":
            small = fit_mpmath(data["grid"], data["small_values"], key[-1])
            coefficients[key+("small",)] = small
        if key not in actual:
            continue
        row = actual[key]
        verify_metadata(row, metadata[pattern], errors, key)
        grid = [D(str(x)) for x in data["grid"]]
        if audit_logic.grid_values(row["grid"]) != grid:
            errors.append(f"Measured fit grid mismatch: {key}")
        for field, value in (("n", len(grid)), ("parameter_count", len(MODEL_COLUMNS[key[-1]])),
                             ("residual_dof", len(grid)-len(MODEL_COLUMNS[key[-1]]))):
            if int(row[field]) != value:
                errors.append(f"Fit dimensions differ: {key}/{field}")
        differences = {}
        for name in MODEL_COLUMNS[key[-1]]:
            differences[name] = assert_close(errors, f"{key}/{name}", fit[name], row[aliases.get(name, name)],
                                            absolute=D("1e-9") if name == "c" else D("1e-7"))
        maxima["c"] = max(maxima["c"], differences["c"])
        maxima["parameter"] = max(maxima["parameter"], max(differences.values()))
        for field in ("rmse", "max_abs_residual"):
            maxima["residual"] = max(maxima["residual"], assert_close(errors, f"{key}/{field}", fit[field], row[field], absolute=D("1e-10")))
        for field, value in (("x_min", min(grid)), ("x_max", max(grid))):
            assert_close(errors, f"{key}/{field}", value, row[field], absolute=D(0), relative=D(0))
        if domain == "equilibrium" and (row["extent_complete"].lower() == "true") != data["extent_complete"]:
            errors.append(f"Equilibrium endpoint status mismatch: {key}")
        log_span = abs(fit["c"])*(max(grid)/min(grid)).ln()
        correction = D(0)
        numbers, booleans = {}, {}
        for power, name in ((1, "d"), (2, "e"), (3, "f")):
            if name in fit:
                span = abs(fit[name])*(min(grid)**(-power)-max(grid)**(-power))
                correction += span
                numbers[f"inverse{power}_to_log_span"] = audit_logic.ratio(span, log_span)
        numbers["total_correction_to_log_span"] = audit_logic.ratio(correction, log_span)
        local_expected = local_expected_groups[pattern]
        local_summary = group_summary(summary, pattern)
        local_key = key[1:]
        if domain == "time_pair":
            delta = max(abs(a-b) for a, b in zip(data["values"], data["small_values"]))
            numbers.update(c_small=small["c"], size_drift=audit_logic.ratio(abs(small["c"]-fit["c"]), abs(fit["c"])),
                           max_absolute_D_difference=delta, D_difference_to_log_span=audit_logic.ratio(delta, log_span))
            booleans.update(size_sign_changed=small["c"]*fit["c"] <= 0,
                            scale_screen_large=audit_logic.time_screen(local_key, local_expected, local_summary),
                            scale_screen_both=audit_logic.time_screen(local_key, local_expected, local_summary, both=True))
        elif domain == "time_individual":
            booleans["scale_screen"] = audit_logic.time_screen(local_key, local_expected, local_summary)
        maxima["metric"] = max(maxima["metric"], audit_logic.compare_record(row, {**numbers, **booleans}, key, errors))
        comparisons.append(dict(**{field: metadata[pattern][field] for field in ("pattern", "m", "family")},
                                domain=domain, identity=str(key[2:]), model=key[-1], grid=row["grid"], n=len(grid),
                                c_high_precision=str(fit["c"]), c_published=row["c"], c_error=str(differences["c"]),
                                maximum_parameter_error=str(max(differences.values())), rmse_high_precision=str(fit["rmse"])))
    return groups, expected, coefficients, comparisons, maxima, hashes


def audit_coverage(merged, metadata, summary, errors):
    expected = set()
    betas = summary["betas"]
    for grid in summary["plan"].get("required_finite_grids", []):
        expected.update((pattern, canonical(beta), int(size), canonical(theta)) for pattern, size, theta, beta in
                        itertools.product(grid.get("patterns", metadata), grid.get("sizes", grid.get("L", [])),
                                          grid.get("times", grid.get("theta", [])), grid.get("betas", grid.get("beta", betas))))
    for grid in summary["plan"].get("required_equilibrium_grids", []):
        expected.update((pattern, canonical(beta), int(size), D("Infinity")) for pattern, size, beta in
                        itertools.product(grid.get("patterns", metadata), grid.get("sizes", grid.get("L", [])),
                                          grid.get("betas", grid.get("beta", betas))))
    missing = expected-merged.keys()
    if len(expected) != summary["required_keys"] or len(missing) != summary["missing_required_keys"]:
        errors.append("Required parameter coverage differs from raw data")
    if len(merged) != summary["rows"]:
        errors.append("Merged observation count differs from source reconstruction")
    raw = (OUT/"observations.csv").read_bytes()
    rows = csv_rows(raw)
    actual = {(r["pattern"], canonical(r["beta"]), int(r["L"]), canonical(r["theta"])): r for r in rows}
    if len(actual) != len(rows) or actual.keys() != merged.keys():
        errors.append("Merged observations key set differs")
    for key in actual.keys() & merged.keys():
        verify_metadata(actual[key], metadata[key[0]], errors, key)
        for field in ("S_A", "S_C", "S_B", "D"):
            assert_close(errors, f"merged{key}/{field}", merged[key][field], actual[key][field], absolute=D("1e-12"))
    return dict(required=len(expected), missing=len(missing), merged=len(merged)), {"observations.csv": sha(raw)}


def audit_diagnostics(groups, metadata, summary, coefficients, expected, errors):
    primary, sensitivity = {}, {}
    for pattern, data in groups.items():
        local_summary = group_summary(summary, pattern)
        local_expected = {key[1:]: value for key, value in expected.items() if key[0] == pattern}
        local_coefficients = {key[1:]: value for key, value in coefficients.items() if key[0] == pattern}
        selected = audit_logic.expected_primary(data, local_summary, local_coefficients, local_expected)
        tags = {field: metadata[pattern][field] for field in ("pattern", "m", "family")}
        for beta, row in selected.items():
            primary[pattern, beta] = {**tags, **row}
            anchor = ("equilibrium", beta, 18, int(local_summary["arguments"]["eq_max"]), "log_inv1")
            if anchor in local_expected:
                reference = local_coefficients[anchor]["c"]
                for key, info in local_expected.items():
                    if key[0] != "equilibrium" or key[1] != beta:
                        continue
                    c = local_coefficients[key]["c"]
                    entry = dict(**tags, domain="equilibrium", beta=beta, model=key[-1], window=f"{key[2]}-{key[3]}",
                                 grid=audit_logic.grid_text(info["grid"]), c=c, reference_c=reference,
                                 relative_drift=audit_logic.ratio(abs(c-reference), abs(reference)),
                                 extent_complete=info["extent_complete"], reference_extent_complete=local_expected[anchor]["extent_complete"])
                    sensitivity[pattern, "equilibrium", beta, entry["window"], key[-1]] = entry
            if row.get("c_neq_diagnostic") is None:
                continue
            size, small, reference = row["primary_L"], row["comparison_L"], row["c_neq_diagnostic"]
            for key, info in local_expected.items():
                if key[0] != "time_individual" or key[1:3] != (beta, size):
                    continue
                window, model = key[3:]
                grid, c = info["grid"], local_coefficients[key]["c"]
                pair = ("time_pair", beta, small, size, window, model)
                common = pair in local_expected
                same_window, same_model = window == row["primary_window"], model == row["primary_model"]
                status = ("available_within_both_size_screens" if audit_logic.time_screen(pair, local_expected, local_summary, both=True)
                          else "available_but_comparison_size_outside_screen") if common else "missing_common_grid_fit"
                entry = dict(**tags, domain="time", beta=beta, L=size, model=model, window=window,
                             grid=audit_logic.grid_text(grid), n=len(grid), residual_dof=len(grid)-len(MODEL_COLUMNS[model]),
                             c=c, reference_c=reference, reference_grid=row["grid"],
                             relative_drift=audit_logic.ratio(abs(c-reference), abs(reference)),
                             comparison_kind="primary_full_grid_check" if same_window and same_model else
                                             "model" if same_window else "window" if same_model else "window_and_model",
                             scale_screen_at_primary_L=audit_logic.time_screen(key, local_expected, local_summary),
                             theta_over_L=max(grid)/size, comparison_L=small, size_check_status=status,
                             size_check_grid=audit_logic.grid_text(local_expected[pair]["grid"]) if common else None,
                             size_drift=audit_logic.ratio(abs(local_coefficients[pair]["c"]-local_coefficients[pair+("small",)]["c"]),
                                                         abs(local_coefficients[pair]["c"])) if common else None,
                             scale_screen_both_sizes=audit_logic.time_screen(pair, local_expected, local_summary, both=True) if common else None)
                sensitivity[pattern, "time", beta, window, model] = entry
    primary_raw = (OUT/"primary_coefficients.csv").read_bytes()
    actual_primary = {(r["pattern"], canonical(r["beta"])): r for r in csv_rows(primary_raw)}
    sensitive_raw = (OUT/"time_sensitivity.csv").read_bytes()
    actual_sensitive = {(r["pattern"], r["domain"], canonical(r["beta"]), r["window"], r["model"]): r for r in csv_rows(sensitive_raw)}
    maximum = D(0)
    for label, desired, actual in (("primary", primary, actual_primary), ("sensitivity", sensitivity, actual_sensitive)):
        if desired.keys() != actual.keys():
            errors.append(f"{label} key coverage mismatch")
        for key in desired.keys() & actual.keys():
            maximum = max(maximum, audit_logic.compare_record(actual[key], desired[key], (label, *key), errors))
    write_csv(OUT/"independent_fit_validation_primary.csv", list(primary.values()))
    write_csv(OUT/"independent_fit_validation_sensitivity.csv", list(sensitivity.values()))
    return dict(primary_rows=len(primary), sensitivity_rows=len(sensitivity), max_numeric_error=str(maximum)), {
        "primary_coefficients.csv": sha(primary_raw), "time_sensitivity.csv": sha(sensitive_raw)}


def audit_raw_entropies(merged, metadata, summary, errors):
    expected = {}
    eq_max = int(summary["plan"]["equilibrium_primary"]["window"][1])
    for pattern, beta, region, low, order in itertools.product(metadata, map(canonical, summary["betas"]),
                                                              ("S_A", "S_C", "S_B"), (18, 36, 54), (1, 2)):
        points = {size: values[region] for (name, b, size, theta), values in merged.items()
                  if name == pattern and b == beta and theta.is_infinite() and low <= size <= eq_max}
        grid = sorted(points)
        model = "area_log_inv" if order == 1 else "area_log_inv2"
        if len(grid) >= len(MODEL_COLUMNS[model]) + 2:
            expected[pattern, beta, region, low, eq_max, f"area_log_inv{order}"] = (grid, [points[x] for x in grid], model)
    raw = (OUT/"raw_entropy_fits.csv").read_bytes()
    rows = csv_rows(raw)
    actual = {(r["pattern"], canonical(r["beta"]), r["region"], int(r["requested_Lmin"]),
               int(r["requested_Lmax"]), r["model"]): r for r in rows}
    if len(actual) != len(rows) or expected.keys() != actual.keys():
        errors.append("Raw entropy model/grid coverage mismatch")
    comparisons = []
    max_c, max_parameter = D(0), D(0)
    for key, (grid, values, model) in expected.items():
        result = fit_mpmath(grid, values, model)
        if key not in actual:
            continue
        row = actual[key]
        verify_metadata(row, metadata[key[0]], errors, key)
        if audit_logic.grid_values(row["grid"]) != list(map(D, grid)):
            errors.append(f"Raw entropy grid mismatch {key}")
        fields = {"a": "a", "c": "log_coefficient", "b": "b", "d": "inv1", "e": "inv2"}
        differences = {}
        for name in MODEL_COLUMNS[model]:
            differences[name] = assert_close(errors, f"raw{key}/{name}", result[name], row[fields[name]],
                                            absolute=D("1e-8") if name == "c" else D("1e-6"))
        max_c = max(max_c, differences["c"])
        max_parameter = max(max_parameter, max(differences.values()))
        expected_values = dict(n=len(grid), parameter_count=len(MODEL_COLUMNS[model]),
                               residual_dof=len(grid)-len(MODEL_COLUMNS[model]), rmse=result["rmse"],
                               max_abs_residual=result["max_abs_residual"],
                               extent_complete=grid[0] == key[3] and grid[-1] == key[4],
                               corner_c=-result["c"] if key[2] == "S_B" else None)
        audit_logic.compare_record(row, expected_values, ("raw_entropy", *key), errors)
        comparisons.append(dict(pattern=key[0], beta=key[1], region=key[2], model=key[-1],
                                requested_Lmin=key[3], requested_Lmax=key[4], grid=row["grid"],
                                log_coefficient_high_precision=str(result["c"]),
                                log_coefficient_published=row["log_coefficient"], c_error=str(differences["c"]),
                                maximum_parameter_error=str(max(differences.values()))))
    write_csv(OUT/"independent_fit_validation_raw_entropies.csv", comparisons)
    return dict(fits=len(comparisons), max_log_coefficient_error=str(max_c), max_parameter_error=str(max_parameter)), {
        "raw_entropy_fits.csv": sha(raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    args = parser.parse_args()
    summary_path = OUT/"summary.json"
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    errors = []
    if args.stage == "final" and summary["stage"] != "final":
        errors.append("Final audit requested for preliminary analysis")
    if sha((AI/"analyze_beta_patterns.py").read_bytes()) != summary["script_sha256"]:
        errors.append("Primary analysis changed after summary")
    plan_path = Path(summary["arguments"]["plan"])
    if sha(plan_path.read_bytes()) != summary["plan_sha256"]:
        errors.append("Plan changed after summary")
    for dependency in summary["dependencies"]:
        if sha(Path(dependency["path"]).read_bytes()) != dependency["sha256"]:
            errors.append(f"Primary dependency changed: {dependency['path']}")
    merged, metadata, sources = load_sources(summary, errors, args.stage)
    coverage, hashes = audit_coverage(merged, metadata, summary, errors)
    if args.stage == "final" and coverage["missing"]:
        errors.append("Final coverage incomplete")
    groups, expected, coefficients, comparisons, maxima, fit_hashes = audit_fits(merged, metadata, summary, errors)
    hashes.update(fit_hashes)
    diagnostics, diagnostic_hashes = audit_diagnostics(groups, metadata, summary, coefficients, expected, errors)
    hashes.update(diagnostic_hashes)
    raw_entropies, raw_hashes = audit_raw_entropies(merged, metadata, summary, errors)
    hashes.update(raw_hashes)
    if sha(summary_path.read_bytes()) != sha(summary_raw):
        errors.append("Summary changed during audit")
    for filename, digest in hashes.items():
        if sha((OUT/filename).read_bytes()) != digest:
            errors.append(f"Published table changed during audit: {filename}")
    write_csv(OUT/"independent_fit_validation_fits.csv", comparisons)
    receipt = dict(status="PASS" if not errors else "FAIL", stage=args.stage,
                   generated_utc=datetime.now(timezone.utc).isoformat(), primary_summary_sha256=sha(summary_raw),
                   primary_script_sha256=summary["script_sha256"], primary_dependencies=summary["dependencies"],
                   plan_sha256=summary["plan_sha256"], audit_script_sha256=sha(Path(__file__).read_bytes()),
                   arithmetic=dependency_receipt(), source_receipts=sources, table_sha256=hashes,
                   patterns=metadata, coverage=coverage, core_fit_count=len(comparisons),
                   maxima={key: str(value) for key, value in maxima.items()}, diagnostics=diagnostics,
                   raw_entropy_diagnostics=raw_entropies, errors=errors,
                   scope="Independent source merge and 80-digit mpmath QR fits, pattern-specific raw entropies without SA=SC, common-grid size diagnostics, primary selections and local labels. No simulation rerun or asymptotic-equivalence claim.")
    (OUT/"independent_fit_validation.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: receipt[key] for key in ("status", "stage", "coverage", "core_fit_count", "maxima", "diagnostics", "raw_entropy_diagnostics", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
