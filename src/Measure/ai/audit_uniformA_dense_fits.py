"""Independent 80-digit audit of uniform-A dense scaling fits.

The physical simulation and the primary fitting code are never imported or run.
All least-squares coefficients are reconstructed from source CSV decimal values.
Preliminary runs audit the captured source snapshots; final runs also require
unchanged original sources, the recorded plan, and the final analysis stage.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from decimal import Decimal, localcontext
import hashlib
import io
import itertools
import json
import math
from pathlib import Path

AI = Path(__file__).resolve().parent
OUT = AI / "analysis_uniformA_dense"
D = Decimal
PRECISION = 80
MODEL_COLUMNS = {
    "log": ("c", "b"),
    "log_inv": ("c", "b", "d"),
    "log_inv2": ("c", "b", "d", "e"),
    "log_inv1": ("c", "b", "d"),
    "log_inv1_inv2": ("c", "b", "d", "e"),
    "log_inv1_inv2_inv3": ("c", "b", "d", "e", "f"),
    "log_plus_inv1": ("c", "b", "d"),
    "log_plus_inv2": ("c", "b", "e"),
    "log_plus_inv1_inv2": ("c", "b", "d", "e"),
    "log_plus_inv1_inv2_inv3": ("c", "b", "d", "e", "f"),
    "area_log": ("a", "c", "b"),
    "area_log_inv": ("a", "c", "b", "d"),
    "area_log_inv2": ("a", "c", "b", "d", "e"),
}


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def csv_rows(raw: bytes) -> list[dict[str, str]]:
    return list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))


def read_csv(path: Path) -> list[dict[str, str]]:
    return csv_rows(path.read_bytes())


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("\n", encoding="utf-8")
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def decimal_solve(matrix: list[list[Decimal]], rhs: list[Decimal]) -> list[Decimal]:
    """Partial-pivot Gaussian elimination; no numpy or primary fit dependency."""
    count = len(rhs)
    augmented = [list(row) + [value] for row, value in zip(matrix, rhs)]
    for column in range(count):
        pivot = max(range(column, count), key=lambda i: abs(augmented[i][column]))
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        if augmented[column][column] == 0:
            raise ArithmeticError("Singular model in independent fit")
        for i in range(column + 1, count):
            multiplier = augmented[i][column] / augmented[column][column]
            for j in range(column, count + 1):
                augmented[i][j] -= multiplier * augmented[column][j]
    answer = [D(0)] * count
    for i in range(count - 1, -1, -1):
        answer[i] = (augmented[i][-1] - sum(augmented[i][j] * answer[j]
                     for j in range(i + 1, count))) / augmented[i][i]
    return answer


def fit_decimal(grid: list, values: list, model: str) -> dict:
    """Unweighted least squares with independent 80-digit natural logarithms."""
    names = MODEL_COLUMNS[model]
    with localcontext() as context:
        context.prec = PRECISION
        design = []
        for raw in grid:
            x = D(str(raw))
            if x <= 0:
                raise ValueError("Logarithmic fit requires positive arguments")
            terms = {"a": x, "c": x.ln(), "b": D(1),
                     "d": 1 / x, "e": 1 / x**2, "f": 1 / x**3}
            design.append([terms[name] for name in names])
        y = [D(str(value)) for value in values]
        if len(y) != len(design) or len(y) < len(names) + 2:
            raise ValueError("Insufficient observations for parameter-count plus two rule")
        gram = [[sum(row[i] * row[j] for row in design)
                 for j in range(len(names))] for i in range(len(names))]
        rhs = [sum(row[i] * value for row, value in zip(design, y))
               for i in range(len(names))]
        coefficients = decimal_solve(gram, rhs)
        residuals = [value - sum(v * c for v, c in zip(row, coefficients))
                     for row, value in zip(design, y)]
        return {**dict(zip(names, coefficients)),
                "rmse": (sum(value**2 for value in residuals) / len(y)).sqrt(),
                "max_abs_residual": max(abs(value) for value in residuals),
                "n": len(y)}


def grid_values(text: str) -> list[Decimal]:
    raw = text.strip()
    values = json.loads(raw) if raw.startswith("[") else raw.split(",")
    return [D(str(value)) for value in values]


def raw_key(row: dict) -> tuple[str, Decimal, int, Decimal]:
    return row["pattern"], D(row["beta"]), int(row["L"]), D(row["theta"])


def json_safe(value):
    if isinstance(value, D):
        return str(value)
    raise TypeError(type(value).__name__)


def assert_close(errors: list[str], label: str, expected, actual,
                 absolute: Decimal = D("1e-8"), relative: Decimal = D("1e-10")) -> Decimal:
    if expected is None:
        if actual not in (None, ""):
            errors.append(f"{label}: expected missing, found {actual}")
        return D(0)
    if actual in (None, ""):
        errors.append(f"{label}: missing published value")
        return D("Infinity")
    expected, actual = D(str(expected)), D(str(actual))
    difference = abs(expected - actual)
    if difference > absolute + relative * abs(expected):
        errors.append(f"{label}: difference {difference}")
    return difference


def canonical(value) -> Decimal:
    number = D(str(value))
    return number.quantize(D("1e-12")) if number.is_finite() else number


def ratio(numerator: Decimal, denominator: Decimal):
    return numerator / denominator if abs(denominator) >= D("1e-10") else None


def load_sources(summary: dict, errors: list[str], stage: str):
    selected = {}
    receipts = []
    betas = {canonical(x) for x in summary["betas"]}
    for source in summary["sources"]:
        origin = Path(source["path"])
        if source.get("missing"):
            if stage == "final" and origin.exists():
                errors.append(f"Previously absent input now exists: {origin}")
            receipts.append(dict(path=str(origin), missing=True))
            continue
        snapshot = Path(source["parsed_snapshot"])
        raw = snapshot.read_bytes()
        if sha256(raw) != source["parsed_sha256"]:
            errors.append(f"Parsed snapshot hash mismatch: {snapshot}")
        if stage == "final":
            if sha256(origin.read_bytes()) != source["sha256"]:
                errors.append(f"Final source changed: {origin}")
            if source["ignored_incomplete_tail_bytes"]:
                errors.append(f"Incomplete source tail in final analysis: {origin}")
        rows = csv_rows(raw)
        if len(rows) != source["rows"]:
            errors.append(f"Source row count mismatch: {origin}")
        receipts.append(dict(path=str(origin), source_sha256=source["sha256"],
                             snapshot_sha256=sha256(raw), rows=len(rows)))
        for line, row in enumerate(rows, 2):
            if row.get("pattern") != "uniform_A":
                continue
            beta, theta = canonical(row["beta"]), canonical(row["theta"])
            if beta not in betas or theta <= 0 or theta.is_nan():
                continue
            size = int(row["L"])
            values = {name: D(row[name]) for name in ("S_A", "S_C", "S_B", "D")}
            if not all(value.is_finite() for value in values.values()):
                errors.append(f"Nonfinite entropy: {origin}:{line}")
            if abs((2 * values["S_A"] + values["S_C"]) / 3 - values["S_B"] - values["D"]) > D("1e-9"):
                errors.append(f"Raw D identity failed: {origin}:{line}")
            key = beta, size, theta
            if key in selected:
                difference = max(abs(selected[key][name] - value) for name, value in values.items())
                if difference > D("1e-9"):
                    errors.append(f"Duplicate physical observation disagrees: {key}")
            selected[key] = {**values, "source_file": str(origin), "source_row": line}
    return selected, receipts


def fit_key(domain: str, row: dict) -> tuple:
    beta = canonical(row["beta"])
    if domain == "equilibrium":
        return domain, beta, int(row["requested_Lmin"]), int(row["requested_Lmax"]), row["model"]
    if domain == "time_individual":
        return domain, beta, int(row["L"]), row["window"], row["model"]
    return domain, beta, int(row["L_small"]), int(row["L_large"]), row["window"], row["model"]


def expected_fits(merged: dict, summary: dict):
    expected = {}
    eq_max = int(summary["arguments"]["eq_max"])
    windows = [(canonical(a), canonical(b)) for a, b in summary["windows"]]
    models = ("log_inv1", "log_inv1_inv2", "log_inv1_inv2_inv3")
    for beta in map(canonical, summary["betas"]):
        equilibrium = {size: values["D"] for (b, size, time), values in merged.items()
                       if b == beta and time.is_infinite()}
        for low, model in itertools.product((18, 36, 54), models[:2]):
            grid = sorted(size for size in equilibrium if low <= size <= eq_max)
            if len(grid) >= len(MODEL_COLUMNS[model]) + 2:
                expected["equilibrium", beta, low, eq_max, model] = dict(
                    grid=grid, values=[equilibrium[size] for size in grid],
                    extent_complete=grid[0] == low and grid[-1] == eq_max)
        sizes = sorted({size for b, size, time in merged if b == beta and time.is_finite()})
        finite = {size: {time: values["D"] for (b, L, time), values in merged.items()
                        if b == beta and L == size and time.is_finite()} for size in sizes}
        for low, high in windows:
            name = f"{float(low):g}-{float(high):g}"
            for size in sizes:
                data = finite[size]
                grid = sorted(time for time in data if low <= time <= high)
                if not grid or grid[0] != low or grid[-1] != high:
                    continue
                for model in models:
                    if len(grid) >= len(MODEL_COLUMNS[model]) + 2:
                        expected["time_individual", beta, size, name, model] = dict(
                            grid=grid, values=[data[time] for time in grid])
            for small, large in itertools.combinations(sizes, 2):
                a, b = finite[small], finite[large]
                grid = sorted(time for time in a.keys() & b.keys() if low <= time <= high)
                if not grid or grid[0] != low or grid[-1] != high:
                    continue
                for model in models:
                    if len(grid) >= len(MODEL_COLUMNS[model]) + 2:
                        expected["time_pair", beta, small, large, name, model] = dict(
                            grid=grid, values=[b[time] for time in grid],
                            small_values=[a[time] for time in grid])
    return expected


def audit_fits(merged: dict, summary: dict, errors: list[str]):
    expected = expected_fits(merged, summary)
    actual = {}
    table_hashes = {}
    for domain, name in (("equilibrium", "equilibrium_fits.csv"),
                         ("time_individual", "time_individual_fits.csv"),
                         ("time_pair", "time_pair_fits.csv")):
        raw = (OUT / name).read_bytes()
        table_hashes[name] = sha256(raw)
        for row in csv_rows(raw):
            key = fit_key(domain, row)
            if key in actual:
                errors.append(f"Duplicate fitted row: {key}")
            actual[key] = row
    if expected.keys() != actual.keys():
        errors.append(f"Fit coverage mismatch: missing={sorted(expected.keys()-actual.keys())}, extra={sorted(actual.keys()-expected.keys())}")
    comparisons, coefficient_cache = [], {}
    largest = dict(c=D(0), parameter=D(0), residual=D(0), metric=D(0))
    aliases = {"d": "inv1", "e": "inv2", "f": "inv3"}
    theta_min = canonical(summary["arguments"]["theta_min"])
    ratio_max = D(str(summary["arguments"]["max_theta_over_L"]))
    for key, data in expected.items():
        fit = fit_decimal(data["grid"], data["values"], key[-1])
        coefficient_cache[key] = fit
        if key not in actual:
            continue
        row = actual[key]
        grid = [D(str(x)) for x in data["grid"]]
        if grid_values(row["grid"]) != grid:
            errors.append(f"Actual grid mismatch: {key}")
        for field, value in (("n", len(grid)), ("parameter_count", len(MODEL_COLUMNS[key[-1]])),
                             ("residual_dof", len(grid) - len(MODEL_COLUMNS[key[-1]]))):
            if int(row[field]) != value:
                errors.append(f"Fit count mismatch: {key}/{field}")
        parameter_errors = {}
        for name in MODEL_COLUMNS[key[-1]]:
            field = aliases.get(name, name)
            parameter_errors[name] = assert_close(errors, f"{key}/{field}", fit[name], row[field],
                                                 absolute=D("1e-9") if name == "c" else D("1e-7"))
        largest["c"] = max(largest["c"], parameter_errors["c"])
        largest["parameter"] = max(largest["parameter"], max(parameter_errors.values()))
        for field in ("rmse", "max_abs_residual"):
            error = assert_close(errors, f"{key}/{field}", fit[field], row[field], absolute=D("1e-10"))
            largest["residual"] = max(largest["residual"], error)
        for field, value in (("x_min", min(grid)), ("x_max", max(grid))):
            assert_close(errors, f"{key}/{field}", value, row[field], absolute=D(0), relative=D(0))
        if key[0] == "equilibrium":
            if (row["extent_complete"].lower() == "true") != data["extent_complete"]:
                errors.append(f"Equilibrium extent mismatch: {key}")
        with localcontext() as context:
            context.prec = PRECISION
            log_span = abs(fit["c"]) * (max(grid) / min(grid)).ln()
            correction = D(0)
            metrics = {}
            for power, name in ((1, "d"), (2, "e"), (3, "f")):
                if name in fit:
                    span = abs(fit[name]) * (min(grid)**(-power) - max(grid)**(-power))
                    correction += span
                    metrics[f"inverse{power}_to_log_span"] = ratio(span, log_span)
            metrics["total_correction_to_log_span"] = ratio(correction, log_span)
            if key[0] == "time_pair":
                small_fit = fit_decimal(grid, data["small_values"], key[-1])
                coefficient_cache[key + ("small",)] = small_fit
                raw_difference = max(abs(a - b) for a, b in zip(data["values"], data["small_values"]))
                metrics.update(c_small=small_fit["c"], size_drift=ratio(abs(small_fit["c"] - fit["c"]), abs(fit["c"])),
                               max_absolute_D_difference=raw_difference,
                               D_difference_to_log_span=ratio(raw_difference, log_span))
                boolean = {"size_sign_changed": fit["c"] * small_fit["c"] <= 0,
                           "scale_screen_large": min(grid) >= theta_min and max(grid)/key[3] <= ratio_max,
                           "scale_screen_both": min(grid) >= theta_min and max(grid)/key[2] <= ratio_max}
            elif key[0] == "time_individual":
                boolean = {"scale_screen": min(grid) >= theta_min and max(grid)/key[2] <= ratio_max}
            else:
                boolean = {}
            for field, value in metrics.items():
                difference = assert_close(errors, f"{key}/{field}", value, row[field], absolute=D("1e-7"))
                largest["metric"] = max(largest["metric"], difference)
            for field, value in boolean.items():
                if (row[field].lower() == "true") != value:
                    errors.append(f"Fit boolean mismatch: {key}/{field}")
        comparisons.append(dict(domain=key[0], identity=str(key[1:]), model=key[-1], grid=row["grid"],
                                n=len(grid), c_high_precision=str(fit["c"]), c_published=row["c"],
                                c_error=str(parameter_errors["c"]),
                                maximum_parameter_error=str(max(parameter_errors.values())),
                                rmse_high_precision=str(fit["rmse"])))
    return comparisons, coefficient_cache, expected, largest, table_hashes


def audit_coverage(merged: dict, summary: dict, errors: list[str]):
    expected = set()
    betas = summary["betas"]
    for grid in summary["plan"].get("required_finite_grids", []):
        expected.update((canonical(beta), int(size), canonical(time)) for size, time, beta in
                        itertools.product(grid.get("sizes", grid.get("L", [])),
                                          grid.get("times", grid.get("theta", [])),
                                          grid.get("betas", grid.get("beta", betas))))
    for grid in summary["plan"].get("required_equilibrium_grids", []):
        expected.update((canonical(beta), int(size), D("Infinity")) for size, beta in
                        itertools.product(grid.get("sizes", grid.get("L", [])),
                                          grid.get("betas", grid.get("beta", betas))))
    missing = expected - merged.keys()
    if len(expected) != summary["required_keys"] or len(missing) != summary["missing_required_keys"]:
        errors.append("Required coverage summary disagrees with raw sources")
    if summary["rows"] != len(merged):
        errors.append("Merged row count mismatch")
    published = read_csv(OUT / "observations.csv")
    actual = {(canonical(r["beta"]), int(r["L"]), canonical(r["theta"])): r for r in published}
    if len(actual) != len(published) or actual.keys() != merged.keys():
        errors.append("Merged observations keys mismatch")
    for key in actual.keys() & merged.keys():
        for name in ("S_A", "S_C", "S_B", "D"):
            assert_close(errors, f"merged{key}/{name}", merged[key][name], actual[key][name], absolute=D("1e-12"))
    return dict(expected=len(expected), missing=len(missing), merged=len(merged))


def grid_text(grid):
    return ",".join(f"{float(x):g}" for x in grid)


def time_screen(key, expected, summary, both=False):
    grid = expected[key]["grid"]
    size = key[2] if key[0] == "time_individual" or both else key[3]
    return min(grid) >= D(str(summary["arguments"]["theta_min"])) and max(grid) / size <= D(str(summary["arguments"]["max_theta_over_L"]))


def compare_record(actual, expected, identity, errors):
    maximum = D(0)
    for field, value in expected.items():
        raw = actual.get(field)
        if isinstance(value, bool):
            if raw is None or (raw.lower() == "true") != value:
                errors.append(f"Diagnostic boolean mismatch {identity}/{field}")
        elif value is None or isinstance(value, (int, float, Decimal)):
            maximum = max(maximum, assert_close(errors, f"{identity}/{field}", value, raw, absolute=D("1e-7")))
        elif raw != value:
            errors.append(f"Diagnostic label mismatch {identity}/{field}: {raw!r} != {value!r}")
    return maximum


def expected_primary(merged, summary, coefficients, expected):
    answer = {}
    eq_max = int(summary["arguments"]["eq_max"])
    configuration = summary["primary"]
    for beta in map(canonical, summary["betas"]):
        config = dict(configuration.get("default", {}))
        config.update(next((value for b, value in configuration.get("by_beta", {}).items()
                           if canonical(b) == beta), {}))
        all_eq = [key for key in expected if key[0] == "equilibrium" and key[1] == beta and expected[key]["extent_complete"]]
        anchor = ("equilibrium", beta, 18, eq_max, "log_inv1")
        eq_c = coefficients[anchor]["c"] if anchor in all_eq else None
        eq_range = [coefficients[key]["c"] for key in all_eq]
        row = dict(beta=beta, c_eq=eq_c,
                   eq_status="available" if eq_c is not None else "missing_complete_18_to_max_extent",
                   eq_range_min=min(eq_range) if eq_range else None,
                   eq_range_max=max(eq_range) if eq_range else None, eq_range_n=len(eq_range),
                   neq_status="primary_not_selected")
        answer[beta] = row
        if not config:
            continue
        window = "-".join(f"{x:g}" for x in config["window"])
        size, model = config["L"], config["model"]
        row.update(primary_L=size, primary_window=window, primary_model=model,
                   neq_status="missing_common_grid_fit")
        candidates = [key for key in expected if key[0] == "time_pair" and key[1] == beta and
                      key[3:] == (size, window, model) and
                      ("comparison_L" not in config or key[2] == config["comparison_L"])]
        if not candidates:
            continue
        key = max(candidates, key=lambda k: k[2])
        small = key[2]
        fit, small_fit = coefficients[key], coefficients[key + ("small",)]
        grid = [D(str(x)) for x in expected[key]["grid"]]
        c, cs = fit["c"], small_fit["c"]
        accepted = time_screen(key, expected, summary, both=True)
        size_drift = ratio(abs(c - cs), abs(c))
        log_span = abs(c) * (max(grid) / min(grid)).ln()
        delta = max(abs(a - b) for a, b in zip(expected[key]["values"], expected[key]["small_values"]))
        correction = sum(abs(fit[name]) * (min(grid)**(-power) - max(grid)**(-power))
                         for power, name in ((1, "d"), (2, "e"), (3, "f")) if name in fit)
        row.update(c_neq=c if accepted else None, c_neq_diagnostic=c,
                   neq_status="finite_window_estimate" if accepted else "excluded_by_operational_scale_screen",
                   comparison_L=small, grid=grid_text(grid), n=len(grid), size_drift=size_drift,
                   size_sign_changed=c * cs <= 0, theta_over_L=max(grid)/size,
                   theta_over_L_comparison=max(grid)/small, theta_over_region_width=3*max(grid)/size,
                   scale_screen_both=accepted, D_difference_to_log_span=ratio(delta, log_span),
                   correction_to_log_span=ratio(correction, log_span))
        models = [k for k in expected if k[0] == "time_pair" and k[1:5] == (beta, small, size, window)]
        windows = [k for k in expected if k[0] == "time_individual" and k[1:3] == (beta, size) and
                   k[-1] == model and time_screen(k, expected, summary)]
        band = [coefficients[k]["c"] for k in expected if k[0] == "time_individual" and
                k[1:3] == (beta, size) and time_screen(k, expected, summary)]
        md = [ratio(abs(coefficients[k]["c"] - c), abs(c)) for k in models]
        wd = [ratio(abs(coefficients[k]["c"] - c), abs(c)) for k in windows]
        row.update(model_drift=max((x for x in md if x is not None), default=None),
                   window_drift=max((x for x in wd if x is not None), default=None),
                   model_check_count=len(models), window_check_count=len(windows),
                   model_sign_changed=any(coefficients[k]["c"] * c <= 0 for k in models),
                   window_sign_changed=any(coefficients[k]["c"] * c <= 0 for k in windows),
                   window_check_scope="individual measured grids at primary L; each window has separate size-check status",
                   neq_range_min=min(band) if band and accepted else None,
                   neq_range_max=max(band) if band and accepted else None, neq_range_n=len(band))
        available_sizes = {L for b, L, t in merged if b == beta and t.is_finite()}
        alternatives = [L for L in available_sizes - {small, size} if all((beta, L, t) in merged for t in grid)
                        and min(grid) >= D(str(summary["arguments"]["theta_min"])) and
                        max(grid)/L <= D(str(summary["arguments"]["max_theta_over_L"]))]
        if alternatives:
            larger = [L for L in alternatives if L > size]
            third = min(larger) if larger else max(alternatives)
            third_fit = fit_decimal(grid, [merged[beta, third, t]["D"] for t in grid], model)
            ct = third_fit["c"]
            extra = ratio(abs(ct - c), abs(ct)) if third > size else ratio(abs(ct - cs), abs(cs))
            decreasing = None if extra is None or size_drift is None else (extra < size_drift if third > size else size_drift < extra)
            row.update(third_L=third, c_third=ct, additional_size_drift=extra,
                       third_size_position="above_primary" if third > size else "below_comparison",
                       size_drift_decreasing=decreasing)
        available = len(models) == 3 and len(windows) >= 2
        third_missing = "third_L" not in row
        third_pass = third_missing or (row.get("additional_size_drift") is not None and
                     row["additional_size_drift"] <= D(".02") and row.get("size_drift_decreasing") is True)
        passed = accepted and available and third_pass and all(row[field] is not None and row[field] <= limit
                  for field, limit in (("size_drift", D(".02")), ("D_difference_to_log_span", D(".01")),
                                       ("model_drift", D(".05")), ("window_drift", D(".05")))) and not any(
                  row[field] for field in ("size_sign_changed", "model_sign_changed", "window_sign_changed"))
        row.update(local_diagnostic_pass=passed,
                   reliability="local_checks_only_not_certified_asymptotic" if passed else "unresolved_diagnostic_estimate",
                   third_size_missing=third_missing,
                   third_size_status="missing_two_size_local_check_only" if third_missing else
                       "checked_stable_and_decreasing" if third_pass else "checked_unstable_or_not_decreasing",
                   third_size_gate=None if third_missing else third_pass,
                   narrow_window_warning=max(grid)/min(grid) < 2,
                   five_parameter_residual_dof=len(grid)-5, local_diagnostic_missing_checks=not available)
        if eq_c is not None and accepted:
            row.update(ratio_neq_to_eq=ratio(c, eq_c), relative_gap=ratio(c-eq_c, abs(eq_c)))
    return answer


def audit_diagnostics(merged, summary, coefficients, expected, errors):
    primary = expected_primary(merged, summary, coefficients, expected)
    primary_path = OUT / "primary_coefficients.csv"
    primary_raw = primary_path.read_bytes()
    actual = {canonical(r["beta"]): r for r in csv_rows(primary_raw)}
    if actual.keys() != primary.keys():
        errors.append("Primary coefficient beta coverage mismatch")
    maximum = D(0)
    for beta, row in primary.items():
        if beta in actual:
            maximum = max(maximum, compare_record(actual[beta], row, ("primary", beta), errors))
    sensitivity = {}
    eq_max = int(summary["arguments"]["eq_max"])
    for beta, selected in primary.items():
        anchor = ("equilibrium", beta, 18, eq_max, "log_inv1")
        if anchor in expected:
            reference = coefficients[anchor]["c"]
            for key in expected:
                if key[0] != "equilibrium" or key[1] != beta:
                    continue
                row = dict(domain="equilibrium", beta=beta, model=key[-1], window=f"{key[2]}-{key[3]}",
                           grid=grid_text(expected[key]["grid"]), c=coefficients[key]["c"], reference_c=reference,
                           relative_drift=ratio(abs(coefficients[key]["c"] - reference), abs(reference)),
                           extent_complete=expected[key]["extent_complete"],
                           reference_extent_complete=expected[anchor]["extent_complete"])
                sensitivity["equilibrium", beta, row["window"], key[-1]] = row
        if selected.get("c_neq_diagnostic") is None:
            continue
        size, small = selected["primary_L"], selected["comparison_L"]
        reference = selected["c_neq_diagnostic"]
        for key in expected:
            if key[0] != "time_individual" or key[1:3] != (beta, size):
                continue
            window, model = key[3:]
            grid, c = expected[key]["grid"], coefficients[key]["c"]
            pair = ("time_pair", beta, small, size, window, model)
            common = pair in expected
            same_window, same_model = window == selected["primary_window"], model == selected["primary_model"]
            status = ("available_within_both_size_screens" if time_screen(pair, expected, summary, both=True)
                      else "available_but_comparison_size_outside_screen") if common else "missing_common_grid_fit"
            row = dict(domain="time", beta=beta, L=size, model=model, window=window,
                       grid=grid_text(grid), n=len(grid), residual_dof=len(grid)-len(MODEL_COLUMNS[model]),
                       c=c, reference_c=reference, reference_grid=selected["grid"],
                       relative_drift=ratio(abs(c-reference), abs(reference)),
                       comparison_kind="primary_full_grid_check" if same_window and same_model else
                                       "model" if same_window else "window" if same_model else "window_and_model",
                       scale_screen_at_primary_L=time_screen(key, expected, summary),
                       theta_over_L=max(grid)/size, comparison_L=small, size_check_status=status,
                       size_check_grid=grid_text(expected[pair]["grid"]) if common else None,
                       size_drift=ratio(abs(coefficients[pair]["c"]-coefficients[pair+("small",)]["c"]), abs(coefficients[pair]["c"])) if common else None,
                       scale_screen_both_sizes=time_screen(pair, expected, summary, both=True) if common else None)
            sensitivity["time", beta, window, model] = row
    sensitivity_path = OUT / "time_sensitivity.csv"
    sensitivity_raw = sensitivity_path.read_bytes()
    actual_sensitivity = {(r["domain"], canonical(r["beta"]), r["window"], r["model"]): r for r in csv_rows(sensitivity_raw)}
    if actual_sensitivity.keys() != sensitivity.keys():
        errors.append("Sensitivity table key coverage mismatch")
    for key, row in sensitivity.items():
        if key in actual_sensitivity:
            maximum = max(maximum, compare_record(actual_sensitivity[key], row, key, errors))
    write_csv(OUT / "independent_fit_validation_primary.csv", list(primary.values()))
    write_csv(OUT / "independent_fit_validation_sensitivity.csv", list(sensitivity.values()))
    return dict(primary_rows=len(primary), sensitivity_rows=len(sensitivity), maximum_numeric_error=str(maximum)), {
        "primary_coefficients.csv": sha256(primary_raw), "time_sensitivity.csv": sha256(sensitivity_raw)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("preliminary", "final"), default="preliminary")
    args = parser.parse_args()
    summary_path = OUT / "summary.json"
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    errors = []
    if args.stage == "final" and summary["stage"] != "final":
        errors.append("Final audit requested for a non-final primary analysis")
    script_path = AI / "analyze_uniformA_dense.py"
    if sha256(script_path.read_bytes()) != summary["script_sha256"]:
        errors.append("Primary analysis code changed since summary")
    plan_path = summary["arguments"].get("plan")
    if plan_path and sha256(Path(plan_path).read_bytes()) != summary["plan_sha256"]:
        errors.append("Primary plan changed since summary")
    merged, sources = load_sources(summary, errors, args.stage)
    coverage = audit_coverage(merged, summary, errors)
    if args.stage == "final" and coverage["missing"]:
        errors.append("Final audit has missing required observations")
    comparisons, coefficients, expected, maxima, hashes = audit_fits(merged, summary, errors)
    diagnostics, diagnostic_hashes = audit_diagnostics(merged, summary, coefficients, expected, errors)
    hashes.update(diagnostic_hashes)
    if sha256(summary_path.read_bytes()) != sha256(summary_raw):
        errors.append("Primary summary changed during audit")
    for name, digest in hashes.items():
        if sha256((OUT / name).read_bytes()) != digest:
            errors.append(f"Primary fit table changed during audit: {name}")
    write_csv(OUT / "independent_fit_validation_fits.csv", comparisons)
    receipt = dict(status="PASS" if not errors else "FAIL", stage=args.stage,
                   generated_utc=datetime.now(timezone.utc).isoformat(), precision_digits=PRECISION,
                   primary_summary_sha256=sha256(summary_raw), primary_script_sha256=summary["script_sha256"],
                   plan_sha256=summary["plan_sha256"], audit_script_sha256=sha256(Path(__file__).read_bytes()),
                   source_receipts=sources, fit_table_sha256=hashes, coverage=coverage,
                   fit_count=len(comparisons), maxima={k: str(v) for k, v in maxima.items()}, errors=errors,
                   diagnostics=diagnostics,
                   scope="Independent raw-source merge, fit coverage, 80-digit unweighted least squares, common grids, residuals and time size diagnostics. No simulation rerun; no claim that finite-window fits are asymptotic coefficients.")
    (OUT / "independent_fit_validation.json").write_text(json.dumps(receipt, indent=2)+"\n", encoding="utf-8")
    print(json.dumps({k: receipt[k] for k in ("status", "stage", "coverage", "fit_count", "maxima", "errors")}, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
