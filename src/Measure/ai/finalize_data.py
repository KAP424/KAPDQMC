"""Merge the completed large-L/time extensions, validate coverage, and record hashes.

Run after both Julia scans have completed. Re-running is idempotent; conflicting
duplicate rows are rejected. All paths are relative to this script, not cwd.
"""
from pathlib import Path
import csv
import hashlib
import io
import json
import math
from datetime import datetime, timezone

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def read_csv(path):
    raw = path.read_bytes()
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    return reader.fieldnames, list(reader), hashlib.sha256(raw).hexdigest()


def key(row):
    return row["pattern"], int(row["L"]), float(row["theta"]), float(row["beta"])


def expected_keys(patterns):
    result = set()
    finite_patterns = {"uniform_A", "stripe_AB_x", "full_empty_checker", "two_B_3", "random_half_3"}
    times = (0, .25, .5, .75, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 8, 12, 16)
    for row in patterns:
        name, period = row["pattern"], int(row["m"])
        if name == "uniform_A":
            sizes = [12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 90]
        elif name == "uniform_B":
            sizes = [12, 24, 36]
        else:
            sizes = list(range(3 * period, 55, 3 * period))
        result.update((name, size, math.inf, beta)
                      for size in sizes for beta in (0, .25, .5, 1, 2, 4))
        result.add((name, max(sizes), math.inf, 8))
        if name in finite_patterns:
            sizes = [18, 30, 42, 54, 72] if name == "uniform_A" else [18, 36, 54]
            result.update((name, size, theta, beta)
                          for size in sizes for theta in times for beta in (0, .5, 1, 2))
    result.update(("uniform_A", size, theta, beta)
                  for size in (90, 108) for theta in (2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6, 7, 8, 10, 12, 16)
                  for beta in (0, .5, 1, 2))
    return result


def main():
    target = DATA / "observations.csv"
    fields, base, before_hash = read_csv(target)
    extensions, extension_hashes = [], {}
    for filename, expected_rows in (("extended_observations.csv", 88), ("late_time_observations.csv", 24)):
        extension_fields, extension, extension_hash = read_csv(DATA / filename)
        if fields != extension_fields:
            raise ValueError(f"Main and {filename} CSV schemas differ")
        if len(extension) != expected_rows:
            raise ValueError(f"{filename} incomplete: {len(extension)} of {expected_rows} rows")
        extensions.append(extension)
        extension_hashes[filename] = extension_hash
    combined = {}
    for source in [base] + extensions:
        seen = set()
        for row in source:
            identity = key(row)
            if identity in seen:
                raise ValueError(f"Duplicate within input: {identity}")
            seen.add(identity)
            if identity in combined and combined[identity] != row:
                raise ValueError(f"Conflicting duplicate: {identity}")
            combined[identity] = row
    _, patterns, _ = read_csv(DATA / "patterns.csv")
    expected = expected_keys(patterns)
    if set(combined) != expected:
        raise ValueError(f"Coverage mismatch: missing={expected-set(combined)}, extra={set(combined)-expected}")
    maximum_identity_error = 0.
    for identity, row in combined.items():
        name, size, theta, beta = identity
        if size % (3 * int(row["m"])):
            raise ValueError(f"Mismatched cut phase: {identity}")
        for column in fields[6:]:
            if not math.isfinite(float(row[column])):
                raise ValueError(f"Non-finite {column}: {identity}")
        a, c, b, delta, d = (float(row[column]) for column in ("S_A", "S_C", "S_B", "delta", "D"))
        maximum_identity_error = max(maximum_identity_error, abs(a-b-delta), abs((2*a+c)/3-b-d))
        if min(a, b, c) < -1e-9:
            raise ValueError(f"Negative entropy: {identity}")
        if float(row["projector_error"]) > 1e-10 or float(row["reconstruction_error"]) > 1e-10:
            raise ValueError(f"Failed numerical diagnostic: {identity}")
        if theta == 0 and max(abs(a), abs(b), abs(c)) > 1e-12:
            raise ValueError(f"Product state has nonzero entropy: {identity}")
        if theta == math.inf and beta == 0 and float(row["residual"]) > 1e-9:
            raise ValueError(f"Infinite-time limit not stationary: {identity}")
    if maximum_identity_error > 1e-10:
        raise ValueError("Entropy-difference identities failed")
    # Stable order, with full finite-time scans followed by the limit at each L.
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(combined[identity] for identity in sorted(combined))
    blob = output.getvalue().encode("utf-8")
    temporary = target.with_suffix(".csv.tmp")
    temporary.write_bytes(blob)
    temporary.replace(target)
    hashes = {}
    for path in sorted(HERE.glob("*.jl")) + sorted(HERE.glob("*.py")) + sorted(DATA.glob("*.csv")):
        hashes[path.relative_to(HERE).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "input_main_sha256": before_hash,
        "input_extension_sha256": extension_hashes,
        "merged_sha256": hashlib.sha256(blob).hexdigest(),
        "rows": len(combined),
        "equilibrium_rows": sum(math.isinf(identity[2]) for identity in combined),
        "finite_time_rows": sum(math.isfinite(identity[2]) for identity in combined),
        "extension_rows": sum(map(len, extensions)),
        "exact_requested_parameter_coverage": True,
        "maximum_entropy_identity_error": maximum_identity_error,
        "max_projector_error": max(float(r["projector_error"]) for r in combined.values()),
        "max_reconstruction_error": max(float(r["reconstruction_error"]) for r in combined.values()),
        "minimum_initial_sv": min(float(r["min_initial_sv"]) for r in combined.values()),
        "sha256": hashes,
    }
    (DATA / "validation_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "sha256"}, indent=2))


if __name__ == "__main__":
    main()
