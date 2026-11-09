"""Validate the complete second-round equilibrium scans and their provenance."""
import csv
import hashlib
import json
import math
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
GS_CANDIDATES = {
    "uniform_A", "uniform_B", "one_B_2", "one_B_3", "two_B_3",
    "diagonal_B_3", "full_empty_cluster3", "random_half_3",
}
BETAS = (0., .25, .5, .75, 1., 1.5, 2., 3., 4.)


def read(name):
    with (DATA / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def key(row):
    return row["pattern"], int(row["L"]), float(row["beta"])


def main():
    original = {key(r): r for r in read("observations.csv") if math.isinf(float(r["theta"]))}
    expected = set()
    for p in read("patterns.csv"):
        name, period = p["pattern"], int(p["m"])
        if name in GS_CANDIDATES:
            sizes = range(12, 91, 6) if period == 1 else range(3*period, 91, 3*period)
        else:
            sizes = list(range(6, 55, 6)) + [72, 90]
        expected.update((name, size, beta) for size in sizes for beta in BETAS)
    expected.update(k for k in original if k[2] == 8)
    control_expected = {(r["pattern"], size, beta) for r in read("arrangement_selection.csv")
                        for size in range(9, 91, 9) for beta in (0., .5, 1., 2.)}
    files = [("equilibrium_extended.csv", expected),
             ("equilibrium_arrangements.csv", control_expected)]
    rows, hashes, counts = [], {}, {}
    for filename, requested in files:
        block = read(filename)
        identities = [key(r) for r in block]
        if len(set(identities)) != len(identities):
            raise ValueError(f"Duplicate row: {filename}")
        if set(identities) != requested:
            raise ValueError(f"Coverage mismatch in {filename}: missing={requested-set(identities)}, extra={set(identities)-requested}")
        counts[filename] = len(block)
        hashes[filename] = hashlib.sha256((DATA / filename).read_bytes()).hexdigest()
        rows.extend(block)
    classifications, fingerprints = {}, {}
    max_identity_error = max_projector_error = max_limit_residual = 0.
    provenance_hashes = {}
    for row in rows:
        identity = key(row)
        if float(row["theta"]) != math.inf:
            raise ValueError(f"Not an infinite-time state: {identity}")
        if int(row["L"]) % (3*int(row["m"])):
            raise ValueError(f"Cut phase mismatch: {identity}")
        a, c, b, delta, d = (float(row[k]) for k in ("S_A", "S_C", "S_B", "delta", "D"))
        numeric = ("S_A", "S_C", "S_B", "D", "energy", "excess_energy",
                   "premeasurement_energy", "premeasurement_excess",
                   "premeasurement_residual", "ground_energy",
                   "projector_error", "reconstruction_error", "min_initial_sv")
        if not all(math.isfinite(float(row[k])) for k in numeric):
            raise ValueError(f"Non-finite observation: {identity}")
        if min(a, b, c) < -1e-9:
            raise ValueError(f"Negative entropy: {identity}")
        actual_ground = abs(float(row["premeasurement_excess"])) < 1e-8
        if (row["premeasurement_ground"] == "true") != actual_ground:
            raise ValueError(f"Misclassified ground state: {identity}")
        if float(row["premeasurement_excess"]) < -1e-8:
            raise ValueError(f"Energy below ground state: {identity}")
        if abs(float(row["premeasurement_energy"]) - float(row["ground_energy"]) -
               float(row["premeasurement_excess"])) > 1e-8:
            raise ValueError(f"Energy identity failed: {identity}")
        group = identity[:2]
        if group in classifications and classifications[group] != actual_ground:
            raise ValueError(f"Ground-state classification changed with beta: {identity}")
        classifications[group] = actual_ground
        if group in fingerprints and fingerprints[group] != row["limit_fingerprint"]:
            raise ValueError(f"Unmeasured state changed with beta: {identity}")
        fingerprints[group] = row["limit_fingerprint"]
        maximum = max(abs(a-b-delta), abs((2*a+c)/3-b-d))
        max_identity_error = max(max_identity_error, maximum)
        max_projector_error = max(max_projector_error, float(row["projector_error"]))
        max_limit_residual = max(max_limit_residual, float(row["premeasurement_residual"]))
        if float(row["reconstruction_error"]) > 1e-10:
            raise ValueError(f"Initial-state reconstruction failed: {identity}")
        source = row["source_file"]
        source_path = DATA / source if row["source"] == "reused" else HERE / source
        if source not in provenance_hashes:
            provenance_hashes[source] = hashlib.sha256(source_path.read_bytes()).hexdigest()
        if provenance_hashes[source] != row["source_sha256"]:
            raise ValueError(f"Source has changed since this row was generated: {source}, {identity}")
        if row["source"] == "reused":
            old = original[identity]
            if any(old[k] != row[k] for k in old):
                raise ValueError(f"Reused observation changed: {identity}")
    if max_identity_error > 1e-9 or max_projector_error > 1e-10 or max_limit_residual > 1e-9:
        raise ValueError("Failed entropy/projector/stationarity check")
    source_counts = dict(Counter(r["source"] for r in rows))
    ground_rows = sum(abs(float(r["premeasurement_excess"])) < 1e-8 for r in rows)
    summary = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(rows), "rows_by_file": counts,
        "source_counts": source_counts,
        "ground_state_rows": ground_rows,
        "blocked_control_rows": len(rows)-ground_rows,
        "patterns": sorted({r["pattern"] for r in rows}),
        "complete_requested_grid": True,
        "source_hashes_match": True,
        "reused_rows_identical_to_original": True,
        "unmeasured_state_fixed_across_beta": True,
        "max_entropy_identity_error": max_identity_error,
        "max_projector_error": max_projector_error,
        "max_unmeasured_stationarity_residual": max_limit_residual,
        "sha256": hashes, "provenance_sha256": provenance_hashes,
        "pattern_ground_sizes": {
            name: [size for (p, size), ground in sorted(classifications.items()) if p == name and ground]
            for name in sorted({r["pattern"] for r in rows})
        },
    }
    target = DATA / "equilibrium_validation.json"
    target.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k != "pattern_ground_sizes"}, indent=2))


if __name__ == "__main__":
    main()
