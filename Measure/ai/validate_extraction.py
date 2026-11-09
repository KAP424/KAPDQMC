"""Independently audit the finite-time extraction scan and its row provenance."""
from __future__ import annotations

import csv
import hashlib
import itertools
import json
import math
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def key(row):
    return row["pattern"], int(row["L"]), float(row["theta"]), float(row["beta"])


def main():
    path = DATA / "extraction_observations.csv"
    provenance = DATA / "extraction_provenance.tsv"
    manifest = DATA / "extraction_manifest.txt"
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    observations = list(csv.DictReader(lines))
    journal = list(csv.DictReader(provenance.open(encoding="utf-8-sig", newline=""), delimiter="\t"))
    assert len(observations) == len(journal)
    assert len({key(r) for r in observations}) == len(observations), "duplicate observation"
    entries = {key(r): r for r in journal}
    assert len(entries) == len(journal), "duplicate provenance"
    catalogues = [DATA / "patterns.csv", DATA / "arrangement_selection.csv"]
    pattern_bits = {}
    for catalogue in catalogues:
        for row in csv.DictReader(catalogue.open(encoding="utf-8-sig", newline="")):
            name, bits = row["pattern"], row["occupied_bits"]
            assert name not in pattern_bits or pattern_bits[name] == bits
            pattern_bits[name] = bits
    source_bytes = {}
    maxima = defaultdict(float)
    source_counts = Counter()
    groups = defaultdict(set)
    for line, row in zip(lines[1:], observations):
        identity = key(row)
        entry = entries[identity]
        assert line == entry["observation_csv"]
        row_hash = hashlib.sha256(line.encode("utf-8")).hexdigest()
        assert row_hash == entry["row_sha256"] == entry["source_row_sha256"]
        m, L = int(row["m"]), int(row["L"])
        assert L % (3 * m) == 0
        bits = entry["pattern_bits"]
        assert bits == pattern_bits[row["pattern"]], "named configuration differs from catalogue"
        assert len(bits) == 2 * m * m and bits.count("1") == m * m
        special = all(bits[j:j+2].count("1") == 1 for j in range(0, len(bits), 2))
        assert special == (row["family"] == "special")
        numeric = {k: float(v) for k, v in row.items() if k not in ("pattern", "family")}
        assert all(math.isfinite(x) for x in numeric.values())
        assert numeric["theta"] >= 0 and numeric["beta"] >= 0
        assert min(numeric[k] for k in ("S_A", "S_C", "S_B", "empty_occupancy")) >= -1e-9
        sa, sc, sb = (numeric[k] for k in ("S_A", "S_C", "S_B"))
        identity_error = max(abs(numeric["delta"] - sa + sb),
                             abs(numeric["D"] - ((2 * sa + sc) / 3 - sb)))
        maxima["entropy_identity_error"] = max(maxima["entropy_identity_error"], identity_error)
        assert identity_error < 1e-10
        for name, tolerance in (("projector_error", 1e-10), ("reconstruction_error", 1e-9)):
            maxima[name] = max(maxima[name], abs(numeric[name]))
            assert abs(numeric[name]) < tolerance
        limit_excess = float(entry["premeasurement_limit_excess"])
        limit_residual = float(entry["premeasurement_limit_residual"])
        assert abs(limit_excess - numeric["accessible_excess"]) < 1e-7
        assert (entry["premeasurement_ground"] == "true") == (abs(limit_excess) < 1e-8)
        assert limit_residual < 1e-9
        maxima["limit_stationarity_residual"] = max(maxima["limit_stationarity_residual"], limit_residual)
        source = Path(entry["source_file"])
        if source not in source_bytes:
            source_bytes[source] = source.read_bytes()
        raw = source_bytes[source]
        assert hashlib.sha256(raw).hexdigest() == entry["source_file_sha256"]
        if entry["source"] == "reused":
            old_line = raw.decode("utf-8-sig").splitlines()[int(entry["source_line"]) - 1]
            assert old_line == line, "reused observation changed"
        else:
            assert entry["source"] == "computed"
            assert entry["source_file_sha256"] == entry["code_sha256"]
        source_counts[entry["source"]] += 1
        groups[(row["pattern"], float(row["beta"]))].add(L)

    # Every explicitly requested grid in the run receipt must be present, including
    # supplementary runs. This does not infer a rectangular grid across all runs.
    observed = set(entries)
    requested_grids = []
    code_hashes = {}
    for line in manifest.read_text(encoding="utf-8-sig").splitlines():
        if line.startswith("CODE_SHA256 "):
            _, filename, digest = line.split()
            assert sha(HERE / filename) == digest, f"code changed: {filename}"
            assert filename not in code_hashes or code_hashes[filename] == digest
            code_hashes[filename] = digest
        if not line.startswith("patterns="):
            continue
        fields = dict(part.strip().split("=", 1) for part in line.split(";") if "=" in part)
        patterns = fields["patterns"].split(":")
        sizes = list(map(int, fields["L"].split(":")))
        times = list(map(float, fields["theta"].split(":")))
        betas = list(map(float, fields["beta"].split(":")))
        requested = set(itertools.product(patterns, sizes, times, betas))
        assert requested <= observed, f"incomplete requested grid: {fields}"
        requested_grids.append({"patterns": patterns, "L": sizes, "theta": times,
                                "beta": betas, "requested_rows": len(requested)})
    assert requested_grids
    assert set(code_hashes) == {"fermion_experiment.jl", "experiment_patterns.jl",
                                "arrangement_patterns.jl", "extraction_experiments.jl"}
    report = {
        "validated_utc": datetime.now(timezone.utc).isoformat(),
        "rows": len(observations), "duplicates": 0,
        "source_counts": dict(source_counts),
        "ground_limit_rows": sum(e["premeasurement_ground"] == "true" for e in journal),
        "complete_requested_grids": True, "provenance_rows_match": True,
        "reused_rows_identical": True, "source_hashes_match": True,
        "generating_code_hashes_match": True, "named_pattern_bits_match": True,
        "numerical_maxima": dict(maxima), "requested_grids": requested_grids,
        "sizes_by_pattern_beta": [dict(pattern=p, beta=b, sizes=sorted(s))
                                  for (p, b), s in sorted(groups.items())],
        "sha256": {p.name: sha(p) for p in (path, provenance, manifest)},
        "source_sha256": {str(p): hashlib.sha256(raw).hexdigest() for p, raw in source_bytes.items()},
        "code_sha256": code_hashes,
        "catalogue_sha256": {p.name: sha(p) for p in catalogues},
        "scope": "Raw-data provenance and diagnostics audit; superseded extraction classifications have been removed.",
        "validator_sha256": sha(Path(__file__)),
    }
    output = DATA / "extraction_validation.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"PASS: {len(observations)} observations; {dict(source_counts)}; "
          f"{len(requested_grids)} complete requested grids. Receipt: {output}")


if __name__ == "__main__":
    main()
