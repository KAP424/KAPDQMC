"""Independent arithmetic, provenance, coverage and numerical checks for the targeted scan."""
from pathlib import Path
import csv
import hashlib
import json
import math
import tomllib

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
OUT = ROOT / "analysis_random_half3_beta1"
BASE_TIMES = [2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
LARGE_TIMES = [4, 5, 6, 7, 8, 10, 12, 14, 16, 18, 20, 24, 28, 32]
LOCKED_SOURCES = {
    "data/observations.csv": "b6e6403a7cf5a228b77f37c6f167bff7da93cc800f768de662a0ae9be34666dc",
    "data/extraction_observations.csv": "aabd6735c4e8ab7d5f8ea74181a4b24b7b32508ac5ed18e9a0919ea752b93b37",
    "data/equilibrium_extended.csv": "8d26ae7ebb69e7b378b62e40329f749f266288634a944e0cce36c73fca7ecb23",
    "data/equilibrium_arrangements.csv": "f6be3cd1d0d03622d278ad793ad56556240e851014417053cd4889d4d61b073b",
    "fermion_experiment.jl": "9efd053e7d469082ad8cead86ec256dade17e40279902842013c9a8dc9864fab",
}
# The removed classification table was recorded by the original scan. It is
# neither a source of entropy data nor a requirement for validating this scan.
HISTORICAL_ARTIFACTS = {
    "analysis_extraction/support_summary.csv": "fb4f6ba5ab403098383002a542a6c49ea717fecc0ce07eecee3d9b2b0884a518",
}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_rows(path):
    assert path.read_bytes().endswith(b"\n"), f"unfinished row: {path}"
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    keys = [(int(x["L"]), float(x["theta"])) for x in rows]
    assert len(keys) == len(set(keys)), f"duplicate keys: {path}"
    return rows


def main():
    for path, expected in LOCKED_SOURCES.items():
        assert sha(ROOT / path) == expected, f"locked source changed: {path}"
    historical_records = []
    for path, expected in HISTORICAL_ARTIFACTS.items():
        current = sha(ROOT / path) if (ROOT / path).is_file() else None
        historical_records.append(dict(path=path, historical_sha256=expected,
                                       current_sha256=current, currently_present=current is not None,
                                       matches_historical_hash=current == expected if current is not None else None,
                                       required_for_validation=False))
    paths = [DATA / "random_half3_beta1_observations.csv", DATA / "random_half3_beta1_equilibrium.csv"]
    grouped = {str(p.resolve()): read_rows(p) for p in paths}
    rows = [x for rr in grouped.values() for x in rr]
    combined = {}
    maxima = dict(D_arithmetic=0.0, delta_arithmetic=0.0, ground_excess=0.0,
                  ground_residual=0.0, projector_error=0.0, reconstruction_error=0.0,
                  rectangle_imag_error=0.0, rectangle_real_error_bound=0.0)
    for x in rows:
        assert x["pattern"] == "random_half_3" and x["family"] == "general"
        assert int(x["m"]) == 3 and int(x["L"]) % 9 == 0
        assert float(x["beta"]) == 1.0 and float(x["rank_tol"]) == 1e-10
        assert x["premeasurement_ground"] == "true"
        assert x["entropy_method"] in ("eigen", "cholesky")
        for col in ("S_A", "S_C", "S_B", "D", "energy", "accessible_excess"):
            assert math.isfinite(float(x[col])), col
        sa, sc, sb = (float(x[k]) for k in ("S_A", "S_C", "S_B"))
        assert min(sa, sc, sb) >= 0
        maxima["D_arithmetic"] = max(maxima["D_arithmetic"], abs((2*sa+sc)/3-sb-float(x["D"])))
        maxima["delta_arithmetic"] = max(maxima["delta_arithmetic"], abs(sa-sb-float(x["delta"])))
        for dest, col in (("ground_excess", "premeasurement_limit_excess"),
                          ("ground_residual", "premeasurement_limit_residual"),
                          ("projector_error", "projector_error"),
                          ("reconstruction_error", "reconstruction_error"),
                          ("rectangle_imag_error", "rectangle_imag_error"),
                          ("rectangle_real_error_bound", "rectangle_real_error_bound")):
            maxima[dest] = max(maxima[dest], abs(float(x[col])))
        assert abs(float(x["accessible_excess"])-float(x["premeasurement_limit_excess"])) < 1e-7
        key = int(x["L"]), float(x["theta"])
        if key in combined:
            assert all(abs(float(x[k])-float(combined[key][k])) < 1e-8 for k in ("S_A", "S_C", "S_B", "D"))
        combined[key] = x
    for col, threshold in dict(D_arithmetic=1e-10, delta_arithmetic=1e-10,
                               ground_excess=1e-8, ground_residual=1e-9,
                               projector_error=1e-10, reconstruction_error=1e-9,
                               rectangle_imag_error=2e-10, rectangle_real_error_bound=1e-9).items():
        assert maxima[col] < threshold, (col, maxima[col])

    required = {(L, float(t)) for L in (90, 108, 144) for t in BASE_TIMES}
    required |= {(L, float(t)) for L in (180, 216) for t in LARGE_TIMES}
    required |= {(L, float(t)) for L in (144, 180, 216) for t in (9, 11)}
    required |= {(L, math.inf) for L in range(99, 181, 9)}
    required |= {(198, math.inf), (216, math.inf)}
    assert required <= combined.keys(), f"missing coverage: {sorted(required-combined.keys())}"

    runs = {}
    for p in sorted(DATA.glob("random_half3_beta1_run_*.toml")):
        m = tomllib.loads(p.read_text(encoding="utf-8"))
        assert m["status"] == "complete", p.name
        assert m["pattern_bits"] == "100110001011010110"
        assert m["beta"] == 1.0 and m["rank_tol"] == 1e-10
        for filename, expected in m["code_sha256"].items():
            assert sha(ROOT / filename) == expected, filename
        file_rows = grouped[str(Path(m["output"]).resolve())]
        file_keys = {(int(x["L"]), float(x["theta"])) for x in file_rows}
        assert {(L, float(t)) for L in m["L"] for t in m["theta"]} <= file_keys
        run_rows = [x for x in file_rows if x["run_id"] == m["run_id"]]
        assert len(run_rows) == m["new_rows"]
        assert all(x["code_sha256"] == m["code_sha256"]["random_half3_beta1_experiments.jl"] for x in run_rows)
        runs[m["run_id"]] = {"manifest": p.name, "manifest_sha256": sha(p), "rows": len(run_rows)}
    assert all(x["run_id"] in runs for x in rows)

    audit_path = DATA / "random_half3_beta1_solver_audit.csv"
    audit = list(csv.DictReader(audit_path.open(encoding="utf-8-sig", newline="")))
    assert len(audit) == 48
    audit_manifest = tomllib.loads((DATA / "random_half3_beta1_solver_audit.toml").read_text(encoding="utf-8"))
    for name, expected in audit_manifest["sha256"].items():
        target = DATA / name if name.endswith(".csv") else ROOT / name
        assert sha(target) == expected, f"solver audit source changed: {name}"
    audit_maxima = {}
    for column, threshold in {
        "normalized_projector_difference": 1e-10,
        "max_block_projector_difference": 1e-9,
        "energy_difference": 1e-7,
        "premeasurement_limit_excess": 1e-8,
        "premeasurement_limit_residual": 1e-9,
        "reconstruction_error": 1e-9,
        "orthogonality_error": 1e-10,
    }.items():
        audit_maxima[column] = max(abs(float(x[column])) for x in audit)
        assert audit_maxima[column] < threshold, (column, audit_maxima[column])
    assert all(x["original_occupied_rows_preserved"] == "true" for x in audit)
    OUT.mkdir(exist_ok=True)
    receipt = dict(status="PASS", new_rows=len(rows), distinct_new_keys=len(combined),
                   coverage_checked=len(required), maxima=maxima, runs=runs,
                   locked_sources_unchanged=LOCKED_SOURCES,
                   historical_artifacts=historical_records,
                   sources=[dict(path=str(p.relative_to(ROOT)), sha256=sha(p)) for p in paths],
                   solver_audit=dict(path=str(audit_path.relative_to(ROOT)), sha256=sha(audit_path),
                                     rows=len(audit), maxima=audit_maxima),
                   script_sha256=sha(Path(__file__)),
                   note="No fitted coefficient or equilibrium agreement is used by these validation checks.")
    (OUT / "independent_validation.json").write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"PASS: {len(rows)} new rows, {len(required)} requested keys, source hashes and numerical checks.")
    print(json.dumps(maxima, indent=2))


if __name__ == "__main__":
    main()
