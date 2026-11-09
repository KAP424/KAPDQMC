"""Path-schema compatibility entrypoint for the frozen boundary analyzer.

The production plan binds analyze_born_boundary_check.py at SHA256
d29682d79e29cacf278ca5b7c93ea758f7a768ff3ad86e35bc2089dcda8e9925.
That file and all production plans/data are preserved. Its evidence reader
mistakes bare source filenames inside pilot_summary.files for sibling files.
This entrypoint changes only evidence-path interpretation using explicit receipt
schemas; all sampling, statistics, fits, eligibility rules and plotting remain
the frozen implementation. No path search or hash-matching fallback is used.
"""
from pathlib import Path
import argparse
import json
import sys

import analyze_born_boundary_check as frozen

FROZEN_SHA256 = "d29682d79e29cacf278ca5b7c93ea758f7a768ff3ad86e35bc2089dcda8e9925"
BOUNDARY_AUDIT = "data/born_scaling/boundary_check/boundary_audit.json"
INDEPENDENT_AUDIT = "data/born_scaling/boundary_check/independent_validation.json"


def evidence_base(receipt, field):
    """Only the two specified geometry schemas use receipt-relative file maps."""
    path = Path(receipt).resolve()
    if field == "files" and path == (frozen.ROOT / BOUNDARY_AUDIT).resolve():
        return path.parent
    if field == "inputs" and path == (frozen.ROOT / INDEPENDENT_AUDIT).resolve():
        return path.parent
    return frozen.ROOT


def compatibility_source(dependencies):
    path = Path(__file__).resolve()
    digest = frozen.base.sha256(path)
    snapshot = frozen.ROOT / "analysis/born_scaling/provenance" / f"{path.stem}_{digest}.py"
    snapshot.parent.mkdir(parents=True, exist_ok=True)
    if snapshot.exists():
        if frozen.base.sha256(snapshot) != digest:
            raise ValueError("Changed content-addressed compatibility entrypoint snapshot")
    else:
        with snapshot.open("xb") as stream:
            stream.write(path.read_bytes())
    frozen.remember(snapshot, dependencies)
    return dict(entrypoint=frozen.base.relative(path), executed_sha256=digest,
                source_snapshot=frozen.base.relative(snapshot),
                frozen_analyzer_sha256=FROZEN_SHA256,
                repair="Evidence path bases only: geometry audit files and independent audit inputs are receipt-relative; pilot summary/manifest files and all source maps are born_ai-relative. No statistical change.")


def verify_frozen_evidence(plan, dependencies):
    frozen.verify_mapping(plan.get("evidence_sha256", {}), dependencies)
    for stem in ("allocation", "pilot_summary"):
        path_key, hash_key = f"{stem}_path", f"{stem}_sha256"
        if (path_key in plan) != (hash_key in plan):
            raise ValueError(f"Incomplete {stem} path/hash binding")
        if path_key in plan:
            frozen.verify_mapping({plan[path_key]: plan[hash_key]}, dependencies)
    for name in plan.get("evidence_sha256", {}):
        path = frozen.resolve_path(name)
        evidence = frozen.read_metadata(path)
        if "status" in evidence and evidence["status"].lower() not in ("pass", "complete", "completed"):
            raise ValueError(f"Frozen evidence has not passed: {path}")
        for field in ("source_sha256", "source_hashes", "reference_hashes"):
            frozen.verify_mapping(evidence.get(field, {}), dependencies)
        for field in ("files", "inputs"):
            frozen.verify_mapping(evidence.get(field, {}), dependencies, evidence_base(path, field))
    compatibility_source(dependencies)


def main():
    actual = frozen.base.sha256(Path(frozen.__file__))
    if actual != FROZEN_SHA256:
        raise ValueError("Compatibility entrypoint requires the unchanged frozen analyzer")
    frozen.verify_frozen_evidence = verify_frozen_evidence
    original_write = frozen.base.write_json

    def write_with_provenance(path, payload):
        if Path(path).name in ("summary.json", "pilot_summary.json") and isinstance(payload, dict) and "stage" in payload:
            payload["analysis_path_compatibility"] = compatibility_source(payload["files"])
        return original_write(path, payload)

    frozen.base.write_json = write_with_provenance
    if "--check-evidence" in sys.argv[1:]:
        parser = argparse.ArgumentParser(description=__doc__)
        parser.add_argument("--check-evidence", action="store_true")
        parser.add_argument("--plan", type=Path,
            default=frozen.ROOT / "data/born_scaling/boundary_check/production/plan.toml")
        args = parser.parse_args()
        plan = frozen.read_metadata(args.plan)
        dependencies = {}
        frozen.verify_mapping(plan["source_sha256"], dependencies)
        verify_frozen_evidence(plan, dependencies)
        print(json.dumps(dict(status="PASS", role="Dependencies only; no production observations analyzed",
            verified_files=len(dependencies), compatibility=compatibility_source(dependencies)), indent=2))
    else:
        frozen.main()


if __name__ == "__main__":
    main()
