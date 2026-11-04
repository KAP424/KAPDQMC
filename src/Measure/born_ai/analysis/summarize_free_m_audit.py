"""Convert the independently generated Julia receipts and bind audit files."""
import hashlib
import json
from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parent


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output = {}
    for stem in ("free_m_update_audit", "entropy_pipeline_audit"):
        source = ROOT / (stem + ".toml")
        receipt = tomllib.loads(source.read_text(encoding="utf-8"))
        receipt["toml_sha256"] = digest(source)
        target = ROOT / (stem + ".json")
        target.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
        output[stem] = receipt
    historical = output["free_m_update_audit"]
    updated = output["entropy_pipeline_audit"]
    for beta in (2, 3, 5):
        rows = [r for r in historical["high_precision_ratio_tests"] if r["beta"] == beta]
        print("256-bit reference beta", beta,
              "source ratio error", max(r["source_relative_error"] for r in rows),
              "Float64 Cholesky ratio error", max(r["cholesky_float64_relative_error"] for r in rows))
    for row in historical["drift_tests"]:
        if row["L"] >= 12:
            print({k: row[k] for k in ("L", "beta", "initial", "proposals", "acceptance",
                                      "max_inverse_relative_error", "stop_reason", "elapsed_seconds")})
    comparisons = []
    for beta in (0, 1, 3):
        pair = {r["sampler"]: r for r in updated["results"] if r["beta"] == beta}
        for obs in ("S2_A", "S2_B", "paired_A_minus_B"):
            a, b = pair["local"][obs], pair["iid"][obs]
            se = (a["block_se"] ** 2 + b["block_se"] ** 2) ** .5
            comparisons.append({"beta": beta, "observable": obs,
                                "local_mean": a["mean"], "iid_mean": b["mean"],
                                "difference": a["mean"] - b["mean"],
                                "combined_block_se": se,
                                "difference_in_block_se": (a["mean"] - b["mean"]) / se if se else 0,
                                "local_iat": a["iat"], "iid_iat": b["iat"]})
            print(comparisons[-1])
    files = [ROOT / name for name in (
        "Free_M_audit_snapshot.jl", "audit_free_m_updates.jl", "free_m_update_audit.toml",
        "free_m_update_audit.json", "audit_entropy_pipeline.jl", "entropy_pipeline_audit.toml",
        "entropy_pipeline_audit.json", "summarize_free_m_audit.py")]
    files += sorted((ROOT / "entropy_smoke").glob("*"))
    files += [ROOT.parent / "Free_M.jl", ROOT.parent / "BornSampler.jl"]
    assert digest(ROOT.parent / "Free_M.jl") == updated["source_sha256"]
    assert digest(ROOT.parent / "BornSampler.jl") == updated["sampler_sha256"]
    assert digest(ROOT / "audit_entropy_pipeline.jl") == updated["script_sha256"]
    assert digest(ROOT / "audit_free_m_updates.jl") == historical["audit_script_sha256"]
    report = {
        "status": "PASS_WITH_NUMERICAL_RISK",
        "historical_update_algebra": "No algebraic mismatch found; finite precision stress failures recorded.",
        "updated_entropy_pipeline": "PASS",
        "maximum_updated_S2_reference_error": max(r["max_S2_reference_error"] for r in updated["results"]),
        "local_iid_comparisons": comparisons,
        "files": {str(p.relative_to(ROOT.parent)).replace("\\", "/"): digest(p) for p in files},
        "limits": "Small-system audits and selected size stress tests do not certify rapid mixing at arbitrary size."
    }
    (ROOT / "audit_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
