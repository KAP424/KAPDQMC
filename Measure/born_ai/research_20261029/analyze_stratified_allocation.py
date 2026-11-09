"""Separate equal-allocation waste from the explanatory power of latent j."""
from pathlib import Path
import csv
import hashlib
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
OUT = HERE / "pilot_analysis"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    source = OUT / "strata.csv"
    with source.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    summary = json.loads((OUT / "summary.json").read_text())
    assert sha(source) == summary["outputs_sha256"]["strata.csv"]
    result = []
    for L, case, observable in sorted({(int(r["L"]), r["case"], r["observable"]) for r in rows}):
        selected = [r for r in rows if int(r["L"]) == L and r["case"] == case and r["observable"] == observable]
        strata = [r for r in selected if r["method"] == "stratified"]
        iid = next(r for r in selected if r["method"] == "iid_latent")
        w = np.array([float(r["weight"]) for r in strata])
        mu = np.array([float(r["mean"]) for r in strata])
        variance = np.array([float(r["variance"]) for r in strata])
        cost = np.array([float(r["seconds_per_sample_median"]) for r in strata])
        iid_cost = float(iid["seconds_per_sample_median"])
        within = float(w @ variance)
        between = float(w @ ((mu - w @ mu)**2))
        iid_variance = within + between
        oracle_samples = float(np.sum(w*np.sqrt(variance))**2)
        oracle_time = float(np.sum(w*np.sqrt(variance*cost))**2)
        equal_time = float(len(w)*np.sum(w*w*variance)*np.mean(cost))
        result.append(dict(L=L, case=case, beta=float(iid["beta"]), observable=observable,
            within_variance=within, between_variance=between, reconstructed_iid_variance=iid_variance,
            explained_between_fraction=between/iid_variance,
            proportional_allocation_gain=iid_variance/within,
            oracle_neyman_equal_draw_gain=iid_variance/oracle_samples,
            oracle_neyman_equal_time_gain=iid_variance*iid_cost/oracle_time,
            equal_allocation_equal_time_gain=iid_variance*iid_cost/equal_time,
            oracle_draw_fractions=";".join(str(v) for v in w*np.sqrt(variance)/np.sum(w*np.sqrt(variance))),
            warning="Pilot plug-in oracle, not achievable performance proof or a calibrated confidence interval"))
    path = OUT / "allocation_comparison.csv"
    if path.exists():
        raise FileExistsError(path)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(result[0]))
        writer.writeheader(); writer.writerows(result)
    receipt = dict(status="PASS", rows=len(result),
        definition="At fixed time T, oracle Var(mean)=(sum_j w_j sigma_j sqrt(cost_j))^2/T; gain uses reconstructed iid variance times iid median cost.",
        limitations="Conditional moments use only64 draws per stratum. No claim of tail coverage; the entire j>=4 tail is one conditional mixture stratum.",
        inputs_sha256={str(source.relative_to(HERE)):sha(source), str(Path(__file__).relative_to(HERE)):sha(Path(__file__))},
        outputs_sha256={str(path.relative_to(HERE)):sha(path)})
    (OUT / "allocation_comparison.json").write_text(json.dumps(receipt, indent=2)+"\n")
    for row in result:
        if row["observable"] in ("D", "SA", "DV_full"):
            print(row["L"], row["case"], row["observable"], "between", round(row["explained_between_fraction"], 4),
                  "oracle_time_gain", round(row["oracle_neyman_equal_time_gain"], 3), "equal_time_gain", round(row["equal_allocation_equal_time_gain"], 3))


if __name__ == "__main__":
    main()
