"""Freeze the authorized pilot-variance budget before any new production sample."""
from pathlib import Path
import csv
import hashlib
import json
import math
from datetime import datetime, timezone

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"analysis/born_scaling/pilot_beta3_large"


def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    output=OUT/"production_allocation.csv"
    receipt=OUT/"production_allocation_design.json"
    if output.exists() or receipt.exists():raise RuntimeError("Allocation already frozen")
    summary=OUT/"summary.json"
    source=json.loads(summary.read_text())
    assert source["status"]=="PASS" and source["stage"]=="pilot_beta3_large"
    assert all(r["burnin256_screen_pass"] for r in source["burnin_screens"])
    for file,digest in source["files"].items():assert sha(ROOT/file)==digest,file
    candidate=OUT/"allocation_candidates.csv"
    with candidate.open(newline="") as stream:rows=list(csv.DictReader(stream))
    designs=[]; allocations=[]
    for L,expected_keep in ((27,2048),(30,2560),(36,2816)):
        rr=[r for r in rows if int(r["L"])==L and int(r["pilot_burnin"]) in (128,256)]
        assert len(rr)==2
        need_D=max(float(r["estimated_needed_D"]) for r in rr)
        need_SA=max(float(r["estimated_needed_SA"]) for r in rr)
        raw_need=max(need_D,need_SA)
        keep=max(2048,256*math.ceil(1.20*raw_need/256))
        assert keep==expected_keep
        allocations.append(dict(L=L,beta=3.0,retained_sweeps=keep,burnin=256,grid="main"))
        designs.append(dict(L=L,beta=3.0,pilot_burnin_cuts="128;256",maximum_needed_D=need_D,
                            maximum_needed_SA=need_SA,maximum_needed=raw_need,variance_budget_factor=1.20,
                            budget_before_rounding=1.20*raw_need,rounding_multiple=256,minimum_keep=2048,
                            retained_sweeps=keep,burnin=256,n_chains=4,
                            predicted_D_MCSE=.003*math.sqrt(need_D/keep),
                            predicted_SA_MCSE=.01*math.sqrt(need_SA/keep)))
    details=OUT/"production_allocation_design.csv"
    for path,items in ((output,allocations),(details,designs)):
        with path.open("x",newline="",encoding="utf-8") as stream:
            writer=csv.DictWriter(stream,fieldnames=list(items[0]));writer.writeheader();writer.writerows(items)
    bound=[summary,candidate,output,details,Path(__file__).resolve()]
    result=dict(status="FROZEN_BEFORE_PRODUCTION",created_utc=datetime.now(timezone.utc).isoformat(),
                rule="max(2048, ceil_to_256(1.20*max needed count over pilot burnin128/256 and D/SA)))",
                purpose="Independent variance-based fixed allocation, not selection by agreement with a scaling law",
                burnin=256,chains_per_size=4,allocation=designs,retained_total=sum(4*r["retained_sweeps"] for r in allocations),
                warmup_total=3*4*256,all_rows_total=sum(4*(r["retained_sweeps"]+256) for r in allocations),
                files={p.relative_to(ROOT).as_posix():sha(p) for p in bound},
                note="Targets are absolute estimated Monte Carlo precision, not guaranteed accuracy. Pilot samples never enter production estimates. Twenty percent is variance budget, not a post-run error-bar adjustment.")
    receipt.write_text(json.dumps(result,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({k:v for k,v in result.items() if k!="files"},indent=2))


if __name__=="__main__":main()
