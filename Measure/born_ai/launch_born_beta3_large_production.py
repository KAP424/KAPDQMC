"""Launch frozen beta=3 production with a benchmark-selected worker count.

No automatic retry, extra samples, count changes, or unrelated process actions.
The pilot plan and each later production plan must be prepared independently.
"""
import argparse
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
import tomllib

ROOT=Path(__file__).resolve().parent
WORKSPACE=ROOT.parents[2]
WRAPPER=ROOT/"run_born_beta3_fresh_chain.jl"
FINALIZER=ROOT/"run_born_scaling_fused.jl"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save(path,value):
    temp=path.with_suffix(path.suffix+".partial")
    temp.write_text(json.dumps(value,indent=2),encoding="utf-8")
    temp.replace(path)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--stage",default="production_beta3_large")
    args=parser.parse_args()
    if not args.stage.startswith("production_beta3_large"):
        raise ValueError("Only independently frozen beta3 extension stages are supported")
    directory=ROOT/"data/born_scaling"/args.stage
    path=directory/f"{args.stage}_plan.toml"
    receipt=directory/"launch_receipt.toml"
    status_path=directory/"launch_status.json"
    if receipt.exists():
        raise RuntimeError("Launch receipt exists; duplicate worker launch refused")
    plan=tomllib.loads(path.read_text(encoding="utf-8"))
    assert plan["stage"]==args.stage and plan["blas_threads"]==1
    nworkers=plan["execution_workers"]
    assert nworkers in (1,2,4)
    assert plan["execution_mode"]=="fresh_process_per_chain"
    assert plan["source_sha256"][WRAPPER.name]==sha(WRAPPER)
    benchmark_path=ROOT/plan["throughput_benchmark_path"]
    assert sha(benchmark_path)==plan["throughput_benchmark_sha256"]
    benchmark=json.loads(benchmark_path.read_text(encoding="utf-8"))
    assert benchmark["status"]=="PASS" and benchmark["recommended_workers"]==nworkers
    for file,digest in benchmark["files"].items():
        assert sha(ROOT/file)==digest,file
    assert sha(ROOT/plan["analysis_protocol_path"])==plan["analysis_protocol_sha256"]
    assert plan["update_kernel"]=="fused_rank_updates"
    assert all(j["beta"]==3. and j["mu"]==1e-8 and j["measure_every"]==1 and j["refresh_every"]==5 for j in plan["jobs"])
    assert len(plan["jobs"])%4==0 and len(set(j["seed"] for j in plan["jobs"]))==len(plan["jobs"])
    assert len(plan["jobs"])==12 and sorted({j["L"] for j in plan["jobs"]})==[27,30,36]
    assert all(j["retained_sweeps"]>=2048 for j in plan["jobs"])
    for file,digest in plan["source_sha256"].items():
        assert sha(ROOT/file)==digest,file
    assert plan["source_sha256"][Path(__file__).name]==sha(__file__)
    for key,value in plan.get("pilot_evidence",{}).items():
        assert sha(ROOT/key)==value,key
    if "allocation_path" in plan:
        assert sha(ROOT/plan["allocation_path"])==plan["allocation_sha256"]
    if "kernel_validation_path" in plan:
        assert sha(ROOT/plan["kernel_validation_path"])==plan["kernel_validation_sha256"]
    for model in plan["models"]:
        for file,digest in model["files"].items():
            assert sha(ROOT/file)==digest,file
    julia=shutil.which("julia")
    if julia is None:
        raise RuntimeError("Julia unavailable")
    env=dict(os.environ,OPENBLAS_NUM_THREADS="1",JULIA_NUM_THREADS="1",MKL_NUM_THREADS="1")
    hidden=subprocess.CREATE_NO_WINDOW if os.name=="nt" else 0
    started=utc();workers=[]
    with receipt.open("x",encoding="utf-8") as stream:
        for key,value in dict(status="launched",started_at=started,plan_path=str(path),plan_sha256=sha(path),
                              launcher_sha256=sha(__file__),working_directory=str(WORKSPACE),
                              window_mode="Windows CREATE_NO_WINDOW",update_kernel="fused_rank_updates",
                              execution_mode="fresh_process_per_chain").items():
            stream.write(f"{key} = {json.dumps(value)}\n")
        stream.write(f"workers = {nworkers}\nblas_threads_per_worker = 1\n");stream.flush()
    pending=list(plan["jobs"])

    def launch_job(job,slot):
        command=[julia,"--project=.",str(WRAPPER),"--plan="+str(path),"--run-id="+job["run_id"]]
        logfile=directory/(job["run_id"]+".log")
        with logfile.open("xb") as log:
            process=subprocess.Popen(command,cwd=WORKSPACE,env=env,stdin=subprocess.DEVNULL,
                                     stdout=log,stderr=subprocess.STDOUT,creationflags=hidden)
        info=dict(slot=slot,run_id=job["run_id"],pid=process.pid,command=command,log=str(logfile),started_at=utc())
        workers.append((process,info))
        with receipt.open("a",encoding="utf-8") as stream:
            stream.write("\n[[worker_processes]]\n")
            for key,value in info.items():stream.write(f"{key} = {json.dumps(value)}\n")
        print(f"LAUNCHED slot={slot} run_id={job['run_id']} pid={process.pid}",flush=True)
    announced=set();cache={};expected=collections.Counter(j["L"] for j in plan["jobs"])
    while True:
        active_slots={info["slot"] for process,info in workers if process.poll() is None}
        launch_failure=any(process.poll() not in (None,0) for process,_ in workers)
        while pending and len(active_slots)<nworkers and not launch_failure:
            slot=next(i for i in range(1,nworkers+1) if i not in active_slots)
            launch_job(pending.pop(0),slot)
            active_slots.add(slot)
        complete=[];failed=[];running=[]
        for job in plan["jobs"]:
            jobpath=ROOT/job["output_dir"]/(job["run_id"]+"_receipt.toml")
            if not jobpath.exists():continue
            try:
                stamp=jobpath.stat().st_mtime_ns
                if jobpath not in cache or cache[jobpath][0]!=stamp:
                    cache[jobpath]=(stamp,tomllib.loads(jobpath.read_text(encoding="utf-8")))
                value=cache[jobpath][1]
            except (OSError,tomllib.TOMLDecodeError):continue
            if value["status"]=="complete":complete.append((job,value))
            elif value["status"]=="failed":failed.append(dict(run_id=job["run_id"],error=value.get("error","unknown")))
            else:running.append(job["run_id"])
        counts=collections.Counter(j["L"] for j,_ in complete)
        exits={info["run_id"]:process.poll() for process,info in workers}
        state=dict(status="running",stage=args.stage,started_at=started,checked_at=utc(),
                   completed_chains=len(complete),total_chains=len(plan["jobs"]),completed_by_L=dict(counts),
                   expected_by_L=dict(expected),running_chains=running,failed_chains=failed,
                   worker_exit_codes=exits,worker_processes=[info for _,info in workers],
                   unlaunched_chains=len(pending),execution_workers=nworkers,execution_mode="fresh_process_per_chain",
                   plan_sha256=sha(path),launch_receipt_sha256=sha(receipt),
                   maximum_inverse_drift=max((r["maximum_inverse_drift"] for _,r in complete),default=0.),
                   maximum_logweight_drift=max((r["maximum_logweight_drift"] for _,r in complete),default=0.))
        save(status_path,state)
        for L,target in expected.items():
            if counts[L]==target and L not in announced:
                announced.add(L);print(f"SIZE COMPLETE L={L}, chains={target}",flush=True)
        if all(code is not None for code in exits.values()) and (not pending or launch_failure):break
        time.sleep(15)
    if failed or any(code!=0 for code in exits.values()) or len(complete)!=len(plan["jobs"]):
        state["status"]="failed_or_incomplete";save(status_path,state)
        raise RuntimeError("Frozen stage failed/incomplete; no retries or count changes made")
    command=[julia,"--project=.",str(FINALIZER),"--plan="+str(path),"--finalize"]
    with (directory/"finalize.log").open("xb") as log:
        result=subprocess.run(command,cwd=WORKSPACE,env=env,stdin=subprocess.DEVNULL,
                              stdout=log,stderr=subprocess.STDOUT,creationflags=hidden)
    state.update(finalize_command=command,finalize_exit_code=result.returncode,completed_at=utc(),
                 status="complete" if result.returncode==0 else "finalization_failed")
    if result.returncode==0:
        state["manifest_sha256"]=sha(directory/f"{args.stage}_manifest.toml")
    save(status_path,state);print("FINAL "+state["status"],flush=True)
    if result.returncode:raise RuntimeError("Finalization failed; inspect finalize.log")


if __name__=="__main__":main()
