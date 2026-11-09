"""Resume the frozen boundary production after the original status writer failed.

Never restart a job having any existing output. Completed receipts are fully
verified. Live jobs are identified by executable command, plan, run-id and
creation time, rather than a remembered PID. Numerical sources/plan are unchanged.
"""
import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import tomllib
import uuid

import psutil

ROOT = Path(__file__).resolve().parent
RUN = ROOT / "data/born_scaling/boundary_check/production"
PLAN = RUN / "plan.toml"
RUNNER = ROOT / "run_born_boundary_check_fast.jl"
REASON = "Original launcher exited after Windows PermissionError (WinError 5) replacing launch_status.json; recorded status stopped at 2026-10-28T06:35:51.660332+00:00. Existing Julia jobs and their data were not restarted."


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def rel(path):
    return Path(path).resolve().relative_to(ROOT).as_posix()


def norm(path):
    return os.path.normcase(os.path.abspath(path))


def atomic_json(path, value, replace=os.replace, attempts=30, initial_delay=.1):
    """Unique temporary files and bounded retry of Windows sharing failures."""
    path = Path(path)
    temporary = path.with_name(path.name + f".{os.getpid()}.{uuid.uuid4().hex}.partial")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    for attempt in range(attempts):
        try:
            replace(temporary, path)
            return
        except OSError as error:
            if not isinstance(error, PermissionError) and getattr(error, "winerror", None) not in (5, 32, 33):
                raise
            if attempt + 1 == attempts:
                # Keep the unique temporary file as recovery evidence.
                raise
            time.sleep(min(2., initial_delay * 1.5 ** attempt))


def read_receipt(job):
    return tomllib.loads((ROOT / job["receipt_path"]).read_text(encoding="utf-8"))


def validate_complete(job, plan_hash):
    receipt = read_receipt(job)
    if receipt.get("status") != "complete" or receipt.get("plan_sha256") != plan_hash or receipt.get("job") != job:
        raise ValueError(f"Incomplete/wrong receipt: {job['run_id']}")
    count = job["sweeps"] // job["measure_every"]
    if receipt["records"] != count or receipt["rows_per_measurement"] != 13:
        raise ValueError("Wrong sample count")
    for field in ("maximum_inverse_drift", "maximum_logweight_drift"):
        if not 0 <= receipt[field] < 1e-5:
            raise ValueError("Numerical drift failure")
    if receipt["legacy_entropy_max_difference"] >= 1e-8:
        raise ValueError("Entropy equivalence failure")
    for name, expected in receipt["files"].items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Changed completed output: {name}")
    if (ROOT / job["snapshot_path"]).stat().st_size != 4 * job["L"]**2 * count:
        raise ValueError("Snapshot byte count is wrong")
    with (ROOT / job["sample_path"]).open(encoding="utf-8") as stream:
        if sum(1 for _ in stream) != 1 + count * 13:
            raise ValueError("CSV row count is wrong")
    if sha(ROOT / job["model_manifest"]) != job["model_manifest_sha256"]:
        raise ValueError("Ground-state model manifest changed")
    return receipt


def matched_processes(jobs, recorded):
    by_id = {job["run_id"]: job for job in jobs}
    matches = []
    for process in psutil.process_iter(["pid", "name"]):
        name = (process.info["name"] or "").lower()
        if "julia" not in name:
            continue
        try:
            command = process.cmdline()
            ids = [arg.split("=", 1)[1] for arg in command if arg.startswith("--run-id=")]
            plans = [arg.split("=", 1)[1] for arg in command if arg.startswith("--plan=")]
            if len(ids) != 1 or ids[0] not in by_id:
                continue
            if len(plans) != 1 or norm(plans[0]) != norm(PLAN) or not any(norm(arg) == norm(RUNNER) for arg in command):
                raise ValueError("A matching run-id has a different runner or plan")
            run_id = ids[0]
            if run_id not in recorded:
                raise ValueError("Live job has no recorded original launch")
            created = process.create_time()
            started = dt.datetime.fromisoformat(recorded[run_id]["started_at"]).timestamp()
            if not started - 5 <= created <= started + 120:
                raise ValueError("Live process creation time differs from original launch")
            matches.append(dict(run_id=run_id, pid=process.pid, ppid=process.ppid(),
                create_time=created, executable=process.exe(), command=command))
        except psutil.NoSuchProcess:
            continue
    # A Juliaup alias may remain as the parent of the actual Julia worker.
    parents = {item["ppid"] for item in matches}
    leaves = [item for item in matches if item["pid"] not in parents]
    if len({item["run_id"] for item in leaves}) != len(leaves):
        raise ValueError("Multiple unrelated live workers share one run-id")
    return {item["run_id"]: item for item in leaves}


def same_process_alive(info):
    try:
        process = psutil.Process(info["pid"])
        return (abs(process.create_time() - info["create_time"]) < .001 and
                process.is_running() and process.status() != psutil.STATUS_ZOMBIE)
    except psutil.NoSuchProcess:
        return False


def job_paths(job):
    return [ROOT / job[key] for key in ("sample_path", "receipt_path", "snapshot_path")] + [RUN / (job["run_id"] + ".log")]


def preflight():
    plan = tomllib.loads(PLAN.read_text(encoding="utf-8"))
    plan_hash = sha(PLAN)
    if (RUN / "manifest.json").exists():
        raise ValueError("Production already has a manifest; refusing duplicate recovery")
    for name, expected in (plan["source_sha256"] | plan.get("evidence_sha256", {})).items():
        if sha(ROOT / name) != expected:
            raise ValueError(f"Frozen dependency changed: {name}")
    status = json.loads((RUN / "launch_status.json").read_text(encoding="utf-8"))
    launch = json.loads((RUN / "launch_receipt.json").read_text(encoding="utf-8"))
    if launch["plan_sha256"] != plan_hash:
        raise ValueError("Original launch used another plan")
    recorded = {item["run_id"]: item for item in status["processes"]}
    live = matched_processes(plan["jobs"], recorded)
    done, pending, adopted = [], [], []
    for job in plan["jobs"]:
        receipt_path = ROOT / job["receipt_path"]
        receipt = read_receipt(job) if receipt_path.exists() else None
        if receipt and receipt.get("status") == "complete":
            validate_complete(job, plan_hash)
            if job["run_id"] in live:
                adopted.append((job, live[job["run_id"]]))
            else:
                done.append(job)
        elif job["run_id"] in live:
            if receipt and (receipt.get("plan_sha256") != plan_hash or receipt.get("job") != job):
                raise ValueError("Adopted job receipt does not match frozen plan")
            adopted.append((job, live[job["run_id"]]))
        elif any(path.exists() for path in job_paths(job)):
            raise ValueError(f"Orphan partial output, refusing restart: {job['run_id']}")
        else:
            pending.append(job)
    if len(adopted) > plan["execution_workers"]:
        raise ValueError("Too many live workers")
    return plan, plan_hash, status, done, pending, adopted


def self_test():
    with tempfile.TemporaryDirectory(prefix="boundary_recovery_atomic_") as directory:
        path = Path(directory) / "state.json"
        calls = []
        def retry_replace(source, destination):
            calls.append(str(source))
            if len(calls) <= 3:
                raise PermissionError("simulated Windows sharing violation")
            os.replace(source, destination)
        atomic_json(path, {"pass": True}, replace=retry_replace, initial_delay=.001)
        if json.loads(path.read_text()) != {"pass": True} or len(calls) != 4:
            raise AssertionError("Atomic permission retry failed")
        atomic_json(path, {"pass": "second"})
        if list(Path(directory).glob("*.partial")):
            raise AssertionError("Successful writes left partial files")
    print("SELF TEST PASS: unique temporary names and PermissionError retry", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preflight", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    plan, plan_hash, original_status, done, pending, adopted = preflight()
    print(json.dumps(dict(preflight="PASS", complete=[j["run_id"] for j in done],
        pending=len(pending), adopted=[info for _, info in adopted]), indent=2), flush=True)
    if args.preflight:
        return
    lock = RUN / "recovery_launcher.lock"
    with lock.open("x", encoding="utf-8") as stream:
        json.dump(dict(pid=os.getpid(), create_time=psutil.Process().create_time(),
                       started_at=utc(), source_sha256=sha(__file__)), stream)
    recovery = RUN / ("recovery_" + dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8])
    recovery.mkdir()
    preserved = {}
    for name in ("launch_status.json", "launch_receipt.json"):
        destination = recovery / ("original_" + name)
        with destination.open("xb") as stream:
            stream.write((RUN / name).read_bytes())
        preserved[rel(destination)] = sha(destination)
    source = recovery / Path(__file__).name
    with source.open("xb") as stream:
        stream.write(Path(__file__).read_bytes())
    preserved[rel(source)] = sha(source)
    receipt_path = recovery / "recovery_receipt.json"
    receipt = dict(status="running", started_at=utc(), reason=REASON, plan_path=rel(PLAN), plan_sha256=plan_hash,
        recovery_source=rel(source), recovery_source_sha256=sha(source), files=preserved,
        initial_complete_jobs=[j["run_id"] for j in done], initial_adopted_processes=[info for _, info in adopted],
        initial_pending_jobs=[j["run_id"] for j in pending], numerical_parameters_unchanged=True,
        initial_complete_receipts={j["receipt_path"]:sha(ROOT/j["receipt_path"]) for j in done})
    atomic_json(receipt_path, receipt)
    environment = dict(os.environ, JULIA_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    hidden = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    processes = [dict(item) for item in original_status["processes"]]
    running = [(job, None, dict(info, adopted=True)) for job, info in adopted]
    pending.sort(key=lambda job:(-job["L"], job["chain"]))
    failed = []
    while pending or running:
        while pending and len(running) < plan["execution_workers"] and not failed:
            job = pending.pop(0)
            if any(path.exists() for path in job_paths(job)):
                raise ValueError("A pending job acquired output before launch; refusing overwrite")
            command = [shutil.which("julia"), "--project=.", str(RUNNER), "--plan="+str(PLAN), "--run-id="+job["run_id"]]
            with (RUN / (job["run_id"] + ".log")).open("xb") as log:
                process = subprocess.Popen(command, cwd=ROOT.parents[2], env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=hidden)
            info = dict(run_id=job["run_id"], pid=process.pid, started_at=utc(), command=command,
                        recovery_launch=True, create_time=psutil.Process(process.pid).create_time())
            processes.append(info)
            running.append((job, process, info))
            print("LAUNCHED " + job["run_id"] + " pid=" + str(process.pid), flush=True)
        for job, process, info in running[:]:
            if process is None:
                if same_process_alive(info):
                    continue
                code = None
            else:
                code = process.poll()
                if code is None:
                    continue
            running.remove((job, process, info))
            info.update(exit_code=code, finished_at=utc())
            try:
                if code not in (None, 0):
                    raise RuntimeError(f"Julia exited with code {code}")
                validate_complete(job, plan_hash)
                done.append(job)
                info["completion_validation"] = "receipt, plan, row counts, binary size, numerical diagnostics and all hashes verified"
            except Exception as error:
                info["error"] = str(error)
                failed.append(dict(info))
            print("FINISHED " + job["run_id"] + " receipt_verified=" + str(job in done), flush=True)
        state = dict(status="running" if not failed else "failed", started_at=original_status["started_at"],
            updated_at=utc(), completed=len(done), total=len(plan["jobs"]), running=[j["run_id"] for j,_,_ in running],
            pending=len(pending), failed=failed, processes=processes, recovery_receipt_path=rel(receipt_path),
            recovery_source_sha256=sha(source))
        try:
            atomic_json(RUN / "launch_status.json", state)
        except PermissionError as error:
            # Telemetry must not kill already-running scientific calculations.
            print("WARNING: status sharing violation persists; unique partial retained: " + str(error), flush=True)
        if failed and not running:
            break
        if pending or running:
            time.sleep(5)
    receipt.update(status="failed" if failed else "complete", completed_at=utc(), completed_jobs=[j["run_id"] for j in done],
                   failed=failed, processes=processes)
    atomic_json(receipt_path, receipt)
    if failed:
        raise RuntimeError("Recovery stopped after a failed job; all outputs retained")
    if len(done) != len(plan["jobs"]):
        raise RuntimeError("Incomplete production; no manifest emitted")
    files = {rel(PLAN):plan_hash, rel(receipt_path):sha(receipt_path), **preserved}
    for job in done:
        output_receipt = validate_complete(job, plan_hash)
        files[job["receipt_path"]] = sha(ROOT / job["receipt_path"])
        files.update(output_receipt["files"])
        files[job["model_manifest"]] = sha(ROOT / job["model_manifest"])
    manifest = dict(status="complete", stage="production", completed_at=utc(), plan_path=rel(PLAN),
        plan_sha256=plan_hash, jobs=plan["jobs"], files=files, source_sha256=plan["source_sha256"],
        total_records=sum(j["sweeps"]//j["measure_every"] for j in done),
        retained_records=sum(j["retained_records"] for j in done), recovery_receipt_path=rel(receipt_path),
        recovery_receipt_sha256=sha(receipt_path), recovery_source_sha256=sha(source))
    atomic_json(RUN / "manifest.json", manifest)
    state.update(status="complete", completed_at=utc(), manifest_sha256=sha(RUN / "manifest.json"))
    atomic_json(RUN / "launch_status.json", state)
    print("COMPLETE production: original four completed jobs preserved, remaining jobs recovered", flush=True)


if __name__ == "__main__":
    main()
