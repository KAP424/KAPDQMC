"""Launch exactly four hidden workers of the frozen plan and monitor only those workers.

This launcher does not choose counts, mutate plans, retry chains, or create extra workers.
"""
from __future__ import annotations
import collections
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import tomllib

ROOT = Path(__file__).resolve().parent
WORKSPACE = ROOT.parents[2]
DIRECTORY = ROOT / "data/born_scaling/production"
PLAN = DIRECTORY / "production_plan.toml"
WRAPPER = ROOT / "run_born_scaling_fused.jl"
LAUNCH_RECEIPT = DIRECTORY / "launch_receipt.toml"
STATUS = DIRECTORY / "launch_status.json"

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def utc():
    return dt.datetime.now(dt.timezone.utc).isoformat()

def save_json(path, data):
    temporary = path.with_suffix(path.suffix + ".partial")
    temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
    temporary.replace(path)

def main():
    if LAUNCH_RECEIPT.exists():
        raise RuntimeError("Launch receipt already exists; refusing to launch duplicate workers")
    plan = tomllib.loads(PLAN.read_text(encoding="utf-8"))
    assert plan["stage"] == "production" and plan["update_kernel"] == "fused_rank_updates"
    assert plan["blas_threads"] == 1 and len(plan["jobs"]) == 240
    assert len({(j["L"], j["beta"], j["mu"]) for j in plan["jobs"]}) == 60
    assert sum(j["retained_sweeps"] for j in plan["jobs"]) == 958464
    assert sum(j["analysis_burnin"] for j in plan["jobs"]) == 61440
    for file, digest in plan["source_sha256"].items():
        assert sha(ROOT / file) == digest, file
    for file, digest in plan["pilot_evidence"].items():
        assert sha(ROOT / file) == digest, file
    assert sha(ROOT / plan["allocation_path"]) == plan["allocation_sha256"]
    for model in plan["models"]:
        for file, digest in model["files"].items():
            assert sha(ROOT / file) == digest, file
    julia = shutil.which("julia")
    if julia is None:
        raise RuntimeError("Julia executable unavailable")
    environment = dict(os.environ)
    environment.update(OPENBLAS_NUM_THREADS="1", JULIA_NUM_THREADS="1", MKL_NUM_THREADS="1")
    hidden = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    commands = [[julia, "--project=.", str(WRAPPER), "--plan=" + str(PLAN),
                 f"--worker={worker}", "--workers=4"] for worker in range(1, 5)]
    started = utc()
    workers = []
    # Open the receipt exclusively before launching any process to prevent duplicate launches.
    with LAUNCH_RECEIPT.open("x", encoding="utf-8") as receipt:
        for key, value in {"status": "launched", "started_at": started,
                           "plan_path": str(PLAN), "plan_sha256": sha(PLAN),
                           "working_directory": str(WORKSPACE), "launcher_sha256": sha(__file__),
                           "window_mode": "Windows CREATE_NO_WINDOW", "update_kernel": "fused_rank_updates"}.items():
            receipt.write(f"{key} = {json.dumps(value)}\n")
        receipt.write("workers = 4\nblas_threads_per_worker = 1\n")
        receipt.flush()
        for worker, command in enumerate(commands, 1):
            log_path = DIRECTORY / f"worker_{worker}.log"
            with log_path.open("xb") as log:
                process = subprocess.Popen(command, cwd=WORKSPACE, env=environment,
                    stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=hidden)
            info = {"worker": worker, "pid": process.pid, "command": command,
                    "log": str(log_path), "started_at": utc()}
            workers.append((process, info))
            receipt.write("\n[[worker_processes]]\n")
            for key, value in info.items():
                receipt.write(f"{key} = {json.dumps(value)}\n")
            receipt.flush()
            print(f"LAUNCHED worker={worker} pid={process.pid} log={log_path.name}", flush=True)
    expected_by_L = collections.Counter(j["L"] for j in plan["jobs"])
    announced = set()
    announced_failures = set()
    cache = {}
    last_progress = None
    last_print = 0.0
    while True:
        complete = []
        failed = []
        running = []
        for job in plan["jobs"]:
            path = ROOT / job["output_dir"] / (job["run_id"] + "_receipt.toml")
            if not path.exists():
                continue
            try:
                stamp = path.stat().st_mtime_ns
                if path not in cache or cache[path][0] != stamp:
                    cache[path] = (stamp, tomllib.loads(path.read_text(encoding="utf-8")))
                data = cache[path][1]
            except (OSError, tomllib.TOMLDecodeError):
                continue  # A running receipt can be observed during its short atomicity gap.
            if data["status"] == "complete":
                complete.append((job, data))
            elif data["status"] == "failed":
                failed.append({"run_id": job["run_id"], "error": data.get("error", "unknown")})
            else:
                running.append(job["run_id"])
        completed_by_L = collections.Counter(j["L"] for j, _ in complete)
        exits = {str(info["worker"]): proc.poll() for proc, info in workers}
        state = {"status": "running", "started_at": started, "checked_at": utc(),
            "plan_sha256": sha(PLAN), "launch_receipt_sha256": sha(LAUNCH_RECEIPT),
            "completed_chains": len(complete), "total_chains": len(plan["jobs"]),
            "completed_by_L": dict(completed_by_L), "expected_by_L": dict(expected_by_L),
            "running_chains": running, "failed_chains": failed,
            "worker_exit_codes": exits, "worker_processes": [i for _, i in workers],
            "maximum_inverse_drift": max((r["maximum_inverse_drift"] for _, r in complete), default=0.),
            "maximum_logweight_drift": max((r["maximum_logweight_drift"] for _, r in complete), default=0.)}
        save_json(STATUS, state)
        for L, target in expected_by_L.items():
            if completed_by_L[L] == target and L not in announced:
                print(f"SIZE COMPLETE L={L} chains={target}", flush=True)
                announced.add(L)
        for failure in failed:
            if failure["run_id"] not in announced_failures:
                print("FAILURE " + json.dumps(failure), flush=True)
                announced_failures.add(failure["run_id"])
        progress = (len(complete), tuple(running), tuple(exits.items()))
        if progress != last_progress and time.monotonic() - last_print >= 60:
            print(f"PROGRESS complete={len(complete)}/240 running={running} exits={exits}", flush=True)
            last_progress = progress
            last_print = time.monotonic()
        if all(code is not None for code in exits.values()):
            break
        time.sleep(15)
    if failed or any(code != 0 for code in exits.values()) or len(complete) != 240:
        state["status"] = "failed_or_incomplete"
        save_json(STATUS, state)
        raise RuntimeError("One or more frozen workers failed/incomplete; no retries or count changes were made")
    command = [julia, "--project=.", str(WRAPPER), "--plan=" + str(PLAN), "--finalize"]
    with (DIRECTORY / "finalize.log").open("xb") as log:
        result = subprocess.run(command, cwd=WORKSPACE, env=environment, stdin=subprocess.DEVNULL,
                                stdout=log, stderr=subprocess.STDOUT, creationflags=hidden)
    state["finalize_command"] = command
    state["finalize_exit_code"] = result.returncode
    state["completed_at"] = utc()
    state["status"] = "complete" if result.returncode == 0 else "finalization_failed"
    if result.returncode == 0:
        state["production_manifest_sha256"] = sha(DIRECTORY / "production_manifest.toml")
    save_json(STATUS, state)
    print("FINAL " + state["status"], flush=True)
    if result.returncode:
        raise RuntimeError("Finalization failed; see finalize.log")

if __name__ == "__main__":
    main()
