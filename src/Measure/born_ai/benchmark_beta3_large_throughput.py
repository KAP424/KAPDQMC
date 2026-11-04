"""Timing-only 1/2/4-worker comparison after the separate pilot completes.

Six fixed-order rounds, startup/warmup excluded by a common timing barrier.
Runs no production and never changes any preexisting source or data file.
"""
from pathlib import Path
import hashlib
import json
import os
import shutil
import subprocess
import time
import tomllib
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "data/born_scaling/beta3_large_throughput"
SCRIPT = ROOT / "benchmark_beta3_large_throughput.jl"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if OUT.exists():
        raise RuntimeError("Timing experiment directory already exists; no overwrite or duplicate launch")
    pilot_manifest = ROOT / "data/born_scaling/pilot_beta3_large/pilot_beta3_large_manifest.toml"
    pilot = tomllib.loads(pilot_manifest.read_text(encoding="utf-8"))
    if pilot["status"] != "complete":
        raise RuntimeError("Wait for the complete pilot before starting any benchmark worker")
    OUT.mkdir()
    env = dict(os.environ, OPENBLAS_NUM_THREADS="1", JULIA_NUM_THREADS="1", MKL_NUM_THREADS="1")
    julia = shutil.which("julia")
    if not julia:
        raise RuntimeError("Julia unavailable")
    hidden = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
    files = {str(p.relative_to(ROOT)).replace("\\", "/"):digest(p) for p in
             (Path(__file__), SCRIPT, pilot_manifest)}
    rows = []
    equivalence = {}
    started = datetime.now(timezone.utc).isoformat()
    for round_id, nworkers in enumerate((1, 2, 4, 4, 2, 1), 1):
        directory = OUT / f"round_{round_id}_workers_{nworkers}"
        directory.mkdir()
        processes = []
        commands = []
        for worker in range(1, nworkers+1):
            log = (directory / f"worker_{worker}.log").open("xb")
            command = [julia, "--project=.", str(SCRIPT), f"--worker={worker}",
                       f"--output={directory}", "--timed-sweeps=20"]
            commands.append(command)
            process = subprocess.Popen(command, cwd=ROOT.parents[2], env=env, stdin=subprocess.DEVNULL,
                                       stdout=log, stderr=subprocess.STDOUT, creationflags=hidden)
            log.close()
            processes.append(process)
        (directory / "launch.json").write_text(json.dumps(dict(nworkers=nworkers, pids=[p.pid for p in processes],
            started_utc=datetime.now(timezone.utc).isoformat(), commands=commands), indent=2), encoding="utf-8")
        print(f"WARMUP round={round_id} workers={nworkers}", flush=True)
        start = time.monotonic()
        while not all((directory / f"worker_{worker}_ready").exists() for worker in range(1, nworkers+1)):
            if any(p.poll() is not None for p in processes):
                raise RuntimeError(f"Worker exited during warmup in {directory}; inspect logs")
            if time.monotonic()-start > 300:
                raise RuntimeError("Warmup exceeded fixed 300-second bound; inspect only launched workers")
            time.sleep(.2)
        (directory / "go").write_text("start\n", encoding="utf-8")
        for process in processes:
            if process.wait() != 0:
                raise RuntimeError(f"Benchmark worker failed in {directory}")
        receipts = []
        for worker in range(1, nworkers+1):
            path = directory / f"worker_{worker}.toml"
            receipt = tomllib.loads(path.read_text(encoding="utf-8"))
            receipts.append(receipt)
            files[path.relative_to(ROOT).as_posix()] = digest(path)
            signature = (receipt["accepted"], receipt["final_E_sha256"], receipt["final_log_weight"])
            if worker in equivalence and equivalence[worker] != signature:
                raise RuntimeError("Worker concurrency changed identical-seed benchmark trajectory")
            equivalence[worker] = signature
            if max(receipt["maximum_inverse_drift"], receipt["maximum_logweight_drift"]) > 1e-5:
                raise RuntimeError("Benchmark drift threshold exceeded")
        elapsed = max(r["elapsed_seconds"] for r in receipts)
        throughput = sum(r["timed_sweeps"] for r in receipts) / elapsed
        rows.append(dict(round=round_id, workers=nworkers, slowest_worker_seconds=elapsed,
                         aggregate_sweeps_per_second=throughput,
                         update_seconds_sum=sum(r["update_seconds"] for r in receipts),
                         entropy_seconds_sum=sum(r["entropy_seconds"] for r in receipts)))
        print(f"TIMED round={round_id} workers={nworkers} sweeps_per_second={throughput:.6f}", flush=True)
    average = {w:sum(r["aggregate_sweeps_per_second"] for r in rows if r["workers"] == w)/2 for w in (1,2,4)}
    fastest = max(average.values())
    recommended = min(w for w in average if average[w] >= .95*fastest)
    result = dict(status="PASS", purpose="Timing-only; excluded from all physical inference", started_utc=started,
        finished_utc=datetime.now(timezone.utc).isoformat(), rounds=rows,
        mean_aggregate_sweeps_per_second=average, recommended_workers=recommended,
        selection_rule="Fewest workers within five percent of highest two-round mean throughput",
        same_seed_trajectory_invariant=True, files=files)
    (OUT / "summary.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)


if __name__ == "__main__":
    main()
