#!/usr/bin/env python3
"""Run an explicitly requested experiment with bounded resources and append-only logs."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import resource
import subprocess
import time

ROOT = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("label")
    p.add_argument("--cwd", default=str(ROOT))
    p.add_argument("--threads", type=int, default=32)
    p.add_argument("--timeout", type=int, default=1800)
    p.add_argument("command", nargs=argparse.REMAINDER)
    args = p.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not command:
        p.error("a command is required")
    stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    out = ROOT / "sym1/results" / f"{stamp}-{args.label}"
    out.mkdir(parents=True, exist_ok=False)
    # chenyi9: decision start — the session authorizes at most 32 CPU cores and 300 GB RAM.
    cpus = sorted(os.sched_getaffinity(0))[:min(args.threads, 32)]
    os.sched_setaffinity(0, cpus)
    resource.setrlimit(resource.RLIMIT_AS, (300_000_000_000, 300_000_000_000))
    # chenyi9: decision end
    env = os.environ.copy()
    for key in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        env[key] = str(len(cpus))
    env["OMP_PROC_BIND"] = "close"
    env["OMP_PLACES"] = "cores"
    env["PYTHONUNBUFFERED"] = "1"
    env["CCACHE_DISABLE"] = "1"
    env["SYMFABRIC_RUN_DIR"] = str(out)
    metadata = {"command": command, "cwd": str(Path(args.cwd).resolve()), "utc": stamp,
                "cpu_affinity": cpus, "address_space_limit_bytes": 300_000_000_000,
                "thread_environment": {k: env[k] for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
                "provenance": "measured local process execution; not accelerator hardware timing"}
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(out, flush=True)
    start = time.monotonic()
    with (out / "stdout.log").open("w") as log:
        proc = subprocess.Popen(command, cwd=args.cwd, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            code = proc.wait(timeout=args.timeout)
        except subprocess.TimeoutExpired:
            import signal
            os.killpg(proc.pid, signal.SIGTERM)
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            code = 124
    metadata.update(exit_code=code, wall_s=time.monotonic() - start,
                    child_max_rss_kib=resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss)
    (out / "completion.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print((out / "stdout.log").read_text(errors="replace")[-5000:])
    print(json.dumps({"result_dir": str(out), "exit_code": code, "wall_s": metadata["wall_s"]}))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
