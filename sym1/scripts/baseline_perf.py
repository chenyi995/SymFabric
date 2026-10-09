#!/usr/bin/env python3
"""Collect CPU counters and flat cycle samples separately from validated kernel timings."""
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = Path(os.environ["SYMFABRIC_RUN_DIR"])
    rows = []
    # claude: decision start — sample validated binaries without timing verifier or per-iteration printing.
    # Repetitions and sample frequency are diagnostic controls, not hardware constants.
    for name,kernel in (("roadNet-CA","bfs"), ("roadNet-CA","sssp"), ("email-Enron","tc")):
        meta = json.loads((ROOT/f"sym1/data/{name}.json").read_text())
        for threads in (1,32):
            env = dict(os.environ, OMP_NUM_THREADS=str(threads))
            cmd = [str(ROOT/"softwareworkload/upstream/gapbs"/kernel), "-f", str(ROOT/f"sym1/data/{name}.el"), "-s", "-n", "20"]
            if kernel != "tc":
                cmd += ["-r", str(meta["source"])]
            if kernel == "sssp":
                cmd += ["-d", "2"]
            label = f"{name}-{kernel}-t{threads}"
            counter = out/f"{label}-counters.csv"
            statcmd = ["perf", "stat", "-x,", "-e", "task-clock,cycles,instructions,branches,branch-misses,cache-references,cache-misses", "-o", str(counter), "--"]+cmd
            result = subprocess.run(statcmd, env=env, capture_output=True, text=True, timeout=180)
            (out/f"{label}-stat.log").write_text(result.stdout+result.stderr)
            assert result.returncode == 0, result.stderr
            record = out/f"{label}.data"
            recmd = ["perf", "record", "-F", "499", "-e", "cycles:u", "-o", str(record), "--"]+cmd
            result = subprocess.run(recmd, env=env, capture_output=True, text=True, timeout=180)
            (out/f"{label}-record.log").write_text(result.stdout+result.stderr)
            assert result.returncode == 0, result.stderr
            report = subprocess.run(["perf","report","--stdio","--no-children","--percent-limit","0.5","-i",str(record)],
                                    capture_output=True,text=True,timeout=60)
            assert report.returncode == 0, report.stderr
            (out/f"{label}-report.txt").write_text(report.stdout+report.stderr)
            rows.append(dict(dataset=name,kernel=kernel,threads=threads,command=cmd,
                             counters=str(counter), report=str(out/f"{label}-report.txt"),
                             provenance="measured whole-process counters and cycle samples; graph loading included"))
            (out/"perf.json").write_text(json.dumps(rows,indent=2)+"\n")
            print(label,flush=True)
    # claude: decision end


if __name__ == "__main__":
    main()
