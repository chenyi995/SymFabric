#!/usr/bin/env python3
"""Check bucket-width sensitivity and record iteration/frontier diagnostics."""
import json
import os
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def main():
    out = Path(os.environ["SYMFABRIC_RUN_DIR"])
    meta = json.loads((ROOT/"sym1/data/roadNet-CA.json").read_text())
    rows = []
    # claude: decision start — GAPBS explicitly requires choosing delta for each input.
    # Bucket widths below are diagnostic experiment controls; generated edge weights are unchanged.
    for delta in (2,16,64,256):
        for threads in (1,32):
            for diagnostic in (False,True):
                n = 1 if diagnostic else 5
                cmd = [str(ROOT/"softwareworkload/upstream/gapbs/sssp"), "-f", str(ROOT/"sym1/data/roadNet-CA.el"),
                       "-s","-r",str(meta["source"]),"-d",str(delta),"-n",str(n),"-v"]
                if diagnostic:
                    cmd.append("-l")
                result = subprocess.run(cmd,env=dict(os.environ,OMP_NUM_THREADS=str(threads)),
                                        capture_output=True,text=True,timeout=180)
                label = f"delta{delta}-t{threads}-{'frontiers' if diagnostic else 'timing'}"
                logfile = out/f"{label}.log"
                logfile.write_text(result.stdout+result.stderr)
                assert result.returncode == 0 and result.stdout.count("PASS") == n, label
                trials = [float(x) for x in re.findall(r"Trial Time:\s+([0-9.]+)",result.stdout)]
                assert len(trials) == n
                iterations = [int(x) for x in re.findall(r"took (\d+) iterations",result.stdout)]
                # Source: GAPBS src/util.h PrintStep prints bin, frontier count, milliseconds.
                steps = re.findall(r"^\s*(\d+)\s+(\d+)\s+([0-9.]+)\s*$",result.stdout,re.M)
                if diagnostic:
                    assert iterations and len(steps) == iterations[0], label
                rows.append(dict(delta=delta,threads=threads,diagnostic=diagnostic,trial_s=trials,
                                 iteration_counts=iterations,steps=steps,log=str(logfile),command=cmd,pass_check=True,
                                 provenance="measured CPU; logging runs separate from timing runs"))
                (out/"buckets.json").write_text(json.dumps(rows,indent=2)+"\n")
                print(label,flush=True)
    # claude: decision end


if __name__ == "__main__":
    main()
