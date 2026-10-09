"""Working-set extrapolation (DERIVED, not measured): scale every size in a saved trace by a factor
and re-map it, to find the working set at which a configuration becomes memory-bound and the
memory side pays off. Tuple counts, candidates and semiring operations are all multiplied by the
same factor, i.e. the algorithmic shape (fan-out, rounds, density) is kept and only the instance
grows. Results are labelled "derived (scaled trace)" wherever they are reported.

    python3 eval/scale_traces.py eval/results/<dir-with-traces> --factors 1,4,16,64,256 \
        --configs edge28,dc_hbm3 --pim lpddr5_pim,hbm3_pim
"""
from __future__ import annotations

import argparse
import copy
import csv
import datetime as dt
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from algebra.trace import OpRecord                           # noqa: E402
from systolicarch import CONFIGS, map_trace, summarize       # noqa: E402
from pim import PIM_CONFIGS, select_placement                # noqa: E402


def load(path: Path) -> list[OpRecord]:
    blob = json.loads(path.read_text())
    return [OpRecord(**{k: (tuple(v) if k in ("n_in", "in_uids", "in_bits") else v) for k, v in r.items()})
            for r in blob["records"]]


def scaled(records: list[OpRecord], f: float) -> list[OpRecord]:
    out = []
    for r in records:
        s = copy.deepcopy(r)
        s.n_in = tuple(int(round(n * f)) for n in r.n_in)
        s.n_out = int(round(r.n_out * f))
        s.candidates = int(round(r.candidates * f))
        s.mul_ops = int(round(r.mul_ops * f))
        s.add_ops = int(round(r.add_ops * f))
        h = dict(r.hint)
        for k in ("state", "build", "probe", "groups", "emitted"):
            if k in h and isinstance(h[k], (int, float)):
                h[k] = int(round(h[k] * f))
        for k in ("dom_R", "dom_S", "shared_dom"):
            if k in h and h[k]:
                h[k] = [int(round(x * f ** 0.5)) if x else x for x in h[k]]   # keys widen with sqrt(f)
        if h.get("dense_volume"):
            h["dense_volume"] = int(round(h["dense_volume"] * f ** 1.5))
        s.hint = h
        # key widths grow by log2(sqrt f) bits per attribute; approximate with +log2(f)/2 per record
        import math
        s.key_bits = int(r.key_bits + math.log2(f) / 2) if f > 1 else r.key_bits
        s.in_bits = tuple(int(b + math.log2(f) / 2) if f > 1 else b for b in r.in_bits)
        out.append(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results_dir")
    ap.add_argument("--factors", default="1,4,16,64,256")
    ap.add_argument("--configs", default="edge28,dc_hbm3")
    ap.add_argument("--pim", default="lpddr5_pim,hbm3_pim")
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    rd = Path(args.results_dir).resolve()
    ts = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    out = ROOT / "eval" / "results" / f"{ts}-scaled-{rd.name}"
    out.mkdir(parents=True)
    rows = []
    pim_by_cfg = {"edge28": "lpddr5_pim", "dc_hbm3": "hbm3_pim", "dc_hbm4": "hbm3_pim", "tpu_like": "hbm3_pim"}
    for tf in sorted((rd / "traces").glob("*.json")):
        wl = tf.stem
        if args.only and wl != args.only:
            continue
        base = load(tf)
        for f in [float(x) for x in args.factors.split(",")]:
            recs = scaled(base, f) if f != 1 else base
            for cname in args.configs.split(","):
                cfg = CONFIGS[cname]
                costs = map_trace(recs, cfg)
                s = summarize(costs, cfg)
                pim_name = pim_by_cfg.get(cname, args.pim.split(",")[0])
                _, ps = select_placement(recs, costs, cfg, PIM_CONFIGS[pim_name])
                ws = max((r.n_in[i] * -(-r.in_bits[i] // 8) for r in recs for i in range(len(r.n_in))), default=0)
                rows.append({"workload": wl, "factor": f, "config": cname, "largest_operand_MB": ws / 1e6,
                             "time_ms": s["time_ms"], "frac_memory_bound": s["frac_memory_bound"],
                             "avg_bw_util": s["avg_bw_util"], "pe_util": s["pe_util"],
                             "pim": pim_name, "pim_speedup": ps["speedup_from_pim"],
                             "time_share_on_pim": ps["time_share_on_pim"]})
        print(wl, "done")
    with open(out / "scaled.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print("results:", out)


if __name__ == "__main__":
    main()
