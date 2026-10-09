"""Copy seven numbers from the run results into docs/findings.md and docs/arch-pim.md (aimd rules
section 16: every number moved between files goes through a script that records source path,
field names and accounting, keeps full precision and formats only on output).

The seven statements were found to disagree with the raw data in the 2026-09-26 audit. Each entry
below names the source, the fields read and the accounting used; the script asserts that the old
sentence occurs exactly once, replaces it, and prints the full-precision values it used so that the
re-check required by section 16 can be done against the printed table.

    python3 eval/update_findings_numbers.py [--dry-run]

Sources
  S1  eval/results/20260925-203019-final6-large/summary.csv            fields: workload, config, cycles
  S2  eval/results/20260925-203019-final6-large/per_op/hmm_viterbi_{edge28,dc_hbm3}.csv
                                                                      fields: mode, cycles_compute, mul_ops
  S3  eval/results/20260925-203156-scaled-20260925-202252-final-large/scaled.csv
                                                                      fields: workload, factor, config,
                                                                      largest_operand_MB, frac_memory_bound, pim_speedup
  S4  eval/results/20260925-203158-sens-large/sensitivity.csv         fields: workload, config, param, time_ms
  S5  pim/config.py HBM3_PIM (rows_per_s, internal_row_bw_Bps)        derived properties of sourced constants
  S6  `python3 -m pytest --collect-only -q tests/` (static count) and the retained log of the full run
      eval/results/<ts>-pytest-full/pytest.log (last line "N passed in ...")
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from systolicarch.config import CONFIGS          # noqa: E402
from pim.config import HBM3_PIM                  # noqa: E402

FINAL6 = ROOT / "eval/results/20260925-203019-final6-large"
SCALED = ROOT / "eval/results/20260925-203156-scaled-20260925-202252-final-large/scaled.csv"
SENS = ROOT / "eval/results/20260925-203158-sens-large/sensitivity.csv"
FINDINGS = ROOT / "docs/findings.md"
ARCH_PIM = ROOT / "docs/arch-pim.md"
MIB_24 = 24 * 1024 * 1024 / 1e6          # the dc_hbm3 buffer in MB (decimal), 25.165824


def tpu_ratios() -> dict:
    """S1: modelled cycles(tpu_like) / cycles(dc_hbm3), same trace, same clock."""
    s = pd.read_csv(FINAL6 / "summary.csv").set_index(["workload", "config"])["cycles"]
    return {w: s[(w, "tpu_like")] / s[(w, "dc_hbm3")]
            for w in ("graph_bfs_sssp", "triangle_count", "kg_multihop", "astar_grid")}


def hmm_dense_step() -> dict:
    """S2: the dense_gemm record of one HMM step on each array; all 238 steps of a run are
    identical (asserted). Utilisation = mul_ops / (cycles_compute x PEs)."""
    out = {}
    for cfg in ("edge28", "dc_hbm3"):
        df = pd.read_csv(FINAL6 / "per_op" / f"hmm_viterbi_{cfg}.csv")
        d = df[df["mode"] == "dense_gemm"]
        assert d["cycles_compute"].nunique() == 1 and d["mul_ops"].nunique() == 1, cfg
        cyc = float(d["cycles_compute"].iloc[0])
        muls = float(d["mul_ops"].iloc[0])
        out[cfg] = {"cycles": cyc, "mul_ops": muls, "rows": CONFIGS[cfg].rows,
                    "util": muls / (cyc * CONFIGS[cfg].pes)}
    return out


def scaled_bands() -> dict:
    """S3: dc_hbm3 rows. For each closure workload, the factors at which the largest operand exceeds
    the 24 MiB buffer, and over those factors the range of frac_memory_bound and of the HBM3-PIM
    placement speedup; plus the fuzzy factor-4 point that does not exceed the buffer."""
    sc = pd.read_csv(SCALED)
    d = sc[sc["config"] == "dc_hbm3"]
    out = {}
    for w in ("datalog_tc", "automata_rpq", "triangle_count", "fuzzy_inference"):
        x = d[d["workload"] == w].sort_values("factor")
        over = x[x["largest_operand_MB"] > MIB_24]
        out[w] = {"first_factor_over": float(over["factor"].iloc[0]),
                  "mb_at_first_over": float(over["largest_operand_MB"].iloc[0]),
                  "mem_lo": float(over["frac_memory_bound"].min()),
                  "mem_hi": float(over["frac_memory_bound"].max()),
                  "by_factor": {float(r.factor): (float(r.largest_operand_MB), float(r.frac_memory_bound),
                                                  float(r.pim_speedup)) for r in x.itertuples()}}
    return out


def pim_thresholds(bands: dict) -> dict:
    """S3: first factor at which the HBM3-PIM speedup of fuzzy / triangle exceeds 1.1x, the speedup
    range from there on, and the range below it."""
    out = {}
    for w in ("fuzzy_inference", "triangle_count"):
        bf = bands[w]["by_factor"]
        above = {f: v[2] for f, v in bf.items() if v[2] > 1.1}
        below = {f: v[2] for f, v in bf.items() if v[2] <= 1.1}
        out[w] = {"first_factor": min(above), "lo": min(above.values()), "hi": max(above.values()),
                  "below_hi": max(below.values())}
    return out


def op_launch_medians() -> dict:
    """S4: for param op_launch_cycles, spread = max/min of time_ms over the sweep per workload;
    the table reports the median of the spread over the 18 workloads."""
    se = pd.read_csv(SENS)
    o = se[se["param"] == "op_launch_cycles"]
    out = {}
    for cfg in ("dc_hbm3", "dc_hbm3_allpe"):
        g = o[o["config"] == cfg].groupby("workload")["time_ms"].agg(lambda s: s.max() / s.min())
        assert len(g) == 18, (cfg, len(g))
        out[cfg] = float(g.median())
    return out


def row_bandwidth() -> dict:
    """S5: HBM3_PIM.rows_per_s x row_bytes (the derived property used by pim/model.py)."""
    return {"rows_per_s": HBM3_PIM.rows_per_s, "row_bw_Bps": HBM3_PIM.internal_row_bw_Bps}


def test_count() -> dict:
    """S6: static collection count and the retained full-run log."""
    col = subprocess.run([sys.executable, "-m", "pytest", "--collect-only", "-q", "tests/",
                          "-p", "no:cacheprovider"], cwd=ROOT, capture_output=True, text=True).stdout
    m = re.search(r"(\d+) tests collected", col)
    assert m, col[-300:]
    logs = sorted((ROOT / "eval/results").glob("*-pytest-full/pytest.log"))
    assert logs, "no retained pytest log under eval/results/*-pytest-full/"
    last = logs[-1].read_text().strip().splitlines()
    passed = next(l for l in reversed(last) if "passed" in l)
    mp = re.search(r"(\d+) passed", passed)
    date = logs[-1].parent.name[:8]
    return {"collected": int(m.group(1)), "passed": int(mp.group(1)), "log": str(logs[-1].relative_to(ROOT)),
            "date": f"{date[:4]}-{date[4:6]}-{date[6:8]}"}


def _range(lo: float, hi: float, fmt: str = ".1f") -> str:
    """'a–b' with one decimal, or a single value when both ends round to the same string."""
    a, b = format(lo, fmt), format(hi, fmt)
    return a if a == b else f"{a}–{b}"


def build_edits() -> list[tuple[Path, str, str, dict]]:
    r = tpu_ratios()
    h = hmm_dense_step()
    b = scaled_bands()
    p = pim_thresholds(b)
    m = op_launch_medians()
    bw = row_bandwidth()
    t = test_count()
    fz4 = b["fuzzy_inference"]["by_factor"][4.0]
    # chenyi9: decision start — 2026-09-26 ruling: correct the seven audited statements in place;
    # the surrounding text is left unchanged
    edits = [
        (FINDINGS,
         "(129 tests, `python3 -m pytest -q tests/`)",
         f"({t['collected']} tests, `python3 -m pytest -q tests/`; {t['passed']} passed on {t['date']})",
         t),
        (FINDINGS,
         "0.4–1.5 million x (A*,\nSSSP, KG multi-hop, triangle counting)",
         f"{r['graph_bfs_sssp']/1e6:.2g} million x (SSSP),\n{r['triangle_count']/1e6:.2g} million x "
         f"(triangle counting), {r['kg_multihop']/1e6:.2g} million x (KG multi-hop) and "
         f"{r['astar_grid']/1e6:.2g} million x (A*)",
         r),
        (FINDINGS,
         "on a 32x32 array the step costs 3D cycles at 2.3 % utilisation and on 128x128 it is slower\n"
         "in cycles than on 32x32.",
         f"on a 32x32 array the step costs {h['edge28']['cycles']:.0f} cycles (about "
         f"{h['edge28']['cycles']/h['edge28']['rows']:.0f}D) at {100*h['edge28']['util']:.1f} % utilisation and "
         f"on 128x128 it costs\n{h['dc_hbm3']['cycles']:.0f} cycles, slower than on 32x32.",
         h),
        (FINDINGS,
         "triangle, 18 MB for fuzzy) `dc_hbm3` becomes memory-bound (0.69–0.77) and stays there up to 6 GB;",
         f"triangle; fuzzy reaches {b['fuzzy_inference']['mb_at_first_over']:.0f} MB only at factor "
         f"{b['fuzzy_inference']['first_factor_over']:.0f}) `dc_hbm3` becomes memory-bound (share of\n"
         f"memory-bound cycles {b['datalog_tc']['mem_lo']:.2f}–{b['datalog_tc']['mem_hi']:.2f} datalog_tc, "
         f"{b['automata_rpq']['mem_lo']:.2f}–{b['automata_rpq']['mem_hi']:.2f} rpq, "
         f"{b['triangle_count']['mem_lo']:.2f}–{b['triangle_count']['mem_hi']:.2f} triangle, "
         f"{b['fuzzy_inference']['mem_lo']:.2f}–{b['fuzzy_inference']['mem_hi']:.2f} fuzzy from factor "
         f"{b['fuzzy_inference']['first_factor_over']:.0f};\nfuzzy at factor 4 is {fz4[0]:.0f} MB and "
         f"{fz4[1]:.2f}) and stays there up to 6 GB;",
         b),
        (FINDINGS,
         "1.4x (fuzzy, triangle),",
         f"{_range(min(p['fuzzy_inference']['lo'], p['triangle_count']['lo']), max(p['fuzzy_inference']['hi'], p['triangle_count']['hi']))}x (fuzzy from factor "
         f"{p['fuzzy_inference']['first_factor']:.0f}, triangle from factor {p['triangle_count']['first_factor']:.0f}; "
         f"at most {max(p['fuzzy_inference']['below_hi'], p['triangle_count']['below_hi']):.2f}x below),",
         p),
        (FINDINGS,
         "| 2.0x (dc_hbm3), 13.6x (dc_hbm3_allpe) |",
         f"| {m['dc_hbm3']:.1f}x (dc_hbm3), {m['dc_hbm3_allpe']:.1f}x (dc_hbm3_allpe) |",
         m),
        (ARCH_PIM,
         "i.e. 8.5 TB/s of",
         f"i.e. {bw['row_bw_Bps']/1e12:.1f} TB/s of",
         bw),
    ]
    # chenyi9: decision end
    return edits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    edits = build_edits()
    texts = {}
    for path, old, new, values in edits:
        s = texts.get(path) or path.read_text()
        n_old, n_new = s.count(old), s.count(new)
        if n_old == 0 and n_new == 1:
            state = "already applied"
        else:
            assert n_old == 1, f"{path.name}: expected the old sentence once, found {n_old}:\n{old!r}"
            s = s.replace(old, new)
            state = "replaced"
        texts[path] = s
        print(f"[{state}] {path.relative_to(ROOT)}")
        print(f"    old: {old!r}")
        print(f"    new: {new!r}")
        print(f"    values: {values}")
    if not args.dry_run:
        for path, s in texts.items():
            path.write_text(s)
        print("written:", ", ".join(str(p.relative_to(ROOT)) for p in texts))
    else:
        print("dry run: nothing written")


if __name__ == "__main__":
    main()
