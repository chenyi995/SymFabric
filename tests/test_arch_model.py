import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from algebra.trace import OpRecord
from systolicarch import CONFIGS, map_trace, summarize
from systolicarch import model as M


def test_dense_gemm_formula_matches_tessera_closed_form():
    cfg = CONFIGS["dc_hbm3"]              # twist: tiles*max(M,D) + 2D-1
    cyc, det = M.dense_gemm_cycles(cfg, M=256, K=128, N=128)
    assert cyc == 1 * 256 + 2 * 128 - 1
    assert det["tiles"] == 1
    cyc2, _ = M.dense_gemm_cycles(cfg, M=1, K=128, N=128)   # M < D costs D per fold
    assert cyc2 == 128 + 2 * 128 - 1
    tpu = CONFIGS["tpu_like"]                                # classical: tiles*max(M,D) + 3D-2
    cyc3, _ = M.dense_gemm_cycles(tpu, M=256, K=128, N=128)
    assert cyc3 == 256 + 3 * 128 - 2


def test_fracturing_runs_narrow_tiles_in_parallel():
    cfg = CONFIGS["dc_hbm3"]
    # K=1024 rows of weights, N=8 columns: 8 tiles of 128x8 fit side by side in 16 strips
    cyc, det = M.dense_gemm_cycles(cfg, M=64, K=1024, N=8)
    assert det["cols_eff"] == 8 and det["parallel_strips"] == 16 and det["tiles"] == 8
    assert cyc == 1 * 128 + 2 * 128 - 1
    mono = CONFIGS["tpu_like"]
    cyc_m, det_m = M.dense_gemm_cycles(mono, M=64, K=1024, N=8)
    assert det_m["parallel_strips"] == 1 and cyc_m > cyc


def rec(op, n_in, n_out, cand, mul, add, key_bits=24, val_bits=1, hint=None, uids=(1, 2), out=3,
        in_bits=None):
    return OpRecord(op=op, semiring="boolean", n_in=tuple(n_in), n_out=n_out, candidates=cand,
                    mul_ops=mul, add_ops=add, key_bits=key_bits, val_bits=val_bits, hint=hint or {},
                    in_uids=tuple(uids), out_uid=out,
                    in_bits=tuple(in_bits) if in_bits else tuple(key_bits + val_bits for _ in n_in))


def test_sparse_combine_uses_merge_on_reconfigurable_and_dense_on_tpu_like():
    h = {"shared_attrs": 1, "dense": False, "fused_reduce": True, "dom_R": [1000, 1000],
         "dom_S": [1000, 1000], "shared_dom": [1000], "dense_volume": 10 ** 9}
    r = rec("combine", (5000, 5000), 8000, 20000, 20000, 12000, hint=h)
    c_re = map_trace([r], CONFIGS["dc_hbm3"])[0]
    c_tpu = map_trace([r], CONFIGS["tpu_like"])[0]
    assert c_re.mode == "merge_join" and c_tpu.mode == "dense_gemm"
    assert c_tpu.cycles > 50 * c_re.cycles          # dense over 1000^3 vs 30k sparse events


def test_probe_table_too_large_becomes_memory_bound():
    cfg = CONFIGS["edge28"]                         # 1.25 MB SRAM
    big_table = 4_000_000                            # 4 M facts * 4 B = 16 MB > SRAM
    build = rec("scan", (100,), big_table, 100, 0, 0, key_bits=24, val_bits=1, uids=(9,), out=2)
    probe = rec("mask", (200_000, big_table), 100_000, 200_000, 0, 0, uids=(1, 2), out=3)
    costs = map_trace([build, probe], cfg)
    m = costs[1]
    assert m.mode in ("random_probe", "partitioned_probe")
    assert m.bound == "memory"


def test_select_with_pq_is_incremental_after_update():
    cfg = CONFIGS["dc_hbm3"]
    upd = rec("update", (10_000, 50), 40, 50, 0, 50, hint={"state": 10_040, "state_uid": 7}, uids=(5, 6), out=8)
    sel = rec("select", (10_040,), 1, 10_040, 0, 0, hint={"k": 1}, uids=(7,), out=9)
    costs = map_trace([upd, sel], cfg)
    assert costs[1].mode == "systolic_pq"
    assert costs[1].detail["inserts"] == 40 and costs[1].cycles_compute == 41
    tpu = map_trace([upd, sel], CONFIGS["tpu_like"])
    assert tpu[1].mode == "sort_select" and tpu[1].cycles_compute > costs[1].cycles_compute


def test_summary_fields():
    r = rec("filter", (1000,), 500, 1000, 0, 0)
    s = summarize(map_trace([r], CONFIGS["edge28"]), CONFIGS["edge28"])
    assert s["ops"] == 1 and s["cycles"] > 0 and 0 <= s["frac_memory_bound"] <= 1


def test_pop_delete_after_select_is_priced_as_a_pop():
    cfg = CONFIGS["dc_hbm3"]
    upd = rec("update", (5000, 30), 30, 30, 0, 30, hint={"state": 5030, "state_uid": 7}, uids=(5, 6), out=8)
    sel = rec("select", (5030,), 1, 5030, 0, 0, hint={"k": 1}, uids=(7,), out=9)
    pop = rec("mask", (5030, 1), 5029, 5030, 0, 0, hint={"negate": True}, uids=(7, 9), out=10)
    costs = map_trace([upd, sel, pop], cfg)
    assert costs[1].mode == "systolic_pq" and costs[1].cycles_compute == 31
    assert costs[2].mode == "pq_pop_delete" and costs[2].cycles_compute == 1
    tpu = map_trace([upd, sel, pop], CONFIGS["tpu_like"])
    assert tpu[2].mode != "pq_pop_delete" and tpu[2].cycles_compute > 1
