import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import pytest
from pim import PIM_CONFIGS, pim_cost_group, fuse_groups
from pim.model import mq_scan, fused_round
from algebra.trace import OpRecord


def test_internal_row_bandwidth_exceeds_external():
    cfg = PIM_CONFIGS["hbm3_pim"]
    assert cfg.banks == 16 * 2 * 16
    # tFAW binds before tRC: 4 rows per 15 ns per pseudo channel
    assert cfg.rows_per_s == pytest.approx(32 * 4 / (24 * 625e-12))
    assert cfg.internal_row_bw_Bps > 8 * cfg.ext_bw_Bps


def test_mq_scan_is_pe_bound_with_baseline_pe():
    cfg = PIM_CONFIGS["hbm3_pim"]
    c = mq_scan(cfg, n_kb=4_000_000, kb_tuple_bytes=4, n_q=8, q_tuple_bytes=4, n_hits=1000,
                out_tuple_bytes=4)
    assert c.t_pe_s > c.t_rows_s           # 16 lanes at 0.666 GHz cannot keep up with the rows
    fast = PIM_CONFIGS["hbm3_pim_fast"]
    c2 = mq_scan(fast, n_kb=4_000_000, kb_tuple_bytes=4, n_q=8, q_tuple_bytes=4, n_hits=1000,
                 out_tuple_bytes=4)
    assert c2.t_pe_s < c.t_pe_s


def test_query_slots_cause_passes():
    cfg = PIM_CONFIGS["hbm3_pim"]
    a = mq_scan(cfg, 1_000_000, 4, 8, 4, 0, 4)
    b = mq_scan(cfg, 1_000_000, 4, 64, 4, 0, 4)
    assert b.detail["passes"] == 8 and b.t_rows_s == pytest.approx(8 * a.t_rows_s)


def rec(op, n_in, n_out, cand, it=0, in_bits=(25, 25)):
    return OpRecord(op=op, semiring="boolean", n_in=tuple(n_in), n_out=n_out, candidates=cand,
                    mul_ops=cand if op == "combine" else 0, add_ops=0, key_bits=24, val_bits=1,
                    hint={}, iteration=it, in_uids=(1, 2), out_uid=3, in_bits=tuple(in_bits[:len(n_in)]))


def test_fuse_groups_detects_fixpoint_round():
    recs = [rec("combine", (100, 5000), 300, 300), rec("mask", (300, 20000), 120, 300),
            rec("update", (20000, 120), 120, 120), rec("select", (120,), 1, 120)]
    g = fuse_groups(recs)
    assert g == [[0, 1, 2], [3]]
    cfg = PIM_CONFIGS["hbm3_pim"]
    pc = pim_cost_group(cfg, recs, g[0])
    assert pc.shape == "fused_round" and pc.t_total_s > 0
    assert pim_cost_group(cfg, recs, g[1]) is None


def test_multi_stack_shuffle_crosses_the_mesh():
    one = PIM_CONFIGS["hbm3_pim"]
    four = PIM_CONFIGS["hbm3_pim_4s"]
    sixteen = PIM_CONFIGS["hbm3_pim_16s"]
    assert four.mesh_side == 2 and sixteen.mesh_side == 4 and sixteen.mesh_diameter == 6
    a = fused_round(one, 10_000, 4, 5_000_000, 4, 20_000_000, 4, 8_000_000, 4, 100_000)
    b = fused_round(sixteen, 10_000, 4, 5_000_000, 4, 20_000_000, 4, 8_000_000, 4, 100_000)
    assert b.t_pe_s < a.t_pe_s and b.t_rows_s < a.t_rows_s        # 16x the banks
    assert b.detail["stacks"] == 16 and b.t_global_s > a.t_global_s   # mesh diameter latency
