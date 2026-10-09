#!/usr/bin/env python3
"""Run upstream SAM sparse pipelines or SCALE-Sim dense shape experiments."""
import argparse
import configparser
import csv
import json
import os
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/"systolicarch/upstream/sam"), str(ROOT/"systolicarch/upstream/SCALE-Sim")]


def sam_dot(left, right, depth, skip):
    from sam.sim.src.rd_scanner import CompressedCrdRdScan
    from sam.sim.src.joiner import Intersect2
    from sam.sim.src.array import Array
    from sam.sim.src.compute import Multiply2
    from sam.sim.src.accumulator import Reduce
    from sam.sim.src.wr_scanner import ValsWrScan
    opts = dict(statistics=True, back_en=True, depth=depth)
    rd1 = CompressedCrdRdScan(seg_arr=[0, len(left)], crd_arr=left, **opts)
    rd2 = CompressedCrdRdScan(seg_arr=[0, len(right)], crd_arr=right, **opts)
    inter = Intersect2(skip=skip, **opts)
    val1, val2 = Array(init_arr=[1]*len(left), **opts), Array(init_arr=[1]*len(right), **opts)
    mul, reduce = Multiply2(**opts), Reduce(**opts)
    writer = ValsWrScan(size=4, fill=0, **opts)
    blocks = dict(rd1=rd1, rd2=rd2, intersect=inter, val1=val1, val2=val2,
                  multiply=mul, reduce=reduce, writer=writer)
    got_coordinates = []
    # Diagnostic timeout only; the actual cycles are measured by stepping upstream blocks.
    for tick in range(100*(len(left)+len(right)+10)):
        if tick < 2:
            rd1.set_in_ref([0, "D"][tick], "")
            rd2.set_in_ref([0, "D"][tick], "")
        inter.set_in1(rd1.out_ref(), rd1.out_crd(), rd1)
        inter.set_in2(rd2.out_ref(), rd2.out_crd(), rd2)
        if skip:
            rd1.set_crd_skip(inter.out_crd_skip1(), inter)
            rd2.set_crd_skip(inter.out_crd_skip2(), inter)
        val1.set_load(inter.out_ref1(), inter)
        val2.set_load(inter.out_ref2(), inter)
        val = inter.out_crd()
        if isinstance(val, int):
            got_coordinates.append(val)
        mul.set_in1(val1.out_load(), val1)
        mul.set_in2(val2.out_load(), val2)
        reduce.set_in_val(mul.out_val(), mul)
        writer.set_input(reduce.out_val(), reduce)
        for block in blocks.values():
            block.update()
        if writer.out_done():
            break
    else:
        raise TimeoutError("SAM pipeline failed to drain")
    expected = sorted(set(left).intersection(right))
    writer.autosize()
    assert got_coordinates == expected, (got_coordinates, expected)
    assert writer.get_arr() == [len(expected)], (writer.get_arr(), expected)
    return {"cycles": tick+1, "matches": len(expected), "pass": True,
            "stats": {name: block.return_statistics() for name, block in blocks.items()}}


def sam(out):
    import numpy as np
    import scipy.sparse as sp
    rows = []
    # claude: decision start — diagnostic sample includes uniform edges and high-degree edges.
    # Sampling parameters are experiment controls; this is not full-workload cycle timing.
    for name in ("ca-GrQc", "email-Enron", "roadNet-CA"):
        edges = np.loadtxt(ROOT/f"sym1/data/{name}.el", dtype=np.int64)
        n = int(edges.max()+1)
        graph = sp.coo_matrix((np.ones(len(edges)*2),
                              (np.concatenate([edges[:, 0], edges[:, 1]]),
                               np.concatenate([edges[:, 1], edges[:, 0]]))), shape=(n, n)).tocsr()
        degree = np.diff(graph.indptr)
        score = degree[edges[:, 0]] + degree[edges[:, 1]]
        hubs = np.argsort(score)[-16:]
        for seed in (0, 1):
            indices = random.Random(seed).sample(range(len(edges)), 64)
            cases = [(int(i), "uniform") for i in indices]+[(int(i), "hub") for i in hubs]
            for index, kind in cases:
                u, v = map(int, edges[index])
                left = graph.indices[graph.indptr[u]:graph.indptr[u+1]].tolist()
                right = graph.indices[graph.indptr[v]:graph.indptr[v+1]].tolist()
                for depth, skip in ((2, False), (16, False), (2, True)):
                    result = sam_dot(left, right, depth, skip)
                    rows.append(dict(dataset=name, seed=seed, sample=kind, u=u, v=v,
                                     left_nnz=len(left), right_nnz=len(right), depth=depth,
                                     skip=skip, **result))
            print(json.dumps({"dataset": name, "seed": seed, "completed_cases": len(rows)}), flush=True)
            (out/"sam.json").write_text(json.dumps(rows, indent=2)+"\n")
    # claude: decision end


def scale(out):
    from scalesim.scale_sim import scalesim
    template = ROOT/"systolicarch/upstream/SCALE-Sim/configs/scale.cfg"
    topology = out/"shapes.csv"
    shapes = [(q, states, states) for states in (48, 128) for q in (1, 8, 64)]
    topology.write_text("Layer,M,N,K,\n"+"".join(f"q{m}_s{n},{m},{n},{k},\n" for m,n,k in shapes))
    # claude: decision start — this upstream revision requires a layout CSV even when disabled.
    # Source: SCALE-Sim layouts/GEMM_mnk/vit_s_downscale_MK_KN.csv; custom layouts stay disabled.
    layout_source = ROOT/"systolicarch/upstream/SCALE-Sim/layouts/GEMM_mnk/vit_s_downscale_MK_KN.csv"
    layout_lines = layout_source.read_text().splitlines()
    layout_fields = layout_lines[1].split(",")[1:]
    layout = out/"layout.csv"
    layout.write_text(layout_lines[0]+"\n"+"".join(
        ",".join([f"q{m}_s{n}"]+layout_fields)+"\n" for m,n,k in shapes))
    # claude: decision end
    rows = []
    # source: upstream scale.cfg for SRAM/layout; in-repo edge28/dc configs motivate 32/128 arrays.
    # USER mode uses the upstream 10 words/cycle parameter, not a physical HBM claim.
    for size in (32, 128):
        for bandwidth_mode in ("CALC", "USER"):
            config = configparser.ConfigParser()
            config.read(template)
            label = f"array{size}-{bandwidth_mode}"
            config["general"]["run_name"] = label
            config["architecture_presets"]["ArrayHeight"] = str(size)
            config["architecture_presets"]["ArrayWidth"] = str(size)
            config["run_presets"]["InterfaceBandwidth"] = bandwidth_mode
            cfg = out/f"{label}.cfg"
            with cfg.open("w") as f:
                config.write(f)
            sim = scalesim(save_disk_space=True, verbose=False, config=str(cfg),
                           topology=str(topology), layout=str(layout), input_type_gemm=True)
            sim.run_scale(top_path=str(out))
            report = out/label/"COMPUTE_REPORT.csv"
            assert report.exists(), "SCALE-Sim exited without a compute report"
            with report.open() as f:
                for row in csv.DictReader(f, skipinitialspace=True):
                    index = int(row["LayerID"])
                    m, n, k = shapes[index]
                    rows.append({"array": size, "mode": bandwidth_mode, "M": m, "N": n, "K": k,
                                 "report": str(report), "raw": row,
                                 "provenance": "upstream dense scheduling simulation; scalar semiring latency unmodelled"})
            (out/"scale.json").write_text(json.dumps(rows, indent=2)+"\n")
            print(label, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("engine", choices=("sam", "scale"))
    args = p.parse_args()
    globals()[args.engine](Path(os.environ["SYMFABRIC_RUN_DIR"]))
