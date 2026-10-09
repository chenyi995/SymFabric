"""Common driver for workload modules.

A workload module (softwareworkload/<name>/workload.py) exports:

  NAME            : str, equals the directory name
  CATEGORY        : one of logic | probabilistic | graph | search | planning | vsa | automata | parsing
  SEMIRINGS       : list of semiring names used
  STRUCTURE       : short label of the execution structure, e.g. "sparse-fixpoint",
                    "dense-iterate", "dag", "frontier-fixpoint", "select-loop"
  HOTSPOT_EVENT   : one sentence naming the event that `native` counts (the same event the
                    algebra's semiring operations correspond to)
  SCALES          : dict scale_name -> parameters (at least "small" and "large")
  make_instance(scale, seed) -> inst
  native(inst)   -> (result, native_ops)  independent golden, plain Python / numpy, never
                                            imports algebra.ops; counts HOTSPOT_EVENT
  algebra(inst)  -> result                 uses only algebra operators; runs inside a trace
  compare(res_native, res_algebra) -> bool

Ratios reported (all relative to native_ops, the hotspot events of the reference algorithm):
  ratio_mul  = trace.mul_ops   / native_ops   PRIMARY: one hotspot event <-> one semiring mul
                                               (a MAC, a relaxation, a clause-literal check ...)
  ratio_sr   = (mul + add ops) / native_ops   arithmetic including the merges
  ratio_cand = all candidates  / native_ops   everything touched, including mask/filter/select/update
The admission threshold of the plan (work-efficiency <= 10x) is checked on ratio_cand, the most
pessimistic of the three.
"""
from __future__ import annotations

import importlib
import json
import pkgutil
import sys
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from algebra import use_trace  # noqa: E402


@dataclass
class Result:
    name: str
    category: str
    scale: str
    seed: int
    ok: bool
    native_ops: int
    algebra_mul_ops: int
    algebra_sr_ops: int
    algebra_candidates: int
    ratio_mul: float
    ratio_sr: float
    ratio_cand: float
    n_records: int
    by_op: dict
    semirings: list
    structure: str
    t_native_s: float
    t_algebra_s: float
    extra: dict = field(default_factory=dict)

    def as_dict(self):
        return asdict(self)


def discover() -> list:
    """Import every softwareworkload/<name>/workload.py module."""
    mods = []
    base = Path(__file__).resolve().parent
    for d in sorted(p for p in base.iterdir() if p.is_dir() and not p.name.startswith(("_", "."))
                    and p.name != "archived"):
        if (d / "workload.py").exists():
            mods.append(importlib.import_module(f"softwareworkload.{d.name}.workload"))
    return mods


def run(mod, scale: str = "small", seed: int = 0, keep_trace: bool = False):
    inst = mod.make_instance(scale, seed)
    t0 = time.perf_counter()
    res_n, native_ops = mod.native(inst)
    t1 = time.perf_counter()
    with use_trace() as tr:
        res_a = mod.algebra(inst)
    t2 = time.perf_counter()
    ok = bool(mod.compare(res_n, res_a))
    s = tr.summary()
    r = Result(
        name=mod.NAME, category=mod.CATEGORY, scale=scale, seed=seed, ok=ok,
        native_ops=int(native_ops), algebra_mul_ops=s["mul_ops"], algebra_sr_ops=s["sr_ops"],
        algebra_candidates=s["candidates"],
        ratio_mul=(s["mul_ops"] / native_ops) if native_ops else float("nan"),
        ratio_sr=(s["sr_ops"] / native_ops) if native_ops else float("nan"),
        ratio_cand=(s["candidates"] / native_ops) if native_ops else float("nan"),
        n_records=s["records"], by_op=s["by_op"], semirings=list(mod.SEMIRINGS),
        structure=mod.STRUCTURE, t_native_s=t1 - t0, t_algebra_s=t2 - t1,
        extra=getattr(mod, "extra_info", lambda inst, res: {})(inst, res_a),
    )
    return (r, tr) if keep_trace else r


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default="small")
    ap.add_argument("--only", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    for m in discover():
        if args.only and m.NAME != args.only:
            continue
        r = run(m, args.scale, args.seed)
        print(json.dumps({k: v for k, v in r.as_dict().items() if k != "by_op"}, default=str))
