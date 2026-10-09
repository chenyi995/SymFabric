"""Exact inference adapter: inspect fact descriptors before building states.

The full original capped fact domain is checked for proofs, including matches
beyond K_max. Only the first K_max matching fact successors are materialized.
An optional (predicate, bound arguments) index preserves original fact order.
The remainder of the pinned upstream derivation function is reused unchanged.
"""
from __future__ import annotations

from collections import defaultdict
import hashlib
import inspect
import sys
import textwrap
import time
import types

import torch
import triton
import triton.language as tl
import kge_experiments.unification as u


@triton.jit
def _contains(Keys, key, valid, N: tl.constexpr, STEPS: tl.constexpr):
    lo = tl.full(key.shape, 0, tl.int32)
    hi = tl.full(key.shape, N, tl.int32)
    for _ in range(STEPS):
        mid = (lo + hi) // 2
        value = tl.load(Keys + mid, valid & (mid < N), other=0)
        below = value < key
        move = (lo < hi) & valid
        lo = tl.where(move & below, mid + 1, lo)
        hi = tl.where(move & ~below, mid, hi)
    value = tl.load(Keys + lo, valid & (lo < N), other=-1)
    return valid & (lo < N) & (value == key)


@triton.jit
def _scan(Q, Remaining, Starts, Lengths, Ids, Facts, Hashes, Excluded,
          Active, Matches, Proofs, FMAX: tl.constexpr, G: tl.constexpr,
          Q0: tl.constexpr, Q1: tl.constexpr, R0: tl.constexpr,
          R1: tl.constexpr, R2: tl.constexpr, NF: tl.constexpr,
          NH: tl.constexpr, NSTEPS: tl.constexpr, BASE: tl.constexpr,
          CONST: tl.constexpr, PAD: tl.constexpr, EXCL: tl.constexpr,
          BLOCK: tl.constexpr):
    b = tl.program_id(0)
    x = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    length = tl.load(Lengths + b)
    active = tl.load(Active + b)
    # Uniform block condition avoids fact/state reads for empty lookup tiles.
    if active & (tl.program_id(1) * BLOCK < length):
        start = tl.load(Starts + b)
        ids = tl.load(Ids + start + x, (x < length) & (x < FMAX), other=0)
        p = tl.load(Facts + ids * 3, (x < length) & (x < FMAX), other=PAD)
        a = tl.load(Facts + ids * 3 + 1, (x < length) & (x < FMAX), other=PAD)
        c = tl.load(Facts + ids * 3 + 2, (x < length) & (x < FMAX), other=PAD)
        qp = tl.load(Q + b * Q0)
        qa = tl.load(Q + b * Q0 + Q1)
        qc = tl.load(Q + b * Q0 + 2 * Q1)
        va = (qa > CONST) & (qa != PAD)
        vc = (qc > CONST) & (qc != PAD)
        ok = (x < length) & (x < FMAX) & (p == qp)
        ok &= (va | (qa == a)) & (vc | (qc == c))
        ok &= ~((va & vc & (qa == qc)) & (a != c) & (a != 0) & (c != 0))
        if EXCL:
            ep = tl.load(Excluded + b * 3)
            ea = tl.load(Excluded + b * 3 + 1)
            ec = tl.load(Excluded + b * 3 + 2)
            ok &= ~((p == ep) & (a == ea) & (c == ec))
        proof = ok
        # Successor atoms live in registers: no [B,FMAX,G,3] state tensor.
        for atom in range(G):
            rp = tl.load(Remaining + b * R0 + atom * R1)
            ra = tl.load(Remaining + b * R0 + atom * R1 + R2)
            rc = tl.load(Remaining + b * R0 + atom * R1 + 2 * R2)
            # Facts are ground, so variable bindings cannot alias another var.
            sa = tl.where(va & (ra == qa) & (a != 0), a, ra)
            sc = tl.where(va & (rc == qa) & (a != 0), a, rc)
            sa = tl.where(vc & (sa == qc) & (c != 0), c, sa)
            sc = tl.where(vc & (sc == qc) & (c != 0), c, sc)
            valid_atom = (rp != PAD) & (rp != 0)
            can_lookup = proof & valid_atom & (sa <= CONST) & (sc <= CONST)
            key = ((rp.to(tl.int64) * BASE) + sa.to(tl.int64)) * BASE + sc.to(tl.int64)
            known = _contains(Hashes, key, can_lookup, NH, NSTEPS)
            if EXCL:
                known &= ~((rp == ep) & (sa == ea) & (sc == ec))
            proof &= (~valid_atom) | known
        tl.store(Matches + b * FMAX + x, ok, x < FMAX)
        tl.store(Proofs + b * FMAX + x, proof, x < FMAX)
    else:
        tl.store(Matches + b * FMAX + x, False, x < FMAX)
        tl.store(Proofs + b * FMAX + x, False, x < FMAX)


@triton.jit
def _select(Matches, Proofs, Starts, Ids, Selected, ProofBatch,
            FMAX: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    valid = tl.load(Matches + b * FMAX + x, x < FMAX, other=0).to(tl.int1)
    proof = tl.load(Proofs + b * FMAX + x, x < FMAX, other=0).to(tl.int1)
    rank = tl.cumsum(valid.to(tl.int32)) - 1
    start = tl.load(Starts + b)
    ids = tl.load(Ids + start + x, valid & (rank < K), other=-1)
    tl.store(Selected + b * K + rank, ids, valid & (rank < K))
    tl.store(ProofBatch + b, tl.sum(proof.to(tl.int32)) > 0)


def fact_stage(engine, queries, remaining, remaining_counts, active, excluded):
    b = queries.shape[0]
    if engine.descriptor_mode == "indexed":
        base = engine.argument_base
        args = torch.where(queries[:, 1:] > engine.constant_no,
                           engine.constant_no + 1, queries[:, 1:])
        key = (queries[:, 0] * base + args[:, 0]) * base + args[:, 1]
        pos = torch.searchsorted(engine.lookup_keys, key)
        safe = pos.clamp(max=engine.lookup_keys.numel() - 1)
        found = (pos < engine.lookup_keys.numel()) & (engine.lookup_keys[safe] == key)
        starts = engine.lookup_starts[safe]
        lengths = torch.where(found, engine.lookup_lengths[safe], 0)
        ids = engine.lookup_ids
    else:
        starts = engine.fact_seg_starts[queries[:, 0]].contiguous()
        lengths = engine.fact_seg_lens[queries[:, 0]].clamp(max=engine.max_fact_pairs).contiguous()
        ids = engine.identity_ids
    matches = torch.empty((b, engine.max_fact_pairs), dtype=torch.bool, device=queries.device)
    proofs = torch.empty_like(matches)
    selected = torch.full((b, engine.K_max), -1, dtype=torch.long, device=queries.device)
    proof_batch = torch.empty(b, dtype=torch.bool, device=queries.device)
    # Views passed from EnvVec can have non-contiguous row strides.
    excl = excluded[:, 0, :].contiguous() if excluded is not None else queries
    nf = engine.facts_idx.shape[0]
    nh = engine.fact_hashes.numel()
    _scan[(b, triton.cdiv(engine.max_fact_pairs, 128))](
        queries, remaining, starts, lengths, ids, engine.facts_idx,
        engine.fact_hashes, excl, active, matches, proofs,
        engine.max_fact_pairs, remaining.shape[1], *queries.stride(), *remaining.stride(),
        nf, nh, max(1, nh.bit_length() + 1), engine.pack_base,
        engine.constant_no, engine.padding_idx, excluded is not None, 128)
    _select[(b,)](matches, proofs, starts, ids, selected, proof_batch,
                  engine.max_fact_pairs, engine.K_max,
                  triton.next_power_of_2(engine.max_fact_pairs), num_warps=8)
    valid = selected >= 0
    states, success, _ = u.unify_with_facts(
        queries, remaining, remaining_counts, selected.clamp(min=0), valid,
        engine.facts_idx, engine.constant_no, engine.padding_idx)
    return states, success, proof_batch


def install(engine, mode="indexed"):
    """Modify a newly created engine only, leaving the native control intact."""
    if mode not in ("compact", "indexed") or engine.parity_mode:
        raise ValueError("This experiment supports compact/indexed non-parity inference")
    if engine.padding_idx != 0 or engine.facts_idx.numel() == 0:
        raise ValueError("Adapter requires the recorded zero padding and nonempty facts")
    started = time.perf_counter()
    facts = engine.facts_idx.cpu().tolist()
    if any(a > engine.constant_no or c > engine.constant_no for _, a, c in facts):
        raise ValueError("Fact descriptors require ground facts")
    engine.descriptor_mode = mode
    engine.identity_ids = torch.arange(len(facts), device=engine.device)
    engine.argument_base = engine.constant_no + 2
    groups = defaultdict(list)
    starts = engine.fact_seg_starts.cpu().tolist()
    lengths = engine.fact_seg_lens.cpu().tolist()
    if mode == "indexed":
        wildcard = engine.constant_no + 1
        base = engine.argument_base
        # Index only the ORIGINAL capped prefix, after upstream shuffling.
        for pred, (start, length) in enumerate(zip(starts, lengths)):
            for rid in range(start, start + min(length, engine.max_fact_pairs)):
                p, a, c = facts[rid]
                assert p == pred
                for x, y in ((a, c), (a, wildcard), (wildcard, c), (wildcard, wildcard)):
                    groups[(p * base + x) * base + y].append(rid)
        keys, offsets, counts, rows = [], [], [], []
        for key, members in sorted(groups.items()):
            keys.append(key); offsets.append(len(rows)); counts.append(len(members)); rows.extend(members)
        for name, values in (("keys", keys), ("starts", offsets), ("lengths", counts), ("ids", rows)):
            setattr(engine, "lookup_" + name, torch.tensor(values, dtype=torch.long, device=engine.device))
    # Preserve upstream terminal handling, rule expansion, pruning, variable
    # renaming and first-K packing. Only replace its fact-generation section.
    source = textwrap.dedent(inspect.getsource(u.UnificationEngineVectorized.get_derived_states_compiled))
    first = source.index("    fact_item_idx, fact_valid, _ = pairs_via_predicate_ranges(")
    last = source.index("    rule_item_idx, rule_valid, _ = pairs_via_predicate_ranges(")
    source = source[:first] + (
        "    fact_states, fact_success, fact_proof_batch = _descriptor_fact_stage(\n"
        "        self, queries, remaining, remaining_counts, active_mask, excluded_queries)\n\n"
    ) + source[last:]
    # Dynamo resolves generated-function globals through __module__. Give the
    # adapter a real module rather than pretending its new globals live in u.
    module_name = '_symfabric_descriptor_' + str(id(engine))
    module = types.ModuleType(module_name)
    namespace = module.__dict__
    namespace.update(vars(u))
    namespace['__name__'] = module_name
    namespace["_descriptor_fact_stage"] = fact_stage
    sys.modules[module_name] = module
    exec(compile(source, "<dprl_descriptor_fact_adapter>", "exec"), namespace)
    engine.get_derived_states_compiled = types.MethodType(namespace["get_derived_states_compiled"], engine)
    if engine.facts_idx.is_cuda:
        torch.cuda.synchronize()
    size = sum(getattr(engine, "lookup_" + n).numel() * 8 for n in ("keys", "starts", "lengths", "ids")) if mode == "indexed" else 0
    return {"mode": mode, "build_s": time.perf_counter() - started,
            "additional_index_bytes": size, "lookup_keys": len(groups),
            "generated_source": source,
            "generated_source_sha256": hashlib.sha256(source.encode()).hexdigest(),
            "contract": "unchanged capped domain/order; proof detection includes overflow candidates"}
