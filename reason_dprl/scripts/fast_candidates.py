"""Exact first-K candidate packing without contended invalid-slot atomics.

The semantic oracle is stable_candidates.pack_results. No candidate is pruned
and the fact-then-rule order and original capacity are retained. Triton kernels
are called directly so torch.compile can incorporate them (PyTorch's documented
user-defined Triton integration). This is an inference-only adapter.
"""
import torch
import triton
import triton.language as tl


@triton.jit
def _indices(FM, RM, Selected, Counts, NF: tl.constexpr, NR: tl.constexpr,
             F0: tl.constexpr, F1: tl.constexpr, R0: tl.constexpr, R1: tl.constexpr,
             CAP: tl.constexpr, BLOCK: tl.constexpr):
    b = tl.program_id(0)
    x = tl.arange(0, BLOCK)
    f = tl.load(FM + b * F0 + x * F1, x < NF, other=0).to(tl.int1)
    r = tl.load(RM + b * R0 + (x - NF) * R1,
                (x >= NF) & (x < NF + NR), other=0).to(tl.int1)
    valid = f | r
    rank = tl.cumsum(valid.to(tl.int32), 0) - 1
    # Every retained destination has exactly one writer. Invalid lanes do not
    # write to a shared sink, removing both the race and atomic contention.
    tl.store(Selected + b * CAP + rank, x, valid & (rank < CAP))
    count = tl.minimum(tl.sum(valid.to(tl.int32), 0), CAP)
    tl.store(Counts + b, count)


@triton.jit
def _gather(Facts, Rules, Selected, Output, TOTAL: tl.constexpr,
            NF: tl.constexpr, NR: tl.constexpr, CAP: tl.constexpr, ATOMS: tl.constexpr,
            FA: tl.constexpr, RA: tl.constexpr,
            F0: tl.constexpr, F1: tl.constexpr, F2: tl.constexpr, F3: tl.constexpr,
            R0: tl.constexpr, R1: tl.constexpr, R2: tl.constexpr, R3: tl.constexpr,
            PAD: tl.constexpr, BLOCK: tl.constexpr):
    x = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    b = x // (CAP * ATOMS * 3)
    k = x // (ATOMS * 3) % CAP
    a = x // 3 % ATOMS
    term = x % 3
    selected = tl.load(Selected + b * CAP + k, x < TOTAL, other=NF+NR)
    fv = tl.load(Facts + b*F0 + selected*F1 + a*F2 + term*F3,
                 (x < TOTAL) & (selected < NF) & (a < FA), other=PAD)
    rv = tl.load(Rules + b*R0 + (selected-NF)*R1 + a*R2 + term*R3,
                 (x < TOTAL) & (selected >= NF) & (selected < NF+NR) & (a < RA), other=PAD)
    value = tl.where(selected < NF, fv, rv)
    tl.store(Output + x, value, x < TOTAL)


def pack_results(fact_states, fact_mask, rule_states, rule_mask, K_max, M_max, padding_idx):
    b, nf, fa, _ = fact_states.shape
    nr, ra = rule_states.shape[1:3]
    if not fact_states.is_cuda or nf + nr == 0:
        from stable_candidates import pack_results as reference
        return reference(fact_states, fact_mask, rule_states, rule_mask, K_max, M_max, padding_idx)
    selected = torch.full((b, K_max), nf+nr, dtype=torch.long, device=fact_states.device)
    counts = torch.empty(b, dtype=torch.long, device=fact_states.device)
    result = torch.empty((b, K_max, M_max, 3), dtype=fact_states.dtype, device=fact_states.device)
    _indices[(b,)](fact_mask, rule_mask, selected, counts, nf, nr,
                  *fact_mask.stride(), *rule_mask.stride(), K_max,
                  triton.next_power_of_2(nf+nr), num_warps=8)
    total = b*K_max*M_max*3
    _gather[(triton.cdiv(total, 256),)](fact_states, rule_states, selected, result,
        total, nf, nr, K_max, M_max, fa, ra, *fact_states.stride(), *rule_states.stride(),
        padding_idx, 256)
    return result, counts


def install():
    import kge_experiments.unification as unification
    unification.pack_results = pack_results
    return {'name': 'triton_stable_first_k', 'semantics': 'identical to stable_candidates',
            'change': 'masked unique writes and direct gather; no invalid-slot atomic reduction'}
