# planning_hadd — delete-relaxation heuristics h^add and h^max

**Category**: planning · **Semiring(s)**: minplus (min,+), bottleneck (min,max) · **Execution structure**: sparse-fixpoint
**Module**: `softwareworkload/planning_hadd/workload.py` · **Test**: `tests/test_workload_planning_hadd.py`

## 1. Problem and reference algorithm
A STRIPS task has P propositions, A actions (1–3 preconditions, 1–2 add effects, integer cost
1–10) and an initial state I. The delete-relaxation heuristics are the least fixpoint of

    h_add(p) = min_{a : p ∈ add(a)} [ cost(a) + Σ_{q ∈ pre(a)} h_add(q) ]
    h_max(p) = min_{a : p ∈ add(a)} [ cost(a) + max_{q ∈ pre(a)} h_max(q) ]

with value 0 on I and +inf for propositions that are never reached. `native()` is the queue-based
generalised Dijkstra with one counter of unsatisfied preconditions per action, the way Fast
Downward evaluates h^add / h^max: pop the cheapest unfinalised proposition, decrement the counter
of every action that has it as a precondition and fold its cost into the action's accumulator,
and when a counter reaches zero relax the action's add effects. Each proposition is finalised
once, so every (action, precondition) pair and every (fired action, add effect) pair is handled
exactly once.

Instances (`make_instance`): propositions get a random rank, an action's preconditions have lower
rank than its add effects (a random DAG, so most propositions are reachable), the 5% lowest ranks
form I, the 5% highest ranks have no achiever ("dead"), and 3% of the actions carry one dead
precondition so that the complete-group rule is exercised. Scales: small P=400 / A=1,500, large
P=4,000 / A=15,000 (preconditions uniform over all lower ranks), and deep = large with
preconditions drawn from a window of 50 ranks below the effect (long causal chains; added to make
the depth sensitivity of section 7 reproducible through the harness).

## 2. Mathematical decomposition
State c(q) (absent = +inf). Per round, over the relations pre(a,q), add(a,p) with value cost(a):

    pre_cost(a) = ⊗_{q ∈ pre(a)} c(q)                      fold with the semiring ⊗, defined only
                                                           when pre(a) ⊆ supp(c) (complete group)
    ach(p)      = ⊕_{a : p ∈ add(a)} cost(a) ⊗ pre_cost(a)   ⊕ = min, ⊗ = +
    c'          = c ⊕ ach                                   fixpoint until c' = c

h_add is the (min,+) instance: fold ⊗ = + and composition ⊗ = +. h_max needs fold = max but
composition = +, i.e. two different ⊗ in one round; the fold is run in (min,max) and the rest in
(min,+). The fold is not invertible in either semiring, so the semi-naive round recomputes an
action from all its preconditions whenever one of them changed (Δ = changed propositions).

## 3. Algebra program
One round of `fixpoint(step, c_0)`, c_0 = {(q): 0 | q ∈ I}, Δ_0 = c_0; `one` = 0.0 in minplus:

    D1       = scan(Δ, const_one)                        (q)->one          support of Δ
    touched  = combine(pre(a,q), D1(q), keep=(a,))        (a,q)x(q)->(a)    actions with a changed pre
    pre_t    = combine(touched(a), pre(a,q))              (a)x(a,q)->(a,q)  every pre of those actions
    J        = combine(pre_t(a,q), c(q))                  (a,q)x(q)->(a,q)  c(q) for reachable q
    [h_max]  J = scan(J, identity, sr=bottleneck)         re-type for the max fold
    pre_cost = reduce(J, keep=(a,), use_mul=True)         (a,q)->(a)        fold: + / max
    [h_max]  pre_cost = scan(pre_cost, identity, sr=minplus)
    missing  = mask(pre_t, c, negate=True)                (a,q) with unreachable q
    bad      = reduce(missing, keep=(a,))                 (a)               incomplete actions
    ok       = mask(pre_cost, bad, negate=True)           (a)               complete actions only
    ach      = combine(add(a,p)[cost], ok(a), keep=(p,))  (a,p)x(a)->(p)    cost(a)+pre_cost(a), min over a
    c, Δ     = update(c, rename(ach, p->q))

`algebra()` runs the program twice: (minplus, minplus) for h_add and (bottleneck fold, minplus
composition) for h_max. `algebra_bottleneck()` runs it unchanged in bottleneck (section 7).

## 4. Hotspot event and work efficiency
Event (both sides, summed over h_add and h_max): one (action, precondition) accumulation — the
golden's counter decrement plus fold, the algebra's fold of one J tuple — plus one (action,
add-effect) relaxation when the action fires — the golden's `c(p) = min(c(p), val)`, the
algebra's `ach` candidate. native_ops = decrements + relaxations; the algebra side is
`trace.mul_ops` over the whole program.

Measured (harness, 2026-09-25; small = 3 seeds, range; large and deep = seed 0):

| scale | ratio_mul | ratio_sr | ratio_cand | rounds h_add / h_max | re-touch factor add / max | fire factor add / max | t_native / t_algebra (s) |
|---|---|---|---|---|---|---|---|
| small (seeds 0–2) | 3.904–4.213 | 4.624–4.976 | 7.432–8.051 | 5–7 / 6–7 | 1.21–1.25 / 1.25–1.32 | 1.32–1.40 / 1.37–1.51 | 0.003 / 0.09 (seed 0) |
| large (seed 0) | 4.018 | 4.758 | 7.667 | 6 / 6 | 1.23 / 1.29 | 1.34 / 1.43 | 0.037 / 1.06 |
| deep (seed 0) | 6.518 | 7.720 | 12.423 | 14 / 14 | 1.85 / 2.00 | 2.18 / 2.42 | 0.045 / 1.67 |

Re-touch factor = (pre, Δ) pairs of the `touched` join per counter decrement of the golden; fire
factor = `ach` candidates per relaxation of the golden. Small seed 0: native_ops 10,216
(decrements 2,939 + relaxations 2,169 per heuristic), 380 of 400
propositions reachable; large: 3,794 of 4,000; deep: 3,795 of 4,000.

Where the ~4x of ratio_mul comes from (small seed 0, ⊗ counted per phase; derived from the
harness phase counters in `extra`):

| phase | h_add ⊗ | h_add candidates | h_max ⊗ | h_max candidates |
|---|---|---|---|---|
| touch (D1 scan + pre ⋈ Δ) | 3,632 | 4,139 | 3,714 | 4,237 |
| gather (touched ⋈ pre) | 6,008 | 6,008 | 6,124 | 6,124 |
| fold (pre_t ⋈ c, casts, reduce) | 7,280 | 10,018 | 7,449 | 18,166 |
| complete (antijoin, reduce, mask) | 0 | 9,745 | 0 | 9,922 |
| fire (add ⋈ ok) | 2,892 | 2,892 | 2,977 | 2,977 |
| update | 0 | 1,111 | 0 | 1,124 |

Per touched (action, precondition) tuple the algebra performs four ⊗-counted operations where the
golden performs one fold: the touch join (one ⊗ per (pre, changed q) pair, identity value), the
gather join (one ⊗ per precondition of a touched action, including unreachable ones), the join
with c (one ⊗ per reachable precondition, identity value) and the fold itself (the only real ⊗).
The three identity ⊗ are index operations that the trace counts as multiplications. Multiplied by
the re-touch factor (1.21–1.32 on the uniform tasks, small seeds 0–2 and large;
1.85–2.00 on the deep task) and adding the fire phase (1.32–1.51 candidates
per golden relaxation on the uniform tasks, 2.18–2.42 on the deep task, because a
touched complete action re-emits its add effects even when its cost did not improve), this gives
the measured ratio_mul. ratio_sr adds the min-merges of the fused reduces and of `update`.
ratio_cand adds the passes with no arithmetic: the antijoin over pre_t, the two masks, the D1
scan, `update`, and for h_max the two casts (h_max fold candidates 18,166 versus
10,018 for h_add at small seed 0).

## 5. scan() functions used
- `const_one(k, v) -> [(k, one)]`: keeps the key, replaces the value by the semiring one; O(1).
  Turns Δ into a support relation so that the touch join does not multiply by the changed cost.
- `identity(k, v) -> [(k, v)]`: re-types a relation into another semiring (h_max only, twice per
  round); O(1) per tuple, no arithmetic.

## 6. Control residue
None beyond the fixpoint termination test (|Δ| = 0, one scalar per round). Rounds: small
5–7 (h_add) / 6–7 (h_max), large 6 / 6, deep 14 / 14.
The bottleneck diagnostic (`algebra_bottleneck`) is not part of `algebra()`.

## 7. Where the algebra is insufficient or awkward
1. **h^max is not one semiring.** "Same program, different semiring" with bottleneck (min,max)
   computes min_a max(cost(a), max_q c(q)), a bottleneck heuristic. Measured (diagnostic in
   `tests/test_workload_planning_hadd.py::test_same_program_in_bottleneck_is_not_hmax`, small,
   seed 0, 2026-09-25): the pure-bottleneck run equals its own Dijkstra golden
   (True) and differs from h^max on 282 of 380 reachable propositions.
   h^max needs ⊗_fold = max and ⊗_comp = +; this program pays two identity casts per round
   because `combine` / `reduce` require both operands in one semiring object. A hardware REDUCE
   would need a fold operator selectable independently of the pipeline's ⊗ (or a two-⊗ value
   algebra); the casts themselves are free in hardware (same bits, different PE op).
2. **No fused combine-with-⊗-fold.** `combine(..., keep=)` fuses only the ⊕-reduction. The
   precondition fold needs J materialised and a second `reduce(use_mul=True)` pass: two ⊗ per
   (a,q) tuple instead of one.
3. **Support-only joins are counted as ⊗.** `touched` and `pre_t` carry the value `one`; they are
   index operations (which actions does q feed, which q does a read). Without an INDEX / semijoin
   that emits the joined tuples, half of the counted ⊗ are identities.
4. **Round-based fixpoint versus Dijkstra.** The golden evaluates each action once; the fixpoint
   re-touches an action every time a precondition improves. On the uniform random tasks this costs
   a factor 1.21–1.32; on the deep task 1.85–2.00, and ratio_cand rises to
   12.423, above the plan's 10x admission threshold (section 4.5). A Dijkstra-ordered
   program needs `select_k(1)` over the frontier every round: P rounds with a frontier of up to P
   entries is O(P^2) select candidates (derived, not measured), so the algebra has no cheap
   equivalent of the priority queue — the plan's class K operator.
5. **Complete-group check.** Implemented as an antijoin on supp(c) (`missing`, `bad`, `ok`)
   rather than a COUNTING comparison against |pre(a)|: the count version needs a second semiring,
   a scan that re-keys the count to (a, n) and a mask against a static (a, |pre(a)|) table — one
   more pass over the touched actions; the antijoin is "∀q ∈ pre(a): q ∈ supp(c)" as ¬∃ and costs
   one support probe per precondition of a touched action (complete phase: 9,745 of
   76,463 candidates at small seed 0). Both are O(|pre_t|) per round; neither is free.

## 8. Shape seen by the hardware
Sparse relations throughout. pre has 1–3 tuples per action (~2,986 at small,
~30,011 at large), add 1–2 per action; c grows to the reachable set (380 /
3,794); Δ per round for h_add at small seed 0: [156, 219, 85, 20, 6, 1, 0], at deep seed 0:
[112, 306, 635, 1165, 1633, 1345, 809, 399, 170, 58, 11, 7, 1, 0] (h_max: [112, 306, 637, 1179, 1653, 1402, 898, 489, 273, 135, 57, 15, 3, 0]). The static operand pre is probed three times per
round (by q in touch, by a in gather, by q in the antijoin) and add once: a weight-stationary
placement of pre indexed both ways is the natural mapping. h_add and h_max are the same dataflow
with a different fold op in one PE stage; their Δ sequences differ slightly, so they run as two
fixpoints here. Output is a dense vector over propositions; no multi-query sharing in this
formulation (a batch of states evaluated against one task would add a Q dimension sharing pre and
add).
