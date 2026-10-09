# sat_bcp — unit propagation for Q cubes over one clause database

**Category**: logic · **Semiring(s)**: boolean, counting, minplus (grouped select-1) · **Execution structure**: frontier-fixpoint
**Module**: `softwareworkload/sat_bcp/workload.py` · **Test**: `tests/test_workload_sat_bcp.py`

## 1. Problem and reference algorithm
Random k-SAT CNF with n variables and m clauses (85% 3-literal, 10% 4-literal, 5% 5-literal
clauses on distinct variables, random polarity) and Q cubes, each c literals on distinct
variables. Literal encoding 2v / 2v+1, negation l ^ 1. For every cube: propagate units to the
fixpoint or to a conflict. Small: n = 300, m = 1200, Q = 16, c = 34. Large: n = 2000, m = 8000,
Q = 64, c = 140. The plan's c = 12 / 20 gave trivial propagation (measured seed 0: 0 conflicts,
2.4 / 0.4 implied literals per cube, 1,008 / 5,352 clause visits), so c was raised until some
cubes conflict and the rest propagate tens of literals, as the plan asks. Measured seed 0: small
7 of 16 cubes conflict (0.44), the others imply 5-67 literals (mean 17.3); large 16 of 64 (0.25),
17-67 literals (mean 36.1). Seeds 1-2 small: 4 and 3 conflicts, means 21.1 and 16.2.

`native()` is a two-watched-literal propagator (MiniSat scheme): per cube a fresh trail, watch
lists initialised from the first two literals of every clause, a propagation head; when a literal
becomes false every clause watching it is visited: if the other watch is true the watch stays;
otherwise the first non-false literal beyond the two watches becomes the new watch; otherwise the
clause is unit (other watch unassigned) or conflicting (other watch false). Hotspot event:
clause visits. Secondary event: literal visits (the other-watch look-up plus every literal
examined in the replacement search).

Result per cube: (conflict, set of true literals). Unit propagation is confluent, so without a
conflict the fixpoint set is unique whatever the order; a conflict's existence is also order-
independent (an all-false clause stays all-false under every extension), but the partial
assignment at which it is detected is not, so on conflict only the verdict is compared.

## 2. Mathematical decomposition
Relations: clause(c, w, l) [boolean; w = width(c) travels in the key], asg(q, l) [true literals of
cube q], nasg(q, l) = asg(q, ¬l) [false literals]. Per round:
    false_count(q, c, w) = Σ_l clause(c, w, l) · nasg(q, l)          (counting)
    sat(q, c)            = ⋁_l clause(c, w, l) ∧ asg(q, l)            (boolean)
    unit(q, c, w)        = [false_count = w − 1] ∧ ¬sat
    conflict(q, c, w)    = [false_count = w]                          (¬sat is implied: asg is consistent)
    implied(q, l)        = ⋁_c unit(q, c) ∧ clause(c, l) ∧ ¬nasg(q, l)  (the one non-false literal)
    conf(q)             ∨= ∃c conflict(q, c)  ∨  ∃l implied(q, l) ∧ implied(q, ¬l)
    asg, Δ               = update(asg, implied ∧ ¬conf)
Fixpoint until Δ = ∅, over all Q cubes at once (the Q queries share clause).
(a) evaluates these equations on the full asg every round. (b) evaluates them only for the
clauses touched by the newly false literals Δfalse(q, l) = ¬Δ(q, l):
    touched(q, l, c) = Δfalse(q, l) ∧ watch(q, c, l)
and maintains watch: for a touched clause that is neither satisfied, unit nor conflicting, the
fallen watch is replaced by min{ l' : clause(c, l'), ¬nasg(q, l'), ¬watch(q, c, l') } — a reduce
under (min, +) with the literal id as value; if both watches fell, a second minimum excluding the
first. Watches on false literals are never deleted: a literal becomes false once, so the tuple is
never touched again.

## 3. Algebra program
(b) watched form, `algebra` (one round; state: ASG, NASG, WATCH(q,c,l), CONF(q)):
    DF    = scan(Δ, negate_literal)                                    # (q,l) newly false
    T3    = combine(DF(q,l), WATCH(q,c,l))                             # (q,l,c) touched  [∝ clause visits]
    TOUCH = reduce(T3, keep=(q,c))                                     # boolean dedupe
    KF    = scan(T3, count_fallen_watches)                             # (q,c) -> 1 or 2   (counting)
    TCL   = combine(TOUCH(q,c), CLAUSE(c,w,l))                         # all literals of touched clauses
    SAT   = combine(TCL, ASG(q,l), keep=(q,c))                         # satisfied touched clauses
    UNS   = mask(TCL, SAT, negate=True)                                # literals of unsatisfied ones
    FL    = mask(UNS, NASG);   FC = scan(FL, count_false_literals)     # (q,c,w) -> false_count
    UNIT  = filter(FC, v == w-1);   CONF = filter(FC, v == w)
    NF    = mask(UNS, NASG, negate=True)                               # non-false literals
    IMPL  = reduce(mask(NF, UNIT), keep=(q,l))                         # the implied literal per unit clause
    NEED  = mask(mask(NF, UNIT, negate=True), CONF, negate=True)       # clauses needing a new watch
    CANDL = mask(NEED, WATCH, negate=True)                             # non-false, non-watched literals
    P1    = reduce(scan(CANDL, literal_as_value [minplus]), keep=(q,c))   # grouped select-1 (min literal)
    NW1   = scan(P1, value_to_key)                                     # (q,c,l) new watches
    NW2   = same on mask(mask(CANDL, filter(KF, v == 2)), NW1, negate=True)   # second watch if both fell
    WATCH, _ = update(WATCH, update(NW1, NW2))
    CONF  ∪= scan(CONF_clauses, to_cube) ∪ scan(combine(IMPL, scan(IMPL, negate_literal)), to_cube)
    ASG, Δ = update(ASG, mask(IMPL, CONF, negate=True));  NASG ∪= scan(Δ, negate_literal)
About 38 operator records per round. `algebra_watched(inst, materialized=False)` stores WATCH as
W0(c, l) (initial watches, shared by all cubes) plus WMOV(q, c, l) (moves): T3 = update(combine(DF,
W0), combine(DF, WMOV)) and CANDL takes two antijoins.

(a) scan form, `algebra_scan` (one round; state ASG, CONF):
    NASG  = scan(ASG, negate_literal [counting])
    FC    = combine(CLAUSE_cnt(c,w,l), NASG(q,l), keep=(q,c,w))         # false_count for every clause with a false literal
    SAT   = combine(CLAUSE(c,w,l), ASG(q,l), keep=(q,c))
    UNIT  = mask(filter(FC, v == w-1), SAT, negate=True);  CONF = filter(FC, v == w)
    UCL   = combine(scan(UNIT, to_bool)(q,c), CLAUSE(c,w,l))
    IMPL  = reduce(mask(UCL, NASG, negate=True), keep=(q,l))
    CONF, ASG, Δ as in (b).

## 4. Hotspot event and work efficiency
Event: one clause visit of the golden; on the algebra side one combine candidate.
Measured (harness, small, seed 0, 2026-09-25), golden 2,689 clause visits, 5,206 literal visits:

| program | mul_ops | sr_ops | candidates | ratio_mul | ratio_sr | ratio_cand | rounds |
|---|---|---|---|---|---|---|---|
| (b) watched, materialised WATCH (primary) | 13,992 | 14,934 | 98,256 | 5.20 | 5.55 | 36.5 | 15 |
| (b) watched, factored W0 + WMOV | 13,992 | 14,934 | 101,169 | 5.20 | — | 37.6 | 15 |
| (a) scan form | 146,286 | 158,751 | 295,703 | 54.4 | 59.0 | 110.0 | 15 |

Seeds 1, 2 small: (b) ratio_mul 4.69, 4.87, ratio_cand 33.3, 35.0; (a) ratio_mul 43.9, 32.2.
Large (seed 0), golden 38,694 clause visits, 76,277 literal visits:

| program | mul_ops | sr_ops | candidates | ratio_mul | ratio_sr | ratio_cand | rounds |
|---|---|---|---|---|---|---|---|
| (b) watched, materialised (primary) | 184,543 | 195,806 | 1,402,945 | 4.77 | 5.06 | 36.3 | 9 |
| (b) watched, factored | 184,543 | 195,806 | 1,447,907 | 4.77 | — | 37.4 | 9 |
| (a) scan form | 1,233,609 | 1,289,701 | 2,536,954 | 31.9 | 33.3 | 65.6 | 9 |

Where (b)'s work goes (small seed 0 / large seed 0, combine candidates by position): touched
T3 3,277 / 43,640 = 1.22× / 1.13× the golden's clause visits; TCL 10,228 / 137,594 = 1.97× /
1.80× the golden's literal visits and 3.1× T3 (the join reads every literal of a touched clause,
the golden's replacement search stops at the first non-false one: 1.9-2.0 literal visits per
clause visit); SAT 485 / 3,305; opposite-implication check 2 / 4. So ratio_mul = 1.22 + 3.80 +
0.18 = 5.20 at small seed 0 (derived from the positions above). ratio_cand ≈ 36 comes from the unary
passes: masks alone are 52,503 / 771,527 candidates (53% / 55% of all), then scans 13,637 /
192,373, filters 8,729 / 122,528, reduces 6,418 / 90,574, updates 2,977 / 41,400. The touched
set is 13-22% larger than the golden's visits because the replacement rule differs (smallest
literal id vs first in clause order), because the algebra finishes a whole round after a cube's
conflict while the golden stops at once, and because an implication found early in a golden
round satisfies clauses before they are visited.

Where (a)'s work goes: FC 69,789 / 604,743 and SAT 73,740 / 614,431 candidates, i.e. per round
every (cube, false literal) pair is re-joined with its 6.4 clause occurrences on average (3,825
/ 600 and 25,635 / 4,000 clause literals per literal, derived) and every (cube, true literal)
pair likewise, for all cubes including the converged and the conflicted ones; the
filters then re-read every FC tuple twice (127,422 / 1,155,046). The plan's prediction of a ratio
far above 1 holds: 54.4 (small) and 31.9 (large), 10.5× and 6.7× the watched form. The ratio
falls with the larger instance only because it has fewer rounds (9 vs 15).

Timing (harness): small (b) 0.28 s, (a) 0.22 s, golden 0.01 s; large (b) 4.8 s (the interpreter
probes the 1,024,000-tuple WATCH relation every round; the factored variant with 16,000 + moved
tuples takes 1.2 s for the same candidates), (a) 1.9 s, golden 0.19 s; large harness run 8.1 s
including the extra runs.

## 5. scan() functions used
All O(1) per tuple, no look-ups outside the tuple:
* `negate_literal`: (q, l) -> (q, l ^ 1), boolean or counting value 1.
* `to_cube`: (q, …) -> (q,); `to_bool` / `to_qc`: (q, c, w) -> (q, c).
* `count_fallen_watches`: (q, l, c) -> ((q, c), 1) counting, merged by the scan's own add.
* `count_false_literals`: (q, c, w, l) -> ((q, c, w), 1) counting, merged: the false count.
* `literal_as_value`: (q, c, w, l) -> ((q, c, l), l) under (min, +): the literal id as the value.
* `value_to_key`: ((q, c), l) -> ((q, c, l), True): the chosen literal back into the key.
Filter predicates: `false_count == w-1`, `false_count == w`, `fallen_watches == 2`, all on the
tuple's own key and value (the clause width is part of the key).

## 6. Control residue
None beyond loop control: the per-round state (CONF, NASG, WATCH) is held in closure variables
between rounds, and the final per-cube result is read out of ASG and CONF (one pass over ASG).
The grouped select-1 for the watch replacement, the double-fall second replacement, the conflict
freeze and the polarity-conflict test are all operator chains; no plain-Python step had to be
counted separately.

## 7. Where the algebra is insufficient or awkward
* The clean form (a) is asymptotically the wrong algorithm, not a constant-factor loss: its work
  per round is proportional to the total number of assigned literals of all cubes (times clause
  occurrence), repeated for every round, while unit propagation's work is proportional to the
  literals that changed. Measured 54.4× / 31.9× the golden's clause visits (ratio_mul), 110× /
  66× on ratio_cand; both far outside the plan's 10× admission threshold. Batching the Q cubes
  makes it worse: converged and conflicted cubes are re-scanned until the last cube converges.
* The watched form (b) recovers the golden's per-round proportionality: the touched relation is
  1.13-1.22× the golden's clause visits and the delta sequence tracks the propagation depth
  (15 / 9 rounds). Its remaining factor of ≈ 5 on mul_ops has one cause, the join with clause
  reads all literals of a touched clause where the golden stops at the first non-false literal
  (1.9-2.0 look-ups per visit); a "first match" join would need an early-exit gather, which the
  algebra does not have.
* ratio_cand ≈ 36 is the cost of writing an inner loop of tests (satisfied? false? watched?
  unit? conflict? both watches fell?) as ≈ 38 relational passes per round; more than half of
  the candidates are mask passes over the same touched-literal tuples. The plan's admission
  threshold of 10× on ratio_cand is not met by (b) either. A hardware operator that gathers a
  touched clause's literals once and classifies them in one pass (a per-group state machine
  over the (q, c) groups) would collapse those passes to about two candidates per literal.
* `select_k(k=1)` could not be used for the watch replacement: it selects k tuples over the
  whole relation and has no group-by. The replacement was expressed as a reduce under the
  selective (min, +) semiring with the literal id as value, which is a grouped select-1; a
  grouped select operator would be the direct form. Two consecutive selections handle clauses
  whose two watches fell in the same round.
* There is no delete: moved watches stay in WATCH. This is harmless here (a false literal never
  becomes newly false again) but relies on the monotone assignment; a general watch scheme
  would need tuple deletion.
* Q cubes run in lock-step: the fixpoint ends when every cube has converged. In (b) a converged
  cube costs nothing (empty Δfalse), so lock-step only lengthens the round count.
* Interpreter note: candidates are identical for the materialised WATCH (Q · 2m = 1,024,000
  tuples at large) and the factored W0 + WMOV form, but the reference interpreter probes the
  larger operand of a combine, so wall time differs 4× at large; a hardware join needs the
  (q, l) index on the watch relation, or the factored representation.

## 8. Shape seen by the hardware
clause(c, w, l): 3,825 / 25,635 tuples, shared by all Q cubes (multi-query operand of every
combine in both forms). WATCH: 38,400 / 1,024,000 tuples materialised, or W0 2,400 / 16,000 plus
moves in the factored form. asg starts at Q · c = 544 / 8,960 tuples and grows by the implied
literals. Δ sizes (b), small seed 0: [79, 54, 28, 15, 15, 11, 11, 11, 13, 12, 3, 1, 2, 1, 0];
large seed 0: [1143, 387, 158, 56, 27, 17, 4, 1, 0]. Rounds 15 / 9. Per round in (b) the touched
relation is a few hundred to a few thousand (q, c) groups and the largest intermediate is TCL,
about 3.1 tuples per group; the dominant operators are mask by candidates and combine by
mul_ops. In (a) the two combines with clause dominate (FC and SAT, each ≈ 16× the golden's visits
at large) and their probe side is the shared clause relation.
