# <workload name>

**Category**: <logic | probabilistic | graph | search | planning | vsa | automata | parsing>
**Semiring(s)**: <names> · **Execution structure**: <dense-iterate | sparse-fixpoint | dag | frontier-fixpoint | select-loop>
**Module**: `softwareworkload/<name>/workload.py` · **Test**: `tests/test_workload_<name>.py`

## 1. Problem and reference algorithm
What is computed; which textbook / production algorithm `native()` implements; what it counts.

## 2. Mathematical decomposition
Equations in the weighted-relation form Y(o) = ⊕_u ⊗_r X_r(π_r(o,u)); the iteration / fixpoint / selection structure around them.

## 3. Algebra program
The operator chain, one line per operator, with input and output schemas.

## 4. Hotspot event and work efficiency
Definition of the counted event on both sides. Measured ratios from the harness (state the
scale, seed, and date of the run; mark them "measured"). Explain any ratio above 1.

## 5. scan() functions used
Each function, one line, and why it is O(1) per tuple. "none" if none.

## 6. Control residue
What stays in plain Python outside the operators (loop control, back-tracking, id assignment),
and its cost relative to the hotspot.

## 7. Where the algebra is insufficient or awkward
Honest list. If a construct could not be expressed, say so and say what a hardware operator
would need. If the clean mathematical form is asymptotically worse than the reference algorithm
(the plan's section 4.5 cases), say so with the measured ratio.

## 8. Shape seen by the hardware
Dense or sparse operands; sizes at each scale; number of rounds and delta sizes for fixpoints;
whether Q queries share an operand (multi-query); output growth; which operator dominates.
