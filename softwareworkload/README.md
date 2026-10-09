# Software workloads

Each subdirectory is one symbolic workload written twice: once as an independent reference
(`native`) and once as a composition of the operators in `algebra/` (`algebra`). The harness in
`_harness.py` runs both, checks that the results agree, and reports the work-efficiency ratio
(semiring operations performed by the algebra program divided by the hotspot events of the
reference algorithm). The module interface is documented in `_harness.py`.

Rules for a workload module:

1. `native()` never imports `algebra.ops`; it is the golden (aimd rules section 19).
2. `algebra()` uses only the operators exported by `algebra/`; if a workload cannot be expressed,
   the module must say so in its docstring and in `docs/decompositions/<name>.md` instead of
   hiding a solver inside `scan()`'s function argument.
3. Every `scan()` function is O(1) per tuple and is listed in the decomposition document.
4. `HOTSPOT_EVENT` names the same kind of event on both sides (e.g. "edge relaxations"), never a
   generic FLOP count.
5. "small" runs in about a second; "large" in well under a minute, in pure Python.

Run everything: `python3 -m softwareworkload._harness --scale small` from the repository root.
Tests: `python3 -m pytest -q tests/test_workload_<name>.py`.
