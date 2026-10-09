> General working rules: `/data2/chenyi9/SymFabric/aimd/AI-RULES.en.md`
> (Chinese version: `aimd/AI-RULES.zh.md`). Read them at the start of the session.
> This file contains project-specific conventions only; where the two conflict, this
> file takes precedence.

# SymFabric project conventions

## Session start
- Run `whoami` for the account name (rules section 1) and address the user by it.
- Reply language follows the user's first message of the session (rules section 24).
  Repository content (code, comments, docs, commit messages) is English (section 25).
- Agent identifier for decision markers (section 23): `claude`.
- Placeholder token for a number without evidence (section 17): `TBD[src]`.

## Layout
| Path | Content |
|---|---|
| `plan/` | research plans (user-owned drafts) |
| `ref/` | reference papers (PDFs ignored by git) and `ref/notes/` one-page notes |
| `docs/` | algebra spec, decompositions, architecture notes, `docs/session/` records |
| `algebra/` | Python package: semirings, weighted relations, operators, reference interpreter |
| `softwareworkload/` | symbolic workloads written against `algebra/`, each with an independent golden |
| `systolicarch/` | reconfigurable systolic array model and operator mapping |
| `pim/` | memory-side (PIM / HBM-to-HBM) model; only used when a bottleneck is shown |
| `profiling/` | GPU / CPU profiling scripts with complete run instructions (no GPU on this host) |
| `eval/` | experiment drivers; results in `eval/results/<YYYYMMDD-HHMMSS>-<name>/`, never overwritten |
| `tests/` | correctness tests (algebra vs golden) |
| `local/` | Chinese drafts and summaries for the user, ignored by git |

## Archiving (rules section 22)
Superseded code goes to `<same dir>/archived/<MMDD>/<file>`; nothing is deleted or overwritten.

## Tests and regression (rules sections 19–20)
- Full test command: `python3 -m pytest -q tests/` (from repository root; about 5 minutes,
  the large-scale workload tests dominate).
- Change in `algebra/` -> run all of `tests/`.
- Change in `softwareworkload/<w>/` -> run `tests/test_workload_<w>.py`.
- Change in `systolicarch/` -> run `tests/test_arch_model.py`; in `pim/` -> `tests/test_pim_model.py`.
- Change in `eval/` -> re-run `python3 eval/run_arch.py --scale small` and compare `summary.csv`
  with the previous results directory.
- Goldens are independent re-implementations (plain Python, no `algebra/` imports).

## Evaluation drivers
- `eval/run_arch.py --scale {small,large} --pim <pim configs>`: runs every workload, maps the
  trace on every architecture config, writes `eval/results/<ts>-<name>-<scale>/` with
  `summary.csv`, `workloads.json`, `per_op/`, `traces/` (re-map without re-running:
  `--from-traces <dir>`).
- `eval/make_coverage.py <dir>` regenerates `docs/coverage.md`; `eval/analyze_bottleneck.py <dir>`
  writes `<dir>/bottleneck.md`; `eval/plot_results.py <dir>` writes `<dir>/plots/`;
  `eval/run_sensitivity.py` sweeps the TBD[src] parameters.

## Numbers (rules sections 12, 13, 17, 18)
Every hardware constant in `systolicarch/` and `pim/` carries a `# source:` comment naming
the paper and section/table, or the profile directory. Every number reported to the user is
labelled measured / derived / speculated.

## Environment facts (checked 2026-09-25)
- Host: 128 CPU cores, 754 GB RAM, no working NVIDIA driver; `/data2` had 22 GB free.
- Python 3.10.13 with numpy 2.2.6, pandas 2.3.3, matplotlib 3.10.8. No scipy / networkx.
- `pdftotext` available for reading `ref/*.pdf`.
