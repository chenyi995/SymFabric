# profiling — GPU / CPU profiling kit for the SymFabric workloads

Purpose: measure, for each target workload at two problem scales, the nine metrics of the project plan
(section 12): working set W, symbolic fraction S, control residue R, sharing degree Q, access pattern,
semiring type, arithmetic intensity, DRAM-bandwidth utilisation, and (elsewhere) work efficiency. The rows
feed experiments E1 (working set), E2 (sharing) and E3 (bandwidth wall). Method follows REASON's
(`ref/notes/2026-HPCA-REASON.md`, "Simulation / implementation method"): Nsight Systems + Nsight Compute on the
GPU, `perf` (in place of VTune) on the CPU, cProfile for Python.

**Untested on this host.** This kit was written on a machine without a working NVIDIA driver and without
`ncu`; nothing has been executed end to end. Shell scripts passed `bash -n`, Python scripts passed `py_compile`,
and the parsers were exercised on hand-written fixture files only. `nsys` (2025.1.3) and `perf` (4.18) flags were
checked against `--help` here; `ncu` flags and metric names, the external software's command lines and Python
APIs come from documentation memory and are marked "verify" where they appear. Expect to adjust symbol regexes,
demo names and API keywords on the first run.

## Prerequisites (on the GPU server)

- CUDA 12.x driver and toolkit; `nvidia-smi` working.
- Nsight Systems `nsys` and Nsight Compute `ncu` on `PATH` (CUDA toolkit or standalone installers). `ncu`
  needs access to GPU performance counters (`NVreg_RestrictProfilingToAdminUsers=0` or root).
- Linux `perf` matching the kernel; `kernel.perf_event_paranoid` <= 1 for `perf stat/record` on own processes
  without the `:u` suffix (see `scripts/perf_wrap.sh`). `perf mem` needs Intel PEBS or AMD IBS.
- conda; Python 3.10 environment `symfabric-prof` created by `scripts/setup_env.sh` (PyTorch 2.5 as in
  REASON, transformers, torch-hd, pyjuice, cuGraph wheels).
- Build tools for Souffle (cmake, bison >= 3.6, flex, mcpp, libffi, sqlite3), Kissat, GraphBLAS/LAGraph, Fast Downward.
- External disk space for clones, builds, datasets and traces: `$SYMFABRIC_EXT` (default
  `/data2/$USER/symfabric_ext`), never inside the repository.

## Layout

| path | content |
|---|---|
| `scripts/setup_env.sh` | conda env, clones into `$SYMFABRIC_EXT`, builds, small datasets; every step idempotent |
| `scripts/nsys_wrap.sh`, `ncu_wrap.sh`, `perf_wrap.sh` | run a command under one tool; raw outputs to `traces/<workload>/<timestamp>-<label>/` |
| `scripts/_wrap_common.sh` | shared by the wrappers: trace directory, `versions.txt`, `cmd.txt`, wall time |
| `scripts/parse_nsys.py`, `parse_ncu.py`, `parse_perf.py` | raw outputs -> one row appended to `results/<workload>.csv` |
| `scripts/_common.py` | CSV schema, `TBD[src]` filling, `SYMFABRIC_METRIC` reader shared by the parsers |
| `workloads/<w>/run.sh` | the concrete command lines for both scales, wrapped and parsed; `notes.md` next to it |
| `workloads/<w>/*.py`, `*.dl` | generators and stand-in drivers (documented in each `notes.md`) |
| `results/` | metric tables; schema in `results/README.md` |
| `traces/` | raw profiler output; git-ignored (`traces/.gitignore`), only `.gitkeep` is tracked |

Workloads: `datalog` (Souffle), `sat` (Kissat), `pc` (pyjuice), `hmm` (Ctrl-G or PyTorch stand-in),
`graph` (cuGraph, SuiteSparse:GraphBLAS via LAGraph), `kg` (NBFNet or sparse (max, x) stand-in),
`planning` (Fast Downward h^add / h^max), `vsa` (torchhd), `beam` (Hugging Face beam search).
Tier column: sat, pc, hmm = 1; datalog, kg, planning, beam = 2, as in plan section 8. graph is not in the
plan's tier table and is filed under 2 next to KG multi-hop; vsa is filed under 3 because NVSA is the plan's
Tier 3 candidate (both are claude's filing, not the user's ruling).

## Run order

```bash
export SYMFABRIC_EXT=/data2/$USER/symfabric_ext          # external directory (default shown)
bash profiling/scripts/setup_env.sh                       # or: setup_env.sh env clone build data (one step at a time)
conda activate symfabric-prof
export HF_HOME=$SYMFABRIC_EXT/hf                          # keeps Hugging Face downloads external
bash profiling/workloads/sat/run.sh small                 # quickest CPU check of the perf path
bash profiling/workloads/hmm/run.sh small                 # quickest GPU check of the nsys + ncu path
for w in datalog sat pc hmm graph kg planning vsa beam; do bash profiling/workloads/$w/run.sh all; done
```
Each `run.sh [small|large|all]` runs its scales, and each scale runs: wrapper -> raw files in
`traces/<workload>/<timestamp>-<label>/` -> parser -> one row in `results/<workload>.csv`. Runs are never
overwritten; re-running appends. Tools that are missing produce a row full of `TBD[src]` plus a note.

## Recording hardware and software versions

Every wrapper writes `versions.txt` in the trace directory from: `nvidia-smi --query-gpu=name,driver_version,memory.total
--format=csv,noheader`, `lscpu` (model name), `nsys --version`, `ncu --version`, `perf --version`,
`python3 --version`, `python -c "import torch; print(torch.__version__, torch.version.cuda)"`, and the
workload's own revision passed through `SW_VERSION_EXTRA` by `run.sh` (git rev, `kissat --version`, package
version). The parsers copy these into the `hardware` and `sw_version` columns; `date` and `wall_s` come from
the same file.

## Metric-to-command table

| # | metric | command / source | column(s) |
|---|---|---|---|
| 1 | working set W | driver scripts print `SYMFABRIC_METRIC W_bytes` (tensor bytes); `sat/gen_cnf.py --stats` (4 B x literal occurrences); datalog: 8 B x tuples; planning: registered states x bytes per state; graph: CSR bytes. Reuse distance: `PERF_MEM=1 perf_wrap.sh` -> `perf mem record` + `perf mem report --stdio --sort mem` (memory-level breakdown). Pin is not scripted. | `W_bytes`, `W_method`, `reuse_distance` |
| 2 | symbolic fraction S | GPU: `nsys profile --trace=cuda,nvtx,osrt --stats=true -o profile` then `nsys stats --report cuda_gpu_kern_sum,cuda_gpu_mem_time_sum,nvtx_pushpop_sum,cuda_api_sum --format csv`; `parse_nsys.py --nvtx-range <hot range>` or `--symbolic-regex <kernels>` over the wrapper wall time. CPU programs: `perf report` sample share of hot + control symbols. Python-level: `python3 -m cProfile -o cprof.out script.py` and `python3 -c "import pstats; pstats.Stats('cprof.out').sort_stats('cumtime').print_stats(25)"` (manual), or `NSYS_EXTRA=--python-sampling=true`. | `S_symbolic_fraction`, `S_method`, `gpu_kernel_time_s`, `wall_time_s` |
| 3 | control residue R | `perf record -g -F 999` + `perf report --stdio --no-children --sort dso,sym --percent-limit 0.1`; `parse_perf.py --hot-regex ... --control-regex ...` (the regexes in each `run.sh` are the manual annotation; re-check against `perf_report.txt`) | `R_control_residue`, `R_method` |
| 4 | sharing degree Q | the software's own configuration / logs: driver scripts print `SYMFABRIC_METRIC Q` (batch, sources, beams x batch); Souffle `-j` threads; Kissat, Fast Downward, cuGraph single-source calls = 1 | `Q_sharing_degree` |
| 5 | access pattern | `ncu --metrics l1tex__t_sector_hit_rate.pct,lts__t_sector_hit_rate.pct,l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum,l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum` (last two: verify with `ncu --query-metrics`); CPU: `perf stat -e cycles,instructions,cache-references,cache-misses,LLC-loads,LLC-load-misses,L1-dcache-loads,L1-dcache-load-misses,branches,branch-misses` and `perf mem` | `access_pattern`, `l1_hit_pct`, `l2_hit_pct`, `sectors_per_request` |
| 6 | semiring type | code inspection, written in each `workloads/<w>/notes.md` and passed as `--semiring` | `semiring` |
| 7 | arithmetic intensity | `ncu --metrics smsp__inst_executed.sum,dram__bytes_read.sum,dram__bytes_write.sum`; `parse_ncu.py` = instructions / DRAM bytes. FLOP-level counts: add `smsp__sass_thread_inst_executed_op_fadd_pred_on.sum,..._fmul_pred_on.sum,..._ffma_pred_on.sum` to `NCU_METRICS` (names to verify) | `arith_intensity`, `inst_executed`, `dram_bytes` |
| 8 | DRAM bandwidth utilisation | `ncu --metrics dram__throughput.avg.pct_of_peak_sustained_elapsed,sm__throughput.avg.pct_of_peak_sustained_elapsed --csv` (duration-weighted by `gpu__time_duration.sum`); timeline: `NSYS_EXTRA=--gpu-metrics-devices=cuda-visible` (view in the nsys GUI) | `dram_bw_util_pct`, `sm_throughput_pct` |
| 9 | work efficiency | not measured here; `softwareworkload/_harness.py` | `work_efficiency` = `see softwareworkload harness` |

Full wrapper invocations: `scripts/ncu_wrap.sh` adds `--target-processes all --csv --print-units base
--log-file ncu.csv --launch-skip 20 --launch-count 200` and optional `--kernel-name regex:` /
`--nvtx --nvtx-include`; `scripts/perf_wrap.sh` runs `perf stat -x ';'`, then `perf record`, then
(optionally) `perf mem`, i.e. the command runs two or three times.

## How results are parsed

`parse_nsys.py` reads `stats_*.csv`; `parse_ncu.py` reads `ncu.csv` (one row per launch and metric, units
normalised); `parse_perf.py` reads `perf_stat.txt`, `perf_report.txt`, `perf_mem.txt`. All read `versions.txt`
and `stdout.log` (`SYMFABRIC_METRIC name value` lines and `--counter-regex` matches such as Kissat's
`propagations:` or Fast Downward's `Expanded N state(s)`). Each call appends exactly one row; a metric absent from
the raw files is written as `TBD[src]`, never estimated. `--help` on any parser lists the identification
arguments (`--workload --tier --scale --instance --semiring --Q --W-bytes`).

## Items to verify on the GPU host before trusting a row

`ncu` metric names (`ncu --query-metrics`), `ncu --print-units base` and `--nvtx-include "<range>/"` syntax;
pyjuice `HCLT` / `compile` / `missing_mask` keywords; Ctrl-G and NBFNet command lines (kept as unverified
options, off by default); LAGraph demo executable names and whether they read the `.mtx` from argv or stdin;
cuGraph wheel names; `kissat --time`; downward-benchmarks problem file names; every symbol regex in the
`run.sh` files (first guesses, to be re-classified from `perf_report.txt`); SAT Competition download URL pattern.
