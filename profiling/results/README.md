# profiling/results — CSV schema

One file per workload, `profiling/results/<workload>.csv`, first row = header. Every parser run appends
one row; rows are never edited or deleted (aimd rules section 21). A run under several tools produces one
row per tool (`metric_source`), all sharing `trace_dir`, which is the join key. A cell that no measurement
in the trace directory supports holds the placeholder `TBD[src]` (rules section 17); nothing is estimated
by the parsers. Values are as measured by the tool; derived or lower-bound quantities say so in their
`*_method` cell.

| column | content |
|---|---|
| workload | directory name under `profiling/workloads/` (datalog, sat, pc, hmm, graph, kg, planning, vsa, beam) |
| tier | plan section 8 tier of the workload (1 memory-resident static, 2 dynamic structure, 3 held-out candidate) |
| scale | `small` or `large` (every workload has at least these two) |
| instance | dataset / instance / model / configuration identifier, plus the algorithm or phase when one run has several |
| hardware | GPU name, driver, memory (`nvidia-smi`) and CPU model (`lscpu`), from `versions.txt` |
| sw_version | `nsys`, `ncu`, `perf`, `python`, `torch` versions and the workload software revision (`SW_VERSION_EXTRA`) |
| date | run date, YYYY-MM-DD |
| trace_dir | `profiling/traces/<workload>/<timestamp>-<label>/` the row was parsed from (not in git) |
| metric_source | parser that wrote the row: `parse_nsys`, `parse_ncu` or `parse_perf` |
| W_bytes | metric 1, working set in bytes; the driver script's tensor bytes, or a formula stated in W_method |
| W_method | how W_bytes was obtained (counter or formula; "lower bound" when structure overhead is excluded) |
| reuse_distance | metric 1 companion: memory-level breakdown of sampled loads from `perf mem` (`PERF_MEM=1`), else TBD |
| S_symbolic_fraction | metric 2, 0..1: symbolic / probabilistic time over end-to-end time (definition in S_method) |
| S_method | what was divided by what (NVTX range or kernel regex over wrapper wall time; perf sample shares for CPU programs) |
| R_control_residue | metric 3, 0..1: share of samples in branch-heavy control symbols (`--control-regex` of run.sh) |
| R_method | classification used; the regexes are the manual annotation and must be re-checked against perf_report.txt |
| Q_sharing_degree | metric 4: concurrent queries reading the same data (batch, beams x batch, threads, or 1) |
| access_pattern | metric 5: `sequential-ish` / `random` from ncu sectors per request (rule in parse_ncu.py), or CPU counter ratios |
| l1_hit_pct | metric 5: ncu `l1tex__t_sector_hit_rate.pct`, duration-weighted over profiled launches |
| l2_hit_pct | metric 5: ncu `lts__t_sector_hit_rate.pct`, duration-weighted |
| sectors_per_request | metric 5: ncu global-load sectors / requests (32 B sectors; 4 = fully coalesced) |
| semiring | metric 6: (add, mul) of the hot loop from code inspection (`--semiring` in run.sh, text in notes.md) |
| arith_intensity | metric 7: `smsp__inst_executed.sum` / (`dram__bytes_read.sum` + `dram__bytes_write.sum`), warp instructions per DRAM byte |
| inst_executed | metric 7 numerator, summed over profiled launches |
| dram_bytes | metric 7 denominator, summed over profiled launches |
| dram_bw_util_pct | metric 8: ncu `dram__throughput.avg.pct_of_peak_sustained_elapsed`, duration-weighted |
| sm_throughput_pct | metric 8 companion: ncu `sm__throughput.avg.pct_of_peak_sustained_elapsed`, duration-weighted |
| gpu_kernel_time_s | sum of GPU kernel time in the nsys summary |
| wall_time_s | wall time of the profiled run measured by the wrapper (includes profiler overhead) |
| work_efficiency | metric 9: always `see softwareworkload harness` (measured by `softwareworkload/_harness.py`, not here) |
| sw_counters | the software's own counters: `SYMFABRIC_METRIC` lines and `--counter-regex` matches from stdout.log, `name=value;...` |
| notes | kernel-time shares, top kernels or symbols, caveats written by the parser and by run.sh |
