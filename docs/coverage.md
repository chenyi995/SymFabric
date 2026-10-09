# Operator coverage and work efficiency (generated)

Source: `eval/results/20260925-203019-final6-large` (scale large, seed 0, 20260925-203019, git uncommitted). Regenerate with `eval/make_coverage.py`; do not edit by hand.

All ratios are measured by the harness: ratio_mul = semiring multiplications of the algebra program / hotspot events of the native algorithm; ratio_cand = every tuple touched by any operator / hotspot events. `ok` = algebra result equals the independent golden.

## 1. Algebra coverage

| workload | category | structure | semirings | ops used (count) | ok | native events | ratio_mul | ratio_cand | residue |
|---|---|---|---|---|---|---|---|---|---|
| astar_grid | search | select-loop | minplus | combine(10311), mask(6874), update(6874), select(3438), scan(6874) | yes | 10,257 | 1.87 | 201.31 | - |
| automata_rpq | automata | frontier-fixpoint | boolean | combine(88), reduce(1), mask(45), update(44) | yes | 3,545,411 | 2.00 | 4.08 | - |
| bayes_net_ve | probabilistic | dag | prob, logprob | combine(118), reduce(32), filter(46), scan(2) | yes | 102,982 | 1.00 | 1.12 | residue_minfill_pair_checks=36111 |
| beam_search | search | select-loop | viterbi | combine(20), select(20), scan(20) | yes | 19,488 | 1.00 | 2.03 | control_residue={'rank_sort_items': 640, 'backpointer_stores': 640, 'backpointer_reads': 640} |
| cky_parsing | parsing | dag | prob, viterbi | combine(178), update(88), scan(176) | yes | 1,237,468 | 5.98 | 6.82 | - |
| ctrlg_dfa_hmm | probabilistic | dense-iterate | prob, logprob, boolean | combine(296), reduce(2), mask(5), update(4), scan(2) | yes | 224,352 | 1.06 | 1.08 | - |
| datalog_nonlinear_tc | logic | sparse-fixpoint | boolean | combine(10), mask(5), update(10) | yes | 164,033 | 71.49 | 73.56 | - |
| datalog_samegen | logic | sparse-fixpoint | boolean | combine(12), mask(6), update(6) | yes | 383,247 | 1.00 | 2.57 | - |
| datalog_tc | logic | sparse-fixpoint | boolean | combine(12), mask(12), update(12) | yes | 23,992,885 | 1.00 | 1.88 | - |
| fuzzy_inference | logic | sparse-fixpoint | fuzzy | combine(49), update(48) | yes | 4,238,976 | 4.84 | 6.80 | - |
| graph_bfs_sssp | graph | frontier-fixpoint + select-loop | minplus, boolean | combine(19889), mask(39733), update(39761), select(19861), scan(11) | yes | 297,966 | 1.46 | 813.90 | - |
| hmm_viterbi | probabilistic | dense-iterate | logprob, viterbi_bp | combine(476), filter(119) | yes | 548,352 | 1.02 | 1.03 | - |
| kg_multihop | graph | frontier-fixpoint (multi-query) | viterbi | combine(4), mask(1), update(4), select(256) | yes | 394,115 | 1.00 | 3.95 | - |
| planning_hadd | planning | sparse-fixpoint | minplus, bottleneck | combine(48), reduce(24), mask(24), update(12), scan(24) | yes | 102,524 | 4.02 | 7.67 | control_residue=none beyond the fixpoint termination test (|Delta| = 0 per round) |
| prob_circuit | probabilistic | dag | prob, logprob, counting | combine(48) | yes | 811,008 | 1.10 | 1.10 | - |
| sat_bcp | logic | frontier-fixpoint | boolean, counting, minplus | combine(36), reduce(36), mask(90), update(54), filter(27), scan(100) | yes | 38,694 | 4.77 | 36.26 | - |
| triangle_count | graph | sparse join (one-shot, no fixpoint) | counting | combine(3), reduce(2), mask(1) | yes | 1,572,228 | 1.62 | 2.47 | - |
| vsa_cleanup | vsa | dag | real | combine(8), select(128), filter(128) | yes | 19,275,776 | 1.12 | 1.14 | control_residue=none (loop over the Q queries only) |

## 2. Operator usage matrix (semiring operations per operator, share of the workload's total)

| workload | combine | reduce | mask | update | select | filter | scan |
|---|---|---|---|---|---|---|---|
| astar_grid | 1% | - | 50% | 0% | 49% | - | 0% |
| automata_rpq | 60% | 0% | 40% | 0% | - | - | - |
| bayes_net_ve | 96% | 3% | - | - | - | 1% | 0% |
| beam_search | 49% | - | - | - | 49% | - | 2% |
| cky_parsing | 93% | - | - | 0% | - | - | 7% |
| ctrlg_dfa_hmm | 99% | 0% | 0% | 0% | - | - | 0% |
| datalog_nonlinear_tc | 99% | - | 1% | 0% | - | - | - |
| datalog_samegen | 56% | - | 44% | 0% | - | - | - |
| datalog_tc | 69% | - | 31% | 0% | - | - | - |
| fuzzy_inference | 81% | - | - | 19% | - | - | - |
| graph_bfs_sssp | 0% | - | 50% | 0% | 50% | - | 0% |
| hmm_viterbi | 99% | - | - | - | - | 1% | - |
| kg_multihop | 34% | - | 33% | 0% | 33% | - | - |
| planning_hadd | 55% | 7% | 24% | 2% | - | - | 12% |
| prob_circuit | 100% | - | - | - | - | - | - |
| sat_bcp | 14% | 1% | 60% | 0% | - | 10% | 15% |
| triangle_count | 68% | 2% | 30% | - | - | - | - |
| vsa_cleanup | 99% | - | - | - | 0% | 1% | - |

## 3. Where the time goes on each configuration

time = modelled cycles / clock; mem% = share of cycles in memory-bound operators; pe = useful semiring ops / (cycles x PEs); bw = average DRAM bandwidth utilisation; PIM x = speedup of the placement selector over chip-only (1.0 = nothing moved).

| workload | config | time (ms) | DRAM bytes | bw | pe | mem% | dominant modes | lpddr5 x | hbm3 x | hbm3_fast x | hbm3_16s x |
|---|---|---|---|---|---|---|---|---|---|---|---|
| astar_grid | edge28 | 0.075 | 8.21e+04 | 0.01 | 0.000 | 0.00 | merge_join 27%, systolic_pq 20%, stream 18% | 1.0 | 1.0 | 1.0 | 1.0 |
| astar_grid | edge28_allpe | 0.075 | 8.21e+04 | 0.01 | 0.000 | 0.00 | merge_join 27%, systolic_pq 20%, stream 18% | 1.0 | 1.0 | 1.0 | 1.0 |
| astar_grid | dc_hbm3 | 0.038 | 8.21e+04 | 0.00 | 0.000 | 0.00 | merge_join 27%, systolic_pq 20%, stream 18% | 1.0 | 1.0 | 1.0 | 1.0 |
| astar_grid | dc_hbm3_allpe | 0.038 | 8.21e+04 | 0.00 | 0.000 | 0.00 | merge_join 27%, systolic_pq 20%, stream 18% | 1.0 | 1.0 | 1.0 | 1.0 |
| astar_grid | dc_hbm4 | 0.038 | 8.21e+04 | 0.00 | 0.000 | 0.00 | merge_join 27%, systolic_pq 20%, stream 18% | 1.0 | 1.0 | 1.0 | 1.0 |
| astar_grid | tpu_like | 105199.372 | 1.39e+07 | 0.00 | 0.008 | 0.00 | dense_gemm 100%, sort_select 0%, dense_elementwise 0% | 129069.977 | 129559.051 | 129559.051 | 127717.855 |
| automata_rpq | edge28 | 7.018 | 6.37e+08 | 0.87 | 0.002 | 0.81 | random_rmw 29%, random_probe 28%, merge_join 17% | 10.656 | 47.214 | 49.94 | 66.11 |
| automata_rpq | edge28_allpe | 6.131 | 6.36e+08 | 1.00 | 0.002 | 1.00 | random_rmw 34%, random_probe 30%, partitioned_probe 15% | 10.874 | 46.963 | 48.373 | 68.678 |
| automata_rpq | dc_hbm3 | 0.206 | 7.5e+05 | 0.00 | 0.002 | 0.00 | merge_join 72%, onchip_probe 19%, onchip_update 7% | 1.0 | 1.454 | 1.524 | 2.103 |
| automata_rpq | dc_hbm3_allpe | 0.002 | 7.5e+05 | 0.27 | 0.184 | 0.27 | merge_join 78%, onchip_probe 14%, onchip_update 6% | 1.0 | 1.0 | 1.0 | 1.0 |
| automata_rpq | dc_hbm4 | 0.206 | 7.5e+05 | 0.00 | 0.002 | 0.00 | merge_join 72%, onchip_probe 19%, onchip_update 7% | 1.0 | 1.454 | 1.524 | 2.103 |
| automata_rpq | tpu_like | 27933.862 | 1.84e+10 | 0.00 | 0.997 | 0.00 | dense_gemm 100%, dense_elementwise 0%, partitioned_probe 0% | 48973.01 | 179358.676 | 193055.672 | 245244.047 |
| bayes_net_ve | edge28 | 0.012 | 2.03e+04 | 0.02 | 0.017 | 0.00 | merge_join 74%, dense_gemm 19%, segmented_reduce 6% | 1.0 | 1.0 | 1.038 | 1.0 |
| bayes_net_ve | edge28_allpe | 0.003 | 2.03e+04 | 0.07 | 0.071 | 0.03 | dense_gemm 78%, merge_join 14%, stream 4% | 1.0 | 1.0 | 1.0 | 1.0 |
| bayes_net_ve | dc_hbm3 | 0.005 | 2.03e+04 | 0.00 | 0.001 | 0.00 | dense_gemm 75%, merge_join 22%, segmented_reduce 2% | 1.0 | 1.0 | 1.0 | 1.0 |
| bayes_net_ve | dc_hbm3_allpe | 0.004 | 2.03e+04 | 0.00 | 0.002 | 0.00 | dense_gemm 95%, merge_join 2%, stream 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| bayes_net_ve | dc_hbm4 | 0.005 | 2.03e+04 | 0.00 | 0.001 | 0.00 | dense_gemm 75%, merge_join 22%, segmented_reduce 2% | 1.0 | 1.0 | 1.0 | 1.0 |
| bayes_net_ve | tpu_like | 0.071 | 9.28e+03 | 0.00 | 0.002 | 0.00 | dense_gemm 100%, segmented_reduce 0%, stream 0% | 1.481 | 1.512 | 2.03 | 1.43 |
| beam_search | edge28 | 0.122 | 4.48e+05 | 0.04 | 0.000 | 0.04 | merge_join 67%, systolic_pq 33%, stream 0% | 1.406 | 1.916 | 2.623 | 1.863 |
| beam_search | edge28_allpe | 0.047 | 4.48e+05 | 0.09 | 0.001 | 0.09 | systolic_pq 86%, merge_join 14%, stream 0% | 1.092 | 1.094 | 1.094 | 1.096 |
| beam_search | dc_hbm3 | 0.030 | 4.48e+05 | 0.01 | 0.000 | 0.00 | systolic_pq 66%, merge_join 34%, stream 0% | 1.005 | 1.008 | 1.152 | 1.01 |
| beam_search | dc_hbm3_allpe | 0.021 | 4.48e+05 | 0.02 | 0.000 | 0.02 | systolic_pq 98%, merge_join 2%, stream 0% | 1.002 | 1.006 | 1.006 | 1.008 |
| beam_search | dc_hbm4 | 0.030 | 4.48e+05 | 0.01 | 0.000 | 0.00 | systolic_pq 66%, merge_join 34%, stream 0% | 1.005 | 1.008 | 1.152 | 1.01 |
| beam_search | tpu_like | 0.665 | 0 | 0.00 | 0.235 | 0.00 | dense_gemm 100%, sort_select 0%, stream 0% | 13.876 | 26.755 | 86.24 | 24.929 |
| cky_parsing | edge28 | 0.666 | 2.68e+03 | 0.00 | 0.022 | 0.00 | merge_join 90%, stream 9%, onchip_update 0% | 1.0 | 1.0 | 1.003 | 1.0 |
| cky_parsing | edge28_allpe | 0.021 | 2.68e+03 | 0.00 | 0.679 | 0.00 | merge_join 89%, stream 10%, onchip_update 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| cky_parsing | dc_hbm3 | 0.083 | 2.68e+03 | 0.00 | 0.005 | 0.00 | merge_join 90%, stream 10%, onchip_update 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| cky_parsing | dc_hbm3_allpe | 0.001 | 2.68e+03 | 0.00 | 0.461 | 0.00 | merge_join 70%, stream 21%, onchip_update 9% | 1.0 | 1.0 | 1.0 | 1.0 |
| cky_parsing | dc_hbm4 | 0.083 | 2.68e+03 | 0.00 | 0.005 | 0.00 | merge_join 90%, stream 10%, onchip_update 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| cky_parsing | tpu_like | 0.331 | 4.11e+06 | 0.01 | 0.031 | 0.01 | dense_gemm 97%, stream 2%, random_rmw 1% | 1.007 | 1.007 | 1.043 | 1.001 |
| ctrlg_dfa_hmm | edge28 | 0.032 | 2.03e+05 | 0.06 | 0.014 | 0.00 | merge_join 99%, segmented_reduce 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| ctrlg_dfa_hmm | edge28_allpe | 0.003 | 2.03e+05 | 0.63 | 0.149 | 0.60 | merge_join 98%, segmented_reduce 1%, onchip_probe 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| ctrlg_dfa_hmm | dc_hbm3 | 0.004 | 2.03e+05 | 0.04 | 0.003 | 0.00 | merge_join 99%, segmented_reduce 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| ctrlg_dfa_hmm | dc_hbm3_allpe | 0.000 | 2.03e+05 | 0.37 | 0.032 | 0.33 | merge_join 97%, onchip_probe 1%, segmented_reduce 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| ctrlg_dfa_hmm | dc_hbm4 | 0.004 | 2.03e+05 | 0.02 | 0.003 | 0.00 | merge_join 99%, segmented_reduce 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| ctrlg_dfa_hmm | tpu_like | 0.125 | 1.46e+04 | 0.00 | 0.007 | 0.00 | dense_gemm 100%, dense_elementwise 0%, segmented_reduce 0% | 1.005 | 1.005 | 1.414 | 1.005 |
| datalog_nonlinear_tc | edge28 | 0.377 | 2.43e+05 | 0.01 | 0.061 | 0.00 | merge_join 86%, dense_gemm 9%, onchip_update 3% | 1.003 | 1.023 | 1.045 | 1.023 |
| datalog_nonlinear_tc | edge28_allpe | 0.044 | 2.43e+05 | 0.05 | 0.517 | 0.04 | dense_gemm 73%, merge_join 26%, onchip_update 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| datalog_nonlinear_tc | dc_hbm3 | 0.044 | 2.43e+05 | 0.00 | 0.016 | 0.00 | merge_join 91%, onchip_update 3%, dense_gemm 3% | 1.0 | 1.0 | 1.0 | 1.0 |
| datalog_nonlinear_tc | dc_hbm3_allpe | 0.002 | 2.43e+05 | 0.12 | 0.412 | 0.08 | dense_gemm 72%, merge_join 26%, onchip_update 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| datalog_nonlinear_tc | dc_hbm4 | 0.044 | 2.43e+05 | 0.00 | 0.016 | 0.00 | merge_join 91%, onchip_update 3%, dense_gemm 3% | 1.0 | 1.0 | 1.0 | 1.0 |
| datalog_nonlinear_tc | tpu_like | 0.019 | 3.14e+06 | 0.14 | 0.505 | 0.00 | dense_gemm 73%, stream_merge 15%, partitioned_probe 12% | 1.067 | 1.067 | 1.067 | 1.067 |
| datalog_samegen | edge28 | 0.101 | 1.76e+06 | 0.17 | 0.007 | 0.11 | merge_join 51%, onchip_probe 30%, onchip_update 19% | 1.762 | 2.803 | 2.803 | 2.974 |
| datalog_samegen | edge28_allpe | 0.019 | 1.76e+06 | 0.90 | 0.040 | 0.87 | onchip_probe 85%, merge_join 12%, onchip_update 3% | 2.302 | 2.959 | 2.959 | 3.92 |
| datalog_samegen | dc_hbm3 | 0.011 | 4.32e+04 | 0.00 | 0.002 | 0.00 | merge_join 58%, onchip_probe 21%, onchip_update 21% | 1.0 | 1.195 | 1.195 | 1.484 |
| datalog_samegen | dc_hbm3_allpe | 0.000 | 4.32e+04 | 0.27 | 0.175 | 0.27 | merge_join 69%, onchip_probe 16%, onchip_update 16% | 1.0 | 1.0 | 1.0 | 1.0 |
| datalog_samegen | dc_hbm4 | 0.011 | 4.32e+04 | 0.00 | 0.002 | 0.00 | merge_join 58%, onchip_probe 21%, onchip_update 21% | 1.0 | 1.195 | 1.195 | 1.484 |
| datalog_samegen | tpu_like | 96.013 | 8.85e+06 | 0.00 | 0.954 | 0.00 | dense_gemm 100%, partitioned_probe 0%, stream_merge 0% | 15.701 | 31.207 | 122.747 | 2.697 |
| datalog_tc | edge28 | 9.513 | 8.71e+08 | 0.88 | 0.005 | 0.80 | partitioned_probe 35%, merge_join 20%, random_rmw 19% | 7.627 | 53.291 | 53.574 | 435.618 |
| datalog_tc | edge28_allpe | 8.373 | 8.71e+08 | 1.00 | 0.006 | 1.00 | partitioned_probe 39%, random_rmw 21%, stream_merge 17% | 6.807 | 47.686 | 47.938 | 407.532 |
| datalog_tc | dc_hbm3 | 0.413 | 7.41e+07 | 0.15 | 0.004 | 0.05 | merge_join 57%, onchip_probe 32%, onchip_update 11% | 1.0 | 2.417 | 2.431 | 23.226 |
| datalog_tc | dc_hbm3_allpe | 0.064 | 7.41e+07 | 0.97 | 0.023 | 0.96 | onchip_probe 96%, merge_join 3%, onchip_update 1% | 1.0 | 1.191 | 1.191 | 6.365 |
| datalog_tc | dc_hbm4 | 0.402 | 7.41e+07 | 0.07 | 0.004 | 0.02 | merge_join 59%, onchip_probe 30%, onchip_update 12% | 1.0 | 2.352 | 2.365 | 22.596 |
| datalog_tc | tpu_like | 12.514 | 5.42e+08 | 0.04 | 0.914 | 0.02 | dense_gemm 96%, partitioned_probe 2%, random_probe 1% | 10.034 | 70.104 | 70.477 | 573.055 |
| fuzzy_inference | edge28 | 6.631 | 6e+08 | 0.87 | 0.006 | 0.77 | stream_merge 53%, merge_join 23%, random_rmw 23% | 1.74 | 3.512 | 3.6 | 3.528 |
| fuzzy_inference | edge28_allpe | 5.768 | 6e+08 | 1.00 | 0.007 | 1.00 | stream_merge 61%, random_rmw 26%, merge_join 12% | 1.957 | 5.624 | 5.756 | 5.672 |
| fuzzy_inference | dc_hbm3 | 0.258 | 7.08e+04 | 0.00 | 0.005 | 0.00 | merge_join 75%, onchip_update 25% | 1.0 | 1.0 | 1.0 | 1.0 |
| fuzzy_inference | dc_hbm3_allpe | 0.002 | 7.08e+04 | 0.03 | 0.592 | 0.03 | merge_join 75%, onchip_update 25% | 1.0 | 1.0 | 1.0 | 1.0 |
| fuzzy_inference | dc_hbm4 | 0.258 | 7.08e+04 | 0.00 | 0.005 | 0.00 | merge_join 75%, onchip_update 25% | 1.0 | 1.0 | 1.0 | 1.0 |
| fuzzy_inference | tpu_like | 2.146 | 2.87e+08 | 0.11 | 0.701 | 0.11 | dense_gemm 89%, stream_merge 9%, random_rmw 2% | 1.237 | 1.281 | 1.382 | 1.209 |
| graph_bfs_sssp | edge28 | 8.028 | 1.99e+06 | 0.00 | 0.000 | 0.00 | systolic_pq 95%, merge_join 3%, onchip_update 1% | 1.011 | 1.012 | 1.013 | 1.011 |
| graph_bfs_sssp | edge28_allpe | 0.556 | 1.99e+06 | 0.03 | 0.002 | 0.02 | systolic_pq 63%, onchip_update 13%, merge_join 10% | 1.018 | 1.018 | 1.018 | 1.018 |
| graph_bfs_sssp | dc_hbm3 | 0.169 | 1.99e+06 | 0.01 | 0.000 | 0.00 | systolic_pq 28%, merge_join 27%, onchip_update 22% | 1.004 | 1.006 | 1.011 | 1.007 |
| graph_bfs_sssp | dc_hbm3_allpe | 0.143 | 1.99e+06 | 0.01 | 0.000 | 0.01 | systolic_pq 33%, onchip_update 25%, merge_join 15% | 1.002 | 1.003 | 1.003 | 1.003 |
| graph_bfs_sssp | dc_hbm4 | 0.169 | 1.99e+06 | 0.00 | 0.000 | 0.00 | systolic_pq 28%, merge_join 27%, onchip_update 22% | 1.004 | 1.006 | 1.011 | 1.007 |
| graph_bfs_sssp | tpu_like | 62772.164 | 1.47e+09 | 0.00 | 0.008 | 0.00 | dense_gemm 100%, sort_select 0%, partitioned_probe 0% | 1201.142 | 1297.504 | 1557.311 | 992.571 |
| hmm_viterbi | edge28 | 0.092 | 1.97e+05 | 0.02 | 0.012 | 0.01 | dense_gemm 99%, dense_elementwise 1%, stream 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| hmm_viterbi | edge28_allpe | 0.092 | 1.97e+05 | 0.02 | 0.012 | 0.01 | dense_gemm 99%, dense_elementwise 1%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| hmm_viterbi | dc_hbm3 | 0.092 | 1.97e+05 | 0.00 | 0.000 | 0.00 | dense_gemm 100%, dense_elementwise 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| hmm_viterbi | dc_hbm3_allpe | 0.092 | 1.97e+05 | 0.00 | 0.000 | 0.00 | dense_gemm 100%, dense_elementwise 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| hmm_viterbi | dc_hbm4 | 0.092 | 1.97e+05 | 0.00 | 0.000 | 0.00 | dense_gemm 100%, dense_elementwise 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| hmm_viterbi | tpu_like | 0.122 | 432 | 0.00 | 0.000 | 0.00 | dense_gemm 100%, dense_elementwise 0%, stream 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| kg_multihop | edge28 | 1.018 | 2.53e+07 | 0.24 | 0.001 | 0.21 | systolic_pq 79%, merge_join 13%, onchip_probe 6% | 1.0 | 1.079 | 1.099 | 1.076 |
| kg_multihop | edge28_allpe | 0.993 | 2.53e+07 | 0.24 | 0.001 | 0.21 | systolic_pq 79%, merge_join 13%, onchip_probe 6% | 1.0 | 1.08 | 1.1 | 1.077 |
| kg_multihop | dc_hbm3 | 0.409 | 5.8e+06 | 0.01 | 0.000 | 0.00 | systolic_pq 95%, merge_join 3%, onchip_update 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| kg_multihop | dc_hbm3_allpe | 0.392 | 5.8e+06 | 0.01 | 0.000 | 0.01 | systolic_pq 99%, merge_join 1%, onchip_update 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| kg_multihop | dc_hbm4 | 0.409 | 5.8e+06 | 0.01 | 0.000 | 0.00 | systolic_pq 95%, merge_join 3%, onchip_update 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| kg_multihop | tpu_like | 610976.648 | 8e+11 | 0.00 | 0.999 | 0.00 | dense_gemm 100%, sort_select 0%, stream_merge 0% | 2377.194 | 4859.529 | 19398.632 | 316.775 |
| planning_hadd | edge28 | 0.128 | 8.42e+05 | 0.06 | 0.006 | 0.06 | merge_join 79%, onchip_probe 9%, segmented_reduce 6% | 1.018 | 1.027 | 1.056 | 1.023 |
| planning_hadd | edge28_allpe | 0.012 | 8.42e+05 | 0.68 | 0.067 | 0.68 | merge_join 92%, onchip_probe 3%, segmented_reduce 2% | 1.0 | 1.0 | 1.04 | 1.0 |
| planning_hadd | dc_hbm3 | 0.016 | 8.42e+05 | 0.04 | 0.002 | 0.00 | merge_join 79%, onchip_probe 9%, segmented_reduce 6% | 1.0 | 1.0 | 1.0 | 1.0 |
| planning_hadd | dc_hbm3_allpe | 0.001 | 8.42e+05 | 0.78 | 0.028 | 0.77 | merge_join 90%, onchip_probe 3%, segmented_reduce 3% | 1.0 | 1.0 | 1.0 | 1.0 |
| planning_hadd | dc_hbm4 | 0.016 | 8.42e+05 | 0.02 | 0.002 | 0.00 | merge_join 79%, onchip_probe 9%, segmented_reduce 6% | 1.0 | 1.0 | 1.0 | 1.0 |
| planning_hadd | tpu_like | 16.238 | 5.76e+09 | 0.30 | 0.011 | 0.30 | dense_gemm 70%, dense_elementwise 30%, partitioned_probe 0% | 1.953 | 2.297 | 3.79 | 1.456 |
| prob_circuit | edge28 | 0.068 | 3.56e+05 | 0.05 | 0.025 | 0.00 | merge_join 97%, dense_gemm 3%, dense_elementwise 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| prob_circuit | edge28_allpe | 0.006 | 3.56e+05 | 0.57 | 0.291 | 0.38 | merge_join 65%, dense_gemm 32%, dense_elementwise 3% | 1.0 | 1.0 | 1.0 | 1.0 |
| prob_circuit | dc_hbm3 | 0.009 | 3.56e+05 | 0.03 | 0.006 | 0.00 | merge_join 88%, dense_gemm 12%, dense_elementwise 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| prob_circuit | dc_hbm3_allpe | 0.001 | 3.56e+05 | 0.20 | 0.037 | 0.19 | dense_gemm 79%, merge_join 21%, dense_elementwise 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| prob_circuit | dc_hbm4 | 0.009 | 3.56e+05 | 0.01 | 0.006 | 0.00 | merge_join 88%, dense_gemm 12%, dense_elementwise 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| prob_circuit | tpu_like | 0.031 | 0 | 0.00 | 0.135 | 0.00 | dense_gemm 100%, dense_elementwise 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| sat_bcp | edge28 | 1.538 | 1.25e+08 | 0.78 | 0.000 | 0.54 | merge_join 42%, random_probe 27%, random_rmw 27% | 1.781 | 3.728 | 4.787 | 4.411 |
| sat_bcp | edge28_allpe | 1.206 | 1.25e+08 | 1.00 | 0.000 | 1.00 | random_probe 35%, random_rmw 34%, merge_join 31% | 1.831 | 4.575 | 6.023 | 5.577 |
| sat_bcp | dc_hbm3 | 0.089 | 4.23e+06 | 0.04 | 0.000 | 0.00 | merge_join 89%, onchip_probe 7%, stream 3% | 1.152 | 1.273 | 1.61 | 1.667 |
| sat_bcp | dc_hbm3_allpe | 0.004 | 4.23e+06 | 0.80 | 0.003 | 0.80 | merge_join 93%, stream 3%, onchip_probe 3% | 1.0 | 1.0 | 1.0 | 1.0 |
| sat_bcp | dc_hbm4 | 0.089 | 4.23e+06 | 0.02 | 0.000 | 0.00 | merge_join 89%, onchip_probe 7%, stream 3% | 1.152 | 1.273 | 1.61 | 1.667 |
| sat_bcp | tpu_like | 971.542 | 1.29e+11 | 0.11 | 0.015 | 0.11 | dense_gemm 89%, dense_elementwise 11%, partitioned_probe 0% | 42.229 | 56.072 | 174.57 | 13.111 |
| triangle_count | edge28 | 1.003 | 1.04e+08 | 1.00 | 0.005 | 1.00 | partitioned_probe 54%, merge_join 46%, segmented_reduce 0% | 1.553 | 2.057 | 2.058 | 2.059 |
| triangle_count | edge28_allpe | 0.998 | 1.04e+08 | 1.00 | 0.005 | 1.00 | partitioned_probe 54%, merge_join 46%, segmented_reduce 0% | 1.552 | 2.058 | 2.058 | 2.059 |
| triangle_count | dc_hbm3 | 0.042 | 1.44e+06 | 0.03 | 0.004 | 0.00 | merge_join 75%, onchip_probe 23%, segmented_reduce 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| triangle_count | dc_hbm3_allpe | 0.001 | 1.44e+06 | 0.83 | 0.107 | 0.83 | merge_join 94%, onchip_probe 5%, segmented_reduce 0% | 1.0 | 1.0 | 1.0 | 1.0 |
| triangle_count | dc_hbm4 | 0.042 | 1.44e+06 | 0.01 | 0.004 | 0.00 | merge_join 75%, onchip_probe 23%, segmented_reduce 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| triangle_count | tpu_like | 53829.026 | 6.4e+13 | 0.99 | 0.018 | 0.99 | dense_elementwise 99%, dense_gemm 1%, partitioned_probe 0% | 24.502 | 40.019 | 125.929 | 3.845 |
| vsa_cleanup | edge28 | 1.742 | 6.59e+07 | 0.36 | 0.024 | 0.23 | merge_join 99%, stream 1%, systolic_pq 0% | 1.005 | 1.126 | 1.139 | 1.126 |
| vsa_cleanup | edge28_allpe | 0.638 | 6.59e+07 | 0.99 | 0.066 | 0.99 | merge_join 99%, systolic_pq 1%, stream 0% | 1.014 | 1.44 | 1.501 | 1.441 |
| vsa_cleanup | dc_hbm3 | 0.212 | 7.69e+06 | 0.03 | 0.006 | 0.00 | merge_join 98%, systolic_pq 1%, stream 1% | 1.0 | 1.0 | 1.002 | 1.0 |
| vsa_cleanup | dc_hbm3_allpe | 0.009 | 7.69e+06 | 0.72 | 0.148 | 0.72 | merge_join 74%, systolic_pq 24%, stream 1% | 1.0 | 1.0 | 1.0 | 1.0 |
| vsa_cleanup | dc_hbm4 | 0.212 | 7.69e+06 | 0.01 | 0.006 | 0.00 | merge_join 98%, systolic_pq 1%, stream 1% | 1.0 | 1.0 | 1.002 | 1.0 |
| vsa_cleanup | tpu_like | 1.871 | 6.59e+08 | 0.29 | 0.013 | 0.29 | dense_gemm 71%, dense_elementwise 29%, stream 0% | 1.244 | 1.364 | 1.382 | 1.365 |
