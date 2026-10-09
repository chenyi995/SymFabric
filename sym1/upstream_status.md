# sym1 source acquisition inventory

Verified at 2026-09-26T23:35:38.879874+00:00.

Measured checkout count: 33 primary repositories and 18 dependency checkouts.
Every listed checkout passed Git object connectivity checks; all registered submodules are initialized at their recorded commits.
Clone completeness is separate from build, runtime dependency, dataset, and benchmark reproduction status.

Primary sources are shallow clones. Exact full commits and recursive submodule revisions are in `upstream_lock.json`.
Downloaded code stays under the ignored `upstream/` directories.

| Repository | Local directory | Commit | Purpose |
|---|---|---|---|
| [gapbs](https://github.com/sbeamer/gapbs) | `softwareworkload/upstream/gapbs` | `2972aeb27031` | CPU graph BFS, SSSP and triangle baseline |
| [souffle](https://github.com/souffle-lang/souffle) | `softwareworkload/upstream/souffle` | `a1303be3c016` | CPU Datalog and semi-naive fixpoint |
| [gdlog](https://github.com/harp-lab/gdlog) | `softwareworkload/upstream/gdlog` | `65a6ee960ced` | GPU Datalog runtime and benchmark programs |
| [vflog](https://github.com/harp-lab/vflog) | `softwareworkload/upstream/vflog` | `fe1dcdc84b83` | Columnar GPU Datalog baseline |
| [srdatalog-python](https://github.com/harp-lab/srdatalog-python) | `softwareworkload/upstream/srdatalog-python` | `114331cec452` | Author-group SRDatalog frontend/runtime; paper artifact revision requires alignment |
| [Lobster](https://github.com/P-bibs/Lobster) | `softwareworkload/upstream/Lobster` | `76a151f01a81` | GPU provenance-semiring Datalog |
| [scallop](https://github.com/scallop-lang/scallop) | `softwareworkload/upstream/scallop` | `668bfb6d45ce` | Neurosymbolic Datalog workloads and reference implementation |
| [pyjuice](https://github.com/Tractables/pyjuice) | `softwareworkload/upstream/pyjuice` | `00e8e770ef79` | Probabilistic circuit GPU baseline |
| [pyjuice-benchmarks](https://github.com/liuanji/pyjuice-benchmarks) | `softwareworkload/upstream/pyjuice-benchmarks` | `56c5db0268d5` | Probabilistic circuit benchmark configurations |
| [klay](https://github.com/ML-KULeuven/klay) | `softwareworkload/upstream/klay` | `d5c2907669dc` | Arithmetic circuit layer baseline |
| [pytorch-struct](https://github.com/harvardnlp/pytorch-struct) | `softwareworkload/upstream/pytorch-struct` | `7146de5659ff` | Semiring chain and parsing dynamic programs |
| [Ctrl-G](https://github.com/joshuacnf/Ctrl-G) | `softwareworkload/upstream/Ctrl-G` | `379463249c36` | HMM x DFA constrained decoding |
| [GraphBLAS](https://github.com/DrTimothyAldenDavis/GraphBLAS) | `softwareworkload/upstream/GraphBLAS` | `afbe9aaefe5d` | CPU sparse semiring linear algebra |
| [LAGraph](https://github.com/GraphBLAS/LAGraph) | `softwareworkload/upstream/LAGraph` | `7543de1f3103` | Graph algorithms over GraphBLAS |
| [graphblast](https://github.com/gunrock/graphblast) | `softwareworkload/upstream/graphblast` | `f56f9677276c` | GPU GraphBLAS baseline |
| [gunrock](https://github.com/gunrock/gunrock) | `softwareworkload/upstream/gunrock` | `748f79e66cc2` | GPU graph traversal and frontier baseline |
| [cugraph](https://github.com/rapidsai/cugraph) | `softwareworkload/upstream/cugraph` | `8443253f867d` | GPU graph baseline used by existing profiling kit |
| [kissat](https://github.com/arminbiere/kissat) | `softwareworkload/upstream/kissat` | `8af8e56f174b` | SAT and watched-literal BCP reference |
| [cadical](https://github.com/arminbiere/cadical) | `softwareworkload/upstream/cadical` | `c60730422e75` | SAT and watched-literal BCP comparison |
| [downward](https://github.com/aibasel/downward) | `softwareworkload/upstream/downward` | `9b81c7e422fd` | Planning heuristic and search baseline |
| [downward-benchmarks](https://github.com/aibasel/downward-benchmarks) | `softwareworkload/upstream/downward-benchmarks` | `e21d49c2cb61` | PDDL planning benchmark inputs |
| [NBFNet](https://github.com/DeepGraphLearning/NBFNet) | `softwareworkload/upstream/NBFNet` | `eb49ff2770d1` | Knowledge-graph Bellman-Ford neural reasoning |
| [torchhd](https://github.com/hyperdimensional-computing/torchhd) | `softwareworkload/upstream/torchhd` | `9d73e1bd1f16` | Vector symbolic cleanup and HDC workloads |
| [pgmpy](https://github.com/pgmpy/pgmpy) | `softwareworkload/upstream/pgmpy` | `fa42ee20e094` | Bayesian network inference baseline |
| [transformers](https://github.com/huggingface/transformers) | `softwareworkload/upstream/transformers` | `96331a9f93b7` | Beam search software implementation |
| [xgrammar](https://github.com/mlc-ai/xgrammar) | `softwareworkload/upstream/xgrammar` | `4221346f3d26` | Grammar-constrained decoding comparison and coverage audit |
| [sam](https://github.com/weiya711/sam) | `systolicarch/upstream/sam` | `6e477d142668` | Sparse streaming simulator and SAM-to-Onyx binding |
| [SCALE-Sim](https://github.com/scalesim-project/SCALE-Sim) | `systolicarch/upstream/SCALE-Sim` | `9f98c4371055` | Dense systolic cycle/traffic simulator |
| [DAG_Processor](https://github.com/nimish15shah/DAG_Processor) | `systolicarch/upstream/DAG_Processor` | `cefe56ec57a9` | DPU-v2 DAG compiler, input DAGs and SystemVerilog simulation |
| [GraphLily](https://github.com/cornell-zhang/GraphLily) | `systolicarch/upstream/GraphLily` | `f3438f9d8de0` | HBM FPGA semiring overlay source and middleware |
| [accel-sim-framework](https://github.com/accel-sim/accel-sim-framework) | `systolicarch/upstream/accel-sim-framework` | `d930ad6d02c0` | GPU simulation framework; requires matching workload traces |
| [ramulator2](https://github.com/CMU-SAFARI/ramulator2) | `systolicarch/upstream/ramulator2` | `72427a1bba37` | Detailed DRAM model for memory bottleneck validation |
| [stonne](https://github.com/stonne-simulator/stonne) | `systolicarch/upstream/stonne` | `22634b8b7668` | Dense/sparse accelerator simulator and MAERI-style networks |

## Dependency checkouts

| Directory | Commit |
|---|---|
| `softwareworkload/upstream/Lobster/core/src/gpu_runtime/libsclgpu/cccl` | `5a87e5d5706e` |
| `softwareworkload/upstream/graphblast/ext/cub` | `d622848f9fb6` |
| `softwareworkload/upstream/graphblast/ext/moderngpu` | `9e491c383e93` |
| `softwareworkload/upstream/xgrammar/3rdparty/cpptrace` | `6689d14c203e` |
| `softwareworkload/upstream/xgrammar/3rdparty/dlpack` | `bbd2f4d32427` |
| `softwareworkload/upstream/xgrammar/3rdparty/googletest` | `df1544bcee0c` |
| `systolicarch/upstream/sam/compiler/benchmark` | `6d50251d8e6e` |
| `systolicarch/upstream/sam/compiler/benchmark/googletest` | `4267679b6887` |
| `systolicarch/upstream/sam/compiler/taco` | `cf8f007bab94` |
| `systolicarch/upstream/sam/compiler/taco/python_bindings/pybind11` | `8de7772cc72d` |
| `systolicarch/upstream/sam/compiler/taco/test/bats` | `dcaec03e32e0` |
| `systolicarch/upstream/SCALE-Sim/submodules/ramulator` | `214f63584521` |
| `systolicarch/upstream/accel-sim-framework/gpu-simulator/gpgpu-sim` | `91880c53383d` |
| `systolicarch/upstream/accel-sim-framework/gpu-simulator/extern/pybind11` | `bf21a6dfe775` |
| `systolicarch/upstream/ramulator2/ext/yaml-cpp` | `56e3bb550c91` |
| `systolicarch/upstream/ramulator2/ext/fmt` | `e69e5f977d45` |
| `systolicarch/upstream/ramulator2/ext/nanobind` | `0f9ce749b257` |
| `systolicarch/upstream/ramulator2/ext/nanobind/ext/robin_map` | `188c45569cc2` |

## Artifact availability and remaining requirements

- REASON: no public implementation link located in the paper or the inspected author material; DPU-v2 is a separate baseline, not a reproduction of REASON.
- Revet / Capstan: no matching public simulator was located in the inspected paper and author pages; SAM is a separately identified streaming baseline.
- SAM includes its Onyx binding source. This does not constitute a complete Onyx silicon/toolchain reproduction.
- SRDatalog: the author-group Python frontend and bundled runtime are cloned; matching its revision and benchmarks to the paper remains necessary.
- Lobster's `.gitmodules` declares CCCL without a corresponding gitlink in this revision. CCCL is cloned at that declared repository URL and locked separately.
- GPU engines still need a usable GPU environment and matching toolchain. This host's NVIDIA driver query fails.
- DPU-v2 includes RTL and input DAGs. VCS is on this host's PATH; license checkout and actual simulation are not established by source acquisition.
- Python package installations, model weights and benchmark datasets are not implied by a source clone.
