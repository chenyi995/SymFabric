#!/usr/bin/env bash
# One-time environment setup for the SymFabric profiling kit.  Untested on this host (no GPU, no network use here).
# usage: bash profiling/scripts/setup_env.sh [env|clone|build|data|all]   (default all; every step is idempotent)
# Everything external lives under $SYMFABRIC_EXT (default /data2/$USER/symfabric_ext), never inside the repository.
# env: ENV_NAME (conda env, default symfabric-prof)  JOBS (build parallelism)  SKIP_RAPIDS=1 (no cuGraph wheels)
#      TORCH_INDEX (PyTorch wheel index; pick the CUDA build matching `nvidia-smi`, see pytorch.org/get-started/locally)
set -euo pipefail
EXT=${SYMFABRIC_EXT:-/data2/$USER/symfabric_ext}; ENV_NAME=${ENV_NAME:-symfabric-prof}; JOBS=${JOBS:-16}
mkdir -p "$EXT/data"
clone() { [ -d "$EXT/$2/.git" ] || git clone "$1" "$EXT/$2"; }                     # $1 url  $2 dir
fetch() { [ -s "$EXT/data/$2" ] || curl -L --fail -o "$EXT/data/$2" "$1"; }         # $1 url  $2 file
pipi()  { conda run -n "$ENV_NAME" python -m pip install "$@"; }

step_env() {
  conda env list | grep -q "^$ENV_NAME " || conda create -y -n "$ENV_NAME" python=3.10
  # PyTorch 2.5 = the version REASON profiled with (ref/notes/2026-HPCA-REASON.md); cu124 wheels need driver >= 12.4
  pipi "torch==2.5.*" --index-url "${TORCH_INDEX:-https://download.pytorch.org/whl/cu124}"
  pipi transformers torch-hd pyjuice pandas          # torch-hd = torchhd; pyjuice pulls triton (verify on the GPU host)
  # cuGraph/cudf wheels from the NVIDIA index; package names and CUDA suffix must match https://docs.rapids.ai/install
  [ "${SKIP_RAPIDS:-0}" = 1 ] || pipi --extra-index-url=https://pypi.nvidia.com cugraph-cu12 cudf-cu12
  echo "activate with: conda activate $ENV_NAME"
}
step_clone() {
  clone https://github.com/souffle-lang/souffle souffle
  clone https://github.com/arminbiere/kissat kissat
  clone https://github.com/Tractables/pyjuice pyjuice                  # runtime uses the pip package; repo for examples/
  clone https://github.com/joshuacnf/Ctrl-G Ctrl-G                     # HMM option A; install: see its README
  clone https://github.com/DrTimothyAldenDavis/GraphBLAS GraphBLAS
  clone https://github.com/GraphBLAS/LAGraph LAGraph                   # BFS / SSSP / triangle-count demos over GraphBLAS
  clone https://github.com/DeepGraphLearning/NBFNet NBFNet             # KG option A; needs torchdrug (heavy)
  clone https://github.com/aibasel/downward downward
  clone https://github.com/aibasel/downward-benchmarks downward-benchmarks   # IPC domains (gripper, logistics00, ...)
  clone https://github.com/harp-lab/GDlog GDlog                        # GPU Datalog; maintenance status not verified
  clone https://github.com/rapidsai/cugraph cugraph                    # source only (python/cugraph examples); wheels above
}
step_build() {
  # Souffle needs bison >= 3.6, flex, mcpp, libffi, sqlite3, zlib, ncurses, a C++17 compiler (souffle docs: build)
  [ -x "$EXT/souffle/build/src/souffle" ] || (cd "$EXT/souffle" && cmake -S . -B build && cmake --build build -j "$JOBS")
  [ -x "$EXT/kissat/build/kissat" ] || (cd "$EXT/kissat" && ./configure && make -j "$JOBS")
  [ -e "$EXT/install/lib/libgraphblas.so" ] || (cd "$EXT/GraphBLAS" && cmake -S . -B build -DCMAKE_INSTALL_PREFIX="$EXT/install" \
      && cmake --build build -j "$JOBS" && cmake --install build)
  # LAGraph demo executables (bfs_demo, sssp_demo, tc_demo) live under build/src/benchmark/: verify names with ls
  [ -d "$EXT/LAGraph/build" ] || (cd "$EXT/LAGraph" && cmake -S . -B build -DCMAKE_PREFIX_PATH="$EXT/install" \
      -DCMAKE_INSTALL_PREFIX="$EXT/install" && cmake --build build -j "$JOBS")
  [ -x "$EXT/downward/builds/release/bin/downward" ] || (cd "$EXT/downward" && ./build.py -j "$JOBS")
}
step_data() {
  # SNAP graphs (https://snap.stanford.edu/data/): small / medium / large; converted by workloads/graph/snap_to_mtx.py
  fetch https://snap.stanford.edu/data/wiki-Vote.txt.gz wiki-Vote.txt.gz
  fetch https://snap.stanford.edu/data/web-Google.txt.gz web-Google.txt.gz
  fetch https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz soc-LiveJournal1.txt.gz
  # SAT: SATLIB random 3-SAT (tiny, sanity only).  SAT Competition instances are served by the Global Benchmark
  # Database, https://benchmark-database.de/ (URL pattern believed to be /file/<hash>?context=cnf; verify on the site
  # and put the files in $EXT/data/sat/); workloads/sat/gen_cnf.py generates fallback instances.
  fetch https://www.cs.ubc.ca/~hoos/SATLIB/Benchmarks/SAT/RND3SAT/uf250-1065.tar.gz uf250-1065.tar.gz
  mkdir -p "$EXT/data/sat" && tar -xzf "$EXT/data/uf250-1065.tar.gz" -C "$EXT/data/sat" 2>/dev/null || true
  # GPT-2 weights are downloaded by transformers on first use; keep the cache external: export HF_HOME=$EXT/hf
}
for step in "${@:-all}"; do
  case $step in
    env) step_env ;; clone) step_clone ;; build) step_build ;; data) step_data ;;
    all) step_env; step_clone; step_build; step_data ;;
    *) echo "unknown step $step" >&2; exit 2 ;;
  esac
done
