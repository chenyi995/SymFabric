# SymFabric source preservation before cleanup

cw636 requested removal of the local SymFabric directory after ensuring its code is pushed to the existing remote repository. Root prepared this delivery on `codex/preserve-symfabric-20261027`, starting from master `bce0039151d2e6ded8007caee99b399427221f76`. The local directory removal follows the successful push.

## Delivered commit

- `32b37dbf50e9af06e35cdd47df953103d9329d4b` preserves existing profiling source, program fixtures, source manifests and the current modifications to four upstream checkouts as patches with base commit identities.

The delivery retains 266 original source/configuration files without changing their bytes. The source-preservation manifest and patch files bring the first commit to 273 files. Existing implementation text is retained as requested. Full raw experiment outputs, environments, downloaded dependencies and local draft reports are outside this source snapshot. The user explicitly authorized deletion of the local directory; this commit is not a complete backup of those artifacts.

## Validation

- All 273 staged files were read back from the Git index and matched their working-tree bytes.
- Python AST parsing passed for 121 files; Bash syntax checks passed for 84 files.
- All four current upstream patches passed reverse-application checks against their respective local trees. The untracked Vampire header was copied and verified byte for byte.
- The source manifest records SHA-256 bindings and the revisions of all 22 discovered upstream repositories.
- Workload experiments, GPU benchmarks and behavioral regression suites were not rerun for this unchanged-source preservation delivery. Rust/C++ execution is not newly validated.

A small generated-source receipt and the cleanup audit remain in the separate PIM workspace under `shared/runs/storage-cleanup-20261027-bkrled8j/`. No existing SymFabric source or configuration was archived or rewritten.
