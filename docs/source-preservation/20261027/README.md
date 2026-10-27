# Source preservation before local cleanup

cw636 requested that existing SymFabric code be pushed before removing the local checkout. The original master commit is recorded in manifest.json. Added profiling source is preserved without changing its implementation or existing text. This snapshot does not assert that historical workload tests pass.

The manifest binds each newly retained source file by SHA-256 and records upstream commits. Restore a modified nested checkout at its recorded commit, apply its named patch with git apply, and copy any separately recorded untracked header to its source path. Installed environments, downloaded dependencies and full raw experiment outputs are not included. Restoring source does not restore those deleted local artifacts.
