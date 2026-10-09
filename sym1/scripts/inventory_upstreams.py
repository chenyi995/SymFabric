#!/usr/bin/env python3
"""Record and verify source checkout revisions; never fetch, build, or run benchmarks."""
import datetime as dt
import json
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def git(path, *args):
    return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()


def inspect(path):
    revision = git(path, "rev-parse", "HEAD")
    check = subprocess.run(["git", "-C", str(path), "fsck", "--connectivity-only", "--no-dangling"],
                           capture_output=True, text=True)
    subs = git(path, "submodule", "status", "--recursive")
    uninitialized = [line for line in subs.splitlines() if line.startswith(("-", "+", "U"))]
    assert check.returncode == 0, (path, check.stderr)
    assert not uninitialized, (path, uninitialized)
    assert len(revision) == 40
    return {"path": str(path.relative_to(ROOT)), "url": git(path, "remote", "get-url", "origin"),
            "commit": revision, "shallow": git(path, "rev-parse", "--is-shallow-repository") == "true",
            "tracked_diff": git(path, "diff", "--stat"), "submodules": subs.splitlines(),
            "connectivity_check": "PASS"}


def main():
    catalog = json.loads((ROOT/"sym1/upstream_catalog.json").read_text())
    top_paths = {r["path"] for r in catalog["repositories"]}
    rows = []
    for repo in catalog["repositories"]:
        path = ROOT/repo["path"]
        assert (path/".git").exists(), f"missing checkout: {path}"
        rows.append(dict(repo, **{k:v for k,v in inspect(path).items() if k not in repo}))
    deps = []
    for parent in ("softwareworkload/upstream", "systolicarch/upstream"):
        for directory, dirs, files in os.walk(ROOT/parent):
            if ".git" in dirs or ".git" in files:
                path = Path(directory)
                if str(path.relative_to(ROOT)) not in top_paths:
                    deps.append(inspect(path))
            dirs[:] = [d for d in dirs if d not in (".git", "__pycache__", "build", "node_modules", ".venv")]
    lock = {"recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "provenance": "measured from local Git checkouts; fsck connectivity and submodule status checked",
            "repositories": rows, "dependency_checkouts": deps}
    target = ROOT/"sym1/upstream_lock.json"
    with target.open("x") as f:
        json.dump(lock, f, indent=2)
        f.write("\n")
    # Independently re-read the saved lock and compare hashes against Git.
    saved = json.loads(target.read_text())
    for row in saved["repositories"]+saved["dependency_checkouts"]:
        assert git(ROOT/row["path"], "rev-parse", "HEAD") == row["commit"]
    lines = ["# sym1 source acquisition inventory", "",
             f"Verified at {lock['recorded_at_utc']}.", "",
             f"Measured checkout count: {len(rows)} primary repositories and {len(deps)} dependency checkouts.",
             "Every listed checkout passed Git object connectivity checks; all registered submodules are initialized at their recorded commits.",
             "Clone completeness is separate from build, runtime dependency, dataset, and benchmark reproduction status.", "",
             "Primary sources are shallow clones. Exact full commits and recursive submodule revisions are in `upstream_lock.json`.",
             "Downloaded code stays under the ignored `upstream/` directories.", "",
             "| Repository | Local directory | Commit | Purpose |", "|---|---|---|---|"]
    for row in rows:
        lines.append(f"| [{row['name']}]({row['url'].removesuffix('.git')}) | `{row['path']}` | `{row['commit'][:12]}` | {row['purpose']} |")
    lines += ["", "## Dependency checkouts", "", "| Directory | Commit |", "|---|---|"]
    lines += [f"| `{r['path']}` | `{r['commit'][:12]}` |" for r in deps]
    lines += ["", "## Artifact availability and remaining requirements", "",
              "- REASON: no public implementation link located in the paper or the inspected author material; DPU-v2 is a separate baseline, not a reproduction of REASON.",
              "- Revet / Capstan: no matching public simulator was located in the inspected paper and author pages; SAM is a separately identified streaming baseline.",
              "- SAM includes its Onyx binding source. This does not constitute a complete Onyx silicon/toolchain reproduction.",
              "- SRDatalog: the author-group Python frontend and bundled runtime are cloned; matching its revision and benchmarks to the paper remains necessary.",
              "- Lobster's `.gitmodules` declares CCCL without a corresponding gitlink in this revision. CCCL is cloned at that declared repository URL and locked separately.",
              "- GPU engines still need a usable GPU environment and matching toolchain. This host's NVIDIA driver query fails.",
              "- DPU-v2 includes RTL and input DAGs. VCS is on this host's PATH; license checkout and actual simulation are not established by source acquisition.",
              "- Python package installations, model weights and benchmark datasets are not implied by a source clone.", ""]
    with (ROOT/"sym1/upstream_status.md").open("x") as f:
        f.write("\n".join(lines))
    print(json.dumps({"primary_repositories": len(rows), "dependency_checkouts": len(deps),
                      "lock": str(target), "validation": "PASS"}))


if __name__ == "__main__":
    main()
