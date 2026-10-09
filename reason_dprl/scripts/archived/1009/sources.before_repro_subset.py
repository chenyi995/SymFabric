#!/usr/bin/env python3
"""Record, restore and verify shared source checkouts without running workloads."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
from pathlib import Path
import subprocess
import sys


# codex: decision start — use one shared checkout per source and relative project links.
PROJECT = Path(__file__).resolve().parents[1]
ROOT = PROJECT.parent
CATALOG = (
    ("DPrL-AAAI", "workloads/dprl", "softwareworkload/upstream/DPrL-AAAI"),
    ("accel-sim-framework", "simulators/accel-sim", "systolicarch/upstream/accel-sim-framework"),
    ("DAG_Processor", "simulators/dpu-v2", "systolicarch/upstream/DAG_Processor"),
    ("sam", "simulators/sam", "systolicarch/upstream/sam"),
    ("SCALE-Sim", "simulators/scale-sim", "systolicarch/upstream/SCALE-Sim"),
    ("stonne", "simulators/stonne", "systolicarch/upstream/stonne"),
    ("ramulator2", "simulators/ramulator2", "systolicarch/upstream/ramulator2"),
    ("GraphLily", "simulators/graphlily", "systolicarch/upstream/GraphLily"),
)
# codex: decision end


def git(path: Path, *args: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(path), *args], text=True, stderr=subprocess.STDOUT
    ).strip()


def inspect(checkout: str) -> dict:
    path = ROOT / checkout
    if not (path / ".git").exists():
        raise ValueError(f"Missing Git checkout: {checkout}")
    dirty = git(path, "status", "--porcelain", "--untracked-files=no")
    if dirty:
        raise ValueError(f"Tracked changes in {checkout}; preserve them and resolve manually")
    return {
        "checkout": checkout,
        "url": git(path, "remote", "get-url", "origin"),
        "commit": git(path, "rev-parse", "HEAD"),
    }


def verify(source: dict) -> None:
    actual = inspect(source["checkout"])
    if actual["commit"] != source["commit"]:
        raise ValueError(
            f"Revision mismatch for {source['checkout']}: "
            f"expected {source['commit']}, found {actual['commit']}"
        )


def all_checkouts(lock: dict):
    for source in lock["sources"]:
        yield source
        yield from source["dependencies"]


def record() -> None:
    """Read source revisions from Git, write once, then independently re-check."""
    inventory = json.loads((ROOT / "sym1/upstream_lock.json").read_text())
    sources, links = [], []
    for name, entry, checkout in CATALOG:
        source = {"name": name, **inspect(checkout)}
        source["dependencies"] = [
            inspect(item["path"])
            for item in inventory["dependency_checkouts"]
            if item["path"].startswith(checkout + "/")
        ]
        sources.append(source)
        links.append({"entry": entry, "checkout": checkout})
    links.append({
        "entry": "simulators/gpgpu-sim",
        "checkout": "systolicarch/upstream/accel-sim-framework/gpu-simulator/gpgpu-sim",
    })
    lock = {
        "recorded_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "provenance": "Inspected local Git revisions, re-read after writing; source availability only",
        "sources": sources,
        "links": links,
        "unavailable": [{
            "name": "REASON",
            "status": "No REASON-specific compiler and simulator located in this checkout",
        }],
    }
    with (PROJECT / "sources.lock.json").open("x") as output:
        json.dump(lock, output, indent=2)
        output.write("\n")
    saved = read_lock()
    for source in all_checkouts(saved):
        verify(source)
    print("PASS: recorded source revisions and independently re-checked the saved lock")


def read_lock() -> dict:
    lock = json.loads((PROJECT / "sources.lock.json").read_text())
    for source in all_checkouts(lock):
        relative = Path(source["checkout"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Checkout must be relative to the repository: {relative}")
    checkouts = {source["checkout"] for source in all_checkouts(lock)}
    entries = set()
    for link in lock["links"]:
        relative = Path(link["entry"])
        if relative.is_absolute() or ".." in relative.parts or len(relative.parts) < 2:
            raise ValueError(f"Invalid project entry: {relative}")
        if link["entry"] in entries or link["checkout"] not in checkouts:
            raise ValueError(f"Duplicate entry or unlocked source: {link}")
        entries.add(link["entry"])
    return lock


def link_destination(link: dict) -> tuple[Path, str]:
    entry = PROJECT / link["entry"]
    destination = os.path.relpath(ROOT / link["checkout"], entry.parent)
    return entry, destination


def check_link_conflicts(lock: dict) -> None:
    for link in lock["links"]:
        entry, destination = link_destination(link)
        if entry.is_symlink():
            if os.readlink(entry) != destination:
                raise ValueError(f"Conflicting link at {entry}; will not overwrite it")
        elif entry.exists():
            raise ValueError(f"Existing non-link at {entry}; will not replace it")


def link_sources(lock: dict) -> None:
    for source in all_checkouts(lock):
        verify(source)
    check_link_conflicts(lock)
    for link in lock["links"]:
        entry, destination = link_destination(link)
        if not entry.is_symlink():
            entry.parent.mkdir(parents=True, exist_ok=True)
            entry.symlink_to(destination, target_is_directory=True)
    check(lock)


def check(lock: dict) -> None:
    for source in all_checkouts(lock):
        verify(source)
    check_link_conflicts(lock)
    for link in lock["links"]:
        entry, _ = link_destination(link)
        if not entry.is_symlink() or not entry.is_dir():
            raise ValueError(f"Missing or broken source link: {entry}")
        if entry.resolve() != (ROOT / link["checkout"]).resolve():
            raise ValueError(f"Wrong source destination: {entry}")
        print(f"PASS: {link['entry']} -> {os.readlink(entry)}")
    print("PASS: source wiring and pinned revisions; builds and performance are not tested")
    for item in lock["unavailable"]:
        print(f"UNAVAILABLE: {item['name']}: {item['status']}")


def fetch(lock: dict) -> None:
    """Restore missing checkouts only, including separately recorded dependencies."""
    # codex: decision start — reject conflicts before downloads and never update existing trees.
    check_link_conflicts(lock)
    for source in all_checkouts(lock):
        path = ROOT / source["checkout"]
        if (path / ".git").exists():
            verify(source)
        elif path.is_symlink() or (path.exists() and (not path.is_dir() or any(path.iterdir()))):
            raise ValueError(f"Existing non-checkout at {path}; will not replace it")
    for source in all_checkouts(lock):
        path = ROOT / source["checkout"]
        if (path / ".git").exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run([
            "git", "clone", "--no-checkout", "--depth", "1", source["url"], str(path)
        ], check=True)
        subprocess.run([
            "git", "-C", str(path), "fetch", "--depth", "1", "origin", source["commit"]
        ], check=True)
        subprocess.run([
            "git", "-C", str(path), "checkout", "--detach", source["commit"]
        ], check=True)
        verify(source)
    link_sources(lock)
    # codex: decision end


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("record", "check", "link", "fetch"))
    args = parser.parse_args()
    try:
        if args.command == "record":
            record()
        else:
            lock = read_lock()
            {"check": check, "link": link_sources, "fetch": fetch}[args.command](lock)
    except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        if isinstance(exc, subprocess.CalledProcessError) and exc.output:
            print(exc.output, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
