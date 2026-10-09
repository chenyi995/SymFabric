#!/usr/bin/env python3
"""Create an immutable source bundle for Slurm jobs and avoid stale NFS .pyc."""
import argparse
from pathlib import Path
import tarfile

ap=argparse.ArgumentParser()
ap.add_argument("output",type=Path)
args=ap.parse_args()
with args.output.open("xb") as stream, tarfile.open(fileobj=stream,mode="w:gz") as archive:
    for path in sorted(Path(__file__).parent.glob("*.py")):
        archive.add(path,arcname=path.name)
print(args.output.resolve())
