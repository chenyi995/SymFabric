#!/usr/bin/env python3
"""Recover and validate the dynamic-state follow-up, without resetting work."""
import argparse
import json
from pathlib import Path

import reproduce
import sources
from summarize_dynamic import summarize

ROOT=Path(__file__).resolve().parents[2]


def main():
    ap=argparse.ArgumentParser()
    sub=ap.add_subparsers(dest='command',required=True)
    src=sub.add_parser('sources')
    src.add_argument('--fetch',action='store_true')
    restore=sub.add_parser('restore')
    restore.add_argument('bundle',type=Path,nargs='?')
    sub.add_parser('check')
    args=ap.parse_args()
    lock=json.loads((ROOT/'reason_dprl/dynamic.lock.json').read_text())
    if args.command=='sources':
        manifest={'sources':lock['sources'],'links':[],'unavailable':[]}
        (sources.fetch if args.fetch else sources.check)(manifest)
    elif args.command=='restore':
        bundle=args.bundle or Path(lock['bundle']['server_path'])
        reproduce.restore(ROOT,bundle,lock['bundle'])
    else:
        for item in lock['files']:
            reproduce.verify_file(ROOT,item)
        for source in lock['sources']:
            sources.verify(source)
        for run in lock['primary_runs']:
            summary=summarize(ROOT/run['path'],3,20)
            assert summary['slurm_job_id']==run['slurm_job_id']
        baseline=json.loads((ROOT/'reason_dprl/repro.lock.json').read_text())
        for name in ('family','wn18rr'):
            reproduce.verify_file(ROOT,baseline['checkpoints'][name])
        print('PASS: source pins, checkpoint identities, artifact hashes and complete 3x20 GPU comparisons')


if __name__=='__main__':
    main()
