#!/usr/bin/env python3
"""Opt-in SDD GC DAG visitation guard; SCALLOP_MEMO_GC=1.
Only prune recursion after a node is already in the reachability set.
"""
from pathlib import Path
import sys
p=Path(sys.argv[1])/'lib/sdd/src/bottom_up.rs'
s=p.read_text()
assert 'TXT_FIRST_MEMO_GC' not in s
old='''  fn mark_visited(sdd_nodes: &SDDNodes, node: SDDNodeIndex, visited: &mut HashSet<SDDNodeIndex>) {
    visited.insert(node);'''
new='''  fn mark_visited(sdd_nodes: &SDDNodes, node: SDDNodeIndex, visited: &mut HashSet<SDDNodeIndex>) {
    // TXT_FIRST_MEMO_GC: same reachable-node set, visit each DAG node once.
    static MEMO_GC: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *MEMO_GC.get_or_init(|| std::env::var("SCALLOP_MEMO_GC").as_deref() == Ok("1")) {
      if !visited.insert(node) { return; }
    } else {
      visited.insert(node);
    }'''
assert s.count(old)==1
p.write_text(s.replace(old,new))
print(p)
