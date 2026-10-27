#!/usr/bin/env python3
"""Add process-wide SCALLOP_CACHE_SCORES=1 opt-in stable cached-score ranking.
Usage: python3 cache_scores_scallop.py /path/to/scallop
Compatible with instrument_scallop.py in either order.
"""
from pathlib import Path
import sys
p=Path(sys.argv[1])/'core/src/runtime/provenance/common/dnf_context.rs'
s=p.read_text()
old='''  fn retain_top_k(&self, clauses: &mut Vec<Clause>, k: usize) {
    clauses.sort_by'''
new='''  fn retain_top_k(&self, clauses: &mut Vec<Clause>, k: usize) {
    // TXT_FIRST_CACHE_SCORES: stable ordering and exact probability fold preserved.
    static CACHE_SCORES: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *CACHE_SCORES.get_or_init(|| std::env::var("SCALLOP_CACHE_SCORES").as_deref() == Ok("1")) {
      let mut scored = std::mem::take(clauses).into_iter().map(|c| {
        let score = self.clause_probability(&c);
        (score, c)
      }).collect::<Vec<_>>();
      scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
      scored.truncate(k);
      *clauses = scored.into_iter().map(|(_, c)| c).collect();
      return;
    }
    clauses.sort_by'''
assert s.count(old)==1 and 'TXT_FIRST_CACHE_SCORES' not in s
p.write_text(s.replace(old,new))
print(p)
