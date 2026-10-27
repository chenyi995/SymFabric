#!/usr/bin/env python3
"""Patch only DNF top-k add/mult to expose diagnostic timing and shape counts.
Usage: python3 instrument_scallop.py /path/to/scallop
Caller: profile_reset(); ...run...; let json = profile_report();
The original algorithm is preserved; profile_reset enables diagnostic overhead.
"""
import sys
from pathlib import Path
p=Path(sys.argv[1])/'core/src/runtime/provenance/common/dnf_context.rs'
s=p.read_text()
if 'TXT_FIRST_PROFILE_INSTRUMENTATION' in s:
    raise SystemExit('already instrumented')
old_add='''    let mut t = t1.or(t2);
    t.dedup();
    self.retain_top_k(&mut t.clauses, k);
    t'''
new_add='''    if !profile_active() {
      let mut t = t1.or(t2);
      t.dedup();
      self.retain_top_k(&mut t.clauses, k);
      return t;
    }
    let start = std::time::Instant::now();
    let mut t = t1.or(t2);
    let a = start.elapsed().as_nanos() as u64;
    let n0 = t.clauses.len();
    let start = std::time::Instant::now();
    t.dedup();
    let b = start.elapsed().as_nanos() as u64;
    let n1 = t.clauses.len();
    let start = std::time::Instant::now();
    self.retain_top_k(&mut t.clauses, k);
    let d = start.elapsed().as_nanos() as u64;
    profile_record("add", [a,b,0,d], [n0,n1,n1,t.clauses.len()], t1, t2, k);
    t'''
old_mult='''    let mut t = t1.and(t2);
    t.dedup();
    self.retain_no_conflict(&mut t.clauses);
    self.retain_top_k(&mut t.clauses, k);
    t'''
new_mult='''    if !profile_active() {
      let mut t = t1.and(t2);
      t.dedup();
      self.retain_no_conflict(&mut t.clauses);
      self.retain_top_k(&mut t.clauses, k);
      return t;
    }
    let start = std::time::Instant::now();
    let mut t = t1.and(t2);
    let a = start.elapsed().as_nanos() as u64;
    let n0 = t.clauses.len();
    let start = std::time::Instant::now();
    t.dedup();
    let b = start.elapsed().as_nanos() as u64;
    let n1 = t.clauses.len();
    let start = std::time::Instant::now();
    self.retain_no_conflict(&mut t.clauses);
    let c = start.elapsed().as_nanos() as u64;
    let n2 = t.clauses.len();
    let start = std::time::Instant::now();
    self.retain_top_k(&mut t.clauses, k);
    let d = start.elapsed().as_nanos() as u64;
    profile_record("mult", [a,b,c,d], [n0,n1,n2,t.clauses.len()], t1, t2, k);
    t'''
assert s.count(old_add)==1 and s.count(old_mult)==1
s=s.replace(old_add,new_add).replace(old_mult,new_mult)
s+='''
// TXT_FIRST_PROFILE_INSTRUMENTATION; deterministic algorithms unchanged.
#[derive(Default)]
struct ProfileStats {
  enabled: bool,
  calls: BTreeMap<&'static str, u64>,
  stages: BTreeMap<&'static str, [u64; 4]>,
  candidates: BTreeMap<&'static str, [u64; 4]>,
  shapes: BTreeMap<(&'static str, usize, usize, usize, usize, usize), u64>,
}
std::thread_local! {
  static TXT_PROFILE: std::cell::RefCell<ProfileStats> = std::cell::RefCell::new(ProfileStats::default());
}
fn profile_active() -> bool { TXT_PROFILE.with(|p| p.borrow().enabled) }
/// Enable and clear per-thread diagnostic counters. Time this separately from baseline runs.
pub fn profile_reset() {
  TXT_PROFILE.with(|p| { let mut q = p.borrow_mut(); *q = ProfileStats::default(); q.enabled = true; });
}
/// Disable counters without deleting them.
pub fn profile_disable() { TXT_PROFILE.with(|p| p.borrow_mut().enabled = false); }
fn profile_record(op: &'static str, times: [u64;4], counts: [usize;4],
                  left: &DNFFormula, right: &DNFFormula, k: usize) {
  let lm = left.clauses.iter().map(|c| c.literals.len()).max().unwrap_or(0);
  let rm = right.clauses.iter().map(|c| c.literals.len()).max().unwrap_or(0);
  TXT_PROFILE.with(|p| {
    let mut q = p.borrow_mut();
    *q.calls.entry(op).or_default() += 1;
    let stages = q.stages.entry(op).or_default();
    for i in 0..4 { stages[i] += times[i]; }
    let candidates = q.candidates.entry(op).or_default();
    for i in 0..4 { candidates[i] += counts[i] as u64; }
    *q.shapes.entry((op,k,left.clauses.len(),right.clauses.len(),lm,rm)).or_default() += 1;
  });
}
/// Return counters for the current thread; stage durations omit instrumentation bookkeeping.
pub fn profile_report() -> serde_json::Value {
  TXT_PROFILE.with(|p| {
    let q = p.borrow();
    let shapes: Vec<_> = q.shapes.iter().map(|((op,k,l,r,lm,rm),n)| serde_json::json!({
      "op":op,"k":k,"left_proofs":l,"right_proofs":r,
      "max_left_literals":lm,"max_right_literals":rm,"calls":n
    })).collect();
    serde_json::json!({
      "note":"Diagnostic instrumented native run; counters are per thread; elapsed timings include Instant overhead; bookkeeping omitted from stage times.",
      "stage_order":["and_or_merge","dedup","disjunction_conflict","score_and_top_k"],
      "candidate_order":["after_merge","after_dedup","after_conflict","retained"],
      "calls":q.calls,"stage_ns":q.stages,"candidate_counts":q.candidates,"shapes":shapes
    })
  })
}
'''
p.write_text(s)
print(p)
