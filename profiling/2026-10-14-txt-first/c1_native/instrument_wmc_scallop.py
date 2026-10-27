#!/usr/bin/env python3
"""Add diagnostic WMC stage counters; does not change the WMC algorithm.
Usage: python3 instrument_wmc_scallop.py /path/to/scallop
API: wmc_profile_reset(), wmc_profile_disable(), wmc_profile_report().
"""
from pathlib import Path
import sys
p=Path(sys.argv[1])/'core/src/runtime/provenance/common/as_boolean_formula.rs'
s=p.read_text()
assert 'TXT_FIRST_WMC_PROFILE' not in s
# Work backwards so offsets from the original string remain valid.
for name,formula_name in [('wmc_with_disjunctions','formula_with_disj'),('wmc','formula')]:
    start=s.index(f'  fn {name}<')
    begin=s.index('    let formula = self.as_boolean_formula();',start)
    end=s.index('    let sdd_config = ',begin)
    expression=s[begin:end]
    injected='''    if wmc_profile_active() {
      let formula_start = std::time::Instant::now();
'''+''.join('  '+line+'\n' for line in expression.rstrip().splitlines())+f'''
      let formula_ns = formula_start.elapsed().as_nanos() as u64;
      return profiled_wmc_run({formula_name}, formula_ns, s, v, "{name}");
    }}
'''
    s=s[:begin]+injected+s[begin:]
s+='''
// TXT_FIRST_WMC_PROFILE: diagnostic timing only, no cached/eager evaluation.
#[derive(Default)]
struct WmcProfileStats {
  enabled: bool,
  calls: std::collections::BTreeMap<&'static str, u64>,
  stage_ns: std::collections::BTreeMap<&'static str, [u64;6]>,
  sdd_nodes_total: u64,
  sdd_nodes_max: usize,
  formula_node_total: u64,
  formula_keys: std::collections::BTreeMap<Vec<usize>, u64>,
}
std::thread_local! {
  static TXT_WMC_PROFILE: std::cell::RefCell<WmcProfileStats> = std::cell::RefCell::new(WmcProfileStats::default());
}
fn wmc_profile_active() -> bool { TXT_WMC_PROFILE.with(|p| p.borrow().enabled) }
/// Enable and clear per-thread WMC diagnostics. Does not reset DNF diagnostics.
pub fn wmc_profile_reset() {
  TXT_WMC_PROFILE.with(|p| { let mut q=p.borrow_mut(); *q=WmcProfileStats::default(); q.enabled=true; });
}
pub fn wmc_profile_disable() { TXT_WMC_PROFILE.with(|p| p.borrow_mut().enabled=false); }
/// Exact ordered syntax key, including every variable ID; no hash collisions.
fn wmc_formula_key(f: &sdd::BooleanFormula, key: &mut Vec<usize>) -> u64 {
  use sdd::BooleanFormula::*;
  match f {
    False => {key.push(0);1}, True => {key.push(1);1},
    Pos{var_id} => {key.push(2);key.push(*var_id);1},
    Neg{var_id} => {key.push(3);key.push(*var_id);1},
    Not{form} => {key.push(4);1+wmc_formula_key(form,key)},
    And{left,right} => {key.push(5);1+wmc_formula_key(left,key)+wmc_formula_key(right,key)},
    Or{left,right} => {key.push(6);1+wmc_formula_key(left,key)+wmc_formula_key(right,key)},
  }
}
fn profiled_wmc_run<S,V>(formula:sdd::BooleanFormula, formula_ns:u64,
                         s:&S,v:&V,op:&'static str)->S::Element
where S:Semiring,V:Fn(&usize)->S::Element {
  let mut times=[formula_ns,0,0,0,0,0];
  let start=std::time::Instant::now();
  let config=sdd::bottom_up::SDDBuilderConfig::with_formula(&formula);
  times[1]=start.elapsed().as_nanos() as u64;
  let start=std::time::Instant::now();
  let builder=sdd::bottom_up::SDDBuilder::with_config(config);
  times[2]=start.elapsed().as_nanos() as u64;
  let start=std::time::Instant::now();
  let diagram=builder.build(&formula);
  times[3]=start.elapsed().as_nanos() as u64;
  let start=std::time::Instant::now();
  let value=diagram.eval_t(v,s);
  times[4]=start.elapsed().as_nanos() as u64;
  let nodes=diagram.sdd_nodes.len();
  let mut key=Vec::new();
  let formula_nodes=wmc_formula_key(&formula,&mut key);
  let start=std::time::Instant::now();
  drop(diagram);drop(formula);
  times[5]=start.elapsed().as_nanos() as u64;
  TXT_WMC_PROFILE.with(|p| {
    let mut q=p.borrow_mut();
    *q.calls.entry(op).or_default()+=1;
    let total=q.stage_ns.entry(op).or_default();for i in 0..6 {total[i]+=times[i];}
    q.sdd_nodes_total+=nodes as u64;q.sdd_nodes_max=q.sdd_nodes_max.max(nodes);
    q.formula_node_total+=formula_nodes;
    *q.formula_keys.entry(key).or_default()+=1;
  });
  value
}
pub fn wmc_profile_report()->serde_json::Value {
  TXT_WMC_PROFILE.with(|p| {
    let q=p.borrow();
    let repeated_calls=q.formula_keys.values().map(|n|n-1).sum::<u64>();
    serde_json::json!({
      "note":"Per-thread diagnostic; Instant overhead included; exact formula-key bookkeeping excluded. build includes upstream apply recursion, garbage collection, and consumed-builder cleanup. No evaluation or compilation cache added.",
      "stage_order":["formula","config_vtree","builder_init","build_sdd","eval_sdd","drop_sdd_formula"],
      "calls":q.calls,"stage_ns":q.stage_ns,
      "sdd_nodes_total":q.sdd_nodes_total,"sdd_nodes_max":q.sdd_nodes_max,
      "formula_nodes_total":q.formula_node_total,
      "unique_exact_formula_syntax":q.formula_keys.len(),
      "repeated_exact_formula_calls":repeated_calls,
      "max_exact_formula_reuses":q.formula_keys.values().max().copied().unwrap_or(0)
    })
  })
}
'''
p.write_text(s)
print(p)
