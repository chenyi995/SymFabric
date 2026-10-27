use sdd::{BooleanFormula,SDD,SDDNode,SDDNodeIndex,Semiring};
use std::collections::BTreeSet;
struct Prob;
impl Semiring for Prob {
 type Element=f64;
 fn zero(&self)->f64{0.0}fn one(&self)->f64{1.0}
 fn add(&self,a:f64,b:f64)->f64{a+b}fn mult(&self,a:f64,b:f64)->f64{a*b}
 fn negate(&self,a:f64)->f64{1.0-a}
}
fn reachable(s:&SDD,id:SDDNodeIndex,out:&mut BTreeSet<SDDNodeIndex>){
 if !out.insert(id){return;}
 if let SDDNode::Or{children}=&s.sdd_nodes[id] {
  for e in children {reachable(s,e.prime,out);reachable(s,e.sub,out);}
 }
}
fn make(f:&BooleanFormula)->SDD {
 sdd::bottom_up::SDDBuilder::with_config(sdd::bottom_up::SDDBuilderConfig::with_formula(f)).build(f)
}
fn main(){
 let pos=sdd::bf_pos;
 let f=(pos(0)&pos(1)) | (pos(0)&pos(2)) | (pos(3)&pos(1)) | (pos(3)&pos(2));
 let g=(f.clone()&pos(4)) | (f.clone()&sdd::bf_neg(5)) | (f.clone()&pos(6));
 let cases=[sdd::bf_false(),sdd::bf_true(),f,g];
 let weights=[[0.1,0.2,0.3,0.4,0.5,0.6,0.7],[0.8,0.7,0.6,0.5,0.4,0.3,0.2]];
 for (case,f) in cases.iter().enumerate(){
  let graph=make(f);let mut visited=BTreeSet::new();
  for root in &graph.roots{reachable(&graph,*root,&mut visited);}
  assert_eq!(visited.len(),graph.sdd_nodes.len(),"GC must retain exactly reachable nodes");
  // Canonical node-id/node-content listing is compared across process variants.
  let signature=visited.iter().map(|id|format!("{:?}:{:?}",id,&graph.sdd_nodes[*id])).collect::<Vec<_>>().join(";");
  print!("case={case} nodes={} structure={signature} probabilities=",graph.sdd_nodes.len());
  for p in weights {
   let v=|id:&usize| p[*id];let result=graph.eval_t(&v,&Prob);
   let ab=(p[0]+p[3]-p[0]*p[3])*(p[1]+p[2]-p[1]*p[2]);
   let expected=match case{0=>0.0,1=>1.0,2=>ab,_=>ab*(1.0-(1.0-p[4])*p[5]*(1.0-p[6]))};
   assert!((result-expected).abs()<1e-12,"weighted count differs from direct probability");
   print!("{:016x},",result.to_bits());
  }
  println!();
 }
 println!("PASS: two different probability assignments on the same built SDD; all GC node sets exactly reachable");
}
