#![allow(dead_code)]
mod literal;
mod clause;
mod dnf_formula;
mod dnf_context;
mod disjunction;
use literal::Literal;
use clause::Clause;
use dnf_formula::DNFFormula;
use dnf_context::DNFContextTrait;
use disjunction::Disjunctions;
use std::collections::{BTreeSet, HashSet};
use std::hash::{Hash, Hasher};
use std::hint::black_box;
use std::time::Instant;
use std::fs::File;
use std::io::Write;
#[repr(C)]
struct CpuTimespec { sec: i64, nsec: i64 }
extern "C" { fn clock_gettime(clock_id: i32, value: *mut CpuTimespec) -> i32; }
fn thread_cpu_ns() -> u128 {
    let mut value=CpuTimespec{sec:0,nsec:0};
    assert_eq!(unsafe{clock_gettime(3, &mut value)}, 0); // Linux CLOCK_THREAD_CPUTIME_ID
    value.sec as u128*1_000_000_000+value.nsec as u128
}

struct Context { probabilities: Vec<f64>, disjunctions: Disjunctions }
impl DNFContextTrait for Context {
    fn fact_probability(&self, id: &usize) -> f64 { self.probabilities[*id] }
    fn has_disjunction_conflict(&self, facts: &BTreeSet<usize>) -> bool { self.disjunctions.has_conflict(facts) }
}
impl Hash for Literal {
    fn hash<H: Hasher>(&self, state: &mut H) { self.fact_id().hash(state); self.sign().hash(state); }
}
impl Hash for Clause {
    fn hash<H: Hasher>(&self, state: &mut H) { self.literals.hash(state); }
}
fn input(k: usize, len: usize, pattern: &str, seed: usize) -> (Context, DNFFormula, DNFFormula) {
    let stride = 4*k+8;
    let probabilities = (0..(len+1)*stride).map(|id| 0.65 + ((id.wrapping_mul(1103515245).wrapping_add(seed*12345)) % 349) as f64 / 1000.0).collect();
    let mut disjunctions=Disjunctions::new();
    if pattern=="duplicate_conflict" { disjunctions.add_disjunction(0, 0); disjunctions.add_disjunction(0, 1); }
    let make = |side: usize| DNFFormula::new((0..k).map(|i| {
        let i = if pattern=="duplicate_conflict" { i % ((k+1)/2).max(1) } else {i};
        Clause::new((0..len).map(|j| {
            let id = if pattern=="shared_prefix" && j < len*3/4 { j*stride }
                else if pattern=="duplicate_conflict" && j==0 { i%2 }
                else { j*stride + 2 + side*k+i };
            Literal::Pos(id)
        }).collect())
    }).collect());
    (Context{probabilities,disjunctions},make(0),make(1))
}
fn cached_top_k(ctx: &Context, clauses: Vec<Clause>, k: usize) -> Vec<Clause> {
    let mut scored: Vec<_> = clauses.into_iter().map(|c| {let p=ctx.clause_probability(&c); (p,c)}).collect();
    // Stable sort preserves the upstream tie order.
    scored.sort_by(|a,b| b.0.partial_cmp(&a.0).unwrap());
    scored.truncate(k);
    scored.into_iter().map(|(_,c)| c).collect()
}
fn execute(ctx: &Context, left: &DNFFormula, right: &DNFFormula, k: usize, variant: &str) -> DNFFormula {
    if variant=="upstream" { return ctx.top_k_mult(left,right,k); }
    let mut t=left.and(right);
    if variant=="cached_hash" {
        let mut seen=HashSet::new();
        t.clauses.retain(|c| seen.insert(c.clone()));
    } else { t.dedup(); }
    ctx.retain_no_conflict(&mut t.clauses);
    t.clauses=cached_top_k(ctx,t.clauses,k);
    t
}
fn instrumented(ctx: &Context,left: &DNFFormula,right: &DNFFormula,k:usize)->(DNFFormula,[u128;4]) {
    let mut times=[0;4];
    let mut start=Instant::now(); let mut t=left.and(right); times[0]=start.elapsed().as_nanos();
    start=Instant::now(); t.dedup(); times[1]=start.elapsed().as_nanos();
    start=Instant::now(); ctx.retain_no_conflict(&mut t.clauses); times[2]=start.elapsed().as_nanos();
    start=Instant::now(); ctx.retain_top_k(&mut t.clauses,k); times[3]=start.elapsed().as_nanos();
    (t,times)
}
fn time_run(ctx:&Context,left:&DNFFormula,right:&DNFFormula,k:usize,variant:&str,n:usize)->(f64,f64) {
    let cpu_start=thread_cpu_ns();
    let start=Instant::now();
    for _ in 0..n {black_box(execute(black_box(ctx),black_box(left),black_box(right),k,variant));}
    let wall=start.elapsed().as_nanos() as f64/n as f64;
    (wall,(thread_cpu_ns()-cpu_start) as f64/n as f64)
}
fn median(mut values:Vec<f64>)->f64 {values.sort_by(|a,b|a.partial_cmp(b).unwrap());values[values.len()/2]}
fn main(){
    let output=std::env::args().nth(1).unwrap_or("results.csv".to_string());
    let mut out=File::create(output).unwrap();
    writeln!(out,"pattern,k,input_literals,variant,repetitions,median_ns,min_ns,max_ns,cpu_median_ns,cpu_min_ns,cpu_max_ns,merge_ns,dedup_ns,conflict_ns,ranking_ns,candidates_after_merge,candidates_after_dedup,candidates_after_conflict,score_evaluations,scored_literal_visits,retained_proofs").unwrap();
    let mut checks=0;
    for pattern in ["distinct","shared_prefix","duplicate_conflict"] {for k in [1,2,4,8,16] {for len in [8,32,128] {
        let (ctx,left,right)=input(k,len,pattern,17);
        // Exact output check includes ordering and probabilities, across deterministic seeds.
        for seed in [1,7,17,103,997] {
            let (c,l,r)=input(k,len,pattern,seed);
            let expected=c.top_k_mult(&l,&r,k);
            for variant in ["cached_scores","cached_hash"] { assert_eq!(execute(&c,&l,&r,k,variant),expected);checks+=1; }
        }
        // More semantic edge cases: contradictory literals, repeated facts, and tied scores.
        let probe=time_run(&ctx,&left,&right,k,"upstream",20).1.max(1.0);
        let n=((15_000_000.0/probe) as usize).clamp(10,50000);
        let mut merged=left.and(&right); let nm=merged.clauses.len();
        merged.dedup();let nd=merged.clauses.len();ctx.retain_no_conflict(&mut merged.clauses);let nc=merged.clauses.len();
        let mut evals=0;let mut visits=0;
        merged.clauses.sort_by(|a,b|{evals+=2;visits+=a.len()+b.len();ctx.clause_probability(b).partial_cmp(&ctx.clause_probability(a)).unwrap()});
        let mut stages=[0_u128;4];
        for _ in 0..n {let (result,t)=instrumented(&ctx,&left,&right,k);black_box(result);for j in 0..4{stages[j]+=t[j];}}
        let variants=["upstream","cached_scores","cached_hash"];
        let mut wall_runs: [Vec<f64>;3]=std::array::from_fn(|_|Vec::new());
        let mut cpu_runs: [Vec<f64>;3]=std::array::from_fn(|_|Vec::new());
        // Interleave and rotate ordering to reduce temporal load and thermal bias.
        for round in 0..5 { for shift in 0..3 {let v=(round+shift)%3;
            let (wall,cpu)=time_run(&ctx,&left,&right,k,variants[v],n);
            wall_runs[v].push(wall);cpu_runs[v].push(cpu);
        }}
        for (v,variant) in variants.into_iter().enumerate() {
            let timings=wall_runs[v].clone();
            let lo=timings.iter().copied().fold(f64::INFINITY,f64::min);let hi=timings.iter().copied().fold(0.0,f64::max);let med=median(timings);
            let cpus=cpu_runs[v].clone();
            let clo=cpus.iter().copied().fold(f64::INFINITY,f64::min);let chi=cpus.iter().copied().fold(0.0,f64::max);let cmed=median(cpus);
            let s=if variant=="upstream" {stages.map(|v|v as f64/n as f64)}else{[f64::NAN;4]};
            writeln!(out,"{pattern},{k},{len},{variant},{n},{med:.3},{lo:.3},{hi:.3},{cmed:.3},{clo:.3},{chi:.3},{:.3},{:.3},{:.3},{:.3},{nm},{nd},{nc},{evals},{visits},{}",s[0],s[1],s[2],s[3],nc.min(k)).unwrap();
        }
    }}}
    let mut c=Context{probabilities:vec![0.5;20],disjunctions:Disjunctions::new()};
    c.disjunctions.add_disjunction(0,1);c.disjunctions.add_disjunction(0,2);
    let cases=[
        (DNFFormula::zero(),DNFFormula::one()),
        (DNFFormula::one(),DNFFormula::one()),
        (DNFFormula::singleton(1),DNFFormula::singleton(1)),
        (DNFFormula::singleton(1),DNFFormula::singleton(2)),
        (DNFFormula::singleton(1),DNFFormula::new(vec![Clause::new(vec![Literal::Neg(1)])])),
        (DNFFormula::new(vec![Clause::singleton(Literal::Pos(1)),Clause::singleton(Literal::Pos(3)),Clause::singleton(Literal::Pos(1))]),DNFFormula::singleton(4)),
    ];
    for (left,right) in cases { for k in [0,1,2,4] {let want=c.top_k_mult(&left,&right,k);for variant in ["cached_scores","cached_hash"]{assert_eq!(execute(&c,&left,&right,k,variant),want);checks+=1;}} }
    eprintln!("correctness_checks={checks}; all exact outputs matched; synthetic native microbenchmark, not E2E");
}
