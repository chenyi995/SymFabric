use scallop_core::integrate::IntegrateContext;
use scallop_core::runtime::provenance::top_k_proofs::TopKProofsProvenance;
use scallop_core::utils::RcFamily;
use std::time::Instant;
use std::hint::black_box;
fn main() {
    let a: Vec<String> = std::env::args().collect();
    let path = &a[1];
    let relation = &a[2];
    let k: usize = a[3].parse().unwrap();
    let reps: usize = a.get(4).map(|x| x.parse().unwrap()).unwrap_or(7);
    let mut expected = None;
    for rep in 0..reps+2 {
        let cold = Instant::now();
        let mut ctx = IntegrateContext::<_, RcFamily>::new(TopKProofsProvenance::<RcFamily>::new(k, false));
        ctx.import_file(path).unwrap();
        let load_ms = cold.elapsed().as_secs_f64()*1000.0;
        let start = Instant::now();
        ctx.compile_with_output_relations(Some(vec![relation])).unwrap();
        let compile_ms = start.elapsed().as_secs_f64()*1000.0;
        let profiling = std::env::var("SCALLOP_PROFILE").as_deref() == Ok("1");
        #[cfg(feature = "diagnostics")]
        if profiling { scallop_core::runtime::provenance::profile_reset(); }
        let start = Instant::now();
        ctx.run().unwrap();
        let run_ms = start.elapsed().as_secs_f64()*1000.0;
        #[cfg(feature = "diagnostics")]
        if profiling {
            scallop_core::runtime::provenance::profile_disable();
            eprintln!("PROFILE_JSON {}", serde_json::json!({"file":path,"k":k,"rep":rep,"profile":scallop_core::runtime::provenance::profile_report()}));
        }
        let wmc_profiling = std::env::var("SCALLOP_WMC_PROFILE").as_deref() == Ok("1");
        #[cfg(feature = "diagnostics")]
        if wmc_profiling { scallop_core::runtime::provenance::wmc_profile_reset(); }
        let start = Instant::now();
        let out = ctx.computed_relation_ref(relation).unwrap();
        let recover_ms = start.elapsed().as_secs_f64()*1000.0;
        #[cfg(feature = "diagnostics")]
        if wmc_profiling {
            scallop_core::runtime::provenance::wmc_profile_disable();
            eprintln!("WMC_PROFILE_JSON {}",serde_json::json!({"file":path,"k":k,"rep":rep,"profile":scallop_core::runtime::provenance::wmc_profile_report()}));
        }
        let result = format!("{}", black_box(out));
        if let Some(ref e) = expected { assert_eq!(e, &result, "repeat result mismatch"); }
        else { expected = Some(result.clone()); }
        let output_chars = result.len();
        let output_hash = result.bytes().fold(14695981039346656037u64, |h,b| (h ^ b as u64).wrapping_mul(1099511628211));
        if rep >= 2 {
            println!("{}", serde_json::json!({"file":path,"relation":relation,"k":k,"rep":rep-2,"load_ms":load_ms,"compile_ms":compile_ms,"run_ms":run_ms,"recover_ms":recover_ms,"symbolic_online_ms":run_ms+recover_ms,"output_chars":output_chars,"output_hash":output_hash,"boundary":"fresh context; parse/compile excluded from symbolic_online; no neural stage"}));
        }
    }
}
