#!/usr/bin/env python3
"""Opt-in per-eval SDD DAG memoization; SCALLOP_MEMO_EVAL=1.
Each eval owns a fresh cache. Assumes pure var_assign and semiring operations,
as in Scallop's probability/gradient semirings. Arithmetic ordering unchanged.
"""
from pathlib import Path
import sys
p=Path(sys.argv[1])/'lib/sdd/src/sdd.rs'
s=p.read_text()
assert 'TXT_FIRST_MEMO_EVAL' not in s
old='''    self.eval_node_t(self.roots[0], var_assign, semiring)
  }

  fn eval_node_t'''
new='''    // TXT_FIRST_MEMO_EVAL: never reuse values across evaluations/weights.
    static MEMO_EVAL: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    if *MEMO_EVAL.get_or_init(|| std::env::var("SCALLOP_MEMO_EVAL").as_deref() == Ok("1")) {
      let mut cache = HashMap::new();
      self.eval_node_t_memo(self.roots[0], var_assign, semiring, &mut cache)
    } else {
      self.eval_node_t(self.roots[0], var_assign, semiring)
    }
  }

  fn eval_node_t'''
assert s.count(old)==1
s=s.replace(old,new)
a=s.index('  fn eval_node_t<T, V>')
b=s.index('  pub fn dot(',a)
method=s[a:b]
method=method.replace('fn eval_node_t<T, V>', 'fn eval_node_t_memo<T, V>')
method=method.replace('semiring: &T) -> T::Element', 'semiring: &T, cache: &mut HashMap<SDDNodeIndex, T::Element>) -> T::Element')
method=method.replace('''    match &self.sdd_nodes[node_id] {''','''    if let Some(value) = cache.get(&node_id) { return value.clone(); }
    let result = match &self.sdd_nodes[node_id] {''',1)
method=method.replace('self.eval_node_t(prime.clone(), var_assign, semiring)', 'self.eval_node_t_memo(prime.clone(), var_assign, semiring, cache)')
method=method.replace('self.eval_node_t(sub.clone(), var_assign, semiring)', 'self.eval_node_t_memo(sub.clone(), var_assign, semiring, cache)')
old_end='''    }
  }

'''
assert method.endswith(old_end)
method=method[:-len(old_end)]+'''    };
    cache.insert(node_id, result.clone());
    result
  }

'''
s=s[:b]+method+s[b:]
p.write_text(s)
print(p)
