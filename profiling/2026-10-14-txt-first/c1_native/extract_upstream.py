#!/usr/bin/env python3
"""Recreate vendored Rust adapters from the fixed official Scallop checkout."""
from pathlib import Path
import subprocess,sys
root=Path(sys.argv[1]).resolve()
expected='668bfb6d45ce302fd4ffa7f29916baf3c7ce36ef'
revision=subprocess.check_output(['git','-C',str(root),'rev-parse','HEAD'],text=True).strip()
assert revision==expected,(revision,expected)
p=Path(__file__).resolve().parent
source=root/'core/src/runtime/provenance/common'
for name in ['literal','clause','dnf_formula','dnf_context','disjunction']:
    # git show bypasses diagnostic changes in the worktree.
    rel=f'core/src/runtime/provenance/common/{name}.rs'
    s=subprocess.check_output(['git','-C',str(root),'show',f'{expected}:{rel}'],text=True)
    (p/f'{name}.upstream.rs').write_text(s)
    if name=='dnf_formula':
        s=s.replace('use super::super::*;', 'use super::{Clause, Literal};')
        s=s[:s.index('\nimpl AsBooleanFormula for DNFFormula')]
    elif name=='literal':
        s=s.replace('use super::as_boolean_formula::*;\n','')
        a=s.index('\nimpl AsBooleanFormula for Literal')
        b=s.index('\nimpl std::fmt::Display',a)
        s=s[:a]+s[b:]
    (p/'src'/f'{name}.rs').write_text(s)
