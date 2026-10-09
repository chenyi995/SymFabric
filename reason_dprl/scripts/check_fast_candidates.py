#!/usr/bin/env python3
"""GPU correctness, repeatability and compiled integration for the packing kernel."""
from bench_common import provenance
import torch
from fast_candidates import pack_results
from stable_candidates import pack_results as reference


def check():
    torch.manual_seed(33)
    compiled = torch.compile(pack_results, fullgraph=True)
    torch._dynamo.config.recompile_limit = 64
    cases = [(5, 7, 5, 2, 4, 3, c) for c in (1, 3, 9, 17)]
    cases += [(3, 0, 9, 2, 4, 3, 7), (3, 9, 0, 4, 2, 5, 12),
              (3, 0, 0, 2, 4, 3, 5), (64, 8000, 32, 3, 4, 3, 262)]
    for b, nf, nr, fa, ra, atoms, cap in cases:
        # Deliberately strided sources exercise the generic gather address map.
        facts = torch.randint(0, 1000, (b,nf,fa,6), device='cuda')[:,:,:,::2]
        rules = torch.randint(0, 1000, (b,nr,ra,6), device='cuda')[:,:,:,::2]
        fm = torch.rand(b,nf,device='cuda') > .8
        rm = torch.rand(b,nr,device='cuda') > .5
        fm[0]=False; rm[0]=False
        fm[1]=True; rm[1]=True
        inputs = (facts,fm,rules,rm,cap,atoms,-1)
        expected=reference(*inputs)
        for fn in (pack_results, compiled):
            for _ in range(5):
                actual=fn(*inputs)
                assert all(torch.equal(a,e) for a,e in zip(actual,expected)), cases
        print('PASS', (b,nf,nr,fa,ra,atoms,cap), flush=True)
    return len(cases)


if __name__=='__main__':
    print(provenance())
    print('FAST_CANDIDATES_PASSED', check())
