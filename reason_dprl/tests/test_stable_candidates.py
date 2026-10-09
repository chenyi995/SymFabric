"""Check compaction ordering and capacity boundaries with an independent oracle."""
from pathlib import Path
import sys
import pytest
import torch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from stable_candidates import pack_results


@pytest.mark.parametrize("capacity",[1,3,9,17])
def test_matches_ordered_python_compaction(capacity):
    generator=torch.Generator().manual_seed(17)
    facts=torch.randint(0,100,(5,7,2,3),generator=generator)
    rules=torch.randint(0,100,(5,5,4,3),generator=generator)
    fm=torch.rand((5,7),generator=generator)>.4
    rm=torch.rand((5,5),generator=generator)>.3
    fm[0]=False;rm[0]=False
    fm[1]=True;rm[1]=True
    actual,counts=pack_results(facts,fm,rules,rm,capacity,3,-1)
    expected=torch.full((5,capacity,3,3),-1,dtype=torch.long)
    for b in range(5):
        valid=[]
        for group,mask in ((facts,fm),(rules,rm)):
            for k in range(group.shape[1]):
                if mask[b,k]:
                    row=torch.full((3,3),-1,dtype=torch.long)
                    n=min(3,group.shape[2]);row[:n]=group[b,k,:n]
                    valid.append(row)
        kept=valid[:capacity]
        if kept:
            expected[b,:len(kept)]=torch.stack(kept)
        assert counts[b]==len(kept)
    assert torch.equal(actual,expected)
