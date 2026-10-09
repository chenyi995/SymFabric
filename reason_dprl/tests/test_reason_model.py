"""Independent numerical and resource checks for the REASON reconstruction."""
import itertools
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts"))
from reason_model import Config, MnistProgram, DotProgram, simulate, simulate_microbatches, watched_literal_bcp


def reference(probabilities):
    batch,length=probabilities.shape[:2]
    out=np.zeros((batch,length*10+2),dtype=np.float64)
    for q in range(batch):
        carry=[1.,0.]
        for t in range(length):
            next_carry=[0.,0.]
            for a,b,c in itertools.product(range(10),range(10),range(2)):
                weight=float(probabilities[q,t,0,a])*float(probabilities[q,t,1,b])*carry[c]
                total=a+b+c
                out[q,t*10+total%10]+=weight
                next_carry[total//10]+=weight
            carry=next_carry
        out[q,-2:]=carry
    return out


@pytest.mark.parametrize("length",[1,4])
def test_probabilities_match_independent_enumeration(length):
    rng=np.random.default_rng(11)
    p=rng.random((3,length,2,10),dtype=np.float32)
    p/=p.sum(-1,keepdims=True)
    program=MnistProgram(length)
    np.testing.assert_allclose(program.execute(p),reference(p),atol=1e-6,rtol=1e-6)


def test_one_hot_carries_and_zero_probabilities():
    p=np.zeros((1,4,2,10),np.float32)
    p[:,:,:,9]=1
    values=MnistProgram(4).execute(p)[0]
    assert values[8]==1
    assert values[19]==1
    assert values[-1]==1
    assert np.count_nonzero(values)==5


def test_shared_vector_dot_products():
    rng=np.random.default_rng(12)
    left=rng.normal(size=(4,17)).astype(np.float32)
    right=rng.normal(size=(4,3,17)).astype(np.float32)
    program=DotProgram(17,3)
    expected=np.einsum("be,bse->bs",left.astype(np.float64),right.astype(np.float64))
    np.testing.assert_allclose(program.execute(left,right),expected,atol=2e-6,rtol=2e-6)
    assert simulate(program,2)["counts"]["operations"]==2*3*(2*17-1)


def test_compiler_partitions_once_and_scheduler_obeys_dependencies():
    p=MnistProgram(2)
    blocks,_=p.compile()
    members=[m for b in blocks for m in b["members"]]
    assert len(members)==len(set(members))==len(p.ops)
    assert all(b["depth"]<=3 and b["ops"]<=7 for b in blocks)
    result=simulate(p,2)
    assert result["counts"]["operations"]==2*len(p.ops)
    assert result["peak_register_words"]<=64*32
    trace={(r["query"],r["root"]):r for r in result["instruction_trace"]}
    for b in blocks:
        for q in range(2):
            r=trace[q,b["root"]]
            for x in b["inputs"]:
                if (q,x) in trace:
                    assert trace[q,x]["finish"]<=r["start"]
    assert result["cycles"]>=max(r["finish"] for r in trace.values())


def test_clock_changes_time_but_not_work():
    p=MnistProgram(1)
    a=simulate(p,1,Config(clock_mhz=500))
    b=simulate(p,1,Config(clock_mhz=250))
    assert a["cycles"]==b["cycles"]
    assert b["execution_us"]==2*a["execution_us"]


def test_query_tiling_preserves_work_and_interface_payload():
    p=MnistProgram(1)
    full=simulate(p,5)
    tiled=simulate_microbatches(p,5,2)
    assert tiled["counts"]["operations"]==full["counts"]["operations"]==5*len(p.ops)
    assert tiled["interface_bytes"]==full["interface_bytes"]
    assert tiled["handoff_us"]==full["handoff_us"]
    assert tiled["cycles"]==2*simulate(p,2)["cycles"]+simulate(p,1)["cycles"]


def scan_bcp(clauses,decisions):
    values={}
    for lit in decisions:
        if abs(lit) in values and values[abs(lit)]!=(lit>0):
            return True,values
        values[abs(lit)]=lit>0
    while True:
        changed=False
        for clause in clauses:
            if any(abs(l) in values and values[abs(l)]==(l>0) for l in clause):
                continue
            unknown=[l for l in clause if abs(l) not in values]
            if not unknown:
                return True,values
            if len(unknown)==1:
                values[abs(unknown[0])]=unknown[0]>0
                changed=True
        if not changed:
            return False,values


def test_watch_replacement_implication_and_conflict():
    cases=[([[1],[-1,2],[-2,3]],[]),([[-1,2],[-2,3],[-3]], [1]),
           ([[1,2,3],[-1],[-2]],[]),([[]],[]),([[1],[-1]],[])]
    for clauses,decisions in cases:
        expected,values=scan_bcp(clauses,decisions)
        actual=watched_literal_bcp(clauses,decisions)
        assert (actual["conflict"] is not None)==expected
        if not expected:
            assert actual["values"]==values


def test_watch_lists_against_random_small_formulas():
    rng=np.random.default_rng(17)
    for _ in range(100):
        clauses=[]
        for _ in range(12):
            variables=rng.choice(np.arange(1,7),size=int(rng.integers(1,5)),replace=False)
            clauses.append([int(v)*int(rng.choice([-1,1])) for v in variables])
        decisions=[int(rng.choice([-1,1]))]
        expected,values=scan_bcp(clauses,decisions)
        actual=watched_literal_bcp(clauses,decisions)
        assert (actual["conflict"] is not None)==expected
        if not expected:
            assert actual["values"]==values
