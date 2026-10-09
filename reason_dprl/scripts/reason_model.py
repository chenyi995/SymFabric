"""DPU-v2-backed REASON reconstruction, NOT the authors' simulator.

The public GraphClass and binary tree builder are used without modification.
The scheduler below adds the REASON configuration, finite banked registers,
tree blocks, spill traffic, dependencies, issue limits and interface accounting.
It is a cycle-based model; its timing has not been calibrated against REASON RTL.
"""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
import itertools
import math
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

ROOT = Path(os.environ.get("SYMFABRIC_REPO_ROOT",Path(__file__).resolve().parents[2])).resolve()
DPU = ROOT / "systolicarch/upstream/DAG_Processor"
DPU_PIN = "cefe56ec57a9794e1ae2263f2998da65f1e93340"
sys.path.insert(0, str(DPU))
from src.common_classes import GraphClass
import networkx as nx


@dataclass(frozen=True)
class Config:
    # source: REASON Table III and Sec. V-F.
    trees: int = 12
    nodes: int = 80
    depth: int = 3
    banks: int = 64
    registers: int = 32
    sram_bytes: int = 1_250_000  # Table III 1.25 MB, interpreted as decimal MB.
    # source: Table III footnote, scaled 12/8 nm rows ONLY; not the 28 nm clock.
    clock_mhz: float = 500.0
    # Remaining quantities are explicit reconstruction assumptions, swept below.
    word_bytes: int = 4
    read_ports: int = 1
    write_ports: int = 1
    sram_latency: int = 2
    dram_latency: int = 100
    external_bytes_per_cycle: float = 64.0
    issue_trees_per_cycle: int = 12
    instruction_bytes: int = 16
    output_fifo_entries: int = 16
    handoff_us: float = 1.0


def provenance():
    commit = subprocess.check_output(["git", "-C", str(DPU), "rev-parse", "HEAD"], text=True).strip()
    if commit != DPU_PIN:
        raise RuntimeError(f"DPU-v2 revision mismatch: {commit}")
    return {"base_repository": "https://github.com/nimish15shah/DAG_Processor", "base_commit": commit,
            "reused_code": ["src/common_classes.py:GraphClass", "GraphClass.create_tree_of_nodes"],
            "paper": "https://arxiv.org/html/2601.20784v1", "timing_validated_against_rtl": False,
            "implemented_path": "binary sum-product DAGs; all other DPrL operators explicitly fall back",
            "parameter_sources": {"trees,nodes,sram_bytes": "Table III",
                "depth,banks,registers": "Sec. V-F", "clock_mhz": "scaled 12/8 nm Table III footnote",
                "all_other_fields": "assumptions; not provided by the paper"},
            "limitations": ["No Accel-Sim coupling or shared-L2 contention model",
                "No proprietary VCS/REASON RTL validation", "No first-order unification support",
                "No accuracy-changing pruning", "Single serial query-batch latency, no claimed overlap",
                "Binary float32 functional arithmetic; datapath precision is a reconstruction choice",
                "80 nodes are an aggregate issue limit across 12 depth-3 trees; exact physical packing is unknown"]}


class MnistProgram:
    """Lower the shipped DPrL PI marginal recurrence to a static arithmetic DAG.

    This preserves upstream's unconditional carry recurrence. It does NOT change
    that recurrence into a target-conditioned exact sum-probability algorithm.
    Log/exp conversion and target gathering remain GPU work.
    """
    def __init__(self, length):
        start = time.perf_counter()
        self.length = length
        self.builder = GraphClass(id_iter=itertools.count())
        g = self.builder
        self.inputs = []
        c = [g.create_real_valued_leaf_node(float(x)) for x in (1, 0)]
        self.constants = dict(zip(c, (1.0, 0.0)))
        self.digit_outputs = []
        def tree(nodes):
            return g.create_tree_of_nodes(nodes, "sum", tree_nodes=set(), curr_graph_nx=nx.DiGraph())
        for t in range(length):
            p = [[g.create_real_valued_leaf_node() for _ in range(10)] for _ in range(2)]
            self.inputs.extend((k, t, n, d) for n in range(2) for d,k in enumerate(p[n]))
            digits, carries = [[] for _ in range(10)], [[], []]
            for a in range(10):
                for b in range(10):
                    pair = g.create_2input_node(p[0][a], p[1][b], "prod")
                    for cin in (0,1):
                        joint = g.create_2input_node(pair, c[cin], "prod")
                        total = a+b+cin
                        digits[total % 10].append(joint)
                        carries[total // 10].append(joint)
            self.digit_outputs.append([tree(x) for x in digits])
            c = [tree(x) for x in carries]
        self.carry_outputs = c
        self.outputs = [x for group in self.digit_outputs for x in group] + c
        self.graph = g.graph
        self.ops = [(k, "sum" if v.is_sum() else "prod", tuple(v.child_key_list))
                    for k,v in self.graph.items() if not v.is_leaf()]
        self.leaves = [k for k,v in self.graph.items() if v.is_leaf()]
        self.lowering_s = time.perf_counter() - start

    def execute(self, probabilities):
        """Functional FP32 evaluation; CPU interpreter time is not chip latency."""
        probabilities = np.asarray(probabilities, dtype=np.float32)
        batch = probabilities.shape[0]
        values = {k: np.full(batch, v, np.float32) for k,v in self.constants.items()}
        for k,t,n,d in self.inputs:
            values[k] = probabilities[:,t,n,d]
        for k,op,ch in self.ops:
            if op == "sum":
                values[k] = values[ch[0]] + values[ch[1]]
            else:
                values[k] = values[ch[0]] * values[ch[1]]
        return np.stack([values[k] for k in self.outputs], axis=-1)

    def compile(self, depth=3):
        start = time.perf_counter()
        users = defaultdict(list)
        for k,op,ch in self.ops:
            for x in ch:
                users[x].append(k)
        op_by_id = {k:(op,ch) for k,op,ch in self.ops}
        outputs, assigned, blocks = set(self.outputs), set(), []
        for root,_,_ in reversed(self.ops):
            if root in assigned:
                continue
            members = []
            def walk(k, level):
                members.append(k)
                for child in op_by_id[k][1]:
                    if level < depth and child in op_by_id and child not in assigned and len(users[child]) == 1 and child not in outputs:
                        walk(child, level+1)
            walk(root, 1)
            member_set = set(members)
            inputs = sorted(set(x for k in members for x in op_by_id[k][1] if x not in member_set))
            heights = {}
            for k in sorted(members):
                heights[k] = 1+max((heights.get(x,0) for x in op_by_id[k][1]), default=0)
            blocks.append({"root": root, "members": sorted(members), "inputs": inputs,
                           "depth": heights[root], "ops": len(members)})
            assigned.update(members)
        assert assigned == set(op_by_id)
        blocks.sort(key=lambda b: b["root"])
        level = {k:0 for k in self.leaves}
        for b in blocks:
            level[b["root"]] = 1 + max(level[k] for k in b["inputs"])
            b["level"] = level[b["root"]]
        return blocks, time.perf_counter()-start


class DotProgram(MnistProgram):
    """Shared query vector times candidate matrix, REASON Sec. V-B SpMSpM."""
    def __init__(self, width, candidates):
        start=time.perf_counter()
        self.width,self.candidates=width,candidates
        g=GraphClass(id_iter=itertools.count())
        left=[g.create_real_valued_leaf_node() for _ in range(width)]
        self.inputs=[(k,"left",i) for i,k in enumerate(left)]
        self.outputs=[]
        for s in range(candidates):
            right=[g.create_real_valued_leaf_node() for _ in range(width)]
            self.inputs.extend((k,s,i) for i,k in enumerate(right))
            products=[g.create_2input_node(a,b,"prod") for a,b in zip(left,right)]
            root=g.create_tree_of_nodes(products,"sum",tree_nodes=set(),curr_graph_nx=nx.DiGraph())
            self.outputs.append(root)
        self.graph=g.graph
        self.ops=[(k,"sum" if v.is_sum() else "prod",tuple(v.child_key_list))
                  for k,v in self.graph.items() if not v.is_leaf()]
        self.leaves=[k for k,v in self.graph.items() if v.is_leaf()]
        self.lowering_s=time.perf_counter()-start

    def execute(self,left,right):
        values={}
        for k,side,i in self.inputs:
            values[k]=np.asarray(left[:,i] if side=="left" else right[:,side,i],dtype=np.float32)
        for k,op,ch in self.ops:
            values[k]=values[ch[0]]+values[ch[1]] if op=="sum" else values[ch[0]]*values[ch[1]]
        return np.stack([values[k] for k in self.outputs],axis=-1)


def simulate_tiled_dot(width,candidates,batch,tile=8,config=Config(),query_tile=None):
    """One handoff for all vectors, serial SRAM-sized tiles inside REASON.

    Conservative implementation: register allocation restarts between tiles;
    shared query vectors may be reloaded internally. GPU/interface payload is
    counted once. This is a concrete tiled mapping, not an optimality bound.
    """
    whole,tail=divmod(candidates,tile)
    parts=[]
    for n,repeats in ((tile,whole),(tail,int(tail>0))):
        if not repeats:
            continue
        p=DotProgram(width,n)
        m=simulate(p,batch,config) if query_tile is None else simulate_microbatches(p,batch,query_tile,config)
        parts.append({"tile_candidates":n,"repetitions":repeats,"model":m})
    execution=sum(x["repetitions"]*x["model"]["execution_us"] for x in parts)
    payload=batch*(width*(1+candidates)+candidates)*config.word_bytes
    handoff=2*config.handoff_us+math.ceil(payload/config.external_bytes_per_cycle)/config.clock_mhz
    return {"parts":parts,"query_tile":query_tile or batch,"execution_us":execution,"interface_bytes":payload,
            "handoff_us":handoff,"warm_accelerator_us":execution+handoff,
            "mapping_s":sum(x["model"]["mapping_s"]+x["model"]["lowering_s"] for x in parts),
            "configuration_us":sum(x["model"]["configuration_us"] for x in parts),
            "evidence":"uncalibrated cycle-based tiled SpMSpM reconstruction"}


def simulate(program, batch, config=Config()):
    """Conservative static schedule, with disjoint issue-delay attribution.

    Interleave queries at each dependency level. Each issued block owns its
    inputs until read, writes a fixed PE output bank, and has a depth-dependent
    ready time. Registers use the lowest free address. Spill/reload costs are
    charged at the shared memory front end; delays are not summed with makespan.
    """
    c = config
    if min(c.trees,c.nodes,c.depth,c.banks,c.registers,c.read_ports,c.write_ports,c.output_fifo_entries) < 1:
        raise ValueError("Hardware capacities must be positive")
    blocks, compile_s = program.compile(c.depth)
    blocks = sorted(blocks, key=lambda b: (b["level"], b["root"]))
    remaining = Counter()
    for q in range(batch):
        for b in blocks:
            remaining.update((q,x) for x in b["inputs"])
        remaining.update((q,x) for x in program.outputs)
    rf = {}
    free = [set(range(c.registers)) for _ in range(c.banks)]
    ready = {(q,k):0 for q in range(batch) for k in program.leaves}
    bank_of = {(q,k):(q*len(program.leaves)+i) % c.banks
               for q in range(batch) for i,k in enumerate(program.leaves)}
    # Every input begins in shared SRAM after the separately modeled handoff.
    stored = set(ready)
    bank_read = [0] * c.banks
    bank_write = [0] * c.banks
    pe_issue = [0] * c.trees
    pe_ops = [0] * c.trees
    issue_counts, node_counts = Counter(), Counter()
    pending_outputs = []
    stalls = Counter()
    count = Counter()
    clock = 0
    peak_rf, peak_sram = 0, len(stored)*c.word_bytes
    trace = []

    def move_cost(words):
        nonlocal peak_sram
        peak_sram = max(peak_sram, len(stored)*c.word_bytes)
        overflow = len(stored)*c.word_bytes > c.sram_bytes
        if overflow:
            count["dram_bytes"] += words*c.word_bytes
            return c.dram_latency + math.ceil(words*c.word_bytes/c.external_bytes_per_cycle)
        count["sram_bytes"] += words*c.word_bytes
        return c.sram_latency + math.ceil(words/c.banks)

    def allocate(key, bank, protected):
        nonlocal clock
        if key in rf:
            return
        if not free[bank]:
            victims = [x for x,(b,a) in rf.items() if b == bank and x not in protected]
            if not victims:
                raise RuntimeError("Block needs more registers in one bank than configured")
            victim = min(victims, key=lambda x:(ready[x], x))
            wait = max(0, ready[victim]-clock)
            stalls["register_pending"] += wait
            clock += wait
            _,addr = rf.pop(victim)
            if victim not in stored and remaining[victim]:
                stored.add(victim)
                delay = move_cost(1)
                stalls["spill"] += delay
                clock += delay
                count["spills"] += 1
            free[bank].add(addr)
        address = min(free[bank])
        free[bank].remove(address)
        rf[key] = (bank,address)

    for block in blocks:
        for q in range(batch):
            inputs = [(q,x) for x in block["inputs"]]
            dep_ready = max(ready[x] for x in inputs)
            wait = max(0, dep_ready-clock)
            stalls["dependency"] += wait
            clock += wait
            if len(pending_outputs) >= c.output_fifo_entries:
                pending_outputs = [x for x in pending_outputs if x > clock]
                if len(pending_outputs) >= c.output_fifo_entries:
                    wait = min(pending_outputs)-clock
                    stalls["output_fifo"] += wait
                    clock += wait
                    pending_outputs = [x for x in pending_outputs if x > clock]
            protected = set(inputs)
            loads = 0
            for key in inputs:
                if key not in rf:
                    allocate(key, bank_of[key], protected)
                    loads += 1
            if loads:
                delay = move_cost(loads)
                clock += delay
                stalls["operand_load"] += delay
                count["loads"] += loads
            reads = Counter(rf[x][0] for x in inputs)
            read_start = max([clock] + [bank_read[b] for b in reads])
            read_cycles = max(math.ceil(n/c.read_ports) for n in reads.values())
            stalls["bank_read"] += read_start-clock+read_cycles-1
            clock = read_start+read_cycles-1
            for bank,n in reads.items():
                bank_read[bank] = read_start+math.ceil(n/c.read_ports)
            pe = min(range(c.trees), key=lambda p: max(pe_issue[p],bank_write[p % c.banks]))
            start = max(clock, pe_issue[pe])
            while issue_counts[start] >= c.issue_trees_per_cycle or node_counts[start]+block["ops"] > c.nodes:
                start += 1
            stalls["instruction_issue"] += start-clock
            clock = start
            key, out_bank = (q,block["root"]), pe % c.banks
            allocate(key,out_bank,protected)
            start = clock
            # Allocation stalls can move an instruction into another occupied slot.
            while issue_counts[start] >= c.issue_trees_per_cycle or node_counts[start]+block["ops"] > c.nodes:
                start += 1
                stalls["instruction_issue"] += 1
            clock = start
            finish = start+block["depth"]+1
            write_time = max(finish,bank_write[out_bank])
            stalls["bank_write"] += write_time-finish
            ready[key] = write_time+1
            bank_of[key] = out_bank
            bank_write[out_bank] = write_time+math.ceil(1/c.write_ports)
            pe_issue[pe] = start+1
            pe_ops[pe] += block["ops"]
            issue_counts[start] += 1
            node_counts[start] += block["ops"]
            pending_outputs.append(ready[key])
            count["instructions"] += 1
            count["operations"] += block["ops"]
            peak_rf = max(peak_rf,len(rf))
            if batch <= 2:
                trace.append({"query":q,"root":key[1],"start":start,"finish":ready[key],"pe":pe,
                              "output_bank":out_bank,"output_register":rf[key][1]})
            for x in inputs:
                remaining[x] -= 1
                if remaining[x] == 0:
                    if x in rf:
                        b,a = rf.pop(x)
                        free[b].add(a)
                    stored.discard(x)
    cycles = max(ready[(q,k)] for q in range(batch) for k in program.outputs)
    code_bytes = len(blocks)*c.instruction_bytes
    configuration_cycles = math.ceil(code_bytes/c.external_bytes_per_cycle)+c.sram_latency
    payload_bytes = batch*(len(program.inputs)+len(program.outputs))*c.word_bytes
    transfer_cycles = math.ceil(payload_bytes/c.external_bytes_per_cycle)
    execute_us = cycles/c.clock_mhz
    handoff_us = 2*c.handoff_us+transfer_cycles/c.clock_mhz
    return {"evidence":"simulated, uncalibrated REASON reconstruction", "config":asdict(c),
            "batch":batch,"cycles":cycles,"execution_us":execute_us,"handoff_us":handoff_us,
            "configuration_us":configuration_cycles/c.clock_mhz,
            "warm_accelerator_us":execute_us+handoff_us,
            "cold_accelerator_us":execute_us+handoff_us+configuration_cycles/c.clock_mhz,
            "lowering_s":program.lowering_s,"mapping_s":compile_s,
            "binary_template_bytes":code_bytes,"interface_bytes":payload_bytes,
            "counts":dict(count),"issue_delay_cycles":dict(stalls),
            "peak_register_words":peak_rf,"peak_sram_bytes":peak_sram,
            "node_utilization":count["operations"]/(cycles*c.nodes),"per_pe_operations":pe_ops,
            "instruction_trace":trace,"block_count":len(blocks)}


def simulate_microbatches(program,batch,tile,config=Config()):
    """Serialize bounded groups of independent queries inside one invocation.

    No queries or arithmetic are pruned. The interface payload and two handoffs
    remain for the complete batch; only register live ranges change. This is a
    mapping sensitivity case, not extra hardware or a throughput overlap claim.
    """
    if tile<1 or batch<1:
        raise ValueError("Batch and query tile must be positive")
    whole,tail=divmod(batch,tile)
    parts=[(n,repeats,simulate(program,n,config)) for n,repeats in
           ((tile,whole),(tail,int(tail>0))) if repeats]
    total=dict(parts[0][2])
    for field in ("cycles","execution_us"):
        total[field]=sum(reps*m[field] for _,reps,m in parts)
    for field in ("counts","issue_delay_cycles"):
        total[field]=dict(sum((Counter({k:reps*v for k,v in m[field].items()}) for _,reps,m in parts),Counter()))
    for field in ("peak_register_words","peak_sram_bytes"):
        total[field]=max(m[field] for _,_,m in parts)
    total["interface_bytes"]=batch*(len(program.inputs)+len(program.outputs))*config.word_bytes
    total["handoff_us"]=2*config.handoff_us+math.ceil(total["interface_bytes"]/config.external_bytes_per_cycle)/config.clock_mhz
    total["warm_accelerator_us"]=total["execution_us"]+total["handoff_us"]
    total["cold_accelerator_us"]=total["warm_accelerator_us"]+total["configuration_us"]
    total["node_utilization"]=total["counts"]["operations"]/(total["cycles"]*config.nodes)
    total["per_pe_operations"]=[sum(reps*m["per_pe_operations"][i] for _,reps,m in parts) for i in range(config.trees)]
    total.update(batch=batch,query_tile=tile,instruction_trace=[],
        mapping_s=sum(m["mapping_s"] for _,_,m in parts),
        mapping_scope="serial independent-query microbatches, one complete-batch handoff")
    return total


def sensitivity(program,batch):
    base=Config()
    variants={"nominal":base,"clock250":replace(base,clock_mhz=250),
              "clock1000":replace(base,clock_mhz=1000),
              "handoff10us":replace(base,handoff_us=10),
              "registers16":replace(base,registers=16),
              "sram_latency8":replace(base,sram_latency=8),
              "read_ports2":replace(base,read_ports=2),
              "link16Bcycle":replace(base,external_bytes_per_cycle=16)}
    return {k:simulate(program,batch,v) for k,v in variants.items()}


def watched_literal_bcp(clauses,decisions=()):
    """Functional WL/FIFO reference for Sec. V-D/E, with conflict flush.

    No timing claims: this validates SAT control semantics only; no DPrL
    first-order operations are mapped to it. CDCL search remains a fallback.
    """
    clauses=[tuple(c) for c in clauses]
    watches, watch_lists, values = {}, defaultdict(set), {}
    fifo=deque(decisions)
    visits=0
    for i,clause in enumerate(clauses):
        if not clause:
            return {"conflict":i,"values":{},"clause_visits":0,"fifo_flushed":len(fifo)}
        watches[i]=[0,min(1,len(clause)-1)]
        for pos in set(watches[i]):
            watch_lists[clause[pos]].add(i)
        if len(clause)==1:
            fifo.append(clause[0])
    def value(lit):
        return None if abs(lit) not in values else values[abs(lit)] == (lit>0)
    while fifo:
        lit=fifo.popleft()
        if value(lit) is False:
            return {"conflict":"contradictory implication","values":values,
                    "clause_visits":visits,"fifo_flushed":len(fifo)}
        if value(lit) is True:
            continue
        values[abs(lit)]=lit>0
        for cid in sorted(watch_lists[-lit].copy()):
            visits+=1
            clause=clauses[cid]
            pair=watches[cid]
            side=0 if clause[pair[0]]==-lit else 1
            other=pair[1-side]
            if value(clause[other]) is True:
                continue
            replacement=next((i for i,x in enumerate(clause) if i not in pair and value(x) is not False),None)
            if replacement is not None:
                watch_lists[-lit].remove(cid)
                pair[side]=replacement
                watch_lists[clause[replacement]].add(cid)
            elif value(clause[other]) is False:
                return {"conflict":cid,"values":values,"clause_visits":visits,"fifo_flushed":len(fifo)}
            else:
                fifo.append(clause[other])
    return {"conflict":None,"values":values,"clause_visits":visits,"fifo_flushed":0}
