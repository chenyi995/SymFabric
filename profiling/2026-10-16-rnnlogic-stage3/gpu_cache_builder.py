"""Local GPU int64 cache construction for unique, previously absent keys.

GPU counts never round-trip through CPU. Small per-query row/nonzero totals do
transfer for Python entry binding. The cache retains exactly the stage-2 four
packed storage kinds: candidates/crow/columns int64 and values float32. Counts
convert only after GPU canonical sorting and exact coordinate validation.

This deliberately supports a narrower contract than CPU preload: every input
key must be unique and absent. Existing/repeated keys raise; no checksum is used
as a substitute for exact count comparison. Entries are marked with a sentinel
integer_digest, so the unchanged CPU preloader also rejects existing GPU keys
rather than pretending to verify their original int64 contents. Count semantic
equivalence is verified independently against upstream/native before timing.
"""
from pathlib import Path
import sys
import time
from types import MethodType
import torch

_STAGE2 = Path(__file__).resolve().parent.parent / '2026-10-14-rnnlogic-sparse-stage2'
_STAGE1 = Path(__file__).resolve().parent.parent / '2026-10-14-rnnlogic-strong-baselines'
for directory in [_STAGE1, _STAGE2]:
    if str(directory) not in sys.path:
        sys.path.insert(0,str(directory))
from cached_sparse_predictor import _Entry
from prefix_cache import PrefixGrounder

GPU_DIGEST_SENTINEL = 'not_computed_gpu_unique_absent_keys_only'


def preload_gpu_new(adapter, h_cpu, r_cpu, removed_cpu, terminals, *, namespace=None, rule_version=None, trusted_unique_positive_terminals=False):
    """Validate/canonicalize GPU int64 terminals and upload no count payload.

    Queries/removed-edge ids are CPU int64 as in the original preload API.
    Terminals must be on adapter.device (CUDA) and follow rules_for_relation.
    This function mutates only the adapter's explicit cache after validation.
    trusted_unique_positive_terminals=True skips value/range/coordinate scans
    only for an independently verified source with overflow excluded; shape,
    dtype, device, new-key and unique-key checks always remain enabled.
    """
    adapter._check_version(namespace,rule_version)
    relation = adapter._cpu_queries(h_cpu,r_cpu,removed_cpu)
    device = adapter.device
    if device.type != 'cuda':
        raise ValueError('GPU preload requires CUDA cache/model device')
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    batch = h_cpu.numel(); entities = adapter.model.graph.entity_size
    plan = adapter.plans[relation]; rules = len(plan)
    if bool(((h_cpu < 0) | (h_cpu >= entities)).any()):
        raise ValueError('query head outside graph')
    if len(terminals) != rules:
        raise ValueError('terminal list does not match original relation rule plan')
    heads = h_cpu.tolist()
    removed = [None] * batch if removed_cpu is None else removed_cpu.tolist()
    keys = [adapter._key(relation,head,edge) for head,edge in zip(heads,removed)]
    if len(set(keys)) != len(keys):
        raise ValueError('GPU preload supports unique input query keys only')
    if any(key in adapter.cache for key in keys):
        raise ValueError('GPU preload supports previously absent query keys only; no existing-int64 digest comparison')
    for index,value in terminals:
        if index.dtype != torch.int64 or value.dtype != torch.int64 or index.device != device or value.device != device:
            raise ValueError('GPU terminals must be int64 indices/counts on cache device')
        if index.ndim != 1 or value.shape != index.shape:
            raise ValueError('terminal indices/counts must be matching 1D tensors')
    if batch * entities * max(1,rules) > torch.iinfo(torch.int64).max:
        raise OverflowError('canonical coordinate encoding would exceed int64')
    if terminals:
        flat = torch.cat([index for index,_ in terminals])
        columns = torch.cat([torch.full_like(index,slot) for slot,(index,_) in enumerate(terminals)])
        exact = torch.cat([value for _,value in terminals])
    else:
        flat = torch.empty(0,dtype=torch.int64,device=device)
        columns = torch.empty_like(flat); exact = torch.empty_like(flat)
    if flat.numel() and not trusted_unique_positive_terminals:
        invalid = ((flat < 0) | (flat >= batch * entities) | (exact <= 0)).any()
        if bool(invalid):
            raise ValueError('terminal indices out of range or counts not strictly positive')
    torch.cuda.synchronize(device)
    validation_seconds = time.perf_counter()-start
    canonical_start = time.perf_counter()
    # Sort integer (query, entity, rule) coordinates without reducing counts.
    # Equality of adjacent coordinates detects illegal duplicate per-rule ids.
    encoded = flat * rules + columns
    order = torch.argsort(encoded,stable=True)
    sorted_encoded = encoded[order]
    if not trusted_unique_positive_terminals and sorted_encoded.numel() > 1 and bool((sorted_encoded[1:] == sorted_encoded[:-1]).any()):
        raise ValueError('duplicate terminal index: aggregate paths in exact integer grounding first')
    flat = flat[order]
    packed_columns = columns[order]
    exact = exact[order]
    candidates, row_sizes = torch.unique_consecutive(flat,return_counts=True)
    global_crow = torch.cat([torch.zeros(1,dtype=torch.int64,device=device),row_sizes.cumsum(0)])
    candidate_counts = torch.bincount(candidates // entities,minlength=batch)
    nonzero_counts = torch.bincount(flat // entities,minlength=batch)
    # Only O(batch) metadata transfers to CPU, not candidates, coordinates or counts.
    sizes = torch.stack([candidate_counts,nonzero_counts]).cpu().tolist()
    candidate_sizes, nnz_sizes = sizes
    packed_candidates = candidates.remainder(entities)
    crow_parts = []
    candidate_at = nnz_at = 0
    for count,nnz in zip(candidate_sizes,nnz_sizes):
        crow_parts.append(global_crow[candidate_at:candidate_at+count+1]-nnz_at)
        candidate_at += count; nnz_at += nnz
    packed_crow = torch.cat(crow_parts)
    packed_values = exact.float()
    torch.cuda.synchronize(device)
    canonical_seconds = time.perf_counter()-canonical_start
    bind_start = time.perf_counter()
    candidate_at = crow_at = nnz_at = 0
    for key,count,nnz in zip(keys,candidate_sizes,nnz_sizes):
        candidate = packed_candidates[candidate_at:candidate_at+count]
        crow = packed_crow[crow_at:crow_at+count+1]
        column = packed_columns[nnz_at:nnz_at+nnz]
        value = packed_values[nnz_at:nnz_at+nnz]
        csr = torch.sparse_csr_tensor(crow,column,value,(count,rules),device=device)
        adapter.cache[key] = _Entry(key,candidate,crow,column,value,csr,count,nnz,GPU_DIGEST_SENTINEL)
        candidate_at += count; crow_at += count+1; nnz_at += nnz
    torch.cuda.synchronize(device)
    bind_seconds = time.perf_counter()-bind_start
    return {'namespace':adapter.namespace,'rule_version':adapter.rule_version,
            'rule_fingerprint':adapter.rule_fingerprint,'relation':relation,
            'input_queries':batch,'unique_input_keys':batch,'new_keys':batch,
            'validation_seconds':validation_seconds,'gpu_canonical_csr_seconds':canonical_seconds,
            'gpu_entry_bind_seconds':bind_seconds,'preload_seconds':time.perf_counter()-start,
            'count_build_included':False,'count_payload_D2H_bytes':0,
            'row_nnz_metadata_D2H_bytes':batch*2*8,
            'integer_duplicate_key_check':'unique absent keys required; no integer digest computed',
            'trusted_unique_positive_terminals':bool(trusted_unique_positive_terminals),
            'terminal_coordinate_validation':('externally verified producer contract; dtype/shape/device checked' if trusted_unique_positive_terminals else 'exact int64 range/positivity/uniqueness before float32 conversion'),
            **adapter.cache_statistics()}


def install_gpu_preloader(adapter, *, trusted_unique_positive_terminals=False):
    """Add GPU-terminal dispatch; retain unchanged CPU preload for CPU pairs.

    Empty rule lists dispatch GPU (and have no counts to move). The LRU manager
    can call the regular preload_from_terminals API after this installation.
    """
    if adapter.device.type != 'cuda':
        raise ValueError('GPU preloader installation requires CUDA adapter')
    if getattr(adapter,'_gpu_preloader_installed',False):
        if adapter._gpu_preloader_trusted != bool(trusted_unique_positive_terminals):
            raise ValueError('GPU preloader trust mode cannot change after installation')
        return adapter
    cpu_preload = adapter.preload_from_terminals
    def dispatch(self,h,r,removed,terminals,**kwargs):
        if not terminals or terminals[0][0].device.type == 'cuda':
            return preload_gpu_new(self,h,r,removed,terminals,trusted_unique_positive_terminals=trusted_unique_positive_terminals,**kwargs)
        return cpu_preload(h,r,removed,terminals,**kwargs)
    adapter.preload_from_terminals = MethodType(dispatch,adapter)
    adapter._gpu_preloader_installed = True
    adapter._gpu_preloader_trusted = bool(trusted_unique_positive_terminals)
    return adapter


def gpu_csr_terminals(graph, backend, rule_plan, h, relation, removed=None, *, controller=None):
    """Compute exact sparse GPU terminals via CSR prefix sharing once per batch.

    Query copies, if needed, are the caller's responsibility and must be timed.
    Returns (terminal_list, controller_stats). Dense integer prefix states are
    transient; terminal nonzero extraction/synchronization is included here.
    """
    if h.device != backend.device or h.dtype != torch.int64:
        raise ValueError('GPU count queries must be int64 on backend device')
    if removed is not None and (removed.device != backend.device or removed.dtype != torch.int64 or removed.shape != h.shape):
        raise ValueError('GPU removed edges must match query device/shape and int64 dtype')
    bodies = [body for _,(_,body) in rule_plan]
    if controller is None:
        controller = PrefixGrounder(graph,bodies,policy='last_use',collect_stats=False)
    elif tuple(controller.rule_bodies) != tuple(map(tuple,bodies)):
        raise ValueError('reused GPU controller differs from exact rule plan')
    original_propagate = graph.propagate
    graph.propagate = backend.propagate
    outputs = []
    complete = False
    begun = False
    try:
        controller.begin(h,int(relation),removed)
        begun = True
        for _,(head,body) in rule_plan:
            if int(head) != int(relation):
                raise ValueError('rule head differs from query relation')
            dense = controller.grounding(h,int(head),body,removed)
            rows,columns = torch.nonzero(dense,as_tuple=True)
            outputs.append((rows*graph.entity_size+columns,dense[rows,columns]))
        complete = True
    finally:
        graph.propagate = original_propagate
        if begun:
            stats = controller.finish(require_complete=complete)
    return outputs,stats


class GPUCSRCountBuilder:
    """Reusable static prefix plan; .batch includes all per-query count work.

    Caller's conservative int64 overflow proof and exact source checks must
    precede trusted preloading. Planning/setup is reported separately by callers.
    """
    def __init__(self, graph, backend, rule_plan):
        self.graph=graph;self.backend=backend
        self.plan=tuple((int(i),(int(head),tuple(map(int,body)))) for i,(head,body) in rule_plan)
        self.controller=PrefixGrounder(graph,[body for _,(_,body) in self.plan],policy='last_use',collect_stats=False)

    def batch(self,h,relation,removed=None):
        return gpu_csr_terminals(self.graph,self.backend,self.plan,h,relation,removed,controller=self.controller)


def terminals_to_gpu(terminals, device):
    """Transfer CPU exact-int64 terminal arrays with two bulk H2D copies.

    Concatenation, transfer and reconstruction are explicit caller-timed work.
    Per-rule views preserve original order, duplicate rule columns and empty
    terminals; integer values never convert to float on this path.
    """
    device=torch.device(device)
    if device.type!='cuda':
        raise ValueError('bulk terminal transfer requires CUDA destination')
    for index,value in terminals:
        if index.device.type!='cpu' or value.device.type!='cpu' or index.dtype!=torch.int64 or value.dtype!=torch.int64:
            raise ValueError('bulk terminal transfer requires CPU int64 indices/counts')
        if index.ndim!=1 or value.shape!=index.shape:
            raise ValueError('terminal indices/counts must be matching 1D tensors')
    if not terminals:
        return []
    lengths=[index.numel() for index,_ in terminals]
    flat=torch.cat([index for index,_ in terminals]).to(device)
    values=torch.cat([value for _,value in terminals]).to(device)
    output=[];cursor=0
    for size in lengths:
        output.append((flat[cursor:cursor+size],values[cursor:cursor+size]));cursor+=size
    return output
