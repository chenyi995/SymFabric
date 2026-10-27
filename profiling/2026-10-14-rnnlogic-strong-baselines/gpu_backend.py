"""Local, exact int64 RNNLogic propagation backends; upstream is unchanged.

Atomic: zeros output, then fused edge gather / per-query exclusion / reduction.
CSR: one fused destination-gather kernel, with original edge ids retained for masks.

The backend does not decide when a rule relation equals a query relation. Its caller
must pass removals exactly where upstream grounding does. Graph preparation is
separate from propagate and must be included in cold-start reporting.
"""
from __future__ import annotations

import argparse
import json
import time
from types import SimpleNamespace

import torch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    # Edge counts vary by relation, not by algorithm. Keep E runtime and
    # suppress divisibility/value specialization to avoid one JIT per relation.
    @triton.jit(do_not_specialize=["E"])
    def _edge_scatter(X, SRC, DST, REMOVED, Y, E, B: tl.constexpr,
                      XS0: tl.constexpr, XS1: tl.constexpr,
                      MASKED: tl.constexpr, BLOCK: tl.constexpr):
        offset = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        edge = offset // B
        lane = offset % B
        valid = edge < E
        source = tl.load(SRC + edge, mask=valid, other=0)
        dest = tl.load(DST + edge, mask=valid, other=0)
        if MASKED:
            removed = tl.load(REMOVED + lane, mask=valid, other=-1)
            valid = valid & (edge != removed)
        count = tl.load(X + source * XS0 + lane * XS1, mask=valid, other=0)
        # Omitting zero updates preserves exact integer semantics and reduces
        # unnecessary atomic contention for sparse active counts.
        tl.atomic_add(Y + dest * B + lane, count,
                      mask=valid & (count != 0), sem="relaxed")

    @triton.jit
    def _csr_gather(X, ROWPTR, SRC, EDGE_IDS, REMOVED, Y,
                    N: tl.constexpr, B: tl.constexpr,
                    XS0: tl.constexpr, XS1: tl.constexpr,
                    MASKED: tl.constexpr, ROWS: tl.constexpr,
                    LANES: tl.constexpr):
        row = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        lane = tl.arange(0, LANES)
        start = tl.load(ROWPTR + row, mask=row < N, other=0)
        end = tl.load(ROWPTR + row + 1, mask=row < N, other=0)
        acc = tl.full((ROWS, LANES), 0, tl.int64)
        if MASKED:
            removed = tl.load(REMOVED + lane, mask=lane < B, other=-1)
        # Each destination owns its output, so no atomics or output-zero pass.
        for step in range(tl.max(end - start, axis=0)):
            at = start + step
            active = (row < N) & (at < end)
            source = tl.load(SRC + at, mask=active, other=0)
            valid = active[:, None] & (lane[None, :] < B)
            if MASKED:
                original_edge = tl.load(EDGE_IDS + at, mask=active, other=-1)
                valid = valid & (original_edge[:, None] != removed[None, :])
            count = tl.load(X + source[:, None] * XS0 + lane[None, :] * XS1,
                            mask=valid, other=0)
            acc += count
        tl.store(Y + row[:, None] * B + lane[None, :], acc,
                 mask=(row[:, None] < N) & (lane[None, :] < B))


class GPUBackend:
    """Resident graph and a small fused replacement for graph.propagate.

    Args:
        graph: upstream KnowledgeGraph-like object with entity_size and
            relation2adjacency[r][0] of shape (2, E), destination first.
        device: CUDA device; defaults to the current CUDA device.
        strategy: 'atomic' (edge parallel) or 'csr' (destination gather).
        validate_values: when true, synchronize to validate mask index bounds
            on every call. Keep false in timed runs after input validation.

    Unsupported inputs raise; they never silently switch to a slower backend.
    Counts use int64 throughout and therefore have the upstream int64 range.
    """
    def __init__(self, graph, device=None, strategy="atomic", validate_values=False):
        if triton is None:
            raise RuntimeError("GPUBackend requires Triton; no fallback was run")
        if not torch.cuda.is_available():
            raise RuntimeError("GPUBackend requires a real available CUDA device")
        if strategy not in ("atomic", "csr"):
            raise ValueError("strategy must be atomic or csr")
        self.device = torch.device(device or torch.cuda.current_device())
        if self.device.type != "cuda":
            raise ValueError("GPUBackend only supports CUDA")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.entity_size = int(graph.entity_size)
        self.strategy = strategy
        self.validate_values = bool(validate_values)
        self.jit_metadata = {"edge_count": "runtime_do_not_specialize",
                             "specialized": ["batch", "input_strides", "mask_present", "tile_sizes"],
                             "csr_entity_count": "static_per_dataset",
                             "compilation_in_setup_seconds": False}
        self.adjacency = []
        self.csr = []
        torch.cuda.synchronize(self.device)
        started = time.perf_counter()
        for adj, _weights in graph.relation2adjacency:
            # CPU graph values may already be CUDA resident in a caller's run.
            adj_cpu = adj.detach().to(device="cpu", dtype=torch.int64).contiguous()
            if adj_cpu.ndim != 2 or adj_cpu.shape[0] != 2:
                raise ValueError("expected adjacency shape [2, E]")
            if adj_cpu.numel() and (int(adj_cpu.min()) < 0 or int(adj_cpu.max()) >= self.entity_size):
                raise ValueError("adjacency entity index outside graph")
            dst, src = adj_cpu[0], adj_cpu[1]
            self.adjacency.append((src.to(self.device), dst.to(self.device)))
            if strategy == "csr":
                order = torch.argsort(dst, stable=True)
                degree = torch.bincount(dst, minlength=self.entity_size)
                ptr = torch.cat((torch.zeros(1, dtype=torch.int64), degree.cumsum(0)))
                self.csr.append((ptr.to(self.device), src[order].to(self.device), order.to(self.device)))
        torch.cuda.synchronize(self.device)
        self.setup_seconds = time.perf_counter() - started
        self.resident_bytes = sum(t.numel() * t.element_size() for pair in self.adjacency for t in pair)
        self.resident_bytes += sum(t.numel() * t.element_size() for triple in self.csr for t in triple)

    @torch.no_grad()
    def propagate(self, x, relation, edges_to_remove=None):
        if x.device != self.device or x.dtype != torch.int64:
            raise TypeError(f"expected int64 input on {self.device}, got {x.dtype} on {x.device}")
        if x.ndim != 3 or x.shape[0] != self.entity_size or x.shape[2] != 1:
            raise ValueError("only shape [entity_size, batch, 1] is supported")
        if x.stride(0) < 0 or x.stride(1) < 0:
            raise ValueError("negative strides are unsupported")
        if not isinstance(relation, int):
            raise TypeError("relation must be a Python int (avoid hidden device synchronization)")
        if relation < 0 or relation >= len(self.adjacency):
            raise IndexError("relation outside graph")
        src, dst = self.adjacency[relation]
        edge_count, batch = src.numel(), x.shape[1]
        masked = edges_to_remove is not None
        if masked:
            if edges_to_remove.device != self.device or edges_to_remove.dtype != torch.int64:
                raise TypeError("edges_to_remove must be int64 on the input CUDA device")
            if edges_to_remove.ndim != 1 or edges_to_remove.numel() != batch or not edges_to_remove.is_contiguous():
                raise ValueError("edges_to_remove must be contiguous shape [batch]")
            if self.validate_values and batch:
                if int(edges_to_remove.min()) < 0 or int(edges_to_remove.max()) >= edge_count:
                    raise IndexError("removed edge id outside relation")
        if batch == 0:
            return torch.empty_like(x, memory_format=torch.contiguous_format)
        output = torch.empty((self.entity_size, batch, 1), device=self.device, dtype=torch.int64)
        dummy = edges_to_remove if masked else x
        with torch.cuda.device(self.device):
            if self.strategy == "atomic":
                output.zero_()
                if edge_count:
                    _edge_scatter[(triton.cdiv(edge_count * batch, 256),)](
                        x, src, dst, dummy, output, edge_count, batch,
                        x.stride(0), x.stride(1), masked, 256)
            else:
                ptr, csr_src, original_ids = self.csr[relation]
                _csr_gather[(triton.cdiv(self.entity_size, 4),)](
                    x, ptr, csr_src, original_ids, dummy, output,
                    self.entity_size, batch, x.stride(0), x.stride(1), masked,
                    4, triton.next_power_of_2(batch), num_warps=4)
        return output


def _reference(x, adj, removed):
    dst, src = adj.to(x.device)
    message = x[src].clone()
    if removed is not None:
        message[removed, torch.arange(x.shape[1], device=x.device)] = 0
    out = torch.zeros_like(x, memory_format=torch.contiguous_format)
    out.index_add_(0, dst, message)
    return out


def self_test():
    """Exact real-GPU checks, including noncontiguous one-hot and large counts."""
    torch.manual_seed(789)
    records = []
    for strategy in ("atomic", "csr"):
        # Parallel edges intentionally exercise multiplicity; high in-degree
        # and empty relations exercise reduction and no-edge behavior.
        n = 17
        edges = torch.tensor([[1, 1, 1, 2, 2, 16, 0, 1], [0, 0, 2, 1, 2, 16, 1, 16]])
        empty = torch.empty((2, 0), dtype=torch.int64)
        graph = SimpleNamespace(entity_size=n, relation2adjacency=[
            [edges, torch.ones(edges.shape[1])], [empty, torch.empty(0)]])
        backend = GPUBackend(graph, strategy=strategy, validate_values=True)
        for batch in (1, 3, 8, 32, 65):
            heads = torch.arange(batch, device=backend.device) % n
            onehot = torch.nn.functional.one_hot(heads, n).T.unsqueeze(-1)
            dense = torch.randint(0, 100, (n, batch, 1), dtype=torch.int64, device=backend.device)
            # Above float32 and int32 exact range, comfortably below int64 overflow.
            large = dense * (1 << 36)
            removed = torch.arange(batch, device=backend.device) % edges.shape[1]
            for label, initial in (("noncontiguous_onehot", onehot), ("dense", dense), ("large_int64", large)):
                for mask in (None, removed):
                    actual, expected = initial, initial
                    for _ in range(3):
                        actual = backend.propagate(actual, 0, mask)
                        expected = _reference(expected, edges, mask)
                        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
                    records.append({"strategy":strategy, "batch":batch, "input":label, "masked":mask is not None, "steps":3})
                torch.testing.assert_close(backend.propagate(initial, 1), torch.zeros_like(initial), rtol=0, atol=0)
    # Changing only relation edge count must reuse the same atomic JIT variant.
    # This prevents a real graph with hundreds of relations paying hundreds of
    # compilations for otherwise identical propagation kernels.
    before = sum(len(v) for v in _edge_scatter.cache.values())
    dynamic_edge_checks = []
    x = torch.arange(17 * 32, device="cuda", dtype=torch.int64).reshape(17, 32, 1)
    for edge_count in (1, 7, 16, 33):
        source = torch.arange(edge_count) % 17
        dest = (source * 3 + 1) % 17
        adj = torch.stack((dest, source))
        graph = SimpleNamespace(entity_size=17, relation2adjacency=[[adj, torch.ones(edge_count)]])
        backend = GPUBackend(graph, strategy="atomic", validate_values=True)
        actual = backend.propagate(x, 0)
        expected = _reference(x, adj, None)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        dynamic_edge_checks.append(edge_count)
    after = sum(len(v) for v in _edge_scatter.cache.values())
    assert after == before, ("edge count triggered extra specialization", before, after)
    return {"device":torch.cuda.get_device_name(), "torch":torch.__version__, "triton":triton.__version__,
            "passed":len(records), "cases":records,
            "dynamic_edge_counts_tested":dynamic_edge_checks,
            "extra_jit_variants_for_dynamic_edges":after-before}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--output")
    args = parser.parse_args()
    if not args.self_test:
        parser.error("specify --self-test")
    result = self_test()
    if args.output:
        with open(args.output, "w") as f:
            json.dump(result, f, indent=2)
    print(json.dumps(result), flush=True)
