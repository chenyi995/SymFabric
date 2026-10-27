"""Local native CPU int64 propagation; upstream RNNLogic is not modified.

Usage::
    backend = CPUBackend(graph, threads=8)
    graph.propagate = backend.propagate

The native kernel fuses source gather, per-query edge exclusion, and destination
reduction. CSR rows own their output, eliminating reduction atomics and the
E*B*D gathered message tensor. Setup and compilation are not part of propagate
latency; metadata reports them explicitly. No PyTorch thread settings change.
"""
from __future__ import annotations

import ctypes
import fcntl
import hashlib
import os
from pathlib import Path
import subprocess
import time

import numpy as np
import torch


_I64_PTR = ctypes.POINTER(ctypes.c_int64)


def _array_ptr(array):
    return array.ctypes.data_as(_I64_PTR)


def _tensor_ptr(tensor):
    return ctypes.cast(tensor.data_ptr(), _I64_PTR)


def _load_native():
    source = Path(__file__).with_name("cpu_native.cpp")
    compiler = os.environ.get("CXX", "g++")
    flags = ["-std=c++17", "-O3", "-fPIC", "-shared", "-fopenmp"]
    # Do not use -march=native: build hosts and Slurm nodes may differ.
    digest = hashlib.sha256(source.read_bytes() + repr((compiler, flags)).encode()).hexdigest()[:16]
    library_path = source.with_name(f"cpu_native_{digest}.so")
    started = time.perf_counter()
    compiled = False
    with open(source.with_name("cpu_native_build.lock"), "w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not library_path.exists():
            temporary = library_path.with_suffix(f".{os.getpid()}.so.tmp")
            try:
                subprocess.run([compiler, *flags, str(source), "-o", str(temporary)],
                               check=True, capture_output=True, text=True)
                temporary.replace(library_path)
            except subprocess.CalledProcessError as error:
                raise RuntimeError(f"CPU native compilation failed:\n{error.stderr}") from error
            finally:
                temporary.unlink(missing_ok=True)
            compiled = True
    lib = ctypes.CDLL(str(library_path))
    fn = lib.rnnlogic_propagate_i64
    fn.argtypes = [_I64_PTR] * 6 + [ctypes.c_int64] * 7 + [ctypes.c_int]
    fn.restype = ctypes.c_int
    lib.rnnlogic_openmp_max_threads.restype = ctypes.c_int
    return lib, {
        "library": str(library_path),
        "compiler": compiler,
        "compile_flags": flags,
        "compiled_this_process": compiled,
        "build_load_seconds": time.perf_counter() - started,
    }


class CPUBackend:
    """Destination CSR propagation with explicit per-call OpenMP thread count.

    Input: CPU torch.int64 [N, B, D], including noncontiguous input strides.
    Removal: optional CPU int64 [B], indexing the original relation edge list.
    Output: contiguous CPU int64 [N, B, D].

    Like upstream int64 scatter, reductions have fixed-width wraparound. The
    caller is responsible for checking that application path counts fit int64.
    """

    def __init__(self, graph, threads=1):
        started = time.perf_counter()
        self.set_threads(threads)
        self.entity_size = int(graph.entity_size)
        self._lib, build_info = _load_native()
        self._propagate = self._lib.rnnlogic_propagate_i64
        prep_started = time.perf_counter()
        self._relations = []
        total_bytes = total_edges = 0
        for entry in graph.relation2adjacency:
            index = entry[0]
            if isinstance(index, torch.Tensor):
                index = index.detach().cpu().numpy()
            index = np.asarray(index, dtype=np.int64)
            if index.ndim != 2 or index.shape[0] != 2:
                raise ValueError("relation adjacency must have shape [2, E]")
            destinations, sources = index
            if index.size and (index.min() < 0 or index.max() >= self.entity_size):
                raise ValueError("adjacency contains an out-of-range entity")
            # Stable sorting preserves original order within each destination.
            order = np.argsort(destinations, kind="stable").astype(np.int64, copy=False)
            rowptr = np.empty(self.entity_size + 1, dtype=np.int64)
            rowptr[0] = 0
            np.cumsum(np.bincount(destinations, minlength=self.entity_size), out=rowptr[1:])
            sorted_sources = np.ascontiguousarray(sources[order])
            original_edge_ids = np.ascontiguousarray(order)
            edge_count = int(sources.size)
            # Keep arrays alive; retain pointers to avoid rebuilding ctypes
            # wrappers on every rule step in the measured critical path.
            self._relations.append((rowptr, sorted_sources, original_edge_ids,
                                    _array_ptr(rowptr), _array_ptr(sorted_sources),
                                    _array_ptr(original_edge_ids), edge_count))
            total_bytes += rowptr.nbytes + sorted_sources.nbytes + original_edge_ids.nbytes
            total_edges += edge_count
        self.setup_seconds = time.perf_counter() - started
        self.metadata = {
            "backend": "native_openmp_destination_csr_int64",
            "threads": self.threads,
            "nodes": self.entity_size,
            "relations": len(self._relations),
            "edges": total_edges,
            "prepared_graph_bytes": total_bytes,
            "graph_prepare_seconds": time.perf_counter() - prep_started,
            "setup_seconds": self.setup_seconds,
            "input_strides_supported": True,
            "preserves_original_edge_ids": True,
            "accumulation": "int64 two's-complement modular addition; no overflow detection",
            "schedule": "OpenMP dynamic chunks of 32 destination rows",
            **build_info,
        }

    def set_threads(self, threads):
        if isinstance(threads, bool) or int(threads) != threads or int(threads) < 1:
            raise ValueError("threads must be a positive integer")
        self.threads = int(threads)
        if hasattr(self, "metadata"):
            self.metadata["threads"] = self.threads

    def propagate(self, x, relation, edges_to_remove=None):
        if x.device.type != "cpu" or x.dtype != torch.int64:
            raise TypeError("CPUBackend expects a CPU int64 tensor")
        if x.ndim != 3 or x.shape[0] != self.entity_size:
            raise ValueError("input must have shape [entity_size, batch, channels]")
        relation = int(relation)
        if relation < 0 or relation >= len(self._relations):
            raise IndexError("relation out of range")
        batch, channels = x.shape[1:]
        row = self._relations[relation]
        removed = None
        removed_ptr = _I64_PTR()
        if edges_to_remove is not None:
            if edges_to_remove.device.type != "cpu" or edges_to_remove.dtype != torch.int64:
                raise TypeError("edges_to_remove must be a CPU int64 tensor")
            if edges_to_remove.ndim != 1 or edges_to_remove.numel() != batch:
                raise ValueError("edges_to_remove must have one edge index per batch lane")
            removed = edges_to_remove.contiguous()
            removed_ptr = _tensor_ptr(removed)
        output = torch.empty(tuple(x.shape), dtype=torch.int64, device="cpu")
        status = self._propagate(row[3], row[4], row[5], _tensor_ptr(x),
                                 _tensor_ptr(output), removed_ptr,
                                 self.entity_size, row[6], batch, channels,
                                 *x.stride(), self.threads)
        if status == 2:
            raise IndexError("removed edge index out of range")
        if status:
            raise RuntimeError(f"native propagation failed with status {status}")
        return output

    __call__ = propagate
