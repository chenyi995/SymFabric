"""Local RNNLogic adapter to established integer sparse linear algebra libraries.

No upstream code is changed.  All states are int64 [entity, query].  A relation
adjacency has destination rows and source columns.  Duplicate edges are summed;
one original edge per query is removed by subtracting exactly that edge's source
count, on every step using the query relation.  Nothing is thresholded or Boolean.
"""
from __future__ import annotations

import importlib.metadata
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / 'deps'))
sys.path.insert(0, str(ROOT.parent / '2026-10-14-rnnlogic-strong-baselines'))
import numpy as np
import torch
from prefix_cache import PrefixGrounder


def _numpy_i64(x):
    if isinstance(x, torch.Tensor):
        if x.device.type != 'cpu' or x.dtype != torch.int64:
            raise TypeError('CPU int64 inputs are required')
        return x.detach().numpy()
    x = np.asarray(x)
    if x.dtype != np.int64:
        raise TypeError('int64 inputs are required')
    return x


class SparseCPUBackend:
    """SciPy CSC or SuiteSparse:GraphBLAS plus_times.int64 propagation.

    ``threads`` controls GraphBLAS's global thread limit.  SciPy's sparse matrix
    multiplication here is single-threaded (it is not a BLAS GEMM).  It is not
    honest to label a SciPy result as a 16-thread sparse-kernel result.
    """

    def __init__(self, graph, library='scipy', threads=1):
        started = time.perf_counter()
        if library not in {'scipy', 'graphblas'}:
            raise ValueError('library must be scipy or graphblas')
        if int(threads) != threads or threads < 1:
            raise ValueError('threads must be positive')
        self.library = library
        self.entity_size = int(graph.entity_size)
        self.threads = int(threads)
        if library == 'scipy':
            import scipy
            import scipy.sparse as sp
            self.sp = sp
            versions = {'scipy': scipy.__version__}
        else:
            import graphblas as gb
            import suitesparse_graphblas as ss
            gb.ss.config['nthreads'] = self.threads
            self.gb = gb
            versions = {'python_graphblas': gb.__version__,
                        'suitesparse_graphblas': ss.__version__,
                        'suitesparse_library': list(gb.ss.about['library_version'])}
        self._relations = []
        bytes_total = edges = 0
        for entry in graph.relation2adjacency:
            index = _numpy_i64(entry[0])
            if index.ndim != 2 or index.shape[0] != 2:
                raise ValueError('adjacency shape must be [2,E]')
            dst, src = (np.ascontiguousarray(z) for z in index)
            if index.size and (index.min() < 0 or index.max() >= self.entity_size):
                raise ValueError('invalid adjacency entity')
            if library == 'scipy':
                a = sp.csc_matrix((np.ones(len(src), dtype=np.int64), (dst, src)),
                                  shape=(self.entity_size, self.entity_size), dtype=np.int64)
                a.sum_duplicates()
                a.sort_indices()
            else:
                a = gb.Matrix.from_coo(dst, src, np.ones(len(src), dtype=np.int64),
                    nrows=self.entity_size, ncols=self.entity_size,
                    dtype=gb.dtypes.INT64, dup_op=gb.binary.plus)
                a.ss.config['format'] = 'by_col'
                a.wait()
            self._relations.append((a, dst, src))
            bytes_total += self.nbytes(a) + dst.nbytes + src.nbytes
            edges += len(src)
        self.metadata = {
            'backend': f'{library}_sparse_state_int64', 'library_versions': versions,
            'numpy': np.__version__, 'threads_requested': self.threads,
            'sparse_kernel_threads': 1 if library == 'scipy' else self.threads,
            'thread_scope': 'SuiteSparse global config' if library == 'graphblas' else 'SciPy sparse kernels serial',
            'nodes': self.entity_size, 'relations': len(self._relations), 'edges': edges,
            'prepared_graph_bytes': bytes_total, 'setup_seconds': time.perf_counter() - started,
            'semiring': 'plus_times int64', 'state_shape': '[entity, query]',
            'relation_orientation': 'destination rows, source columns; CSC/by_col',
            'deletion': 'subtract one original edge contribution separately in each query column',
            'duplicate_edges': 'adjacency stores int64 edge multiplicity; deletion subtracts one copy',
            'accumulation': 'fixed-width int64; caller must establish overflow safety',
        }

    def set_threads(self, threads):
        if int(threads) != threads or threads < 1:
            raise ValueError('threads must be positive')
        self.threads = int(threads)
        if self.library == 'graphblas':
            self.gb.ss.config['nthreads'] = self.threads
        self.metadata['threads_requested'] = self.threads
        self.metadata['sparse_kernel_threads'] = 1 if self.library == 'scipy' else self.threads

    def initial(self, h):
        h = _numpy_i64(h)
        if h.ndim != 1 or (h.size and (h.min() < 0 or h.max() >= self.entity_size)):
            raise ValueError('h must contain in-range entity IDs')
        batch = len(h)
        if self.library == 'scipy':
            return self.sp.csc_matrix((np.ones(batch, np.int64), (h, np.arange(batch))),
                                      shape=(self.entity_size, batch), dtype=np.int64)
        x = self.gb.Matrix.from_coo(h, np.arange(batch), np.ones(batch, np.int64),
                nrows=self.entity_size, ncols=batch, dtype=self.gb.dtypes.INT64)
        x.ss.config['format'] = 'by_col'
        return x

    def nbytes(self, x):
        if self.library == 'scipy':
            return x.data.nbytes + x.indices.nbytes + x.indptr.nbytes
        return x.ss.nbytes

    def nnz(self, x):
        return int(x.nnz if self.library == 'scipy' else x.nvals)

    def propagate(self, x, relation, edges_to_remove=None):
        relation = int(relation)
        if relation < 0 or relation >= len(self._relations):
            raise IndexError('relation out of range')
        a, dst, src = self._relations[relation]
        if x.shape[0] != self.entity_size:
            raise ValueError('state entity dimension mismatch')
        batch = x.shape[1]
        removed = None
        if edges_to_remove is not None:
            removed = _numpy_i64(edges_to_remove)
            if removed.shape != (batch,):
                raise ValueError('one removed original edge ID is required per query')
            if removed.size and (removed.min() < 0 or removed.max() >= len(src)):
                raise IndexError('removed edge ID out of range')
        if self.library == 'scipy':
            y = a @ x
            if removed is not None and batch:
                # Advanced sparse indexing is pairwise, not a Cartesian slice.
                values = np.asarray(x[src[removed], np.arange(batch)]).reshape(-1)
                correction = self.sp.csc_matrix((values, (dst[removed], np.arange(batch))), shape=x.shape)
                y = y - correction
            y.eliminate_zeros()
            y.sort_indices()
            return y
        gb = self.gb
        y = a.mxm(x, gb.semiring.plus_times).new()
        if removed is not None and batch:
            # GraphBLAS matrix indexing is Cartesian. Taking its diagonal gives
            # precisely X[source_of_removed_edge[q], q], including duplicate src.
            diagonal = x[src[removed], np.arange(batch)].new().diag()
            queries, values = diagonal.to_coo()
            correction = gb.Matrix.from_coo(dst[removed[queries]], queries, -values,
                            nrows=self.entity_size, ncols=batch, dtype=gb.dtypes.INT64)
            y = y.ewise_add(correction, gb.binary.plus).new()
        # GraphBLAS permits explicit zeros; remove them before state statistics
        # or sparse consumption. Exact cancellation after deletion is common.
        y = y.select(gb.select.valuene, 0).new()
        y.ss.config['format'] = 'by_col'
        y.wait()
        return y

    def to_coo(self, x):
        """Return NumPy (entity, query, int64 count), unique nonzero entries."""
        if self.library == 'scipy':
            x = x.tocoo(copy=False)
            return x.row.astype(np.int64, copy=False), x.col.astype(np.int64, copy=False), x.data
        r, c, v = x.to_coo(sort=False)
        return r.astype(np.int64, copy=False), c.astype(np.int64, copy=False), v

    def to_sparse(self, x):
        """Return CPU torch (query*N+entity, int64 count) for sparse consumer."""
        entity, batch, values = self.to_coo(x)
        return torch.from_numpy(batch * self.entity_size + entity), torch.from_numpy(values)

    def to_dense(self, x):
        """Return exact CPU torch int64 [query,entity] for unchanged predictor."""
        if self.library == 'scipy':
            return torch.from_numpy(x.T.toarray())
        return torch.from_numpy(x.T.to_dense(fill_value=0))


class SparsePrefixGrounder(PrefixGrounder):
    """Existing prefix policy with sparse state and a small terminal adapter.

    Default return_format='dense' keeps the upstream grounding contract.  For a
    sparse consumer choose 'sparse': each rule returns unique (flat_index,count)
    where flat_index=query*N+entity, with no dense terminal materialization.
    Cache is reset for every begin/finish; no cross-forward reuse is implemented.
    """

    def __init__(self, graph, rule_bodies, backend=None, *, library='scipy', threads=1,
                 policy='last_use', budget_bytes=None, collect_stats=True,
                 return_format='dense'):
        if return_format not in {'dense', 'sparse'}:
            raise ValueError('return_format must be dense or sparse')
        self.backend = backend or SparseCPUBackend(graph, library=library, threads=threads)
        self.return_format = return_format
        super().__init__(graph, rule_bodies, policy=policy, budget_bytes=budget_bytes,
                         collect_stats=collect_stats)

    def _storage(self, x):
        return ('cpu_sparse', self.backend.library, id(x)), self.backend.nbytes(x)

    @torch.no_grad()
    def grounding_state(self, h, head, body, removed=None):
        if not self._active:
            raise RuntimeError('call begin before grounding')
        if self._signature(h) != self._h_signature or self._signature(removed) != self._removed_signature:
            raise ValueError('query tensors changed within a forward')
        if self._integer(head) != self._head:
            raise ValueError('query relation changed within a forward')
        body = tuple(self._integer(r) for r in body)
        if self._index >= len(self.rule_bodies) or body != self.rule_bodies[self._index]:
            raise ValueError('grounding calls must follow declared rule order')
        prefixes = self._plans[self._index]
        x, start = None, 0
        for depth in range(len(prefixes) - 1, -1, -1):
            prefix = prefixes[depth]
            if prefix in self._cache:
                x = self._cache[prefix]
                self._cache.move_to_end(prefix)
                if self.collect_stats:
                    self._stats['hits'] += 1
                    self._stats['cache_lookup_hits'] += 1
                start = depth
                break
            if self.collect_stats:
                self._stats['misses'] += 1
                self._stats['cache_lookup_misses'] += 1
        if x is None:
            x = self.backend.initial(h)
            if self.collect_stats:
                self._stats['identity_builds'] += 1
            self._observe(x)
            self._retain((), x)
        for depth in range(start, len(body)):
            relation, prefix = body[depth], prefixes[depth + 1]
            y = self.backend.propagate(x, relation, removed if relation == self._head else None)
            if self.collect_stats:
                self._stats['propagations'] += 1
                self._stats['recomputes'] += int(prefix in self._seen)
                self._seen.add(prefix)
            self._observe(x, y)
            x = y
            del y
            self._retain(prefix, x)
        if self._release_last_use:
            for prefix in self._release_at[self._index]:
                if prefix in self._cache:
                    self._drop(prefix, 'last_use_releases')
        self._observe(x)
        self._index += 1
        self._stats['rule_calls'] += 1
        return x

    def grounding(self, h, head, body, removed=None):
        x = self.grounding_state(h, head, body, removed)
        return self.backend.to_dense(x) if self.return_format == 'dense' else self.backend.to_sparse(x)

    def grounding_sparse(self, h, head, body, removed=None):
        return self.backend.to_sparse(self.grounding_state(h, head, body, removed))
