"""Small GraphBLAS wrapper optimizations, separate from the measured baseline.

All math and library kernels are unchanged. Preselect column orientation before
numeric multiplication; reuse an exactly empty frontier; update newly owned
outputs in place; rely on GraphBLAS dependencies and terminal export to complete
work instead of a redundant explicit wait on every intermediate.
"""
import numpy as np
from sparse_cpu_backend import SparseCPUBackend, _numpy_i64


class SparseCPUBackendOptimized(SparseCPUBackend):
    def __init__(self, graph, library='graphblas', threads=1):
        if library != 'graphblas':
            raise ValueError('This candidate only changes the GraphBLAS wrapper')
        super().__init__(graph, library=library, threads=threads)
        self.metadata['backend'] = 'graphblas_sparse_state_int64_local_wrapper_optimized'
        self.metadata['local_optimizations'] = [
            'select by-column output format before numeric multiply',
            'reuse exactly empty input state (same shape)',
            'in-place correction accumulation on new output',
            'in-place exact-zero pruning on new output',
            'remove per-intermediate explicit wait; dependent kernels and terminal exports complete work',
        ]
        self.metadata['synchronization'] = 'to_coo/to_dense wait before exporting every consumed terminal; finish_state waits for explicit grounding-only consumers'

    def _empty(self, batch):
        y = self.gb.Matrix(self.gb.dtypes.INT64, nrows=self.entity_size, ncols=batch)
        y.ss.config['format'] = 'by_col'
        return y

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
                raise ValueError('one removed original edge ID required per query')
            if removed.size and (removed.min() < 0 or removed.max() >= len(src)):
                raise IndexError('removed edge ID out of range')
        # Validation occurs even on empty states. This aliases a read-only state;
        # the prefix controller accounts shared objects by identity.
        if x.nvals == 0:
            return x
        y = self._empty(batch)
        if len(src) == 0:
            return y
        gb = self.gb
        y << a.mxm(x, gb.semiring.plus_times)
        if removed is not None and batch:
            diagonal = x[src[removed], np.arange(batch)].new().diag()
            queries, values = diagonal.to_coo()
            if len(queries):
                correction = gb.Matrix.from_coo(dst[removed[queries]], queries, -values,
                                nrows=self.entity_size, ncols=batch, dtype=gb.dtypes.INT64)
                y(accum=gb.binary.plus) << correction
        # Keep exact zero pruning even in overflow scenarios; no nonnegative or
        # no-overflow shortcut is used to remove this operation.
        y << y.select(gb.select.valuene, 0)
        return y

    def finish_state(self, x):
        x.wait()
        return x

    def to_coo(self, x):
        x.wait()
        return super().to_coo(x)

    def to_dense(self, x):
        x.wait()
        return super().to_dense(x)
