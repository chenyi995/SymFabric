"""Reuse exact adapter tests with independent optimized GraphBLAS backend."""
import test_sparse_cpu as tests
from sparse_cpu_backend_optimized import SparseCPUBackendOptimized
original=tests.SparseCPUBackend

def factory(graph,library='scipy',threads=1):
    return SparseCPUBackendOptimized(graph,threads=threads) if library=='graphblas' else original(graph,library=library,threads=threads)

tests.SparseCPUBackend=factory
if __name__=='__main__':tests.main()
