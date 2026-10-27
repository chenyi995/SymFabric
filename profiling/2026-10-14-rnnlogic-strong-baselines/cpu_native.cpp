// Local propagation backend for RNNLogic; no dependency on PyTorch's C++ ABI.
#include <cstdint>
#include <cstring>
#include <omp.h>

extern "C" int rnnlogic_openmp_max_threads() { return omp_get_max_threads(); }

// CSR groups by destination but retains each edge's original position. Every
// output row has one owner, so integer reductions need no atomic operations.
extern "C" int rnnlogic_propagate_i64(
    const int64_t* rowptr, const int64_t* sources, const int64_t* edge_ids,
    const int64_t* input, int64_t* output, const int64_t* removed,
    int64_t nodes, int64_t edges, int64_t batch, int64_t channels,
    int64_t stride_n, int64_t stride_b, int64_t stride_d, int threads) {
    if (nodes < 0 || edges < 0 || batch < 0 || channels < 0 || threads < 1)
        return 1;
    // Normalize Python-style negative edge indices when supplied, and reject
    // invalid indices before entering the parallel region or touching output.
    if (removed) {
        for (int64_t b = 0; b < batch; ++b)
            if (removed[b] < -edges || removed[b] >= edges) return 2;
    }
    const int64_t width = batch * channels;
    #pragma omp parallel for schedule(dynamic, 32) num_threads(threads)
    for (int64_t dst = 0; dst < nodes; ++dst) {
        int64_t* result = output + dst * width;
        // Unsigned arithmetic specifies two's-complement wraparound instead of
        // invoking C++ signed-overflow UB. Intended workloads must still audit
        // count ranges: this is fixed-width int64, not arbitrary precision.
        auto* accum = reinterpret_cast<uint64_t*>(result);
        for (int64_t i = 0; i < width; ++i) accum[i] = 0;
        for (int64_t pos = rowptr[dst]; pos < rowptr[dst + 1]; ++pos) {
            const int64_t edge = edge_ids[pos];
            const int64_t* source = input + sources[pos] * stride_n;
            if (!removed) {
                if (stride_b == channels && stride_d == 1) {
                    #pragma omp simd
                    for (int64_t i = 0; i < width; ++i)
                        accum[i] += static_cast<uint64_t>(source[i]);
                } else {
                    for (int64_t b = 0; b < batch; ++b)
                        for (int64_t d = 0; d < channels; ++d)
                            accum[b * channels + d] +=
                                static_cast<uint64_t>(source[b * stride_b + d * stride_d]);
                }
            } else {
                for (int64_t b = 0; b < batch; ++b) {
                    int64_t remove = removed[b];
                    if (remove < 0) remove += edges;
                    if (remove == edge) continue;
                    #pragma omp simd
                    for (int64_t d = 0; d < channels; ++d)
                        accum[b * channels + d] +=
                            static_cast<uint64_t>(source[b * stride_b + d * stride_d]);
                }
            }
        }
    }
    return 0;
}
