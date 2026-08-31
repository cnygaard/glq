/* glq_fht_cpu.cpp — block-diagonal Fast Walsh-Hadamard transform, fp32.
 *
 * Bit-exact port of glq/hadamard.py::_pytorch_fht: ascending-distance butterfly
 * (h = 1, 2, ..., n/2; pair (a, b) -> (a+b, a-b)) then ONE division by (float)sqrt(n).
 * The op order is the contract — the CPU fused path must produce the same fp32 values
 * as the torch fallback so the two paths are interchangeable mid-model.
 * Unlike hadamard.block_diagonal_fht (which mutates), this operates on a caller-owned
 * output buffer; the binding clones.
 */
#include <cmath>
#include <cstdint>

namespace glq_cpu {

void fht_inplace(float* x, int64_t n) {
    for (int64_t h = 1; h < n; h <<= 1) {
        for (int64_t i0 = 0; i0 < n; i0 += 2 * h) {
            for (int64_t j = 0; j < h; ++j) {
                const float a = x[i0 + j];
                const float b = x[i0 + h + j];
                x[i0 + j] = a + b;
                x[i0 + h + j] = a - b;
            }
        }
    }
    const float r = (float)std::sqrt((double)n);
    for (int64_t i = 0; i < n; ++i) x[i] /= r;
}

// meta rows: {offset, bs, log_bs, pad} — the _pack_block_meta layout, already CPU int32.
void blockdiag_fht_rows(float* x, int64_t rows, int64_t n, const int32_t* meta,
                        int64_t nblocks) {
    for (int64_t r = 0; r < rows; ++r) {
        float* row = x + r * n;
        for (int64_t b = 0; b < nblocks; ++b)
            fht_inplace(row + meta[4 * b], meta[4 * b + 1]);
    }
}

}  // namespace glq_cpu
