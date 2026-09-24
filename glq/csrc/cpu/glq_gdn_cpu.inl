/* glq_gdn_cpu.inl — the GatedDeltaNet decode step, compiled once per ISA tier.
 *
 * Included into a namespace per tier behind `#pragma GCC target`, exactly like
 * glq_trellis_cpu_zmm.inl. There are no intrinsics here on purpose: the body is plain fp32
 * FMA over contiguous rows of length V, which GCC vectorises well under each tier's target
 * pragma — the same reason hand-vectorising the trellis bit-unpack was a measured NO-GO.
 *
 * The transformation this implements is a fusion, not a new algorithm. The reference
 * (`torch_recurrent_gated_delta_rule`) walks the K x V state FIVE times per token and
 * materialises four temporaries of that size:
 *
 *     S * decay                     -> temp
 *     (S * k[:,None]).sum(dim=-2)   -> temp, then reduce
 *     S + k[:,None]*d[None,:]       -> temp
 *     (S * q[:,None]).sum(dim=-2)   -> temp, then reduce
 *
 * Here the decay folds into the kv reduction and the rank-1 update folds into the output
 * reduction, so the state is touched exactly TWICE and nothing else is allocated.
 * At 48 heads x 128 x 128 that is ~27 MiB/layer of traffic down to ~6 MiB.
 *
 * NOT bit-exact against the reference, and cannot be: torch reduces over dim=-2 of a
 * (K, V) tensor in its own order while this accumulates along i. Agreement is a tolerance,
 * and tests/test_gdn_cpu_kernel.py sizes it from the measured gap.
 */

/* One decode step for ONE (batch, head).
 *
 *   S     K x V, row-major, MUTATED IN PLACE (this is the whole point)
 *   q, k  K      q arrives pre-scaled by 1/sqrt(K) and l2-normalised by the caller
 *   v     V
 *   decay scalar, ALREADY exponentiated
 *   beta  scalar
 *   out   V, written
 *   scratch V, caller-owned so the hot path allocates nothing
 */
static inline void gdn_head_step(float* __restrict__ S,
                                 const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 float decay, float beta,
                                 float* __restrict__ out,
                                 float* __restrict__ scratch,
                                 int64_t K, int64_t V) {
    // ---- pass 1: decay the state, and reduce kv[j] = sum_i S[i,j]*k[i] in the same sweep
    for (int64_t j = 0; j < V; ++j) scratch[j] = 0.0f;
    for (int64_t i = 0; i < K; ++i) {
        float* __restrict__ Si = S + i * V;
        const float ki = k[i];
        for (int64_t j = 0; j < V; ++j) {
            const float s = Si[j] * decay;
            Si[j] = s;
            scratch[j] += s * ki;
        }
    }

    // delta[j] = (v[j] - kv[j]) * beta, in place over the scratch that held kv
    for (int64_t j = 0; j < V; ++j) scratch[j] = (v[j] - scratch[j]) * beta;

    // ---- pass 2: rank-1 update, and reduce out[j] = sum_i S[i,j]*q[i] in the same sweep.
    // The update must land BEFORE the output reduction reads it -- the reference computes
    // core_attn_out from the ALREADY-updated state (modeling_qwen4_exp.py:453).
    for (int64_t j = 0; j < V; ++j) out[j] = 0.0f;
    for (int64_t i = 0; i < K; ++i) {
        float* __restrict__ Si = S + i * V;
        const float ki = k[i];
        const float qi = q[i];
        for (int64_t j = 0; j < V; ++j) {
            const float s = Si[j] + ki * scratch[j];
            Si[j] = s;
            out[j] += s * qi;
        }
    }
}
