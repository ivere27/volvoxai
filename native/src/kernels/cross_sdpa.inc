// --- 7. Cross SDPA ---
/*
 * Scaled dot-product attention over an explicit K/V memory.
 *
 * The previous implementation walked the key axis three times per (query, head)
 * — once to find the maximum logit, once to accumulate the softmax denominator,
 * and once to accumulate the output — recomputing the same Q.K dot on every
 * pass.  With a 402-row cross-attention memory that made each node cost three
 * times the necessary arithmetic, all of it scalar: 31 ms per
 * node per token, which was 92% of an FP32 decode step.
 *
 * The online-softmax recurrence that replaced it lives in
 * attention_f32_core.h, shared with sdpa.c, which had the same defect in a
 * worse form.  Each key is visited once and the Q.K dot is computed exactly
 * once.  Live state is one head's accumulator rather than a full score row,
 * which is what lets the same code serve the incremental single-row path and a
 * full uncached forward without a scratch allocation.
 *
 * The AVX2 body is selected at runtime through vx_kernel_platform(), so a
 * binary built for a baseline CPU still vectorizes here.  Vectorizing an FP32
 * reduction reorders additions, so the AVX2 and portable paths agree to a
 * floating-point tolerance rather than bit-exactly. The row-versus-full-forward
 * invariant is unaffected: each query
 * row's recurrence reads only that row, so computing one row alone and
 * computing it inside a batch follow the identical sequence of operations.
 */
#include "attention_f32_isa.h"


void cross_sdpa_f32(const float* q_in, const float* k_in, const float* v_in, float* output,
                    int seq_len_q, int seq_len_kv, int d_model, int num_heads,
                    int head_dim, float scale, const int32_t* mask,
                    int mask_mode, int causal) {
#if VX_SDPA_X86_AVX2
    if (head_dim <= VX_SDPA_MAX_HEAD_DIM && vx_kernel_platform()->has_avx2) {
        /* The tiled kernel serves one query row and a full forward alike, and
         * uses it for both deliberately: sharing one path is what makes the
         * incremental row bit-identical to the same row of a full pass. */
        vx_attention_tiled_avx2(q_in, k_in, v_in, d_model, output, d_model,
                                seq_len_q, seq_len_kv, num_heads, head_dim,
                                scale, mask, mask_mode, causal);
        return;
    }
#endif
    vx_attention_f32_portable(q_in, k_in, v_in, d_model, output, d_model,
                              seq_len_q, seq_len_kv, num_heads, head_dim, scale,
                              mask, mask_mode, causal);
}
