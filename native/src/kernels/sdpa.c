#include "mathcompat.h"
#include "attention_f32_isa.h"
// --- 6. SDPA (Self Attention) ---
/*
 * Self attention over a fused QKV tensor.
 *
 * The previous implementation walked the key axis three times per (query, head)
 * and recomputed the Q.K dot on each walk — and the third walk was nested inside
 * a loop over output channels, so it recomputed the whole dot once per channel.
 * That made the final pass O(seq_kv * head_dim^2) and the kernel as a whole cost
 * (head_dim + 2) dot products per (query, head) where one is needed: 42x the
 * necessary arithmetic at this model's head_dim of 40.  cross_sdpa.c had already
 * been given the shared-probability form for exactly this reason; this file had
 * not.
 *
 * Both now share the online-softmax core in attention_f32_core.h, which visits
 * each key once, and select an AVX2 body at runtime through
 * vx_kernel_platform().  Storage is the only difference: Q, K and V are slices
 * of one tensor strided by 3*d_model, while the output is strided by d_model.
 */
void add_f32(const float* a, const float* b, float* out, int n) {
    int i = 0;
    for (; i + 4 <= n; i += 4) {
        v128_t va = wasm_v128_load(a + i);
        v128_t vb = wasm_v128_load(b + i);
        wasm_v128_store(out + i, wasm_f32x4_add(va, vb));
    }
    for (; i < n; i++) out[i] = a[i] + b[i];
}


void sdpa_f32(const float* qkv, float* output, int seq_len, int d_model,
              int num_heads, int head_dim, float scale, const int32_t* mask,
              int mask_mode, int causal) {
    /* Q, K and V are consecutive d_model blocks of each row. */
    const long stride = (long)d_model * 3;
    const float* q_in = qkv;
    const float* k_in = qkv + d_model;
    const float* v_in = qkv + (long)d_model * 2;
#if VX_SDPA_X86_AVX2
    if (head_dim <= VX_SDPA_MAX_HEAD_DIM && vx_kernel_platform()->has_avx2) {
        vx_attention_tiled_avx2(q_in, k_in, v_in, stride, output, d_model,
                                seq_len, seq_len, num_heads, head_dim, scale,
                                mask, mask_mode, causal);
        return;
    }
#endif
    vx_attention_f32_portable(q_in, k_in, v_in, stride, output, d_model,
                              seq_len, seq_len, num_heads, head_dim, scale,
                              mask, mask_mode, causal);
}
