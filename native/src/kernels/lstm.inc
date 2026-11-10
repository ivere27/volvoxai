#include <stdlib.h>
#include "lstm_workspace.h"
#if defined(__wasm__)
#include "gemm_f32.h"
#endif

#if !defined(__wasm__)
#if (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
static VX_FASTMATH_TARGET_AVX2 void vx_lstm_gates_avx2(
    const float* ws_i, const float* ws_o, const float* ws_f, const float* ws_c,
    const float* rg_i, const float* rg_o, const float* rg_f, const float* rg_c,
    float* c_b, float* h_b, float* out_tb, uint32_t hidden) {
    for (uint32_t j = 0; j + 8u <= hidden; j += 8u) {
        __m256 vi = _mm256_add_ps(_mm256_loadu_ps(ws_i + j), _mm256_loadu_ps(rg_i + j));
        __m256 vo = _mm256_add_ps(_mm256_loadu_ps(ws_o + j), _mm256_loadu_ps(rg_o + j));
        __m256 vf = _mm256_add_ps(_mm256_loadu_ps(ws_f + j), _mm256_loadu_ps(rg_f + j));
        __m256 vc = _mm256_add_ps(_mm256_loadu_ps(ws_c + j), _mm256_loadu_ps(rg_c + j));

        __m256 gi = vx_sigmoid_avx2(vi);
        __m256 go = vx_sigmoid_avx2(vo);
        __m256 gf = vx_sigmoid_avx2(vf);
        __m256 gc = vx_tanh_avx2(vc);

        __m256 c_prev = _mm256_loadu_ps(c_b + j);
        __m256 c_curr = _mm256_add_ps(_mm256_mul_ps(gf, c_prev), _mm256_mul_ps(gi, gc));
        __m256 h_curr = _mm256_mul_ps(go, vx_tanh_avx2(c_curr));

        _mm256_storeu_ps(c_b + j, c_curr);
        _mm256_storeu_ps(h_b + j, h_curr);
        _mm256_storeu_ps(out_tb + j, h_curr);
    }
}
#endif
#endif

/* ONNX LSTM, time-major input/output and i/o/f/c gate order. Input projection
 * is one GEMM per direction; only the recurrent projection is sequential. */
static int vx_lstm_f32_impl(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction,
        const float* const* packed_weight, const float* const* packed_recurrent) {
#if !defined(__wasm__)
    (void)packed_weight;
    (void)packed_recurrent;
#endif
    if (!input || !weight || !recurrent || !output || !time || !batch ||
        !width || !hidden || direction > 2u || hidden > INT_MAX / 8u ||
        width > INT_MAX || (uint64_t)time * batch > INT_MAX) return 0;
    uint64_t state_count = (uint64_t)batch * hidden;
    uint64_t projection_count = (uint64_t)time * batch * 4u * hidden;
    size_t workspace_bytes = vx_lstm_workspace_bytes(time, batch, hidden);
    if (!workspace_bytes) return 0;
    /* Admission accounts for this exact temporary allocation. Its lifetime
     * ends with the call, including when the caller uses a short-lived thread. */
    float* workspace = (float*)malloc(workspace_bytes);
    if (!workspace) return 0;
    float* recurrent_gates = workspace + projection_count;
    float* h = recurrent_gates + state_count * 4u;
    float* c = h + state_count;
    uint32_t directions = direction == 2u ? 2u : 1u;
#if !defined(__wasm__) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
    int use_avx2 = vx_kernel_platform()->has_avx2;
#endif

    for (uint32_t d = 0; d < directions; ++d) {
        for (uint64_t index = 0; index < state_count; ++index) {
            h[index] = initial_h ? initial_h[(uint64_t)d * state_count + index] : 0.0f;
            c[index] = initial_c ? initial_c[(uint64_t)d * state_count + index] : 0.0f;
        }
#if defined(__wasm__)
        if (!packed_weight || !packed_weight[d] ||
            !vx_gemm_f32_run_packed(input, packed_weight[d], NULL, workspace,
                time * batch, width, 4u * hidden))
#endif
            matmul_f32_out_in(input, weight + (uint64_t)d * 4u * hidden * width,
                NULL, workspace, (int)(time * batch), (int)width, (int)(4u * hidden));
        if (bias) {
            const float* b1 = bias + (uint64_t)d * 8u * hidden;
            const float* b2 = b1 + 4u * hidden;
            uint64_t total_steps = (uint64_t)time * batch;
            for (uint64_t s_idx = 0; s_idx < total_steps; ++s_idx) {
                float* ws_s = workspace + s_idx * 4u * hidden;
                for (uint32_t g = 0; g < 4u * hidden; ++g) {
                    ws_s[g] += b1[g] + b2[g];
                }
            }
        }
        const float* rw = recurrent + (uint64_t)d * 4u * hidden * hidden;
        for (uint32_t step = 0; step < time; ++step) {
            uint32_t t = direction == 1u || d == 1u ? time - 1u - step : step;
#if defined(__wasm__)
            if (!packed_recurrent || !packed_recurrent[d] ||
                !vx_gemm_f32_run_packed(h, packed_recurrent[d], NULL,
                    recurrent_gates, batch, hidden, 4u * hidden))
#endif
                matmul_f32_out_in(h, rw, NULL, recurrent_gates,
                    (int)batch, (int)hidden, (int)(4u * hidden));
            for (uint32_t b = 0; b < batch; ++b) {
                const float* ws_tb = workspace + ((uint64_t)t * batch + b) * 4u * hidden;
                const float* rg_b = recurrent_gates + (uint64_t)b * 4u * hidden;
                float* c_b = c + (uint64_t)b * hidden;
                float* h_b = h + (uint64_t)b * hidden;
                float* out_tb = output + (((uint64_t)t * directions + d) * batch + b) * hidden;
                const float* ws_i = ws_tb;
                const float* ws_o = ws_tb + hidden;
                const float* ws_f = ws_tb + 2u * hidden;
                const float* ws_c = ws_tb + 3u * hidden;
                const float* rg_i = rg_b;
                const float* rg_o = rg_b + hidden;
                const float* rg_f = rg_b + 2u * hidden;
                const float* rg_c = rg_b + 3u * hidden;
                uint32_t j = 0;

#if defined(__aarch64__) || defined(_M_ARM64)
                for (; j + 4u <= hidden; j += 4u) {
                    float32x4_t vi = vaddq_f32(vld1q_f32(ws_i + j), vld1q_f32(rg_i + j));
                    float32x4_t vo = vaddq_f32(vld1q_f32(ws_o + j), vld1q_f32(rg_o + j));
                    float32x4_t vf = vaddq_f32(vld1q_f32(ws_f + j), vld1q_f32(rg_f + j));
                    float32x4_t vc = vaddq_f32(vld1q_f32(ws_c + j), vld1q_f32(rg_c + j));

                    float32x4_t gi = vx_sigmoid_neon(vi);
                    float32x4_t go = vx_sigmoid_neon(vo);
                    float32x4_t gf = vx_sigmoid_neon(vf);
                    float32x4_t gc = vx_tanh_neon(vc);

                    float32x4_t c_prev = vld1q_f32(c_b + j);
                    float32x4_t c_curr = vaddq_f32(vmulq_f32(gf, c_prev), vmulq_f32(gi, gc));
                    float32x4_t h_curr = vmulq_f32(go, vx_tanh_neon(c_curr));

                    vst1q_f32(c_b + j, c_curr);
                    vst1q_f32(h_b + j, h_curr);
                    vst1q_f32(out_tb + j, h_curr);
                }
#elif !defined(__wasm__) && (defined(__x86_64__) || defined(_M_X64)) && (defined(__GNUC__) || defined(__clang__))
                if (use_avx2) {
                    vx_lstm_gates_avx2(ws_i, ws_o, ws_f, ws_c,
                                       rg_i, rg_o, rg_f, rg_c,
                                       c_b, h_b, out_tb, hidden);
                    j = (hidden / 8u) * 8u;
                }
#endif
                for (; j < hidden; ++j) {
                    float vi = ws_i[j] + rg_i[j];
                    float vo = ws_o[j] + rg_o[j];
                    float vf = ws_f[j] + rg_f[j];
                    float vc = ws_c[j] + rg_c[j];

                    float gi = 1.0f / (1.0f + expf(-vi));
                    float go = 1.0f / (1.0f + expf(-vo));
                    float gf = 1.0f / (1.0f + expf(-vf));
                    float gc = tanhf(vc);

                    float c_curr = gf * c_b[j] + gi * gc;
                    float h_curr = go * tanhf(c_curr);

                    c_b[j] = c_curr;
                    h_b[j] = h_curr;
                    out_tb[j] = h_curr;
                }
            }
        }
    }
    free(workspace);
    return 1;
}

int lstm_f32(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction) {
    return vx_lstm_f32_impl(input, weight, recurrent, bias, initial_h, initial_c,
        output, time, batch, width, hidden, direction, NULL, NULL);
}

#if defined(__wasm__)
/* Internal runtime bridge: packs belong to the prepared engine, while h/c,
 * projections and every input value retain their per-invocation lifetime. */
int lstm_f32_packed(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction,
        const float* const* packed_weight, const float* const* packed_recurrent) {
    return vx_lstm_f32_impl(input, weight, recurrent, bias, initial_h, initial_c,
        output, time, batch, width, hidden, direction, packed_weight, packed_recurrent);
}
#endif
