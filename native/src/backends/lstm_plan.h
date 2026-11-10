#ifndef VOLVOXAI_LSTM_PLAN_H
#define VOLVOXAI_LSTM_PLAN_H
#include <stdint.h>
#include <stddef.h>

typedef struct VxLstmPlan {
    const float* inputs[6];
    size_t bytes[7];
    uint32_t params[8];
    uint32_t groups;
} VxLstmPlan;

static int vx_lstm_plan(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction, VxLstmPlan* plan) {
    if (!plan || !input || !weight || !recurrent || !output || !time || !batch ||
        !width || !hidden || hidden > 1024u || direction > 2u) return 0;
    uint32_t directions = direction == 2u ? 2u : 1u;
    if (batch > 65535u / directions) return 0;
    if ((uint64_t)time * batch > UINT32_MAX / 4u / width ||
        (uint64_t)time * directions * batch > UINT32_MAX / 4u / hidden) return 0;
    uint64_t sizes[7] = {(uint64_t)time * batch * width,
        (uint64_t)directions * 4u * hidden * width,
        (uint64_t)directions * 4u * hidden * hidden,
        (uint64_t)directions * 8u * hidden,
        (uint64_t)directions * batch * hidden,
        (uint64_t)directions * batch * hidden,
        (uint64_t)time * directions * batch * hidden};
    const float* inputs[6] = {input, weight, recurrent, bias, initial_h, initial_c};
    for (unsigned index = 0; index < 7; ++index) {
        if (!sizes[index] || sizes[index] > UINT32_MAX / 4u || sizes[index] > SIZE_MAX / 4u) return 0;
        plan->bytes[index] = (size_t)sizes[index] * sizeof(float);
    }
    uintptr_t out = (uintptr_t)output;
    if (out > UINTPTR_MAX - plan->bytes[6]) return 0;
    for (unsigned index = 0; index < 6; ++index) {
        plan->inputs[index] = inputs[index] ? inputs[index] : input;
        if (!inputs[index]) plan->bytes[index] = sizeof(float);
        uintptr_t in = (uintptr_t)plan->inputs[index];
        if (in > UINTPTR_MAX - plan->bytes[index] ||
            (in < out + plan->bytes[6] && out < in + plan->bytes[index])) return 0;
    }
    plan->params[0] = time; plan->params[1] = batch;
    plan->params[2] = width; plan->params[3] = hidden;
    plan->params[4] = direction;
    plan->params[5] = (bias ? 1u : 0u) | (initial_h ? 2u : 0u) | (initial_c ? 4u : 0u);
    plan->params[6] = plan->params[7] = 0u;
    plan->groups = batch * directions;
    return 1;
}

static inline int vx_packed_lstm_unpack_bank(const void* raw_data, const float* scales,
                                             int rows, int k, int words, int groups,
                                             int group_size, int weight_bits, float* dst) {
    const unsigned char* raw = (const unsigned char*)raw_data;
    const int32_t bias = 1 << (weight_bits - 1);
    const uint32_t mask_tail = ((uint64_t)k * (uint32_t)weight_bits) % 32u
        ? (UINT32_MAX << (uint32_t)(((uint64_t)k * (uint32_t)weight_bits) % 32u))
        : 0u;
    const uint32_t pad_tail = weight_bits == 3 ? 0u
        : weight_bits == 2 ? 0xaaaaaaaau
        : weight_bits == 4 ? 0x88888888u : 0x80808080u;
    if (!raw || !scales || !dst || rows <= 0 || k <= 0 || words <= 0 || groups <= 0 || group_size <= 0) return 0;
    for (long i = 0; i < (long)rows * groups; i++) {
        if (!isfinite(scales[i]) || scales[i] <= 0.0f) return 0;
    }
    for (int row = 0; row < rows; row++) {
        const unsigned char* row_bytes = raw + (size_t)row * (size_t)words * 4u;
        const float* row_scales = scales + (size_t)row * (size_t)groups;
        float* row_dst = dst + (size_t)row * (size_t)k;
        if (mask_tail) {
            uint32_t last_word;
            memcpy(&last_word, row_bytes + (size_t)(words - 1) * 4u, 4u);
            if ((last_word & mask_tail) != (pad_tail & mask_tail)) return 0;
        }
        if (weight_bits == 3) {
            for (int col = 0; col < k; col++) {
                uint64_t bit = (uint64_t)col * 3u;
                uint32_t word_idx = (uint32_t)(bit / 32u);
                uint32_t shift = (uint32_t)(bit % 32u);
                uint32_t w0, code;
                memcpy(&w0, row_bytes + (size_t)word_idx * 4u, 4u);
                code = w0 >> shift;
                if (shift > 29u) {
                    uint32_t w1;
                    memcpy(&w1, row_bytes + (size_t)(word_idx + 1u) * 4u, 4u);
                    code |= w1 << (32u - shift);
                }
                code &= 7u;
                row_dst[col] = (float)((int32_t)code - bias) * row_scales[col / group_size];
            }
        } else {
            const uint32_t per_word = 32u / (uint32_t)weight_bits;
            const uint32_t code_mask = (1u << (uint32_t)weight_bits) - 1u;
            for (int col = 0; col < k; col++) {
                uint32_t word_idx = (uint32_t)col / per_word;
                uint32_t shift = ((uint32_t)col % per_word) * (uint32_t)weight_bits;
                uint32_t w0;
                memcpy(&w0, row_bytes + (size_t)word_idx * 4u, 4u);
                uint32_t code = (w0 >> shift) & code_mask;
                row_dst[col] = (float)((int32_t)code - bias) * row_scales[col / group_size];
            }
        }
    }
    return 1;
}
#endif
