/*
 * Portable forward kernels for sequence-model primitives.
 *
 * These are intentionally scalar correctness kernels shared by native CPU and
 * the ordinary inference WASM sidecar. Architecture-specific acceleration can
 * wrap the same contracts without changing graph semantics.
 */
#include "mathcompat.h"
#include "sequence_ops.h"
#include <stddef.h>
#include <stdint.h>

#ifndef WASM_EXPORT
#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif
#endif

static int vx_sequence_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

static int vx_sequence_product(size_t *value, uint32_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static int32_t vx_sequence_i32_at(const void* values, size_t index) {
    int32_t value;
    __builtin_memcpy(&value,
        (const unsigned char*)values + index * sizeof(value), sizeof(value));
    return value;
}

WASM_EXPORT("sin_f32")
void sin_f32(const float *input, float *output, uint32_t elements) {
    if (!input || !output) return;
    for (uint32_t index = 0; index < elements; index++) {
        output[index] = sinf(input[index]);
    }
}

WASM_EXPORT("cos_f32")
void cos_f32(const float *input, float *output, uint32_t elements) {
    if (!input || !output) return;
    for (uint32_t index = 0; index < elements; index++) {
        output[index] = cosf(input[index]);
    }
}

/*
 * Rotary position embedding over [B,S,D] (rank two uses B=1). The optional
 * position array is either [S] (mode 1) or [B,S] (mode 2); mode 0 derives
 * positions from sequence index plus position_offset. Interleaved mode uses
 * adjacent pairs; the default uses half-split pairs.
 */
WASM_EXPORT("rope_f32")
int rope_f32(const float *input, const void *position_ids, float *output,
        uint32_t batch, uint32_t sequence, uint32_t width,
        uint32_t rotary_width, float theta, int32_t position_offset,
        uint32_t interleaved, uint32_t position_mode) {
    size_t elements = batch;
    if (!input || !output || !batch || !sequence || !width || !rotary_width ||
        rotary_width > width || (rotary_width & 1u) ||
        !vx_sequence_finite_f32(theta) || theta <= 0.0f ||
        position_offset < 0 || sequence > 0x7fffffffu ||
        (uint32_t)position_offset > 0x7fffffffu - (sequence - 1u) ||
        interleaved > 1u || position_mode > 2u ||
        (position_mode && !position_ids) ||
        !vx_sequence_product(&elements, sequence) ||
        !vx_sequence_product(&elements, width)) return 0;

    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t sequence_index = 0; sequence_index < sequence; sequence_index++) {
            int32_t position;
            size_t row = (size_t)batch_index * sequence + sequence_index;
            size_t base = row * width;
            if (position_mode == 1u)
                position = vx_sequence_i32_at(position_ids, sequence_index);
            else if (position_mode == 2u)
                position = vx_sequence_i32_at(position_ids, row);
            else position = position_offset + (int32_t)sequence_index;
            if (position < 0) return 0;

            if (interleaved) {
                for (uint32_t pair = 0; pair < rotary_width / 2u; pair++) {
                    uint32_t left = pair * 2u;
                    uint32_t right = left + 1u;
                    float exponent = (2.0f * (float)pair) / (float)rotary_width;
                    float angle = (float)position / powf(theta, exponent);
                    float cosine = cosf(angle);
                    float sine = sinf(angle);
                    float left_value = input[base + left];
                    float right_value = input[base + right];
                    output[base + left] = left_value * cosine - right_value * sine;
                    output[base + right] = left_value * sine + right_value * cosine;
                }
            } else {
                uint32_t half = rotary_width / 2u;
                for (uint32_t pair = 0; pair < half; pair++) {
                    uint32_t left = pair;
                    uint32_t right = pair + half;
                    float exponent = (2.0f * (float)pair) / (float)rotary_width;
                    float angle = (float)position / powf(theta, exponent);
                    float cosine = cosf(angle);
                    float sine = sinf(angle);
                    float left_value = input[base + left];
                    float right_value = input[base + right];
                    output[base + left] = left_value * cosine - right_value * sine;
                    output[base + right] = left_value * sine + right_value * cosine;
                }
            }
            for (uint32_t dimension = rotary_width; dimension < width; dimension++) {
                output[base + dimension] = input[base + dimension];
            }
        }
    }
    (void)elements;
    return 1;
}

static float vx_sequence_softplus(float value) {
    if (value > 20.0f) return value;
    if (value < -20.0f) return expf(value);
    return logf(1.0f + expf(value));
}

static size_t vx_sequence_bc_index(uint32_t mode, uint32_t batch_index,
        uint32_t sequence_index, uint32_t state_index,
        uint32_t sequence, uint32_t state_width) {
    if (mode == 0u) return state_index;
    if (mode == 1u) return (size_t)sequence_index * state_width + state_index;
    return ((size_t)batch_index * sequence + sequence_index) * state_width + state_index;
}

/*
 * Mamba-style selective scan:
 *   state = exp(dt * A) * state + dt * B * u
 *   y     = sum(state * C) + D * u
 * Optional z applies SiLU(z) as an output gate. B/C mode 0, 1, and 2 mean
 * [N], [S,N], and [B,S,N], respectively. `state_scratch` is caller-owned
 * [B,D,N] storage so the freestanding WASM kernel never allocates.
 */
WASM_EXPORT("ssm_scan_f32")
int ssm_scan_f32(const float *input, const float *delta, const float *a,
        const float *b, const float *c, const float *d, const float *z,
        const float *initial_state, float *output, float *final_state,
        float *state_scratch, uint32_t batch, uint32_t sequence,
        uint32_t channels, uint32_t state_width, uint32_t b_mode,
        uint32_t c_mode, uint32_t delta_softplus) {
    size_t input_elements = batch;
    size_t state_elements = batch;
    if (!input || !delta || !a || !b || !c || !output || !state_scratch ||
        !batch || !sequence || !channels || !state_width || b_mode > 2u ||
        c_mode > 2u || delta_softplus > 1u ||
        !vx_sequence_product(&input_elements, sequence) ||
        !vx_sequence_product(&input_elements, channels) ||
        !vx_sequence_product(&state_elements, channels) ||
        !vx_sequence_product(&state_elements, state_width)) return 0;

    for (size_t index = 0; index < state_elements; index++) {
        state_scratch[index] = initial_state ? initial_state[index] : 0.0f;
    }
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t sequence_index = 0; sequence_index < sequence; sequence_index++) {
            for (uint32_t channel = 0; channel < channels; channel++) {
                size_t input_index = ((size_t)batch_index * sequence + sequence_index) * channels + channel;
                size_t state_base = ((size_t)batch_index * channels + channel) * state_width;
                size_t a_base = (size_t)channel * state_width;
                float input_value = input[input_index];
                float dt = delta[input_index];
                float result = d ? d[channel] * input_value : 0.0f;
                if (delta_softplus) dt = vx_sequence_softplus(dt);
                for (uint32_t state_index = 0; state_index < state_width; state_index++) {
                    size_t b_index = vx_sequence_bc_index(b_mode, batch_index,
                        sequence_index, state_index, sequence, state_width);
                    size_t c_index = vx_sequence_bc_index(c_mode, batch_index,
                        sequence_index, state_index, sequence, state_width);
                    float state = expf(dt * a[a_base + state_index]) *
                        state_scratch[state_base + state_index] +
                        dt * b[b_index] * input_value;
                    state_scratch[state_base + state_index] = state;
                    result += state * c[c_index];
                }
                if (z) {
                    float gate = z[input_index];
                    gate = gate / (1.0f + expf(-gate));
                    result *= gate;
                }
                output[input_index] = result;
            }
        }
    }
    if (final_state) {
        for (size_t index = 0; index < state_elements; index++) {
            final_state[index] = state_scratch[index];
        }
    }
    (void)input_elements;
    return 1;
}
