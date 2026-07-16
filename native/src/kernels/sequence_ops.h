#ifndef VOLVOXAI_SEQUENCE_OPS_H
#define VOLVOXAI_SEQUENCE_OPS_H

#include <stdint.h>

void sin_f32(const float* input, float* output, uint32_t elements);
void cos_f32(const float* input, float* output, uint32_t elements);

int rope_f32(const float* input, const void* position_ids, float* output,
             uint32_t batch, uint32_t sequence, uint32_t width,
             uint32_t rotary_width, float theta, int32_t position_offset,
             uint32_t interleaved, uint32_t position_mode);

int ssm_scan_f32(const float* input, const float* delta, const float* a,
                 const float* b, const float* c, const float* d,
                 const float* z, const float* initial_state, float* output,
                 float* final_state, float* state_scratch, uint32_t batch,
                 uint32_t sequence, uint32_t channels, uint32_t state_width,
                 uint32_t b_mode, uint32_t c_mode,
                 uint32_t delta_softplus);

#endif
