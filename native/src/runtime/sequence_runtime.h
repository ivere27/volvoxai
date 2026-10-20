#ifndef VOLVOX_RUNTIME_SEQUENCE_RUNTIME_H
#define VOLVOX_RUNTIME_SEQUENCE_RUNTIME_H

#include "engine_internal.h"
#include "decode_row_set.h"

enum {
    VX_SEQUENCE_ERROR = -1,
    VX_SEQUENCE_NOT_HANDLED = 0,
    VX_SEQUENCE_HANDLED = 1
};

/* Shared private validation keeps CPU and CUDA RoPE geometry identical. */
typedef struct {
    int batch, sequence, width, heads, rotary_width;
    int position_offset, interleaved, position_mode;
    float theta;
    T* positions;
} VxSequenceRoPE;

int vx_sequence_rope_parameters(Node* node, const T* input, const T* output,
                                VxSequenceRoPE* parameters);

/*
 * Validate and execute a portable sequence operator.  The return value makes
 * this safe to call once from the generic CPU dispatch chain: unrelated ops
 * return NOT_HANDLED without touching tensors, while malformed sequence ops
 * return ERROR.  backend_name receives a stable profiler label on success.
 */
int vx_sequence_node_run(Node* node, long element_offset, int element_count,
                         const char** backend_name);

/* Compute staged token rows, retaining the original position-list contract. */
int vx_sequence_rope_row(Node* node, const T* input, const T* output,
                         const VxDecodeRowSet* rows, const float* input_rows,
                         float* output_rows);

#endif
