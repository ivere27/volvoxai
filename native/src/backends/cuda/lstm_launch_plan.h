#ifndef VOLVOXAI_CUDA_LSTM_LAUNCH_PLAN_H
#define VOLVOXAI_CUDA_LSTM_LAUNCH_PLAN_H
#include <stddef.h>
#include <stdint.h>

typedef struct VxCudaLstmPreparedPlan {
    size_t recurrent_offset, projection_offset, gates_offset, cells_offset, bytes;
    uint32_t directions, gate_width, projection_groups, gate_groups, state_groups;
} VxCudaLstmPreparedPlan;

/* The existing, context-owned dense workspace is already included in CUDA's
 * memory admission and released only after its replay graphs are retired.
 * No allocation or cached activation is introduced by this optional tactic.
 * All five regions are overwritten by the current request before being read. */
static inline int vx_cuda_lstm_prepared_plan(uint32_t time, uint32_t batch,
        uint32_t width, uint32_t hidden, uint32_t direction, size_t capacity,
        VxCudaLstmPreparedPlan* plan) {
    if (!plan || time < 2u || !batch || width < 32u || hidden < 32u ||
        hidden > 1024u || direction > 2u) return 0;
    uint64_t directions = direction == 2u ? 2u : 1u;
    if (batch > 65535u / directions) return 0;
    uint64_t gates = directions * batch * 4u * hidden;
    uint64_t weight = directions * 4u * hidden * width;
    uint64_t recurrent = directions * 4u * hidden * hidden;
    /* Bound individual terms before multiplying by time or adding offsets. */
    if (gates > UINT32_MAX / time || weight > UINT32_MAX / sizeof(float) ||
        recurrent > UINT32_MAX / sizeof(float)) return 0;
    uint64_t projection = gates * time;
    uint64_t elements = weight + recurrent + 2u * projection + gates + gates / 4u;
    if (elements > UINT32_MAX / sizeof(float) || elements > SIZE_MAX / sizeof(float) ||
        elements > capacity / sizeof(float)) return 0;
    plan->recurrent_offset = (size_t)weight * sizeof(float);
    plan->projection_offset = (size_t)(weight + recurrent) * sizeof(float);
    plan->gates_offset = (size_t)(weight + recurrent + 2u * projection) * sizeof(float);
    plan->cells_offset = plan->gates_offset + (size_t)gates * sizeof(float);
    plan->bytes = (size_t)elements * sizeof(float);
    plan->directions = (uint32_t)directions;
    plan->gate_width = 4u * hidden;
    plan->projection_groups = (uint32_t)((projection + 127u) / 128u);
    plan->gate_groups = (uint32_t)((gates + 127u) / 128u);
    plan->state_groups = (uint32_t)((gates / 4u + 127u) / 128u);
    return 1;
}
#endif
