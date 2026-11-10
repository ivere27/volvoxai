#include "lstm_plan.h"

int vk_graph_lstm_f32(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction) {
    VxLstmPlan plan;
    if (!vx_lstm_plan(input, weight, recurrent, bias, initial_h, initial_c,
            output, time, batch, width, hidden, direction, &plan)) return 0;
    graph_scratch_begin();
    VkGraphBinding bindings[8];
    for (unsigned i = 0; i < 6; ++i) {
        VkTensorSlot* slot = graph_ensure_device(plan.inputs[i], plan.bytes[i], i >= 1u && i <= 3u);
        if (!slot) return 0;
        bindings[i] = (VkGraphBinding){slot->offset, plan.bytes[i]};
    }
    VkTensorSlot* dst = graph_output_slot(output, plan.bytes[6]);
    size_t offset = graph_scratch_upload(plan.params, sizeof(plan.params));
    if (!dst || offset == SIZE_MAX) return 0;
    bindings[6] = (VkGraphBinding){dst->offset, plan.bytes[6]};
    bindings[7] = (VkGraphBinding){offset, sizeof(plan.params)};
    if (!vk_dispatch_kernel(&k_lstm, bindings, plan.groups, 1u, 1u)) return 0;
    graph_mark_device(dst);
    return 1;
}
