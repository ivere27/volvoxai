#include "lstm_plan.h"

int metal_graph_lstm_f32(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction) {
    VxLstmPlan plan;
    if (!vx_lstm_plan(input, weight, recurrent, bias, initial_h, initial_c,
            output, time, batch, width, hidden, direction, &plan)) return 0;
    MetalBinding bindings[8];
    for (unsigned i = 0; i < 6; ++i) {
        MetalTensorSlot* slot = graph_ensure_device(plan.inputs[i], plan.bytes[i], i >= 1u && i <= 3u);
        if (!slot) return 0;
        bindings[i] = (MetalBinding){slot->buffer, plan.bytes[i]};
    }
    MetalTensorSlot* dst = graph_output_slot(output, plan.bytes[6]);
    if (!dst) return 0;
    id<MTLBuffer> pb = create_buffer(sizeof(plan.params), plan.params);
    if (!pb) return 0;
    bindings[6] = (MetalBinding){dst->buffer, plan.bytes[6]};
    bindings[7] = (MetalBinding){pb, sizeof(plan.params)};
    int ok = dispatch_kernel(&k_lstm, bindings, plan.groups, 1u, 1u);
    VX_METAL_RELEASE(pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
