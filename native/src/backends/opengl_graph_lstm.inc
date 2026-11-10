#include "lstm_plan.h"

int opengl_graph_lstm_f32(const float* input, const float* weight, const float* recurrent,
        const float* bias, const float* initial_h, const float* initial_c,
        float* output, uint32_t time, uint32_t batch, uint32_t width,
        uint32_t hidden, uint32_t direction) {
    VxLstmPlan plan;
    if (!vx_lstm_plan(input, weight, recurrent, bias, initial_h, initial_c,
            output, time, batch, width, hidden, direction, &plan)) return 0;
    GLuint bindings[8];
    for (unsigned i = 0; i < 6; ++i) {
        OglTensorSlot* slot = graph_ensure_device(plan.inputs[i], plan.bytes[i], i >= 1u && i <= 3u);
        if (!slot) return 0;
        bindings[i] = slot->buffer;
    }
    OglTensorSlot* dst = graph_output_slot(output, plan.bytes[6]);
    if (!dst) return 0;
    bindings[6] = dst->buffer;
    GLuint pb = params_buffer(plan.params, sizeof(plan.params));
    if (!pb) return 0;
    bindings[7] = pb;
    int ok = dispatch_kernel(&k_lstm, bindings, plan.groups, 1u, 1u);
    p_glDeleteBuffers(1, &pb);
    if (!ok) return 0;
    graph_mark_device(dst);
    return 1;
}
