#include "lstm_plan.h"

static int vx_webgpu_plan_lstm(const Node* node, T* input, T* output, VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* recurrent = vx_webgpu_port(node, VX_PORT_RECURRENT_WEIGHT);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    T* h = vx_webgpu_port(node, VX_PORT_INITIAL_H);
    T* c = vx_webgpu_port(node, VX_PORT_INITIAL_C);
    int direction = vx_node_param_i32(node, VX_NODE_PARAM_DIRECTION, 0);
    if (!input || !output || !weight || !recurrent || input->ndim != 3 ||
        output->ndim != 4 || weight->ndim != 3 || recurrent->ndim != 3 ||
        direction < 0 || direction > 2) return 0;
    int hidden = recurrent->shape[2], directions = direction == 2 ? 2 : 1;
    if (hidden <= 0 || hidden > 1024 ||
        weight->shape[0] != directions || weight->shape[1] != 4 * hidden || weight->shape[2] != input->shape[2] ||
        recurrent->shape[0] != directions || recurrent->shape[1] != 4 * hidden ||
        output->shape[0] != input->shape[0] || output->shape[1] != directions ||
        output->shape[2] != input->shape[1] || output->shape[3] != hidden ||
        (bias && (bias->ndim != 2 || bias->shape[0] != directions || bias->shape[1] != 8 * hidden)) ||
        (h && (h->ndim != 3 || h->shape[0] != directions || h->shape[1] != input->shape[1] || h->shape[2] != hidden)) ||
        (c && (c->ndim != 3 || c->shape[0] != directions || c->shape[1] != input->shape[1] || c->shape[2] != hidden))) return 0;
    T* tensors[7] = {input, weight, recurrent, bias, h, c, output};
    for (unsigned i = 0; i < 7; ++i)
        if (tensors[i] && tensors[i]->dtype != T_F32) return 0;
    VxLstmPlan lstm;
    if (!vx_lstm_plan(input->data, weight->data, recurrent->data,
            bias ? bias->data : NULL, h ? h->data : NULL, c ? c->data : NULL,
            output->data, input->shape[0], input->shape[1], input->shape[2], hidden,
            (uint32_t)direction, &lstm)) return 0;
    /* WebGPU residency owns one exact byte extent per host pointer. A short
     * view of X for an absent optional port would replace its full buffer. */
    const uint32_t zero = 0u;
    for (unsigned i = 0; i < 6; ++i)
        if (!(tensors[i] ? vx_webgpu_bind(plan, lstm.inputs[i], lstm.bytes[i])
                         : vx_webgpu_bind_scratch(plan, &zero, 1u))) return 0;
    if (!vx_webgpu_bind(plan, output->data, lstm.bytes[6]) ||
        !vx_webgpu_params(plan, lstm.params, 8u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_LSTM;
    plan->variants[0].workgroup[0] = lstm.groups;
    plan->variants[0].workgroup[1] = plan->variants[0].workgroup[2] = 1u;
    plan->variant_count = 1u;
    return 1;
}

static int vx_webgpu_plan_packed_lstm(const Node* node, T* input, T* output, VxWebGpuNodePlan* plan) {
    T* weight = vx_webgpu_port(node, VX_PORT_WEIGHT);
    T* weight_scale = vx_webgpu_port(node, VX_PORT_SCALE);
    T* recurrent = vx_webgpu_port(node, VX_PORT_RECURRENT_WEIGHT);
    T* recurrent_scale = vx_webgpu_port(node, VX_PORT_RECURRENT_SCALE);
    T* bias = vx_webgpu_port(node, VX_PORT_BIAS);
    T* h = vx_webgpu_port(node, VX_PORT_INITIAL_H);
    T* c = vx_webgpu_port(node, VX_PORT_INITIAL_C);
    int direction = vx_node_param_i32(node, VX_NODE_PARAM_DIRECTION, 0);
    int group_size = vx_node_param_i32(node, VX_NODE_PARAM_GROUP_SIZE, 0);
    int weight_bits = vx_node_param_i32(node, VX_NODE_PARAM_WEIGHT_BITS, 4);
    if (!input || !output || !weight || !weight_scale || !recurrent || !recurrent_scale ||
        input->ndim != 3 || output->ndim != 4 || weight->ndim != 3 || recurrent->ndim != 3 ||
        direction < 0 || direction > 2 || !vx_packed_group_size_supported(group_size) ||
        (weight_bits != 2 && weight_bits != 3 && weight_bits != 4 && weight_bits != 8)) return 0;
    int hidden = recurrent->shape[1] / 4;
    int directions = direction == 2 ? 2 : 1;
    int time = input->shape[0], batch = input->shape[1], width = input->shape[2];
    int rows = directions * 4 * hidden;
    int w_words = (width / 32) * weight_bits + ((width % 32) * weight_bits + 31) / 32;
    int w_groups = width / group_size + (width % group_size != 0);
    int r_words = (hidden / 32) * weight_bits + ((hidden % 32) * weight_bits + 31) / 32;
    int r_groups = hidden / group_size + (hidden % group_size != 0);
    size_t w_elems = (size_t)rows * (size_t)width;
    size_t r_elems = (size_t)rows * (size_t)hidden;
    int idx = (int)(node - g_n);
    int can_cache = idx >= 0 && (size_t)idx < g_node_capacity;
    float* unpacked = can_cache ? g_qwcache[idx] : NULL;
    if (!unpacked) {
        unpacked = (float*)malloc((w_elems + r_elems) * sizeof(float));
        if (!unpacked) return 0;
        if (!vx_packed_lstm_unpack_bank(weight->data, (const float*)weight_scale->data,
                rows, width, w_words, w_groups, group_size, weight_bits, unpacked) ||
            !vx_packed_lstm_unpack_bank(recurrent->data, (const float*)recurrent_scale->data,
                rows, hidden, r_words, r_groups, group_size, weight_bits, unpacked + w_elems)) {
            free(unpacked);
            return 0;
        }
        if (can_cache) g_qwcache[idx] = unpacked;
    }
    VxLstmPlan lstm;
    if (!vx_lstm_plan(input->data, unpacked, unpacked + w_elems,
            bias ? bias->data : NULL, h ? h->data : NULL, c ? c->data : NULL,
            output->data, time, batch, width, hidden, (uint32_t)direction, &lstm)) {
        if (!can_cache) free(unpacked);
        return 0;
    }
    const uint32_t zero = 0u;
    T* opt_tensors[4] = {bias, h, c, NULL};
    for (unsigned i = 0; i < 6; ++i) {
        if (i < 3) {
            if (!vx_webgpu_bind(plan, lstm.inputs[i], lstm.bytes[i])) return 0;
        } else {
            T* opt = opt_tensors[i - 3];
            if (!(opt ? vx_webgpu_bind(plan, lstm.inputs[i], lstm.bytes[i])
                      : vx_webgpu_bind_scratch(plan, &zero, 1u))) return 0;
        }
    }
    if (!vx_webgpu_bind(plan, output->data, lstm.bytes[6]) ||
        !vx_webgpu_params(plan, lstm.params, 8u)) return 0;
    plan->variants[0].shader_id = VX_SHADER_INFERENCE_LSTM;
    plan->variants[0].workgroup[0] = lstm.groups;
    plan->variants[0].workgroup[1] = plan->variants[0].workgroup[2] = 1u;
    plan->variant_count = 1u;
    return 1;
}
