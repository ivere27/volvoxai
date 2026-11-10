#include "../backends/lstm_plan.h"

typedef int (*VxLstmF32Fn)(const float*, const float*, const float*, const float*,
    const float*, const float*, float*, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t);

static int lstm_node_forward(Node* node, T* output, VxLstmF32Fn kernel) {
    if (!node || node->operator_kind != VX_OP_LSTM || !kernel ||
        g_active_row >= 0 || g_prefix_rows > 0 || g_execution_row >= 0) return 0;
    T* x = nin(node, VX_PORT_INPUT);
    T* w = nin(node, VX_PORT_WEIGHT);
    T* r = nin(node, VX_PORT_RECURRENT_WEIGHT);
    T* b = nin(node, VX_PORT_BIAS);
    T* h = nin(node, VX_PORT_INITIAL_H);
    T* c = nin(node, VX_PORT_INITIAL_C);
    int direction = vx_node_param_i32(node, VX_NODE_PARAM_DIRECTION, 0);
    if (!x || !w || !r || !output || x->ndim != 3 || w->ndim != 3 || r->ndim != 3 ||
        output->ndim != 4 || direction < 0 || direction > 2) return 0;
    int time = x->shape[0], batch = x->shape[1], width = x->shape[2];
    int hidden = r->shape[2], directions = direction == 2 ? 2 : 1;
    if (time <= 0 || batch <= 0 || width <= 0 || hidden <= 0 || hidden > INT_MAX / 8 ||
        w->shape[0] != directions || w->shape[1] != 4 * hidden || w->shape[2] != width ||
        r->shape[0] != directions || r->shape[1] != 4 * hidden ||
        output->shape[0] != time || output->shape[1] != directions ||
        output->shape[2] != batch || output->shape[3] != hidden ||
        (b && (b->ndim != 2 || b->shape[0] != directions || b->shape[1] != 8 * hidden)) ||
        (h && (h->ndim != 3 || h->shape[0] != directions || h->shape[1] != batch || h->shape[2] != hidden)) ||
        (c && (c->ndim != 3 || c->shape[0] != directions || c->shape[1] != batch || c->shape[2] != hidden))) return 0;
    T* tensors[] = {x, w, r, b, h, c, output};
    for (unsigned i = 0; i < sizeof(tensors) / sizeof(tensors[0]); ++i) {
        if (!tensors[i]) continue;
        materialize_tensor_f32(tensors[i]);
        if (tensors[i]->dtype != T_F32) return 0;
    }
#if defined(__wasm__)
    if (kernel == lstm_f32 && vx_engine_state_current()->gemm_f32_packed_enabled) {
        const VxPreparedF32Weights* store = vx_engine_state_current()->prepared_f32_weights;
        const float* packed_w[2] = {NULL, NULL};
        const float* packed_r[2] = {NULL, NULL};
        for (int d = 0; d < directions; d++) {
            if (volvoxai_engine_tensor_is_model_weight_locked(w->name))
                packed_w[d] = vx_prepared_f32_weights_find(store,
                    w->data + (size_t)d * 4u * hidden * width,
                    (uint32_t)width, (uint32_t)(4u * hidden), 1);
            if (volvoxai_engine_tensor_is_model_weight_locked(r->name))
                packed_r[d] = vx_prepared_f32_weights_find(store,
                    r->data + (size_t)d * 4u * hidden * hidden,
                    (uint32_t)hidden, (uint32_t)(4u * hidden), 1);
        }
        return lstm_f32_packed(x->data, w->data, r->data, b ? b->data : NULL,
            h ? h->data : NULL, c ? c->data : NULL, output->data,
            (uint32_t)time, (uint32_t)batch, (uint32_t)width,
            (uint32_t)hidden, (uint32_t)direction, packed_w, packed_r);
    }
#endif
    return kernel(x->data, w->data, r->data, b ? b->data : NULL,
        h ? h->data : NULL, c ? c->data : NULL, output->data,
        (uint32_t)time, (uint32_t)batch, (uint32_t)width, (uint32_t)hidden, (uint32_t)direction);
}

static int packed_lstm_unpack_bank(const T* packed, const T* scale,
                                   int rows, int k, int words, int groups,
                                   int group_size, int weight_bits, float* dst) {
    if (!packed || !scale || !dst) return 0;
    return vx_packed_lstm_unpack_bank(packed->data, (const float*)scale->data,
                                      rows, k, words, groups, group_size, weight_bits, dst);
}

static int packed_lstm_node_forward(Node* node, int idx, T* output, VxLstmF32Fn kernel) {
    if (!node || node->operator_kind != VX_OP_PACKED_LSTM || !kernel ||
        g_active_row >= 0 || g_prefix_rows > 0 || g_execution_row >= 0) return 0;
    T* x = nin(node, VX_PORT_INPUT);
    T* w = nin(node, VX_PORT_WEIGHT);
    T* ws = nin(node, VX_PORT_SCALE);
    T* r = nin(node, VX_PORT_RECURRENT_WEIGHT);
    T* rs = nin(node, VX_PORT_RECURRENT_SCALE);
    T* b = nin(node, VX_PORT_BIAS);
    T* h = nin(node, VX_PORT_INITIAL_H);
    T* c = nin(node, VX_PORT_INITIAL_C);
    int direction = vx_node_param_i32(node, VX_NODE_PARAM_DIRECTION, 0);
    int group_size = vx_node_param_i32(node, VX_NODE_PARAM_GROUP_SIZE, 0);
    int weight_bits = vx_node_param_i32(node, VX_NODE_PARAM_WEIGHT_BITS, 4);
    if (!x || !w || !ws || !r || !rs || !output ||
        x->ndim != 3 || w->ndim != 3 || ws->ndim != 3 ||
        r->ndim != 3 || rs->ndim != 3 || output->ndim != 4 ||
        w->dtype != T_I32 || r->dtype != T_I32 ||
        w->quantization.valid || r->quantization.valid ||
        !w->data || !ws->data || !r->data || !rs->data ||
        direction < 0 || direction > 2 ||
        !vx_packed_group_size_supported(group_size) ||
        (weight_bits != 2 && weight_bits != 3 && weight_bits != 4 && weight_bits != 8))
        return 0;
    int time = x->shape[0], batch = x->shape[1], width = x->shape[2];
    int gates_hidden = r->shape[1], directions = direction == 2 ? 2 : 1;
    if (time <= 0 || batch <= 0 || width <= 0 || gates_hidden < 4 || (gates_hidden % 4) != 0)
        return 0;
    int hidden = gates_hidden / 4;
    if (hidden > INT_MAX / 8) return 0;
    int w_words = (width / 32) * weight_bits + ((width % 32) * weight_bits + 31) / 32;
    int w_groups = width / group_size + (width % group_size != 0);
    int r_words = (hidden / 32) * weight_bits + ((hidden % 32) * weight_bits + 31) / 32;
    int r_groups = hidden / group_size + (hidden % group_size != 0);
    if (w->shape[0] != directions || w->shape[1] != gates_hidden || w->shape[2] != w_words ||
        ws->shape[0] != directions || ws->shape[1] != gates_hidden || ws->shape[2] != w_groups ||
        r->shape[0] != directions || r->shape[1] != gates_hidden || r->shape[2] != r_words ||
        rs->shape[0] != directions || rs->shape[1] != gates_hidden || rs->shape[2] != r_groups ||
        output->shape[0] != time || output->shape[1] != directions ||
        output->shape[2] != batch || output->shape[3] != hidden ||
        (b && (b->ndim != 2 || b->shape[0] != directions || b->shape[1] != 8 * hidden)) ||
        (h && (h->ndim != 3 || h->shape[0] != directions || h->shape[1] != batch || h->shape[2] != hidden)) ||
        (c && (c->ndim != 3 || c->shape[0] != directions || c->shape[1] != batch || c->shape[2] != hidden)))
        return 0;
    T* f32_tensors[] = {x, ws, rs, b, h, c, output};
    for (unsigned i = 0; i < sizeof(f32_tensors) / sizeof(f32_tensors[0]); ++i) {
        if (!f32_tensors[i]) continue;
        materialize_tensor_f32(f32_tensors[i]);
        if (f32_tensors[i]->dtype != T_F32 || f32_tensors[i]->quantization.valid) return 0;
    }
    int rows = directions * gates_hidden;
    size_t w_elems = (size_t)rows * (size_t)width;
    size_t r_elems = (size_t)rows * (size_t)hidden;
    if (w_elems > SIZE_MAX / sizeof(float) || r_elems > (SIZE_MAX / sizeof(float)) - w_elems)
        return 0;
    int can_cache = idx >= 0 && (size_t)idx < g_node_capacity &&
        volvoxai_engine_tensor_is_model_weight_locked(w->name) &&
        volvoxai_engine_tensor_is_model_weight_locked(ws->name) &&
        volvoxai_engine_tensor_is_model_weight_locked(r->name) &&
        volvoxai_engine_tensor_is_model_weight_locked(rs->name);
    float* unpacked = can_cache ? g_qwcache[idx] : NULL;
    if (!unpacked) {
        unpacked = (float*)malloc((w_elems + r_elems) * sizeof(float));
        if (!unpacked) return 0;
        if (!packed_lstm_unpack_bank(w, ws, rows, width, w_words, w_groups, group_size, weight_bits, unpacked) ||
            !packed_lstm_unpack_bank(r, rs, rows, hidden, r_words, r_groups, group_size, weight_bits, unpacked + w_elems)) {
            free(unpacked);
            return 0;
        }
        if (can_cache) g_qwcache[idx] = unpacked;
    }
    int status = kernel(x->data, unpacked, unpacked + w_elems, b ? b->data : NULL,
        h ? h->data : NULL, c ? c->data : NULL, output->data,
        (uint32_t)time, (uint32_t)batch, (uint32_t)width, (uint32_t)hidden, (uint32_t)direction);
    if (!can_cache) free(unpacked);
    return status;
}
