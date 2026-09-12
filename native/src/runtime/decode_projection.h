/* Geometry shared by CPU and WebGPU row execution. No allocation, command
 * encoding or numerical kernel belongs in this projection. */
#ifndef VOLVOXAI_DECODE_PROJECTION_H
#define VOLVOXAI_DECODE_PROJECTION_H

/* Row execution is a storage projection of an ordinary C numerical plan.
 * The dependency planner selects the changed closure. This file never asks
 * an invariant encoder branch to support row execution. */
#include "paged_binding.h"
#include "attention_mask.h"

typedef struct {
    T* tensor;
    int axis;
    int sequence;
    size_t width;
    int prefix;
} VxDecodeRowOperand;

typedef struct {
    VxDecodeRowSet rows;
    int scalar_position, scalar_parked, scalar_length;
    VxDecodeLanePages scalar_pages;
    VxDecodeRowOperand output;
    // The supported row operators have at most three varying operands:
    // attention Q/K/V or Where condition/a/b. Other ports are invariant.
    VxDecodeRowOperand inputs[3];
    int input_count;
    int view;
    int attention;
    int causal;
    T* mask;
    int mask_layout;
    int keys;
} VxDecodeRowProjection;

static T* vx_decode_projection_port(const Node* node, VxPortKind port) {
    for (int i = 0; i < node->nin; i++)
        if (node->ins[i].port == port) return t_find(node->ins[i].name);
    return NULL;
}

static int vx_decode_projection_broadcast(const T* input, const T* output, uint32_t* strides) {
    (void)strides;
    if (input->ndim < 0 || output->ndim > 8 || input->ndim > output->ndim) return 0;
    int offset = output->ndim - input->ndim;
    for (int i = 0; i < input->ndim; i++)
        if (input->shape[i] != 1 && input->shape[i] != output->shape[offset + i]) return 0;
    return 1;
}

static int vx_decode_projection_pointwise(const Node* node) {
    VxOperatorKind op = node->operator_kind;
    return (op == VX_OP_RELU) || (op == VX_OP_LEAKY_RELU) || (op == VX_OP_GELU) ||
        (op == VX_OP_SILU) || (op == VX_OP_SIGMOID) || (op == VX_OP_TANH) ||
        (op == VX_OP_SIN) || (op == VX_OP_COS) || (op == VX_OP_HARD_SIGMOID) ||
        (op == VX_OP_HARD_SWISH) || (op == VX_OP_CLIP);
}

static int vx_decode_projection_layout(T* tensor, int lanes, int sequence,
                                VxDecodeRowOperand* layout) {
    if (!tensor || !layout || tensor->ndim < 1 || !tensor->elem_size) return 0;
    int axis = 0;
    if (lanes > 1) {
        if (tensor->ndim < 2 || tensor->shape[0] != lanes) return 0;
        axis = 1;
    } else {
        while (axis < tensor->ndim && tensor->shape[axis] == 1) axis++;
    }
    if (axis == tensor->ndim || tensor->shape[axis] <= 0 ||
        (sequence && tensor->shape[axis] != sequence)) return 0;
    size_t width = 1;
    for (int i = axis + 1; i < tensor->ndim; i++) {
        if (tensor->shape[i] <= 0 || width > UINT32_MAX / (uint32_t)tensor->shape[i]) return 0;
        width *= (uint32_t)tensor->shape[i];
    }
    if (width > UINT32_MAX / tensor->elem_size ||
        (uint64_t)lanes * tensor->shape[axis] * width != (uint64_t)tensor->numel) return 0;
    *layout = (VxDecodeRowOperand){tensor, axis, tensor->shape[axis], width, 0};
    return 1;
}

static int vx_decode_projection_invariant(const Node* node, const T* tensor) {
    return tensor && (volvoxai_engine_tensor_is_model_weight_locked(tensor->name) ||
        !vx_incremental_tensor_dirty_before_node_locked(tensor, (int)(node - g_n)));
}

static int vx_decode_projection_add_input(VxDecodeRowProjection* p, T* tensor,
                                    int sequence, int prefix) {
    for (int i = 0; i < p->input_count; i++)
        if (p->inputs[i].tensor == tensor) return p->inputs[i].prefix == prefix;
    if (p->input_count == 3 ||
        !vx_decode_projection_layout(tensor, p->rows.lanes, sequence, &p->inputs[p->input_count])) return 0;
    p->inputs[p->input_count++].prefix = prefix;
    return 1;
}

static int vx_decode_projection_view(const Node* node, const VxDecodeRowOperand* input,
                               const VxDecodeRowOperand* output) {
    if (input->tensor->dtype != output->tensor->dtype || input->width != output->width ||
        input->tensor->numel != output->tensor->numel) return 0;
    if (node->operator_kind != VX_OP_TRANSPOSE) return 1;
    const T* t = input->tensor;
    if (t->ndim != output->tensor->ndim) return 0;
    int previous = -1;
    for (int axis = 0; axis < t->ndim; axis++) {
        int source = node->parsed_params.has_perm ? node->parsed_params.perm[axis] : t->ndim - 1 - axis;
        if (source < 0 || source >= t->ndim ||
            output->tensor->shape[axis] != t->shape[source]) return 0;
        if (t->shape[source] == 1) continue;
        if (source <= previous) return 0;
        previous = source;
    }
    return 1;
}

static int vx_decode_projection_describe(const Node* node, int row, VxDecodeRowProjection* p) {
    if (!node || node->nout != 1 || row < 0) return 0;
    memset(p, 0, sizeof(*p));
    if (g_decode_lanes > 0) p->rows = g_decode_rows;
    else {
        p->scalar_position = row;
        p->scalar_length = row + 1;
        p->rows.lanes = p->rows.live = 1;
        p->rows.key_capacity = row + 1;
        p->rows.positions = &p->scalar_position;
        p->rows.parked = &p->scalar_parked;
        p->rows.kv_lengths = &p->scalar_length;
        p->rows.pages = &p->scalar_pages;
        p->rows.unpaged = !vx_paged_bound_locked();
        if (!p->rows.unpaged && vx_paged_lane_pages_locked(0, &p->scalar_pages) != 1) return 0;
    }
    if (!vx_decode_projection_layout(t_find(node->out), p->rows.lanes, 0, &p->output)) return 0;
    for (int lane = 0; lane < p->rows.lanes; lane++)
        if (!p->rows.parked[lane] && p->rows.positions[lane] >= p->output.sequence) return 0;
    VxOperatorKind op = node->operator_kind;
    int sequence = p->output.sequence;
    p->attention = (op == VX_OP_Q_SDPA) || (op == VX_OP_CROSS_SDPA);
    if (p->attention) {
        T* q = vx_decode_projection_port(node, VX_PORT_Q);
        T* k = vx_decode_projection_port(node, VX_PORT_K);
        T* v = vx_decode_projection_port(node, VX_PORT_V);
        p->causal = vx_node_param_bool(node, VX_NODE_PARAM_CAUSAL, 0);
        if (!q || !k || !v || (q->ndim != 2 && q->ndim != 3) ||
            !vx_decode_projection_add_input(p, q, sequence, 0)) return 0;
        if (p->causal) {
            if (!vx_decode_projection_add_input(p, k, sequence, 1) ||
                !vx_decode_projection_add_input(p, v, sequence, 1)) return 0;
        } else if (!vx_decode_projection_invariant(node, k) || !vx_decode_projection_invariant(node, v)) return 0;
        if (k->ndim != q->ndim || v->ndim != q->ndim ||
            k->shape[k->ndim - 2] != v->shape[v->ndim - 2]) return 0;
        p->keys = p->causal ? p->rows.key_capacity : k->shape[k->ndim - 2];
        p->mask = vx_decode_projection_port(node, VX_PORT_MASK);
        p->mask_layout = attention_mask_mode(p->mask, p->rows.lanes, sequence, k->shape[k->ndim - 2]);
        if (p->mask_layout < 0 || (!p->causal && p->mask && !vx_decode_projection_invariant(node, p->mask))) return 0;
        return 1;
    }
    int dense = (op == VX_OP_LINEAR) || (op == VX_OP_MATMUL) || (op == VX_OP_GEMM) ||
        (op == VX_OP_Q_LINEAR) || (op == VX_OP_Q_MATMUL) || (op == VX_OP_Q_GEMM);
    int bmm = (op == VX_OP_BATCH_MATMUL) || (op == VX_OP_Q_BATCH_MATMUL);
    int embedding = (op == VX_OP_EMBEDDING) || (op == VX_OP_Q_EMBEDDING);
    int nary = (op == VX_OP_ADD) || (op == VX_OP_MUL) || (op == VX_OP_SUB) ||
        (op == VX_OP_DIV) || (op == VX_OP_Q_ADD) || (op == VX_OP_WHERE) ||
        (op == VX_OP_EQUAL) || (op == VX_OP_GREATER_OR_EQUAL);
    int expand = (op == VX_OP_EXPAND);
    int norm = (op == VX_OP_LAYER_NORM) || (op == VX_OP_RMS_NORM) || (op == VX_OP_Q_LAYER_NORM) ||
        (op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX);
    int argmax = (op == VX_OP_Q_ARG_MAX) || (op == VX_OP_ARG_MAX);
    p->view = (op == VX_OP_RESHAPE) || (op == VX_OP_FLATTEN) || (op == VX_OP_SQUEEZE) ||
        (op == VX_OP_UNSQUEEZE) || (op == VX_OP_IDENTITY) || (op == VX_OP_TRANSPOSE);
    int unary = vx_decode_projection_pointwise(node) || (op == VX_OP_Q_SILU) || (op == VX_OP_Q_GELU) ||
        (op == VX_OP_QUANTIZE_LINEAR) || (op == VX_OP_DEQUANTIZE_LINEAR);
    if (!dense && !bmm && !embedding && !nary && !expand && !norm && !argmax && !p->view && !unary) return 0;
    T* primary = vx_decode_projection_port(node, VX_PORT_INPUT);
    if (!primary) primary = vx_decode_projection_port(node, VX_PORT_X);
    if (!primary) primary = vx_decode_projection_port(node, VX_PORT_DATA);
    if (!primary) primary = vx_decode_projection_port(node, VX_PORT_A);
    if (bmm) primary = vx_decode_projection_port(node, VX_PORT_A);
    for (int port = 0; port < node->nin; port++) {
        T* tensor = t_find(node->ins[port].name);
        if (!tensor) return 0;
        int activation = nary || tensor == primary;
        if (!activation) {
            if (!vx_decode_projection_invariant(node, tensor)) return 0;
            continue;
        }
        VxDecodeRowOperand layout;
        if ((nary || expand) && !vx_decode_projection_layout(tensor, p->rows.lanes, sequence, &layout)) {
            uint32_t strides[8];
            T narrow = *p->output.tensor;
            narrow.shape[p->output.axis] = 1;
            if (!vx_decode_projection_invariant(node, tensor) || !vx_decode_projection_broadcast(tensor, &narrow, strides)) return 0;
            continue;
        }
        if (!vx_decode_projection_add_input(p, tensor, sequence, 0)) return 0;
    }
    if (!p->input_count && !expand && !nary) return 0;
    if (p->view && (p->input_count != 1 || !vx_decode_projection_view(node, &p->inputs[0], &p->output))) return 0;
    if (dense || norm || bmm) {
        if (p->output.axis >= p->output.tensor->ndim - 1 ||
            p->output.width != (size_t)p->output.tensor->shape[p->output.tensor->ndim - 1]) return 0;
        if (!p->input_count || p->inputs[0].width != (size_t)primary->shape[primary->ndim - 1]) return 0;
    }
    if (norm || argmax) {
        int axis = vx_node_param_i32(node, VX_NODE_PARAM_AXIS, -1);
        if (axis < 0) axis += primary->ndim;
        if (((op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX) || argmax) &&
            axis <= p->inputs[0].axis) return 0;
    }
    return 1;
}

#endif
