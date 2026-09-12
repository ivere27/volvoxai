#include "sequence_runtime.h"

#include "sequence_ops.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

static T* sequence_input(Node* node, VxPortKind key) {
    if (!node || !key) return NULL;
    for (int index = 0; index < node->nin; index++) {
        if (node->ins[index].port == key)
            return t_find(node->ins[index].name);
    }
    return NULL;
}

static T* sequence_output(Node* node, VxPortKind key) {
    if (!node || !key) return NULL;
    for (int index = 0; index < node->nout; index++) {
        if (node->outs[index].port == key)
            return t_find(node->outs[index].name);
    }
    return NULL;
}

static T* sequence_primary_output(Node* node) {
    T* output = sequence_output(node, VX_PORT_OUT);
    return output ? output : (node ? t_find(node->out) : NULL);
}

static int sequence_same_shape(const T* left, const T* right) {
    if (!left || !right || left->ndim != right->ndim ||
        left->numel != right->numel) return 0;
    for (int axis = 0; axis < left->ndim; axis++) {
        if (left->shape[axis] != right->shape[axis]) return 0;
    }
    return 1;
}

static int sequence_product(size_t* value, size_t factor) {
    if (!value || (factor && *value > SIZE_MAX / factor)) return 0;
    *value *= factor;
    return 1;
}

static int sequence_exact_bool(const Node* node, VxNodeParamKey key,
                               int default_value, int* output) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!output || !value) return 0;
    if (value->kind == VX_NODE_PARAM_ABSENT) {
        *output = default_value;
        return 1;
    }
    if (value->kind != VX_NODE_PARAM_BOOL) return 0;
    *output = value->value.i32 ? 1 : 0;
    return 1;
}

static int sequence_exact_int(const Node* node, VxNodeParamKey key,
                              int default_value, int* output) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!output || !value) return 0;
    if (value->kind == VX_NODE_PARAM_ABSENT) {
        *output = default_value;
        return 1;
    }
    if (value->kind != VX_NODE_PARAM_I32) return 0;
    *output = value->value.i32;
    return 1;
}

static int sequence_exact_float(const Node* node, VxNodeParamKey key,
                                float default_value, float* output) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!output || !value) return 0;
    if (value->kind == VX_NODE_PARAM_ABSENT) {
        *output = default_value;
        return 1;
    }
    if (value->kind != VX_NODE_PARAM_F32) return 0;
    *output = value->value.f32;
    return 1;
}

static int sequence_bc_mode(const T* tensor, int rank, int batch,
                            int sequence, int state_width) {
    if (!tensor || tensor->dtype != T_F32 || !tensor->data) return -1;
    if (tensor->ndim == 1 && tensor->shape[0] == state_width) return 0;
    if (tensor->ndim == 2 && tensor->shape[0] == sequence &&
        tensor->shape[1] == state_width) return 1;
    if (rank == 3 && tensor->ndim == 3 && tensor->shape[0] == batch &&
        tensor->shape[1] == sequence && tensor->shape[2] == state_width) return 2;
    return -1;
}

static int sequence_state_shape(const T* tensor, int batch, int channels,
                                int state_width) {
    size_t elements = (size_t)batch;
    if (!sequence_product(&elements, (size_t)channels) ||
        !sequence_product(&elements, (size_t)state_width)) return 0;
    return tensor && tensor->dtype == T_F32 && tensor->data &&
        tensor->ndim == 3 && tensor->shape[0] == batch &&
        tensor->shape[1] == channels && tensor->shape[2] == state_width &&
        (size_t)tensor->numel == elements;
}

static int sequence_unary(Node* node, long element_offset, int element_count,
                          int sine) {
    T* input = sequence_input(node, VX_PORT_INPUT);
    T* output = sequence_primary_output(node);
    if (!input) input = sequence_input(node, VX_PORT_X);
    if (!input || !output || input->dtype != T_F32 || output->dtype != T_F32 ||
        !input->data || !output->data || !sequence_same_shape(input, output) ||
        element_offset < 0 || element_count < 0 ||
        (size_t)element_offset > (size_t)input->numel ||
        (size_t)element_count > (size_t)input->numel - (size_t)element_offset)
        return VX_SEQUENCE_ERROR;
    materialize_tensor_f32(input);
    if (sine)
        sin_f32(input->data + element_offset, output->data + element_offset,
                (uint32_t)element_count);
    else
        cos_f32(input->data + element_offset, output->data + element_offset,
                (uint32_t)element_count);
    return VX_SEQUENCE_HANDLED;
}

static int sequence_rope(Node* node) {
    T* input = sequence_input(node, VX_PORT_INPUT);
    T* positions = sequence_input(node, VX_PORT_POSITION_IDS);
    T* output = sequence_primary_output(node);
    int rank, batch, sequence, width, rotary_width, position_offset, interleaved;
    int position_mode = 0;
    float theta;
    if (!input) input = sequence_input(node, VX_PORT_X);
    /*
     * Distinct buffers, stated rather than assumed.
     *
     * This recomputes the whole sequence on every call, including a row step
     * that changed one token -- which is correct only because a rotation of an
     * unchanged row reproduces that row. In place it would not: rope(rope(x))
     * is not rope(x), so each step would rotate every retained row again. The
     * scalar row path has always depended on this; a declared batch depends on
     * it for every lane at once, so it is worth refusing rather than trusting.
     */
    if (!input || !output || input->dtype != T_F32 || output->dtype != T_F32 ||
        !input->data || !output->data || input->data == output->data ||
        !sequence_same_shape(input, output) ||
        (input->ndim != 2 && input->ndim != 3)) return VX_SEQUENCE_ERROR;
    rank = input->ndim;
    batch = rank == 2 ? 1 : input->shape[0];
    sequence = input->shape[rank - 2];
    width = input->shape[rank - 1];
    if (!sequence_exact_int(node, VX_NODE_PARAM_ROTARY_DIM, width,
                            &rotary_width) ||
        !sequence_exact_float(node, VX_NODE_PARAM_THETA, 10000.0f, &theta) ||
        !sequence_exact_int(node, VX_NODE_PARAM_POSITION_OFFSET, 0,
                            &position_offset) ||
        !sequence_exact_bool(node, VX_NODE_PARAM_INTERLEAVED, 0,
                             &interleaved) ||
        batch <= 0 || sequence <= 0 || width <= 0 || rotary_width <= 0 ||
        rotary_width > width || (rotary_width & 1) || !isfinite(theta) ||
        theta <= 0.0f || position_offset < 0) return VX_SEQUENCE_ERROR;
    if (positions) {
        if (positions->dtype != T_I32 || !positions->data) return VX_SEQUENCE_ERROR;
        if (positions->ndim == 1 && positions->shape[0] == sequence)
            position_mode = 1;
        else if (rank == 3 && positions->ndim == 2 &&
                 positions->shape[0] == batch && positions->shape[1] == sequence)
            position_mode = 2;
        else
            return VX_SEQUENCE_ERROR;
    }
    materialize_tensor_f32(input);
    return rope_f32(input->data, positions ? positions->data : NULL,
                    output->data, (uint32_t)batch, (uint32_t)sequence,
                    (uint32_t)width, (uint32_t)rotary_width, theta,
                    position_offset, (uint32_t)interleaved,
                    (uint32_t)position_mode)
        ? VX_SEQUENCE_HANDLED : VX_SEQUENCE_ERROR;
}

static int sequence_scan(Node* node) {
    T* input = sequence_input(node, VX_PORT_INPUT);
    T* delta = sequence_input(node, VX_PORT_DELTA);
    T* a = sequence_input(node, VX_PORT_SSM_A);
    T* b = sequence_input(node, VX_PORT_SSM_B);
    T* c = sequence_input(node, VX_PORT_SSM_C);
    T* d = sequence_input(node, VX_PORT_SSM_D);
    T* z = sequence_input(node, VX_PORT_Z);
    T* initial_state = sequence_input(node, VX_PORT_INITIAL_STATE);
    T* output = sequence_primary_output(node);
    T* final_state = sequence_output(node, VX_PORT_STATE);
    int rank, batch, sequence, channels, state_width;
    int b_mode, c_mode, delta_softplus;
    size_t a_elements, state_elements, state_bytes;
    float* scratch;
    if (!input) input = sequence_input(node, VX_PORT_U);
    if (!a) a = sequence_input(node, VX_PORT_A);
    if (!b) b = sequence_input(node, VX_PORT_B);
    if (!c) c = sequence_input(node, VX_PORT_C);
    if (!d) d = sequence_input(node, VX_PORT_D);
    if (!final_state) final_state = sequence_output(node, VX_PORT_FINAL_STATE);
    if (!input || !delta || !a || !b || !c || !output ||
        input->dtype != T_F32 || delta->dtype != T_F32 || a->dtype != T_F32 ||
        output->dtype != T_F32 || !input->data || !delta->data || !a->data ||
        !output->data || (input->ndim != 2 && input->ndim != 3) ||
        !sequence_same_shape(input, delta) || !sequence_same_shape(input, output))
        return VX_SEQUENCE_ERROR;
    rank = input->ndim;
    batch = rank == 2 ? 1 : input->shape[0];
    sequence = input->shape[rank - 2];
    channels = input->shape[rank - 1];
    state_width = a->ndim == 2 ? a->shape[1] : 0;
    b_mode = sequence_bc_mode(b, rank, batch, sequence, state_width);
    c_mode = sequence_bc_mode(c, rank, batch, sequence, state_width);
    a_elements = (size_t)channels;
    if (batch <= 0 || sequence <= 0 || channels <= 0 || state_width <= 0 ||
        !sequence_product(&a_elements, (size_t)state_width) ||
        a->shape[0] != channels || a->numel < 0 ||
        (size_t)a->numel != a_elements ||
        b_mode < 0 || c_mode < 0 ||
        (d && (d->dtype != T_F32 || !d->data || d->ndim != 1 ||
               d->shape[0] != channels)) ||
        (z && (z->dtype != T_F32 || !z->data || !sequence_same_shape(z, input))) ||
        (initial_state && !sequence_state_shape(initial_state, batch, channels,
                                                state_width)) ||
        (final_state && !sequence_state_shape(final_state, batch, channels,
                                              state_width)) ||
        !sequence_exact_bool(node, VX_NODE_PARAM_DELTA_SOFTPLUS, 1,
                             &delta_softplus)) return VX_SEQUENCE_ERROR;
    state_elements = (size_t)batch;
    if (!sequence_product(&state_elements, (size_t)channels) ||
        !sequence_product(&state_elements, (size_t)state_width) ||
        !sequence_product(&state_elements, sizeof(float))) return VX_SEQUENCE_ERROR;
    state_bytes = state_elements;

    materialize_tensor_f32(input);
    materialize_tensor_f32(delta);
    materialize_tensor_f32(a);
    materialize_tensor_f32(b);
    materialize_tensor_f32(c);
    materialize_tensor_f32(d);
    materialize_tensor_f32(z);
    materialize_tensor_f32(initial_state);
    scratch = (float*)malloc(state_bytes);
    if (!scratch) return VX_SEQUENCE_ERROR;
    int result = ssm_scan_f32(input->data, delta->data, a->data, b->data,
                              c->data, d ? d->data : NULL, z ? z->data : NULL,
                              initial_state ? initial_state->data : NULL,
                              output->data, final_state ? final_state->data : NULL,
                              scratch, (uint32_t)batch, (uint32_t)sequence,
                              (uint32_t)channels, (uint32_t)state_width,
                              (uint32_t)b_mode, (uint32_t)c_mode,
                              (uint32_t)delta_softplus);
    free(scratch);
    if (!result) return VX_SEQUENCE_ERROR;
    if (final_state) {
        vx_runtime_backend_mark_host(final_state->data,
            state_bytes,
            !final_state->owns);
    }
    return VX_SEQUENCE_HANDLED;
}

int vx_sequence_node_run(Node* node, long element_offset, int element_count,
                         const char** backend_name) {
    if (!node) return VX_SEQUENCE_NOT_HANDLED;
    if (node->operator_kind == VX_OP_SIN) {
        if (backend_name) *backend_name = "cpu";
        return sequence_unary(node, element_offset, element_count, 1);
    }
    if (node->operator_kind == VX_OP_COS) {
        if (backend_name) *backend_name = "cpu";
        return sequence_unary(node, element_offset, element_count, 0);
    }
    if (node->operator_kind == VX_OP_ROPE) {
        if (backend_name) *backend_name = "cpu-rope";
        return sequence_rope(node);
    }
    if ((node->operator_kind == VX_OP_SSM_SCAN) || (node->operator_kind == VX_OP_SELECTIVE_SCAN)) {
        if (backend_name) *backend_name = "cpu-ssm-scan";
        return sequence_scan(node);
    }
    return VX_SEQUENCE_NOT_HANDLED;
}
