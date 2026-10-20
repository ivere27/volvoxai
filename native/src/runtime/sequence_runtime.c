#include "sequence_runtime.h"

#include "sequence_ops.h"

#include <math.h>
#include <limits.h>
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

static int32_t sequence_position_at(const T* positions, size_t index) {
    int32_t value;
    memcpy(&value, (const unsigned char*)positions->data + index * sizeof(value),
           sizeof(value));
    return value;
}

int vx_sequence_rope_parameters(Node* node, const T* input, const T* output,
                                VxSequenceRoPE* p) {
    if (!p) return 0;
    memset(p, 0, sizeof(*p));
    p->positions = sequence_input(node, VX_PORT_POSITION_IDS);
    /* A retained token must never be rotated again through an aliased buffer. */
    if (!input || !output || input->dtype != T_F32 || output->dtype != T_F32 ||
        !input->data || !output->data || input->data == output->data ||
        !sequence_same_shape(input, output) ||
        (input->ndim != 2 && input->ndim != 3)) return 0;
    p->batch = input->ndim == 2 ? 1 : input->shape[0];
    p->sequence = input->shape[input->ndim - 2];
    p->width = input->shape[input->ndim - 1];
    if (!sequence_exact_int(node, VX_NODE_PARAM_HEADS, 1, &p->heads) ||
        p->heads <= 0 || p->width <= 0 || p->width % p->heads ||
        !sequence_exact_int(node, VX_NODE_PARAM_ROTARY_DIM, p->width / p->heads,
                            &p->rotary_width) ||
        !sequence_exact_float(node, VX_NODE_PARAM_THETA, 10000.0f, &p->theta) ||
        !sequence_exact_int(node, VX_NODE_PARAM_POSITION_OFFSET, 0,
                            &p->position_offset) ||
        !sequence_exact_bool(node, VX_NODE_PARAM_INTERLEAVED, 0,
                             &p->interleaved) ||
        p->batch <= 0 || p->sequence <= 0 || p->rotary_width <= 0 ||
        p->rotary_width > p->width / p->heads || (p->rotary_width & 1) ||
        !isfinite(p->theta) || p->theta <= 0.0f || p->position_offset < 0 ||
        p->position_offset > INT32_MAX - (p->sequence - 1)) return 0;
    size_t elements = (size_t)p->batch;
    if (!sequence_product(&elements, (size_t)p->sequence) ||
        !sequence_product(&elements, (size_t)p->width) || input->numel <= 0 ||
        (size_t)input->numel != elements || elements > SIZE_MAX / sizeof(float)) return 0;
    T* positions = p->positions;
    if (positions) {
        if (positions->dtype != T_I32 || !positions->data) return 0;
        if (positions->ndim == 1 && positions->shape[0] == p->sequence)
            p->position_mode = 1;
        else if (input->ndim == 3 && positions->ndim == 2 &&
                 positions->shape[0] == p->batch && positions->shape[1] == p->sequence)
            p->position_mode = 2;
        else
            return 0;
        size_t count = p->position_mode == 1 ? (size_t)p->sequence
            : (size_t)p->batch * p->sequence;
        if (positions->numel <= 0 || (size_t)positions->numel != count ||
            count > SIZE_MAX / sizeof(int32_t)) return 0;
        /* Validate the entire supplied list, including inactive and retained
         * tokens, before either full or row execution writes an output. */
        for (size_t index = 0; index < count; index++)
            if (sequence_position_at(positions, index) < 0) return 0;
    }
    return 1;
}

static int sequence_rope(Node* node) {
    T* input = sequence_input(node, VX_PORT_INPUT);
    T* output = sequence_primary_output(node);
    VxSequenceRoPE p;
    if (!input) input = sequence_input(node, VX_PORT_X);
    if (!vx_sequence_rope_parameters(node, input, output, &p)) return VX_SEQUENCE_ERROR;
    materialize_tensor_f32(input);
    return vx_rope_heads_f32(input->data, p.positions ? p.positions->data : NULL,
                    output->data, (uint32_t)p.batch, (uint32_t)p.sequence,
                    (uint32_t)p.width, (uint32_t)p.heads, (uint32_t)p.rotary_width,
                    p.theta, p.position_offset, (uint32_t)p.interleaved,
                    (uint32_t)p.position_mode)
        ? VX_SEQUENCE_HANDLED : VX_SEQUENCE_ERROR;
}

int vx_sequence_rope_row(Node* node, const T* input, const T* output,
                         const VxDecodeRowSet* rows, const float* input_rows,
                         float* output_rows) {
    VxSequenceRoPE p;
    if (!node || node->operator_kind != VX_OP_ROPE || !rows || !input_rows ||
        !output_rows || !rows->positions || !rows->empty || rows->slots <= 0 ||
        !vx_sequence_rope_parameters(node, input, output, &p) ||
        rows->slots != p.batch || (size_t)rows->slots > SIZE_MAX / sizeof(int32_t))
        return VX_SEQUENCE_ERROR;
    int32_t* selected = malloc((size_t)rows->slots * sizeof(int32_t));
    if (!selected) return VX_SEQUENCE_ERROR;
    for (int slot = 0; slot < rows->slots; slot++) {
        int position = rows->positions[slot];
        if (rows->empty[slot]) { selected[slot] = 0; continue; }
        if (position < 0 || position >= p.sequence) {
            free(selected);
            return VX_SEQUENCE_ERROR;
        }
        if (p.position_mode == 1) selected[slot] = sequence_position_at(p.positions, (size_t)position);
        else if (p.position_mode == 2) selected[slot] = sequence_position_at(p.positions,
            (size_t)slot * p.sequence + position);
        else selected[slot] = p.position_offset + position;
    }
    int result = vx_rope_heads_f32(input_rows, selected, output_rows,
        (uint32_t)rows->slots, 1u, (uint32_t)p.width, (uint32_t)p.heads,
        (uint32_t)p.rotary_width, p.theta, 0, (uint32_t)p.interleaved, 2u);
    free(selected);
    return result ? VX_SEQUENCE_HANDLED : VX_SEQUENCE_ERROR;
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
