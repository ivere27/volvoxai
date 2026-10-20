/* Exercise the real typed row projector without a backend or legacy fallback.
 * Storage and ownership queries below are fixture plumbing; geometry is read
 * from application-shaped tensor/node descriptors by decode_projection.h. */
#include "engine_internal.h"
#include "incremental_runtime.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "decode_projection.h"
/* Link-time section collection keeps this fixture independent of engine
 * execution while exercising the planner's actual private prefix guard. */
#ifdef VX_PROJECTION_RUNTIME_SOURCE
#include VX_PROJECTION_RUNTIME_SOURCE
#else
#include "../src/runtime/incremental_runtime.c"
#endif

static VxEngineState state;
static T tensors[3];
static Node nodes[1];
static Ref inputs[3];
static int positions[3], empty[3], lengths[3];
static VxDecodeSlotPages pages[3];

VxEngineState* vx_engine_state_current(void) { return &state; }
uint64_t volvoxai_engine_model_generation_locked(void) { return state.model_generation; }
T* t_find(const char* name) {
    for (int i = 0; i < 3; i++)
        if (!strcmp(tensors[i].name, name)) return &tensors[i];
    return NULL;
}
int volvoxai_engine_tensor_is_model_weight_locked(const char* name) {
    return !strcmp(name, "weight");
}
int vx_paged_bound_locked(void) { return 0; }
int vx_paged_slot_pages_locked(int slot, VxDecodeSlotPages* output) {
    (void)slot; (void)output;
    return 0;
}

static void require(int value, const char* message) {
    if (!value) {
        fprintf(stderr, "projection regression: %s\n", message);
        exit(1);
    }
}

static void tensor(int index, const char* name, VxDataType dtype,
                    int rank, const int* shape) {
    T* value = &tensors[index];
    strcpy(value->name, name);
    value->dtype = dtype;
    value->ndim = rank;
    value->elem_size = dtype == T_I8 ? 1 : 4;
    value->numel = 1;
    for (int axis = 0; axis < rank; axis++) {
        value->shape[axis] = shape[axis];
        value->numel *= shape[axis];
    }
}

static Node* node(VxOperatorKind op, int slots, int scalar) {
    memset(&state, 0, sizeof(state));
    memset(tensors, 0, sizeof(tensors));
    memset(nodes, 0, sizeof(nodes));
    memset(inputs, 0, sizeof(inputs));
    memset(positions, 0, sizeof(positions));
    memset(empty, 0, sizeof(empty));
    memset(pages, 0, sizeof(pages));
    state.nodes = nodes;
    state.node_count = 1;
    state.decode_slots = scalar ? 0 : slots;
    for (int slot = 0; slot < slots; slot++) lengths[slot] = 1;
    state.decode_rows = (VxDecodeRowSet){
        .slots = slots, .live = slots, .positions = positions, .empty = empty,
        .kv_lengths = lengths, .pages = pages, .key_capacity = 1, .unpaged = 1,
    };
    Node* value = &nodes[0];
    value->operator_kind = op;
    value->ins = inputs;
    value->nin = value->nout = 1;
    strcpy(value->out, "out");
    inputs[0].port = VX_PORT_INPUT;
    strcpy(inputs[0].name, "input");
    return value;
}

static void argmax_ids(int batch, int sequence, int scalar, int quantized) {
    Node* value = node(quantized ? VX_OP_Q_ARG_MAX : VX_OP_ARG_MAX, batch, scalar);
    const int input_shape[] = {batch, sequence, 7};
    const int output_shape[] = {batch, sequence};
    tensor(0, "input", quantized ? T_I8 : T_F32, 3, input_shape);
    tensor(1, "out", T_I32, 2, output_shape);
    value->parsed_params.cache[VX_NODE_PARAM_AXIS].kind = VX_NODE_PARAM_I32;
    value->parsed_params.cache[VX_NODE_PARAM_AXIS].value.i32 = -1;
    value->parsed_params.cache[VX_NODE_PARAM_KEEPDIMS].kind = VX_NODE_PARAM_BOOL;
    value->parsed_params.cache[VX_NODE_PARAM_KEEPDIMS].value.i32 = 0;
    VxDecodeRowProjection projection;
    require(vx_decode_projection_describe(value, 0, &projection),
            "feature ArgMax must admit [B,S,V] -> I32[B,S]");
    require(projection.output.axis == 1 && projection.output.sequence == sequence &&
            projection.output.width == 1 && projection.input_count == 1 &&
            projection.inputs[0].axis == 1 && projection.inputs[0].width == 7,
            "token IDs retain their declared batch and sequence axes");
    /* Reducing over S cannot be implemented by projecting an individual token. */
    value->parsed_params.cache[VX_NODE_PARAM_AXIS].value.i32 = 1;
    const int reduced_shape[] = {batch, 7};
    tensor(1, "out", T_I32, 2, reduced_shape);
    require(!vx_decode_projection_describe(value, 0, &projection),
            "sequence reduction must remain outside row admission");
}

static void singleton_head_norm(int batch, int dimension, int scalar) {
    Node* value = node(VX_OP_RMS_NORM, batch, scalar);
    const int shape[] = {batch, 1, 4, dimension};
    const int affine_shape[] = {dimension};
    tensor(0, "input", T_F32, 4, shape);
    tensor(1, "out", T_F32, 4, shape);
    tensor(2, "weight", T_F32, 1, affine_shape);
    value->nin = 2;
    inputs[1].port = VX_PORT_WEIGHT;
    strcpy(inputs[1].name, "weight");
    VxDecodeRowProjection projection;
    require(vx_decode_projection_describe(value, 0, &projection),
            "singleton ordered head norm must retain one token");
    require(projection.output.axis == 1 && projection.output.sequence == 1 &&
            projection.output.width == (size_t)4 * dimension &&
            projection.input_count == 1 && projection.inputs[0].axis == 1 &&
            projection.inputs[0].width == (size_t)4 * dimension,
            "one token spans all independent last-D normalization rows");
}

static void singleton_id_prefix(void) {
    const int shape[] = {1, 1};
    node(VX_OP_ARG_MAX, 1, 1);
    tensor(1, "out", T_I32, 2, shape);
    size_t bytes = 123;
    require(hybrid_tensor_prefix_bytes(&tensors[1], 0, &bytes) == 0 && bytes == 0,
            "admitted singleton IDs have an empty row-zero prefix");
    bytes = 123;
    require(hybrid_tensor_prefix_bytes(&tensors[1], 1, &bytes) == -1 && bytes == 123,
            "a singleton token never acquires a second row");
    const int scalar_shape[] = {1, 1, 1};
    for (int dtype = 0; dtype < 2; dtype++) {
        tensor(1, "out", dtype ? T_I32 : T_F32, 3, scalar_shape);
        bytes = 123;
        require(hybrid_tensor_prefix_bytes(&tensors[1], 0, &bytes) == 0 && bytes == 0,
                "singleton gates and mask views retain an empty row-zero prefix");
        bytes = 123;
        require(hybrid_tensor_prefix_bytes(&tensors[1], 1, &bytes) == -1 && bytes == 123,
                "singleton scalar rows never acquire a second row");
    }
}

static void pointwise_token_ids(int batch,int sequence) {
    Node* value=node(VX_OP_WHERE,batch,0);
    const int shape[]={batch,sequence};
    tensor(0,"input",T_I32,2,shape);
    tensor(1,"out",T_I32,2,shape);
    tensor(2,"mask",T_I32,2,shape);
    value->nin=3;
    inputs[0].port=VX_PORT_A;
    inputs[1].port=VX_PORT_B;strcpy(inputs[1].name,"input");
    inputs[2].port=VX_PORT_CONDITION;strcpy(inputs[2].name,"mask");
    for(int row=0;row<sequence;row++) {
        for(int slot=0;slot<batch;slot++) positions[slot]=row;
        VxDecodeRowProjection p;
        require(vx_decode_projection_describe(value,row,&p),"pointwise I32 token rows remain in the retained domain");
        require(p.output.axis==1 && p.output.sequence==sequence && p.output.width==1 && p.input_count==2,
                "token selection preserves batch/sequence instead of treating all IDs as features");
        for(int i=0;i<p.input_count;i++)
            require(p.inputs[i].axis==1 && p.inputs[i].width==1,"token and condition each project one I32 per slot");
    }
}

int main(void) {
    for (int quantized = 0; quantized < 2; quantized++) {
        argmax_ids(1, 6, 1, quantized);
        argmax_ids(1, 6, 0, quantized);
        argmax_ids(1, 1, 1, quantized);
        argmax_ids(1, 1, 0, quantized);
        argmax_ids(3, 1, 0, quantized);
        argmax_ids(3, 6, 0, quantized);
    }
    for (int dimension = 64; dimension <= 128; dimension *= 2) {
        singleton_head_norm(1, dimension, 1);
        singleton_head_norm(1, dimension, 0);
        singleton_head_norm(3, dimension, 0);
    }
    singleton_id_prefix();
    pointwise_token_ids(1,1);
    pointwise_token_ids(1,6);
    pointwise_token_ids(3,6);
    puts("typed projection: 12 ID geometries, 12 sequence-refusal cases, 6 singleton head norms PASS");
    return 0;
}
