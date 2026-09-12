#include "generated/operator_param_registry.h"
/* See ptq_authoring.h for what this step is and why it moved here.
 *
 * The rewrite in one paragraph: qualify each node, collect every activation
 * tensor the qualified nodes read or write, give each one an affine, replace
 * the nodes with their byte-domain equivalents, and close the resulting island
 * with a QuantizeLinear wherever float enters and a DequantizeLinear wherever
 * it leaves. What comes out is a graph whose interior is entirely bytes and
 * whose boundary is unchanged, which is what makes the package a drop-in
 * replacement for the float one.
 *
 * Nothing is inferred from values. Which nodes qualify is a function of their
 * operator, their ports, and the configuration — never of what the weights
 * happen to contain — so authoring the same graph twice gives the same answer
 * on any machine, which is the property the golden test in
 * python/tests/test_ptq_golden.py depends on.
 */
#include "ptq_authoring.h"
#include "../generated/operator_vocabulary.h"

#include "cJSON.h"
#include "safetensors.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VX_PTQ_MAX_RANK 8
#define VX_PTQ_MAX_ACTIVATION_PORTS 4

/* --- what an operator becomes ---------------------------------------------
 *
 * A dense operator carries a weight that gets packed, so it needs a layer
 * entry and per-channel scales. A byte operator keeps its operands and only
 * changes the domain its activations live in.
 *
 * The attention, convolution and embedding families are matched by the Python
 * authoring pass and not yet by this one. They are refused by name rather
 * than quietly retained: a graph that authors differently here than there is
 * worse than one that refuses, because the difference surfaces as an accuracy
 * gap long after the call that caused it. */

typedef struct VxPtqDenseOp {
    VxOperatorKind source;
    VxOperatorKind quantized;
} VxPtqDenseOp;

static const VxPtqDenseOp VX_PTQ_DENSE_OPS[] = {
    { VX_OP_LINEAR, VX_OP_Q_LINEAR },
    { VX_OP_MATMUL, VX_OP_Q_LINEAR },
};

typedef struct VxPtqByteOp {
    VxOperatorKind source;
    VxOperatorKind quantized;
    /* Input ports carrying activations. Everything else is left alone: a
     * QLayerNorm still reads its F32 scale and bias. */
    const char* activation_ports[VX_PTQ_MAX_ACTIVATION_PORTS];
} VxPtqByteOp;

/* A quantized node's parameters are canonical, not copied.
 *
 * The source node may have written a parameter, omitted it and taken a
 * default, or spelled a float at a precision the kernel cannot hold. The
 * template states the value the kernel will actually use — every parameter
 * present, defaults materialized, floats narrowed to the F32 they are stored
 * as — so that reading the template tells you what will run without also
 * knowing which defaults the authoring implementation had in mind. */
typedef struct VxPtqCanonicalParam {
    const char* name;
    enum {
        VX_PTQ_PARAM_INT,
        VX_PTQ_PARAM_F32,
        VX_PTQ_PARAM_STRING,
        VX_PTQ_PARAM_BOOL,
        VX_PTQ_PARAM_INT_ARRAY
    } kind;
    double number;
    const char* text;
    int64_t values[4];
    size_t value_count;
} VxPtqCanonicalParam;

static const VxPtqByteOp VX_PTQ_BYTE_OPS[] = {
    { VX_OP_LAYER_NORM, VX_OP_Q_LAYER_NORM, { "input", NULL, NULL, NULL } },
    { VX_OP_GROUP_NORM, VX_OP_Q_GROUP_NORM, { "input", NULL, NULL, NULL } },
    { VX_OP_GELU,      VX_OP_Q_GELU,      { "input", NULL, NULL, NULL } },
    { VX_OP_SILU,      VX_OP_Q_SILU,      { "input", NULL, NULL, NULL } },
    { VX_OP_ADD,       VX_OP_Q_ADD,       { "a", "b", NULL, NULL } },
    /* Both operands dynamic: a BatchMatMul against an immutable weight is a
     * dense node wearing the wrong name, and packing it as an activation
     * would quantize a constant against a calibrated range. */
    { VX_OP_BATCH_MATMUL, VX_OP_Q_BATCH_MATMUL, { "a", "b", NULL, NULL } },
    /* Q, K and V are each calibrated; the optional keep mask is copied
     * through as it stands, because it selects rather than scales. */
    { VX_OP_CROSS_SDPA, VX_OP_Q_SDPA, { "q", "k", "v", NULL } },
};

#define VX_PTQ_COUNT(array) (sizeof(array) / sizeof((array)[0]))

/* --- state ---------------------------------------------------------------- */

typedef struct VxPtqTensor {
    char name[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char dtype[24];
    int64_t shape[VX_PTQ_MAX_RANK];
    int rank;
    int is_initializer;
    int is_public_input;
    int is_graph_output;
    int producer; /* node index, or -1 */
} VxPtqTensor;

typedef enum VxPtqNodeRole {
    VX_PTQ_ROLE_RETAINED = 0,
    VX_PTQ_ROLE_DENSE = 1,
    VX_PTQ_ROLE_BYTE = 2,
    VX_PTQ_ROLE_CONV = 3
} VxPtqNodeRole;

/* Roles that pack a weight, and so need a layer entry and per-channel
 * scales. The rest only move activations between domains. */
#define VX_PTQ_ROLE_PACKS_WEIGHT(role) \
    ((role) == VX_PTQ_ROLE_DENSE || (role) == VX_PTQ_ROLE_CONV)

#define VX_PTQ_MAX_CANONICAL_PARAMS 8

typedef struct VxPtqNodePlan {
    VxPtqNodeRole role;
    const char* node_id;
    VxOperatorKind quantized_op;
    const VxPtqByteOp* byte_op;
    VxPtqCanonicalParam params[VX_PTQ_MAX_CANONICAL_PARAMS];
    size_t param_count;
    const char* input_tensor;
    const char* weight_tensor;
    const char* bias_tensor;
    const char* output_tensor;
    int weight_axis;
    /* How many output channels the packed weight will have, and therefore
     * how long its per-channel scale vector is. A bias must match it: a
     * shorter one would be read past its end when folded into the
     * accumulator. */
    int64_t output_channels;
    char packed_weight[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char packed_bias[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char weight_scale[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char weight_zero_point[VX_PTQ_AUTHORING_NAME_CAPACITY];
} VxPtqNodePlan;

/* An activation's byte form, and the two tensors its affine lives in. */
typedef struct VxPtqActivation {
    const char* source;
    char quantized[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char scale[VX_PTQ_AUTHORING_NAME_CAPACITY];
    char zero_point[VX_PTQ_AUTHORING_NAME_CAPACITY];
} VxPtqActivation;

typedef struct VxPtqAuthoringState {
    cJSON* source;
    cJSON* nodes;
    VxPtqTensor* tensors;
    size_t tensor_count;
    size_t tensor_capacity;
    VxPtqNameSet occupied;
    /* The set borrows its strings; allocated names need somewhere to live
     * that outlasts the buffer they were written into. */
    char (*reserved)[VX_PTQ_AUTHORING_NAME_CAPACITY];
    size_t reserved_count;
    size_t reserved_capacity;
    VxPtqNodePlan* plans;
    size_t node_count;
    VxPtqActivation* activations;
    size_t activation_count;
    const VxPtqAuthoringConfig* config;
    VxPtqAuthored* out;
} VxPtqAuthoringState;

static int vx_ptq_fail_v(VxPtqAuthored* out, VxStatus status,
                         const char* format, va_list arguments) {
    if (!out) return -1;
    out->status = status;
    vsnprintf(out->message, sizeof(out->message), format, arguments);
    return -1;
}

#if !defined(VOLVOXAI_PTQ_AUTHORING_NO_FILES)
static int vx_ptq_fail_as(VxPtqAuthored* out, VxStatus status,
                          const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vx_ptq_fail_v(out, status, format, arguments);
    va_end(arguments);
    return -1;
}
#endif

static int vx_ptq_fail(VxPtqAuthored* out, VxStatus status, const char* format, ...) {
    va_list arguments;
    va_start(arguments, format);
    vx_ptq_fail_v(out, status, format, arguments);
    va_end(arguments);
    return -1;
}

/* --- tensor table ---------------------------------------------------------- */

static VxPtqTensor* vx_ptq_find(VxPtqAuthoringState* state, const char* name) {
    size_t index;
    if (!name) return NULL;
    for (index = 0; index < state->tensor_count; index++) {
        if (strcmp(state->tensors[index].name, name) == 0) {
            return &state->tensors[index];
        }
    }
    return NULL;
}

static VxPtqTensor* vx_ptq_intern(VxPtqAuthoringState* state,
                                  const char* name) {
    VxPtqTensor* found = vx_ptq_find(state, name);
    if (found) return found;
    if (!name || !name[0]) {
        vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "tensor name is empty");
        return NULL;
    }
    if (strlen(name) >= VX_PTQ_AUTHORING_NAME_CAPACITY) {
        vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "tensor name exceeds the %u-byte authoring limit",
                    (unsigned)(VX_PTQ_AUTHORING_NAME_CAPACITY - 1u));
        return NULL;
    }
    if (state->tensor_count == state->tensor_capacity) {
        size_t capacity = state->tensor_capacity ? state->tensor_capacity * 2u
                                                 : 64u;
        VxPtqTensor* grown = (VxPtqTensor*)realloc(
            state->tensors, capacity * sizeof(*grown));
        if (!grown) {
            vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            return NULL;
        }
        state->tensors = grown;
        state->tensor_capacity = capacity;
    }
    found = &state->tensors[state->tensor_count++];
    memset(found, 0, sizeof(*found));
    snprintf(found->name, sizeof(found->name), "%s", name);
    snprintf(found->dtype, sizeof(found->dtype), "%s", "float32");
    found->producer = -1;
    return found;
}

/* Allocates a name and keeps a copy alive for the occupied set to point at. */
static int vx_ptq_reserve(VxPtqAuthoringState* state, const char* key,
                          const char* role,
                          char output[VX_PTQ_AUTHORING_NAME_CAPACITY]) {
    char derived[VX_PTQ_ALLOCATED_NAME_CAPACITY];
    if (vx_ptq_allocate_name(&state->occupied, key, role, derived) != 0) {
        return -1;
    }
    if (state->reserved_count == state->reserved_capacity) {
        size_t capacity = state->reserved_capacity ? state->reserved_capacity * 2u
                                                   : 64u;
        char (*grown)[VX_PTQ_AUTHORING_NAME_CAPACITY] = realloc(
            state->reserved, capacity * VX_PTQ_AUTHORING_NAME_CAPACITY);
        if (!grown) return -1;
        state->reserved = grown;
        state->reserved_capacity = capacity;
    }
    snprintf(state->reserved[state->reserved_count],
             VX_PTQ_AUTHORING_NAME_CAPACITY, "%s", derived);
    if (vx_ptq_name_set_add(&state->occupied,
                            state->reserved[state->reserved_count]) != 0) {
        return -1;
    }
    snprintf(output, VX_PTQ_AUTHORING_NAME_CAPACITY, "%s", derived);
    state->reserved_count++;
    return 0;
}

static const char* vx_ptq_string(cJSON* object, const char* key) {
    cJSON* item = object ? cJSON_GetObjectItemCaseSensitive(object, key) : NULL;
    return cJSON_IsString(item) ? item->valuestring : NULL;
}

static void vx_ptq_read_shape(cJSON* shape, int64_t out[VX_PTQ_MAX_RANK],
                              int* rank) {
    cJSON* extent;
    int index = 0;
    *rank = 0;
    if (!cJSON_IsArray(shape)) return;
    cJSON_ArrayForEach(extent, shape) {
        if (index >= VX_PTQ_MAX_RANK) break;
        /* A symbolic extent is a name, not a number. Authoring records it as
         * unresolved: the only extent it needs concrete is the weight's
         * channel count, which never is. */
        out[index] = cJSON_IsNumber(extent) ? (int64_t)extent->valuedouble : -1;
        index++;
    }
    *rank = index;
}

#if !defined(VOLVOXAI_PTQ_AUTHORING_NO_FILES)
/* The only two functions in this file that touch a filesystem. Everything
 * else works on bytes, which is what lets the same authoring serve a
 * WebAssembly build with no `fopen` to call. */
static int vx_ptq_read_binary_file(const char* path, unsigned char** bytes_out,
                                   size_t* size_out) {
    FILE* handle;
    long size;
    unsigned char* bytes;
    size_t read;

    *bytes_out = NULL;
    *size_out = 0u;
    if (!path) return -1;
    handle = fopen(path, "rb");
    if (!handle) return -1;
    if (fseek(handle, 0, SEEK_END) != 0) { fclose(handle); return -1; }
    size = ftell(handle);
    if (size < 0 || fseek(handle, 0, SEEK_SET) != 0) {
        fclose(handle);
        return -1;
    }
    /* One extra NUL so a caller may treat the bytes as a string. */
    bytes = (unsigned char*)malloc((size_t)size + 1u);
    if (!bytes) { fclose(handle); return -1; }
    read = fread(bytes, 1u, (size_t)size, handle);
    fclose(handle);
    if (read != (size_t)size) { free(bytes); return -1; }
    bytes[size] = '\0';
    *bytes_out = bytes;
    *size_out = (size_t)size;
    return 0;
}

static char* vx_ptq_read_file(const char* path) {
    unsigned char* bytes = NULL;
    size_t size = 0u;
    if (vx_ptq_read_binary_file(path, &bytes, &size) != 0) return NULL;
    return (char*)bytes;
}
#endif /* !VOLVOXAI_PTQ_AUTHORING_NO_FILES */

static int vx_ptq_collect_tensors(VxPtqAuthoringState* state,
                                  const SafetensorsBorrowedBytes* shards,
                                  size_t shard_count) {
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(state->source, "inputs");
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(state->source, "outputs");
    cJSON* entry;
    cJSON* node;
    int node_index = 0;
    size_t shard;
    size_t index;

    if (cJSON_IsObject(inputs)) {
        cJSON_ArrayForEach(entry, inputs) {
            VxPtqTensor* tensor = vx_ptq_intern(state, entry->string);
            const char* dtype;
            if (!tensor) return -1;
            tensor->is_public_input = 1;
            dtype = vx_ptq_string(entry, "dtype");
            if (dtype) snprintf(tensor->dtype, sizeof(tensor->dtype), "%s", dtype);
            vx_ptq_read_shape(cJSON_GetObjectItemCaseSensitive(entry, "shape"),
                              tensor->shape, &tensor->rank);
        }
    }

    cJSON_ArrayForEach(node, state->nodes) {
        cJSON* node_outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
        cJSON* node_inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        cJSON* port;
        cJSON_ArrayForEach(port, node_outputs) {
            const char* name = vx_ptq_string(port, "tensor");
            const char* dtype = vx_ptq_string(port, "dtype");
            VxPtqTensor* tensor;
            if (!name) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "node %d output '%s' declares no tensor name", node_index,
                    port->string ? port->string : "?");
            }
            tensor = vx_ptq_intern(state, name);
            if (!tensor) return -1;
            tensor->producer = node_index;
            if (dtype) snprintf(tensor->dtype, sizeof(tensor->dtype), "%s", dtype);
            vx_ptq_read_shape(cJSON_GetObjectItemCaseSensitive(port, "shape"),
                              tensor->shape, &tensor->rank);
        }
        cJSON_ArrayForEach(port, node_inputs) {
            if (cJSON_IsString(port) &&
                !vx_ptq_intern(state, port->valuestring)) {
                return -1;
            }
        }
        node_index++;
    }

    /* Anything a shard provides is immutable. That is what separates a weight
     * from an activation, and it is a property of the package rather than of
     * the graph — which is why the shards are read even though no payload is
     * packed here. */
    for (shard = 0u; shard < shard_count; shard++) {
        int stored_index;
        for (stored_index = 0; stored_index < shards[shard].tensor_count;
             stored_index++) {
            const SafetensorsTensor* stored = &shards[shard].tensors[stored_index];
            VxPtqTensor* tensor = vx_ptq_intern(state, stored->name);
            int axis;
            if (!tensor) return -1;
            tensor->is_initializer = 1;
            tensor->rank = stored->ndim < VX_PTQ_MAX_RANK ? stored->ndim
                                                          : VX_PTQ_MAX_RANK;
            for (axis = 0; axis < tensor->rank; axis++) {
                tensor->shape[axis] = (int64_t)stored->shape[axis];
            }
        }
    }

    if (cJSON_IsArray(outputs)) {
        cJSON_ArrayForEach(entry, outputs) {
            VxPtqTensor* tensor;
            if (!cJSON_IsString(entry)) continue;
            tensor = vx_ptq_find(state, entry->valuestring);
            if (!tensor) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "graph output '%s' is produced by nothing",
                    entry->valuestring);
            }
            tensor->is_graph_output = 1;
        }
    }

    /* Reserve every existing name so an allocated one can never shadow it. */
    for (index = 0; index < state->tensor_count; index++) {
        if (vx_ptq_name_set_add(&state->occupied,
                                state->tensors[index].name) != 0) {
            return vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
        }
    }
    return 0;
}

/* --- qualification --------------------------------------------------------- */

static int vx_ptq_named(const char* const* list, size_t count,
                        const char* value) {
    size_t index;
    if (!list || !value) return 0;
    for (index = 0; index < count; index++) {
        if (list[index] && strcmp(list[index], value) == 0) return 1;
    }
    return 0;
}

static const VxPtqDenseOp* vx_ptq_dense_op(VxOperatorKind kind) {
    size_t index;
    for (index = 0; index < VX_PTQ_COUNT(VX_PTQ_DENSE_OPS); index++) {
        if (VX_PTQ_DENSE_OPS[index].source == kind) {
            return &VX_PTQ_DENSE_OPS[index];
        }
    }
    return NULL;
}

static const VxPtqByteOp* vx_ptq_byte_op(VxOperatorKind kind) {
    size_t index;
    for (index = 0; index < VX_PTQ_COUNT(VX_PTQ_BYTE_OPS); index++) {
        if (VX_PTQ_BYTE_OPS[index].source == kind) {
            return &VX_PTQ_BYTE_OPS[index];
        }
    }
    return NULL;
}

/* A bias is folded into the accumulator one output channel at a time, so its
 * length is not a detail: a short one is read past its end. */
static int vx_ptq_check_bias(VxPtqAuthoringState* state,
                             const VxPtqNodePlan* plan, const char* bias_name) {
    const VxPtqTensor* bias = vx_ptq_find(state, bias_name);
    if (!bias || !bias->is_initializer) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "'%s' bias '%s' must be an immutable initializer", plan->node_id,
            bias_name);
    }
    if (bias->rank != 1 || bias->shape[0] != plan->output_channels) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "'%s' bias '%s' has %lld entries for %lld output channels",
            plan->node_id, bias_name,
            (long long)(bias->rank == 1 ? bias->shape[0] : -1),
            (long long)plan->output_channels);
    }
    return 0;
}

/* A dense node qualifies when it reads exactly one immutable F32 weight
 * through the canonical port, declares which way that weight is laid out, and
 * produces a dynamic F32 result. Anything else is refused: PTQ cannot state
 * what the packed form would mean. */
static int vx_ptq_qualify_dense(VxPtqAuthoringState* state, cJSON* node,
                                const char* op_type, VxPtqNodePlan* plan) {
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    cJSON* out_port = outputs ? cJSON_GetObjectItemCaseSensitive(outputs, "out")
                              : NULL;
    const char* layout = vx_ptq_string(params, "weight_layout");
    const char* input_name = vx_ptq_string(inputs, "input");
    const char* weight_name = vx_ptq_string(inputs, "weight");
    const char* bias_name = vx_ptq_string(inputs, "bias");
    const char* output_name = vx_ptq_string(out_port, "tensor");
    const VxPtqTensor* input;
    const VxPtqTensor* weight;
    const VxPtqTensor* output;

    if (!input_name || !weight_name || !output_name) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "%s '%s' needs input and weight ports and one 'out' result",
            op_type, plan->node_id);
    }
    if (!layout || (vx_generated_node_param_symbol_from_json(layout) != VX_NODE_SYMBOL_DIN_DOUT &&
                    vx_generated_node_param_symbol_from_json(layout) != VX_NODE_SYMBOL_DOUT_DIN)) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "%s '%s' needs an explicit weight_layout of din_dout or dout_din",
            op_type, plan->node_id);
    }
    input = vx_ptq_find(state, input_name);
    weight = vx_ptq_find(state, weight_name);
    output = vx_ptq_find(state, output_name);
    if (!input || !weight || !output) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "%s '%s' names an unknown tensor",
                           op_type, plan->node_id);
    }
    if (!weight->is_initializer) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "%s '%s' reads a dynamic weight '%s'; PTQ packs immutable weights",
            op_type, plan->node_id, weight_name);
    }
    if (input->is_initializer || output->is_initializer) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "%s '%s' needs dynamic activation and result tensors",
            op_type, plan->node_id);
    }
    if (weight->rank != 2) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "%s '%s' weight '%s' has rank %d; dense PTQ packs a matrix",
            op_type, plan->node_id, weight_name, weight->rank);
    }
    /* The packed weight is always [dout, din], so output channels land on
     * axis 0 whichever way the source was stored. din_dout means the source
     * has them last; dout_din means it already has them first. */
    plan->output_channels = vx_generated_node_param_symbol_from_json(layout) == VX_NODE_SYMBOL_DIN_DOUT ? weight->shape[1]
                                                            : weight->shape[0];
    if (bias_name && vx_ptq_check_bias(state, plan, bias_name) != 0) return -1;

    plan->role = VX_PTQ_ROLE_DENSE;
    plan->quantized_op = VX_OP_Q_LINEAR;
    plan->input_tensor = input_name;
    plan->weight_tensor = weight_name;
    plan->bias_tensor = bias_name;
    plan->output_tensor = output_name;
    plan->weight_axis = 0;
    return 0;
}

static double vx_ptq_number(cJSON* params, const char* key, double fallback) {
    cJSON* item = params ? cJSON_GetObjectItemCaseSensitive(params, key) : NULL;
    return cJSON_IsNumber(item) ? item->valuedouble : fallback;
}

/* Reads an integer sequence parameter, or the default when absent.
 * `count` values are required exactly; a shorter or longer list is a graph
 * describing geometry the kernel cannot perform. */
static int vx_ptq_int_sequence(cJSON* params, const char* key, size_t count,
                               const int64_t* fallback, int64_t* out) {
    cJSON* item = params ? cJSON_GetObjectItemCaseSensitive(params, key) : NULL;
    cJSON* entry;
    size_t index = 0u;
    if (!item) {
        for (index = 0; index < count; index++) out[index] = fallback[index];
        return 0;
    }
    if (!cJSON_IsArray(item) || (size_t)cJSON_GetArraySize(item) != count) {
        return -1;
    }
    cJSON_ArrayForEach(entry, item) {
        if (!cJSON_IsNumber(entry) ||
            entry->valuedouble != (double)(int64_t)entry->valuedouble) {
            return -1;
        }
        out[index++] = (int64_t)entry->valuedouble;
    }
    return 0;
}

static void vx_ptq_push_int_array(VxPtqNodePlan* plan, const char* name,
                                  const int64_t* values, size_t count) {
    VxPtqCanonicalParam* param;
    size_t index;
    if (plan->param_count >= VX_PTQ_MAX_CANONICAL_PARAMS || count > 4u) return;
    param = &plan->params[plan->param_count++];
    memset(param, 0, sizeof(*param));
    param->name = name;
    param->kind = VX_PTQ_PARAM_INT_ARRAY;
    param->value_count = count;
    for (index = 0; index < count; index++) param->values[index] = values[index];
}

static void vx_ptq_push_param(VxPtqNodePlan* plan, const char* name, int kind,
                              double number, const char* text) {
    VxPtqCanonicalParam* param;
    if (plan->param_count >= VX_PTQ_MAX_CANONICAL_PARAMS) return;
    param = &plan->params[plan->param_count++];
    memset(param, 0, sizeof(*param));
    param->name = name;
    param->kind = kind;
    /* A float parameter is stored as F32 and read back as F32, so the value
     * the template states is the narrowed one. Writing 1e-5 when the kernel
     * will see 9.9999997e-6 makes the template describe a graph that does not
     * run. */
    param->number = kind == VX_PTQ_PARAM_F32 ? (double)(float)number : number;
    param->text = text;
}

static int vx_ptq_push_f32_param(VxPtqAuthoringState* state,
                                 VxPtqNodePlan* plan,
                                 const char* name,
                                 double number) {
    float narrowed = (float)number;
    if (!isfinite(number) || !isfinite(narrowed)) {
        return vx_ptq_fail(
            state->out, VX_STATUS_INVALID_GRAPH,
            "node '%s' parameter '%s' must be finite and representable as F32",
            plan->node_id, name);
    }
    vx_ptq_push_param(plan, name, VX_PTQ_PARAM_F32, (double)narrowed, NULL);
    return 0;
}

/* Conv2D packs a rank-4 weight, so it is a dense node with geometry.
 *
 * The geometry is stated, never inferred: an implicit stride or padding would
 * make the template describe one convolution and the kernel perform another,
 * and the difference is an output that is the wrong size rather than wrong by
 * a little. */
static int vx_ptq_qualify_conv(VxPtqAuthoringState* state, cJSON* node,
                               VxPtqNodePlan* plan) {
    static const int64_t UNIT[2] = { 1, 1 };
    static const int64_t ZERO[2] = { 0, 0 };
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    cJSON* out_port = outputs ? cJSON_GetObjectItemCaseSensitive(outputs, "out")
                              : NULL;
    cJSON* groups_json = cJSON_GetObjectItemCaseSensitive(params, "groups");
    cJSON* relu_json = cJSON_GetObjectItemCaseSensitive(params, "relu");
    const char* input_name = vx_ptq_string(inputs, "input");
    const char* weight_name = vx_ptq_string(inputs, "weight");
    const char* bias_name = vx_ptq_string(inputs, "bias");
    const char* output_name = vx_ptq_string(out_port, "tensor");
    const char* data_layout = vx_ptq_string(params, "data_layout");
    const char* weight_layout = vx_ptq_string(params, "weight_layout");
    const VxPtqTensor* weight;
    const VxPtqTensor* input;
    int64_t stride[2];
    int64_t dilation[2];
    int64_t padding[2];
    int64_t pads[4];
    int64_t pad_defaults[4];
    double groups = 1.0;
    double relu = 0.0;

    if (!input_name || !weight_name || !output_name) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' needs input and weight ports and one 'out' result",
            plan->node_id);
    }
    if (data_layout && vx_generated_node_param_symbol_from_json(data_layout) != VX_NODE_SYMBOL_NHWC) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' is %s; the byte-domain kernel is NHWC only, and "
            "relabelling another layout is not a transpose",
            plan->node_id, data_layout);
    }
    if (weight_layout && vx_generated_node_param_symbol_from_json(weight_layout) != VX_NODE_SYMBOL_OHWI &&
        vx_generated_node_param_symbol_from_json(weight_layout) != VX_NODE_SYMBOL_HWIO) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' weight layout %s is neither OHWI nor HWIO",
            plan->node_id, weight_layout);
    }
    input = vx_ptq_find(state, input_name);
    weight = vx_ptq_find(state, weight_name);
    if (!input || !weight) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "Conv2D '%s' names an unknown tensor",
                           plan->node_id);
    }
    if (!weight->is_initializer || weight->rank != 4) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' needs an immutable rank-4 weight", plan->node_id);
    }
    if (input->is_initializer) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' needs a dynamic activation", plan->node_id);
    }
    /* Packed OHWI: output channels lead, whichever way the source stored
     * them. */
    plan->output_channels =
        weight_layout && vx_generated_node_param_symbol_from_json(weight_layout) == VX_NODE_SYMBOL_HWIO ? weight->shape[3]
                                                            : weight->shape[0];
    if (bias_name && vx_ptq_check_bias(state, plan, bias_name) != 0) return -1;
    if (vx_ptq_int_sequence(params, "stride", 2u, UNIT, stride) != 0 ||
        vx_ptq_int_sequence(params, "dilation", 2u, UNIT, dilation) != 0 ||
        vx_ptq_int_sequence(params, "padding", 2u, ZERO, padding) != 0) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' has a malformed stride, dilation or padding",
            plan->node_id);
    }
    if (stride[0] <= 0 || stride[1] <= 0 || dilation[0] <= 0 ||
        dilation[1] <= 0) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "Conv2D '%s' needs positive stride and dilation", plan->node_id);
    }
    pad_defaults[0] = padding[0];
    pad_defaults[1] = padding[1];
    pad_defaults[2] = padding[0];
    pad_defaults[3] = padding[1];
    if (vx_ptq_int_sequence(params, "pads", 4u, pad_defaults, pads) != 0) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "Conv2D '%s' has malformed pads",
                           plan->node_id);
    }
    if (groups_json) {
        if (!cJSON_IsNumber(groups_json) || groups_json->valuedouble <= 0.0 ||
            groups_json->valuedouble !=
                (double)(int)groups_json->valuedouble) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "Conv2D '%s' needs a positive integer group count",
                plan->node_id);
        }
        groups = groups_json->valuedouble;
    }
    if (relu_json) {
        if (!cJSON_IsNumber(relu_json) ||
            (relu_json->valuedouble != 0.0 && relu_json->valuedouble != 1.0 &&
             relu_json->valuedouble != 2.0)) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "Conv2D '%s' relu must be 0, 1 or 2", plan->node_id);
        }
        relu = relu_json->valuedouble;
    }

    plan->role = VX_PTQ_ROLE_CONV;
    plan->quantized_op = VX_OP_Q_CONV_2D;
    plan->input_tensor = input_name;
    plan->weight_tensor = weight_name;
    plan->bias_tensor = bias_name;
    plan->output_tensor = output_name;
    /* Packed OHWI: output channels lead, whichever way the source stored
     * them. */
    plan->weight_axis = 0;
    vx_ptq_push_int_array(plan, "stride", stride, 2u);
    vx_ptq_push_int_array(plan, "dilation", dilation, 2u);
    vx_ptq_push_param(plan, "groups", VX_PTQ_PARAM_INT, groups, NULL);
    vx_ptq_push_int_array(plan, "pads", pads, 4u);
    vx_ptq_push_param(plan, "data_layout", VX_PTQ_PARAM_STRING, 0.0, "NHWC");
    vx_ptq_push_param(plan, "weight_layout", VX_PTQ_PARAM_STRING, 0.0, "OHWI");
    vx_ptq_push_param(plan, "relu", VX_PTQ_PARAM_INT, relu, NULL);
    return 0;
}

static int vx_ptq_qualify_byte(VxPtqAuthoringState* state, cJSON* node,
                               const VxPtqByteOp* op, VxPtqNodePlan* plan) {
    cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    cJSON* out_port = outputs ? cJSON_GetObjectItemCaseSensitive(outputs, "out")
                              : NULL;
    cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    const char* output_name = vx_ptq_string(out_port, "tensor");

    if (!output_name) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "%s '%s' needs one 'out' result",
                           vx_operator_kind_name(op->source), plan->node_id);
    }
    plan->role = VX_PTQ_ROLE_BYTE;
    plan->quantized_op = op->quantized;
    plan->byte_op = op;
    plan->output_tensor = output_name;

    if (op->quantized == VX_OP_Q_ADD) {
        vx_ptq_push_param(plan, "relu", VX_PTQ_PARAM_INT,
                          vx_ptq_number(params, "relu", 0.0), NULL);
    } else if (op->quantized == VX_OP_Q_GELU) {
        const char* approximate = vx_ptq_string(params, "approximate");
        if (approximate && vx_generated_node_param_symbol_from_json(approximate) != VX_NODE_SYMBOL_NONE) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "GELU '%s' has approximate='%s'; the byte-domain kernel is "
                "exact and implements only 'none'",
                plan->node_id, approximate);
        }
        vx_ptq_push_param(plan, "approximate", VX_PTQ_PARAM_STRING, 0.0, "none");
    } else if (op->quantized == VX_OP_Q_LAYER_NORM) {
        const char* input_name = vx_ptq_string(inputs, "input");
        const VxPtqTensor* input = vx_ptq_find(state, input_name);
        if (!input || input->rank < 1 || input->shape[input->rank - 1] <= 0) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "LayerNorm '%s' needs an activation with a fixed final "
                "dimension", plan->node_id);
        }
        vx_ptq_push_param(plan, "d_model", VX_PTQ_PARAM_INT,
                          (double)input->shape[input->rank - 1], NULL);
        if (vx_ptq_push_f32_param(
                state, plan, "eps", vx_ptq_number(params, "eps", 1e-5)) != 0) {
            return -1;
        }
    } else if (op->quantized == VX_OP_Q_BATCH_MATMUL) {
        const char* operands[2] = { vx_ptq_string(inputs, "a"),
                                    vx_ptq_string(inputs, "b") };
        size_t which;
        if (cJSON_IsObject(params) && cJSON_GetArraySize(params) != 0) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "BatchMatMul '%s' has parameters; the byte-domain kernel takes "
                "none", plan->node_id);
        }
        for (which = 0u; which < 2u; which++) {
            const VxPtqTensor* operand = vx_ptq_find(state, operands[which]);
            if (!operand) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "BatchMatMul '%s' names an unknown operand", plan->node_id);
            }
            if (operand->is_initializer) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "BatchMatMul '%s' operand '%s' is immutable; a dense node "
                    "against a weight belongs in Linear, where the weight is "
                    "packed rather than calibrated",
                    plan->node_id, operand->name);
            }
        }
    } else if (op->quantized == VX_OP_Q_GROUP_NORM) {
        double groups = vx_ptq_number(params, "num_groups", 0.0);
        const char* layout = vx_ptq_string(params, "data_layout");
        if (groups <= 0.0) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "GroupNorm '%s' needs a positive num_groups", plan->node_id);
        }
        if (layout && vx_generated_node_param_symbol_from_json(layout) != VX_NODE_SYMBOL_NHWC) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "GroupNorm '%s' is %s; the byte-domain kernel is NHWC only",
                plan->node_id, layout);
        }
        vx_ptq_push_param(plan, "num_groups", VX_PTQ_PARAM_INT, groups, NULL);
        if (vx_ptq_push_f32_param(
                state, plan, "eps", vx_ptq_number(params, "eps", 1e-5)) != 0) {
            return -1;
        }
        vx_ptq_push_param(plan, "data_layout", VX_PTQ_PARAM_STRING, 0.0, "NHWC");
    } else if (op->quantized == VX_OP_Q_SDPA) {
        cJSON* heads = cJSON_GetObjectItemCaseSensitive(params, "heads");
        cJSON* causal = cJSON_GetObjectItemCaseSensitive(params, "causal");
        cJSON* scale = cJSON_GetObjectItemCaseSensitive(params, "scale");
        const char* encoding = vx_ptq_string(params, "mask_encoding");
        if (!cJSON_IsNumber(heads) || heads->valuedouble <= 0.0 ||
            heads->valuedouble != (double)(int)heads->valuedouble) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "CrossSDPA '%s' needs an explicit positive head count",
                plan->node_id);
        }
        if (!cJSON_IsBool(causal)) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "CrossSDPA '%s' needs an explicit boolean causal; leaving it "
                "implied would let the mask decide, and the mask is optional",
                plan->node_id);
        }
        if (encoding && vx_generated_node_param_symbol_from_json(encoding) != VX_NODE_SYMBOL_KEEP) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "CrossSDPA '%s' carries a %s mask; the byte-domain kernel "
                "consumes I32 keep masks, not F32 additive ones",
                plan->node_id, encoding);
        }
        vx_ptq_push_param(plan, "heads", VX_PTQ_PARAM_INT,
                          heads->valuedouble, NULL);
        vx_ptq_push_param(plan, "causal", VX_PTQ_PARAM_BOOL,
                          cJSON_IsTrue(causal) ? 1.0 : 0.0, NULL);
        /* Only when the source stated one: an absent scale means the default
         * 1/sqrt(head_dim), and materializing that here would turn a
         * derived value into a declared one. */
        if (scale) {
            if (!cJSON_IsNumber(scale) || !(scale->valuedouble > 0.0)) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "CrossSDPA '%s' has a non-positive scale", plan->node_id);
            }
            if (vx_ptq_push_f32_param(
                    state, plan, "scale", scale->valuedouble) != 0) {
                return -1;
            }
        }
    } else if (op->quantized == VX_OP_Q_SILU) {
        if (cJSON_IsObject(params) && cJSON_GetArraySize(params) != 0) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "SiLU '%s' has parameters; the byte-domain kernel takes none",
                plan->node_id);
        }
    }
    return 0;
}

static int vx_ptq_qualify(VxPtqAuthoringState* state) {
    const VxPtqAuthoringConfig* config = state->config;
    cJSON* node;
    size_t index = 0u;

    cJSON_ArrayForEach(node, state->nodes) {
        VxPtqNodePlan* plan = &state->plans[index++];
        const char* op_type = vx_ptq_string(node, "opType");
        VxOperatorKind kind = vx_operator_kind_from_name(op_type);

        plan->node_id = vx_ptq_string(node, "id");
        if (!op_type || !plan->node_id) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "node %zu has no id or opType",
                               index - 1u);
        }
        if (vx_ptq_named(config->float_operators, config->float_operator_count,
                         op_type) ||
            vx_ptq_named(config->float_nodes, config->float_node_count,
                         plan->node_id) ||
            (config->selected_node_count &&
             !vx_ptq_named(config->selected_nodes, config->selected_node_count,
                           plan->node_id))) {
            plan->role = VX_PTQ_ROLE_RETAINED;
            state->out->retained_float_nodes++;
            continue;
        }
        if (kind == VX_OP_EMBEDDING) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "node '%s' is %s, which this implementation does not author "
                "yet; name it in float_operators to keep it in F32",
                plan->node_id, op_type);
        }
        if (kind == VX_OP_SDPA) {
            /* Same refusal the Python exporter gives, and for the same
             * reason: a packed qkv operand does not expose independently
             * calibrated Q, K and V, so there is no sound lowering. */
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                "node '%s' is a packed SDPA; canonicalize it to an explicit "
                "CrossSDPA before calibration, because packed qkv exposes no "
                "independently calibrated Q/K/V to quantize",
                plan->node_id);
        }
        if (vx_ptq_dense_op(kind)) {
            if (vx_ptq_qualify_dense(state, node, op_type, plan) != 0) return -1;
        } else if (kind == VX_OP_CONV_2D) {
            if (vx_ptq_qualify_conv(state, node, plan) != 0) return -1;
        } else if (vx_ptq_byte_op(kind)) {
            if (vx_ptq_qualify_byte(state, node, vx_ptq_byte_op(kind),
                                    plan) != 0) {
                return -1;
            }
        } else {
            plan->role = VX_PTQ_ROLE_RETAINED;
            state->out->retained_float_nodes++;
            continue;
        }
        state->out->quantized_nodes++;
    }
    if (!state->out->quantized_nodes) {
        return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
            "no node in this graph qualifies for quantization");
    }
    return 0;
}

/* Names the tensors each quantized node introduces. Separate from
 * qualification so that allocation order is the node order, which is what
 * makes the derived names reproducible. */
static int vx_ptq_name_payloads(VxPtqAuthoringState* state) {
    size_t index;
    for (index = 0; index < state->node_count; index++) {
        VxPtqNodePlan* plan = &state->plans[index];
        if (!VX_PTQ_ROLE_PACKS_WEIGHT(plan->role)) continue;
        if (vx_ptq_reserve(state, plan->node_id, "weight",
                           plan->packed_weight) != 0 ||
            vx_ptq_reserve(state, plan->node_id, "weight_scale",
                           plan->weight_scale) != 0 ||
            vx_ptq_reserve(state, plan->node_id, "weight_zero_point",
                           plan->weight_zero_point) != 0) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "cannot name '%s' packed weight",
                               plan->node_id);
        }
        /* Always, even when the float node had no bias. QLinear's bias is an
         * int32 accumulator term that carries the activation zero point's
         * contribution, so it exists whenever the input is asymmetric — and
         * making its presence depend on the source graph would mean the same
         * kernel took two shapes. */
        if (vx_ptq_reserve(state, plan->node_id, "bias",
                           plan->packed_bias) != 0) {
            return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "cannot name '%s' packed bias",
                               plan->node_id);
        }
    }
    return 0;
}

static const VxPtqActivation* vx_ptq_activation(
        const VxPtqAuthoringState* state, const char* source) {
    size_t index;
    for (index = 0; index < state->activation_count; index++) {
        if (strcmp(state->activations[index].source, source) == 0) {
            return &state->activations[index];
        }
    }
    return NULL;
}

/* Every dynamic tensor a quantized node reads or writes crosses into the byte
 * domain and needs an affine. Walking the qualified nodes rather than every
 * tensor is what keeps a retained float region out of the island. */
static int vx_ptq_collect_activations(VxPtqAuthoringState* state) {
    cJSON* node;
    size_t index = 0u;

    cJSON_ArrayForEach(node, state->nodes) {
        VxPtqNodePlan* plan = &state->plans[index++];
        cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const char* names[VX_PTQ_MAX_ACTIVATION_PORTS + 1u];
        size_t named = 0u;
        size_t which;

        if (plan->role == VX_PTQ_ROLE_RETAINED) continue;
        if (VX_PTQ_ROLE_PACKS_WEIGHT(plan->role)) {
            names[named++] = plan->input_tensor;
        } else {
            for (which = 0u; which < VX_PTQ_MAX_ACTIVATION_PORTS; which++) {
                const char* port = plan->byte_op->activation_ports[which];
                const char* value;
                if (!port) break;
                value = vx_ptq_string(inputs, port);
                if (!value) {
                    return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "node '%s' has no '%s' input",
                                       plan->node_id, port);
                }
                names[named++] = value;
            }
        }
        names[named++] = plan->output_tensor;

        for (which = 0u; which < named; which++) {
            VxPtqTensor* tensor = vx_ptq_find(state, names[which]);
            VxPtqActivation* activation;
            if (!tensor) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "node '%s' names an unknown tensor '%s'", plan->node_id,
                    names[which]);
            }
            /* An immutable operand of a byte node — a LayerNorm scale, say —
             * stays in float and gets no affine. */
            if (tensor->is_initializer) continue;
            if (vx_ptq_activation(state, tensor->name)) continue;
            activation = &state->activations[state->activation_count];
            activation->source = tensor->name;
            if (vx_ptq_reserve(state, tensor->name, "activation",
                               activation->quantized) != 0 ||
                vx_ptq_reserve(state, tensor->name, "scale",
                               activation->scale) != 0 ||
                vx_ptq_reserve(state, tensor->name, "zero_point",
                               activation->zero_point) != 0) {
                return vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                    "cannot name the affine for '%s'", tensor->name);
            }
            state->activation_count++;
        }
    }
    return 0;
}

/* --- emission --------------------------------------------------------------- */

static const char* vx_ptq_storage_name(VxPtqStorage storage) {
    return storage == VX_PTQ_STORAGE_U8 ? "uint8" : "int8";
}

/* A float parameter, written so that reading it back gives the same double.
 *
 * cJSON prints numbers at 15 significant digits and keeps that form when it
 * compares equal within an epsilon, which is close enough for most uses and
 * not close enough here: a template is compared against the one another
 * implementation wrote, and a value that differs in the sixteenth digit reads
 * as a disagreement about arithmetic rather than about formatting.
 *
 * Shortest-round-trip is what every other language in this project already
 * emits, so matching it costs one loop and removes the whole question. */
#if defined(VOLVOXAI_PTQ_HOST_FORMAT) && VOLVOXAI_PTQ_HOST_FORMAT
/* Written by the embedding, which has a correct one.
 *
 * Rendering a double as the shortest decimal that reads back as itself is a
 * solved problem with a subtle solution — correct rounding past fifteen
 * significant digits needs more precision than the double itself carries. A
 * freestanding build has no C library to ask, and a hand-written formatter
 * that is right for most values is worse than none: the ones it gets wrong
 * become templates whose parameters mean something slightly different from
 * the same template authored natively.
 *
 * So the WebAssembly build asks its host instead. JavaScript's number-to-
 * string is specified to produce exactly this shortest form, and
 * ts/host/PtqAuthoringWasm.ts lays those digits out the way a C library
 * would. One import, and the question is closed. Returns the length written,
 * or a negative value if the host could not. */
extern int vx_ptq_host_format_double(double value, char* buffer, int capacity);
#endif

static cJSON* vx_ptq_exact_number(double value) {
    char buffer[40];
    if (!isfinite(value)) return NULL;
#if defined(VOLVOXAI_PTQ_HOST_FORMAT) && VOLVOXAI_PTQ_HOST_FORMAT
    int written = vx_ptq_host_format_double(value, buffer,
                                            (int)sizeof(buffer));
    if (written <= 0 || (size_t)written >= sizeof(buffer)) return NULL;
    buffer[written] = '\0';
    return cJSON_CreateRaw(buffer);
#else
    int precision;
    for (precision = 1; precision < 17; precision++) {
        snprintf(buffer, sizeof(buffer), "%.*g", precision, value);
        if (strtod(buffer, NULL) == value) break;
    }
    if (precision >= 17) {
        snprintf(buffer, sizeof(buffer), "%.17g", value);
        /* Seventeen significant digits round-trip for every double, so a
         * value that still does not read back is not an unlucky number — it
         * is a formatter that cannot render this magnitude. The freestanding
         * build has one of those outside a bounded range, and a parameter it
         * cannot write is a parameter that would mean something different
         * wherever the template is read. Refusing beats writing it. */
        if (strtod(buffer, NULL) != value) return NULL;
    }
    return cJSON_CreateRaw(buffer);
#endif
}

static cJSON* vx_ptq_shape_json(const VxPtqTensor* tensor) {
    cJSON* shape = cJSON_CreateArray();
    int axis;
    if (!shape) return NULL;
    for (axis = 0; axis < tensor->rank; axis++) {
        cJSON* extent = cJSON_CreateNumber((double)tensor->shape[axis]);
        if (!extent) { cJSON_Delete(shape); return NULL; }
        cJSON_AddItemToArray(shape, extent);
    }
    return shape;
}

/* `out` descriptor of an emitted node: which tensor, what storage, and the
 * shape it inherits from the value it replaces. */
static cJSON* vx_ptq_out_descriptor(const VxPtqTensor* shape_of,
                                    const char* tensor_name,
                                    const char* dtype) {
    cJSON* outputs = cJSON_CreateObject();
    cJSON* port = cJSON_CreateObject();
    cJSON* shape;
    if (!outputs || !port) {
        cJSON_Delete(outputs);
        cJSON_Delete(port);
        return NULL;
    }
    shape = vx_ptq_shape_json(shape_of);
    if (!shape ||
        !cJSON_AddStringToObject(port, "tensor", tensor_name) ||
        !cJSON_AddStringToObject(port, "dtype", dtype)) {
        cJSON_Delete(shape);
        cJSON_Delete(port);
        cJSON_Delete(outputs);
        return NULL;
    }
    cJSON_AddItemToObject(port, "shape", shape);
    cJSON_AddItemToObject(outputs, "out", port);
    return outputs;
}

static int vx_ptq_add_affine(cJSON* table, const char* tensor_name,
                             int per_axis, int axis, const char* scale_name,
                             const char* zero_point_name) {
    cJSON* entry = cJSON_CreateObject();
    if (!entry) return -1;
    if (!cJSON_AddStringToObject(entry, "scheme",
                                 per_axis ? "per_axis" : "per_tensor") ||
        (per_axis && !cJSON_AddNumberToObject(entry, "axis", (double)axis)) ||
        !cJSON_AddStringToObject(entry, "scale_tensor", scale_name) ||
        !cJSON_AddStringToObject(entry, "zero_point_tensor", zero_point_name)) {
        cJSON_Delete(entry);
        return -1;
    }
    cJSON_AddItemToObject(table, tensor_name, entry);
    return 0;
}

/* Where float enters the island. A public input feeding a quantized node, or
 * any tensor a retained float node produced, needs converting once. */
static int vx_ptq_needs_quantize(const VxPtqAuthoringState* state,
                                 const VxPtqTensor* tensor) {
    if (tensor->is_public_input) return 1;
    if (tensor->producer < 0) return 0;
    return state->plans[tensor->producer].role == VX_PTQ_ROLE_RETAINED;
}

static int vx_ptq_emit(VxPtqAuthoringState* state, char** rendered_out) {
    const VxPtqAuthoringConfig* config = state->config;
    const char* activation_dtype = vx_ptq_storage_name(config->activation_storage);
    const char* weight_dtype = vx_ptq_storage_name(config->weight_storage);
    cJSON* root = cJSON_Duplicate(state->source, 1);
    cJSON* nodes = cJSON_CreateArray();
    cJSON* quantization = cJSON_CreateObject();
    cJSON* affines = cJSON_CreateObject();
    cJSON* source_node;
    size_t index;
    size_t param_index;
    int status = -1;
    char* rendered = NULL;

    *rendered_out = NULL;
    if (!root || !nodes || !quantization || !affines) {
        vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
        goto done;
    }

    /* Quantize boundaries first, then the graph in its original order, then
     * dequantize boundaries. Keeping the original node order is what lets a
     * reader match the template against the graph it came from. */
    for (index = 0; index < state->activation_count; index++) {
        const VxPtqActivation* activation = &state->activations[index];
        const VxPtqTensor* tensor = vx_ptq_find(state, activation->source);
        cJSON* node;
        cJSON* inputs;
        cJSON* outputs;
        char node_id[VX_PTQ_AUTHORING_NAME_CAPACITY];

        if (!vx_ptq_needs_quantize(state, tensor)) continue;
        if (vx_ptq_reserve(state, tensor->name, "quantize", node_id) != 0) {
            vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH, "cannot name the quantize boundary for '%s'",
                        tensor->name);
            goto done;
        }
        node = cJSON_CreateObject();
        inputs = cJSON_CreateObject();
        outputs = vx_ptq_out_descriptor(tensor, activation->quantized,
                                        activation_dtype);
        if (!node || !inputs || !outputs) {
            cJSON_Delete(node); cJSON_Delete(inputs); cJSON_Delete(outputs);
            vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            goto done;
        }
        cJSON_AddStringToObject(inputs, "input", tensor->name);
        cJSON_AddStringToObject(inputs, "scale", activation->scale);
        cJSON_AddStringToObject(inputs, "zero_point", activation->zero_point);
        cJSON_AddStringToObject(node, "id", node_id);
        cJSON_AddStringToObject(node, "opType", "QuantizeLinear");
        cJSON_AddItemToObject(node, "inputs", inputs);
        cJSON_AddItemToObject(node, "outputs", outputs);
        cJSON_AddItemToObject(node, "params", cJSON_CreateObject());
        cJSON_AddItemToArray(nodes, node);
    }

    index = 0u;
    cJSON_ArrayForEach(source_node, state->nodes) {
        VxPtqNodePlan* plan = &state->plans[index++];
        cJSON* node;
        cJSON* inputs;
        cJSON* outputs;
        cJSON* params;
        const VxPtqActivation* produced;
        const VxPtqTensor* output_tensor;

        if (plan->role == VX_PTQ_ROLE_RETAINED) {
            cJSON* copy = cJSON_Duplicate(source_node, 1);
            if (!copy) { vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory"); goto done; }
            cJSON_AddItemToArray(nodes, copy);
            continue;
        }

        output_tensor = vx_ptq_find(state, plan->output_tensor);
        produced = vx_ptq_activation(state, plan->output_tensor);
        node = cJSON_CreateObject();
        inputs = cJSON_CreateObject();
        params = cJSON_CreateObject();
        outputs = vx_ptq_out_descriptor(output_tensor, produced->quantized,
                                        activation_dtype);
        if (!node || !inputs || !outputs || !params) {
            cJSON_Delete(node); cJSON_Delete(inputs);
            cJSON_Delete(outputs); cJSON_Delete(params);
            vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            goto done;
        }
        for (param_index = 0; param_index < plan->param_count; param_index++) {
            const VxPtqCanonicalParam* param = &plan->params[param_index];
            int added;
            if (param->kind == VX_PTQ_PARAM_STRING) {
                added = cJSON_AddStringToObject(params, param->name,
                                                param->text) != NULL;
            } else if (param->kind == VX_PTQ_PARAM_INT) {
                added = cJSON_AddNumberToObject(params, param->name,
                                                param->number) != NULL;
            } else if (param->kind == VX_PTQ_PARAM_BOOL) {
                added = cJSON_AddBoolToObject(params, param->name,
                                              param->number != 0.0) != NULL;
            } else if (param->kind == VX_PTQ_PARAM_INT_ARRAY) {
                cJSON* list = cJSON_CreateArray();
                size_t value_index;
                added = list != NULL;
                for (value_index = 0;
                     added && value_index < param->value_count; value_index++) {
                    cJSON* value = cJSON_CreateNumber(
                        (double)param->values[value_index]);
                    if (!value) { added = 0; break; }
                    cJSON_AddItemToArray(list, value);
                }
                if (added) {
                    added = cJSON_AddItemToObject(params, param->name, list);
                }
                if (!added) cJSON_Delete(list);
            } else {
                cJSON* exact = vx_ptq_exact_number(param->number);
                if (!exact) {
                    cJSON_Delete(node); cJSON_Delete(inputs);
                    cJSON_Delete(outputs); cJSON_Delete(params);
                    vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                        "node '%s' parameter '%s' cannot be written exactly by "
                        "this build's number formatting; a template that "
                        "rounds it would mean something different from one "
                        "that does not",
                        plan->node_id, param->name);
                    goto done;
                }
                added = cJSON_AddItemToObject(params, param->name, exact);
                if (!added) cJSON_Delete(exact);
            }
            if (!added) {
                cJSON_Delete(node); cJSON_Delete(inputs);
                cJSON_Delete(outputs); cJSON_Delete(params);
                vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
                goto done;
            }
        }

        if (VX_PTQ_ROLE_PACKS_WEIGHT(plan->role)) {
            const VxPtqActivation* consumed =
                vx_ptq_activation(state, plan->input_tensor);
            cJSON_AddStringToObject(inputs, "input", consumed->quantized);
            cJSON_AddStringToObject(inputs, "weight", plan->packed_weight);
            cJSON_AddStringToObject(inputs, "bias", plan->packed_bias);
        } else {
            /* Re-point the activation ports; copy every other operand as it
             * stands, because a byte node still reads its float parameters. */
            cJSON* source_inputs =
                cJSON_GetObjectItemCaseSensitive(source_node, "inputs");
            cJSON* port;
            cJSON_ArrayForEach(port, source_inputs) {
                const VxPtqActivation* replacement = NULL;
                size_t which;
                if (!cJSON_IsString(port)) continue;
                for (which = 0u; which < VX_PTQ_MAX_ACTIVATION_PORTS; which++) {
                    const char* named = plan->byte_op->activation_ports[which];
                    if (!named) break;
                    if (strcmp(named, port->string) == 0) {
                        replacement = vx_ptq_activation(state, port->valuestring);
                        break;
                    }
                }
                cJSON_AddStringToObject(inputs, port->string,
                                        replacement ? replacement->quantized
                                                    : port->valuestring);
            }
        }

        cJSON_AddStringToObject(node, "id", plan->node_id);
        cJSON_AddStringToObject(node, "opType", vx_operator_kind_name(plan->quantized_op));
        cJSON_AddItemToObject(node, "inputs", inputs);
        cJSON_AddItemToObject(node, "outputs", outputs);
        cJSON_AddItemToObject(node, "params", params);
        cJSON_AddItemToArray(nodes, node);
    }

    /* Where float leaves. A graph output, or anything a retained float node
     * still reads, has to come back across the boundary. */
    for (index = 0; index < state->activation_count; index++) {
        const VxPtqActivation* activation = &state->activations[index];
        const VxPtqTensor* tensor = vx_ptq_find(state, activation->source);
        cJSON* node;
        cJSON* inputs;
        cJSON* outputs;
        char node_id[VX_PTQ_AUTHORING_NAME_CAPACITY];

        if (!tensor->is_graph_output) continue;
        if (tensor->producer < 0 ||
            state->plans[tensor->producer].role == VX_PTQ_ROLE_RETAINED) {
            continue;
        }
        if (vx_ptq_reserve(state, state->plans[tensor->producer].node_id,
                           "dequantize", node_id) != 0) {
            vx_ptq_fail(state->out, VX_STATUS_INVALID_GRAPH,
                        "cannot name the dequantize boundary for '%s'",
                        tensor->name);
            goto done;
        }
        node = cJSON_CreateObject();
        inputs = cJSON_CreateObject();
        outputs = vx_ptq_out_descriptor(tensor, tensor->name, tensor->dtype);
        if (!node || !inputs || !outputs) {
            cJSON_Delete(node); cJSON_Delete(inputs); cJSON_Delete(outputs);
            vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            goto done;
        }
        cJSON_AddStringToObject(inputs, "input", activation->quantized);
        cJSON_AddStringToObject(inputs, "scale", activation->scale);
        cJSON_AddStringToObject(inputs, "zero_point", activation->zero_point);
        cJSON_AddStringToObject(node, "id", node_id);
        cJSON_AddStringToObject(node, "opType", "DequantizeLinear");
        cJSON_AddItemToObject(node, "inputs", inputs);
        cJSON_AddItemToObject(node, "outputs", outputs);
        cJSON_AddItemToObject(node, "params", cJSON_CreateObject());
        cJSON_AddItemToArray(nodes, node);
    }

    /* Affines. Activations are per-tensor; weights are per-output-channel,
     * which is what keeps one badly scaled column from setting the range for
     * the whole matrix. */
    for (index = 0; index < state->activation_count; index++) {
        const VxPtqActivation* activation = &state->activations[index];
        if (vx_ptq_add_affine(affines, activation->quantized, 0, 0,
                              activation->scale, activation->zero_point) != 0) {
            vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            goto done;
        }
    }
    for (index = 0; index < state->node_count; index++) {
        const VxPtqNodePlan* plan = &state->plans[index];
        if (!VX_PTQ_ROLE_PACKS_WEIGHT(plan->role)) continue;
        if (vx_ptq_add_affine(affines, plan->packed_weight, 1,
                              plan->weight_axis, plan->weight_scale,
                              plan->weight_zero_point) != 0) {
            vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            goto done;
        }
    }
    (void)weight_dtype;

    if (!cJSON_AddStringToObject(quantization, "format",
                                 "volvox-affine-safetensors/v1")) {
        vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
        goto done;
    }
    cJSON_AddItemToObject(quantization, "tensors", affines);
    affines = NULL;

    cJSON_DeleteItemFromObjectCaseSensitive(root, "nodes");
    cJSON_AddItemToObject(root, "nodes", nodes);
    nodes = NULL;
    cJSON_DeleteItemFromObjectCaseSensitive(root, "quantization");
    cJSON_AddItemToObject(root, "quantization", quantization);
    quantization = NULL;

    rendered = cJSON_Print(root);
    if (!rendered) { vx_ptq_fail(state->out, VX_STATUS_OUT_OF_MEMORY, "out of memory"); goto done; }
    *rendered_out = rendered;
    rendered = NULL;
    status = 0;

done:
    free(rendered);
    cJSON_Delete(affines);
    cJSON_Delete(quantization);
    cJSON_Delete(nodes);
    cJSON_Delete(root);
    return status;
}

/* --- publishing the plan --------------------------------------------------- */

static int vx_ptq_publish(VxPtqAuthoringState* state) {
    VxPtqAuthored* out = state->out;
    size_t index;
    size_t layer_index = 0u;

    out->observers = (VxPtqAuthoredObserver*)calloc(
        state->activation_count ? state->activation_count : 1u,
        sizeof(*out->observers));
    out->layers = (VxPtqAuthoredLayer*)calloc(
        state->node_count ? state->node_count : 1u, sizeof(*out->layers));
    if (!out->observers || !out->layers) {
        return vx_ptq_fail(out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
    }

    for (index = 0; index < state->activation_count; index++) {
        VxPtqAuthoredObserver* observer = &out->observers[index];
        snprintf(observer->tensor_name, sizeof(observer->tensor_name), "%s",
                 state->activations[index].source);
        snprintf(observer->quantized_tensor_name,
                 sizeof(observer->quantized_tensor_name), "%s",
                 state->activations[index].quantized);
        observer->storage = state->config->activation_storage;
        observer->scheme = state->config->activation_scheme;
    }
    out->observer_count = state->activation_count;

    for (index = 0; index < state->node_count; index++) {
        const VxPtqNodePlan* plan = &state->plans[index];
        VxPtqAuthoredLayer* layer;
        if (!VX_PTQ_ROLE_PACKS_WEIGHT(plan->role)) continue;
        layer = &out->layers[layer_index++];
        layer->kind = plan->role == VX_PTQ_ROLE_CONV
            ? VX_PTQ_AUTHORED_LAYER_QCONV2D
            : VX_PTQ_AUTHORED_LAYER_QLINEAR;
        layer->node_index = (int32_t)index;
        layer->weight_axis = plan->weight_axis;
        snprintf(layer->node_id, sizeof(layer->node_id), "%s", plan->node_id);
        snprintf(layer->input_tensor_name, sizeof(layer->input_tensor_name),
                 "%s", plan->input_tensor);
        snprintf(layer->output_tensor_name, sizeof(layer->output_tensor_name),
                 "%s", plan->output_tensor);
        snprintf(layer->source_weight_name, sizeof(layer->source_weight_name),
                 "%s", plan->weight_tensor);
        snprintf(layer->packed_weight_name, sizeof(layer->packed_weight_name),
                 "%s", plan->packed_weight);
        if (plan->bias_tensor) {
            snprintf(layer->source_bias_name, sizeof(layer->source_bias_name),
                     "%s", plan->bias_tensor);
            snprintf(layer->packed_bias_name, sizeof(layer->packed_bias_name),
                     "%s", plan->packed_bias);
        }
    }
    out->layer_count = layer_index;
    return 0;
}

/* --- entry point ------------------------------------------------------------ */

int vx_ptq_author_graph(const char* source_graph_json,
                        size_t source_graph_length,
                        const VxPtqWeightBlob* weights,
                        size_t weight_count,
                        const VxPtqAuthoringConfig* config,
                        char** template_json_out,
                        size_t* template_length_out,
                        VxPtqAuthored* out) {
    VxPtqAuthoringConfig defaults = VX_PTQ_AUTHORING_CONFIG_INIT;
    VxPtqAuthoringState state;
    SafetensorsBorrowedBytes* shards = NULL;
    size_t parsed = 0u;
    char* rendered = NULL;
    size_t index;
    int status = -1;

    if (!out) return -1;
    vx_ptq_authored_init(out);
    if (template_json_out) *template_json_out = NULL;
    if (template_length_out) *template_length_out = 0u;
    if (!source_graph_json || !source_graph_length || !template_json_out) {
        return vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "authoring needs a source graph and somewhere "
                                "to put the template");
    }
    if (!config) config = &defaults;
    if (config->struct_size != sizeof(VxPtqAuthoringConfig)) {
        return vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "authoring config has an unexpected size");
    }
    if (config->weight_storage != VX_PTQ_STORAGE_I8) {
        return vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH,
            "weights are packed as int8; a zero point on a weight survives "
            "into the accumulator");
    }

    memset(&state, 0, sizeof(state));
    vx_ptq_name_set_init(&state.occupied);
    state.config = config;
    state.out = out;

    state.source = cJSON_ParseWithLength(source_graph_json,
                                         source_graph_length);
    if (!state.source) {
        vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "the source graph is not valid JSON");
        goto done;
    }
    {
        const char* format = vx_ptq_string(state.source, "format");
        if (!format || strcmp(format, "volvox-graph/v1") != 0) {
            vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "source graph must declare format volvox-graph/v1");
            goto done;
        }
    }
    state.nodes = cJSON_GetObjectItemCaseSensitive(state.source, "nodes");
    if (!cJSON_IsArray(state.nodes)) {
        vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "source graph has no node array");
        goto done;
    }
    state.node_count = (size_t)cJSON_GetArraySize(state.nodes);
    if (!state.node_count) {
        vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "source graph declares no nodes");
        goto done;
    }

    if (weight_count) {
        shards = (SafetensorsBorrowedBytes*)calloc(weight_count,
                                                   sizeof(*shards));
        if (!shards) { vx_ptq_fail(out, VX_STATUS_OUT_OF_MEMORY, "out of memory"); goto done; }
        for (parsed = 0u; parsed < weight_count; parsed++) {
            if (!weights || !weights[parsed].bytes ||
                safetensors_parse_borrowed_bytes(weights[parsed].bytes,
                                                 weights[parsed].byte_count,
                                                 &shards[parsed]) != 0) {
                vx_ptq_fail(out, VX_STATUS_INVALID_GRAPH, "weight shard %zu is not a safetensors file",
                            parsed);
                goto done;
            }
        }
    }

    if (vx_ptq_collect_tensors(&state, shards, parsed) != 0) goto done;

    state.plans = (VxPtqNodePlan*)calloc(state.node_count, sizeof(*state.plans));
    state.activations = (VxPtqActivation*)calloc(
        state.tensor_count ? state.tensor_count : 1u, sizeof(*state.activations));
    if (!state.plans || !state.activations) {
        vx_ptq_fail(out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
        goto done;
    }

    if (vx_ptq_qualify(&state) != 0) goto done;
    if (vx_ptq_name_payloads(&state) != 0) goto done;
    if (vx_ptq_collect_activations(&state) != 0) goto done;
    if (vx_ptq_emit(&state, &rendered) != 0) goto done;
    if (vx_ptq_publish(&state) != 0) goto done;
    *template_json_out = rendered;
    if (template_length_out) *template_length_out = strlen(rendered);
    rendered = NULL;
    status = 0;

done:
    if (status != 0) vx_ptq_authored_free(out);
    free(rendered);
    for (index = 0; index < parsed; index++) {
        safetensors_borrowed_bytes_free(&shards[index]);
    }
    free(shards);
    free(state.plans);
    free(state.activations);
    free(state.reserved);
    free(state.tensors);
    vx_ptq_name_set_free(&state.occupied);
    cJSON_Delete(state.source);
    return status;
}

void vx_ptq_authored_release_template(char* template_json) {
    free(template_json);
}

#if !defined(VOLVOXAI_PTQ_AUTHORING_NO_FILES)
/* The file-backed form. Everything above works on bytes, so this is only
 * reading and writing — which is exactly the part a WebAssembly build cannot
 * have, and the reason the split exists. */
int vx_ptq_write_template_file(const char* template_graph_path,
                               const char* template_json,
                               size_t template_length,
                               VxPtqAuthored* out) {
    FILE* handle;
    int write_failed;

    if (!out) return -1;
    if (!template_graph_path || !template_graph_path[0] || !template_json) {
        return vx_ptq_fail_as(out, VX_STATUS_INVALID_ARGUMENT,
                              "writing a template needs a path and bytes");
    }
    handle = fopen(template_graph_path, "wb");
    if (!handle) {
        return vx_ptq_fail_as(out, VX_STATUS_IO_ERROR,
                              "cannot write template '%s'",
                              template_graph_path);
    }
    write_failed =
        (template_length != 0u &&
         fwrite(template_json, 1u, template_length, handle) !=
             template_length) ||
        fputc('\n', handle) == EOF;
    if (write_failed) {
        (void)fclose(handle);
        return vx_ptq_fail_as(out, VX_STATUS_IO_ERROR,
                              "cannot write template '%s'",
                              template_graph_path);
    }
    if (fclose(handle) != 0) {
        return vx_ptq_fail_as(out, VX_STATUS_IO_ERROR,
                              "cannot flush template '%s'",
                              template_graph_path);
    }
    return 0;
}

static int vx_ptq_author_template_paths(
        const char* source_graph_path,
        const char* const* weight_paths,
        size_t weight_path_count,
        const char* template_graph_path,
        char** template_json_out,
        size_t* template_length_out,
        const VxPtqAuthoringConfig* config,
        VxPtqAuthored* out) {
    VxPtqWeightBlob* blobs = NULL;
    unsigned char** shard_bytes = NULL;
    size_t* shard_sizes = NULL;
    char* source = NULL;
    char* rendered = NULL;
    size_t rendered_length = 0u;
    size_t loaded = 0u;
    size_t index;
    int status = -1;

    if (template_json_out) *template_json_out = NULL;
    if (template_length_out) *template_length_out = 0u;
    if (!out) return -1;
    vx_ptq_authored_init(out);
    if (!source_graph_path || !source_graph_path[0] ||
        ((!template_graph_path || !template_graph_path[0]) &&
         !template_json_out)) {
        return vx_ptq_fail_as(
            out, VX_STATUS_INVALID_ARGUMENT,
            "authoring needs a source path and an output destination");
    }

    source = vx_ptq_read_file(source_graph_path);
    if (!source) {
        return vx_ptq_fail_as(out, VX_STATUS_IO_ERROR,
                              "cannot read source graph '%s'",
                              source_graph_path);
    }

    if (weight_path_count) {
        blobs = (VxPtqWeightBlob*)calloc(weight_path_count, sizeof(*blobs));
        shard_bytes = (unsigned char**)calloc(weight_path_count,
                                              sizeof(*shard_bytes));
        shard_sizes = (size_t*)calloc(weight_path_count, sizeof(*shard_sizes));
        if (!blobs || !shard_bytes || !shard_sizes) {
            vx_ptq_fail(out, VX_STATUS_OUT_OF_MEMORY, "out of memory");
            goto done;
        }
        for (loaded = 0u; loaded < weight_path_count; loaded++) {
            if (vx_ptq_read_binary_file(weight_paths[loaded],
                                        &shard_bytes[loaded],
                                        &shard_sizes[loaded]) != 0) {
                vx_ptq_fail_as(out, VX_STATUS_IO_ERROR,
                               "cannot read weights '%s'",
                               weight_paths[loaded]);
                goto done;
            }
            blobs[loaded].bytes = shard_bytes[loaded];
            blobs[loaded].byte_count = shard_sizes[loaded];
        }
    }

    if (vx_ptq_author_graph(source, strlen(source), blobs, loaded, config,
                            &rendered, &rendered_length, out) != 0) {
        goto done;
    }

    if (template_graph_path && template_graph_path[0]) {
        if (vx_ptq_write_template_file(template_graph_path, rendered,
                                       rendered_length, out) != 0) goto done;
    } else {
        *template_json_out = rendered;
        if (template_length_out) *template_length_out = rendered_length;
        rendered = NULL;
    }
    status = 0;

done:
    if (status != 0) vx_ptq_authored_free(out);
    vx_ptq_authored_release_template(rendered);
    for (index = 0; index < loaded; index++) free(shard_bytes[index]);
    free(shard_bytes);
    free(shard_sizes);
    free(blobs);
    free(source);
    return status;
}

int vx_ptq_author_template(const char* source_graph_path,
                           const char* const* weight_paths,
                           size_t weight_path_count,
                           const char* template_graph_path,
                           const VxPtqAuthoringConfig* config,
                           VxPtqAuthored* out) {
    return vx_ptq_author_template_paths(
        source_graph_path, weight_paths, weight_path_count, template_graph_path,
        NULL, NULL, config, out);
}

int vx_ptq_author_template_to_memory(
        const char* source_graph_path,
        const char* const* weight_paths,
        size_t weight_path_count,
        const VxPtqAuthoringConfig* config,
        char** template_json_out,
        size_t* template_length_out,
        VxPtqAuthored* out) {
    return vx_ptq_author_template_paths(
        source_graph_path, weight_paths, weight_path_count, NULL,
        template_json_out, template_length_out, config, out);
}
#endif /* !VOLVOXAI_PTQ_AUTHORING_NO_FILES */

void vx_ptq_authored_init(VxPtqAuthored* out) {
    if (!out) return;
    out->observers = NULL;
    out->observer_count = 0u;
    out->layers = NULL;
    out->layer_count = 0u;
    out->quantized_nodes = 0u;
    out->retained_float_nodes = 0u;
    out->status = VX_STATUS_OK;
    out->message[0] = '\0';
}

void vx_ptq_authored_free(VxPtqAuthored* out) {
    if (!out) return;
    free(out->observers);
    free(out->layers);
    /* init() clears message too, which is wrong on a failure path — the
     * caller still needs to read it. Preserve it. */
    out->observers = NULL;
    out->observer_count = 0u;
    out->layers = NULL;
    out->layer_count = 0u;
}
