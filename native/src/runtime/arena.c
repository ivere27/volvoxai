#include "engine_internal.h"
#include "vx_platform.h"
#include "engine_core.h"
#include "incremental_runtime.h"
#include "activation_plan.h"
#include "graph_plan.h"
#if VOLVOXAI_ENABLE_WEBGPU
#include "webgpu_domain.h"
#endif
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif
#if VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void* vx_arena_allocate(size_t bytes) {
    void* pointer = malloc(bytes);
    VxMemoryObserver* observer = &vx_engine_state_current()->memory_observer;
    if (pointer && observer->owner)
        vx_memory_record(observer, VX_MEMORY_HOST_ARENA, (uint64_t)(uintptr_t)pointer,
            bytes, VX_TRACE_MEMORY_ACTION_ALLOCATE);
    return pointer;
}
static void vx_arena_release(void* pointer) {
    uint64_t key = (uint64_t)(uintptr_t)pointer;
    free(pointer);
    VxMemoryObserver* observer = &vx_engine_state_current()->memory_observer;
    if (key && observer->owner)
        vx_memory_record(observer, VX_MEMORY_HOST_ARENA, key, 0, VX_TRACE_MEMORY_ACTION_FREE);
}
void vx_engine_memory_begin(VxEngineState* state, const VxTraceScope* scope) {
    if (!state) return;
#if VOLVOXAI_ENABLE_WEBGPU
    if (state->webgpu_state) vx_webgpu_memory_begin((VxTraceScope*)scope);
#endif
    if (!vx_memory_observer_attach(&state->memory_observer, scope)) return;
    state->memory_observer_clear = vx_memory_observer_clear;
    if (state->dynamic_shape_reserved_arena)
        vx_memory_record(&state->memory_observer, VX_MEMORY_HOST_ARENA,
            (uint64_t)(uintptr_t)state->dynamic_shape_reserved_arena,
            state->dynamic_arena_capacity_bytes, VX_TRACE_MEMORY_ACTION_EXISTING);
    for (int b = 0; b < state->arena_buffer_count; b++) {
        size_t capacity = state->dynamic_arena_active ? state->dynamic_arena_capacity_bytes : 0;
        if (!state->dynamic_arena_active) for (int t = 0; t < state->arena_tensor_count; t++) {
            const T* tensor = &state->tensors[state->arena_tensor_indices[t]];
            if ((void*)tensor->data == state->arena_buffers[b]) {
                size_t bytes = (size_t)tensor->numel * tensor->elem_size;
                if (bytes > capacity) capacity = bytes;
            }
        }
        vx_memory_record(&state->memory_observer, VX_MEMORY_HOST_ARENA,
            (uint64_t)(uintptr_t)state->arena_buffers[b], capacity, VX_TRACE_MEMORY_ACTION_EXISTING);
    }
    VxEngineStateScope previous = vx_engine_state_scope_enter(state);
#if VOLVOXAI_ENABLE_CUDA
    cuda_memory_inventory();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    opengl_memory_inventory();
#endif
#if VOLVOXAI_ENABLE_VULKAN
    vk_memory_inventory();
#endif
    vx_engine_state_scope_leave(previous);
}

// ---- Activation arena: reuse physical buffers across non-overlapping lifetimes ----
// Similar to planned tensor arenas in optimized inference runtimes. Transient F32 and
// I8/U8 activations are pooled: owned (calloc'd by us -> NOT weights, which
// point into the blob), with a non-skipped producer node (excludes graph
// inputs/constants), at least one consumer (excludes dead tensors), not declared as a
// graph output, and not aliased by another tensor (excludes Reshape/Flatten
// passthroughs). A declared output remains live until the runtime snapshots it after the
// whole graph, even when an in-graph consumer such as ArgMax has already read it.
// last-use scans ALL nodes INCLUDING skipped ones, so conv+add residuals and other
// fused-away consumers keep their buffer live conservatively. Cuts peak activation memory
// (~88 -> ~10 MB here) which removes most cold-start first-touch page faults.
/* Arena blocks are raw byte allocations. T.data uses the common float-pointer
 * field, but canonical W8A8 intermediates really own one byte per
 * element.  Keeping capacities in elements would reserve four times as much
 * memory and, more importantly, would make a mixed F32/I8 reuse plan reason
 * about the wrong unit. */
static void dynamic_shape_plan_clear(VxDynamicShapePlan* plan) {
    if (!plan) return;
    free(plan->signature);
    free(plan->tensor_indices);
    free(plan->offsets);
    free(plan->tensor_bytes);
    free(plan->tensor_shapes);
    memset(plan, 0, sizeof(*plan));
}

static void dynamic_shape_maximum_layout_clear(
        VxDynamicShapeMaximumLayout* layout) {
    if (!layout) return;
    free(layout->tensor_indices);
    free(layout->offsets);
    free(layout->tensor_bytes);
    free(layout->physical_offsets);
    free(layout->physical_capacities);
    memset(layout, 0, sizeof(*layout));
}

static void dynamic_shape_cache_clear(void) {
    VxEngineState* state = vx_engine_state_current();
    for (uint32_t index = 0; index < state->dynamic_shape_plan_capacity; index++)
        dynamic_shape_plan_clear(&state->dynamic_shape_plans[index]);
    free(state->dynamic_shape_plans);
    state->dynamic_shape_plans = NULL;
    state->dynamic_shape_plan_capacity = 0;
    state->dynamic_shape_plan_metadata_bytes = 0;
    state->dynamic_shape_cache_oversize_skips = 0;
    state->dynamic_shape_cache_bypasses = 0;
    state->dynamic_shape_plan_clock = 0;
}

static int dynamic_passthrough_node(const Node* node) {
    return node && node->skip &&
        ((node->operator_kind == VX_OP_RESHAPE) || (node->operator_kind == VX_OP_FLATTEN) ||
         (node->operator_kind == VX_OP_SQUEEZE) || (node->operator_kind == VX_OP_UNSQUEEZE) ||
         (node->operator_kind == VX_OP_DROPOUT) || (node->operator_kind == VX_OP_IDENTITY));
}

static int arena_tensor_slot(int tensor_index) {
    for (int slot = 0; slot < g_arena_tensor_count; slot++)
        if (g_arena_tensor_indices[slot] == tensor_index) return slot;
    return -1;
}

/* Ordinary lifetime reuse must be split before an incremental prefill, but a
 * skipped storage-view node has no dispatch that can recreate its value. Keep
 * that intentional graph alias attached to its already restored source. */
static int arena_passthrough_source_slot(int target_slot) {
    int target_index;
    T* target;
    if (target_slot < 0 || target_slot >= g_arena_tensor_count) return -1;
    target_index = g_arena_tensor_indices[target_slot];
    if (target_index < 0 || target_index >= g_nt) return -1;
    target = &g_t[target_index];
    for (int node_index = 0; node_index < g_nn; node_index++) {
        Node* node = &g_n[node_index];
        T* source;
        int source_slot;
        int produces_target = 0;
        if (!dynamic_passthrough_node(node) || node->nin <= 0) continue;
        for (int output = 0; output < node->nout; output++)
            if (!strcmp(node->outs[output].name, target->name))
                produces_target = 1;
        if (!produces_target && strcmp(node->out, target->name)) continue;
        source = t_find(node->ins[0].name);
        source_slot = source ? arena_tensor_slot((int)(source - g_t)) : -1;
        if (source_slot < 0 || source_slot >= target_slot ||
            source->data != target->data || source->dtype != target->dtype ||
            source->numel < 0 || target->numel < 0 ||
            source->elem_size != target->elem_size ||
            source->numel != target->numel)
            return -1;
        return source_slot;
    }
    return -1;
}

void volvoxai_engine_free_arena(void) {
    VxEngineState* state = vx_engine_state_current();
    for (int i = 0; i < g_arena_nbufs; i++) vx_arena_release(g_arena_bufs[i]);
    free(g_arena_bufs);
    for (int i = 0; i < g_arena_tensor_count; i++) {
        int index = g_arena_tensor_indices[i];
        if (index < 0 || index >= g_nt) continue;
        g_t[index].data = NULL;
        g_t[index].owns = 0;
    }
    free(g_arena_tensor_indices);
    vx_arena_release(state->dynamic_shape_reserved_arena);
    state->dynamic_shape_reserved_arena = NULL;
    g_arena_bufs = NULL;
    g_arena_nbufs = 0;
    state->arena_allocated_bytes = 0;
    g_arena_tensor_indices = NULL;
    g_arena_tensor_count = 0;
    state->dynamic_arena_active = 0;
    state->dynamic_arena_capacity_bytes = 0;
    state->dynamic_arena_current_bytes = 0;
    state->dynamic_arena_high_water_bytes = 0;
    state->dynamic_arena_grow_count = 0;
    state->dynamic_resource_generation = 0;
    dynamic_shape_cache_clear();
    dynamic_shape_maximum_layout_clear(
        &state->dynamic_shape_maximum_layout);
}

static int restore_arena_tensors(void) {
    if (g_arena_tensor_count <= 0) { volvoxai_engine_free_arena(); return 0; }
    void** replacements = (void**)calloc((size_t)g_arena_tensor_count, sizeof(void*));
    unsigned char* replacement_owns =
        (unsigned char*)calloc((size_t)g_arena_tensor_count, 1u);
    if (!replacements || !replacement_owns) {
        free(replacements);
        free(replacement_owns);
        return -1;
    }
    for (int i = 0; i < g_arena_tensor_count; i++) {
        int index = g_arena_tensor_indices[i];
        int source_slot;
        if (index < 0 || index >= g_nt) goto fail;
        T* tensor = &g_t[index];
        size_t bytes;
        if (!tensor->data || tensor->numel < 0 || !tensor->elem_size ||
            (size_t)tensor->numel > SIZE_MAX / tensor->elem_size) goto fail;
        bytes = (size_t)tensor->numel * tensor->elem_size;
        source_slot = arena_passthrough_source_slot(i);
        if (source_slot >= 0) {
            replacements[i] = replacements[source_slot];
            if (!replacements[i]) goto fail;
            continue;
        }
        /* A full incremental prefill extends every activation lifetime across
         * later steps. Two values that shared one ordinary-forward arena slot
         * must therefore receive independent storage here: preserving the
         * physical alias would let a later branch overwrite a skipped cached
         * value. This is a lifetime split, not a graph-level view alias. */
        replacements[i] = calloc(tensor->numel > 0 ? (size_t)tensor->numel : 1u, tensor->elem_size);
        if (!replacements[i]) goto fail;
        replacement_owns[i] = 1u;
        if (bytes) memcpy(replacements[i], tensor->data, bytes);
    }
    for (int i = 0; i < g_arena_nbufs; i++) vx_arena_release(g_arena_bufs[i]);
    free(g_arena_bufs);
    for (int i = 0; i < g_arena_tensor_count; i++) {
        T* tensor = &g_t[g_arena_tensor_indices[i]];
        tensor->data = (float*)replacements[i];
        tensor->owns = replacement_owns[i] ? 1 : 0;
    }
    free(replacements);
    free(replacement_owns);
    free(g_arena_tensor_indices);
    g_arena_bufs = NULL;
    g_arena_nbufs = 0;
    g_arena_tensor_indices = NULL;
    g_arena_tensor_count = 0;
    vx_engine_state_current()->dynamic_arena_active = 0;
    vx_engine_state_current()->arena_allocated_bytes = 0;
    vx_engine_state_current()->dynamic_arena_capacity_bytes = 0;
    vx_engine_state_current()->dynamic_arena_current_bytes = 0;
    return 0;
fail:
    for (int i = 0; i < g_arena_tensor_count; i++)
        if (replacement_owns[i]) free(replacements[i]);
    free(replacements);
    free(replacement_owns);
    return -1;
}

static int t_index_by_name(const char* name) {
    T* t = t_find(name);
    return t ? (int)(t - g_t) : -1;
}

static int tensor_is_declared_graph_output(const char* name) {
    cJSON* outputs = g_graph_root
        ? cJSON_GetObjectItemCaseSensitive(g_graph_root, "outputs") : NULL;
    if (!name || !cJSON_IsArray(outputs)) return 0;
    for (cJSON* output = outputs->child; output; output = output->next) {
        if (cJSON_IsString(output) && output->valuestring &&
            !strcmp(output->valuestring, name))
            return 1;
    }
    return 0;
}

static size_t dynamic_dtype_size(int dtype) {
    switch (dtype) {
        case T_I8:
        case T_U8: return 1u;
        case T_F32:
        case T_I32: return 4u;
        default: return 0u;
    }
}

static int dynamic_descriptor_bytes(
        const VolvoxAIEngineResolvedTensor* descriptor,
        size_t* out_bytes) {
    size_t count = 1u;
    size_t element_size;
    if (!descriptor || !descriptor->name || !descriptor->name[0] ||
        descriptor->rank < 0 || descriptor->rank > 8 ||
        !(element_size = dynamic_dtype_size(descriptor->dtype))) return -1;
    for (int axis = 0; axis < descriptor->rank; axis++) {
        int extent = descriptor->shape[axis];
        if (extent <= 0 || (size_t)extent > SIZE_MAX / count ||
            (size_t)extent > (size_t)LONG_MAX / count) return -1;
        count *= (size_t)extent;
    }
    if (count > SIZE_MAX / element_size) return -1;
    if (out_bytes) *out_bytes = count * element_size;
    return 0;
}

static int dynamic_descriptor_index(
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        const char* name) {
    if (!name) return -1;
    for (size_t index = 0; index < descriptor_count; index++)
        if (descriptors[index].name &&
            !strcmp(descriptors[index].name, name)) return (int)index;
    return -1;
}

static int dynamic_maximum_descriptor_index(
        const VolvoxAIEngineMaximumTensor* descriptors,
        size_t descriptor_count,
        const char* name) {
    if (!name) return -1;
    for (size_t index = 0; index < descriptor_count; index++)
        if (descriptors[index].name &&
            !strcmp(descriptors[index].name, name)) return (int)index;
    return -1;
}

static int dynamic_quantized_norm_stats_bound(
        const Node* node,
        const VolvoxAIEngineMaximumTensor* descriptors,
        size_t descriptor_count,
        size_t* bytes_out) {
    const T* input = NULL;
    const int* maximum_shape;
    size_t activation_elements;
    size_t element_size;
    size_t units;
    int rank;
    int descriptor_index;
    if (!node || !descriptors || !descriptor_count || !bytes_out) return -1;
    for (int index = 0; index < node->nin; index++) {
        if (node->ins[index].port != VX_PORT_INPUT) continue;
        if (input) return -1;
        input = t_find(node->ins[index].name);
    }
    if (!input || (input->dtype != T_I8 && input->dtype != T_U8) ||
        !(element_size = dynamic_dtype_size(input->dtype))) return -1;
    descriptor_index = dynamic_maximum_descriptor_index(
        descriptors, descriptor_count, input->name);
    if (descriptor_index >= 0) {
        const VolvoxAIEngineMaximumTensor* descriptor =
            &descriptors[descriptor_index];
        if (descriptor->dtype != input->dtype ||
            descriptor->rank <= 0 || descriptor->rank > 8 ||
            !descriptor->maximum_byte_size ||
            descriptor->maximum_byte_size % element_size) return -1;
        rank = descriptor->rank;
        maximum_shape = descriptor->maximum_shape;
        activation_elements = descriptor->maximum_byte_size / element_size;
    } else {
        if (input->numel <= 0 || !input->elem_size ||
            input->elem_size != element_size ||
            (size_t)input->numel > SIZE_MAX / element_size) return -1;
        rank = input->ndim;
        maximum_shape = input->shape;
        activation_elements = (size_t)input->numel;
    }
    if (rank <= 0 || rank > 8) return -1;
    for (int axis = 0; axis < rank; axis++)
        if (maximum_shape[axis] <= 0) return -1;
    if (node->operator_kind == VX_OP_Q_LAYER_NORM) {
        const size_t feature = (size_t)maximum_shape[rank - 1];
        if (!feature || activation_elements % feature) return -1;
        units = activation_elements / feature;
    } else {
        const VxCachedNodeParam* groups_param;
        int groups = 1;
        if (rank != 4) return -1;
        groups_param = vx_node_param(node, VX_NODE_PARAM_NUM_GROUPS);
        if (!groups_param || groups_param->kind == VX_NODE_PARAM_ABSENT)
            groups_param = vx_node_param(node, VX_NODE_PARAM_GROUPS);
        if (groups_param && groups_param->kind != VX_NODE_PARAM_ABSENT) {
            if (groups_param->kind != VX_NODE_PARAM_I32 ||
                groups_param->value.i32 <= 0) return -1;
            groups = groups_param->value.i32;
        }
        if ((size_t)maximum_shape[0] > SIZE_MAX / (size_t)groups)
            return -1;
        units = (size_t)maximum_shape[0] * (size_t)groups;
    }
    /* Both kernels keep one mean and one inverse standard deviation for each
     * normalization unit. Shape-domain proof fixes the final feature/group
     * geometry, so this is the exact maximum rather than an element-count
     * overestimate that could spuriously exhaust Vulkan's fixed scratch. */
    if (!units || units > SIZE_MAX / (2u * sizeof(float))) return -1;
    *bytes_out = units * 2u * sizeof(float);
    return *bytes_out ? 0 : -1;
}

static int dynamic_align_size(size_t value, size_t alignment,
                              size_t* out) {
    size_t remainder;
    size_t adjustment;
    if (!alignment || !out) return -1;
    remainder = value % alignment;
    adjustment = remainder ? alignment - remainder : 0u;
    if (adjustment > SIZE_MAX - value) return -1;
    *out = value + adjustment;
    return 0;
}

typedef struct VxPortableGraphTensorView {
    const uint8_t* name;
    uint32_t name_length;
    uint32_t kind;
    uint32_t birth;
    uint32_t last_use;
    uint32_t alias_root;
    uint32_t flags;
    uint32_t widened_birth;
    int tensor_index;
    int descriptor_index;
    int32_t dtype;
    uint64_t size_bytes;
} VxPortableGraphTensorView;

typedef struct VxPortableGraphWireView {
    const uint8_t* request;
    uint32_t request_bytes;
    const uint8_t* response;
    uint32_t response_bytes;
    uint32_t source_offset;
    uint32_t source_count;
    uint32_t request_edge_offset;
    uint32_t request_edge_count;
    uint32_t string_offset;
    uint32_t string_bytes;
    uint32_t tensor_offset;
    uint32_t tensor_count;
    uint32_t node_count;
} VxPortableGraphWireView;

static uint32_t dynamic_wire_read_u32(const uint8_t* source) {
    return (uint32_t)source[0] |
           ((uint32_t)source[1] << 8u) |
           ((uint32_t)source[2] << 16u) |
           ((uint32_t)source[3] << 24u);
}

static uint64_t dynamic_wire_read_u64(const uint8_t* source) {
    return (uint64_t)dynamic_wire_read_u32(source) |
           ((uint64_t)dynamic_wire_read_u32(source + 4u) << 32u);
}

static void dynamic_wire_write_u32(uint8_t* destination, uint32_t value) {
    destination[0] = (uint8_t)value;
    destination[1] = (uint8_t)(value >> 8u);
    destination[2] = (uint8_t)(value >> 16u);
    destination[3] = (uint8_t)(value >> 24u);
}

static void dynamic_wire_write_u64(uint8_t* destination, uint64_t value) {
    dynamic_wire_write_u32(destination, (uint32_t)value);
    dynamic_wire_write_u32(destination + 4u, (uint32_t)(value >> 32u));
}

static int dynamic_u32_layout_add(uint64_t* cursor, uint64_t amount) {
    if (!cursor || amount > UINT32_MAX || *cursor > UINT32_MAX - amount)
        return -1;
    *cursor += amount;
    return 0;
}

static int dynamic_wire_string_slice(
        const VxPortableGraphWireView* view,
        uint32_t offset,
        uint32_t length,
        const uint8_t** name) {
    if (!view || !name || !length || offset > view->string_bytes ||
        length > view->string_bytes - offset)
        return -1;
    *name = view->request + view->string_offset + offset;
    return 0;
}

static int dynamic_name_slice_equal(const uint8_t* name,
                                    uint32_t name_length,
                                    const char* value) {
    size_t length;
    if (!name || !name_length || !value) return 0;
    length = strlen(value);
    return length == (size_t)name_length &&
        !memcmp(name, value, (size_t)name_length);
}

static int dynamic_tensor_index_slice(const uint8_t* name,
                                      uint32_t name_length) {
    int found = -1;
    for (int index = 0; index < g_nt; index++) {
        if (!dynamic_name_slice_equal(name, name_length, g_t[index].name))
            continue;
        if (found >= 0) return -1;
        found = index;
    }
    return found;
}

static int dynamic_graph_index_name(
        const VxPortableGraphTensorView* tensors,
        uint32_t tensor_count,
        const char* name) {
    int found = -1;
    if (!tensors || !name || !name[0]) return -1;
    for (uint32_t index = 0u; index < tensor_count; index++) {
        if (!dynamic_name_slice_equal(tensors[index].name,
                                      tensors[index].name_length, name))
            continue;
        if (found >= 0) return -1;
        found = (int)index;
    }
    return found;
}

static int dynamic_portable_dtype_rank(int32_t dtype) {
    switch (dtype) {
        case VX_DTYPE_F32: return 0;
        case VX_DTYPE_I32: return 1;
        case VX_DTYPE_I8: return 2;
        case VX_DTYPE_U8: return 3;
        default: return -1;
    }
}

static int dynamic_graph_wire_view_load(
        const uint8_t* request,
        uint32_t request_bytes,
        const uint8_t* response,
        uint32_t response_bytes,
        VxPortableGraphWireView* view) {
    uint32_t request_node_offset;
    uint32_t request_node_count;
    uint32_t request_output_offset;
    uint32_t request_output_count;
    uint32_t response_node_offset;
    uint32_t response_edge_offset;
    uint32_t response_edge_count;
    uint32_t response_output_offset;
    uint32_t response_output_count;
    uint64_t cursor;
    if (!request || request_bytes < sizeof(VxGraphPlanRequestV1) ||
        !response || response_bytes < sizeof(VxGraphPlanResponseV1) ||
        !view)
        return -1;
    memset(view, 0, sizeof(*view));
    if (dynamic_wire_read_u32(request + 0u) !=
            VOLVOXAI_GRAPH_PLAN_REQUEST_MAGIC ||
        dynamic_wire_read_u32(request + 4u) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        dynamic_wire_read_u32(request + 8u) != request_bytes ||
        dynamic_wire_read_u32(request + 12u) != 0u ||
        dynamic_wire_read_u32(request + 56u) != 0u ||
        dynamic_wire_read_u32(request + 60u) != 0u ||
        dynamic_wire_read_u32(response + 0u) !=
            VOLVOXAI_GRAPH_PLAN_RESPONSE_MAGIC ||
        dynamic_wire_read_u32(response + 4u) !=
            VOLVOXAI_GRAPH_PLAN_ABI_VERSION ||
        (int32_t)dynamic_wire_read_u32(response + 8u) !=
            VX_GRAPH_PLAN_STATUS_OK ||
        (int32_t)dynamic_wire_read_u32(response + 12u) !=
            VX_GRAPH_PLAN_ERROR_NONE ||
        dynamic_wire_read_u32(response + 16u) !=
            VX_GRAPH_PLAN_ERROR_SECTION_NONE ||
        dynamic_wire_read_u32(response + 20u) != VX_GRAPH_PLAN_INDEX_NONE ||
        dynamic_wire_read_u32(response + 24u) != VX_GRAPH_PLAN_INDEX_NONE ||
        dynamic_wire_read_u32(response + 28u) != response_bytes ||
        dynamic_wire_read_u32(response + 32u) != response_bytes ||
        dynamic_wire_read_u32(response + 72u) != 0u ||
        dynamic_wire_read_u32(response + 76u) != 0u)
        return -1;

    view->request = request;
    view->request_bytes = request_bytes;
    view->response = response;
    view->response_bytes = response_bytes;
    view->source_offset = dynamic_wire_read_u32(request + 16u);
    view->source_count = dynamic_wire_read_u32(request + 20u);
    request_node_offset = dynamic_wire_read_u32(request + 24u);
    request_node_count = dynamic_wire_read_u32(request + 28u);
    view->request_edge_offset = dynamic_wire_read_u32(request + 32u);
    view->request_edge_count = dynamic_wire_read_u32(request + 36u);
    request_output_offset = dynamic_wire_read_u32(request + 40u);
    request_output_count = dynamic_wire_read_u32(request + 44u);
    view->string_offset = dynamic_wire_read_u32(request + 48u);
    view->string_bytes = dynamic_wire_read_u32(request + 52u);
    cursor = sizeof(VxGraphPlanRequestV1);
    if (view->source_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)view->source_count *
                         sizeof(VxGraphPlanSourceV1)) != 0 ||
        request_node_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)request_node_count *
                         sizeof(VxGraphPlanNodeV1)) != 0 ||
        view->request_edge_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)view->request_edge_count *
                         sizeof(VxGraphPlanEdgeV1)) != 0 ||
        request_output_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)request_output_count *
                         sizeof(VxGraphPlanPublicOutputV1)) != 0 ||
        view->string_offset != cursor ||
        dynamic_u32_layout_add(&cursor, view->string_bytes) != 0 ||
        cursor != request_bytes)
        return -1;

    view->tensor_offset = dynamic_wire_read_u32(response + 40u);
    view->tensor_count = dynamic_wire_read_u32(response + 44u);
    response_node_offset = dynamic_wire_read_u32(response + 48u);
    view->node_count = dynamic_wire_read_u32(response + 52u);
    response_edge_offset = dynamic_wire_read_u32(response + 56u);
    response_edge_count = dynamic_wire_read_u32(response + 60u);
    response_output_offset = dynamic_wire_read_u32(response + 64u);
    response_output_count = dynamic_wire_read_u32(response + 68u);
    cursor = sizeof(VxGraphPlanResponseV1);
    if (!view->tensor_count || !response_output_count ||
        request_node_count != view->node_count ||
        view->request_edge_count != response_edge_count ||
        request_output_count != response_output_count ||
        view->tensor_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)view->tensor_count *
                         sizeof(VxGraphPlanTensorV1)) != 0 ||
        response_node_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)view->node_count *
                         sizeof(VxGraphPlanStepV1)) != 0 ||
        response_edge_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)response_edge_count *
                         sizeof(VxGraphPlanResolvedEdgeV1)) != 0 ||
        response_output_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)response_output_count *
                         sizeof(VxGraphPlanResolvedOutputV1)) != 0 ||
        cursor != response_bytes || view->node_count != (uint32_t)g_nn)
        return -1;
    return 0;
}

static int dynamic_descriptor_index_slice(
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        const uint8_t* name,
        uint32_t name_length) {
    int found = -1;
    for (size_t index = 0; index < descriptor_count; index++) {
        if (!dynamic_name_slice_equal(name, name_length,
                                      descriptors[index].name))
            continue;
        if (found >= 0 || index > (size_t)INT_MAX) return -1;
        found = (int)index;
    }
    return found;
}

static int dynamic_weight_tensor_bytes(const T* tensor, uint64_t* bytes) {
    if (!tensor || !bytes || tensor->numel <= 0) return -1;
    if (tensor->elem_size) {
        if ((uint64_t)tensor->numel > UINT64_MAX / tensor->elem_size)
            return -1;
        *bytes = (uint64_t)tensor->numel * tensor->elem_size;
        return *bytes ? 0 : -1;
    }
    for (int file = g_weight_file_count - 1; file >= 0; file--) {
        const SafetensorsTensor* stored =
            safetensors_find_tensor(&g_weight_files[file], tensor->name);
        if (!stored) continue;
        if (!stored->nbytes) return -1;
        *bytes = stored->nbytes;
        return 0;
    }
    return -1;
}

static int dynamic_graph_tensor_views_load(
        const VxPortableGraphWireView* view,
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        VxPortableGraphTensorView* tensors,
        uint32_t* descriptor_graph_indices) {
    unsigned char* descriptor_seen = NULL;
    int result = -1;
    if (!view || !descriptors || !descriptor_count || !tensors ||
        !descriptor_graph_indices ||
        view->tensor_count > (uint32_t)INT_MAX ||
        descriptor_count > (size_t)INT_MAX)
        return -1;
    descriptor_seen = (unsigned char*)calloc(descriptor_count, 1u);
    if (!descriptor_seen) return -2;
    for (uint32_t index = 0u; index < view->tensor_count; index++) {
        const uint8_t* record = view->response + view->tensor_offset +
            (uint64_t)index * sizeof(VxGraphPlanTensorV1);
        VxPortableGraphTensorView* tensor = &tensors[index];
        uint32_t declaration = dynamic_wire_read_u32(record + 4u);
        const uint8_t* name;
        uint32_t name_offset;
        uint32_t name_length;
        int descriptor_index;
        size_t descriptor_bytes = 0u;
        memset(tensor, 0, sizeof(*tensor));
        tensor->kind = dynamic_wire_read_u32(record + 0u);
        tensor->birth = dynamic_wire_read_u32(record + 16u);
        tensor->last_use = dynamic_wire_read_u32(record + 20u);
        tensor->alias_root = index;
        tensor->widened_birth = 0u;
        if (tensor->kind == VX_GRAPH_PLAN_TENSOR_INPUT ||
            tensor->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            const uint8_t* source;
            uint32_t source_kind;
            if (declaration >= view->source_count) goto done;
            source = view->request + view->source_offset +
                (uint64_t)declaration * sizeof(VxGraphPlanSourceV1);
            name_offset = dynamic_wire_read_u32(source + 0u);
            name_length = dynamic_wire_read_u32(source + 4u);
            source_kind = dynamic_wire_read_u32(source + 8u);
            if (dynamic_wire_read_u32(source + 12u) != 0u ||
                (tensor->kind == VX_GRAPH_PLAN_TENSOR_INPUT &&
                 source_kind != VX_GRAPH_PLAN_SOURCE_INPUT) ||
                (tensor->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT &&
                 source_kind != VX_GRAPH_PLAN_SOURCE_WEIGHT))
                goto done;
        } else if (tensor->kind == VX_GRAPH_PLAN_TENSOR_VALUE) {
            const uint8_t* edge;
            if (declaration >= view->request_edge_count) goto done;
            edge = view->request + view->request_edge_offset +
                (uint64_t)declaration * sizeof(VxGraphPlanEdgeV1);
            name_offset = dynamic_wire_read_u32(edge + 8u);
            name_length = dynamic_wire_read_u32(edge + 12u);
            if (dynamic_wire_read_u32(edge + 16u) != 0u ||
                dynamic_wire_read_u32(edge + 20u) != 0u)
                goto done;
        } else {
            goto done;
        }
        if (dynamic_wire_string_slice(view, name_offset, name_length,
                                      &name) != 0)
            goto done;
        tensor->name = name;
        tensor->name_length = name_length;
        tensor->tensor_index = dynamic_tensor_index_slice(name, name_length);
        if (tensor->tensor_index < 0 || tensor->tensor_index >= g_nt)
            goto done;
        descriptor_index = dynamic_descriptor_index_slice(
            descriptors, descriptor_count, name, name_length);
        tensor->descriptor_index = descriptor_index;
        if (tensor->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            T* engine_tensor;
            if (descriptor_index >= 0) goto done;
            engine_tensor = &g_t[tensor->tensor_index];
            if (engine_tensor->is_graph_input ||
                dynamic_weight_tensor_bytes(engine_tensor,
                                            &tensor->size_bytes) != 0)
                goto done;
            tensor->dtype = engine_tensor->dtype;
            continue;
        }
        if (descriptor_index < 0 ||
            descriptor_seen[descriptor_index] ||
            dynamic_descriptor_bytes(&descriptors[descriptor_index],
                                     &descriptor_bytes) != 0)
            goto done;
        if ((tensor->kind == VX_GRAPH_PLAN_TENSOR_INPUT) !=
                (g_t[tensor->tensor_index].is_graph_input != 0) ||
            g_t[tensor->tensor_index].dtype !=
                descriptors[descriptor_index].dtype)
            goto done;
        descriptor_seen[descriptor_index] = 1u;
        descriptor_graph_indices[descriptor_index] = index;
        tensor->dtype = descriptors[descriptor_index].dtype;
        tensor->size_bytes = descriptor_bytes;
        if (tensor->kind == VX_GRAPH_PLAN_TENSOR_INPUT)
            tensor->flags |= VX_ACTIVATION_PLAN_TENSOR_CROSS_CALL_LIVE;
    }
    for (size_t index = 0; index < descriptor_count; index++)
        if (!descriptor_seen[index]) goto done;
    result = 0;
done:
    free(descriptor_seen);
    return result;
}

static int dynamic_effective_producer_mark(
        const VxPortableGraphTensorView* tensors,
        uint32_t tensor_count,
        const char* name,
        int producer,
        int* producers) {
    int graph_index;
    if (!name || !name[0]) return 0;
    graph_index = dynamic_graph_index_name(tensors, tensor_count, name);
    if (graph_index < 0 ||
        tensors[graph_index].kind != VX_GRAPH_PLAN_TENSOR_VALUE)
        return -1;
    if (producers[graph_index] < 0 || producer < producers[graph_index])
        producers[graph_index] = producer;
    return 0;
}

static int dynamic_effective_producer_mark_node(
        const VxPortableGraphTensorView* tensors,
        uint32_t tensor_count,
        const Node* node,
        int producer,
        int* producers) {
    if (!node) return -1;
    for (int output = 0; output < node->nout; output++)
        if (dynamic_effective_producer_mark(
                tensors, tensor_count, node->outs[output].name,
                producer, producers) != 0)
            return -1;
    if (node->out[0] && dynamic_effective_producer_mark(
            tensors, tensor_count, node->out, producer, producers) != 0)
        return -1;
    return 0;
}

static int dynamic_portable_alias_target(
        VxPortableGraphTensorView* tensors,
        uint32_t tensor_count,
        int source_index,
        const char* target_name) {
    int target_index = dynamic_graph_index_name(
        tensors, tensor_count, target_name);
    VxPortableGraphTensorView* source;
    VxPortableGraphTensorView* target;
    if (source_index < 0 || target_index < 0 ||
        (uint32_t)source_index >= tensor_count ||
        (uint32_t)target_index >= tensor_count)
        return -1;
    source = &tensors[source_index];
    target = &tensors[target_index];
    if (target->kind != VX_GRAPH_PLAN_TENSOR_VALUE ||
        source->dtype != target->dtype ||
        source->size_bytes != target->size_bytes)
        return -1;
    if (source->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
        target->flags |= VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED;
        target->widened_birth = VX_ACTIVATION_PLAN_INDEX_NONE;
    } else {
        target->alias_root = (uint32_t)source_index;
    }
    return 0;
}

static int dynamic_portable_apply_lifetimes(
        VxPortableGraphTensorView* tensors,
        uint32_t tensor_count) {
    int* producers = NULL;
    int result = -1;
    if (!tensors || !tensor_count || tensor_count > (uint32_t)INT_MAX)
        return -1;
    producers = (int*)malloc((size_t)tensor_count * sizeof(*producers));
    if (!producers) return -2;
    for (uint32_t index = 0u; index < tensor_count; index++)
        producers[index] = -1;

    /* Mirror only the optimizer rewrites that can move a declared output's
     * actual write earlier than its canonical producer. Consumer last-use is
     * already conservative because graph-plan includes every skipped edge. */
    for (int node = 0; node < g_nn; node++) {
        int peer = -1;
        if (g_n[node].skip) continue;
        if (dynamic_effective_producer_mark_node(
                tensors, tensor_count, &g_n[node], node, producers) != 0)
            goto done;
        if (g_node_fusion[node].id == GRAPH_FUSION_CONV_ADD ||
            g_node_fusion[node].id == GRAPH_FUSION_DEPTHWISE_POINTWISE ||
            g_node_fusion[node].id == GRAPH_FUSION_CHAINED_ELEMENTWISE)
            peer = g_node_fusion[node].peer_idx;
        if (g_concat_sigmoid_fuse[node] >= 0)
            peer = g_concat_sigmoid_fuse[node];
        if (peer >= 0) {
            if (peer >= g_nn || !g_n[peer].skip ||
                dynamic_effective_producer_mark_node(
                    tensors, tensor_count, &g_n[peer], node,
                    producers) != 0)
                goto done;
        }
    }
    for (int node = 0; node < g_nn; node++) {
        int source_index;
        if (!dynamic_passthrough_node(&g_n[node]) || g_n[node].nin <= 0)
            continue;
        source_index = dynamic_graph_index_name(
            tensors, tensor_count, g_n[node].ins[0].name);
        if (source_index < 0) goto done;
        for (int output = 0; output < g_n[node].nout; output++) {
            int target_index = dynamic_graph_index_name(
                tensors, tensor_count, g_n[node].outs[output].name);
            if (target_index < 0 ||
                dynamic_portable_alias_target(
                    tensors, tensor_count, source_index,
                    g_n[node].outs[output].name) != 0)
                goto done;
            if (producers[target_index] < 0)
                producers[target_index] = node;
        }
    }
    for (uint32_t index = 0u; index < tensor_count; index++) {
        VxPortableGraphTensorView* tensor = &tensors[index];
        if (tensor->kind != VX_GRAPH_PLAN_TENSOR_VALUE) continue;
        if (producers[index] < 0 ||
            (uint32_t)producers[index] > tensor->birth)
            goto done;
        if ((uint32_t)producers[index] < tensor->birth &&
            !(tensor->flags & VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED)) {
            tensor->flags |= VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED;
            tensor->widened_birth = (uint32_t)producers[index];
        }
    }
    result = 0;
done:
    free(producers);
    return result;
}

typedef struct VxPortableExpectedRegion {
    uint64_t size_bytes;
    uint64_t offset_bytes;
    uint32_t birth;
    uint32_t last_use;
    int32_t dtype;
    uint32_t active;
    uint32_t cross_call;
    uint32_t seen;
} VxPortableExpectedRegion;

static int dynamic_portable_lifetime_compare(uint32_t left,
                                             uint32_t right) {
    if (left == right) return 0;
    if (left == VX_ACTIVATION_PLAN_INDEX_NONE) return -1;
    if (right == VX_ACTIVATION_PLAN_INDEX_NONE) return 1;
    return left < right ? -1 : 1;
}

static uint32_t dynamic_portable_lifetime_min(uint32_t left,
                                              uint32_t right) {
    return dynamic_portable_lifetime_compare(left, right) <= 0 ? left : right;
}

static uint32_t dynamic_portable_lifetime_max(uint32_t left,
                                              uint32_t right) {
    return dynamic_portable_lifetime_compare(left, right) >= 0 ? left : right;
}

static int dynamic_portable_lifetimes_overlap(uint32_t left_birth,
                                              uint32_t left_last,
                                              uint32_t right_birth,
                                              uint32_t right_last) {
    return dynamic_portable_lifetime_compare(left_last, right_birth) >= 0 &&
        dynamic_portable_lifetime_compare(right_last, left_birth) >= 0;
}

static int dynamic_portable_resolve_aliases(
        VxPortableGraphTensorView* tensors,
        uint32_t tensor_count) {
    if (!tensors || !tensor_count) return -1;
    for (uint32_t index = 0u; index < tensor_count; index++) {
        uint32_t current = index;
        uint32_t hops;
        for (hops = 0u; hops <= tensor_count; hops++) {
            uint32_t next = tensors[current].alias_root;
            if (next >= tensor_count) return -1;
            if (next == current) break;
            if (tensors[current].kind != VX_GRAPH_PLAN_TENSOR_VALUE ||
                tensors[current].dtype != tensors[next].dtype ||
                dynamic_portable_lifetime_compare(
                    tensors[next].flags &
                            VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED
                        ? tensors[next].widened_birth : tensors[next].birth,
                    tensors[current].flags &
                            VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED
                        ? tensors[current].widened_birth :
                          tensors[current].birth) >= 0)
                return -1;
            current = next;
        }
        if (hops > tensor_count) return -1;
        if (tensors[index].kind != VX_GRAPH_PLAN_TENSOR_WEIGHT &&
            tensors[current].kind == VX_GRAPH_PLAN_TENSOR_WEIGHT)
            return -1;
        tensors[index].alias_root = current;
    }
    return 0;
}

static int dynamic_portable_expected_regions(
        const VxPortableGraphTensorView* tensors,
        uint32_t tensor_count,
        uint32_t node_count,
        VxPortableExpectedRegion* roots,
        uint32_t* active_count,
        uint64_t* unique_bytes) {
    if (!tensors || !roots || !active_count || !unique_bytes) return -1;
    *active_count = 0u;
    *unique_bytes = 0u;
    for (uint32_t index = 0u; index < tensor_count; index++) {
        const VxPortableGraphTensorView* tensor = &tensors[index];
        VxPortableExpectedRegion* root;
        uint32_t birth;
        if (tensor->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) continue;
        if (tensor->alias_root >= tensor_count ||
            tensors[tensor->alias_root].kind == VX_GRAPH_PLAN_TENSOR_WEIGHT)
            return -1;
        root = &roots[tensor->alias_root];
        birth = tensor->flags & VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED
            ? tensor->widened_birth : tensor->birth;
        if (!root->active) {
            root->active = 1u;
            root->dtype = tensor->dtype;
            root->size_bytes = tensor->size_bytes;
            root->birth = birth;
            root->last_use = tensor->last_use;
        } else {
            if (root->dtype != tensor->dtype) return -1;
            if (tensor->size_bytes > root->size_bytes)
                root->size_bytes = tensor->size_bytes;
            root->birth = dynamic_portable_lifetime_min(root->birth, birth);
            root->last_use = dynamic_portable_lifetime_max(
                root->last_use, tensor->last_use);
        }
        if (tensor->flags & VX_ACTIVATION_PLAN_TENSOR_CROSS_CALL_LIVE)
            root->cross_call = 1u;
    }
    for (uint32_t index = 0u; index < tensor_count; index++) {
        VxPortableExpectedRegion* root = &roots[index];
        if (!root->active) continue;
        if (root->cross_call) {
            root->birth = VX_ACTIVATION_PLAN_INDEX_NONE;
            root->last_use = node_count;
        }
        if (dynamic_portable_lifetime_compare(root->birth, root->last_use) > 0 ||
            root->size_bytes > UINT64_MAX - *unique_bytes)
            return -1;
        *unique_bytes += root->size_bytes;
        (*active_count)++;
    }
    return 0;
}

static int dynamic_portable_response_decode(
        const uint8_t* response,
        uint32_t response_capacity,
        const VxPortableGraphTensorView* tensors,
        uint32_t tensor_count,
        uint32_t node_count,
        size_t* graph_offsets,
        size_t* arena_bytes) {
    VxPortableExpectedRegion* roots = NULL;
    uint64_t arena_capacities[4] = {0u, 0u, 0u, 0u};
    uint64_t arena_bases[4] = {0u, 0u, 0u, 0u};
    uint64_t arena_high_water[4] = {0u, 0u, 0u, 0u};
    uint32_t region_offset;
    uint32_t region_count;
    uint32_t response_arena_offset;
    uint32_t response_arena_count;
    uint32_t expected_active_count;
    uint32_t written_bytes;
    uint64_t expected_unique;
    uint64_t expected_capacity = 0u;
    uint64_t host_cursor = 0u;
    uint64_t cursor;
    uint32_t prior_owner = 0u;
    int have_prior_owner = 0;
    int result = -1;
    if (!response || response_capacity < sizeof(VxActivationPlanResponseV1) ||
        !tensors || !tensor_count || tensor_count > UINT32_MAX / 88u ||
        !graph_offsets || !arena_bytes)
        return -1;
    roots = (VxPortableExpectedRegion*)calloc(
        tensor_count, sizeof(*roots));
    if (!roots) return -2;
    if (dynamic_portable_expected_regions(
            tensors, tensor_count, node_count, roots,
            &expected_active_count, &expected_unique) != 0)
        goto done;
    written_bytes = dynamic_wire_read_u32(response + 32u);
    region_offset = dynamic_wire_read_u32(response + 48u);
    region_count = dynamic_wire_read_u32(response + 52u);
    response_arena_offset = dynamic_wire_read_u32(response + 56u);
    response_arena_count = dynamic_wire_read_u32(response + 60u);
    cursor = sizeof(VxActivationPlanResponseV1);
    if (dynamic_wire_read_u32(response + 0u) !=
            VOLVOXAI_ACTIVATION_PLAN_RESPONSE_MAGIC ||
        dynamic_wire_read_u32(response + 4u) !=
            VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION ||
        (int32_t)dynamic_wire_read_u32(response + 8u) !=
            VX_ACTIVATION_PLAN_STATUS_OK ||
        (int32_t)dynamic_wire_read_u32(response + 12u) !=
            VX_ACTIVATION_PLAN_ERROR_NONE ||
        dynamic_wire_read_u32(response + 16u) !=
            VX_ACTIVATION_PLAN_ERROR_SECTION_NONE ||
        dynamic_wire_read_u32(response + 20u) !=
            VX_ACTIVATION_PLAN_INDEX_NONE ||
        dynamic_wire_read_u32(response + 24u) !=
            VX_ACTIVATION_PLAN_INDEX_NONE ||
        dynamic_wire_read_u32(response + 28u) != 0u ||
        written_bytes > response_capacity ||
        written_bytes < sizeof(VxActivationPlanResponseV1) ||
        dynamic_wire_read_u32(response + 36u) != written_bytes ||
        dynamic_wire_read_u32(response + 40u) != tensor_count * UINT32_C(88) ||
        dynamic_wire_read_u32(response + 44u) != tensor_count ||
        region_count != expected_active_count ||
        response_arena_count > 4u ||
        dynamic_wire_read_u64(response + 64u) != expected_unique ||
        dynamic_wire_read_u32(response + 80u) != 0u ||
        dynamic_wire_read_u32(response + 84u) != 0u ||
        dynamic_wire_read_u32(response + 88u) != 0u ||
        dynamic_wire_read_u32(response + 92u) != 0u ||
        region_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)region_count *
                         sizeof(VxActivationPlanRegionV1)) != 0 ||
        response_arena_offset != cursor ||
        dynamic_u32_layout_add(
            &cursor, (uint64_t)response_arena_count *
                         sizeof(VxActivationPlanArenaV1)) != 0 ||
        cursor != written_bytes)
        goto done;

    {
        int prior_rank = -1;
        for (uint32_t index = 0u; index < response_arena_count; index++) {
            const uint8_t* arena = response + response_arena_offset +
                (uint64_t)index * sizeof(VxActivationPlanArenaV1);
            int32_t dtype = (int32_t)dynamic_wire_read_u32(arena + 0u);
            int rank = dynamic_portable_dtype_rank(dtype);
            uint64_t capacity = dynamic_wire_read_u64(arena + 8u);
            size_t aligned;
            if (rank <= prior_rank || rank < 0 ||
                dynamic_wire_read_u32(arena + 4u) != 0u || !capacity ||
                capacity > UINT64_MAX - expected_capacity ||
                host_cursor > SIZE_MAX ||
                dynamic_align_size((size_t)host_cursor, 64u, &aligned) != 0)
                goto done;
            host_cursor = aligned;
            arena_bases[rank] = host_cursor;
            arena_capacities[rank] = capacity;
            if (capacity > UINT64_MAX - host_cursor) goto done;
            host_cursor += capacity;
            expected_capacity += capacity;
            prior_rank = rank;
        }
    }
    if (dynamic_wire_read_u64(response + 72u) != expected_capacity ||
        host_cursor > SIZE_MAX)
        goto done;

    for (uint32_t index = 0u; index < region_count; index++) {
        const uint8_t* region = response + region_offset +
            (uint64_t)index * sizeof(VxActivationPlanRegionV1);
        uint32_t owner = dynamic_wire_read_u32(region + 0u);
        int32_t dtype = (int32_t)dynamic_wire_read_u32(region + 4u);
        uint32_t birth = dynamic_wire_read_u32(region + 8u);
        uint32_t last_use = dynamic_wire_read_u32(region + 12u);
        uint64_t offset = dynamic_wire_read_u64(region + 16u);
        uint64_t size = dynamic_wire_read_u64(region + 24u);
        uint32_t flags = dynamic_wire_read_u32(region + 32u);
        VxPortableExpectedRegion* root;
        uint64_t end;
        int rank;
        if (owner >= tensor_count ||
            (have_prior_owner && owner <= prior_owner) ||
            dynamic_wire_read_u32(region + 36u) != 0u)
            goto done;
        prior_owner = owner;
        have_prior_owner = 1;
        root = &roots[owner];
        rank = dynamic_portable_dtype_rank(dtype);
        if (!root->active || root->seen || rank < 0 ||
            root->dtype != dtype || root->birth != birth ||
            root->last_use != last_use || root->size_bytes != size ||
            flags != (root->cross_call
                ? VX_ACTIVATION_PLAN_REGION_CROSS_CALL_LIVE : 0u) ||
            offset % (dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32
                ? 4u : 1u) != 0u ||
            size > UINT64_MAX - offset)
            goto done;
        end = offset + size;
        if (!arena_capacities[rank] || end > arena_capacities[rank] ||
            arena_bases[rank] > UINT64_MAX - offset ||
            arena_bases[rank] + offset > SIZE_MAX)
            goto done;
        root->offset_bytes = offset;
        root->seen = 1u;
        if (end > arena_high_water[rank]) arena_high_water[rank] = end;
    }
    for (uint32_t index = 0u; index < tensor_count; index++) {
        const VxPortableExpectedRegion* left = &roots[index];
        int rank;
        if (!left->active || !left->seen) {
            if (left->active != left->seen) goto done;
            continue;
        }
        rank = dynamic_portable_dtype_rank(left->dtype);
        if (rank < 0) goto done;
        for (uint32_t other = index + 1u; other < tensor_count; other++) {
            const VxPortableExpectedRegion* right = &roots[other];
            uint64_t left_end;
            uint64_t right_end;
            if (!right->active || right->dtype != left->dtype ||
                !dynamic_portable_lifetimes_overlap(
                    left->birth, left->last_use,
                    right->birth, right->last_use))
                continue;
            if (left->size_bytes > UINT64_MAX - left->offset_bytes ||
                right->size_bytes > UINT64_MAX - right->offset_bytes)
                goto done;
            left_end = left->offset_bytes + left->size_bytes;
            right_end = right->offset_bytes + right->size_bytes;
            if (left->offset_bytes < right_end &&
                right->offset_bytes < left_end)
                goto done;
        }
    }
    for (int rank = 0; rank < 4; rank++)
        if (arena_capacities[rank] != arena_high_water[rank]) goto done;
    for (uint32_t index = 0u; index < tensor_count; index++) {
        const VxPortableGraphTensorView* tensor = &tensors[index];
        const VxPortableExpectedRegion* root;
        int rank;
        uint64_t physical;
        if (tensor->kind == VX_GRAPH_PLAN_TENSOR_WEIGHT) {
            graph_offsets[index] = SIZE_MAX;
            continue;
        }
        root = &roots[tensor->alias_root];
        rank = dynamic_portable_dtype_rank(root->dtype);
        if (rank < 0 || !root->active || !root->seen ||
            arena_bases[rank] > UINT64_MAX - root->offset_bytes)
            goto done;
        physical = arena_bases[rank] + root->offset_bytes;
        if (physical > SIZE_MAX) goto done;
        graph_offsets[index] = (size_t)physical;
    }
    *arena_bytes = (size_t)host_cursor;
    result = 0;
done:
    free(roots);
    return result;
}

int volvoxai_engine_configure_portable_cpu_activation_plan(
        const uint8_t* graph_plan_request,
        uint32_t graph_plan_request_bytes,
        const uint8_t* graph_plan_response,
        uint32_t graph_plan_response_bytes) {
    VxEngineState* state = vx_engine_state_current();
    VxPortableGraphWireView view;
    int result = -1;
    if (!state || !graph_plan_request || !graph_plan_request_bytes ||
        !graph_plan_response || !graph_plan_response_bytes)
        return -1;
    volvoxai_engine_metadata_lock();
    if (!g_loaded || state->backend != VX_PORTABLE_BACKEND_KIND ||
        state->dynamic_arena_active ||
        state->dynamic_shape_maximum_layout.configured)
        goto done;
    for (uint32_t index = 0; index < state->dynamic_shape_plan_capacity; index++)
        if (state->dynamic_shape_plans[index].signature) goto done;
    if (dynamic_graph_wire_view_load(
            graph_plan_request, graph_plan_request_bytes,
            graph_plan_response, graph_plan_response_bytes, &view) != 0)
        goto done;
    state->portable_graph_plan_request_v1 = graph_plan_request;
    state->portable_graph_plan_request_v1_bytes = graph_plan_request_bytes;
    state->portable_graph_plan_v1 = graph_plan_response;
    state->portable_graph_plan_v1_bytes = graph_plan_response_bytes;
    state->portable_cpu_activation_plan_enabled = 1;
    result = 0;
done:
    volvoxai_engine_metadata_unlock();
    return result;
}

static int dynamic_shape_plan_build_portable(
        const char* signature,
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        VxDynamicShapePlan* out) {
    VxEngineState* state = vx_engine_state_current();
    VxPortableGraphWireView graph;
    VxPortableGraphTensorView* tensors = NULL;
    uint32_t* descriptor_graph_indices = NULL;
    size_t* graph_offsets = NULL;
    uint8_t* request = NULL;
    uint8_t* response = NULL;
    void* scratch_allocation = NULL;
    uint8_t* scratch = NULL;
    uint64_t layout;
    uint32_t graph_offset;
    uint32_t tensor_offset;
    uint32_t request_bytes;
    uint32_t response_capacity;
    uint32_t scratch_bytes;
    size_t host_arena_bytes = 0u;
    size_t logical_bytes = 0u;
    size_t signature_bytes;
    VxActivationPlanResponseV1 query = {0};
    int allocation_failed = 0;
    int result = -1;
    int32_t core_status;
    if (!state || !state->portable_cpu_activation_plan_enabled ||
        !signature || !signature[0] || !descriptors || !descriptor_count ||
        descriptor_count > (size_t)INT_MAX || !out)
        return -1;
    memset(out, 0, sizeof(*out));
    if (dynamic_graph_wire_view_load(
            state->portable_graph_plan_request_v1,
            state->portable_graph_plan_request_v1_bytes,
            state->portable_graph_plan_v1,
            state->portable_graph_plan_v1_bytes, &graph) != 0 ||
        graph.tensor_count > UINT32_MAX / 88u)
        return -1;
    if ((size_t)graph.tensor_count > SIZE_MAX / sizeof(*tensors) ||
        (size_t)graph.tensor_count > SIZE_MAX / sizeof(*graph_offsets) ||
        descriptor_count > SIZE_MAX / sizeof(*descriptor_graph_indices))
        return -1;
    tensors = (VxPortableGraphTensorView*)calloc(
        graph.tensor_count, sizeof(*tensors));
    descriptor_graph_indices = (uint32_t*)malloc(
        descriptor_count * sizeof(*descriptor_graph_indices));
    graph_offsets = (size_t*)malloc(
        (size_t)graph.tensor_count * sizeof(*graph_offsets));
    if (!tensors || !descriptor_graph_indices || !graph_offsets) {
        allocation_failed = 1;
        goto done;
    }
    result = dynamic_graph_tensor_views_load(
        &graph, descriptors, descriptor_count, tensors,
        descriptor_graph_indices);
    if (result == -2) allocation_failed = 1;
    if (result != 0) {
        result = -1;
        goto done;
    }
    result = dynamic_portable_apply_lifetimes(tensors, graph.tensor_count);
    if (result == -2) allocation_failed = 1;
    if (result != 0) {
        result = -1;
        goto done;
    }
    if (dynamic_portable_resolve_aliases(
            tensors, graph.tensor_count) != 0) {
        result = -1;
        goto done;
    }

    layout = sizeof(VxActivationPlanRequestV1);
    graph_offset = (uint32_t)layout;
    if (dynamic_u32_layout_add(&layout, graph.response_bytes) != 0)
        goto done;
    tensor_offset = (uint32_t)layout;
    if (dynamic_u32_layout_add(
            &layout, (uint64_t)graph.tensor_count *
                         sizeof(VxActivationPlanTensorV1)) != 0)
        goto done;
    request_bytes = (uint32_t)layout;
    request = (uint8_t*)calloc(1u, request_bytes);
    if (!request) {
        allocation_failed = 1;
        goto done;
    }
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, magic),
        VOLVOXAI_ACTIVATION_PLAN_REQUEST_MAGIC);
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, abi_version),
        VOLVOXAI_ACTIVATION_PLAN_ABI_VERSION);
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, total_bytes),
        request_bytes);
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, graph_plan_offset),
        graph_offset);
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, graph_plan_bytes),
        graph.response_bytes);
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, tensor_offset),
        tensor_offset);
    dynamic_wire_write_u32(
        request + offsetof(VxActivationPlanRequestV1, tensor_count),
        graph.tensor_count);
    memcpy(request + graph_offset, graph.response, graph.response_bytes);
    for (uint32_t index = 0u; index < graph.tensor_count; index++) {
        uint8_t* record = request + tensor_offset +
            (uint64_t)index * sizeof(VxActivationPlanTensorV1);
        dynamic_wire_write_u32(record + 0u, (uint32_t)tensors[index].dtype);
        dynamic_wire_write_u32(record + 4u, tensors[index].flags);
        dynamic_wire_write_u64(record + 8u, tensors[index].size_bytes);
        dynamic_wire_write_u32(record + 16u, tensors[index].alias_root);
        dynamic_wire_write_u32(record + 20u,
            tensors[index].flags & VX_ACTIVATION_PLAN_TENSOR_BIRTH_WIDENED
                ? tensors[index].widened_birth : 0u);
    }

    core_status = vx_activation_plan_compile_v1(
        request, request_bytes, (uint8_t*)&query, (uint32_t)sizeof(query),
        NULL, 0u);
    response_capacity = dynamic_wire_read_u32(
        (const uint8_t*)&query +
        offsetof(VxActivationPlanResponseV1, required_response_bytes));
    scratch_bytes = dynamic_wire_read_u32(
        (const uint8_t*)&query +
        offsetof(VxActivationPlanResponseV1, required_scratch_bytes));
    if ((core_status != VX_ACTIVATION_PLAN_STATUS_RESPONSE_TOO_SMALL &&
         core_status != VX_ACTIVATION_PLAN_STATUS_SCRATCH_TOO_SMALL) ||
        response_capacity < sizeof(VxActivationPlanResponseV1) ||
        scratch_bytes != graph.tensor_count * 88u)
        goto done;
    response = (uint8_t*)malloc(response_capacity);
    if (!response) {
        allocation_failed = 1;
        goto done;
    }
    if (scratch_bytes) {
        uintptr_t address;
        if ((size_t)scratch_bytes > SIZE_MAX - 15u) goto done;
        scratch_allocation = malloc((size_t)scratch_bytes + 15u);
        if (!scratch_allocation) {
            allocation_failed = 1;
            goto done;
        }
        address = (uintptr_t)scratch_allocation;
        scratch = (uint8_t*)((address + 15u) & ~(uintptr_t)15u);
    }
    core_status = vx_activation_plan_compile_v1(
        request, request_bytes, response, response_capacity,
        scratch, scratch_bytes);
    if (core_status != VX_ACTIVATION_PLAN_STATUS_OK) goto done;
    result = dynamic_portable_response_decode(
        response, response_capacity, tensors, graph.tensor_count,
        graph.node_count, graph_offsets, &host_arena_bytes);
    if (result == -2) allocation_failed = 1;
    if (result != 0 || !host_arena_bytes) {
        result = -1;
        goto done;
    }

    if (descriptor_count > SIZE_MAX / sizeof(*out->tensor_indices) ||
        descriptor_count > SIZE_MAX / sizeof(*out->offsets) ||
        descriptor_count > SIZE_MAX / sizeof(*out->tensor_bytes) ||
        descriptor_count > SIZE_MAX / (9u * sizeof(*out->tensor_shapes)))
        goto done;
    signature_bytes = strlen(signature) + 1u;
    out->signature = (char*)malloc(signature_bytes);
    out->tensor_indices = (int*)malloc(
        descriptor_count * sizeof(*out->tensor_indices));
    out->offsets = (size_t*)malloc(
        descriptor_count * sizeof(*out->offsets));
    out->tensor_bytes = (size_t*)malloc(
        descriptor_count * sizeof(*out->tensor_bytes));
    out->tensor_shapes = (int*)calloc(
        descriptor_count * 9u, sizeof(*out->tensor_shapes));
    if (!out->signature || !out->tensor_indices || !out->offsets ||
        !out->tensor_bytes || !out->tensor_shapes) {
        allocation_failed = 1;
        goto done;
    }
    memcpy(out->signature, signature, signature_bytes);
    for (size_t index = 0; index < descriptor_count; index++) {
        uint32_t graph_index = descriptor_graph_indices[index];
        size_t bytes;
        if (graph_index >= graph.tensor_count ||
            tensors[graph_index].descriptor_index != (int)index ||
            graph_offsets[graph_index] == SIZE_MAX ||
            dynamic_descriptor_bytes(&descriptors[index], &bytes) != 0 ||
            bytes > SIZE_MAX - logical_bytes)
            goto done;
        out->tensor_indices[index] = tensors[graph_index].tensor_index;
        out->offsets[index] = graph_offsets[graph_index];
        out->tensor_bytes[index] = bytes;
        out->tensor_shapes[index * 9u] = descriptors[index].rank;
        for (int axis = 0; axis < descriptors[index].rank; axis++)
            out->tensor_shapes[index * 9u + 1u + (size_t)axis] =
                descriptors[index].shape[axis];
        logical_bytes += bytes;
    }
    out->tensor_count = (int)descriptor_count;
    out->arena_bytes = host_arena_bytes;
    out->logical_bytes = logical_bytes;
    out->model_generation = volvoxai_engine_model_generation_locked();
    result = 0;
done:
    free(scratch_allocation);
    free(response);
    free(request);
    free(graph_offsets);
    free(descriptor_graph_indices);
    free(tensors);
    if (result != 0) dynamic_shape_plan_clear(out);
    return allocation_failed ? -2 : result;
}

static int dynamic_shape_maximum_layout_build(
        const VolvoxAIEngineMaximumTensor* descriptors,
        size_t descriptor_count,
        VxDynamicShapeMaximumLayout* out) {
    int* starts = NULL;
    int* ends = NULL;
    int* aliases = NULL;
    int* buffer_of = NULL;
    size_t* buffer_capacity = NULL;
    int* buffer_until = NULL;
    size_t* buffer_offsets = NULL;
    int* descriptor_by_tensor = NULL;
    int* declared_producer_by_tensor = NULL;
    int* producer_by_tensor = NULL;
    int buffer_count = 0;
    size_t arena_bytes = 0;
    int result = -1;
    int allocation_failed = 0;
    if (!descriptors || !descriptor_count ||
        descriptor_count > (size_t)INT_MAX || !out) return -1;
    memset(out, 0, sizeof(*out));
    if (descriptor_count > SIZE_MAX / sizeof(*out->tensor_indices) ||
        descriptor_count > SIZE_MAX / sizeof(*out->offsets) ||
        descriptor_count > SIZE_MAX / sizeof(*out->tensor_bytes))
        return -1;
    out->tensor_indices = (int*)malloc(
        descriptor_count * sizeof(*out->tensor_indices));
    out->offsets = (size_t*)malloc(
        descriptor_count * sizeof(*out->offsets));
    out->tensor_bytes = (size_t*)malloc(
        descriptor_count * sizeof(*out->tensor_bytes));
    out->physical_offsets = (size_t*)malloc(
        descriptor_count * sizeof(*out->physical_offsets));
    out->physical_capacities = (size_t*)malloc(
        descriptor_count * sizeof(*out->physical_capacities));
    starts = (int*)malloc(descriptor_count * sizeof(*starts));
    ends = (int*)malloc(descriptor_count * sizeof(*ends));
    aliases = (int*)malloc(descriptor_count * sizeof(*aliases));
    buffer_of = (int*)malloc(descriptor_count * sizeof(*buffer_of));
    buffer_capacity = (size_t*)calloc(
        descriptor_count, sizeof(*buffer_capacity));
    buffer_until = (int*)malloc(descriptor_count * sizeof(*buffer_until));
    buffer_offsets = (size_t*)malloc(
        descriptor_count * sizeof(*buffer_offsets));
    descriptor_by_tensor = (int*)malloc((size_t)g_nt *
                                         sizeof(*descriptor_by_tensor));
    declared_producer_by_tensor = (int*)malloc(
        (size_t)g_nt * sizeof(*declared_producer_by_tensor));
    producer_by_tensor = (int*)malloc((size_t)g_nt *
                                       sizeof(*producer_by_tensor));
    if (!out->tensor_indices || !out->offsets || !out->tensor_bytes ||
        !out->physical_offsets || !out->physical_capacities ||
        !starts || !ends || !aliases || !buffer_of ||
        !buffer_capacity || !buffer_until || !buffer_offsets ||
        (g_nt && (!descriptor_by_tensor ||
                  !declared_producer_by_tensor || !producer_by_tensor))) {
        allocation_failed = 1;
        goto done;
    }

    for (int tensor = 0; tensor < g_nt; tensor++) {
        descriptor_by_tensor[tensor] = -1;
        declared_producer_by_tensor[tensor] = -1;
        producer_by_tensor[tensor] = -1;
    }
    /* First validate the closed logical declarations independently from the
     * optimized execution schedule. Fusion may rewrite Node.out or make an
     * earlier node produce a skipped peer's output, while Node.outs retains
     * the canonical graph declaration used by shape resolution. */
    for (int node = 0; node < g_nn; node++) {
        for (int output = 0; output < g_n[node].nout; output++) {
            T* tensor = t_find(g_n[node].outs[output].name);
            int tensor_index;
            if (!tensor) goto done;
            tensor_index = (int)(tensor - g_t);
            if (tensor_index < 0 || tensor_index >= g_nt ||
                declared_producer_by_tensor[tensor_index] >= 0) goto done;
            declared_producer_by_tensor[tensor_index] = node;
        }
    }
    /* Derive the earliest node that actually materializes each tensor after
     * optimization. Using only the canonical producer is unsafe: Conv+Add,
     * chained elementwise, ReLU6, depthwise+pointwise, and concat+sigmoid can
     * write a skipped peer's final output before that peer's declared node.
     * A lifetime beginning at the later declaration could then alias storage
     * that is still live when the fused producer writes it. */
    for (int node = 0; node < g_nn; node++) {
        Node* current = &g_n[node];
        int peer = -1;
        if (current->skip) continue;
        for (int output = 0; output < current->nout; output++) {
            T* tensor = t_find(current->outs[output].name);
            int tensor_index = tensor ? (int)(tensor - g_t) : -1;
            if (tensor_index < 0 || tensor_index >= g_nt) goto done;
            if (producer_by_tensor[tensor_index] < 0 ||
                node < producer_by_tensor[tensor_index])
                producer_by_tensor[tensor_index] = node;
        }
        if (current->out[0]) {
            T* tensor = t_find(current->out);
            int tensor_index = tensor ? (int)(tensor - g_t) : -1;
            if (tensor_index < 0 || tensor_index >= g_nt) goto done;
            if (producer_by_tensor[tensor_index] < 0 ||
                node < producer_by_tensor[tensor_index])
                producer_by_tensor[tensor_index] = node;
        }
        if (g_node_fusion[node].id == GRAPH_FUSION_CONV_ADD ||
            g_node_fusion[node].id == GRAPH_FUSION_DEPTHWISE_POINTWISE ||
            g_node_fusion[node].id == GRAPH_FUSION_CHAINED_ELEMENTWISE)
            peer = g_node_fusion[node].peer_idx;
        if (g_concat_sigmoid_fuse[node] >= 0)
            peer = g_concat_sigmoid_fuse[node];
        if (peer >= 0) {
            Node* fused;
            if (peer >= g_nn || !g_n[peer].skip) goto done;
            fused = &g_n[peer];
            for (int output = 0; output < fused->nout; output++) {
                T* tensor = t_find(fused->outs[output].name);
                int tensor_index = tensor ? (int)(tensor - g_t) : -1;
                if (tensor_index < 0 || tensor_index >= g_nt) goto done;
                if (producer_by_tensor[tensor_index] < 0 ||
                    node < producer_by_tensor[tensor_index])
                    producer_by_tensor[tensor_index] = node;
            }
            if (fused->out[0]) {
                T* tensor = t_find(fused->out);
                int tensor_index = tensor ? (int)(tensor - g_t) : -1;
                if (tensor_index < 0 || tensor_index >= g_nt) goto done;
                if (producer_by_tensor[tensor_index] < 0 ||
                    node < producer_by_tensor[tensor_index])
                    producer_by_tensor[tensor_index] = node;
            }
        }
    }
    /* Storage-view nodes do not dispatch, but their declared outputs alias a
     * resolved source below and therefore have a valid logical producer. Any
     * other skipped output without an effective fused producer fails closed. */
    for (int node = 0; node < g_nn; node++) {
        if (!dynamic_passthrough_node(&g_n[node])) continue;
        for (int output = 0; output < g_n[node].nout; output++) {
            T* tensor = t_find(g_n[node].outs[output].name);
            int tensor_index = tensor ? (int)(tensor - g_t) : -1;
            if (tensor_index < 0 || tensor_index >= g_nt) goto done;
            if (producer_by_tensor[tensor_index] < 0)
                producer_by_tensor[tensor_index] = node;
        }
    }

    for (size_t index = 0; index < descriptor_count; index++) {
        const VolvoxAIEngineMaximumTensor* descriptor = &descriptors[index];
        T* tensor;
        int tensor_index;
        int producer;
        size_t bytes = descriptor->maximum_byte_size;
        size_t element_size = dynamic_dtype_size(descriptor->dtype);
        size_t elements = 1u;
        if (!descriptor->name || !descriptor->name[0] || !bytes ||
            !element_size || descriptor->rank < 0 || descriptor->rank > 8)
            goto done;
        for (int axis = 0; axis < descriptor->rank; axis++) {
            int extent = descriptor->maximum_shape[axis];
            if (extent <= 0 || (size_t)extent > SIZE_MAX / elements)
                goto done;
            elements *= (size_t)extent;
        }
        if (elements > SIZE_MAX / element_size ||
            elements * element_size != bytes) goto done;
        tensor = t_find(descriptor->name);
        if (!tensor) goto done;
        tensor_index = (int)(tensor - g_t);
        if (tensor_index < 0 || tensor_index >= g_nt ||
            descriptor_by_tensor[tensor_index] >= 0) goto done;
        producer = producer_by_tensor[tensor_index];
        if ((tensor->dtype != descriptor->dtype) ||
            tensor->ndim != descriptor->rank ||
            tensor->elem_size != dynamic_dtype_size(descriptor->dtype) ||
            (!tensor->is_graph_input && producer < 0)) goto done;
        descriptor_by_tensor[tensor_index] = (int)index;
        out->tensor_indices[index] = tensor_index;
        out->tensor_bytes[index] = bytes;
        starts[index] = tensor->is_graph_input ? 0 : producer + 1;
        ends[index] = starts[index];
        aliases[index] = -1;
        buffer_of[index] = -1;
    }

    for (int node = 0; node < g_nn; node++) {
        size_t stats_bytes;
        size_t* maximum;
        if (g_n[node].operator_kind == VX_OP_Q_GROUP_NORM)
            maximum = &out->qgroupnorm_stats_bytes;
        else if (g_n[node].operator_kind == VX_OP_Q_LAYER_NORM)
            maximum = &out->qlayernorm_stats_bytes;
        else
            continue;
        if (dynamic_quantized_norm_stats_bound(
                &g_n[node], descriptors, descriptor_count,
                &stats_bytes) != 0) goto done;
        if (stats_bytes > *maximum) *maximum = stats_bytes;
    }

    /* The descriptor batch is closed: every graph input and every node output
     * appears exactly once, and no model weight may be rebound as activation
     * storage. */
    for (int tensor = 0; tensor < g_nt; tensor++) {
        if (g_t[tensor].is_graph_input &&
            descriptor_by_tensor[tensor] < 0) goto done;
    }
    for (int node = 0; node < g_nn; node++) {
        for (int output = 0; output < g_n[node].nout; output++) {
            T* tensor = t_find(g_n[node].outs[output].name);
            if (!tensor || descriptor_by_tensor[tensor - g_t] < 0) goto done;
        }
    }

    for (int node = 0; node < g_nn; node++) {
        for (int input = 0; input < g_n[node].nin; input++) {
            T* tensor = t_find(g_n[node].ins[input].name);
            int index = tensor ? descriptor_by_tensor[tensor - g_t] : -1;
            if (index >= 0 && node + 1 > ends[index]) ends[index] = node + 1;
        }
    }
    for (size_t index = 0; index < descriptor_count; index++) {
        T* tensor = &g_t[out->tensor_indices[index]];
        if (vx_engine_state_current()->shape_policy.retain_activations || tensor->is_graph_input ||
            tensor_is_declared_graph_output(tensor->name))
            ends[index] = g_nn + 2;
    }

    /* Optimized copy-only nodes retain their alias contract.  Folding the
     * alias into the plan also extends the source lifetime through every
     * consumer of the logical output. */
    for (int node = 0; node < g_nn; node++) {
        int source;
        T* source_tensor;
        if (!dynamic_passthrough_node(&g_n[node]) || g_n[node].nin <= 0)
            continue;
        source_tensor = t_find(g_n[node].ins[0].name);
        source = source_tensor ? descriptor_by_tensor[source_tensor - g_t] : -1;
        if (source < 0) {
            size_t source_bytes;
            if (!source_tensor || source_tensor->is_graph_input ||
                producer_by_tensor[source_tensor - g_t] >= 0 ||
                source_tensor->numel < 0 || !source_tensor->elem_size ||
                (size_t)source_tensor->numel >
                    SIZE_MAX / source_tensor->elem_size) goto done;
            source_bytes = (size_t)source_tensor->numel *
                source_tensor->elem_size;
            for (int output = 0; output < g_n[node].nout; output++) {
                T* target_tensor = t_find(g_n[node].outs[output].name);
                int target = target_tensor ?
                    descriptor_by_tensor[target_tensor - g_t] : -1;
                if (target < 0 || out->tensor_bytes[target] != source_bytes ||
                    descriptors[target].dtype != source_tensor->dtype)
                    goto done;
                /* This skipped node has no dispatch at which to copy its
                 * immutable source. Reserve its target from the beginning of
                 * the execution so a transactional pre-dispatch copy cannot
                 * be overwritten by an earlier arena lifetime. */
                starts[target] = 0;
            }
            continue;
        }
        for (int output = 0; output < g_n[node].nout; output++) {
            T* target_tensor = t_find(g_n[node].outs[output].name);
            int target = target_tensor ?
                descriptor_by_tensor[target_tensor - g_t] : -1;
            if (target < 0 || out->tensor_bytes[target] !=
                                  out->tensor_bytes[source] ||
                descriptors[target].dtype != descriptors[source].dtype)
                goto done;
            aliases[target] = source;
            if (ends[target] > ends[source]) ends[source] = ends[target];
        }
    }

    /* Assign logical lifetimes to reusable byte buffers in closed descriptor
     * order. Fusion may move a later declaration's effective producer earlier;
     * reuse therefore relies only on the explicit prior-end < new-start test,
     * not on producer monotonicity. */
    for (size_t index = 0; index < descriptor_count; index++) {
        int best = -1;
        size_t need = out->tensor_bytes[index];
        /* Native GPU byte kernels bind packed u32 storage. Keep the logical
         * tensor byte count exact, but reserve the physical lifetime through
         * the final packed word so odd I8/U8 extents cannot overrun a proved
         * maximum-domain span. */
        if (descriptors[index].dtype == T_I8 ||
            descriptors[index].dtype == T_U8) {
            if (need > SIZE_MAX - 3u) goto done;
            need = (need + 3u) & ~(size_t)3u;
        }
        if (aliases[index] >= 0) {
            int root = aliases[index];
            while (root >= 0 && aliases[root] >= 0) root = aliases[root];
            if (root < 0 || root >= (int)index || buffer_of[root] < 0)
                goto done;
            buffer_of[index] = buffer_of[root];
            if (ends[index] > buffer_until[buffer_of[index]])
                buffer_until[buffer_of[index]] = ends[index];
            continue;
        }
        for (int buffer = 0; buffer < buffer_count; buffer++) {
            if (buffer_until[buffer] >= starts[index] ||
                buffer_capacity[buffer] < need) continue;
            if (best < 0 || buffer_capacity[buffer] < buffer_capacity[best])
                best = buffer;
        }
        if (best < 0) {
            for (int buffer = 0; buffer < buffer_count; buffer++) {
                if (buffer_until[buffer] >= starts[index]) continue;
                if (best < 0 ||
                    buffer_capacity[buffer] > buffer_capacity[best])
                    best = buffer;
            }
        }
        if (best < 0) {
            if (buffer_count >= (int)descriptor_count) goto done;
            best = buffer_count++;
            buffer_capacity[best] = 0;
        }
        if (buffer_capacity[best] < need) buffer_capacity[best] = need;
        buffer_until[best] = ends[index];
        buffer_of[index] = best;
    }
    for (int buffer = 0; buffer < buffer_count; buffer++) {
        if (dynamic_align_size(arena_bytes, 64u, &arena_bytes) != 0)
            goto done;
        buffer_offsets[buffer] = arena_bytes;
        out->physical_offsets[buffer] = arena_bytes;
        out->physical_capacities[buffer] = buffer_capacity[buffer];
        if (buffer_capacity[buffer] > SIZE_MAX - arena_bytes) goto done;
        arena_bytes += buffer_capacity[buffer];
    }
    for (size_t index = 0; index < descriptor_count; index++) {
        if (buffer_of[index] < 0 || buffer_of[index] >= buffer_count)
            goto done;
        out->offsets[index] = buffer_offsets[buffer_of[index]];
        if (out->tensor_bytes[index] > arena_bytes - out->offsets[index])
            goto done;
    }
    out->tensor_count = (int)descriptor_count;
    out->physical_span_count = buffer_count;
    out->arena_bytes = arena_bytes;
    out->model_generation = volvoxai_engine_model_generation_locked();
    out->configured = 1;
    result = 0;
done:
    free(starts);
    free(ends);
    free(aliases);
    free(buffer_of);
    free(buffer_capacity);
    free(buffer_until);
    free(buffer_offsets);
    free(descriptor_by_tensor);
    free(declared_producer_by_tensor);
    free(producer_by_tensor);
    if (result != 0) dynamic_shape_maximum_layout_clear(out);
    return allocation_failed ? -2 : result;
}

/* Project one exact logical shape onto the independently proved maximum
 * layout. Do not repack the smaller byte sizes: best-fit fragmentation is not
 * monotone and a concrete packing can exceed the maximum packing even when
 * every tensor shrinks. */
static int dynamic_shape_plan_project_maximum(
        const char* signature,
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        const VxDynamicShapeMaximumLayout* maximum,
        VxDynamicShapePlan* out) {
    int* descriptor_by_tensor = NULL;
    size_t logical_bytes = 0;
    size_t signature_size;
    int allocation_failed = 0;
    int result = -1;
    if (!signature || !signature[0] || !descriptors || !descriptor_count ||
        descriptor_count > (size_t)INT_MAX || !maximum || !maximum->configured ||
        maximum->model_generation !=
            volvoxai_engine_model_generation_locked() ||
        maximum->tensor_count < 0 ||
        (size_t)maximum->tensor_count != descriptor_count || !out)
        return -1;
    memset(out, 0, sizeof(*out));
    if (descriptor_count > SIZE_MAX / sizeof(*out->tensor_indices) ||
        descriptor_count > SIZE_MAX / sizeof(*out->offsets) ||
        descriptor_count > SIZE_MAX / sizeof(*out->tensor_bytes) ||
        descriptor_count > SIZE_MAX / (9u * sizeof(*out->tensor_shapes)))
        return -1;
    out->tensor_indices = (int*)malloc(
        descriptor_count * sizeof(*out->tensor_indices));
    out->offsets = (size_t*)malloc(
        descriptor_count * sizeof(*out->offsets));
    out->tensor_bytes = (size_t*)malloc(
        descriptor_count * sizeof(*out->tensor_bytes));
    out->tensor_shapes = (int*)calloc(
        descriptor_count * 9u, sizeof(*out->tensor_shapes));
    descriptor_by_tensor = (int*)malloc(
        (size_t)g_nt * sizeof(*descriptor_by_tensor));
    signature_size = strlen(signature) + 1u;
    out->signature = (char*)malloc(signature_size);
    if (!out->tensor_indices || !out->offsets || !out->tensor_bytes ||
        !out->tensor_shapes || (g_nt && !descriptor_by_tensor) ||
        !out->signature) {
        allocation_failed = 1;
        goto done;
    }
    memcpy(out->signature, signature, signature_size);
    for (int tensor = 0; tensor < g_nt; tensor++)
        descriptor_by_tensor[tensor] = -1;
    for (size_t index = 0; index < descriptor_count; index++) {
        const VolvoxAIEngineResolvedTensor* descriptor = &descriptors[index];
        T* tensor;
        int tensor_index;
        int maximum_index = -1;
        size_t bytes;
        if (dynamic_descriptor_bytes(descriptor, &bytes) != 0) goto done;
        tensor = t_find(descriptor->name);
        if (!tensor || tensor->dtype != descriptor->dtype ||
            tensor->elem_size != dynamic_dtype_size(descriptor->dtype))
            goto done;
        tensor_index = (int)(tensor - g_t);
        if (tensor_index < 0 || tensor_index >= g_nt ||
            descriptor_by_tensor[tensor_index] >= 0) goto done;
        for (int candidate = 0; candidate < maximum->tensor_count;
             candidate++) {
            if (maximum->tensor_indices[candidate] == tensor_index) {
                maximum_index = candidate;
                break;
            }
        }
        if (maximum_index < 0 ||
            bytes > maximum->tensor_bytes[maximum_index] ||
            maximum->offsets[maximum_index] > maximum->arena_bytes ||
            bytes > maximum->arena_bytes -
                maximum->offsets[maximum_index] ||
            bytes > SIZE_MAX - logical_bytes) goto done;
        descriptor_by_tensor[tensor_index] = (int)index;
        out->tensor_indices[index] = tensor_index;
        out->offsets[index] = maximum->offsets[maximum_index];
        out->tensor_bytes[index] = bytes;
        out->tensor_shapes[index * 9u] = descriptor->rank;
        for (int axis = 0; axis < descriptor->rank; axis++)
            out->tensor_shapes[index * 9u + 1u + (size_t)axis] =
                descriptor->shape[axis];
        logical_bytes += bytes;
    }
    for (int index = 0; index < maximum->tensor_count; index++) {
        int tensor_index = maximum->tensor_indices[index];
        if (tensor_index < 0 || tensor_index >= g_nt ||
            descriptor_by_tensor[tensor_index] < 0) goto done;
    }
    out->tensor_count = (int)descriptor_count;
    out->arena_bytes = maximum->arena_bytes;
    out->logical_bytes = logical_bytes;
    out->model_generation = maximum->model_generation;
    result = 0;
done:
    free(descriptor_by_tensor);
    if (result != 0) dynamic_shape_plan_clear(out);
    return allocation_failed ? -2 : result;
}

static int dynamic_shape_plan_build(
        const char* signature,
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        const VxDynamicShapeMaximumLayout* maximum,
        VxDynamicShapePlan* out) {
    VxEngineState* state = vx_engine_state_current();
    VolvoxAIEngineMaximumTensor* exact_descriptors = NULL;
    VxDynamicShapeMaximumLayout exact_layout = {0};
    int result;
    if (state && state->portable_cpu_activation_plan_enabled) {
        if (maximum && maximum->configured) return -1;
        return dynamic_shape_plan_build_portable(
            signature, descriptors, descriptor_count, out);
    }
    if (maximum && maximum->configured)
        return dynamic_shape_plan_project_maximum(
            signature, descriptors, descriptor_count, maximum, out);

    /* CPU retains its exact-shape liveness layout and geometric growth path.
     * Dynamic native-GPU contexts configure a fixed maximum-domain layout. */
    if (!descriptors || !descriptor_count || descriptor_count > (size_t)INT_MAX)
        return -1;
    exact_descriptors = (VolvoxAIEngineMaximumTensor*)calloc(
        descriptor_count, sizeof(*exact_descriptors));
    if (!exact_descriptors) return -2;
    for (size_t index = 0; index < descriptor_count; index++) {
        size_t bytes;
        if (dynamic_descriptor_bytes(&descriptors[index], &bytes) != 0) {
            free(exact_descriptors);
            return -1;
        }
        exact_descriptors[index].name = descriptors[index].name;
        exact_descriptors[index].dtype = descriptors[index].dtype;
        exact_descriptors[index].rank = descriptors[index].rank;
        memcpy(exact_descriptors[index].maximum_shape,
               descriptors[index].shape,
               sizeof(exact_descriptors[index].maximum_shape));
        exact_descriptors[index].maximum_byte_size = bytes;
    }
    result = dynamic_shape_maximum_layout_build(
        exact_descriptors, descriptor_count, &exact_layout);
    free(exact_descriptors);
    if (result == 0)
        result = dynamic_shape_plan_project_maximum(
            signature, descriptors, descriptor_count, &exact_layout, out);
    dynamic_shape_maximum_layout_clear(&exact_layout);
    return result;
}

static int dynamic_shape_plan_matches(
        const VxDynamicShapePlan* plan,
        const char* signature,
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count) {
    if (!plan || !plan->signature || strcmp(plan->signature, signature) ||
        plan->model_generation != volvoxai_engine_model_generation_locked() ||
        plan->tensor_count < 0 || (size_t)plan->tensor_count != descriptor_count)
        return 0;
    for (size_t index = 0; index < descriptor_count; index++) {
        T* tensor = t_find(descriptors[index].name);
        size_t bytes;
        if (!tensor || tensor->dtype != descriptors[index].dtype ||
            dynamic_descriptor_bytes(&descriptors[index], &bytes) != 0 ||
            plan->tensor_indices[index] != (int)(tensor - g_t) ||
            plan->tensor_bytes[index] != bytes ||
            plan->tensor_shapes[index * 9u] != descriptors[index].rank)
            return 0;
        for (int axis = 0; axis < descriptors[index].rank; axis++)
            if (plan->tensor_shapes[index * 9u + 1u + (size_t)axis] !=
                descriptors[index].shape[axis]) return 0;
    }
    return 1;
}

static int dynamic_bindings_validate(
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        const VolvoxAIEngineInputBinding* inputs,
        size_t input_count,
        int require_complete) {
    size_t graph_input_count = 0;
    if ((input_count && !inputs) || (descriptor_count && !descriptors))
        return -1;
    for (size_t index = 0; index < descriptor_count; index++) {
        T* tensor = t_find(descriptors[index].name);
        if (tensor && tensor->is_graph_input) graph_input_count++;
    }
    if (require_complete && input_count != graph_input_count) return -1;
    for (size_t index = 0; index < input_count; index++) {
        int descriptor_index;
        size_t bytes;
        T* tensor;
        if (!inputs[index].name || !inputs[index].name[0] ||
            !inputs[index].data) return -1;
        descriptor_index = dynamic_descriptor_index(
            descriptors, descriptor_count, inputs[index].name);
        if (descriptor_index < 0) return -1;
        tensor = t_find(inputs[index].name);
        if (!tensor || !tensor->is_graph_input ||
            descriptors[descriptor_index].dtype != inputs[index].dtype ||
            dynamic_descriptor_bytes(&descriptors[descriptor_index], &bytes) != 0 ||
            inputs[index].byte_size != bytes) return -1;
        for (size_t prior = 0; prior < index; prior++)
            if (!strcmp(inputs[prior].name, inputs[index].name)) return -1;
    }
    return 0;
}

int volvoxai_engine_configure_dynamic_shape_domain(
        const VolvoxAIEngineMaximumTensor* descriptors,
        size_t descriptor_count) {
    VxEngineState* state = vx_engine_state_current();
    VxDynamicShapeMaximumLayout candidate = {0};
    int took_model_lock;
    int result = -1;
    if (!state || !descriptors || !descriptor_count ||
        descriptor_count > (size_t)INT_MAX) return -1;
    took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    if (!g_loaded || state->dynamic_arena_active) goto done;
    for (uint32_t index = 0; index < state->dynamic_shape_plan_capacity; index++)
        if (state->dynamic_shape_plans[index].signature) goto done;
    result = dynamic_shape_maximum_layout_build(
        descriptors, descriptor_count, &candidate);
    if (result != 0) goto done;
    dynamic_shape_maximum_layout_clear(
        &state->dynamic_shape_maximum_layout);
    state->dynamic_shape_maximum_layout = candidate;
    memset(&candidate, 0, sizeof(candidate));
done:
    dynamic_shape_maximum_layout_clear(&candidate);
    volvoxai_engine_metadata_unlock();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

static size_t dynamic_geometric_capacity(size_t current, size_t required) {
    const VolvoxAIEngineShapePolicy* policy = &vx_engine_state_current()->shape_policy;
    if (policy->capacity_growth_factor > 1.0) {
        double proposed = current ? ceil(current * policy->capacity_growth_factor) : (double)required;
        if (!isfinite(proposed) || proposed >= (double)SIZE_MAX) return 0;
        size_t capacity = (size_t)proposed;
        if (capacity < required) capacity = required;
        return policy->max_activation_capacity_bytes && capacity > policy->max_activation_capacity_bytes ? 0 : capacity;
    }
    size_t capacity = current ? current : 4096u;
    if (capacity < current || !required) return required;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) return required;
        capacity *= 2u;
    }
    return capacity;
}

static uint64_t dynamic_shape_plan_metadata(const VxDynamicShapePlan* plan) {
    return plan->signature ? sizeof(*plan) + strlen(plan->signature) + 1u +
        (uint64_t)plan->tensor_count * (10u * sizeof(int) + 2u * sizeof(size_t)) : 0;
}

/* Publication-only LRU mutation. All fallible binding work finishes first. */
static VxDynamicShapePlan* dynamic_shape_cache_publish(VxEngineState* state,
        VxDynamicShapePlan* candidate, uint32_t* evictions) {
    uint64_t bytes = dynamic_shape_plan_metadata(candidate);
    candidate->metadata_bytes = bytes;
    uint64_t limit = state->shape_policy.plan_cache_metadata_bytes;
    if (limit && bytes > limit) {
        state->dynamic_shape_cache_oversize_skips++;
        state->dynamic_shape_cache_bypasses++;
        return candidate;
    }
    VxDynamicShapePlan* vacant = NULL;
    for (;;) {
        VxDynamicShapePlan* oldest = NULL;
        for (uint32_t i = 0; i < state->dynamic_shape_plan_capacity; ++i) {
            VxDynamicShapePlan* entry = &state->dynamic_shape_plans[i];
            if (!entry->signature) vacant = entry;
            else if (!oldest || entry->last_use < oldest->last_use) oldest = entry;
        }
        uint32_t entry_limit = state->shape_policy.plan_cache_entries ?
            state->shape_policy.plan_cache_entries : VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY;
        if (!vacant && state->dynamic_shape_plan_capacity < entry_limit &&
            (!limit || state->dynamic_shape_plan_metadata_bytes <= limit - bytes)) {
            uint32_t capacity = state->dynamic_shape_plan_capacity;
            uint32_t next = capacity > entry_limit / 2u ? entry_limit : capacity ? capacity * 2u : 1u;
            if ((uint64_t)next * sizeof(*state->dynamic_shape_plans) > SIZE_MAX) {
                state->dynamic_shape_cache_bypasses++;
                return candidate;
            }
            VxDynamicShapePlan* entries = realloc(state->dynamic_shape_plans, (size_t)next * sizeof(*entries));
            /* Cache allocation is optional after a valid binding. An absent
             * cache slot must not fail execution or evict a working entry. */
            if (!entries) {
                state->dynamic_shape_cache_bypasses++;
                return candidate;
            }
            memset(entries + capacity, 0, (size_t)(next - capacity) * sizeof(*entries));
            state->dynamic_shape_plans = entries;
            state->dynamic_shape_plan_capacity = next;
            vacant = &entries[capacity];
        }
        if (vacant && (!limit || state->dynamic_shape_plan_metadata_bytes <= limit - bytes)) break;
        if (!oldest) {
            state->dynamic_shape_cache_bypasses++;
            return candidate;
        }
        state->dynamic_shape_plan_metadata_bytes -= dynamic_shape_plan_metadata(oldest);
        dynamic_shape_plan_clear(oldest);
        ++*evictions;
    }
    *vacant = *candidate;
    memset(candidate, 0, sizeof(*candidate));
    state->dynamic_shape_plan_metadata_bytes += bytes;
    return vacant;
}

int volvoxai_engine_reserve_dynamic_shape_domain(void) {
    static const char reservation_signature[] =
        "$volvoxai.maximum-domain-reservation";
    VxEngineState* state = vx_engine_state_current();
    const VxDynamicShapeMaximumLayout* maximum;
    VolvoxAIEnginePhysicalSpan* spans = NULL;
    void* arena = NULL;
    int took_model_lock;
    int result = -1;
    if (!state) return -1;
    took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    maximum = &state->dynamic_shape_maximum_layout;
    if (!g_loaded || !maximum->configured ||
        maximum->model_generation !=
            volvoxai_engine_model_generation_locked() ||
        maximum->physical_span_count <= 0 || !maximum->arena_bytes ||
        state->dynamic_arena_active || state->dynamic_shape_reserved_arena)
        goto done;
    arena = vx_arena_allocate(maximum->arena_bytes);
    if (!arena) {
        result = -2;
        goto done;
    }
    spans = (VolvoxAIEnginePhysicalSpan*)malloc(
        (size_t)maximum->physical_span_count * sizeof(*spans));
    if (!spans) {
        result = -2;
        goto done;
    }
    for (int index = 0; index < maximum->physical_span_count; index++) {
        size_t offset = maximum->physical_offsets[index];
        size_t capacity = maximum->physical_capacities[index];
        if (!capacity || offset > maximum->arena_bytes ||
            capacity > maximum->arena_bytes - offset) goto done;
        spans[index].host = (const unsigned char*)arena + offset;
        spans[index].capacity_bytes = capacity;
    }
    if (vx_runtime_backend_has_graph()) {
        result = vx_runtime_backend_bind_shape_domain(
            reservation_signature, spans,
            (size_t)maximum->physical_span_count,
            maximum->qgroupnorm_stats_bytes,
            maximum->qlayernorm_stats_bytes);
        if (result != 0) goto done;
    }
    state->dynamic_shape_reserved_arena = arena;
    arena = NULL;
    state->dynamic_arena_capacity_bytes = maximum->arena_bytes;
    state->dynamic_arena_high_water_bytes = maximum->arena_bytes;
    state->dynamic_arena_grow_count = 1u;
    result = 0;
done:
    free(spans);
    vx_arena_release(arena);
    volvoxai_engine_metadata_unlock();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_commit_dynamic_shape(
        const char* signature,
        const VolvoxAIEngineResolvedTensor* descriptors,
        size_t descriptor_count,
        const VolvoxAIEngineInputBinding* inputs,
        size_t input_count,
        VolvoxAIEngineDynamicShapeStats* stats) {
    VxEngineState* state = vx_engine_state_current();
    VxDynamicShapePlan candidate_plan = {0};
    VxDynamicShapePlan* plan = NULL;
    void* candidate_arena = NULL;
    size_t candidate_capacity = 0;
    VolvoxAIEnginePhysicalSpan* physical_spans = NULL;
    void** next_buffers = NULL;
    int* next_indices = NULL;
    void** owned_storage = NULL;
    size_t owned_count = 0;
    int cache_hit = 0;
    int grew = 0;
    int consumed_reservation = 0;
    int took_model_lock;
    int result = -1;
    if (!state || !signature || !signature[0] || !descriptors ||
        !descriptor_count || descriptor_count > (size_t)INT_MAX ||
        !stats || stats->struct_size != sizeof(*stats)) return -1;
    took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    if (!g_loaded) goto done;
    state->activation_budget_exceeded = 0;
    if (dynamic_bindings_validate(descriptors, descriptor_count, inputs,
                                  input_count, 1) != 0) goto done;
    for (uint32_t index = 0; index < state->dynamic_shape_plan_capacity; index++) {
        if (dynamic_shape_plan_matches(&state->dynamic_shape_plans[index],
                                       signature, descriptors,
                                       descriptor_count)) {
            plan = &state->dynamic_shape_plans[index];
            cache_hit = 1;
            break;
        }
    }
    if (!plan) {
        int build_status = dynamic_shape_plan_build(
            signature, descriptors, descriptor_count,
            &state->dynamic_shape_maximum_layout, &candidate_plan);
        if (build_status != 0) {
            result = build_status;
            goto done;
        }
        plan = &candidate_plan;
    }
    if (state->dynamic_arena_active && g_arena_nbufs == 1 &&
        g_arena_bufs && g_arena_bufs[0] &&
        state->dynamic_arena_capacity_bytes >= plan->arena_bytes) {
        candidate_arena = g_arena_bufs[0];
        candidate_capacity = state->dynamic_arena_capacity_bytes;
    } else if (state->dynamic_shape_reserved_arena &&
               state->dynamic_arena_capacity_bytes == plan->arena_bytes) {
        candidate_arena = state->dynamic_shape_reserved_arena;
        candidate_capacity = state->dynamic_arena_capacity_bytes;
        consumed_reservation = 1;
    } else if (state->dynamic_shape_maximum_layout.configured) {
        /* A bounded accelerator domain is published only after reserve() has
         * acquired its complete host/device capacity. Never turn an absent or
         * inconsistent reservation into first-execute arena growth. */
        goto done;
    } else {
        candidate_capacity = dynamic_geometric_capacity(
            state->dynamic_arena_capacity_bytes, plan->arena_bytes);
        if (candidate_capacity < plan->arena_bytes) {
            state->activation_budget_exceeded = 1; result = -2; goto done;
        }
        candidate_arena = vx_arena_allocate(candidate_capacity ? candidate_capacity : 1u);
        if (!candidate_arena) {
            result = -2;
            goto done;
        }
        grew = 1;
    }
    next_buffers = (void**)malloc(sizeof(*next_buffers));
    next_indices = (int*)malloc(descriptor_count * sizeof(*next_indices));
    owned_storage = (void**)malloc(descriptor_count * sizeof(*owned_storage));
    if (!next_buffers || !next_indices || !owned_storage) {
        result = -2;
        goto done;
    }
    next_buffers[0] = candidate_arena;
    for (size_t index = 0; index < descriptor_count; index++) {
        T* tensor = &g_t[plan->tensor_indices[index]];
        next_indices[index] = plan->tensor_indices[index];
        if (tensor->owns && tensor->data) {
            int duplicate = 0;
            for (size_t prior = 0; prior < owned_count; prior++)
                if (owned_storage[prior] == tensor->data) duplicate = 1;
            if (!duplicate) owned_storage[owned_count++] = tensor->data;
        }
    }

    if (state->dynamic_shape_maximum_layout.physical_span_count < 0 ||
        (size_t)state->dynamic_shape_maximum_layout.physical_span_count >
            descriptor_count) goto done;
    if (state->dynamic_shape_maximum_layout.physical_span_count) {
        physical_spans = (VolvoxAIEnginePhysicalSpan*)malloc(
            (size_t)state->dynamic_shape_maximum_layout.physical_span_count *
            sizeof(*physical_spans));
        if (!physical_spans) {
            result = -2;
            goto done;
        }
    }
    for (int index = 0;
         index < state->dynamic_shape_maximum_layout.physical_span_count;
         index++) {
        size_t offset =
            state->dynamic_shape_maximum_layout.physical_offsets[index];
        size_t capacity =
            state->dynamic_shape_maximum_layout.physical_capacities[index];
        if (!capacity || offset > candidate_capacity ||
            capacity > candidate_capacity - offset) goto done;
        physical_spans[index].host =
            (const unsigned char*)candidate_arena + offset;
        physical_spans[index].capacity_bytes = capacity;
    }

    /* Native graph backends reserve the complete fixed physical domain and
     * stage their exact semantic shape key before the host tensor table moves.
     * Every host operation after this point is non-failing, so a rejected
     * device reservation leaves the previous graph and arena untouched. */
    if (vx_runtime_backend_has_graph()) {
        int bind_status = state->dynamic_shape_maximum_layout.configured
            ? vx_runtime_backend_bind_shape_domain(
                signature, physical_spans,
                (size_t)state->dynamic_shape_maximum_layout.physical_span_count,
                state->dynamic_shape_maximum_layout.qgroupnorm_stats_bytes,
                state->dynamic_shape_maximum_layout.qlayernorm_stats_bytes)
            : vx_runtime_backend_bind_shape(signature);
        if (bind_status != 0) {
            result = bind_status;
            goto done;
        }
    }

    /* Commit boundary. Every lookup, byte calculation, metadata allocation,
     * plan construction, and arena growth above is side-effect free. */
    for (size_t index = 0; index < descriptor_count; index++) {
        const VolvoxAIEngineResolvedTensor* descriptor = &descriptors[index];
        T* tensor = &g_t[plan->tensor_indices[index]];
        tensor->ndim = descriptor->rank;
        memset(tensor->shape, 0, sizeof(tensor->shape));
        tensor->numel = 1;
        for (int axis = 0; axis < descriptor->rank; axis++) {
            tensor->shape[axis] = descriptor->shape[axis];
            tensor->numel *= descriptor->shape[axis];
        }
        tensor->data = (float*)((unsigned char*)candidate_arena +
                                plan->offsets[index]);
        tensor->owns = 0;
    }
    /* Skipped passthroughs normally alias another resolved activation. When
     * their source is an immutable model tensor outside the activation plan,
     * materialize it into the dedicated lifetime reserved by plan_build(). */
    for (int node = 0; node < g_nn; node++) {
        int source_index;
        T* source;
        size_t source_bytes;
        if (!dynamic_passthrough_node(&g_n[node]) || g_n[node].nin <= 0)
            continue;
        source_index = dynamic_descriptor_index(
            descriptors, descriptor_count, g_n[node].ins[0].name);
        if (source_index >= 0) continue;
        source = t_find(g_n[node].ins[0].name);
        source_bytes = (size_t)source->numel * source->elem_size;
        for (int output = 0; output < g_n[node].nout; output++) {
            int target_index = dynamic_descriptor_index(
                descriptors, descriptor_count, g_n[node].outs[output].name);
            T* target = &g_t[plan->tensor_indices[target_index]];
            if (source_bytes) {
                memcpy(target->data, source->data, source_bytes);
                if (vx_runtime_backend_has_graph())
                    vx_runtime_backend_mark_host(
                        target->data, source_bytes, 0);
            }
        }
    }
    for (size_t index = 0; index < input_count; index++) {
        T* tensor = t_find(inputs[index].name);
        if (inputs[index].byte_size && inputs[index].location == VX_MEMORY_HOST)
            memcpy(tensor->data, inputs[index].data, inputs[index].byte_size);
        vx_incremental_mark_tensor_locked(tensor);
        if (vx_runtime_backend_has_graph() && inputs[index].byte_size &&
            inputs[index].location == VX_MEMORY_HOST)
            vx_runtime_backend_mark_host(
                tensor->data, inputs[index].byte_size, 0);
    }
    for (size_t index = 0; index < owned_count; index++)
        free(owned_storage[index]);
    for (int index = 0; index < g_arena_nbufs; index++)
        if (g_arena_bufs[index] != candidate_arena) vx_arena_release(g_arena_bufs[index]);
    free(g_arena_bufs);
    free(g_arena_tensor_indices);
    g_arena_bufs = next_buffers;
    state->arena_allocated_bytes = candidate_capacity;
    g_arena_nbufs = 1;
    g_arena_tensor_indices = next_indices;
    g_arena_tensor_count = (int)descriptor_count;
    if (consumed_reservation)
        state->dynamic_shape_reserved_arena = NULL;
    next_buffers = NULL;
    next_indices = NULL;
    uint32_t cache_evictions = 0;
    if (!cache_hit) plan = dynamic_shape_cache_publish(state, &candidate_plan, &cache_evictions);
    plan->last_use = ++state->dynamic_shape_plan_clock;
    state->dynamic_arena_active = 1;
    state->dynamic_arena_capacity_bytes = candidate_capacity;
    state->dynamic_arena_current_bytes = plan->arena_bytes;
    if (candidate_capacity > state->dynamic_arena_high_water_bytes)
        state->dynamic_arena_high_water_bytes = candidate_capacity;
    if (grew) state->dynamic_arena_grow_count++;
    state->dynamic_resource_generation++;
    g_arena_detached = 0;
    vx_incremental_invalidate_locked();
    if (!vx_runtime_backend_has_graph())
        vx_runtime_backend_reset_transients();
    *stats = (VolvoxAIEngineDynamicShapeStats)
        VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
    stats->plan_cache_hit = cache_hit;
    stats->plan_cache_evictions = cache_evictions;
    stats->logical_bytes = plan->logical_bytes;
    stats->required_arena_bytes = plan->arena_bytes;
    stats->arena_capacity_bytes = candidate_capacity;
    stats->arena_high_water_bytes = state->dynamic_arena_high_water_bytes;
    stats->arena_grow_count = state->dynamic_arena_grow_count;
    stats->resource_generation = state->dynamic_resource_generation;
    result = 0;
done:
    if (result != 0 && grew) vx_arena_release(candidate_arena);
    free(next_buffers);
    free(next_indices);
    free(owned_storage);
    free(physical_spans);
    dynamic_shape_plan_clear(&candidate_plan);
    volvoxai_engine_metadata_unlock();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_commit_input_bindings(
        const VolvoxAIEngineInputBinding* inputs,
        size_t input_count) {
    int took_model_lock;
    int result = -1;
    if ((input_count && !inputs) || !vx_engine_state_current()) return -1;
    took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    if (!g_loaded) goto done;
    for (size_t index = 0; index < input_count; index++) {
        T* tensor = inputs[index].name ? t_find(inputs[index].name) : NULL;
        size_t bytes;
        if (!tensor || !tensor->is_graph_input || !tensor->data ||
            tensor->dtype != inputs[index].dtype || tensor->numel < 0 ||
            !tensor->elem_size ||
            (size_t)tensor->numel > SIZE_MAX / tensor->elem_size ||
            inputs[index].byte_size !=
                (bytes = (size_t)tensor->numel * tensor->elem_size) ||
            (!inputs[index].data && bytes)) goto done;
        for (size_t prior = 0; prior < index; prior++)
            if (!strcmp(inputs[prior].name, inputs[index].name)) goto done;
    }
    for (size_t index = 0; index < input_count; index++) {
        T* tensor = t_find(inputs[index].name);
        if (inputs[index].byte_size && inputs[index].location == VX_MEMORY_HOST)
            memcpy(tensor->data, inputs[index].data, inputs[index].byte_size);
        vx_incremental_mark_tensor_locked(tensor);
        if (vx_runtime_backend_has_graph() && inputs[index].byte_size &&
            inputs[index].location == VX_MEMORY_HOST)
            vx_runtime_backend_mark_host(
                tensor->data, inputs[index].byte_size, 0);
    }
    result = 0;
done:
    volvoxai_engine_metadata_unlock();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

static void plan_memory_arena(void) {
    /* A private Trainer retains all forward values for backward. Its exact
     * dynamic arena survives the step; inference liveness reuse would destroy
     * both those values and the configured geometric capacity policy. */
    if (vx_engine_state_current()->shape_policy.retain_activations) return;
    const char* en = vx_engine_env("VOLVOX_ARENA");
    if (en && en[0] && !strcmp(en, "0")) return;   // default-on; VOLVOX_ARENA=0 disables
    if (g_use_vulkan || g_use_opengl || g_use_metal
#if VOLVOXAI_ENABLE_CUDA
        || g_use_cuda
#endif
    ) return;
    int nt = g_nt, nn = g_nn;
    if (nt <= 0 || nn <= 0) return;

    int* prod = (int*)malloc(sizeof(int) * nt);
    int* last = (int*)malloc(sizeof(int) * nt);
    int* buf_of = (int*)malloc(sizeof(int) * nt);
    char* poolable = (char*)calloc(nt, 1);
    size_t* cap = (size_t*)malloc(sizeof(size_t) * nt);
    char* busy = (char*)calloc(nt, 1);
    void** bufs = (void**)calloc((size_t)nt, sizeof(void*));
    if (!prod || !last || !buf_of || !poolable || !cap || !busy || !bufs) {
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }
    for (int i = 0; i < nt; i++) { prod[i] = -1; last[i] = -1; buf_of[i] = -1; }

    for (int j = 0; j < nn; j++) {
        if (g_n[j].skip) continue;
        int ti = t_index_by_name(g_n[j].out);
        if (ti >= 0) prod[ti] = j;
    }
    for (int j = 0; j < nn; j++) {
        for (int k = 0; k < g_n[j].nin; k++) {
            int ti = t_index_by_name(g_n[j].ins[k].name);
            if (ti >= 0 && j > last[ti]) last[ti] = j;
        }
    }
    for (int i = 0; i < nt; i++) {
        T* t = &g_t[i];
        if (!(t->owns && (t->dtype == T_F32 || t->dtype == T_I8 || t->dtype == T_U8) &&
              t->numel > 0 && t->elem_size > 0 &&
              (size_t)t->numel <= SIZE_MAX / t->elem_size)) continue;
        if (prod[i] < 0 || last[i] < 0 ||
            tensor_is_declared_graph_output(t->name))
            continue;                                      // graph input/const, output, or dead
        int aliased = 0;
        for (int u = 0; u < nt; u++) if (u != i && g_t[u].data == t->data) { aliased = 1; break; }
        if (aliased) continue;
        poolable[i] = 1;
    }

    // Phase 1: assign each poolable tensor a buffer INDEX, growing byte capacities.
    // (Buffers are allocated once after sizing so pointers never move — a tensor is written
    // every forward within its lifetime, so its data pointer must stay valid for good.)
    int nbufs = 0;
    size_t orig_bytes = 0;
    for (int i = 0; i < nn; i++) {
        for (int t = 0; t < nt; t++) {                     // tensors produced at node i
            if (!poolable[t] || prod[t] != i) continue;
            size_t need = (size_t)g_t[t].numel * g_t[t].elem_size;
            if (need > SIZE_MAX - orig_bytes) {
                free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
                return;
            }
            orig_bytes += need;
            int best = -1;                                  // best-fit among free buffers
            for (int b = 0; b < nbufs; b++)
                if (!busy[b] && cap[b] >= need && (best < 0 || cap[b] < cap[best])) best = b;
            if (best < 0) {                                 // none fit: grow largest free, else new
                for (int b = 0; b < nbufs; b++)
                    if (!busy[b] && (best < 0 || cap[b] > cap[best])) best = b;
                if (best < 0) { best = nbufs++; cap[best] = 0; }
                if (cap[best] < need) cap[best] = need;
            }
            busy[best] = 1;
            buf_of[t] = best;
        }
        for (int t = 0; t < nt; t++)                        // free tensors last-used at node i
            if (poolable[t] && last[t] == i && buf_of[t] >= 0) busy[buf_of[t]] = 0;
    }

    // Phase 2: allocate the arena buffers once at their final sizes.
    int ok = 1;
    for (int b = 0; b < nbufs; b++) {
        bufs[b] = vx_arena_allocate(cap[b] > 0 ? cap[b] : 1u);
        if (!bufs[b]) { ok = 0; break; }
    }
    if (!ok) {                                              // OOM: undo, keep original calloc buffers
        for (int b = 0; b < nbufs; b++) vx_arena_release(bufs[b]);
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }

    // Phase 3: point each pooled tensor at its buffer, dropping its own calloc buffer.
    int pooled_count = 0;
    for (int t = 0; t < nt; t++) if (poolable[t] && buf_of[t] >= 0) pooled_count++;
    int* arena_tensor_indices = pooled_count > 0 ? (int*)malloc((size_t)pooled_count * sizeof(int)) : NULL;
    if (pooled_count > 0 && !arena_tensor_indices) {
        for (int b = 0; b < nbufs; b++) vx_arena_release(bufs[b]);
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }
    int pooled_index = 0;
    for (int t = 0; t < nt; t++) {
        if (!poolable[t] || buf_of[t] < 0) continue;
        free(g_t[t].data);
        g_t[t].data = (float*)bufs[buf_of[t]];
        g_t[t].owns = 0;
        arena_tensor_indices[pooled_index++] = t;
    }

    size_t arena_bytes = 0;
    for (int b = 0; b < nbufs; b++) {
        if (cap[b] > SIZE_MAX - arena_bytes) {
            /* The allocations are already valid; keep the plan and avoid an
             * overflowing debug-only total. */
            arena_bytes = SIZE_MAX;
            break;
        }
        arena_bytes += cap[b];
    }
    g_arena_bufs = bufs;
    vx_engine_state_current()->arena_allocated_bytes = arena_bytes;
    g_arena_nbufs = nbufs;
    g_arena_tensor_indices = arena_tensor_indices;
    g_arena_tensor_count = pooled_count;
    if (g_debug)
        vx_engine_log("[debug] arena_plan pooled=%.1f MB -> arena=%.1f MB (%d buffers)\n",
                (double)orig_bytes / 1048576.0, (double)arena_bytes / 1048576.0, nbufs);

    free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy);
}

int volvoxai_engine_prepare_tensor_table_mutation(void) {
    return restore_arena_tensors();
}

#if VOLVOXAI_ENABLE_TRAINING
int volvoxai_engine_prepare_training_transition(void) {
    if (vx_engine_state_current()->shape_policy.retain_activations) return 0;
    /* Training temporarily unfuses this exact topology and then restores it.
     * Resolved plans contain indices and offsets, not tensor data pointers.
     * Keep those immutable plans when no arena remains to detach. Structural
     * mutation continues to clear them through the ordinary entry above. */
    return g_arena_tensor_count ? restore_arena_tensors() : 0;
}
#endif

void volvoxai_engine_finish_tensor_table_mutation(void) {
    /* Structural callers hold both locks until this boundary and may publish
     * a rebuilt immutable index. Non-structural arena restoration reaches the
     * same hook with an already-current index, making this a read-only no-op. */
    volvoxai_engine_tensor_name_index_rebuild();
    plan_memory_arena();
}
