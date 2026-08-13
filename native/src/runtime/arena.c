#include "engine_internal.h"
#include "engine_core.h"
#include "incremental_runtime.h"

#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    for (int index = 0; index < VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY; index++)
        dynamic_shape_plan_clear(&state->dynamic_shape_plans[index]);
    state->dynamic_shape_plan_clock = 0;
}

static int dynamic_passthrough_node(const Node* node) {
    return node && node->skip &&
        (!strcmp(node->op, "Reshape") || !strcmp(node->op, "Flatten") ||
         !strcmp(node->op, "Squeeze") || !strcmp(node->op, "Unsqueeze") ||
         !strcmp(node->op, "Dropout") || !strcmp(node->op, "Identity"));
}

static int arena_tensor_slot(int tensor_index) {
    for (int slot = 0; slot < g_arena_tensor_count; slot++)
        if (g_arena_tensor_indices[slot] == tensor_index) return slot;
    return -1;
}

/* Ordinary lifetime reuse must be split before an incremental seed, but a
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
    for (int i = 0; i < g_arena_nbufs; i++) free(g_arena_bufs[i]);
    free(g_arena_bufs);
    for (int i = 0; i < g_arena_tensor_count; i++) {
        int index = g_arena_tensor_indices[i];
        if (index < 0 || index >= g_nt) continue;
        g_t[index].data = NULL;
        g_t[index].owns = 0;
    }
    free(g_arena_tensor_indices);
    free(state->dynamic_shape_reserved_arena);
    state->dynamic_shape_reserved_arena = NULL;
    g_arena_bufs = NULL;
    g_arena_nbufs = 0;
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
        /* A full incremental seed extends every activation lifetime across
         * later steps. Two values that shared one ordinary-forward arena slot
         * must therefore receive independent storage here: preserving the
         * physical alias would let a later branch overwrite a skipped cached
         * value. This is a lifetime split, not a graph-level view alias. */
        replacements[i] = calloc(tensor->numel > 0 ? (size_t)tensor->numel : 1u, tensor->elem_size);
        if (!replacements[i]) goto fail;
        replacement_owns[i] = 1u;
        if (bytes) memcpy(replacements[i], tensor->data, bytes);
    }
    for (int i = 0; i < g_arena_nbufs; i++) free(g_arena_bufs[i]);
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
        if (strcmp(node->ins[index].key, "input")) continue;
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
    if (!strcmp(node->op, "QLayerNorm")) {
        const size_t feature = (size_t)maximum_shape[rank - 1];
        if (!feature || activation_elements % feature) return -1;
        units = activation_elements / feature;
    } else {
        cJSON* groups_json;
        int groups = 1;
        if (rank != 4) return -1;
        groups_json = node->params
            ? cJSON_GetObjectItemCaseSensitive(node->params, "num_groups")
            : NULL;
        if (!groups_json && node->params)
            groups_json = cJSON_GetObjectItemCaseSensitive(
                node->params, "groups");
        if (groups_json) {
            if (!cJSON_IsNumber(groups_json) || groups_json->valueint <= 0 ||
                groups_json->valuedouble != (double)groups_json->valueint)
                return -1;
            groups = groups_json->valueint;
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
        descriptor_count > MAXT || !out) return -1;
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
        if (!strcmp(g_n[node].op, "QGroupNorm"))
            maximum = &out->qgroupnorm_stats_bytes;
        else if (!strcmp(g_n[node].op, "QLayerNorm"))
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
        if (tensor->is_graph_input ||
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
        descriptor_count > MAXT || !maximum || !maximum->configured ||
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
    VolvoxAIEngineMaximumTensor* exact_descriptors = NULL;
    VxDynamicShapeMaximumLayout exact_layout = {0};
    int result;
    if (maximum && maximum->configured)
        return dynamic_shape_plan_project_maximum(
            signature, descriptors, descriptor_count, maximum, out);

    /* CPU retains its exact-shape liveness layout and geometric growth path.
     * Dynamic native-GPU contexts configure a fixed maximum-domain layout. */
    if (!descriptors || !descriptor_count || descriptor_count > MAXT)
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
        descriptor_count > MAXT) return -1;
    took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    if (!g_loaded || state->dynamic_arena_active) goto done;
    for (int index = 0; index < VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY; index++)
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
    size_t capacity = current ? current : 4096u;
    if (capacity < current || !required) return required;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) return required;
        capacity *= 2u;
    }
    return capacity;
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
    arena = malloc(maximum->arena_bytes);
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
    free(arena);
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
    VxDynamicShapePlan* victim = NULL;
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
        !descriptor_count || descriptor_count > MAXT ||
        !stats || stats->struct_size != sizeof(*stats)) return -1;
    took_model_lock = !volvoxai_engine_model_route_lease_active();
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    if (!g_loaded) goto done;
    if (dynamic_bindings_validate(descriptors, descriptor_count, inputs,
                                  input_count, 1) != 0) goto done;
    for (int index = 0; index < VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY; index++) {
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
        for (int index = 0; index < VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY;
             index++) {
            VxDynamicShapePlan* entry = &state->dynamic_shape_plans[index];
            if (!entry->signature || !victim ||
                entry->last_use < victim->last_use) victim = entry;
            if (!entry->signature) break;
        }
        if (!victim) goto done;
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
        if (candidate_capacity < plan->arena_bytes) goto done;
        candidate_arena = malloc(candidate_capacity ? candidate_capacity : 1u);
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
        if (inputs[index].byte_size)
            memcpy(tensor->data, inputs[index].data, inputs[index].byte_size);
        vx_incremental_mark_tensor_locked(tensor);
        if (vx_runtime_backend_has_graph() && inputs[index].byte_size)
            vx_runtime_backend_mark_host(
                tensor->data, inputs[index].byte_size, 0);
    }
    for (size_t index = 0; index < owned_count; index++)
        free(owned_storage[index]);
    for (int index = 0; index < g_arena_nbufs; index++)
        if (g_arena_bufs[index] != candidate_arena) free(g_arena_bufs[index]);
    free(g_arena_bufs);
    free(g_arena_tensor_indices);
    g_arena_bufs = next_buffers;
    g_arena_nbufs = 1;
    g_arena_tensor_indices = next_indices;
    g_arena_tensor_count = (int)descriptor_count;
    if (consumed_reservation)
        state->dynamic_shape_reserved_arena = NULL;
    next_buffers = NULL;
    next_indices = NULL;
    if (!cache_hit) {
        dynamic_shape_plan_clear(victim);
        *victim = candidate_plan;
        memset(&candidate_plan, 0, sizeof(candidate_plan));
        plan = victim;
    }
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
    stats->logical_bytes = plan->logical_bytes;
    stats->required_arena_bytes = plan->arena_bytes;
    stats->arena_capacity_bytes = candidate_capacity;
    stats->arena_high_water_bytes = state->dynamic_arena_high_water_bytes;
    stats->arena_grow_count = state->dynamic_arena_grow_count;
    stats->resource_generation = state->dynamic_resource_generation;
    result = 0;
done:
    if (result != 0 && grew) free(candidate_arena);
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
        if (inputs[index].byte_size)
            memcpy(tensor->data, inputs[index].data, inputs[index].byte_size);
        vx_incremental_mark_tensor_locked(tensor);
        if (vx_runtime_backend_has_graph() && inputs[index].byte_size)
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
    const char* en = getenv("VOLVOX_ARENA");
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
        bufs[b] = malloc(cap[b] > 0 ? cap[b] : 1u);
        if (!bufs[b]) { ok = 0; break; }
    }
    if (!ok) {                                              // OOM: undo, keep original calloc buffers
        for (int b = 0; b < nbufs; b++) free(bufs[b]);
        free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy); free(bufs);
        return;
    }

    // Phase 3: point each pooled tensor at its buffer, dropping its own calloc buffer.
    int pooled_count = 0;
    for (int t = 0; t < nt; t++) if (poolable[t] && buf_of[t] >= 0) pooled_count++;
    int* arena_tensor_indices = pooled_count > 0 ? (int*)malloc((size_t)pooled_count * sizeof(int)) : NULL;
    if (pooled_count > 0 && !arena_tensor_indices) {
        for (int b = 0; b < nbufs; b++) free(bufs[b]);
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
    g_arena_nbufs = nbufs;
    g_arena_tensor_indices = arena_tensor_indices;
    g_arena_tensor_count = pooled_count;
    if (g_debug)
        fprintf(stderr, "[debug] arena_plan pooled=%.1f MB -> arena=%.1f MB (%d buffers)\n",
                (double)orig_bytes / 1048576.0, (double)arena_bytes / 1048576.0, nbufs);

    free(prod); free(last); free(buf_of); free(poolable); free(cap); free(busy);
}

int volvoxai_engine_prepare_tensor_table_mutation(void) {
    return restore_arena_tensors();
}

void volvoxai_engine_finish_tensor_table_mutation(void) {
    /* Structural callers hold both locks until this boundary and may publish
     * a rebuilt immutable index. Non-structural arena restoration reaches the
     * same hook with an already-current index, making this a read-only no-op. */
    volvoxai_engine_tensor_name_index_rebuild();
    plan_memory_arena();
}
