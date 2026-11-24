/* Explicit CPU placement within a CUDA graph. No placement is inferred from
 * observed fallback, and logical node order is preserved by compilation. */
#include "fixed_partition.h"

static int vx_fixed_partition_cpu_operator(VxOperatorKind op) {
    /* These decoder operators borrow immutable weight storage directly.
     * Convolution/recurrent/quantized authoring routes have separate host
     * materializations and need their own placement resource proof. */
    switch (op) {
        case VX_OP_LINEAR: case VX_OP_MATMUL: case VX_OP_GEMM:
        case VX_OP_BATCH_MATMUL: case VX_OP_EMBEDDING: case VX_OP_GATHER:
        case VX_OP_MOE_ROUTER: case VX_OP_MOE_LINEAR: case VX_OP_MOE_GATED_MLP:
        case VX_OP_SDPA: case VX_OP_CROSS_SDPA:
        case VX_OP_LAYER_NORM: case VX_OP_RMS_NORM: case VX_OP_ROPE:
        case VX_OP_RELU: case VX_OP_GELU: case VX_OP_SILU:
        case VX_OP_SIGMOID: case VX_OP_TANH: case VX_OP_SIN: case VX_OP_COS:
        case VX_OP_CLIP: case VX_OP_EXP: case VX_OP_SQRT: case VX_OP_LOG:
        case VX_OP_NEG: case VX_OP_ADD: case VX_OP_SUB: case VX_OP_MUL:
        case VX_OP_DIV: case VX_OP_SOFTMAX: case VX_OP_LOG_SOFTMAX:
        case VX_OP_REDUCE_SUM: case VX_OP_REDUCE_MEAN: case VX_OP_ARG_MAX:
        case VX_OP_EQUAL: case VX_OP_GREATER_OR_EQUAL: case VX_OP_NOT:
        case VX_OP_WHERE: case VX_OP_MASK: case VX_OP_CUM_SUM:
        case VX_OP_RESHAPE: case VX_OP_FLATTEN: case VX_OP_SQUEEZE:
        case VX_OP_UNSQUEEZE: case VX_OP_TRANSPOSE: case VX_OP_CONCAT:
        case VX_OP_CONCAT2: case VX_OP_SPLIT: case VX_OP_SLICE:
        case VX_OP_PAD: case VX_OP_EXPAND: case VX_OP_GATHER_ELEMENTS:
        case VX_OP_IDENTITY: case VX_OP_BROADCAST: case VX_OP_CAST:
            return 1;
        default: return 0;
    }
}

static int vx_fixed_partition_name_compare(const void* left, const void* right) {
    return strcmp(*(const char* const*)left, *(const char* const*)right);
}

static VxStatus vx_fixed_partition_prepare(VxEngineState* state,
        const char* const* names, size_t count, char* error, size_t capacity) {
    if (error && capacity) error[0] = '\0';
    if (!count) return VX_STATUS_OK;
    if (!state || !names || count > 65536u || state->fixed_partition)
        return VX_STATUS_INVALID_ARGUMENT;
#if !VOLVOXAI_ENABLE_CUDA
    if (error && capacity) snprintf(error, capacity, "fixed CPU placement requires native CUDA");
    return VX_STATUS_BACKEND_UNSUPPORTED;
#else
    const char* message = "fixed CPU placement requires a preserved native CUDA graph";
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    VxFixedPartition* plan = NULL;
    const char** sorted = NULL;
    unsigned char* matched = NULL;
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(state->graph_root, "nodes");
    if (!state->use_cuda || !state->weight_files_borrowed ||
        !state->preserve_node_boundaries || state->node_count <= 0 ||
        !cJSON_IsArray(nodes) || cJSON_GetArraySize(nodes) != state->node_count)
        goto done;
    if ((size_t)state->node_count > SIZE_MAX - sizeof(*plan)) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    plan = calloc(1, sizeof(*plan) + (size_t)state->node_count);
    if (!plan) return VX_STATUS_OUT_OF_MEMORY;
    plan->node_count = (size_t)state->node_count;
    sorted = malloc(count * sizeof(*sorted));
    matched = calloc(count, sizeof(*matched));
    if (!sorted || !matched) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    for (size_t i = 0; i < count; ++i) {
        if (!names[i] || !names[i][0] || strlen(names[i]) >= 128u) {
            status = VX_STATUS_INVALID_ARGUMENT;
            message = "CPU placement names must be nonempty logical node IDs shorter than 128 bytes";
            goto done;
        }
        sorted[i] = names[i];
    }
    qsort(sorted, count, sizeof(*sorted), vx_fixed_partition_name_compare);
    for (size_t i = 1; i < count; ++i) {
        if (!strcmp(sorted[i - 1], sorted[i])) {
            status = VX_STATUS_INVALID_ARGUMENT;
            message = "CPU placement node IDs must be unique";
            goto done;
        }
    }
    size_t index = 0, resolved = 0;
    for (const cJSON* node = nodes->child; node; node = node->next, ++index) {
        const char* id = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(node, "id"));
        const char* const* hit = id
            ? bsearch(&id, sorted, count, sizeof(*sorted), vx_fixed_partition_name_compare)
            : NULL;
        if (!hit) continue;
        size_t slot = (size_t)(hit - sorted);
        if (matched[slot]) {
            status = VX_STATUS_INVALID_ARGUMENT;
            message = "CPU placement names an unknown or ambiguous logical node ID";
            goto done;
        }
        matched[slot] = 1;
        const Node* runtime = &state->nodes[index];
        VxOperatorKind op = vx_operator_kind_from_name(
            cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(node, "opType")));
        if (runtime->skip || runtime->disabled || runtime->operator_kind != op ||
            !vx_fixed_partition_cpu_operator(op)) {
            message = "CPU placement requires a supported unfused decoder operator";
            goto done;
        }
        plan->cpu_nodes[index] = 1;
        resolved++;
    }
    if (resolved != count) {
        status = VX_STATUS_INVALID_ARGUMENT;
        message = "CPU placement names an unknown or ambiguous logical node ID";
        goto done;
    }
    plan->cpu_count = count;
    /* CPU dense nodes use direct F32/compact loads, avoiding a second complete
     * packed host copy that the CUDA weight envelope does not otherwise own. */
    state->gemm_f32_packed_enabled = 0;
    state->fixed_partition = plan;
    free(matched);
    free(sorted);
    return VX_STATUS_OK;
done:
    free(matched);
    free(sorted);
    vx_fixed_partition_destroy(plan);
    if (error && capacity) snprintf(error, capacity, "%s", message);
    return status;
#endif
}

static uint64_t vx_fixed_partition_reserved_host_bytes(const VxEngineState* state) {
    uint64_t bytes = state ? vx_fixed_partition_bytes(state->fixed_partition) : 0u;
#if VOLVOXAI_ENABLE_CUDA
    /* Cache admission already charges this same per-context transfer slab. */
    if (bytes && !state->weight_bank_cache_plan)
        bytes += cuda_graph_moe_cache_staging_bound();
#endif
    return bytes;
}
