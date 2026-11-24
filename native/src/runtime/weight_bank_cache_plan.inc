/* Compile/context admission for explicit CUDA weight-bank caches. Included by
 * public_api.c; the immutable source graph and tensor shapes are unchanged. */
#include "weight_bank_cache.h"

#if VOLVOXAI_ENABLE_CUDA
static const T* vx_weight_bank_cache_tensor(const VxEngineState* state,
                                           const char* name) {
    if (!state || !name) return NULL;
    for (int i = 0; i < state->tensor_count; ++i)
        if (!strcmp(state->tensors[i].name, name)) return &state->tensors[i];
    return NULL;
}

static const T* vx_weight_bank_cache_input(const VxEngineState* state,
                                          const Node* node, VxPortKind port) {
    for (int i = 0; node && i < node->nin; ++i)
        if (node->ins[i].port == port)
            return vx_weight_bank_cache_tensor(state, node->ins[i].name);
    return NULL;
}

static int vx_weight_bank_cache_span_bytes(const T* tensor, size_t* bytes) {
    if (!tensor || !tensor->data || !tensor->elem_size || tensor->numel <= 0 ||
        (uint64_t)tensor->numel > SIZE_MAX / tensor->elem_size) return 0;
    *bytes = (size_t)tensor->numel * tensor->elem_size;
    return 1;
}

static int vx_weight_bank_cache_tensor_bytes(const T* tensor, size_t* bytes) {
    return tensor && ((tensor->dtype == T_F32 && tensor->elem_size == sizeof(float)) ||
        ((tensor->dtype == T_F16 || tensor->dtype == T_BF16) && tensor->elem_size == sizeof(uint16_t))) &&
        vx_weight_bank_cache_span_bytes(tensor, bytes);
}

static int vx_weight_bank_cache_immutable_source(const VxEngineState* state,
                                                const T* tensor) {
    size_t bytes;
    if (!state->weight_files_borrowed || !tensor || tensor->owns ||
        tensor->is_graph_input || !vx_weight_bank_cache_tensor_bytes(tensor, &bytes))
        return 0;
    const VxConvertedWeight* converted = vx_compiled_weight_conversions_find(
        state->compiled_weight_conversions, tensor->name);
    if (tensor->dtype == T_F32 && converted && converted->data == tensor->data && converted->byte_size == bytes)
        return 1;
    for (int i = 0; i < state->weight_file_count; ++i) {
        const SafetensorsTensor* source =
            safetensors_find_tensor(&state->weight_files[i], tensor->name);
        if (source && (uint32_t)source->dtype == (uint32_t)tensor->dtype &&
            source->data == tensor->data && source->nbytes == bytes) return 1;
    }
    return 0;
}

static int vx_weight_bank_cache_overlap(const void* a, size_t an,
                                        const void* b, size_t bn) {
    if (!a || !b || !an || !bn) return 0;
    uintptr_t x = (uintptr_t)a, y = (uintptr_t)b;
    return x <= y ? y - x < an : x - y < bn;
}

static int vx_weight_bank_cache_group_overlap(const VxWeightBankCacheGroup* group,
                                              const void* base, size_t bytes) {
    for (size_t i = 0; i < 3; ++i)
        if (vx_weight_bank_cache_overlap(group->weights[i], group->weight_bytes[i],
                                         base, bytes)) return 1;
    return vx_weight_bank_cache_overlap(group->bias, group->bias_bytes, base, bytes);
}
#endif

static const VxWeightBankCacheGroup* vx_weight_bank_cache_group(
        const VxWeightBankCachePlan* plan, const void* primary) {
    for (size_t i = 0; plan && i < plan->count; ++i)
        if (plan->groups[i].weights[0] == primary) return &plan->groups[i];
    return NULL;
}

static int vx_weight_bank_cache_source(const VxWeightBankCachePlan* plan,
                                       const void* base, size_t bytes) {
    if (!base || !bytes) return 0;
    for (size_t i = 0; plan && i < plan->count; ++i) {
        const VxWeightBankCacheGroup* group = &plan->groups[i];
        for (size_t j = 0; j < 3; ++j)
            if (group->weights[j] == base && group->weight_bytes[j] == bytes)
                return 1;
        if (group->bias == base && group->bias_bytes == bytes) return 1;
    }
    return 0;
}

#if VOLVOXAI_ENABLE_CUDA
static int vx_weight_bank_cache_resolve(const VxEngineState* state,
        const Node* node, uint32_t slots, VxWeightBankCacheGroup* group) {
    const T* weights[3] = {NULL};
    const T* bias = NULL;
    if (node->resident_slot_rows || node->resident_slot_domain) return 0;
    if (node->operator_kind == VX_OP_MOE_LINEAR) {
        weights[0] = vx_weight_bank_cache_input(state, node, VX_PORT_EXPERT_WEIGHT);
        bias = vx_weight_bank_cache_input(state, node, VX_PORT_EXPERT_BIAS);
    } else if (node->operator_kind == VX_OP_MOE_GATED_MLP) {
        weights[0] = vx_weight_bank_cache_input(state, node, VX_PORT_GATE_WEIGHT);
        weights[1] = vx_weight_bank_cache_input(state, node, VX_PORT_UP_WEIGHT);
        weights[2] = vx_weight_bank_cache_input(state, node, VX_PORT_DOWN_WEIGHT);
    } else return 0;
    memset(group, 0, sizeof(*group));
    for (size_t i = 0; i < (node->operator_kind == VX_OP_MOE_LINEAR ? 1u : 3u); ++i) {
        if (!vx_weight_bank_cache_immutable_source(state, weights[i]) ||
            weights[i]->ndim != 3 || weights[i]->shape[0] <= 0 ||
            weights[i]->shape[1] <= 0 || weights[i]->shape[2] <= 0 ||
            !vx_weight_bank_cache_tensor_bytes(weights[i], &group->weight_bytes[i]))
            return 0;
        group->weights[i] = weights[i]->data;
        group->weight_dtype[i] = (uint32_t)weights[i]->dtype;
    }
    group->operator_kind = (uint32_t)node->operator_kind;
    snprintf(group->bank, sizeof(group->bank), "%s", weights[0]->name);
    group->experts = (uint32_t)weights[0]->shape[0];
    group->d_in = (uint32_t)weights[0]->shape[1];
    group->resident_slots = slots;
    /* The bootstrap projects symbolic dimensions to their minima. Reserve a
     * conservative K bound here; declared-domain proof checks maximum K. */
    group->max_top_k = slots;
    if (!slots || slots > group->experts) return 0;
    if (node->operator_kind == VX_OP_MOE_LINEAR) {
        group->d_out = (uint32_t)weights[0]->shape[2];
        if (bias) {
            if (bias->dtype != T_F32 || !vx_weight_bank_cache_immutable_source(state, bias) ||
                (uint64_t)bias->numel != (uint64_t)group->experts * group->d_out ||
                !vx_weight_bank_cache_tensor_bytes(bias, &group->bias_bytes))
                return 0;
            group->bias = bias->data;
        }
    } else {
        group->hidden = (uint32_t)weights[0]->shape[2];
        group->d_out = (uint32_t)weights[2]->shape[2];
        for (int axis = 0; axis < 3; ++axis)
            if (weights[1]->shape[axis] != weights[0]->shape[axis]) return 0;
        if (weights[2]->shape[0] != (int)group->experts ||
            weights[2]->shape[1] != (int)group->hidden || group->hidden > 8192u)
            return 0;
    }
    return 1;
}

static int vx_weight_bank_cache_groups_equal(const VxWeightBankCacheGroup* a,
                                             const VxWeightBankCacheGroup* b) {
    return a->operator_kind == b->operator_kind && a->experts == b->experts &&
        a->d_in == b->d_in && a->hidden == b->hidden && a->d_out == b->d_out &&
        a->resident_slots == b->resident_slots && a->max_top_k == b->max_top_k &&
        a->weights[0] == b->weights[0] && a->weights[1] == b->weights[1] &&
        a->weights[2] == b->weights[2] && a->bias == b->bias &&
        !memcmp(a->weight_dtype,b->weight_dtype,sizeof(a->weight_dtype));
}
#endif

static VxStatus vx_weight_bank_cache_prepare(VxEngineState* state,
        const VxWeightBankCache* configs, size_t count,
        char* error, size_t error_capacity) {
    if (error && error_capacity) error[0] = '\0';
    if (!count) return VX_STATUS_OK;
    if (!state || !configs || count > 128u || state->weight_bank_cache_plan)
        return VX_STATUS_INVALID_ARGUMENT;
#if !VOLVOXAI_ENABLE_CUDA
    if (error && error_capacity)
        snprintf(error, error_capacity, "weight-bank caching requires native CUDA");
    return VX_STATUS_BACKEND_UNSUPPORTED;
#else
    VxWeightBankCachePlan* plan = NULL;
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    const char* message = "weight-bank caching requires immutable F32/F16/BF16 MoE banks";
    if (!state->use_cuda || !state->weight_files_borrowed) goto done;
    plan = (VxWeightBankCachePlan*)calloc(1, sizeof(*plan));
    if (!plan) return VX_STATUS_OUT_OF_MEMORY;
    plan->groups = (VxWeightBankCacheGroup*)calloc(count, sizeof(*plan->groups));
    if (!plan->groups) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    const cJSON* banks = cJSON_GetObjectItemCaseSensitive(state->graph_root, "banks");
    for (size_t ci = 0; ci < count; ++ci) {
        const VxWeightBankCache* config = &configs[ci];
        int found = 0;
        if (!config->bank || !config->bank[0] || strlen(config->bank) >= 128u ||
            !config->resident_slots ||
            !cJSON_IsString(cJSON_GetObjectItemCaseSensitive(banks, config->bank))) {
            status = VX_STATUS_INVALID_ARGUMENT;
            message = "weight-bank cache names must identify declared banks with positive slots";
            goto done;
        }
        for (size_t earlier = 0; earlier < ci; ++earlier)
            if (!strcmp(configs[earlier].bank, config->bank)) {
                status = VX_STATUS_INVALID_ARGUMENT;
                message = "weight-bank cache names must be unique";
                goto done;
            }
        for (int ni = 0; ni < state->node_count; ++ni) {
            const Node* node = &state->nodes[ni];
            VxPortKind port = node->operator_kind == VX_OP_MOE_LINEAR
                ? VX_PORT_EXPERT_WEIGHT : node->operator_kind == VX_OP_MOE_GATED_MLP
                ? VX_PORT_GATE_WEIGHT : VX_PORT_UNSPECIFIED;
            if (port == VX_PORT_UNSPECIFIED) continue;
            const T* primary = vx_weight_bank_cache_input(state, node, port);
            if (!primary || strcmp(primary->name, config->bank)) continue;
            if (vx_fixed_partition_cpu(state->fixed_partition, (size_t)ni)) {
                message = "cached expert banks cannot be consumed by CPU-assigned nodes";
                goto done;
            }
            VxWeightBankCacheGroup group;
            if (!vx_weight_bank_cache_resolve(state, node, config->resident_slots, &group))
                goto done;
            if (!found) plan->groups[ci] = group;
            else if (!vx_weight_bank_cache_groups_equal(&plan->groups[ci], &group)) {
                message = "cached bank consumers must use the same complete MoE weight group";
                goto done;
            }
            found = 1;
        }
        if (!found) { message = "cache bank is not a MoE primary weight"; goto done; }
        plan->count = ci + 1u;
    }
    /* Aliases or other consumers could trigger a whole-source upload. Admit
     * only exact weight ports of the resolved MoE group, including shared use. */
    for (size_t gi = 0; gi < plan->count; ++gi) {
        const VxWeightBankCacheGroup* group = &plan->groups[gi];
        for (size_t gj = 0; gj < gi; ++gj) {
            for (size_t wi = 0; wi < 3; ++wi)
                if (vx_weight_bank_cache_group_overlap(&plan->groups[gj],
                        group->weights[wi], group->weight_bytes[wi])) {
                    message = "distinct cache entries cannot share source storage";
                    goto done;
                }
            if (vx_weight_bank_cache_group_overlap(&plan->groups[gj],
                    group->bias, group->bias_bytes)) goto done;
        }
        for (int ni = 0; ni < state->node_count; ++ni) {
            const Node* node = &state->nodes[ni];
            VxWeightBankCacheGroup candidate;
            int same_group = vx_weight_bank_cache_resolve(state, node,
                group->resident_slots, &candidate) &&
                vx_weight_bank_cache_groups_equal(group, &candidate);
            for (int pi = 0; pi < node->nin + node->nout; ++pi) {
                int is_output = pi >= node->nin;
                const Ref* ref = is_output ? &node->outs[pi - node->nin] : &node->ins[pi];
                const T* tensor = vx_weight_bank_cache_tensor(state, ref->name);
                size_t bytes;
                if (!tensor || !vx_weight_bank_cache_span_bytes(tensor, &bytes) ||
                    !vx_weight_bank_cache_group_overlap(group, tensor->data, bytes)) continue;
                int allowed = !is_output && same_group &&
                    ((ref->port == VX_PORT_EXPERT_WEIGHT && tensor->data == group->weights[0]) ||
                     (ref->port == VX_PORT_EXPERT_BIAS && tensor->data == group->bias) ||
                     (ref->port == VX_PORT_GATE_WEIGHT && tensor->data == group->weights[0]) ||
                     (ref->port == VX_PORT_UP_WEIGHT && tensor->data == group->weights[1]) ||
                     (ref->port == VX_PORT_DOWN_WEIGHT && tensor->data == group->weights[2]));
                if (!allowed) {
                    message = "cached weights cannot be graph outputs or inputs of other operators";
                    goto done;
                }
            }
        }
        const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(state->graph_root, "outputs");
        for (const cJSON* output = outputs ? outputs->child : NULL; output; output = output->next) {
            const T* tensor = vx_weight_bank_cache_tensor(state, cJSON_GetStringValue(output));
            size_t bytes;
            if (vx_weight_bank_cache_span_bytes(tensor, &bytes) &&
                vx_weight_bank_cache_group_overlap(group, tensor->data, bytes)) {
                message = "cached weights cannot be graph outputs";
                goto done;
            }
        }
    }
    VxEngineStateScope scope = vx_engine_state_scope_enter(state);
    cuda_graph_clear_allocation_failure();
    int registered = 1;
    for (size_t i = 0; i < plan->count && registered; ++i) {
        const VxWeightBankCacheGroup* group = &plan->groups[i];
        registered = group->operator_kind == VX_OP_MOE_LINEAR
            ? cuda_graph_moe_cache_register_linear_storage(group->weights[0], group->weight_dtype[0], group->bias,
                group->experts, group->d_in, group->d_out,
                group->max_top_k, group->resident_slots)
            : cuda_graph_moe_cache_register_gated_storage(group->weights[0], group->weight_dtype[0],
                group->weights[1], group->weight_dtype[1], group->weights[2], group->weight_dtype[2],
                group->experts, group->d_in, group->hidden,
                group->d_out, group->max_top_k, group->resident_slots);
    }
    if (!registered) status = cuda_graph_last_allocation_failed()
        ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_BACKEND_UNSUPPORTED;
    vx_engine_state_scope_leave(scope);
    if (!registered) { message = "CUDA weight-bank cache could not be reserved"; goto done; }
    state->weight_bank_cache_plan = plan;
    return VX_STATUS_OK;
done:
    vx_weight_bank_cache_plan_destroy(plan);
    if (error && error_capacity) snprintf(error, error_capacity, "%s", message);
    return status;
#endif
}

static int vx_weight_bank_cache_bounds(const VxWeightBankCachePlan* plan,
        uint64_t alignment, uint64_t* host_out, uint64_t* device_out) {
    if (!alignment || !host_out || !device_out) return 0;
    *host_out = *device_out = 0;
    if (!plan) return 1;
#if !VOLVOXAI_ENABLE_CUDA
    return 0;
#else
    if (plan->count > (UINT64_MAX - sizeof(*plan)) / sizeof(*plan->groups)) return 0;
    uint64_t host = sizeof(*plan) + plan->count * sizeof(*plan->groups), device = 0;
    /* Every group shares the context's bounded pinned upload storage. */
    uint64_t staging = cuda_graph_moe_cache_staging_bound();
    if (staging > UINT64_MAX - host) return 0;
    host += staging;
    for (size_t i = 0; i < plan->count; ++i) {
        const VxWeightBankCacheGroup* group = &plan->groups[i];
        uint64_t group_host = 0, group_device = 0;
        int ok = group->operator_kind == VX_OP_MOE_LINEAR
            ? cuda_graph_moe_cache_linear_storage_bounds(group->experts, group->d_in,
                group->d_out, group->max_top_k, group->resident_slots,
                group->weight_dtype[0], group->bias != NULL, &group_host, &group_device)
            : cuda_graph_moe_cache_gated_storage_bounds(group->experts, group->d_in,
                group->hidden, group->d_out, group->max_top_k,
                group->resident_slots, group->weight_dtype[0], group->weight_dtype[1],
                group->weight_dtype[2], &group_host, &group_device);
        /* The backend places weights and route scratch in one allocation. */
        uint64_t padding = (alignment - group_device % alignment) % alignment;
        if (!ok || group_device > UINT64_MAX - padding) return 0;
        group_device += padding;
        if (group_host > UINT64_MAX - host || group_device > UINT64_MAX - device)
            return 0;
        host += group_host;
        device += group_device;
    }
    *host_out = host;
    *device_out = device;
    return 1;
#endif
}
