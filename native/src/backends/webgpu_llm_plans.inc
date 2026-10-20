/* The same checked shader job describes Vulkan, OpenGL, Metal and WebGPU.
 * Two owner-local workspaces are reserved before encoding a forward and reused
 * across its serial operators. Model weight buffers are bound directly. */
static int vx_webgpu_llm_operator(VxOperatorKind op) {
    return (op >= VX_OP_GATED_DELTA_NET && op <= VX_OP_PACKED_LINEAR) ||
        op == VX_OP_MOE_ROUTER || op == VX_OP_MOE_GATED_MLP;
}

static int vx_webgpu_llm_node(const Node* node) {
    if (!node || !vx_webgpu_llm_operator(node->operator_kind)) return 0;
    if (node->operator_kind != VX_OP_MOE_ROUTER) return 1;
    /* Keep the existing basic router's expert-bank domain. The extended
     * router uses parallel workgroup storage with a checked bank limit. */
    return vx_node_param_has(node, VX_NODE_PARAM_SCORING_FUNC) ||
        vx_node_param_has(node, VX_NODE_PARAM_NUM_GROUPS) ||
        vx_node_param_has(node, VX_NODE_PARAM_TOPK_GROUPS) ||
        vx_node_param_has(node, VX_NODE_PARAM_GROUP_SCORE_TOP_K) ||
        vx_node_param_has(node, VX_NODE_PARAM_ROUTED_SCALING_FACTOR) ||
        vx_node_param_has(node, VX_NODE_PARAM_NORMALIZATION_EPSILON) ||
        vx_webgpu_port(node, VX_PORT_CORRECTION_BIAS) ||
        vx_webgpu_port(node, VX_PORT_HASH_INDICES) ||
        vx_webgpu_port(node, VX_PORT_TOKEN_IDS) ||
        vx_node_param_i32(node, VX_NODE_PARAM_TOP_K, 2) > 8;
}

static int vx_webgpu_llm_job(const Node* node, uint64_t workspace[2],
                            VxLlmGpuEmit emit, void* user) {
    workspace[0] = workspace[1] = 0;
    if (!node || !vx_webgpu_llm_operator(node->operator_kind)) return 0;
    if (node->operator_kind == VX_OP_MOE_ROUTER) {
        VxLlmGpuRouterArgs args;
        return vx_engine_llm_router_args((Node*)node, &args) &&
            vx_llm_gpu_router_plan(&args, emit, user);
    }
    if (node->operator_kind == VX_OP_MOE_GATED_MLP) {
        VxLlmGpuGatedArgs args;
        return vx_engine_llm_gated_args((Node*)node, &args) &&
            vx_llm_gpu_gated_plan(&args, workspace, emit, user);
    }
    if (node->operator_kind == VX_OP_N_GRAM_IDS) {
        VxLlmGpuNgramArgs args;
        return vx_engine_llm_ngram_args((Node*)node, &args) &&
            vx_llm_gpu_ngram_plan(&args, emit, user);
    }
    VxLLMOpArgs args;
    return vx_engine_llm_node_args((Node*)node, &args) &&
        vx_llm_gpu_plan(&args, workspace, emit, user);
}

static int vx_webgpu_llm_reserve(const uint64_t bytes[2]) {
    for (unsigned slot = 0; slot < 2; ++slot) {
        if (!bytes[slot] || bytes[slot] <= g_webgpu.llm_workspace_bytes[slot]) continue;
        if (bytes[slot] > UINT32_MAX - 15u || bytes[slot] > SIZE_MAX ||
            bytes[slot] > g_webgpu.limits.maxStorageBufferBindingSize ||
            bytes[slot] > g_webgpu.limits.maxBufferSize) return 0;
        void* host = vx_webgpu_arena((size_t)bytes[slot]);
        if (!host) return 0;
        g_webgpu.llm_workspace[slot] = host;
        g_webgpu.llm_workspace_bytes[slot] = (size_t)bytes[slot];
    }
    if (!g_webgpu.llm_zero) {
        /* Distinct writable optional outputs cannot alias the read-only zero
         * binding, even when a uniform prevents all accesses to either. */
        g_webgpu.llm_zero = vx_webgpu_arena(9u * 16u);
        if (!g_webgpu.llm_zero) return 0;
        memset(g_webgpu.llm_zero, 0, 9u * 16u);
        if (!validating)
            for (unsigned slot = 0; slot < 9; ++slot)
                vx_webgpu_mark_host(NULL, (unsigned char*)g_webgpu.llm_zero + slot * 16u, 4u, 0);
    }
    return 1;
}

static int vx_webgpu_llm_prepare_workspace(void) {
    uint64_t maximum[2] = {0, 0};
    int has_job = 0;
    memset(g_webgpu.llm_workspace, 0, sizeof(g_webgpu.llm_workspace));
    memset(g_webgpu.llm_workspace_bytes, 0, sizeof(g_webgpu.llm_workspace_bytes));
    g_webgpu.llm_zero = NULL;
#if VOLVOXAI_ENABLE_TRAINING
    if (g_native_training_mode) return 1;
#endif
    for (int i = 0; i < g_nn; ++i) {
        if (!vx_webgpu_llm_node(&g_n[i])) continue;
        uint64_t bytes[2];
        if (!vx_webgpu_llm_job(&g_n[i], bytes, NULL, NULL)) return 0;
        has_job = 1;
        for (unsigned slot = 0; slot < 2; ++slot)
            if (bytes[slot] > maximum[slot]) maximum[slot] = bytes[slot];
    }
    /* A bounded graph already proved its largest shape. Reserve that shape's
     * serial workspace before the first token so later prefixes keep the same
     * host identity and GPUBuffer instead of reallocating at each step. */
    const VxDynamicShapeMaximumLayout* domain =
        &vx_engine_state_current()->dynamic_shape_maximum_layout;
    for (unsigned slot = 0; slot < 2; ++slot)
        if (domain->modern_workspace_slot_bytes[slot] > maximum[slot])
            maximum[slot] = domain->modern_workspace_slot_bytes[slot];
    return !has_job || vx_webgpu_llm_reserve(maximum);
}

typedef struct {
    VxWebGpuNodePlan* plans;
    uint32_t count, capacity;
} VxWebGpuLlmEmission;

static int vx_webgpu_llm_emit(void* user, const VxLlmGpuDispatch* job) {
    VxWebGpuLlmEmission* emission = user;
    if (!job || emission->count >= emission->capacity || job->buffer_count > 8u) return 0;
    const char* name = vx_llm_gpu_shader_name(job->shader);
    uint32_t shader = VX_SHADER_COUNT;
    if (!name) return 0;
    for (uint32_t i = 0; i < VX_SHADER_COUNT; ++i)
        if (!strcmp(vx_shader_programs[i].name, name) &&
            !strcmp(vx_shader_programs[i].entry, "main")) { shader = i; break; }
    if (shader == VX_SHADER_COUNT) return 0;
    VxWebGpuNodePlan* plan = &emission->plans[emission->count];
    memset(plan, 0, sizeof(*plan));
    for (uint32_t i = 0; i < job->buffer_count; ++i) {
        const VxLlmGpuBuffer* buffer = &job->buffers[i];
        const void* host = buffer->host;
        if (buffer->workspace == 1u || buffer->workspace == 2u) {
            unsigned slot = buffer->workspace - 1u;
            if (buffer->bytes > g_webgpu.llm_workspace_bytes[slot]) return 0;
            host = g_webgpu.llm_workspace[slot];
            /* Residency owns the maximum allocation; each dispatch exposes
             * only its checked slice. Different layers reuse one GPUBuffer. */
            if (!vx_webgpu_bind(plan, host, g_webgpu.llm_workspace_bytes[slot])) return 0;
            plan->bindings[plan->binding_count - 1u].bytes = (uint32_t)buffer->bytes;
            continue;
        } else if (buffer->workspace == 3u) {
            host = (unsigned char*)g_webgpu.llm_zero + (buffer->output ? (i + 1u) * 16u : 0u);
            if (buffer->bytes != 4u) return 0;
        } else if (buffer->workspace != 0u) return 0;
        if (buffer->bytes > SIZE_MAX || !vx_webgpu_bind(plan, host, (size_t)buffer->bytes)) return 0;
    }
    if (!vx_webgpu_params(plan, job->uniform, job->shader == VX_LLM_GPU_COPY ? 4u : 32u) ||
        !vx_webgpu_grid(plan, shader, job->groups[0], job->groups[1], job->groups[2])) return 0;
    emission->count++;
    return 1;
}

static int vx_webgpu_plan_llm_sequence(const Node* node,
                                      VxWebGpuNodePlan* plans, uint32_t* count) {
    uint64_t bytes[2];
    if (!vx_webgpu_llm_job(node, bytes, NULL, NULL)) return 0;
    /* Maximum-domain compilation uses an isolated proof owner, so it reserves
     * this node's workspace instead of scanning a mixed projected graph. */
    if (validating && !vx_webgpu_llm_reserve(bytes)) return 0;
    VxWebGpuLlmEmission emission = {plans, 0, vx_webgpu_plan_capacity(node)};
    if (!vx_webgpu_llm_job(node, bytes, vx_webgpu_llm_emit, &emission) || !emission.count) return 0;
    *count = emission.count;
    return 1;
}
