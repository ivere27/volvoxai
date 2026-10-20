/* Generation policy belongs to a context, independently of its graph and of
 * other contexts. The graph's ArgMax remains the immutable ID-domain proof;
 * RANDOM replaces only the newly selected row from its declared F32 logits. */
static void vx_full_prefix_state_free(VxFullPrefixState* state) {
    if (!state) return;
    for (size_t i = 0; i < state->count; ++i) free((void*)state->inputs[i].data);
    free(state->inputs);
    free(state);
}

static void vx_generation_policy_clear(VxExecutionContext* context) {
    free(context->generation_candidates);
    free(context->generation_policy_ids);
    free(context->generation_sequences);
    free(context->generation_history);
    context->generation_candidates = NULL;
    context->generation_policy_ids = NULL;
    context->generation_sequences = NULL;
    context->generation_history = NULL;
    vx_full_prefix_state_free(context->generation_full_prefix);
    context->generation_full_prefix = NULL;
}

static VxStatus vx_generation_config_validate(const VxExecutionConfig* config,
    const cJSON* generation, size_t* id_count, size_t* history_capacity) {
    *id_count = 0; *history_capacity = 0;
    if (config->sampling != VX_SAMPLING_GREEDY && config->sampling != VX_SAMPLING_RANDOM)
        return VX_STATUS_INVALID_ARGUMENT;
    if (!config->max_steps || (config->sampling == VX_SAMPLING_GREEDY &&
        (config->has_seed || config->has_temperature || config->has_top_k || config->has_top_p)))
        return VX_STATUS_INVALID_ARGUMENT;
    if ((config->has_temperature && (!isfinite(config->temperature) || config->temperature <= 0)) ||
        (config->has_top_p && (!isfinite(config->top_p) || config->top_p <= 0 || config->top_p > 1)) ||
        config->stop_id_count > 4096 || config->eos_id_count > 4096 || config->stop_sequence_count > 128 ||
        (config->stop_id_count && !config->stop_ids) || (config->eos_id_count && !config->eos_ids) ||
        (config->stop_sequence_count && !config->stop_sequences)) return VX_STATUS_INVALID_ARGUMENT;
    const cJSON* domain = cJSON_GetObjectItemCaseSensitive(generation, "id_domain_size");
    if (!domain || domain->valueint <= 0 || (config->has_top_k && config->top_k > (uint32_t)domain->valueint))
        return VX_STATUS_INVALID_ARGUMENT;
    if (config->sampling == VX_SAMPLING_RANDOM && domain->valueint > 1048576)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    *id_count = config->stop_id_count + config->eos_id_count;
    for (size_t i = 0; i < config->stop_sequence_count; i++) {
        const VxGenerationIdSequence* sequence = &config->stop_sequences[i];
        if (!sequence->count || !sequence->ids || sequence->count > 4096 ||
            *id_count > 4096 - sequence->count) return VX_STATUS_INVALID_ARGUMENT;
        *id_count += sequence->count;
        if (sequence->count > *history_capacity) *history_capacity = sequence->count;
    }
    if (*id_count > 4096) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < config->stop_id_count; i++)
        if (config->stop_ids[i] < 0 || config->stop_ids[i] >= domain->valueint) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < config->eos_id_count; i++)
        if (config->eos_ids[i] < 0 || config->eos_ids[i] >= domain->valueint) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t i = 0; i < config->stop_sequence_count; i++)
        for (size_t j = 0; j < config->stop_sequences[i].count; j++)
            if (config->stop_sequences[i].ids[j] < 0 || config->stop_sequences[i].ids[j] >= domain->valueint)
                return VX_STATUS_INVALID_ARGUMENT;
    return VX_STATUS_OK;
}

static VxStatus vx_generation_policy_initialize(VxExecutionContext* context,
    const VxExecutionConfig* config, const cJSON* generation, size_t id_count, size_t history_capacity) {
    const cJSON* domain = cJSON_GetObjectItemCaseSensitive(generation, "id_domain_size");
    context->generation_sampling = config->sampling;
    context->generation_rng = config->has_seed ? config->seed : 0;
    context->generation_temperature = config->has_temperature ? config->temperature : 1;
    context->generation_top_p = config->has_top_p ? config->top_p : 1;
    context->generation_top_k = config->has_top_k ? config->top_k : 0;
    context->generation_vocabulary = (size_t)domain->valueint;
    context->generation_stop_id_count = config->stop_id_count;
    context->generation_eos_id_count = config->eos_id_count;
    context->generation_sequence_count = config->stop_sequence_count;
    context->generation_history_capacity = history_capacity;
    uint64_t bytes = id_count * sizeof(int32_t) + history_capacity * sizeof(int32_t) +
        config->stop_sequence_count * sizeof(VxGenerationSequence);
    int device_sampling = config->sampling == VX_SAMPLING_RANDOM &&
        context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA;
    if (config->sampling == VX_SAMPLING_RANDOM) bytes += context->generation_vocabulary * sizeof(VxGenerationCandidate);
    if (device_sampling)
        bytes += context->generation_vocabulary * sizeof(VxGenerationCandidate) + sizeof(int32_t);
    if (bytes > UINT64_MAX - context->allocated_bytes) return VX_STATUS_OUT_OF_MEMORY;
    if (context->execution_memory_limit) {
        uint64_t bound;
        if (!vx_execution_memory_bound(context, &bound) || bytes > UINT64_MAX - bound ||
            bound + bytes > context->execution_memory_limit) return VX_STATUS_OUT_OF_MEMORY;
    }
    if (id_count) context->generation_policy_ids = malloc(id_count * sizeof(int32_t));
    if (config->stop_sequence_count)
        context->generation_sequences = calloc(config->stop_sequence_count, sizeof(VxGenerationSequence));
    if (history_capacity) context->generation_history = calloc(history_capacity, sizeof(int32_t));
    /* CUDA reserves both sort buffers on DEVICE. CPU/WASM retain their one
     * HOST buffer. Both choices fit the same admission bound as before. */
    if (config->sampling == VX_SAMPLING_RANDOM && !device_sampling) {
        if (context->generation_vocabulary > SIZE_MAX / sizeof(VxGenerationCandidate)) return VX_STATUS_OUT_OF_MEMORY;
        context->generation_candidates = malloc(context->generation_vocabulary * sizeof(VxGenerationCandidate));
    }
    if ((id_count && !context->generation_policy_ids) ||
        (config->stop_sequence_count && !context->generation_sequences) ||
        (history_capacity && !context->generation_history) ||
        (config->sampling == VX_SAMPLING_RANDOM && !device_sampling &&
            !context->generation_candidates)) return VX_STATUS_OUT_OF_MEMORY;
    size_t offset = 0;
    if (config->stop_id_count) {
        memcpy(context->generation_policy_ids, config->stop_ids, config->stop_id_count * sizeof(int32_t));
        offset += config->stop_id_count;
    }
    if (config->eos_id_count) {
        memcpy(context->generation_policy_ids + offset, config->eos_ids, config->eos_id_count * sizeof(int32_t));
        offset += config->eos_id_count;
    }
    for (size_t i = 0; i < config->stop_sequence_count; i++) {
        size_t count = config->stop_sequences[i].count;
        context->generation_sequences[i] = (VxGenerationSequence){offset, count};
        memcpy(context->generation_policy_ids + offset, config->stop_sequences[i].ids, count * sizeof(int32_t));
        offset += count;
    }
#if VOLVOXAI_ENABLE_CUDA
    if (config->sampling == VX_SAMPLING_RANDOM && context->compiled->builtin_backend == VX_BACKEND_KIND_CUDA) {
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        int ready = cuda_generation_prepare((uint32_t)context->generation_vocabulary);
        vx_engine_state_scope_leave(scope);
        if (!ready) return VX_STATUS_OUT_OF_MEMORY;
    }
#endif
    context->allocated_bytes += bytes;
    return VX_STATUS_OK;
}

/* SplitMix64 uses unsigned arithmetic with a defined wraparound. Exactly one
 * draw is consumed per selected ID, including a one-candidate distribution. */
static double vx_generation_uniform(VxExecutionContext* context) {
    uint64_t value = (context->generation_rng += UINT64_C(0x9e3779b97f4a7c15));
    value = (value ^ (value >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    value = (value ^ (value >> 27)) * UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31;
    return (double)(value >> 11) * 0x1p-53;
}

static int vx_generation_worse(VxGenerationCandidate a, VxGenerationCandidate b) {
    return a.score < b.score || (a.score == b.score && a.id > b.id);
}
static void vx_generation_sift(VxGenerationCandidate* items, size_t root, size_t count) {
    while (root < count / 2) {
        size_t child = root * 2 + 1;
        if (child + 1 < count && vx_generation_worse(items[child + 1], items[child])) child++;
        if (!vx_generation_worse(items[child], items[root])) return;
        VxGenerationCandidate swap = items[root]; items[root] = items[child]; items[child] = swap;
        root = child;
    }
}

static int vx_generation_sample_cpu(VxExecutionContext* context, const T* scores, int row, int32_t* id) {
    if (!scores || scores->dtype != T_F32 || scores->ndim != 3 || scores->shape[0] != 1 ||
        scores->shape[2] != (int)context->generation_vocabulary || row < 0 || row >= scores->shape[1] || !scores->data)
        return -1;
    size_t count = context->generation_vocabulary;
    const float* values = (const float*)scores->data + (size_t)row * count;
    VxGenerationCandidate* items = context->generation_candidates;
    for (size_t i = 0; i < count; i++) {
        if (isnan(values[i])) return -1;
        items[i] = (VxGenerationCandidate){values[i], (int32_t)i};
    }
    for (size_t root = count / 2; root > 0; root--) vx_generation_sift(items, root - 1, count);
    for (size_t end = count; end > 1; end--) {
        VxGenerationCandidate swap = items[0]; items[0] = items[end - 1]; items[end - 1] = swap;
        vx_generation_sift(items, 0, end - 1);
    }
    if (context->generation_top_k && count > context->generation_top_k) count = context->generation_top_k;
    double maximum = items[0].score;
    if (maximum == -INFINITY) return -1;
    double total = 0;
    for (size_t i = 0; i < count; i++) {
        items[i].score = maximum == INFINITY ? (items[i].score == INFINITY ? 1 : 0)
            : exp((items[i].score - maximum) / context->generation_temperature);
        total += items[i].score;
    }
    if (!isfinite(total) || total <= 0) return -1;
    double retained = 0, threshold = context->generation_top_p * total;
    size_t admitted = 0;
    do { retained += items[admitted++].score; } while (admitted < count && retained < threshold);
    double target = vx_generation_uniform(context) * retained, cumulative = 0;
    *id = items[admitted - 1].id;
    for (size_t i = 0; i < admitted; i++) {
        cumulative += items[i].score;
        if (target < cumulative) { *id = items[i].id; break; }
    }
    return 0;
}

static void vx_generation_observe_id(VxExecutionContext* context, int32_t id) {
    if (context->generation_history_capacity)
        context->generation_history[context->generation_selected % context->generation_history_capacity] = id;
    context->generation_selected++;
    const int32_t* ids = context->generation_policy_ids;
    for (size_t i = 0; i < context->generation_eos_id_count; i++)
        if (ids[context->generation_stop_id_count + i] == id) {
            context->generation_stop_reason = VX_EXECUTION_STOP_EOS; return;
        }
    for (size_t i = 0; i < context->generation_stop_id_count; i++)
        if (ids[i] == id) { context->generation_stop_reason = VX_EXECUTION_STOP_ID_STOP; return; }
    for (size_t i = 0; i < context->generation_sequence_count; i++) {
        VxGenerationSequence sequence = context->generation_sequences[i];
        if (context->generation_selected < sequence.count) continue;
        int matched = 1;
        for (size_t j = 0; j < sequence.count; j++)
            if (context->generation_history[(context->generation_selected - sequence.count + j) %
                    context->generation_history_capacity] != ids[sequence.offset + j]) { matched = 0; break; }
        if (matched) { context->generation_stop_reason = VX_EXECUTION_STOP_ID_STOP; return; }
    }
}

static int vx_generation_select_id(VxExecutionContext* context, int row, int32_t* selected) {
    if (!context->execution_contract_enabled || !context->execution_autoregressive) return 0;
    const cJSON* generation = cJSON_GetObjectItemCaseSensitive(vx_execution_document(context->compiled->model), "generation");
    T* output = t_find(vx_execution_string(generation, "token_output"));
    T* scores = t_find(vx_execution_string(generation, "logits_output"));
    if (!output || output->dtype != T_I32 || output->ndim != 2 || output->shape[0] != 1 ||
        row < 0 || row >= output->shape[1] || !output->data) return -1;
    int32_t id;
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda) {
        if (context->generation_sampling == VX_SAMPLING_RANDOM) {
            if (!scores || scores->dtype != T_F32 || scores->ndim != 3 || scores->shape[0] != 1 ||
                scores->shape[1] != output->shape[1] || scores->shape[2] != (int)context->generation_vocabulary ||
                !scores->data || !cuda_generation_sample_f32(scores->data, (int32_t*)output->data,
                    (uint32_t)output->shape[1], (uint32_t)context->generation_vocabulary, (uint32_t)row,
                    context->generation_temperature, context->generation_top_k, context->generation_top_p,
                    vx_generation_uniform(context), &id)) return -1;
        } else if (!cuda_generation_read_id((const int32_t*)output->data, (uint32_t)output->shape[1], (uint32_t)row, &id)) return -1;
    } else
#endif
#if VOLVOXAI_ENABLE_VULKAN
    if (g_use_vulkan) {
        if (context->generation_sampling != VX_SAMPLING_GREEDY ||
            !vk_generation_read_id((const int32_t*)output->data, (uint32_t)output->shape[1], (uint32_t)row, &id)) return -1;
    } else
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (g_use_opengl) {
        if (context->generation_sampling != VX_SAMPLING_GREEDY ||
            !opengl_generation_read_id((const int32_t*)output->data, (uint32_t)output->shape[1], (uint32_t)row, &id)) return -1;
    } else
#endif
#if VOLVOXAI_ENABLE_METAL
    if (g_use_metal) {
        if (context->generation_sampling != VX_SAMPLING_GREEDY ||
            !metal_generation_read_id((const int32_t*)output->data, (uint32_t)output->shape[1], (uint32_t)row, &id)) return -1;
    } else
#endif
    if (context->generation_sampling == VX_SAMPLING_RANDOM) {
        if (vx_generation_sample_cpu(context, scores, row, &id)) return -1;
        ((int32_t*)output->data)[row] = id;
    } else id = ((const int32_t*)output->data)[row];
    if (id < 0 || id >= (int32_t)context->generation_vocabulary) return -1;
    if (selected) *selected = id;
    vx_generation_observe_id(context, id);
    return 0;
}

static int vx_generation_select_row(VxExecutionContext* context, int row) {
    if (context->generation_sampling != VX_SAMPLING_RANDOM && !context->generation_policy_ids) return 0;
    return vx_generation_select_id(context, row, NULL);
}
