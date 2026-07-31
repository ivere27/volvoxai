static int cuda_training_attention_mask_bytes(
        uint32_t mode, uint32_t batch, uint32_t seq_q, uint32_t seq_kv,
        size_t no_mask_bytes, size_t* expected_bytes) {
    uint32_t count;
    uint32_t factors[3];
    int factor_count;
    if (!expected_bytes || mode > 4u) return 0;
    if (!mode) {
        *expected_bytes = no_mask_bytes;
        return no_mask_bytes != 0;
    }
    if (mode == 1u) {
        factors[0] = seq_kv;
        factor_count = 1;
    } else if (mode == 2u) {
        factors[0] = batch;
        factors[1] = seq_kv;
        factor_count = 2;
    } else if (mode == 3u) {
        factors[0] = seq_q;
        factors[1] = seq_kv;
        factor_count = 2;
    } else {
        factors[0] = batch;
        factors[1] = seq_q;
        factors[2] = seq_kv;
        factor_count = 3;
    }
    return cuda_training_product_bytes(factors, factor_count, sizeof(int32_t),
                                       &count, expected_bytes);
}

/* P and dS are CUDA-private staging matrices, not graph tensors. One combined
 * allocation is reused in stream order across training forward/backward
 * attention launches, so it never incurs host transfers or consumes a graph
 * slot. Grow transactionally: the previous allocation remains authoritative
 * until its replacement has been allocated and all queued users have
 * completed. */
static int cuda_training_attention_workspace_reserve(
        size_t probability_bytes, CUdeviceptr* probabilities,
        CUdeviceptr* dscore) {
    size_t required_bytes;
    CUdeviceptr next = 0;
    CudaContextGuard guard;
    int ok = 0;

    if (!probability_bytes || !probabilities || !dscore ||
        probability_bytes > SIZE_MAX / 2u ||
        !p_cuMemAlloc || !p_cuMemFree) return 0;
    required_bytes = probability_bytes * 2u;
    if (!cuda_training_attention_workspace ||
        cuda_training_attention_workspace_bytes < required_bytes) {
        if (!cuda_context_enter(&guard)) return 0;
        if (!cuda_ok(p_cuMemAlloc(&next, required_bytes),
                     "cuMemAlloc(CUDA training attention workspace)"))
            goto done;
        if (cuda_training_attention_workspace && cuda_stream &&
            !cuda_ok(p_cuStreamSynchronize(cuda_stream),
                     "cuStreamSynchronize(CUDA training attention workspace)"))
            goto done;
        if (cuda_training_attention_workspace &&
            !cuda_ok(p_cuMemFree(cuda_training_attention_workspace),
                     "cuMemFree(CUDA training attention workspace)"))
            goto done;
        cuda_training_attention_workspace = next;
        cuda_training_attention_workspace_bytes = required_bytes;
        next = 0;
        ok = 1;
done:
        if (next) (void)p_cuMemFree(next);
        if (!cuda_context_leave(&guard)) ok = 0;
        if (!ok) return 0;
    }
    if (cuda_training_attention_workspace >
        UINTPTR_MAX - (CUdeviceptr)probability_bytes) return 0;
    *probabilities = cuda_training_attention_workspace;
    *dscore = cuda_training_attention_workspace +
        (CUdeviceptr)probability_bytes;
    return 1;
}

static int cuda_training_dispatch_sdpa(
        void* const* hosts, const size_t* bytes,
        const unsigned char* access, const unsigned char* is_weight,
        uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[12];
    uint32_t qkv_elements;
    uint32_t output_elements;
    uint32_t probability_elements;
    uint32_t qkv_factors[4];
    uint32_t output_factors[3];
    uint32_t probability_factors[4];
    size_t qkv_bytes;
    size_t output_bytes;
    size_t probability_bytes;
    size_t mask_bytes;
    float scale;
    float dropout_scale;
    CudaTensorView views[5];
    CUdeviceptr devices[5];
    CUdeviceptr probabilities;
    CUdeviceptr dscore;
    void* probability_args[4];
    void* dscore_args[5];
    void* backward_args[7];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ_WRITE ||
        access[4] != CUDA_TRAINING_READ || is_weight[4] != 0u) return -1;
    for (int binding = 0; binding < 4; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[4], sizeof(params));
    memcpy(&scale, &params[5], sizeof(scale));
    memcpy(&dropout_scale, &params[11], sizeof(dropout_scale));
    if (!params[0] || !params[1] || !params[2] || !params[3] || !params[4] ||
        params[3] > 64u || (uint64_t)params[2] * params[3] != params[1] ||
        !isfinite(scale) || params[6] > 1u || params[7] > 4u ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f) return -1;
    qkv_factors[0] = params[4];
    qkv_factors[1] = params[0];
    qkv_factors[2] = params[1];
    qkv_factors[3] = 3u;
    output_factors[0] = params[4];
    output_factors[1] = params[0];
    output_factors[2] = params[1];
    probability_factors[0] = params[4];
    probability_factors[1] = params[2];
    probability_factors[2] = params[0];
    probability_factors[3] = params[0];
    if (!cuda_training_product_bytes(qkv_factors, 4, sizeof(float),
                                     &qkv_elements, &qkv_bytes) ||
        !cuda_training_product_bytes(output_factors, 3, sizeof(float),
                                     &output_elements, &output_bytes) ||
        !cuda_training_product_bytes(probability_factors, 4, sizeof(float),
                                     &probability_elements,
                                     &probability_bytes) ||
        !cuda_training_attention_mask_bytes(params[7], params[4], params[0],
                                             params[0], qkv_bytes,
                                             &mask_bytes) ||
        bytes[0] != qkv_bytes || bytes[1] != mask_bytes ||
        bytes[2] != output_bytes || bytes[3] != qkv_bytes ||
        groups_x != params[0] || groups_y != params[2] ||
        params[4] > UINT32_MAX / 3u || groups_z != params[4] * 3u)
        return -1;
    (void)qkv_elements;
    (void)output_elements;
    (void)probability_elements;
    if (!cuda_training_attention_workspace_reserve(
            probability_bytes, &probabilities, &dscore)) return -1;
    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < 5; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     binding < 4 ? is_weight[binding] : 0,
                                     &views[binding])) return -1;
        devices[binding] = views[binding].device;
    }
    probability_args[0] = &devices[0];
    probability_args[1] = &devices[1];
    probability_args[2] = &probabilities;
    probability_args[3] = &devices[4];
    dscore_args[0] = &devices[0];
    dscore_args[1] = &devices[2];
    dscore_args[2] = &probabilities;
    dscore_args[3] = &dscore;
    dscore_args[4] = &devices[4];
    backward_args[0] = &devices[0];
    backward_args[1] = &devices[1];
    backward_args[2] = &devices[2];
    backward_args[3] = &devices[3];
    backward_args[4] = &probabilities;
    backward_args[5] = &dscore;
    backward_args[6] = &devices[4];
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sdpa_probabilities),
                     params[0], params[2], params[4],
                     64u, 1u, 1u, 0u, probability_args) ||
        !cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sdpa_dscore),
                     params[0], params[2], params[4],
                     64u, 1u, 1u, 0u, dscore_args)) return -1;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sdpa_backward), groups_x, groups_y, groups_z,
                     64u, 1u, 1u, 0u, backward_args)) return -1;
    graph_mark_device(views[3].slot);
    return 0;
}

static int cuda_training_dispatch_cross_sdpa(
        CudaTrainingCommand command, void* const* hosts,
        const size_t* bytes, const unsigned char* access,
        const unsigned char* is_weight, uint32_t groups_x,
        uint32_t groups_y, uint32_t groups_z) {
    uint32_t params[16];
    uint32_t q_elements;
    uint32_t kv_elements;
    uint32_t probability_elements;
    uint32_t q_factors[3];
    uint32_t kv_factors[3];
    uint32_t probability_factors[4];
    size_t q_bytes;
    size_t kv_bytes;
    size_t probability_bytes;
    size_t mask_bytes;
    float scale;
    float dropout_scale;
    CudaTensorView views[9];
    CUdeviceptr devices[9];
    CUfunction function;
    uint32_t expected_x;
    int destination;
    CUdeviceptr probabilities;
    CUdeviceptr dscore;
    void* probability_args[5];
    void* dscore_args[5];
    void* backward_args[11];

    if (!hosts[0] || !hosts[1] || !hosts[2] || !hosts[3] || !hosts[4] ||
        !hosts[5] || !hosts[6] || !hosts[7] || !hosts[8] ||
        access[0] != CUDA_TRAINING_READ ||
        access[1] != CUDA_TRAINING_READ ||
        access[2] != CUDA_TRAINING_READ ||
        access[3] != CUDA_TRAINING_READ ||
        access[4] != CUDA_TRAINING_READ ||
        access[5] != CUDA_TRAINING_READ_WRITE ||
        access[6] != CUDA_TRAINING_READ_WRITE ||
        access[7] != CUDA_TRAINING_READ_WRITE ||
        access[8] != CUDA_TRAINING_READ || is_weight[8] != 0u) return -1;
    for (int binding = 0; binding < 8; binding++)
        if (is_weight[binding] > 1u) return -1;
    memcpy(params, hosts[8], sizeof(params));
    memcpy(&scale, &params[6], sizeof(scale));
    memcpy(&dropout_scale, &params[12], sizeof(dropout_scale));
    if (!params[0] || !params[1] || !params[2] || !params[3] || !params[4] ||
        !params[5] || params[4] > 64u ||
        (uint64_t)params[3] * params[4] != params[2] ||
        !isfinite(scale) || params[7] > 1u || params[8] > 4u ||
        !isfinite(dropout_scale) || dropout_scale < 1.0f ||
        params[13] || params[14] || params[15]) return -1;
    q_factors[0] = params[5];
    q_factors[1] = params[0];
    q_factors[2] = params[2];
    kv_factors[0] = params[5];
    kv_factors[1] = params[1];
    kv_factors[2] = params[2];
    probability_factors[0] = params[5];
    probability_factors[1] = params[3];
    probability_factors[2] = params[0];
    probability_factors[3] = params[1];
    if (!cuda_training_product_bytes(q_factors, 3, sizeof(float),
                                     &q_elements, &q_bytes) ||
        !cuda_training_product_bytes(kv_factors, 3, sizeof(float),
                                     &kv_elements, &kv_bytes) ||
        !cuda_training_product_bytes(probability_factors, 4, sizeof(float),
                                     &probability_elements,
                                     &probability_bytes) ||
        !cuda_training_attention_mask_bytes(params[8], params[5], params[0],
                                             params[1], q_bytes,
                                             &mask_bytes) ||
        bytes[0] != q_bytes || bytes[1] != kv_bytes || bytes[2] != kv_bytes ||
        bytes[3] != mask_bytes || bytes[4] != q_bytes ||
        bytes[5] != q_bytes || bytes[6] != kv_bytes || bytes[7] != kv_bytes ||
        groups_y != params[3] || groups_z != params[5]) return -1;
    (void)q_elements;
    (void)kv_elements;
    (void)probability_elements;
    if (!cuda_training_attention_workspace_reserve(
            probability_bytes, &probabilities, &dscore)) return -1;
    if (command == CUDA_TRAINING_COMMAND_CROSS_SDPA_Q_BACKWARD) {
        expected_x = params[0];
        destination = 5;
        function = CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_backward_q);
    } else if (command == CUDA_TRAINING_COMMAND_CROSS_SDPA_K_BACKWARD) {
        expected_x = params[1];
        destination = 6;
        function = CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_backward_k);
    } else {
        expected_x = params[1];
        destination = 7;
        function = CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_backward_v);
    }
    if (groups_x != expected_x) return -1;
    memset(views, 0, sizeof(views));
    memset(devices, 0, sizeof(devices));
    for (int binding = 0; binding < 5; binding++) {
        if (!graph_ensure_input_view(hosts[binding], bytes[binding],
                                     is_weight[binding], &views[binding]))
            return -1;
        devices[binding] = views[binding].device;
    }
    if (!graph_ensure_input_view(hosts[destination], bytes[destination],
                                 is_weight[destination], &views[destination]) ||
        !graph_ensure_input_view(hosts[8], bytes[8], 0, &views[8])) return -1;
    devices[destination] = views[destination].device;
    devices[8] = views[8].device;
    probability_args[0] = &devices[0];
    probability_args[1] = &devices[1];
    probability_args[2] = &devices[3];
    probability_args[3] = &probabilities;
    probability_args[4] = &devices[8];
    dscore_args[0] = &devices[2];
    dscore_args[1] = &devices[4];
    dscore_args[2] = &probabilities;
    dscore_args[3] = &dscore;
    dscore_args[4] = &devices[8];
    for (int binding = 0; binding < 8; binding++)
        backward_args[binding] = &devices[binding];
    backward_args[8] = &probabilities;
    backward_args[9] = &dscore;
    backward_args[10] = &devices[8];
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_probabilities),
                     params[0], params[3], params[5],
                     64u, 1u, 1u, 0u, probability_args) ||
        !cuda_launch(CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_dscore),
                     params[0], params[3], params[5],
                     64u, 1u, 1u, 0u, dscore_args)) return -1;
    if (!cuda_launch(function, groups_x, groups_y, groups_z,
                     64u, 1u, 1u, 0u, backward_args)) return -1;
    graph_mark_device(views[destination].slot);
    return 0;
}

int cuda_graph_sdpa_training_f32(
        const float* qkv, const int32_t* mask, long mask_elements,
        float* output, int sequence, int d_model, int heads, int head_dim,
        int batch, float attention_scale, int causal, int mask_mode,
        uint32_t threshold, uint32_t seed, uint32_t counter,
        float dropout_scale) {
    uint32_t params[12] = {0};
    uint32_t qkv_count;
    uint32_t output_count;
    uint32_t probability_count;
    uint32_t factors[4];
    size_t qkv_bytes;
    size_t output_bytes;
    size_t probability_bytes;
    size_t expected_mask_bytes = 0;
    CudaTensorView qkv_view;
    CudaTensorView mask_view;
    CudaTensorView output_view;
    CudaTensorView params_view;
    CUdeviceptr mask_device = 0;
    CUdeviceptr probabilities;
    CUdeviceptr dscore;
    void* probability_args[4];
    void* forward_args[5];

    if (!cuda_ready || !cuda_training_module ||
        !CUDA_TRAINING_FUNCTION(k_training_sdpa_probabilities) || !CUDA_TRAINING_FUNCTION(k_training_sdpa_forward) ||
        !qkv || !output || sequence <= 0 || d_model <= 0 || heads <= 0 ||
        head_dim <= 0 || head_dim > 64 || batch <= 0 ||
        (uint64_t)(uint32_t)heads * (uint32_t)head_dim != (uint32_t)d_model ||
        !isfinite(attention_scale) || (causal != 0 && causal != 1) ||
        mask_mode < 0 || mask_mode > 4 || !isfinite(dropout_scale) ||
        dropout_scale < 1.0f) return 0;
    factors[0] = (uint32_t)batch;
    factors[1] = (uint32_t)sequence;
    factors[2] = (uint32_t)d_model;
    factors[3] = 3u;
    if (!cuda_training_product_bytes(factors, 4, sizeof(float), &qkv_count,
                                     &qkv_bytes)) return 0;
    if (!cuda_training_product_bytes(factors, 3, sizeof(float), &output_count,
                                     &output_bytes)) return 0;
    factors[0] = (uint32_t)batch;
    factors[1] = (uint32_t)heads;
    factors[2] = (uint32_t)sequence;
    factors[3] = (uint32_t)sequence;
    if (!cuda_training_product_bytes(factors, 4, sizeof(float),
                                     &probability_count,
                                     &probability_bytes)) return 0;
    if (!mask_mode) {
        if (mask || mask_elements != 0) return 0;
    } else {
        if (!mask || mask_elements <= 0 ||
            !cuda_training_attention_mask_bytes(
                (uint32_t)mask_mode, (uint32_t)batch, (uint32_t)sequence,
                (uint32_t)sequence, 0, &expected_mask_bytes) ||
            (uint64_t)mask_elements * sizeof(int32_t) != expected_mask_bytes)
            return 0;
    }
    params[0] = (uint32_t)sequence;
    params[1] = (uint32_t)d_model;
    params[2] = (uint32_t)heads;
    params[3] = (uint32_t)head_dim;
    params[4] = (uint32_t)batch;
    memcpy(&params[5], &attention_scale, sizeof(attention_scale));
    params[6] = (uint32_t)causal;
    params[7] = (uint32_t)mask_mode;
    params[8] = threshold;
    params[9] = seed;
    params[10] = counter;
    memcpy(&params[11], &dropout_scale, sizeof(dropout_scale));
    if (!cuda_training_attention_workspace_reserve(
            probability_bytes, &probabilities, &dscore)) return 0;
    cuda_graph_mark_host(params, sizeof(params), 0);
    if (!graph_ensure_input_view(qkv, qkv_bytes, 0, &qkv_view) ||
        !graph_output_view(output, output_bytes, &output_view) ||
        !graph_ensure_input_view(params, sizeof(params), 0, &params_view))
        return 0;
    if (mask_mode) {
        if (!graph_ensure_input_view(mask, expected_mask_bytes, 0, &mask_view))
            return 0;
        mask_device = mask_view.device;
    }
    probability_args[0] = &qkv_view.device;
    probability_args[1] = &mask_device;
    probability_args[2] = &probabilities;
    probability_args[3] = &params_view.device;
    forward_args[0] = &qkv_view.device;
    forward_args[1] = &mask_device;
    forward_args[2] = &output_view.device;
    forward_args[3] = &probabilities;
    forward_args[4] = &params_view.device;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sdpa_probabilities),
                     (uint32_t)sequence, (uint32_t)heads,
                     (uint32_t)batch, 64u, 1u, 1u, 0u,
                     probability_args)) return 0;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_sdpa_forward), (uint32_t)sequence,
                     (uint32_t)heads, (uint32_t)batch,
                     64u, 1u, 1u, 0u, forward_args)) return 0;
    graph_mark_device(output_view.slot);
    (void)qkv_count;
    (void)output_count;
    (void)probability_count;
    (void)dscore;
    return 1;
}


int cuda_graph_cross_sdpa_training_f32(
        const float* query, const float* key, const float* value,
        const int32_t* mask, long mask_elements, float* output,
        int query_sequence, int key_sequence, int d_model, int heads,
        int head_dim, int batch, float attention_scale, int causal,
        int mask_mode, uint32_t threshold, uint32_t seed, uint32_t counter,
        float dropout_scale) {
    uint32_t params[16] = {0};
    uint32_t query_count;
    uint32_t key_count;
    uint32_t probability_count;
    uint32_t factors[4];
    size_t query_bytes;
    size_t key_bytes;
    size_t probability_bytes;
    size_t expected_mask_bytes = 0;
    CudaTensorView query_view;
    CudaTensorView key_view;
    CudaTensorView value_view;
    CudaTensorView mask_view;
    CudaTensorView output_view;
    CudaTensorView params_view;
    CUdeviceptr mask_device = 0;
    CUdeviceptr probabilities;
    CUdeviceptr dscore;
    void* probability_args[5];
    void* forward_args[7];

    if (!cuda_ready || !cuda_training_module ||
        !CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_probabilities) ||
        !CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_forward) || !query || !key || !value ||
        !output || query_sequence <= 0 || key_sequence <= 0 || d_model <= 0 ||
        heads <= 0 || head_dim <= 0 || head_dim > 64 || batch <= 0 ||
        (uint64_t)(uint32_t)heads * (uint32_t)head_dim != (uint32_t)d_model ||
        !isfinite(attention_scale) || (causal != 0 && causal != 1) ||
        mask_mode < 0 || mask_mode > 4 || !isfinite(dropout_scale) ||
        dropout_scale < 1.0f) return 0;
    factors[0] = (uint32_t)batch;
    factors[1] = (uint32_t)query_sequence;
    factors[2] = (uint32_t)d_model;
    if (!cuda_training_product_bytes(factors, 3, sizeof(float), &query_count,
                                     &query_bytes)) return 0;
    factors[1] = (uint32_t)key_sequence;
    if (!cuda_training_product_bytes(factors, 3, sizeof(float), &key_count,
                                     &key_bytes)) return 0;
    factors[0] = (uint32_t)batch;
    factors[1] = (uint32_t)heads;
    factors[2] = (uint32_t)query_sequence;
    factors[3] = (uint32_t)key_sequence;
    if (!cuda_training_product_bytes(factors, 4, sizeof(float),
                                     &probability_count,
                                     &probability_bytes)) return 0;
    if (!mask_mode) {
        if (mask || mask_elements != 0) return 0;
    } else {
        if (!mask || mask_elements <= 0 ||
            !cuda_training_attention_mask_bytes(
                (uint32_t)mask_mode, (uint32_t)batch,
                (uint32_t)query_sequence, (uint32_t)key_sequence, 0,
                &expected_mask_bytes) ||
            (uint64_t)mask_elements * sizeof(int32_t) != expected_mask_bytes)
            return 0;
    }
    params[0] = (uint32_t)query_sequence;
    params[1] = (uint32_t)key_sequence;
    params[2] = (uint32_t)d_model;
    params[3] = (uint32_t)heads;
    params[4] = (uint32_t)head_dim;
    params[5] = (uint32_t)batch;
    memcpy(&params[6], &attention_scale, sizeof(attention_scale));
    params[7] = (uint32_t)causal;
    params[8] = (uint32_t)mask_mode;
    params[9] = threshold;
    params[10] = seed;
    params[11] = counter;
    memcpy(&params[12], &dropout_scale, sizeof(dropout_scale));
    if (!cuda_training_attention_workspace_reserve(
            probability_bytes, &probabilities, &dscore)) return 0;
    cuda_graph_mark_host(params, sizeof(params), 0);
    if (!graph_ensure_input_view(query, query_bytes, 0, &query_view) ||
        !graph_ensure_input_view(key, key_bytes, 0, &key_view) ||
        !graph_ensure_input_view(value, key_bytes, 0, &value_view) ||
        !graph_output_view(output, query_bytes, &output_view) ||
        !graph_ensure_input_view(params, sizeof(params), 0, &params_view))
        return 0;
    if (mask_mode) {
        if (!graph_ensure_input_view(mask, expected_mask_bytes, 0, &mask_view))
            return 0;
        mask_device = mask_view.device;
    }
    probability_args[0] = &query_view.device;
    probability_args[1] = &key_view.device;
    probability_args[2] = &mask_device;
    probability_args[3] = &probabilities;
    probability_args[4] = &params_view.device;
    forward_args[0] = &query_view.device;
    forward_args[1] = &key_view.device;
    forward_args[2] = &value_view.device;
    forward_args[3] = &mask_device;
    forward_args[4] = &output_view.device;
    forward_args[5] = &probabilities;
    forward_args[6] = &params_view.device;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_probabilities),
                     (uint32_t)query_sequence, (uint32_t)heads,
                     (uint32_t)batch, 64u, 1u, 1u, 0u,
                     probability_args)) return 0;
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_training_cross_sdpa_forward),
                     (uint32_t)query_sequence, (uint32_t)heads,
                     (uint32_t)batch, 64u, 1u, 1u, 0u,
                     forward_args)) return 0;
    graph_mark_device(output_view.slot);
    (void)query_count;
    (void)key_count;
    (void)probability_count;
    (void)dscore;
    return 1;
}
