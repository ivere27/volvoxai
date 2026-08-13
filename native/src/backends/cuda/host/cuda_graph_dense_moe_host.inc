static int cuda_matmul_impl(const float* input, const float* weight,
                            const float* bias, float* output,
                            int rows, int d_in, int d_out,
                            int out_in, int refresh_host_input,
                            int sync_host_output) {
    CudaTensorView input_view;
    CudaTensorSlot* weight_slot;
    CudaTensorSlot* bias_slot = NULL;
    CudaTensorView output_view;
    uint32_t row_count;
    uint32_t input_width;
    uint32_t output_width;
    size_t input_bytes, weight_bytes, output_bytes;
    void* args[7];
    unsigned int gx, gy;
    if (!input || !weight || !output || rows <= 0 || d_in <= 0 || d_out <= 0 ||
        (out_in != 0 && out_in != 1) ||
        !size_product_u32((uint32_t)rows, (uint32_t)d_in, sizeof(float), &input_bytes) ||
        !size_product_u32((uint32_t)d_in, (uint32_t)d_out, sizeof(float), &weight_bytes) ||
        !size_product_u32((uint32_t)rows, (uint32_t)d_out, sizeof(float), &output_bytes)) return 0;
    row_count = (uint32_t)rows;
    input_width = (uint32_t)d_in;
    output_width = (uint32_t)d_out;
    if (refresh_host_input) {
        /* Standalone callers own host coherence and may pass an interior row
         * pointer.  Refresh its owning allocation on every invocation.  The
         * graph route deliberately skips this block so a preceding CUDA node
         * stays authoritative without an intermediate D2H/H2D round trip. */
        if (!graph_get_view(input, input_bytes, 0, &input_view)) return 0;
        if (input_view.slot) {
            input_view.slot->host_dirty = 1;
            input_view.slot->device_dirty = 0;
        }
    }
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view)) return 0;
    weight_slot = graph_ensure_input(weight, weight_bytes, 1);
    if (bias) bias_slot = graph_ensure_input(bias, (size_t)d_out * sizeof(float), 1);
    if (!graph_output_view(output, output_bytes, &output_view) || !weight_slot ||
        (bias && !bias_slot)) return 0;
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_view.device;
    args[1] = &weight_slot->device;
    args[2] = &bias_device;
    args[3] = &output_view.device;
    args[4] = &row_count;
    args[5] = &input_width;
    args[6] = &output_width;
    gx = (output_width + 15u) / 16u;
    gy = (row_count + 15u) / 16u;
    if (!gx || !gy || gy > 65535u ||
        !cuda_launch(out_in ? CUDA_FORWARD_FUNCTION(k_matmul_out_in) : CUDA_FORWARD_FUNCTION(k_matmul),
                     gx, gy, 1, 16, 16, 1, 0, args)) return 0;
    graph_mark_device(output_view.slot);
    return !sync_host_output || cuda_graph_sync_host(output, output_bytes, 0);
}

int cuda_matmul(const float* input, const float* weight, const float* bias,
                float* output, int rows, int d_in, int d_out) {
    return cuda_matmul_impl(input, weight, bias, output, rows, d_in, d_out,
                            0, 1, 1);
}

int cuda_graph_matmul_f32(const float* input, const float* weight,
                          const float* bias, float* output,
                          int rows, int d_in, int d_out) {
    return cuda_matmul_impl(input, weight, bias, output, rows, d_in, d_out,
                            0, 0, 0);
}

int cuda_graph_matmul_layout_f32(const float* input, const float* weight,
                                 const float* bias, float* output,
                                 int rows, int d_in, int d_out, int out_in) {
    return cuda_matmul_impl(input, weight, bias, output, rows, d_in, d_out,
                            out_in, 0, 0);
}

static int cuda_lora_workspace_reserve(size_t bytes) {
    CUdeviceptr next = 0;
    int ok = 0;
    CudaContextGuard guard;
    if (!bytes || !p_cuMemAlloc || !p_cuMemFree) return 0;
    if (cuda_lora_workspace && cuda_lora_workspace_bytes >= bytes) return 1;
    if (cuda_domain_enforced) return 0;
    if (!cuda_context_enter(&guard)) return 0;
    if (!cuda_ok(p_cuMemAlloc(&next, bytes),
                 "cuMemAlloc(CUDA LoRA workspace)")) {
        cuda_graph_allocation_failed = 1;
        goto done;
    }
    if (cuda_lora_workspace && cuda_stream &&
        !cuda_ok(p_cuStreamSynchronize(cuda_stream),
                 "cuStreamSynchronize(CUDA LoRA workspace)")) goto done;
    if (cuda_lora_workspace &&
        !cuda_ok(p_cuMemFree(cuda_lora_workspace),
                 "cuMemFree(CUDA LoRA workspace)")) goto done;
    cuda_lora_workspace = next;
    cuda_lora_workspace_bytes = bytes;
    next = 0;
    ok = 1;
done:
    if (next) (void)p_cuMemFree(next);
    if (!cuda_context_leave(&guard)) ok = 0;
    return ok;
}

int cuda_graph_lora_apply_f32(const float* input, const float* a,
                              const float* b, float* output,
                              int rows, int d_in, int rank, int d_out,
                              float scale) {
    CudaTensorView input_view, a_view, b_view, output_view;
    uint32_t row_count, input_width, rank_width, output_width;
    size_t input_bytes, a_bytes, b_bytes, output_bytes, workspace_bytes;
    unsigned int rank_gx, output_gx, gy;
    void* rank_args[6];
    void* apply_args[7];
    if (!input || !a || !b || !output || rows <= 0 || d_in <= 0 ||
        rank <= 0 || d_out <= 0 || !isfinite(scale) ||
        !size_product_u32((uint32_t)rows, (uint32_t)d_in,
                          sizeof(float), &input_bytes) ||
        !size_product_u32((uint32_t)d_in, (uint32_t)rank,
                          sizeof(float), &a_bytes) ||
        !size_product_u32((uint32_t)rank, (uint32_t)d_out,
                          sizeof(float), &b_bytes) ||
        !size_product_u32((uint32_t)rows, (uint32_t)d_out,
                          sizeof(float), &output_bytes) ||
        !size_product_u32((uint32_t)rows, (uint32_t)rank,
                          sizeof(float), &workspace_bytes) ||
        ranges_overlap(output, output_bytes, a, a_bytes) ||
        ranges_overlap(output, output_bytes, b, b_bytes)) return 0;
    if (scale == 0.0f) return 1;
    row_count = (uint32_t)rows;
    input_width = (uint32_t)d_in;
    rank_width = (uint32_t)rank;
    output_width = (uint32_t)d_out;

    /* Adapter versions can change between request-scoped routes while their
     * allocator reuses a host address. Mark A/B current at this boundary so
     * pointer reuse can never select stale device bytes. */
    cuda_graph_mark_host(a, a_bytes, 1);
    cuda_graph_mark_host(b, b_bytes, 1);
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view) ||
        !graph_ensure_input_view(a, a_bytes, 1, &a_view) ||
        !graph_ensure_input_view(b, b_bytes, 1, &b_view) ||
        !graph_output_view(output, output_bytes, &output_view) ||
        !cuda_lora_workspace_reserve(workspace_bytes)) return 0;

    rank_args[0] = &input_view.device;
    rank_args[1] = &a_view.device;
    rank_args[2] = &cuda_lora_workspace;
    rank_args[3] = &row_count;
    rank_args[4] = &input_width;
    rank_args[5] = &rank_width;
    rank_gx = (rank_width + 15u) / 16u;
    gy = (row_count + 15u) / 16u;
    if (!rank_gx || !gy || gy > 65535u ||
        !cuda_launch(CUDA_FORWARD_FUNCTION(k_lora_rank), rank_gx, gy, 1u,
                     16u, 16u, 1u, 0u, rank_args)) return 0;

    apply_args[0] = &cuda_lora_workspace;
    apply_args[1] = &b_view.device;
    apply_args[2] = &output_view.device;
    apply_args[3] = &row_count;
    apply_args[4] = &rank_width;
    apply_args[5] = &output_width;
    apply_args[6] = &scale;
    output_gx = (output_width + 15u) / 16u;
    if (!output_gx ||
        !cuda_launch(CUDA_FORWARD_FUNCTION(k_lora_apply), output_gx, gy, 1u,
                     16u, 16u, 1u, 0u, apply_args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_moe_router_f32(
        const float* input, const float* weight, const float* bias,
        float* route_indices, float* route_weights, int rows, int d_model,
        int experts, int top_k, float temperature, int normalize) {
    uint32_t row_count;
    uint32_t model_width;
    uint32_t expert_count;
    uint32_t route_width;
    uint32_t normalize_flag;
    uint32_t has_bias;
    uint32_t input_elements;
    uint32_t weight_elements;
    uint32_t route_elements;
    size_t input_bytes;
    size_t weight_bytes;
    size_t route_bytes;
    size_t bias_bytes;
    CudaTensorView input_view;
    CudaTensorView weight_view;
    CudaTensorView bias_view;
    CudaTensorView indices_view;
    CudaTensorView routes_view;
    CUdeviceptr bias_device = 0;
    void* args[12];
    if (!input || !weight || !route_indices || !route_weights || rows <= 0 ||
        d_model <= 0 || experts <= 0 || top_k <= 0 || top_k > experts ||
        top_k > 8 || !isfinite(temperature) || temperature <= 0.0f ||
        (normalize != 0 && normalize != 1)) return 0;
    row_count = (uint32_t)rows;
    model_width = (uint32_t)d_model;
    expert_count = (uint32_t)experts;
    route_width = (uint32_t)top_k;
    normalize_flag = (uint32_t)normalize;
    has_bias = bias ? 1u : 0u;
    if (!size_product_u32(row_count, model_width, sizeof(float),
                          &input_bytes) ||
        !size_product_u32(model_width, expert_count, sizeof(float),
                          &weight_bytes) ||
        !size_product_u32(row_count, route_width, sizeof(float),
                          &route_bytes) ||
        !size_product_u32(1u, expert_count, sizeof(float), &bias_bytes) ||
        ranges_overlap(route_indices, route_bytes, route_weights,
                       route_bytes)) return 0;
    input_elements = row_count * model_width;
    weight_elements = model_width * expert_count;
    route_elements = row_count * route_width;
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view) ||
        !graph_ensure_input_view(weight, weight_bytes, 1, &weight_view) ||
        !graph_output_view(route_indices, route_bytes, &indices_view) ||
        !graph_output_view(route_weights, route_bytes, &routes_view)) return 0;
    if (bias) {
        if (!graph_ensure_input_view(bias, bias_bytes, 1, &bias_view)) return 0;
        bias_device = bias_view.device;
    }
    args[0] = &input_view.device;
    args[1] = &weight_view.device;
    args[2] = &bias_device;
    args[3] = &indices_view.device;
    args[4] = &routes_view.device;
    args[5] = &row_count;
    args[6] = &model_width;
    args[7] = &expert_count;
    args[8] = &route_width;
    args[9] = &temperature;
    args[10] = &normalize_flag;
    args[11] = &has_bias;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_moe_router), row_count, args)) return 0;
    graph_mark_device(indices_view.slot);
    graph_mark_device(routes_view.slot);
    (void)input_elements;
    (void)weight_elements;
    (void)route_elements;
    return 1;
}


int cuda_graph_moe_linear_f32(
        const float* input, const float* expert_weight,
        const float* expert_bias, const float* route_indices,
        const float* route_weights, float* output, int rows, int d_in,
        int d_out, int experts, int top_k, const unsigned int* slot_rows,
        unsigned int slot_domain) {
    uint32_t row_count;
    uint32_t input_width;
    uint32_t output_width;
    uint32_t expert_count;
    uint32_t route_width;
    uint32_t has_bias;
    uint32_t input_elements;
    uint32_t weight_elements;
    uint32_t output_elements;
    uint32_t route_elements;
    size_t input_bytes;
    size_t weight_bytes;
    size_t output_bytes;
    size_t route_bytes;
    size_t bias_bytes;
    CudaTensorView input_view;
    CudaTensorView weight_view;
    CudaTensorView bias_view;
    CudaTensorView indices_view;
    CudaTensorView routes_view;
    CudaTensorView output_view;
    CudaTensorView slot_view;
    CUdeviceptr bias_device = 0;
    CUdeviceptr slot_device = 0;
    uint32_t slot_extent = slot_rows ? (uint32_t)slot_domain : 0u;
    size_t slot_bytes;
    void* args[14];
    if (!input || !expert_weight || !route_indices || !route_weights ||
        !output || rows <= 0 || d_in <= 0 || d_out <= 0 || experts <= 0 ||
        top_k <= 0 || top_k > 8) return 0;
    if (slot_rows ? (slot_extent < (uint32_t)experts ||
                     (uint32_t)top_k > slot_extent)
                  : top_k > experts) return 0;
    row_count = (uint32_t)rows;
    input_width = (uint32_t)d_in;
    output_width = (uint32_t)d_out;
    expert_count = (uint32_t)experts;
    route_width = (uint32_t)top_k;
    has_bias = expert_bias ? 1u : 0u;
    if (!size_product_u32(row_count, input_width, sizeof(float),
                          &input_bytes) ||
        !checked_product4_bytes(expert_count, input_width, output_width, 1u,
                                sizeof(float), &weight_elements,
                                &weight_bytes) ||
        !size_product_u32(row_count, output_width, sizeof(float),
                          &output_bytes) ||
        !size_product_u32(row_count, route_width, sizeof(float),
                          &route_bytes) ||
        !size_product_u32(expert_count, output_width, sizeof(float),
                          &bias_bytes) ||
        !size_product_u32(1u, slot_extent ? slot_extent : 1u,
                          sizeof(uint32_t), &slot_bytes)) return 0;
    input_elements = row_count * input_width;
    output_elements = row_count * output_width;
    route_elements = row_count * route_width;
    if (!graph_ensure_input_view(input, input_bytes, 0, &input_view) ||
        !graph_ensure_input_view(expert_weight, weight_bytes, 1,
                                 &weight_view) ||
        !graph_ensure_input_view(route_indices, route_bytes, 0,
                                 &indices_view) ||
        !graph_ensure_input_view(route_weights, route_bytes, 0,
                                 &routes_view) ||
        !graph_output_view(output, output_bytes, &output_view)) return 0;
    if (expert_bias) {
        if (!graph_ensure_input_view(expert_bias, bias_bytes, 1, &bias_view))
            return 0;
        bias_device = bias_view.device;
    }
    if (slot_extent) {
        /* The table is immutable for the life of a context residency, so it is
         * uploaded through the same invariant-input path as the weights. */
        if (!graph_ensure_input_view((const float*)slot_rows, slot_bytes, 1,
                                     &slot_view)) return 0;
        slot_device = slot_view.device;
    }
    args[0] = &input_view.device;
    args[1] = &weight_view.device;
    args[2] = &bias_device;
    args[3] = &indices_view.device;
    args[4] = &routes_view.device;
    args[5] = &output_view.device;
    args[6] = &row_count;
    args[7] = &input_width;
    args[8] = &output_width;
    args[9] = &expert_count;
    args[10] = &route_width;
    args[11] = &has_bias;
    args[12] = &slot_device;
    args[13] = &slot_extent;
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_moe_linear), output_elements, args)) return 0;
    graph_mark_device(output_view.slot);
    (void)input_elements;
    (void)weight_elements;
    (void)route_elements;
    return 1;
}
