/* Both row tactics bind five full ranges: input, a resident weight,
 * a second weight or residual, then two outputs. A tactic miss must precede coherence; after the
 * first owner read, a failure is terminal because an upload may be in flight. */
static int cuda_gemv_row_bind_owners(const void* const hosts[5],
        const size_t bytes[5], int weight_count, unsigned int grid,
        CudaTensorView views[5]) {
    for (int index = 0; index < 5; index++) {
        int owner = graph_find_containing_slot(hosts[index], bytes[index]);
        if (owner < 0) return 0;
        CudaTensorSlot* slot = &graph_slots[owner];
        size_t offset = (uintptr_t)hosts[index] - (uintptr_t)slot->host;
        int is_weight = index > 0 && index <= weight_count;
        if (!slot->device || (is_weight ? !graph_slot_is_resident_weight(slot) : slot->is_weight) ||
            slot->device > UINT64_MAX - offset ||
            bytes[index] > UINT64_MAX - (slot->device + offset)) return 0;
        views[index] = (CudaTensorView){slot, slot->device + offset};
    }
    for (int destination = 3; destination < 5; destination++)
        for (int source = 0; source < destination; source++)
            if (ranges_overlap(hosts[destination], bytes[destination], hosts[source], bytes[source]) ||
                (views[destination].device <= views[source].device
                    ? bytes[destination] > views[source].device - views[destination].device
                    : bytes[source] > views[destination].device - views[source].device)) return 0;
    if (!cuda_launch_dimensions_valid(grid, 1u, 1u, 128u, 1u, 1u, 0u)) return 0;
    /* Read both complete output owners before either selected row is written. */
    for (int index = 0; index < 5; index++)
        if (!graph_ensure_input_view(hosts[index], bytes[index],
                index > 0 && index <= weight_count, &views[index])) return -1;
    return 1;
}

static int cuda_matmul_impl(const float* input, const float* weight,
                            const float* bias, float* output,
                            int rows, int d_in, int d_out,
                            int out_in, int refresh_host_input,
                            int sync_host_output, const float* residual, float* final_output,
                            int residual_first, int sequence, int row) {
    CudaTensorView input_view;
    CudaTensorSlot* weight_slot;
    CudaTensorSlot* bias_slot = NULL;
    CudaTensorView output_view;
    uint32_t row_count;
    uint32_t input_width;
    uint32_t output_width;
    size_t input_bytes, weight_bytes, output_bytes;
    void* args[10];
    CudaTensorView residual_view = {0}, final_view = {0};
    int dual = residual || final_output;
    CUdeviceptr weight_device;
    unsigned int gx, gy;
    if (!input || !weight || !output || rows <= 0 || d_in <= 0 || d_out <= 0 ||
        (out_in != 0 && out_in != 1) ||
        !size_product_u32((uint32_t)rows, (uint32_t)d_in, sizeof(float), &input_bytes) ||
        !size_product_u32((uint32_t)d_in, (uint32_t)d_out, sizeof(float), &weight_bytes) ||
        !size_product_u32((uint32_t)rows, (uint32_t)d_out, sizeof(float), &output_bytes)) return 0;
    row_count = (uint32_t)rows;
    input_width = (uint32_t)d_in;
    output_width = (uint32_t)d_out;
    if (dual) {
        /* A tactic miss must precede any coherence work. Both full output
         * owners are then read before their selected row is written. */
        if (!residual || !final_output || !out_in || bias || rows != 1 ||
            refresh_host_input || sync_host_output || !cuda_domain_enforced ||
            !cuda_forward_scope_active || !cuda_forward_context_ready ||
            cuda_forward_context_failed || sequence <= 0 || row < 0 || row >= sequence ||
            (residual_first != 0 && residual_first != 1) ||
            !size_product_u32((uint32_t)sequence, input_width, sizeof(float), &input_bytes) ||
            !size_product_u32((uint32_t)sequence, output_width, sizeof(float), &output_bytes)) return 0;
        const void* hosts[5] = {input, weight, residual, output, final_output};
        size_t bytes[5] = {input_bytes, weight_bytes, output_bytes, output_bytes, output_bytes};
        CudaTensorView views[5];
        int bound = cuda_gemv_row_bind_owners(hosts, bytes, 1,
            (output_width + 3u) / 4u, views);
        if (bound != 1) return bound;
        input_view = views[0]; weight_slot = views[1].slot; output_view = views[3];
        weight_device = views[1].device;
        residual_view = views[2]; final_view = views[4];
        input_view.device += (uint64_t)row * input_width * sizeof(float);
        output_view.device += (uint64_t)row * output_width * sizeof(float);
        residual_view.device += (uint64_t)row * output_width * sizeof(float);
        final_view.device += (uint64_t)row * output_width * sizeof(float);
    } else {
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
        weight_device = weight_slot->device;
    }
    CUdeviceptr no_bias = 0;
    CUdeviceptr bias_device = bias_slot ? bias_slot->device : no_bias;
    args[0] = &input_view.device;
    args[1] = &weight_device;
    args[2] = &bias_device;
    args[3] = &output_view.device;
    args[4] = &row_count;
    args[5] = &input_width;
    args[6] = &output_width;
    uint32_t order = (uint32_t)residual_first;
    args[7] = &residual_view.device; args[8] = &final_view.device; args[9] = &order;
    unsigned int columns_per_block = out_in ? 4u : 8u;
    unsigned int gemv_grid = (output_width + columns_per_block - 1u) /
        columns_per_block;
    /* The compile proof retains the original tiled launch as the universal
     * route. Specialize only when the concrete device also admits this block,
     * grid and (for IN_OUT) its 576 bytes of static shared storage. */
    size_t split_bytes = vx_cuda_matmul_split_workspace_bytes(
        row_count, input_width, output_width, 2u);
    CudaContextState* state = split_bytes ? cuda_context_state_get() : NULL;
    unsigned int split_grid = 0u, reduce_grid = 0u;
    uint32_t split_k_slices = 0u;
    int split = split_bytes && state && state->matmul_workspace && !dual &&
        CUDA_FORWARD_FUNCTION(k_matmul_split) &&
        CUDA_FORWARD_FUNCTION(k_matmul_split_out_in) &&
        CUDA_FORWARD_FUNCTION(k_matmul_split_reduce);
    if (split) {
        split_grid = (output_width + 31u) / 32u;
        reduce_grid = (row_count * output_width + 127u) / 128u;
        split = cuda_launch_dimensions_valid(split_grid, 1u, 2u, 8u, 16u, 1u, 17152u) &&
            cuda_launch_dimensions_valid(reduce_grid, 1u, 1u, 128u, 1u, 1u, 0u) &&
            cuda_launch_dimensions_valid((output_width + 15u) / 16u,
                (row_count + 15u) / 16u, 1u, 8u, 16u, 1u, 8704u);
    }
    if (split) {
        const void* hosts[4] = {input, weight, bias, output};
        CUdeviceptr devices[4] = {input_view.device, weight_device, bias_device, output_view.device};
        size_t bytes[4] = {input_bytes, weight_bytes, bias ? (size_t)d_out * sizeof(float) : 0u, output_bytes};
        for (unsigned int index = 0; index < 4u && split; index++) {
            if (!bytes[index]) continue;
            if (ranges_overlap((const void*)(uintptr_t)state->matmul_workspace,
                    VX_CUDA_MATMUL_WORKSPACE_BYTES,
                    (const void*)(uintptr_t)devices[index], bytes[index]) ||
                (state->matmul_tuning.output && ranges_overlap(
                    (const void*)(uintptr_t)state->matmul_tuning.output,
                    VX_CUDA_MATMUL_CALIBRATION_OUTPUT_BYTES,
                    (const void*)(uintptr_t)devices[index], bytes[index]))) split = 0;
            if (index < 3u && (ranges_overlap(output, output_bytes, hosts[index], bytes[index]) ||
                ranges_overlap((const void*)(uintptr_t)output_view.device, output_bytes,
                    (const void*)(uintptr_t)devices[index], bytes[index]))) split = 0;
        }
    }
    if (split) {
        if (!cuda_matmul_choose_slices(state, input_view.device, weight_device,
                bias_device, row_count, input_width, output_width, out_in, &split_k_slices)) return 0;
        split_bytes = vx_cuda_matmul_split_workspace_bytes(
            row_count, input_width, output_width, split_k_slices);
        split = split_bytes && cuda_launch_dimensions_valid(split_grid, 1u, split_k_slices,
            8u, 16u, 1u, 17152u);
    }
    if (split) {
        uint32_t layout = (uint32_t)out_in;
        void* partial_args[7] = {&input_view.device, &weight_device, &state->matmul_workspace,
            &row_count, &input_width, &output_width, &split_k_slices};
        void* reduce_args[10] = {&input_view.device, &weight_device, &bias_device,
            &state->matmul_workspace, &output_view.device, &row_count,
            &input_width, &output_width, &layout, &split_k_slices};
        /* Every tactic miss precedes this pair. A submitted failure is terminal,
         * never a retry through the ordinary tiled kernel. */
        if (!cuda_launch(out_in ? CUDA_FORWARD_FUNCTION(k_matmul_split_out_in)
                               : CUDA_FORWARD_FUNCTION(k_matmul_split),
                split_grid, 1u, split_k_slices, 8u, 16u, 1u, 0u, partial_args) ||
            !cuda_launch(CUDA_FORWARD_FUNCTION(k_matmul_split_reduce),
                reduce_grid, 1u, 1u, 128u, 1u, 1u, 0u, reduce_args)) return 0;
    } else if (row_count == 1u &&
        (out_in || cuda_max_shared_memory_per_block >= 576u) &&
        cuda_launch_dimensions_valid(gemv_grid, 1u, 1u, 128u, 1u, 1u, 0u) &&
        !ranges_overlap(output, output_bytes, input, input_bytes) &&
        !ranges_overlap(output, output_bytes, weight, weight_bytes) &&
        (!bias || !ranges_overlap(output, output_bytes, bias,
                                  (size_t)d_out * sizeof(float)))) {
        gx = gemv_grid;
        if (!gx || !cuda_launch(out_in ? CUDA_FORWARD_FUNCTION(k_gemv_out_in)
                                      : CUDA_FORWARD_FUNCTION(k_gemv),
                                gx, 1u, 1u, 128u, 1u, 1u, 0u, args)) return dual ? -1 : 0;
    } else {
        int register_tile = row_count > 32u && output_width >= 1024u;
        unsigned int tile_columns = register_tile ? 64u : 16u;
        unsigned int tile_rows = register_tile ? 32u : 16u;
        unsigned int shared_bytes = register_tile ? 6336u : 8704u;
        CUfunction kernel = register_tile
            ? (out_in ? CUDA_FORWARD_FUNCTION(k_matmul_register_out_in)
                      : CUDA_FORWARD_FUNCTION(k_matmul_register))
            : (out_in ? CUDA_FORWARD_FUNCTION(k_matmul_out_in)
                      : CUDA_FORWARD_FUNCTION(k_matmul));
        gx = (output_width + tile_columns - 1u) / tile_columns;
        gy = (row_count + tile_rows - 1u) / tile_rows;
        if (!gx || !gy || gy > 65535u ||
            cuda_max_shared_memory_per_block < shared_bytes ||
            !cuda_launch(kernel, gx, gy, 1, 8, 16, 1, 0, args)) return dual ? -1 : 0;
    }
    graph_mark_device(output_view.slot);
    if (dual) graph_mark_device(final_view.slot);
    return !sync_host_output || cuda_graph_sync_host(output, output_bytes, 0);
}

int cuda_matmul(const float* input, const float* weight, const float* bias,
                float* output, int rows, int d_in, int d_out) {
    return cuda_matmul_impl(input, weight, bias, output, rows, d_in, d_out,
                            0, 1, 1, NULL, NULL, 0, 0, 0);
}

int cuda_graph_matmul_f32(const float* input, const float* weight,
                          const float* bias, float* output,
                          int rows, int d_in, int d_out) {
    return cuda_matmul_impl(input, weight, bias, output, rows, d_in, d_out,
                            0, 0, 0, NULL, NULL, 0, 0, 0);
}

int cuda_graph_matmul_layout_f32(const float* input, const float* weight,
                                 const float* bias, float* output,
                                 int rows, int d_in, int d_out, int out_in) {
    return cuda_matmul_impl(input, weight, bias, output, rows, d_in, d_out,
                            out_in, 0, 0, NULL, NULL, 0, 0, 0);
}

int cuda_graph_gemv_add_dual_f32(const float* input, const float* weight,
    const float* residual, float* dense, float* output,
    int sequence, int row, int d_in, int d_out, int residual_first) {
    if (!residual || !output) return 0;
    return cuda_matmul_impl(input, weight, NULL, dense, 1, d_in, d_out,
                            1, 0, 0, residual, output, residual_first, sequence, row);
}

/* A refused row tactic submits nothing. Once owner coherence starts,
 * any failure is terminal, because uploads or a kernel may be in flight. */
int cuda_graph_gemv_pair_out_in_f32(const float* input,
        const float* weight0, const float* weight1,
        float* output0, float* output1,
        int sequence, int row, int d_in, int d_out) {
    size_t input_bytes, weight_bytes, output_bytes;
    if (!input || !weight0 || !weight1 || !output0 || !output1 ||
        !cuda_domain_enforced || !cuda_forward_scope_active ||
        !cuda_forward_context_ready || cuda_forward_context_failed ||
        sequence <= 0 || row < 0 || row >= sequence || d_in <= 0 || d_out <= 0 ||
        !CUDA_FORWARD_FUNCTION(k_gemv_pair_out_in) ||
        !size_product_u32((uint32_t)sequence, (uint32_t)d_in, sizeof(float), &input_bytes) ||
        !size_product_u32((uint32_t)d_out, (uint32_t)d_in, sizeof(float), &weight_bytes) ||
        !size_product_u32((uint32_t)sequence, (uint32_t)d_out, sizeof(float), &output_bytes)) return 0;
    const void* hosts[5] = {input, weight0, weight1, output0, output1};
    size_t bytes[5] = {input_bytes, weight_bytes, weight_bytes, output_bytes, output_bytes};
    CudaTensorView views[5];
    uint32_t rows = 1u, input_width = (uint32_t)d_in, output_width = (uint32_t)d_out;
    unsigned int grid = (output_width + 3u) / 4u;
    int bound = cuda_gemv_row_bind_owners(hosts, bytes, 2, grid, views);
    if (bound != 1) return bound;
    views[0].device += (uint64_t)row * input_width * sizeof(float);
    views[3].device += (uint64_t)row * output_width * sizeof(float);
    views[4].device += (uint64_t)row * output_width * sizeof(float);
    void* args[8] = {&views[0].device, &views[1].device, &views[2].device,
        &views[3].device, &views[4].device, &rows, &input_width, &output_width};
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_gemv_pair_out_in), grid, 1u, 1u,
            128u, 1u, 1u, 0u, args)) return -1;
    graph_mark_device(views[3].slot);
    graph_mark_device(views[4].slot);
    return 1;
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
        !cuda_ok(cuda_trace_wait_stream(cuda_stream),
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

int cuda_graph_moe_router_extended_f32(const float* input, const float* weight,
        const float* bias, const float* correction, const int32_t* token_ids,
        const int32_t* hash_indices, float* indices, float* routes,
        uint32_t rows, uint32_t width, uint32_t experts, uint32_t top_k,
        float temperature, uint32_t normalize, uint32_t scoring,
        uint32_t groups, uint32_t selected_groups, uint32_t group_top_k,
        float scale, float epsilon, uint32_t vocabulary) {
    if (!input || !weight || !indices || !routes || !rows || !width || !experts ||
        !top_k || top_k > experts || scoring > 2 || normalize > 1 || !groups ||
        groups > experts || experts > 4096 || 2u * experts + groups > 8192 ||
        experts % groups || !selected_groups || selected_groups > groups ||
        !group_top_k || group_top_k > experts / groups ||
        top_k > selected_groups * (experts / groups) ||
        !isfinite(temperature) || temperature <= 0 || !isfinite(scale) || scale <= 0 ||
        !isfinite(epsilon) || epsilon < 0 || (!!token_ids != !!hash_indices) ||
        (hash_indices && (!vocabulary || groups != 1))) return 0;
    const void* pointers[8] = {input, weight, bias, correction, token_ids, hash_indices, indices, routes};
    uint32_t a[8] = {rows,width,1,1,rows,vocabulary,rows,rows};
    uint32_t b[8] = {width,experts,experts,experts,1,top_k,top_k,top_k};
    CudaTensorView views[8] = {{0}};
    CUdeviceptr devices[8] = {0};
    size_t bytes[8] = {0};
    for (int i = 0; i < 8; ++i) if (pointers[i]) {
        if (!size_product_u32(a[i], b[i], 4u, &bytes[i])) return 0;
        if (i < 6 ? !graph_ensure_input_view((const float*)pointers[i], bytes[i], i == 1 || i == 2 || i == 3 || i == 5, &views[i])
                  : !graph_output_view((float*)pointers[i], bytes[i], &views[i])) return 0;
        devices[i] = views[i].device;
    }
    for (int i = 0; i < 7; ++i)
        if (pointers[i] && ranges_overlap(routes, bytes[7], pointers[i], bytes[i])) return 0;
    for (int i = 0; i < 6; ++i)
        if (pointers[i] && ranges_overlap(indices, bytes[6], pointers[i], bytes[i])) return 0;
    void* args[] = {&devices[0],&devices[1],&devices[2],&devices[3],&devices[4],&devices[5],
        &devices[6],&devices[7],&rows,&width,&experts,&top_k,&temperature,&normalize,&scoring,
        &groups,&selected_groups,&group_top_k,&scale,&epsilon,&vocabulary};
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_moe_router_extended), rows,1,1,128,1,1,0,args)) return 0;
    graph_mark_device(views[6].slot); graph_mark_device(views[7].slot);
    return 1;
}

int cuda_graph_moe_gated_mlp_f32(const float* input, const float* gate_weight,
        const float* up_weight, const float* down_weight, const float* indices,
        const float* routes, float* output, uint32_t rows, uint32_t d_in,
        uint32_t hidden, uint32_t d_out, uint32_t experts, uint32_t top_k,
        uint32_t input_scale, float limit) {
    if (!input || !gate_weight || !up_weight || !down_weight || !indices || !routes ||
        !output || !rows || !d_in || !hidden || hidden > 8192 || !d_out || !experts ||
        !top_k || top_k > experts || input_scale > 1 || !isfinite(limit) || limit < 0) return 0;
    const float* pointers[7] = {input,gate_weight,up_weight,down_weight,indices,routes,output};
    uint32_t a[7] = {rows,experts,experts,experts,rows,rows,rows};
    uint32_t b[7] = {d_in,d_in,d_in,hidden,top_k,top_k,d_out};
    uint32_t c[7] = {1,hidden,hidden,d_out,1,1,1};
    size_t bytes[7]; CudaTensorView views[7]; uint32_t count;
    for (int i = 0; i < 7; ++i) {
        if (!checked_product4_bytes(a[i],b[i],c[i],1,4,&count,&bytes[i])) return 0;
        if (i < 6 ? !graph_ensure_input_view(pointers[i],bytes[i],i >= 1 && i <= 3,&views[i])
                  : !graph_output_view(output,bytes[i],&views[i])) return 0;
    }
    for (int i = 0; i < 6; ++i)
        if (ranges_overlap(output,bytes[6],pointers[i],bytes[i])) return 0;
    void* args[] = {&views[0].device,&views[1].device,&views[2].device,&views[3].device,
        &views[4].device,&views[5].device,&views[6].device,&rows,&d_in,&hidden,&d_out,
        &experts,&top_k,&input_scale,&limit};
    if (!cuda_launch(CUDA_FORWARD_FUNCTION(k_moe_gated_mlp),rows,1,1,128,1,1,0,args)) return 0;
    graph_mark_device(views[6].slot);
    return 1;
}
