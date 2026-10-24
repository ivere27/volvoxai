/* Select an immutable table before any tensor command. Capture/validation
 * cannot introduce a new table: the preceding observed pass must publish all
 * of its keys successfully. No cached address depends on a host tensor pointer
 * or on a shape binding. */
static int cuda_rope_powers_view(CudaContextState* state, int rotary_width,
        float theta, CUdeviceptr* device) {
    uint32_t theta_bits;
    uint32_t index;
    if (!state || !state->rope_powers_device || !device || rotary_width <= 0 ||
        rotary_width > (int)(CUDA_ROPE_POWER_CAPACITY * 2u) || (rotary_width & 1) ||
        !isfinite(theta) || theta < 1.0f ||
        state->rope_power_cache.count > CUDA_ROPE_CACHE_CAPACITY) return 0;
    memcpy(&theta_bits, &theta, sizeof(theta_bits));
    for (index = 0; index < state->rope_power_cache.count; index++) {
        const CudaRopePowerKey* key = &state->rope_power_cache.keys[index];
        if (key->theta_bits == theta_bits && key->rotary_width == (uint32_t)rotary_width) {
            *device = state->rope_powers_device + (size_t)index * CUDA_ROPE_POWER_TABLE_BYTES;
            return 1;
        }
    }
    if (cuda_forward_scope_active &&
        (cuda_replay.pass == CUDA_REPLAY_PASS_CAPTURE ||
         cuda_replay.pass == CUDA_REPLAY_PASS_VALIDATE)) {
        cuda_replay_fail();
        return 0;
    }
    index = state->rope_power_cache.count;
    if (index == CUDA_ROPE_CACHE_CAPACITY && cuda_forward_scope_active &&
        cuda_replay.pass == CUDA_REPLAY_PASS_OBSERVE)
        cuda_replay.unsafe = 1;
    *device = state->rope_powers_device + (size_t)index * CUDA_ROPE_POWER_TABLE_BYTES;
    for (uint32_t pair = 0; pair < (uint32_t)rotary_width / 2u; pair++) {
        float exponent = (2.0f * (float)pair) / (float)rotary_width;
        state->rope_powers_host[pair] = powf(theta, exponent);
    }
    /* Only a miss uses the synchronous transfer slab. It completes before the
     * scratch host table is reused; the immutable device table is never
     * overwritten. Overflow retains the prior uncaptured sequential behavior. */
    if (!cuda_ok(cuda_copy_htod_current(*device, state->rope_powers_host,
            (size_t)rotary_width / 2u * sizeof(float)),
            "cuMemcpyHtoD(RoPE operator metadata)")) return 0;
    if (index < CUDA_ROPE_CACHE_CAPACITY) {
        state->rope_power_cache.keys[index] =
            (CudaRopePowerKey){theta_bits, (uint32_t)rotary_width};
        state->rope_power_cache.count++;
    }
    return 1;
}

int cuda_graph_rope_heads_f32(const float* input, const int32_t* position_ids,
    long position_elements, float* output, int batch, int sequence, int width,
    int heads, int rotary_width, float theta, int32_t position_offset,
    int interleaved, int position_mode) {
    CudaTensorView input_view, positions_view, output_view;
    uint32_t elements, params[8];
    size_t bytes, position_bytes = 0;
    uint64_t expected_positions = 0;
    CUdeviceptr no_positions = 0, positions_device;
    void* args[12];
    if (!input || !output || batch <= 0 || sequence <= 0 || width <= 0 ||
        heads <= 0 || width % heads || width / heads > 128 ||
        rotary_width <= 0 || rotary_width > width / heads || (rotary_width & 1) ||
        !isfinite(theta) || theta < 1.0f || position_offset < 0 ||
        (uint32_t)position_offset > INT32_MAX - ((uint32_t)sequence - 1u) ||
        (interleaved != 0 && interleaved != 1) || position_mode < 0 || position_mode > 2 ||
        !checked_product4_bytes((uint32_t)batch, (uint32_t)sequence,
            (uint32_t)width, 1u, sizeof(float), &elements, &bytes)) return 0;
    if (position_mode == 1) expected_positions = (uint32_t)sequence;
    else if (position_mode == 2)
        expected_positions = (uint64_t)(uint32_t)batch * (uint32_t)sequence;
    if ((!expected_positions && (position_ids || position_elements)) ||
        (expected_positions && (!position_ids || position_elements <= 0 ||
            (uint64_t)position_elements != expected_positions ||
            expected_positions > UINT32_MAX ||
            expected_positions > SIZE_MAX / sizeof(int32_t)))) return 0;
    if (expected_positions) {
        position_bytes = (size_t)expected_positions * sizeof(int32_t);
        /* The physical admission contract restricts these metadata values to
         * authoritative host inputs/constants. Scan the whole list before any
         * allocation/upload/launch, including entries outside an active row. */
        for (size_t index = 0; index < expected_positions; index++) {
            int32_t position;
            memcpy(&position, (const unsigned char*)position_ids + index * sizeof(position), sizeof(position));
            if (position < 0) return 0;
        }
    }
    if (ranges_overlap(output, bytes, input, bytes) ||
        (position_ids && ranges_overlap(output, bytes, position_ids, position_bytes))) return 0;
    CudaContextState* state = cuda_context_state_get();
    CUdeviceptr powers_device;
    if (!state || !state->ready ||
        !cuda_rope_powers_view(state, rotary_width, theta, &powers_device)) return 0;
    params[0] = elements;
    params[1] = (uint32_t)sequence;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)heads;
    params[4] = (uint32_t)rotary_width;
    params[5] = (uint32_t)position_offset;
    params[6] = (uint32_t)interleaved;
    params[7] = (uint32_t)position_mode;
    if (!graph_ensure_input_view(input, bytes, 0, &input_view) ||
        (position_ids && !graph_ensure_input_view(position_ids, position_bytes, 0, &positions_view)) ||
        !graph_output_view(output, bytes, &output_view)) return 0;
    positions_device = position_ids ? positions_view.device : no_positions;
    args[0] = &input_view.device;
    args[1] = &positions_device;
    args[2] = &output_view.device;
    for (int index = 0; index < 5; index++) args[index + 3] = &params[index];
    args[8] = &powers_device;
    args[9] = &params[5];
    args[10] = &params[6];
    args[11] = &params[7];
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_rope_heads), elements, args)) return 0;
    graph_mark_device(output_view.slot);
    return 1;
}

int cuda_graph_rope_pair_heads_f32(
    const float* input0, float* output0, int width0, int heads0,
    const float* input1, float* output1, int width1, int heads1,
    const int32_t* position_ids, long position_elements,
    int sequence, int row, int rotary_width, float theta,
    int32_t position_offset, int interleaved, int position_mode) {
    if (!input0 || !output0 || !input1 || !output1 ||
        sequence <= 0 || row < 0 || row >= sequence ||
        width0 <= 0 || heads0 <= 0 || width0 % heads0 ||
        width1 <= 0 || heads1 <= 0 || width1 % heads1 ||
        rotary_width <= 0 || (rotary_width & 1) ||
        rotary_width != width0 / heads0 || rotary_width != width1 / heads1 ||
        !isfinite(theta) || theta < 1.0f ||
        (interleaved != 0 && interleaved != 1) || position_mode < 0 || position_mode > 2 ||
        !CUDA_FORWARD_FUNCTION(k_rope_pair_heads)) return 0;

    CudaContextState* state = cuda_context_state_get();
    CUdeviceptr powers_device;
    if (!state || !state->ready ||
        !cuda_rope_powers_view(state, rotary_width, theta, &powers_device)) return 0;

    size_t in0_bytes = (size_t)sequence * width0 * sizeof(float);
    size_t out0_bytes = (size_t)sequence * width0 * sizeof(float);
    size_t in1_bytes = (size_t)sequence * width1 * sizeof(float);
    size_t out1_bytes = (size_t)sequence * width1 * sizeof(float);
    size_t pos_bytes = (size_t)position_elements * sizeof(int32_t);

    CudaTensorView in0_view, out0_view, in1_view, out1_view, pos_view;
    if (!graph_ensure_input_view(input0, in0_bytes, 0, &in0_view) ||
        !graph_output_view(output0, out0_bytes, &out0_view) ||
        !graph_ensure_input_view(input1, in1_bytes, 0, &in1_view) ||
        !graph_output_view(output1, out1_bytes, &out1_view)) return 0;

    if (position_ids && !graph_ensure_input_view(position_ids, pos_bytes, 0, &pos_view)) return 0;
    CUdeviceptr pos_device = position_ids ? pos_view.device : 0;

    uint32_t elements0 = (uint32_t)width0;
    uint32_t elements1 = (uint32_t)width1;
    uint32_t u_width0 = (uint32_t)width0;
    uint32_t u_heads0 = (uint32_t)heads0;
    uint32_t u_width1 = (uint32_t)width1;
    uint32_t u_heads1 = (uint32_t)heads1;
    uint32_t u_rotary = (uint32_t)rotary_width;
    uint32_t u_offset = (uint32_t)position_offset;
    uint32_t u_interleaved = (uint32_t)interleaved;
    uint32_t u_mode = (uint32_t)position_mode;
    CUdeviceptr argument_storage = 0;

    in0_view.device += (uint64_t)row * width0 * sizeof(float);
    out0_view.device += (uint64_t)row * width0 * sizeof(float);
    in1_view.device += (uint64_t)row * width1 * sizeof(float);
    out1_view.device += (uint64_t)row * width1 * sizeof(float);

    void* args[17] = {
        &in0_view.device,
        &pos_device,
        &out0_view.device,
        &in1_view.device,
        &out1_view.device,
        &elements0,
        &elements1,
        &u_width0,
        &u_heads0,
        &u_width1,
        &u_heads1,
        &u_rotary,
        &powers_device,
        &u_offset,
        &u_interleaved,
        &u_mode,
        &argument_storage
    };

    uint32_t total_pairs = (u_heads0 + u_heads1) * (u_rotary / 2u);
    if (!cuda_launch_1d(CUDA_FORWARD_FUNCTION(k_rope_pair_heads), total_pairs, args)) return 0;

    graph_mark_device(out0_view.slot);
    graph_mark_device(out1_view.slot);
    return 1;
}
