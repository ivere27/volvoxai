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
