/* Full-profile host orchestration for cuda_quantize_w8_kernels.inc. This
 * fragment is composed only into cuda_engine.c when training/PTQ is enabled.
 * It uses the already-loaded Driver API table and never links CUDA libraries. */

typedef struct {
    CUdeviceptr source;
    CUdeviceptr output;
    CUdeviceptr scales;
    CUdeviceptr zero_points;
    CUdeviceptr minima;
    CUdeviceptr maxima;
    CUdeviceptr candidate_scales;
    CUdeviceptr candidate_zero_points;
    CUdeviceptr status;
    CUdeviceptr saturation_by_block;
    CUdeviceptr saturation_total;
    CUdeviceptr bias;
    CUdeviceptr packed_bias;
} CudaQuantizeW8Storage;

static int cuda_quantize_w8_alloc(CUdeviceptr* output, size_t bytes) {
    return output && bytes &&
        cuda_ok(p_cuMemAlloc(output, bytes), "cuMemAlloc(quantize W8)");
}

static int cuda_quantize_w8_copy_h2d(CUdeviceptr output, const void* input,
                                     size_t bytes) {
    int copied;
    if (!output || !input || !bytes) return 0;
    copied = cuda_ok(cuda_copy_htod_current(output, input, bytes),
                     "cuMemcpyHtoD(quantize W8)");
    return copied;
}

static int cuda_quantize_w8_copy_d2h(void* output, CUdeviceptr input,
                                     size_t bytes) {
    int copied;
    if (!output || !input || !bytes) return 0;
    copied = cuda_ok(cuda_copy_dtoh_current(output, input, bytes),
                     "cuMemcpyDtoH(quantize W8)");
    return copied;
}

static void cuda_quantize_w8_free_device(CudaQuantizeW8Storage* storage) {
    if (!storage || !p_cuMemFree) return;
    CUdeviceptr* allocations[] = {
        &storage->packed_bias, &storage->bias,
        &storage->saturation_total, &storage->saturation_by_block,
        &storage->status, &storage->candidate_zero_points,
        &storage->candidate_scales, &storage->maxima, &storage->minima,
        &storage->zero_points, &storage->scales, &storage->output,
        &storage->source,
    };
    for (size_t index = 0;
         index < sizeof(allocations) / sizeof(allocations[0]); index++) {
        if (*allocations[index]) {
            (void)p_cuMemFree(*allocations[index]);
            *allocations[index] = 0;
        }
    }
}

static int cuda_quantize_w8_host_ranges_valid(
        const float* source, size_t source_bytes, const int8_t* output,
        size_t output_bytes, const float* scales, size_t scales_bytes,
        const int32_t* zero_points, size_t zero_point_bytes,
        const float* bias, const int32_t* packed_bias,
        size_t bias_bytes, const uint64_t* saturation_count) {
    size_t saturation_bytes = saturation_count ? sizeof(*saturation_count) : 0u;
    if (ranges_overlap(source, source_bytes, output, output_bytes) ||
        ranges_overlap(source, source_bytes, scales, scales_bytes) ||
        ranges_overlap(source, source_bytes, zero_points, zero_point_bytes) ||
        ranges_overlap(output, output_bytes, scales, scales_bytes) ||
        ranges_overlap(output, output_bytes, zero_points, zero_point_bytes) ||
        ranges_overlap(scales, scales_bytes, zero_points, zero_point_bytes) ||
        ranges_overlap(bias, bias_bytes, packed_bias, bias_bytes) ||
        ranges_overlap(source, source_bytes, packed_bias, bias_bytes) ||
        ranges_overlap(output, output_bytes, bias, bias_bytes) ||
        ranges_overlap(output, output_bytes, packed_bias, bias_bytes) ||
        ranges_overlap(scales, scales_bytes, bias, bias_bytes) ||
        ranges_overlap(scales, scales_bytes, packed_bias, bias_bytes) ||
        ranges_overlap(zero_points, zero_point_bytes, bias, bias_bytes) ||
        ranges_overlap(zero_points, zero_point_bytes, packed_bias, bias_bytes) ||
        ranges_overlap(saturation_count, saturation_bytes,
                       source, source_bytes) ||
        ranges_overlap(saturation_count, saturation_bytes,
                       output, output_bytes) ||
        ranges_overlap(saturation_count, saturation_bytes,
                       scales, scales_bytes) ||
        ranges_overlap(saturation_count, saturation_bytes,
                       zero_points, zero_point_bytes) ||
        ranges_overlap(saturation_count, saturation_bytes,
                       bias, bias_bytes) ||
        ranges_overlap(saturation_count, saturation_bytes,
                       packed_bias, bias_bytes)) return 0;
    return 1;
}

int cuda_training_quantize_w8_available(void) {
    return cuda_ready && cuda_training_module &&
        CUDA_TRAINING_FUNCTION(k_quantize_w8_reset) && CUDA_TRAINING_FUNCTION(k_quantize_w8_reduce) &&
        CUDA_TRAINING_FUNCTION(k_quantize_w8_derive_params) &&
        CUDA_TRAINING_FUNCTION(k_quantize_w8_validate_preserved_params) &&
        CUDA_TRAINING_FUNCTION(k_quantize_w8_commit_params) && CUDA_TRAINING_FUNCTION(k_quantize_w8_f32) &&
        CUDA_TRAINING_FUNCTION(k_quantize_w8_sum_saturation) &&
        CUDA_TRAINING_FUNCTION(k_quantize_w8_validate_bias) && CUDA_TRAINING_FUNCTION(k_quantize_w8_pack_bias) &&
        !cuda_forward_scope_active && !cuda_training_active;
}

int cuda_training_quantize_w8_f32(
        const float* source, int8_t* output, float* scales,
        int32_t* zero_points, uint32_t rows, uint32_t columns,
        uint32_t transpose_source, uint32_t scheme, uint32_t scale_policy,
        const float* bias, float input_scale, int32_t* packed_bias,
        uint64_t* saturation_count, uint32_t* out_status) {
    enum {
        CUDA_QUANTIZE_W8_INVALID_ARGUMENT = 1u,
        CUDA_QUANTIZE_W8_RECOMPUTE = 0u,
        CUDA_QUANTIZE_W8_PRESERVE = 1u,
    };
    CudaQuantizeW8Storage device;
    CudaContextGuard guard;
    uint64_t element_count64;
    uint32_t element_count;
    uint32_t checked_count;
    uint32_t block_count;
    uint32_t row_groups;
    size_t source_bytes;
    size_t output_bytes;
    size_t scales_bytes;
    size_t zero_point_bytes;
    size_t bias_bytes;
    size_t saturation_blocks_bytes;
    int has_bias = bias && packed_bias;
    int context_entered = 0;
    int success = 0;
    uint32_t device_status = CUDA_QUANTIZE_W8_INVALID_ARGUMENT;
    uint64_t staged_saturation = 0;
    int8_t* staged_output = NULL;
    float* staged_scales = NULL;
    int32_t* staged_zero_points = NULL;
    int32_t* staged_bias = NULL;
    void* args[10];

    memset(&device, 0, sizeof(device));
    memset(&guard, 0, sizeof(guard));
    if (out_status) *out_status = CUDA_QUANTIZE_W8_INVALID_ARGUMENT;
    element_count64 = (uint64_t)rows * (uint64_t)columns;
    if (!cuda_training_quantize_w8_available() || !source || !output ||
        !scales || !zero_points || !rows || !columns ||
        element_count64 > UINT32_MAX || transpose_source > 1u ||
        scheme > 1u || scale_policy > CUDA_QUANTIZE_W8_PRESERVE ||
        (!!bias != !!packed_bias) ||
        (has_bias && (!isfinite(input_scale) || input_scale <= 0.0f)))
        return 0;

    element_count = (uint32_t)element_count64;
    if (!checked_elements_bytes(element_count64, sizeof(float),
                                &checked_count, &source_bytes) ||
        !checked_elements_bytes(rows, sizeof(float),
                                &checked_count, &scales_bytes) ||
        !checked_elements_bytes(rows, sizeof(int32_t),
                                &checked_count, &zero_point_bytes)) return 0;
    block_count = element_count / 256u + (element_count % 256u != 0u);
    row_groups = rows / 256u + (rows % 256u != 0u);
    output_bytes = (size_t)element_count;
    bias_bytes = has_bias ? (size_t)rows * sizeof(float) : 0u;
    saturation_blocks_bytes = (size_t)block_count * sizeof(uint32_t);
    if (!cuda_quantize_w8_host_ranges_valid(
            source, source_bytes, output, output_bytes, scales, scales_bytes,
            zero_points, zero_point_bytes, bias, packed_bias, bias_bytes,
            saturation_count)) return 0;

    staged_output = (int8_t*)malloc(output_bytes);
    staged_scales = (float*)malloc(scales_bytes);
    staged_zero_points = (int32_t*)malloc(zero_point_bytes);
    staged_bias = has_bias ? (int32_t*)malloc(zero_point_bytes) : NULL;
    if (!staged_output || !staged_scales || !staged_zero_points ||
        (has_bias && !staged_bias)) goto done;

    if (!cuda_context_enter_owned(&guard)) goto done;
    context_entered = 1;
    if (!cuda_ok(p_cuStreamSynchronize(cuda_stream),
                 "cuStreamSynchronize(quantize W8 begin)")) goto device_done;
    if (!cuda_quantize_w8_alloc(&device.source, source_bytes) ||
        !cuda_quantize_w8_alloc(&device.output, output_bytes) ||
        !cuda_quantize_w8_alloc(&device.scales, scales_bytes) ||
        !cuda_quantize_w8_alloc(&device.zero_points, zero_point_bytes) ||
        !cuda_quantize_w8_alloc(&device.status, sizeof(uint32_t)) ||
        !cuda_quantize_w8_alloc(&device.saturation_by_block,
                                saturation_blocks_bytes) ||
        !cuda_quantize_w8_alloc(&device.saturation_total,
                                sizeof(uint64_t)) ||
        (scale_policy == CUDA_QUANTIZE_W8_RECOMPUTE &&
         (!cuda_quantize_w8_alloc(&device.minima, scales_bytes) ||
          !cuda_quantize_w8_alloc(&device.maxima, scales_bytes) ||
          !cuda_quantize_w8_alloc(&device.candidate_scales, scales_bytes) ||
          !cuda_quantize_w8_alloc(&device.candidate_zero_points,
                                  zero_point_bytes))) ||
        (has_bias &&
         (!cuda_quantize_w8_alloc(&device.bias, bias_bytes) ||
          !cuda_quantize_w8_alloc(&device.packed_bias,
                                  zero_point_bytes)))) goto device_done;

    if (!cuda_quantize_w8_copy_h2d(device.source, source, source_bytes) ||
        (scale_policy == CUDA_QUANTIZE_W8_PRESERVE &&
         (!cuda_quantize_w8_copy_h2d(device.scales, scales, scales_bytes) ||
          !cuda_quantize_w8_copy_h2d(device.zero_points, zero_points,
                                     zero_point_bytes))) ||
        (has_bias &&
         !cuda_quantize_w8_copy_h2d(device.bias, bias, bias_bytes)))
        goto device_done;

    args[0] = &device.status;
    if (!cuda_launch_in_owned_context(
            CUDA_TRAINING_FUNCTION(k_quantize_w8_reset), 1u, 1u, 1u,
            1u, 1u, 1u, 0u, args)) goto device_done;

    if (scale_policy == CUDA_QUANTIZE_W8_RECOMPUTE) {
        args[0] = &device.source;
        args[1] = &device.minima;
        args[2] = &device.maxima;
        args[3] = &device.status;
        args[4] = &rows;
        args[5] = &columns;
        args[6] = &transpose_source;
        if (!cuda_launch_in_owned_context(
                CUDA_TRAINING_FUNCTION(k_quantize_w8_reduce), rows, 1u, 1u,
                256u, 1u, 1u, 0u, args)) goto device_done;
        args[0] = &device.minima;
        args[1] = &device.maxima;
        args[2] = &device.candidate_scales;
        args[3] = &device.candidate_zero_points;
        args[4] = &device.status;
        args[5] = &rows;
        args[6] = &scheme;
        if (!cuda_launch_in_owned_context(
                CUDA_TRAINING_FUNCTION(k_quantize_w8_derive_params),
                row_groups, 1u, 1u, 256u, 1u, 1u, 0u, args))
            goto device_done;
    } else {
        args[0] = &device.scales;
        args[1] = &device.zero_points;
        args[2] = &device.status;
        args[3] = &rows;
        args[4] = &scheme;
        if (!cuda_launch_in_owned_context(
                CUDA_TRAINING_FUNCTION(k_quantize_w8_validate_preserved_params),
                row_groups, 1u, 1u, 256u, 1u, 1u, 0u, args))
            goto device_done;
    }

    if (has_bias) {
        CUdeviceptr bias_scales = scale_policy == CUDA_QUANTIZE_W8_RECOMPUTE
            ? device.candidate_scales : device.scales;
        args[0] = &device.bias;
        args[1] = &input_scale;
        args[2] = &bias_scales;
        args[3] = &device.status;
        args[4] = &rows;
        if (!cuda_launch_in_owned_context(
                CUDA_TRAINING_FUNCTION(k_quantize_w8_validate_bias),
                row_groups, 1u, 1u, 256u, 1u, 1u, 0u, args))
            goto device_done;
    }

    if (scale_policy == CUDA_QUANTIZE_W8_RECOMPUTE) {
        args[0] = &device.candidate_scales;
        args[1] = &device.candidate_zero_points;
        args[2] = &device.scales;
        args[3] = &device.zero_points;
        args[4] = &device.status;
        args[5] = &rows;
        args[6] = &scale_policy;
        if (!cuda_launch_in_owned_context(
                CUDA_TRAINING_FUNCTION(k_quantize_w8_commit_params),
                row_groups, 1u, 1u, 256u, 1u, 1u, 0u, args))
            goto device_done;
    }

    args[0] = &device.source;
    args[1] = &device.scales;
    args[2] = &device.zero_points;
    args[3] = &device.output;
    args[4] = &device.saturation_by_block;
    args[5] = &device.status;
    args[6] = &rows;
    args[7] = &columns;
    args[8] = &transpose_source;
    args[9] = &scheme;
    if (!cuda_launch_in_owned_context(
            CUDA_TRAINING_FUNCTION(k_quantize_w8_f32), block_count, 1u, 1u,
            256u, 1u, 1u, 0u, args)) goto device_done;
    args[0] = &device.saturation_by_block;
    args[1] = &device.saturation_total;
    args[2] = &device.status;
    args[3] = &block_count;
    if (!cuda_launch_in_owned_context(
            CUDA_TRAINING_FUNCTION(k_quantize_w8_sum_saturation),
            1u, 1u, 1u, 1u, 1u, 1u, 0u, args)) goto device_done;
    if (has_bias) {
        args[0] = &device.bias;
        args[1] = &input_scale;
        args[2] = &device.scales;
        args[3] = &device.packed_bias;
        args[4] = &device.status;
        args[5] = &rows;
        if (!cuda_launch_in_owned_context(
                CUDA_TRAINING_FUNCTION(k_quantize_w8_pack_bias),
                row_groups, 1u, 1u, 256u, 1u, 1u, 0u, args))
            goto device_done;
    }
    if (!cuda_ok(p_cuStreamSynchronize(cuda_stream),
                 "cuStreamSynchronize(quantize W8)")) goto device_done;
    if (!cuda_quantize_w8_copy_d2h(&device_status, device.status,
                                   sizeof(device_status))) goto device_done;
    if (out_status) *out_status = device_status;
    if (device_status != 0u) goto device_done;
    if (!cuda_quantize_w8_copy_d2h(staged_output, device.output,
                                   output_bytes) ||
        !cuda_quantize_w8_copy_d2h(staged_scales, device.scales,
                                   scales_bytes) ||
        !cuda_quantize_w8_copy_d2h(staged_zero_points, device.zero_points,
                                   zero_point_bytes) ||
        !cuda_quantize_w8_copy_d2h(&staged_saturation,
                                   device.saturation_total,
                                   sizeof(staged_saturation)) ||
        (has_bias &&
         !cuda_quantize_w8_copy_d2h(staged_bias, device.packed_bias,
                                    zero_point_bytes))) goto device_done;
    success = 1;

device_done:
    if (cuda_stream && p_cuStreamSynchronize)
        (void)p_cuStreamSynchronize(cuda_stream);
    cuda_quantize_w8_free_device(&device);
    if (!cuda_context_leave(&guard)) success = 0;
    context_entered = 0;

    if (success) {
        memcpy(output, staged_output, output_bytes);
        memcpy(scales, staged_scales, scales_bytes);
        memcpy(zero_points, staged_zero_points, zero_point_bytes);
        if (has_bias) memcpy(packed_bias, staged_bias, zero_point_bytes);
        if (saturation_count) *saturation_count = staged_saturation;
        if (out_status) *out_status = 0u;
    }

done:
    if (context_entered) {
        cuda_quantize_w8_free_device(&device);
        (void)cuda_context_leave(&guard);
    }
    free(staged_output);
    free(staged_scales);
    free(staged_zero_points);
    free(staged_bias);
    return success;
}
