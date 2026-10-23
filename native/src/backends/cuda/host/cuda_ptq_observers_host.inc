/* Included only in cuda_engine.c's full profile, after training-step helpers.
 * PTQ consumes existing forward device views and downloads sufficient
 * statistics only. A missing/host-dirty view is an error, never a CPU retry. */
typedef struct { float minimum, maximum; uint32_t invalid; } CudaPTQRange;
_Static_assert(sizeof(CudaPTQRange) == 12u, "CUDA PTQ range ABI");

static int cuda_ptq_device_input(const float* input, uint64_t count,
                                 CudaTensorView* view) {
    if (!cuda_ready || !cuda_training_module || cuda_forward_scope_active ||
        !input || !count || count > SIZE_MAX / sizeof(float)) return 0;
    size_t bytes = (size_t)count * sizeof(float);
    int index = graph_find_containing_slot(input, bytes);
    if (index < 0 || !graph_slots[index].device || graph_slots[index].host_dirty)
        return 0;
    return graph_get_view(input, bytes, 0, view);
}

int cuda_ptq_observe_f32(const float* input, uint64_t count,
                        float* minimum, float* maximum) {
    CudaTensorView view;
    CUdeviceptr partials = 0, result = 0;
    CudaPTQRange range;
    uint32_t blocks = count / 256u >= 256u ? 256u :
        (uint32_t)(count / 256u + (count % 256u != 0u));
    int status = -1;
    if (!minimum || !maximum || !blocks ||
        !CUDA_TRAINING_FUNCTION(k_ptq_observer_partials) ||
        !CUDA_TRAINING_FUNCTION(k_ptq_observer_finish) ||
        !cuda_ptq_device_input(input, count, &view)) return -1;
    if (!cuda_training_step_raw_alloc(&partials, (size_t)blocks * sizeof(range)) ||
        !cuda_training_step_raw_alloc(&result, sizeof(range))) goto done;
    void* first[] = {&view.device, &partials, &count};
    void* last[] = {&partials, &result, &blocks};
    if (!cuda_launch(CUDA_TRAINING_FUNCTION(k_ptq_observer_partials),
            blocks, 1, 1, 256, 1, 1, 0, first) ||
        !cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_ptq_observer_finish), 1, last) ||
        !cuda_training_step_sync_stream() ||
        !cuda_training_step_copy_dtoh(&range, result, sizeof(range)) ||
        range.invalid || !isfinite(range.minimum) || !isfinite(range.maximum) ||
        range.minimum > range.maximum) goto done;
    *minimum = range.minimum; *maximum = range.maximum;
    status = 0;
done:
    /* cuMemFree is ordered with outstanding uses, including launch failure. */
    cuda_training_step_raw_free(&result);
    cuda_training_step_raw_free(&partials);
    return status;
}

static int cuda_ptq_moments_finish(CUdeviceptr device, double* moments, size_t count) {
    if (!cuda_training_step_sync_stream() ||
        !cuda_training_step_copy_dtoh(moments, device, count * sizeof(double))) return -1;
    for (size_t i = 0; i < count; i++)
        if (!isfinite(moments[i]) || moments[i] < 0.0) return -1;
    return 0;
}

int cuda_ptq_importance_linear_f32(const float* input, uint64_t count,
                                  uint32_t channels, double* moments) {
    CudaTensorView view;
    CUdeviceptr sums = 0;
    float minimum, maximum;
    int status = -1;
    if (!moments || !channels || sizeof(double) > SIZE_MAX / channels ||
        !count || count % channels ||
        !CUDA_TRAINING_FUNCTION(k_ptq_importance_linear) ||
        !cuda_ptq_device_input(input, count, &view) ||
        cuda_ptq_observe_f32(input, count, &minimum, &maximum)) return -1;
    size_t bytes = (size_t)channels * sizeof(double);
    if (!cuda_training_step_raw_alloc(&sums, bytes) ||
        !cuda_training_step_copy_htod(sums, moments, bytes)) goto done;
    void* args[] = {&view.device, &sums, &count, &channels};
    uint32_t work = channels > 65536u ? 65536u : channels;
    if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_ptq_importance_linear), work, args)) goto done;
    status = cuda_ptq_moments_finish(sums, moments, channels);
done:
    cuda_training_step_raw_free(&sums);
    return status;
}

int cuda_ptq_importance_conv_f32(const float* input,
        const CudaPTQConvGeometry* geometry, double* moments) {
    CudaTensorView view;
    CUdeviceptr sums = 0;
    CudaPTQConvGeometry g;
    float minimum, maximum;
    uint64_t elements = 1, columns = 1;
    int status = -1;
    if (!geometry || !moments || !CUDA_TRAINING_FUNCTION(k_ptq_importance_conv)) return -1;
    g = *geometry;
    int dims[] = {g.n, g.h, g.w, g.c};
    int kernel[] = {g.kh, g.kw, g.ic_per_group};
    for (size_t i = 0; i < 4; i++) {
        if (dims[i] <= 0 || elements > SIZE_MAX / sizeof(float) / (uint32_t)dims[i]) return -1;
        elements *= (uint32_t)dims[i];
    }
    for (size_t i = 0; i < 3; i++) {
        if (kernel[i] <= 0 || columns > SIZE_MAX / sizeof(double) / (uint32_t)kernel[i]) return -1;
        columns *= (uint32_t)kernel[i];
    }
    if (g.groups <= 0 || g.oh <= 0 || g.ow <= 0 || g.stride_y <= 0 || g.stride_x <= 0 ||
        g.dilation_y <= 0 || g.dilation_x <= 0 || g.pad_top < 0 || g.pad_left < 0 ||
        (uint64_t)(uint32_t)g.groups * (uint32_t)g.ic_per_group != (uint32_t)g.c ||
        columns > SIZE_MAX / sizeof(double) / (uint32_t)g.groups ||
        !cuda_ptq_device_input(input, elements, &view) ||
        cuda_ptq_observe_f32(input, elements, &minimum, &maximum)) return -1;
    size_t count = (size_t)columns * (uint32_t)g.groups;
    size_t bytes = count * sizeof(double);
    if (!cuda_training_step_raw_alloc(&sums, bytes) ||
        !cuda_training_step_copy_htod(sums, moments, bytes)) goto done;
    void* args[] = {&view.device, &sums, &columns, &g.n, &g.h, &g.w, &g.c,
        &g.oh, &g.ow, &g.kh, &g.kw, &g.ic_per_group, &g.groups,
        &g.stride_y, &g.stride_x, &g.dilation_y, &g.dilation_x, &g.pad_top, &g.pad_left};
    uint32_t work = count > 65536u ? 65536u : (uint32_t)count;
    if (!cuda_launch_1d(CUDA_TRAINING_FUNCTION(k_ptq_importance_conv), work, args)) goto done;
    status = cuda_ptq_moments_finish(sums, moments, count);
done:
    cuda_training_step_raw_free(&sums);
    return status;
}
