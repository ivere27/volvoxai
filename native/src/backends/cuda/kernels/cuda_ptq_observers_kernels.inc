/* Full-profile PTQ statistics. No activation leaves the device. The range
 * reduction has bounded partial storage; second moments use one ordered F64
 * accumulator per logical column, including the previously committed sum. */
#if defined(__clang__)
#define VX_PTQ_SHARED __attribute__((shared))
#else
#define VX_PTQ_SHARED __shared__
#endif

struct VxCudaPTQRange { float minimum, maximum; vx_u32 invalid; };

extern "C" VX_CUDA_GLOBAL void vx_cuda_ptq_observer_partials_f32(
        const float* input, VxCudaPTQRange* partials, unsigned long long count) {
    VX_PTQ_SHARED float minima[256], maxima[256];
    VX_PTQ_SHARED vx_u32 invalid[256];
    vx_u32 lane = threadIdx.x;
    unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + lane;
    unsigned long long stride = (unsigned long long)gridDim.x * blockDim.x;
    float lo = 3.4028234663852886e38f, hi = -3.4028234663852886e38f;
    vx_u32 bad = 0;
    for (; index < count; index += stride) {
        float value = input[index];
        if (!vx_cuda_training_step_isfinite(value)) bad = 1;
        else { if (value < lo) lo = value; if (value > hi) hi = value; }
    }
    minima[lane] = lo; maxima[lane] = hi; invalid[lane] = bad;
    vx_cuda_training_step_barrier();
    for (vx_u32 offset = 128; offset; offset >>= 1) {
        if (lane < offset) {
            if (minima[lane + offset] < minima[lane]) minima[lane] = minima[lane + offset];
            if (maxima[lane + offset] > maxima[lane]) maxima[lane] = maxima[lane + offset];
            invalid[lane] |= invalid[lane + offset];
        }
        vx_cuda_training_step_barrier();
    }
    if (!lane) partials[blockIdx.x] = {minima[0], maxima[0], invalid[0]};
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_ptq_observer_finish_f32(
        const VxCudaPTQRange* partials, VxCudaPTQRange* result, vx_u32 count) {
    if (vx_training_global_x()) return;
    VxCudaPTQRange total = partials[0];
    for (vx_u32 i = 1; i < count; i++) {
        if (partials[i].minimum < total.minimum) total.minimum = partials[i].minimum;
        if (partials[i].maximum > total.maximum) total.maximum = partials[i].maximum;
        total.invalid |= partials[i].invalid;
    }
    *result = total;
}

/* Separate correctly rounded instructions retain the CPU observer's F64
 * square/add order even when the CUDA compiler otherwise contracts FMA. */
static VX_CUDA_DEVICE double vx_cuda_ptq_add_square(double sum, float input) {
    double value = (double)input, square, result;
    asm("mul.rn.f64 %0, %1, %1;" : "=d"(square) : "d"(value));
    asm("add.rn.f64 %0, %1, %2;" : "=d"(result) : "d"(sum), "d"(square));
    return result;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_ptq_importance_linear_f32(
        const float* input, double* sums, unsigned long long count, vx_u32 channels) {
    for (unsigned long long column = vx_training_global_x(); column < channels;
         column += (unsigned long long)gridDim.x * blockDim.x) {
        double sum = sums[column];
        for (unsigned long long i = column; i < count; i += channels)
            sum = vx_cuda_ptq_add_square(sum, input[i]);
        sums[column] = sum;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_ptq_importance_conv_f32(
        const float* input, double* sums, unsigned long long columns,
        int n, int h, int w, int c, int oh, int ow, int kh, int kw,
        int ic_per_group, int groups, int stride_y, int stride_x,
        int dilation_y, int dilation_x, int pad_top, int pad_left) {
    unsigned long long count = columns * (unsigned)groups;
    for (unsigned long long index = vx_training_global_x(); index < count;
         index += (unsigned long long)gridDim.x * blockDim.x) {
        unsigned long long column = index % columns;
        unsigned group = (unsigned)(index / columns);
        unsigned ic = (unsigned)(column % (unsigned)ic_per_group);
        unsigned x = (unsigned)(column / (unsigned)ic_per_group % (unsigned)kw);
        unsigned y = (unsigned)(column / (unsigned)ic_per_group / (unsigned)kw);
        double sum = sums[index];
        for (int batch = 0; batch < n; batch++)
        for (int out_y = 0; out_y < oh; out_y++) {
            long long in_y = (long long)out_y * stride_y - pad_top + (long long)y * dilation_y;
            if (in_y < 0 || in_y >= h) continue;
            for (int out_x = 0; out_x < ow; out_x++) {
                long long in_x = (long long)out_x * stride_x - pad_left + (long long)x * dilation_x;
                if (in_x < 0 || in_x >= w) continue;
                unsigned long long base = (((unsigned long long)batch * (unsigned)h + in_y) *
                    (unsigned)w + in_x) * (unsigned)c;
                sum = vx_cuda_ptq_add_square(sum, input[base +
                    (unsigned long long)group * (unsigned)ic_per_group + ic]);
            }
        }
        sums[index] = sum;
    }
    (void)kh;
}
#undef VX_PTQ_SHARED
