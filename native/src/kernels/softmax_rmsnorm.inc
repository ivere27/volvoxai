/* Softmax and RMSNorm row reductions, with NEON and WASM SIMD row kernels. */

void rmsnorm_f32(const float* input, const float* weight, float* output, int seq_len, int d_model, double eps) {
    for (int s = 0; s < seq_len; s++) {
        double var = 0.0;
        for (int d = 0; d < d_model; d++) {
            double value = input[s*d_model + d];
            var += value * value;
        }
        var /= (double)d_model;
        double inv_std = 1.0 / __builtin_sqrt(var + eps);
        for (int d = 0; d < d_model; d++) {
            output[s*d_model + d] = (float)((double)input[s*d_model + d] * inv_std * weight[d]);
        }
    }
}
typedef struct VxSoftmaxF32RowsContext {
    const float* input;
    float* output;
    int width;
    int use_avx2;
    int use_neon;
} VxSoftmaxF32RowsContext;

#if VX_FASTMATH_ARM_NEON
static int vx_softmax_f32_finite_max_neon(
        const float* input, int width, float* maximum_out) {
    const uint32x4_t exponent_mask =
        vdupq_n_u32(UINT32_C(0x7f800000));
    float32x4_t maximum = vld1q_f32(input);
    uint32x4_t nonfinite = vceqq_u32(
        vandq_u32(vreinterpretq_u32_f32(maximum), exponent_mask),
        exponent_mask);
    int column = 4;
    for (; column + 16 <= width; column += 16) {
        const float32x4_t value0 = vld1q_f32(input + column);
        const float32x4_t value1 = vld1q_f32(input + column + 4);
        const float32x4_t value2 = vld1q_f32(input + column + 8);
        const float32x4_t value3 = vld1q_f32(input + column + 12);
        maximum = vmaxq_f32(maximum, value0);
        maximum = vmaxq_f32(maximum, value1);
        maximum = vmaxq_f32(maximum, value2);
        maximum = vmaxq_f32(maximum, value3);
        nonfinite = vorrq_u32(nonfinite, vceqq_u32(
            vandq_u32(vreinterpretq_u32_f32(value0), exponent_mask),
            exponent_mask));
        nonfinite = vorrq_u32(nonfinite, vceqq_u32(
            vandq_u32(vreinterpretq_u32_f32(value1), exponent_mask),
            exponent_mask));
        nonfinite = vorrq_u32(nonfinite, vceqq_u32(
            vandq_u32(vreinterpretq_u32_f32(value2), exponent_mask),
            exponent_mask));
        nonfinite = vorrq_u32(nonfinite, vceqq_u32(
            vandq_u32(vreinterpretq_u32_f32(value3), exponent_mask),
            exponent_mask));
    }
    for (; column + 4 <= width; column += 4) {
        const float32x4_t value = vld1q_f32(input + column);
        maximum = vmaxq_f32(maximum, value);
        nonfinite = vorrq_u32(nonfinite, vceqq_u32(
            vandq_u32(vreinterpretq_u32_f32(value), exponent_mask),
            exponent_mask));
    }
    if (vmaxvq_u32(nonfinite) != 0u) return 0;
    *maximum_out = vmaxvq_f32(maximum);
    for (; column < width; column++) {
        if (!__builtin_isfinite(input[column])) return 0;
        if (input[column] > *maximum_out) *maximum_out = input[column];
    }
    return 1;
}

static void vx_softmax_f32_finite_row_neon(
        const float* input, float* output, int width, float maximum) {
    const float32x4_t maximum_lanes = vdupq_n_f32(maximum);
    const float32x4_t minimum_delta = vdupq_n_f32(-88.0f);
    float sum = 0.0f;
    int column = 0;
    /* Four independent exponent chains give an out-of-order Arm core enough
     * work to cover the FMLA latency in the range-reduction polynomial.  The
     * scalar additions below deliberately remain in element order, preserving
     * the canonical denominator bit-for-bit. */
    for (; column + 16 <= width; column += 16) {
        float32x4_t exponential[4];
        for (int block = 0; block < 4; block++) {
            float32x4_t delta = vsubq_f32(
                vld1q_f32(input + column + block * 4), maximum_lanes);
            delta = vmaxq_f32(delta, minimum_delta);
            exponential[block] = vx_accurate_exp_neon(delta);
            vst1q_f32(output + column + block * 4, exponential[block]);
        }
        for (int block = 0; block < 4; block++) {
            sum += vgetq_lane_f32(exponential[block], 0);
            sum += vgetq_lane_f32(exponential[block], 1);
            sum += vgetq_lane_f32(exponential[block], 2);
            sum += vgetq_lane_f32(exponential[block], 3);
        }
    }
    for (; column + 4 <= width; column += 4) {
        float32x4_t delta = vsubq_f32(
            vld1q_f32(input + column), maximum_lanes);
        const float32x4_t exponential =
            vx_accurate_exp_neon(vmaxq_f32(delta, minimum_delta));
        vst1q_f32(output + column, exponential);
        sum += vgetq_lane_f32(exponential, 0);
        sum += vgetq_lane_f32(exponential, 1);
        sum += vgetq_lane_f32(exponential, 2);
        sum += vgetq_lane_f32(exponential, 3);
    }
    for (; column < width; column++) {
        output[column] = accurate_expf(input[column] - maximum);
        sum += output[column];
    }
    {
        const float32x4_t denominator = vdupq_n_f32(sum);
        column = 0;
        for (; column + 16 <= width; column += 16) {
            const float32x4_t value0 = vld1q_f32(output + column);
            const float32x4_t value1 = vld1q_f32(output + column + 4);
            const float32x4_t value2 = vld1q_f32(output + column + 8);
            const float32x4_t value3 = vld1q_f32(output + column + 12);
            vst1q_f32(output + column, vdivq_f32(value0, denominator));
            vst1q_f32(output + column + 4,
                vdivq_f32(value1, denominator));
            vst1q_f32(output + column + 8,
                vdivq_f32(value2, denominator));
            vst1q_f32(output + column + 12,
                vdivq_f32(value3, denominator));
        }
        for (; column + 4 <= width; column += 4) {
            vst1q_f32(output + column,
                vdivq_f32(vld1q_f32(output + column), denominator));
        }
    }
    for (; column < width; column++) output[column] /= sum;
}
#endif

#if VX_FASTMATH_X86_AVX2
/* Keep the scalar maximum and sum reductions in their original left-to-right
 * order. Only the lane-independent exponential and normalization work is
 * vectorized, so the AVX2 tier stays byte-identical to the scalar tier for
 * finite rows. Non-finite rows never enter this helper. */
static VX_FASTMATH_TARGET_AVX2 void vx_softmax_f32_finite_row_avx2(
        const float* input, float* output, int width, float maximum) {
    const __m256 maximum_lanes = _mm256_set1_ps(maximum);
    float sum = 0.0f;
    int column = 0;
    for (; column + 8 <= width; column += 8) {
        __m256 delta = _mm256_sub_ps(_mm256_loadu_ps(input + column),
                                     maximum_lanes);
        /* Scalar accurate_expf returns zero before range reduction below -87.
         * Keeping those lanes finite avoids a spurious invalid conversion for
         * an extreme finite subtraction that rounds to negative infinity. */
        delta = _mm256_max_ps(delta, _mm256_set1_ps(-88.0f));
        _mm256_storeu_ps(output + column, vx_accurate_exp_avx2(delta));
        sum += output[column + 0];
        sum += output[column + 1];
        sum += output[column + 2];
        sum += output[column + 3];
        sum += output[column + 4];
        sum += output[column + 5];
        sum += output[column + 6];
        sum += output[column + 7];
    }
    for (; column < width; column++) {
        output[column] = accurate_expf(input[column] - maximum);
        sum += output[column];
    }
    {
        const __m256 denominator = _mm256_set1_ps(sum);
        column = 0;
        for (; column + 8 <= width; column += 8) {
            _mm256_storeu_ps(output + column, _mm256_div_ps(
                _mm256_loadu_ps(output + column), denominator));
        }
    }
    for (; column < width; column++) output[column] /= sum;
}
#endif

#if VX_FASTMATH_WASM_SIMD
/* The maximum and denominator reductions retain scalar left-to-right order;
 * only the lane-independent exponential and normalization use SIMD128.  This
 * mirrors the AVX2 route while avoiding four scalar polynomial evaluations per
 * vector on WASM. Non-finite rows remain on the scalar contract below. */
static void vx_softmax_f32_finite_row_wasm_simd(
        const float* input, float* output, int width, float maximum) {
    const v128_t maximum_lanes = wasm_f32x4_splat(maximum);
    float sum = 0.0f;
    int column = 0;
    for (; column + 4 <= width; column += 4) {
        v128_t delta = wasm_f32x4_sub(
            wasm_v128_load(input + column), maximum_lanes);
        delta = wasm_f32x4_max(delta, wasm_f32x4_splat(-88.0f));
        wasm_v128_store(output + column,
            vx_accurate_exp_wasm_simd(delta));
        sum += output[column + 0];
        sum += output[column + 1];
        sum += output[column + 2];
        sum += output[column + 3];
    }
    for (; column < width; column++) {
        output[column] = accurate_expf(input[column] - maximum);
        sum += output[column];
    }
    {
        const v128_t denominator = wasm_f32x4_splat(sum);
        column = 0;
        for (; column + 4 <= width; column += 4) {
            wasm_v128_store(output + column, wasm_f32x4_div(
                wasm_v128_load(output + column), denominator));
        }
    }
    for (; column < width; column++) output[column] /= sum;
}
#endif

static void vx_softmax_f32_rows(void* opaque, int begin, int end) {
    VxSoftmaxF32RowsContext* context =
        (VxSoftmaxF32RowsContext*)opaque;
    const int d = context->width;
    for (int i = begin; i < end; i++) {
        const size_t offset = (size_t)i * (size_t)d;
        const float* input = context->input + offset;
        float* output = context->output + offset;
        float max_val = input[0];
#if VX_FASTMATH_WASM_SIMD
        if (d >= 4) {
            int finite = __builtin_isfinite(max_val);
            for (int j = 1; j < d; j++) {
                if (!__builtin_isfinite(input[j])) finite = 0;
                if (input[j] > max_val) max_val = input[j];
            }
            if (finite) {
                vx_softmax_f32_finite_row_wasm_simd(
                    input, output, d, max_val);
                continue;
            }
        } else {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#elif VX_FASTMATH_X86_AVX2
        if (context->use_avx2 && d >= 8) {
            int finite = __builtin_isfinite(max_val);
            for (int j = 1; j < d; j++) {
                if (!__builtin_isfinite(input[j])) finite = 0;
                if (input[j] > max_val) max_val = input[j];
            }
            if (finite) {
                vx_softmax_f32_finite_row_avx2(input, output, d, max_val);
                continue;
            }
        } else {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#elif VX_FASTMATH_ARM_NEON
        if (context->use_neon && d >= 4) {
            if (vx_softmax_f32_finite_max_neon(
                    input, d, &max_val)) {
                vx_softmax_f32_finite_row_neon(input, output, d, max_val);
                continue;
            }
            max_val = input[0];
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        } else {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#else
        {
            for (int j = 1; j < d; j++) {
                if (input[j] > max_val) max_val = input[j];
            }
        }
#endif
        float sum = 0.0f;
        for (int j = 0; j < d; j++) {
            output[j] = accurate_expf(input[j] - max_val);
            sum += output[j];
        }
        for (int j = 0; j < d; j++) {
            output[j] /= sum;
        }
    }
}

void softmax_f32(const float* input, float* output, int b, int d) {
    VxSoftmaxF32RowsContext context;
    int threads;
    int grain;
    if (!input || !output || b <= 0 || d <= 0) return;
    context.input = input;
    context.output = output;
    context.width = d;
#if VX_FASTMATH_X86_AVX2
    context.use_avx2 = vx_kernel_platform()->has_avx2;
#else
    context.use_avx2 = 0;
#endif
#if VX_FASTMATH_ARM_NEON
    context.use_neon = vx_kernel_platform()->has_neon;
#else
    context.use_neon = 0;
#endif
    threads = vx_kernels_thread_count();
    /* Keep short row batches on the caller and split only when the exponential
     * work amortizes a pool wake-up. Every worker owns complete rows, so each
     * reduction retains the scalar operation order and 1T/NT results remain
     * byte-identical. */
    if (threads <= 1 || b <= 1 ||
        (uint64_t)(uint32_t)b * (uint64_t)(uint32_t)d < 65536u) {
        vx_softmax_f32_rows(&context, 0, b);
        return;
    }
    {
        const uint64_t target_chunks = (uint64_t)(uint32_t)threads * 4u;
        grain = (int)((uint64_t)(uint32_t)b / target_chunks +
                      ((uint64_t)(uint32_t)b % target_chunks != 0u));
    }
    if (grain < 1) grain = 1;
    vx_kernels_parallel_for(b, grain, vx_softmax_f32_rows, &context);
}
