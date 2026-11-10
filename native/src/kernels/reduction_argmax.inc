/* ReduceMean / ReduceSum / ArgMax, with the typed element accessors they
 * share for dtype-generic axis walks. */

typedef double VxReduceF64x2 __attribute__((vector_size(16)));

typedef struct VxOrderedReductionRows {
    const float* input;
    float* output;
    size_t row_offset;
    int width;
    float scale;
} VxOrderedReductionRows;

static void vx_reduce_ordered_rows(void* opaque, int begin, int end) {
    const VxOrderedReductionRows* context = (const VxOrderedReductionRows*)opaque;
    const size_t width = (size_t)context->width;
    size_t row = context->row_offset + (size_t)begin;
    const size_t stop = context->row_offset + (size_t)end;
    /* Each lane is a different row. Columns still accumulate left to right
     * in double precision, including the final F32 cast before scaling.
     * Two independent vector chains hide add latency without reassociation.
     * This maps to F64 SIMD128 on WASM and the native vector extensions. */
    for (; stop - row >= 4u; row += 4u) {
        const float* input0 = context->input + row * width;
        const float* input1 = input0 + width;
        const float* input2 = input1 + width;
        const float* input3 = input2 + width;
        double sum0 = 0.0;
        double sum1 = 0.0;
        double sum2 = 0.0;
        double sum3 = 0.0;
        for (size_t column = 0; column < width; column++) {
            sum0 += (double)input0[column];
            sum1 += (double)input1[column];
            sum2 += (double)input2[column];
            sum3 += (double)input3[column];
        }
        context->output[row] = (float)sum0 * context->scale;
        context->output[row + 1u] = (float)sum1 * context->scale;
        context->output[row + 2u] = (float)sum2 * context->scale;
        context->output[row + 3u] = (float)sum3 * context->scale;
    }
    for (; row < stop; row++) {
        double sum = 0.0;
        for (size_t column = 0; column < width; column++)
            sum += context->input[row * width + column];
        context->output[row] = (float)sum * context->scale;
    }
}

void reduce_last_axis_ordered_f32(const float* input, float* output,
                                size_t rows, int width, float scale) {
    VxOrderedReductionRows context = {input, output, 0u, width, scale};
    size_t input_bytes, output_bytes;
    uintptr_t input_address, output_address;
    int threads;
    if (!input || !output || !rows || width <= 0 ||
        rows > SIZE_MAX / sizeof(float) / (size_t)width) return;
    input_bytes = rows * (size_t)width * sizeof(float);
    output_bytes = rows * sizeof(float);
    input_address = (uintptr_t)input;
    output_address = (uintptr_t)output;
    /* Keep the old ordered traversal if a private caller overlaps the input
     * and output. Parallel row writes could otherwise overwrite unread rows. */
    if ((output_address >= input_address &&
         output_address - input_address < input_bytes) ||
        (input_address > output_address &&
         input_address - output_address < output_bytes)) {
        for (size_t row = 0; row < rows; row++) {
            double sum = 0.0;
            for (int column = 0; column < width; column++)
                sum += input[row * (size_t)width + (size_t)column];
            output[row] = (float)sum * scale;
        }
        return;
    }
    threads = vx_kernels_thread_count();
    /* The pool ABI uses int ranges. Bound each synchronous dispatch before
     * narrowing; row_offset retains the full tensor extent. */
    while (context.row_offset < rows) {
        size_t remaining = rows - context.row_offset;
        int count = remaining > INT_MAX ? INT_MAX : (int)remaining;
        if (threads > 1 && count > 1 &&
            (uint64_t)(uint32_t)count * (uint64_t)(uint32_t)width >= 65536u) {
            uint64_t chunks = (uint64_t)(uint32_t)threads * 4u;
            uint64_t grain = ((uint64_t)(uint32_t)count + chunks - 1u) / chunks;
            grain = (grain + 3u) & ~UINT64_C(3);
            if (grain < 4u) grain = 4u;
            if (grain > INT_MAX) grain = INT_MAX;
            vx_kernels_parallel_for(count, (int)grain,
                                    vx_reduce_ordered_rows, &context);
        } else {
            vx_reduce_ordered_rows(&context, 0, count);
        }
        context.row_offset += (size_t)count;
    }
}

void reduce_mean_f32(const float* input, float* output, int b, int d) {
    for (int i = 0; i < b; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += input[i*d + j];
        output[i] = sum / (float)d;
    }
}
void reduce_sum_f32(const float* input, float* output, int b, int d) {
    for (int i = 0; i < b; i++) {
        float sum = 0.0f;
        for (int j = 0; j < d; j++) sum += input[i*d + j];
        output[i] = sum;
    }
}

/* Ordered scans use F32 rounding at each addition and modulo-2^32 I32
 * arithmetic on every backend. Each lane is independent, with no scratch. */
static int cumsum_geometry(const int* shape, int ndim, int axis,
                           size_t* outer, size_t* length, size_t* inner) {
    size_t elements = 1;
    if (!shape || ndim <= 0) return -1;
    if (axis < 0) axis += ndim;
    if (axis < 0 || axis >= ndim) return -1;
    *outer = *inner = 1;
    for (int i = 0; i < ndim; ++i) {
        if (shape[i] <= 0 || elements > SIZE_MAX / sizeof(float) / (size_t)shape[i]) return -1;
        elements *= (size_t)shape[i];
        if (i < axis) *outer *= (size_t)shape[i];
        if (i > axis) *inner *= (size_t)shape[i];
    }
    *length = (size_t)shape[axis];
    return 0;
}

int cumsum_forward_f32(const float* input, float* output, const int* shape,
                      int ndim, int axis, int exclusive, int reverse) {
    size_t outer, length, inner;
    if (!input || !output || (exclusive != 0 && exclusive != 1) ||
        (reverse != 0 && reverse != 1) ||
        cumsum_geometry(shape, ndim, axis, &outer, &length, &inner)) return -1;
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            float accum = 0.0f;
            for (size_t t = 0; t < length; ++t) {
                size_t position = reverse ? length - 1 - t : t;
                size_t idx = (o * length + position) * inner + i;
                float value = input[idx];
                if (exclusive) output[idx] = accum;
                accum += value;
                if (!exclusive) output[idx] = accum;
            }
        }
    }
    return 0;
}

int cumsum_forward_i32(const int32_t* input, int32_t* output, const int* shape,
                      int ndim, int axis, int exclusive, int reverse) {
    size_t outer, length, inner;
    if (!input || !output || (exclusive != 0 && exclusive != 1) ||
        (reverse != 0 && reverse != 1) ||
        cumsum_geometry(shape, ndim, axis, &outer, &length, &inner)) return -1;
    for (size_t o = 0; o < outer; ++o) {
        for (size_t i = 0; i < inner; ++i) {
            uint32_t accum = 0;
            for (size_t t = 0; t < length; ++t) {
                size_t position = reverse ? length - 1 - t : t;
                size_t idx = (o * length + position) * inner + i;
                uint32_t value = (uint32_t)input[idx];
                if (!exclusive) accum += value;
                memcpy(output + idx, &accum, sizeof(accum));
                if (exclusive) accum += value;
            }
        }
    }
    return 0;
}

static double vx_kernel_value_at(const void* values, int dtype, long index) {
    switch (dtype) {
        case VX_DTYPE_F32: return ((const float*)values)[index];
        case VX_DTYPE_I32: return ((const int32_t*)values)[index];
        case VX_DTYPE_I8: return ((const int8_t*)values)[index];
        case VX_DTYPE_U8: return ((const uint8_t*)values)[index];
        default: return 0.0;
    }
}

static int vx_kernel_dtype_supported(int dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

static void vx_kernel_store_index(void* values, int dtype, long index, int value) {
    switch (dtype) {
        case VX_DTYPE_F32:
            ((float*)values)[index] = (float)value;
            break;
        case VX_DTYPE_I32:
            ((int32_t*)values)[index] = (int32_t)value;
            break;
        case VX_DTYPE_I8:
        case VX_DTYPE_U8:
            /* Store the byte representation directly so Int8Array and
             * Uint8Array retain JavaScript TypedArray wrapping semantics. */
            ((uint8_t*)values)[index] = (uint8_t)value;
            break;
    }
}

/* General ONNX-style ArgMax indexing: [outer, axis, inner].  Values are read
 * and indices are written with the tensor's actual WASM storage dtype. */
void argmax_axis_typed(const void* input, void* output, int outer, int axis_size,
                       int inner, int input_dtype, int output_dtype) {
    if (!input || !output || outer <= 0 || axis_size <= 0 || inner <= 0 ||
        !vx_kernel_dtype_supported(input_dtype) ||
        !vx_kernel_dtype_supported(output_dtype)) return;
    for (int outer_index = 0; outer_index < outer; outer_index++) {
        for (int inner_index = 0; inner_index < inner; inner_index++) {
            long base = ((long)outer_index * axis_size * inner) + inner_index;
            double best = vx_kernel_value_at(input, input_dtype, base);
            int best_index = 0;
            for (int axis_index = 1; axis_index < axis_size; axis_index++) {
                double value = vx_kernel_value_at(input, input_dtype,
                    base + (long)axis_index * inner);
                /* Strict comparison deliberately preserves the first index on
                 * ties and follows JavaScript's NaN comparison behavior. */
                if (value > best) {
                    best = value;
                    best_index = axis_index;
                }
            }
            vx_kernel_store_index(output, output_dtype,
                (long)outer_index * inner + inner_index, best_index);
        }
    }
}

void argmax_f32(const float* input, float* output, int b, int d) {
    argmax_axis_typed(input, output, b, d, 1,
        VX_DTYPE_F32, VX_DTYPE_F32);
}

static double vx_kernel_max(double left, double right) {
    /* Math.max returns NaN when either input is NaN. */
    if (left != left || right != right) return left + right;
    return left > right ? left : right;
}

static double vx_kernel_min(double left, double right) {
    /* Math.min returns NaN when either input is NaN. */
    if (left != left || right != right) return left + right;
    return left < right ? left : right;
}

