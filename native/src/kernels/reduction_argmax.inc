/* ReduceMean / ReduceSum / ArgMax, with the typed element accessors they
 * share for dtype-generic axis walks. */

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

