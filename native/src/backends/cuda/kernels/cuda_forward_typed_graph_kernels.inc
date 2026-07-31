extern "C" VX_CUDA_GLOBAL void vx_cuda_batch_matmul_f32(
        const float* a, const float* b, float* output, vx_u32 elements,
        VxCudaBatchMatMulParams params) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;

    vx_u32 matrix_elements = params.m * params.n;
    vx_u32 batch_index = index / matrix_elements;
    vx_u32 matrix_index = index - batch_index * matrix_elements;
    vx_u32 row = matrix_index / params.n;
    vx_u32 column = matrix_index - row * params.n;
    vx_u32 a_batch_offset = 0u;
    vx_u32 b_batch_offset = 0u;
    vx_u32 remainder = batch_index;
    for (vx_u32 dimension = 0u;
         dimension < params.batch_rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_batch_strides[dimension];
        remainder -=
            coordinate * params.output_batch_strides[dimension];
        a_batch_offset += coordinate * params.a_batch_strides[dimension];
        b_batch_offset += coordinate * params.b_batch_strides[dimension];
    }

    float sum = 0.0f;
    vx_u32 a_row = a_batch_offset + row * params.k;
    vx_u32 b_column = b_batch_offset + column;
    for (vx_u32 inner = 0u; inner < params.k; inner++) {
        sum = vx_add_rn(sum, vx_mul_rn(
            a[a_row + inner],
            b[b_column + inner * params.n]));
    }
    output[index] = sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_qbatch_matmul_i8u8(
        const vx_u8* a, const vx_u8* b, vx_u8* output, vx_u32 elements,
        VxCudaBatchMatMulParams params,
        float a_scale, vx_i32 a_zero_point,
        float b_scale, vx_i32 b_zero_point,
        float output_scale, vx_i32 output_zero_point,
        vx_u32 a_dtype, vx_u32 b_dtype, vx_u32 output_dtype) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;

    vx_u32 matrix_elements = params.m * params.n;
    vx_u32 batch_index = index / matrix_elements;
    vx_u32 matrix_index = index - batch_index * matrix_elements;
    vx_u32 row = matrix_index / params.n;
    vx_u32 column = matrix_index - row * params.n;
    vx_u32 a_batch_offset = 0u;
    vx_u32 b_batch_offset = 0u;
    vx_u32 remainder = batch_index;
    for (vx_u32 dimension = 0u;
         dimension < params.batch_rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_batch_strides[dimension];
        remainder -=
            coordinate * params.output_batch_strides[dimension];
        a_batch_offset += coordinate * params.a_batch_strides[dimension];
        b_batch_offset += coordinate * params.b_batch_strides[dimension];
    }

    vx_i32 accumulator = 0;
    vx_u32 a_row = a_batch_offset + row * params.k;
    vx_u32 b_column = b_batch_offset + column;
    for (vx_u32 inner = 0u; inner < params.k; inner++) {
        vx_i32 a_value =
            vx_typed_byte(a, a_row + inner, a_dtype) - a_zero_point;
        vx_i32 b_value =
            vx_typed_byte(
                b, b_column + inner * params.n, b_dtype) - b_zero_point;
        accumulator += a_value * b_value;
    }
    float multiplier = vx_mul_rn(a_scale, b_scale) / output_scale;
    float transformed = vx_add_rn(
        vx_mul_rn((float)accumulator, multiplier),
        (float)output_zero_point);
    output[index] = vx_quantize_transformed(
        transformed, output_zero_point, output_dtype);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_argmax_f32_i32(
        const float* input, vx_i32* output, vx_u32 outer,
        vx_u32 axis_size, vx_u32 inner) {
    vx_u32 output_index = vx_global_x();
    if (output_index >= outer * inner) return;

    vx_u32 outer_index = output_index / inner;
    vx_u32 inner_index = output_index - outer_index * inner;
    vx_u32 base = outer_index * axis_size * inner + inner_index;
    float best = input[base];
    vx_i32 best_index = 0;
    for (vx_u32 axis_index = 1u; axis_index < axis_size; axis_index++) {
        float value = input[base + axis_index * inner];
        /* Strict comparison preserves the first index on ties and matches
         * CPU/JavaScript NaN comparison behavior. */
        if (value > best) {
            best = value;
            best_index = (vx_i32)axis_index;
        }
    }
    output[output_index] = best_index;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_compare_i32(
        const vx_i32* a, const vx_i32* b, vx_i32* output,
        vx_u32 elements, VxCudaBroadcastParams params, vx_i32 op) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 a_index = 0u;
    vx_u32 b_index = 0u;
    vx_u32 remainder = index;
    for (vx_u32 dimension = 0u; dimension < params.rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        a_index += coordinate * params.a_strides[dimension];
        b_index += coordinate * params.b_strides[dimension];
    }
    if (op == 1)
        output[index] = a[a_index] >= b[b_index] ? 1 : 0;
    else
        output[index] = a[a_index] == b[b_index] ? 1 : 0;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_not_i32(
        const vx_i32* input, vx_i32* output, vx_u32 elements) {
    vx_u32 index = vx_global_x();
    if (index < elements) output[index] = input[index] == 0 ? 1 : 0;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_clip_i32(
        const vx_i32* input, vx_i32* output, vx_u32 elements,
        vx_i32 minimum, vx_i32 maximum) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_i32 value = input[index];
    if (value < minimum) value = minimum;
    if (value > maximum) value = maximum;
    output[index] = value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_where_32(
        const vx_i32* condition, const vx_u32* a, const vx_u32* b,
        vx_u32* output, vx_u32 elements) {
    vx_u32 index = vx_global_x();
    if (index < elements)
        output[index] = condition[index] != 0 ? a[index] : b[index];
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_f32_bits(float value) {
    vx_u32 bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}

/* Match the portable/TypedArray F32 -> I32 contract: non-finite values become
 * zero, finite values truncate toward zero, then wrap modulo 2^32. */
static VX_CUDA_DEVICE vx_i32 vx_cuda_f32_to_i32(float value) {
    vx_u32 bits = vx_cuda_f32_bits(value);
    vx_u32 exponent = (bits >> 23u) & 0xffu;
    if (exponent == 0xffu || exponent < 127u) return 0;
    vx_u32 significand = (bits & 0x7fffffu) | 0x800000u;
    vx_i32 shift = (vx_i32)exponent - 127 - 23;
    vx_u32 magnitude;
    if (shift >= 32) magnitude = 0u;
    else if (shift >= 0) magnitude = significand << (vx_u32)shift;
    else magnitude = significand >> (vx_u32)(-shift);
    vx_u32 wrapped = (bits & 0x80000000u) ? 0u - magnitude : magnitude;
    vx_i32 result;
    asm("mov.b32 %0, %1;" : "=r"(result) : "r"(wrapped));
    return result;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_cast_typed(
        const void* input, void* output, vx_u32 elements,
        vx_i32 input_dtype, vx_i32 output_dtype) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    if (input_dtype == VX_DTYPE_F32 && output_dtype == VX_DTYPE_F32) {
        ((float*)output)[index] = ((const float*)input)[index];
        return;
    }
    vx_i32 integer;
    if (input_dtype == VX_DTYPE_F32)
        integer = vx_cuda_f32_to_i32(((const float*)input)[index]);
    else if (input_dtype == VX_DTYPE_I32)
        integer = ((const vx_i32*)input)[index];
    else if (input_dtype == VX_DTYPE_I8)
        integer = (vx_i32)((const vx_i8*)input)[index];
    else
        integer = (vx_i32)((const vx_u8*)input)[index];
    if (output_dtype == VX_DTYPE_F32)
        ((float*)output)[index] = (float)integer;
    else if (output_dtype == VX_DTYPE_I32)
        ((vx_i32*)output)[index] = integer;
    else
        ((vx_u8*)output)[index] = (vx_u8)integer;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_copy_32(
        const vx_u32* input, vx_u32* output, vx_u32 elements) {
    vx_u32 index = vx_global_x();
    if (index < elements) output[index] = input[index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_transpose_32(
        const vx_u32* input, vx_u32* output, vx_u32 elements,
        VxCudaTransposeParams params) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 remainder = index;
    vx_u32 input_index = 0u;
    for (vx_u32 dimension = 0u; dimension < params.rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        input_index += coordinate * params.input_strides[dimension];
    }
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_transpose_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 elements,
        VxCudaTransposeParams params) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 remainder = index;
    vx_u32 input_index = 0u;
    for (vx_u32 dimension = 0u; dimension < params.rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        input_index += coordinate * params.input_strides[dimension];
    }
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_expand_32(
        const vx_u32* input, vx_u32* output, vx_u32 elements,
        VxCudaTransposeParams params) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 remainder = index;
    vx_u32 input_index = 0u;
    for (vx_u32 dimension = 0u; dimension < params.rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        input_index += coordinate * params.input_strides[dimension];
    }
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_slice_32(
        const vx_u32* input, vx_u32* output, vx_u32 elements,
        VxCudaSliceParams params) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 remainder = index;
    vx_u32 input_index = 0u;
    for (vx_u32 dimension = 0u; dimension < params.rank; dimension++) {
        vx_u32 coordinate =
            remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        input_index +=
            (params.starts[dimension] +
             coordinate * params.steps[dimension]) *
            params.input_strides[dimension];
    }
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_split_32(
        const vx_u32* input, vx_u32* output, vx_u32 elements,
        vx_u32 inner, vx_u32 split_size, vx_u32 input_axis,
        vx_u32 axis_offset) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 inner_index = index % inner;
    vx_u32 split_index = (index / inner) % split_size;
    vx_u32 outer_index = index / (split_size * inner);
    vx_u32 input_index = outer_index * input_axis * inner +
        (axis_offset + split_index) * inner + inner_index;
    output[index] = input[input_index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_concat_32(
        const vx_u32* input, vx_u32* output, vx_u32 elements,
        vx_u32 axis_offset, vx_u32 input_axis, vx_u32 output_axis,
        vx_u32 inner) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 inner_index = index % inner;
    vx_u32 outer_axis = index / inner;
    vx_u32 axis_index = outer_axis % input_axis;
    vx_u32 outer_index = outer_axis / input_axis;
    vx_u32 output_index = ((outer_index * output_axis + axis_offset +
        axis_index) * inner) + inner_index;
    output[output_index] = input[index];
}
