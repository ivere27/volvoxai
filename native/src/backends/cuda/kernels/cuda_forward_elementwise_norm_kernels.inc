extern "C" VX_CUDA_GLOBAL void vx_cuda_copy_f32(
        const float* input, float* output, vx_u32 elements,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, output, elements);
    vx_u32 index = vx_global_x();
    if (index < elements) output[index] = input[index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_copy_i8u8(
        const vx_u8* input, vx_u8* output, vx_u32 elements) {
    vx_u32 index = vx_global_x();
    if (index < elements) output[index] = input[index];
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_binary_f32(
        const float* a, const float* b, float* output, vx_u32 elements,
        VxCudaBroadcastParams params, vx_i32 op) {
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    vx_u32 a_index = 0;
    vx_u32 b_index = 0;
    vx_u32 remainder = index;
    for (vx_u32 dimension = 0; dimension < params.rank; dimension++) {
        vx_u32 coordinate = remainder / params.output_strides[dimension];
        remainder -= coordinate * params.output_strides[dimension];
        a_index += coordinate * params.a_strides[dimension];
        b_index += coordinate * params.b_strides[dimension];
    }
    float av = a[a_index];
    float bv = b[b_index];
    if (op == 1) output[index] = av - bv;
    else if (op == 2) output[index] = av / bv;
    else if (op == 3) output[index] = av + bv;
    else output[index] = av * bv;
}

/* Unary operation IDs are private to cuda_engine.c. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_unary_f32(
        const float* input, float* output, vx_u32 elements,
        vx_i32 op, float alpha, float beta,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, output, elements, op, alpha, beta);
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float x = input[index];
    float y = x;
    if (op == 0) y = vx_max(x, 0.0f);
    else if (op == 1) y = vx_sigmoid(x);
    else if (op == 2) {
        if (alpha != 0.0f) {
            float c = 0.7978845608028654f * (x + 0.044715f * x * x * x);
            y = 0.5f * x * (1.0f + vx_tanh(c));
        } else {
            y = 0.5f * x *
                (1.0f + vx_erf_approx(x * 0.7071067811865476f));
        }
    } else if (op == 3) y = x * vx_sigmoid(x);
    else if (op == 4) y = vx_tanh(x);
    else if (op == 5) y = x * vx_min(6.0f, vx_max(0.0f, x + 3.0f)) / 6.0f;
    else if (op == 6) {
        float v = x + 3.0f;
        if (v < 0.0f) v = 0.0f;
        if (v > 6.0f) v = 6.0f;
        y = v / 6.0f;
    } else if (op == 7) y = x > 0.0f ? x : alpha * x;
    else if (op == 8) y = vx_min(beta, vx_max(alpha, x));
    output[index] = y;
}

/* Preserve both semantic node outputs: gate/up remain unchanged and the
 * separately observable SiLU activation is stored before its Mul result.
 * This uses exactly the ordinary SiLU and Mul F32 arithmetic. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_silu_mul_dual_f32(
        const float* input, const float* other, float* activation,
        float* output, vx_u32 elements,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, other, activation, output, elements);
    vx_u32 index = vx_global_x();
    if (index >= elements) return;
    float value = input[index];
    float activated = value * vx_sigmoid(value);
    float product = activated * other[index];
    activation[index] = activated;
    output[index] = product;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_layernorm_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 width, float epsilon) {
    vx_u32 row = vx_global_x();
    if (row >= rows) return;
    vx_u32 base = row * width;
    float mean = 0.0f;
    for (vx_u32 column = 0; column < width; column++) mean += input[base + column];
    mean /= (float)width;
    float variance = 0.0f;
    for (vx_u32 column = 0; column < width; column++) {
        float delta = input[base + column] - mean;
        variance += delta * delta;
    }
    float inverse = 1.0f / vx_sqrt(variance / (float)width + epsilon);
    for (vx_u32 column = 0; column < width; column++)
        output[base + column] = (input[base + column] - mean) * inverse *
            weight[column] + (bias ? bias[column] : 0.0f);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_rmsnorm_f32(
        const float* input, const float* weight, float* output,
        vx_u32 rows, vx_u32 width, float epsilon,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, output, rows, width, epsilon);
    vx_u32 row = vx_global_x();
    if (row >= rows) return;
    vx_u32 base = row * width;
    float sum_squares = 0.0f;
    for (vx_u32 column = 0; column < width; column++) {
        float value = input[base + column];
        sum_squares += value * value;
    }
    float inverse = 1.0f / vx_sqrt(sum_squares / (float)width + epsilon);
    for (vx_u32 column = 0; column < width; column++)
        output[base + column] = input[base + column] * inverse * weight[column];
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_cuda_norm_shuffle_down(
        float value, vx_u32 delta) {
    float shuffled;
    asm("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=f"(shuffled) : "f"(value), "r"(delta));
    return shuffled;
}

/* One block owns a row. The host admits 32, 64, 128 or 256 threads and
 * proves the 32 bytes of static shared storage separately from the launch's
 * zero dynamic shared bytes. All lanes participate, including width tails.
 * Strict F32 multiply/add and sqrt remain separate; only the positive-square
 * summation order changes. The second barrier also permits exact in-place
 * input/output: no lane writes before the complete row has been reduced. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_rmsnorm_cooperative_f32(
        const float* input, const float* weight, float* output,
        vx_u32 rows, vx_u32 width, float epsilon,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, output, rows, width, epsilon);
    VX_CUDA_SHARED float warp_sums[8];
    vx_u32 row = blockIdx.x;
    if (row >= rows) return;
    vx_u32 lane = threadIdx.x;
    vx_u32 base = row * width;
    float sum_squares = 0.0f;
    for (vx_u32 column = lane; column < width; column += blockDim.x) {
        float value = input[base + column];
        sum_squares = vx_add_rn(sum_squares, vx_mul_rn(value, value));
    }
    for (vx_u32 delta = 16u; delta; delta /= 2u)
        sum_squares = vx_add_rn(
            sum_squares, vx_cuda_norm_shuffle_down(sum_squares, delta));
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = sum_squares;
    vx_sync_threads();
    if (lane == 0u) {
        float total = warp_sums[0];
        for (vx_u32 warp = 1u; warp < blockDim.x / 32u; ++warp)
            total = vx_add_rn(total, warp_sums[warp]);
        warp_sums[0] = 1.0f / vx_sqrt(total / (float)width + epsilon);
    }
    vx_sync_threads();
    float inverse = warp_sums[0];
    for (vx_u32 column = lane; column < width; column += blockDim.x)
        output[base + column] = vx_mul_rn(
            vx_mul_rn(input[base + column], inverse), weight[column]);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_softmax_f32(
        const float* input, float* output, vx_u32 rows, vx_u32 width,
        vx_i32 logarithmic) {
    vx_u32 row = vx_global_x();
    if (row >= rows) return;
    vx_u32 base = row * width;
    float maximum = input[base];
    for (vx_u32 column = 1; column < width; column++)
        maximum = vx_max(maximum, input[base + column]);
    float sum = 0.0f;
    for (vx_u32 column = 0; column < width; column++)
        sum += vx_exp(input[base + column] - maximum);
    if (logarithmic) {
        /* log(sum) = ln(2) * log2(sum). */
        float log2_sum;
        asm("lg2.approx.f32 %0, %1;" : "=f"(log2_sum) : "f"(sum));
        float normalizer = maximum + 0.6931471805599453f * log2_sum;
        for (vx_u32 column = 0; column < width; column++)
            output[base + column] = input[base + column] - normalizer;
    } else {
        float inverse = 1.0f / sum;
        for (vx_u32 column = 0; column < width; column++)
            output[base + column] = vx_exp(input[base + column] - maximum) * inverse;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_embedding_f32(
        const vx_i32* tokens, const float* weight, float* output,
        vx_u32 token_count, vx_u32 width,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        tokens, weight, output, token_count, width);
    vx_u32 index = vx_global_x();
    vx_u32 elements = token_count * width;
    if (index >= elements) return;
    vx_u32 token = index / width;
    vx_u32 column = index - token * width;
    output[index] = weight[(vx_u32)tokens[token] * width + column];
}
