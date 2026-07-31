extern "C" VX_CUDA_GLOBAL void vx_cuda_copy_f32(
        const float* input, float* output, vx_u32 elements) {
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
        vx_i32 op, float alpha, float beta) {
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
        vx_u32 rows, vx_u32 width, float epsilon) {
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
        vx_u32 token_count, vx_u32 width) {
    vx_u32 index = vx_global_x();
    vx_u32 elements = token_count * width;
    if (index >= elements) return;
    vx_u32 token = index / width;
    vx_u32 column = index - token * width;
    output[index] = weight[(vx_u32)tokens[token] * width + column];
}
