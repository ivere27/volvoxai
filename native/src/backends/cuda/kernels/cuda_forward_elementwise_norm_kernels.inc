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
    float av, bv;
    if (params.rank == 0u) {
        av = a[index];
        bv = b[index];
    } else if (params.rank == 254u) {
        av = a[index];
        bv = b[0];
    } else if (params.rank == 255u) {
        av = a[0];
        bv = b[index];
    } else {
        vx_u32 a_index = 0;
        vx_u32 b_index = 0;
        vx_u32 remainder = index;
        vx_u32 last_dim = params.rank - 1u;
        for (vx_u32 dimension = 0; dimension < last_dim; dimension++) {
            vx_u32 coordinate = remainder / params.output_strides[dimension];
            remainder -= coordinate * params.output_strides[dimension];
            a_index += coordinate * params.a_strides[dimension];
            b_index += coordinate * params.b_strides[dimension];
        }
        a_index += remainder * params.a_strides[last_dim];
        b_index += remainder * params.b_strides[last_dim];
        av = a[a_index];
        bv = b[b_index];
    }
    if (op == 1) output[index] = av - bv;
    else if (op == 2) output[index] = av / bv;
    else if (op == 3) output[index] = av + bv;
    else output[index] = av * bv;
}

/* Standalone Exp has the full IEEE F32 range; the clamped helper used by
 * softmax/sigmoid is intentionally a different numerical operation. Double
 * range reduction keeps the result accurate at F32 overflow/subnormal edges. */
static VX_CUDA_DEVICE float vx_cuda_exp_full_f32(float value) {
    if (value != value) return value;
    if (value >= 89.0f) return __builtin_inff();
    if (value <= -104.0f) return 0.0f;
    double scaled = (double)value * 1.4426950408889634074;
    int exponent = (int)(scaled + (scaled >= 0.0 ? 0.5 : -0.5));
    double reduced = ((double)value - exponent * 0.69314718036912381649) -
        exponent * 1.9082149292705877000e-10;
    double sum = 1.0 + reduced * (1.0 + reduced * (0.5 + reduced * (0.16666666666666666 + reduced * (0.041666666666666664 + reduced * (0.008333333333333333 + reduced * (0.0013888888888888889 + reduced * (0.0001984126984126984 + reduced * (2.4801587301587302e-5 + reduced * (2.755731922398589e-6 + reduced * (2.755731922398589e-7 + reduced * (2.5052108385441719e-8 + reduced * 2.0876756987868099e-9)))))))))));
    union { vx_u64 bits; double value; } power;
    power.bits = (vx_u64)(exponent + 1023) << 52;
    return (float)(sum * power.value);
}

static VX_CUDA_DEVICE vx_u32 vx_cuda_trig_bit(const vx_u32* words, int bit) {
    return bit < 0 || bit >= 224 ? 0u : (words[bit / 32] >> (bit % 32)) & 1u;
}

/* F32 Payne--Hanek reduction needs only 192 reciprocal bits: even FLT_MAX
 * retains 64 fractional bits after the exact 24-bit significand product.
 * Small inputs use split pi/2 subtraction; both paths produce [-pi/4,pi/4].
 * No host reads or CUDA math-library dependency are involved. */
static VX_CUDA_DEVICE float vx_cuda_trig_f32(float value, int cosine) {
    union { float value; vx_u32 bits; } input = {value};
    vx_u32 magnitude_bits = input.bits & 0x7fffffffu;
    if (magnitude_bits >= 0x7f800000u) return value - value;
    union { vx_u32 bits; float value; } magnitude = {magnitude_bits};
    double reduced = (double)magnitude.value;
    vx_u32 quadrant = 0u;
    if (magnitude.value < 1048576.0f) {
        int multiple = (int)(reduced * 0.63661977236758134308 + 0.5);
        quadrant = (vx_u32)multiple & 3u;
        reduced = ((reduced - multiple * 1.5707963267341256142) -
            multiple * 6.0771005063039659766e-11) -
            multiple * 2.0222662487959506315e-21;
    } else {
        const vx_u32 reciprocal[6] = {0x3c439041u, 0xdb629599u,
            0xf534ddc0u, 0xfc2757d1u, 0x4e441529u, 0xa2f9836eu};
        vx_u32 product[7];
        vx_u32 mantissa = (magnitude_bits & 0x7fffffu) | 0x800000u;
        vx_u64 carry = 0u;
        for (int word = 0; word < 6; ++word) {
            vx_u64 part = (vx_u64)reciprocal[word] * mantissa + carry;
            product[word] = (vx_u32)part;
            carry = part >> 32;
        }
        product[6] = (vx_u32)carry;
        int point = 342 - (int)(magnitude_bits >> 23);
        vx_u32 up = vx_cuda_trig_bit(product, point - 1);
        quadrant = (vx_cuda_trig_bit(product, point) +
            2u * vx_cuda_trig_bit(product, point + 1) + up) & 3u;
        if (up) {
            carry = 1u;
            for (int word = 0; word < 7; ++word) {
                carry += (vx_u64)(vx_u32)~product[word];
                product[word] = (vx_u32)carry;
                carry >>= 32;
            }
        }
        double fraction = 0.0, weight = 0.5;
        for (int bit = 1; bit <= 96; ++bit) {
            fraction += (double)vx_cuda_trig_bit(product, point - bit) * weight;
            weight *= 0.5;
        }
        reduced = fraction * 1.5707963267948966192;
        if (up) reduced = -reduced;
    }
    if (cosine) quadrant = (quadrant + 1u) & 3u;
    double square = reduced * reduced;
    double result;
    if (quadrant & 1u) {
        result = 1.0 + square * (-0.5 + square * (0.041666666666666664 + square * (-0.0013888888888888889 + square * (2.4801587301587302e-5 + square * (-2.755731922398589e-7 + square * (2.0876756987868099e-9 - square * 1.1470745597729725e-11))))));
    } else {
        result = reduced * (1.0 + square * (-0.16666666666666666 + square * (0.008333333333333333 + square * (-0.0001984126984126984 + square * (2.755731922398589e-6 + square * (-2.5052108385441719e-8 + square * (1.6059043836821615e-10 - square * 7.647163731819816e-13)))))));
    }
    if (quadrant & 2u) result = -result;
    if (!cosine && (input.bits >> 31)) result = -result;
    return (float)result;
}

/* Unary operation IDs are private to cuda_engine.c. */
static VX_CUDA_DEVICE float vx_cuda_log_full_f32(float x) {
    float value;
    asm("lg2.approx.f32 %0, %1;" : "=f"(value) : "f"(x));
    return value * 0.6931471805599453f;
}
static VX_CUDA_DEVICE float vx_cuda_unary_atan(float value) {
    if (value != value || value == 0.0f) return value;
    float x = vx_abs(value), offset = 0.0f;
    int reciprocal = x > 1.0f;
    if (reciprocal) x = 1.0f / x;
    if (x > 0.414213562373095f) { x = (x-1.0f)/(x+1.0f); offset = 0.7853981633974483f; }
    float q = x*x;
    float result = offset + x * (1.0f + q * (-1.0f/3.0f + q * (1.0f/5.0f + q *
        (-1.0f/7.0f + q * (1.0f/9.0f + q * (-1.0f/11.0f + q * (1.0f/13.0f + q *
        (-1.0f/15.0f + q * (1.0f/17.0f - q/19.0f)))))))));
    if (reciprocal) result = 1.5707963267948966f - result;
    return value < 0.0f ? -result : result;
}
static VX_CUDA_DEVICE float vx_cuda_log1p_positive(float z) {
    if (z < 0.0001f) return z * (1.0f - z * 0.5f + z * z / 3.0f);
    float r = z/(2.0f+z), q = r*r;
    return 2.0f*r*(1.0f+q*(1.0f/3.0f+q*(1.0f/5.0f+q*(1.0f/7.0f+
        q*(1.0f/9.0f+q*(1.0f/11.0f+q*(1.0f/13.0f+q/15.0f)))))));
}

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
    else if (op == 9) y = vx_cuda_exp_full_f32(x);
    else if (op == 10) y = vx_cuda_trig_f32(x, 0);
    else if (op == 11) y = vx_cuda_trig_f32(x, 1);
    else if (op == 12) y = vx_sqrt(x);
    else if (op == 13) y = vx_cuda_log_full_f32(x);
    else if (op == 14) y = x == 0.0f ? x : vx_erf_approx(x);
    else if (op == 15) asm("cvt.rmi.f32.f32 %0, %1;" : "=f"(y) : "f"(x));
    else if (op == 16) asm("cvt.rpi.f32.f32 %0, %1;" : "=f"(y) : "f"(x));
    else if (op == 17) asm("cvt.rni.f32.f32 %0, %1;" : "=f"(y) : "f"(x));
    else if (op == 18) y = vx_cuda_unary_atan(x);
    else if (op == 19) y = (x > 0.0f ? x : 0.0f) + vx_cuda_log1p_positive(vx_cuda_exp_full_f32(-vx_abs(x)));
    else if (op == 20) y = -x;
    output[index] = y;
}

/* Each thread owns one strided scan. Work is linear in the tensor size and
 * accumulation follows axis order exactly, including exclusive/reverse.
 * Loading each input before writing also permits an exact in-place scan. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_cumsum_f32(
        const float* input, float* output, vx_u32 scans, vx_u32 length,
        vx_u32 inner, vx_u32 exclusive, vx_u32 reverse) {
    vx_u32 scan = vx_global_x();
    if (scan >= scans) return;
    vx_u32 base = (scan / inner) * length * inner + scan % inner;
    float sum = 0.0f;
    for (vx_u32 step = 0; step < length; ++step) {
        vx_u32 position = reverse ? length - 1u - step : step;
        vx_u32 index = base + position * inner;
        float value = input[index];
        if (exclusive) output[index] = sum;
        sum = vx_add_rn(sum, value);
        if (!exclusive) output[index] = sum;
    }
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_cumsum_i32(
        const vx_u32* input, vx_u32* output, vx_u32 scans, vx_u32 length,
        vx_u32 inner, vx_u32 exclusive, vx_u32 reverse) {
    vx_u32 scan = vx_global_x();
    if (scan >= scans) return;
    vx_u32 base = (scan / inner) * length * inner + scan % inner;
    vx_u32 sum = 0u;
    for (vx_u32 step = 0; step < length; ++step) {
        vx_u32 position = reverse ? length - 1u - step : step;
        vx_u32 index = base + position * inner;
        vx_u32 value = input[index];
        if (exclusive) output[index] = sum;
        sum += value;
        if (!exclusive) output[index] = sum;
    }
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

extern "C" VX_CUDA_GLOBAL void vx_cuda_layernorm_cooperative_f32(
        const float* input, const float* weight, const float* bias,
        float* output, vx_u32 rows, vx_u32 width, float epsilon) {
    VX_CUDA_SHARED float warp_sums[8];
    VX_CUDA_SHARED float s_mean;
    VX_CUDA_SHARED float s_inverse;
    vx_u32 row = blockIdx.x;
    if (row >= rows) return;
    vx_u32 lane = threadIdx.x;
    vx_u32 base = row * width;

    float sum = 0.0f;
    for (vx_u32 column = lane; column < width; column += blockDim.x)
        sum = vx_add_rn(sum, input[base + column]);
    for (vx_u32 delta = 16u; delta; delta /= 2u)
        sum = vx_add_rn(sum, vx_cuda_norm_shuffle_down(sum, delta));
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = sum;
    vx_sync_threads();
    if (lane == 0u) {
        float total = warp_sums[0];
        for (vx_u32 warp = 1u; warp < blockDim.x / 32u; ++warp)
            total = vx_add_rn(total, warp_sums[warp]);
        s_mean = total / (float)width;
    }
    vx_sync_threads();
    float mean = s_mean;

    float sum_squares = 0.0f;
    for (vx_u32 column = lane; column < width; column += blockDim.x) {
        float delta = input[base + column] - mean;
        sum_squares = vx_add_rn(sum_squares, vx_mul_rn(delta, delta));
    }
    for (vx_u32 delta = 16u; delta; delta /= 2u)
        sum_squares = vx_add_rn(sum_squares, vx_cuda_norm_shuffle_down(sum_squares, delta));
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = sum_squares;
    vx_sync_threads();
    if (lane == 0u) {
        float total = warp_sums[0];
        for (vx_u32 warp = 1u; warp < blockDim.x / 32u; ++warp)
            total = vx_add_rn(total, warp_sums[warp]);
        s_inverse = 1.0f / vx_sqrt(total / (float)width + epsilon);
    }
    vx_sync_threads();
    float inverse = s_inverse;

    for (vx_u32 column = lane; column < width; column += blockDim.x) {
        float val = vx_mul_rn(vx_mul_rn(input[base + column] - mean, inverse), weight[column]);
        if (bias) val = vx_add_rn(val, bias[column]);
        output[base + column] = val;
    }
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
