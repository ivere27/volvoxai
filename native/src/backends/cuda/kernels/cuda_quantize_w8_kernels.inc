/*
 * Manual CUDA Driver API/PTX primitives for F32-master -> canonical W8
 * materialization.  This file is intended to be included by the full-profile
 * CUDA kernel translation unit after the freestanding CUDA macros have been
 * defined.  It deliberately imports no CUDA headers and calls no CUDA Runtime,
 * cuBLAS, cuBLASLt, cuDNN, or libdevice routine.
 *
 * Canonical storage contract
 * --------------------------
 * `rows` is the output-channel dimension and `columns` is the flattened
 * reduction dimension.  The destination is always row-major [rows,columns]:
 * QLinear/QGemm weights are [O,I], QConv2D weights are flattened OHWI
 * [O,H*W*I_per_group], and QEmbedding weights are [vocab,hidden].  Setting
 * transpose_source maps an F32 IN_OUT [columns,rows] master without a staging
 * transpose. Scale and zero-point tensors contain one entry per row. Both are
 * published in safetensors and referenced by the graph's central
 * `volvox-affine-safetensors/v1` table; numeric affine values never enter JSON.
 *
 * Scheme 0 is VolvoxAI's authored W8 contract: symmetric signed-I8 in the
 * narrow [-127,127] range, scale=max(abs(row))/127 (or 1 for an all-zero row),
 * and zero point 0.  It matches volvoxai_ptq_pack_weight_i8() and
 * volvoxai_training_quantize_weight_f32_to_i8().  Scheme 1 is per-row
 * asymmetric signed-I8 in the full [-128,127] range.  Its range includes real
 * zero and its scale/zero point use the same 255-bin and ties-to-even rules as
 * volvoxai_ptq_calculate_params().
 *
 * Transaction and launch contract
 * -------------------------------
 * The host owns one zero-initialized status word and launches all stages in a
 * single ordered CUDA stream.  Source/parameter/bias validation writes only
 * scratch plus status.  Public scales, zero points, W8 bytes, I32 bias, and
 * saturation total are committed only by a later stage after status is still
 * zero, so invalid input cannot leave a partially materialized artifact.
 *
 * Recommended recompute sequence:
 *   reset -> reduce -> derive scratch params -> validate optional bias ->
 *   commit params -> quantize -> sum saturation -> pack optional bias.
 * Preserve mode replaces derive+commit with validate_preserved_params.  The
 * host must stop and report failure when status is nonzero after stream sync.
 * Dimensions, pointer ranges/non-aliasing, allocation sizes, exact grid shape,
 * and rows*columns <= UINT32_MAX remain host validation responsibilities.
 *
 * Status bits:
 *   bit 0 (1): invalid scalar/dimension/launch argument observed on device
 *   bit 1 (2): source contains NaN or infinity
 *   bit 2 (4): scale/zero-point derivation or preserved metadata is invalid
 *   bit 3 (8): bias, accumulator scale, or rounded I32 bias is invalid
 */
#ifndef VOLVOXAI_CUDA_QUANTIZE_W8_KERNELS_INC
#define VOLVOXAI_CUDA_QUANTIZE_W8_KERNELS_INC

/* Keep this include independently compilable for a focused PTX smoke test. */
#ifndef VX_CUDA_GLOBAL
#if defined(__clang__)
#define VX_CUDA_GLOBAL __attribute__((global))
#define VX_CUDA_DEVICE __attribute__((device))
#include <__clang_cuda_builtin_vars.h>
#else
#define VX_CUDA_GLOBAL __global__
#define VX_CUDA_DEVICE __device__
#endif
#endif

#if defined(__clang__)
#define VX_CUDA_QUANTIZE_W8_SHARED __attribute__((shared))
#else
#define VX_CUDA_QUANTIZE_W8_SHARED __shared__
#endif

typedef unsigned int vx_cuda_w8_u32;
typedef int vx_cuda_w8_i32;
typedef signed char vx_cuda_w8_i8;
typedef unsigned long long vx_cuda_w8_u64;

enum VxCudaQuantizeW8Status {
    VX_CUDA_QUANTIZE_W8_INVALID_ARGUMENT = 1u,
    VX_CUDA_QUANTIZE_W8_NONFINITE_SOURCE = 2u,
    VX_CUDA_QUANTIZE_W8_INVALID_PARAMS = 4u,
    VX_CUDA_QUANTIZE_W8_INVALID_BIAS = 8u
};

enum VxCudaQuantizeW8Scheme {
    VX_CUDA_QUANTIZE_W8_SYMMETRIC = 0u,
    VX_CUDA_QUANTIZE_W8_ASYMMETRIC = 1u
};

enum VxCudaQuantizeW8ScalePolicy {
    VX_CUDA_QUANTIZE_W8_RECOMPUTE = 0u,
    VX_CUDA_QUANTIZE_W8_PRESERVE = 1u
};

static VX_CUDA_DEVICE vx_cuda_w8_u32 vx_cuda_w8_global_x() {
    return blockIdx.x * blockDim.x + threadIdx.x;
}

static VX_CUDA_DEVICE void vx_cuda_w8_barrier() {
    asm volatile("bar.sync 0;" : : : "memory");
}

static VX_CUDA_DEVICE vx_cuda_w8_u32 vx_cuda_w8_float_bits(float value) {
    vx_cuda_w8_u32 bits;
    asm("mov.b32 %0, %1;" : "=r"(bits) : "f"(value));
    return bits;
}

static VX_CUDA_DEVICE int vx_cuda_w8_isfinite(float value) {
    return (vx_cuda_w8_float_bits(value) & 0x7f800000u) != 0x7f800000u;
}

static VX_CUDA_DEVICE vx_cuda_w8_u32 vx_cuda_w8_atomic_or(
        vx_cuda_w8_u32* address, vx_cuda_w8_u32 value) {
    vx_cuda_w8_u32 prior;
    vx_cuda_w8_u64 raw_address = (vx_cuda_w8_u64)address;
    asm volatile("atom.global.or.b32 %0, [%1], %2;"
                 : "=r"(prior)
                 : "l"(raw_address), "r"(value)
                 : "memory");
    return prior;
}

static VX_CUDA_DEVICE float vx_cuda_w8_mul_rn(float left, float right) {
    float result;
    asm("mul.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
}

static VX_CUDA_DEVICE float vx_cuda_w8_div_rn(float left, float right) {
    float result;
    asm("div.rn.f32 %0, %1, %2;" : "=f"(result) : "f"(left), "f"(right));
    return result;
}

static VX_CUDA_DEVICE vx_cuda_w8_i32 vx_cuda_w8_round_even_f64(double value) {
    vx_cuda_w8_i32 result;
    asm("cvt.rni.s32.f64 %0, %1;" : "=r"(result) : "d"(value));
    return result;
}

static VX_CUDA_DEVICE long long vx_cuda_w8_round_even_i64(double value) {
    long long result;
    asm("cvt.rni.s64.f64 %0, %1;" : "=l"(result) : "d"(value));
    return result;
}

/* Reset only scratch status.  The public saturation total remains untouched
 * until sum_saturation commits a successful result.
 * Required launch: grid=(1,1,1), block=(1,1,1). */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_reset(
        vx_cuda_w8_u32* status) {
    if (vx_cuda_w8_global_x() != 0u) return;
    if (status) *status = 0u;
}

/*
 * Per-row min/max reduction, including real zero in every range.  Scratch
 * minima/maxima each contain `rows` floats.  Exactly 256 lanes cooperate on
 * one row so comparison order cannot affect the extrema.
 * Required launch: grid=(rows,1,1), block=(256,1,1).
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_reduce_f32(
        const float* source, float* minima, float* maxima,
        vx_cuda_w8_u32* status, vx_cuda_w8_u32 rows,
        vx_cuda_w8_u32 columns, vx_cuda_w8_u32 transpose_source) {
    VX_CUDA_QUANTIZE_W8_SHARED float shared_minimum[256];
    VX_CUDA_QUANTIZE_W8_SHARED float shared_maximum[256];
    VX_CUDA_QUANTIZE_W8_SHARED vx_cuda_w8_u32 shared_invalid[256];
    vx_cuda_w8_u32 lane = threadIdx.x;
    vx_cuda_w8_u32 row = blockIdx.x;

    if (blockDim.x != 256u) {
        if (lane == 0u && status)
            vx_cuda_w8_atomic_or(status,
                                 VX_CUDA_QUANTIZE_W8_INVALID_ARGUMENT);
        return;
    }
    if (row >= rows) return;
    if (!source || !minima || !maxima || !status || rows == 0u ||
        columns == 0u || transpose_source > 1u) {
        if (lane == 0u && status)
            vx_cuda_w8_atomic_or(status,
                                 VX_CUDA_QUANTIZE_W8_INVALID_ARGUMENT);
        return;
    }

    float minimum = 0.0f;
    float maximum = 0.0f;
    vx_cuda_w8_u32 invalid = 0u;
    for (vx_cuda_w8_u32 column = lane; column < columns; column += 256u) {
        vx_cuda_w8_u64 index = transpose_source != 0u
            ? (vx_cuda_w8_u64)column * rows + row
            : (vx_cuda_w8_u64)row * columns + column;
        float value = source[index];
        if (!vx_cuda_w8_isfinite(value)) {
            invalid = 1u;
        } else {
            if (value < minimum) minimum = value;
            if (value > maximum) maximum = value;
        }
    }
    shared_minimum[lane] = minimum;
    shared_maximum[lane] = maximum;
    shared_invalid[lane] = invalid;
    vx_cuda_w8_barrier();

    for (vx_cuda_w8_u32 stride = 128u; stride != 0u; stride >>= 1u) {
        if (lane < stride) {
            float other_minimum = shared_minimum[lane + stride];
            float other_maximum = shared_maximum[lane + stride];
            if (other_minimum < shared_minimum[lane])
                shared_minimum[lane] = other_minimum;
            if (other_maximum > shared_maximum[lane])
                shared_maximum[lane] = other_maximum;
            shared_invalid[lane] |= shared_invalid[lane + stride];
        }
        vx_cuda_w8_barrier();
    }

    if (lane == 0u) {
        minima[row] = shared_minimum[0];
        maxima[row] = shared_maximum[0];
        if (shared_invalid[0] != 0u)
            vx_cuda_w8_atomic_or(status,
                                 VX_CUDA_QUANTIZE_W8_NONFINITE_SOURCE);
    }
}

/*
 * Derive candidate metadata into scratch.  This stage never mutates public
 * scale/zero-point arrays.  Required launch: block=(256,1,1),
 * grid=(ceil(rows/256),1,1).
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_derive_params(
        const float* minima, const float* maxima, float* candidate_scales,
        vx_cuda_w8_i32* candidate_zero_points, vx_cuda_w8_u32* status,
        vx_cuda_w8_u32 rows, vx_cuda_w8_u32 scheme) {
    vx_cuda_w8_u32 row = vx_cuda_w8_global_x();
    if (!status || !minima || !maxima || !candidate_scales ||
        !candidate_zero_points || rows == 0u || scheme > 1u) {
        if (status && row == 0u)
            vx_cuda_w8_atomic_or(status,
                                 VX_CUDA_QUANTIZE_W8_INVALID_ARGUMENT);
        return;
    }
    if (row >= rows || *status != 0u) return;

    float minimum = minima[row];
    float maximum = maxima[row];
    float scale;
    vx_cuda_w8_i32 zero_point;
    if (!vx_cuda_w8_isfinite(minimum) || !vx_cuda_w8_isfinite(maximum) ||
        minimum > 0.0f || maximum < 0.0f || minimum > maximum) {
        vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_PARAMS);
        return;
    }
    if (scheme == VX_CUDA_QUANTIZE_W8_SYMMETRIC) {
        float magnitude = -minimum > maximum ? -minimum : maximum;
        scale = magnitude == 0.0f ? 1.0f : (float)((double)magnitude / 127.0);
        /* Match ptq_positive_f32(): a finite, positive binary64 scale that
         * rounds below the F32 subnormal domain commits FLT_MIN rather than
         * turning an otherwise valid row into an invalid zero scale. */
        if (magnitude != 0.0f && scale == 0.0f)
            scale = 1.1754943508222875e-38f;
        zero_point = 0;
    } else if (minimum == maximum) {
        scale = 1.0f;
        zero_point = 0;
    } else {
        scale = (float)(((double)maximum - (double)minimum) / 255.0);
        if (maximum != minimum && scale == 0.0f)
            scale = 1.1754943508222875e-38f;
        if (!vx_cuda_w8_isfinite(scale) || !(scale > 0.0f)) {
            vx_cuda_w8_atomic_or(status,
                                 VX_CUDA_QUANTIZE_W8_INVALID_PARAMS);
            return;
        }
        double transformed = -128.0 - (double)minimum / (double)scale;
        if (transformed <= -128.0) zero_point = -128;
        else if (transformed >= 127.0) zero_point = 127;
        else zero_point = vx_cuda_w8_round_even_f64(transformed);
    }
    if (!vx_cuda_w8_isfinite(scale) || !(scale > 0.0f)) {
        vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_PARAMS);
        return;
    }
    candidate_scales[row] = scale;
    candidate_zero_points[row] = zero_point;
}

/*
 * Validate caller-owned metadata for scale-preserving requantization.  A
 * symmetric artifact must retain zero point 0.  An asymmetric artifact accepts
 * any signed-I8 zero point.  Required launch is the same as derive_params.
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_validate_preserved_params(
        const float* scales, const vx_cuda_w8_i32* zero_points,
        vx_cuda_w8_u32* status, vx_cuda_w8_u32 rows,
        vx_cuda_w8_u32 scheme) {
    vx_cuda_w8_u32 row = vx_cuda_w8_global_x();
    if (!status || !scales || !zero_points || rows == 0u || scheme > 1u) {
        if (status && row == 0u)
            vx_cuda_w8_atomic_or(status,
                                 VX_CUDA_QUANTIZE_W8_INVALID_ARGUMENT);
        return;
    }
    if (row >= rows || *status != 0u) return;
    float scale = scales[row];
    vx_cuda_w8_i32 zero_point = zero_points[row];
    if (!vx_cuda_w8_isfinite(scale) || !(scale > 0.0f) ||
        zero_point < -128 || zero_point > 127 ||
        (scheme == VX_CUDA_QUANTIZE_W8_SYMMETRIC && zero_point != 0))
        vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_PARAMS);
}

/*
 * Transactional metadata commit after derive_params.  `scale_policy` must be
 * RECOMPUTE; PRESERVE is intentionally a no-op because validation must not
 * rewrite caller-owned descriptor bytes.  Required launch is the same as
 * derive_params.
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_commit_params(
        const float* candidate_scales,
        const vx_cuda_w8_i32* candidate_zero_points, float* scales,
        vx_cuda_w8_i32* zero_points, const vx_cuda_w8_u32* status,
        vx_cuda_w8_u32 rows, vx_cuda_w8_u32 scale_policy) {
    vx_cuda_w8_u32 row = vx_cuda_w8_global_x();
    if (!status || !candidate_scales || !candidate_zero_points || !scales ||
        !zero_points || rows == 0u || scale_policy > 1u || row >= rows ||
        *status != 0u || scale_policy == VX_CUDA_QUANTIZE_W8_PRESERVE) return;
    scales[row] = candidate_scales[row];
    zero_points[row] = candidate_zero_points[row];
}

/*
 * Quantize one canonical destination element per lane and reduce saturation
 * flags to one u32 count per block.  `saturation_by_block` has exactly
 * ceil(rows*columns/256) entries.  Required launch uses block=(256,1,1) and
 * that exact one-dimensional grid.
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_f32(
        const float* source, const float* scales,
        const vx_cuda_w8_i32* zero_points, vx_cuda_w8_i8* output,
        vx_cuda_w8_u32* saturation_by_block,
        const vx_cuda_w8_u32* status, vx_cuda_w8_u32 rows,
        vx_cuda_w8_u32 columns, vx_cuda_w8_u32 transpose_source,
        vx_cuda_w8_u32 scheme) {
    VX_CUDA_QUANTIZE_W8_SHARED vx_cuda_w8_u32 shared_saturated[256];
    vx_cuda_w8_u32 lane = threadIdx.x;
    vx_cuda_w8_u32 index = vx_cuda_w8_global_x();
    vx_cuda_w8_u32 elements = rows * columns;
    vx_cuda_w8_u32 saturated = 0u;

    if (blockDim.x != 256u || !status || !source || !scales || !zero_points ||
        !output || !saturation_by_block || rows == 0u || columns == 0u ||
        rows > 0xffffffffu / columns || transpose_source > 1u || scheme > 1u ||
        *status != 0u) return;

    if (index < elements) {
        vx_cuda_w8_u32 row = index / columns;
        vx_cuda_w8_u32 column = index - row * columns;
        vx_cuda_w8_u64 source_index = transpose_source != 0u
            ? (vx_cuda_w8_u64)column * rows + row : index;
        double transformed = (double)source[source_index] /
            (double)scales[row] + (double)zero_points[row];
        vx_cuda_w8_i32 minimum = scheme == VX_CUDA_QUANTIZE_W8_SYMMETRIC
            ? -127 : -128;
        vx_cuda_w8_i32 maximum = 127;
        vx_cuda_w8_i32 quantized;

        /* Avoid an out-of-range f64->s32 conversion while retaining the CPU
         * contract of rounding first and then deciding whether it saturated. */
        if (transformed < (double)minimum - 1.0) {
            quantized = minimum;
            saturated = 1u;
        } else if (transformed > (double)maximum + 1.0) {
            quantized = maximum;
            saturated = 1u;
        } else {
            quantized = vx_cuda_w8_round_even_f64(transformed);
            if (quantized < minimum) {
                quantized = minimum;
                saturated = 1u;
            } else if (quantized > maximum) {
                quantized = maximum;
                saturated = 1u;
            }
        }
        output[index] = (vx_cuda_w8_i8)quantized;
    }

    shared_saturated[lane] = saturated;
    vx_cuda_w8_barrier();
    for (vx_cuda_w8_u32 stride = 128u; stride != 0u; stride >>= 1u) {
        if (lane < stride)
            shared_saturated[lane] += shared_saturated[lane + stride];
        vx_cuda_w8_barrier();
    }
    if (lane == 0u) saturation_by_block[blockIdx.x] = shared_saturated[0];
}

/* Required launch: grid=(1,1,1), block=(1,1,1). */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_sum_saturation(
        const vx_cuda_w8_u32* saturation_by_block,
        vx_cuda_w8_u64* saturation_total, const vx_cuda_w8_u32* status,
        vx_cuda_w8_u32 block_count) {
    if (vx_cuda_w8_global_x() != 0u || !status || *status != 0u ||
        !saturation_by_block || !saturation_total || block_count == 0u) return;
    vx_cuda_w8_u64 total = 0ull;
    for (vx_cuda_w8_u32 block = 0u; block < block_count; ++block)
        total += saturation_by_block[block];
    *saturation_total = total;
}

/*
 * Preflight F32 bias for canonical W8A8 I32 accumulator storage.  The mapping
 * is round_ties_even(bias[row] / (input_scale * weight_scale[row])).
 * Float multiply/divide are explicitly round-to-nearest to match the native C
 * implementation before its binary64 rounding step.  Required launch:
 * block=(256,1,1), grid=(ceil(rows/256),1,1).
 */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_validate_bias_f32(
        const float* bias, float input_scale, const float* weight_scales,
        vx_cuda_w8_u32* status, vx_cuda_w8_u32 rows) {
    vx_cuda_w8_u32 row = vx_cuda_w8_global_x();
    if (!status || !bias || !weight_scales || rows == 0u ||
        !vx_cuda_w8_isfinite(input_scale) || !(input_scale > 0.0f)) {
        if (status && row == 0u)
            vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_BIAS);
        return;
    }
    if (row >= rows || *status != 0u) return;
    float weight_scale = weight_scales[row];
    float value = bias[row];
    float accumulator_scale = vx_cuda_w8_mul_rn(input_scale, weight_scale);
    float ratio = vx_cuda_w8_div_rn(value, accumulator_scale);
    if (!vx_cuda_w8_isfinite(weight_scale) || !(weight_scale > 0.0f) ||
        !vx_cuda_w8_isfinite(value) ||
        !vx_cuda_w8_isfinite(accumulator_scale) ||
        !(accumulator_scale > 0.0f) || !vx_cuda_w8_isfinite(ratio)) {
        vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_BIAS);
        return;
    }
    double rounded_source = (double)ratio;
    if (rounded_source < -2147483649.0 || rounded_source > 2147483648.0) {
        vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_BIAS);
        return;
    }
    long long rounded = vx_cuda_w8_round_even_i64(rounded_source);
    if (rounded < -2147483648ll || rounded > 2147483647ll)
        vx_cuda_w8_atomic_or(status, VX_CUDA_QUANTIZE_W8_INVALID_BIAS);
}

/* Transactional bias commit after validate_bias_f32; same launch shape. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_quantize_w8_pack_bias_f32(
        const float* bias, float input_scale, const float* weight_scales,
        vx_cuda_w8_i32* output, const vx_cuda_w8_u32* status,
        vx_cuda_w8_u32 rows) {
    vx_cuda_w8_u32 row = vx_cuda_w8_global_x();
    if (!status || *status != 0u || !bias || !weight_scales || !output ||
        row >= rows) return;
    float accumulator_scale = vx_cuda_w8_mul_rn(input_scale,
                                                 weight_scales[row]);
    float ratio = vx_cuda_w8_div_rn(bias[row], accumulator_scale);
    output[row] = (vx_cuda_w8_i32)vx_cuda_w8_round_even_i64((double)ratio);
}

#undef VX_CUDA_QUANTIZE_W8_SHARED
#endif
