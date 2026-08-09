/*
 * Forward-only portable kernels used by the ordinary WASM inference engine.
 *
 * Keep this file independent of training_kernels.c: volvoxai.wasm is the
 * inference sidecar and must not acquire a training dependency or export a
 * training symbol.  The full sidecar links this file through kernels.c as a
 * forward-compatible superset.
 */
#include "mathcompat.h"
#include "inference_kernels.h"
#include "kernel_platform.h"
#include "fast_exp.h"
#include "w8a8_affine.h"
#include <stddef.h>
#include <stdint.h>
#if !defined(__wasm__)
#include <string.h>
#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#endif

static int vx_pf_finite_f32(float value) {
    union { float f; uint32_t u; } bits = { value };
    return ((bits.u >> 23u) & 0xffu) != 0xffu;
}

static int vx_pf_finite_f64(double value) {
    union { double f; uint64_t u; } bits = { value };
    return ((bits.u >> 52u) & 0x7ffu) != 0x7ffu;
}

static int vx_pf_mul_size(size_t *value, size_t factor) {
    if (factor && *value > (size_t)-1 / factor) return 0;
    *value *= factor;
    return 1;
}

static int vx_pf_contiguous_strides(const uint32_t *shape, uint32_t rank,
        size_t *strides) {
    size_t stride = 1;
    if (!shape || !strides || rank == 0 || rank > 8) return 0;
    for (uint32_t reverse = rank; reverse-- > 0;) {
        if (!shape[reverse]) return 0;
        strides[reverse] = stride;
        if (!vx_pf_mul_size(&stride, shape[reverse])) return 0;
    }
    return 1;
}

static size_t vx_pf_broadcast_index(uint32_t output_index,
        const uint32_t *input_shape, uint32_t input_rank,
        const uint32_t *output_shape, uint32_t output_rank,
        const size_t *input_strides) {
    size_t remaining = output_index;
    size_t input_index = 0;
    uint32_t offset = output_rank - input_rank;
    for (uint32_t reverse = output_rank; reverse-- > 0;) {
        uint32_t coordinate = (uint32_t)(remaining % output_shape[reverse]);
        remaining /= output_shape[reverse];
        if (reverse >= offset && input_shape[reverse - offset] != 1u) {
            input_index += (size_t)coordinate * input_strides[reverse - offset];
        }
    }
    return input_index;
}

static int vx_pf_broadcast_validate(const uint32_t *a_shape,
        const uint32_t *b_shape, const uint32_t *output_shape,
        uint32_t a_rank, uint32_t b_rank, uint32_t output_rank,
        uint32_t elements, size_t *a_strides, size_t *b_strides) {
    size_t product = 1;
    if (!a_shape || !b_shape || !output_shape || !a_strides || !b_strides ||
        a_rank == 0 || b_rank == 0 || output_rank == 0 ||
        a_rank > output_rank || b_rank > output_rank || output_rank > 8) return 0;
    if (!vx_pf_contiguous_strides(a_shape, a_rank, a_strides) ||
        !vx_pf_contiguous_strides(b_shape, b_rank, b_strides)) return 0;
    for (uint32_t axis = 0; axis < output_rank; axis++) {
        uint32_t a_dimension = axis < output_rank - a_rank ? 1u :
            a_shape[axis - (output_rank - a_rank)];
        uint32_t b_dimension = axis < output_rank - b_rank ? 1u :
            b_shape[axis - (output_rank - b_rank)];
        uint32_t output_dimension = output_shape[axis];
        if (!output_dimension || (a_dimension != 1u && a_dimension != output_dimension) ||
            (b_dimension != 1u && b_dimension != output_dimension) ||
            !vx_pf_mul_size(&product, output_dimension)) return 0;
    }
    return product == elements;
}

/*
 * Shape-specialized entries into the broadcast loop.
 *
 * The generic body below resolves two broadcast indices per element, each a loop
 * over the output rank.  That index arithmetic — not the arithmetic the node
 * actually asks for — is what the kernel spends its time on, and it runs even
 * when nothing is being broadcast.  A W8A8 package leaves these ops in F32, so
 * they do not shrink when a model is quantized: the twenty-six Mul nodes the
 * encoder gets from decomposing SiLU into Sigmoid*Mul cost the same in the INT8
 * package as in the FP32 one.
 *
 * Two shapes are worth naming.  When both operands already have as many elements
 * as the output, no axis is stretched and the operation is plain elementwise
 * whatever the ranks look like.  When one operand has a single element it is a
 * scale or an offset.  Both reduce to a contiguous walk with no index math.
 */
static size_t vx_pf_shape_elements(const uint32_t *shape, uint32_t rank) {
    size_t product = 1;
    uint32_t axis;
    for (axis = 0; axis < rank; axis++) product *= (size_t)shape[axis];
    return product;
}

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_BINARY_X86_AVX2 1
#define VX_BINARY_TARGET_AVX2 __attribute__((target("avx2")))

/* `b_step` is 1 for the elementwise case and 0 to hold one operand fixed, which
 * lets both fast shapes share a body. */
static VX_BINARY_TARGET_AVX2 uint32_t vx_pf_binary_contiguous_avx2(
        const float *a, const float *b, float *output, uint32_t elements,
        uint32_t kind, int a_step, int b_step) {
    const __m256 fixed_a = a_step ? _mm256_setzero_ps() : _mm256_set1_ps(a[0]);
    const __m256 fixed_b = b_step ? _mm256_setzero_ps() : _mm256_set1_ps(b[0]);
    uint32_t index = 0;
    for (; index + 8u <= elements; index += 8u) {
        const __m256 left = a_step ? _mm256_loadu_ps(a + index) : fixed_a;
        const __m256 right = b_step ? _mm256_loadu_ps(b + index) : fixed_b;
        __m256 result;
        if (kind == VX_PORTABLE_BINARY_ADD) result = _mm256_add_ps(left, right);
        else if (kind == VX_PORTABLE_BINARY_MUL) result = _mm256_mul_ps(left, right);
        else if (kind == VX_PORTABLE_BINARY_SUB) result = _mm256_sub_ps(left, right);
        else result = _mm256_div_ps(left, right);
        _mm256_storeu_ps(output + index, result);
    }
    return index;
}
#else
#define VX_BINARY_X86_AVX2 0
#endif

static void vx_pf_binary_contiguous(const float *a, const float *b,
        float *output, uint32_t elements, uint32_t kind,
        int a_step, int b_step) {
    uint32_t index = 0;
#if VX_BINARY_X86_AVX2
    if (vx_kernel_platform()->has_avx2)
        index = vx_pf_binary_contiguous_avx2(a, b, output, elements, kind,
                                             a_step, b_step);
#endif
    for (; index < elements; index++) {
        const float left = a[a_step ? index : 0];
        const float right = b[b_step ? index : 0];
        if (kind == VX_PORTABLE_BINARY_ADD) output[index] = left + right;
        else if (kind == VX_PORTABLE_BINARY_MUL) output[index] = left * right;
        else if (kind == VX_PORTABLE_BINARY_SUB) output[index] = left - right;
        else output[index] = left / right;
    }
}

/* Right-aligned F32 broadcast. kind is VX_PORTABLE_BINARY_{ADD,MUL,SUB,DIV}.
 * Division uses IEEE division without an added epsilon, matching the
 * JavaScript portable reference. */
WASM_EXPORT("binary_broadcast_f32")
int binary_broadcast_f32(const float *a, const float *b, float *output,
        const uint32_t *a_shape, const uint32_t *b_shape,
        const uint32_t *output_shape, uint32_t a_rank, uint32_t b_rank,
        uint32_t output_rank, uint32_t elements, uint32_t kind) {
    size_t a_strides[8], b_strides[8];
    size_t a_elements, b_elements;
    if (!a || !b || !output || kind > VX_PORTABLE_BINARY_DIV ||
        !vx_pf_broadcast_validate(a_shape, b_shape, output_shape, a_rank, b_rank,
            output_rank, elements, a_strides, b_strides)) return 0;
    /* Validation has already established the shapes broadcast to `elements`, so
     * an operand carrying that many elements cannot be stretched along any axis
     * and one carrying a single element is stretched along all of them. */
    a_elements = vx_pf_shape_elements(a_shape, a_rank);
    b_elements = vx_pf_shape_elements(b_shape, b_rank);
    if ((a_elements == (size_t)elements || a_elements == 1u) &&
        (b_elements == (size_t)elements || b_elements == 1u)) {
        vx_pf_binary_contiguous(a, b, output, elements, kind,
                                a_elements == 1u ? 0 : 1,
                                b_elements == 1u ? 0 : 1);
        return 1;
    }
    for (uint32_t index = 0; index < elements; index++) {
        size_t a_index = vx_pf_broadcast_index(index, a_shape, a_rank,
            output_shape, output_rank, a_strides);
        size_t b_index = vx_pf_broadcast_index(index, b_shape, b_rank,
            output_shape, output_rank, b_strides);
        if (kind == VX_PORTABLE_BINARY_ADD) output[index] = a[a_index] + b[b_index];
        else if (kind == VX_PORTABLE_BINARY_MUL) output[index] = a[a_index] * b[b_index];
        else if (kind == VX_PORTABLE_BINARY_SUB) output[index] = a[a_index] - b[b_index];
        else output[index] = a[a_index] / b[b_index];
    }
    return 1;
}

/* Canonical ONNX comparisons over I32 values. The output uses I32 0/1
 * storage so boolean-valued edges remain inside the runtime dtype vocabulary. */
WASM_EXPORT("compare_broadcast_i32")
int compare_broadcast_i32(const int32_t *a, const int32_t *b, int32_t *output,
        const uint32_t *a_shape, const uint32_t *b_shape,
        const uint32_t *output_shape, uint32_t a_rank, uint32_t b_rank,
        uint32_t output_rank, uint32_t elements, uint32_t kind) {
    size_t a_strides[8], b_strides[8];
    if (!a || !b || !output || kind > 1u ||
        !vx_pf_broadcast_validate(a_shape, b_shape, output_shape, a_rank, b_rank,
            output_rank, elements, a_strides, b_strides)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        size_t a_index = vx_pf_broadcast_index(index, a_shape, a_rank,
            output_shape, output_rank, a_strides);
        size_t b_index = vx_pf_broadcast_index(index, b_shape, b_rank,
            output_shape, output_rank, b_strides);
        output[index] = kind == 0u
            ? a[a_index] == b[b_index]
            : a[a_index] >= b[b_index];
    }
    return 1;
}

/* Canonical Where broadcasts the condition and both data branches
 * independently to the output shape. Branch storage is copied as U32 words so
 * both F32 and I32 values retain their exact bit patterns. */
WASM_EXPORT("where_broadcast_32")
int where_broadcast_32(const void *condition, uint32_t condition_dtype,
        const uint32_t *a, const uint32_t *b, uint32_t *output,
        const uint32_t *condition_shape, const uint32_t *a_shape,
        const uint32_t *b_shape, const uint32_t *output_shape,
        uint32_t condition_rank, uint32_t a_rank, uint32_t b_rank,
        uint32_t output_rank, uint32_t elements, uint32_t data_dtype) {
    size_t condition_strides[8], condition_b_strides[8];
    size_t a_strides[8], b_strides[8];
    if (!condition || !a || !b || !output ||
        (condition_dtype != VX_DTYPE_F32 && condition_dtype != VX_DTYPE_I32) ||
        (data_dtype != VX_DTYPE_F32 && data_dtype != VX_DTYPE_I32) ||
        !vx_pf_broadcast_validate(condition_shape, a_shape, output_shape,
            condition_rank, a_rank, output_rank, elements,
            condition_strides, a_strides) ||
        !vx_pf_broadcast_validate(condition_shape, b_shape, output_shape,
            condition_rank, b_rank, output_rank, elements,
            condition_b_strides, b_strides)) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        size_t condition_index = vx_pf_broadcast_index(index, condition_shape,
            condition_rank, output_shape, output_rank, condition_strides);
        size_t a_index = vx_pf_broadcast_index(index, a_shape, a_rank,
            output_shape, output_rank, a_strides);
        size_t b_index = vx_pf_broadcast_index(index, b_shape, b_rank,
            output_shape, output_rank, b_strides);
        int selected = condition_dtype == VX_DTYPE_I32
            ? ((const int32_t *)condition)[condition_index] != 0
            : ((const float *)condition)[condition_index] != 0.0f;
        output[index] = selected ? a[a_index] : b[b_index];
    }
    return 1;
}

WASM_EXPORT("not_i32")
int not_i32(const int32_t *input, int32_t *output, uint32_t elements) {
    if (!input || !output || !elements) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        output[index] = input[index] == 0;
    }
    return 1;
}

static uint32_t vx_pf_load_u32_le(const void *data, size_t index) {
    const uint8_t *bytes = (const uint8_t *)data + index * 4u;
    return (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
}

static float vx_pf_load_f32_le(const void *data, size_t index) {
    union { uint32_t u; float f; } value = { vx_pf_load_u32_le(data, index) };
    return value.f;
}

static double vx_pf_typed_value(const void *data, uint32_t dtype, size_t index) {
    if (dtype == VX_DTYPE_F32) return vx_pf_load_f32_le(data, index);
    if (dtype == VX_DTYPE_I32)
        return (int32_t)vx_pf_load_u32_le(data, index);
    if (dtype == VX_DTYPE_I8) return ((const int8_t *)data)[index];
    return ((const uint8_t *)data)[index];
}

static int vx_pf_typed_number_dtype(uint32_t dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

/* Output-major quantized matrix multiply.  The graph representation stores
 * raw I8/U8 bytes as [d_out, d_in], so this deliberately does not require
 * padded 4-byte rows. */
WASM_EXPORT("matmul_quantized_f32")
int matmul_quantized_f32(const float *input, const void *weight,
        const float *scale, const void *zero_point, const float *bias,
        float *output, uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t weight_dtype, uint32_t scale_elements,
        uint32_t zero_point_dtype, uint32_t zero_point_elements) {
    size_t count = 1;
    if (!input || !weight || !scale || !output || !rows || !d_in || !d_out ||
        (weight_dtype != VX_DTYPE_I8 && weight_dtype != VX_DTYPE_U8) ||
        (scale_elements != 1u && scale_elements != d_out) ||
        (zero_point_elements && (!zero_point ||
            !vx_pf_typed_number_dtype(zero_point_dtype) ||
            (zero_point_elements != 1u && zero_point_elements != d_out)))) return 0;
    if (!vx_pf_mul_size(&count, rows) || !vx_pf_mul_size(&count, d_in)) return 0;
    count = 1;
    if (!vx_pf_mul_size(&count, d_out) || !vx_pf_mul_size(&count, d_in)) return 0;
    count = 1;
    if (!vx_pf_mul_size(&count, rows) || !vx_pf_mul_size(&count, d_out)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < d_out; column++) {
            double sum = 0.0;
            double zero = zero_point ? vx_pf_typed_value(zero_point, zero_point_dtype,
                zero_point_elements == 1u ? 0u : column) : 0.0;
            size_t weight_offset = (size_t)column * d_in;
            for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                sum += (double)input[(size_t)row * d_in + dimension] *
                    (vx_pf_typed_value(weight, weight_dtype, weight_offset + dimension) - zero);
            }
            sum *= vx_pf_load_f32_le(scale,
                scale_elements == 1u ? 0u : column);
            if (bias) sum += vx_pf_load_f32_le(bias, column);
            output[(size_t)row * d_out + column] = (float)sum;
        }
    }
    return 1;
}

/* The caller clamps to I8/U8 range first, so floorf() and the integer
 * conversion are both bounded.  This mirrors the portable JavaScript
 * reference's IEEE ties-to-even requantization. */
/* Canonical W8A8 dense inference.  Activations and output use their declared
 * I8/U8 storage, the [d_out,d_in] weight has per-output-channel scale/zero
 * point metadata, and bias is already in the I32 accumulator domain. */
WASM_EXPORT("qlinear_i8u8")
int qlinear_i8u8(const void *input, const void *weight, const int32_t *bias,
        const float *weight_scales, const int32_t *weight_zero_points,
        void *output, uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    size_t input_elements = rows;
    size_t weight_elements = d_out;
    size_t output_elements = rows;
    int32_t output_minimum, output_maximum;
    if (!input || !weight || !bias || !weight_scales || !weight_zero_points || !output ||
        !rows || !d_in || !d_out || !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(weight_dtype) || !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_pf_mul_size(&input_elements, d_in) ||
        !vx_pf_mul_size(&weight_elements, d_in) ||
        !vx_pf_mul_size(&output_elements, d_out)) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t column = 0; column < d_out; column++) {
        volatile float product_scale;
        volatile float multiplier;
        if (!vx_pf_finite_f32(weight_scales[column]) ||
            weight_scales[column] <= 0.0f ||
            !vx_w8a8_zero_point_valid(weight_zero_points[column],
                                         weight_dtype))
            return 0;
        product_scale = input_scale * weight_scales[column];
        multiplier = product_scale / output_scale;
        if (!vx_pf_finite_f32(multiplier) || multiplier <= 0.0f) return 0;
    }
    for (uint32_t row = 0; row < rows; row++) {
        size_t input_offset = (size_t)row * d_in;
        size_t output_offset = (size_t)row * d_out;
        for (uint32_t column = 0; column < d_out; column++) {
            int64_t accumulator = (int64_t)bias[column];
            size_t weight_offset = (size_t)column * d_in;
            const float product_scale = input_scale * weight_scales[column];
            const float multiplier = product_scale / output_scale;
            for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                int32_t input_value = vx_w8a8_byte_value(input, input_dtype,
                    input_offset + dimension);
                int32_t weight_value = vx_w8a8_byte_value(weight, weight_dtype,
                    weight_offset + dimension);
                accumulator += (int64_t)(input_value - input_zero_point) *
                    (int64_t)(weight_value - weight_zero_points[column]);
                if (accumulator < (int64_t)INT32_MIN || accumulator > (int64_t)INT32_MAX) {
                    return 0;
                }
            }
            {
                const float scaled = (float)accumulator * multiplier;
                const float transformed = scaled + (float)output_zero_point;
                int32_t quantized;
                quantized = vx_w8a8_requantize(transformed,
                    output_minimum, output_maximum, output_zero_point);
                vx_w8a8_store_byte(output, output_dtype, output_offset + column, quantized);
            }
        }
    }
    return 1;
}

typedef struct {
    uint32_t m;
    uint32_t k;
    uint32_t n;
    int32_t a_zero_point;
    int32_t b_zero_point;
    int32_t output_zero_point;
    uint32_t a_dtype;
    uint32_t b_dtype;
    uint32_t output_dtype;
    int32_t output_minimum;
    int32_t output_maximum;
    float multiplier;
} VxPfQBatchMatMulDescriptor;

/* Validate the complete one-pair ABI before either the scalar or SIMD kernel
 * writes output.  In particular, the maximum centered-byte product is checked
 * against the canonical I32 accumulation contract.  The two volatile stages
 * make the multiplier identical to Math.fround(fround(a*b)/out) at the JS
 * boundary even when this source is compiled as part of a native amalgam. */
static int vx_pf_qbatch_matmul_validate(const void *a, const void *b,
        void *output, uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype,
        VxPfQBatchMatMulDescriptor *descriptor) {
    size_t a_elements = m;
    size_t b_elements = k;
    size_t output_elements = m;
    int32_t a_minimum, a_maximum, b_minimum, b_maximum;
    uint64_t a_magnitude, b_magnitude, accumulator_bound;
    volatile float product_scale;
    volatile float multiplier;
    VxPfQBatchMatMulDescriptor parsed;
    if (!descriptor || !a || !b || !output || !m || !k || !n ||
        !vx_w8a8_byte_dtype(a_dtype) ||
        !vx_w8a8_byte_dtype(b_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(a_scale) || a_scale <= 0.0f ||
        !vx_pf_finite_f32(b_scale) || b_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(a_zero_point, a_dtype) ||
        !vx_w8a8_zero_point_valid(b_zero_point, b_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_pf_mul_size(&a_elements, k) ||
        !vx_pf_mul_size(&b_elements, n) ||
        !vx_pf_mul_size(&output_elements, n)) return 0;
    a_minimum = a_dtype == VX_DTYPE_I8 ? -128 : 0;
    a_maximum = a_dtype == VX_DTYPE_I8 ? 127 : 255;
    b_minimum = b_dtype == VX_DTYPE_I8 ? -128 : 0;
    b_maximum = b_dtype == VX_DTYPE_I8 ? 127 : 255;
    a_magnitude = (uint64_t)(
        (-((int64_t)a_minimum - a_zero_point) >
         (int64_t)a_maximum - a_zero_point)
            ? -((int64_t)a_minimum - a_zero_point)
            : (int64_t)a_maximum - a_zero_point);
    b_magnitude = (uint64_t)(
        (-((int64_t)b_minimum - b_zero_point) >
         (int64_t)b_maximum - b_zero_point)
            ? -((int64_t)b_minimum - b_zero_point)
            : (int64_t)b_maximum - b_zero_point);
    accumulator_bound = a_magnitude * b_magnitude * (uint64_t)k;
    if (accumulator_bound > (uint64_t)INT32_MAX) return 0;
    product_scale = a_scale * b_scale;
    multiplier = product_scale / output_scale;
    if (!vx_pf_finite_f32(multiplier) || multiplier <= 0.0f) return 0;
    parsed.m = m;
    parsed.k = k;
    parsed.n = n;
    parsed.a_zero_point = a_zero_point;
    parsed.b_zero_point = b_zero_point;
    parsed.output_zero_point = output_zero_point;
    parsed.a_dtype = a_dtype;
    parsed.b_dtype = b_dtype;
    parsed.output_dtype = output_dtype;
    parsed.output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    parsed.output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    parsed.multiplier = multiplier;
    *descriptor = parsed;
    return 1;
}

static void vx_pf_qbatch_matmul_store(void *output, size_t index,
        int32_t accumulator,
        const VxPfQBatchMatMulDescriptor *descriptor) {
    volatile float scaled = (float)accumulator * descriptor->multiplier;
    volatile float transformed =
        scaled + (float)descriptor->output_zero_point;
    const int32_t quantized = vx_w8a8_requantize(
        transformed, descriptor->output_minimum, descriptor->output_maximum,
        descriptor->output_zero_point);
    vx_w8a8_store_byte(
        output, descriptor->output_dtype, index, quantized);
}

static int vx_pf_qbatch_matmul_scalar(const void *a, const void *b,
        void *output, const VxPfQBatchMatMulDescriptor *descriptor) {
    for (uint32_t row = 0; row < descriptor->m; row++) {
        for (uint32_t column = 0; column < descriptor->n; column++) {
            int32_t accumulator = 0;
            for (uint32_t inner = 0; inner < descriptor->k; inner++) {
                int32_t a_value = vx_w8a8_byte_value(
                    a, descriptor->a_dtype,
                    (size_t)row * descriptor->k + inner);
                int32_t b_value = vx_w8a8_byte_value(
                    b, descriptor->b_dtype,
                    (size_t)inner * descriptor->n + column);
                accumulator += (a_value - descriptor->a_zero_point) *
                    (b_value - descriptor->b_zero_point);
            }
            vx_pf_qbatch_matmul_store(
                output, (size_t)row * descriptor->n + column, accumulator,
                descriptor);
        }
    }
    return 1;
}

/* Physical-byte ONNX MatMul for one [M,K] @ [K,N] matrix pair.  Batch
 * broadcasting is resolved by the caller, so this ABI stays usable by both
 * the portable WASM engine and the native runtime.  Both operands are
 * per-tensor I8/U8 tensors and the result is requantized directly into its
 * declared per-tensor byte domain.  This export is also the authoritative
 * scalar oracle for the separately feature-gated WASM SIMD128 kernel. */
WASM_EXPORT("qbatch_matmul_i8u8")
int qbatch_matmul_i8u8(const void *a, const void *b, void *output,
        uint32_t m, uint32_t k, uint32_t n,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype) {
    VxPfQBatchMatMulDescriptor descriptor;
    if (!vx_pf_qbatch_matmul_validate(
            a, b, output, m, k, n, a_scale, a_zero_point,
            b_scale, b_zero_point, output_scale, output_zero_point,
            a_dtype, b_dtype, output_dtype, &descriptor)) return 0;
    return vx_pf_qbatch_matmul_scalar(a, b, output, &descriptor);
}

/* Canonical W8A8 embedding gather. Token IDs use conventional I32 storage;
 * the rank-2 [vocab, hidden] table has one scale/zero point per vocabulary
 * row, and the gathered values are requantized directly into the output's
 * declared I8/U8 domain. Validate every ID and row descriptor before writing
 * an output byte so a malformed call is fail-closed and leaves output intact. */
WASM_EXPORT("qembedding_i8u8")
int qembedding_i8u8(const int32_t *tokens_data, const void *weight,
        const float *weight_scales, const int32_t *weight_zero_points,
        void *output, uint32_t tokens, uint32_t vocab, uint32_t hidden,
        float output_scale, int32_t output_zero_point,
        uint32_t weight_dtype, uint32_t output_dtype) {
    size_t weight_elements = vocab;
    size_t output_elements = tokens;
    int32_t output_minimum, output_maximum;
    if (!tokens_data || !weight || !weight_scales || !weight_zero_points || !output ||
        !tokens || !vocab || !hidden || !vx_w8a8_byte_dtype(weight_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_pf_mul_size(&weight_elements, hidden) ||
        !vx_pf_mul_size(&output_elements, hidden)) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t row = 0; row < vocab; row++) {
        if (!vx_pf_finite_f32(weight_scales[row]) || weight_scales[row] <= 0.0f ||
            !vx_w8a8_zero_point_valid(weight_zero_points[row], weight_dtype)) return 0;
    }
    for (uint32_t token = 0; token < tokens; token++) {
        if (tokens_data[token] < 0 || (uint32_t)tokens_data[token] >= vocab) return 0;
    }
    for (uint32_t token = 0; token < tokens; token++) {
        const uint32_t row = (uint32_t)tokens_data[token];
        const size_t weight_offset = (size_t)row * hidden;
        const size_t output_offset = (size_t)token * hidden;
        for (uint32_t column = 0; column < hidden; column++) {
            const int32_t centered = vx_w8a8_byte_value(weight, weight_dtype,
                weight_offset + column) - weight_zero_points[row];
            const float dequantized = (float)centered * weight_scales[row];
            const float transformed = dequantized / output_scale +
                (float)output_zero_point;
            const int32_t quantized = vx_w8a8_requantize(transformed,
                output_minimum, output_maximum, output_zero_point);
            vx_w8a8_store_byte(output, output_dtype, output_offset + column, quantized);
        }
    }
    return 1;
}

#if !defined(__wasm__)
#include "quant_cpu_opt.h"
#endif

/* Exact-shape W8A8 addition.  Each operand has its own per-tensor mapping;
 * the result is requantized directly into the declared I8/U8 output bytes. */
WASM_EXPORT("qadd_i8u8")
int qadd_i8u8(const void *a, const void *b, void *output, uint32_t elements,
        float a_scale, int32_t a_zero_point,
        float b_scale, int32_t b_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t a_dtype, uint32_t b_dtype, uint32_t output_dtype, uint32_t relu) {
    int32_t output_minimum, output_maximum, relu6_upper = 0;
    if (!a || !b || !output || !elements || !vx_w8a8_byte_dtype(a_dtype) ||
        !vx_w8a8_byte_dtype(b_dtype) || !vx_w8a8_byte_dtype(output_dtype) ||
        relu > 2u ||
        !vx_pf_finite_f32(a_scale) || a_scale <= 0.0f ||
        !vx_pf_finite_f32(b_scale) || b_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(a_zero_point, a_dtype) ||
        !vx_w8a8_zero_point_valid(b_zero_point, b_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype)) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
#if !defined(__wasm__)
    if (vx_qadd_i8u8_native_try(
            a, b, output, elements, a_scale, a_zero_point,
            b_scale, b_zero_point, output_scale, output_zero_point,
            a_dtype, b_dtype, output_dtype, relu)) return 1;
#endif
    for (uint32_t index = 0; index < elements; index++) {
        const float a_delta = (float)(vx_w8a8_byte_value(a, a_dtype, index) -
            a_zero_point);
        const float b_delta = (float)(vx_w8a8_byte_value(b, b_dtype, index) -
            b_zero_point);
        const float a_value = a_delta * a_scale;
        const float b_value = b_delta * b_scale;
        const float sum = a_value + b_value;
        const float scaled = sum / output_scale;
        const float transformed = scaled + (float)output_zero_point;
        int32_t quantized = vx_w8a8_requantize(transformed,
            output_minimum, output_maximum, output_zero_point);
        if (relu) {
            if (quantized < output_zero_point) quantized = output_zero_point;
            if (relu >= 2u && quantized > relu6_upper) quantized = relu6_upper;
        }
        vx_w8a8_store_byte(output, output_dtype, index, quantized);
    }
    return 1;
}

/* I8/U8 activation descriptors are immutable for a loaded graph, while each
 * input has only 256 possible byte values.  Cache the exact byte transform so
 * incremental rows pay one indexed load rather than one expf per element.
 *
 * A four-way thread-local cache avoids locks in parallel native callers and
 * keeps independent WASM instances self-contained.  Cold, small activations
 * stay on the scalar route: building all 256 entries would cost more than the
 * work it replaces.  A table populated by a full-sequence seed remains useful
 * even when a later incremental row is smaller than the build threshold. */
enum {
    VX_PF_QACT_LUT_VALUES = 256,
    VX_PF_QACT_LUT_SETS = 8,
    VX_PF_QACT_LUT_WAYS = 4,
    VX_PF_QACT_LUT_BUILD_THRESHOLD = 512,
};

typedef struct {
    uint32_t input_scale_bits;
    uint32_t output_scale_bits;
    int32_t input_zero_point;
    int32_t output_zero_point;
    uint8_t input_dtype;
    uint8_t output_dtype;
    uint8_t valid;
    uint8_t values[VX_PF_QACT_LUT_VALUES];
} VxPfQactLutEntry;

typedef struct {
    VxPfQactLutEntry entries[VX_PF_QACT_LUT_SETS][VX_PF_QACT_LUT_WAYS];
    uint8_t next_victim[VX_PF_QACT_LUT_SETS];
} VxPfQactLutCache;

static _Thread_local VxPfQactLutCache g_vx_pf_qsilu_lut_cache;
static _Thread_local VxPfQactLutCache g_vx_pf_qgelu_lut_cache;

static uint32_t vx_pf_qact_f32_bits(float value) {
    union { float f; uint32_t u; } bits = { value };
    return bits.u;
}

static uint32_t vx_pf_qact_lut_hash(uint32_t input_scale_bits,
        int32_t input_zero_point, uint32_t output_scale_bits,
        int32_t output_zero_point, uint32_t input_dtype, uint32_t output_dtype) {
    uint32_t hash = input_scale_bits * 0x9e3779b1u;
    hash ^= output_scale_bits + 0x85ebca6bu + (hash << 6u) + (hash >> 2u);
    hash ^= (uint32_t)input_zero_point * 0xc2b2ae35u;
    hash ^= (uint32_t)output_zero_point * 0x27d4eb2fu;
    hash ^= input_dtype * 0x165667b1u;
    hash ^= output_dtype * 0xd3a2646cu;
    hash ^= hash >> 16u;
    return hash;
}

static int vx_pf_qact_lut_matches(const VxPfQactLutEntry *entry,
        uint32_t input_scale_bits, int32_t input_zero_point,
        uint32_t output_scale_bits, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    return entry->valid && entry->input_scale_bits == input_scale_bits &&
        entry->output_scale_bits == output_scale_bits &&
        entry->input_zero_point == input_zero_point &&
        entry->output_zero_point == output_zero_point &&
        entry->input_dtype == input_dtype && entry->output_dtype == output_dtype;
}

/* Return a matching entry, reserve one for a sufficiently large cold call, or
 * return NULL to keep a small cold call scalar.  valid remains clear while the
 * caller fills a reserved table. */
static VxPfQactLutEntry *vx_pf_qact_lut_acquire(VxPfQactLutCache *cache,
        uint32_t elements, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype, int *needs_build) {
    const uint32_t input_scale_bits = vx_pf_qact_f32_bits(input_scale);
    const uint32_t output_scale_bits = vx_pf_qact_f32_bits(output_scale);
    const uint32_t set_index = vx_pf_qact_lut_hash(input_scale_bits,
        input_zero_point, output_scale_bits, output_zero_point,
        input_dtype, output_dtype) & (VX_PF_QACT_LUT_SETS - 1u);
    VxPfQactLutEntry *set = cache->entries[set_index];
    uint32_t way;
    *needs_build = 0;
    for (way = 0; way < VX_PF_QACT_LUT_WAYS; way++) {
        if (vx_pf_qact_lut_matches(&set[way], input_scale_bits,
                input_zero_point, output_scale_bits, output_zero_point,
                input_dtype, output_dtype)) return &set[way];
    }
    if (elements < VX_PF_QACT_LUT_BUILD_THRESHOLD) return NULL;
    for (way = 0; way < VX_PF_QACT_LUT_WAYS && set[way].valid; way++) {}
    if (way == VX_PF_QACT_LUT_WAYS) {
        way = cache->next_victim[set_index];
        cache->next_victim[set_index] =
            (uint8_t)((way + 1u) % VX_PF_QACT_LUT_WAYS);
    }
    set[way].valid = 0;
    set[way].input_scale_bits = input_scale_bits;
    set[way].output_scale_bits = output_scale_bits;
    set[way].input_zero_point = input_zero_point;
    set[way].output_zero_point = output_zero_point;
    set[way].input_dtype = (uint8_t)input_dtype;
    set[way].output_dtype = (uint8_t)output_dtype;
    *needs_build = 1;
    return &set[way];
}

static int32_t vx_pf_qact_raw_value(uint8_t raw_byte, uint32_t input_dtype) {
    if (input_dtype == VX_DTYPE_I8 && raw_byte >= 128u)
        return (int32_t)raw_byte - 256;
    return (int32_t)raw_byte;
}

static uint8_t vx_pf_qsilu_quantized_byte(uint8_t raw_byte,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, int32_t output_minimum, int32_t output_maximum) {
    const float input_value = (float)(vx_pf_qact_raw_value(raw_byte, input_dtype) -
        input_zero_point) * input_scale;
    const float activated = input_value / (1.0f + expf(-input_value));
    const float transformed = activated / output_scale + (float)output_zero_point;
    const int32_t quantized = vx_w8a8_requantize(transformed,
        output_minimum, output_maximum, output_zero_point);
    return (uint8_t)quantized;
}

/* Canonical byte-quantized SiLU.  It keeps the activation in its declared
 * I8/U8 representation at both graph boundaries: decode one scalar using the
 * immutable input descriptor, apply x * sigmoid(x), then requantize straight
 * into the immutable output descriptor.  Validate all metadata before the
 * first store so malformed direct WASM/native calls fail closed. */
WASM_EXPORT("qsilu_i8u8")
int qsilu_i8u8(const void *input, void *output, uint32_t elements,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    int32_t output_minimum, output_maximum;
    int needs_build;
    VxPfQactLutEntry *entry;
    const uint8_t *input_bytes = (const uint8_t *)input;
    uint8_t *output_bytes = (uint8_t *)output;
    if (!input || !output || input == output || !elements ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype)) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    entry = vx_pf_qact_lut_acquire(&g_vx_pf_qsilu_lut_cache, elements,
        input_scale, input_zero_point, output_scale, output_zero_point,
        input_dtype, output_dtype, &needs_build);
    if (entry && needs_build) {
        for (uint32_t raw_byte = 0; raw_byte < VX_PF_QACT_LUT_VALUES; raw_byte++) {
            entry->values[raw_byte] = vx_pf_qsilu_quantized_byte((uint8_t)raw_byte,
                input_scale, input_zero_point, output_scale, output_zero_point,
                input_dtype, output_minimum, output_maximum);
        }
        entry->valid = 1;
    }
    if (entry) {
        for (uint32_t index = 0; index < elements; index++)
            output_bytes[index] = entry->values[input_bytes[index]];
        return 1;
    }
    for (uint32_t index = 0; index < elements; index++) {
        output_bytes[index] = vx_pf_qsilu_quantized_byte(input_bytes[index],
            input_scale, input_zero_point, output_scale, output_zero_point,
            input_dtype, output_minimum, output_maximum);
    }
    return 1;
}

/* Keep this explicit F32 operation order aligned with gELU.wgsl's fixed
 * Abramowitz-Stegun 7.1.26 path.  In particular, do not call native erff():
 * QGELU must produce the same canonical quantization boundary in WASM and
 * shader-backed execution. */
static float vx_pf_qgelu_erf_approx(float value) {
    const float sign = value >= 0.0f ? 1.0f : -1.0f;
    const float magnitude = fabsf(value);
    const float denominator = 1.0f + 0.3275911f * magnitude;
    const float t = 1.0f / denominator;
    float polynomial = 1.061405429f * t;
    polynomial = polynomial - 1.453152027f;
    polynomial = polynomial * t;
    polynomial = polynomial + 1.421413741f;
    polynomial = polynomial * t;
    polynomial = polynomial - 0.284496736f;
    polynomial = polynomial * t;
    polynomial = polynomial + 0.254829592f;
    polynomial = polynomial * t;
    {
        const float exponential = expf(-(magnitude * magnitude));
        return sign * (1.0f - polynomial * exponential);
    }
}

static uint8_t vx_pf_qgelu_quantized_byte(uint8_t raw_byte,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, int32_t output_minimum, int32_t output_maximum) {
    const float value = (float)(vx_pf_qact_raw_value(raw_byte, input_dtype) -
        input_zero_point) * input_scale;
    const float erf_input = value * 0.7071067811865476f;
    const float cdf = 0.5f * (1.0f + vx_pf_qgelu_erf_approx(erf_input));
    const float gelu = value * cdf;
    const float scaled = gelu / output_scale;
    const float transformed = scaled + (float)output_zero_point;
    const int32_t quantized = vx_w8a8_requantize(transformed,
        output_minimum, output_maximum, output_zero_point);
    return (uint8_t)quantized;
}

/* Canonical byte-quantized GELU with fixed approximate='none' semantics.  It
 * owns no approximation argument: graph wrappers may accept either their
 * parameter-free spelling or an explicit approximate='none', but this ABI is
 * deliberately one canonical A-S polynomial.  As with QSiLU, all descriptor
 * checks precede the first output byte write. */
WASM_EXPORT("qgelu_i8u8")
int qgelu_i8u8(const void *input, void *output, uint32_t elements,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    int32_t output_minimum, output_maximum;
    int needs_build;
    VxPfQactLutEntry *entry;
    const uint8_t *input_bytes = (const uint8_t *)input;
    uint8_t *output_bytes = (uint8_t *)output;
    if (!input || !output || input == output || !elements ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype)) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    entry = vx_pf_qact_lut_acquire(&g_vx_pf_qgelu_lut_cache, elements,
        input_scale, input_zero_point, output_scale, output_zero_point,
        input_dtype, output_dtype, &needs_build);
    if (entry && needs_build) {
        for (uint32_t raw_byte = 0; raw_byte < VX_PF_QACT_LUT_VALUES; raw_byte++) {
            entry->values[raw_byte] = vx_pf_qgelu_quantized_byte((uint8_t)raw_byte,
                input_scale, input_zero_point, output_scale, output_zero_point,
                input_dtype, output_minimum, output_maximum);
        }
        entry->valid = 1;
    }
    if (entry) {
        for (uint32_t index = 0; index < elements; index++)
            output_bytes[index] = entry->values[input_bytes[index]];
        return 1;
    }
    for (uint32_t index = 0; index < elements; index++) {
        output_bytes[index] = vx_pf_qgelu_quantized_byte(input_bytes[index],
            input_scale, input_zero_point, output_scale, output_zero_point,
            input_dtype, output_minimum, output_maximum);
    }
    return 1;
}

/* SafeTensors byte offsets are not required to be F32-aligned.  Read affine
 * coefficients bytewise so the portable ABI remains valid for mapped model
 * storage as well as naturally aligned caller arrays.  SafeTensors and the
 * supported native/WASM targets use IEEE little-endian F32. */
/* Canonical NHWC byte-domain GroupNorm.  Statistics deliberately stay in
 * centered raw-byte units until the biased variance is complete; only then is
 * it converted into real-value units through input_scale^2.  This matches the
 * paired qGroupNormStats/qGroupNormApply shaders without materializing an F32
 * activation tensor.  Every descriptor, affine coefficient, and overlap is
 * preflighted before output storage is touched. */
/* Channel ceiling for the per-channel statistic fan-out below. Anything wider
 * falls back to the strided per-group transform, which needs no scratch. */
#define VX_QGROUPNORM_MAX_CHANNELS 4096

WASM_EXPORT("qgroupnorm_i8u8")
int qgroupnorm_i8u8(const void *input, const float *weight, const float *bias,
        void *output, uint32_t batch, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t groups, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype) {
    size_t area = height;
    size_t elements = batch;
    size_t values_per_group;
    size_t affine_bytes;
    int32_t output_minimum;
    int32_t output_maximum;
    if (!input || !weight || !bias || !output || !batch || !height || !width ||
        !channels || !groups || channels % groups ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_pf_finite_f32(epsilon) || epsilon <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_pf_mul_size(&area, width) || !vx_pf_mul_size(&elements, height) ||
        !vx_pf_mul_size(&elements, width) || !vx_pf_mul_size(&elements, channels) ||
        elements > UINT32_MAX) return 0;
    values_per_group = area;
    if (!vx_pf_mul_size(&values_per_group, channels / groups)) return 0;
    affine_bytes = channels;
    if (!vx_pf_mul_size(&affine_bytes, sizeof(float))) return 0;
    if (vx_w8a8_ranges_overlap(output, elements, input, elements) ||
        vx_w8a8_ranges_overlap(output, elements, weight, affine_bytes) ||
        vx_w8a8_ranges_overlap(output, elements, bias, affine_bytes)) return 0;
    for (uint32_t channel = 0; channel < channels; channel++) {
        if (!vx_pf_finite_f32(vx_w8a8_affine_f32_at(weight, channel)) ||
            !vx_pf_finite_f32(vx_w8a8_affine_f32_at(bias, channel))) return 0;
    }
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    {
        const uint32_t channels_per_group = channels / groups;
        const size_t sample_stride = area * channels;
        /* Per-channel copies of each group's (mean, 1/sigma), so the final
         * elementwise transform can sweep the activation contiguously. */
        float channel_mean[VX_QGROUPNORM_MAX_CHANNELS];
        float channel_inverse[VX_QGROUPNORM_MAX_CHANNELS];
        const int deferred = channels <= VX_QGROUPNORM_MAX_CHANNELS;
        for (uint32_t sample = 0; sample < batch; sample++) {
            const size_t sample_offset = (size_t)sample * sample_stride;
            for (uint32_t group = 0; group < groups; group++) {
                const uint32_t channel_start = group * channels_per_group;
                float sum = 0.0f;
                for (size_t spatial = 0; spatial < area; spatial++) {
                    const size_t offset = sample_offset + spatial * channels + channel_start;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const float raw = (float)(vx_w8a8_byte_value(input, input_dtype,
                            offset + local) - input_zero_point);
                        sum = sum + raw;
                    }
                }
                {
                    const float mean_raw = sum / (float)values_per_group;
                    float centered_square_sum = 0.0f;
                    for (size_t spatial = 0; spatial < area; spatial++) {
                        const size_t offset = sample_offset + spatial * channels + channel_start;
                        for (uint32_t local = 0; local < channels_per_group; local++) {
                            const float raw = (float)(vx_w8a8_byte_value(input, input_dtype,
                                offset + local) - input_zero_point);
                            const float centered = raw - mean_raw;
                            centered_square_sum = centered_square_sum + centered * centered;
                        }
                    }
                    {
                        float raw_variance = centered_square_sum / (float)values_per_group;
                        if (raw_variance < 0.0f) raw_variance = 0.0f;
                        {
                            const float scaled_variance = raw_variance * input_scale;
                            float real_variance = scaled_variance * input_scale;
                            if (real_variance < 0.0f) real_variance = 0.0f;
                            {
                                const float inverse_stddev = 1.0f / sqrtf(real_variance + epsilon);
                                /* The statistics above are order-dependent
                                 * reductions and stay exactly as written. This
                                 * final transform is elementwise, so publishing
                                 * (mean, 1/sigma) per channel lets a single
                                 * linear vectorized sweep replace the strided
                                 * per-group walk without touching any partial
                                 * sum. See the sweep after this loop nest. */
                                if (deferred) {
                                    for (uint32_t local = 0; local < channels_per_group; local++) {
                                        channel_mean[channel_start + local] = mean_raw;
                                        channel_inverse[channel_start + local] = inverse_stddev;
                                    }
                                } else {
                                    /* Wider than the scratch: transform in place
                                     * with the original strided walk. */
                                    for (size_t spatial = 0; spatial < area; spatial++) {
                                        const size_t offset = sample_offset +
                                            spatial * channels + channel_start;
                                        for (uint32_t local = 0; local < channels_per_group; local++) {
                                            const uint32_t channel = channel_start + local;
                                            const float raw = (float)(vx_w8a8_byte_value(input,
                                                input_dtype, offset + local) - input_zero_point);
                                            const float scaled = (raw - mean_raw) * input_scale;
                                            const float normalized = scaled * inverse_stddev;
                                            const float affine = normalized *
                                                vx_w8a8_affine_f32_at(weight, channel) +
                                                vx_w8a8_affine_f32_at(bias, channel);
                                            const float output_scaled = affine / output_scale;
                                            const float transformed = output_scaled +
                                                (float)output_zero_point;
                                            vx_w8a8_store_byte(output, output_dtype, offset + local,
                                                vx_w8a8_requantize(transformed, output_minimum,
                                                    output_maximum, output_zero_point));
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
            /* Elementwise transform, one linear sweep per sample.  Each output
             * runs the identical scalar sequence the per-group walk ran, so the
             * result is bit-identical; only the traversal order changed, and
             * this order is contiguous instead of `groups` strided passes. */
            if (deferred) for (size_t spatial = 0; spatial < area; spatial++) {
                const size_t row = sample_offset + spatial * channels;
                for (uint32_t channel = 0; channel < channels; channel++) {
                    const size_t index = row + channel;
                    const float raw = (float)(vx_w8a8_byte_value(input,
                        input_dtype, index) - input_zero_point);
                    const float scaled = (raw - channel_mean[channel]) * input_scale;
                    const float normalized = scaled * channel_inverse[channel];
                    const float affine = normalized *
                        vx_w8a8_affine_f32_at(weight, channel) +
                        vx_w8a8_affine_f32_at(bias, channel);
                    const float output_scaled = affine / output_scale;
                    const float transformed = output_scaled + (float)output_zero_point;
                    vx_w8a8_store_byte(output, output_dtype, index,
                        vx_w8a8_requantize(transformed, output_minimum,
                            output_maximum, output_zero_point));
                }
            }
        }
    }
    return 1;
}

/* Canonical byte-domain LayerNorm over `rows` contiguous final-axis rows.
 * Mean and biased variance are reduced in centered raw-byte units, then the
 * variance is converted to real-value units with input_scale^2.  This is the
 * portable counterpart to qLayerNormStats/qLayerNormApply and never expands a
 * byte activation into an F32 activation tensor. */
WASM_EXPORT("qlayernorm_i8u8")
int qlayernorm_i8u8(const void *input, const float *weight, const float *bias,
        void *output, uint32_t rows, uint32_t d_model, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, float epsilon, uint32_t input_dtype,
        uint32_t output_dtype) {
    size_t elements = rows;
    size_t affine_bytes = d_model;
    int32_t output_minimum;
    int32_t output_maximum;
    if (!input || !weight || !bias || !output || !rows || !d_model ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_pf_finite_f32(epsilon) || epsilon <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_pf_mul_size(&elements, d_model) || elements > UINT32_MAX ||
        !vx_pf_mul_size(&affine_bytes, sizeof(float))) return 0;
    if (vx_w8a8_ranges_overlap(output, elements, input, elements) ||
        vx_w8a8_ranges_overlap(output, elements, weight, affine_bytes) ||
        vx_w8a8_ranges_overlap(output, elements, bias, affine_bytes)) return 0;
    for (uint32_t channel = 0; channel < d_model; channel++) {
        if (!vx_pf_finite_f32(vx_w8a8_affine_f32_at(weight, channel)) ||
            !vx_pf_finite_f32(vx_w8a8_affine_f32_at(bias, channel))) return 0;
    }
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t row = 0; row < rows; row++) {
        const size_t offset = (size_t)row * d_model;
        float sum = 0.0f;
        for (uint32_t channel = 0; channel < d_model; channel++) {
            const float raw = (float)(vx_w8a8_byte_value(input, input_dtype,
                offset + channel) - input_zero_point);
            sum = sum + raw;
        }
        {
            const float mean_raw = sum / (float)d_model;
            float centered_square_sum = 0.0f;
            for (uint32_t channel = 0; channel < d_model; channel++) {
                const float raw = (float)(vx_w8a8_byte_value(input, input_dtype,
                    offset + channel) - input_zero_point);
                const float centered = raw - mean_raw;
                centered_square_sum = centered_square_sum + centered * centered;
            }
            {
                float raw_variance = centered_square_sum / (float)d_model;
                if (raw_variance < 0.0f) raw_variance = 0.0f;
                {
                    const float scaled_variance = raw_variance * input_scale;
                    float real_variance = scaled_variance * input_scale;
                    if (real_variance < 0.0f) real_variance = 0.0f;
                    {
                        const float inverse_stddev = 1.0f / sqrtf(real_variance + epsilon);
                        for (uint32_t channel = 0; channel < d_model; channel++) {
                            const float raw = (float)(vx_w8a8_byte_value(input,
                                input_dtype, offset + channel) - input_zero_point);
                            const float normalized = (raw - mean_raw) * input_scale *
                                inverse_stddev;
                            const float affine = normalized *
                                vx_w8a8_affine_f32_at(weight, channel) +
                                vx_w8a8_affine_f32_at(bias, channel);
                            const float transformed = affine / output_scale +
                                (float)output_zero_point;
                            const int32_t quantized = vx_w8a8_requantize(
                                transformed, output_minimum, output_maximum,
                                output_zero_point);
                            vx_w8a8_store_byte(output, output_dtype,
                                                    offset + channel, quantized);
                        }
                    }
                }
            }
        }
    }
    return 1;
}

/* The canonical QSDPA mask layouts deliberately match the native graph and
 * browser spellings: no mask, [K], [B,K], [Q,K], or [B,Q,K].  A nonzero I32
 * entry keeps a key; zero masks it. */
static int vx_pf_qsdpa_mask_elements(uint32_t batch, uint32_t seq_q,
        uint32_t seq_kv, uint32_t mask_mode, size_t *elements_out) {
    size_t elements;
    if (!elements_out) return 0;
    if (mask_mode == 0u) {
        *elements_out = 0;
        return 1;
    }
    elements = seq_kv;
    if (mask_mode == 1u) {
        *elements_out = elements;
        return 1;
    }
    if (mask_mode == 2u) {
        if (!vx_pf_mul_size(&elements, batch)) return 0;
        *elements_out = elements;
        return 1;
    }
    if (mask_mode == 3u) {
        if (!vx_pf_mul_size(&elements, seq_q)) return 0;
        *elements_out = elements;
        return 1;
    }
    if (mask_mode == 4u) {
        if (!vx_pf_mul_size(&elements, seq_q) ||
            !vx_pf_mul_size(&elements, batch)) return 0;
        *elements_out = elements;
        return 1;
    }
    return 0;
}

/* SafeTensors offsets need not preserve I32 alignment.  The portable ABI is
 * little-endian on every supported native/WASM target, so decode keep-mask
 * values bytewise rather than imposing an alignment requirement on callers. */
static int32_t vx_pf_qsdpa_mask_i32_at(const int32_t *mask, size_t index) {
    const unsigned char *bytes = (const unsigned char *)mask + index * sizeof(int32_t);
    const uint32_t bits = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
    return (int32_t)bits;
}

static int vx_pf_qsdpa_key_allowed(const int32_t *mask, uint32_t mask_mode,
        uint32_t batch_index, uint32_t query, uint32_t key, uint32_t seq_q,
        uint32_t seq_kv, uint32_t causal) {
    size_t index;
    if (causal && key > query) return 0;
    if (mask_mode == 0u) return 1;
    if (mask_mode == 1u) return vx_pf_qsdpa_mask_i32_at(mask, key) != 0;
    if (mask_mode == 2u) {
        index = (size_t)batch_index * seq_kv + key;
        return vx_pf_qsdpa_mask_i32_at(mask, index) != 0;
    }
    if (mask_mode == 3u) {
        index = (size_t)query * seq_kv + key;
        return vx_pf_qsdpa_mask_i32_at(mask, index) != 0;
    }
    index = ((size_t)batch_index * seq_q + query) * seq_kv + key;
    return vx_pf_qsdpa_mask_i32_at(mask, index) != 0;
}

#if defined(__wasm_simd128__)
/* QSDPA is one of the largest remaining WASM seed kernels. Its centered
 * byte products are proven I32-safe by qsdpa_i8u8() before execution, so SIMD
 * may regroup the exact integer sum without changing the result.  Keep this
 * local to the WASM build: native execution has its own parallel dispatcher. */
static int32_t vx_pf_qsdpa_wasm_centered_dot(const void *q, const void *k,
        size_t q_base, size_t k_base, uint32_t dimensions,
        uint32_t q_dtype, uint32_t k_dtype, int32_t q_zero_point,
        int32_t k_zero_point) {
    const uint8_t *q_bytes = (const uint8_t *)q + q_base;
    const uint8_t *k_bytes = (const uint8_t *)k + k_base;
    const v128_t q_zp = wasm_i16x8_splat((int16_t)q_zero_point);
    const v128_t k_zp = wasm_i16x8_splat((int16_t)k_zero_point);
    v128_t sums = wasm_i32x4_splat(0);
    uint32_t dimension = 0;
    for (; dimension + 16u <= dimensions; dimension += 16u) {
        const v128_t q8 = wasm_v128_load(q_bytes + dimension);
        const v128_t k8 = wasm_v128_load(k_bytes + dimension);
        const v128_t q_lo = wasm_i16x8_sub(q_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_low_i8x16(q8)
            : wasm_u16x8_extend_low_u8x16(q8), q_zp);
        const v128_t q_hi = wasm_i16x8_sub(q_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_high_i8x16(q8)
            : wasm_u16x8_extend_high_u8x16(q8), q_zp);
        const v128_t k_lo = wasm_i16x8_sub(k_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_low_i8x16(k8)
            : wasm_u16x8_extend_low_u8x16(k8), k_zp);
        const v128_t k_hi = wasm_i16x8_sub(k_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_high_i8x16(k8)
            : wasm_u16x8_extend_high_u8x16(k8), k_zp);
        sums = wasm_i32x4_add(sums, wasm_i32x4_dot_i16x8(q_lo, k_lo));
        sums = wasm_i32x4_add(sums, wasm_i32x4_dot_i16x8(q_hi, k_hi));
    }
    if (dimension + 8u <= dimensions) {
        const v128_t q8 = wasm_v128_load64_zero(q_bytes + dimension);
        const v128_t k8 = wasm_v128_load64_zero(k_bytes + dimension);
        const v128_t q16 = wasm_i16x8_sub(q_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_low_i8x16(q8)
            : wasm_u16x8_extend_low_u8x16(q8), q_zp);
        const v128_t k16 = wasm_i16x8_sub(k_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_low_i8x16(k8)
            : wasm_u16x8_extend_low_u8x16(k8), k_zp);
        sums = wasm_i32x4_add(sums, wasm_i32x4_dot_i16x8(q16, k16));
        dimension += 8u;
    }
    {
        int32_t sum = wasm_i32x4_extract_lane(sums, 0) +
            wasm_i32x4_extract_lane(sums, 1) +
            wasm_i32x4_extract_lane(sums, 2) +
            wasm_i32x4_extract_lane(sums, 3);
        for (; dimension < dimensions; dimension++) {
            const int32_t q_value = vx_w8a8_byte_value(
                q, q_dtype, q_base + dimension) - q_zero_point;
            const int32_t k_value = vx_w8a8_byte_value(
                k, k_dtype, k_base + dimension) - k_zero_point;
            sum += q_value * k_value;
        }
        return sum;
    }
}

/* Dot over Q and K already centered into I16.
 *
 * The byte helper re-widens and re-centers K for every query, so each K element
 * is decoded seq_q times. Centering one head's K once leaves only the loads and
 * the dot here. The products are the same integers either way, and the sum is
 * proven I32-safe before execution, so the result is identical. */
static int32_t vx_pf_qsdpa_wasm_dot_i16(const int16_t *q, const int16_t *k,
        uint32_t dimensions) {
    v128_t sums = wasm_i32x4_splat(0);
    uint32_t dimension = 0;
    int32_t sum;
    for (; dimension + 8u <= dimensions; dimension += 8u) {
        sums = wasm_i32x4_add(sums, wasm_i32x4_dot_i16x8(
            wasm_v128_load(q + dimension), wasm_v128_load(k + dimension)));
    }
    sum = wasm_i32x4_extract_lane(sums, 0) + wasm_i32x4_extract_lane(sums, 1) +
          wasm_i32x4_extract_lane(sums, 2) + wasm_i32x4_extract_lane(sums, 3);
    for (; dimension < dimensions; dimension++)
        sum += (int32_t)q[dimension] * (int32_t)k[dimension];
    return sum;
}

enum {
    VX_PF_QSDPA_ACCUMULATOR_INITIALIZE = 0,
    VX_PF_QSDPA_ACCUMULATOR_RESCALE = 1,
    VX_PF_QSDPA_ACCUMULATOR_ADD = 2,
};

/* Same recurrence as the byte helper below, reading V already centered and
 * widened to F32.
 *
 * The byte helper re-decodes V for every (query, key) pair, so each V element
 * is converted seq_q times -- 402 here. Converting one head's V once and
 * reusing it across every query is the same arithmetic: the scale is still
 * applied in the original position of each expression, so `(weight * value) *
 * scale` stays exactly that and the result is bit-identical. */
static void vx_pf_qsdpa_wasm_accumulate_centered(float *accumulator,
        const float *v_centered, uint32_t dimensions, float v_scale,
        float weight, uint32_t mode) {
    const v128_t scale = wasm_f32x4_splat(v_scale);
    const v128_t weight_vector = wasm_f32x4_splat(weight);
    for (uint32_t dimension = 0; dimension < dimensions; dimension += 4u) {
        const v128_t values = wasm_v128_load(v_centered + dimension);
        v128_t next;
        if (mode == VX_PF_QSDPA_ACCUMULATOR_INITIALIZE) {
            next = wasm_f32x4_mul(values, scale);
        } else if (mode == VX_PF_QSDPA_ACCUMULATOR_RESCALE) {
            next = wasm_f32x4_add(
                wasm_f32x4_mul(wasm_v128_load(accumulator + dimension),
                               weight_vector),
                wasm_f32x4_mul(values, scale));
        } else {
            const v128_t contribution = wasm_f32x4_mul(
                wasm_f32x4_mul(weight_vector, values), scale);
            next = wasm_f32x4_add(wasm_v128_load(accumulator + dimension),
                                  contribution);
        }
        wasm_v128_store(accumulator + dimension, next);
    }
}

/* Each F32 lane still observes keys in the original order.  The explicit
 * multiply ordering mirrors the scalar recurrence so baseline SIMD remains
 * byte-identical at the final ties-to-even requantization boundary. */
static void vx_pf_qsdpa_wasm_accumulate(float *accumulator, const void *v,
        size_t v_base, uint32_t dimensions, uint32_t v_dtype,
        int32_t v_zero_point, float v_scale, float weight, uint32_t mode) {
    const uint8_t *v_bytes = (const uint8_t *)v + v_base;
    const v128_t zero_point = wasm_i32x4_splat(v_zero_point);
    const v128_t scale = wasm_f32x4_splat(v_scale);
    const v128_t weight_vector = wasm_f32x4_splat(weight);
    for (uint32_t dimension = 0; dimension < dimensions; dimension += 4u) {
        const v128_t bytes = wasm_v128_load32_zero(v_bytes + dimension);
        const v128_t values16 = v_dtype == VX_DTYPE_I8
            ? wasm_i16x8_extend_low_i8x16(bytes)
            : wasm_u16x8_extend_low_u8x16(bytes);
        const v128_t centered = wasm_i32x4_sub(
            wasm_i32x4_extend_low_i16x8(values16), zero_point);
        const v128_t values = wasm_f32x4_convert_i32x4(centered);
        v128_t next;
        if (mode == VX_PF_QSDPA_ACCUMULATOR_INITIALIZE) {
            next = wasm_f32x4_mul(values, scale);
        } else if (mode == VX_PF_QSDPA_ACCUMULATOR_RESCALE) {
            next = wasm_f32x4_add(
                wasm_f32x4_mul(wasm_v128_load(accumulator + dimension),
                               weight_vector),
                wasm_f32x4_mul(values, scale));
        } else {
            const v128_t contribution = wasm_f32x4_mul(
                wasm_f32x4_mul(weight_vector, values), scale);
            next = wasm_f32x4_add(wasm_v128_load(accumulator + dimension),
                                  contribution);
        }
        wasm_v128_store(accumulator + dimension, next);
    }
}
#endif

/* Canonical W8A8 cross/self attention over distinct Q, K, and V tensors.
 * QK accumulation is exact centered-byte I32 arithmetic.  A stable online
 * softmax keeps just a max, a sum, and one F32 value vector (head_dim <= 64)
 * in local storage, so scores and dequantized activations never become graph
 * tensors.  The all-masked case is intentionally the output zero point. */
/* One head's V, centered and widened to F32, reused by every query of that
 * head. Sized for seq_kv * head_dim; larger heads keep the per-key byte path. */
enum { VX_PF_QSDPA_V_SCRATCH_FLOATS = 65536u };
static float vx_pf_qsdpa_v_centered[VX_PF_QSDPA_V_SCRATCH_FLOATS];
enum { VX_PF_QSDPA_ALLOWED_SCRATCH = 8192u };
static unsigned char vx_pf_qsdpa_key_allowed_cache[VX_PF_QSDPA_ALLOWED_SCRATCH];
/* One head's K, centered into I16 and padded to a multiple of eight so the dot
 * never falls into its scalar tail. */
static int16_t vx_pf_qsdpa_k_centered[VX_PF_QSDPA_V_SCRATCH_FLOATS + 8u];

WASM_EXPORT("qsdpa_i8u8")
int qsdpa_i8u8(const void *q, const void *k, const void *v,
        const int32_t *mask, void *output, uint32_t batch, uint32_t seq_q,
        uint32_t seq_kv, uint32_t d_model, uint32_t heads, float q_scale,
        int32_t q_zero_point, float k_scale, int32_t k_zero_point,
        float v_scale, int32_t v_zero_point, float output_scale,
        int32_t output_zero_point, float attention_scale, uint32_t q_dtype,
        uint32_t k_dtype, uint32_t v_dtype, uint32_t output_dtype,
        uint32_t causal, uint32_t mask_mode) {
    size_t q_elements = batch;
    size_t kv_elements = batch;
    size_t mask_elements = 0;
    size_t mask_bytes = 0;
    uint32_t head_dim;
    int32_t output_minimum;
    int32_t output_maximum;
    float qk_scale;
    float score_scale;
    uint64_t q_magnitude;
    uint64_t k_magnitude;
    uint64_t maximum_dot;
    if (!q || !k || !v || !output || !batch || !seq_q || !seq_kv ||
        !d_model || !heads || causal > 1u ||
        !vx_w8a8_byte_dtype(q_dtype) ||
        !vx_w8a8_byte_dtype(k_dtype) ||
        !vx_w8a8_byte_dtype(v_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(q_scale) || q_scale <= 0.0f ||
        !vx_pf_finite_f32(k_scale) || k_scale <= 0.0f ||
        !vx_pf_finite_f32(v_scale) || v_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_pf_finite_f32(attention_scale) || attention_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(q_zero_point, q_dtype) ||
        !vx_w8a8_zero_point_valid(k_zero_point, k_dtype) ||
        !vx_w8a8_zero_point_valid(v_zero_point, v_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        d_model % heads || d_model % 4u) return 0;
    head_dim = d_model / heads;
    if (!head_dim || head_dim % 4u || head_dim > 64u ||
        !vx_pf_mul_size(&q_elements, seq_q) ||
        !vx_pf_mul_size(&q_elements, d_model) ||
        !vx_pf_mul_size(&kv_elements, seq_kv) ||
        !vx_pf_mul_size(&kv_elements, d_model) ||
        q_elements > UINT32_MAX || kv_elements > UINT32_MAX ||
        !vx_pf_qsdpa_mask_elements(batch, seq_q, seq_kv, mask_mode,
                                    &mask_elements) ||
        mask_elements > UINT32_MAX ||
        (mask_mode == 0u ? mask != NULL : mask == NULL) ||
        (mask_elements && !vx_pf_mul_size(&mask_elements, sizeof(*mask)))) return 0;
    mask_bytes = mask_elements;
    qk_scale = q_scale * k_scale;
    score_scale = qk_scale * attention_scale;
    /* An infinite score multiplier would make +/-infinity subtraction in an
     * online recurrence undefined.  The same finite F32 multiplier is checked
     * by the GPU descriptor path. */
    if (!vx_pf_finite_f32(qk_scale) || qk_scale <= 0.0f ||
        !vx_pf_finite_f32(score_scale) || score_scale <= 0.0f) return 0;
    {
        const int64_t q_low =
            (int64_t)(q_dtype == VX_DTYPE_I8 ? -128 : 0) - q_zero_point;
        const int64_t q_high =
            (int64_t)(q_dtype == VX_DTYPE_I8 ? 127 : 255) - q_zero_point;
        const int64_t k_low =
            (int64_t)(k_dtype == VX_DTYPE_I8 ? -128 : 0) - k_zero_point;
        const int64_t k_high =
            (int64_t)(k_dtype == VX_DTYPE_I8 ? 127 : 255) - k_zero_point;
        q_magnitude = (uint64_t)(q_low < 0 ? -q_low : q_low);
        if ((uint64_t)(q_high < 0 ? -q_high : q_high) > q_magnitude)
            q_magnitude = (uint64_t)(q_high < 0 ? -q_high : q_high);
        k_magnitude = (uint64_t)(k_low < 0 ? -k_low : k_low);
        if ((uint64_t)(k_high < 0 ? -k_high : k_high) > k_magnitude)
            k_magnitude = (uint64_t)(k_high < 0 ? -k_high : k_high);
    }
    maximum_dot = q_magnitude * k_magnitude * head_dim;
    if (!vx_pf_finite_f32((float)maximum_dot * score_scale)) return 0;
    if (vx_w8a8_ranges_overlap(output, q_elements, q, q_elements) ||
        vx_w8a8_ranges_overlap(output, q_elements, k, kv_elements) ||
        vx_w8a8_ranges_overlap(output, q_elements, v, kv_elements) ||
        (mask_bytes && vx_w8a8_ranges_overlap(output, q_elements, mask, mask_bytes))) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    /* Head outside query so one head's V decode is shared by every query. Each
     * (query, head) output is independent, so the swap changes no result. */
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        const size_t q_batch_base = (size_t)batch_index * seq_q * d_model;
        const size_t kv_batch_base = (size_t)batch_index * seq_kv * d_model;
        const int use_v_scratch =
            (size_t)seq_kv * head_dim <= VX_PF_QSDPA_V_SCRATCH_FLOATS;
        /* For these modes the allowance depends only on the key, so resolve it
         * once per batch rather than dispatching mask_mode per (query, head,
         * key) -- ten million times on this encoder. */
        const int allowance_is_key_only =
            !causal && mask_mode <= 2u &&
            (size_t)seq_kv <= VX_PF_QSDPA_ALLOWED_SCRATCH;
        if (allowance_is_key_only) {
            for (uint32_t key = 0; key < seq_kv; key++)
                vx_pf_qsdpa_key_allowed_cache[key] = (unsigned char)
                    vx_pf_qsdpa_key_allowed(mask, mask_mode, batch_index,
                                            0u, key, seq_q, seq_kv, causal);
        }
        for (uint32_t head = 0; head < heads; head++) {
            const uint32_t head_base = head * head_dim;
            if (use_v_scratch) {
                for (uint32_t key = 0; key < seq_kv; key++) {
                    const size_t row = kv_batch_base + (size_t)key * d_model + head_base;
                    float *destination = vx_pf_qsdpa_v_centered +
                        (size_t)key * head_dim;
                    int16_t *k_destination = vx_pf_qsdpa_k_centered +
                        (size_t)key * head_dim;
                    for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                        destination[dimension] = (float)(vx_w8a8_byte_value(
                            v, v_dtype, row + dimension) - v_zero_point);
                        k_destination[dimension] = (int16_t)(vx_w8a8_byte_value(
                            k, k_dtype, row + dimension) - k_zero_point);
                    }
                }
                /* The dot reads eight at a time; zero the padding so a
                 * head_dim that is not a multiple of eight still reads
                 * defined lanes that contribute nothing. */
                for (uint32_t pad = 0; pad < 8u; pad++)
                    vx_pf_qsdpa_k_centered[(size_t)seq_kv * head_dim + pad] = 0;
            }
            for (uint32_t query = 0; query < seq_q; query++) {
                const size_t q_row_base = q_batch_base + (size_t)query * d_model;
                float accumulator[64] = {0.0f};
                int16_t q_centered[72] = {0};
                if (use_v_scratch) {
                    for (uint32_t dimension = 0; dimension < head_dim; dimension++)
                        q_centered[dimension] = (int16_t)(vx_w8a8_byte_value(
                            q, q_dtype, q_row_base + head_base + dimension) -
                            q_zero_point);
                }
                float maximum_score = 0.0f;
                float sum = 0.0f;
                int have_key = 0;
                for (uint32_t key = 0; key < seq_kv; key++) {
                    int32_t dot = 0;
                    float score;
                    float weight;
                    const size_t kv_row_base = kv_batch_base + (size_t)key * d_model;
                    if (allowance_is_key_only
                            ? !vx_pf_qsdpa_key_allowed_cache[key]
                            : !vx_pf_qsdpa_key_allowed(mask, mask_mode,
                                  batch_index, query, key, seq_q, seq_kv,
                                  causal)) continue;
#if defined(__wasm_simd128__)
                    dot = use_v_scratch
                        ? vx_pf_qsdpa_wasm_dot_i16(q_centered,
                              vx_pf_qsdpa_k_centered + (size_t)key * head_dim,
                              head_dim)
                        : vx_pf_qsdpa_wasm_centered_dot(q, k,
                              q_row_base + head_base, kv_row_base + head_base,
                              head_dim, q_dtype, k_dtype, q_zero_point,
                              k_zero_point);
#else
                    for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                        const int32_t q_value = vx_w8a8_byte_value(q, q_dtype,
                            q_row_base + head_base + dimension) - q_zero_point;
                        const int32_t k_value = vx_w8a8_byte_value(k, k_dtype,
                            kv_row_base + head_base + dimension) - k_zero_point;
                        dot += q_value * k_value;
                    }
#endif
                    score = (float)dot * score_scale;
                    if (!have_key) {
                        maximum_score = score;
                        sum = 1.0f;
#if defined(__wasm_simd128__)
                        if (use_v_scratch)
                            vx_pf_qsdpa_wasm_accumulate_centered(accumulator,
                                vx_pf_qsdpa_v_centered + (size_t)key * head_dim,
                                head_dim, v_scale, 1.0f,
                                VX_PF_QSDPA_ACCUMULATOR_INITIALIZE);
                        else
                            vx_pf_qsdpa_wasm_accumulate(accumulator, v,
                                kv_row_base + head_base, head_dim, v_dtype,
                                v_zero_point, v_scale, 1.0f,
                                VX_PF_QSDPA_ACCUMULATOR_INITIALIZE);
#else
                        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                            const int32_t v_value = vx_w8a8_byte_value(v, v_dtype,
                                kv_row_base + head_base + dimension) - v_zero_point;
                            accumulator[dimension] = (float)v_value * v_scale;
                        }
#endif
                        have_key = 1;
                        continue;
                    }
                    if (score > maximum_score) {
                        weight = accurate_expf(maximum_score - score);
                        sum = sum * weight + 1.0f;
#if defined(__wasm_simd128__)
                        if (use_v_scratch)
                            vx_pf_qsdpa_wasm_accumulate_centered(accumulator,
                                vx_pf_qsdpa_v_centered + (size_t)key * head_dim,
                                head_dim, v_scale, weight,
                                VX_PF_QSDPA_ACCUMULATOR_RESCALE);
                        else
                            vx_pf_qsdpa_wasm_accumulate(accumulator, v,
                                kv_row_base + head_base, head_dim, v_dtype,
                                v_zero_point, v_scale, weight,
                                VX_PF_QSDPA_ACCUMULATOR_RESCALE);
#else
                        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                            const int32_t v_value = vx_w8a8_byte_value(v, v_dtype,
                                kv_row_base + head_base + dimension) - v_zero_point;
                            accumulator[dimension] = accumulator[dimension] * weight +
                                (float)v_value * v_scale;
                        }
#endif
                        maximum_score = score;
                    } else {
                        weight = accurate_expf(score - maximum_score);
                        sum += weight;
#if defined(__wasm_simd128__)
                        if (use_v_scratch)
                            vx_pf_qsdpa_wasm_accumulate_centered(accumulator,
                                vx_pf_qsdpa_v_centered + (size_t)key * head_dim,
                                head_dim, v_scale, weight,
                                VX_PF_QSDPA_ACCUMULATOR_ADD);
                        else
                            vx_pf_qsdpa_wasm_accumulate(accumulator, v,
                                kv_row_base + head_base, head_dim, v_dtype,
                                v_zero_point, v_scale, weight,
                                VX_PF_QSDPA_ACCUMULATOR_ADD);
#else
                        for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                            const int32_t v_value = vx_w8a8_byte_value(v, v_dtype,
                                kv_row_base + head_base + dimension) - v_zero_point;
                            accumulator[dimension] += weight * (float)v_value * v_scale;
                        }
#endif
                    }
                }
                for (uint32_t dimension = 0; dimension < head_dim; dimension++) {
                    const size_t output_index = q_row_base + head_base + dimension;
                    int32_t quantized = output_zero_point;
                    if (have_key && sum > 0.0f) {
                        const float value = accumulator[dimension] / sum;
                        const float transformed = value / output_scale +
                            (float)output_zero_point;
                        quantized = vx_w8a8_requantize(transformed,
                            output_minimum, output_maximum, output_zero_point);
                    }
                    vx_w8a8_store_byte(output, output_dtype, output_index,
                                           quantized);
                }
            }
        }
    }
    return 1;
}

/* Canonical byte-domain ArgMax over a flattened [outer, axis, inner] view.
 * Quantized per-tensor scales are positive, so comparing the stored signed or
 * unsigned byte values is exactly equivalent to comparing dequantized values;
 * do not introduce an F32 logits boundary merely to select an index. */
static void vx_pf_qargmax_store_i32(int32_t *output, size_t index,
                                    int32_t value) {
    unsigned char *bytes = (unsigned char *)output + index * sizeof(value);
    uint32_t bits = (uint32_t)value;
    bytes[0] = (unsigned char)(bits & 0xffu);
    bytes[1] = (unsigned char)((bits >> 8u) & 0xffu);
    bytes[2] = (unsigned char)((bits >> 16u) & 0xffu);
    bytes[3] = (unsigned char)((bits >> 24u) & 0xffu);
}

/* I8/U8 input, I32 output, first tie wins. All shape/range checks precede
 * every store so malformed direct native/WASM calls leave output untouched. */
WASM_EXPORT("qargmax_i8u8")
int qargmax_i8u8(const void *input, int32_t *output, uint32_t outer,
                 uint32_t axis_size, uint32_t inner, uint32_t input_dtype) {
    size_t input_elements = outer;
    size_t output_elements = outer;
    size_t output_bytes;
    if (!input || !output || !outer || !axis_size || !inner ||
        axis_size > (uint32_t)INT32_MAX ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_pf_mul_size(&input_elements, axis_size) ||
        !vx_pf_mul_size(&input_elements, inner) ||
        !vx_pf_mul_size(&output_elements, inner) ||
        input_elements > UINT32_MAX || output_elements > UINT32_MAX ||
        output_elements > (size_t)-1 / sizeof(*output)) return 0;
    output_bytes = output_elements * sizeof(*output);
    if (vx_w8a8_ranges_overlap(output, output_bytes, input, input_elements)) return 0;
    for (uint32_t outer_index = 0; outer_index < outer; outer_index++) {
        for (uint32_t inner_index = 0; inner_index < inner; inner_index++) {
            size_t input_index = ((size_t)outer_index * axis_size) * inner +
                inner_index;
            int32_t best_value = vx_w8a8_byte_value(input, input_dtype,
                                                        input_index);
            uint32_t best_index = 0;
            for (uint32_t axis_index = 1; axis_index < axis_size; axis_index++) {
                int32_t value = vx_w8a8_byte_value(input, input_dtype,
                    input_index + (size_t)axis_index * inner);
                /* Strictly greater preserves the earliest equal maximum. */
                if (value > best_value) {
                    best_value = value;
                    best_index = axis_index;
                }
            }
            vx_pf_qargmax_store_i32(output,
                (size_t)outer_index * inner + inner_index, (int32_t)best_index);
        }
    }
    return 1;
}

/* SafeTensors permits an I32 tensor to begin at an arbitrary byte offset.
 * Keep-mask values are therefore decoded bytewise instead of assuming native
 * I32 alignment.  Every supported portable target is little endian. */
static int32_t vx_pf_qmaskedmean_mask_i32_at(const int32_t *mask, size_t index) {
    const unsigned char *bytes = (const unsigned char *)mask +
        index * sizeof(int32_t);
    uint32_t bits = (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u);
    return (int32_t)bits;
}

/* The canonical contract makes the reduction domain I32, matching the GPU
 * shader.  Bound the largest possible centered byte sum before any reduction
 * starts so signed accumulation cannot overflow on malformed direct calls. */
static int vx_pf_qmaskedmean_sum_i32_valid(uint32_t dtype, int32_t zero_point,
                                           uint32_t sequence) {
    int64_t low;
    int64_t high;
    uint64_t magnitude;
    if (dtype == VX_DTYPE_I8) {
        low = -128 - (int64_t)zero_point;
        high = 127 - (int64_t)zero_point;
    } else if (dtype == VX_DTYPE_U8) {
        low = -(int64_t)zero_point;
        high = 255 - (int64_t)zero_point;
    } else {
        return 0;
    }
    magnitude = (uint64_t)(low < 0 ? -low : low);
    if ((uint64_t)(high < 0 ? -high : high) > magnitude)
        magnitude = (uint64_t)(high < 0 ? -high : high);
    return magnitude == 0 || (uint64_t)sequence <= (uint64_t)INT32_MAX / magnitude;
}

/* Canonical token-mask mean for quantized sequence graphs. Input stays
 * [B,S,D] physical I8/U8 bytes, mask is ordinary I32 [B,S], and output is
 * [B,D] physical bytes.  Accumulating centered raw values before the single
 * requantization avoids an F32 graph activation; only scalar local math is
 * used.  A batch row with no kept tokens denotes the real zero vector and
 * consequently writes the output zero point. */
WASM_EXPORT("qmaskedmean_i8u8")
int qmaskedmean_i8u8(const void *input, const int32_t *mask, void *output,
        uint32_t batch, uint32_t sequence, uint32_t width,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    size_t input_elements = batch;
    size_t mask_elements = batch;
    size_t output_elements = batch;
    size_t mask_bytes;
    float multiplier;
    int32_t output_minimum;
    int32_t output_maximum;
    if (!input || !mask || !output || !batch || !sequence || !width ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_pf_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_pf_qmaskedmean_sum_i32_valid(input_dtype, input_zero_point, sequence) ||
        !vx_pf_mul_size(&input_elements, sequence) ||
        !vx_pf_mul_size(&input_elements, width) ||
        !vx_pf_mul_size(&mask_elements, sequence) ||
        !vx_pf_mul_size(&output_elements, width) ||
        input_elements > UINT32_MAX || mask_elements > UINT32_MAX ||
        output_elements > UINT32_MAX ||
        mask_elements > (size_t)-1 / sizeof(int32_t)) return 0;
    multiplier = input_scale / output_scale;
    if (!vx_pf_finite_f32(multiplier) || multiplier <= 0.0f) return 0;
    mask_bytes = mask_elements * sizeof(int32_t);
    if (vx_w8a8_ranges_overlap(output, output_elements, input, input_elements) ||
        vx_w8a8_ranges_overlap(output, output_elements, mask, mask_bytes)) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        uint32_t keep_count = 0;
        const size_t mask_base = (size_t)batch_index * sequence;
        const size_t input_batch_base = mask_base * width;
        const size_t output_batch_base = (size_t)batch_index * width;
        for (uint32_t token = 0; token < sequence; token++) {
            if (vx_pf_qmaskedmean_mask_i32_at(mask, mask_base + token) != 0)
                keep_count++;
        }
        for (uint32_t dimension = 0; dimension < width; dimension++) {
            int32_t quantized = output_zero_point;
            if (keep_count) {
                int32_t centered_sum = 0;
                for (uint32_t token = 0; token < sequence; token++) {
                    if (vx_pf_qmaskedmean_mask_i32_at(mask, mask_base + token) != 0) {
                        const size_t input_index = input_batch_base +
                            (size_t)token * width + dimension;
                        centered_sum += vx_w8a8_byte_value(input, input_dtype,
                            input_index) - input_zero_point;
                    }
                }
                {
                    const float mean = (float)centered_sum /
                        (float)keep_count;
                    /* Keep the F32 quantization schedule identical to the
                     * authoritative WGSL: a separately rounded multiply
                     * precedes the zero-point addition.  `volatile` prevents
                     * an FMA contraction on native AVX/FMA builds. */
                    volatile float scaled = mean * multiplier;
                    const float transformed = scaled + (float)output_zero_point;
                    quantized = vx_w8a8_requantize(transformed,
                        output_minimum, output_maximum, output_zero_point);
                }
            }
            vx_w8a8_store_byte(output, output_dtype,
                                   output_batch_base + dimension, quantized);
        }
    }
    return 1;
}

/* Metadata-only typed bridge between quantized activation domains.  Unlike
 * QuantizeLinear, both sides already own immutable scale/zero-point metadata
 * and no mutable scale tensors participate in execution. */
WASM_EXPORT("requantize_linear_i8u8")
int requantize_linear_i8u8(const void *input, void *output, uint32_t elements,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    int32_t output_minimum, output_maximum;
    float multiplier;
    if (!input || !output || !elements || !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) || !vx_pf_finite_f32(input_scale) ||
        input_scale <= 0.0f || !vx_pf_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype)) return 0;
    multiplier = input_scale / output_scale;
    if (!vx_pf_finite_f32(multiplier) || multiplier <= 0.0f) return 0;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t index = 0; index < elements; index++) {
        const float centered = (float)(vx_w8a8_byte_value(input, input_dtype, index) -
            input_zero_point);
        const float scaled = centered * multiplier;
        const float transformed = scaled + (float)output_zero_point;
        const int32_t quantized = vx_w8a8_requantize(transformed,
            output_minimum, output_maximum, output_zero_point);
        vx_w8a8_store_byte(output, output_dtype, index, quantized);
    }
    return 1;
}

static int vx_pf_qconv_accumulator_i32_valid(const int32_t *bias,
        const int32_t *weight_zero_points, uint32_t output_channels,
        uint64_t terms, int32_t input_zero_point, uint32_t input_dtype,
        uint32_t weight_dtype) {
    uint64_t input_magnitude;
    int64_t input_low, input_high;
    uint64_t low_magnitude, high_magnitude;
    if (!weight_zero_points || !output_channels || !terms ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_byte_dtype(weight_dtype) ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype)) return 0;
    input_low =
        (int64_t)(input_dtype == VX_DTYPE_I8 ? -128 : 0) - input_zero_point;
    input_high =
        (int64_t)(input_dtype == VX_DTYPE_I8 ? 127 : 255) - input_zero_point;
    low_magnitude = (uint64_t)(input_low < 0 ? -input_low : input_low);
    high_magnitude = (uint64_t)(input_high < 0 ? -input_high : input_high);
    input_magnitude = low_magnitude > high_magnitude ? low_magnitude : high_magnitude;
    for (uint32_t output_channel = 0; output_channel < output_channels;
            output_channel++) {
        int64_t weight_low, weight_high;
        uint64_t weight_magnitude, accumulator_bound, bias_magnitude = 0;
        int32_t weight_zero_point = weight_zero_points[output_channel];
        if (!vx_w8a8_zero_point_valid(weight_zero_point, weight_dtype)) return 0;
        weight_low = (int64_t)(weight_dtype == VX_DTYPE_I8 ? -128 : 0) -
            weight_zero_point;
        weight_high = (int64_t)(weight_dtype == VX_DTYPE_I8 ? 127 : 255) -
            weight_zero_point;
        weight_magnitude = (uint64_t)(weight_low < 0 ? -weight_low : weight_low);
        high_magnitude = (uint64_t)(weight_high < 0 ? -weight_high : weight_high);
        if (high_magnitude > weight_magnitude) weight_magnitude = high_magnitude;
        if (input_magnitude && weight_magnitude &&
            terms > (uint64_t)INT32_MAX / input_magnitude / weight_magnitude) return 0;
        accumulator_bound = input_magnitude * weight_magnitude * terms;
        if (bias) {
            int64_t bias_value = bias[output_channel];
            bias_magnitude = (uint64_t)(bias_value < 0 ? -bias_value : bias_value);
        }
        if (accumulator_bound > (uint64_t)INT32_MAX ||
            bias_magnitude > (uint64_t)INT32_MAX - accumulator_bound) return 0;
    }
    return 1;
}

#ifdef __wasm__
static void vx_pf_qconv_copy_bytes(uint8_t *destination,
        const uint8_t *source, size_t bytes) {
    size_t index = 0u;
#if defined(__wasm_simd128__)
    for (; bytes - index >= 16u; index += 16u) {
        wasm_v128_store(destination + index, wasm_v128_load(source + index));
    }
#endif
    for (; index < bytes; index++) destination[index] = source[index];
}

static void vx_pf_qconv_fill_bytes(uint8_t *destination, uint8_t value,
        size_t bytes) {
    size_t index = 0u;
#if defined(__wasm_simd128__)
    const v128_t values = wasm_i8x16_splat((int8_t)value);
    for (; bytes - index >= 16u; index += 16u) {
        wasm_v128_store(destination + index, values);
    }
#endif
    for (; index < bytes; index++) destination[index] = value;
}

/* WASM-internal layout-only groups=1 bridge for the packed-GEMM convolution
 * route.  It is intentionally absent from the native public kernel ABI.
 * Padding stores the raw activation zero point, making every padded product
 * exactly zero after QLinear centering.  The shared accumulator proof keeps
 * its fail-closed domain identical to canonical QConv before scratch is
 * touched; arithmetic and requantization remain owned by packed QLinear. */
WASM_EXPORT("qconv2d_im2col_i8u8")
int qconv2d_im2col_i8u8(const void *input, void *columns,
        const int32_t *bias, const int32_t *weight_zero_points,
        uint32_t batch, uint32_t input_height, uint32_t input_width,
        uint32_t input_channels, uint32_t output_height,
        uint32_t output_width, uint32_t output_channels, uint32_t kernel_height,
        uint32_t kernel_width, uint32_t stride_y, uint32_t stride_x,
        uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
        uint32_t padding_left, uint32_t padding_bottom,
        uint32_t padding_right, int32_t input_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype) {
    const uint8_t *input_bytes = (const uint8_t *)input;
    uint8_t *column_bytes = (uint8_t *)columns;
    size_t input_elements = batch;
    size_t column_elements = batch;
    size_t patch_elements = kernel_height;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width;
    uint8_t padding_byte;
    if (!input || !columns || !batch || !input_height || !input_width ||
        !input_channels || !output_height || !output_width || !output_channels ||
        !kernel_height || !kernel_width || !stride_y || !stride_x ||
        !dilation_y || !dilation_x ||
        !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_pf_mul_size(&input_elements, input_height) ||
        !vx_pf_mul_size(&input_elements, input_width) ||
        !vx_pf_mul_size(&input_elements, input_channels) ||
        !vx_pf_mul_size(&patch_elements, kernel_width) ||
        !vx_pf_mul_size(&patch_elements, input_channels) ||
        !vx_pf_mul_size(&column_elements, output_height) ||
        !vx_pf_mul_size(&column_elements, output_width) ||
        !vx_pf_mul_size(&column_elements, patch_elements) ||
        vx_w8a8_ranges_overlap(input, input_elements, columns, column_elements)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    if (!vx_pf_qconv_accumulator_i32_valid(bias, weight_zero_points,
            output_channels, patch_elements, input_zero_point, input_dtype,
            weight_dtype)) return 0;
    padding_byte = input_dtype == VX_DTYPE_I8
        ? (uint8_t)(int8_t)input_zero_point :
        (uint8_t)input_zero_point;
    if (kernel_height == 3u && kernel_width == 3u &&
        dilation_y == 1u && dilation_x == 1u) {
        const size_t input_plane = (size_t)input_height * input_width;
        const size_t output_plane = (size_t)output_height * output_width;
        const size_t strip_bytes = (size_t)3u * input_channels;
        for (uint32_t sample = 0; sample < batch; sample++) {
            for (uint32_t output_y = 0; output_y < output_height; output_y++) {
                const int64_t input_y_origin =
                    (int64_t)output_y * stride_y - padding_top;
                for (uint32_t output_x = 0; output_x < output_width;
                        output_x++) {
                    const size_t location = (size_t)sample * output_plane +
                        (size_t)output_y * output_width + output_x;
                    uint8_t *row = column_bytes + location * patch_elements;
                    const int64_t input_x_origin =
                        (int64_t)output_x * stride_x - padding_left;
                    for (uint32_t kernel_y = 0; kernel_y < 3u; kernel_y++) {
                        const int64_t input_y = input_y_origin + kernel_y;
                        uint8_t *block = row + (size_t)kernel_y * strip_bytes;
                        if (input_y < 0 || input_y >= input_height) {
                            vx_pf_qconv_fill_bytes(
                                block, padding_byte, strip_bytes);
                        } else if (input_x_origin >= 0 &&
                                   input_x_origin + 2 < input_width) {
                            const size_t source =
                                ((size_t)sample * input_plane +
                                 (size_t)input_y * input_width +
                                 (size_t)input_x_origin) * input_channels;
                            vx_pf_qconv_copy_bytes(
                                block, input_bytes + source, strip_bytes);
                        } else {
                            for (uint32_t kernel_x = 0; kernel_x < 3u;
                                    kernel_x++) {
                                const int64_t input_x =
                                    input_x_origin + kernel_x;
                                uint8_t *channels = block +
                                    (size_t)kernel_x * input_channels;
                                if (input_x < 0 || input_x >= input_width) {
                                    vx_pf_qconv_fill_bytes(channels,
                                        padding_byte, input_channels);
                                } else {
                                    const size_t source =
                                        ((size_t)sample * input_plane +
                                         (size_t)input_y * input_width +
                                         (size_t)input_x) * input_channels;
                                    vx_pf_qconv_copy_bytes(channels,
                                        input_bytes + source, input_channels);
                                }
                            }
                        }
                    }
                }
            }
        }
        return 1;
    }
    for (uint32_t sample = 0; sample < batch; sample++) {
        for (uint32_t output_y = 0; output_y < output_height; output_y++) {
            for (uint32_t output_x = 0; output_x < output_width; output_x++) {
                size_t destination = (((size_t)sample * output_height + output_y) *
                    output_width + output_x) * patch_elements;
                for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                    uint64_t padded_y = (uint64_t)output_y * stride_y +
                        (uint64_t)kernel_y * dilation_y;
                    for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                        uint64_t padded_x = (uint64_t)output_x * stride_x +
                            (uint64_t)kernel_x * dilation_x;
                        const int valid = padded_y >= padding_top &&
                            padded_y - padding_top < input_height &&
                            padded_x >= padding_left &&
                            padded_x - padding_left < input_width;
                        if (valid) {
                            size_t source = (((size_t)sample * input_height +
                                (uint32_t)(padded_y - padding_top)) * input_width +
                                (uint32_t)(padded_x - padding_left)) * input_channels;
                            for (uint32_t channel = 0; channel < input_channels; channel++)
                                column_bytes[destination++] = input_bytes[source + channel];
                        } else {
                            for (uint32_t channel = 0; channel < input_channels; channel++)
                                column_bytes[destination++] = padding_byte;
                        }
                    }
                }
            }
        }
    }
    return 1;
}
#endif

/* Canonical W8A8 Conv2D: NHWC activations and [O,H,W,I/group] OHWI weights.
 * Weight scale/zero-point are per output channel, and an optional I32 bias is
 * in the accumulator domain.  The JavaScript caller validates tensor shapes;
 * this ABI repeats the arithmetic checks so direct WASM users cannot overflow
 * an I32 accumulation or address a wrapped layout. */
WASM_EXPORT("qconv2d_i8u8")
int qconv2d_i8u8(const void *input, const void *weight, const int32_t *bias,
        const float *weight_scales, const int32_t *weight_zero_points,
        void *output, uint32_t batch, uint32_t input_height,
        uint32_t input_width, uint32_t input_channels,
        uint32_t output_height, uint32_t output_width,
        uint32_t output_channels, uint32_t kernel_height,
        uint32_t kernel_width, uint32_t input_per_group,
        uint32_t stride_y, uint32_t stride_x, uint32_t dilation_y,
        uint32_t dilation_x, uint32_t padding_top, uint32_t padding_left,
        uint32_t padding_bottom, uint32_t padding_right, uint32_t groups,
        uint32_t relu, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    size_t input_elements = batch;
    size_t weight_elements = output_channels;
    size_t output_elements = batch;
    uint64_t padded_height, padded_width, effective_height, effective_width;
    uint64_t expected_height, expected_width, terms;
    uint32_t output_channels_per_group;
    int32_t output_minimum, output_maximum, relu6_upper = 0;
    if (!input || !weight || !weight_scales || !weight_zero_points || !output ||
        !batch || !input_height || !input_width || !input_channels ||
        !output_height || !output_width || !output_channels || !kernel_height ||
        !kernel_width || !input_per_group || !stride_y || !stride_x ||
        !dilation_y || !dilation_x || !groups || relu > 2u ||
        !vx_w8a8_byte_dtype(input_dtype) || !vx_w8a8_byte_dtype(weight_dtype) ||
        !vx_w8a8_byte_dtype(output_dtype) || !vx_pf_finite_f32(input_scale) ||
        input_scale <= 0.0f || !vx_pf_finite_f32(output_scale) ||
        output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        input_channels % groups || output_channels % groups ||
        (uint64_t)input_per_group * groups != input_channels) return 0;
    if (!vx_pf_mul_size(&input_elements, input_height) ||
        !vx_pf_mul_size(&input_elements, input_width) ||
        !vx_pf_mul_size(&input_elements, input_channels) ||
        !vx_pf_mul_size(&weight_elements, kernel_height) ||
        !vx_pf_mul_size(&weight_elements, kernel_width) ||
        !vx_pf_mul_size(&weight_elements, input_per_group) ||
        !vx_pf_mul_size(&output_elements, output_height) ||
        !vx_pf_mul_size(&output_elements, output_width) ||
        !vx_pf_mul_size(&output_elements, output_channels)) return 0;
    padded_height = (uint64_t)input_height + padding_top + padding_bottom;
    padded_width = (uint64_t)input_width + padding_left + padding_right;
    effective_height = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_width = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_height < effective_height || padded_width < effective_width) return 0;
    expected_height = (padded_height - effective_height) / stride_y + 1u;
    expected_width = (padded_width - effective_width) / stride_x + 1u;
    if (expected_height != output_height || expected_width != output_width) return 0;
    output_channels_per_group = output_channels / groups;
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    terms = (uint64_t)(weight_elements / output_channels);
    if (!vx_pf_qconv_accumulator_i32_valid(bias, weight_zero_points,
            output_channels, terms, input_zero_point, input_dtype,
            weight_dtype)) return 0;
    for (uint32_t output_channel = 0; output_channel < output_channels; output_channel++) {
        if (!vx_pf_finite_f32(weight_scales[output_channel]) ||
            weight_scales[output_channel] <= 0.0f) return 0;
    }
    if (relu >= 2u) {
        const float relu6_scaled = 6.0f / output_scale;
        const float relu6_transformed = relu6_scaled + (float)output_zero_point;
        relu6_upper = vx_w8a8_requantize(relu6_transformed,
            output_minimum, output_maximum, 0);
    }
    /* The absolute-value proof above bounds the bias plus every possible
     * product in the complete reduction.  It therefore also bounds every
     * accumulator prefix, so the hot K loop needs no redundant range branch. */
    for (uint32_t batch_index = 0; batch_index < batch; batch_index++) {
        for (uint32_t output_y = 0; output_y < output_height; output_y++) {
            for (uint32_t output_x = 0; output_x < output_width; output_x++) {
                for (uint32_t output_channel = 0; output_channel < output_channels;
                        output_channel++) {
                    uint32_t group = output_channel / output_channels_per_group;
                    size_t output_index = (((size_t)batch_index * output_height + output_y) *
                        output_width + output_x) * output_channels + output_channel;
                    size_t weight_channel_offset = (size_t)output_channel * kernel_height *
                        kernel_width * input_per_group;
                    int64_t accumulator = bias ? (int64_t)bias[output_channel] : 0;
                    const float product_scale = input_scale * weight_scales[output_channel];
                    const float multiplier = product_scale / output_scale;
                    for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                        uint64_t padded_y = (uint64_t)output_y * stride_y +
                            (uint64_t)kernel_y * dilation_y;
                        if (padded_y < padding_top || padded_y - padding_top >= input_height) continue;
                        for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                            uint64_t padded_x = (uint64_t)output_x * stride_x +
                                (uint64_t)kernel_x * dilation_x;
                            if (padded_x < padding_left || padded_x - padding_left >= input_width) continue;
                            for (uint32_t local_channel = 0; local_channel < input_per_group;
                                    local_channel++) {
                                uint32_t input_channel = group * input_per_group + local_channel;
                                size_t input_index = (((size_t)batch_index * input_height +
                                    (uint32_t)(padded_y - padding_top)) * input_width +
                                    (uint32_t)(padded_x - padding_left)) * input_channels + input_channel;
                                size_t weight_index = weight_channel_offset +
                                    ((size_t)kernel_y * kernel_width + kernel_x) * input_per_group +
                                    local_channel;
                                accumulator += (int64_t)(vx_w8a8_byte_value(input, input_dtype,
                                    input_index) - input_zero_point) *
                                    (int64_t)(vx_w8a8_byte_value(weight, weight_dtype,
                                    weight_index) - weight_zero_points[output_channel]);
                            }
                        }
                    }
                    {
                        const float scaled = (float)accumulator * multiplier;
                        const float transformed = scaled + (float)output_zero_point;
                        const int transformed_nan = transformed != transformed;
                        int32_t quantized = vx_w8a8_requantize(transformed,
                            output_minimum, output_maximum, output_zero_point);
                        if (!transformed_nan && relu) {
                            if (quantized < output_zero_point) quantized = output_zero_point;
                            if (relu >= 2u && quantized > relu6_upper) quantized = relu6_upper;
                        }
                        vx_w8a8_store_byte(output, output_dtype, output_index, quantized);
                    }
                }
            }
        }
    }
    return 1;
}

static int vx_pf_expand_validate(const uint32_t *input_shape,
        const uint32_t *output_shape, uint32_t input_rank,
        uint32_t output_rank, uint32_t output_elements, size_t *input_strides) {
    size_t output_product = 1;
    if (!input_shape || !output_shape || !input_strides || input_rank == 0 ||
        output_rank == 0 || input_rank > output_rank || output_rank > 8 ||
        !vx_pf_contiguous_strides(input_shape, input_rank, input_strides)) return 0;
    for (uint32_t axis = 0; axis < output_rank; axis++) {
        uint32_t input_dimension = axis < output_rank - input_rank ? 1u :
            input_shape[axis - (output_rank - input_rank)];
        uint32_t output_dimension = output_shape[axis];
        if (!output_dimension || (input_dimension != 1u && input_dimension != output_dimension) ||
            !vx_pf_mul_size(&output_product, output_dimension)) return 0;
    }
    return output_product == output_elements;
}

/* Return the largest contiguous suffix that Expand can copy without any
 * broadcast-index arithmetic. Attention masks and position/adapter constants
 * commonly widen only leading axes, leaving hundreds of trailing values
 * contiguous. Resolving the right-aligned index once per such block avoids a
 * rank-sized chain of integer divisions for every output element. */
static uint32_t vx_pf_expand_contiguous_suffix(
        const uint32_t *input_shape, const uint32_t *output_shape,
        uint32_t input_rank, uint32_t output_rank) {
    uint32_t elements = 1u;
    uint32_t offset = output_rank - input_rank;
    for (uint32_t reverse = output_rank; reverse-- > offset;) {
        uint32_t input_dimension = input_shape[reverse - offset];
        if (input_dimension != output_shape[reverse]) break;
        elements *= output_shape[reverse];
    }
    return elements;
}

static void vx_pf_expand_copy_contiguous(const uint8_t *source,
        uint8_t *destination, uint32_t elements, size_t element_size) {
#if defined(__wasm__)
    uint32_t index = 0u;
    /* The freestanding module's general memcpy is deliberately a volatile
     * byte loop so LLVM cannot turn its definition into an unresolved libc
     * call. Expand moves large aligned F32/I32 suffixes, so spelling out the
     * element-width loop here lets the SIMD build move them a vector at a time
     * without routing every four-byte element through that byte loop. */
    if (element_size == sizeof(uint32_t)) {
#if defined(__wasm_simd128__)
        for (; elements - index >= 4u; index += 4u) {
            wasm_v128_store(destination + (size_t)index * sizeof(uint32_t),
                wasm_v128_load(source + (size_t)index * sizeof(uint32_t)));
        }
#endif
        for (; index < elements; index++) {
            uint32_t value;
            /* Constant-size builtins lower to one unaligned i32 load/store in
             * wasm, while retaining memcpy's alias-safe bit-copy semantics for
             * the float entry point. */
            __builtin_memcpy(&value,
                source + (size_t)index * sizeof(value), sizeof(value));
            __builtin_memcpy(destination + (size_t)index * sizeof(value),
                &value, sizeof(value));
        }
        return;
    }
#if defined(__wasm_simd128__)
    for (; elements - index >= 16u; index += 16u) {
        wasm_v128_store(destination + index, wasm_v128_load(source + index));
    }
#endif
    for (; index < elements; index++) destination[index] = source[index];
#else
    memcpy(destination, source, (size_t)elements * element_size);
#endif
}

static int vx_pf_expand_copy_blocks(const void *input, void *output,
        const uint32_t *input_shape, const uint32_t *output_shape,
        uint32_t input_rank, uint32_t output_rank,
        uint32_t output_elements, size_t element_size,
        const size_t *input_strides) {
    const uint32_t block_elements = vx_pf_expand_contiguous_suffix(
        input_shape, output_shape, input_rank, output_rank);
    const uint32_t block_count = output_elements / block_elements;
    uint32_t suffix_rank = 0u;
    uint32_t suffix_elements = 1u;
    uint32_t prefix_rank;
    const uint8_t *source = (const uint8_t *)input;
    uint8_t *destination = (uint8_t *)output;
    size_t input_steps[8] = {0};
    uint32_t coordinates[8] = {0};
    size_t input_index = 0;
    const uint32_t offset = output_rank - input_rank;
    if (!element_size || (size_t)output_elements > SIZE_MAX / element_size)
        return 0;
    if (input == output) return block_count == 1u;
    while (suffix_rank < output_rank && suffix_elements < block_elements) {
        suffix_elements *= output_shape[output_rank - 1u - suffix_rank];
        suffix_rank++;
    }
    prefix_rank = output_rank - suffix_rank;
    for (uint32_t axis = 0; axis < prefix_rank; axis++) {
        if (axis >= offset && input_shape[axis - offset] != 1u)
            input_steps[axis] = input_strides[axis - offset];
    }
    for (uint32_t block = 0; block < block_count; block++) {
        const uint32_t output_index = block * block_elements;
        vx_pf_expand_copy_contiguous(source + input_index * element_size,
            destination + (size_t)output_index * element_size,
            block_elements, element_size);
        for (uint32_t reverse = prefix_rank; reverse-- > 0;) {
            coordinates[reverse]++;
            if (coordinates[reverse] < output_shape[reverse]) {
                input_index += input_steps[reverse];
                break;
            }
            coordinates[reverse] = 0u;
            input_index -= (size_t)(output_shape[reverse] - 1u) *
                input_steps[reverse];
        }
    }
    return 1;
}

WASM_EXPORT("expand_nd_f32")
int expand_nd_f32(const float *input, float *output, const uint32_t *input_shape,
        const uint32_t *output_shape, uint32_t input_rank, uint32_t output_rank,
        uint32_t output_elements) {
    size_t input_strides[8];
    if (!input || !output || !vx_pf_expand_validate(input_shape, output_shape,
        input_rank, output_rank, output_elements, input_strides)) return 0;
    return vx_pf_expand_copy_blocks(input, output, input_shape, output_shape,
        input_rank, output_rank, output_elements, sizeof(*output), input_strides);
}

WASM_EXPORT("expand_nd_u32")
int expand_nd_u32(const uint32_t *input, uint32_t *output,
        const uint32_t *input_shape, const uint32_t *output_shape,
        uint32_t input_rank, uint32_t output_rank, uint32_t output_elements) {
    size_t input_strides[8];
    if (!input || !output || !vx_pf_expand_validate(input_shape, output_shape,
        input_rank, output_rank, output_elements, input_strides)) return 0;
    return vx_pf_expand_copy_blocks(input, output, input_shape, output_shape,
        input_rank, output_rank, output_elements, sizeof(*output), input_strides);
}

WASM_EXPORT("expand_nd_i8u8")
int expand_nd_i8u8(const uint8_t *input, uint8_t *output,
        const uint32_t *input_shape, const uint32_t *output_shape,
        uint32_t input_rank, uint32_t output_rank, uint32_t output_elements) {
    size_t input_strides[8];
    if (!input || !output || input == output ||
        !vx_pf_expand_validate(input_shape, output_shape,
            input_rank, output_rank, output_elements, input_strides)) return 0;
    return vx_pf_expand_copy_blocks(input, output, input_shape, output_shape,
        input_rank, output_rank, output_elements, sizeof(*output), input_strides);
}

/* Shared transpose plan: validates the permutation and precomputes, per output
 * axis, the extent and the input-element step that advancing it costs.
 *
 * The obvious loop recovers each output coordinate with a `%` and a `/` per
 * rank per element -- eight integer divisions per byte at rank 4, on runtime
 * shapes the compiler cannot strength-reduce. Walking the output linearly makes
 * those coordinates an odometer instead: increment the innermost counter, and
 * on wrap subtract the axis span and carry. Every element then costs one add.
 */
/* Square tile for the 2-D path, in elements. 32x32 keeps both the strided reads
 * and the contiguous writes of a tile inside L1 for every element width here. */
#define VX_PF_TRANSPOSE_TILE 32

typedef struct {
    size_t step[8];      /* input elements advanced by +1 on this output axis */
    size_t span[8];      /* step * (extent - 1): undone when the axis wraps */
    uint32_t extent[8];
    uint32_t counter[8];
    size_t run;          /* contiguous elements copied per block */
    size_t blocks;       /* number of runs covering the output */
    int carry_axis;      /* first axis the odometer carries from, -1 if none */
    /* Set when the whole transpose collapses to a 2-D swap: the output is
     * [rows][cols] and the input element for (r, c) sits at
     * r*row_step + c*col_step. Worth separating because that case is a strided
     * gather the odometer cannot make cache-friendly -- NHWC<->NCHW is exactly
     * this, and it is where the encoder spends its transpose time. */
    int tiled;
    size_t rows, cols, row_step, col_step;
} VxTransposePlan;

static int vx_pf_transpose_plan(const uint32_t *input_shape,
        const uint32_t *permutation, uint32_t rank, uint32_t elements,
        VxTransposePlan *plan) {
    size_t input_strides[8], product = 1;
    uint32_t seen = 0;
    if (!input_shape || !permutation || rank == 0 || rank > 8 ||
        !vx_pf_contiguous_strides(input_shape, rank, input_strides)) return 0;
    for (uint32_t axis = 0; axis < rank; axis++) {
        uint32_t source_axis = permutation[axis];
        if (source_axis >= rank || (seen & (1u << source_axis)) ||
            !vx_pf_mul_size(&product, input_shape[source_axis])) return 0;
        seen |= 1u << source_axis;
    }
    if (product != elements) return 0;
    for (uint32_t axis = 0; axis < rank; axis++) {
        uint32_t source_axis = permutation[axis];
        plan->extent[axis] = input_shape[source_axis];
        plan->step[axis] = input_strides[source_axis];
        plan->span[axis] = (size_t)(plan->extent[axis] - 1) * plan->step[axis];
        plan->counter[axis] = 0;
    }
    /* When the innermost output axis is also the contiguous input axis, its
     * whole run is contiguous on both sides and copies as one block. */
    plan->run = plan->step[rank - 1] == 1 ? plan->extent[rank - 1] : 1;
    plan->blocks = plan->run ? elements / plan->run : 0;
    plan->carry_axis = (int)rank - 1 - (plan->run > 1 ? 1 : 0);

    /* Merge output axes that are already adjacent in the input, then drop the
     * unit axes. Rank-4 NHWC<->NCHW collapses to two axes this way. */
    size_t merged_extent[8], merged_step[8];
    uint32_t merged = 0;
    for (uint32_t axis = 0; axis < rank; axis++) {
        size_t extent = plan->extent[axis], step = plan->step[axis];
        if (merged && merged_step[merged - 1] == step * extent) {
            merged_extent[merged - 1] *= extent;
            merged_step[merged - 1] = step;
        } else {
            merged_extent[merged] = extent;
            merged_step[merged] = step;
            merged++;
        }
    }
    uint32_t significant = 0;
    for (uint32_t axis = 0; axis < merged; axis++) {
        if (merged_extent[axis] == 1) continue;
        merged_extent[significant] = merged_extent[axis];
        merged_step[significant] = merged_step[axis];
        significant++;
    }
    /* Two strided axes is the transpose proper; one axis (or none) is already a
     * copy or a single strided walk the odometer handles fine. */
    plan->tiled = significant == 2 && merged_step[1] != 1;
    if (plan->tiled) {
        plan->rows = merged_extent[0];
        plan->cols = merged_extent[1];
        plan->row_step = merged_step[0];
        plan->col_step = merged_step[1];
    }
    return elements == 0 || plan->run != 0;
}

/* Tile bounds shared by the three element widths. */
#define VX_PF_TRANSPOSE_TILED(TYPE)                                            \
    for (size_t row_base = 0; row_base < plan.rows; row_base += VX_PF_TRANSPOSE_TILE) { \
        size_t row_end = row_base + VX_PF_TRANSPOSE_TILE;                      \
        if (row_end > plan.rows) row_end = plan.rows;                          \
        for (size_t col_base = 0; col_base < plan.cols; col_base += VX_PF_TRANSPOSE_TILE) { \
            size_t col_end = col_base + VX_PF_TRANSPOSE_TILE;                  \
            if (col_end > plan.cols) col_end = plan.cols;                      \
            for (size_t row = row_base; row < row_end; row++) {                \
                const TYPE *source = input + row * plan.row_step;              \
                TYPE *destination = output + row * plan.cols;                  \
                for (size_t col = col_base; col < col_end; col++)              \
                    destination[col] = source[col * plan.col_step];            \
            }                                                                  \
        }                                                                      \
    }

/* Advance the odometer past one emitted run. Returns the new input index. */
static size_t vx_pf_transpose_advance(VxTransposePlan *plan, size_t input_index) {
    for (int axis = plan->carry_axis; axis >= 0; axis--) {
        if (++plan->counter[axis] < plan->extent[axis]) return input_index + plan->step[axis];
        plan->counter[axis] = 0;
        input_index -= plan->span[axis];
    }
    return input_index;
}

WASM_EXPORT("transpose_nd_f32")
int transpose_nd_f32(const float *input, float *output, const uint32_t *input_shape,
        const uint32_t *permutation, uint32_t rank, uint32_t elements) {
    VxTransposePlan plan;
    size_t input_index = 0, written = 0;
    if (!input || !output ||
        !vx_pf_transpose_plan(input_shape, permutation, rank, elements, &plan)) return 0;
    if (plan.tiled) {
        VX_PF_TRANSPOSE_TILED(float)
        return 1;
    }
    for (size_t block = 0; block < plan.blocks; block++) {
        for (size_t offset = 0; offset < plan.run; offset++)
            output[written + offset] = input[input_index + offset];
        written += plan.run;
        input_index = vx_pf_transpose_advance(&plan, input_index);
    }
    return 1;
}

WASM_EXPORT("transpose_nd_u32")
int transpose_nd_u32(const uint32_t *input, uint32_t *output,
        const uint32_t *input_shape, const uint32_t *permutation,
        uint32_t rank, uint32_t elements) {
    VxTransposePlan plan;
    size_t input_index = 0, written = 0;
    if (!input || !output ||
        !vx_pf_transpose_plan(input_shape, permutation, rank, elements, &plan)) return 0;
    if (plan.tiled) {
        VX_PF_TRANSPOSE_TILED(uint32_t)
        return 1;
    }
    for (size_t block = 0; block < plan.blocks; block++) {
        for (size_t offset = 0; offset < plan.run; offset++)
            output[written + offset] = input[input_index + offset];
        written += plan.run;
        input_index = vx_pf_transpose_advance(&plan, input_index);
    }
    return 1;
}

WASM_EXPORT("transpose_nd_i8u8")
int transpose_nd_i8u8(const uint8_t *input, uint8_t *output,
        const uint32_t *input_shape, const uint32_t *permutation,
        uint32_t rank, uint32_t elements, uint32_t dtype) {
    VxTransposePlan plan;
    size_t input_index = 0, written = 0;
    if (!input || !output || input == output ||
        (dtype != VX_DTYPE_I8 && dtype != VX_DTYPE_U8) ||
        !vx_pf_transpose_plan(input_shape, permutation, rank, elements, &plan)) return 0;
    if (plan.tiled) {
        VX_PF_TRANSPOSE_TILED(uint8_t)
        return 1;
    }
    for (size_t block = 0; block < plan.blocks; block++) {
        for (size_t offset = 0; offset < plan.run; offset++)
            output[written + offset] = input[input_index + offset];
        written += plan.run;
        input_index = vx_pf_transpose_advance(&plan, input_index);
    }
    return 1;
}

static int vx_pf_slice_dimensions_valid(uint32_t outer, uint32_t input_axis,
        uint32_t output_axis, uint32_t inner, uint32_t axis_offset,
        size_t *input_block, size_t *output_block) {
    size_t input_count = 1, output_count = 1;
    if (!outer || !input_axis || !output_axis || !inner || axis_offset > input_axis ||
        output_axis > input_axis - axis_offset) return 0;
    if (!vx_pf_mul_size(&input_count, input_axis) || !vx_pf_mul_size(&input_count, inner) ||
        !vx_pf_mul_size(&input_count, outer) || !vx_pf_mul_size(&output_count, output_axis) ||
        !vx_pf_mul_size(&output_count, inner) || !vx_pf_mul_size(&output_count, outer)) return 0;
    if (input_block) *input_block = (size_t)input_axis * inner;
    if (output_block) *output_block = (size_t)output_axis * inner;
    return 1;
}

static int vx_pf_concat_dimensions_valid(uint32_t outer, uint32_t input_axis,
        uint32_t output_axis, uint32_t inner, uint32_t axis_offset,
        size_t *input_block) {
    size_t source_count = 1, destination_count = 1;
    if (!outer || !input_axis || !output_axis || !inner || axis_offset > output_axis ||
        input_axis > output_axis - axis_offset) return 0;
    if (!vx_pf_mul_size(&source_count, input_axis) || !vx_pf_mul_size(&source_count, inner) ||
        !vx_pf_mul_size(&source_count, outer) || !vx_pf_mul_size(&destination_count, output_axis) ||
        !vx_pf_mul_size(&destination_count, inner) || !vx_pf_mul_size(&destination_count, outer)) return 0;
    if (input_block) *input_block = (size_t)input_axis * inner;
    return 1;
}

/* Copy one input into a position in an N-way concat. apply_sigmoid is kept
 * here because the CPU portable operation supports the same fused flag. */
WASM_EXPORT("concat_slice_f32")
int concat_slice_f32(const float *input, float *output, uint32_t outer,
        uint32_t input_axis, uint32_t output_axis, uint32_t inner,
        uint32_t axis_offset, uint32_t apply_sigmoid) {
    size_t input_block;
    if (!input || !output || apply_sigmoid > 1u ||
        !vx_pf_concat_dimensions_valid(outer, input_axis, output_axis, inner,
            axis_offset, &input_block)) return 0;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = (size_t)group * input_block;
        size_t destination = ((size_t)group * output_axis + axis_offset) * inner;
        for (size_t index = 0; index < input_block; index++) {
            float value = input[source + index];
            output[destination + index] = apply_sigmoid ?
                1.0f / (1.0f + expf(-value)) : value;
        }
    }
    return 1;
}

WASM_EXPORT("concat_slice_u32")
int concat_slice_u32(const uint32_t *input, uint32_t *output, uint32_t outer,
        uint32_t input_axis, uint32_t output_axis, uint32_t inner,
        uint32_t axis_offset) {
    size_t input_block;
    if (!input || !output ||
        !vx_pf_concat_dimensions_valid(outer, input_axis, output_axis, inner,
            axis_offset, &input_block)) return 0;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = (size_t)group * input_block;
        size_t destination = ((size_t)group * output_axis + axis_offset) * inner;
        for (size_t index = 0; index < input_block; index++) {
            output[destination + index] = input[source + index];
        }
    }
    return 1;
}

/* These byte kernels deliberately do not dequantize or requantize.  Their
 * callers must prove that every edge has the same immutable quantization
 * descriptor before moving I8/U8 storage. */
WASM_EXPORT("copy_i8u8")
int copy_i8u8(const uint8_t *input, uint8_t *output, uint32_t elements,
        uint32_t dtype) {
    if (!input || !output || !elements || !vx_w8a8_byte_dtype(dtype)) return 0;
    if (input == output) return 1;
    for (uint32_t index = 0; index < elements; index++) {
        output[index] = input[index];
    }
    return 1;
}

/* Raw byte concat is valid only when JavaScript has already verified matching
 * quantization descriptors.  Unlike concat_slice_f32(), it has no fused
 * transformation. */
WASM_EXPORT("concat_slice_i8u8")
int concat_slice_i8u8(const uint8_t *input, uint8_t *output, uint32_t outer,
        uint32_t input_axis, uint32_t output_axis, uint32_t inner,
        uint32_t axis_offset, uint32_t dtype) {
    size_t input_block;
    if (!input || !output || !vx_w8a8_byte_dtype(dtype) ||
        !vx_pf_concat_dimensions_valid(outer, input_axis, output_axis, inner,
            axis_offset, &input_block)) return 0;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = (size_t)group * input_block;
        size_t destination = ((size_t)group * output_axis + axis_offset) * inner;
        for (size_t index = 0; index < input_block; index++) {
            output[destination + index] = input[source + index];
        }
    }
    return 1;
}

/* Copy a single equal-sized Split output. input_axis is the full input axis;
 * output_axis is the selected slice extent. */
WASM_EXPORT("split_slice_f32")
int split_slice_f32(const float *input, float *output, uint32_t outer,
        uint32_t input_axis, uint32_t output_axis, uint32_t inner,
        uint32_t axis_offset) {
    size_t output_block;
    if (!input || !output || !vx_pf_slice_dimensions_valid(outer, input_axis,
        output_axis, inner, axis_offset, NULL, &output_block)) return 0;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = ((size_t)group * input_axis + axis_offset) * inner;
        size_t destination = (size_t)group * output_block;
        for (size_t index = 0; index < output_block; index++) {
            output[destination + index] = input[source + index];
        }
    }
    return 1;
}

WASM_EXPORT("split_slice_u32")
int split_slice_u32(const uint32_t *input, uint32_t *output, uint32_t outer,
        uint32_t input_axis, uint32_t output_axis, uint32_t inner,
        uint32_t axis_offset) {
    size_t output_block;
    if (!input || !output || !vx_pf_slice_dimensions_valid(outer, input_axis,
        output_axis, inner, axis_offset, NULL, &output_block)) return 0;
    for (uint32_t group = 0; group < outer; group++) {
        size_t source = ((size_t)group * input_axis + axis_offset) * inner;
        size_t destination = (size_t)group * output_block;
        for (size_t index = 0; index < output_block; index++) {
            output[destination + index] = input[source + index];
        }
    }
    return 1;
}

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#define VX_GROUPNORM_X86_AVX2 1
#define VX_GROUPNORM_TARGET_AVX2 __attribute__((target("avx2")))

/* Sum a contiguous run into a pair of double accumulators.
 *
 * GroupNorm's reductions accumulate in double, so a vector lane is four wide
 * rather than eight and each float load feeds two converts.  That still beats
 * the scalar loop, and keeping the accumulator type means the group statistics
 * do not change precision — only the order in which equal-magnitude terms are
 * added, which for double accumulators over O(1) activations moves the result by
 * far less than the F32 output can represent. */
static inline VX_GROUPNORM_TARGET_AVX2 void vx_groupnorm_accumulate(
        const float *values, uint32_t count, __m256d *low, __m256d *high) {
    uint32_t index = 0;
    for (; index + 8u <= count; index += 8u) {
        const __m256 loaded = _mm256_loadu_ps(values + index);
        *low = _mm256_add_pd(*low,
            _mm256_cvtps_pd(_mm256_castps256_ps128(loaded)));
        *high = _mm256_add_pd(*high,
            _mm256_cvtps_pd(_mm256_extractf128_ps(loaded, 1)));
    }
    for (; index + 4u <= count; index += 4u)
        *low = _mm256_add_pd(*low, _mm256_cvtps_pd(_mm_loadu_ps(values + index)));
    {
        double tail = 0.0;
        for (; index < count; index++) tail += (double)values[index];
        if (tail != 0.0)
            *low = _mm256_add_pd(*low, _mm256_set_pd(0.0, 0.0, 0.0, tail));
    }
}

static inline VX_GROUPNORM_TARGET_AVX2 double vx_groupnorm_reduce(
        __m256d low, __m256d high) {
    double lanes[4];
    _mm256_storeu_pd(lanes, _mm256_add_pd(low, high));
    return (lanes[0] + lanes[1]) + (lanes[2] + lanes[3]);
}
#else
#define VX_GROUPNORM_X86_AVX2 0
#define VX_GROUPNORM_TARGET_AVX2
#endif

#if VX_GROUPNORM_X86_AVX2
/* Mirrors the portable body's three passes over each group: mean, centered
 * variance, then the affine normalization.  Fusing the first two into a
 * sum/sum-of-squares pair would halve the memory traffic but subtracts two
 * nearly equal quantities, so the passes stay separate. */
static VX_GROUPNORM_TARGET_AVX2 void vx_groupnorm_group_avx2(
        const float *input, const float *weight, const float *bias,
        float *output, size_t sample_offset, uint32_t first_channel,
        uint32_t channels, uint32_t channels_per_group, size_t spatial,
        size_t values, double epsilon) {
    __m256d mean_low = _mm256_setzero_pd(), mean_high = _mm256_setzero_pd();
    __m256d variance_low = _mm256_setzero_pd();
    __m256d variance_high = _mm256_setzero_pd();
    double mean, inverse;
    size_t point;
    for (point = 0; point < spatial; point++)
        vx_groupnorm_accumulate(
            input + sample_offset + point * channels + first_channel,
            channels_per_group, &mean_low, &mean_high);
    mean = vx_groupnorm_reduce(mean_low, mean_high) / (double)values;
    {
        const __m256d means = _mm256_set1_pd(mean);
        for (point = 0; point < spatial; point++) {
            const float *row = input + sample_offset + point * channels +
                first_channel;
            uint32_t local = 0;
            for (; local + 4u <= channels_per_group; local += 4u) {
                const __m256d centered = _mm256_sub_pd(
                    _mm256_cvtps_pd(_mm_loadu_ps(row + local)), means);
                variance_low = _mm256_add_pd(variance_low,
                    _mm256_mul_pd(centered, centered));
            }
            for (; local < channels_per_group; local++) {
                const double centered = (double)row[local] - mean;
                variance_high = _mm256_add_pd(variance_high,
                    _mm256_set_pd(0.0, 0.0, 0.0, centered * centered));
            }
        }
    }
    inverse = 1.0 / __builtin_sqrt(
        vx_groupnorm_reduce(variance_low, variance_high) / (double)values +
        epsilon);
    /* The affine pass is the only one that is not a reduction, so it runs eight
     * lanes wide in F32.  mean and inverse are already resolved scalars, and the
     * portable body rounds this product to F32 on store regardless. */
    {
        const __m256 means = _mm256_set1_ps((float)mean);
        const __m256 inverses = _mm256_set1_ps((float)inverse);
        for (point = 0; point < spatial; point++) {
            const size_t offset = sample_offset + point * channels +
                first_channel;
            const float *row = input + offset;
            float *destination = output + offset;
            uint32_t local = 0;
            for (; local + 8u <= channels_per_group; local += 8u) {
                const __m256 scaled = _mm256_mul_ps(
                    _mm256_sub_ps(_mm256_loadu_ps(row + local), means), inverses);
                const __m256 gain = _mm256_loadu_ps(weight + first_channel + local);
                const __m256 offsets = bias
                    ? _mm256_loadu_ps(bias + first_channel + local)
                    : _mm256_setzero_ps();
                _mm256_storeu_ps(destination + local,
                    _mm256_add_ps(_mm256_mul_ps(scaled, gain), offsets));
            }
            for (; local < channels_per_group; local++) {
                const uint32_t channel = first_channel + local;
                destination[local] = (float)(((double)row[local] - mean) *
                    inverse * weight[channel] + (bias ? bias[channel] : 0.0f));
            }
        }
    }
}
#endif

/* One accumulator pair per group lives on the stack during the streaming pass;
 * beyond this the original per-group loop still runs (correct, just slower). */
#define VX_GROUPNORM_MAX_STREAMED_GROUPS 512
/* A group narrower than one AVX2 vector never enters the vector body. */
#define VX_GROUPNORM_MIN_VECTOR_CHANNELS 8

#if VX_GROUPNORM_X86_AVX2
/* Keep a bounded NHWC row tile resident while vectorizing four independent
 * narrow groups in the F64 lanes.  Sixteen common image-encoder rows fit in
 * L1; wider legal tensors keep the same bounded, allocation-free algorithm
 * and merely lose some locality. */
#define VX_GROUPNORM_AVX2_SPATIAL_TILE 16u

static VX_GROUPNORM_TARGET_AVX2 void vx_groupnorm_streamed_avx2(
        const float *input, const float *weight, const float *bias,
        float *output, uint32_t batch, uint32_t channels, uint32_t groups,
        uint32_t channels_per_group, size_t spatial, size_t values,
        size_t sample_stride, double epsilon) {
    double sums[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    double squares[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    double means[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    double inverses[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    for (uint32_t sample = 0; sample < batch; sample++) {
        const size_t sample_offset = (size_t)sample * sample_stride;
        size_t tile;
        uint32_t group;
        for (group = 0; group < groups; group++) sums[group] = 0.0;
        for (tile = 0; tile < spatial; tile += VX_GROUPNORM_AVX2_SPATIAL_TILE) {
            size_t limit = tile + VX_GROUPNORM_AVX2_SPATIAL_TILE;
            if (limit > spatial) limit = spatial;
            for (group = 0; group + 3u < groups; group += 4u) {
                __m256d total = _mm256_loadu_pd(sums + group);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span0 = input + sample_offset +
                        point * channels + first;
                    const float *span1 = span0 + channels_per_group;
                    const float *span2 = span1 + channels_per_group;
                    const float *span3 = span2 + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++)
                        total = _mm256_add_pd(total, _mm256_set_pd(
                            (double)span3[local], (double)span2[local],
                            (double)span1[local], (double)span0[local]));
                }
                _mm256_storeu_pd(sums + group, total);
            }
            for (; group < groups; group++) {
                double total = sums[group];
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span = input + sample_offset +
                        point * channels + first;
                    for (uint32_t local = 0; local < channels_per_group; local++)
                        total += (double)span[local];
                }
                sums[group] = total;
            }
        }
        for (group = 0; group < groups; group++) {
            means[group] = sums[group] / (double)values;
            squares[group] = 0.0;
        }
        for (tile = 0; tile < spatial; tile += VX_GROUPNORM_AVX2_SPATIAL_TILE) {
            size_t limit = tile + VX_GROUPNORM_AVX2_SPATIAL_TILE;
            if (limit > spatial) limit = spatial;
            for (group = 0; group + 3u < groups; group += 4u) {
                __m256d total = _mm256_loadu_pd(squares + group);
                const __m256d mean = _mm256_loadu_pd(means + group);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span0 = input + sample_offset +
                        point * channels + first;
                    const float *span1 = span0 + channels_per_group;
                    const float *span2 = span1 + channels_per_group;
                    const float *span3 = span2 + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const __m256d centered = _mm256_sub_pd(_mm256_set_pd(
                            (double)span3[local], (double)span2[local],
                            (double)span1[local], (double)span0[local]), mean);
                        total = _mm256_add_pd(total,
                            _mm256_mul_pd(centered, centered));
                    }
                }
                _mm256_storeu_pd(squares + group, total);
            }
            for (; group < groups; group++) {
                double total = squares[group];
                const double mean = means[group];
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span = input + sample_offset +
                        point * channels + first;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const double centered = (double)span[local] - mean;
                        total += centered * centered;
                    }
                }
                squares[group] = total;
            }
        }
        for (group = 0; group < groups; group++)
            inverses[group] = 1.0 /
                __builtin_sqrt(squares[group] / (double)values + epsilon);

        /* Four adjacent groups form one contiguous channel bundle.  Build each
         * statistics vector once per tile/channel lane, then apply it to the
         * resident rows without changing the scalar operation order. */
        for (tile = 0; tile < spatial; tile += VX_GROUPNORM_AVX2_SPATIAL_TILE) {
            size_t limit = tile + VX_GROUPNORM_AVX2_SPATIAL_TILE;
            if (limit > spatial) limit = spatial;
            for (group = 0; group + 3u < groups; group += 4u) {
                const uint32_t first = group * channels_per_group;
                const uint32_t bundle = 4u * channels_per_group;
                uint32_t local = 0;
                for (; local + 4u <= bundle; local += 4u) {
                    const uint32_t c0 = first + local;
                    const uint32_t g0 = group + local / channels_per_group;
                    const uint32_t g1 = group + (local + 1u) / channels_per_group;
                    const uint32_t g2 = group + (local + 2u) / channels_per_group;
                    const uint32_t g3 = group + (local + 3u) / channels_per_group;
                    const __m256d mean = _mm256_set_pd(
                        means[g3], means[g2], means[g1], means[g0]);
                    const __m256d inverse = _mm256_set_pd(
                        inverses[g3], inverses[g2], inverses[g1], inverses[g0]);
                    const __m256d gain = _mm256_cvtps_pd(
                        _mm_loadu_ps(weight + c0));
                    const __m256d offset = bias
                        ? _mm256_cvtps_pd(_mm_loadu_ps(bias + c0))
                        : _mm256_setzero_pd();
                    for (size_t point = tile; point < limit; point++) {
                        const size_t row = sample_offset + point * channels + c0;
                        __m256d normalized = _mm256_sub_pd(
                            _mm256_cvtps_pd(_mm_loadu_ps(input + row)), mean);
                        normalized = _mm256_mul_pd(normalized, inverse);
                        normalized = _mm256_mul_pd(normalized, gain);
                        normalized = _mm256_add_pd(normalized, offset);
                        _mm_storeu_ps(output + row,
                            _mm256_cvtpd_ps(normalized));
                    }
                }
                for (; local < bundle; local++) {
                    const uint32_t channel = first + local;
                    const uint32_t lane_group =
                        group + local / channels_per_group;
                    for (size_t point = tile; point < limit; point++) {
                        const size_t row = sample_offset + point * channels + channel;
                        output[row] = (float)(((double)input[row] -
                            means[lane_group]) * inverses[lane_group] *
                            weight[channel] + (bias ? bias[channel] : 0.0f));
                    }
                }
            }
            for (; group < groups; group++) {
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const size_t row = sample_offset + point * channels + first;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const uint32_t channel = first + local;
                        output[row + local] = (float)(((double)input[row + local] -
                            means[group]) * inverses[group] * weight[channel] +
                            (bias ? bias[channel] : 0.0f));
                    }
                }
            }
        }
    }
}
#endif

#if defined(__wasm__) && defined(__wasm_simd128__) && \
    !defined(VOLVOXAI_DISABLE_GROUPNORM_WASM_SIMD)
#define VX_GROUPNORM_WASM_SIMD 1
#else
#define VX_GROUPNORM_WASM_SIMD 0
#endif

#if VX_GROUPNORM_WASM_SIMD
/* Keep one small NHWC row tile resident while visiting independent groups.
 * Sixteen 320-channel F32 rows occupy 20 KiB, leaving room in a conservative
 * 32 KiB L1 for statistics and affine vectors. Wider tensors still retain
 * exact behavior and merely lose some locality. */
#ifndef VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE
#define VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE 16u
#endif
#if VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE == 0
#error "VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE must be positive"
#endif

#if defined(VOLVOXAI_GROUPNORM_WASM_SIMD_TESTING)
static uint32_t vx_groupnorm_wasm_simd_call_count = 0;

WASM_EXPORT("groupnorm_wasm_simd_calls")
uint32_t groupnorm_wasm_simd_calls(void) {
    return vx_groupnorm_wasm_simd_call_count;
}

WASM_EXPORT("reset_groupnorm_wasm_simd_calls")
void reset_groupnorm_wasm_simd_calls(void) {
    vx_groupnorm_wasm_simd_call_count = 0;
}
#endif

/* Pair independent groups in the two F64 lanes.  Each lane observes points
 * and channels in exactly the scalar order, so this is not a reassociation of
 * either reduction.  Tiling amortizes the load/store of the accumulator pair
 * while keeping the strided group spans in L1. */
static void vx_groupnorm_streamed_wasm_simd(
        const float *input, const float *weight, const float *bias,
        float *output, uint32_t batch, uint32_t channels, uint32_t groups,
        uint32_t channels_per_group, size_t spatial, size_t values,
        size_t sample_stride, double epsilon) {
    double sums[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    double squares[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    double means[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    double inverses[VX_GROUPNORM_MAX_STREAMED_GROUPS];
    for (uint32_t sample = 0; sample < batch; sample++) {
        const size_t sample_offset = (size_t)sample * sample_stride;
        uint32_t group;
        size_t tile;
        for (group = 0; group < groups; group++) sums[group] = 0.0;
        for (tile = 0; tile < spatial;) {
            const size_t remaining = spatial - tile;
            const size_t limit = remaining < VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE
                ? spatial : tile + VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE;
            for (group = 0; group + 7u < groups; group += 8u) {
                v128_t total01 = wasm_v128_load(sums + group);
                v128_t total23 = wasm_v128_load(sums + group + 2u);
                v128_t total45 = wasm_v128_load(sums + group + 4u);
                v128_t total67 = wasm_v128_load(sums + group + 6u);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span0 = input + sample_offset +
                        point * channels + first;
                    const float *span1 = span0 + channels_per_group;
                    const float *span2 = span1 + channels_per_group;
                    const float *span3 = span2 + channels_per_group;
                    const float *span4 = span3 + channels_per_group;
                    const float *span5 = span4 + channels_per_group;
                    const float *span6 = span5 + channels_per_group;
                    const float *span7 = span6 + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        total01 = wasm_f64x2_add(total01, wasm_f64x2_make(
                            (double)span0[local], (double)span1[local]));
                        total23 = wasm_f64x2_add(total23, wasm_f64x2_make(
                            (double)span2[local], (double)span3[local]));
                        total45 = wasm_f64x2_add(total45, wasm_f64x2_make(
                            (double)span4[local], (double)span5[local]));
                        total67 = wasm_f64x2_add(total67, wasm_f64x2_make(
                            (double)span6[local], (double)span7[local]));
                    }
                }
                wasm_v128_store(sums + group, total01);
                wasm_v128_store(sums + group + 2u, total23);
                wasm_v128_store(sums + group + 4u, total45);
                wasm_v128_store(sums + group + 6u, total67);
            }
            for (; group + 3u < groups; group += 4u) {
                v128_t total01 = wasm_v128_load(sums + group);
                v128_t total23 = wasm_v128_load(sums + group + 2u);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span0 = input + sample_offset +
                        point * channels + first;
                    const float *span1 = span0 + channels_per_group;
                    const float *span2 = span1 + channels_per_group;
                    const float *span3 = span2 + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        total01 = wasm_f64x2_add(total01, wasm_f64x2_make(
                            (double)span0[local], (double)span1[local]));
                        total23 = wasm_f64x2_add(total23, wasm_f64x2_make(
                            (double)span2[local], (double)span3[local]));
                    }
                }
                wasm_v128_store(sums + group, total01);
                wasm_v128_store(sums + group + 2u, total23);
            }
            for (; group + 1u < groups; group += 2u) {
                v128_t total = wasm_v128_load(sums + group);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *low = input + sample_offset +
                        point * channels + first;
                    const float *high = low + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++)
                        total = wasm_f64x2_add(total, wasm_f64x2_make(
                            (double)low[local], (double)high[local]));
                }
                wasm_v128_store(sums + group, total);
            }
            if (group < groups) {
                double total = sums[group];
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span = input + sample_offset +
                        point * channels + first;
                    for (uint32_t local = 0; local < channels_per_group; local++)
                        total += span[local];
                }
                sums[group] = total;
            }
            tile = limit;
        }
        for (group = 0; group < groups; group++) {
            means[group] = sums[group] / (double)values;
            squares[group] = 0.0;
        }
        for (tile = 0; tile < spatial;) {
            const size_t remaining = spatial - tile;
            const size_t limit = remaining < VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE
                ? spatial : tile + VOLVOXAI_GROUPNORM_WASM_SPATIAL_TILE;
            for (group = 0; group + 7u < groups; group += 8u) {
                v128_t total01 = wasm_v128_load(squares + group);
                v128_t total23 = wasm_v128_load(squares + group + 2u);
                v128_t total45 = wasm_v128_load(squares + group + 4u);
                v128_t total67 = wasm_v128_load(squares + group + 6u);
                const v128_t mean01 = wasm_v128_load(means + group);
                const v128_t mean23 = wasm_v128_load(means + group + 2u);
                const v128_t mean45 = wasm_v128_load(means + group + 4u);
                const v128_t mean67 = wasm_v128_load(means + group + 6u);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span0 = input + sample_offset +
                        point * channels + first;
                    const float *span1 = span0 + channels_per_group;
                    const float *span2 = span1 + channels_per_group;
                    const float *span3 = span2 + channels_per_group;
                    const float *span4 = span3 + channels_per_group;
                    const float *span5 = span4 + channels_per_group;
                    const float *span6 = span5 + channels_per_group;
                    const float *span7 = span6 + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const v128_t centered01 = wasm_f64x2_sub(
                            wasm_f64x2_make((double)span0[local],
                                            (double)span1[local]), mean01);
                        const v128_t centered23 = wasm_f64x2_sub(
                            wasm_f64x2_make((double)span2[local],
                                            (double)span3[local]), mean23);
                        const v128_t centered45 = wasm_f64x2_sub(
                            wasm_f64x2_make((double)span4[local],
                                            (double)span5[local]), mean45);
                        const v128_t centered67 = wasm_f64x2_sub(
                            wasm_f64x2_make((double)span6[local],
                                            (double)span7[local]), mean67);
                        total01 = wasm_f64x2_add(total01,
                            wasm_f64x2_mul(centered01, centered01));
                        total23 = wasm_f64x2_add(total23,
                            wasm_f64x2_mul(centered23, centered23));
                        total45 = wasm_f64x2_add(total45,
                            wasm_f64x2_mul(centered45, centered45));
                        total67 = wasm_f64x2_add(total67,
                            wasm_f64x2_mul(centered67, centered67));
                    }
                }
                wasm_v128_store(squares + group, total01);
                wasm_v128_store(squares + group + 2u, total23);
                wasm_v128_store(squares + group + 4u, total45);
                wasm_v128_store(squares + group + 6u, total67);
            }
            for (; group + 3u < groups; group += 4u) {
                v128_t total01 = wasm_v128_load(squares + group);
                v128_t total23 = wasm_v128_load(squares + group + 2u);
                const v128_t mean01 = wasm_v128_load(means + group);
                const v128_t mean23 = wasm_v128_load(means + group + 2u);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span0 = input + sample_offset +
                        point * channels + first;
                    const float *span1 = span0 + channels_per_group;
                    const float *span2 = span1 + channels_per_group;
                    const float *span3 = span2 + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const v128_t centered01 = wasm_f64x2_sub(
                            wasm_f64x2_make((double)span0[local],
                                            (double)span1[local]), mean01);
                        const v128_t centered23 = wasm_f64x2_sub(
                            wasm_f64x2_make((double)span2[local],
                                            (double)span3[local]), mean23);
                        total01 = wasm_f64x2_add(total01,
                            wasm_f64x2_mul(centered01, centered01));
                        total23 = wasm_f64x2_add(total23,
                            wasm_f64x2_mul(centered23, centered23));
                    }
                }
                wasm_v128_store(squares + group, total01);
                wasm_v128_store(squares + group + 2u, total23);
            }
            for (; group + 1u < groups; group += 2u) {
                v128_t total = wasm_v128_load(squares + group);
                const v128_t mean = wasm_v128_load(means + group);
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *low = input + sample_offset +
                        point * channels + first;
                    const float *high = low + channels_per_group;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const v128_t centered = wasm_f64x2_sub(
                            wasm_f64x2_make((double)low[local],
                                            (double)high[local]), mean);
                        total = wasm_f64x2_add(total,
                            wasm_f64x2_mul(centered, centered));
                    }
                }
                wasm_v128_store(squares + group, total);
            }
            if (group < groups) {
                double total = squares[group];
                const double mean = means[group];
                const uint32_t first = group * channels_per_group;
                for (size_t point = tile; point < limit; point++) {
                    const float *span = input + sample_offset +
                        point * channels + first;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        const double centered = (double)span[local] - mean;
                        total += centered * centered;
                    }
                }
                squares[group] = total;
            }
            tile = limit;
        }
        for (group = 0; group < groups; group++)
            inverses[group] = 1.0 /
                __builtin_sqrt(squares[group] / (double)values + epsilon);
        for (size_t point = 0; point < spatial; point++) {
            const size_t row = sample_offset + point * channels;
            for (group = 0; group < groups; group++) {
                const uint32_t first = group * channels_per_group;
                const v128_t mean = wasm_f64x2_splat(means[group]);
                const v128_t inverse = wasm_f64x2_splat(inverses[group]);
                uint32_t local = 0;
                for (; local + 2u <= channels_per_group; local += 2u) {
                    const uint32_t channel = first + local;
                    const v128_t centered = wasm_f64x2_sub(
                        wasm_f64x2_promote_low_f32x4(
                            wasm_v128_load64_zero(input + row + channel)), mean);
                    const v128_t gain = wasm_f64x2_promote_low_f32x4(
                        wasm_v128_load64_zero(weight + channel));
                    v128_t normalized = wasm_f64x2_mul(
                        wasm_f64x2_mul(centered, inverse), gain);
                    if (bias)
                        normalized = wasm_f64x2_add(normalized,
                            wasm_f64x2_promote_low_f32x4(
                                wasm_v128_load64_zero(bias + channel)));
                    wasm_v128_store64_lane(output + row + channel,
                        wasm_f32x4_demote_f64x2_zero(normalized), 0);
                }
                for (; local < channels_per_group; local++) {
                    const uint32_t channel = first + local;
                    output[row + channel] = (float)(((double)input[row + channel] -
                        means[group]) * inverses[group] * weight[channel] +
                        (bias ? bias[channel] : 0.0f));
                }
            }
        }
    }
#if defined(VOLVOXAI_GROUPNORM_WASM_SIMD_TESTING)
    vx_groupnorm_wasm_simd_call_count++;
#endif
}
#endif

WASM_EXPORT("groupnorm_f32")
int groupnorm_f32(const float *input, const float *weight, const float *bias,
        float *output, uint32_t batch, uint32_t height, uint32_t width,
        uint32_t channels, uint32_t groups, double epsilon) {
    size_t spatial = 1, values = 1, sample_stride = 1;
    uint32_t channels_per_group;
    if (!input || !weight || !output || !batch || !height || !width ||
        !channels || !groups || channels % groups || !(epsilon > 0.0) ||
        !vx_pf_finite_f64(epsilon)) return 0;
    channels_per_group = channels / groups;
    if (!vx_pf_mul_size(&spatial, height) || !vx_pf_mul_size(&spatial, width) ||
        !vx_pf_mul_size(&values, spatial) || !vx_pf_mul_size(&values, channels_per_group) ||
        !vx_pf_mul_size(&sample_stride, spatial) || !vx_pf_mul_size(&sample_stride, channels) ||
        batch > (size_t)-1 / sample_stride) return 0;
#if VX_GROUPNORM_X86_AVX2
    if (vx_kernel_platform()->has_avx2) {
        if (channels_per_group < VX_GROUPNORM_MIN_VECTOR_CHANNELS &&
            groups <= VX_GROUPNORM_MAX_STREAMED_GROUPS) {
            vx_groupnorm_streamed_avx2(input, weight, bias, output, batch,
                channels, groups, channels_per_group, spatial, values,
                sample_stride, epsilon);
            return 1;
        }
        if (channels_per_group >= VX_GROUPNORM_MIN_VECTOR_CHANNELS) {
            for (uint32_t sample = 0; sample < batch; sample++) {
                const size_t sample_offset = (size_t)sample * sample_stride;
                for (uint32_t group = 0; group < groups; group++)
                    vx_groupnorm_group_avx2(input, weight, bias, output,
                        sample_offset, group * channels_per_group, channels,
                        channels_per_group, spatial, values, epsilon);
            }
            return 1;
        }
    }
#endif
#if VX_GROUPNORM_WASM_SIMD
    if (groups <= VX_GROUPNORM_MAX_STREAMED_GROUPS) {
        vx_groupnorm_streamed_wasm_simd(input, weight, bias, output, batch,
            channels, groups, channels_per_group, spatial, values,
            sample_stride, epsilon);
        return 1;
    }
#endif
    /* Stream every group at once with spatial outermost.
     *
     * Walking one group at a time reads `channels_per_group` floats out of each
     * `channels`-float row, so with cpg=3 of 96 each pass touches every cache
     * line to use 12 of its 384 bytes -- and there are three such passes per
     * group. Hoisting spatial outside and carrying one accumulator per group
     * turns that into three sequential passes over the tensor, total.
     *
     * Each group's additions still happen in the original order (point ascending,
     * then channel), and accumulation stays in double, so results are unchanged.
     * Variance keeps the two-pass form on purpose: folding it into sum/sumsq
     * would subtract two nearly equal numbers and lose the low bits. */
    if (groups <= VX_GROUPNORM_MAX_STREAMED_GROUPS) {
        double sums[VX_GROUPNORM_MAX_STREAMED_GROUPS];
        double squares[VX_GROUPNORM_MAX_STREAMED_GROUPS];
        double means[VX_GROUPNORM_MAX_STREAMED_GROUPS];
        double inverses[VX_GROUPNORM_MAX_STREAMED_GROUPS];
        for (uint32_t sample = 0; sample < batch; sample++) {
            const size_t sample_offset = (size_t)sample * sample_stride;
            for (uint32_t group = 0; group < groups; group++) sums[group] = 0.0;
            for (size_t point = 0; point < spatial; point++) {
                const float *row = input + sample_offset + point * channels;
                for (uint32_t group = 0; group < groups; group++) {
                    const float *span = row + (size_t)group * channels_per_group;
                    double total = sums[group];
                    for (uint32_t local = 0; local < channels_per_group; local++)
                        total += span[local];
                    sums[group] = total;
                }
            }
            for (uint32_t group = 0; group < groups; group++) {
                means[group] = sums[group] / (double)values;
                squares[group] = 0.0;
            }
            for (size_t point = 0; point < spatial; point++) {
                const float *row = input + sample_offset + point * channels;
                for (uint32_t group = 0; group < groups; group++) {
                    const float *span = row + (size_t)group * channels_per_group;
                    double mean = means[group], total = squares[group];
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        double centered = (double)span[local] - mean;
                        total += centered * centered;
                    }
                    squares[group] = total;
                }
            }
            for (uint32_t group = 0; group < groups; group++)
                inverses[group] = 1.0 /
                    __builtin_sqrt(squares[group] / (double)values + epsilon);
            for (size_t point = 0; point < spatial; point++) {
                const size_t row = sample_offset + point * channels;
                for (uint32_t group = 0; group < groups; group++) {
                    const uint32_t first_channel = group * channels_per_group;
                    const double mean = means[group], inverse = inverses[group];
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        uint32_t channel = first_channel + local;
                        output[row + channel] = (float)(((double)input[row + channel] - mean) *
                            inverse * weight[channel] + (bias ? bias[channel] : 0.0f));
                    }
                }
            }
        }
        return 1;
    }
    for (uint32_t sample = 0; sample < batch; sample++) {
        size_t sample_offset = (size_t)sample * sample_stride;
        for (uint32_t group = 0; group < groups; group++) {
            uint32_t first_channel = group * channels_per_group;
            double mean = 0.0;
            double variance = 0.0;
            for (size_t point = 0; point < spatial; point++) {
                size_t offset = sample_offset + point * channels + first_channel;
                for (uint32_t local = 0; local < channels_per_group; local++) {
                    mean += input[offset + local];
                }
            }
            mean /= (double)values;
            for (size_t point = 0; point < spatial; point++) {
                size_t offset = sample_offset + point * channels + first_channel;
                for (uint32_t local = 0; local < channels_per_group; local++) {
                    double centered = (double)input[offset + local] - mean;
                    variance += centered * centered;
                }
            }
            {
                double inverse = 1.0 / __builtin_sqrt(variance / (double)values + epsilon);
                for (size_t point = 0; point < spatial; point++) {
                    size_t offset = sample_offset + point * channels + first_channel;
                    for (uint32_t local = 0; local < channels_per_group; local++) {
                        uint32_t channel = first_channel + local;
                        output[offset + local] = (float)(((double)input[offset + local] - mean) *
                            inverse * weight[channel] + (bias ? bias[channel] : 0.0f));
                    }
                }
            }
        }
    }
    return 1;
}

WASM_EXPORT("prelu_generic_f32")
int prelu_generic_f32(const float *input, const float *slope, float *output,
        uint32_t elements, uint32_t slope_elements, uint32_t channels) {
    if (!input || !slope || !output || !elements || !slope_elements || !channels) return 0;
    for (uint32_t index = 0; index < elements; index++) {
        float value = input[index];
        uint32_t slope_index = slope_elements == channels ? index % channels : index % slope_elements;
        output[index] = value < 0.0f ? value * slope[slope_index] : value;
    }
    return 1;
}

WASM_EXPORT("resize_nearest2d_f32")
int resize_nearest2d_f32(const float *input, float *output, uint32_t batch,
        uint32_t input_height, uint32_t input_width, uint32_t channels,
        uint32_t output_height, uint32_t output_width) {
    size_t input_elements = 1, output_elements = 1;
    if (!input || !output || !batch || !input_height || !input_width || !channels ||
        !output_height || !output_width) return 0;
    if (!vx_pf_mul_size(&input_elements, batch) || !vx_pf_mul_size(&input_elements, input_height) ||
        !vx_pf_mul_size(&input_elements, input_width) || !vx_pf_mul_size(&input_elements, channels) ||
        !vx_pf_mul_size(&output_elements, batch) || !vx_pf_mul_size(&output_elements, output_height) ||
        !vx_pf_mul_size(&output_elements, output_width) || !vx_pf_mul_size(&output_elements, channels)) return 0;
    for (uint32_t sample = 0; sample < batch; sample++) {
        for (uint32_t y = 0; y < output_height; y++) {
            uint32_t source_y = (uint32_t)(((uint64_t)y * input_height) / output_height);
            for (uint32_t x = 0; x < output_width; x++) {
                uint32_t source_x = (uint32_t)(((uint64_t)x * input_width) / output_width);
                size_t source = (((size_t)sample * input_height + source_y) * input_width + source_x) * channels;
                size_t destination = (((size_t)sample * output_height + y) * output_width + x) * channels;
                for (uint32_t channel = 0; channel < channels; channel++) {
                    output[destination + channel] = input[source + channel];
                }
            }
        }
    }
    return 1;
}

WASM_EXPORT("resize_nearest2d_i8u8")
int resize_nearest2d_i8u8(const uint8_t *input, uint8_t *output, uint32_t batch,
        uint32_t input_height, uint32_t input_width, uint32_t channels,
        uint32_t output_height, uint32_t output_width, uint32_t dtype) {
    size_t input_elements = 1, output_elements = 1;
    if (!input || !output || !batch || !input_height || !input_width || !channels ||
        !output_height || !output_width || !vx_w8a8_byte_dtype(dtype)) return 0;
    if (!vx_pf_mul_size(&input_elements, batch) || !vx_pf_mul_size(&input_elements, input_height) ||
        !vx_pf_mul_size(&input_elements, input_width) || !vx_pf_mul_size(&input_elements, channels) ||
        !vx_pf_mul_size(&output_elements, batch) || !vx_pf_mul_size(&output_elements, output_height) ||
        !vx_pf_mul_size(&output_elements, output_width) || !vx_pf_mul_size(&output_elements, channels)) return 0;
    for (uint32_t sample = 0; sample < batch; sample++) {
        for (uint32_t y = 0; y < output_height; y++) {
            uint32_t source_y = (uint32_t)(((uint64_t)y * input_height) / output_height);
            for (uint32_t x = 0; x < output_width; x++) {
                uint32_t source_x = (uint32_t)(((uint64_t)x * input_width) / output_width);
                size_t source = (((size_t)sample * input_height + source_y) * input_width + source_x) * channels;
                size_t destination = (((size_t)sample * output_height + y) * output_width + x) * channels;
                for (uint32_t channel = 0; channel < channels; channel++) {
                    output[destination + channel] = input[source + channel];
                }
            }
        }
    }
    return 1;
}

WASM_EXPORT("maxpool2d_i8u8")
int maxpool2d_i8u8(const uint8_t *input, uint8_t *output, uint32_t batch,
        uint32_t input_height, uint32_t input_width, uint32_t channels,
        uint32_t output_height, uint32_t output_width, uint32_t kernel_y,
        uint32_t kernel_x, uint32_t stride_y, uint32_t stride_x,
        uint32_t padding_y, uint32_t padding_x, uint32_t dtype) {
    size_t input_elements = 1, output_elements = 1;
    if (!input || !output || !batch || !input_height || !input_width || !channels ||
        !output_height || !output_width || !kernel_y || !kernel_x || !stride_y || !stride_x ||
        !vx_w8a8_byte_dtype(dtype)) return 0;
    if (!vx_pf_mul_size(&input_elements, batch) || !vx_pf_mul_size(&input_elements, input_height) ||
        !vx_pf_mul_size(&input_elements, input_width) || !vx_pf_mul_size(&input_elements, channels) ||
        !vx_pf_mul_size(&output_elements, batch) || !vx_pf_mul_size(&output_elements, output_height) ||
        !vx_pf_mul_size(&output_elements, output_width) || !vx_pf_mul_size(&output_elements, channels)) return 0;
    for (uint32_t sample = 0; sample < batch; sample++) {
        for (uint32_t output_y = 0; output_y < output_height; output_y++) {
            uint64_t origin_y = (uint64_t)output_y * stride_y;
            for (uint32_t output_x = 0; output_x < output_width; output_x++) {
                uint64_t origin_x = (uint64_t)output_x * stride_x;
                for (uint32_t channel = 0; channel < channels; channel++) {
                    size_t destination = (((size_t)sample * output_height + output_y) * output_width + output_x) * channels + channel;
                    if (dtype == VX_DTYPE_I8) {
                        int8_t best = -128;
                        for (uint32_t kernel_row = 0; kernel_row < kernel_y; kernel_row++) {
                            uint64_t padded_y = origin_y + kernel_row;
                            if (padded_y < padding_y || padded_y - padding_y >= input_height) continue;
                            for (uint32_t kernel_column = 0; kernel_column < kernel_x; kernel_column++) {
                                uint64_t padded_x = origin_x + kernel_column;
                                if (padded_x < padding_x || padded_x - padding_x >= input_width) continue;
                                size_t source = (((size_t)sample * input_height +
                                    (uint32_t)(padded_y - padding_y)) * input_width +
                                    (uint32_t)(padded_x - padding_x)) * channels + channel;
                                int8_t value = ((const int8_t *)input)[source];
                                if (value > best) best = value;
                            }
                        }
                        output[destination] = (uint8_t)best;
                    } else {
                        uint8_t best = 0;
                        for (uint32_t kernel_row = 0; kernel_row < kernel_y; kernel_row++) {
                            uint64_t padded_y = origin_y + kernel_row;
                            if (padded_y < padding_y || padded_y - padding_y >= input_height) continue;
                            for (uint32_t kernel_column = 0; kernel_column < kernel_x; kernel_column++) {
                                uint64_t padded_x = origin_x + kernel_column;
                                if (padded_x < padding_x || padded_x - padding_x >= input_width) continue;
                                size_t source = (((size_t)sample * input_height +
                                    (uint32_t)(padded_y - padding_y)) * input_width +
                                    (uint32_t)(padded_x - padding_x)) * channels + channel;
                                uint8_t value = input[source];
                                if (value > best) best = value;
                            }
                        }
                        output[destination] = best;
                    }
                }
            }
        }
    }
    return 1;
}

static int vx_pf_moe_router_dimensions_valid(uint32_t rows, uint32_t d_model,
        uint32_t experts, uint32_t top_k, double temperature, uint32_t normalize) {
    size_t product = 1;
    if (!rows || !d_model || !experts || !top_k || top_k > experts || normalize > 1u ||
        !(temperature > 0.0) || !vx_pf_finite_f64(temperature)) return 0;
    if (!vx_pf_mul_size(&product, rows) || !vx_pf_mul_size(&product, d_model)) return 0;
    product = 1;
    if (!vx_pf_mul_size(&product, d_model) || !vx_pf_mul_size(&product, experts)) return 0;
    product = 1;
    return vx_pf_mul_size(&product, rows) && vx_pf_mul_size(&product, top_k);
}

static int vx_pf_moe_linear_dimensions_valid(uint32_t rows, uint32_t d_in,
        uint32_t d_out, uint32_t experts, uint32_t top_k) {
    size_t product = 1;
    if (!rows || !d_in || !d_out || !experts || !top_k || top_k > experts) return 0;
    if (!vx_pf_mul_size(&product, rows) || !vx_pf_mul_size(&product, d_in)) return 0;
    product = 1;
    if (!vx_pf_mul_size(&product, experts) || !vx_pf_mul_size(&product, d_in) ||
        !vx_pf_mul_size(&product, d_out)) return 0;
    product = 1;
    if (!vx_pf_mul_size(&product, rows) || !vx_pf_mul_size(&product, d_out)) return 0;
    product = 1;
    return vx_pf_mul_size(&product, rows) && vx_pf_mul_size(&product, top_k);
}

static float vx_pf_moe_router_logit(const float *input, const float *weight,
        const float *bias, uint32_t row, uint32_t expert, uint32_t d_model,
        uint32_t experts, double temperature) {
    double value = bias ? bias[expert] : 0.0;
    size_t input_offset = (size_t)row * d_model;
    for (uint32_t dimension = 0; dimension < d_model; dimension++) {
        value += (double)input[input_offset + dimension] *
            weight[(size_t)dimension * experts + expert];
    }
    return (float)(value / temperature);
}

static int vx_pf_route_index(const float *indices, uint32_t offset,
        uint32_t experts, uint32_t *expert) {
    float raw = indices[offset];
    uint32_t value;
    if (!vx_pf_finite_f32(raw) || raw < 0.0f || raw >= (float)experts) return 0;
    value = (uint32_t)raw;
    if ((float)value != raw) return 0;
    *expert = value;
    return 1;
}

/* Resolve one global slot id against a partially resident bank.  slot_rows maps
 * every id in [0, slot_domain) to its staged row, or VX_MOE_SLOT_ABSENT when the
 * context did not materialize it.  A NULL table means the bank is fully
 * resident and ids are already rows.  Routing to an absent slot is an error
 * rather than a silent read of a neighbouring expert. */
static int vx_pf_moe_resident_row(const uint32_t *slot_rows, uint32_t slot_domain,
        uint32_t staged_rows, uint32_t slot, uint32_t *row) {
    uint32_t mapped;
    if (!slot_rows) {
        if (slot >= staged_rows) return 0;
        *row = slot;
        return 1;
    }
    if (slot >= slot_domain) return 0;
    mapped = slot_rows[slot];
    if (mapped >= staged_rows) return 0;
    *row = mapped;
    return 1;
}

WASM_EXPORT("moe_router_f32")
int moe_router_f32(const float *input, const float *weight, const float *bias,
        float *indices, float *route_weights, uint32_t rows, uint32_t d_model,
        uint32_t experts, uint32_t top_k, double temperature, uint32_t normalize) {
    if (!input || !weight || !indices || !route_weights ||
        !vx_pf_moe_router_dimensions_valid(rows, d_model, experts, top_k,
            temperature, normalize)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        float maximum = 0.0f;
        uint32_t have_maximum = 0;
        for (uint32_t expert = 0; expert < experts; expert++) {
            float logit = vx_pf_moe_router_logit(input, weight, bias, row, expert,
                d_model, experts, temperature);
            if (!vx_pf_finite_f32(logit)) return 0;
            if (!have_maximum || logit > maximum) {
                maximum = logit;
                have_maximum = 1;
            }
        }
        for (uint32_t slot = 0; slot < top_k; slot++) {
            uint32_t best = 0;
            float best_value = 0.0f;
            uint32_t have_best = 0;
            for (uint32_t expert = 0; expert < experts; expert++) {
                uint32_t selected = 0;
                for (uint32_t prior = 0; prior < slot; prior++) {
                    if ((uint32_t)indices[(size_t)row * top_k + prior] == expert) {
                        selected = 1;
                        break;
                    }
                }
                if (selected) continue;
                {
                    float logit = vx_pf_moe_router_logit(input, weight, bias, row,
                        expert, d_model, experts, temperature);
                    if (!vx_pf_finite_f32(logit)) return 0;
                    if (!have_best || logit > best_value ||
                        (logit == best_value && expert < best)) {
                        best = expert;
                        best_value = logit;
                        have_best = 1;
                    }
                }
            }
            if (!have_best) return 0;
            indices[(size_t)row * top_k + slot] = (float)best;
        }
        {
            double denominator = 0.0;
            uint32_t count = normalize ? top_k : experts;
            for (uint32_t item = 0; item < count; item++) {
                uint32_t expert = normalize ?
                    (uint32_t)indices[(size_t)row * top_k + item] : item;
                denominator += expf(vx_pf_moe_router_logit(input, weight, bias, row,
                    expert, d_model, experts, temperature) - maximum);
            }
            if (!(denominator > 0.0)) return 0;
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t expert = (uint32_t)indices[(size_t)row * top_k + slot];
                route_weights[(size_t)row * top_k + slot] = (float)(expf(
                    vx_pf_moe_router_logit(input, weight, bias, row, expert,
                        d_model, experts, temperature) - maximum) / denominator);
            }
        }
    }
    return 1;
}

WASM_EXPORT("moe_linear_f32")
int vx_moe_linear_banked_f32(const float *, const float *, const float *,
        const float *, const float *, float *, uint32_t, uint32_t, uint32_t,
        uint32_t, uint32_t, const uint32_t *, uint32_t);

int moe_linear_f32(const float *input, const float *expert_weight,
        const float *expert_bias, const float *route_indices,
        const float *route_weights, float *output, uint32_t rows,
        uint32_t d_in, uint32_t d_out, uint32_t experts, uint32_t top_k) {
    return vx_moe_linear_banked_f32(input, expert_weight, expert_bias,
        route_indices, route_weights, output, rows, d_in, d_out, experts,
        top_k, NULL, 0);
}

WASM_EXPORT("vx_moe_linear_banked_f32")
/* Banked form.  expert_weight holds only the resident slots, so `experts` is
 * the staged row count while route indices stay in the model's global slot
 * space.  slot_rows/slot_domain carry the mapping; NULL means fully resident. */
int vx_moe_linear_banked_f32(const float *input, const float *expert_weight,
        const float *expert_bias, const float *route_indices,
        const float *route_weights, float *output, uint32_t rows,
        uint32_t d_in, uint32_t d_out, uint32_t experts, uint32_t top_k,
        const uint32_t *slot_rows, uint32_t slot_domain) {
    uint32_t route_domain = slot_rows ? slot_domain : experts;
    if (!input || !expert_weight || !route_indices || !route_weights || !output ||
        !vx_pf_moe_linear_dimensions_valid(rows, d_in, d_out, experts, top_k)) return 0;
    if (slot_rows && (slot_domain < experts || top_k > slot_domain)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < d_out; column++) {
            double sum = 0.0;
            for (uint32_t slot = 0; slot < top_k; slot++) {
                uint32_t route_offset = row * top_k + slot;
                uint32_t expert;
                uint32_t staged;
                float gate = route_weights[route_offset];
                size_t expert_offset;
                double value;
                if (!vx_pf_route_index(route_indices, route_offset, route_domain, &expert) ||
                    !vx_pf_finite_f32(gate)) return 0;
                if (!vx_pf_moe_resident_row(slot_rows, slot_domain, experts, expert,
                        &staged)) return 0;
                expert = staged;
                expert_offset = (size_t)expert * d_in * d_out;
                value = expert_bias ? expert_bias[(size_t)expert * d_out + column] : 0.0;
                for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                    value += (double)input[(size_t)row * d_in + dimension] *
                        expert_weight[expert_offset + (size_t)dimension * d_out + column];
                }
                sum += gate * value;
            }
            output[(size_t)row * d_out + column] = (float)sum;
        }
    }
    return 1;
}
