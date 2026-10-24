#include "training_core.h"
#include "../kernels/training_kernels.h"
#include "../kernels/packed_linear.h"

#include <float.h>
#include <limits.h>
#include <stdint.h>

#if !defined(__wasm__)
#include <math.h>
#endif

static int ptq_finite_f32(float value) {
    union { float value; uint32_t bits; } encoded = { value };
    return (encoded.bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static int ptq_finite_f64(double value) {
    union { double value; uint64_t bits; } encoded = { value };
    return (encoded.bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

static double ptq_abs_f64(double value) {
    return value < 0.0 ? -value : value;
}

static double ptq_floor_f64(double value) {
#if defined(__wasm__)
    return __builtin_floor(value);
#else
    return floor(value);
#endif
}

static int ptq_product_fits_u32(uint32_t left, uint32_t right, size_t* output) {
    if (!output || (left != 0 && (size_t)right > SIZE_MAX / (size_t)left)) return 0;
    *output = (size_t)left * (size_t)right;
    return 1;
}

static int ptq_domain(int32_t dtype, int32_t* minimum, int32_t* maximum) {
    if (dtype == VOLVOXAI_DTYPE_I8) {
        *minimum = -128;
        *maximum = 127;
        return 0;
    }
    if (dtype == VOLVOXAI_DTYPE_U8) {
        *minimum = 0;
        *maximum = 255;
        return 0;
    }
    return -1;
}

static int ptq_positive_f32(double value, float* output) {
    if (!output || !ptq_finite_f64(value) || value <= 0.0) return -1;
    float rounded = (float)value;
    if (ptq_finite_f32(rounded) && rounded > 0.0f) {
        *output = rounded;
        return 0;
    }
    if (rounded == 0.0f) {
        *output = FLT_MIN;
        return 0;
    }
    return -1;
}

/* max_abs / bound can round upward enough that reconstructing the endpoint
 * overflows F32. Only computed weight scales use this one-ULP correction;
 * caller-supplied scales retain their exact values. */
static float ptq_finite_weight_scale(float scale, int bound) {
    if (!ptq_finite_f32((float)bound * scale)) {
        union { uint32_t bits; float value; } encoded = {0};
        encoded.value = scale;
        --encoded.bits;
        return encoded.value;
    }
    return scale;
}

static double ptq_round_ties_even(double value) {
    double lower = ptq_floor_f64(value);
    double fraction = value - lower;
    if (fraction < 0.5) return lower;
    if (fraction > 0.5) return lower + 1.0;
    /* A binary64 value can retain an exact .5 fraction only while its integer
       part fits safely in int64_t. Keep rounding-mode-independent ties-to-even
       without requiring the freestanding WASM module to import fmod(). */
    int64_t integer = (int64_t)lower;
    return integer % 2 == 0 ? lower : lower + 1.0;
}

static int ptq_values_finite(const float* values, int64_t count) {
    if (count < 0 || (uint64_t)count > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        (count > 0 && !values)) return -1;
    for (int64_t index = 0; index < count; index++) {
        if (!ptq_finite_f32(values[index])) return -1;
    }
    return 0;
}

uint32_t volvoxai_ptq_abi_version(void) {
    return VOLVOXAI_PTQ_ABI_VERSION;
}

uint32_t volvoxai_ptq_capabilities(void) {
    return VOLVOXAI_PTQ_CAP_ALL;
}

void volvoxai_ptq_observer_reset(volvoxai_ptq_observer_t* observer) {
    if (!observer) return;
    observer->minimum = __builtin_inff();
    observer->maximum = -__builtin_inff();
    observer->sample_count = 0;
}

int volvoxai_ptq_observer_observe_f32(volvoxai_ptq_observer_t* observer,
                                      const float* values, int64_t count) {
    if (!observer || ptq_values_finite(values, count) != 0) return -1;
    if (count == 0) return 0;
    if ((uint64_t)count > UINT64_MAX - observer->sample_count) return -1;
    float minimum = values[0];
    float maximum = values[0];
    for (int64_t index = 1; index < count; index++) {
        if (values[index] < minimum) minimum = values[index];
        if (values[index] > maximum) maximum = values[index];
    }
    if (observer->sample_count == 0) {
        observer->minimum = minimum;
        observer->maximum = maximum;
    } else {
        if (minimum < observer->minimum) observer->minimum = minimum;
        if (maximum > observer->maximum) observer->maximum = maximum;
    }
    observer->sample_count += (uint64_t)count;
    return 0;
}

int volvoxai_ptq_calculate_params(const volvoxai_ptq_observer_t* observer,
                                  int32_t dtype, int32_t scheme,
                                  volvoxai_ptq_params_t* out_params) {
    int32_t qmin = 0;
    int32_t qmax = 0;
    if (!observer || !out_params || observer->sample_count == 0 ||
        !ptq_finite_f32(observer->minimum) || !ptq_finite_f32(observer->maximum) ||
        observer->minimum > observer->maximum ||
        ptq_domain(dtype, &qmin, &qmax) != 0) return -1;
    float scale = 0.0f;
    int32_t zero_point = 0;
    if (scheme == VX_AFFINE_SCHEME_SYMMETRIC) {
        double minimum_magnitude = ptq_abs_f64((double)observer->minimum);
        double maximum_magnitude = ptq_abs_f64((double)observer->maximum);
        double magnitude = minimum_magnitude > maximum_magnitude
            ? minimum_magnitude : maximum_magnitude;
        if (ptq_positive_f32(magnitude == 0.0 ? 1.0 : magnitude / 127.0, &scale) != 0) return -1;
        zero_point = dtype == VOLVOXAI_DTYPE_I8 ? 0 : 128;
    } else if (scheme == VX_AFFINE_SCHEME_ASYMMETRIC) {
        double minimum = observer->minimum < 0.0f ? (double)observer->minimum : 0.0;
        double maximum = observer->maximum > 0.0f ? (double)observer->maximum : 0.0;
        if (minimum == maximum) {
            scale = 1.0f;
            zero_point = 0;
        } else {
            if (ptq_positive_f32((maximum - minimum) / (double)(qmax - qmin), &scale) != 0) return -1;
            double rounded = ptq_round_ties_even((double)qmin - minimum / (double)scale);
            if (rounded < (double)qmin) rounded = (double)qmin;
            if (rounded > (double)qmax) rounded = (double)qmax;
            zero_point = (int)rounded;
        }
    } else {
        return -1;
    }
    out_params->dtype = dtype;
    out_params->scheme = scheme;
    out_params->scale = scale;
    out_params->zero_point = zero_point;
    out_params->observed_min = observer->minimum;
    out_params->observed_max = observer->maximum;
    out_params->sample_count = observer->sample_count;
    return 0;
}

int volvoxai_ptq_quantize_f32(const float* values, int64_t count,
                              const volvoxai_ptq_params_t* params,
                              void* output, uint64_t* saturation_count) {
    int32_t qmin = 0;
    int32_t qmax = 0;
    if (!params || (count > 0 && !output) || ptq_values_finite(values, count) != 0 ||
        ptq_domain(params->dtype, &qmin, &qmax) != 0 ||
        !ptq_finite_f32(params->scale) || params->scale <= 0.0f ||
        params->zero_point < qmin || params->zero_point > qmax) return -1;
    uint64_t saturated = 0;
    for (int64_t index = 0; index < count; index++) {
        double rounded = ptq_round_ties_even((double)values[index] / (double)params->scale +
                                             (double)params->zero_point);
        if (rounded < (double)qmin) {
            rounded = (double)qmin;
            saturated++;
        } else if (rounded > (double)qmax) {
            rounded = (double)qmax;
            saturated++;
        }
        if (params->dtype == VOLVOXAI_DTYPE_I8) ((int8_t*)output)[index] = (int8_t)rounded;
        else ((uint8_t*)output)[index] = (uint8_t)rounded;
    }
    if (saturation_count) *saturation_count = saturated;
    return 0;
}

static int ptq_shape_info(const int32_t* shape, int32_t ndim, int32_t axis,
                          int64_t* elements, int32_t* normalized_axis,
                          int32_t* channels, int64_t* inner) {
    if (!shape || ndim <= 0 || ndim > 8) return -1;
    int32_t resolved = axis < 0 ? axis + ndim : axis;
    if (resolved < 0 || resolved >= ndim) return -1;
    int64_t total = 1;
    for (int32_t index = 0; index < ndim; index++) {
        if (shape[index] <= 0 || total > INT64_MAX / shape[index]) return -1;
        total *= shape[index];
    }
    if ((uint64_t)total > (uint64_t)(SIZE_MAX / sizeof(float))) return -1;
    int64_t stride = 1;
    for (int32_t index = resolved + 1; index < ndim; index++) stride *= shape[index];
    *elements = total;
    *normalized_axis = resolved;
    *channels = shape[resolved];
    *inner = stride;
    return 0;
}

static int ptq_ranges_overlap(const void* left, size_t left_size,
                              const void* right, size_t right_size) {
    if (!left_size || !right_size) return 0;
    uintptr_t left_start = (uintptr_t)left;
    uintptr_t right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - left_size ||
        right_start > UINTPTR_MAX - right_size) return 1;
    return left_start < right_start + right_size &&
           right_start < left_start + left_size;
}

static int8_t ptq_quantize_symmetric_i8(float value, float scale,
                                        int32_t bound, uint64_t* saturation_count) {
    double rounded = ptq_round_ties_even((double)value / (double)scale);
    if (rounded < -(double)bound) {
        rounded = -(double)bound;
        (*saturation_count)++;
    } else if (rounded > (double)bound) {
        rounded = (double)bound;
        (*saturation_count)++;
    }
    return (int8_t)rounded;
}

int volvoxai_ptq_pack_weight_i8(const float* values, const int32_t* shape,
                                int32_t ndim, int32_t axis, int8_t* output,
                                float* scales, int32_t scale_count,
                                uint64_t* saturation_count) {
    return volvoxai_ptq_pack_weight_i8_range(values, shape, ndim, axis, output,
        scales, scale_count, 127, saturation_count);
}

int volvoxai_ptq_pack_weight_i8_range(const float* values, const int32_t* shape,
                                      int32_t ndim, int32_t axis, int8_t* output,
                                      float* scales, int32_t scale_count,
                                      int32_t bound, uint64_t* saturation_count) {
    int64_t elements = 0;
    int64_t inner = 0;
    int32_t resolved_axis = 0;
    int32_t channels = 0;
    if ((bound != 64 && bound != 127) ||
        ptq_shape_info(shape, ndim, axis, &elements, &resolved_axis, &channels, &inner) != 0 ||
        !values || !output || !scales || scale_count != channels ||
        ptq_values_finite(values, elements) != 0) return -1;
    (void)resolved_axis;
    if ((uint64_t)channels > (uint64_t)(SIZE_MAX / sizeof(float))) return -1;
    size_t values_bytes = (size_t)elements * sizeof(*values);
    size_t output_bytes = (size_t)elements * sizeof(*output);
    size_t scales_bytes = (size_t)channels * sizeof(*scales);
    if (ptq_ranges_overlap(values, values_bytes, output, output_bytes) ||
        ptq_ranges_overlap(values, values_bytes, scales, scales_bytes) ||
        ptq_ranges_overlap(output, output_bytes, scales, scales_bytes)) return -1;
    for (int32_t channel = 0; channel < channels; channel++) scales[channel] = 0.0f;
    for (int64_t index = 0; index < elements; index++) {
        int32_t channel = (int32_t)((index / inner) % channels);
        float magnitude = values[index] < 0.0f ? -values[index] : values[index];
        if (magnitude > scales[channel]) scales[channel] = magnitude;
    }
    for (int32_t channel = 0; channel < channels; channel++) {
        double magnitude = scales[channel];
        if (ptq_positive_f32(magnitude == 0.0 ? 1.0 : magnitude / (double)bound,
                             &scales[channel]) != 0) return -1;
        scales[channel] = ptq_finite_weight_scale(scales[channel], bound);
    }
    uint64_t saturated = 0;
    for (int64_t index = 0; index < elements; index++) {
        int32_t channel = (int32_t)((index / inner) % channels);
        output[index] = ptq_quantize_symmetric_i8(values[index], scales[channel],
                                                   bound, &saturated);
    }
    if (saturation_count) *saturation_count = saturated;
    return 0;
}

static int ptq_validate_weight_scales(const float* scales, int32_t count) {
    if (count < 0 || (count > 0 && !scales)) return -1;
    for (int32_t index = 0; index < count; index++) {
        if (!ptq_finite_f32(scales[index]) || scales[index] <= 0.0f) return -1;
    }
    return 0;
}

uint32_t volvoxai_training_quantize_weight_f32_to_i8(
        const float* source, int8_t* output, float* scales,
        uint32_t rows, uint32_t columns, uint32_t transpose_source,
        uint32_t scale_policy, uint64_t* saturation_count) {
    if (!rows || !columns || rows > INT32_MAX || transpose_source > 1 ||
        scale_policy > VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE) return 0;
    size_t elements = 0;
    if ((uint64_t)rows * (uint64_t)columns > INT64_MAX ||
        !ptq_product_fits_u32(rows, columns, &elements) ||
        elements > SIZE_MAX / sizeof(*source) || !source || !output || !scales ||
        ptq_values_finite(source, (int64_t)elements) != 0) return 0;
    size_t source_bytes = elements * sizeof(*source);
    size_t output_bytes = elements * sizeof(*output);
    size_t scales_bytes = (size_t)rows * sizeof(*scales);
    if (ptq_ranges_overlap(source, source_bytes, output, output_bytes) ||
        ptq_ranges_overlap(source, source_bytes, scales, scales_bytes) ||
        ptq_ranges_overlap(output, output_bytes, scales, scales_bytes)) return 0;
    if (scale_policy == VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE) {
        if (ptq_validate_weight_scales(scales, (int32_t)rows) != 0) return 0;
    } else {
        for (uint32_t row = 0; row < rows; row++) {
            float maximum = 0.0f;
            for (uint32_t column = 0; column < columns; column++) {
                size_t index = transpose_source
                    ? (size_t)column * rows + row
                    : (size_t)row * columns + column;
                float magnitude = source[index] < 0.0f ? -source[index] : source[index];
                if (magnitude > maximum) maximum = magnitude;
            }
            if (ptq_positive_f32(maximum == 0.0f ? 1.0 : (double)maximum / 127.0,
                                 &scales[row]) != 0) return 0;
            scales[row] = ptq_finite_weight_scale(scales[row], 127);
        }
    }
    uint64_t saturated = 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < columns; column++) {
            size_t source_index = transpose_source
                ? (size_t)column * rows + row
                : (size_t)row * columns + column;
            size_t output_index = (size_t)row * columns + column;
            output[output_index] = ptq_quantize_symmetric_i8(
                source[source_index], scales[row], 127, &saturated);
        }
    }
    if (saturation_count) *saturation_count = saturated;
    return 1;
}

uint32_t volvoxai_training_dequantize_weight_i8_to_f32(
        const int8_t* source, const float* scales, float* output,
        uint32_t rows, uint32_t columns, uint32_t transpose_destination) {
    if (!rows || !columns || transpose_destination > 1 || rows > INT32_MAX ||
        ptq_validate_weight_scales(scales, (int32_t)rows) != 0) return 0;
    size_t elements = 0;
    if (!ptq_product_fits_u32(rows, columns, &elements) ||
        elements > SIZE_MAX / sizeof(*output) || !source || !output) return 0;
    size_t source_bytes = elements * sizeof(*source);
    size_t output_bytes = elements * sizeof(*output);
    size_t scales_bytes = (size_t)rows * sizeof(*scales);
    if (ptq_ranges_overlap(source, source_bytes, output, output_bytes) ||
        ptq_ranges_overlap(source, source_bytes, scales, scales_bytes) ||
        ptq_ranges_overlap(output, output_bytes, scales, scales_bytes)) return 0;
    for (uint32_t row = 0; row < rows; row++) {
        for (uint32_t column = 0; column < columns; column++) {
            size_t source_index = (size_t)row * columns + column;
            size_t output_index = transpose_destination
                ? (size_t)column * rows + row
                : source_index;
            output[output_index] = (float)source[source_index] * scales[row];
        }
    }
    return 1;
}

int volvoxai_ptq_pack_bias_i32(const float* values, int32_t count,
                               float input_scale, const float* weight_scales,
                               int32_t scale_count, int32_t* output) {
    if (count < 0 || count != scale_count ||
        (uint64_t)count > (uint64_t)(SIZE_MAX / sizeof(float)) ||
        (uint64_t)count > (uint64_t)(SIZE_MAX / sizeof(int32_t)) ||
        (count > 0 && (!values || !weight_scales || !output)) ||
        !ptq_finite_f32(input_scale) || input_scale <= 0.0f ||
        ptq_values_finite(values, count) != 0) return -1;
    for (int32_t index = 0; index < count; index++) {
        if (!ptq_finite_f32(weight_scales[index]) || weight_scales[index] <= 0.0f) return -1;
        float accumulator_scale = input_scale * weight_scales[index];
        if (!ptq_finite_f32(accumulator_scale) || accumulator_scale <= 0.0f) return -1;
        float ratio = values[index] / accumulator_scale;
        if (!ptq_finite_f32(ratio)) return -1;
        double rounded = ptq_round_ties_even((double)ratio);
        if (rounded < (double)INT32_MIN || rounded > (double)INT32_MAX) return -1;
    }
    for (int32_t index = 0; index < count; index++) {
        float accumulator_scale = input_scale * weight_scales[index];
        output[index] = (int32_t)ptq_round_ties_even((double)(values[index] / accumulator_scale));
    }
    return 0;
}

/* Keep source F32 division/scale rounding identical in tensor helpers and
 * package authoring. Clamp before conversion so tiny preserved scales cannot
 * turn an infinite ratio into an undefined float-to-integer conversion. */
static int ptq_group_code(float value, float scale, int minimum, int maximum,
                          uint64_t* saturation) {
    float ratio = value / scale;
    if (ratio < (float)(minimum - 1)) {
        if (saturation) ++*saturation;
        return minimum;
    }
    if (ratio > (float)(maximum + 1)) {
        if (saturation) ++*saturation;
        return maximum;
    }
    int rounded = (int)ptq_round_ties_even((double)ratio);
    if (rounded < minimum) {
        if (saturation) ++*saturation;
        return minimum;
    }
    if (rounded > maximum) {
        if (saturation) ++*saturation;
        return maximum;
    }
    return rounded;
}

/* Output-channel fits share the same logical codebook as grouped weights.
 * Importance is the diagonal im2col second moment, independently accumulated
 * for each convolution group. It is never a full Hessian/reconstruction fit. */
int volvoxai_ptq_pack_conv_weight(const float* source, int32_t channels,
                                  int64_t columns, uint32_t bits,
                                  int32_t algorithm, const double* importance,
                                  uint32_t importance_groups,
                                  uint8_t* output, float* scales) {
    if (!source || !output || !scales || channels <= 0 || columns <= 0 ||
        (bits != 2 && bits != 3 && bits != 4) || columns > INT64_MAX / channels ||
        (algorithm != VX_WEIGHT_ALGORITHM_RTN &&
         algorithm != VX_WEIGHT_ALGORITHM_MSE_CLIP &&
         algorithm != VX_WEIGHT_ALGORITHM_IMPORTANCE) ||
        (algorithm == VX_WEIGHT_ALGORITHM_IMPORTANCE
            ? (!importance || !importance_groups ||
               (uint32_t)channels % importance_groups)
            : (importance != NULL || importance_groups != 0)) ||
        ptq_values_finite(source, columns * channels) != 0) return -1;
    const unsigned zero = 1u << (bits - 1u);
    const int high = (int)zero - 1, low = -(int)zero;
    const unsigned mask = (1u << bits) - 1u;
    uint64_t row_bits = (uint64_t)columns * bits;
    size_t row_bytes = (size_t)(row_bits / 8u + (row_bits % 8u != 0));
    if (row_bytes > SIZE_MAX / (size_t)channels ||
        (size_t)channels > SIZE_MAX / sizeof(float)) return -1;
    size_t source_bytes = (size_t)(columns * channels) * sizeof(float);
    size_t output_bytes = row_bytes * (size_t)channels;
    size_t scales_bytes = (size_t)channels * sizeof(float);
    if (ptq_ranges_overlap(source, source_bytes, output, output_bytes) ||
        ptq_ranges_overlap(source, source_bytes, scales, scales_bytes) ||
        ptq_ranges_overlap(output, output_bytes, scales, scales_bytes)) return -1;
    if (importance) {
        if ((uint64_t)columns > SIZE_MAX / sizeof(double) / importance_groups) return -1;
        size_t count = (size_t)columns * importance_groups;
        if (ptq_ranges_overlap(importance, count * sizeof(double), output, output_bytes) ||
            ptq_ranges_overlap(importance, count * sizeof(double), scales, scales_bytes)) return -1;
        for (size_t k = 0; k < count; k++)
            if (!ptq_finite_f64(importance[k]) || importance[k] < 0.0) return -1;
    }
    for (int32_t channel = 0; channel < channels; channel++) {
        const float* row = source + (size_t)channel * (size_t)columns;
        const double* energy = importance ? importance +
            ((size_t)channel / ((uint32_t)channels / importance_groups)) * (size_t)columns : NULL;
        float maximum = 0.0f;
        for (int64_t k = 0; k < columns; k++) {
            float magnitude = row[k] < 0.0f ? -row[k] : row[k];
            if (magnitude > maximum) maximum = magnitude;
        }
        float base;
        if (ptq_positive_f32(maximum == 0.0f ? 1.0 : (double)maximum / high, &base)) return -1;
        if (base > FLT_MAX / (float)zero) base = FLT_MAX / (float)zero;
        double best_error = __builtin_inf();
        float best_scale = base;
        int last = algorithm == VX_WEIGHT_ALGORITHM_RTN ? 100 : bits == 2 ? 1 : 50;
        for (int percent = 100; percent >= last; percent--) {
            float scale = base * ((float)percent / 100.0f);
            if (scale <= 0.0f) scale = FLT_MIN;
            double error = 0.0;
            for (int64_t k = 0; k < columns; k++) {
                int code = ptq_group_code(row[k], scale, low, high, NULL);
                float restored = (float)code * scale;
                double delta = (double)row[k] - restored;
                error += delta * delta * (energy ? energy[k] : 1.0);
            }
            if (error < best_error) { best_error = error; best_scale = scale; }
        }
        scales[channel] = best_scale;
        uint8_t* packed = output + (size_t)channel * row_bytes;
        for (size_t k = 0; k < row_bytes; k++) packed[k] = bits == 2 ? 0xaau : bits == 3 ? 0u : 0x88u;
        for (int64_t k = 0; k < columns; k++) {
            unsigned code = (unsigned)(ptq_group_code(row[k], best_scale, low, high, NULL) - low);
            size_t byte = (size_t)((uint64_t)k * bits / 8u);
            unsigned shift = (unsigned)((uint64_t)k * bits % 8u);
            packed[byte] = (uint8_t)((packed[byte] & ~(mask << shift)) | (code << shift));
            if (shift + bits > 8u)
                packed[byte + 1] = (uint8_t)((packed[byte + 1] & ~(mask >> (8u - shift))) | (code >> (8u - shift)));
        }
    }
    return 0;
}

int volvoxai_ptq_pack_conv_i4(const float* source, int32_t channels,
                              int64_t columns, int32_t algorithm,
                              uint8_t* output, float* scales) {
    return volvoxai_ptq_pack_conv_weight(source, channels, columns, 4u,
        algorithm, NULL, 0u, output, scales);
}

int volvoxai_training_quantize_grouped_weight(
        const float* source, uint32_t rows, uint32_t columns, int transpose,
        uint32_t bits, uint32_t group_size, int32_t algorithm, const double* importance,
        const float* preserve_scales, uint32_t* codes, float* scales,
        uint64_t* saturation_count) {
    size_t elements;
    if (!rows || !columns || (transpose != 0 && transpose != 1) ||
        (bits != 2 && bits != 3 && bits != 4 && bits != 8) || !vx_packed_group_size_supported((int)group_size) ||
        !source || !codes || !scales ||
        !ptq_product_fits_u32(rows, columns, &elements) ||
        elements > INT64_MAX || elements > SIZE_MAX / sizeof(float) ||
        ptq_values_finite(source, (int64_t)elements) != 0 ||
        (algorithm != VX_WEIGHT_ALGORITHM_RTN &&
         algorithm != VX_WEIGHT_ALGORITHM_MSE_CLIP &&
         algorithm != VX_WEIGHT_ALGORITHM_IMPORTANCE) ||
        (preserve_scales && algorithm != VX_WEIGHT_ALGORITHM_RTN) ||
        (algorithm == VX_WEIGHT_ALGORITHM_IMPORTANCE && !importance)) return 0;
    uint32_t groups = columns / group_size + (columns % group_size != 0);
    uint64_t row_bits = (uint64_t)columns * bits;
    uint32_t words = (uint32_t)(row_bits / 32u + (row_bits % 32u != 0));
    size_t word_count, scale_count;
    if (!ptq_product_fits_u32(rows, words, &word_count) ||
        !ptq_product_fits_u32(rows, groups, &scale_count) ||
        word_count > SIZE_MAX / sizeof(uint32_t) ||
        scale_count > SIZE_MAX / sizeof(float)) return 0;
    size_t source_bytes = elements * sizeof(float);
    size_t code_bytes = word_count * sizeof(uint32_t);
    size_t scale_bytes = scale_count * sizeof(float);
    if (ptq_ranges_overlap(source, source_bytes, codes, code_bytes) ||
        ptq_ranges_overlap(source, source_bytes, scales, scale_bytes) ||
        ptq_ranges_overlap(codes, code_bytes, scales, scale_bytes) ||
        (preserve_scales &&
         (ptq_ranges_overlap(preserve_scales, scale_bytes, codes, code_bytes) ||
          ptq_ranges_overlap(preserve_scales, scale_bytes, scales, scale_bytes)))) return 0;
    if (importance) {
        if ((uint64_t)columns > SIZE_MAX / sizeof(double)) return 0;
        size_t importance_bytes = (size_t)columns * sizeof(double);
        if (ptq_ranges_overlap(importance, importance_bytes, codes, code_bytes) ||
            ptq_ranges_overlap(importance, importance_bytes, scales, scale_bytes)) return 0;
    }
    if (importance) for (uint32_t c = 0; c < columns; ++c)
        if (!ptq_finite_f64(importance[c]) || importance[c] < 0.0) return 0;
    if (preserve_scales) for (size_t i = 0; i < scale_count; ++i)
        if (!ptq_finite_f32(preserve_scales[i]) || preserve_scales[i] <= 0.0f) return 0;
    const int high = (1 << (bits - 1u)) - 1, low = -high - 1;
    union { uint32_t bits; float value; } smallest = { 1u };
    for (size_t i = 0; i < word_count; ++i)
        codes[i] = bits == 2 ? UINT32_C(0xaaaaaaaa) : bits == 3 ? 0u :
            bits == 4 ? UINT32_C(0x88888888) : UINT32_C(0x80808080);
    uint64_t saturated = 0;
    for (uint32_t r = 0; r < rows; ++r) {
        for (uint32_t g = 0; g < groups; ++g) {
            uint32_t begin = g * group_size, length = columns - begin;
            if (length > group_size) length = group_size;
            float values[VX_PACKED_MAX_GROUP_SIZE], maximum = 0.0f;
            for (uint32_t j = 0; j < length; ++j) {
                size_t index = transpose ? (size_t)(begin + j) * rows + r
                                         : (size_t)r * columns + begin + j;
                float v = source[index];
                values[j] = v;
                float magnitude = v < 0.0f ? -v : v;
                if (magnitude > maximum) maximum = magnitude;
            }
            float scale = maximum == 0.0f ? 1.0f : maximum / (float)high;
            if (scale == 0.0f) scale = smallest.value;
            scale = ptq_finite_weight_scale(scale, high);
            if (preserve_scales) scale = preserve_scales[(size_t)r * groups + g];
            else if (algorithm != VX_WEIGHT_ALGORITHM_RTN && maximum > 0.0f) {
                float base = scale, best = scale;
                double best_error = __builtin_inf();
                for (int step = 0; step <= (bits == 2 ? 99 : bits == 3 ? 50 : 25); ++step) {
                    /* W2 must search below 2/3 maxabs to use -2 at all.
                     * Keep the existing I4/I8 grid/arithmetic unchanged. */
                    float candidate = base * (bits <= 3
                        ? (float)(100 - step) / 100.0f : 1.0f - step * 0.01f);
                    if (candidate == 0.0f) candidate = smallest.value;
                    double error = 0.0;
                    for (uint32_t j = 0; j < length; ++j) {
                        int q = ptq_group_code(values[j], candidate, low, high, NULL);
                        float reconstructed = (float)q * candidate;
                        if (!ptq_finite_f32(reconstructed)) {
                            error = __builtin_inf();
                            break;
                        }
                        double diff = (double)values[j] - (double)reconstructed;
                        double weight = algorithm == VX_WEIGHT_ALGORITHM_IMPORTANCE
                            ? importance[begin + j] : 1.0;
                        error += diff * diff * weight;
                    }
                    if (error < best_error) { best_error = error; best = candidate; }
                }
                scale = best;
            }
            scales[(size_t)r * groups + g] = scale;
            for (uint32_t j = 0; j < length; ++j) {
                uint32_t k = begin + j;
                uint64_t bit = (uint64_t)k * bits;
                uint32_t shift = (uint32_t)(bit % 32u);
                int q = ptq_group_code(values[j], scale, low, high, &saturated);
                uint32_t mask = ((1u << bits) - 1u) << shift;
                uint32_t* word = &codes[(size_t)r * words + (size_t)(bit / 32u)];
                *word = (*word & ~mask) | ((uint32_t)(q - low) << shift);
                if (shift + bits > 32u) {
                    uint32_t high_mask = ((1u << bits) - 1u) >> (32u - shift);
                    word[1] = (word[1] & ~high_mask) | ((uint32_t)(q - low) >> (32u - shift));
                }
            }
        }
    }
    if (saturation_count) *saturation_count = saturated;
    return 1;
}
