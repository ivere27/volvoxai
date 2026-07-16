#include "volvoxai_training.h"
#include "../kernels/training_kernels.h"

#include <float.h>
#include <limits.h>
#include <stdint.h>

#if !defined(__wasm__)
#include <math.h>
#endif

#if defined(__wasm__)
#define VX_PTQ_EXPORT(name) __attribute__((export_name(name)))
#else
#define VX_PTQ_EXPORT(name)
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

VX_PTQ_EXPORT("volvoxai_ptq_abi_version")
uint32_t volvoxai_ptq_abi_version(void) {
    return VOLVOXAI_PTQ_ABI_VERSION;
}

VX_PTQ_EXPORT("volvoxai_ptq_capabilities")
uint32_t volvoxai_ptq_capabilities(void) {
    return VOLVOXAI_PTQ_CAP_ALL;
}

VX_PTQ_EXPORT("volvoxai_ptq_observer_reset")
void volvoxai_ptq_observer_reset(volvoxai_ptq_observer_t* observer) {
    if (!observer) return;
    observer->minimum = __builtin_inff();
    observer->maximum = -__builtin_inff();
    observer->sample_count = 0;
}

VX_PTQ_EXPORT("volvoxai_ptq_observer_observe_f32")
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

VX_PTQ_EXPORT("volvoxai_ptq_calculate_params")
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
    if (scheme == VOLVOXAI_PTQ_SYMMETRIC) {
        double minimum_magnitude = ptq_abs_f64((double)observer->minimum);
        double maximum_magnitude = ptq_abs_f64((double)observer->maximum);
        double magnitude = minimum_magnitude > maximum_magnitude
            ? minimum_magnitude : maximum_magnitude;
        if (ptq_positive_f32(magnitude == 0.0 ? 1.0 : magnitude / 127.0, &scale) != 0) return -1;
        zero_point = dtype == VOLVOXAI_DTYPE_I8 ? 0 : 128;
    } else if (scheme == VOLVOXAI_PTQ_ASYMMETRIC) {
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

VX_PTQ_EXPORT("volvoxai_ptq_quantize_f32")
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
                                        uint64_t* saturation_count) {
    double rounded = ptq_round_ties_even((double)value / (double)scale);
    if (rounded < -127.0) {
        rounded = -127.0;
        (*saturation_count)++;
    } else if (rounded > 127.0) {
        rounded = 127.0;
        (*saturation_count)++;
    }
    return (int8_t)rounded;
}

VX_PTQ_EXPORT("volvoxai_ptq_pack_weight_i8")
int volvoxai_ptq_pack_weight_i8(const float* values, const int32_t* shape,
                                int32_t ndim, int32_t axis, int8_t* output,
                                float* scales, int32_t scale_count,
                                uint64_t* saturation_count) {
    int64_t elements = 0;
    int64_t inner = 0;
    int32_t resolved_axis = 0;
    int32_t channels = 0;
    if (ptq_shape_info(shape, ndim, axis, &elements, &resolved_axis, &channels, &inner) != 0 ||
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
        if (ptq_positive_f32(magnitude == 0.0 ? 1.0 : magnitude / 127.0,
                             &scales[channel]) != 0) return -1;
    }
    uint64_t saturated = 0;
    for (int64_t index = 0; index < elements; index++) {
        int32_t channel = (int32_t)((index / inner) % channels);
        output[index] = ptq_quantize_symmetric_i8(values[index], scales[channel],
                                                   &saturated);
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
                source[source_index], scales[row], &saturated);
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

VX_PTQ_EXPORT("volvoxai_ptq_pack_bias_i32")
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
