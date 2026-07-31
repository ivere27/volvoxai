#ifndef VOLVOXAI_TYPED_CONTROL_PLAN_H
#define VOLVOXAI_TYPED_CONTROL_PLAN_H

#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "../../include/volvoxai_enums.h"

enum {
    VX_TYPED_CONTROL_EQUAL_I32 = 0,
    VX_TYPED_CONTROL_GREATER_OR_EQUAL_I32 = 1,
    VX_TYPED_CONTROL_NOT_I32 = 2,
    VX_TYPED_CONTROL_CLIP_I32 = 3,
    VX_TYPED_CONTROL_COPY_32 = 4,
    VX_TYPED_CONTROL_CAST_32 = 5,
};

typedef struct {
    /* Mirrors shaders/inference/typedControl32Native.wgsl exactly. */
    uint32_t values[32];
} VxTypedControlMetadata;

_Static_assert(sizeof(VxTypedControlMetadata) == 128,
               "typedControl32Native storage ABI");

static inline int vx_typed_control_checked_elements(
        long elements, uint32_t* count, size_t* bytes) {
    if (!count || !bytes || elements <= 0 ||
        (uint64_t)elements > UINT32_MAX ||
        (uint64_t)elements > SIZE_MAX / sizeof(uint32_t))
        return 0;
    *count = (uint32_t)elements;
    *bytes = (size_t)elements * sizeof(uint32_t);
    return 1;
}

static inline int vx_typed_control_ranges_overlap(
        const void* left, size_t left_bytes,
        const void* right, size_t right_bytes) {
    uintptr_t left_begin;
    uintptr_t right_begin;
    if (!left || !right || left_bytes == 0 || right_bytes == 0) return 0;
    left_begin = (uintptr_t)left;
    right_begin = (uintptr_t)right;
    if (left_begin > UINTPTR_MAX - left_bytes ||
        right_begin > UINTPTR_MAX - right_bytes)
        return 1;
    return left_begin < right_begin + right_bytes &&
           right_begin < left_begin + left_bytes;
}

static inline void vx_typed_control_metadata_init(
        VxTypedControlMetadata* metadata, uint32_t elements,
        uint32_t operation, uint32_t input_dtype,
        uint32_t output_dtype, int32_t minimum, int32_t maximum) {
    memset(metadata, 0, sizeof(*metadata));
    metadata->values[0] = elements;
    metadata->values[2] = operation;
    metadata->values[3] = input_dtype;
    metadata->values[4] = output_dtype;
    metadata->values[5] = (uint32_t)minimum;
    metadata->values[6] = (uint32_t)maximum;
}

static inline int vx_typed_control_compare_plan(
        const int32_t* a, long a_elements,
        const int32_t* b, long b_elements,
        int32_t* output, long output_elements,
        const uint32_t* output_strides,
        const uint32_t* a_strides,
        const uint32_t* b_strides,
        int rank, int operation,
        VxTypedControlMetadata* metadata,
        size_t* a_bytes, size_t* b_bytes, size_t* output_bytes) {
    uint32_t a_count;
    uint32_t b_count;
    uint32_t output_count;
    uint32_t dimensions[8] = {0};
    uint64_t a_max_index = 0;
    uint64_t b_max_index = 0;
    if (!a || !b || !output || !output_strides ||
        !a_strides || !b_strides || !metadata ||
        !a_bytes || !b_bytes || !output_bytes ||
        rank <= 0 || rank > 8 ||
        (operation != VX_TYPED_CONTROL_EQUAL_I32 &&
         operation != VX_TYPED_CONTROL_GREATER_OR_EQUAL_I32) ||
        !vx_typed_control_checked_elements(
            a_elements, &a_count, a_bytes) ||
        !vx_typed_control_checked_elements(
            b_elements, &b_count, b_bytes) ||
        !vx_typed_control_checked_elements(
            output_elements, &output_count, output_bytes) ||
        vx_typed_control_ranges_overlap(
            output, *output_bytes, a, *a_bytes) ||
        vx_typed_control_ranges_overlap(
            output, *output_bytes, b, *b_bytes) ||
        output_strides[0] == 0u ||
        output_strides[rank - 1] != 1u ||
        output_count % output_strides[0] != 0u)
        return 0;

    dimensions[0] = output_count / output_strides[0];
    if (dimensions[0] == 0u) return 0;
    for (int dimension = 1; dimension < rank; dimension++) {
        if (output_strides[dimension] == 0u ||
            output_strides[dimension - 1] %
                output_strides[dimension] != 0u)
            return 0;
        dimensions[dimension] =
            output_strides[dimension - 1] /
            output_strides[dimension];
        if (dimensions[dimension] == 0u) return 0;
    }
    for (int dimension = 0; dimension < rank; dimension++) {
        a_max_index +=
            (uint64_t)(dimensions[dimension] - 1u) *
            a_strides[dimension];
        b_max_index +=
            (uint64_t)(dimensions[dimension] - 1u) *
            b_strides[dimension];
    }
    if (a_max_index >= a_count || b_max_index >= b_count) return 0;

    vx_typed_control_metadata_init(
        metadata, output_count, (uint32_t)operation,
        VX_DTYPE_I32, VX_DTYPE_I32, 0, 0);
    metadata->values[1] = (uint32_t)rank;
    memcpy(metadata->values + 8, output_strides,
           (size_t)rank * sizeof(uint32_t));
    memcpy(metadata->values + 16, a_strides,
           (size_t)rank * sizeof(uint32_t));
    memcpy(metadata->values + 24, b_strides,
           (size_t)rank * sizeof(uint32_t));
    return 1;
}

static inline int vx_typed_control_unary_plan(
        const int32_t* input, int32_t* output, long elements,
        int operation, int32_t minimum, int32_t maximum,
        VxTypedControlMetadata* metadata, size_t* bytes) {
    uint32_t count;
    if (!input || !output || !metadata || !bytes ||
        (operation != VX_TYPED_CONTROL_NOT_I32 &&
         operation != VX_TYPED_CONTROL_CLIP_I32) ||
        (operation == VX_TYPED_CONTROL_CLIP_I32 && minimum > maximum) ||
        !vx_typed_control_checked_elements(elements, &count, bytes) ||
        vx_typed_control_ranges_overlap(output, *bytes, input, *bytes))
        return 0;
    vx_typed_control_metadata_init(
        metadata, count, (uint32_t)operation,
        VX_DTYPE_I32, VX_DTYPE_I32, minimum, maximum);
    return 1;
}

static inline int vx_typed_control_copy_plan(
        const void* input, void* output, long elements,
        VxTypedControlMetadata* metadata, size_t* bytes) {
    uint32_t count;
    if (!input || !output || !metadata || !bytes ||
        !vx_typed_control_checked_elements(elements, &count, bytes) ||
        (input != output &&
         vx_typed_control_ranges_overlap(output, *bytes, input, *bytes)))
        return 0;
    vx_typed_control_metadata_init(
        metadata, count, VX_TYPED_CONTROL_COPY_32,
        VX_DTYPE_UNSPECIFIED, VX_DTYPE_UNSPECIFIED, 0, 0);
    return 1;
}

static inline int vx_typed_control_cast_plan(
        const void* input, int input_dtype,
        void* output, int output_dtype, long elements,
        VxTypedControlMetadata* metadata, size_t* bytes) {
    uint32_t count;
    if (!input || !output || !metadata || !bytes ||
        (input_dtype != VX_DTYPE_F32 && input_dtype != VX_DTYPE_I32) ||
        (output_dtype != VX_DTYPE_F32 && output_dtype != VX_DTYPE_I32) ||
        !vx_typed_control_checked_elements(elements, &count, bytes) ||
        vx_typed_control_ranges_overlap(output, *bytes, input, *bytes))
        return 0;
    vx_typed_control_metadata_init(
        metadata, count, VX_TYPED_CONTROL_CAST_32,
        (uint32_t)input_dtype, (uint32_t)output_dtype, 0, 0);
    return 1;
}

static inline int vx_typed_control_where_plan(
        const int32_t* condition, const void* a, const void* b,
        void* output, long elements, uint32_t* count, size_t* bytes) {
    if (!condition || !a || !b || !output || !count || !bytes ||
        !vx_typed_control_checked_elements(elements, count, bytes) ||
        vx_typed_control_ranges_overlap(output, *bytes, condition, *bytes) ||
        vx_typed_control_ranges_overlap(output, *bytes, a, *bytes) ||
        vx_typed_control_ranges_overlap(output, *bytes, b, *bytes))
        return 0;
    return 1;
}

static inline int vx_typed_control_argmax_plan(
        const float* input, int32_t* output,
        uint32_t outer, uint32_t axis_size, uint32_t inner,
        uint32_t* input_elements, uint32_t* output_elements,
        size_t* input_bytes, size_t* output_bytes) {
    uint64_t input_count = (uint64_t)outer * axis_size * inner;
    uint64_t output_count = (uint64_t)outer * inner;
    if (!input || !output || !input_elements || !output_elements ||
        !input_bytes || !output_bytes ||
        outer == 0u || axis_size == 0u || inner == 0u ||
        input_count > UINT32_MAX || output_count > UINT32_MAX ||
        input_count > SIZE_MAX / sizeof(uint32_t) ||
        output_count > SIZE_MAX / sizeof(uint32_t))
        return 0;
    *input_elements = (uint32_t)input_count;
    *output_elements = (uint32_t)output_count;
    *input_bytes = (size_t)input_count * sizeof(uint32_t);
    *output_bytes = (size_t)output_count * sizeof(uint32_t);
    if (vx_typed_control_ranges_overlap(
            output, *output_bytes, input, *input_bytes))
        return 0;
    return 1;
}

static inline int vx_typed_control_concat_plan(
        const void* const* inputs, const long* sizes,
        const int* input_axes, int count, void* output,
        int output_axis, int inner,
        uint32_t* output_elements, size_t* output_bytes) {
    uint64_t total = 0u;
    uint64_t summed_axis = 0u;
    uint64_t common_outer = 0u;
    uint64_t output_row_elements;
    if (!inputs || !sizes || !input_axes || !output ||
        !output_elements || !output_bytes ||
        count <= 0 || output_axis <= 0 || inner <= 0)
        return 0;
    for (int index = 0; index < count; index++) {
        uint64_t denominator;
        uint64_t outer;
        if (!inputs[index] || sizes[index] <= 0 ||
            input_axes[index] <= 0)
            return 0;
        denominator =
            (uint64_t)(uint32_t)input_axes[index] * (uint32_t)inner;
        if (denominator == 0u ||
            (uint64_t)sizes[index] % denominator != 0u)
            return 0;
        outer = (uint64_t)sizes[index] / denominator;
        if (outer == 0u || (index != 0 && outer != common_outer))
            return 0;
        common_outer = outer;
        total += (uint64_t)sizes[index];
        summed_axis += (uint32_t)input_axes[index];
        if (total > UINT32_MAX || summed_axis > UINT32_MAX)
            return 0;
    }
    output_row_elements =
        (uint64_t)(uint32_t)output_axis * (uint32_t)inner;
    if (summed_axis != (uint32_t)output_axis ||
        output_row_elements == 0u ||
        common_outer > UINT64_MAX / output_row_elements ||
        total != common_outer * output_row_elements ||
        total > SIZE_MAX / sizeof(uint32_t))
        return 0;
    *output_elements = (uint32_t)total;
    *output_bytes = (size_t)total * sizeof(uint32_t);
    for (int index = 0; index < count; index++) {
        size_t input_bytes =
            (size_t)sizes[index] * sizeof(uint32_t);
        if (vx_typed_control_ranges_overlap(
                output, *output_bytes, inputs[index], input_bytes))
            return 0;
    }
    return 1;
}

#endif
