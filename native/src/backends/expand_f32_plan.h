#ifndef VOLVOXAI_EXPAND_F32_PLAN_H
#define VOLVOXAI_EXPAND_F32_PLAN_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t input_elements;
    uint32_t output_elements;
    size_t input_bytes;
    size_t output_bytes;
    /* Mirrors shaders/inference/expand.wgsl exactly. */
    uint32_t params[20];
} VxExpandF32Plan;

static inline int vx_expand_f32_ranges_overlap(
        const void* left, size_t left_bytes,
        const void* right, size_t right_bytes) {
    uintptr_t left_begin;
    uintptr_t right_begin;
    if (!left || !right || left_bytes == 0u || right_bytes == 0u) return 0;
    left_begin = (uintptr_t)left;
    right_begin = (uintptr_t)right;
    if (left_begin > UINTPTR_MAX - left_bytes ||
        right_begin > UINTPTR_MAX - right_bytes)
        return 1;
    return left_begin < right_begin + right_bytes &&
        right_begin < left_begin + left_bytes;
}

static inline int vx_expand_f32_shape_size(
        const int* shape, int rank, uint32_t* elements, size_t* bytes) {
    uint64_t product = 1u;
    uint64_t max_elements = (uint64_t)UINT32_MAX / sizeof(float);
    if (!shape || !elements || !bytes || rank <= 0 || rank > 8)
        return 0;
    if ((uint64_t)SIZE_MAX / sizeof(float) < max_elements)
        max_elements = (uint64_t)SIZE_MAX / sizeof(float);
    for (int dimension = 0; dimension < rank; dimension++) {
        uint32_t extent;
        if (shape[dimension] <= 0) return 0;
        extent = (uint32_t)shape[dimension];
        if (product > max_elements / extent) return 0;
        product *= extent;
    }
    *elements = (uint32_t)product;
    *bytes = (size_t)product * sizeof(float);
    return 1;
}

static inline int vx_expand_f32_plan(
        const float* input, float* output,
        const int* input_shape, int input_rank,
        const int* output_shape, int output_rank,
        VxExpandF32Plan* plan_out) {
    VxExpandF32Plan plan = {0};
    int offset;
    if (!input || !output || !input_shape || !output_shape || !plan_out ||
        input_rank <= 0 || input_rank > 8 ||
        output_rank <= 0 || output_rank > 8 ||
        input_rank > output_rank ||
        !vx_expand_f32_shape_size(
            input_shape, input_rank, &plan.input_elements,
            &plan.input_bytes) ||
        !vx_expand_f32_shape_size(
            output_shape, output_rank, &plan.output_elements,
            &plan.output_bytes) ||
        vx_expand_f32_ranges_overlap(
            output, plan.output_bytes, input, plan.input_bytes))
        return 0;
    offset = output_rank - input_rank;
    for (int dimension = 0; dimension < input_rank; dimension++) {
        int input_extent = input_shape[dimension];
        int output_extent = output_shape[offset + dimension];
        if (input_extent != 1 && input_extent != output_extent) return 0;
    }
    plan.params[0] = (uint32_t)input_rank;
    plan.params[1] = (uint32_t)output_rank;
    plan.params[3] = plan.output_elements;
    for (int dimension = 0; dimension < input_rank; dimension++)
        plan.params[4 + dimension] = (uint32_t)input_shape[dimension];
    for (int dimension = 0; dimension < output_rank; dimension++)
        plan.params[12 + dimension] = (uint32_t)output_shape[dimension];
    *plan_out = plan;
    return 1;
}

#endif
