#ifndef VOLVOXAI_QBATCH_MATMUL_PLAN_H
#define VOLVOXAI_QBATCH_MATMUL_PLAN_H

#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "../../include/volvoxai_enums.h"

typedef struct {
    uint32_t output_batch_strides[8];
    uint32_t a_batch_strides[8];
    uint32_t b_batch_strides[8];
    uint32_t batch_rank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t a_elements;
    uint32_t b_elements;
    uint32_t output_elements;
    size_t a_bytes;
    size_t b_bytes;
    size_t output_bytes;
} VxQBatchMatMulDevicePlan;

static int vx_qbatch_shape_strides(const int* shape, int rank,
                                   uint32_t strides[8],
                                   uint32_t* elements_out) {
    uint64_t stride = 1u;
    if (!shape || !strides || !elements_out || rank <= 0 || rank > 8)
        return 0;
    for (int dimension = 0; dimension < 8; dimension++)
        strides[dimension] = 0u;
    for (int dimension = rank - 1; dimension >= 0; dimension--) {
        if (shape[dimension] <= 0 || stride > UINT32_MAX) return 0;
        strides[dimension] = (uint32_t)stride;
        stride *= (uint32_t)shape[dimension];
        if (stride > UINT32_MAX) return 0;
    }
    *elements_out = (uint32_t)stride;
    return 1;
}

static int vx_qbatch_qdesc_valid(float scale, int32_t zero_point,
                                 uint32_t dtype) {
    if (!isfinite(scale) || scale <= 0.0f) return 0;
    if (dtype == VX_DTYPE_I8)
        return zero_point >= -128 && zero_point <= 127;
    if (dtype == VX_DTYPE_U8)
        return zero_point >= 0 && zero_point <= 255;
    return 0;
}

static uint64_t vx_qbatch_centered_magnitude(uint32_t dtype,
                                             int32_t zero_point) {
    int64_t low =
        (int64_t)(dtype == VX_DTYPE_I8 ? -128 : 0) - zero_point;
    int64_t high =
        (int64_t)(dtype == VX_DTYPE_I8 ? 127 : 255) - zero_point;
    uint64_t low_magnitude = (uint64_t)(low < 0 ? -low : low);
    uint64_t high_magnitude = (uint64_t)(high < 0 ? -high : high);
    return low_magnitude > high_magnitude
        ? low_magnitude : high_magnitude;
}

static int vx_qbatch_ranges_overlap(const void* left, size_t left_bytes,
                                    const void* right, size_t right_bytes) {
    uintptr_t left_start = (uintptr_t)left;
    uintptr_t right_start = (uintptr_t)right;
    if (!left || !right || !left_bytes || !right_bytes) return 0;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes)
        return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

static int vx_qbatch_matmul_device_plan(
        const void* a, const int* a_shape, int a_rank,
        float a_scale, int32_t a_zero_point, uint32_t a_dtype,
        const void* b, const int* b_shape, int b_rank,
        float b_scale, int32_t b_zero_point, uint32_t b_dtype,
        void* output, const int* output_shape, int output_rank,
        float output_scale, int32_t output_zero_point,
        uint32_t output_dtype, VxQBatchMatMulDevicePlan* plan_out) {
    VxQBatchMatMulDevicePlan plan = {0};
    uint32_t a_strides[8];
    uint32_t b_strides[8];
    uint32_t output_strides[8];
    uint64_t output_matrix_elements;
    uint64_t accumulator_bound;
    float multiplier;
    int a_leading;
    int b_leading;
    if (!a || !b || !output || !plan_out ||
        a_rank < 2 || a_rank > 8 || b_rank < 2 || b_rank > 8 ||
        output_rank < 2 || output_rank > 8 ||
        output_rank != (a_rank > b_rank ? a_rank : b_rank) ||
        !vx_qbatch_qdesc_valid(a_scale, a_zero_point, a_dtype) ||
        !vx_qbatch_qdesc_valid(b_scale, b_zero_point, b_dtype) ||
        !vx_qbatch_qdesc_valid(
            output_scale, output_zero_point, output_dtype) ||
        !vx_qbatch_shape_strides(
            a_shape, a_rank, a_strides, &plan.a_elements) ||
        !vx_qbatch_shape_strides(
            b_shape, b_rank, b_strides, &plan.b_elements) ||
        !vx_qbatch_shape_strides(
            output_shape, output_rank, output_strides,
            &plan.output_elements))
        return 0;
    plan.batch_rank = (uint32_t)output_rank - 2u;
    plan.m = (uint32_t)a_shape[a_rank - 2];
    plan.k = (uint32_t)a_shape[a_rank - 1];
    plan.n = (uint32_t)b_shape[b_rank - 1];
    output_matrix_elements = (uint64_t)plan.m * plan.n;
    if (a_shape[a_rank - 1] != b_shape[b_rank - 2] ||
        output_shape[output_rank - 2] != (int)plan.m ||
        output_shape[output_rank - 1] != (int)plan.n ||
        output_matrix_elements == 0u ||
        output_matrix_elements > UINT32_MAX)
        return 0;
    a_leading = output_rank - a_rank;
    b_leading = output_rank - b_rank;
    for (uint32_t dimension = 0; dimension < plan.batch_rank; dimension++) {
        int a_dimension = (int)dimension - a_leading;
        int b_dimension = (int)dimension - b_leading;
        int a_extent = a_dimension >= 0 ? a_shape[a_dimension] : 1;
        int b_extent = b_dimension >= 0 ? b_shape[b_dimension] : 1;
        int expected_extent =
            a_extent > b_extent ? a_extent : b_extent;
        if ((a_extent != 1 && b_extent != 1 && a_extent != b_extent) ||
            output_shape[dimension] != expected_extent)
            return 0;
        plan.output_batch_strides[dimension] =
            output_strides[dimension] / (uint32_t)output_matrix_elements;
        if (a_dimension >= 0 && a_extent != 1)
            plan.a_batch_strides[dimension] = a_strides[a_dimension];
        if (b_dimension >= 0 && b_extent != 1)
            plan.b_batch_strides[dimension] = b_strides[b_dimension];
    }
    accumulator_bound =
        vx_qbatch_centered_magnitude(a_dtype, a_zero_point) *
        vx_qbatch_centered_magnitude(b_dtype, b_zero_point) *
        (uint64_t)plan.k;
    multiplier = (a_scale * b_scale) / output_scale;
    plan.a_bytes = plan.a_elements;
    plan.b_bytes = plan.b_elements;
    plan.output_bytes = plan.output_elements;
    if ((uint64_t)plan.m * plan.k > plan.a_elements ||
        (uint64_t)plan.k * plan.n > plan.b_elements ||
        output_matrix_elements > plan.output_elements ||
        accumulator_bound > INT32_MAX ||
        !isfinite(multiplier) || multiplier <= 0.0f ||
        vx_qbatch_ranges_overlap(
            output, plan.output_bytes, a, plan.a_bytes) ||
        vx_qbatch_ranges_overlap(
            output, plan.output_bytes, b, plan.b_bytes))
        return 0;
    *plan_out = plan;
    return 1;
}

#endif
