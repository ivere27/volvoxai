#ifndef VOLVOXAI_BATCH_MATMUL_F32_PLAN_H
#define VOLVOXAI_BATCH_MATMUL_F32_PLAN_H

#include <stddef.h>
#include <stdint.h>

typedef struct {
    uint32_t metadata[23];
    uint32_t metadata_words;
    uint32_t batch_rank;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t output_batches;
    uint32_t a_elements;
    uint32_t b_elements;
    uint32_t output_elements;
    size_t a_bytes;
    size_t b_bytes;
    size_t output_bytes;
} VxBatchMatMulF32Plan;

static inline int vx_batch_matmul_f32_shape_strides(
        const int* shape, int rank, uint32_t strides[8],
        uint32_t* elements_out, size_t* bytes_out) {
    uint64_t stride = 1u;
    uint64_t maximum_elements = UINT32_MAX;
    if (!shape || !strides || !elements_out || !bytes_out ||
        rank <= 0 || rank > 8)
        return 0;
    if ((uint64_t)SIZE_MAX / sizeof(float) < maximum_elements)
        maximum_elements = (uint64_t)SIZE_MAX / sizeof(float);
    for (int dimension = 0; dimension < 8; dimension++)
        strides[dimension] = 0u;
    for (int dimension = rank - 1; dimension >= 0; dimension--) {
        uint32_t extent;
        if (shape[dimension] <= 0 || stride > maximum_elements) return 0;
        extent = (uint32_t)shape[dimension];
        strides[dimension] = (uint32_t)stride;
        if ((uint64_t)extent > maximum_elements / stride) return 0;
        stride *= extent;
    }
    *elements_out = (uint32_t)stride;
    *bytes_out = (size_t)stride * sizeof(float);
    return 1;
}

static inline int vx_batch_matmul_f32_ranges_overlap(
        const void* left, size_t left_bytes,
        const void* right, size_t right_bytes) {
    uintptr_t left_start;
    uintptr_t right_start;
    if (!left || !right || !left_bytes || !right_bytes) return 0;
    left_start = (uintptr_t)left;
    right_start = (uintptr_t)right;
    if (left_start > UINTPTR_MAX - left_bytes ||
        right_start > UINTPTR_MAX - right_bytes)
        return 1;
    return left_start < right_start + right_bytes &&
        right_start < left_start + left_bytes;
}

static inline int vx_batch_matmul_f32_plan(
        const float* a, const int* a_shape, int a_rank,
        const float* b, const int* b_shape, int b_rank,
        float* output, const int* output_shape, int output_rank,
        VxBatchMatMulF32Plan* plan_out) {
    VxBatchMatMulF32Plan plan = {0};
    uint32_t a_strides[8];
    uint32_t b_strides[8];
    uint32_t output_strides[8];
    uint64_t matrix_elements;
    int a_leading;
    int b_leading;
    if (!a || !b || !output || !plan_out ||
        a_rank < 2 || a_rank > 8 || b_rank < 2 || b_rank > 8 ||
        output_rank < 2 || output_rank > 8 ||
        output_rank != (a_rank > b_rank ? a_rank : b_rank) ||
        !vx_batch_matmul_f32_shape_strides(
            a_shape, a_rank, a_strides,
            &plan.a_elements, &plan.a_bytes) ||
        !vx_batch_matmul_f32_shape_strides(
            b_shape, b_rank, b_strides,
            &plan.b_elements, &plan.b_bytes) ||
        !vx_batch_matmul_f32_shape_strides(
            output_shape, output_rank, output_strides,
            &plan.output_elements, &plan.output_bytes))
        return 0;
    plan.batch_rank = (uint32_t)output_rank - 2u;
    plan.m = (uint32_t)a_shape[a_rank - 2];
    plan.k = (uint32_t)a_shape[a_rank - 1];
    plan.n = (uint32_t)b_shape[b_rank - 1];
    matrix_elements = (uint64_t)plan.m * plan.n;
    if (a_shape[a_rank - 1] != b_shape[b_rank - 2] ||
        output_shape[output_rank - 2] != (int)plan.m ||
        output_shape[output_rank - 1] != (int)plan.n ||
        !matrix_elements || matrix_elements > plan.output_elements ||
        plan.output_elements % (uint32_t)matrix_elements != 0u ||
        (uint64_t)plan.m * plan.k > plan.a_elements ||
        (uint64_t)plan.k * plan.n > plan.b_elements)
        return 0;
    plan.output_batches =
        plan.output_elements / (uint32_t)matrix_elements;
    a_leading = output_rank - a_rank;
    b_leading = output_rank - b_rank;
    plan.metadata[0] = plan.batch_rank;
    plan.metadata[1] = plan.m;
    plan.metadata[2] = plan.k;
    plan.metadata[3] = plan.n;
    plan.metadata[4] = plan.output_batches;
    for (uint32_t dimension = 0; dimension < plan.batch_rank; dimension++) {
        int a_dimension = (int)dimension - a_leading;
        int b_dimension = (int)dimension - b_leading;
        int a_extent = a_dimension >= 0 ? a_shape[a_dimension] : 1;
        int b_extent = b_dimension >= 0 ? b_shape[b_dimension] : 1;
        int expected_extent = a_extent > b_extent ? a_extent : b_extent;
        if ((a_extent != 1 && b_extent != 1 && a_extent != b_extent) ||
            output_shape[dimension] != expected_extent)
            return 0;
        plan.metadata[5u + dimension] =
            output_strides[dimension] / (uint32_t)matrix_elements;
        if (a_dimension >= 0 && a_extent != 1)
            plan.metadata[5u + plan.batch_rank + dimension] =
                a_strides[a_dimension];
        if (b_dimension >= 0 && b_extent != 1)
            plan.metadata[5u + 2u * plan.batch_rank + dimension] =
                b_strides[b_dimension];
    }
    plan.metadata_words = 5u + 3u * plan.batch_rank;
    if (!plan.output_batches || plan.metadata_words > 23u ||
        vx_batch_matmul_f32_ranges_overlap(
            output, plan.output_bytes, a, plan.a_bytes) ||
        vx_batch_matmul_f32_ranges_overlap(
            output, plan.output_bytes, b, plan.b_bytes))
        return 0;
    *plan_out = plan;
    return 1;
}

#endif
