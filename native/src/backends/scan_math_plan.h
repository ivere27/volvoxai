#ifndef VOLVOXAI_SCAN_MATH_PLAN_H
#define VOLVOXAI_SCAN_MATH_PLAN_H

#include "typed_control_plan.h"

/* Shared checked geometry for the WGSL unaryMath and cumSum kernels. */
typedef struct {
    uint32_t params[8];
    uint32_t grid[3];
    size_t bytes;
} VxScanMathPlan;

static inline int vx_scan_math_plan_base(
        const void* input, void* output, long elements,
        VxScanMathPlan* plan) {
    uint32_t count;
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (!input || !output ||
        !vx_typed_control_checked_elements(elements, &count, &plan->bytes) ||
        count > UINT32_MAX - 63u ||
        vx_typed_control_ranges_overlap(input, plan->bytes, output, plan->bytes))
        return 0;
    return 1;
}

static inline void vx_scan_math_grid(uint32_t work, uint32_t grid[3]) {
    uint32_t groups = (work + 63u) / 64u;
    grid[0] = groups < 65535u ? groups : 65535u;
    grid[1] = (groups + grid[0] - 1u) / grid[0];
    grid[2] = 1u;
}

/* operation: Exp/Sin/Cos/Sqrt/Log/Erf/Floor/Ceil/Round/Atan/Softplus/Neg. */
static inline int vx_unary_math_plan(
        const float* input, float* output, long elements, int operation,
        VxScanMathPlan* plan) {
    if (operation < 0 || operation > 11 ||
        !vx_scan_math_plan_base(input, output, elements, plan)) return 0;
    plan->params[0] = (uint32_t)elements;
    plan->params[1] = (uint32_t)operation;
    vx_scan_math_grid((uint32_t)elements, plan->grid);
    return 1;
}

static inline int vx_cumsum_plan(
        const void* input, void* output, long elements,
        const int* shape, int rank, int axis, int exclusive, int reverse,
        int dtype, VxScanMathPlan* plan) {
    uint64_t count = 1u, inner = 1u;
    if (!shape || rank < 1 || rank > 8 ||
        (dtype != VX_DTYPE_F32 && dtype != VX_DTYPE_I32) ||
        (exclusive != 0 && exclusive != 1) || (reverse != 0 && reverse != 1) ||
        !vx_scan_math_plan_base(input, output, elements, plan)) return 0;
    if (axis < 0) axis += rank;
    if (axis < 0 || axis >= rank) return 0;
    for (int i = 0; i < rank; ++i) {
        if (shape[i] <= 0 || count > (uint64_t)elements / (uint32_t)shape[i]) return 0;
        count *= (uint32_t)shape[i];
        if (i > axis) inner *= (uint32_t)shape[i];
    }
    if (count != (uint64_t)elements) return 0;
    plan->params[0] = (uint32_t)(count / (uint32_t)shape[axis]);
    plan->params[1] = (uint32_t)shape[axis];
    plan->params[2] = (uint32_t)inner;
    plan->params[3] = (uint32_t)exclusive;
    plan->params[4] = (uint32_t)reverse;
    plan->params[5] = dtype == VX_DTYPE_I32 ? 1u : 0u;
    vx_scan_math_grid(plan->params[0], plan->grid);
    return 1;
}

#endif
