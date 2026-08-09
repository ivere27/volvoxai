#include "thread_pool.h"

#include <stdatomic.h>
#include <stdio.h>

enum { VX_TEST_TASKS = 97 };

typedef struct {
    int total;
    atomic_int invalid_range;
    atomic_uint visits[VX_TEST_TASKS];
} VxParallelForTestContext;

static void vx_test_range_worker(void* opaque, int begin, int end) {
    VxParallelForTestContext* context =
        (VxParallelForTestContext*)opaque;
    if (begin < 0 || begin >= end || end > context->total) {
        atomic_store_explicit(
            &context->invalid_range, 1, memory_order_relaxed);
        return;
    }
    for (int task = begin; task < end; task++) {
        atomic_fetch_add_explicit(
            &context->visits[task], 1u, memory_order_relaxed);
    }
}

static int vx_test_exact_coverage(VxKernelThreadPool* pool, int grain) {
    VxParallelForTestContext context = {0};
    VxKernelThreadPoolScope scope;
    context.total = VX_TEST_TASKS;
    if (pool) scope = vx_kernel_thread_pool_scope_enter(pool);
    vx_kernels_parallel_for(
        context.total, grain, vx_test_range_worker, &context);
    if (pool) vx_kernel_thread_pool_scope_leave(scope);
    if (atomic_load_explicit(
            &context.invalid_range, memory_order_relaxed) != 0) return 0;
    for (int task = 0; task < context.total; task++) {
        if (atomic_load_explicit(
                &context.visits[task], memory_order_relaxed) != 1u) return 0;
    }
    return 1;
}

int main(void) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    VxParallelForTestContext empty = {0};
    if (!pool) return 1;
    if (!vx_test_exact_coverage(NULL, 3) ||
        !vx_test_exact_coverage(pool, 3) ||
        !vx_test_exact_coverage(pool, VX_TEST_TASKS)) {
        vx_kernel_thread_pool_destroy(pool);
        return 1;
    }
    empty.total = VX_TEST_TASKS;
    vx_kernels_parallel_for(0, 1, vx_test_range_worker, &empty);
    for (int task = 0; task < empty.total; task++) {
        if (atomic_load_explicit(
                &empty.visits[task], memory_order_relaxed) != 0u) {
            vx_kernel_thread_pool_destroy(pool);
            return 1;
        }
    }
    vx_kernel_thread_pool_destroy(pool);
    puts("kernel parallel-for range contract tests passed");
    return 0;
}
