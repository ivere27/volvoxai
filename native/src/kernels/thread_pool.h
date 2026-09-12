#ifndef VOLVOXAI_KERNEL_THREAD_POOL_H
#define VOLVOXAI_KERNEL_THREAD_POOL_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*VxKernelParallelFn)(void* context, int begin, int end);

typedef struct VxKernelThreadPool VxKernelThreadPool;

typedef struct VxKernelThreadPoolScope {
    VxKernelThreadPool* previous;
    VxKernelThreadPool* bound;
    int retained;
} VxKernelThreadPoolScope;

VxKernelThreadPool* vx_kernel_thread_pool_create(int thread_count);
/* On POSIX worker builds the pool is a synchronous multi-producer executor and
 * a successfully bound scope pins its lifetime. Before destroy, the owner must
 * establish external quiescence: no scope-enter call may be in flight or start
 * later. Destroy then stops admission, drains every already-bound scope and
 * accepted call, and joins the persistent workers. Destruction is owner-only,
 * non-concurrent, and must not be invoked from a bound scope or callback.
 * Windows/WASM retain their serial, owner-quiesced implementation. */
void vx_kernel_thread_pool_destroy(VxKernelThreadPool* pool);
/* Returns the allocations/reservations owned by a fully initialized pool.
 * The stack byte count includes every requested worker stack and guard range;
 * workers are lazy, but the bound covers the maximum configured population. */
int vx_kernel_thread_pool_resource_bound(
    VxKernelThreadPool* pool,
    size_t* out_pool_heap_bytes,
    size_t* out_worker_count,
    size_t* out_worker_stack_reservation_bytes);
VxKernelThreadPoolScope vx_kernel_thread_pool_scope_enter(
    VxKernelThreadPool* pool);
void vx_kernel_thread_pool_scope_leave(VxKernelThreadPoolScope scope);

/* These operations target the pool bound by the current owner scope. Multiple
 * owner threads may submit concurrently. Persistent workers claim ready jobs
 * one chunk at a time in round-robin order; a single bounded caller lane only
 * executes its submitting thread's own callbacks. Each call is synchronous
 * and returns after all of its chunks complete. A parallel_for invoked by one
 * of its own callbacks runs the nested range serially, which makes nesting
 * deadlock-free. Without a bound owner, execution is serial and retains no
 * process-default state. */
void vx_set_num_threads(int count);
int vx_kernels_thread_count(void);
void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context);
void vx_kernels_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
