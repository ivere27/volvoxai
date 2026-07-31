#ifndef VOLVOXAI_KERNEL_THREAD_POOL_H
#define VOLVOXAI_KERNEL_THREAD_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*VxKernelParallelFn)(void* context, int begin, int end);

typedef struct VxKernelThreadPool VxKernelThreadPool;

typedef struct VxKernelThreadPoolScope {
    VxKernelThreadPool* previous;
    VxKernelThreadPool* bound;
} VxKernelThreadPoolScope;

VxKernelThreadPool* vx_kernel_thread_pool_create(int thread_count);
void vx_kernel_thread_pool_destroy(VxKernelThreadPool* pool);
VxKernelThreadPoolScope vx_kernel_thread_pool_scope_enter(
    VxKernelThreadPool* pool);
void vx_kernel_thread_pool_scope_leave(VxKernelThreadPoolScope scope);

/* These operations target the pool bound by the current owner scope. Without
 * a bound owner they execute serially and retain no process-default state. */
void vx_set_num_threads(int count);
int vx_kernels_thread_count(void);
void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context);
void vx_kernels_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
