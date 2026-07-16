#ifndef VOLVOXAI_KERNEL_THREAD_POOL_H
#define VOLVOXAI_KERNEL_THREAD_POOL_H

#ifdef __cplusplus
extern "C" {
#endif

typedef void (*VxKernelParallelFn)(void* context, int begin, int end);

/* Thread configuration and shutdown must be performed while no kernel call is
 * active.  A pool that has been shut down is initialized again on demand. */
void vx_set_num_threads(int count);
int vx_kernels_thread_count(void);
void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context);
void vx_kernels_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
