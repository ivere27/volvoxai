#include "thread_pool.h"

#if !defined(__wasm__) && !defined(_WIN32)

#include <pthread.h>
#include <unistd.h>

#define VX_MAX_THREADS 16

typedef struct {
    pthread_t threads[VX_MAX_THREADS - 1];
    int nthreads;
    int nworkers;
    int initialized;
    int stop;
    int generation;
    int remaining;
    int next;
    int total;
    int grain;
    VxKernelParallelFn fn;
    void* context;
    pthread_mutex_t mutex;
    pthread_cond_t start;
    pthread_cond_t done;
} VxKernelThreadPool;

static VxKernelThreadPool g_kernel_pool = {
    .mutex = PTHREAD_MUTEX_INITIALIZER,
    .start = PTHREAD_COND_INITIALIZER,
    .done = PTHREAD_COND_INITIALIZER,
};
static int g_num_threads_override;
static int g_thread_count_cached;

static int vx_native_thread_count(void) {
    int count;
    if (g_thread_count_cached > 0) return g_thread_count_cached;
    count = g_num_threads_override;
    if (count <= 0) {
        long processors = sysconf(_SC_NPROCESSORS_ONLN);
        count = processors > 0 ? (int)processors : 1;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        if (count > 2) count = 2;
#else
        if (count > 4) count = 4;
#endif
    }
    if (count < 1) count = 1;
    if (count > VX_MAX_THREADS) count = VX_MAX_THREADS;
    g_thread_count_cached = count;
    return count;
}

static int vx_pool_take_chunk(int* begin, int* end) {
    int chunk_begin;
    int chunk_end;
    pthread_mutex_lock(&g_kernel_pool.mutex);
    chunk_begin = g_kernel_pool.next;
    if (chunk_begin >= g_kernel_pool.total) {
        pthread_mutex_unlock(&g_kernel_pool.mutex);
        return 0;
    }
    chunk_end = chunk_begin + g_kernel_pool.grain;
    if (chunk_end > g_kernel_pool.total) chunk_end = g_kernel_pool.total;
    g_kernel_pool.next = chunk_end;
    pthread_mutex_unlock(&g_kernel_pool.mutex);
    *begin = chunk_begin;
    *end = chunk_end;
    return 1;
}

static void vx_pool_run_chunks(void) {
    int begin;
    int end;
    while (vx_pool_take_chunk(&begin, &end))
        g_kernel_pool.fn(g_kernel_pool.context, begin, end);
}

static void* vx_worker_main(void* unused) {
    int seen_generation = 0;
    (void)unused;
    for (;;) {
        pthread_mutex_lock(&g_kernel_pool.mutex);
        while (!g_kernel_pool.stop &&
               g_kernel_pool.generation == seen_generation)
            pthread_cond_wait(&g_kernel_pool.start, &g_kernel_pool.mutex);
        if (g_kernel_pool.stop) {
            pthread_mutex_unlock(&g_kernel_pool.mutex);
            return NULL;
        }
        seen_generation = g_kernel_pool.generation;
        pthread_mutex_unlock(&g_kernel_pool.mutex);

        vx_pool_run_chunks();

        pthread_mutex_lock(&g_kernel_pool.mutex);
        g_kernel_pool.remaining--;
        if (g_kernel_pool.remaining == 0)
            pthread_cond_signal(&g_kernel_pool.done);
        pthread_mutex_unlock(&g_kernel_pool.mutex);
    }
}

static void vx_pool_init(void) {
    int created = 0;
    if (g_kernel_pool.initialized) return;
    g_kernel_pool.nthreads = vx_native_thread_count();
    g_kernel_pool.nworkers = g_kernel_pool.nthreads - 1;
    for (int index = 0; index < g_kernel_pool.nworkers; index++) {
        if (pthread_create(&g_kernel_pool.threads[index], NULL,
                           vx_worker_main, NULL) != 0)
            break;
        created++;
    }
    g_kernel_pool.nworkers = created;
    g_kernel_pool.nthreads = created + 1;
    g_kernel_pool.initialized = 1;
}

void vx_kernels_shutdown(void) {
    int workers;
    if (!g_kernel_pool.initialized) return;
    pthread_mutex_lock(&g_kernel_pool.mutex);
    g_kernel_pool.stop = 1;
    g_kernel_pool.generation++;
    pthread_cond_broadcast(&g_kernel_pool.start);
    workers = g_kernel_pool.nworkers;
    pthread_mutex_unlock(&g_kernel_pool.mutex);
    for (int index = 0; index < workers; index++)
        pthread_join(g_kernel_pool.threads[index], NULL);
    pthread_mutex_lock(&g_kernel_pool.mutex);
    g_kernel_pool.nthreads = 0;
    g_kernel_pool.nworkers = 0;
    g_kernel_pool.initialized = 0;
    g_kernel_pool.stop = 0;
    g_kernel_pool.generation = 0;
    g_kernel_pool.remaining = 0;
    g_kernel_pool.next = 0;
    g_kernel_pool.total = 0;
    g_kernel_pool.grain = 0;
    g_kernel_pool.fn = NULL;
    g_kernel_pool.context = NULL;
    pthread_mutex_unlock(&g_kernel_pool.mutex);
}

void vx_set_num_threads(int count) {
    vx_kernels_shutdown();
    g_num_threads_override = count > 0 ? count : 0;
    g_thread_count_cached = 0;
}

void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context) {
    int requested_threads;
    if (total <= 0 || !fn) return;
    if (grain < 1) grain = 1;
    requested_threads = vx_native_thread_count();
    if (requested_threads <= 1 || total <= grain) {
        fn(context, 0, total);
        return;
    }
    vx_pool_init();
    pthread_mutex_lock(&g_kernel_pool.mutex);
    g_kernel_pool.fn = fn;
    g_kernel_pool.context = context;
    g_kernel_pool.total = total;
    g_kernel_pool.grain = grain;
    g_kernel_pool.next = 0;
    g_kernel_pool.remaining = g_kernel_pool.nthreads;
    g_kernel_pool.generation++;
    pthread_cond_broadcast(&g_kernel_pool.start);
    pthread_mutex_unlock(&g_kernel_pool.mutex);

    vx_pool_run_chunks();

    pthread_mutex_lock(&g_kernel_pool.mutex);
    g_kernel_pool.remaining--;
    if (g_kernel_pool.remaining == 0)
        pthread_cond_signal(&g_kernel_pool.done);
    while (g_kernel_pool.remaining != 0)
        pthread_cond_wait(&g_kernel_pool.done, &g_kernel_pool.mutex);
    pthread_mutex_unlock(&g_kernel_pool.mutex);
}

int vx_kernels_thread_count(void) {
    return vx_native_thread_count();
}

#else

void vx_set_num_threads(int count) {
    (void)count;
}

int vx_kernels_thread_count(void) {
    return 1;
}

void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context) {
    (void)grain;
    if (total > 0 && fn) fn(context, 0, total);
}

void vx_kernels_shutdown(void) {
}

#endif
