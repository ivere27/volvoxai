/* cpu_set_t and the CPU_* macros are GNU extensions; this must precede every
 * system header so features.h sees it. */
#if !defined(__wasm__) && !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "thread_pool.h"

#if !defined(__wasm__)
#include <stdlib.h>
#endif

#if !defined(__wasm__) && !defined(_WIN32)

#include <pthread.h>
#include <sched.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>

#define VX_MAX_THREADS 16

struct VxKernelThreadPool {
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
    int thread_count_override;
    int thread_count_cached;
    VxKernelParallelFn fn;
    void* context;
    pthread_mutex_t mutex;
    pthread_cond_t start;
    pthread_cond_t done;
};

static _Thread_local VxKernelThreadPool* g_current_kernel_pool;

#if !defined(__ARM_NEON) && !defined(__ARM_NEON__)
/* Count distinct physical cores from the sysfs topology.  Byte-domain GEMM and
 * norm kernels are load/store bound, so a second SMT sibling on the same core
 * adds scheduling and L1 pressure without adding throughput.  Returns 0 when
 * the topology is unavailable and the caller must fall back to the online
 * processor count. */
static int vx_native_physical_cores(void) {
    cpu_set_t allowed;
    int physical = 0;
    /* Count only CPUs this process may actually run on.  A container quota or
     * an explicit affinity mask usually pins a non-contiguous set, so walking
     * indices 0..online-1 would describe a different machine than the one the
     * pool will schedule on. */
    CPU_ZERO(&allowed);
    if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return 0;
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++) {
        char path[128];
        char list[256];
        FILE* file;
        size_t length;
        int first;
        if (!CPU_ISSET(cpu, &allowed)) continue;
        snprintf(path, sizeof path,
                 "/sys/devices/system/cpu/cpu%d/topology/thread_siblings_list",
                 cpu);
        file = fopen(path, "r");
        if (!file) return 0;
        length = fread(list, 1, sizeof list - 1, file);
        fclose(file);
        if (!length) return 0;
        list[length] = '\0';
        /* The lowest allowed sibling represents the physical core, so each core
         * is counted exactly once even when only one of its threads is
         * available to this process. */
        first = (int)strtol(list, NULL, 10);
        while (first >= 0 && first < cpu && !CPU_ISSET(first, &allowed)) {
            const char* next = strchr(list, ',');
            if (!next) { first = cpu; break; }
            memmove(list, next + 1, strlen(next + 1) + 1);
            first = (int)strtol(list, NULL, 10);
        }
        if (first == cpu) physical++;
    }
    return physical;
}

/* CPU topology cannot change under a running process, and the scan opens one
 * sysfs file per allowed CPU, so resolve it at most once per process.  A benign
 * race recomputes the same value. */
static int vx_native_cached_physical_cores(void) {
    static int cached;
    int value = cached;
    if (value == 0) {
        value = vx_native_physical_cores();
        if (value <= 0) value = -1;
        cached = value;
    }
    return value > 0 ? value : 0;
}
#endif

static int vx_native_thread_count(VxKernelThreadPool* pool) {
    int count;
    if (!pool) return 1;
    if (pool->thread_count_cached > 0) return pool->thread_count_cached;
    count = pool->thread_count_override;
    if (count <= 0) {
        const char* requested = getenv("VOLVOXAI_THREADS");
        count = requested ? (int)strtol(requested, NULL, 10) : 0;
    }
    if (count <= 0) {
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        /* Unchanged ARM policy.  Heterogeneous big.LITTLE clusters were tuned
         * to two workers and that choice has not been re-measured here. */
        long processors = sysconf(_SC_NPROCESSORS_ONLN);
        count = processors > 0 ? (int)processors : 1;
        if (count > 2) count = 2;
#else
        /* One worker per physical core.  The previous fixed cap of four left
         * measurable throughput unused above four cores, while running one
         * worker per SMT sibling was slower than one per core. */
        count = vx_native_cached_physical_cores();
        if (count <= 0) {
            long processors = sysconf(_SC_NPROCESSORS_ONLN);
            count = processors > 0 ? (int)processors : 1;
            if (count > 4) count = 4;
        }
#endif
    }
    if (count < 1) count = 1;
    if (count > VX_MAX_THREADS) count = VX_MAX_THREADS;
    pool->thread_count_cached = count;
    return count;
}

static int vx_pool_take_chunk(VxKernelThreadPool* pool, int* begin, int* end) {
    int chunk_begin;
    int chunk_end;
    pthread_mutex_lock(&pool->mutex);
    chunk_begin = pool->next;
    if (chunk_begin >= pool->total) {
        pthread_mutex_unlock(&pool->mutex);
        return 0;
    }
    chunk_end = chunk_begin + pool->grain;
    if (chunk_end > pool->total) chunk_end = pool->total;
    pool->next = chunk_end;
    pthread_mutex_unlock(&pool->mutex);
    *begin = chunk_begin;
    *end = chunk_end;
    return 1;
}

static void vx_pool_run_chunks(VxKernelThreadPool* pool) {
    int begin;
    int end;
    while (vx_pool_take_chunk(pool, &begin, &end))
        pool->fn(pool->context, begin, end);
}

static void* vx_worker_main(void* opaque) {
    VxKernelThreadPool* pool = (VxKernelThreadPool*)opaque;
    int seen_generation = 0;
    for (;;) {
        pthread_mutex_lock(&pool->mutex);
        while (!pool->stop && pool->generation == seen_generation)
            pthread_cond_wait(&pool->start, &pool->mutex);
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        seen_generation = pool->generation;
        pthread_mutex_unlock(&pool->mutex);

        vx_pool_run_chunks(pool);

        pthread_mutex_lock(&pool->mutex);
        pool->remaining--;
        if (pool->remaining == 0) pthread_cond_signal(&pool->done);
        pthread_mutex_unlock(&pool->mutex);
    }
}

static void vx_pool_init(VxKernelThreadPool* pool) {
    int created = 0;
    if (!pool || pool->initialized) return;
    pool->nthreads = vx_native_thread_count(pool);
    pool->nworkers = pool->nthreads - 1;
    for (int index = 0; index < pool->nworkers; index++) {
        if (pthread_create(&pool->threads[index], NULL, vx_worker_main, pool) != 0)
            break;
        created++;
    }
    pool->nworkers = created;
    pool->nthreads = created + 1;
    pool->initialized = 1;
}

static void vx_kernel_thread_pool_shutdown(VxKernelThreadPool* pool) {
    int workers;
    if (!pool || !pool->initialized) return;
    pthread_mutex_lock(&pool->mutex);
    pool->stop = 1;
    pool->generation++;
    pthread_cond_broadcast(&pool->start);
    workers = pool->nworkers;
    pthread_mutex_unlock(&pool->mutex);
    for (int index = 0; index < workers; index++)
        pthread_join(pool->threads[index], NULL);
    pthread_mutex_lock(&pool->mutex);
    pool->nthreads = 0;
    pool->nworkers = 0;
    pool->initialized = 0;
    pool->stop = 0;
    pool->generation = 0;
    pool->remaining = 0;
    pool->next = 0;
    pool->total = 0;
    pool->grain = 0;
    pool->fn = NULL;
    pool->context = NULL;
    pthread_mutex_unlock(&pool->mutex);
}

VxKernelThreadPool* vx_kernel_thread_pool_create(int thread_count) {
    VxKernelThreadPool* pool = (VxKernelThreadPool*)calloc(1, sizeof(*pool));
    if (!pool) return NULL;
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->start, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->done, NULL) != 0) {
        pthread_cond_destroy(&pool->start);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    pool->thread_count_override = thread_count > 0 ? thread_count : 0;
    return pool;
}

void vx_kernel_thread_pool_destroy(VxKernelThreadPool* pool) {
    if (!pool) return;
    vx_kernel_thread_pool_shutdown(pool);
    pthread_cond_destroy(&pool->done);
    pthread_cond_destroy(&pool->start);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
}

VxKernelThreadPoolScope vx_kernel_thread_pool_scope_enter(
    VxKernelThreadPool* pool) {
    VxKernelThreadPoolScope scope = {g_current_kernel_pool, pool};
    g_current_kernel_pool = pool;
    return scope;
}

void vx_kernel_thread_pool_scope_leave(VxKernelThreadPoolScope scope) {
    if (g_current_kernel_pool == scope.bound)
        g_current_kernel_pool = scope.previous;
}

void vx_kernels_shutdown(void) {
    vx_kernel_thread_pool_shutdown(g_current_kernel_pool);
}

void vx_set_num_threads(int count) {
    VxKernelThreadPool* pool = g_current_kernel_pool;
    if (!pool) return;
    vx_kernel_thread_pool_shutdown(pool);
    pool->thread_count_override = count > 0 ? count : 0;
    pool->thread_count_cached = 0;
}

void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context) {
    VxKernelThreadPool* pool = g_current_kernel_pool;
    int requested_threads;
    if (total <= 0 || !fn) return;
    if (grain < 1) grain = 1;
    requested_threads = vx_native_thread_count(pool);
    if (!pool || requested_threads <= 1 || total <= grain) {
        fn(context, 0, total);
        return;
    }
    vx_pool_init(pool);
    pthread_mutex_lock(&pool->mutex);
    pool->fn = fn;
    pool->context = context;
    pool->total = total;
    pool->grain = grain;
    pool->next = 0;
    pool->remaining = pool->nthreads;
    pool->generation++;
    pthread_cond_broadcast(&pool->start);
    pthread_mutex_unlock(&pool->mutex);

    vx_pool_run_chunks(pool);

    pthread_mutex_lock(&pool->mutex);
    pool->remaining--;
    if (pool->remaining == 0) pthread_cond_signal(&pool->done);
    while (pool->remaining != 0)
        pthread_cond_wait(&pool->done, &pool->mutex);
    pthread_mutex_unlock(&pool->mutex);
}

int vx_kernels_thread_count(void) {
    return vx_native_thread_count(g_current_kernel_pool);
}

#else

struct VxKernelThreadPool { int unused; };
static _Thread_local VxKernelThreadPool* g_current_kernel_pool;

VxKernelThreadPool* vx_kernel_thread_pool_create(int thread_count) {
    (void)thread_count;
#if defined(__wasm__)
    return (VxKernelThreadPool*)1;
#else
    return (VxKernelThreadPool*)calloc(1, sizeof(VxKernelThreadPool));
#endif
}

void vx_kernel_thread_pool_destroy(VxKernelThreadPool* pool) {
#if defined(__wasm__)
    (void)pool;
#else
    free(pool);
#endif
}

VxKernelThreadPoolScope vx_kernel_thread_pool_scope_enter(
    VxKernelThreadPool* pool) {
    VxKernelThreadPoolScope scope = {g_current_kernel_pool, pool};
    g_current_kernel_pool = pool;
    return scope;
}

void vx_kernel_thread_pool_scope_leave(VxKernelThreadPoolScope scope) {
    if (g_current_kernel_pool == scope.bound)
        g_current_kernel_pool = scope.previous;
}

void vx_set_num_threads(int count) { (void)count; }
int vx_kernels_thread_count(void) { return 1; }
void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context) {
    (void)grain;
    if (total > 0 && fn) fn(context, 0, total);
}
void vx_kernels_shutdown(void) {}

#endif
