/* cpu_set_t and the CPU_* macros are GNU extensions; this must precede every
 * system header so features.h sees it. */
#if !defined(__wasm__) && !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "thread_pool.h"

#include <stdint.h>

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

typedef struct VxKernelJob {
    VxKernelParallelFn fn;
    void* context;
    int total;
    int grain;
    int next_index;
    int active_chunks;
    int queued;
    int completed;
    pthread_cond_t completion;
    struct VxKernelJob* previous_ready;
    struct VxKernelJob* next_ready;
} VxKernelJob;

struct VxKernelThreadPool {
    pthread_t threads[VX_MAX_THREADS - 1];
    int nthreads;
    int nworkers;
    int initialized;
    int stop;
    int accepting;
    int destroying;
    int lifecycle_active;
    int caller_leader_active;
    int thread_count_override;
    int thread_count_cached;
    size_t worker_stack_size;
    size_t worker_guard_size;
    size_t active_scopes;
    size_t active_calls;
    size_t active_jobs;
    VxKernelJob* ready_head;
    VxKernelJob* ready_tail;
    pthread_mutex_t mutex;
    pthread_cond_t work_available;
    pthread_cond_t idle;
    pthread_cond_t lifecycle;
};

static _Thread_local VxKernelThreadPool* g_current_kernel_pool;
/* A worker callback, or a callback executed by the one caller-leader permit,
 * must not recursively submit to the same bounded pool.  Running the nested
 * range inline preserves progress even when every worker is already occupied
 * by an outer job. */
static _Thread_local unsigned int g_parallel_callback_depth;
static pthread_once_t g_worker_reservation_once = PTHREAD_ONCE_INIT;
static size_t g_worker_stack_size;
static size_t g_worker_guard_size;
static int g_worker_reservation_status = -1;

static int vx_size_round_up(size_t value, size_t alignment,
                            size_t* out_value) {
    size_t remainder;
    if (!alignment || !out_value) return -1;
    remainder = value % alignment;
    if (remainder && value > SIZE_MAX - (alignment - remainder)) return -1;
    *out_value = remainder ? value + alignment - remainder : value;
    return 0;
}

/* Snapshot the platform's pthread defaults once for the process, normalize the
 * implementation-rounded guard range, and later pass those exact values back
 * to every pthread_create.  A later pthread_setattr_default_np() call cannot
 * make a context reserve more than the attributes used by compilation proof. */
static void vx_worker_reservation_initialize(void) {
    pthread_attr_t attributes;
    size_t stack_size = 0;
    size_t guard_size = 0;
    size_t page_size;
    long page_size_long;
    int status;
    status = pthread_attr_init(&attributes);
    if (status != 0) return;
    status = pthread_attr_getstacksize(&attributes, &stack_size);
    if (status == 0)
        status = pthread_attr_getguardsize(&attributes, &guard_size);
    page_size_long = sysconf(_SC_PAGESIZE);
    if (status != 0 || page_size_long <= 0) {
        pthread_attr_destroy(&attributes);
        return;
    }
    page_size = (size_t)page_size_long;
    if (vx_size_round_up(stack_size, page_size, &stack_size) != 0 ||
        vx_size_round_up(guard_size, page_size, &guard_size) != 0 ||
        !stack_size || stack_size > SIZE_MAX - guard_size ||
        pthread_attr_setstacksize(&attributes, stack_size) != 0 ||
        pthread_attr_setguardsize(&attributes, guard_size) != 0) {
        pthread_attr_destroy(&attributes);
        return;
    }
    g_worker_stack_size = stack_size;
    g_worker_guard_size = guard_size;
    g_worker_reservation_status = 0;
    pthread_attr_destroy(&attributes);
}

static int vx_pool_configure_worker_reservation(VxKernelThreadPool* pool) {
    if (!pool || pthread_once(&g_worker_reservation_once,
                              vx_worker_reservation_initialize) != 0 ||
        g_worker_reservation_status != 0) return -1;
    pool->worker_stack_size = g_worker_stack_size;
    pool->worker_guard_size = g_worker_guard_size;
    return 0;
}

static int vx_pool_explicit_worker_attributes(
        const VxKernelThreadPool* pool, pthread_attr_t* attributes) {
    if (!pool || !attributes || !pool->worker_stack_size ||
        pthread_attr_init(attributes) != 0) return -1;
    if (pthread_attr_setstacksize(attributes, pool->worker_stack_size) != 0 ||
        pthread_attr_setguardsize(attributes, pool->worker_guard_size) != 0) {
        pthread_attr_destroy(attributes);
        return -1;
    }
    return 0;
}

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

static pthread_once_t g_physical_core_once = PTHREAD_ONCE_INIT;
static int g_physical_core_count;

/* CPU topology cannot change under a running process, and the scan opens one
 * sysfs file per allowed CPU, so resolve it once without a cross-pool data
 * race when independent pools make their first submission concurrently. */
static void vx_native_physical_cores_initialize(void) {
    g_physical_core_count = vx_native_physical_cores();
}

static int vx_native_cached_physical_cores(void) {
    if (pthread_once(&g_physical_core_once,
                     vx_native_physical_cores_initialize) != 0)
        return 0;
    return g_physical_core_count > 0 ? g_physical_core_count : 0;
}
#endif

/* pool->mutex protects the cached resolution and every configuration change. */
static int vx_native_thread_count_locked(VxKernelThreadPool* pool) {
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

static int vx_native_thread_count(VxKernelThreadPool* pool) {
    int count;
    if (!pool) return 1;
    pthread_mutex_lock(&pool->mutex);
    count = vx_native_thread_count_locked(pool);
    pthread_mutex_unlock(&pool->mutex);
    return count;
}

static int vx_native_maximum_thread_count(
        const VxKernelThreadPool* pool) {
    int count;
    if (!pool) return 1;
    count = pool->thread_count_override;
    /* Auto mode consults process environment/topology separately for each
     * state.  Its proof must therefore cover the full structural cap even if
     * the validation state happens to resolve fewer workers. */
    if (count <= 0) count = VX_MAX_THREADS;
    if (count > VX_MAX_THREADS) count = VX_MAX_THREADS;
    return count;
}

static void vx_parallel_invoke(VxKernelParallelFn fn, void* context,
                               int begin, int end) {
    g_parallel_callback_depth++;
    fn(context, begin, end);
    g_parallel_callback_depth--;
}

static void vx_pool_ready_append_locked(VxKernelThreadPool* pool,
                                        VxKernelJob* job) {
    job->previous_ready = pool->ready_tail;
    job->next_ready = NULL;
    if (pool->ready_tail)
        pool->ready_tail->next_ready = job;
    else
        pool->ready_head = job;
    pool->ready_tail = job;
    job->queued = 1;
}

static void vx_pool_ready_remove_locked(VxKernelThreadPool* pool,
                                        VxKernelJob* job) {
    if (!job->queued) return;
    if (job->previous_ready)
        job->previous_ready->next_ready = job->next_ready;
    else
        pool->ready_head = job->next_ready;
    if (job->next_ready)
        job->next_ready->previous_ready = job->previous_ready;
    else
        pool->ready_tail = job->previous_ready;
    job->previous_ready = NULL;
    job->next_ready = NULL;
    job->queued = 0;
}

static void vx_pool_ready_rotate_locked(VxKernelThreadPool* pool,
                                        VxKernelJob* job) {
    if (!job->queued || pool->ready_tail == job) return;
    vx_pool_ready_remove_locked(pool, job);
    vx_pool_ready_append_locked(pool, job);
}

static int vx_pool_claim_job_chunk_locked(VxKernelThreadPool* pool,
                                          VxKernelJob* job,
                                          int* begin, int* end) {
    int chunk_begin;
    int chunk_end;
    if (!job || job->completed || job->next_index >= job->total) return 0;
    chunk_begin = job->next_index;
    chunk_end = job->grain >= job->total - chunk_begin
        ? job->total : chunk_begin + job->grain;
    job->next_index = chunk_end;
    job->active_chunks++;
    if (job->next_index >= job->total)
        vx_pool_ready_remove_locked(pool, job);
    *begin = chunk_begin;
    *end = chunk_end;
    return 1;
}

/* Workers take one chunk and rotate that job behind every other ready job.
 * Large jobs therefore cannot monopolize a shared Runtime pool while a short
 * job waits behind their complete range. */
static VxKernelJob* vx_pool_claim_fair_chunk_locked(
        VxKernelThreadPool* pool, int* begin, int* end) {
    VxKernelJob* job;
    while ((job = pool->ready_head) != NULL) {
        if (!vx_pool_claim_job_chunk_locked(pool, job, begin, end)) {
            vx_pool_ready_remove_locked(pool, job);
            continue;
        }
        if (job->queued) vx_pool_ready_rotate_locked(pool, job);
        return job;
    }
    return NULL;
}

static void vx_pool_finish_chunk_locked(VxKernelThreadPool* pool,
                                        VxKernelJob* job) {
    if (!job || job->active_chunks <= 0) return;
    job->active_chunks--;
    if (job->next_index < job->total || job->active_chunks != 0 ||
        job->completed)
        return;
    job->completed = 1;
    if (pool->active_jobs) pool->active_jobs--;
    pthread_cond_broadcast(&job->completion);
    if (!pool->active_jobs) pthread_cond_broadcast(&pool->idle);
}

/* Only one submitting thread may execute callbacks alongside the persistent
 * workers. When its job is no longer at the fair queue head, wake every ready
 * job's caller; the current head acquires the permit while the rest resume
 * waiting on their own per-job condition. This also provides progress and
 * chunk-level fairness when width one has no persistent worker. */
static void vx_pool_release_caller_leader_locked(
        VxKernelThreadPool* pool) {
    VxKernelJob* ready;
    if (!pool->caller_leader_active) return;
    pool->caller_leader_active = 0;
    for (ready = pool->ready_head; ready; ready = ready->next_ready)
        pthread_cond_signal(&ready->completion);
    pthread_cond_broadcast(&pool->lifecycle);
}

static void* vx_worker_main(void* opaque) {
    VxKernelThreadPool* pool = (VxKernelThreadPool*)opaque;
    for (;;) {
        VxKernelJob* job;
        VxKernelParallelFn fn;
        void* context;
        int begin;
        int end;
        pthread_mutex_lock(&pool->mutex);
        while (!pool->stop && !pool->ready_head)
            pthread_cond_wait(&pool->work_available, &pool->mutex);
        if (pool->stop) {
            pthread_mutex_unlock(&pool->mutex);
            return NULL;
        }
        job = vx_pool_claim_fair_chunk_locked(pool, &begin, &end);
        if (!job) {
            pthread_mutex_unlock(&pool->mutex);
            continue;
        }
        fn = job->fn;
        context = job->context;
        pthread_mutex_unlock(&pool->mutex);

        vx_parallel_invoke(fn, context, begin, end);

        pthread_mutex_lock(&pool->mutex);
        vx_pool_finish_chunk_locked(pool, job);
        pthread_mutex_unlock(&pool->mutex);
    }
}

/* pool->mutex is held across initialization so at most one submitter creates
 * the persistent worker set. */
static void vx_pool_init_locked(VxKernelThreadPool* pool) {
    int created = 0;
    pthread_attr_t attributes;
    if (!pool || pool->initialized) return;
    pool->nthreads = vx_native_thread_count_locked(pool);
    pool->nworkers = pool->nthreads - 1;
    if (pool->nworkers > 0 &&
        vx_pool_explicit_worker_attributes(pool, &attributes) != 0) {
        pool->nworkers = 0;
        pool->nthreads = 1;
        pool->initialized = 1;
        return;
    }
    for (int index = 0; index < pool->nworkers; index++) {
        if (pthread_create(&pool->threads[index], &attributes,
                           vx_worker_main, pool) != 0)
            break;
        created++;
    }
    if (pool->nworkers > 0) pthread_attr_destroy(&attributes);
    pool->nworkers = created;
    pool->nthreads = created + 1;
    pool->initialized = 1;
}

static void vx_pool_wait_for_admission_locked(VxKernelThreadPool* pool) {
    /* A retained scope pins the pool while restart/reconfiguration is active.
     * Destruction cannot advance to its non-accepting phase until every such
     * scope leaves, so a retained submitter always eventually observes an open
     * admission gate. */
    while (pool->lifecycle_active || !pool->accepting)
        pthread_cond_wait(&pool->lifecycle, &pool->mutex);
}

/* pthread_cond_init failure is exceptional, but it must not escape the same
 * concurrency bound as an ordinary queued job. Borrow the single caller lane
 * and account the callback as an active synchronous call. */
static void vx_pool_run_bounded_inline(VxKernelThreadPool* pool,
                                       VxKernelParallelFn fn,
                                       void* context, int total) {
    pthread_mutex_lock(&pool->mutex);
    vx_pool_wait_for_admission_locked(pool);
    vx_pool_init_locked(pool);
    while (pool->caller_leader_active) {
        pthread_cond_wait(&pool->lifecycle, &pool->mutex);
        vx_pool_wait_for_admission_locked(pool);
    }
    pool->caller_leader_active = 1;
    pool->active_calls++;
    pthread_mutex_unlock(&pool->mutex);

    vx_parallel_invoke(fn, context, 0, total);

    pthread_mutex_lock(&pool->mutex);
    if (pool->active_calls) pool->active_calls--;
    vx_pool_release_caller_leader_locked(pool);
    if (!pool->active_calls) pthread_cond_broadcast(&pool->idle);
    pthread_mutex_unlock(&pool->mutex);
}

typedef enum VxPoolStopAction {
    VX_POOL_STOP_RESTARTABLE = 0,
    VX_POOL_STOP_RECONFIGURE = 1,
    VX_POOL_STOP_DESTROY = 2,
} VxPoolStopAction;

/* Stop accepts no new jobs, drains every accepted synchronous call, and only
 * then joins the persistent workers.  Reconfiguration publishes its new width
 * before admission reopens, so no call can enter with half-applied settings. */
static int vx_kernel_thread_pool_stop(VxKernelThreadPool* pool,
                                      VxPoolStopAction action,
                                      int thread_count) {
    int workers;
    if (!pool) return -1;
    pthread_mutex_lock(&pool->mutex);
    while (pool->lifecycle_active && !pool->destroying)
        pthread_cond_wait(&pool->lifecycle, &pool->mutex);
    if (pool->destroying) {
        pthread_mutex_unlock(&pool->mutex);
        return -1;
    }
    if (action == VX_POOL_STOP_DESTROY) {
        /* Mark destruction before waiting so a scope which starts before the
         * owner's external quiescence barrier cannot become newly retained.
         * Existing retained scopes may continue submitting while destroy
         * waits; this closes the pre-admission UAF window without exceeding
         * the pool's execution width. */
        pool->destroying = 1;
        while (pool->active_scopes)
            pthread_cond_wait(&pool->idle, &pool->mutex);
    }
    pool->lifecycle_active = 1;
    pool->accepting = 0;
    while (pool->active_calls)
        pthread_cond_wait(&pool->idle, &pool->mutex);
    workers = pool->initialized ? pool->nworkers : 0;
    pool->stop = 1;
    pthread_cond_broadcast(&pool->work_available);
    pthread_mutex_unlock(&pool->mutex);
    for (int index = 0; index < workers; index++)
        pthread_join(pool->threads[index], NULL);
    pthread_mutex_lock(&pool->mutex);
    pool->nthreads = 0;
    pool->nworkers = 0;
    pool->initialized = 0;
    pool->stop = 0;
    pool->ready_head = NULL;
    pool->ready_tail = NULL;
    pool->caller_leader_active = 0;
    if (action == VX_POOL_STOP_RECONFIGURE) {
        pool->thread_count_override = thread_count > 0 ? thread_count : 0;
        pool->thread_count_cached = 0;
    }
    pool->lifecycle_active = 0;
    if (action != VX_POOL_STOP_DESTROY) pool->accepting = 1;
    pthread_cond_broadcast(&pool->lifecycle);
    pthread_mutex_unlock(&pool->mutex);
    return 0;
}

VxKernelThreadPool* vx_kernel_thread_pool_create(int thread_count) {
    VxKernelThreadPool* pool = (VxKernelThreadPool*)calloc(1, sizeof(*pool));
    if (!pool) return NULL;
    if (pthread_mutex_init(&pool->mutex, NULL) != 0) {
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->work_available, NULL) != 0) {
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->idle, NULL) != 0) {
        pthread_cond_destroy(&pool->work_available);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    if (pthread_cond_init(&pool->lifecycle, NULL) != 0) {
        pthread_cond_destroy(&pool->idle);
        pthread_cond_destroy(&pool->work_available);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    if (vx_pool_configure_worker_reservation(pool) != 0) {
        pthread_cond_destroy(&pool->lifecycle);
        pthread_cond_destroy(&pool->idle);
        pthread_cond_destroy(&pool->work_available);
        pthread_mutex_destroy(&pool->mutex);
        free(pool);
        return NULL;
    }
    pool->thread_count_override = thread_count > 0 ? thread_count : 0;
    pool->accepting = 1;
    return pool;
}

int vx_kernel_thread_pool_resource_bound(
        VxKernelThreadPool* pool,
        size_t* out_pool_heap_bytes,
        size_t* out_worker_count,
        size_t* out_worker_stack_reservation_bytes) {
    size_t workers;
    size_t per_worker;
    if (!pool) return -1;
    pthread_mutex_lock(&pool->mutex);
    workers = (size_t)(vx_native_maximum_thread_count(pool) - 1);
    if (pool->worker_stack_size > SIZE_MAX - pool->worker_guard_size)
        goto fail;
    per_worker = pool->worker_stack_size + pool->worker_guard_size;
    if (per_worker && workers > SIZE_MAX / per_worker) goto fail;
    if (out_pool_heap_bytes) *out_pool_heap_bytes = sizeof(*pool);
    if (out_worker_count) *out_worker_count = workers;
    if (out_worker_stack_reservation_bytes)
        *out_worker_stack_reservation_bytes = workers * per_worker;
    pthread_mutex_unlock(&pool->mutex);
    return 0;
fail:
    pthread_mutex_unlock(&pool->mutex);
    return -1;
}

void vx_kernel_thread_pool_destroy(VxKernelThreadPool* pool) {
    if (!pool) return;
    if (vx_kernel_thread_pool_stop(pool, VX_POOL_STOP_DESTROY, 0) != 0)
        return;
    pthread_cond_destroy(&pool->lifecycle);
    pthread_cond_destroy(&pool->idle);
    pthread_cond_destroy(&pool->work_available);
    pthread_mutex_destroy(&pool->mutex);
    free(pool);
}

VxKernelThreadPoolScope vx_kernel_thread_pool_scope_enter(
    VxKernelThreadPool* pool) {
    VxKernelThreadPoolScope scope = {g_current_kernel_pool, pool, 0};
    if (pool) {
        pthread_mutex_lock(&pool->mutex);
        if (!pool->destroying) {
            pool->active_scopes++;
            scope.retained = 1;
        } else {
            scope.bound = NULL;
        }
        pthread_mutex_unlock(&pool->mutex);
    }
    g_current_kernel_pool = scope.bound;
    return scope;
}

void vx_kernel_thread_pool_scope_leave(VxKernelThreadPoolScope scope) {
    if (g_current_kernel_pool == scope.bound) {
        g_current_kernel_pool = scope.previous;
        if (scope.retained && scope.bound) {
            pthread_mutex_lock(&scope.bound->mutex);
            if (scope.bound->active_scopes)
                scope.bound->active_scopes--;
            if (!scope.bound->active_scopes)
                pthread_cond_broadcast(&scope.bound->idle);
            pthread_mutex_unlock(&scope.bound->mutex);
        }
    }
}

void vx_kernels_shutdown(void) {
    if (g_parallel_callback_depth) return;
    (void)vx_kernel_thread_pool_stop(
        g_current_kernel_pool, VX_POOL_STOP_RESTARTABLE, 0);
}

void vx_set_num_threads(int count) {
    VxKernelThreadPool* pool = g_current_kernel_pool;
    if (!pool || g_parallel_callback_depth) return;
    (void)vx_kernel_thread_pool_stop(
        pool, VX_POOL_STOP_RECONFIGURE, count);
}

void vx_kernels_parallel_for(int total, int grain, VxKernelParallelFn fn,
                             void* context) {
    VxKernelThreadPool* pool = g_current_kernel_pool;
    VxKernelJob job = {0};
    int caller_leader = 0;
    if (total <= 0 || !fn) return;
    if (grain < 1) grain = 1;
    if (g_parallel_callback_depth) {
        vx_parallel_invoke(fn, context, 0, total);
        return;
    }
    if (!pool) {
        vx_parallel_invoke(fn, context, 0, total);
        return;
    }
    if (pthread_cond_init(&job.completion, NULL) != 0) {
        vx_pool_run_bounded_inline(pool, fn, context, total);
        return;
    }
    job.fn = fn;
    job.context = context;
    job.total = total;
    job.grain = grain;

    pthread_mutex_lock(&pool->mutex);
    vx_pool_wait_for_admission_locked(pool);
    vx_pool_init_locked(pool);
    pool->active_calls++;
    pool->active_jobs++;
    vx_pool_ready_append_locked(pool, &job);
    pthread_cond_broadcast(&pool->work_available);
    pthread_mutex_unlock(&pool->mutex);

    for (;;) {
        int begin;
        int end;
        int claimed = 0;
        pthread_mutex_lock(&pool->mutex);
        if (!caller_leader && !job.completed &&
            pool->ready_head == &job &&
            !pool->caller_leader_active) {
            pool->caller_leader_active = 1;
            caller_leader = 1;
        }
        if (caller_leader && pool->ready_head == &job) {
            claimed = vx_pool_claim_job_chunk_locked(
                pool, &job, &begin, &end);
            if (claimed && job.queued)
                vx_pool_ready_rotate_locked(pool, &job);
        }
        if (!claimed && caller_leader) {
            vx_pool_release_caller_leader_locked(pool);
            caller_leader = 0;
        }
        if (!claimed && job.completed) {
            if (pool->active_calls) pool->active_calls--;
            if (!pool->active_calls) pthread_cond_broadcast(&pool->idle);
            pthread_mutex_unlock(&pool->mutex);
            break;
        }
        if (!claimed) {
            pthread_cond_wait(&job.completion, &pool->mutex);
            pthread_mutex_unlock(&pool->mutex);
            continue;
        }
        pthread_mutex_unlock(&pool->mutex);
        vx_parallel_invoke(fn, context, begin, end);
        pthread_mutex_lock(&pool->mutex);
        vx_pool_finish_chunk_locked(pool, &job);
        pthread_mutex_unlock(&pool->mutex);
    }
    pthread_cond_destroy(&job.completion);
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

int vx_kernel_thread_pool_resource_bound(
        VxKernelThreadPool* pool,
        size_t* out_pool_heap_bytes,
        size_t* out_worker_count,
        size_t* out_worker_stack_reservation_bytes) {
    if (!pool) return -1;
#if defined(__wasm__)
    if (out_pool_heap_bytes) *out_pool_heap_bytes = 0u;
#else
    if (out_pool_heap_bytes) *out_pool_heap_bytes = sizeof(*pool);
#endif
    if (out_worker_count) *out_worker_count = 0u;
    if (out_worker_stack_reservation_bytes)
        *out_worker_stack_reservation_bytes = 0u;
    return 0;
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
    VxKernelThreadPoolScope scope = {g_current_kernel_pool, pool, 0};
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
