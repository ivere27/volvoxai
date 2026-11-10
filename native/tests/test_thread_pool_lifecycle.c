#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "thread_pool.h"

#include <errno.h>
#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { TASKS = 67, OUTER_TASKS = 13, INNER_TASKS = 11 };

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                #condition);                                                 \
        return 0;                                                            \
    }                                                                        \
} while (0)

typedef struct {
    int total;
    atomic_int invalid;
    atomic_uint visits[TASKS];
} Coverage;

static void cover_range(Coverage* coverage, int begin, int end) {
    if (begin < 0 || begin >= end || end > coverage->total) {
        atomic_store_explicit(&coverage->invalid, 1, memory_order_relaxed);
        return;
    }
    for (int index = begin; index < end; index++)
        atomic_fetch_add_explicit(&coverage->visits[index], 1u,
                                  memory_order_relaxed);
}

static int coverage_is_exact(const Coverage* coverage) {
    if (atomic_load_explicit(&coverage->invalid, memory_order_relaxed))
        return 0;
    for (int index = 0; index < coverage->total; index++)
        if (atomic_load_explicit(&coverage->visits[index],
                                 memory_order_relaxed) != 1u)
            return 0;
    return 1;
}

static void deadline_after(struct timespec* deadline, time_t seconds) {
    timespec_get(deadline, TIME_UTC);
    deadline->tv_sec += seconds;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int submitters_ready;
    int submitters_released;
    int first_callbacks;
    int timed_out;
} SubmitGate;

typedef struct {
    VxKernelThreadPool* pool;
    SubmitGate* gate;
    Coverage coverage;
    atomic_int first_callback;
} Submission;

static void concurrent_worker(void* opaque, int begin, int end) {
    Submission* submission = (Submission*)opaque;
    if (!atomic_exchange_explicit(&submission->first_callback, 1,
                                  memory_order_relaxed)) {
        struct timespec deadline;
        int status = 0;
        pthread_mutex_lock(&submission->gate->mutex);
        submission->gate->first_callbacks++;
        pthread_cond_broadcast(&submission->gate->condition);
        deadline_after(&deadline, 5);
        while (submission->gate->first_callbacks < 2 &&
               status != ETIMEDOUT)
            status = pthread_cond_timedwait(&submission->gate->condition,
                                            &submission->gate->mutex,
                                            &deadline);
        if (submission->gate->first_callbacks < 2)
            submission->gate->timed_out = 1;
        pthread_cond_broadcast(&submission->gate->condition);
        pthread_mutex_unlock(&submission->gate->mutex);
    }
    cover_range(&submission->coverage, begin, end);
}

static void* submit_concurrently(void* opaque) {
    Submission* submission = (Submission*)opaque;
    VxKernelThreadPoolScope scope =
        vx_kernel_thread_pool_scope_enter(submission->pool);
    pthread_mutex_lock(&submission->gate->mutex);
    submission->gate->submitters_ready++;
    pthread_cond_broadcast(&submission->gate->condition);
    while (!submission->gate->submitters_released)
        pthread_cond_wait(&submission->gate->condition,
                          &submission->gate->mutex);
    pthread_mutex_unlock(&submission->gate->mutex);
    vx_kernels_parallel_for(TASKS, 1, concurrent_worker, submission);
    vx_kernel_thread_pool_scope_leave(scope);
    return NULL;
}

static int test_concurrent_submit_exact(VxKernelThreadPool* pool) {
    SubmitGate gate = {0};
    Submission first = {0};
    Submission second = {0};
    pthread_t threads[2];
    struct timespec deadline;
    int status = 0;
    int created[2] = {0, 0};
    int valid;
    if (pthread_mutex_init(&gate.mutex, NULL) != 0) return 0;
    if (pthread_cond_init(&gate.condition, NULL) != 0) {
        pthread_mutex_destroy(&gate.mutex);
        return 0;
    }
    first.pool = second.pool = pool;
    first.gate = second.gate = &gate;
    first.coverage.total = second.coverage.total = TASKS;
    created[0] = pthread_create(
        &threads[0], NULL, submit_concurrently, &first) == 0;
    created[1] = created[0] && pthread_create(
        &threads[1], NULL, submit_concurrently, &second) == 0;
    if (!created[0] || !created[1]) {
        pthread_mutex_lock(&gate.mutex);
        gate.submitters_released = 1;
        pthread_cond_broadcast(&gate.condition);
        pthread_mutex_unlock(&gate.mutex);
        if (created[0]) pthread_join(threads[0], NULL);
        if (created[1]) pthread_join(threads[1], NULL);
        pthread_cond_destroy(&gate.condition);
        pthread_mutex_destroy(&gate.mutex);
        return 0;
    }
    pthread_mutex_lock(&gate.mutex);
    deadline_after(&deadline, 5);
    while (gate.submitters_ready < 2 && status != ETIMEDOUT)
        status = pthread_cond_timedwait(&gate.condition, &gate.mutex,
                                        &deadline);
    valid = gate.submitters_ready == 2;
    gate.submitters_released = 1;
    pthread_cond_broadcast(&gate.condition);
    pthread_mutex_unlock(&gate.mutex);
    if (pthread_join(threads[0], NULL) != 0) valid = 0;
    if (pthread_join(threads[1], NULL) != 0) valid = 0;
    valid = valid && gate.first_callbacks == 2 && !gate.timed_out &&
        coverage_is_exact(&first.coverage) &&
        coverage_is_exact(&second.coverage);
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    return valid;
}

typedef struct {
    atomic_uint outer[OUTER_TASKS];
    atomic_uint inner[INNER_TASKS];
    atomic_uint inner_callbacks;
    atomic_int invalid_width;
} Nested;

static void nested_inner(void* opaque, int begin, int end) {
    Nested* nested = (Nested*)opaque;
    if (vx_kernels_thread_count() != 1)
        atomic_store(&nested->invalid_width, 1);
    /* Lifecycle calls from a callback must neither deadlock nor change the
     * enclosing owner's configuration, on the caller lane or a worker. */
    vx_set_num_threads(1);
    vx_kernels_shutdown();
    atomic_fetch_add_explicit(&nested->inner_callbacks, 1u,
                              memory_order_relaxed);
    for (int index = begin; index < end; index++)
        atomic_fetch_add_explicit(&nested->inner[index], 1u,
                                  memory_order_relaxed);
}

static void nested_outer(void* opaque, int begin, int end) {
    Nested* nested = (Nested*)opaque;
    for (int index = begin; index < end; index++) {
        atomic_fetch_add_explicit(&nested->outer[index], 1u,
                                  memory_order_relaxed);
        vx_kernels_parallel_for(INNER_TASKS, 1, nested_inner, nested);
    }
}

static int test_nested_serial_fallback(VxKernelThreadPool* pool) {
    Nested nested = {0};
    VxKernelThreadPoolScope scope = vx_kernel_thread_pool_scope_enter(pool);
    vx_kernels_parallel_for(OUTER_TASKS, 1, nested_outer, &nested);
    vx_kernel_thread_pool_scope_leave(scope);
    for (int index = 0; index < OUTER_TASKS; index++)
        CHECK(atomic_load_explicit(&nested.outer[index],
                                   memory_order_relaxed) == 1u);
    for (int index = 0; index < INNER_TASKS; index++)
        CHECK(atomic_load_explicit(&nested.inner[index],
                                   memory_order_relaxed) == OUTER_TASKS);
    CHECK(atomic_load_explicit(&nested.inner_callbacks,
                               memory_order_relaxed) == OUTER_TASKS);
    CHECK(!atomic_load(&nested.invalid_width));
    scope = vx_kernel_thread_pool_scope_enter(pool);
    int width = vx_kernels_thread_count();
    vx_kernel_thread_pool_scope_leave(scope);
    CHECK(width == 4);
    return 1;
}

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered;
    int released;
    Coverage coverage;
} DrainGate;

typedef struct {
    VxKernelThreadPool* pool;
    DrainGate* gate;
    int before_admission;
} DrainSubmission;

typedef struct {
    VxKernelThreadPool* pool;
    atomic_int started;
    atomic_int completed;
} DestroySubmission;

static void drain_worker(void* opaque, int begin, int end) {
    DrainGate* gate = (DrainGate*)opaque;
    pthread_mutex_lock(&gate->mutex);
    if (!gate->entered) {
        gate->entered = 1;
        pthread_cond_broadcast(&gate->condition);
        while (!gate->released)
            pthread_cond_wait(&gate->condition, &gate->mutex);
    }
    pthread_mutex_unlock(&gate->mutex);
    cover_range(&gate->coverage, begin, end);
}

static void* submit_drain_job(void* opaque) {
    DrainSubmission* submission = (DrainSubmission*)opaque;
    VxKernelThreadPoolScope scope =
        vx_kernel_thread_pool_scope_enter(submission->pool);
    if (submission->before_admission) {
        pthread_mutex_lock(&submission->gate->mutex);
        submission->gate->entered = 1;
        pthread_cond_broadcast(&submission->gate->condition);
        while (!submission->gate->released)
            pthread_cond_wait(&submission->gate->condition,
                              &submission->gate->mutex);
        pthread_mutex_unlock(&submission->gate->mutex);
    }
    vx_kernels_parallel_for(TASKS, 1, drain_worker, submission->gate);
    vx_kernel_thread_pool_scope_leave(scope);
    return NULL;
}

static void* destroy_pool(void* opaque) {
    DestroySubmission* submission = (DestroySubmission*)opaque;
    atomic_store_explicit(&submission->started, 1, memory_order_release);
    vx_kernel_thread_pool_destroy(submission->pool);
    atomic_store_explicit(&submission->completed, 1, memory_order_release);
    return NULL;
}

static int test_destroy_drains(int before_admission) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    DrainGate gate = {0};
    DrainSubmission job = {pool, &gate, before_admission};
    DestroySubmission destroy = {.pool = pool};
    pthread_t job_thread;
    pthread_t destroy_thread;
    struct timespec deadline;
    struct timespec pause = {0, 20 * 1000 * 1000};
    int status = 0;
    int valid;
    if (!pool || pthread_mutex_init(&gate.mutex, NULL) != 0) {
        vx_kernel_thread_pool_destroy(pool);
        return 0;
    }
    if (pthread_cond_init(&gate.condition, NULL) != 0) {
        pthread_mutex_destroy(&gate.mutex);
        vx_kernel_thread_pool_destroy(pool);
        return 0;
    }
    gate.coverage.total = TASKS;
    if (pthread_create(&job_thread, NULL, submit_drain_job, &job) != 0) {
        pthread_cond_destroy(&gate.condition);
        pthread_mutex_destroy(&gate.mutex);
        vx_kernel_thread_pool_destroy(pool);
        return 0;
    }
    pthread_mutex_lock(&gate.mutex);
    deadline_after(&deadline, 5);
    while (!gate.entered && status != ETIMEDOUT)
        status = pthread_cond_timedwait(&gate.condition, &gate.mutex,
                                        &deadline);
    valid = gate.entered;
    pthread_mutex_unlock(&gate.mutex);
    if (!valid || pthread_create(
            &destroy_thread, NULL, destroy_pool, &destroy) != 0) {
        pthread_mutex_lock(&gate.mutex);
        gate.released = 1;
        pthread_cond_broadcast(&gate.condition);
        pthread_mutex_unlock(&gate.mutex);
        pthread_join(job_thread, NULL);
        vx_kernel_thread_pool_destroy(pool);
        pthread_cond_destroy(&gate.condition);
        pthread_mutex_destroy(&gate.mutex);
        return 0;
    }
    while (!atomic_load_explicit(&destroy.started, memory_order_acquire)) { }
    nanosleep(&pause, NULL);
    valid = !atomic_load_explicit(&destroy.completed, memory_order_acquire);
    pthread_mutex_lock(&gate.mutex);
    gate.released = 1;
    pthread_cond_broadcast(&gate.condition);
    pthread_mutex_unlock(&gate.mutex);
    if (pthread_join(job_thread, NULL) != 0) valid = 0;
    if (pthread_join(destroy_thread, NULL) != 0) valid = 0;
    valid = valid &&
        atomic_load_explicit(&destroy.completed, memory_order_acquire) &&
        coverage_is_exact(&gate.coverage);
    pthread_cond_destroy(&gate.condition);
    pthread_mutex_destroy(&gate.mutex);
    return valid;
}

typedef struct {
    Coverage coverage;
    pthread_t owner;
    int callbacks;
    int wrong_thread;
} Unbound;

static void unbound_worker(void* opaque, int begin, int end) {
    Unbound* check = (Unbound*)opaque;
    check->callbacks++;
    check->wrong_thread |= !pthread_equal(check->owner, pthread_self());
    cover_range(&check->coverage, begin, end);
}

static int test_unbound_is_serial(void) {
    Unbound check = {.coverage.total = TASKS, .owner = pthread_self()};
    CHECK(vx_kernels_thread_count() == 1);
    vx_set_num_threads(16);
    CHECK(vx_kernels_thread_count() == 1);
    vx_kernels_parallel_for(0, 1, unbound_worker, &check);
    vx_kernels_parallel_for(TASKS, 1, NULL, &check);
    CHECK(check.callbacks == 0);
    vx_kernels_parallel_for(TASKS, 1, unbound_worker, &check);
    CHECK(check.callbacks == 1 && !check.wrong_thread);
    CHECK(coverage_is_exact(&check.coverage));
    vx_kernels_shutdown();
    CHECK(vx_kernels_thread_count() == 1);
    return 1;
}

typedef struct {
    VxKernelThreadPool* pool;
    atomic_int invalid;
    atomic_int active;
    atomic_int peak;
    int iterations;
} Stress;

typedef struct {
    Coverage coverage;
    Stress* stress;
    float values[TASKS];
    float output[TASKS];
} NumericJob;

static void numeric_worker(void* opaque, int begin, int end) {
    NumericJob* job = (NumericJob*)opaque;
    int active = atomic_fetch_add(&job->stress->active, 1) + 1;
    int peak = atomic_load(&job->stress->peak);
    while (peak < active &&
           !atomic_compare_exchange_weak(&job->stress->peak, &peak, active)) { }
    cover_range(&job->coverage, begin, end);
    for (int i = begin; i < end; i++)
        job->output[i] = job->values[i] * 0.5f + 3.0f;
    sched_yield();
    atomic_fetch_sub(&job->stress->active, 1);
}

static void* stress_submit(void* opaque) {
    Stress* stress = (Stress*)opaque;
    VxKernelThreadPoolScope scope =
        vx_kernel_thread_pool_scope_enter(stress->pool);
    for (int iteration = 0; iteration < stress->iterations; iteration++) {
        NumericJob job = {.coverage.total = TASKS, .stress = stress};
        for (int i = 0; i < TASKS; i++) job.values[i] = (float)(i - iteration);
        vx_kernels_parallel_for(TASKS, iteration % 9, numeric_worker, &job);
        if (!coverage_is_exact(&job.coverage)) atomic_store(&stress->invalid, 1);
        for (int i = 0; i < TASKS; i++)
            if (job.output[i] != job.values[i] * 0.5f + 3.0f)
                atomic_store(&stress->invalid, 1);
    }
    vx_kernel_thread_pool_scope_leave(scope);
    return NULL;
}

static void* stress_restart(void* opaque) {
    Stress* stress = (Stress*)opaque;
    const int widths[] = {1, 4, 0, 2, 99};
    VxKernelThreadPoolScope scope =
        vx_kernel_thread_pool_scope_enter(stress->pool);
    for (int i = 0; i < 30; i++) {
        int requested = widths[i % 5];
        vx_set_num_threads(requested);
        int width = vx_kernels_thread_count();
        if (width < 1 || width > 16 ||
            (requested > 0 && width != (requested < 16 ? requested : 16)))
            atomic_store(&stress->invalid, 1);
        size_t workers = 0, heap = 0, stack = 0;
        if (vx_kernel_thread_pool_resource_bound(
                stress->pool, &heap, &workers, &stack) != 0 || !heap ||
            workers != (size_t)(requested <= 0 || requested > 16
                               ? 15 : requested - 1) ||
            (workers && !stack) || (!workers && stack))
            atomic_store(&stress->invalid, 1);
        vx_kernels_shutdown();
    }
    vx_kernel_thread_pool_scope_leave(scope);
    return NULL;
}

static int test_shared_and_independent_owners(void) {
    VxKernelThreadPool* first = vx_kernel_thread_pool_create(1);
    VxKernelThreadPool* second = vx_kernel_thread_pool_create(4);
    Stress shared = {.pool = first, .iterations = 120};
    Stress independent = {.pool = second, .iterations = 120};
    pthread_t threads[4];
    CHECK(first && second);
    CHECK(pthread_create(&threads[0], NULL, stress_submit, &shared) == 0);
    CHECK(pthread_create(&threads[1], NULL, stress_submit, &shared) == 0);
    CHECK(pthread_create(&threads[2], NULL, stress_restart, &shared) == 0);
    CHECK(pthread_create(&threads[3], NULL, stress_submit, &independent) == 0);
    for (int i = 0; i < 4; i++) CHECK(pthread_join(threads[i], NULL) == 0);
    CHECK(!atomic_load(&shared.invalid) && !atomic_load(&independent.invalid));
    CHECK(atomic_load(&shared.peak) <= 16 && atomic_load(&shared.active) == 0);
    CHECK(atomic_load(&independent.peak) <= 4 && atomic_load(&independent.active) == 0);
    VxKernelThreadPoolScope a = vx_kernel_thread_pool_scope_enter(first);
    VxKernelThreadPoolScope b = vx_kernel_thread_pool_scope_enter(second);
    CHECK(vx_kernels_thread_count() == 4);
    vx_kernel_thread_pool_scope_leave(b);
    CHECK(vx_kernels_thread_count() == 16);
    vx_kernel_thread_pool_scope_leave(a);
    CHECK(vx_kernels_thread_count() == 1);
    vx_kernel_thread_pool_destroy(first);
    vx_kernel_thread_pool_destroy(second);
    return 1;
}

static int test_caller_handoff_respects_width(void) {
    const int widths[] = {1, 2, 4};
    for (unsigned test = 0; test < sizeof(widths) / sizeof(widths[0]); test++) {
        VxKernelThreadPool* pool = vx_kernel_thread_pool_create(widths[test]);
        Stress shared = {.pool = pool, .iterations = 48};
        pthread_t threads[4];
        CHECK(pool);
        for (int i = 0; i < 4; i++)
            CHECK(pthread_create(&threads[i], NULL, stress_submit, &shared) == 0);
        for (int i = 0; i < 4; i++) CHECK(pthread_join(threads[i], NULL) == 0);
        CHECK(!atomic_load(&shared.invalid));
        CHECK(atomic_load(&shared.active) == 0);
        CHECK(atomic_load(&shared.peak) <= widths[test]);
        vx_kernel_thread_pool_destroy(pool);
    }
    return 1;
}

static int test_automatic_affinity_is_per_pool(void) {
#if defined(__linux__)
    cpu_set_t original, single;
    int cpus[2] = {-1, -1};
    if (sched_getaffinity(0, sizeof(original), &original) != 0) {
        puts("SKIP automatic affinity: sched_getaffinity unavailable");
        return 1;
    }
    for (int cpu = 0; cpu < CPU_SETSIZE; cpu++)
        if (CPU_ISSET(cpu, &original)) {
            if (cpus[0] < 0) cpus[0] = cpu;
            else { cpus[1] = cpu; break; }
        }
    if (cpus[1] < 0) {
        puts("SKIP automatic affinity: fewer than two allowed CPUs");
        return 1;
    }
    const char* requested = getenv("VOLVOXAI_THREADS");
    char* saved = requested ? strdup(requested) : NULL;
    if (requested && !saved) return 0;
    unsetenv("VOLVOXAI_THREADS");
    VxKernelThreadPool* wide = vx_kernel_thread_pool_create(0);
    VxKernelThreadPool* pinned[2] = {NULL, NULL};
    int valid = wide != NULL;
    int wide_width = 0;
    if (wide) {
        VxKernelThreadPoolScope scope = vx_kernel_thread_pool_scope_enter(wide);
        wide_width = vx_kernels_thread_count();
        vx_kernel_thread_pool_scope_leave(scope);
    }
    for (int i = 0; i < 2 && valid; i++) {
        CPU_ZERO(&single);
        CPU_SET(cpus[i], &single);
        if (sched_setaffinity(0, sizeof(single), &single) != 0) {
            puts("SKIP automatic affinity: sched_setaffinity unavailable");
            break;
        }
        pinned[i] = vx_kernel_thread_pool_create(0);
        if (!pinned[i]) { valid = 0; break; }
        VxKernelThreadPoolScope scope = vx_kernel_thread_pool_scope_enter(pinned[i]);
        int width = vx_kernels_thread_count();
        vx_kernel_thread_pool_scope_leave(scope);
        if (width != 1) {
            fprintf(stderr, "automatic affinity CPU %d: width=%d expected=1\n", cpus[i], width);
            valid = 0;
        }
    }
    if (sched_setaffinity(0, sizeof(original), &original) != 0) valid = 0;
    if (valid && pinned[0]) {
        VxKernelThreadPoolScope scope = vx_kernel_thread_pool_scope_enter(pinned[0]);
        vx_set_num_threads(0);
        valid = vx_kernels_thread_count() == wide_width;
        vx_kernel_thread_pool_scope_leave(scope);
    }
    vx_kernel_thread_pool_destroy(wide);
    vx_kernel_thread_pool_destroy(pinned[0]);
    vx_kernel_thread_pool_destroy(pinned[1]);
    if (saved) { setenv("VOLVOXAI_THREADS", saved, 1); free(saved); }
    else unsetenv("VOLVOXAI_THREADS");
    return valid;
#else
    puts("SKIP automatic affinity: physical-core affinity policy unavailable");
    return 1;
#endif
}

int main(void) {
    if (!test_unbound_is_serial()) return 1;
    if (!test_automatic_affinity_is_per_pool()) return 1;
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    if (!pool) return 1;
    if (!test_concurrent_submit_exact(pool) ||
        !test_nested_serial_fallback(pool)) {
        vx_kernel_thread_pool_destroy(pool);
        return 1;
    }
    vx_kernel_thread_pool_destroy(pool);
    if (!test_destroy_drains(0) || !test_destroy_drains(1) ||
        !test_shared_and_independent_owners() ||
        !test_caller_handoff_respects_width()) return 1;
    puts("thread-pool lifecycle contract passed");
    return 0;
}
