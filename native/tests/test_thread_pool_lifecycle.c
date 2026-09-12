#include "thread_pool.h"

#include <errno.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
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
} Nested;

static void nested_inner(void* opaque, int begin, int end) {
    Nested* nested = (Nested*)opaque;
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

static int test_destroy_drains(void) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    DrainGate gate = {0};
    DrainSubmission job = {pool, &gate};
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

int main(void) {
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(4);
    if (!pool) return 1;
    if (!test_concurrent_submit_exact(pool) ||
        !test_nested_serial_fallback(pool)) {
        vx_kernel_thread_pool_destroy(pool);
        return 1;
    }
    vx_kernel_thread_pool_destroy(pool);
    if (!test_destroy_drains()) return 1;
    puts("thread-pool lifecycle contract passed");
    return 0;
}
