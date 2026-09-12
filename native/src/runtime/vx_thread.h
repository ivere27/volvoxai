#ifndef VOLVOXAI_RUNTIME_VX_THREAD_H
#define VOLVOXAI_RUNTIME_VX_THREAD_H

/*
 * One mutual-exclusion vocabulary for every target the engine runs on.
 *
 * The engine's own state (model, metadata, adapter admin, backend registry)
 * is guarded by mutexes that only ever protect against other threads in the
 * same process. A wasm32 build has no such threads, so the same state machine
 * runs correctly with primitives that do nothing. That is the whole reason
 * this header exists: the runtime source stays identical across targets and
 * only the primitive changes, instead of the runtime being reimplemented for
 * the browser.
 *
 * `vx_kernel_thread_pool` already follows this shape for data parallelism --
 * its wasm path executes the range serially. This is the same idea for
 * mutual exclusion.
 *
 * Deliberately narrow: mutex and once, no condition variables and no thread
 * creation. A threadless target cannot honour a condition wait, so a caller
 * that needs one must not be reachable there.
 */

#if !defined(VOLVOXAI_NO_THREADS)
#if defined(__wasm__) && !defined(__EMSCRIPTEN_PTHREADS__)
#define VOLVOXAI_NO_THREADS 1
#endif
#endif

#if defined(VOLVOXAI_NO_THREADS)

typedef unsigned char VxMutex;
typedef unsigned char VxOnce;
typedef unsigned char VxCond;

#define VX_MUTEX_INITIALIZER 0
#define VX_ONCE_INITIALIZER 0
#define VX_COND_INITIALIZER 0

static inline int vx_mutex_init(VxMutex* mutex) { *mutex = 0; return 0; }
static inline void vx_mutex_destroy(VxMutex* mutex) { (void)mutex; }
static inline void vx_mutex_lock(VxMutex* mutex) { (void)mutex; }
static inline void vx_mutex_unlock(VxMutex* mutex) { (void)mutex; }
/* Reports success like its pthread counterpart: an uncontended lock always
 * succeeds when no other thread can hold it. */
static inline int vx_mutex_trylock(VxMutex* mutex) { (void)mutex; return 0; }

static inline void vx_once(VxOnce* once, void (*routine)(void)) {
    if (!*once) {
        *once = 1u;
        routine();
    }
}

static inline int vx_cond_init(VxCond* cond) { *cond = 0; return 0; }
static inline void vx_cond_destroy(VxCond* cond) { (void)cond; }
static inline void vx_cond_signal(VxCond* cond) { (void)cond; }
static inline void vx_cond_broadcast(VxCond* cond) { (void)cond; }

/*
 * Deliberately no vx_cond_wait or pthread_cond_wait: threadless targets cannot
 * honour a condition wait. Callers must drive the manual poll core or branch.
 */

typedef VxMutex pthread_mutex_t;
typedef VxCond pthread_cond_t;
typedef int pthread_t;

static inline int pthread_equal(pthread_t a, pthread_t b) { return a == b; }
static inline pthread_t pthread_self(void) { return 0; }

#define PTHREAD_MUTEX_INITIALIZER VX_MUTEX_INITIALIZER
#define PTHREAD_COND_INITIALIZER VX_COND_INITIALIZER

static inline int pthread_mutex_init(pthread_mutex_t* m, const void* attr) { (void)attr; return vx_mutex_init(m); }
static inline int pthread_mutex_destroy(pthread_mutex_t* m) { vx_mutex_destroy(m); return 0; }
static inline int pthread_mutex_lock(pthread_mutex_t* m) { vx_mutex_lock(m); return 0; }
static inline int pthread_mutex_unlock(pthread_mutex_t* m) { vx_mutex_unlock(m); return 0; }
static inline int pthread_mutex_trylock(pthread_mutex_t* m) { return vx_mutex_trylock(m); }

static inline int pthread_cond_init(pthread_cond_t* c, const void* attr) { (void)attr; return vx_cond_init(c); }
static inline int pthread_cond_destroy(pthread_cond_t* c) { vx_cond_destroy(c); return 0; }
static inline int pthread_cond_signal(pthread_cond_t* c) { vx_cond_signal(c); return 0; }
static inline int pthread_cond_broadcast(pthread_cond_t* c) { vx_cond_broadcast(c); return 0; }

#else

#include <pthread.h>

typedef pthread_mutex_t VxMutex;
typedef pthread_once_t VxOnce;
typedef pthread_cond_t VxCond;

#define VX_MUTEX_INITIALIZER PTHREAD_MUTEX_INITIALIZER
#define VX_ONCE_INITIALIZER PTHREAD_ONCE_INIT
#define VX_COND_INITIALIZER PTHREAD_COND_INITIALIZER

static inline int vx_mutex_init(VxMutex* mutex) {
    return pthread_mutex_init(mutex, NULL);
}
static inline void vx_mutex_destroy(VxMutex* mutex) {
    (void)pthread_mutex_destroy(mutex);
}
static inline void vx_mutex_lock(VxMutex* mutex) {
    (void)pthread_mutex_lock(mutex);
}
static inline void vx_mutex_unlock(VxMutex* mutex) {
    (void)pthread_mutex_unlock(mutex);
}
static inline int vx_mutex_trylock(VxMutex* mutex) {
    return pthread_mutex_trylock(mutex);
}

static inline void vx_once(VxOnce* once, void (*routine)(void)) {
    (void)pthread_once(once, routine);
}

static inline int vx_cond_init(VxCond* cond) {
    return pthread_cond_init(cond, NULL);
}
static inline void vx_cond_destroy(VxCond* cond) {
    (void)pthread_cond_destroy(cond);
}
static inline void vx_cond_signal(VxCond* cond) {
    (void)pthread_cond_signal(cond);
}
static inline void vx_cond_broadcast(VxCond* cond) {
    (void)pthread_cond_broadcast(cond);
}
static inline int vx_cond_wait(VxCond* cond, VxMutex* mutex) {
    return pthread_cond_wait(cond, mutex);
}
static inline int vx_cond_timedwait(VxCond* cond, VxMutex* mutex, const struct timespec* abstime) {
    return pthread_cond_timedwait(cond, mutex, abstime);
}

#endif

#endif
