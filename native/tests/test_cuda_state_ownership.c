#include "cuda_engine.h"
#include "runtime_state.h"

#include <pthread.h>
#include <sched.h>
#include <stdatomic.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", \
                __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

typedef struct {
    VxEngineState* engine;
    uint64_t probe;
    uintptr_t identity;
    _Atomic int* ready;
    int ok;
} OwnershipThread;

static void* run_owner(void* opaque) {
    OwnershipThread* thread = (OwnershipThread*)opaque;
    VxEngineStateScope scope = vx_engine_state_scope_enter(thread->engine);
    thread->identity = cuda_test_context_state_identity();
    cuda_test_context_state_set_probe(thread->probe);
    (void)atomic_fetch_add_explicit(thread->ready, 1, memory_order_release);
    while (atomic_load_explicit(thread->ready, memory_order_acquire) != 2)
        sched_yield();
    thread->ok = thread->identity != 0u &&
        cuda_test_context_state_identity() == thread->identity &&
        cuda_test_context_state_probe() == thread->probe;
    vx_engine_state_scope_leave(scope);
    return NULL;
}

int main(void) {
    VxEngineState* first = (VxEngineState*)calloc(1, sizeof(*first));
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    _Atomic int ready = 0;
    OwnershipThread owners[2] = {
        {first, UINT64_C(0x1111222233334444), 0u, &ready, 0},
        {second, UINT64_C(0xaaaabbbbccccdddd), 0u, &ready, 0},
    };
    pthread_t threads[2];
    CHECK(first && second);
    CHECK(vx_engine_state_init(first) == 0);
    CHECK(vx_engine_state_init(second) == 0);
    {
        const uint8_t host_value = 7u;
        VxEngineStateScope scope = vx_engine_state_scope_enter(first);
        CHECK(first->cuda_context_state == NULL);
        cuda_graph_reset();
        CHECK(cuda_graph_sync_host(&host_value, sizeof(host_value), 0));
        CHECK(first->cuda_context_state == NULL);
        vx_engine_state_scope_leave(scope);
    }
    CHECK(pthread_create(&threads[0], NULL, run_owner, &owners[0]) == 0);
    CHECK(pthread_create(&threads[1], NULL, run_owner, &owners[1]) == 0);
    CHECK(pthread_join(threads[0], NULL) == 0);
    CHECK(pthread_join(threads[1], NULL) == 0);
    CHECK(owners[0].ok && owners[1].ok);
    CHECK(owners[0].identity != owners[1].identity);
    vx_engine_state_deinit(second);
    vx_engine_state_deinit(first);
    free(second);
    free(first);
    puts("CUDA per-engine state ownership test passed");
    return 0;
}
