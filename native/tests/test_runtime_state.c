#include "runtime_state.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

typedef struct {
    VxEngineState* state;
    int ok;
} ThreadProbe;

static void* probe_thread_state(void* opaque) {
    ThreadProbe* probe = (ThreadProbe*)opaque;
    if (vx_engine_state_current() != NULL) return NULL;
    VxEngineStateScope scope = vx_engine_state_scope_enter(probe->state);
    if (vx_engine_state_current() == probe->state &&
        probe->state->tensor_count == 23)
        probe->ok = 1;
    vx_engine_state_scope_leave(scope);
    return NULL;
}

int main(void) {
    VxEngineState* first = (VxEngineState*)calloc(1, sizeof(*first));
    VxEngineState* second = (VxEngineState*)calloc(1, sizeof(*second));
    CHECK(first && second);
    CHECK(vx_engine_state_init(first) == 0);
    CHECK(vx_engine_state_init(second) == 0);

    CHECK(vx_engine_state_current() == NULL);
    CHECK(first->active_row == -1);
    CHECK(first->execution_row == -1);
    CHECK(first->model_generation == 1);

    first->tensor_count = 11;
    second->tensor_count = 23;
    VxEngineStateScope first_scope = vx_engine_state_scope_enter(first);
    CHECK(vx_engine_state_current() == first);
    CHECK(vx_engine_state_current()->tensor_count == 11);
    CHECK(vx_engine_state_route_lease_begin() == 0);
    CHECK(vx_engine_state_route_lease_active());
    CHECK(vx_engine_state_route_lease_begin() != 0);
    vx_engine_state_route_lease_end();

    VxEngineStateScope second_scope = vx_engine_state_scope_enter(second);
    CHECK(vx_engine_state_current() == second);
    CHECK(vx_engine_state_current()->tensor_count == 23);
    vx_engine_state_scope_leave(second_scope);
    CHECK(vx_engine_state_current() == first);
    vx_engine_state_scope_leave(first_scope);
    CHECK(vx_engine_state_current() == NULL);

    ThreadProbe probe = { .state = second, .ok = 0 };
    pthread_t thread;
    CHECK(pthread_create(&thread, NULL, probe_thread_state, &probe) == 0);
    CHECK(pthread_join(thread, NULL) == 0);
    CHECK(probe.ok);
    CHECK(vx_engine_state_current() == NULL);

    pthread_mutex_lock(&first->model_mutex);
    pthread_mutex_unlock(&first->model_mutex);
    vx_engine_state_deinit(second);
    vx_engine_state_deinit(first);
    free(second);
    free(first);
    puts("native runtime state tests passed");
    return 0;
}
