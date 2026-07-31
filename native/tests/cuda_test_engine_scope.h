#ifndef VOLVOXAI_CUDA_TEST_ENGINE_SCOPE_H
#define VOLVOXAI_CUDA_TEST_ENGINE_SCOPE_H

#include "runtime_state.h"

#include <stdlib.h>

typedef struct {
    VxEngineState* engine;
    VxEngineStateScope scope;
    int active;
} CudaTestEngineScope;

static int cuda_test_engine_scope_begin(CudaTestEngineScope* owner) {
    if (!owner) return -1;
    owner->engine = (VxEngineState*)calloc(1, sizeof(*owner->engine));
    if (!owner->engine || vx_engine_state_init(owner->engine) != 0) {
        free(owner->engine);
        owner->engine = NULL;
        return -1;
    }
    owner->scope = vx_engine_state_scope_enter(owner->engine);
    owner->active = 1;
    return 0;
}

static void cuda_test_engine_scope_end(CudaTestEngineScope* owner) {
    if (!owner || !owner->active) return;
    vx_engine_state_scope_leave(owner->scope);
    vx_engine_state_deinit(owner->engine);
    free(owner->engine);
    owner->engine = NULL;
    owner->active = 0;
}

#endif
