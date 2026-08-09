#include "runtime_state.h"

#include <assert.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local VxEngineState* g_current_engine_state;
static _Thread_local VxEngineState* g_route_lease_owner;

int vx_engine_state_init(VxEngineState* state) {
    if (!state) return -1;
    memset(state, 0, sizeof(*state));
    state->active_row = -1;
    state->execution_row = -1;
    state->tensor_name_index_count = -1;
    state->incremental_hybrid_prepared_row = -1;
    state->last_failure_node_index = -1;
    state->model_generation = 1;
    {
        const char* value = getenv("VOLVOX_PW_GEMM");
        state->conv_pw_gemm_enabled =
            !(value && value[0] && strcmp(value, "0") == 0);
    }
    {
        /* Default on, and the shape decides the rest.
         *
         * This was default off, from a comment citing three image-encoder
         * shapes where the packed kernel measured 8.16 GMAC/s against 20.38 for
         * unpacked.  That measurement was real but the conclusion did not
         * generalize: it inverted at LLM shapes, where the unpacked kernel
         * collapses to 4-7 GMAC/s because it has no cache blocking and rereads
         * the whole weight per row block, while packed stayed flat.  A
         * process-wide boolean cannot express that, which is why the packed
         * kernel now carries a plan (vx_gemm_f32_plan) and picks its blocking
         * and regime from M, K and N.  Measured after that change, one thread:
         * 402x320x320 42.1 -> 56.6, 512x2048x2048 7.6 -> 52.6, 1x4096x4096
         * 1.2 -> 4.2 GMAC/s, bit-identical to matmul_f32 throughout.
         *
         * The switch remains as an escape hatch to the unpacked kernel, not as
         * the place the policy lives. */
        const char* value = getenv("VOLVOX_F32_GEMM_PACKED");
        state->gemm_f32_packed_enabled =
            !(value && value[0] && strcmp(value, "0") == 0);
    }
#if VOLVOXAI_ENABLE_TRAINING
    state->dynamic_autograd_forward_backend = -1;
#endif
    if (pthread_mutex_init(&state->adapter_admin_mutex, NULL) != 0)
        return -1;
    if (pthread_mutex_init(&state->model_mutex, NULL) != 0) {
        pthread_mutex_destroy(&state->adapter_admin_mutex);
        return -1;
    }
    if (pthread_mutex_init(&state->metadata_mutex, NULL) != 0) {
        pthread_mutex_destroy(&state->model_mutex);
        pthread_mutex_destroy(&state->adapter_admin_mutex);
        return -1;
    }
    state->kernel_thread_pool = vx_kernel_thread_pool_create(0);
    if (!state->kernel_thread_pool) {
        pthread_mutex_destroy(&state->metadata_mutex);
        pthread_mutex_destroy(&state->model_mutex);
        pthread_mutex_destroy(&state->adapter_admin_mutex);
        return -1;
    }
    return 0;
}

void vx_engine_state_deinit(VxEngineState* state) {
    if (!state) return;
    assert(!state->loaded);
    assert(!state->graph_root);
    assert(!state->merged_adapter_weights);
    /* Shutdown normally owns this cleanup. Keep deinit complete for an engine
     * whose load was abandoned after residency registration but before init. */
    for (size_t index = 0; index < state->bank_residency_count; index++)
        free(state->bank_residency[index].slot_rows);
    free(state->bank_residency);
    state->bank_residency = NULL;
    state->bank_residency_count = 0;
    state->bank_residency_capacity = 0;
    if (state->adapter_registry_state_destroy)
        state->adapter_registry_state_destroy(state->adapter_registry_state);
    state->adapter_registry_state = NULL;
    state->adapter_registry_state_destroy = NULL;
    if (state->vulkan_context_state_destroy)
        state->vulkan_context_state_destroy(state->vulkan_context_state);
    state->vulkan_context_state = NULL;
    state->vulkan_context_state_destroy = NULL;
    if (state->opengl_context_state_destroy)
        state->opengl_context_state_destroy(state->opengl_context_state);
    state->opengl_context_state = NULL;
    state->opengl_context_state_destroy = NULL;
    if (state->metal_context_state_destroy)
        state->metal_context_state_destroy(state->metal_context_state);
    state->metal_context_state = NULL;
    state->metal_context_state_destroy = NULL;
    if (state->nnapi_context_state_destroy)
        state->nnapi_context_state_destroy(state->nnapi_context_state);
    state->nnapi_context_state = NULL;
    state->nnapi_context_state_destroy = NULL;
    if (state->cuda_context_state_destroy)
        state->cuda_context_state_destroy(state->cuda_context_state);
    state->cuda_context_state = NULL;
    state->cuda_context_state_destroy = NULL;
    /* Ordinary shutdown releases this first. Keep deinit defensive for a
     * partially initialized context whose graph load never reached shutdown. */
    free(state->cpu_typed_workspace);
    state->cpu_typed_workspace = NULL;
    state->cpu_typed_workspace_bound_bytes = 0u;
    state->cpu_typed_workspace_capacity_bytes = 0u;
    state->cpu_typed_workspace_configured = 0;
    vx_kernel_thread_pool_destroy(state->kernel_thread_pool);
    state->kernel_thread_pool = NULL;
    pthread_mutex_destroy(&state->metadata_mutex);
    pthread_mutex_destroy(&state->model_mutex);
    pthread_mutex_destroy(&state->adapter_admin_mutex);
}

VxEngineState* vx_engine_state_current(void) {
    return g_current_engine_state;
}

VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state) {
    assert(state);
    VxEngineStateScope scope = {
        .previous = g_current_engine_state,
        .bound = state,
        .kernel_pool_scope = vx_kernel_thread_pool_scope_enter(
            state->kernel_thread_pool),
    };
    assert(!g_route_lease_owner || g_route_lease_owner == scope.bound);
    g_current_engine_state = scope.bound;
    return scope;
}

void vx_engine_state_scope_leave(VxEngineStateScope scope) {
    assert(g_current_engine_state == scope.bound);
    assert(!g_route_lease_owner || g_route_lease_owner == scope.previous);
    vx_kernel_thread_pool_scope_leave(scope.kernel_pool_scope);
    g_current_engine_state = scope.previous;
}

int vx_engine_state_route_lease_active(void) {
    return g_route_lease_owner &&
        g_route_lease_owner == vx_engine_state_current();
}

int vx_engine_state_route_lease_begin(void) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || g_route_lease_owner) return -1;
    g_route_lease_owner = state;
    return 0;
}

void vx_engine_state_route_lease_end(void) {
    assert(g_route_lease_owner == vx_engine_state_current());
    g_route_lease_owner = NULL;
}
