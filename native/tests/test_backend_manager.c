#include "backend_manager.h"
#include "runtime_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int g_init_result;
static int g_init_calls;
static int g_cleanup_calls;
static int g_opengl_init_result;
static int g_opengl_init_calls;
static int g_opengl_cleanup_calls;
static int g_cuda_init_result;
static int g_cuda_init_calls;
static int g_cuda_cleanup_calls;
static int g_device_release_calls;
static int g_opengl_device_release_calls;

int vk_init(void) {
    g_init_calls++;
    return g_init_result;
}

void vk_cleanup(void) {
    g_cleanup_calls++;
}

/* The shared device outlives individual contexts, so deactivation releases it
 * separately from the per-state cleanup. */
void vk_device_release(void) {
    g_device_release_calls++;
}

int opengl_init(void) {
    g_opengl_init_calls++;
    return g_opengl_init_result;
}

void opengl_cleanup(void) {
    g_opengl_cleanup_calls++;
}

void opengl_device_release(void) {
    g_opengl_device_release_calls++;
}

int cuda_init(void) {
    g_cuda_init_calls++;
    return g_cuda_init_result;
}

void cuda_cleanup(void) {
    g_cuda_cleanup_calls++;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    CHECK(VOLVOXAI_BACKEND_CUDA == 4);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);
    CHECK(vx_backend_manager_activate((VolvoxAIEngineBackend)5) == -1);

    g_init_result = -1;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_VULKAN) == -1);
    CHECK(g_init_calls == 1);
    CHECK(g_cleanup_calls == 1);
    /* Deactivation drops only the per-state context; the shared device is
     * released at process exit so compile and execute reuse one device. */
    CHECK(g_device_release_calls == 0);
    CHECK(state->use_vulkan == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    g_cuda_init_result = -1;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_CUDA) == -1);
    CHECK(g_cuda_init_calls == 1);
    CHECK(g_cuda_cleanup_calls == 1);
    CHECK(state->use_cuda == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    g_cuda_init_result = 0;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_CUDA) == 0);
    CHECK(g_cuda_init_calls == 2);
    CHECK(g_cuda_cleanup_calls == 1);
    CHECK(state->use_cuda == 1);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CUDA);
    CHECK(!strcmp(vx_backend_manager_name(), "CUDA"));
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_CUDA) == 0);
    CHECK(g_cuda_init_calls == 2);

    vx_backend_manager_deactivate();
    CHECK(g_cuda_cleanup_calls == 2);
    CHECK(state->use_cuda == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    g_init_result = 0;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_VULKAN) == 0);
    CHECK(g_init_calls == 2);
    CHECK(g_cleanup_calls == 1);
    CHECK(state->use_vulkan == 1);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_VULKAN);
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_VULKAN) == 0);
    CHECK(g_init_calls == 2);

    g_opengl_init_result = -1;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_OPENGL) == -1);
    CHECK(g_opengl_init_calls == 1);
    CHECK(g_opengl_cleanup_calls == 1);
    CHECK(g_cleanup_calls == 1);
    CHECK(state->use_vulkan == 1 && state->use_opengl == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_VULKAN);

    vx_backend_manager_deactivate();
    CHECK(g_cleanup_calls == 2);
    CHECK(g_device_release_calls == 0);
    CHECK(state->use_vulkan == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("native backend manager tests passed");
    return 0;
}
