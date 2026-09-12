#include "backend_manager.h"
#include "runtime_state.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", \
                __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

typedef struct BackendProbe {
    int init_result;
    int init_calls;
    int cleanup_calls;
    char tag;
} BackendProbe;

static BackendProbe probes[] = {
    [VX_PORTABLE_BACKEND_KIND] = {0, 0, 0, 'c'},
    [VX_BACKEND_KIND_VULKAN] = {0, 0, 0, 'v'},
    [VX_BACKEND_KIND_OPENGL] = {0, 0, 0, 'g'},
    [VX_BACKEND_KIND_METAL] = {0, 0, 0, 'm'},
    [VX_BACKEND_KIND_CUDA] = {0, 0, 0, 'u'},
};
static char events[128];
static size_t event_count;

static void record_event(char backend, char operation) {
    if (event_count + 2u > sizeof(events)) abort();
    events[event_count++] = backend;
    events[event_count++] = operation;
}

static int backend_init(VxBackendKind backend) {
    BackendProbe* probe = &probes[(int)backend];
    probe->init_calls++;
    record_event(probe->tag, 'i');
    return probe->init_result;
}

static void backend_cleanup(VxBackendKind backend) {
    BackendProbe* probe = &probes[(int)backend];
    probe->cleanup_calls++;
    record_event(probe->tag, 'c');
}

int vk_init(void) { return backend_init(VX_BACKEND_KIND_VULKAN); }
void vk_cleanup(void) { backend_cleanup(VX_BACKEND_KIND_VULKAN); }
int opengl_init(void) { return backend_init(VX_BACKEND_KIND_OPENGL); }
void opengl_cleanup(void) { backend_cleanup(VX_BACKEND_KIND_OPENGL); }
int metal_init(void) { return backend_init(VX_BACKEND_KIND_METAL); }
void metal_cleanup(void) { backend_cleanup(VX_BACKEND_KIND_METAL); }
int cuda_init(void) { return backend_init(VX_BACKEND_KIND_CUDA); }
void cuda_cleanup(void) { backend_cleanup(VX_BACKEND_KIND_CUDA); }

static int selected_flags_match(const VxEngineState* state,
                                VxBackendKind backend) {
    return state && state->backend == (int)backend &&
#if VOLVOXAI_ENABLE_WEBGPU
        state->use_webgpu == (backend == VX_BACKEND_KIND_WEBGPU) &&
#endif
        state->use_vulkan == (backend == VX_BACKEND_KIND_VULKAN) &&
        state->use_opengl == (backend == VX_BACKEND_KIND_OPENGL) &&
        state->use_metal == (backend == VX_BACKEND_KIND_METAL) &&
        state->use_cuda == (backend == VX_BACKEND_KIND_CUDA);
}

/*
 * Selection and reporting have to be one vocabulary.
 *
 * Restating the names here would only copy the table, so this asserts the
 * property the table exists for instead: the name the engine reports for a
 * backend is a name it accepts back, and it names that same backend. Three
 * separate spellings of this list is what let `webgpu` be reportable through
 * one entry point and unselectable through another.
 */
static int name_round_trips(VxBackendKind backend) {
    VxBackendKind parsed;
    const char* name = vx_backend_manager_name_of(backend);
    return name && name[0] &&
        vx_backend_manager_by_name(name, &parsed) && parsed == backend;
}

static int activate_and_check(VxEngineState* state,
                              VxBackendKind backend) {
    int init_before = probes[(int)backend].init_calls;
    int cleanup_before = probes[(int)backend].cleanup_calls;
    if (vx_backend_manager_activate(backend) != 0 ||
        probes[(int)backend].init_calls != init_before + 1 ||
        probes[(int)backend].cleanup_calls != cleanup_before ||
        vx_backend_manager_current() != backend ||
        !selected_flags_match(state, backend) ||
        strcmp(vx_backend_manager_name(),
               vx_backend_manager_name_of(backend)) != 0 ||
        !name_round_trips(backend))
        return 0;
    /* Re-selecting an active backend is side-effect-free. */
    if (vx_backend_manager_activate(backend) != 0 ||
        probes[(int)backend].init_calls != init_before + 1 ||
        probes[(int)backend].cleanup_calls != cleanup_before)
        return 0;
    return 1;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    const VxBackendKind gpu_backends[] = {
        VX_BACKEND_KIND_VULKAN,
        VX_BACKEND_KIND_OPENGL,
        VX_BACKEND_KIND_METAL,
        VX_BACKEND_KIND_CUDA,
    };

    CHECK(state != NULL);
    CHECK(vx_engine_state_init(state) == 0);
    scope = vx_engine_state_scope_enter(state);

    {
        /* Every row, including backends this build cannot run. The name space
         * is the same on every target; whether a backend is there to run is
         * what `activate` answers, and the two must not be conflated. */
        VxBackendKind row;
        VxBackendKind parsed;
        for (row = VX_PORTABLE_BACKEND_KIND; row <= VX_BACKEND_KIND_WEBGPU; ++row)
            CHECK(name_round_trips(row));
        CHECK(!vx_backend_manager_by_name("not-a-backend", &parsed));
        CHECK(!vx_backend_manager_by_name(NULL, &parsed));
        CHECK(!strcmp(vx_backend_manager_name_of(VX_PORTABLE_BACKEND_KIND),
                      VX_PORTABLE_BACKEND_NAME));
    }
    CHECK(vx_backend_manager_current() == VX_PORTABLE_BACKEND_KIND);
    CHECK(selected_flags_match(state, VX_PORTABLE_BACKEND_KIND));
    CHECK(strcmp(vx_backend_manager_name(), VX_PORTABLE_BACKEND_NAME) == 0);

    /* Every optional backend must fail closed, clean its partial attempt, and
     * leave the selected backend and flags untouched. The same backend must
     * then be reusable without recreating the engine state. */
    for (size_t index = 0;
         index < sizeof(gpu_backends) / sizeof(gpu_backends[0]); index++) {
        VxBackendKind backend = gpu_backends[index];
        BackendProbe* probe = &probes[(int)backend];
        int init_before = probe->init_calls;
        int cleanup_before = probe->cleanup_calls;
        probe->init_result = -1;
        CHECK(vx_backend_manager_activate(backend) == -1);
        CHECK(probe->init_calls == init_before + 1);
        CHECK(probe->cleanup_calls == cleanup_before + 1);
        CHECK(vx_backend_manager_current() == VX_PORTABLE_BACKEND_KIND);
        CHECK(selected_flags_match(state, VX_PORTABLE_BACKEND_KIND));

        probe->init_result = 0;
        CHECK(activate_and_check(state, backend));
        vx_backend_manager_deactivate();
        CHECK(probe->cleanup_calls == cleanup_before + 2);
        CHECK(vx_backend_manager_current() == VX_PORTABLE_BACKEND_KIND);
        CHECK(selected_flags_match(state, VX_PORTABLE_BACKEND_KIND));
    }

    /* A failed replacement never cleans or demotes the live backend. */
    CHECK(activate_and_check(state, VX_BACKEND_KIND_VULKAN));
    {
        int vulkan_cleanup_before =
            probes[VX_BACKEND_KIND_VULKAN].cleanup_calls;
        int opengl_cleanup_before =
            probes[VX_BACKEND_KIND_OPENGL].cleanup_calls;
        probes[VX_BACKEND_KIND_OPENGL].init_result = -1;
        CHECK(vx_backend_manager_activate(VX_BACKEND_KIND_OPENGL) == -1);
        CHECK(probes[VX_BACKEND_KIND_OPENGL].cleanup_calls ==
              opengl_cleanup_before + 1);
        CHECK(probes[VX_BACKEND_KIND_VULKAN].cleanup_calls ==
              vulkan_cleanup_before);
        CHECK(vx_backend_manager_current() == VX_BACKEND_KIND_VULKAN);
        CHECK(selected_flags_match(state, VX_BACKEND_KIND_VULKAN));
    }

    /* A successful replacement initializes the candidate before retiring the
     * old backend, then publishes exactly one new selection. */
    probes[VX_BACKEND_KIND_OPENGL].init_result = 0;
    {
        size_t before = event_count;
        int vulkan_cleanup_before =
            probes[VX_BACKEND_KIND_VULKAN].cleanup_calls;
        CHECK(vx_backend_manager_activate(VX_BACKEND_KIND_OPENGL) == 0);
        CHECK(event_count == before + 4u);
        CHECK(events[before] == 'g' && events[before + 1u] == 'i');
        CHECK(events[before + 2u] == 'v' && events[before + 3u] == 'c');
        CHECK(probes[VX_BACKEND_KIND_VULKAN].cleanup_calls ==
              vulkan_cleanup_before + 1);
        CHECK(vx_backend_manager_current() == VX_BACKEND_KIND_OPENGL);
        CHECK(selected_flags_match(state, VX_BACKEND_KIND_OPENGL));
    }

    /* Invalid enum values are rejected without an implicit CPU fallback. */
    CHECK(vx_backend_manager_activate((VxBackendKind)-1) == -1);
    CHECK(vx_backend_manager_activate(VX_BACKEND_KIND_UNSPECIFIED) == -1);
    CHECK(vx_backend_manager_current() == VX_BACKEND_KIND_OPENGL);
    CHECK(selected_flags_match(state, VX_BACKEND_KIND_OPENGL));

    vx_backend_manager_deactivate();
    vx_backend_manager_deactivate();
    CHECK(vx_backend_manager_current() == VX_PORTABLE_BACKEND_KIND);
    CHECK(selected_flags_match(state, VX_PORTABLE_BACKEND_KIND));

    /* Cleanup is not terminal: a previously failed and cleaned backend can be
     * selected and released again in the same state. */
    probes[VX_BACKEND_KIND_CUDA].init_result = 0;
    CHECK(activate_and_check(state, VX_BACKEND_KIND_CUDA));
    vx_backend_manager_deactivate();
    CHECK(vx_backend_manager_current() == VX_PORTABLE_BACKEND_KIND);

    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("native GPU backend contract passed");
    return 0;
}
