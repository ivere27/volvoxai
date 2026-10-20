"""Exercise cache activation and retained reads through production C boundaries."""
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

from tools.tests import function_source


PREAMBLE = r'''
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
typedef int VxStatus;
enum { VX_STATUS_OK, VX_STATUS_EXECUTION_FAILED, VX_STATUS_INVALID_ARGUMENT,
       VX_STATUS_OUT_OF_MEMORY };
typedef struct {
    int device_acquired, forward_active, transient_active, implicit_forward;
    void* trace_pass;
} OpenGLContextState;
typedef struct { OpenGLContextState gl; } VxEngineState;
typedef struct { VxEngineState* previous; } VxEngineStateScope;
typedef struct { unsigned tensor_count; } VxDecodeCacheCommand;
typedef struct {
    VxDecodeCacheCommand execution_cache;
    void* decode_cache;
    VxEngineState* engine_state;
} VxExecutionContext;
static VxEngineState state, *current;
static int g_use_opengl, route_lease, model_locks, scope_entries;
static int configure_calls, transfer_count, fail_after, configure_status, close_unready;
static uint32_t cache_rows[4], stage_rows[4];
static const uint32_t expected_output[4] = {0x80000000u, 0x7fc01234u, 0x3f800000u, 0xff800000u};
static VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* owner) {
    VxEngineStateScope scope = {current}; current = owner; ++scope_entries; return scope;
}
static void vx_engine_state_scope_leave(VxEngineStateScope scope) { current = scope.previous; }
static int volvoxai_engine_model_route_lease_active(void) { return route_lease; }
static void volvoxai_engine_model_lock(void) { assert(model_locks == 0); ++model_locks; }
static void volvoxai_engine_model_unlock(void) { assert(model_locks == 1); --model_locks; }

#if VOLVOXAI_ENABLE_OPENGL
typedef unsigned int GLuint, GLenum;
#define EGL_NO_CONTEXT 0
#define GL_ALL_BARRIER_BITS 0xffffffffu
#define GL_BUFFER_UPDATE_BARRIER_BIT 0x200u
#define GL_SHADER_STORAGE_BUFFER 0x90d2u
#define GL_NO_ERROR 0
typedef struct { struct { uint64_t handle; } buffer; } VxNativeStorage;
static pthread_mutex_t device_mutex = PTHREAD_MUTEX_INITIALIZER;
static int device_locks, ready, finishes, bound_buffer, releases;
static int egl_context = 1;
static OpenGLContextState* opengl_context_state_get(int create) {
    (void)create; return current ? &current->gl : NULL;
}
/* An attempted recursive acquisition fails immediately in this regression
 * instead of leaving its runner blocked in pthread_mutex_lock. */
static void opengl_device_lock(void) {
    int result = pthread_mutex_trylock(&device_mutex);
    assert(result == 0); assert(device_locks == 0); ++device_locks;
}
static void opengl_device_unlock(void) {
    assert(device_locks == 1); --device_locks;
    assert(pthread_mutex_unlock(&device_mutex) == 0);
}
static int ogl_ready(void) { return ready; }
static int opengl_make_current_locked(void) { assert(device_locks == 1); return 0; }
static void opengl_release_current_locked(void) {
    assert(device_locks == 1);
    if (current) assert(model_locks || route_lease);
    ++releases;
}
static void finish(void) { assert(device_locks == 1); ++finishes; }
static void (*p_glFinish)(void) = finish;
static void opengl_trace_end_record(OpenGLContextState* owner) { (void)owner; }
static void opengl_trace_finish(OpenGLContextState* owner, int ok) { (void)owner; (void)ok; }
static void opengl_clear_errors(void) { assert(device_locks == 1); }
static void p_glMemoryBarrier(GLenum flags) { (void)flags; assert(device_locks == 1); }
static void p_glBindBuffer(GLenum target, GLuint buffer) {
    assert(target == GL_SHADER_STORAGE_BUFFER && device_locks == 1); bound_buffer = (int)buffer;
}
static GLenum p_glGetError(void) { return GL_NO_ERROR; }
static int read_buffer(GLenum target, size_t bytes, void* destination) {
    assert(target == GL_SHADER_STORAGE_BUFFER && bound_buffer == 77 && device_locks == 1);
    assert(bytes == sizeof(expected_output)); memcpy(destination, expected_output, bytes); return 1;
}
'''

CONFIGURE = r'''
#endif
/* Device calls are doubles; cache activation, GL pass ownership and retained
 * read boundaries below are extracted unchanged from their production files. */
static VxStatus vx_cache_configure(VxExecutionContext* context,
        const VxDecodeCacheCommand* command, int* mutated) {
    assert(current == context->engine_state && command->tensor_count == 2);
    assert(model_locks || route_lease); ++configure_calls;
    if (configure_status != VX_STATUS_OK && fail_after == 0) return configure_status;
    for (int index = 0; index < 2; ++index) {
#if VOLVOXAI_ENABLE_OPENGL
        if (g_use_opengl) assert(opengl_graph_access_ensure());
#endif
        *mutated = 1; ++transfer_count;
        if (index == 0) memcpy(stage_rows, cache_rows, sizeof(cache_rows));
        else for (int row = 0; row < 4; ++row) cache_rows[row] = stage_rows[3 - row];
        if (fail_after == transfer_count) {
#if VOLVOXAI_ENABLE_OPENGL
            if (g_use_opengl && close_unready) ready = 0;
#endif
            return configure_status;
        }
    }
    context->decode_cache = context;
    return configure_status;
}
'''

FIXTURE = r'''
static VxExecutionContext reset(int opengl) {
    assert(current == NULL && model_locks == 0);
    memset(&state, 0, sizeof(state)); state.gl.device_acquired = 1;
    g_use_opengl = opengl; route_lease = 0; scope_entries = configure_calls = transfer_count = 0;
    fail_after = close_unready = 0; configure_status = VX_STATUS_OK;
    for (int row = 0; row < 4; ++row) cache_rows[row] = expected_output[row];
#if VOLVOXAI_ENABLE_OPENGL
    assert(device_locks == 0); ready = 1; finishes = releases = 0;
#endif
    return (VxExecutionContext){.execution_cache = {2}, .engine_state = &state};
}
#if VOLVOXAI_ENABLE_OPENGL
static void owned_read_after_return(void) {
    uint32_t output[4] = {0};
    VxNativeStorage retained = {.buffer = {77}};
    assert(current == NULL && model_locks == 0);
    assert(ogl_tensor_read(&retained, output, sizeof(output)));
    assert(memcmp(output, expected_output, sizeof(output)) == 0);
    assert(device_locks == 0);
    assert(state.gl.forward_active == 0 && state.gl.implicit_forward == 0);
}
static void check_gl(void) {
    VxExecutionContext context = reset(1);
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OK);
    assert(transfer_count == 2);
    for (int row = 0; row < 4; ++row) assert(cache_rows[row] == expected_output[3 - row]);
    owned_read_after_return();
    assert(finishes == 2 && releases == 2);

    /* A partially submitted transfer still drops its pass before error
     * publication, retained reads and context retirement. */
    context = reset(1); fail_after = 1; configure_status = VX_STATUS_OUT_OF_MEMORY;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OUT_OF_MEMORY);
    assert(transfer_count == 1 && context.decode_cache == NULL);
    assert(memcmp(stage_rows, expected_output, sizeof(stage_rows)) == 0);
    owned_read_after_return();
    assert(finishes == 2 && releases == 2);

    context = reset(1); configure_status = VX_STATUS_INVALID_ARGUMENT;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_INVALID_ARGUMENT);
    assert(!transfer_count && !finishes && !releases); owned_read_after_return();

    context = reset(1); route_lease = 1;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OK);
    assert(model_locks == 0 && finishes == 1); route_lease = 0; owned_read_after_return();

    /* Ending an acquired but no-longer-ready pass reports execution failure
     * while preserving a more specific configure failure. */
    context = reset(1); fail_after = 1; close_unready = 1;
    configure_status = VX_STATUS_INVALID_ARGUMENT;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_INVALID_ARGUMENT);
    assert(device_locks == 0 && !state.gl.forward_active);
    context = reset(1); fail_after = 1; close_unready = 1;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_EXECUTION_FAILED);
    assert(device_locks == 0 && !state.gl.forward_active);
}
#endif
static void check_early_returns(void) {
    VxExecutionContext context = reset(1); context.execution_cache.tensor_count = 0;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OK);
    assert(!scope_entries && !configure_calls && !transfer_count);
    context = reset(1); context.decode_cache = &context;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OK);
    assert(!scope_entries && !configure_calls && !transfer_count);
#if VOLVOXAI_ENABLE_OPENGL
    assert(!finishes && !releases && !device_locks);
#endif
}
static void check_other_backend(void) {
    VxExecutionContext context = reset(0);
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OK);
    assert(scope_entries == 1 && configure_calls == 1 && transfer_count == 2 && model_locks == 0);
#if VOLVOXAI_ENABLE_OPENGL
    assert(!finishes && !releases && !device_locks);
#endif
    context = reset(0); fail_after = 1; configure_status = VX_STATUS_OUT_OF_MEMORY;
    assert(vx_execution_apply_cache(&context) == VX_STATUS_OUT_OF_MEMORY);
    assert(transfer_count == 1 && current == NULL && model_locks == 0);
}
int main(void) {
#if VOLVOXAI_ENABLE_OPENGL
    check_gl();
#endif
    check_early_returns(); check_other_backend();
    puts("PASS production cache activation closes its GL pass before retained read and failure cleanup");
}
'''


class OpenGLCachePassTests(unittest.TestCase):
    def run_fixture(self, enabled):
        compiler = shutil.which("cc")
        if not compiler:
            self.skipTest("C compiler unavailable")
        root = Path(__file__).resolve().parents[2]
        backend = root / "native/src/backends"
        translation = PREAMBLE
        for name in ("opengl_graph_access_ensure", "opengl_graph_end_forward"):
            translation += function_source((backend / "opengl_engine.c").read_text(), name)
        translation += function_source((backend / "opengl_tensor_interop.inc").read_text(), "ogl_tensor_read")
        translation += CONFIGURE
        translation += function_source((root / "native/src/runtime/public_api_execution_driver.inc").read_text(),
                                       "vx_execution_apply_cache")
        translation += FIXTURE
        with tempfile.TemporaryDirectory(prefix="volvoxai-gl-cache-pass-") as directory:
            source, binary = Path(directory) / "cache.c", Path(directory) / "cache"
            source.write_text(translation)
            built = subprocess.run([compiler, "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread",
                "-DVOLVOXAI_ENABLE_TRAINING=0", f"-DVOLVOXAI_ENABLE_OPENGL={enabled}", str(source), "-o", str(binary)],
                capture_output=True, text=True)
            self.assertEqual(built.returncode, 0, built.stdout + built.stderr)
            checked = subprocess.run([str(binary)], capture_output=True, text=True, timeout=5)
            self.assertEqual(checked.returncode, 0, checked.stdout + checked.stderr)
            self.assertIn("PASS production cache activation closes its GL pass", checked.stdout)

    def test_paged_activation_retained_read_and_failure_cleanup(self):
        self.run_fixture(1)

    def test_non_gl_profile_keeps_existing_cache_behavior(self):
        self.run_fixture(0)


if __name__ == "__main__":
    unittest.main()
