/* Linux-only hardware qualification shim. Intercepts dynamically resolved
 * CUDA, OpenGL and Vulkan driver calls; forwards every call unchanged. Never linked into the product.
 *
 * cc -shared -fPIC -O2 native/tests/gpu_profiling_probe.c -ldl -o build/gpu_profiling_probe.so
 * LD_PRELOAD=.../gpu_profiling_probe.so python tools/qualify_profiling.py --backend cuda --gpu-probe
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdatomic.h>
#include <stdint.h>
#include <string.h>

enum { GRAPH_LAUNCH, GRAPH_CAPTURE, TIMING_CREATE, ELAPSED, STREAM_SYNC,
       EVENT_SYNC, CONTEXT_SYNC, EXTERNAL_RECORD, GL_CREATE, GL_WRITE,
       GL_READ, GL_FINISH, VK_CREATE, VK_WRITE, VK_READ, VK_WAIT, EGL_PROC, VK_PROC,
       GL_CALIBRATE, VK_CALIBRATE, COUNT };
static _Atomic uint64_t counts[COUNT];
static _Atomic(void*) functions[COUNT];

uint64_t vx_test_gpu_counter(unsigned index) {
    return index < COUNT ? atomic_load(&counts[index]) : 0;
}
static int graph_launch(void* graph, void* stream) {
    atomic_fetch_add(&counts[GRAPH_LAUNCH], 1);
    return ((int (*)(void*, void*))atomic_load(&functions[GRAPH_LAUNCH]))(graph, stream);
}
static int graph_capture(void* stream, int mode) {
    atomic_fetch_add(&counts[GRAPH_CAPTURE], 1);
    return ((int (*)(void*, int))atomic_load(&functions[GRAPH_CAPTURE]))(stream, mode);
}
static int event_create(void** event, unsigned flags) {
    if (!(flags & 2u)) atomic_fetch_add(&counts[TIMING_CREATE], 1);
    return ((int (*)(void**, unsigned))atomic_load(&functions[TIMING_CREATE]))(event, flags);
}
static int event_elapsed(float* elapsed, void* start, void* end) {
    atomic_fetch_add(&counts[ELAPSED], 1);
    return ((int (*)(float*, void*, void*))atomic_load(&functions[ELAPSED]))(elapsed, start, end);
}
static int stream_sync(void* stream) {
    atomic_fetch_add(&counts[STREAM_SYNC], 1);
    return ((int (*)(void*))atomic_load(&functions[STREAM_SYNC]))(stream);
}
static int event_sync(void* event) {
    atomic_fetch_add(&counts[EVENT_SYNC], 1);
    return ((int (*)(void*))atomic_load(&functions[EVENT_SYNC]))(event);
}
static int context_sync(void) {
    atomic_fetch_add(&counts[CONTEXT_SYNC], 1);
    return ((int (*)(void))atomic_load(&functions[CONTEXT_SYNC]))();
}

static int external_record(void* event, void* stream, unsigned flags) {
    atomic_fetch_add(&counts[EXTERNAL_RECORD], 1);
    return ((int (*)(void*, void*, unsigned))atomic_load(&functions[EXTERNAL_RECORD]))(event, stream, flags);
}
static void gl_create(int count, unsigned* queries) {
    atomic_fetch_add(&counts[GL_CREATE], 1);
    ((void (*)(int, unsigned*))atomic_load(&functions[GL_CREATE]))(count, queries);
}
static void gl_write(unsigned query, unsigned target) {
    atomic_fetch_add(&counts[GL_WRITE], 1);
    ((void (*)(unsigned, unsigned))atomic_load(&functions[GL_WRITE]))(query, target);
}
static void gl_read(unsigned query, unsigned name, uint64_t* value) {
    atomic_fetch_add(&counts[GL_READ], 1);
    ((void (*)(unsigned, unsigned, uint64_t*))atomic_load(&functions[GL_READ]))(query, name, value);
}
static void gl_finish(void) {
    atomic_fetch_add(&counts[GL_FINISH], 1);
    ((void (*)(void))atomic_load(&functions[GL_FINISH]))();
}
static int vk_create(void* device, const void* info, const void* allocator, uint64_t* pool) {
    atomic_fetch_add(&counts[VK_CREATE], 1);
    return ((int (*)(void*, const void*, const void*, uint64_t*))atomic_load(&functions[VK_CREATE]))(device, info, allocator, pool);
}
static void vk_write(void* cmd, unsigned stage, uint64_t pool, unsigned query) {
    atomic_fetch_add(&counts[VK_WRITE], 1);
    ((void (*)(void*, unsigned, uint64_t, unsigned))atomic_load(&functions[VK_WRITE]))(cmd, stage, pool, query);
}
static int vk_read(void* device, uint64_t pool, unsigned first, unsigned count,
                   size_t bytes, void* data, uint64_t stride, unsigned flags) {
    atomic_fetch_add(&counts[VK_READ], 1);
    return ((int (*)(void*, uint64_t, unsigned, unsigned, size_t, void*, uint64_t, unsigned))
        atomic_load(&functions[VK_READ]))(device, pool, first, count, bytes, data, stride, flags);
}
static int vk_wait(void* device, unsigned count, const uint64_t* fences, unsigned all, uint64_t timeout) {
    atomic_fetch_add(&counts[VK_WAIT], 1);
    return ((int (*)(void*, unsigned, const uint64_t*, unsigned, uint64_t))
        atomic_load(&functions[VK_WAIT]))(device, count, fences, all, timeout);
}
static void gl_calibrate(unsigned name, uint64_t* value) {
    if (name == 0x8E28u) atomic_fetch_add(&counts[GL_CALIBRATE], 1); /* GL_TIMESTAMP */
    ((void (*)(unsigned, uint64_t*))atomic_load(&functions[GL_CALIBRATE]))(name, value);
}
static int vk_calibrate(void* device, unsigned count, const void* infos, uint64_t* values, uint64_t* deviation) {
    atomic_fetch_add(&counts[VK_CALIBRATE], 1);
    return ((int (*)(void*, unsigned, const void*, uint64_t*, uint64_t*))
        atomic_load(&functions[VK_CALIBRATE]))(device, count, infos, values, deviation);
}
static void* intercept(const char* name, void* address);
static void* egl_proc(const char* name) {
    return intercept(name, ((void* (*)(const char*))atomic_load(&functions[EGL_PROC]))(name));
}
static void* vk_proc(void* instance, const char* name) {
    return intercept(name, ((void* (*)(void*, const char*))atomic_load(&functions[VK_PROC]))(instance, name));
}
static void* intercept(const char* name, void* address) {
    if (!address) return address;
    static const char* const symbols[COUNT] = {"cuGraphLaunch", "cuStreamBeginCapture_v2",
        "cuEventCreate", "cuEventElapsedTime", "cuStreamSynchronize", "cuEventSynchronize", "cuCtxSynchronize",
        "cuEventRecordWithFlags", "glGenQueries", "glQueryCounter", "glGetQueryObjectui64v", "glFinish",
        "vkCreateQueryPool", "vkCmdWriteTimestamp", "vkGetQueryPoolResults", "vkWaitForFences",
        "eglGetProcAddress", "vkGetInstanceProcAddr", "glGetInteger64v", "vkGetCalibratedTimestampsEXT"};
    static void* const wrappers[COUNT] = {(void*)graph_launch, (void*)graph_capture,
        (void*)event_create, (void*)event_elapsed, (void*)stream_sync, (void*)event_sync, (void*)context_sync,
        (void*)external_record, (void*)gl_create, (void*)gl_write, (void*)gl_read, (void*)gl_finish,
        (void*)vk_create, (void*)vk_write, (void*)vk_read, (void*)vk_wait, (void*)egl_proc, (void*)vk_proc,
        (void*)gl_calibrate, (void*)vk_calibrate};
    for (unsigned i = 0; i < COUNT; i++) {
        if (strcmp(name, symbols[i]) == 0 ||
            (i == GRAPH_CAPTURE && strcmp(name, "cuStreamBeginCapture") == 0)) {
            if (address != wrappers[i]) atomic_store(&functions[i], address);
            return wrappers[i];
        }
    }
    return address;
}
void* dlsym(void* handle, const char* name) {
    void* (*resolve)(void*, const char*) = dlvsym(RTLD_NEXT, "dlsym", "GLIBC_2.2.5");
    return intercept(name, resolve(handle, name));
}
