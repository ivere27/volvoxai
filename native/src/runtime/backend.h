#ifndef VOLVOX_RUNTIME_BACKEND_H
#define VOLVOX_RUNTIME_BACKEND_H

/*
 * Private backend registry used by the native runtime.  This is deliberately
 * not a public header: Node and T are runtime-owned types, and device
 * registration must not become part of the installed C ABI.
 */
#include "engine_internal.h"

#include <stddef.h>

/* `supports` and `run` return one of these values. A declined `run` must not
 * write an output tensor or claim device ownership; validation caches may be
 * populated while deciding whether a legacy kernel can run. */
enum {
    VX_BACKEND_ERROR = -1,
    VX_BACKEND_DECLINED = 0,
    VX_BACKEND_HANDLED = 1,
};

/* `init` uses conventional C success while retaining a non-fatal unavailable
 * result for optional devices.  CPU, the required final fallback, may not be
 * unavailable.  teardown is called after every non-ready init attempt. */
enum {
    VX_BACKEND_INIT_ERROR = -1,
    VX_BACKEND_INIT_READY = 0,
    VX_BACKEND_INIT_UNAVAILABLE = 1,
};

typedef struct VxBackend {
    const char* name;
    void* user_data;
    int required;

    int (*init)(void* user_data);
    int (*supports)(void* user_data, const Node* node);
    int (*run)(void* user_data, const Node* node, T* input, T* output);

    void* (*alloc)(void* user_data, size_t bytes);
    void (*upload)(void* user_data, void* destination, const void* source, size_t bytes);
    void (*download)(void* user_data, void* destination, const void* source, size_t bytes);

    /* A graph route writes its output into backend-owned storage. One-shot
     * routes (NNAPI/Vulkan/OpenGL Linear) write host memory and therefore need
     * the normal host-dirty propagation after dispatch. */
    int outputs_device_resident;

    /* Optional graph-storage lifecycle hooks.  Their signatures mirror the
     * existing Vulkan/OpenGL/Metal graph helpers so runtime integration can
     * migrate without changing synchronization semantics. */
    void (*reset)(void* user_data);
    void (*begin_forward)(void* user_data);
    int (*end_forward)(void* user_data);       /* zero is success */
    void (*mark_host)(void* user_data, const void* host, size_t bytes, int is_weight);
    /* Private device hooks retain the historic nonzero-success result. */
    int (*sync_host)(void* user_data, const void* host, size_t bytes, int is_weight);

    void (*teardown)(void* user_data);
} VxBackend;

enum { VX_BACKEND_MAX = 16 };

typedef struct {
    const VxBackend* backends[VX_BACKEND_MAX];
    unsigned char initialized[VX_BACKEND_MAX];
    size_t count;
    size_t cpu_index;
    int ready;
} VxBackendRegistry;

/* The supplied CPU backend must be the last entry.  The registry copies only
 * backend pointers; each VxBackend object must outlive the registry.  Create
 * a registry with `{0}` and call teardown before reinitializing it. */
int vx_backend_registry_validate(const VxBackend* const* backends,
                                 size_t count,
                                 const VxBackend* cpu_backend);
int vx_backend_registry_init(VxBackendRegistry* registry,
                             const VxBackend* const* backends,
                             size_t count,
                             const VxBackend* cpu_backend);
void vx_backend_registry_teardown(VxBackendRegistry* registry);

/* Successful dispatch returns VX_BACKEND_HANDLED and stores the selected
 * backend when `selected` is non-NULL.  An error never falls through to CPU.
 * Because CPU is required and final, a normal dispatch never returns DECLINED. */
int vx_backend_registry_dispatch(const VxBackendRegistry* registry,
                                 const Node* node,
                                 T* input,
                                 T* output,
                                 const VxBackend** selected);

/* Fan out existing graph lifecycle and host/device coherence operations to
 * initialized backends in registration order. */
void vx_backend_registry_reset(VxBackendRegistry* registry);
void vx_backend_registry_begin_forward(VxBackendRegistry* registry);
int vx_backend_registry_end_forward(VxBackendRegistry* registry);
void vx_backend_registry_mark_host(VxBackendRegistry* registry,
                                   const void* host,
                                   size_t bytes,
                                   int is_weight);
int vx_backend_registry_sync_host(VxBackendRegistry* registry,
                                  const void* host,
                                  size_t bytes,
                                  int is_weight);

#endif
