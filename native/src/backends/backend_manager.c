#include <string.h>

#include "backend_manager.h"

#include "backend_config.h"
#if VOLVOXAI_ENABLE_WEBGPU
#include "gpu_bridge.h"
#include "shader_catalog_profile.h"
#endif
#include "runtime_state.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#if VOLVOXAI_ENABLE_WEBGPU
#define g_use_webgpu (vx_engine_state_current()->use_webgpu)
#endif
#define g_use_vulkan (vx_engine_state_current()->use_vulkan)
#define g_use_opengl (vx_engine_state_current()->use_opengl)
#define g_use_metal (vx_engine_state_current()->use_metal)
#define g_use_cuda (vx_engine_state_current()->use_cuda)
#define g_backend (vx_engine_state_current()->backend)

static void clear_backend_flags(void) {
#if VOLVOXAI_ENABLE_WEBGPU
    g_use_webgpu = 0;
#endif
    g_use_vulkan = 0;
    g_use_opengl = 0;
    g_use_metal = 0;
    g_use_cuda = 0;
}

static void cleanup_backend(VxBackendKind backend) {
    switch (backend) {
        case VX_BACKEND_KIND_VULKAN:
#if VOLVOXAI_ENABLE_VULKAN
            /* Drops this engine state's context only. The shared device is not
             * released here: vx_model_compile unloads its scratch engine state
             * through this path, so releasing would rebuild the whole device
             * before execute. It is torn down at process exit instead. */
            vk_cleanup();
#endif
            break;
        case VX_BACKEND_KIND_OPENGL:
#if VOLVOXAI_ENABLE_OPENGL
            opengl_cleanup();
#endif
            break;
        case VX_BACKEND_KIND_METAL:
#if VOLVOXAI_ENABLE_METAL
            metal_cleanup();
#endif
            break;
        case VX_BACKEND_KIND_CUDA:
#if VOLVOXAI_ENABLE_CUDA
            cuda_cleanup();
#endif
            break;
        case VX_PORTABLE_BACKEND_KIND:
        default:
            break;
    }
}

static void select_backend(VxBackendKind backend) {
    clear_backend_flags();
    g_backend = backend;
    switch (backend) {
#if VOLVOXAI_ENABLE_WEBGPU
        case VX_BACKEND_KIND_WEBGPU: g_use_webgpu = 1; break;
#endif
        case VX_BACKEND_KIND_VULKAN: g_use_vulkan = 1; break;
        case VX_BACKEND_KIND_OPENGL: g_use_opengl = 1; break;
        case VX_BACKEND_KIND_METAL: g_use_metal = 1; break;
        case VX_BACKEND_KIND_CUDA: g_use_cuda = 1; break;
        case VX_PORTABLE_BACKEND_KIND:
        default: break;
    }
}

void vx_backend_manager_deactivate(void) {
    cleanup_backend(g_backend);
    select_backend(VX_PORTABLE_BACKEND_KIND);
}

int vx_backend_manager_activate(VxBackendKind backend) {
    if (!vx_backend_kind_name(backend) ||
        (backend != VX_PORTABLE_BACKEND_KIND &&
         (backend == VX_BACKEND_KIND_WASM || backend == VX_BACKEND_KIND_NATIVE_CPU))) return -1;
    if ((int)backend == g_backend) return 0;
    int status = -1;
    switch (backend) {
        case VX_PORTABLE_BACKEND_KIND:
            status = 0;
            break;
        case VX_BACKEND_KIND_WEBGPU:
#if VOLVOXAI_ENABLE_WEBGPU
            /* The device belongs to the host. Activation only asks whether one
             * is there; the backend registry reports it unavailable otherwise
             * and the engine stays where it was. */
            status = vx_gpu_available((uint32_t)(uintptr_t)VX_GPU_BRIDGE_ABI_HASH,
                (uint32_t)(uintptr_t)VX_SHADER_CATALOG_HASH) ? 0 : -1;
#endif
            break;
        case VX_BACKEND_KIND_VULKAN:
#if VOLVOXAI_ENABLE_VULKAN
            status = vk_init();
#endif
            break;
        case VX_BACKEND_KIND_OPENGL:
#if VOLVOXAI_ENABLE_OPENGL
            status = opengl_init();
#endif
            break;
        case VX_BACKEND_KIND_METAL:
#if VOLVOXAI_ENABLE_METAL
            status = metal_init();
#endif
            break;
        case VX_BACKEND_KIND_CUDA:
#if VOLVOXAI_ENABLE_CUDA
            status = cuda_init();
#endif
            break;
        default:
            break;
    }
    if (status != 0) {
        /* Initializers may allocate partially before reporting failure. Clean
         * only the attempted backend; the active backend and its flags remain
         * untouched until a replacement has initialized successfully. */
        cleanup_backend(backend);
        return -1;
    }
    cleanup_backend(g_backend);
    select_backend(backend);
    return 0;
}

VxBackendKind vx_backend_manager_current(void) { return g_backend; }

const char* vx_backend_manager_name_of(VxBackendKind backend) {
    const char* name = vx_backend_kind_name(backend);
    return name ? name : VX_PORTABLE_BACKEND_NAME;
}

const char* vx_backend_manager_name(void) {
    return vx_backend_manager_name_of(g_backend);
}

int vx_backend_manager_by_name(const char* name, VxBackendKind* backend) {
    VxBackendKind kind = vx_backend_kind_from_name(name);
    if (!backend || kind == VX_BACKEND_KIND_UNSPECIFIED ||
        (kind != VX_PORTABLE_BACKEND_KIND &&
         (kind == VX_BACKEND_KIND_WASM || kind == VX_BACKEND_KIND_NATIVE_CPU))) return 0;
    *backend = kind;
    return 1;
}
