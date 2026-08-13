#include "backend_manager.h"

#include "backend_config.h"
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

#define g_use_vulkan (vx_engine_state_current()->use_vulkan)
#define g_use_opengl (vx_engine_state_current()->use_opengl)
#define g_use_metal (vx_engine_state_current()->use_metal)
#define g_use_cuda (vx_engine_state_current()->use_cuda)
#define g_backend (vx_engine_state_current()->backend)

static void clear_backend_flags(void) {
    g_use_vulkan = 0;
    g_use_opengl = 0;
    g_use_metal = 0;
    g_use_cuda = 0;
}

static void cleanup_backend(VolvoxAIEngineBackend backend) {
    switch (backend) {
        case VOLVOXAI_BACKEND_VULKAN:
#if VOLVOXAI_ENABLE_VULKAN
            /* Drops this engine state's context only. The shared device is not
             * released here: vx_model_compile unloads its scratch engine state
             * through this path, so releasing would rebuild the whole device
             * before execute. It is torn down at process exit instead. */
            vk_cleanup();
#endif
            break;
        case VOLVOXAI_BACKEND_OPENGL:
#if VOLVOXAI_ENABLE_OPENGL
            opengl_cleanup();
#endif
            break;
        case VOLVOXAI_BACKEND_METAL:
#if VOLVOXAI_ENABLE_METAL
            metal_cleanup();
#endif
            break;
        case VOLVOXAI_BACKEND_CUDA:
#if VOLVOXAI_ENABLE_CUDA
            cuda_cleanup();
#endif
            break;
        case VOLVOXAI_BACKEND_CPU:
        default:
            break;
    }
}

static void select_backend(VolvoxAIEngineBackend backend) {
    clear_backend_flags();
    g_backend = backend;
    switch (backend) {
        case VOLVOXAI_BACKEND_VULKAN: g_use_vulkan = 1; break;
        case VOLVOXAI_BACKEND_OPENGL: g_use_opengl = 1; break;
        case VOLVOXAI_BACKEND_METAL: g_use_metal = 1; break;
        case VOLVOXAI_BACKEND_CUDA: g_use_cuda = 1; break;
        case VOLVOXAI_BACKEND_CPU:
        default: break;
    }
}

void vx_backend_manager_deactivate(void) {
    cleanup_backend(g_backend);
    select_backend(VOLVOXAI_BACKEND_CPU);
}

int vx_backend_manager_activate(VolvoxAIEngineBackend backend) {
    if (backend < VOLVOXAI_BACKEND_CPU || backend > VOLVOXAI_BACKEND_CUDA)
        return -1;
    if ((int)backend == g_backend) return 0;
    int status = -1;
    switch (backend) {
        case VOLVOXAI_BACKEND_CPU:
            status = 0;
            break;
        case VOLVOXAI_BACKEND_VULKAN:
#if VOLVOXAI_ENABLE_VULKAN
            status = vk_init();
#endif
            break;
        case VOLVOXAI_BACKEND_OPENGL:
#if VOLVOXAI_ENABLE_OPENGL
            status = opengl_init();
#endif
            break;
        case VOLVOXAI_BACKEND_METAL:
#if VOLVOXAI_ENABLE_METAL
            status = metal_init();
#endif
            break;
        case VOLVOXAI_BACKEND_CUDA:
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

VolvoxAIEngineBackend vx_backend_manager_current(void) { return g_backend; }

const char* vx_backend_manager_name(void) {
    switch (g_backend) {
        case VOLVOXAI_BACKEND_VULKAN: return "Vulkan";
        case VOLVOXAI_BACKEND_OPENGL: return "OpenGL";
        case VOLVOXAI_BACKEND_METAL: return "Metal";
        case VOLVOXAI_BACKEND_CUDA: return "CUDA";
        case VOLVOXAI_BACKEND_CPU:
        default: return "CPU";
    }
}
