#include "backend_manager.h"

#include "backend_config.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if VOLVOXAI_ENABLE_NNAPI
#include "nnapi_engine.h"
#endif

/* Runtime dispatch reads these private flags. Their storage belongs to the
 * backend integration layer, never to a CLI or embedding application. */
int g_use_vulkan;
int g_use_nnapi;
int g_use_opengl;
int g_use_metal;

static VolvoxAIEngineBackend g_backend = VOLVOXAI_BACKEND_CPU;

static void clear_backend_flags(void) {
    g_use_vulkan = 0;
    g_use_nnapi = 0;
    g_use_opengl = 0;
    g_use_metal = 0;
}

static void cleanup_backend(VolvoxAIEngineBackend backend) {
    switch (backend) {
        case VOLVOXAI_BACKEND_VULKAN:
#if VOLVOXAI_ENABLE_VULKAN
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
        case VOLVOXAI_BACKEND_NNAPI:
#if VOLVOXAI_ENABLE_NNAPI
            nnapi_cleanup();
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
        case VOLVOXAI_BACKEND_NNAPI: g_use_nnapi = 1; break;
        case VOLVOXAI_BACKEND_CPU:
        default: break;
    }
}

void vx_backend_manager_deactivate(void) {
    cleanup_backend(g_backend);
    select_backend(VOLVOXAI_BACKEND_CPU);
}

int vx_backend_manager_activate(VolvoxAIEngineBackend backend) {
    if (backend < VOLVOXAI_BACKEND_CPU || backend > VOLVOXAI_BACKEND_NNAPI) return -1;
    if (backend == g_backend) return 0;
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
        case VOLVOXAI_BACKEND_NNAPI:
#if VOLVOXAI_ENABLE_NNAPI
            status = nnapi_init();
#endif
            break;
        case VOLVOXAI_BACKEND_CUSTOM:
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
        case VOLVOXAI_BACKEND_NNAPI: return "NNAPI";
        case VOLVOXAI_BACKEND_CPU:
        default: return "CPU";
    }
}
