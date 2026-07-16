#include "backend_manager.h"

#include <stdio.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

extern int g_use_vulkan;
extern int g_use_opengl;

static int g_init_result;
static int g_init_calls;
static int g_cleanup_calls;
static int g_opengl_init_result;
static int g_opengl_init_calls;
static int g_opengl_cleanup_calls;

int vk_init(void) {
    g_init_calls++;
    return g_init_result;
}

void vk_cleanup(void) {
    g_cleanup_calls++;
}

int opengl_init(void) {
    g_opengl_init_calls++;
    return g_opengl_init_result;
}

void opengl_cleanup(void) {
    g_opengl_cleanup_calls++;
}

int main(void) {
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    g_init_result = -1;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_VULKAN) == -1);
    CHECK(g_init_calls == 1);
    CHECK(g_cleanup_calls == 1);
    CHECK(g_use_vulkan == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    g_init_result = 0;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_VULKAN) == 0);
    CHECK(g_init_calls == 2);
    CHECK(g_cleanup_calls == 1);
    CHECK(g_use_vulkan == 1);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_VULKAN);
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_VULKAN) == 0);
    CHECK(g_init_calls == 2);

    g_opengl_init_result = -1;
    CHECK(vx_backend_manager_activate(VOLVOXAI_BACKEND_OPENGL) == -1);
    CHECK(g_opengl_init_calls == 1);
    CHECK(g_opengl_cleanup_calls == 1);
    CHECK(g_cleanup_calls == 1);
    CHECK(g_use_vulkan == 1 && g_use_opengl == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_VULKAN);

    vx_backend_manager_deactivate();
    CHECK(g_cleanup_calls == 2);
    CHECK(g_use_vulkan == 0);
    CHECK(vx_backend_manager_current() == VOLVOXAI_BACKEND_CPU);

    puts("native backend manager tests passed");
    return 0;
}
