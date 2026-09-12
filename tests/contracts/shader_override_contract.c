#define _POSIX_C_SOURCE 200809L

#include "shader_store.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>


static int expect_view(
    const VolvoxAIShaderView* view,
    const unsigned char* expected,
    size_t expected_size,
    const char* label) {
    if (
        !view->data
        || view->size != expected_size
        || memcmp(view->data, expected, expected_size) != 0
    ) {
        fprintf(stderr, "contract fixture: unexpected %s shader bytes\n", label);
        return 0;
    }
    return 1;
}


int main(int argc, char** argv) {
    static const unsigned char embedded[] = "embedded shader fixture";
    static const unsigned char external[] = "external shader fixture";
    const char* shader_path = "spv/test.spv";
    VolvoxAIShaderView view = {(const unsigned char*)1, 1};

    if (argc != 3) {
        fprintf(stderr, "usage: %s MISSING_OVERRIDE VALID_OVERRIDE\n", argv[0]);
        return 2;
    }

    /* One rejected request must not poison the process-wide store. */
    if (
        volvoxai_shader_store_get("../test.spv", &view)
            != VOLVOXAI_SHADER_STORE_INVALID_ARGUMENT
        || view.data != NULL
        || view.size != 0
    ) {
        fprintf(stderr, "contract fixture: invalid lookup contract failed\n");
        return 3;
    }

    if (setenv("VOLVOXAI_SHADER_DIR", argv[1], 1) != 0) {
        perror("setenv missing override");
        return 4;
    }
    if (
        volvoxai_shader_store_get(shader_path, &view)
            != VOLVOXAI_SHADER_STORE_OK
        || !expect_view(&view, embedded, sizeof(embedded) - 1, "embedded")
    ) {
        return 5;
    }

    if (setenv("VOLVOXAI_SHADER_DIR", argv[2], 1) != 0) {
        perror("setenv valid override");
        return 6;
    }
    if (
        volvoxai_shader_store_get(shader_path, &view)
            != VOLVOXAI_SHADER_STORE_OK
        || !expect_view(&view, external, sizeof(external) - 1, "external")
    ) {
        return 7;
    }
    if (
        volvoxai_shader_store_get(shader_path, &view)
            != VOLVOXAI_SHADER_STORE_OK
        || !expect_view(&view, external, sizeof(external) - 1, "cached external")
    ) {
        return 8;
    }

    /* Log-once is process-scoped, not cache-lifetime-scoped. */
    volvoxai_shader_store_shutdown();
    if (
        volvoxai_shader_store_get(shader_path, &view)
            != VOLVOXAI_SHADER_STORE_OK
        || !expect_view(&view, external, sizeof(external) - 1, "reloaded external")
    ) {
        return 9;
    }
    volvoxai_shader_store_shutdown();
    return 0;
}
