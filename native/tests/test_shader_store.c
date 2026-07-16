#define _POSIX_C_SOURCE 200809L

#include "shader_store.h"

#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

typedef struct LookupThreadArgs {
    const char* path;
    const void* expected;
    size_t expected_size;
    int failed;
} LookupThreadArgs;

typedef struct ExpectedShader {
    unsigned char* data;
    size_t size;
} ExpectedShader;

static int view_equals(const VolvoxAIShaderView* view, const void* expected, size_t size) {
    return view->size == size && memcmp(view->data, expected, size) == 0;
}

static int read_expected_shader(
    const char* root, const char* path, ExpectedShader* expected) {
    size_t root_size = strlen(root);
    size_t path_size = strlen(path);
    char* file_path = (char*)malloc(root_size + path_size + 2);
    if (!file_path) return 0;
    memcpy(file_path, root, root_size);
    if (root_size > 0 && root[root_size - 1] != '/') file_path[root_size++] = '/';
    memcpy(file_path + root_size, path, path_size + 1);

    FILE* file = fopen(file_path, "rb");
    free(file_path);
    if (!file || fseek(file, 0, SEEK_END) != 0) {
        if (file) fclose(file);
        return 0;
    }
    long end = ftell(file);
    if (end <= 0 || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    expected->size = (size_t)end;
    expected->data = (unsigned char*)malloc(expected->size);
    if (!expected->data) {
        fclose(file);
        return 0;
    }
    size_t read_size = fread(expected->data, 1, expected->size, file);
    int failed = read_size != expected->size || ferror(file);
    fclose(file);
    if (failed) {
        free(expected->data);
        expected->data = NULL;
        expected->size = 0;
        return 0;
    }
    return 1;
}

static void* lookup_thread(void* opaque) {
    LookupThreadArgs* args = (LookupThreadArgs*)opaque;
    for (int i = 0; i < 200; i++) {
        VolvoxAIShaderView view;
        if (volvoxai_shader_store_get(args->path, &view) != VOLVOXAI_SHADER_STORE_OK ||
            !view_equals(&view, args->expected, args->expected_size)) {
            args->failed = 1;
            break;
        }
    }
    return NULL;
}

int main(int argc, char** argv) {
    if (argc == 2 && strcmp(argv[1], "--expect-corrupt-lazy") == 0) {
        VolvoxAIShaderView view;
        /* A corrupt GLSL block must not affect the untouched SPIR-V block. */
        CHECK(volvoxai_shader_store_get("spv/foo.spv", &view) ==
              VOLVOXAI_SHADER_STORE_OK);
        CHECK(view_equals(&view, "embedded-foo", strlen("embedded-foo")));
        CHECK(volvoxai_shader_store_get("glsl/foo.comp", &view) ==
              VOLVOXAI_SHADER_STORE_INVALID_DATA);
        CHECK(view.data == NULL && view.size == 0);
        volvoxai_shader_store_shutdown();
        puts("shader store corrupt/lazy test passed");
        return 0;
    }

    CHECK(argc == 3);
    const char* missing_override_root = argv[1];
    const char* valid_override_root = argv[2];
    const char* const paths[] = {
        "spv/add.spv",
        "spv/conv2D.spv",
        "glsl/add.comp",
        "spv/matMulBackward.spv",
    };
    ExpectedShader expected[sizeof(paths) / sizeof(paths[0])] = {{0}};
    for (size_t i = 0; i < sizeof(paths) / sizeof(paths[0]); i++) {
        CHECK(read_expected_shader(valid_override_root, paths[i], &expected[i]));
    }

    CHECK(unsetenv("VOLVOXAI_SHADER_DIR") == 0);
    VolvoxAIShaderView first;
    CHECK(volvoxai_shader_store_get(paths[0], &first) == VOLVOXAI_SHADER_STORE_OK);
    CHECK(view_equals(&first, expected[0].data, expected[0].size));

    VolvoxAIShaderView second;
    CHECK(volvoxai_shader_store_get(paths[1], &second) == VOLVOXAI_SHADER_STORE_OK);
    CHECK(view_equals(&second, expected[1].data, expected[1].size));
    /* A later lookup must not invalidate an earlier view from the same block. */
    CHECK(view_equals(&first, expected[0].data, expected[0].size));

    VolvoxAIShaderView training;
    CHECK(volvoxai_shader_store_get(paths[3], &training) ==
          VOLVOXAI_SHADER_STORE_OK);
    CHECK(view_equals(&training, expected[3].data, expected[3].size));

    VolvoxAIShaderView invalid = {(const unsigned char*)1, 1};
    CHECK(volvoxai_shader_store_get("../spv/foo.spv", &invalid) ==
          VOLVOXAI_SHADER_STORE_INVALID_ARGUMENT);
    CHECK(invalid.data == NULL && invalid.size == 0);
    CHECK(volvoxai_shader_store_get("spv/does-not-exist.spv", &invalid) ==
          VOLVOXAI_SHADER_STORE_NOT_FOUND);

    LookupThreadArgs thread_args[] = {
        {paths[0], expected[0].data, expected[0].size, 0},
        {paths[2], expected[2].data, expected[2].size, 0},
        {paths[1], expected[1].data, expected[1].size, 0},
        {paths[3], expected[3].data, expected[3].size, 0},
    };
    pthread_t threads[sizeof(thread_args) / sizeof(thread_args[0])];
    for (size_t i = 0; i < sizeof(thread_args) / sizeof(thread_args[0]); i++) {
        CHECK(pthread_create(&threads[i], NULL, lookup_thread, &thread_args[i]) == 0);
    }
    for (size_t i = 0; i < sizeof(thread_args) / sizeof(thread_args[0]); i++) {
        CHECK(pthread_join(threads[i], NULL) == 0);
        CHECK(!thread_args[i].failed);
    }

    char* copied_root = strdup(valid_override_root);
    CHECK(copied_root != NULL);
    CHECK(volvoxai_shader_store_set_override_root(copied_root) ==
          VOLVOXAI_SHADER_STORE_OK);
    memset(copied_root, 'x', strlen(copied_root));
    free(copied_root);

    /* A nonempty environment root takes precedence, even when it misses. */
    CHECK(setenv("VOLVOXAI_SHADER_DIR", missing_override_root, 1) == 0);
    CHECK(volvoxai_shader_store_get(paths[0], &second) == VOLVOXAI_SHADER_STORE_OK);
    CHECK(second.data == first.data);
    /* A repeated miss must neither fail nor invalidate the embedded cache. */
    CHECK(volvoxai_shader_store_get(paths[0], &second) == VOLVOXAI_SHADER_STORE_OK);

    CHECK(unsetenv("VOLVOXAI_SHADER_DIR") == 0);
    CHECK(volvoxai_shader_store_get(paths[0], &second) == VOLVOXAI_SHADER_STORE_OK);
    CHECK(view_equals(&second, expected[0].data, expected[0].size));
    CHECK(second.data != first.data);

    CHECK(setenv("VOLVOXAI_SHADER_DIR", valid_override_root, 1) == 0);
    CHECK(volvoxai_shader_store_get(paths[0], &second) == VOLVOXAI_SHADER_STORE_OK);
    CHECK(view_equals(&second, expected[0].data, expected[0].size));
    CHECK(second.data != first.data);
    CHECK(volvoxai_shader_store_get(paths[0], &second) == VOLVOXAI_SHADER_STORE_OK);

    volvoxai_shader_store_shutdown();
    CHECK(unsetenv("VOLVOXAI_SHADER_DIR") == 0);
    CHECK(volvoxai_shader_store_get(paths[0], &second) == VOLVOXAI_SHADER_STORE_OK);
    CHECK(view_equals(&second, expected[0].data, expected[0].size));
    volvoxai_shader_store_shutdown();

    for (size_t i = 0; i < sizeof(expected) / sizeof(expected[0]); i++) {
        free(expected[i].data);
    }

    puts("shader store tests passed");
    return 0;
}
