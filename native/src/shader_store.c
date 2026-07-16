#include "shader_store.h"

#include "embedded_shaders.h"
#include "../third_party/xz-embedded/xz.h"

#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE ((size_t)4096)
#define VOLVOXAI_SHADER_STORE_MAX_BLOCK_COUNT ((size_t)64)
#define VOLVOXAI_SHADER_STORE_MAX_BLOCK_SIZE ((size_t)64 * 1024 * 1024)
#define VOLVOXAI_SHADER_STORE_MAX_OVERRIDE_SIZE ((size_t)16 * 1024 * 1024)
#define VOLVOXAI_SHADER_STORE_MAX_OVERRIDE_TOTAL ((size_t)128 * 1024 * 1024)

#if !defined(VOLVOXAI_EMBEDDED_SHADER_PACK_VERSION)
#error "embedded_shaders.h is missing its shader pack version"
#elif VOLVOXAI_EMBEDDED_SHADER_PACK_VERSION != 1
#error "unsupported embedded shader pack version"
#endif

typedef struct ShaderBlockCache {
    unsigned char* data;
    /* 0 = untouched, 1 = decoded, -1 = permanently failed. */
    int state;
} ShaderBlockCache;

typedef struct ShaderOverrideCache {
    char* file_path;
    unsigned char* data;
    size_t size;
    struct ShaderOverrideCache* next;
} ShaderOverrideCache;

static pthread_mutex_t g_shader_store_mutex = PTHREAD_MUTEX_INITIALIZER;
static ShaderBlockCache* g_block_cache;
static size_t g_block_cache_count;
static ShaderOverrideCache* g_override_cache;
static size_t g_override_cache_bytes;
static char* g_configured_override_root;
static int g_env_override_used_logged;
static int g_env_override_fallback_logged;
static int g_configured_override_fallback_logged;
static int g_crc32_initialized;

static size_t bounded_strlen(const char* value, size_t maximum) {
    size_t length = 0;
    if (!value) return maximum + 1;
    while (length <= maximum && value[length] != '\0') length++;
    return length;
}

static int valid_lookup_path(const char* path) {
    size_t length = bounded_strlen(path, VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE);
    if (length == 0 || length > VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE ||
        path[0] == '/' || path[0] == '\\') {
        return 0;
    }

    const char* component = path;
    for (const char* cursor = path;; cursor++) {
        if (*cursor == '\\') return 0;
        if (*cursor == '/' || *cursor == '\0') {
            size_t component_length = (size_t)(cursor - component);
            if (component_length == 0 ||
                (component_length == 1 && component[0] == '.') ||
                (component_length == 2 && component[0] == '.' && component[1] == '.')) {
                return 0;
            }
            if (*cursor == '\0') break;
            component = cursor + 1;
        }
    }
    return 1;
}

VolvoxAIShaderStoreResult volvoxai_shader_store_set_override_root(const char* root) {
    char* copy = NULL;
    if (root && root[0] != '\0') {
        size_t length = bounded_strlen(root, VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE);
        if (length == 0 || length > VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE) {
            return VOLVOXAI_SHADER_STORE_INVALID_ARGUMENT;
        }
        copy = (char*)malloc(length + 1);
        if (!copy) return VOLVOXAI_SHADER_STORE_OUT_OF_MEMORY;
        memcpy(copy, root, length + 1);
    }

    pthread_mutex_lock(&g_shader_store_mutex);
    char* previous = g_configured_override_root;
    g_configured_override_root = copy;
    pthread_mutex_unlock(&g_shader_store_mutex);
    free(previous);
    return VOLVOXAI_SHADER_STORE_OK;
}

static const VolvoxAIEmbeddedShaderRecord* find_embedded_record(const char* path) {
    size_t low = 0;
    size_t high = volvoxai_embedded_shader_record_count;
    while (low < high) {
        size_t middle = low + (high - low) / 2;
        int comparison = strcmp(path, volvoxai_embedded_shader_records[middle].path);
        if (comparison == 0) return &volvoxai_embedded_shader_records[middle];
        if (comparison < 0) high = middle;
        else low = middle + 1;
    }
    return NULL;
}

static char* join_override_path(const char* root, const char* path) {
    size_t root_length = bounded_strlen(root, VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE);
    size_t path_length = strlen(path);
    if (root_length == 0 || root_length > VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE) return NULL;

    int add_separator = root[root_length - 1] != '/';
    if (path_length > VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE ||
        root_length > VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE - path_length) {
        return NULL;
    }
    size_t length = root_length + path_length;
    if (add_separator && length == VOLVOXAI_SHADER_STORE_MAX_PATH_SIZE) return NULL;
    length += (size_t)add_separator;
    char* joined = (char*)malloc(length + 1);
    if (!joined) return NULL;
    memcpy(joined, root, root_length);
    if (add_separator) joined[root_length++] = '/';
    memcpy(joined + root_length, path, path_length + 1);
    return joined;
}

static ShaderOverrideCache* find_override_cache(const char* file_path) {
    for (ShaderOverrideCache* item = g_override_cache; item; item = item->next) {
        if (strcmp(item->file_path, file_path) == 0) return item;
    }
    return NULL;
}

/* Return 1 on success, 0 when the configured override cannot supply path. */
static int try_override_locked(
    const char* root, const char* path, VolvoxAIShaderView* out_view) {
    char* file_path = join_override_path(root, path);
    if (!file_path) return 0;

    ShaderOverrideCache* cached = find_override_cache(file_path);
    if (cached) {
        free(file_path);
        out_view->data = cached->data;
        out_view->size = cached->size;
        return 1;
    }

    FILE* file = fopen(file_path, "rb");
    if (!file) {
        free(file_path);
        return 0;
    }
    if (fseek(file, 0, SEEK_END) != 0) {
        fclose(file);
        free(file_path);
        return 0;
    }
    long end = ftell(file);
    if (end <= 0 || (unsigned long)end > VOLVOXAI_SHADER_STORE_MAX_OVERRIDE_SIZE ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        free(file_path);
        return 0;
    }

    size_t size = (size_t)end;
    if (size > VOLVOXAI_SHADER_STORE_MAX_OVERRIDE_TOTAL - g_override_cache_bytes) {
        fclose(file);
        free(file_path);
        return 0;
    }
    unsigned char* data = (unsigned char*)malloc(size);
    if (!data) {
        fclose(file);
        free(file_path);
        return 0;
    }
    size_t read_size = fread(data, 1, size, file);
    int read_failed = read_size != size || ferror(file);
    fclose(file);
    if (read_failed) {
        free(data);
        free(file_path);
        return 0;
    }

    ShaderOverrideCache* entry = (ShaderOverrideCache*)malloc(sizeof(*entry));
    if (!entry) {
        free(data);
        free(file_path);
        return 0;
    }
    entry->file_path = file_path;
    entry->data = data;
    entry->size = size;
    entry->next = g_override_cache;
    g_override_cache = entry;
    g_override_cache_bytes += size;
    out_view->data = data;
    out_view->size = size;
    return 1;
}

static VolvoxAIShaderStoreResult ensure_block_cache_locked(void) {
    if (g_block_cache) return VOLVOXAI_SHADER_STORE_OK;
    if (volvoxai_embedded_shader_block_count == 0 ||
        volvoxai_embedded_shader_block_count > VOLVOXAI_SHADER_STORE_MAX_BLOCK_COUNT) {
        return VOLVOXAI_SHADER_STORE_INVALID_DATA;
    }
    g_block_cache = (ShaderBlockCache*)calloc(
        volvoxai_embedded_shader_block_count, sizeof(*g_block_cache));
    if (!g_block_cache) return VOLVOXAI_SHADER_STORE_OUT_OF_MEMORY;
    g_block_cache_count = volvoxai_embedded_shader_block_count;
    return VOLVOXAI_SHADER_STORE_OK;
}

static VolvoxAIShaderStoreResult decode_block_locked(uint32_t block_index) {
    VolvoxAIShaderStoreResult result = ensure_block_cache_locked();
    if (result != VOLVOXAI_SHADER_STORE_OK) return result;
    if ((size_t)block_index >= g_block_cache_count) {
        return VOLVOXAI_SHADER_STORE_INVALID_DATA;
    }

    ShaderBlockCache* cache = &g_block_cache[block_index];
    if (cache->state == 1) return VOLVOXAI_SHADER_STORE_OK;
    if (cache->state == -1) return VOLVOXAI_SHADER_STORE_INVALID_DATA;

    const VolvoxAIEmbeddedShaderBlock* block =
        &volvoxai_embedded_shader_blocks[block_index];
    if (!block->compressed_data || block->compressed_size == 0 ||
        block->compressed_size > VOLVOXAI_SHADER_STORE_MAX_BLOCK_SIZE ||
        block->uncompressed_size == 0 ||
        block->uncompressed_size > VOLVOXAI_SHADER_STORE_MAX_BLOCK_SIZE) {
        cache->state = -1;
        return VOLVOXAI_SHADER_STORE_INVALID_DATA;
    }

    unsigned char* output = (unsigned char*)malloc(block->uncompressed_size);
    if (!output) return VOLVOXAI_SHADER_STORE_OUT_OF_MEMORY;

    if (!g_crc32_initialized) {
        xz_crc32_init();
        g_crc32_initialized = 1;
    }
    struct xz_dec* decoder = xz_dec_init(XZ_SINGLE, 0);
    if (!decoder) {
        free(output);
        return VOLVOXAI_SHADER_STORE_OUT_OF_MEMORY;
    }
    struct xz_buf buffer = {
        .in = block->compressed_data,
        .in_pos = 0,
        .in_size = block->compressed_size,
        .out = output,
        .out_pos = 0,
        .out_size = block->uncompressed_size,
    };
    enum xz_ret xz_result = xz_dec_run(decoder, &buffer);
    xz_dec_end(decoder);
    if (xz_result != XZ_STREAM_END || buffer.in_pos != buffer.in_size ||
        buffer.out_pos != buffer.out_size) {
        free(output);
        cache->state = -1;
        fprintf(stderr,
                "[VolvoxAI] Embedded shader block %u failed XZ/CRC32 validation\n",
                (unsigned)block_index);
        return VOLVOXAI_SHADER_STORE_INVALID_DATA;
    }

    cache->data = output;
    cache->state = 1;
    return VOLVOXAI_SHADER_STORE_OK;
}

static VolvoxAIShaderStoreResult embedded_view_locked(
    const VolvoxAIEmbeddedShaderRecord* record, VolvoxAIShaderView* out_view) {
    if (record->block >= volvoxai_embedded_shader_block_count ||
        record->offset % VOLVOXAI_EMBEDDED_SHADER_ALIGNMENT != 0) {
        return VOLVOXAI_SHADER_STORE_INVALID_DATA;
    }
    const VolvoxAIEmbeddedShaderBlock* block =
        &volvoxai_embedded_shader_blocks[record->block];
    if (record->size == 0 || record->offset > block->uncompressed_size ||
        record->size > block->uncompressed_size - record->offset) {
        return VOLVOXAI_SHADER_STORE_INVALID_DATA;
    }

    VolvoxAIShaderStoreResult result = decode_block_locked(record->block);
    if (result != VOLVOXAI_SHADER_STORE_OK) return result;
    out_view->data = g_block_cache[record->block].data + record->offset;
    out_view->size = record->size;
    return VOLVOXAI_SHADER_STORE_OK;
}

VolvoxAIShaderStoreResult volvoxai_shader_store_get(
    const char* path, VolvoxAIShaderView* out_view) {
    if (!out_view) return VOLVOXAI_SHADER_STORE_INVALID_ARGUMENT;
    out_view->data = NULL;
    out_view->size = 0;
    if (!valid_lookup_path(path)) return VOLVOXAI_SHADER_STORE_INVALID_ARGUMENT;

    pthread_mutex_lock(&g_shader_store_mutex);
    const VolvoxAIEmbeddedShaderRecord* record = find_embedded_record(path);
    if (!record) {
        pthread_mutex_unlock(&g_shader_store_mutex);
        return VOLVOXAI_SHADER_STORE_NOT_FOUND;
    }

    const char* env_override_root = getenv("VOLVOXAI_SHADER_DIR");
    if (env_override_root && env_override_root[0] != '\0') {
        if (try_override_locked(env_override_root, path, out_view)) {
            if (!g_env_override_used_logged) {
                fprintf(stderr,
                        "[VolvoxAI] Using shader override: VOLVOXAI_SHADER_DIR=%s\n",
                        env_override_root);
                g_env_override_used_logged = 1;
            }
            pthread_mutex_unlock(&g_shader_store_mutex);
            return VOLVOXAI_SHADER_STORE_OK;
        }
        if (!g_env_override_fallback_logged) {
            fprintf(stderr,
                    "[VolvoxAI] Shader override '%s' is unavailable; using embedded shaders\n",
                    env_override_root);
            g_env_override_fallback_logged = 1;
        }
    } else if (g_configured_override_root) {
        if (try_override_locked(g_configured_override_root, path, out_view)) {
            pthread_mutex_unlock(&g_shader_store_mutex);
            return VOLVOXAI_SHADER_STORE_OK;
        }
        if (!g_configured_override_fallback_logged) {
            fprintf(stderr,
                    "[VolvoxAI] Configured shader root '%s' is unavailable; using embedded shaders\n",
                    g_configured_override_root);
            g_configured_override_fallback_logged = 1;
        }
    }

    VolvoxAIShaderStoreResult result = embedded_view_locked(record, out_view);
    pthread_mutex_unlock(&g_shader_store_mutex);
    return result;
}

void volvoxai_shader_store_shutdown(void) {
    pthread_mutex_lock(&g_shader_store_mutex);
    for (size_t i = 0; i < g_block_cache_count; i++) free(g_block_cache[i].data);
    free(g_block_cache);
    g_block_cache = NULL;
    g_block_cache_count = 0;

    while (g_override_cache) {
        ShaderOverrideCache* next = g_override_cache->next;
        free(g_override_cache->file_path);
        free(g_override_cache->data);
        free(g_override_cache);
        g_override_cache = next;
    }
    g_override_cache_bytes = 0;
    free(g_configured_override_root);
    g_configured_override_root = NULL;
    g_env_override_used_logged = 0;
    g_env_override_fallback_logged = 0;
    g_configured_override_fallback_logged = 0;
    pthread_mutex_unlock(&g_shader_store_mutex);
}
