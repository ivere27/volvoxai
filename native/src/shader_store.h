#ifndef VOLVOXAI_SHADER_STORE_H
#define VOLVOXAI_SHADER_STORE_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VolvoxAIShaderView {
    /* Shader bytes are not guaranteed to have a trailing NUL byte. */
    const unsigned char* data;
    size_t size;
} VolvoxAIShaderView;

typedef enum VolvoxAIShaderStoreResult {
    VOLVOXAI_SHADER_STORE_OK = 0,
    VOLVOXAI_SHADER_STORE_INVALID_ARGUMENT = -1,
    VOLVOXAI_SHADER_STORE_NOT_FOUND = -2,
    VOLVOXAI_SHADER_STORE_OUT_OF_MEMORY = -3,
    VOLVOXAI_SHADER_STORE_INVALID_DATA = -4
} VolvoxAIShaderStoreResult;

/*
 * Set a process-wide development shader root, copying root before returning.
 * Passing NULL or an empty string clears it. A nonempty VOLVOXAI_SHADER_DIR
 * takes precedence over this root; failure of that explicit environment
 * override falls directly back to embedded data.
 */
VolvoxAIShaderStoreResult volvoxai_shader_store_set_override_root(const char* root);

/*
 * Look up a compiled-root-relative path such as "spv/add.spv".
 *
 * The lookup order is a nonempty VOLVOXAI_SHADER_DIR, the programmatic root,
 * then the embedded record. Missing, unreadable, empty, or oversized override
 * files fall back to the embedded record. Embedded backend/scope blocks are
 * decompressed only when a record in that block is requested.
 *
 * This function is thread-safe. A successful view remains valid until
 * volvoxai_shader_store_shutdown(); shutdown must not race with code using a
 * previously returned view.
 */
VolvoxAIShaderStoreResult volvoxai_shader_store_get(
    const char* path, VolvoxAIShaderView* out_view);

/* Release all decompressed blocks and cached override files. */
void volvoxai_shader_store_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif
