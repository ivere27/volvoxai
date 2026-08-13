#ifndef VOLVOXAI_NATIVE_DYNAMIC_BATCH_EVIDENCE_TOKENS_H
#define VOLVOXAI_NATIVE_DYNAMIC_BATCH_EVIDENCE_TOKENS_H

#include <stddef.h>
#include <string.h>

static size_t vx_native_batch_evidence_key_count(const char* evidence,
                                                 const char* key) {
    size_t key_length;
    size_t count = 0u;
    const char* cursor;
    if (!evidence || !key || !key[0]) return 0u;
    key_length = strlen(key);
    cursor = evidence;
    while (*cursor) {
        const char* end = strchr(cursor, ';');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length >= key_length + 1u &&
            !memcmp(cursor, key, key_length) && cursor[key_length] == '=')
            count++;
        if (!end) break;
        cursor = end + 1u;
    }
    return count;
}

static int vx_native_batch_evidence_token(const char* evidence,
                                          const char* key, char* value,
                                          size_t capacity) {
    size_t key_length;
    const char* cursor;
    int found = 0;
    if (!evidence || !key || !key[0] || !value || capacity < 2u) return 0;
    key_length = strlen(key);
    cursor = evidence;
    while (*cursor) {
        const char* end = strchr(cursor, ';');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length >= key_length + 1u &&
            !memcmp(cursor, key, key_length) && cursor[key_length] == '=') {
            size_t value_length = length - key_length - 1u;
            if (!value_length || found || value_length >= capacity) return 0;
            memcpy(value, cursor + key_length + 1u, value_length);
            value[value_length] = '\0';
            found = 1;
        }
        if (!end) break;
        cursor = end + 1u;
    }
    return found;
}

#endif
