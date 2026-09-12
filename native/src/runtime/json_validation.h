#ifndef VOLVOXAI_JSON_VALIDATION_H
#define VOLVOXAI_JSON_VALIDATION_H

#include "cJSON.h"
#include <stdlib.h>
#include <string.h>

/* cJSON exposes decoded JSON strings only as NUL-terminated C strings and does
 * not retain their decoded byte length. A decoded U+0000 would therefore make
 * every later lookup, comparison, and wire adapter observe a truncated value.
 *
 * This length-aware pre-parse scan is intentionally narrower than a JSON
 * parser. A literal zero byte is invalid JSON and is rejected regardless of
 * parser state, preventing cJSON from accepting a valid prefix before hidden
 * trailing bytes. The valid JSON escape "\\u0000" is identified only inside a
 * JSON string. Escaped backslashes are consumed in pairs, so the JSON spelling
 * "\\\\u0000" (a literal backslash followed by the characters u0000) remains
 * accepted. Full syntax and UTF-8 validation remain the responsibility of
 * cJSON. */
static inline int vx_json_text_contains_decoded_nul(
        const unsigned char* bytes, size_t byte_count) {
    int in_string = 0;
    if (!bytes) return 0;
    for (size_t index = 0; index < byte_count; index++) {
        unsigned char value = bytes[index];
        if (value == 0u) return 1;
        if (!in_string) {
            if (value == (unsigned char)'"') in_string = 1;
            continue;
        }
        if (value == (unsigned char)'"') {
            in_string = 0;
            continue;
        }
        if (value == (unsigned char)'\\') {
            unsigned char escaped;
            if (++index >= byte_count) break;
            escaped = bytes[index];
            if (escaped == 0u) return 1;
            if (escaped == (unsigned char)'u' &&
                byte_count - index > 4u &&
                bytes[index + 1u] == (unsigned char)'0' &&
                bytes[index + 2u] == (unsigned char)'0' &&
                bytes[index + 3u] == (unsigned char)'0' &&
                bytes[index + 4u] == (unsigned char)'0') return 1;
            continue;
        }
    }
    return 0;
}

/* cJSON intentionally preserves duplicate object members. Model and tensor
 * formats use JSON objects as maps, so reject ambiguity before any lookup can
 * select one duplicate and silently ignore another. */
static int vx_json_key_pointer_compare(const void* left, const void* right) {
    const char* const* l = (const char* const*)left;
    const char* const* r = (const char* const*)right;
    return strcmp(*l, *r);
}

static int vx_json_object_keys_unique_recursive(const cJSON* value) {
    if (cJSON_IsObject(value)) {
        size_t count = 0;
        for (const cJSON* item = value->child; item; item = item->next) {
            if (!item->string || count == (size_t)-1) return 0;
            count++;
        }
        if (count > (size_t)-1 / sizeof(const char*)) return 0;
        const char** keys = count
            ? (const char**)malloc(count * sizeof(*keys)) : NULL;
        if (count && !keys) return 0;
        size_t index = 0;
        for (const cJSON* item = value->child; item; item = item->next) {
            keys[index++] = item->string;
        }
        if (count > 1) {
            qsort(keys, count, sizeof(*keys), vx_json_key_pointer_compare);
            for (index = 1; index < count; index++) {
                if (!strcmp(keys[index - 1], keys[index])) {
                    free(keys);
                    return 0;
                }
            }
        }
        free(keys);
        for (const cJSON* item = value->child; item; item = item->next) {
            if (!vx_json_object_keys_unique_recursive(item)) return 0;
        }
    } else if (cJSON_IsArray(value)) {
        for (const cJSON* item = value->child; item; item = item->next) {
            if (!vx_json_object_keys_unique_recursive(item)) return 0;
        }
    }
    return 1;
}

#endif
