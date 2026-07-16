#ifndef VOLVOXAI_JSON_VALIDATION_H
#define VOLVOXAI_JSON_VALIDATION_H

#include "cJSON.h"
#include <stdlib.h>
#include <string.h>

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
