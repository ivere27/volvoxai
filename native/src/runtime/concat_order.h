/* Private canonical operand order shared by graph loading and byte routes. */
#ifndef VOLVOXAI_CONCAT_ORDER_H
#define VOLVOXAI_CONCAT_ORDER_H
#include "runtime_state.h"
#include <string.h>

static inline int vx_concat_numeric_input_key(const char* key, uint64_t* value) {
    uint64_t parsed = 0;
    const char* cursor;
    int overflow = 0;
    if (!key || strncmp(key, "input", 5) || !key[5]) return 0;
    for (cursor = key + 5; *cursor; cursor++) {
        unsigned digit;
        if (*cursor < '0' || *cursor > '9') return 0;
        digit = (unsigned)(*cursor - '0');
        if (parsed > (UINT64_MAX - digit) / 10u) overflow = 1;
        else if (!overflow) parsed = parsed * 10u + digit;
    }
    if (value) *value = overflow ? UINT64_MAX : parsed;
    return 1;
}

static inline int vx_concat_key_compare(const char* left, const char* right) {
    uint64_t left_index;
    uint64_t right_index;
    int left_numeric = vx_concat_numeric_input_key(left, &left_index);
    int right_numeric = vx_concat_numeric_input_key(right, &right_index);
    if (left_numeric && right_numeric) {
        if (left_index < right_index) return -1;
        if (left_index > right_index) return 1;
        return 0;  /* JavaScript's stable sort retains declaration order on ties. */
    }
    return strcmp(left, right);
}

static inline int vx_concat_preferred_key(VxPortKind port) {
    static const VxPortKind preferred[] = {VX_PORT_INPUT, VX_PORT_A, VX_PORT_B, VX_PORT_C, VX_PORT_D, VX_PORT_E, VX_PORT_F, VX_PORT_G, VX_PORT_H};
    for (size_t i = 0; i < sizeof(preferred) / sizeof(preferred[0]); ++i)
        if (port == preferred[i]) return (int)i;
    return 9;
}

static inline int vx_concat_sort(Node* node) {
    if (!node || node->nin <= 0 || !node->ins) return -1;
    for (int left = 0; left < node->nin; left++) {
        if (!node->ins[left].key[0]) return -1;
        for (int right = left + 1; right < node->nin; right++) {
            if (!strcmp(node->ins[left].key, node->ins[right].key)) return -1;
        }
    }
    for (int input_index = 1; input_index < node->nin; input_index++) {
        Ref current = node->ins[input_index];
        int insert = input_index;
        while (insert > 0) {
            const Ref* previous = &node->ins[insert - 1];
            int left = vx_concat_preferred_key(current.port);
            int right = vx_concat_preferred_key(previous->port);
            if (left > right || (left == right &&
                vx_concat_key_compare(current.key, previous->key) >= 0)) break;
            node->ins[insert] = *previous;
            insert--;
        }
        node->ins[insert] = current;
    }
    return 0;
}

#endif
