#ifndef VOLVOX_RUNTIME_SEQUENCE_RUNTIME_H
#define VOLVOX_RUNTIME_SEQUENCE_RUNTIME_H

#include "engine_internal.h"

enum {
    VX_SEQUENCE_ERROR = -1,
    VX_SEQUENCE_NOT_HANDLED = 0,
    VX_SEQUENCE_HANDLED = 1
};

/*
 * Validate and execute a portable sequence operator.  The return value makes
 * this safe to call once from the generic CPU dispatch chain: unrelated ops
 * return NOT_HANDLED without touching tensors, while malformed sequence ops
 * return ERROR.  backend_name receives a stable profiler label on success.
 */
int vx_sequence_node_run(Node* node, long element_offset, int element_count,
                         const char** backend_name);

#endif
