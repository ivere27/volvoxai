#ifndef VOLVOXAI_LLM_WORKSPACE_H
#define VOLVOXAI_LLM_WORKSPACE_H

#include <stdint.h>

/* Serial operators reuse two persistent buffers. Their largest extents may
 * belong to different operators, so retain each maximum independently. */
static inline int vx_llm_workspace_align(uint64_t bytes, uint64_t alignment,
                                        uint64_t* aligned) {
    if (!alignment || !aligned) return 0;
    uint64_t tail = bytes % alignment;
    uint64_t padding = tail ? alignment - tail : 0u;
    if (bytes > UINT64_MAX - padding) return 0;
    *aligned = bytes + padding;
    return 1;
}

static inline int vx_llm_workspace_capacity(const uint64_t slots[2],
        uint64_t alignment, uint64_t dummy_bytes, uint32_t dummy_count,
        uint64_t* capacity) {
    uint64_t total = 0u, aligned;
    if (!slots || !capacity || !alignment) return 0;
    for (unsigned slot = 0; slot < 2u; ++slot) {
        if (!vx_llm_workspace_align(slots[slot], alignment, &aligned) ||
            aligned > UINT64_MAX - total) return 0;
        total += aligned;
    }
    if (!vx_llm_workspace_align(dummy_bytes, alignment, &aligned) ||
        (dummy_count && aligned > (UINT64_MAX - total) / dummy_count)) return 0;
    *capacity = total + aligned * dummy_count;
    return 1;
}

static inline int vx_llm_workspace_maxima(uint64_t maximum[2],
                                         const uint64_t node[2]) {
    uint64_t candidate[2], total;
    if (!maximum || !node) return 0;
    for (unsigned slot = 0; slot < 2u; ++slot)
        candidate[slot] = node[slot] > maximum[slot] ? node[slot] : maximum[slot];
    if (!vx_llm_workspace_capacity(candidate, 1u, 0u, 0u, &total)) return 0;
    maximum[0] = candidate[0]; maximum[1] = candidate[1];
    return 1;
}

#endif
