#ifndef VOLVOXAI_FIXED_PARTITION_H
#define VOLVOXAI_FIXED_PARTITION_H

#include <stddef.h>
#include <stdlib.h>

/* Private immutable placement indexed by the preserved logical node order. */
typedef struct {
    size_t node_count;
    size_t cpu_count;
    unsigned char cpu_nodes[];
} VxFixedPartition;

static inline int vx_fixed_partition_cpu(const VxFixedPartition* plan, size_t index) {
    return plan && index < plan->node_count && plan->cpu_nodes[index] != 0;
}

static inline size_t vx_fixed_partition_bytes(const VxFixedPartition* plan) {
    return plan ? sizeof(*plan) + plan->node_count : 0u;
}

static inline void vx_fixed_partition_destroy(VxFixedPartition* plan) {
    free(plan);
}

#endif
