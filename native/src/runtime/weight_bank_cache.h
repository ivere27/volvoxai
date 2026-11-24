#ifndef VOLVOXAI_WEIGHT_BANK_CACHE_H
#define VOLVOXAI_WEIGHT_BANK_CACHE_H

#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>

/* A context cache keeps the complete immutable source shape. Only the
 * separately allocated device slots change their expert assignment. */
typedef struct {
    char bank[128];
    const void* weights[3];
    uint32_t weight_dtype[3];
    size_t weight_bytes[3];
    const float* bias;
    size_t bias_bytes;
    uint32_t operator_kind;
    uint32_t experts;
    uint32_t d_in;
    uint32_t hidden;
    uint32_t d_out;
    uint32_t max_top_k;
    uint32_t resident_slots;
} VxWeightBankCacheGroup;

typedef struct {
    VxWeightBankCacheGroup* groups;
    size_t count;
} VxWeightBankCachePlan;

static inline void vx_weight_bank_cache_plan_destroy(VxWeightBankCachePlan* plan) {
    if (!plan) return;
    free(plan->groups);
    free(plan);
}

#endif
