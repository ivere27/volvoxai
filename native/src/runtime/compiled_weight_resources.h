#ifndef VOLVOXAI_COMPILED_WEIGHT_RESOURCES_H
#define VOLVOXAI_COMPILED_WEIGHT_RESOURCES_H

#include "safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VxConvertedWeight {
    char name[128];
    const float* data;
    size_t byte_size;
    size_t source_byte_size;
} VxConvertedWeight;

/* Immutable F32 materializations derived from one exact compiled weight
 * revision. BF16 sources widen on every backend; CPU also prepares F16.
 * Selected banks remain Context-owned copy-on-write overlays. */
typedef struct VxCompiledWeightConversions {
    VxConvertedWeight* weights;
    size_t count;
    uint64_t allocated_bytes;
    uint64_t materialized_bytes;
    uint64_t materialization_count;
} VxCompiledWeightConversions;

int vx_compiled_weight_conversions_create(
    const SafetensorsFile* files,
    size_t file_count,
    const char* const* excluded_weight_names,
    size_t excluded_weight_count,
    int include_f16,
    VxCompiledWeightConversions** out_store);

const VxConvertedWeight* vx_compiled_weight_conversions_find(
    const VxCompiledWeightConversions* store,
    const char* name);

void vx_compiled_weight_conversions_destroy(VxCompiledWeightConversions* store);

/* Immutable portable F32 transforms prepared before execution admission.
 * The engine owns this store with its exact model revision; lookups never
 * allocate or retain activations. Unsupported or over-budget packs fall back
 * to the ordinary kernel. */
typedef struct VxPreparedF32Weight {
    const float* source;
    float* packed;
    uint32_t k, n;
    int out_in;
    struct VxPreparedF32Weight* next;
} VxPreparedF32Weight;

typedef struct VxPreparedF32Weights {
    VxPreparedF32Weight* first;
    uint64_t allocated_bytes;
} VxPreparedF32Weights;

#define VX_PREPARED_F32_WEIGHT_LIMIT_BYTES (64u * 1024u * 1024u)
/* Allocation-independent upper bound. Repeated candidates may overcount, but
 * failed allocations cannot reduce the admission bound for a later engine. */
uint64_t vx_prepared_f32_weights_bound_add(uint64_t used, uint32_t k, uint32_t n);
const float* vx_prepared_f32_weights_find(const VxPreparedF32Weights* store,
    const float* source, uint32_t k, uint32_t n, int out_in);
const float* vx_prepared_f32_weights_add(VxPreparedF32Weights** store,
    const float* source, uint32_t k, uint32_t n, int out_in);
void vx_prepared_f32_weights_clear(VxPreparedF32Weights** store);

#ifdef __cplusplus
}
#endif

#endif
