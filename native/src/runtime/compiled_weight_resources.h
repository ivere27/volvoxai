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

#ifdef __cplusplus
}
#endif

#endif
