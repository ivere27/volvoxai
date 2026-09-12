#ifndef VOLVOXAI_COMPILED_WEIGHT_RESOURCES_H
#define VOLVOXAI_COMPILED_WEIGHT_RESOURCES_H

#include "safetensors.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct VxCompiledCpuWeight {
    char name[128];
    const float* data;
    size_t byte_size;
    size_t source_byte_size;
} VxCompiledCpuWeight;

/* Immutable CPU-only materializations derived from one exact compiled weight
 * revision. The store deliberately excludes selected banks: those remain
 * Context-owned copy-on-write overlays. */
typedef struct VxCompiledCpuWeightStore {
    VxCompiledCpuWeight* weights;
    size_t count;
    uint64_t allocated_bytes;
    uint64_t materialized_bytes;
    uint64_t materialization_count;
} VxCompiledCpuWeightStore;

int vx_compiled_cpu_weight_store_create(
    const SafetensorsFile* files,
    size_t file_count,
    const char* const* excluded_weight_names,
    size_t excluded_weight_count,
    VxCompiledCpuWeightStore** out_store);

const VxCompiledCpuWeight* vx_compiled_cpu_weight_store_find(
    const VxCompiledCpuWeightStore* store,
    const char* name);

void vx_compiled_cpu_weight_store_destroy(VxCompiledCpuWeightStore* store);

#ifdef __cplusplus
}
#endif

#endif
