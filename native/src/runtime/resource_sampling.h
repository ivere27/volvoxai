/* Private fixed-size observations. No engine object, allocation or GPU work
 * is retained by a sample. Public contracts are generated from the proto. */
#ifndef VOLVOXAI_RESOURCE_SAMPLING_H
#define VOLVOXAI_RESOURCE_SAMPLING_H

#include "vx_lifecycle_types.h"

#define VX_RESOURCE_MAX_GPUS 8

typedef struct {
    VxObservationStatus status;
    uint64_t linear, allocated, free, metadata, prefix, slack, untracked;
    uint64_t largest_free, allocated_blocks, free_blocks;
} VxWasmMemorySample;

typedef struct {
    VxObservationStatus status;
    uint64_t process_time_ns;
    uint32_t online_processors;
    const char* source;
} VxProcessCpuSample;

typedef struct {
    char id[96], name[96];
    VxObservationStatus utilization, memory;
    double compute_active, memory_active;
    uint64_t capacity_bytes, used_bytes;
} VxGpuResourceSample;

typedef struct {
    uint64_t start_ns, end_ns;
    VxWasmMemorySample wasm;
    VxProcessMemorySampleV1 process;
    VxProcessCpuSample cpu;
    VxObservationStatus gpu_status;
    const char* gpu_source;
    uint32_t gpu_count;
    int gpu_truncated;
    VxGpuResourceSample gpus[VX_RESOURCE_MAX_GPUS];
} VxResourceSample;

void vx_resource_sample(VxResourceSample* sample, int cpu, int gpu);
#if defined(__wasm__)
void vx_wasm_memory_sample(VxWasmMemorySample* sample);
#endif

#endif
