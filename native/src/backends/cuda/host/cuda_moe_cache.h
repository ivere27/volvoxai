#ifndef VOLVOXAI_CUDA_MOE_CACHE_H
#define VOLVOXAI_CUDA_MOE_CACHE_H
#include <stddef.h>
#include <stdint.h>
#include "cuda_transfer_limits.h"

typedef struct CudaMoeCache {
    struct CudaMoeCache* next;
    const void* source[3];
    uint32_t dtype[3];
    const float* bias;
    size_t expert_bytes[3], bias_bytes;
    size_t host_bytes, device_bytes;
    uintptr_t allocation, weight[3], bias_device, routes_device;
    uint32_t experts, d_in, hidden, d_out, maximum_top_k, slots;
    uint32_t* expert_ids;
    uint64_t* ages;
    unsigned char* pinned;
    float* routes_host;
    uint64_t clock;
} CudaMoeCache;
#endif
