#ifndef VOLVOXAI_CUDA_ROPE_CACHE_H
#define VOLVOXAI_CUDA_ROPE_CACHE_H

#include <stddef.h>
#include <stdint.h>

/* Small immutable tables are shared by theta/rotary-width, not by node or
 * shape. Once published, an entry keeps the same device address and contents
 * until its context retires. The final table is a synchronous overflow path;
 * it never enters a captured executable. */
#define CUDA_ROPE_POWER_CAPACITY 64u
#define CUDA_ROPE_CACHE_CAPACITY 8u
#define CUDA_ROPE_POWER_TABLE_BYTES \
    ((size_t)CUDA_ROPE_POWER_CAPACITY * sizeof(float))
#define CUDA_ROPE_POWER_STORAGE_BYTES \
    ((size_t)(CUDA_ROPE_CACHE_CAPACITY + 1u) * CUDA_ROPE_POWER_TABLE_BYTES)

typedef struct {
    uint32_t theta_bits;
    uint32_t rotary_width;
} CudaRopePowerKey;

typedef struct {
    CudaRopePowerKey keys[CUDA_ROPE_CACHE_CAPACITY];
    uint32_t count;
} CudaRopePowerCache;

#endif
