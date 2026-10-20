#ifndef VOLVOXAI_CUDA_ZERO_RANGES_H
#define VOLVOXAI_CUDA_ZERO_RANGES_H

#include <stdint.h>

/* A launch-owned parameter image, never a persistent device allocation. */
#define VX_CUDA_ZERO_RANGE_CAPACITY 64u
typedef struct {
    uint64_t address;
    uint64_t bytes;
} VxCudaZeroRange;
typedef struct {
    VxCudaZeroRange ranges[VX_CUDA_ZERO_RANGE_CAPACITY];
} VxCudaZeroRanges;

#endif
