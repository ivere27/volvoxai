#ifndef VOLVOXAI_CUDA_DENSE_LAUNCH_PLAN_H
#define VOLVOXAI_CUDA_DENSE_LAUNCH_PLAN_H
#include <stddef.h>
#include <stdint.h>
#include "packed_launch_config.h"

/* Adaptive shared matrix CTAs must cover the device before reducing output parallelism.
 * Device capacity and the compiled PTX target are fixed for the context, so
 * graph preparation and replay select identical routes without calibration. */
static inline uint64_t vx_cuda_packed_linear_matrix_tiles(
        uint64_t rows, uint32_t output_width, uint32_t activation_bits,
        uint32_t sm_count, uint32_t compiled_arch) {
    if(rows<16u || !output_width || !sm_count || compiled_arch<75u)return 0u;
    uint32_t tile_columns=vx_cuda_packed_linear_tile_columns(
        rows,output_width,activation_bits,sm_count);
    if(!tile_columns)return 0u;
    uint64_t row_tiles=rows/16u+(rows%16u!=0u);
    uint64_t column_tiles=output_width/tile_columns+(output_width%tile_columns!=0u);
    if(row_tiles>UINT64_MAX/column_tiles)return 0u;
    uint64_t tiles=row_tiles*column_tiles;
    return tiles>=sm_count?tiles:0u;
}

/* Row caching stays within 16KiB per CTA. Larger K retains group streaming,
 * so model shape never creates an unbounded allocation or a new owner. */
static inline uint32_t vx_cuda_packed_linear_row_cache_bytes(uint32_t input_width) {
    if(!input_width || input_width>16384u)return 0u;
    return ((input_width+127u)/128u)*128u;
}

/* Fixed for the context lifetime; absent scratch keeps the ordinary tiled route. */
#define VX_CUDA_MATMUL_WORKSPACE_BYTES (2u * 1024u * 1024u)
#define VX_CUDA_CONTEXT_WORKSPACE_BYTES (32u * 1024u * 1024u)
#define VX_CUDA_PACKED_WORKSPACE_BYTES VX_CUDA_CONTEXT_WORKSPACE_BYTES
#define VX_CUDA_MATMUL_CALIBRATION_OUTPUT_BYTES (1u * 1024u * 1024u)
static inline int vx_cuda_matmul_split_slices_supported(uint32_t split_k_slices) {
    return split_k_slices == 2u || split_k_slices == 4u ||
        split_k_slices == 8u || split_k_slices == 16u;
}
static inline size_t vx_cuda_matmul_split_workspace_bytes(
        uint32_t rows, uint32_t input_width, uint32_t output_width,
        uint32_t split_k_slices) {
    if (rows < 17u || rows > 32u || input_width < 1024u ||
        output_width < 1024u || !vx_cuda_matmul_split_slices_supported(split_k_slices)) return 0u;
    uint64_t bytes = (uint64_t)rows * output_width * split_k_slices * sizeof(float);
    return bytes <= VX_CUDA_MATMUL_WORKSPACE_BYTES ? (size_t)bytes : 0u;
}
/* A two-wave starting estimate, not an optimality guarantee. Calibration may
 * choose any capacity-valid candidate, including ones excluded by this minimum
 * of two K16 stages per inner slice. Unknown device capacity keeps tiled GEMM. */
static inline uint32_t vx_cuda_matmul_split_default_slices(
        uint32_t rows, uint32_t input_width, uint32_t output_width,
        uint32_t sm_count, uint32_t resident_blocks) {
    if (!sm_count || !resident_blocks ||
        !vx_cuda_matmul_split_workspace_bytes(rows, input_width, output_width, 2u)) return 0u;
    uint64_t tiles = ((uint64_t)rows + 31u) / 32u *
        (((uint64_t)output_width + 31u) / 32u);
    uint64_t resident_capacity = (uint64_t)sm_count * resident_blocks;
    uint32_t selected = 0u;
    for (uint32_t split_k_slices = 2u; split_k_slices <= 16u; split_k_slices *= 2u) {
        if (!vx_cuda_matmul_split_workspace_bytes(rows, input_width, output_width, split_k_slices) ||
            input_width / (4u * split_k_slices) < 32u) continue;
        selected = split_k_slices;
        /* Equivalent to 2*resident_capacity <= tiles*split_k_slices without
         * overflowing for large device metadata. The right product is even. */
        if (resident_capacity <= tiles * split_k_slices / 2u) break;
    }
    return selected;
}
#endif
