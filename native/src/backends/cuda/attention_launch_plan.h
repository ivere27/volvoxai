#ifndef VOLVOXAI_CUDA_ATTENTION_LAUNCH_PLAN_H
#define VOLVOXAI_CUDA_ATTENTION_LAUNCH_PLAN_H

#include <stdint.h>

/* Private physical launch contract for the cooperative F32 SDPA kernels.
 * Static shared storage is rounded to the driver's 16-byte allocation unit;
 * launches still request zero dynamic shared bytes. */
enum {
    VX_CUDA_ATTENTION_BLOCK_LANES = 128,
    VX_CUDA_ATTENTION_TILE_KEYS = 48,
    VX_CUDA_ATTENTION_STATISTICS = 3,
    VX_CUDA_ATTENTION_STATIC_SHARED_BYTES =
        ((VX_CUDA_ATTENTION_TILE_KEYS + VX_CUDA_ATTENTION_STATISTICS) * 4 + 15) & ~15
};

typedef struct VxCudaAttentionLaunchPlan {
    uint32_t grid[3];
    uint32_t block[3];
    uint32_t static_shared_bytes;
} VxCudaAttentionLaunchPlan;

/* Each task owns a block. U64 arithmetic permits padding past UINT32_MAX;
 * the kernel checks its U64 flattened block index before converting to U32.
 * Maximum-domain tasks also bound every smaller ROW/range dispatch. */
static inline int vx_cuda_attention_launch_plan(
        uint64_t tasks,
        const uint32_t maximum_grid[3],
        const uint32_t maximum_block[3],
        uint32_t maximum_threads_per_block,
        uint32_t maximum_shared_memory_bytes,
        VxCudaAttentionLaunchPlan* plan) {
    uint64_t grid_x;
    uint64_t grid_y;
    if (!plan) return 0;
    *plan = (VxCudaAttentionLaunchPlan){0};
    if (!tasks || tasks > UINT32_MAX || !maximum_grid || !maximum_block ||
        !maximum_grid[0] || !maximum_grid[1] || !maximum_grid[2] ||
        maximum_block[0] < VX_CUDA_ATTENTION_BLOCK_LANES ||
        !maximum_block[1] || !maximum_block[2] ||
        maximum_threads_per_block < VX_CUDA_ATTENTION_BLOCK_LANES ||
        maximum_shared_memory_bytes < VX_CUDA_ATTENTION_STATIC_SHARED_BYTES)
        return 0;
    grid_x = tasks < maximum_grid[0] ? tasks : maximum_grid[0];
    grid_y = tasks / grid_x + (tasks % grid_x != 0u);
    if (grid_y > maximum_grid[1]) return 0;
    plan->grid[0] = (uint32_t)grid_x;
    plan->grid[1] = (uint32_t)grid_y;
    plan->grid[2] = 1u;
    plan->block[0] = VX_CUDA_ATTENTION_BLOCK_LANES;
    plan->block[1] = 1u;
    plan->block[2] = 1u;
    plan->static_shared_bytes = VX_CUDA_ATTENTION_STATIC_SHARED_BYTES;
    return 1;
}

/* Cross output tiles use four warp partials for 32 contiguous channels.
 * Reserve the entry's static storage even when a longer row uses its fallback.
 * Check the full task/tile product before any allocation or submission. */
enum {
    VX_CUDA_CROSS_ATTENTION_OUTPUT_CHANNELS = 32,
    VX_CUDA_CROSS_ATTENTION_CACHE_KEYS = 2048,
    VX_CUDA_CROSS_ATTENTION_STATIC_SHARED_BYTES =
        ((VX_CUDA_CROSS_ATTENTION_CACHE_KEYS + VX_CUDA_ATTENTION_STATISTICS +
          VX_CUDA_ATTENTION_BLOCK_LANES) * 4 + 15) & ~15
};

static inline int vx_cuda_cross_attention_launch_plan(
        uint64_t tasks, uint32_t query_sequence, uint32_t key_sequence, uint32_t head_dim,
        const uint32_t maximum_grid[3],
        const uint32_t maximum_block[3],
        uint32_t maximum_threads_per_block,
        uint32_t maximum_shared_memory_bytes,
        VxCudaAttentionLaunchPlan* plan) {
    uint32_t tiles = 1u;
    if (!plan) return 0;
    *plan = (VxCudaAttentionLaunchPlan){0};
    if (!query_sequence || !key_sequence || !head_dim || !maximum_grid ||
        maximum_shared_memory_bytes < VX_CUDA_CROSS_ATTENTION_STATIC_SHARED_BYTES)
        return 0;
    if (query_sequence == 1u && key_sequence <= VX_CUDA_ATTENTION_TILE_KEYS && head_dim <= 128u)
        tiles = head_dim / VX_CUDA_CROSS_ATTENTION_OUTPUT_CHANNELS +
            (head_dim % VX_CUDA_CROSS_ATTENTION_OUTPUT_CHANNELS != 0u);
    if (tiles > maximum_grid[2] || tasks > UINT32_MAX / tiles ||
        !vx_cuda_attention_launch_plan(tasks, maximum_grid, maximum_block,
            maximum_threads_per_block, maximum_shared_memory_bytes, plan))
        return 0;
    plan->grid[2] = tiles;
    plan->static_shared_bytes = VX_CUDA_CROSS_ATTENTION_STATIC_SHARED_BYTES;
    return 1;
}

typedef struct VxCudaAttentionPrefillPlan {
    uint32_t rows;
    uint32_t first_query;
    uint32_t grid_y;
    uint32_t static_shared_bytes;
    uint32_t workspace_bytes;
} VxCudaAttentionPrefillPlan;

/* The transposed-K path preserves the cached arithmetic only for K>48.
 * Short causal queries retain their separate dot/reduction path. Both the
 * transpose and compute geometry must fit before either launch is submitted. */
static inline int vx_cuda_attention_prefill_plan(
        uint32_t queries, uint32_t keys, uint32_t causal,
        const uint32_t maximum_grid[3], const uint32_t maximum_block[3],
        uint32_t maximum_threads_per_block, uint32_t maximum_shared_memory_bytes,
        VxCudaAttentionPrefillPlan* plan) {
    if (!plan) return 0;
    *plan = (VxCudaAttentionPrefillPlan){0};
    if (queries < 128u || queries > 2048u || keys < 49u || keys > 2048u ||
        causal > 1u || !maximum_grid || !maximum_block ||
        maximum_block[0] < 256u || maximum_block[1] < 8u || !maximum_block[2] ||
        maximum_threads_per_block < 256u || !maximum_grid[2]) return 0;
    /* Causal rows only address scores through min(queries, keys). The
     * physical K extent still determines transpose storage and input strides. */
    uint32_t active_keys = causal && queries < keys ? queries : keys;
    uint32_t rows = active_keys <= 512u ? 4u : 2u;
    uint32_t capacity = active_keys <= 512u ? 512u : 2048u;
    uint32_t first = causal ? 48u : 0u;
    uint32_t grid_y = (queries - first + rows - 1u) / rows;
    uint32_t shared_bytes = (rows * (capacity + 3u + 256u + 64u) * 4u + 15u) & ~15u;
    if (maximum_grid[0] < 32u || maximum_grid[0] < (keys + 31u) / 32u ||
        maximum_grid[0] < first * 32u || maximum_grid[1] < 8u ||
        maximum_grid[1] < grid_y || maximum_shared_memory_bytes < shared_bytes)
        return 0;
    plan->rows = rows;
    plan->first_query = first;
    plan->grid_y = grid_y;
    plan->static_shared_bytes = shared_bytes;
    plan->workspace_bytes = keys * 256u * 4u;
    return 1;
}

#endif
