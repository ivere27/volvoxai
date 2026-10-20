#include "../src/backends/cuda/attention_launch_plan.h"

#include <assert.h>
#include <stdio.h>
#include <string.h>

static int plan_empty(const VxCudaAttentionLaunchPlan* plan) {
    const VxCudaAttentionLaunchPlan empty = {0};
    return memcmp(plan, &empty, sizeof(empty)) == 0;
}

int main(void) {
    uint32_t grid[3] = {7u, 4u, 1u};
    uint32_t block[3] = {128u, 1u, 1u};
    VxCudaAttentionLaunchPlan plan;
    assert(VX_CUDA_ATTENTION_STATIC_SHARED_BYTES == 208u);
    assert(vx_cuda_attention_launch_plan(28u, grid, block, 128u, 208u, &plan));
    assert(plan.grid[0] == 7u && plan.grid[1] == 4u && plan.grid[2] == 1u);
    assert(plan.block[0] == 128u && plan.block[1] == 1u && plan.block[2] == 1u);
    assert(plan.static_shared_bytes == 208u);
    assert(!vx_cuda_attention_launch_plan(29u, grid, block, 128u, 208u, &plan));
    assert(plan_empty(&plan));
    assert(!vx_cuda_attention_launch_plan(1u, grid, block, 127u, 208u, &plan));
    assert(!vx_cuda_attention_launch_plan(1u, grid, block, 128u, 207u, &plan));
    assert(!vx_cuda_attention_launch_plan(1u, grid, block, 0u, 208u, &plan));
    assert(!vx_cuda_attention_launch_plan(1u, grid, block, 128u, 0u, &plan));
    assert(!vx_cuda_attention_launch_plan(0u, grid, block, 128u, 208u, &plan));
    assert(!vx_cuda_attention_launch_plan((uint64_t)UINT32_MAX + 1u,
        grid, block, 128u, 208u, &plan));
    assert(!vx_cuda_attention_launch_plan(1u, NULL, block, 128u, 208u, &plan));
    assert(!vx_cuda_attention_launch_plan(1u, grid, NULL, 128u, 208u, &plan));
    assert(!vx_cuda_attention_launch_plan(1u, grid, block, 128u, 208u, NULL));
    for (unsigned int axis = 0u; axis < 3u; ++axis) {
        uint32_t saved = grid[axis];
        grid[axis] = 0u;
        assert(!vx_cuda_attention_launch_plan(1u, grid, block, 128u, 208u, &plan));
        grid[axis] = saved;
        saved = block[axis];
        block[axis] = 0u;
        assert(!vx_cuda_attention_launch_plan(1u, grid, block, 128u, 208u, &plan));
        block[axis] = saved;
    }
    block[0] = 127u;
    assert(!vx_cuda_attention_launch_plan(1u, grid, block, 128u, 208u, &plan));
    block[0] = 128u;

    /* The old ceil(tasks/256) proof admitted this one-block grid, although
     * the cooperative kernel now needs one block for each of 256 tasks. */
    grid[0] = grid[1] = 1u;
    assert((256u + 255u) / 256u <= grid[0]);
    assert(!vx_cuda_attention_launch_plan(256u, grid, block, 256u, 208u, &plan));
    /* Conversely a 128-thread physical limit supports the new kernel. */
    assert(vx_cuda_attention_launch_plan(1u, grid, block, 128u, 208u, &plan));

    /* Every smaller ROW/range task count fits the same maximum-domain caps. */
    grid[0] = 7u;
    grid[1] = 4u;
    for (uint64_t tasks = 1u; tasks <= 28u; ++tasks) {
        assert(vx_cuda_attention_launch_plan(tasks, grid, block, 128u, 208u, &plan));
        assert((uint64_t)plan.grid[0] * plan.grid[1] >= tasks);
        assert(plan.grid[0] <= 7u && plan.grid[1] <= 4u);
    }

    grid[0] = UINT32_MAX - 1u;
    grid[1] = 2u;
    assert(vx_cuda_attention_launch_plan(UINT32_MAX, grid, block, 128u, 208u, &plan));
    assert(plan.grid[0] == UINT32_MAX - 1u && plan.grid[1] == 2u);
    {
        uint64_t last_index = (uint64_t)(plan.grid[1] - 1u) * plan.grid[0] +
            plan.grid[0] - 1u;
        assert(last_index > UINT32_MAX);
        assert(last_index >= (uint64_t)UINT32_MAX);
        /* A U32 conversion before the comparison would alias a valid task. */
        assert((uint32_t)last_index < UINT32_MAX);
    }
    grid[1] = 1u;
    assert(!vx_cuda_attention_launch_plan(UINT32_MAX, grid, block, 128u, 208u, &plan));
    grid[0] = UINT32_MAX;
    assert(vx_cuda_attention_launch_plan(UINT32_MAX, grid, block, 128u, 208u, &plan));
    assert(plan.grid[0] == UINT32_MAX && plan.grid[1] == 1u);
    /* Cross entries reserve the bounded score cache even for short rows or
     * rows beyond that cache. Check the actual aligned launch boundary. */
    assert(VX_CUDA_CROSS_ATTENTION_CACHE_KEYS == 2048u);
    assert(VX_CUDA_CROSS_ATTENTION_STATIC_SHARED_BYTES == 8720u);
    grid[0] = 65535u;
    grid[1] = grid[2] = 65535u;
    const uint32_t key_counts[] = {1u, 48u, 49u, 2048u, 2049u};
    for (unsigned int i = 0u; i < sizeof key_counts / sizeof key_counts[0]; ++i) {
        assert(!vx_cuda_cross_attention_launch_plan(32u, 1u, key_counts[i], 64u,
            grid, block, 128u, 8719u, &plan));
        assert(plan_empty(&plan));
        assert(vx_cuda_cross_attention_launch_plan(32u, 1u, key_counts[i], 64u,
            grid, block, 128u, 8720u, &plan));
        assert(plan.static_shared_bytes == 8720u);
        assert(plan.grid[2] == (key_counts[i] <= 48u ? 2u : 1u));
    }
    /* Prefill uses a transpose plus a multi-row kernel; reject every limit
     * before the first stage. Capacity changes alter both row reuse and
     * shared storage, and short rows must stay on their original algorithm. */
    VxCudaAttentionPrefillPlan prefill;
    block[0] = 256u; block[1] = 8u;
    assert(vx_cuda_attention_prefill_plan(512u, 512u, 1u,
        grid, block, 256u, 13360u, &prefill));
    assert(prefill.rows == 4u && prefill.first_query == 48u && prefill.grid_y == 116u);
    assert(prefill.static_shared_bytes == 13360u && prefill.workspace_bytes == 524288u);
    assert(vx_cuda_attention_prefill_plan(1536u, 1536u, 1u,
        grid, block, 256u, 18976u, &prefill));
    assert(prefill.rows == 2u && prefill.grid_y == 744u);
    assert(prefill.static_shared_bytes == 18976u && prefill.workspace_bytes == 1572864u);
    assert(vx_cuda_attention_prefill_plan(2048u, 2048u, 0u,
        grid, block, 256u, 18976u, &prefill));
    assert(prefill.first_query == 0u && prefill.grid_y == 1024u &&
        prefill.workspace_bytes == 2u * 1024u * 1024u);
    assert(vx_cuda_attention_prefill_plan(129u, 513u, 1u,
        grid, block, 256u, 18976u, &prefill));
    assert(prefill.rows == 4u && prefill.grid_y == 21u);
    assert(prefill.static_shared_bytes == 13360u && prefill.workspace_bytes == 513u * 1024u);
    assert(vx_cuda_attention_prefill_plan(512u, 2048u, 1u,
        grid, block, 256u, 13360u, &prefill));
    assert(prefill.rows == 4u && prefill.grid_y == 116u &&
        prefill.workspace_bytes == 2u * 1024u * 1024u);
    assert(vx_cuda_attention_prefill_plan(513u, 2048u, 1u,
        grid, block, 256u, 18976u, &prefill));
    assert(prefill.rows == 2u && prefill.grid_y == 233u);
    assert(vx_cuda_attention_prefill_plan(512u, 2048u, 0u,
        grid, block, 256u, 18976u, &prefill));
    assert(prefill.rows == 2u && prefill.grid_y == 256u);
    const uint32_t admitted_queries[] = {128u, 129u, 511u, 512u, 513u, 1536u, 2048u};
    const uint32_t admitted_keys[] = {49u, 511u, 512u, 513u, 1567u, 2048u};
    for (unsigned int q = 0; q < sizeof admitted_queries / sizeof admitted_queries[0]; ++q)
        for (unsigned int k = 0; k < sizeof admitted_keys / sizeof admitted_keys[0]; ++k)
            for (uint32_t causal = 0; causal < 2u; ++causal) {
                assert(vx_cuda_attention_prefill_plan(admitted_queries[q], admitted_keys[k], causal,
                    grid, block, 256u, 18976u, &prefill));
                assert(prefill.first_query + prefill.grid_y * prefill.rows >= admitted_queries[q]);
                assert(prefill.first_query + (prefill.grid_y - 1u) * prefill.rows < admitted_queries[q]);
                uint32_t active_keys = causal && admitted_queries[q] < admitted_keys[k] ?
                    admitted_queries[q] : admitted_keys[k];
                uint32_t capacity = prefill.rows == 4u ? 512u : 2048u;
                assert(prefill.rows == (active_keys <= 512u ? 4u : 2u));
                assert(prefill.workspace_bytes == admitted_keys[k] * 1024u);
                for (uint32_t row = prefill.first_query; row < admitted_queries[q]; ++row) {
                    uint32_t accessed = causal && row + 1u < admitted_keys[k] ?
                        row + 1u : admitted_keys[k];
                    assert(accessed <= capacity);
                }
            }
    assert(!vx_cuda_attention_prefill_plan(127u, 512u, 1u, grid, block, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(2049u, 512u, 1u, grid, block, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 48u, 0u, grid, block, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 2049u, 1u, grid, block, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 2u, grid, block, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 255u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 256u, 13359u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 2048u, 1u, grid, block, 256u, 13359u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(513u, 2048u, 1u, grid, block, 256u, 18975u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 2048u, 0u, grid, block, 256u, 18975u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, NULL, block, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, NULL, 256u, 18976u, &prefill));
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 256u, 18976u, NULL));
    block[0] = 255u;
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 256u, 18976u, &prefill));
    block[0] = 256u; block[1] = 7u;
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 256u, 18976u, &prefill));
    block[1] = 8u;
    grid[0] = 1535u;
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 256u, 18976u, &prefill));
    grid[0] = 65535u; grid[1] = 115u;
    assert(!vx_cuda_attention_prefill_plan(512u, 512u, 1u, grid, block, 256u, 18976u, &prefill));
    const VxCudaAttentionPrefillPlan empty_prefill = {0};
    assert(memcmp(&prefill, &empty_prefill, sizeof prefill) == 0);
    puts("PASS CUDA attention physical launch plan");
    return 0;
}
