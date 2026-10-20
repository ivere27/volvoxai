#ifndef VOLVOXAI_LLM_GPU_PLAN_H
#define VOLVOXAI_LLM_GPU_PLAN_H

#include <stddef.h>
#include <stdint.h>
#include "../kernels/llm_ops.h"

/* Physical plans shared by the native adapters and the WebGPU C owner.
 * A buffer names an existing tensor span, one reusable device workspace or
 * the four-byte zero binding. No plan copies or joins model weights. */
typedef enum {
    VX_LLM_GPU_DELTA_GATES, VX_LLM_GPU_DELTA_NET,
    VX_LLM_GPU_SPARSE_SELECT, VX_LLM_GPU_SPARSE_ATTENTION,
    VX_LLM_GPU_KV_COMPRESS, VX_LLM_GPU_HYPER_CONNECTION,
    VX_LLM_GPU_HYPER_HEAD, VX_LLM_GPU_HYPER_RESIDUAL,
    VX_LLM_GPU_ROTARY, VX_LLM_GPU_NGRAM,
    VX_LLM_GPU_QSA_SCORES, VX_LLM_GPU_QSA_MASK,
    VX_LLM_GPU_SIGNED_SQRT, VX_LLM_GPU_EMBEDDING_SHARD,
    VX_LLM_GPU_SPARSE_INDEX_SCORES, VX_LLM_GPU_SPARSE_INDEX_MASK,
    VX_LLM_GPU_PACKED_LINEAR,
    VX_LLM_GPU_MOE_ROUTER, VX_LLM_GPU_MOE_UP, VX_LLM_GPU_MOE_DOWN,
    VX_LLM_GPU_ROPE, VX_LLM_GPU_COPY, VX_LLM_GPU_SHADER_COUNT
} VxLlmGpuShader;

typedef struct {
    const void* host;
    uint64_t bytes;
    uint32_t workspace; /* 0=tensor, 1/2=workspace, 3=zero/discard, 4=constants. */
    uint32_t output;
} VxLlmGpuBuffer;

typedef struct {
    VxLlmGpuShader shader;
    VxLlmGpuBuffer buffers[8];
    uint32_t buffer_count;
    uint32_t uniform[32];
    uint32_t groups[3];
    float constants[128];
    uint32_t constants_bytes;
} VxLlmGpuDispatch;

typedef int (*VxLlmGpuEmit)(void* user, const VxLlmGpuDispatch* dispatch);

typedef struct {
    const float *input, *weight, *bias, *correction;
    const int32_t *tokens, *hash;
    float *indices, *routes;
    uint32_t rows, d_in, experts, top_k, normalize, scoring, groups;
    uint32_t selected_groups, group_top_k, vocabulary;
    float temperature, scale, epsilon;
} VxLlmGpuRouterArgs;

typedef struct {
    const float *input, *gate, *up, *down, *indices, *routes;
    float* output;
    uint32_t rows, d_in, hidden, d_out, experts, top_k, input_scale;
    float limit;
} VxLlmGpuGatedArgs;

typedef struct {
    const int32_t *tokens, *multipliers, *sizes, *offsets;
    int32_t* output;
    uint32_t batch, sequence, ngram, heads_per_ngram;
    int32_t eos, pad, vocabulary;
    uint64_t first_unit, unit_count;
} VxLlmGpuNgramArgs;

typedef struct {
    const float* input;
    const int32_t* positions;
    float* output;
    uint32_t batch, sequence, width, heads, rotary_width, position_offset;
    uint32_t interleaved, position_mode;
    float theta;
} VxLlmGpuRopeArgs;

int vx_llm_gpu_workspace_slots(size_t total, const size_t raw[2], size_t slots[2]);
const char* vx_llm_gpu_shader_name(VxLlmGpuShader shader);
uint32_t vx_llm_gpu_shader_bindings(VxLlmGpuShader shader);
int vx_llm_gpu_plan(const VxLLMOpArgs* args, uint64_t workspace[2],
                     VxLlmGpuEmit emit, void* user);
int vx_llm_gpu_router_plan(const VxLlmGpuRouterArgs* args,
                            VxLlmGpuEmit emit, void* user);
int vx_llm_gpu_gated_plan(const VxLlmGpuGatedArgs* args,
                           uint64_t workspace[2], VxLlmGpuEmit emit, void* user);
int vx_llm_gpu_ngram_plan(const VxLlmGpuNgramArgs* args,
                           VxLlmGpuEmit emit, void* user);
int vx_llm_gpu_rope_plan(const VxLlmGpuRopeArgs* args,
                          VxLlmGpuEmit emit, void* user);

#endif
