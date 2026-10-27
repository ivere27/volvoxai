/* Shared F32 numerical implementations: deliberately no training dependency. */
#include "llm_math.h"
#include <stdlib.h>
#include <limits.h>
#include "packed_linear_ops.inc"

uint64_t vx_llm_work_items(const VxLLMOpArgs* a) {
    if (!a || a->batch <= 0 || a->sequence <= 0) return 0;
    switch (a->kind) {
        case VX_OP_PACKED_LINEAR: {
            uint64_t rows=(uint64_t)a->batch*(uint32_t)a->sequence;
            return a->value_dim>0 && rows<=UINT64_MAX/(uint32_t)a->value_dim
                ? rows*(uint32_t)a->value_dim : 0;
        }
        case VX_OP_SIGNED_SQRT:
        case VX_OP_EMBEDDING_SHARD: return a->y_elements[0];
        case VX_OP_GATED_DELTA_NET: return (uint64_t)a->batch * a->heads;
        case VX_OP_SPARSE_ATTENTION: return (uint64_t)a->batch * a->sequence * a->heads;
        case VX_OP_KV_COMPRESS:
        case VX_OP_SPARSE_INDEXER:
        case VX_OP_QSA_INDEXER:
        case VX_OP_HYPER_CONNECTION:
        case VX_OP_HYPER_HEAD: return (uint64_t)a->batch * a->sequence;
        case VX_OP_HYPER_RESIDUAL: return (uint64_t)a->batch * a->sequence * a->streams * a->key_dim;
        case VX_OP_ROTARY_EMBEDDING: return (uint64_t)a->batch * a->sequence * a->heads * a->key_dim;
        default: return 0;
    }
}
uint64_t vx_llm_scratch_elements(const VxLLMOpArgs* a) {
    if (!a) return 0;
    if (a->kind == VX_OP_PACKED_LINEAR)
        return a->key_dim > 0 && a->batch > 0 && a->sequence > 0
            ? vx_packed_linear_scratch_elements((uint64_t)a->key_dim,
                (uint64_t)a->batch * (uint32_t)a->sequence,
                a->groups ? (uint32_t)a->groups : 4u,(uint32_t)a->value_dim) : 0;
    if (a->kind == VX_OP_GATED_DELTA_NET) return (uint64_t)a->key_dim * a->value_dim;
    if (a->kind == VX_OP_SPARSE_INDEXER) return (uint64_t)a->keys;
    if (a->kind == VX_OP_SPARSE_ATTENTION) return (uint64_t)a->keys * 2;
    return 0;
}
int vx_llm_forward_f32(const VxLLMOpArgs* args) {
    uint64_t work = vx_llm_work_items(args), count = vx_llm_scratch_elements(args);
    uint64_t first = args ? args->first_unit : 0;
    uint64_t units = args && args->unit_count ? args->unit_count : work;
    if (!work || first > work || units > work - first || count > SIZE_MAX / sizeof(float)) return -1;
    float* scratch = count ? (float*)malloc((size_t)count * sizeof(float)) : NULL;
    if (count && !scratch) return -1;
    if(args->kind==VX_OP_PACKED_LINEAR) {
        int status=vx_packed_linear_forward(args,scratch);
        free(scratch);
        return status;
    }
    for (uint64_t unit = first; unit < first + units; ++unit) vx_llm_run_unit(args, unit, scratch);
    free(scratch);
    return 0;
}
