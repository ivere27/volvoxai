/* Shared, immutable execution layout for profiling and node debugging. */
#ifndef VOLVOXAI_EXECUTION_PLAN_H
#define VOLVOXAI_EXECUTION_PLAN_H
#include "vx_lifecycle.h"
#include <stdint.h>
#include <stddef.h>
#include <string.h>
typedef struct {
    uint64_t runtime_id, model_id, compiled_model_id, context_id;
    uint64_t execution_id, graph_id, graph_revision;
} VxExecutionIdentity;
/* Allocator labels are private implementation details, not public switches. */
typedef enum {
    VX_MEMORY_HOST_ARENA, VX_MEMORY_HOST_TENSOR, VX_MEMORY_CUDA_GRAPH,
    VX_MEMORY_OPENGL_GRAPH, VX_MEMORY_VULKAN_GRAPH, VX_MEMORY_WEBGPU_BUFFER,
    VX_MEMORY_RESULT_HOST, VX_MEMORY_RESULT_CUDA, VX_MEMORY_RESULT_OPENGL,
    VX_MEMORY_RESULT_VULKAN, VX_MEMORY_RESULT_METAL, VX_MEMORY_HOST_WEIGHTS,
    VX_MEMORY_HOST_PACKED_WEIGHTS, VX_MEMORY_HOST_METADATA, VX_MEMORY_HOST_SCRATCH,
    VX_MEMORY_HOST_STAGING, VX_MEMORY_ALLOCATOR_COUNT
} VxMemoryAllocator;
typedef struct {
    uint64_t key, owner_id, capacity, trace_id;
    VxMemoryOwnerKind owner_kind;
    VxMemoryResourceRole role;
    VxMemorySpace space;
    VxMemoryAllocator allocator;
    char name[128];
} VxMemoryAllocation;
typedef struct {
    char name[128];
    /* The float tensor this value represents (PTQ provenance); empty when it
     * is the same name or unknown. */
    char source_name[128];
    VxDataType dtype;
    int rank, first, last, retained;
    /* Training debug plans: gradient tensors name the tensor they belong to. */
    int has_gradient_of;
    uint32_t gradient_of;
    int64_t shape[8];
    uint64_t bytes, allocation_id, offset;
} VxExecutionPlanTensor;
typedef struct {
    char operator_name[48], source_node_id[128], fused_source_node_id[128];
    uint32_t first_input, input_count, first_output, output_count;
    int skipped, fused;
    /* Training debug plans: which part of the step (VxTracePhase). */
    int phase;
    /* StepCost: counts for one call at these shapes, including a fused peer. */
    int cost_status;
    uint64_t cost_macs, cost_elementwise, cost_transcendental;
    uint64_t cost_input_bytes, cost_output_bytes;
} VxExecutionPlanStep;
typedef struct {
    size_t bytes;
    uint64_t id, storage_generation;
    VxExecutionIdentity identity;
    char fingerprint[64], backend[32];
    uint32_t step_count, tensor_count, reference_count, allocation_count;
    int placement_complete, source_mapping_complete, metadata_truncated;
    VxExecutionPlanStep* steps;
    VxExecutionPlanTensor* tensors;
    uint32_t* references;
    VxMemoryAllocation* allocations;
    char* signature;
} VxExecutionPlan;

/* A single allocation contains the entire fixed plan. All element strides
 * preserve pointer alignment; the trailing signature needs byte alignment. */
static inline size_t vx_execution_plan_size(uint32_t steps, uint32_t tensors,
    uint32_t references, uint32_t allocations, const char* signature) {
    uint64_t bytes = sizeof(VxExecutionPlan) + (uint64_t)steps * sizeof(VxExecutionPlanStep) +
        (uint64_t)tensors * sizeof(VxExecutionPlanTensor) +
        (uint64_t)allocations * sizeof(VxMemoryAllocation) +
        (uint64_t)references * sizeof(uint32_t) + strlen(signature) + 1;
    bytes = (bytes + 15u) & ~UINT64_C(15);
    return bytes <= SIZE_MAX ? (size_t)bytes : 0;
}
static inline void vx_execution_plan_init(VxExecutionPlan* plan, size_t bytes,
    uint32_t steps, uint32_t tensors, uint32_t references, uint32_t allocations,
    const char* signature) {
    memset(plan, 0, bytes);
    plan->bytes = bytes; plan->step_count = steps; plan->tensor_count = tensors;
    plan->reference_count = references; plan->allocation_count = allocations;
    plan->steps = (VxExecutionPlanStep*)(plan + 1);
    plan->tensors = (VxExecutionPlanTensor*)(plan->steps + steps);
    plan->allocations = (VxMemoryAllocation*)(plan->tensors + tensors);
    plan->references = (uint32_t*)(plan->allocations + allocations);
    plan->signature = (char*)(plan->references + references);
    strcpy(plan->signature, signature);
}
#endif
