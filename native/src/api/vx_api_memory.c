/* On-demand memory observations and independent compilation bounds. */
#include "vx_api_convert.h"
#include "public_api_internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int vx_api_set_bytes(const SynurangLiteAllocator* allocator,
    SynurangLiteBytes* field, const char* value) {
    return synurang_lite_bytes_assign(allocator, field, value, strlen(value)) == SYNURANG_LITE_OK;
}
static int vx_api_capture_bound(
    const SynurangLiteAllocator* allocator,
    VolvoxaiV1MemoryDomainAttestation* attestation,
    const char* budget_domain_id,
    VolvoxaiV1MemoryBoundKind kind,
    uint64_t maximum_bytes,
    int has_limit,
    uint64_t limit_bytes) {
    VolvoxaiV1MemoryBoundProof* bound =
        volvoxai_v1_memory_domain_attestation_add_bounds(attestation);
    VolvoxaiV1MemoryByteSize* maximum;
    if (!bound ||
        !vx_api_set_bytes(allocator, &bound->field_budget_domain_id,
                          budget_domain_id)) {
        return 0;
    }
    bound->field_kind = kind;
    maximum = (VolvoxaiV1MemoryByteSize*)allocator->allocate(
        allocator->context, sizeof(*maximum));
    if (!maximum) return 0;
    volvoxai_v1_memory_byte_size_init_with_allocator(maximum, allocator);
    maximum->field_bytes = maximum_bytes;
    bound->field_maximum_bytes = maximum;
    if (has_limit) {
        VolvoxaiV1MemoryByteSize* limit;
        if (maximum_bytes > limit_bytes) return 0;
        limit = (VolvoxaiV1MemoryByteSize*)allocator->allocate(
            allocator->context, sizeof(*limit));
        if (!limit) return 0;
        volvoxai_v1_memory_byte_size_init_with_allocator(limit, allocator);
        limit->field_bytes = limit_bytes;
        bound->field_limit_bytes = limit;
    }
    return 1;
}

int vx_api_compiled_memory_bounds(
    VolvoxaiV1CompiledModelHandle* response,
    const VxCompiledDomainAttestationView* domain) {
    const SynurangLiteAllocator* allocator;
    VolvoxaiV1MemoryDomainAttestation* attestation;

    if (!response || !domain) return 0;
    allocator = response->_allocator;
    attestation = (VolvoxaiV1MemoryDomainAttestation*)allocator->allocate(
        allocator->context, sizeof(*attestation));
    if (!attestation) return 0;
    volvoxai_v1_memory_domain_attestation_init_with_allocator(
        attestation, allocator);
    response->field_memory_bounds = attestation;
    if (!vx_api_set_bytes(allocator, &attestation->field_proof_protocol,
                          "canonical-symbolic-domain-proof/v1") ||
        !vx_api_set_bytes(allocator, &attestation->field_resource_protocol,
                          "bounded-resource-maxima/v1") ||
        !vx_api_set_bytes(allocator, &attestation->field_graph_fingerprint,
                          domain->graph_fingerprint) ||
        !vx_api_set_bytes(
            allocator, &attestation->field_shape_domain_proof_identity,
            domain->shape_domain_proof_identity) ||
        !vx_api_capture_bound(
            allocator, attestation, "maximum-tensor",
            VOLVOXAI_V1_MEMORY_BOUND_KIND_MAXIMUM_TENSOR,
            domain->maximum_tensor_bytes, 0, 0u) ||
        !vx_api_capture_bound(
            allocator, attestation, "provider-resident",
            VOLVOXAI_V1_MEMORY_BOUND_KIND_ORDINARY_RESIDENT,
            domain->maximum_resident_bytes,
            domain->has_resource_limit,
            domain->resource_limit_bytes)) {
        return 0;
    }
    return 1;
}

int vx_api_execution_metrics(const SynurangLiteAllocator* allocator,
    VolvoxaiV1ExecutionMetrics** output, const VxReport* report) {
    *output = allocator->allocate(allocator->context, sizeof(**output));
    if (!*output) return 0;
    volvoxai_v1_execution_metrics_init_with_allocator(*output, allocator);
    (*output)->field_host_time_ns = vx_api_duration_ns(report->execution_time_ms);
    (*output)->field_output_bytes = report->result_bytes;
    return 1;
}

int vx_api_compilation_metrics(VolvoxaiV1CompiledModelHandle* response, const VxReport* report) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    response->field_metrics = allocator->allocate(allocator->context, sizeof(*response->field_metrics));
    if (!response->field_metrics) return 0;
    volvoxai_v1_compilation_metrics_init_with_allocator(response->field_metrics, allocator);
    response->field_metrics->field_host_time_ns = vx_api_duration_ns(report->compile_time_ms);
    return 1;
}

/* AIP-158 page tokens are opaque to callers; the engine spells record offsets
 * in decimal. Records are immutable once published, so tokens stay valid. */
int vx_api_page_token_offset(const SynurangLiteBytes* token, uint64_t total, uint64_t* offset) {
    *offset = 0;
    if (!token->len) return 1;
    if (token->len > 20) return 0;
    for (size_t i = 0; i < token->len; i++) {
        unsigned digit = (unsigned)token->data[i] - '0';
        if (digit > 9 || (i == 0 && digit == 0 && token->len > 1)) return 0;
        if (*offset > (UINT64_MAX - digit) / 10) return 0;
        *offset = *offset * 10 + digit;
    }
    return *offset <= total;
}
int vx_api_next_page_token(const SynurangLiteAllocator* allocator, SynurangLiteBytes* token,
    uint64_t next, uint64_t total) {
    if (next >= total) return 1;
    char text[24]; snprintf(text, sizeof(text), "%llu", (unsigned long long)next);
    return vx_api_set_bytes(allocator, token, text);
}

int vx_api_execution_result_fields(VolvoxaiV1ExecutionResultHandle* output,
    const VxResult* result, const VxReport* report) {
    output->field_execution_id = vx_result_execution_id(result);
    output->field_state = (int)vx_result_state(result);
    return vx_api_execution_metrics(output->_allocator, &output->field_metrics, report);
}

/* All optional observations keep availability separate from an exact zero. */
#define VX_NEW(parent, field, type) do { \
    (parent)->field_##field = (parent)->_allocator->allocate((parent)->_allocator->context, sizeof(*(parent)->field_##field)); \
    if (!(parent)->field_##field) return 0; \
    volvoxai_v1_##type##_init_with_allocator((parent)->field_##field, (parent)->_allocator); \
} while (0)
static int vx_api_observed_bytes(VolvoxaiV1ObservedBytes* value,
    VxObservationStatus status, uint64_t bytes) {
    value->field_status = status;
    value->has_bytes = status == VX_OBSERVATION_AVAILABLE;
    if (value->has_bytes) value->field_bytes = bytes;
    return 1;
}
int vx_api_resource_snapshot(VolvoxaiV1ResourceSnapshot* output, const VxResourceSample* sample) {
    const SynurangLiteAllocator* allocator = output->_allocator;
    output->field_observation_start_ns = sample->start_ns;
    output->field_observation_end_ns = sample->end_ns;
    VX_NEW(output, wasm, wasm_memory_snapshot);
    VolvoxaiV1WasmMemorySnapshot* wasm = output->field_wasm;
    wasm->field_status = sample->wasm.status;
    if (sample->wasm.status == VX_OBSERVATION_AVAILABLE) {
        wasm->field_linear_bytes = sample->wasm.linear;
        wasm->field_allocated_bytes = sample->wasm.allocated;
        wasm->field_free_bytes = sample->wasm.free;
        wasm->field_allocator_metadata_bytes = sample->wasm.metadata;
        wasm->field_module_prefix_bytes = sample->wasm.prefix;
        wasm->field_page_slack_bytes = sample->wasm.slack;
        wasm->field_untracked_bytes = sample->wasm.untracked;
        wasm->field_largest_free_block_bytes = sample->wasm.largest_free;
        wasm->field_allocated_blocks = sample->wasm.allocated_blocks;
        wasm->field_free_blocks = sample->wasm.free_blocks;
    }
    VX_NEW(output, process, process_memory_snapshot);
    VolvoxaiV1ProcessMemorySnapshot* process = output->field_process;
    if (!vx_api_set_bytes(allocator, &process->field_source, "process OS sampler")) return 0;
    VX_NEW(process, resident, observed_bytes); VX_NEW(process, peak_resident, observed_bytes);
#if defined(__wasm__)
    VxObservationStatus missing = VX_OBSERVATION_UNSUPPORTED;
#else
    VxObservationStatus missing = VX_OBSERVATION_FAILED;
#endif
    vx_api_observed_bytes(process->field_resident,
        sample->process.available_mask & VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS ? VX_OBSERVATION_AVAILABLE : missing, sample->process.rss_bytes);
    vx_api_observed_bytes(process->field_peak_resident,
        sample->process.available_mask & VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS ? VX_OBSERVATION_AVAILABLE : missing, sample->process.peak_rss_bytes);
    VX_NEW(output, cpu, process_cpu_sample);
    output->field_cpu->field_status = sample->cpu.status;
    if (!vx_api_set_bytes(allocator, &output->field_cpu->field_source, sample->cpu.source)) return 0;
    if (sample->cpu.status == VX_OBSERVATION_AVAILABLE) {
        output->field_cpu->field_process_time_ns = sample->cpu.process_time_ns;
        output->field_cpu->has_online_processors = sample->cpu.online_processors != 0;
        output->field_cpu->field_online_processors = sample->cpu.online_processors;
    }
    VX_NEW(output, gpu, gpu_resource_samples);
    output->field_gpu->field_status = sample->gpu_status;
    output->field_gpu->field_truncated = sample->gpu_truncated;
    if (!vx_api_set_bytes(allocator, &output->field_gpu->field_source, sample->gpu_source)) return 0;
    for (uint32_t i = 0; i < sample->gpu_count; i++) {
        const VxGpuResourceSample* in = &sample->gpus[i];
        VolvoxaiV1GpuResourceSample* gpu = volvoxai_v1_gpu_resource_samples_add_devices(output->field_gpu);
        if (!gpu || !vx_api_set_bytes(allocator, &gpu->field_device_id, in->id) ||
            !vx_api_set_bytes(allocator, &gpu->field_name, in->name) ||
            !vx_api_set_bytes(allocator, &gpu->field_source, sample->gpu_source)) return 0;
        gpu->field_utilization_status = in->utilization;
        if (in->utilization == VX_OBSERVATION_AVAILABLE) {
            gpu->has_gpu_utilization = gpu->has_memory_utilization = 1;
            gpu->field_gpu_utilization = in->compute_active;
            gpu->field_memory_utilization = in->memory_active;
        }
        VX_NEW(gpu, memory_total, observed_bytes); VX_NEW(gpu, memory_used, observed_bytes);
        vx_api_observed_bytes(gpu->field_memory_total, in->memory, in->capacity_bytes);
        vx_api_observed_bytes(gpu->field_memory_used, in->memory, in->used_bytes);
    }
    return 1;
}
static int vx_api_memory_owner(VolvoxaiV1MemoryOwnerRef* owner, int kind, uint64_t id) {
    char text[32]; snprintf(text, sizeof(text), "%llu", (unsigned long long)id);
    owner->field_kind = kind;
    return vx_api_set_bytes(owner->_allocator, &owner->field_owner_id, text);
}
static int vx_api_memory_allocation(VolvoxaiV1MemoryAllocation* out,
    const VxMemoryAllocation* in, uint64_t id) {
    out->field_allocation_id = id; out->field_role = in->role;
    out->has_trace_allocation_id = in->trace_id != 0; out->field_trace_allocation_id = in->trace_id;
    out->field_space = in->space; out->field_capacity_bytes = in->capacity;
    VX_NEW(out, owner, memory_owner_ref);
    return vx_api_memory_owner(out->field_owner, in->owner_kind, in->owner_id) &&
        vx_api_set_bytes(out->_allocator, &out->field_allocator, vx_memory_allocator_name(in->allocator)) &&
        vx_api_set_bytes(out->_allocator, &out->field_name, in->name);
}
int vx_api_memory_inventory(VolvoxaiV1ResourceSnapshot* output, int kind,
    uint64_t id, const VxMemoryInventory* inventory, const VxMemoryView* view, int runtime) {
    VX_NEW(output, memory, memory_snapshot);
    VolvoxaiV1MemorySnapshot* memory = output->field_memory;
    VX_NEW(memory, subject, memory_owner_ref);
    if (!vx_api_memory_owner(memory->field_subject, kind, id)) return 0;
    memory->field_inventory = VX_MEMORY_INVENTORY_PARTIAL;
    memory->field_truncated = inventory->truncated;
    for (size_t i = 0; i < inventory->count; i++) {
        VolvoxaiV1MemoryAllocation* allocation = volvoxai_v1_memory_snapshot_add_allocations(memory);
        if (!allocation || !vx_api_memory_allocation(allocation, &inventory->items[i], i + 1)) return 0;
    }
    if (view) {
        if (runtime) {
            memory->has_unconsumed_result_bytes = 1;
            memory->field_unconsumed_result_bytes = view->active_output_bytes;
        } else {
            memory->has_retained_result_capacity_bytes = memory->has_idle_result_capacity_bytes = 1;
            memory->field_retained_result_capacity_bytes = view->result_capacity_bytes;
            memory->field_idle_result_capacity_bytes = view->idle_result_bytes;
        }
    }
    return 1;
}
int vx_api_step_cost(const SynurangLiteAllocator* allocator, VolvoxaiV1StepCost** slot,
                     const VxExecutionPlanStep* step) {
    if (!step->cost_status) return 1;
    *slot = allocator->allocate(allocator->context, sizeof(**slot));
    if (!*slot) return 0;
    volvoxai_v1_step_cost_init_with_allocator(*slot, allocator);
    VolvoxaiV1StepCost* cost = *slot;
    cost->field_status = step->cost_status;
    cost->field_multiply_accumulates = step->cost_macs;
    cost->field_elementwise_operations = step->cost_elementwise;
    cost->field_transcendental_operations = step->cost_transcendental;
    cost->field_input_bytes = step->cost_input_bytes;
    cost->field_output_bytes = step->cost_output_bytes;
    return 1;
}

int vx_api_execution_plan(VolvoxaiV1ExecutionPlan* output, const VxExecutionPlan* plan) {
    const SynurangLiteAllocator* allocator = output->_allocator;
    output->field_plan_id = plan->id; output->field_placement_complete = plan->placement_complete;
    output->field_source_mapping_complete = plan->source_mapping_complete;
    output->field_metadata_truncated = plan->metadata_truncated;
    if (!vx_api_set_bytes(allocator, &output->field_graph_fingerprint, plan->fingerprint) ||
        !vx_api_set_bytes(allocator, &output->field_shape_signature, plan->signature) ||
        !vx_api_set_bytes(allocator, &output->field_backend, plan->backend)) return 0;
    VX_NEW(output, lineage, lineage);
    output->field_lineage->field_runtime_id = plan->identity.runtime_id;
    output->field_lineage->field_model_id = plan->identity.model_id;
    output->field_lineage->field_compiled_model_id = plan->identity.compiled_model_id;
    output->field_lineage->field_context_id = plan->identity.context_id;
    output->field_lineage->field_execution_id = plan->identity.execution_id;
    output->field_lineage->field_graph_id = plan->identity.graph_id;
    output->field_lineage->field_graph_revision = plan->identity.graph_revision;
    for (uint32_t i = 0; i < plan->allocation_count; i++) {
        VolvoxaiV1MemoryAllocation* allocation = volvoxai_v1_execution_plan_add_allocations(output);
        if (!allocation || !vx_api_memory_allocation(allocation, &plan->allocations[i], i + 1)) return 0;
    }
    for (uint32_t i = 0; i < plan->tensor_count; i++) {
        const VxExecutionPlanTensor* in = &plan->tensors[i];
        VolvoxaiV1ExecutionPlanTensor* out = volvoxai_v1_execution_plan_add_tensors(output);
        if (!out || !vx_api_set_bytes(allocator, &out->field_name, in->name) ||
            !vx_api_set_bytes(allocator, &out->field_source_tensor_name, in->source_name)) return 0;
        out->field_tensor_id = i; out->field_dtype = in->dtype; out->field_logical_bytes = in->bytes;
        out->field_retained = in->retained;
        out->has_gradient_of = in->has_gradient_of != 0;
        out->field_gradient_of = in->gradient_of;
        if (in->allocation_id) {
            out->has_allocation_id = out->has_offset_bytes = 1;
            out->field_allocation_id = in->allocation_id; out->field_offset_bytes = in->offset;
        }
        if (in->first >= 0 && in->last >= in->first) {
            out->has_first_step = out->has_last_step = 1;
            out->field_first_step = (uint32_t)in->first; out->field_last_step = (uint32_t)in->last;
        }
        for (int d = 0; d < in->rank; d++) {
            int64_t* dim = volvoxai_v1_execution_plan_tensor_add_shape(out);
            if (!dim) return 0;
            *dim = in->shape[d];
        }
    }
    for (uint32_t i = 0; i < plan->step_count; i++) {
        const VxExecutionPlanStep* in = &plan->steps[i];
        VolvoxaiV1ExecutionPlanStep* out = volvoxai_v1_execution_plan_add_steps(output);
        if (!out || !vx_api_set_bytes(allocator, &out->field_operator_name, in->operator_name)) return 0;
        out->field_schedule_index = i; out->field_skipped = in->skipped; out->field_fused = in->fused;
        out->field_phase = in->phase;
        if (!vx_api_step_cost(allocator, &out->field_cost, in)) return 0;
        if (in->source_node_id[0]) {
            SynurangLiteBytes* source = volvoxai_v1_execution_plan_step_add_source_node_ids(out);
            if (!source || !vx_api_set_bytes(allocator, source, in->source_node_id)) return 0;
        }
        if (in->fused_source_node_id[0]) {
            SynurangLiteBytes* source = volvoxai_v1_execution_plan_step_add_source_node_ids(out);
            if (!source || !vx_api_set_bytes(allocator, source, in->fused_source_node_id)) return 0;
        }
        for (uint32_t r = 0; r < in->input_count; r++) {
            uint32_t* value = volvoxai_v1_execution_plan_step_add_inputs(out);
            if (!value) return 0;
            *value = plan->references[in->first_input + r];
        }
        for (uint32_t r = 0; r < in->output_count; r++) {
            uint32_t* value = volvoxai_v1_execution_plan_step_add_outputs(out);
            if (!value) return 0;
            *value = plan->references[in->first_output + r];
        }
    }
    return 1;
}
#undef VX_NEW
