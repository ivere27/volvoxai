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

int vx_api_execution_result_fields(VolvoxaiV1ExecutionResultHandle* output,
    const VxResult* result, const VxReport* report) {
    output->field_execution_id = vx_result_execution_id(result);
    output->field_state = (int)vx_result_state(result);
    return vx_api_execution_metrics(output->_allocator, &output->field_metrics, report);
}

int vx_api_memory_snapshot(VolvoxaiV1MemorySnapshot* snapshot,
    int owner_kind, uint64_t owner_id, const VxProcessMemorySampleV1* sample,
    uint64_t start, uint64_t end) {
    const SynurangLiteAllocator* allocator = snapshot->_allocator;
    snapshot->field_observation_start_ns = start;
    snapshot->field_observation_end_ns = end;
    snapshot->field_resource_inventory = VOLVOXAI_V1_MEMORY_INVENTORY_KIND_PARTIAL;
    snapshot->field_subject = allocator->allocate(allocator->context, sizeof(*snapshot->field_subject));
    if (!snapshot->field_subject) return 0;
    volvoxai_v1_memory_owner_ref_init_with_allocator(snapshot->field_subject, allocator);
    snapshot->field_subject->field_kind = owner_kind;
    char identity[32];
    snprintf(identity, sizeof(identity), "%llu", (unsigned long long)owner_id);
    if (!vx_api_set_bytes(allocator, &snapshot->field_subject->field_owner_id, identity)) return 0;
    if (!sample) return 1;
    snapshot->field_envelopes.data = allocator->allocate(allocator->context,
        2 * sizeof(*snapshot->field_envelopes.data));
    if (!snapshot->field_envelopes.data) return 0;
    snapshot->field_envelopes.len = snapshot->field_envelopes.cap = 2;
    for (size_t i = 0; i < 2; i++)
        volvoxai_v1_memory_envelope_evidence_init_with_allocator(&snapshot->field_envelopes.data[i], allocator);
    for (size_t i = 0; i < 2; i++) {
        VolvoxaiV1MemoryEnvelopeEvidence* envelope = &snapshot->field_envelopes.data[i];
        int available = sample->available_mask & (i ? VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS : VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS);
        envelope->field_kind = i ? VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_PEAK_RSS : VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_RSS;
        envelope->field_source = VOLVOXAI_V1_MEMORY_EVIDENCE_SOURCE_OS_SAMPLER;
        envelope->field_temporal_coverage = i ? VOLVOXAI_V1_MEMORY_TEMPORAL_COVERAGE_PROCESS_LIFETIME : VOLVOXAI_V1_MEMORY_TEMPORAL_COVERAGE_INSTANT;
        envelope->field_value_relation = available ? VOLVOXAI_V1_MEMORY_VALUE_RELATION_EXACT : VOLVOXAI_V1_MEMORY_VALUE_RELATION_UNAVAILABLE;
        if (!vx_api_set_bytes(allocator, &envelope->field_sampler, "process")) return 0;
        if (available) {
            envelope->field_bytes = allocator->allocate(allocator->context, sizeof(*envelope->field_bytes));
            if (!envelope->field_bytes) return 0;
            volvoxai_v1_memory_byte_size_init_with_allocator(envelope->field_bytes, allocator);
            envelope->field_bytes->field_bytes = i ? sample->peak_rss_bytes : sample->rss_bytes;
        }
    }
    return 1;
}

int vx_api_memory_counter(VolvoxaiV1MemorySnapshot* snapshot, const char* name,
    uint64_t value, int metric) {
    const SynurangLiteAllocator* allocator = snapshot->_allocator;
    VolvoxaiV1MemoryCounter* counter = volvoxai_v1_memory_snapshot_add_counters(snapshot);
    if (!counter) return 0;
    if (!vx_api_set_bytes(allocator, &counter->field_name, name)) return 0;
    counter->field_owner = allocator->allocate(allocator->context, sizeof(*counter->field_owner));
    if (!counter->field_owner) return 0;
    volvoxai_v1_memory_owner_ref_init_with_allocator(counter->field_owner, allocator);
    counter->field_owner->field_kind = snapshot->field_subject->field_kind;
    if (synurang_lite_bytes_assign(allocator, &counter->field_owner->field_owner_id,
        snapshot->field_subject->field_owner_id.data, snapshot->field_subject->field_owner_id.len) != SYNURANG_LITE_OK) return 0;
    counter->field_measurement = allocator->allocate(allocator->context, sizeof(*counter->field_measurement));
    if (!counter->field_measurement) return 0;
    VolvoxaiV1MemoryMeasurement* measurement = counter->field_measurement;
    volvoxai_v1_memory_measurement_init_with_allocator(measurement, allocator);
    measurement->field_metric = metric;
    measurement->field_source = VOLVOXAI_V1_MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER;
    measurement->field_value_relation = VOLVOXAI_V1_MEMORY_VALUE_RELATION_EXACT;
    measurement->field_temporal_coverage = VOLVOXAI_V1_MEMORY_TEMPORAL_COVERAGE_INSTANT;
    measurement->field_bytes = allocator->allocate(allocator->context, sizeof(*measurement->field_bytes));
    if (!measurement->field_bytes) return 0;
    volvoxai_v1_memory_byte_size_init_with_allocator(measurement->field_bytes, allocator);
    measurement->field_bytes->field_bytes = value;
    return 1;
}
