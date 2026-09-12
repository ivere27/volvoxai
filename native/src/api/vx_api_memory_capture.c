/* Opt-in process memory capture for the protobuf API.
 *
 * `CreateRuntimeRequest.memory_capture` opts a Runtime and the objects it owns
 * into the memory-evidence contract. This is deliberately best-effort: a
 * platform without a sampler reports its envelopes as UNAVAILABLE rather than
 * failing an otherwise valid operation.
 *
 * Scope is exactly what the engine can prove. `vx_process_memory_sample_v1`
 * supplies instant RSS and process-lifetime peak RSS, so those two envelopes
 * carry byte values and every other requested envelope is present with
 * UNAVAILABLE and no value. Resource inventories need backend-specific
 * collectors that do not exist, so the single snapshot is always an empty
 * PARTIAL inventory — never a COMPLETE one, which would license a consumer to
 * treat it as a physical total.
 *
 * Policy is keyed by the native runtime lineage id that every report carries,
 * so a report from any descendant object finds its Runtime's policy without a
 * separate lease tree.
 */
#include "vx_api_convert.h"
#include "public_api_internal.h"

#include "vx_thread.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VX_API_CAPTURE_PROTOCOL "volvoxai-memory-capture/v1"
#define VX_API_EVIDENCE_FORMAT "volvoxai-memory-evidence/v1"
#define VX_API_MAX_PERIODIC_SNAPSHOTS 4096u

typedef struct VxApiCapturePolicy {
    struct VxApiCapturePolicy* next;
    uint64_t runtime_id;
    uint64_t capture_sequence;
    int include_resource_inventory;
    int include_domain_attestation;
    size_t envelope_count;
    VolvoxaiV1MemoryEnvelopeKind* envelopes;
} VxApiCapturePolicy;

static pthread_mutex_t vx_api_capture_mutex = PTHREAD_MUTEX_INITIALIZER;
static VxApiCapturePolicy* vx_api_capture_policies;

/* Caller holds the mutex. */
static VxApiCapturePolicy* vx_api_capture_find(uint64_t runtime_id) {
    VxApiCapturePolicy* policy = vx_api_capture_policies;
    while (policy) {
        if (policy->runtime_id == runtime_id) return policy;
        policy = policy->next;
    }
    return NULL;
}

const char* vx_api_capture_validate(const VolvoxaiV1MemoryCaptureOptions* options) {
    size_t index;
    size_t other;

    if (!options) return NULL;
    if (options->field_protocol.len != strlen(VX_API_CAPTURE_PROTOCOL) ||
        memcmp(options->field_protocol.data, VX_API_CAPTURE_PROTOCOL,
               options->field_protocol.len) != 0) {
        return "memory_capture.protocol must be exactly " VX_API_CAPTURE_PROTOCOL;
    }
    for (index = 0; index < options->field_requested_envelopes.len; index++) {
        VolvoxaiV1MemoryEnvelopeKind kind = options->field_requested_envelopes.data[index];
        if (kind == VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_UNSPECIFIED) {
            return "memory_capture.requested_envelopes must not contain UNSPECIFIED";
        }
        for (other = 0; other < index; other++) {
            if (options->field_requested_envelopes.data[other] == kind) {
                return "memory_capture.requested_envelopes must be unique";
            }
        }
    }
    if (!options->field_include_resource_inventory &&
        !options->field_include_domain_attestation &&
        options->field_requested_envelopes.len == 0u) {
        return "memory_capture must select an inventory, an attestation, or an envelope";
    }
    if (options->has_sampling_interval_nanoseconds != options->has_max_periodic_snapshots) {
        return "memory_capture sampling interval and snapshot limit must both be present";
    }
    if (options->has_max_periodic_snapshots) {
        if (options->field_sampling_interval_nanoseconds == 0u) {
            return "memory_capture.sampling_interval_nanoseconds must be nonzero";
        }
        if (options->field_max_periodic_snapshots == 0u ||
            options->field_max_periodic_snapshots > VX_API_MAX_PERIODIC_SNAPSHOTS) {
            return "memory_capture.max_periodic_snapshots must be in [1, 4096]";
        }
        /* Rejected rather than silently ignored: a caller that asked for a
         * sampled window must not receive a single AFTER snapshot and believe
         * it covers that window. */
        return "periodic memory capture is not implemented; omit the sampling interval";
    }
    return NULL;
}

int vx_api_capture_register(uint64_t runtime_id,
                            const VolvoxaiV1MemoryCaptureOptions* options) {
    VxApiCapturePolicy* policy;
    size_t index;

    if (!options || runtime_id == 0u) return 1;
    policy = (VxApiCapturePolicy*)calloc(1u, sizeof(*policy));
    if (!policy) return 0;
    policy->runtime_id = runtime_id;
    policy->include_resource_inventory = options->field_include_resource_inventory;
    policy->include_domain_attestation = options->field_include_domain_attestation;
    policy->envelope_count = options->field_requested_envelopes.len;
    if (policy->envelope_count) {
        policy->envelopes = (VolvoxaiV1MemoryEnvelopeKind*)calloc(
            policy->envelope_count, sizeof(*policy->envelopes));
        if (!policy->envelopes) {
            free(policy);
            return 0;
        }
        for (index = 0; index < policy->envelope_count; index++) {
            policy->envelopes[index] = options->field_requested_envelopes.data[index];
        }
    }
    pthread_mutex_lock(&vx_api_capture_mutex);
    policy->next = vx_api_capture_policies;
    vx_api_capture_policies = policy;
    pthread_mutex_unlock(&vx_api_capture_mutex);
    return 1;
}

void vx_api_capture_forget(uint64_t runtime_id) {
    VxApiCapturePolicy** link;

    pthread_mutex_lock(&vx_api_capture_mutex);
    link = &vx_api_capture_policies;
    while (*link) {
        VxApiCapturePolicy* policy = *link;
        if (policy->runtime_id == runtime_id) {
            *link = policy->next;
            free(policy->envelopes);
            free(policy);
            break;
        }
        link = &policy->next;
    }
    pthread_mutex_unlock(&vx_api_capture_mutex);
}

/* Which sampler and coverage each envelope kind would come from, and whether
 * this platform actually supplied a value for it. */
static void vx_api_envelope_shape(VolvoxaiV1MemoryEnvelopeKind kind,
                                  const VxProcessMemorySampleV1* sample,
                                  int sampled,
                                  int* available,
                                  uint64_t* bytes,
                                  VolvoxaiV1MemoryEvidenceSource* source,
                                  VolvoxaiV1MemoryTemporalCoverage* coverage,
                                  const char** sampler) {
    *available = 0;
    *bytes = 0u;
    *source = VOLVOXAI_V1_MEMORY_EVIDENCE_SOURCE_OS_SAMPLER;
    *coverage = VOLVOXAI_V1_MEMORY_TEMPORAL_COVERAGE_INSTANT;
    *sampler = "vx_process_memory_sample_v1";

    switch (kind) {
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_RSS:
            *available = sampled &&
                (sample->available_mask & VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS) != 0u;
            *bytes = sample->rss_bytes;
            return;
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_PEAK_RSS:
            *available = sampled &&
                (sample->available_mask & VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS) != 0u;
            *bytes = sample->peak_rss_bytes;
            *coverage =
                VOLVOXAI_V1_MEMORY_TEMPORAL_COVERAGE_PROCESS_LIFETIME;
            return;
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_MANAGED_HEAP_USED:
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_EXTERNAL_BYTES:
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_PROCESS_ARRAY_BUFFER_BYTES:
            *source =
                VOLVOXAI_V1_MEMORY_EVIDENCE_SOURCE_RUNTIME_COUNTER;
            *sampler = "native-runtime-memory-counters/v1";
            return;
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_DEVICE_PROCESS_USED:
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_DEVICE_TOTAL_USED:
        case VOLVOXAI_V1_MEMORY_ENVELOPE_KIND_DEVICE_TOTAL_CAPACITY:
            *source =
                VOLVOXAI_V1_MEMORY_EVIDENCE_SOURCE_DRIVER_SAMPLER;
            *sampler = "native-driver-memory-sampler/v1";
            return;
        default:
            /* PSS and private bytes have no portable sampler here. */
            return;
    }
}

static int vx_api_set_bytes(const SynurangLiteAllocator* allocator,
                            SynurangLiteBytes* field,
                            const char* text) {
    return synurang_lite_bytes_assign(allocator, field, text, strlen(text)) ==
           SYNURANG_LITE_OK;
}

int vx_api_capture_attach(VolvoxaiV1OperationReport* report, const VxReport* native) {
    const SynurangLiteAllocator* allocator;
    VxProcessMemorySampleV1 sample = VX_PROCESS_MEMORY_SAMPLE_V1_INIT;
    VxApiCapturePolicy* policy;
    VolvoxaiV1MemoryEvidence* evidence;
    VolvoxaiV1MemorySnapshot* snapshot;
    VolvoxaiV1MemoryEnvelopeEvidence* envelopes = NULL;
    VolvoxaiV1MemoryOwnerRef* subject;
    uint64_t capture_sequence;
    uint64_t subject_id;
    size_t envelope_count;
    size_t index;
    char identity[96];
    int sampled;
    VolvoxaiV1MemoryOwnerKind subject_kind;

    if (!report || !native) return 1;

    pthread_mutex_lock(&vx_api_capture_mutex);
    policy = vx_api_capture_find(native->runtime_id);
    if (policy) {
        capture_sequence = ++policy->capture_sequence;
        envelope_count = policy->envelope_count;
    }
    pthread_mutex_unlock(&vx_api_capture_mutex);
    if (!policy) return 1;

    allocator = report->_allocator;
    sampled = vx_process_memory_sample_v1(&sample) != 0;

    evidence = (VolvoxaiV1MemoryEvidence*)allocator->allocate(allocator->context,
                                                              sizeof(*evidence));
    if (!evidence) return 0;
    volvoxai_v1_memory_evidence_init_with_allocator(evidence, allocator);
    report->field_memory_evidence = evidence;
    if (!vx_api_set_bytes(allocator, &evidence->field_format, VX_API_EVIDENCE_FORMAT)) return 0;
    snprintf(identity, sizeof(identity), "native-%llu-capture-%llu",
             (unsigned long long)native->runtime_id, (unsigned long long)capture_sequence);
    if (!vx_api_set_bytes(allocator, &evidence->field_capture_id, identity)) return 0;

    snapshot = (VolvoxaiV1MemorySnapshot*)allocator->allocate(allocator->context,
                                                              sizeof(*snapshot));
    if (!snapshot) return 0;
    volvoxai_v1_memory_snapshot_init_with_allocator(snapshot, allocator);
    evidence->field_snapshots.data = snapshot;
    evidence->field_snapshots.len = 1u;
    evidence->field_snapshots.cap = 1u;

    snapshot->field_sequence = 1u;
    snapshot->field_stage = (VolvoxaiV1OperationStage)native->stage;
    snapshot->field_point = VOLVOXAI_V1_MEMORY_SNAPSHOT_POINT_AFTER;
    /* Never COMPLETE: resource collectors do not exist, and a COMPLETE empty
     * inventory would assert an exact zero. */
    snapshot->field_resource_inventory =
        VOLVOXAI_V1_MEMORY_INVENTORY_KIND_PARTIAL;
    snprintf(identity, sizeof(identity), "native-%llu-phase-%llu",
             (unsigned long long)native->runtime_id, (unsigned long long)capture_sequence);
    if (!vx_api_set_bytes(allocator, &snapshot->field_phase_occurrence_id, identity)) return 0;
    if (native->backend[0] &&
        !vx_api_set_bytes(allocator, &snapshot->field_backend, native->backend)) return 0;
    if (native->device[0] &&
        !vx_api_set_bytes(allocator, &snapshot->field_device, native->device)) return 0;

    if (sampled && (sample.available_mask & VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME)) {
        VolvoxaiV1MemoryMonotonicTime* clock =
            (VolvoxaiV1MemoryMonotonicTime*)allocator->allocate(allocator->context,
                                                                sizeof(*clock));
        if (!clock) return 0;
        volvoxai_v1_memory_monotonic_time_init_with_allocator(clock, allocator);
        clock->field_nanoseconds = sample.monotonic_nanoseconds;
        snapshot->field_monotonic_time = clock;
    }

    subject = (VolvoxaiV1MemoryOwnerRef*)allocator->allocate(allocator->context,
                                                             sizeof(*subject));
    if (!subject) return 0;
    volvoxai_v1_memory_owner_ref_init_with_allocator(subject, allocator);
    /* Use the deepest lifecycle owner affected by the operation. These are
     * native generation identities, which MemoryOwnerRef explicitly permits;
     * public handle lineage is projected independently after conversion. */
    if (native->context_id != 0u) {
        subject_kind = VOLVOXAI_V1_MEMORY_OWNER_KIND_EXECUTION_CONTEXT;
        subject_id = native->context_id;
    } else if (native->compiled_model_id != 0u) {
        subject_kind = VOLVOXAI_V1_MEMORY_OWNER_KIND_COMPILED_MODEL;
        subject_id = native->compiled_model_id;
    } else if (native->model_id != 0u) {
        subject_kind = VOLVOXAI_V1_MEMORY_OWNER_KIND_MODEL;
        subject_id = native->model_id;
    } else {
        subject_kind = VOLVOXAI_V1_MEMORY_OWNER_KIND_RUNTIME;
        subject_id = native->runtime_id;
    }
    subject->field_kind = subject_kind;
    snprintf(identity, sizeof(identity), "%llu", (unsigned long long)subject_id);
    if (!vx_api_set_bytes(allocator, &subject->field_owner_id, identity)) return 0;
    snapshot->field_subject = subject;

    if (envelope_count == 0u) return 1;
    envelopes = (VolvoxaiV1MemoryEnvelopeEvidence*)allocator->allocate(
        allocator->context, sizeof(*envelopes) * envelope_count);
    if (!envelopes) return 0;
    snapshot->field_envelopes.data = envelopes;
    snapshot->field_envelopes.len = envelope_count;
    snapshot->field_envelopes.cap = envelope_count;

    for (index = 0; index < envelope_count; index++) {
        VolvoxaiV1MemoryEnvelopeEvidence* entry = &envelopes[index];
        VolvoxaiV1MemoryEvidenceSource source;
        VolvoxaiV1MemoryTemporalCoverage coverage;
        const char* sampler;
        uint64_t bytes;
        int available;

        volvoxai_v1_memory_envelope_evidence_init_with_allocator(entry, allocator);
        pthread_mutex_lock(&vx_api_capture_mutex);
        entry->field_kind = policy->envelopes[index];
        pthread_mutex_unlock(&vx_api_capture_mutex);

        vx_api_envelope_shape(entry->field_kind, &sample, sampled, &available, &bytes,
                              &source, &coverage, &sampler);
        entry->field_source = source;
        entry->field_temporal_coverage = coverage;
        if (!vx_api_set_bytes(allocator, &entry->field_sampler, sampler)) return 0;
        if (available) {
            VolvoxaiV1MemoryByteSize* size = (VolvoxaiV1MemoryByteSize*)allocator->allocate(
                allocator->context, sizeof(*size));
            if (!size) return 0;
            volvoxai_v1_memory_byte_size_init_with_allocator(size, allocator);
            size->field_bytes = bytes;
            entry->field_bytes = size;
            entry->field_value_relation =
                VOLVOXAI_V1_MEMORY_VALUE_RELATION_EXACT;
        } else {
            /* UNAVAILABLE carries no MemoryByteSize, which is how a consumer
             * tells "this platform has no sampler" from "the value is zero". */
            entry->field_value_relation =
                VOLVOXAI_V1_MEMORY_VALUE_RELATION_UNAVAILABLE;
        }
    }
    return 1;
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

int vx_api_capture_attach_compiled_domain(
    VolvoxaiV1OperationReport* report,
    const VxReport* native,
    const VxCompiledDomainAttestationView* domain) {
    const SynurangLiteAllocator* allocator;
    VolvoxaiV1MemorySnapshot* snapshot;
    VolvoxaiV1MemoryDomainAttestation* attestation;
    VxApiCapturePolicy* policy;
    int requested = 0;

    if (!report || !native || !domain || !domain->graph_fingerprint ||
        !domain->graph_fingerprint[0] ||
        !domain->shape_domain_proof_identity ||
        !domain->shape_domain_proof_identity[0]) {
        return 0;
    }
    pthread_mutex_lock(&vx_api_capture_mutex);
    policy = vx_api_capture_find(native->runtime_id);
    if (policy) requested = policy->include_domain_attestation;
    pthread_mutex_unlock(&vx_api_capture_mutex);
    if (!requested) return 1;
    if (!report->field_memory_evidence ||
        report->field_memory_evidence->field_snapshots.len == 0u) {
        return 0;
    }

    allocator = report->_allocator;
    snapshot = &report->field_memory_evidence->field_snapshots.data[0];
    attestation = (VolvoxaiV1MemoryDomainAttestation*)allocator->allocate(
        allocator->context, sizeof(*attestation));
    if (!attestation) return 0;
    volvoxai_v1_memory_domain_attestation_init_with_allocator(
        attestation, allocator);
    snapshot->field_domain_attestation = attestation;
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
