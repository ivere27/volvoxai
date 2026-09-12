#include "vx_api_convert.h"

#include <stdint.h>
#include <stddef.h>
#include <stdlib.h>
#include <string.h>

/* ---------------------------------------------------------------------------
 * Call-scoped scratch
 * ------------------------------------------------------------------------ */

struct VxApiScratchBlock {
    VxApiScratchBlock* next;
    /* Storage follows the header; max_align_t keeps it usable for any type. */
    max_align_t payload;
};

void* vx_api_scratch_alloc(VxApiScratch* scratch, size_t size) {
    VxApiScratchBlock* block;
    size_t header = offsetof(VxApiScratchBlock, payload);

    if (!scratch || size == 0u) return NULL;
    if (size > (size_t)-1 - header) {
        if (scratch) scratch->failed = 1;
        return NULL;
    }
    block = (VxApiScratchBlock*)calloc(1u, header + size);
    if (!block) {
        scratch->failed = 1;
        return NULL;
    }
    block->next = scratch->head;
    scratch->head = block;
    return (char*)block + header;
}

const char* vx_api_scratch_cstr(VxApiScratch* scratch,
                                const SynurangLiteBytes* bytes) {
    char* copy;
    if (!scratch || !bytes || bytes->len == 0u) return NULL;
    copy = (char*)vx_api_scratch_alloc(scratch, bytes->len + 1u);
    if (!copy) return NULL;
    memcpy(copy, bytes->data, bytes->len);
    copy[bytes->len] = '\0';
    return copy;
}

const char* const* vx_api_scratch_cstr_array(VxApiScratch* scratch,
                                             const SynurangLiteBytes* items,
                                             size_t stride,
                                             size_t count) {
    const char** array;
    size_t index;

    if (!scratch || count == 0u) return NULL;
    array = (const char**)vx_api_scratch_alloc(scratch, sizeof(*array) * count);
    if (!array) return NULL;
    for (index = 0; index < count; index++) {
        const SynurangLiteBytes* entry =
            (const SynurangLiteBytes*)((const char*)items + stride * index);
        array[index] = vx_api_scratch_cstr(scratch, entry);
    }
    return array;
}

int vx_api_scratch_failed(const VxApiScratch* scratch) {
    return !scratch || scratch->failed;
}

void vx_api_scratch_release(VxApiScratch* scratch) {
    VxApiScratchBlock* block;
    if (!scratch) return;
    block = scratch->head;
    while (block) {
        VxApiScratchBlock* next = block->next;
        free(block);
        block = next;
    }
    scratch->head = NULL;
    scratch->failed = 0;
}

/* ---------------------------------------------------------------------------
 * Small helpers over the generated lite messages
 * ------------------------------------------------------------------------ */

static int vx_api_set_text(const SynurangLiteAllocator* allocator,
                           SynurangLiteBytes* field,
                           const char* text) {
    if (!text || !*text) return 1;
    return synurang_lite_bytes_assign(allocator, field, text, strlen(text)) ==
           SYNURANG_LITE_OK;
}

#define VX_API_NEW_CHILD(allocator, slot, type, init_fn)                       \
    do {                                                                       \
        if (!*(slot)) {                                                        \
            *(slot) = (type*)(allocator)->allocate((allocator)->context,        \
                                                   sizeof(type));               \
            if (!*(slot)) return 0;                                            \
            init_fn(*(slot), (allocator));                                     \
        }                                                                      \
    } while (0)

/* ---------------------------------------------------------------------------
 * Packed "key=value;" evidence scanner
 *
 * The engine writes routing, fallback and decode evidence as a semicolon
 * separated list of key=value pairs (see public_api_report_diagnostics.inc).
 * These helpers read one field at a time without mutating the source buffer.
 * ------------------------------------------------------------------------ */

/* Points value/length at the value for key, or returns 0 when absent. */
static int vx_api_field(const char* packed,
                        const char* key,
                        const char** value,
                        size_t* length) {
    size_t key_length = strlen(key);
    const char* cursor = packed;

    if (!packed || !*packed) return 0;
    while (*cursor) {
        const char* end = strchr(cursor, ';');
        size_t span = end ? (size_t)(end - cursor) : strlen(cursor);
        if (span > key_length && cursor[key_length] == '=' &&
            strncmp(cursor, key, key_length) == 0) {
            *value = cursor + key_length + 1u;
            *length = span - key_length - 1u;
            return 1;
        }
        if (!end) break;
        cursor = end + 1;
    }
    return 0;
}

static int vx_api_field_u64(const char* packed, const char* key, uint64_t* out) {
    const char* value;
    size_t length;
    char buffer[64];
    if (!vx_api_field(packed, key, &value, &length)) return 0;
    if (length == 0u || length >= sizeof(buffer)) return 0;
    memcpy(buffer, value, length);
    buffer[length] = '\0';
    *out = strtoull(buffer, NULL, 10);
    return 1;
}

static int vx_api_field_i32(const char* packed, const char* key, int32_t* out) {
    const char* value;
    size_t length;
    char buffer[32];
    char* end;
    long parsed;
    if (!vx_api_field(packed, key, &value, &length)) return 0;
    if (length == 0u || length >= sizeof(buffer)) return 0;
    memcpy(buffer, value, length);
    buffer[length] = '\0';
    parsed = strtol(buffer, &end, 10);
    if (end != buffer + length || parsed < INT32_MIN || parsed > INT32_MAX)
        return 0;
    *out = (int32_t)parsed;
    return 1;
}

static int vx_api_field_hex64(const char* packed, const char* key, uint64_t* out) {
    const char* value;
    size_t length;
    char buffer[32];
    const char* colon;
    if (!vx_api_field(packed, key, &value, &length)) return 0;
    /* shape_signature packs "<hex>:<signature>"; the digest ends at the colon. */
    colon = (const char*)memchr(value, ':', length);
    if (colon) length = (size_t)(colon - value);
    if (length == 0u || length >= sizeof(buffer)) return 0;
    memcpy(buffer, value, length);
    buffer[length] = '\0';
    *out = strtoull(buffer, NULL, 16);
    return 1;
}

static int vx_api_field_double(const char* packed, const char* key, double* out) {
    const char* value;
    size_t length;
    char buffer[64];
    if (!vx_api_field(packed, key, &value, &length)) return 0;
    if (length == 0u || length >= sizeof(buffer)) return 0;
    memcpy(buffer, value, length);
    buffer[length] = '\0';
    *out = strtod(buffer, NULL);
    return 1;
}

static int vx_api_field_text(const SynurangLiteAllocator* allocator,
                             const char* packed,
                             const char* key,
                             SynurangLiteBytes* field) {
    const char* value;
    size_t length;
    if (!vx_api_field(packed, key, &value, &length)) return 1;
    if (length == 0u) return 1;
    return synurang_lite_bytes_assign(allocator, field, value, length) ==
           SYNURANG_LITE_OK;
}

static int vx_api_field_equals(const char* packed, const char* key, const char* expect) {
    const char* value;
    size_t length;
    if (!vx_api_field(packed, key, &value, &length)) return 0;
    return length == strlen(expect) && memcmp(value, expect, length) == 0;
}

/* ---------------------------------------------------------------------------
 * Typed evidence
 * ------------------------------------------------------------------------ */

/* "index=%d;op=%s;output=%s" */
static int vx_api_node_ref(const SynurangLiteAllocator* allocator,
                           VolvoxaiV1NodeRef** slot,
                           const char* packed) {
    uint64_t index = 0u;
    if (!packed || !*packed) return 1;
    VX_API_NEW_CHILD(allocator, slot, VolvoxaiV1NodeRef,
                     volvoxai_v1_node_ref_init_with_allocator);
    if (vx_api_field_u64(packed, "index", &index)) {
        (*slot)->field_index = (int32_t)index;
    }
    if (!vx_api_field_text(allocator, packed, "op", &(*slot)->field_op)) return 0;
    if (!vx_api_field_text(allocator, packed, "output", &(*slot)->field_output)) return 0;
    return 1;
}

/* "provider=builtin:<backend>;nodes=..;selected=..;fallback=..;missing=..;
 *  digest=<hex>" optionally followed by the dynamic-shape record. */
static int vx_api_route_evidence(const SynurangLiteAllocator* allocator,
                                 VolvoxaiV1OperationReport* report,
                                 const VxReport* native) {
    const char* packed = native->route_evidence;
    VolvoxaiV1RouteEvidence* route;
    const char* provider;
    size_t provider_length;
    uint64_t number = 0u;
    double seconds = 0.0;

    if (!packed || !*packed) return 1;
    VX_API_NEW_CHILD(allocator, &report->field_route, VolvoxaiV1RouteEvidence,
                     volvoxai_v1_route_evidence_init_with_allocator);
    route = report->field_route;
    route->field_attested = native->route_attested != 0;

    if (vx_api_field(packed, "provider", &provider, &provider_length)) {
        /* "builtin:<name>" marks an in-tree backend; anything else is the
         * provider name as the engine reported it. */
        static const char builtin[] = "builtin:";
        const size_t builtin_length = sizeof(builtin) - 1u;
        if (provider_length > builtin_length &&
            memcmp(provider, builtin, builtin_length) == 0) {
            route->field_builtin = 1;
            provider += builtin_length;
            provider_length -= builtin_length;
        }
        if (provider_length &&
            synurang_lite_bytes_assign(allocator, &route->field_provider, provider,
                                       provider_length) != SYNURANG_LITE_OK) {
            return 0;
        }
    }
    if (vx_api_field_u64(packed, "nodes", &number)) route->field_active_nodes = (uint32_t)number;
    if (vx_api_field_u64(packed, "selected", &number)) route->field_selected_nodes = (uint32_t)number;
    if (vx_api_field_u64(packed, "fallback", &number)) route->field_fallback_nodes = (uint32_t)number;
    if (vx_api_field_u64(packed, "missing", &number)) route->field_missing_nodes = (uint32_t)number;
    if (vx_api_field_hex64(packed, "digest", &number)) route->field_digest = number;

    /* The dynamic-shape record is appended only by built-in backends that
     * bound a symbolic signature for this execution. */
    if (vx_api_field(packed, "shape_signature", &provider, &provider_length)) {
        VolvoxaiV1ShapePlanEvidence* plan;
        const char* colon;
        VX_API_NEW_CHILD(allocator, &route->field_shape_plan,
                         VolvoxaiV1ShapePlanEvidence,
                         volvoxai_v1_shape_plan_evidence_init_with_allocator);
        plan = route->field_shape_plan;
        if (vx_api_field_hex64(packed, "shape_signature", &number)) {
            plan->field_signature_digest = number;
        }
        colon = (const char*)memchr(provider, ':', provider_length);
        if (colon) {
            size_t signature_length = provider_length - (size_t)(colon - provider) - 1u;
            if (signature_length &&
                synurang_lite_bytes_assign(allocator, &plan->field_signature, colon + 1,
                                           signature_length) != SYNURANG_LITE_OK) {
                return 0;
            }
        }
        plan->field_origin = vx_api_field_equals(packed, "shape_plan", "hit")
                                 ? VOLVOXAI_V1_SHAPE_PLAN_ORIGIN_CACHE_HIT
                                 : VOLVOXAI_V1_SHAPE_PLAN_ORIGIN_COLD;
        if (vx_api_field_double(packed, "shape_bind_ms", &seconds)) {
            plan->field_bind_time_ms = seconds;
        }
        if (vx_api_field_u64(packed, "logical_bytes", &number)) plan->field_logical_bytes = number;
        if (vx_api_field_u64(packed, "arena_required", &number)) plan->field_arena_required_bytes = number;
        if (vx_api_field_u64(packed, "arena_capacity", &number)) plan->field_arena_capacity_bytes = number;
        if (vx_api_field_u64(packed, "arena_high_water", &number)) plan->field_arena_high_water_bytes = number;
        if (vx_api_field_u64(packed, "arena_grows", &number)) plan->field_arena_grow_count = number;
        if (vx_api_field_u64(packed, "resource_generation", &number)) plan->field_resource_generation = number;
    }
    return 1;
}

/* "operator=none" | "operator=used;count=N;first=index:I,op:OP" */
static int vx_api_fallback_evidence(const SynurangLiteAllocator* allocator,
                                    VolvoxaiV1OperationReport* report,
                                    const VxReport* native) {
    const char* packed = native->fallback_evidence;
    VolvoxaiV1FallbackEvidence* fallback;
    uint64_t count = 0u;

    if (!packed || !*packed) return 1;
    VX_API_NEW_CHILD(allocator, &report->field_fallback, VolvoxaiV1FallbackEvidence,
                     volvoxai_v1_fallback_evidence_init_with_allocator);
    fallback = report->field_fallback;
    fallback->field_operator_fallback_used = native->operator_fallback_used != 0;
    if (vx_api_field_u64(packed, "count", &count)) fallback->field_count = (uint32_t)count;
    /* The first offending node is reported separately and more completely in
     * VxReport.offending_node, so reuse that rather than reparsing "first". */
    if (native->offending_node[0]) {
        if (!vx_api_node_ref(allocator, &fallback->field_first, native->offending_node)) return 0;
    }
    return 1;
}

/* "enabled=1;prefilled=<0|1|unknown>;mode=<name>;last=<...>[;position=<i32>]" */
static int vx_api_decode_state(const SynurangLiteAllocator* allocator,
                               VolvoxaiV1OperationReport* report,
                               const VxReport* native) {
    const char* packed = native->decode_state;
    VolvoxaiV1DecodeState* decode;
    int32_t last_position;

    if (!packed || !*packed) return 1;
    VX_API_NEW_CHILD(allocator, &report->field_decode, VolvoxaiV1DecodeState,
                     volvoxai_v1_decode_state_init_with_allocator);
    decode = report->field_decode;
    decode->field_enabled = vx_api_field_equals(packed, "enabled", "1");
    decode->field_prefilled = vx_api_field_equals(packed, "prefilled", "1");
    if (vx_api_field_equals(packed, "mode", "provider")) {
        decode->field_mode = VOLVOXAI_V1_DECODE_MODE_PROVIDER;
    } else if (vx_api_field_equals(packed, "mode", "row")) {
        decode->field_mode = VOLVOXAI_V1_DECODE_MODE_ROW;
    } else if (vx_api_field_equals(packed, "mode", "dependency")) {
        decode->field_mode = VOLVOXAI_V1_DECODE_MODE_DEPENDENCY;
    } else if (decode->field_enabled) {
        decode->field_mode = VOLVOXAI_V1_DECODE_MODE_UNSPECIFIED;
    } else {
        decode->field_mode = VOLVOXAI_V1_DECODE_MODE_DISABLED;
    }
    if (vx_api_field_i32(packed, "position", &last_position)) {
        decode->has_last_position = 1;
        decode->field_last_position = last_position;
    }
    return 1;
}

static VolvoxaiV1CandidateOutcome vx_api_candidate_outcome(
    const char* value,
    size_t length) {
#define VX_API_OUTCOME_IS(literal) \
    (length == sizeof(literal) - 1u && \
     memcmp(value, literal, sizeof(literal) - 1u) == 0)
    if (VX_API_OUTCOME_IS("selected"))
        return VOLVOXAI_V1_CANDIDATE_OUTCOME_SELECTED;
    if (VX_API_OUTCOME_IS("unavailable"))
        return VOLVOXAI_V1_CANDIDATE_OUTCOME_UNAVAILABLE;
    if (VX_API_OUTCOME_IS("domain-unsupported") ||
        VX_API_OUTCOME_IS("domain-unattested"))
        return VOLVOXAI_V1_CANDIDATE_OUTCOME_DOMAIN_UNSUPPORTED;
    if (VX_API_OUTCOME_IS("operator-fallback-forbidden") ||
        VX_API_OUTCOME_IS("unsupported") ||
        VX_API_OUTCOME_IS("route-unattested"))
        return VOLVOXAI_V1_CANDIDATE_OUTCOME_OPERATOR_UNSUPPORTED;
    if (VX_API_OUTCOME_IS("not-tried") ||
        VX_API_OUTCOME_IS("not-evaluated"))
        return VOLVOXAI_V1_CANDIDATE_OUTCOME_NOT_EVALUATED;
    if (VX_API_OUTCOME_IS("compile-failed") ||
        VX_API_OUTCOME_IS("graph-rejected") ||
        VX_API_OUTCOME_IS("out-of-memory") ||
        VX_API_OUTCOME_IS("rejected"))
        return VOLVOXAI_V1_CANDIDATE_OUTCOME_COMPILE_FAILED;
    return VOLVOXAI_V1_CANDIDATE_OUTCOME_UNSPECIFIED;
#undef VX_API_OUTCOME_IS
}

/* "<backend>:<status>:<outcome>;..." — one typed record per policy candidate. */
static int vx_api_compilation_evidence(const SynurangLiteAllocator* allocator,
                                       VolvoxaiV1OperationReport* report,
                                       const VxReport* native) {
    VolvoxaiV1CompilationEvidence* compilation;
    const char* packed = native->candidate_outcomes;

    VX_API_NEW_CHILD(allocator, &report->field_compilation, VolvoxaiV1CompilationEvidence,
                     volvoxai_v1_compilation_evidence_init_with_allocator);
    compilation = report->field_compilation;
    compilation->field_policy_mode = (VolvoxaiV1BackendPolicyMode)native->policy_mode;
    compilation->field_operator_fallback = (VolvoxaiV1OperatorFallback)native->operator_fallback;
    compilation->field_tier_fallback_used = native->tier_fallback_used != 0;

    if (packed && *packed) {
        const char* cursor = packed;
        size_t count = 1u;
        size_t index = 0u;
        VolvoxaiV1CompilationCandidate* candidates;
        for (const char* item = packed; *item; item++)
            if (*item == ';') count++;
        candidates = (VolvoxaiV1CompilationCandidate*)allocator->allocate(
            allocator->context, sizeof(*candidates) * count);
        if (!candidates) return 0;
        compilation->field_candidates.data = candidates;
        compilation->field_candidates.len = count;
        compilation->field_candidates.cap = count;
        for (size_t candidate_index = 0u; candidate_index < count;
             candidate_index++)
            volvoxai_v1_compilation_candidate_init_with_allocator(
                &candidates[candidate_index], allocator);

        while (*cursor && index < count) {
            const char* end = strchr(cursor, ';');
            const char* first = strchr(cursor, ':');
            const char* second;
            char status_text[32];
            char* status_end = NULL;
            long status;
            size_t span = end ? (size_t)(end - cursor) : strlen(cursor);
            size_t backend_length;
            size_t status_length;
            size_t outcome_length;
            VolvoxaiV1CompilationCandidate* candidate = &candidates[index];
            if (!first || (size_t)(first - cursor) >= span) return 0;
            second = (const char*)memchr(first + 1, ':',
                                         span - (size_t)(first + 1 - cursor));
            if (!second) return 0;
            backend_length = (size_t)(first - cursor);
            status_length = (size_t)(second - first - 1);
            outcome_length = span - (size_t)(second + 1 - cursor);
            if (!backend_length || !status_length ||
                status_length >= sizeof(status_text) || !outcome_length)
                return 0;

            if (synurang_lite_bytes_assign(allocator, &candidate->field_backend,
                                           cursor, backend_length) != SYNURANG_LITE_OK)
                return 0;
            memcpy(status_text, first + 1, status_length);
            status_text[status_length] = '\0';
            status = strtol(status_text, &status_end, 10);
            if (!status_end || *status_end != '\0' || status < INT32_MIN ||
                status > INT32_MAX)
                return 0;
            candidate->field_status = (VolvoxaiV1NativeStatus)status;
            candidate->field_outcome = vx_api_candidate_outcome(
                second + 1, outcome_length);
            if (synurang_lite_bytes_assign(allocator, &candidate->field_reason,
                                           second + 1, outcome_length) !=
                SYNURANG_LITE_OK)
                return 0;
            index++;
            if (!end) break;
            cursor = end + 1;
        }
    }
    return 1;
}

/* ---------------------------------------------------------------------------
 * Report
 * ------------------------------------------------------------------------ */

static int vx_api_input_issue(const SynurangLiteAllocator* allocator,
                               VolvoxaiV1InputValidationIssue** slot,
                               const VxInputValidationEvidence* evidence) {
    VolvoxaiV1InputValidationIssue* issue;
    if (evidence->code == VX_INPUT_UNSPECIFIED) return 1;
    VX_API_NEW_CHILD(allocator, slot, VolvoxaiV1InputValidationIssue,
                     volvoxai_v1_input_validation_issue_init_with_allocator);
    issue = *slot;
    issue->field_code = (VolvoxaiV1InputValidationCode)evidence->code;
    issue->has_input_index = evidence->has_input_index;
    issue->field_input_index = evidence->input_index;
    issue->has_axis = evidence->has_axis;
    issue->field_axis = evidence->axis;
    issue->has_expected_byte_size = evidence->has_expected_bytes;
    issue->field_expected_byte_size = evidence->expected_bytes;
    issue->has_required_extent = evidence->has_required_extent;
    issue->field_required_extent = evidence->required_extent;
    issue->field_names_truncated = evidence->names_truncated;
    if (!vx_api_set_text(allocator, &issue->field_input_name, evidence->name)) return 0;
    if (evidence->has_expected) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        VX_API_NEW_CHILD(allocator, &issue->field_expected, VolvoxaiV1TensorSpec,
                         volvoxai_v1_tensor_spec_init_with_allocator);
        spec.name = evidence->name;
        spec.dtype = evidence->expected_dtype;
        spec.rank = evidence->expected_rank < VX_MAX_TENSOR_RANK
                        ? evidence->expected_rank : VX_MAX_TENSOR_RANK;
        for (uint32_t axis = 0; axis < spec.rank; axis++) {
            spec.dimensions[axis].kind = evidence->kinds[axis];
            spec.dimensions[axis].symbol = evidence->symbols[axis];
            spec.dimensions[axis].min = evidence->minimums[axis];
            spec.dimensions[axis].max = evidence->maximums[axis];
            spec.dimensions[axis].multiple_of = evidence->multiples[axis];
        }
        if (!vx_api_tensor_spec_from_native(allocator, issue->field_expected, &spec)) return 0;
    }
    if (evidence->has_actual) {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        VX_API_NEW_CHILD(allocator, &issue->field_actual, VolvoxaiV1TensorInfo,
                         volvoxai_v1_tensor_info_init_with_allocator);
        info.name = evidence->name;
        info.dtype = evidence->actual_dtype;
        info.location = evidence->actual_location;
        info.byte_size = evidence->actual_bytes;
        info.rank = evidence->actual_rank < VX_MAX_TENSOR_RANK
                        ? evidence->actual_rank : VX_MAX_TENSOR_RANK;
        memcpy(info.shape, evidence->actual_shape, info.rank * sizeof(int64_t));
        issue->has_actual_rank = 1;
        issue->field_actual_rank = evidence->actual_rank;
        if (!vx_api_tensor_info_from_native(allocator, issue->field_actual, &info)) return 0;
    }
    return 1;
}

int vx_api_report_from_native(VolvoxaiV1OperationReport* report,
                              const VxReport* native) {
    const SynurangLiteAllocator* allocator;

    if (!report || !native || native->struct_size != sizeof(*native)) return 0;
    allocator = report->_allocator;

    report->field_status = (VolvoxaiV1NativeStatus)native->status;
    report->field_stage = (VolvoxaiV1OperationStage)native->stage;
    report->field_code = (VolvoxaiV1OperationCode)native->code;
    if (!vx_api_set_text(allocator, &report->field_message, native->message)) return 0;
    if (!vx_api_set_text(allocator, &report->field_backend, native->backend)) return 0;
    if (!vx_api_set_text(allocator, &report->field_device, native->device)) return 0;

    if (native->offending_node[0]) {
        if (!vx_api_node_ref(allocator, &report->field_offending_node,
                             native->offending_node)) {
            return 0;
        }
    }

    VX_API_NEW_CHILD(allocator, &report->field_lineage, VolvoxaiV1Lineage,
                     volvoxai_v1_lineage_init_with_allocator);
    report->field_lineage->field_execution_id = native->execution_id;
    report->field_lineage->field_runtime_id = native->runtime_id;
    report->field_lineage->field_model_id = native->model_id;
    report->field_lineage->field_compiled_model_id = native->compiled_model_id;
    report->field_lineage->field_context_id = native->context_id;
    report->field_lineage->field_graph_id = native->graph_id;
    report->field_lineage->field_graph_revision = native->graph_revision;
    report->field_lineage->field_weight_id = native->weight_id;
    report->field_lineage->field_weight_revision = native->weight_revision;
    report->field_lineage->field_adapter_id = native->adapter_id;
    report->field_lineage->field_adapter_revision = native->adapter_revision;

    VX_API_NEW_CHILD(allocator, &report->field_timings, VolvoxaiV1OperationTimings,
                     volvoxai_v1_operation_timings_init_with_allocator);
    report->field_timings->field_compile_time_ms = native->compile_time_ms;
    report->field_timings->field_execution_time_ms = native->execution_time_ms;

    VX_API_NEW_CHILD(allocator, &report->field_accounting, VolvoxaiV1OperationAccounting,
                     volvoxai_v1_operation_accounting_init_with_allocator);
    report->field_accounting->field_lineage_allocated_bytes = native->allocated_bytes;
    report->field_accounting->field_result_bytes = native->result_bytes;

    if (!vx_api_compilation_evidence(allocator, report, native)) return 0;
    if (!vx_api_route_evidence(allocator, report, native)) return 0;
    if (!vx_api_fallback_evidence(allocator, report, native)) return 0;
    if (!vx_api_decode_state(allocator, report, native)) return 0;
    /* Only runtimes that opted in produce evidence; every other report is
     * left without the field rather than carrying an empty capture. */
    if (!vx_api_capture_attach(report, native)) return 0;
    if (!vx_api_input_issue(allocator, &report->field_input_issue, &native->input_issue)) return 0;
    return 1;
}

int vx_api_report_attach(const SynurangLiteAllocator* owner_alloc,
                         VolvoxaiV1OperationReport** slot,
                         const VxReport* native) {
    if (!owner_alloc || !slot) return 0;
    VX_API_NEW_CHILD(owner_alloc, slot, VolvoxaiV1OperationReport,
                     volvoxai_v1_operation_report_init_with_allocator);
    return vx_api_report_from_native(*slot, native);
}

void vx_api_report_set_public_lineage(VolvoxaiV1OperationReport* report,
                                      uint64_t runtime_id,
                                      uint64_t model_id,
                                      uint64_t compiled_model_id,
                                      uint64_t context_id) {
    VolvoxaiV1Lineage* lineage;
    if (!report || !report->field_lineage) return;
    lineage = report->field_lineage;
    lineage->field_runtime_id = runtime_id;
    lineage->field_model_id = model_id;
    lineage->field_compiled_model_id = compiled_model_id;
    lineage->field_context_id = context_id;
}

void vx_api_report_set_graph_plan_lineage(VolvoxaiV1OperationReport* report,
                                          uint64_t graph_plan_id) {
    if (!report || !report->field_lineage) return;
    report->field_lineage->field_graph_plan_id = graph_plan_id;
}

int vx_api_report_ok(const SynurangLiteAllocator* owner_alloc,
                     VolvoxaiV1OperationReport** slot,
                     VxStage stage) {
    return vx_api_report_fail(owner_alloc, slot, VX_STATUS_OK, stage, VX_CODE_NONE, "");
}

int vx_api_report_fail(const SynurangLiteAllocator* owner_alloc,
                       VolvoxaiV1OperationReport** slot,
                       VxStatus status,
                       VxStage stage,
                       VxOperationCode code,
                       const char* message) {
    if (!owner_alloc || !slot) return 0;
    VX_API_NEW_CHILD(owner_alloc, slot, VolvoxaiV1OperationReport,
                     volvoxai_v1_operation_report_init_with_allocator);
    (*slot)->field_status = (VolvoxaiV1NativeStatus)status;
    (*slot)->field_stage = (VolvoxaiV1OperationStage)stage;
    (*slot)->field_code = (VolvoxaiV1OperationCode)code;
    if (!vx_api_set_text(owner_alloc, &(*slot)->field_message, message)) return 0;
    return 1;
}

/* ---------------------------------------------------------------------------
 * Tensors
 * ------------------------------------------------------------------------ */

int vx_api_tensor_spec_from_native(const SynurangLiteAllocator* allocator,
                                   VolvoxaiV1TensorSpec* spec,
                                   const VxTensorSpec* native) {
    uint32_t axis;
    VolvoxaiV1DimensionConstraint* dimensions;

    if (!spec || !native) return 0;
    if (!vx_api_set_text(allocator, &spec->field_name, native->name)) return 0;
    spec->field_dtype = (VolvoxaiV1DataType)native->dtype;
    spec->field_location = (VolvoxaiV1MemoryLocation)native->location;
    if (native->rank == 0u) return 1;

    dimensions = (VolvoxaiV1DimensionConstraint*)allocator->allocate(
        allocator->context, sizeof(*dimensions) * native->rank);
    if (!dimensions) return 0;
    spec->field_dimensions.data = dimensions;
    spec->field_dimensions.len = native->rank;
    spec->field_dimensions.cap = native->rank;

    for (axis = 0; axis < native->rank; axis++) {
        const VxDimensionConstraint* source = &native->dimensions[axis];
        VolvoxaiV1DimensionConstraint* target = &dimensions[axis];
        volvoxai_v1_dimension_constraint_init_with_allocator(target, allocator);
        target->field_kind = (VolvoxaiV1DimensionKind)source->kind;
        target->field_min = source->min;
        target->field_max = source->max;
        target->field_multiple_of = source->multiple_of;
        if (!vx_api_set_text(allocator, &target->field_symbol, source->symbol)) return 0;
    }
    return 1;
}

int vx_api_tensor_info_from_native(const SynurangLiteAllocator* allocator,
                                   VolvoxaiV1TensorInfo* info,
                                   const VxTensorInfo* native) {
    uint32_t axis;
    int64_t* shape;

    if (!info || !native) return 0;
    if (!vx_api_set_text(allocator, &info->field_name, native->name)) return 0;
    info->field_dtype = (VolvoxaiV1DataType)native->dtype;
    info->field_byte_size = native->byte_size;
    info->field_location = (VolvoxaiV1MemoryLocation)native->location;
    if (native->rank == 0u) return 1;

    shape = (int64_t*)allocator->allocate(allocator->context,
                                          sizeof(*shape) * native->rank);
    if (!shape) return 0;
    for (axis = 0; axis < native->rank; axis++) shape[axis] = native->shape[axis];
    info->field_shape.data = shape;
    info->field_shape.len = native->rank;
    info->field_shape.cap = native->rank;
    return 1;
}

/* Oneof discriminators are the raw protobuf field numbers of Tensor.payload. */
#define VX_API_TENSOR_PAYLOAD_INLINE 5
#define VX_API_TENSOR_PAYLOAD_VIEW 6

VxStatus vx_api_buffer_view_resolve(const VolvoxaiV1BufferView* view,
                                    void** data,
                                    size_t* byte_size) {
    if (!view || !data || !byte_size) return VX_STATUS_INVALID_ARGUMENT;
    *data = NULL;
    *byte_size = 0u;

#if defined(__wasm__)
    /* A protobuf BufferView received by generated browser dispatch contains a
     * foreign transport value, not a trusted offset allocated by this module.
     * Reject it before validating or constructing an address. */
    return VX_STATUS_TRANSPORT_UNSUPPORTED;
#else
    uint64_t base;
    uint64_t offset;

    if (view->field_handle <= 0 || view->field_offset < 0 ||
        view->field_length < 0) {
        return VX_STATUS_INVALID_ARGUMENT;
    }
    base = (uint64_t)view->field_handle;
    offset = (uint64_t)view->field_offset;
    if (offset > UINT64_MAX - base || base + offset > (uint64_t)UINTPTR_MAX ||
        (uint64_t)view->field_length > (uint64_t)SIZE_MAX) {
        return VX_STATUS_INVALID_ARGUMENT;
    }
    *data = (void*)(uintptr_t)(base + offset);
    *byte_size = (size_t)view->field_length;
    return VX_STATUS_OK;
#endif
}

int vx_api_report_binding_fail(const SynurangLiteAllocator* owner_alloc,
                               VolvoxaiV1OperationReport** slot,
                               VxStatus status, VxStage stage,
                               const char* invalid_message,
                               const VolvoxaiV1Tensor* tensor, size_t input_index) {
    VxInputValidationEvidence evidence = {0};
    VxOperationCode code = status == VX_STATUS_TRANSPORT_UNSUPPORTED
        ? VX_CODE_TRANSPORT_UNSUPPORTED : status == VX_STATUS_OUT_OF_MEMORY
        ? VX_CODE_OUT_OF_MEMORY : VX_CODE_INVALID_ARGUMENT;
    if (!vx_api_report_fail(owner_alloc, slot, status, stage, code,
            status == VX_STATUS_TRANSPORT_UNSUPPORTED
                ? "this transport cannot dereference a BufferView" : invalid_message)) return 0;
    if (!tensor || status == VX_STATUS_OUT_OF_MEMORY) return 1;
    evidence.has_input_index = 1;
    evidence.input_index = input_index;
    evidence.has_actual = 1;
    evidence.actual_dtype = (VxDataType)tensor->field_dtype;
    evidence.actual_location = (VxMemoryLocation)tensor->field_location;
    evidence.actual_rank = tensor->field_shape.len > UINT32_MAX ? UINT32_MAX : (uint32_t)tensor->field_shape.len;
    for (size_t axis = 0; axis < tensor->field_shape.len && axis < VX_MAX_TENSOR_RANK; axis++)
        evidence.actual_shape[axis] = tensor->field_shape.data[axis];
    evidence.actual_bytes = tensor->which_payload == VX_API_TENSOR_PAYLOAD_INLINE
        ? tensor->field_inline.len : tensor->field_view && tensor->field_view->field_length > 0
        ? (uint64_t)tensor->field_view->field_length : 0;
    evidence.code = status == VX_STATUS_TRANSPORT_UNSUPPORTED ? VX_INPUT_TRANSPORT_UNSUPPORTED
        : tensor->field_shape.len > VX_MAX_TENSOR_RANK ? VX_INPUT_RANK_MISMATCH
        : !tensor->field_name.len || memchr(tensor->field_name.data, 0, tensor->field_name.len)
        ? VX_INPUT_INVALID_NAME : tensor->which_payload == VX_API_TENSOR_PAYLOAD_VIEW
        ? VX_INPUT_INVALID_BUFFER_VIEW : VX_INPUT_PAYLOAD_REQUIRED;
    if (!vx_api_input_issue(owner_alloc, &(*slot)->field_input_issue, &evidence)) return 0;
    /* Preserve length-aware names, including malformed NUL-containing input,
     * without ever interpreting them as a C string. */
    return synurang_lite_bytes_assign(owner_alloc,
        &(*slot)->field_input_issue->field_input_name, tensor->field_name.data,
        tensor->field_name.len) == SYNURANG_LITE_OK &&
        synurang_lite_bytes_assign(owner_alloc,
        &(*slot)->field_input_issue->field_actual->field_name, tensor->field_name.data,
        tensor->field_name.len) == SYNURANG_LITE_OK;
}

VxStatus vx_api_binding_from_tensor(VxApiScratch* scratch,
                                    VxTensorBinding* binding,
                                    const VolvoxaiV1Tensor* tensor) {
    size_t axis;
    void* data;
    size_t byte_size;

    if (!scratch || !binding || !tensor) return VX_STATUS_INVALID_ARGUMENT;
    *binding = (VxTensorBinding)VX_TENSOR_BINDING_INIT;

    if (!tensor->field_name.len ||
        memchr(tensor->field_name.data, 0, tensor->field_name.len))
        return VX_STATUS_INVALID_ARGUMENT;
    if (tensor->field_shape.len > (size_t)VX_MAX_TENSOR_RANK) {
        return VX_STATUS_INVALID_ARGUMENT;
    }
    binding->name = vx_api_scratch_cstr(scratch, &tensor->field_name);
    if (!binding->name) return VX_STATUS_OUT_OF_MEMORY;
    binding->dtype = (VxDataType)tensor->field_dtype;
    binding->location = (VxMemoryLocation)tensor->field_location;
    binding->rank = (uint32_t)tensor->field_shape.len;
    for (axis = 0; axis < tensor->field_shape.len; axis++) {
        binding->shape[axis] = tensor->field_shape.data[axis];
    }

    switch (tensor->which_payload) {
        case VX_API_TENSOR_PAYLOAD_INLINE:
            binding->data = tensor->field_inline.data;
            binding->byte_size = tensor->field_inline.len;
            return VX_STATUS_OK;
        case VX_API_TENSOR_PAYLOAD_VIEW:
            {
                VxStatus status = vx_api_buffer_view_resolve(
                    tensor->field_view, &data, &byte_size);
                if (status != VX_STATUS_OK) return status;
            }
            binding->data = data;
            binding->byte_size = byte_size;
            return VX_STATUS_OK;
        default:
            /* A descriptor without a payload cannot be bound as an input. */
            return VX_STATUS_INVALID_ARGUMENT;
    }
}
