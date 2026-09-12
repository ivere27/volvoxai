/* Public value vocabulary shared by the engine and its backend SPI.
 *
 * proto/volvoxai.proto is the public application API; the lifecycle functions
 * that used to live here are an implementation detail and now sit in
 * native/src/runtime/vx_lifecycle.h. What remains public is the type
 * vocabulary that volvoxai_backend.h needs, because deployment code
 * implements provider callbacks against these structs.
 */
#ifndef VOLVOXAI_TYPES_H
#define VOLVOXAI_TYPES_H

#include <stddef.h>
#include <stdint.h>

#include "volvoxai_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define VX_API
#elif defined(__GNUC__) || defined(__clang__)
#define VX_API __attribute__((visibility("default")))
#else
#define VX_API
#endif

#define VX_MAX_TENSOR_RANK UINT32_C(8)
#define VX_REPORT_BACKEND_CAPACITY 64u
#define VX_REPORT_DEVICE_CAPACITY 128u
#define VX_REPORT_MESSAGE_CAPACITY 256u
#define VX_REPORT_CANDIDATES_CAPACITY 2048u
#define VX_REPORT_ROUTE_CAPACITY 512u
#define VX_REPORT_FALLBACK_CAPACITY 256u
#define VX_REPORT_NODE_CAPACITY 128u
#define VX_REPORT_DECODE_CAPACITY 128u

/* Inline, retained input evidence. No pointers into a caller's tensor batch
 * survive validation; public messages are projected from this snapshot. */
typedef struct VxInputValidationEvidence {
    VxInputValidationCode code;
    int has_input_index;
    uint64_t input_index;
    int has_axis;
    uint32_t axis;
    int has_expected;
    int has_actual;
    int has_expected_bytes;
    uint64_t expected_bytes;
    int has_required_extent;
    int64_t required_extent;
    int names_truncated;
    char name[256];
    VxDataType actual_dtype;
    VxMemoryLocation actual_location;
    uint32_t actual_rank;
    int64_t actual_shape[VX_MAX_TENSOR_RANK];
    uint64_t actual_bytes;
    VxDataType expected_dtype;
    uint32_t expected_rank;
    VxDimensionKind kinds[VX_MAX_TENSOR_RANK];
    int64_t minimums[VX_MAX_TENSOR_RANK];
    int64_t maximums[VX_MAX_TENSOR_RANK];
    int64_t multiples[VX_MAX_TENSOR_RANK];
    char symbols[VX_MAX_TENSOR_RANK][64];
} VxInputValidationEvidence;

typedef struct VxReport {
    size_t struct_size;
    VxStatus status;
    VxStage stage;
    uint64_t execution_id;
    uint64_t runtime_id;
    uint64_t model_id;
    uint64_t compiled_model_id;
    uint64_t context_id;
    uint64_t graph_id;
    uint64_t graph_revision;
    uint64_t weight_id;
    uint64_t weight_revision;
    uint64_t adapter_id;
    uint64_t adapter_revision;
    uint64_t allocated_bytes;
    uint64_t result_bytes;
    double compile_time_ms;
    double execution_time_ms;
    VxBackendPolicyMode policy_mode;
    VxOperatorFallback operator_fallback;
    uint32_t candidate_count;
    int32_t tier_fallback_used;
    int32_t operator_fallback_used;
    int32_t route_attested;
    char backend[VX_REPORT_BACKEND_CAPACITY];
    char device[VX_REPORT_DEVICE_CAPACITY];
    VxOperationCode code;
    char message[VX_REPORT_MESSAGE_CAPACITY];
    char candidate_outcomes[VX_REPORT_CANDIDATES_CAPACITY];
    char route_evidence[VX_REPORT_ROUTE_CAPACITY];
    char fallback_evidence[VX_REPORT_FALLBACK_CAPACITY];
    char offending_node[VX_REPORT_NODE_CAPACITY];
    char decode_state[VX_REPORT_DECODE_CAPACITY];
    VxInputValidationEvidence input_issue;
} VxReport;

#define VX_REPORT_INIT { \
    sizeof(VxReport), VX_STATUS_OK, VX_STAGE_NONE, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0.0, 0.0, VX_BACKEND_PREFER, VX_OPERATOR_FALLBACK_ALLOW, \
    0, 0, 0, 0, \
    {0}, {0}, VX_CODE_NONE, {0}, {0}, {0}, {0}, {0}, {0}, {0} \
}

typedef struct VxRuntimeOptions {
    size_t struct_size;
    int32_t debug;
    int32_t cpu_threads;
    /* DIRECT disables scheduler submission. SCHEDULED permits direct and
     * scheduled execution. The enum is generated from the public schema. */
    VxExecutionMode execution_mode;
    size_t max_scheduled_requests;
    size_t max_scheduled_input_bytes;
    /* Maximum coalescing delay for scheduled requests. Zero dispatches
     * immediately; deadlines may shorten a nonzero delay. */
    uint32_t max_batch_delay_milliseconds;
    /* Queued, in-flight, and caller-retained results share these Runtime-wide
     * bounds. Admission precharges each route's declared worst-case output sum;
     * a validated success shrinks that charge to its owned snapshot bytes. The
     * slot and remaining bytes return on failure or final result release. */
    size_t max_unconsumed_results;
    size_t max_unconsumed_result_bytes;
} VxRuntimeOptions;

#define VX_RUNTIME_OPTIONS_INIT \
    { sizeof(VxRuntimeOptions), 0, 0, VX_EXECUTION_MODE_SCHEDULED, 64u, \
      64u * 1024u * 1024u, 0u, 64u, 64u * 1024u * 1024u }

/* Which slots of a declared weight bank to materialize.
 *
 * Residency is part of the immutable loaded Model source. Every compiled
 * context owns a private engine and descriptor table, borrows the compiled
 * immutable weight blobs, and materializes the same selected rows in a
 * context-owned copy-on-write overlay. Slot ids are ascending, unique, and
 * index the bank's full extent. Route indices stay in that global slot space
 * at execution, so a model loaded with a subset still routes by the ids the
 * exporter emitted. */
typedef struct VxBankResidency {
    size_t struct_size;
    /* Weight tensor named by the graph document's "banks" table. */
    const char* bank;
    const uint32_t* slots;
    size_t slot_count;
} VxBankResidency;

#define VX_BANK_RESIDENCY_INIT { sizeof(VxBankResidency), NULL, NULL, 0 }

typedef struct VxModelSource {
    size_t struct_size;
    /* Basename is graph.json or a named *.graph.json document. */
    const char* graph_path;
    const char* const* weight_paths;
    size_t weight_path_count;
    /* Optional; a bank left out is fully resident. */
    const VxBankResidency* bank_residency;
    size_t bank_residency_count;
} VxModelSource;

#define VX_MODEL_SOURCE_INIT { sizeof(VxModelSource), NULL, NULL, 0, NULL, 0 }

typedef struct VxBackendPolicy {
    size_t struct_size;
    VxBackendPolicyMode mode;
    VxOperatorFallback operator_fallback;
    /* Ordered 1..63-character canonical provider names. Each begins with a
     * lower-case letter and then uses lower-case letters, digits, dot,
     * underscore, or dash.
     * PREFER tries candidates in order. REQUIRE accepts exactly one candidate.
     * NULL/zero selects the default PREFER policy whose sole candidate is
     * "cpu". */
    const char* const* backends;
    size_t backend_count;
} VxBackendPolicy;

#define VX_BACKEND_POLICY_INIT \
    { sizeof(VxBackendPolicy), VX_BACKEND_PREFER, VX_OPERATOR_FALLBACK_ALLOW, \
      NULL, 0 }

typedef struct VxContextOptions {
    size_t struct_size;
    VxDecodeRowMode decode_row_mode;
    int32_t require_incremental;
    uint32_t decode_lanes;
    const char* const* decode_inputs;
    size_t decode_input_count;
} VxContextOptions;

#define VX_CONTEXT_OPTIONS_INIT \
    { sizeof(VxContextOptions), VX_DECODE_ROW_DISABLED, 0, 1, NULL, 0 }

/* Logical tensor dimensions are either one positive fixed extent or one
 * canonical bounded symbol. Malformed mixed states are rejected: fixed axes
 * require symbol == NULL, min == max, and multiple_of == 1; symbolic axes
 * require a non-empty canonical symbol and positive min/max/multiple_of.
 * VxDimensionKind is generated from proto/volvoxai.proto. */

typedef struct VxDimensionConstraint {
    size_t struct_size;
    VxDimensionKind kind;
    const char* symbol;
    int64_t min;
    int64_t max;
    int64_t multiple_of;
} VxDimensionConstraint;

#define VX_DIMENSION_CONSTRAINT_INIT \
    { sizeof(VxDimensionConstraint), VX_DIMENSION_FIXED, NULL, 1, 1, 1 }

/* Logical model contract. Provider compile callbacks borrow every string for
 * the callback duration. No byte size is reported because a symbolic tensor
 * has many concrete byte sizes within its bounded domain. */
typedef struct VxTensorSpec {
    size_t struct_size;
    const char* name;
    VxDataType dtype;
    uint32_t rank;
    VxDimensionConstraint dimensions[VX_MAX_TENSOR_RANK];
    VxMemoryLocation location;
} VxTensorSpec;

#define VX_TENSOR_SPEC_INIT \
    { sizeof(VxTensorSpec), NULL, VX_DTYPE_F32, 0, {{0}}, VX_MEMORY_HOST }

/* One concrete host tensor supplied as part of an atomic execution batch.
 * Provider callbacks borrow the descriptor, name, and data for the callback
 * duration. Device bindings are rejected. */
typedef struct VxTensorBinding {
    size_t struct_size;
    const char* name;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    const void* data;
    size_t byte_size;
    VxMemoryLocation location;
} VxTensorBinding;

#define VX_TENSOR_BINDING_INIT \
    { sizeof(VxTensorBinding), NULL, VX_DTYPE_F32, 0, {0}, NULL, 0, \
      VX_MEMORY_HOST }

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_TYPES_H */
