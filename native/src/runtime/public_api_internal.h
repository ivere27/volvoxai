#ifndef VOLVOXAI_PUBLIC_API_INTERNAL_H
#define VOLVOXAI_PUBLIC_API_INTERNAL_H

#include "vx_lifecycle.h"
#include "engine_core.h"
#include "paged_kv.h"

typedef struct VxWeightRevisionRecord VxWeightRevisionRecord;
typedef struct VxEngineState VxEngineState;
typedef struct VxGraphPlan VxGraphPlan;
typedef struct VxResolvedGraphPlan VxResolvedGraphPlan;

typedef struct {
    uint32_t lanes;
    int prefilled;
    VxDecodeMode mode;
    const uint32_t* active_lengths;
    const uint8_t* parked;
    uint64_t cache_generation;
} VxDecodeStateView;

typedef VxStatus (*VxDecodeStateWriter)(const VxDecodeStateView* state, void* user);
VxStatus vx_execution_context_inspect_decode(VxExecutionContext* context,
    VxDecodeStateWriter write, void* user, VxReport* report);
/* Private lane encoding: -1 is parked; -2 is idle. Public callers use the
 * generated DecodeLaneAction oneof instead of numeric sentinels. */
VxStatus vx_execution_context_decode_lanes(VxExecutionContext* context,
    int prefill, int32_t position, const int32_t* positions, size_t lane_count,
    const VxTensorBinding* inputs, size_t input_count, VxResult** result, VxReport* report);

typedef struct {
    const char* token_input;
    const char* keep_input;
    const char* token_output;
    uint32_t token_count;
    int has_cache_generation;
    uint64_t cache_generation;
} VxDecodeFeedbackOptions;
VxStatus vx_execution_context_decode_generate(VxExecutionContext* context,
    const VxDecodeFeedbackOptions* options, VxResult** result, VxReport* report);

typedef enum {
    VX_DECODE_CACHE_INSPECT, VX_DECODE_CACHE_CONFIGURE, VX_DECODE_CACHE_PUBLISH,
    VX_DECODE_CACHE_REUSE, VX_DECODE_CACHE_RELEASE_LANE, VX_DECODE_CACHE_EVICT
} VxDecodeCacheAction;
typedef struct {
    VxDecodeCacheAction action;
    const char* const* tensors;
    size_t tensor_count;
    uint32_t page_tokens, max_pages;
    int has_max_pages, policy, clear_on_recycle;
    const char* key;
    uint32_t lane, tokens, free_pages;
} VxDecodeCacheCommand;
typedef struct {
    const VxPagedKVCache* cache;
    const char* const* tensors;
    size_t tensor_count;
    int policy;
    uint64_t pool_bytes, prefix_snapshot_bytes, cache_generation;
} VxDecodeCacheView;
typedef VxStatus (*VxDecodeCacheWriter)(const VxDecodeCacheView*, void*);
VxStatus vx_execution_context_decode_cache(VxExecutionContext* context,
    const VxDecodeCacheCommand* command, VxDecodeCacheWriter write, void* user, VxReport* report);

typedef struct VxPlanningWeightSpec {
    const char* name;
    VxDataType dtype;
    const int64_t* shape;
    size_t rank;
} VxPlanningWeightSpec;

typedef struct VxPlanningQuantizationSpec {
    const char* tensor_name;
    uint32_t axis;
    const float* scales;
    const int32_t* zero_points;
    size_t count;
    int per_axis;
} VxPlanningQuantizationSpec;

typedef struct VxStandaloneGraphPlanSource {
    const uint8_t* graph_document;
    size_t graph_document_bytes;
    const VxPlanningWeightSpec* weights;
    size_t weight_count;
    const VxPlanningQuantizationSpec* quantization;
    size_t quantization_count;
} VxStandaloneGraphPlanSource;

typedef struct VxPlanningInputShape {
    const char* name;
    VxDataType dtype;
    const int64_t* shape;
    size_t rank;
} VxPlanningInputShape;

typedef struct VxGraphPlanResolveSource {
    int minimum_binding;
    const VxPlanningInputShape* inputs;
    size_t input_count;
    uint32_t call_kind;
    const char* const* changed_inputs;
    size_t changed_input_count;
    const VxBankResidency* bank_residency;
    size_t bank_residency_count;
} VxGraphPlanResolveSource;

typedef struct VxPlanningStoredWeight {
    char name[128];
    VxDataType dtype;
    int64_t shape[8];
    size_t rank;
} VxPlanningStoredWeight;

typedef struct VxGraphPlanView {
    /* Borrowed immutable cJSON document; private to native projections. */
    const void* logical_graph_document;
    const VxPlanningStoredWeight* stored_weights;
    size_t stored_weight_count;
    uint32_t source_kind;
    const char* graph_fingerprint;
    const char* plan_identity;
    const char* shape_domain_proof_identity;
    uint64_t graph_identity;
    uint64_t graph_revision;
    uint64_t weight_identity;
    uint64_t weight_revision;
    uint64_t adapter_identity;
    uint64_t adapter_revision;
    const uint8_t* graph_plan_request_v1;
    uint32_t graph_plan_request_v1_bytes;
    const uint8_t* graph_plan_response_v1;
    uint32_t graph_plan_response_v1_bytes;
    const uint8_t* graph_bind_definition_v1;
    uint32_t graph_bind_definition_v1_bytes;
    const uint8_t* graph_domain_response_v1;
    uint32_t graph_domain_response_v1_bytes;
    uint32_t graph_domain_terminal_kind;
    int32_t graph_bind_definition_status_v1;
    uint32_t graph_bind_definition_error_section_v1;
    uint32_t graph_bind_definition_error_index_v1;
    uint32_t graph_bind_definition_error_subindex_v1;
    int32_t graph_domain_status_v1;
    /* Canonical public refusal projection; UINT32_MAX means no typed ref. */
    uint32_t shape_diagnostic_code;
    uint32_t shape_diagnostic_node_index;
    uint32_t shape_diagnostic_tensor_index;
    const uint8_t* independent_batch_response_v1;
    uint32_t independent_batch_response_v1_bytes;
    int independent_batch_validated;
    int independent_batch_supported;
    uint32_t independent_batch_reason;
    uint32_t independent_batch_covered_nodes;
    uint32_t independent_batch_failed_node_index;
    uint32_t independent_batch_failed_tensor_index;
    uint32_t independent_batch_axis;
    uint64_t independent_batch_minimum;
    uint64_t independent_batch_maximum;
    uint64_t independent_batch_multiple_of;
    const char* independent_batch_symbol;
    const char* independent_batch_message;
    const char* independent_batch_failed_node;
    const char* independent_batch_failed_tensor;
    const char* independent_batch_proof_identity;
    const VxBankResidency* bank_residency;
    size_t bank_residency_count;
} VxGraphPlanView;

typedef struct VxResolvedGraphPlanView {
    const char* graph_fingerprint;
    const char* plan_identity;
    uint32_t source_kind;
    const char* signature;
    uint64_t signature_digest;
    const uint8_t* graph_bind_response_v1;
    uint32_t graph_bind_response_v1_bytes;
    const uint8_t* call_sequence_request_v1;
    uint32_t call_sequence_request_v1_bytes;
    const uint8_t* call_sequence_response_v1;
    uint32_t call_sequence_response_v1_bytes;
} VxResolvedGraphPlanView;

/* Immutable compile-time domain evidence already proved by the C engine.
 * The generated protobuf boundary borrows this view only while projecting a
 * successful CompileModel response. */
typedef struct VxCompiledDomainAttestationView {
    const char* graph_fingerprint;
    const char* shape_domain_proof_identity;
    uint64_t maximum_tensor_bytes;
    uint64_t maximum_resident_bytes;
    uint64_t resource_limit_bytes;
    int32_t has_resource_limit;
} VxCompiledDomainAttestationView;

#if defined(VOLVOXAI_TEST_IDENTITY_LAYOUT)
typedef struct VxGraphPlanIdentitySnapshot {
    char plan_identity[96];
    char shape_domain_proof_identity[96];
    char independent_batch_proof_identity[128];
} VxGraphPlanIdentitySnapshot;
#endif

#if defined(__GNUC__) && !defined(_WIN32)
#define VX_PUBLIC_INTERNAL __attribute__((visibility("hidden")))
#else
#define VX_PUBLIC_INTERNAL
#endif

/* Snapshot the providers that completed initialization for this Runtime.
 * The first entry is always the built-in CPU provider. The caller owns the
 * returned contiguous name array and releases it with free(). */
VX_PUBLIC_INTERNAL VxStatus vx_runtime_internal_backend_names(
    VxRuntime* runtime,
    char (**out_names)[VX_REPORT_BACKEND_CAPACITY],
    size_t* out_count,
    VxReport* report);

/* Side-effect-free validation used by the browser path transport before it
 * fetches caller-named files. These functions deliberately stop before file
 * I/O, object/revision identity allocation, or publication. The corresponding
 * public lifecycle operations use the same implementation, so this private
 * boundary cannot drift into a second model/adapter contract. */
#if defined(__wasm__)
VX_PUBLIC_INTERNAL VxStatus vx_runtime_internal_preflight_load_model(
    VxRuntime* runtime,
    const VxModelSource* source,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_preflight_publish_adapter(
    VxModel* model,
    const VxAdapterSource* source,
    VxReport* report);
#endif

typedef struct VxSourceBytes {
    const void* data;
    size_t size;
} VxSourceBytes;

typedef struct VxModelPackageSource {
    VxSourceBytes graph;
    const VxSourceBytes* weights;
    size_t weight_count;
} VxModelPackageSource;

/* Uses the same validation/snapshot owner as path loading. Preflight performs
 * no I/O or publication. The public operation is LoadModel in the proto. */
VX_PUBLIC_INTERNAL VxStatus vx_runtime_internal_load_model_package(
    VxRuntime* runtime, const VxModelSource* source,
    const VxModelPackageSource* package, VxModel** out_model,
    VxReport* report, int preflight_only);

VX_PUBLIC_INTERNAL size_t vx_model_internal_input_count(const VxModel* model);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_input_spec(
    const VxModel* model, size_t index, VxTensorSpec* spec);
VX_PUBLIC_INTERNAL size_t vx_model_internal_output_count(const VxModel* model);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_output_spec(
    const VxModel* model, size_t index, VxTensorSpec* spec);

VX_PUBLIC_INTERNAL VxStatus vx_graph_plan_internal_create_model(
    VxModel* model,
    VxGraphPlan** out_plan,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_graph_plan_internal_create_standalone(
    const VxStandaloneGraphPlanSource* source,
    VxGraphPlan** out_plan,
    VxReport* report);
VX_PUBLIC_INTERNAL void vx_graph_plan_internal_retain(VxGraphPlan* plan);
VX_PUBLIC_INTERNAL void vx_graph_plan_internal_release(VxGraphPlan* plan);
VX_PUBLIC_INTERNAL VxStatus vx_graph_plan_internal_view(
    const VxGraphPlan* plan,
    VxGraphPlanView* out_view);
#if defined(VOLVOXAI_TEST_IDENTITY_LAYOUT)
VX_PUBLIC_INTERNAL VxStatus vx_graph_plan_test_identity_snapshot(
    const VxGraphPlan* plan,
    VxGraphPlanIdentitySnapshot* out_snapshot);
VX_PUBLIC_INTERNAL VxStatus vx_model_test_proof_identity_snapshot(
    const VxModel* model,
    VxGraphPlanIdentitySnapshot* out_snapshot);
#endif
VX_PUBLIC_INTERNAL VxStatus vx_graph_plan_internal_resolve(
    const VxGraphPlan* plan,
    const VxGraphPlanResolveSource* source,
    VxResolvedGraphPlan** out_resolved,
    VxReport* report);
VX_PUBLIC_INTERNAL void vx_resolved_graph_plan_internal_release(
    VxResolvedGraphPlan* resolved);
VX_PUBLIC_INTERNAL VxStatus vx_resolved_graph_plan_internal_view(
    const VxResolvedGraphPlan* resolved,
    VxResolvedGraphPlanView* out_view);
VX_PUBLIC_INTERNAL VxStatus vx_compiled_model_internal_domain_attestation(
    const VxCompiledModel* compiled,
    VxCompiledDomainAttestationView* out_view);

#if VOLVOXAI_ENABLE_TRAINING
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_accept_full_owner(
    VxModel* model,
    VxWeightRevisionRecord** out_revision,
    VxReport* report);
VX_PUBLIC_INTERNAL void vx_model_internal_release_weight_revision(
    VxWeightRevisionRecord* revision);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_create_authoring_engine(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    VxBackendKind backend,
    const VolvoxAIEngineShapePolicy* shape_policy,
    VxEngineState** out_state,
    VxReport* report);
VX_PUBLIC_INTERNAL void vx_model_internal_destroy_authoring_engine(
    VxEngineState* state);
VX_PUBLIC_INTERNAL const char* vx_model_internal_graph_path(const VxModel* model);
VX_PUBLIC_INTERNAL const char* vx_model_internal_logical_fingerprint(
    const VxModel* model);
/* Validate one complete shaped batch, resolve the logical graph, optionally
 * require an existing accumulation signature, and atomically bind the private
 * authoring engine. `out_shape_signature` is heap-owned by the caller. */
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_bind_authoring_inputs(
    VxModel* model,
    VxEngineState* state,
    VxBackendKind backend,
    const VxTensorBinding* inputs,
    size_t input_count,
    const char* required_shape_signature,
    char** out_shape_signature,
    VolvoxAIEngineDynamicShapeStats* out_stats,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_validate_authoring_revision(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    uint64_t adapter_id,
    uint64_t adapter_revision,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_prepare_weight_revision(
    VxModel* model,
    const char* const* weight_paths,
    size_t weight_path_count,
    VxWeightRevisionRecord** out_revision,
    VxReport* report);
VX_PUBLIC_INTERNAL VxStatus vx_model_internal_publish_weight_revision(
    VxModel* model,
    VxWeightRevisionRecord* expected,
    VxWeightRevisionRecord* successor,
    VxReport* report);
#endif

#undef VX_PUBLIC_INTERNAL

#endif /* VOLVOXAI_PUBLIC_API_INTERNAL_H */
