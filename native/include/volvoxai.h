#ifndef VOLVOXAI_H
#define VOLVOXAI_H

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

#define VX_NATIVE_API_VERSION UINT32_C(1)
#define VX_MAX_TENSOR_RANK UINT32_C(8)
#define VX_REPORT_BACKEND_CAPACITY 64u
#define VX_REPORT_DEVICE_CAPACITY 128u
#define VX_REPORT_REASON_CAPACITY 64u
#define VX_REPORT_MESSAGE_CAPACITY 256u
#define VX_REPORT_CANDIDATES_CAPACITY 2048u
#define VX_REPORT_ROUTE_CAPACITY 512u
#define VX_REPORT_FALLBACK_CAPACITY 256u
#define VX_REPORT_NODE_CAPACITY 128u
#define VX_REPORT_DECODE_CAPACITY 128u
#define VX_MAX_BACKEND_CANDIDATES 16u

/* Process-envelope sampling is an optional, read-only diagnostic surface.
 * A successful call reports each acquired signal through available_mask;
 * unavailable or failed platform samplers leave the corresponding bit clear
 * and value zero. This makes an available exact zero distinct from an
 * unavailable value without introducing sampling failures into inference. */
#define VX_PROCESS_MEMORY_SAMPLE_ABI_VERSION UINT32_C(1)
#define VX_PROCESS_MEMORY_AVAILABLE_CURRENT_RSS UINT32_C(1)
#define VX_PROCESS_MEMORY_AVAILABLE_PEAK_RSS UINT32_C(2)
#define VX_PROCESS_MEMORY_AVAILABLE_MONOTONIC_TIME UINT32_C(4)

typedef struct VxProcessMemorySampleV1 {
    size_t struct_size;
    uint32_t abi_version;
    uint32_t available_mask;
    uint64_t rss_bytes;
    uint64_t peak_rss_bytes;
    uint64_t monotonic_nanoseconds;
} VxProcessMemorySampleV1;

#define VX_PROCESS_MEMORY_SAMPLE_V1_INIT { \
    sizeof(VxProcessMemorySampleV1), VX_PROCESS_MEMORY_SAMPLE_ABI_VERSION, \
    0, 0, 0, 0 \
}

typedef struct VxRuntime VxRuntime;
typedef struct VxModel VxModel;
typedef struct VxCompiledModel VxCompiledModel;
typedef struct VxExecutionContext VxExecutionContext;
typedef struct VxResult VxResult;
typedef struct VxRequest VxRequest;

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
    char reason[VX_REPORT_REASON_CAPACITY];
    char message[VX_REPORT_MESSAGE_CAPACITY];
    char candidate_outcomes[VX_REPORT_CANDIDATES_CAPACITY];
    char route_evidence[VX_REPORT_ROUTE_CAPACITY];
    char fallback_evidence[VX_REPORT_FALLBACK_CAPACITY];
    char offending_node[VX_REPORT_NODE_CAPACITY];
    char decode_state[VX_REPORT_DECODE_CAPACITY];
} VxReport;

#define VX_REPORT_INIT { \
    sizeof(VxReport), VX_STATUS_OK, VX_STAGE_NONE, 0, \
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, \
    0.0, 0.0, VX_BACKEND_PREFER, VX_OPERATOR_FALLBACK_ALLOW, \
    0, 0, 0, 0, \
    {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0}, {0} \
}

typedef struct VxRuntimeOptions {
    size_t struct_size;
    int32_t debug;
    int32_t cpu_threads;
    /* DIRECT disables scheduled submission. SCHEDULED permits both direct
     * vx_runtime_run calls and scheduled vx_runtime_submit calls. The enum is
     * generated from proto/volvoxai.proto. */
    VxExecutionMode execution_mode;
    size_t max_scheduled_requests;
    size_t max_scheduled_input_bytes;
    /* Maximum coalescing delay for scheduled requests. Zero dispatches
     * immediately; deadlines may shorten a nonzero delay. */
    uint32_t max_batch_delay_milliseconds;
    /* Queued, in-flight, and caller-retained results share these Runtime-wide
     * bounds. Admission precharges each route's declared worst-case output sum;
     * a validated success shrinks that charge to its owned snapshot bytes. The
     * slot and remaining bytes return on failure or final VxResult release. */
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
    /* Ordered canonical lower-case provider names. PREFER tries candidates in
     * order. REQUIRE accepts exactly one candidate. NULL/zero selects the
     * default PREFER policy whose sole candidate is "cpu". */
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
} VxContextOptions;

#define VX_CONTEXT_OPTIONS_INIT \
    { sizeof(VxContextOptions), VX_DECODE_ROW_DISABLED, 0 }

typedef struct VxRevisionInfo {
    size_t struct_size;
    uint64_t graph_id;
    uint64_t graph_revision;
    uint64_t weight_id;
    uint64_t weight_revision;
    uint64_t adapter_id;
    uint64_t adapter_revision;
} VxRevisionInfo;

#define VX_REVISION_INFO_INIT \
    { sizeof(VxRevisionInfo), 0, 0, 0, 0, 0, 0 }

typedef struct VxAdapterSource {
    size_t struct_size;
    /* Stable logical name. Publishing another package under the same name
     * creates the next immutable revision of that adapter. */
    const char* adapter_name;
    /* Optional safetensors adapter package. NULL publishes a metadata-only
     * route revision, which is useful to providers that own adapter storage. */
    const char* package_path;
    /* Optional package-local version override. */
    const char* version_name;
} VxAdapterSource;

#define VX_ADAPTER_SOURCE_INIT \
    { sizeof(VxAdapterSource), NULL, NULL, NULL }

typedef struct VxAdapterRevision {
    size_t struct_size;
    uint64_t adapter_id;
    uint64_t adapter_revision;
} VxAdapterRevision;

#define VX_ADAPTER_REVISION_INIT \
    { sizeof(VxAdapterRevision), 0, 0 }

typedef struct VxTensorInfo {
    size_t struct_size;
    const char* name;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
    VxMemoryLocation location;
} VxTensorInfo;

#define VX_TENSOR_INFO_INIT \
    { sizeof(VxTensorInfo), NULL, VX_DTYPE_F32, 0, {0}, 0, VX_MEMORY_HOST }

/* Logical tensor dimensions are either one positive fixed extent or one
 * canonical bounded symbol. Malformed mixed states are rejected: fixed axes
 * require symbol == NULL, min == max, and multiple_of == 1; symbolic axes
 * require a non-empty canonical symbol and positive min/max/multiple_of. */
typedef enum VxDimensionKind {
    VX_DIMENSION_FIXED = 1,
    VX_DIMENSION_SYMBOLIC = 2
} VxDimensionKind;

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

/* Logical model contract. Returned name and symbolic-dimension strings are
 * borrowed from the execution context and remain valid until that context is
 * released. No byte size is reported because a symbolic tensor has many
 * concrete byte sizes within its bounded domain. */
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
 * The runtime borrows the descriptor, name, and data only until the
 * synchronous API call returns. Device bindings are rejected. */
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

typedef enum VxRuntimeFreshness {
    /* Every admitted request remains eligible until cancelled or completed. */
    VX_RUNTIME_FRESHNESS_ALL = 0,
    /* A request with the same compiled route and nonzero stream key is
     * superseded by a newer admitted LATEST request. Queued replacement is
     * transactional: equal payload storage may be reused across dynamic shapes,
     * while a larger payload requires temporary byte-budget headroom. Already
     * submitted work remains physical but its result is discarded at completion. */
    VX_RUNTIME_FRESHNESS_LATEST = 1,
    /* Refuse or retire queued/completed work whose monotonic deadline elapsed. */
    VX_RUNTIME_FRESHNESS_DROP_IF_LATE = 2
} VxRuntimeFreshness;

#define VX_RUNTIME_PRIORITY_MIN (-1000)
#define VX_RUNTIME_PRIORITY_MAX 1000

typedef struct VxRuntimeSubmitOptions {
    size_t struct_size;
    /* Higher values run first. Queued age raises effective priority; equal
     * effective priorities use earliest-deadline-first, then request id. */
    int32_t priority;
    /* Absolute CLOCK_MONOTONIC microseconds. Zero disables the deadline. It is
     * always an admission/EDF target; only DROP_IF_LATE makes it a hard
     * queued/completion cutoff. */
    uint64_t deadline_monotonic_micros;
    VxRuntimeFreshness freshness;
    /* Required and nonzero for LATEST. Numeric identity is bounded and copied
     * by value; it is never interpreted as a caller-owned string. */
    uint64_t stream_key;
} VxRuntimeSubmitOptions;

#define VX_RUNTIME_SUBMIT_OPTIONS_INIT \
    { sizeof(VxRuntimeSubmitOptions), 0, 0, VX_RUNTIME_FRESHNESS_ALL, 0 }

typedef enum VxRuntimeRequestState {
    VX_RUNTIME_REQUEST_QUEUED = 0,
    VX_RUNTIME_REQUEST_RUNNING = 1,
    VX_RUNTIME_REQUEST_SUCCEEDED = 2,
    VX_RUNTIME_REQUEST_CANCELLED = 3,
    VX_RUNTIME_REQUEST_FAILED = 4,
    VX_RUNTIME_REQUEST_SUPERSEDED = 5
} VxRuntimeRequestState;

typedef struct VxRequestInfo {
    size_t struct_size;
    uint64_t request_id;
    VxRuntimeRequestState state;
    /* BUSY while queued/running; otherwise the terminal execution status. */
    VxStatus status;
    size_t owned_input_bytes;
    /* Set when accepted work crossed its target. ALL/LATEST still publish a
     * successful result; DROP_IF_LATE instead returns DEADLINE_EXCEEDED. */
    int32_t deadline_missed;
} VxRequestInfo;

#define VX_REQUEST_INFO_INIT \
    { sizeof(VxRequestInfo), 0, VX_RUNTIME_REQUEST_QUEUED, \
      VX_STATUS_BUSY, 0, 0 }

#define VX_REQUEST_WAIT_INFINITE UINT64_MAX

/* Resolved per-tensor affine metadata. Numeric values are loaded from the
 * safetensors tensors referenced by graph.quantization.tensors; they are not
 * graph JSON parameters. */
typedef struct VxAffineQuantization {
    size_t struct_size;
    int32_t defined;
    float scale;
    int32_t zero_point;
} VxAffineQuantization;

#define VX_AFFINE_QUANTIZATION_INIT \
    { sizeof(VxAffineQuantization), 0, 0.0f, 0 }

VX_API const char* vx_status_string(VxStatus status);
/* Current CLOCK_MONOTONIC time in microseconds, or zero when unavailable.
 * Submit deadlines use this clock domain. */
VX_API uint64_t vx_runtime_monotonic_time_micros(void);

/* Returns one when the exact v1 descriptor was accepted, even if no platform
 * signal was available, and zero for NULL, struct-size, or ABI-version
 * negotiation failure. The function has no runtime handle and cannot change
 * inference state. */
VX_API int vx_process_memory_sample_v1(VxProcessMemorySampleV1* sample);

VX_API VxStatus vx_runtime_create(const VxRuntimeOptions* options,
                           VxRuntime** out_runtime,
                           VxReport* report);
VX_API void vx_runtime_retain(VxRuntime* runtime);
VX_API void vx_runtime_release(VxRuntime* runtime);
/* Logical close is idempotent. It rejects provider registration, model load,
 * and compilation of already loaded models. Compiled models, their contexts,
 * and stable results retain the pinned runtime state and remain usable. */
VX_API VxStatus vx_runtime_close(VxRuntime* runtime, VxReport* report);

VX_API VxStatus vx_runtime_load_model(VxRuntime* runtime,
                               const VxModelSource* source,
                               VxModel** out_model,
                               VxReport* report);
VX_API void vx_model_retain(VxModel* model);
VX_API void vx_model_release(VxModel* model);
VX_API VxStatus vx_model_revision_info(VxModel* model,
                                VxRevisionInfo* info,
                                VxReport* report);
VX_API VxStatus vx_model_publish_adapter(VxModel* model,
                                  const VxAdapterSource* source,
                                  VxAdapterRevision* published,
                                  VxReport* report);

VX_API VxStatus vx_model_compile(VxModel* model,
                          const VxBackendPolicy* policy,
                          VxCompiledModel** out_compiled,
                          VxReport* report);
VX_API void vx_compiled_model_retain(VxCompiledModel* compiled);
VX_API void vx_compiled_model_release(VxCompiledModel* compiled);
VX_API VxStatus vx_compiled_model_report(const VxCompiledModel* compiled,
                                  VxReport* report);

/* Direct synchronous execution of one logical request, including a
 * caller-authored bulk B=N binding. The Runtime borrows inputs only until this
 * call returns, reuses the exact compiled route's retained mutable context,
 * and allocates neither a request handle nor the Runtime coordinator. It still
 * reserves one bounded result ticket before entering the provider. A busy
 * route returns VX_STATUS_BUSY and never falls back to scheduling. */
VX_API VxStatus vx_runtime_run(
    VxRuntime* runtime,
    VxCompiledModel* compiled,
    const VxTensorBinding* inputs,
    size_t input_count,
    VxResult** out_result,
    VxReport* report);

/* Scheduled stateless Runtime execution. A Runtime created in DIRECT mode
 * rejects submission. SCHEDULED validates normalized metadata, reserves the
 * Runtime request/input and result budgets, then copies HOST payload bytes.
 * Normalized names borrow the retained model's immutable storage; caller
 * descriptors, names, and payloads may be released once submit returns. */
VX_API VxStatus vx_runtime_submit(
    VxRuntime* runtime,
    VxCompiledModel* compiled,
    const VxTensorBinding* inputs,
    size_t input_count,
    const VxRuntimeSubmitOptions* options,
    VxRequest** out_request,
    VxReport* report);
VX_API VxStatus vx_request_poll(const VxRequest* request,
                                VxRequestInfo* info,
                                VxReport* report);
/* Zero is a non-blocking wait. VX_REQUEST_WAIT_INFINITE waits without a
 * timeout. A finite timeout returns BUSY while the request remains live. */
VX_API VxStatus vx_request_wait(VxRequest* request,
                                uint64_t timeout_milliseconds,
                                VxReport* report);
/* Queued work is removed immediately. Running device/CPU work is logically
 * cancelled and its result suppressed after the physical execution returns. */
VX_API VxStatus vx_request_cancel(VxRequest* request, VxReport* report);
/* Returns a retained immutable result on success. The caller releases it with
 * vx_result_release(). */
VX_API VxStatus vx_request_result(VxRequest* request,
                                  VxResult** out_result,
                                  VxReport* report);
VX_API void vx_request_release(VxRequest* request);

VX_API VxStatus vx_compiled_model_create_context(
    VxCompiledModel* compiled,
    const VxContextOptions* options,
    VxExecutionContext** out_context,
    VxReport* report);
VX_API void vx_execution_context_retain(VxExecutionContext* context);
VX_API void vx_execution_context_release(VxExecutionContext* context);
/* Rejects newly submitted operations, drains already accepted operations, and
 * releases runtime-owned execution storage. Safe to call repeatedly. */
VX_API VxStatus vx_execution_context_close(VxExecutionContext* context,
                                    VxReport* report);

VX_API size_t vx_execution_context_input_count(VxExecutionContext* context);
VX_API VxStatus vx_execution_context_input_spec(VxExecutionContext* context,
                                         size_t index,
                                         VxTensorSpec* spec,
                                         VxReport* report);
VX_API VxStatus vx_execution_context_input_affine_quantization(
    VxExecutionContext* context,
    const char* name,
    VxAffineQuantization* quantization,
    VxReport* report);
VX_API VxStatus vx_execution_context_execute(VxExecutionContext* context,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report);
/* Recompute only the leading row_count rows of a fixed-shape sequence graph.
 * This is ordinary execution without retained decode/KV state. Built-in
 * backends support it when the graph's row-aware operators accept the prefix;
 * external providers return BACKEND_UNSUPPORTED. */
VX_API VxStatus vx_execution_context_execute_prefix(
                                      VxExecutionContext* context,
                                      int32_t row_count,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report);
/* Decode contexts are enabled through VxContextOptions. Seed executes the
 * complete graph, step executes the negotiated dependency/row path, and reset
 * clears the context-local decode/KV state. Seed and step return ordinary
 * immutable result snapshots. */
VX_API VxStatus vx_execution_context_decode_seed(VxExecutionContext* context,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report);
VX_API VxStatus vx_execution_context_decode_step(VxExecutionContext* context,
                                          int32_t position,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report);
VX_API VxStatus vx_execution_context_decode_reset(VxExecutionContext* context,
                                           VxReport* report);
/* Adapter changes are FIFO context operations. select pins an exact published
 * revision. rebind explicitly adopts the model's currently published adapter;
 * no context changes revisions implicitly. */
VX_API VxStatus vx_execution_context_select_adapter(
    VxExecutionContext* context,
    const VxAdapterRevision* revision,
    VxReport* report);
VX_API VxStatus vx_execution_context_rebind_adapter(
    VxExecutionContext* context,
    VxReport* report);

VX_API void vx_result_retain(VxResult* result);
VX_API void vx_result_release(VxResult* result);
VX_API uint64_t vx_result_execution_id(const VxResult* result);
VX_API size_t vx_result_output_count(const VxResult* result);
VX_API VxStatus vx_result_output_info(const VxResult* result,
                               size_t index,
                               VxTensorInfo* info,
                               VxReport* report);
/* Copies a named immutable snapshot into caller-owned storage. Passing NULL
 * with capacity zero queries the required byte count. */
VX_API VxStatus vx_result_read(const VxResult* result,
                        const char* name,
                        void* destination,
                        size_t capacity,
                        size_t* required,
                        VxReport* report);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_H */
