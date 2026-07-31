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

typedef struct VxRuntime VxRuntime;
typedef struct VxModel VxModel;
typedef struct VxCompiledModel VxCompiledModel;
typedef struct VxExecutionContext VxExecutionContext;
typedef struct VxResult VxResult;

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
} VxRuntimeOptions;

#define VX_RUNTIME_OPTIONS_INIT { sizeof(VxRuntimeOptions), 0, 0 }

typedef struct VxModelSource {
    size_t struct_size;
    /* Basename is graph.json or a named *.graph.json document. */
    const char* graph_path;
    const char* const* weight_paths;
    size_t weight_path_count;
} VxModelSource;

#define VX_MODEL_SOURCE_INIT { sizeof(VxModelSource), NULL, NULL, 0 }

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
VX_API VxStatus vx_execution_context_input_info(VxExecutionContext* context,
                                         size_t index,
                                         VxTensorInfo* info,
                                         VxReport* report);
VX_API VxStatus vx_execution_context_input_affine_quantization(
    VxExecutionContext* context,
    const char* name,
    VxAffineQuantization* quantization,
    VxReport* report);
VX_API VxStatus vx_execution_context_set_input(VxExecutionContext* context,
                                        const char* name,
                                        VxDataType dtype,
                                        const void* data,
                                        size_t byte_size,
                                        VxReport* report);
VX_API VxStatus vx_execution_context_execute(VxExecutionContext* context,
                                      VxResult** out_result,
                                      VxReport* report);
/* Recompute only the leading row_count rows of a fixed-shape sequence graph.
 * This is ordinary execution without retained decode/KV state. Built-in
 * backends support it when the graph's row-aware operators accept the prefix;
 * external providers return BACKEND_UNSUPPORTED. */
VX_API VxStatus vx_execution_context_execute_prefix(
                                      VxExecutionContext* context,
                                      int32_t row_count,
                                      VxResult** out_result,
                                      VxReport* report);
/* Decode contexts are enabled through VxContextOptions. Seed executes the
 * complete graph, step executes the negotiated dependency/row path, and reset
 * clears the context-local decode/KV state. Seed and step return ordinary
 * immutable result snapshots. */
VX_API VxStatus vx_execution_context_decode_seed(VxExecutionContext* context,
                                          VxResult** out_result,
                                          VxReport* report);
VX_API VxStatus vx_execution_context_decode_step(VxExecutionContext* context,
                                          int32_t position,
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
