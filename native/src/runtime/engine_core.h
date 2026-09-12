#ifndef VOLVOXAI_ENGINE_CORE_H
#define VOLVOXAI_ENGINE_CORE_H

#include "volvoxai_enums.h"
#include "generated/backend_vocabulary.h"

#include <stddef.h>
#include <stdint.h>

/* Private immutable device copy. Poll returns 1 pending, 0 ready, -1 failed.
 * The ticket owns its source snapshot independently of every engine context. */
typedef struct {
    uint32_t ticket;
    size_t byte_size;
    void (*convert)(void* destination, size_t bytes);
    int (*poll)(uint32_t ticket, void* destination, size_t bytes);
    void (*release)(uint32_t ticket);
} VxDeviceSnapshot;
int volvoxai_engine_snapshot_tensor(const char* name, VxDeviceSnapshot* snapshot);

#ifdef __cplusplus
extern "C" {
#endif

typedef enum {
    VOLVOXAI_ENGINE_NODE_PATCH_MODE_UNSPECIFIED = 0,
    VOLVOXAI_ENGINE_NODE_PATCH_MODE_MERGE = 1,
    VOLVOXAI_ENGINE_NODE_PATCH_MODE_REPLACE = 2,
    VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_BEFORE = 3,
    VOLVOXAI_ENGINE_NODE_PATCH_MODE_INSERT_AFTER = 4,
    VOLVOXAI_ENGINE_NODE_PATCH_MODE_DELETE = 5
} VolvoxAIEngineNodePatchMode;

/* Compatibility spellings for the canonical protobuf-generated dtype. */
enum {
    VOLVOXAI_DTYPE_F32 = VX_DTYPE_F32,
    VOLVOXAI_DTYPE_I8 = VX_DTYPE_I8,
    VOLVOXAI_DTYPE_U8 = VX_DTYPE_U8,
    VOLVOXAI_DTYPE_I32 = VX_DTYPE_I32,
    VOLVOXAI_DTYPE_F16 = VX_DTYPE_F16
};


typedef struct {
    VxBackendKind backend;
    int debug;
    int cpu_threads; /* zero keeps the runtime default */
} VolvoxAIEngineOptions;

/* Concrete projection of one logical graph tensor for a single execution.
 * These descriptors are internal to the native engine boundary: public
 * callers bind VxTensorBinding values through volvoxai.h. */
typedef struct {
    const char* name;
    int dtype;
    int rank;
    int shape[8];
} VolvoxAIEngineResolvedTensor;

/* Pointwise shape/byte maxima for the complete declared shape domain.  This
 * is an allocation proof input, not a jointly executable shape: correlated
 * tensor maxima need not be realizable by one request. */
typedef struct {
    const char* name;
    int dtype;
    int rank;
    int maximum_shape[8];
    size_t maximum_byte_size;
} VolvoxAIEngineMaximumTensor;

/* One context-owned physical activation span.  Several logical tensors may
 * share a span only when their immutable topology lifetimes do not overlap. */
typedef struct {
    const void* host;
    size_t capacity_bytes;
} VolvoxAIEnginePhysicalSpan;

typedef struct {
    const char* name;
    int dtype;
    const void* data;
    size_t byte_size;
} VolvoxAIEngineInputBinding;

typedef struct {
    size_t struct_size;
    int plan_cache_hit;
    size_t logical_bytes;
    size_t required_arena_bytes;
    size_t arena_capacity_bytes;
    size_t arena_high_water_bytes;
    uint64_t arena_grow_count;
    uint64_t resource_generation;
    uint32_t plan_cache_evictions;
} VolvoxAIEngineDynamicShapeStats;

/* Private execution policy. Zero fields retain inference defaults. */
typedef struct {
    uint32_t plan_cache_entries;
    uint64_t plan_cache_metadata_bytes;
    uint64_t max_activation_capacity_bytes;
    double capacity_growth_factor;
    int retain_activations;
} VolvoxAIEngineShapePolicy;

#define VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT \
    { sizeof(VolvoxAIEngineDynamicShapeStats), 0, 0, 0, 0, 0, 0, 0, 0 }

/* Backend-neutral autoregressive execution. A decode session is created after
 * model initialization, then prefilled once after the caller writes all inputs.
 * Later steps rerun the dependency closure; CPU may additionally refresh one
 * decoder row/KV prefix when a positive position is supplied. Ordinary
 * forwards, explicit incremental resets, and graph/weight mutations clear the
 * session's prefilled state, so the next session operation must be prefill(). */
typedef enum {
    VOLVOXAI_DECODE_MODE_NONE = 0,
    VOLVOXAI_DECODE_MODE_ORDINARY_FORWARD = 1,
    VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY = 2,
    VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW = 3
} VolvoxAIDecodeMode;

typedef enum {
    VOLVOXAI_DECODE_ROW_AUTO = 0,
    VOLVOXAI_DECODE_ROW_REQUIRED = 1,
    VOLVOXAI_DECODE_ROW_DISABLED = 2
} VolvoxAIDecodeRowMode;

typedef struct {
    size_t struct_size;
    VolvoxAIDecodeRowMode row_mode;
    int require_incremental;
    uint32_t lanes;
} VolvoxAIDecodeSessionOptions;

#define VOLVOXAI_DECODE_SESSION_OPTIONS_INIT \
    { sizeof(VolvoxAIDecodeSessionOptions), VOLVOXAI_DECODE_ROW_AUTO, 0, 1 }

typedef struct VolvoxAIDecodeSession VolvoxAIDecodeSession;

// Native graph engine: parses a Volvox graph package (graph.json + safetensors),
// builds the graph once, and runs it node-by-node on the CPU/GPU kernels.
// Training-specific declarations live in training_core.h.
//
// Configure optional device/debug policy before loading a graph. Calling this
// while a graph is loaded is rejected; explicit device requests never silently
// fall back to CPU. A rejected option set or failed backend initialization
// leaves the previously configured options/backend unchanged.
int    volvoxai_engine_configure(const VolvoxAIEngineOptions* options);
int    volvoxai_engine_configure_backend(const char* name);
int    volvoxai_engine_get_options(VolvoxAIEngineOptions* options);
const char* volvoxai_engine_backend_name(void);
int    volvoxai_engine_set_debug(int enabled);
int    volvoxai_engine_debug(void);
/* Select an optional row for row-aware execution/output paths. -1 restores
 * whole-tensor execution. This is graph scheduling state, not token policy. */
int    volvoxai_engine_set_execution_row(int row);
int    volvoxai_engine_execution_row(void);
/* Graph-declared interface names in declaration order. Returned strings remain
 * owned by the engine and are valid until shutdown or the next init. */
int    volvoxai_engine_graph_input_count(void);
const char* volvoxai_engine_graph_input_name(int index);
int    volvoxai_engine_graph_output_count(void);
const char* volvoxai_engine_graph_output_name(int index);

/* weights_path may be NULL for a graph whose declared dependencies are fully
 * satisfied by graph inputs and node outputs. Strict graph preflight still
 * rejects unresolved weight names. Both init entry points return 0 on success,
 * -2 for graph metadata allocation failure, and -1 otherwise. */
int    volvoxai_engine_init(const char* graph_path, const char* weights_path);
/* weight_file_count may be zero when weight_file_paths is NULL. */
int    volvoxai_engine_init_with_weight_files(const char* graph_path,
                                              const char* const* weight_file_paths,
                                              int weight_file_count);
/* Mutable view of an F32 graph input; typed inputs return NULL. */
float* volvoxai_engine_input_ptr(const char* name, long* numel);
int    volvoxai_engine_set_input_raw(const char* name, int dtype, const void* data, size_t nbytes);
/* Validate the complete resolved descriptor/input batch, prepare a cached
 * liveness plan, grow one reusable host activation arena transactionally,
 * prebind any selected native graph backend, and only then publish tensor
 * metadata and input bytes. A negative return leaves the previously committed
 * tensor descriptors, pointers, and arena intact. */
int    volvoxai_engine_commit_dynamic_shape(
           const char* signature,
           const VolvoxAIEngineResolvedTensor* tensors,
           size_t tensor_count,
           const VolvoxAIEngineInputBinding* inputs,
           size_t input_count,
           VolvoxAIEngineDynamicShapeStats* stats);
/* Build the fixed topology-liveness fallback from pointwise tensor byte
 * maxima. Every later concrete plan is checked against this bound and uses
 * its fixed offsets. The function never executes a synthetic maximum shape. */
int    volvoxai_engine_configure_dynamic_shape_domain(
           const VolvoxAIEngineMaximumTensor* tensors,
           size_t tensor_count);
/* Bootstrap kernels classify operands by port use. Clear any temporary
 * weight classification attached to logical tensor identities before true
 * immutable model origins are retained. */
int    volvoxai_engine_demote_preloaded_logical_tensors_locked(void);
/* Allocate the proved host arena and reserve every backend physical span while
 * the owning context is still unpublished. A later exact bind consumes this
 * reservation without growing host or device capacity. */
int    volvoxai_engine_reserve_dynamic_shape_domain(void);
/* Atomic validation/copy path used by decode steps after a successful prefill.
 * It deliberately does not re-plan: decode steps must keep the prefilled shape
 * signature and retained incremental storage exactly. */
int    volvoxai_engine_commit_input_bindings(
           const VolvoxAIEngineInputBinding* inputs,
           size_t input_count);
/* Reserve one context-owned CPU workspace at its already proved maximum.
 * Kernels may borrow it only for the duration of a serialized engine call and
 * must accept an undersized/NULL view by selecting an allocation-free route. */
int    volvoxai_engine_configure_cpu_typed_workspace(size_t bounded_bytes);
void*  volvoxai_engine_cpu_typed_workspace(size_t* capacity_bytes);
/* Copy real F32 values into an F32 graph input, or quantize them into a
 * per-tensor I8/U8 graph input using its declared scale and zero point. */
int    volvoxai_engine_set_input_f32(const char* name, const float* data, long numel);
int    volvoxai_engine_is_graph_input(const char* name);
int    volvoxai_engine_is_model_weight(const char* name);
int    volvoxai_engine_copy_tensor_f32(const char* name, float* out, long numel);
int    volvoxai_engine_copy_tensor_raw(const char* name, void* out, size_t nbytes);
int    volvoxai_engine_tensor_info(const char* name, long* numel, int* shape, int* ndim);
int    volvoxai_engine_tensor_info_ex(const char* name, long* numel, int* shape, int* ndim, int* dtype, size_t* elem_size);
/* Returns 1 for a graph input with resolved per-tensor affine metadata, 0 for
 * an unquantized graph input, and -1 for an unknown/non-input tensor. */
int    volvoxai_engine_input_affine_quantization(const char* name,
                                                 float* scale,
                                                 int* zero_point);
int    volvoxai_engine_set_tensor_f32(const char* name, const float* data, long numel);
int    volvoxai_engine_set_tensor_raw(const char* name, int dtype, const void* data, size_t nbytes);
int    volvoxai_engine_add_model_tensor_raw(const char* name, const int* shape, int ndim,
                                            int dtype, const void* data, size_t nbytes);
int    volvoxai_engine_remove_model_tensor(const char* name);
int    volvoxai_engine_prepare_tensor_table_mutation(void);
#if VOLVOXAI_ENABLE_TRAINING
int    volvoxai_engine_prepare_training_transition(void);
#endif
void   volvoxai_engine_finish_tensor_table_mutation(void);
int    volvoxai_engine_tensor_weight_file_index(const char* name);
int    volvoxai_engine_linear_weight_layout(const char* weight_name, int* d_in, int* d_out, int* out_in);
int    volvoxai_engine_save_weight_file(int weight_file_index, const char* path);
int    volvoxai_engine_weight_file_bytes(int weight_file_index,
                                          unsigned char** bytes, size_t* size);
int    volvoxai_engine_adapter_stage_json(const char* manifest_json,
                                 const char* const* tensor_names, const void* const* tensor_data,
                                 const int* tensor_dtypes, const size_t* tensor_nbytes, int tensor_count);
int    volvoxai_engine_adapter_clone_update(const char* source_version, const char* new_adapter_id,
                                   const char* new_version_id, const char* const* tensor_names,
                                   const void* const* tensor_data, const int* tensor_dtypes,
                                   const size_t* tensor_nbytes, const int* update_modes, int tensor_count);
int    volvoxai_engine_adapter_clone_update_with_metadata(const char* source_version, const char* new_adapter_id,
                                                 const char* new_version_id, const char* const* tensor_names,
                                                 const void* const* tensor_data, const int* tensor_dtypes,
                                                 const size_t* tensor_nbytes, const int* update_modes,
                                                 int tensor_count, const char* metadata_json);
int    volvoxai_engine_adapter_load(const char* path, const char* version_override);
int    volvoxai_engine_adapter_save(const char* version_id, const char* path);
char*  volvoxai_engine_adapter_list_json(void);
int    volvoxai_engine_adapter_activate(const char* version_id);
int    volvoxai_engine_adapter_remove(const char* version_id);
int    volvoxai_engine_adapter_merge(const char* version_id);
int    volvoxai_engine_adapter_unmerge(void);
int    volvoxai_engine_adapter_route_begin(const char* version_id);
int    volvoxai_engine_adapter_route_begin_many(const char* const* version_ids, const float* scales, int count);
void   volvoxai_engine_adapter_route_end(void);
char*  volvoxai_engine_inspect_model_json(int include_graph, int include_tensors, int include_weight_files, int include_params);
// For INSERT_BEFORE/INSERT_AFTER, node_index == -1 appends (and creates the first node of an empty graph).
int    volvoxai_engine_patch_node_json(int node_index, const char* patch_json,
                                       VolvoxAIEngineNodePatchMode mode,
                                       int rebuild_execution_plan);
int    volvoxai_engine_patch_graph_json(const char* patches_json,
                                        int rebuild_execution_plan,
                                        int reoptimize);
int    volvoxai_engine_save_graph(const char* path);
int    volvoxai_engine_forward(void);                              // run the whole graph (all rows)
/* Opt-in dependency-aware forward. The first call runs the complete graph;
 * later calls execute only nodes reachable from graph inputs written through
 * set_input_raw() since the previous incremental call. Cached intermediates
 * remain owned by the engine. Call incremental_reset() after mutating an input
 * through input_ptr() or changing graph execution state outside this API. */
int    volvoxai_engine_forward_incremental(void);
/* Refresh one row after a successful whole-graph/incremental prefill. Since the
 * preceding successful call, every modified row-shaped graph input must differ
 * only at `row`; set_input_raw may submit the whole buffer, but all other rows
 * must retain their prefilled values. For multi-row changes, call
 * incremental_reset() and use forward_incremental() instead. Availability is
 * reported by incremental_row_supported(); provider-owned contexts and native
 * device graphs use their own complete-node execution contracts. */
int    volvoxai_engine_forward_incremental_row(int row);
/* A lane holding no request. It still occupies a row of the dense batch, so
 * every operand keeps its shape; nothing is written back for it. */
#define VOLVOXAI_DECODE_LANE_PARKED (-1)
/* Refresh one row of every lane of a `[lanes,S,...]` batch, where
 * `positions[lane]` is that lane's row and lanes may sit at different
 * positions. `lanes` is declared here rather than read from a shape, because
 * the leading extent of a sequence-major activation is its token count and
 * inferring from it would run an S-lane step in a one-lane context.
 *
 * The per-row contract of forward_incremental_row() applies per lane. Refused
 * when the graph contains an operator whose row path does not stage lanes; the
 * caller falls back to forward_incremental(). */
int    volvoxai_engine_forward_incremental_rows(const int* positions, int lanes);
int    volvoxai_engine_incremental_row_supported(void);
void   volvoxai_engine_incremental_reset(void);
VolvoxAIDecodeSession* volvoxai_engine_decode_session_create(
    const VolvoxAIDecodeSessionOptions* options);
VolvoxAIDecodeMode volvoxai_engine_decode_session_mode(
    const VolvoxAIDecodeSession* session);
VolvoxAIDecodeMode volvoxai_engine_decode_session_last_execution_mode(
    const VolvoxAIDecodeSession* session);
int    volvoxai_engine_decode_session_prefilled(const VolvoxAIDecodeSession* session);
int    volvoxai_engine_decode_session_prefill(VolvoxAIDecodeSession* session);
/* position >= 1 selects row mode when negotiated; -1 uses dependency mode. */
int    volvoxai_engine_decode_session_step(VolvoxAIDecodeSession* session, int position);
int    volvoxai_engine_decode_session_step_rows(VolvoxAIDecodeSession* session,
                                               const int* positions, int lanes);
int    volvoxai_engine_decode_session_reset(VolvoxAIDecodeSession* session);
void   volvoxai_engine_decode_session_destroy(VolvoxAIDecodeSession* session);
int    volvoxai_engine_forward_prefix(int row_count);
int    volvoxai_engine_forward_row(int row);
const float* volvoxai_engine_tensor_row_f32(const char* name, int row, int* count);
void   volvoxai_engine_profile_reset(void);                       // reset debug per-op timing aggregation
void   volvoxai_engine_profile_report(void);                      // print debug per-op timing aggregation
void   volvoxai_engine_shutdown(void);

// Convenience single-pass wrapper (loads input file, runs once, writes output file).
int volvoxai_engine_run(const char* graph_path, const char* weights_path,
               const char* input_path, const char* output_path);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_ENGINE_CORE_H */
