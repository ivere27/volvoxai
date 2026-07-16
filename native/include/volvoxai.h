#ifndef VOLVOXAI_H
#define VOLVOXAI_H

#include <stddef.h>

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

/* Raw tensor APIs use these stable storage dtypes.  They are public so a CLI
 * or embedding application can use set_input_raw/copy_tensor_raw without
 * importing native runtime internals. */
typedef enum {
    VOLVOXAI_DTYPE_F32 = 0,
    VOLVOXAI_DTYPE_I8 = 1,
    VOLVOXAI_DTYPE_U8 = 2,
    VOLVOXAI_DTYPE_I32 = 3,
    VOLVOXAI_DTYPE_F16 = 4
} VolvoxAIDataType;

typedef enum {
    VOLVOXAI_BACKEND_CPU = 0,
    VOLVOXAI_BACKEND_VULKAN = 1,
    VOLVOXAI_BACKEND_OPENGL = 2,
    VOLVOXAI_BACKEND_METAL = 3,
    VOLVOXAI_BACKEND_NNAPI = 4,
    /* Reported by get_options() after name-based SDK selection.  Pass a
     * concrete built-in value to configure(); select custom backends through
     * volvoxai_engine_configure_backend() in volvoxai_backend.h. */
    VOLVOXAI_BACKEND_CUSTOM = 5
} VolvoxAIEngineBackend;

typedef struct {
    VolvoxAIEngineBackend backend;
    int debug;
    int cpu_threads; /* zero keeps the runtime default */
} VolvoxAIEngineOptions;

/* Backend-neutral autoregressive execution. A decode session is created after
 * model initialization, then seeded once after the caller writes all inputs.
 * Later steps rerun the dependency closure; CPU may additionally refresh one
 * decoder row/KV prefix when a positive position is supplied. Ordinary
 * forwards, explicit incremental resets, and graph/weight mutations clear the
 * session's seeded state, so the next session operation must be seed(). */
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
} VolvoxAIDecodeSessionOptions;

#define VOLVOXAI_DECODE_SESSION_OPTIONS_INIT \
    { sizeof(VolvoxAIDecodeSessionOptions), VOLVOXAI_DECODE_ROW_AUTO, 0 }

typedef struct VolvoxAIDecodeSession VolvoxAIDecodeSession;

// Native graph engine: parses a Volvox blueprint (config.json + safetensors),
// builds the graph once, and runs it node-by-node on the CPU/GPU kernels.
// Training-specific declarations live in volvoxai_training.h.
//
// Configure optional device/debug policy before loading a graph. Calling this
// while a graph is loaded is rejected; explicit device requests never silently
// fall back to CPU. A rejected option set or failed backend initialization
// leaves the previously configured options/backend unchanged.
int    volvoxai_engine_configure(const VolvoxAIEngineOptions* options);
int    volvoxai_engine_get_options(VolvoxAIEngineOptions* options);
const char* volvoxai_engine_backend_name(void);
int    volvoxai_engine_set_debug(int enabled);
int    volvoxai_engine_debug(void);
/* Select an optional row for row-aware execution/output paths. -1 restores
 * whole-tensor execution. This is graph scheduling state, not token policy. */
int    volvoxai_engine_set_execution_row(int row);
int    volvoxai_engine_execution_row(void);
/* Graph-declared interface names in config order. Returned strings remain
 * owned by the engine and are valid until shutdown or the next init. */
int    volvoxai_engine_graph_input_count(void);
const char* volvoxai_engine_graph_input_name(int index);
int    volvoxai_engine_graph_output_count(void);
const char* volvoxai_engine_graph_output_name(int index);

/* weights_path may be NULL for a graph whose declared dependencies are fully
 * satisfied by graph inputs and node outputs. Strict graph preflight still
 * rejects unresolved weight names. */
int    volvoxai_engine_init(const char* config_path, const char* weights_path);
/* weight_file_count may be zero when weight_file_paths is NULL. */
int    volvoxai_engine_init_with_weight_files(const char* config_path,
                                              const char* const* weight_file_paths,
                                              int weight_file_count);
/* Mutable view of an F32 graph input; typed inputs return NULL. */
float* volvoxai_engine_input_ptr(const char* name, long* numel);
int    volvoxai_engine_set_input_raw(const char* name, int dtype, const void* data, size_t nbytes);
/* Copy real F32 values into an F32 graph input, or quantize them into a
 * per-tensor I8/U8 graph input using its declared scale and zero point. */
int    volvoxai_engine_set_input_f32(const char* name, const float* data, long numel);
int    volvoxai_engine_is_graph_input(const char* name);
int    volvoxai_engine_is_model_weight(const char* name);
int    volvoxai_engine_copy_tensor_f32(const char* name, float* out, long numel);
int    volvoxai_engine_copy_tensor_raw(const char* name, void* out, size_t nbytes);
int    volvoxai_engine_tensor_info(const char* name, long* numel, int* shape, int* ndim);
int    volvoxai_engine_tensor_info_ex(const char* name, long* numel, int* shape, int* ndim, int* dtype, size_t* elem_size);
int    volvoxai_engine_set_tensor_f32(const char* name, const float* data, long numel);
int    volvoxai_engine_set_tensor_raw(const char* name, int dtype, const void* data, size_t nbytes);
int    volvoxai_engine_add_model_tensor_raw(const char* name, const int* shape, int ndim,
                                            int dtype, const void* data, size_t nbytes);
int    volvoxai_engine_remove_model_tensor(const char* name);
int    volvoxai_engine_prepare_tensor_table_mutation(void);
void   volvoxai_engine_finish_tensor_table_mutation(void);
int    volvoxai_engine_tensor_weight_file_index(const char* name);
int    volvoxai_engine_linear_weight_layout(const char* weight_name, int* d_in, int* d_out, int* out_in);
int    volvoxai_engine_save_weight_file(int weight_file_index, const char* path);
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
int    volvoxai_engine_save_config(const char* path);
int    volvoxai_engine_forward(void);                              // run the whole graph (all rows)
/* Opt-in dependency-aware forward. The first call runs the complete graph;
 * later calls execute only nodes reachable from graph inputs written through
 * set_input_raw() since the previous incremental call. Cached intermediates
 * remain owned by the engine. Call incremental_reset() after mutating an input
 * through input_ptr() or changing graph execution state outside this API. */
int    volvoxai_engine_forward_incremental(void);
/* Refresh one row after a successful whole-graph/incremental seed. Since the
 * preceding successful call, every modified row-shaped graph input must differ
 * only at `row`; set_input_raw may submit the whole buffer, but all other rows
 * must retain their seeded values. For multi-row changes, call
 * incremental_reset() and use forward_incremental() instead. Availability is
 * reported by incremental_row_supported(); public V1 backends and native
 * device-graph backends use complete-node dependency execution instead. */
int    volvoxai_engine_forward_incremental_row(int row);
int    volvoxai_engine_incremental_row_supported(void);
void   volvoxai_engine_incremental_reset(void);
VolvoxAIDecodeSession* volvoxai_engine_decode_session_create(
    const VolvoxAIDecodeSessionOptions* options);
VolvoxAIDecodeMode volvoxai_engine_decode_session_mode(
    const VolvoxAIDecodeSession* session);
VolvoxAIDecodeMode volvoxai_engine_decode_session_last_execution_mode(
    const VolvoxAIDecodeSession* session);
int    volvoxai_engine_decode_session_seeded(const VolvoxAIDecodeSession* session);
int    volvoxai_engine_decode_session_seed(VolvoxAIDecodeSession* session);
/* position >= 1 selects row mode when negotiated; -1 uses dependency mode. */
int    volvoxai_engine_decode_session_step(VolvoxAIDecodeSession* session, int position);
int    volvoxai_engine_decode_session_reset(VolvoxAIDecodeSession* session);
void   volvoxai_engine_decode_session_destroy(VolvoxAIDecodeSession* session);
int    volvoxai_engine_forward_prefix(int row_count);
int    volvoxai_engine_forward_row(int row);
const float* volvoxai_engine_tensor_row_f32(const char* name, int row, int* count);
void   volvoxai_engine_profile_reset(void);                       // reset debug per-op timing aggregation
void   volvoxai_engine_profile_report(void);                      // print debug per-op timing aggregation
void   volvoxai_engine_shutdown(void);

// Convenience single-pass wrapper (loads input file, runs once, writes output file).
int volvoxai_engine_run(const char* config_path, const char* weights_path,
               const char* input_path, const char* output_path);

#ifdef __cplusplus
}
#endif

#endif
