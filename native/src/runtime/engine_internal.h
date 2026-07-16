#ifndef VOLVOX_ENGINE_INTERNAL_H
#define VOLVOX_ENGINE_INTERNAL_H

#include "cJSON.h"
#include "safetensors.h"
#include "backend_config.h"
#include <stddef.h>
#include <stdint.h>

typedef struct VxBackend VxBackend;

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

/* Keep enough metadata capacity for large materialized graphs whose persisted
 * parameters and intermediates exceed the former 1024-entry ceiling. */
#define MAXT 2048
#define MAXN 1024
#define MAXIN 12
#define MAX_WEIGHT_FILES 16

enum { T_F32 = 0, T_I8 = 1, T_U8 = 2, T_I32 = 3, T_F16 = 4 };

/* Physical I8/U8 activation storage owns its real-value mapping here.  This
 * is intentionally separate from the legacy QTensor sidecar, which remains
 * an internal compatibility path for F32 graphs. */
typedef struct {
    int valid;
    float scale;
    int zero_point;
} TensorQuantization;

typedef struct {
    char name[128];
    int shape[8];
    int ndim;
    float* data;
    long numel;
    int owns;
    int dtype;
    size_t elem_size;
    int is_graph_input;
    TensorQuantization quantization;
} T;
typedef struct { char key[24]; char name[128]; } Ref;
typedef struct { char op[40]; Ref ins[MAXIN]; int nin; Ref outs[MAXIN]; int nout; char out[128]; cJSON* params;
                 int owns_params; int disabled; int fuse_relu6; int skip; } Node;
typedef struct {
    signed char* data;
    long cap;
    float scale;
    int zp;
    int has_params;
    int valid;
} QTensor;

/* Parsed once from the JS-compatible `weights_quantization` object for an
 * explicit physical-byte QLinear node. The aligned bias copy also avoids
 * assuming safetensors byte offsets satisfy I32 alignment. */
typedef struct {
    int valid;
    int d_in;
    int d_out;
    int input_dtype;
    int weight_dtype;
    int output_dtype;
    float* weight_scales;
    int32_t* weight_zero_points;
    int32_t* bias;
    void* packed_weight;
    uint32_t packed_weight_bytes;
} QLinearMetadata;

/* Canonical physical-byte QConv2D metadata.  The direct CPU path accepts only
 * NHWC activations and [O,H,W,I/group] OHWI weights; the optional bias has an
 * aligned I32 copy in the accumulator domain. */
typedef struct {
    int valid;
    uint32_t batch;
    uint32_t input_height;
    uint32_t input_width;
    uint32_t input_channels;
    uint32_t output_height;
    uint32_t output_width;
    uint32_t output_channels;
    uint32_t kernel_height;
    uint32_t kernel_width;
    uint32_t input_per_group;
    uint32_t stride_y;
    uint32_t stride_x;
    uint32_t dilation_y;
    uint32_t dilation_x;
    uint32_t padding_top;
    uint32_t padding_left;
    uint32_t padding_bottom;
    uint32_t padding_right;
    uint32_t groups;
    uint32_t relu;
    int input_dtype;
    int weight_dtype;
    int output_dtype;
    float* weight_scales;
    int32_t* weight_zero_points;
    int32_t* bias;
} QConv2DMetadata;

/* Canonical physical-byte QEmbedding metadata. Token IDs remain conventional
 * I32 storage; the rank-2 [vocab, hidden] table owns per-row quantization and
 * the output owns its immutable per-tensor I8/U8 descriptor. */
typedef struct {
    int valid;
    uint32_t vocab;
    uint32_t hidden;
    int weight_dtype;
    int output_dtype;
    float* weight_scales;
    int32_t* weight_zero_points;
} QEmbeddingMetadata;

extern T g_t[MAXT];
extern int g_nt;
extern Node g_n[MAXN];
extern int g_nn;
extern QTensor g_qt[MAXT];
extern QLinearMetadata g_qlinear_meta[MAXN];
extern QConv2DMetadata g_qconv_meta[MAXN];
extern QEmbeddingMetadata g_qembedding_meta[MAXN];
extern char* g_blob;
extern SafetensorsFile g_weight_files[MAX_WEIGHT_FILES];
extern char g_weight_paths[MAX_WEIGHT_FILES][4096];
extern int g_weight_file_count;
extern cJSON* g_cfg_root;
extern char g_first_input[128];
extern int g_loaded;
extern int g_weight_caches_dirty;
extern int g_active_row;
extern int g_prefix_rows;
extern float* g_kcache[MAXN];
extern float* g_vcache[MAXN];
extern float* g_qwcache[MAXN];
extern float* g_f16wcache[MAXN];
extern float* g_f16bcache[MAXN];
extern float* g_conv_wcache[MAXN];
extern void* g_q8wcache[MAXN];
extern uint32_t g_q8wcache_bytes[MAXN];
extern int g_concat_sigmoid_fuse[MAXN];

extern int g_use_vulkan;
extern int g_use_nnapi;
extern int g_use_opengl;
extern int g_use_metal;
extern int g_debug;
extern int g_execution_row;

double volvoxai_engine_now_ms(void);
T* t_find(const char* name);
/* The tensor table is a dense array, while steady-state graph execution needs
 * name resolution to stay independent of the model's tensor count.  Table
 * mutators keep this private hash index synchronized through these hooks. */
void volvoxai_engine_tensor_name_index_invalidate(void);
void volvoxai_engine_tensor_name_index_add(int tensor_index);
void volvoxai_engine_tensor_name_index_rebuild(void);
int volvoxai_engine_tensor_is_removed_output(const char* name);
void materialize_tensor_f32(T* t);
void qt_invalidate_all(void);
void vx_runtime_backend_reset(void);
void vx_runtime_backend_teardown(void);
void vx_runtime_backend_begin_forward(void);
int vx_runtime_backend_end_forward(void);
void vx_runtime_backend_mark_host(const void* host, size_t bytes, int is_weight);
int vx_runtime_backend_sync_host(const void* host, size_t bytes, int is_weight);
int vx_runtime_backend_has_graph(void);
int vx_runtime_backend_stage(const VxBackend* sdk_backend);
int vx_runtime_backend_sdk_execution_eligible(void);
int vx_runtime_backend_sdk_node_eligible(const Node* node);
void vx_decode_session_invalidate_model_locked(void);
int vx_decode_session_active_locked(void);
void vx_decode_session_invalidate_cache_locked(void);
int vx_incremental_hybrid_row_active_locked(void);
void volvoxai_engine_clear_qlinear_metadata(void);
void volvoxai_engine_clear_qconv_metadata(void);
void volvoxai_engine_clear_qembedding_metadata(void);
int vk_sync_host_tensor(T* t);
void vk_mark_owned_tensors_host_dirty(void);
char* read_file(const char* path, long* out_size);
int volvoxai_engine_load_weight_files(const char* const* paths, int count);
int build_graph(const char* config_path);
void volvoxai_engine_free_arena(void);
int volvoxai_engine_prepare_tensor_table_mutation(void);
void volvoxai_engine_finish_tensor_table_mutation(void);
int volvoxai_engine_refresh_weight_caches(void);
void volvoxai_engine_model_lock(void);
void volvoxai_engine_model_unlock(void);
uint64_t volvoxai_engine_model_generation_locked(void);
void volvoxai_engine_model_generation_advance_locked(void);
int volvoxai_engine_adapter_effect_active_locked(void);
int volvoxai_engine_model_route_lease_active(void);
void volvoxai_engine_metadata_lock(void);
void volvoxai_engine_metadata_unlock(void);
#if VOLVOXAI_ENABLE_TRAINING
int volvoxai_engine_training_accumulation_pending(void);
void volvoxai_engine_reset_training_accumulation(void);
#else
static inline int volvoxai_engine_training_accumulation_pending(void) { return 0; }
static inline void volvoxai_engine_reset_training_accumulation(void) {}
#endif
int volvoxai_engine_linear_weight_layout(const char* weight_name, int* d_in, int* d_out, int* out_in);
int volvoxai_engine_forward(void);
int volvoxai_engine_forward_locked(void);
int volvoxai_engine_forward_incremental_locked(void);
int volvoxai_engine_forward_incremental_row_locked(int row);
int volvoxai_engine_tensor_is_model_weight_locked(const char* name);
int volvoxai_engine_tensor_is_internal_companion_locked(const char* name);
int volvoxai_engine_add_model_tensor_raw_locked(const char* name,
                                                const int* shape, int ndim,
                                                int dtype, const void* data,
                                                size_t nbytes);
int volvoxai_engine_remove_model_tensor_locked(const char* name);
#if VOLVOXAI_ENABLE_TRAINING
int volvoxai_engine_apply_tensor_update_f32(const char* name, const float* update, long numel,
                                   int mode, float learning_rate, float beta1, float beta2,
                                   float epsilon, float weight_decay, float max_grad_norm,
                                   long step);
int volvoxai_engine_prepare_tensor_update_f32(const char* name, long numel, int mode);
int volvoxai_engine_apply_tensor_update_f32_locked(const char* name, const float* update, long numel,
                                           int mode, float learning_rate, float beta1, float beta2,
                                           float epsilon, float weight_decay, float max_grad_norm,
                                           long step);
#endif
int prepack_conv_weights(void);
int run_node(Node* n, int idx, int is_last);
int run_node_cpu_direct(Node* n, int idx, int is_last);
/* Side-effect-free preflight for canonical operators whose CPU implementation
 * can refresh exactly one [1,S,...] row after a device seed. */
int vx_runtime_node_incremental_row_compatible(Node* n, int idx, int row);
void prof_reset(void);
void prof_add_entry(const char* op, double ms);
void prof_report(void);

#endif
