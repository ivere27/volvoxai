#ifndef VOLVOXAI_RUNTIME_STATE_H
#define VOLVOXAI_RUNTIME_STATE_H

#include "cJSON.h"
#include "gemm_f32.h"
#include "safetensors.h"
#include "thread_pool.h"
#include "volvoxai_enums.h"

#include <pthread.h>
#include <stddef.h>
#include <stdint.h>

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

/* Keep enough metadata capacity for large materialized graphs whose persisted
 * parameters and intermediates exceed the former 1024-entry ceiling. */
#define MAXT 2048
#define MAXN 1024
#define MAXIN 12
#define MAX_WEIGHT_FILES 16
#define TENSOR_NAME_INDEX_CAPACITY (MAXT * 2)

#if VOLVOXAI_ENABLE_TRAINING
#include "../training/training_state.h"
#endif

enum {
    T_F32 = VX_DTYPE_F32,
    T_I8 = VX_DTYPE_I8,
    T_U8 = VX_DTYPE_U8,
    T_I32 = VX_DTYPE_I32,
    T_F16 = VX_DTYPE_F16
};

/* Physical I8/U8 activation storage owns its real-value mapping here. */
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
    VxDataType dtype;
    size_t elem_size;
    int is_graph_input;
    TensorQuantization quantization;
} T;

typedef struct { char key[24]; char name[128]; } Ref;

typedef struct {
    char op[40];
    Ref ins[MAXIN];
    int nin;
    Ref outs[MAXIN];
    int nout;
    char out[128];
    cJSON* params;
    int owns_params;
    int disabled;
    int fuse_relu6;
    int skip;
} Node;

/* Parsed once from the safetensors-backed affine descriptor table for an
 * explicit physical-byte QLinear node. The aligned bias copy also avoids
 * assuming safetensors byte offsets satisfy I32 alignment. */
typedef struct {
    int valid;
    int d_in;
    int d_out;
    VxDataType input_dtype;
    VxDataType weight_dtype;
    VxDataType output_dtype;
    float* weight_scales;
    int32_t* weight_zero_points;
    int32_t* bias;
    int weight_zero_all_zero;
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
    VxDataType input_dtype;
    VxDataType weight_dtype;
    VxDataType output_dtype;
    float* weight_scales;
    int32_t* weight_zero_points;
    int32_t* bias;
    void* packed_weight;
    uint32_t packed_weight_bytes;
} QConv2DMetadata;

/* Canonical physical-byte QEmbedding metadata. Token IDs remain conventional
 * I32 storage; the rank-2 [vocab, hidden] table owns per-row quantization and
 * the output owns its immutable per-tensor I8/U8 descriptor. */
typedef struct {
    int valid;
    uint32_t vocab;
    uint32_t hidden;
    VxDataType weight_dtype;
    VxDataType output_dtype;
    float* weight_scales;
    int32_t* weight_zero_points;
} QEmbeddingMetadata;

typedef enum {
    GRAPH_FUSION_CONV_BIAS = 1,
    GRAPH_FUSION_CONV_BIAS_ACT,
    GRAPH_FUSION_CONV_BIAS_HARDSWISH,
    GRAPH_FUSION_CONV_ADD,
    GRAPH_FUSION_CONV_ADD_ACT,
    GRAPH_FUSION_DEPTHWISE_ACT,
    GRAPH_FUSION_DEPTHWISE_POINTWISE,
    GRAPH_FUSION_DECONV_ACT,
    GRAPH_FUSION_CONV1D_ACT,
    GRAPH_FUSION_CONV3D_ACT,
    GRAPH_FUSION_MATMUL_BIAS,
    GRAPH_FUSION_MATMUL_BIAS_ACT,
    GRAPH_FUSION_MATMUL_ADD,
    GRAPH_FUSION_MATMUL_SOFTMAX,
    GRAPH_FUSION_ATTENTION_SCALE_MASK_SOFTMAX,
    GRAPH_FUSION_SWIGLU,
    GRAPH_FUSION_ADD_ACT,
    GRAPH_FUSION_MUL_ADD,
    GRAPH_FUSION_MUL_SIGMOID,
    GRAPH_FUSION_CHAINED_ELEMENTWISE,
    GRAPH_FUSION_ADD_CLAMP,
    GRAPH_FUSION_SOFTMAX_DECOMPOSED,
    GRAPH_FUSION_LAYERNORM_MATMUL,
    GRAPH_FUSION_RMSNORM_MUL,
    GRAPH_FUSION_INSTANCENORM_ACT,
    GRAPH_FUSION_GLOBAL_AVGPOOL_FLATTEN,
    GRAPH_FUSION_MAXPOOL_ACT,
    GRAPH_FUSION_ARGMAX_GATHER,
    GRAPH_FUSION_SPLIT_MATMUL,
    GRAPH_FUSION_QDQ_REQUANT,
    GRAPH_FUSION_CONCAT_SIGMOID_INTERNAL,
    GRAPH_FUSION_ALIAS_PASSTHROUGH_INTERNAL
} GraphFusionId;

typedef struct {
    int relu6;
    int alias;
    int concat_sigmoid;
    int conv_add;
    int depthwise_pointwise;
    int chained_add;
    int skipped;
} GraphOptStats;

typedef struct {
    GraphFusionId id;
    unsigned backend_mask;
    int peer_idx;
    char source_tensor[128];
    char aux_tensor[128];
} GraphNodeFusion;

typedef struct {
    uint64_t generation;
    int tensor_count;
    int node_count;
    int input_indices[MAXN][MAXIN];
    int output_indices[MAXN][MAXIN + 1];
    unsigned char input_counts[MAXN];
    unsigned char output_counts[MAXN];
    int valid;
} VxIncrementalPlan;

typedef struct {
    char op[40];
    double ms;
    int count;
} ProfEntry;

typedef struct {
    int active;
    int idx;
    int is_last;
    int adapter_linear;
    int native_physical_qlinear;
    int native_physical_qbatch_matmul;
    int native_physical_qembedding;
    int native_physical_qconv;
    int native_physical_qadd;
    int native_physical_qsilu;
    int native_physical_qgelu;
    int native_physical_qgroupnorm;
    int native_physical_qlayernorm;
    int native_physical_qsdpa;
    int native_physical_qargmax;
    int native_physical_qmaskedmean;
    int native_physical_requantize;
    int native_physical_quantize;
    int native_physical_dequantize;
    int native_physical_shape;
    const char** node_backend;
} VxRuntimeNodeState;

enum { VX_BACKEND_MAX = 16 };
typedef struct VxBackendRegistry {
    const struct VxBackend* backends[VX_BACKEND_MAX];
    unsigned char initialized[VX_BACKEND_MAX];
    size_t count;
    size_t cpu_index;
    int ready;
} VxBackendRegistry;

typedef struct {
    T* tensor;
    unsigned char* backup;
    unsigned char* merged;
    size_t nbytes;
} EngineMergedAdapterWeight;

/* Context-owned state used by the private native graph implementation. */
typedef struct VxEngineState {
    T tensors[MAXT];
    int tensor_count;
    Node nodes[MAXN];
    int node_count;
    QLinearMetadata qlinear_metadata[MAXN];
    QConv2DMetadata qconv_metadata[MAXN];
    QEmbeddingMetadata qembedding_metadata[MAXN];

    char* weight_blob;
    SafetensorsFile weight_files[MAX_WEIGHT_FILES];
    char weight_paths[MAX_WEIGHT_FILES][4096];
    int weight_file_count;
    cJSON* graph_root;
    char first_input[128];
    int loaded;
    int weight_caches_dirty;

    int backend;
    int use_vulkan;
    int use_nnapi;
    int use_opengl;
    int use_metal;
    int use_cuda;

    int active_row;
    int prefix_rows;
    long prefix_row_capacity;
    int execution_row;
    int debug;

    float* kcache[MAXN];
    float* vcache[MAXN];
    float* qwcache[MAXN];
    float* f16wcache[MAXN];
    float* f16bcache[MAXN];
    float* conv_wcache[MAXN];
    void* q8wcache[MAXN];
    uint32_t q8wcache_bytes[MAXN];
    int concat_sigmoid_fuse[MAXN];
    GraphOptStats graph_opt_stats;
    GraphNodeFusion node_fusion[MAXN];

    int tensor_name_index[TENSOR_NAME_INDEX_CAPACITY];
    int tensor_name_index_count;
    void* retired_f16_storage[MAXT];
    int retired_f16_storage_count;

    void** arena_buffers;
    int arena_buffer_count;
    int* arena_tensor_indices;
    int arena_tensor_count;

    char removed_graph_outputs[MAXT][128];
    int removed_graph_output_count;
    T graph_patch_reused_old_tensors[MAXT];
    int graph_patch_reused_indices[MAXT];
    int graph_patch_reused_count;

    unsigned char incremental_dirty[MAXT];
    int incremental_cache_valid;
    int incremental_arena_detached;
    int incremental_hybrid_row_active;
    unsigned char incremental_hybrid_row_nodes[MAXN];
    int incremental_hybrid_prepared_row;
    VxIncrementalPlan incremental_plan;
    struct VolvoxAIDecodeSession* active_decode_session;

    ProfEntry profiler_entries[128];
    int profiler_entry_count;

    VxRuntimeNodeState runtime_node;
    int runtime_cpu_dispatch;
    int runtime_cuda_replay_eligible;
    int runtime_cuda_route_tracking;
    uint64_t runtime_cuda_replay_generation;
    int runtime_forward_ok;
    int last_failure_node_index;
    /* Exact per-node route from the most recent forward. Values point to
     * static backend labels owned by the private engine implementation. */
    const char* runtime_route_backend[MAXN];
    VxBackendRegistry backend_registry;

    float* conv_pwf32_pack[MAXN];
    float* conv_pwf32_pack_plain[MAXN];
    float* conv_dw_pw_tmp[MAXN];
    long conv_dw_pw_tmp_cap[MAXN];
    const float** conv_f32_igemm_indir[MAXN];
    long conv_f32_igemm_indir_cap[MAXN];
    uint64_t conv_f32_igemm_indir_key[MAXN];
    float* conv_f32_igemm_zero[MAXN];
    int conv_f32_igemm_zero_cap[MAXN];
    int conv_pw_gemm_enabled;
    VxGemmF32Cache gemm_f32_cache;

    EngineMergedAdapterWeight* merged_adapter_weights;
    int merged_adapter_count;
    char merged_adapter_version[128];
    char pre_merge_active_version[128];

    pthread_mutex_t adapter_admin_mutex;
    pthread_mutex_t model_mutex;
    pthread_mutex_t metadata_mutex;
    VxKernelThreadPool* kernel_thread_pool;
    int cpu_threads;
    uint64_t model_generation;

    /* Backend-private request state. Device backends keep graph residency,
     * command, scratch, descriptor, and training state in these capsules; the
     * generic runtime only owns their lifetime. */
    void* vulkan_context_state;
    void (*vulkan_context_state_destroy)(void* state);
    void* opengl_context_state;
    void (*opengl_context_state_destroy)(void* state);
    void* metal_context_state;
    void (*metal_context_state_destroy)(void* state);
    void* nnapi_context_state;
    void (*nnapi_context_state_destroy)(void* state);
    void* cuda_context_state;
    void (*cuda_context_state_destroy)(void* state);

#if VOLVOXAI_ENABLE_TRAINING
    EngineOptimizerState optimizer_states[MAXT];
    TrainingAccumulationState training_accumulation;
    struct VolvoxAIAutogradContext* dynamic_autograd_context;
    int dynamic_autograd_forward_capture;
    int dynamic_autograd_forward_backend;
    int native_training_mode;
    uint32_t native_training_counter;
    /* Trainer-owned stream key mixed into packaged-graph Dropout counters.
     * Zero selects the internal counter used by non-Trainer tests. */
    uint32_t native_training_rng_seed;
    int required_training_backend;
    int last_training_backend;
    uint32_t gpu_training_dummy[4];
#endif

    void* adapter_registry_state;
    void (*adapter_registry_state_destroy)(void* state);
} VxEngineState;

typedef struct {
    VxEngineState* previous;
    VxEngineState* bound;
    VxKernelThreadPoolScope kernel_pool_scope;
} VxEngineStateScope;

/* Initialize only the state capsule itself. Runtime-owned graph/backend
 * resources must already be released before deinit. */
int vx_engine_state_init(VxEngineState* state);
void vx_engine_state_deinit(VxEngineState* state);

/* A public context binds its state for the duration of each private engine
 * call. NULL means no engine operation is currently scoped on this thread. */
VxEngineState* vx_engine_state_current(void);
VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state);
void vx_engine_state_scope_leave(VxEngineStateScope scope);

/* Adapter route leases are thread-local but tied to the state whose model
 * mutex they retain. They therefore cannot live as a plain state field. */
int vx_engine_state_route_lease_active(void);
int vx_engine_state_route_lease_begin(void);
void vx_engine_state_route_lease_end(void);

#endif
