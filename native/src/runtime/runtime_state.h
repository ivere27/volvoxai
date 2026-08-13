#ifndef VOLVOXAI_RUNTIME_STATE_H
#define VOLVOXAI_RUNTIME_STATE_H

#include "cJSON.h"
#include "decode_row_set.h"
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

/* I8/U8 activation storage owns its real-value mapping here. */
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

/* Longest paged-tensor set a context binds: the attention K and V operands of
 * every attention node in one decoder. */
#define VX_PAGED_BINDING_MAX_TENSORS 16

/*
 * One lane of one paged KV cache, bound to this context.
 *
 * Context state rather than a global, so two contexts paging independently
 * cannot observe each other's page tables. The cache is borrowed; the caller
 * that bound it outlives the binding.
 */
typedef struct {
    struct VxPagedKVCache* cache;
    int lane;
    int bound;
    int name_count;
    char names[VX_PAGED_BINDING_MAX_TENSORS][128];
    /* Two gather buffers, one per attention operand, grown geometrically and
     * never returned: a decode loop's prefix grows by one token per generated
     * token, so an exact-fit buffer would reallocate once per token.
     *
     * Bytes rather than floats: a W8A8 pool gathers int8 rows through the same
     * arithmetic, and the element type only ever decided how wide a row is. */
    unsigned char* gather[2];
    size_t gather_bytes[2];
} VxPagedBindingState;

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
    /* Partially resident weight bank read by this node: global slot id ->
     * staged row, VX_MOE_SLOT_ABSENT where the context did not materialize it.
     * NULL/0 means the bank is fully resident and ids are already rows. */
    const uint32_t* resident_slot_rows;
    uint32_t resident_slot_domain;
} Node;

/* Parsed once from the safetensors-backed affine descriptor table for an
 * explicit byte QLinear node. The aligned bias copy also avoids
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

/* Canonical byte QConv2D metadata.  The direct CPU path accepts only
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
    /* OHWI transposed once for the narrow-input AVX2 path, which cannot read
     * eight adjacent output channels contiguously out of OHWI. */
    void* small_c_packed_weight;
} QConv2DMetadata;

/* Canonical byte QEmbedding metadata. Token IDs remain conventional
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

/* Load-time bank selection is part of one private engine snapshot. Nodes keep
 * borrowed pointers into these tables after the requested rows are staged. */
typedef struct VxBankResidencyState {
    char bank[128];
    /* Global-id -> staged row, VX_MOE_SLOT_ABSENT where not resident. */
    uint32_t* slot_rows;
    uint32_t slot_domain;
    uint32_t staged_rows;
    int applied;
} VxBankResidencyState;

enum { VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY = 4 };

/* A resolved plan is context-local because VxEngineState is owned by exactly
 * one public execution context.  Topology stays in Node/T; cache entries keep
 * only the concrete byte layout for an exact logical input signature. */
typedef struct {
    char* signature;
    int tensor_count;
    int* tensor_indices;
    size_t* offsets;
    size_t* tensor_bytes;
    int* tensor_shapes; /* [rank, axis0, ..., axis7] per tensor */
    size_t arena_bytes;
    size_t logical_bytes;
    uint64_t last_use;
    uint64_t model_generation;
} VxDynamicShapePlan;

/* Independently packed from pointwise tensor maxima and immutable topology
 * lifetimes. Concrete best-fit packing is not monotone as tensor sizes shrink,
 * so every concrete native plan projects onto these fixed offsets instead. */
typedef struct {
    int tensor_count;
    int* tensor_indices;
    size_t* offsets;
    size_t* tensor_bytes;
    int physical_span_count;
    size_t* physical_offsets;
    size_t* physical_capacities;
    size_t arena_bytes;
    /* Maximum backend-private F32 statistics storage for the quantized norm
     * kernels. These buffers are not logical tensors, so activation spans do
     * not account for them. Domain reservation must publish them atomically
     * with the physical activation layout. */
    size_t qgroupnorm_stats_bytes;
    size_t qlayernorm_stats_bytes;
    uint64_t model_generation;
    int configured;
} VxDynamicShapeMaximumLayout;

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
    /* Set only for compiled-model contexts. Their file table is a shallow
     * immutable view retained by the compiled weight store. */
    int weight_files_borrowed;
    cJSON* graph_root;
    char first_input[128];
    int loaded;
    int weight_caches_dirty;
    VxBankResidencyState* bank_residency;
    size_t bank_residency_count;
    size_t bank_residency_capacity;

    int backend;
    int use_vulkan;
    int use_opengl;
    int use_metal;
    int use_cuda;

    VxPagedBindingState paged;

    int active_row;
    int prefix_rows;
    long prefix_row_capacity;
    int execution_row;
    /*
     * The lanes a decode step declares, and where each one writes.
     *
     * Zero for every ordinary forward and for the scalar `active_row` path, so
     * a context that never asks for a batch is untouched. Above one it is what
     * the *caller* declared, never what a shape suggested: the leading extent
     * of `[S,1,D]` is S, so inferring the lane count from a shape compiles an
     * S-lane pipeline for a one-lane context. `decode_rows` is only meaningful
     * while `decode_lanes` exceeds one.
     */
    int decode_lanes;
    VxDecodeRowSet decode_rows;
    /*
     * Scratch for one step's staged rows, grown on demand and never shrunk.
     *
     * A batch's write rows are `lane * S + position[lane]`, which is contiguous
     * only when the batch has one lane, so a kernel that wants `lanes` adjacent
     * rows needs them copied somewhere first. Per context because two contexts
     * may decode at once; kept across steps because the size is a function of
     * the model, not of the step.
     */
    unsigned char* decode_stage;
    size_t decode_stage_bytes;
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

    VxDynamicShapePlan dynamic_shape_plans[
        VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY];
    VxDynamicShapeMaximumLayout dynamic_shape_maximum_layout;
    void* dynamic_shape_reserved_arena;
    uint64_t dynamic_shape_plan_clock;
    size_t dynamic_arena_capacity_bytes;
    size_t dynamic_arena_current_bytes;
    size_t dynamic_arena_high_water_bytes;
    uint64_t dynamic_arena_grow_count;
    uint64_t dynamic_resource_generation;
    int dynamic_arena_active;

    /* Set only by the public built-in compiler after graph-wide native-GPU
     * value-origin proof succeeds. Public I32/F32 sources are then preflighted
     * before each binding commit, so GPU wrappers must not reread potentially
     * stale host mirrors of device-produced values. Raw private-engine users
     * leave this clear and retain the legacy host-value validation. */
    int bounded_gpu_value_domain_proven;

    /* One audited transient workspace shared sequentially by CPU kernels in
     * this execution context. Its immutable bound is derived from the model's
     * complete declared shape domain before the context is created. */
    void* cpu_typed_workspace;
    size_t cpu_typed_workspace_bound_bytes;
    size_t cpu_typed_workspace_capacity_bytes;
    int cpu_typed_workspace_configured;

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
    /* Spatial convolution weights repacked to [oc/16][tap][ic][16]. In HWIO the
     * igemm inner loop steps `out_c * 4` bytes per input channel — 1280 at 320
     * channels, which is 3.2 cache lines per 4K page across 100 pages, so no
     * prefetch stream forms and the L1 dTLB thrashes. Packed, the same loop
     * advances one full cache line per step. */
    float* conv_f32_igemm_pack[MAXN];
    long conv_f32_igemm_pack_cap[MAXN];
    const float* conv_f32_igemm_pack_src[MAXN];
    int conv_pw_gemm_enabled;
    /* The packed F32 GEMM is measurably slower than the unpacked kernel at
     * this model's shapes, so the choice is a switch rather than a constant.
     * See VOLVOX_F32_GEMM_PACKED in runtime_state.c. */
    int gemm_f32_packed_enabled;
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
    /* Exact concrete activation signature mixed with seed and counter. */
    uint32_t native_training_shape_hash;
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
