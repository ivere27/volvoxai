#ifndef VOLVOXAI_RUNTIME_STATE_H
#define VOLVOXAI_RUNTIME_STATE_H

/* Backend availability decides one field below, so the state layout must be
 * settled the same way in every translation unit that sees this header. */
#include "backend_config.h"
#include "cJSON.h"
#include "compiled_weight_resources.h"
#include "decode_row_set.h"
#include "engine_core.h"
#include "gemm_f32.h"
#include "safetensors.h"
#include "thread_pool.h"
#include "volvoxai_enums.h"
#include "generated/operator_param_ids.h"
#include "generated/operator_vocabulary.h"

#include "vx_thread.h"
#include <stddef.h>
#include <stdint.h>

#ifndef VOLVOXAI_ENABLE_TRAINING
#define VOLVOXAI_ENABLE_TRAINING 0
#endif

#define MAX_WEIGHT_FILES 32

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

typedef struct {
    char key[24];  /* Document spelling, including arbitrary variadic ports. */
    char name[128];
    int tensor_index;
    VxPortKind port;  /* Fixed role resolved when the reference is created. */
} Ref;

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
    char (*names)[128];
    /* Two gather buffers, one per attention operand, grown geometrically and
     * never returned: a decode loop's prefix grows by one token per generated
     * token, so an exact-fit buffer would reallocate once per token.
     *
     * Bytes rather than floats: a W8A8 pool gathers int8 rows through the same
     * arithmetic, and the element type only ever decided how wide a row is. */
    unsigned char* gather[2];
    size_t gather_bytes[2];
} VxPagedBindingState;

typedef enum {
    VX_CONV_WEIGHT_LAYOUT_INVALID = 0,
    VX_CONV_WEIGHT_LAYOUT_HWIO,
    VX_CONV_WEIGHT_LAYOUT_HWCM,
    VX_CONV_WEIGHT_LAYOUT_OHWI,
    VX_CONV_WEIGHT_LAYOUT_1HWO,
    VX_CONV_WEIGHT_LAYOUT_1HWM,
} VxConvWeightLayout;

typedef enum {
    VX_DATA_LAYOUT_INVALID = 0,
    VX_DATA_LAYOUT_NHWC,
} VxDataLayout;

/*
 * Graph parameters are decoded exactly once, when a Node is compiled.  The
 * raw cJSON object remains attached to Node for graph persistence and patch
 * transactions, but execution code reads this fixed, typed cache only.
 *
 * The key, value-kind and symbol numbers come from the internal operator
 * parameter proto registry.  A fixed numeric key is deliberately used instead
 * of a string-keyed map so a forward pass never walks a cJSON linked list or
 * compares parameter names.
 */
typedef enum {
    VX_NODE_ARRAY_STRIDE = 0,
    VX_NODE_ARRAY_DILATION,
    VX_NODE_ARRAY_PADDING,
    VX_NODE_ARRAY_KERNEL,
    VX_NODE_ARRAY_KERNEL_SIZE,
    VX_NODE_ARRAY_PADS,
    VX_NODE_ARRAY_PERM,
    VX_NODE_ARRAY_STARTS,
    VX_NODE_ARRAY_ENDS,
    VX_NODE_ARRAY_AXES,
    VX_NODE_ARRAY_STEPS,
    VX_NODE_ARRAY_SHAPE,
    VX_NODE_ARRAY_SPLIT,
    VX_NODE_ARRAY_SIZE,
    VX_NODE_ARRAY_COUNT
} VxNodeParamArraySlot;

typedef struct {
    uint8_t kind;
    uint16_t symbol;
    int count;
    union {
        int32_t i32;
        uint32_t u32;
        float f32;
    } value;
} VxCachedNodeParam;

typedef struct {
    VxCachedNodeParam cache[VX_NODE_PARAM_COUNT];
    int32_t* arrays[VX_NODE_ARRAY_COUNT];
    size_t array_bytes[VX_NODE_ARRAY_COUNT];
    uint16_t parameter_count;
    uint16_t cached_parameter_count;
    int axis;
    int keepdims;
    int select_last_index;
    float min_val;
    float max_val;
    int min_val_i32;
    int max_val_i32;
    int has_min_val;
    int has_max_val;
    int relu;
    int groups;
    int num_groups;
    int d_model;
    int heads;
    int causal;
    int top_k;
    int num_experts;
    int normalize;
    float temperature;
    float attention_dropout_probability;
    uint32_t attention_dropout_seed;
    int transB;
    float alpha;
    float beta;
    float epsilon;
    float eps;
    float scale;
    int has_scale;
    int sigmoid;
    int weight_only;
    int weight_only_valid;
    int data_layout;
    int weight_layout;
    uint32_t strict_num_groups;
    uint32_t strict_heads;
    int qgroupnorm_params_valid;
    int qsdpa_params_valid;
    int qargmax_params_valid;
    int perm[8];
    int perm_rank;
    int has_perm;
    int transpose_params_valid;
    int starts[8];
    int ends[8];
    int axes[8];
    int steps[8];
    int slice_rank;
    int has_slice;
    int pads[4];
    int has_pads;
    int kernel_size[2];
    int stride[2];
    int dilation[2];
    /* 0 = not parsed, 1 = valid cache, -1 = invalid typed parameters. */
    int parsed;
} VxNodeParams;

typedef struct {
    VxOperatorKind operator_kind;
    Ref* ins;
    int nin;
    int input_capacity;
    Ref* outs;
    int nout;
    int output_capacity;
    int dependency_output_count;
    int primary_output_index;
    char out[128];
    cJSON* params;
    int owns_params;
    int disabled;
    int fuse_relu6;
    int skip;
    VxNodeParams parsed_params;
    /* Partially resident weight bank read by this node: global slot id ->
     * staged row, VX_MOE_SLOT_ABSENT where the context did not materialize it.
     * NULL/0 means the bank is fully resident and ids are already rows. */
    const uint32_t* resident_slot_rows;
    uint32_t resident_slot_domain;
} Node;

/* Graph nodes own their variable port and typed-parameter storage. Temporary
 * execution projections may borrow a Node; only its owner disposes it. */
int vx_node_reserve_refs(Node* node, int inputs, int outputs);
void vx_node_clear_param_cache(Node* node);
void vx_node_dispose(Node* node);

static inline const VxCachedNodeParam* vx_node_param(
        const Node* node, VxNodeParamKey key) {
    if (!node || (unsigned)key >= (unsigned)VX_NODE_PARAM_COUNT) return NULL;
    return &node->parsed_params.cache[key];
}

static inline int vx_node_param_has(const Node* node, VxNodeParamKey key) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    return value && value->kind != VX_NODE_PARAM_ABSENT &&
        value->kind != VX_NODE_PARAM_NULL;
}

static inline int vx_node_param_i32(const Node* node, VxNodeParamKey key,
                                    int fallback) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!value) return fallback;
    if (value->kind == VX_NODE_PARAM_I32) return value->value.i32;
    if (value->kind == VX_NODE_PARAM_BOOL) return value->value.i32 ? 1 : 0;
    return fallback;
}

static inline float vx_node_param_f32(const Node* node, VxNodeParamKey key,
                                      float fallback) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!value) return fallback;
    if (value->kind == VX_NODE_PARAM_F32) return value->value.f32;
    if (value->kind == VX_NODE_PARAM_I32) return (float)value->value.i32;
    return fallback;
}

static inline uint32_t vx_node_param_u32(const Node* node,
                                         VxNodeParamKey key,
                                         uint32_t fallback) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!value) return fallback;
    if (value->kind == VX_NODE_PARAM_U32) return value->value.u32;
    if (value->kind == VX_NODE_PARAM_I32 && value->value.i32 >= 0)
        return (uint32_t)value->value.i32;
    return fallback;
}

static inline int vx_node_param_bool(const Node* node, VxNodeParamKey key,
                                     int fallback) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!value) return fallback;
    if (value->kind == VX_NODE_PARAM_BOOL || value->kind == VX_NODE_PARAM_I32)
        return value->value.i32 != 0;
    return fallback;
}

static inline VxNodeParamSymbol vx_node_param_symbol(
        const Node* node, VxNodeParamKey key, VxNodeParamSymbol fallback) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (!value || value->kind != VX_NODE_PARAM_SYMBOL) return fallback;
    return (VxNodeParamSymbol)value->symbol;
}

static inline const int32_t* vx_node_param_array(
        const Node* node, VxNodeParamKey key, VxNodeParamArraySlot slot,
        int* count) {
    const VxCachedNodeParam* value = vx_node_param(node, key);
    if (count) *count = 0;
    if (!node || !value || value->kind != VX_NODE_PARAM_I32_ARRAY ||
        (unsigned)slot >= (unsigned)VX_NODE_ARRAY_COUNT) return NULL;
    if (count) *count = (int)value->count;
    return node->parsed_params.arrays[slot];
}

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
    const unsigned char* row_probe_tensors;
    size_t node_capacity;
    int valid;
} VxIncrementalPlan;

/* Execution-quiescent scratch sized with the graph metadata block.  Row
 * compatibility and hybrid planning share this context-owned storage instead
 * of allocating after the context's resident bound has been accepted. */
typedef struct {
    unsigned char* compatibility_dirty;
    unsigned char* hybrid_tensors;
    unsigned char* hybrid_nodes;
    unsigned char* hybrid_boundary_inputs;
    size_t* hybrid_prefix_bytes;
    size_t tensor_capacity;
    size_t node_capacity;
} VxIncrementalScratch;

typedef enum {
    VX_ENGINE_RESULT_OK = 0,
    VX_ENGINE_RESULT_ERROR = -1,
    VX_ENGINE_RESULT_OUT_OF_MEMORY = -2
} VxEngineResult;

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
    uint64_t metadata_bytes;
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
#include "profiling.h"

typedef struct VxEngineState {
    struct VxTraceScope* profiling;
    VxMemoryObserver memory_observer;
    void (*memory_observer_clear)(VxMemoryObserver*);
    T* tensors;
    int tensor_count;
    size_t tensor_capacity;
    Node* nodes;
    int node_count;
    size_t node_capacity;
    QLinearMetadata* qlinear_metadata;
    QConv2DMetadata* qconv_metadata;
    QEmbeddingMetadata* qembedding_metadata;
    void* graph_metadata_storage;
    size_t graph_metadata_bytes;
    size_t graph_metadata_transaction_peak_bytes;

    char* weight_blob;
    SafetensorsFile weight_files[MAX_WEIGHT_FILES];
    char weight_paths[MAX_WEIGHT_FILES][4096];
    int weight_file_count;
    /* Set only for compiled-model contexts. Their file table is a shallow
     * immutable view retained by the compiled weight store. */
    int weight_files_borrowed;
    const VxCompiledCpuWeightStore* compiled_cpu_weights;
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
#if VOLVOXAI_ENABLE_WEBGPU
    /* Declared only where the backend is compiled in: a native build has no
     * WebGPU and must keep the layout it had before this backend existed. */
    int use_webgpu;
    void* webgpu_state;
    int webgpu_validation;
#endif

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

    float** kcache;
    float** vcache;
    float** qwcache;
    float** f16wcache;
    float** f16bcache;
    float** conv_wcache;
    void** q8wcache;
    uint32_t* q8wcache_bytes;
    int* concat_sigmoid_fuse;
    GraphOptStats graph_opt_stats;
    GraphNodeFusion* node_fusion;

    int* tensor_name_index;
    size_t tensor_name_index_capacity;
    int tensor_name_index_count;
    void** retired_f16_storage;
    int retired_f16_storage_count;

    void** arena_buffers;
    int arena_buffer_count;
    size_t arena_allocated_bytes;
    int* arena_tensor_indices;
    int arena_tensor_count;

    VxDynamicShapePlan* dynamic_shape_plans;
    uint32_t dynamic_shape_plan_capacity;
    uint64_t dynamic_shape_plan_metadata_bytes;
    uint64_t dynamic_shape_cache_oversize_skips;
    uint64_t dynamic_shape_cache_bypasses;
    VolvoxAIEngineShapePolicy shape_policy;
    uint64_t bootstrap_activation_bytes;
    int activation_budget_exceeded;
    VxDynamicShapeMaximumLayout dynamic_shape_maximum_layout;
    /* Immutable model-owned topology definition/plan borrowed only by public
     * CPU contexts. The activation core consumes these bytes while this
     * context's retained CompiledModel keeps the Model alive. */
    const uint8_t* portable_graph_plan_request_v1;
    uint32_t portable_graph_plan_request_v1_bytes;
    const uint8_t* portable_graph_plan_v1;
    uint32_t portable_graph_plan_v1_bytes;
    int portable_cpu_activation_plan_enabled;
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

    char (*removed_graph_outputs)[128];
    int removed_graph_output_count;
    T* graph_patch_reused_old_tensors;
    int* graph_patch_reused_indices;
    int graph_patch_reused_count;

    unsigned char* incremental_dirty;
    int incremental_cache_valid;
    int incremental_arena_detached;
    int incremental_hybrid_row_active;
    unsigned char* incremental_hybrid_row_nodes;
    /* Exact logical FIXED selection plus any physical fusion owners promoted
     * by native lowering. It is context-owned graph metadata so no per-token
     * allocation escapes the compiled resident domain. */
    unsigned char* incremental_portable_selected_nodes;
    int incremental_portable_selection_active;
    int incremental_hybrid_prepared_row;
    VxIncrementalPlan incremental_plan;
    VxIncrementalScratch incremental_scratch;
    struct VolvoxAIDecodeSession* active_decode_session;

    VxRuntimeNodeState runtime_node;
    int runtime_cpu_dispatch;
    int runtime_cuda_replay_eligible;
    int runtime_cuda_route_tracking;
    uint64_t runtime_cuda_replay_generation;
    int runtime_forward_ok;
    int last_failure_node_index;
    /* Exact per-node route from the most recent forward. Values point to
     * static backend labels owned by the private engine implementation. */
    const char** runtime_route_backend;
    VxBackendRegistry backend_registry;

    float** conv_pwf32_pack;
    float** conv_pwf32_pack_plain;
    float** conv_dw_pw_tmp;
    long* conv_dw_pw_tmp_cap;
    const float*** conv_f32_igemm_indir;
    long* conv_f32_igemm_indir_cap;
    uint64_t* conv_f32_igemm_indir_key;
    float** conv_f32_igemm_zero;
    int* conv_f32_igemm_zero_cap;
    /* Spatial convolution weights repacked to [oc/16][tap][ic][16]. In HWIO the
     * igemm inner loop steps `out_c * 4` bytes per input channel — 1280 at 320
     * channels, which is 3.2 cache lines per 4K page across 100 pages, so no
     * prefetch stream forms and the L1 dTLB thrashes. Packed, the same loop
     * advances one full cache line per step. */
    float** conv_f32_igemm_pack;
    long* conv_f32_igemm_pack_cap;
    const float** conv_f32_igemm_pack_src;
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

    VxMutex adapter_admin_mutex;
    VxMutex model_mutex;
    VxMutex metadata_mutex;
    VxKernelThreadPool* kernel_thread_pool;
    int kernel_thread_pool_owned;
    int kernel_thread_pool_thread_count;
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
    EngineOptimizerState* optimizer_states;
    TrainingAccumulationState training_accumulation;
    struct VolvoxAIAutogradContext* dynamic_autograd_context;
    int dynamic_autograd_forward_capture;
    int dynamic_autograd_forward_backend;
    int native_training_mode;
    uint32_t native_training_counter;
    /* Trainer-owned stream key mixed into packaged-graph Dropout counters.
     * Zero selects the internal counter for direct engine training. */
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
/* Public Runtime children share one CPU resource-domain pool. Passing NULL
 * retains the standalone/private-engine behavior and creates a state-owned
 * pool. A borrowed pool must outlive the state and cannot be reconfigured by
 * engine configure/shutdown. */
int vx_engine_state_init_with_kernel_pool(
    VxEngineState* state,
    VxKernelThreadPool* kernel_thread_pool,
    int thread_count);
void vx_engine_state_deinit(VxEngineState* state);

/* Graph-size-dependent metadata is reserved as one failure-atomic owned
 * block. The byte value includes alignment padding and is therefore the exact
 * retained allocation. The state also records the checked old-plus-candidate
 * transaction peak across successful reserves. */
int vx_engine_state_graph_metadata_bytes_for(size_t tensor_capacity,
                                             size_t node_capacity,
                                             size_t* bytes_out);
/* Include variable node ports and typed parameter arrays in resident admission.
 * The peak also covers their old-plus-candidate transactional replacement. */
int vx_engine_state_graph_metadata_owned_bytes(const VxEngineState* state,
                                               uint64_t* resident, uint64_t* peak);
VxEngineResult vx_engine_state_reserve_graph_metadata(
    VxEngineState* state, size_t tensor_capacity, size_t node_capacity);
void vx_engine_state_release_graph_metadata(VxEngineState* state);


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
