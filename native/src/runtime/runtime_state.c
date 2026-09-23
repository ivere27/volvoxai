#include "runtime_state.h"
#include "vx_platform.h"

#include <assert.h>
#include <limits.h>
#include <stdlib.h>
#include <string.h>

static int vx_node_grow_refs(Ref** refs, int* capacity, int count) {
    if (count < 0 || (size_t)count > SIZE_MAX / sizeof(Ref)) return -1;
    if (count <= *capacity) return 0;
    Ref* next = realloc(*refs, (size_t)count * sizeof(Ref));
    if (!next) return -1;
    memset(next + *capacity, 0, (size_t)(count - *capacity) * sizeof(Ref));
    *refs = next; *capacity = count;
    return 0;
}

int vx_node_reserve_refs(Node* node, int inputs, int outputs) {
    if (!node || vx_node_grow_refs(&node->ins, &node->input_capacity, inputs) ||
        vx_node_grow_refs(&node->outs, &node->output_capacity, outputs)) return -1;
    return 0;
}

void vx_node_clear_param_cache(Node* node) {
    if (!node) return;
    for (int slot = 0; slot < VX_NODE_ARRAY_COUNT; ++slot)
        free(node->parsed_params.arrays[slot]);
    memset(&node->parsed_params, 0, sizeof(node->parsed_params));
}

void vx_node_dispose(Node* node) {
    if (!node) return;
    if (node->owns_params) cJSON_Delete(node->params);
    free(node->ins); free(node->outs);
    vx_node_clear_param_cache(node);
    memset(node, 0, sizeof(*node));
}

static _Thread_local VxEngineState* g_current_engine_state;
static _Thread_local VxEngineState* g_route_lease_owner;


typedef struct VxGraphMetadataView {
    T* tensors;
    Node* nodes;
    QLinearMetadata* qlinear_metadata;
    QConv2DMetadata* qconv_metadata;
    QEmbeddingMetadata* qembedding_metadata;
    float** kcache;
    float** vcache;
    float** qwcache;
    float** f16wcache;
    float** f16bcache;
    float** conv_wcache;
    void** q8wcache;
    uint32_t* q8wcache_bytes;
    int* concat_sigmoid_fuse;
    GraphNodeFusion* node_fusion;
    int* tensor_name_index;
    void** retired_f16_storage;
    char (*removed_graph_outputs)[128];
    T* graph_patch_reused_old_tensors;
    int* graph_patch_reused_indices;
    unsigned char* incremental_dirty;
    unsigned char* incremental_compatibility_dirty;
    unsigned char* incremental_hybrid_tensors;
    unsigned char* incremental_hybrid_boundary_inputs;
    size_t* incremental_hybrid_prefix_bytes;
    unsigned char* incremental_hybrid_row_nodes;
    unsigned char* incremental_portable_selected_nodes;
    unsigned char* incremental_hybrid_plan_nodes;
    const char** runtime_route_backend;
    float** conv_pwf32_pack;
    float** conv_pwf32_pack_plain;
    float** conv_dw_pw_tmp;
    long* conv_dw_pw_tmp_cap;
    const float*** conv_f32_igemm_indir;
    long* conv_f32_igemm_indir_cap;
    uint64_t* conv_f32_igemm_indir_key;
    float** conv_f32_igemm_zero;
    int* conv_f32_igemm_zero_cap;
    float** conv_f32_igemm_pack;
    long* conv_f32_igemm_pack_cap;
    const float** conv_f32_igemm_pack_src;
#if VOLVOXAI_ENABLE_TRAINING
    EngineOptimizerState* optimizer_states;
#endif
    size_t tensor_name_index_capacity;
} VxGraphMetadataView;

static int vx_size_add(size_t left, size_t right, size_t* result) {
    if (!result || left > SIZE_MAX - right) return -1;
    *result = left + right;
    return 0;
}

static int vx_size_mul(size_t left, size_t right, size_t* result) {
    if (!result || (left && right > SIZE_MAX / left)) return -1;
    *result = left * right;
    return 0;
}

static int vx_layout_array(void* base, size_t* offset, size_t alignment,
                           size_t count, size_t element_size, void** result) {
    size_t padding;
    size_t bytes;
    if (!offset || !alignment || !result) return -1;
    padding = (alignment - (*offset % alignment)) % alignment;
    if (vx_size_add(*offset, padding, offset) != 0 ||
        vx_size_mul(count, element_size, &bytes) != 0) return -1;
    *result = base ? (void*)((unsigned char*)base + *offset) : NULL;
    return vx_size_add(*offset, bytes, offset);
}

static int vx_tensor_name_index_capacity(size_t tensor_capacity,
                                         size_t* result) {
    size_t required;
    size_t capacity = 1;
    if (!result) return -1;
    if (!tensor_capacity) {
        *result = 0;
        return 0;
    }
    if (vx_size_mul(tensor_capacity, 2u, &required) != 0) return -1;
    while (capacity < required) {
        if (capacity > SIZE_MAX / 2u) return -1;
        capacity *= 2u;
    }
    *result = capacity;
    return 0;
}

static void* vx_graph_metadata_allocate(size_t bytes) {
    return bytes ? calloc(1, bytes) : NULL;
}

#define VX_LAYOUT_ARRAY(view, base, offset, field, type, count) \
    do { \
        void* vx_layout_pointer = NULL; \
        if (vx_layout_array((base), &(offset), _Alignof(type), (count), \
                            sizeof(type), &vx_layout_pointer) != 0) return -1; \
        (view)->field = vx_layout_pointer; \
    } while (0)

static int vx_graph_metadata_layout(void* base, size_t tensor_capacity,
                                    size_t node_capacity,
                                    VxGraphMetadataView* view,
                                    size_t* bytes_out) {
    size_t offset = 0;
    size_t name_index_capacity;
    if (!view || !bytes_out ||
        tensor_capacity > (size_t)INT_MAX || node_capacity > (size_t)INT_MAX ||
        vx_tensor_name_index_capacity(tensor_capacity,
                                      &name_index_capacity) != 0) return -1;
    memset(view, 0, sizeof(*view));
    view->tensor_name_index_capacity = name_index_capacity;
    VX_LAYOUT_ARRAY(view, base, offset, tensors, T, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, retired_f16_storage, void*, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, removed_graph_outputs, char[128], tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, graph_patch_reused_old_tensors, T, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, graph_patch_reused_indices, int, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_dirty, unsigned char, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_compatibility_dirty,
                    unsigned char, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_hybrid_tensors,
                    unsigned char, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_hybrid_boundary_inputs,
                    unsigned char, tensor_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_hybrid_prefix_bytes,
                    size_t, tensor_capacity);
#if VOLVOXAI_ENABLE_TRAINING
    VX_LAYOUT_ARRAY(view, base, offset, optimizer_states, EngineOptimizerState,
                    tensor_capacity);
#endif
    VX_LAYOUT_ARRAY(view, base, offset, tensor_name_index, int, name_index_capacity);

    VX_LAYOUT_ARRAY(view, base, offset, nodes, Node, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, qlinear_metadata, QLinearMetadata, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, qconv_metadata, QConv2DMetadata, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, qembedding_metadata, QEmbeddingMetadata,
                    node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, kcache, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, vcache, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, qwcache, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, f16wcache, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, f16bcache, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_wcache, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, q8wcache, void*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, q8wcache_bytes, uint32_t, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, concat_sigmoid_fuse, int, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, node_fusion, GraphNodeFusion, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_hybrid_row_nodes, unsigned char,
                    node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_portable_selected_nodes,
                    unsigned char, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, incremental_hybrid_plan_nodes,
                    unsigned char, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, runtime_route_backend, const char*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_pwf32_pack, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_pwf32_pack_plain, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_dw_pw_tmp, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_dw_pw_tmp_cap, long, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_indir, const float**, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_indir_cap, long, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_indir_key, uint64_t, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_zero, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_zero_cap, int, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_pack, float*, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_pack_cap, long, node_capacity);
    VX_LAYOUT_ARRAY(view, base, offset, conv_f32_igemm_pack_src, const float*,
                    node_capacity);
    *bytes_out = offset;
    return 0;
}

#undef VX_LAYOUT_ARRAY

static void vx_graph_metadata_install(VxEngineState* state,
                                      const VxGraphMetadataView* view,
                                      void* storage, size_t bytes,
                                      size_t tensor_capacity,
                                      size_t node_capacity) {
#define VX_INSTALL(field) state->field = view->field
    VX_INSTALL(tensors); VX_INSTALL(nodes);
    VX_INSTALL(qlinear_metadata); VX_INSTALL(qconv_metadata);
    VX_INSTALL(qembedding_metadata);
    VX_INSTALL(kcache); VX_INSTALL(vcache); VX_INSTALL(qwcache);
    VX_INSTALL(f16wcache); VX_INSTALL(f16bcache); VX_INSTALL(conv_wcache);
    VX_INSTALL(q8wcache); VX_INSTALL(q8wcache_bytes);
    VX_INSTALL(concat_sigmoid_fuse); VX_INSTALL(node_fusion);
    VX_INSTALL(tensor_name_index); VX_INSTALL(retired_f16_storage);
    VX_INSTALL(removed_graph_outputs);
    VX_INSTALL(graph_patch_reused_old_tensors);
    VX_INSTALL(graph_patch_reused_indices); VX_INSTALL(incremental_dirty);
    VX_INSTALL(incremental_hybrid_row_nodes);
    VX_INSTALL(incremental_portable_selected_nodes);
    VX_INSTALL(runtime_route_backend);
    VX_INSTALL(conv_pwf32_pack); VX_INSTALL(conv_pwf32_pack_plain);
    VX_INSTALL(conv_dw_pw_tmp); VX_INSTALL(conv_dw_pw_tmp_cap);
    VX_INSTALL(conv_f32_igemm_indir); VX_INSTALL(conv_f32_igemm_indir_cap);
    VX_INSTALL(conv_f32_igemm_indir_key); VX_INSTALL(conv_f32_igemm_zero);
    VX_INSTALL(conv_f32_igemm_zero_cap); VX_INSTALL(conv_f32_igemm_pack);
    VX_INSTALL(conv_f32_igemm_pack_cap); VX_INSTALL(conv_f32_igemm_pack_src);
#if VOLVOXAI_ENABLE_TRAINING
    VX_INSTALL(optimizer_states);
#endif
#undef VX_INSTALL
    state->incremental_plan.node_capacity = node_capacity;
    state->incremental_scratch.compatibility_dirty =
        view->incremental_compatibility_dirty;
    state->incremental_scratch.hybrid_tensors =
        view->incremental_hybrid_tensors;
    state->incremental_scratch.hybrid_nodes =
        view->incremental_hybrid_plan_nodes;
    state->incremental_scratch.hybrid_boundary_inputs =
        view->incremental_hybrid_boundary_inputs;
    state->incremental_scratch.hybrid_prefix_bytes =
        view->incremental_hybrid_prefix_bytes;
    state->incremental_scratch.tensor_capacity = tensor_capacity;
    state->incremental_scratch.node_capacity = node_capacity;
    state->tensor_name_index_capacity = view->tensor_name_index_capacity;
    state->tensor_capacity = tensor_capacity;
    state->node_capacity = node_capacity;
    state->graph_metadata_storage = storage;
    state->graph_metadata_bytes = bytes;
}

int vx_engine_state_graph_metadata_bytes_for(size_t tensor_capacity,
                                             size_t node_capacity,
                                             size_t* bytes_out) {
    VxGraphMetadataView view;
    return vx_graph_metadata_layout(NULL, tensor_capacity, node_capacity,
                                    &view, bytes_out);
}

#define VX_COPY_ARRAY(destination, source, type, count) \
    do { \
        if ((count)) memcpy((destination), (source), (count) * sizeof(type)); \
    } while (0)

VxEngineResult vx_engine_state_reserve_graph_metadata(
        VxEngineState* state, size_t tensor_capacity, size_t node_capacity) {
    VxGraphMetadataView next;
    size_t bytes;
    size_t transaction_bytes;
    size_t tensor_copy;
    size_t node_copy;
    void* storage;
    if (!state || tensor_capacity > (size_t)INT_MAX ||
        node_capacity > (size_t)INT_MAX || state->tensor_count < 0 ||
        state->node_count < 0 ||
        state->incremental_portable_selection_active)
        return VX_ENGINE_RESULT_ERROR;
    if (tensor_capacity < state->tensor_capacity)
        tensor_capacity = state->tensor_capacity;
    if (node_capacity < state->node_capacity) node_capacity = state->node_capacity;
    if ((size_t)state->tensor_count > tensor_capacity ||
        (size_t)state->node_count > node_capacity) return VX_ENGINE_RESULT_ERROR;
    if (tensor_capacity == state->tensor_capacity &&
        node_capacity == state->node_capacity) return VX_ENGINE_RESULT_OK;
    if (vx_graph_metadata_layout(NULL, tensor_capacity, node_capacity,
                                 &next, &bytes) != 0) return VX_ENGINE_RESULT_ERROR;
    if (vx_size_add(state->graph_metadata_bytes, bytes,
                    &transaction_bytes) != 0)
        return VX_ENGINE_RESULT_OUT_OF_MEMORY;
    storage = vx_graph_metadata_allocate(bytes);
    if (bytes && !storage) return VX_ENGINE_RESULT_OUT_OF_MEMORY;
    if (vx_graph_metadata_layout(storage, tensor_capacity, node_capacity,
                                 &next, &bytes) != 0) {
        free(storage);
        return VX_ENGINE_RESULT_ERROR;
    }
    tensor_copy = state->tensor_capacity < tensor_capacity
        ? state->tensor_capacity : tensor_capacity;
    node_copy = state->node_capacity < node_capacity
        ? state->node_capacity : node_capacity;
    for (size_t index = 0; index < node_capacity; index++)
        next.concat_sigmoid_fuse[index] = -1;
    if (state->graph_metadata_storage) {
        VX_COPY_ARRAY(next.tensors, state->tensors, T, tensor_copy);
        VX_COPY_ARRAY(next.retired_f16_storage, state->retired_f16_storage,
                      void*, tensor_copy);
        VX_COPY_ARRAY(next.removed_graph_outputs, state->removed_graph_outputs,
                      char[128], tensor_copy);
        VX_COPY_ARRAY(next.graph_patch_reused_old_tensors,
                      state->graph_patch_reused_old_tensors, T, tensor_copy);
        VX_COPY_ARRAY(next.graph_patch_reused_indices,
                      state->graph_patch_reused_indices, int, tensor_copy);
        VX_COPY_ARRAY(next.incremental_dirty, state->incremental_dirty,
                      unsigned char, tensor_copy);
#if VOLVOXAI_ENABLE_TRAINING
        VX_COPY_ARRAY(next.optimizer_states, state->optimizer_states,
                      EngineOptimizerState, tensor_copy);
#endif
        VX_COPY_ARRAY(next.nodes, state->nodes, Node, node_copy);
        VX_COPY_ARRAY(next.qlinear_metadata, state->qlinear_metadata,
                      QLinearMetadata, node_copy);
        VX_COPY_ARRAY(next.qconv_metadata, state->qconv_metadata,
                      QConv2DMetadata, node_copy);
        VX_COPY_ARRAY(next.qembedding_metadata, state->qembedding_metadata,
                      QEmbeddingMetadata, node_copy);
#define VX_COPY_NODE(field, type) \
        VX_COPY_ARRAY(next.field, state->field, type, node_copy)
        VX_COPY_NODE(kcache, float*); VX_COPY_NODE(vcache, float*);
        VX_COPY_NODE(qwcache, float*); VX_COPY_NODE(f16wcache, float*);
        VX_COPY_NODE(f16bcache, float*); VX_COPY_NODE(conv_wcache, float*);
        VX_COPY_NODE(q8wcache, void*); VX_COPY_NODE(q8wcache_bytes, uint32_t);
        VX_COPY_NODE(concat_sigmoid_fuse, int);
        VX_COPY_NODE(node_fusion, GraphNodeFusion);
        VX_COPY_NODE(incremental_hybrid_row_nodes, unsigned char);
        VX_COPY_NODE(runtime_route_backend, const char*);
        VX_COPY_NODE(conv_pwf32_pack, float*);
        VX_COPY_NODE(conv_pwf32_pack_plain, float*);
        VX_COPY_NODE(conv_dw_pw_tmp, float*); VX_COPY_NODE(conv_dw_pw_tmp_cap, long);
        VX_COPY_NODE(conv_f32_igemm_indir, const float**);
        VX_COPY_NODE(conv_f32_igemm_indir_cap, long);
        VX_COPY_NODE(conv_f32_igemm_indir_key, uint64_t);
        VX_COPY_NODE(conv_f32_igemm_zero, float*);
        VX_COPY_NODE(conv_f32_igemm_zero_cap, int);
        VX_COPY_NODE(conv_f32_igemm_pack, float*);
        VX_COPY_NODE(conv_f32_igemm_pack_cap, long);
        VX_COPY_NODE(conv_f32_igemm_pack_src, const float*);
#undef VX_COPY_NODE

    }
    if (state->tensors && next.tensors && state->merged_adapter_weights) {
        uintptr_t old_begin = (uintptr_t)state->tensors;
        uintptr_t old_end = old_begin + state->tensor_capacity * sizeof(T);
        for (int index = 0; index < state->merged_adapter_count; index++) {
            uintptr_t pointer = (uintptr_t)state->merged_adapter_weights[index].tensor;
            if (pointer >= old_begin && pointer < old_end &&
                (pointer - old_begin) % sizeof(T) == 0) {
                size_t tensor_index = (pointer - old_begin) / sizeof(T);
                state->merged_adapter_weights[index].tensor = &next.tensors[tensor_index];
            }
        }
    }
    free(state->graph_metadata_storage);
    vx_graph_metadata_install(state, &next, storage, bytes,
                              tensor_capacity, node_capacity);
    if (state->graph_metadata_transaction_peak_bytes < transaction_bytes)
        state->graph_metadata_transaction_peak_bytes = transaction_bytes;
    state->tensor_name_index_count = -1;
    return VX_ENGINE_RESULT_OK;
}

#undef VX_COPY_ARRAY

void vx_engine_state_release_graph_metadata(VxEngineState* state) {
    VxGraphMetadataView empty;
    if (!state) return;
    for (size_t i = 0; i < state->node_capacity; ++i) vx_node_dispose(&state->nodes[i]);
    free(state->graph_metadata_storage);
    memset(&empty, 0, sizeof(empty));
    vx_graph_metadata_install(state, &empty, NULL, 0, 0, 0);
    state->graph_metadata_transaction_peak_bytes = 0;
    state->tensor_count = 0;
    state->node_count = 0;
    state->tensor_name_index_count = -1;
    state->retired_f16_storage_count = 0;
    state->removed_graph_output_count = 0;
    state->graph_patch_reused_count = 0;
    state->incremental_plan.generation = 0;
    state->incremental_plan.tensor_count = 0;
    state->incremental_plan.node_count = 0;
    state->incremental_plan.valid = 0;
    state->incremental_portable_selection_active = 0;
}

int vx_engine_state_graph_metadata_owned_bytes(const VxEngineState* state,
                                               uint64_t* resident, uint64_t* peak) {
    if (!state || !resident || !peak) return -1;
    uint64_t owned = 0;
    for (size_t i = 0; i < state->node_capacity; ++i) {
        const Node* node = &state->nodes[i];
        if (node->input_capacity < 0 || node->output_capacity < 0) return -1;
        uint64_t bytes = ((uint64_t)node->input_capacity + (uint64_t)node->output_capacity) * sizeof(Ref);
        for (int slot = 0; slot < VX_NODE_ARRAY_COUNT; ++slot) {
            if (node->parsed_params.array_bytes[slot] > UINT64_MAX - bytes) return -1;
            bytes += node->parsed_params.array_bytes[slot];
        }
        if (bytes > UINT64_MAX - owned) return -1;
        owned += bytes;
    }
    if (owned > UINT64_MAX - state->graph_metadata_bytes ||
        owned > (UINT64_MAX - state->graph_metadata_transaction_peak_bytes) / 2u) return -1;
    *resident = state->graph_metadata_bytes + owned;
    *peak = state->graph_metadata_transaction_peak_bytes + 2u * owned;
    return *peak >= *resident ? 0 : -1;
}

int vx_engine_state_init_with_kernel_pool(
        VxEngineState* state,
        VxKernelThreadPool* kernel_thread_pool,
        int thread_count) {
    if (!state || thread_count < 0) return -1;
    memset(state, 0, sizeof(*state));
    state->active_row = -1;
    state->execution_row = -1;
    state->tensor_name_index_count = -1;
    state->incremental_hybrid_prepared_row = -1;
    state->last_failure_node_index = -1;
    state->model_generation = 1;
    {
        const char* value = vx_engine_env("VOLVOX_PW_GEMM");
        state->conv_pw_gemm_enabled =
            !(value && value[0] && strcmp(value, "0") == 0);
    }
    {
        /* Default on, and the shape decides the rest.
         *
         * This was default off, from a comment citing three image-encoder
         * shapes where the packed kernel measured 8.16 GMAC/s against 20.38 for
         * unpacked.  That measurement was real but the conclusion did not
         * generalize: it inverted at LLM shapes, where the unpacked kernel
         * collapses to 4-7 GMAC/s because it has no cache blocking and rereads
         * the whole weight per row block, while packed stayed flat.  A
         * process-wide boolean cannot express that, which is why the packed
         * kernel now carries a plan (vx_gemm_f32_plan) and picks its blocking
         * and regime from M, K and N.  Measured after that change, one thread:
         * 402x320x320 42.1 -> 56.6, 512x2048x2048 7.6 -> 52.6, 1x4096x4096
         * 1.2 -> 4.2 GMAC/s, bit-identical to matmul_f32 throughout.
         *
         * The switch remains as an escape hatch to the unpacked kernel, not as
         * the place the policy lives. */
        const char* value = vx_engine_env("VOLVOX_F32_GEMM_PACKED");
        state->gemm_f32_packed_enabled =
            !(value && value[0] && strcmp(value, "0") == 0);
    }
    state->backend = VX_PORTABLE_BACKEND_KIND;
#if VOLVOXAI_ENABLE_TRAINING
    state->dynamic_autograd_forward_backend = -1;
#endif
    if (vx_mutex_init(&state->adapter_admin_mutex) != 0)
        return -1;
    if (vx_mutex_init(&state->model_mutex) != 0) {
        vx_mutex_destroy(&state->adapter_admin_mutex);
        return -1;
    }
    if (vx_mutex_init(&state->metadata_mutex) != 0) {
        vx_mutex_destroy(&state->model_mutex);
        vx_mutex_destroy(&state->adapter_admin_mutex);
        return -1;
    }
    state->kernel_thread_pool = kernel_thread_pool
        ? kernel_thread_pool
        : vx_kernel_thread_pool_create(thread_count);
    if (!state->kernel_thread_pool) {
        vx_mutex_destroy(&state->metadata_mutex);
        vx_mutex_destroy(&state->model_mutex);
        vx_mutex_destroy(&state->adapter_admin_mutex);
        return -1;
    }
    state->kernel_thread_pool_owned = kernel_thread_pool ? 0 : 1;
    state->kernel_thread_pool_thread_count = thread_count;
    state->cpu_threads = thread_count;
    return 0;
}

int vx_engine_state_init(VxEngineState* state) {
    return vx_engine_state_init_with_kernel_pool(state, NULL, 0);
}

void vx_engine_state_deinit(VxEngineState* state) {
    if (!state) return;
    assert(!state->loaded);
    assert(!state->graph_root);
    assert(!state->merged_adapter_weights);
    /* Shutdown normally owns this cleanup. Keep deinit complete for an engine
     * whose load was abandoned after residency registration but before init. */
    for (size_t index = 0; index < state->bank_residency_count; index++)
        free(state->bank_residency[index].slot_rows);
    free(state->bank_residency);
    state->bank_residency = NULL;
    state->bank_residency_count = 0;
    state->bank_residency_capacity = 0;
    free(state->decode_stage);
    state->decode_stage = NULL;
    state->decode_stage_bytes = 0;
    state->decode_lanes = 0;
    if (state->adapter_registry_state_destroy)
        state->adapter_registry_state_destroy(state->adapter_registry_state);
    state->adapter_registry_state = NULL;
    state->adapter_registry_state_destroy = NULL;
    if (state->vulkan_context_state_destroy)
        state->vulkan_context_state_destroy(state->vulkan_context_state);
    state->vulkan_context_state = NULL;
    state->vulkan_context_state_destroy = NULL;
    if (state->opengl_context_state_destroy)
        state->opengl_context_state_destroy(state->opengl_context_state);
    state->opengl_context_state = NULL;
    state->opengl_context_state_destroy = NULL;
    if (state->metal_context_state_destroy)
        state->metal_context_state_destroy(state->metal_context_state);
    state->metal_context_state = NULL;
    state->metal_context_state_destroy = NULL;
    if (state->cuda_context_state_destroy)
        state->cuda_context_state_destroy(state->cuda_context_state);
    state->cuda_context_state = NULL;
    state->cuda_context_state_destroy = NULL;
    /* Ordinary shutdown releases this first. Keep deinit defensive for a
     * partially initialized context whose graph load never reached shutdown. */
    free(state->cpu_typed_workspace);
    state->cpu_typed_workspace = NULL;
    state->cpu_typed_workspace_bound_bytes = 0u;
    state->cpu_typed_workspace_capacity_bytes = 0u;
    state->cpu_typed_workspace_configured = 0;
    state->portable_graph_plan_request_v1 = NULL;
    state->portable_graph_plan_request_v1_bytes = 0u;
    state->portable_graph_plan_v1 = NULL;
    state->portable_graph_plan_v1_bytes = 0u;
    state->portable_cpu_activation_plan_enabled = 0;
    vx_engine_state_release_graph_metadata(state);
    if (state->kernel_thread_pool_owned)
        vx_kernel_thread_pool_destroy(state->kernel_thread_pool);
    state->kernel_thread_pool = NULL;
    state->kernel_thread_pool_owned = 0;
    state->kernel_thread_pool_thread_count = 0;
    if (state->memory_observer_clear) state->memory_observer_clear(&state->memory_observer);
    vx_mutex_destroy(&state->metadata_mutex);
    vx_mutex_destroy(&state->model_mutex);
    vx_mutex_destroy(&state->adapter_admin_mutex);
}

VxEngineState* vx_engine_state_current(void) {
    return g_current_engine_state;
}

VxEngineStateScope vx_engine_state_scope_enter(VxEngineState* state) {
    assert(state);
    VxEngineStateScope scope = {
        .previous = g_current_engine_state,
        .bound = state,
        .kernel_pool_scope = vx_kernel_thread_pool_scope_enter(
            state->kernel_thread_pool),
    };
    assert(!g_route_lease_owner || g_route_lease_owner == scope.bound);
    g_current_engine_state = scope.bound;
    return scope;
}

void vx_engine_state_scope_leave(VxEngineStateScope scope) {
    assert(g_current_engine_state == scope.bound);
    assert(!g_route_lease_owner || g_route_lease_owner == scope.previous);
    vx_kernel_thread_pool_scope_leave(scope.kernel_pool_scope);
    g_current_engine_state = scope.previous;
}

int vx_engine_state_route_lease_active(void) {
    return g_route_lease_owner &&
        g_route_lease_owner == vx_engine_state_current();
}

int vx_engine_state_route_lease_begin(void) {
    VxEngineState* state = vx_engine_state_current();
    if (!state || g_route_lease_owner) return -1;
    g_route_lease_owner = state;
    return 0;
}

void vx_engine_state_route_lease_end(void) {
    assert(g_route_lease_owner == vx_engine_state_current());
    g_route_lease_owner = NULL;
}
