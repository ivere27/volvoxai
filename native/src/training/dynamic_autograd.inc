/*
 * Full-profile native eager autograd.
 *
 * This fragment is composed into engine_runtime.c through training_runtime.inc
 * after the portable backward driver and GPU planner.  It deliberately owns
 * no inference-profile symbol.  A context keeps its tensors/nodes outside the
 * packaged-model globals and projects borrowed metadata into empty g_t/g_n
 * tables only while one public operation holds the model mutex.
 */

#define VX_AUTOGRAD_CONTEXT_MAGIC UINT64_C(0x5658414743545831)
#define VX_AUTOGRAD_TENSOR_MAGIC  UINT64_C(0x56584147544e5331)

enum {
    VX_DYNAMIC_AUTOGRAD_TENSOR_CAPACITY = 8192,
    VX_DYNAMIC_AUTOGRAD_NODE_CAPACITY = 4096
};

typedef struct VxDynamicAutogradNode {
    Node runtime;
    struct VolvoxAIAutogradTensor** inputs;
    struct VolvoxAIAutogradTensor** outputs;
    uint64_t* input_versions;
    int recorded;
    int history_released;
} VxDynamicAutogradNode;

static void dynamic_autograd_node_clear(VxDynamicAutogradNode* node) {
    vx_node_dispose(&node->runtime);
    free(node->inputs); free(node->outputs); free(node->input_versions);
    memset(node, 0, sizeof(*node));
}


struct VolvoxAIAutogradContext {
    uint64_t magic;
    int backend;
    uint32_t training_counter;
    uint32_t unrecorded_counter;
    int no_grad_depth;
    int poisoned;
    uint64_t next_tensor_id;
    struct VolvoxAIAutogradTensor** tensors;
    int tensor_count;
    struct VolvoxAIAutogradTensor** handles;
    int handle_count;
    int handle_capacity;
    VxDynamicAutogradNode* nodes;
    int node_count;
};

struct VolvoxAIAutogradTensor {
    uint64_t magic;
    VolvoxAIAutogradContext* context;
    char name[128];
    int shape[8];
    int ndim;
    int dtype;
    size_t elem_size;
    long numel;
    void* data;
    float* grad;
    uint64_t version;
    int slot;
    int requires_grad;
    int is_leaf;
    int alive;
};

int vx_dynamic_autograd_active_locked(void) {
    return g_dynamic_autograd_context &&
        g_dynamic_autograd_context->magic == VX_AUTOGRAD_CONTEXT_MAGIC;
}

/* run_node() calls this after registry selection.  It is private to this
   translation unit and inert outside one eager apply, so ordinary inference
   dispatch retains no additional state or branch-visible ABI. */
static void dynamic_autograd_note_forward_backend(VxBackendKind kind) {
    if (g_dynamic_autograd_forward_capture)
        g_dynamic_autograd_forward_backend = kind;
}

static size_t dynamic_autograd_dtype_size(int dtype) {
    if (dtype == VOLVOXAI_DTYPE_F32 || dtype == VOLVOXAI_DTYPE_I32)
        return sizeof(uint32_t);
    return 0;
}

static int dynamic_autograd_shape(const int32_t shape[8], int ndim,
                                  long* out_numel) {
    long numel = 1;
    if (!out_numel || ndim < 0 || ndim > 8) return -1;
    for (int dimension = 0; dimension < ndim; dimension++) {
        int value = shape[dimension];
        if (value <= 0 || numel > LONG_MAX / value) return -1;
        numel *= value;
    }
    *out_numel = numel;
    return 0;
}

static int dynamic_autograd_current_backend(void) {
    if (g_use_cuda) return VX_BACKEND_KIND_CUDA;
    if (g_use_metal) return VX_BACKEND_KIND_METAL;
    if (g_use_opengl) return VX_BACKEND_KIND_OPENGL;
    if (g_use_vulkan) return VX_BACKEND_KIND_VULKAN;
    return VX_PORTABLE_BACKEND_KIND;
}

static int dynamic_autograd_backend_supported(int backend) {
    if (backend == VX_PORTABLE_BACKEND_KIND) return 1;
    if (backend == VX_BACKEND_KIND_VULKAN) return VOLVOXAI_ENABLE_VULKAN;
    if (backend == VX_BACKEND_KIND_OPENGL) return VOLVOXAI_ENABLE_OPENGL;
    if (backend == VX_BACKEND_KIND_METAL) return VOLVOXAI_ENABLE_METAL;
    if (backend == VX_BACKEND_KIND_CUDA) return VOLVOXAI_ENABLE_CUDA;
    return 0;
}

static NativeGpuTrainingBackend dynamic_autograd_gpu_backend(int backend) {
    if (backend == VX_BACKEND_KIND_VULKAN)
        return NATIVE_GPU_TRAINING_VULKAN;
    if (backend == VX_BACKEND_KIND_OPENGL)
        return NATIVE_GPU_TRAINING_OPENGL;
    if (backend == VX_BACKEND_KIND_METAL)
        return NATIVE_GPU_TRAINING_METAL;
    if (backend == VX_BACKEND_KIND_CUDA)
        return NATIVE_GPU_TRAINING_CUDA;
    return NATIVE_GPU_TRAINING_NONE;
}

static int dynamic_autograd_context_identity_valid(
        const VolvoxAIAutogradContext* context) {
    return context && context == g_dynamic_autograd_context &&
        context->magic == VX_AUTOGRAD_CONTEXT_MAGIC;
}

static int dynamic_autograd_context_valid(
        const VolvoxAIAutogradContext* context) {
    return dynamic_autograd_context_identity_valid(context) &&
        !context->poisoned;
}

static int dynamic_autograd_tensor_valid(
        const VolvoxAIAutogradContext* context,
        const VolvoxAIAutogradTensor* tensor) {
    return dynamic_autograd_context_valid(context) && tensor && tensor->alive &&
        tensor->magic == VX_AUTOGRAD_TENSOR_MAGIC &&
        tensor->context == context && tensor->slot >= 0 &&
        tensor->slot < context->tensor_count &&
        context->tensors[tensor->slot] == tensor;
}

/* This list is intentionally identical to the native reverse driver's public
 * graph intersection.  New forward operators do not silently become autograd
 * promises merely because run_node() can execute them. */
static int dynamic_autograd_operator_supported(VxOperatorKind op) {
    if (op == VX_OP_UNSPECIFIED) return 0;
    return (op == VX_OP_MATMUL) || (op == VX_OP_LINEAR) ||
        (op == VX_OP_GEMM) || (op == VX_OP_ADD) ||
        (op == VX_OP_MUL) || (op == VX_OP_EMBEDDING) ||
        (op == VX_OP_LAYER_NORM) || (op == VX_OP_GROUP_NORM) ||
        (op == VX_OP_RMS_NORM) || (op == VX_OP_RELU) ||
        (op == VX_OP_GELU) || (op == VX_OP_SILU) ||
        (op == VX_OP_SIGMOID) ||
        (op == VX_OP_TANH) || (op == VX_OP_HARD_SWISH) ||
        (op == VX_OP_HARD_SIGMOID) || (op == VX_OP_LEAKY_RELU) ||
        (op == VX_OP_CLIP) || (op == VX_OP_PRELU) ||
        (op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX) ||
        (op == VX_OP_SDPA) || (op == VX_OP_CROSS_SDPA) ||
        (op == VX_OP_MOE_LINEAR) || (op == VX_OP_MOE_ROUTER) ||
        (op == VX_OP_CONV_2D) || (op == VX_OP_BATCH_NORM_2D) ||
        (op == VX_OP_MAX_POOL_2D) || (op == VX_OP_GLOBAL_AVERAGE_POOL) ||
        (op == VX_OP_RESIZE_NEAREST_2D) ||
        (op == VX_OP_UPSAMPLE_NEAREST_2D) || (op == VX_OP_CONCAT) ||
        (op == VX_OP_TRANSPOSE) || (op == VX_OP_DROPOUT) ||
        (op == VX_OP_REDUCE_SUM) || (op == VX_OP_REDUCE_MEAN) ||
        (op == VX_OP_IDENTITY) || (op == VX_OP_RESHAPE) ||
        (op == VX_OP_FLATTEN) || (op == VX_OP_SQUEEZE) ||
        (op == VX_OP_UNSQUEEZE) || (op == VX_OP_SPLIT);
}

static int dynamic_autograd_node_has_input(const Node* node,
                                           VxPortKind key) {
    if (!node || !key) return 0;
    for (int index = 0; index < node->nin; index++)
        if (node->ins[index].port == key) return 1;
    return 0;
}

static int dynamic_autograd_node_has_output(const Node* node,
                                            VxPortKind key) {
    if (!node || !key) return 0;
    for (int index = 0; index < node->nout; index++)
        if (node->outs[index].port == key) return 1;
    return 0;
}

static int dynamic_autograd_key_in(VxPortKind key,
                                   const VxPortKind* keys, int count) {
    if (!key || !keys || count < 0) return 0;
    for (int index = 0; index < count; index++)
        if (!(key != keys[index])) return 1;
    return 0;
}

static int dynamic_autograd_fixed_inputs(
        const Node* node, const VxPortKind* required, int required_count,
        const VxPortKind* optional, int optional_count) {
    if (!node || required_count < 0 || optional_count < 0 ||
        node->nin < required_count ||
        node->nin > required_count + optional_count) return 0;
    for (int index = 0; index < required_count; index++)
        if (!dynamic_autograd_node_has_input(node, required[index])) return 0;
    for (int index = 0; index < node->nin; index++) {
        VxPortKind key = node->ins[index].port;
        if (!dynamic_autograd_key_in(key, required, required_count) &&
            !dynamic_autograd_key_in(key, optional, optional_count)) return 0;
    }
    return 1;
}

/* run_node() consumes a graph-loader-validated Node and several CPU
 * branches dereference canonical bindings before their kernel validator runs.
 * The eager API therefore admits a deliberately strict canonical spelling,
 * then uses the backward-plan builders below as the common shape validator. */
static int dynamic_autograd_node_schema_supported(const Node* node) {
    static const VxPortKind input[] = {VX_PORT_INPUT};
    static const VxPortKind input_weight[] = {VX_PORT_INPUT, VX_PORT_WEIGHT};
    static const VxPortKind bias[] = {VX_PORT_BIAS};
    static const VxPortKind binary[] = {VX_PORT_A, VX_PORT_B};
    static const VxPortKind batchnorm[] = {
        VX_PORT_INPUT, VX_PORT_WEIGHT, VX_PORT_BIAS, VX_PORT_RUNNING_MEAN, VX_PORT_RUNNING_VAR
    };
    static const VxPortKind qkv[] = {VX_PORT_QKV};
    static const VxPortKind qkv_optional[] = {VX_PORT_MASK};
    static const VxPortKind cross[] = {VX_PORT_Q, VX_PORT_K, VX_PORT_V};
    static const VxPortKind moe_router[] = {VX_PORT_INPUT, VX_PORT_WEIGHT};
    static const VxPortKind moe_linear_canonical[] = {
        VX_PORT_INPUT, VX_PORT_EXPERT_WEIGHT, VX_PORT_ROUTE_INDICES, VX_PORT_ROUTE_WEIGHTS
    };
    static const VxPortKind moe_linear_short[] = {
        VX_PORT_INPUT, VX_PORT_EXPERT_WEIGHT, VX_PORT_INDICES, VX_PORT_WEIGHTS
    };
    static const VxPortKind expert_bias[] = {VX_PORT_EXPERT_BIAS};
    VxOperatorKind op;
    if (!node || !(op = node->operator_kind) || node->nout <= 0) return 0;
    if (op == VX_OP_SPLIT) {
        if (!dynamic_autograd_fixed_inputs(node, input, 1, NULL, 0)) return 0;
    } else if (op == VX_OP_MOE_ROUTER) {
        if (!dynamic_autograd_fixed_inputs(node, moe_router, 2, bias, 1) ||
            node->nout != 2 ||
            !(dynamic_autograd_node_has_output(node, VX_PORT_INDICES) ||
              dynamic_autograd_node_has_output(node, VX_PORT_ROUTE_INDICES)) ||
            !(dynamic_autograd_node_has_output(node, VX_PORT_WEIGHTS) ||
              dynamic_autograd_node_has_output(node, VX_PORT_ROUTE_WEIGHTS))) return 0;
    } else {
        if (node->nout != 1) return 0;
        if ((op == VX_OP_MATMUL) || (op == VX_OP_LINEAR) ||
            (op == VX_OP_GEMM) || (op == VX_OP_CONV_2D) ||
            (op == VX_OP_LAYER_NORM)) {
            if (!dynamic_autograd_fixed_inputs(node, input_weight, 2, bias, 1))
                return 0;
        } else if ((op == VX_OP_EMBEDDING) || (op == VX_OP_RMS_NORM)) {
            if (!dynamic_autograd_fixed_inputs(node, input_weight, 2, NULL, 0))
                return 0;
        } else if (op == VX_OP_PRELU) {
            if (!dynamic_autograd_node_has_input(node, VX_PORT_INPUT) ||
                (dynamic_autograd_node_has_input(node, VX_PORT_WEIGHT) +
                 dynamic_autograd_node_has_input(node, VX_PORT_SLOPE) != 1) ||
                node->nin != 2) return 0;
            for (int index = 0; index < node->nin; index++) {
                VxPortKind key = node->ins[index].port;
                if ((key != VX_PORT_INPUT) && (key != VX_PORT_WEIGHT) &&
                    (key != VX_PORT_SLOPE)) return 0;
            }
        } else if (op == VX_OP_GROUP_NORM) {
            if (!dynamic_autograd_node_has_input(node, VX_PORT_INPUT) ||
                (dynamic_autograd_node_has_input(node, VX_PORT_WEIGHT) +
                 dynamic_autograd_node_has_input(node, VX_PORT_SCALE) != 1) ||
                node->nin < 2 || node->nin > 3) return 0;
            for (int index = 0; index < node->nin; index++) {
                VxPortKind key = node->ins[index].port;
                if ((key != VX_PORT_INPUT) && (key != VX_PORT_WEIGHT) &&
                    (key != VX_PORT_SCALE) && (key != VX_PORT_BIAS)) return 0;
            }
        } else if (op == VX_OP_BATCH_NORM_2D) {
            if (!dynamic_autograd_fixed_inputs(node, batchnorm, 5, NULL, 0))
                return 0;
        } else if ((op == VX_OP_ADD) || (op == VX_OP_MUL)) {
            if (!dynamic_autograd_fixed_inputs(node, binary, 2, NULL, 0)) return 0;
        } else if (op == VX_OP_SDPA) {
            if (!dynamic_autograd_fixed_inputs(node, qkv, 1, qkv_optional, 1))
                return 0;
        } else if (op == VX_OP_CROSS_SDPA) {
            if (!dynamic_autograd_fixed_inputs(node, cross, 3, qkv_optional, 1))
                return 0;
        } else if (op == VX_OP_MOE_LINEAR) {
            if (!dynamic_autograd_fixed_inputs(
                    node, moe_linear_canonical, 4, expert_bias, 1) &&
                !dynamic_autograd_fixed_inputs(
                    node, moe_linear_short, 4, expert_bias, 1)) return 0;
        } else if (op == VX_OP_CONCAT) {
            if (node->nin <= 0) return 0;
        } else {
            if (!dynamic_autograd_fixed_inputs(node, input, 1, NULL, 0)) return 0;
        }
    }
    return 1;
}

/* These operands affect forward execution but have no VJP in the native
   backward contract.  Do not let a generic F32 leaf opt into a gradient that
   backward would silently omit. */
static int dynamic_autograd_input_role_differentiable(VxOperatorKind op,
                                                      VxPortKind key) {
    if (!op || !key) return 0;
    if ((op == VX_OP_BATCH_NORM_2D) &&
        (!(key != VX_PORT_RUNNING_MEAN) || !(key != VX_PORT_RUNNING_VAR)))
        return 0;
    if ((op == VX_OP_MOE_LINEAR) &&
        (!(key != VX_PORT_ROUTE_INDICES) || !(key != VX_PORT_INDICES)))
        return 0;
    if (((op == VX_OP_SDPA) || (op == VX_OP_CROSS_SDPA)) &&
        !(key != VX_PORT_MASK)) return 0;
    return 1;
}

static void dynamic_autograd_free_node_cache(int index) {
    if (index < 0 || (size_t)index >= g_node_capacity) return;
    free(g_kcache[index]); g_kcache[index] = NULL;
    free(g_vcache[index]); g_vcache[index] = NULL;
    free(g_qwcache[index]); g_qwcache[index] = NULL;
    free(g_f16wcache[index]); g_f16wcache[index] = NULL;
    free(g_f16bcache[index]); g_f16bcache[index] = NULL;
    free(g_conv_wcache[index]); g_conv_wcache[index] = NULL;
    free(g_q8wcache[index]); g_q8wcache[index] = NULL;
    g_q8wcache_bytes[index] = 0;
}

static void dynamic_autograd_free_all_node_caches(void) {
    for (size_t index = 0; index < g_node_capacity; index++)
        dynamic_autograd_free_node_cache((int)index);
}

static int dynamic_autograd_materialize_tensor(
        const VolvoxAIAutogradTensor* tensor) {
    if (!tensor || !tensor->data || tensor->numel <= 0 || !tensor->elem_size ||
        (size_t)tensor->numel > SIZE_MAX / tensor->elem_size) return -1;
    if (!vx_runtime_backend_has_graph()) return 0;
    return vx_runtime_backend_sync_host(
        tensor->data, (size_t)tensor->numel * tensor->elem_size, 0);
}

static int dynamic_autograd_materialize_prefix(
        const VolvoxAIAutogradContext* context, int tensor_count) {
    if (!context || tensor_count < 0 || tensor_count > context->tensor_count)
        return -1;
    for (int index = 0; index < tensor_count; index++) {
        if (dynamic_autograd_materialize_tensor(context->tensors[index]) != 0)
            return -1;
    }
    return 0;
}

/* A failed eager forward may have made both old and unpublished values
   device-authoritative.  Backend address tables must be discarded before the
   unpublished host pointers are freed.  If preserving the old prefix fails,
   reject every later operation rather than exposing stale host data. */
static void dynamic_autograd_rollback_backend(
        VolvoxAIAutogradContext* context, int old_tensor_count) {
    if (!context) return;
    if (dynamic_autograd_materialize_prefix(context, old_tensor_count) != 0)
        context->poisoned = 1;
    /* Eager operands used in semantic weight positions may have been promoted
       to persistent model slots by a graph backend.  An autograd context owns
       no packaged model, so discard the complete address table after a failed
       apply instead of retaining a host identity that the eager API may later
       release. */
    vx_runtime_backend_reset();
}

static int dynamic_autograd_project(
        VolvoxAIAutogradContext* context, const unsigned char* live_nodes,
        int for_backward) {
    if (!dynamic_autograd_context_valid(context) || g_loaded || g_nt || g_nn ||
        g_active_row != -1 || g_prefix_rows != 0 || g_execution_row != -1 ||
        context->tensor_count < 0 || context->node_count < 0 ||
        dynamic_autograd_current_backend() != context->backend)
        return -1;
    if (vx_engine_state_reserve_graph_metadata(
            vx_engine_state_current(), (size_t)context->tensor_count,
            (size_t)context->node_count) != 0) return -1;
    graph_opt_reset_state();
    g_active_row = -1;
    g_prefix_rows = 0;
    g_execution_row = -1;
    for (int index = 0; index < context->tensor_count; index++) {
        VolvoxAIAutogradTensor* tensor = context->tensors[index];
        if (!dynamic_autograd_tensor_valid(context, tensor)) goto fail;
        memset(&g_t[index], 0, sizeof(g_t[index]));
        memcpy(g_t[index].name, tensor->name, sizeof(g_t[index].name));
        memcpy(g_t[index].shape, tensor->shape, sizeof(g_t[index].shape));
        /* The packaged runtime represents one value as [1] in its operator
           kernels, while the eager ABI admits a conventional rank-0 scalar.
           Project that one metadata difference without changing public shape. */
        g_t[index].ndim = tensor->ndim ? tensor->ndim : 1;
        if (!tensor->ndim) g_t[index].shape[0] = 1;
        g_t[index].data = (float*)tensor->data;
        g_t[index].numel = tensor->numel;
        g_t[index].owns = 0;
        g_t[index].dtype = tensor->dtype;
        g_t[index].elem_size = tensor->elem_size;
        g_t[index].is_graph_input = 0;
    }
    g_nt = context->tensor_count;
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    for (int index = 0; index < context->node_count; index++) {
        g_n[index] = context->nodes[index].runtime;
        g_n[index].owns_params = 0;
        g_n[index].skip = for_backward &&
            (!context->nodes[index].recorded || !live_nodes || !live_nodes[index]);
    }
    g_nn = context->node_count;
    return 0;
fail:
    g_nt = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    return -1;
}

static void dynamic_autograd_unproject(VolvoxAIAutogradContext* context) {
    int tensor_count = context ? context->tensor_count : g_nt;
    int node_count = context ? context->node_count : g_nn;
    if ((size_t)tensor_count > g_tensor_capacity)
        tensor_count = (int)g_tensor_capacity;
    if ((size_t)node_count > g_node_capacity)
        node_count = (int)g_node_capacity;
    memset(g_t, 0, (size_t)tensor_count * sizeof(g_t[0]));
    memset(g_n, 0, (size_t)node_count * sizeof(g_n[0]));
    g_nt = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    g_nn = 0;
    graph_opt_reset_state();
    g_active_row = -1;
    g_prefix_rows = 0;
    g_execution_row = -1;
}

static int dynamic_autograd_append_handle(VolvoxAIAutogradContext* context,
                                          VolvoxAIAutogradTensor* tensor) {
    if (!context || !tensor) return -1;
    if (context->handle_count == context->handle_capacity) {
        int capacity = context->handle_capacity ? context->handle_capacity * 2 : 32;
        if (capacity < context->handle_count ||
            (size_t)capacity > SIZE_MAX / sizeof(context->handles[0]))
            return -1;
        VolvoxAIAutogradTensor** handles =
            (VolvoxAIAutogradTensor**)realloc(context->handles,
                (size_t)capacity * sizeof(context->handles[0]));
        if (!handles) return -1;
        context->handles = handles;
        context->handle_capacity = capacity;
    }
    context->handles[context->handle_count++] = tensor;
    return 0;
}

static VolvoxAIAutogradTensor* dynamic_autograd_tensor_create_locked(
        VolvoxAIAutogradContext* context, int dtype, const int32_t shape[8],
        int ndim, const void* data, uint64_t nbytes, int requires_grad,
        int is_leaf) {
    long numel;
    size_t elem_size = dynamic_autograd_dtype_size(dtype);
    if (!dynamic_autograd_context_valid(context) || !shape || !elem_size ||
        context->tensor_count >= VX_DYNAMIC_AUTOGRAD_TENSOR_CAPACITY ||
        dynamic_autograd_shape(shape, ndim, &numel) != 0 || numel <= 0 ||
        (size_t)numel > SIZE_MAX / elem_size ||
        (requires_grad && dtype != VOLVOXAI_DTYPE_F32)) return NULL;
    size_t bytes = (size_t)numel * elem_size;
    if ((data && nbytes != bytes) || (!data && nbytes != 0)) return NULL;
    VolvoxAIAutogradTensor* tensor =
        (VolvoxAIAutogradTensor*)calloc(1, sizeof(*tensor));
    if (!tensor) return NULL;
    tensor->data = calloc(1, bytes);
    if (!tensor->data) { free(tensor); return NULL; }
    if (data) memcpy(tensor->data, data, bytes);
    tensor->magic = VX_AUTOGRAD_TENSOR_MAGIC;
    tensor->context = context;
    tensor->ndim = ndim;
    for (int dimension = 0; dimension < ndim; dimension++)
        tensor->shape[dimension] = shape[dimension];
    tensor->dtype = dtype;
    tensor->elem_size = elem_size;
    tensor->numel = numel;
    tensor->slot = context->tensor_count;
    tensor->requires_grad = requires_grad ? 1 : 0;
    tensor->is_leaf = is_leaf ? 1 : 0;
    tensor->alive = 1;
    uint64_t id = context->next_tensor_id++;
    int written = snprintf(tensor->name, sizeof(tensor->name),
                           "__vx_autograd_%llu",
                           (unsigned long long)id);
    if (written <= 0 || written >= (int)sizeof(tensor->name) ||
        dynamic_autograd_append_handle(context, tensor) != 0) {
        free(tensor->data);
        free(tensor);
        return NULL;
    }
    context->tensors[context->tensor_count++] = tensor;
    if (vx_runtime_backend_has_graph())
        vx_runtime_backend_mark_host(tensor->data, bytes, 0);
    return tensor;
}

static void dynamic_autograd_invalidate_tensor(
        VolvoxAIAutogradTensor* tensor) {
    if (!tensor) return;
    free(tensor->data);
    free(tensor->grad);
    tensor->data = NULL;
    tensor->grad = NULL;
    tensor->alive = 0;
    tensor->slot = -1;
}

static int dynamic_autograd_handle_index(
        const VolvoxAIAutogradContext* context,
        const VolvoxAIAutogradTensor* tensor) {
    if (!context || !tensor) return -1;
    for (int index = 0; index < context->handle_count; index++)
        if (context->handles[index] == tensor) return index;
    return -1;
}

/* Destroy only handles that the caller can no longer observe: unpublished
   failed-apply outputs, or pointers explicitly surrendered through release(). */
static int dynamic_autograd_destroy_handle(
        VolvoxAIAutogradContext* context, VolvoxAIAutogradTensor* tensor) {
    int handle_index = dynamic_autograd_handle_index(context, tensor);
    if (handle_index < 0) return -1;
    if (tensor->alive) dynamic_autograd_invalidate_tensor(tensor);
    if (handle_index + 1 < context->handle_count) {
        memmove(&context->handles[handle_index],
                &context->handles[handle_index + 1],
                (size_t)(context->handle_count - handle_index - 1) *
                    sizeof(context->handles[0]));
    }
    context->handles[--context->handle_count] = NULL;
    tensor->magic = 0;
    tensor->context = NULL;
    free(tensor);
    return 0;
}

static int dynamic_autograd_build_reachability(
        VolvoxAIAutogradContext* context,
        const VolvoxAIAutogradTensor* output,
        unsigned char* live_nodes) {
    unsigned char* live_tensors;
    if (!context || !output || !live_nodes) return -1;
    live_tensors = (unsigned char*)calloc((size_t)context->tensor_count, 1);
    if (!live_tensors) return -1;
    live_tensors[output->slot] = 1;
    for (int node_index = context->node_count - 1; node_index >= 0;
         node_index--) {
        VxDynamicAutogradNode* node = &context->nodes[node_index];
        int live = 0;
        for (int output_index = 0; output_index < node->runtime.nout;
             output_index++) {
            VolvoxAIAutogradTensor* candidate = node->outputs[output_index];
            if (candidate && candidate->alive &&
                live_tensors[candidate->slot]) { live = 1; break; }
        }
        if (!live) continue;
        if (node->history_released) {
            free(live_tensors);
            return -1;
        }
        if (!node->recorded) continue;
        live_nodes[node_index] = 1;
        for (int input_index = 0; input_index < node->runtime.nin;
             input_index++) {
            VolvoxAIAutogradTensor* input = node->inputs[input_index];
            if (!dynamic_autograd_tensor_valid(context, input) ||
                input->version != node->input_versions[input_index]) {
                free(live_tensors);
                return -1;
            }
            if (input->requires_grad) live_tensors[input->slot] = 1;
        }
    }
    free(live_tensors);
    return 0;
}

/* The GPU-plan builders are side-effect-free descriptor/shape validators
 * shared by every native training backend.  Validate the candidate VJP even
 * for CPU and no-grad execution, then additionally preflight every concrete
 * GPU command before running forward or publishing output handles. */
static int dynamic_autograd_validate_candidate(
        VolvoxAIAutogradContext* context,
        VxDynamicAutogradNode* candidate) {
    float** grads = NULL;
    unsigned char* live_nodes = NULL;
    GpuTrainingPlan plan = {0};
    int projected = 0;
    int rc = -1;
    NativeGpuTrainingBackend backend =
        dynamic_autograd_gpu_backend(context ? context->backend : -1);
    int candidate_index;
    int saved_recorded = candidate ? candidate->recorded : 0;
    int moe_indices_slot = -1;
    if (!context || !candidate || !dynamic_autograd_node_schema_supported(
            &candidate->runtime) || context->node_count <= 0) return -1;
    candidate_index = context->node_count - 1;
    if (&context->nodes[candidate_index] != candidate) return -1;
    if (context->backend != VX_PORTABLE_BACKEND_KIND &&
        (backend == NATIVE_GPU_TRAINING_NONE ||
         gpu_training_select_backend() != backend)) return -1;
    if (candidate->runtime.operator_kind == VX_OP_MOE_ROUTER) {
        VolvoxAIAutogradTensor* route_weights = NULL;
        VolvoxAIAutogradTensor* route_indices = NULL;
        for (int output_index = 0; output_index < candidate->runtime.nout;
             output_index++) {
            VxPortKind key = candidate->runtime.outs[output_index].port;
            if (!(key != VX_PORT_INDICES) || !(key != VX_PORT_ROUTE_INDICES))
                route_indices = candidate->outputs[output_index];
            else if (!(key != VX_PORT_WEIGHTS) ||
                     !(key != VX_PORT_ROUTE_WEIGHTS))
                route_weights = candidate->outputs[output_index];
        }
        if (!route_indices ||
            (route_indices->dtype != VOLVOXAI_DTYPE_F32 &&
             route_indices->dtype != VOLVOXAI_DTYPE_I32) ||
            !route_weights || route_weights->dtype != VOLVOXAI_DTYPE_F32)
            return -1;
        /* Native GPU MoE kernels physically bind route indices as F32.  CPU
           also accepts semantic I32 indices through route_index_read(). */
        if (context->backend != VX_PORTABLE_BACKEND_KIND &&
            route_indices->dtype != VOLVOXAI_DTYPE_F32) return -1;
        if (route_indices->dtype == VOLVOXAI_DTYPE_I32)
            moe_indices_slot = route_indices->slot;
    } else if (candidate->runtime.operator_kind == VX_OP_MOE_LINEAR) {
        VolvoxAIAutogradTensor* route_indices = NULL;
        for (int input_index = 0; input_index < candidate->runtime.nin;
             input_index++) {
            VxPortKind key = candidate->runtime.ins[input_index].port;
            if (!(key != VX_PORT_INDICES) || !(key != VX_PORT_ROUTE_INDICES)) {
                route_indices = candidate->inputs[input_index];
                break;
            }
        }
        if (!route_indices ||
            (route_indices->dtype != VOLVOXAI_DTYPE_F32 &&
             route_indices->dtype != VOLVOXAI_DTYPE_I32)) return -1;
        if (context->backend != VX_PORTABLE_BACKEND_KIND &&
            route_indices->dtype != VOLVOXAI_DTYPE_F32) return -1;
        if (route_indices->dtype == VOLVOXAI_DTYPE_I32)
            moe_indices_slot = route_indices->slot;
    }
    grads = (float**)calloc((size_t)context->tensor_count, sizeof(grads[0]));
    live_nodes = (unsigned char*)calloc(
        (size_t)(context->node_count > 0 ? context->node_count : 1), 1);
    if (!grads || !live_nodes) goto done;
    int seeded = 0;
    for (int output_index = 0; output_index < candidate->runtime.nout;
         output_index++) {
        VolvoxAIAutogradTensor* output = candidate->outputs[output_index];
        VxPortKind key = candidate->runtime.outs[output_index].port;
        int differentiable = (candidate->runtime.operator_kind != VX_OP_MOE_ROUTER) ||
            !(key != VX_PORT_WEIGHTS) || !(key != VX_PORT_ROUTE_WEIGHTS);
        if (!output || output->dtype != VOLVOXAI_DTYPE_F32 ||
            !differentiable) continue;
        grads[output->slot] = (float*)calloc(
            (size_t)output->numel, sizeof(float));
        if (!grads[output->slot]) goto done;
        seeded = 1;
    }
    if (!seeded) goto done;
    live_nodes[candidate_index] = 1;
    candidate->recorded = 1;
    if (dynamic_autograd_project(context, live_nodes, 1) != 0) goto done;
    candidate->recorded = saved_recorded;
    projected = 1;
    /* The common planner is descriptor-only on CPU. Temporarily present its
       F32 route-index contract so it can validate all other MoE
       dimensions; actual CPU forward/backward retain the public I32 tensor. */
    if (moe_indices_slot >= 0) g_t[moe_indices_slot].dtype = T_F32;
    if (gpu_build_backward_plan(&plan, grads) != 0 || plan.command_count <= 0)
        goto done;
    if (context->backend != VX_PORTABLE_BACKEND_KIND &&
        gpu_preflight_backward_plan(backend, &plan) != 0) goto done;
    rc = 0;
done:
    /* Projection failure happens before the saved value is restored above. */
    if (candidate && candidate->recorded != saved_recorded)
        candidate->recorded = saved_recorded;
    if (projected) dynamic_autograd_unproject(context);
    gpu_training_plan_free(&plan);
    if (grads) {
        for (int index = 0; index < context->tensor_count; index++)
            free(grads[index]);
    }
    free(grads);
    free(live_nodes);
    return rc;
}

uint32_t volvoxai_autograd_abi_version(void) {
    return VOLVOXAI_AUTOGRAD_ABI_VERSION;
}

VolvoxAIAutogradContext* volvoxai_autograd_context_create(
        const volvoxai_autograd_options_t* options) {
    volvoxai_autograd_options_t defaults = VOLVOXAI_AUTOGRAD_OPTIONS_INIT;
    const volvoxai_autograd_options_t* selected = options ? options : &defaults;
    if (selected->struct_size != sizeof(*selected) ||
        volvoxai_engine_model_route_lease_active()) return NULL;
    volvoxai_engine_model_lock();
    int current_backend = dynamic_autograd_current_backend();
    int requested_backend = selected->backend == VOLVOXAI_AUTOGRAD_CURRENT_BACKEND
        ? current_backend : selected->backend;
    VolvoxAIAutogradContext* context = NULL;
    if (g_dynamic_autograd_context || g_loaded || g_nt || g_nn ||
        g_active_row != -1 || g_prefix_rows != 0 || g_execution_row != -1 ||
        volvoxai_engine_model_route_lease_active() ||
        requested_backend != current_backend ||
        !dynamic_autograd_backend_supported(requested_backend)) goto done;
    context = (VolvoxAIAutogradContext*)calloc(1, sizeof(*context));
    if (!context) goto done;
    context->tensors = (VolvoxAIAutogradTensor**)calloc(
        VX_DYNAMIC_AUTOGRAD_TENSOR_CAPACITY, sizeof(context->tensors[0]));
    context->nodes = (VxDynamicAutogradNode*)calloc(
        VX_DYNAMIC_AUTOGRAD_NODE_CAPACITY, sizeof(context->nodes[0]));
    if (!context->tensors || !context->nodes) {
        free(context->tensors); free(context->nodes); free(context);
        context = NULL;
        goto done;
    }
    context->magic = VX_AUTOGRAD_CONTEXT_MAGIC;
    context->backend = requested_backend;
    context->training_counter = selected->training_counter;
    context->next_tensor_id = 1;
    g_dynamic_autograd_context = context;
done:
    volvoxai_engine_model_unlock();
    return context;
}

void volvoxai_autograd_context_destroy(VolvoxAIAutogradContext* context) {
    if (!context) return;
    volvoxai_engine_model_lock();
    if (!dynamic_autograd_context_identity_valid(context)) {
        volvoxai_engine_model_unlock();
        return;
    }
    if (!g_loaded) {
        vx_runtime_backend_reset();
        dynamic_autograd_free_all_node_caches();
    }
    for (int index = 0; index < context->node_count; index++)
        dynamic_autograd_node_clear(&context->nodes[index]);
    for (int index = 0; index < context->handle_count; index++) {
        VolvoxAIAutogradTensor* tensor = context->handles[index];
        if (tensor) {
            if (tensor->alive) dynamic_autograd_invalidate_tensor(tensor);
            tensor->magic = 0;
            tensor->context = NULL;
            free(tensor);
        }
    }
    free(context->handles);
    free(context->tensors);
    free(context->nodes);
    context->magic = 0;
    g_dynamic_autograd_context = NULL;
    free(context);
    volvoxai_engine_model_unlock();
}

int volvoxai_autograd_context_backend(
        const VolvoxAIAutogradContext* context) {
    volvoxai_engine_model_lock();
    int backend = dynamic_autograd_context_identity_valid(context)
        ? context->backend : -1;
    volvoxai_engine_model_unlock();
    return backend;
}

VolvoxAIAutogradTensor* volvoxai_autograd_tensor_create(
        VolvoxAIAutogradContext* context,
        const volvoxai_autograd_tensor_spec_t* spec) {
    if (!spec || spec->struct_size != sizeof(*spec)) return NULL;
    volvoxai_engine_model_lock();
    VolvoxAIAutogradTensor* tensor = NULL;
    if (dynamic_autograd_context_valid(context) && !g_loaded && !g_nt && !g_nn)
        tensor = dynamic_autograd_tensor_create_locked(
            context, spec->dtype, spec->shape, spec->ndim, spec->data,
            spec->nbytes, spec->requires_grad, 1);
    volvoxai_engine_model_unlock();
    return tensor;
}

VolvoxAIAutogradTensor* volvoxai_autograd_tensor_detach(
        VolvoxAIAutogradContext* context,
        const VolvoxAIAutogradTensor* source) {
    volvoxai_engine_model_lock();
    VolvoxAIAutogradTensor* tensor = NULL;
    if (dynamic_autograd_tensor_valid(context, source) && !g_loaded && !g_nt &&
        !g_nn && dynamic_autograd_materialize_tensor(source) == 0) {
        int32_t shape[8] = {0};
        for (int dimension = 0; dimension < source->ndim; dimension++)
            shape[dimension] = source->shape[dimension];
        tensor = dynamic_autograd_tensor_create_locked(
            context, source->dtype, shape, source->ndim, source->data,
            (uint64_t)source->numel * source->elem_size, 0, 1);
    }
    volvoxai_engine_model_unlock();
    return tensor;
}

int volvoxai_autograd_tensor_set_requires_grad(
        VolvoxAIAutogradTensor* tensor, int requires_grad) {
    volvoxai_engine_model_lock();
    VolvoxAIAutogradContext* context = tensor ? tensor->context : NULL;
    int rc = -1;
    int requested = requires_grad ? 1 : 0;
    int already_used = 0;
    if (dynamic_autograd_tensor_valid(context, tensor) &&
        tensor->requires_grad != requested) {
        for (int node_index = 0; node_index < context->node_count &&
             !already_used; node_index++) {
            VxDynamicAutogradNode* node = &context->nodes[node_index];
            for (int input_index = 0; input_index < node->runtime.nin;
                 input_index++) {
                if (node->inputs[input_index] == tensor) {
                    already_used = 1;
                    break;
                }
            }
        }
    }
    if (dynamic_autograd_tensor_valid(context, tensor) && tensor->is_leaf &&
        tensor->dtype == VOLVOXAI_DTYPE_F32 && !already_used) {
        tensor->requires_grad = requested;
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_tensor_set_data(VolvoxAIAutogradTensor* tensor,
                                      const void* data, uint64_t nbytes) {
    if (!data) return -1;
    volvoxai_engine_model_lock();
    VolvoxAIAutogradContext* context = tensor ? tensor->context : NULL;
    int rc = -1;
    int saved_by_live_graph = 0;
    if (dynamic_autograd_tensor_valid(context, tensor)) {
        for (int node_index = 0; node_index < context->node_count &&
             !saved_by_live_graph; node_index++) {
            VxDynamicAutogradNode* node = &context->nodes[node_index];
            if (!node->recorded || node->history_released) continue;
            for (int input_index = 0; input_index < node->runtime.nin;
                 input_index++) {
                if (node->inputs[input_index] == tensor) {
                    saved_by_live_graph = 1;
                    break;
                }
            }
        }
    }
    if (dynamic_autograd_tensor_valid(context, tensor) && tensor->is_leaf &&
        !saved_by_live_graph &&
        (uint64_t)tensor->numel * tensor->elem_size == nbytes) {
        memcpy(tensor->data, data, (size_t)nbytes);
        tensor->version++;
        if (!tensor->version) tensor->version = 1;
        if (vx_runtime_backend_has_graph())
            vx_runtime_backend_mark_host(tensor->data, (size_t)nbytes, 0);
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_tensor_copy_data(const VolvoxAIAutogradTensor* tensor,
                                       void* output, uint64_t nbytes) {
    if (!output) return -1;
    volvoxai_engine_model_lock();
    const VolvoxAIAutogradContext* context = tensor ? tensor->context : NULL;
    int rc = -1;
    if (dynamic_autograd_tensor_valid(context, tensor) &&
        (uint64_t)tensor->numel * tensor->elem_size == nbytes &&
        dynamic_autograd_materialize_tensor(tensor) == 0) {
        memcpy(output, tensor->data, (size_t)nbytes);
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_tensor_copy_grad(const VolvoxAIAutogradTensor* tensor,
                                       float* output, int64_t numel) {
    if (!output) return -1;
    volvoxai_engine_model_lock();
    const VolvoxAIAutogradContext* context = tensor ? tensor->context : NULL;
    int rc = -1;
    if (dynamic_autograd_tensor_valid(context, tensor) && tensor->is_leaf &&
        tensor->requires_grad && tensor->grad && numel == tensor->numel) {
        memcpy(output, tensor->grad, (size_t)tensor->numel * sizeof(float));
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_tensor_info(const VolvoxAIAutogradTensor* tensor,
                                  int64_t* numel, int32_t* shape,
                                  int32_t* ndim, int32_t* dtype,
                                  int32_t* requires_grad, int32_t* is_leaf) {
    volvoxai_engine_model_lock();
    const VolvoxAIAutogradContext* context = tensor ? tensor->context : NULL;
    int rc = -1;
    if (dynamic_autograd_tensor_valid(context, tensor)) {
        if (numel) *numel = tensor->numel;
        if (shape)
            for (int dimension = 0; dimension < tensor->ndim; dimension++)
                shape[dimension] = tensor->shape[dimension];
        if (ndim) *ndim = tensor->ndim;
        if (dtype) *dtype = tensor->dtype;
        if (requires_grad) *requires_grad = tensor->requires_grad;
        if (is_leaf) *is_leaf = tensor->is_leaf;
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_tensor_release(VolvoxAIAutogradTensor* tensor) {
    if (!tensor) return -1;
    volvoxai_engine_model_lock();
    VolvoxAIAutogradContext* context = tensor->context;
    int rc = -1;
    if (!dynamic_autograd_context_valid(context) ||
        tensor->magic != VX_AUTOGRAD_TENSOR_MAGIC ||
        dynamic_autograd_handle_index(context, tensor) < 0 || g_loaded ||
        g_nt || g_nn) goto done;
    if (!tensor->alive) {
        rc = dynamic_autograd_destroy_handle(context, tensor);
        goto done;
    }
    /* A node owns internal references to all of its inputs and outputs until
       clear_graph().  Never recycle those addresses: explicit release is the
       point at which the caller promises no stale pointer will be used. */
    for (int node_index = 0; node_index < context->node_count; node_index++) {
        VxDynamicAutogradNode* node = &context->nodes[node_index];
        for (int input_index = 0; input_index < node->runtime.nin; input_index++)
            if (node->inputs[input_index] == tensor) goto done;
        for (int output_index = 0; output_index < node->runtime.nout;
             output_index++)
            if (node->outputs[output_index] == tensor) goto done;
    }
    if (tensor->slot < 0 || tensor->slot >= context->tensor_count ||
        context->tensors[tensor->slot] != tensor) goto done;
    /* Resetting a backend address table before freeing this host key is safe
       only after every other active value is host-authoritative. */
    if (dynamic_autograd_materialize_prefix(context, context->tensor_count) != 0)
        goto done;
    /* A dynamic leaf may have occupied a semantic weight binding.  There is no
       loaded model whose resident slot must survive, and this host key is
       about to be freed, so invalidate the complete backend address table. */
    vx_runtime_backend_reset();
    int removed_slot = tensor->slot;
    for (int slot = removed_slot + 1; slot < context->tensor_count; slot++) {
        context->tensors[slot - 1] = context->tensors[slot];
        context->tensors[slot - 1]->slot = slot - 1;
    }
    context->tensors[--context->tensor_count] = NULL;
    rc = dynamic_autograd_destroy_handle(context, tensor);
done:
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_apply(VolvoxAIAutogradContext* context,
                            const volvoxai_autograd_op_t* descriptor,
                            VolvoxAIAutogradTensor** outputs,
                            int32_t output_capacity) {
    if (outputs && output_capacity > 0)
        for (int index = 0; index < output_capacity; index++) outputs[index] = NULL;
    if (!descriptor || descriptor->struct_size != sizeof(*descriptor) ||
        !descriptor->operator_kind || !outputs || descriptor->input_count <= 0 ||
        descriptor->output_count <= 0 ||
        output_capacity != descriptor->output_count || !descriptor->inputs ||
        !descriptor->outputs ||
        !dynamic_autograd_operator_supported(descriptor->operator_kind)) return -1;
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int rc = -1;
    int projected = 0;
    int forward_started = 0;
    int transients_reset = 0;
    uint32_t unrecorded_counter = context
        ? context->unrecorded_counter + 1u : 0u;
    int old_tensor_count = context ? context->tensor_count : 0;
    int old_node_count = context ? context->node_count : 0;
    int saved_training_mode = g_native_training_mode;
    uint32_t saved_training_counter = g_native_training_counter;
    cJSON* params = NULL;
    if (!dynamic_autograd_context_valid(context) || g_loaded || g_nt || g_nn ||
        context->node_count >= VX_DYNAMIC_AUTOGRAD_NODE_CAPACITY ||
        dynamic_autograd_current_backend() != context->backend) goto done;
    if (descriptor->params_json) {
        params = cJSON_Parse(descriptor->params_json);
        if (!cJSON_IsObject(params)) goto done;
    }
    VxDynamicAutogradNode* dynamic_node = &context->nodes[context->node_count];
    memset(dynamic_node, 0, sizeof(*dynamic_node));
    Node* node = &dynamic_node->runtime;
    node->operator_kind = descriptor->operator_kind;
    node->params = params;
    node->owns_params = params ? 1 : 0;
    params = NULL;
    if (vx_node_reserve_refs(node, descriptor->input_count, descriptor->output_count)) goto rollback;
    dynamic_node->inputs = calloc((size_t)descriptor->input_count, sizeof(*dynamic_node->inputs));
    dynamic_node->outputs = calloc((size_t)descriptor->output_count, sizeof(*dynamic_node->outputs));
    dynamic_node->input_versions = calloc((size_t)descriptor->input_count, sizeof(*dynamic_node->input_versions));
    if (!dynamic_node->inputs || !dynamic_node->outputs || !dynamic_node->input_versions) goto rollback;
    int propagates_grad = 0;
    for (int index = 0; index < descriptor->input_count; index++) {
        const volvoxai_autograd_input_t* binding = &descriptor->inputs[index];
        const VolvoxAIAutogradTensor* input = binding->tensor;
        if (!binding->key || !binding->key[0] ||
            strlen(binding->key) >= sizeof(node->ins[0].key) ||
            !dynamic_autograd_tensor_valid(context, input) ||
            (input->requires_grad &&
             !dynamic_autograd_input_role_differentiable(
                 descriptor->operator_kind, vx_port_kind_from_name(binding->key)))) goto rollback;
        for (int prior = 0; prior < index; prior++)
            if (!strcmp(binding->key, node->ins[prior].key)) goto rollback;
        memcpy(node->ins[index].key, binding->key, strlen(binding->key) + 1);
        node->ins[index].port = vx_port_kind_from_name(binding->key);
        memcpy(node->ins[index].name, input->name, strlen(input->name) + 1);
        dynamic_node->inputs[index] = (VolvoxAIAutogradTensor*)input;
        dynamic_node->input_versions[index] = input->version;
        if (input->requires_grad) propagates_grad = 1;
    }
    node->nin = descriptor->input_count;
    dynamic_node->recorded = propagates_grad && context->no_grad_depth == 0;
    for (int index = 0; index < descriptor->output_count; index++) {
        const volvoxai_autograd_output_spec_t* output_spec =
            &descriptor->outputs[index];
        if (!output_spec->key || !output_spec->key[0] ||
            strlen(output_spec->key) >= sizeof(node->outs[0].key)) goto rollback;
        for (int prior = 0; prior < index; prior++)
            if (!strcmp(output_spec->key, node->outs[prior].key)) goto rollback;
        int output_requires_grad = dynamic_node->recorded &&
            output_spec->dtype == VOLVOXAI_DTYPE_F32;
        VxPortKind output_port = vx_port_kind_from_name(output_spec->key);
        if ((descriptor->operator_kind == VX_OP_MOE_ROUTER) &&
            (output_port == VX_PORT_INDICES ||
             output_port == VX_PORT_ROUTE_INDICES))
            output_requires_grad = 0;
        VolvoxAIAutogradTensor* output = dynamic_autograd_tensor_create_locked(
            context, output_spec->dtype, output_spec->shape, output_spec->ndim,
            NULL, 0, output_requires_grad, 0);
        if (!output) goto rollback;
        node->outs[index].port = output_port;
        memcpy(node->outs[index].key, output_spec->key,
               strlen(output_spec->key) + 1);
        memcpy(node->outs[index].name, output->name, strlen(output->name) + 1);
        dynamic_node->outputs[index] = output;
        if (index == 0) memcpy(node->out, output->name, strlen(output->name) + 1);
    }
    node->nout = descriptor->output_count;
    if (volvoxai_engine_compile_node_params(node) != 0 ||
        node->parsed_params.parsed != 1) goto rollback;
    context->node_count++;
    g_native_training_mode = 1;
    g_native_training_counter = dynamic_node->recorded
        ? context->training_counter
        : context->training_counter + unrecorded_counter;
    if (dynamic_autograd_validate_candidate(context, dynamic_node) != 0)
        goto rollback_committed;
    if (dynamic_autograd_project(context, NULL, 0) != 0) goto rollback_committed;
    projected = 1;
    g_native_training_mode = 1;
    g_native_training_counter = dynamic_node->recorded
        ? context->training_counter
        : context->training_counter + unrecorded_counter;
    if (vx_runtime_backend_has_graph()) {
        /* Zero generation is the private eager/non-loaded sentinel; it keeps
           these individual eager operations out of inference graph replay. */
        vx_runtime_backend_begin_forward(0, 0);
        forward_started = 1;
        if (vx_runtime_backend_prepare_forward() != 0) goto forward_done;
    }
    g_dynamic_autograd_forward_backend = -1;
    g_dynamic_autograd_forward_capture = 1;
    rc = run_node(&g_n[context->node_count - 1], context->node_count - 1, 1);
    g_dynamic_autograd_forward_capture = 0;
    if (rc == 0 && g_dynamic_autograd_forward_backend != context->backend)
        rc = -1;
forward_done:
    g_dynamic_autograd_forward_capture = 0;
    if (forward_started && vx_runtime_backend_end_forward(rc == 0) != 0) rc = -1;
    forward_started = 0;
    g_native_training_mode = saved_training_mode;
    g_native_training_counter = saved_training_counter;
    dynamic_autograd_unproject(context);
    projected = 0;
    if (rc != 0) goto rollback_committed;
    for (int index = 0; index < descriptor->output_count; index++) {
        outputs[index] = dynamic_node->outputs[index];
        if (!dynamic_node->recorded) outputs[index]->is_leaf = 1;
    }
    if (!dynamic_node->recorded) {
        /* An operation outside gradient recording is an immediate graph break,
           not a permanent tape node.  Its outputs remain live detached leaves,
           while the descriptor and any node-indexed cache can be retired now.
           A separate counter keeps repeated no-grad Dropout invocations from
           reusing the same mask after this storage index is reused. */
        context->unrecorded_counter = unrecorded_counter;
        context->node_count = old_node_count;
        dynamic_autograd_free_node_cache(old_node_count);
        dynamic_autograd_node_clear(dynamic_node);
    }
    goto done;

rollback_committed:
    if (forward_started) (void)vx_runtime_backend_end_forward(0);
    g_native_training_mode = saved_training_mode;
    g_native_training_counter = saved_training_counter;
    if (projected) dynamic_autograd_unproject(context);
    dynamic_autograd_rollback_backend(context, old_tensor_count);
    transients_reset = 1;
    if (context->node_count > old_node_count) context->node_count = old_node_count;
    dynamic_autograd_free_node_cache(old_node_count);
rollback:
    if (context && context->tensor_count > old_tensor_count &&
        !transients_reset) {
        dynamic_autograd_rollback_backend(context, old_tensor_count);
        transients_reset = 1;
    }
    cJSON_Delete(params);
    if (old_node_count < VX_DYNAMIC_AUTOGRAD_NODE_CAPACITY) {
        VxDynamicAutogradNode* failed = &context->nodes[old_node_count];
        dynamic_autograd_node_clear(failed);
    }
    while (context->tensor_count > old_tensor_count) {
        VolvoxAIAutogradTensor* output =
            context->tensors[--context->tensor_count];
        context->tensors[context->tensor_count] = NULL;
        (void)dynamic_autograd_destroy_handle(context, output);
    }
    rc = -1;
done:
    g_native_training_mode = saved_training_mode;
    g_native_training_counter = saved_training_counter;
    /* Invalid/non-object params JSON exits before the Node takes ownership. */
    cJSON_Delete(params);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_no_grad_begin(VolvoxAIAutogradContext* context) {
    volvoxai_engine_model_lock();
    int rc = -1;
    if (dynamic_autograd_context_valid(context) && context->no_grad_depth < INT_MAX) {
        context->no_grad_depth++;
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_no_grad_end(VolvoxAIAutogradContext* context) {
    volvoxai_engine_model_lock();
    int rc = -1;
    if (dynamic_autograd_context_valid(context) && context->no_grad_depth > 0) {
        context->no_grad_depth--;
        rc = 0;
    }
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_backward(VolvoxAIAutogradContext* context,
                               const VolvoxAIAutogradTensor* output,
                               const float* seed, int64_t seed_numel,
                               int retain_graph) {
    volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    int rc = -1;
    int projected = 0;
    float** grads = NULL;
    unsigned char* live_nodes = NULL;
    const char** trainable_names = NULL;
    VolvoxAIAutogradTensor** newly_allocated = NULL;
    int newly_allocated_count = 0;
    GpuTrainingPlan plan = {0};
    int saved_training_mode = g_native_training_mode;
    uint32_t saved_training_counter = g_native_training_counter;
    if (!dynamic_autograd_tensor_valid(context, output) || g_loaded || g_nt ||
        g_nn || !output->requires_grad || output->dtype != VOLVOXAI_DTYPE_F32 ||
        output->numel <= 0 || context->no_grad_depth ||
        dynamic_autograd_current_backend() != context->backend ||
        (seed ? seed_numel != output->numel
              : (output->numel != 1 || (seed_numel != 0 && seed_numel != 1))))
        goto done;
    grads = (float**)calloc((size_t)context->tensor_count, sizeof(grads[0]));
    live_nodes = (unsigned char*)calloc(
        (size_t)(context->node_count > 0 ? context->node_count : 1), 1);
    trainable_names = (const char**)calloc(
        (size_t)context->tensor_count, sizeof(trainable_names[0]));
    newly_allocated = (VolvoxAIAutogradTensor**)calloc(
        (size_t)context->tensor_count, sizeof(newly_allocated[0]));
    if (!grads || !live_nodes || !trainable_names || !newly_allocated) goto done;
    grads[output->slot] = (float*)calloc((size_t)output->numel, sizeof(float));
    if (!grads[output->slot]) goto done;
    if (seed) memcpy(grads[output->slot], seed,
                     (size_t)output->numel * sizeof(float));
    else grads[output->slot][0] = 1.0f;
    if (dynamic_autograd_build_reachability(context, output, live_nodes) != 0)
        goto done;
    /* Saved forward values may still be device-authoritative.  Materializing
       once before the backward scope also makes retain_graph correct after a
       GPU training end releases transient graph slots. */
    if (dynamic_autograd_materialize_prefix(context, context->tensor_count) != 0)
        goto done;
    if (dynamic_autograd_project(context, live_nodes, 1) != 0) goto done;
    projected = 1;
    g_native_training_mode = 1;
    g_native_training_counter = context->training_counter;
    if (context->backend == VX_PORTABLE_BACKEND_KIND) {
        rc = backward_supported_graph(grads);
    } else {
        NativeGpuTrainingBackend backend =
            dynamic_autograd_gpu_backend(context->backend);
        if (backend == NATIVE_GPU_TRAINING_NONE ||
            gpu_training_select_backend() != backend ||
            gpu_build_backward_plan(&plan, grads) != 0) {
            rc = -1;
        } else {
            int trainable_count = 0;
            for (int index = 0; index < context->tensor_count; index++) {
                VolvoxAIAutogradTensor* tensor = context->tensors[index];
                if (tensor->is_leaf && tensor->requires_grad && grads[index])
                    trainable_names[trainable_count++] = tensor->name;
            }
            if (plan.command_count == 0) rc = 0;
            else if (gpu_preflight_backward_plan(backend, &plan) != 0) rc = -1;
            else if (trainable_count > 0)
                rc = gpu_execute_preflighted_backward_plan(
                    backend, &plan, grads, trainable_names, trainable_count,
                    GPU_BACKWARD_EXECUTION_SYNC_TRAINABLES_AND_END);
            else {
                if (gpu_training_begin(backend) != 0) rc = -1;
                else {
                    rc = gpu_dispatch_preflighted_backward_plan(backend, &plan);
                    gpu_complete_backward_plan_execution(backend);
                }
            }
        }
    }
    gpu_training_plan_free(&plan);
    g_native_training_mode = saved_training_mode;
    g_native_training_counter = saved_training_counter;
    dynamic_autograd_unproject(context);
    projected = 0;
    if (rc != 0) goto done;
    /* Allocate every missing persistent leaf gradient before changing any
       existing one, preserving all-or-nothing accumulation on OOM. */
    for (int index = 0; index < context->tensor_count; index++) {
        VolvoxAIAutogradTensor* tensor = context->tensors[index];
        if (!tensor->is_leaf || !tensor->requires_grad || !grads[index] ||
            tensor->grad) continue;
        tensor->grad = (float*)calloc((size_t)tensor->numel, sizeof(float));
        if (!tensor->grad) {
            for (int allocated = 0; allocated < newly_allocated_count; allocated++) {
                free(newly_allocated[allocated]->grad);
                newly_allocated[allocated]->grad = NULL;
            }
            rc = -1;
            goto done;
        }
        newly_allocated[newly_allocated_count++] = tensor;
    }
    for (int index = 0; index < context->tensor_count; index++) {
        VolvoxAIAutogradTensor* tensor = context->tensors[index];
        if (!tensor->is_leaf || !tensor->requires_grad || !grads[index]) continue;
        grad_add_full(tensor->grad, grads[index], tensor->numel, 1.0f);
    }
    if (!retain_graph && !output->is_leaf) {
        for (int node_index = 0; node_index < context->node_count; node_index++)
            if (live_nodes[node_index])
                context->nodes[node_index].history_released = 1;
    }
    rc = 0;
done:
    if (projected) dynamic_autograd_unproject(context);
    g_native_training_mode = saved_training_mode;
    g_native_training_counter = saved_training_counter;
    gpu_training_plan_free(&plan);
    if (grads) {
        int count = context && context->tensor_count >= 0 &&
            context->tensor_count <= VX_DYNAMIC_AUTOGRAD_TENSOR_CAPACITY
                ? context->tensor_count : 0;
        for (int index = 0; index < count; index++) free(grads[index]);
    }
    free(grads);
    free(live_nodes);
    free(trainable_names);
    free(newly_allocated);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_zero_grad(VolvoxAIAutogradContext* context,
                                VolvoxAIAutogradTensor* tensor) {
    volvoxai_engine_model_lock();
    int rc = -1;
    if (!dynamic_autograd_context_valid(context)) goto done;
    if (tensor) {
        if (!dynamic_autograd_tensor_valid(context, tensor) || !tensor->is_leaf)
            goto done;
        if (tensor->grad)
            memset(tensor->grad, 0, (size_t)tensor->numel * sizeof(float));
    } else {
        for (int index = 0; index < context->tensor_count; index++) {
            VolvoxAIAutogradTensor* leaf = context->tensors[index];
            if (leaf->is_leaf && leaf->grad)
                memset(leaf->grad, 0,
                       (size_t)leaf->numel * sizeof(float));
        }
    }
    rc = 0;
done:
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_autograd_clear_graph(VolvoxAIAutogradContext* context) {
    volvoxai_engine_model_lock();
    int rc = -1;
    if (!dynamic_autograd_context_valid(context) || g_loaded || g_nt || g_nn ||
        context->no_grad_depth) goto done;
    /* Keep leaf values coherent before invalidating device slots that may be
       keyed by soon-to-be-freed non-leaf addresses. */
    for (int index = 0; index < context->tensor_count; index++) {
        VolvoxAIAutogradTensor* tensor = context->tensors[index];
        if (tensor->is_leaf && dynamic_autograd_materialize_tensor(tensor) != 0)
            goto done;
    }
    /* Non-leaf eager values can themselves be consumed as later weights.
       Those slots look persistent to CUDA even though their host allocations
       are about to be invalidated.  A dynamic context excludes loaded models,
       making a full slot reset both safe and required here. */
    vx_runtime_backend_reset();
    dynamic_autograd_free_all_node_caches();
    for (int index = 0; index < context->node_count; index++) {
        dynamic_autograd_node_clear(&context->nodes[index]);
    }
    int retained = 0;
    for (int index = 0; index < context->tensor_count; index++) {
        VolvoxAIAutogradTensor* tensor = context->tensors[index];
        if (tensor->is_leaf) {
            tensor->slot = retained;
            context->tensors[retained++] = tensor;
        } else {
            dynamic_autograd_invalidate_tensor(tensor);
        }
    }
    for (int index = retained; index < context->tensor_count; index++)
        context->tensors[index] = NULL;
    context->tensor_count = retained;
    context->node_count = 0;
    context->training_counter++;
    rc = 0;
done:
    volvoxai_engine_model_unlock();
    return rc;
}
