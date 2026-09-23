/*
 * GPU backward plan builder and dispatch for native training.
 *
 * Fragment of the training runtime translation unit: included only from
 * training_runtime.inc, never compiled or included on its own. Selects a GPU
 * training backend, records per-operator dispatches into a GpuTrainingPlan, and
 * executes/synchronizes them. Falls back to the CPU kernels when unsupported.
 */

int volvoxai_engine_require_training_backend(int backend) {
    if (backend != NATIVE_GPU_TRAINING_NONE &&
        backend != NATIVE_GPU_TRAINING_VULKAN &&
        backend != NATIVE_GPU_TRAINING_OPENGL &&
        backend != NATIVE_GPU_TRAINING_METAL &&
        backend != NATIVE_GPU_TRAINING_CUDA) return -1;
    if ((backend == NATIVE_GPU_TRAINING_VULKAN && !VOLVOXAI_ENABLE_VULKAN) ||
        (backend == NATIVE_GPU_TRAINING_OPENGL && !VOLVOXAI_ENABLE_OPENGL) ||
        (backend == NATIVE_GPU_TRAINING_METAL && !VOLVOXAI_ENABLE_METAL) ||
        (backend == NATIVE_GPU_TRAINING_CUDA && !VOLVOXAI_ENABLE_CUDA)) return -1;
    volvoxai_engine_model_lock();
    g_required_training_backend = (NativeGpuTrainingBackend)backend;
    volvoxai_engine_model_unlock();
    return 0;
}

int volvoxai_engine_last_training_backend(void) {
    volvoxai_engine_model_lock();
    int backend = (int)g_last_training_backend;
    volvoxai_engine_model_unlock();
    return backend;
}

enum {
    GPU_TRAINING_READ = 1,
    GPU_TRAINING_READ_WRITE = 3,
    GPU_TRAINING_MAX_BINDINGS = 13,
    GPU_TRAINING_CUDA_BASIC_SCALAR_ITEMS_PER_PARTIAL = 4096
};

typedef struct {
    const char* shader;
    const char* entry;
    void* hosts[GPU_TRAINING_MAX_BINDINGS];
    size_t bytes[GPU_TRAINING_MAX_BINDINGS];
    unsigned char access[GPU_TRAINING_MAX_BINDINGS];
    unsigned char is_weight[GPU_TRAINING_MAX_BINDINGS];
    int binding_count;
    uint32_t groups[3];
} GpuTrainingCommand;

typedef struct {
    GpuTrainingCommand* commands;
    int command_count;
    int command_capacity;
    void** owned;
    int owned_count;
    int owned_capacity;
    /* Present only for NODES captures. Numeric commands keep their old layout. */
    VxTraceWork* origins;
    size_t origin_capacity;
    VxTraceWork origin;
} GpuTrainingPlan;

static void gpu_training_plan_trace(GpuTrainingPlan* plan) {
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (plan->origins || !vx_trace_nodes(scope)) return;
    plan->origin_capacity = vx_trace_device_capacity(scope, 1024);
    if (plan->origin_capacity) plan->origins = calloc(plan->origin_capacity, sizeof(*plan->origins));
    if (!plan->origins) { plan->origin_capacity = 0; vx_trace_drop(scope); }
    plan->origin.index = -1;
}
static VxTraceWork gpu_training_command_origin(const GpuTrainingPlan* plan, int index) {
    if (plan->origins && index >= 0 && (size_t)index < plan->origin_capacity)
        return plan->origins[index];
    return (VxTraceWork){.index = -1};
}

typedef enum {
    /* Preserve the original executor contract: copy every trainable gradient
       written by the plan back to its host allocation, then close the backend
       training scope. */
    GPU_BACKWARD_EXECUTION_SYNC_TRAINABLES_AND_END = 0,
    /* CUDA-only train-step plumbing may keep graph slots device-authoritative
       after dispatch.  A successful call in this mode deliberately leaves the
       CUDA training scope open until gpu_complete_backward_plan_execution(). */
    GPU_BACKWARD_EXECUTION_CUDA_DEVICE_RESIDENT = 1
} GpuBackwardExecutionMode;

#if VOLVOXAI_ENABLE_VULKAN
static int gpu_vulkan_tensor_requirement_cmp(const void* lhs, const void* rhs) {
    uintptr_t a = (uintptr_t)((const VkTrainingTensorRequirement*)lhs)->host;
    uintptr_t b = (uintptr_t)((const VkTrainingTensorRequirement*)rhs)->host;
    return a < b ? -1 : (a > b ? 1 : 0);
}
#endif

static NativeGpuTrainingBackend gpu_training_select_backend(void) {
#if VOLVOXAI_ENABLE_VULKAN
    if (g_use_vulkan && vk_training_available()) return NATIVE_GPU_TRAINING_VULKAN;
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (g_use_opengl && opengl_training_available()) return NATIVE_GPU_TRAINING_OPENGL;
#endif
#if VOLVOXAI_ENABLE_METAL
    if (g_use_metal && metal_training_available()) return NATIVE_GPU_TRAINING_METAL;
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda && cuda_training_available()) return NATIVE_GPU_TRAINING_CUDA;
#endif
    return NATIVE_GPU_TRAINING_NONE;
}

static int gpu_training_begin(NativeGpuTrainingBackend backend) {
#if VOLVOXAI_ENABLE_VULKAN
    if (backend == NATIVE_GPU_TRAINING_VULKAN) return vk_training_begin();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (backend == NATIVE_GPU_TRAINING_OPENGL) return opengl_training_begin();
#endif
#if VOLVOXAI_ENABLE_METAL
    if (backend == NATIVE_GPU_TRAINING_METAL) return metal_training_begin();
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (backend == NATIVE_GPU_TRAINING_CUDA) return cuda_training_begin();
#endif
    return -1;
}

static int gpu_training_dispatch(NativeGpuTrainingBackend backend, const GpuTrainingCommand* command) {
    if (!command) return -1;
#if VOLVOXAI_ENABLE_VULKAN
    if (backend == NATIVE_GPU_TRAINING_VULKAN) {
        return vk_training_dispatch(command->shader, command->entry, command->hosts, command->bytes,
                                    command->access, command->is_weight, command->binding_count,
                                    command->groups[0], command->groups[1], command->groups[2]);
    }
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (backend == NATIVE_GPU_TRAINING_OPENGL) {
        return opengl_training_dispatch(command->shader, command->entry, command->hosts, command->bytes,
                                        command->access, command->is_weight, command->binding_count,
                                        command->groups[0], command->groups[1], command->groups[2]);
    }
#endif
#if VOLVOXAI_ENABLE_METAL
    if (backend == NATIVE_GPU_TRAINING_METAL) {
        return metal_training_dispatch(command->shader, command->entry, command->hosts, command->bytes,
                                       command->access, command->is_weight, command->binding_count,
                                       command->groups[0], command->groups[1], command->groups[2]);
    }
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (backend == NATIVE_GPU_TRAINING_CUDA) {
        return cuda_training_dispatch(command->shader, command->entry, command->hosts, command->bytes,
                                      command->access, command->is_weight, command->binding_count,
                                      command->groups[0], command->groups[1], command->groups[2]);
    }
#endif
    return -1;
}

static int gpu_training_command_supported(NativeGpuTrainingBackend backend,
                                          const GpuTrainingCommand* command) {
    if (!command) return 0;
#if VOLVOXAI_ENABLE_VULKAN
    if (backend == NATIVE_GPU_TRAINING_VULKAN) {
        return vk_training_supports(command->shader, command->entry, command->bytes,
                                    command->binding_count, command->groups[0],
                                    command->groups[1], command->groups[2]);
    }
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (backend == NATIVE_GPU_TRAINING_OPENGL) {
        return opengl_training_supports(command->shader, command->entry, command->bytes,
                                        command->binding_count, command->groups[0],
                                        command->groups[1], command->groups[2]);
    }
#endif
#if VOLVOXAI_ENABLE_METAL
    if (backend == NATIVE_GPU_TRAINING_METAL) {
        return metal_training_supports(command->shader, command->entry, command->bytes,
                                       command->binding_count, command->groups[0],
                                       command->groups[1], command->groups[2]);
    }
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (backend == NATIVE_GPU_TRAINING_CUDA) {
        return cuda_training_preflight(command->shader, command->entry,
                                       command->bytes,
                                       command->binding_count,
                                       command->groups[0], command->groups[1],
                                       command->groups[2]);
    }
#endif
    return 0;
}

static int gpu_training_sync(NativeGpuTrainingBackend backend, void* host, size_t bytes) {
#if VOLVOXAI_ENABLE_VULKAN
    if (backend == NATIVE_GPU_TRAINING_VULKAN) return vk_training_sync(host, bytes);
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (backend == NATIVE_GPU_TRAINING_OPENGL) return opengl_training_sync(host, bytes);
#endif
#if VOLVOXAI_ENABLE_METAL
    if (backend == NATIVE_GPU_TRAINING_METAL) return metal_training_sync(host, bytes);
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (backend == NATIVE_GPU_TRAINING_CUDA) return cuda_training_sync(host, bytes);
#endif
    return -1;
}

static void gpu_training_end(NativeGpuTrainingBackend backend) {
#if VOLVOXAI_ENABLE_VULKAN
    if (backend == NATIVE_GPU_TRAINING_VULKAN) vk_training_end();
#endif
#if VOLVOXAI_ENABLE_OPENGL
    if (backend == NATIVE_GPU_TRAINING_OPENGL) opengl_training_end();
#endif
#if VOLVOXAI_ENABLE_METAL
    if (backend == NATIVE_GPU_TRAINING_METAL) metal_training_end();
#endif
#if VOLVOXAI_ENABLE_CUDA
    if (backend == NATIVE_GPU_TRAINING_CUDA) cuda_training_end();
#endif
    (void)backend;
}

static void gpu_training_plan_free(GpuTrainingPlan* plan) {
    if (!plan) return;
    for (int i = 0; i < plan->owned_count; i++) free(plan->owned[i]);
    free(plan->owned);
    free(plan->commands);
    free(plan->origins);
    memset(plan, 0, sizeof(*plan));
}

static void* gpu_training_plan_alloc(GpuTrainingPlan* plan, size_t bytes) {
    if (!plan || bytes == 0) return NULL;
    if (plan->owned_count == plan->owned_capacity) {
        int capacity = plan->owned_capacity ? plan->owned_capacity * 2 : 32;
        void** owned = (void**)realloc(plan->owned, (size_t)capacity * sizeof(*owned));
        if (!owned) return NULL;
        plan->owned = owned;
        plan->owned_capacity = capacity;
    }
    void* allocation = calloc(1, bytes);
    if (!allocation) return NULL;
    plan->owned[plan->owned_count++] = allocation;
    return allocation;
}

static int gpu_training_host_is_weight(void* host) {
    if (!host || host == g_gpu_training_dummy) return 0;
    for (int i = 0; i < g_nt; i++) {
        /* owns describes allocation lifetime, not model semantics: graph
           inputs can borrow storage and dynamically added model weights can
           own it.  Identify persistent weights by the authoritative model
           tensor name/storage relationship instead.  Keep scanning aliases so
           a non-weight tensor sharing the allocation cannot hide the weight. */
        if (g_t[i].data == host &&
            volvoxai_engine_tensor_is_model_weight_locked(g_t[i].name)) return 1;
    }
    return 0;
}

static int gpu_training_plan_add(GpuTrainingPlan* plan, const char* shader, const char* entry,
                                 void* const* hosts, const size_t* bytes,
                                 const unsigned char* access, int binding_count,
                                 uint32_t groups_x, uint32_t groups_y, uint32_t groups_z) {
    if (!plan || !shader || !entry || !hosts || !bytes || !access ||
        binding_count <= 0 || binding_count > GPU_TRAINING_MAX_BINDINGS ||
        groups_x == 0 || groups_y == 0 || groups_z == 0) return -1;
    if (plan->command_count == plan->command_capacity) {
        int capacity = plan->command_capacity ? plan->command_capacity * 2 : 64;
        GpuTrainingCommand* commands = (GpuTrainingCommand*)realloc(
            plan->commands, (size_t)capacity * sizeof(*commands));
        if (!commands) return -1;
        plan->commands = commands;
        plan->command_capacity = capacity;
    }
    if (plan->origins && (size_t)plan->command_count < plan->origin_capacity)
        plan->origins[plan->command_count] = plan->origin;
    else if (plan->origins) vx_trace_drop(vx_engine_state_current()->profiling);
    GpuTrainingCommand* command = &plan->commands[plan->command_count++];
    memset(command, 0, sizeof(*command));
    command->shader = shader;
    command->entry = entry;
    command->binding_count = binding_count;
    command->groups[0] = groups_x;
    command->groups[1] = groups_y;
    command->groups[2] = groups_z;
    for (int i = 0; i < binding_count; i++) {
        if (!hosts[i] || bytes[i] == 0 || (access[i] != GPU_TRAINING_READ &&
            access[i] != GPU_TRAINING_READ_WRITE)) return -1;
        command->hosts[i] = hosts[i];
        command->bytes[i] = bytes[i];
        command->access[i] = access[i];
        command->is_weight[i] = (unsigned char)gpu_training_host_is_weight(hosts[i]);
    }
    return 0;
}

static uint32_t gpu_groups_64(long count) {
    if (count <= 0 || (uint64_t)count > (uint64_t)UINT32_MAX * 64u) return 0;
    return (uint32_t)(((uint64_t)count + 63u) / 64u);
}

static uint32_t gpu_groups_n(long count, uint32_t group_size) {
    if (count <= 0 || group_size == 0 ||
        (uint64_t)count > (uint64_t)UINT32_MAX * group_size) return 0;
    return (uint32_t)(((uint64_t)count + group_size - 1u) / group_size);
}

static uint32_t f32_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    return bits;
}

static void gpu_binding_tensor(void** hosts, size_t* bytes, int binding, T* tensor) {
    hosts[binding] = tensor ? (void*)tensor->data : (void*)g_gpu_training_dummy;
    bytes[binding] = tensor ? tensor_bytes(tensor) : sizeof(g_gpu_training_dummy);
}

static void gpu_binding_gradient(void** hosts, size_t* bytes, int binding, float* gradient, long numel) {
    hosts[binding] = gradient ? (void*)gradient : (void*)g_gpu_training_dummy;
    bytes[binding] = gradient && numel > 0 ? (size_t)numel * sizeof(float) : sizeof(g_gpu_training_dummy);
}

static void gpu_binding_params(void** hosts, size_t* bytes, int binding, void* params, size_t params_bytes) {
    hosts[binding] = params;
    bytes[binding] = params_bytes;
}

static int gpu_training_sync_all_f32_tensors(void) {
    for (int i = 0; i < g_nt; i++) {
        if (g_t[i].dtype == T_F32 && g_t[i].data && g_t[i].numel > 0 &&
            vk_sync_host_tensor(&g_t[i]) != 0) return -1;
    }
    return 0;
}

static void gpu_bindings_init(void** hosts, size_t* bytes, unsigned char* access, int count) {
    for (int i = 0; i < count; i++) {
        hosts[i] = g_gpu_training_dummy;
        bytes[i] = sizeof(g_gpu_training_dummy);
        access[i] = GPU_TRAINING_READ;
    }
}

static int gpu_plan_linear(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    T* bias = nin(n, VX_PORT_BIAS);
    if (!in || !weight || !out || !go || in->dtype != T_F32 || weight->dtype != T_F32 ||
        out->dtype != T_F32 || in->ndim <= 0 || out->ndim <= 0 ||
        (bias && bias->dtype != T_F32)) return -1;
    int d_in = in->shape[in->ndim - 1];
    int d_out = out->shape[out->ndim - 1];
    long rows = d_out > 0 ? out->numel / d_out : 0;
    int out_in = 0;
    if (d_in <= 0 || d_out <= 0 || rows <= 0 || in->numel != rows * d_in ||
        weight->numel != (long)d_in * d_out || (bias && bias->numel != d_out) ||
        linear_node_weight_layout(n, in, out, weight, &out_in) != 0) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
    if (!params || rows > UINT32_MAX) return -1;
    params[0] = (uint32_t)rows; params[1] = (uint32_t)d_in; params[2] = (uint32_t)d_out;
    params[3] = bias ? 1u : 0u;
    params[4] = out_in ? 0u : 1u;
    params[5] = out_in ? 0u : 1u;
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    float* gb = bias ? grad_for(grads, bias) : NULL;
    if (!gi || !gw || (bias && !gb)) return -1;
    void* hosts[7]; size_t bytes[7]; unsigned char access[7];
    gpu_bindings_init(hosts, bytes, access, 7);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 4, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 5, gb, bias ? bias->numel : 0);
    gpu_binding_params(hosts, bytes, 6, params, 8 * sizeof(uint32_t));
    access[3] = access[4] = access[5] = GPU_TRAINING_READ_WRITE;
    uint32_t input_groups_x = gpu_groups_n(d_in, 16);
    uint32_t input_groups_y = gpu_groups_n(rows, 16);
    uint32_t weight_groups_x = gpu_groups_n(d_out, 16);
    uint32_t weight_groups_y = gpu_groups_n(d_in, 16);
    uint32_t bias_groups = gpu_groups_64(d_out);
    if (!input_groups_x || !input_groups_y || !weight_groups_x || !weight_groups_y || !bias_groups ||
        gpu_training_plan_add(plan, "matMulBackward", "input_main", hosts, bytes, access, 7,
                              input_groups_x, input_groups_y, 1) != 0 ||
        gpu_training_plan_add(plan, "matMulBackward", "weight_main", hosts, bytes, access, 7,
                              weight_groups_x, weight_groups_y, 1) != 0) return -1;
    if (bias && gpu_training_plan_add(plan, "matMulBackward", "bias_main", hosts, bytes, access, 7,
                                      bias_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_binary(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go,
                           uint32_t kind) {
    T* a_tensor = nin(n, VX_PORT_A);
    if (!a_tensor) a_tensor = node_input_default(n);
    T* b_tensor = nin(n, VX_PORT_B);
    int relu = kind == 0
        ? vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0) : 0;
    if (!a_tensor || !b_tensor || !out || !go || a_tensor->dtype != T_F32 ||
        b_tensor->dtype != T_F32 || out->dtype != T_F32 || out->ndim < 0 || out->ndim > 8 ||
        relu < 0 || relu > 2) return -1;
    uint32_t a_strides[8], b_strides[8], output_strides[8] = {0};
    if (gpu_broadcast_strides(a_tensor, out, a_strides) != 0 ||
        gpu_broadcast_strides(b_tensor, out, b_strides) != 0) return -1;
    long stride = 1;
    for (int i = out->ndim - 1; i >= 0; i--) {
        if (stride > UINT32_MAX) return -1;
        output_strides[i] = (uint32_t)stride;
        stride *= out->shape[i];
    }
    if (out->numel > UINT32_MAX || a_tensor->numel > UINT32_MAX || b_tensor->numel > UINT32_MAX) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 32 * sizeof(uint32_t));
    if (!params) return -1;
    params[0] = (uint32_t)out->ndim; params[1] = (uint32_t)out->numel;
    params[2] = (uint32_t)a_tensor->numel; params[3] = (uint32_t)b_tensor->numel; params[4] = kind;
    params[5] = (uint32_t)relu;
    memcpy(params + 8, output_strides, sizeof(output_strides));
    memcpy(params + 16, a_strides, sizeof(a_strides));
    memcpy(params + 24, b_strides, sizeof(b_strides));
    float* ga = grad_for(grads, a_tensor);
    float* gb = grad_for(grads, b_tensor);
    if (!ga || !gb) return -1;
    void* hosts[6]; size_t bytes[6]; unsigned char access[6];
    gpu_bindings_init(hosts, bytes, access, 6);
    gpu_binding_tensor(hosts, bytes, 0, a_tensor);
    gpu_binding_tensor(hosts, bytes, 1, b_tensor);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, ga, a_tensor->numel);
    gpu_binding_gradient(hosts, bytes, 4, gb, b_tensor->numel);
    gpu_binding_params(hosts, bytes, 5, params, 32 * sizeof(uint32_t));
    access[3] = access[4] = GPU_TRAINING_READ_WRITE;
    uint32_t ga_groups = gpu_groups_64(a_tensor->numel);
    uint32_t gb_groups = gpu_groups_64(b_tensor->numel);
    const char* ga_entry = "a_main";
    const char* gb_entry = "b_main";
    if (gpu_training_select_backend() == NATIVE_GPU_TRAINING_CUDA) {
        /* Explicit CUDA tactics: large scalar broadcasts use a deterministic
           staged reduction; small genuine broadcasts use one cooperative
           block per destination. Identity and larger general broadcasts
           retain their established entry points/contracts. */
        if (a_tensor->numel == 1 && out->numel >
                GPU_TRAINING_CUDA_BASIC_SCALAR_ITEMS_PER_PARTIAL) {
            ga_entry = "a_scalar_staged";
            ga_groups = gpu_groups_n(
                out->numel,
                GPU_TRAINING_CUDA_BASIC_SCALAR_ITEMS_PER_PARTIAL);
        } else if (a_tensor->numel > 1 && a_tensor->numel <= 1024 &&
            a_tensor->numel < out->numel) {
            ga_entry = "a_cooperative";
            ga_groups = (uint32_t)a_tensor->numel;
        }
        if (b_tensor->numel == 1 && out->numel >
                GPU_TRAINING_CUDA_BASIC_SCALAR_ITEMS_PER_PARTIAL) {
            gb_entry = "b_scalar_staged";
            gb_groups = gpu_groups_n(
                out->numel,
                GPU_TRAINING_CUDA_BASIC_SCALAR_ITEMS_PER_PARTIAL);
        } else if (b_tensor->numel > 1 && b_tensor->numel <= 1024 &&
            b_tensor->numel < out->numel) {
            gb_entry = "b_cooperative";
            gb_groups = (uint32_t)b_tensor->numel;
        }
    }
    if (!ga_groups || !gb_groups ||
        gpu_training_plan_add(plan, "basicBackward", ga_entry, hosts, bytes, access, 6,
                              ga_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "basicBackward", gb_entry, hosts, bytes, access, 6,
                              gb_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_copy(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 || in->numel != out->numel) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(out->numel);
    if (!params || !gi || !groups || out->numel > UINT32_MAX) return -1;
    params[0] = (uint32_t)out->numel;
    void* hosts[3]; size_t bytes[3]; unsigned char access[3];
    gpu_bindings_init(hosts, bytes, access, 3);
    gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
    gpu_binding_params(hosts, bytes, 2, params, 4 * sizeof(uint32_t));
    access[1] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "copyBackward", "main", hosts, bytes, access, 3, groups, 1, 1);
}

static int gpu_plan_dropout(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    int node_index = (int)(n - g_n);
    if (!in || !out || !go || node_index < 0 || node_index >= g_nn ||
        in->dtype != T_F32 || out->dtype != T_F32 || in->numel != out->numel ||
        out->numel <= 0 || out->numel > UINT32_MAX) return -1;
    NativeDropoutParams dropout;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(out->numel);
    if (native_dropout_params(n, node_index, &dropout) != 0 || !params || !gi || !groups)
        return -1;
    params[0] = (uint32_t)out->numel;
    params[1] = dropout.threshold;
    params[2] = dropout.seed;
    params[3] = dropout.counter;
    params[4] = f32_bits(dropout.scale);
    void* hosts[3]; size_t bytes[3]; unsigned char access[3];
    gpu_bindings_init(hosts, bytes, access, 3);
    gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
    gpu_binding_params(hosts, bytes, 2, params, 8 * sizeof(uint32_t));
    access[1] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "dropoutBackward", "main", hosts, bytes, access, 3,
                                 groups, 1, 1);
}

static int gpu_plan_last_axis_reduction(GpuTrainingPlan* plan, float** grads, Node* n,
                                        T* out, float* go) {
    T* in = NULL;
    long rows = 0;
    int width = 0;
    float scale = 1.0f;
    if (!go || last_axis_reduction_dimensions(n, out, &in, &rows, &width, &scale) != 0 ||
        rows > UINT32_MAX || in->numel > UINT32_MAX) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(in->numel);
    if (!params || !gi || !groups) return -1;
    params[0] = (uint32_t)in->numel;
    params[1] = (uint32_t)width;
    params[2] = f32_bits(scale);
    void* hosts[3]; size_t bytes[3]; unsigned char access[3];
    gpu_bindings_init(hosts, bytes, access, 3);
    gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
    gpu_binding_params(hosts, bytes, 2, params, 4 * sizeof(uint32_t));
    access[1] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "reduceBackward", "main", hosts, bytes, access, 3,
                                 groups, 1, 1);
}

static int gpu_plan_transpose(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel || in->ndim <= 0 || in->ndim > 8 || out->ndim != in->ndim) return -1;
    int rank = in->ndim;
    uint32_t perm[8] = {0}, input_strides[8] = {0}, output_strides[8] = {0};
    int seen[8] = {0};
    int permutation_count = 0;
    const int32_t* permutation = vx_node_param_array(
        n, VX_NODE_PARAM_PERM, VX_NODE_ARRAY_PERM, &permutation_count);
    if (permutation && permutation_count != rank) return -1;
    if (!permutation && vx_node_param(n, VX_NODE_PARAM_PERM)->kind !=
        VX_NODE_PARAM_ABSENT) return -1;
    for (int i = 0; i < rank; i++) {
        int value = permutation ? permutation[i] : rank - 1 - i;
        if (value < 0 || value >= rank || seen[value] || out->shape[i] != in->shape[value]) return -1;
        seen[value] = 1;
        perm[i] = (uint32_t)value;
    }
    long stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stride > UINT32_MAX) return -1;
        input_strides[i] = (uint32_t)stride;
        stride *= in->shape[i];
    }
    stride = 1;
    for (int i = rank - 1; i >= 0; i--) {
        if (stride > UINT32_MAX) return -1;
        output_strides[i] = (uint32_t)stride;
        stride *= out->shape[i];
    }
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 34 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(out->numel);
    if (!params || !gi || !groups || out->numel > UINT32_MAX) return -1;
    params[0] = (uint32_t)rank; params[1] = (uint32_t)out->numel;
    for (int i = 0; i < 8; i++) params[2 + i] = i < rank ? (uint32_t)in->shape[i] : 1u;
    memcpy(params + 10, perm, sizeof(perm));
    memcpy(params + 18, input_strides, sizeof(input_strides));
    memcpy(params + 26, output_strides, sizeof(output_strides));
    void* hosts[3]; size_t bytes[3]; unsigned char access[3];
    gpu_bindings_init(hosts, bytes, access, 3);
    gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
    gpu_binding_params(hosts, bytes, 2, params, 34 * sizeof(uint32_t));
    access[1] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "transposeBackward", "main", hosts, bytes, access, 3,
                                 groups, 1, 1);
}

static int gpu_activation_kind(VxOperatorKind op) {
    if (op == VX_OP_RELU) return 0;
    if (op == VX_OP_GELU) return 1;
    if (op == VX_OP_SILU) return 2;
    if (op == VX_OP_SIGMOID) return 3;
    if (op == VX_OP_TANH) return 4;
    if (op == VX_OP_LEAKY_RELU) return 5;
    if (op == VX_OP_HARD_SIGMOID) return 6;
    if (op == VX_OP_HARD_SWISH) return 7;
    if (op == VX_OP_CLIP) return 8;
    return -1;
}

static int gpu_plan_activation(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    int kind = gpu_activation_kind(n->operator_kind);
    if (!in || !out || !go || kind < 0 || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel || out->numel > UINT32_MAX) return -1;
    int gelu_tanh = 0;
    if (kind == 1 && training_gelu_approximate_tanh(n, &gelu_tanh) != 0)
        return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(out->numel);
    if (!params || !gi || !groups) return -1;
    params[0] = (uint32_t)out->numel;
    params[1] = (uint32_t)(kind == 1 && gelu_tanh ? 9 : kind);
    params[2] = f32_bits(kind == 8
        ? vx_node_param_f32(n, VX_NODE_PARAM_MIN, -1.0e30f)
        : vx_node_param_f32(n, VX_NODE_PARAM_ALPHA, 0.01f));
    params[3] = f32_bits(kind == 8
        ? vx_node_param_f32(n, VX_NODE_PARAM_MAX, 1.0e30f) : 0.0f);
    if (kind == 8) {
        T* lower = nin(n, VX_PORT_MIN);
        T* upper = nin(n, VX_PORT_MAX);
        T* bounds[2] = {lower, upper};
        for (size_t i = 0; i < 2; ++i) if (bounds[i] &&
            (bounds[i]->dtype != T_F32 || bounds[i]->numel != 1 || !bounds[i]->data ||
             !volvoxai_engine_tensor_is_model_weight_locked(bounds[i]->name))) return -1;
        float lo = lower ? lower->data[0] : n->parsed_params.min_val;
        float hi = upper ? upper->data[0] : n->parsed_params.max_val;
        if (isnan(lo) || isnan(hi) || lo > hi) return -1;
        params[2] = f32_bits(lo); params[3] = f32_bits(hi);
    }
    void* hosts[5]; size_t bytes[5]; unsigned char access[5];
    gpu_bindings_init(hosts, bytes, access, 5);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, out);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gi, in->numel);
    gpu_binding_params(hosts, bytes, 4, params, 4 * sizeof(uint32_t));
    access[3] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "activationBackward", "main", hosts, bytes, access, 5,
                                 groups, 1, 1);
}

static int gpu_plan_prelu(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    if (!weight) weight = nin(n, VX_PORT_SLOPE);
    if (!in || !weight || !out || !go || in->dtype != T_F32 || weight->dtype != T_F32 ||
        out->dtype != T_F32 || in->ndim <= 0 || in->numel != out->numel ||
        in->numel > UINT32_MAX) return -1;
    int channels = in->shape[in->ndim - 1];
    if (channels <= 0 || (weight->numel != 1 && weight->numel != channels)) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    uint32_t input_groups = gpu_groups_64(in->numel);
    uint32_t weight_groups = gpu_groups_64(weight->numel);
    if (!params || !gi || !gw || !input_groups || !weight_groups) return -1;
    params[0] = (uint32_t)in->numel;
    params[1] = (uint32_t)channels;
    params[2] = (uint32_t)weight->numel;
    void* hosts[6]; size_t bytes[6]; unsigned char access[6];
    gpu_bindings_init(hosts, bytes, access, 6);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 4, gw, weight->numel);
    gpu_binding_params(hosts, bytes, 5, params, 4 * sizeof(uint32_t));
    access[3] = access[4] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "preluBackward", "input_main", hosts, bytes, access, 6,
                              input_groups, 1, 1) != 0) return -1;
    return gpu_training_plan_add(plan, "preluBackward", "weight_main", hosts, bytes, access, 6,
                                 weight_groups, 1, 1);
}

static int gpu_plan_softmax(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->numel != out->numel || in->ndim <= 0) return -1;
    int axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, in->ndim - 1);
    if (axis < 0) axis += in->ndim;
    int width = in->shape[in->ndim - 1];
    long rows = width > 0 ? in->numel / width : 0;
    if (axis != in->ndim - 1 || width <= 0 || rows <= 0 || rows > UINT32_MAX) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(in->numel);
    if (!params || !gi || !groups) return -1;
    params[0] = (uint32_t)rows; params[1] = (uint32_t)width;
    params[2] = (n->operator_kind == VX_OP_LOG_SOFTMAX) ? 1u : 0u;
    void* hosts[4]; size_t bytes[4]; unsigned char access[4];
    gpu_bindings_init(hosts, bytes, access, 4);
    gpu_binding_tensor(hosts, bytes, 0, out);
    gpu_binding_gradient(hosts, bytes, 1, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 2, gi, in->numel);
    gpu_binding_params(hosts, bytes, 3, params, 4 * sizeof(uint32_t));
    access[2] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "softmaxBackward", "main", hosts, bytes, access, 4,
                                 groups, 1, 1);
}

static int gpu_plan_embedding(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* tokens = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    if (!tokens || !weight || !out || !go || tokens->dtype != T_I32 || weight->dtype != T_F32 ||
        out->dtype != T_F32 || weight->ndim <= 0 || out->ndim <= 0) return -1;
    int vocab = weight->shape[0];
    int d_model = out->shape[out->ndim - 1];
    long token_count = tokens->numel;
    if (vocab <= 0 || d_model <= 0 || token_count <= 0 || token_count > UINT32_MAX ||
        weight->numel != (long)vocab * d_model || out->numel != token_count * d_model) return -1;
    const int32_t* ids = (const int32_t*)tokens->data;
    for (long token = 0; token < token_count; token++) {
        if (ids[token] < 0 || ids[token] >= vocab) return -1;
    }
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gw = grad_for(grads, weight);
    uint32_t groups = gpu_groups_64(weight->numel);
    if (!params || !gw || !groups) return -1;
    params[0] = (uint32_t)token_count; params[1] = (uint32_t)d_model; params[2] = (uint32_t)vocab;
    void* hosts[4]; size_t bytes[4]; unsigned char access[4];
    gpu_bindings_init(hosts, bytes, access, 4);
    gpu_binding_tensor(hosts, bytes, 0, tokens);
    gpu_binding_gradient(hosts, bytes, 1, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 2, gw, weight->numel);
    gpu_binding_params(hosts, bytes, 3, params, 4 * sizeof(uint32_t));
    access[2] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "embeddingBackward", "main", hosts, bytes, access, 4,
                                 groups, 1, 1);
}

static int gpu_plan_layernorm(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    T* bias = nin(n, VX_PORT_BIAS);
    if (!in || !weight || !out || !go || in->dtype != T_F32 || weight->dtype != T_F32 ||
        out->dtype != T_F32 || in->ndim <= 0 || (bias && bias->dtype != T_F32)) return -1;
    int d_model = in->shape[in->ndim - 1];
    long rows = d_model > 0 ? in->numel / d_model : 0;
    int requested_d_model = vx_node_param_i32(
        n, VX_NODE_PARAM_D_MODEL, d_model);
    float epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-6f);
    if (d_model <= 0 || requested_d_model != d_model || rows <= 0 || rows > UINT32_MAX ||
        in->numel > UINT32_MAX || in->numel != rows * (long)d_model ||
        !(epsilon > 0.0f) || !isfinite(epsilon) ||
        weight->numel < d_model || (bias && bias->numel < d_model) ||
        out->numel != in->numel) return -1;
    NativeGpuTrainingBackend backend = gpu_training_select_backend();
    int cooperative_param = backend == NATIVE_GPU_TRAINING_CUDA && rows > 64;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    float* gb = bias ? grad_for(grads, bias) : NULL;
    uint32_t input_groups = gpu_groups_64(in->numel);
    uint32_t param_groups = cooperative_param
        ? (uint32_t)d_model : gpu_groups_64(d_model);
    const char* param_entry = cooperative_param
        ? "param_cooperative" : "param_main";
    if (!params || !gi || !gw || (bias && !gb) || !input_groups || !param_groups) return -1;
    params[0] = (uint32_t)rows; params[1] = (uint32_t)d_model;
    params[2] = f32_bits(epsilon); params[3] = bias ? 1u : 0u;
    void* hosts[7]; size_t bytes[7]; unsigned char access[7];
    gpu_bindings_init(hosts, bytes, access, 7);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 4, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 5, gb, bias ? bias->numel : 0);
    gpu_binding_params(hosts, bytes, 6, params, 4 * sizeof(uint32_t));
    access[3] = access[4] = access[5] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "layerNormBackward", "input_main", hosts, bytes, access, 7,
                              input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "layerNormBackward", param_entry, hosts, bytes, access, 7,
                              param_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_groupnorm(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    if (!weight) weight = nin(n, VX_PORT_SCALE);
    T* bias = nin(n, VX_PORT_BIAS);
    int groups = vx_node_param_i32(n, VX_NODE_PARAM_NUM_GROUPS,
        vx_node_param_i32(n, VX_NODE_PARAM_GROUPS, 1));
    float epsilon = vx_node_param_f32(n, VX_NODE_PARAM_EPS, 1.0e-5f);
    int batch, height, width, channels;
    if (!go || !(epsilon > 0.0f) || !isfinite(epsilon) ||
        groupnorm_dimensions(in, out, weight, bias, groups, &batch, &height,
                             &width, &channels) != 0) return -1;
    NativeGpuTrainingBackend backend = gpu_training_select_backend();
    uint64_t total_groups = (uint64_t)(uint32_t)batch * (uint32_t)groups;
    if (total_groups == 0 || total_groups > UINT32_MAX ||
        (backend == NATIVE_GPU_TRAINING_CUDA && total_groups > INT_MAX))
        return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    float* gb = bias ? grad_for(grads, bias) : NULL;
    /* CUDA assigns one cooperative 64-lane block to each sample/group.
       Its parameter kernel likewise assigns one block to each channel.
       Shader backends retain their one-invocation-per-output contract. */
    uint32_t input_groups = backend == NATIVE_GPU_TRAINING_CUDA
        ? (uint32_t)total_groups : gpu_groups_64((long)total_groups);
    uint32_t parameter_groups = backend == NATIVE_GPU_TRAINING_CUDA
        ? (uint32_t)channels : gpu_groups_64(channels);
    const char* parameter_entry = backend == NATIVE_GPU_TRAINING_CUDA
        ? "param_cooperative" : "param_main";
    if (!params || !gi || !gw || (bias && !gb) || !input_groups || !parameter_groups)
        return -1;
    params[0] = (uint32_t)batch;
    params[1] = (uint32_t)height;
    params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    params[4] = (uint32_t)groups;
    params[5] = bias ? 1u : 0u;
    params[6] = f32_bits(epsilon);
    void* hosts[7]; size_t bytes[7]; unsigned char access[7];
    gpu_bindings_init(hosts, bytes, access, 7);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 4, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 5, gb, bias ? bias->numel : 0);
    gpu_binding_params(hosts, bytes, 6, params, 8 * sizeof(uint32_t));
    access[3] = access[4] = access[5] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "groupNormBackward", "input_main", hosts, bytes,
                              access, 7, input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "groupNormBackward", parameter_entry, hosts, bytes,
                              access, 7, parameter_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_rmsnorm(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    if (!in || !weight || !out || !go || in->dtype != T_F32 || weight->dtype != T_F32 ||
        out->dtype != T_F32 || in->ndim <= 0) return -1;
    int d_model = vx_node_param_i32(
        n, VX_NODE_PARAM_D_MODEL, in->shape[in->ndim - 1]);
    long rows = d_model > 0 ? in->numel / d_model : 0;
    if (d_model <= 0 || rows <= 0 || in->numel != rows * d_model || rows > UINT32_MAX ||
        weight->numel < d_model || out->numel != in->numel) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    uint32_t input_groups = gpu_groups_64(in->numel);
    uint32_t weight_groups = gpu_groups_64(d_model);
    if (!params || !gi || !gw || !input_groups || !weight_groups) return -1;
    params[0] = (uint32_t)rows; params[1] = (uint32_t)d_model;
    params[2] = f32_bits(vx_node_param_f32(
        n, VX_NODE_PARAM_EPS, 1.0e-6f));
    void* hosts[6]; size_t bytes[6]; unsigned char access[6];
    gpu_bindings_init(hosts, bytes, access, 6);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 4, gw, weight->numel);
    gpu_binding_params(hosts, bytes, 5, params, 4 * sizeof(uint32_t));
    access[3] = access[4] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "rmsNormBackward", "input_main", hosts, bytes, access, 6,
                              input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "rmsNormBackward", "weight_main", hosts, bytes, access, 6,
                              weight_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_batchnorm(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    T* bias = nin(n, VX_PORT_BIAS);
    T* mean = nin(n, VX_PORT_RUNNING_MEAN);
    T* variance = nin(n, VX_PORT_RUNNING_VAR);
    if (!in || !weight || !bias || !mean || !variance || !out || !go || in->dtype != T_F32 ||
        weight->dtype != T_F32 || bias->dtype != T_F32 || mean->dtype != T_F32 ||
        variance->dtype != T_F32 || out->dtype != T_F32 || in->ndim != 4 || out->numel != in->numel) return -1;
    int channels = in->shape[3];
    if (channels <= 0 || weight->numel < channels || bias->numel < channels ||
        mean->numel < channels || variance->numel < channels || in->numel > UINT32_MAX) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 4 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    float* gb = grad_for(grads, bias);
    uint32_t input_groups = gpu_groups_64(in->numel);
    uint32_t param_groups = gpu_groups_64(channels);
    if (!params || !gi || !gw || !gb || !input_groups || !param_groups) return -1;
    params[0] = (uint32_t)in->numel; params[1] = (uint32_t)channels;
    params[2] = f32_bits(vx_node_param_f32(
        n, VX_NODE_PARAM_EPS, 1.0e-5f));
    void* hosts[9]; size_t bytes[9]; unsigned char access[9];
    gpu_bindings_init(hosts, bytes, access, 9);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_tensor(hosts, bytes, 2, mean);
    gpu_binding_tensor(hosts, bytes, 3, variance);
    gpu_binding_gradient(hosts, bytes, 4, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 5, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 6, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 7, gb, bias->numel);
    gpu_binding_params(hosts, bytes, 8, params, 4 * sizeof(uint32_t));
    access[5] = access[6] = access[7] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "batchNorm2DBackward", "input_main", hosts, bytes, access, 9,
                              input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "batchNorm2DBackward", "param_main", hosts, bytes, access, 9,
                              param_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_pooling(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go,
                            int max_pool) {
    T* in = node_input_default(n);
    TrainingMaxPool2DGeometry geometry;
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim != 4 || in->shape[0] <= 0 || in->shape[1] <= 0 ||
        in->shape[2] <= 0 || in->shape[3] <= 0) return -1;
    if (max_pool && training_maxpool2d_geometry(n, in, out, &geometry) != 0)
        return -1;
    int batch = in->shape[0], h = in->shape[1], width = in->shape[2], channels = in->shape[3];
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 12 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(in->numel);
    if (!params || !gi || !groups) return -1;
    params[0] = (uint32_t)batch; params[1] = (uint32_t)h; params[2] = (uint32_t)width;
    params[3] = (uint32_t)channels;
    if (max_pool) {
        params[4] = (uint32_t)geometry.output_height;
        params[5] = (uint32_t)geometry.output_width;
        params[6] = (uint32_t)geometry.kernel_y;
        params[7] = (uint32_t)geometry.kernel_x;
        params[8] = (uint32_t)geometry.stride_y;
        params[9] = (uint32_t)geometry.stride_x;
        params[10] = (uint32_t)geometry.pads[0];
        params[11] = (uint32_t)geometry.pads[1];
    } else if (out->numel != (long)batch * channels) {
        return -1;
    }
    void* hosts[4]; size_t bytes[4]; unsigned char access[4];
    gpu_bindings_init(hosts, bytes, access, 4);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_gradient(hosts, bytes, 1, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 2, gi, in->numel);
    gpu_binding_params(hosts, bytes, 3, params, 12 * sizeof(uint32_t));
    access[2] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "poolingBackward",
                                 max_pool ? "max_pool_main" : "global_average_main",
                                 hosts, bytes, access, 4, groups, 1, 1);
}

static int gpu_plan_resize(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    int nearest = 1;
    if ((n->operator_kind == VX_OP_RESIZE) && resize_f32_parameters(n, &nearest)) return -1;
    if (!in || !out || !go || in->dtype != T_F32 || out->dtype != T_F32 ||
        in->ndim != 4 || out->ndim != 4 || in->shape[0] != out->shape[0] ||
        in->shape[3] != out->shape[3]) return -1;
    for (int i = 0; i < 4; i++) if (in->shape[i] <= 0 || out->shape[i] <= 0) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    uint32_t groups = gpu_groups_64(in->numel);
    if (!params || !gi || !groups) return -1;
    params[0] = (uint32_t)in->shape[0]; params[1] = (uint32_t)in->shape[1];
    params[2] = (uint32_t)in->shape[2]; params[3] = (uint32_t)in->shape[3];
    params[4] = (uint32_t)out->shape[1]; params[5] = (uint32_t)out->shape[2];
    params[6] = nearest ? 0u : 1u; params[7] = 0u;
    void* hosts[3]; size_t bytes[3]; unsigned char access[3];
    gpu_bindings_init(hosts, bytes, access, 3);
    gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
    gpu_binding_params(hosts, bytes, 2, params, 8 * sizeof(uint32_t));
    access[1] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "resizeBackward", "main", hosts, bytes, access, 3,
                                 groups, 1, 1);
}

static int gpu_plan_concat(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    if (!n || !out || !go || out->dtype != T_F32 || out->ndim <= 0 ||
        n->nin <= 0) return -1;
    if (vx_node_param_i32(n, VX_NODE_PARAM_SIGMOID, 0)) {
        uint32_t groups = gpu_groups_64(out->numel);
        if (!groups || out->numel <= 0 || (uint64_t)out->numel > SIZE_MAX / sizeof(float)) return -1;
        float* adjusted = gpu_training_plan_alloc(plan, (size_t)out->numel * sizeof(float));
        uint32_t* params = gpu_training_plan_alloc(plan, 4u * sizeof(uint32_t));
        if (!adjusted || !params) return -1;
        params[0] = (uint32_t)out->numel; params[1] = 3u;
        params[2] = params[3] = 0u;
        void* hosts[5]; size_t bytes[5]; unsigned char access[5];
        gpu_bindings_init(hosts, bytes, access, 5);
        gpu_binding_tensor(hosts, bytes, 0, out);
        gpu_binding_tensor(hosts, bytes, 1, out);
        gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
        gpu_binding_gradient(hosts, bytes, 3, adjusted, out->numel);
        gpu_binding_params(hosts, bytes, 4, params, 4u * sizeof(uint32_t));
        access[3] = GPU_TRAINING_READ_WRITE;
        if (gpu_training_plan_add(plan, "activationBackward", "main", hosts, bytes, access, 5,
                                  groups, 1, 1)) return -1;
        go = adjusted;
    }
    int axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0);
    if (axis < 0) axis += out->ndim;
    if (axis < 0 || axis >= out->ndim) return -1;
    long inner = 1;
    for (int i = axis + 1; i < out->ndim; i++) inner *= out->shape[i];
    int axis_offset = 0;
    for (int input_index = 0; input_index < n->nin; input_index++) {
        T* in = t_find(n->ins[input_index].name);
        if (!in || in->dtype != T_F32 || in->ndim != out->ndim || in->numel > UINT32_MAX) return -1;
        for (int dimension = 0; dimension < out->ndim; dimension++) {
            if (dimension != axis && in->shape[dimension] != out->shape[dimension]) return -1;
        }
        if (axis_offset < 0 || inner <= 0 || inner > UINT32_MAX || out->shape[axis] <= 0) return -1;
        uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
        float* gi = grad_for(grads, in);
        uint32_t groups = gpu_groups_64(in->numel);
        if (!params || !gi || !groups) return -1;
        params[0] = (uint32_t)in->numel; params[1] = (uint32_t)axis_offset;
        params[2] = (uint32_t)in->shape[axis]; params[3] = (uint32_t)out->shape[axis];
        params[4] = (uint32_t)inner;
        void* hosts[3]; size_t bytes[3]; unsigned char access[3];
        gpu_bindings_init(hosts, bytes, access, 3);
        gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
        gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
        gpu_binding_params(hosts, bytes, 2, params, 8 * sizeof(uint32_t));
        access[1] = GPU_TRAINING_READ_WRITE;
        if (gpu_training_plan_add(plan, "concatBackward", "main", hosts, bytes, access, 3,
                                  groups, 1, 1) != 0) return -1;
        axis_offset += in->shape[axis];
    }
    return axis_offset == out->shape[axis] ? 0 : -1;
}

static int gpu_plan_split(GpuTrainingPlan* plan, float** grads, Node* n) {
    T* in = node_input_default(n);
    if (!plan || !grads || !n || !in || in->dtype != T_F32 || in->ndim <= 0 ||
        n->nout <= 0 || in->numel <= 0 || in->numel > UINT32_MAX) return -1;
    int axis = vx_node_param_i32(n, VX_NODE_PARAM_AXIS, 0);
    if (axis < 0) axis += in->ndim;
    if (axis < 0 || axis >= in->ndim || in->shape[axis] <= 0) return -1;
    long inner = 1;
    for (int d = axis + 1; d < in->ndim; d++) {
        if (in->shape[d] <= 0 || inner > UINT32_MAX) return -1;
        inner *= in->shape[d];
    }
    if (inner <= 0 || inner > UINT32_MAX) return -1;
    long outer = in->numel / ((long)in->shape[axis] * inner);
    if (outer <= 0 || in->numel != outer * in->shape[axis] * inner) return -1;

    float* gi = grad_for(grads, in);
    if (!gi) return -1;
    int axis_offset = 0;
    int live_outputs = 0;
    for (int oi = 0; oi < n->nout; oi++) {
        T* out = t_find(n->outs[oi].name);
        if (!out || out->dtype != T_F32 || out->ndim != in->ndim ||
            out->shape[axis] <= 0 || out->numel > UINT32_MAX) return -1;
        for (int d = 0; d < in->ndim; d++) {
            if (d != axis && out->shape[d] != in->shape[d]) return -1;
        }
        long block = (long)out->shape[axis] * inner;
        if (out->numel != outer * block ||
            axis_offset > in->shape[axis] - out->shape[axis]) return -1;
        float* go = grad_existing(grads, out);
        if (go) {
            uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 5 * sizeof(uint32_t));
            uint32_t groups = gpu_groups_64(out->numel);
            if (!params || !groups) return -1;
            params[0] = (uint32_t)out->numel;
            params[1] = (uint32_t)inner;
            params[2] = (uint32_t)out->shape[axis];
            params[3] = (uint32_t)in->shape[axis];
            params[4] = (uint32_t)axis_offset;
            void* hosts[3]; size_t bytes[3]; unsigned char access[3];
            gpu_bindings_init(hosts, bytes, access, 3);
            gpu_binding_gradient(hosts, bytes, 0, go, out->numel);
            gpu_binding_gradient(hosts, bytes, 1, gi, in->numel);
            gpu_binding_params(hosts, bytes, 2, params, 5 * sizeof(uint32_t));
            access[1] = GPU_TRAINING_READ_WRITE;
            if (gpu_training_plan_add(plan, "splitBackward", "main", hosts, bytes, access, 3,
                                      groups, 1, 1) != 0) return -1;
            live_outputs++;
        }
        axis_offset += out->shape[axis];
    }
    return axis_offset == in->shape[axis] && live_outputs > 0 ? 0 : -1;
}

static int gpu_plan_conv2d(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT);
    T* bias = nin(n, VX_PORT_BIAS);
    if (!in || !weight || !out || !go || in->dtype != T_F32 || weight->dtype != T_F32 ||
        out->dtype != T_F32 || (bias && bias->dtype != T_F32) || in->ndim != 4 ||
        weight->ndim != 4 || out->ndim != 4) return -1;
    int batch = in->shape[0], in_h = in->shape[1], in_w = in->shape[2], in_c = in->shape[3];
    int out_h = out->shape[1], out_w = out->shape[2], out_c = out->shape[3];
    int stride[2], padding[2], dilation[2];
    if (training_node_pair(n, VX_NODE_PARAM_STRIDE, VX_NODE_ARRAY_STRIDE,
                           1, INT_MIN, 0, stride) != 0 ||
        training_node_pair(n, VX_NODE_PARAM_PADDING, VX_NODE_ARRAY_PADDING,
                           0, INT_MIN, 0, padding) != 0 ||
        training_node_pair(n, VX_NODE_PARAM_DILATION, VX_NODE_ARRAY_DILATION,
                           1, INT_MIN, 0, dilation) != 0) return -1;
    int sy = stride[0], sx = stride[1];
    int py = padding[0], px = padding[1];
    int dy = dilation[0], dx = dilation[1];
    int pads[4] = {py, px, py, px};
    int pads_count = 0;
    const int32_t* cached_pads = vx_node_param_array(
        n, VX_NODE_PARAM_PADS, VX_NODE_ARRAY_PADS, &pads_count);
    if (cached_pads) {
        if (pads_count != 4) return -1;
        for (int index = 0; index < 4; index++) pads[index] = cached_pads[index];
    } else if (vx_node_param(n, VX_NODE_PARAM_PADS)->kind !=
               VX_NODE_PARAM_ABSENT) {
        return -1;
    }
    int groups = vx_node_param_i32(n, VX_NODE_PARAM_GROUPS, 1);
    int relu = n->fuse_relu6 ? 2 :
        vx_node_param_i32(n, VX_NODE_PARAM_RELU, 0);
    VxNodeParamSymbol layout = vx_node_param_symbol(
        n, VX_NODE_PARAM_WEIGHT_LAYOUT, VX_NODE_SYMBOL_HWIO);
    int kh = 0, kw = 0;
    uint32_t layout_kind = 0u;
    if (batch <= 0 || in_h <= 0 || in_w <= 0 || in_c <= 0 || out_h <= 0 || out_w <= 0 ||
        out_c <= 0 || out->shape[0] != batch || sy <= 0 || sx <= 0 || dy <= 0 || dx <= 0 ||
        pads[0] < 0 || pads[1] < 0 || groups <= 0 || in_c % groups || out_c % groups ||
        relu < 0 || relu > 2 ||
        (bias && bias->numel != out_c)) return -1;
    if (groups == in_c && layout == VX_NODE_SYMBOL_HWCM) {
        int multiplier = out_c / in_c;
        if (out_c % in_c || weight->shape[2] != in_c || weight->shape[3] != multiplier) return -1;
        kh = weight->shape[0]; kw = weight->shape[1];
        layout_kind = 1u;
    } else if (groups == in_c &&
               (layout == VX_NODE_SYMBOL_1HWO ||
                layout == VX_NODE_SYMBOL_1HWM)) {
        if (out_c % in_c || weight->shape[0] != 1 ||
            weight->shape[3] != out_c) return -1;
        kh = weight->shape[1]; kw = weight->shape[2];
        layout_kind = 3u;
    } else if (layout == VX_NODE_SYMBOL_HWIO) {
        kh = weight->shape[0]; kw = weight->shape[1];
        if (weight->shape[2] != in_c / groups || weight->shape[3] != out_c) return -1;
        layout_kind = 0u;
    } else if (layout == VX_NODE_SYMBOL_OHWI) {
        if (weight->shape[0] != out_c ||
            weight->shape[3] != in_c / groups) return -1;
        kh = weight->shape[1]; kw = weight->shape[2];
        layout_kind = 2u;
    } else {
        return -1;
    }
    if (kh <= 0 || kw <= 0 || weight->numel > UINT32_MAX) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 20 * sizeof(uint32_t));
    float* gi = grad_for(grads, in);
    float* gw = grad_for(grads, weight);
    float* gb = bias ? grad_for(grads, bias) : NULL;
    uint32_t input_groups = gpu_groups_64(in->numel);
    uint32_t weight_groups = gpu_groups_64(weight->numel);
    uint32_t bias_groups = gpu_groups_64(out_c);
    if (!params || !gi || !gw || (bias && !gb) || !input_groups || !weight_groups || !bias_groups) return -1;
    params[0] = (uint32_t)batch; params[1] = (uint32_t)in_h; params[2] = (uint32_t)in_w;
    params[3] = (uint32_t)in_c; params[4] = (uint32_t)out_h; params[5] = (uint32_t)out_w;
    params[6] = (uint32_t)out_c; params[7] = (uint32_t)kh; params[8] = (uint32_t)kw;
    params[9] = (uint32_t)sy; params[10] = (uint32_t)sx; params[11] = (uint32_t)pads[0];
    params[12] = (uint32_t)pads[1]; params[13] = (uint32_t)groups; params[14] = (uint32_t)dy;
    params[15] = (uint32_t)dx; params[16] = (uint32_t)relu;
    params[17] = bias ? 1u : 0u; params[18] = (uint32_t)weight->numel;
    params[19] = layout_kind;
    void* hosts[8]; size_t bytes[8]; unsigned char access[8];
    gpu_bindings_init(hosts, bytes, access, 8);
    gpu_binding_tensor(hosts, bytes, 0, in);
    gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_tensor(hosts, bytes, 2, out);
    gpu_binding_gradient(hosts, bytes, 3, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 4, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 5, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 6, gb, bias ? bias->numel : 0);
    gpu_binding_params(hosts, bytes, 7, params, 20 * sizeof(uint32_t));
    access[4] = access[5] = access[6] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "conv2DBackward", "input_main", hosts, bytes, access, 8,
                              input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "conv2DBackward", "weight_main", hosts, bytes, access, 8,
                              weight_groups, 1, 1) != 0) return -1;
    if (bias && gpu_training_plan_add(plan, "conv2DBackward", "bias_main", hosts, bytes, access, 8,
                                      bias_groups, 1, 1) != 0) return -1;
    return 0;
}

static int gpu_plan_sdpa(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* qkv = nin(n, VX_PORT_QKV);
    if (!qkv) qkv = node_input_default(n);
    T* mask = nin(n, VX_PORT_MASK);
    if (!qkv || !out || !go || qkv->dtype != T_F32 || out->dtype != T_F32 ||
        (qkv->ndim != 2 && qkv->ndim != 3) || out->ndim != qkv->ndim) return -1;
    int batch = qkv->ndim == 3 ? qkv->shape[0] : 1;
    int seq = qkv->shape[qkv->ndim - 2];
    int d_model = out->shape[out->ndim - 1];
    int heads = vx_node_param_i32(n, VX_NODE_PARAM_HEADS, 8);
    int head_dim = heads > 0 ? d_model / heads : 0;
    int mask_mode = attention_mask_mode(mask, batch, seq, seq);
    if (batch <= 0 || seq <= 0 || d_model <= 0 || heads <= 0 || d_model % heads ||
        head_dim <= 0 || head_dim > 64 || out->shape[out->ndim - 2] != seq ||
        (out->ndim == 3 && out->shape[0] != batch) || qkv->shape[qkv->ndim - 1] != 3 * d_model ||
        qkv->numel != (long)batch * seq * 3 * d_model || out->numel != (long)batch * seq * d_model ||
        batch > (int)(UINT32_MAX / 3u) || mask_mode < 0) return -1;
    NativeAttentionDropoutParams dropout;
    if (native_attention_dropout_params(n, (int)(n - g_n), &dropout) != 0) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 12 * sizeof(uint32_t));
    float* gqkv = grad_for(grads, qkv);
    if (!params || !gqkv) return -1;
    params[0] = (uint32_t)seq; params[1] = (uint32_t)d_model; params[2] = (uint32_t)heads;
    params[3] = (uint32_t)head_dim; params[4] = (uint32_t)batch;
    params[5] = f32_bits(vx_node_param_f32(
        n, VX_NODE_PARAM_SCALE, 1.0f / sqrtf((float)head_dim)));
    params[6] = vx_node_param_i32(
        n, VX_NODE_PARAM_CAUSAL, 1) ? 1u : 0u;
    params[7] = (uint32_t)mask_mode;
    params[8] = dropout.threshold; params[9] = dropout.seed;
    params[10] = dropout.counter; params[11] = f32_bits(dropout.scale);
    void* hosts[5]; size_t bytes[5]; unsigned char access[5];
    gpu_bindings_init(hosts, bytes, access, 5);
    gpu_binding_tensor(hosts, bytes, 0, qkv);
    gpu_binding_tensor(hosts, bytes, 1, mask ? mask : qkv);
    gpu_binding_gradient(hosts, bytes, 2, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 3, gqkv, qkv->numel);
    gpu_binding_params(hosts, bytes, 4, params, 12 * sizeof(uint32_t));
    access[3] = GPU_TRAINING_READ_WRITE;
    return gpu_training_plan_add(plan, "sdpaBackward", "main", hosts, bytes, access, 5,
                                 (uint32_t)seq, (uint32_t)heads, (uint32_t)batch * 3u);
}

static int gpu_plan_cross_sdpa(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* q = nin(n, VX_PORT_Q); T* k = nin(n, VX_PORT_K); T* v = nin(n, VX_PORT_V);
    T* mask = nin(n, VX_PORT_MASK);
    if (!q || !k || !v || !out || !go || q->dtype != T_F32 || k->dtype != T_F32 ||
        v->dtype != T_F32 || out->dtype != T_F32 || (q->ndim != 2 && q->ndim != 3) ||
        k->ndim != q->ndim || v->ndim != q->ndim || out->ndim != q->ndim) return -1;
    int rank = q->ndim;
    int batch = rank == 3 ? q->shape[0] : 1;
    int seq_q = q->shape[rank - 2], seq_kv = k->shape[rank - 2], d_model = q->shape[rank - 1];
    int heads = vx_node_param_i32(n, VX_NODE_PARAM_HEADS, 8);
    int head_dim = heads > 0 ? d_model / heads : 0;
    int mask_mode = attention_mask_mode(mask, batch, seq_q, seq_kv);
    if (batch <= 0 || seq_q <= 0 || seq_kv <= 0 || d_model <= 0 || heads <= 0 ||
        d_model % heads || head_dim <= 0 || head_dim > 64 ||
        (rank == 3 && (k->shape[0] != batch || v->shape[0] != batch || out->shape[0] != batch)) ||
        k->shape[rank - 1] != d_model || v->shape[rank - 2] != seq_kv ||
        v->shape[rank - 1] != d_model || out->shape[rank - 2] != seq_q ||
        out->shape[rank - 1] != d_model || q->numel != (long)batch * seq_q * d_model ||
        k->numel != (long)batch * seq_kv * d_model || v->numel != k->numel ||
        out->numel != q->numel || mask_mode < 0) return -1;
    NativeAttentionDropoutParams dropout;
    if (native_attention_dropout_params(n, (int)(n - g_n), &dropout) != 0) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 16 * sizeof(uint32_t));
    float* gq = grad_for(grads, q); float* gk = grad_for(grads, k); float* gv = grad_for(grads, v);
    if (!params || !gq || !gk || !gv) return -1;
    params[0] = (uint32_t)seq_q; params[1] = (uint32_t)seq_kv; params[2] = (uint32_t)d_model;
    params[3] = (uint32_t)heads; params[4] = (uint32_t)head_dim; params[5] = (uint32_t)batch;
    params[6] = f32_bits(vx_node_param_f32(
        n, VX_NODE_PARAM_SCALE, 1.0f / sqrtf((float)head_dim)));
    params[7] = vx_node_param_i32(
        n, VX_NODE_PARAM_CAUSAL, 0) ? 1u : 0u;
    params[8] = (uint32_t)mask_mode;
    params[9] = dropout.threshold; params[10] = dropout.seed;
    params[11] = dropout.counter; params[12] = f32_bits(dropout.scale);
    void* hosts[9]; size_t bytes[9]; unsigned char access[9];
    gpu_bindings_init(hosts, bytes, access, 9);
    gpu_binding_tensor(hosts, bytes, 0, q); gpu_binding_tensor(hosts, bytes, 1, k);
    gpu_binding_tensor(hosts, bytes, 2, v); gpu_binding_tensor(hosts, bytes, 3, mask ? mask : q);
    gpu_binding_gradient(hosts, bytes, 4, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 5, gq, q->numel);
    gpu_binding_gradient(hosts, bytes, 6, gk, k->numel);
    gpu_binding_gradient(hosts, bytes, 7, gv, v->numel);
    gpu_binding_params(hosts, bytes, 8, params, 16 * sizeof(uint32_t));
    access[5] = access[6] = access[7] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "crossSdpaBackward", "q_main", hosts, bytes, access, 9,
                              (uint32_t)seq_q, (uint32_t)heads, (uint32_t)batch) != 0 ||
        gpu_training_plan_add(plan, "crossSdpaBackward", "k_main", hosts, bytes, access, 9,
                              (uint32_t)seq_kv, (uint32_t)heads, (uint32_t)batch) != 0 ||
        gpu_training_plan_add(plan, "crossSdpaBackward", "v_main", hosts, bytes, access, 9,
                              (uint32_t)seq_kv, (uint32_t)heads, (uint32_t)batch) != 0) return -1;
    return 0;
}

static int gpu_plan_moe_linear(GpuTrainingPlan* plan, float** grads, Node* n, T* out, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_EXPERT_WEIGHT); if (!weight) weight = nin(n, VX_PORT_WEIGHT);
    T* bias = nin(n, VX_PORT_EXPERT_BIAS); if (!bias) bias = nin(n, VX_PORT_BIAS);
    T* indices = nin(n, VX_PORT_ROUTE_INDICES); if (!indices) indices = nin(n, VX_PORT_INDICES);
    T* gates = nin(n, VX_PORT_ROUTE_WEIGHTS); if (!gates) gates = nin(n, VX_PORT_WEIGHTS);
    if (!in || !weight || !indices || !gates || !out || !go || in->dtype != T_F32 ||
        weight->dtype != T_F32 || indices->dtype != T_F32 || gates->dtype != T_F32 ||
        out->dtype != T_F32 || (bias && bias->dtype != T_F32) || in->ndim <= 0 ||
        weight->ndim != 3 || out->ndim <= 0) return -1;
    int experts = weight->shape[0], d_in = weight->shape[1], d_out = weight->shape[2];
    long rows = d_in > 0 ? in->numel / d_in : 0;
    int top_k = rows > 0 && indices->numel % rows == 0 ? (int)(indices->numel / rows) : 0;
    if (experts <= 0 || d_in <= 0 || d_out <= 0 || rows <= 0 || rows > UINT32_MAX ||
        in->shape[in->ndim - 1] != d_in || out->shape[out->ndim - 1] != d_out ||
        out->numel != rows * d_out || indices->numel != gates->numel || top_k <= 0 ||
        top_k > experts || top_k > 8 || (bias && bias->numel != (long)experts * d_out)) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
    uint32_t* slot_rows = NULL;
    uint32_t* row_slots = NULL;
    float* gi = grad_for(grads, in); float* gw = grad_for(grads, weight);
    float* gb = bias ? grad_for(grads, bias) : NULL; float* gg = grad_for(grads, gates);
    uint32_t input_groups = gpu_groups_64(rows * d_in);
    uint32_t weight_groups = gpu_groups_64(weight->numel);
    uint32_t bias_groups = gpu_groups_64((long)experts * d_out);
    uint32_t gate_groups = gpu_groups_64(rows * top_k);
    if (!params || !gi || !gw || !gg || (bias && !gb) || !input_groups || !weight_groups ||
        !bias_groups || !gate_groups) return -1;
    params[0] = (uint32_t)rows; params[1] = (uint32_t)d_in; params[2] = (uint32_t)d_out;
    params[3] = (uint32_t)experts; params[4] = (uint32_t)top_k; params[5] = bias ? 1u : 0u;
    if (n->resident_slot_rows) {
        const uint32_t domain = n->resident_slot_domain;
        if (domain < (uint32_t)experts || top_k > (int)domain ||
            (size_t)domain > SIZE_MAX / sizeof(*slot_rows)) return -1;
        slot_rows = (uint32_t*)gpu_training_plan_alloc(
            plan, (size_t)domain * sizeof(*slot_rows));
        row_slots = (uint32_t*)gpu_training_plan_alloc(
            plan, (size_t)experts * sizeof(*row_slots));
        if (!slot_rows || !row_slots) return -1;
        memcpy(slot_rows, n->resident_slot_rows,
               (size_t)domain * sizeof(*slot_rows));
        for (int row = 0; row < experts; row++) row_slots[row] = VX_MOE_SLOT_ABSENT;
        for (uint32_t slot = 0; slot < domain; slot++) {
            const uint32_t row = slot_rows[slot];
            if (row == VX_MOE_SLOT_ABSENT) continue;
            if (row >= (uint32_t)experts || row_slots[row] != VX_MOE_SLOT_ABSENT)
                return -1;
            row_slots[row] = slot;
        }
        for (int row = 0; row < experts; row++)
            if (row_slots[row] == VX_MOE_SLOT_ABSENT) return -1;
        params[6] = domain;
    } else {
        slot_rows = (uint32_t*)gpu_training_plan_alloc(plan, sizeof(*slot_rows));
        row_slots = (uint32_t*)gpu_training_plan_alloc(plan, sizeof(*row_slots));
        if (!slot_rows || !row_slots) return -1;
        params[6] = 0u;
    }
    params[7] = 0u;
    void* hosts[13]; size_t bytes[13]; unsigned char access[13];
    gpu_bindings_init(hosts, bytes, access, 13);
    gpu_binding_tensor(hosts, bytes, 0, in); gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_tensor(hosts, bytes, 2, bias); gpu_binding_tensor(hosts, bytes, 3, indices);
    gpu_binding_tensor(hosts, bytes, 4, gates); gpu_binding_gradient(hosts, bytes, 5, go, out->numel);
    gpu_binding_gradient(hosts, bytes, 6, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 7, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 8, gb, bias ? bias->numel : 0);
    gpu_binding_gradient(hosts, bytes, 9, gg, gates->numel);
    hosts[10] = slot_rows;
    bytes[10] = n->resident_slot_rows
        ? (size_t)n->resident_slot_domain * sizeof(*slot_rows)
        : sizeof(*slot_rows);
    hosts[11] = row_slots;
    bytes[11] = n->resident_slot_rows
        ? (size_t)experts * sizeof(*row_slots)
        : sizeof(*row_slots);
    gpu_binding_params(hosts, bytes, 12, params, 8 * sizeof(uint32_t));
    access[6] = access[7] = access[8] = access[9] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "moeLinearBackward", "input_main", hosts, bytes, access, 13,
                              input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "moeLinearBackward", "weight_main", hosts, bytes, access, 13,
                              weight_groups, 1, 1) != 0) return -1;
    if (bias && gpu_training_plan_add(plan, "moeLinearBackward", "bias_main", hosts, bytes, access, 13,
                                      bias_groups, 1, 1) != 0) return -1;
    return gpu_training_plan_add(plan, "moeLinearBackward", "route_main", hosts, bytes, access, 13,
                                 gate_groups, 1, 1);
}

static int gpu_plan_moe_router(GpuTrainingPlan* plan, float** grads, Node* n, T* route_weights, float* go) {
    T* in = node_input_default(n);
    T* weight = nin(n, VX_PORT_WEIGHT); if (!weight) weight = nin(n, VX_PORT_ROUTER_WEIGHT);
    T* bias = nin(n, VX_PORT_BIAS);
    T* indices = nout(n, VX_PORT_INDICES); if (!indices) indices = nout(n, VX_PORT_ROUTE_INDICES);
    if (!in || !weight || !indices || !route_weights || !go || in->dtype != T_F32 ||
        weight->dtype != T_F32 || indices->dtype != T_F32 || route_weights->dtype != T_F32 ||
        (bias && bias->dtype != T_F32) || in->ndim <= 0 || weight->ndim != 2) return -1;
    int d_model = in->shape[in->ndim - 1];
    int experts = vx_node_param_i32(
        n, VX_NODE_PARAM_NUM_EXPERTS, weight->shape[1]);
    long rows = d_model > 0 ? in->numel / d_model : 0;
    int top_k = rows > 0 && indices->numel % rows == 0 ? (int)(indices->numel / rows) : 0;
    float temperature = vx_node_param_f32(
        n, VX_NODE_PARAM_TEMPERATURE, 1.0f);
    if (d_model <= 0 || experts <= 0 || rows <= 0 || rows > UINT32_MAX || top_k <= 0 ||
        top_k > experts || top_k > 8 || weight->shape[0] != d_model || weight->shape[1] != experts ||
        indices->numel != rows * top_k || route_weights->numel != rows * top_k ||
        (bias && bias->numel < experts) || !isfinite(temperature) || temperature <= 0.0f) return -1;
    uint32_t* params = (uint32_t*)gpu_training_plan_alloc(plan, 8 * sizeof(uint32_t));
    float* grad_logits = (float*)gpu_training_plan_alloc(plan, (size_t)rows * experts * sizeof(float));
    float* gi = grad_for(grads, in); float* gw = grad_for(grads, weight);
    float* gb = bias ? grad_for(grads, bias) : NULL;
    uint32_t logit_groups = gpu_groups_64(rows * experts);
    uint32_t input_groups = gpu_groups_64(rows * d_model);
    uint32_t weight_groups = gpu_groups_64((long)d_model * experts);
    uint32_t bias_groups = gpu_groups_64(experts);
    if (!params || !grad_logits || !gi || !gw || (bias && !gb) || !logit_groups || !input_groups ||
        !weight_groups || !bias_groups) return -1;
    params[0] = (uint32_t)rows; params[1] = (uint32_t)d_model; params[2] = (uint32_t)experts;
    params[3] = (uint32_t)top_k;
    params[4] = vx_node_param_i32(
        n, VX_NODE_PARAM_NORMALIZE, 1) ? 1u : 0u;
    params[5] = bias ? 1u : 0u; params[6] = f32_bits(temperature);
    void* hosts[11]; size_t bytes[11]; unsigned char access[11];
    gpu_bindings_init(hosts, bytes, access, 11);
    gpu_binding_tensor(hosts, bytes, 0, in); gpu_binding_tensor(hosts, bytes, 1, weight);
    gpu_binding_tensor(hosts, bytes, 2, bias); gpu_binding_tensor(hosts, bytes, 3, indices);
    gpu_binding_tensor(hosts, bytes, 4, route_weights);
    gpu_binding_gradient(hosts, bytes, 5, go, route_weights->numel);
    gpu_binding_gradient(hosts, bytes, 6, grad_logits, rows * experts);
    gpu_binding_gradient(hosts, bytes, 7, gi, in->numel);
    gpu_binding_gradient(hosts, bytes, 8, gw, weight->numel);
    gpu_binding_gradient(hosts, bytes, 9, gb, bias ? bias->numel : 0);
    gpu_binding_params(hosts, bytes, 10, params, 8 * sizeof(uint32_t));
    access[6] = access[7] = access[8] = access[9] = GPU_TRAINING_READ_WRITE;
    if (gpu_training_plan_add(plan, "moeRouterBackward", "logit_main", hosts, bytes, access, 11,
                              logit_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "moeRouterBackward", "input_main", hosts, bytes, access, 11,
                              input_groups, 1, 1) != 0 ||
        gpu_training_plan_add(plan, "moeRouterBackward", "weight_main", hosts, bytes, access, 11,
                              weight_groups, 1, 1) != 0) return -1;
    if (bias && gpu_training_plan_add(plan, "moeRouterBackward", "bias_main", hosts, bytes, access, 11,
                                      bias_groups, 1, 1) != 0) return -1;
    return 0;
}

#include "training_extended_backward.inc"

static int gpu_build_backward_plan(GpuTrainingPlan* plan, float** grads) {
    if (!plan || !grads) return -1;
    gpu_training_plan_trace(plan);
    for (int i = g_nn - 1; i >= 0; i--) {
        Node* n = &g_n[i];
        if (n->disabled || n->skip) continue;
        if (plan->origins) plan->origin = (VxTraceWork){VX_TRACE_PHASE_BACKWARD, i, 0,
            vx_operator_kind_name(n->operator_kind), n->out, NULL};
        int is_split = (n->operator_kind == VX_OP_SPLIT);
        T* out = (n->operator_kind == VX_OP_MOE_ROUTER) ? nout(n, VX_PORT_WEIGHTS) : t_find(n->out);
        if (!out && (n->operator_kind == VX_OP_MOE_ROUTER)) out = nout(n, VX_PORT_ROUTE_WEIGHTS);
        for (int output_index = 0; output_index < n->nout; output_index++) {
            T* candidate = t_find(n->outs[output_index].name);
            if (!is_split && candidate && candidate != out && grad_existing(grads, candidate)) return -1;
        }
        if (is_split) {
            int has_live_output = 0;
            for (int output_index = 0; output_index < n->nout; output_index++) {
                T* candidate = t_find(n->outs[output_index].name);
                if (candidate && grad_existing(grads, candidate)) { has_live_output = 1; break; }
            }
            if (!has_live_output) continue;
            if (gpu_plan_split(plan, grads, n) != 0) {
                if (g_debug) fprintf(stderr, "[debug] GPU training fallback at op '%s'\n", vx_operator_kind_name(n->operator_kind));
                return -1;
            }
            continue;
        }
        float* go = grad_existing(grads, out);
        if (!out || !go) continue;
        VxOperatorKind op = n->operator_kind;
        int rc = -1;
        if ((op == VX_OP_MATMUL) || (op == VX_OP_LINEAR) || (op == VX_OP_GEMM)) {
            rc = gpu_plan_linear(plan, grads, n, out, go);
        } else if (op == VX_OP_ADD) {
            rc = gpu_plan_binary(plan, grads, n, out, go, 0);
        } else if (op == VX_OP_MUL) {
            rc = gpu_plan_binary(plan, grads, n, out, go, 1);
        } else if (op == VX_OP_EMBEDDING) {
            rc = gpu_plan_embedding(plan, grads, n, out, go);
        } else if (op == VX_OP_LAYER_NORM) {
            rc = gpu_plan_layernorm(plan, grads, n, out, go);
        } else if (op == VX_OP_GROUP_NORM) {
            rc = gpu_plan_groupnorm(plan, grads, n, out, go);
        } else if (op == VX_OP_RMS_NORM) {
            rc = gpu_plan_rmsnorm(plan, grads, n, out, go);
        } else if (gpu_activation_kind(op) >= 0) {
            rc = gpu_plan_activation(plan, grads, n, out, go);
        } else if (op == VX_OP_PRELU) {
            rc = gpu_plan_prelu(plan, grads, n, out, go);
        } else if ((op == VX_OP_SOFTMAX) || (op == VX_OP_LOG_SOFTMAX)) {
            rc = gpu_plan_softmax(plan, grads, n, out, go);
        } else if (op == VX_OP_SDPA) {
            rc = gpu_plan_sdpa(plan, grads, n, out, go);
        } else if (op == VX_OP_CROSS_SDPA) {
            rc = gpu_plan_cross_sdpa(plan, grads, n, out, go);
        } else if (op == VX_OP_MOE_LINEAR) {
            rc = gpu_plan_moe_linear(plan, grads, n, out, go);
        } else if (op == VX_OP_MOE_ROUTER) {
            rc = gpu_plan_moe_router(plan, grads, n, out, go);
        } else if (op == VX_OP_CONV_2D) {
            rc = gpu_plan_conv2d(plan, grads, n, out, go);
        } else if (op == VX_OP_BATCH_NORM_2D) {
            rc = gpu_plan_batchnorm(plan, grads, n, out, go);
        } else if (op == VX_OP_MAX_POOL_2D) {
            rc = gpu_plan_pooling(plan, grads, n, out, go, 1);
        } else if (op == VX_OP_GLOBAL_AVERAGE_POOL) {
            rc = gpu_plan_pooling(plan, grads, n, out, go, 0);
        } else if ((op == VX_OP_RESIZE) || (op == VX_OP_RESIZE_NEAREST_2D) || (op == VX_OP_UPSAMPLE_NEAREST_2D)) {
            rc = gpu_plan_resize(plan, grads, n, out, go);
        } else if (op == VX_OP_CONCAT) {
            rc = gpu_plan_concat(plan, grads, n, out, go);
        } else if (op == VX_OP_TRANSPOSE) {
            rc = gpu_plan_transpose(plan, grads, n, out, go);
        } else if (op == VX_OP_DROPOUT) {
            rc = gpu_plan_dropout(plan, grads, n, out, go);
        } else if ((op == VX_OP_REDUCE_SUM) || (op == VX_OP_REDUCE_MEAN)) {
            rc = gpu_plan_last_axis_reduction(plan, grads, n, out, go);
        } else if ((op == VX_OP_IDENTITY) || (op == VX_OP_RESHAPE) || (op == VX_OP_FLATTEN) ||
                   (op == VX_OP_SQUEEZE) || (op == VX_OP_UNSQUEEZE)) {
            rc = gpu_plan_copy(plan, grads, n, out, go);
        } else {
            rc = training_extended_backward(plan, grads, n, out, go);
        }
        if (rc != 0) {
            if (g_debug) fprintf(stderr, "[debug] GPU training fallback at op '%s'\n", vx_operator_kind_name(op));
            return -1;
        }
    }
    return 0;
}

/* Validate an entire plan before gpu_training_begin() can mutate backend
   state.  Return 0 when every command is supported, 1 when the selected
   backend cannot execute the plan, and -1 for an internal/allocation error. */
static int gpu_preflight_backward_plan(NativeGpuTrainingBackend backend,
                                       const GpuTrainingPlan* plan) {
    if (backend == NATIVE_GPU_TRAINING_NONE || !plan) return -1;
#if VOLVOXAI_ENABLE_VULKAN
    if (backend == NATIVE_GPU_TRAINING_VULKAN) {
        if (plan->command_count <= 0) return 1;
        size_t requirement_capacity = (size_t)plan->command_count *
                                      (GPU_TRAINING_MAX_BINDINGS - 1u);
        if (requirement_capacity > (size_t)INT32_MAX ||
            requirement_capacity > SIZE_MAX / sizeof(VkTrainingTensorRequirement)) return 1;
        size_t* params_bytes = (size_t*)malloc(
            (size_t)plan->command_count * sizeof(*params_bytes));
        VkTrainingTensorRequirement* requirements =
            (VkTrainingTensorRequirement*)malloc(requirement_capacity * sizeof(*requirements));
        if (!params_bytes || !requirements) {
            free(params_bytes);
            free(requirements);
            return -1;
        }
        int requirement_count = 0;
        for (int i = 0; i < plan->command_count; i++) {
            const GpuTrainingCommand* command = &plan->commands[i];
            params_bytes[i] = command->binding_count > 0
                ? command->bytes[command->binding_count - 1] : 0;
            for (int binding = 0; binding + 1 < command->binding_count; binding++) {
                requirements[requirement_count].host = command->hosts[binding];
                requirements[requirement_count].bytes = command->bytes[binding];
                requirement_count++;
            }
        }
        qsort(requirements, (size_t)requirement_count, sizeof(*requirements),
              gpu_vulkan_tensor_requirement_cmp);
        int unique_count = 0;
        for (int i = 0; i < requirement_count; i++) {
            if (unique_count > 0 &&
                requirements[i].host == requirements[unique_count - 1].host) {
                if (requirements[i].bytes > requirements[unique_count - 1].bytes) {
                    requirements[unique_count - 1].bytes = requirements[i].bytes;
                }
            } else {
                requirements[unique_count++] = requirements[i];
            }
        }
        int supported = vk_training_plan_supported(plan->command_count, params_bytes,
                                                   requirements, unique_count);
        free(params_bytes);
        free(requirements);
        if (!supported) {
            if (g_debug) fprintf(stderr,
                "[debug] Vulkan backward plan rejected commands=%d unique_buffers=%d\n",
                plan->command_count, unique_count);
            return 1;
        }
    }
#endif
    for (int i = 0; i < plan->command_count; i++) {
        if (!gpu_training_command_supported(backend, &plan->commands[i])) {
            if (g_debug) fprintf(stderr,
                "[debug] GPU backward command unsupported index=%d shader='%s' entry='%s' bindings=%d groups=[%u,%u,%u]\n",
                i, plan->commands[i].shader, plan->commands[i].entry,
                plan->commands[i].binding_count, plan->commands[i].groups[0],
                plan->commands[i].groups[1], plan->commands[i].groups[2]);
            return 1;
        }
    }
    return 0;
}

/* Close the scope opened by gpu_execute_preflighted_backward_plan().  For the
   CUDA resident mode this is intentionally a separate caller-controlled step:
   cuda_training_end() synchronizes the stream and releases transient
   activation/gradient slots, so accumulation/optimizer work must be enqueued
   before that boundary. Stable model-weight slots remain resident. */
static void gpu_complete_backward_plan_execution(NativeGpuTrainingBackend backend) {
    gpu_training_end(backend);
}

static int gpu_sync_backward_plan_trainable_gradients(
        NativeGpuTrainingBackend backend, const GpuTrainingPlan* plan,
        float** grads, const char* const* trainable_names, int trainable_count) {
    if (backend == NATIVE_GPU_TRAINING_NONE || !plan || !grads || !trainable_names ||
        trainable_count <= 0) return -1;
    for (int i = 0; i < trainable_count; i++) {
        T* tensor = t_find(trainable_names[i]);
        float* gradient = grad_existing(grads, tensor);
        if (!tensor || !gradient) return -1;
        int written_on_gpu = 0;
        for (int command_index = 0; command_index < plan->command_count && !written_on_gpu;
             command_index++) {
            const GpuTrainingCommand* command = &plan->commands[command_index];
            for (int binding = 0; binding < command->binding_count; binding++) {
                if (command->hosts[binding] == gradient &&
                    command->access[binding] == GPU_TRAINING_READ_WRITE) {
                    written_on_gpu = 1;
                    break;
                }
            }
        }
        if (written_on_gpu && gpu_training_sync(backend, gradient,
                (size_t)tensor->numel * sizeof(float)) != 0) return -1;
    }
    return 0;
}

/* Dispatch only. The caller owns an already-open backend training scope and
   is responsible for closing it on every outcome. This seam lets CUDA seed
   loss gradients and zero tape buffers on the same stream before backward. */
static int gpu_dispatch_preflighted_backward_plan(
        NativeGpuTrainingBackend backend, const GpuTrainingPlan* plan) {
    if (backend == NATIVE_GPU_TRAINING_NONE || !plan) return -1;
    VxTraceScope* scope = vx_engine_state_current()->profiling;
    if (vx_trace_nodes(scope)) {
        VxTraceWork saved = scope->work;
        for (int i = 0; i < plan->command_count; i++) {
            scope->work = gpu_training_command_origin(plan, i);
            uint64_t start = vx_trace_now_ns();
            int result = gpu_training_dispatch(backend, &plan->commands[i]);
            vx_trace_host_program(scope, start, plan->commands[i].shader, plan->commands[i].entry);
            if (result != 0) { scope->work = saved; return -1; }
        }
        scope->work = saved;
        return 0;
    }
    for (int i = 0; i < plan->command_count; i++) {
        if (gpu_training_dispatch(backend, &plan->commands[i]) != 0) {
            if (g_debug) fprintf(stderr,
                "[debug] GPU backward dispatch failed index=%d shader='%s' entry='%s'\n",
                i, plan->commands[i].shader, plan->commands[i].entry);
            return -1;
        }
    }
    return 0;
}

/* Dispatch a plan whose full command set has already passed
   gpu_preflight_backward_plan(). The host-synchronized mode copies trainable
   gradients and ends the scope. The CUDA resident mode performs no trainable-gradient
   D2H copies and, on success, leaves the CUDA training scope open; its caller
   must eventually call gpu_complete_backward_plan_execution().  Dispatch
   failure always closes the scope because no further device work is safe. */
static int gpu_execute_preflighted_backward_plan(
        NativeGpuTrainingBackend backend, const GpuTrainingPlan* plan,
        float** grads, const char* const* trainable_names, int trainable_count,
        GpuBackwardExecutionMode mode) {
    if (backend == NATIVE_GPU_TRAINING_NONE || !plan ||
        (mode != GPU_BACKWARD_EXECUTION_SYNC_TRAINABLES_AND_END &&
         mode != GPU_BACKWARD_EXECUTION_CUDA_DEVICE_RESIDENT) ||
        (mode == GPU_BACKWARD_EXECUTION_CUDA_DEVICE_RESIDENT &&
         backend != NATIVE_GPU_TRAINING_CUDA) ||
        (mode == GPU_BACKWARD_EXECUTION_SYNC_TRAINABLES_AND_END &&
         (!grads || !trainable_names || trainable_count <= 0))) return -1;
    if (gpu_training_begin(backend) != 0) {
        if (g_debug) fprintf(stderr, "[debug] GPU backward begin failed backend=%d\n", backend);
        return -1;
    }
    int rc = gpu_dispatch_preflighted_backward_plan(backend, plan);
    if (rc != 0) {
        gpu_complete_backward_plan_execution(backend);
        return -1;
    }
    if (mode == GPU_BACKWARD_EXECUTION_CUDA_DEVICE_RESIDENT) return 0;
    rc = gpu_sync_backward_plan_trainable_gradients(
        backend, plan, grads, trainable_names, trainable_count);
    gpu_complete_backward_plan_execution(backend);
    return rc;
}

/* Compatibility executor used by every existing training path.  Its return
   contract and lifecycle remain unchanged: 1 means unsupported/preflight
   rejection, 0 means gradients are host-visible and the backend is closed,
   and -1 means execution failed with the backend closed when it had begun. */
static int gpu_execute_backward_plan(NativeGpuTrainingBackend backend, const GpuTrainingPlan* plan,
                                     float** grads, const char* const* trainable_names,
                                     int trainable_count) {
    if (backend == NATIVE_GPU_TRAINING_NONE || !plan || !grads || !trainable_names ||
        trainable_count <= 0) return -1;
    int preflight = gpu_preflight_backward_plan(backend, plan);
    if (preflight != 0) return preflight;
    return gpu_execute_preflighted_backward_plan(
        backend, plan, grads, trainable_names, trainable_count,
        GPU_BACKWARD_EXECUTION_SYNC_TRAINABLES_AND_END);
}
