#include "engine_core.h"
#include "engine_internal.h"
#include "incremental_runtime.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "training/training_core.h"
#endif
#include "shader_store.h"
#include "adapter_runtime_internal.h"
#include "conv_f32_opt.h"
#include "gemm_f32.h"
#include "inference_kernels.h"
#include "backend_manager.h"
#include "thread_pool.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if VOLVOXAI_ENABLE_NNAPI
#include "nnapi_engine.h"
#endif
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif
#include <math.h>
#include <limits.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float volvoxai_engine_adapter_f16_at(const void* data, size_t index);
static float volvoxai_engine_adapter_f32_at(const void* data, size_t index);

#define g_merged_adapter_weights \
    (vx_engine_state_current()->merged_adapter_weights)
#define g_merged_adapter_count \
    (vx_engine_state_current()->merged_adapter_count)
#define g_merged_adapter_version \
    (vx_engine_state_current()->merged_adapter_version)
#define g_pre_merge_active_version \
    (vx_engine_state_current()->pre_merge_active_version)
#define g_adapter_admin_mutex \
    (vx_engine_state_current()->adapter_admin_mutex)
#define g_engine_model_mutex \
    (vx_engine_state_current()->model_mutex)
#define g_engine_metadata_mutex \
    (vx_engine_state_current()->metadata_mutex)
#define g_engine_route_lease (vx_engine_state_route_lease_active())
#define g_cpu_threads (vx_engine_state_current()->cpu_threads)
#define g_engine_model_generation \
    (vx_engine_state_current()->model_generation)

void volvoxai_engine_model_lock(void) { pthread_mutex_lock(&g_engine_model_mutex); }
void volvoxai_engine_model_unlock(void) { pthread_mutex_unlock(&g_engine_model_mutex); }
uint64_t volvoxai_engine_model_generation_locked(void) {
    return g_engine_model_generation;
}
void volvoxai_engine_model_generation_advance_locked(void) {
    g_engine_model_generation++;
    if (!g_engine_model_generation) g_engine_model_generation = 1;
    /* Graph/weight mutations invalidate both the dependency cache and the
     * seeded state exposed by a live DecodeSession. */
    vx_incremental_invalidate_locked();
}
int volvoxai_engine_model_route_lease_active(void) { return g_engine_route_lease; }
int volvoxai_engine_adapter_effect_active_locked(void) {
    char active[128] = {0};
    return g_merged_adapter_count > 0 ||
           (vx_adapter_get_active(active, sizeof(active)) == 0 && active[0]);
}
void volvoxai_engine_metadata_lock(void) { pthread_mutex_lock(&g_engine_metadata_mutex); }
void volvoxai_engine_metadata_unlock(void) { pthread_mutex_unlock(&g_engine_metadata_mutex); }
static void volvoxai_engine_shutdown_impl(void);

static void volvoxai_engine_free_cpu_typed_workspace_locked(void) {
    VxEngineState* state = vx_engine_state_current();
    if (!state) return;
    free(state->cpu_typed_workspace);
    state->cpu_typed_workspace = NULL;
    state->cpu_typed_workspace_bound_bytes = 0u;
    state->cpu_typed_workspace_capacity_bytes = 0u;
    state->cpu_typed_workspace_configured = 0;
}

int volvoxai_engine_configure_cpu_typed_workspace(size_t bounded_bytes) {
    VxEngineState* state = vx_engine_state_current();
    void* candidate = NULL;
    int took_model_lock;
    int result = -1;
    if (!state) return -1;
    took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    if (bounded_bytes && !g_loaded) goto done;
    /* The compiled proof fixes this bound for the context lifetime. Refusing
     * replacement also guarantees there is never an uncharged old+candidate
     * workspace allocation phase. Shutdown uses the private release helper. */
    if (state->cpu_typed_workspace_configured) {
        if (bounded_bytes == state->cpu_typed_workspace_bound_bytes &&
            bounded_bytes == state->cpu_typed_workspace_capacity_bytes &&
            (!bounded_bytes || state->cpu_typed_workspace))
            result = 0;
        goto done;
    }
    if (bounded_bytes) {
        candidate = malloc(bounded_bytes);
        if (!candidate) goto done;
    }
    free(state->cpu_typed_workspace);
    state->cpu_typed_workspace = candidate;
    state->cpu_typed_workspace_bound_bytes = bounded_bytes;
    state->cpu_typed_workspace_capacity_bytes = bounded_bytes;
    state->cpu_typed_workspace_configured = 1;
    candidate = NULL;
    result = 0;
done:
    free(candidate);
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

void* volvoxai_engine_cpu_typed_workspace(size_t* capacity_bytes) {
    VxEngineState* state = vx_engine_state_current();
    if (capacity_bytes)
        *capacity_bytes = state ? state->cpu_typed_workspace_capacity_bytes : 0u;
    return state ? state->cpu_typed_workspace : NULL;
}

#include "engine_bank_residency.inc"

static int volvoxai_engine_merged_contains_tensor(const char* name) {
    if (!name) return 0;
    pthread_mutex_lock(&g_adapter_admin_mutex);
    int found = 0;
    for (int i = 0; i < g_merged_adapter_count; i++) {
        if (g_merged_adapter_weights[i].tensor && !strcmp(g_merged_adapter_weights[i].tensor->name, name)) { found = 1; break; }
    }
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    return found;
}

#if VOLVOXAI_ENABLE_TRAINING
#include "../training/optimizer_runtime.inc"
#else
/* Common model mutation and shutdown paths call these hooks. Inference owns no
   optimizer state, so compile them to private no-ops instead of exporting a
   training API or spreading feature checks through lifecycle code. */
static void volvoxai_engine_free_optimizer_states(void) { }
static void optimizer_state_remove(const char* name) { (void)name; }
#endif

static size_t engine_dtype_size_local(int dtype) {
    switch (dtype) {
        case T_F32: case T_I32: return 4;
        case T_F16: return 2;
        case T_I8: case T_U8: return 1;
        default: return 0;
    }
}

static VxDataType safetensors_dtype_from_engine(int dtype) {
    switch (dtype) {
        case VX_DTYPE_F32:
        case VX_DTYPE_I8:
        case VX_DTYPE_U8:
        case VX_DTYPE_I32:
        case VX_DTYPE_F16:
            return dtype;
        default: return SAFETENSORS_DTYPE_UNKNOWN;
    }
}

static int volvoxai_engine_refresh_weight_caches_if_dirty(void) {
    if (!g_weight_caches_dirty) return 0;
    return volvoxai_engine_refresh_weight_caches();
}

int volvoxai_engine_configure(const VolvoxAIEngineOptions* options) {
    if (!options) return -1;
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (g_loaded || vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    if (options->backend < VOLVOXAI_BACKEND_CPU ||
        options->backend > VOLVOXAI_BACKEND_CUDA || options->cpu_threads < 0) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    if (vx_backend_manager_activate(options->backend) != 0) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    vx_set_num_threads(options->cpu_threads);
    g_cpu_threads = options->cpu_threads;
    g_debug = options->debug ? 1 : 0;
    g_execution_row = -1;
    volvoxai_engine_model_unlock();
    return 0;
}

int volvoxai_engine_get_options(VolvoxAIEngineOptions* options) {
    if (!options) return -1;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    options->backend = vx_backend_manager_current();
    options->debug = g_debug;
    options->cpu_threads = g_cpu_threads;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return 0;
}

const char* volvoxai_engine_backend_name(void) {
    return vx_backend_manager_name();
}

static int volvoxai_builtin_backend_by_name(const char* name,
                                             VolvoxAIEngineBackend* backend) {
    if (!name || !backend) return 0;
    if (!strcmp(name, "cpu")) *backend = VOLVOXAI_BACKEND_CPU;
    else if (!strcmp(name, "vulkan")) *backend = VOLVOXAI_BACKEND_VULKAN;
    else if (!strcmp(name, "opengl")) *backend = VOLVOXAI_BACKEND_OPENGL;
    else if (!strcmp(name, "metal")) *backend = VOLVOXAI_BACKEND_METAL;
    else if (!strcmp(name, "nnapi")) *backend = VOLVOXAI_BACKEND_NNAPI;
    else if (!strcmp(name, "cuda")) *backend = VOLVOXAI_BACKEND_CUDA;
    else return 0;
    return 1;
}

int volvoxai_engine_configure_backend(const char* name) {
    VolvoxAIEngineBackend previous_builtin;
    VolvoxAIEngineBackend builtin;
    int result = -1;
    if (!name || !name[0] || g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    if (g_loaded) goto done;
    previous_builtin = vx_backend_manager_current();
    if (volvoxai_builtin_backend_by_name(name, &builtin)) {
        if (vx_backend_manager_activate(builtin) != 0) goto done;
        if (vx_runtime_backend_stage() != 0) {
            (void)vx_backend_manager_activate(previous_builtin);
            goto done;
        }
        result = 0;
        goto done;
    }
done:
    volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_set_debug(int enabled) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    g_debug = enabled ? 1 : 0;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return 0;
}

int volvoxai_engine_debug(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    int enabled = g_debug;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return enabled;
}

static int volvoxai_engine_tensor_is_activation_locked(const T* tensor) {
    if (!tensor) return 0;
    if (tensor->is_graph_input) return 1;
    for (int node_index = 0; node_index < g_nn; node_index++) {
        const Node* node = &g_n[node_index];
        for (int output_index = 0; output_index < node->nout; output_index++) {
            if (!strcmp(node->outs[output_index].name, tensor->name)) return 1;
        }
    }
    return 0;
}

static long volvoxai_engine_tensor_row_capacity(const T* tensor) {
    if (!tensor || tensor->numel <= 0) return 0;
    if (tensor->ndim <= 0) return 1;
    int width = tensor->shape[tensor->ndim - 1];
    if (width <= 0 || tensor->numel % width != 0) return 0;
    return tensor->numel / width;
}

/* Row-aware operators may reduce or add a trailing feature dimension (for
 * example ArgMax and Embedding). The largest activation-side row capacity at
 * the graph tail therefore describes the legal scheduling range without
 * assuming a particular model output dtype or rank. Individual operators
 * still validate their stricter row contracts. */
static long volvoxai_engine_execution_row_capacity_locked(void) {
    if (!g_loaded || g_nn <= 0) return 0;
    const Node* final_node = &g_n[g_nn - 1];
    long capacity = volvoxai_engine_tensor_row_capacity(t_find(final_node->out));
    for (int input_index = 0; input_index < final_node->nin; input_index++) {
        T* input = t_find(final_node->ins[input_index].name);
        if (!volvoxai_engine_tensor_is_activation_locked(input)) continue;
        long input_capacity = volvoxai_engine_tensor_row_capacity(input);
        if (input_capacity > capacity) capacity = input_capacity;
    }
    return capacity;
}

static int volvoxai_engine_execution_row_valid_locked(int row) {
    if (row < -1) return 0;
    if (row < 0 || !g_loaded) return 1;
    long capacity = volvoxai_engine_execution_row_capacity_locked();
    return capacity > 0 && (long)row < capacity;
}

int volvoxai_engine_set_execution_row(int row) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked() || vx_decode_session_active_locked() ||
        !volvoxai_engine_execution_row_valid_locked(row)) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        return -1;
    }
    g_execution_row = row;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return 0;
}

int volvoxai_engine_execution_row(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    int row = g_execution_row;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return row;
}

static cJSON* volvoxai_engine_graph_interface(const char* key) {
    return g_graph_root ? cJSON_GetObjectItem(g_graph_root, key) : NULL;
}

int volvoxai_engine_graph_input_count(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    cJSON* inputs = volvoxai_engine_graph_interface("inputs");
    int count = cJSON_IsObject(inputs) ? cJSON_GetArraySize(inputs) : 0;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return count;
}

const char* volvoxai_engine_graph_input_name(int index) {
    if (index < 0) return NULL;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    cJSON* inputs = volvoxai_engine_graph_interface("inputs");
    cJSON* input = cJSON_IsObject(inputs) ? cJSON_GetArrayItem(inputs, index) : NULL;
    const char* name = input ? input->string : NULL;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return name;
}

int volvoxai_engine_graph_output_count(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    cJSON* outputs = volvoxai_engine_graph_interface("outputs");
    int count = cJSON_IsArray(outputs) ? cJSON_GetArraySize(outputs) : 0;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return count;
}

const char* volvoxai_engine_graph_output_name(int index) {
    if (index < 0) return NULL;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    cJSON* outputs = volvoxai_engine_graph_interface("outputs");
    cJSON* output = cJSON_IsArray(outputs)
        ? cJSON_GetArrayItem(outputs, index) : NULL;
    const char* name = cJSON_IsString(output) ? output->valuestring : NULL;
    if (took_model_lock) volvoxai_engine_model_unlock();
    return name;
}

int volvoxai_engine_init(const char* graph_path, const char* weights_path) {
    const char* paths[1] = { weights_path };
    return volvoxai_engine_init_with_weight_files(graph_path, paths, weights_path ? 1 : 0);
}

static int volvoxai_engine_init_with_weight_files_impl(const char* graph_path,
                                                       const char* const* weight_file_paths,
                                                       int weight_file_count) {
    if (g_loaded) volvoxai_engine_shutdown_impl();
    g_nt = 0;
    volvoxai_engine_tensor_name_index_rebuild();
    g_nn = 0;
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    g_first_input[0] = 0;
    g_active_row = -1;
    g_prefix_rows = 0;
    g_prefix_row_capacity = 0;
    g_execution_row = -1;
    vx_incremental_invalidate_locked();
    vx_runtime_backend_reset();
    if (volvoxai_engine_load_weight_files(weight_file_paths, weight_file_count) != 0) goto fail;
    /* Slice requested banks before the graph is built so every downstream shape
     * check sees the staged extent rather than the bank's full one. */
    if (vx_bank_residency_apply() != 0) goto fail;
    if (build_graph(graph_path) != 0) goto fail;
    vx_bank_residency_bind_nodes();
    if (prepack_cpu_weights() != 0) goto fail;
    g_weight_caches_dirty = 0;
    g_loaded = 1;
    volvoxai_engine_model_generation_advance_locked();
    return 0;
fail:
    volvoxai_engine_shutdown_impl();
    return -1;
}

int volvoxai_engine_init_with_weight_files(const char* graph_path,
                                           const char* const* weight_file_paths,
                                           int weight_file_count) {
    if (!graph_path || !graph_path[0] || weight_file_count < 0 ||
        weight_file_count > MAX_WEIGHT_FILES ||
        (weight_file_count > 0 && !weight_file_paths)) return -1;
    for (int index = 0; index < weight_file_count; index++) {
        if (!weight_file_paths[index] || !weight_file_paths[index][0]) return -1;
    }
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_init_with_weight_files_impl(
        graph_path, weight_file_paths, weight_file_count);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

float* volvoxai_engine_input_ptr(const char* name, long* numel) {
    if (numel) *numel = 0;
    volvoxai_engine_metadata_lock();
    T* t = t_find(name && name[0] ? name : g_first_input);
    float* data = NULL;
    if (t && t->is_graph_input && t->dtype == T_F32 && t->data) {
        /* The returned pointer is mutable, so a later incremental call cannot
         * know which bytes changed. Force one complete refresh. */
        vx_incremental_invalidate_locked();
        if (vx_runtime_backend_has_graph() && t->numel > 0 && t->elem_size &&
            (size_t)t->numel <= SIZE_MAX / t->elem_size)
            vx_runtime_backend_mark_host(
                t->data, (size_t)t->numel * t->elem_size, 0);
        if (numel) *numel = t->numel;
        data = t->data;
    }
    volvoxai_engine_metadata_unlock();
    return data;
}

int volvoxai_engine_set_input_raw(const char* name, int dtype, const void* data, size_t nbytes) {
    if (!name || !name[0] || (!data && nbytes > 0)) return -1;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    T* tensor = t_find(name);
    int rc = -1;
    if (tensor && tensor->is_graph_input && tensor->dtype == dtype && tensor->data &&
        tensor->numel >= 0 && (size_t)tensor->numel <= SIZE_MAX / tensor->elem_size &&
        nbytes == (size_t)tensor->numel * tensor->elem_size) {
        if (nbytes > 0) memcpy(tensor->data, data, nbytes);
        vx_incremental_mark_tensor_locked(tensor);
        if (vx_runtime_backend_has_graph() && nbytes > 0)
            vx_runtime_backend_mark_host(tensor->data, nbytes, 0);
        rc = 0;
    }
    volvoxai_engine_metadata_unlock();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_set_input_f32(const char* name, const float* data, long numel) {
    if (!name || !name[0] || numel <= 0 || !data) return -1;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    volvoxai_engine_metadata_lock();
    T* tensor = t_find(name);
    int rc = -1;
    if (tensor && tensor->is_graph_input && tensor->data && tensor->numel == numel) {
        if (tensor->dtype == T_F32 && tensor->elem_size == sizeof(float) &&
            (size_t)numel <= SIZE_MAX / sizeof(float)) {
            memcpy(tensor->data, data, (size_t)numel * sizeof(float));
            rc = 0;
        } else if ((tensor->dtype == T_I8 || tensor->dtype == T_U8) &&
                   tensor->elem_size == 1 && tensor->quantization.valid &&
                   isfinite(tensor->quantization.scale) &&
                   tensor->quantization.scale > 0.0f && numel <= UINT32_MAX &&
                   ((tensor->dtype == T_I8 && tensor->quantization.zero_point >= -128 &&
                     tensor->quantization.zero_point <= 127) ||
                    (tensor->dtype == T_U8 && tensor->quantization.zero_point >= 0 &&
                     tensor->quantization.zero_point <= 255))) {
            int8_t zero_i8 = (int8_t)tensor->quantization.zero_point;
            uint8_t zero_u8 = (uint8_t)tensor->quantization.zero_point;
            const void* zero = tensor->dtype == T_I8
                ? (const void*)&zero_i8 : (const void*)&zero_u8;
            uint32_t dtype = (uint32_t)tensor->dtype;
            if (quantize_linear_typed(data, &tensor->quantization.scale, zero,
                    dtype, tensor->data, dtype,
                    (uint32_t)numel) == 1) rc = 0;
        }
        if (rc == 0) {
            size_t nbytes = (size_t)numel * tensor->elem_size;
            vx_incremental_mark_tensor_locked(tensor);
            if (vx_runtime_backend_has_graph() && nbytes > 0)
                vx_runtime_backend_mark_host(tensor->data, nbytes, 0);
        }
    }
    volvoxai_engine_metadata_unlock();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_input_affine_quantization(const char* name,
                                              float* scale,
                                              int* zero_point) {
    int result = -1;
    if (!name || !name[0] || !scale || !zero_point) return -1;
    volvoxai_engine_metadata_lock();
    T* tensor = t_find(name);
    if (tensor && tensor->is_graph_input) {
        result = tensor->quantization.valid ? 1 : 0;
        *scale = tensor->quantization.valid ? tensor->quantization.scale : 0.0f;
        *zero_point = tensor->quantization.valid
            ? tensor->quantization.zero_point : 0;
    }
    volvoxai_engine_metadata_unlock();
    return result;
}

int volvoxai_engine_is_graph_input(const char* name) {
    volvoxai_engine_metadata_lock();
    T* t = t_find(name && name[0] ? name : g_first_input);
    int result = t && t->is_graph_input;
    volvoxai_engine_metadata_unlock();
    return result;
}

static int volvoxai_engine_tensor_weight_file_index_raw(const char* name) {
    if (!name || !name[0]) return -1;
    for (int f = g_weight_file_count - 1; f >= 0; f--) {
        if (safetensors_find_tensor(&g_weight_files[f], name)) return f;
    }
    return -1;
}

int volvoxai_engine_tensor_is_model_weight_locked(const char* name) {
    if (!name || !name[0]) return 0;
    T* tensor = t_find(name);
    if (!tensor || tensor->is_graph_input || !tensor->data || tensor->numel <= 0) return 0;
    int file_index = volvoxai_engine_tensor_weight_file_index_raw(name);
    if (file_index < 0) return 0;
    SafetensorsTensor* stored = safetensors_find_tensor_mutable(&g_weight_files[file_index], name);
    if (!stored || stored->dtype != safetensors_dtype_from_engine(tensor->dtype) ||
        tensor->elem_size == 0 || (size_t)tensor->numel > SIZE_MAX / tensor->elem_size ||
        stored->nbytes != (size_t)tensor->numel * tensor->elem_size || stored->data != tensor->data) {
        return 0;
    }
    return 1;
}

/* A generic F32 operator may widen a safetensors F16 constant in place. The
 * execution tensor then intentionally no longer aliases the stored source,
 * so the strict mutable-weight predicate above becomes false even though the
 * widened copy is still immutable for this context. Keep that narrower
 * provenance rule separate from public mutable-weight identity. */
static int volvoxai_engine_tensor_has_immutable_weight_origin_locked(
        const T* tensor) {
    int file_index;
    const SafetensorsTensor* stored;
    if (!tensor || !tensor->name[0] || tensor->is_graph_input ||
        !tensor->data || tensor->numel <= 0)
        return 0;
    if (volvoxai_engine_tensor_is_model_weight_locked(tensor->name))
        return 1;
    file_index = volvoxai_engine_tensor_weight_file_index_raw(tensor->name);
    if (file_index < 0 || tensor->dtype != T_F32 ||
        tensor->elem_size != sizeof(float) || !tensor->owns)
        return 0;
    stored = safetensors_find_tensor(&g_weight_files[file_index],
                                     tensor->name);
    if (!stored || stored->dtype != SAFETENSORS_DTYPE_F16 ||
        stored->ndim != tensor->ndim)
        return 0;
    for (int axis = 0; axis < tensor->ndim; axis++)
        if (stored->shape[axis] != tensor->shape[axis]) return 0;
    return 1;
}

int volvoxai_engine_sync_model_weights_locked(void) {
    for (int index = 0; index < g_nt; index++) {
        T* tensor = &g_t[index];
        if (volvoxai_engine_tensor_is_model_weight_locked(tensor->name) &&
            vk_sync_host_tensor(tensor) != 0) return -1;
    }
    return 0;
}

int volvoxai_engine_demote_preloaded_logical_tensors_locked(void) {
    VxEngineState* state = vx_engine_state_current();
    const VxDynamicShapeMaximumLayout* maximum;
    if (!state) return -1;
    maximum = &state->dynamic_shape_maximum_layout;
    if (!maximum->configured || maximum->tensor_count <= 0 ||
        !maximum->tensor_indices)
        return -1;
    for (int logical_index = 0; logical_index < maximum->tensor_count;
         logical_index++) {
        int tensor_index = maximum->tensor_indices[logical_index];
        T* tensor;
        size_t bytes;
        if (tensor_index < 0 || tensor_index >= g_nt) return -1;
        tensor = &g_t[tensor_index];
        if (!tensor->data || tensor->numel <= 0 || !tensor->elem_size ||
            (size_t)tensor->numel > SIZE_MAX / tensor->elem_size)
            return -1;
        bytes = (size_t)tensor->numel * tensor->elem_size;
        /* Exact bootstrap identity only. Parsed affine arrays, conversion
         * caches, LUTs, and backend synthetic constants are not logical T
         * buffers and therefore retain their invariant lifetime. */
        vx_runtime_backend_demote_weight(tensor->data, bytes);
    }
    return 0;
}

int volvoxai_engine_retain_preloaded_model_weights_locked(void) {
    for (int index = 0; index < g_nt; index++) {
        T* tensor = &g_t[index];
        size_t bytes;
        if (!volvoxai_engine_tensor_has_immutable_weight_origin_locked(tensor) ||
            !tensor->data || tensor->numel <= 0 || !tensor->elem_size ||
            (size_t)tensor->numel > SIZE_MAX / tensor->elem_size)
            continue;
        bytes = (size_t)tensor->numel * tensor->elem_size;
        /* Classification-only: the backend promotes a slot only when the
         * bootstrap forward actually materialized it. Unused weights are not
         * allocated or uploaded as a side effect of the domain proof. */
        vx_runtime_backend_retain_weight(tensor->data, bytes);
    }
    return 0;
}

int volvoxai_engine_is_model_weight(const char* name) {
    volvoxai_engine_metadata_lock();
    int result = !volvoxai_engine_tensor_is_quantization_parameter_locked(name) &&
        volvoxai_engine_tensor_is_model_weight_locked(name);
    volvoxai_engine_metadata_unlock();
    return result;
}

int volvoxai_engine_copy_tensor_f32(const char* name, float* out, long numel) {
    if (!out || numel < 0) return -1;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    const char* tensor_name = name && name[0] ? name : g_first_input;
    T* t = (volvoxai_engine_tensor_is_removed_output(tensor_name) ||
            volvoxai_engine_tensor_is_quantization_parameter_locked(tensor_name))
        ? NULL : t_find(tensor_name);
    if (!t || t->numel != numel) { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    if (vk_sync_host_tensor(t) != 0) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        return -1;
    }
    if (t->dtype == T_F32) materialize_tensor_f32(t);
    if (t->dtype == T_F32) memcpy(out, t->data, (size_t)numel * sizeof(float));
    else for (long i = 0; i < numel; i++) {
        if (t->dtype == T_F16) out[i] = volvoxai_engine_adapter_f16_at(t->data, (size_t)i);
        else if (t->dtype == T_I8) out[i] = (float)((const int8_t*)t->data)[i];
        else if (t->dtype == T_U8) out[i] = (float)((const uint8_t*)t->data)[i];
        else if (t->dtype == T_I32) {
            int32_t value; memcpy(&value, (const unsigned char*)t->data + (size_t)i * sizeof(value), sizeof(value));
            out[i] = (float)value;
        } else { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    }
    if (took_model_lock) volvoxai_engine_model_unlock();
    return 0;
}

int volvoxai_engine_copy_tensor_raw(const char* name, void* out, size_t nbytes) {
    if (!out && nbytes > 0) return -1;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    const char* tensor_name = name && name[0] ? name : g_first_input;
    T* t = (volvoxai_engine_tensor_is_removed_output(tensor_name) ||
            volvoxai_engine_tensor_is_quantization_parameter_locked(tensor_name))
        ? NULL : t_find(tensor_name);
    size_t expected = t ? (size_t)t->numel * t->elem_size : 0;
    if (!t || expected != nbytes) { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    if (vk_sync_host_tensor(t) != 0) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        return -1;
    }
    if (t->dtype == T_F32) materialize_tensor_f32(t);
    if (nbytes > 0) memcpy(out, t->data, nbytes);
    if (took_model_lock) volvoxai_engine_model_unlock();
    return 0;
}

int volvoxai_engine_tensor_info(const char* name, long* numel, int* shape, int* ndim) {
    volvoxai_engine_metadata_lock();
    const char* tensor_name = name && name[0] ? name : g_first_input;
    T* t = (volvoxai_engine_tensor_is_removed_output(tensor_name) ||
            volvoxai_engine_tensor_is_quantization_parameter_locked(tensor_name))
        ? NULL : t_find(tensor_name);
    if (!t) { volvoxai_engine_metadata_unlock(); return -1; }
    if (numel) *numel = t->numel;
    if (ndim) *ndim = t->ndim;
    if (shape) {
        for (int i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    }
    volvoxai_engine_metadata_unlock();
    return 0;
}

int volvoxai_engine_tensor_info_ex(const char* name, long* numel, int* shape, int* ndim, int* dtype, size_t* elem_size) {
    volvoxai_engine_metadata_lock();
    const char* tensor_name = name && name[0] ? name : g_first_input;
    T* t = (volvoxai_engine_tensor_is_removed_output(tensor_name) ||
            volvoxai_engine_tensor_is_quantization_parameter_locked(tensor_name))
        ? NULL : t_find(tensor_name);
    if (!t) { volvoxai_engine_metadata_unlock(); return -1; }
    if (numel) *numel = t->numel;
    if (ndim) *ndim = t->ndim;
    if (shape) {
        for (int i = 0; i < t->ndim; i++) shape[i] = t->shape[i];
    }
    if (dtype) *dtype = t->dtype;
    if (elem_size) *elem_size = t->elem_size;
    volvoxai_engine_metadata_unlock();
    return 0;
}

int volvoxai_engine_tensor_weight_file_index(const char* name) {
    if (volvoxai_engine_tensor_is_quantization_parameter_locked(name)) return -1;
    return volvoxai_engine_tensor_weight_file_index_raw(name);
}

static int volvoxai_engine_set_tensor_raw_impl(const char* name, int dtype, const void* data, size_t nbytes) {
    if (!name || (!data && nbytes > 0)) return -1;
    if (volvoxai_engine_tensor_is_removed_output(name)) return -1;
    if (volvoxai_engine_tensor_is_quantization_parameter_locked(name)) return -1;
    if (volvoxai_engine_merged_contains_tensor(name)) return -1;
    T* t = t_find(name);
    if (!t || t->is_graph_input || t->dtype != dtype) return -1;
    size_t want = (size_t)t->numel * t->elem_size;
    if (nbytes != want) return -1;
    if (vk_sync_host_tensor(t) != 0) return -1;
    if (nbytes > 0) memcpy(t->data, data, nbytes);
    if (vx_runtime_backend_has_graph() && nbytes > 0)
        vx_runtime_backend_mark_host(t->data, nbytes, 1);

    int file_index = volvoxai_engine_tensor_weight_file_index_raw(name);
    if (file_index >= 0) {
        VxDataType st_dtype = safetensors_dtype_from_engine(dtype);
        SafetensorsTensor* st = safetensors_find_tensor_mutable(&g_weight_files[file_index], name);
        if (st && st->dtype == st_dtype && st->nbytes == nbytes) {
            safetensors_set_tensor_data(&g_weight_files[file_index], name, data, nbytes);
        }
    }
    vk_mark_owned_tensors_host_dirty();
    g_weight_caches_dirty = 1;
    volvoxai_engine_model_generation_advance_locked();
    return 0;
}

int volvoxai_engine_set_tensor_raw(const char* name, int dtype, const void* data, size_t nbytes) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_set_tensor_raw_impl(name, dtype, data, nbytes);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_add_model_tensor_raw_locked(const char* name, const int* shape,
                                                int ndim, int dtype,
                                                const void* data, size_t nbytes) {
    if (!name || !name[0] || strlen(name) >= sizeof(g_t[0].name) ||
        !shape || ndim < 0 || ndim > 8 || (!data && nbytes > 0)) return -1;
    size_t elem_size = engine_dtype_size_local(dtype);
    VxDataType st_dtype = safetensors_dtype_from_engine(dtype);
    if (!elem_size || st_dtype == SAFETENSORS_DTYPE_UNKNOWN) return -1;
    long numel = 1;
    for (int i = 0; i < ndim; i++) {
        if (shape[i] <= 0 || numel > LONG_MAX / shape[i]) return -1;
        numel *= shape[i];
    }
    if ((size_t)numel > SIZE_MAX / elem_size || nbytes != (size_t)numel * elem_size) return -1;

    int rc = -1;
    int mutation_prepared = 0;
    if (!g_loaded || g_weight_file_count <= 0 || g_nt >= MAXT || t_find(name)) goto done;
    if (volvoxai_engine_prepare_tensor_table_mutation() != 0) goto done;
    mutation_prepared = 1;
    SafetensorsFile* weights = &g_weight_files[0];
    if (safetensors_add_tensor(weights, name, st_dtype, shape, ndim, data, nbytes) != 0) goto done;
    SafetensorsTensor* stored = safetensors_find_tensor_mutable(weights, name);
    if (!stored || (!stored->data && nbytes > 0)) {
        safetensors_remove_tensor(weights, name);
        goto done;
    }
    T* tensor = &g_t[g_nt];
    memset(tensor, 0, sizeof(*tensor));
    strncpy(tensor->name, name, sizeof(tensor->name) - 1);
    tensor->ndim = ndim;
    for (int i = 0; i < ndim; i++) tensor->shape[i] = shape[i];
    tensor->data = (float*)stored->data;
    tensor->numel = numel;
    tensor->owns = 0;
    tensor->dtype = dtype;
    tensor->elem_size = elem_size;
    g_nt++;
    volvoxai_engine_tensor_name_index_add(g_nt - 1);
    g_weight_caches_dirty = 1;
    volvoxai_engine_model_generation_advance_locked();
    rc = 0;
done:
    if (mutation_prepared) volvoxai_engine_finish_tensor_table_mutation();
    return rc;
}

int volvoxai_engine_add_model_tensor_raw(const char* name, const int* shape, int ndim,
                                         int dtype, const void* data, size_t nbytes) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_add_model_tensor_raw_locked(
        name, shape, ndim, dtype, data, nbytes);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_remove_model_tensor_locked(const char* name) {
    if (!name || !name[0]) return -1;
    int rc = -1;
    int mutation_prepared = 0;
    if (!g_loaded || g_merged_adapter_count ||
        volvoxai_engine_tensor_is_quantization_parameter_locked(name)) goto done;
    T* tensor = t_find(name);
    if (!tensor || tensor->is_graph_input) goto done;
    for (int i = 0; i < g_nn; i++) {
        for (int j = 0; j < g_n[i].nin; j++) if (!strcmp(g_n[i].ins[j].name, name)) goto done;
        for (int j = 0; j < g_n[i].nout; j++) if (!strcmp(g_n[i].outs[j].name, name)) goto done;
        if (!strcmp(g_n[i].out, name)) goto done;
    }
    int file_index = -1;
    for (int f = 0; f < g_weight_file_count; f++) {
        if (safetensors_find_tensor(&g_weight_files[f], name)) {
            if (file_index >= 0) goto done; /* Ambiguous shadowed shard. */
            file_index = f;
        }
    }
    if (file_index < 0) goto done;
    int tensor_index = (int)(tensor - g_t);
    if (tensor_index < 0 || tensor_index >= g_nt) goto done;
    /* The reset below discards every backend slot, not only the tensor being
     * removed. Preserve other device-authoritative trained weights first. */
    if (volvoxai_engine_sync_model_weights_locked() != 0) goto done;
    if (volvoxai_engine_prepare_tensor_table_mutation() != 0) goto done;
    mutation_prepared = 1;
    vx_runtime_backend_reset();
    if (safetensors_remove_tensor(&g_weight_files[file_index], name) != 0) goto done;
    optimizer_state_remove(name);
    if (tensor_index + 1 < g_nt) {
        memmove(&g_t[tensor_index], &g_t[tensor_index + 1],
                (size_t)(g_nt - tensor_index - 1) * sizeof(g_t[0]));
    }
    g_nt--;
    volvoxai_engine_tensor_name_index_invalidate();
    memset(&g_t[g_nt], 0, sizeof(g_t[0]));
    g_weight_caches_dirty = 1;
    volvoxai_engine_model_generation_advance_locked();
    rc = 0;
done:
    if (mutation_prepared) volvoxai_engine_finish_tensor_table_mutation();
    return rc;
}

int volvoxai_engine_remove_model_tensor(const char* name) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_remove_model_tensor_locked(name);
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_set_tensor_f32(const char* name, const float* data, long numel) {
    if (!name || !data || numel < 0) return -1;
    return volvoxai_engine_set_tensor_raw(name, T_F32, data, (size_t)numel * sizeof(float));
}

int volvoxai_engine_save_weight_file(int weight_file_index, const char* path) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    int rc = -1;
    if (volvoxai_engine_training_accumulation_pending() ||
        g_merged_adapter_count || weight_file_index < 0 ||
        weight_file_index >= g_weight_file_count) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    const char* out = (path && path[0]) ? path : g_weight_paths[weight_file_index];
    if (out && out[0]) {
        for (int tensor_index = 0; tensor_index < g_nt; tensor_index++) {
            T* tensor = &g_t[tensor_index];
            if (volvoxai_engine_tensor_weight_file_index_raw(tensor->name) ==
                    weight_file_index &&
                vk_sync_host_tensor(tensor) != 0) {
                out = NULL;
                break;
            }
        }
    }
    if (out && out[0]) {
        rc = safetensors_save(out, &g_weight_files[weight_file_index]);
    }
    volvoxai_engine_model_unlock();
    return rc;
}

typedef struct {
    const void* data;
    size_t nbytes;
    int dtype;
} EngineAdapterBinding;

static cJSON* volvoxai_engine_json_string(cJSON* object, const char* name) {
    cJSON* item = object ? cJSON_GetObjectItem(object, name) : NULL;
    return cJSON_IsString(item) && item->valuestring ? item : NULL;
}

static int volvoxai_engine_adapter_kind(const char* name, VxAdapterKind* out) {
    if (!name || !out) return -1;
    if (strcmp(name, "lora")) return -1;
    *out = VX_ADAPTER_LORA;
    return 0;
}

static EngineAdapterBinding volvoxai_engine_adapter_binding(const char* name,
                                                    const char* const* tensor_names,
                                                    const void* const* tensor_data,
                                                    const int* tensor_dtypes,
                                                    const size_t* tensor_nbytes,
                                                    int tensor_count) {
    EngineAdapterBinding out = {0};
    if (!name) return out;
    for (int i = 0; i < tensor_count; i++) {
        if (tensor_names[i] && !strcmp(tensor_names[i], name)) {
            out.data = tensor_data[i];
            out.dtype = tensor_dtypes[i];
            out.nbytes = tensor_nbytes[i];
            return out;
        }
    }
    return out;
}

static float volvoxai_engine_adapter_f16(uint16_t h) {
    uint32_t sign = ((uint32_t)h & 0x8000u) << 16;
    uint32_t exp = ((uint32_t)h >> 10) & 0x1fu;
    uint32_t mant = (uint32_t)h & 0x03ffu;
    uint32_t bits;
    if (exp == 0) {
        if (!mant) bits = sign;
        else {
            int e = -14;
            while (!(mant & 0x0400u)) { mant <<= 1; e--; }
            mant &= 0x03ffu;
            bits = sign | ((uint32_t)(e + 127) << 23) | (mant << 13);
        }
    } else if (exp == 31) bits = sign | 0x7f800000u | (mant << 13);
    else bits = sign | ((exp + 112u) << 23) | (mant << 13);
    float value;
    memcpy(&value, &bits, sizeof(value));
    return value;
}

static float volvoxai_engine_adapter_f16_at(const void* data, size_t index) {
    const unsigned char* bytes = (const unsigned char*)data + index * 2u;
    return volvoxai_engine_adapter_f16((uint16_t)bytes[0] | ((uint16_t)bytes[1] << 8));
}

static float volvoxai_engine_adapter_f32_at(const void* data, size_t index) {
    float value;
    memcpy(&value, (const unsigned char*)data + index * sizeof(float), sizeof(value));
    return value;
}

static uint16_t volvoxai_engine_adapter_f16_bits(float value) {
    uint32_t bits;
    memcpy(&bits, &value, sizeof(bits));
    uint16_t sign = (uint16_t)((bits >> 16) & 0x8000u);
    uint32_t exponent = (bits >> 23) & 0xffu;
    uint32_t mantissa = bits & 0x7fffffu;
    if (exponent == 0xffu) {
        return (uint16_t)(sign | 0x7c00u | (mantissa ? 0x0200u : 0u));
    }
    int half_exponent = (int)exponent - 127 + 15;
    if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
    if (half_exponent <= 0) {
        if (half_exponent < -10) return sign;
        mantissa |= 0x800000u;
        unsigned shift = (unsigned)(14 - half_exponent);
        uint32_t rounded = mantissa >> shift;
        uint32_t remainder = mantissa & ((1u << shift) - 1u);
        uint32_t halfway = 1u << (shift - 1u);
        if (remainder > halfway || (remainder == halfway && (rounded & 1u))) rounded++;
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = mantissa >> 13;
    uint32_t remainder = mantissa & 0x1fffu;
    if (remainder > 0x1000u || (remainder == 0x1000u && (rounded & 1u))) {
        rounded++;
        if (rounded == 0x400u) {
            rounded = 0;
            half_exponent++;
            if (half_exponent >= 31) return (uint16_t)(sign | 0x7c00u);
        }
    }
    return (uint16_t)(sign | (uint16_t)(half_exponent << 10) | (uint16_t)rounded);
}

static void volvoxai_engine_adapter_store_weight(unsigned char* data, int dtype, size_t index, float value) {
    if (dtype == T_F32) {
        memcpy(data + index * sizeof(float), &value, sizeof(value));
    } else {
        uint16_t bits = volvoxai_engine_adapter_f16_bits(value);
        data[index * 2u] = (unsigned char)(bits & 0xffu);
        data[index * 2u + 1u] = (unsigned char)(bits >> 8);
    }
}

static int volvoxai_engine_adapter_tensor_spec(VxAdapterTensorSpec* out, EngineAdapterBinding binding,
                                      int rows, int cols, int transpose, void** owned) {
    if (!out || !binding.data || rows <= 0 || cols <= 0 || !owned) return -1;
    size_t elem_size = binding.dtype == T_F32 ? sizeof(float) : binding.dtype == T_F16 ? 2u : 0u;
    size_t count = (size_t)rows * cols;
    if (!elem_size || count > SIZE_MAX / elem_size || binding.nbytes != count * elem_size) return -1;
    memset(out, 0, sizeof(*out));
    out->dtype = binding.dtype;
    out->nbytes = binding.nbytes;
    out->data = binding.data;
    out->rows = rows;
    out->cols = cols;
    if (!transpose) return 0;
    float* values = (float*)malloc(count * sizeof(float));
    if (!values) return -1;
    for (int r = 0; r < rows; r++) {
        for (int c = 0; c < cols; c++) {
            size_t src = (size_t)r * cols + c;
            float value = binding.dtype == T_F32 ? volvoxai_engine_adapter_f32_at(binding.data, src) :
                          volvoxai_engine_adapter_f16_at(binding.data, src);
            values[(size_t)c * rows + r] = value;
        }
    }
    out->data = values;
    out->dtype = T_F32;
    out->nbytes = count * sizeof(float);
    out->rows = cols;
    out->cols = rows;
    *owned = values;
    return 0;
}

static int volvoxai_engine_adapter_manifest_specs_match(cJSON* target, const char* a_name, const char* b_name,
                                               EngineAdapterBinding a, EngineAdapterBinding b,
                                               int a_rows, int a_cols, int b_rows, int b_cols) {
    cJSON* specs = cJSON_GetObjectItemCaseSensitive(target, "tensors");
    if (!specs) return 1;
    if (!cJSON_IsArray(specs) || cJSON_GetArraySize(specs) != 2) return 0;
    int seen_a = 0, seen_b = 0;
    for (int i = 0; i < 2; i++) {
        cJSON* spec = cJSON_GetArrayItem(specs, i);
        cJSON* role = cJSON_GetObjectItemCaseSensitive(spec, "role");
        cJSON* name = cJSON_GetObjectItemCaseSensitive(spec, "name");
        cJSON* shape = cJSON_GetObjectItemCaseSensitive(spec, "shape");
        cJSON* dtype = cJSON_GetObjectItemCaseSensitive(spec, "dtype");
        if (!cJSON_IsString(role) || !cJSON_IsString(name) || !cJSON_IsArray(shape) ||
            cJSON_GetArraySize(shape) != 2 || !cJSON_IsString(dtype)) return 0;
        int is_a = !strcmp(role->valuestring, "a");
        if ((!is_a && strcmp(role->valuestring, "b")) || (is_a ? seen_a++ : seen_b++)) return 0;
        const char* expected_name = is_a ? a_name : b_name;
        EngineAdapterBinding binding = is_a ? a : b;
        int rows = is_a ? a_rows : b_rows;
        int cols = is_a ? a_cols : b_cols;
        cJSON* row = cJSON_GetArrayItem(shape, 0);
        cJSON* col = cJSON_GetArrayItem(shape, 1);
        const char* expected_dtype = binding.dtype == T_F32 ? "F32" : binding.dtype == T_F16 ? "F16" : "";
        if (strcmp(name->valuestring, expected_name) || !cJSON_IsNumber(row) || !cJSON_IsNumber(col) ||
            row->valuedouble != rows || col->valuedouble != cols || strcmp(dtype->valuestring, expected_dtype)) return 0;
    }
    return seen_a == 1 && seen_b == 1;
}

static T* volvoxai_engine_node_input(Node* node, const char* key) {
    if (!node || !key) return NULL;
    for (int i = 0; i < node->nin; i++) {
        if (!strcmp(node->ins[i].key, key)) return t_find(node->ins[i].name);
    }
    return NULL;
}

static int volvoxai_engine_linear_weight_layout_impl(const char* weight_name, int* d_in, int* d_out, int* out_in) {
    T* weight = t_find(weight_name);
    if (!weight || weight->ndim != 2) return -1;
    int found = 0;
    int resolved_in = 0, resolved_out = 0, resolved_layout = 0;
    for (int i = 0; i < g_nn; i++) {
        Node* node = &g_n[i];
        if (strcmp(node->op, "MatMul") && strcmp(node->op, "Gemm") && strcmp(node->op, "Linear")) continue;
        if (volvoxai_engine_node_input(node, "weight") != weight) continue;
        T* input = volvoxai_engine_node_input(node, "input");
        if (!input) input = volvoxai_engine_node_input(node, "x");
        if (!input) input = volvoxai_engine_node_input(node, "a");
        T* output = t_find(node->out);
        if (!input || input->ndim <= 0 || !output || output->ndim <= 0) return -1;
        int din = input->shape[input->ndim - 1];
        int dout = output->shape[output->ndim - 1];
        int layout = -1;
        cJSON* layout_json = node->params ? cJSON_GetObjectItem(node->params, "weight_layout") : NULL;
        if (cJSON_IsString(layout_json) && layout_json->valuestring) {
            const char* value = layout_json->valuestring;
            if (!strcmp(value, "dout_din")) layout = 1;
            else if (!strcmp(value, "din_dout")) layout = 0;
            else return -1;
        }
        cJSON* trans_b = node->params ? cJSON_GetObjectItem(node->params, "transB") : NULL;
        if (layout < 0 && cJSON_IsNumber(trans_b) && trans_b->valueint) layout = 1;
        int matches_in_out = weight->shape[0] == din && weight->shape[1] == dout;
        int matches_out_in = weight->shape[0] == dout && weight->shape[1] == din;
        if (layout < 0) {
            if (matches_in_out && !matches_out_in) layout = 0;
            else if (matches_out_in && !matches_in_out) layout = 1;
            else if (matches_in_out) layout = 0;
            else return -1;
        }
        if ((layout == 0 && !matches_in_out) || (layout == 1 && !matches_out_in)) return -1;
        if (found && (resolved_in != din || resolved_out != dout || resolved_layout != layout)) return -1;
        found = 1; resolved_in = din; resolved_out = dout; resolved_layout = layout;
    }
    if (!found) return -1;
    if (d_in) *d_in = resolved_in;
    if (d_out) *d_out = resolved_out;
    if (out_in) *out_in = resolved_layout;
    return 0;
}

int volvoxai_engine_linear_weight_layout(const char* weight_name, int* d_in, int* d_out, int* out_in) {
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_linear_weight_layout_impl(weight_name, d_in, d_out, out_in);
    volvoxai_engine_metadata_unlock();
    return rc;
}

static int volvoxai_engine_adapter_stage_json_impl(const char* manifest_json,
                                          const char* const* tensor_names, const void* const* tensor_data,
                                          const int* tensor_dtypes, const size_t* tensor_nbytes, int tensor_count) {
    if (!manifest_json || tensor_count < 0 ||
        (tensor_count > 0 && (!tensor_names || !tensor_data || !tensor_dtypes || !tensor_nbytes))) return -1;
    for (int i = 0; i < tensor_count; i++) {
        if (!tensor_names[i] || !tensor_names[i][0] || !tensor_data[i]) return -1;
        for (int j = 0; j < i; j++) if (!strcmp(tensor_names[i], tensor_names[j])) return -1;
    }
    cJSON* wrapper = cJSON_Parse(manifest_json);
    if (!wrapper) return -1;
    cJSON* encoded = volvoxai_engine_json_string(wrapper, "volvox_adapter_manifest");
    cJSON* manifest = encoded ? cJSON_Parse(encoded->valuestring) : wrapper;
    if (!manifest) { cJSON_Delete(wrapper); return -1; }
    cJSON* format = volvoxai_engine_json_string(manifest, "format");
    cJSON* adapter_id = volvoxai_engine_json_string(manifest, "adapter_id");
    cJSON* version_id = volvoxai_engine_json_string(manifest, "version_id");
    cJSON* root_kind = volvoxai_engine_json_string(manifest, "kind");
    cJSON* targets_json = cJSON_GetObjectItem(manifest, "targets");
    int count = cJSON_IsArray(targets_json) ? cJSON_GetArraySize(targets_json) : 0;
    VxAdapterKind manifest_kind;
    if (!format || strcmp(format->valuestring, "volvox.adapter.v1") || !adapter_id || !version_id || !root_kind ||
        volvoxai_engine_adapter_kind(root_kind->valuestring, &manifest_kind) != 0 || count <= 0) {
        if (manifest != wrapper) cJSON_Delete(manifest); cJSON_Delete(wrapper); return -1;
    }
    if (count > INT_MAX / 2 || tensor_count != count * 2) {
        if (manifest != wrapper) cJSON_Delete(manifest);
        cJSON_Delete(wrapper);
        return -1;
    }
    VxAdapterTargetSpec* targets = (VxAdapterTargetSpec*)calloc((size_t)count, sizeof(*targets));
    void** owned_a = (void**)calloc((size_t)count, sizeof(void*));
    void** owned_b = (void**)calloc((size_t)count, sizeof(void*));
    if (!targets || !owned_a || !owned_b) {
        free(targets); free(owned_a); free(owned_b);
        if (manifest != wrapper) cJSON_Delete(manifest); cJSON_Delete(wrapper); return -1;
    }
    int ok = 1;
    for (int i = 0; i < count && ok; i++) {
        cJSON* item = cJSON_GetArrayItem(targets_json, i);
        cJSON* weight_name = volvoxai_engine_json_string(item, "weight");
        cJSON* a_name = volvoxai_engine_json_string(item, "a");
        cJSON* b_name = volvoxai_engine_json_string(item, "b");
        cJSON* kind = volvoxai_engine_json_string(item, "kind");
        cJSON* layout = volvoxai_engine_json_string(item, "layout");
        cJSON* rank = cJSON_GetObjectItem(item, "rank");
        cJSON* alpha = cJSON_GetObjectItem(item, "alpha");
        cJSON* scale = cJSON_GetObjectItem(item, "scale");
        T* weight = weight_name ? t_find(weight_name->valuestring) : NULL;
        int base_d_in = 0, base_d_out = 0, base_out_in = 0;
        VxAdapterKind target_kind = manifest_kind;
        if (kind && (volvoxai_engine_adapter_kind(kind->valuestring, &target_kind) != 0 || target_kind != manifest_kind)) { ok = 0; break; }
        double rank_value = cJSON_IsNumber(rank) ? rank->valuedouble : 0.0;
        if (!weight_name || !a_name || !b_name || !weight || weight->ndim != 2 ||
            (weight->dtype != T_F32 && weight->dtype != T_F16) ||
            !cJSON_IsNumber(rank) || !isfinite(rank_value) || rank_value < 1.0 || rank_value > INT_MAX ||
            rank_value != floor(rank_value) ||
            !cJSON_IsNumber(alpha) || !isfinite(alpha->valuedouble) || alpha->valuedouble <= 0.0 ||
            (scale && (!cJSON_IsNumber(scale) || !isfinite(scale->valuedouble))) ||
            volvoxai_engine_linear_weight_layout_impl(weight_name->valuestring, &base_d_in, &base_d_out, &base_out_in) != 0) { ok = 0; break; }
        targets[i].kind = VX_ADAPTER_LORA;
        int peft = layout && !strcmp(layout->valuestring, "peft");
        if (!layout || (!peft && strcmp(layout->valuestring, "din_r_r_dout"))) { ok = 0; break; }
        targets[i].weight_name = weight_name->valuestring;
        targets[i].d_in = base_d_in; targets[i].d_out = base_d_out; targets[i].rank = (int)rank_value;
        targets[i].alpha = (float)alpha->valuedouble;
        targets[i].scale = cJSON_IsNumber(scale) ? (float)scale->valuedouble : NAN;
        targets[i].a_name = a_name->valuestring; targets[i].b_name = b_name->valuestring;
        EngineAdapterBinding a = volvoxai_engine_adapter_binding(a_name->valuestring, tensor_names, tensor_data, tensor_dtypes, tensor_nbytes, tensor_count);
        EngineAdapterBinding b = volvoxai_engine_adapter_binding(b_name->valuestring, tensor_names, tensor_data, tensor_dtypes, tensor_nbytes, tensor_count);
        int a_rows = peft ? targets[i].rank : targets[i].d_in;
        int a_cols = peft ? targets[i].d_in : targets[i].rank;
        int b_rows = peft ? targets[i].d_out : targets[i].rank;
        int b_cols = peft ? targets[i].rank : targets[i].d_out;
        if (!volvoxai_engine_adapter_manifest_specs_match(item, a_name->valuestring, b_name->valuestring,
                                                 a, b, a_rows, a_cols, b_rows, b_cols) ||
            volvoxai_engine_adapter_tensor_spec(&targets[i].a, a, a_rows, a_cols, peft, &owned_a[i]) != 0 ||
            volvoxai_engine_adapter_tensor_spec(&targets[i].b, b, b_rows, b_cols, peft, &owned_b[i]) != 0) { ok = 0; break; }
    }
    char* canonical_manifest = cJSON_PrintUnformatted(manifest);
    VxAdapterVersionSpec spec = { adapter_id->valuestring, version_id->valuestring, targets, count, canonical_manifest };
    int rc = ok && canonical_manifest ? vx_adapter_stage(&spec) : -1;
    free(canonical_manifest);
    for (int i = 0; i < count; i++) {
        free(owned_a[i]); free(owned_b[i]);
    }
    free(owned_a); free(owned_b); free(targets);
    if (manifest != wrapper) cJSON_Delete(manifest);
    cJSON_Delete(wrapper);
    return rc;
}

int volvoxai_engine_adapter_stage_json(const char* manifest_json,
                              const char* const* tensor_names, const void* const* tensor_data,
                              const int* tensor_dtypes, const size_t* tensor_nbytes, int tensor_count) {
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_adapter_stage_json_impl(manifest_json, tensor_names, tensor_data,
                                            tensor_dtypes, tensor_nbytes, tensor_count);
    volvoxai_engine_metadata_unlock();
    return rc;
}

int volvoxai_engine_adapter_clone_update_with_metadata(const char* source_version, const char* new_adapter_id,
                                              const char* new_version_id, const char* const* tensor_names,
                                              const void* const* tensor_data, const int* tensor_dtypes,
                                              const size_t* tensor_nbytes, const int* update_modes,
                                              int tensor_count, const char* metadata_json) {
    if (tensor_count <= 0 || !tensor_names || !tensor_data || !tensor_dtypes || !tensor_nbytes || !update_modes) return -1;
    VxAdapterTensorUpdate* updates = (VxAdapterTensorUpdate*)calloc((size_t)tensor_count, sizeof(*updates));
    if (!updates) return -1;
    for (int i = 0; i < tensor_count; i++) {
        updates[i].tensor_name = tensor_names[i]; updates[i].data = tensor_data[i];
        updates[i].dtype = tensor_dtypes[i]; updates[i].nbytes = tensor_nbytes[i];
        updates[i].mode = (VxAdapterUpdateMode)update_modes[i];
    }
    int rc = vx_adapter_clone_update_with_metadata(source_version, new_adapter_id, new_version_id,
                                                    updates, tensor_count, metadata_json);
    free(updates);
    return rc;
}

int volvoxai_engine_adapter_clone_update(const char* source_version, const char* new_adapter_id,
                                const char* new_version_id, const char* const* tensor_names,
                                const void* const* tensor_data, const int* tensor_dtypes,
                                const size_t* tensor_nbytes, const int* update_modes, int tensor_count) {
    return volvoxai_engine_adapter_clone_update_with_metadata(source_version, new_adapter_id, new_version_id,
                                                     tensor_names, tensor_data, tensor_dtypes,
                                                     tensor_nbytes, update_modes, tensor_count, NULL);
}

static int volvoxai_engine_adapter_load_impl(const char* path, const char* version_override) {
    char loaded_version[128];
    if (vx_adapter_load_safetensors(path, version_override, loaded_version, sizeof(loaded_version)) != 0) return -1;
    int count = vx_adapter_target_count(loaded_version);
    int valid = count > 0;
    for (int i = 0; i < count && valid; i++) {
        VxAdapterTargetInfo info;
        int d_in = 0, d_out = 0, out_in = 0;
        T* weight = NULL;
        if (vx_adapter_target_info(loaded_version, i, &info) != 0 ||
            !(weight = t_find(info.weight_name)) ||
            volvoxai_engine_linear_weight_layout_impl(info.weight_name, &d_in, &d_out, &out_in) != 0 ||
            d_in != info.d_in || d_out != info.d_out || info.kind != VX_ADAPTER_LORA ||
            (weight->dtype != T_F32 && weight->dtype != T_F16)) valid = 0;
    }
    if (!valid) { vx_adapter_remove(loaded_version); return -1; }
    return 0;
}

int volvoxai_engine_adapter_load(const char* path, const char* version_override) {
    volvoxai_engine_metadata_lock();
    int rc = volvoxai_engine_adapter_load_impl(path, version_override);
    volvoxai_engine_metadata_unlock();
    return rc;
}

int volvoxai_engine_adapter_save(const char* version_id, const char* path) {
    return vx_adapter_save_safetensors(version_id, path);
}

char* volvoxai_engine_adapter_list_json(void) {
    return vx_adapter_list_json();
}

int volvoxai_engine_adapter_activate(const char* version_id) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    pthread_mutex_lock(&g_adapter_admin_mutex);
    int rc = g_merged_adapter_count ? -1 : vx_adapter_activate(version_id);
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    if (rc == 0) volvoxai_engine_model_generation_advance_locked();
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_adapter_remove(const char* version_id) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    pthread_mutex_lock(&g_adapter_admin_mutex);
    char active_version[128];
    int active = version_id && vx_adapter_get_active(active_version, sizeof(active_version)) == 0 &&
                 active_version[0] && !strcmp(version_id, active_version);
    int protected_version = g_merged_adapter_count && version_id &&
        (!strcmp(version_id, g_merged_adapter_version) || !strcmp(version_id, g_pre_merge_active_version));
    int rc = -1;
    if (!active && !protected_version &&
        volvoxai_engine_sync_model_weights_locked() == 0)
        rc = vx_adapter_remove(version_id);
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    if (rc == 0) {
        /* Adapter A/B buffers are stable CUDA weight identities while loaded.
         * Removal frees them, so use the full lifetime reset, not transient
         * training cleanup. */
        vx_runtime_backend_reset();
        volvoxai_engine_model_generation_advance_locked();
    }
    volvoxai_engine_model_unlock();
    return rc;
}

static void volvoxai_engine_adapter_free_merge_state(void) {
    for (int i = 0; i < g_merged_adapter_count; i++) {
        free(g_merged_adapter_weights[i].backup);
        free(g_merged_adapter_weights[i].merged);
    }
    free(g_merged_adapter_weights);
    g_merged_adapter_weights = NULL;
    g_merged_adapter_count = 0;
    g_merged_adapter_version[0] = 0;
    g_pre_merge_active_version[0] = 0;
}

static int volvoxai_engine_adapter_merge_impl(const char* version_id) {
    if (!version_id || !version_id[0] || g_merged_adapter_count) return -1;
    int count = vx_adapter_target_count(version_id);
    if (count <= 0) return -1;
    EngineMergedAdapterWeight* entries = (EngineMergedAdapterWeight*)calloc((size_t)count, sizeof(*entries));
    if (!entries) return -1;
    int ok = 1;
    for (int i = 0; i < count; i++) {
        VxAdapterTargetInfo info;
        if (vx_adapter_target_info(version_id, i, &info) != 0) { ok = 0; break; }
        T* tensor = t_find(info.weight_name);
        int d_in = 0, d_out = 0, out_in = 0;
        if (!tensor || (tensor->dtype != T_F32 && tensor->dtype != T_F16) || tensor->ndim != 2 ||
            volvoxai_engine_linear_weight_layout_impl(info.weight_name, &d_in, &d_out, &out_in) != 0 ||
            d_in != info.d_in || d_out != info.d_out) { ok = 0; break; }
        int file_index = volvoxai_engine_tensor_weight_file_index(info.weight_name);
        const SafetensorsTensor* backing = file_index >= 0 ? safetensors_find_tensor(&g_weight_files[file_index], info.weight_name) : NULL;
        VxDataType expected_dtype = tensor->dtype == T_F32 ? SAFETENSORS_DTYPE_F32 : SAFETENSORS_DTYPE_F16;
        if (backing && backing->dtype != expected_dtype) { ok = 0; break; }
        if (vk_sync_host_tensor(tensor) != 0) { ok = 0; break; }
        size_t bytes = (size_t)tensor->numel * tensor->elem_size;
        entries[i].tensor = tensor;
        entries[i].nbytes = bytes;
        entries[i].backup = (unsigned char*)malloc(bytes);
        entries[i].merged = (unsigned char*)malloc(bytes);
        if (!entries[i].backup || !entries[i].merged) { ok = 0; break; }
        memcpy(entries[i].backup, tensor->data, bytes);
        float* canonical = (float*)malloc((size_t)tensor->numel * sizeof(float));
        if (!canonical || vx_adapter_materialize_weight(version_id, info.weight_name, tensor->data, tensor->dtype,
                                                         info.d_in, info.d_out, out_in, canonical) != 0) {
            free(canonical);
            ok = 0; break;
        }
        for (int k = 0; k < info.d_in; k++) {
            for (int j = 0; j < info.d_out; j++) {
                size_t canonical_index = (size_t)k * info.d_out + j;
                size_t storage_index = out_in ? (size_t)j * info.d_in + k : canonical_index;
                volvoxai_engine_adapter_store_weight(entries[i].merged, tensor->dtype, storage_index,
                                            canonical[canonical_index]);
            }
        }
        free(canonical);
    }
    if (!ok) {
        for (int i = 0; i < count; i++) { free(entries[i].backup); free(entries[i].merged); }
        free(entries);
        return -1;
    }
    vx_adapter_get_active(g_pre_merge_active_version, sizeof(g_pre_merge_active_version));
    for (int i = 0; i < count; i++) {
        memcpy(entries[i].tensor->data, entries[i].merged, entries[i].nbytes);
        if (vx_runtime_backend_has_graph() && entries[i].nbytes > 0)
            vx_runtime_backend_mark_host(entries[i].tensor->data,
                                         entries[i].nbytes, 1);
    }
    g_merged_adapter_weights = entries;
    g_merged_adapter_count = count;
    strncpy(g_merged_adapter_version, version_id, sizeof(g_merged_adapter_version) - 1);
    g_merged_adapter_version[sizeof(g_merged_adapter_version) - 1] = 0;
    vx_adapter_activate("");
    vk_mark_owned_tensors_host_dirty(); g_weight_caches_dirty = 1;
    volvoxai_engine_model_generation_advance_locked();
    return 0;
}

int volvoxai_engine_adapter_merge(const char* version_id) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    pthread_mutex_lock(&g_adapter_admin_mutex);
    int rc = volvoxai_engine_adapter_merge_impl(version_id);
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    volvoxai_engine_model_unlock();
    return rc;
}

static int volvoxai_engine_adapter_unmerge_impl(void) {
    if (!g_merged_adapter_count) return -1;
    char restore[128];
    memcpy(restore, g_pre_merge_active_version, sizeof(restore));
    for (int i = 0; i < g_merged_adapter_count; i++) {
        T* tensor = g_merged_adapter_weights[i].tensor;
        if (vk_sync_host_tensor(tensor) != 0) return -1;
        if (memcmp(tensor->data, g_merged_adapter_weights[i].merged,
                   g_merged_adapter_weights[i].nbytes) != 0) return -1;
    }
    for (int i = 0; i < g_merged_adapter_count; i++) {
        T* tensor = g_merged_adapter_weights[i].tensor;
        memcpy(tensor->data, g_merged_adapter_weights[i].backup, g_merged_adapter_weights[i].nbytes);
        if (vx_runtime_backend_has_graph() &&
            g_merged_adapter_weights[i].nbytes > 0)
            vx_runtime_backend_mark_host(
                tensor->data, g_merged_adapter_weights[i].nbytes, 1);
    }
    volvoxai_engine_adapter_free_merge_state();
    if (vx_adapter_activate(restore) != 0) vx_adapter_activate("");
    vk_mark_owned_tensors_host_dirty(); g_weight_caches_dirty = 1;
    volvoxai_engine_model_generation_advance_locked();
    return 0;
}

int volvoxai_engine_adapter_unmerge(void) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (volvoxai_engine_training_accumulation_pending()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    pthread_mutex_lock(&g_adapter_admin_mutex);
    int rc = volvoxai_engine_adapter_unmerge_impl();
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_adapter_route_begin(const char* version_id) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    pthread_mutex_lock(&g_adapter_admin_mutex);
    int rc = g_merged_adapter_count && version_id && version_id[0] ? -1 : vx_adapter_request_begin(version_id);
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    if (rc == 0 && vx_engine_state_route_lease_begin() != 0) {
        vx_adapter_request_end();
        rc = -1;
    }
    if (rc != 0) volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_adapter_route_begin_many(const char* const* version_ids, const float* scales, int count) {
    if (g_engine_route_lease) return -1;
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return -1;
    }
    pthread_mutex_lock(&g_adapter_admin_mutex);
    if (g_merged_adapter_count) {
        for (int i = 0; i < count; i++) {
            if (version_ids && version_ids[i] && version_ids[i][0]) {
                pthread_mutex_unlock(&g_adapter_admin_mutex);
                volvoxai_engine_model_unlock();
                return -1;
            }
        }
    }
    int rc = vx_adapter_request_begin_many(version_ids, scales, count);
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    if (rc == 0 && vx_engine_state_route_lease_begin() != 0) {
        vx_adapter_request_end();
        rc = -1;
    }
    if (rc != 0) volvoxai_engine_model_unlock();
    return rc;
}

void volvoxai_engine_adapter_route_end(void) {
    vx_adapter_request_end();
    if (g_engine_route_lease) {
        vx_engine_state_route_lease_end();
        volvoxai_engine_model_unlock();
    }
}

int volvoxai_engine_forward_locked(void) {
    if (!g_loaded) return -1;
    if (g_nn == 0) {
        vx_engine_state_current()->last_failure_node_index = -1;
        memset(vx_engine_state_current()->runtime_route_backend, 0,
               sizeof(vx_engine_state_current()->runtime_route_backend));
        return 0;
    }
    vx_incremental_prepare_ordinary_locked();
    if (volvoxai_engine_refresh_weight_caches_if_dirty() != 0) return -1;
    if (vx_adapter_run_begin() != 0) return -1;
    int rc = -1;
    int backend_forward_started = 0;
    g_active_row = -1;
    g_prefix_rows = 0;
    g_prefix_row_capacity = 0;
    if (vx_runtime_backend_has_graph()) {
        vx_runtime_backend_begin_forward(
            vx_runtime_backend_cuda_replay_eligible(),
            volvoxai_engine_model_generation_locked());
        backend_forward_started = 1;
    }
    vk_mark_owned_tensors_host_dirty();
    if (backend_forward_started && vx_runtime_backend_prepare_forward() != 0)
        goto done;
    if (g_debug) prof_reset();
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    vx_engine_state_current()->last_failure_node_index = -1;
    memset(vx_engine_state_current()->runtime_route_backend, 0,
           sizeof(vx_engine_state_current()->runtime_route_backend));
    for (int i = 0; i < g_nn; i++) {
        if (run_node(&g_n[i], i, i == g_nn - 1) != 0) {
            vx_engine_state_current()->last_failure_node_index = i;
            goto done;
        }
    }
    rc = 0;
done:
    if (backend_forward_started) {
        double wait_t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
        if (vx_runtime_backend_end_forward(rc == 0) != 0) rc = -1;
        if (g_debug) prof_add_entry("GPUWait", volvoxai_engine_now_ms() - wait_t0);
    }
    if (rc == 0 && g_debug) {
        fprintf(stderr, "[debug] volvoxai_engine_forward nodes=%d %.3f ms\n",
                g_nn, volvoxai_engine_now_ms() - t0);
        prof_report();
    }
    vx_adapter_run_end();
    return rc;
}

int volvoxai_engine_forward(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    int rc = volvoxai_engine_forward_locked();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_forward_incremental_locked(void) {
    return vx_incremental_forward_locked(-1);
}

int volvoxai_engine_forward_incremental_row_locked(int row) {
    return volvoxai_engine_execution_row_valid_locked(row)
        ? vx_incremental_forward_locked(row) : -1;
}

int volvoxai_engine_forward_incremental(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    int rc = volvoxai_engine_forward_incremental_locked();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_forward_incremental_row(int row) {
    if (row < 0) return -1;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    int rc = volvoxai_engine_forward_incremental_row_locked(row);
    if (took_model_lock) volvoxai_engine_model_unlock();
    return rc;
}

int volvoxai_engine_incremental_row_supported(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    int supported = vx_incremental_row_supported_locked();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return supported;
}

void volvoxai_engine_incremental_reset(void) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    if (!vx_dynamic_autograd_active_locked())
        vx_incremental_prepare_ordinary_locked();
    if (took_model_lock) volvoxai_engine_model_unlock();
}

void volvoxai_engine_profile_reset(void) {
    prof_reset();
}

void volvoxai_engine_profile_report(void) {
    prof_report();
}

int volvoxai_engine_forward_prefix(int row_count) {
    long row_capacity;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    row_capacity = volvoxai_engine_execution_row_capacity_locked();
    if (!g_loaded || g_nn == 0 || row_count <= 0 || row_capacity <= 0 ||
        (long)row_count > row_capacity ||
        !volvoxai_engine_execution_row_valid_locked(row_count - 1)) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        return -1;
    }
    vx_incremental_prepare_ordinary_locked();
    if (volvoxai_engine_refresh_weight_caches_if_dirty() != 0) { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    if (vx_adapter_run_begin() != 0) { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    int result = -1;
    int backend_forward_started = 0;
    g_active_row = -1;
    g_prefix_rows = row_count;
    g_prefix_row_capacity = row_capacity;
    g_execution_row = row_count - 1;
    if (vx_runtime_backend_has_graph()) {
        vx_runtime_backend_begin_forward(
            0, volvoxai_engine_model_generation_locked());
        backend_forward_started = 1;
    }
    /* Prefix execution may be the first device pass after model load.  Create
     * full-tensor graph slots before row-sliced CUDA kernels request interior
     * views; later row calls deliberately retain those resident allocations. */
    vk_mark_owned_tensors_host_dirty();
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    for (int i = 0; i < g_nn; i++) {
        if (run_node(&g_n[i], i, i == g_nn - 1) != 0) goto done;
    }
    result = 0;
done:
    if (backend_forward_started &&
        vx_runtime_backend_end_forward(result == 0) != 0)
        result = -1;
    g_prefix_rows = 0;
    g_prefix_row_capacity = 0;
    if (result == 0 && g_debug) {
        fprintf(stderr, "[debug] volvoxai_engine_forward_prefix rows=%d nodes=%d %.3f ms\n",
                row_count, g_nn, volvoxai_engine_now_ms() - t0);
    }
    vx_adapter_run_end();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

int volvoxai_engine_forward_row(int row) {
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    if (!g_loaded || g_nn == 0 || row < 0 ||
        !volvoxai_engine_execution_row_valid_locked(row)) {
        if (took_model_lock) volvoxai_engine_model_unlock();
        return -1;
    }
    vx_incremental_prepare_ordinary_locked();
    if (volvoxai_engine_refresh_weight_caches_if_dirty() != 0) { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    if (vx_adapter_run_begin() != 0) { if (took_model_lock) volvoxai_engine_model_unlock(); return -1; }
    int result = -1;
    int backend_forward_started = 0;
    g_active_row = row;
    g_execution_row = row;
    g_prefix_rows = 0;
    g_prefix_row_capacity = 0;
    if (vx_runtime_backend_has_graph()) {
        vx_runtime_backend_begin_forward(
            0, volvoxai_engine_model_generation_locked());
        backend_forward_started = 1;
    }
    double t0 = g_debug ? volvoxai_engine_now_ms() : 0.0;
    for (int i = 0; i < g_nn; i++) {
        if (run_node(&g_n[i], i, i == g_nn - 1) != 0) goto done;
    }
    result = 0;
done:
    if (backend_forward_started &&
        vx_runtime_backend_end_forward(result == 0) != 0)
        result = -1;
    g_active_row = -1;
    if (result == 0 && g_debug) {
        fprintf(stderr, "[debug] volvoxai_engine_forward_row row=%d nodes=%d %.3f ms\n",
                row, g_nn, volvoxai_engine_now_ms() - t0);
    }
    vx_adapter_run_end();
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

const float* volvoxai_engine_tensor_row_f32(const char* name, int row, int* count) {
    if (count) *count = 0;
    if (!name || !name[0] || row < -1) return NULL;
    int took_model_lock = !g_engine_route_lease;
    if (took_model_lock) volvoxai_engine_model_lock();
    T* tensor = volvoxai_engine_tensor_is_quantization_parameter_locked(name)
        ? NULL : t_find(name);
    const float* result = NULL;
    int row_synced = 0;
    if (!tensor || !tensor->data || tensor->numel <= 0) goto done;
#if VOLVOXAI_ENABLE_CUDA
    if (g_use_cuda && row >= 0 && tensor->dtype == T_F32 &&
        tensor->ndim > 0 && tensor->shape[tensor->ndim - 1] > 0) {
        int row_width = tensor->shape[tensor->ndim - 1];
        long row_total = tensor->numel / row_width;
        if (tensor->numel != row_total * row_width || row >= row_total ||
            (size_t)row_width > SIZE_MAX / sizeof(float) ||
            vx_runtime_backend_sync_host(
                tensor->data + (long)row * row_width,
                (size_t)row_width * sizeof(float),
                volvoxai_engine_tensor_is_model_weight_locked(tensor->name)) != 0)
            goto done;
        row_synced = 1;
    }
#endif
    if (!row_synced && vk_sync_host_tensor(tensor) != 0) goto done;
    materialize_tensor_f32(tensor);
    if (tensor->dtype != T_F32 || !tensor->data || tensor->numel > INT_MAX) goto done;
    if (row < 0) {
        if (count) *count = (int)tensor->numel;
        result = tensor->data;
        goto done;
    }
    if (tensor->ndim <= 0 || tensor->shape[tensor->ndim - 1] <= 0) goto done;
    int width = tensor->shape[tensor->ndim - 1];
    long rows = tensor->numel / width;
    if (tensor->numel != rows * width || row >= rows) goto done;
    if (count) *count = width;
    result = tensor->data + (long)row * width;
done:
    if (took_model_lock) volvoxai_engine_model_unlock();
    return result;
}

static void volvoxai_engine_shutdown_impl(void) {
    vx_bank_residency_clear();
    volvoxai_engine_model_generation_advance_locked();
    vx_decode_session_invalidate_model_locked();
    volvoxai_engine_reset_training_accumulation();
    volvoxai_engine_free_optimizer_states();
    pthread_mutex_lock(&g_adapter_admin_mutex);
    volvoxai_engine_adapter_free_merge_state();
    pthread_mutex_unlock(&g_adapter_admin_mutex);
    vx_adapter_reset();
#if VOLVOXAI_ENABLE_VULKAN
    vk_free_weight_cache();
#endif
    vx_runtime_backend_reset();
#if VOLVOXAI_ENABLE_OPENGL
    opengl_free_weight_cache();
#endif
#if VOLVOXAI_ENABLE_NNAPI
    /* NNAPI models retain pointers to constant operands. Destroy those models
     * before the graph-owned safetensors/tensor storage below is released. */
    nnapi_free_weight_cache();
#endif
    vx_runtime_backend_teardown();
    volvoxai_shader_store_shutdown();
    vx_conv_f32_opt_free_all();
    vx_gemm_f32_cache_free_all(&vx_engine_state_current()->gemm_f32_cache);
    volvoxai_engine_free_arena();
    volvoxai_engine_free_cpu_typed_workspace_locked();
    volvoxai_engine_clear_qlinear_metadata();
    volvoxai_engine_clear_qconv_metadata();
    volvoxai_engine_clear_qembedding_metadata();
    for (int i = 0; i < g_nn; i++) {
        if (g_n[i].owns_params && g_n[i].params) {
            cJSON_Delete(g_n[i].params);
            g_n[i].params = NULL;
            g_n[i].owns_params = 0;
        }
        free(g_kcache[i]);
        free(g_vcache[i]);
        free(g_qwcache[i]);
        free(g_f16wcache[i]);
        free(g_f16bcache[i]);
        free(g_conv_wcache[i]);
        free(g_q8wcache[i]);
        g_kcache[i] = NULL;
        g_vcache[i] = NULL;
        g_qwcache[i] = NULL;
        g_f16wcache[i] = NULL;
        g_f16bcache[i] = NULL;
        g_conv_wcache[i] = NULL;
        g_q8wcache[i] = NULL;
        g_q8wcache_bytes[i] = 0;
    }
    for (int i = 0; i < g_nt; i++) {
        if (g_t[i].owns && g_t[i].data) {
            free(g_t[i].data);
            g_t[i].data = NULL;
        }
    }
    if (g_graph_root) {
        cJSON_Delete(g_graph_root);
        g_graph_root = NULL;
    }
    for (int i = 0; i < g_weight_file_count; i++) {
        safetensors_free(&g_weight_files[i]);
        g_weight_paths[i][0] = 0;
    }
    g_weight_file_count = 0;
    g_blob = NULL;
    g_nt = 0;
    volvoxai_engine_tensor_name_index_rebuild();
    g_nn = 0;
    g_loaded = 0;
    g_weight_caches_dirty = 0;
    g_first_input[0] = 0;
    g_active_row = -1;
    g_prefix_rows = 0;
    g_prefix_row_capacity = 0;
    g_execution_row = -1;
    vx_incremental_shutdown_locked();
}

void volvoxai_engine_shutdown(void) {
    if (g_engine_route_lease) volvoxai_engine_adapter_route_end();
    volvoxai_engine_model_lock();
    if (vx_dynamic_autograd_active_locked()) {
        volvoxai_engine_model_unlock();
        return;
    }
    volvoxai_engine_metadata_lock();
    volvoxai_engine_shutdown_impl();
    vx_backend_manager_deactivate();
    vx_set_num_threads(0);
    g_cpu_threads = 0;
    g_debug = 0;
    volvoxai_engine_metadata_unlock();
    volvoxai_engine_model_unlock();
}

int volvoxai_engine_run(const char* graph_path, const char* weights_path,
               const char* input_path, const char* output_path) {
    int result = -1;
    if (volvoxai_engine_init(graph_path, weights_path) != 0) return -1;

    if (input_path) {
        long n;
        const char* input_name = volvoxai_engine_graph_input_name(0);
        float* dst = volvoxai_engine_input_ptr(input_name, &n);
        long isz;
        char* idata = read_file(input_path, &isz);
        if (!dst || !idata || isz < 0) { free(idata); goto done; }
        long c = isz / 4;
        if (c > n) c = n;
        memcpy(dst, idata, (size_t)c * 4);
        free(idata);
    }
    if (volvoxai_engine_forward() != 0) goto done;

    if (output_path) {
        const char* output_name = volvoxai_engine_graph_output_name(0);
        int count = 0;
        const float* output = volvoxai_engine_tensor_row_f32(
            output_name, volvoxai_engine_execution_row(), &count);
        if (!output || count <= 0) goto done;
        FILE* f = fopen(output_path, "wb");
        if (!f) goto done;
        size_t written = fwrite(output, sizeof(float), (size_t)count, f);
        int close_status = fclose(f);
        if (written != (size_t)count || close_status != 0) goto done;
    }
    result = 0;
done:
    volvoxai_engine_shutdown();
    return result;
}
