#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "volvoxai.h"
#include "volvoxai_backend.h"

#include "backend_sdk.h"
#include "engine_internal.h"
#include "engine_core.h"
#include "generated/kernel_registry.h"
#include "inference_kernels.h"
#include "json_validation.h"
#include "public_api_internal.h"
#include "runtime_state.h"
#include "safetensors.h"
#include "shape_contract.h"
#include "thread_pool.h"

#if defined(VOLVOXAI_ENABLE_VULKAN) && VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#endif
#if defined(VOLVOXAI_ENABLE_OPENGL) && VOLVOXAI_ENABLE_OPENGL
#include "opengl_engine.h"
#endif
#if defined(VOLVOXAI_ENABLE_METAL) && VOLVOXAI_ENABLE_METAL
#include "metal_engine.h"
#endif
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct VxOwnedOutput {
    char* name;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
    unsigned char* data;
} VxOwnedOutput;

typedef struct VxDeclaredTensor {
    char* name;
    char* symbols[VX_MAX_TENSOR_RANK];
    VxDataType dtype;
    uint32_t rank;
    VxDimensionKind kinds[VX_MAX_TENSOR_RANK];
    int64_t minimums[VX_MAX_TENSOR_RANK];
    int64_t maximums[VX_MAX_TENSOR_RANK];
    int64_t multiples[VX_MAX_TENSOR_RANK];
    size_t maximum_byte_size;
    /* Concrete mirrors are populated only for singleton/static descriptors;
     * they keep private-engine result validation isolated from dynamic arena
     * work deferred to the next tranche. */
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
} VxDeclaredTensor;

typedef VxDeclaredTensor VxDeclaredOutput;

struct VxWeightRevisionRecord {
    atomic_uint references;
    uint64_t identity;
    uint64_t revision;
    uint64_t content_hash;
    char** paths;
    size_t path_count;
    uint64_t allocated_bytes;
};

typedef struct VxAdapterRevisionRecord {
    atomic_uint references;
    uint64_t identity;
    uint64_t revision;
    uint64_t content_hash;
    char* name;
    char* package_path;
    char* version_name;
    uint64_t allocated_bytes;
    struct VxAdapterRevisionRecord* next;
} VxAdapterRevisionRecord;

struct VxRuntime {
    atomic_uint references;
    atomic_uint_fast64_t next_object_identity;
    atomic_uint_fast64_t next_execution_identity;
    pthread_mutex_t mutex;
    int closed;
    uint64_t identity;
    VxRuntimeOptions options;
    VxProviderRegistry* providers;
};

struct VxModel {
    atomic_uint references;
    VxRuntime* runtime;
    char* graph_path;
    cJSON* logical_graph;
    VxDeclaredTensor* inputs;
    size_t input_count;
    VxDeclaredTensor* outputs;
    size_t output_count;
    char graph_fingerprint[64];
    char shape_domain_proof_identity[96];
    uint64_t maximum_tensor_bytes;
    pthread_mutex_t revision_mutex;
    _Atomic(VxWeightRevisionRecord*) current_weights;
    _Atomic(VxAdapterRevisionRecord*) current_adapter;
    VxAdapterRevisionRecord* adapters;
    uint64_t identity;
    uint64_t graph_identity;
    uint64_t graph_revision;
    uint64_t weight_identity;
    uint64_t allocated_bytes;
    /* Load-time weight-bank residency, replayed into the engine on each
     * compile because the engine owns one global weight table. */
    VxBankResidency* bank_residency;
    uint32_t** bank_residency_slots;
    size_t bank_residency_count;
};

struct VxCompiledModel {
    atomic_uint references;
    VxModel* model;
    VxWeightRevisionRecord* weights;
    VxAdapterRevisionRecord* adapter;
    uint64_t identity;
    uint64_t allocated_bytes;
    VxBackendPolicyMode mode;
    VxOperatorFallback operator_fallback;
    size_t maximum_cpu_typed_workspace_bytes;
    uint64_t maximum_resident_bytes;
    char backend[VX_BACKEND_NAME_CAPACITY];
    VolvoxAIEngineBackend builtin_backend;
    const VxBackendProvider* provider;
    void* provider_runtime;
    void* provider_compiled;
    VxReport report;
};

struct VxExecutionContext {
    atomic_uint references;
    VxCompiledModel* compiled;
    uint64_t identity;
    uint64_t allocated_bytes;
    pthread_mutex_t queue_mutex;
    pthread_cond_t queue_condition;
    uint64_t next_ticket;
    uint64_t serving_ticket;
    int closing;
    int closed;
    VxEngineState* engine_state;
    int engine_state_initialized;
    int engine_loaded;
    VolvoxAIDecodeSession* decode_session;
    VxDecodeRowMode decode_row_mode;
    int require_incremental;
    VxAdapterRevisionRecord* adapter;
    char adapter_route_version[96];
    void* provider_context;
    int decode_seeded;
    int64_t* decode_input_shapes;
    VxDeclaredTensor* logical_tensors;
    size_t logical_tensor_count;
    size_t* logical_tensor_name_slots;
    size_t logical_tensor_name_capacity;
    char* committed_shape_signature;
    VolvoxAIEngineDynamicShapeStats dynamic_shape_stats;
    double dynamic_shape_bind_time_ms;
};

struct VxResult {
    atomic_uint references;
    VxExecutionContext* context;
    uint64_t execution_id;
    VxOwnedOutput* outputs;
    size_t output_count;
    size_t output_capacity;
    VxStatus output_sink_status;
    uint64_t snapshot_bytes;
    uint64_t evidence_allocated_bytes;
    uint64_t adapter_id;
    uint64_t adapter_revision;
    int32_t operator_fallback_used;
    int32_t route_attested;
    char route_evidence[VX_REPORT_ROUTE_CAPACITY];
    char fallback_evidence[VX_REPORT_FALLBACK_CAPACITY];
    char offending_node[VX_REPORT_NODE_CAPACITY];
    char decode_state[VX_REPORT_DECODE_CAPACITY];
    int64_t* input_shapes;
    size_t input_shape_count;
};

typedef struct VxContextOperation {
    uint64_t ticket;
    int accepted;
} VxContextOperation;

static int vx_builtin_dynamic_shape_backend(const char* backend) {
    return backend && (!strcmp(backend, "cpu") ||
        !strcmp(backend, "vulkan") || !strcmp(backend, "opengl") ||
        !strcmp(backend, "metal") || !strcmp(backend, "cuda"));
}

static int vx_native_gpu_backend(VolvoxAIEngineBackend backend) {
    return backend == VOLVOXAI_BACKEND_VULKAN ||
        backend == VOLVOXAI_BACKEND_OPENGL ||
        backend == VOLVOXAI_BACKEND_METAL ||
        backend == VOLVOXAI_BACKEND_CUDA;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
typedef void (*VxPublicApiCpuExecuteHook)(void* user_data);
static VxPublicApiCpuExecuteHook g_cpu_execute_hook;
static void* g_cpu_execute_hook_user_data;

void vx_public_api_test_set_cpu_execute_hook(VxPublicApiCpuExecuteHook hook,
                                              void* user_data) {
    g_cpu_execute_hook = hook;
    g_cpu_execute_hook_user_data = user_data;
}

int vx_public_api_test_dynamic_shape_state(
        const VxExecutionContext* context,
        uint64_t* resource_generation,
        size_t* arena_capacity,
        size_t* arena_high_water,
        uint64_t* arena_grow_count) {
    if (!context || !context->engine_state ||
        !vx_builtin_dynamic_shape_backend(context->compiled->backend))
        return -1;
    if (resource_generation)
        *resource_generation =
            context->engine_state->dynamic_resource_generation;
    if (arena_capacity)
        *arena_capacity =
            context->engine_state->dynamic_arena_capacity_bytes;
    if (arena_high_water)
        *arena_high_water =
            context->engine_state->dynamic_arena_high_water_bytes;
    if (arena_grow_count)
        *arena_grow_count =
            context->engine_state->dynamic_arena_grow_count;
    return 0;
}

int vx_public_api_test_fill_engine_i32_tensor(
        VxExecutionContext* context, const char* name, int32_t value) {
    T* tensor = NULL;
    if (!context || !context->engine_state || !name) return -1;
    for (int index = 0; index < context->engine_state->tensor_count; index++)
        if (!strcmp(context->engine_state->tensors[index].name, name)) {
            tensor = &context->engine_state->tensors[index];
            break;
        }
    if (!tensor || tensor->dtype != T_I32 || !tensor->data ||
        tensor->numel <= 0) return -1;
    for (long index = 0; index < tensor->numel; index++)
        memcpy((unsigned char*)tensor->data +
                   (size_t)index * sizeof(value),
               &value, sizeof(value));
    return 0;
}

int vx_public_api_test_conv_cache_state(
        const VxExecutionContext* context,
        int node_index,
        int* transformed_weight,
        int* cpu_pack,
        int* cpu_indirection) {
    const VxEngineState* state;
    if (!context || !(state = context->engine_state) ||
        node_index < 0 || node_index >= state->node_count)
        return -1;
    if (transformed_weight)
        *transformed_weight = state->conv_wcache[node_index] != NULL;
    if (cpu_pack)
        *cpu_pack = state->conv_pwf32_pack[node_index] != NULL ||
            state->conv_pwf32_pack_plain[node_index] != NULL ||
            state->conv_f32_igemm_pack[node_index] != NULL;
    if (cpu_indirection)
        *cpu_indirection =
            state->conv_f32_igemm_indir[node_index] != NULL ||
            state->conv_f32_igemm_zero[node_index] != NULL;
    return 0;
}

#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA && \
    defined(VOLVOXAI_CUDA_TESTING)
int vx_public_api_test_cuda_dynamic_reservation(
        const VxExecutionContext* context,
        uint64_t* allocation_count,
        size_t* span_count,
        int* preload_complete,
        int* enforced,
        int* replay_plan) {
    VxEngineStateScope scope;
    int result;
    if (!context || !context->engine_state ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CUDA)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    if (allocation_count)
        *allocation_count = cuda_test_graph_allocation_count();
    result = cuda_test_graph_domain_reservation(
        span_count, preload_complete, enforced, replay_plan);
    vx_engine_state_scope_leave(scope);
    return result;
}
#endif

int vx_public_api_test_cpu_typed_workspace_state(
        const VxExecutionContext* context,
        uintptr_t* address,
        size_t* bound_bytes,
        size_t* capacity_bytes) {
    if (!context || !context->engine_state || !context->compiled ||
        context->compiled->provider ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CPU)
        return -1;
    if (address)
        *address = (uintptr_t)context->engine_state->cpu_typed_workspace;
    if (bound_bytes)
        *bound_bytes =
            context->engine_state->cpu_typed_workspace_bound_bytes;
    if (capacity_bytes)
        *capacity_bytes =
            context->engine_state->cpu_typed_workspace_capacity_bytes;
    return 0;
}

int vx_public_api_test_cpu_typed_workspace_reconfigure(
        VxExecutionContext* context,
        size_t bounded_bytes) {
    VxEngineStateScope scope;
    int result;
    if (!context || !context->engine_state || !context->compiled ||
        context->compiled->provider ||
        context->compiled->builtin_backend != VOLVOXAI_BACKEND_CPU)
        return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = volvoxai_engine_configure_cpu_typed_workspace(bounded_bytes);
    vx_engine_state_scope_leave(scope);
    return result;
}

int vx_public_api_test_compiled_resource_bounds(
        const VxCompiledModel* compiled,
        size_t* maximum_typed_scratch_bytes,
        uint64_t* maximum_resident_bytes) {
    if (!compiled || compiled->provider ||
        compiled->builtin_backend != VOLVOXAI_BACKEND_CPU)
        return -1;
    if (maximum_typed_scratch_bytes)
        *maximum_typed_scratch_bytes =
            compiled->maximum_cpu_typed_workspace_bytes;
    if (maximum_resident_bytes)
        *maximum_resident_bytes = compiled->maximum_resident_bytes;
    return 0;
}

int vx_public_api_test_copy_tensor(const VxExecutionContext* context,
                                   const char* name,
                                   void* bytes,
                                   size_t byte_size) {
    VxEngineStateScope scope;
    int result;
    if (!context || !context->engine_state || !name || !bytes ||
        strcmp(context->compiled->backend, "cpu")) return -1;
    scope = vx_engine_state_scope_enter(context->engine_state);
    result = volvoxai_engine_copy_tensor_raw(name, bytes, byte_size);
    vx_engine_state_scope_leave(scope);
    return result;
}
#endif

static uint64_t vx_runtime_identity(const VxRuntime* runtime) {
    struct timespec now = {0};
    uint64_t identity = (uint64_t)(uintptr_t)runtime;
    if (timespec_get(&now, TIME_UTC) == TIME_UTC) {
        identity ^= (uint64_t)now.tv_sec;
        identity ^= (uint64_t)now.tv_nsec << 32u;
    }
    identity ^= identity >> 30u;
    identity *= UINT64_C(0xbf58476d1ce4e5b9);
    identity ^= identity >> 27u;
    identity *= UINT64_C(0x94d049bb133111eb);
    identity ^= identity >> 31u;
    return identity ? identity : UINT64_C(1);
}

static uint64_t vx_next_object_identity(VxRuntime* runtime) {
    uint64_t identity = atomic_fetch_add_explicit(&runtime->next_object_identity, 1,
                                                   memory_order_relaxed);
    if (!identity)
        identity = atomic_fetch_add_explicit(&runtime->next_object_identity, 1,
                                              memory_order_relaxed);
    return identity;
}

static uint64_t vx_next_execution_identity(VxRuntime* runtime) {
    uint64_t identity = atomic_fetch_add_explicit(
        &runtime->next_execution_identity, 1, memory_order_relaxed);
    if (!identity)
        identity = atomic_fetch_add_explicit(
            &runtime->next_execution_identity, 1, memory_order_relaxed);
    return identity;
}

static double vx_report_now_ms(void) {
    struct timespec now;
    if (timespec_get(&now, TIME_UTC) != TIME_UTC) return 0.0;
    return (double)now.tv_sec * 1000.0 + (double)now.tv_nsec / 1000000.0;
}

static uint64_t vx_revision_update(uint64_t revision,
                                   const void* bytes,
                                   size_t byte_count) {
    const unsigned char* cursor = (const unsigned char*)bytes;
    for (size_t index = 0; index < byte_count; index++) {
        revision ^= cursor[index];
        revision *= UINT64_C(1099511628211);
    }
    return revision;
}

static void vx_snapshot_path_release(char* path) {
    if (!path) return;
    (void)remove(path);
    free(path);
}

static int vx_is_canonical_graph_path(const char* path) {
    const char* basename;
    const char* slash;
    const char* backslash;
    size_t length;
    static const char named_suffix[] = ".graph.json";
    if (!path || !path[0]) return 0;
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    basename = slash && (!backslash || slash > backslash) ? slash + 1
        : backslash ? backslash + 1 : path;
    if (strcmp(basename, "graph.json") == 0) return 1;
    length = strlen(basename);
    return length > sizeof(named_suffix) - 1u &&
        strcmp(basename + length - (sizeof(named_suffix) - 1u), named_suffix) == 0;
}

/* The private engine still consumes filesystem paths. Copy each accepted
 * source into a process-owned, mode-0600 snapshot so later compilation and
 * context creation never reread caller-owned mutable files. */
static VxStatus vx_file_snapshot(const char* source_path,
                                 char** out_snapshot_path,
                                 uint64_t* out_revision) {
    unsigned char bytes[16384];
    uint64_t revision = UINT64_C(1469598103934665603);
    FILE* source = NULL;
    FILE* snapshot = NULL;
    char path[PATH_MAX];
    char* owned_path = NULL;
    size_t count;
    int failed = 0;
    if (!source_path || !source_path[0] || !out_snapshot_path)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_snapshot_path = NULL;
    source = fopen(source_path, "rb");
    if (!source) return VX_STATUS_IO_ERROR;
#ifdef _WIN32
    {
        char directory[MAX_PATH];
        DWORD length = GetTempPathA((DWORD)sizeof(directory), directory);
        if (!length || length >= sizeof(directory) ||
            !GetTempFileNameA(directory, "vxr", 0, path)) {
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
        snapshot = fopen(path, "wb");
        if (!snapshot) {
            (void)remove(path);
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
    }
#else
    {
        const char* directory = getenv("TMPDIR");
        int descriptor;
        int written;
        if (!directory || !directory[0]) directory = "/tmp";
        written = snprintf(path, sizeof(path), "%s%svolvoxai-snapshot-XXXXXX",
                           directory,
                           directory[strlen(directory) - 1u] == '/' ? "" : "/");
        if (written < 0 || (size_t)written >= sizeof(path)) {
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
        descriptor = mkstemp(path);
        if (descriptor < 0) {
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
        snapshot = fdopen(descriptor, "wb");
        if (!snapshot) {
            close(descriptor);
            (void)remove(path);
            fclose(source);
            return VX_STATUS_IO_ERROR;
        }
    }
#endif
    owned_path = (char*)malloc(strlen(path) + 1u);
    if (!owned_path) {
        fclose(snapshot);
        (void)remove(path);
        fclose(source);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    memcpy(owned_path, path, strlen(path) + 1u);
    while ((count = fread(bytes, 1, sizeof(bytes), source)) != 0) {
        revision = vx_revision_update(revision, bytes, count);
        if (fwrite(bytes, 1, count, snapshot) != count) {
            failed = 1;
            break;
        }
    }
    if (ferror(source)) failed = 1;
    if (fflush(snapshot) != 0) failed = 1;
    if (fclose(source) != 0) failed = 1;
    if (fclose(snapshot) != 0) failed = 1;
    if (failed) {
        vx_snapshot_path_release(owned_path);
        return VX_STATUS_IO_ERROR;
    }
    if (out_revision)
        *out_revision = revision ? revision : UINT64_C(1);
    *out_snapshot_path = owned_path;
    return VX_STATUS_OK;
}

static int vx_graph_outputs_envelope_valid(const cJSON* outputs) {
    int count;
    if (!cJSON_IsArray(outputs) || !outputs->child) return 0;
    count = cJSON_GetArraySize(outputs);
    if (count <= 0 || count > MAXT) return 0;
    for (const cJSON* item = outputs->child; item; item = item->next) {
        if (!cJSON_IsString(item) || !item->valuestring ||
            !item->valuestring[0] || strlen(item->valuestring) >= 128u) return 0;
        for (const cJSON* prior = outputs->child; prior && prior != item;
             prior = prior->next) {
            if (cJSON_IsString(prior) && prior->valuestring &&
                !strcmp(prior->valuestring, item->valuestring)) return 0;
        }
    }
    return 1;
}

static int vx_graph_quantization_envelope_valid(const cJSON* root) {
    const cJSON* quantization;
    const cJSON* format;
    const cJSON* tensors;
    if (cJSON_GetObjectItemCaseSensitive(root, "weights_quantization") ||
        cJSON_GetObjectItemCaseSensitive(
            root, "weights_quantization_storage")) return 0;
    quantization = cJSON_GetObjectItemCaseSensitive(root, "quantization");
    if (!quantization) return 1;
    if (!cJSON_IsObject(quantization) ||
        cJSON_GetArraySize(quantization) != 2) return 0;
    format = cJSON_GetObjectItemCaseSensitive(quantization, "format");
    tensors = cJSON_GetObjectItemCaseSensitive(quantization, "tensors");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-affine-safetensors/v1") ||
        !cJSON_IsObject(tensors) || !tensors->child) return 0;
    for (const cJSON* item = quantization->child; item; item = item->next) {
        if (!item->string || (strcmp(item->string, "format") &&
                              strcmp(item->string, "tensors"))) return 0;
    }
    for (const cJSON* descriptor = tensors->child; descriptor;
         descriptor = descriptor->next) {
        const cJSON* scheme;
        const cJSON* axis;
        const cJSON* scale;
        const cJSON* zero;
        int per_axis;
        int expected_fields;
        if (!descriptor->string || !descriptor->string[0] ||
            strlen(descriptor->string) >= 128u ||
            !cJSON_IsObject(descriptor)) return 0;
        scheme = cJSON_GetObjectItemCaseSensitive(descriptor, "scheme");
        axis = cJSON_GetObjectItemCaseSensitive(descriptor, "axis");
        scale = cJSON_GetObjectItemCaseSensitive(descriptor, "scale_tensor");
        zero = cJSON_GetObjectItemCaseSensitive(
            descriptor, "zero_point_tensor");
        if (!cJSON_IsString(scheme) || !scheme->valuestring) return 0;
        if (!strcmp(scheme->valuestring, "per_tensor")) per_axis = 0;
        else if (!strcmp(scheme->valuestring, "per_axis")) per_axis = 1;
        else return 0;
        expected_fields = per_axis ? 4 : 3;
        if (cJSON_GetArraySize(descriptor) != expected_fields ||
            (per_axis && (!cJSON_IsNumber(axis) ||
                          axis->valuedouble != (double)axis->valueint)) ||
            (!per_axis && axis) ||
            !cJSON_IsString(scale) || !scale->valuestring ||
            !scale->valuestring[0] || strlen(scale->valuestring) >= 128u ||
            !cJSON_IsString(zero) || !zero->valuestring ||
            !zero->valuestring[0] || strlen(zero->valuestring) >= 128u ||
            !strcmp(scale->valuestring, zero->valuestring)) return 0;
        for (const cJSON* item = descriptor->child; item; item = item->next) {
            if (!item->string ||
                (strcmp(item->string, "scheme") &&
                 strcmp(item->string, "axis") &&
                 strcmp(item->string, "scale_tensor") &&
                 strcmp(item->string, "zero_point_tensor"))) return 0;
        }
    }
    return 1;
}

static int vx_graph_value_contains_affine_alias(const cJSON* value) {
    static const char* const forbidden[] = {
        "quantization",
        "input_scale", "input_zero_point",
        "output_scale", "output_zero_point",
        "weight_scale", "weight_zero_point",
        "zero_point",
        "scales", "zero_points",
        "scale_tensor", "zero_point_tensor",
    };
    for (const cJSON* item = value ? value->child : NULL;
         item; item = item->next) {
        for (size_t index = 0;
             item->string && index < sizeof(forbidden) / sizeof(forbidden[0]);
             index++) {
            if (!strcmp(item->string, forbidden[index])) return 1;
        }
        if (vx_graph_value_contains_affine_alias(item)) return 1;
    }
    return 0;
}

static int vx_graph_params_affine_payload_free(const cJSON* params) {
    if (!params) return 1;
    return cJSON_IsObject(params) &&
           !vx_graph_value_contains_affine_alias(params);
}

static int vx_graph_envelope_schema_valid(const cJSON* root) {
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    if ((inputs && !cJSON_IsObject(inputs)) ||
        (nodes && !cJSON_IsArray(nodes)) ||
        !vx_graph_quantization_envelope_valid(root)) return 0;
    for (const cJSON* input = inputs ? inputs->child : NULL;
         input; input = input->next) {
        const cJSON* dtype = cJSON_GetObjectItemCaseSensitive(input, "dtype");
        if (!cJSON_IsObject(input) || !cJSON_IsString(dtype) ||
            !dtype->valuestring ||
            cJSON_GetObjectItemCaseSensitive(input, "quantization")) return 0;
        if (strcmp(dtype->valuestring, "float32") &&
            strcmp(dtype->valuestring, "int32") &&
            strcmp(dtype->valuestring, "int8") &&
            strcmp(dtype->valuestring, "uint8")) return 0;
    }
    for (const cJSON* node = nodes ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* output_dtypes =
            cJSON_GetObjectItemCaseSensitive(node, "outputs_dtype");
        const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
        if (!cJSON_IsObject(node) || !cJSON_IsString(op) ||
            !op->valuestring || !op->valuestring[0] ||
            !vx_graph_runtime_operator_name_is_current(op->valuestring) ||
            cJSON_GetObjectItemCaseSensitive(node, "op") ||
            cJSON_GetObjectItemCaseSensitive(node, "outputs_quantization") ||
            !vx_graph_params_affine_payload_free(params) ||
            (output_dtypes && !cJSON_IsObject(output_dtypes))) return 0;
        for (const cJSON* dtype = output_dtypes ? output_dtypes->child : NULL;
             dtype; dtype = dtype->next) {
            if (!cJSON_IsString(dtype) || !dtype->valuestring ||
                (strcmp(dtype->valuestring, "float32") &&
                 strcmp(dtype->valuestring, "int32") &&
                 strcmp(dtype->valuestring, "int8") &&
                 strcmp(dtype->valuestring, "uint8"))) return 0;
        }
    }
    return 1;
}

static char* vx_string_copy(const char* value);
static void vx_model_bank_residency_clear(VxModel* model);

static size_t vx_execution_dtype_size(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_I8:
        case VX_DTYPE_U8: return 1u;
        case VX_DTYPE_F32:
        case VX_DTYPE_I32: return 4u;
        default: return 0u;
    }
}

static void vx_declared_outputs_free(VxDeclaredOutput* outputs, size_t count) {
    if (!outputs) return;
    for (size_t index = 0; index < count; index++) {
        free(outputs[index].name);
        for (uint32_t axis = 0; axis < outputs[index].rank; axis++)
            free(outputs[index].symbols[axis]);
    }
    free(outputs);
}

static int vx_graph_execution_dtype(const cJSON* item, VxDataType* out) {
    if (!cJSON_IsString(item) || !item->valuestring || !out) return -1;
    if (!strcmp(item->valuestring, "float32")) *out = VX_DTYPE_F32;
    else if (!strcmp(item->valuestring, "int32")) *out = VX_DTYPE_I32;
    else if (!strcmp(item->valuestring, "int8")) *out = VX_DTYPE_I8;
    else if (!strcmp(item->valuestring, "uint8")) *out = VX_DTYPE_U8;
    else return -1;
    return 0;
}

static int vx_declared_output_set_json(VxDeclaredOutput* output,
                                       const cJSON* shape,
                                       const cJSON* dtype) {
    size_t elements = 1u;
    size_t element_size;
    int rank;
    if (!output || !cJSON_IsArray(shape) ||
        vx_graph_execution_dtype(dtype, &output->dtype) != 0)
        return -1;
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return -1;
    memset(output->shape, 0, sizeof(output->shape));
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* dimension = cJSON_GetArrayItem(shape, axis);
        if (!cJSON_IsNumber(dimension) ||
            dimension->valuedouble != (double)dimension->valueint ||
            dimension->valueint <= 0 ||
            (size_t)dimension->valueint > SIZE_MAX / elements)
            return -1;
        output->shape[axis] = dimension->valueint;
        elements *= (size_t)dimension->valueint;
    }
    element_size = vx_execution_dtype_size(output->dtype);
    if (!element_size || elements > SIZE_MAX / element_size) return -1;
    output->rank = (uint32_t)rank;
    output->byte_size = elements * element_size;
    return 0;
}

static int vx_declared_output_set_weight(VxDeclaredOutput* output,
                                         const SafetensorsTensor* tensor) {
    size_t storage_size;
    size_t execution_size;
    size_t elements = 1u;
    VxDataType dtype;
    if (!output || !tensor || tensor->ndim < 0 ||
        tensor->ndim > (int)VX_MAX_TENSOR_RANK)
        return -1;
    switch (tensor->dtype) {
        case SAFETENSORS_DTYPE_F32: dtype = VX_DTYPE_F32; break;
        case SAFETENSORS_DTYPE_F16: dtype = VX_DTYPE_F32; break;
        case SAFETENSORS_DTYPE_I32: dtype = VX_DTYPE_I32; break;
        case SAFETENSORS_DTYPE_I8: dtype = VX_DTYPE_I8; break;
        case SAFETENSORS_DTYPE_U8: dtype = VX_DTYPE_U8; break;
        default: return -1;
    }
    for (int axis = 0; axis < tensor->ndim; axis++) {
        if (tensor->shape[axis] <= 0 ||
            (size_t)tensor->shape[axis] > SIZE_MAX / elements)
            return -1;
        elements *= (size_t)tensor->shape[axis];
    }
    if (safetensors_tensor_nbytes(tensor->dtype, tensor->shape,
                                  tensor->ndim, &storage_size) != 0 ||
        storage_size != tensor->nbytes ||
        elements > SIZE_MAX / vx_execution_dtype_size(dtype))
        return -1;
    execution_size = elements * vx_execution_dtype_size(dtype);
    output->dtype = dtype;
    output->rank = (uint32_t)tensor->ndim;
    memset(output->shape, 0, sizeof(output->shape));
    for (int axis = 0; axis < tensor->ndim; axis++)
        output->shape[axis] = tensor->shape[axis];
    output->byte_size = execution_size;
    return 0;
}

static VxStatus vx_graph_package_validate_envelope(
    const char* path,
    const char* const* weight_paths,
    size_t weight_path_count,
    int require_resolved_outputs,
    VxDeclaredOutput** out_outputs,
    size_t* out_output_count) {
    FILE* file;
    long signed_size;
    size_t size;
    size_t read_size;
    char* bytes;
    cJSON* root;
    cJSON* format;
    cJSON* outputs;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    VxDeclaredOutput* declared_outputs = NULL;
    unsigned char* graph_resolved = NULL;
    size_t output_count = 0;
    if (!out_outputs || !out_output_count ||
        (weight_path_count && !weight_paths))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_outputs = NULL;
    *out_output_count = 0;
    file = fopen(path, "rb");
    if (!file) return VX_STATUS_IO_ERROR;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (signed_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return VX_STATUS_IO_ERROR;
    }
    size = (size_t)signed_size;
    if ((long)size != signed_size || size == SIZE_MAX) {
        fclose(file);
        return VX_STATUS_INVALID_GRAPH;
    }
    bytes = (char*)malloc(size + 1u);
    if (!bytes) {
        fclose(file);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    read_size = fread(bytes, 1, size, file);
    if (read_size != size || ferror(file)) {
        fclose(file);
        free(bytes);
        return VX_STATUS_IO_ERROR;
    }
    if (fclose(file) != 0) {
        free(bytes);
        return VX_STATUS_IO_ERROR;
    }
    bytes[size] = '\0';
    root = cJSON_ParseWithLength(bytes, size + 1u);
    free(bytes);
    if (!root) return VX_STATUS_INVALID_GRAPH;
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if (cJSON_IsObject(root) && vx_json_object_keys_unique_recursive(root) &&
        cJSON_IsString(format) && format->valuestring &&
        !strcmp(format->valuestring, "volvox-graph/v1") &&
        vx_graph_outputs_envelope_valid(outputs) &&
        vx_graph_envelope_schema_valid(root)) {
        output_count = (size_t)cJSON_GetArraySize(outputs);
        declared_outputs = (VxDeclaredOutput*)calloc(
            output_count, sizeof(*declared_outputs));
        graph_resolved = (unsigned char*)calloc(output_count, 1u);
        if (!declared_outputs || !graph_resolved) {
            status = VX_STATUS_OUT_OF_MEMORY;
        } else {
            status = VX_STATUS_OK;
            for (size_t index = 0; index < output_count; index++) {
                const cJSON* item = cJSON_GetArrayItem(
                    outputs, (int)index);
                const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
                    root, "inputs");
                const cJSON* input = cJSON_IsObject(inputs)
                    ? cJSON_GetObjectItemCaseSensitive(inputs,
                                                       item->valuestring)
                    : NULL;
                declared_outputs[index].dtype = (VxDataType)-1;
                declared_outputs[index].name = vx_string_copy(
                    item->valuestring);
                if (!declared_outputs[index].name) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    break;
                }
                if (input) {
                    const cJSON* shape = cJSON_GetObjectItemCaseSensitive(
                        input, "shape");
                    const cJSON* dtype = cJSON_GetObjectItemCaseSensitive(
                        input, "dtype");
                    if (vx_declared_output_set_json(
                            &declared_outputs[index], shape, dtype) != 0) {
                        status = VX_STATUS_INVALID_GRAPH;
                        break;
                    }
                    graph_resolved[index] = 1u;
                    continue;
                }
                const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(
                    root, "nodes");
                for (const cJSON* node = nodes ? nodes->child : NULL;
                     node; node = node->next) {
                    const cJSON* node_outputs =
                        cJSON_GetObjectItemCaseSensitive(node, "outputs");
                    for (const cJSON* node_output = node_outputs
                             ? node_outputs->child : NULL;
                         node_output; node_output = node_output->next) {
                        if (!cJSON_IsString(node_output) ||
                            !node_output->valuestring ||
                            strcmp(node_output->valuestring,
                                   item->valuestring))
                            continue;
                        const cJSON* shapes =
                            cJSON_GetObjectItemCaseSensitive(
                                node, "outputs_shape");
                        const cJSON* dtypes =
                            cJSON_GetObjectItemCaseSensitive(
                                node, "outputs_dtype");
                        const cJSON* shape = cJSON_IsObject(shapes)
                            ? cJSON_GetObjectItemCaseSensitive(
                                  shapes, node_output->string)
                            : NULL;
                        const cJSON* dtype = dtypes
                            ? cJSON_GetObjectItemCaseSensitive(
                                  dtypes, node_output->string)
                            : NULL;
                        cJSON default_dtype = {0};
                        if (graph_resolved[index] || !shape ||
                            (dtypes && !dtype)) {
                            status = VX_STATUS_INVALID_GRAPH;
                            break;
                        }
                        if (!dtype) {
                            default_dtype.type = cJSON_String;
                            default_dtype.valuestring = (char*)"float32";
                            dtype = &default_dtype;
                        }
                        if (vx_declared_output_set_json(
                                &declared_outputs[index], shape, dtype) != 0) {
                            status = VX_STATUS_INVALID_GRAPH;
                            break;
                        }
                        graph_resolved[index] = 1u;
                    }
                    if (status != VX_STATUS_OK) break;
                }
                if (status != VX_STATUS_OK) break;
            }
            for (size_t weight_index = 0;
                 status == VX_STATUS_OK && weight_index < weight_path_count;
                 weight_index++) {
                SafetensorsFile weights = {0};
                int needed = 0;
                for (size_t output_index = 0;
                     output_index < output_count; output_index++)
                    if (!graph_resolved[output_index]) needed = 1;
                if (!needed) break;
                if (safetensors_load(weight_paths[weight_index], &weights) != 0) {
                    status = VX_STATUS_INVALID_GRAPH;
                    break;
                }
                for (size_t output_index = 0;
                     output_index < output_count; output_index++) {
                    const SafetensorsTensor* tensor;
                    if (graph_resolved[output_index]) continue;
                    tensor = safetensors_find_tensor(
                        &weights, declared_outputs[output_index].name);
                    if (tensor && vx_declared_output_set_weight(
                                      &declared_outputs[output_index],
                                      tensor) != 0)
                        declared_outputs[output_index].dtype =
                            (VxDataType)-2;
                }
                safetensors_free(&weights);
            }
            if (require_resolved_outputs) {
                for (size_t index = 0;
                     status == VX_STATUS_OK && index < output_count; index++)
                    if (!vx_execution_dtype_size(declared_outputs[index].dtype))
                        status = VX_STATUS_INVALID_GRAPH;
            }
        }
    }
    cJSON_Delete(root);
    free(graph_resolved);
    if (status != VX_STATUS_OK) {
        vx_declared_outputs_free(declared_outputs, output_count);
        return status;
    }
    *out_outputs = declared_outputs;
    *out_output_count = output_count;
    return status;
}

static int vx_json_object_fields(const cJSON* object,
                                 const char* const* allowed,
                                 size_t allowed_count,
                                 size_t minimum_count,
                                 size_t maximum_count) {
    size_t count = 0;
    if (!cJSON_IsObject(object)) return 0;
    for (const cJSON* item = object->child; item; item = item->next) {
        int known = 0;
        count++;
        if (!item->string) return 0;
        for (size_t index = 0; index < allowed_count; index++)
            if (!strcmp(item->string, allowed[index])) known = 1;
        if (!known) return 0;
    }
    return count >= minimum_count && count <= maximum_count;
}

static int vx_shape_symbol_valid(const char* symbol) {
    size_t length;
    if (!symbol || !symbol[0]) return 0;
    length = strlen(symbol);
    if (length > 64u || !((symbol[0] >= 'A' && symbol[0] <= 'Z') ||
                          (symbol[0] >= 'a' && symbol[0] <= 'z'))) return 0;
    for (size_t index = 1; index < length; index++) {
        unsigned char value = (unsigned char)symbol[index];
        if (!((value >= 'A' && value <= 'Z') ||
              (value >= 'a' && value <= 'z') ||
              (value >= '0' && value <= '9') || value == '_')) return 0;
    }
    return 1;
}

static int vx_json_positive_safe_i64(const cJSON* value, int64_t* out) {
    const double safe_max = 9007199254740991.0;
    int64_t converted;
    if (!cJSON_IsNumber(value) || value->valuedouble <= 0.0 ||
        value->valuedouble > safe_max) return 0;
    converted = (int64_t)value->valuedouble;
    if ((double)converted != value->valuedouble) return 0;
    if (out) *out = converted;
    return 1;
}

static int vx_dimension_json(const cJSON* dimensions,
                             const char* symbol,
                             int64_t* minimum,
                             int64_t* maximum,
                             int64_t* multiple) {
    static const char* const fields[] = {"min", "max", "multiple_of"};
    const cJSON* descriptor;
    const cJSON* min_item;
    const cJSON* max_item;
    const cJSON* multiple_item;
    int64_t min_value;
    int64_t max_value;
    int64_t multiple_value = 1;
    int64_t remainder;
    int64_t adjustment;
    if (!vx_shape_symbol_valid(symbol) || !cJSON_IsObject(dimensions)) return 0;
    descriptor = cJSON_GetObjectItemCaseSensitive(dimensions, symbol);
    if (!descriptor ||
        !vx_json_object_fields(descriptor, fields, 3u, 2u, 3u)) return 0;
    min_item = cJSON_GetObjectItemCaseSensitive(descriptor, "min");
    max_item = cJSON_GetObjectItemCaseSensitive(descriptor, "max");
    multiple_item = cJSON_GetObjectItemCaseSensitive(descriptor, "multiple_of");
    if (!vx_json_positive_safe_i64(min_item, &min_value) ||
        !vx_json_positive_safe_i64(max_item, &max_value) ||
        min_value > max_value ||
        (multiple_item &&
         !vx_json_positive_safe_i64(multiple_item, &multiple_value))) return 0;
    remainder = min_value % multiple_value;
    adjustment = remainder ? multiple_value - remainder : 0;
    if (adjustment > max_value - min_value) return 0;
    *minimum = min_value;
    *maximum = max_value;
    *multiple = multiple_value;
    return 1;
}

static int vx_declared_tensor_set(VxDeclaredTensor* tensor,
                                  const char* name,
                                  const cJSON* shape,
                                  const cJSON* dtype,
                                  const cJSON* dimensions) {
    size_t maximum_elements = 1u;
    size_t element_size;
    int rank;
    int all_fixed = 1;
    if (!tensor || !name || !name[0] || strlen(name) >= 128u ||
        !cJSON_IsArray(shape) ||
        vx_graph_execution_dtype(dtype, &tensor->dtype) != 0) return -1;
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return -1;
    tensor->name = vx_string_copy(name);
    if (!tensor->name) return -2;
    tensor->rank = (uint32_t)rank;
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* dimension = cJSON_GetArrayItem(shape, axis);
        int64_t minimum;
        int64_t maximum;
        int64_t multiple;
        if (vx_json_positive_safe_i64(dimension, &minimum)) {
            maximum = minimum;
            multiple = 1;
            tensor->kinds[axis] = VX_DIMENSION_FIXED;
            tensor->shape[axis] = minimum;
        } else if (cJSON_IsString(dimension) && dimension->valuestring &&
                   vx_dimension_json(dimensions, dimension->valuestring,
                                     &minimum, &maximum, &multiple)) {
            tensor->symbols[axis] = vx_string_copy(dimension->valuestring);
            if (!tensor->symbols[axis]) return -2;
            tensor->kinds[axis] = VX_DIMENSION_SYMBOLIC;
            all_fixed = 0;
        } else {
            return -1;
        }
        tensor->minimums[axis] = minimum;
        tensor->maximums[axis] = maximum;
        tensor->multiples[axis] = multiple;
        if ((uint64_t)maximum > SIZE_MAX / maximum_elements) return -1;
        maximum_elements *= (size_t)maximum;
    }
    element_size = vx_execution_dtype_size(tensor->dtype);
    if (!element_size || maximum_elements > SIZE_MAX / element_size) return -1;
    tensor->maximum_byte_size = maximum_elements * element_size;
    if (all_fixed) tensor->byte_size = tensor->maximum_byte_size;
    return 0;
}

static int vx_declared_tensor_clone(VxDeclaredTensor* target,
                                    const VxDeclaredTensor* source) {
    *target = *source;
    target->name = vx_string_copy(source->name);
    memset(target->symbols, 0, sizeof(target->symbols));
    if (!target->name) return -1;
    for (uint32_t axis = 0; axis < source->rank; axis++) {
        if (!source->symbols[axis]) continue;
        target->symbols[axis] = vx_string_copy(source->symbols[axis]);
        if (!target->symbols[axis]) return -1;
    }
    return 0;
}

static int vx_declared_tensor_set_weight(VxDeclaredTensor* tensor,
                                         const char* name,
                                         const SafetensorsTensor* source) {
    VxDeclaredOutput concrete = {0};
    if (vx_declared_output_set_weight(&concrete, source) != 0) return -1;
    concrete.name = vx_string_copy(name);
    if (!concrete.name) return -2;
    for (uint32_t axis = 0; axis < concrete.rank; axis++) {
        concrete.kinds[axis] = VX_DIMENSION_FIXED;
        concrete.minimums[axis] = concrete.shape[axis];
        concrete.maximums[axis] = concrete.shape[axis];
        concrete.multiples[axis] = 1;
    }
    concrete.maximum_byte_size = concrete.byte_size;
    *tensor = concrete;
    return 0;
}

typedef struct VxValidatedWeightBank {
    const char* name;
    int64_t minimum;
    int64_t maximum;
    int64_t multiple;
    uint64_t slot_count;
    size_t occurrences;
} VxValidatedWeightBank;

static int vx_graph_schema_valid(const cJSON* root) {
    static const char* const root_fields[] = {
        "format", "dimensions", "inputs", "nodes",
        "outputs", "banks", "quantization"
    };
    static const char* const input_fields[] = {"dtype", "shape"};
    static const char* const node_fields[] = {
        "id", "opType", "inputs", "outputs", "params"
    };
    static const char* const output_fields[] = {"tensor", "dtype", "shape"};
    const cJSON* format = cJSON_GetObjectItemCaseSensitive(root, "format");
    const cJSON* dimensions =
        cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    const cJSON* nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(root, "outputs");
    if (!cJSON_IsObject(root) ||
        !vx_json_object_fields(root, root_fields, 7u, 5u, 7u) ||
        !cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-graph/v1") ||
        !cJSON_IsObject(dimensions) || !cJSON_IsObject(inputs) ||
        !cJSON_IsArray(nodes) || !vx_graph_outputs_envelope_valid(outputs) ||
        !vx_graph_quantization_envelope_valid(root)) return 0;
    for (const cJSON* dimension = dimensions->child; dimension;
         dimension = dimension->next) {
        int64_t minimum;
        int64_t maximum;
        int64_t multiple;
        if (!dimension->string ||
            !vx_dimension_json(dimensions, dimension->string,
                               &minimum, &maximum, &multiple)) return 0;
    }
    /* Optional weight banks: tensor name -> the bounded dimension governing its
     * slot axis.  Slot residency is a context concern; the document only
     * declares that axis 0 of that weight is runtime-selected. */
    {
        const cJSON* banks = cJSON_GetObjectItemCaseSensitive(root, "banks");
        if (banks) {
            if (!cJSON_IsObject(banks)) return 0;
            for (const cJSON* bank = banks->child; bank; bank = bank->next) {
                int64_t minimum;
                int64_t maximum;
                int64_t multiple;
                if (!bank->string || !cJSON_IsString(bank) || !bank->valuestring ||
                    !vx_dimension_json(dimensions, bank->valuestring,
                                       &minimum, &maximum, &multiple)) return 0;
            }
        }
    }
    for (const cJSON* input = inputs->child; input; input = input->next) {
        VxDeclaredTensor temporary = {0};
        int parsed;
        if (!input->string ||
            !vx_json_object_fields(input, input_fields, 2u, 2u, 2u)) return 0;
        parsed = vx_declared_tensor_set(
            &temporary, input->string,
            cJSON_GetObjectItemCaseSensitive(input, "shape"),
            cJSON_GetObjectItemCaseSensitive(input, "dtype"), dimensions);
        if (parsed != 0) {
            free(temporary.name);
            for (uint32_t axis = 0; axis < temporary.rank; axis++)
                free(temporary.symbols[axis]);
            return 0;
        }
        free(temporary.name);
        for (uint32_t axis = 0; axis < temporary.rank; axis++)
            free(temporary.symbols[axis]);
    }
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* node_inputs =
            cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const cJSON* node_outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
        if (!vx_json_object_fields(node, node_fields, 5u, 5u, 5u) ||
            !cJSON_IsString(id) || !id->valuestring || !id->valuestring[0] ||
            !cJSON_IsString(op) || !op->valuestring ||
            !vx_graph_runtime_operator_name_is_current(op->valuestring) ||
            !cJSON_IsObject(node_inputs) || !cJSON_IsObject(node_outputs) ||
            !node_outputs->child || !cJSON_IsObject(params) ||
            !vx_graph_params_affine_payload_free(params)) return 0;
        for (const cJSON* output = node_outputs->child; output;
             output = output->next) {
            VxDeclaredTensor temporary = {0};
            const cJSON* tensor;
            int parsed;
            if (!vx_json_object_fields(output, output_fields, 3u, 3u, 3u))
                return 0;
            tensor = cJSON_GetObjectItemCaseSensitive(output, "tensor");
            if (!cJSON_IsString(tensor) || !tensor->valuestring) return 0;
            parsed = vx_declared_tensor_set(
                &temporary, tensor->valuestring,
                cJSON_GetObjectItemCaseSensitive(output, "shape"),
                cJSON_GetObjectItemCaseSensitive(output, "dtype"), dimensions);
            free(temporary.name);
            for (uint32_t axis = 0; axis < temporary.rank; axis++)
                free(temporary.symbols[axis]);
            if (parsed != 0) return 0;
        }
    }
    return 1;
}

static VxStatus vx_graph_package_validate(
    const char* path,
    const char* const* weight_paths,
    size_t weight_path_count,
    int require_resolved_outputs,
    const VxBankResidency* bank_residency,
    size_t bank_residency_count,
    VxDeclaredTensor** out_inputs,
    size_t* out_input_count,
    VxDeclaredTensor** out_outputs,
    size_t* out_output_count) {
    FILE* file = NULL;
    char* bytes = NULL;
    cJSON* root = NULL;
    VxDeclaredTensor* inputs = NULL;
    VxDeclaredTensor* outputs = NULL;
    VxValidatedWeightBank* validated_banks = NULL;
    unsigned char* resolved = NULL;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    long signed_size;
    size_t size;
    size_t input_count = 0u;
    size_t output_count = 0u;
    if (!path || !out_inputs || !out_input_count || !out_outputs ||
        !out_output_count || (weight_path_count && !weight_paths) ||
        (bank_residency_count && !bank_residency))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_inputs = NULL;
    *out_input_count = 0;
    *out_outputs = NULL;
    *out_output_count = 0;
    file = fopen(path, "rb");
    if (!file) return VX_STATUS_IO_ERROR;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (signed_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 ||
        (size_t)signed_size == SIZE_MAX) {
        fclose(file);
        return VX_STATUS_IO_ERROR;
    }
    size = (size_t)signed_size;
    bytes = (char*)malloc(size + 1u);
    if (!bytes) {
        fclose(file);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    {
        size_t bytes_read = fread(bytes, 1, size, file);
        int read_failed = bytes_read != size || ferror(file);
        int close_failed = fclose(file) != 0;
        if (read_failed || close_failed) {
            free(bytes);
            return VX_STATUS_IO_ERROR;
        }
    }
    bytes[size] = '\0';
    root = cJSON_ParseWithLength(bytes, size + 1u);
    free(bytes);
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) goto done;
    {
        const cJSON* dimensions =
            cJSON_GetObjectItemCaseSensitive(root, "dimensions");
        const cJSON* input_object =
            cJSON_GetObjectItemCaseSensitive(root, "inputs");
        const cJSON* output_array =
            cJSON_GetObjectItemCaseSensitive(root, "outputs");
        const cJSON* bank_object =
            cJSON_GetObjectItemCaseSensitive(root, "banks");
        size_t bank_count = bank_object
            ? (size_t)cJSON_GetArraySize(bank_object) : 0u;
        input_count = (size_t)cJSON_GetArraySize(input_object);
        output_count = (size_t)cJSON_GetArraySize(output_array);
        inputs = input_count
            ? (VxDeclaredTensor*)calloc(input_count, sizeof(*inputs)) : NULL;
        outputs = (VxDeclaredTensor*)calloc(output_count, sizeof(*outputs));
        resolved = (unsigned char*)calloc(output_count, 1u);
        validated_banks = bank_count
            ? (VxValidatedWeightBank*)calloc(
                  bank_count, sizeof(*validated_banks))
            : NULL;
        if ((input_count && !inputs) || !outputs || !resolved ||
            (bank_count && !validated_banks)) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        if (bank_count) {
            size_t bank_index = 0;
            for (const cJSON* bank = bank_object->child; bank;
                 bank = bank->next, bank_index++) {
                VxValidatedWeightBank* validated =
                    &validated_banks[bank_index];
                validated->name = bank->string;
                if (!vx_dimension_json(
                        dimensions, bank->valuestring,
                        &validated->minimum, &validated->maximum,
                        &validated->multiple)) {
                    status = VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
            }
        }
        size_t index = 0;
        for (const cJSON* item = input_object->child; item;
             item = item->next, index++) {
            int parsed = vx_declared_tensor_set(
                &inputs[index], item->string,
                cJSON_GetObjectItemCaseSensitive(item, "shape"),
                cJSON_GetObjectItemCaseSensitive(item, "dtype"), dimensions);
            if (parsed != 0) {
                status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                        VX_STATUS_INVALID_GRAPH;
                goto done;
            }
        }
        status = VX_STATUS_OK;
        for (index = 0; index < output_count; index++) {
            const cJSON* output_name =
                cJSON_GetArrayItem(output_array, (int)index);
            for (size_t input_index = 0; input_index < input_count;
                 input_index++) {
                if (strcmp(inputs[input_index].name,
                           output_name->valuestring)) continue;
                if (vx_declared_tensor_clone(&outputs[index],
                                             &inputs[input_index]) != 0) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    goto done;
                }
                resolved[index] = 1u;
            }
            const cJSON* nodes =
                cJSON_GetObjectItemCaseSensitive(root, "nodes");
            for (const cJSON* node = nodes->child; node; node = node->next) {
                const cJSON* node_outputs =
                    cJSON_GetObjectItemCaseSensitive(node, "outputs");
                for (const cJSON* descriptor = node_outputs->child; descriptor;
                     descriptor = descriptor->next) {
                    const cJSON* tensor =
                        cJSON_GetObjectItemCaseSensitive(descriptor, "tensor");
                    int parsed;
                    if (strcmp(tensor->valuestring,
                               output_name->valuestring)) continue;
                    if (resolved[index]) {
                        status = VX_STATUS_INVALID_GRAPH;
                        goto done;
                    }
                    parsed = vx_declared_tensor_set(
                        &outputs[index], tensor->valuestring,
                        cJSON_GetObjectItemCaseSensitive(descriptor, "shape"),
                        cJSON_GetObjectItemCaseSensitive(descriptor, "dtype"),
                        dimensions);
                    if (parsed != 0) {
                        status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                                VX_STATUS_INVALID_GRAPH;
                        goto done;
                    }
                    resolved[index] = 1u;
                }
            }
            if (!resolved[index]) {
                outputs[index].name = vx_string_copy(output_name->valuestring);
                outputs[index].dtype = (VxDataType)-1;
                if (!outputs[index].name) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
        for (size_t weight_index = 0; weight_index < weight_path_count;
             weight_index++) {
            SafetensorsFile weights = {0};
            if (safetensors_load(weight_paths[weight_index], &weights) != 0) {
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
            for (index = 0; index < output_count; index++) {
                const SafetensorsTensor* tensor;
                if (resolved[index]) continue;
                tensor = safetensors_find_tensor(&weights, outputs[index].name);
                if (!tensor) continue;
                char* name = outputs[index].name;
                outputs[index].name = NULL;
                int parsed = vx_declared_tensor_set_weight(
                    &outputs[index], name, tensor);
                free(name);
                if (parsed != 0) {
                    safetensors_free(&weights);
                    status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                            VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
                resolved[index] = 1u;
            }
            for (int tensor_index = 0; tensor_index < weights.tensor_count;
                 tensor_index++) {
                const SafetensorsTensor* tensor =
                    &weights.tensors[tensor_index];
                VxValidatedWeightBank* validated = NULL;
                int shape_positive;
                uint64_t slot_count;
                for (size_t bank_index = 0; bank_index < bank_count;
                     bank_index++)
                    if (!strcmp(validated_banks[bank_index].name,
                                tensor->name)) {
                        validated = &validated_banks[bank_index];
                        break;
                    }
                if (!validated) continue;
                shape_positive = tensor->ndim >= 2;
                for (int axis = 0; shape_positive && axis < tensor->ndim;
                     axis++)
                    shape_positive = tensor->shape[axis] > 0;
                if (validated->occurrences != 0u || !shape_positive) {
                    safetensors_free(&weights);
                    status = VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
                slot_count = (uint64_t)tensor->shape[0];
                if (slot_count < (uint64_t)validated->minimum ||
                    slot_count > (uint64_t)validated->maximum ||
                    slot_count % (uint64_t)validated->multiple != 0u) {
                    safetensors_free(&weights);
                    status = VX_STATUS_INVALID_GRAPH;
                    goto done;
                }
                validated->slot_count = slot_count;
                validated->occurrences = 1u;
            }
            safetensors_free(&weights);
        }
        if (require_resolved_outputs) {
            for (index = 0; index < output_count; index++)
                if (!resolved[index]) status = VX_STATUS_INVALID_GRAPH;
            for (size_t bank_index = 0;
                 status == VX_STATUS_OK && bank_index < bank_count;
                 bank_index++)
                if (validated_banks[bank_index].occurrences != 1u)
                    status = VX_STATUS_INVALID_GRAPH;
            for (size_t residency_index = 0;
                 status == VX_STATUS_OK &&
                     residency_index < bank_residency_count;
                 residency_index++) {
                const VxBankResidency* requested =
                    &bank_residency[residency_index];
                const VxValidatedWeightBank* validated = NULL;
                for (size_t bank_index = 0; bank_index < bank_count;
                     bank_index++)
                    if (!strcmp(validated_banks[bank_index].name,
                                requested->bank)) {
                        validated = &validated_banks[bank_index];
                        break;
                    }
                if (!validated || !requested->slots ||
                    !requested->slot_count ||
                    (uint64_t)requested->slots[requested->slot_count - 1u] >=
                        validated->slot_count)
                    status = VX_STATUS_INVALID_ARGUMENT;
            }
        }
        if (status == VX_STATUS_OK) {
            *out_inputs = inputs;
            *out_input_count = input_count;
            *out_outputs = outputs;
            *out_output_count = output_count;
            inputs = NULL;
            outputs = NULL;
        }
    }
done:
    cJSON_Delete(root);
    free(validated_banks);
    free(resolved);
    vx_declared_outputs_free(inputs, inputs ? input_count : 0u);
    vx_declared_outputs_free(outputs, outputs ? output_count : 0u);
    return status;
}

static cJSON* vx_json_file_load(const char* path) {
    FILE* file = NULL;
    char* bytes = NULL;
    cJSON* root = NULL;
    long signed_size;
    size_t size;
    if (!path || !(file = fopen(path, "rb"))) return NULL;
    if (fseek(file, 0, SEEK_END) != 0 ||
        (signed_size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0 ||
        (size_t)signed_size == SIZE_MAX) goto done;
    size = (size_t)signed_size;
    bytes = (char*)malloc(size + 1u);
    if (!bytes) goto done;
    if (fread(bytes, 1, size, file) != size || ferror(file)) goto done;
    bytes[size] = '\0';
    root = cJSON_ParseWithLength(bytes, size + 1u);
done:
    free(bytes);
    fclose(file);
    return root;
}

static VxStatus vx_text_snapshot(const char* text, char** out_path) {
    FILE* snapshot = NULL;
    char path[PATH_MAX];
    char* owned = NULL;
    size_t length;
    if (!text || !out_path) return VX_STATUS_INVALID_ARGUMENT;
    *out_path = NULL;
    length = strlen(text);
#ifdef _WIN32
    {
        char directory[MAX_PATH];
        DWORD directory_length = GetTempPathA(
            (DWORD)sizeof(directory), directory);
        if (!directory_length || directory_length >= sizeof(directory) ||
            !GetTempFileNameA(directory, "vxl", 0, path))
            return VX_STATUS_IO_ERROR;
        snapshot = fopen(path, "wb");
        if (!snapshot) {
            (void)remove(path);
            return VX_STATUS_IO_ERROR;
        }
    }
#else
    {
        const char* directory = getenv("TMPDIR");
        int descriptor;
        int written;
        if (!directory || !directory[0]) directory = "/tmp";
        written = snprintf(path, sizeof(path),
                           "%s%svolvoxai-lowered-XXXXXX", directory,
                           directory[strlen(directory) - 1u] == '/' ? "" : "/");
        if (written < 0 || (size_t)written >= sizeof(path))
            return VX_STATUS_IO_ERROR;
        descriptor = mkstemp(path);
        if (descriptor < 0) return VX_STATUS_IO_ERROR;
        snapshot = fdopen(descriptor, "wb");
        if (!snapshot) {
            close(descriptor);
            (void)remove(path);
            return VX_STATUS_IO_ERROR;
        }
    }
#endif
    owned = vx_string_copy(path);
    if (!owned || (length && fwrite(text, 1, length, snapshot) != length) ||
        fflush(snapshot) != 0 || fclose(snapshot) != 0) {
        if (owned) vx_snapshot_path_release(owned);
        else {
            fclose(snapshot);
            (void)remove(path);
        }
        return owned ? VX_STATUS_IO_ERROR : VX_STATUS_OUT_OF_MEMORY;
    }
    *out_path = owned;
    return VX_STATUS_OK;
}

static cJSON* vx_shape_minimum_projection(const cJSON* shape,
                                          const cJSON* dimensions) {
    cJSON* projected;
    int rank;
    if (!cJSON_IsArray(shape)) return NULL;
    rank = cJSON_GetArraySize(shape);
    if (rank < 0 || rank > (int)VX_MAX_TENSOR_RANK) return NULL;
    projected = cJSON_CreateArray();
    if (!projected) return NULL;
    for (int axis = 0; axis < rank; axis++) {
        const cJSON* item = cJSON_GetArrayItem(shape, axis);
        int64_t extent;
        if (!vx_json_positive_safe_i64(item, &extent)) {
            int64_t minimum;
            int64_t maximum;
            int64_t multiple;
            int64_t remainder;
            int64_t adjustment;
            if (!cJSON_IsString(item) || !item->valuestring ||
                !vx_dimension_json(dimensions, item->valuestring,
                                   &minimum, &maximum, &multiple)) {
                cJSON_Delete(projected);
                return NULL;
            }
            remainder = minimum % multiple;
            adjustment = remainder ? multiple - remainder : 0;
            if (adjustment > maximum - minimum) {
                cJSON_Delete(projected);
                return NULL;
            }
            extent = minimum + adjustment;
        }
        /* The private physical tensor table intentionally remains int-sized.
         * Providers may support wider logical dimensions, but the built-in
         * native route must reject such a domain at context creation. */
        if (extent <= 0 || extent > INT_MAX ||
            !cJSON_AddItemToArray(projected,
                                  cJSON_CreateNumber((double)extent))) {
            cJSON_Delete(projected);
            return NULL;
        }
    }
    return projected;
}

/* Lower the closed logical v1 envelope into the private engine's concrete
 * bootstrap projection. This file is process-owned and immediately removed
 * after build_graph() has parsed it; it is not a compatibility input path. */
static VxStatus vx_graph_lower_private_bootstrap(const char* graph_path,
                                                 char** out_path) {
    cJSON* root = NULL;
    cJSON* dimensions;
    cJSON* inputs;
    cJSON* nodes;
    char* encoded = NULL;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    if (!graph_path || !out_path) return VX_STATUS_INVALID_ARGUMENT;
    *out_path = NULL;
    root = vx_json_file_load(graph_path);
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) goto done;
    dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    for (cJSON* input = inputs->child; input; input = input->next) {
        cJSON* projected = vx_shape_minimum_projection(
            cJSON_GetObjectItemCaseSensitive(input, "shape"), dimensions);
        if (!projected || !cJSON_ReplaceItemInObjectCaseSensitive(
                              input, "shape", projected)) {
            cJSON_Delete(projected);
            goto done;
        }
    }
    for (cJSON* node = nodes->child; node; node = node->next) {
        cJSON* logical_outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        cJSON* physical_outputs = cJSON_CreateObject();
        cJSON* output_shapes = cJSON_CreateObject();
        cJSON* output_dtypes = cJSON_CreateObject();
        if (!physical_outputs || !output_shapes || !output_dtypes) {
            cJSON_Delete(physical_outputs);
            cJSON_Delete(output_shapes);
            cJSON_Delete(output_dtypes);
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        for (cJSON* output = logical_outputs->child; output;
             output = output->next) {
            cJSON* tensor = cJSON_GetObjectItemCaseSensitive(output, "tensor");
            cJSON* dtype = cJSON_GetObjectItemCaseSensitive(output, "dtype");
            cJSON* projected = vx_shape_minimum_projection(
                cJSON_GetObjectItemCaseSensitive(output, "shape"), dimensions);
            if (!output->string || !cJSON_IsString(tensor) ||
                !tensor->valuestring || !cJSON_IsString(dtype) ||
                !dtype->valuestring || !projected) {
                cJSON_Delete(projected);
                cJSON_Delete(physical_outputs);
                cJSON_Delete(output_shapes);
                cJSON_Delete(output_dtypes);
                goto done;
            }
            if (!cJSON_AddStringToObject(physical_outputs, output->string,
                                         tensor->valuestring) ||
                !cJSON_AddItemToObject(output_shapes, output->string,
                                       projected)) {
                cJSON_Delete(projected);
                cJSON_Delete(physical_outputs);
                cJSON_Delete(output_shapes);
                cJSON_Delete(output_dtypes);
                goto done;
            }
            projected = NULL; /* output_shapes owns it now. */
            if (!cJSON_AddStringToObject(output_dtypes, output->string,
                                         dtype->valuestring)) {
                cJSON_Delete(physical_outputs);
                cJSON_Delete(output_shapes);
                cJSON_Delete(output_dtypes);
                goto done;
            }
        }
        if (!cJSON_ReplaceItemInObjectCaseSensitive(node, "outputs",
                                                     physical_outputs)) {
            cJSON_Delete(physical_outputs);
            cJSON_Delete(output_shapes);
            cJSON_Delete(output_dtypes);
            goto done;
        }
        cJSON_AddItemToObject(node, "outputs_shape", output_shapes);
        cJSON_AddItemToObject(node, "outputs_dtype", output_dtypes);
    }
    cJSON_DeleteItemFromObjectCaseSensitive(root, "dimensions");
    /* "banks" is kept: vx_runtime_load_model checks residency requests against
     * it, and the engine needs it to know which weights are sliceable. */
    encoded = cJSON_PrintUnformatted(root);
    if (!encoded) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    status = vx_text_snapshot(encoded, out_path);
done:
    free(encoded);
    cJSON_Delete(root);
    return status;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
/* Tests that exercise full-profile engine internals still enter through the
 * canonical bounded graph contract.  Keep the concrete bootstrap format
 * private by exposing only this test-build bridge, never a legacy graph input
 * path or a production symbol. */
int vx_public_api_test_private_engine_init(const char* graph_path,
                                           const char* weights_path) {
    char* lowered_graph_path = NULL;
    VxStatus status = vx_graph_lower_private_bootstrap(
        graph_path, &lowered_graph_path);
    int result;
    if (status != VX_STATUS_OK) return -1;
    result = volvoxai_engine_init(lowered_graph_path, weights_path);
    vx_snapshot_path_release(lowered_graph_path);
    return result;
}
#endif

static VxStatus vx_logical_tensors_load(const VxModel* model,
                                        VxDeclaredTensor** out_tensors,
                                        size_t* out_count) {
    const cJSON* root = NULL;
    const cJSON* dimensions;
    const cJSON* inputs;
    const cJSON* nodes;
    VxDeclaredTensor* tensors = NULL;
    size_t count = 0;
    size_t cursor = 0;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    if (!model || !out_tensors || !out_count) return VX_STATUS_INVALID_ARGUMENT;
    *out_tensors = NULL;
    *out_count = 0;
    root = model->logical_graph;
    if (!root || !vx_json_object_keys_unique_recursive(root) ||
        !vx_graph_schema_valid(root)) goto done;
    dimensions = cJSON_GetObjectItemCaseSensitive(root, "dimensions");
    inputs = cJSON_GetObjectItemCaseSensitive(root, "inputs");
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    count = (size_t)cJSON_GetArraySize(inputs);
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        size_t output_count = (size_t)cJSON_GetArraySize(outputs);
        if (output_count > MAXT - count) goto done;
        count += output_count;
    }
    if (count > MAXT) goto done;
    tensors = (VxDeclaredTensor*)calloc(count, sizeof(*tensors));
    if (!tensors) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    for (const cJSON* input = inputs->child; input;
         input = input->next, cursor++) {
        int parsed = vx_declared_tensor_set(
            &tensors[cursor], input->string,
            cJSON_GetObjectItemCaseSensitive(input, "shape"),
            cJSON_GetObjectItemCaseSensitive(input, "dtype"), dimensions);
        if (parsed != 0) {
            status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                    VX_STATUS_INVALID_GRAPH;
            goto done;
        }
    }
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        for (const cJSON* output = outputs->child; output;
             output = output->next, cursor++) {
            const cJSON* tensor =
                cJSON_GetObjectItemCaseSensitive(output, "tensor");
            int parsed = vx_declared_tensor_set(
                &tensors[cursor], tensor->valuestring,
                cJSON_GetObjectItemCaseSensitive(output, "shape"),
                cJSON_GetObjectItemCaseSensitive(output, "dtype"), dimensions);
            if (parsed != 0) {
                status = parsed == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                        VX_STATUS_INVALID_GRAPH;
                goto done;
            }
            for (size_t prior = 0; prior < cursor; prior++)
                if (!strcmp(tensors[prior].name, tensors[cursor].name))
                    goto done;
        }
    }
    /* Output-only symbols are legal. They remain unbound until the first
     * canonical node inference that produces them, then participate in the
     * same graph-wide equality class as public-input symbols. */
    *out_tensors = tensors;
    *out_count = count;
    tensors = NULL;
    status = VX_STATUS_OK;
done:
    vx_declared_outputs_free(tensors, tensors ? count : 0u);
    return status;
}

static int vx_dimension_domain_is_dynamic(const VxDeclaredTensor* tensor,
                                          uint32_t axis) {
    int64_t first;
    int64_t last;
    int64_t remainder;
    int64_t adjustment;
    if (!tensor || axis >= tensor->rank ||
        tensor->kinds[axis] != VX_DIMENSION_SYMBOLIC) return 0;
    remainder = tensor->minimums[axis] % tensor->multiples[axis];
    adjustment = remainder ? tensor->multiples[axis] - remainder : 0;
    if (adjustment > tensor->maximums[axis] - tensor->minimums[axis]) return 0;
    first = tensor->minimums[axis] + adjustment;
    last = tensor->maximums[axis] -
        tensor->maximums[axis] % tensor->multiples[axis];
    return first < last;
}

/* Concat is the v1 exception that may introduce a new symbolic dimension:
 * exactly one prior non-affine source plus fixed extents. Keep that affine
 * definition graph-owned so a later node can only repeat the identical fact. */
typedef enum VxAffineSymbolDefinitionKind {
    VX_AFFINE_SYMBOL_NON_AFFINE = 1,
    VX_AFFINE_SYMBOL_RELATION = 2,
} VxAffineSymbolDefinitionKind;

typedef struct VxAffineSymbolDefinition {
    const char* symbol;
    VxAffineSymbolDefinitionKind kind;
    const char* source;
    int64_t offset;
} VxAffineSymbolDefinition;

typedef struct VxConcatAffineCandidate {
    const char* target;
    const char* source;
    int64_t offset;
    int present;
} VxConcatAffineCandidate;

typedef struct VxDimensionProgression {
    int64_t first;
    int64_t last;
    int64_t step;
    uint64_t count;
} VxDimensionProgression;

static const VxDeclaredTensor* vx_declared_tensor_find(
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const char* name) {
    if (!tensors || !name) return NULL;
    for (size_t index = 0; index < tensor_count; index++)
        if (tensors[index].name && !strcmp(tensors[index].name, name))
            return &tensors[index];
    return NULL;
}

static VxAffineSymbolDefinition* vx_affine_symbol_find(
        VxAffineSymbolDefinition* definitions,
        size_t definition_count,
        const char* symbol) {
    if (!definitions || !symbol) return NULL;
    for (size_t index = 0; index < definition_count; index++)
        if (!strcmp(definitions[index].symbol, symbol))
            return &definitions[index];
    return NULL;
}

static int vx_affine_symbol_define_non_affine(
        VxAffineSymbolDefinition* definitions,
        size_t* definition_count,
        size_t definition_capacity,
        const char* symbol) {
    VxAffineSymbolDefinition* definition;
    if (!definitions || !definition_count || !symbol) return -1;
    definition = vx_affine_symbol_find(
        definitions, *definition_count, symbol);
    if (definition) return 0;
    if (*definition_count >= definition_capacity) return -1;
    definition = &definitions[(*definition_count)++];
    definition->symbol = symbol;
    definition->kind = VX_AFFINE_SYMBOL_NON_AFFINE;
    return 0;
}

static VxStatus vx_concat_domain_reject(const cJSON* node,
                                        const char* reason,
                                        char* evidence,
                                        size_t evidence_capacity) {
    const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(node, "id");
    snprintf(evidence, evidence_capacity,
             "unsupported_node=%s;unsupported_op=Concat;"
             "required=canonical-affine-concat-domain;relation_reason=%s",
             cJSON_IsString(node_id) && node_id->valuestring
                 ? node_id->valuestring : "<unknown>",
             reason ? reason : "invalid");
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static int vx_concat_input_port_index(const char* name, size_t* out_index) {
    const char* suffix;
    size_t value = 0;
    if (!name || strncmp(name, "input", 5u) || !(suffix = name + 5u)[0])
        return 0;
    if (suffix[0] == '0' && suffix[1]) return 0;
    for (; *suffix; suffix++) {
        size_t digit;
        if (*suffix < '0' || *suffix > '9') return 0;
        digit = (size_t)(*suffix - '0');
        if (value > (SIZE_MAX - digit) / 10u) return 0;
        value = value * 10u + digit;
    }
    *out_index = value;
    return 1;
}

static int vx_concat_axis(const cJSON* params,
                          uint32_t rank,
                          size_t* out_axis) {
    static const char* const fields[] = {"axis"};
    const cJSON* item;
    int64_t raw = 0;
    if (!params || !out_axis || !rank ||
        !vx_json_object_fields(params, fields, 1u, 0u, 1u)) return 0;
    item = cJSON_GetObjectItemCaseSensitive(params, "axis");
    if (item) {
        if (!cJSON_IsNumber(item) ||
            item->valuedouble < -(double)rank ||
            item->valuedouble >= (double)rank) return 0;
        raw = (int64_t)item->valuedouble;
        if ((double)raw != item->valuedouble) return 0;
    }
    if (raw < 0) raw += rank;
    *out_axis = (size_t)raw;
    return 1;
}

static int vx_declared_dimension_fixed_value(
        const VxDeclaredTensor* tensor,
        size_t axis,
        int64_t* out_value) {
    if (!tensor || axis >= tensor->rank || !out_value) return 0;
    if (tensor->kinds[axis] == VX_DIMENSION_FIXED ||
        (tensor->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
         tensor->minimums[axis] == tensor->maximums[axis])) {
        *out_value = tensor->minimums[axis];
        return 1;
    }
    return 0;
}

static int vx_declared_dimensions_provably_equal(
        const VxDeclaredTensor* left,
        size_t left_axis,
        const VxDeclaredTensor* right,
        size_t right_axis) {
    int64_t left_fixed;
    int64_t right_fixed;
    if (!left || !right || left_axis >= left->rank ||
        right_axis >= right->rank) return 0;
    if (left->kinds[left_axis] == VX_DIMENSION_SYMBOLIC &&
        right->kinds[right_axis] == VX_DIMENSION_SYMBOLIC &&
        left->symbols[left_axis] && right->symbols[right_axis] &&
        !strcmp(left->symbols[left_axis], right->symbols[right_axis]))
        return 1;
    return vx_declared_dimension_fixed_value(left, left_axis, &left_fixed) &&
        vx_declared_dimension_fixed_value(right, right_axis, &right_fixed) &&
        left_fixed == right_fixed;
}

static int vx_declared_dimension_progression(
        const VxDeclaredTensor* tensor,
        size_t axis,
        VxDimensionProgression* progression) {
    int64_t remainder;
    int64_t adjustment;
    if (!tensor || axis >= tensor->rank || !progression ||
        tensor->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
        tensor->minimums[axis] <= 0 ||
        tensor->maximums[axis] < tensor->minimums[axis] ||
        tensor->multiples[axis] <= 0) return 0;
    remainder = tensor->minimums[axis] % tensor->multiples[axis];
    adjustment = remainder ? tensor->multiples[axis] - remainder : 0;
    if (adjustment > tensor->maximums[axis] - tensor->minimums[axis])
        return 0;
    progression->first = tensor->minimums[axis] + adjustment;
    progression->last = tensor->maximums[axis] -
        tensor->maximums[axis] % tensor->multiples[axis];
    progression->step = tensor->multiples[axis];
    progression->count =
        (uint64_t)((progression->last - progression->first) /
                   progression->step) + 1u;
    return progression->first <= progression->last;
}

static VxStatus vx_concat_affine_candidate_prove(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        VxConcatAffineCandidate* candidate,
        char* evidence,
        size_t evidence_capacity) {
    const int64_t safe_max = INT64_C(9007199254740991);
    const cJSON* node_inputs =
        cJSON_GetObjectItemCaseSensitive(node, "inputs");
    const cJSON* node_outputs =
        cJSON_GetObjectItemCaseSensitive(node, "outputs");
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    const cJSON* output_port;
    const cJSON* output_name;
    const VxDeclaredTensor** inputs = NULL;
    const VxDeclaredTensor* output = NULL;
    const VxDeclaredTensor* dynamic_input = NULL;
    size_t input_count;
    size_t axis;
    int64_t fixed_axis_sum = 0;
    VxStatus status = VX_STATUS_BACKEND_UNSUPPORTED;
    int raw_input_count;
    if (!candidate) return VX_STATUS_INVALID_ARGUMENT;
    memset(candidate, 0, sizeof(*candidate));
    raw_input_count = cJSON_GetArraySize(node_inputs);
    if (!cJSON_IsObject(node_inputs) || raw_input_count < 2 ||
        raw_input_count > MAXT)
        return vx_concat_domain_reject(
            node, "invalid-input-ports", evidence, evidence_capacity);
    input_count = (size_t)raw_input_count;
    inputs = (const VxDeclaredTensor**)calloc(input_count, sizeof(*inputs));
    if (!inputs) return VX_STATUS_OUT_OF_MEMORY;
    for (const cJSON* input = node_inputs->child; input; input = input->next) {
        size_t index;
        if (!input->string ||
            !vx_concat_input_port_index(input->string, &index) ||
            index >= input_count || inputs[index] ||
            !cJSON_IsString(input) || !input->valuestring ||
            !(inputs[index] = vx_declared_tensor_find(
                  tensors, tensor_count, input->valuestring))) {
            status = vx_concat_domain_reject(
                node, "invalid-input-ports", evidence, evidence_capacity);
            goto done;
        }
    }
    for (size_t index = 0; index < input_count; index++) {
        if (!inputs[index]) {
            status = vx_concat_domain_reject(
                node, "nonconsecutive-input-ports", evidence,
                evidence_capacity);
            goto done;
        }
    }
    output_port = cJSON_IsObject(node_outputs) ? node_outputs->child : NULL;
    output_name = output_port
        ? cJSON_GetObjectItemCaseSensitive(output_port, "tensor") : NULL;
    if (!output_port || output_port->next || !output_port->string ||
        strcmp(output_port->string, "out") ||
        !cJSON_IsString(output_name) || !output_name->valuestring ||
        !(output = vx_declared_tensor_find(
              tensors, tensor_count, output_name->valuestring))) {
        status = vx_concat_domain_reject(
            node, "invalid-output-ports", evidence, evidence_capacity);
        goto done;
    }
    if (inputs[0]->rank < 1u || inputs[0]->rank > 8u ||
        !vx_concat_axis(params, inputs[0]->rank, &axis)) {
        status = vx_concat_domain_reject(
            node, "invalid-rank-or-axis", evidence, evidence_capacity);
        goto done;
    }
    for (size_t input_index = 0; input_index < input_count; input_index++) {
        int64_t fixed_axis;
        if (inputs[input_index]->rank != inputs[0]->rank) {
            status = vx_concat_domain_reject(
                node, "rank-mismatch", evidence, evidence_capacity);
            goto done;
        }
        if (inputs[input_index]->dtype != inputs[0]->dtype) {
            status = vx_concat_domain_reject(
                node, "dtype-mismatch", evidence, evidence_capacity);
            goto done;
        }
        for (size_t dimension = 0; dimension < inputs[0]->rank; dimension++) {
            if (dimension != axis &&
                !vx_declared_dimensions_provably_equal(
                    inputs[input_index], dimension, inputs[0], dimension)) {
                status = vx_concat_domain_reject(
                    node, "nonaxis-shape-mismatch", evidence,
                    evidence_capacity);
                goto done;
            }
        }
        if (!vx_declared_dimension_fixed_value(
                inputs[input_index], axis, &fixed_axis)) {
            if (dynamic_input) {
                status = vx_concat_domain_reject(
                    node, "two-dynamic-terms", evidence, evidence_capacity);
                goto done;
            }
            dynamic_input = inputs[input_index];
        } else {
            if (fixed_axis > safe_max - fixed_axis_sum) {
                status = vx_concat_domain_reject(
                    node, "axis-sum-overflow", evidence, evidence_capacity);
                goto done;
            }
            fixed_axis_sum += fixed_axis;
        }
    }
    if (output->rank != inputs[0]->rank) {
        status = vx_concat_domain_reject(
            node, "output-rank-mismatch", evidence, evidence_capacity);
        goto done;
    }
    if (output->dtype != inputs[0]->dtype) {
        status = vx_concat_domain_reject(
            node, "output-dtype-mismatch", evidence, evidence_capacity);
        goto done;
    }
    for (size_t dimension = 0; dimension < output->rank; dimension++) {
        if (dimension != axis &&
            !vx_declared_dimensions_provably_equal(
                output, dimension, inputs[0], dimension)) {
            status = vx_concat_domain_reject(
                node, "output-nonaxis-shape-mismatch", evidence,
                evidence_capacity);
            goto done;
        }
    }
    if (!dynamic_input) {
        int64_t output_axis;
        if (!vx_declared_dimension_fixed_value(output, axis, &output_axis) ||
            output_axis != fixed_axis_sum) {
            status = vx_concat_domain_reject(
                node, "fixed-axis-sum-mismatch", evidence,
                evidence_capacity);
            goto done;
        }
        status = VX_STATUS_OK;
        goto done;
    }
    if (output->kinds[axis] != VX_DIMENSION_SYMBOLIC ||
        !output->symbols[axis]) {
        status = vx_concat_domain_reject(
            node, "dynamic-target-not-symbolic", evidence,
            evidence_capacity);
        goto done;
    }
    for (size_t input_index = 0; input_index < input_count; input_index++) {
        for (size_t dimension = 0; dimension < inputs[input_index]->rank;
             dimension++) {
            if (inputs[input_index]->symbols[dimension] &&
                !strcmp(inputs[input_index]->symbols[dimension],
                        output->symbols[axis])) {
                status = vx_concat_domain_reject(
                    node, "target-not-output-only", evidence,
                    evidence_capacity);
                goto done;
            }
        }
    }
    {
        VxDimensionProgression source_progression;
        VxDimensionProgression target_progression;
        int64_t expected_first;
        int64_t expected_last;
        if (!vx_declared_dimension_progression(
                dynamic_input, axis, &source_progression) ||
            !vx_declared_dimension_progression(
                output, axis, &target_progression) ||
            source_progression.first > safe_max - fixed_axis_sum ||
            source_progression.last > safe_max - fixed_axis_sum) {
            status = vx_concat_domain_reject(
                node, "invalid-affine-progression", evidence,
                evidence_capacity);
            goto done;
        }
        expected_first = source_progression.first + fixed_axis_sum;
        expected_last = source_progression.last + fixed_axis_sum;
        if (target_progression.first != expected_first ||
            target_progression.last != expected_last ||
            target_progression.count != source_progression.count) {
            status = vx_concat_domain_reject(
                node, "affine-progression-mismatch", evidence,
                evidence_capacity);
            goto done;
        }
    }
    candidate->target = output->symbols[axis];
    candidate->source = dynamic_input->symbols[axis];
    candidate->offset = fixed_axis_sum;
    candidate->present = 1;
    status = VX_STATUS_OK;
done:
    free(inputs);
    return status;
}

static VxStatus vx_graph_affine_concat_domain_proof(
        const VxModel* model,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        uint64_t* digest,
        size_t* affine_relation_count,
        char* evidence,
        size_t evidence_capacity) {
    const cJSON* nodes;
    VxAffineSymbolDefinition* definitions = NULL;
    size_t definition_count = 0;
    size_t definition_capacity;
    VxStatus status = VX_STATUS_OK;
    if (!model || !tensors || !digest || !affine_relation_count ||
        !evidence || !evidence_capacity) return VX_STATUS_INVALID_ARGUMENT;
    *affine_relation_count = 0;
    definition_capacity = tensor_count * VX_MAX_TENSOR_RANK;
    if (!definition_capacity) definition_capacity = 1u;
    definitions = (VxAffineSymbolDefinition*)calloc(
        definition_capacity, sizeof(*definitions));
    if (!definitions) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < model->input_count; index++) {
        for (size_t axis = 0; axis < tensors[index].rank; axis++) {
            if (tensors[index].symbols[axis] &&
                vx_affine_symbol_define_non_affine(
                    definitions, &definition_count, definition_capacity,
                    tensors[index].symbols[axis]) != 0) {
                status = VX_STATUS_OUT_OF_MEMORY;
                goto done;
            }
        }
    }
    nodes = cJSON_GetObjectItemCaseSensitive(model->logical_graph, "nodes");
    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* outputs =
            cJSON_GetObjectItemCaseSensitive(node, "outputs");
        VxConcatAffineCandidate candidate = {0};
        if (cJSON_IsString(op) && op->valuestring &&
            !strcmp(op->valuestring, "Concat")) {
            VxAffineSymbolDefinition* source_definition;
            VxAffineSymbolDefinition* target_definition;
            status = vx_concat_affine_candidate_prove(
                node, tensors, tensor_count, &candidate,
                evidence, evidence_capacity);
            if (status != VX_STATUS_OK) goto done;
            if (candidate.present) {
                source_definition = vx_affine_symbol_find(
                    definitions, definition_count, candidate.source);
                if (!source_definition ||
                    source_definition->kind != VX_AFFINE_SYMBOL_NON_AFFINE) {
                    status = vx_concat_domain_reject(
                        node, "source-not-predefined-non-affine", evidence,
                        evidence_capacity);
                    goto done;
                }
                target_definition = vx_affine_symbol_find(
                    definitions, definition_count, candidate.target);
                if (target_definition) {
                    if (target_definition->kind != VX_AFFINE_SYMBOL_RELATION) {
                        status = vx_concat_domain_reject(
                            node, "target-predefined", evidence,
                            evidence_capacity);
                        goto done;
                    }
                    if (strcmp(target_definition->source, candidate.source) ||
                        target_definition->offset != candidate.offset) {
                        status = vx_concat_domain_reject(
                            node, "affine-relation-conflict", evidence,
                            evidence_capacity);
                        goto done;
                    }
                } else {
                    if (definition_count >= definition_capacity) {
                        status = VX_STATUS_OUT_OF_MEMORY;
                        goto done;
                    }
                    target_definition = &definitions[definition_count++];
                    target_definition->symbol = candidate.target;
                    target_definition->kind = VX_AFFINE_SYMBOL_RELATION;
                    target_definition->source = candidate.source;
                    target_definition->offset = candidate.offset;
                    *digest = vx_revision_update(
                        *digest, candidate.target,
                        strlen(candidate.target) + 1u);
                    *digest = vx_revision_update(
                        *digest, candidate.source,
                        strlen(candidate.source) + 1u);
                    *digest = vx_revision_update(
                        *digest, &candidate.offset,
                        sizeof(candidate.offset));
                    (*affine_relation_count)++;
                }
            }
        }
        for (const cJSON* output_port = cJSON_IsObject(outputs)
                 ? outputs->child : NULL;
             output_port; output_port = output_port->next) {
            const cJSON* tensor_name =
                cJSON_GetObjectItemCaseSensitive(output_port, "tensor");
            const VxDeclaredTensor* output =
                cJSON_IsString(tensor_name) && tensor_name->valuestring
                ? vx_declared_tensor_find(
                      tensors, tensor_count, tensor_name->valuestring)
                : NULL;
            if (!output) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                goto done;
            }
            for (size_t axis = 0; axis < output->rank; axis++) {
                if (output->symbols[axis] &&
                    vx_affine_symbol_define_non_affine(
                        definitions, &definition_count, definition_capacity,
                        output->symbols[axis]) != 0) {
                    status = VX_STATUS_OUT_OF_MEMORY;
                    goto done;
                }
            }
        }
    }
done:
    free(definitions);
    return status;
}

/* The AVX2 QBatch route packs both dynamic operands into one transient buffer:
 *
 *   pairs   = ceil(K / 2)
 *   nblocks = ceil(N / 16)
 *   bytes   = nblocks * pairs * 64 + M * pairs * sizeof(int32_t)
 *
 * The expression is monotone in M/K/N, so evaluating declared maxima is a
 * conservative full-domain bound even when symbols correlate those axes. A
 * node whose operands are not logical activations contributes zero: it then
 * remains on the allocation-free native/portable route unless another already
 * charged shared workspace happens to fit its concrete call. */
static size_t vx_qbatch_cpu_workspace_domain_bound(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* a_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "a") : NULL;
    const cJSON* b_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "b") : NULL;
    const VxDeclaredTensor* a;
    const VxDeclaredTensor* b;
    int64_t m;
    int64_t k;
    int64_t b_k;
    int64_t n;
    if (!cJSON_IsString(a_ref) || !a_ref->valuestring ||
        !cJSON_IsString(b_ref) || !b_ref->valuestring ||
        !(a = vx_declared_tensor_find(
              tensors, tensor_count, a_ref->valuestring)) ||
        !(b = vx_declared_tensor_find(
              tensors, tensor_count, b_ref->valuestring)) ||
        a->rank < 2u || b->rank < 2u)
        return 0u;
    m = a->maximums[a->rank - 2u];
    k = a->maximums[a->rank - 1u];
    b_k = b->maximums[b->rank - 2u];
    n = b->maximums[b->rank - 1u];
    if (b_k > k) k = b_k;
    if (m <= 0 || k <= 0 || n <= 0 ||
        (uint64_t)m > UINT32_MAX || (uint64_t)k > UINT32_MAX ||
        (uint64_t)n > UINT32_MAX)
        return 0u;
    return vx_qbatch_matmul_i8u8_native_workspace_bytes(
        (uint32_t)m, (uint32_t)k, (uint32_t)n);
}

typedef struct VxNativeGpuDomainLimits {
    uint64_t maximum_buffer_bytes;
    uint64_t maximum_uniform_bytes;
    uint64_t maximum_total_span_bytes;
    uint64_t maximum_scratch_bytes;
    uint64_t current_graph_scratch_bytes;
    uint64_t storage_alignment;
    uint32_t maximum_grid[3];
    uint32_t maximum_block[3];
    uint32_t maximum_threads_per_block;
    uint32_t maximum_tensor_slots;
    uint32_t maximum_storage_bindings;
    uint32_t maximum_uniform_bindings;
} VxNativeGpuDomainLimits;

enum {
    VX_NATIVE_GPU_LAYOUT_LIMIT_QUERY_FAILED = -10,
    VX_NATIVE_GPU_LAYOUT_LIMIT_SET_INVALID = -11,
    VX_NATIVE_GPU_LAYOUT_SPAN_INVALID = -12,
    VX_NATIVE_GPU_LAYOUT_ARENA_EXHAUSTED = -13,
    VX_NATIVE_GPU_LAYOUT_QNORM_EXHAUSTED = -14,
    VX_NATIVE_GPU_LAYOUT_SLOT_EXHAUSTED = -15,
    VX_NATIVE_GPU_LAYOUT_SCRATCH_EXHAUSTED = -16,
};

static int vx_native_gpu_query_domain_limits(
        VolvoxAIEngineBackend backend,
        VxNativeGpuDomainLimits* limits) {
    if (!limits) return -1;
    memset(limits, 0, sizeof(*limits));
#if defined(VOLVOXAI_ENABLE_VULKAN) && VOLVOXAI_ENABLE_VULKAN
    if (backend == VOLVOXAI_BACKEND_VULKAN) {
        VulkanDomainLimits physical = {0};
        if (vk_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes = physical.maximum_storage_buffer_bytes;
        limits->maximum_uniform_bytes = physical.maximum_uniform_buffer_bytes;
        limits->maximum_total_span_bytes = physical.maximum_total_span_bytes;
        limits->maximum_scratch_bytes = physical.maximum_scratch_bytes;
        limits->current_graph_scratch_bytes =
            physical.current_graph_scratch_bytes;
        limits->storage_alignment = physical.storage_alignment;
        memcpy(limits->maximum_grid, physical.maximum_workgroups,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_workgroup_size,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_workgroup_invocations;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings =
            physical.maximum_storage_bindings;
        limits->maximum_uniform_bindings =
            physical.maximum_uniform_bindings;
    return 0;
}
#endif
#if defined(VOLVOXAI_ENABLE_OPENGL) && VOLVOXAI_ENABLE_OPENGL
    if (backend == VOLVOXAI_BACKEND_OPENGL) {
        OpenGLDomainLimits physical = {0};
        if (opengl_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes =
            physical.maximum_storage_buffer_bytes;
        limits->maximum_uniform_bytes =
            physical.maximum_uniform_buffer_bytes;
        limits->storage_alignment = 1u;
        memcpy(limits->maximum_grid, physical.maximum_workgroups,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_workgroup_size,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_workgroup_invocations;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings =
            physical.maximum_storage_bindings;
        limits->maximum_uniform_bindings =
            physical.maximum_uniform_bindings;
        return 0;
    }
#endif
#if defined(VOLVOXAI_ENABLE_METAL) && VOLVOXAI_ENABLE_METAL
    if (backend == VOLVOXAI_BACKEND_METAL) {
        MetalDomainLimits physical = {0};
        if (metal_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes = physical.maximum_buffer_bytes;
        limits->maximum_uniform_bytes = physical.maximum_buffer_bytes;
        limits->storage_alignment = 1u;
        memcpy(limits->maximum_grid, physical.maximum_workgroups,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_workgroup_size,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_threads_per_workgroup;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings = physical.maximum_bindings;
        limits->maximum_uniform_bindings = 1u;
        return 0;
    }
#endif
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (backend == VOLVOXAI_BACKEND_CUDA) {
        CudaDomainLimits physical = {0};
        if (cuda_query_domain_limits(&physical) != 0) return -1;
        limits->maximum_buffer_bytes = physical.total_memory_bytes;
        limits->maximum_uniform_bytes = physical.total_memory_bytes;
        limits->maximum_total_span_bytes = physical.total_memory_bytes;
        limits->storage_alignment = 1u;
        memcpy(limits->maximum_grid, physical.maximum_grid,
               sizeof(limits->maximum_grid));
        memcpy(limits->maximum_block, physical.maximum_block,
               sizeof(limits->maximum_block));
        limits->maximum_threads_per_block =
            physical.maximum_threads_per_block;
        limits->maximum_tensor_slots = physical.maximum_tensor_slots;
        limits->maximum_storage_bindings = UINT32_MAX;
        limits->maximum_uniform_bindings = UINT32_MAX;
        return 0;
    }
#endif
    return -1;
}

static int vx_native_gpu_slot_capacity_proven(
        uint64_t tensor_count,
        uint64_t node_count,
        uint64_t physical_span_count,
        uint64_t slot_limit,
        uint64_t* required_out) {
    uint64_t required;
    if (!required_out || !slot_limit ||
        node_count > (UINT64_MAX - tensor_count) / 4u ||
        physical_span_count >
            UINT64_MAX - tensor_count - node_count * 4u)
        return 0;
    required = tensor_count + node_count * 4u + physical_span_count;
    *required_out = required;
    return required <= slot_limit;
}

static int vx_native_gpu_accumulate_resource_bytes(
    uint64_t* total, uint64_t bytes, uint64_t copies);
static int vx_native_gpu_align_resource_bytes(
    uint64_t bytes, uint64_t alignment, uint64_t* aligned_out);

/* Vulkan reserves one aligned tail arena for maximum-domain normalization
 * statistics followed by all shape-invariant dispatch data observed during
 * the validation forward.  This mirrors vk_graph_domain_scratch_layout: each
 * statistics range ends at an independently aligned cursor and the measured
 * bootstrap high-water mark is then reserved after both ranges. */
static int vx_native_gpu_vulkan_scratch_capacity_proven(
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t validation_graph_scratch_bytes,
        uint64_t storage_alignment,
        uint64_t maximum_scratch_bytes,
        uint64_t* required_out) {
    uint64_t required = 0u;
    uint64_t aligned;
    if (!required_out || !storage_alignment || !maximum_scratch_bytes ||
        validation_graph_scratch_bytes % storage_alignment)
        return 0;
    if (qgroupnorm_stats_bytes) {
        if (!vx_native_gpu_align_resource_bytes(
                qgroupnorm_stats_bytes, storage_alignment, &aligned) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &required, aligned, 1u))
            return 0;
    }
    if (qlayernorm_stats_bytes) {
        if (!vx_native_gpu_align_resource_bytes(
                qlayernorm_stats_bytes, storage_alignment, &aligned) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &required, aligned, 1u))
            return 0;
    }
    if (!vx_native_gpu_accumulate_resource_bytes(
            &required, validation_graph_scratch_bytes, 1u))
        return 0;
    *required_out = required;
    return required <= maximum_scratch_bytes;
}

static int vx_native_gpu_maximum_layout_proof(
        VxEngineState* validation,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        VolvoxAIEngineBackend backend,
        VxNativeGpuDomainLimits* limits,
        uint64_t* arena_bytes_out,
        uint64_t* device_span_bytes_out,
        uint64_t* span_count_out,
        uint64_t* qgroupnorm_stats_bytes_out,
        uint64_t* qlayernorm_stats_bytes_out,
        uint64_t* scratch_required_bytes_out) {
    VolvoxAIEngineMaximumTensor* maximums = NULL;
    VxEngineStateScope scope;
    const VxDynamicShapeMaximumLayout* layout;
    uint64_t maximum_slots;
    int configured;
    int result = -1;
    if (!validation || !tensors || !tensor_count || !limits ||
        !arena_bytes_out || !device_span_bytes_out || !span_count_out ||
        !qgroupnorm_stats_bytes_out || !qlayernorm_stats_bytes_out ||
        !scratch_required_bytes_out ||
        !vx_native_gpu_backend(backend) || validation->tensor_count < 0 ||
        validation->node_count < 0)
        return -1;
    maximums = (VolvoxAIEngineMaximumTensor*)calloc(
        tensor_count, sizeof(*maximums));
    if (!maximums) return -2;
    for (size_t index = 0; index < tensor_count; index++) {
        maximums[index].name = tensors[index].name;
        maximums[index].dtype = (int)tensors[index].dtype;
        maximums[index].rank = (int)tensors[index].rank;
        for (uint32_t axis = 0; axis < tensors[index].rank; axis++)
            maximums[index].maximum_shape[axis] =
                (int)tensors[index].maximums[axis];
        maximums[index].maximum_byte_size = tensors[index].maximum_byte_size;
    }
    scope = vx_engine_state_scope_enter(validation);
    configured = volvoxai_engine_configure_dynamic_shape_domain(
        maximums, tensor_count);
    if (configured != 0) {
        result = configured;
        goto done;
    }
    if (vx_native_gpu_query_domain_limits(backend, limits) != 0) {
        result = VX_NATIVE_GPU_LAYOUT_LIMIT_QUERY_FAILED;
        goto done;
    }
    layout = &validation->dynamic_shape_maximum_layout;
    if (!layout->configured || layout->tensor_count < 0 ||
        (size_t)layout->tensor_count != tensor_count ||
        layout->physical_span_count <= 0 || !layout->arena_bytes ||
        !layout->physical_capacities ||
        !limits->maximum_buffer_bytes || !limits->maximum_uniform_bytes ||
        !limits->maximum_grid[0] || !limits->maximum_grid[1] ||
        !limits->storage_alignment || !limits->maximum_grid[2] ||
        !limits->maximum_block[0] ||
        !limits->maximum_block[1] || !limits->maximum_block[2] ||
        !limits->maximum_threads_per_block ||
        (backend == VOLVOXAI_BACKEND_VULKAN &&
         (!limits->maximum_scratch_bytes ||
          limits->current_graph_scratch_bytes >
              limits->maximum_scratch_bytes ||
          limits->current_graph_scratch_bytes %
              limits->storage_alignment)) ||
        limits->maximum_storage_bindings < 7u ||
        limits->maximum_uniform_bindings < 1u) {
        result = VX_NATIVE_GPU_LAYOUT_LIMIT_SET_INVALID;
        goto done;
    }
    *device_span_bytes_out = 0u;
    for (int index = 0; index < layout->physical_span_count; index++) {
        uint64_t capacity = (uint64_t)layout->physical_capacities[index];
        uint64_t remainder;
        uint64_t aligned;
        if (!capacity || capacity > limits->maximum_buffer_bytes) {
            result = VX_NATIVE_GPU_LAYOUT_SPAN_INVALID;
            goto done;
        }
        remainder = capacity % limits->storage_alignment;
        if (remainder && capacity >
                UINT64_MAX - (limits->storage_alignment - remainder)) {
            result = VX_NATIVE_GPU_LAYOUT_SPAN_INVALID;
            goto done;
        }
        aligned = remainder
            ? capacity + (limits->storage_alignment - remainder)
            : capacity;
        if (aligned > UINT64_MAX - *device_span_bytes_out) {
            result = VX_NATIVE_GPU_LAYOUT_SPAN_INVALID;
            goto done;
        }
        *device_span_bytes_out += aligned;
    }
    if (limits->maximum_total_span_bytes &&
        *device_span_bytes_out > limits->maximum_total_span_bytes) {
        result = VX_NATIVE_GPU_LAYOUT_ARENA_EXHAUSTED;
        goto done;
    }
    if ((layout->qgroupnorm_stats_bytes &&
         layout->qgroupnorm_stats_bytes > limits->maximum_buffer_bytes) ||
        (layout->qlayernorm_stats_bytes &&
         layout->qlayernorm_stats_bytes > limits->maximum_buffer_bytes)) {
        result = VX_NATIVE_GPU_LAYOUT_QNORM_EXHAUSTED;
        goto done;
    }
    *scratch_required_bytes_out = 0u;
    if (backend == VOLVOXAI_BACKEND_VULKAN &&
        !vx_native_gpu_vulkan_scratch_capacity_proven(
            (uint64_t)layout->qgroupnorm_stats_bytes,
            (uint64_t)layout->qlayernorm_stats_bytes,
            limits->current_graph_scratch_bytes,
            limits->storage_alignment,
            limits->maximum_scratch_bytes,
            scratch_required_bytes_out)) {
        result = VX_NATIVE_GPU_LAYOUT_SCRATCH_EXHAUSTED;
        goto done;
    }
    /* Bootstrap graph slots are keyed by host address: every T can own at
     * most one slot, while the qualified quantized routes add at most four
     * persistent metadata keys per node (QConv2D is the largest at three).
     * Domain publication discards logical transient slots, retains the
     * immutable subset, and adds one slot per physical span. Keeping the full
     * T count plus both conservative additions therefore bounds both phases
     * without charging the logical tensor table twice. */
    if (!vx_native_gpu_slot_capacity_proven(
            (uint64_t)validation->tensor_count,
            (uint64_t)validation->node_count,
            (uint64_t)layout->physical_span_count,
            (uint64_t)limits->maximum_tensor_slots,
            &maximum_slots)) {
        result = VX_NATIVE_GPU_LAYOUT_SLOT_EXHAUSTED;
        goto done;
    }
    *arena_bytes_out = (uint64_t)layout->arena_bytes;
    *span_count_out = (uint64_t)layout->physical_span_count;
    *qgroupnorm_stats_bytes_out =
        (uint64_t)layout->qgroupnorm_stats_bytes;
    *qlayernorm_stats_bytes_out =
        (uint64_t)layout->qlayernorm_stats_bytes;
    result = 0;
done:
    vx_engine_state_scope_leave(scope);
    free(maximums);
    return result;
}

static VxStatus vx_native_gpu_domain_reject(const cJSON* node,
                                      const char* reason,
                                      char* evidence,
                                      size_t evidence_capacity) {
    const cJSON* node_id = node
        ? cJSON_GetObjectItemCaseSensitive(node, "id") : NULL;
    const cJSON* op = node
        ? cJSON_GetObjectItemCaseSensitive(node, "opType") : NULL;
    snprintf(evidence, evidence_capacity,
             "unsupported_node=%s;unsupported_op=%s;"
             "required=native-gpu-full-bounded-domain;predicate_reason=%s",
             cJSON_IsString(node_id) && node_id->valuestring
                 ? node_id->valuestring : "<unknown>",
             cJSON_IsString(op) && op->valuestring
                 ? op->valuestring : "<unknown>",
             reason ? reason : "invalid");
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static const VxDeclaredTensor* vx_native_gpu_node_input_named(
        const cJSON* node,
        const char* role,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* reference = cJSON_IsObject(inputs) && role
        ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
    return cJSON_IsString(reference) && reference->valuestring
        ? vx_declared_tensor_find(tensors, tensor_count, reference->valuestring)
        : NULL;
}

static const VxDeclaredTensor* vx_native_gpu_node_input_domain(
        const cJSON* node,
        const char* role,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        VxDeclaredTensor* fixed_storage) {
    const VxDeclaredTensor* logical = vx_native_gpu_node_input_named(
        node, role, tensors, tensor_count);
    const cJSON* inputs;
    const cJSON* reference;
    if (logical) return logical;
    if (!validation || !fixed_storage) return NULL;
    inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    reference = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, role) : NULL;
    if (!cJSON_IsString(reference) || !reference->valuestring) return NULL;
    for (int index = 0; index < validation->tensor_count; index++) {
        const T* tensor = &validation->tensors[index];
        if (strcmp(tensor->name, reference->valuestring)) continue;
        if (tensor->ndim <= 0 || tensor->ndim > (int)VX_MAX_TENSOR_RANK ||
            tensor->numel <= 0 || !tensor->elem_size) return NULL;
        memset(fixed_storage, 0, sizeof(*fixed_storage));
        fixed_storage->name = (char*)tensor->name;
        fixed_storage->dtype = tensor->dtype;
        fixed_storage->rank = (uint32_t)tensor->ndim;
        for (uint32_t axis = 0; axis < fixed_storage->rank; axis++) {
            if (tensor->shape[axis] <= 0) return NULL;
            fixed_storage->kinds[axis] = VX_DIMENSION_FIXED;
            fixed_storage->minimums[axis] = tensor->shape[axis];
            fixed_storage->maximums[axis] = tensor->shape[axis];
            fixed_storage->multiples[axis] = 1;
        }
        if ((uint64_t)tensor->numel > SIZE_MAX / tensor->elem_size)
            return NULL;
        fixed_storage->maximum_byte_size =
            (size_t)tensor->numel * tensor->elem_size;
        return fixed_storage;
    }
    return NULL;
}

static const T* vx_native_gpu_validation_tensor_find(
        const VxEngineState* validation,
        const char* name) {
    if (!validation || !name) return NULL;
    for (int index = 0; index < validation->tensor_count; index++)
        if (!strcmp(validation->tensors[index].name, name))
            return &validation->tensors[index];
    return NULL;
}

static const T* vx_native_gpu_validation_node_input(
        const VxEngineState* validation,
        const Node* node,
        const char* role) {
    if (!validation || !node || !role) return NULL;
    for (int index = 0; index < node->nin; index++)
        if (!strcmp(node->ins[index].key, role))
            return vx_native_gpu_validation_tensor_find(
                validation, node->ins[index].name);
    return NULL;
}

static int vx_native_gpu_accumulate_resource_bytes(uint64_t* total,
                                             uint64_t bytes,
                                             uint64_t copies) {
    if (!total || !copies || bytes > (UINT64_MAX - *total) / copies)
        return 0;
    *total += bytes * copies;
    return 1;
}

static int vx_native_gpu_align_resource_bytes(uint64_t bytes,
                                               uint64_t alignment,
                                               uint64_t* aligned_out) {
    uint64_t remainder;
    if (!bytes || !alignment || !aligned_out) return 0;
    remainder = bytes % alignment;
    if (remainder && bytes > UINT64_MAX - (alignment - remainder)) return 0;
    *aligned_out = remainder ? bytes + (alignment - remainder) : bytes;
    return 1;
}

/* Public built-in results are published one output at a time.  For output i,
 * all preceding snapshots, the temporary readback buffer, and the newly
 * appended owned snapshot coexist until the temporary is released.  This is
 * intentionally an ordered peak rather than 2 * sum(outputs): only the
 * current output is duplicated. */
static int vx_native_gpu_result_publication_peak(
        const VxModel* model,
        uint64_t* snapshot_bytes_out,
        uint64_t* publication_peak_out) {
    uint64_t published = 0u;
    uint64_t peak = 0u;
    if (!model || !snapshot_bytes_out || !publication_peak_out)
        return 0;
    for (size_t index = 0; index < model->output_count; index++) {
        uint64_t bytes = (uint64_t)model->outputs[index].maximum_byte_size;
        uint64_t candidate;
        if (bytes > (UINT64_MAX - published) / 2u)
            return 0;
        candidate = published + bytes * 2u;
        if (candidate > peak) peak = candidate;
        if (bytes > UINT64_MAX - published) return 0;
        published += bytes;
    }
    *snapshot_bytes_out = published;
    *publication_peak_out = peak;
    return 1;
}

static int vx_native_gpu_maximum_signature_bytes(
        const VxModel* model, uint64_t* bytes_out) {
    uint64_t bytes = 4u;
    if (!model || !bytes_out) return 0;
    for (size_t index = 0; index < model->input_count; index++) {
        uint64_t name_bytes = (uint64_t)strlen(model->inputs[index].name);
        uint64_t axes_bytes = (uint64_t)model->inputs[index].rank * 24u;
        if (name_bytes > UINT64_MAX - 4u - axes_bytes ||
            bytes > UINT64_MAX - name_bytes - 4u - axes_bytes)
            return 0;
        bytes += name_bytes + 4u + axes_bytes;
    }
    *bytes_out = bytes;
    return 1;
}

/* Bound every heap-owned maximum-layout/plan allocation in a dynamic native
 * GPU context.  A cold bind can hold a full four-entry LRU while constructing
 * its fifth candidate, together with the resolved request/signature and the
 * larger of plan-build or bind-publication temporaries.  Maximum-layout build
 * has a separate phase; take the larger phase instead of summing allocations
 * that cannot coexist. */
static int vx_native_gpu_dynamic_metadata_peak(
        const VxModel* model,
        uint64_t logical_tensor_count,
        uint64_t engine_tensor_count,
        uint64_t physical_span_count,
        uint64_t* peak_out) {
    uint64_t signature_bytes;
    uint64_t plan_bytes = 0u;
    uint64_t maximum_layout_bytes = 0u;
    uint64_t layout_build_temporary = 0u;
    uint64_t layout_build_peak;
    uint64_t resolved_request_bytes = 0u;
    uint64_t plan_build_temporary = 0u;
    uint64_t bind_temporary = sizeof(void*);
    uint64_t transaction_temporary;
    uint64_t plan_transaction_peak = 0u;
    if (!model || !logical_tensor_count || !peak_out ||
        !vx_native_gpu_maximum_signature_bytes(model, &signature_bytes))
        return 0;

    if (!vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count, 9u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count,
            4u * sizeof(size_t)))
        return 0;

    /* The public maximum descriptors remain live while the private maximum
     * layout builder owns starts, ends, aliases, buffer_of, buffer_until and
     * its two size_t arrays. descriptor/producers are indexed by the complete
     * private T table. */
    if (!vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, logical_tensor_count,
            sizeof(VolvoxAIEngineMaximumTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, logical_tensor_count,
            5u * sizeof(int) + 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_build_temporary, engine_tensor_count,
            3u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, logical_tensor_count,
            sizeof(VolvoxAIEngineResolvedTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &resolved_request_bytes, (uint64_t)model->input_count,
            sizeof(VolvoxAIEngineInputBinding)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_build_temporary, engine_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, logical_tensor_count,
            sizeof(int) + sizeof(void*)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, physical_span_count,
            sizeof(VolvoxAIEnginePhysicalSpan)))
        return 0;
    if (maximum_layout_bytes > UINT64_MAX - layout_build_temporary)
        return 0;
    layout_build_peak = maximum_layout_bytes + layout_build_temporary;
    transaction_temporary = plan_build_temporary > bind_temporary
        ? plan_build_temporary : bind_temporary;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, maximum_layout_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, plan_bytes,
            VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY + 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, resolved_request_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_transaction_peak, transaction_temporary, 1u))
        return 0;
    *peak_out = layout_build_peak > plan_transaction_peak
        ? layout_build_peak : plan_transaction_peak;
    return 1;
}

static int vx_native_gpu_resident_capacity_proven(
        uint64_t base_bytes,
        uint64_t result_publication_bytes,
        uint64_t dynamic_metadata_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* required_out) {
    uint64_t required = base_bytes;
    if (!required_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &required, result_publication_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &required, dynamic_metadata_bytes, 1u))
        return 0;
    *required_out = required;
    return !hard_limit_bytes || required <= hard_limit_bytes;
}

static int vx_native_gpu_accumulate_immutable_allocation(
        uint64_t host_bytes,
        uint64_t device_bytes,
        uint64_t device_alignment,
        uint64_t* resident_total,
        uint64_t* device_total) {
    uint64_t aligned_device;
    if (!host_bytes || !device_bytes || !resident_total || !device_total ||
        !vx_native_gpu_align_resource_bytes(
            device_bytes, device_alignment, &aligned_device) ||
        !vx_native_gpu_accumulate_resource_bytes(
            resident_total, host_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            resident_total, aligned_device, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            device_total, aligned_device, 1u))
        return 0;
    return 1;
}

static int vx_native_gpu_maskless_qsdpa_immutable_bound(
        VolvoxAIEngineBackend backend,
        int has_mask,
        uint64_t device_alignment,
        uint64_t* resident_total,
        uint64_t* device_total) {
    /* Vulkan/OpenGL/Metal bind a shared static I32 zero when QSDPA has no
     * graph mask. It is not a logical/validation tensor, but graph_ensure_*()
     * retains it as an immutable slot. CUDA passes a null device pointer and
     * therefore owns no corresponding allocation. */
    if (has_mask || backend == VOLVOXAI_BACKEND_CUDA) return 1;
    return vx_native_gpu_accumulate_immutable_allocation(
        sizeof(int32_t), sizeof(int32_t), device_alignment,
        resident_total, device_total);
}

typedef struct {
    void* values;
    size_t elements;
    void* next;
} VxNativeGpuZeroBiasBackingProof;

static int vx_native_gpu_accumulate_zero_bias_backing(
        uint64_t* resident_total) {
    return vx_native_gpu_accumulate_resource_bytes(
        resident_total, sizeof(VxNativeGpuZeroBiasBackingProof), 1u);
}

/* A native GPU may materialize an immutable input into a node-local representation
 * (for example an F16-to-F32 or convolution-layout cache), and quantized
 * nodes own aligned affine metadata that is not itself represented by T.
 * Charge one host representation and one alignment-rounded device allocation
 * per consumer. This is deliberately conservative when consumers share a
 * cache key, but it remains sound for Vulkan's suballocated hard arena. Byte
 * tensors occupy complete u32 words on Vulkan/OpenGL/Metal, so their device
 * allocation is rounded independently from the exact host representation. */
static int vx_native_gpu_validation_immutable_bytes_bound(
        const VxEngineState* validation,
        const VxDeclaredTensor* logical_tensors,
        size_t logical_tensor_count,
        VolvoxAIEngineBackend backend,
        uint64_t device_alignment,
        uint64_t* bytes_out,
        uint64_t* device_bytes_out) {
    uint64_t total = 0u;
    uint64_t device_total = 0u;
    if (!validation || !logical_tensors || !device_alignment || !bytes_out ||
        !device_bytes_out ||
        validation->node_count < 0 || validation->tensor_count < 0)
        return 0;
    for (int node_index = 0; node_index < validation->node_count;
         node_index++) {
        const Node* node = &validation->nodes[node_index];
        for (int input_index = 0; input_index < node->nin; input_index++) {
            const T* tensor = vx_native_gpu_validation_tensor_find(
                validation, node->ins[input_index].name);
            uint64_t element_size;
            uint64_t bytes;
            uint64_t device_bytes;
            if (!tensor || tensor->numel <= 0 || !tensor->elem_size)
                return 0;
            if (vx_declared_tensor_find(
                    logical_tensors, logical_tensor_count, tensor->name))
                continue;
            element_size = tensor->dtype == T_F16
                ? (uint64_t)sizeof(float) : (uint64_t)tensor->elem_size;
            if ((uint64_t)tensor->numel > UINT64_MAX / element_size)
                return 0;
            bytes = (uint64_t)tensor->numel * element_size;
            device_bytes = bytes;
            if (tensor->dtype == T_I8 || tensor->dtype == T_U8) {
                if (device_bytes > UINT64_MAX - 3u) return 0;
                device_bytes = (device_bytes + 3u) & ~UINT64_C(3);
            }
            if (!vx_native_gpu_accumulate_immutable_allocation(
                    bytes, device_bytes, device_alignment,
                    &total, &device_total))
                return 0;
        }
        if ((!strcmp(node->op, "Linear") || !strcmp(node->op, "MatMul") ||
             !strcmp(node->op, "Gemm")) &&
            backend != VOLVOXAI_BACKEND_CUDA) {
            const T* output = vx_native_gpu_validation_tensor_find(
                validation, node->out);
            const T* bias = vx_native_gpu_validation_node_input(
                validation, node, "bias");
            uint64_t aligned_dummy;
            /* Vulkan/OpenGL retain these graph bindings. Metal creates the
             * equivalent buffers per dispatch. Charging one independently
             * aligned allocation per node is conservative for both lifetime
             * models and covers output-major's otherwise hidden scale slot. */
            if (!output || output->ndim <= 0 ||
                output->shape[output->ndim - 1] <= 0 ||
                (!bias &&
                 !vx_native_gpu_accumulate_immutable_allocation(
                     (uint64_t)output->shape[output->ndim - 1] *
                         sizeof(float),
                     (uint64_t)output->shape[output->ndim - 1] *
                         sizeof(float),
                     device_alignment, &total, &device_total)) ||
                !vx_native_gpu_align_resource_bytes(
                    sizeof(float), device_alignment, &aligned_dummy) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &total, aligned_dummy, 1u) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &device_total, aligned_dummy, 1u))
                return 0;
            if (!bias &&
                (backend == VOLVOXAI_BACKEND_VULKAN ||
                 backend == VOLVOXAI_BACKEND_OPENGL) &&
                !vx_native_gpu_accumulate_zero_bias_backing(&total))
                return 0;
        }
        if ((!strcmp(node->op, "QLinear") ||
             !strcmp(node->op, "QMatMul") ||
             !strcmp(node->op, "QGemm")) &&
            validation->qlinear_metadata[node_index].valid) {
            const QLinearMetadata* metadata =
                &validation->qlinear_metadata[node_index];
            uint64_t component_bytes = (uint64_t)metadata->d_out *
                sizeof(uint32_t);
            if (metadata->d_out <= 0 ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                (metadata->packed_weight &&
                 !vx_native_gpu_accumulate_resource_bytes(
                    &total, metadata->packed_weight_bytes, 1u)))
                return 0;
        } else if (!strcmp(node->op, "QConv2D") &&
                   validation->qconv_metadata[node_index].valid) {
            const QConv2DMetadata* metadata =
                &validation->qconv_metadata[node_index];
            uint64_t component_bytes =
                (uint64_t)metadata->output_channels *
                sizeof(uint32_t);
            uint64_t small_c_bytes = 0u;
            if (metadata->small_c_packed_weight) {
                small_c_bytes = (uint64_t)metadata->kernel_height;
                if (!metadata->kernel_width ||
                    small_c_bytes > UINT64_MAX / metadata->kernel_width)
                    return 0;
                small_c_bytes *= metadata->kernel_width;
                if (!metadata->input_per_group ||
                    small_c_bytes > UINT64_MAX / metadata->input_per_group)
                    return 0;
                small_c_bytes *= metadata->input_per_group;
                if (!metadata->output_channels ||
                    small_c_bytes > UINT64_MAX / metadata->output_channels)
                    return 0;
                small_c_bytes *= metadata->output_channels;
            }
            if (!metadata->output_channels ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                (metadata->packed_weight &&
                 !vx_native_gpu_accumulate_resource_bytes(
                    &total, metadata->packed_weight_bytes, 1u)) ||
                (small_c_bytes && !vx_native_gpu_accumulate_resource_bytes(
                    &total, small_c_bytes, 1u)))
                return 0;
            if (!metadata->bias && backend != VOLVOXAI_BACKEND_CUDA &&
                !vx_native_gpu_accumulate_zero_bias_backing(&total))
                return 0;
        } else if (!strcmp(node->op, "QEmbedding") &&
                   validation->qembedding_metadata[node_index].valid) {
            const QEmbeddingMetadata* metadata =
                &validation->qembedding_metadata[node_index];
            uint64_t component_bytes = (uint64_t)metadata->vocab *
                sizeof(uint32_t);
            if (!metadata->vocab ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total) ||
                !vx_native_gpu_accumulate_immutable_allocation(
                    component_bytes, component_bytes, device_alignment,
                    &total, &device_total))
                return 0;
        }
        if (!strcmp(node->op, "QSDPA") &&
            !vx_native_gpu_maskless_qsdpa_immutable_bound(
                backend,
                vx_native_gpu_validation_node_input(
                    validation, node, "mask") != NULL,
                device_alignment, &total, &device_total))
            return 0;
        /* CUDA retains each QSiLU/QGELU descriptor's 256-byte lookup table as
         * an immutable device slot through invariant preload/domain
         * publication. Charging one table per node is conservative when two
         * nodes share the cache key and exact when their descriptors differ. */
        if (backend == VOLVOXAI_BACKEND_CUDA &&
            (!strcmp(node->op, "QSiLU") || !strcmp(node->op, "QGELU")) &&
            !vx_native_gpu_accumulate_immutable_allocation(
                256u, 256u, device_alignment, &total, &device_total))
            return 0;
    }
    *bytes_out = total;
    *device_bytes_out = device_total;
    return 1;
}

static int vx_native_gpu_validation_bootstrap_activation_bound(
        const VxEngineState* validation,
        const VxDeclaredTensor* logical_tensors,
        size_t logical_tensor_count,
        uint64_t device_alignment,
        uint64_t* host_bytes_out,
        uint64_t* device_bytes_out) {
    uint64_t host_total = 0u;
    uint64_t device_total = 0u;
    if (!validation || !logical_tensors || !logical_tensor_count ||
        !device_alignment || !host_bytes_out || !device_bytes_out)
        return 0;
    for (size_t index = 0; index < logical_tensor_count; index++) {
        const T* tensor = vx_native_gpu_validation_tensor_find(
            validation, logical_tensors[index].name);
        uint64_t host_bytes;
        uint64_t device_bytes;
        uint64_t aligned_device;
        if (!tensor || !tensor->data || tensor->numel <= 0 ||
            !tensor->elem_size ||
            (uint64_t)tensor->numel > UINT64_MAX / tensor->elem_size)
            return 0;
        host_bytes = (uint64_t)tensor->numel * tensor->elem_size;
        device_bytes = host_bytes;
        if (tensor->dtype == T_I8 || tensor->dtype == T_U8) {
            if (device_bytes > UINT64_MAX - 3u) return 0;
            device_bytes = (device_bytes + 3u) & ~UINT64_C(3);
        }
        if (!vx_native_gpu_align_resource_bytes(
                device_bytes, device_alignment, &aligned_device) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &host_total, host_bytes, 1u) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &device_total, aligned_device, 1u))
            return 0;
    }
    *host_bytes_out = host_total;
    *device_bytes_out = device_total;
    return 1;
}

static int vx_native_gpu_dispatch_conv2d_bias_bound(
        uint64_t output_channels,
        uint64_t device_alignment,
        uint64_t* total) {
    uint64_t bias_bytes;
    uint64_t aligned_bias_bytes;
    if (!output_channels || !device_alignment || !total ||
        output_channels > UINT64_MAX / sizeof(float))
        return 0;
    bias_bytes = output_channels * sizeof(float);
    return vx_native_gpu_align_resource_bytes(
               bias_bytes, device_alignment, &aligned_bias_bytes) &&
        vx_native_gpu_accumulate_resource_bytes(total, bias_bytes, 1u) &&
        vx_native_gpu_accumulate_resource_bytes(
            total, aligned_bias_bytes, 1u);
}

static int vx_native_gpu_dispatch_qlinear_host_bound(
        uint64_t output_channels, uint64_t* total) {
    if (!output_channels || !total ||
        output_channels > UINT64_MAX / sizeof(float))
        return 0;
    return vx_native_gpu_accumulate_resource_bytes(
        total, output_channels * sizeof(float), 1u);
}

static int vx_native_gpu_dispatch_opengl_packed_clear_bound(
        uint64_t output_bytes, uint64_t* total) {
    uint64_t packed_bytes;
    if (!output_bytes || !total || output_bytes > UINT64_MAX - 3u)
        return 0;
    packed_bytes = (output_bytes + 3u) & ~UINT64_C(3);
    return vx_native_gpu_accumulate_resource_bytes(
        total, packed_bytes, 1u);
}

static int vx_native_gpu_validation_dispatch_temporary_bound(
        const VxEngineState* validation,
        const VxDeclaredTensor* logical_tensors,
        size_t logical_tensor_count,
        VolvoxAIEngineBackend backend,
        uint64_t device_alignment,
        uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!validation || !logical_tensors || !device_alignment || !bytes_out ||
        validation->node_count < 0)
        return 0;
    if (backend == VOLVOXAI_BACKEND_CUDA) {
        *bytes_out = 0u;
        return 1;
    }
    for (int index = 0; index < validation->node_count; index++) {
        const Node* node = &validation->nodes[index];
        uint64_t dispatch_count = 1u;
        uint64_t aligned_bytes;
        /* Metal retains every bound buffer until its batched graph command
         * completes; OpenGL deletion is likewise deferred while commands
         * reference a buffer.  Qualified parameter blocks are at most 17
         * u32 words. Reserve one independently aligned 256-byte block for
         * each possible per-input/per-output dispatch of a node (Concat and
         * Split are the multi-dispatch cases). Quantize/Dequantize expose
         * their scale and zero-point ports, so max(nin,nout) also covers their
         * three scalar/parameter buffers. */
        if (backend != VOLVOXAI_BACKEND_VULKAN) {
            if (node->nin > 1) dispatch_count = (uint64_t)node->nin;
            if (node->nout > 1 && (uint64_t)node->nout > dispatch_count)
                dispatch_count = (uint64_t)node->nout;
            if (!vx_native_gpu_align_resource_bytes(
                    256u, device_alignment, &aligned_bytes) ||
                !vx_native_gpu_accumulate_resource_bytes(
                    &total, aligned_bytes, dispatch_count))
                return 0;
        }

        /* QLinear/QMatMul/QGemm build d_out F32 multipliers in a host malloc
         * on Vulkan/OpenGL/Metal. Their device copy is already covered by the
         * immutable-component/scratch proof, but the simultaneously live host
         * array is an additional resident transient. */
        if ((!strcmp(node->op, "QLinear") ||
             !strcmp(node->op, "QMatMul") ||
             !strcmp(node->op, "QGemm")) &&
            validation->qlinear_metadata[index].valid &&
            !vx_native_gpu_dispatch_qlinear_host_bound(
                (uint64_t)validation->qlinear_metadata[index].d_out,
                &total))
            return 0;

        /* Of the currently exporter-qualified native-GPU operators, Conv2D
         * has one additional shape-sized dispatch allocation: when bias is
         * absent, OpenGL/Metal create a zero-filled out_c F32 host array and a
         * device buffer before releasing either. Conv1D and MoE have similar
         * private paths but are deliberately outside the qualified set; they
         * must extend this proof before their registry qualification changes. */
        if (backend != VOLVOXAI_BACKEND_VULKAN &&
            !strcmp(node->op, "Conv2D") &&
            !vx_native_gpu_validation_node_input(
                validation, node, "bias")) {
            const char* output_name = node->out[0]
                ? node->out
                : node->nout > 0 ? node->outs[0].name : NULL;
            const VxDeclaredTensor* output = vx_declared_tensor_find(
                logical_tensors, logical_tensor_count, output_name);
            if (!output || !output->rank ||
                output->maximums[output->rank - 1u] <= 0 ||
                !vx_native_gpu_dispatch_conv2d_bias_bound(
                    (uint64_t)output->maximums[output->rank - 1u],
                    device_alignment, &total))
                return 0;
        }

        /* OpenGL initializes a qualified byte Concat output through a host
         * zero array rounded to its packed-u32 allocation before uploading it.
         * Metal clears its shared device buffer in place and Vulkan uses its
         * fixed scratch/device arena, so only OpenGL owns this host transient. */
        if (backend == VOLVOXAI_BACKEND_OPENGL &&
            !strcmp(node->op, "Concat")) {
            const char* output_name = node->out[0]
                ? node->out
                : node->nout > 0 ? node->outs[0].name : NULL;
            const VxDeclaredTensor* output = vx_declared_tensor_find(
                logical_tensors, logical_tensor_count, output_name);
            if (!output ||
                ((output->dtype == VX_DTYPE_I8 ||
                  output->dtype == VX_DTYPE_U8) &&
                 !vx_native_gpu_dispatch_opengl_packed_clear_bound(
                     (uint64_t)output->maximum_byte_size, &total)))
                return 0;
        }
    }
    *bytes_out = total;
    return 1;
}

/* Context creation retains the minimum-shape host tensors until the first
 * exact binding publishes the fixed maximum host arena. OpenGL and Metal also
 * retain every bootstrap device buffer until all candidate domain and stats
 * buffers have been allocated transactionally. Vulkan reuses ranges in one
 * fixed arena, and CUDA explicitly releases bootstrap transients first, so
 * those backends need the larger phase rather than their sum. */
static int vx_native_gpu_activation_resource_peak(
        VolvoxAIEngineBackend backend,
        uint64_t bootstrap_host_bytes,
        uint64_t maximum_host_arena_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t dispatch_temporary_bytes,
        uint64_t vulkan_reserved_scratch_bytes,
        uint64_t* peak_out) {
    uint64_t peak = 0u;
    uint64_t device_phase;
    uint64_t qnorm_bytes;
    if (!peak_out || !vx_native_gpu_backend(backend) ||
        qgroupnorm_stats_bytes > UINT64_MAX - qlayernorm_stats_bytes)
        return 0;
    qnorm_bytes = qgroupnorm_stats_bytes + qlayernorm_stats_bytes;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &peak, bootstrap_host_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &peak, maximum_host_arena_bytes, 1u))
        return 0;
    if (backend == VOLVOXAI_BACKEND_OPENGL ||
        backend == VOLVOXAI_BACKEND_METAL) {
        if (bootstrap_device_bytes > UINT64_MAX - candidate_device_bytes)
            return 0;
        device_phase = bootstrap_device_bytes + candidate_device_bytes;
        if (!vx_native_gpu_accumulate_resource_bytes(
                &peak, qnorm_bytes, 2u))
            return 0;
    } else {
        device_phase = bootstrap_device_bytes > candidate_device_bytes
            ? bootstrap_device_bytes : candidate_device_bytes;
        if (backend == VOLVOXAI_BACKEND_VULKAN &&
            !vx_native_gpu_accumulate_resource_bytes(
                &peak, vulkan_reserved_scratch_bytes, 1u))
            return 0;
    }
    if (!vx_native_gpu_accumulate_resource_bytes(&peak, device_phase, 1u))
        return 0;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &peak, dispatch_temporary_bytes, 1u))
        return 0;
    *peak_out = peak;
    return 1;
}

static int vx_native_gpu_device_capacity_proven(
        uint64_t immutable_device_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* peak_out) {
    uint64_t transient_phase = bootstrap_device_bytes > candidate_device_bytes
        ? bootstrap_device_bytes : candidate_device_bytes;
    uint64_t peak;
    if (!peak_out || immutable_device_bytes > UINT64_MAX - transient_phase)
        return 0;
    peak = immutable_device_bytes + transient_phase;
    *peak_out = peak;
    return !hard_limit_bytes || peak <= hard_limit_bytes;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_align_resource_bytes(
        uint64_t bytes, uint64_t alignment, uint64_t* aligned_out) {
    return vx_native_gpu_align_resource_bytes(bytes, alignment, aligned_out);
}

int vx_public_api_test_native_gpu_immutable_components(
        uint64_t component_bytes,
        uint64_t component_count,
        uint64_t device_alignment,
        uint64_t* resident_out,
        uint64_t* device_out) {
    uint64_t resident = 0u;
    uint64_t device = 0u;
    if (!component_count || !resident_out || !device_out) return 0;
    for (uint64_t index = 0; index < component_count; index++)
        if (!vx_native_gpu_accumulate_immutable_allocation(
                component_bytes, component_bytes, device_alignment,
                &resident, &device))
            return 0;
    *resident_out = resident;
    *device_out = device;
    return 1;
}

int vx_public_api_test_native_gpu_maskless_qsdpa_immutable_bound(
        VolvoxAIEngineBackend backend,
        int has_mask,
        uint64_t device_alignment,
        uint64_t* resident_out,
        uint64_t* device_out) {
    uint64_t resident = 0u;
    uint64_t device = 0u;
    if (!resident_out || !device_out ||
        !vx_native_gpu_maskless_qsdpa_immutable_bound(
            backend, has_mask, device_alignment, &resident, &device))
        return 0;
    *resident_out = resident;
    *device_out = device;
    return 1;
}

int vx_public_api_test_native_gpu_zero_bias_backing_bound(
        uint64_t count, uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!count || !bytes_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &total, sizeof(VxNativeGpuZeroBiasBackingProof), count))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_activation_resource_peak(
        VolvoxAIEngineBackend backend,
        uint64_t bootstrap_host_bytes,
        uint64_t maximum_host_arena_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t dispatch_temporary_bytes,
        uint64_t vulkan_reserved_scratch_bytes,
        uint64_t* peak_out) {
    return vx_native_gpu_activation_resource_peak(
        backend, bootstrap_host_bytes, maximum_host_arena_bytes,
        bootstrap_device_bytes, candidate_device_bytes,
        qgroupnorm_stats_bytes, qlayernorm_stats_bytes,
        dispatch_temporary_bytes, vulkan_reserved_scratch_bytes, peak_out);
}

int vx_public_api_test_native_gpu_conv2d_dispatch_temporary_bound(
        uint64_t output_channels,
        int has_bias,
        uint64_t parameter_blocks,
        uint64_t device_alignment,
        uint64_t* bytes_out) {
    uint64_t total = 0u;
    uint64_t aligned_parameter;
    if (!bytes_out || !parameter_blocks ||
        !vx_native_gpu_align_resource_bytes(
            256u, device_alignment, &aligned_parameter) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &total, aligned_parameter, parameter_blocks) ||
        (!has_bias && !vx_native_gpu_dispatch_conv2d_bias_bound(
            output_channels, device_alignment, &total)))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_qlinear_dispatch_host_bound(
        uint64_t output_channels, uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!bytes_out || !vx_native_gpu_dispatch_qlinear_host_bound(
            output_channels, &total))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_opengl_packed_clear_bound(
        uint64_t output_bytes, uint64_t* bytes_out) {
    uint64_t total = 0u;
    if (!bytes_out ||
        !vx_native_gpu_dispatch_opengl_packed_clear_bound(
            output_bytes, &total))
        return 0;
    *bytes_out = total;
    return 1;
}

int vx_public_api_test_native_gpu_vulkan_scratch_capacity_proven(
        uint64_t qgroupnorm_stats_bytes,
        uint64_t qlayernorm_stats_bytes,
        uint64_t validation_graph_scratch_bytes,
        uint64_t storage_alignment,
        uint64_t maximum_scratch_bytes,
        uint64_t* required_out) {
    return vx_native_gpu_vulkan_scratch_capacity_proven(
        qgroupnorm_stats_bytes, qlayernorm_stats_bytes,
        validation_graph_scratch_bytes, storage_alignment,
        maximum_scratch_bytes, required_out);
}

int vx_public_api_test_native_gpu_device_capacity_proven(
        uint64_t immutable_device_bytes,
        uint64_t bootstrap_device_bytes,
        uint64_t candidate_device_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* peak_out) {
    return vx_native_gpu_device_capacity_proven(
        immutable_device_bytes, bootstrap_device_bytes,
        candidate_device_bytes, hard_limit_bytes, peak_out);
}

int vx_public_api_test_native_gpu_slot_capacity_proven(
        uint64_t tensor_count,
        uint64_t node_count,
        uint64_t physical_span_count,
        uint64_t slot_limit,
        uint64_t* required_out) {
    return vx_native_gpu_slot_capacity_proven(
        tensor_count, node_count, physical_span_count,
        slot_limit, required_out);
}

int vx_public_api_test_native_gpu_result_publication_peak(
        const uint64_t* output_bytes,
        size_t output_count,
        uint64_t* snapshot_bytes_out,
        uint64_t* publication_peak_out) {
    uint64_t published = 0u;
    uint64_t peak = 0u;
    if ((!output_bytes && output_count) || !snapshot_bytes_out ||
        !publication_peak_out)
        return 0;
    for (size_t index = 0; index < output_count; index++) {
        uint64_t bytes = output_bytes[index];
        uint64_t candidate;
        if (bytes > (UINT64_MAX - published) / 2u) return 0;
        candidate = published + bytes * 2u;
        if (candidate > peak) peak = candidate;
        if (bytes > UINT64_MAX - published) return 0;
        published += bytes;
    }
    *snapshot_bytes_out = published;
    *publication_peak_out = peak;
    return 1;
}

int vx_public_api_test_native_gpu_dynamic_metadata_peak(
        uint64_t signature_bytes,
        uint64_t input_count,
        uint64_t logical_tensor_count,
        uint64_t engine_tensor_count,
        uint64_t physical_span_count,
        uint64_t* peak_out) {
    uint64_t plan_bytes = 0u;
    uint64_t maximum_layout_bytes = 0u;
    uint64_t layout_temporary = 0u;
    uint64_t layout_peak;
    uint64_t request_bytes = 0u;
    uint64_t build_temporary = 0u;
    uint64_t bind_temporary = sizeof(void*);
    uint64_t transaction_temporary;
    uint64_t transaction_peak = 0u;
    if (!signature_bytes || !logical_tensor_count || !peak_out ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &plan_bytes, logical_tensor_count,
            sizeof(int) + 2u * sizeof(size_t) + 9u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &maximum_layout_bytes, logical_tensor_count,
            sizeof(int) + 4u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_temporary, logical_tensor_count,
            sizeof(VolvoxAIEngineMaximumTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_temporary, logical_tensor_count,
            5u * sizeof(int) + 2u * sizeof(size_t)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &layout_temporary, engine_tensor_count, 3u * sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &request_bytes, logical_tensor_count,
            sizeof(VolvoxAIEngineResolvedTensor)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &request_bytes, signature_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &request_bytes, input_count,
            sizeof(VolvoxAIEngineInputBinding)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &build_temporary, engine_tensor_count, sizeof(int)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, logical_tensor_count,
            sizeof(int) + sizeof(void*)) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &bind_temporary, physical_span_count,
            sizeof(VolvoxAIEnginePhysicalSpan)))
        return 0;
    if (maximum_layout_bytes > UINT64_MAX - layout_temporary) return 0;
    layout_peak = maximum_layout_bytes + layout_temporary;
    transaction_temporary = build_temporary > bind_temporary
        ? build_temporary : bind_temporary;
    if (!vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, maximum_layout_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, plan_bytes,
            VX_DYNAMIC_SHAPE_PLAN_CACHE_CAPACITY + 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, request_bytes, 1u) ||
        !vx_native_gpu_accumulate_resource_bytes(
            &transaction_peak, transaction_temporary, 1u))
        return 0;
    *peak_out = layout_peak > transaction_peak
        ? layout_peak : transaction_peak;
    return 1;
}

int vx_public_api_test_native_gpu_resident_capacity_proven(
        uint64_t base_bytes,
        uint64_t result_publication_bytes,
        uint64_t dynamic_metadata_bytes,
        uint64_t hard_limit_bytes,
        uint64_t* required_out) {
    return vx_native_gpu_resident_capacity_proven(
        base_bytes, result_publication_bytes, dynamic_metadata_bytes,
        hard_limit_bytes, required_out);
}
#endif

static const VxDeclaredTensor* vx_native_gpu_node_data_domain(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        VxDeclaredTensor* fixed_storage) {
    static const char* const roles[] = {"input", "data", "x", "a"};
    for (size_t index = 0; index < sizeof(roles) / sizeof(roles[0]); index++) {
        const VxDeclaredTensor* tensor = vx_native_gpu_node_input_domain(
            node, roles[index], tensors, tensor_count, validation,
            fixed_storage);
        if (tensor) return tensor;
    }
    return NULL;
}

static const VxDeclaredTensor* vx_native_gpu_node_output(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count) {
    const cJSON* outputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "outputs") : NULL;
    const cJSON* port = cJSON_IsObject(outputs)
        ? cJSON_GetObjectItemCaseSensitive(outputs, "out") : NULL;
    const cJSON* reference;
    if (!port && cJSON_IsObject(outputs)) port = outputs->child;
    reference = cJSON_IsObject(port)
        ? cJSON_GetObjectItemCaseSensitive(port, "tensor") : NULL;
    return cJSON_IsString(reference) && reference->valuestring
        ? vx_declared_tensor_find(tensors, tensor_count, reference->valuestring)
        : NULL;
}

static int vx_native_gpu_tensor_maximum_elements(
        const VxDeclaredTensor* tensor,
        uint64_t* elements) {
    uint64_t product = 1u;
    if (!tensor || !elements || !tensor->rank ||
        tensor->rank > VX_MAX_TENSOR_RANK) return 0;
    for (uint32_t axis = 0; axis < tensor->rank; axis++) {
        uint64_t extent;
        if (tensor->maximums[axis] <= 0) return 0;
        extent = (uint64_t)tensor->maximums[axis];
        if (product > UINT64_MAX / extent) return 0;
        product *= extent;
    }
    *elements = product;
    return 1;
}

static int vx_native_gpu_tensor_maximum_product(
        const VxDeclaredTensor* tensor,
        uint32_t begin,
        uint32_t end,
        uint64_t limit,
        uint64_t* product_out) {
    uint64_t product = 1u;
    if (!tensor || begin > end || end > tensor->rank) return 0;
    for (uint32_t axis = begin; axis < end; axis++) {
        uint64_t extent;
        if (tensor->maximums[axis] <= 0) return 0;
        extent = (uint64_t)tensor->maximums[axis];
        if (product > limit / extent) return 0;
        product *= extent;
    }
    if (product_out) *product_out = product;
    return 1;
}

static int vx_native_gpu_integer_parameter(const cJSON* params,
                                     const char* name,
                                     int64_t default_value,
                                     int64_t minimum,
                                     int64_t maximum,
                                     int64_t* value_out) {
    const cJSON* value = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, name) : NULL;
    int64_t converted = default_value;
    if (value) {
        if (cJSON_IsBool(value)) {
            converted = cJSON_IsTrue(value) ? 1 : 0;
        } else {
            if (!cJSON_IsNumber(value) ||
                value->valuedouble < (double)INT_MIN ||
                value->valuedouble > (double)INT_MAX)
                return 0;
            converted = (int64_t)value->valuedouble;
            if ((double)converted != value->valuedouble) return 0;
        }
    }
    if (converted < minimum || converted > maximum) return 0;
    if (value_out) *value_out = converted;
    return 1;
}

static int vx_native_gpu_integer_array(const cJSON* value,
                                 int64_t* values,
                                 size_t capacity,
                                 size_t* count_out) {
    size_t count;
    size_t index = 0;
    if (!cJSON_IsArray(value) || !values || !count_out) return 0;
    count = (size_t)cJSON_GetArraySize(value);
    if (!count || count > capacity) return 0;
    for (const cJSON* item = value->child; item; item = item->next) {
        int64_t converted;
        if (!cJSON_IsNumber(item) ||
            item->valuedouble < (double)INT_MIN ||
            item->valuedouble > (double)INT_MAX)
            return 0;
        converted = (int64_t)item->valuedouble;
        if ((double)converted != item->valuedouble || index >= count) return 0;
        values[index++] = converted;
    }
    if (index != count) return 0;
    *count_out = count;
    return 1;
}

static int vx_native_gpu_tensor_shapes_equal(const VxDeclaredTensor* left,
                                       const VxDeclaredTensor* right) {
    if (!left || !right || left->rank != right->rank) return 0;
    for (uint32_t axis = 0; axis < left->rank; axis++)
        if (!vx_declared_dimensions_provably_equal(
                left, axis, right, axis)) return 0;
    return 1;
}

static int vx_native_gpu_tensor_domains_equal(const VxDeclaredTensor* left,
                                        const VxDeclaredTensor* right) {
    return left && right && left->dtype == right->dtype &&
        vx_native_gpu_tensor_shapes_equal(left, right);
}

static int vx_native_gpu_one_d_launch_proven(
        uint64_t elements,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits);

static int vx_native_gpu_dense_launch_proven(
        uint64_t rows,
        uint64_t input_width,
        uint64_t output_width,
        int quantized,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t output_elements;
    if (!rows || !input_width || !output_width || !limits ||
        rows > UINT32_MAX / output_width)
        return 0;
    output_elements = rows * output_width;
    if (backend == VOLVOXAI_BACKEND_CUDA) {
        if (quantized)
            return vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits);
        /* cuda_matmul_impl() has one fixed 16x16 launch. CUDA additionally
         * rejects grid-y above 65535 even when the physical device reports a
         * wider architectural grid domain. */
        return limits->maximum_block[0] >= 16u &&
            limits->maximum_block[1] >= 16u &&
            limits->maximum_threads_per_block >= 256u &&
            (output_width + 15u) / 16u <= limits->maximum_grid[0] &&
            (rows + 15u) / 16u <= limits->maximum_grid[1] &&
            (rows + 15u) / 16u <= 65535u;
    }
    if (!quantized) {
        uint64_t scalar_grid_x = (output_width + 63u) / 64u;
        /* Every dynamic row domain can select the scalar M=1 route. */
        if (limits->maximum_block[0] < 64u ||
            limits->maximum_threads_per_block < 64u ||
            scalar_grid_x > limits->maximum_grid[0] ||
            rows > limits->maximum_grid[1])
            return 0;
        /* Multi-row matrices use an 8x8 workgroup that emits a 16x16 tile. */
        if (rows > 1u && input_width >= 16u && output_width >= 16u &&
            (limits->maximum_block[0] < 8u ||
             limits->maximum_block[1] < 8u ||
             limits->maximum_threads_per_block < 64u ||
             (output_width + 15u) / 16u > limits->maximum_grid[0] ||
             (rows + 15u) / 16u > limits->maximum_grid[1]))
            return 0;
    } else {
        int tiled = input_width >= 16u && output_width >= 32u &&
            (output_width & 3u) == 0u;
        if (!vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return 0;
        if (tiled &&
            (limits->maximum_block[0] < 8u ||
             limits->maximum_block[1] < 8u ||
             limits->maximum_threads_per_block < 64u ||
             (output_width + 31u) / 32u > limits->maximum_grid[0] ||
             (rows + 7u) / 8u > limits->maximum_grid[1]))
            return 0;
    }
    return 1;
}

static int vx_native_gpu_dense_dynamic_rows_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        int quantized,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    uint64_t rows;
    uint64_t input_width;
    uint64_t output_width;
    if (!input || !output || !limits || !input->rank ||
        input->rank != output->rank ||
        vx_dimension_domain_is_dynamic(input, input->rank - 1u) ||
        vx_dimension_domain_is_dynamic(output, output->rank - 1u))
        return 0;
    for (uint32_t axis = 0; axis + 1u < input->rank; axis++)
        if (!vx_declared_dimensions_provably_equal(
                input, axis, output, axis)) return 0;
    /* seq_range and the ordinary one-shot routes carry rows as signed int. */
    if (!vx_native_gpu_tensor_maximum_product(
            input, 0u, input->rank - 1u, INT_MAX, &rows)) return 0;
    input_width = (uint64_t)input->maximums[input->rank - 1u];
    output_width = (uint64_t)output->maximums[output->rank - 1u];
    return vx_native_gpu_dense_launch_proven(
        rows, input_width, output_width, quantized, backend, limits);
}

static int vx_native_gpu_linear_storage_domain_proven(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    VxDeclaredTensor weight_fixed = {0};
    VxDeclaredTensor bias_fixed = {0};
    const VxDeclaredTensor* weight;
    const VxDeclaredTensor* bias = NULL;
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* bias_ref = cJSON_IsObject(inputs)
        ? cJSON_GetObjectItemCaseSensitive(inputs, "bias") : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    const cJSON* layout = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout") : NULL;
    int64_t trans_b = 0;
    int64_t weight_rows;
    int64_t weight_columns;
    int64_t d_in;
    int64_t d_out;
    int output_major = -1;
    uint64_t bias_elements;
    if (!node || !validation || !input || !output || !input->rank ||
        !output->rank ||
        vx_dimension_domain_is_dynamic(input, input->rank - 1u) ||
        vx_dimension_domain_is_dynamic(output, output->rank - 1u))
        return 0;
    d_in = input->maximums[input->rank - 1u];
    d_out = output->maximums[output->rank - 1u];
    weight = vx_native_gpu_node_input_domain(
        node, "weight", tensors, tensor_count, validation, &weight_fixed);
    if (!weight || weight->dtype != VX_DTYPE_F32 || weight->rank != 2u ||
        vx_declared_tensor_find(
            tensors, tensor_count, weight->name) != NULL ||
        !vx_declared_dimension_fixed_value(weight, 0u, &weight_rows) ||
        !vx_declared_dimension_fixed_value(weight, 1u, &weight_columns))
        return 0;
    if (layout) {
        if (!cJSON_IsString(layout) || !layout->valuestring) return 0;
        if (!strcmp(layout->valuestring, "dout_din")) output_major = 1;
        else if (!strcmp(layout->valuestring, "din_dout")) output_major = 0;
        else if (layout->valuestring[0]) return 0;
    }
    if (output_major < 0) {
        if (!vx_native_gpu_integer_parameter(
                params, "transB", 0, 0, 1, &trans_b))
            return 0;
        if (trans_b) output_major = 1;
    }
    if (output_major < 0) {
        int canonical = weight_rows == d_in && weight_columns == d_out;
        int transposed = weight_rows == d_out && weight_columns == d_in;
        output_major = transposed && !canonical ? 1 : canonical ? 0 : -1;
    }
    if (output_major < 0 ||
        (!output_major &&
         (weight_rows != d_in || weight_columns != d_out)) ||
        (output_major &&
         (weight_rows != d_out || weight_columns != d_in)))
        return 0;
    if (bias_ref) {
        if (!cJSON_IsString(bias_ref) || !bias_ref->valuestring) return 0;
        bias = vx_native_gpu_node_input_domain(
            node, "bias", tensors, tensor_count, validation, &bias_fixed);
        if (!bias || bias->dtype != VX_DTYPE_F32 ||
            vx_declared_tensor_find(tensors, tensor_count, bias->name) != NULL ||
            !vx_native_gpu_tensor_maximum_elements(bias, &bias_elements) ||
            bias_elements != (uint64_t)d_out)
            return 0;
        for (uint32_t axis = 0; axis < bias->rank; axis++)
            if (vx_dimension_domain_is_dynamic(bias, axis)) return 0;
    }
    return 1;
}

static int vx_native_gpu_centered_byte_distance(
        VxDataType dtype, int32_t zero_point, uint64_t* distance_out) {
    int64_t zero = zero_point;
    if (!distance_out) return 0;
    if (dtype == VX_DTYPE_I8) {
        if (zero < -128 || zero > 127) return 0;
        *distance_out = (uint64_t)(zero < 0 ? 127 - zero : zero + 128);
        return 1;
    }
    if (dtype == VX_DTYPE_U8) {
        if (zero < 0 || zero > 255) return 0;
        *distance_out = (uint64_t)(zero < 128 ? 255 - zero : zero);
        return 1;
    }
    return 0;
}

static int vx_native_gpu_quantized_accumulator_channel_proven(
        uint64_t terms,
        VxDataType input_dtype,
        int32_t input_zero_point,
        VxDataType weight_dtype,
        int32_t weight_zero_point,
        const int32_t* bias) {
    uint64_t input_distance;
    uint64_t weight_distance;
    uint64_t bias_distance = 0u;
    uint64_t product;
    if (!terms ||
        !vx_native_gpu_centered_byte_distance(
            input_dtype, input_zero_point, &input_distance) ||
        !vx_native_gpu_centered_byte_distance(
            weight_dtype, weight_zero_point, &weight_distance) ||
        terms > UINT64_MAX / input_distance ||
        terms * input_distance > UINT64_MAX / weight_distance)
        return 0;
    if (bias) {
        int64_t value = *bias;
        bias_distance = (uint64_t)(value < 0 ? -value : value);
    }
    product = terms * input_distance * weight_distance;
    return product <= INT32_MAX &&
        bias_distance <= (uint64_t)INT32_MAX - product;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_quantized_accumulator_channel_proven(
        uint64_t terms,
        VxDataType input_dtype,
        int32_t input_zero_point,
        VxDataType weight_dtype,
        int32_t weight_zero_point,
        const int32_t* bias) {
    return vx_native_gpu_quantized_accumulator_channel_proven(
        terms, input_dtype, input_zero_point, weight_dtype,
        weight_zero_point, bias);
}
#endif

static int vx_native_gpu_qlinear_accumulator_domain_proven(
        const VxEngineState* validation,
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    const Node* runtime_node = NULL;
    const T* runtime_input = NULL;
    const QLinearMetadata* metadata = NULL;
    int64_t input_distance;
    if (!validation || !input || !output || !input->rank || !output->rank ||
        !output->name) return 0;
    for (int index = 0; index < validation->node_count; index++) {
        const Node* candidate = &validation->nodes[index];
        if ((!strcmp(candidate->op, "QLinear") ||
             !strcmp(candidate->op, "QMatMul") ||
             !strcmp(candidate->op, "QGemm")) &&
            !strcmp(candidate->out, output->name)) {
            if (runtime_node) return 0;
            runtime_node = candidate;
            metadata = &validation->qlinear_metadata[index];
        }
    }
    if (!runtime_node || !metadata || !metadata->valid ||
        !metadata->weight_zero_points || !metadata->bias ||
        metadata->d_in <= 0 || metadata->d_out <= 0 ||
        input->maximums[input->rank - 1u] != metadata->d_in ||
        output->maximums[output->rank - 1u] != metadata->d_out)
        return 0;
    runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "input");
    if (!runtime_input) runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "x");
    if (!runtime_input) runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "a");
    if (!runtime_input || !runtime_input->quantization.valid ||
        (metadata->input_dtype != VX_DTYPE_I8 &&
         metadata->input_dtype != VX_DTYPE_U8) ||
        (metadata->weight_dtype != VX_DTYPE_I8 &&
         metadata->weight_dtype != VX_DTYPE_U8))
        return 0;
    if ((metadata->input_dtype == VX_DTYPE_I8 &&
         (runtime_input->quantization.zero_point < -128 ||
          runtime_input->quantization.zero_point > 127)) ||
        (metadata->input_dtype == VX_DTYPE_U8 &&
         (runtime_input->quantization.zero_point < 0 ||
          runtime_input->quantization.zero_point > 255)))
        return 0;
    if (metadata->input_dtype == VX_DTYPE_I8) {
        int64_t zero = runtime_input->quantization.zero_point;
        input_distance = zero < 0 ? 127 - zero : zero + 128;
    } else {
        int64_t zero = runtime_input->quantization.zero_point;
        input_distance = zero < 128 ? 255 - zero : zero;
    }
    for (int channel = 0; channel < metadata->d_out; channel++) {
        int64_t zero = metadata->weight_zero_points[channel];
        if ((metadata->weight_dtype == VX_DTYPE_I8 &&
             (zero < -128 || zero > 127)) ||
            (metadata->weight_dtype == VX_DTYPE_U8 &&
             (zero < 0 || zero > 255)))
            return 0;
        int64_t weight_distance = metadata->weight_dtype == VX_DTYPE_I8
            ? (zero < 0 ? 127 - zero : zero + 128)
            : (zero < 128 ? 255 - zero : zero);
        int64_t bias_distance = metadata->bias[channel] < 0
            ? -(int64_t)metadata->bias[channel]
            : (int64_t)metadata->bias[channel];
        uint64_t product = (uint64_t)metadata->d_in *
            (uint64_t)input_distance * (uint64_t)weight_distance;
        if (product > INT32_MAX ||
            (uint64_t)bias_distance > (uint64_t)INT32_MAX - product)
            return 0;
    }
    return 1;
}

static int vx_native_gpu_qconv_accumulator_domain_proven(
        const VxEngineState* validation,
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    const Node* runtime_node = NULL;
    const T* runtime_input = NULL;
    const QConv2DMetadata* metadata = NULL;
    uint64_t terms;
    if (!validation || !input || !output || input->rank != 4u ||
        output->rank != 4u || !output->name)
        return 0;
    for (int index = 0; index < validation->node_count; index++) {
        const Node* candidate = &validation->nodes[index];
        if (!strcmp(candidate->op, "QConv2D") &&
            !strcmp(candidate->out, output->name)) {
            if (runtime_node) return 0;
            runtime_node = candidate;
            metadata = &validation->qconv_metadata[index];
        }
    }
    if (!runtime_node || !metadata || !metadata->valid ||
        !metadata->weight_zero_points || !metadata->kernel_height ||
        !metadata->kernel_width || !metadata->input_per_group ||
        !metadata->input_channels || !metadata->output_channels ||
        input->maximums[3] != (int64_t)metadata->input_channels ||
        output->maximums[3] != (int64_t)metadata->output_channels ||
        (metadata->input_dtype != VX_DTYPE_I8 &&
         metadata->input_dtype != VX_DTYPE_U8) ||
        (metadata->weight_dtype != VX_DTYPE_I8 &&
         metadata->weight_dtype != VX_DTYPE_U8))
        return 0;
    runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "input");
    if (!runtime_input) runtime_input = vx_native_gpu_validation_node_input(
        validation, runtime_node, "x");
    if (!runtime_input || !runtime_input->quantization.valid ||
        runtime_input->dtype != metadata->input_dtype)
        return 0;
    terms = metadata->kernel_height;
    if (terms > UINT64_MAX / metadata->kernel_width) return 0;
    terms *= metadata->kernel_width;
    if (terms > UINT64_MAX / metadata->input_per_group) return 0;
    terms *= metadata->input_per_group;
    for (uint32_t channel = 0; channel < metadata->output_channels;
         channel++) {
        const int32_t* bias = metadata->bias
            ? &metadata->bias[channel] : NULL;
        if (!vx_native_gpu_quantized_accumulator_channel_proven(
                terms, metadata->input_dtype,
                runtime_input->quantization.zero_point,
                metadata->weight_dtype,
                metadata->weight_zero_points[channel], bias))
            return 0;
    }
    return 1;
}

static int vx_native_gpu_expand_domain_proven(const VxDeclaredTensor* input,
                                        const VxDeclaredTensor* output) {
    uint32_t leading;
    if (!input || !output || !input->rank || input->rank > output->rank)
        return 0;
    leading = output->rank - input->rank;
    for (uint32_t axis = 0; axis < input->rank; axis++) {
        int64_t fixed;
        if (vx_declared_dimension_fixed_value(input, axis, &fixed) &&
            fixed == 1) continue;
        if (!vx_declared_dimensions_provably_equal(
                input, axis, output, leading + axis)) return 0;
    }
    return 1;
}

static int vx_native_gpu_one_d_launch_proven(
        uint64_t elements,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits) {
    const uint64_t lanes = backend == VOLVOXAI_BACKEND_CUDA ? 256u : 64u;
    uint64_t grid;
    if (!elements || elements > UINT32_MAX || !limits ||
        limits->maximum_block[0] < lanes ||
        limits->maximum_threads_per_block < lanes)
        return 0;
    /* CUDA's common 1-D launcher uses 256 threads; Vulkan/OpenGL/Metal shader
     * routes use 64 scalar or packed-u32 lanes. Keep the proof identical to
     * the selected backend instead of relying on customary device minima. */
    grid = elements / lanes + (elements % lanes != 0u);
    return grid <= limits->maximum_grid[0];
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int vx_public_api_test_native_gpu_dense_launch_proven(
        uint64_t rows,
        uint64_t input_width,
        uint64_t output_width,
        int quantized,
        VolvoxAIEngineBackend backend,
        uint64_t maximum_grid_x,
        uint64_t maximum_grid_y,
        uint64_t maximum_block_x,
        uint64_t maximum_block_y,
        uint64_t maximum_threads_per_block) {
    VxNativeGpuDomainLimits limits = {0};
    limits.maximum_grid[0] = maximum_grid_x;
    limits.maximum_grid[1] = maximum_grid_y;
    limits.maximum_grid[2] = 1u;
    limits.maximum_block[0] = maximum_block_x;
    limits.maximum_block[1] = maximum_block_y;
    limits.maximum_block[2] = 1u;
    limits.maximum_threads_per_block = maximum_threads_per_block;
    return vx_native_gpu_dense_launch_proven(
        rows, input_width, output_width, quantized, backend, &limits);
}
#endif

static int vx_native_gpu_broadcast_axis_proven(
        const VxDeclaredTensor* operand,
        uint32_t operand_axis,
        const VxDeclaredTensor* output,
        uint32_t output_axis) {
    int64_t fixed;
    return (vx_declared_dimension_fixed_value(
                operand, operand_axis, &fixed) && fixed == 1) ||
        vx_declared_dimensions_provably_equal(
            operand, operand_axis, output, output_axis);
}

static int vx_native_gpu_binary_broadcast_domain_proven(
        const VxDeclaredTensor* a,
        const VxDeclaredTensor* b,
        const VxDeclaredTensor* output,
        int exact) {
    uint32_t a_leading;
    uint32_t b_leading;
    if (!a || !b || !output || !a->rank || !b->rank ||
        a->rank > output->rank || b->rank > output->rank)
        return 0;
    a_leading = output->rank - a->rank;
    b_leading = output->rank - b->rank;
    for (uint32_t axis = 0; axis < output->rank; axis++) {
        int a_present = axis >= a_leading;
        int b_present = axis >= b_leading;
        int a_equal = !a_present || vx_declared_dimensions_provably_equal(
            a, axis - a_leading, output, axis);
        int b_equal = !b_present || vx_declared_dimensions_provably_equal(
            b, axis - b_leading, output, axis);
        if (exact) {
            if (!a_present || !b_present || !a_equal || !b_equal) return 0;
            continue;
        }
        if ((a_present && !vx_native_gpu_broadcast_axis_proven(
                 a, axis - a_leading, output, axis)) ||
            (b_present && !vx_native_gpu_broadcast_axis_proven(
                 b, axis - b_leading, output, axis)) ||
            (!a_equal && !b_equal))
            return 0;
    }
    return 1;
}

static int vx_native_gpu_batch_matmul_domain_proven(
        const VxDeclaredTensor* a,
        const VxDeclaredTensor* b,
        const VxDeclaredTensor* output) {
    uint32_t output_batch_rank;
    uint32_t a_batch_rank;
    uint32_t b_batch_rank;
    uint32_t a_leading;
    uint32_t b_leading;
    if (!a || !b || !output || a->rank < 2u || a->rank > 8u ||
        b->rank < 2u || b->rank > 8u || output->rank < 2u ||
        output->rank > 8u ||
        output->rank != (a->rank > b->rank ? a->rank : b->rank) ||
        !vx_declared_dimensions_provably_equal(
            a, a->rank - 1u, b, b->rank - 2u) ||
        !vx_declared_dimensions_provably_equal(
            a, a->rank - 2u, output, output->rank - 2u) ||
        !vx_declared_dimensions_provably_equal(
            b, b->rank - 1u, output, output->rank - 1u))
        return 0;
    output_batch_rank = output->rank - 2u;
    a_batch_rank = a->rank - 2u;
    b_batch_rank = b->rank - 2u;
    a_leading = output_batch_rank - a_batch_rank;
    b_leading = output_batch_rank - b_batch_rank;
    for (uint32_t axis = 0; axis < output_batch_rank; axis++) {
        int a_present = axis >= a_leading;
        int b_present = axis >= b_leading;
        int a_equal = !a_present || vx_declared_dimensions_provably_equal(
            a, axis - a_leading, output, axis);
        int b_equal = !b_present || vx_declared_dimensions_provably_equal(
            b, axis - b_leading, output, axis);
        if ((a_present && !vx_native_gpu_broadcast_axis_proven(
                 a, axis - a_leading, output, axis)) ||
            (b_present && !vx_native_gpu_broadcast_axis_proven(
                 b, axis - b_leading, output, axis)) ||
            (!a_equal && !b_equal))
            return 0;
    }
    return 1;
}

static size_t vx_native_gpu_dynamic_symbol_count(
        const VxDeclaredTensor* tensor,
        const char* symbol) {
    size_t count = 0u;
    if (!tensor || !symbol) return 0u;
    for (uint32_t axis = 0; axis < tensor->rank; axis++)
        if (vx_dimension_domain_is_dynamic(tensor, axis) &&
            tensor->symbols[axis] &&
            !strcmp(tensor->symbols[axis], symbol))
            count++;
    return count;
}

static int vx_native_gpu_reshape_element_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output) {
    uint64_t input_fixed = 1u;
    uint64_t output_fixed = 1u;
    if (!input || !output || !input->rank || !output->rank) return 0;
    for (uint32_t axis = 0; axis < input->rank; axis++) {
        if (vx_dimension_domain_is_dynamic(input, axis)) {
            if (!input->symbols[axis] ||
                vx_native_gpu_dynamic_symbol_count(
                    input, input->symbols[axis]) !=
                vx_native_gpu_dynamic_symbol_count(
                    output, input->symbols[axis]))
                return 0;
        } else {
            uint64_t extent = (uint64_t)input->maximums[axis];
            if (!extent || input_fixed > UINT64_MAX / extent) return 0;
            input_fixed *= extent;
        }
    }
    for (uint32_t axis = 0; axis < output->rank; axis++) {
        if (vx_dimension_domain_is_dynamic(output, axis)) {
            if (!output->symbols[axis] ||
                vx_native_gpu_dynamic_symbol_count(
                    output, output->symbols[axis]) !=
                vx_native_gpu_dynamic_symbol_count(
                    input, output->symbols[axis]))
                return 0;
        } else {
            uint64_t extent = (uint64_t)output->maximums[axis];
            if (!extent || output_fixed > UINT64_MAX / extent) return 0;
            output_fixed *= extent;
        }
    }
    return input_fixed == output_fixed;
}

static int vx_native_gpu_transpose_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        const cJSON* params) {
    int64_t permutation[8] = {0};
    const cJSON* perm = cJSON_IsObject(params)
        ? cJSON_GetObjectItemCaseSensitive(params, "perm") : NULL;
    size_t count = 0u;
    unsigned int seen = 0u;
    if (!input || !output || !input->rank || input->rank > 8u ||
        input->rank != output->rank) return 0;
    if (perm) {
        if (!vx_native_gpu_integer_array(perm, permutation, 8u, &count) ||
            count != input->rank) return 0;
    } else {
        count = input->rank;
        for (size_t axis = 0; axis < count; axis++)
            permutation[axis] = (int64_t)count - 1 - (int64_t)axis;
    }
    for (uint32_t output_axis = 0; output_axis < output->rank; output_axis++) {
        int64_t input_axis = permutation[output_axis];
        if (input_axis < 0) input_axis += input->rank;
        if (input_axis < 0 || input_axis >= input->rank ||
            (seen & (1u << (unsigned int)input_axis)) ||
            !vx_declared_dimensions_provably_equal(
                input, (uint32_t)input_axis, output, output_axis))
            return 0;
        seen |= 1u << (unsigned int)input_axis;
    }
    return 1;
}

static int vx_native_gpu_slice_extent_at(int64_t input_extent,
                                   int64_t raw_start,
                                   int64_t raw_end,
                                   int64_t step,
                                   int64_t* output_extent) {
    int64_t start;
    int64_t end;
    if (input_extent <= 0 || raw_start < 0 || step <= 0 ||
        !output_extent) return 0;
    start = raw_start;
    end = raw_end < 0 ? raw_end + input_extent : raw_end;
    if (start > input_extent) start = input_extent;
    if (end < 0) end = 0;
    if (end > input_extent) end = input_extent;
    if (end <= start) return 0;
    *output_extent = (end - start - 1) / step + 1;
    return *output_extent > 0;
}

static int vx_native_gpu_slice_axis_domain_proven(
        const VxDeclaredTensor* input,
        const VxDeclaredTensor* output,
        uint32_t axis,
        int selected,
        int64_t raw_start,
        int64_t raw_end,
        int64_t step) {
    VxDimensionProgression input_domain;
    int64_t output_fixed;
    int64_t first_output;
    int64_t last_output;
    uint64_t last_index;
    if (!input || !output || axis >= input->rank || axis >= output->rank)
        return 0;
    if (!selected)
        return vx_declared_dimensions_provably_equal(
            input, axis, output, axis);
    if (!vx_declared_dimension_progression(
            input, axis, &input_domain)) {
        int64_t fixed;
        if (!vx_declared_dimension_fixed_value(input, axis, &fixed) ||
            fixed <= 0) return 0;
        input_domain.first = fixed;
        input_domain.last = fixed;
        input_domain.step = 1;
        input_domain.count = 1u;
    }

    /* A symbolic output can only repeat the input equality class here. The
     * static Slice contract has no graph-owned affine relation with which to
     * bind a distinct output symbol. The sole varying case representable by
     * that equality class is an unclipped identity slice. */
    if (vx_declared_dimensions_provably_equal(
            input, axis, output, axis) &&
        vx_dimension_domain_is_dynamic(input, axis)) {
        return raw_start == 0 && step == 1 && raw_end >= input_domain.last;
    }

    if (!vx_declared_dimension_fixed_value(output, axis, &output_fixed) ||
        output_fixed <= 0 ||
        !vx_native_gpu_slice_extent_at(
            input_domain.first, raw_start, raw_end, step, &first_output) ||
        !vx_native_gpu_slice_extent_at(
            input_domain.last, raw_start, raw_end, step, &last_output) ||
        first_output != output_fixed || last_output != output_fixed)
        return 0;

    /* The native GPU wrappers address the last selected element directly. Prove
     * their exact runtime guard for the smallest legal input extent; the
     * output is fixed above, so every larger extent is then safe as well. */
    if ((uint64_t)(output_fixed - 1) >
            (UINT64_MAX - (uint64_t)raw_start) / (uint64_t)step)
        return 0;
    last_index = (uint64_t)raw_start +
        (uint64_t)(output_fixed - 1) * (uint64_t)step;
    return last_index < (uint64_t)input_domain.first;
}

static int vx_native_gpu_concat_launch_domain_proven(
        const cJSON* node,
        const VxDeclaredTensor* output) {
    const cJSON* inputs = node
        ? cJSON_GetObjectItemCaseSensitive(node, "inputs") : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    int input_count = cJSON_IsObject(inputs)
        ? cJSON_GetArraySize(inputs) : 0;
    size_t axis;
    uint64_t inner;
    if (!output || !output->rank || input_count < 2 ||
        input_count > MAXIN || input_count > 256 ||
        !vx_concat_axis(params, output->rank, &axis) ||
        output->maximums[axis] <= 0 ||
        output->maximums[axis] > INT_MAX ||
        !vx_native_gpu_tensor_maximum_product(
            output, (uint32_t)axis + 1u, output->rank,
            INT_MAX, &inner))
        return 0;
    return inner > 0u;
}

enum VxNativeGpuValueProof {
    VX_NATIVE_GPU_VALUE_REJECTED = 0,
    VX_NATIVE_GPU_VALUE_PUBLIC_PREFLIGHT = 1,
    VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN = 2,
};

static const char* vx_native_gpu_validation_node_input_name(
        const Node* node, const char* role) {
    if (!node || !role) return NULL;
    for (int index = 0; index < node->nin; index++)
        if (!strcmp(node->ins[index].key, role))
            return node->ins[index].name;
    return NULL;
}

static const char* vx_native_gpu_validation_node_data_name(
        const Node* node) {
    static const char* const roles[] = {"input", "data", "x", "a"};
    if (!node) return NULL;
    for (size_t index = 0; index < sizeof(roles) / sizeof(roles[0]); index++) {
        const char* name = vx_native_gpu_validation_node_input_name(
            node, roles[index]);
        if (name) return name;
    }
    return node->nin > 0 ? node->ins[0].name : NULL;
}

static const Node* vx_native_gpu_validation_node_producer(
        const VxEngineState* validation, const char* tensor_name) {
    if (!validation || !tensor_name) return NULL;
    for (int index = 0; index < validation->node_count; index++) {
        const Node* node = &validation->nodes[index];
        if (node->out[0] && !strcmp(node->out, tensor_name)) return node;
        for (int output = 0; output < node->nout; output++)
            if (!strcmp(node->outs[output].name, tensor_name)) return node;
    }
    return NULL;
}

static int vx_native_gpu_validation_node_index(
        const VxEngineState* validation, const Node* node) {
    if (!validation || !node || node < validation->nodes ||
        node >= validation->nodes + validation->node_count)
        return -1;
    return (int)(node - validation->nodes);
}

static int vx_native_gpu_value_preserving_node(const Node* producer) {
    return producer &&
        (!strcmp(producer->op, "Reshape") ||
         !strcmp(producer->op, "Squeeze") ||
         !strcmp(producer->op, "Unsqueeze") ||
         !strcmp(producer->op, "Identity") ||
         !strcmp(producer->op, "Expand"));
}

/* Prove the origin of values that a GPU wrapper otherwise examines through a
 * host pointer. Public values are deferred to the execution-wide preflight;
 * immutable values are checked here while their host storage is authoritative.
 * A deliberately small producer proof prevents device-authored data from
 * being mistaken for current host data. */
static int vx_native_gpu_i32_value_source_proof(
        const VxEngineState* validation,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const char* tensor_name,
        int64_t minimum,
        int64_t maximum_exclusive,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!validation || !tensor_name || minimum >= maximum_exclusive ||
        before_node < 0 || before_node > validation->node_count ||
        depth < 0 || depth > validation->node_count)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    tensor = vx_native_gpu_validation_tensor_find(validation, tensor_name);
    if (!tensor || tensor->dtype != T_I32 || !tensor->data ||
        tensor->numel <= 0)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    if (tensor->is_graph_input)
        return VX_NATIVE_GPU_VALUE_PUBLIC_PREFLIGHT;
    producer = vx_native_gpu_validation_node_producer(validation, tensor_name);
    if (!producer) {
        for (long index = 0; index < tensor->numel; index++) {
            int32_t value;
            memcpy(&value,
                   (const unsigned char*)tensor->data +
                       (size_t)index * sizeof(value),
                   sizeof(value));
            if ((int64_t)value < minimum ||
                (int64_t)value >= maximum_exclusive)
                return VX_NATIVE_GPU_VALUE_REJECTED;
        }
        return VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN;
    }
    {
        int producer_index = vx_native_gpu_validation_node_index(
            validation, producer);
        if (producer_index < 0 || producer_index >= before_node)
            return VX_NATIVE_GPU_VALUE_REJECTED;
        before_node = producer_index;
    }
    if (!strcmp(producer->op, "Clip")) {
        int64_t clip_min;
        int64_t clip_max;
        if (!vx_native_gpu_integer_parameter(
                producer->params, "min", INT32_MIN,
                INT32_MIN, INT32_MAX, &clip_min) ||
            !vx_native_gpu_integer_parameter(
                producer->params, "max", INT32_MAX,
                INT32_MIN, INT32_MAX, &clip_max) ||
            clip_min > clip_max || clip_min < minimum ||
            clip_max >= maximum_exclusive)
            return VX_NATIVE_GPU_VALUE_REJECTED;
        return VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN;
    }
    if (!strcmp(producer->op, "ArgMax") ||
        !strcmp(producer->op, "QArgMax")) {
        const char* source_name = vx_native_gpu_validation_node_data_name(
            producer);
        const VxDeclaredTensor* source = vx_declared_tensor_find(
            tensors, tensor_count, source_name);
        int64_t axis;
        int qargmax = !strcmp(producer->op, "QArgMax");
        if (!source || !source->rank || minimum > 0 ||
            (qargmax &&
             (!producer->params ||
              cJSON_GetArraySize(producer->params) != 1)) ||
            !vx_native_gpu_integer_parameter(
                producer->params, "axis", qargmax ? -1 : 0,
                qargmax ? -1 : -(int64_t)source->rank,
                qargmax ? -1 : (int64_t)source->rank - 1, &axis))
            return VX_NATIVE_GPU_VALUE_REJECTED;
        if (axis < 0) axis += source->rank;
        return source->maximums[axis] > 0 &&
               source->maximums[axis] <= maximum_exclusive
            ? VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN
            : VX_NATIVE_GPU_VALUE_REJECTED;
    }
    if (vx_native_gpu_value_preserving_node(producer)) {
        const char* source_name = vx_native_gpu_validation_node_data_name(
            producer);
        return vx_native_gpu_i32_value_source_proof(
            validation, tensors, tensor_count, source_name,
            minimum, maximum_exclusive, before_node, depth + 1);
    }
    return VX_NATIVE_GPU_VALUE_REJECTED;
}

static int vx_native_gpu_finite_f32_value_source_proof(
        const VxEngineState* validation,
        const char* tensor_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!validation || !tensor_name || before_node < 0 ||
        before_node > validation->node_count || depth < 0 ||
        depth > validation->node_count)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    tensor = vx_native_gpu_validation_tensor_find(validation, tensor_name);
    if (!tensor || tensor->dtype != T_F32 || !tensor->data ||
        tensor->numel <= 0)
        return VX_NATIVE_GPU_VALUE_REJECTED;
    if (tensor->is_graph_input)
        return VX_NATIVE_GPU_VALUE_PUBLIC_PREFLIGHT;
    producer = vx_native_gpu_validation_node_producer(validation, tensor_name);
    if (!producer) {
        for (long index = 0; index < tensor->numel; index++) {
            float value;
            memcpy(&value,
                   (const unsigned char*)tensor->data +
                       (size_t)index * sizeof(value),
                   sizeof(value));
            if (!isfinite(value))
                return VX_NATIVE_GPU_VALUE_REJECTED;
        }
        return VX_NATIVE_GPU_VALUE_STATICALLY_PROVEN;
    }
    {
        int producer_index = vx_native_gpu_validation_node_index(
            validation, producer);
        if (producer_index < 0 || producer_index >= before_node)
            return VX_NATIVE_GPU_VALUE_REJECTED;
        before_node = producer_index;
    }
    if (vx_native_gpu_value_preserving_node(producer)) {
        const char* source_name = vx_native_gpu_validation_node_data_name(
            producer);
        return vx_native_gpu_finite_f32_value_source_proof(
            validation, source_name, before_node, depth + 1);
    }
    return VX_NATIVE_GPU_VALUE_REJECTED;
}

static VxStatus vx_native_gpu_node_domain_proof(
        const cJSON* node,
        const VxDeclaredTensor* tensors,
        size_t tensor_count,
        const VxEngineState* validation,
        VolvoxAIEngineBackend backend,
        const VxNativeGpuDomainLimits* limits,
        char* evidence,
        size_t evidence_capacity) {
    const cJSON* op_json = node
        ? cJSON_GetObjectItemCaseSensitive(node, "opType") : NULL;
    const cJSON* params = node
        ? cJSON_GetObjectItemCaseSensitive(node, "params") : NULL;
    const char* op = cJSON_IsString(op_json) ? op_json->valuestring : NULL;
    VxDeclaredTensor input_fixed = {0};
    const VxDeclaredTensor* input = vx_native_gpu_node_data_domain(
        node, tensors, tensor_count, validation, &input_fixed);
    const VxDeclaredTensor* output = vx_native_gpu_node_output(
        node, tensors, tensor_count);
    const Node* validation_node = output
        ? vx_native_gpu_validation_node_producer(validation, output->name)
        : NULL;
    const int validation_node_index = vx_native_gpu_validation_node_index(
        validation, validation_node);
    int64_t value;
    if (!op || !output || !limits)
        return vx_native_gpu_domain_reject(
            node, "missing-logical-output", evidence, evidence_capacity);

    if (!strcmp(op, "QConv2D")) {
        const cJSON* weight_only = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "weight_only")
            : NULL;
        uint64_t output_elements;
        /* Weight-only QConv2D dequantizes into a CPU cache and has no strict
         * native-GPU device route. Reject before validation forward instead
         * of allocating an unbounded CPU-only cache on a doomed route. */
        if (weight_only && (!cJSON_IsBool(weight_only) ||
                            cJSON_IsTrue(weight_only)))
            return vx_native_gpu_domain_reject(
                node, "qconv-weight-only-device-route", evidence,
                evidence_capacity);
        if (!input || input->rank != 4u || output->rank != 4u ||
            !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
            vx_dimension_domain_is_dynamic(input, 3u) ||
            vx_dimension_domain_is_dynamic(output, 3u) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qconv-channel-or-batch-domain", evidence,
                evidence_capacity);
        /* Kernel geometry, activation zero point, per-channel weight zero
         * points, and the optional aligned bias copy are invariant across a
         * shape rebind. Prove the exact worst-case I32 sum for every output
         * channel before admitting the device route. */
        if (!vx_native_gpu_qconv_accumulator_domain_proven(
                validation, input, output))
            return vx_native_gpu_domain_reject(
                node, "qconv-accumulator-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "QLinear") || !strcmp(op, "QMatMul") ||
               !strcmp(op, "QGemm")) {
        if (!input ||
            (input->dtype != VX_DTYPE_I8 &&
             input->dtype != VX_DTYPE_U8) ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !vx_native_gpu_dense_dynamic_rows_proven(
                input, output, 1, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qlinear-nonrow-dynamic-axis", evidence,
                evidence_capacity);
        /* K, affine metadata, per-channel zero points, and bias are immutable
         * across a shape rebind. Prove their exact I32 accumulator bound once;
         * only the independent row count may vary at execution. */
        if (!vx_native_gpu_qlinear_accumulator_domain_proven(
                validation, input, output))
            return vx_native_gpu_domain_reject(
                node, "qlinear-accumulator-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Linear")) {
        if (!input || input->dtype != VX_DTYPE_F32 ||
            output->dtype != VX_DTYPE_F32 ||
            !vx_native_gpu_dense_dynamic_rows_proven(
                input, output, 0, backend, limits) ||
            !vx_native_gpu_linear_storage_domain_proven(
                node, tensors, tensor_count, validation, input, output))
            return vx_native_gpu_domain_reject(
                node, "linear-f32-immutable-layout-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "GELU") || !strcmp(op, "SiLU") ||
               !strcmp(op, "QGELU") || !strcmp(op, "QSiLU") ||
               !strcmp(op, "Clip") || !strcmp(op, "Not") ||
               !strcmp(op, "Cast") ||
               !strcmp(op, "QuantizeLinear") ||
               !strcmp(op, "DequantizeLinear") ||
               !strcmp(op, "RequantizeLinear")) {
        uint64_t output_elements;
        if (!vx_native_gpu_tensor_shapes_equal(input, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "shape-preserving-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Reshape") || !strcmp(op, "Squeeze") ||
               !strcmp(op, "Unsqueeze")) {
        uint64_t output_elements;
        if (!vx_native_gpu_reshape_element_domain_proven(input, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "reshape-element-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Transpose")) {
        uint64_t output_elements;
        if (!vx_native_gpu_transpose_domain_proven(input, output, params) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "transpose-permutation-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Concat")) {
        /* The affine shape proof above establishes every input/output shape
         * relation. This native-GPU tail covers the host stack arrays and
         * its signed-int inner/output-axis launch parameters. */
        uint64_t output_elements;
        if (!vx_native_gpu_concat_launch_domain_proven(node, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "concat-count-or-index-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Add") || !strcmp(op, "QAdd") ||
               !strcmp(op, "Mul") ||
               !strcmp(op, "Sub") || !strcmp(op, "Div") ||
               !strcmp(op, "Equal") ||
               !strcmp(op, "GreaterOrEqual")) {
        VxDeclaredTensor a_fixed = {0};
        VxDeclaredTensor b_fixed = {0};
        const VxDeclaredTensor* a = vx_native_gpu_node_input_domain(
            node, "a", tensors, tensor_count, validation, &a_fixed);
        const VxDeclaredTensor* b = vx_native_gpu_node_input_domain(
            node, "b", tensors, tensor_count, validation, &b_fixed);
        int exact = !strcmp(op, "Add") || !strcmp(op, "QAdd");
        if (!a) a = vx_native_gpu_node_input_domain(
            node, "input", tensors, tensor_count, validation, &a_fixed);
        if (!a) a = vx_native_gpu_node_input_domain(
            node, "data", tensors, tensor_count, validation, &a_fixed);
        if (!vx_native_gpu_binary_broadcast_domain_proven(a, b, output, exact))
            return vx_native_gpu_domain_reject(
                node, exact ? "binary-exact-domain" :
                    "binary-broadcast-domain",
                evidence, evidence_capacity);
        {
            uint64_t output_elements;
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "binary-launch-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "BatchMatMul") ||
               !strcmp(op, "QBatchMatMul")) {
        VxDeclaredTensor a_fixed = {0};
        VxDeclaredTensor b_fixed = {0};
        const VxDeclaredTensor* a = vx_native_gpu_node_input_domain(
            node, "a", tensors, tensor_count, validation, &a_fixed);
        const VxDeclaredTensor* b = vx_native_gpu_node_input_domain(
            node, "b", tensors, tensor_count, validation, &b_fixed);
        int quantized = !strcmp(op, "QBatchMatMul");
        if (!a || !b ||
            (!quantized &&
             (a->dtype != VX_DTYPE_F32 || b->dtype != VX_DTYPE_F32 ||
              output->dtype != VX_DTYPE_F32)) ||
            (quantized &&
             ((a->dtype != VX_DTYPE_I8 && a->dtype != VX_DTYPE_U8) ||
              (b->dtype != VX_DTYPE_I8 && b->dtype != VX_DTYPE_U8) ||
              (output->dtype != VX_DTYPE_I8 &&
               output->dtype != VX_DTYPE_U8))) ||
            !vx_native_gpu_batch_matmul_domain_proven(a, b, output))
            return vx_native_gpu_domain_reject(
                node, "batch-matmul-broadcast-or-contract-domain",
                evidence, evidence_capacity);
        if (quantized) {
            uint64_t output_elements;
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "batch-matmul-launch-domain", evidence,
                    evidence_capacity);
            uint64_t k = (uint64_t)a->maximums[a->rank - 1u];
            if (k > (uint64_t)INT32_MAX / UINT64_C(65025))
                return vx_native_gpu_domain_reject(
                    node, "qbatch-accumulator-domain", evidence,
                    evidence_capacity);
        } else if (backend == VOLVOXAI_BACKEND_CUDA) {
            uint64_t output_elements;
            /* CUDA's F32 implementation is a flat output-element launch,
             * unlike the 8x8 tiled V/O/M kernels below. Prove that launch
             * against the queried grid as well as the tensor byte domain. */
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "batch-matmul-launch-domain", evidence,
                    evidence_capacity);
        } else {
            uint64_t batches;
            uint64_t m = (uint64_t)output->maximums[output->rank - 2u];
            uint64_t n = (uint64_t)output->maximums[output->rank - 1u];
            if (!vx_native_gpu_tensor_maximum_product(
                    output, 0u, output->rank - 2u,
                    UINT32_MAX, &batches) ||
                limits->maximum_block[0] < 8u ||
                limits->maximum_block[1] < 8u ||
                limits->maximum_threads_per_block < 64u ||
                (n + 7u) / 8u > limits->maximum_grid[0] ||
                (m + 7u) / 8u > limits->maximum_grid[1] ||
                batches > limits->maximum_grid[2])
                return vx_native_gpu_domain_reject(
                    node, "batch-matmul-tiled-launch-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "QSDPA")) {
        VxDeclaredTensor q_fixed = {0};
        VxDeclaredTensor k_fixed = {0};
        VxDeclaredTensor v_fixed = {0};
        VxDeclaredTensor mask_fixed = {0};
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* mask_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "mask") : NULL;
        const VxDeclaredTensor* q = vx_native_gpu_node_input_domain(
            node, "q", tensors, tensor_count, validation, &q_fixed);
        const VxDeclaredTensor* k = vx_native_gpu_node_input_domain(
            node, "k", tensors, tensor_count, validation, &k_fixed);
        const VxDeclaredTensor* v = vx_native_gpu_node_input_domain(
            node, "v", tensors, tensor_count, validation, &v_fixed);
        const VxDeclaredTensor* mask = vx_native_gpu_node_input_domain(
            node, "mask", tensors, tensor_count, validation, &mask_fixed);
        int64_t heads;
        int64_t d_model;
        uint64_t batch;
        uint64_t seq_q;
        uint64_t tasks;
        int mask_proven = !mask_reference;
        if (!q || !k || !v || (q->rank != 2u && q->rank != 3u) ||
            k->rank != q->rank || v->rank != q->rank ||
            output->rank != q->rank ||
            (q->dtype != VX_DTYPE_I8 && q->dtype != VX_DTYPE_U8) ||
            (k->dtype != VX_DTYPE_I8 && k->dtype != VX_DTYPE_U8) ||
            (v->dtype != VX_DTYPE_I8 && v->dtype != VX_DTYPE_U8) ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !vx_native_gpu_tensor_shapes_equal(q, output) ||
            !vx_native_gpu_tensor_shapes_equal(k, v) ||
            !vx_declared_dimensions_provably_equal(
                q, q->rank - 1u, k, k->rank - 1u) ||
            !vx_declared_dimension_fixed_value(
                q, q->rank - 1u, &d_model) ||
            d_model <= 0 ||
            !vx_native_gpu_integer_parameter(
                params, "heads", 0, 1, INT_MAX, &heads) ||
            d_model % heads || d_model % 4 ||
            d_model / heads > 64 || (d_model / heads) % 4)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-rank-feature-or-head-domain", evidence,
                evidence_capacity);
        if (q->rank == 3u &&
            (!vx_declared_dimensions_provably_equal(q, 0u, k, 0u) ||
             !vx_declared_dimensions_provably_equal(q, 0u, v, 0u)))
            return vx_native_gpu_domain_reject(
                node, "qsdpa-batch-domain", evidence, evidence_capacity);
        batch = q->rank == 3u ? (uint64_t)q->maximums[0] : 1u;
        seq_q = (uint64_t)q->maximums[q->rank - 2u];
        if (!batch || !seq_q || (uint64_t)heads > UINT32_MAX ||
            batch > UINT32_MAX / seq_q ||
            batch * seq_q > UINT32_MAX / (uint64_t)heads)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-task-domain", evidence, evidence_capacity);
        tasks = batch * seq_q * (uint64_t)heads;
        if (seq_q > limits->maximum_grid[0] ||
            (uint64_t)heads > limits->maximum_grid[1] ||
            batch > limits->maximum_grid[2] ||
            !vx_native_gpu_one_d_launch_proven(tasks, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qsdpa-launch-domain", evidence, evidence_capacity);
        if (mask_reference && !mask)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-unresolved-mask-domain", evidence,
                evidence_capacity);
        if (mask) {
            int64_t fixed_first = 0;
            if (mask->dtype == VX_DTYPE_I32 && mask->rank == 1u &&
                vx_declared_dimensions_provably_equal(
                    mask, 0u, k, k->rank - 2u)) {
                mask_proven = 1;
            } else if (mask->dtype == VX_DTYPE_I32 && mask->rank == 2u &&
                       vx_declared_dimensions_provably_equal(
                           mask, 1u, k, k->rank - 2u) &&
                       (vx_declared_dimensions_provably_equal(
                            mask, 0u, q, q->rank - 2u) ||
                        (q->rank == 3u &&
                         vx_declared_dimensions_provably_equal(
                             mask, 0u, q, 0u)) ||
                        (q->rank == 2u &&
                         vx_declared_dimension_fixed_value(
                             mask, 0u, &fixed_first) && fixed_first == 1))) {
                mask_proven = 1;
            } else if (mask->dtype == VX_DTYPE_I32 && mask->rank == 3u &&
                       q->rank == 3u &&
                       vx_declared_dimensions_provably_equal(
                           mask, 0u, q, 0u) &&
                       vx_declared_dimensions_provably_equal(
                           mask, 1u, q, 1u) &&
                       vx_declared_dimensions_provably_equal(
                           mask, 2u, k, 1u)) {
                mask_proven = 1;
            }
        }
        if (!mask_proven)
            return vx_native_gpu_domain_reject(
                node, "qsdpa-mask-domain", evidence, evidence_capacity);
    } else if (!strcmp(op, "QMaskedMean")) {
        VxDeclaredTensor mask_fixed = {0};
        const VxDeclaredTensor* mask = vx_native_gpu_node_input_domain(
            node, "mask", tensors, tensor_count, validation, &mask_fixed);
        uint64_t output_elements;
        if (!input || !mask || input->rank != 3u || mask->rank != 2u ||
            output->rank != 2u || mask->dtype != VX_DTYPE_I32 ||
            (input->dtype != VX_DTYPE_I8 &&
             input->dtype != VX_DTYPE_U8) ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !vx_declared_dimensions_provably_equal(input, 0u, mask, 0u) ||
            !vx_declared_dimensions_provably_equal(input, 1u, mask, 1u) ||
            !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
            !vx_declared_dimensions_provably_equal(input, 2u, output, 1u) ||
            (uint64_t)input->maximums[1] >
                (uint64_t)INT32_MAX / 255u ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qmaskedmean-shape-accumulator-or-launch-domain",
                evidence, evidence_capacity);
    } else if (!strcmp(op, "QArgMax")) {
        uint64_t outer;
        uint64_t inner;
        uint64_t output_elements;
        int64_t axis;
        if (!input || input->rank < 2u || input->rank > 8u ||
            output->rank + 1u != input->rank ||
            output->dtype != VX_DTYPE_I32 ||
            !vx_native_gpu_integer_parameter(
                params, "axis", 0, -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis))
            return vx_native_gpu_domain_reject(
                node, "qargmax-parameters-or-rank", evidence,
                evidence_capacity);
        if (axis < 0) axis += input->rank;
        for (uint32_t input_axis = 0, output_axis = 0;
             input_axis < input->rank; input_axis++) {
            if (input_axis == (uint32_t)axis) continue;
            if (output_axis >= output->rank ||
                !vx_declared_dimensions_provably_equal(
                    input, input_axis, output, output_axis++))
                return vx_native_gpu_domain_reject(
                    node, "qargmax-output-domain", evidence,
                    evidence_capacity);
        }
        if (!vx_native_gpu_tensor_maximum_product(
                input, 0u, (uint32_t)axis, UINT32_MAX, &outer) ||
            !vx_native_gpu_tensor_maximum_product(
                input, (uint32_t)axis + 1u, input->rank,
                UINT32_MAX, &inner) ||
            outer > UINT32_MAX / inner ||
            (uint64_t)input->maximums[axis] >
                UINT32_MAX / (outer * inner))
            return vx_native_gpu_domain_reject(
                node, "qargmax-index-domain", evidence,
                evidence_capacity);
        output_elements = outer * inner;
        if (!vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "qargmax-launch-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "ArgMax")) {
        uint64_t outer;
        uint64_t inner;
        uint64_t output_elements;
        int64_t axis;
        int64_t keepdims;
        int64_t select_last;
        if (!input || input->rank < 1u || input->rank > 8u ||
            !vx_native_gpu_integer_parameter(
                params, "axis", 0, -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis) ||
            !vx_native_gpu_integer_parameter(
                params, "keepdims", 1, 0, 1, &keepdims) ||
            !vx_native_gpu_integer_parameter(
                params, "select_last_index", 0, 0, 0, &select_last))
            return vx_native_gpu_domain_reject(
                node, "argmax-parameters", evidence, evidence_capacity);
        if (axis < 0) axis += input->rank;
        if (!vx_native_gpu_tensor_maximum_product(
                input, 0u, (uint32_t)axis, UINT32_MAX, &outer) ||
            !vx_native_gpu_tensor_maximum_product(
                input, (uint32_t)axis + 1u, input->rank,
                UINT32_MAX, &inner) ||
            outer > UINT32_MAX / inner)
            return vx_native_gpu_domain_reject(
                node, "argmax-index-domain", evidence, evidence_capacity);
        output_elements = outer * inner;
        if ((uint64_t)input->maximums[axis] >
                UINT32_MAX / output_elements ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "argmax-element-or-launch-domain", evidence,
                evidence_capacity);
        (void)keepdims;
        (void)select_last;
    } else if (!strcmp(op, "Gather")) {
        const VxDeclaredTensor* indices = vx_native_gpu_node_input_named(
            node, "indices", tensors, tensor_count);
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* indices_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "indices") : NULL;
        uint64_t indices_elements;
        uint64_t output_elements;
        if (!input || input->dtype != VX_DTYPE_F32 ||
            !vx_native_gpu_integer_parameter(params, "axis", 0, 0, 0, &value) ||
            !indices || indices->dtype != VX_DTYPE_I32 ||
            !cJSON_IsString(indices_reference) ||
            !indices_reference->valuestring ||
            !validation_node || validation_node->resident_slot_rows ||
            input->minimums[0] <= 0 || input->minimums[0] > INT_MAX ||
            vx_native_gpu_i32_value_source_proof(
                validation, tensors, tensor_count,
                indices_reference->valuestring,
                -input->minimums[0], input->minimums[0],
                validation_node_index, 0) ==
                VX_NATIVE_GPU_VALUE_REJECTED ||
            !vx_native_gpu_tensor_maximum_elements(indices, &indices_elements) ||
            !vx_native_gpu_tensor_maximum_elements(output, &output_elements) ||
            indices_elements > INT_MAX || output_elements > INT_MAX ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, validation_node && validation_node->resident_slot_rows
                    ? "banked-gather-global-slot-domain"
                    : "gather-axis0-count-or-index-origin-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Slice")) {
        const cJSON* starts = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "starts") : NULL;
        const cJSON* ends = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "ends") : NULL;
        const cJSON* steps = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "steps") : NULL;
        const cJSON* axes = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "axes") : NULL;
        int64_t starts_values[8] = {0};
        int64_t ends_values[8] = {0};
        int64_t steps_values[8] = {0};
        int64_t axes_values[8] = {0};
        size_t start_count = 0;
        size_t end_count = 0;
        size_t step_count = 0;
        size_t axis_count = 0;
        unsigned int seen = 0u;
        int64_t axis_starts[8] = {0};
        int64_t axis_ends[8] = {0};
        int64_t axis_steps[8] = {1, 1, 1, 1, 1, 1, 1, 1};
        if (!input || !input->rank || input->rank > 8u ||
            output->rank != input->rank || output->dtype != input->dtype ||
            (input->dtype != VX_DTYPE_F32 &&
             input->dtype != VX_DTYPE_I32) ||
            !vx_native_gpu_integer_array(
                starts, starts_values, 8u, &start_count) ||
            !vx_native_gpu_integer_array(
                ends, ends_values, 8u, &end_count) ||
            end_count != start_count ||
            (steps && (!vx_native_gpu_integer_array(
                steps, steps_values, 8u, &step_count) ||
                step_count != start_count)) ||
            (axes && (!vx_native_gpu_integer_array(
                axes, axes_values, 8u, &axis_count) ||
                axis_count != start_count)) || start_count > input->rank)
            return vx_native_gpu_domain_reject(
                node, "slice-static-parameters", evidence,
                evidence_capacity);
        for (size_t index = 0; index < start_count; index++) {
            int64_t axis = axes ? axes_values[index] : (int64_t)index;
            int64_t step = steps ? steps_values[index] : 1;
            if (axis < 0) axis += input->rank;
            if (axis < 0 || axis >= input->rank || step <= 0 ||
                starts_values[index] < 0 ||
                (seen & (1u << (unsigned int)axis)))
                return vx_native_gpu_domain_reject(
                    node, "slice-positive-static-domain", evidence,
                    evidence_capacity);
            seen |= 1u << (unsigned int)axis;
            axis_starts[axis] = starts_values[index];
            axis_ends[axis] = ends_values[index];
            axis_steps[axis] = step;
        }
        for (uint32_t axis = 0; axis < input->rank; axis++)
            if (!vx_native_gpu_slice_axis_domain_proven(
                    input, output, axis,
                    (seen & (1u << axis)) != 0u,
                    axis_starts[axis], axis_ends[axis], axis_steps[axis]))
                return vx_native_gpu_domain_reject(
                    node, "slice-output-or-index-domain", evidence,
                    evidence_capacity);
        {
            uint64_t output_elements;
            if (!vx_native_gpu_tensor_maximum_elements(
                    output, &output_elements) ||
                !vx_native_gpu_one_d_launch_proven(
                    output_elements, backend, limits))
                return vx_native_gpu_domain_reject(
                    node, "slice-launch-domain", evidence,
                    evidence_capacity);
        }
    } else if (!strcmp(op, "Softmax")) {
        int64_t axis;
        uint64_t rows;
        if (!input || !input->rank ||
            !vx_native_gpu_integer_parameter(
                params, "axis", (int64_t)input->rank - 1,
                -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis))
            return vx_native_gpu_domain_reject(
                node, "softmax-parameters", evidence, evidence_capacity);
        if (axis < 0) axis += input->rank;
        if (axis != (int64_t)input->rank - 1 ||
            !vx_native_gpu_tensor_maximum_product(
                input, 0u, input->rank - 1u, INT_MAX, &rows) ||
            !vx_native_gpu_one_d_launch_proven(rows, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "softmax-last-axis-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "ReduceSum")) {
        int64_t axis;
        uint64_t rows;
        if (!input || !input->rank ||
            !vx_native_gpu_integer_parameter(
                params, "axis", (int64_t)input->rank - 1,
                -(int64_t)input->rank,
                (int64_t)input->rank - 1, &axis))
            return vx_native_gpu_domain_reject(
                node, "reduce-parameters", evidence, evidence_capacity);
        if (axis < 0) axis += input->rank;
        if (axis != (int64_t)input->rank - 1 ||
            !vx_native_gpu_tensor_maximum_product(
                input, 0u, input->rank - 1u, INT_MAX, &rows) ||
            !vx_native_gpu_one_d_launch_proven(rows, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "reduce-last-axis-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Where")) {
        const VxDeclaredTensor* condition = vx_native_gpu_node_input_named(
            node, "condition", tensors, tensor_count);
        const VxDeclaredTensor* a = vx_native_gpu_node_input_named(
            node, "x", tensors, tensor_count);
        const VxDeclaredTensor* b = vx_native_gpu_node_input_named(
            node, "y", tensors, tensor_count);
        if (!condition) condition = vx_native_gpu_node_input_named(
            node, "cond", tensors, tensor_count);
        if (!a) a = vx_native_gpu_node_input_named(
            node, "a", tensors, tensor_count);
        if (!b) b = vx_native_gpu_node_input_named(
            node, "b", tensors, tensor_count);
        uint64_t output_elements;
        if (!condition || !a || !b ||
            condition->dtype != VX_DTYPE_I32 ||
            (a->dtype != VX_DTYPE_F32 && a->dtype != VX_DTYPE_I32) ||
            !vx_native_gpu_tensor_shapes_equal(condition, output) ||
            !vx_native_gpu_tensor_domains_equal(a, output) ||
            !vx_native_gpu_tensor_domains_equal(b, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "where-exact-i32-condition-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Expand")) {
        uint64_t output_elements;
        if (!input ||
            (input->dtype != VX_DTYPE_F32 &&
             input->dtype != VX_DTYPE_I32) ||
            input->dtype != output->dtype ||
            !vx_native_gpu_expand_domain_proven(input, output) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "expand-broadcast-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "LayerNorm") || !strcmp(op, "QLayerNorm")) {
        const char* weight_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "weight") : NULL;
        const char* bias_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "bias") : NULL;
        uint64_t rows;
        uint64_t output_elements;
        if (!input || !input->rank ||
            !vx_native_gpu_tensor_shapes_equal(input, output) ||
            vx_dimension_domain_is_dynamic(input, input->rank - 1u) ||
            vx_dimension_domain_is_dynamic(output, output->rank - 1u) ||
            !vx_native_gpu_tensor_maximum_product(
                input, 0u, input->rank - 1u, INT_MAX, &rows) ||
            rows > limits->maximum_grid[0] ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "layernorm-feature-row-or-launch-domain", evidence,
                evidence_capacity);
        if (!strcmp(op, "QLayerNorm") &&
            (!weight_name || !bias_name ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, weight_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, bias_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED))
            return vx_native_gpu_domain_reject(
                node, "qlayernorm-affine-value-origin-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "GroupNorm") || !strcmp(op, "QGroupNorm")) {
        const char* weight_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "weight") : NULL;
        const char* bias_name = validation_node
            ? vx_native_gpu_validation_node_input_name(
                validation_node, "bias") : NULL;
        const cJSON* num_groups = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "num_groups") : NULL;
        int64_t groups = 1;
        uint64_t tasks;
        uint64_t output_elements;
        if (!input || input->rank != 4u || output->rank != 4u ||
            !vx_native_gpu_tensor_shapes_equal(input, output) ||
            vx_dimension_domain_is_dynamic(input, 3u) ||
            vx_dimension_domain_is_dynamic(output, 3u) ||
            !vx_native_gpu_integer_parameter(
                params, num_groups ? "num_groups" : "groups", 1,
                1, INT_MAX, &groups) ||
            groups > input->maximums[3] ||
            input->maximums[3] % groups ||
            (uint64_t)input->maximums[0] >
                (uint64_t)INT_MAX / (uint64_t)groups)
            return vx_native_gpu_domain_reject(
                node, "groupnorm-channel-or-task-domain", evidence,
                evidence_capacity);
        tasks = (uint64_t)input->maximums[0] * (uint64_t)groups;
        if (limits->maximum_block[0] < 64u ||
            limits->maximum_threads_per_block < 64u ||
            tasks > limits->maximum_grid[0])
            return vx_native_gpu_domain_reject(
                node, "groupnorm-launch-domain", evidence,
                evidence_capacity);
        if (!vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, "groupnorm-apply-launch-domain", evidence,
                evidence_capacity);
        if (!strcmp(op, "QGroupNorm") &&
            (!weight_name || !bias_name ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, weight_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED ||
             vx_native_gpu_finite_f32_value_source_proof(
                 validation, bias_name, validation_node_index, 0) ==
                 VX_NATIVE_GPU_VALUE_REJECTED))
            return vx_native_gpu_domain_reject(
                node, "qgroupnorm-affine-value-origin-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "Conv2D")) {
        VxDeclaredTensor weight_fixed = {0};
        const VxDeclaredTensor* logical_weight =
            vx_native_gpu_node_input_named(
                node, "weight", tensors, tensor_count);
        const VxDeclaredTensor* logical_bias =
            vx_native_gpu_node_input_named(
                node, "bias", tensors, tensor_count);
        const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
            node, "weight", tensors, tensor_count, validation,
            &weight_fixed);
        const cJSON* layout_item = cJSON_IsObject(params)
            ? cJSON_GetObjectItemCaseSensitive(params, "weight_layout")
            : NULL;
        const char* layout = !layout_item || cJSON_IsNull(layout_item)
            ? "HWIO"
            : cJSON_IsString(layout_item) && layout_item->valuestring
                ? layout_item->valuestring : NULL;
        int direct_f32_layout = layout &&
            (!strcmp(layout, "HWIO") || !strcmp(layout, "HWCM"));
        uint64_t grid_x;
        uint64_t grid_y;
        uint64_t grid_z;
        uint64_t output_channels;
        if (!input || !weight || input->rank != 4u ||
            weight->rank != 4u || output->rank != 4u ||
            !vx_declared_dimensions_provably_equal(input, 0u, output, 0u) ||
            vx_dimension_domain_is_dynamic(input, 3u) ||
            vx_dimension_domain_is_dynamic(output, 3u))
            return vx_native_gpu_domain_reject(
                node, "conv-dynamic-channel", evidence,
                evidence_capacity);
        /* The current Conv2D shape contract is canonical HWIO/HWCM. Reject
         * legacy transformed layouts even for immutable storage rather than
         * compiling a route whose first public execute cannot shape-bind.
         * Within the canonical layouts, immutable F16 may own a widening
         * cache while a logical/public weight must remain direct F32. */
        if (!direct_f32_layout)
            return vx_native_gpu_domain_reject(
                node, "conv-noncanonical-weight-layout-domain",
                evidence, evidence_capacity);
        if ((logical_weight && weight->dtype != VX_DTYPE_F32) ||
            (logical_bias && logical_bias->dtype == VX_DTYPE_F16))
            return vx_native_gpu_domain_reject(
                node, logical_weight
                    ? "conv-mutable-transformed-weight-domain"
                    : "conv-mutable-f16-bias-domain",
                evidence, evidence_capacity);
        output_channels = (uint64_t)output->maximums[3];
        grid_x = (uint64_t)output->maximums[2] / 8u +
            ((uint64_t)output->maximums[2] % 8u != 0u);
        grid_y = (uint64_t)output->maximums[1] / 8u +
            ((uint64_t)output->maximums[1] % 8u != 0u);
        if (!output_channels ||
            (uint64_t)output->maximums[0] > UINT64_MAX / output_channels)
            return vx_native_gpu_domain_reject(
                node, "conv-grid-domain", evidence, evidence_capacity);
        grid_z = (uint64_t)output->maximums[0] * output_channels;
        /* Every V/O/M Conv2D tactic, including its generic failure fallback,
         * launches 8x8 over output width/height. The generic route owns one Z
         * workgroup per batch/output-channel pair, so it is the complete
         * tactic-independent upper bound. */
        if (limits->maximum_block[0] < 8u ||
            limits->maximum_block[1] < 8u ||
            limits->maximum_threads_per_block < 64u ||
            grid_x > limits->maximum_grid[0] ||
            grid_y > limits->maximum_grid[1] ||
            grid_z > limits->maximum_grid[2])
            return vx_native_gpu_domain_reject(
                node, "conv-grid-domain", evidence,
                evidence_capacity);
    } else if (!strcmp(op, "QEmbedding")) {
        VxDeclaredTensor weight_fixed = {0};
        const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
            node, "weight", tensors, tensor_count, validation,
            &weight_fixed);
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* ids_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "input") : NULL;
        uint64_t token_count;
        uint64_t output_elements;
        if (!input || !input->rank || output->rank != input->rank + 1u ||
            input->dtype != VX_DTYPE_I32 ||
            (output->dtype != VX_DTYPE_I8 &&
             output->dtype != VX_DTYPE_U8) ||
            !weight || weight->rank != 2u ||
            (weight->dtype != VX_DTYPE_I8 &&
             weight->dtype != VX_DTYPE_U8) ||
            !cJSON_IsString(ids_reference) || !ids_reference->valuestring ||
            !validation_node || validation_node->resident_slot_rows ||
            vx_dimension_domain_is_dynamic(weight, 0u) ||
            vx_dimension_domain_is_dynamic(weight, 1u) ||
            weight->minimums[0] <= 0 || weight->minimums[0] > INT_MAX ||
            vx_native_gpu_i32_value_source_proof(
                validation, tensors, tensor_count,
                ids_reference->valuestring, 0, weight->minimums[0],
                validation_node_index, 0) ==
                VX_NATIVE_GPU_VALUE_REJECTED ||
            vx_dimension_domain_is_dynamic(output, output->rank - 1u) ||
            !vx_declared_dimensions_provably_equal(
                weight, 1u, output, output->rank - 1u) ||
            !vx_native_gpu_tensor_maximum_elements(input, &token_count) ||
            !vx_native_gpu_tensor_maximum_elements(
                output, &output_elements) ||
            token_count > UINT32_MAX || output_elements > UINT32_MAX ||
            !vx_native_gpu_one_d_launch_proven(
                output_elements, backend, limits))
            return vx_native_gpu_domain_reject(
                node, validation_node && validation_node->resident_slot_rows
                    ? "banked-qembedding-global-slot-domain"
                    : "qembedding-feature-or-index-origin-domain", evidence,
                evidence_capacity);
        for (uint32_t axis = 0; axis < input->rank; axis++)
            if (!vx_declared_dimensions_provably_equal(
                    input, axis, output, axis))
                return vx_native_gpu_domain_reject(
                    node, "qembedding-prefix-domain", evidence,
                    evidence_capacity);
    } else if (!strcmp(op, "Embedding")) {
        VxDeclaredTensor weight_fixed = {0};
        const VxDeclaredTensor* weight = vx_native_gpu_node_input_domain(
            node, "weight", tensors, tensor_count, validation,
            &weight_fixed);
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const cJSON* ids_reference = cJSON_IsObject(inputs)
            ? cJSON_GetObjectItemCaseSensitive(inputs, "input") : NULL;
        uint64_t token_count;
        uint64_t vocabulary;
        if (!input || !input->rank || output->rank != input->rank + 1u ||
            input->dtype != VX_DTYPE_I32 || output->dtype != VX_DTYPE_F32 ||
            !weight || weight->dtype != VX_DTYPE_F32 || !weight->rank ||
            !cJSON_IsString(ids_reference) || !ids_reference->valuestring ||
            !validation_node || validation_node->resident_slot_rows ||
            vx_dimension_domain_is_dynamic(output, output->rank - 1u) ||
            !vx_declared_dimensions_provably_equal(
                weight, weight->rank - 1u,
                output, output->rank - 1u) ||
            !vx_native_gpu_tensor_maximum_elements(input, &token_count) ||
            token_count > INT_MAX ||
            !vx_native_gpu_tensor_maximum_product(
                weight, 0u, weight->rank - 1u, INT_MAX, &vocabulary) ||
            vocabulary > INT_MAX ||
            vx_native_gpu_i32_value_source_proof(
                validation, tensors, tensor_count,
                ids_reference->valuestring, 0, (int64_t)vocabulary,
                validation_node_index, 0) ==
                VX_NATIVE_GPU_VALUE_REJECTED)
            return vx_native_gpu_domain_reject(
                node, validation_node && validation_node->resident_slot_rows
                    ? "banked-embedding-global-slot-domain"
                    : "embedding-feature-or-index-origin-domain", evidence,
                evidence_capacity);
        if (!vx_native_gpu_one_d_launch_proven(
                (uint64_t)output->maximum_byte_size / sizeof(float),
                backend, limits))
            return vx_native_gpu_domain_reject(
                node, "embedding-launch-domain", evidence,
                evidence_capacity);
        for (uint32_t axis = 0; axis < input->rank; axis++)
            if (!vx_declared_dimensions_provably_equal(
                    input, axis, output, axis))
                return vx_native_gpu_domain_reject(
                    node, "embedding-prefix-domain", evidence,
                    evidence_capacity);
    } else {
        return vx_native_gpu_domain_reject(
            node, "missing-native-gpu-domain-proof", evidence,
            evidence_capacity);
    }
    return VX_STATUS_OK;
}

typedef struct VxBuiltinResourceDomain {
    size_t maximum_typed_scratch_bytes;
    uint64_t maximum_resident_bytes;
} VxBuiltinResourceDomain;

/* Compilation proves the declared domain from immutable logical metadata and
 * the generated native descriptor-predicate registry. The private minimum
 * projection remains a bootstrap/prepack validation only; it is not used as
 * evidence that other legal shapes are routable. */
static VxStatus vx_builtin_bounded_domain_proof(
        const VxModel* model,
        const VxWeightRevisionRecord* weights,
        VolvoxAIEngineBackend backend,
        VxEngineState* validation,
        VxBuiltinResourceDomain* resources,
        char* evidence,
        size_t evidence_capacity) {
    VxDeclaredTensor* tensors = NULL;
    size_t tensor_count = 0;
    const cJSON* root = NULL;
    const cJSON* nodes;
    const char* registry_backend;
    uint64_t digest = UINT64_C(1469598103934665603);
    uint64_t maximum_resident = 0;
    uint64_t maximum_weight_bytes = 0;
    uint64_t native_gpu_immutable_bytes = 0;
    uint64_t native_gpu_device_immutable_bytes = 0;
    uint64_t native_gpu_bootstrap_host_bytes = 0;
    uint64_t native_gpu_bootstrap_device_bytes = 0;
    uint64_t native_gpu_arena_bytes = 0;
    uint64_t native_gpu_device_span_bytes = 0;
    uint64_t native_gpu_dispatch_temporary_bytes = 0;
    uint64_t native_gpu_activation_peak_bytes = 0;
    uint64_t native_gpu_device_peak_bytes = 0;
    uint64_t native_gpu_span_count = 0;
    uint64_t native_gpu_qgroupnorm_stats_bytes = 0;
    uint64_t native_gpu_qlayernorm_stats_bytes = 0;
    uint64_t native_gpu_scratch_required_bytes = 0;
    uint64_t native_gpu_result_snapshot_bytes = 0;
    uint64_t native_gpu_result_publication_peak_bytes = 0;
    uint64_t native_gpu_dynamic_metadata_peak_bytes = 0;
    uint64_t maximum_launch_elements = 0;
    size_t maximum_typed_scratch = 0u;
    size_t operator_count = 0;
    size_t canonical_count = 0;
    size_t affine_relation_count = 0;
    int native_gpu_validation_bounds_ready = 0;
    int dynamic = 0;
    VxStatus status;
    VxNativeGpuDomainLimits native_gpu_limits = {0};
    if (!model || !resources || !evidence || !evidence_capacity)
        return VX_STATUS_INVALID_ARGUMENT;
    memset(resources, 0, sizeof(*resources));
    evidence[0] = '\0';
    registry_backend = backend == VOLVOXAI_BACKEND_CPU ? "native-cpu" :
        backend == VOLVOXAI_BACKEND_VULKAN ? "vulkan" :
        backend == VOLVOXAI_BACKEND_OPENGL ? "opengl" :
        backend == VOLVOXAI_BACKEND_METAL ? "metal" :
        backend == VOLVOXAI_BACKEND_NNAPI ? "nnapi" :
        backend == VOLVOXAI_BACKEND_CUDA ? "cuda" : NULL;
    if (!registry_backend) return VX_STATUS_BACKEND_UNSUPPORTED;
    status = vx_logical_tensors_load(model, &tensors, &tensor_count);
    if (status != VX_STATUS_OK) return status;
    for (size_t index = 0; index < tensor_count; index++)
        for (uint32_t axis = 0; axis < tensors[index].rank; axis++)
            dynamic |= vx_dimension_domain_is_dynamic(
                &tensors[index], axis);
    for (size_t index = 0; index < tensor_count; index++) {
        const VxDeclaredTensor* tensor = &tensors[index];
        uint64_t resident_charge = (uint64_t)tensor->maximum_byte_size;
        uint64_t maximum_elements = 0u;
        int tensor_dynamic = 0;
        if (dynamic && vx_native_gpu_backend(backend) &&
            tensor->dtype == VX_DTYPE_F16) {
            /* The native graph ABI has no F16 activation slots. Some legacy
             * copy-shaped routes widen their source to F32, so reserving the
             * declared two-byte domain would under-size the rebound output.
             * Immutable F16 model weights remain supported: they are outside
             * this closed logical activation table and preload as F32. */
            snprintf(evidence, evidence_capacity,
                     "unsupported_tensor=%s;"
                     "required=native-gpu-full-bounded-domain;"
                     "predicate_reason="
                     "native-gpu-f32-or-quantized-logical-domain",
                     tensor->name ? tensor->name : "<unknown>");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (vx_native_gpu_backend(backend) &&
            (!vx_native_gpu_tensor_maximum_elements(tensor, &maximum_elements) ||
             tensor->rank > 8u || maximum_elements > UINT32_MAX ||
             tensor->maximum_byte_size > UINT32_MAX ||
             (tensor->dtype == VX_DTYPE_F16 &&
              maximum_elements > UINT32_MAX / sizeof(float)))) {
            snprintf(evidence, evidence_capacity,
                     "unsupported_tensor=%s;"
                     "required=native-gpu-u32-tensor-domain",
                     tensor->name ? tensor->name : "<unknown>");
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (vx_native_gpu_backend(backend) &&
            tensor->dtype == VX_DTYPE_F16)
            resident_charge = maximum_elements * sizeof(float);
        if (maximum_elements > maximum_launch_elements)
            maximum_launch_elements = maximum_elements;
        for (uint32_t axis = 0; axis < tensor->rank; axis++)
            tensor_dynamic |= vx_dimension_domain_is_dynamic(tensor, axis);
        /* Native-GPU dynamic activation ownership is charged from the exact
         * maximum layout and the backend-specific bootstrap/candidate phases
         * after that layout has been built. */
        if (!dynamic || !vx_native_gpu_backend(backend)) {
            if (tensor_dynamic) {
                uint64_t copies = vx_native_gpu_backend(backend) &&
                    tensor->dtype == VX_DTYPE_F16 ? 4u : 3u;
                /* Widening an owned F16 bootstrap tensor keeps its retired
                 * F16 storage until teardown, in addition to the F32
                 * bootstrap, maximum host arena, and device span. Four
                 * F32-width copies cover that exceptional lifetime. */
                if (resident_charge > UINT64_MAX / copies) {
                    status = VX_STATUS_BACKEND_UNSUPPORTED;
                    goto done;
                }
                resident_charge *= copies;
            }
        } else {
            resident_charge = 0u;
        }
        if (resident_charge > UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident += resident_charge;
        digest = vx_revision_update(digest, tensor->name,
                                    strlen(tensor->name) + 1u);
        digest = vx_revision_update(digest, &tensor->dtype,
                                    sizeof(tensor->dtype));
        digest = vx_revision_update(digest, &tensor->rank,
                                    sizeof(tensor->rank));
        for (uint32_t axis = 0; axis < tensor->rank; axis++) {
            if (tensor->maximums[axis] > INT_MAX) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                goto done;
            }
            digest = vx_revision_update(digest, &tensor->kinds[axis],
                                        sizeof(tensor->kinds[axis]));
            digest = vx_revision_update(digest, &tensor->minimums[axis],
                                        sizeof(tensor->minimums[axis]));
            digest = vx_revision_update(digest, &tensor->maximums[axis],
                                        sizeof(tensor->maximums[axis]));
            digest = vx_revision_update(digest, &tensor->multiples[axis],
                                        sizeof(tensor->multiples[axis]));
            if (tensor->symbols[axis])
                digest = vx_revision_update(
                    digest, tensor->symbols[axis],
                    strlen(tensor->symbols[axis]) + 1u);
        }
    }
    for (size_t index = 0; weights && index < weights->path_count; index++) {
        FILE* file = fopen(weights->paths[index], "rb");
        long size;
        if (!file) {
            status = VX_STATUS_IO_ERROR;
            goto done;
        }
        if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
            (uint64_t)size > UINT64_MAX - maximum_weight_bytes) {
            (void)fclose(file);
            status = VX_STATUS_IO_ERROR;
            goto done;
        }
        if (fclose(file) != 0) {
            status = VX_STATUS_IO_ERROR;
            goto done;
        }
        maximum_weight_bytes += (uint64_t)size;
    }
    if (vx_native_gpu_backend(backend)) {
        if (!vx_native_gpu_validation_immutable_bytes_bound(
                validation, tensors, tensor_count,
                backend,
                1u, &native_gpu_immutable_bytes,
                &native_gpu_device_immutable_bytes)) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        /* The mapped host revision remains alive while the native GPU owns
         * converted, packed, and aligned immutable copies. They are simultaneous
         * resources, so max(file, device) would understate the context. */
        if (native_gpu_immutable_bytes >
            UINT64_MAX - maximum_weight_bytes) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_weight_bytes += native_gpu_immutable_bytes;
        native_gpu_validation_bounds_ready = 1;
    }
    if (maximum_weight_bytes > UINT64_MAX - maximum_resident) {
        status = VX_STATUS_BACKEND_UNSUPPORTED;
        goto done;
    }
    maximum_resident += maximum_weight_bytes;
    if (dynamic && backend != VOLVOXAI_BACKEND_CPU &&
        !vx_native_gpu_backend(backend)) {
        status = VX_STATUS_BACKEND_UNSUPPORTED;
        goto done;
    }
    /* Value-origin and exact launch predicates are safety contracts for every
     * native-GPU graph, not only graphs whose logical extents vary.  Dynamic
     * graphs query these limits as part of maximum-layout construction below;
     * fixed graphs still need the same device limits before per-node proof. */
    if (!dynamic && vx_native_gpu_backend(backend)) {
        if (!validation || !native_gpu_validation_bounds_ready ||
            vx_native_gpu_query_domain_limits(
                backend, &native_gpu_limits) != 0) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        digest = vx_revision_update(
            digest, &native_gpu_limits, sizeof(native_gpu_limits));
    }
    if (dynamic && vx_native_gpu_backend(backend)) {
        int proof_status;
        uint64_t aligned_immutable_bytes = 0u;
        uint64_t aligned_device_immutable_bytes = 0u;
        uint64_t immutable_alignment_delta = 0u;
        if (!validation || !native_gpu_validation_bounds_ready) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        proof_status = vx_native_gpu_maximum_layout_proof(
            validation, tensors, tensor_count, backend, &native_gpu_limits,
            &native_gpu_arena_bytes, &native_gpu_device_span_bytes,
            &native_gpu_span_count,
            &native_gpu_qgroupnorm_stats_bytes,
            &native_gpu_qlayernorm_stats_bytes,
            &native_gpu_scratch_required_bytes);
        if (proof_status != 0) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-device-domain-capacity;"
                "backend=%s;layout_status=%d;arena_bytes=%" PRIu64
                ";immutable_bytes=%" PRIu64,
                registry_backend, proof_status,
                native_gpu_arena_bytes, native_gpu_immutable_bytes);
            status = proof_status == -2
                ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (!vx_native_gpu_validation_immutable_bytes_bound(
                validation, tensors, tensor_count,
                backend,
                native_gpu_limits.storage_alignment,
                &aligned_immutable_bytes,
                &aligned_device_immutable_bytes) ||
            aligned_immutable_bytes < native_gpu_immutable_bytes ||
            aligned_device_immutable_bytes <
                native_gpu_device_immutable_bytes) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        immutable_alignment_delta =
            aligned_immutable_bytes - native_gpu_immutable_bytes;
        if (immutable_alignment_delta > UINT64_MAX - maximum_weight_bytes ||
            immutable_alignment_delta > UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_weight_bytes += immutable_alignment_delta;
        maximum_resident += immutable_alignment_delta;
        native_gpu_immutable_bytes = aligned_immutable_bytes;
        native_gpu_device_immutable_bytes =
            aligned_device_immutable_bytes;
        if (!vx_native_gpu_validation_bootstrap_activation_bound(
                validation, tensors, tensor_count,
                native_gpu_limits.storage_alignment,
                &native_gpu_bootstrap_host_bytes,
                &native_gpu_bootstrap_device_bytes) ||
            !vx_native_gpu_validation_dispatch_temporary_bound(
                validation, tensors, tensor_count, backend,
                native_gpu_limits.storage_alignment,
                &native_gpu_dispatch_temporary_bytes) ||
            !vx_native_gpu_activation_resource_peak(
                backend, native_gpu_bootstrap_host_bytes,
                native_gpu_arena_bytes,
                native_gpu_bootstrap_device_bytes,
                native_gpu_device_span_bytes,
                native_gpu_qgroupnorm_stats_bytes,
                native_gpu_qlayernorm_stats_bytes,
                native_gpu_dispatch_temporary_bytes,
                native_gpu_scratch_required_bytes,
                &native_gpu_activation_peak_bytes) ||
            native_gpu_activation_peak_bytes >
                UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        /* maximum_total_span_bytes covers the graph arena shared by logical
         * activations and resident weights. Each Vulkan suballocation is
         * rounded to the queried device alignment before this comparison.
         * CUDA frees bootstrap transients before reservation; Vulkan returns
         * their ranges to the same arena, so the larger phase is sufficient.
         * Quantized-norm statistics live in separately proved scratch/buffers
         * and are not part of that address range. */
        if (!vx_native_gpu_device_capacity_proven(
                native_gpu_device_immutable_bytes,
                native_gpu_bootstrap_device_bytes,
                native_gpu_device_span_bytes,
                native_gpu_limits.maximum_total_span_bytes,
                &native_gpu_device_peak_bytes)) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-device-domain-capacity;"
                "backend=%s;device_bytes=%" PRIu64
                ";device_limit=%" PRIu64,
                registry_backend,
                native_gpu_device_peak_bytes,
                native_gpu_limits.maximum_total_span_bytes);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        digest = vx_revision_update(
            digest, &native_gpu_limits, sizeof(native_gpu_limits));
        digest = vx_revision_update(
            digest, &native_gpu_arena_bytes, sizeof(native_gpu_arena_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_device_span_bytes,
            sizeof(native_gpu_device_span_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_bootstrap_host_bytes,
            sizeof(native_gpu_bootstrap_host_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_bootstrap_device_bytes,
            sizeof(native_gpu_bootstrap_device_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_dispatch_temporary_bytes,
            sizeof(native_gpu_dispatch_temporary_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_span_count, sizeof(native_gpu_span_count));
        digest = vx_revision_update(
            digest, &native_gpu_qgroupnorm_stats_bytes,
            sizeof(native_gpu_qgroupnorm_stats_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_qlayernorm_stats_bytes,
            sizeof(native_gpu_qlayernorm_stats_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_scratch_required_bytes,
            sizeof(native_gpu_scratch_required_bytes));
        maximum_resident += native_gpu_activation_peak_bytes;
    }
    if (vx_native_gpu_backend(backend)) {
        uint64_t proved_resident = 0u;
        if (!vx_native_gpu_result_publication_peak(
                model, &native_gpu_result_snapshot_bytes,
                &native_gpu_result_publication_peak_bytes) ||
            (dynamic && !vx_native_gpu_dynamic_metadata_peak(
                model, (uint64_t)tensor_count,
                validation ? (uint64_t)validation->tensor_count : 0u,
                native_gpu_span_count,
                &native_gpu_dynamic_metadata_peak_bytes)) ||
            !vx_native_gpu_resident_capacity_proven(
                maximum_resident,
                native_gpu_result_publication_peak_bytes,
                native_gpu_dynamic_metadata_peak_bytes,
                (uint64_t)SIZE_MAX, &proved_resident)) {
            snprintf(
                evidence, evidence_capacity,
                "required=native-gpu-resident-capacity;"
                "backend=%s;resident_base=%" PRIu64
                ";result_peak=%" PRIu64 ";plan_metadata_peak=%" PRIu64
                ";resident_limit=%zu",
                registry_backend,
                maximum_resident,
                native_gpu_result_publication_peak_bytes,
                native_gpu_dynamic_metadata_peak_bytes, SIZE_MAX);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident = proved_resident;
        digest = vx_revision_update(
            digest, &native_gpu_result_snapshot_bytes,
            sizeof(native_gpu_result_snapshot_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_result_publication_peak_bytes,
            sizeof(native_gpu_result_publication_peak_bytes));
        digest = vx_revision_update(
            digest, &native_gpu_dynamic_metadata_peak_bytes,
            sizeof(native_gpu_dynamic_metadata_peak_bytes));
    }
    if (dynamic && backend == VOLVOXAI_BACKEND_CUDA) {
        uint64_t one_d_grid;
        one_d_grid = maximum_launch_elements / 256u +
            (maximum_launch_elements % 256u != 0u);
        if (native_gpu_limits.maximum_block[0] < 256u ||
            native_gpu_limits.maximum_threads_per_block < 256u ||
            one_d_grid > native_gpu_limits.maximum_grid[0]) {
            snprintf(
                evidence, evidence_capacity,
                "required=cuda-device-domain-capacity;"
                "launch_elements=%" PRIu64 ";grid_x=%" PRIu32,
                maximum_launch_elements,
                native_gpu_limits.maximum_grid[0]);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
    }
    root = model->logical_graph;
    if (!root) {
        status = VX_STATUS_IO_ERROR;
        goto done;
    }
    nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
    status = vx_graph_affine_concat_domain_proof(
        model, tensors, tensor_count, &digest, &affine_relation_count,
        evidence, evidence_capacity);
    if (status != VX_STATUS_OK) goto done;
    for (const cJSON* node = cJSON_IsArray(nodes) ? nodes->child : NULL;
         node; node = node->next) {
        const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
        const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(node, "id");
        const VxGeneratedShapeContractRoute* shape_route;
        const VxKernelRegistration* backend_route;
        const char* implemented_shape_function;
        if (!cJSON_IsString(op) || !op->valuestring ||
            !(shape_route = vx_kernel_shape_contract_find(op->valuestring)) ||
            !(backend_route = vx_kernel_registry_find(
                  registry_backend, op->valuestring)) ||
            !shape_route->shape_function_id ||
            !shape_route->shape_function_id[0] ||
            !backend_route->route || !backend_route->route[0] ||
            !backend_route->predicate_id || !backend_route->predicate_id[0] ||
            backend_route->operator_kind != shape_route->operator_kind) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        implemented_shape_function =
            vx_shape_contract_function_id(op->valuestring);
        if (shape_route->classification !=
                VX_SHAPE_CONTRACT_CLASSIFICATION_CANONICAL ||
            !implemented_shape_function ||
            strcmp(implemented_shape_function,
                   shape_route->shape_function_id) ||
            !backend_route->exporter_qualified) {
            snprintf(
                evidence, evidence_capacity,
                "unsupported_node=%s;unsupported_op=%s;"
                "required=canonical-native-shape-contract;reexport=required",
                cJSON_IsString(node_id) && node_id->valuestring
                    ? node_id->valuestring : "<unknown>",
                op->valuestring);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (dynamic && !backend_route->dynamic) {
            snprintf(
                evidence, evidence_capacity,
                "unsupported_node=%s;unsupported_op=%s;"
                "required=dynamic-native-route;reexport=required",
                cJSON_IsString(node_id) && node_id->valuestring
                    ? node_id->valuestring : "<unknown>",
                op->valuestring);
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        if (vx_native_gpu_backend(backend)) {
            status = vx_native_gpu_node_domain_proof(
                node, tensors, tensor_count, validation, backend,
                &native_gpu_limits,
                evidence, evidence_capacity);
            if (status != VX_STATUS_OK) goto done;
        }
        digest = vx_revision_update(digest, shape_route->shape_function_id,
                                    strlen(shape_route->shape_function_id) + 1u);
        digest = vx_revision_update(digest, &shape_route->classification,
                                    sizeof(shape_route->classification));
        digest = vx_revision_update(digest, backend_route->route,
                                    strlen(backend_route->route) + 1u);
        digest = vx_revision_update(digest, backend_route->predicate_id,
                                    strlen(backend_route->predicate_id) + 1u);
        if (backend == VOLVOXAI_BACKEND_CPU &&
            !strcmp(op->valuestring, "QBatchMatMul")) {
            const size_t node_workspace =
                vx_qbatch_cpu_workspace_domain_bound(
                    node, tensors, tensor_count);
            if (node_workspace > maximum_typed_scratch)
                maximum_typed_scratch = node_workspace;
            digest = vx_revision_update(
                digest, &node_workspace, sizeof(node_workspace));
        }
        canonical_count++;
        operator_count++;
    }
    if (backend == VOLVOXAI_BACKEND_CPU) {
        static const char workspace_contract[] =
            "native.cpu.typed-workspace.qbatch-centered-i16/v1";
        if ((uint64_t)maximum_typed_scratch >
            UINT64_MAX - maximum_resident) {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
            goto done;
        }
        maximum_resident += (uint64_t)maximum_typed_scratch;
        digest = vx_revision_update(
            digest, workspace_contract, sizeof(workspace_contract));
        digest = vx_revision_update(
            digest, &maximum_typed_scratch, sizeof(maximum_typed_scratch));
    }
    if (vx_native_gpu_backend(backend)) {
        /* The public route-evidence ABI is fixed at 512 bytes. Backend and
         * capacity fields are more actionable here than repeating the equal
         * canonical/native-route counts already implied by domain_route=all.
         * Keep every attested field in one bounded record; ordering the
         * dispatch, result, and plan peaks before the digest prevents silent
         * truncation when the built-in route prefix is prepended. */
        snprintf(evidence, evidence_capacity,
                 "shape_proof=%s;domain_route=all;dynamic=%d;"
                 "native_gpu_backend=%s"
                ";native_gpu_fixed_scratch=%" PRIu64 "/%" PRIu64
                "/%" PRIu64
                ";native_gpu_domain_spans=%" PRIu64
                ";native_gpu_domain_bytes=%" PRIu64
                ";native_gpu_storage_alignment=%" PRIu64
                 ";native_gpu_result_peak=%" PRIu64
                 ";native_gpu_plan_metadata_peak=%" PRIu64
                 ";max_resident=%" PRIu64
                 ";domain_digest=%016" PRIx64,
                model->shape_domain_proof_identity, dynamic,
                registry_backend,
                native_gpu_scratch_required_bytes,
                native_gpu_limits.current_graph_scratch_bytes,
                native_gpu_limits.maximum_scratch_bytes,
                native_gpu_span_count,
                native_gpu_arena_bytes,
                native_gpu_limits.storage_alignment,
                native_gpu_result_publication_peak_bytes,
                native_gpu_dynamic_metadata_peak_bytes,
                maximum_resident, digest);
    } else {
        snprintf(evidence, evidence_capacity,
                 "shape_proof=%s;domain_route=all;dynamic=%d;"
                 "operators=%zu;canonical=%zu;native_routes=%zu;"
                 "affine_relations=%zu;"
                 "max_tensor=%" PRIu64 ";max_weights=%" PRIu64
                 ";max_typed_scratch=%zu;max_resident=%" PRIu64
                 ";domain_digest=%016" PRIx64,
                 model->shape_domain_proof_identity, dynamic,
                 operator_count, canonical_count, operator_count,
                 affine_relation_count, model->maximum_tensor_bytes,
                 maximum_weight_bytes, maximum_typed_scratch,
                 maximum_resident, digest);
    }
    resources->maximum_typed_scratch_bytes = maximum_typed_scratch;
    resources->maximum_resident_bytes = maximum_resident;
    status = VX_STATUS_OK;
done:
    vx_declared_outputs_free(tensors, tensor_count);
    return status;
}

typedef struct VxShapeParamStorage {
    VxShapeParam* values;
    double** arrays;
    size_t count;
} VxShapeParamStorage;

static void vx_shape_param_storage_clear(VxShapeParamStorage* storage) {
    if (!storage) return;
    for (size_t index = 0; index < storage->count; index++)
        free(storage->arrays ? storage->arrays[index] : NULL);
    free(storage->arrays);
    free(storage->values);
    memset(storage, 0, sizeof(*storage));
}

static const VxShapeTensorDescriptor* vx_shape_param_declared_output(
        const VxShapeNamedTensor* declared_outputs,
        size_t declared_output_count) {
    const VxShapeTensorDescriptor* output = NULL;
    for (size_t index = 0; index < declared_output_count; index++) {
        if (!declared_outputs[index].name ||
            strcmp(declared_outputs[index].name, "out")) continue;
        if (output) return NULL;
        output = &declared_outputs[index].descriptor;
    }
    return output;
}

static int vx_shape_params_materialize(
        const cJSON* object,
        const char* operator_name,
        const VxShapeNamedTensor* declared_outputs,
        size_t declared_output_count,
        VxShapeParamStorage* storage) {
    size_t count;
    size_t index = 0;
    if (!cJSON_IsObject(object) || !storage) return -1;
    memset(storage, 0, sizeof(*storage));
    count = (size_t)cJSON_GetArraySize(object);
    storage->values = count
        ? (VxShapeParam*)calloc(count, sizeof(*storage->values)) : NULL;
    storage->arrays = count
        ? (double**)calloc(count, sizeof(*storage->arrays)) : NULL;
    storage->count = count;
    if (count && (!storage->values || !storage->arrays)) return -2;
    for (const cJSON* item = object->child; item;
         item = item->next, index++) {
        VxShapeParam* value = &storage->values[index];
        if (!item->string || !item->string[0]) return -1;
        value->name = item->string;
        if (cJSON_IsNumber(item)) {
            value->kind = VX_SHAPE_PARAM_NUMBER;
            value->value.number = item->valuedouble;
        } else if (cJSON_IsBool(item)) {
            value->kind = VX_SHAPE_PARAM_BOOLEAN;
            value->value.boolean = cJSON_IsTrue(item) ? 1 : 0;
        } else if (cJSON_IsString(item) && item->valuestring) {
            value->kind = VX_SHAPE_PARAM_STRING;
            value->value.string = item->valuestring;
        } else if (cJSON_IsArray(item)) {
            size_t array_count = (size_t)cJSON_GetArraySize(item);
            size_t array_index = 0;
            const VxShapeTensorDescriptor* declared =
                vx_shape_param_declared_output(
                    declared_outputs, declared_output_count);
            int symbolic_target = operator_name && item->string &&
                !strcmp(item->string, "shape") &&
                (!strcmp(operator_name, "Reshape") ||
                 !strcmp(operator_name, "Expand"));
            double* numbers = array_count
                ? (double*)malloc(array_count * sizeof(*numbers)) : NULL;
            if (array_count && !numbers) return -2;
            storage->arrays[index] = numbers;
            for (const cJSON* element = item->child; element;
                 element = element->next, array_index++) {
                if (cJSON_IsNumber(element)) {
                    numbers[array_index] = element->valuedouble;
                } else if (cJSON_IsString(element) &&
                           element->valuestring && symbolic_target &&
                           declared && declared->shape &&
                           declared->rank == array_count &&
                           array_index < declared->rank &&
                           vx_shape_symbol_valid(element->valuestring) &&
                           declared->shape[array_index] > 0u &&
                           declared->shape[array_index] <=
                               VX_SHAPE_CONTRACT_MAX_SAFE_INTEGER) {
                    uint64_t extent = declared->shape[array_index];
                    size_t prior_index = 0;
                    for (const cJSON* prior = item->child;
                         prior && prior != element;
                         prior = prior->next, prior_index++) {
                        if (cJSON_IsString(prior) && prior->valuestring &&
                            !strcmp(prior->valuestring,
                                    element->valuestring) &&
                            numbers[prior_index] != (double)extent)
                            return -1;
                    }
                    numbers[array_index] = (double)extent;
                } else {
                    return -1;
                }
            }
            value->kind = VX_SHAPE_PARAM_NUMBER_ARRAY;
            value->value.number_array.count = array_count;
            value->value.number_array.values = numbers;
        } else {
            return -1;
        }
    }
    return 0;
}

static uint64_t vx_logical_tensor_name_hash(const char* name) {
    uint64_t hash = UINT64_C(14695981039346656037);
    if (!name) return 0;
    while (*name) {
        hash ^= (unsigned char)*name++;
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static VxStatus vx_logical_tensor_name_index_build(
        VxExecutionContext* context) {
    size_t capacity = 1u;
    size_t* slots;
    if (!context || context->logical_tensor_name_slots ||
        context->logical_tensor_name_capacity) return VX_STATUS_INVALID_ARGUMENT;
    if (!context->logical_tensor_count) return VX_STATUS_OK;
    if (context->logical_tensor_count > SIZE_MAX / 2u)
        return VX_STATUS_OUT_OF_MEMORY;
    while (capacity < context->logical_tensor_count * 2u) {
        if (capacity > SIZE_MAX / 2u) return VX_STATUS_OUT_OF_MEMORY;
        capacity *= 2u;
    }
    if (capacity > SIZE_MAX / sizeof(*slots)) return VX_STATUS_OUT_OF_MEMORY;
    slots = (size_t*)calloc(capacity, sizeof(*slots));
    if (!slots) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const char* name = context->logical_tensors[index].name;
        size_t slot;
        if (!name || !name[0]) {
            free(slots);
            return VX_STATUS_INVALID_GRAPH;
        }
        slot = (size_t)vx_logical_tensor_name_hash(name) & (capacity - 1u);
        while (slots[slot]) {
            if (!strcmp(context->logical_tensors[slots[slot] - 1u].name,
                        name)) {
                free(slots);
                return VX_STATUS_INVALID_GRAPH;
            }
            slot = (slot + 1u) & (capacity - 1u);
        }
        slots[slot] = index + 1u;
    }
    if (context->allocated_bytes > SIZE_MAX - capacity * sizeof(*slots)) {
        free(slots);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    context->logical_tensor_name_slots = slots;
    context->logical_tensor_name_capacity = capacity;
    context->allocated_bytes += capacity * sizeof(*slots);
    return VX_STATUS_OK;
}

static size_t vx_logical_tensor_index(const VxExecutionContext* context,
                                      const char* name) {
    if (!context || !name) return SIZE_MAX;
    if (context->logical_tensor_name_slots &&
        context->logical_tensor_name_capacity) {
        size_t slot = (size_t)vx_logical_tensor_name_hash(name) &
            (context->logical_tensor_name_capacity - 1u);
        for (size_t probe = 0;
             probe < context->logical_tensor_name_capacity; probe++) {
            size_t encoded = context->logical_tensor_name_slots[slot];
            if (!encoded) return SIZE_MAX;
            if (!strcmp(context->logical_tensors[encoded - 1u].name, name))
                return encoded - 1u;
            slot = (slot + 1u) &
                (context->logical_tensor_name_capacity - 1u);
        }
        return SIZE_MAX;
    }
    for (size_t index = 0; index < context->logical_tensor_count; index++)
        if (!strcmp(context->logical_tensors[index].name, name)) return index;
    return SIZE_MAX;
}

/* Logical tensors are input-first and then topological node-output order. The
 * first declaration of a symbol is therefore its canonical representative:
 * all inputs have already been checked for equality by input staging, and
 * every later inferred output is checked against this representative before
 * it is committed. Looking past the first declaration only repeats that same
 * equality proof and makes each dynamic bind quadratic in graph size. */
static int vx_resolved_symbol_from_partial(
        const VxExecutionContext* context,
        const VolvoxAIEngineResolvedTensor* resolved,
        const char* symbol,
        int64_t* out_extent) {
    if (!context || !resolved || !symbol || !out_extent) return -1;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const VxDeclaredTensor* logical = &context->logical_tensors[index];
        for (uint32_t axis = 0; axis < logical->rank; axis++) {
            if (!logical->symbols[axis] ||
                strcmp(logical->symbols[axis], symbol)) continue;
            if (resolved[index].shape[axis] <= 0) return 0;
            *out_extent = resolved[index].shape[axis];
            return 1;
        }
    }
    return 0;
}

static void vx_shape_quantization_from_private(
        const VxExecutionContext* context,
        size_t node_index,
        const char* input_port,
        const char* tensor_name,
        VxShapeTensorDescriptor* descriptor) {
    T* tensor = tensor_name ? t_find(tensor_name) : NULL;
    if (!tensor || !descriptor) return;
    if (tensor->quantization.valid) {
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_TENSOR;
        descriptor->quantization.scale = tensor->quantization.scale;
        descriptor->quantization.zero_point = tensor->quantization.zero_point;
        return;
    }
    /* Per-axis weight metadata is normalized and aligned once by the private
     * engine. Reuse those central tables here instead of re-reading possibly
     * unaligned safetensors parameter bytes for every concrete shape plan. */
    if (!context || !context->engine_state || node_index >= (size_t)MAXN ||
        !input_port || strcmp(input_port, "weight")) return;
    if (context->engine_state->qlinear_metadata[node_index].valid) {
        const QLinearMetadata* metadata =
            &context->engine_state->qlinear_metadata[node_index];
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_AXIS;
        descriptor->quantization.axis = 0u;
        descriptor->quantization.count = (size_t)metadata->d_out;
        descriptor->quantization.scales = metadata->weight_scales;
        descriptor->quantization.zero_points = metadata->weight_zero_points;
    } else if (context->engine_state->qconv_metadata[node_index].valid) {
        const QConv2DMetadata* metadata =
            &context->engine_state->qconv_metadata[node_index];
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_AXIS;
        descriptor->quantization.axis = 0u;
        descriptor->quantization.count = metadata->output_channels;
        descriptor->quantization.scales = metadata->weight_scales;
        descriptor->quantization.zero_points = metadata->weight_zero_points;
    } else if (context->engine_state->qembedding_metadata[node_index].valid) {
        const QEmbeddingMetadata* metadata =
            &context->engine_state->qembedding_metadata[node_index];
        descriptor->quantization.scheme = VX_SHAPE_QUANTIZATION_PER_AXIS;
        descriptor->quantization.axis = 0u;
        descriptor->quantization.count = metadata->vocab;
        descriptor->quantization.scales = metadata->weight_scales;
        descriptor->quantization.zero_points = metadata->weight_zero_points;
    }
}

static int vx_shape_input_descriptor(
        const VxExecutionContext* context,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t node_index,
        const char* input_port,
        const char* tensor_name,
        VxShapeTensorDescriptor* descriptor,
        uint64_t shape[VX_MAX_TENSOR_RANK]) {
    size_t logical_index = vx_logical_tensor_index(context, tensor_name);
    memset(descriptor, 0, sizeof(*descriptor));
    memset(shape, 0, VX_MAX_TENSOR_RANK * sizeof(*shape));
    if (logical_index != SIZE_MAX) {
        const VolvoxAIEngineResolvedTensor* concrete =
            &resolved[logical_index];
        if (concrete->rank < 0 ||
            concrete->rank > (int)VX_MAX_TENSOR_RANK) return -1;
        descriptor->rank = (size_t)concrete->rank;
        descriptor->dtype = (VxDataType)concrete->dtype;
        for (int axis = 0; axis < concrete->rank; axis++) {
            if (concrete->shape[axis] <= 0) return -1;
            shape[axis] = (uint64_t)concrete->shape[axis];
        }
    } else {
        T* tensor = t_find(tensor_name);
        if (!tensor || tensor->ndim < 0 ||
            tensor->ndim > (int)VX_MAX_TENSOR_RANK || tensor->numel <= 0)
            return -1;
        descriptor->rank = (size_t)tensor->ndim;
        /* F16 is an immutable private storage optimization. Native v1 exposes
         * and computes it as F32, matching output snapshot conversion. */
        descriptor->dtype = tensor->dtype == T_F16
            ? VX_DTYPE_F32 : tensor->dtype;
        for (int axis = 0; axis < tensor->ndim; axis++) {
            if (tensor->shape[axis] <= 0) return -1;
            shape[axis] = (uint64_t)tensor->shape[axis];
        }
    }
    descriptor->shape = shape;
    vx_shape_quantization_from_private(
        context, node_index, input_port, tensor_name, descriptor);
    return 0;
}

static int vx_shape_declared_output_descriptor(
        const VxExecutionContext* context,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t logical_index,
        VxShapeTensorDescriptor* descriptor,
        uint64_t shape[VX_MAX_TENSOR_RANK]) {
    const VxDeclaredTensor* logical;
    if (!context || !resolved || logical_index >= context->logical_tensor_count)
        return -1;
    logical = &context->logical_tensors[logical_index];
    memset(descriptor, 0, sizeof(*descriptor));
    memset(shape, 0, VX_MAX_TENSOR_RANK * sizeof(*shape));
    descriptor->rank = logical->rank;
    descriptor->dtype = logical->dtype;
    for (uint32_t axis = 0; axis < logical->rank; axis++) {
        int64_t extent = logical->minimums[axis];
        if (logical->kinds[axis] == VX_DIMENSION_SYMBOLIC) {
            int found = vx_resolved_symbol_from_partial(
                context, resolved, logical->symbols[axis], &extent);
            if (found <= 0) return found;
        }
        if (extent <= 0) return -1;
        shape[axis] = (uint64_t)extent;
    }
    descriptor->shape = shape;
    vx_shape_quantization_from_private(
        context, SIZE_MAX, NULL, logical->name, descriptor);
    return 1;
}

static int vx_resolved_assert_output(
        const VxExecutionContext* context,
        VolvoxAIEngineResolvedTensor* resolved,
        size_t logical_index,
        const VxShapeTensorDescriptor* inferred) {
    const VxDeclaredTensor* logical;
    VolvoxAIEngineResolvedTensor* concrete;
    if (!context || !resolved || !inferred ||
        logical_index >= context->logical_tensor_count) return -1;
    logical = &context->logical_tensors[logical_index];
    concrete = &resolved[logical_index];
    if (inferred->dtype != logical->dtype || inferred->rank != logical->rank)
        return -1;
    for (uint32_t axis = 0; axis < logical->rank; axis++) {
        uint64_t raw_extent = inferred->shape[axis];
        int64_t bound_extent = 0;
        int bound;
        if (!raw_extent || raw_extent > INT_MAX ||
            raw_extent < (uint64_t)logical->minimums[axis] ||
            raw_extent > (uint64_t)logical->maximums[axis] ||
            raw_extent % (uint64_t)logical->multiples[axis]) return -1;
        if (logical->kinds[axis] == VX_DIMENSION_FIXED) {
            if (logical->symbols[axis] ||
                raw_extent != (uint64_t)logical->minimums[axis]) return -1;
        } else {
            if (!logical->symbols[axis] || !logical->symbols[axis][0]) return -1;
            bound = vx_resolved_symbol_from_partial(
                context, resolved, logical->symbols[axis], &bound_extent);
            if (bound < 0 || (bound && raw_extent != (uint64_t)bound_extent))
                return -1;
        }
        concrete->shape[axis] = (int)raw_extent;
    }
    return 0;
}

static VxStatus vx_resolve_node_shapes(
        const VxExecutionContext* context,
        const cJSON* node,
        size_t node_index,
        VolvoxAIEngineResolvedTensor* resolved) {
    const cJSON* op = cJSON_GetObjectItemCaseSensitive(node, "opType");
    const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
    const cJSON* outputs = cJSON_GetObjectItemCaseSensitive(node, "outputs");
    const cJSON* params = cJSON_GetObjectItemCaseSensitive(node, "params");
    const char* shape_function;
    VxShapeNamedTensor request_inputs[MAXIN] = {0};
    VxShapeNamedTensor request_outputs[MAXIN] = {0};
    uint64_t input_shapes[MAXIN * VX_MAX_TENSOR_RANK] = {0};
    uint64_t output_shapes[MAXIN * VX_MAX_TENSOR_RANK] = {0};
    size_t output_indices[MAXIN] = {0};
    VxShapeParamStorage param_storage = {0};
    VxConcreteShapeRequest request = {0};
    VxConcreteShapeResult result = VX_CONCRETE_SHAPE_RESULT_INITIALIZER;
    VxShapeContractError error = {0};
    size_t input_count;
    size_t output_count;
    size_t index;
    int declarations_bound = 1;
    VxStatus status = VX_STATUS_INVALID_GRAPH;
    if (!cJSON_IsString(op) || !op->valuestring ||
        !(shape_function = vx_shape_contract_function_id(op->valuestring)) ||
        !cJSON_IsObject(inputs) || !cJSON_IsObject(outputs) ||
        !cJSON_IsObject(params)) return VX_STATUS_BACKEND_UNSUPPORTED;
    input_count = (size_t)cJSON_GetArraySize(inputs);
    output_count = (size_t)cJSON_GetArraySize(outputs);
    if (!input_count || !output_count || input_count > MAXIN ||
        output_count > MAXIN) return VX_STATUS_INVALID_GRAPH;
    index = 0;
    for (const cJSON* input = inputs->child; input;
         input = input->next, index++) {
        if (!input->string || !input->string[0] ||
            !cJSON_IsString(input) || !input->valuestring ||
            vx_shape_input_descriptor(
                context, resolved, node_index, input->string,
                input->valuestring,
                &request_inputs[index].descriptor,
                input_shapes + index * VX_MAX_TENSOR_RANK) != 0)
            goto done;
        request_inputs[index].name = input->string;
    }
    index = 0;
    for (const cJSON* output = outputs->child; output;
         output = output->next, index++) {
        const cJSON* tensor =
            cJSON_GetObjectItemCaseSensitive(output, "tensor");
        int descriptor_status;
        if (!output->string || !output->string[0] ||
            !cJSON_IsString(tensor) || !tensor->valuestring ||
            (output_indices[index] = vx_logical_tensor_index(
                 context, tensor->valuestring)) == SIZE_MAX) goto done;
        request_outputs[index].name = output->string;
        descriptor_status = vx_shape_declared_output_descriptor(
            context, resolved, output_indices[index],
            &request_outputs[index].descriptor,
            output_shapes + index * VX_MAX_TENSOR_RANK);
        if (descriptor_status < 0) goto done;
        if (!descriptor_status) declarations_bound = 0;
    }
    {
        int params_status = vx_shape_params_materialize(
            params, op->valuestring,
            declarations_bound ? request_outputs : NULL,
            declarations_bound ? output_count : 0u,
            &param_storage);
        if (params_status != 0) {
            status = params_status == -2 ? VX_STATUS_OUT_OF_MEMORY :
                                           VX_STATUS_INVALID_GRAPH;
            goto done;
        }
    }
    request.inputs = request_inputs;
    request.input_count = input_count;
    request.params = param_storage.values;
    request.param_count = param_storage.count;
    request.declared_outputs = declarations_bound ? request_outputs : NULL;
    request.declared_output_count = declarations_bound ? output_count : 0u;
    if (vx_shape_contract_infer(op->valuestring, &request, &result, &error) != 0) {
        if (context->engine_state && context->engine_state->debug)
            fprintf(stderr,
                    "[debug] shape contract failed op=%s code=%d path=%s message=%s\n",
                    op->valuestring, (int)error.code,
                    error.path[0] ? error.path : "<none>",
                    error.detail[0] ? error.detail : "<none>");
        status = error.code == VX_SHAPE_CONTRACT_ERROR_OUT_OF_MEMORY
            ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_INVALID_GRAPH;
        goto done;
    }
    if (!result.shape_function_id ||
        strcmp(result.shape_function_id, shape_function) ||
        result.output_count != output_count) goto done;
    for (index = 0; index < output_count; index++) {
        const char* port = request_outputs[index].name;
        const VxShapeTensorDescriptor* inferred = NULL;
        for (size_t candidate = 0; candidate < result.output_count;
             candidate++) {
            if (result.outputs[candidate].name &&
                !strcmp(result.outputs[candidate].name, port)) {
                inferred = &result.outputs[candidate].descriptor;
                break;
            }
        }
        if (!inferred || vx_resolved_assert_output(
                context, resolved, output_indices[index], inferred) != 0) {
            if (context->engine_state && context->engine_state->debug)
                fprintf(stderr,
                        "[debug] inferred output assertion failed op=%s port=%s tensor=%s\n",
                        op->valuestring, port,
                        context->logical_tensors[output_indices[index]].name);
            goto done;
        }
    }
    status = VX_STATUS_OK;
done:
    vx_shape_contract_result_clear(&result);
    vx_shape_param_storage_clear(&param_storage);
    return status;
}

static VxStatus vx_resolve_graph_shapes(
        const VxExecutionContext* context,
        VolvoxAIEngineResolvedTensor* resolved) {
    const cJSON* nodes;
    if (!context || !context->compiled || !context->compiled->model ||
        !resolved || !context->compiled->model->logical_graph)
        return VX_STATUS_INVALID_ARGUMENT;
    nodes = cJSON_GetObjectItemCaseSensitive(
        context->compiled->model->logical_graph, "nodes");
    if (!cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    {
        size_t node_index = 0;
        for (const cJSON* node = nodes->child; node;
             node = node->next, node_index++) {
            VxStatus status = vx_resolve_node_shapes(
                context, node, node_index, resolved);
            if (status != VX_STATUS_OK) {
                if (context->engine_state && context->engine_state->debug) {
                    const cJSON* node_id = cJSON_GetObjectItemCaseSensitive(
                        node, "id");
                    const cJSON* op = cJSON_GetObjectItemCaseSensitive(
                        node, "opType");
                    fprintf(stderr,
                            "[debug] graph shape resolution stopped node=%zu id=%s op=%s status=%d\n",
                            node_index,
                            cJSON_IsString(node_id) && node_id->valuestring
                                ? node_id->valuestring : "<unknown>",
                            cJSON_IsString(op) && op->valuestring
                                ? op->valuestring : "<unknown>",
                            (int)status);
                }
                return status;
            }
        }
    }
    for (size_t index = 0; index < context->logical_tensor_count; index++)
        for (int axis = 0; axis < resolved[index].rank; axis++)
            if (resolved[index].shape[axis] <= 0)
                return VX_STATUS_INVALID_GRAPH;
    return VX_STATUS_OK;
}

static VxStatus vx_resolved_plan_create(
        const VxExecutionContext* context,
        const int64_t* input_shapes,
        VolvoxAIEngineResolvedTensor** out_tensors,
        size_t* out_tensor_count,
        char** out_signature) {
    const VxModel* model;
    VolvoxAIEngineResolvedTensor* resolved = NULL;
    char* signature = NULL;
    size_t signature_capacity = 4u;
    size_t signature_offset = 0;
    if (!context || (!input_shapes && context->compiled->model->input_count) ||
        !out_tensors || !out_tensor_count ||
        !out_signature) return VX_STATUS_INVALID_ARGUMENT;
    *out_tensors = NULL;
    *out_tensor_count = 0;
    *out_signature = NULL;
    model = context->compiled->model;
    for (size_t input = 0; input < model->input_count; input++) {
        size_t name_length = strlen(model->inputs[input].name);
        if (name_length > SIZE_MAX - signature_capacity - 4u ||
            model->inputs[input].rank >
                (SIZE_MAX - signature_capacity - name_length - 4u) / 24u)
            return VX_STATUS_OUT_OF_MEMORY;
        signature_capacity += name_length + 4u +
            (size_t)model->inputs[input].rank * 24u;
    }
    signature = (char*)malloc(signature_capacity);
    resolved = (VolvoxAIEngineResolvedTensor*)calloc(
        context->logical_tensor_count, sizeof(*resolved));
    if (!signature || !resolved) {
        free(signature);
        free(resolved);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    signature_offset = (size_t)snprintf(signature, signature_capacity, "v1;");
    {
        const char* previous = NULL;
        for (size_t position = 0; position < model->input_count; position++) {
            size_t input = SIZE_MAX;
            int written;
            for (size_t candidate = 0; candidate < model->input_count;
                 candidate++) {
                const char* name = model->inputs[candidate].name;
                if ((previous && strcmp(name, previous) <= 0) ||
                    (input != SIZE_MAX &&
                     strcmp(name, model->inputs[input].name) >= 0)) continue;
                input = candidate;
            }
            if (input == SIZE_MAX) goto oom;
            written = snprintf(signature + signature_offset,
                               signature_capacity - signature_offset,
                               "%s[", model->inputs[input].name);
        if (written < 0 || (size_t)written >=
                             signature_capacity - signature_offset) goto oom;
        signature_offset += (size_t)written;
        for (uint32_t axis = 0; axis < model->inputs[input].rank; axis++) {
            written = snprintf(
                signature + signature_offset,
                signature_capacity - signature_offset,
                "%s%" PRId64,
                axis ? "," : "",
                input_shapes[input * VX_MAX_TENSOR_RANK + axis]);
            if (written < 0 || (size_t)written >=
                                 signature_capacity - signature_offset) goto oom;
            signature_offset += (size_t)written;
        }
        written = snprintf(signature + signature_offset,
                           signature_capacity - signature_offset, "];");
        if (written < 0 || (size_t)written >=
                             signature_capacity - signature_offset) goto oom;
        signature_offset += (size_t)written;
            previous = model->inputs[input].name;
        }
    }
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const VxDeclaredTensor* logical = &context->logical_tensors[index];
        VolvoxAIEngineResolvedTensor* concrete = &resolved[index];
        concrete->name = logical->name;
        concrete->dtype = (int)logical->dtype;
        concrete->rank = (int)logical->rank;
    }
    for (size_t input = 0; input < model->input_count; input++) {
        size_t logical_index = vx_logical_tensor_index(
            context, model->inputs[input].name);
        if (logical_index == SIZE_MAX) goto invalid;
        for (uint32_t axis = 0; axis < model->inputs[input].rank; axis++) {
            int64_t extent =
                input_shapes[input * VX_MAX_TENSOR_RANK + axis];
            if (extent <= 0 || extent > INT_MAX) goto invalid;
            resolved[logical_index].shape[axis] = (int)extent;
        }
    }
    {
        VxStatus shape_status = vx_resolve_graph_shapes(context, resolved);
        if (shape_status != VX_STATUS_OK) {
            free(signature);
            free(resolved);
            return shape_status;
        }
    }
    *out_tensors = resolved;
    *out_tensor_count = context->logical_tensor_count;
    *out_signature = signature;
    return VX_STATUS_OK;
invalid:
    free(signature);
    free(resolved);
    return VX_STATUS_INVALID_GRAPH;
oom:
    free(signature);
    free(resolved);
    return VX_STATUS_OUT_OF_MEMORY;
}

static char* vx_string_copy(const char* value) {
    size_t length;
    char* copy;
    if (!value) return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1u);
    if (copy) memcpy(copy, value, length + 1u);
    return copy;
}

static void vx_model_bank_residency_clear(VxModel* model) {
    if (!model) return;
    if (model->bank_residency_slots) {
        for (size_t index = 0; index < model->bank_residency_count; index++)
            free(model->bank_residency_slots[index]);
    }
    free(model->bank_residency_slots);
    free(model->bank_residency);
    model->bank_residency_slots = NULL;
    model->bank_residency = NULL;
    model->bank_residency_count = 0;
}

static void vx_weight_revision_retain(VxWeightRevisionRecord* revision) {
    if (revision)
        atomic_fetch_add_explicit(&revision->references, 1,
                                  memory_order_relaxed);
}

static void vx_weight_revision_release(VxWeightRevisionRecord* revision) {
    if (!revision || atomic_fetch_sub_explicit(&revision->references, 1,
                                               memory_order_acq_rel) != 1)
        return;
    for (size_t index = 0; index < revision->path_count; index++)
        vx_snapshot_path_release(revision->paths[index]);
    free(revision->paths);
    free(revision);
}

static void vx_adapter_revision_retain(VxAdapterRevisionRecord* revision) {
    if (revision)
        atomic_fetch_add_explicit(&revision->references, 1,
                                  memory_order_relaxed);
}

static void vx_adapter_revision_release(VxAdapterRevisionRecord* revision) {
    if (!revision || atomic_fetch_sub_explicit(&revision->references, 1,
                                               memory_order_acq_rel) != 1)
        return;
    free(revision->name);
    vx_snapshot_path_release(revision->package_path);
    free(revision->version_name);
    free(revision);
}

static VxWeightRevisionRecord* vx_weight_revision_create(
    const char* const* paths,
    size_t path_count,
    uint64_t identity,
    uint64_t revision,
    VxStatus* out_status) {
    VxWeightRevisionRecord* record;
    uint64_t content_hash = UINT64_C(1469598103934665603);
    if (out_status) *out_status = VX_STATUS_INVALID_ARGUMENT;
    if (path_count > (size_t)MAX_WEIGHT_FILES || (path_count && !paths))
        return NULL;
    record = (VxWeightRevisionRecord*)calloc(1, sizeof(*record));
    if (!record) {
        if (out_status) *out_status = VX_STATUS_OUT_OF_MEMORY;
        return NULL;
    }
    atomic_init(&record->references, 1);
    record->identity = identity;
    record->revision = revision;
    content_hash = vx_revision_update(content_hash, &path_count,
                                      sizeof(path_count));
    if (path_count)
        record->paths = (char**)calloc(path_count, sizeof(*record->paths));
    if (path_count && !record->paths) goto oom;
    for (size_t index = 0; index < path_count; index++) {
        uint64_t file_hash;
        VxStatus snapshot_status;
        if (!paths[index] || !paths[index][0]) {
            if (out_status) *out_status = VX_STATUS_INVALID_ARGUMENT;
            vx_weight_revision_release(record);
            return NULL;
        }
        snapshot_status = vx_file_snapshot(paths[index], &record->paths[index],
                                           &file_hash);
        if (snapshot_status != VX_STATUS_OK) {
            if (out_status) *out_status = snapshot_status;
            vx_weight_revision_release(record);
            return NULL;
        }
        record->path_count = index + 1u;
        content_hash = vx_revision_update(content_hash, &index, sizeof(index));
        content_hash = vx_revision_update(content_hash, &file_hash,
                                          sizeof(file_hash));
    }
    record->path_count = path_count;
    record->content_hash = content_hash ? content_hash : UINT64_C(1);
    record->allocated_bytes = sizeof(*record) +
        path_count * sizeof(*record->paths);
    for (size_t index = 0; index < path_count; index++)
        record->allocated_bytes += strlen(record->paths[index]) + 1u;
    if (out_status) *out_status = VX_STATUS_OK;
    return record;
oom:
    if (out_status) *out_status = VX_STATUS_OUT_OF_MEMORY;
    vx_weight_revision_release(record);
    return NULL;
}

static VxAdapterRevisionRecord* vx_adapter_revision_create(
    const char* name,
    const char* package_path,
    const char* version_name,
    uint64_t identity,
    uint64_t revision,
    VxStatus* out_status) {
    VxAdapterRevisionRecord* record;
    VxStatus snapshot_status = VX_STATUS_OK;
    uint64_t content_hash = UINT64_C(1);
    if (out_status) *out_status = VX_STATUS_OUT_OF_MEMORY;
    if (!name || !name[0]) return NULL;
    record = (VxAdapterRevisionRecord*)calloc(1, sizeof(*record));
    if (!record) return NULL;
    atomic_init(&record->references, 1);
    record->identity = identity;
    record->revision = revision;
    record->name = vx_string_copy(name);
    if (package_path)
        snapshot_status = vx_file_snapshot(package_path, &record->package_path,
                                           &content_hash);
    record->content_hash = content_hash;
    record->version_name = version_name ? vx_string_copy(version_name) : NULL;
    if (snapshot_status != VX_STATUS_OK || !record->name ||
        (package_path && !record->package_path) ||
        (version_name && !record->version_name)) {
        if (out_status) *out_status = snapshot_status != VX_STATUS_OK
            ? snapshot_status : VX_STATUS_OUT_OF_MEMORY;
        vx_adapter_revision_release(record);
        return NULL;
    }
    record->allocated_bytes = sizeof(*record) + strlen(record->name) + 1u;
    if (record->package_path)
        record->allocated_bytes += strlen(record->package_path) + 1u;
    if (record->version_name)
        record->allocated_bytes += strlen(record->version_name) + 1u;
    if (out_status) *out_status = VX_STATUS_OK;
    return record;
}

static void vx_report_write(VxReport* report,
                            VxStatus status,
                            VxStage stage,
                            const char* backend,
                            const char* device,
                            const char* reason,
                            const char* message,
                            uint64_t execution_id) {
    size_t struct_size;
    if (!report || report->struct_size != sizeof(*report)) return;
    struct_size = report->struct_size;
    memset(report, 0, sizeof(*report));
    report->struct_size = struct_size;
    report->status = status;
    report->stage = stage;
    report->execution_id = execution_id;
    snprintf(report->backend, sizeof(report->backend), "%s", backend ? backend : "");
    snprintf(report->device, sizeof(report->device), "%s", device ? device : "");
    snprintf(report->reason, sizeof(report->reason), "%s", reason ? reason : "");
    snprintf(report->message, sizeof(report->message), "%s", message ? message : "");
}

static int vx_report_valid(const VxReport* report) {
    return !report || report->struct_size == sizeof(*report);
}

static void vx_report_set_lineage(VxReport* report,
                                  const VxRuntime* runtime,
                                  const VxModel* model,
                                  const VxCompiledModel* compiled,
                                  const VxExecutionContext* context) {
    const VxWeightRevisionRecord* weights = NULL;
    const VxAdapterRevisionRecord* adapter = NULL;
    uint64_t allocated_bytes = 0;
    if (!report || report->struct_size != sizeof(*report)) return;
    if (context && !compiled) compiled = context->compiled;
    if (compiled && !model) model = compiled->model;
    if (model && !runtime) runtime = model->runtime;
    if (runtime) {
        report->runtime_id = runtime->identity;
        allocated_bytes += sizeof(*runtime);
    }
    if (model) {
        report->model_id = model->identity;
        report->graph_id = model->graph_identity;
        report->graph_revision = model->graph_revision;
        allocated_bytes += model->allocated_bytes;
    }
    if (compiled) {
        weights = compiled->weights;
        adapter = compiled->adapter;
        report->compiled_model_id = compiled->identity;
        report->policy_mode = compiled->mode;
        report->operator_fallback = compiled->operator_fallback;
        allocated_bytes += compiled->allocated_bytes;
    }
    if (context) {
        adapter = context->adapter;
        report->context_id = context->identity;
        allocated_bytes += context->allocated_bytes;
    }
    if (model && !weights) {
        pthread_mutex_lock((pthread_mutex_t*)&model->revision_mutex);
        weights = atomic_load_explicit(&model->current_weights,
                                       memory_order_acquire);
        adapter = atomic_load_explicit(&model->current_adapter,
                                       memory_order_acquire);
        if (weights) {
            report->weight_id = weights->identity;
            report->weight_revision = weights->revision;
            allocated_bytes += weights->allocated_bytes;
        }
        if (adapter) {
            report->adapter_id = adapter->identity;
            report->adapter_revision = adapter->revision;
            allocated_bytes += adapter->allocated_bytes;
        }
        pthread_mutex_unlock((pthread_mutex_t*)&model->revision_mutex);
    } else {
        if (weights) {
            report->weight_id = weights->identity;
            report->weight_revision = weights->revision;
            allocated_bytes += weights->allocated_bytes;
        }
        if (adapter) {
            report->adapter_id = adapter->identity;
            report->adapter_revision = adapter->revision;
            allocated_bytes += adapter->allocated_bytes;
        }
    }
    report->allocated_bytes = allocated_bytes;
}

static int vx_route_label_matches_backend(const char* route,
                                          const char* backend) {
    size_t length;
    if (!route || !backend) return 0;
    length = strlen(backend);
    return !strncmp(route, backend, length) &&
        (route[length] == '\0' || route[length] == '-' || route[length] == '+');
}

static void vx_report_set_builtin_route(VxReport* report,
                                        const VxEngineState* state,
                                        const char* backend) {
    uint64_t digest = UINT64_C(1469598103934665603);
    int node_count = state ? state->node_count : 0;
    int active_nodes = 0;
    int selected_nodes = 0;
    int fallback_nodes = 0;
    int missing_nodes = 0;
    int first_fallback = -1;
    if (!report || report->struct_size != sizeof(*report)) return;
    for (int index = 0; index < node_count; index++) {
        const Node* node = &state->nodes[index];
        const char* route = state->runtime_route_backend[index];
        digest = vx_revision_update(digest, &index, sizeof(index));
        digest = vx_revision_update(digest, node->op, strlen(node->op) + 1u);
        digest = vx_revision_update(digest, node->out, strlen(node->out) + 1u);
        digest = vx_revision_update(digest, &node->disabled, sizeof(node->disabled));
        digest = vx_revision_update(digest, &node->skip, sizeof(node->skip));
        if (node->disabled || node->skip) continue;
        active_nodes++;
        /* CPU compilation does not execute a validation forward: every active
         * node has only the CPU registry route in that configuration. */
        if (backend && !strcmp(backend, "cpu")) route = "cpu";
        if (!route) {
            missing_nodes++;
            continue;
        }
        digest = vx_revision_update(digest, route, strlen(route) + 1u);
        if (vx_route_label_matches_backend(route, backend)) {
            selected_nodes++;
        } else {
            fallback_nodes++;
            if (first_fallback < 0) first_fallback = index;
        }
    }
    report->route_attested = missing_nodes == 0;
    report->operator_fallback_used = fallback_nodes != 0;
    snprintf(report->route_evidence, sizeof(report->route_evidence),
             "provider=builtin:%s;nodes=%d;selected=%d;fallback=%d;missing=%d;"
             "digest=%016" PRIx64,
             backend ? backend : "unknown", active_nodes, selected_nodes,
             fallback_nodes, missing_nodes, digest);
    if (fallback_nodes) {
        const Node* first = first_fallback >= 0 ? &state->nodes[first_fallback] : NULL;
        snprintf(report->fallback_evidence, sizeof(report->fallback_evidence),
                 "operator=used;count=%d;first=index:%d,op:%s", fallback_nodes,
                 first_fallback, first ? first->op : "unknown");
        if (first)
            snprintf(report->offending_node, sizeof(report->offending_node),
                     "index=%d;op=%s;output=%s", first_fallback, first->op,
                     first->out);
    } else {
        snprintf(report->fallback_evidence, sizeof(report->fallback_evidence),
                 "%s", "operator=none");
    }
}

static void vx_report_append_dynamic_shape(VxReport* report,
                                           const VxExecutionContext* context) {
    uint64_t digest = UINT64_C(1469598103934665603);
    size_t offset;
    const char* signature;
    const VolvoxAIEngineDynamicShapeStats* stats;
    if (!report || report->struct_size != sizeof(*report) || !context ||
        !context->committed_shape_signature ||
        !vx_builtin_dynamic_shape_backend(context->compiled->backend)) return;
    signature = context->committed_shape_signature;
    stats = &context->dynamic_shape_stats;
    digest = vx_revision_update(digest, signature, strlen(signature) + 1u);
    offset = strlen(report->route_evidence);
    if (offset >= sizeof(report->route_evidence)) return;
    snprintf(report->route_evidence + offset,
             sizeof(report->route_evidence) - offset,
             ";shape_signature=%016" PRIx64 ":%.128s;shape_plan=%s;"
             "shape_bind_ms=%.3f;logical_bytes=%zu;arena_required=%zu;"
             "arena_capacity=%zu;arena_high_water=%zu;arena_grows=%" PRIu64
             ";resource_generation=%" PRIu64,
             digest, signature,
             stats->plan_cache_hit ? "hit" : "cold",
             context->dynamic_shape_bind_time_ms,
             stats->logical_bytes, stats->required_arena_bytes,
             stats->arena_capacity_bytes, stats->arena_high_water_bytes,
             stats->arena_grow_count, stats->resource_generation);
    offset = strlen(report->route_evidence);
    if (offset < sizeof(report->route_evidence)) {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        (void)vx_runtime_backend_append_dynamic_telemetry(
            report->route_evidence + offset,
            sizeof(report->route_evidence) - offset);
        vx_engine_state_scope_leave(scope);
    }
}

static VxStatus vx_builtin_compile_route_status(
    VxReport* report,
    VxOperatorFallback operator_fallback) {
    if (!report || report->struct_size != sizeof(*report))
        return VX_STATUS_INTERNAL;
    if (!report->route_attested) {
        report->status = VX_STATUS_BACKEND_UNSUPPORTED;
        snprintf(report->reason, sizeof(report->reason), "%s",
                 "BACKEND_UNSUPPORTED");
        snprintf(report->message, sizeof(report->message), "%s",
                 "built-in backend route could not be attested");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    if (report->operator_fallback_used &&
        operator_fallback == VX_OPERATOR_FALLBACK_FORBID) {
        report->status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
        snprintf(report->reason, sizeof(report->reason), "%s",
                 "OPERATOR_FALLBACK_FORBIDDEN");
        snprintf(report->message, sizeof(report->message), "%s",
                 "built-in backend route contains a CPU operator fallback");
        return VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
    }
    return VX_STATUS_OK;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
VxStatus vx_public_api_test_evaluate_builtin_route(
    const char* selected_backend,
    const char* actual_route,
    VxOperatorFallback operator_fallback,
    VxReport* report) {
    VxEngineState* state;
    VxStatus status;
    if (!selected_backend || !selected_backend[0] || !actual_route ||
        !actual_route[0] ||
        (operator_fallback != VX_OPERATOR_FALLBACK_ALLOW &&
         operator_fallback != VX_OPERATOR_FALLBACK_FORBID))
        return VX_STATUS_INVALID_ARGUMENT;
    state = (VxEngineState*)calloc(1, sizeof(*state));
    if (!state) return VX_STATUS_OUT_OF_MEMORY;
    state->node_count = 1;
    snprintf(state->nodes[0].op, sizeof(state->nodes[0].op), "%s", "Identity");
    snprintf(state->nodes[0].out, sizeof(state->nodes[0].out), "%s", "y");
    state->runtime_route_backend[0] = actual_route;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_COMPILE,
                    selected_backend, NULL, "OK", "test route", 0);
    vx_report_set_builtin_route(report, state, selected_backend);
    status = vx_builtin_compile_route_status(report, operator_fallback);
    free(state);
    return status;
}
#endif

static void vx_report_set_builtin_failure_node(VxReport* report,
                                               const VxEngineState* state) {
    int index;
    if (!report || report->struct_size != sizeof(*report) || !state) return;
    index = state->last_failure_node_index;
    if (index < 0 || index >= state->node_count) return;
    snprintf(report->offending_node, sizeof(report->offending_node),
             "index=%d;op=%s;output=%s", index, state->nodes[index].op,
             state->nodes[index].out);
}

static void vx_report_publish(VxReport* destination,
                              const VxReport* source) {
    size_t struct_size;
    if (!destination || destination->struct_size != sizeof(*destination) ||
        !source || source->struct_size != sizeof(*source)) return;
    struct_size = destination->struct_size;
    *destination = *source;
    destination->struct_size = struct_size;
}

static void vx_report_set_plan_evidence(VxReport* report,
                                        const VxCompiledModel* compiled) {
    const VxReport* plan;
    if (!report || report->struct_size != sizeof(*report) || !compiled) return;
    plan = &compiled->report;
    report->policy_mode = compiled->mode;
    report->operator_fallback = compiled->operator_fallback;
    report->candidate_count = plan->candidate_count;
    report->tier_fallback_used = plan->tier_fallback_used;
    report->operator_fallback_used = plan->operator_fallback_used;
    report->route_attested = plan->route_attested;
    snprintf(report->candidate_outcomes, sizeof(report->candidate_outcomes),
             "%s", plan->candidate_outcomes);
    snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
             plan->route_evidence);
    snprintf(report->fallback_evidence, sizeof(report->fallback_evidence), "%s",
             plan->fallback_evidence);
    snprintf(report->decode_state, sizeof(report->decode_state), "%s",
             plan->decode_state);
}

static void vx_report_add_result_allocation(VxReport* report,
                                            const VxResult* result) {
    uint64_t result_allocation;
    if (!report || report->struct_size != sizeof(*report) || !result) return;
    result_allocation = sizeof(*result);
    if (result->output_capacity <=
        (UINT64_MAX - result_allocation) / sizeof(*result->outputs))
        result_allocation += result->output_capacity * sizeof(*result->outputs);
    if (UINT64_MAX - result_allocation >= result->snapshot_bytes)
        result_allocation += result->snapshot_bytes;
    if (UINT64_MAX - report->allocated_bytes >= result_allocation)
        report->allocated_bytes += result_allocation;
}

static void vx_report_set_result_evidence(VxReport* report,
                                          const VxResult* result) {
    if (!report || report->struct_size != sizeof(*report) || !result) return;
    vx_report_set_lineage(report, NULL, NULL, NULL, result->context);
    vx_report_set_plan_evidence(report, result->context->compiled);
    report->adapter_id = result->adapter_id;
    report->adapter_revision = result->adapter_revision;
    report->operator_fallback_used = result->operator_fallback_used;
    report->route_attested = result->route_attested;
    snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
             result->route_evidence);
    snprintf(report->fallback_evidence, sizeof(report->fallback_evidence), "%s",
             result->fallback_evidence);
    snprintf(report->offending_node, sizeof(report->offending_node), "%s",
             result->offending_node);
    snprintf(report->decode_state, sizeof(report->decode_state), "%s",
             result->decode_state);
    report->execution_id = result->execution_id;
    report->result_bytes = result->snapshot_bytes;
    report->allocated_bytes = result->evidence_allocated_bytes;
}

static void vx_result_capture_evidence(VxResult* result,
                                       const VxReport* report) {
    if (!result || !report || report->struct_size != sizeof(*report)) return;
    result->evidence_allocated_bytes = report->allocated_bytes;
    result->adapter_id = report->adapter_id;
    result->adapter_revision = report->adapter_revision;
    result->operator_fallback_used = report->operator_fallback_used;
    result->route_attested = report->route_attested;
    snprintf(result->route_evidence, sizeof(result->route_evidence), "%s",
             report->route_evidence);
    snprintf(result->fallback_evidence, sizeof(result->fallback_evidence), "%s",
             report->fallback_evidence);
    snprintf(result->offending_node, sizeof(result->offending_node), "%s",
             report->offending_node);
    snprintf(result->decode_state, sizeof(result->decode_state), "%s",
             report->decode_state);
}

static VxStatus vx_fail(VxReport* report,
                        VxStatus status,
                        VxStage stage,
                        const char* backend,
                        const char* reason,
                        const char* message) {
    vx_report_write(report, status, stage, backend, NULL, reason, message, 0);
    return status;
}

const char* vx_status_string(VxStatus status) {
    switch (status) {
        case VX_STATUS_OK: return "OK";
        case VX_STATUS_INVALID_ARGUMENT: return "INVALID_ARGUMENT";
        case VX_STATUS_OUT_OF_MEMORY: return "OUT_OF_MEMORY";
        case VX_STATUS_HANDLE_DISPOSED: return "HANDLE_DISPOSED";
        case VX_STATUS_IO_ERROR: return "IO_ERROR";
        case VX_STATUS_INVALID_GRAPH: return "INVALID_GRAPH";
        case VX_STATUS_BACKEND_UNAVAILABLE: return "BACKEND_UNAVAILABLE";
        case VX_STATUS_BACKEND_UNSUPPORTED: return "BACKEND_UNSUPPORTED";
        case VX_STATUS_BACKEND_REQUIRED: return "BACKEND_REQUIRED";
        case VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN:
            return "OPERATOR_FALLBACK_FORBIDDEN";
        case VX_STATUS_EXECUTION_FAILED: return "EXECUTION_FAILED";
        case VX_STATUS_RESULT_DISPOSED: return "RESULT_DISPOSED";
        case VX_STATUS_DEVICE_LOST: return "DEVICE_LOST";
        case VX_STATUS_ABI_UNSUPPORTED: return "ABI_UNSUPPORTED";
        case VX_STATUS_NOT_FOUND: return "NOT_FOUND";
        case VX_STATUS_BUFFER_TOO_SMALL: return "BUFFER_TOO_SMALL";
        case VX_STATUS_INTERNAL: return "INTERNAL";
#if VOLVOXAI_ENABLE_TRAINING
        case (VxStatus)-17: return "REVISION_CONFLICT";
#endif
        default: return "UNKNOWN";
    }
}

static void vx_model_source_view(const VxModel* model,
                                 const VxWeightRevisionRecord* weights,
                                 VxModelSource* source) {
    *source = (VxModelSource)VX_MODEL_SOURCE_INIT;
    source->struct_size = sizeof(*source);
    source->graph_path = model->graph_path;
    source->weight_paths = weights
        ? (const char* const*)weights->paths : NULL;
    source->weight_path_count = weights ? weights->path_count : 0;
    source->bank_residency = model->bank_residency;
    source->bank_residency_count = model->bank_residency_count;
}

static void vx_declared_tensor_spec(const VxDeclaredTensor* declared,
                                    VxTensorSpec* spec) {
    memset(spec, 0, sizeof(*spec));
    spec->struct_size = sizeof(*spec);
    spec->name = declared->name;
    spec->dtype = declared->dtype;
    spec->rank = declared->rank;
    spec->location = VX_MEMORY_HOST;
    for (uint32_t axis = 0; axis < declared->rank; axis++) {
        VxDimensionConstraint* dimension = &spec->dimensions[axis];
        dimension->struct_size = sizeof(*dimension);
        dimension->kind = declared->kinds[axis];
        dimension->symbol = declared->symbols[axis];
        dimension->min = declared->minimums[axis];
        dimension->max = declared->maximums[axis];
        dimension->multiple_of = declared->multiples[axis];
    }
}

static VxTensorSpec* vx_declared_tensor_specs(
    const VxDeclaredTensor* declared,
    size_t count) {
    VxTensorSpec* specs = count
        ? (VxTensorSpec*)calloc(count, sizeof(*specs)) : NULL;
    if (count && !specs) return NULL;
    for (size_t index = 0; index < count; index++)
        vx_declared_tensor_spec(&declared[index], &specs[index]);
    return specs;
}

static int vx_provider_attestation_valid(
    const VxModel* model,
    const VxBackendShapeDomainAttestation* attestation) {
    if (!model || !attestation ||
        attestation->struct_size != sizeof(*attestation) ||
        !attestation->graph_fingerprint ||
        !attestation->shape_domain_proof_identity ||
        strcmp(attestation->graph_fingerprint, model->graph_fingerprint) ||
        strcmp(attestation->shape_domain_proof_identity,
               model->shape_domain_proof_identity) ||
        attestation->maximum_tensor_bytes != model->maximum_tensor_bytes ||
        attestation->maximum_tensor_bytes >
            attestation->maximum_resident_bytes ||
        (attestation->has_resource_limit != 0 &&
         attestation->has_resource_limit != 1) ||
        (attestation->has_resource_limit &&
         attestation->maximum_resident_bytes >
             attestation->resource_limit_bytes)) return 0;
    return 1;
}

static int vx_policy_valid(const VxBackendPolicy* policy) {
    if (!policy || policy->struct_size != sizeof(*policy) ||
        (policy->mode != VX_BACKEND_PREFER &&
         policy->mode != VX_BACKEND_REQUIRE) ||
        (policy->operator_fallback != VX_OPERATOR_FALLBACK_ALLOW &&
         policy->operator_fallback != VX_OPERATOR_FALLBACK_FORBID) ||
        policy->backend_count > VX_MAX_BACKEND_CANDIDATES ||
        (policy->backend_count && !policy->backends) ||
        (policy->mode == VX_BACKEND_REQUIRE && policy->backend_count != 1u) ||
        (policy->mode == VX_BACKEND_PREFER && policy->backends == NULL &&
         policy->backend_count != 0u) ||
        (policy->mode == VX_BACKEND_PREFER && policy->backends != NULL &&
         policy->backend_count == 0u)) return 0;
    for (size_t index = 0; index < policy->backend_count; index++) {
        const char* name = policy->backends[index];
        size_t length;
        if (!name || !name[0]) return 0;
        length = strlen(name);
        if (length >= VX_BACKEND_NAME_CAPACITY || name[0] < 'a' || name[0] > 'z')
            return 0;
        for (size_t offset = 1; offset < length; offset++) {
            unsigned char c = (unsigned char)name[offset];
            if (!((c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') ||
                  c == '-')) return 0;
        }
        for (size_t previous = 0; previous < index; previous++)
            if (!strcmp(policy->backends[previous], name)) return 0;
    }
    return 1;
}

static int vx_builtin_backend_by_name(const char* name,
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

#if VOLVOXAI_ENABLE_TRAINING
static const char* vx_builtin_backend_name(VolvoxAIEngineBackend backend) {
    switch (backend) {
        case VOLVOXAI_BACKEND_CPU: return "cpu";
        case VOLVOXAI_BACKEND_VULKAN: return "vulkan";
        case VOLVOXAI_BACKEND_OPENGL: return "opengl";
        case VOLVOXAI_BACKEND_METAL: return "metal";
        case VOLVOXAI_BACKEND_NNAPI: return "nnapi";
        case VOLVOXAI_BACKEND_CUDA: return "cuda";
        default: return NULL;
    }
}
#endif

static const char* vx_builtin_device_identity(const char* backend) {
    if (backend && !strcmp(backend, "cpu")) return "host";
    if (backend && !strcmp(backend, "vulkan")) return "builtin:vulkan";
    if (backend && !strcmp(backend, "opengl")) return "builtin:opengl";
    if (backend && !strcmp(backend, "metal")) return "builtin:metal";
    if (backend && !strcmp(backend, "nnapi")) return "builtin:nnapi";
    if (backend && !strcmp(backend, "cuda")) return "builtin:cuda";
    return "builtin:unknown";
}

static VxStatus vx_private_engine_load(VxEngineState* state,
                                       const VxModel* model,
                                       const VxWeightRevisionRecord* weights,
                                       const VxRuntimeOptions* options,
                                       VolvoxAIEngineBackend backend,
                                       const char* backend_name,
                                       VxReport* report,
                                       VxStage stage) {
    VolvoxAIEngineOptions core_options = {
        .backend = backend,
        .debug = options->debug ? 1 : 0,
        .cpu_threads = options->cpu_threads,
    };
    VxEngineStateScope scope;
    VolvoxAIEngineOptions selected_options = {0};
    char* lowered_graph_path = NULL;
    VxStatus lower_status;
    int configured = 0;
    int status;
    lower_status = vx_graph_lower_private_bootstrap(
        model->graph_path, &lowered_graph_path);
    if (lower_status != VX_STATUS_OK) {
        vx_report_write(report, lower_status, stage, backend_name, NULL,
                        lower_status == VX_STATUS_OUT_OF_MEMORY
                            ? "OUT_OF_MEMORY" : "GRAPH_LOWERING_FAILED",
                        lower_status == VX_STATUS_OUT_OF_MEMORY
                            ? "private graph lowering allocation failed"
                            : "closed v1 graph could not be lowered for the built-in engine",
                        0);
        return lower_status;
    }
    scope = vx_engine_state_scope_enter(state);
    volvoxai_engine_tensor_name_index_invalidate();
    status = volvoxai_engine_configure(&core_options);
    if (status == 0 &&
        volvoxai_engine_get_options(&selected_options) == 0 &&
        selected_options.backend == backend)
        configured = 1;
    else
        status = -1;
    if (status == 0) {
        /* Register residency before init: the engine slices the banks between
         * loading the weight files and building the graph. */
        for (size_t index = 0; status == 0 && index < model->bank_residency_count;
             index++) {
            const VxBankResidency* residency = &model->bank_residency[index];
            status = volvoxai_engine_add_bank_residency(
                residency->bank, residency->slots, residency->slot_count);
        }
    }
    if (status == 0) {
        status = volvoxai_engine_init_with_weight_files(
            lowered_graph_path,
            weights ? (const char* const*)weights->paths : NULL,
            weights ? (int)weights->path_count : 0);
    }
    vx_engine_state_scope_leave(scope);
    vx_snapshot_path_release(lowered_graph_path);
    if (status != 0) {
        VxStatus failure = configured
            ? VX_STATUS_INVALID_GRAPH : VX_STATUS_BACKEND_UNAVAILABLE;
        vx_report_write(report, failure, stage, backend_name, NULL,
                        failure == VX_STATUS_BACKEND_UNAVAILABLE
                            ? "BACKEND_UNAVAILABLE" : "GRAPH_INITIALIZATION_FAILED",
                        failure == VX_STATUS_BACKEND_UNAVAILABLE
                            ? "requested built-in backend is unavailable"
                            : "native graph validation or initialization failed",
                        0);
        return failure;
    }
    vx_report_write(report, VX_STATUS_OK, stage, backend_name,
                    vx_builtin_device_identity(backend_name), "OK",
                    "native built-in graph initialized", 0);
    return VX_STATUS_OK;
}

static void vx_private_engine_unload(VxExecutionContext* context) {
    VxEngineStateScope scope;
    if (!context) return;
    vx_declared_outputs_free(context->logical_tensors,
                             context->logical_tensor_count);
    context->logical_tensors = NULL;
    context->logical_tensor_count = 0;
    free(context->logical_tensor_name_slots);
    context->logical_tensor_name_slots = NULL;
    context->logical_tensor_name_capacity = 0;
    free(context->committed_shape_signature);
    context->committed_shape_signature = NULL;
    if (!context->engine_state_initialized) return;
    if (context->engine_loaded) {
        scope = vx_engine_state_scope_enter(context->engine_state);
        if (context->decode_session) {
            volvoxai_engine_decode_session_destroy(context->decode_session);
            context->decode_session = NULL;
        }
        volvoxai_engine_tensor_name_index_invalidate();
        volvoxai_engine_tensor_name_index_rebuild();
        volvoxai_engine_shutdown();
        vx_engine_state_scope_leave(scope);
        context->engine_loaded = 0;
    }
    vx_engine_state_deinit(context->engine_state);
    context->engine_state_initialized = 0;
    free(context->engine_state);
    context->engine_state = NULL;
}

#if VOLVOXAI_ENABLE_TRAINING
static VxStatus vx_private_weight_revision_validate(
    const VxModel* model,
    const VxWeightRevisionRecord* weights,
    VxReport* report) {
    VxEngineState* validation =
        (VxEngineState*)calloc(1, sizeof(*validation));
    VxStatus status;
    if (!validation || vx_engine_state_init(validation) != 0) {
        free(validation);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_MODEL_LOAD, "cpu", "OUT_OF_MEMORY",
                       "weight validation state allocation failed");
    }
    status = vx_private_engine_load(validation, model, weights,
                                    &model->runtime->options,
                                    VOLVOXAI_BACKEND_CPU, "cpu", report,
                                    VX_STAGE_MODEL_LOAD);
    {
        VxExecutionContext temporary = {0};
        temporary.engine_state = validation;
        temporary.engine_state_initialized = 1;
        /* Shutdown is the rollback path for both complete and partial loads. */
        temporary.engine_loaded = 1;
        vx_private_engine_unload(&temporary);
    }
    return status;
}
#endif

VxStatus vx_runtime_create(const VxRuntimeOptions* options,
                           VxRuntime** out_runtime,
                           VxReport* report) {
    VxRuntimeOptions resolved = VX_RUNTIME_OPTIONS_INIT;
    VxRuntime* runtime;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!out_runtime || (options && options->struct_size != sizeof(*options)) ||
        (options && options->cpu_threads < 0))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_RUNTIME_CREATE, NULL, "INVALID_OPTIONS",
                       "runtime options are invalid");
    *out_runtime = NULL;
    if (options) resolved = *options;
    runtime = (VxRuntime*)calloc(1, sizeof(*runtime));
    if (!runtime)
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_RUNTIME_CREATE, NULL, "OUT_OF_MEMORY",
                       "runtime allocation failed");
    atomic_init(&runtime->references, 1);
    atomic_init(&runtime->next_object_identity, 1);
    atomic_init(&runtime->next_execution_identity, 1);
    if (pthread_mutex_init(&runtime->mutex, NULL) != 0) {
        free(runtime);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_RUNTIME_CREATE,
                       NULL, "MUTEX_INIT_FAILED", "runtime mutex initialization failed");
    }
    runtime->providers = vx_provider_registry_create();
    if (!runtime->providers) {
        pthread_mutex_destroy(&runtime->mutex);
        free(runtime);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_RUNTIME_CREATE, NULL, "OUT_OF_MEMORY",
                       "provider registry allocation failed");
    }
    runtime->options = resolved;
    runtime->identity = vx_runtime_identity(runtime);
    *out_runtime = runtime;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_RUNTIME_CREATE, NULL, NULL,
                    "OK", "runtime created", 0);
    vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
    return VX_STATUS_OK;
}

void vx_runtime_retain(VxRuntime* runtime) {
    if (runtime) atomic_fetch_add_explicit(&runtime->references, 1, memory_order_relaxed);
}

void vx_runtime_release(VxRuntime* runtime) {
    if (!runtime || atomic_fetch_sub_explicit(&runtime->references, 1,
                                              memory_order_acq_rel) != 1) return;
    vx_provider_registry_destroy(runtime->providers);
    pthread_mutex_destroy(&runtime->mutex);
    free(runtime);
}

VxStatus vx_runtime_register_provider(VxRuntime* runtime,
                                      const VxBackendProvider* provider,
                                      VxReport* report) {
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_RUNTIME_CREATE, NULL, "INVALID_RUNTIME",
                       "runtime is NULL");
    pthread_mutex_lock(&runtime->mutex);
    if (runtime->closed) {
        pthread_mutex_unlock(&runtime->mutex);
        status = vx_fail(report, VX_STATUS_HANDLE_DISPOSED,
                         VX_STAGE_RUNTIME_CREATE, NULL, "HANDLE_DISPOSED",
                         "runtime is closed");
        vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
        return status;
    }
    status = vx_provider_registry_register(runtime->providers,
                                           &runtime->options,
                                           provider, report);
    pthread_mutex_unlock(&runtime->mutex);
    vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
    return status;
}

VxStatus vx_runtime_close(VxRuntime* runtime, VxReport* report) {
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CLOSE,
                       NULL, "INVALID_RUNTIME", "runtime is NULL");
    pthread_mutex_lock(&runtime->mutex);
    runtime->closed = 1;
    pthread_mutex_unlock(&runtime->mutex);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE, NULL, NULL, "OK",
                    "runtime closed", 0);
    vx_report_set_lineage(report, runtime, NULL, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_runtime_load_model(VxRuntime* runtime,
                               const VxModelSource* source,
                               VxModel** out_model,
                               VxReport* report) {
    VxModel* model;
    VxWeightRevisionRecord* weights = NULL;
    VxAdapterRevisionRecord* base_adapter = NULL;
    VxStatus envelope_status;
    VxStatus weight_status;
    VxStatus adapter_status;
    VxStatus graph_snapshot_status;
    uint64_t graph_revision;
    char* graph_snapshot = NULL;
    VxDeclaredTensor* declared_inputs = NULL;
    size_t input_count = 0;
    VxDeclaredOutput* declared_outputs = NULL;
    size_t output_count = 0;
    VxDeclaredTensor* domain_tensors = NULL;
    size_t domain_tensor_count = 0;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!runtime || !source || source->struct_size != sizeof(*source) ||
        !source->graph_path || !source->graph_path[0] || !out_model ||
        source->weight_path_count > (size_t)MAX_WEIGHT_FILES ||
        (source->weight_path_count && !source->weight_paths) ||
        (source->bank_residency_count && !source->bank_residency) ||
        source->bank_residency_count >
            SIZE_MAX / sizeof(*source->bank_residency) ||
        source->bank_residency_count >
            SIZE_MAX / sizeof(uint32_t*))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_MODEL_SOURCE", "model source is invalid");
    for (size_t index = 0; index < source->bank_residency_count; index++) {
        const VxBankResidency* residency = &source->bank_residency[index];
        if (residency->struct_size != sizeof(*residency) || !residency->bank ||
            !residency->bank[0] || !residency->slots || !residency->slot_count ||
            residency->slot_count > SIZE_MAX / sizeof(*residency->slots))
            return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                           NULL, "INVALID_BANK_RESIDENCY",
                           "bank residency entry is invalid");
        for (size_t prior = 0; prior < index; prior++)
            if (!strcmp(source->bank_residency[prior].bank, residency->bank))
                return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                               VX_STAGE_MODEL_LOAD, NULL,
                               "INVALID_BANK_RESIDENCY",
                               "bank residency entries must name unique banks");
        for (size_t slot = 1; slot < residency->slot_count; slot++)
            if (residency->slots[slot] <= residency->slots[slot - 1])
                return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                               VX_STAGE_MODEL_LOAD, NULL,
                               "INVALID_BANK_RESIDENCY",
                               "bank residency slots must be strictly ascending");
    }

    if (!vx_is_canonical_graph_path(source->graph_path))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_GRAPH_PATH",
                       "graph_path must name graph.json or a named *.graph.json document");
    *out_model = NULL;
    pthread_mutex_lock(&runtime->mutex);
    int runtime_closed = runtime->closed;
    pthread_mutex_unlock(&runtime->mutex);
    if (runtime_closed)
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_MODEL_LOAD,
                       NULL, "HANDLE_DISPOSED", "runtime is closed");
    graph_snapshot_status = vx_file_snapshot(
        source->graph_path, &graph_snapshot, &graph_revision);
    if (graph_snapshot_status != VX_STATUS_OK)
        return vx_fail(
            report, graph_snapshot_status, VX_STAGE_MODEL_LOAD, NULL,
            graph_snapshot_status == VX_STATUS_OUT_OF_MEMORY
                ? "OUT_OF_MEMORY" : "GRAPH_NOT_READABLE",
            graph_snapshot_status == VX_STATUS_OUT_OF_MEMORY
                ? "graph snapshot allocation failed"
                : "graph package file could not be snapshotted");
    envelope_status = vx_graph_package_validate(
        graph_snapshot, NULL, 0, 0, NULL, 0,
        &declared_inputs, &input_count,
        &declared_outputs, &output_count);
    if (envelope_status != VX_STATUS_OK) {
        vx_snapshot_path_release(graph_snapshot);
        if (envelope_status == VX_STATUS_OUT_OF_MEMORY)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "OUT_OF_MEMORY", "graph package validation allocation failed");
        if (envelope_status == VX_STATUS_IO_ERROR)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "GRAPH_NOT_READABLE", "graph package file could not be read");
        return vx_fail(report, VX_STATUS_INVALID_GRAPH, VX_STAGE_MODEL_LOAD, NULL,
                       "INVALID_GRAPH_CONTRACT",
                       "graph.json must use the volvox-graph/v1 contract");
    }
    vx_declared_outputs_free(declared_inputs, input_count);
    declared_inputs = NULL;
    input_count = 0;
    vx_declared_outputs_free(declared_outputs, output_count);
    declared_outputs = NULL;
    output_count = 0;
    model = (VxModel*)calloc(1, sizeof(*model));
    if (!model) {
        vx_snapshot_path_release(graph_snapshot);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD,
                       NULL, "OUT_OF_MEMORY", "model allocation failed");
    }
    atomic_init(&model->references, 1);
    if (pthread_mutex_init(&model->revision_mutex, NULL) != 0) {
        vx_snapshot_path_release(graph_snapshot);
        free(model);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_MODEL_LOAD, NULL,
                       "MUTEX_INIT_FAILED",
                       "model revision mutex initialization failed");
    }
    model->graph_path = graph_snapshot;
    graph_snapshot = NULL;
    model->logical_graph = vx_json_file_load(model->graph_path);
    if (!model->logical_graph) goto oom;
    if (source->bank_residency_count) {
        const cJSON* declared =
            cJSON_GetObjectItemCaseSensitive(model->logical_graph, "banks");
        const cJSON* dimensions =
            cJSON_GetObjectItemCaseSensitive(model->logical_graph,
                                             "dimensions");
        model->bank_residency = (VxBankResidency*)calloc(
            source->bank_residency_count, sizeof(*model->bank_residency));
        model->bank_residency_slots = (uint32_t**)calloc(
            source->bank_residency_count, sizeof(*model->bank_residency_slots));
        if (!model->bank_residency || !model->bank_residency_slots) goto oom;
        for (size_t index = 0; index < source->bank_residency_count; index++) {
            const VxBankResidency* requested = &source->bank_residency[index];
            const cJSON* declared_bank = declared
                ? cJSON_GetObjectItemCaseSensitive(declared, requested->bank)
                : NULL;
            int64_t bank_minimum;
            int64_t bank_maximum;
            int64_t bank_multiple;
            uint32_t* slots;
            if (!declared_bank || !cJSON_IsString(declared_bank) ||
                !declared_bank->valuestring ||
                !vx_dimension_json(dimensions, declared_bank->valuestring,
                                   &bank_minimum, &bank_maximum,
                                   &bank_multiple) ||
                (uint64_t)requested->slots[requested->slot_count - 1u] >=
                    (uint64_t)bank_maximum)
                goto invalid_bank_residency;
            slots = (uint32_t*)malloc(requested->slot_count * sizeof(uint32_t));
            if (!slots) goto oom;
            memcpy(slots, requested->slots,
                   requested->slot_count * sizeof(uint32_t));
            model->bank_residency_slots[index] = slots;
            model->bank_residency[index] = *requested;
            model->bank_residency[index].struct_size =
                sizeof(model->bank_residency[index]);
            /* The parsed graph owns this canonical key for the whole model
             * lifetime; never retain the caller's transient bank string. */
            model->bank_residency[index].bank = declared_bank->string;
            model->bank_residency[index].slots = slots;
            model->bank_residency_count = index + 1;
        }
    }
    model->runtime = runtime;
    model->identity = vx_next_object_identity(runtime);
    model->graph_identity = vx_next_object_identity(runtime);
    model->graph_revision = graph_revision;
    model->weight_identity = vx_next_object_identity(runtime);
    weights = vx_weight_revision_create(source->weight_paths,
                                        source->weight_path_count,
                                        model->weight_identity, 1,
                                        &weight_status);
    if (!weights) {
        if (weight_status == VX_STATUS_IO_ERROR) goto weights_unreadable;
        if (weight_status == VX_STATUS_INVALID_ARGUMENT)
            goto weights_invalid;
        goto oom;
    }
    envelope_status = vx_graph_package_validate(
        model->graph_path, (const char* const*)weights->paths,
        weights->path_count, 1,
        model->bank_residency, model->bank_residency_count,
        &declared_inputs, &input_count,
        &declared_outputs, &output_count);
    if (envelope_status != VX_STATUS_OK) {
        vx_weight_revision_release(weights);
        weights = NULL;
        vx_declared_outputs_free(declared_inputs, input_count);
        vx_declared_outputs_free(declared_outputs, output_count);
        vx_model_bank_residency_clear(model);
        cJSON_Delete(model->logical_graph);
        vx_snapshot_path_release(model->graph_path);
        pthread_mutex_destroy(&model->revision_mutex);
        free(model);
        if (envelope_status == VX_STATUS_OUT_OF_MEMORY)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "OUT_OF_MEMORY",
                           "output descriptor allocation failed");
        if (envelope_status == VX_STATUS_IO_ERROR)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "GRAPH_NOT_READABLE",
                           "graph package snapshot could not be read");
        if (envelope_status == VX_STATUS_INVALID_ARGUMENT)
            return vx_fail(report, envelope_status, VX_STAGE_MODEL_LOAD, NULL,
                           "INVALID_BANK_RESIDENCY",
                           "bank residency exceeds the supplied bank extent");
        return vx_fail(report, VX_STATUS_INVALID_GRAPH, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_GRAPH_CONTRACT",
                       "every graph output must have one canonical execution descriptor");
    }
    model->inputs = declared_inputs;
    model->input_count = input_count;
    declared_inputs = NULL;
    input_count = 0;
    model->outputs = declared_outputs;
    model->output_count = output_count;
    declared_outputs = NULL;
    output_count = 0;
    envelope_status = vx_logical_tensors_load(
        model, &domain_tensors, &domain_tensor_count);
    if (envelope_status != VX_STATUS_OK) {
        if (envelope_status == VX_STATUS_OUT_OF_MEMORY) goto oom;
        vx_weight_revision_release(weights);
        vx_declared_outputs_free(model->inputs, model->input_count);
        vx_declared_outputs_free(model->outputs, model->output_count);
        vx_model_bank_residency_clear(model);
        cJSON_Delete(model->logical_graph);
        vx_snapshot_path_release(model->graph_path);
        pthread_mutex_destroy(&model->revision_mutex);
        free(model);
        return vx_fail(report, VX_STATUS_INVALID_GRAPH,
                       VX_STAGE_MODEL_LOAD, NULL,
                       "INVALID_LOGICAL_DOMAIN",
                       "logical activation descriptors do not form one closed bounded domain");
    }
    for (size_t index = 0; index < domain_tensor_count; index++)
        if (domain_tensors[index].maximum_byte_size >
            model->maximum_tensor_bytes)
            model->maximum_tensor_bytes =
                (uint64_t)domain_tensors[index].maximum_byte_size;
    vx_declared_outputs_free(domain_tensors, domain_tensor_count);
    domain_tensors = NULL;
    domain_tensor_count = 0;
    base_adapter = vx_adapter_revision_create(
        "__base__", NULL, NULL, vx_next_object_identity(runtime), 1,
        &adapter_status);
    if (!base_adapter) goto oom;
    model->adapters = base_adapter;
    atomic_init(&model->current_weights, weights);
    atomic_init(&model->current_adapter, base_adapter);
    model->allocated_bytes = sizeof(*model) + strlen(model->graph_path) + 1u;
    snprintf(model->graph_fingerprint, sizeof(model->graph_fingerprint),
             "fnv1a64:%016" PRIx64, model->graph_revision);
    snprintf(model->shape_domain_proof_identity,
             sizeof(model->shape_domain_proof_identity),
             "%s:%016" PRIx64,
             VX_BACKEND_SHAPE_PROOF_PROTOCOL, model->graph_revision);
    model->allocated_bytes += model->input_count * sizeof(*model->inputs);
    for (size_t index = 0; index < model->input_count; index++) {
        model->allocated_bytes += strlen(model->inputs[index].name) + 1u;
        if (model->inputs[index].maximum_byte_size > model->maximum_tensor_bytes)
            model->maximum_tensor_bytes =
                (uint64_t)model->inputs[index].maximum_byte_size;
        for (uint32_t axis = 0; axis < model->inputs[index].rank; axis++)
            if (model->inputs[index].symbols[axis])
                model->allocated_bytes +=
                    strlen(model->inputs[index].symbols[axis]) + 1u;
    }
    model->allocated_bytes += model->output_count * sizeof(*model->outputs);
    for (size_t index = 0; index < model->output_count; index++) {
        model->allocated_bytes += strlen(model->outputs[index].name) + 1u;
        if (model->outputs[index].maximum_byte_size > model->maximum_tensor_bytes)
            model->maximum_tensor_bytes =
                (uint64_t)model->outputs[index].maximum_byte_size;
        for (uint32_t axis = 0; axis < model->outputs[index].rank; axis++)
            if (model->outputs[index].symbols[axis])
                model->allocated_bytes +=
                    strlen(model->outputs[index].symbols[axis]) + 1u;
    }
    if (model->bank_residency_count) {
        if (!vx_native_gpu_accumulate_resource_bytes(
                &model->allocated_bytes,
                sizeof(*model->bank_residency),
                (uint64_t)model->bank_residency_count) ||
            !vx_native_gpu_accumulate_resource_bytes(
                &model->allocated_bytes,
                sizeof(*model->bank_residency_slots),
                (uint64_t)model->bank_residency_count)) goto oom;
        for (size_t index = 0; index < model->bank_residency_count; index++)
            if (!vx_native_gpu_accumulate_resource_bytes(
                    &model->allocated_bytes,
                    sizeof(*model->bank_residency[index].slots),
                    (uint64_t)model->bank_residency[index].slot_count))
                goto oom;
    }
    vx_runtime_retain(runtime);
    *out_model = model;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "model source retained", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
oom:
    vx_declared_outputs_free(domain_tensors, domain_tensor_count);
    vx_weight_revision_release(weights);
    vx_adapter_revision_release(base_adapter);
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD, NULL,
                   "OUT_OF_MEMORY", "model source copy failed");
weights_unreadable:
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_IO_ERROR, VX_STAGE_MODEL_LOAD, NULL,
                   "WEIGHTS_NOT_READABLE",
                   "a weight revision file could not be read");
weights_invalid:
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                   NULL, "INVALID_MODEL_SOURCE",
                   "a weight source path is empty");
invalid_bank_residency:
    vx_declared_outputs_free(declared_inputs, input_count);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                   NULL, "INVALID_BANK_RESIDENCY",
                   "bank residency does not match the graph's declared bank domain");
}

void vx_model_retain(VxModel* model) {
    if (model) atomic_fetch_add_explicit(&model->references, 1, memory_order_relaxed);
}

void vx_model_release(VxModel* model) {
    VxAdapterRevisionRecord* adapter;
    if (!model || atomic_fetch_sub_explicit(&model->references, 1,
                                            memory_order_acq_rel) != 1) return;
    vx_weight_revision_release(atomic_load_explicit(
        &model->current_weights, memory_order_relaxed));
    adapter = model->adapters;
    while (adapter) {
        VxAdapterRevisionRecord* next = adapter->next;
        vx_adapter_revision_release(adapter);
        adapter = next;
    }
    vx_declared_outputs_free(model->inputs, model->input_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_model_bank_residency_clear(model);
    cJSON_Delete(model->logical_graph);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    vx_runtime_release(model->runtime);
    free(model);
}

VxStatus vx_model_revision_info(VxModel* model,
                                VxRevisionInfo* info,
                                VxReport* report) {
    VxWeightRevisionRecord* weights;
    VxAdapterRevisionRecord* adapter;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !info || info->struct_size != sizeof(*info))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_REVISION_INFO",
                       "model or revision info buffer is invalid");
    pthread_mutex_lock(&model->revision_mutex);
    weights = atomic_load_explicit(&model->current_weights,
                                   memory_order_acquire);
    adapter = atomic_load_explicit(&model->current_adapter,
                                   memory_order_acquire);
    info->graph_id = model->graph_identity;
    info->graph_revision = model->graph_revision;
    info->weight_id = weights ? weights->identity : 0;
    info->weight_revision = weights ? weights->revision : 0;
    info->adapter_id = adapter ? adapter->identity : 0;
    info->adapter_revision = adapter ? adapter->revision : 0;
    pthread_mutex_unlock(&model->revision_mutex);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "published model revision returned", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_model_publish_adapter(VxModel* model,
                                  const VxAdapterSource* source,
                                  VxAdapterRevision* published,
                                  VxReport* report) {
    VxAdapterRevisionRecord* previous = NULL;
    VxAdapterRevisionRecord* record;
    VxStatus record_status;
    uint64_t identity;
    uint64_t revision;
    int runtime_closed;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !source || source->struct_size != sizeof(*source) ||
        !source->adapter_name || !source->adapter_name[0] ||
        strlen(source->adapter_name) >= 128u ||
        (source->package_path && !source->package_path[0]) ||
        (source->version_name &&
         (!source->version_name[0] || strlen(source->version_name) >= 96u)) ||
        (published && published->struct_size != sizeof(*published)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_ADAPTER,
                       NULL, "INVALID_ADAPTER_SOURCE",
                       "adapter source or revision output is invalid");
    pthread_mutex_lock(&model->runtime->mutex);
    runtime_closed = model->runtime->closed;
    pthread_mutex_unlock(&model->runtime->mutex);
    if (runtime_closed)
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_ADAPTER,
                       NULL, "HANDLE_DISPOSED", "runtime is closed");
    pthread_mutex_lock(&model->revision_mutex);
    for (VxAdapterRevisionRecord* item = model->adapters;
         item; item = item->next) {
        if (!strcmp(item->name, source->adapter_name) &&
            (!previous || item->revision > previous->revision))
            previous = item;
    }
    identity = previous ? previous->identity :
        vx_next_object_identity(model->runtime);
    revision = previous ? previous->revision + 1u : 1u;
    if (!revision) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_ADAPTER, NULL,
                       "REVISION_OVERFLOW", "adapter revision overflowed");
    }
    record = vx_adapter_revision_create(source->adapter_name,
                                        source->package_path,
                                        source->version_name,
                                        identity, revision, &record_status);
    if (!record) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(
            report, record_status, VX_STAGE_ADAPTER, NULL,
            record_status == VX_STATUS_IO_ERROR ? "ADAPTER_NOT_READABLE" :
                                                  "OUT_OF_MEMORY",
            record_status == VX_STATUS_IO_ERROR
                ? "adapter package file could not be snapshotted"
                : "adapter revision allocation failed");
    }
    record->next = model->adapters;
    model->adapters = record;
    atomic_store_explicit(&model->current_adapter, record,
                          memory_order_release);
    pthread_mutex_unlock(&model->revision_mutex);
    if (published) {
        published->adapter_id = identity;
        published->adapter_revision = revision;
    }
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_ADAPTER, NULL, NULL,
                    "OK", "adapter revision published atomically", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

#if VOLVOXAI_ENABLE_TRAINING
VxStatus vx_model_internal_accept_full_owner(
    VxModel* model,
    VxWeightRevisionRecord** out_revision,
    VxReport* report) {
    VxWeightRevisionRecord* revision;
    if (!model || !out_revision)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_TRAINER",
                       "trainer acceptance arguments are invalid");
    *out_revision = NULL;
    /* Runtime close and Trainer acceptance linearize on this mutex. A Trainer
     * that retains its exact base revision before close remains a live child;
     * a later creation attempt is rejected. */
    pthread_mutex_lock(&model->runtime->mutex);
    if (model->runtime->closed) {
        pthread_mutex_unlock(&model->runtime->mutex);
        return vx_fail(report, VX_STATUS_HANDLE_DISPOSED,
                       VX_STAGE_MODEL_LOAD, NULL, "HANDLE_DISPOSED",
                       "runtime is closed");
    }
    pthread_mutex_lock(&model->revision_mutex);
    revision = atomic_load_explicit(&model->current_weights,
                                    memory_order_acquire);
    vx_weight_revision_retain(revision);
    pthread_mutex_unlock(&model->revision_mutex);
    pthread_mutex_unlock(&model->runtime->mutex);
    if (!revision)
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_MODEL_LOAD, NULL,
                       "REVISION_MISSING",
                       "model has no published weight revision");
    *out_revision = revision;
    return VX_STATUS_OK;
}

VxStatus vx_model_internal_validate_authoring_revision(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    uint64_t adapter_id,
    uint64_t adapter_revision,
    VxReport* report) {
    VxWeightRevisionRecord* current;
    VxAdapterRevisionRecord* adapter;
    VxStatus status = VX_STATUS_OK;
    if (!model || !exact_revision ||
        exact_revision->identity != model->weight_identity)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL,
                       "INVALID_AUTHORING_REVISION",
                       "authoring revision arguments are invalid");
    pthread_mutex_lock(&model->runtime->mutex);
    if (model->runtime->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
    } else {
        pthread_mutex_lock(&model->revision_mutex);
        current = atomic_load_explicit(&model->current_weights,
                                       memory_order_acquire);
        adapter = atomic_load_explicit(&model->current_adapter,
                                       memory_order_acquire);
        /* Authoring engines currently materialize the exact base-weight
         * revision only.  They do not materialize an adapter package, so a
         * plan may be created only from the no-adapter lineage and becomes
         * stale as soon as any adapter is published. */
        if (current != exact_revision ||
            (adapter && (!adapter->name || strcmp(adapter->name, "__base__"))) ||
            adapter_id != 0u || adapter_revision != 0u)
            status = VX_STATUS_REVISION_CONFLICT;
        pthread_mutex_unlock(&model->revision_mutex);
    }
    pthread_mutex_unlock(&model->runtime->mutex);
    if (status == VX_STATUS_OK) {
        vx_report_write(report, status, VX_STAGE_MODEL_LOAD, NULL, NULL,
                        "OK", "authoring revision is current", 0);
        vx_report_set_lineage(report, NULL, model, NULL, NULL);
        return status;
    }
    return vx_fail(
        report, status, VX_STAGE_MODEL_LOAD, NULL,
        status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
                                              "REVISION_CONFLICT",
        status == VX_STATUS_HANDLE_DISPOSED
            ? "runtime closed after authoring owner creation"
            : "model weights or adapter changed after authoring owner creation");
}

void vx_model_internal_release_weight_revision(
    VxWeightRevisionRecord* revision) {
    vx_weight_revision_release(revision);
}

void vx_model_internal_destroy_authoring_engine(VxEngineState* state) {
    VxEngineStateScope scope;
    if (!state) return;
    scope = vx_engine_state_scope_enter(state);
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    /* Shutdown is also the rollback path for a partially loaded graph. */
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
}

size_t vx_model_internal_input_count(const VxModel* model) {
    return model ? model->input_count : 0u;
}

VxStatus vx_model_internal_input_spec(const VxModel* model,
                                      size_t index,
                                      VxTensorSpec* spec) {
    if (!model || !spec || spec->struct_size != sizeof(*spec))
        return VX_STATUS_INVALID_ARGUMENT;
    if (index >= model->input_count) return VX_STATUS_NOT_FOUND;
    vx_declared_tensor_spec(&model->inputs[index], spec);
    return VX_STATUS_OK;
}

const char* vx_model_internal_logical_fingerprint(const VxModel* model) {
    return model ? model->graph_fingerprint : NULL;
}

VxStatus vx_model_internal_create_authoring_engine(
    VxModel* model,
    VxWeightRevisionRecord* exact_revision,
    VolvoxAIEngineBackend backend,
    VxEngineState** out_state,
    VxReport* report) {
    const char* backend_name = vx_builtin_backend_name(backend);
    VxEngineState* state;
    VxStatus status;
    if (!model || !exact_revision || !out_state || !backend_name ||
        exact_revision->identity != model->weight_identity)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, backend_name,
                       "INVALID_TRAINING_ENGINE",
                       "training engine arguments are invalid");
    *out_state = NULL;
    /* Only accepted Trainers call this helper. Their retained Model/Runtime
     * ownership survives logical Runtime close, including rollback engine
     * recreation. Public Trainer creation performs the close check. */
    state = (VxEngineState*)calloc(1, sizeof(*state));
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_MODEL_LOAD, backend_name, "OUT_OF_MEMORY",
                       "training engine state allocation failed");
    }
    status = vx_private_engine_load(state, model, exact_revision,
                                    &model->runtime->options, backend,
                                    backend_name, report,
                                    VX_STAGE_MODEL_LOAD);
    if (status != VX_STATUS_OK) {
        vx_model_internal_destroy_authoring_engine(state);
        return status;
    }
    *out_state = state;
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_model_internal_prepare_weight_revision(
    VxModel* model,
    const char* const* weight_paths,
    size_t weight_path_count,
    VxWeightRevisionRecord** out_revision,
    VxReport* report) {
    VxStatus status;
    VxWeightRevisionRecord* successor;
    if (!model || !out_revision ||
        weight_path_count > (size_t)MAX_WEIGHT_FILES ||
        (weight_path_count && !weight_paths))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_WEIGHT_REVISION",
                       "successor weight source is invalid");
    *out_revision = NULL;
    /* This hidden path is used by an already-accepted Trainer. The Trainer
     * retains Model and Runtime, so logical Runtime close must not invalidate
     * its private work or atomic publication. New Trainer creation remains
     * rejected by vx_model_internal_create_authoring_engine(). */
    successor = vx_weight_revision_create(weight_paths, weight_path_count,
                                          model->weight_identity, 0, &status);
    if (!successor)
        return vx_fail(report, status, VX_STAGE_MODEL_LOAD, NULL,
                       status == VX_STATUS_IO_ERROR ?
                           "WEIGHTS_NOT_READABLE" : "OUT_OF_MEMORY",
                       status == VX_STATUS_IO_ERROR ?
                           "successor weight file could not be read" :
                           "successor weight revision allocation failed");
    status = vx_private_weight_revision_validate(model, successor, report);
    if (status != VX_STATUS_OK) {
        vx_weight_revision_release(successor);
        return status;
    }
    *out_revision = successor;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "private weight successor prepared", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}

VxStatus vx_model_internal_publish_weight_revision(
    VxModel* model,
    VxWeightRevisionRecord* expected,
    VxWeightRevisionRecord* successor,
    VxReport* report) {
    VxWeightRevisionRecord* current;
    uint64_t next_revision;
    if (!model || !expected || !successor || successor->revision != 0 ||
        successor->identity != model->weight_identity)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_MODEL_LOAD, NULL, "INVALID_WEIGHT_REVISION",
                       "weight publication arguments are invalid");
    pthread_mutex_lock(&model->revision_mutex);
    current = atomic_load_explicit(&model->current_weights,
                                   memory_order_acquire);
    if (current != expected) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(report, VX_STATUS_REVISION_CONFLICT,
                       VX_STAGE_MODEL_LOAD, NULL,
                       "REVISION_CONFLICT",
                       "model weight revision changed since trainer creation");
    }
    next_revision = current->revision + 1u;
    if (!next_revision) {
        pthread_mutex_unlock(&model->revision_mutex);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_MODEL_LOAD, NULL,
                       "REVISION_OVERFLOW", "weight revision overflowed");
    }
    successor->revision = next_revision;
    vx_weight_revision_retain(successor);
    atomic_store_explicit(&model->current_weights, successor,
                          memory_order_release);
    pthread_mutex_unlock(&model->revision_mutex);
    vx_weight_revision_release(current);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "weight successor published atomically", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
}
#endif

static void vx_compiled_destroy(VxCompiledModel* compiled) {
    if (compiled->provider) {
        if (compiled->provider_compiled)
            compiled->provider->compiled_destroy(compiled->provider_compiled);
    }
    vx_weight_revision_release(compiled->weights);
    vx_adapter_revision_release(compiled->adapter);
    vx_model_release(compiled->model);
    free(compiled);
}

static VxStatus vx_model_compile_candidate(VxModel* model,
                                           const VxBackendPolicy* policy,
                                           const char* requested_backend,
                                           VxCompiledModel** out_compiled,
                                           VxReport* report) {
    VxBackendPolicy resolved = *policy;
    const VxBackendProvider* provider = NULL;
    const char* selected_backend;
    VolvoxAIEngineBackend builtin_backend = VOLVOXAI_BACKEND_CPU;
    VxProviderBinding binding = {0};
    VxCompiledModel* compiled;
    VxStatus status;
    VxReport compile_report = VX_REPORT_INIT;
    double compile_started_ms = vx_report_now_ms();
    int builtin;
    VxStatus missing_status;
    *out_compiled = NULL;
    selected_backend = requested_backend;
    builtin = vx_builtin_backend_by_name(requested_backend, &builtin_backend);
    if (!builtin)
        if (vx_provider_registry_find(model->runtime->providers,
                                      requested_backend, &binding))
            provider = binding.provider;
    if (!builtin && !provider) {
        missing_status = resolved.mode == VX_BACKEND_REQUIRE
            ? VX_STATUS_BACKEND_REQUIRED : VX_STATUS_BACKEND_UNAVAILABLE;
        vx_report_write(&compile_report, missing_status,
                        VX_STAGE_COMPILE, requested_backend, NULL,
                        resolved.mode == VX_BACKEND_REQUIRE
                            ? "BACKEND_REQUIRED" : "BACKEND_UNAVAILABLE",
                        resolved.mode == VX_BACKEND_REQUIRE
                            ? "required backend provider is unavailable"
                            : "preferred backend provider is unavailable",
                        0);
        vx_report_set_lineage(&compile_report, NULL, model, NULL, NULL);
        compile_report.policy_mode = resolved.mode;
        compile_report.operator_fallback = resolved.operator_fallback;
        compile_report.candidate_count = 1;
        snprintf(compile_report.candidate_outcomes,
                 sizeof(compile_report.candidate_outcomes),
                 "%s:unavailable", requested_backend);
        snprintf(compile_report.fallback_evidence,
                 sizeof(compile_report.fallback_evidence), "%s",
                 "tier=forbidden;operator=not-evaluated");
        compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
        vx_report_publish(report, &compile_report);
        return missing_status;
    }
    compiled = (VxCompiledModel*)calloc(1, sizeof(*compiled));
    if (!compiled) {
        vx_report_write(&compile_report, VX_STATUS_OUT_OF_MEMORY,
                        VX_STAGE_COMPILE, selected_backend, NULL,
                        "OUT_OF_MEMORY", "compiled model allocation failed", 0);
        vx_report_set_lineage(&compile_report, NULL, model, NULL, NULL);
        compile_report.policy_mode = resolved.mode;
        compile_report.operator_fallback = resolved.operator_fallback;
        compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
        vx_report_publish(report, &compile_report);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    atomic_init(&compiled->references, 1);
    compiled->model = model;
    vx_model_retain(model);
    pthread_mutex_lock(&model->revision_mutex);
    compiled->weights = atomic_load_explicit(&model->current_weights,
                                              memory_order_acquire);
    compiled->adapter = atomic_load_explicit(&model->current_adapter,
                                              memory_order_acquire);
    vx_weight_revision_retain(compiled->weights);
    vx_adapter_revision_retain(compiled->adapter);
    pthread_mutex_unlock(&model->revision_mutex);
    compiled->identity = vx_next_object_identity(model->runtime);
    compiled->allocated_bytes = sizeof(*compiled);
    compiled->mode = resolved.mode;
    compiled->operator_fallback = resolved.operator_fallback;
    compiled->builtin_backend = builtin_backend;
    snprintf(compiled->backend, sizeof(compiled->backend), "%s", selected_backend);
    compiled->report = (VxReport)VX_REPORT_INIT;
    if (provider) {
        VxModelSource source;
        VxBackendCompileInput compile_input = VX_BACKEND_COMPILE_INPUT_INIT;
        VxBackendShapeDomainAttestation attestation =
            VX_BACKEND_SHAPE_DOMAIN_ATTESTATION_INIT;
        VxTensorSpec* input_specs = NULL;
        VxTensorSpec* output_specs = NULL;
        VxReport provider_report = VX_REPORT_INIT;
        compiled->provider = provider;
        compiled->provider_runtime = binding.runtime_instance;
        if (provider->shape_domain.support != VX_BACKEND_SHAPE_DOMAIN_FULL) {
            vx_report_write(&compile_report, VX_STATUS_BACKEND_UNSUPPORTED,
                            VX_STAGE_COMPILE, selected_backend, NULL,
                            "BOUNDED_DOMAIN_UNSUPPORTED",
                            "provider does not support the complete bounded shape domain",
                            0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:domain-unsupported", selected_backend);
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return VX_STATUS_BACKEND_UNSUPPORTED;
        }
        input_specs = vx_declared_tensor_specs(model->inputs,
                                               model->input_count);
        output_specs = vx_declared_tensor_specs(model->outputs,
                                                model->output_count);
        if ((model->input_count && !input_specs) ||
            (model->output_count && !output_specs)) {
            free(input_specs);
            free(output_specs);
            vx_compiled_destroy(compiled);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_COMPILE, selected_backend,
                           "OUT_OF_MEMORY", "compile-spec allocation failed");
        }
        vx_model_source_view(model, compiled->weights, &source);
        compile_input.source = &source;
        compile_input.graph_fingerprint = model->graph_fingerprint;
        compile_input.shape_domain_proof_identity =
            model->shape_domain_proof_identity;
        compile_input.inputs = input_specs;
        compile_input.input_count = model->input_count;
        compile_input.outputs = output_specs;
        compile_input.output_count = model->output_count;
        provider_report = (VxReport)VX_REPORT_INIT;
        status = provider->compile(compiled->provider_runtime, &compile_input,
                                   &resolved, &compiled->provider_compiled,
                                   &attestation, &provider_report);
        free(input_specs);
        free(output_specs);
        if (status != VX_STATUS_OK || !compiled->provider_compiled) {
            VxStatus failure = status == VX_STATUS_OK
                ? VX_STATUS_BACKEND_UNSUPPORTED : status;
            vx_report_write(&compile_report, failure, VX_STAGE_COMPILE,
                            selected_backend, provider_report.device,
                            "BACKEND_UNSUPPORTED", "backend compilation failed", 0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:compile-failed", selected_backend);
            snprintf(compile_report.offending_node,
                     sizeof(compile_report.offending_node), "%s",
                     provider_report.offending_node);
            snprintf(compile_report.route_evidence,
                     sizeof(compile_report.route_evidence), "%s",
                     provider_report.route_evidence);
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return failure;
        }
        if (!vx_provider_attestation_valid(model, &attestation)) {
            vx_report_write(&compile_report, VX_STATUS_ABI_UNSUPPORTED,
                            VX_STAGE_COMPILE, selected_backend,
                            provider_report.device,
                            "SHAPE_DOMAIN_ATTESTATION_INVALID",
                            "provider did not attest the exact bounded shape domain and resources",
                            0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:domain-unattested", selected_backend);
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return VX_STATUS_ABI_UNSUPPORTED;
        }
        if (!provider_report.route_attested || !provider_report.route_evidence[0] ||
            !provider_report.device[0] ||
            (resolved.operator_fallback == VX_OPERATOR_FALLBACK_FORBID &&
             provider_report.operator_fallback_used)) {
            const int strict = resolved.operator_fallback == VX_OPERATOR_FALLBACK_FORBID;
            vx_report_write(&compile_report,
                            strict ? VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN :
                                     VX_STATUS_BACKEND_UNSUPPORTED,
                            VX_STAGE_COMPILE, selected_backend,
                            provider_report.device,
                            strict ? "OPERATOR_FALLBACK_FORBIDDEN" :
                                     "BACKEND_UNSUPPORTED",
                            strict ? "provider did not attest an all-provider route" :
                                     "provider did not report exact route evidence",
                            0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.operator_fallback_used =
                provider_report.operator_fallback_used;
            compile_report.route_attested = provider_report.route_attested;
            compile_report.candidate_count = 1;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes),
                     "%s:route-unattested", selected_backend);
            snprintf(compile_report.route_evidence,
                     sizeof(compile_report.route_evidence), "%s",
                     provider_report.route_evidence);
            snprintf(compile_report.fallback_evidence,
                     sizeof(compile_report.fallback_evidence), "%s",
                     provider_report.fallback_evidence[0]
                         ? provider_report.fallback_evidence :
                           "operator=unattested");
            snprintf(compile_report.offending_node,
                     sizeof(compile_report.offending_node), "%s",
                     provider_report.offending_node);
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return strict ? VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN :
                            VX_STATUS_BACKEND_UNSUPPORTED;
        }
        compile_report = provider_report;
        compile_report.struct_size = sizeof(compile_report);
        compile_report.status = VX_STATUS_OK;
        compile_report.stage = VX_STAGE_COMPILE;
        compile_report.execution_id = 0;
        snprintf(compile_report.backend, sizeof(compile_report.backend), "%s",
                 selected_backend);
        snprintf(compile_report.reason, sizeof(compile_report.reason), "%s", "OK");
        if (!compile_report.message[0])
            snprintf(compile_report.message, sizeof(compile_report.message), "%s",
                     "backend compilation completed");
    } else {
        VxEngineState* validation = (VxEngineState*)calloc(1, sizeof(*validation));
        VxBuiltinResourceDomain resource_domain = {0};
        char domain_evidence[VX_REPORT_ROUTE_CAPACITY] = {0};
        if (!validation || vx_engine_state_init(validation) != 0) {
            free(validation);
            vx_report_write(&compile_report, VX_STATUS_OUT_OF_MEMORY,
                            VX_STAGE_COMPILE, selected_backend,
                            vx_builtin_device_identity(selected_backend),
                            "OUT_OF_MEMORY", "validation state allocation failed", 0);
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return VX_STATUS_OUT_OF_MEMORY;
        }
        status = vx_private_engine_load(validation, model, compiled->weights,
                                        &model->runtime->options,
                                        compiled->builtin_backend,
                                        selected_backend,
                                        &compile_report, VX_STAGE_COMPILE);
        if (status == VX_STATUS_OK) {
            status = vx_builtin_bounded_domain_proof(
                model, compiled->weights, compiled->builtin_backend,
                validation,
                &resource_domain,
                domain_evidence,
                sizeof(domain_evidence));
            if (status != VX_STATUS_OK) {
                int contract_unsupported =
                    strstr(domain_evidence,
                           "required=canonical-native-shape-contract") != NULL;
                vx_report_write(
                    &compile_report, status, VX_STAGE_COMPILE,
                    selected_backend,
                    vx_builtin_device_identity(selected_backend),
                    status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
                    contract_unsupported
                        ? "CANONICAL_SHAPE_CONTRACT_UNSUPPORTED"
                        : "BOUNDED_DOMAIN_UNSUPPORTED",
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "bounded-domain proof allocation failed"
                    : contract_unsupported
                        ? "node lacks a canonical native shape contract; re-export after native contract support is added"
                        : "built-in backend cannot attest the complete declared shape domain",
                    0);
            }
        }
        if (status == VX_STATUS_OK &&
            compiled->builtin_backend != VOLVOXAI_BACKEND_CPU &&
            validation->node_count > 0) {
            validation->bounded_gpu_value_domain_proven =
                vx_native_gpu_backend(compiled->builtin_backend);
            VxEngineStateScope validation_scope =
                vx_engine_state_scope_enter(validation);
            int forward_status = volvoxai_engine_forward();
            vx_engine_state_scope_leave(validation_scope);
            if (forward_status != 0) {
                status = VX_STATUS_BACKEND_UNSUPPORTED;
                vx_report_write(&compile_report, status, VX_STAGE_COMPILE,
                                selected_backend,
                                vx_builtin_device_identity(selected_backend),
                                "BACKEND_UNSUPPORTED",
                                "built-in backend cannot execute the complete graph",
                                0);
            }
        }
        /* Vulkan's fixed tail scratch also contains every shape-invariant
         * dispatch payload from one whole graph forward. The first proof is
         * intentionally only a gate before execution. Re-run it after the
         * validation forward so the queried high-water mark is included in
         * the authoritative resource bound and maximum-domain reservation. */
        if (status == VX_STATUS_OK &&
            compiled->builtin_backend == VOLVOXAI_BACKEND_VULKAN) {
            status = vx_builtin_bounded_domain_proof(
                model, compiled->weights, compiled->builtin_backend,
                validation, &resource_domain, domain_evidence,
                sizeof(domain_evidence));
            if (status != VX_STATUS_OK) {
                int contract_unsupported =
                    strstr(domain_evidence,
                           "required=canonical-native-shape-contract") != NULL;
                vx_report_write(
                    &compile_report, status, VX_STAGE_COMPILE,
                    selected_backend,
                    vx_builtin_device_identity(selected_backend),
                    status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
                    contract_unsupported
                        ? "CANONICAL_SHAPE_CONTRACT_UNSUPPORTED"
                        : "BOUNDED_DOMAIN_UNSUPPORTED",
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "bounded-domain proof allocation failed"
                    : contract_unsupported
                        ? "node lacks a canonical native shape contract; re-export after native contract support is added"
                        : "built-in backend cannot attest the complete declared shape domain",
                    0);
            }
        }
        if (status == VX_STATUS_OK) {
            compiled->maximum_cpu_typed_workspace_bytes =
                resource_domain.maximum_typed_scratch_bytes;
            compiled->maximum_resident_bytes =
                resource_domain.maximum_resident_bytes;
        }
        if (status == VX_STATUS_OK || status == VX_STATUS_BACKEND_UNSUPPORTED)
            vx_report_set_builtin_route(&compile_report, validation,
                                        selected_backend);
        if (domain_evidence[0]) {
            size_t used = strlen(compile_report.route_evidence);
            if (used < sizeof(compile_report.route_evidence))
                snprintf(compile_report.route_evidence + used,
                         sizeof(compile_report.route_evidence) - used,
                         ";%s", domain_evidence);
        }
        if (status == VX_STATUS_OK)
            status = vx_builtin_compile_route_status(
                &compile_report, resolved.operator_fallback);
        if (status != VX_STATUS_OK)
            vx_report_set_builtin_failure_node(&compile_report, validation);
        {
            VxExecutionContext temporary = {0};
            temporary.engine_state = validation;
            temporary.engine_state_initialized = 1;
            /* shutdown is also the rollback path for partial initialization */
            temporary.engine_loaded = 1;
            vx_private_engine_unload(&temporary);
        }
        if (status != VX_STATUS_OK) {
            vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
            compile_report.policy_mode = resolved.mode;
            compile_report.operator_fallback = resolved.operator_fallback;
            compile_report.candidate_count = 1u;
            snprintf(compile_report.candidate_outcomes,
                     sizeof(compile_report.candidate_outcomes), "%s:rejected",
                     selected_backend);
            compile_report.tier_fallback_used = 0;
            compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
            vx_report_publish(report, &compile_report);
            vx_compiled_destroy(compiled);
            return status;
        }
    }
    vx_report_set_lineage(&compile_report, NULL, model, compiled, NULL);
    compile_report.policy_mode = resolved.mode;
    compile_report.operator_fallback = resolved.operator_fallback;
    compile_report.candidate_count = 1u;
    compile_report.tier_fallback_used = 0;
    snprintf(compile_report.candidate_outcomes,
             sizeof(compile_report.candidate_outcomes),
             "%s:selected", selected_backend);
    if (!compile_report.fallback_evidence[0])
        snprintf(compile_report.fallback_evidence,
                 sizeof(compile_report.fallback_evidence), "%s",
                 "tier=none;operator=none");
    if (!compile_report.decode_state[0])
        snprintf(compile_report.decode_state, sizeof(compile_report.decode_state),
                 "%s", "none");
    compile_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
    compiled->report = compile_report;
    vx_report_publish(report, &compile_report);
    *out_compiled = compiled;
    return VX_STATUS_OK;
}

static const char* vx_candidate_outcome(VxStatus status,
                                        const VxReport* report) {
    if (status == VX_STATUS_OK) return "selected";
    if (status == VX_STATUS_BACKEND_UNAVAILABLE ||
        status == VX_STATUS_BACKEND_REQUIRED) return "unavailable";
    if (status == VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN)
        return "operator-fallback-forbidden";
    if (status == VX_STATUS_INVALID_GRAPH) return "graph-rejected";
    if (status == VX_STATUS_OUT_OF_MEMORY) return "out-of-memory";
    if (report && !strcmp(report->reason, "BACKEND_UNSUPPORTED"))
        return "unsupported";
    return "rejected";
}

static void vx_candidate_outcomes_append(char* outcomes,
                                         size_t capacity,
                                         const char* backend,
                                         const char* outcome) {
    size_t used;
    if (!outcomes || !capacity || !backend || !outcome) return;
    used = strlen(outcomes);
    if (used >= capacity) return;
    snprintf(outcomes + used, capacity - used, "%s%s:%s",
             used ? ";" : "", backend, outcome);
}

VxStatus vx_model_compile(VxModel* model,
                          const VxBackendPolicy* policy,
                          VxCompiledModel** out_compiled,
                          VxReport* report) {
    static const char* const default_backends[] = { "cpu" };
    VxBackendPolicy resolved = VX_BACKEND_POLICY_INIT;
    VxReport last_report = VX_REPORT_INIT;
    char outcomes[VX_REPORT_CANDIDATES_CAPACITY] = {0};
    double compile_started_ms = vx_report_now_ms();
    VxStatus status = VX_STATUS_BACKEND_UNAVAILABLE;
    int runtime_closed;

    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !out_compiled || (policy && !vx_policy_valid(policy)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_COMPILE,
                       NULL, "INVALID_POLICY", "backend policy is invalid");
    *out_compiled = NULL;
    if (policy) resolved = *policy;
    if (!resolved.backends && resolved.backend_count == 0u) {
        resolved.backends = default_backends;
        resolved.backend_count = 1u;
    }
    if (!vx_policy_valid(&resolved))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_COMPILE,
                       NULL, "INVALID_POLICY", "backend policy is invalid");

    pthread_mutex_lock(&model->runtime->mutex);
    runtime_closed = model->runtime->closed;
    pthread_mutex_unlock(&model->runtime->mutex);
    if (runtime_closed) {
        status = vx_fail(report, VX_STATUS_HANDLE_DISPOSED, VX_STAGE_COMPILE,
                         NULL, "HANDLE_DISPOSED", "runtime is closed");
        vx_report_set_lineage(report, NULL, model, NULL, NULL);
        return status;
    }

    for (size_t index = 0; index < resolved.backend_count; index++) {
        VxCompiledModel* candidate = NULL;
        VxReport candidate_report = VX_REPORT_INIT;
        status = vx_model_compile_candidate(model, &resolved,
                                            resolved.backends[index],
                                            &candidate, &candidate_report);
        vx_candidate_outcomes_append(outcomes, sizeof(outcomes),
                                     resolved.backends[index],
                                     vx_candidate_outcome(status,
                                                          &candidate_report));
        last_report = candidate_report;
        if (status == VX_STATUS_OK) {
            for (size_t skipped = index + 1u;
                 skipped < resolved.backend_count; skipped++)
                vx_candidate_outcomes_append(outcomes, sizeof(outcomes),
                                             resolved.backends[skipped],
                                             "not-tried");
            candidate_report.candidate_count =
                (uint32_t)resolved.backend_count;
            candidate_report.tier_fallback_used = index > 0u;
            snprintf(candidate_report.candidate_outcomes,
                     sizeof(candidate_report.candidate_outcomes), "%s",
                     outcomes);
            if (index > 0u) {
                snprintf(candidate_report.fallback_evidence,
                         sizeof(candidate_report.fallback_evidence),
                         "tier=%s->%s;selected-index=%zu;operator=%s",
                         resolved.backends[0], resolved.backends[index], index,
                         candidate_report.operator_fallback_used ?
                             "used" : "none");
            }
            candidate_report.compile_time_ms =
                vx_report_now_ms() - compile_started_ms;
            candidate->report = candidate_report;
            *out_compiled = candidate;
            vx_report_publish(report, &candidate_report);
            return VX_STATUS_OK;
        }
        if (resolved.mode == VX_BACKEND_REQUIRE ||
            status == VX_STATUS_OUT_OF_MEMORY ||
            status == VX_STATUS_INTERNAL) break;
    }

    last_report.candidate_count = (uint32_t)resolved.backend_count;
    last_report.tier_fallback_used = 0;
    snprintf(last_report.candidate_outcomes,
             sizeof(last_report.candidate_outcomes), "%s", outcomes);
    last_report.compile_time_ms = vx_report_now_ms() - compile_started_ms;
    last_report.policy_mode = resolved.mode;
    last_report.operator_fallback = resolved.operator_fallback;
    if (resolved.mode == VX_BACKEND_PREFER &&
        status == VX_STATUS_BACKEND_UNAVAILABLE) {
        snprintf(last_report.reason, sizeof(last_report.reason), "%s",
                 "BACKEND_UNAVAILABLE");
        snprintf(last_report.message, sizeof(last_report.message), "%s",
                 "no preferred backend candidate is available");
    }
    vx_report_set_lineage(&last_report, NULL, model, NULL, NULL);
    vx_report_publish(report, &last_report);
    return status;
}

void vx_compiled_model_retain(VxCompiledModel* compiled) {
    if (compiled)
        atomic_fetch_add_explicit(&compiled->references, 1, memory_order_relaxed);
}

void vx_compiled_model_release(VxCompiledModel* compiled) {
    if (!compiled || atomic_fetch_sub_explicit(&compiled->references, 1,
                                               memory_order_acq_rel) != 1) return;
    vx_compiled_destroy(compiled);
}

VxStatus vx_compiled_model_report(const VxCompiledModel* compiled,
                                  VxReport* report) {
    if (!compiled || !report || report->struct_size != sizeof(*report))
        return VX_STATUS_INVALID_ARGUMENT;
    *report = compiled->report;
    return VX_STATUS_OK;
}

static VxStatus vx_context_operation_begin(VxExecutionContext* context,
                                           VxContextOperation* operation,
                                           VxStage stage,
                                           VxReport* report) {
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!context || !operation)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage, NULL,
                       "INVALID_CONTEXT", "execution context is NULL");
    memset(operation, 0, sizeof(*operation));
    pthread_mutex_lock(&context->queue_mutex);
    if (context->closing || context->closed) {
        VxStatus closed_status;
        pthread_mutex_unlock(&context->queue_mutex);
        closed_status = vx_fail(report, VX_STATUS_HANDLE_DISPOSED, stage,
                                context->compiled->backend,
                                "HANDLE_DISPOSED",
                                "execution context is closing or closed");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return closed_status;
    }
    operation->ticket = context->next_ticket++;
    operation->accepted = 1;
    while (operation->ticket != context->serving_ticket)
        pthread_cond_wait(&context->queue_condition, &context->queue_mutex);
    pthread_mutex_unlock(&context->queue_mutex);
    return VX_STATUS_OK;
}

static void vx_context_operation_end(VxExecutionContext* context,
                                     VxContextOperation* operation) {
    if (!context || !operation || !operation->accepted) return;
    pthread_mutex_lock(&context->queue_mutex);
    context->serving_ticket++;
    pthread_cond_broadcast(&context->queue_condition);
    pthread_mutex_unlock(&context->queue_mutex);
    operation->accepted = 0;
}

static int vx_context_options_valid(const VxContextOptions* options) {
    return options && options->struct_size == sizeof(*options) &&
        (options->decode_row_mode == VX_DECODE_ROW_DISABLED ||
         options->decode_row_mode == VX_DECODE_ROW_AUTO ||
         options->decode_row_mode == VX_DECODE_ROW_REQUIRED) &&
        (options->require_incremental == 0 ||
         options->require_incremental == 1) &&
        (options->decode_row_mode != VX_DECODE_ROW_DISABLED ||
         !options->require_incremental);
}

static const char* vx_decode_mode_name(VolvoxAIDecodeMode mode) {
    switch (mode) {
        case VOLVOXAI_DECODE_MODE_ORDINARY_FORWARD: return "ordinary";
        case VOLVOXAI_DECODE_MODE_INCREMENTAL_DEPENDENCY: return "dependency";
        case VOLVOXAI_DECODE_MODE_INCREMENTAL_ROW: return "row";
        default: return "none";
    }
}

static void vx_report_set_builtin_decode_state(VxReport* report,
                                               VxExecutionContext* context) {
    VolvoxAIDecodeMode mode;
    VolvoxAIDecodeMode last;
    int seeded;
    if (!report || report->struct_size != sizeof(*report) || !context ||
        !context->decode_session) return;
    mode = volvoxai_engine_decode_session_mode(context->decode_session);
    last = volvoxai_engine_decode_session_last_execution_mode(
        context->decode_session);
    seeded = volvoxai_engine_decode_session_seeded(context->decode_session);
    snprintf(report->decode_state, sizeof(report->decode_state),
             "enabled=1;seeded=%d;mode=%s;last=%s", seeded,
             vx_decode_mode_name(mode), vx_decode_mode_name(last));
}

static void vx_report_set_context_decode_state(VxReport* report,
                                               VxExecutionContext* context) {
    if (!report || report->struct_size != sizeof(*report) || !context ||
        context->decode_row_mode == VX_DECODE_ROW_DISABLED) return;
    if (context->compiled->provider) {
        snprintf(report->decode_state, sizeof(report->decode_state),
                 "enabled=1;seeded=%d;mode=provider;last=none",
                 context->decode_seeded ? 1 : 0);
    } else {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        vx_report_set_builtin_decode_state(report, context);
        vx_engine_state_scope_leave(scope);
    }
}

static int vx_adapter_is_base(const VxAdapterRevisionRecord* adapter) {
    return adapter && adapter->name && !strcmp(adapter->name, "__base__");
}

static VxStatus vx_context_prepare_adapter(
    VxExecutionContext* context,
    const VxAdapterRevisionRecord* adapter,
    char* prepared_route,
    size_t prepared_route_capacity,
    VxReport* report) {
    if (!context || !adapter || !prepared_route || !prepared_route_capacity)
        return VX_STATUS_INVALID_ARGUMENT;
    prepared_route[0] = '\0';
    if (context->compiled->provider) {
        if (vx_adapter_is_base(adapter) &&
            (!context->adapter || vx_adapter_is_base(context->adapter)))
            return VX_STATUS_OK;
        if (!context->compiled->provider->context_select_adapter)
            return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                           VX_STAGE_ADAPTER, context->compiled->backend,
                           "ADAPTER_UNSUPPORTED",
                           "backend does not implement adapter selection");
        return context->compiled->provider->context_select_adapter(
            context->provider_context, adapter->identity, adapter->revision,
            adapter->package_path, adapter->version_name, report);
    }
    if (!vx_adapter_is_base(adapter) && adapter->package_path) {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        int written = snprintf(prepared_route,
                               prepared_route_capacity,
                               "vx-%016" PRIx64 "-%016" PRIx64,
                               adapter->identity, adapter->revision);
        int loaded = written > 0 &&
            (size_t)written < prepared_route_capacity &&
            volvoxai_engine_adapter_load(
                adapter->package_path, prepared_route) == 0;
        if (!loaded && prepared_route[0]) {
            loaded = volvoxai_engine_adapter_route_begin(
                prepared_route) == 0;
            if (loaded) volvoxai_engine_adapter_route_end();
        }
        vx_engine_state_scope_leave(scope);
        if (!loaded) {
            return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                           VX_STAGE_ADAPTER, context->compiled->backend,
                           "ADAPTER_PACKAGE_REJECTED",
                           "adapter package is incompatible with the model");
        }
    }
    return VX_STATUS_OK;
}

static int vx_context_has_dynamic_logical_domain(
        const VxExecutionContext* context) {
    if (!context) return 0;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        const VxDeclaredTensor* tensor = &context->logical_tensors[index];
        for (uint32_t axis = 0; axis < tensor->rank; axis++)
            if (vx_dimension_domain_is_dynamic(tensor, axis)) return 1;
    }
    return 0;
}

static VxStatus vx_context_prepare_native_gpu_dynamic_reservation(
        VxExecutionContext* context) {
    VolvoxAIEngineMaximumTensor* maximums;
    VxEngineStateScope scope;
    int configured;
    int prewarmed;
    int logical_preload_demoted;
    int weights_retained;
    int preload_complete = 0;
    int reserved;
    int allocation_failed = 0;
    size_t arena_bytes;
    if (!context || !context->engine_state ||
        !vx_native_gpu_backend(context->compiled->builtin_backend) ||
        !context->logical_tensor_count)
        return VX_STATUS_INVALID_ARGUMENT;
    maximums = (VolvoxAIEngineMaximumTensor*)calloc(
        context->logical_tensor_count, sizeof(*maximums));
    if (!maximums) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < context->logical_tensor_count; index++) {
        maximums[index].name = context->logical_tensors[index].name;
        maximums[index].dtype = (int)context->logical_tensors[index].dtype;
        maximums[index].rank = (int)context->logical_tensors[index].rank;
        for (uint32_t axis = 0;
             axis < context->logical_tensors[index].rank; axis++)
            maximums[index].maximum_shape[axis] = (int)
                context->logical_tensors[index].maximums[axis];
        maximums[index].maximum_byte_size =
            context->logical_tensors[index].maximum_byte_size;
    }
    scope = vx_engine_state_scope_enter(context->engine_state);
    configured = volvoxai_engine_configure_dynamic_shape_domain(
        maximums, context->logical_tensor_count);
    /* The bootstrap projection is one legal minimum binding used only to
     * materialize every invariant weight/metadata/LUT route. The pointwise
     * maximum layout above, never this execution sample, proves the domain. */
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (configured == 0 &&
        context->compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA)
        cuda_graph_clear_allocation_failure();
#endif
    prewarmed = configured == 0 ? volvoxai_engine_forward() : -1;
    logical_preload_demoted = prewarmed == 0
        ? volvoxai_engine_demote_preloaded_logical_tensors_locked() : -1;
    weights_retained = logical_preload_demoted == 0
        ? volvoxai_engine_retain_preloaded_model_weights_locked() : -1;
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (weights_retained == 0 &&
        context->compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA)
        preload_complete = cuda_graph_complete_invariant_preload();
#endif
    reserved = preload_complete == 0
        ? volvoxai_engine_reserve_dynamic_shape_domain() : -1;
#if defined(VOLVOXAI_ENABLE_CUDA) && VOLVOXAI_ENABLE_CUDA
    if (context->compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA)
        allocation_failed = cuda_graph_last_allocation_failed();
#endif
    arena_bytes = context->engine_state->dynamic_arena_capacity_bytes;
    vx_engine_state_scope_leave(scope);
    free(maximums);
    if (configured == -2 || reserved == -2 ||
        (prewarmed != 0 && allocation_failed))
        return VX_STATUS_OUT_OF_MEMORY;
    if (configured != 0 || prewarmed != 0 ||
        logical_preload_demoted != 0 || weights_retained != 0 ||
        preload_complete != 0 || reserved != 0)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    if ((uint64_t)arena_bytes > UINT64_MAX - context->allocated_bytes)
        return VX_STATUS_OUT_OF_MEMORY;
    context->allocated_bytes += (uint64_t)arena_bytes;
    return VX_STATUS_OK;
}

VxStatus vx_compiled_model_create_context(
    VxCompiledModel* compiled,
    const VxContextOptions* options,
    VxExecutionContext** out_context,
    VxReport* report) {
    VxContextOptions resolved = VX_CONTEXT_OPTIONS_INIT;
    VxExecutionContext* context;
    VxStatus status;
    char prepared_route[96];
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!compiled || !out_context ||
        (options && !vx_context_options_valid(options)))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_CONTEXT_CREATE, compiled ? compiled->backend : NULL,
                       "INVALID_CONTEXT_OPTIONS", "context options are invalid");
    *out_context = NULL;
    if (options) resolved = *options;
    context = (VxExecutionContext*)calloc(1, sizeof(*context));
    if (!context)
        return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_CONTEXT_CREATE, compiled->backend,
                       "OUT_OF_MEMORY", "context allocation failed");
    atomic_init(&context->references, 1);
    context->compiled = compiled;
    context->adapter = compiled->adapter;
    vx_adapter_revision_retain(context->adapter);
    context->identity = vx_next_object_identity(compiled->model->runtime);
    context->allocated_bytes = sizeof(*context);
    context->decode_row_mode = resolved.decode_row_mode;
    context->require_incremental = resolved.require_incremental;
    vx_compiled_model_retain(compiled);
    if (pthread_mutex_init(&context->queue_mutex, NULL) != 0) {
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_CONTEXT_CREATE,
                       compiled->backend, "QUEUE_INIT_FAILED",
                       "context queue initialization failed");
    }
    if (pthread_cond_init(&context->queue_condition, NULL) != 0) {
        pthread_mutex_destroy(&context->queue_mutex);
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_CONTEXT_CREATE,
                       compiled->backend, "QUEUE_INIT_FAILED",
                       "context queue initialization failed");
    }
    if (compiled->provider) {
        status = compiled->provider->context_create(compiled->provider_compiled,
                                                     &resolved,
                                                     &context->provider_context,
                                                     report);
        if (status != VX_STATUS_OK || !context->provider_context) {
            if (context->provider_context)
                compiled->provider->context_destroy(context->provider_context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report,
                           status == VX_STATUS_OK ? VX_STATUS_BACKEND_UNAVAILABLE : status,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           "BACKEND_CONTEXT_CREATE_FAILED",
                           "backend context instance creation failed");
        }
    } else {
        context->engine_state = (VxEngineState*)calloc(1, sizeof(*context->engine_state));
        if (!context->engine_state ||
            vx_engine_state_init(context->engine_state) != 0) {
            free(context->engine_state);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           "STATE_ALLOCATION_FAILED", "context state allocation failed");
        }
        context->engine_state_initialized = 1;
        context->allocated_bytes += sizeof(*context->engine_state);
        context->dynamic_shape_stats =
            (VolvoxAIEngineDynamicShapeStats)
                VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
        status = vx_logical_tensors_load(
            compiled->model, &context->logical_tensors,
            &context->logical_tensor_count);
        if (status == VX_STATUS_OK)
            status = vx_logical_tensor_name_index_build(context);
        if (status != VX_STATUS_OK) {
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, status, VX_STAGE_CONTEXT_CREATE,
                           compiled->backend,
                           status == VX_STATUS_OUT_OF_MEMORY
                               ? "OUT_OF_MEMORY" : "LOGICAL_GRAPH_INVALID",
                           status == VX_STATUS_OUT_OF_MEMORY
                               ? "logical tensor table allocation failed"
                               : "closed v1 logical tensor table is invalid");
        }
        /* shutdown owns cleanup for both successful and partial loads */
        context->engine_loaded = 1;
        status = vx_private_engine_load(context->engine_state, compiled->model,
                                        compiled->weights,
                                        &compiled->model->runtime->options,
                                        compiled->builtin_backend,
                                        compiled->backend,
                                        report, VX_STAGE_CONTEXT_CREATE);
        if (status != VX_STATUS_OK) {
            vx_private_engine_unload(context);
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return status;
        }
        context->engine_state->bounded_gpu_value_domain_proven =
            vx_native_gpu_backend(compiled->builtin_backend);
        if (vx_native_gpu_backend(compiled->builtin_backend) &&
            vx_context_has_dynamic_logical_domain(context)) {
            if (compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA &&
                (!vx_adapter_is_base(context->adapter) ||
                 resolved.decode_row_mode != VX_DECODE_ROW_DISABLED)) {
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return vx_fail(
                    report, VX_STATUS_BACKEND_UNSUPPORTED,
                    VX_STAGE_CONTEXT_CREATE, compiled->backend,
                    !vx_adapter_is_base(compiled->adapter)
                        ? "DYNAMIC_CUDA_ADAPTER_UNSUPPORTED"
                        : "DYNAMIC_CUDA_DECODE_UNSUPPORTED",
                    !vx_adapter_is_base(compiled->adapter)
                        ? "dynamic CUDA contexts require the base adapter until adapter residency is proved"
                        : "dynamic CUDA decode requires a separately proved persistent device layout");
            }
            status = vx_context_prepare_native_gpu_dynamic_reservation(context);
            if (status != VX_STATUS_OK) {
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return vx_fail(
                    report, status, VX_STAGE_CONTEXT_CREATE,
                    compiled->backend,
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "DYNAMIC_DOMAIN_OUT_OF_MEMORY"
                        : "DYNAMIC_DOMAIN_RESERVATION_FAILED",
                    status == VX_STATUS_OUT_OF_MEMORY
                        ? "fixed native GPU dynamic-domain reservation exhausted host or device memory"
                        : "native GPU could not preload invariants and reserve the complete fixed dynamic domain");
            }
        }
        if (compiled->builtin_backend == VOLVOXAI_BACKEND_CPU) {
            const size_t workspace_bound =
                compiled->maximum_cpu_typed_workspace_bytes;
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            const int workspace_status =
                volvoxai_engine_configure_cpu_typed_workspace(
                    workspace_bound);
            vx_engine_state_scope_leave(scope);
            if (workspace_status != 0 || (uint64_t)workspace_bound >
                    UINT64_MAX - context->allocated_bytes) {
                VxStatus failure = vx_fail(
                    report, VX_STATUS_OUT_OF_MEMORY,
                    VX_STAGE_CONTEXT_CREATE, compiled->backend,
                    "WORKSPACE_ALLOCATION_FAILED",
                    "proved context CPU workspace allocation failed");
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return failure;
            }
            context->allocated_bytes += (uint64_t)workspace_bound;
        }
        if (resolved.decode_row_mode != VX_DECODE_ROW_DISABLED) {
            VolvoxAIDecodeSessionOptions decode_options =
                VOLVOXAI_DECODE_SESSION_OPTIONS_INIT;
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            decode_options.row_mode = resolved.decode_row_mode ==
                VX_DECODE_ROW_REQUIRED ? VOLVOXAI_DECODE_ROW_REQUIRED :
                                         VOLVOXAI_DECODE_ROW_AUTO;
            decode_options.require_incremental = resolved.require_incremental;
            context->decode_session = volvoxai_engine_decode_session_create(
                &decode_options);
            vx_engine_state_scope_leave(scope);
            if (!context->decode_session) {
                vx_private_engine_unload(context);
                pthread_cond_destroy(&context->queue_condition);
                pthread_mutex_destroy(&context->queue_mutex);
                vx_adapter_revision_release(context->adapter);
                vx_compiled_model_release(compiled);
                free(context);
                return vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                               VX_STAGE_CONTEXT_CREATE, compiled->backend,
                               "DECODE_MODE_UNSUPPORTED",
                               "requested decode mode is unavailable");
            }
        }
    }
    if (resolved.decode_row_mode != VX_DECODE_ROW_DISABLED &&
        compiled->model->input_count) {
        context->decode_input_shapes = (int64_t*)calloc(
            compiled->model->input_count * VX_MAX_TENSOR_RANK,
            sizeof(*context->decode_input_shapes));
        if (!context->decode_input_shapes) {
            if (compiled->provider) {
                if (compiled->provider->context_close)
                    (void)compiled->provider->context_close(
                        context->provider_context, NULL);
                compiled->provider->context_destroy(context->provider_context);
            } else {
                vx_private_engine_unload(context);
            }
            pthread_cond_destroy(&context->queue_condition);
            pthread_mutex_destroy(&context->queue_mutex);
            vx_adapter_revision_release(context->adapter);
            vx_compiled_model_release(compiled);
            free(context);
            return vx_fail(report, VX_STATUS_OUT_OF_MEMORY,
                           VX_STAGE_CONTEXT_CREATE, compiled->backend,
                           "OUT_OF_MEMORY",
                           "decode shape signature allocation failed");
        }
    }
    status = vx_context_prepare_adapter(context, context->adapter,
                                        prepared_route,
                                        sizeof(prepared_route), report);
    if (status != VX_STATUS_OK) {
        if (compiled->provider) {
            if (compiled->provider->context_close)
                (void)compiled->provider->context_close(
                    context->provider_context, NULL);
            compiled->provider->context_destroy(context->provider_context);
        } else {
            vx_private_engine_unload(context);
        }
        pthread_cond_destroy(&context->queue_condition);
        pthread_mutex_destroy(&context->queue_mutex);
        free(context->decode_input_shapes);
        vx_adapter_revision_release(context->adapter);
        vx_compiled_model_release(compiled);
        free(context);
        return status;
    }
    snprintf(context->adapter_route_version,
             sizeof(context->adapter_route_version), "%s", prepared_route);
    *out_context = context;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_CONTEXT_CREATE,
                    compiled->backend,
                    compiled->provider ? NULL :
                        vx_builtin_device_identity(compiled->backend),
                    "OK",
                    "execution context created", 0);
    vx_report_set_lineage(report, NULL, NULL, compiled, context);
    vx_report_set_plan_evidence(report, compiled);
    if (context->decode_row_mode != VX_DECODE_ROW_DISABLED) {
        if (!compiled->provider) {
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            vx_report_set_builtin_decode_state(report, context);
            vx_engine_state_scope_leave(scope);
        } else {
            snprintf(report->decode_state, sizeof(report->decode_state), "%s",
                     "enabled=1;seeded=0;mode=provider;last=none");
        }
    }
    return VX_STATUS_OK;
}

void vx_execution_context_retain(VxExecutionContext* context) {
    if (context)
        atomic_fetch_add_explicit(&context->references, 1, memory_order_relaxed);
}

VxStatus vx_execution_context_close(VxExecutionContext* context,
                                    VxReport* report) {
    uint64_t ticket;
    VxStatus status = VX_STATUS_OK;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!context)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_CLOSE,
                       NULL, "INVALID_CONTEXT", "execution context is NULL");
    pthread_mutex_lock(&context->queue_mutex);
    if (context->closed) {
        pthread_mutex_unlock(&context->queue_mutex);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE,
                        context->compiled->backend, NULL, "OK",
                        "execution context already closed", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return VX_STATUS_OK;
    }
    if (context->closing) {
        while (!context->closed)
            pthread_cond_wait(&context->queue_condition, &context->queue_mutex);
        pthread_mutex_unlock(&context->queue_mutex);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE,
                        context->compiled->backend, NULL, "OK",
                        "execution context closed", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return VX_STATUS_OK;
    }
    context->closing = 1;
    ticket = context->next_ticket++;
    while (ticket != context->serving_ticket)
        pthread_cond_wait(&context->queue_condition, &context->queue_mutex);
    pthread_mutex_unlock(&context->queue_mutex);

    if (context->compiled->provider) {
        if (context->compiled->provider->context_close)
            status = context->compiled->provider->context_close(
                context->provider_context, report);
    } else {
        vx_private_engine_unload(context);
    }

    pthread_mutex_lock(&context->queue_mutex);
    context->closed = 1;
    context->serving_ticket++;
    pthread_cond_broadcast(&context->queue_condition);
    pthread_mutex_unlock(&context->queue_mutex);
    if (status == VX_STATUS_OK)
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_CLOSE,
                        context->compiled->backend, NULL, "OK",
                        "execution context closed", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    return status;
}

void vx_execution_context_release(VxExecutionContext* context) {
    if (!context || atomic_fetch_sub_explicit(&context->references, 1,
                                              memory_order_acq_rel) != 1) return;
    (void)vx_execution_context_close(context, NULL);
    if (context->compiled->provider && context->provider_context)
        context->compiled->provider->context_destroy(context->provider_context);
    free(context->decode_input_shapes);
    pthread_cond_destroy(&context->queue_condition);
    pthread_mutex_destroy(&context->queue_mutex);
    vx_adapter_revision_release(context->adapter);
    vx_compiled_model_release(context->compiled);
    free(context);
}

size_t vx_execution_context_input_count(VxExecutionContext* context) {
    VxContextOperation operation;
    size_t count = 0;
    if (vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, NULL) !=
        VX_STATUS_OK) return 0;
    count = context->compiled->model->input_count;
    vx_context_operation_end(context, &operation);
    return count;
}

VxStatus vx_execution_context_input_spec(VxExecutionContext* context,
                                         size_t index,
                                         VxTensorSpec* spec,
                                         VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!spec || spec->struct_size != sizeof(*spec))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_INPUT,
                       context ? context->compiled->backend : NULL,
                       "INVALID_TENSOR_SPEC", "tensor spec buffer is invalid");
    status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    if (index >= context->compiled->model->input_count) {
        status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_INPUT,
                         context->compiled->backend, "INPUT_NOT_FOUND",
                         "input index is out of range");
    } else {
        vx_declared_tensor_spec(&context->compiled->model->inputs[index], spec);
        status = VX_STATUS_OK;
        vx_report_write(report, status, VX_STAGE_INPUT,
                        context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        "OK", "logical input spec returned", 0);
    }
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    vx_context_operation_end(context, &operation);
    return status;
}

VxStatus vx_execution_context_input_affine_quantization(
    VxExecutionContext* context,
    const char* name,
    VxAffineQuantization* quantization,
    VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!name || !name[0] || !quantization ||
        quantization->struct_size != sizeof(*quantization))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_INPUT,
                       context ? context->compiled->backend : NULL,
                       "INVALID_AFFINE_QUANTIZATION_REQUEST",
                       "input affine quantization request is invalid");
    quantization->defined = 0;
    quantization->scale = 0.0f;
    quantization->zero_point = 0;
    status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    if (context->compiled->provider) {
        status = vx_fail(
            report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_INPUT,
            context->compiled->backend, "INPUT_QUANTIZATION_INTROSPECTION_UNSUPPORTED",
            "provider does not expose input affine quantization introspection");
    } else {
        VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
        float scale = 0.0f;
        int zero_point = 0;
        int result = volvoxai_engine_input_affine_quantization(
            name, &scale, &zero_point);
        vx_engine_state_scope_leave(scope);
        if (result < 0) {
            status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_INPUT,
                             context->compiled->backend, "INPUT_NOT_FOUND",
                             "input name was not found");
        } else {
            quantization->defined = result;
            quantization->scale = result ? scale : 0.0f;
            quantization->zero_point = result ? zero_point : 0;
            status = VX_STATUS_OK;
            vx_report_write(report, status, VX_STAGE_INPUT,
                            context->compiled->backend,
                            vx_builtin_device_identity(context->compiled->backend),
                            "OK", "input affine quantization returned", 0);
        }
    }
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    vx_context_operation_end(context, &operation);
    return status;
}

typedef struct VxStagedBindingBatch {
    VxTensorBinding* bindings;
    size_t count;
    int64_t* resolved_input_shapes;
} VxStagedBindingBatch;

typedef enum VxContextRunKind {
    VX_CONTEXT_RUN_FORWARD = 0,
    VX_CONTEXT_RUN_DECODE_SEED = 1,
    VX_CONTEXT_RUN_DECODE_STEP = 2,
    VX_CONTEXT_RUN_PREFIX = 3
} VxContextRunKind;

static void vx_staged_binding_batch_clear(VxStagedBindingBatch* batch) {
    if (!batch) return;
    free(batch->bindings);
    free(batch->resolved_input_shapes);
    memset(batch, 0, sizeof(*batch));
}

static size_t vx_model_input_index(const VxModel* model, const char* name) {
    if (!model || !name) return SIZE_MAX;
    for (size_t index = 0; index < model->input_count; index++)
        if (!strcmp(model->inputs[index].name, name)) return index;
    return SIZE_MAX;
}

static int vx_binding_symbol_consistent(const VxModel* model,
                                        const int64_t* resolved_shapes,
                                        size_t input_index,
                                        uint32_t axis,
                                        int64_t extent) {
    const char* symbol = model->inputs[input_index].symbols[axis];
    if (!symbol) return 1;
    for (size_t other = 0; other < model->input_count; other++) {
        for (uint32_t other_axis = 0;
             other_axis < model->inputs[other].rank; other_axis++) {
            int64_t prior;
            if (!model->inputs[other].symbols[other_axis] ||
                strcmp(symbol, model->inputs[other].symbols[other_axis]))
                continue;
            prior = resolved_shapes[other * VX_MAX_TENSOR_RANK + other_axis];
            if (prior && prior != extent) return 0;
        }
    }
    return 1;
}

static VxStatus vx_stage_binding_batch(
    VxExecutionContext* context,
    const VxTensorBinding* inputs,
    size_t input_count,
    int require_complete,
    int decode_step,
    VxStagedBindingBatch* staged) {
    const VxModel* model = context->compiled->model;
    unsigned char* seen = NULL;
    size_t* indices = NULL;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    memset(staged, 0, sizeof(*staged));
    if ((input_count && !inputs) || input_count > model->input_count ||
        (require_complete && input_count != model->input_count) ||
        (decode_step && !context->decode_seeded)) return status;
    seen = model->input_count
        ? (unsigned char*)calloc(model->input_count, 1u) : NULL;
    indices = input_count ? (size_t*)calloc(input_count, sizeof(*indices)) : NULL;
    staged->resolved_input_shapes = model->input_count
        ? (int64_t*)calloc(model->input_count * VX_MAX_TENSOR_RANK,
                           sizeof(*staged->resolved_input_shapes)) : NULL;
    if ((model->input_count && (!seen || !staged->resolved_input_shapes)) ||
        (input_count && !indices)) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    if (decode_step && model->input_count)
        memcpy(staged->resolved_input_shapes, context->decode_input_shapes,
               model->input_count * VX_MAX_TENSOR_RANK * sizeof(int64_t));
    for (size_t index = 0; index < input_count; index++) {
        const VxTensorBinding* binding = &inputs[index];
        const VxDeclaredTensor* spec;
        size_t model_index;
        size_t elements = 1u;
        size_t element_size;
        size_t expected_bytes;
        if (binding->struct_size != sizeof(*binding) ||
            !binding->name || !binding->name[0] ||
            binding->location != VX_MEMORY_HOST ||
            binding->rank > VX_MAX_TENSOR_RANK || !binding->data) goto done;
        model_index = vx_model_input_index(model, binding->name);
        if (model_index == SIZE_MAX || seen[model_index]) goto done;
        seen[model_index] = 1u;
        indices[index] = model_index;
        spec = &model->inputs[model_index];
        if (binding->dtype != spec->dtype || binding->rank != spec->rank)
            goto done;
        for (uint32_t axis = 0; axis < binding->rank; axis++) {
            int64_t extent = binding->shape[axis];
            if (extent <= 0 || extent < spec->minimums[axis] ||
                extent > spec->maximums[axis] ||
                extent % spec->multiples[axis] != 0 ||
                (spec->kinds[axis] == VX_DIMENSION_FIXED &&
                 (spec->symbols[axis] || extent != spec->minimums[axis])) ||
                (spec->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
                 (!spec->symbols[axis] || !spec->symbols[axis][0])) ||
                (decode_step && extent !=
                    context->decode_input_shapes[
                        model_index * VX_MAX_TENSOR_RANK + axis]) ||
                !vx_binding_symbol_consistent(
                    model, staged->resolved_input_shapes,
                    model_index, axis, extent) ||
                (uint64_t)extent > SIZE_MAX / elements) goto done;
            elements *= (size_t)extent;
            staged->resolved_input_shapes[
                model_index * VX_MAX_TENSOR_RANK + axis] = extent;
        }
        element_size = vx_execution_dtype_size(binding->dtype);
        if (!element_size || elements > SIZE_MAX / element_size) goto done;
        expected_bytes = elements * element_size;
        if (binding->byte_size != expected_bytes) goto done;
    }
    if (require_complete)
        for (size_t index = 0; index < model->input_count; index++)
            if (!seen[index]) goto done;
    staged->bindings = input_count
        ? (VxTensorBinding*)calloc(input_count, sizeof(*staged->bindings)) : NULL;
    if (input_count && !staged->bindings) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    staged->count = input_count;
    for (size_t index = 0; index < input_count; index++) {
        const VxTensorBinding* source = &inputs[index];
        VxTensorBinding* target = &staged->bindings[index];
        *target = *source;
        target->struct_size = sizeof(*target);
        target->name = model->inputs[indices[index]].name;
        target->location = VX_MEMORY_HOST;
        /* The synchronous execution API borrows host input storage until it
         * returns. Keep normalization copy-free; the selected backend copies
         * or uploads only after the complete graph plan commits. */
        target->data = source->data;
    }
    status = VX_STATUS_OK;
done:
    free(indices);
    free(seen);
    if (status != VX_STATUS_OK) vx_staged_binding_batch_clear(staged);
    return status;
}

static const VxTensorBinding* vx_staged_binding_find(
        const VxStagedBindingBatch* staged, const char* name) {
    if (!staged || !name) return NULL;
    for (size_t index = 0; index < staged->count; index++)
        if (!strcmp(staged->bindings[index].name, name))
            return &staged->bindings[index];
    return NULL;
}

/* Return 1 with a public graph-input name, 2 for an immutable/statically
 * ranged source, and 0 for an unexpected device-produced origin. The compile
 * proof admits exactly this same producer subset. */
static int vx_native_gpu_public_i32_source(
        const VxEngineState* state,
        const char* tensor_name,
        const char** public_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!state || !tensor_name || !public_name || before_node < 0 ||
        before_node > state->node_count || depth < 0 ||
        depth > state->node_count)
        return 0;
    tensor = vx_native_gpu_validation_tensor_find(state, tensor_name);
    if (!tensor || tensor->dtype != T_I32) return 0;
    if (tensor->is_graph_input) {
        *public_name = tensor->name;
        return 1;
    }
    producer = vx_native_gpu_validation_node_producer(state, tensor_name);
    if (!producer) return 2;
    {
        int producer_index = vx_native_gpu_validation_node_index(
            state, producer);
        if (producer_index < 0 || producer_index >= before_node) return 0;
        before_node = producer_index;
    }
    if (!strcmp(producer->op, "Clip") ||
        !strcmp(producer->op, "ArgMax") ||
        !strcmp(producer->op, "QArgMax"))
        return 2;
    if (vx_native_gpu_value_preserving_node(producer))
        return vx_native_gpu_public_i32_source(
            state, vx_native_gpu_validation_node_data_name(producer),
            public_name, before_node, depth + 1);
    return 0;
}

static int vx_native_gpu_public_f32_source(
        const VxEngineState* state,
        const char* tensor_name,
        const char** public_name,
        int before_node,
        int depth) {
    const T* tensor;
    const Node* producer;
    if (!state || !tensor_name || !public_name || before_node < 0 ||
        before_node > state->node_count || depth < 0 ||
        depth > state->node_count)
        return 0;
    tensor = vx_native_gpu_validation_tensor_find(state, tensor_name);
    if (!tensor || tensor->dtype != T_F32) return 0;
    if (tensor->is_graph_input) {
        *public_name = tensor->name;
        return 1;
    }
    producer = vx_native_gpu_validation_node_producer(state, tensor_name);
    if (!producer) return 2;
    {
        int producer_index = vx_native_gpu_validation_node_index(
            state, producer);
        if (producer_index < 0 || producer_index >= before_node) return 0;
        before_node = producer_index;
    }
    if (vx_native_gpu_value_preserving_node(producer))
        return vx_native_gpu_public_f32_source(
            state, vx_native_gpu_validation_node_data_name(producer),
            public_name, before_node, depth + 1);
    return 0;
}

static const VolvoxAIEngineResolvedTensor* vx_resolved_tensor_find(
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t resolved_count,
        const char* name) {
    if (!resolved || !name) return NULL;
    for (size_t index = 0; index < resolved_count; index++)
        if (!strcmp(resolved[index].name, name)) return &resolved[index];
    return NULL;
}

static int vx_native_gpu_preflight_i32_binding(
        const VxStagedBindingBatch* staged,
        const char* public_name,
        int64_t minimum,
        int64_t maximum_exclusive) {
    const VxTensorBinding* binding = vx_staged_binding_find(
        staged, public_name);
    if (!binding) return 1; /* An unchanged decode-step input was preflighted. */
    if (binding->dtype != VX_DTYPE_I32 || !binding->data ||
        binding->byte_size % sizeof(int32_t) ||
        minimum >= maximum_exclusive)
        return 0;
    size_t count = binding->byte_size / sizeof(int32_t);
    for (size_t index = 0; index < count; index++) {
        int32_t value;
        memcpy(&value,
               (const unsigned char*)binding->data +
                   index * sizeof(value),
               sizeof(value));
        if ((int64_t)value < minimum ||
            (int64_t)value >= maximum_exclusive)
            return 0;
    }
    return 1;
}

static int vx_native_gpu_preflight_finite_binding(
        const VxStagedBindingBatch* staged,
        const char* public_name) {
    const VxTensorBinding* binding = vx_staged_binding_find(
        staged, public_name);
    if (!binding) return 1;
    if (binding->dtype != VX_DTYPE_F32 || !binding->data ||
        binding->byte_size % sizeof(float))
        return 0;
    size_t count = binding->byte_size / sizeof(float);
    for (size_t index = 0; index < count; index++) {
        float value;
        memcpy(&value,
               (const unsigned char*)binding->data +
                   index * sizeof(value),
               sizeof(value));
        if (!isfinite(value)) return 0;
    }
    return 1;
}

/* Value predicates are graph-wide and run after canonical shape resolution
 * but before volvoxai_engine_commit_dynamic_shape copies an input, publishes a
 * plan, uploads a byte, or allows the first GPU dispatch. */
static VxStatus vx_native_gpu_preflight_public_values(
        const VxExecutionContext* context,
        const VxStagedBindingBatch* staged,
        const VolvoxAIEngineResolvedTensor* resolved,
        size_t resolved_count) {
    const cJSON* nodes;
    const VxEngineState* state;
    if (!context || !staged ||
        !vx_native_gpu_backend(context->compiled->builtin_backend))
        return VX_STATUS_OK;
    state = context->engine_state;
    nodes = cJSON_GetObjectItemCaseSensitive(
        context->compiled->model->logical_graph, "nodes");
    if (!state || !cJSON_IsArray(nodes)) return VX_STATUS_INVALID_GRAPH;
    for (const cJSON* node = nodes->child; node; node = node->next) {
        const cJSON* op_item = cJSON_GetObjectItemCaseSensitive(
            node, "opType");
        const cJSON* inputs = cJSON_GetObjectItemCaseSensitive(
            node, "inputs");
        const char* op = cJSON_IsString(op_item)
            ? op_item->valuestring : NULL;
        const VxDeclaredTensor* logical_output = vx_native_gpu_node_output(
            node, context->logical_tensors, context->logical_tensor_count);
        const Node* validation_node = logical_output
            ? vx_native_gpu_validation_node_producer(
                  state, logical_output->name)
            : NULL;
        const int validation_node_index =
            vx_native_gpu_validation_node_index(state, validation_node);
        if (!op || !cJSON_IsObject(inputs) || validation_node_index < 0)
            return VX_STATUS_INVALID_GRAPH;
        if (!strcmp(op, "Gather")) {
            const cJSON* data_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "input");
            const cJSON* indices_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "indices");
            const char* public_name = NULL;
            int source_kind;
            int64_t axis_size = 0;
            if (!cJSON_IsString(data_ref) || !data_ref->valuestring ||
                !cJSON_IsString(indices_ref) || !indices_ref->valuestring)
                return VX_STATUS_INVALID_GRAPH;
            source_kind = vx_native_gpu_public_i32_source(
                state, indices_ref->valuestring, &public_name,
                validation_node_index, 0);
            if (!source_kind) return VX_STATUS_INVALID_GRAPH;
            if (source_kind != 1) continue;
            {
                const VolvoxAIEngineResolvedTensor* data =
                    vx_resolved_tensor_find(
                        resolved, resolved_count, data_ref->valuestring);
                const T* fixed = vx_native_gpu_validation_tensor_find(
                    state, data_ref->valuestring);
                if (data && data->rank > 0) axis_size = data->shape[0];
                else if (fixed && fixed->ndim > 0) axis_size = fixed->shape[0];
            }
            if (axis_size <= 0 ||
                !vx_native_gpu_preflight_i32_binding(
                    staged, public_name, -axis_size, axis_size))
                return VX_STATUS_INVALID_ARGUMENT;
        } else if (!strcmp(op, "Embedding") ||
                   !strcmp(op, "QEmbedding")) {
            const cJSON* ids_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "input");
            const cJSON* weight_ref = cJSON_GetObjectItemCaseSensitive(
                inputs, "weight");
            const T* weight;
            const char* public_name = NULL;
            int source_kind;
            if (!cJSON_IsString(ids_ref) || !ids_ref->valuestring ||
                !cJSON_IsString(weight_ref) || !weight_ref->valuestring)
                return VX_STATUS_INVALID_GRAPH;
            source_kind = vx_native_gpu_public_i32_source(
                state, ids_ref->valuestring, &public_name,
                validation_node_index, 0);
            if (!source_kind) return VX_STATUS_INVALID_GRAPH;
            if (source_kind != 1) continue;
            weight = vx_native_gpu_validation_tensor_find(
                state, weight_ref->valuestring);
            if (!weight || weight->ndim <= 0 || weight->shape[0] <= 0 ||
                !vx_native_gpu_preflight_i32_binding(
                    staged, public_name, 0, weight->shape[0]))
                return VX_STATUS_INVALID_ARGUMENT;
        } else if (!strcmp(op, "QGroupNorm") ||
                   !strcmp(op, "QLayerNorm")) {
            static const char* const roles[] = {"weight", "bias"};
            for (size_t role = 0; role < 2u; role++) {
                const cJSON* reference = cJSON_GetObjectItemCaseSensitive(
                    inputs, roles[role]);
                const char* public_name = NULL;
                int source_kind;
                if (!cJSON_IsString(reference) || !reference->valuestring)
                    return VX_STATUS_INVALID_GRAPH;
                source_kind = vx_native_gpu_public_f32_source(
                    state, reference->valuestring, &public_name,
                    validation_node_index, 0);
                if (!source_kind) return VX_STATUS_INVALID_GRAPH;
                if (source_kind == 1 &&
                    !vx_native_gpu_preflight_finite_binding(
                        staged, public_name))
                    return VX_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return VX_STATUS_OK;
}

static VxStatus vx_commit_builtin_bindings(
    VxExecutionContext* context,
    const VxStagedBindingBatch* staged,
    const int64_t* resolved_input_shapes,
    VxContextRunKind kind,
    const char* required_shape_signature) {
    VolvoxAIEngineInputBinding* engine_inputs = NULL;
    VolvoxAIEngineResolvedTensor* resolved = NULL;
    VolvoxAIEngineDynamicShapeStats stats =
        VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
    char* signature = NULL;
    size_t resolved_count = 0;
    double started_ms = vx_report_now_ms();
    VxEngineStateScope scope = vx_engine_state_scope_enter(context->engine_state);
    VxStatus failure_status = VX_STATUS_OK;
    int result = 0;
    volvoxai_engine_tensor_name_index_invalidate();
    volvoxai_engine_tensor_name_index_rebuild();
    if (staged->count)
        engine_inputs = (VolvoxAIEngineInputBinding*)calloc(
            staged->count, sizeof(*engine_inputs));
    if (staged->count && !engine_inputs) {
        failure_status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    for (size_t index = 0; index < staged->count; index++) {
        engine_inputs[index].name = staged->bindings[index].name;
        engine_inputs[index].dtype = (int)staged->bindings[index].dtype;
        engine_inputs[index].data = staged->bindings[index].data;
        engine_inputs[index].byte_size = staged->bindings[index].byte_size;
    }
    if (!vx_builtin_dynamic_shape_backend(context->compiled->backend)) {
        failure_status = vx_native_gpu_preflight_public_values(
            context, staged, NULL, 0u);
        if (failure_status != VX_STATUS_OK) goto done;
        result = volvoxai_engine_commit_input_bindings(
            engine_inputs, staged->count);
        goto done;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_STEP) {
        failure_status = vx_native_gpu_preflight_public_values(
            context, staged, NULL, 0u);
        if (failure_status != VX_STATUS_OK) goto done;
        result = context->committed_shape_signature
            ? volvoxai_engine_commit_input_bindings(
                  engine_inputs, staged->count)
            : -1;
        if (result == 0) stats = context->dynamic_shape_stats;
        stats.plan_cache_hit = 1;
        goto done;
    }
    if (!context->logical_tensor_count) {
        result = volvoxai_engine_commit_input_bindings(
            engine_inputs, staged->count);
        if (result == 0) stats.plan_cache_hit = 1;
        goto done;
    }
    {
        VxStatus plan_status = vx_resolved_plan_create(
            context, resolved_input_shapes, &resolved,
            &resolved_count, &signature);
        if (plan_status != VX_STATUS_OK) {
            if (context->engine_state->debug)
                fprintf(stderr,
                        "[debug] dynamic shape resolution failed status=%d tensors=%zu\n",
                        (int)plan_status, context->logical_tensor_count);
            failure_status = plan_status;
            goto done;
        }
    }
    if (required_shape_signature &&
        strcmp(required_shape_signature, signature)) {
        result = -4;
        goto done;
    }
    failure_status = vx_native_gpu_preflight_public_values(
        context, staged, resolved, resolved_count);
    if (failure_status != VX_STATUS_OK) goto done;
    result = volvoxai_engine_commit_dynamic_shape(
        signature, resolved, resolved_count, engine_inputs, staged->count,
        &stats);
    if (result == 0) {
        free(context->committed_shape_signature);
        context->committed_shape_signature = signature;
        signature = NULL;
        context->dynamic_shape_stats = stats;
    }
done:
    if (result == 0)
        context->dynamic_shape_bind_time_ms =
            vx_report_now_ms() - started_ms;
    vx_engine_state_scope_leave(scope);
    free(engine_inputs);
    free(resolved);
    free(signature);
    return failure_status != VX_STATUS_OK ? failure_status :
        result == 0 ? VX_STATUS_OK :
        result == -2 ? VX_STATUS_OUT_OF_MEMORY :
        result == -3 ? VX_STATUS_BACKEND_UNSUPPORTED :
        result == -4 ? VX_STATUS_INVALID_ARGUMENT :
                       VX_STATUS_EXECUTION_FAILED;
}

#if VOLVOXAI_ENABLE_TRAINING
VxStatus vx_model_internal_bind_authoring_inputs(
        VxModel* model,
        VxEngineState* state,
        VolvoxAIEngineBackend backend,
        const VxTensorBinding* inputs,
        size_t input_count,
        const char* required_shape_signature,
        char** out_shape_signature,
        VxReport* report) {
    VxCompiledModel compiled;
    VxExecutionContext context;
    VxStagedBindingBatch staged = {0};
    VxStatus status;
    const char* backend_name = vx_builtin_backend_name(backend);
    if (!model || !state || !backend_name || !out_shape_signature) {
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_INPUT, backend_name,
                       "INVALID_INPUT_BATCH",
                       "authoring input binding arguments are invalid");
    }
    *out_shape_signature = NULL;
    memset(&compiled, 0, sizeof(compiled));
    memset(&context, 0, sizeof(context));
    compiled.model = model;
    compiled.builtin_backend = backend;
    snprintf(compiled.backend, sizeof(compiled.backend), "%s", backend_name);
    context.compiled = &compiled;
    context.engine_state = state;
    status = vx_logical_tensors_load(
        model, &context.logical_tensors, &context.logical_tensor_count);
    if (status == VX_STATUS_OK)
        status = vx_logical_tensor_name_index_build(&context);
    if (status != VX_STATUS_OK) goto done;
    status = vx_stage_binding_batch(
        &context, inputs, input_count, 1, 0, &staged);
    if (status != VX_STATUS_OK) goto done;
    status = vx_commit_builtin_bindings(
        &context, &staged, staged.resolved_input_shapes,
        VX_CONTEXT_RUN_FORWARD, required_shape_signature);
    if (status == VX_STATUS_OK) {
        if (!context.committed_shape_signature) {
            status = VX_STATUS_INTERNAL;
            goto done;
        }
        *out_shape_signature = context.committed_shape_signature;
        context.committed_shape_signature = NULL;
    }
done:
    vx_staged_binding_batch_clear(&staged);
    vx_declared_outputs_free(context.logical_tensors,
                             context.logical_tensor_count);
    free(context.logical_tensor_name_slots);
    free(context.committed_shape_signature);
    if (status != VX_STATUS_OK) {
        vx_report_write(
            report, status, VX_STAGE_TRAINER_INPUT, backend_name,
            vx_builtin_device_identity(backend_name),
            status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
            status == VX_STATUS_BACKEND_UNSUPPORTED
                ? "BOUNDED_DOMAIN_UNSUPPORTED" :
            status == VX_STATUS_INVALID_GRAPH ? "INVALID_GRAPH" :
            status == VX_STATUS_INVALID_ARGUMENT && required_shape_signature
                ? "ACTIVATION_SIGNATURE_MISMATCH" :
                "INVALID_INPUT_BATCH",
            status == VX_STATUS_INVALID_ARGUMENT && required_shape_signature
                ? "input shape changed inside an accumulation window"
                : status == VX_STATUS_INVALID_GRAPH
                    ? "resolved shape plan violates the canonical graph shape contract"
                : "input batch violates the complete logical shape contract",
            0);
    }
    return status;
}
#endif

static void vx_owned_output_clear(VxOwnedOutput* output) {
    if (!output) return;
    free(output->name);
    free(output->data);
    memset(output, 0, sizeof(*output));
}

static void vx_result_destroy(VxResult* result) {
    for (size_t index = 0; index < result->output_count; index++)
        vx_owned_output_clear(&result->outputs[index]);
    free(result->outputs);
    free(result->input_shapes);
    vx_execution_context_release(result->context);
    free(result);
}

static const VxDeclaredOutput* vx_model_output_descriptor(
    const VxModel* model,
    const char* name) {
    if (!model || !name) return NULL;
    for (size_t index = 0; index < model->output_count; index++)
        if (!strcmp(model->outputs[index].name, name))
            return &model->outputs[index];
    return NULL;
}

static int vx_result_symbol_extent(const VxResult* result,
                                   const char* symbol,
                                   int64_t* out_extent) {
    const VxModel* model = result->context->compiled->model;
    int found = 0;
    int64_t extent = 0;
    if (!symbol || !result->input_shapes) return 0;
    for (size_t input = 0; input < model->input_count; input++) {
        for (uint32_t axis = 0; axis < model->inputs[input].rank; axis++) {
            int64_t candidate;
            if (!model->inputs[input].symbols[axis] ||
                strcmp(model->inputs[input].symbols[axis], symbol)) continue;
            candidate = result->input_shapes[
                input * VX_MAX_TENSOR_RANK + axis];
            if (!candidate || (found && candidate != extent)) return 0;
            extent = candidate;
            found = 1;
        }
    }
    if (found && out_extent) *out_extent = extent;
    return found;
}

static VxStatus vx_result_append(VxResult* result,
                                 const char* name,
                                 VxDataType dtype,
                                 const int64_t* shape,
                                 uint32_t rank,
                                 const void* data,
                                 size_t byte_size) {
    VxOwnedOutput* output;
    const VxDeclaredOutput* descriptor;
    size_t elements = 1u;
    size_t element_size;
    size_t expected_bytes;
    if (!result || !name || !name[0] ||
        rank > VX_MAX_TENSOR_RANK ||
        (rank && !shape) || (!data && byte_size))
        return VX_STATUS_INVALID_ARGUMENT;
    descriptor = vx_model_output_descriptor(
        result->context->compiled->model, name);
    if (!descriptor || !vx_execution_dtype_size(dtype) ||
        dtype != descriptor->dtype || rank != descriptor->rank)
        return VX_STATUS_INVALID_ARGUMENT;
    for (uint32_t axis = 0; axis < rank; axis++) {
        int64_t extent = shape[axis];
        int64_t resolved_symbol = 0;
        if (extent <= 0 || extent < descriptor->minimums[axis] ||
            extent > descriptor->maximums[axis] ||
            extent % descriptor->multiples[axis] != 0 ||
            (descriptor->kinds[axis] == VX_DIMENSION_FIXED &&
             (descriptor->symbols[axis] ||
              extent != descriptor->minimums[axis])) ||
            (descriptor->kinds[axis] == VX_DIMENSION_SYMBOLIC &&
             (!descriptor->symbols[axis] ||
              (vx_result_symbol_extent(result, descriptor->symbols[axis],
                                       &resolved_symbol) &&
               extent != resolved_symbol))) ||
            (uint64_t)extent > SIZE_MAX / elements)
            return VX_STATUS_INVALID_ARGUMENT;
        elements *= (size_t)extent;
    }
    element_size = vx_execution_dtype_size(dtype);
    if (!element_size || elements > SIZE_MAX / element_size)
        return VX_STATUS_INVALID_ARGUMENT;
    expected_bytes = elements * element_size;
    if (byte_size != expected_bytes) return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < result->output_count; index++)
        if (!strcmp(result->outputs[index].name, name))
            return VX_STATUS_INVALID_ARGUMENT;
    if (result->output_count == result->output_capacity) {
        size_t next_capacity = result->output_capacity
            ? result->output_capacity * 2u : 4u;
        VxOwnedOutput* next = (VxOwnedOutput*)realloc(
            result->outputs, next_capacity * sizeof(*next));
        if (!next) return VX_STATUS_OUT_OF_MEMORY;
        memset(next + result->output_capacity, 0,
               (next_capacity - result->output_capacity) * sizeof(*next));
        result->outputs = next;
        result->output_capacity = next_capacity;
    }
    output = &result->outputs[result->output_count];
    output->name = vx_string_copy(name);
    if (byte_size) output->data = (unsigned char*)malloc(byte_size);
    if (!output->name || (byte_size && !output->data)) {
        vx_owned_output_clear(output);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    output->dtype = dtype;
    output->rank = rank;
    for (uint32_t axis = 0; axis < rank; axis++) output->shape[axis] = shape[axis];
    output->byte_size = byte_size;
    if (byte_size) memcpy(output->data, data, byte_size);
    if (UINT64_MAX - result->snapshot_bytes < byte_size) {
        vx_owned_output_clear(output);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    result->snapshot_bytes += byte_size;
    result->output_count++;
    return VX_STATUS_OK;
}

static VxStatus vx_provider_output_write(void* user_data,
                                         const char* name,
                                         VxDataType dtype,
                                         const int64_t* shape,
                                         uint32_t rank,
                                         const void* data,
                                         size_t byte_size) {
    VxResult* result = (VxResult*)user_data;
    VxStatus status;
    if (!result) return VX_STATUS_INVALID_ARGUMENT;
    if (result->output_sink_status != VX_STATUS_OK)
        return result->output_sink_status;
    status = vx_result_append(result, name, dtype, shape, rank, data,
                              byte_size);
    if (status != VX_STATUS_OK) result->output_sink_status = status;
    return status;
}

static VxStatus vx_snapshot_cpu_outputs(VxExecutionContext* context,
                                        VxResult* result) {
    int output_count = volvoxai_engine_graph_output_count();
    if (output_count < 0) return VX_STATUS_INTERNAL;
    for (int index = 0; index < output_count; index++) {
        const char* name = volvoxai_engine_graph_output_name(index);
        long numel;
        int shape[8] = {0};
        int rank;
        int dtype;
        VxDataType execution_dtype;
        size_t element_size;
        size_t byte_size;
        int64_t shape64[VX_MAX_TENSOR_RANK] = {0};
        unsigned char* bytes;
        if (!name || volvoxai_engine_tensor_info_ex(name, &numel, shape, &rank,
                                                     &dtype, &element_size) != 0 ||
            rank < 0 || rank > (int)VX_MAX_TENSOR_RANK || numel < 0 ||
            (numel > 0 && (size_t)numel > SIZE_MAX / element_size))
            return VX_STATUS_INTERNAL;
        execution_dtype = dtype == T_F16 ? VX_DTYPE_F32 : (VxDataType)dtype;
        element_size = dtype == T_F16 ? sizeof(float) : element_size;
        byte_size = (size_t)numel * element_size;
        bytes = byte_size ? (unsigned char*)malloc(byte_size) : NULL;
        if (byte_size && !bytes) return VX_STATUS_OUT_OF_MEMORY;
        if (byte_size &&
            (dtype == T_F16
                 ? volvoxai_engine_copy_tensor_f32(
                       name, (float*)bytes, numel)
                 : volvoxai_engine_copy_tensor_raw(name, bytes, byte_size)) != 0) {
            free(bytes);
            return VX_STATUS_EXECUTION_FAILED;
        }
        for (int axis = 0; axis < rank; axis++) shape64[axis] = shape[axis];
        VxStatus status = vx_result_append(
            result, name, execution_dtype, shape64, (uint32_t)rank, bytes,
            byte_size);
        free(bytes);
        if (status != VX_STATUS_OK)
            return status == VX_STATUS_OUT_OF_MEMORY
                ? status : VX_STATUS_EXECUTION_FAILED;
    }
    (void)context;
    return VX_STATUS_OK;
}

static VxStatus vx_execution_context_run(VxExecutionContext* context,
                                         VxContextRunKind kind,
                                         int32_t position,
                                         const VxTensorBinding* inputs,
                                         size_t input_count,
                                         VxResult** out_result,
                                         VxReport* report) {
    VxContextOperation operation;
    VxStagedBindingBatch staged = {0};
    VxResult* result;
    VxStatus status;
    uint64_t execution_id;
    VxReport execution_report = VX_REPORT_INIT;
    double execution_started_ms;
    int attestation_failure = 0;
    int fallback_forbidden_failure = 0;
    const int decode_operation = kind == VX_CONTEXT_RUN_DECODE_SEED ||
        kind == VX_CONTEXT_RUN_DECODE_STEP;
    VxStage stage = decode_operation ? VX_STAGE_DECODE : VX_STAGE_EXECUTE;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!out_result)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                       context ? context->compiled->backend : NULL,
                       "INVALID_RESULT_POINTER", "result output pointer is NULL");
    *out_result = NULL;
    status = vx_context_operation_begin(context, &operation, stage, report);
    if (status != VX_STATUS_OK) return status;
    if (decode_operation &&
        context->decode_row_mode == VX_DECODE_ROW_DISABLED) {
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, stage,
                         context->compiled->backend,
                         "DECODE_NOT_ENABLED",
                         "execution context was not created for decode");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_STEP && position < -1) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                         context->compiled->backend,
                         "INVALID_DECODE_POSITION",
                         "decode position must be -1 or non-negative");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        if (decode_operation)
            vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_PREFIX && position <= 0) {
        status = vx_fail(report, VX_STATUS_INVALID_ARGUMENT, stage,
                         context->compiled->backend,
                         "INVALID_PREFIX_ROW_COUNT",
                         "prefix row count must be positive");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        vx_context_operation_end(context, &operation);
        return status;
    }
    status = vx_stage_binding_batch(
        context, inputs, input_count,
        kind != VX_CONTEXT_RUN_DECODE_STEP,
        kind == VX_CONTEXT_RUN_DECODE_STEP, &staged);
    if (status != VX_STATUS_OK) {
        status = vx_fail(report, status, stage, context->compiled->backend,
                         status == VX_STATUS_OUT_OF_MEMORY ?
                             "OUT_OF_MEMORY" : "INVALID_INPUT_BATCH",
                         status == VX_STATUS_OUT_OF_MEMORY ?
                             "input batch staging allocation failed" :
                             "input batch violates the complete logical shape contract");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        if (decode_operation)
            vx_report_set_context_decode_state(report, context);
        vx_context_operation_end(context, &operation);
        return status;
    }
    execution_started_ms = vx_report_now_ms();
    result = (VxResult*)calloc(1, sizeof(*result));
    if (!result) {
        vx_staged_binding_batch_clear(&staged);
        vx_context_operation_end(context, &operation);
        vx_report_write(&execution_report, VX_STATUS_OUT_OF_MEMORY,
                        stage, context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        "OUT_OF_MEMORY", "result allocation failed", 0);
        vx_report_set_lineage(&execution_report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(&execution_report, context->compiled);
        execution_report.execution_time_ms =
            vx_report_now_ms() - execution_started_ms;
        vx_report_publish(report, &execution_report);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    atomic_init(&result->references, 1);
    result->context = context;
    result->input_shapes = staged.resolved_input_shapes;
    result->input_shape_count = context->compiled->model->input_count;
    staged.resolved_input_shapes = NULL;
    vx_execution_context_retain(context);
    execution_id = vx_next_execution_identity(
        context->compiled->model->runtime);
    result->execution_id = execution_id;
    execution_report = context->compiled->report;
    execution_report.struct_size = sizeof(execution_report);
    execution_report.execution_id = execution_id;
    execution_report.stage = stage;
    execution_report.compile_time_ms = context->compiled->report.compile_time_ms;
    execution_report.execution_time_ms = 0.0;
    execution_report.offending_node[0] = '\0';
    if (context->compiled->provider) {
        VxBackendOutputSink sink = {
            .struct_size = sizeof(sink),
            .user_data = result,
            .write = vx_provider_output_write,
        };
        if (kind == VX_CONTEXT_RUN_FORWARD) {
            status = context->compiled->provider->context_execute(
                context->provider_context, staged.bindings, staged.count,
                &sink, &execution_report);
        } else if (kind == VX_CONTEXT_RUN_DECODE_SEED &&
                   context->compiled->provider->context_decode_seed) {
            status = context->compiled->provider->context_decode_seed(
                context->provider_context, staged.bindings, staged.count,
                &sink, &execution_report);
        } else if (kind == VX_CONTEXT_RUN_DECODE_STEP &&
                   context->compiled->provider->context_decode_step) {
            status = context->compiled->provider->context_decode_step(
                context->provider_context, position,
                staged.bindings, staged.count, &sink, &execution_report);
        } else {
            status = VX_STATUS_BACKEND_UNSUPPORTED;
        }
        if (result->output_sink_status != VX_STATUS_OK)
            status = result->output_sink_status == VX_STATUS_OUT_OF_MEMORY
                ? VX_STATUS_OUT_OF_MEMORY : VX_STATUS_EXECUTION_FAILED;
        if (status == VX_STATUS_OK) {
            attestation_failure = !execution_report.route_attested ||
                                  !execution_report.route_evidence[0];
            fallback_forbidden_failure =
                context->compiled->operator_fallback ==
                    VX_OPERATOR_FALLBACK_FORBID &&
                execution_report.operator_fallback_used;
            if (fallback_forbidden_failure)
                status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
            else if (attestation_failure)
                status = VX_STATUS_EXECUTION_FAILED;
        }
    } else {
        VxEngineStateScope scope;
        VxStatus binding_status =
            vx_commit_builtin_bindings(
                context, &staged, result->input_shapes, kind, NULL);
        int forward_status = binding_status == VX_STATUS_OK ? 0 : -1;
        int adapter_route_active = 0;
        scope = vx_engine_state_scope_enter(context->engine_state);
#if defined(VOLVOXAI_PUBLIC_API_TESTING)
        if (g_cpu_execute_hook)
            g_cpu_execute_hook(g_cpu_execute_hook_user_data);
#endif
        volvoxai_engine_tensor_name_index_invalidate();
        volvoxai_engine_tensor_name_index_rebuild();
        if (context->adapter_route_version[0]) {
            adapter_route_active = volvoxai_engine_adapter_route_begin(
                context->adapter_route_version) == 0;
            if (!adapter_route_active) forward_status = -1;
        }
        if (forward_status == 0 && kind == VX_CONTEXT_RUN_DECODE_SEED)
            forward_status = volvoxai_engine_decode_session_seed(
                context->decode_session);
        else if (forward_status == 0 && kind == VX_CONTEXT_RUN_DECODE_STEP)
            forward_status = volvoxai_engine_decode_session_step(
                context->decode_session, position);
        else if (forward_status == 0 && kind == VX_CONTEXT_RUN_PREFIX)
            forward_status = volvoxai_engine_forward_prefix(position);
        else if (forward_status == 0)
            forward_status = volvoxai_engine_forward();
        status = forward_status == 0 ? vx_snapshot_cpu_outputs(context, result) :
                 binding_status != VX_STATUS_OK ? binding_status :
                                                   VX_STATUS_EXECUTION_FAILED;
        if (adapter_route_active) volvoxai_engine_adapter_route_end();
        if (decode_operation)
            vx_report_set_builtin_decode_state(&execution_report, context);
        vx_engine_state_scope_leave(scope);
        vx_report_set_builtin_route(&execution_report, context->engine_state,
                                    context->compiled->backend);
        vx_report_append_dynamic_shape(&execution_report, context);
        if (status == VX_STATUS_OK) {
            attestation_failure = !execution_report.route_attested ||
                                  !execution_report.route_evidence[0];
            fallback_forbidden_failure =
                context->compiled->operator_fallback ==
                    VX_OPERATOR_FALLBACK_FORBID &&
                execution_report.operator_fallback_used;
            if (fallback_forbidden_failure)
                status = VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN;
            else if (attestation_failure)
                status = VX_STATUS_EXECUTION_FAILED;
        }
        if (status != VX_STATUS_OK)
            vx_report_set_builtin_failure_node(&execution_report,
                                               context->engine_state);
    }
    if (status == VX_STATUS_OK &&
        result->output_count != context->compiled->model->output_count)
        status = VX_STATUS_EXECUTION_FAILED;
    if (status != VX_STATUS_OK) {
        execution_report.status = status;
        execution_report.stage = stage;
        execution_report.execution_id = execution_id;
        snprintf(execution_report.backend, sizeof(execution_report.backend), "%s",
                 context->compiled->backend);
        if (!context->compiled->provider)
            snprintf(execution_report.device, sizeof(execution_report.device),
                     "%s", vx_builtin_device_identity(
                         context->compiled->backend));
        if (status == VX_STATUS_OUT_OF_MEMORY) {
            snprintf(execution_report.reason,
                     sizeof(execution_report.reason), "%s",
                     "OUT_OF_MEMORY");
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "resolved shape plan or arena growth allocation failed");
        } else if (status == VX_STATUS_INVALID_ARGUMENT) {
            snprintf(execution_report.reason,
                     sizeof(execution_report.reason), "%s",
                     "INVALID_INPUT_VALUES");
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "input values violate a graph-wide native GPU preflight predicate");
        } else if (status == VX_STATUS_INVALID_GRAPH) {
            snprintf(execution_report.reason,
                     sizeof(execution_report.reason), "%s",
                     "INVALID_GRAPH");
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "resolved shape plan violates the canonical graph shape contract");
        } else if (status == VX_STATUS_BACKEND_UNSUPPORTED) {
            snprintf(execution_report.reason,
                     sizeof(execution_report.reason), "%s",
                     "BACKEND_UNSUPPORTED");
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "backend does not implement the requested execution operation");
        } else if (status == VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN) {
            snprintf(execution_report.reason,
                     sizeof(execution_report.reason), "%s",
                     "OPERATOR_FALLBACK_FORBIDDEN");
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     "execution used an operator fallback forbidden by the compiled policy");
        } else {
            snprintf(execution_report.reason,
                     sizeof(execution_report.reason), "%s",
                     "EXECUTION_FAILED");
            snprintf(execution_report.message,
                     sizeof(execution_report.message), "%s",
                     attestation_failure
                         ? "execution route did not provide its compiled attestation"
                         : "execution or output snapshot failed");
        }
        if (decode_operation &&
            context->compiled->provider && !execution_report.decode_state[0])
            snprintf(execution_report.decode_state,
                     sizeof(execution_report.decode_state), "%s",
                     "enabled=1;seeded=unknown;mode=provider;last=failed");
        vx_report_set_lineage(&execution_report, NULL, NULL, NULL, context);
        execution_report.result_bytes = result->snapshot_bytes;
        execution_report.execution_time_ms =
            vx_report_now_ms() - execution_started_ms;
        vx_report_publish(report, &execution_report);
        vx_staged_binding_batch_clear(&staged);
        vx_result_destroy(result);
        vx_context_operation_end(context, &operation);
        return status;
    }
    if (kind == VX_CONTEXT_RUN_DECODE_SEED) {
        if (context->compiled->model->input_count)
            memcpy(context->decode_input_shapes, result->input_shapes,
                   context->compiled->model->input_count *
                       VX_MAX_TENSOR_RANK * sizeof(int64_t));
        context->decode_seeded = 1;
    }
    *out_result = result;
    execution_report.status = VX_STATUS_OK;
    execution_report.stage = stage;
    execution_report.execution_id = execution_id;
    snprintf(execution_report.backend, sizeof(execution_report.backend), "%s",
             context->compiled->backend);
    if (!context->compiled->provider)
        snprintf(execution_report.device, sizeof(execution_report.device), "%s",
                 vx_builtin_device_identity(context->compiled->backend));
    snprintf(execution_report.reason, sizeof(execution_report.reason), "%s", "OK");
    snprintf(execution_report.message, sizeof(execution_report.message), "%s",
             !decode_operation ?
                 "execution completed with owned output snapshots" :
                 "decode operation completed with owned output snapshots");
    if (decode_operation && context->compiled->provider &&
        !execution_report.decode_state[0])
        snprintf(execution_report.decode_state,
                 sizeof(execution_report.decode_state), "%s",
                 kind == VX_CONTEXT_RUN_DECODE_SEED ?
                     "enabled=1;seeded=1;mode=provider;last=seed" :
                     "enabled=1;seeded=1;mode=provider;last=step");
    execution_report.offending_node[0] = '\0';
    vx_report_set_lineage(&execution_report, NULL, NULL, NULL, context);
    execution_report.result_bytes = result->snapshot_bytes;
    vx_report_add_result_allocation(&execution_report, result);
    execution_report.execution_time_ms = vx_report_now_ms() - execution_started_ms;
    vx_result_capture_evidence(result, &execution_report);
    vx_report_publish(report, &execution_report);
    vx_staged_binding_batch_clear(&staged);
    vx_context_operation_end(context, &operation);
    return VX_STATUS_OK;
}

VxStatus vx_execution_context_execute(VxExecutionContext* context,
                                      const VxTensorBinding* inputs,
                                      size_t input_count,
                                      VxResult** out_result,
                                      VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_FORWARD, -1,
                                    inputs, input_count,
                                    out_result, report);
}

VxStatus vx_execution_context_execute_prefix(VxExecutionContext* context,
                                             int32_t row_count,
                                             const VxTensorBinding* inputs,
                                             size_t input_count,
                                             VxResult** out_result,
                                             VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_PREFIX, row_count,
                                    inputs, input_count,
                                    out_result, report);
}

VxStatus vx_execution_context_decode_seed(VxExecutionContext* context,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_DECODE_SEED, -1,
                                    inputs, input_count,
                                    out_result, report);
}

VxStatus vx_execution_context_decode_step(VxExecutionContext* context,
                                          int32_t position,
                                          const VxTensorBinding* inputs,
                                          size_t input_count,
                                          VxResult** out_result,
                                          VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_DECODE_STEP,
                                    position, inputs, input_count,
                                    out_result, report);
}

VxStatus vx_execution_context_decode_reset(VxExecutionContext* context,
                                           VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    status = vx_context_operation_begin(context, &operation, VX_STAGE_DECODE,
                                        report);
    if (status != VX_STATUS_OK) return status;
    if (context->decode_row_mode == VX_DECODE_ROW_DISABLED) {
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                         VX_STAGE_DECODE, context->compiled->backend,
                         "DECODE_NOT_ENABLED",
                         "execution context was not created for decode");
    } else if (context->compiled->provider) {
        if (context->compiled->provider->context_decode_reset)
            status = context->compiled->provider->context_decode_reset(
                context->provider_context, report);
        else
            status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED,
                             VX_STAGE_DECODE, context->compiled->backend,
                             "BACKEND_UNSUPPORTED",
                             "backend does not implement decode reset");
        if (status == VX_STATUS_OK) {
            vx_report_write(report, status, VX_STAGE_DECODE,
                            context->compiled->backend, NULL, "OK",
                            "decode state reset", 0);
            if (report && report->struct_size == sizeof(*report))
                snprintf(report->decode_state, sizeof(report->decode_state),
                         "%s",
                         "enabled=1;seeded=0;mode=provider;last=none");
        }
    } else {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        status = volvoxai_engine_decode_session_reset(context->decode_session) == 0
            ? VX_STATUS_OK : VX_STATUS_EXECUTION_FAILED;
        if (status == VX_STATUS_OK)
            vx_report_write(report, status, VX_STAGE_DECODE,
                            context->compiled->backend,
                            vx_builtin_device_identity(context->compiled->backend),
                            "OK", "decode state reset", 0);
        else
            vx_fail(report, status, VX_STAGE_DECODE,
                    context->compiled->backend,
                    "EXECUTION_FAILED", "decode reset failed");
        vx_report_set_builtin_decode_state(report, context);
        vx_engine_state_scope_leave(scope);
    }
    if (status == VX_STATUS_OK) {
        context->decode_seeded = 0;
        if (context->decode_input_shapes)
            memset(context->decode_input_shapes, 0,
                   context->compiled->model->input_count *
                       VX_MAX_TENSOR_RANK * sizeof(int64_t));
    }
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    if (status == VX_STATUS_OK &&
        report && report->struct_size == sizeof(*report)) {
        if (context->compiled->provider)
            snprintf(report->decode_state, sizeof(report->decode_state), "%s",
                     "enabled=1;seeded=0;mode=provider;last=none");
        else {
            VxEngineStateScope scope =
                vx_engine_state_scope_enter(context->engine_state);
            vx_report_set_builtin_decode_state(report, context);
            vx_engine_state_scope_leave(scope);
        }
    }
    vx_context_operation_end(context, &operation);
    return status;
}

static VxAdapterRevisionRecord* vx_model_find_adapter_revision(
    VxModel* model,
    uint64_t identity,
    uint64_t revision,
    int current) {
    VxAdapterRevisionRecord* selected = NULL;
    pthread_mutex_lock(&model->revision_mutex);
    if (current) {
        selected = atomic_load_explicit(&model->current_adapter,
                                        memory_order_acquire);
    } else {
        for (VxAdapterRevisionRecord* item = model->adapters;
             item; item = item->next) {
            if (item->identity == identity && item->revision == revision) {
                selected = item;
                break;
            }
        }
    }
    vx_adapter_revision_retain(selected);
    pthread_mutex_unlock(&model->revision_mutex);
    return selected;
}

static VxStatus vx_context_select_adapter_record(
    VxExecutionContext* context,
    VxAdapterRevisionRecord* selected,
    VxReport* report) {
    VxStatus status;
    char prepared_route[96];
    if (!context->compiled->provider &&
        context->compiled->builtin_backend == VOLVOXAI_BACKEND_CUDA &&
        vx_context_has_dynamic_logical_domain(context) &&
        !vx_adapter_is_base(selected)) {
        VxStatus failure;
        vx_adapter_revision_release(selected);
        failure = vx_fail(
            report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_ADAPTER,
            context->compiled->backend,
            "DYNAMIC_CUDA_ADAPTER_UNSUPPORTED",
            "dynamic CUDA adapter selection requires a separately proved atomic residency reservation");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return failure;
    }
    if (selected == context->adapter) {
        vx_adapter_revision_release(selected);
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_ADAPTER,
                        context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        "OK",
                        "adapter revision already selected", 0);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return VX_STATUS_OK;
    }
    status = vx_context_prepare_adapter(context, selected, prepared_route,
                                        sizeof(prepared_route), report);
    if (status != VX_STATUS_OK) {
        vx_adapter_revision_release(selected);
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
        return status;
    }
    if (!context->compiled->provider && context->decode_session) {
        VxEngineStateScope scope =
            vx_engine_state_scope_enter(context->engine_state);
        int reset = volvoxai_engine_decode_session_reset(
            context->decode_session);
        vx_engine_state_scope_leave(scope);
        if (reset != 0) {
            VxStatus failure;
            vx_adapter_revision_release(selected);
            failure = vx_fail(
                report, VX_STATUS_EXECUTION_FAILED, VX_STAGE_ADAPTER,
                context->compiled->backend,
                "DECODE_RESET_FAILED",
                "decode state could not be reset before adapter selection");
            vx_report_set_lineage(report, NULL, NULL, NULL, context);
            vx_report_set_plan_evidence(report, context->compiled);
            return failure;
        }
    }
    {
        VxAdapterRevisionRecord* previous = context->adapter;
        context->adapter = selected;
        snprintf(context->adapter_route_version,
                 sizeof(context->adapter_route_version), "%s",
                 prepared_route);
        vx_adapter_revision_release(previous);
    }
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_ADAPTER,
                    context->compiled->backend,
                    context->compiled->provider ? NULL :
                        vx_builtin_device_identity(context->compiled->backend),
                    "OK",
                    "adapter revision selected", 0);
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    return status;
}

VxStatus vx_execution_context_select_adapter(
    VxExecutionContext* context,
    const VxAdapterRevision* revision,
    VxReport* report) {
    VxContextOperation operation;
    VxAdapterRevisionRecord* selected;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!revision || revision->struct_size != sizeof(*revision) ||
        !revision->adapter_id || !revision->adapter_revision)
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_ADAPTER,
                       context ? context->compiled->backend : NULL,
                       "INVALID_ADAPTER_REVISION",
                       "adapter revision key is invalid");
    status = vx_context_operation_begin(context, &operation,
                                        VX_STAGE_ADAPTER, report);
    if (status != VX_STATUS_OK) return status;
    selected = vx_model_find_adapter_revision(context->compiled->model,
                                              revision->adapter_id,
                                              revision->adapter_revision, 0);
    if (!selected) {
        status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_ADAPTER,
                         context->compiled->backend,
                         "ADAPTER_REVISION_NOT_FOUND",
                         "adapter revision is not published by this model");
        vx_report_set_lineage(report, NULL, NULL, NULL, context);
        vx_report_set_plan_evidence(report, context->compiled);
    } else {
        status = vx_context_select_adapter_record(context, selected, report);
    }
    vx_context_operation_end(context, &operation);
    return status;
}

VxStatus vx_execution_context_rebind_adapter(
    VxExecutionContext* context,
    VxReport* report) {
    VxContextOperation operation;
    VxAdapterRevisionRecord* selected;
    VxStatus status;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    status = vx_context_operation_begin(context, &operation,
                                        VX_STAGE_ADAPTER, report);
    if (status != VX_STATUS_OK) return status;
    selected = vx_model_find_adapter_revision(context->compiled->model,
                                              0, 0, 1);
    if (!selected) {
        status = vx_fail(report, VX_STATUS_INTERNAL, VX_STAGE_ADAPTER,
                         context->compiled->backend,
                         "ADAPTER_REVISION_MISSING",
                         "model has no published adapter revision");
    } else {
        status = vx_context_select_adapter_record(context, selected, report);
    }
    vx_context_operation_end(context, &operation);
    return status;
}

void vx_result_retain(VxResult* result) {
    if (result) atomic_fetch_add_explicit(&result->references, 1, memory_order_relaxed);
}

void vx_result_release(VxResult* result) {
    if (!result || atomic_fetch_sub_explicit(&result->references, 1,
                                             memory_order_acq_rel) != 1) return;
    vx_result_destroy(result);
}

uint64_t vx_result_execution_id(const VxResult* result) {
    return result ? result->execution_id : 0;
}

size_t vx_result_output_count(const VxResult* result) {
    return result ? result->output_count : 0;
}

VxStatus vx_result_output_info(const VxResult* result,
                               size_t index,
                               VxTensorInfo* info,
                               VxReport* report) {
    const VxOwnedOutput* output;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!result || !info || info->struct_size != sizeof(*info))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
                       NULL, "INVALID_OUTPUT_INFO", "output descriptor request is invalid");
    if (index >= result->output_count) {
        VxStatus status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_READBACK,
                                  result->context->compiled->backend,
                                  "OUTPUT_NOT_FOUND",
                                  "output index is out of range");
        vx_report_set_result_evidence(report, result);
        return status;
    }
    output = &result->outputs[index];
    info->name = output->name;
    info->dtype = output->dtype;
    info->rank = output->rank;
    memcpy(info->shape, output->shape, sizeof(info->shape));
    info->byte_size = output->byte_size;
    info->location = VX_MEMORY_HOST;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_READBACK,
                    result->context->compiled->backend, "host", "OK",
                    "output descriptor returned", result->execution_id);
    vx_report_set_result_evidence(report, result);
    return VX_STATUS_OK;
}

VxStatus vx_result_read(const VxResult* result,
                        const char* name,
                        void* destination,
                        size_t capacity,
                        size_t* required,
                        VxReport* report) {
    const VxOwnedOutput* output = NULL;
    if (!vx_report_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!result || !name || !name[0] || (!destination && capacity))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_READBACK,
                       NULL, "INVALID_READ", "output read request is invalid");
    for (size_t index = 0; index < result->output_count; index++) {
        if (!strcmp(result->outputs[index].name, name)) {
            output = &result->outputs[index];
            break;
        }
    }
    if (!output) {
        VxStatus status = vx_fail(report, VX_STATUS_NOT_FOUND, VX_STAGE_READBACK,
                                  result->context->compiled->backend,
                                  "OUTPUT_NOT_FOUND",
                                  "named output does not exist");
        vx_report_set_result_evidence(report, result);
        return status;
    }
    if (required) *required = output->byte_size;
    if (!destination && capacity == 0) {
        vx_report_write(report, VX_STATUS_OK, VX_STAGE_READBACK,
                        result->context->compiled->backend, "host", "OK",
                        "output byte count returned", result->execution_id);
        vx_report_set_result_evidence(report, result);
        return VX_STATUS_OK;
    }
    if (capacity < output->byte_size) {
        VxStatus status = vx_fail(report, VX_STATUS_BUFFER_TOO_SMALL,
                                  VX_STAGE_READBACK,
                                  result->context->compiled->backend,
                                  "BUFFER_TOO_SMALL",
                                  "destination cannot hold the output snapshot");
        vx_report_set_result_evidence(report, result);
        return status;
    }
    if (output->byte_size) memcpy(destination, output->data, output->byte_size);
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_READBACK,
                    result->context->compiled->backend, "host", "OK",
                    "output snapshot copied", result->execution_id);
    vx_report_set_result_evidence(report, result);
    return VX_STATUS_OK;
}
