#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "volvoxai.h"
#include "volvoxai_backend.h"

#include "backend_sdk.h"
#include "engine_internal.h"
#include "engine_core.h"
#include "json_validation.h"
#include "public_api_internal.h"
#include "runtime_state.h"
#include "safetensors.h"
#include "thread_pool.h"

#include <inttypes.h>
#include <limits.h>
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

typedef struct VxDeclaredOutput {
    char* name;
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
} VxDeclaredOutput;

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
    VxDeclaredOutput* outputs;
    size_t output_count;
    pthread_mutex_t revision_mutex;
    _Atomic(VxWeightRevisionRecord*) current_weights;
    _Atomic(VxAdapterRevisionRecord*) current_adapter;
    VxAdapterRevisionRecord* adapters;
    uint64_t identity;
    uint64_t graph_identity;
    uint64_t graph_revision;
    uint64_t weight_identity;
    uint64_t allocated_bytes;
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
};

typedef struct VxContextOperation {
    uint64_t ticket;
    int accepted;
} VxContextOperation;

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
typedef void (*VxPublicApiCpuExecuteHook)(void* user_data);
static VxPublicApiCpuExecuteHook g_cpu_execute_hook;
static void* g_cpu_execute_hook_user_data;

void vx_public_api_test_set_cpu_execute_hook(VxPublicApiCpuExecuteHook hook,
                                              void* user_data) {
    g_cpu_execute_hook = hook;
    g_cpu_execute_hook_user_data = user_data;
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
    for (size_t index = 0; index < count; index++) free(outputs[index].name);
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

static char* vx_string_copy(const char* value) {
    size_t length;
    char* copy;
    if (!value) return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1u);
    if (copy) memcpy(copy, value, length + 1u);
    return copy;
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
    if (!report || report->struct_size < sizeof(*report)) return;
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

static void vx_report_set_lineage(VxReport* report,
                                  const VxRuntime* runtime,
                                  const VxModel* model,
                                  const VxCompiledModel* compiled,
                                  const VxExecutionContext* context) {
    const VxWeightRevisionRecord* weights = NULL;
    const VxAdapterRevisionRecord* adapter = NULL;
    uint64_t allocated_bytes = 0;
    if (!report || report->struct_size < sizeof(*report)) return;
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
    if (!report || report->struct_size < sizeof(*report)) return;
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

static VxStatus vx_builtin_compile_route_status(
    VxReport* report,
    VxOperatorFallback operator_fallback) {
    if (!report || report->struct_size < sizeof(*report))
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
    if (!report || report->struct_size < sizeof(*report) || !state) return;
    index = state->last_failure_node_index;
    if (index < 0 || index >= state->node_count) return;
    snprintf(report->offending_node, sizeof(report->offending_node),
             "index=%d;op=%s;output=%s", index, state->nodes[index].op,
             state->nodes[index].out);
}

static void vx_report_publish(VxReport* destination,
                              const VxReport* source) {
    size_t struct_size;
    if (!destination || destination->struct_size < sizeof(*destination) ||
        !source || source->struct_size < sizeof(*source)) return;
    struct_size = destination->struct_size;
    *destination = *source;
    destination->struct_size = struct_size;
}

static void vx_report_set_plan_evidence(VxReport* report,
                                        const VxCompiledModel* compiled) {
    const VxReport* plan;
    if (!report || report->struct_size < sizeof(*report) || !compiled) return;
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
    if (!report || report->struct_size < sizeof(*report) || !result) return;
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
    if (!report || report->struct_size < sizeof(*report) || !result) return;
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
    if (!result || !report || report->struct_size < sizeof(*report)) return;
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
    source->struct_size = sizeof(*source);
    source->graph_path = model->graph_path;
    source->weight_paths = weights
        ? (const char* const*)weights->paths : NULL;
    source->weight_path_count = weights ? weights->path_count : 0;
}

static int vx_policy_valid(const VxBackendPolicy* policy) {
    if (!policy || policy->struct_size < sizeof(*policy) ||
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
    int configured = 0;
    int status;
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
        status = volvoxai_engine_init_with_weight_files(
            model->graph_path,
            weights ? (const char* const*)weights->paths : NULL,
            weights ? (int)weights->path_count : 0);
    }
    vx_engine_state_scope_leave(scope);
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
    if (!context || !context->engine_state_initialized) return;
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
    if (!out_runtime || (options && options->struct_size < sizeof(*options)) ||
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
    VxDeclaredOutput* declared_outputs = NULL;
    size_t output_count = 0;
    if (!runtime || !source || source->struct_size < sizeof(*source) ||
        !source->graph_path || !source->graph_path[0] || !out_model ||
        source->weight_path_count > (size_t)MAX_WEIGHT_FILES ||
        (source->weight_path_count && !source->weight_paths))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_MODEL_SOURCE", "model source is invalid");
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
    envelope_status = vx_graph_package_validate_envelope(
        graph_snapshot, NULL, 0, 0, &declared_outputs, &output_count);
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
    envelope_status = vx_graph_package_validate_envelope(
        model->graph_path, (const char* const*)weights->paths,
        weights->path_count, 1, &declared_outputs, &output_count);
    if (envelope_status != VX_STATUS_OK) {
        vx_weight_revision_release(weights);
        weights = NULL;
        vx_declared_outputs_free(declared_outputs, output_count);
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
        return vx_fail(report, VX_STATUS_INVALID_GRAPH, VX_STAGE_MODEL_LOAD,
                       NULL, "INVALID_GRAPH_CONTRACT",
                       "every graph output must have one canonical execution descriptor");
    }
    model->outputs = declared_outputs;
    model->output_count = output_count;
    declared_outputs = NULL;
    output_count = 0;
    base_adapter = vx_adapter_revision_create(
        "__base__", NULL, NULL, vx_next_object_identity(runtime), 1,
        &adapter_status);
    if (!base_adapter) goto oom;
    model->adapters = base_adapter;
    atomic_init(&model->current_weights, weights);
    atomic_init(&model->current_adapter, base_adapter);
    model->allocated_bytes = sizeof(*model) + strlen(model->graph_path) + 1u;
    model->allocated_bytes += model->output_count * sizeof(*model->outputs);
    for (size_t index = 0; index < model->output_count; index++)
        model->allocated_bytes += strlen(model->outputs[index].name) + 1u;
    vx_runtime_retain(runtime);
    *out_model = model;
    vx_report_write(report, VX_STATUS_OK, VX_STAGE_MODEL_LOAD, NULL, NULL,
                    "OK", "model source retained", 0);
    vx_report_set_lineage(report, NULL, model, NULL, NULL);
    return VX_STATUS_OK;
oom:
    vx_weight_revision_release(weights);
    vx_adapter_revision_release(base_adapter);
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_OUT_OF_MEMORY, VX_STAGE_MODEL_LOAD, NULL,
                   "OUT_OF_MEMORY", "model source copy failed");
weights_unreadable:
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_IO_ERROR, VX_STAGE_MODEL_LOAD, NULL,
                   "WEIGHTS_NOT_READABLE",
                   "a weight revision file could not be read");
weights_invalid:
    vx_declared_outputs_free(declared_outputs, output_count);
    vx_declared_outputs_free(model->outputs, model->output_count);
    vx_snapshot_path_release(model->graph_path);
    pthread_mutex_destroy(&model->revision_mutex);
    free(model);
    return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_MODEL_LOAD,
                   NULL, "INVALID_MODEL_SOURCE",
                   "a weight source path is empty");
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
    vx_declared_outputs_free(model->outputs, model->output_count);
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
    if (!model || !info || info->struct_size < sizeof(*info))
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
    if (!model || !source || source->struct_size < sizeof(*source) ||
        !source->adapter_name || !source->adapter_name[0] ||
        strlen(source->adapter_name) >= 128u ||
        (source->package_path && !source->package_path[0]) ||
        (source->version_name &&
         (!source->version_name[0] || strlen(source->version_name) >= 96u)) ||
        (published && published->struct_size < sizeof(*published)))
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
        VxReport provider_report = VX_REPORT_INIT;
        compiled->provider = provider;
        compiled->provider_runtime = binding.runtime_instance;
        vx_model_source_view(model, compiled->weights, &source);
        provider_report = (VxReport)VX_REPORT_INIT;
        status = provider->compile(compiled->provider_runtime, &source, &resolved,
                                   &compiled->provider_compiled, &provider_report);
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
        if (status == VX_STATUS_OK &&
            compiled->builtin_backend != VOLVOXAI_BACKEND_CPU &&
            validation->node_count > 0) {
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
        if (status == VX_STATUS_OK || status == VX_STATUS_BACKEND_UNSUPPORTED)
            vx_report_set_builtin_route(&compile_report, validation,
                                        selected_backend);
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
    if (!compiled || !report || report->struct_size < sizeof(*report))
        return VX_STATUS_INVALID_ARGUMENT;
    *report = compiled->report;
    return VX_STATUS_OK;
}

static VxStatus vx_context_operation_begin(VxExecutionContext* context,
                                           VxContextOperation* operation,
                                           VxStage stage,
                                           VxReport* report) {
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
    return options && options->struct_size >= sizeof(*options) &&
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
    if (!report || report->struct_size < sizeof(*report) || !context ||
        !context->decode_session) return;
    mode = volvoxai_engine_decode_session_mode(context->decode_session);
    last = volvoxai_engine_decode_session_last_execution_mode(
        context->decode_session);
    seeded = volvoxai_engine_decode_session_seeded(context->decode_session);
    snprintf(report->decode_state, sizeof(report->decode_state),
             "enabled=1;seeded=%d;mode=%s;last=%s", seeded,
             vx_decode_mode_name(mode), vx_decode_mode_name(last));
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

VxStatus vx_compiled_model_create_context(
    VxCompiledModel* compiled,
    const VxContextOptions* options,
    VxExecutionContext** out_context,
    VxReport* report) {
    VxContextOptions resolved = VX_CONTEXT_OPTIONS_INIT;
    VxExecutionContext* context;
    VxStatus status;
    char prepared_route[96];
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
    if (!context->compiled->provider) {
        VxEngineStateScope scope;
        scope = vx_engine_state_scope_enter(context->engine_state);
        count = (size_t)volvoxai_engine_graph_input_count();
        vx_engine_state_scope_leave(scope);
    }
    vx_context_operation_end(context, &operation);
    return count;
}

VxStatus vx_execution_context_input_info(VxExecutionContext* context,
                                         size_t index,
                                         VxTensorInfo* info,
                                         VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!info || info->struct_size < sizeof(*info))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_INPUT,
                       context ? context->compiled->backend : NULL,
                       "INVALID_TENSOR_INFO", "tensor info buffer is invalid");
    status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    if (context->compiled->provider) {
        status = vx_fail(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_INPUT,
                         context->compiled->backend, "INPUT_INTROSPECTION_UNSUPPORTED",
                         "provider does not expose input introspection");
    } else {
        VxEngineStateScope scope;
        const char* name;
        long numel;
        int shape[8] = {0};
        int rank;
        int dtype;
        size_t element_size;
        scope = vx_engine_state_scope_enter(context->engine_state);
        volvoxai_engine_tensor_name_index_invalidate();
        volvoxai_engine_tensor_name_index_rebuild();
        name = index < (size_t)volvoxai_engine_graph_input_count()
            ? volvoxai_engine_graph_input_name((int)index) : NULL;
        if (!name || volvoxai_engine_tensor_info_ex(name, &numel, shape, &rank,
                                                     &dtype, &element_size) != 0) {
            status = VX_STATUS_NOT_FOUND;
        } else {
            info->name = name;
            info->dtype = (VxDataType)dtype;
            info->rank = (uint32_t)rank;
            memset(info->shape, 0, sizeof(info->shape));
            for (int axis = 0; axis < rank; axis++) info->shape[axis] = shape[axis];
            info->byte_size = numel > 0 && (size_t)numel <= SIZE_MAX / element_size
                ? (size_t)numel * element_size : 0;
            info->location = VX_MEMORY_HOST;
            status = VX_STATUS_OK;
        }
        vx_engine_state_scope_leave(scope);
        if (status == VX_STATUS_OK)
            vx_report_write(report, status, VX_STAGE_INPUT,
                            context->compiled->backend,
                            vx_builtin_device_identity(context->compiled->backend),
                            "OK",
                            "input descriptor returned", 0);
        else
            vx_fail(report, status, VX_STAGE_INPUT,
                    context->compiled->backend, "INPUT_NOT_FOUND",
                    "input index is out of range");
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
    if (!name || !name[0] || !quantization ||
        quantization->struct_size < sizeof(*quantization))
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

VxStatus vx_execution_context_set_input(VxExecutionContext* context,
                                        const char* name,
                                        VxDataType dtype,
                                        const void* data,
                                        size_t byte_size,
                                        VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
    if (!name || !name[0] || (!data && byte_size) ||
        !vx_execution_dtype_size(dtype))
        return vx_fail(report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_INPUT,
                       context ? context->compiled->backend : NULL,
                       "INVALID_INPUT", "input name or storage is invalid");
    status = vx_context_operation_begin(context, &operation, VX_STAGE_INPUT, report);
    if (status != VX_STATUS_OK) return status;
    if (context->compiled->provider) {
        status = context->compiled->provider->context_set_input(
            context->provider_context, name, dtype, data, byte_size, report);
    } else {
        VxEngineStateScope scope;
        int result;
        scope = vx_engine_state_scope_enter(context->engine_state);
        volvoxai_engine_tensor_name_index_invalidate();
        volvoxai_engine_tensor_name_index_rebuild();
        result = volvoxai_engine_set_input_raw(name, (int)dtype, data, byte_size);
        vx_engine_state_scope_leave(scope);
        status = result == 0 ? VX_STATUS_OK : VX_STATUS_INVALID_ARGUMENT;
    }
    if (status == VX_STATUS_OK)
        vx_report_write(report, status, VX_STAGE_INPUT, context->compiled->backend,
                        context->compiled->provider ? NULL :
                            vx_builtin_device_identity(context->compiled->backend),
                        "OK",
                        "input copied", 0);
    else
        vx_fail(report, status, VX_STAGE_INPUT, context->compiled->backend,
                "INPUT_REJECTED", "input was rejected");
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    vx_context_operation_end(context, &operation);
    return status;
}

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

static VxStatus vx_result_append(VxResult* result,
                                 const char* name,
                                 VxDataType dtype,
                                 const int64_t* shape,
                                 uint32_t rank,
                                 const void* data,
                                 size_t byte_size) {
    VxOwnedOutput* output;
    const VxDeclaredOutput* descriptor;
    if (!result || !name || !name[0] ||
        rank > VX_MAX_TENSOR_RANK ||
        (rank && !shape) || (!data && byte_size))
        return VX_STATUS_INVALID_ARGUMENT;
    descriptor = vx_model_output_descriptor(
        result->context->compiled->model, name);
    if (!descriptor || !vx_execution_dtype_size(dtype) ||
        dtype != descriptor->dtype || rank != descriptor->rank ||
        byte_size != descriptor->byte_size)
        return VX_STATUS_INVALID_ARGUMENT;
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (shape[axis] != descriptor->shape[axis])
            return VX_STATUS_INVALID_ARGUMENT;
    }
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

typedef enum VxContextRunKind {
    VX_CONTEXT_RUN_FORWARD = 0,
    VX_CONTEXT_RUN_DECODE_SEED = 1,
    VX_CONTEXT_RUN_DECODE_STEP = 2,
    VX_CONTEXT_RUN_PREFIX = 3
} VxContextRunKind;

static VxStatus vx_execution_context_run(VxExecutionContext* context,
                                         VxContextRunKind kind,
                                         int32_t position,
                                         VxResult** out_result,
                                         VxReport* report) {
    VxContextOperation operation;
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
    execution_started_ms = vx_report_now_ms();
    result = (VxResult*)calloc(1, sizeof(*result));
    if (!result) {
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
                context->provider_context, &sink, &execution_report);
        } else if (kind == VX_CONTEXT_RUN_DECODE_SEED &&
                   context->compiled->provider->context_decode_seed) {
            status = context->compiled->provider->context_decode_seed(
                context->provider_context, &sink, &execution_report);
        } else if (kind == VX_CONTEXT_RUN_DECODE_STEP &&
                   context->compiled->provider->context_decode_step) {
            status = context->compiled->provider->context_decode_step(
                context->provider_context, position, &sink, &execution_report);
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
        int forward_status = 0;
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
                                       VX_STATUS_EXECUTION_FAILED;
        if (adapter_route_active) volvoxai_engine_adapter_route_end();
        if (decode_operation)
            vx_report_set_builtin_decode_state(&execution_report, context);
        vx_engine_state_scope_leave(scope);
        vx_report_set_builtin_route(&execution_report, context->engine_state,
                                    context->compiled->backend);
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
        if (status == VX_STATUS_BACKEND_UNSUPPORTED) {
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
        vx_result_destroy(result);
        vx_context_operation_end(context, &operation);
        return status;
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
    vx_context_operation_end(context, &operation);
    return VX_STATUS_OK;
}

VxStatus vx_execution_context_execute(VxExecutionContext* context,
                                      VxResult** out_result,
                                      VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_FORWARD, -1,
                                    out_result, report);
}

VxStatus vx_execution_context_execute_prefix(VxExecutionContext* context,
                                             int32_t row_count,
                                             VxResult** out_result,
                                             VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_PREFIX, row_count,
                                    out_result, report);
}

VxStatus vx_execution_context_decode_seed(VxExecutionContext* context,
                                          VxResult** out_result,
                                          VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_DECODE_SEED, -1,
                                    out_result, report);
}

VxStatus vx_execution_context_decode_step(VxExecutionContext* context,
                                          int32_t position,
                                          VxResult** out_result,
                                          VxReport* report) {
    return vx_execution_context_run(context, VX_CONTEXT_RUN_DECODE_STEP,
                                    position, out_result, report);
}

VxStatus vx_execution_context_decode_reset(VxExecutionContext* context,
                                           VxReport* report) {
    VxContextOperation operation;
    VxStatus status;
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
            if (report && report->struct_size >= sizeof(*report))
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
    vx_report_set_lineage(report, NULL, NULL, NULL, context);
    vx_report_set_plan_evidence(report, context->compiled);
    if (status == VX_STATUS_OK &&
        report && report->struct_size >= sizeof(*report)) {
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
    if (!revision || revision->struct_size < sizeof(*revision) ||
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
    VxStatus status = vx_context_operation_begin(
        context, &operation, VX_STAGE_ADAPTER, report);
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
    if (!result || !info || info->struct_size < sizeof(*info))
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
