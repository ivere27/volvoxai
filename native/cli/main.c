#include "volvoxai.h"
#include "cJSON.h"
#if VOLVOXAI_ENABLE_TRAINING
#include "volvoxai_full.h"
#endif

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#ifndef VOLVOXAI_VERSION
#define VOLVOXAI_VERSION "unknown"
#endif
#ifndef VOLVOXAI_GIT_COMMIT
#define VOLVOXAI_GIT_COMMIT "unknown"
#endif
#ifndef VOLVOXAI_BUILD_DATE
#define VOLVOXAI_BUILD_DATE "unknown"
#endif

#define MAX_BINDINGS 32
#define MAX_WEIGHT_PATHS 32

typedef struct Binding {
    char name[128];
    char path[PATH_MAX];
    int64_t shape[VX_MAX_TENSOR_RANK];
    uint32_t rank;
    int has_shape;
} Binding;

typedef struct ModelPaths {
    char graph[PATH_MAX];
    char default_weights[PATH_MAX];
} ModelPaths;

typedef struct RunOptions {
    const char* weight_paths[MAX_WEIGHT_PATHS];
    size_t weight_path_count;
    Binding inputs[MAX_BINDINGS];
    Binding outputs[MAX_BINDINGS];
    size_t input_count;
    size_t output_count;
    const char* backend;
    const char* report_json;
    int debug;
    int cpu_threads;
    int output_row;
} RunOptions;

#if VOLVOXAI_ENABLE_TRAINING
typedef struct TrainOptions {
    RunOptions run;
    const char* targets_path;
    const char* logits_name;
    const char* trainable_names[MAX_BINDINGS];
    size_t trainable_count;
    const char* output_weight_paths[MAX_WEIGHT_PATHS];
    size_t output_weight_path_count;
    long microbatches;
    uint32_t accumulation_steps;
    VxOptimizerOptions optimizer;
    int32_t ignore_index;
    int32_t loss_row;
} TrainOptions;
#endif

static int file_exists(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int is_directory(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int copy_path(char* destination, size_t capacity, const char* value) {
    int written;
    if (!destination || !capacity || !value) return -1;
    written = snprintf(destination, capacity, "%s", value);
    return written >= 0 && (size_t)written < capacity ? 0 : -1;
}

static int join_path(char* destination, size_t capacity,
                     const char* directory, const char* leaf) {
    size_t length;
    int written;
    if (!destination || !capacity || !directory || !leaf) return -1;
    length = strlen(directory);
    written = snprintf(destination, capacity, "%s%s%s", directory,
                       length && directory[length - 1] == '/' ? "" : "/", leaf);
    return written >= 0 && (size_t)written < capacity ? 0 : -1;
}

static int resolve_model_paths(const char* model, ModelPaths* paths) {
    if (!model || !model[0] || !paths) return -1;
    memset(paths, 0, sizeof(*paths));
    if (is_directory(model)) {
        if (join_path(paths->graph, sizeof(paths->graph), model, "graph.json") != 0 ||
            join_path(paths->default_weights, sizeof(paths->default_weights), model,
                      "model.safetensors") != 0) {
            fprintf(stderr, "Model path is too long: %s\n", model);
            return -1;
        }
        if (!file_exists(paths->default_weights)) paths->default_weights[0] = 0;
    } else if (copy_path(paths->graph, sizeof(paths->graph), model) != 0) {
        fprintf(stderr, "Graph path is too long: %s\n", model);
        return -1;
    }
    if (!file_exists(paths->graph)) {
        fprintf(stderr, "Missing graph.json: %s\n", paths->graph);
        return -1;
    }
    return 0;
}

static int parse_binding(const char* argument, Binding* binding,
                         int require_name) {
    const char* equals;
    const char* shape_open = NULL;
    size_t name_length;
    if (!argument || !binding) return -1;
    memset(binding, 0, sizeof(*binding));
    equals = strchr(argument, '=');
    if (!equals) {
        if (require_name || copy_path(binding->path, sizeof(binding->path), argument) != 0)
            return -1;
        return binding->path[0] ? 0 : -1;
    }
    name_length = (size_t)(equals - argument);
    if (name_length && argument[name_length - 1u] == ']') {
        const char* cursor;
        const char* shape_end = equals - 1;
        for (cursor = shape_end; cursor > argument; cursor--)
            if (cursor[-1] == '[') {
                shape_open = cursor - 1;
                break;
            }
        if (!shape_open || shape_open == argument) return -1;
        cursor = shape_open + 1;
        while (cursor < shape_end) {
            char* parsed_end = NULL;
            long long extent;
            if (binding->rank == VX_MAX_TENSOR_RANK) return -1;
            errno = 0;
            extent = strtoll(cursor, &parsed_end, 10);
            if (errno == ERANGE || !parsed_end || parsed_end == cursor ||
                extent <= 0 || extent > INT64_MAX || parsed_end > shape_end ||
                (parsed_end < shape_end && *parsed_end != ',')) return -1;
            binding->shape[binding->rank++] = (int64_t)extent;
            cursor = parsed_end < shape_end ? parsed_end + 1 : parsed_end;
        }
        if (!binding->rank) return -1;
        binding->has_shape = 1;
        name_length = (size_t)(shape_open - argument);
    }
    if (!name_length || name_length >= sizeof(binding->name) || !equals[1]) return -1;
    memcpy(binding->name, argument, name_length);
    binding->name[name_length] = 0;
    return copy_path(binding->path, sizeof(binding->path), equals + 1);
}

static int parse_nonnegative_int(const char* value, int allow_minus_one,
                                 int* parsed) {
    char* end = NULL;
    long number;
    if (!value || !value[0] || !parsed) return -1;
    errno = 0;
    number = strtol(value, &end, 10);
    if (errno == ERANGE || !end || *end || number > INT_MAX ||
        number < (allow_minus_one ? -1 : 0)) return -1;
    *parsed = (int)number;
    return 0;
}

static int select_backend(RunOptions* options, const char* backend) {
    if (options->backend && strcmp(options->backend, backend)) {
        fprintf(stderr, "Pass at most one accelerator backend flag.\n");
        return -1;
    }
    options->backend = backend;
    return 0;
}

static int parse_run_option(int argc, char** argv, int* index,
                            RunOptions* options) {
    const char* argument = argv[*index];
    const char* backend = NULL;
    if (!strcmp(argument, "--cpu")) backend = "cpu";
    else if (!strcmp(argument, "--vulkan")) backend = "vulkan";
    else if (!strcmp(argument, "--opengl")) backend = "opengl";
    else if (!strcmp(argument, "--metal")) backend = "metal";
    else if (!strcmp(argument, "--nnapi")) backend = "nnapi";
    else if (!strcmp(argument, "--cuda")) backend = "cuda";
    if (backend) return select_backend(options, backend) == 0 ? 1 : -1;
    if (!strcmp(argument, "--debug")) {
        options->debug = 1;
        return 1;
    }
    if (!strcmp(argument, "--report-json")) {
        if (*index + 1 >= argc || options->report_json) {
            fprintf(stderr, "--report-json expects exactly one output file.\n");
            return -1;
        }
        options->report_json = argv[++(*index)];
        return options->report_json[0] ? 1 : -1;
    }
    if (!strcmp(argument, "--threads")) {
        if (*index + 1 >= argc ||
            parse_nonnegative_int(argv[++(*index)], 0, &options->cpu_threads) != 0 ||
            options->cpu_threads == 0) {
            fprintf(stderr, "--threads must be a positive 32-bit integer.\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argument, "--row")) {
        if (*index + 1 >= argc ||
            parse_nonnegative_int(argv[++(*index)], 1, &options->output_row) != 0) {
            fprintf(stderr, "--row must be -1 or a non-negative 32-bit integer.\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argument, "--weights")) {
        if (*index + 1 >= argc ||
            options->weight_path_count == MAX_WEIGHT_PATHS) {
            fprintf(stderr, "--weights expects a file and may be repeated at most %d times.\n",
                    MAX_WEIGHT_PATHS);
            return -1;
        }
        options->weight_paths[options->weight_path_count++] = argv[++(*index)];
        return 1;
    }
    if (!strcmp(argument, "--input")) {
        if (*index + 1 >= argc || options->input_count == MAX_BINDINGS ||
            parse_binding(argv[++(*index)], &options->inputs[options->input_count], 1) != 0) {
            fprintf(stderr,
                    "--input expects name=file for a fixed input or "
                    "name[d0,d1,...]=file for a dynamic input.\n");
            return -1;
        }
        options->input_count++;
        return 1;
    }
    if (!strcmp(argument, "--output")) {
        if (*index + 1 >= argc || options->output_count == MAX_BINDINGS ||
            parse_binding(argv[++(*index)], &options->outputs[options->output_count], 0) != 0) {
            fprintf(stderr, "--output expects name=file or file.\n");
            return -1;
        }
        options->output_count++;
        return 1;
    }
    return 0;
}

static const char* dtype_name(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_F32: return "F32";
        case VX_DTYPE_I8: return "I8";
        case VX_DTYPE_U8: return "U8";
        case VX_DTYPE_I32: return "I32";
        default: return "unknown";
    }
}

static const char* dtype_suffix(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_F32: return ".f32";
        case VX_DTYPE_I8: return ".i8";
        case VX_DTYPE_U8: return ".u8";
        case VX_DTYPE_I32: return ".i32";
        default: return NULL;
    }
}

static int has_suffix(const char* path, const char* suffix) {
    size_t path_length;
    size_t suffix_length;
    if (!path || !suffix) return 0;
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    return path_length >= suffix_length &&
           !strcmp(path + path_length - suffix_length, suffix);
}

static void print_failure(const char* operation, VxStatus status,
                          const VxReport* report) {
    fprintf(stderr, "%s failed: %s", operation, vx_status_string(status));
    if (report && report->reason[0]) fprintf(stderr, " [%s]", report->reason);
    if (report && report->message[0]) fprintf(stderr, ": %s", report->message);
    fputc('\n', stderr);
}

static int read_exact_file(const char* path, size_t expected, void** data) {
    FILE* file;
    long size;
    void* bytes;
    if (!path || !data) return -1;
    *data = NULL;
    file = fopen(path, "rb");
    if (!file) {
        fprintf(stderr, "Cannot read input file: %s\n", path);
        return -1;
    }
    if (fseek(file, 0, SEEK_END) != 0 || (size = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fprintf(stderr, "Cannot inspect input file: %s\n", path);
        fclose(file);
        return -1;
    }
    if ((unsigned long)size > SIZE_MAX || (size_t)size != expected) {
        fprintf(stderr, "Input expects %zu raw bytes, got %ld from %s\n",
                expected, size, path);
        fclose(file);
        return -1;
    }
    bytes = malloc(expected ? expected : 1);
    if (!bytes) {
        fprintf(stderr, "Cannot allocate %zu input bytes.\n", expected);
        fclose(file);
        return -1;
    }
    if (expected && fread(bytes, 1, expected, file) != expected) {
        fprintf(stderr, "Cannot read input file: %s\n", path);
        free(bytes);
        fclose(file);
        return -1;
    }
    if (fclose(file) != 0) {
        fprintf(stderr, "Cannot finish reading input file: %s\n", path);
        free(bytes);
        return -1;
    }
    *data = bytes;
    return 0;
}

static int find_input(VxExecutionContext* context, const char* name,
                      VxTensorSpec* found, VxReport* report) {
    size_t count = vx_execution_context_input_count(context);
    for (size_t index = 0; index < count; index++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        if (vx_execution_context_input_spec(context, index, &spec, report) ==
                VX_STATUS_OK &&
            spec.name && !strcmp(spec.name, name)) {
            *found = spec;
            return 0;
        }
    }
    fprintf(stderr, "Unknown input tensor: %s\n", name);
    return -1;
}

static size_t dtype_byte_size(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_I8:
        case VX_DTYPE_U8: return 1u;
        case VX_DTYPE_F32:
        case VX_DTYPE_I32: return 4u;
        default: return 0u;
    }
}

static int prepare_input_binding(VxExecutionContext* context,
                                 const Binding* binding,
                                 VxTensorBinding* prepared,
                                 void** storage,
                                 VxReport* report) {
    VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
    const char* suffix;
    void* bytes = NULL;
    size_t byte_size;
    size_t element_size;
    if (!prepared || !storage ||
        find_input(context, binding->name, &spec, report) != 0) return -1;
    if (binding->has_shape && binding->rank != spec.rank) {
        fprintf(stderr, "Input %s expects rank %u, got rank %u.\n",
                binding->name, spec.rank, binding->rank);
        return -1;
    }
    *prepared = (VxTensorBinding)VX_TENSOR_BINDING_INIT;
    prepared->name = binding->name;
    prepared->dtype = spec.dtype;
    prepared->rank = spec.rank;
    byte_size = element_size = dtype_byte_size(spec.dtype);
    if (!element_size) return -1;
    for (uint32_t axis = 0; axis < spec.rank; axis++) {
        int64_t extent;
        if (binding->has_shape) {
            extent = binding->shape[axis];
        } else {
            if (spec.dimensions[axis].kind != VX_DIMENSION_FIXED) {
                fprintf(stderr,
                        "Dynamic input %s requires an explicit shape: "
                        "--input %s[d0,d1,...]=%s\n",
                        binding->name, binding->name, binding->path);
                return -1;
            }
            extent = spec.dimensions[axis].min;
        }
        if (extent <= 0 || (uint64_t)extent > SIZE_MAX / byte_size) {
            fprintf(stderr, "Input %s shape is too large.\n", binding->name);
            return -1;
        }
        prepared->shape[axis] = extent;
        byte_size *= (size_t)extent;
    }
    suffix = dtype_suffix(spec.dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Input %s has dtype %s and requires a %s file: %s\n",
                binding->name, dtype_name(spec.dtype), suffix ? suffix : "supported raw",
                binding->path);
        return -1;
    }
    if (read_exact_file(binding->path, byte_size, &bytes) != 0) {
        if (file_exists(binding->path)) {
            fprintf(stderr, "Input %s expects %zu raw bytes.\n",
                    binding->name, byte_size);
        }
        return -1;
    }
    prepared->data = bytes;
    prepared->byte_size = byte_size;
    prepared->location = VX_MEMORY_HOST;
    *storage = bytes;
    return 0;
}

static int find_output(const VxResult* result, const char* name,
                       VxTensorInfo* found, VxReport* report) {
    size_t count = vx_result_output_count(result);
    for (size_t index = 0; index < count; index++) {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        if (vx_result_output_info(result, index, &info, report) == VX_STATUS_OK &&
            ((!name && index == 0) || (name && info.name && !strcmp(info.name, name)))) {
            *found = info;
            return 0;
        }
    }
    fprintf(stderr, "Unknown output tensor: %s\n", name ? name : "<first>");
    return -1;
}

static int write_bytes(const char* path, const void* bytes, size_t size) {
    FILE* file = fopen(path, "wb");
    int failed = 0;
    if (!file) {
        fprintf(stderr, "Cannot write output file: %s\n", path);
        return -1;
    }
    if (size && fwrite(bytes, 1, size, file) != size) failed = 1;
    if (fclose(file) != 0) failed = 1;
    if (failed) {
        fprintf(stderr, "Cannot write output file: %s\n", path);
        return -1;
    }
    return 0;
}

static int write_output_binding(const VxResult* result, const Binding* binding,
                                int row, VxReport* report) {
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    const char* suffix;
    void* bytes;
    VxStatus status;
    size_t offset = 0;
    size_t byte_size;
    if (find_output(result, binding->name[0] ? binding->name : NULL,
                    &info, report) != 0) return -1;
    suffix = dtype_suffix(info.dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Output %s has dtype %s and requires a %s file: %s\n",
                info.name, dtype_name(info.dtype), suffix ? suffix : "supported raw",
                binding->path);
        return -1;
    }
    byte_size = info.byte_size;
    if (row >= 0) {
        if (info.dtype != VX_DTYPE_F32 || info.rank < 2 || info.shape[0] <= 0 ||
            (uint64_t)row >= (uint64_t)info.shape[0] ||
            info.byte_size % (size_t)info.shape[0] != 0) {
            fprintf(stderr, "--row %d is invalid for output %s.\n", row, info.name);
            return -1;
        }
        byte_size = info.byte_size / (size_t)info.shape[0];
        offset = (size_t)row * byte_size;
    }
    bytes = malloc(info.byte_size ? info.byte_size : 1);
    if (!bytes) {
        fprintf(stderr, "Cannot allocate %zu output bytes.\n", info.byte_size);
        return -1;
    }
    status = vx_result_read(result, info.name, bytes, info.byte_size, NULL, report);
    if (status != VX_STATUS_OK) {
        print_failure("Reading output", status, report);
        free(bytes);
        return -1;
    }
    status = write_bytes(binding->path, (const unsigned char*)bytes + offset, byte_size);
    free(bytes);
    return status;
}

typedef struct StableReadEvidence {
    VxTensorInfo info;
    void* first_read;
} StableReadEvidence;

static const char* runtime_dtype_name(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_F32: return "float32";
        case VX_DTYPE_I32: return "int32";
        case VX_DTYPE_I8: return "int8";
        case VX_DTYPE_U8: return "uint8";
        default: return NULL;
    }
}

static const char* runtime_stage_name(VxStage stage) {
    switch (stage) {
        case VX_STAGE_NONE: return "none";
        case VX_STAGE_RUNTIME_CREATE: return "runtime-create";
        case VX_STAGE_MODEL_LOAD: return "model-load";
        case VX_STAGE_COMPILE: return "compile";
        case VX_STAGE_CONTEXT_CREATE: return "context-create";
        case VX_STAGE_INPUT: return "input";
        case VX_STAGE_EXECUTE: return "execute";
        case VX_STAGE_READBACK: return "readback";
        case VX_STAGE_CLOSE: return "close";
        case VX_STAGE_DECODE: return "decode";
        case VX_STAGE_ADAPTER: return "adapter";
        default: return "unknown";
    }
}

static int json_add_uint64_string(cJSON* object, const char* key,
                                  uint64_t value) {
    char text[32];
    if (snprintf(text, sizeof(text), "%llu",
                 (unsigned long long)value) < 0)
        return -1;
    return cJSON_AddStringToObject(object, key, text) ? 0 : -1;
}

static int json_add_identity(cJSON* object, const char* key,
                             const char* kind, uint64_t identity) {
    char text[80];
    int written = snprintf(text, sizeof(text), "native-%s-%llu", kind,
                           (unsigned long long)identity);
    if (written < 0 || (size_t)written >= sizeof(text)) return -1;
    return cJSON_AddStringToObject(object, key, text) ? 0 : -1;
}

static int json_add_optional_uint64_string(cJSON* object, const char* key,
                                           uint64_t value) {
    if (!value) return cJSON_AddNullToObject(object, key) ? 0 : -1;
    return json_add_uint64_string(object, key, value);
}

static int json_add_optional_identity(cJSON* object, const char* key,
                                      const char* kind, uint64_t identity) {
    if (!identity) return cJSON_AddNullToObject(object, key) ? 0 : -1;
    return json_add_identity(object, key, kind, identity);
}

static int json_add_adapter_identity(cJSON* object, const char* key,
                                     uint64_t identity, uint64_t revision) {
    char text[96];
    int written = snprintf(text, sizeof(text), "native-adapter-%llu:%llu",
                           (unsigned long long)identity,
                           (unsigned long long)revision);
    if (written < 0 || (size_t)written >= sizeof(text)) return -1;
    return cJSON_AddStringToObject(object, key, text) ? 0 : -1;
}

static cJSON* report_device_json(const VxReport* report) {
    cJSON* device;
    if (!report->device[0]) return cJSON_CreateNull();
    device = cJSON_CreateObject();
    if (!device ||
        !cJSON_AddStringToObject(device, "backend", report->backend) ||
        !cJSON_AddStringToObject(device, "device", report->device)) {
        cJSON_Delete(device);
        return NULL;
    }
    return device;
}

static cJSON* report_route_json(const VxReport* report) {
    cJSON* route = cJSON_CreateObject();
    cJSON* operator = cJSON_CreateObject();
    if (!route || !operator ||
        !cJSON_AddBoolToObject(route, "tierFallback",
                              report->tier_fallback_used != 0) ||
        !cJSON_AddStringToObject(operator, "attestation",
                                 report->route_attested ? "reported" : "unknown"))
        goto fail;
    if (report->route_attested) {
        if (!cJSON_AddBoolToObject(operator, "used",
                                  report->operator_fallback_used != 0))
            goto fail;
    } else if (!cJSON_AddNullToObject(operator, "used")) {
        goto fail;
    }
    if (report->offending_node[0]) {
        if (!cJSON_AddStringToObject(operator, "offendingNode",
                                     report->offending_node))
            goto fail;
    } else if (!cJSON_AddNullToObject(operator, "offendingNode")) {
        goto fail;
    }
    if (!cJSON_AddItemToObject(route, "operator", operator)) goto fail;
    return route;
fail:
    cJSON_Delete(operator);
    cJSON_Delete(route);
    return NULL;
}

static cJSON* report_revisions_json(const VxReport* report, int execution) {
    cJSON* revisions = cJSON_CreateObject();
    cJSON* adapters = cJSON_CreateArray();
    cJSON* adapter = NULL;
    if (!revisions || !adapters ||
        json_add_uint64_string(revisions, "topologyRevision",
                               report->graph_revision) != 0 ||
        json_add_uint64_string(revisions, "weightRevision",
                               report->weight_revision) != 0 ||
        json_add_identity(revisions, "weightRevisionId", "weight",
                          report->weight_id) != 0)
        goto fail;
    if (report->adapter_id && report->adapter_revision) {
        if (json_add_adapter_identity(revisions, "adapterRevisionId",
                                      report->adapter_id,
                                      report->adapter_revision) != 0)
            goto fail;
        adapter = cJSON_CreateObject();
        if (!adapter || json_add_adapter_identity(
                adapter, "value", report->adapter_id,
                report->adapter_revision) != 0)
            goto fail;
        {
            cJSON* value = cJSON_DetachItemFromObject(adapter, "value");
            cJSON_Delete(adapter);
            adapter = value;
        }
        if (!adapter || !cJSON_AddItemToArray(adapters, adapter)) goto fail;
        adapter = NULL;
    } else {
        if (!cJSON_AddNullToObject(revisions, "adapterRevisionId")) goto fail;
        if (execution) {
            adapter = cJSON_CreateNull();
            if (!adapter || !cJSON_AddItemToArray(adapters, adapter)) goto fail;
            adapter = NULL;
        }
    }
    if (!cJSON_AddItemToObject(revisions, "adapterRevisionIds", adapters))
        goto fail;
    return revisions;
fail:
    cJSON_Delete(adapter);
    cJSON_Delete(adapters);
    cJSON_Delete(revisions);
    return NULL;
}

static cJSON* compilation_evidence_json(const VxReport* report) {
    cJSON* compilation = cJSON_CreateObject();
    cJSON* policy = cJSON_CreateObject();
    cJSON* order = NULL;
    cJSON* device = NULL;
    cJSON* revisions = NULL;
    cJSON* route = NULL;
    if (!compilation || !policy ||
        json_add_identity(compilation, "compilationId", "compiled",
                          report->compiled_model_id) != 0)
        goto fail;
    if (report->policy_mode == VX_BACKEND_REQUIRE) {
        if (!cJSON_AddStringToObject(policy, "mode", "require") ||
            !cJSON_AddStringToObject(policy, "backend", report->backend))
            goto fail;
    } else {
        order = cJSON_CreateArray();
        if (!order || !cJSON_AddStringToObject(policy, "mode", "prefer") ||
            !cJSON_AddItemToArray(order, cJSON_CreateString(report->backend)) ||
            !cJSON_AddItemToObject(policy, "order", order))
            goto fail;
        order = NULL;
    }
    if (!cJSON_AddStringToObject(
            policy, "operatorFallback",
            report->operator_fallback == VX_OPERATOR_FALLBACK_FORBID
                ? "forbid" : "allow") ||
        !cJSON_AddItemToObject(compilation, "policy", policy))
        goto fail;
    policy = NULL;
    if (!cJSON_AddStringToObject(compilation, "selectedBackend",
                                 report->backend))
        goto fail;
    device = report_device_json(report);
    if (!device || !cJSON_AddItemToObject(compilation, "selectedDevice", device))
        goto fail;
    device = NULL;
    if (json_add_identity(compilation, "definitionId", "graph",
                          report->graph_id) != 0)
        goto fail;
    revisions = report_revisions_json(report, 0);
    route = report_route_json(report);
    if (!revisions || !route ||
        !cJSON_AddItemToObject(compilation, "revisions", revisions))
        goto fail;
    revisions = NULL;
    if (!cJSON_AddItemToObject(compilation, "route", route)) goto fail;
    route = NULL;
    return compilation;
fail:
    cJSON_Delete(route);
    cJSON_Delete(revisions);
    cJSON_Delete(device);
    cJSON_Delete(order);
    cJSON_Delete(policy);
    cJSON_Delete(compilation);
    return NULL;
}

static cJSON* execution_evidence_json(const VxReport* report) {
    cJSON* execution = cJSON_CreateObject();
    cJSON* device = NULL;
    cJSON* revisions = NULL;
    cJSON* route = NULL;
    cJSON* decode = NULL;
    if (!execution ||
        json_add_identity(execution, "executionId", "execution",
                          report->execution_id) != 0 ||
        json_add_identity(execution, "contextId", "context",
                          report->context_id) != 0 ||
        !cJSON_AddStringToObject(execution, "backend", report->backend))
        goto fail;
    device = report_device_json(report);
    if (!device || !cJSON_AddItemToObject(execution, "device", device))
        goto fail;
    device = NULL;
    if (!cJSON_AddStringToObject(execution, "outcome", "success"))
        goto fail;
    revisions = report_revisions_json(report, 1);
    route = report_route_json(report);
    decode = cJSON_CreateObject();
    if (!revisions || !route || !decode ||
        !cJSON_AddStringToObject(decode, "operation", "execute") ||
        !cJSON_AddNullToObject(decode, "mode") ||
        !cJSON_AddStringToObject(decode, "cacheState", "not-applicable") ||
        !cJSON_AddNullToObject(decode, "cacheGeneration") ||
        !cJSON_AddNullToObject(decode, "position") ||
        !cJSON_AddItemToObject(execution, "revisions", revisions))
        goto fail;
    revisions = NULL;
    if (!cJSON_AddItemToObject(execution, "route", route)) goto fail;
    route = NULL;
    if (!cJSON_AddItemToObject(execution, "decodeState", decode)) goto fail;
    decode = NULL;
    return execution;
fail:
    cJSON_Delete(decode);
    cJSON_Delete(route);
    cJSON_Delete(revisions);
    cJSON_Delete(device);
    cJSON_Delete(execution);
    return NULL;
}

static void stable_read_evidence_free(StableReadEvidence* records,
                                      size_t count) {
    if (!records) return;
    for (size_t index = 0; index < count; index++)
        free(records[index].first_read);
    free(records);
}

static cJSON* verify_stable_result(VxResult* result,
                                   VxExecutionContext* context,
                                   VxReport* report) {
    const uint64_t json_safe_integer = UINT64_C(9007199254740991);
    const size_t count = vx_result_output_count(result);
    StableReadEvidence* records = NULL;
    void* scratch = NULL;
    size_t scratch_size = 0;
    cJSON* stable = NULL;
    cJSON* outputs = NULL;
    VxStatus status;
    if (!count) {
        fprintf(stderr, "Lifecycle evidence requires at least one result output.\n");
        return NULL;
    }
    records = (StableReadEvidence*)calloc(count, sizeof(*records));
    stable = cJSON_CreateObject();
    outputs = cJSON_CreateArray();
    if (!records || !stable || !outputs) goto fail;
    for (size_t index = 0; index < count; index++) {
        cJSON* output = NULL;
        cJSON* shape = NULL;
        const char* dtype;
        records[index].info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
        status = vx_result_output_info(result, index, &records[index].info,
                                       report);
        if (status != VX_STATUS_OK) {
            print_failure("Inspecting stable output", status, report);
            goto fail;
        }
        dtype = runtime_dtype_name(records[index].info.dtype);
        if (!dtype || !records[index].info.name ||
            records[index].info.byte_size > json_safe_integer) {
            fprintf(stderr, "Stable output descriptor is not JSON-safe.\n");
            goto fail;
        }
        output = cJSON_CreateObject();
        shape = cJSON_CreateArray();
        if (!output || !shape ||
            !cJSON_AddStringToObject(output, "name", records[index].info.name)) {
            cJSON_Delete(shape);
            cJSON_Delete(output);
            goto fail;
        }
        for (uint32_t axis = 0; axis < records[index].info.rank; axis++) {
            int64_t dimension = records[index].info.shape[axis];
            cJSON* value;
            if (dimension <= 0 || (uint64_t)dimension > json_safe_integer) {
                cJSON_Delete(shape);
                cJSON_Delete(output);
                goto fail;
            }
            value = cJSON_CreateNumber((double)dimension);
            if (!value || !cJSON_AddItemToArray(shape, value)) {
                cJSON_Delete(value);
                cJSON_Delete(shape);
                cJSON_Delete(output);
                goto fail;
            }
        }
        if (!cJSON_AddItemToObject(output, "shape", shape) ||
            !cJSON_AddStringToObject(output, "dtype", dtype) ||
            !cJSON_AddStringToObject(
                output, "location",
                records[index].info.location == VX_MEMORY_DEVICE
                    ? "device" : "host") ||
            !cJSON_AddNumberToObject(output, "byteLength",
                                     (double)records[index].info.byte_size) ||
            !cJSON_AddItemToArray(outputs, output)) {
            cJSON_Delete(output);
            goto fail;
        }
        records[index].first_read = malloc(
            records[index].info.byte_size ? records[index].info.byte_size : 1u);
        if (!records[index].first_read) goto fail;
        if (records[index].info.byte_size > scratch_size)
            scratch_size = records[index].info.byte_size;
    }
    scratch = malloc(scratch_size ? scratch_size : 1u);
    if (!scratch) goto fail;
    for (size_t index = 0; index < count; index++) {
        const VxTensorInfo* info = &records[index].info;
        status = vx_result_read(result, info->name, records[index].first_read,
                                info->byte_size, NULL, report);
        if (status == VX_STATUS_OK)
            status = vx_result_read(result, info->name, scratch,
                                    info->byte_size, NULL, report);
        if (status != VX_STATUS_OK) {
            print_failure("Verifying fresh result reads", status, report);
            goto fail;
        }
        if (info->byte_size &&
            memcmp(records[index].first_read, scratch, info->byte_size)) {
            fprintf(stderr, "Repeated result reads differ for output %s.\n",
                    info->name);
            goto fail;
        }
    }
    status = vx_execution_context_close(context, report);
    if (status != VX_STATUS_OK) {
        print_failure("Closing context before stable result verification",
                      status, report);
        goto fail;
    }
    for (size_t index = 0; index < count; index++) {
        const VxTensorInfo* info = &records[index].info;
        status = vx_result_read(result, info->name, scratch, info->byte_size,
                                NULL, report);
        if (status != VX_STATUS_OK) {
            print_failure("Reading result after context close", status, report);
            goto fail;
        }
        if (info->byte_size &&
            memcmp(records[index].first_read, scratch, info->byte_size)) {
            fprintf(stderr,
                    "Result changed after context close for output %s.\n",
                    info->name);
            goto fail;
        }
    }
    if (!cJSON_AddItemToObject(stable, "outputs", outputs)) goto fail;
    outputs = NULL;
    if (!cJSON_AddBoolToObject(stable, "freshCallerOwnedReads", 1) ||
        !cJSON_AddBoolToObject(stable, "readableAfterContextClose", 1) ||
        !cJSON_AddBoolToObject(stable, "contextClosedBeforeResult", 1))
        goto fail;
    free(scratch);
    stable_read_evidence_free(records, count);
    return stable;
fail:
    free(scratch);
    stable_read_evidence_free(records, count);
    cJSON_Delete(outputs);
    cJSON_Delete(stable);
    return NULL;
}

static cJSON* runtime_evidence_json(const VxReport* compilation_report,
                                    const VxReport* execution_report,
                                    cJSON* stable_result) {
    cJSON* root = cJSON_CreateObject();
    cJSON* compilation = compilation_evidence_json(compilation_report);
    cJSON* execution = execution_evidence_json(execution_report);
    if (!root || !compilation || !execution || !stable_result ||
        !cJSON_AddStringToObject(root, "schema", "volvoxai.runtime-evidence") ||
        !cJSON_AddNumberToObject(root, "version", 1) ||
        !cJSON_AddItemToObject(root, "compilation", compilation)) {
        cJSON_Delete(execution);
        cJSON_Delete(compilation);
        cJSON_Delete(stable_result);
        cJSON_Delete(root);
        return NULL;
    }
    compilation = NULL;
    if (!cJSON_AddItemToObject(root, "execution", execution)) goto fail;
    execution = NULL;
    if (!cJSON_AddBoolToObject(stable_result,
                               "resultClosedAfterVerification", 1) ||
        !cJSON_AddItemToObject(root, "stableResult", stable_result))
        goto fail;
    return root;
fail:
    cJSON_Delete(execution);
    cJSON_Delete(compilation);
    cJSON_Delete(stable_result);
    cJSON_Delete(root);
    return NULL;
}

/* `--report-json` also carries typed public-runtime/API failures when a
 * VxReport is available. A strict physical backend probe can therefore
 * distinguish a proved compile rejection from missing-device or execution
 * infrastructure without scraping human text. CLI parsing, path resolution,
 * local binding validation, and output-file failures are not VxReport events.
 * Keep this schema separate from successful lifecycle evidence: a failed API
 * operation has no stable result and must never look like partial success. */
static cJSON* runtime_failure_evidence_json(const VxReport* report,
                                            const char* requested_backend,
                                            const VxModelSource* source) {
    cJSON* root = cJSON_CreateObject();
    cJSON* request = cJSON_CreateObject();
    cJSON* policy = cJSON_CreateObject();
    cJSON* source_json = cJSON_CreateObject();
    cJSON* weight_paths = cJSON_CreateArray();
    cJSON* reported = cJSON_CreateObject();
    cJSON* device = NULL;
    cJSON* lineage = cJSON_CreateObject();
    if (!report || report->status == VX_STATUS_OK ||
        !source || !source->graph_path || !source->graph_path[0] ||
        !root || !request || !policy || !source_json || !weight_paths ||
        !reported || !lineage ||
        !cJSON_AddStringToObject(
            root, "schema", "volvoxai.runtime-failure-evidence") ||
        !cJSON_AddNumberToObject(root, "version", 1) ||
        !cJSON_AddStringToObject(root, "outcome", "failure") ||
        !cJSON_AddStringToObject(root, "status",
                                 vx_status_string(report->status)) ||
        !cJSON_AddStringToObject(root, "stage",
                                 runtime_stage_name(report->stage)))
        goto fail;
    if (requested_backend) {
        if (!cJSON_AddStringToObject(request, "backend", requested_backend) ||
            !cJSON_AddStringToObject(policy, "mode", "require") ||
            !cJSON_AddStringToObject(policy, "operatorFallback", "forbid"))
            goto fail;
    } else if (!cJSON_AddNullToObject(request, "backend") ||
               !cJSON_AddStringToObject(policy, "mode", "prefer") ||
               !cJSON_AddStringToObject(policy, "operatorFallback", "allow")) {
        goto fail;
    }
    if (!cJSON_AddItemToObject(request, "policy", policy)) goto fail;
    policy = NULL;
    if (!cJSON_AddItemToObject(root, "request", request)) goto fail;
    request = NULL;

    if (!cJSON_AddStringToObject(source_json, "graphPath",
                                 source->graph_path))
        goto fail;
    for (size_t index = 0; index < source->weight_path_count; index++) {
        cJSON* weight_path;
        if (!source->weight_paths || !source->weight_paths[index] ||
            !source->weight_paths[index][0])
            goto fail;
        weight_path = cJSON_CreateString(source->weight_paths[index]);
        if (!weight_path || !cJSON_AddItemToArray(weight_paths, weight_path)) {
            cJSON_Delete(weight_path);
            goto fail;
        }
    }
    if (!cJSON_AddItemToObject(source_json, "weightPaths", weight_paths))
        goto fail;
    weight_paths = NULL;
    if (!cJSON_AddItemToObject(root, "source", source_json)) goto fail;
    source_json = NULL;

    if (report->backend[0]) {
        if (!cJSON_AddStringToObject(reported, "backend", report->backend))
            goto fail;
    } else if (!cJSON_AddNullToObject(reported, "backend")) {
        goto fail;
    }
    device = report_device_json(report);
    if (!device || !cJSON_AddItemToObject(reported, "device", device))
        goto fail;
    device = NULL;
#define ADD_REPORT_TEXT(name, field) \
    do { \
        if (report->field[0]) { \
            if (!cJSON_AddStringToObject(reported, name, report->field)) \
                goto fail; \
        } else if (!cJSON_AddNullToObject(reported, name)) { \
            goto fail; \
        } \
    } while (0)
    ADD_REPORT_TEXT("reason", reason);
    ADD_REPORT_TEXT("message", message);
    ADD_REPORT_TEXT("offendingNode", offending_node);
    ADD_REPORT_TEXT("candidateOutcomes", candidate_outcomes);
    ADD_REPORT_TEXT("routeEvidence", route_evidence);
    ADD_REPORT_TEXT("fallbackEvidence", fallback_evidence);
#undef ADD_REPORT_TEXT
    if (!cJSON_AddBoolToObject(reported, "tierFallback",
                               report->tier_fallback_used != 0) ||
        !cJSON_AddBoolToObject(reported, "operatorFallbackUsed",
                               report->operator_fallback_used != 0) ||
        !cJSON_AddBoolToObject(reported, "routeAttested",
                               report->route_attested != 0) ||
        !cJSON_AddItemToObject(root, "report", reported))
        goto fail;
    reported = NULL;

    if (json_add_optional_identity(lineage, "runtimeId", "runtime",
                                   report->runtime_id) != 0 ||
        json_add_optional_identity(lineage, "modelId", "model",
                                   report->model_id) != 0 ||
        json_add_optional_identity(lineage, "compilationId", "compiled",
                                   report->compiled_model_id) != 0 ||
        json_add_optional_identity(lineage, "definitionId", "graph",
                                   report->graph_id) != 0 ||
        json_add_optional_identity(lineage, "weightRevisionId", "weight",
                                   report->weight_id) != 0 ||
        json_add_optional_uint64_string(lineage, "topologyRevision",
                                        report->graph_revision) != 0 ||
        json_add_optional_uint64_string(lineage, "weightRevision",
                                        report->weight_revision) != 0 ||
        !cJSON_AddItemToObject(root, "lineage", lineage))
        goto fail;
    return root;
fail:
    cJSON_Delete(lineage);
    cJSON_Delete(device);
    cJSON_Delete(reported);
    cJSON_Delete(weight_paths);
    cJSON_Delete(source_json);
    cJSON_Delete(policy);
    cJSON_Delete(request);
    cJSON_Delete(root);
    return NULL;
}

static int write_runtime_evidence(const char* path, cJSON* evidence) {
    char* serialized;
    int result;
    if (!path || !evidence) return -1;
    serialized = cJSON_PrintUnformatted(evidence);
    if (!serialized) {
        fprintf(stderr, "Cannot serialize lifecycle evidence.\n");
        return -1;
    }
    result = write_bytes(path, serialized, strlen(serialized));
    free(serialized);
    return result;
}

static void print_release_info(void) {
    printf("VolvoxAI Native Engine %s (commit %s, built %s)\n",
           VOLVOXAI_VERSION, VOLVOXAI_GIT_COMMIT, VOLVOXAI_BUILD_DATE);
}

static void print_root_help(const char* argv0) {
    printf("VolvoxAI Native Engine\n\n");
    printf("Usage: %s <command> [options]\n\n", argv0);
    printf("Commands:\n");
    printf("  run <model-dir|graph.json>   Execute one generic tensor graph.\n");
#if VOLVOXAI_ENABLE_TRAINING
    printf("  train <model-dir|graph.json> Train selected weights privately, then commit.\n");
#endif
    printf("  version                      Print release information.\n\n");
    printf("The fixed CLI uses only the public opaque-handle runtime API.\n");
}

static void print_run_help(const char* argv0) {
    printf("Usage: %s run <model-dir|graph.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>             Add a safetensors weight shard (repeatable).\n");
    printf("  --input <name[shape]=file>   Bind one exact typed input; shape is required for dynamic axes.\n");
    printf("  --output <name=file|file>    Write exact typed raw output data.\n");
    printf("  --row <index>                Write one row from each selected F32 output.\n");
    printf("  --report-json <file>         Write lifecycle success or typed runtime/API failure evidence when available.\n");
    printf("  --threads <n>                Set the CPU worker count.\n");
    printf("  --cpu | --vulkan | --opengl | --metal | --nnapi | --cuda\n");
    printf("  --debug\n");
}

#if VOLVOXAI_ENABLE_TRAINING
static void print_train_help(const char* argv0) {
    printf("Usage: %s train <model-dir|graph.json> [options]\n\n", argv0);
    printf("Required options:\n");
    printf("  --targets <file.i32>         Raw cross-entropy target indices.\n");
    printf("  --logits <tensor>            F32 logits tensor name.\n");
    printf("  --trainable <tensor>         F32 model weight (repeatable).\n");
    printf("  --output-weights <file>      Export committed shard (repeatable).\n\n");
    printf("Other options:\n");
    printf("  --weights <file>             Add an input weight shard (repeatable).\n");
    printf("  --input <name=file>          Load exact typed raw graph input.\n");
    printf("  --microbatches <n>           Number of microbatches (default 1).\n");
    printf("  --accumulation-steps <n>     Private gradient window (default 1).\n");
    printf("  --optimizer <sgd|adamw>      Optimizer kind (default sgd).\n");
    printf("  --learning-rate <value>      Non-negative learning rate.\n");
    printf("  --beta1 <value> --beta2 <value> --epsilon <value>\n");
    printf("  --weight-decay <value> --max-gradient-norm <value>\n");
    printf("  --ignore-index <i> --row <i> --threads <n> --debug\n");
    printf("  --vulkan | --opengl | --metal | --cuda\n");
}
#endif

static int command_run(int argc, char** argv) {
    RunOptions options;
    ModelPaths paths;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxReport compilation_report = VX_REPORT_INIT;
    VxReport execution_report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    VxTensorBinding input_bindings[MAX_BINDINGS];
    void* input_storage[MAX_BINDINGS] = {0};
    cJSON* stable_result = NULL;
    cJSON* runtime_evidence = NULL;
    cJSON* runtime_failure_evidence = NULL;
    const char* selected_backends[1];
    VxStatus status;
    int return_code = 1;
    int report_json_written = 0;

    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_run_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    memset(&options, 0, sizeof(options));
    options.output_row = -1;
    for (int index = 3; index < argc; index++) {
        int parsed = parse_run_option(argc, argv, &index, &options);
        if (parsed < 0) return 2;
        if (!parsed) {
            fprintf(stderr, "Unknown run option: %s\n", argv[index]);
            return 2;
        }
    }
    if (resolve_model_paths(argv[2], &paths) != 0) return 1;
    if (!options.weight_path_count && paths.default_weights[0]) {
        options.weight_paths[options.weight_path_count++] = paths.default_weights;
    }

    runtime_options.debug = options.debug;
    runtime_options.cpu_threads = options.cpu_threads;
    source.graph_path = paths.graph;
    source.weight_paths = options.weight_paths;
    source.weight_path_count = options.weight_path_count;
    if (options.backend) {
        selected_backends[0] = options.backend;
        policy.mode = VX_BACKEND_REQUIRE;
        policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
        policy.backends = selected_backends;
        policy.backend_count = 1;
    }

    status = vx_runtime_create(&runtime_options, &runtime, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Native init", status, &report);
        goto cleanup;
    }
    status = vx_runtime_load_model(runtime, &source, &model, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Native init", status, &report);
        goto cleanup;
    }
    status = vx_model_compile(model, &policy, &compiled, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Native init", status, &report);
        goto cleanup;
    }
    compilation_report = report;
    printf("VolvoxAI Native Engine\nBackend: %s\n",
           report.backend[0] ? report.backend : "unknown");
    status = vx_compiled_model_create_context(compiled, &context_options,
                                              &context, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Native init", status, &report);
        goto cleanup;
    }
    if (options.input_count != vx_execution_context_input_count(context)) {
        fprintf(stderr,
                "Execution requires one atomic binding for each of the %zu graph inputs; got %zu.\n",
                vx_execution_context_input_count(context), options.input_count);
        goto cleanup;
    }
    for (size_t index = 0; index < options.input_count; index++) {
        if (prepare_input_binding(context, &options.inputs[index],
                                  &input_bindings[index],
                                  &input_storage[index], &report) != 0)
            goto cleanup;
    }
    status = vx_execution_context_execute(
        context, input_bindings, options.input_count, &result, &report);
    for (size_t index = 0; index < options.input_count; index++) {
        free(input_storage[index]);
        input_storage[index] = NULL;
    }
    if (status != VX_STATUS_OK) {
        print_failure("Native inference", status, &report);
        goto cleanup;
    }
    execution_report = report;
    if (options.debug) {
        fprintf(stderr,
                "[debug] execution=%llu backend=%s device=%s time_ms=%.3f outputs=%zu\n",
                (unsigned long long)vx_result_execution_id(result), report.backend,
                report.device, report.execution_time_ms,
                vx_result_output_count(result));
    }
    if (options.report_json) {
        stable_result = verify_stable_result(result, context, &report);
        if (!stable_result) goto cleanup;
    }
    for (size_t index = 0; index < options.output_count; index++) {
        if (write_output_binding(result, &options.outputs[index],
                                 options.output_row, &report) != 0)
            goto cleanup;
    }
    if (options.report_json) {
        vx_result_release(result);
        result = NULL;
        runtime_evidence = runtime_evidence_json(
            &compilation_report, &execution_report, stable_result);
        stable_result = NULL;
        if (!runtime_evidence ||
            write_runtime_evidence(options.report_json, runtime_evidence) != 0)
            goto cleanup;
        report_json_written = 1;
    }
    return_code = 0;

cleanup:
    if (return_code != 0 && options.report_json && !report_json_written &&
        report.status != VX_STATUS_OK) {
        runtime_failure_evidence = runtime_failure_evidence_json(
            &report, options.backend, &source);
        if (!runtime_failure_evidence ||
            write_runtime_evidence(options.report_json,
                                   runtime_failure_evidence) != 0) {
            fprintf(stderr, "Cannot write machine-readable failure evidence.\n");
        }
    }
    for (size_t index = 0; index < MAX_BINDINGS; index++)
        free(input_storage[index]);
    cJSON_Delete(runtime_failure_evidence);
    cJSON_Delete(runtime_evidence);
    cJSON_Delete(stable_result);
    vx_result_release(result);
    if (context) (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    return return_code;
}

#if VOLVOXAI_ENABLE_TRAINING
static int parse_train_float(const char* value, float* parsed) {
    char* end = NULL;
    float number;
    if (!value || !value[0] || !parsed) return -1;
    errno = 0;
    number = strtof(value, &end);
    if (errno == ERANGE || !end || *end || !isfinite(number)) return -1;
    *parsed = number;
    return 0;
}

static int parse_train_i32(const char* value, int32_t* parsed) {
    char* end = NULL;
    long number;
    if (!value || !value[0] || !parsed) return -1;
    errno = 0;
    number = strtol(value, &end, 10);
    if (errno == ERANGE || !end || *end || number < INT32_MIN ||
        number > INT32_MAX) return -1;
    *parsed = (int32_t)number;
    return 0;
}

static int read_whole_file(const char* path, void** data, size_t* byte_size) {
    FILE* file;
    long size;
    void* bytes;
    int failed = 0;
    if (!path || !data || !byte_size) return -1;
    *data = NULL;
    *byte_size = 0;
    file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0 ||
        (size = ftell(file)) < 0 || fseek(file, 0, SEEK_SET) != 0) {
        if (file) fclose(file);
        return -1;
    }
    if ((unsigned long)size > SIZE_MAX) {
        fclose(file);
        return -1;
    }
    bytes = malloc(size > 0 ? (size_t)size : 1u);
    if (!bytes) {
        fclose(file);
        return -1;
    }
    if (size > 0 && fread(bytes, 1, (size_t)size, file) != (size_t)size)
        failed = 1;
    if (fclose(file) != 0) failed = 1;
    if (failed) {
        free(bytes);
        return -1;
    }
    *data = bytes;
    *byte_size = (size_t)size;
    return 0;
}

static int find_trainer_input(VxTrainer* trainer, const char* name,
                              VxTensorSpec* found, VxReport* report) {
    size_t count = vx_trainer_input_count(trainer);
    for (size_t index = 0; index < count; index++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        if (vx_trainer_input_spec(trainer, index, &spec, report) == VX_STATUS_OK &&
            spec.name && !strcmp(spec.name, name)) {
            *found = spec;
            return 0;
        }
    }
    fprintf(stderr, "Unknown training input tensor: %s\n", name);
    return -1;
}

static int prepare_trainer_input_binding(VxTrainer* trainer,
                                         const Binding* binding,
                                         VxTensorBinding* prepared,
                                         void** storage,
                                         VxReport* report) {
    VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
    const char* suffix;
    void* bytes = NULL;
    size_t byte_size;
    size_t element_size;
    if (!prepared || !storage ||
        find_trainer_input(trainer, binding->name, &spec, report) != 0)
        return -1;
    if (binding->has_shape && binding->rank != spec.rank) {
        fprintf(stderr, "Training input %s expects rank %u, got rank %u.\n",
                binding->name, spec.rank, binding->rank);
        return -1;
    }
    *prepared = (VxTensorBinding)VX_TENSOR_BINDING_INIT;
    prepared->name = binding->name;
    prepared->dtype = spec.dtype;
    prepared->rank = spec.rank;
    byte_size = element_size = dtype_byte_size(spec.dtype);
    if (!element_size) return -1;
    for (uint32_t axis = 0; axis < spec.rank; axis++) {
        int64_t extent;
        if (binding->has_shape) {
            extent = binding->shape[axis];
        } else {
            if (spec.dimensions[axis].kind != VX_DIMENSION_FIXED) {
                fprintf(stderr,
                        "Dynamic training input %s requires an explicit shape.\n",
                        binding->name);
                return -1;
            }
            extent = spec.dimensions[axis].min;
        }
        if (extent <= 0 || (uint64_t)extent > SIZE_MAX / byte_size)
            return -1;
        prepared->shape[axis] = extent;
        byte_size *= (size_t)extent;
    }
    suffix = dtype_suffix(spec.dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Training input %s has dtype %s and requires a %s file: %s\n",
                binding->name, dtype_name(spec.dtype),
                suffix ? suffix : "supported raw", binding->path);
        return -1;
    }
    if (read_exact_file(binding->path, byte_size, &bytes) != 0)
        return -1;
    prepared->data = bytes;
    prepared->byte_size = byte_size;
    prepared->location = VX_MEMORY_HOST;
    *storage = bytes;
    return 0;
}

static int trainable_duplicate(const TrainOptions* options, const char* name) {
    for (size_t index = 0; index < options->trainable_count; index++)
        if (!strcmp(options->trainable_names[index], name)) return 1;
    return 0;
}

static int command_train(int argc, char** argv) {
    TrainOptions options;
    ModelPaths paths;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxTrainerOptions trainer_options = VX_TRAINER_OPTIONS_INIT;
    VxCrossEntropyLoss loss = VX_CROSS_ENTROPY_LOSS_INIT;
    VxTrainStepOptions step_options = VX_TRAIN_STEP_OPTIONS_INIT;
    VxTrainStepResult step_result = VX_TRAIN_STEP_RESULT_INIT;
    VxRevisionInfo published = VX_REVISION_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxTrainer* trainer = NULL;
    VxTensorBinding input_bindings[MAX_BINDINGS];
    void* input_storage[MAX_BINDINGS] = {0};
    void* targets_storage = NULL;
    size_t targets_bytes = 0;
    const char* selected_backend;
    int return_code = 1;
    VxStatus status;

    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_train_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    memset(&options, 0, sizeof(options));
    options.run.output_row = -1;
    options.microbatches = 1;
    options.accumulation_steps = 1;
    options.optimizer = (VxOptimizerOptions)VX_OPTIMIZER_OPTIONS_INIT;
    options.ignore_index = -100;
    options.loss_row = -1;
    for (int index = 3; index < argc; index++) {
        const char* argument = argv[index];
        if (!strcmp(argument, "--targets")) {
            if (++index >= argc || options.targets_path) goto usage_error;
            options.targets_path = argv[index];
        } else if (!strcmp(argument, "--logits")) {
            if (++index >= argc || options.logits_name) goto usage_error;
            options.logits_name = argv[index];
        } else if (!strcmp(argument, "--trainable")) {
            const char* name;
            if (++index >= argc || options.trainable_count == MAX_BINDINGS)
                goto usage_error;
            name = argv[index];
            if (!name[0] || trainable_duplicate(&options, name)) goto usage_error;
            options.trainable_names[options.trainable_count++] = name;
        } else if (!strcmp(argument, "--output-weights")) {
            if (++index >= argc ||
                options.output_weight_path_count == MAX_WEIGHT_PATHS)
                goto usage_error;
            options.output_weight_paths[options.output_weight_path_count++] =
                argv[index];
        } else if (!strcmp(argument, "--microbatches")) {
            int parsed;
            if (++index >= argc ||
                parse_nonnegative_int(argv[index], 0, &parsed) != 0 || !parsed)
                goto usage_error;
            options.microbatches = parsed;
        } else if (!strcmp(argument, "--accumulation-steps")) {
            int parsed;
            if (++index >= argc ||
                parse_nonnegative_int(argv[index], 0, &parsed) != 0 || !parsed)
                goto usage_error;
            options.accumulation_steps = (uint32_t)parsed;
        } else if (!strcmp(argument, "--optimizer")) {
            if (++index >= argc) goto usage_error;
            if (!strcmp(argv[index], "sgd"))
                options.optimizer.kind = VX_OPTIMIZER_SGD;
            else if (!strcmp(argv[index], "adamw"))
                options.optimizer.kind = VX_OPTIMIZER_ADAMW;
            else goto usage_error;
        } else if (!strcmp(argument, "--learning-rate")) {
            if (++index >= argc || parse_train_float(
                    argv[index], &options.optimizer.learning_rate) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--beta1")) {
            if (++index >= argc || parse_train_float(
                    argv[index], &options.optimizer.beta1) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--beta2")) {
            if (++index >= argc || parse_train_float(
                    argv[index], &options.optimizer.beta2) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--epsilon")) {
            if (++index >= argc || parse_train_float(
                    argv[index], &options.optimizer.epsilon) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--weight-decay")) {
            if (++index >= argc || parse_train_float(
                    argv[index], &options.optimizer.weight_decay) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--max-gradient-norm")) {
            if (++index >= argc || parse_train_float(
                    argv[index], &options.optimizer.max_gradient_norm) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--ignore-index")) {
            if (++index >= argc || parse_train_i32(
                    argv[index], &options.ignore_index) != 0)
                goto usage_error;
        } else if (!strcmp(argument, "--row")) {
            if (++index >= argc || parse_train_i32(
                    argv[index], &options.loss_row) != 0 || options.loss_row < -1)
                goto usage_error;
        } else {
            int parsed = parse_run_option(argc, argv, &index, &options.run);
            if (parsed <= 0) goto usage_error;
        }
    }
    if (!options.targets_path || !options.logits_name ||
        !options.trainable_count || !options.output_weight_path_count ||
        options.run.output_count || options.run.output_row != -1 ||
        options.run.report_json ||
        options.optimizer.learning_rate < 0.0f ||
        options.optimizer.beta1 < 0.0f || options.optimizer.beta1 >= 1.0f ||
        options.optimizer.beta2 < 0.0f || options.optimizer.beta2 >= 1.0f ||
        options.optimizer.epsilon <= 0.0f ||
        options.optimizer.weight_decay < 0.0f ||
        options.optimizer.max_gradient_norm < 0.0f) goto usage_error;
    if (!has_suffix(options.targets_path, ".i32") ||
        read_whole_file(options.targets_path, &targets_storage,
                        &targets_bytes) != 0 || !targets_bytes ||
        targets_bytes % sizeof(int32_t) != 0 ||
        targets_bytes / sizeof(int32_t) > INT_MAX) {
        fprintf(stderr, "--targets must be a nonempty raw .i32 file.\n");
        goto cleanup;
    }
    if (resolve_model_paths(argv[2], &paths) != 0) goto cleanup;
    if (!options.run.weight_path_count && paths.default_weights[0])
        options.run.weight_paths[options.run.weight_path_count++] =
            paths.default_weights;
    if (options.output_weight_path_count != options.run.weight_path_count) {
        fprintf(stderr,
                "--output-weights count must match the %zu input weight shard(s).\n",
                options.run.weight_path_count);
        return_code = 2;
        goto cleanup;
    }
    runtime_options.debug = options.run.debug;
    runtime_options.cpu_threads = options.run.cpu_threads;
    source.graph_path = paths.graph;
    source.weight_paths = options.run.weight_paths;
    source.weight_path_count = options.run.weight_path_count;
    selected_backend = options.run.backend ? options.run.backend : "cpu";
    trainer_options.backend = selected_backend;

    status = vx_runtime_create(&runtime_options, &runtime, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Training runtime creation", status, &report);
        goto cleanup;
    }
    status = vx_runtime_load_model(runtime, &source, &model, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Training model load", status, &report);
        goto cleanup;
    }
    status = vx_model_create_trainer(model, &trainer_options, &trainer, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Trainer creation", status, &report);
        goto cleanup;
    }
    if (options.run.input_count != vx_trainer_input_count(trainer)) {
        fprintf(stderr,
                "Training requires one atomic shaped binding for each of the %zu graph inputs; got %zu.\n",
                vx_trainer_input_count(trainer), options.run.input_count);
        goto cleanup;
    }
    for (size_t index = 0; index < options.run.input_count; index++)
        if (prepare_trainer_input_binding(
                trainer, &options.run.inputs[index], &input_bindings[index],
                &input_storage[index], &report) != 0)
            goto cleanup;

    loss.logits_name = options.logits_name;
    loss.targets = (const int32_t*)targets_storage;
    loss.target_count = targets_bytes / sizeof(int32_t);
    loss.ignore_index = options.ignore_index;
    loss.row_index = options.loss_row;
    if (options.accumulation_steps > 1u)
        loss.normalizer = (float)loss.target_count *
                          (float)options.accumulation_steps;
    step_options.inputs = input_bindings;
    step_options.input_count = options.run.input_count;
    step_options.losses = &loss;
    step_options.loss_count = 1;
    step_options.trainable_names = options.trainable_names;
    step_options.trainable_count = options.trainable_count;
    step_options.optimizer = options.optimizer;
    step_options.accumulation_steps = options.accumulation_steps;
    printf("Training backend=%s optimizer=%s microbatches=%ld accumulation=%u\n",
           selected_backend,
           options.optimizer.kind == VX_OPTIMIZER_ADAMW ? "adamw" : "sgd",
           options.microbatches, options.accumulation_steps);
    for (long index = 0; index < options.microbatches; index++) {
        step_result = (VxTrainStepResult)VX_TRAIN_STEP_RESULT_INIT;
        step_options.flush_accumulation =
            index + 1 == options.microbatches ? 1 : 0;
        status = vx_trainer_train_step(
            trainer, &step_options, &step_result, &report);
        if (status != VX_STATUS_OK) {
            print_failure("Training step", status, &report);
            goto cleanup;
        }
        printf("microbatch=%llu optimizer_step=%llu loss=%.8g "
               "accumulated=%u update=%s backend=%s\n",
               (unsigned long long)step_result.microbatch_id,
               (unsigned long long)step_result.optimizer_step,
               step_result.loss, step_result.accumulated_microbatches,
               step_result.update_applied ? "yes" : "no",
               step_result.backend);
    }
    status = vx_trainer_commit(trainer, &published, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Training commit", status, &report);
        goto cleanup;
    }
    status = vx_trainer_export_weights(
        trainer, options.output_weight_paths,
        options.output_weight_path_count, &report);
    if (status != VX_STATUS_OK) {
        print_failure("Training export", status, &report);
        goto cleanup;
    }
    printf("Published weight revision %llu and exported %zu shard(s).\n",
           (unsigned long long)published.weight_revision,
           options.output_weight_path_count);
    return_code = 0;
    goto cleanup;

usage_error:
    fprintf(stderr, "Invalid train options. Use '%s train --help'.\n", argv[0]);
    return_code = 2;

cleanup:
    for (size_t index = 0; index < MAX_BINDINGS; index++)
        free(input_storage[index]);
    free(targets_storage);
    if (trainer) (void)vx_trainer_close(trainer, NULL);
    vx_trainer_release(trainer);
    vx_model_release(model);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    return return_code;
}
#endif

int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        print_root_help(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    if (!strcmp(argv[1], "--version") || !strcmp(argv[1], "-V") ||
        !strcmp(argv[1], "version")) {
        print_release_info();
        return 0;
    }
    if (!strcmp(argv[1], "run")) return command_run(argc, argv);
#if VOLVOXAI_ENABLE_TRAINING
    if (!strcmp(argv[1], "train")) return command_train(argc, argv);
#endif
    fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
    print_root_help(argv[0]);
    return 2;
}
