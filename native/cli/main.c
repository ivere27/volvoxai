#include "cJSON.h"
#include "call_client.h"
#include "../src/generated/proto_methods.h"
#include "volvoxai_lite.h"

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
#define CLI_MAX_TENSOR_RANK 8u
#define TENSOR_INLINE_PAYLOAD 5

/* Keep the version and its release-context marker contiguous in allocated
 * data. Native cross builds cannot execute the target ELF during packaging,
 * so the finalizer verifies this exact prefix instead of accepting a null
 * version provenance record. */
static const char vx_release_version_prefix[] =
    "VolvoxAI Native Engine " VOLVOXAI_VERSION " (";

typedef struct Binding {
    char name[128];
    char path[PATH_MAX];
    int64_t shape[CLI_MAX_TENSOR_RANK];
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
    VolvoxaiV1TrainingOptimizerKind optimizer_kind;
    int has_learning_rate;
    float learning_rate;
    int has_beta1;
    float beta1;
    int has_beta2;
    float beta2;
    int has_epsilon;
    float epsilon;
    int has_weight_decay;
    float weight_decay;
    int has_max_gradient_norm;
    float max_gradient_norm;
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
        for (cursor = shape_end; cursor > argument; cursor--) {
            if (cursor[-1] == '[') {
                shape_open = cursor - 1;
                break;
            }
        }
        if (!shape_open || shape_open == argument) return -1;
        cursor = shape_open + 1;
        while (cursor < shape_end) {
            char* parsed_end = NULL;
            long long extent;
            if (binding->rank == CLI_MAX_TENSOR_RANK) return -1;
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
        if (*index + 1 >= argc || options->weight_path_count == MAX_WEIGHT_PATHS) {
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

static const char* dtype_name(VolvoxaiV1DataType dtype) {
    const char* name = volvoxai_v1_data_type_name(dtype);
    return name ? name : "DATA_TYPE_UNSPECIFIED";
}

static const char* dtype_suffix(VolvoxaiV1DataType dtype) {
    switch (dtype) {
        case VOLVOXAI_V1_DATA_TYPE_F32: return ".f32";
        case VOLVOXAI_V1_DATA_TYPE_I8: return ".i8";
        case VOLVOXAI_V1_DATA_TYPE_U8: return ".u8";
        case VOLVOXAI_V1_DATA_TYPE_I32: return ".i32";
        default: return NULL;
    }
}

static size_t dtype_byte_size(VolvoxaiV1DataType dtype) {
    switch (dtype) {
        case VOLVOXAI_V1_DATA_TYPE_I8:
        case VOLVOXAI_V1_DATA_TYPE_U8: return 1u;
        case VOLVOXAI_V1_DATA_TYPE_F32:
        case VOLVOXAI_V1_DATA_TYPE_I32: return 4u;
        default: return 0u;
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
    bytes = malloc(expected ? expected : 1u);
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
    if (size > 0 && fread(bytes, 1, (size_t)size, file) != (size_t)size) failed = 1;
    if (fclose(file) != 0) failed = 1;
    if (failed) {
        free(bytes);
        return -1;
    }
    *data = bytes;
    *byte_size = (size_t)size;
    return 0;
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

static int assign_text(const SynurangLiteAllocator* allocator,
                       SynurangLiteBytes* field, const char* text) {
    return synurang_lite_bytes_assign(
               allocator, field, text, text ? strlen(text) : 0u) == SYNURANG_LITE_OK
               ? 0
               : -1;
}

static int copy_last_error_text(VxCallClient* client,
                                char* buffer, size_t capacity) {
    int32_t length = 0;
    const uint8_t* bytes;
    size_t copy_length;
    if (!buffer || !capacity) return -1;
    buffer[0] = 0;
    bytes = vx_call_error(client, &length);
    if (!bytes || length <= 0) return 0;
    copy_length = (size_t)length;
    if (copy_length >= capacity) copy_length = capacity - 1u;
    memcpy(buffer, bytes, copy_length);
    buffer[copy_length] = 0;
    return 0;
}

static void print_dispatch_error(const char* operation, VxCallClient* client) {
    int32_t length = 0;
    const uint8_t* bytes = vx_call_error(client, &length);
    fprintf(stderr, "%s failed", operation);
    if (bytes && length > 0) {
        fprintf(stderr, ": %.*s", (int)length, (const char*)bytes);
    }
    fputc('\n', stderr);
}

static const char* native_status_name(VolvoxaiV1NativeStatus status) {
    const char* name = volvoxai_v1_native_status_name(status);
    return name ? name : "NATIVE_STATUS_UNSPECIFIED";
}

static const char* stage_name(VolvoxaiV1OperationStage stage) {
    const char* name = volvoxai_v1_operation_stage_name(stage);
    return name ? name : "OPERATION_STAGE_UNSPECIFIED";
}

static int report_ok(const char* operation, const VolvoxaiV1OperationReport* report) {
    if (report && report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK) return 1;
    fprintf(stderr, "%s failed", operation);
    if (report) {
        fprintf(stderr, ": %s", native_status_name(report->field_status));
        if (report->field_code != VOLVOXAI_V1_OPERATION_CODE_NONE) {
            fprintf(stderr, " [code=%d]", (int)report->field_code);
        }
        if (report->field_message.len) {
            fprintf(stderr, ": %.*s", (int)report->field_message.len,
                    (const char*)report->field_message.data);
        }
    } else {
        fprintf(stderr, ": missing operation report");
    }
    fputc('\n', stderr);
    return 0;
}


static int bytes_equal_string(const SynurangLiteBytes* value, const char* text) {
    size_t text_len;
    if (!value || !text) return 0;
    text_len = strlen(text);
    return value->len == text_len &&
           (text_len == 0 || memcmp(value->data, text, text_len) == 0);
}

static const VolvoxaiV1TensorSpec* find_tensor_spec(
    const VolvoxaiV1TensorSpec* specs, size_t count, const char* name) {
    size_t index;
    for (index = 0; index < count; index++) {
        if (bytes_equal_string(&specs[index].field_name, name)) return &specs[index];
    }
    return NULL;
}

static const VolvoxaiV1TensorInfo* find_tensor_info(
    const VolvoxaiV1TensorInfo* infos, size_t count, const char* name) {
    size_t index;
    if (!count) return NULL;
    if (!name) return &infos[0];
    for (index = 0; index < count; index++) {
        if (bytes_equal_string(&infos[index].field_name, name)) return &infos[index];
    }
    return NULL;
}

static int validate_dimension(const char* input_name,
                              const VolvoxaiV1DimensionConstraint* constraint,
                              int64_t extent) {
    if (!constraint || extent <= 0) {
        fprintf(stderr, "Input %s has an invalid extent.\n", input_name);
        return -1;
    }
    if (constraint->field_min > 0 && extent < constraint->field_min) {
        fprintf(stderr, "Input %s extent %lld is below declared minimum %lld.\n",
                input_name, (long long)extent, (long long)constraint->field_min);
        return -1;
    }
    if (constraint->field_max > 0 && extent > constraint->field_max) {
        fprintf(stderr, "Input %s extent %lld exceeds declared maximum %lld.\n",
                input_name, (long long)extent, (long long)constraint->field_max);
        return -1;
    }
    if (constraint->field_multiple_of > 0 &&
        extent % constraint->field_multiple_of != 0) {
        fprintf(stderr,
                "Input %s extent %lld is not a multiple of %lld.\n",
                input_name, (long long)extent,
                (long long)constraint->field_multiple_of);
        return -1;
    }
    return 0;
}

static int fill_tensor_from_binding(const Binding* binding,
                                    const VolvoxaiV1TensorSpec* spec,
                                    VolvoxaiV1Tensor* tensor,
                                    const SynurangLiteAllocator* allocator) {
    void* raw = NULL;
    size_t byte_size;
    size_t element_size;
    size_t axis;
    const char* suffix;
    if (!binding || !spec || !tensor) return -1;
    volvoxai_v1_tensor_init_with_allocator(tensor, allocator);
    if (assign_text(allocator, &tensor->field_name, binding->name) != 0) goto oom;
    tensor->field_dtype = spec->field_dtype;
    element_size = dtype_byte_size(spec->field_dtype);
    if (!element_size) {
        fprintf(stderr, "Input %s uses unsupported CLI dtype %s.\n",
                binding->name, dtype_name(spec->field_dtype));
        return -1;
    }
    if (binding->has_shape && binding->rank != spec->field_dimensions.len) {
        fprintf(stderr, "Input %s expects rank %zu, got rank %u.\n",
                binding->name, spec->field_dimensions.len, binding->rank);
        return -1;
    }
    byte_size = element_size;
    for (axis = 0; axis < spec->field_dimensions.len; axis++) {
        const VolvoxaiV1DimensionConstraint* constraint = &spec->field_dimensions.data[axis];
        int64_t extent;
        int64_t* slot;
        if (binding->has_shape) {
            extent = binding->shape[axis];
        } else {
            if (constraint->field_kind != VOLVOXAI_V1_DIMENSION_KIND_FIXED) {
                fprintf(stderr,
                        "Dynamic input %s requires an explicit shape: "
                        "--input %s[d0,d1,...]=%s\n",
                        binding->name, binding->name, binding->path);
                return -1;
            }
            extent = constraint->field_min;
        }
        if (validate_dimension(binding->name, constraint, extent) != 0) return -1;
        if ((uint64_t)extent > SIZE_MAX / byte_size) {
            fprintf(stderr, "Input %s shape is too large.\n", binding->name);
            return -1;
        }
        slot = volvoxai_v1_tensor_add_shape(tensor);
        if (!slot) goto oom;
        *slot = extent;
        byte_size *= (size_t)extent;
    }
    suffix = dtype_suffix(spec->field_dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Input %s has dtype %s and requires a %s file: %s\n",
                binding->name, dtype_name(spec->field_dtype),
                suffix ? suffix : "supported raw", binding->path);
        return -1;
    }
    if (read_exact_file(binding->path, byte_size, &raw) != 0) {
        if (file_exists(binding->path)) {
            fprintf(stderr, "Input %s expects %zu raw bytes.\n",
                    binding->name, byte_size);
        }
        return -1;
    }
    if (synurang_lite_bytes_assign(allocator, &tensor->field_inline, raw, byte_size) !=
        SYNURANG_LITE_OK) {
        free(raw);
        goto oom;
    }
    free(raw);
    tensor->which_payload = TENSOR_INLINE_PAYLOAD;
    return 0;

oom:
    fprintf(stderr, "Out of memory while encoding input %s.\n", binding->name);
    return -1;
}

static int encode_create_runtime_request(const RunOptions* options,
                                         VolvoxaiV1ExecutionMode mode,
                                         uint8_t** encoded,
                                         size_t* encoded_len) {
    VolvoxaiV1CreateRuntimeRequest request;
    volvoxai_v1_create_runtime_request_init(&request);
    request.field_debug = options->debug;
    if (options->cpu_threads > 0) request.field_cpu_threads = options->cpu_threads;
    request.has_execution_mode = 1;
    request.field_execution_mode = mode;
    if (volvoxai_v1_create_runtime_request_encode(&request, encoded, encoded_len) !=
        SYNURANG_LITE_OK) {
        volvoxai_v1_create_runtime_request_free(&request);
        return -1;
    }
    volvoxai_v1_create_runtime_request_free(&request);
    return 0;
}

static int encode_load_model_request(int64_t runtime_id, const ModelPaths* paths,
                                     const RunOptions* options,
                                     uint8_t** encoded, size_t* encoded_len) {
    VolvoxaiV1LoadModelRequest request;
    size_t index;
    volvoxai_v1_load_model_request_init(&request);
    request.field_runtime_id = runtime_id;
    if (assign_text(request._allocator, &request.field_graph_path, paths->graph) != 0) {
        volvoxai_v1_load_model_request_free(&request);
        return -1;
    }
    for (index = 0; index < options->weight_path_count; index++) {
        SynurangLiteBytes* slot = volvoxai_v1_load_model_request_add_weight_paths(&request);
        if (!slot ||
            assign_text(request._allocator, slot, options->weight_paths[index]) != 0) {
            volvoxai_v1_load_model_request_free(&request);
            return -1;
        }
    }
    if (volvoxai_v1_load_model_request_encode(&request, encoded, encoded_len) !=
        SYNURANG_LITE_OK) {
        volvoxai_v1_load_model_request_free(&request);
        return -1;
    }
    volvoxai_v1_load_model_request_free(&request);
    return 0;
}

static int encode_compile_model_request(int64_t model_id, const char* backend,
                                         uint8_t** encoded, size_t* encoded_len) {
    VolvoxaiV1CompileModelRequest request;
    VolvoxaiV1BackendPolicy policy;
    int result = -1;
    volvoxai_v1_compile_model_request_init(&request);
    volvoxai_v1_backend_policy_init(&policy);
    request.field_model_id = model_id;
    if (backend) {
        policy.field_mode = VOLVOXAI_V1_BACKEND_POLICY_MODE_REQUIRE;
        policy.field_operator_fallback = VOLVOXAI_V1_OPERATOR_FALLBACK_FORBID;
        SynurangLiteBytes* entry = volvoxai_v1_backend_policy_add_backends(&policy);
        if (!entry || assign_text(policy._allocator, entry, backend) != 0) goto cleanup;
        request.field_policy = &policy;
    }
    result = volvoxai_v1_compile_model_request_encode(&request, encoded, encoded_len) ==
             SYNURANG_LITE_OK ? 0 : -1;
cleanup:
    request.field_policy = NULL;
    volvoxai_v1_backend_policy_free(&policy);
    volvoxai_v1_compile_model_request_free(&request);
    return result;
}

static void print_debug_route(const VolvoxaiV1OperationReport* report,
                              const char* label, double duration_ms) {
    const VolvoxaiV1RouteEvidence* route;
    if (!report) return;
    route = report->field_route;
    fprintf(stderr, "[debug] %s status=%s", label, native_status_name(report->field_status));
    if (report->field_backend.len) {
        fprintf(stderr, " backend=%.*s", (int)report->field_backend.len,
                (const char*)report->field_backend.data);
    }
    if (report->field_device.len) {
        fprintf(stderr, " device=%.*s", (int)report->field_device.len,
                (const char*)report->field_device.data);
    }
    fprintf(stderr, " host_ms=%.3f", duration_ms);
    if (route) {
        fprintf(stderr, " route=%.*s attested=%d active=%u selected=%u",
                (int)route->field_provider.len,
                route->field_provider.data ? (const char*)route->field_provider.data : "",
                route->field_attested, route->field_active_nodes,
                route->field_selected_nodes);
    }
    fputc('\n', stderr);
}

static int json_add_string_copy(cJSON* object, const char* key,
                                const char* value) {
    return cJSON_AddStringToObject(object, key, value ? value : "") ? 0 : -1;
}

static int json_add_optional_string_copy(cJSON* object, const char* key,
                                         const char* value) {
    if (!value || !value[0]) return cJSON_AddNullToObject(object, key) ? 0 : -1;
    return json_add_string_copy(object, key, value);
}

static int json_add_bytes_copy(cJSON* object, const char* key,
                               const SynurangLiteBytes* value) {
    char* text;
    cJSON* item;
    if (!object || !key || !value) return -1;
    text = (char*)malloc(value->len + 1u);
    if (!text) return -1;
    if (value->len) memcpy(text, value->data, value->len);
    text[value->len] = 0;
    item = cJSON_CreateString(text);
    free(text);
    if (!item) return -1;
    if (!cJSON_AddItemToObject(object, key, item)) {
        cJSON_Delete(item);
        return -1;
    }
    return 0;
}

static int json_add_optional_bytes_copy(cJSON* object, const char* key,
                                        const SynurangLiteBytes* value) {
    if (!value || !value->len) return cJSON_AddNullToObject(object, key) ? 0 : -1;
    return json_add_bytes_copy(object, key, value);
}

static cJSON* json_build_report(const VolvoxaiV1OperationReport* report) {
    cJSON* object;
    cJSON* lineage = NULL;
    cJSON* route = NULL;
    if (!report) return cJSON_CreateNull();
    object = cJSON_CreateObject();
    if (!object ||
        json_add_string_copy(object, "status",
                             native_status_name(report->field_status)) != 0 ||
        json_add_string_copy(object, "stage",
                             stage_name(report->field_stage)) != 0 ||
        !cJSON_AddNumberToObject(object, "code", report->field_code) ||
        json_add_optional_bytes_copy(object, "message",
                                     &report->field_message) != 0 ||
        json_add_optional_bytes_copy(object, "backend",
                                     &report->field_backend) != 0 ||
        json_add_optional_bytes_copy(object, "device",
                                     &report->field_device) != 0) {
        cJSON_Delete(object);
        return NULL;
    }
    if (report->field_lineage) {
        lineage = cJSON_CreateObject();
        if (!lineage ||
            !cJSON_AddNumberToObject(lineage, "runtimeId",
                                     (double)report->field_lineage->field_runtime_id) ||
            !cJSON_AddNumberToObject(lineage, "modelId",
                                     (double)report->field_lineage->field_model_id) ||
            !cJSON_AddNumberToObject(lineage, "compiledModelId",
                                     (double)report->field_lineage->field_compiled_model_id) ||
            !cJSON_AddNumberToObject(lineage, "contextId",
                                     (double)report->field_lineage->field_context_id) ||
            !cJSON_AddNumberToObject(lineage, "executionId",
                                     (double)report->field_lineage->field_execution_id) ||
            !cJSON_AddItemToObject(object, "lineage", lineage)) {
            cJSON_Delete(lineage);
            cJSON_Delete(object);
            return NULL;
        }
    } else if (!cJSON_AddNullToObject(object, "lineage")) {
        cJSON_Delete(object);
        return NULL;
    }
    if (report->field_route) {
        route = cJSON_CreateObject();
        if (!route ||
            json_add_optional_bytes_copy(route, "provider",
                                         &report->field_route->field_provider) != 0 ||
            !cJSON_AddBoolToObject(route, "builtin", report->field_route->field_builtin) ||
            !cJSON_AddBoolToObject(route, "attested", report->field_route->field_attested) ||
            !cJSON_AddNumberToObject(route, "activeNodes",
                                     (double)report->field_route->field_active_nodes) ||
            !cJSON_AddNumberToObject(route, "selectedNodes",
                                     (double)report->field_route->field_selected_nodes) ||
            !cJSON_AddNumberToObject(route, "fallbackNodes",
                                     (double)report->field_route->field_fallback_nodes) ||
            !cJSON_AddNumberToObject(route, "missingNodes",
                                     (double)report->field_route->field_missing_nodes) ||
            !cJSON_AddItemToObject(object, "route", route)) {
            cJSON_Delete(route);
            cJSON_Delete(object);
            return NULL;
        }
    } else if (!cJSON_AddNullToObject(object, "route")) {
        cJSON_Delete(object);
        return NULL;
    }
    return object;
}

static cJSON* json_build_outputs(const VolvoxaiV1ResultInfo* info) {
    cJSON* array;
    size_t index;
    if (!info) return cJSON_CreateNull();
    array = cJSON_CreateArray();
    if (!array) return NULL;
    for (index = 0; index < info->field_outputs.len; index++) {
        const VolvoxaiV1TensorInfo* output = &info->field_outputs.data[index];
        cJSON* item = cJSON_CreateObject();
        cJSON* shape = cJSON_CreateArray();
        size_t axis;
        if (!item || !shape ||
            json_add_bytes_copy(item, "name", &output->field_name) != 0 ||
            json_add_string_copy(item, "dtype", dtype_name(output->field_dtype)) != 0 ||
            !cJSON_AddNumberToObject(item, "byteLength",
                                     (double)output->field_byte_size)) {
            cJSON_Delete(shape);
            cJSON_Delete(item);
            cJSON_Delete(array);
            return NULL;
        }
        for (axis = 0; axis < output->field_shape.len; axis++) {
            cJSON* dimension = cJSON_CreateNumber((double)output->field_shape.data[axis]);
            if (!dimension || !cJSON_AddItemToArray(shape, dimension)) {
                cJSON_Delete(dimension);
                cJSON_Delete(shape);
                cJSON_Delete(item);
                cJSON_Delete(array);
                return NULL;
            }
        }
        if (!cJSON_AddItemToObject(item, "shape", shape) ||
            !cJSON_AddItemToArray(array, item)) {
            cJSON_Delete(item);
            cJSON_Delete(array);
            return NULL;
        }
    }
    return array;
}

static int write_report_json(const char* path, cJSON* root) {
    char* serialized;
    int result;
    if (!path || !root) return -1;
    serialized = cJSON_PrintUnformatted(root);
    if (!serialized) return -1;
    result = write_bytes(path, serialized, strlen(serialized));
    free(serialized);
    return result;
}

static cJSON* json_build_run_success(const ModelPaths* paths, const RunOptions* options,
                                     const VolvoxaiV1OperationReport* compile_report,
                                     const VolvoxaiV1ExecutionResultHandle* execution,
                                     const VolvoxaiV1ResultInfo* result_info) {
    cJSON* root = cJSON_CreateObject();
    cJSON* request = cJSON_CreateObject();
    cJSON* weights = cJSON_CreateArray();
    cJSON* compile = json_build_report(compile_report);
    cJSON* execute = json_build_report(execution ? execution->field_report : NULL);
    cJSON* outputs = json_build_outputs(result_info);
    size_t index;
    if (!root || !request || !weights || !compile || !execute || !outputs) goto fail;
    if (json_add_string_copy(root, "schema", "volvoxai.native-cli.run-report") != 0 ||
        !cJSON_AddNumberToObject(root, "version", 1) ||
        json_add_string_copy(request, "graphPath", paths->graph) != 0 ||
        json_add_optional_string_copy(request, "backend", options->backend) != 0) {
        goto fail;
    }
    for (index = 0; index < options->weight_path_count; index++) {
        cJSON* entry = cJSON_CreateString(options->weight_paths[index]);
        if (!entry || !cJSON_AddItemToArray(weights, entry)) {
            cJSON_Delete(entry);
            goto fail;
        }
    }
    if (!cJSON_AddItemToObject(request, "weightPaths", weights) ||
        !cJSON_AddItemToObject(root, "request", request) ||
        !cJSON_AddItemToObject(root, "compile", compile) ||
        !cJSON_AddItemToObject(root, "execute", execute) ||
        !cJSON_AddItemToObject(root, "outputs", outputs)) {
        goto fail;
    }
    if (!cJSON_AddNumberToObject(root, "resultId", (double)execution->field_result_id) ||
        !cJSON_AddNumberToObject(root, "executionId",
                                 (double)execution->field_execution_id)) {
        goto fail;
    }
    return root;

fail:
    cJSON_Delete(outputs);
    cJSON_Delete(execute);
    cJSON_Delete(compile);
    cJSON_Delete(weights);
    cJSON_Delete(request);
    cJSON_Delete(root);
    return NULL;
}

static cJSON* json_build_failure(const ModelPaths* paths, const RunOptions* options,
                                 const char* operation,
                                 const VolvoxaiV1OperationReport* report,
                                 const char* dispatch_error) {
    cJSON* root = cJSON_CreateObject();
    cJSON* request = cJSON_CreateObject();
    cJSON* weights = cJSON_CreateArray();
    cJSON* report_json = json_build_report(report);
    size_t index;
    if (!root || !request || !weights || !report_json) goto fail;
    if (json_add_string_copy(root, "schema", "volvoxai.native-cli.failure") != 0 ||
        !cJSON_AddNumberToObject(root, "version", 1) ||
        json_add_string_copy(root, "operation", operation) != 0 ||
        json_add_string_copy(request, "graphPath",
                             paths && paths->graph[0] ? paths->graph : "") != 0 ||
        json_add_optional_string_copy(request, "backend", options ? options->backend : NULL) != 0 ||
        json_add_optional_string_copy(root, "dispatchError", dispatch_error) != 0) {
        goto fail;
    }
    if (options) {
        for (index = 0; index < options->weight_path_count; index++) {
            cJSON* entry = cJSON_CreateString(options->weight_paths[index]);
            if (!entry || !cJSON_AddItemToArray(weights, entry)) {
                cJSON_Delete(entry);
                goto fail;
            }
        }
    }
    if (!cJSON_AddItemToObject(request, "weightPaths", weights) ||
        !cJSON_AddItemToObject(root, "request", request) ||
        !cJSON_AddItemToObject(root, "report", report_json)) {
        goto fail;
    }
    return root;

fail:
    cJSON_Delete(report_json);
    cJSON_Delete(weights);
    cJSON_Delete(request);
    cJSON_Delete(root);
    return NULL;
}

static int write_output_binding(VxCallClient* client, const Binding* binding,
                                const VolvoxaiV1ResultInfo* result_info,
                                int output_row) {
    const VolvoxaiV1TensorInfo* info;
    const char* requested_name;
    const char* suffix;
    char output_name[256];
    uint8_t* payload = NULL;
    int32_t payload_len = 0;
    VolvoxaiV1ReadOutputResponse response;
    size_t name_length;
    size_t offset = 0;
    size_t write_length;
    int result = -1;

    requested_name = binding->name[0] ? binding->name : NULL;
    info = find_tensor_info(result_info->field_outputs.data,
                            result_info->field_outputs.len, requested_name);
    if (!info) {
        fprintf(stderr, "Unknown output tensor: %s\n",
                requested_name ? requested_name : "<first>");
        return -1;
    }
    suffix = dtype_suffix(info->field_dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Output %.*s has dtype %s and requires a %s file: %s\n",
                (int)info->field_name.len,
                info->field_name.data ? (const char*)info->field_name.data : "",
                dtype_name(info->field_dtype), suffix ? suffix : "supported raw",
                binding->path);
        return -1;
    }
    name_length = info->field_name.len;
    if (name_length >= sizeof(output_name)) name_length = sizeof(output_name) - 1u;
    memcpy(output_name, info->field_name.data, name_length);
    output_name[name_length] = 0;

    {
        VolvoxaiV1ReadOutputRequest request;
        volvoxai_v1_read_output_request_init(&request);
        request.field_result_id = result_info->field_result_id;
        if (assign_text(request._allocator, &request.field_name, output_name) != 0) {
            volvoxai_v1_read_output_request_free(&request);
            return -1;
        }
        VX_CALL_MESSAGE(client, VX_RPC_VX_INFERENCE_SERVICE_READ_OUTPUT,
                        volvoxai_v1_read_output_request, &request, payload, payload_len);
        volvoxai_v1_read_output_request_free(&request);
    }
    if (!payload) {
        print_dispatch_error("ReadOutput", client);
        return -1;
    }
    volvoxai_v1_read_output_response_init(&response);
    if (volvoxai_v1_read_output_response_decode(&response, payload,
                                                (size_t)payload_len) != SYNURANG_LITE_OK) {
        fprintf(stderr, "ReadOutput returned an undecodable response.\n");
        goto cleanup;
    }
    if (!report_ok("ReadOutput", response.field_report)) goto cleanup;
    if (!response.field_tensor ||
        response.field_tensor->which_payload != TENSOR_INLINE_PAYLOAD) {
        fprintf(stderr, "ReadOutput did not return an inline tensor payload.\n");
        goto cleanup;
    }
    write_length = response.field_tensor->field_inline.len;
    if (output_row >= 0) {
        if (info->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
            info->field_shape.len < 2 || info->field_shape.data[0] <= 0 ||
            (uint64_t)output_row >= (uint64_t)info->field_shape.data[0] ||
            write_length % (size_t)info->field_shape.data[0] != 0) {
            fprintf(stderr, "--row %d is invalid for output %s.\n",
                    output_row, output_name);
            goto cleanup;
        }
        write_length /= (size_t)info->field_shape.data[0];
        offset = (size_t)output_row * write_length;
    }
    result = write_bytes(binding->path,
                         response.field_tensor->field_inline.data + offset,
                         write_length);

cleanup:
    volvoxai_v1_read_output_response_free(&response);
    vx_call_free(client, payload);
    return result;
}

static void print_release_info(void) {
    printf("%scommit %s, built %s)\n", vx_release_version_prefix,
           VOLVOXAI_GIT_COMMIT, VOLVOXAI_BUILD_DATE);
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
    printf("The fixed CLI uses only the generated public proto API.\n");
}

static void print_run_help(const char* argv0) {
    printf("Usage: %s run <model-dir|graph.json> [options]\n\n", argv0);
    printf("Options:\n");
    printf("  --weights <file>             Add a safetensors weight shard (repeatable).\n");
    printf("  --input <name[shape]=file>   Bind one exact typed input; shape is required for dynamic axes.\n");
    printf("  --output <name=file|file>    Write exact typed raw output data.\n");
    printf("  --row <index>                Write one row from each selected F32 output.\n");
    printf("  --report-json <file>         Write machine-readable success or failure reports.\n");
    printf("  --threads <n>                Set the CPU worker count.\n");
    printf("  --cpu | --vulkan | --opengl | --metal | --cuda\n");
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
    printf("  --cpu | --vulkan | --opengl | --metal | --cuda\n");
}
#endif

static int command_run(VxCallClient* client, int argc, char** argv) {
    RunOptions options;
    ModelPaths paths;
    uint8_t* encoded = NULL;
    size_t encoded_len = 0u;
    uint8_t* policy = NULL;
    size_t policy_len = 0u;
    uint8_t* payload = NULL;
    int32_t payload_len = 0;
    VolvoxaiV1RuntimeHandle runtime = {0};
    VolvoxaiV1ModelHandle model = {0};
    VolvoxaiV1CompiledModelHandle compiled = {0};
    VolvoxaiV1ExecutionContextHandle context = {0};
    VolvoxaiV1ExecutionResultHandle execution = {0};
    VolvoxaiV1ResultInfo result_info = {0};
    int have_runtime = 0;
    int have_model = 0;
    int have_compiled = 0;
    int have_context = 0;
    int have_execution = 0;
    int have_result_info = 0;
    int return_code = 1;
    size_t index;
    char dispatch_error[512] = {0};
    const char* failed_operation = NULL;
    const VolvoxaiV1OperationReport* failed_report = NULL;

    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_run_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    memset(&options, 0, sizeof(options));
    options.output_row = -1;
    for (index = 3; index < (size_t)argc; index++) {
        int option_index = (int)index;
        int parsed = parse_run_option(argc, argv, &option_index, &options);
        if (parsed <= 0) {
            fprintf(stderr, "Invalid run options. Use '%s run --help'.\n", argv[0]);
            return 2;
        }
        index = (size_t)option_index;
    }
    if (resolve_model_paths(argv[2], &paths) != 0) return 1;
    if (!options.weight_path_count && paths.default_weights[0]) {
        options.weight_paths[options.weight_path_count++] = paths.default_weights;
    }
    if (options.input_count == 0) {
        fprintf(stderr, "Pass one --input for each declared model input.\n");
        return 2;
    }

    if (encode_create_runtime_request(&options, VOLVOXAI_V1_EXECUTION_MODE_DIRECT,
                                      &encoded, &encoded_len) != 0) {
        fprintf(stderr, "Cannot encode CreateRuntime request.\n");
        return 1;
    }
    failed_operation = "CreateRuntime";
    payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME, encoded, (int32_t)encoded_len, &payload_len);
    synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, encoded);
    encoded = NULL;
    encoded_len = 0u;
    if (!payload) {
        copy_last_error_text(client, dispatch_error, sizeof(dispatch_error));
        print_dispatch_error(failed_operation, client);
        goto cleanup;
    }
    volvoxai_v1_runtime_handle_init(&runtime);
    if (volvoxai_v1_runtime_handle_decode(&runtime, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "CreateRuntime returned an undecodable response.\n");
        goto cleanup;
    }
    have_runtime = 1;
    failed_report = runtime.field_report;
    if (!report_ok(failed_operation, runtime.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    if (encode_load_model_request(runtime.field_runtime_id, &paths, &options,
                                  &encoded, &encoded_len) != 0) {
        fprintf(stderr, "Cannot encode LoadModel request.\n");
        goto cleanup;
    }
    failed_operation = "LoadModel";
    payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL, encoded, (int32_t)encoded_len, &payload_len);
    synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, encoded);
    encoded = NULL;
    encoded_len = 0u;
    if (!payload) {
        copy_last_error_text(client, dispatch_error, sizeof(dispatch_error));
        print_dispatch_error(failed_operation, client);
        goto cleanup;
    }
    volvoxai_v1_model_handle_init(&model);
    if (volvoxai_v1_model_handle_decode(&model, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "LoadModel returned an undecodable response.\n");
        goto cleanup;
    }
    have_model = 1;
    failed_report = model.field_report;
    if (!report_ok(failed_operation, model.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    if (encode_compile_model_request(model.field_model_id, options.backend, &policy, &policy_len) != 0) {
        fprintf(stderr, "Cannot encode CompileModel policy.\n");
        goto cleanup;
    }
    failed_operation = "CompileModel";
    payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL,
                            policy, policy_len, &payload_len);
    if (policy) {
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, policy);
        policy = NULL;
    }
    policy_len = 0u;
    if (!payload) {
        copy_last_error_text(client, dispatch_error, sizeof(dispatch_error));
        print_dispatch_error(failed_operation, client);
        goto cleanup;
    }
    volvoxai_v1_compiled_model_handle_init(&compiled);
    if (volvoxai_v1_compiled_model_handle_decode(&compiled, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "CompileModel returned an undecodable response.\n");
        goto cleanup;
    }
    have_compiled = 1;
    failed_report = compiled.field_report;
    if (!report_ok(failed_operation, compiled.field_report)) goto cleanup;
    if (options.debug) print_debug_route(compiled.field_report, "compile", ((double)compiled.field_compile_time_ns / 1e6));
    vx_call_free(client, payload);
    payload = NULL;

    failed_operation = "CreateExecutionContext";
    {
        VolvoxaiV1CreateExecutionContextRequest request;
        volvoxai_v1_create_execution_context_request_init(&request);
        request.field_compiled_model_id = compiled.field_compiled_model_id;
        if (volvoxai_v1_create_execution_context_request_encode(
                &request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
            volvoxai_v1_create_execution_context_request_free(&request);
            fprintf(stderr, "Cannot encode CreateExecutionContext request.\n");
            goto cleanup;
        }
        volvoxai_v1_create_execution_context_request_free(&request);
        payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_EXECUTION_CONTEXT,
            encoded, (int32_t)encoded_len, &payload_len);
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, encoded);
        encoded = NULL;
        encoded_len = 0u;
    }
    if (!payload) {
        copy_last_error_text(client, dispatch_error, sizeof(dispatch_error));
        print_dispatch_error(failed_operation, client);
        goto cleanup;
    }
    volvoxai_v1_execution_context_handle_init(&context);
    if (volvoxai_v1_execution_context_handle_decode(&context, payload,
                                                    (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "CreateExecutionContext returned an undecodable response.\n");
        goto cleanup;
    }
    have_context = 1;
    failed_report = context.field_report;
    if (!report_ok(failed_operation, context.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    if (options.input_count != context.field_inputs.len) {
        fprintf(stderr,
                "Execution requires one atomic binding for each of the %zu graph inputs; got %zu.\n",
                context.field_inputs.len, options.input_count);
        return_code = 2;
        goto cleanup;
    }

    {
        VolvoxaiV1ExecuteRequest request;
        volvoxai_v1_execute_request_init(&request);
        request.field_context_id = context.field_context_id;
        for (index = 0; index < options.input_count; index++) {
            const VolvoxaiV1TensorSpec* spec = find_tensor_spec(
                context.field_inputs.data, context.field_inputs.len,
                options.inputs[index].name);
            VolvoxaiV1Tensor* tensor;
            if (!spec) {
                fprintf(stderr, "Unknown input tensor: %s\n", options.inputs[index].name);
                volvoxai_v1_execute_request_free(&request);
                return_code = 2;
                goto cleanup;
            }
            tensor = volvoxai_v1_execute_request_add_inputs(&request);
            if (!tensor ||
                fill_tensor_from_binding(&options.inputs[index], spec, tensor,
                                         request._allocator) != 0) {
                volvoxai_v1_execute_request_free(&request);
                goto cleanup;
            }
        }
        if (volvoxai_v1_execute_request_encode(&request, &encoded, &encoded_len) !=
            SYNURANG_LITE_OK) {
            volvoxai_v1_execute_request_free(&request);
            fprintf(stderr, "Cannot encode Execute request.\n");
            goto cleanup;
        }
        volvoxai_v1_execute_request_free(&request);
    }

    failed_operation = "Execute";
    payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_EXECUTE, encoded, (int32_t)encoded_len, &payload_len);
    synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, encoded);
    encoded = NULL;
    encoded_len = 0u;
    if (!payload) {
        copy_last_error_text(client, dispatch_error, sizeof(dispatch_error));
        print_dispatch_error(failed_operation, client);
        goto cleanup;
    }
    volvoxai_v1_execution_result_handle_init(&execution);
    if (volvoxai_v1_execution_result_handle_decode(&execution, payload,
                                                   (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "Execute returned an undecodable response.\n");
        goto cleanup;
    }
    have_execution = 1;
    failed_report = execution.field_report;
    if (!report_ok(failed_operation, execution.field_report)) goto cleanup;
    if (options.debug) print_debug_route(execution.field_report, "execute", execution.field_metrics ? ((double)execution.field_metrics->field_host_time_ns / 1e6) : 0);
    vx_call_free(client, payload);
    payload = NULL;

    failed_operation = "GetResult";
    {
        VolvoxaiV1ResultRef request;
        volvoxai_v1_result_ref_init(&request);
        request.field_result_id = execution.field_result_id;
        VX_CALL_MESSAGE(client, VX_RPC_VX_INFERENCE_SERVICE_GET_RESULT,
                        volvoxai_v1_result_ref, &request, payload, payload_len);
        volvoxai_v1_result_ref_free(&request);
    }
    if (!payload) {
        copy_last_error_text(client, dispatch_error, sizeof(dispatch_error));
        print_dispatch_error(failed_operation, client);
        goto cleanup;
    }
    volvoxai_v1_result_info_init(&result_info);
    if (volvoxai_v1_result_info_decode(&result_info, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "GetResult returned an undecodable response.\n");
        goto cleanup;
    }
    have_result_info = 1;
    failed_report = result_info.field_report;
    if (!report_ok(failed_operation, result_info.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    for (index = 0; index < options.output_count; index++) {
        if (write_output_binding(client, &options.outputs[index], &result_info,
                                 options.output_row) != 0) {
            goto cleanup;
        }
    }

    if (options.report_json) {
        cJSON* root = json_build_run_success(&paths, &options,
                                             compiled.field_report,
                                             &execution, &result_info);
        if (!root || write_report_json(options.report_json, root) != 0) {
            cJSON_Delete(root);
            fprintf(stderr, "Cannot write report JSON: %s\n", options.report_json);
            goto cleanup;
        }
        cJSON_Delete(root);
    }
    return_code = 0;

cleanup:
    if (payload) vx_call_free(client, payload);
    if (encoded) {
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, encoded);
    }
    if (policy) {
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, policy);
    }
    if (return_code != 0 && options.report_json) {
        cJSON* root = json_build_failure(&paths, &options, failed_operation ? failed_operation : "run",
                                         failed_report, dispatch_error);
        if (root) {
            if (write_report_json(options.report_json, root) != 0) {
                fprintf(stderr, "Cannot write report JSON: %s\n", options.report_json);
            }
            cJSON_Delete(root);
        }
    }
    if (have_result_info) volvoxai_v1_result_info_free(&result_info);
    if (have_execution) volvoxai_v1_execution_result_handle_free(&execution);
    if (have_context) volvoxai_v1_execution_context_handle_free(&context);
    if (have_compiled) volvoxai_v1_compiled_model_handle_free(&compiled);
    if (have_model) volvoxai_v1_model_handle_free(&model);
    if (have_runtime) volvoxai_v1_runtime_handle_free(&runtime);
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

static int trainable_duplicate(const TrainOptions* options, const char* name) {
    size_t index;
    for (index = 0; index < options->trainable_count; index++) {
        if (!strcmp(options->trainable_names[index], name)) return 1;
    }
    return 0;
}

static int encode_optimizer(VolvoxaiV1TrainStepRequest* request,
                            const TrainOptions* options) {
    VolvoxaiV1TrainerOptimizerOptions* optimizer;
    optimizer = (VolvoxaiV1TrainerOptimizerOptions*)request->_allocator->allocate(
        request->_allocator->context, sizeof(*optimizer));
    if (!optimizer) return -1;
    volvoxai_v1_trainer_optimizer_options_init_with_allocator(optimizer,
                                                              request->_allocator);
    optimizer->field_kind = options->optimizer_kind;
    optimizer->has_kind = 1;
    optimizer->has_learning_rate = options->has_learning_rate;
    optimizer->field_learning_rate = options->learning_rate;
    optimizer->has_beta1 = options->has_beta1;
    optimizer->field_beta1 = options->beta1;
    optimizer->has_beta2 = options->has_beta2;
    optimizer->field_beta2 = options->beta2;
    optimizer->has_epsilon = options->has_epsilon;
    optimizer->field_epsilon = options->epsilon;
    optimizer->has_weight_decay = options->has_weight_decay;
    optimizer->field_weight_decay = options->weight_decay;
    optimizer->has_max_gradient_norm = options->has_max_gradient_norm;
    optimizer->field_max_gradient_norm = options->max_gradient_norm;
    request->field_optimizer = optimizer;
    return 0;
}

static int command_train(VxCallClient* client, int argc, char** argv) {
    TrainOptions options;
    ModelPaths paths;
    uint8_t* encoded = NULL;
    size_t encoded_len = 0u;
    uint8_t* payload = NULL;
    int32_t payload_len = 0;
    void* targets_storage = NULL;
    size_t targets_bytes = 0u;
    const int32_t* targets = NULL;
    size_t target_count = 0u;
    VolvoxaiV1RuntimeHandle runtime = {0};
    VolvoxaiV1ModelHandle model = {0};
    VolvoxaiV1TrainerHandle trainer = {0};
    VolvoxaiV1TrainStepResult step = {0};
    VolvoxaiV1RevisionInfo published = {0};
    int have_runtime = 0;
    int have_model = 0;
    int have_trainer = 0;
    int have_step = 0;
    int have_published = 0;
    int return_code = 1;
    size_t index;

    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_train_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    memset(&options, 0, sizeof(options));
    options.run.output_row = -1;
    options.microbatches = 1;
    options.accumulation_steps = 1u;
    options.optimizer_kind = VOLVOXAI_V1_TRAINING_OPTIMIZER_KIND_SGD;
    options.ignore_index = -100;
    options.loss_row = -1;

    for (index = 3; index < (size_t)argc; index++) {
        const char* argument = argv[index];
        if (!strcmp(argument, "--targets")) {
            if (++index >= (size_t)argc || options.targets_path) goto usage_error;
            options.targets_path = argv[index];
        } else if (!strcmp(argument, "--logits")) {
            if (++index >= (size_t)argc || options.logits_name) goto usage_error;
            options.logits_name = argv[index];
        } else if (!strcmp(argument, "--trainable")) {
            const char* name;
            if (++index >= (size_t)argc || options.trainable_count == MAX_BINDINGS) {
                goto usage_error;
            }
            name = argv[index];
            if (!name[0] || trainable_duplicate(&options, name)) goto usage_error;
            options.trainable_names[options.trainable_count++] = name;
        } else if (!strcmp(argument, "--output-weights")) {
            if (++index >= (size_t)argc ||
                options.output_weight_path_count == MAX_WEIGHT_PATHS) {
                goto usage_error;
            }
            options.output_weight_paths[options.output_weight_path_count++] = argv[index];
        } else if (!strcmp(argument, "--microbatches")) {
            int parsed;
            if (++index >= (size_t)argc ||
                parse_nonnegative_int(argv[index], 0, &parsed) != 0 || !parsed) {
                goto usage_error;
            }
            options.microbatches = parsed;
        } else if (!strcmp(argument, "--accumulation-steps")) {
            int parsed;
            if (++index >= (size_t)argc ||
                parse_nonnegative_int(argv[index], 0, &parsed) != 0 || !parsed) {
                goto usage_error;
            }
            options.accumulation_steps = (uint32_t)parsed;
        } else if (!strcmp(argument, "--optimizer")) {
            if (++index >= (size_t)argc) goto usage_error;
            if (!strcmp(argv[index], "sgd")) {
                options.optimizer_kind = VOLVOXAI_V1_TRAINING_OPTIMIZER_KIND_SGD;
            } else if (!strcmp(argv[index], "adamw")) {
                options.optimizer_kind = VOLVOXAI_V1_TRAINING_OPTIMIZER_KIND_ADAMW;
            } else {
                goto usage_error;
            }
        } else if (!strcmp(argument, "--learning-rate")) {
            if (++index >= (size_t)argc ||
                parse_train_float(argv[index], &options.learning_rate) != 0) {
                goto usage_error;
            }
            options.has_learning_rate = 1;
        } else if (!strcmp(argument, "--beta1")) {
            if (++index >= (size_t)argc ||
                parse_train_float(argv[index], &options.beta1) != 0) {
                goto usage_error;
            }
            options.has_beta1 = 1;
        } else if (!strcmp(argument, "--beta2")) {
            if (++index >= (size_t)argc ||
                parse_train_float(argv[index], &options.beta2) != 0) {
                goto usage_error;
            }
            options.has_beta2 = 1;
        } else if (!strcmp(argument, "--epsilon")) {
            if (++index >= (size_t)argc ||
                parse_train_float(argv[index], &options.epsilon) != 0) {
                goto usage_error;
            }
            options.has_epsilon = 1;
        } else if (!strcmp(argument, "--weight-decay")) {
            if (++index >= (size_t)argc ||
                parse_train_float(argv[index], &options.weight_decay) != 0) {
                goto usage_error;
            }
            options.has_weight_decay = 1;
        } else if (!strcmp(argument, "--max-gradient-norm")) {
            if (++index >= (size_t)argc ||
                parse_train_float(argv[index], &options.max_gradient_norm) != 0) {
                goto usage_error;
            }
            options.has_max_gradient_norm = 1;
        } else if (!strcmp(argument, "--ignore-index")) {
            if (++index >= (size_t)argc ||
                parse_train_i32(argv[index], &options.ignore_index) != 0) {
                goto usage_error;
            }
        } else if (!strcmp(argument, "--row")) {
            if (++index >= (size_t)argc ||
                parse_train_i32(argv[index], &options.loss_row) != 0 ||
                options.loss_row < -1) {
                goto usage_error;
            }
        } else {
            int option_index = (int)index;
            int parsed = parse_run_option(argc, argv, &option_index, &options.run);
            if (parsed <= 0) goto usage_error;
            index = (size_t)option_index;
        }
    }

    if (!options.targets_path || !options.logits_name ||
        !options.trainable_count || !options.output_weight_path_count ||
        options.run.output_count || options.run.output_row != -1 ||
        options.run.report_json ||
        (options.has_learning_rate && options.learning_rate < 0.0f) ||
        (options.has_beta1 && (options.beta1 < 0.0f || options.beta1 >= 1.0f)) ||
        (options.has_beta2 && (options.beta2 < 0.0f || options.beta2 >= 1.0f)) ||
        (options.has_epsilon && options.epsilon <= 0.0f) ||
        (options.has_weight_decay && options.weight_decay < 0.0f) ||
        (options.has_max_gradient_norm && options.max_gradient_norm < 0.0f)) {
        goto usage_error;
    }
    if (!has_suffix(options.targets_path, ".i32") ||
        read_whole_file(options.targets_path, &targets_storage, &targets_bytes) != 0 ||
        !targets_bytes || targets_bytes % sizeof(int32_t) != 0) {
        fprintf(stderr, "--targets must be a nonempty raw .i32 file.\n");
        goto cleanup;
    }
    targets = (const int32_t*)targets_storage;
    target_count = targets_bytes / sizeof(int32_t);
    if (resolve_model_paths(argv[2], &paths) != 0) goto cleanup;
    if (!options.run.weight_path_count && paths.default_weights[0]) {
        options.run.weight_paths[options.run.weight_path_count++] = paths.default_weights;
    }
    if (options.output_weight_path_count != options.run.weight_path_count) {
        fprintf(stderr,
                "--output-weights count must match the %zu input weight shard(s).\n",
                options.run.weight_path_count);
        return_code = 2;
        goto cleanup;
    }

    if (encode_create_runtime_request(&options.run, VOLVOXAI_V1_EXECUTION_MODE_DIRECT,
                                      &encoded, &encoded_len) != 0) {
        fprintf(stderr, "Cannot encode CreateRuntime request.\n");
        goto cleanup;
    }
    payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME, encoded, (int32_t)encoded_len, &payload_len);
    synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, encoded);
    encoded = NULL;
    encoded_len = 0u;
    if (!payload) {
        print_dispatch_error("CreateRuntime", client);
        goto cleanup;
    }
    volvoxai_v1_runtime_handle_init(&runtime);
    if (volvoxai_v1_runtime_handle_decode(&runtime, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "CreateRuntime returned an undecodable response.\n");
        goto cleanup;
    }
    have_runtime = 1;
    if (!report_ok("CreateRuntime", runtime.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    if (encode_load_model_request(runtime.field_runtime_id, &paths, &options.run,
                                  &encoded, &encoded_len) != 0) {
        fprintf(stderr, "Cannot encode LoadModel request.\n");
        goto cleanup;
    }
    payload = vx_call_bytes(client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL, encoded, (int32_t)encoded_len, &payload_len);
    synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, encoded);
    encoded = NULL;
    encoded_len = 0u;
    if (!payload) {
        print_dispatch_error("LoadModel", client);
        goto cleanup;
    }
    volvoxai_v1_model_handle_init(&model);
    if (volvoxai_v1_model_handle_decode(&model, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "LoadModel returned an undecodable response.\n");
        goto cleanup;
    }
    have_model = 1;
    if (!report_ok("LoadModel", model.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    {
        VolvoxaiV1CreateTrainerRequest request;
        volvoxai_v1_create_trainer_request_init(&request);
        request.field_model_id = model.field_model_id;
        if (assign_text(request._allocator, &request.field_backend, options.run.backend) != 0) {
            volvoxai_v1_create_trainer_request_free(&request);
            goto cleanup;
        }
        VX_CALL_MESSAGE(client, VX_RPC_VX_TRAINING_SERVICE_CREATE_TRAINER,
                        volvoxai_v1_create_trainer_request, &request, payload, payload_len);
        volvoxai_v1_create_trainer_request_free(&request);
    }
    if (!payload) {
        print_dispatch_error("CreateTrainer", client);
        goto cleanup;
    }
    volvoxai_v1_trainer_handle_init(&trainer);
    if (volvoxai_v1_trainer_handle_decode(&trainer, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "CreateTrainer returned an undecodable response.\n");
        goto cleanup;
    }
    have_trainer = 1;
    if (!report_ok("CreateTrainer", trainer.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    if (options.run.input_count != trainer.field_inputs.len) {
        fprintf(stderr,
                "Training requires one atomic shaped binding for each of the %zu graph inputs; got %zu.\n",
                trainer.field_inputs.len, options.run.input_count);
        return_code = 2;
        goto cleanup;
    }

    printf("Training backend=%s optimizer=%s microbatches=%ld accumulation=%u\n",
           options.run.backend ? options.run.backend : "cpu",
           options.optimizer_kind == VOLVOXAI_V1_TRAINING_OPTIMIZER_KIND_ADAMW
               ? "adamw"
               : "sgd",
           options.microbatches, options.accumulation_steps);

    for (index = 0; index < (size_t)options.microbatches; index++) {
        VolvoxaiV1TrainStepRequest request;
        VolvoxaiV1CrossEntropyLoss* loss;
        size_t input_index;
        size_t trainable_index;
        volvoxai_v1_train_step_request_init(&request);
        request.field_trainer_id = trainer.field_trainer_id;
        request.has_accumulation_steps = 1;
        request.field_accumulation_steps = options.accumulation_steps;
        request.field_flush_accumulation = (index + 1u == (size_t)options.microbatches);
        request.field_reset_accumulation = 0;
        for (input_index = 0; input_index < options.run.input_count; input_index++) {
            const VolvoxaiV1TensorSpec* spec = find_tensor_spec(
                trainer.field_inputs.data, trainer.field_inputs.len,
                options.run.inputs[input_index].name);
            VolvoxaiV1Tensor* tensor;
            if (!spec) {
                fprintf(stderr, "Unknown training input tensor: %s\n",
                        options.run.inputs[input_index].name);
                volvoxai_v1_train_step_request_free(&request);
                return_code = 2;
                goto cleanup;
            }
            tensor = volvoxai_v1_train_step_request_add_inputs(&request);
            if (!tensor ||
                fill_tensor_from_binding(&options.run.inputs[input_index], spec, tensor,
                                         request._allocator) != 0) {
                volvoxai_v1_train_step_request_free(&request);
                goto cleanup;
            }
        }
        loss = volvoxai_v1_train_step_request_add_losses(&request);
        if (!loss) {
            volvoxai_v1_train_step_request_free(&request);
            fprintf(stderr, "Cannot allocate training loss request.\n");
            goto cleanup;
        }
        volvoxai_v1_cross_entropy_loss_init_with_allocator(loss, request._allocator);
        if (assign_text(request._allocator, &loss->field_name, "cross_entropy") != 0 ||
            assign_text(request._allocator, &loss->field_logits_name,
                        options.logits_name) != 0) {
            volvoxai_v1_train_step_request_free(&request);
            fprintf(stderr, "Cannot encode loss names.\n");
            goto cleanup;
        }
        loss->field_targets = request._allocator->allocate(request._allocator->context, sizeof(*loss->field_targets));
        if (!loss->field_targets) {
            volvoxai_v1_train_step_request_free(&request);
            goto cleanup;
        }
        volvoxai_v1_tensor_init_with_allocator(loss->field_targets, request._allocator);
        loss->field_targets->field_dtype = VOLVOXAI_V1_DATA_TYPE_I32;
        loss->field_targets->which_payload = 5;
        int64_t* target_axis = volvoxai_v1_tensor_add_shape(loss->field_targets);
        if (!target_axis || synurang_lite_bytes_assign(request._allocator,
            &loss->field_targets->field_inline, targets, targets_bytes) != SYNURANG_LITE_OK) {
            volvoxai_v1_train_step_request_free(&request);
            goto cleanup;
        }
        *target_axis = (int64_t)target_count;
        loss->has_ignore_index = 1;
        loss->field_ignore_index = options.ignore_index;
        loss->has_row_index = 1;
        loss->field_row_index = options.loss_row;
        if (options.accumulation_steps > 1u) {
            loss->field_normalizer =
                (float)target_count * (float)options.accumulation_steps;
        }
        for (trainable_index = 0; trainable_index < options.trainable_count;
             trainable_index++) {
            SynurangLiteBytes* slot =
                volvoxai_v1_train_step_request_add_trainable_names(&request);
            if (!slot ||
                assign_text(request._allocator, slot,
                            options.trainable_names[trainable_index]) != 0) {
                volvoxai_v1_train_step_request_free(&request);
                fprintf(stderr, "Cannot encode trainable names.\n");
                goto cleanup;
            }
        }
        if (encode_optimizer(&request, &options) != 0) {
            volvoxai_v1_train_step_request_free(&request);
            fprintf(stderr, "Cannot encode optimizer options.\n");
            goto cleanup;
        }
        if (volvoxai_v1_train_step_request_encode(&request, &encoded, &encoded_len) !=
            SYNURANG_LITE_OK) {
            volvoxai_v1_train_step_request_free(&request);
            fprintf(stderr, "Cannot encode TrainStep request.\n");
            goto cleanup;
        }
        volvoxai_v1_train_step_request_free(&request);

        payload = vx_call_bytes(client, VX_RPC_VX_TRAINING_SERVICE_TRAIN_STEP, encoded, (int32_t)encoded_len, &payload_len);
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, encoded);
        encoded = NULL;
        encoded_len = 0u;
        if (!payload) {
            print_dispatch_error("TrainStep", client);
            goto cleanup;
        }
        if (have_step) {
            volvoxai_v1_train_step_result_free(&step);
            have_step = 0;
        }
        volvoxai_v1_train_step_result_init(&step);
        if (volvoxai_v1_train_step_result_decode(&step, payload, (size_t)payload_len) !=
            SYNURANG_LITE_OK) {
            fprintf(stderr, "TrainStep returned an undecodable response.\n");
            goto cleanup;
        }
        have_step = 1;
        if (!report_ok("TrainStep", step.field_report)) goto cleanup;
        printf("microbatch=%llu optimizer_step=%llu loss=%.8g accumulated=%u update=%s backend=%.*s\n",
               (unsigned long long)step.field_microbatch_id,
               (unsigned long long)step.field_optimizer_step,
               step.field_loss, step.field_accumulated_microbatches,
               step.field_update_applied ? "yes" : "no",
               (int)step.field_backend.len,
               step.field_backend.data ? (const char*)step.field_backend.data : "");
        vx_call_free(client, payload);
        payload = NULL;
    }

    {
        VolvoxaiV1TrainerRef request;
        volvoxai_v1_trainer_ref_init(&request);
        request.field_trainer_id = trainer.field_trainer_id;
        VX_CALL_MESSAGE(client, VX_RPC_VX_TRAINING_SERVICE_COMMIT_TRAINER,
                        volvoxai_v1_trainer_ref, &request, payload, payload_len);
        volvoxai_v1_trainer_ref_free(&request);
    }
    if (!payload) {
        print_dispatch_error("CommitTrainer", client);
        goto cleanup;
    }
    volvoxai_v1_revision_info_init(&published);
    if (volvoxai_v1_revision_info_decode(&published, payload, (size_t)payload_len) !=
        SYNURANG_LITE_OK) {
        fprintf(stderr, "CommitTrainer returned an undecodable response.\n");
        goto cleanup;
    }
    have_published = 1;
    if (!report_ok("CommitTrainer", published.field_report)) goto cleanup;
    vx_call_free(client, payload);
    payload = NULL;

    {
        VolvoxaiV1ExportTrainerWeightsRequest request;
        volvoxai_v1_export_trainer_weights_request_init(&request);
        request.field_trainer_id = trainer.field_trainer_id;
        for (index = 0; index < options.output_weight_path_count; index++) {
            SynurangLiteBytes* slot =
                volvoxai_v1_export_trainer_weights_request_add_output_paths(&request);
            if (!slot ||
                assign_text(request._allocator, slot,
                            options.output_weight_paths[index]) != 0) {
                volvoxai_v1_export_trainer_weights_request_free(&request);
                fprintf(stderr, "Cannot encode ExportTrainerWeights request.\n");
                goto cleanup;
            }
        }
        if (volvoxai_v1_export_trainer_weights_request_encode(
                &request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
            volvoxai_v1_export_trainer_weights_request_free(&request);
            fprintf(stderr, "Cannot encode ExportTrainerWeights request.\n");
            goto cleanup;
        }
        volvoxai_v1_export_trainer_weights_request_free(&request);
    }
    payload = vx_call_bytes(client, VX_RPC_VX_TRAINING_SERVICE_EXPORT_TRAINER_WEIGHTS, encoded, (int32_t)encoded_len,
                                                    &payload_len);
    synurang_lite_default_allocator()->deallocate(
        synurang_lite_default_allocator()->context, encoded);
    encoded = NULL;
    encoded_len = 0u;
    if (!payload) {
        print_dispatch_error("ExportTrainerWeights", client);
        goto cleanup;
    }
    {
        VolvoxaiV1TrainerWeights weights;
        volvoxai_v1_trainer_weights_init(&weights);
        if (volvoxai_v1_trainer_weights_decode(&weights, payload, (size_t)payload_len) !=
            SYNURANG_LITE_OK) {
            volvoxai_v1_trainer_weights_free(&weights);
            fprintf(stderr, "ExportTrainerWeights returned an undecodable response.\n");
            goto cleanup;
        }
        if (!report_ok("ExportTrainerWeights", weights.field_report)) {
            volvoxai_v1_trainer_weights_free(&weights);
            goto cleanup;
        }
        volvoxai_v1_trainer_weights_free(&weights);
    }
    vx_call_free(client, payload);
    payload = NULL;

    printf("Published weight revision %llu and exported %zu shard(s).\n",
           (unsigned long long)published.field_weight_revision,
           options.output_weight_path_count);
    return_code = 0;
    goto cleanup;

usage_error:
    fprintf(stderr, "Invalid train options. Use '%s train --help'.\n", argv[0]);
    return_code = 2;

cleanup:
    if (payload) {
        if (have_trainer || have_step || have_published) {
            vx_call_free(client, payload);
        } else {
            vx_call_free(client, payload);
        }
    }
    if (encoded) {
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, encoded);
    }
    free(targets_storage);
    if (have_published) volvoxai_v1_revision_info_free(&published);
    if (have_step) volvoxai_v1_train_step_result_free(&step);
    if (have_trainer) volvoxai_v1_trainer_handle_free(&trainer);
    if (have_model) volvoxai_v1_model_handle_free(&model);
    if (have_runtime) volvoxai_v1_runtime_handle_free(&runtime);
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
    if (!strcmp(argv[1], "run")) {
        VxCallClient client;
        if (!vx_call_client_open(&client)) return 1;
        int result = command_run(&client, argc, argv);
        vx_call_client_close(&client);
        return result;
    }
#if VOLVOXAI_ENABLE_TRAINING
    if (!strcmp(argv[1], "train")) {
        VxCallClient client;
        if (!vx_call_client_open(&client)) return 1;
        int result = command_train(&client, argc, argv);
        vx_call_client_close(&client);
        return result;
    }
#endif
    fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
    print_root_help(argv[0]);
    return 2;
}
