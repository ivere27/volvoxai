#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "../../native/cli/call_client.h"
#include "../../native/src/generated/proto_methods.h"
#include "volvoxai_lite.h"

#include "image_io.h"

#include <errno.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

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
#define TASK_MAX_TENSOR_RANK 8u
#define TENSOR_INLINE_PAYLOAD 5

typedef struct Binding {
    char name[128];
    char path[PATH_MAX];
    int64_t shape[TASK_MAX_TENSOR_RANK];
    uint32_t rank;
    int has_shape;
} Binding;

typedef struct ModelPaths {
    char graph[PATH_MAX];
    char default_weights[PATH_MAX];
} ModelPaths;

typedef struct TaskOptions {
    const char* weight_paths[MAX_WEIGHT_PATHS];
    size_t weight_path_count;
    Binding inputs[MAX_BINDINGS];
    Binding images[MAX_BINDINGS];
    Binding outputs[MAX_BINDINGS];
    size_t input_count;
    size_t image_count;
    size_t output_count;
    const char* backend;
    int debug;
    int cpu_threads;
    int image_normalization;
    int warmup_runs;
    int timed_runs;
    int include_transfers;
} TaskOptions;

typedef struct PreparedInput {
    char name[128];
    VolvoxaiV1DataType dtype;
    int64_t shape[TASK_MAX_TENSOR_RANK];
    size_t rank;
    void* data;
    size_t byte_size;
} PreparedInput;

typedef struct TaskSession {
    VxCallClient client;
    int64_t runtime_id;
    int64_t model_id;
    int64_t compiled_model_id;
    int64_t context_id;
    int64_t result_id;
    VolvoxaiV1ExecutionContextHandle context;
    int context_initialized;
    VolvoxaiV1ResultInfo result_info;
    int result_info_initialized;
    PreparedInput inputs[MAX_BINDINGS];
    size_t input_count;
    uint8_t* input_request;
    size_t input_request_len;
    int32_t prefill_position;
    int decode;
    int debug;
} TaskSession;

typedef struct LabelList {
    char* storage;
    char** items;
    size_t count;
} LabelList;

typedef struct RankedValue {
    size_t index;
    int class_id;
    float score;
} RankedValue;

static double now_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static int file_exists(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

static int is_directory(const char* path) {
    struct stat st;
    return path && path[0] && stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

static int copy_string(char* destination, size_t capacity, const char* value) {
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
    } else if (copy_string(paths->graph, sizeof(paths->graph), model) != 0) {
        fprintf(stderr, "Graph path is too long: %s\n", model);
        return -1;
    }
    if (!file_exists(paths->graph)) {
        fprintf(stderr, "Missing graph.json: %s\n", paths->graph);
        return -1;
    }
    return 0;
}

static const char* discover_file(const char* model, const char* explicit_path,
                                 const char* leaf, char* buffer, size_t capacity) {
    if (explicit_path) return explicit_path;
    if (!is_directory(model) || join_path(buffer, capacity, model, leaf) != 0)
        return NULL;
    return file_exists(buffer) ? buffer : NULL;
}

static int parse_binding(const char* argument, Binding* binding, int require_name) {
    const char* equals;
    const char* shape_open = NULL;
    size_t name_length;
    if (!argument || !binding) return -1;
    memset(binding, 0, sizeof(*binding));
    equals = strchr(argument, '=');
    if (!equals) {
        if (require_name || copy_string(binding->path, sizeof(binding->path), argument) != 0)
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
            unsigned long long extent;
            if (binding->rank == TASK_MAX_TENSOR_RANK ||
                *cursor < '1' || *cursor > '9') return -1;
            errno = 0;
            extent = strtoull(cursor, &parsed_end, 10);
            if (errno == ERANGE || !parsed_end || parsed_end == cursor ||
                extent > (unsigned long long)INT64_MAX ||
                parsed_end > shape_end ||
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
    return copy_string(binding->path, sizeof(binding->path), equals + 1);
}

static int parse_int(const char* text, int minimum, int* result) {
    char* end = NULL;
    long value;
    if (!text || !text[0] || !result) return -1;
    errno = 0;
    value = strtol(text, &end, 10);
    if (errno == ERANGE || !end || *end || value < minimum || value > INT_MAX)
        return -1;
    *result = (int)value;
    return 0;
}

static int parse_normalization(const char* value, int* mode) {
    if (!strcmp(value, "zero-one")) *mode = VOLVOX_IMAGE_ZERO_ONE;
    else if (!strcmp(value, "minus-one-one")) *mode = VOLVOX_IMAGE_MINUS_ONE_ONE;
    else if (!strcmp(value, "raw-255")) *mode = VOLVOX_IMAGE_RAW_255;
    else return -1;
    return 0;
}

static int select_backend(TaskOptions* options, const char* backend) {
    if (options->backend && strcmp(options->backend, backend)) {
        fprintf(stderr, "Pass at most one accelerator backend flag.\n");
        return -1;
    }
    options->backend = backend;
    return 0;
}

static void task_options_init(TaskOptions* options) {
    memset(options, 0, sizeof(*options));
    options->image_normalization = -1;
    options->timed_runs = 1;
}

static int parse_common_option(int argc, char** argv, int* index,
                               TaskOptions* options) {
    const char* argument = argv[*index];
    const char* backend = NULL;
    if (!strcmp(argument, "--vulkan")) backend = "vulkan";
    else if (!strcmp(argument, "--opengl")) backend = "opengl";
    else if (!strcmp(argument, "--metal")) backend = "metal";
    else if (!strcmp(argument, "--cuda")) backend = "cuda";
    if (backend) return select_backend(options, backend) == 0 ? 1 : -1;
    if (!strcmp(argument, "--debug")) {
        options->debug = 1;
        return 1;
    }
    if (!strcmp(argument, "--include_transfers")) {
        options->include_transfers = 1;
        return 1;
    }
    if (!strcmp(argument, "--threads")) {
        if (*index + 1 >= argc || parse_int(argv[++(*index)], 1,
                                            &options->cpu_threads) != 0) {
            fprintf(stderr, "--threads must be a positive 32-bit integer.\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argument, "--warmup_runs")) {
        if (*index + 1 >= argc || parse_int(argv[++(*index)], 0,
                                            &options->warmup_runs) != 0) {
            fprintf(stderr, "--warmup_runs must be a non-negative integer.\n");
            return -1;
        }
        return 1;
    }
    if (!strcmp(argument, "--num_runs")) {
        if (*index + 1 >= argc || parse_int(argv[++(*index)], 1,
                                            &options->timed_runs) != 0) {
            fprintf(stderr, "--num_runs must be a positive integer.\n");
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
    if (!strcmp(argument, "--image")) {
        if (*index + 1 >= argc || options->image_count == MAX_BINDINGS ||
            parse_binding(argv[++(*index)], &options->images[options->image_count], 1) != 0) {
            fprintf(stderr,
                    "--image expects name=file for a fixed input or "
                    "name[d0,d1,...]=file for a dynamic input.\n");
            return -1;
        }
        options->image_count++;
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
    if (!strcmp(argument, "--image-normalize")) {
        if (*index + 1 >= argc ||
            parse_normalization(argv[++(*index)], &options->image_normalization) != 0) {
            fprintf(stderr,
                    "--image-normalize expects zero-one, minus-one-one, or raw-255.\n");
            return -1;
        }
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

static int has_suffix(const char* path, const char* suffix) {
    size_t path_length;
    size_t suffix_length;
    if (!path || !suffix) return 0;
    path_length = strlen(path);
    suffix_length = strlen(suffix);
    return path_length >= suffix_length &&
           !strcmp(path + path_length - suffix_length, suffix);
}

static int read_file(const char* path, void** bytes, size_t* size) {
    FILE* file;
    long length;
    void* data;
    if (!path || !bytes || !size) return -1;
    *bytes = NULL;
    *size = 0;
    file = fopen(path, "rb");
    if (!file) return -1;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    data = malloc(length ? (size_t)length : 1);
    if (!data) {
        fclose(file);
        return -1;
    }
    if (length && fread(data, 1, (size_t)length, file) != (size_t)length) {
        free(data);
        fclose(file);
        return -1;
    }
    if (fclose(file) != 0) {
        free(data);
        return -1;
    }
    *bytes = data;
    *size = (size_t)length;
    return 0;
}

static int assign_text(const SynurangLiteAllocator* allocator,
                       SynurangLiteBytes* field, const char* text) {
    return synurang_lite_bytes_assign(
               allocator, field, text, text ? strlen(text) : 0u) == SYNURANG_LITE_OK
               ? 0
               : -1;
}

static int bytes_equal_string(const SynurangLiteBytes* value, const char* text) {
    size_t text_length;
    if (!value || !text) return 0;
    text_length = strlen(text);
    return value->len == text_length &&
           (!text_length || !memcmp(value->data, text, text_length));
}

static void print_dispatch_error(const char* operation, VxCallClient* client) {
    int32_t length = 0;
    const uint8_t* message = vx_call_error(client, &length);
    fprintf(stderr, "%s failed", operation);
    if (message && length > 0)
        fprintf(stderr, ": %.*s", (int)length, (const char*)message);
    fputc('\n', stderr);
}

static int report_ok(const char* operation,
                     const VolvoxaiV1OperationReport* report) {
    const char* status_name;
    if (report && report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK) return 1;
    status_name = report ? volvoxai_v1_native_status_name(report->field_status) : NULL;
    fprintf(stderr, "%s failed: %s", operation,
            status_name ? status_name : "missing operation report");
    if (report && report->field_code != VOLVOXAI_V1_OPERATION_CODE_NONE)
        fprintf(stderr, " [code=%d]", (int)report->field_code);
    if (report && report->field_message.len)
        fprintf(stderr, ": %.*s", (int)report->field_message.len,
                (const char*)report->field_message.data);
    fputc('\n', stderr);
    return 0;
}

static void release_result_id(TaskSession* session, int64_t id) {
    VolvoxaiV1ResultRef request;
    uint8_t* payload;
    int32_t payload_length;
    if (id <= 0) return;
    volvoxai_v1_result_ref_init(&request);
    request.field_result_id = id;
    VX_CALL_MESSAGE(&session->client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RESULT,
                    volvoxai_v1_result_ref, &request, payload, payload_length);
    volvoxai_v1_result_ref_free(&request);
    vx_call_free(&session->client, payload);
}

static void free_encoded(uint8_t* encoded) {
    const SynurangLiteAllocator* allocator = synurang_lite_default_allocator();
    if (encoded) allocator->deallocate(allocator->context, encoded);
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

static const VolvoxaiV1TensorSpec* find_input(const TaskSession* session,
                                               const char* name) {
    if (!session || !session->context_initialized) return NULL;
    for (size_t index = 0; index < session->context.field_inputs.len; index++) {
        const VolvoxaiV1TensorSpec* spec = &session->context.field_inputs.data[index];
        if (bytes_equal_string(&spec->field_name, name)) return spec;
    }
    fprintf(stderr, "Unknown input tensor: %s\n", name);
    return NULL;
}

static int validate_dimension(const char* name,
                              const VolvoxaiV1DimensionConstraint* constraint,
                              int64_t extent) {
    if (!constraint || extent <= 0 ||
        (constraint->field_kind != VOLVOXAI_V1_DIMENSION_KIND_FIXED &&
         constraint->field_kind != VOLVOXAI_V1_DIMENSION_KIND_SYMBOLIC)) {
        fprintf(stderr, "Input %s has an invalid dimension.\n", name);
        return -1;
    }
    if (extent < constraint->field_min || extent > constraint->field_max ||
        constraint->field_multiple_of <= 0 ||
        extent % constraint->field_multiple_of) {
        fprintf(stderr,
                "Input %s extent %lld violates the declared [%lld, %lld] x %lld domain.\n",
                name, (long long)extent, (long long)constraint->field_min,
                (long long)constraint->field_max,
                (long long)constraint->field_multiple_of);
        return -1;
    }
    return 0;
}

static int prepare_input_descriptor(const Binding* source,
                                    const VolvoxaiV1TensorSpec* spec,
                                    PreparedInput* prepared) {
    size_t byte_size;
    if (!source || !spec || !prepared) return -1;
    byte_size = dtype_byte_size(spec->field_dtype);
    if (!byte_size || spec->field_dimensions.len > TASK_MAX_TENSOR_RANK ||
        spec->field_location != VOLVOXAI_V1_MEMORY_LOCATION_HOST ||
        (source->has_shape && source->rank != spec->field_dimensions.len)) {
        fprintf(stderr, "Input %s has an unsupported dtype, rank, or location.\n",
                source->name);
        return -1;
    }
    memset(prepared, 0, sizeof(*prepared));
    if (copy_string(prepared->name, sizeof(prepared->name), source->name) != 0)
        return -1;
    prepared->dtype = spec->field_dtype;
    prepared->rank = spec->field_dimensions.len;
    for (size_t axis = 0; axis < prepared->rank; axis++) {
        const VolvoxaiV1DimensionConstraint* dimension =
            &spec->field_dimensions.data[axis];
        int64_t extent;
        if (source->has_shape) {
            extent = source->shape[axis];
        } else if (dimension->field_kind == VOLVOXAI_V1_DIMENSION_KIND_FIXED) {
            extent = dimension->field_min;
        } else {
            fprintf(stderr,
                    "Dynamic input %s requires an explicit shape: "
                    "%s[d0,d1,...]=file.\n",
                    source->name, source->name);
            return -1;
        }
        if (validate_dimension(source->name, dimension, extent) != 0 ||
            (uint64_t)extent > SIZE_MAX / byte_size) {
            fprintf(stderr, "Input %s shape is too large or outside its domain.\n",
                    source->name);
            return -1;
        }
        prepared->shape[axis] = extent;
        byte_size *= (size_t)extent;
    }
    prepared->byte_size = byte_size;
    return 0;
}

static int append_prepared_input(TaskSession* session,
                                 const PreparedInput* prepared,
                                 void* storage) {
    if (!session || !prepared || !storage || session->input_count >= MAX_BINDINGS)
        return -1;
    for (size_t index = 0; index < session->input_count; index++) {
        if (!strcmp(session->inputs[index].name, prepared->name)) {
            fprintf(stderr, "Input %s was bound more than once.\n", prepared->name);
            return -1;
        }
    }
    session->inputs[session->input_count] = *prepared;
    session->inputs[session->input_count].data = storage;
    session->input_count++;
    return 0;
}

static int set_raw_input(TaskSession* session, const Binding* binding) {
    const VolvoxaiV1TensorSpec* spec = find_input(session, binding->name);
    PreparedInput prepared;
    const char* suffix;
    void* bytes = NULL;
    size_t size = 0;
    if (!spec || prepare_input_descriptor(binding, spec, &prepared) != 0) return -1;
    suffix = dtype_suffix(spec->field_dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Input %s has dtype %s and requires a %s file: %s\n",
                binding->name, dtype_name(spec->field_dtype),
                suffix ? suffix : "supported raw", binding->path);
        return -1;
    }
    if (read_file(binding->path, &bytes, &size) != 0) {
        fprintf(stderr, "Cannot read input file: %s\n", binding->path);
        return -1;
    }
    if (size != prepared.byte_size) {
        fprintf(stderr, "Input %s expects %zu raw bytes, got %zu from %s\n",
                binding->name, prepared.byte_size, size, binding->path);
        free(bytes);
        return -1;
    }
    if (append_prepared_input(session, &prepared, bytes) != 0) {
        free(bytes);
        return -1;
    }
    return 0;
}

static int clamp_rounded(float value, int minimum, int maximum) {
    long rounded = lroundf(value);
    if (rounded < minimum) return minimum;
    if (rounded > maximum) return maximum;
    return (int)rounded;
}

static int set_image_input(TaskSession* session, const Binding* binding,
                           const TaskOptions* options) {
    const VolvoxaiV1TensorSpec* spec = find_input(session, binding->name);
    PreparedInput prepared;
    int shape[TASK_MAX_TENSOR_RANK] = {0};
    size_t element_count;
    float* decoded = NULL;
    void* storage = NULL;
    int normalization = -1;
    char error[256] = {0};
    if (!spec || prepare_input_descriptor(binding, spec, &prepared) != 0) return -1;
    if (spec->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 &&
        spec->field_dtype != VOLVOXAI_V1_DATA_TYPE_I8 &&
        spec->field_dtype != VOLVOXAI_V1_DATA_TYPE_U8) {
        fprintf(stderr, "Image input %s requires F32, I8, or U8, got %s.\n",
                binding->name, dtype_name(spec->field_dtype));
        return -1;
    }
    for (size_t axis = 0; axis < prepared.rank; axis++) {
        if (prepared.shape[axis] <= 0 || prepared.shape[axis] > INT_MAX) {
            fprintf(stderr, "Image input %s has an unsupported shape.\n", binding->name);
            return -1;
        }
        shape[axis] = (int)prepared.shape[axis];
    }
    normalization = options->image_normalization;
    if (normalization < 0) {
        fprintf(stderr,
                "Image input %s requires explicit --image-normalize; graph metadata is never used for application preprocessing.\n",
                binding->name);
        return -1;
    }
    element_count = spec->field_dtype == VOLVOXAI_V1_DATA_TYPE_F32
        ? prepared.byte_size / sizeof(float) : prepared.byte_size;
    decoded = (float*)malloc(element_count * sizeof(float));
    storage = malloc(prepared.byte_size ? prepared.byte_size : 1);
    if (!decoded || !storage) {
        fprintf(stderr, "Cannot allocate image input %s.\n", binding->name);
        free(decoded);
        free(storage);
        return -1;
    }
    if (volvox_load_image_to_tensor(binding->path, decoded, shape,
                                    (int)prepared.rank,
                                    normalization, error, sizeof(error)) != 0) {
        fprintf(stderr, "Cannot decode image %s: %s\n", binding->path, error);
        free(decoded);
        free(storage);
        return -1;
    }
    if (spec->field_dtype == VOLVOXAI_V1_DATA_TYPE_F32) {
        memcpy(storage, decoded, prepared.byte_size);
    } else if (normalization == VOLVOX_IMAGE_RAW_255) {
        for (size_t index = 0; index < element_count; index++) {
            if (spec->field_dtype == VOLVOXAI_V1_DATA_TYPE_U8)
                ((uint8_t*)storage)[index] = (uint8_t)clamp_rounded(decoded[index], 0, 255);
            else
                ((int8_t*)storage)[index] = (int8_t)clamp_rounded(decoded[index] - 128.0f,
                                                                 -128, 127);
        }
    } else {
        uint8_t* payload;
        int32_t payload_length = 0;
        VolvoxaiV1AffineQuantization quantization;
        int quantization_initialized = 0;
        VolvoxaiV1GetInputAffineQuantizationRequest request;
        volvoxai_v1_get_input_affine_quantization_request_init(&request);
        request.field_context_id = session->context_id;
        if (assign_text(request._allocator, &request.field_name, binding->name) != 0) {
            volvoxai_v1_get_input_affine_quantization_request_free(&request);
            free(decoded);
            free(storage);
            return -1;
        }
        VX_CALL_MESSAGE(&session->client,
                        VX_RPC_VX_INFERENCE_SERVICE_GET_INPUT_AFFINE_QUANTIZATION,
                        volvoxai_v1_get_input_affine_quantization_request,
                        &request, payload, payload_length);
        volvoxai_v1_get_input_affine_quantization_request_free(&request);
        if (!payload) {
            print_dispatch_error("GetInputAffineQuantization",
                                 &session->client);
            free(decoded);
            free(storage);
            return -1;
        }
        volvoxai_v1_affine_quantization_init(&quantization);
        quantization_initialized = 1;
        if (volvoxai_v1_affine_quantization_decode(
                &quantization, payload, (size_t)payload_length) != SYNURANG_LITE_OK ||
            !report_ok("GetInputAffineQuantization", quantization.field_report)) {
            if (quantization_initialized)
                volvoxai_v1_affine_quantization_free(&quantization);
            vx_call_free(&session->client, payload);
            free(decoded);
            free(storage);
            return -1;
        }
        if (!quantization.field_defined || !(quantization.field_scale > 0.0f)) {
            fprintf(stderr,
                    "Normalized byte image input %s requires per-tensor quantization metadata.\n",
                    binding->name);
            volvoxai_v1_affine_quantization_free(&quantization);
            vx_call_free(&session->client, payload);
            free(decoded);
            free(storage);
            return -1;
        }
        for (size_t index = 0; index < element_count; index++) {
            float quantized = decoded[index] / quantization.field_scale +
                              (float)quantization.field_zero_point;
            if (spec->field_dtype == VOLVOXAI_V1_DATA_TYPE_U8)
                ((uint8_t*)storage)[index] = (uint8_t)clamp_rounded(quantized, 0, 255);
            else
                ((int8_t*)storage)[index] = (int8_t)clamp_rounded(quantized, -128, 127);
        }
        volvoxai_v1_affine_quantization_free(&quantization);
        vx_call_free(&session->client, payload);
    }
    free(decoded);
    if (append_prepared_input(session, &prepared, storage) != 0) {
        free(storage);
        return -1;
    }
    return 0;
}

static int fill_tensor(const PreparedInput* input, VolvoxaiV1Tensor* tensor) {
    if (!input || !tensor || !tensor->_allocator) return -1;
    if (assign_text(tensor->_allocator, &tensor->field_name, input->name) != 0)
        return -1;
    tensor->field_dtype = input->dtype;
    for (size_t axis = 0; axis < input->rank; axis++) {
        int64_t* extent = volvoxai_v1_tensor_add_shape(tensor);
        if (!extent) return -1;
        *extent = input->shape[axis];
    }
    if (synurang_lite_bytes_assign(tensor->_allocator, &tensor->field_inline,
                                   input->data, input->byte_size) != SYNURANG_LITE_OK)
        return -1;
    tensor->which_payload = TENSOR_INLINE_PAYLOAD;
    return 0;
}

static int encode_input_request(TaskSession* session) {
    if (session->decode) {
        VolvoxaiV1DecodePrefillRequest request;
        volvoxai_v1_decode_prefill_request_init(&request);
        request.field_context_id = session->context_id;
        request.which_cursor = 3; /* DecodePrefillRequest.position */
        request.field_position = session->prefill_position;
        for (size_t index = 0; index < session->input_count; index++) {
            VolvoxaiV1Tensor* tensor =
                volvoxai_v1_decode_prefill_request_add_inputs(&request);
            if (!tensor || fill_tensor(&session->inputs[index], tensor) != 0) {
                volvoxai_v1_decode_prefill_request_free(&request);
                return -1;
            }
        }
        if (volvoxai_v1_decode_prefill_request_encode(
                &request, &session->input_request,
                &session->input_request_len) != SYNURANG_LITE_OK) {
            volvoxai_v1_decode_prefill_request_free(&request);
            return -1;
        }
        volvoxai_v1_decode_prefill_request_free(&request);
    } else {
        VolvoxaiV1ExecuteRequest request;
        volvoxai_v1_execute_request_init(&request);
        request.field_context_id = session->context_id;
        for (size_t index = 0; index < session->input_count; index++) {
            VolvoxaiV1Tensor* tensor = volvoxai_v1_execute_request_add_inputs(&request);
            if (!tensor || fill_tensor(&session->inputs[index], tensor) != 0) {
                volvoxai_v1_execute_request_free(&request);
                return -1;
            }
        }
        if (volvoxai_v1_execute_request_encode(
                &request, &session->input_request,
                &session->input_request_len) != SYNURANG_LITE_OK) {
            volvoxai_v1_execute_request_free(&request);
            return -1;
        }
        volvoxai_v1_execute_request_free(&request);
    }
    if (session->input_request_len > INT32_MAX) {
        free_encoded(session->input_request);
        session->input_request = NULL;
        session->input_request_len = 0;
        return -1;
    }
    return 0;
}

static int encode_create_runtime_request(const TaskOptions* options,
                                         uint8_t** encoded,
                                         size_t* encoded_length) {
    VolvoxaiV1CreateRuntimeRequest request;
    volvoxai_v1_create_runtime_request_init(&request);
    request.field_debug = options->debug;
    request.field_cpu_threads = options->cpu_threads;
    request.has_execution_mode = 1;
    request.field_execution_mode = VOLVOXAI_V1_EXECUTION_MODE_DIRECT;
    if (volvoxai_v1_create_runtime_request_encode(
            &request, encoded, encoded_length) != SYNURANG_LITE_OK) {
        volvoxai_v1_create_runtime_request_free(&request);
        return -1;
    }
    volvoxai_v1_create_runtime_request_free(&request);
    return *encoded_length <= INT32_MAX ? 0 : -1;
}

static int encode_load_model_request(int64_t runtime_id,
                                     const ModelPaths* paths,
                                     const TaskOptions* options,
                                     uint8_t** encoded,
                                     size_t* encoded_length) {
    VolvoxaiV1LoadModelRequest request;
    volvoxai_v1_load_model_request_init(&request);
    request.field_runtime_id = runtime_id;
    if (assign_text(request._allocator, &request.field_graph_path, paths->graph) != 0)
        goto fail;
    for (size_t index = 0; index < options->weight_path_count; index++) {
        SynurangLiteBytes* path =
            volvoxai_v1_load_model_request_add_weight_paths(&request);
        if (!path || assign_text(request._allocator, path,
                                 options->weight_paths[index]) != 0)
            goto fail;
    }
    if (volvoxai_v1_load_model_request_encode(
            &request, encoded, encoded_length) != SYNURANG_LITE_OK)
        goto fail;
    volvoxai_v1_load_model_request_free(&request);
    return *encoded_length <= INT32_MAX ? 0 : -1;
fail:
    volvoxai_v1_load_model_request_free(&request);
    return -1;
}

static int encode_compile_model_request(int64_t model_id, const char* backend,
                                         uint8_t** encoded, size_t* encoded_length) {
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
    result = volvoxai_v1_compile_model_request_encode(&request, encoded, encoded_length) ==
             SYNURANG_LITE_OK ? 0 : -1;
cleanup:
    request.field_policy = NULL;
    volvoxai_v1_backend_policy_free(&policy);
    volvoxai_v1_compile_model_request_free(&request);
    return result;
}

static void print_backend(const VolvoxaiV1OperationReport* report) {
    const SynurangLiteBytes* backend = report ? &report->field_backend : NULL;
    if (report && report->field_route && report->field_route->field_provider.len)
        backend = &report->field_route->field_provider;
    printf("Backend: %.*s\n", backend && backend->data ? (int)backend->len : 7,
           backend && backend->data ? (const char*)backend->data : "unknown");
}

static void print_debug_report(const char* operation,
                               const VolvoxaiV1OperationReport* report) {
    const char* status;
    if (!report) return;
    status = volvoxai_v1_native_status_name(report->field_status);
    fprintf(stderr, "[debug] %s status=%s", operation,
            status ? status : "NATIVE_STATUS_UNSPECIFIED");
    if (report->field_backend.len)
        fprintf(stderr, " backend=%.*s", (int)report->field_backend.len,
                (const char*)report->field_backend.data);
    if (report->field_timings && report->field_timings->field_execution_time_ms > 0.0)
        fprintf(stderr, " execution_ms=%.3f",
                report->field_timings->field_execution_time_ms);
    fputc('\n', stderr);
}

static int create_runtime(TaskSession* session, const TaskOptions* options) {
    uint8_t* encoded = NULL;
    size_t encoded_length = 0;
    uint8_t* payload = NULL;
    int32_t payload_length = 0;
    VolvoxaiV1RuntimeHandle handle;
    int initialized = 0;
    int result = -1;
    if (encode_create_runtime_request(options, &encoded, &encoded_length) != 0)
        goto cleanup;
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME,
        encoded, (int32_t)encoded_length, &payload_length);
    if (!payload) {
        print_dispatch_error("CreateRuntime", &session->client);
        goto cleanup;
    }
    volvoxai_v1_runtime_handle_init(&handle);
    initialized = 1;
    if (volvoxai_v1_runtime_handle_decode(
            &handle, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "CreateRuntime returned an undecodable response.\n");
        goto cleanup;
    }
    session->runtime_id = handle.field_runtime_id;
    if (!report_ok("CreateRuntime", handle.field_report)) goto cleanup;
    result = session->runtime_id > 0 ? 0 : -1;
cleanup:
    if (initialized) volvoxai_v1_runtime_handle_free(&handle);
    if (payload) vx_call_free(&session->client, payload);
    free_encoded(encoded);
    return result;
}

static int load_model(TaskSession* session, const ModelPaths* paths,
                      const TaskOptions* options) {
    uint8_t* encoded = NULL;
    size_t encoded_length = 0;
    uint8_t* payload = NULL;
    int32_t payload_length = 0;
    VolvoxaiV1ModelHandle handle;
    int initialized = 0;
    int result = -1;
    if (encode_load_model_request(session->runtime_id, paths, options,
                                  &encoded, &encoded_length) != 0)
        goto cleanup;
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL,
        encoded, (int32_t)encoded_length, &payload_length);
    if (!payload) {
        print_dispatch_error("LoadModel", &session->client);
        goto cleanup;
    }
    volvoxai_v1_model_handle_init(&handle);
    initialized = 1;
    if (volvoxai_v1_model_handle_decode(
            &handle, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "LoadModel returned an undecodable response.\n");
        goto cleanup;
    }
    session->model_id = handle.field_model_id;
    if (!report_ok("LoadModel", handle.field_report)) goto cleanup;
    result = session->model_id > 0 ? 0 : -1;
cleanup:
    if (initialized) volvoxai_v1_model_handle_free(&handle);
    if (payload) vx_call_free(&session->client, payload);
    free_encoded(encoded);
    return result;
}

static int compile_model(TaskSession* session, const TaskOptions* options) {
    uint8_t* policy = NULL;
    size_t policy_length = 0;
    uint8_t* payload = NULL;
    int32_t payload_length = 0;
    VolvoxaiV1CompiledModelHandle handle;
    int initialized = 0;
    int result = -1;
    if (encode_compile_model_request(session->model_id, options->backend, &policy, &policy_length) != 0)
        goto cleanup;
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL,
                            policy, policy_length, &payload_length);
    if (!payload) {
        print_dispatch_error("CompileModel", &session->client);
        goto cleanup;
    }
    volvoxai_v1_compiled_model_handle_init(&handle);
    initialized = 1;
    if (volvoxai_v1_compiled_model_handle_decode(
            &handle, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "CompileModel returned an undecodable response.\n");
        goto cleanup;
    }
    session->compiled_model_id = handle.field_compiled_model_id;
    if (!report_ok("CompileModel", handle.field_report)) goto cleanup;
    print_backend(handle.field_report);
    if (options->debug) print_debug_report("compile", handle.field_report);
    result = session->compiled_model_id > 0 ? 0 : -1;
cleanup:
    if (initialized) volvoxai_v1_compiled_model_handle_free(&handle);
    if (payload) vx_call_free(&session->client, payload);
    free_encoded(policy);
    return result;
}

static int create_context(TaskSession* session, int decode) {
    uint8_t* payload;
    uint8_t* encoded = NULL;
    size_t encoded_length = 0u;
    int32_t payload_length = 0;
    VolvoxaiV1CreateExecutionContextRequest request;
    volvoxai_v1_create_execution_context_request_init(&request);
    request.field_compiled_model_id = session->compiled_model_id;
    request.field_decode_row_mode = decode ? VOLVOXAI_V1_DECODE_ROW_MODE_AUTO
                                          : VOLVOXAI_V1_DECODE_ROW_MODE_DISABLED;
    if (volvoxai_v1_create_execution_context_request_encode(
            &request, &encoded, &encoded_length) != SYNURANG_LITE_OK) {
        volvoxai_v1_create_execution_context_request_free(&request);
        return -1;
    }
    volvoxai_v1_create_execution_context_request_free(&request);
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_EXECUTION_CONTEXT,
        encoded, (int32_t)encoded_length, &payload_length);
    free_encoded(encoded);
    if (!payload) {
        print_dispatch_error("CreateExecutionContext", &session->client);
        return -1;
    }
    volvoxai_v1_execution_context_handle_init(&session->context);
    session->context_initialized = 1;
    if (volvoxai_v1_execution_context_handle_decode(
            &session->context, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "CreateExecutionContext returned an undecodable response.\n");
        vx_call_free(&session->client, payload);
        return -1;
    }
    vx_call_free(&session->client, payload);
    session->context_id = session->context.field_context_id;
    if (!report_ok("CreateExecutionContext", session->context.field_report)) return -1;
    return session->context_id > 0 ? 0 : -1;
}

static void session_release_result(TaskSession* session) {
    if (session->result_info_initialized) {
        volvoxai_v1_result_info_free(&session->result_info);
        session->result_info_initialized = 0;
    }
    release_result_id(session, session->result_id);
    session->result_id = 0;
}

static void session_close(TaskSession* session) {
    if (!session) return;
    session_release_result(session);
    free_encoded(session->input_request);
    for (size_t index = 0; index < session->input_count; index++)
        free(session->inputs[index].data);
    if (session->context_initialized)
        volvoxai_v1_execution_context_handle_free(&session->context);
    vx_call_client_close(&session->client);
    memset(session, 0, sizeof(*session));
}

static int session_open(TaskSession* session, const char* model_path,
                        TaskOptions* options, int decode,
                        int32_t prefill_position) {
    ModelPaths paths;
    memset(session, 0, sizeof(*session));
    session->debug = options->debug;
    session->decode = decode;
    session->prefill_position = prefill_position;
    if (resolve_model_paths(model_path, &paths) != 0) return -1;
    if (!options->weight_path_count && paths.default_weights[0])
        options->weight_paths[options->weight_path_count++] = paths.default_weights;
    if (!vx_call_client_open(&session->client)) goto fail;
    if (create_runtime(session, options) != 0 ||
        load_model(session, &paths, options) != 0 ||
        compile_model(session, options) != 0 ||
        create_context(session, decode) != 0)
        goto fail;
    for (size_t index = 0; index < options->input_count; index++)
        if (set_raw_input(session, &options->inputs[index]) != 0) goto fail;
    for (size_t index = 0; index < options->image_count; index++)
        if (set_image_input(session, &options->images[index], options) != 0)
            goto fail;
    if (session->input_count != session->context.field_inputs.len) {
        fprintf(stderr,
                "Execution requires one binding for each of %zu inputs; got %zu.\n",
                session->context.field_inputs.len, session->input_count);
        goto fail;
    }
    if (encode_input_request(session) != 0) {
        fprintf(stderr, "Cannot encode the generated input request.\n");
        goto fail;
    }
    return 0;
fail:
    session_close(session);
    return -1;
}

static int refresh_result_info(TaskSession* session) {
    uint8_t* payload;
    int32_t payload_length = 0;
    if (session->result_info_initialized) {
        volvoxai_v1_result_info_free(&session->result_info);
        session->result_info_initialized = 0;
    }
    {
        VolvoxaiV1ResultRef request;
        volvoxai_v1_result_ref_init(&request);
        request.field_result_id = session->result_id;
        VX_CALL_MESSAGE(&session->client, VX_RPC_VX_INFERENCE_SERVICE_GET_RESULT,
                        volvoxai_v1_result_ref, &request, payload, payload_length);
        volvoxai_v1_result_ref_free(&request);
    }
    if (!payload) {
        print_dispatch_error("GetResult", &session->client);
        return -1;
    }
    volvoxai_v1_result_info_init(&session->result_info);
    session->result_info_initialized = 1;
    if (volvoxai_v1_result_info_decode(
            &session->result_info, payload,
            (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "GetResult returned an undecodable response.\n");
        vx_call_free(&session->client, payload);
        return -1;
    }
    vx_call_free(&session->client, payload);
    return report_ok("GetResult", session->result_info.field_report) ? 0 : -1;
}

static int accept_execution_result(TaskSession* session, uint8_t* payload,
                                   int32_t payload_length,
                                   const char* operation) {
    VolvoxaiV1ExecutionResultHandle handle;
    int result = -1;
    volvoxai_v1_execution_result_handle_init(&handle);
    if (volvoxai_v1_execution_result_handle_decode(
            &handle, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "%s returned an undecodable response.\n", operation);
        goto cleanup;
    }
    session->result_id = handle.field_result_id;
    if (!report_ok(operation, handle.field_report)) goto cleanup;
    if (session->debug) print_debug_report(operation, handle.field_report);
    result = session->result_id > 0 ? 0 : -1;
cleanup:
    if (result != 0 && handle.field_result_id > 0) {
        release_result_id(session, handle.field_result_id);
        session->result_id = 0;
    }
    volvoxai_v1_execution_result_handle_free(&handle);
    return result;
}

static int execute_once(TaskSession* session) {
    uint8_t* payload;
    int32_t payload_length = 0;
    session_release_result(session);
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_EXECUTE,
        session->input_request, (int32_t)session->input_request_len,
        &payload_length);
    if (!payload) {
        print_dispatch_error("Execute", &session->client);
        return -1;
    }
    if (accept_execution_result(session, payload, payload_length, "Execute") != 0) {
        vx_call_free(&session->client, payload);
        return -1;
    }
    vx_call_free(&session->client, payload);
    return 0;
}

static int decode_prefill_once(TaskSession* session) {
    uint8_t* payload;
    int32_t payload_length = 0;
    session_release_result(session);
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_DECODE_PREFILL,
        session->input_request, (int32_t)session->input_request_len,
        &payload_length);
    if (!payload) {
        print_dispatch_error("DecodePrefill", &session->client);
        return -1;
    }
    if (accept_execution_result(
            session, payload, payload_length, "DecodePrefill") != 0) {
        vx_call_free(&session->client, payload);
        return -1;
    }
    vx_call_free(&session->client, payload);
    return 0;
}

static int decode_step_once(TaskSession* session) {
    VolvoxaiV1DecodeStepRequest request;
    uint8_t* encoded = NULL;
    size_t encoded_length = 0;
    uint8_t* payload = NULL;
    int32_t payload_length = 0;
    int result = -1;
    volvoxai_v1_decode_step_request_init(&request);
    request.field_context_id = session->context_id;
    if (volvoxai_v1_decode_step_request_encode(
            &request, &encoded, &encoded_length) != SYNURANG_LITE_OK ||
        encoded_length > INT32_MAX) {
        fprintf(stderr, "Cannot encode DecodeStep request.\n");
        goto cleanup;
    }
    session_release_result(session);
    payload = vx_call_bytes(&session->client, VX_RPC_VX_INFERENCE_SERVICE_DECODE_STEP,
        encoded, (int32_t)encoded_length, &payload_length);
    if (!payload) {
        print_dispatch_error("DecodeStep", &session->client);
        goto cleanup;
    }
    result = accept_execution_result(
        session, payload, payload_length, "DecodeStep");
cleanup:
    if (payload) vx_call_free(&session->client, payload);
    free_encoded(encoded);
    volvoxai_v1_decode_step_request_free(&request);
    return result;
}

static int materialize_all_outputs(TaskSession* session);

static int execute_timed(TaskSession* session, const TaskOptions* options,
                         const char* label) {
    double first = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double total = 0.0;
    for (int index = 0; index < options->warmup_runs; index++) {
        if (execute_once(session) != 0 ||
            (options->include_transfers &&
             (refresh_result_info(session) != 0 ||
              materialize_all_outputs(session) != 0)))
            return -1;
    }
    for (int index = 0; index < options->timed_runs; index++) {
        double start = now_ms();
        double elapsed;
        if (execute_once(session) != 0 ||
            (options->include_transfers &&
             (refresh_result_info(session) != 0 ||
              materialize_all_outputs(session) != 0)))
            return -1;
        elapsed = now_ms() - start;
        if (!index) first = minimum = maximum = elapsed;
        if (elapsed < minimum) minimum = elapsed;
        if (elapsed > maximum) maximum = elapsed;
        total += elapsed;
    }
    if (options->warmup_runs || options->timed_runs != 1 || options->include_transfers) {
        fprintf(stderr,
                "[bench] %s: first=%.3f avg=%.3f min=%.3f max=%.3f ms "
                "(warmup=%d, runs=%d, transfers=%s)\n",
                label, first, total / options->timed_runs, minimum, maximum,
                options->warmup_runs, options->timed_runs,
                options->include_transfers ? "on" : "off");
    }
    return session->result_info_initialized ? 0 : refresh_result_info(session);
}

static const VolvoxaiV1TensorInfo* find_output(const TaskSession* session,
                                               const char* name) {
    if (!session || !session->result_info_initialized) return NULL;
    if (!name && session->result_info.field_outputs.len)
        return &session->result_info.field_outputs.data[0];
    for (size_t index = 0; index < session->result_info.field_outputs.len; index++) {
        const VolvoxaiV1TensorInfo* info =
            &session->result_info.field_outputs.data[index];
        if (bytes_equal_string(&info->field_name, name)) return info;
    }
    fprintf(stderr, "Unknown output tensor: %s\n", name ? name : "<first>");
    return NULL;
}

static int copy_output_info(TaskSession* session,
                            const VolvoxaiV1TensorInfo* info,
                            void** bytes) {
    uint8_t* payload;
    int32_t payload_length = 0;
    VolvoxaiV1ReadOutputResponse response;
    int response_initialized = 0;
    int result = -1;
    if (!info || !bytes || info->field_name.len > INT32_MAX) return -1;
    *bytes = NULL;
    if (info->field_byte_size > SIZE_MAX) return -1;
    {
        VolvoxaiV1ReadOutputRequest request;
        volvoxai_v1_read_output_request_init(&request);
        request.field_result_id = session->result_id;
        if (synurang_lite_bytes_assign(request._allocator, &request.field_name,
                                      info->field_name.data, info->field_name.len) != SYNURANG_LITE_OK) {
            volvoxai_v1_read_output_request_free(&request);
            return -1;
        }
        VX_CALL_MESSAGE(&session->client, VX_RPC_VX_INFERENCE_SERVICE_READ_OUTPUT,
                        volvoxai_v1_read_output_request, &request, payload, payload_length);
        volvoxai_v1_read_output_request_free(&request);
    }
    if (!payload) {
        print_dispatch_error("ReadOutput", &session->client);
        return -1;
    }
    volvoxai_v1_read_output_response_init(&response);
    response_initialized = 1;
    if (volvoxai_v1_read_output_response_decode(
            &response, payload, (size_t)payload_length) != SYNURANG_LITE_OK) {
        fprintf(stderr, "ReadOutput returned an undecodable response.\n");
        goto cleanup;
    }
    if (!report_ok("ReadOutput", response.field_report) || !response.field_tensor ||
        response.field_tensor->which_payload != TENSOR_INLINE_PAYLOAD ||
        response.field_tensor->field_inline.len != (size_t)info->field_byte_size) {
        if (response.field_report &&
            response.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK)
            fprintf(stderr, "ReadOutput returned an invalid inline tensor.\n");
        goto cleanup;
    }
    *bytes = malloc(info->field_byte_size ? (size_t)info->field_byte_size : 1u);
    if (!*bytes) goto cleanup;
    if (info->field_byte_size)
        memcpy(*bytes, response.field_tensor->field_inline.data,
               (size_t)info->field_byte_size);
    result = 0;
cleanup:
    if (result != 0) {
        free(*bytes);
        *bytes = NULL;
    }
    if (response_initialized) volvoxai_v1_read_output_response_free(&response);
    vx_call_free(&session->client, payload);
    return result;
}

static int copy_output(TaskSession* session, const char* name,
                       const VolvoxaiV1TensorInfo** found, void** bytes) {
    const VolvoxaiV1TensorInfo* info = find_output(session, name);
    if (!info || copy_output_info(session, info, bytes) != 0) return -1;
    if (found) *found = info;
    return 0;
}

static int write_file(const char* path, const void* bytes, size_t size) {
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

static int write_selected_outputs(TaskSession* session, const TaskOptions* options) {
    for (size_t index = 0; index < options->output_count; index++) {
        const Binding* binding = &options->outputs[index];
        const VolvoxaiV1TensorInfo* info = NULL;
        void* bytes = NULL;
        const char* suffix;
        if (copy_output(session, binding->name[0] ? binding->name : NULL,
                        &info, &bytes) != 0) return -1;
        suffix = dtype_suffix(info->field_dtype);
        if (!suffix || !has_suffix(binding->path, suffix)) {
            fprintf(stderr,
                    "Output %.*s has dtype %s and requires a %s file: %s\n",
                    (int)info->field_name.len,
                    info->field_name.data ? (const char*)info->field_name.data : "",
                    dtype_name(info->field_dtype),
                    suffix ? suffix : "supported raw", binding->path);
            free(bytes);
            return -1;
        }
        if (write_file(binding->path, bytes, (size_t)info->field_byte_size) != 0) {
            free(bytes);
            return -1;
        }
        free(bytes);
    }
    return 0;
}

static int materialize_all_outputs(TaskSession* session) {
    for (size_t index = 0; index < session->result_info.field_outputs.len; index++) {
        const VolvoxaiV1TensorInfo* info =
            &session->result_info.field_outputs.data[index];
        void* bytes = NULL;
        if (copy_output_info(session, info, &bytes) != 0) return -1;
        free(bytes);
    }
    return 0;
}

static int load_labels(const char* path, LabelList* labels) {
    void* bytes = NULL;
    size_t size = 0;
    char* cursor;
    size_t capacity = 0;
    memset(labels, 0, sizeof(*labels));
    if (!path) return 0;
    if (read_file(path, &bytes, &size) != 0 || size == SIZE_MAX) {
        fprintf(stderr, "Cannot read labels file: %s\n", path);
        free(bytes);
        return -1;
    }
    labels->storage = (char*)realloc(bytes, size + 1);
    if (!labels->storage) {
        free(bytes);
        return -1;
    }
    labels->storage[size] = 0;
    cursor = labels->storage;
    while (*cursor) {
        char* line = cursor;
        char* end = strchr(cursor, '\n');
        if (end) {
            *end = 0;
            cursor = end + 1;
        } else {
            cursor += strlen(cursor);
        }
        end = line + strlen(line);
        while (end > line && (end[-1] == '\r' || end[-1] == ' ' || end[-1] == '\t'))
            *--end = 0;
        if (!line[0]) continue;
        if (labels->count == capacity) {
            size_t next = capacity ? capacity * 2 : 16;
            char** grown = (char**)realloc(labels->items, next * sizeof(char*));
            if (!grown) return -1;
            labels->items = grown;
            capacity = next;
        }
        labels->items[labels->count++] = line;
    }
    return 0;
}

static void free_labels(LabelList* labels) {
    if (!labels) return;
    free(labels->items);
    free(labels->storage);
    memset(labels, 0, sizeof(*labels));
}

static const char* label_at(const LabelList* labels, int class_id) {
    return labels && class_id >= 0 && (size_t)class_id < labels->count
        ? labels->items[class_id] : NULL;
}

static int compare_ranked_value(const void* left, const void* right) {
    const RankedValue* a = (const RankedValue*)left;
    const RankedValue* b = (const RankedValue*)right;
    if (a->score > b->score) return -1;
    if (a->score < b->score) return 1;
    return a->index < b->index ? -1 : a->index > b->index;
}

static void print_root_help(const char* argv0) {
    printf("VolvoxAI Native Task Example\n\n");
    printf("Usage: %s <command> [options]\n\n", argv0);
    printf("Commands:\n");
    printf("  run <model>       Execute typed raw/image inputs and write raw outputs.\n");
    printf("  classify <model>  Rank a declared F32 logits output.\n");
    printf("  detect <model>    Rank declared F32 boxes and scores outputs.\n");
    printf("  decode <model>    Run public decode prefill/step operations.\n");
    printf("  version           Print release information.\n\n");
    printf("All commands use the generated protobuf C service API.\n");
}

static void print_common_help(void) {
    printf("  --weights <file>             Add a safetensors shard (repeatable).\n");
    printf("  --input <name[shape]=file>   Load one exact typed input; shape is required for dynamic axes.\n");
    printf("  --image <name[shape]=file>   Decode an image; shape is required for dynamic axes.\n");
    printf("  --image-normalize <mode>     zero-one, minus-one-one, or raw-255.\n");
    printf("  --output <name=file|file>    Write exact typed raw output data.\n");
    printf("  --vulkan | --opengl | --metal | --cuda\n");
    printf("  --debug\n");
}

static void print_run_help(const char* argv0) {
    printf("Usage: %s run <model-dir|graph.json> [options]\n\n", argv0);
    print_common_help();
}

static void print_classify_help(const char* argv0) {
    printf("Usage: %s classify <model-dir|graph.json> [options]\n\n", argv0);
    printf("  --logits <name>              Declared F32 logits output (default logits).\n");
    printf("  --labels <file>              Optional class labels (discovers labels.txt).\n");
    printf("  --top-k <n>                  Number of rows to print (default 5).\n");
    print_common_help();
}

static void print_detect_help(const char* argv0) {
    printf("Usage: %s detect <model-dir|graph.json> [options]\n\n", argv0);
    printf("  --boxes <name>               Declared F32 boxes output (default boxes).\n");
    printf("  --scores <name>              Declared F32 scores output (default scores).\n");
    printf("  --labels <file>              Optional class labels (discovers labels.txt).\n");
    printf("  --max-det <n>                Maximum detections to print (default 20).\n");
    printf("  --warmup_runs <n>            Discarded executions before timing.\n");
    printf("  --num_runs <n>               Timed executions (default 1).\n");
    printf("  --include_transfers          Attest stable-result readback in timing scope.\n");
    print_common_help();
}

static void print_decode_help(const char* argv0) {
    printf("Usage: %s decode <model-dir|graph.json> [options]\n\n", argv0);
    printf("  --steps <n>                  Decode steps after prefill (default 1).\n");
    printf("  --prefill-position <n>       Final active prompt position (default 0).\n");
    print_common_help();
}

static int command_run(int argc, char** argv) {
    TaskOptions options;
    TaskSession session;
    int result = 1;
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_run_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    task_options_init(&options);
    for (int index = 3; index < argc; index++) {
        int parsed = parse_common_option(argc, argv, &index, &options);
        if (parsed < 0) return 2;
        if (!parsed) {
            fprintf(stderr, "Unknown run option: %s\n", argv[index]);
            return 2;
        }
    }
    if (session_open(&session, argv[2], &options, 0, 0) != 0) return 1;
    if (execute_timed(&session, &options, "run") == 0 &&
        write_selected_outputs(&session, &options) == 0)
        result = 0;
    session_close(&session);
    return result;
}

static int command_classify(int argc, char** argv) {
    TaskOptions options;
    TaskSession session;
    LabelList labels;
    const char* logits_name = "logits";
    const char* labels_path = NULL;
    char discovered_labels[PATH_MAX];
    int top_k = 5;
    int result = 1;
    const VolvoxaiV1TensorInfo* info = NULL;
    void* bytes = NULL;
    RankedValue* ranked = NULL;
    size_t count;
    memset(&labels, 0, sizeof(labels));
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_classify_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    task_options_init(&options);
    for (int index = 3; index < argc; index++) {
        if (!strcmp(argv[index], "--logits") && index + 1 < argc) {
            logits_name = argv[++index];
        } else if (!strcmp(argv[index], "--labels") && index + 1 < argc) {
            labels_path = argv[++index];
        } else if (!strcmp(argv[index], "--top-k") && index + 1 < argc) {
            if (parse_int(argv[++index], 1, &top_k) != 0) return 2;
        } else {
            int parsed = parse_common_option(argc, argv, &index, &options);
            if (parsed < 0) return 2;
            if (!parsed) {
                fprintf(stderr, "Unknown classify option: %s\n", argv[index]);
                return 2;
            }
        }
    }
    labels_path = discover_file(argv[2], labels_path, "labels.txt",
                                discovered_labels, sizeof(discovered_labels));
    if (labels_path && load_labels(labels_path, &labels) != 0) return 1;
    if (session_open(&session, argv[2], &options, 0, 0) != 0) goto cleanup_labels;
    if (execute_timed(&session, &options, "classify") != 0 ||
        copy_output(&session, logits_name, &info, &bytes) != 0)
        goto cleanup_session;
    if (info->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
        info->field_byte_size % sizeof(float)) {
        fprintf(stderr, "Classification output %s must be F32.\n", logits_name);
        goto cleanup_session;
    }
    count = (size_t)info->field_byte_size / sizeof(float);
    ranked = (RankedValue*)calloc(count ? count : 1, sizeof(*ranked));
    if (!ranked) goto cleanup_session;
    for (size_t index = 0; index < count; index++) {
        ranked[index].index = index;
        ranked[index].class_id = (int)index;
        ranked[index].score = ((float*)bytes)[index];
    }
    qsort(ranked, count, sizeof(*ranked), compare_ranked_value);
    printf(labels.count ? "rank\tclass\tlabel\tscore\n" : "rank\tclass\tscore\n");
    for (size_t rank = 0; rank < count && rank < (size_t)top_k; rank++) {
        const char* label = label_at(&labels, ranked[rank].class_id);
        if (labels.count)
            printf("%zu\t%d\t%s\t%g\n", rank + 1, ranked[rank].class_id,
                   label ? label : "<unknown>", ranked[rank].score);
        else
            printf("%zu\t%d\t%g\n", rank + 1, ranked[rank].class_id,
                   ranked[rank].score);
    }
    if (write_selected_outputs(&session, &options) == 0) result = 0;
cleanup_session:
    free(ranked);
    free(bytes);
    session_close(&session);
cleanup_labels:
    free_labels(&labels);
    return result;
}

static int command_detect(int argc, char** argv) {
    TaskOptions options;
    TaskSession session;
    LabelList labels;
    const char* boxes_name = "boxes";
    const char* scores_name = "scores";
    const char* labels_path = NULL;
    char discovered_labels[PATH_MAX];
    int maximum_detections = 20;
    int result = 1;
    const VolvoxaiV1TensorInfo* boxes_info = NULL;
    const VolvoxaiV1TensorInfo* scores_info = NULL;
    void* boxes_bytes = NULL;
    void* scores_bytes = NULL;
    RankedValue* detections = NULL;
    size_t box_count;
    size_t score_count;
    size_t detection_count;
    size_t class_count;
    memset(&labels, 0, sizeof(labels));
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_detect_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    task_options_init(&options);
    for (int index = 3; index < argc; index++) {
        if (!strcmp(argv[index], "--boxes") && index + 1 < argc) {
            boxes_name = argv[++index];
        } else if (!strcmp(argv[index], "--scores") && index + 1 < argc) {
            scores_name = argv[++index];
        } else if (!strcmp(argv[index], "--labels") && index + 1 < argc) {
            labels_path = argv[++index];
        } else if (!strcmp(argv[index], "--max-det") && index + 1 < argc) {
            if (parse_int(argv[++index], 1, &maximum_detections) != 0) return 2;
        } else {
            int parsed = parse_common_option(argc, argv, &index, &options);
            if (parsed < 0) return 2;
            if (!parsed) {
                fprintf(stderr, "Unknown detect option: %s\n", argv[index]);
                return 2;
            }
        }
    }
    labels_path = discover_file(argv[2], labels_path, "labels.txt",
                                discovered_labels, sizeof(discovered_labels));
    if (labels_path && load_labels(labels_path, &labels) != 0) return 1;
    if (session_open(&session, argv[2], &options, 0, 0) != 0) goto cleanup_labels;
    if (execute_timed(&session, &options, "detect") != 0 ||
        copy_output(&session, boxes_name, &boxes_info, &boxes_bytes) != 0 ||
        copy_output(&session, scores_name, &scores_info, &scores_bytes) != 0)
        goto cleanup_session;
    if (boxes_info->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
        scores_info->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32 ||
        boxes_info->field_byte_size % sizeof(float) ||
        scores_info->field_byte_size % sizeof(float)) {
        fprintf(stderr, "Detection boxes and scores must be F32.\n");
        goto cleanup_session;
    }
    box_count = (size_t)boxes_info->field_byte_size / sizeof(float);
    score_count = (size_t)scores_info->field_byte_size / sizeof(float);
    if (!box_count || box_count % 4) {
        fprintf(stderr, "Detection boxes must contain N rows of four coordinates.\n");
        goto cleanup_session;
    }
    detection_count = box_count / 4;
    if (!score_count || score_count % detection_count) {
        fprintf(stderr, "Detection scores must contain one class row per box.\n");
        goto cleanup_session;
    }
    class_count = score_count / detection_count;
    detections = (RankedValue*)calloc(detection_count, sizeof(*detections));
    if (!detections) goto cleanup_session;
    for (size_t index = 0; index < detection_count; index++) {
        const float* row = (const float*)scores_bytes + index * class_count;
        size_t best = 0;
        for (size_t class_id = 1; class_id < class_count; class_id++)
            if (row[class_id] > row[best]) best = class_id;
        detections[index].index = index;
        detections[index].class_id = (int)best;
        detections[index].score = row[best];
    }
    qsort(detections, detection_count, sizeof(*detections), compare_ranked_value);
    printf(labels.count
           ? "rank\tindex\tscore\tscore_pct\tclass\tlabel\tx0\ty0\tx1\ty1\n"
           : "rank\tindex\tscore\tscore_pct\tclass\tx0\ty0\tx1\ty1\n");
    for (size_t rank = 0; rank < detection_count &&
                          rank < (size_t)maximum_detections; rank++) {
        const RankedValue* detection = &detections[rank];
        const float* box = (const float*)boxes_bytes + detection->index * 4;
        const char* label = label_at(&labels, detection->class_id);
        if (labels.count)
            printf("%zu\t%zu\t%g\t%.2f%%\t%d\t%s\t%g\t%g\t%g\t%g\n",
                   rank + 1, detection->index, detection->score,
                   detection->score * 100.0f, detection->class_id,
                   label ? label : "<unknown>", box[0], box[1], box[2], box[3]);
        else
            printf("%zu\t%zu\t%g\t%.2f%%\t%d\t%g\t%g\t%g\t%g\n",
                   rank + 1, detection->index, detection->score,
                   detection->score * 100.0f, detection->class_id,
                   box[0], box[1], box[2], box[3]);
    }
    if (write_selected_outputs(&session, &options) == 0) result = 0;
cleanup_session:
    free(detections);
    free(boxes_bytes);
    free(scores_bytes);
    session_close(&session);
cleanup_labels:
    free_labels(&labels);
    return result;
}

static int command_decode(int argc, char** argv) {
    TaskOptions options;
    TaskSession session;
    int steps = 1;
    int prefill_position = 0;
    int result = 1;
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_decode_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    task_options_init(&options);
    for (int index = 3; index < argc; index++) {
        if (!strcmp(argv[index], "--steps") && index + 1 < argc) {
            if (parse_int(argv[++index], 0, &steps) != 0) return 2;
        } else if (!strcmp(argv[index], "--prefill-position") && index + 1 < argc) {
            if (parse_int(argv[++index], 0, &prefill_position) != 0) return 2;
        } else {
            int parsed = parse_common_option(argc, argv, &index, &options);
            if (parsed < 0) return 2;
            if (!parsed) {
                fprintf(stderr, "Unknown decode option: %s\n", argv[index]);
                return 2;
            }
        }
    }
    if (session_open(&session, argv[2], &options, 1,
                     (int32_t)prefill_position) != 0)
        return 1;
    if (decode_prefill_once(&session) != 0) goto cleanup;
    for (int index = 0; index < steps; index++) {
        if (decode_step_once(&session) != 0) goto cleanup;
    }
    if (refresh_result_info(&session) == 0 &&
        write_selected_outputs(&session, &options) == 0)
        result = 0;
cleanup:
    session_close(&session);
    return result;
}

int main(int argc, char** argv) {
    if (argc < 2 || !strcmp(argv[1], "--help") || !strcmp(argv[1], "-h")) {
        print_root_help(argv[0]);
        return argc < 2 ? 1 : 0;
    }
    if (!strcmp(argv[1], "--version") || !strcmp(argv[1], "-V") ||
        !strcmp(argv[1], "version")) {
        printf("VolvoxAI Native Task Example %s (commit %s, built %s)\n",
               VOLVOXAI_VERSION, VOLVOXAI_GIT_COMMIT, VOLVOXAI_BUILD_DATE);
        return 0;
    }
    if (!strcmp(argv[1], "run")) return command_run(argc, argv);
    if (!strcmp(argv[1], "classify")) return command_classify(argc, argv);
    if (!strcmp(argv[1], "detect")) return command_detect(argc, argv);
    if (!strcmp(argv[1], "decode")) return command_decode(argc, argv);
    fprintf(stderr, "Unknown command: %s\n\n", argv[1]);
    print_root_help(argv[0]);
    return 2;
}
