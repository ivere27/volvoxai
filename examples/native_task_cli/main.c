#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "volvoxai.h"

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

typedef struct TaskSession {
    VxRuntime* runtime;
    VxModel* model;
    VxCompiledModel* compiled;
    VxExecutionContext* context;
    VxResult* result;
    VxTensorBinding input_bindings[MAX_BINDINGS];
    void* input_storage[MAX_BINDINGS];
    size_t input_binding_count;
    VxReport report;
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
            if (binding->rank == VX_MAX_TENSOR_RANK ||
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
    else if (!strcmp(argument, "--nnapi")) backend = "nnapi";
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

static int find_input(TaskSession* session, const char* name,
                      VxTensorSpec* found) {
    size_t count = vx_execution_context_input_count(session->context);
    for (size_t index = 0; index < count; index++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        if (vx_execution_context_input_spec(session->context, index, &spec,
                                            &session->report) == VX_STATUS_OK &&
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

static int prepare_binding_descriptor(TaskSession* session,
                                      const Binding* source,
                                      const VxTensorSpec* spec,
                                      VxTensorBinding* prepared) {
    size_t byte_size = dtype_byte_size(spec->dtype);
    if (!byte_size || spec->rank > VX_MAX_TENSOR_RANK ||
        (source->has_shape && source->rank != spec->rank)) {
        fprintf(stderr, "Input %s has an unsupported dtype or rank.\n",
                source->name);
        return -1;
    }
    *prepared = (VxTensorBinding)VX_TENSOR_BINDING_INIT;
    prepared->name = spec->name;
    prepared->dtype = spec->dtype;
    prepared->rank = spec->rank;
    for (uint32_t axis = 0; axis < spec->rank; axis++) {
        int64_t extent;
        if (source->has_shape) {
            extent = source->shape[axis];
        } else if (spec->dimensions[axis].kind == VX_DIMENSION_FIXED) {
            extent = spec->dimensions[axis].min;
        } else {
            fprintf(stderr,
                    "Dynamic input %s requires an explicit shape: "
                    "%s[d0,d1,...]=file.\n",
                    source->name, source->name);
            return -1;
        }
        if (extent <= 0 || (uint64_t)extent > SIZE_MAX / byte_size) {
            fprintf(stderr, "Input %s shape is too large.\n", source->name);
            return -1;
        }
        prepared->shape[axis] = extent;
        byte_size *= (size_t)extent;
    }
    prepared->byte_size = byte_size;
    prepared->location = VX_MEMORY_HOST;
    (void)session;
    return 0;
}

static int append_prepared_input(TaskSession* session,
                                 const VxTensorBinding* prepared,
                                 void* storage) {
    if (!session || !prepared || !storage ||
        session->input_binding_count >= MAX_BINDINGS) return -1;
    for (size_t index = 0; index < session->input_binding_count; index++)
        if (!strcmp(session->input_bindings[index].name, prepared->name)) {
            fprintf(stderr, "Input %s was bound more than once.\n",
                    prepared->name);
            return -1;
        }
    session->input_bindings[session->input_binding_count] = *prepared;
    session->input_bindings[session->input_binding_count].data = storage;
    session->input_storage[session->input_binding_count] = storage;
    session->input_binding_count++;
    return 0;
}

static int set_raw_input(TaskSession* session, const Binding* binding) {
    VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
    VxTensorBinding prepared = VX_TENSOR_BINDING_INIT;
    const char* suffix;
    void* bytes = NULL;
    size_t size = 0;
    if (find_input(session, binding->name, &spec) != 0 ||
        prepare_binding_descriptor(session, binding, &spec, &prepared) != 0)
        return -1;
    suffix = dtype_suffix(spec.dtype);
    if (!suffix || !has_suffix(binding->path, suffix)) {
        fprintf(stderr, "Input %s has dtype %s and requires a %s file: %s\n",
                binding->name, dtype_name(spec.dtype), suffix ? suffix : "supported raw",
                binding->path);
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
    VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
    VxTensorBinding prepared = VX_TENSOR_BINDING_INIT;
    VxAffineQuantization quantization = VX_AFFINE_QUANTIZATION_INIT;
    int shape[VX_MAX_TENSOR_RANK] = {0};
    size_t element_count;
    float* decoded = NULL;
    void* storage = NULL;
    int normalization = -1;
    char error[256] = {0};
    VxStatus status;
    if (find_input(session, binding->name, &spec) != 0 ||
        prepare_binding_descriptor(session, binding, &spec, &prepared) != 0)
        return -1;
    if (spec.dtype != VX_DTYPE_F32 && spec.dtype != VX_DTYPE_I8 &&
        spec.dtype != VX_DTYPE_U8) {
        fprintf(stderr, "Image input %s requires F32, I8, or U8, got %s.\n",
                binding->name, dtype_name(spec.dtype));
        return -1;
    }
    for (uint32_t axis = 0; axis < prepared.rank; axis++) {
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
    element_count = spec.dtype == VX_DTYPE_F32
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
    if (spec.dtype == VX_DTYPE_F32) {
        memcpy(storage, decoded, prepared.byte_size);
    } else if (normalization == VOLVOX_IMAGE_RAW_255) {
        for (size_t index = 0; index < element_count; index++) {
            if (spec.dtype == VX_DTYPE_U8)
                ((uint8_t*)storage)[index] = (uint8_t)clamp_rounded(decoded[index], 0, 255);
            else
                ((int8_t*)storage)[index] = (int8_t)clamp_rounded(decoded[index] - 128.0f,
                                                                 -128, 127);
        }
    } else {
        status = vx_execution_context_input_affine_quantization(
            session->context, binding->name, &quantization, &session->report);
        if (status != VX_STATUS_OK) {
            print_failure("Reading image input quantization", status,
                          &session->report);
            free(decoded);
            free(storage);
            return -1;
        }
        if (!quantization.defined) {
            fprintf(stderr,
                    "Normalized byte image input %s requires per-tensor quantization metadata.\n",
                    binding->name);
            free(decoded);
            free(storage);
            return -1;
        }
        for (size_t index = 0; index < element_count; index++) {
            float quantized = decoded[index] / quantization.scale +
                              (float)quantization.zero_point;
            if (spec.dtype == VX_DTYPE_U8)
                ((uint8_t*)storage)[index] = (uint8_t)clamp_rounded(quantized, 0, 255);
            else
                ((int8_t*)storage)[index] = (int8_t)clamp_rounded(quantized, -128, 127);
        }
    }
    free(decoded);
    if (append_prepared_input(session, &prepared, storage) != 0) {
        free(storage);
        return -1;
    }
    return 0;
}

static void session_close(TaskSession* session) {
    if (!session) return;
    vx_result_release(session->result);
    session->result = NULL;
    if (session->context) (void)vx_execution_context_close(session->context, NULL);
    vx_execution_context_release(session->context);
    for (size_t index = 0; index < session->input_binding_count; index++)
        free(session->input_storage[index]);
    vx_compiled_model_release(session->compiled);
    vx_model_release(session->model);
    if (session->runtime) (void)vx_runtime_close(session->runtime, NULL);
    vx_runtime_release(session->runtime);
    memset(session, 0, sizeof(*session));
}

static int session_open(TaskSession* session, const char* model_path,
                        TaskOptions* options, int decode) {
    ModelPaths paths;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    const char* selected_backends[1];
    VxStatus status;
    memset(session, 0, sizeof(*session));
    session->report = (VxReport)VX_REPORT_INIT;
    if (resolve_model_paths(model_path, &paths) != 0) return -1;
    if (!options->weight_path_count && paths.default_weights[0])
        options->weight_paths[options->weight_path_count++] = paths.default_weights;
    runtime_options.debug = options->debug;
    runtime_options.cpu_threads = options->cpu_threads;
    source.graph_path = paths.graph;
    source.weight_paths = options->weight_paths;
    source.weight_path_count = options->weight_path_count;
    if (options->backend) {
        selected_backends[0] = options->backend;
        policy.mode = VX_BACKEND_REQUIRE;
        policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
        policy.backends = selected_backends;
        policy.backend_count = 1;
    }
    if (decode) context_options.decode_row_mode = VX_DECODE_ROW_AUTO;
    status = vx_runtime_create(&runtime_options, &session->runtime, &session->report);
    if (status == VX_STATUS_OK)
        status = vx_runtime_load_model(session->runtime, &source, &session->model,
                                       &session->report);
    if (status == VX_STATUS_OK)
        status = vx_model_compile(session->model, &policy, &session->compiled,
                                  &session->report);
    if (status == VX_STATUS_OK)
        status = vx_compiled_model_create_context(session->compiled, &context_options,
                                                  &session->context, &session->report);
    if (status != VX_STATUS_OK) {
        print_failure("Native init", status, &session->report);
        session_close(session);
        return -1;
    }
    printf("Backend: %s\n", session->report.backend[0]
           ? session->report.backend : "unknown");
    for (size_t index = 0; index < options->input_count; index++)
        if (set_raw_input(session, &options->inputs[index]) != 0) {
            session_close(session);
            return -1;
        }
    for (size_t index = 0; index < options->image_count; index++)
        if (set_image_input(session, &options->images[index], options) != 0) {
            session_close(session);
            return -1;
        }
    return 0;
}

static int execute_once(TaskSession* session) {
    VxStatus status;
    vx_result_release(session->result);
    session->result = NULL;
    status = vx_execution_context_execute(
        session->context, session->input_bindings,
        session->input_binding_count, &session->result, &session->report);
    if (status != VX_STATUS_OK) {
        print_failure("Native inference", status, &session->report);
        return -1;
    }
    return 0;
}

static int execute_timed(TaskSession* session, const TaskOptions* options,
                         const char* label) {
    double first = 0.0;
    double minimum = 0.0;
    double maximum = 0.0;
    double total = 0.0;
    for (int index = 0; index < options->warmup_runs; index++)
        if (execute_once(session) != 0) return -1;
    for (int index = 0; index < options->timed_runs; index++) {
        double start = now_ms();
        double elapsed;
        if (execute_once(session) != 0) return -1;
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
    return 0;
}

static int find_output(TaskSession* session, const char* name, VxTensorInfo* found) {
    size_t count = session->result ? vx_result_output_count(session->result) : 0;
    for (size_t index = 0; index < count; index++) {
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        if (vx_result_output_info(session->result, index, &info, &session->report) ==
                VX_STATUS_OK &&
            ((!name && index == 0) || (name && info.name && !strcmp(info.name, name)))) {
            *found = info;
            return 0;
        }
    }
    fprintf(stderr, "Unknown output tensor: %s\n", name ? name : "<first>");
    return -1;
}

static int copy_output(TaskSession* session, const char* name,
                       VxTensorInfo* info, void** bytes) {
    VxStatus status;
    if (!bytes || find_output(session, name, info) != 0) return -1;
    *bytes = malloc(info->byte_size ? info->byte_size : 1);
    if (!*bytes) return -1;
    status = vx_result_read(session->result, info->name, *bytes, info->byte_size,
                            NULL, &session->report);
    if (status != VX_STATUS_OK) {
        print_failure("Reading output", status, &session->report);
        free(*bytes);
        *bytes = NULL;
        return -1;
    }
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
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        void* bytes = NULL;
        const char* suffix;
        if (copy_output(session, binding->name[0] ? binding->name : NULL,
                        &info, &bytes) != 0) return -1;
        suffix = dtype_suffix(info.dtype);
        if (!suffix || !has_suffix(binding->path, suffix)) {
            fprintf(stderr, "Output %s has dtype %s and requires a %s file: %s\n",
                    info.name, dtype_name(info.dtype), suffix ? suffix : "supported raw",
                    binding->path);
            free(bytes);
            return -1;
        }
        if (write_file(binding->path, bytes, info.byte_size) != 0) {
            free(bytes);
            return -1;
        }
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
    printf("  decode <model>    Run public decode seed/step operations.\n");
    printf("  version           Print release information.\n\n");
    printf("All commands use only volvoxai.h opaque handles.\n");
}

static void print_common_help(void) {
    printf("  --weights <file>             Add a safetensors shard (repeatable).\n");
    printf("  --input <name[shape]=file>   Load one exact typed input; shape is required for dynamic axes.\n");
    printf("  --image <name[shape]=file>   Decode an image; shape is required for dynamic axes.\n");
    printf("  --image-normalize <mode>     zero-one, minus-one-one, or raw-255.\n");
    printf("  --output <name=file|file>    Write exact typed raw output data.\n");
    printf("  --vulkan | --opengl | --metal | --nnapi | --cuda\n");
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
    printf("  --steps <n>                  Decode steps after seed (default 1).\n");
    printf("  --start-position <n>         Position of the first step (default 0).\n");
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
    if (session_open(&session, argv[2], &options, 0) != 0) return 1;
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
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
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
    if (session_open(&session, argv[2], &options, 0) != 0) goto cleanup_labels;
    if (execute_timed(&session, &options, "classify") != 0 ||
        copy_output(&session, logits_name, &info, &bytes) != 0)
        goto cleanup_session;
    if (info.dtype != VX_DTYPE_F32 || info.byte_size % sizeof(float)) {
        fprintf(stderr, "Classification output %s must be F32.\n", logits_name);
        goto cleanup_session;
    }
    count = info.byte_size / sizeof(float);
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
    VxTensorInfo boxes_info = VX_TENSOR_INFO_INIT;
    VxTensorInfo scores_info = VX_TENSOR_INFO_INIT;
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
    if (session_open(&session, argv[2], &options, 0) != 0) goto cleanup_labels;
    if (execute_timed(&session, &options, "detect") != 0 ||
        copy_output(&session, boxes_name, &boxes_info, &boxes_bytes) != 0 ||
        copy_output(&session, scores_name, &scores_info, &scores_bytes) != 0)
        goto cleanup_session;
    if (boxes_info.dtype != VX_DTYPE_F32 || scores_info.dtype != VX_DTYPE_F32 ||
        boxes_info.byte_size % sizeof(float) || scores_info.byte_size % sizeof(float)) {
        fprintf(stderr, "Detection boxes and scores must be F32.\n");
        goto cleanup_session;
    }
    box_count = boxes_info.byte_size / sizeof(float);
    score_count = scores_info.byte_size / sizeof(float);
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
    int start_position = 0;
    int result = 1;
    VxStatus status;
    if (argc < 3 || !strcmp(argv[2], "--help") || !strcmp(argv[2], "-h")) {
        print_decode_help(argv[0]);
        return argc < 3 ? 1 : 0;
    }
    task_options_init(&options);
    for (int index = 3; index < argc; index++) {
        if (!strcmp(argv[index], "--steps") && index + 1 < argc) {
            if (parse_int(argv[++index], 0, &steps) != 0) return 2;
        } else if (!strcmp(argv[index], "--start-position") && index + 1 < argc) {
            if (parse_int(argv[++index], 0, &start_position) != 0) return 2;
        } else {
            int parsed = parse_common_option(argc, argv, &index, &options);
            if (parsed < 0) return 2;
            if (!parsed) {
                fprintf(stderr, "Unknown decode option: %s\n", argv[index]);
                return 2;
            }
        }
    }
    if (session_open(&session, argv[2], &options, 1) != 0) return 1;
    status = vx_execution_context_decode_seed(
        session.context, session.input_bindings,
        session.input_binding_count, &session.result, &session.report);
    if (status != VX_STATUS_OK) {
        print_failure("Decode seed", status, &session.report);
        goto cleanup;
    }
    for (int index = 0; index < steps; index++) {
        VxResult* next = NULL;
        status = vx_execution_context_decode_step(session.context,
                                                  start_position + index,
                                                  NULL, 0u, &next,
                                                  &session.report);
        if (status != VX_STATUS_OK) {
            print_failure("Decode step", status, &session.report);
            vx_result_release(next);
            goto cleanup;
        }
        vx_result_release(session.result);
        session.result = next;
    }
    if (write_selected_outputs(&session, &options) == 0) result = 0;
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
