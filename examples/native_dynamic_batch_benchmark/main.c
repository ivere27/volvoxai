#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "evidence_tokens.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define BENCHMARK_RECORD "native-dynamic-batch-case"
#define FIXTURE_SEED UINT32_C(20260821)
#define MAX_WEIGHTS 8u
#define MAX_INPUTS 256u
#define MAX_SAMPLES 1000u
#define VULKAN_STAGING_BYTES UINT64_C(33554432)

#define FAIL(...) do { \
    fprintf(stderr, "native dynamic-batch benchmark: "); \
    fprintf(stderr, __VA_ARGS__); \
    fputc('\n', stderr); \
    goto cleanup; \
} while (0)

typedef enum InputMode {
    INPUT_RECEIPT = 1,
    INPUT_VQA_ENCODER = 2,
    INPUT_VQA_DECODER = 3
} InputMode;

typedef struct Options {
    const char* case_id;
    const char* graph;
    const char* weights[MAX_WEIGHTS];
    size_t weight_count;
    const char* backend;
    InputMode input_mode;
    const char* input_mode_name;
    const char** lane_files;
    size_t lane_file_count;
    size_t lane_file_capacity;
    size_t batch_size;
    size_t warmup;
    size_t repeat;
    uint32_t batch_delay_ms;
    uint32_t vulkan_arena_mb;
    int cuda_device_set;
    uint32_t cuda_device;
} Options;

typedef struct Fixture {
    size_t lane_count;
    size_t input_count;
    size_t unique_input_classes;
    const char* reuse;
    char* batch_symbol;
    VxTensorSpec* specs;
    char** names;
    VxTensorBinding* bindings;
    void** storage;
} Fixture;

typedef struct GroupResults {
    VxResult** results;
    VxRequest** requests;
    VxReport* reports;
    double elapsed_ms;
    uint64_t physical_execution_id;
} GroupResults;

typedef struct ParityStats {
    uint64_t compared_elements;
    uint64_t output_tensors;
    double max_abs;
    double max_rel;
    uint64_t receipt_decoded_outputs;
} ParityStats;

typedef struct BackendExecutionEvidence {
    uint64_t vulkan_arena_bytes;
    uint64_t vulkan_staging_bytes;
    int vulkan_stage_coherent;
    int vulkan_stage_coherent_set;
} BackendExecutionEvidence;

static int safe_token(const char* text) {
    const unsigned char* cursor = (const unsigned char*)text;
    if (!cursor || !*cursor) return 0;
    while (*cursor) {
        unsigned char c = *cursor++;
        if (!((c >= 'a' && c <= 'z') || (c >= 'A' && c <= 'Z') ||
              (c >= '0' && c <= '9') || c == '.' || c == '_' || c == '-'))
            return 0;
    }
    return 1;
}

static int parse_size(const char* text, size_t minimum, size_t maximum,
                      size_t* value) {
    char* end = NULL;
    unsigned long long parsed;
    if (!text || !text[0] || !value) return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !end || *end || parsed < minimum || parsed > maximum ||
        parsed > SIZE_MAX)
        return 0;
    *value = (size_t)parsed;
    return 1;
}

static int parse_options(int argc, char** argv, Options* options) {
    memset(options, 0, sizeof(*options));
    if (argc < 1) return 0;
    options->lane_file_capacity = (size_t)argc;
    options->lane_files = calloc(
        options->lane_file_capacity, sizeof(*options->lane_files));
    if (!options->lane_files) return 0;
    for (int index = 1; index < argc; index++) {
        const char* flag = argv[index];
        const char* value = index + 1 < argc ? argv[++index] : NULL;
        size_t number = 0;
        if (!value) return 0;
        if (!strcmp(flag, "--case-id")) options->case_id = value;
        else if (!strcmp(flag, "--graph")) options->graph = value;
        else if (!strcmp(flag, "--weight")) {
            if (options->weight_count >= MAX_WEIGHTS) return 0;
            options->weights[options->weight_count++] = value;
        } else if (!strcmp(flag, "--backend")) options->backend = value;
        else if (!strcmp(flag, "--input-mode")) {
            options->input_mode_name = value;
            if (!strcmp(value, "receipt")) options->input_mode = INPUT_RECEIPT;
            else if (!strcmp(value, "vqa-encoder"))
                options->input_mode = INPUT_VQA_ENCODER;
            else if (!strcmp(value, "vqa-decoder"))
                options->input_mode = INPUT_VQA_DECODER;
            else return 0;
        } else if (!strcmp(flag, "--lane-file")) {
            if (options->lane_file_count >= options->lane_file_capacity)
                return 0;
            options->lane_files[options->lane_file_count++] = value;
        } else if (!strcmp(flag, "--batch-size")) {
            if (!parse_size(value, 2u, SIZE_MAX / 2u,
                            &options->batch_size))
                return 0;
        } else if (!strcmp(flag, "--warmup")) {
            if (!parse_size(value, 1u, MAX_SAMPLES, &options->warmup))
                return 0;
        } else if (!strcmp(flag, "--repeat")) {
            if (!parse_size(value, 1u, MAX_SAMPLES, &options->repeat))
                return 0;
        } else if (!strcmp(flag, "--max-batch-delay-ms")) {
            if (!parse_size(value, 1u, UINT32_MAX, &number)) return 0;
            options->batch_delay_ms = (uint32_t)number;
        } else if (!strcmp(flag, "--vulkan-arena-mb")) {
            if (!parse_size(value, 1u, UINT32_MAX, &number)) return 0;
            options->vulkan_arena_mb = (uint32_t)number;
        } else if (!strcmp(flag, "--cuda-device")) {
            if (!parse_size(value, 0u, UINT32_MAX, &number)) return 0;
            options->cuda_device_set = 1;
            options->cuda_device = (uint32_t)number;
        } else return 0;
    }
    if (!safe_token(options->case_id) || !options->graph ||
        !options->weight_count || !options->backend || !options->input_mode ||
        !options->batch_size || !options->warmup || !options->repeat ||
        !options->batch_delay_ms)
        return 0;
    if (strcmp(options->backend, "vulkan") &&
        strcmp(options->backend, "opengl") &&
        strcmp(options->backend, "cuda"))
        return 0;
    if (!strcmp(options->backend, "vulkan")) {
        char expected[32];
        const char* configured = getenv("VOLVOX_VULKAN_MB");
        snprintf(expected, sizeof(expected), "%" PRIu32,
                 options->vulkan_arena_mb);
        if (!options->vulkan_arena_mb || !configured ||
            strcmp(configured, expected))
            return 0;
    } else if (options->vulkan_arena_mb) return 0;
    if (!strcmp(options->backend, "cuda")) {
        const char* configured = getenv("VOLVOXAI_CUDA_DEVICE");
        if (!options->cuda_device_set || options->cuda_device != 0u ||
            !configured || strcmp(configured, "0"))
            return 0;
    } else if (options->cuda_device_set) {
        return 0;
    }
    if (options->input_mode == INPUT_RECEIPT) {
        if (!options->lane_file_count ||
            options->lane_file_count > options->batch_size)
            return 0;
    } else if (options->lane_file_count) {
        return 0;
    }
    return 1;
}

static size_t dtype_size(VxDataType dtype) {
    switch (dtype) {
        case VX_DTYPE_BOOL:
        case VX_DTYPE_U8:
        case VX_DTYPE_I8: return 1u;
        case VX_DTYPE_I16:
        case VX_DTYPE_U16:
        case VX_DTYPE_F16:
        case VX_DTYPE_BF16: return 2u;
        case VX_DTYPE_I32:
        case VX_DTYPE_U32:
        case VX_DTYPE_F32: return 4u;
        case VX_DTYPE_I64:
        case VX_DTYPE_U64:
        case VX_DTYPE_F64: return 8u;
        default: return 0u;
    }
}

static int checked_mul(size_t left, size_t right, size_t* product) {
    if (!product || (right && left > SIZE_MAX / right)) return 0;
    *product = left * right;
    return 1;
}

static int read_exact_file(const char* path, void* data, size_t bytes) {
    FILE* file;
    long length;
    size_t read_count;
    if (!path || !data || !bytes) return 0;
    file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        (uint64_t)length != (uint64_t)bytes || fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    read_count = fread(data, 1u, bytes, file);
    {
        int close_status = fclose(file);
        return read_count == bytes && close_status == 0;
    }
}

static uint32_t prng_next(uint32_t* state) {
    uint32_t value = *state;
    value ^= value << 13;
    value ^= value >> 17;
    value ^= value << 5;
    *state = value;
    return value;
}

static void fill_synthetic(InputMode mode, const VxTensorSpec* spec, size_t lane,
                           size_t input_index, void* data, size_t bytes) {
    uint32_t state = FIXTURE_SEED ^
        (uint32_t)(lane + 1u) * UINT32_C(0x9e3779b9) ^
        (uint32_t)(input_index + 1u) * UINT32_C(0x85ebca6b);
    size_t elements = bytes / dtype_size(spec->dtype);
    if (spec->dtype == VX_DTYPE_F32) {
        float* output = (float*)data;
        int cache_index =
            mode == INPUT_VQA_DECODER && input_index >= 5u
                ? (int)input_index - 5 : -1;
        for (size_t index = 0; index < elements; index++) {
            if (mode == INPUT_VQA_ENCODER && input_index == 0u) {
                output[index] = (float)lane * 0.03125f;
            } else if (mode == INPUT_VQA_DECODER && cache_index >= 0) {
                int32_t centered = (int32_t)
                    ((index * (size_t)(cache_index + 3)) % 97u) - 48;
                output[index] =
                    (float)centered * 0.002f +
                    (float)lane * 0.03125f +
                    (float)cache_index * 0.0078125f;
            } else {
                int32_t signed_value =
                    (int32_t)(prng_next(&state) & 0xffffu) - 32768;
                output[index] = (float)signed_value / 327680.0f;
            }
        }
    } else if (spec->dtype == VX_DTYPE_I32) {
        int32_t* output = (int32_t*)data;
        for (size_t index = 0; index < elements; index++) {
            if (mode == INPUT_VQA_ENCODER && input_index == 1u)
                output[index] = (int32_t)((index + 4u + lane * 17u) % 256u);
            else if (mode == INPUT_VQA_ENCODER && input_index == 3u)
                output[index] = (int32_t)index;
            else if ((mode == INPUT_VQA_ENCODER && input_index == 2u) ||
                     (mode == INPUT_VQA_DECODER && input_index == 2u))
                output[index] = (int32_t)(lane % 8u);
            else if (mode == INPUT_VQA_DECODER && input_index == 0u)
                output[index] = (int32_t)(lane + 1u);
            else if (mode == INPUT_VQA_DECODER && input_index == 4u)
                output[index] = 1;
            else
                output[index] = 0;
        }
    } else {
        memset(data, 0, bytes);
    }
}

static int vqa_spec_matches(InputMode mode, size_t input,
                            const VxTensorSpec* spec) {
    static const int64_t encoder_shapes[4][4] = {
        {1, 1, 320, 672}, {1, 1, 0, 0}, {1, 0, 0, 0}, {1, 1, 0, 0},
    };
    int64_t expected[4] = {0};
    uint32_t rank;
    VxDataType dtype;
    char expected_name[32];
    snprintf(expected_name, sizeof(expected_name), "input%zu", input);
    if (!spec->name || strcmp(spec->name, expected_name)) return 0;
    if (mode == INPUT_VQA_ENCODER) {
        if (input >= 4u) return 0;
        rank = input == 0u ? 4u : input == 2u ? 1u : 2u;
        dtype = input == 0u ? VX_DTYPE_F32 : VX_DTYPE_I32;
        memcpy(expected, encoder_shapes[input], sizeof(expected));
    } else if (mode == INPUT_VQA_DECODER) {
        if (input >= 21u) return 0;
        if (input < 3u) {
            rank = input == 0u ? 2u : 1u;
            expected[0] = 1;
            expected[1] = input == 0u ? 1 : 0;
            dtype = VX_DTYPE_I32;
        } else if (input < 5u) {
            rank = 2u;
            expected[0] = 1;
            expected[1] = input == 3u ? 211 : 1;
            dtype = VX_DTYPE_I32;
        } else {
            rank = 4u;
            expected[0] = 1;
            expected[1] = 8;
            expected[2] = input < 13u ? 211 : 1;
            expected[3] = 40;
            dtype = VX_DTYPE_F32;
        }
    } else {
        return 0;
    }
    if (spec->rank != rank || spec->dtype != dtype) return 0;
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (spec->dimensions[axis].min != expected[axis]) return 0;
    }
    return 1;
}

static void fixture_clear(Fixture* fixture) {
    if (!fixture) return;
    if (fixture->storage) {
        size_t count = fixture->lane_count * fixture->input_count;
        for (size_t index = 0; index < count; index++)
            free(fixture->storage[index]);
    }
    free(fixture->storage);
    free(fixture->bindings);
    if (fixture->names) {
        for (size_t input = 0; input < fixture->input_count; input++)
            free(fixture->names[input]);
    }
    free(fixture->names);
    free(fixture->batch_symbol);
    free(fixture->specs);
    memset(fixture, 0, sizeof(*fixture));
}

static int fixture_create(const Options* options, VxCompiledModel* compiled,
                          Fixture* fixture, VxReport* report) {
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxExecutionContext* context = NULL;
    size_t binding_count;
    memset(fixture, 0, sizeof(*fixture));
    if (vx_compiled_model_create_context(compiled, &context_options, &context,
                                         report) != VX_STATUS_OK)
        return 0;
    fixture->lane_count = options->batch_size;
    fixture->input_count = vx_execution_context_input_count(context);
    if (!fixture->input_count || fixture->input_count > MAX_INPUTS ||
        !checked_mul(fixture->lane_count, fixture->input_count,
                     &binding_count))
        goto fail;
    if ((options->input_mode == INPUT_VQA_ENCODER &&
         fixture->input_count != 4u) ||
        (options->input_mode == INPUT_VQA_DECODER &&
         fixture->input_count != 21u))
        goto fail;
    fixture->specs = calloc(fixture->input_count, sizeof(*fixture->specs));
    fixture->names = calloc(fixture->input_count, sizeof(*fixture->names));
    fixture->bindings = calloc(binding_count, sizeof(*fixture->bindings));
    fixture->storage = calloc(binding_count, sizeof(*fixture->storage));
    if (!fixture->specs || !fixture->names || !fixture->bindings ||
        !fixture->storage)
        goto fail;
    for (size_t input = 0; input < fixture->input_count; input++) {
        VxTensorSpec* spec = &fixture->specs[input];
        size_t elements = 1u;
        size_t bytes;
        *spec = (VxTensorSpec)VX_TENSOR_SPEC_INIT;
        if (vx_execution_context_input_spec(context, input, spec, report) !=
                VX_STATUS_OK || !spec->name || !spec->name[0] || !spec->rank ||
            spec->rank > VX_MAX_TENSOR_RANK || !dtype_size(spec->dtype))
            goto fail;
        fixture->names[input] = malloc(strlen(spec->name) + 1u);
        if (!fixture->names[input]) goto fail;
        memcpy(fixture->names[input], spec->name, strlen(spec->name) + 1u);
        if (spec->dimensions[0].kind != VX_DIMENSION_SYMBOLIC ||
            !spec->dimensions[0].symbol || !spec->dimensions[0].symbol[0] ||
            spec->dimensions[0].min != 1 ||
            spec->dimensions[0].max < (int64_t)options->batch_size)
            goto fail;
        if (!fixture->batch_symbol) {
            fixture->batch_symbol =
                malloc(strlen(spec->dimensions[0].symbol) + 1u);
            if (!fixture->batch_symbol) goto fail;
            memcpy(fixture->batch_symbol, spec->dimensions[0].symbol,
                   strlen(spec->dimensions[0].symbol) + 1u);
        } else if (strcmp(fixture->batch_symbol,
                          spec->dimensions[0].symbol)) {
            goto fail;
        }
        if (options->input_mode == INPUT_RECEIPT &&
            (fixture->input_count != 1u || spec->dtype != VX_DTYPE_F32 ||
             spec->rank != 4u || strcmp(spec->name, "input0") ||
             spec->dimensions[0].min != 1 ||
             spec->dimensions[1].min != 1 ||
             spec->dimensions[2].min != 320 ||
             spec->dimensions[3].min != 672))
            goto fail;
        if (options->input_mode != INPUT_RECEIPT &&
            !vqa_spec_matches(options->input_mode, input, spec))
            goto fail;
        for (uint32_t axis = 0; axis < spec->rank; axis++) {
            int64_t extent = spec->dimensions[axis].min;
            if (extent <= 0 || (uint64_t)extent > SIZE_MAX ||
                !checked_mul(elements, (size_t)extent, &elements))
                goto fail;
        }
        if (!checked_mul(elements, dtype_size(spec->dtype), &bytes) || !bytes)
            goto fail;
        for (size_t lane = 0; lane < fixture->lane_count; lane++) {
            size_t slot = lane * fixture->input_count + input;
            VxTensorBinding* binding = &fixture->bindings[slot];
            void* storage = malloc(bytes);
            if (!storage) goto fail;
            fixture->storage[slot] = storage;
            *binding = (VxTensorBinding)VX_TENSOR_BINDING_INIT;
            binding->name = fixture->names[input];
            binding->dtype = spec->dtype;
            binding->rank = spec->rank;
            binding->data = storage;
            binding->byte_size = bytes;
            for (uint32_t axis = 0; axis < spec->rank; axis++)
                binding->shape[axis] = spec->dimensions[axis].min;
            if (options->input_mode == INPUT_RECEIPT) {
                size_t fixture_index = lane % options->lane_file_count;
                if (!read_exact_file(options->lane_files[fixture_index],
                                     storage, bytes))
                    goto fail;
            } else {
                fill_synthetic(options->input_mode, spec, lane, input,
                               storage, bytes);
            }
        }
    }
    /* Claim a distinct input class only after comparing complete binding
     * sets. No content or digest crosses the benchmark JSON boundary. */
    {
        size_t classes = options->input_mode == INPUT_RECEIPT
            ? options->lane_file_count : fixture->lane_count;
        for (size_t left = 0; left < classes; left++) {
            for (size_t right = left + 1u; right < classes; right++) {
                int differs = 0;
                for (size_t input = 0; input < fixture->input_count; input++) {
                    const VxTensorBinding* a =
                        &fixture->bindings[left * fixture->input_count + input];
                    const VxTensorBinding* b =
                        &fixture->bindings[right * fixture->input_count + input];
                    if (a->byte_size != b->byte_size ||
                        memcmp(a->data, b->data, a->byte_size)) {
                        differs = 1;
                        break;
                    }
                }
                if (!differs) goto fail;
            }
        }
    }
    if (options->input_mode == INPUT_RECEIPT) {
        fixture->unique_input_classes = options->lane_file_count;
        fixture->reuse = options->batch_size > options->lane_file_count
            ? "cyclic" : "none";
    } else {
        fixture->unique_input_classes = options->batch_size;
        fixture->reuse = "none";
    }
    (void)vx_execution_context_close(context, report);
    vx_execution_context_release(context);
    return 1;

fail:
    if (context) {
        (void)vx_execution_context_close(context, report);
        vx_execution_context_release(context);
    }
    fixture_clear(fixture);
    return 0;
}

static int report_token(const char* evidence, const char* key,
                        char* value, size_t capacity) {
    return vx_native_batch_evidence_token(
        evidence, key, value, capacity);
}

static size_t report_key_count(const char* evidence, const char* key) {
    return vx_native_batch_evidence_key_count(evidence, key);
}

static int report_u64(const VxReport* report, const char* key,
                      uint64_t* value) {
    char text[64];
    char* end = NULL;
    unsigned long long parsed;
    if (!report || !value ||
        !report_token(report->route_evidence, key, text, sizeof(text)))
        return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int report_u64_canonical(const VxReport* report, const char* key,
                                uint64_t* value) {
    char text[64];
    char canonical[64];
    uint64_t parsed = 0;
    if (!report || !value ||
        !report_token(report->route_evidence, key, text, sizeof(text)) ||
        !report_u64(report, key, &parsed))
        return 0;
    snprintf(canonical, sizeof(canonical), "%" PRIu64, parsed);
    if (strcmp(text, canonical)) return 0;
    *value = parsed;
    return 1;
}

static int report_string(const VxReport* report, const char* key,
                         const char* expected) {
    char value[256];
    return report && expected &&
        report_token(report->route_evidence, key, value, sizeof(value)) &&
        !strcmp(value, expected);
}

static const char* backend_execution_failure(
    const VxReport* report, const Options* options, int measured,
    BackendExecutionEvidence* evidence) {
    if (!report || !options || !evidence)
        return "backend-execution-arguments";
    if (!strcmp(options->backend, "vulkan")) {
        uint64_t arena = 0;
        uint64_t staging = 0;
        uint64_t uploads = 0;
        uint64_t downloads = 0;
        uint64_t expected_arena =
            (uint64_t)options->vulkan_arena_mb * UINT64_C(1024) *
            UINT64_C(1024);
        char coherent[4];
        int coherent_value;
        if (report_key_count(report->route_evidence, "vk_mem") != 1u ||
            report_key_count(report->route_evidence, "vk_arena") != 1u ||
            report_key_count(report->route_evidence, "vk_stage") != 1u ||
            report_key_count(report->route_evidence,
                             "vk_stage_coherent") != 1u ||
            report_key_count(report->route_evidence, "vk_up") != 1u ||
            report_key_count(report->route_evidence, "vk_down") != 1u)
            return "vulkan-proof-cardinality";
        if (!report_string(report, "vk_mem", "device-local"))
            return "vulkan-memory-placement";
        if (!report_u64_canonical(report, "vk_arena", &arena) ||
            arena != expected_arena)
            return "vulkan-arena-bytes";
        if (!report_u64_canonical(report, "vk_stage", &staging) ||
            staging != VULKAN_STAGING_BYTES)
            return "vulkan-staging-bytes";
        if (!report_token(report->route_evidence, "vk_stage_coherent",
                          coherent, sizeof(coherent)) ||
            (strcmp(coherent, "0") && strcmp(coherent, "1")))
            return "vulkan-staging-coherence";
        coherent_value = coherent[0] == '1';
        if (!report_u64_canonical(report, "vk_up", &uploads) || !uploads)
            return "vulkan-staging-upload";
        if (!report_u64_canonical(report, "vk_down", &downloads) ||
            !downloads)
            return "vulkan-staging-download";
        if (report_key_count(report->route_evidence,
                             "cuda_graph_replay"))
            return "vulkan-unexpected-cuda-proof";
        if (evidence->vulkan_stage_coherent_set &&
            (evidence->vulkan_arena_bytes != arena ||
             evidence->vulkan_staging_bytes != staging ||
             evidence->vulkan_stage_coherent != coherent_value))
            return "vulkan-placement-changed";
        evidence->vulkan_arena_bytes = arena;
        evidence->vulkan_staging_bytes = staging;
        evidence->vulkan_stage_coherent = coherent_value;
        evidence->vulkan_stage_coherent_set = 1;
    } else if (!strcmp(options->backend, "cuda")) {
        char replay[4];
        if (report_key_count(report->route_evidence,
                             "cuda_graph_replay") != 1u)
            return "cuda-graph-replay-cardinality";
        if (!report_token(report->route_evidence, "cuda_graph_replay",
                          replay, sizeof(replay)) ||
            (strcmp(replay, "0") && strcmp(replay, "1")))
            return "cuda-graph-replay-proof";
        if (measured && strcmp(replay, "1"))
            return "cuda-graph-replay-not-measured";
        if (report_key_count(report->route_evidence, "vk_mem") ||
            report_key_count(report->route_evidence, "vk_arena") ||
            report_key_count(report->route_evidence, "vk_stage") ||
            report_key_count(report->route_evidence,
                             "vk_stage_coherent") ||
            report_key_count(report->route_evidence, "vk_up") ||
            report_key_count(report->route_evidence, "vk_down"))
            return "cuda-unexpected-vulkan-proof";
    } else {
        if (report_key_count(report->route_evidence,
                             "cuda_graph_replay") ||
            report_key_count(report->route_evidence, "vk_mem") ||
            report_key_count(report->route_evidence, "vk_arena") ||
            report_key_count(report->route_evidence, "vk_stage") ||
            report_key_count(report->route_evidence,
                             "vk_stage_coherent") ||
            report_key_count(report->route_evidence, "vk_up") ||
            report_key_count(report->route_evidence, "vk_down"))
            return "opengl-unexpected-backend-proof";
    }
    return NULL;
}

static const char* strict_route_failure(const VxReport* report,
                                        const char* backend) {
    char expected_provider[96];
    char provider[96];
    uint64_t nodes = 0;
    uint64_t selected = 0;
    uint64_t fallback = 0;
    uint64_t missing = 0;
    if (!report || !backend) return "route-arguments";
    snprintf(expected_provider, sizeof(expected_provider), "builtin:%s",
             backend);
    if (report->status != VX_STATUS_OK) return "route-report-status";
    if (!report->route_attested) return "route-not-attested";
    if (report->tier_fallback_used) return "route-tier-fallback";
    if (report->operator_fallback_used) return "route-operator-fallback";
    if (strcmp(report->backend, backend)) return "route-backend";
    if (strcmp(report->fallback_evidence, "operator=none"))
        return "route-operator-evidence";
    if (!report_token(report->route_evidence, "provider", provider,
                      sizeof(provider)))
        return "route-provider-missing-or-duplicate";
    if (strcmp(provider, expected_provider)) return "route-provider";
    if (!report_u64(report, "nodes", &nodes) || !nodes)
        return "route-nodes";
    if (!report_u64(report, "selected", &selected) || selected != nodes)
        return "route-selected";
    if (!report_u64(report, "fallback", &fallback) || fallback)
        return "route-fallback";
    if (!report_u64(report, "missing", &missing) || missing)
        return "route-missing";
    return NULL;
}

static int strict_route(const VxReport* report, const char* backend) {
    return strict_route_failure(report, backend) == NULL;
}

static int diagnostic_failure(const char* stage, const char* check,
                              size_t lane, size_t output, int call_status,
                              const VxReport* report) {
    fprintf(stderr,
            "native dynamic-batch diagnostic: stage=%s check=%s "
            "lane=%zu output=%zu callStatus=%d reportStatus=%d "
            "routeAttested=%d tierFallback=%d operatorFallback=%d\n",
            stage, check, lane, output, call_status,
            report ? (int)report->status : -1,
            report ? (int)report->route_attested : -1,
            report ? (int)report->tier_fallback_used : -1,
            report ? (int)report->operator_fallback_used : -1);
    return 0;
}

static int diagnostic_u64(const char* stage, const char* check, size_t lane,
                          uint64_t observed, uint64_t expected) {
    fprintf(stderr,
            "native dynamic-batch diagnostic: stage=%s check=%s "
            "lane=%zu observed=%" PRIu64 " expected=%" PRIu64 "\n",
            stage, check, lane, observed, expected);
    return 0;
}

static void group_clear(GroupResults* group, size_t lane_count) {
    if (!group) return;
    if (group->results) {
        for (size_t lane = 0; lane < lane_count; lane++)
            vx_result_release(group->results[lane]);
    }
    if (group->requests) {
        for (size_t lane = 0; lane < lane_count; lane++)
            vx_request_release(group->requests[lane]);
    }
    free(group->results);
    free(group->requests);
    free(group->reports);
    memset(group, 0, sizeof(*group));
}

static int group_allocate(size_t lanes, int scheduled, GroupResults* group) {
    memset(group, 0, sizeof(*group));
    group->results = calloc(lanes, sizeof(*group->results));
    group->reports = calloc(lanes, sizeof(*group->reports));
    group->requests =
        scheduled ? calloc(lanes, sizeof(*group->requests)) : NULL;
    if (!group->results || !group->reports ||
        (scheduled && !group->requests))
        return 0;
    for (size_t lane = 0; lane < lanes; lane++)
        group->reports[lane] = (VxReport)VX_REPORT_INIT;
    return 1;
}

static int run_direct_group(VxRuntime* runtime, VxCompiledModel* compiled,
                            const Fixture* fixture, const Options* options,
                            int measured,
                            BackendExecutionEvidence* backend_evidence,
                            GroupResults* group) {
    uint64_t started;
    if (!group_allocate(fixture->lane_count, 0, group))
        return diagnostic_failure("direct", "allocate", 0u, 0u, -1, NULL);
    started = vx_runtime_monotonic_time_micros();
    if (!started)
        return diagnostic_failure("direct", "clock-start", 0u, 0u, -1,
                                  NULL);
    for (size_t lane = 0; lane < fixture->lane_count; lane++) {
        VxTensorBinding* bindings =
            fixture->bindings + lane * fixture->input_count;
        VxStatus status = vx_runtime_run(
            runtime, compiled, bindings, fixture->input_count,
            &group->results[lane], &group->reports[lane]);
        if (status != VX_STATUS_OK)
            return diagnostic_failure("direct", "runtime-run", lane, 0u,
                                      (int)status, &group->reports[lane]);
    }
    {
        uint64_t ended = vx_runtime_monotonic_time_micros();
        if (!ended || ended < started)
            return diagnostic_failure("direct", "clock-end", 0u, 0u, -1,
                                      NULL);
        group->elapsed_ms = (double)(ended - started) / 1000.0;
    }
    for (size_t lane = 0; lane < fixture->lane_count; lane++) {
        const char* route_failure =
            strict_route_failure(&group->reports[lane], options->backend);
        const char* execution_failure = backend_execution_failure(
            &group->reports[lane], options, measured, backend_evidence);
        size_t physical_id_count = report_key_count(
            group->reports[lane].route_evidence, "physicalExecutionId");
        size_t batch_size_count = report_key_count(
            group->reports[lane].route_evidence, "batchSize");
        size_t invocation_count = report_key_count(
            group->reports[lane].route_evidence, "trueBackendInvocations");
        if (route_failure)
            return diagnostic_failure("direct", route_failure, lane, 0u,
                                      VX_STATUS_OK, &group->reports[lane]);
        if (execution_failure)
            return diagnostic_failure("direct", execution_failure, lane, 0u,
                                      VX_STATUS_OK, &group->reports[lane]);
        if (!group->results[lane])
            return diagnostic_failure("direct", "result-null", lane, 0u,
                                      VX_STATUS_OK, &group->reports[lane]);
        if (physical_id_count)
            return diagnostic_u64("direct", "unexpected-physical-id-token",
                                  lane, physical_id_count, 0u);
        if (batch_size_count)
            return diagnostic_u64("direct", "unexpected-batch-size-token",
                                  lane, batch_size_count, 0u);
        if (invocation_count)
            return diagnostic_u64("direct", "unexpected-invocation-token",
                                  lane, invocation_count, 0u);
    }
    return 1;
}

static int run_scheduled_group(VxRuntime* runtime, VxCompiledModel* compiled,
                               const Fixture* fixture,
                               const Options* options, int measured,
                               BackendExecutionEvidence* backend_evidence,
                               const char* proof_identity,
                               uint64_t previous_physical_id,
                               GroupResults* group) {
    VxRuntimeSubmitOptions submit = VX_RUNTIME_SUBMIT_OPTIONS_INIT;
    uint64_t started;
    uint64_t first_id = 0;
    submit.freshness = VX_RUNTIME_FRESHNESS_ALL;
    if (!group_allocate(fixture->lane_count, 1, group))
        return diagnostic_failure("scheduled", "allocate", 0u, 0u, -1,
                                  NULL);
    started = vx_runtime_monotonic_time_micros();
    if (!started)
        return diagnostic_failure("scheduled", "clock-start", 0u, 0u, -1,
                                  NULL);
    for (size_t lane = 0; lane < fixture->lane_count; lane++) {
        VxTensorBinding* bindings =
            fixture->bindings + lane * fixture->input_count;
        VxStatus status = vx_runtime_submit(
            runtime, compiled, bindings, fixture->input_count, &submit,
            &group->requests[lane], &group->reports[lane]);
        if (status != VX_STATUS_OK)
            return diagnostic_failure("scheduled", "submit", lane, 0u,
                                      (int)status, &group->reports[lane]);
    }
    for (size_t lane = 0; lane < fixture->lane_count; lane++) {
        VxStatus status = vx_request_wait(
            group->requests[lane], VX_REQUEST_WAIT_INFINITE,
            &group->reports[lane]);
        if (status != VX_STATUS_OK)
            return diagnostic_failure("scheduled", "wait", lane, 0u,
                                      (int)status, &group->reports[lane]);
    }
    for (size_t lane = 0; lane < fixture->lane_count; lane++) {
        VxStatus status = vx_request_result(
            group->requests[lane], &group->results[lane],
            &group->reports[lane]);
        if (status != VX_STATUS_OK)
            return diagnostic_failure("scheduled", "result", lane, 0u,
                                      (int)status, &group->reports[lane]);
    }
    {
        uint64_t ended = vx_runtime_monotonic_time_micros();
        if (!ended || ended < started)
            return diagnostic_failure("scheduled", "clock-end", 0u, 0u,
                                      -1, NULL);
        group->elapsed_ms = (double)(ended - started) / 1000.0;
    }
    for (size_t lane = 0; lane < fixture->lane_count; lane++) {
        uint64_t invocations = 0;
        uint64_t batch_size = 0;
        uint64_t physical_id = 0;
        const char* prefix = "trueBackendInvocations=1;batchSize=";
        const char* route_failure =
            strict_route_failure(&group->reports[lane], options->backend);
        const char* execution_failure = backend_execution_failure(
            &group->reports[lane], options, measured, backend_evidence);
        if (route_failure)
            return diagnostic_failure("scheduled", route_failure, lane, 0u,
                                      VX_STATUS_OK, &group->reports[lane]);
        if (execution_failure)
            return diagnostic_failure("scheduled", execution_failure, lane,
                                      0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (!group->results[lane])
            return diagnostic_failure("scheduled", "result-null", lane, 0u,
                                      VX_STATUS_OK, &group->reports[lane]);
        if (strncmp(group->reports[lane].route_evidence, prefix,
                    strlen(prefix)))
            return diagnostic_failure("scheduled", "proof-prefix", lane, 0u,
                                      VX_STATUS_OK, &group->reports[lane]);
        if (!strstr(group->reports[lane].message,
                    "trueBackendInvocations=1"))
            return diagnostic_failure("scheduled", "message-invocation",
                                      lane, 0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (!report_u64(&group->reports[lane], "trueBackendInvocations",
                        &invocations))
            return diagnostic_failure("scheduled", "invocation-token", lane,
                                      0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (invocations != 1u)
            return diagnostic_u64("scheduled", "invocation-value", lane,
                                  invocations, 1u);
        if (!report_u64(&group->reports[lane], "batchSize", &batch_size))
            return diagnostic_failure("scheduled", "batch-size-token", lane,
                                      0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (batch_size != fixture->lane_count)
            return diagnostic_u64("scheduled", "batch-size-value", lane,
                                  batch_size, fixture->lane_count);
        if (!report_u64(&group->reports[lane], "physicalExecutionId",
                        &physical_id) || !physical_id)
            return diagnostic_failure("scheduled", "physical-id-token", lane,
                                      0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (!report_string(&group->reports[lane], "batchProtocol",
                           VX_BACKEND_BATCH_PROTOCOL))
            return diagnostic_failure("scheduled", "batch-protocol-token",
                                      lane, 0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (!report_string(&group->reports[lane], "batchProof",
                           proof_identity))
            return diagnostic_failure("scheduled", "batch-proof-token", lane,
                                      0u, VX_STATUS_OK,
                                      &group->reports[lane]);
        if (!lane) first_id = physical_id;
        else if (physical_id != first_id)
            return diagnostic_u64("scheduled", "physical-id-shared", lane,
                                  physical_id, first_id);
    }
    if (first_id == previous_physical_id)
        return diagnostic_u64("scheduled", "physical-id-duplicate", 0u,
                              first_id, previous_physical_id);
    group->physical_execution_id = first_id;
    return 1;
}

static int compare_results(const GroupResults* direct,
                           const GroupResults* scheduled, size_t lanes,
                           InputMode input_mode, ParityStats* parity) {
    const double absolute_tolerance = 1.0e-4;
    const double relative_tolerance = 1.0e-4;
    for (size_t lane = 0; lane < lanes; lane++) {
        size_t output_count = vx_result_output_count(direct->results[lane]);
        size_t scheduled_output_count =
            vx_result_output_count(scheduled->results[lane]);
        if (output_count != scheduled_output_count)
            return diagnostic_u64("parity", "output-count", lane,
                                  scheduled_output_count, output_count);
        for (size_t output = 0; output < output_count; output++) {
            VxTensorInfo direct_info = VX_TENSOR_INFO_INIT;
            VxTensorInfo scheduled_info = VX_TENSOR_INFO_INIT;
            VxReport report = VX_REPORT_INIT;
            void* direct_data = NULL;
            void* scheduled_data = NULL;
            int equal = 1;
            const char* mismatch = NULL;
            VxStatus status = vx_result_output_info(
                direct->results[lane], output, &direct_info, &report);
            if (status != VX_STATUS_OK)
                return diagnostic_failure("parity", "direct-output-info",
                                          lane, output, (int)status, &report);
            status = vx_result_output_info(
                scheduled->results[lane], output, &scheduled_info, &report);
            if (status != VX_STATUS_OK)
                return diagnostic_failure("parity", "scheduled-output-info",
                                          lane, output, (int)status, &report);
            if (!direct_info.name || !scheduled_info.name)
                return diagnostic_failure("parity", "output-name-null", lane,
                                          output, VX_STATUS_OK, NULL);
            if (strcmp(direct_info.name, scheduled_info.name))
                return diagnostic_failure("parity", "output-name", lane,
                                          output, VX_STATUS_OK, NULL);
            if (direct_info.dtype != scheduled_info.dtype)
                return diagnostic_failure("parity", "output-dtype", lane,
                                          output, VX_STATUS_OK, NULL);
            if (direct_info.rank != scheduled_info.rank)
                return diagnostic_failure("parity", "output-rank", lane,
                                          output, VX_STATUS_OK, NULL);
            if (direct_info.byte_size != scheduled_info.byte_size)
                return diagnostic_failure("parity", "output-byte-size", lane,
                                          output, VX_STATUS_OK, NULL);
            if (memcmp(direct_info.shape, scheduled_info.shape,
                       sizeof(direct_info.shape)))
                return diagnostic_failure("parity", "output-shape", lane,
                                          output, VX_STATUS_OK, NULL);
            direct_data =
                malloc(direct_info.byte_size ? direct_info.byte_size : 1u);
            scheduled_data =
                malloc(scheduled_info.byte_size ? scheduled_info.byte_size
                                                : 1u);
            if (!direct_data || !scheduled_data) {
                free(direct_data);
                free(scheduled_data);
                return diagnostic_failure("parity", "output-allocation", lane,
                                          output, -1, NULL);
            }
            status = vx_result_read(
                direct->results[lane], direct_info.name, direct_data,
                direct_info.byte_size, NULL, &report);
            if (status != VX_STATUS_OK) {
                free(direct_data);
                free(scheduled_data);
                return diagnostic_failure("parity", "direct-result-read",
                                          lane, output, (int)status, &report);
            }
            status = vx_result_read(
                scheduled->results[lane], scheduled_info.name, scheduled_data,
                scheduled_info.byte_size, NULL, &report);
            if (status != VX_STATUS_OK) {
                free(direct_data);
                free(scheduled_data);
                return diagnostic_failure("parity", "scheduled-result-read",
                                          lane, output, (int)status, &report);
            }
            if (direct_info.dtype == VX_DTYPE_F32) {
                size_t elements = direct_info.byte_size / sizeof(float);
                const float* left = (const float*)direct_data;
                const float* right = (const float*)scheduled_data;
                for (size_t index = 0; index < elements; index++) {
                    double a = left[index];
                    double b = right[index];
                    double absolute;
                    double relative;
                    if (!isfinite(a) || !isfinite(b)) {
                        equal = 0;
                        mismatch = "f32-nonfinite";
                        continue;
                    }
                    absolute = fabs(a - b);
                    relative =
                        absolute / fmax(fmax(fabs(a), fabs(b)), 1.0e-12);
                    if (absolute > parity->max_abs)
                        parity->max_abs = absolute;
                    if (relative > parity->max_rel)
                        parity->max_rel = relative;
                    /* Combined allclose: near-zero values are governed by
                     * atol and ordinary values receive the rtol allowance. */
                    if (absolute >
                        absolute_tolerance +
                            relative_tolerance * fmax(fabs(a), fabs(b))) {
                        equal = 0;
                        if (!mismatch) mismatch = "f32-allclose";
                    }
                }
                if (input_mode == INPUT_RECEIPT &&
                    !strcmp(direct_info.name, "slot_logits")) {
                    size_t classes = 0u;
                    size_t rows;
                    if (direct_info.rank < 2u) {
                        equal = 0;
                        if (!mismatch) mismatch = "receipt-output-rank";
                    } else {
                        classes = (size_t)
                            direct_info.shape[direct_info.rank - 1u];
                        if (classes < 2u || elements % classes) {
                            equal = 0;
                            if (!mismatch)
                                mismatch = "receipt-output-classes";
                            classes = 0u;
                        }
                    }
                    if (classes) {
                        rows = elements / classes;
                        for (size_t row = 0; row < rows; row++) {
                            size_t left_best = 0u;
                            size_t right_best = 0u;
                            for (size_t cls = 1u; cls < classes; cls++) {
                                if (left[row * classes + cls] >
                                    left[row * classes + left_best])
                                    left_best = cls;
                                if (right[row * classes + cls] >
                                    right[row * classes + right_best])
                                    right_best = cls;
                            }
                            if (left_best != right_best) {
                                equal = 0;
                                if (!mismatch)
                                    mismatch = "receipt-decoded-equality";
                            }
                        }
                        parity->receipt_decoded_outputs++;
                    }
                }
                parity->compared_elements += elements;
            } else {
                if (!dtype_size(direct_info.dtype)) {
                    free(direct_data);
                    free(scheduled_data);
                    return diagnostic_failure(
                        "parity", "unsupported-output-dtype", lane, output,
                        VX_STATUS_OK, NULL);
                }
                equal = !memcmp(direct_data, scheduled_data,
                                direct_info.byte_size);
                if (!equal) mismatch = "integer-byte-equality";
                parity->compared_elements +=
                    direct_info.byte_size / dtype_size(direct_info.dtype);
            }
            parity->output_tensors++;
            free(direct_data);
            free(scheduled_data);
            if (!equal)
                return diagnostic_failure("parity", mismatch ? mismatch :
                                          "output-equality", lane, output,
                                          VX_STATUS_OK, NULL);
        }
    }
    return 1;
}

static int double_compare(const void* left, const void* right) {
    double a = *(const double*)left;
    double b = *(const double*)right;
    return a < b ? -1 : a > b ? 1 : 0;
}

static double median(const double* values, size_t count) {
    double* copy = malloc(count * sizeof(*copy));
    double result;
    if (!copy) return -1.0;
    memcpy(copy, values, count * sizeof(*copy));
    qsort(copy, count, sizeof(*copy), double_compare);
    result = count & 1u
        ? copy[count / 2u]
        : (copy[count / 2u - 1u] + copy[count / 2u]) * 0.5;
    free(copy);
    return result;
}

static void json_string(const char* value) {
    const unsigned char* cursor = (const unsigned char*)value;
    putchar('"');
    while (cursor && *cursor) {
        unsigned char c = *cursor++;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c >= 0x20u && c < 0x7fu) putchar((int)c);
        else printf("\\u%04x", (unsigned)c);
    }
    putchar('"');
}

static void print_samples(const double* samples, size_t count) {
    putchar('[');
    for (size_t index = 0; index < count; index++) {
        if (index) putchar(',');
        printf("%.6f", samples[index]);
    }
    putchar(']');
}

static void print_u64_values(const uint64_t* values, size_t count) {
    putchar('[');
    for (size_t index = 0; index < count; index++) {
        if (index) putchar(',');
        printf("%" PRIu64, values[index]);
    }
    putchar(']');
}

static void print_shapes(const Fixture* fixture) {
    putchar('[');
    for (size_t input = 0; input < fixture->input_count; input++) {
        const VxTensorBinding* binding = &fixture->bindings[input];
        if (input) putchar(',');
        printf("{\"name\":");
        json_string(binding->name);
        printf(",\"dtype\":%" PRId32 ",\"shape\":[", binding->dtype);
        for (uint32_t axis = 0; axis < binding->rank; axis++) {
            if (axis) putchar(',');
            printf("%" PRId64, binding->shape[axis]);
        }
        printf("]}");
    }
    putchar(']');
}

int main(int argc, char** argv) {
    Options options;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    Fixture fixture = {0};
    double* direct_samples = NULL;
    double* scheduled_samples = NULL;
    uint64_t* scheduled_physical_ids = NULL;
    uint64_t previous_physical_id = 0;
    ParityStats parity = {0};
    BackendExecutionEvidence backend_evidence = {0};
    char proof_identity[256] = {0};
    uint64_t batch_axis = 0;
    uint64_t batch_min = 0;
    uint64_t batch_max = 0;
    uint64_t batch_multiple = 0;
    uint64_t route_nodes = 0;
    double direct_median = -1.0;
    double scheduled_median = -1.0;
    const char* backends[1];
    int exit_code = 1;

    if (!parse_options(argc, argv, &options)) {
        fprintf(stderr,
                "usage: %s --case-id TOKEN --graph PATH --weight PATH "
                "[--weight PATH ...] --backend vulkan|opengl|cuda "
                "--input-mode receipt|vqa-encoder|vqa-decoder "
                "[--lane-file PATH ...] --batch-size N --warmup N --repeat N "
                "--max-batch-delay-ms N [--vulkan-arena-mb N] "
                "[--cuda-device 0]\n",
                argv[0]);
        free(options.lane_files);
        return 2;
    }
    direct_samples = calloc(options.repeat, sizeof(*direct_samples));
    scheduled_samples = calloc(options.repeat, sizeof(*scheduled_samples));
    scheduled_physical_ids =
        calloc(options.repeat, sizeof(*scheduled_physical_ids));
    if (!direct_samples || !scheduled_samples || !scheduled_physical_ids)
        FAIL("timing allocation failed");
    runtime_options.execution_mode = VX_EXECUTION_MODE_SCHEDULED;
    runtime_options.cpu_threads = 1;
    runtime_options.max_scheduled_requests = options.batch_size;
    runtime_options.max_scheduled_input_bytes =
        (size_t)UINT64_C(2) << 30;
    runtime_options.max_batch_delay_milliseconds = options.batch_delay_ms;
    runtime_options.max_unconsumed_results = options.batch_size * 2u;
    runtime_options.max_unconsumed_result_bytes =
        (size_t)UINT64_C(2) << 30;
    if (vx_runtime_create(&runtime_options, &runtime, &report) != VX_STATUS_OK)
        FAIL("runtime create failed: %s %s", report.reason, report.message);
    source.graph_path = options.graph;
    source.weight_paths = options.weights;
    source.weight_path_count = options.weight_count;
    if (vx_runtime_load_model(runtime, &source, &model, &report) !=
        VX_STATUS_OK)
        FAIL("model load failed: %s %s", report.reason, report.message);
    backends[0] = options.backend;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = backends;
    policy.backend_count = 1u;
    if (vx_model_compile(model, &policy, &compiled, &report) != VX_STATUS_OK)
        FAIL("compile failed: %s %s route=%s", report.reason, report.message,
             report.route_evidence);
    if (vx_compiled_model_report(compiled, &report) != VX_STATUS_OK ||
        !strict_route(&report, options.backend) ||
        !report_string(&report, "batchProtocol", VX_BACKEND_BATCH_PROTOCOL) ||
        !report_token(report.route_evidence, "batchProof", proof_identity,
                      sizeof(proof_identity)) ||
        strncmp(proof_identity, VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL,
                strlen(VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL)) ||
        !report_u64(&report, "batchAxis", &batch_axis) || batch_axis != 0u ||
        !report_u64(&report, "batchMin", &batch_min) || batch_min != 1u ||
        !report_u64(&report, "batchMax", &batch_max) ||
        batch_max < options.batch_size ||
        !report_u64(&report, "batchMultiple", &batch_multiple) ||
        !batch_multiple || options.batch_size % batch_multiple ||
        !report_u64(&report, "nodes", &route_nodes) || !route_nodes)
        FAIL("compiled public batch contract is missing or incompatible");
    if (!fixture_create(&options, compiled, &fixture, &report))
        FAIL("fixture creation/min-domain validation failed: %s %s",
             report.reason, report.message);

    for (size_t iteration = 0;
         iteration < options.warmup + options.repeat; iteration++) {
        GroupResults direct = {0};
        GroupResults scheduled = {0};
        int direct_first = (iteration & 1u) == 0u;
        int measured = iteration >= options.warmup;
        int ok;
        if (direct_first) {
            ok = run_direct_group(runtime, compiled, &fixture,
                                  &options, measured, &backend_evidence,
                                  &direct) &&
                 run_scheduled_group(runtime, compiled, &fixture,
                                     &options, measured, &backend_evidence,
                                     proof_identity,
                                     previous_physical_id, &scheduled);
        } else {
            ok = run_scheduled_group(runtime, compiled, &fixture,
                                     &options, measured, &backend_evidence,
                                     proof_identity,
                                     previous_physical_id, &scheduled) &&
                 run_direct_group(runtime, compiled, &fixture,
                                  &options, measured, &backend_evidence,
                                  &direct);
        }
        if (!ok ||
            !compare_results(&direct, &scheduled, fixture.lane_count,
                             options.input_mode,
                             &parity)) {
            group_clear(&direct, fixture.lane_count);
            group_clear(&scheduled, fixture.lane_count);
            FAIL("execution, public physical proof, or lane parity failed "
                 "at group %zu", iteration);
        }
        previous_physical_id = scheduled.physical_execution_id;
        if (iteration >= options.warmup) {
            size_t sample = iteration - options.warmup;
            direct_samples[sample] = direct.elapsed_ms;
            scheduled_samples[sample] = scheduled.elapsed_ms;
            scheduled_physical_ids[sample] =
                scheduled.physical_execution_id;
        }
        group_clear(&direct, fixture.lane_count);
        group_clear(&scheduled, fixture.lane_count);
    }
    if (options.input_mode == INPUT_RECEIPT &&
        !parity.receipt_decoded_outputs)
        FAIL("receipt output did not expose slot_logits for decoded parity");
    direct_median = median(direct_samples, options.repeat);
    scheduled_median = median(scheduled_samples, options.repeat);
    if (!isfinite(direct_median) || direct_median < 0.0 ||
        !isfinite(scheduled_median) || scheduled_median < 0.0)
        FAIL("median calculation failed");
    if (!strcmp(options.backend, "vulkan") &&
        !backend_evidence.vulkan_stage_coherent_set)
        FAIL("Vulkan execution placement proof was not observed");

    printf("{\"record\":\"%s\",\"caseId\":", BENCHMARK_RECORD);
    json_string(options.case_id);
    printf(",\"backend\":");
    json_string(options.backend);
    printf(",\"inputMode\":");
    json_string(options.input_mode_name);
    printf(",\"status\":\"pass\",\"logicalLanes\":%zu,"
           "\"uniqueInputClasses\":%zu,\"fixtureReuse\":",
           fixture.lane_count, fixture.unique_input_classes);
    json_string(fixture.reuse);
    printf(",\"syntheticFixture\":{");
    if (options.input_mode == INPUT_RECEIPT) {
        printf("\"kind\":\"private-f32-lanes\",\"seed\":null");
    } else if (options.input_mode == INPUT_VQA_ENCODER) {
        printf("\"kind\":\"component-only-graph-domain-minimum\","
               "\"seed\":%" PRIu32 ",\"Q\":1,\"M\":211",
               FIXTURE_SEED);
    } else {
        printf("\"kind\":\"component-only-graph-domain-minimum\","
               "\"seed\":%" PRIu32 ",\"M\":211,\"P\":1,\"R\":2",
               FIXTURE_SEED);
    }
    printf("},\"resolvedInputs\":");
    print_shapes(&fixture);
    printf(",\"runtime\":{\"executionMode\":\"scheduled\",\"cpuThreads\":1,"
           "\"freshness\":\"ALL\","
           "\"maxScheduledRequests\":%zu,"
           "\"maxScheduledInputBytes\":2147483648,"
           "\"maxUnconsumedResults\":%zu,"
           "\"maxUnconsumedResultBytes\":2147483648,"
           "\"requestedMaxBatchDelayMs\":%" PRIu32 ","
           "\"effectiveMaxBatchDelayMs\":%" PRIu32 ","
           "\"vulkanArenaMB\":",
           runtime_options.max_scheduled_requests,
           runtime_options.max_unconsumed_results, options.batch_delay_ms,
           runtime_options.max_batch_delay_milliseconds);
    if (options.vulkan_arena_mb)
        printf("%" PRIu32, options.vulkan_arena_mb);
    else
        printf("null");
    printf("},\"batchContract\":{\"protocol\":");
    json_string(VX_BACKEND_BATCH_PROTOCOL);
    printf(",\"axis\":%" PRIu64 ",\"symbol\":",
           batch_axis);
    json_string(fixture.batch_symbol);
    printf(",\"min\":%" PRIu64
           ",\"max\":%" PRIu64 ",\"multiple\":%" PRIu64
           ",\"proofIdentity\":",
           batch_min, batch_max, batch_multiple);
    json_string(proof_identity);
    printf("},\"strictRoute\":{\"routeAttested\":true,"
           "\"provider\":\"builtin:%s\","
           "\"nodes\":%" PRIu64 ",\"selected\":%" PRIu64
           ",\"fallback\":0,\"missing\":0,"
           "\"operatorFallbackEvidence\":\"operator=none\"",
           options.backend, route_nodes, route_nodes);
    printf("},\"measurement\":{\"warmupGroups\":%zu,"
           "\"repeatGroups\":%zu,"
           "\"order\":\"alternating-direct-scheduled\","
           "\"clock\":\"CLOCK_MONOTONIC\","
           "\"timingBoundary\":\"direct:first-vx_runtime_run-to-Nth-"
           "vx_runtime_run-owned-result;scheduled:first-vx_runtime_submit-"
           "through-all-waits-to-Nth-vx_request_result-owned-result;"
           "read-parity-release-untimed\"},"
           "\"direct\":{\"logicalLanes\":%zu,\"physicalBatchSize\":1,"
           "\"trueBackendInvocationsPerGroup\":%zu,\"samplesMs\":",
           options.warmup, options.repeat, fixture.lane_count,
           fixture.lane_count);
    print_samples(direct_samples, options.repeat);
    printf(",\"medianMs\":%.6f},"
           "\"scheduled\":{\"logicalLanes\":%zu,"
           "\"physicalBatchSize\":%zu,"
           "\"trueBackendInvocationsPerGroup\":1,"
           "\"allLanesSharedPhysicalExecutionId\":true,"
           "\"physicalExecutionIdChangesAcrossGroups\":true,"
           "\"physicalExecutionIdScope\":\"process-local-runtime-counter\","
           "\"physicalExecutionIds\":",
           direct_median, fixture.lane_count, fixture.lane_count);
    print_u64_values(scheduled_physical_ids, options.repeat);
    printf(",\"samplesMs\":");
    print_samples(scheduled_samples, options.repeat);
    printf(",\"medianMs\":%.6f},"
           "\"parity\":{\"status\":\"pass\","
           "\"absoluteTolerance\":0.0001,\"relativeTolerance\":0.0001,"
           "\"combinedAllclose\":true,\"nonfiniteRejected\":true,"
           "\"integerOutputsByteExact\":true,"
           "\"receiptDecodedEquality\":%s,"
           "\"comparedElements\":%" PRIu64 ","
           "\"outputTensors\":%" PRIu64 ","
           "\"maxAbs\":%.9g,\"maxRel\":%.9g},"
           "\"publicEvidence\":{"
           "\"source\":\"VxReport.route_evidence\","
           "\"compiledContractExact\":true,"
           "\"scheduledLaneTokensExact\":true,"
           "\"directBatchTokensAbsent\":true,"
           "\"backendExecutionProofExact\":true,"
           "\"vulkanComputeDeviceLocal\":",
           scheduled_median,
           options.input_mode == INPUT_RECEIPT ? "true" : "null",
           parity.compared_elements, parity.output_tensors,
           parity.max_abs, parity.max_rel);
    if (!strcmp(options.backend, "vulkan"))
        printf("true,\"vulkanArenaBytes\":%" PRIu64
               ",\"vulkanStagingBytes\":%" PRIu64
               ",\"vulkanStagingHostCoherent\":%s,"
               "\"cudaGraphReplayMeasured\":null}}\n",
               backend_evidence.vulkan_arena_bytes,
               backend_evidence.vulkan_staging_bytes,
               backend_evidence.vulkan_stage_coherent ? "true" : "false");
    else if (!strcmp(options.backend, "cuda"))
        printf("null,\"vulkanArenaBytes\":null,"
               "\"vulkanStagingBytes\":null,"
               "\"vulkanStagingHostCoherent\":null,"
               "\"cudaGraphReplayMeasured\":true}}\n");
    else
        printf("null,\"vulkanArenaBytes\":null,"
               "\"vulkanStagingBytes\":null,"
               "\"vulkanStagingHostCoherent\":null,"
               "\"cudaGraphReplayMeasured\":null}}\n");
    fflush(stdout);
    exit_code = 0;

cleanup:
    fixture_clear(&fixture);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    if (runtime) {
        (void)vx_runtime_close(runtime, &report);
        vx_runtime_release(runtime);
    }
    free(direct_samples);
    free(scheduled_samples);
    free(scheduled_physical_ids);
    free(options.lane_files);
    return exit_code;
}
