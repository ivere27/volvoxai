#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "vx_training_lifecycle.h"

#include "cJSON.h"
#include "engine_core.h"
#include "json_validation.h"
#include "public_api_internal.h"
#include "runtime_state.h"
#include "training_core.h"

#include <limits.h>
#include "vx_thread.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#elif !defined(__wasm__)
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

#define VX_PTQ_MAX_SAFE_INTEGER UINT64_C(9007199254740991)

typedef struct VxOwnedPTQObserver {
    char name[VX_PTQ_NAME_CAPACITY];
    VxDataType dtype;
    VxPTQScheme scheme;
} VxOwnedPTQObserver;

typedef struct VxOwnedPTQSymbolCoverage {
    char name[VX_PTQ_NAME_CAPACITY];
    int64_t minimum;
    int64_t maximum;
} VxOwnedPTQSymbolCoverage;

typedef struct VxOwnedPTQActivationCoverage {
    char name[VX_PTQ_NAME_CAPACITY];
    uint64_t values;
} VxOwnedPTQActivationCoverage;

typedef struct VxOwnedPTQProfile {
    char name[VX_PTQ_NAME_CAPACITY];
    uint64_t batch_count;
    uint64_t sample_count;
    char** signatures;
    size_t signature_count;
    size_t signature_capacity;
    VxOwnedPTQSymbolCoverage* symbols;
    size_t symbol_count;
    VxOwnedPTQActivationCoverage* activations;
    size_t activation_count;
} VxOwnedPTQProfile;

struct VxPTQPlan {
    atomic_uint references;
    pthread_mutex_t mutex;
    int closed;
    VxModel* model;
    VxWeightRevisionRecord* exact_revision;
    VxEngineState* engine;
    VolvoxAIPTQPlan* core;
    char* template_graph_snapshot;
    VxOwnedPTQObserver* observers;
    size_t observer_count;
    size_t input_count;
    VxOwnedPTQProfile* profiles;
    size_t profile_count;
    char** sample_names;
    size_t sample_count;
    uint64_t represented_sample_count;
    VxRevisionInfo revision;
    uint64_t runtime_id;
    uint64_t model_id;
};

static char* ptq_string_copy(const char* value) {
    size_t length;
    char* copy;
    if (!value) return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1u);
    if (copy) memcpy(copy, value, length + 1u);
    return copy;
}

static int ptq_name_valid(const char* value) {
    return value && value[0] && strlen(value) < VX_PTQ_NAME_CAPACITY;
}

static int ptq_profile_name_valid(const char* value) {
    size_t length;
    if (!value || !(('A' <= value[0] && value[0] <= 'Z') ||
                    ('a' <= value[0] && value[0] <= 'z')))
        return 0;
    length = strlen(value);
    if (!length || length > 64u) return 0;
    for (size_t index = 1; index < length; index++) {
        char character = value[index];
        if (!(('A' <= character && character <= 'Z') ||
              ('a' <= character && character <= 'z') ||
              ('0' <= character && character <= '9') ||
              character == '.' || character == '_' || character == '-'))
            return 0;
    }
    return 1;
}

static int ptq_report_argument_valid(const VxReport* report) {
    return !report || report->struct_size == sizeof(*report);
}

static void ptq_report(VxPTQPlan* plan,
                       VxReport* report,
                       VxStatus status,
                       VxStage stage,
                       VxOperationCode reason,
                       const char* message) {
    size_t struct_size;
    if (!report || report->struct_size != sizeof(*report)) return;
    struct_size = report->struct_size;
    memset(report, 0, sizeof(*report));
    report->struct_size = struct_size;
    report->status = status;
    report->stage = stage;
    report->policy_mode = VX_BACKEND_REQUIRE;
    report->operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    if (plan) {
        report->runtime_id = plan->runtime_id;
        report->model_id = plan->model_id;
        report->graph_id = plan->revision.graph_id;
        report->graph_revision = plan->revision.graph_revision;
        report->weight_id = plan->revision.weight_id;
        report->weight_revision = plan->revision.weight_revision;
        report->adapter_id = plan->revision.adapter_id;
        report->adapter_revision = plan->revision.adapter_revision;
#if defined(__wasm__)
        snprintf(report->backend, sizeof(report->backend), "%s", "wasm");
        snprintf(report->route_evidence, sizeof(report->route_evidence),
                 "%s", "ptq-calibration:wasm-required");
#else
        snprintf(report->backend, sizeof(report->backend), "%s", "cpu");
        snprintf(report->route_evidence, sizeof(report->route_evidence),
                 "%s", "ptq-calibration:cpu-required");
#endif
        report->route_attested = 1;
    }
    report->code = reason;
    snprintf(report->message, sizeof(report->message), "%s",
             message ? message : "");
}

static int ptq_read_file(const char* path, unsigned char** bytes, size_t* size) {
    FILE* file;
    long length;
    unsigned char* data;
    if (!path || !path[0] || !bytes || !size) return -1;
    *bytes = NULL;
    *size = 0;
    file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END) != 0) {
        if (file) fclose(file);
        return -1;
    }
    length = ftell(file);
    if (length < 0 || (unsigned long)length >= SIZE_MAX ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return -1;
    }
    data = (unsigned char*)malloc((size_t)length + 1u);
    if (!data) {
        fclose(file);
        return -2;
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
    data[length] = 0;
    *bytes = data;
    *size = (size_t)length;
    return 0;
}

static int ptq_graph_document_valid(const unsigned char* bytes, size_t size) {
    cJSON* root;
    const cJSON* format;
    if (!bytes || !size || bytes[size] != 0) return 0;
    root = cJSON_ParseWithLength((const char*)bytes, size);
    if (!cJSON_IsObject(root) || !vx_json_object_keys_unique_recursive(root)) {
        cJSON_Delete(root);
        return 0;
    }
    format = cJSON_GetObjectItemCaseSensitive(root, "format");
    if (!cJSON_IsString(format) || !format->valuestring ||
        strcmp(format->valuestring, "volvox-graph/v1")) {
        cJSON_Delete(root);
        return 0;
    }
    cJSON_Delete(root);
    return 1;
}

static int ptq_snapshot_template(const VxPTQPlanOptions* options, char** output_path) {
    unsigned char* bytes = NULL;
    size_t size = 0;
    char path[PATH_MAX] = {0};
    FILE* output = NULL;
    char* owned = NULL;
    int read_status;
    int write_failed = 0;
    if (!output_path) return -1;
    *output_path = NULL;
    if (options->template_graph_size) {
        size = options->template_graph_size;
        if (size == SIZE_MAX) return -2;
        bytes = (unsigned char*)malloc(size + 1u);
        if (!bytes) return -2;
        memcpy(bytes, options->template_graph, size);
        bytes[size] = 0;
        read_status = 0;
    } else {
        read_status = ptq_read_file(options->template_graph_path, &bytes, &size);
    }
    if (read_status != 0) return read_status;
    if (!ptq_graph_document_valid(bytes, size)) {
        free(bytes);
        return -3;
    }
#ifdef _WIN32
    {
        char directory[MAX_PATH];
        DWORD length = GetTempPathA((DWORD)sizeof(directory), directory);
        if (!length || length >= sizeof(directory) ||
            !GetTempFileNameA(directory, "vxp", 0, path)) {
            free(bytes);
            return -1;
        }
        output = fopen(path, "wb");
    }
#elif defined(__wasm__)
    {
        static uint64_t sequence;
        if (sequence == UINT64_MAX) { free(bytes); return -1; }
        int written = snprintf(path, sizeof(path), "/ptq/%llu", ++sequence);
        if (written < 0 || (size_t)written >= sizeof(path)) { free(bytes); return -1; }
        output = fopen(path, "wb");
    }
#else
    {
        const char* directory = getenv("TMPDIR");
        int descriptor;
        int written;
        if (!directory || !directory[0]) directory = "/tmp";
        written = snprintf(path, sizeof(path), "%s%svolvoxai-ptq-XXXXXX",
                           directory,
                           directory[strlen(directory) - 1u] == '/' ? "" : "/");
        if (written < 0 || (size_t)written >= sizeof(path)) {
            free(bytes);
            return -1;
        }
        descriptor = mkstemp(path);
        if (descriptor >= 0) output = fdopen(descriptor, "wb");
        if (descriptor >= 0 && !output) close(descriptor);
    }
#endif
    if (!output) {
        write_failed = 1;
    } else {
        if (size && fwrite(bytes, 1, size, output) != size) write_failed = 1;
        if (fflush(output) != 0) write_failed = 1;
        if (fclose(output) != 0) write_failed = 1;
        output = NULL;
    }
    if (write_failed) {
        if (path[0]) (void)remove(path);
        free(bytes);
        return -1;
    }
    free(bytes);
    owned = ptq_string_copy(path);
    if (!owned) {
        (void)remove(path);
        return -2;
    }
    *output_path = owned;
    return 0;
}

static void ptq_snapshot_release(char* path) {
    if (!path) return;
    (void)remove(path);
    free(path);
}

static int ptq_output_graph_name_valid(const char* path) {
    const char* basename;
    const char* slash;
    const char* backslash;
    if (!path || !path[0]) return 0;
    slash = strrchr(path, '/');
    backslash = strrchr(path, '\\');
    basename = slash && (!backslash || slash > backslash) ? slash + 1u
        : backslash ? backslash + 1u : path;
    return !strcmp(basename, "graph.json");
}

static int ptq_weights_name_valid(const char* path) {
    static const char suffix[] = ".safetensors";
    size_t length;
    if (!path || !path[0]) return 0;
    length = strlen(path);
    return length > sizeof(suffix) - 1u &&
           !strcmp(path + length - (sizeof(suffix) - 1u), suffix);
}

static size_t ptq_parent_span(const char* path) {
    const char* slash = strrchr(path, '/');
    const char* backslash = strrchr(path, '\\');
    const char* separator = slash && (!backslash || slash > backslash)
        ? slash : backslash;
    return separator ? (size_t)(separator - path) + 1u : 0u;
}

static int ptq_paths_are_siblings(const char* left, const char* right) {
    size_t left_parent;
    size_t right_parent;
    if (!left || !right) return 0;
    left_parent = ptq_parent_span(left);
    right_parent = ptq_parent_span(right);
    return left_parent == right_parent &&
           !memcmp(left, right, left_parent);
}

static VxStatus ptq_current_locked(VxPTQPlan* plan) {
    VxReport ignored = VX_REPORT_INIT;
    if (!plan || plan->closed || !plan->engine || !plan->core)
        return VX_STATUS_HANDLE_DISPOSED;
    return vx_model_internal_validate_authoring_revision(
        plan->model, plan->exact_revision, plan->revision.adapter_id,
        plan->revision.adapter_revision, &ignored);
}

static void ptq_samples_release(char** names, size_t count) {
    if (!names) return;
    for (size_t index = 0; index < count; index++) free(names[index]);
    free(names);
}

static int ptq_sample_exists(const VxPTQPlan* plan, const char* name) {
    for (size_t index = 0; index < plan->sample_count; index++) {
        if (!strcmp(plan->sample_names[index], name)) return 1;
    }
    return 0;
}

static void ptq_profiles_release(VxOwnedPTQProfile* profiles, size_t count) {
    if (!profiles) return;
    for (size_t index = 0; index < count; index++) {
        for (size_t signature = 0;
             signature < profiles[index].signature_count; signature++)
            free(profiles[index].signatures[signature]);
        free(profiles[index].signatures);
        free(profiles[index].symbols);
        free(profiles[index].activations);
    }
    free(profiles);
}

static VxOwnedPTQProfile* ptq_profile_find(VxPTQPlan* plan,
                                           const char* name) {
    if (!plan || !name) return NULL;
    for (size_t index = 0; index < plan->profile_count; index++)
        if (!strcmp(plan->profiles[index].name, name))
            return &plan->profiles[index];
    return NULL;
}

static VxStatus ptq_profiles_initialize(
        VxPTQPlan* plan,
        const char* const* names,
        size_t count) {
    VxOwnedPTQSymbolCoverage* symbols = NULL;
    size_t symbol_capacity;
    size_t symbol_count = 0;
    VxStatus status = VX_STATUS_OK;
    if (!plan || !names || !count || count > (size_t)INT32_MAX)
        return VX_STATUS_INVALID_ARGUMENT;
    plan->input_count = vx_model_internal_input_count(plan->model);
    if (!plan->input_count) return VX_STATUS_INVALID_GRAPH;
    if (plan->input_count > SIZE_MAX / VX_MAX_TENSOR_RANK ||
        plan->input_count * VX_MAX_TENSOR_RANK >
            SIZE_MAX / sizeof(*symbols)) return VX_STATUS_OUT_OF_MEMORY;
    symbol_capacity = plan->input_count * VX_MAX_TENSOR_RANK;
    symbols = (VxOwnedPTQSymbolCoverage*)calloc(
        symbol_capacity, sizeof(*symbols));
    if (!symbols) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t input = 0; input < plan->input_count; input++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        if (vx_model_internal_input_spec(plan->model, input, &spec) !=
            VX_STATUS_OK) {
            status = VX_STATUS_INVALID_GRAPH;
            goto done;
        }
        for (uint32_t axis = 0; axis < spec.rank; axis++) {
            const char* symbol = spec.dimensions[axis].symbol;
            int duplicate = 0;
            if (spec.dimensions[axis].kind != VX_DIMENSION_SYMBOLIC ||
                !ptq_name_valid(symbol)) continue;
            for (size_t prior = 0; prior < symbol_count; prior++)
                if (!strcmp(symbols[prior].name, symbol)) duplicate = 1;
            if (duplicate) continue;
            if (symbol_count >= symbol_capacity) {
                status = VX_STATUS_INVALID_GRAPH;
                goto done;
            }
            snprintf(symbols[symbol_count].name,
                     sizeof(symbols[symbol_count].name), "%s", symbol);
            symbols[symbol_count].minimum = INT64_MAX;
            symbols[symbol_count].maximum = 0;
            symbol_count++;
        }
    }
    plan->profiles = (VxOwnedPTQProfile*)calloc(count, sizeof(*plan->profiles));
    if (!plan->profiles) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    plan->profile_count = count;
    for (size_t index = 0; index < count; index++) {
        VxOwnedPTQProfile* profile = &plan->profiles[index];
        if (!ptq_profile_name_valid(names[index])) {
            status = VX_STATUS_INVALID_ARGUMENT;
            goto done;
        }
        for (size_t prior = 0; prior < index; prior++)
            if (!strcmp(names[prior], names[index])) {
                status = VX_STATUS_INVALID_ARGUMENT;
                goto done;
            }
        snprintf(profile->name, sizeof(profile->name), "%s", names[index]);
        if (symbol_count) {
            profile->symbols = (VxOwnedPTQSymbolCoverage*)malloc(
                symbol_count * sizeof(*profile->symbols));
            if (!profile->symbols) {
                status = VX_STATUS_OUT_OF_MEMORY;
                goto done;
            }
            memcpy(profile->symbols, symbols,
                   symbol_count * sizeof(*profile->symbols));
        }
        profile->symbol_count = symbol_count;
        if (plan->observer_count) {
            profile->activations = (VxOwnedPTQActivationCoverage*)calloc(
                plan->observer_count, sizeof(*profile->activations));
            if (!profile->activations) {
                status = VX_STATUS_OUT_OF_MEMORY;
                goto done;
            }
            profile->activation_count = plan->observer_count;
            for (size_t observer = 0; observer < plan->observer_count;
                 observer++)
                snprintf(profile->activations[observer].name,
                         sizeof(profile->activations[observer].name), "%s",
                         plan->observers[observer].name);
        }
    }
done:
    free(symbols);
    return status;
}

static void ptq_owner_destroy(VxPTQPlan* plan) {
    if (!plan) return;
    (void)vx_ptq_plan_close(plan, NULL);
    vx_model_release(plan->model);
    pthread_mutex_destroy(&plan->mutex);
    free(plan);
}

VxStatus vx_model_create_ptq_plan(VxModel* model,
                                  const VxPTQPlanOptions* options,
                                  VxPTQPlan** out_plan,
                                  VxReport* report) {
    VxPTQPlan* plan = NULL;
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    char* template_snapshot = NULL;
    int snapshot_status;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !options || options->struct_size != sizeof(*options) ||
        !out_plan ||
        (!!(options->template_graph_path && options->template_graph_path[0]) ==
         !!options->template_graph_size) ||
        (options->template_graph_size && !options->template_graph) ||
        !options->profile_names ||
        !options->profile_count || options->profile_count > (size_t)INT32_MAX ||
        !options->observers ||
        !options->observer_count || options->observer_count > (size_t)INT32_MAX ||
        (options->layer_count && !options->layers) ||
        options->layer_count > (size_t)INT32_MAX) {
        if (out_plan) *out_plan = NULL;
        ptq_report(NULL, report, status, VX_STAGE_PTQ_CREATE,
                   VX_CODE_INVALID_PTQ_PLAN, "PTQ plan options are invalid");
        return status;
    }
    *out_plan = NULL;
    snapshot_status = ptq_snapshot_template(options,
                                            &template_snapshot);
    if (snapshot_status != 0) {
        status = snapshot_status == -2 ? VX_STATUS_OUT_OF_MEMORY :
                 snapshot_status == -3 ? VX_STATUS_INVALID_GRAPH :
                                         VX_STATUS_IO_ERROR;
        ptq_report(NULL, report, status, VX_STAGE_PTQ_CREATE,
                   snapshot_status == -3 ? VX_CODE_INVALID_TEMPLATE_GRAPH :
                                           VX_CODE_TEMPLATE_SNAPSHOT_FAILED,
                   snapshot_status == -3
                       ? "template graph format must be exactly volvox-graph/v1"
                       : "template graph could not be snapshotted");
        return status;
    }
    plan = (VxPTQPlan*)calloc(1, sizeof(*plan));
    if (!plan) {
        ptq_snapshot_release(template_snapshot);
        ptq_report(NULL, report, VX_STATUS_OUT_OF_MEMORY,
                   VX_STAGE_PTQ_CREATE, VX_CODE_OUT_OF_MEMORY,
                   "PTQ plan allocation failed");
        return VX_STATUS_OUT_OF_MEMORY;
    }
    atomic_init(&plan->references, 1u);
    if (pthread_mutex_init(&plan->mutex, NULL) != 0) {
        free(plan);
        ptq_snapshot_release(template_snapshot);
        ptq_report(NULL, report, VX_STATUS_INTERNAL, VX_STAGE_PTQ_CREATE,
                   VX_CODE_MUTEX_INIT_FAILED, "PTQ plan mutex initialization failed");
        return VX_STATUS_INTERNAL;
    }
    plan->template_graph_snapshot = template_snapshot;
    plan->model = model;
    vx_model_retain(model);
    status = vx_model_internal_accept_full_owner(
        model, &plan->exact_revision, report);
    if (status != VX_STATUS_OK) goto fail;
    plan->revision = (VxRevisionInfo)VX_REVISION_INFO_INIT;
    {
        VxReport lineage = VX_REPORT_INIT;
        status = vx_model_revision_info(model, &plan->revision, &lineage);
        if (status == VX_STATUS_OK) {
            plan->runtime_id = lineage.runtime_id;
            plan->model_id = lineage.model_id;
            /* The synthetic __base__ adapter is an internal selection record,
             * not an adapter materialized by the private PTQ engine. */
            plan->revision.adapter_id = 0u;
            plan->revision.adapter_revision = 0u;
        }
    }
    if (status != VX_STATUS_OK) goto fail;
    status = vx_model_internal_validate_authoring_revision(
        model, plan->exact_revision, plan->revision.adapter_id,
        plan->revision.adapter_revision, report);
    if (status != VX_STATUS_OK) goto fail;
    /* Observers read intermediates after the complete forward pass. Inference
     * fusion and arena reuse would erase those values before calibration. */
    const VolvoxAIEngineShapePolicy shape_policy = {.retain_activations = 1};
    status = vx_model_internal_create_authoring_engine(
        model, plan->exact_revision, VX_PORTABLE_BACKEND_KIND, &shape_policy,
        &plan->engine, report);
    if (status != VX_STATUS_OK) goto fail;
    if (plan->engine->weight_file_count != 1) {
        status = VX_STATUS_INVALID_GRAPH;
        goto fail;
    }
    plan->observers = (VxOwnedPTQObserver*)calloc(
        options->observer_count, sizeof(*plan->observers));
    if (!plan->observers) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto fail;
    }
    {
        VxEngineStateScope scope = vx_engine_state_scope_enter(plan->engine);
        plan->core = volvoxai_ptq_plan_create();
        if (!plan->core) status = VX_STATUS_INVALID_GRAPH;
        for (size_t index = 0; status == VX_STATUS_OK &&
                               index < options->observer_count; index++) {
            const VxPTQObserverSpec* source = &options->observers[index];
            volvoxai_ptq_tensor_spec_t target = VOLVOXAI_PTQ_TENSOR_SPEC_INIT;
            if (source->struct_size != sizeof(*source) ||
                !ptq_name_valid(source->tensor_name) ||
                (source->dtype != VX_DTYPE_I8 && source->dtype != VX_DTYPE_U8) ||
                (source->scheme != VX_PTQ_SCHEME_SYMMETRIC &&
                 source->scheme != VX_PTQ_SCHEME_ASYMMETRIC)) {
                status = VX_STATUS_INVALID_ARGUMENT;
                break;
            }
            snprintf(plan->observers[index].name,
                     sizeof(plan->observers[index].name), "%s",
                     source->tensor_name);
            plan->observers[index].dtype = source->dtype;
            plan->observers[index].scheme = source->scheme;
            target.tensor_name = source->tensor_name;
            target.dtype = source->dtype;
            target.scheme = source->scheme;
            target.quantized_tensor_name = source->quantized_tensor_name;
            if (volvoxai_ptq_plan_add_tensor(plan->core, &target) != 0)
                status = VX_STATUS_INVALID_GRAPH;
        }
        for (size_t index = 0; status == VX_STATUS_OK &&
                               index < options->layer_count; index++) {
            const VxPTQLayerSpec* source = &options->layers[index];
            volvoxai_ptq_layer_spec_t target = VOLVOXAI_PTQ_LAYER_SPEC_INIT;
            if (source->struct_size != sizeof(*source) ||
                source->mode != VX_PTQ_MODE_W8A8 ||
                (source->kind != VX_PTQ_LAYER_QLINEAR &&
                 source->kind != VX_PTQ_LAYER_QCONV2D)) {
                status = VX_STATUS_INVALID_ARGUMENT;
                break;
            }
            target.kind = source->kind == VX_PTQ_LAYER_QLINEAR
                ? VX_PTQ_LAYER_QLINEAR : VX_PTQ_LAYER_QCONV2D;
            target.node_index = source->node_index;
            target.weight_axis = source->weight_axis;
            target.input_tensor_name = source->input_tensor_name;
            target.output_tensor_name = source->output_tensor_name;
            target.source_weight_name = source->source_weight_name;
            target.packed_weight_name = source->packed_weight_name;
            target.source_bias_name = source->source_bias_name;
            target.packed_bias_name = source->packed_bias_name;
            target.node_id = source->node_id;
            if (volvoxai_ptq_plan_add_layer(plan->core, &target) != 0)
                status = VX_STATUS_INVALID_GRAPH;
        }
        vx_engine_state_scope_leave(scope);
    }
    if (status != VX_STATUS_OK) goto fail;
    plan->observer_count = options->observer_count;
    status = ptq_profiles_initialize(
        plan, options->profile_names, options->profile_count);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_model_internal_validate_authoring_revision(
        model, plan->exact_revision, plan->revision.adapter_id,
        plan->revision.adapter_revision, report);
    if (status != VX_STATUS_OK) goto fail;
    *out_plan = plan;
    ptq_report(plan, report, VX_STATUS_OK, VX_STAGE_PTQ_CREATE, VX_CODE_NONE,
               "W8A8 PTQ plan created with a private exact-revision CPU engine");
    return VX_STATUS_OK;

fail:
    ptq_report(
        plan, report, status, VX_STAGE_PTQ_CREATE,
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
        status == VX_STATUS_INVALID_ARGUMENT ? VX_CODE_INVALID_PTQ_SPEC :
        plan && plan->engine && plan->engine->weight_file_count != 1
            ? VX_CODE_PTQ_SINGLE_SHARD_REQUIRED : VX_CODE_PTQ_PLAN_CREATE_FAILED,
        status == VX_STATUS_INVALID_GRAPH && plan && plan->engine &&
                plan->engine->weight_file_count != 1
            ? "PTQ authoring currently requires exactly one source safetensors shard"
            : "PTQ plan creation failed validation");
    ptq_owner_destroy(plan);
    return status;
}

void vx_ptq_plan_retain(VxPTQPlan* plan) {
    if (plan)
        atomic_fetch_add_explicit(&plan->references, 1u, memory_order_relaxed);
}

VxStatus vx_ptq_plan_close(VxPTQPlan* plan, VxReport* report) {
    VolvoxAIPTQPlan* core;
    VxEngineState* engine;
    VxWeightRevisionRecord* revision;
    char* template_snapshot;
    VxOwnedPTQObserver* observers;
    VxOwnedPTQProfile* profiles;
    size_t profile_count;
    char** sample_names;
    size_t sample_count;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan) {
        ptq_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_CLOSE, VX_CODE_INVALID_PTQ_PLAN, "PTQ plan is NULL");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    if (plan->closed) {
        pthread_mutex_unlock(&plan->mutex);
        ptq_report(plan, report, VX_STATUS_OK, VX_STAGE_PTQ_CLOSE, VX_CODE_NONE,
                   "PTQ plan already closed");
        return VX_STATUS_OK;
    }
    plan->closed = 1;
    core = plan->core;
    engine = plan->engine;
    revision = plan->exact_revision;
    template_snapshot = plan->template_graph_snapshot;
    observers = plan->observers;
    profiles = plan->profiles;
    profile_count = plan->profile_count;
    sample_names = plan->sample_names;
    sample_count = plan->sample_count;
    plan->core = NULL;
    plan->engine = NULL;
    plan->exact_revision = NULL;
    plan->template_graph_snapshot = NULL;
    plan->observers = NULL;
    plan->observer_count = 0;
    plan->profiles = NULL;
    plan->profile_count = 0;
    plan->sample_names = NULL;
    plan->sample_count = 0;
    pthread_mutex_unlock(&plan->mutex);
    if (engine && core) {
        VxEngineStateScope scope = vx_engine_state_scope_enter(engine);
        volvoxai_ptq_plan_destroy(core);
        vx_engine_state_scope_leave(scope);
    }
    vx_model_internal_destroy_authoring_engine(engine);
    vx_model_internal_release_weight_revision(revision);
    ptq_snapshot_release(template_snapshot);
    free(observers);
    ptq_profiles_release(profiles, profile_count);
    ptq_samples_release(sample_names, sample_count);
    ptq_report(plan, report, VX_STATUS_OK, VX_STAGE_PTQ_CLOSE, VX_CODE_NONE,
               "PTQ plan and private calibration state released");
    return VX_STATUS_OK;
}

void vx_ptq_plan_release(VxPTQPlan* plan) {
    if (!plan || atomic_fetch_sub_explicit(&plan->references, 1u,
                                           memory_order_acq_rel) != 1u)
        return;
    ptq_owner_destroy(plan);
}

size_t vx_ptq_plan_input_count(VxPTQPlan* plan) {
    size_t count = 0;
    if (!plan) return 0;
    pthread_mutex_lock(&plan->mutex);
    if (ptq_current_locked(plan) == VX_STATUS_OK)
        count = vx_model_internal_input_count(plan->model);
    pthread_mutex_unlock(&plan->mutex);
    return count;
}

VxStatus vx_ptq_plan_input_spec(VxPTQPlan* plan,
                                size_t index,
                                VxTensorSpec* spec,
                                VxReport* report) {
    VxStatus status;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !spec || spec->struct_size != sizeof(*spec)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, VX_CODE_INVALID_INPUT_QUERY,
                   "PTQ plan or tensor spec buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK)
        status = vx_model_internal_input_spec(plan->model, index, spec);
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_NOT_FOUND ? VX_CODE_INPUT_NOT_FOUND :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_INPUT_QUERY_FAILED,
               status == VX_STATUS_OK ? "logical PTQ input spec returned" :
                                        "logical PTQ input spec is unavailable");
    return status;
}

static int ptq_coverage_complete_locked(const VxPTQPlan* plan) {
    if (!plan || !plan->profile_count) return 0;
    for (size_t index = 0; index < plan->profile_count; index++)
        if (!plan->profiles[index].batch_count ||
            !plan->profiles[index].sample_count) return 0;
    return 1;
}

static void ptq_fill_info_locked(const VxPTQPlan* plan,
                                 VxPTQPlanInfo* info) {
    size_t covered = 0;
    size_t struct_size = info->struct_size;
    for (size_t index = 0; index < plan->profile_count; index++)
        if (plan->profiles[index].batch_count &&
            plan->profiles[index].sample_count) covered++;
    memset(info, 0, sizeof(*info));
    info->struct_size = struct_size;
    info->calibration_batches = plan->sample_count;
    info->calibration_samples = plan->represented_sample_count;
    info->tensor_count = plan->observer_count;
    info->profile_count = plan->profile_count;
    info->covered_profile_count = covered;
    info->coverage_complete = covered == plan->profile_count;
    info->revision = plan->revision;
}

static const VxTensorBinding* ptq_batch_binding(
        const VxPTQCalibrationBatch* batch,
        const char* name) {
    for (size_t index = 0; index < batch->input_count; index++)
        if (!strcmp(batch->inputs[index].name, name)) return &batch->inputs[index];
    return NULL;
}

static int ptq_profile_signature_exists(const VxOwnedPTQProfile* profile,
                                        const char* signature) {
    for (size_t index = 0; index < profile->signature_count; index++)
        if (!strcmp(profile->signatures[index], signature)) return 1;
    return 0;
}

static VxStatus ptq_profile_reserve_signature(VxOwnedPTQProfile* profile) {
    char** expanded;
    size_t capacity;
    if (profile->signature_count < profile->signature_capacity)
        return VX_STATUS_OK;
    capacity = profile->signature_capacity
        ? profile->signature_capacity * 2u : 4u;
    if (capacity < profile->signature_capacity ||
        capacity > SIZE_MAX / sizeof(*expanded)) return VX_STATUS_OUT_OF_MEMORY;
    expanded = (char**)realloc(profile->signatures,
                               capacity * sizeof(*expanded));
    if (!expanded) return VX_STATUS_OUT_OF_MEMORY;
    profile->signatures = expanded;
    profile->signature_capacity = capacity;
    return VX_STATUS_OK;
}

static VxStatus ptq_activation_sizes_locked(
        VxPTQPlan* plan,
        const VxOwnedPTQProfile* profile,
        uint64_t* values) {
    VxEngineStateScope scope = vx_engine_state_scope_enter(plan->engine);
    VxStatus status = VX_STATUS_OK;
    for (size_t index = 0; index < plan->observer_count; index++) {
        long numel = 0;
        int shape[VX_MAX_TENSOR_RANK] = {0};
        int rank = 0;
        int dtype = -1;
        size_t element_size = 0;
        if (volvoxai_engine_tensor_info_ex(
                plan->observers[index].name, &numel, shape, &rank,
                &dtype, &element_size) != 0 || numel <= 0 ||
            dtype != VX_DTYPE_F32 ||
            (uint64_t)numel > UINT64_MAX - profile->activations[index].values) {
            status = VX_STATUS_INVALID_GRAPH;
            break;
        }
        values[index] = (uint64_t)numel;
    }
    vx_engine_state_scope_leave(scope);
    return status;
}

static void ptq_profile_update_symbols(
        VxPTQPlan* plan,
        VxOwnedPTQProfile* profile,
        const VxPTQCalibrationBatch* batch) {
    for (size_t input = 0; input < plan->input_count; input++) {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        if (vx_model_internal_input_spec(plan->model, input, &spec) !=
            VX_STATUS_OK) continue;
        const VxTensorBinding* binding = ptq_batch_binding(batch, spec.name);
        if (!binding) continue;
        for (uint32_t axis = 0; axis < spec.rank; axis++) {
            const char* symbol = spec.dimensions[axis].symbol;
            if (!symbol) continue;
            for (size_t index = 0; index < profile->symbol_count; index++) {
                VxOwnedPTQSymbolCoverage* coverage = &profile->symbols[index];
                int64_t extent = binding->shape[axis];
                if (strcmp(coverage->name, symbol)) continue;
                if (!profile->batch_count || extent < coverage->minimum)
                    coverage->minimum = extent;
                if (!profile->batch_count || extent > coverage->maximum)
                    coverage->maximum = extent;
                break;
            }
        }
    }
}

VxStatus vx_ptq_plan_calibrate(VxPTQPlan* plan,
                               const VxPTQCalibrationBatch* batch,
                               VxPTQPlanInfo* info,
                               VxReport* report) {
    VxStatus status;
    VxReport input_report = VX_REPORT_INIT;
    volvoxai_ptq_input_binding_t* bindings = NULL;
    char* owned_sample = NULL;
    char* shape_signature = NULL;
    char* owned_signature = NULL;
    char** next_names = NULL;
    uint64_t* activation_values = NULL;
    VxOwnedPTQProfile* profile = NULL;
    uint64_t core_batches = 0;
    int add_signature = 0;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !batch || batch->struct_size != sizeof(*batch) ||
        !ptq_profile_name_valid(batch->profile_name) ||
        !ptq_name_valid(batch->sample_name) || !batch->sample_count ||
        !batch->inputs || !batch->input_count ||
        batch->input_count > (size_t)INT32_MAX ||
        !info || info->struct_size != sizeof(*info)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_CALIBRATE, VX_CODE_INVALID_CALIBRATION_SAMPLE,
                   "named shape-bearing calibration batch is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status != VX_STATUS_OK) goto done;
    profile = ptq_profile_find(plan, batch->profile_name);
    if (!profile || ptq_sample_exists(plan, batch->sample_name) ||
        profile->batch_count == UINT64_MAX ||
        batch->sample_count > UINT64_MAX - profile->sample_count ||
        batch->sample_count > UINT64_MAX - plan->represented_sample_count ||
        profile->sample_count + batch->sample_count > VX_PTQ_MAX_SAFE_INTEGER ||
        plan->represented_sample_count + batch->sample_count >
            VX_PTQ_MAX_SAFE_INTEGER ||
        plan->sample_count >= (size_t)VX_PTQ_MAX_SAFE_INTEGER) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    status = vx_model_internal_bind_authoring_inputs(
        plan->model, plan->engine, VX_PORTABLE_BACKEND_KIND,
        batch->inputs, batch->input_count, NULL, &shape_signature, NULL, &input_report);
    if (status != VX_STATUS_OK) goto done;
    if (plan->sample_count == SIZE_MAX ||
        plan->sample_count + 1u > SIZE_MAX / sizeof(*next_names)) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    add_signature = !ptq_profile_signature_exists(profile, shape_signature);
    if (add_signature) {
        status = ptq_profile_reserve_signature(profile);
        if (status != VX_STATUS_OK) goto done;
        owned_signature = ptq_string_copy(shape_signature);
        if (!owned_signature) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
    }
    activation_values = (uint64_t*)calloc(
        plan->observer_count, sizeof(*activation_values));
    if (!activation_values) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    status = ptq_activation_sizes_locked(plan, profile, activation_values);
    if (status != VX_STATUS_OK) goto done;
    owned_sample = ptq_string_copy(batch->sample_name);
    bindings = (volvoxai_ptq_input_binding_t*)calloc(
        batch->input_count, sizeof(*bindings));
    next_names = (char**)malloc((plan->sample_count + 1u) * sizeof(*next_names));
    if (!owned_sample || !bindings || !next_names) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    if (plan->sample_count)
        memcpy(next_names, plan->sample_names,
               plan->sample_count * sizeof(*next_names));
    for (size_t index = 0; index < batch->input_count; index++) {
        bindings[index] = (volvoxai_ptq_input_binding_t)
            VOLVOXAI_PTQ_INPUT_BINDING_INIT;
        bindings[index].tensor_name = batch->inputs[index].name;
        bindings[index].dtype = batch->inputs[index].dtype;
        bindings[index].data = batch->inputs[index].data;
        bindings[index].nbytes = batch->inputs[index].byte_size;
    }
    {
        VxEngineStateScope scope = vx_engine_state_scope_enter(plan->engine);
        int result = volvoxai_engine_ptq_plan_calibrate_sample(
            plan->core, batch->sample_name, bindings,
            (int32_t)batch->input_count);
        if (result == 0)
            core_batches = volvoxai_ptq_plan_calibration_samples(plan->core);
        vx_engine_state_scope_leave(scope);
        if (result != 0 || core_batches != plan->sample_count + 1u) {
            status = VX_STATUS_EXECUTION_FAILED;
            goto done;
        }
    }
    next_names[plan->sample_count] = owned_sample;
    owned_sample = NULL;
    free(plan->sample_names);
    plan->sample_names = next_names;
    next_names = NULL;
    plan->sample_count++;
    plan->represented_sample_count += batch->sample_count;
    ptq_profile_update_symbols(plan, profile, batch);
    profile->batch_count++;
    profile->sample_count += batch->sample_count;
    if (add_signature) {
        profile->signatures[profile->signature_count++] = owned_signature;
        owned_signature = NULL;
    }
    for (size_t index = 0; index < profile->activation_count; index++)
        profile->activations[index].values += activation_values[index];
    ptq_fill_info_locked(plan, info);
    status = VX_STATUS_OK;

done:
    free(owned_sample);
    free(owned_signature);
    free(shape_signature);
    free(activation_values);
    free(bindings);
    free(next_names);
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_CALIBRATE,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_INVALID_ARGUMENT ? VX_CODE_INVALID_CALIBRATION_SAMPLE :
               status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_CALIBRATION_EXECUTION_FAILED,
               status == VX_STATUS_OK
                   ? "named shape profile calibration batch committed atomically"
                   : "calibration batch was not committed");
    if (report && input_report.status != VX_STATUS_OK)
        report->input_issue = input_report.input_issue;
    return status;
}

VxStatus vx_ptq_plan_info(VxPTQPlan* plan,
                          VxPTQPlanInfo* info,
                          VxReport* report) {
    VxStatus status;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !info || info->struct_size != sizeof(*info)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, VX_CODE_INVALID_PTQ_INFO,
                   "PTQ plan info buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK) ptq_fill_info_locked(plan, info);
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_PTQ_INSPECT_FAILED,
               status == VX_STATUS_OK ? "PTQ plan metadata returned" :
                                        "PTQ plan metadata is unavailable");
    return status;
}

VxStatus vx_ptq_plan_profile_coverage(
        VxPTQPlan* plan,
        size_t index,
        VxPTQProfileCoverage* coverage,
        VxReport* report) {
    VxStatus status;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !coverage || coverage->struct_size != sizeof(*coverage)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, VX_CODE_INVALID_PTQ_PROFILE_QUERY,
                   "PTQ profile coverage buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK && index >= plan->profile_count)
        status = VX_STATUS_NOT_FOUND;
    if (status == VX_STATUS_OK) {
        const VxOwnedPTQProfile* profile = &plan->profiles[index];
        size_t struct_size = coverage->struct_size;
        memset(coverage, 0, sizeof(*coverage));
        coverage->struct_size = struct_size;
        snprintf(coverage->profile_name, sizeof(coverage->profile_name),
                 "%s", profile->name);
        coverage->calibration_batches = profile->batch_count;
        coverage->calibration_samples = profile->sample_count;
        coverage->shape_signature_count = profile->signature_count;
        coverage->symbol_count = profile->symbol_count;
    }
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_NOT_FOUND ? VX_CODE_PTQ_PROFILE_NOT_FOUND :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_PTQ_PROFILE_QUERY_FAILED,
               status == VX_STATUS_OK ? "PTQ profile coverage returned" :
                                        "PTQ profile coverage is unavailable");
    return status;
}

static char* ptq_coverage_json_locked(const VxPTQPlan* plan) {
    cJSON* root = cJSON_CreateObject();
    cJSON* profiles = NULL;
    char* encoded = NULL;
    const char* fingerprint =
        vx_model_internal_logical_fingerprint(plan->model);
    if (!root || !fingerprint || !fingerprint[0] ||
        !cJSON_AddStringToObject(root, "format", "volvox.ptq-coverage/v1") ||
        !cJSON_AddStringToObject(root, "logicalFingerprint", fingerprint) ||
        !cJSON_AddBoolToObject(root, "complete",
                               ptq_coverage_complete_locked(plan)) ||
        !cJSON_AddNumberToObject(root, "totalBatches",
                                 (double)plan->sample_count) ||
        !cJSON_AddNumberToObject(root, "totalSamples",
                                 (double)plan->represented_sample_count))
        goto done;
    profiles = cJSON_CreateArray();
    if (!profiles) goto done;
    if (!cJSON_AddItemToObject(root, "profiles", profiles)) {
        cJSON_Delete(profiles);
        goto done;
    }
    for (size_t index = 0; index < plan->profile_count; index++) {
        const VxOwnedPTQProfile* source = &plan->profiles[index];
        cJSON* profile = cJSON_CreateObject();
        cJSON* signatures = NULL;
        cJSON* symbols = NULL;
        cJSON* activations = NULL;
        if (!profile ||
            !cJSON_AddStringToObject(profile, "name", source->name) ||
            !cJSON_AddNumberToObject(profile, "batches",
                                     (double)source->batch_count) ||
            !cJSON_AddNumberToObject(profile, "samples",
                                     (double)source->sample_count))
            goto profile_failed;
        signatures = cJSON_CreateArray();
        if (!signatures) goto profile_failed;
        if (!cJSON_AddItemToObject(profile, "signatures", signatures)) {
            cJSON_Delete(signatures);
            goto profile_failed;
        }
        for (size_t signature = 0; signature < source->signature_count;
             signature++) {
            cJSON* value = cJSON_CreateString(source->signatures[signature]);
            if (!value || !cJSON_AddItemToArray(signatures, value)) {
                cJSON_Delete(value);
                goto profile_failed;
            }
        }
        symbols = cJSON_CreateObject();
        if (!symbols) goto profile_failed;
        if (!cJSON_AddItemToObject(profile, "symbols", symbols)) {
            cJSON_Delete(symbols);
            goto profile_failed;
        }
        if (source->batch_count) {
            for (size_t symbol = 0; symbol < source->symbol_count; symbol++) {
                cJSON* range = cJSON_CreateObject();
                if (!range || !cJSON_AddNumberToObject(
                        range, "minimum", (double)source->symbols[symbol].minimum) ||
                    !cJSON_AddNumberToObject(
                        range, "maximum", (double)source->symbols[symbol].maximum) ||
                    !cJSON_AddItemToObject(
                        symbols, source->symbols[symbol].name, range)) {
                    cJSON_Delete(range);
                    goto profile_failed;
                }
            }
        }
        activations = cJSON_CreateObject();
        if (!activations) goto profile_failed;
        if (!cJSON_AddItemToObject(
                profile, "activationSamples", activations)) {
            cJSON_Delete(activations);
            goto profile_failed;
        }
        for (size_t activation = 0;
             activation < source->activation_count; activation++) {
            if (!cJSON_AddNumberToObject(
                    activations, source->activations[activation].name,
                    (double)source->activations[activation].values)) {
                goto profile_failed;
            }
        }
        if (!cJSON_AddItemToArray(profiles, profile)) {
            goto profile_failed;
        }
        continue;
profile_failed:
        cJSON_Delete(profile);
        goto done;
    }
    encoded = cJSON_PrintUnformatted(root);
done:
    cJSON_Delete(root);
    return encoded;
}

VxStatus vx_ptq_plan_coverage_json(VxPTQPlan* plan,
                                   char* output,
                                   size_t output_capacity,
                                   size_t* required_size,
    VxReport* report) {
    VxStatus status;
    char* encoded = NULL;
    size_t required = 0;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !required_size || (!output && output_capacity)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, VX_CODE_INVALID_PTQ_COVERAGE_QUERY,
                   "PTQ coverage JSON query is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    *required_size = 0;
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK) {
        encoded = ptq_coverage_json_locked(plan);
        if (!encoded) {
            status = VX_STATUS_OUT_OF_MEMORY;
        } else {
            required = strlen(encoded) + 1u;
            *required_size = required;
            if (output) {
                if (output_capacity < required) {
                    status = VX_STATUS_INVALID_ARGUMENT;
                } else {
                    memcpy(output, encoded, required);
                }
            }
        }
    }
    pthread_mutex_unlock(&plan->mutex);
    free(encoded);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_INVALID_ARGUMENT
                   ? VX_CODE_PTQ_COVERAGE_BUFFER_TOO_SMALL :
               status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_PTQ_COVERAGE_QUERY_FAILED,
               status == VX_STATUS_OK ? "PTQ profile coverage JSON returned" :
                                        "PTQ profile coverage JSON is unavailable");
    return status;
}

VxStatus vx_ptq_plan_tensor_parameters(
    VxPTQPlan* plan,
    size_t index,
    VxPTQTensorParameters* parameters,
    VxReport* report) {
    VxStatus status;
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !parameters ||
        parameters->struct_size != sizeof(*parameters)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, VX_CODE_INVALID_PTQ_TENSOR_QUERY,
                   "PTQ tensor parameter buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK && index >= plan->observer_count)
        status = VX_STATUS_NOT_FOUND;
    if (status == VX_STATUS_OK) {
        size_t struct_size = parameters->struct_size;
        memset(parameters, 0, sizeof(*parameters));
        parameters->struct_size = struct_size;
        snprintf(parameters->tensor_name, sizeof(parameters->tensor_name), "%s",
                 plan->observers[index].name);
        parameters->dtype = plan->observers[index].dtype;
        parameters->scheme = plan->observers[index].scheme;
        if (plan->sample_count) {
            volvoxai_ptq_params_t source;
            VxEngineStateScope scope = vx_engine_state_scope_enter(plan->engine);
            int result = volvoxai_ptq_plan_tensor_params(
                plan->core, plan->observers[index].name, &source);
            vx_engine_state_scope_leave(scope);
            if (result != 0) {
                status = VX_STATUS_INTERNAL;
            } else {
                parameters->scale = source.scale;
                parameters->zero_point = source.zero_point;
                parameters->observed_min = source.observed_min;
                parameters->observed_max = source.observed_max;
                parameters->observed_values = source.sample_count;
            }
        }
    }
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_NOT_FOUND ? VX_CODE_PTQ_TENSOR_NOT_FOUND :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_PTQ_TENSOR_QUERY_FAILED,
               status == VX_STATUS_OK ? "PTQ tensor parameters returned" :
                                        "PTQ tensor parameters are unavailable");
    return status;
}

VxStatus vx_ptq_plan_write_package(VxPTQPlan* plan,
                                   const VxPTQPackageOptions* options,
                                   VxReport* report) {
    VxStatus status;
    int byte_export = options && options->graph_bytes && options->graph_size &&
        options->weights_bytes && options->weights_size &&
        !(options->output_graph_path && options->output_graph_path[0]) &&
        !(options->output_weights_path && options->output_weights_path[0]);
    if (!ptq_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!plan || !options || options->struct_size != sizeof(*options) ||
        (!byte_export && (!ptq_output_graph_name_valid(options->output_graph_path) ||
        !ptq_weights_name_valid(options->output_weights_path) ||
        !ptq_paths_are_siblings(options->output_graph_path,
                                options->output_weights_path) ||
        !strcmp(options->output_graph_path, options->output_weights_path)))) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_WRITE, VX_CODE_INVALID_PTQ_OUTPUT_PATHS,
                   "PTQ output must be sibling graph.json and *.safetensors paths");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK && !ptq_coverage_complete_locked(plan))
        status = VX_STATUS_INVALID_ARGUMENT;
    if (status == VX_STATUS_OK) {
        volvoxai_ptq_package_options_t target = VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT;
        VxEngineStateScope scope;
        char* coverage_json = ptq_coverage_json_locked(plan);
        if (!coverage_json) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto write_done;
        }
        target.template_graph_path = plan->template_graph_snapshot;
        target.source_weights_path = plan->engine->weight_paths[0];
        target.output_graph_path = options->output_graph_path;
        target.output_weights_path = options->output_weights_path;
        target.graph_bytes = options->graph_bytes;
        target.graph_size = options->graph_size;
        target.weights_bytes = options->weights_bytes;
        target.weights_size = options->weights_size;
        target.logical_fingerprint =
            vx_model_internal_logical_fingerprint(plan->model);
        target.profile_coverage_json = coverage_json;
        scope = vx_engine_state_scope_enter(plan->engine);
        if (volvoxai_ptq_plan_write_package(plan->core, &target) != 0)
            status = VX_STATUS_INVALID_GRAPH;
        vx_engine_state_scope_leave(scope);
        free(coverage_json);
    }
write_done:
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_WRITE,
               status == VX_STATUS_OK ? VX_CODE_NONE :
               status == VX_STATUS_INVALID_ARGUMENT ? VX_CODE_PTQ_PROFILE_COVERAGE_REQUIRED :
               status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
               status == VX_STATUS_INVALID_GRAPH ? VX_CODE_PTQ_PACKAGE_WRITE_FAILED :
               status == VX_STATUS_REVISION_CONFLICT ? VX_CODE_REVISION_CONFLICT :
               status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
               VX_CODE_PTQ_WRITE_FAILED,
               status == VX_STATUS_OK
                   ? "W8A8 package written with complete named-profile coverage"
                   : "PTQ package was not written");
    return status;
}
