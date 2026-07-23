#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "volvoxai_full.h"

#include "cJSON.h"
#include "engine_core.h"
#include "json_validation.h"
#include "public_api_internal.h"
#include "runtime_state.h"
#include "training_core.h"

#include <limits.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

typedef struct VxOwnedPTQObserver {
    char name[VX_PTQ_NAME_CAPACITY];
    VxDataType dtype;
    VxPTQScheme scheme;
} VxOwnedPTQObserver;

typedef struct VxOwnedPTQInput {
    char name[VX_PTQ_NAME_CAPACITY];
    VxDataType dtype;
    uint32_t rank;
    int64_t shape[VX_MAX_TENSOR_RANK];
    size_t byte_size;
} VxOwnedPTQInput;

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
    VxOwnedPTQInput* inputs;
    size_t input_count;
    char** sample_names;
    size_t sample_count;
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

static void ptq_report(VxPTQPlan* plan,
                       VxReport* report,
                       VxStatus status,
                       VxStage stage,
                       const char* reason,
                       const char* message) {
    size_t struct_size;
    if (!report || report->struct_size < sizeof(*report)) return;
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
        snprintf(report->backend, sizeof(report->backend), "%s", "cpu");
        snprintf(report->route_evidence, sizeof(report->route_evidence),
                 "%s", "ptq-calibration:cpu-required");
        report->route_attested = 1;
    }
    snprintf(report->reason, sizeof(report->reason), "%s",
             reason ? reason : "");
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

static int ptq_snapshot_template(const char* source_path, char** output_path) {
    unsigned char* bytes = NULL;
    size_t size = 0;
    char path[PATH_MAX] = {0};
    FILE* output = NULL;
    char* owned = NULL;
    int read_status;
    int write_failed = 0;
    if (!output_path) return -1;
    *output_path = NULL;
    read_status = ptq_read_file(source_path, &bytes, &size);
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

static VxStatus ptq_cache_inputs(VxPTQPlan* plan) {
    VxEngineStateScope scope;
    int count;
    if (!plan || !plan->engine) return VX_STATUS_INVALID_ARGUMENT;
    scope = vx_engine_state_scope_enter(plan->engine);
    count = volvoxai_engine_graph_input_count();
    if (count <= 0 || (size_t)count > SIZE_MAX / sizeof(*plan->inputs)) {
        vx_engine_state_scope_leave(scope);
        return VX_STATUS_INVALID_GRAPH;
    }
    plan->inputs = (VxOwnedPTQInput*)calloc((size_t)count,
                                            sizeof(*plan->inputs));
    if (!plan->inputs) {
        vx_engine_state_scope_leave(scope);
        return VX_STATUS_OUT_OF_MEMORY;
    }
    for (int index = 0; index < count; index++) {
        const char* name = volvoxai_engine_graph_input_name(index);
        long numel = 0;
        int shape[VX_MAX_TENSOR_RANK] = {0};
        int rank = 0;
        int dtype = -1;
        size_t element_size = 0;
        VxOwnedPTQInput* target = &plan->inputs[index];
        if (!ptq_name_valid(name) || volvoxai_engine_tensor_info_ex(
                name, &numel, shape, &rank, &dtype, &element_size) != 0 ||
            numel < 0 || rank < 0 || rank > (int)VX_MAX_TENSOR_RANK ||
            !element_size || (size_t)numel > SIZE_MAX / element_size) {
            free(plan->inputs);
            plan->inputs = NULL;
            vx_engine_state_scope_leave(scope);
            return VX_STATUS_INVALID_GRAPH;
        }
        snprintf(target->name, sizeof(target->name), "%s", name);
        target->dtype = (VxDataType)dtype;
        target->rank = (uint32_t)rank;
        for (int axis = 0; axis < rank; axis++) target->shape[axis] = shape[axis];
        target->byte_size = (size_t)numel * element_size;
    }
    plan->input_count = (size_t)count;
    vx_engine_state_scope_leave(scope);
    return VX_STATUS_OK;
}

static void ptq_owner_destroy(VxPTQPlan* plan) {
    if (!plan) return;
    (void)vx_ptq_plan_close(plan, NULL);
    free(plan->inputs);
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
    if (!model || !options || options->struct_size < sizeof(*options) ||
        !out_plan || !options->template_graph_path ||
        !options->template_graph_path[0] || !options->observers ||
        !options->observer_count || options->observer_count > (size_t)INT32_MAX ||
        !options->layers || !options->layer_count ||
        options->layer_count > (size_t)INT32_MAX) {
        if (out_plan) *out_plan = NULL;
        ptq_report(NULL, report, status, VX_STAGE_PTQ_CREATE,
                   "INVALID_PTQ_PLAN", "PTQ plan options are invalid");
        return status;
    }
    *out_plan = NULL;
    snapshot_status = ptq_snapshot_template(options->template_graph_path,
                                            &template_snapshot);
    if (snapshot_status != 0) {
        status = snapshot_status == -2 ? VX_STATUS_OUT_OF_MEMORY :
                 snapshot_status == -3 ? VX_STATUS_INVALID_GRAPH :
                                         VX_STATUS_IO_ERROR;
        ptq_report(NULL, report, status, VX_STAGE_PTQ_CREATE,
                   snapshot_status == -3 ? "INVALID_TEMPLATE_GRAPH" :
                                           "TEMPLATE_SNAPSHOT_FAILED",
                   snapshot_status == -3
                       ? "template graph format must be exactly volvox-graph/v1"
                       : "template graph could not be snapshotted");
        return status;
    }
    plan = (VxPTQPlan*)calloc(1, sizeof(*plan));
    if (!plan) {
        ptq_snapshot_release(template_snapshot);
        ptq_report(NULL, report, VX_STATUS_OUT_OF_MEMORY,
                   VX_STAGE_PTQ_CREATE, "OUT_OF_MEMORY",
                   "PTQ plan allocation failed");
        return VX_STATUS_OUT_OF_MEMORY;
    }
    atomic_init(&plan->references, 1u);
    if (pthread_mutex_init(&plan->mutex, NULL) != 0) {
        free(plan);
        ptq_snapshot_release(template_snapshot);
        ptq_report(NULL, report, VX_STATUS_INTERNAL, VX_STAGE_PTQ_CREATE,
                   "MUTEX_INIT_FAILED", "PTQ plan mutex initialization failed");
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
    status = vx_model_internal_create_authoring_engine(
        model, plan->exact_revision, VOLVOXAI_BACKEND_CPU,
        &plan->engine, report);
    if (status != VX_STATUS_OK) goto fail;
    if (plan->engine->weight_file_count != 1) {
        status = VX_STATUS_INVALID_GRAPH;
        goto fail;
    }
    status = ptq_cache_inputs(plan);
    if (status != VX_STATUS_OK) goto fail;
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
            if (source->struct_size < sizeof(*source) ||
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
            if (volvoxai_ptq_plan_add_tensor(plan->core, &target) != 0)
                status = VX_STATUS_INVALID_GRAPH;
        }
        for (size_t index = 0; status == VX_STATUS_OK &&
                               index < options->layer_count; index++) {
            const VxPTQLayerSpec* source = &options->layers[index];
            volvoxai_ptq_layer_spec_t target = VOLVOXAI_PTQ_LAYER_SPEC_INIT;
            if (source->struct_size < sizeof(*source) ||
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
            if (volvoxai_ptq_plan_add_layer(plan->core, &target) != 0)
                status = VX_STATUS_INVALID_GRAPH;
        }
        vx_engine_state_scope_leave(scope);
    }
    if (status != VX_STATUS_OK) goto fail;
    plan->observer_count = options->observer_count;
    status = vx_model_internal_validate_authoring_revision(
        model, plan->exact_revision, plan->revision.adapter_id,
        plan->revision.adapter_revision, report);
    if (status != VX_STATUS_OK) goto fail;
    *out_plan = plan;
    ptq_report(plan, report, VX_STATUS_OK, VX_STAGE_PTQ_CREATE, "OK",
               "W8A8 PTQ plan created with a private exact-revision CPU engine");
    return VX_STATUS_OK;

fail:
    ptq_report(
        plan, report, status, VX_STAGE_PTQ_CREATE,
        status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
        status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
        status == VX_STATUS_REVISION_CONFLICT ? "REVISION_CONFLICT" :
        status == VX_STATUS_INVALID_ARGUMENT ? "INVALID_PTQ_SPEC" :
        plan && plan->engine && plan->engine->weight_file_count != 1
            ? "PTQ_SINGLE_SHARD_REQUIRED" : "PTQ_PLAN_CREATE_FAILED",
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
    char** sample_names;
    size_t sample_count;
    if (!plan) {
        ptq_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_CLOSE, "INVALID_PTQ_PLAN", "PTQ plan is NULL");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    if (plan->closed) {
        pthread_mutex_unlock(&plan->mutex);
        ptq_report(plan, report, VX_STATUS_OK, VX_STAGE_PTQ_CLOSE, "OK",
                   "PTQ plan already closed");
        return VX_STATUS_OK;
    }
    plan->closed = 1;
    core = plan->core;
    engine = plan->engine;
    revision = plan->exact_revision;
    template_snapshot = plan->template_graph_snapshot;
    observers = plan->observers;
    sample_names = plan->sample_names;
    sample_count = plan->sample_count;
    plan->core = NULL;
    plan->engine = NULL;
    plan->exact_revision = NULL;
    plan->template_graph_snapshot = NULL;
    plan->observers = NULL;
    plan->observer_count = 0;
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
    ptq_samples_release(sample_names, sample_count);
    ptq_report(plan, report, VX_STATUS_OK, VX_STAGE_PTQ_CLOSE, "OK",
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
    if (ptq_current_locked(plan) == VX_STATUS_OK) count = plan->input_count;
    pthread_mutex_unlock(&plan->mutex);
    return count;
}

VxStatus vx_ptq_plan_input_info(VxPTQPlan* plan,
                                size_t index,
                                VxTensorInfo* info,
                                VxReport* report) {
    VxStatus status;
    if (!plan || !info || info->struct_size < sizeof(*info)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, "INVALID_INPUT_QUERY",
                   "PTQ plan or tensor info buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK) {
        if (index >= plan->input_count) {
            status = VX_STATUS_NOT_FOUND;
        } else {
            const VxOwnedPTQInput* source = &plan->inputs[index];
            size_t struct_size = info->struct_size;
            memset(info, 0, sizeof(*info));
            info->struct_size = struct_size;
            info->name = source->name;
            info->dtype = source->dtype;
            info->rank = source->rank;
            memcpy(info->shape, source->shape, sizeof(info->shape));
            info->byte_size = source->byte_size;
            info->location = VX_MEMORY_HOST;
        }
    }
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? "OK" :
               status == VX_STATUS_NOT_FOUND ? "INPUT_NOT_FOUND" :
               status == VX_STATUS_REVISION_CONFLICT ? "REVISION_CONFLICT" :
               status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
               "INPUT_QUERY_FAILED",
               status == VX_STATUS_OK ? "PTQ input metadata returned" :
                                        "PTQ input metadata is unavailable");
    return status;
}

static VxStatus ptq_validate_inputs_locked(VxPTQPlan* plan,
                                           const VxPTQInput* inputs,
                                           size_t input_count) {
    VxEngineStateScope scope;
    int expected_count;
    if (!inputs || !input_count || input_count > (size_t)INT32_MAX)
        return VX_STATUS_INVALID_ARGUMENT;
    scope = vx_engine_state_scope_enter(plan->engine);
    expected_count = volvoxai_engine_graph_input_count();
    if (expected_count <= 0 || input_count != (size_t)expected_count) {
        vx_engine_state_scope_leave(scope);
        return VX_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < input_count; index++) {
        const VxPTQInput* input = &inputs[index];
        long numel = 0;
        int shape[VX_MAX_TENSOR_RANK] = {0};
        int rank = 0;
        int dtype = -1;
        size_t element_size = 0;
        int found = 0;
        if (input->struct_size < sizeof(*input) ||
            !ptq_name_valid(input->name) ||
            (input->byte_size && !input->data)) {
            vx_engine_state_scope_leave(scope);
            return VX_STATUS_INVALID_ARGUMENT;
        }
        for (size_t prior = 0; prior < index; prior++) {
            if (!strcmp(inputs[prior].name, input->name)) {
                vx_engine_state_scope_leave(scope);
                return VX_STATUS_INVALID_ARGUMENT;
            }
        }
        for (int descriptor = 0; descriptor < expected_count; descriptor++) {
            const char* name = volvoxai_engine_graph_input_name(descriptor);
            if (name && !strcmp(name, input->name)) {
                found = 1;
                break;
            }
        }
        if (!found || volvoxai_engine_tensor_info_ex(
                input->name, &numel, shape, &rank, &dtype, &element_size) != 0 ||
            numel < 0 || !element_size ||
            (size_t)numel > SIZE_MAX / element_size ||
            dtype != input->dtype ||
            (size_t)numel * element_size != input->byte_size) {
            vx_engine_state_scope_leave(scope);
            return VX_STATUS_INVALID_ARGUMENT;
        }
    }
    vx_engine_state_scope_leave(scope);
    return VX_STATUS_OK;
}

VxStatus vx_ptq_plan_calibrate(VxPTQPlan* plan,
                               const char* sample_name,
                               const VxPTQInput* inputs,
                               size_t input_count,
                               uint64_t* calibration_samples,
                               VxReport* report) {
    VxStatus status;
    volvoxai_ptq_input_binding_t* bindings = NULL;
    char* owned_sample = NULL;
    char** next_names = NULL;
    uint64_t samples = 0;
    if (!plan || !ptq_name_valid(sample_name) || !calibration_samples) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_CALIBRATE, "INVALID_CALIBRATION_SAMPLE",
                   "named calibration sample arguments are invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    *calibration_samples = 0;
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status != VX_STATUS_OK) goto done;
    status = ptq_validate_inputs_locked(plan, inputs, input_count);
    if (status != VX_STATUS_OK || ptq_sample_exists(plan, sample_name)) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    if (plan->sample_count == SIZE_MAX ||
        plan->sample_count + 1u > SIZE_MAX / sizeof(*next_names)) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    owned_sample = ptq_string_copy(sample_name);
    bindings = (volvoxai_ptq_input_binding_t*)calloc(
        input_count, sizeof(*bindings));
    next_names = (char**)malloc((plan->sample_count + 1u) * sizeof(*next_names));
    if (!owned_sample || !bindings || !next_names) {
        status = VX_STATUS_OUT_OF_MEMORY;
        goto done;
    }
    if (plan->sample_count)
        memcpy(next_names, plan->sample_names,
               plan->sample_count * sizeof(*next_names));
    for (size_t index = 0; index < input_count; index++) {
        bindings[index] = (volvoxai_ptq_input_binding_t)
            VOLVOXAI_PTQ_INPUT_BINDING_INIT;
        bindings[index].tensor_name = inputs[index].name;
        bindings[index].dtype = inputs[index].dtype;
        bindings[index].data = inputs[index].data;
        bindings[index].nbytes = inputs[index].byte_size;
    }
    {
        VxEngineStateScope scope = vx_engine_state_scope_enter(plan->engine);
        int result = volvoxai_engine_ptq_plan_calibrate_sample(
            plan->core, sample_name, bindings, (int32_t)input_count);
        if (result == 0)
            samples = volvoxai_ptq_plan_calibration_samples(plan->core);
        vx_engine_state_scope_leave(scope);
        if (result != 0 || samples != plan->sample_count + 1u) {
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
    *calibration_samples = samples;
    status = VX_STATUS_OK;

done:
    free(owned_sample);
    free(bindings);
    free(next_names);
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_CALIBRATE,
               status == VX_STATUS_OK ? "OK" :
               status == VX_STATUS_INVALID_ARGUMENT ? "INVALID_CALIBRATION_SAMPLE" :
               status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
               status == VX_STATUS_REVISION_CONFLICT ? "REVISION_CONFLICT" :
               status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
               "CALIBRATION_EXECUTION_FAILED",
               status == VX_STATUS_OK ? "calibration sample committed atomically" :
                                        "calibration sample was not committed");
    return status;
}

VxStatus vx_ptq_plan_info(VxPTQPlan* plan,
                          VxPTQPlanInfo* info,
                          VxReport* report) {
    VxStatus status;
    if (!plan || !info || info->struct_size < sizeof(*info)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, "INVALID_PTQ_INFO",
                   "PTQ plan info buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK) {
        size_t struct_size = info->struct_size;
        memset(info, 0, sizeof(*info));
        info->struct_size = struct_size;
        info->calibration_samples = plan->sample_count;
        info->tensor_count = plan->observer_count;
        info->revision = plan->revision;
    }
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_INSPECT,
               status == VX_STATUS_OK ? "OK" :
               status == VX_STATUS_REVISION_CONFLICT ? "REVISION_CONFLICT" :
               status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
               "PTQ_INSPECT_FAILED",
               status == VX_STATUS_OK ? "PTQ plan metadata returned" :
                                        "PTQ plan metadata is unavailable");
    return status;
}

VxStatus vx_ptq_plan_tensor_parameters(
    VxPTQPlan* plan,
    size_t index,
    VxPTQTensorParameters* parameters,
    VxReport* report) {
    VxStatus status;
    if (!plan || !parameters || parameters->struct_size < sizeof(*parameters)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_INSPECT, "INVALID_PTQ_TENSOR_QUERY",
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
               status == VX_STATUS_OK ? "OK" :
               status == VX_STATUS_NOT_FOUND ? "PTQ_TENSOR_NOT_FOUND" :
               status == VX_STATUS_REVISION_CONFLICT ? "REVISION_CONFLICT" :
               status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
               "PTQ_TENSOR_QUERY_FAILED",
               status == VX_STATUS_OK ? "PTQ tensor parameters returned" :
                                        "PTQ tensor parameters are unavailable");
    return status;
}

VxStatus vx_ptq_plan_write_package(VxPTQPlan* plan,
                                   const VxPTQPackageOptions* options,
                                   VxReport* report) {
    VxStatus status;
    if (!plan || !options || options->struct_size < sizeof(*options) ||
        !ptq_output_graph_name_valid(options->output_graph_path) ||
        !ptq_weights_name_valid(options->output_weights_path) ||
        !ptq_paths_are_siblings(options->output_graph_path,
                                options->output_weights_path) ||
        !strcmp(options->output_graph_path, options->output_weights_path)) {
        ptq_report(plan, report, VX_STATUS_INVALID_ARGUMENT,
                   VX_STAGE_PTQ_WRITE, "INVALID_PTQ_OUTPUT_PATHS",
                   "PTQ output must be sibling graph.json and *.safetensors paths");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&plan->mutex);
    status = ptq_current_locked(plan);
    if (status == VX_STATUS_OK && !plan->sample_count)
        status = VX_STATUS_INVALID_ARGUMENT;
    if (status == VX_STATUS_OK) {
        volvoxai_ptq_package_options_t target = VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT;
        VxEngineStateScope scope;
        target.template_graph_path = plan->template_graph_snapshot;
        target.source_weights_path = plan->engine->weight_paths[0];
        target.output_graph_path = options->output_graph_path;
        target.output_weights_path = options->output_weights_path;
        scope = vx_engine_state_scope_enter(plan->engine);
        if (volvoxai_ptq_plan_write_package(plan->core, &target) != 0)
            status = VX_STATUS_INVALID_GRAPH;
        vx_engine_state_scope_leave(scope);
    }
    pthread_mutex_unlock(&plan->mutex);
    ptq_report(plan, report, status, VX_STAGE_PTQ_WRITE,
               status == VX_STATUS_OK ? "OK" :
               status == VX_STATUS_INVALID_ARGUMENT ? "CALIBRATION_REQUIRED" :
               status == VX_STATUS_INVALID_GRAPH ? "PTQ_PACKAGE_WRITE_FAILED" :
               status == VX_STATUS_REVISION_CONFLICT ? "REVISION_CONFLICT" :
               status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
               "PTQ_WRITE_FAILED",
               status == VX_STATUS_OK ? "W8A8 package written as graph.json plus safetensors" :
                                        "PTQ package was not written");
    return status;
}
