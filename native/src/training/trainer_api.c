#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "vx_training_lifecycle.h"

#include "public_api_internal.h"
#include "profiling.h"
#include "runtime_state.h"
#include "training_core.h"
#include "volvoxai_lite.h"

#include <limits.h>
#include <math.h>
#include "vx_thread.h"
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#elif !defined(__wasm__)
#include <unistd.h>
#endif

#ifndef PATH_MAX
#define PATH_MAX 4096
#endif

_Static_assert(sizeof(int32_t) == sizeof(int),
               "native cross-entropy targets require 32-bit int");

struct VxTrainer {
    atomic_uint references;
    pthread_mutex_t mutex;
    int closed;
    atomic_uint shared_readers;
    int poisoned;
    VxModel* model;
    VxWeightRevisionRecord* base_revision;
    VxWeightRevisionRecord* checkpoint_revision;
    uint64_t rng_seed;
    unsigned char* checkpoint_metadata;
    size_t checkpoint_metadata_bytes;
    VxEngineState* engine;
    VolvoxAIEngineShapePolicy shape_policy;
    VxBackendKind backend;
    int training_backend;
    char backend_name[VX_REPORT_BACKEND_CAPACITY];
    uint32_t rng_stream_key;
    uint64_t microbatch_id;
    uint64_t optimizer_step;
    uint64_t baseline_optimizer_step;
    VxOptimizerOptions optimizer_config;
    VxOptimizerOptions baseline_optimizer_config;
    char* binding_signature;
    uint64_t plan_cache_hits;
    uint64_t plan_cache_misses;
    uint64_t plan_cache_evictions;
    uint64_t activation_capacity;
    uint64_t activation_high_water;
    uint64_t activation_grow_count;
    char* baseline_optimizer_path;
    char* accumulation_shape_signature;
    int working_dirty;
    uint32_t accumulated_microbatches;
    VxTrainStepResult last_step;
    VxReport last_step_report;
#if VOLVOXAI_ENABLE_WEBGPU
    VxWebGpuTrainStep* pending_step;
    char* pending_shape_signature;
    VxOptimizerOptions pending_optimizer_config;
#endif
};

static char* trainer_string_copy(const char* value) {
    size_t length;
    char* copy;
    if (!value) return NULL;
    length = strlen(value);
    copy = (char*)malloc(length + 1u);
    if (copy) memcpy(copy, value, length + 1u);
    return copy;
}

static uint64_t trainer_activation_storage(const VxTrainer* trainer) {
    const VxEngineState* engine = trainer->engine;
    uint64_t bytes = engine->arena_allocated_bytes;
    for (int i = 0; i < engine->tensor_count; i++) {
        const T* tensor = &engine->tensors[i];
        if (!tensor->owns || !tensor->data || tensor->numel <= 0) continue;
        int weight = 0;
        for (int f = 0; f < engine->weight_file_count && !weight; f++)
            weight = safetensors_find_tensor(&engine->weight_files[f], tensor->name) != NULL;
        if (!weight) bytes += (uint64_t)tensor->numel * tensor->elem_size;
    }
    return bytes;
}

static void trainer_record_activation_storage(VxTrainer* trainer) {
    uint64_t bytes = trainer_activation_storage(trainer);
    if (bytes > trainer->activation_capacity) trainer->activation_grow_count++;
    trainer->activation_capacity = bytes;
    if (bytes > trainer->activation_high_water) trainer->activation_high_water = bytes;
    if (trainer->engine->bootstrap_activation_bytes > trainer->activation_high_water)
        trainer->activation_high_water = trainer->engine->bootstrap_activation_bytes;
}

static int trainer_report_argument_valid(const VxReport* report) {
    return !report || report->struct_size == sizeof(*report);
}

static void trainer_report(VxTrainer* trainer,
                           VxReport* report,
                           VxStatus status,
                           VxStage stage,
                           VxOperationCode reason,
                           const char* message) {
    if (!report || report->struct_size != sizeof(*report)) return;
    if (trainer && trainer->model) {
        (void)vx_model_revision_info(trainer->model,
                                     &(VxRevisionInfo)VX_REVISION_INFO_INIT,
                                     report);
    } else {
        size_t struct_size = report->struct_size;
        memset(report, 0, sizeof(*report));
        report->struct_size = struct_size;
    }
    report->status = status;
    report->stage = stage;
    if (trainer)
        snprintf(report->backend, sizeof(report->backend), "%s",
                 trainer->backend_name);
    report->code = reason;
    snprintf(report->message, sizeof(report->message), "%s",
             message ? message : "");
}

static int trainer_backend_parse(const char* name,
                                 VxBackendKind* backend,
                                 int* training_backend) {
    VxBackendKind kind = name && name[0] ? vx_backend_kind_from_name(name) : VX_PORTABLE_BACKEND_KIND;
    if (!backend || !training_backend) return -1;
    switch (kind) {
        case VX_PORTABLE_BACKEND_KIND:
            *training_backend = VOLVOXAI_TRAINING_BACKEND_ALLOW_FALLBACK; break;
        case VX_BACKEND_KIND_VULKAN:
            *training_backend = VOLVOXAI_TRAINING_BACKEND_VULKAN; break;
        case VX_BACKEND_KIND_OPENGL:
            *training_backend = VOLVOXAI_TRAINING_BACKEND_OPENGL; break;
        case VX_BACKEND_KIND_METAL:
            *training_backend = VOLVOXAI_TRAINING_BACKEND_METAL; break;
        case VX_BACKEND_KIND_CUDA:
            *training_backend = VOLVOXAI_TRAINING_BACKEND_CUDA; break;
#if VOLVOXAI_ENABLE_WEBGPU
        case VX_BACKEND_KIND_WEBGPU:
            *training_backend = VOLVOXAI_TRAINING_BACKEND_ALLOW_FALLBACK; break;
#endif
        default: return -1;
    }
    *backend = kind;
    return 0;
}

static uint32_t trainer_rng_stream_key(uint64_t seed) {
    uint64_t value = seed ? seed : UINT64_C(0x6a09e667f3bcc909);
    value ^= value >> 30u;
    value *= UINT64_C(0xbf58476d1ce4e5b9);
    value ^= value >> 27u;
    value *= UINT64_C(0x94d049bb133111eb);
    value ^= value >> 31u;
    return (uint32_t)(value ^ (value >> 32u));
}

static uint32_t trainer_shape_stream_key(const char* signature) {
    uint32_t value = UINT32_C(2166136261);
    if (!signature) return 0u;
    for (const unsigned char* cursor = (const unsigned char*)signature;
         *cursor; cursor++) {
        value ^= *cursor;
        value *= UINT32_C(16777619);
    }
    return value;
}

static int trainer_temp_path(char** out_path) {
    char path[PATH_MAX];
    char* owned;
    if (!out_path) return -1;
    *out_path = NULL;
#ifdef _WIN32
    {
        char directory[MAX_PATH];
        DWORD length = GetTempPathA((DWORD)sizeof(directory), directory);
        if (!length || length >= sizeof(directory) ||
            !GetTempFileNameA(directory, "vxt", 0, path))
            return -1;
    }
#elif defined(__wasm__)
    {
        static uint64_t sequence;
        if (sequence == UINT64_MAX) return -1;
        int written = snprintf(path, sizeof(path), "/trainer/%llu", ++sequence);
        if (written < 0 || (size_t)written >= sizeof(path)) return -1;
        FILE* file = fopen(path, "wb");
        if (!file) return -1;
        if (fclose(file) != 0) { (void)remove(path); return -1; }
    }
#else
    {
        const char* directory = getenv("TMPDIR");
        int descriptor;
        int written;
        if (!directory || !directory[0]) directory = "/tmp";
        written = snprintf(path, sizeof(path), "%s%svolvoxai-trainer-XXXXXX",
                           directory,
                           directory[strlen(directory) - 1u] == '/' ? "" : "/");
        if (written < 0 || (size_t)written >= sizeof(path)) return -1;
        descriptor = mkstemp(path);
        if (descriptor < 0) return -1;
        if (close(descriptor) != 0) {
            (void)remove(path);
            return -1;
        }
    }
#endif
    owned = trainer_string_copy(path);
    if (!owned) {
        (void)remove(path);
        return -1;
    }
    *out_path = owned;
    return 0;
}

static void trainer_owned_path_release(char* path) {
    if (!path) return;
    (void)remove(path);
    free(path);
}

static VxStatus trainer_configure_engine_locked(VxTrainer* trainer,
                                                 VxEngineState* engine,
                                                 VxReport* report) {
    VxEngineStateScope scope;
    int status;
    if (!trainer || !engine) return VX_STATUS_INVALID_ARGUMENT;
    engine->native_training_rng_seed = trainer->rng_stream_key;
    scope = vx_engine_state_scope_enter(engine);
    status = trainer->backend == VX_BACKEND_KIND_WEBGPU ? 0 :
        volvoxai_engine_require_training_backend(trainer->training_backend);
    if (status == 0 && trainer->baseline_optimizer_path) {
        long loaded_step = -1;
        status = volvoxai_engine_load_optimizer_state(
            trainer->baseline_optimizer_path, &loaded_step);
        if (status == 0 &&
            (loaded_step < 0 || (uint64_t)loaded_step !=
                                  trainer->baseline_optimizer_step))
            status = -1;
    }
    vx_engine_state_scope_leave(scope);
    if (status == 0) return VX_STATUS_OK;
    trainer_report(trainer, report, VX_STATUS_BACKEND_UNSUPPORTED,
                   VX_STAGE_TRAINER_CREATE, VX_CODE_TRAINING_PREFLIGHT_FAILED,
                   "required training backend or optimizer baseline is unavailable");
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static VxStatus trainer_restore_baseline_locked(VxTrainer* trainer,
                                                 VxReport* report) {
    VxEngineState* replacement = NULL;
    VxStatus status;
    status = vx_model_internal_create_authoring_engine(
        trainer->model, trainer->checkpoint_revision ? trainer->checkpoint_revision : trainer->base_revision,
        trainer->backend, &trainer->shape_policy, &replacement, report);
    if (status != VX_STATUS_OK) return status;
    status = trainer_configure_engine_locked(trainer, replacement, report);
    if (status != VX_STATUS_OK) {
        vx_model_internal_destroy_authoring_engine(replacement);
        return status;
    }
    vx_model_internal_destroy_authoring_engine(trainer->engine);
    trainer->engine = replacement;
    trainer->optimizer_step = trainer->baseline_optimizer_step;
    trainer->optimizer_config = trainer->baseline_optimizer_config;
    free(trainer->binding_signature);
    trainer->binding_signature = NULL;
    trainer->plan_cache_hits = trainer->plan_cache_misses = trainer->plan_cache_evictions = 0;
    trainer->activation_capacity = trainer->activation_high_water = trainer->activation_grow_count = 0;
    trainer_record_activation_storage(trainer);
    trainer->working_dirty = trainer->checkpoint_revision != NULL;
    trainer->accumulated_microbatches = 0;
    free(trainer->accumulation_shape_signature);
    trainer->accumulation_shape_signature = NULL;
    trainer->poisoned = 0;
    return VX_STATUS_OK;
}

static void trainer_temp_paths_release(char** paths, size_t count) {
    if (!paths) return;
    for (size_t index = 0; index < count; index++)
        trainer_owned_path_release(paths[index]);
    free(paths);
}

static VxStatus trainer_snapshot_weights_locked(VxTrainer* trainer,
                                                 char*** out_paths,
                                                 size_t* out_count) {
    char** paths;
    size_t count;
    VxEngineStateScope scope;
    int failed = 0;
    if (!trainer || !trainer->engine || !out_paths || !out_count)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_paths = NULL;
    *out_count = 0;
    count = (size_t)trainer->engine->weight_file_count;
    if (!count || count > (size_t)MAX_WEIGHT_FILES)
        return VX_STATUS_INVALID_GRAPH;
    paths = (char**)calloc(count, sizeof(*paths));
    if (!paths) return VX_STATUS_OUT_OF_MEMORY;
    for (size_t index = 0; index < count; index++) {
        if (trainer_temp_path(&paths[index]) != 0) {
            trainer_temp_paths_release(paths, count);
            return VX_STATUS_OUT_OF_MEMORY;
        }
    }
    scope = vx_engine_state_scope_enter(trainer->engine);
    for (size_t index = 0; index < count; index++) {
        if (volvoxai_engine_save_weight_file((int)index, paths[index]) != 0) {
            failed = 1;
            break;
        }
    }
    vx_engine_state_scope_leave(scope);
    if (failed) {
        trainer_temp_paths_release(paths, count);
        return VX_STATUS_IO_ERROR;
    }
    *out_paths = paths;
    *out_count = count;
    return VX_STATUS_OK;
}

static VxStatus trainer_snapshot_optimizer_locked(VxTrainer* trainer,
                                                   char** out_path) {
    VxEngineStateScope scope;
    char* path = NULL;
    int status;
    if (!trainer || !trainer->engine || !out_path ||
        trainer->optimizer_step > (uint64_t)LONG_MAX)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_path = NULL;
    if (trainer_temp_path(&path) != 0) return VX_STATUS_OUT_OF_MEMORY;
    scope = vx_engine_state_scope_enter(trainer->engine);
    status = volvoxai_engine_save_optimizer_state(
        path, (long)trainer->optimizer_step);
    vx_engine_state_scope_leave(scope);
    if (status != 0) {
        trainer_owned_path_release(path);
        return VX_STATUS_IO_ERROR;
    }
    *out_path = path;
    return VX_STATUS_OK;
}

VxStatus vx_model_create_trainer(VxModel* model,
                                 const VxTrainerOptions* options,
                                 VxTrainer** out_trainer,
                                 VxReport* report) {
    VxTrainerOptions resolved = VX_TRAINER_OPTIONS_INIT;
    VxTrainer* trainer;
    VxStatus status;
    const char* backend_name;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!model || !out_trainer ||
        (options && options->struct_size != sizeof(*options))) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_CREATE, VX_CODE_INVALID_TRAINER_OPTIONS,
                       "model, options, or trainer output is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    *out_trainer = NULL;
    if (options) resolved = *options;
    if (resolved.backend && !resolved.backend[0]) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_CREATE, VX_CODE_INVALID_TRAINING_BACKEND,
                       "training backend name must be non-empty when provided");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    backend_name = resolved.backend && resolved.backend[0]
        ? resolved.backend :
#if defined(__wasm__)
          "wasm";
#else
          "cpu";
#endif
    if (strlen(backend_name) >= VX_REPORT_BACKEND_CAPACITY) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_CREATE, VX_CODE_INVALID_TRAINING_BACKEND,
                       "training backend name is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    trainer = (VxTrainer*)calloc(1, sizeof(*trainer));
    if (!trainer) {
        trainer_report(NULL, report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_TRAINER_CREATE, VX_CODE_OUT_OF_MEMORY,
                       "trainer allocation failed");
        return VX_STATUS_OUT_OF_MEMORY;
    }
    if (trainer_backend_parse(backend_name, &trainer->backend,
                              &trainer->training_backend) != 0) {
        free(trainer);
        trainer_report(NULL, report, VX_STATUS_BACKEND_UNSUPPORTED,
                       VX_STAGE_TRAINER_CREATE, VX_CODE_TRAINING_BACKEND_UNSUPPORTED,
                       "trainer requires cpu, vulkan, opengl, metal, or cuda");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    atomic_init(&trainer->references, 1);
    atomic_init(&trainer->shared_readers, 0);
    if (pthread_mutex_init(&trainer->mutex, NULL) != 0) {
        free(trainer);
        trainer_report(NULL, report, VX_STATUS_INTERNAL,
                       VX_STAGE_TRAINER_CREATE, VX_CODE_MUTEX_INIT_FAILED,
                       "trainer mutex initialization failed");
        return VX_STATUS_INTERNAL;
    }
    trainer->shape_policy = (VolvoxAIEngineShapePolicy){8u, 1024u * 1024u,
        512u * 1024u * 1024u, 2.0, 1};
    if (resolved.shape_options) {
        const VolvoxaiV1TrainerShapeOptions* p = resolved.shape_options;
#define VX_SHAPE_OPTION(field) if (p->has_##field) trainer->shape_policy.field = p->field_##field
        VX_SHAPE_OPTION(plan_cache_entries);
        VX_SHAPE_OPTION(plan_cache_metadata_bytes);
        VX_SHAPE_OPTION(max_activation_capacity_bytes);
        VX_SHAPE_OPTION(capacity_growth_factor);
#undef VX_SHAPE_OPTION
    }
    const VolvoxAIEngineShapePolicy* policy = &trainer->shape_policy;
    if (!policy->plan_cache_entries || !policy->plan_cache_metadata_bytes ||
        !policy->max_activation_capacity_bytes ||
        policy->plan_cache_metadata_bytes > SIZE_MAX ||
        policy->max_activation_capacity_bytes > SIZE_MAX ||
        !isfinite(policy->capacity_growth_factor) || policy->capacity_growth_factor <= 1.0) {
        pthread_mutex_destroy(&trainer->mutex); free(trainer);
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT, VX_STAGE_TRAINER_CREATE,
            VX_CODE_INVALID_SHAPE_OPTIONS, "shape limits must be positive and growth factor finite and greater than one");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    trainer->model = model;
    vx_model_retain(model);
    trainer->rng_seed = resolved.rng_seed;
    trainer->optimizer_config = trainer->baseline_optimizer_config =
        (VxOptimizerOptions)VX_OPTIMIZER_OPTIONS_INIT;
    trainer->rng_stream_key = trainer_rng_stream_key(resolved.rng_seed);
    snprintf(trainer->backend_name, sizeof(trainer->backend_name), "%s",
             backend_name);
    status = vx_model_internal_accept_full_owner(
        model, &trainer->base_revision, report);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_model_internal_create_authoring_engine(
        model, trainer->base_revision, trainer->backend, &trainer->shape_policy,
        &trainer->engine, report);
    if (status != VX_STATUS_OK) goto fail;
    status = trainer_configure_engine_locked(trainer, trainer->engine, report);
    if (status != VX_STATUS_OK) goto fail;
    trainer_record_activation_storage(trainer);
    *out_trainer = trainer;
    trainer_report(trainer, report, VX_STATUS_OK, VX_STAGE_TRAINER_CREATE,
                   VX_CODE_NONE, "trainer created with a private exact-revision engine");
    return VX_STATUS_OK;

fail:
    if (!report || report->status == VX_STATUS_OK) trainer_report(
        trainer, report, status, VX_STAGE_TRAINER_CREATE,
        status == VX_STATUS_INTERNAL ? VX_CODE_REVISION_MISSING :
        status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
        status == VX_STATUS_BACKEND_UNAVAILABLE ? VX_CODE_TRAINING_BACKEND_UNAVAILABLE :
        status == VX_STATUS_BACKEND_UNSUPPORTED ? VX_CODE_TRAINING_PREFLIGHT_FAILED :
        status == VX_STATUS_OUT_OF_MEMORY ? VX_CODE_OUT_OF_MEMORY :
        VX_CODE_TRAINER_CREATE_FAILED,
        status == VX_STATUS_INTERNAL
            ? "model has no published weight revision"
            : status == VX_STATUS_HANDLE_DISPOSED
            ? "runtime is closed"
            : "private exact-revision training engine creation failed");
    vx_model_internal_destroy_authoring_engine(trainer->engine);
    vx_model_internal_release_weight_revision(trainer->base_revision);
    vx_model_release(model);
    pthread_mutex_destroy(&trainer->mutex);
    free(trainer);
    return status;
}

void vx_trainer_retain(VxTrainer* trainer) {
    if (trainer)
        atomic_fetch_add_explicit(&trainer->references, 1,
                                  memory_order_relaxed);
}

VxStatus vx_trainer_close(VxTrainer* trainer, VxReport* report) {
    VxEngineState* engine;
    VxWeightRevisionRecord* base;
    char* optimizer_path;
    char* accumulation_shape_signature;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_CLOSE, VX_CODE_INVALID_TRAINER, "trainer is NULL");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        pthread_mutex_unlock(&trainer->mutex);
        trainer_report(trainer, report, VX_STATUS_OK, VX_STAGE_CLOSE, VX_CODE_NONE,
                       "trainer already closed");
        return VX_STATUS_OK;
    }
    if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire)) {
        pthread_mutex_unlock(&trainer->mutex);
        trainer_report(trainer, report, VX_STATUS_BUSY, VX_STAGE_CLOSE, VX_CODE_BUSY,
            "shared parameter views are live");
        return VX_STATUS_BUSY;
    }
    trainer->closed = 1;
#if VOLVOXAI_ENABLE_WEBGPU
    if (trainer->pending_step) {
        VxEngineStateScope scope = vx_engine_state_scope_enter(trainer->engine);
        vx_webgpu_train_release(trainer->pending_step);
        vx_engine_state_scope_leave(scope);
        trainer->pending_step = NULL;
    }
    free(trainer->pending_shape_signature);
    trainer->pending_shape_signature = NULL;
#endif
    engine = trainer->engine;
    base = trainer->base_revision;
    optimizer_path = trainer->baseline_optimizer_path;
    accumulation_shape_signature = trainer->accumulation_shape_signature;
    trainer->engine = NULL;
    trainer->base_revision = NULL;
    trainer->baseline_optimizer_path = NULL;
    trainer->accumulation_shape_signature = NULL;
    pthread_mutex_unlock(&trainer->mutex);
    vx_model_internal_destroy_authoring_engine(engine);
    vx_model_internal_release_weight_revision(base);
    vx_model_internal_release_weight_revision(trainer->checkpoint_revision);
    trainer->checkpoint_revision = NULL;
    free(trainer->checkpoint_metadata);
    trainer->checkpoint_metadata = NULL;
    trainer->checkpoint_metadata_bytes = 0;
    trainer_owned_path_release(optimizer_path);
    free(accumulation_shape_signature);
    free(trainer->binding_signature);
    trainer->binding_signature = NULL;
    trainer_report(trainer, report, VX_STATUS_OK, VX_STAGE_CLOSE, VX_CODE_NONE,
                   "trainer closed and private training state released");
    return VX_STATUS_OK;
}

void vx_trainer_release(VxTrainer* trainer) {
    if (!trainer || atomic_fetch_sub_explicit(&trainer->references, 1,
                                              memory_order_acq_rel) != 1)
        return;
    (void)vx_trainer_close(trainer, NULL);
    vx_model_release(trainer->model);
    pthread_mutex_destroy(&trainer->mutex);
    free(trainer);
}

size_t vx_trainer_input_count(VxTrainer* trainer) {
    size_t count = 0;
    if (!trainer) return 0;
    pthread_mutex_lock(&trainer->mutex);
    if (!trainer->closed && trainer->model)
        count = vx_model_internal_input_count(trainer->model);
    pthread_mutex_unlock(&trainer->mutex);
    return count;
}

VxStatus vx_trainer_input_spec(VxTrainer* trainer,
                               size_t index,
                               VxTensorSpec* spec,
                               VxReport* report) {
    VxStatus status = VX_STATUS_NOT_FOUND;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer || !spec || spec->struct_size != sizeof(*spec)) {
        trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_INPUT, VX_CODE_INVALID_INPUT_QUERY,
                       "trainer or tensor spec buffer is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
    } else {
        status = vx_model_internal_input_spec(trainer->model, index, spec);
    }
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_INPUT,
                   status == VX_STATUS_OK ? VX_CODE_NONE :
                   status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
                   status == VX_STATUS_NOT_FOUND ? VX_CODE_INPUT_NOT_FOUND :
                   VX_CODE_INPUT_QUERY_FAILED,
                   status == VX_STATUS_OK ? "logical trainer input spec returned" :
                   "logical trainer input spec is unavailable");
    return status;
}

static int trainer_optimizer_valid(const VxOptimizerOptions* optimizer) {
    return optimizer && optimizer->struct_size == sizeof(*optimizer) &&
        (optimizer->kind == VX_OPTIMIZER_SGD || optimizer->kind == VX_OPTIMIZER_ADAMW) &&
        isfinite(optimizer->learning_rate) && optimizer->learning_rate >= 0.0f &&
        isfinite(optimizer->beta1) && optimizer->beta1 >= 0.0f && optimizer->beta1 < 1.0f &&
        isfinite(optimizer->beta2) && optimizer->beta2 >= 0.0f && optimizer->beta2 < 1.0f &&
        isfinite(optimizer->epsilon) && optimizer->epsilon > 0.0f &&
        isfinite(optimizer->weight_decay) && optimizer->weight_decay >= 0.0f &&
        isfinite(optimizer->max_gradient_norm) && optimizer->max_gradient_norm >= 0.0f;
}

static VxOptimizerOptions trainer_optimizer_resolve(const VxTrainer* trainer,
                                                    const VxTrainStepOptions* request) {
    VxOptimizerOptions value = trainer->optimizer_config;
    uint32_t fields = request->optimizer_fields;
    if (fields & VX_OPTIMIZER_FIELD_KIND) {
        if (value.kind != request->optimizer.kind)
            value = (VxOptimizerOptions)VX_OPTIMIZER_OPTIONS_INIT;
        value.kind = request->optimizer.kind;
    }
#define VX_OPTIMIZER_OVERRIDE(flag, member) \
    if (fields & flag) value.member = request->optimizer.member
    VX_OPTIMIZER_OVERRIDE(VX_OPTIMIZER_FIELD_LEARNING_RATE, learning_rate);
    VX_OPTIMIZER_OVERRIDE(VX_OPTIMIZER_FIELD_BETA1, beta1);
    VX_OPTIMIZER_OVERRIDE(VX_OPTIMIZER_FIELD_BETA2, beta2);
    VX_OPTIMIZER_OVERRIDE(VX_OPTIMIZER_FIELD_EPSILON, epsilon);
    VX_OPTIMIZER_OVERRIDE(VX_OPTIMIZER_FIELD_WEIGHT_DECAY, weight_decay);
    VX_OPTIMIZER_OVERRIDE(VX_OPTIMIZER_FIELD_MAX_GRADIENT_NORM, max_gradient_norm);
#undef VX_OPTIMIZER_OVERRIDE
    return value;
}

static int trainer_step_options_valid(const VxTrainStepOptions* options,
                                      const VxTrainStepResult* result) {
    if (!options || options->struct_size != sizeof(*options) ||
        !result || result->struct_size != sizeof(*result) ||
        options->optimizer.struct_size != sizeof(options->optimizer) ||
        !options->inputs || !options->input_count ||
        !options->losses || !options->loss_count ||
        options->loss_count > VX_MAX_TRAINING_LOSSES ||
        options->loss_count > (size_t)INT_MAX ||
        !options->trainable_names || !options->trainable_count ||
        options->trainable_count > (size_t)INT_MAX ||
        !options->accumulation_steps ||
        options->accumulation_steps > (uint32_t)INT_MAX ||
        (options->flush_accumulation != 0 &&
         options->flush_accumulation != 1) ||
        (options->reset_accumulation != 0 &&
         options->reset_accumulation != 1) ||
        !trainer_optimizer_valid(&options->optimizer))
        return 0;
    for (size_t index = 0; index < options->loss_count; index++) {
        const VxCrossEntropyLoss* loss = &options->losses[index];
        if (loss->struct_size != sizeof(*loss) || !loss->name || !loss->name[0] ||
            strlen(loss->name) >= VX_TRAINING_NAME_CAPACITY ||
            !loss->logits_name || !loss->logits_name[0] ||
            !loss->targets || !loss->target_count ||
            loss->target_count > (size_t)INT_MAX || loss->row_index < -1 ||
            !isfinite(loss->weight) || loss->weight < 0.0f ||
            !isfinite(loss->normalizer) || loss->normalizer < 0.0f ||
            (options->accumulation_steps > 1u && loss->normalizer <= 0.0f))
            return 0;
        for (size_t previous = 0; previous < index; previous++)
            if (!strcmp(options->losses[previous].name, loss->name)) return 0;
    }
    for (size_t index = 0; index < options->trainable_count; index++) {
        if (!options->trainable_names[index] ||
            !options->trainable_names[index][0]) return 0;
        for (size_t previous = 0; previous < index; previous++)
            if (!strcmp(options->trainable_names[previous],
                        options->trainable_names[index])) return 0;
    }
    return 1;
}

static void trainer_shared_release(void* pointer) {
    VxTrainer* trainer = pointer;
    atomic_fetch_sub_explicit(&trainer->shared_readers, 1, memory_order_release);
    vx_trainer_release(trainer);
}
static void trainer_storage_release(void* pointer) { vx_native_storage_release(pointer); }

VxStatus vx_trainer_read_parameters(VxTrainer* trainer, const char* const* names,
    size_t count, int shared, VxTrainerTensor* outputs, VxReport* report) {
    if (!trainer || !names || !count || !outputs) return VX_STATUS_INVALID_ARGUMENT;
    VxStatus status = VX_STATUS_OK;
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) { status = VX_STATUS_HANDLE_DISPOSED; goto done; }
    if (trainer->last_step.state == VX_RESULT_STATE_PENDING) { status = VX_STATUS_BUSY; goto done; }
    if (trainer->poisoned || !trainer->engine) { status = VX_STATUS_INTERNAL; goto done; }
    /* Shared views are initially CPU-only. GPU exports remain snapshots until
     * their backend can expose a stable parameter allocation with read access. */
    if (shared && trainer->backend != VX_PORTABLE_BACKEND_KIND) {
        status = VX_STATUS_BACKEND_UNSUPPORTED; goto done;
    }
    VxEngineStateScope scope = vx_engine_state_scope_enter(trainer->engine);
    int batch = vx_native_tensor_batch_begin(trainer->backend == VX_BACKEND_KIND_WASM
        ? VX_BACKEND_KIND_NATIVE_CPU : trainer->backend, 1);
    if (!batch) status = VX_STATUS_BACKEND_UNSUPPORTED;
    for (size_t i = 0; i < count && status == VX_STATUS_OK; i++) {
        T* tensor = NULL;
        if (!names[i]) { status = VX_STATUS_INVALID_ARGUMENT; break; }
        for (int j = 0; j < trainer->engine->tensor_count; j++)
            if (!strcmp(trainer->engine->tensors[j].name, names[i])) { tensor = &trainer->engine->tensors[j]; break; }
        int weight = 0;
        for (int f = 0; f < trainer->engine->weight_file_count && !weight; f++)
            weight = safetensors_find_tensor(&trainer->engine->weight_files[f], names[i]) != NULL;
        if (!tensor || !weight || !tensor->data || tensor->numel <= 0 ||
            tensor->ndim < 0 || tensor->ndim > VX_MAX_TENSOR_RANK || !tensor->elem_size ||
            (size_t)tensor->numel > SIZE_MAX / tensor->elem_size) { status = VX_STATUS_INVALID_ARGUMENT; break; }
        VxTrainerTensor* output = &outputs[i];
        output->info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
        output->info.name = names[i];
        output->info.dtype = tensor->dtype;
        output->info.rank = (uint32_t)tensor->ndim;
        output->info.byte_size = (size_t)tensor->numel * tensor->elem_size;
        for (int j = 0; j < tensor->ndim; j++) output->info.shape[j] = tensor->shape[j];
        if (shared) {
            output->memory = (VxNativeBuffer){.kind = VX_NATIVE_BUFFER_HOST,
                .handle = (uint64_t)(uintptr_t)tensor->data, .length = output->info.byte_size};
            vx_trainer_retain(trainer);
            atomic_fetch_add_explicit(&trainer->shared_readers, 1, memory_order_relaxed);
            output->owner = trainer; output->release = trainer_shared_release;
        } else {
            VxBackendKind backend = trainer->backend == VX_BACKEND_KIND_WASM ? VX_BACKEND_KIND_NATIVE_CPU : trainer->backend;
            if (volvoxai_engine_snapshot_native_tensor(names[i], backend, NULL, &output->storage) != 0) {
                status = VX_STATUS_EXECUTION_FAILED; break;
            }
            output->memory = output->storage->buffer;
            output->owner = output->storage; output->release = trainer_storage_release;
        }
        output->info.location = output->memory.kind == VX_NATIVE_BUFFER_HOST ? VX_MEMORY_HOST : VX_MEMORY_DEVICE;
    }
    if (batch && !vx_native_tensor_batch_end(trainer->backend == VX_BACKEND_KIND_WASM
        ? VX_BACKEND_KIND_NATIVE_CPU : trainer->backend)) status = VX_STATUS_EXECUTION_FAILED;
    vx_engine_state_scope_leave(scope);
done:
    pthread_mutex_unlock(&trainer->mutex);
    if (status != VX_STATUS_OK) for (size_t i = 0; i < count; i++) {
        if (outputs[i].release) outputs[i].release(outputs[i].owner);
        memset(&outputs[i], 0, sizeof(outputs[i]));
    }
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_EXPORT,
        status == VX_STATUS_OK ? VX_CODE_NONE : status == VX_STATUS_BUSY ? VX_CODE_BUSY : VX_CODE_INVALID_ARGUMENT,
        status == VX_STATUS_OK ? "parameter storage retained" : "parameter export is unavailable");
    return status;
}

typedef struct {
    VxTrainer* trainer;
    const VxTrainStepOptions* options;
} VxTrainerCapture;
static int trainer_prepare_device_inputs(void* pointer) {
    VxTrainerCapture* capture = pointer;
    const VxTrainStepOptions* options = capture->options;
    VxTrainer* trainer = capture->trainer;
    int handoff = 2;
    for (size_t i = 0; i < options->input_count; i++)
        if (options->inputs[i].location == VX_MEMORY_DEVICE && !options->inputs[i].native_ready) handoff = 0;
    if (!vx_native_tensor_batch_begin(trainer->backend, handoff)) return -1;
    int ok = 1;
    for (size_t i = 0; i < options->input_count && ok; i++) {
        const VxTensorBinding* input = &options->inputs[i];
        if (input->location != VX_MEMORY_DEVICE) continue;
        T* tensor = NULL;
        for (int j = 0; j < trainer->engine->tensor_count; j++)
            if (!strcmp(trainer->engine->tensors[j].name, input->name)) {
                tensor = &trainer->engine->tensors[j]; break;
            }
        ok = tensor && vx_native_tensor_copy_input(trainer->backend,
            tensor->data, &input->native_buffer, 0);
    }
    if (!vx_native_tensor_batch_end(trainer->backend)) ok = 0;
    return ok ? 0 : -1;
}
static int trainer_capture_forward(void* pointer) {
    VxTrainerCapture* capture = pointer;
    const VxTrainStepOptions* options = capture->options;
    VxTrainer* trainer = capture->trainer;
    VxBackendKind backend = trainer->backend == VX_BACKEND_KIND_WASM ? VX_BACKEND_KIND_NATIVE_CPU : trainer->backend;
    if (!options->output_count) return 0;
    if (!vx_native_tensor_batch_begin(backend, 1)) return -1;
    int ok = 1;
    for (size_t i = 0; i < options->output_count && ok; i++) {
        T* tensor = NULL;
        for (int j = 0; j < trainer->engine->tensor_count; j++)
            if (!strcmp(trainer->engine->tensors[j].name, options->output_names[i])) {
                tensor = &trainer->engine->tensors[j]; break;
            }
        if (!tensor || tensor->numel <= 0 || !tensor->elem_size ||
            (size_t)tensor->numel > SIZE_MAX / tensor->elem_size) { ok = 0; break; }
        VxTrainerTensor* output = &options->outputs[i];
        output->info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
        output->info.name = options->output_names[i];
        output->info.dtype = tensor->dtype;
        output->info.rank = (uint32_t)tensor->ndim;
        output->info.byte_size = (size_t)tensor->numel * tensor->elem_size;
        for (int j = 0; j < tensor->ndim; j++) output->info.shape[j] = tensor->shape[j];
        output->storage = vx_native_storage_snapshot(NULL, backend, tensor->data, output->info.byte_size);
        if (!output->storage) { ok = 0; break; }
        output->memory = output->storage->buffer;
        output->info.location = output->memory.kind == VX_NATIVE_BUFFER_HOST ? VX_MEMORY_HOST : VX_MEMORY_DEVICE;
        output->owner = output->storage;
        output->release = trainer_storage_release;
    }
    if (!vx_native_tensor_batch_end(backend)) ok = 0;
    return ok ? 0 : -1;
}

VxStatus vx_trainer_train_step(VxTrainer* trainer,
                               const VxTrainStepOptions* requested,
                               VxTrainStepResult* result,
                               VxReport* report) {
    volvoxai_cross_entropy_loss_t losses[VX_MAX_TRAINING_LOSSES];
    volvoxai_cross_entropy_metric_t metrics[VX_MAX_TRAINING_LOSSES];
    VxEngineStateScope scope;
    VxTraceScope profiling = {0};
    VxStatus status = VX_STATUS_EXECUTION_FAILED;
    float aggregate_loss = 0.0f;
    int accumulated = 0;
    int update_applied = 0;
    int update_mode;
    int core_status;
    char* shape_signature = NULL;
    char gpu_evidence[256] = {0};
    VxReport input_report = VX_REPORT_INIT;
    VxTrainStepOptions resolved;
    const VxTrainStepOptions* options = &resolved;
    VolvoxAIEngineDynamicShapeStats binding_stats = VOLVOXAI_ENGINE_DYNAMIC_SHAPE_STATS_INIT;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer || !requested || requested->struct_size != sizeof(*requested) ||
        !result || result->struct_size != sizeof(*result) ||
        requested->optimizer_fields & ~VX_OPTIMIZER_FIELDS_ALL) {
        trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_STEP, VX_CODE_INVALID_TRAIN_STEP,
                       "loss, optimizer, accumulation, trainable, or result options are invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
        goto done;
    }
    if (trainer->poisoned || !trainer->engine ||
        trainer->optimizer_step >= (uint64_t)LONG_MAX) {
        status = VX_STATUS_INTERNAL;
        goto done;
    }
    if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire) || trainer->last_step.state == VX_RESULT_STATE_PENDING) {
        status = VX_STATUS_BUSY;
        goto done;
    }
    if (requested->output_count && (trainer->backend == VX_BACKEND_KIND_WEBGPU ||
        !requested->output_names || !requested->outputs)) { status = VX_STATUS_BACKEND_UNSUPPORTED; goto done; }
    if (requested->output_count) {
        VxEngineStateScope validation = vx_engine_state_scope_enter(trainer->engine);
        int valid = 1;
        for (size_t i = 0; i < requested->output_count; i++) {
            int found = 0;
            for (int j = 0; requested->output_names[i] && j < volvoxai_engine_graph_output_count(); j++)
                if (!strcmp(requested->output_names[i], volvoxai_engine_graph_output_name(j))) found = 1;
            for (size_t j = 0; found && j < i; j++)
                if (!strcmp(requested->output_names[j], requested->output_names[i])) found = 0;
            if (!found) valid = 0;
        }
        vx_engine_state_scope_leave(validation);
        if (!valid) { status = VX_STATUS_INVALID_ARGUMENT; goto done; }
    }
    vx_model_profile_begin(trainer->model, &profiling, "TrainStep", trainer->backend_name);
    trainer->engine->profiling = profiling.trace ? &profiling : NULL;
    if (profiling.trace) vx_engine_memory_begin(trainer->engine, &profiling);
    resolved = *requested;
    resolved.optimizer = trainer_optimizer_resolve(trainer, requested);
    if (!trainer_step_options_valid(options, result)) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto done;
    }
    {
        size_t struct_size = result->struct_size;
        memset(result, 0, sizeof(*result));
        result->struct_size = struct_size;
    }
    memset(losses, 0, sizeof(losses));
    memset(metrics, 0, sizeof(metrics));
    for (size_t index = 0; index < options->loss_count; index++) {
        losses[index].name = options->losses[index].name;
        losses[index].logits_name = options->losses[index].logits_name;
        losses[index].targets = (const int*)options->losses[index].targets;
        losses[index].target_count = (int)options->losses[index].target_count;
        losses[index].ignore_index = options->losses[index].ignore_index;
        losses[index].row_index = options->losses[index].row_index;
        losses[index].weight = options->losses[index].weight;
        losses[index].normalizer = options->losses[index].normalizer;
    }
    update_mode = options->optimizer.kind == VX_OPTIMIZER_ADAMW
        ? VOLVOXAI_TENSOR_UPDATE_ADAMW : VOLVOXAI_TENSOR_UPDATE_SGD;
    status = vx_model_internal_bind_authoring_inputs(
        trainer->model, trainer->engine, trainer->backend,
        options->inputs, options->input_count,
        trainer->accumulated_microbatches && !options->reset_accumulation
            ? trainer->accumulation_shape_signature : NULL,
        &shape_signature, &binding_stats, &input_report);
    if (status != VX_STATUS_OK) goto done;
    if (binding_stats.plan_cache_hit) trainer->plan_cache_hits++;
    else {
        trainer->plan_cache_misses++;
        trainer->plan_cache_evictions += binding_stats.plan_cache_evictions;
    }
    free(trainer->binding_signature);
    trainer->binding_signature = trainer_string_copy(shape_signature);
    if (!trainer->binding_signature) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    trainer_record_activation_storage(trainer);
    trainer->engine->native_training_shape_hash =
        trainer_shape_stream_key(shape_signature);
    scope = vx_engine_state_scope_enter(trainer->engine);
#if VOLVOXAI_ENABLE_WEBGPU
    if (trainer->backend == VX_BACKEND_KIND_WEBGPU) {
        status = vx_webgpu_train_begin(options, (long)(trainer->optimizer_step + 1u),
                                         &trainer->pending_step, gpu_evidence, sizeof(gpu_evidence));
        vx_engine_state_scope_leave(scope);
        if (status == VX_STATUS_OK) {
            trainer_record_activation_storage(trainer);
            result->microbatch_id = ++trainer->microbatch_id;
            result->optimizer_step = trainer->optimizer_step;
            result->state = VX_RESULT_STATE_PENDING;
            snprintf(result->backend, sizeof(result->backend), "%s", trainer->backend_name);
            trainer->pending_optimizer_config = options->optimizer;
            trainer->pending_shape_signature = shape_signature;
            shape_signature = NULL;
            trainer->last_step = *result;
            trainer_report(trainer, &trainer->last_step_report, VX_STATUS_OK,
                           VX_STAGE_TRAINER_STEP, VX_CODE_PENDING, "GPU training submitted; query GetTrainStep");
        } else if (status == VX_STATUS_EXECUTION_FAILED) {
            if (trainer_restore_baseline_locked(trainer, report) != VX_STATUS_OK) trainer->poisoned = 1;
        }
        goto done;
    }
#endif
    VxTrainerCapture capture = {trainer, options};
    int device_inputs = 0;
    for (size_t i = 0; i < options->input_count; i++)
        if (options->inputs[i].location == VX_MEMORY_DEVICE) device_inputs = 1;
    core_status = volvoxai_engine_train_step_multi_capture(
        losses, (int)options->loss_count,
        options->trainable_names, (int)options->trainable_count,
        update_mode, options->optimizer.learning_rate,
        options->optimizer.beta1, options->optimizer.beta2,
        options->optimizer.epsilon, options->optimizer.weight_decay,
        options->optimizer.max_gradient_norm,
        (long)(trainer->optimizer_step + 1u),
        (int)options->accumulation_steps,
        options->flush_accumulation, options->reset_accumulation,
        &aggregate_loss, metrics, &accumulated, &update_applied,
        device_inputs ? trainer_prepare_device_inputs : NULL, trainer_capture_forward, &capture);
    vx_engine_state_scope_leave(scope);
    if (core_status != 0 || !isfinite(aggregate_loss) || accumulated < 0 ||
        (update_applied != 0 && update_applied != 1)) {
        VxStatus restore = trainer_restore_baseline_locked(trainer, report);
        if (restore != VX_STATUS_OK) trainer->poisoned = 1;
        status = VX_STATUS_EXECUTION_FAILED;
        goto done;
    }
    trainer->optimizer_config = options->optimizer;
    trainer_record_activation_storage(trainer);
    trainer->microbatch_id++;
    trainer->accumulated_microbatches =
        trainer->engine->training_accumulation.active
            ? (uint32_t)trainer->engine->training_accumulation.microbatches : 0u;
    free(trainer->accumulation_shape_signature);
    trainer->accumulation_shape_signature = NULL;
    if (trainer->accumulated_microbatches) {
        trainer->accumulation_shape_signature = shape_signature;
        shape_signature = NULL;
    }
    if (update_applied) {
        trainer->optimizer_step++;
        trainer->working_dirty = 1;
    }
    {
        result->microbatch_id = trainer->microbatch_id;
        result->optimizer_step = trainer->optimizer_step;
        result->accumulated_microbatches = (uint32_t)accumulated;
        result->update_applied = update_applied;
        result->loss = aggregate_loss;
        result->metric_count = options->loss_count;
        snprintf(result->backend, sizeof(result->backend), "%s",
                 trainer->backend_name);
        for (size_t index = 0; index < options->loss_count; index++) {
            snprintf(result->metrics[index].name,
                     sizeof(result->metrics[index].name), "%s",
                     options->losses[index].name);
            result->metrics[index].loss = metrics[index].loss;
            result->metrics[index].correct = metrics[index].correct;
            result->metrics[index].examples = metrics[index].examples;
            result->metrics[index].normalizer = metrics[index].normalizer;
        }
    }
    status = VX_STATUS_OK;
    result->state = VX_RESULT_STATE_READY;
    trainer->last_step = *result;

done:
    if (status != VX_STATUS_OK && requested->outputs) for (size_t i = 0; i < requested->output_count; i++) {
        if (requested->outputs[i].release) requested->outputs[i].release(requested->outputs[i].owner);
        memset(&requested->outputs[i], 0, sizeof(requested->outputs[i]));
    }
    free(shape_signature);
    if (profiling.trace) {
        trainer->engine->profiling = NULL;
        vx_model_profile_end(trainer->model, &profiling);
    }
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_STEP,
                   status == VX_STATUS_OK ? VX_CODE_NONE :
                   status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
                   status == VX_STATUS_BUSY ? VX_CODE_BUSY :
                   status == VX_STATUS_BACKEND_UNSUPPORTED ? VX_CODE_TRAINING_GRAPH_UNSUPPORTED :
                   status == VX_STATUS_INTERNAL ? VX_CODE_TRAINER_POISONED :
                   input_report.status != VX_STATUS_OK ? input_report.code : VX_CODE_TRAIN_STEP_FAILED,
                   status == VX_STATUS_OK
                       ? (result->state == VX_RESULT_STATE_PENDING
                              ? "GPU training submitted; query GetTrainStep"
                              : update_applied
                              ? "private optimizer update applied; model remains unpublished"
                              : "private gradient accumulation advanced; model remains unpublished")
                       : status == VX_STATUS_BUSY ? "previous training step is pending" :
                         status == VX_STATUS_BACKEND_UNSUPPORTED && gpu_evidence[0] ? gpu_evidence :
                         input_report.status != VX_STATUS_OK ? input_report.message :
                         "training step failed and private work was restored or discarded");
    if (report && input_report.status != VX_STATUS_OK)
        report->input_issue = input_report.input_issue;
    if (status == VX_STATUS_OK) {
        trainer->last_step_report = report ? *report : (VxReport)VX_REPORT_INIT;
    }
    return status;
}

VxStatus vx_trainer_get_step(VxTrainer* trainer, uint64_t microbatch_id,
                              VxTrainStepResult* result, VxReport* report) {
    if (!trainer || !result || result->struct_size != sizeof(*result) ||
        !trainer_report_argument_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&trainer->mutex);
    VxStatus status = trainer->closed ? VX_STATUS_HANDLE_DISPOSED :
        !microbatch_id || microbatch_id != trainer->last_step.microbatch_id
        ? VX_STATUS_NOT_FOUND : VX_STATUS_OK;
    if (status != VX_STATUS_OK) {
        trainer_report(trainer, report, status, VX_STAGE_TRAINER_STEP,
                       VX_CODE_TRAIN_STEP_NOT_FOUND, "trainer has no retained step with this microbatch id");
        pthread_mutex_unlock(&trainer->mutex);
        return status;
    }
#if VOLVOXAI_ENABLE_WEBGPU
    if (trainer->pending_step) {
        VxEngineStateScope scope = vx_engine_state_scope_enter(trainer->engine);
        status = vx_webgpu_train_poll(trainer->pending_step, &trainer->last_step);
        if (status != VX_STATUS_OK || trainer->last_step.state == VX_RESULT_STATE_READY) {
            vx_webgpu_train_release(trainer->pending_step);
            trainer->pending_step = NULL;
            vx_engine_state_scope_leave(scope);
            if (status == VX_STATUS_OK) {
                trainer_record_activation_storage(trainer);
                trainer->optimizer_config = trainer->pending_optimizer_config;
                trainer->accumulated_microbatches = trainer->engine->training_accumulation.active
                    ? (uint32_t)trainer->engine->training_accumulation.microbatches : 0u;
                free(trainer->accumulation_shape_signature);
                trainer->accumulation_shape_signature = trainer->accumulated_microbatches
                    ? trainer->pending_shape_signature : NULL;
                if (trainer->accumulated_microbatches) trainer->pending_shape_signature = NULL;
                if (trainer->last_step.update_applied) {
                    trainer->optimizer_step++;
                    trainer->working_dirty = 1;
                }
                trainer->last_step.optimizer_step = trainer->optimizer_step;
            } else {
                if (trainer_restore_baseline_locked(trainer, report) != VX_STATUS_OK) trainer->poisoned = 1;
                trainer->last_step = (VxTrainStepResult)VX_TRAIN_STEP_RESULT_INIT;
                trainer->last_step.microbatch_id = microbatch_id;
                trainer->last_step.state = VX_RESULT_STATE_FAILED;
                snprintf(trainer->last_step.backend, sizeof(trainer->last_step.backend), "%s", trainer->backend_name);
            }
            free(trainer->pending_shape_signature);
            trainer->pending_shape_signature = NULL;
            trainer->last_step_report = (VxReport)VX_REPORT_INIT;
            trainer_report(trainer, &trainer->last_step_report, status,
                           VX_STAGE_TRAINER_STEP, status == VX_STATUS_OK ? VX_CODE_NONE : VX_CODE_TRAIN_STEP_FAILED,
                           status == VX_STATUS_OK ? "GPU training completed; private state published" :
                           "GPU training failed; committed baseline restored or trainer poisoned");
        } else vx_engine_state_scope_leave(scope);
    }
#endif
    *result = trainer->last_step;
    if (report) *report = trainer->last_step_report;
    status = trainer->last_step_report.status;
    pthread_mutex_unlock(&trainer->mutex);
    return status;
}

VxStatus vx_trainer_commit(VxTrainer* trainer,
                           VxRevisionInfo* published,
                           VxReport* report) {
    VxWeightRevisionRecord* successor = NULL;
    char** weight_paths = NULL;
    size_t weight_path_count = 0;
    char* optimizer_path = NULL;
    VxStatus status;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer ||
        (published && published->struct_size != sizeof(*published))) {
        trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_COMMIT, VX_CODE_INVALID_COMMIT,
                       "trainer or revision output is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
        goto done;
    }
    if (trainer->poisoned || !trainer->engine) {
        status = VX_STATUS_INTERNAL;
        goto done;
    }
    if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire) || trainer->last_step.state == VX_RESULT_STATE_PENDING) {
        status = VX_STATUS_BUSY;
        goto done;
    }
    if (trainer->accumulated_microbatches) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto accumulation_pending;
    }
    if (!trainer->working_dirty) {
        status = VX_STATUS_INVALID_ARGUMENT;
        goto no_update;
    }
    status = trainer_snapshot_weights_locked(
        trainer, &weight_paths, &weight_path_count);
    if (status != VX_STATUS_OK) goto done;
    status = vx_model_internal_prepare_weight_revision(
        trainer->model, (const char* const*)weight_paths, weight_path_count,
        &successor, report);
    if (status != VX_STATUS_OK) goto done;
    status = trainer_snapshot_optimizer_locked(trainer, &optimizer_path);
    if (status != VX_STATUS_OK) goto done;
    status = vx_model_internal_publish_weight_revision(
        trainer->model, trainer->base_revision, successor, report);
    if (status == VX_STATUS_OK) {
        vx_model_internal_release_weight_revision(trainer->base_revision);
        trainer->base_revision = successor;
        successor = NULL;
        vx_model_internal_release_weight_revision(trainer->checkpoint_revision);
        trainer->checkpoint_revision = NULL;
        trainer_owned_path_release(trainer->baseline_optimizer_path);
        trainer->baseline_optimizer_path = optimizer_path;
        optimizer_path = NULL;
        trainer->baseline_optimizer_step = trainer->optimizer_step;
        trainer->baseline_optimizer_config = trainer->optimizer_config;
        trainer->working_dirty = 0;
        if (published) {
            VxRevisionInfo snapshot = VX_REVISION_INFO_INIT;
            if (vx_model_revision_info(trainer->model, &snapshot, NULL) ==
                VX_STATUS_OK) {
                size_t struct_size = published->struct_size;
                *published = snapshot;
                published->struct_size = struct_size;
            }
        }
    }
    goto done;

accumulation_pending:
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_COMMIT,
                   VX_CODE_ACCUMULATION_PENDING,
                   "commit requires an applied optimizer update and no unfinished accumulation");
    return status;
no_update:
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_COMMIT,
                   VX_CODE_NO_APPLIED_UPDATE,
                   "trainer has no private optimizer update to publish");
    return status;

done:
    trainer_temp_paths_release(weight_paths, weight_path_count);
    trainer_owned_path_release(optimizer_path);
    vx_model_internal_release_weight_revision(successor);
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_COMMIT,
                   status == VX_STATUS_OK ? VX_CODE_NONE :
                   status == VX_STATUS_REVISION_CONFLICT
                       ? VX_CODE_REVISION_CONFLICT :
                   status == VX_STATUS_HANDLE_DISPOSED
                       ? VX_CODE_HANDLE_DISPOSED :
                   status == VX_STATUS_INTERNAL
                       ? VX_CODE_TRAINER_POISONED : VX_CODE_COMMIT_FAILED,
                   status == VX_STATUS_OK
                       ? "private weights published as one immutable successor"
                       : "private weights were not published");
    return status;
}

VxStatus vx_trainer_rollback(VxTrainer* trainer, VxReport* report) {
    VxStatus status;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_ROLLBACK, VX_CODE_INVALID_TRAINER,
                       "trainer is NULL");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
    } else if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire) || trainer->last_step.state == VX_RESULT_STATE_PENDING) {
        status = VX_STATUS_BUSY;
    } else {
        status = trainer_restore_baseline_locked(trainer, report);
    }
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_ROLLBACK,
                   status == VX_STATUS_OK ? VX_CODE_NONE :
                   status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
                   VX_CODE_ROLLBACK_FAILED,
                   status == VX_STATUS_OK
                       ? "private training state restored to the committed baseline"
                       : "private training state could not be restored");
    return status;
}

VxStatus vx_trainer_export_weight_bytes(VxTrainer* trainer,
                                         VxTrainerWeightSink sink, void* user,
                                         VxReport* report) {
    VxStatus status = VX_STATUS_OK;
    if (!trainer_report_argument_valid(report)) return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer || !sink) return VX_STATUS_INVALID_ARGUMENT;
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) status = VX_STATUS_HANDLE_DISPOSED;
    else if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire) || trainer->last_step.state == VX_RESULT_STATE_PENDING) status = VX_STATUS_BUSY;
    else if (trainer->poisoned || !trainer->engine) status = VX_STATUS_INTERNAL;
    else if (trainer->accumulated_microbatches) status = VX_STATUS_INVALID_ARGUMENT;
    else {
        VxEngineStateScope scope = vx_engine_state_scope_enter(trainer->engine);
        size_t count = (size_t)trainer->engine->weight_file_count;
        for (size_t index = 0; index < count; index++) {
            unsigned char* bytes = NULL;
            size_t size = 0;
            if (volvoxai_engine_weight_file_bytes((int)index, &bytes, &size) != 0)
                status = VX_STATUS_OUT_OF_MEMORY;
            else if (!sink(user, index, count, bytes, size))
                status = VX_STATUS_OUT_OF_MEMORY;
            free(bytes);
            if (status != VX_STATUS_OK) break;
        }
        vx_engine_state_scope_leave(scope);
    }
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_EXPORT,
                   status == VX_STATUS_OK ? VX_CODE_NONE :
                   status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
                   status == VX_STATUS_INTERNAL ? VX_CODE_TRAINER_POISONED :
                   status == VX_STATUS_INVALID_ARGUMENT ? VX_CODE_EXPORT_MISMATCH :
                   VX_CODE_OUT_OF_MEMORY,
                   status == VX_STATUS_OK ? "private trainer weights exported" :
                   "private trainer weights were not exported");
    return status;
}

VxStatus vx_trainer_export_weights(VxTrainer* trainer,
                                   const char* const* output_paths,
                                   size_t output_path_count,
                                   VxReport* report) {
    VxEngineStateScope scope;
    VxStatus status = VX_STATUS_OK;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer || !output_paths || !output_path_count) {
        trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_EXPORT, VX_CODE_INVALID_EXPORT,
                       "trainer weight export paths are invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < output_path_count; index++) {
        if (!output_paths[index] || !output_paths[index][0]) {
            trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                           VX_STAGE_TRAINER_EXPORT, VX_CODE_INVALID_EXPORT,
                           "trainer weight export path is empty");
            return VX_STATUS_INVALID_ARGUMENT;
        }
        for (size_t previous = 0; previous < index; previous++) {
            if (!strcmp(output_paths[previous], output_paths[index])) {
                trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                               VX_STAGE_TRAINER_EXPORT, VX_CODE_DUPLICATE_EXPORT_PATH,
                               "trainer weight export paths must be unique");
                return VX_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
    } else if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire) || trainer->last_step.state == VX_RESULT_STATE_PENDING) {
        status = VX_STATUS_BUSY;
    } else if (trainer->poisoned || !trainer->engine) {
        status = VX_STATUS_INTERNAL;
    } else if (trainer->accumulated_microbatches) {
        status = VX_STATUS_INVALID_ARGUMENT;
    } else if (output_path_count !=
               (size_t)trainer->engine->weight_file_count) {
        status = VX_STATUS_INVALID_ARGUMENT;
    } else {
        scope = vx_engine_state_scope_enter(trainer->engine);
        for (size_t index = 0; index < output_path_count; index++) {
            if (volvoxai_engine_save_weight_file(
                    (int)index, output_paths[index]) != 0) {
                status = VX_STATUS_IO_ERROR;
                break;
            }
        }
        vx_engine_state_scope_leave(scope);
    }
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_EXPORT,
                   status == VX_STATUS_OK ? VX_CODE_NONE :
                   status == VX_STATUS_HANDLE_DISPOSED ? VX_CODE_HANDLE_DISPOSED :
                   status == VX_STATUS_INTERNAL ? VX_CODE_TRAINER_POISONED :
                   status == VX_STATUS_IO_ERROR ? VX_CODE_EXPORT_FAILED :
                   VX_CODE_EXPORT_MISMATCH,
                   status == VX_STATUS_OK ? "private trainer weights exported" :
                   "private trainer weights were not exported");
    return status;
}

#include "trainer_checkpoint.inc"
