#ifndef _POSIX_C_SOURCE
#define _POSIX_C_SOURCE 200809L
#endif

#include "volvoxai_full.h"

#include "public_api_internal.h"
#include "runtime_state.h"
#include "training_core.h"

#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
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
    int poisoned;
    VxModel* model;
    VxWeightRevisionRecord* base_revision;
    VxEngineState* engine;
    VolvoxAIEngineBackend backend;
    int training_backend;
    char backend_name[VX_REPORT_BACKEND_CAPACITY];
    uint32_t rng_stream_key;
    uint64_t microbatch_id;
    uint64_t optimizer_step;
    uint64_t baseline_optimizer_step;
    char* baseline_optimizer_path;
    char* accumulation_shape_signature;
    int working_dirty;
    uint32_t accumulated_microbatches;
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

static int trainer_report_argument_valid(const VxReport* report) {
    return !report || report->struct_size == sizeof(*report);
}

static void trainer_report(VxTrainer* trainer,
                           VxReport* report,
                           VxStatus status,
                           VxStage stage,
                           const char* reason,
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
    snprintf(report->reason, sizeof(report->reason), "%s",
             reason ? reason : "");
    snprintf(report->message, sizeof(report->message), "%s",
             message ? message : "");
}

static int trainer_backend_parse(const char* name,
                                 VolvoxAIEngineBackend* backend,
                                 int* training_backend) {
    const char* selected = name && name[0] ? name : "cpu";
    if (!backend || !training_backend) return -1;
    if (!strcmp(selected, "cpu")) {
        *backend = VOLVOXAI_BACKEND_CPU;
        *training_backend = VOLVOXAI_TRAINING_BACKEND_ALLOW_FALLBACK;
    } else if (!strcmp(selected, "vulkan")) {
        *backend = VOLVOXAI_BACKEND_VULKAN;
        *training_backend = VOLVOXAI_TRAINING_BACKEND_VULKAN;
    } else if (!strcmp(selected, "opengl")) {
        *backend = VOLVOXAI_BACKEND_OPENGL;
        *training_backend = VOLVOXAI_TRAINING_BACKEND_OPENGL;
    } else if (!strcmp(selected, "metal")) {
        *backend = VOLVOXAI_BACKEND_METAL;
        *training_backend = VOLVOXAI_TRAINING_BACKEND_METAL;
    } else if (!strcmp(selected, "cuda")) {
        *backend = VOLVOXAI_BACKEND_CUDA;
        *training_backend = VOLVOXAI_TRAINING_BACKEND_CUDA;
    } else {
        return -1;
    }
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
    status = volvoxai_engine_require_training_backend(
        trainer->training_backend);
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
                   VX_STAGE_TRAINER_CREATE, "TRAINING_PREFLIGHT_FAILED",
                   "required training backend or optimizer baseline is unavailable");
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static VxStatus trainer_restore_baseline_locked(VxTrainer* trainer,
                                                 VxReport* report) {
    VxEngineState* replacement = NULL;
    VxStatus status;
    status = vx_model_internal_create_authoring_engine(
        trainer->model, trainer->base_revision, trainer->backend,
        &replacement, report);
    if (status != VX_STATUS_OK) return status;
    status = trainer_configure_engine_locked(trainer, replacement, report);
    if (status != VX_STATUS_OK) {
        vx_model_internal_destroy_authoring_engine(replacement);
        return status;
    }
    vx_model_internal_destroy_authoring_engine(trainer->engine);
    trainer->engine = replacement;
    trainer->optimizer_step = trainer->baseline_optimizer_step;
    trainer->working_dirty = 0;
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
                       VX_STAGE_TRAINER_CREATE, "INVALID_TRAINER_OPTIONS",
                       "model, options, or trainer output is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    *out_trainer = NULL;
    if (options) resolved = *options;
    if (resolved.backend && !resolved.backend[0]) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_CREATE, "INVALID_TRAINING_BACKEND",
                       "training backend name must be non-empty when provided");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    backend_name = resolved.backend && resolved.backend[0]
        ? resolved.backend : "cpu";
    if (strlen(backend_name) >= VX_REPORT_BACKEND_CAPACITY) {
        trainer_report(NULL, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_CREATE, "INVALID_TRAINING_BACKEND",
                       "training backend name is invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    trainer = (VxTrainer*)calloc(1, sizeof(*trainer));
    if (!trainer) {
        trainer_report(NULL, report, VX_STATUS_OUT_OF_MEMORY,
                       VX_STAGE_TRAINER_CREATE, "OUT_OF_MEMORY",
                       "trainer allocation failed");
        return VX_STATUS_OUT_OF_MEMORY;
    }
    if (trainer_backend_parse(backend_name, &trainer->backend,
                              &trainer->training_backend) != 0) {
        free(trainer);
        trainer_report(NULL, report, VX_STATUS_BACKEND_UNSUPPORTED,
                       VX_STAGE_TRAINER_CREATE, "TRAINING_BACKEND_UNSUPPORTED",
                       "trainer requires cpu, vulkan, opengl, metal, or cuda");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    atomic_init(&trainer->references, 1);
    if (pthread_mutex_init(&trainer->mutex, NULL) != 0) {
        free(trainer);
        trainer_report(NULL, report, VX_STATUS_INTERNAL,
                       VX_STAGE_TRAINER_CREATE, "MUTEX_INIT_FAILED",
                       "trainer mutex initialization failed");
        return VX_STATUS_INTERNAL;
    }
    trainer->model = model;
    vx_model_retain(model);
    trainer->rng_stream_key = trainer_rng_stream_key(resolved.rng_seed);
    snprintf(trainer->backend_name, sizeof(trainer->backend_name), "%s",
             backend_name);
    status = vx_model_internal_accept_full_owner(
        model, &trainer->base_revision, report);
    if (status != VX_STATUS_OK) goto fail;
    status = vx_model_internal_create_authoring_engine(
        model, trainer->base_revision, trainer->backend,
        &trainer->engine, report);
    if (status != VX_STATUS_OK) goto fail;
    status = trainer_configure_engine_locked(trainer, trainer->engine, report);
    if (status != VX_STATUS_OK) goto fail;
    *out_trainer = trainer;
    trainer_report(trainer, report, VX_STATUS_OK, VX_STAGE_TRAINER_CREATE,
                   "OK", "trainer created with a private exact-revision engine");
    return VX_STATUS_OK;

fail:
    trainer_report(
        trainer, report, status, VX_STAGE_TRAINER_CREATE,
        status == VX_STATUS_INTERNAL ? "REVISION_MISSING" :
        status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
        status == VX_STATUS_BACKEND_UNAVAILABLE ? "TRAINING_BACKEND_UNAVAILABLE" :
        status == VX_STATUS_BACKEND_UNSUPPORTED ? "TRAINING_PREFLIGHT_FAILED" :
        status == VX_STATUS_OUT_OF_MEMORY ? "OUT_OF_MEMORY" :
        "TRAINER_CREATE_FAILED",
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
                       VX_STAGE_CLOSE, "INVALID_TRAINER", "trainer is NULL");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        pthread_mutex_unlock(&trainer->mutex);
        trainer_report(trainer, report, VX_STATUS_OK, VX_STAGE_CLOSE, "OK",
                       "trainer already closed");
        return VX_STATUS_OK;
    }
    trainer->closed = 1;
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
    trainer_owned_path_release(optimizer_path);
    free(accumulation_shape_signature);
    trainer_report(trainer, report, VX_STATUS_OK, VX_STAGE_CLOSE, "OK",
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
                       VX_STAGE_TRAINER_INPUT, "INVALID_INPUT_QUERY",
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
                   status == VX_STATUS_OK ? "OK" :
                   status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
                   status == VX_STATUS_NOT_FOUND ? "INPUT_NOT_FOUND" :
                   "INPUT_QUERY_FAILED",
                   status == VX_STATUS_OK ? "logical trainer input spec returned" :
                   "logical trainer input spec is unavailable");
    return status;
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
        (options->optimizer.kind != VX_OPTIMIZER_SGD &&
         options->optimizer.kind != VX_OPTIMIZER_ADAMW) ||
        !isfinite(options->optimizer.learning_rate) ||
        options->optimizer.learning_rate < 0.0f ||
        !isfinite(options->optimizer.beta1) ||
        options->optimizer.beta1 < 0.0f ||
        options->optimizer.beta1 >= 1.0f ||
        !isfinite(options->optimizer.beta2) ||
        options->optimizer.beta2 < 0.0f ||
        options->optimizer.beta2 >= 1.0f ||
        !isfinite(options->optimizer.epsilon) ||
        options->optimizer.epsilon <= 0.0f ||
        !isfinite(options->optimizer.weight_decay) ||
        options->optimizer.weight_decay < 0.0f ||
        !isfinite(options->optimizer.max_gradient_norm) ||
        options->optimizer.max_gradient_norm < 0.0f)
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

VxStatus vx_trainer_train_step(VxTrainer* trainer,
                               const VxTrainStepOptions* options,
                               VxTrainStepResult* result,
                               VxReport* report) {
    volvoxai_cross_entropy_loss_t losses[VX_MAX_TRAINING_LOSSES];
    volvoxai_cross_entropy_metric_t metrics[VX_MAX_TRAINING_LOSSES];
    VxEngineStateScope scope;
    VxStatus status = VX_STATUS_EXECUTION_FAILED;
    float aggregate_loss = 0.0f;
    int accumulated = 0;
    int update_applied = 0;
    int update_mode;
    int core_status;
    char* shape_signature = NULL;
    if (!trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!trainer || !trainer_step_options_valid(options, result)) {
        trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                       VX_STAGE_TRAINER_STEP, "INVALID_TRAIN_STEP",
                       "loss, optimizer, accumulation, trainable, or result options are invalid");
        return VX_STATUS_INVALID_ARGUMENT;
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
    status = vx_model_internal_bind_authoring_inputs(
        trainer->model, trainer->engine, trainer->backend,
        options->inputs, options->input_count,
        trainer->accumulated_microbatches && !options->reset_accumulation
            ? trainer->accumulation_shape_signature : NULL,
        &shape_signature, report);
    if (status != VX_STATUS_OK) goto done;
    trainer->engine->native_training_shape_hash =
        trainer_shape_stream_key(shape_signature);
    scope = vx_engine_state_scope_enter(trainer->engine);
    core_status = volvoxai_engine_train_step_multi(
        losses, (int)options->loss_count,
        options->trainable_names, (int)options->trainable_count,
        update_mode, options->optimizer.learning_rate,
        options->optimizer.beta1, options->optimizer.beta2,
        options->optimizer.epsilon, options->optimizer.weight_decay,
        options->optimizer.max_gradient_norm,
        (long)(trainer->optimizer_step + 1u),
        (int)options->accumulation_steps,
        options->flush_accumulation, options->reset_accumulation,
        &aggregate_loss, metrics, &accumulated, &update_applied);
    vx_engine_state_scope_leave(scope);
    if (core_status != 0 || !isfinite(aggregate_loss) || accumulated < 0 ||
        (update_applied != 0 && update_applied != 1)) {
        VxStatus restore = trainer_restore_baseline_locked(trainer, report);
        if (restore != VX_STATUS_OK) trainer->poisoned = 1;
        status = VX_STATUS_EXECUTION_FAILED;
        goto done;
    }
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

done:
    free(shape_signature);
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_STEP,
                   status == VX_STATUS_OK ? "OK" :
                   status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
                   status == VX_STATUS_INTERNAL ? "TRAINER_POISONED" :
                   "TRAIN_STEP_FAILED",
                   status == VX_STATUS_OK
                       ? (update_applied
                              ? "private optimizer update applied; model remains unpublished"
                              : "private gradient accumulation advanced; model remains unpublished")
                       : "training step failed and private work was restored or discarded");
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
                       VX_STAGE_TRAINER_COMMIT, "INVALID_COMMIT",
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
        trainer_owned_path_release(trainer->baseline_optimizer_path);
        trainer->baseline_optimizer_path = optimizer_path;
        optimizer_path = NULL;
        trainer->baseline_optimizer_step = trainer->optimizer_step;
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
                   "ACCUMULATION_PENDING",
                   "commit requires an applied optimizer update and no unfinished accumulation");
    return status;
no_update:
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_COMMIT,
                   "NO_APPLIED_UPDATE",
                   "trainer has no private optimizer update to publish");
    return status;

done:
    trainer_temp_paths_release(weight_paths, weight_path_count);
    trainer_owned_path_release(optimizer_path);
    vx_model_internal_release_weight_revision(successor);
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_COMMIT,
                   status == VX_STATUS_OK ? "OK" :
                   status == VX_STATUS_REVISION_CONFLICT
                       ? "REVISION_CONFLICT" :
                   status == VX_STATUS_HANDLE_DISPOSED
                       ? "HANDLE_DISPOSED" :
                   status == VX_STATUS_INTERNAL
                       ? "TRAINER_POISONED" : "COMMIT_FAILED",
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
                       VX_STAGE_TRAINER_ROLLBACK, "INVALID_TRAINER",
                       "trainer is NULL");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
    } else {
        status = trainer_restore_baseline_locked(trainer, report);
    }
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_ROLLBACK,
                   status == VX_STATUS_OK ? "OK" :
                   status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
                   "ROLLBACK_FAILED",
                   status == VX_STATUS_OK
                       ? "private training state restored to the committed baseline"
                       : "private training state could not be restored");
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
                       VX_STAGE_TRAINER_EXPORT, "INVALID_EXPORT",
                       "trainer weight export paths are invalid");
        return VX_STATUS_INVALID_ARGUMENT;
    }
    for (size_t index = 0; index < output_path_count; index++) {
        if (!output_paths[index] || !output_paths[index][0]) {
            trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                           VX_STAGE_TRAINER_EXPORT, "INVALID_EXPORT",
                           "trainer weight export path is empty");
            return VX_STATUS_INVALID_ARGUMENT;
        }
        for (size_t previous = 0; previous < index; previous++) {
            if (!strcmp(output_paths[previous], output_paths[index])) {
                trainer_report(trainer, report, VX_STATUS_INVALID_ARGUMENT,
                               VX_STAGE_TRAINER_EXPORT, "DUPLICATE_EXPORT_PATH",
                               "trainer weight export paths must be unique");
                return VX_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    pthread_mutex_lock(&trainer->mutex);
    if (trainer->closed) {
        status = VX_STATUS_HANDLE_DISPOSED;
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
                   status == VX_STATUS_OK ? "OK" :
                   status == VX_STATUS_HANDLE_DISPOSED ? "HANDLE_DISPOSED" :
                   status == VX_STATUS_INTERNAL ? "TRAINER_POISONED" :
                   status == VX_STATUS_IO_ERROR ? "EXPORT_FAILED" :
                   "EXPORT_MISMATCH",
                   status == VX_STATUS_OK ? "private trainer weights exported" :
                   "private trainer weights were not exported");
    return status;
}
