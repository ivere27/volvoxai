/* Portable Trainer state. Included only by trainer_api.c in the full profile. */
#include "volvoxai_lite.h"

static VolvoxaiV1TrainerOptimizerOptions* trainer_optimizer_message(
    const SynurangLiteAllocator* allocator, const VxOptimizerOptions* value) {
    VolvoxaiV1TrainerOptimizerOptions* out = allocator->allocate(allocator->context, sizeof(*out));
    if (!out) return NULL;
    volvoxai_v1_trainer_optimizer_options_init_with_allocator(out, allocator);
    out->has_kind = 1;
    out->field_kind = (VolvoxaiV1TrainingOptimizerKind)value->kind;
#define VX_OPTIMIZER_OUTPUT(member) out->has_##member = 1; out->field_##member = value->member
    VX_OPTIMIZER_OUTPUT(learning_rate);
    VX_OPTIMIZER_OUTPUT(beta1);
    VX_OPTIMIZER_OUTPUT(beta2);
    VX_OPTIMIZER_OUTPUT(epsilon);
    VX_OPTIMIZER_OUTPUT(weight_decay);
    VX_OPTIMIZER_OUTPUT(max_gradient_norm);
#undef VX_OPTIMIZER_OUTPUT
    return out;
}

static int trainer_optimizer_checkpoint(const VolvoxaiV1TrainerOptimizerOptions* in,
                                        VxOptimizerOptions* out) {
    if (!in || !in->has_kind || !in->has_learning_rate || !in->has_beta1 || !in->has_beta2 ||
        !in->has_epsilon || !in->has_weight_decay || !in->has_max_gradient_norm) return 0;
    *out = (VxOptimizerOptions) {sizeof(*out), (VxOptimizerKind)in->field_kind,
        in->field_learning_rate, in->field_beta1, in->field_beta2, in->field_epsilon,
        in->field_weight_decay, in->field_max_gradient_norm};
    return trainer_optimizer_valid(out);
}

static int trainer_tensor_info(VolvoxaiV1TensorInfo* info, const T* tensor, int location) {
    if (!info || synurang_lite_bytes_assign(info->_allocator, &info->field_name,
        tensor->name, strlen(tensor->name))) return 0;
    info->field_dtype = (VolvoxaiV1DataType)tensor->dtype;
    info->field_byte_size = (uint64_t)tensor->numel * tensor->elem_size;
    info->field_location = location;
    for (int axis = 0; axis < tensor->ndim; ++axis) {
        int64_t* extent = volvoxai_v1_tensor_info_add_shape(info);
        if (!extent) return 0;
        *extent = tensor->shape[axis];
    }
    return 1;
}

static int trainer_shape_state(VxTrainer* trainer, VolvoxaiV1TrainerShapeState* out) {
    VxEngineState* engine = trainer->engine;
    if (trainer->binding_signature && synurang_lite_bytes_assign(out->_allocator,
        &out->field_signature, trainer->binding_signature, strlen(trainer->binding_signature))) return 0;
    out->field_activation_capacity_bytes = trainer_activation_storage(trainer);
    out->field_activation_high_water_bytes = trainer->activation_high_water;
    out->field_activation_grow_count = trainer->activation_grow_count;
    out->field_plan_cache_hits = trainer->plan_cache_hits;
    out->field_plan_cache_misses = trainer->plan_cache_misses;
    out->field_plan_cache_evictions = trainer->plan_cache_evictions;
    out->field_plan_cache_oversize_skips = engine->dynamic_shape_cache_oversize_skips;
    out->field_plan_cache_bypasses = engine->dynamic_shape_cache_bypasses;
    if (synurang_lite_bytes_assign(out->_allocator, &out->field_backend,
        trainer->backend_name, strlen(trainer->backend_name))) return 0;
    uint64_t previous_use = 0;
    for (;;) {
        const VxDynamicShapePlan* plan = NULL;
        for (uint32_t i = 0; i < engine->dynamic_shape_plan_capacity; ++i) {
            const VxDynamicShapePlan* candidate = &engine->dynamic_shape_plans[i];
            if (candidate->signature && candidate->last_use > previous_use &&
                (!plan || candidate->last_use < plan->last_use)) plan = candidate;
        }
        if (!plan) break;
        VolvoxaiV1TrainerCachedPlan* cached = volvoxai_v1_trainer_shape_state_add_cached_plans(out);
        if (!cached || synurang_lite_bytes_assign(out->_allocator, &cached->field_signature,
            plan->signature, strlen(plan->signature))) return 0;
        cached->field_metadata_bytes = plan->metadata_bytes;
        cached->field_logical_bytes = plan->logical_bytes;
        cached->field_arena_bytes = plan->arena_bytes;
        cached->field_last_used = plan->last_use;
        previous_use = plan->last_use;
        out->field_plan_cache_entries++;
    }
    out->field_plan_cache_metadata_bytes = engine->dynamic_shape_plan_metadata_bytes;
    out->field_plan_cache_storage_bytes = engine->dynamic_shape_plan_metadata_bytes +
        (engine->dynamic_shape_plan_capacity - out->field_plan_cache_entries) * (uint64_t)sizeof(VxDynamicShapePlan);
    out->field_options = out->_allocator->allocate(out->_allocator->context, sizeof(*out->field_options));
    if (!out->field_options) return 0;
    volvoxai_v1_trainer_shape_options_init_with_allocator(out->field_options, out->_allocator);
#define VX_SHAPE_POLICY(field) out->field_options->has_##field = 1; \
    out->field_options->field_##field = trainer->shape_policy.field
    VX_SHAPE_POLICY(plan_cache_entries);
    VX_SHAPE_POLICY(plan_cache_metadata_bytes);
    VX_SHAPE_POLICY(max_activation_capacity_bytes);
    VX_SHAPE_POLICY(capacity_growth_factor);
#undef VX_SHAPE_POLICY
    /* A weight may share its mapping with many nodes. Count its storage once
     * in the shard inventory, and leave it out of the activation descriptors. */
    for (int f = 0; f < engine->weight_file_count; f++)
        for (int i = 0; i < engine->weight_files[f].tensor_count; i++)
            out->field_parameter_bytes += engine->weight_files[f].tensors[i].nbytes;
    for (int i = 0; i < engine->tensor_count; i++) {
        const T* tensor = &engine->tensors[i];
        int weight = 0;
        for (int f = 0; f < engine->weight_file_count && !weight; f++)
            weight = safetensors_find_tensor(&engine->weight_files[f], tensor->name) != NULL;
        if (tensor->numel <= 0) continue;
        int location = trainer->backend == VX_PORTABLE_BACKEND_KIND ?
            VOLVOXAI_V1_MEMORY_LOCATION_HOST : VOLVOXAI_V1_MEMORY_LOCATION_DEVICE;
        if (!weight && !trainer_tensor_info(
            volvoxai_v1_trainer_shape_state_add_activations(out), tensor, location)) return 0;
        if (tensor->dtype == T_F32 && !trainer_tensor_info(weight ?
            volvoxai_v1_trainer_shape_state_add_parameter_gradients(out) :
            volvoxai_v1_trainer_shape_state_add_activation_gradients(out), tensor, location)) return 0;
    }
    return 1;
}

static int trainer_read_file(const char* path, unsigned char** bytes, size_t* size) {
    FILE* file = fopen(path, "rb");
    long length;
    *bytes = NULL; *size = 0;
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) || (length = ftell(file)) < 0 ||
        (uint64_t)length > SIZE_MAX || fseek(file, 0, SEEK_SET)) {
        fclose(file); return 0;
    }
    unsigned char* data = malloc(length ? (size_t)length : 1u);
    int ok = data && fread(data, 1, (size_t)length, file) == (size_t)length;
    if (fclose(file)) ok = 0;
    if (!ok) { free(data); return 0; }
    *bytes = data; *size = (size_t)length;
    return 1;
}

static int trainer_stage_bytes(const SynurangLiteBytes* bytes, char** path) {
    if (!bytes || !bytes->data || !bytes->len || trainer_temp_path(path)) return 0;
    FILE* file = fopen(*path, "wb");
    if (!file) return 0;
    int ok = fwrite(bytes->data, 1, bytes->len, file) == bytes->len;
    if (fclose(file)) ok = 0;
    return ok;
}

static VxStatus trainer_idle_status(const VxTrainer* trainer) {
    if (trainer->closed) return VX_STATUS_HANDLE_DISPOSED;
    if (atomic_load_explicit(&trainer->shared_readers, memory_order_acquire) ||
        trainer->last_step.state == VX_RESULT_STATE_PENDING) return VX_STATUS_BUSY;
    if (trainer->poisoned || !trainer->engine) return VX_STATUS_INTERNAL;
    return VX_STATUS_OK;
}

VxStatus vx_trainer_state(VxTrainer* trainer, VolvoxaiV1TrainerState* state,
                          int reset, VxReport* report) {
    if (!trainer || !state || !trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    VxStatus status;
    pthread_mutex_lock(&trainer->mutex);
    status = trainer_idle_status(trainer);
    if (status == VX_STATUS_BUSY && !reset) status = VX_STATUS_OK;
    if (status == VX_STATUS_OK) {
        VxEngineState* engine = trainer->engine;
        if (reset) {
            VxEngineStateScope scope = vx_engine_state_scope_enter(engine);
            int reset_status = volvoxai_engine_reset_gradient_accumulation();
            vx_engine_state_scope_leave(scope);
            if (reset_status) { status = VX_STATUS_INTERNAL; goto done; }
            trainer->accumulated_microbatches = 0;
            free(trainer->accumulation_shape_signature);
            trainer->accumulation_shape_signature = NULL;
        }
        state->field_optimizer_config = trainer_optimizer_message(state->_allocator, &trainer->optimizer_config);
        state->field_shape = state->_allocator->allocate(state->_allocator->context, sizeof(*state->field_shape));
        if (state->field_shape)
            volvoxai_v1_trainer_shape_state_init_with_allocator(state->field_shape, state->_allocator);
        if (!state->field_optimizer_config || !state->field_shape || !trainer_shape_state(trainer, state->field_shape)) {
            status = VX_STATUS_OUT_OF_MEMORY;
            goto done;
        }
        const TrainingAccumulationState* window = &engine->training_accumulation;
        state->field_optimizer_step = trainer->optimizer_step;
        state->field_rng_seed = trainer->rng_seed;
        state->field_accumulated_microbatches = trainer->accumulated_microbatches;
        state->field_accumulation_steps = window->active ? window->accumulation_steps : 0;
        state->field_step_pending = trainer->last_step.state == VX_RESULT_STATE_PENDING;
        state->field_has_private_updates = trainer->working_dirty != 0;
        if (trainer->accumulation_shape_signature && synurang_lite_bytes_assign(
            state->_allocator, &state->field_activation_signature,
            trainer->accumulation_shape_signature, strlen(trainer->accumulation_shape_signature)))
            status = VX_STATUS_OUT_OF_MEMORY;
        for (size_t i = 0; engine->optimizer_states && i < engine->tensor_capacity; i++) {
            const EngineOptimizerState* slot = &engine->optimizer_states[i];
            if (slot->numel > 0)
                state->field_optimizer_bytes += (uint64_t)slot->numel * sizeof(float) *
                    ((slot->m != NULL) + (slot->v != NULL));
        }
        for (int i = 0; i < window->trainable_count; i++)
            state->field_gradient_bytes += (uint64_t)window->trainable_numel[i] * sizeof(float);
        for (int i = 0; i < window->loss_count && status == VX_STATUS_OK; i++) {
            const TrainingAccumulationLoss* loss = &window->losses[i];
            VolvoxaiV1TrainingMetric* metric = volvoxai_v1_trainer_state_add_accumulated_metrics(state);
            if (!metric || synurang_lite_bytes_assign(state->_allocator, &metric->field_name,
                loss->name, strlen(loss->name))) { status = VX_STATUS_OUT_OF_MEMORY; break; }
            metric->field_loss = loss->metric.loss;
            metric->field_correct = loss->metric.correct;
            metric->field_examples = loss->metric.examples;
            metric->field_normalizer = loss->metric.normalizer;
        }
    }
done:
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_STEP,
        status == VX_STATUS_OK ? VX_CODE_NONE : status == VX_STATUS_BUSY ? VX_CODE_BUSY : VX_CODE_TRAINER_STATE_FAILED,
        status == VX_STATUS_OK ? (reset ? "unfinished gradients discarded" : "trainer state inspected") :
            "trainer state is unavailable for this operation");
    return status;
}

VxStatus vx_trainer_export_checkpoint(VxTrainer* trainer, VolvoxaiV1TrainerCheckpoint* output,
    const void* metadata, size_t metadata_bytes, int replace_metadata, VxReport* report) {
    if (!trainer || !output || (metadata_bytes && !metadata) || !trainer_report_argument_valid(report))
        return VX_STATUS_INVALID_ARGUMENT;
    const SynurangLiteAllocator* allocator = output->_allocator;
    VxStatus status;
    char* optimizer_path = NULL;
    unsigned char* data = NULL;
    size_t size = 0;
    pthread_mutex_lock(&trainer->mutex);
    status = trainer_idle_status(trainer);
    if (status != VX_STATUS_OK) goto done;
    if (trainer->accumulated_microbatches) { status = VX_STATUS_BUSY; goto done; }
    status = VX_STATUS_OUT_OF_MEMORY;
    output->field_optimizer_config = trainer_optimizer_message(allocator, &trainer->optimizer_config);
    if (!output->field_optimizer_config) goto done;
    if (!trainer_read_file(vx_model_internal_graph_path(trainer->model), &data, &size) ||
        synurang_lite_bytes_assign(allocator, &output->field_graph, data, size)) goto done;
    free(data); data = NULL;
    if (synurang_lite_bytes_assign(allocator, &output->field_metadata,
        replace_metadata ? metadata : trainer->checkpoint_metadata,
        replace_metadata ? metadata_bytes : trainer->checkpoint_metadata_bytes)) goto done;
    {
        VxEngineStateScope scope = vx_engine_state_scope_enter(trainer->engine);
        for (int i = 0; i < trainer->engine->weight_file_count; i++) {
            SynurangLiteBytes* shard = volvoxai_v1_trainer_checkpoint_add_weight_shards(output);
            if (!shard || volvoxai_engine_weight_file_bytes(i, &data, &size) ||
                synurang_lite_bytes_assign(allocator, shard, data, size)) break;
            free(data); data = NULL;
            if (i == trainer->engine->weight_file_count - 1) status = VX_STATUS_OK;
        }
        vx_engine_state_scope_leave(scope);
    }
    if (status != VX_STATUS_OK) goto done;
    status = trainer_snapshot_optimizer_locked(trainer, &optimizer_path);
    if (status != VX_STATUS_OK) goto done;
    if (!trainer_read_file(optimizer_path, &data, &size) ||
        synurang_lite_bytes_assign(allocator, &output->field_optimizer, data, size)) {
        status = VX_STATUS_OUT_OF_MEMORY; goto done;
    }
    output->field_version = 1;
    output->field_optimizer_step = trainer->optimizer_step;
    output->field_rng_seed = trainer->rng_seed;
done:
    free(data);
    trainer_owned_path_release(optimizer_path);
    pthread_mutex_unlock(&trainer->mutex);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_EXPORT,
        status == VX_STATUS_OK ? VX_CODE_NONE : status == VX_STATUS_BUSY ? VX_CODE_BUSY : VX_CODE_CHECKPOINT_EXPORT_FAILED,
        status == VX_STATUS_OK ? "portable trainer checkpoint exported" :
            "checkpoint requires an idle trainer with no unfinished accumulation");
    return status;
}

/* Called during CreateTrainer, before the new owner is visible to any caller. */
VxStatus vx_trainer_restore_checkpoint(VxTrainer* trainer, const VolvoxaiV1TrainerCheckpoint* input,
                                       VxReport* report) {
    VxStatus status = VX_STATUS_INVALID_ARGUMENT;
    char** paths = NULL;
    char* optimizer_path = NULL;
    unsigned char* graph = NULL;
    unsigned char* metadata = NULL;
    size_t graph_bytes = 0, count = input ? input->field_weight_shards.len : 0;
    VxWeightRevisionRecord* revision = NULL;
    VxEngineState* engine = NULL;
    VxOptimizerOptions optimizer_config;
    if (!trainer || !input || !trainer_report_argument_valid(report)) return status;
    if (input->field_version != 1 || input->field_optimizer_step > (uint64_t)LONG_MAX ||
        !count || count > MAX_WEIGHT_FILES || !input->field_optimizer.len || !input->field_graph.len ||
        !trainer_optimizer_checkpoint(input->field_optimizer_config, &optimizer_config))
        goto done;
    if (!trainer_read_file(vx_model_internal_graph_path(trainer->model), &graph, &graph_bytes)) {
        status = VX_STATUS_IO_ERROR; goto done;
    }
    if (graph_bytes != input->field_graph.len || memcmp(graph, input->field_graph.data, graph_bytes))
        goto done;
    paths = calloc(count, sizeof(*paths));
    if (!paths) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
    for (size_t i = 0; i < count; i++)
        if (!trainer_stage_bytes(&input->field_weight_shards.data[i], &paths[i])) {
            status = VX_STATUS_IO_ERROR; goto done;
        }
    if (!trainer_stage_bytes(&input->field_optimizer, &optimizer_path)) {
        status = VX_STATUS_IO_ERROR; goto done;
    }
    status = vx_model_internal_prepare_weight_revision(trainer->model,
        (const char* const*)paths, count, &revision, report);
    if (status != VX_STATUS_OK) goto done;
    status = vx_model_internal_create_authoring_engine(trainer->model, revision,
        trainer->backend, &trainer->shape_policy, &engine, report);
    if (status != VX_STATUS_OK) goto done;
    status = VX_STATUS_INVALID_ARGUMENT;
    if (engine->weight_file_count != trainer->engine->weight_file_count) goto done;
    for (int i = 0; i < engine->weight_file_count; i++) {
        const SafetensorsFile* next = &engine->weight_files[i];
        const SafetensorsFile* original = &trainer->engine->weight_files[i];
        if (next->tensor_count != original->tensor_count) goto done;
        for (int j = 0; j < next->tensor_count; j++) {
            const SafetensorsTensor* value = &next->tensors[j];
            const SafetensorsTensor* expected = safetensors_find_tensor(original, value->name);
            if (!expected || value->dtype != expected->dtype || value->ndim != expected->ndim ||
                value->nbytes != expected->nbytes || memcmp(value->shape, expected->shape, value->ndim * sizeof(int))) goto done;
            if (value->dtype == SAFETENSORS_DTYPE_F32)
                for (size_t k = 0; k < value->nbytes / sizeof(float); k++)
                    if (!isfinite(((const float*)value->data)[k])) goto done;
        }
    }
    {
        VxEngineStateScope scope = vx_engine_state_scope_enter(engine);
        long step = -1;
        int ok = (trainer->backend == VX_BACKEND_KIND_WEBGPU ||
            !volvoxai_engine_require_training_backend(trainer->training_backend)) &&
            !volvoxai_engine_load_optimizer_state(optimizer_path, &step) &&
            step >= 0 && (uint64_t)step == input->field_optimizer_step;
        vx_engine_state_scope_leave(scope);
        if (!ok) goto done;
    }
    if (input->field_metadata.len) {
        metadata = malloc(input->field_metadata.len);
        if (!metadata) { status = VX_STATUS_OUT_OF_MEMORY; goto done; }
        memcpy(metadata, input->field_metadata.data, input->field_metadata.len);
    }
    vx_model_internal_destroy_authoring_engine(trainer->engine);
    trainer->engine = engine; engine = NULL;
    trainer->activation_capacity = trainer->activation_high_water = trainer->activation_grow_count = 0;
    trainer_record_activation_storage(trainer);
    trainer->checkpoint_revision = revision; revision = NULL;
    trainer->baseline_optimizer_path = optimizer_path; optimizer_path = NULL;
    trainer->baseline_optimizer_step = trainer->optimizer_step = input->field_optimizer_step;
    trainer->baseline_optimizer_config = trainer->optimizer_config = optimizer_config;
    trainer->rng_seed = input->field_rng_seed;
    trainer->rng_stream_key = trainer_rng_stream_key(trainer->rng_seed);
    trainer->engine->native_training_rng_seed = trainer->rng_stream_key;
    trainer->working_dirty = 1;
    trainer->checkpoint_metadata = metadata; metadata = NULL;
    trainer->checkpoint_metadata_bytes = input->field_metadata.len;
    status = VX_STATUS_OK;
done:
    free(graph); free(metadata);
    trainer_temp_paths_release(paths, count);
    trainer_owned_path_release(optimizer_path);
    vx_model_internal_destroy_authoring_engine(engine);
    vx_model_internal_release_weight_revision(revision);
    trainer_report(trainer, report, status, VX_STAGE_TRAINER_CREATE,
        status == VX_STATUS_OK ? VX_CODE_NONE : VX_CODE_INVALID_CHECKPOINT,
        status == VX_STATUS_OK ? "private trainer restored from portable checkpoint" :
            "checkpoint graph, weights, optimizer or version is invalid");
    return status;
}
