/* VxTrainingService — full-profile training authoring.
 *
 * Trainer state is private: a step mutates only working weights, gradients,
 * optimizer slots, accumulation and RNG, and CommitTrainer is the only
 * operation that publishes a successor Model weight revision.
 *
 * This file exists only in the full profile. The inference profile never
 * registers the service, so its RPCs report "not implemented" rather than
 * silently succeeding.
 */
#include "vx_api_convert.h"
#include "vx_api_buffer.h"
#include "vx_api_handles.h"
#include "vx_training_lifecycle.h"
#include "volvoxai_ffi.h"
#include "vx_api.h"
#include "vx_api_authoring.h"

#include <string.h>

static void vx_api_retain_trainer(void* pointer) {
    vx_trainer_retain((VxTrainer*)pointer);
}

static void vx_api_release_trainer_handle(void* pointer) {
    vx_trainer_release((VxTrainer*)pointer);
}

static void vx_api_training_lineage(VolvoxaiV1OperationReport* report, const VxApiHandleLease* lease) {
    vx_api_report_set_public_lineage(report, lease->lineage.runtime_id, lease->lineage.model_id,
        lease->lineage.compiled_model_id, lease->lineage.context_id);
}

static int vx_api_create_trainer(const VolvoxaiV1CreateTrainerRequest* request,
                                 VolvoxaiV1TrainerHandle* response,
                                 void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxTrainerOptions options = VX_TRAINER_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxTrainer* trainer = NULL;
    VxStatus status;
    size_t count;
    size_t index;
    VolvoxaiV1TensorSpec* specs;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxModel* model;
    int result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_MODEL,
                               request->field_model_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_CREATE,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released model")
                   ? 0 : -1;
    }
    model = (VxModel*)lease.pointer;
    /* An empty backend selects CPU; a named one is an exact requirement. */
    options.backend = vx_api_scratch_cstr(&scratch, &request->field_backend);
    options.rng_seed = request->field_rng_seed;
    options.shape_options = request->field_shape_options;

    status = vx_model_create_trainer(model, &options, &trainer, &report);
    if (status == VX_STATUS_OK && request->field_checkpoint)
        status = vx_trainer_restore_checkpoint(trainer, request->field_checkpoint, &report);
    vx_api_scratch_release(&scratch);
    if (!vx_api_report_attach(allocator, &response->field_report, &report)) {
        result = -1;
        goto done;
    }
    if (status != VX_STATUS_OK) goto done;

    count = vx_trainer_input_count(trainer);
    if (count != 0u) {
        specs = (VolvoxaiV1TensorSpec*)allocator->allocate(allocator->context,
                                                           sizeof(*specs) * count);
        if (!specs) {
            result = -1;
            goto done;
        }
        response->field_inputs.data = specs;
        response->field_inputs.len = count;
        response->field_inputs.cap = count;
        for (index = 0; index < count; index++) {
            VxTensorSpec native = VX_TENSOR_SPEC_INIT;
            VxReport ignored = VX_REPORT_INIT;
            volvoxai_v1_tensor_spec_init_with_allocator(&specs[index], allocator);
            if (vx_trainer_input_spec(trainer, index, &native, &ignored) !=
                VX_STATUS_OK) {
                continue;
            }
            if (!vx_api_tensor_spec_from_native(allocator, &specs[index], &native)) {
                result = -1;
                goto done;
            }
        }
    }

    response->field_trainer_id = vx_api_handle_insert_with_lineage(user_data,
        VX_API_HANDLE_TRAINER, trainer,
        vx_api_retain_trainer, vx_api_release_trainer_handle, &lease.lineage);
    if (!response->field_trainer_id) {
        result = vx_api_report_fail(allocator, &response->field_report,
                                    VX_STATUS_OUT_OF_MEMORY, VX_STAGE_TRAINER_CREATE,
                                    VX_CODE_OUT_OF_MEMORY, "handle registry allocation failed")
                     ? 0 : -1;
    } else {
        trainer = NULL; /* ownership transferred to the registry */
    }
done:
    if (trainer) vx_trainer_release(trainer);
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_release_trainer(const VolvoxaiV1TrainerRef* request,
                                  VolvoxaiV1OperationReport* response,
                                  void* user_data) {
    (void)user_data;
    (void)vx_api_handle_remove(user_data, VX_API_HANDLE_TRAINER, request->field_trainer_id);
    response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_CLOSE;
    return 0;
}

static int vx_api_training_result(VolvoxaiV1TrainStepResult* response,
    const VxTrainStepResult* result, const VxReport* report) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VolvoxaiV1TrainingMetric* metrics;
    size_t index;
    response->field_state = result->state;
    response->field_microbatch_id = result->microbatch_id;
    response->field_optimizer_step = result->optimizer_step;
    response->field_accumulated_microbatches = result->accumulated_microbatches;
    response->field_update_applied = result->update_applied != 0;
    response->field_loss = result->loss;
    if (result->backend[0] &&
        synurang_lite_bytes_assign(allocator, &response->field_backend, result->backend,
                                   strlen(result->backend)) != SYNURANG_LITE_OK) {
        return -1;
    }
    if (result->metric_count) {
        metrics = (VolvoxaiV1TrainingMetric*)allocator->allocate(
            allocator->context, sizeof(*metrics) * result->metric_count);
        if (!metrics) {
            return -1;
        }
        response->field_metrics.data = metrics;
        response->field_metrics.len = 0;
        response->field_metrics.cap = result->metric_count;
        for (index = 0; index < result->metric_count; index++) {
            const VxTrainingMetric* source = &result->metrics[index];
            volvoxai_v1_training_metric_init_with_allocator(&metrics[index], allocator);
            response->field_metrics.len++;
            if (source->name[0] &&
                synurang_lite_bytes_assign(allocator, &metrics[index].field_name,
                                           source->name,
                                           strlen(source->name)) != SYNURANG_LITE_OK) {
                return -1;
            }
            metrics[index].field_loss = source->loss;
            metrics[index].field_correct = source->correct;
            metrics[index].field_examples = source->examples;
            metrics[index].field_normalizer = source->normalizer;
        }
    }
    return vx_api_report_attach(allocator, &response->field_report, report)
                     ? 0 : -1;
}

static int vx_api_train_step(const VolvoxaiV1TrainStepRequest* request,
                             VolvoxaiV1TrainStepResult* response,
                             void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxTrainStepOptions options = VX_TRAIN_STEP_OPTIONS_INIT;
    VxTrainStepResult result = VX_TRAIN_STEP_RESULT_INIT;
    VxReport report = VX_REPORT_INIT;
    VxTensorBinding* bindings = NULL;
    VxCrossEntropyLoss* losses = NULL;
    VxStatus status;
    size_t index;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTrainer* trainer;
    int api_result = 0;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER,
                               request->field_trainer_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_STEP,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released trainer")
                   ? 0 : -1;
    }
    trainer = (VxTrainer*)lease.pointer;

    if (request->field_inputs.len) {
        bindings = (VxTensorBinding*)vx_api_scratch_alloc(
            &scratch, sizeof(*bindings) * request->field_inputs.len);
        if (!bindings) {
            api_result = -1;
            goto done;
        }
        for (index = 0; index < request->field_inputs.len; index++) {
            status = vx_api_binding_from_tensor(&scratch, &bindings[index],
                                                &request->field_inputs.data[index]);
            if (status != VX_STATUS_OK) {
                api_result = vx_api_report_binding_fail(
                    allocator, &response->field_report, status,
                    VX_STAGE_TRAINER_INPUT,
                    "training input could not be bound", &request->field_inputs.data[index], index)
                                 ? 0 : -1;
                goto done;
            }
        }
    }
    options.inputs = bindings;
    options.input_count = request->field_inputs.len;

    if (request->field_losses.len) {
        losses = (VxCrossEntropyLoss*)vx_api_scratch_alloc(
            &scratch, sizeof(*losses) * request->field_losses.len);
        if (!losses) {
            api_result = -1;
            goto done;
        }
        for (index = 0; index < request->field_losses.len; index++) {
            const VolvoxaiV1CrossEntropyLoss* source = &request->field_losses.data[index];
            losses[index] = (VxCrossEntropyLoss)VX_CROSS_ENTROPY_LOSS_INIT;
            losses[index].name = vx_api_scratch_cstr(&scratch, &source->field_name);
            losses[index].logits_name =
                vx_api_scratch_cstr(&scratch, &source->field_logits_name);
            size_t bytes = 0;
            status = vx_api_tensor_bytes(source->field_targets, &bytes);
            if (status != VX_STATUS_OK || source->field_targets->field_dtype != VOLVOXAI_V1_DATA_TYPE_I32) {
                api_result = vx_api_report_fail(allocator, &response->field_report,
                    VX_STATUS_INVALID_ARGUMENT, VX_STAGE_TRAINER_INPUT, VX_CODE_INVALID_ARGUMENT,
                    "cross entropy targets must be dense I32 tensors") ? 0 : -1;
                goto done;
            }
            void* targets = vx_api_scratch_alloc(&scratch, bytes);
            status = targets ? vx_api_tensor_read(&scratch, source->field_targets, targets, bytes) : VX_STATUS_OUT_OF_MEMORY;
            if (status != VX_STATUS_OK) {
                api_result = vx_api_report_fail(allocator, &response->field_report, status,
                    VX_STAGE_TRAINER_INPUT, VX_CODE_INVALID_ARGUMENT, "target storage is unavailable") ? 0 : -1;
                goto done;
            }
            losses[index].targets = targets;
            losses[index].target_count = bytes / sizeof(int32_t);
            /* Absent optional fields keep the engine default. */
            if (source->has_ignore_index) losses[index].ignore_index = source->field_ignore_index;
            if (source->has_row_index) losses[index].row_index = source->field_row_index;
            if (source->has_weight) losses[index].weight = source->field_weight;
            losses[index].normalizer = source->field_normalizer;
        }
        options.losses = losses;
        options.loss_count = request->field_losses.len;
    }

    if (request->field_trainable_names.len) {
        options.trainable_names =
            vx_api_scratch_cstr_array(&scratch, request->field_trainable_names.data,
                                      sizeof(SynurangLiteBytes),
                                      request->field_trainable_names.len);
        options.trainable_count = request->field_trainable_names.len;
    }

    options.optimizer_fields = 0;
    if (request->field_optimizer) {
        const VolvoxaiV1TrainerOptimizerOptions* source = request->field_optimizer;
        if (source->has_kind) {
            options.optimizer.kind = (VxOptimizerKind)source->field_kind;
            options.optimizer_fields |= VX_OPTIMIZER_FIELD_KIND;
        }
#define VX_OPTIMIZER_INPUT(flag, member) \
        if (source->has_##member) { \
            options.optimizer.member = source->field_##member; \
            options.optimizer_fields |= flag; \
        }
        VX_OPTIMIZER_INPUT(VX_OPTIMIZER_FIELD_LEARNING_RATE, learning_rate);
        VX_OPTIMIZER_INPUT(VX_OPTIMIZER_FIELD_BETA1, beta1);
        VX_OPTIMIZER_INPUT(VX_OPTIMIZER_FIELD_BETA2, beta2);
        VX_OPTIMIZER_INPUT(VX_OPTIMIZER_FIELD_EPSILON, epsilon);
        VX_OPTIMIZER_INPUT(VX_OPTIMIZER_FIELD_WEIGHT_DECAY, weight_decay);
        VX_OPTIMIZER_INPUT(VX_OPTIMIZER_FIELD_MAX_GRADIENT_NORM, max_gradient_norm);
#undef VX_OPTIMIZER_INPUT
    }
    if (request->has_accumulation_steps) {
        options.accumulation_steps = request->field_accumulation_steps;
    }
    options.flush_accumulation = request->field_flush_accumulation;
    options.reset_accumulation = request->field_reset_accumulation;

    if (request->field_outputs) {
        options.output_count = request->field_outputs->field_names.len;
        options.output_names = vx_api_scratch_cstr_array(&scratch,
            request->field_outputs->field_names.data, sizeof(SynurangLiteBytes), options.output_count);
        options.outputs = options.output_count ? vx_api_scratch_alloc(&scratch,
            options.output_count * sizeof(*options.outputs)) : NULL;
    }
    if (vx_api_scratch_failed(&scratch)) { api_result = -1; goto done; }
    status = vx_trainer_train_step(trainer, &options, &result, &report);
    if (vx_api_scratch_failed(&scratch)) {
        api_result = -1;
        goto done;
    }
    if (status != VX_STATUS_OK) {
        api_result = vx_api_report_attach(allocator, &response->field_report, &report)
                         ? 0 : -1;
        goto done;
    }

    api_result = vx_api_training_result(response, &result, &report);
    for (size_t i = 0; !api_result && i < options.output_count; i++) {
        VxTrainerTensor* output = &options.outputs[i];
        VolvoxaiV1Tensor* tensor = volvoxai_v1_train_step_result_add_outputs(response);
        VxApiBuffer* buffer = tensor ? vx_api_buffer_create(&output->memory, output->storage,
            output->owner, output->release, 0) : NULL;
        if (!buffer) { api_result = -1; break; }
        output->release = NULL;
        int64_t id = vx_api_buffer_publish(user_data, buffer);
        if (!id) { vx_api_buffer_release(buffer); api_result = -1; break; }
        if (!vx_api_buffer_tensor(allocator, tensor, &output->info, id)) {
            vx_api_handle_remove(user_data, VX_API_HANDLE_BUFFER, id); api_result = -1; break;
        }
    }
done:
    if (options.outputs) for (size_t i = 0; i < options.output_count; i++)
        if (options.outputs[i].release) options.outputs[i].release(options.outputs[i].owner);
    if (api_result) for (size_t i = 0; i < response->field_outputs.len; i++) {
        VolvoxaiV1BufferView* view = response->field_outputs.data[i].field_buffer;
        if (view) { vx_api_handle_remove(user_data, VX_API_HANDLE_BUFFER, view->field_buffer_id); view->field_buffer_id = 0; }
    }
    vx_api_scratch_release(&scratch);
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return api_result;
}

static int vx_api_get_train_step(const VolvoxaiV1TrainStepRef* request,
    VolvoxaiV1TrainStepResult* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER, request->field_trainer_id, &lease))
        return vx_api_report_fail(response->_allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_STEP,
            VX_CODE_HANDLE_DISPOSED, "unknown or released trainer") ? 0 : -1;
    VxTrainStepResult result = VX_TRAIN_STEP_RESULT_INIT;
    VxReport report = VX_REPORT_INIT;
    (void)vx_trainer_get_step(lease.pointer, request->field_microbatch_id, &result, &report);
    int status = vx_api_training_result(response, &result, &report);
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return status;
}

static int vx_api_commit_trainer(const VolvoxaiV1TrainerRef* request,
                                 VolvoxaiV1RevisionInfo* response,
                                 void* user_data) {
    (void)user_data;
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxRevisionInfo published = VX_REVISION_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTrainer* trainer;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER,
                               request->field_trainer_id, &lease)) {
        return vx_api_report_fail(allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_COMMIT,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released trainer")
                   ? 0 : -1;
    }
    trainer = (VxTrainer*)lease.pointer;
    if (vx_trainer_commit(trainer, &published, &report) == VX_STATUS_OK) {
        response->field_graph_id = published.graph_id;
        response->field_graph_revision = published.graph_revision;
        response->field_weight_id = published.weight_id;
        response->field_weight_revision = published.weight_revision;
        response->field_adapter_id = published.adapter_id;
        response->field_adapter_revision = published.adapter_revision;
    }
    result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_rollback_trainer(const VolvoxaiV1TrainerRef* request,
                                   VolvoxaiV1OperationReport* response,
                                   void* user_data) {
    (void)user_data;
    VxReport report = VX_REPORT_INIT;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTrainer* trainer;
    int result;

    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER,
                               request->field_trainer_id, &lease)) {
        response->field_status = VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED;
        response->field_stage = VOLVOXAI_V1_OPERATION_STAGE_TRAINER_ROLLBACK;
        return 0;
    }
    trainer = (VxTrainer*)lease.pointer;
    (void)vx_trainer_rollback(trainer, &report);
    result = vx_api_report_from_native(response, &report) ? 0 : -1;
    vx_api_training_lineage(response, &lease);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_trainer_weight_sink(void* user, size_t index, size_t count,
                                      const unsigned char* bytes, size_t size) {
    VolvoxaiV1TrainerWeights* response = (VolvoxaiV1TrainerWeights*)user;
    const SynurangLiteAllocator* allocator = response->_allocator;
    if (!index) {
        response->field_shards.data = (SynurangLiteBytes*)allocator->allocate(
            allocator->context, count * sizeof(SynurangLiteBytes));
        if (!response->field_shards.data) return 0;
        memset(response->field_shards.data, 0, count * sizeof(SynurangLiteBytes));
        response->field_shards.len = response->field_shards.cap = count;
    }
    return synurang_lite_bytes_assign(allocator, &response->field_shards.data[index],
                                      bytes, size) == SYNURANG_LITE_OK;
}

static int vx_api_export_trainer_weights(
    const VolvoxaiV1ExportTrainerWeightsRequest* request,
    VolvoxaiV1TrainerWeights* response,
    void* user_data) {
    (void)user_data;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxReport report = VX_REPORT_INIT;
    const char* const* paths;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxTrainer* trainer;
    int result;

#if defined(__wasm__)
    if (request->field_output_paths.len)
        return vx_api_report_fail(response->_allocator, &response->field_report,
                                  VX_STATUS_TRANSPORT_UNSUPPORTED, VX_STAGE_TRAINER_EXPORT,
                                  VX_CODE_TRANSPORT_UNSUPPORTED, "WASM exports weight shards as bytes; omit output_paths") ? 0 : -1;
#endif
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER,
                               request->field_trainer_id, &lease)) {
        return vx_api_report_fail(response->_allocator, &response->field_report,
                                  VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_EXPORT,
                                  VX_CODE_HANDLE_DISPOSED, "unknown or released trainer")
                   ? 0 : -1;
    }
    trainer = (VxTrainer*)lease.pointer;
    paths = vx_api_scratch_cstr_array(&scratch, request->field_output_paths.data,
                                      sizeof(SynurangLiteBytes),
                                      request->field_output_paths.len);
    if (vx_api_scratch_failed(&scratch)) {
        vx_api_scratch_release(&scratch);
        vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
        return -1;
    }
    if (!request->field_output_paths.len) {
        (void)vx_trainer_export_weight_bytes(
            trainer, vx_api_trainer_weight_sink, response, &report);
        if (report.status != VX_STATUS_OK) {
            for (size_t index = 0; index < response->field_shards.len; index++)
                synurang_lite_bytes_clear(response->_allocator, &response->field_shards.data[index]);
            response->field_shards.len = 0;
        }
    } else {
        (void)vx_trainer_export_weights(trainer, paths, request->field_output_paths.len,
                                        &report);
    }
    vx_api_scratch_release(&scratch);
    result = vx_api_report_attach(response->_allocator, &response->field_report,
                                  &report) ? 0 : -1;
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_trainer_state_common(VxApiRegistry* user_data, const VolvoxaiV1TrainerRef* request,
    VolvoxaiV1TrainerState* response, int reset) {
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER, request->field_trainer_id, &lease))
        return vx_api_report_fail(response->_allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_STEP, VX_CODE_HANDLE_DISPOSED,
            "unknown or released trainer") ? 0 : -1;
    vx_trainer_state(lease.pointer, response, reset, &report);
    if (report.status != VX_STATUS_OK) {
        const SynurangLiteAllocator* allocator = response->_allocator;
        volvoxai_v1_trainer_state_free(response);
        volvoxai_v1_trainer_state_init_with_allocator(response, allocator);
    }
    int result = vx_api_report_attach(response->_allocator, &response->field_report, &report) ? 0 : -1;
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_get_trainer_state(const VolvoxaiV1TrainerRef* request,
    VolvoxaiV1TrainerState* response, void* user_data) {
    (void)user_data;
    return vx_api_trainer_state_common(user_data, request, response, 0);
}

static int vx_api_reset_trainer_accumulation(const VolvoxaiV1TrainerRef* request,
    VolvoxaiV1TrainerState* response, void* user_data) {
    (void)user_data;
    return vx_api_trainer_state_common(user_data, request, response, 1);
}

static int vx_api_export_trainer_checkpoint(const VolvoxaiV1ExportTrainerCheckpointRequest* request,
    VolvoxaiV1TrainerCheckpointInfo* response, void* user_data) {
    (void)user_data;
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    const SynurangLiteAllocator* allocator = response->_allocator;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER, request->field_trainer_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report,
            VX_STATUS_HANDLE_DISPOSED, VX_STAGE_TRAINER_EXPORT, VX_CODE_HANDLE_DISPOSED,
            "unknown or released trainer") ? 0 : -1;
    response->field_checkpoint = allocator->allocate(allocator->context, sizeof(*response->field_checkpoint));
    if (!response->field_checkpoint) { vx_api_handle_lease_release(&lease); return -1; }
    volvoxai_v1_trainer_checkpoint_init_with_allocator(response->field_checkpoint, allocator);
    vx_trainer_export_checkpoint(lease.pointer, response->field_checkpoint,
        request->field_metadata.data, request->field_metadata.len, request->has_metadata, &report);
    if (report.status != VX_STATUS_OK) {
        volvoxai_v1_trainer_checkpoint_free(response->field_checkpoint);
        allocator->deallocate(allocator->context, response->field_checkpoint);
        response->field_checkpoint = NULL;
    }
    int result = vx_api_report_attach(allocator, &response->field_report, &report) ? 0 : -1;
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_handle_lease_release(&lease);
    return result;
}

static int vx_api_read_trainer_parameters(const VolvoxaiV1ReadTrainerParametersRequest* request,
    VolvoxaiV1TensorBatch* response, void* user_data) {
    const SynurangLiteAllocator* allocator = response->_allocator;
    VxApiScratch scratch = VX_API_SCRATCH_OWNER(user_data);
    VxApiHandleLease lease = VX_API_HANDLE_LEASE_INIT;
    VxReport report = VX_REPORT_INIT;
    int ok = 1;
    if (!vx_api_handle_acquire(user_data, VX_API_HANDLE_TRAINER, request->field_trainer_id, &lease))
        return vx_api_report_fail(allocator, &response->field_report, VX_STATUS_HANDLE_DISPOSED,
            VX_STAGE_TRAINER_EXPORT, VX_CODE_HANDLE_DISPOSED, "trainer is not live") ? 0 : -1;
    size_t count = request->field_names.len;
    const char* const* names = vx_api_scratch_cstr_array(&scratch, request->field_names.data,
        sizeof(SynurangLiteBytes), count);
    VxTrainerTensor* outputs = count ? vx_api_scratch_alloc(&scratch, count * sizeof(*outputs)) : NULL;
    VxStatus status = vx_api_scratch_failed(&scratch) ? VX_STATUS_OUT_OF_MEMORY :
        request->field_mode != VOLVOXAI_V1_PARAMETER_EXPORT_MODE_SNAPSHOT &&
        request->field_mode != VOLVOXAI_V1_PARAMETER_EXPORT_MODE_SHARED_READ ? VX_STATUS_INVALID_ARGUMENT :
        vx_trainer_read_parameters(lease.pointer, names, count,
            request->field_mode == VOLVOXAI_V1_PARAMETER_EXPORT_MODE_SHARED_READ, outputs, &report);
    if (status != VX_STATUS_OK) {
        ok = vx_api_report_fail(allocator, &response->field_report, status, VX_STAGE_TRAINER_EXPORT,
            status == VX_STATUS_BUSY ? VX_CODE_BUSY : VX_CODE_INVALID_ARGUMENT, "parameter storage is unavailable");
        goto done;
    }
    for (size_t i = 0; i < count; i++) {
        VolvoxaiV1Tensor* tensor = volvoxai_v1_tensor_batch_add_outputs(response);
        VxApiBuffer* buffer = tensor ? vx_api_buffer_create(&outputs[i].memory, outputs[i].storage,
            outputs[i].owner, outputs[i].release,
            request->field_mode == VOLVOXAI_V1_PARAMETER_EXPORT_MODE_SHARED_READ) : NULL;
        if (!buffer) { ok = 0; break; }
        outputs[i].release = NULL;
        int64_t id = vx_api_buffer_publish(user_data, buffer);
        if (!id) { vx_api_buffer_release(buffer); ok = 0; break; }
        if (!vx_api_buffer_tensor(allocator, tensor, &outputs[i].info, id)) {
            vx_api_handle_remove(user_data, VX_API_HANDLE_BUFFER, id); ok = 0; break;
        }
    }
    if (ok) ok = vx_api_report_attach(allocator, &response->field_report, &report);
done:
    if (outputs) for (size_t i = 0; i < count; i++)
        if (outputs[i].release) outputs[i].release(outputs[i].owner);
    if (!ok) for (size_t i = 0; i < response->field_outputs.len; i++) {
        VolvoxaiV1BufferView* view = response->field_outputs.data[i].field_buffer;
        if (view) { vx_api_handle_remove(user_data, VX_API_HANDLE_BUFFER, view->field_buffer_id); view->field_buffer_id = 0; }
    }
    vx_api_training_lineage(response->field_report, &lease);
    vx_api_scratch_release(&scratch);
    vx_api_handle_lease_release(&lease);
    return ok ? 0 : -1;
}

#include "vx_api_initializer.inc"

VX_API_UNARY(vx_api_initialize_tensor, VolvoxaiV1InitializeTensorRequest, VolvoxaiV1InitializedTensor,
    volvoxai_v1_initialized_tensor, vx_training_initialize_tensor_respond)
VX_API_UNARY(vx_api_build_lora_linear, VolvoxaiV1BuildLoraLinearRequest, VolvoxaiV1LoraLinearPlan,
    volvoxai_v1_lora_linear_plan, vx_training_build_lora_linear_respond)
VX_API_UNARY(vx_api_build_routed_adapter, VolvoxaiV1BuildRoutedAdapterRequest, VolvoxaiV1RoutedAdapterPlan,
    volvoxai_v1_routed_adapter_plan, vx_training_build_routed_adapter_respond)
VX_API_UNARY(vx_api_create_trainer, VolvoxaiV1CreateTrainerRequest, VolvoxaiV1TrainerHandle,
    volvoxai_v1_trainer_handle, vx_training_create_trainer_respond)
VX_API_UNARY(vx_api_release_trainer, VolvoxaiV1TrainerRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_training_release_trainer_respond)
VX_API_UNARY(vx_api_train_step, VolvoxaiV1TrainStepRequest, VolvoxaiV1TrainStepResult,
    volvoxai_v1_train_step_result, vx_training_train_step_respond)
VX_API_UNARY(vx_api_read_trainer_parameters, VolvoxaiV1ReadTrainerParametersRequest, VolvoxaiV1TensorBatch,
    volvoxai_v1_tensor_batch, vx_training_read_trainer_parameters_respond)
VX_API_UNARY(vx_api_get_train_step, VolvoxaiV1TrainStepRef, VolvoxaiV1TrainStepResult,
    volvoxai_v1_train_step_result, vx_training_get_train_step_respond)
VX_API_UNARY(vx_api_commit_trainer, VolvoxaiV1TrainerRef, VolvoxaiV1RevisionInfo,
    volvoxai_v1_revision_info, vx_training_commit_trainer_respond)
VX_API_UNARY(vx_api_rollback_trainer, VolvoxaiV1TrainerRef, VolvoxaiV1OperationReport,
    volvoxai_v1_operation_report, vx_training_rollback_trainer_respond)
VX_API_UNARY(vx_api_export_trainer_weights, VolvoxaiV1ExportTrainerWeightsRequest, VolvoxaiV1TrainerWeights,
    volvoxai_v1_trainer_weights, vx_training_export_trainer_weights_respond)
VX_API_UNARY(vx_api_get_trainer_state, VolvoxaiV1TrainerRef, VolvoxaiV1TrainerState,
    volvoxai_v1_trainer_state, vx_training_get_trainer_state_respond)
VX_API_UNARY(vx_api_reset_trainer_accumulation, VolvoxaiV1TrainerRef, VolvoxaiV1TrainerState,
    volvoxai_v1_trainer_state, vx_training_reset_trainer_accumulation_respond)
VX_API_UNARY(vx_api_export_trainer_checkpoint, VolvoxaiV1ExportTrainerCheckpointRequest, VolvoxaiV1TrainerCheckpointInfo,
    volvoxai_v1_trainer_checkpoint_info, vx_training_export_trainer_checkpoint_respond)

int vx_api_install_training_handlers(SynurangInstance* instance, VxApiRegistry* registry) {
    VxTrainingServiceHandlers handlers;
    memset(&handlers, 0, sizeof(handlers));
    handlers.initialize_tensor.message = vx_api_initialize_tensor_call;
    handlers.build_lora_linear.message = vx_api_build_lora_linear_call;
    handlers.build_routed_adapter.message = vx_api_build_routed_adapter_call;
    handlers.create_trainer.message = vx_api_create_trainer_call;
    handlers.release_trainer.message = vx_api_release_trainer_call;
    handlers.train_step.message = vx_api_train_step_call;
    handlers.read_trainer_parameters.message = vx_api_read_trainer_parameters_call;
    handlers.get_train_step.message = vx_api_get_train_step_call;
    handlers.commit_trainer.message = vx_api_commit_trainer_call;
    handlers.rollback_trainer.message = vx_api_rollback_trainer_call;
    handlers.export_trainer_weights.message = vx_api_export_trainer_weights_call;
    handlers.get_trainer_state.message = vx_api_get_trainer_state_call;
    handlers.reset_trainer_accumulation.message = vx_api_reset_trainer_accumulation_call;
    handlers.export_trainer_checkpoint.message = vx_api_export_trainer_checkpoint_call;
    return vx_training_register(instance, &handlers, registry);
}
