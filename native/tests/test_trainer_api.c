#include "volvoxai.h"
#include "volvoxai_full.h"
#include "safetensors.h"

#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

_Static_assert(VX_STATUS_REVISION_CONFLICT == -17,
               "protobuf training status contract");
_Static_assert(VX_STAGE_TRAINER_CREATE == 11 &&
               VX_STAGE_TRAINER_INPUT == 12 &&
               VX_STAGE_TRAINER_STEP == 13 &&
               VX_STAGE_TRAINER_COMMIT == 14 &&
               VX_STAGE_TRAINER_ROLLBACK == 15 &&
               VX_STAGE_TRAINER_EXPORT == 16,
               "protobuf training stage contract");
_Static_assert(VX_OPTIMIZER_ADAMW == 0 && VX_OPTIMIZER_SGD == 1,
               "protobuf optimizer contract");

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length = strlen(text);
    if (!file) return -1;
    if (fwrite(text, 1, length, file) != length) {
        fclose(file);
        return -1;
    }
    return fclose(file) == 0 ? 0 : -1;
}

static int write_weights(const char* path) {
    const int shape[2] = {2, 2};
    const float values[4] = {0.25f, -0.40f, 0.15f, 0.30f};
    SafetensorsFile file;
    int status;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_F32,
                               shape, 2, values, sizeof(values)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

static int read_weight(const char* path, float values[4]) {
    SafetensorsFile file;
    const SafetensorsTensor* tensor;
    memset(&file, 0, sizeof(file));
    if (safetensors_load(path, &file) != 0) return -1;
    tensor = safetensors_find_tensor(&file, "w");
    if (!tensor || tensor->dtype != SAFETENSORS_DTYPE_F32 ||
        tensor->nbytes != 4u * sizeof(float)) {
        safetensors_free(&file);
        return -1;
    }
    memcpy(values, tensor->data, 4u * sizeof(float));
    safetensors_free(&file);
    return 0;
}

static int weights_differ(const float left[4], const float right[4]) {
    for (int index = 0; index < 4; index++)
        if (fabsf(left[index] - right[index]) > 1.0e-7f) return 1;
    return 0;
}

static VxStatus trainer_bind_and_step_batch(VxTrainer* trainer,
                                      const float* input,
                                      size_t batch_size,
                                      const int32_t* targets,
                                      size_t target_count,
                                      const char* logits_name,
                                      VxOptimizerKind optimizer,
                                      uint32_t accumulation_steps,
                                      int flush,
                                      float normalizer,
                                      VxTrainStepResult* result,
                                      VxReport* report) {
    const char* trainables[] = {"w"};
    VxCrossEntropyLoss loss = VX_CROSS_ENTROPY_LOSS_INIT;
    VxTrainStepOptions options = VX_TRAIN_STEP_OPTIONS_INIT;
    VxTensorBinding binding = VX_TENSOR_BINDING_INIT;
    binding.name = "x";
    binding.dtype = VX_DTYPE_F32;
    binding.rank = 2;
    binding.shape[0] = (int64_t)batch_size;
    binding.shape[1] = 2;
    binding.data = input;
    binding.byte_size = batch_size * 2u * sizeof(float);
    binding.location = VX_MEMORY_HOST;
    loss.logits_name = logits_name;
    loss.targets = targets;
    loss.target_count = target_count;
    loss.normalizer = normalizer;
    options.inputs = &binding;
    options.input_count = 1;
    options.losses = &loss;
    options.loss_count = 1;
    options.trainable_names = trainables;
    options.trainable_count = 1;
    options.optimizer.kind = optimizer;
    options.optimizer.learning_rate = 0.05f;
    options.optimizer.weight_decay = optimizer == VX_OPTIMIZER_ADAMW
        ? 0.01f : 0.0f;
    options.accumulation_steps = accumulation_steps;
    options.flush_accumulation = flush;
    return vx_trainer_train_step(trainer, &options, result, report);
}

static VxStatus trainer_bind_and_step(VxTrainer* trainer,
                                      const float input[2],
                                      int32_t target,
                                      const char* logits_name,
                                      VxOptimizerKind optimizer,
                                      uint32_t accumulation_steps,
                                      int flush,
                                      float normalizer,
                                      VxTrainStepResult* result,
                                      VxReport* report) {
    return trainer_bind_and_step_batch(
        trainer, input, 1u, &target, 1u, logits_name, optimizer,
        accumulation_steps, flush, normalizer, result, report);
}

static int oversized_step_descriptors_are_rejected(
        VxTrainer* trainer, const float input_values[2], VxReport* report) {
    const char* trainables[] = {"w"};
    const int32_t target = 0;
    VxTensorBinding input = VX_TENSOR_BINDING_INIT;
    VxCrossEntropyLoss loss = VX_CROSS_ENTROPY_LOSS_INIT;
    VxTrainStepOptions options = VX_TRAIN_STEP_OPTIONS_INIT;
    VxTrainStepResult result = VX_TRAIN_STEP_RESULT_INIT;
    VxReport oversized_report = VX_REPORT_INIT;

    input.name = "x";
    input.dtype = VX_DTYPE_F32;
    input.rank = 2;
    input.shape[0] = 1;
    input.shape[1] = 2;
    input.data = input_values;
    input.byte_size = 2u * sizeof(float);
    input.location = VX_MEMORY_HOST;
    loss.logits_name = "logits";
    loss.targets = &target;
    loss.target_count = 1;
    options.inputs = &input;
    options.input_count = 1;
    options.losses = &loss;
    options.loss_count = 1;
    options.trainable_names = trainables;
    options.trainable_count = 1;

    options.struct_size++;
    CHECK(vx_trainer_train_step(trainer, &options, &result, report) ==
          VX_STATUS_INVALID_ARGUMENT);
    options.struct_size = sizeof(options);
    result.struct_size++;
    CHECK(vx_trainer_train_step(trainer, &options, &result, report) ==
          VX_STATUS_INVALID_ARGUMENT);
    result.struct_size = sizeof(result);
    options.optimizer.struct_size++;
    CHECK(vx_trainer_train_step(trainer, &options, &result, report) ==
          VX_STATUS_INVALID_ARGUMENT);
    options.optimizer.struct_size = sizeof(options.optimizer);
    loss.struct_size++;
    CHECK(vx_trainer_train_step(trainer, &options, &result, report) ==
          VX_STATUS_INVALID_ARGUMENT);
    loss.struct_size = sizeof(loss);
    input.struct_size++;
    CHECK(vx_trainer_train_step(trainer, &options, &result, report) ==
          VX_STATUS_INVALID_ARGUMENT);
    input.struct_size = sizeof(input);
    oversized_report.struct_size++;
    CHECK(vx_trainer_train_step(
              trainer, &options, &result, &oversized_report) ==
          VX_STATUS_INVALID_ARGUMENT);
    return 0;
}

static int execute_compiled(VxCompiledModel* compiled,
                            const float input[2],
                            float output[2]) {
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    const VxTensorBinding binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 2u, {1, 2},
        input, 2u * sizeof(float), VX_MEMORY_HOST,
    };
    VxStatus status = vx_compiled_model_create_context(
        compiled, &options, &context, &report);
    if (status == VX_STATUS_OK)
        status = vx_execution_context_execute(
            context, &binding, 1u, &result, &report);
    if (status == VX_STATUS_OK)
        status = vx_result_read(result, "logits", output,
                                2u * sizeof(float), NULL, &report);
    vx_result_release(result);
    if (context) (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    return status == VX_STATUS_OK ? 0 : -1;
}

typedef struct ThreadStep {
    VxTrainer* trainer;
    float input[2];
    int32_t target;
    VxTrainStepResult result;
    VxReport report;
    VxStatus status;
} ThreadStep;

static void* thread_step_main(void* argument) {
    ThreadStep* step = (ThreadStep*)argument;
    step->status = trainer_bind_and_step(
        step->trainer, step->input, step->target, "logits",
        VX_OPTIMIZER_ADAMW, 1, 0, 0.0f, &step->result, &step->report);
    return NULL;
}

int main(void) {
    const char* graph_path = "/tmp/volvox-trainer.graph.json";
    const char* initial_weights_path = "/tmp/volvox-trainer-initial.safetensors";
    const char* baseline_export_path = "/tmp/volvox-trainer-baseline.safetensors";
    const char* dirty_export_path = "/tmp/volvox-trainer-dirty.safetensors";
    const char* restored_export_path = "/tmp/volvox-trainer-restored.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\",2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"matmul\",\"opType\":\"MatMul\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"logits\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",2]}},\"params\":{}}],"
        "\"outputs\":[\"logits\"]}";
    const char* initial_paths[] = {initial_weights_path};
    const char* baseline_exports[] = {baseline_export_path};
    const char* dirty_exports[] = {dirty_export_path};
    const char* restored_exports[] = {restored_export_path};
    const char* cpu_backends[] = {"cpu"};
    const float input[2] = {1.0f, -0.5f};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource model_source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxTrainerOptions trainer_options = VX_TRAINER_OPTIONS_INIT;
    VxRevisionInfo initial = VX_REVISION_INFO_INIT;
    VxRevisionInfo after_accumulation = VX_REVISION_INFO_INIT;
    VxRevisionInfo first_published = VX_REVISION_INFO_INIT;
    VxRevisionInfo current = VX_REVISION_INFO_INIT;
    VxRevisionInfo threaded_published = VX_REVISION_INFO_INIT;
    VxTrainStepResult step_result = VX_TRAIN_STEP_RESULT_INIT;
    VxTrainStepResult second_result = VX_TRAIN_STEP_RESULT_INIT;
    VxTrainStepResult failed_result = VX_TRAIN_STEP_RESULT_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* old_compiled = NULL;
    VxCompiledModel* new_compiled = NULL;
    VxTrainer* trainer = NULL;
    VxTrainer* failed = NULL;
    VxTrainer* left = NULL;
    VxTrainer* right = NULL;
    VxTrainer* retained = NULL;
    float initial_weight[4];
    float baseline_weight[4];
    float dirty_weight[4];
    float restored_weight[4];
    float old_output[2];
    float new_output[2];

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(write_weights(initial_weights_path) == 0);
    CHECK(read_weight(initial_weights_path, initial_weight) == 0);
    model_source.graph_path = graph_path;
    model_source.weight_paths = initial_paths;
    model_source.weight_path_count = 1;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = cpu_backends;
    policy.backend_count = 1;
    trainer_options.backend = "cpu";
    trainer_options.rng_seed = UINT64_C(1234);

    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_load_model(runtime, &model_source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(vx_model_revision_info(model, &initial, &report) == VX_STATUS_OK);
    CHECK(initial.weight_id && initial.weight_revision == 1);
    CHECK(vx_model_compile(model, &policy, &old_compiled, &report) ==
          VX_STATUS_OK);

    {
        VxTrainer* rejected = NULL;
        VxTrainerOptions oversized = trainer_options;
        VxReport oversized_report = VX_REPORT_INIT;
        oversized.struct_size++;
        CHECK(vx_model_create_trainer(
                  model, &oversized, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL);
        oversized_report.struct_size++;
        CHECK(vx_model_create_trainer(
                  model, &trainer_options, &rejected, &oversized_report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL);
    }

    {
        VxTrainer* unavailable = NULL;
        VxTrainerOptions exact = VX_TRAINER_OPTIONS_INIT;
        exact.backend = "nnapi";
        CHECK(vx_model_create_trainer(model, &exact, &unavailable, &report) ==
              VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(unavailable == NULL &&
              !strcmp(report.reason, "TRAINING_BACKEND_UNSUPPORTED"));
        exact.backend = "cuda";
        CHECK(vx_model_create_trainer(model, &exact, &unavailable, &report) ==
              VX_STATUS_BACKEND_UNAVAILABLE);
        CHECK(unavailable == NULL && !strcmp(report.backend, "cuda"));
        exact.backend = "";
        CHECK(vx_model_create_trainer(model, &exact, &unavailable, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(unavailable == NULL);
    }

    CHECK(vx_model_create_trainer(model, &trainer_options, &trainer, &report) ==
          VX_STATUS_OK);
    CHECK(!strcmp(report.backend, "cpu"));
    CHECK(vx_trainer_input_count(trainer) == 1);
    {
        VxTensorSpec spec = VX_TENSOR_SPEC_INIT;
        VxRevisionInfo published = VX_REVISION_INFO_INIT;
        VxReport oversized_report = VX_REPORT_INIT;
        spec.struct_size++;
        CHECK(vx_trainer_input_spec(trainer, 0, &spec, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        published.struct_size++;
        CHECK(vx_trainer_commit(trainer, &published, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        oversized_report.struct_size++;
        CHECK(vx_trainer_close(trainer, &oversized_report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(vx_trainer_input_count(trainer) == 1);
        spec = (VxTensorSpec)VX_TENSOR_SPEC_INIT;
        CHECK(vx_trainer_input_spec(trainer, 0, &spec, &report) == VX_STATUS_OK);
        CHECK(spec.name && !strcmp(spec.name, "x") &&
              spec.dtype == VX_DTYPE_F32 && spec.rank == 2 &&
              spec.dimensions[0].kind == VX_DIMENSION_SYMBOLIC &&
              !strcmp(spec.dimensions[0].symbol, "B") &&
              spec.dimensions[0].min == 1 && spec.dimensions[0].max == 2 &&
              spec.dimensions[1].kind == VX_DIMENSION_FIXED &&
              spec.dimensions[1].min == 2);
    }
    CHECK(oversized_step_descriptors_are_rejected(trainer, input, &report) == 0);

    /* An unfinished accumulation window is private and cannot be committed. */
    CHECK(trainer_bind_and_step(trainer, input, 0, "logits",
                                VX_OPTIMIZER_SGD, 2, 0, 2.0f,
                                &step_result, &report) == VX_STATUS_OK);
    CHECK(!step_result.update_applied &&
          step_result.accumulated_microbatches == 1 &&
          step_result.optimizer_step == 0 && isfinite(step_result.loss));
    CHECK(vx_model_revision_info(model, &after_accumulation, &report) ==
          VX_STATUS_OK);
    CHECK(after_accumulation.weight_revision == initial.weight_revision);
    CHECK(vx_trainer_commit(trainer, NULL, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(!strcmp(report.reason, "ACCUMULATION_PENDING"));

    {
        const float other_shape[4] = {1.0f, -0.5f, 0.25f, 0.75f};
        const int32_t other_targets[2] = {0, 1};
        VxTrainStepResult rejected = VX_TRAIN_STEP_RESULT_INIT;
        CHECK(trainer_bind_and_step_batch(
                  trainer, other_shape, 2u, other_targets, 2u, "logits",
                  VX_OPTIMIZER_SGD, 2, 0, 2.0f, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
    }

    CHECK(trainer_bind_and_step(trainer, input, 0, "logits",
                                VX_OPTIMIZER_SGD, 2, 0, 2.0f,
                                &second_result, &report) == VX_STATUS_OK);
    CHECK(second_result.update_applied && second_result.optimizer_step == 1 &&
          second_result.metric_count == 1 &&
          !strcmp(second_result.metrics[0].name, "loss"));
    CHECK(vx_model_revision_info(model, &current, &report) == VX_STATUS_OK);
    CHECK(current.weight_revision == initial.weight_revision);
    CHECK(vx_trainer_commit(trainer, &first_published, &report) == VX_STATUS_OK);
    CHECK(first_published.weight_id == initial.weight_id &&
          first_published.weight_revision == initial.weight_revision + 1u);

    /* Compiled models pin their exact old/new immutable revisions. */
    CHECK(vx_model_compile(model, &policy, &new_compiled, &report) ==
          VX_STATUS_OK);
    CHECK(execute_compiled(old_compiled, input, old_output) == 0);
    CHECK(execute_compiled(new_compiled, input, new_output) == 0);
    CHECK(fabsf(old_output[0] - new_output[0]) > 1.0e-7f ||
          fabsf(old_output[1] - new_output[1]) > 1.0e-7f);

    /* Rollback restores committed weights and optimizer baseline. */
    CHECK(vx_trainer_export_weights(trainer, baseline_exports, 1, &report) ==
          VX_STATUS_OK);
    CHECK(read_weight(baseline_export_path, baseline_weight) == 0);
    CHECK(weights_differ(initial_weight, baseline_weight));
    step_result = (VxTrainStepResult)VX_TRAIN_STEP_RESULT_INIT;
    CHECK(trainer_bind_and_step(trainer, input, 1, "logits",
                                VX_OPTIMIZER_ADAMW, 1, 0, 0.0f,
                                &step_result, &report) == VX_STATUS_OK);
    CHECK(step_result.update_applied && step_result.optimizer_step == 2);
    CHECK(vx_trainer_export_weights(trainer, dirty_exports, 1, &report) ==
          VX_STATUS_OK);
    CHECK(read_weight(dirty_export_path, dirty_weight) == 0);
    CHECK(weights_differ(baseline_weight, dirty_weight));
    CHECK(vx_trainer_rollback(trainer, &report) == VX_STATUS_OK);
    CHECK(vx_trainer_export_weights(trainer, restored_exports, 1, &report) ==
          VX_STATUS_OK);
    CHECK(read_weight(restored_export_path, restored_weight) == 0);
    CHECK(!weights_differ(baseline_weight, restored_weight));
    CHECK(vx_trainer_commit(trainer, NULL, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(!strcmp(report.reason, "NO_APPLIED_UPDATE"));

    /* A failed step restores/discards private work and publishes nothing. */
    CHECK(vx_model_create_trainer(model, NULL, &failed, &report) == VX_STATUS_OK);
    CHECK(trainer_bind_and_step(failed, input, 1, "logits",
                                VX_OPTIMIZER_ADAMW, 1, 0, 0.0f,
                                &failed_result, &report) == VX_STATUS_OK);
    CHECK(failed_result.update_applied);
    failed_result = (VxTrainStepResult)VX_TRAIN_STEP_RESULT_INIT;
    CHECK(trainer_bind_and_step(failed, input, 0, "missing_logits",
                                VX_OPTIMIZER_SGD, 1, 0, 0.0f,
                                &failed_result, &report) ==
          VX_STATUS_EXECUTION_FAILED);
    CHECK(vx_model_revision_info(model, &current, &report) == VX_STATUS_OK);
    CHECK(current.weight_revision == first_published.weight_revision);
    CHECK(vx_trainer_commit(failed, NULL, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(!strcmp(report.reason, "NO_APPLIED_UPDATE"));
    CHECK(vx_trainer_export_weights(failed, restored_exports, 1, &report) ==
          VX_STATUS_OK);
    CHECK(read_weight(restored_export_path, restored_weight) == 0);
    CHECK(!weights_differ(baseline_weight, restored_weight));

    /* Two trainers execute simultaneously in separate engine/optimizer/input
     * owners. Exactly one compare-and-publish can win. */
    CHECK(vx_model_create_trainer(model, NULL, &left, &report) == VX_STATUS_OK);
    CHECK(vx_model_create_trainer(model, NULL, &right, &report) == VX_STATUS_OK);
    {
        pthread_t left_thread;
        pthread_t right_thread;
        ThreadStep left_step = {
            .trainer = left,
            .input = {1.0f, 0.25f},
            .target = 0,
            .result = VX_TRAIN_STEP_RESULT_INIT,
            .report = VX_REPORT_INIT,
            .status = VX_STATUS_INTERNAL,
        };
        ThreadStep right_step = {
            .trainer = right,
            .input = {-0.75f, 1.0f},
            .target = 1,
            .result = VX_TRAIN_STEP_RESULT_INIT,
            .report = VX_REPORT_INIT,
            .status = VX_STATUS_INTERNAL,
        };
        CHECK(pthread_create(&left_thread, NULL, thread_step_main, &left_step) == 0);
        CHECK(pthread_create(&right_thread, NULL, thread_step_main, &right_step) == 0);
        CHECK(pthread_join(left_thread, NULL) == 0);
        CHECK(pthread_join(right_thread, NULL) == 0);
        CHECK(left_step.status == VX_STATUS_OK &&
              right_step.status == VX_STATUS_OK);
        CHECK(left_step.result.update_applied && right_step.result.update_applied);
        CHECK(vx_trainer_commit(left, &threaded_published, &report) ==
              VX_STATUS_OK);
        CHECK(vx_trainer_commit(right, NULL, &report) ==
              VX_STATUS_REVISION_CONFLICT);
        CHECK(!strcmp(report.reason, "REVISION_CONFLICT"));
        CHECK(vx_trainer_rollback(right, &report) == VX_STATUS_OK);
    }
    CHECK(vx_model_revision_info(model, &current, &report) == VX_STATUS_OK);
    CHECK(current.weight_revision == threaded_published.weight_revision);

    /* Logical Runtime close rejects new root work but retained Trainers keep
     * their private engine, may recreate it for rollback, and may publish a
     * later already-accepted update. */
    step_result = (VxTrainStepResult)VX_TRAIN_STEP_RESULT_INIT;
    CHECK(vx_model_create_trainer(model, NULL, &retained, &report) == VX_STATUS_OK);
    CHECK(trainer_bind_and_step(retained, input, 0, "logits",
                                VX_OPTIMIZER_SGD, 1, 0, 0.0f,
                                &step_result, &report) == VX_STATUS_OK);
    CHECK(step_result.update_applied);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    CHECK(vx_trainer_rollback(retained, &report) == VX_STATUS_OK);
    CHECK(vx_trainer_commit(retained, NULL, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(!strcmp(report.reason, "NO_APPLIED_UPDATE"));
    step_result = (VxTrainStepResult)VX_TRAIN_STEP_RESULT_INIT;
    CHECK(trainer_bind_and_step(retained, input, 0, "logits",
                                VX_OPTIMIZER_SGD, 1, 0, 0.0f,
                                &step_result, &report) == VX_STATUS_OK);
    CHECK(step_result.update_applied);
    CHECK(vx_trainer_commit(retained, &current, &report) == VX_STATUS_OK);
    CHECK(current.weight_revision == threaded_published.weight_revision + 1u);
    {
        VxTrainer* rejected = NULL;
        CHECK(vx_model_create_trainer(model, NULL, &rejected, &report) ==
              VX_STATUS_HANDLE_DISPOSED);
        CHECK(rejected == NULL && report.stage == VX_STAGE_TRAINER_CREATE);
    }

    CHECK(vx_trainer_close(trainer, &report) == VX_STATUS_OK);
    CHECK(vx_trainer_close(trainer, &report) == VX_STATUS_OK);
    {
        VxTensorSpec closed_spec = VX_TENSOR_SPEC_INIT;
        CHECK(vx_trainer_input_spec(trainer, 0, &closed_spec, &report) ==
              VX_STATUS_HANDLE_DISPOSED);
    }

    vx_trainer_release(retained);
    vx_trainer_release(right);
    vx_trainer_release(left);
    vx_trainer_release(failed);
    vx_trainer_release(trainer);
    vx_compiled_model_release(new_compiled);
    vx_compiled_model_release(old_compiled);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);

    remove(graph_path);
    remove(initial_weights_path);
    remove(baseline_export_path);
    remove(dirty_export_path);
    remove(restored_export_path);
    puts("native Trainer ownership, training, publication, rollback, and isolation tests passed");
    return 0;
}
