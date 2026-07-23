#ifndef VOLVOXAI_FULL_H
#define VOLVOXAI_FULL_H

#include "volvoxai.h"
#include "volvoxai_full_enums.h"

#ifdef __cplusplus
extern "C" {
#endif

/* This header and every symbol declared by it exist only in the full native
 * profile. Training state is owned by one opaque Trainer; PTQ calibration is
 * owned by one opaque PTQ plan. Neither is attached to an inference
 * ExecutionContext. */
typedef struct VxTrainer VxTrainer;
typedef struct VxPTQPlan VxPTQPlan;

#define VX_MAX_TRAINING_LOSSES 8u
#define VX_TRAINING_NAME_CAPACITY 128u

typedef struct VxTrainerOptions {
    size_t struct_size;
    /* NULL selects CPU. A non-NULL backend is an exact requirement. The full
     * profile accepts differentiable built-ins only: cpu, vulkan, opengl,
     * metal, or cuda. It never retries another backend after creation. */
    const char* backend;
    uint64_t rng_seed;
} VxTrainerOptions;

#define VX_TRAINER_OPTIONS_INIT \
    { sizeof(VxTrainerOptions), NULL, UINT64_C(0) }

typedef struct VxCrossEntropyLoss {
    size_t struct_size;
    const char* name;
    const char* logits_name;
    const int32_t* targets;
    size_t target_count;
    int32_t ignore_index;
    /* -1 consumes the complete logits tensor. A non-negative value selects a
     * row-aware loss when the graph supports that execution shape. */
    int32_t row_index;
    float weight;
    /* Zero selects the active-label count for a one-microbatch update.
     * Accumulation windows larger than one require an explicit positive
     * denominator. */
    float normalizer;
} VxCrossEntropyLoss;

#define VX_CROSS_ENTROPY_LOSS_INIT \
    { sizeof(VxCrossEntropyLoss), "loss", NULL, NULL, 0, -1, -1, 1.0f, 0.0f }

typedef struct VxOptimizerOptions {
    size_t struct_size;
    VxOptimizerKind kind;
    float learning_rate;
    float beta1;
    float beta2;
    float epsilon;
    float weight_decay;
    float max_gradient_norm;
} VxOptimizerOptions;

#define VX_OPTIMIZER_OPTIONS_INIT \
    { sizeof(VxOptimizerOptions), VX_OPTIMIZER_ADAMW, 1.0e-3f, 0.9f, 0.999f, \
      1.0e-8f, 0.0f, 0.0f }

typedef struct VxTrainStepOptions {
    size_t struct_size;
    const VxCrossEntropyLoss* losses;
    size_t loss_count;
    const char* const* trainable_names;
    size_t trainable_count;
    VxOptimizerOptions optimizer;
    uint32_t accumulation_steps;
    int32_t flush_accumulation;
    int32_t reset_accumulation;
} VxTrainStepOptions;

#define VX_TRAIN_STEP_OPTIONS_INIT \
    { sizeof(VxTrainStepOptions), NULL, 0, NULL, 0, \
      VX_OPTIMIZER_OPTIONS_INIT, 1u, 0, 0 }

typedef struct VxTrainingMetric {
    char name[VX_TRAINING_NAME_CAPACITY];
    float loss;
    int32_t correct;
    int32_t examples;
    float normalizer;
} VxTrainingMetric;

typedef struct VxTrainStepResult {
    size_t struct_size;
    uint64_t microbatch_id;
    uint64_t optimizer_step;
    uint32_t accumulated_microbatches;
    int32_t update_applied;
    float loss;
    size_t metric_count;
    VxTrainingMetric metrics[VX_MAX_TRAINING_LOSSES];
    char backend[VX_REPORT_BACKEND_CAPACITY];
} VxTrainStepResult;

#define VX_TRAIN_STEP_RESULT_INIT \
    { sizeof(VxTrainStepResult), 0, 0, 0, 0, 0.0f, 0, \
      { { { 0 }, 0.0f, 0, 0, 0.0f } }, { 0 } }

VX_API VxStatus vx_model_create_trainer(VxModel* model,
                                         const VxTrainerOptions* options,
                                         VxTrainer** out_trainer,
                                         VxReport* report);
VX_API void vx_trainer_retain(VxTrainer* trainer);
VX_API void vx_trainer_release(VxTrainer* trainer);
VX_API VxStatus vx_trainer_close(VxTrainer* trainer, VxReport* report);

VX_API size_t vx_trainer_input_count(VxTrainer* trainer);
VX_API VxStatus vx_trainer_input_info(VxTrainer* trainer,
                                      size_t index,
                                      VxTensorInfo* info,
                                      VxReport* report);
VX_API VxStatus vx_trainer_set_input(VxTrainer* trainer,
                                     const char* name,
                                     VxDataType dtype,
                                     const void* data,
                                     size_t byte_size,
                                     VxReport* report);

/* A step mutates only Trainer-owned weights, gradients, optimizer slots, RNG,
 * and accumulation state. update_applied is false for an unfinished
 * accumulation window, and Model remains unchanged in either case. */
VX_API VxStatus vx_trainer_train_step(VxTrainer* trainer,
                                      const VxTrainStepOptions* options,
                                      VxTrainStepResult* result,
                                      VxReport* report);

/* Commit rejects unfinished accumulation and publishes exactly one immutable
 * successor only when at least one optimizer update has been applied. It uses
 * compare-and-publish against the Trainer's retained base revision. */
VX_API VxStatus vx_trainer_commit(VxTrainer* trainer,
                                  VxRevisionInfo* published,
                                  VxReport* report);

/* Rollback discards all uncommitted weights, gradients, optimizer changes,
 * accumulation, and RNG progress, then restores the last committed Trainer
 * baseline. */
VX_API VxStatus vx_trainer_rollback(VxTrainer* trainer, VxReport* report);

/* Export the Trainer's current private weight files without publishing them.
 * output_path_count must exactly match the model package's weight shard count.
 * Export rejects an unfinished accumulation window. */
VX_API VxStatus vx_trainer_export_weights(VxTrainer* trainer,
                                          const char* const* output_paths,
                                          size_t output_path_count,
                                          VxReport* report);

/* Full-profile post-training quantization authoring. A plan retains its Model,
 * pins the exact current revision, snapshots the caller-authored graph
 * template at creation, and owns a private CPU engine used only for
 * calibration. The current writer deliberately supports W8A8 QLinear and
 * QConv2D packages backed by exactly one source safetensors shard. */
#define VX_PTQ_NAME_CAPACITY 128u

typedef struct VxPTQObserverSpec {
    size_t struct_size;
    const char* tensor_name;
    VxDataType dtype;
    VxPTQScheme scheme;
} VxPTQObserverSpec;

#define VX_PTQ_OBSERVER_SPEC_INIT \
    { sizeof(VxPTQObserverSpec), NULL, VX_DTYPE_I8, VX_PTQ_SCHEME_SYMMETRIC }

typedef struct VxPTQLayerSpec {
    size_t struct_size;
    VxPTQMode mode;
    VxPTQLayerKind kind;
    int32_t node_index;
    int32_t weight_axis;
    const char* input_tensor_name;
    const char* output_tensor_name;
    const char* source_weight_name;
    const char* packed_weight_name;
    const char* source_bias_name;
    const char* packed_bias_name;
} VxPTQLayerSpec;

#define VX_PTQ_LAYER_SPEC_INIT \
    { sizeof(VxPTQLayerSpec), VX_PTQ_MODE_W8A8, VX_PTQ_LAYER_QLINEAR, \
      -1, 0, NULL, NULL, NULL, NULL, NULL, NULL }

typedef struct VxPTQPlanOptions {
    size_t struct_size;
    const char* template_graph_path;
    const VxPTQObserverSpec* observers;
    size_t observer_count;
    const VxPTQLayerSpec* layers;
    size_t layer_count;
} VxPTQPlanOptions;

#define VX_PTQ_PLAN_OPTIONS_INIT \
    { sizeof(VxPTQPlanOptions), NULL, NULL, 0, NULL, 0 }

typedef struct VxPTQInput {
    size_t struct_size;
    const char* name;
    VxDataType dtype;
    const void* data;
    size_t byte_size;
} VxPTQInput;

#define VX_PTQ_INPUT_INIT \
    { sizeof(VxPTQInput), NULL, VX_DTYPE_F32, NULL, 0 }

typedef struct VxPTQPlanInfo {
    size_t struct_size;
    uint64_t calibration_samples;
    size_t tensor_count;
    VxRevisionInfo revision;
} VxPTQPlanInfo;

#define VX_PTQ_PLAN_INFO_INIT \
    { sizeof(VxPTQPlanInfo), 0, 0, VX_REVISION_INFO_INIT }

typedef struct VxPTQTensorParameters {
    size_t struct_size;
    char tensor_name[VX_PTQ_NAME_CAPACITY];
    VxDataType dtype;
    VxPTQScheme scheme;
    float scale;
    int32_t zero_point;
    float observed_min;
    float observed_max;
    uint64_t observed_values;
} VxPTQTensorParameters;

#define VX_PTQ_TENSOR_PARAMETERS_INIT \
    { sizeof(VxPTQTensorParameters), { 0 }, VX_DTYPE_I8, \
      VX_PTQ_SCHEME_SYMMETRIC, 0.0f, 0, 0.0f, 0.0f, 0 }

typedef struct VxPTQPackageOptions {
    size_t struct_size;
    /* The graph basename must be exactly graph.json. The safetensors file must
     * be a sibling path with a .safetensors suffix. Neither output may exist. */
    const char* output_graph_path;
    const char* output_weights_path;
} VxPTQPackageOptions;

#define VX_PTQ_PACKAGE_OPTIONS_INIT \
    { sizeof(VxPTQPackageOptions), NULL, NULL }

VX_API VxStatus vx_model_create_ptq_plan(VxModel* model,
                                         const VxPTQPlanOptions* options,
                                         VxPTQPlan** out_plan,
                                         VxReport* report);
VX_API void vx_ptq_plan_retain(VxPTQPlan* plan);
VX_API void vx_ptq_plan_release(VxPTQPlan* plan);
VX_API VxStatus vx_ptq_plan_close(VxPTQPlan* plan, VxReport* report);
VX_API size_t vx_ptq_plan_input_count(VxPTQPlan* plan);
VX_API VxStatus vx_ptq_plan_input_info(VxPTQPlan* plan,
                                       size_t index,
                                       VxTensorInfo* info,
                                       VxReport* report);
VX_API VxStatus vx_ptq_plan_calibrate(VxPTQPlan* plan,
                                      const char* sample_name,
                                      const VxPTQInput* inputs,
                                      size_t input_count,
                                      uint64_t* calibration_samples,
                                      VxReport* report);
VX_API VxStatus vx_ptq_plan_info(VxPTQPlan* plan,
                                 VxPTQPlanInfo* info,
                                 VxReport* report);
VX_API VxStatus vx_ptq_plan_tensor_parameters(
    VxPTQPlan* plan,
    size_t index,
    VxPTQTensorParameters* parameters,
    VxReport* report);
VX_API VxStatus vx_ptq_plan_write_package(
    VxPTQPlan* plan,
    const VxPTQPackageOptions* options,
    VxReport* report);

#ifdef __cplusplus
}
#endif

#endif /* VOLVOXAI_FULL_H */
