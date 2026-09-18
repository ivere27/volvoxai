/* Full-profile authoring lifecycle — internal implementation surface.
 *
 * Like vx_lifecycle.h this is not the public API. VxTrainingService and
 * VxQuantizationService in proto/volvoxai.proto are, and the handler tables
 * in native/src/api/ reach these functions on their behalf.
 */
#ifndef VOLVOXAI_TRAINING_LIFECYCLE_H
#define VOLVOXAI_TRAINING_LIFECYCLE_H

#include "vx_lifecycle.h"
#include "native_tensor.h"
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
struct VolvoxaiV1TrainerCheckpoint;
struct VolvoxaiV1TrainerState;
struct VolvoxaiV1TrainerShapeOptions;
VxStatus vx_trainer_restore_checkpoint(VxTrainer*, const struct VolvoxaiV1TrainerCheckpoint*, VxReport*);
VxStatus vx_trainer_export_checkpoint(VxTrainer*, struct VolvoxaiV1TrainerCheckpoint*,
    const void* metadata, size_t metadata_bytes, int replace_metadata, VxReport*);
VxStatus vx_trainer_state(VxTrainer*, struct VolvoxaiV1TrainerState*, int reset, VxReport*);

#define VX_MAX_TRAINING_LOSSES 8u
#define VX_TRAINING_NAME_CAPACITY 128u

typedef struct VxTrainerOptions {
    size_t struct_size;
    /* NULL selects CPU. A non-NULL backend is an exact requirement. The full
     * profile accepts differentiable built-ins only: cpu, vulkan, opengl,
     * metal, or cuda. It never retries another backend after creation. */
    const char* backend;
    uint64_t rng_seed;
    const struct VolvoxaiV1TrainerShapeOptions* shape_options;
} VxTrainerOptions;

#define VX_TRAINER_OPTIONS_INIT \
    { sizeof(VxTrainerOptions), NULL, UINT64_C(0), NULL }

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

/* Presence bits for the generated per-call optimizer overrides. */
enum {
    VX_OPTIMIZER_FIELD_KIND = 1u,
    VX_OPTIMIZER_FIELD_LEARNING_RATE = 2u,
    VX_OPTIMIZER_FIELD_BETA1 = 4u,
    VX_OPTIMIZER_FIELD_BETA2 = 8u,
    VX_OPTIMIZER_FIELD_EPSILON = 16u,
    VX_OPTIMIZER_FIELD_WEIGHT_DECAY = 32u,
    VX_OPTIMIZER_FIELD_MAX_GRADIENT_NORM = 64u,
    VX_OPTIMIZER_FIELDS_ALL = 127u
};

typedef struct VxTrainerTensor VxTrainerTensor;
typedef struct VxTrainStepOptions {
    size_t struct_size;
    /* One complete logical input batch. Every binding is validated and copied
     * atomically before the private forward/backward step may begin. */
    const VxTensorBinding* inputs;
    size_t input_count;
    const VxCrossEntropyLoss* losses;
    size_t loss_count;
    const char* const* trainable_names;
    size_t trainable_count;
    VxOptimizerOptions optimizer;
    uint32_t accumulation_steps;
    int32_t flush_accumulation;
    int32_t reset_accumulation;
    uint32_t optimizer_fields;
    const char* const* output_names;
    size_t output_count;
    VxTrainerTensor* outputs;
} VxTrainStepOptions;

#define VX_TRAIN_STEP_OPTIONS_INIT \
    { sizeof(VxTrainStepOptions), NULL, 0, NULL, 0, NULL, 0, \
      VX_OPTIMIZER_OPTIONS_INIT, 1u, 0, 0, VX_OPTIMIZER_FIELDS_ALL, NULL, 0, NULL }

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
    VxResultState state;
} VxTrainStepResult;

#define VX_TRAIN_STEP_RESULT_INIT \
    { sizeof(VxTrainStepResult), 0, 0, 0, 0, 0.0f, 0, \
      { { { 0 }, 0.0f, 0, 0, 0.0f } }, { 0 }, VX_RESULT_STATE_UNSPECIFIED }

VxStatus vx_trainer_get_step(VxTrainer* trainer, uint64_t microbatch_id,
                              VxTrainStepResult* result, VxReport* report);

#if VOLVOXAI_ENABLE_WEBGPU
typedef struct VxWebGpuTrainStep VxWebGpuTrainStep;
VxStatus vx_webgpu_train_begin(const VxTrainStepOptions* options, long step,
                                 VxWebGpuTrainStep** pending, char* evidence, size_t capacity);
VxStatus vx_webgpu_train_poll(VxWebGpuTrainStep* pending, VxTrainStepResult* result);
void vx_webgpu_train_release(VxWebGpuTrainStep* pending);
#endif

struct VxTrainerTensor {
    VxTensorInfo info;
    VxNativeBuffer memory;
    VxNativeStorage* storage;
    void* owner;
    void (*release)(void*);
};
VxStatus vx_trainer_read_parameters(VxTrainer* trainer, const char* const* names,
    size_t count, int shared, VxTrainerTensor* outputs, VxReport* report);

VX_API VxStatus vx_model_create_trainer(VxModel* model,
                                         const VxTrainerOptions* options,
                                         VxTrainer** out_trainer,
                                         VxReport* report);
VX_API void vx_trainer_retain(VxTrainer* trainer);
VX_API void vx_trainer_release(VxTrainer* trainer);
VX_API VxStatus vx_trainer_close(VxTrainer* trainer, VxReport* report);

VX_API size_t vx_trainer_input_count(VxTrainer* trainer);
VX_API VxStatus vx_trainer_input_spec(VxTrainer* trainer,
                                      size_t index,
                                      VxTensorSpec* spec,
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

/* The sink borrows one serialized shard only for the duration of its call. */
typedef int (*VxTrainerWeightSink)(void* user, size_t index, size_t count,
                                    const unsigned char* bytes, size_t size);
VxStatus vx_trainer_export_weight_bytes(VxTrainer* trainer,
                                         VxTrainerWeightSink sink, void* user,
                                         VxReport* report);

typedef struct VxPTQObserverSpec {
    size_t struct_size;
    const char* tensor_name;
    VxDataType dtype;
    VxPTQScheme scheme;
    /* The byte tensor `tensor_name` becomes in the template, and so the one
     * whose affine the writer fills in. NULL means the template did not
     * rename the value and the writer names the affine itself. */
    const char* quantized_tensor_name;
} VxPTQObserverSpec;

#define VX_PTQ_OBSERVER_SPEC_INIT                                      \
    {                                                                  \
        sizeof(VxPTQObserverSpec), NULL, VX_DTYPE_I8,                  \
            VX_PTQ_SCHEME_SYMMETRIC, NULL                              \
    }

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
    /* The node's id, which is what binds it to the template. Authoring
     * inserts the affine boundaries, so positions no longer agree between
     * the source graph and the template; ids do. */
    const char* node_id;
} VxPTQLayerSpec;

#define VX_PTQ_LAYER_SPEC_INIT \
    { sizeof(VxPTQLayerSpec), VX_PTQ_MODE_W8A8, VX_PTQ_LAYER_QLINEAR, \
      -1, 0, NULL, NULL, NULL, NULL, NULL, NULL, NULL }

typedef struct VxPTQPlanOptions {
    size_t struct_size;
    const char* template_graph_path;
    /* Every declared profile must receive at least one successful calibration
     * batch before package materialization is allowed. */
    const char* const* profile_names;
    size_t profile_count;
    const VxPTQObserverSpec* observers;
    size_t observer_count;
    const VxPTQLayerSpec* layers;
    size_t layer_count;
    const unsigned char* template_graph;
    size_t template_graph_size;
} VxPTQPlanOptions;

#define VX_PTQ_PLAN_OPTIONS_INIT \
    { sizeof(VxPTQPlanOptions), NULL, NULL, 0, NULL, 0, NULL, 0, NULL, 0 }

typedef struct VxPTQCalibrationBatch {
    size_t struct_size;
    const char* profile_name;
    const char* sample_name;
    /* Number of logical examples represented by this batch. */
    uint64_t sample_count;
    const VxTensorBinding* inputs;
    size_t input_count;
} VxPTQCalibrationBatch;

#define VX_PTQ_CALIBRATION_BATCH_INIT \
    { sizeof(VxPTQCalibrationBatch), NULL, NULL, 0, NULL, 0 }

typedef struct VxPTQPlanInfo {
    size_t struct_size;
    uint64_t calibration_batches;
    uint64_t calibration_samples;
    size_t tensor_count;
    size_t profile_count;
    size_t covered_profile_count;
    int32_t coverage_complete;
    VxRevisionInfo revision;
} VxPTQPlanInfo;

#define VX_PTQ_PLAN_INFO_INIT \
    { sizeof(VxPTQPlanInfo), 0, 0, 0, 0, 0, 0, VX_REVISION_INFO_INIT }

typedef struct VxPTQProfileCoverage {
    size_t struct_size;
    char profile_name[VX_PTQ_NAME_CAPACITY];
    uint64_t calibration_batches;
    uint64_t calibration_samples;
    size_t shape_signature_count;
    size_t symbol_count;
} VxPTQProfileCoverage;

#define VX_PTQ_PROFILE_COVERAGE_INIT \
    { sizeof(VxPTQProfileCoverage), { 0 }, 0, 0, 0, 0 }

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
    unsigned char** graph_bytes;
    size_t* graph_size;
    unsigned char** weights_bytes;
    size_t* weights_size;
} VxPTQPackageOptions;

#define VX_PTQ_PACKAGE_OPTIONS_INIT \
    { sizeof(VxPTQPackageOptions), NULL, NULL, NULL, NULL, NULL, NULL }

VX_API VxStatus vx_model_create_ptq_plan(VxModel* model,
                                         const VxPTQPlanOptions* options,
                                         VxPTQPlan** out_plan,
                                         VxReport* report);
VX_API void vx_ptq_plan_retain(VxPTQPlan* plan);
VX_API void vx_ptq_plan_release(VxPTQPlan* plan);
VX_API VxStatus vx_ptq_plan_close(VxPTQPlan* plan, VxReport* report);
VX_API size_t vx_ptq_plan_input_count(VxPTQPlan* plan);
VX_API VxStatus vx_ptq_plan_input_spec(VxPTQPlan* plan,
                                       size_t index,
                                       VxTensorSpec* spec,
                                       VxReport* report);
VX_API VxStatus vx_ptq_plan_calibrate(VxPTQPlan* plan,
                                      const VxPTQCalibrationBatch* batch,
                                      VxPTQPlanInfo* info,
                                      VxReport* report);
VX_API VxStatus vx_ptq_plan_info(VxPTQPlan* plan,
                                 VxPTQPlanInfo* info,
                                 VxReport* report);
VX_API VxStatus vx_ptq_plan_profile_coverage(
    VxPTQPlan* plan,
    size_t index,
    VxPTQProfileCoverage* coverage,
    VxReport* report);
/* UTF-8 JSON in the shared `volvox.ptq-coverage/v1` format. The first call
 * may pass NULL/zero to query the required byte count including the NUL. */
VX_API VxStatus vx_ptq_plan_coverage_json(VxPTQPlan* plan,
                                          char* output,
                                          size_t output_capacity,
                                          size_t* required_size,
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

#endif /* VOLVOXAI_TRAINING_LIFECYCLE_H */
