#ifndef VOLVOXAI_TRAINING_H
#define VOLVOXAI_TRAINING_H

#include "volvoxai.h"
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct volvoxai_cross_entropy_loss {
    const char* name;
    const char* logits_name;
    const int* targets;
    int target_count;
    int ignore_index;
    int row_index;
    float weight;
    /* 0 selects this microbatch's active-label count; positive values are an
       explicit (normally full accumulation-window) denominator. */
    float normalizer;
} volvoxai_cross_entropy_loss_t;

typedef struct volvoxai_cross_entropy_metric {
    float loss;
    int correct;
    int examples;
    float normalizer;
} volvoxai_cross_entropy_metric_t;

typedef enum {
    VOLVOXAI_TENSOR_UPDATE_ASSIGN = 0,
    VOLVOXAI_TENSOR_UPDATE_ADD = 1,
    VOLVOXAI_TENSOR_UPDATE_SGD = 2,
    VOLVOXAI_TENSOR_UPDATE_ADAMW = 3
} VolvoxAITensorUpdateMode;

typedef enum {
    VOLVOXAI_TRAINING_BACKEND_ALLOW_FALLBACK = 0,
    VOLVOXAI_TRAINING_BACKEND_VULKAN = 1,
    VOLVOXAI_TRAINING_BACKEND_OPENGL = 2,
    VOLVOXAI_TRAINING_BACKEND_METAL = 3
} VolvoxAITrainingBackend;

typedef enum {
    VOLVOXAI_PTQ_SYMMETRIC = 0,
    VOLVOXAI_PTQ_ASYMMETRIC = 1
} VolvoxAIPTQScheme;

/* Stable, freestanding PTQ ABI implemented by the full native/WASM profile.
   Capability bits let small clients use only the helpers they require without
   assuming that future sidecars expose the complete authoring surface. */
#define VOLVOXAI_PTQ_ABI_VERSION 1u

typedef enum {
    VOLVOXAI_PTQ_CAP_OBSERVER = 1u << 0,
    VOLVOXAI_PTQ_CAP_PARAMETERS = 1u << 1,
    VOLVOXAI_PTQ_CAP_QUANTIZE = 1u << 2,
    VOLVOXAI_PTQ_CAP_WEIGHT_I8 = 1u << 3,
    VOLVOXAI_PTQ_CAP_BIAS_I32 = 1u << 4,
    VOLVOXAI_PTQ_CAP_ALL = (1u << 5) - 1u
} VolvoxAIPTQCapability;

typedef struct volvoxai_ptq_observer {
    float minimum;
    float maximum;
    uint64_t sample_count;
} volvoxai_ptq_observer_t;

typedef struct volvoxai_ptq_params {
    int32_t dtype;
    int32_t scheme;
    float scale;
    int32_t zero_point;
    float observed_min;
    float observed_max;
    uint64_t sample_count;
} volvoxai_ptq_params_t;

/* Full-profile post-training quantization. Observers are reusable across
   ordinary calibration forwards. All functions reject non-finite source data
   without partially updating their destination. */
uint32_t volvoxai_ptq_abi_version(void);
uint32_t volvoxai_ptq_capabilities(void);
void volvoxai_ptq_observer_reset(volvoxai_ptq_observer_t* observer);
int volvoxai_ptq_observer_observe_f32(volvoxai_ptq_observer_t* observer,
                                      const float* values, int64_t count);
int volvoxai_ptq_calculate_params(const volvoxai_ptq_observer_t* observer,
                                  int32_t dtype, int32_t scheme,
                                  volvoxai_ptq_params_t* out_params);
/* Quantize into the dtype's full physical storage domain: [-128,127] for I8
   and [0,255] for U8. Symmetric parameters derived above use a divisor of 127,
   so values inside their observed range normally occupy only [-127,127]. */
int volvoxai_ptq_quantize_f32(const float* values, int64_t count,
                              const volvoxai_ptq_params_t* params,
                              void* output, uint64_t* saturation_count);

/* Pack row-major weights in the symmetric narrow I8 range [-127,127], with one
   scale per `axis` element and zero point 0. Canonical QLinear [O,I] and QConv
   OHWI weights use axis 0. `output` must not overlap `values`, and `scales`
   must overlap neither buffer; scale_count must equal shape[axis]. */
int volvoxai_ptq_pack_weight_i8(const float* values, const int32_t* shape,
                                int32_t ndim, int32_t axis, int8_t* output,
                                float* scales, int32_t scale_count,
                                uint64_t* saturation_count);
int volvoxai_ptq_pack_bias_i32(const float* values, int32_t count,
                               float input_scale, const float* weight_scales,
                               int32_t scale_count, int32_t* output);

/* Observe a named CPU-visible F32 tensor after volvoxai_engine_forward(). */
int volvoxai_engine_ptq_observe_tensor(const char* tensor_name,
                                       volvoxai_ptq_observer_t* observer);

/* Quantize one loaded F32 model weight into newly named I8 + F32 scale tensors.
   This deliberately does not rewrite graph operators: patch the consuming node
   to QLinear/QConv and copy the returned scale tensor values into the saved
   config's weights_quantization descriptor. The new tensors are added to weight
   file 0, which can then be persisted with
   volvoxai_engine_save_weight_file(0, path). */
int volvoxai_engine_ptq_materialize_weight_i8(const char* source_name,
                                              const char* output_name,
                                              const char* scale_name,
                                              int32_t axis,
                                              int32_t* out_scale_count);

/* Explicit full-profile PTQ package authoring. A plan is bound to the exact
   loaded-model generation that created it and becomes permanently stale after
   reload, shutdown, graph/weight/training mutation, or adapter activation.
   Active or merged adapters are rejected. The plan observes tensors from a
   loaded FP32 graph, but never rewrites that graph. Package creation consumes
   a caller-authored quantized blueprint template and validates every selected
   QLinear/QConv2D node before filling its quantization metadata and adding
   packed tensors to a pass-through copy of the source safetensors file. */
typedef struct VolvoxAIPTQPlan VolvoxAIPTQPlan;

typedef enum {
    VOLVOXAI_PTQ_LAYER_QLINEAR = 1,
    VOLVOXAI_PTQ_LAYER_QCONV2D = 2
} VolvoxAIPTQLayerKind;

typedef struct volvoxai_ptq_tensor_spec {
    uint32_t struct_size;
    const char* tensor_name;
    int32_t dtype;
    int32_t scheme;
} volvoxai_ptq_tensor_spec_t;

#define VOLVOXAI_PTQ_TENSOR_SPEC_INIT \
    { sizeof(volvoxai_ptq_tensor_spec_t), NULL, VOLVOXAI_DTYPE_I8, \
      VOLVOXAI_PTQ_SYMMETRIC }

typedef struct volvoxai_ptq_layer_spec {
    uint32_t struct_size;
    int32_t kind;
    /* The same index must identify the canonical FP32 source node in the
       loaded graph and its explicit QLinear/QConv2D replacement in the
       quantized template. */
    int32_t node_index;
    int32_t weight_axis;
    const char* input_tensor_name;
    const char* output_tensor_name;
    const char* source_weight_name;
    const char* packed_weight_name;
    /* QLinear requires both bias names. QConv2D accepts either both names or
       neither, matching the physical-byte runtime contract. */
    const char* source_bias_name;
    const char* packed_bias_name;
} volvoxai_ptq_layer_spec_t;

#define VOLVOXAI_PTQ_LAYER_SPEC_INIT \
    { sizeof(volvoxai_ptq_layer_spec_t), VOLVOXAI_PTQ_LAYER_QLINEAR, -1, 0, \
      NULL, NULL, NULL, NULL, NULL, NULL }

typedef struct volvoxai_ptq_input_binding {
    uint32_t struct_size;
    const char* tensor_name;
    int32_t dtype;
    const void* data;
    uint64_t nbytes;
} volvoxai_ptq_input_binding_t;

#define VOLVOXAI_PTQ_INPUT_BINDING_INIT \
    { sizeof(volvoxai_ptq_input_binding_t), NULL, VOLVOXAI_DTYPE_F32, NULL, 0 }

typedef struct volvoxai_ptq_package_options {
    uint32_t struct_size;
    /* The template/source remain unchanged. Output paths must not exist. */
    const char* template_config_path;
    const char* source_weights_path;
    const char* output_config_path;
    const char* output_weights_path;
} volvoxai_ptq_package_options_t;

#define VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT \
    { sizeof(volvoxai_ptq_package_options_t), NULL, NULL, NULL, NULL }

/* Returns NULL unless an FP32 model is loaded with no active/merged adapter.
   Operations on a stale plan return -1; its sample-count query returns zero. */
VolvoxAIPTQPlan* volvoxai_ptq_plan_create(void);
void volvoxai_ptq_plan_destroy(VolvoxAIPTQPlan* plan);
int volvoxai_ptq_plan_add_tensor(VolvoxAIPTQPlan* plan,
                                 const volvoxai_ptq_tensor_spec_t* spec);
int volvoxai_ptq_plan_add_layer(VolvoxAIPTQPlan* plan,
                                const volvoxai_ptq_layer_spec_t* spec);
/* Every graph input must appear exactly once. Input sizes/dtypes and every
   observed F32 tensor are preflighted before observers are committed, so a
   failed sample never partially changes calibration ranges. */
int volvoxai_engine_ptq_plan_calibrate(
    VolvoxAIPTQPlan* plan, const volvoxai_ptq_input_binding_t* bindings,
    int32_t binding_count);
/* Named form recorded in the package's ptq_authoring.samples audit list.
   Sample names must be non-empty and unique within a plan. */
int volvoxai_engine_ptq_plan_calibrate_sample(
    VolvoxAIPTQPlan* plan, const char* sample_name,
    const volvoxai_ptq_input_binding_t* bindings, int32_t binding_count);
int volvoxai_ptq_plan_tensor_params(const VolvoxAIPTQPlan* plan,
                                    const char* tensor_name,
                                    volvoxai_ptq_params_t* out_params);
uint64_t volvoxai_ptq_plan_calibration_samples(const VolvoxAIPTQPlan* plan);
int volvoxai_ptq_plan_write_package(
    const VolvoxAIPTQPlan* plan,
    const volvoxai_ptq_package_options_t* options);

/* Strict training backend policy used by service runtimes. */
int volvoxai_engine_require_training_backend(int backend);
int volvoxai_engine_last_training_backend(void);

int volvoxai_engine_apply_tensor_update_f32(const char* name, const float* update, long numel,
                                            int mode, float learning_rate, float beta1,
                                            float beta2, float epsilon, float weight_decay,
                                            float max_grad_norm, long step);
int volvoxai_engine_train_step_multi(
    const volvoxai_cross_entropy_loss_t* losses, int loss_count,
    const char* const* trainable_names, int trainable_count,
    int update_mode, float learning_rate, float beta1, float beta2,
    float epsilon, float weight_decay, float max_grad_norm,
    long step, int accumulation_steps, int flush_accumulation,
    int reset_accumulation, float* out_loss,
    volvoxai_cross_entropy_metric_t* out_metrics,
    int* out_accumulated_microbatches, int* out_update_applied);

/* Discard an unfinished native accumulation window without processing another
   microbatch. */
int volvoxai_engine_reset_gradient_accumulation(void);
int volvoxai_engine_train_step(const char* logits_name, const int* targets, int target_count,
                               int ignore_index,
                               const char* const* trainable_names, int trainable_count,
                               int update_mode, float learning_rate, float beta1, float beta2,
                               float epsilon, float weight_decay, float max_grad_norm,
                               long step, float* out_loss, int* out_correct, int* out_examples);
int volvoxai_engine_lora_train_step(const char* logits_name, const int* targets, int target_count,
                                    int ignore_index,
                                    const char* const* trainable_names, int trainable_count,
                                    int update_mode, float learning_rate, float beta1, float beta2,
                                    float epsilon, float weight_decay, float max_grad_norm,
                                    long step, float* out_loss, int* out_correct, int* out_examples);

int volvoxai_engine_save_optimizer_state(const char* path, long training_step);
int volvoxai_engine_load_optimizer_state(const char* path, long* training_step);

#ifdef __cplusplus
}
#endif

#endif
