#ifndef VOLVOXAI_TRAINING_CORE_H
#define VOLVOXAI_TRAINING_CORE_H

#include "../runtime/engine_core.h"
#include "../generated/operator_vocabulary.h"
#include "volvoxai_full_enums.h"
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
    VOLVOXAI_TRAINING_BACKEND_METAL = 3,
    /* Keep this equal to VX_BACKEND_KIND_CUDA. */
    VOLVOXAI_TRAINING_BACKEND_CUDA = 4
} VolvoxAITrainingBackend;

/* Full-profile eager, first-order reverse-mode autograd.  A context owns an
   independent define-by-run tape and temporarily projects it into the native
   executor only while an API call holds the engine model lock.  Consequently
   exactly one context may exist and it may be created only while no packaged
   model is loaded.  These declarations have no implementation or exported
   symbols in the inference profile. */
#define VOLVOXAI_AUTOGRAD_ABI_VERSION 1u
#define VOLVOXAI_AUTOGRAD_CURRENT_BACKEND (-1)

typedef struct VolvoxAIAutogradContext VolvoxAIAutogradContext;
typedef struct VolvoxAIAutogradTensor VolvoxAIAutogradTensor;

typedef struct volvoxai_autograd_options {
    uint32_t struct_size;
    /* VOLVOXAI_AUTOGRAD_CURRENT_BACKEND binds the context to the currently
       configured built-in backend.  A concrete VOLVOXAI_BACKEND_* value is an
       assertion, not a request to reconfigure the engine. CPU and native GPU
       backends are accepted; custom backends are not differentiable. */
    int32_t backend;
    /* Deterministic counter mixed into Dropout/attention-dropout masks. */
    uint32_t training_counter;
} volvoxai_autograd_options_t;

#define VOLVOXAI_AUTOGRAD_OPTIONS_INIT \
    { sizeof(volvoxai_autograd_options_t), \
      VOLVOXAI_AUTOGRAD_CURRENT_BACKEND, 0u }

typedef struct volvoxai_autograd_tensor_spec {
    uint32_t struct_size;
    /* ABI v1 admits F32 values and I32 discrete inputs/masks. */
    int32_t dtype;
    int32_t ndim;
    int32_t shape[8];
    const void* data;
    uint64_t nbytes;
    int32_t requires_grad;
} volvoxai_autograd_tensor_spec_t;

#define VOLVOXAI_AUTOGRAD_TENSOR_SPEC_INIT \
    { sizeof(volvoxai_autograd_tensor_spec_t), VOLVOXAI_DTYPE_F32, 0, \
      { 0, 0, 0, 0, 0, 0, 0, 0 }, NULL, 0u, 0 }

/* Inputs and outputs use the same semantic keys as graph nodes (for
   example input/weight/bias or a/b).  Output shapes are always explicit: the
   eager ABI does not contain model-specific shape inference. */
typedef struct volvoxai_autograd_input {
    const char* key;
    const VolvoxAIAutogradTensor* tensor;
} volvoxai_autograd_input_t;

typedef struct volvoxai_autograd_output_spec {
    const char* key;
    /* ABI v1 admits F32 values and I32 discrete outputs. */
    int32_t dtype;
    int32_t ndim;
    int32_t shape[8];
} volvoxai_autograd_output_spec_t;

typedef struct volvoxai_autograd_op {
    uint32_t struct_size;
    VxOperatorKind operator_kind;
    const volvoxai_autograd_input_t* inputs;
    int32_t input_count;
    const volvoxai_autograd_output_spec_t* outputs;
    int32_t output_count;
    /* Optional JSON object using the ordinary native operator parameter
       schema.  The call parses and owns a deep copy. */
    const char* params_json;
} volvoxai_autograd_op_t;

#define VOLVOXAI_AUTOGRAD_OP_INIT \
    { sizeof(volvoxai_autograd_op_t), VX_OP_UNSPECIFIED, NULL, 0, NULL, 0, NULL }

uint32_t volvoxai_autograd_abi_version(void);
VolvoxAIAutogradContext* volvoxai_autograd_context_create(
    const volvoxai_autograd_options_t* options);
void volvoxai_autograd_context_destroy(VolvoxAIAutogradContext* context);
int volvoxai_autograd_context_backend(const VolvoxAIAutogradContext* context);

VolvoxAIAutogradTensor* volvoxai_autograd_tensor_create(
    VolvoxAIAutogradContext* context,
    const volvoxai_autograd_tensor_spec_t* spec);
/* Detach is an independent value copy in ABI v1.  The result is a leaf with
   requires_grad disabled; callers may enable it explicitly below. */
VolvoxAIAutogradTensor* volvoxai_autograd_tensor_detach(
    VolvoxAIAutogradContext* context,
    const VolvoxAIAutogradTensor* source);
int volvoxai_autograd_tensor_set_requires_grad(
    VolvoxAIAutogradTensor* tensor, int requires_grad);
int volvoxai_autograd_tensor_set_data(VolvoxAIAutogradTensor* tensor,
                                      const void* data, uint64_t nbytes);
int volvoxai_autograd_tensor_copy_data(const VolvoxAIAutogradTensor* tensor,
                                       void* output, uint64_t nbytes);
int volvoxai_autograd_tensor_copy_grad(const VolvoxAIAutogradTensor* tensor,
                                       float* output, int64_t numel);
int volvoxai_autograd_tensor_info(const VolvoxAIAutogradTensor* tensor,
                                  int64_t* numel, int32_t* shape,
                                  int32_t* ndim, int32_t* dtype,
                                  int32_t* requires_grad, int32_t* is_leaf);
/* Releases caller ownership of a tensor handle.  A live tensor is releasable
   when no current tape node references it.  Non-leaf handles invalidated by
   clear_graph() are always releasable.  After success the pointer itself is
   invalid and must not be used again. */
int volvoxai_autograd_tensor_release(VolvoxAIAutogradTensor* tensor);

/* Executes one operation immediately and appends it to the tape when gradient
   recording is enabled and an F32 input requires gradients.  Otherwise the
   temporary node is retired and its results are detached leaves.
   output_capacity must equal descriptor->output_count.  On failure no output
   handle escapes and the previous tape remains usable.  A backend coherence
   failure while rolling back is fatal to the context: subsequent calls reject
   it and the caller must destroy the context. */
int volvoxai_autograd_apply(VolvoxAIAutogradContext* context,
                            const volvoxai_autograd_op_t* descriptor,
                            VolvoxAIAutogradTensor** outputs,
                            int32_t output_capacity);

int volvoxai_autograd_no_grad_begin(VolvoxAIAutogradContext* context);
int volvoxai_autograd_no_grad_end(VolvoxAIAutogradContext* context);

/* The seed must contain exactly output.numel F32 values.  NULL is accepted
   only for a scalar output and means a seed of one.  Leaf gradients accumulate
   across calls until zero_grad(); gradients are ordinary detached buffers, so
   higher-order differentiation is deliberately outside ABI v1. */
int volvoxai_autograd_backward(VolvoxAIAutogradContext* context,
                               const VolvoxAIAutogradTensor* output,
                               const float* seed, int64_t seed_numel,
                               int retain_graph);
/* NULL zeros every live leaf. */
int volvoxai_autograd_zero_grad(VolvoxAIAutogradContext* context,
                                VolvoxAIAutogradTensor* tensor);
/* Invalidates non-leaf handles and clears executed nodes while preserving
   leaves, their values, and accumulated gradients for the next eager tape.
   Release invalidated handles explicitly with tensor_release(). */
int volvoxai_autograd_clear_graph(VolvoxAIAutogradContext* context);

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

/* Quantize one loaded F32 model weight into newly named I8 weight, F32 scale,
   and I8 zero-point tensors. This deliberately does not rewrite graph
   operators: the caller must reference the two parameter tensors from the
   graph's canonical `quantization.tensors` entry. The new tensors are added to
   weight file 0, which can then be persisted with
   volvoxai_engine_save_weight_file(0, path). */
int volvoxai_engine_ptq_materialize_weight_i8(const char* source_name,
                                              const char* output_name,
                                              const char* scale_name,
                                              const char* zero_point_name,
                                              int32_t axis,
                                              int32_t* out_scale_count);

/* Explicit full-profile PTQ package authoring. A plan is bound to the exact
   loaded-model generation that created it and becomes permanently stale after
   reload, shutdown, graph/weight/training mutation, or adapter activation.
   Active or merged adapters are rejected. The plan observes tensors from a
   loaded FP32 graph, but never rewrites that graph. Package creation consumes
   a caller-authored quantized graph template and validates every selected
   QLinear/QConv2D node before filling its quantization metadata and adding
   packed tensors to a pass-through copy of the source safetensors file. */
typedef struct VolvoxAIPTQPlan VolvoxAIPTQPlan;

typedef struct volvoxai_ptq_tensor_spec {
    uint32_t struct_size;
    const char* tensor_name;
    int32_t dtype;
    int32_t scheme;
    /* The byte tensor the template renamed this value to, or NULL. */
    const char* quantized_tensor_name;
} volvoxai_ptq_tensor_spec_t;

#define VOLVOXAI_PTQ_TENSOR_SPEC_INIT \
    { sizeof(volvoxai_ptq_tensor_spec_t), NULL, VOLVOXAI_DTYPE_I8, \
      VX_PTQ_SCHEME_SYMMETRIC, NULL }

typedef struct volvoxai_ptq_layer_spec {
    uint32_t struct_size;
    int32_t kind;
    /* Position of the canonical FP32 source node in the loaded graph. The
       template is located by node_id instead: authoring inserts the affine
       boundaries, so the two graphs no longer agree about positions. */
    int32_t node_index;
    int32_t weight_axis;
    const char* input_tensor_name;
    const char* output_tensor_name;
    const char* source_weight_name;
    const char* packed_weight_name;
    /* QLinear requires both bias names. QConv2D accepts either both names or
       neither, matching the byte runtime contract. */
    const char* source_bias_name;
    const char* packed_bias_name;
    /* Identifies the same node inside the quantized template. */
    const char* node_id;
} volvoxai_ptq_layer_spec_t;

#define VOLVOXAI_PTQ_LAYER_SPEC_INIT \
    { sizeof(volvoxai_ptq_layer_spec_t), VX_PTQ_LAYER_QLINEAR, -1, 0, \
      NULL, NULL, NULL, NULL, NULL, NULL, NULL }

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
    const char* template_graph_path;
    const char* source_weights_path;
    const char* output_graph_path;
    const char* output_weights_path;
    const char* logical_fingerprint;
    const char* profile_coverage_json;
    unsigned char** graph_bytes;
    size_t* graph_size;
    unsigned char** weights_bytes;
    size_t* weights_size;
} volvoxai_ptq_package_options_t;

#define VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT \
    { sizeof(volvoxai_ptq_package_options_t), NULL, NULL, NULL, NULL, NULL, NULL, \
      NULL, NULL, NULL, NULL }

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
int volvoxai_engine_finalize_host_tensor_update_f32(const char* name, long numel);
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

#endif /* VOLVOXAI_TRAINING_CORE_H */
