#include "training/training_core.h"
#include "runtime/runtime_state.h"
#include "adapter_runtime_internal.h"
#include "safetensors.h"
#include "volvoxai.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
#include <sys/wait.h>
#include <unistd.h>
#endif

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int result = length && fwrite(text, 1, length, file) != length ? -1 : 0;
    if (fclose(file) != 0) result = -1;
    return result;
}

static int file_contains(const char* path, const char* needle) {
    char buffer[4096];
    FILE* file = fopen(path, "rb");
    if (!file) return 0;
    size_t length = fread(buffer, 1, sizeof(buffer) - 1, file);
    int complete = feof(file);
    if (fclose(file) != 0) complete = 0;
    buffer[length] = 0;
    return complete && strstr(buffer, needle) != NULL;
}

static int closef(float left, float right, float tolerance) {
    return fabsf(left - right) <= tolerance;
}

static int stage_ptq_adapter(void) {
    static const float a[3] = {1.0f, 0.0f, -1.0f};
    static const float b[2] = {0.5f, -0.5f};
    VxAdapterTargetSpec target;
    memset(&target, 0, sizeof(target));
    target.weight_name = "weight";
    target.kind = VX_ADAPTER_LORA;
    target.d_in = 3;
    target.d_out = 2;
    target.rank = 1;
    target.alpha = 1.0f;
    target.scale = 1.0f;
    target.a.data = a;
    target.a.nbytes = sizeof(a);
    target.a.dtype = VX_ADAPTER_DTYPE_F32;
    target.a.rows = 3;
    target.a.cols = 1;
    target.b.data = b;
    target.b.nbytes = sizeof(b);
    target.b.dtype = VX_ADAPTER_DTYPE_F32;
    target.b.rows = 1;
    target.b.cols = 2;
    VxAdapterVersionSpec version = {
        "ptq-active", "ptq-active", &target, 1, NULL
    };
    return vx_adapter_stage(&version);
}

static int execute_public_i8_package(
        const char* graph_path, const char* weights_path,
        const char* input_name, uint32_t input_rank,
        const int64_t* input_shape, const int8_t* input_data,
        size_t input_bytes, const char* output_name,
        int8_t* output_data, size_t output_bytes) {
    const char* weight_paths[1] = {weights_path};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxTensorBinding binding = VX_TENSOR_BINDING_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    int status = -1;

    if (!graph_path || !weights_path || !input_name || !input_shape ||
        !input_data || !input_bytes || !output_name || !output_data ||
        !output_bytes || input_rank > VX_MAX_TENSOR_RANK) goto done;
    source.graph_path = graph_path;
    source.weight_paths = weight_paths;
    source.weight_path_count = 1u;
    binding.name = input_name;
    binding.dtype = VX_DTYPE_I8;
    binding.rank = input_rank;
    for (uint32_t axis = 0; axis < input_rank; axis++)
        binding.shape[axis] = input_shape[axis];
    binding.data = input_data;
    binding.byte_size = input_bytes;
    binding.location = VX_MEMORY_HOST;
    if (vx_runtime_create(&runtime_options, &runtime, &report) != VX_STATUS_OK ||
        vx_runtime_load_model(runtime, &source, &model, &report) != VX_STATUS_OK ||
        vx_model_compile(model, &policy, &compiled, &report) != VX_STATUS_OK ||
        vx_compiled_model_create_context(
            compiled, &context_options, &context, &report) != VX_STATUS_OK ||
        vx_execution_context_execute(
            context, &binding, 1u, &result, &report) != VX_STATUS_OK ||
        vx_result_read(result, output_name, output_data, output_bytes,
                       NULL, &report) != VX_STATUS_OK) goto done;
    status = 0;
done:
    vx_result_release(result);
    if (context) (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    return status;
}

static int test_qlinear_package(void) {
    const char* source_graph_path = "/tmp/volvox-ptq-package-source.graph.json";
    const char* template_graph_path = "/tmp/volvox-ptq-package-template.graph.json";
    const char* invalid_template_path = "/tmp/volvox-ptq-package-invalid-template.json";
    const char* retired_shape_system_template_path =
        "/tmp/volvox-ptq-package-retired-shape-system-template.json";
    const char* wrong_layout_template_path = "/tmp/volvox-ptq-package-in-out-template.json";
    const char* missing_dependency_template_path = "/tmp/volvox-ptq-package-missing-template.json";
    const char* namespace_collision_template_path = "/tmp/volvox-ptq-package-collision-template.json";
    const char* bias_descriptor_template_path = "/tmp/volvox-ptq-package-bias-descriptor.json";
    const char* mixed_template_path = "/tmp/volvox-ptq-package-mixed-template.json";
    const char* source_weights_path = "/tmp/volvox-ptq-package-source.safetensors";
    const char* legacy_weights_path = "/tmp/volvox-ptq-package-legacy.safetensors";
    const char* mixed_weights_path = "/tmp/volvox-ptq-package-mixed.safetensors";
    const char* output_graph_path = "/tmp/volvox-ptq-package-output.graph.json";
    const char* output_weights_path = "/tmp/volvox-ptq-package-output.safetensors";
    const char* source_config =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}}],\"outputs\":[\"y\"]}";
    const char* target_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const char* invalid_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"linear\",\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const char* retired_shape_system_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"shape_system\":\"volvox-bounded-shape/v1\",\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const char* wrong_layout_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{\"weight_layout\":\"din_dout\"}}],\"outputs\":[\"y\"]}";
    const char* missing_dependency_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}},{\"id\":\"missing-add\",\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"y\",\"b\":\"missing.weight\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"z\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}}],\"outputs\":[\"z\"]}";
    const char* namespace_collision_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"},"
        "\"weight.i8\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const char* bias_descriptor_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}}],\"weights_quantization\":{\"bias.i32\":{}},"
        "\"outputs\":[\"y\"]}";
    const char* mixed_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qlinear\",\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\",\"bias\":\"bias.i32\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"dtype\":\"int8\",\"shape\":[2,2]}},"
        "\"params\":{}},{\"id\":\"pass\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"other.bias\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"pass\",\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"params\":{}}],\"outputs\":[\"y\",\"pass\"]}";
    const int matrix_shape[2] = {2, 3};
    const int vector_shape[1] = {2};
    const float weights[6] = {1.0f, 2.0f, -1.0f, -2.0f, 0.5f, 3.0f};
    const float other_weights[6] = {9.0f, 8.0f, 7.0f, 6.0f, 5.0f, 4.0f};
    const float bias[2] = {0.25f, -0.5f};
    const float other_bias[2] = {3.0f, 4.0f};
    const float mixed_other_bias[2] = {30.0f, 40.0f};
    const float sample0[6] = {1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -1.0f};
    const float sample1[6] = {-2.0f, 0.5f, 1.5f, 3.0f, -4.0f, 2.0f};
    remove(source_graph_path);
    remove(template_graph_path);
    remove(invalid_template_path);
    remove(retired_shape_system_template_path);
    remove(wrong_layout_template_path);
    remove(missing_dependency_template_path);
    remove(namespace_collision_template_path);
    remove(bias_descriptor_template_path);
    remove(mixed_template_path);
    remove(source_weights_path);
    remove(legacy_weights_path);
    remove(mixed_weights_path);
    remove(output_graph_path);
    remove(output_weights_path);

    CHECK(write_text(source_graph_path, source_config) == 0);
    CHECK(write_text(template_graph_path, target_template) == 0);
    CHECK(write_text(invalid_template_path, invalid_template) == 0);
    CHECK(write_text(retired_shape_system_template_path,
                     retired_shape_system_template) == 0);
    CHECK(write_text(wrong_layout_template_path, wrong_layout_template) == 0);
    CHECK(write_text(missing_dependency_template_path,
                     missing_dependency_template) == 0);
    CHECK(write_text(namespace_collision_template_path,
                     namespace_collision_template) == 0);
    CHECK(write_text(bias_descriptor_template_path,
                     bias_descriptor_template) == 0);
    CHECK(write_text(mixed_template_path, mixed_template) == 0);
    SafetensorsFile source_file;
    CHECK(safetensors_init_empty(&source_file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&source_file, "weight", SAFETENSORS_DTYPE_F32,
                                 matrix_shape, 2, weights, sizeof(weights)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "bias", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "other.weight", SAFETENSORS_DTYPE_F32,
                                 matrix_shape, 2, other_weights,
                                 sizeof(other_weights)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "other.bias", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, other_bias,
                                 sizeof(other_bias)) == 0);
    CHECK(safetensors_save(source_weights_path, &source_file) == 0);
    CHECK(safetensors_set_metadata_json(
              &source_file, "{\"weights_quantization\":\"legacy\"}") == 0);
    CHECK(safetensors_save(legacy_weights_path, &source_file) == 0);
    safetensors_free(&source_file);
    CHECK(safetensors_init_empty(&source_file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&source_file, "weight", SAFETENSORS_DTYPE_F32,
                                 matrix_shape, 2, weights, sizeof(weights)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "bias", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "other.weight", SAFETENSORS_DTYPE_F32,
                                 matrix_shape, 2, other_weights,
                                 sizeof(other_weights)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "other.bias", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, mixed_other_bias,
                                 sizeof(mixed_other_bias)) == 0);
    CHECK(safetensors_save(mixed_weights_path, &source_file) == 0);
    safetensors_free(&source_file);

    CHECK(volvoxai_engine_init(source_graph_path, source_weights_path) == 0);
    int helper_scale_count = 0;
    CHECK(volvoxai_engine_ptq_materialize_weight_i8(
              "weight", "scratch.weight.i8", "scratch.weight.scale",
              "scratch.weight.zero_point", 0, &helper_scale_count) == 0);
    CHECK(helper_scale_count == 2);
    long helper_count = 0;
    int helper_dtype = -1;
    CHECK(volvoxai_engine_tensor_info_ex("scratch.weight.i8", &helper_count,
                                         NULL, NULL, &helper_dtype, NULL) == 0);
    CHECK(helper_count == 6 && helper_dtype == VOLVOXAI_DTYPE_I8);
    CHECK(volvoxai_engine_tensor_info_ex(
              "scratch.weight.zero_point", &helper_count, NULL, NULL,
              &helper_dtype, NULL) == 0);
    CHECK(helper_count == 2 && helper_dtype == VOLVOXAI_DTYPE_I8);
    int8_t helper_zero_points[2] = {1, 1};
    CHECK(volvoxai_engine_copy_tensor_raw(
              "scratch.weight.zero_point", helper_zero_points,
              sizeof(helper_zero_points)) == 0);
    CHECK(helper_zero_points[0] == 0 && helper_zero_points[1] == 0);
    CHECK(volvoxai_engine_remove_model_tensor(
              "scratch.weight.zero_point") == 0);
    CHECK(volvoxai_engine_remove_model_tensor("scratch.weight.scale") == 0);
    CHECK(volvoxai_engine_remove_model_tensor("scratch.weight.i8") == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        sample0, sizeof(sample0)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    volvoxai_ptq_observer_t helper_observer;
    volvoxai_ptq_observer_reset(&helper_observer);
    CHECK(volvoxai_engine_ptq_observe_tensor("y", &helper_observer) == 0);
    CHECK(closef(helper_observer.minimum, -2.5f, 1.0e-6f));
    CHECK(closef(helper_observer.maximum, 5.25f, 1.0e-6f));
    CHECK(stage_ptq_adapter() == 0);
    CHECK(volvoxai_engine_adapter_activate("ptq-active") == 0);
    CHECK(volvoxai_ptq_plan_create() == NULL);
    CHECK(volvoxai_engine_ptq_materialize_weight_i8(
              "weight", "active.weight.i8", "active.weight.scale",
              "active.weight.zero_point", 0, NULL) != 0);
    CHECK(volvoxai_engine_tensor_info("active.weight.i8", NULL, NULL, NULL) != 0);
    CHECK(volvoxai_engine_adapter_activate("") == 0);
    VolvoxAIPTQPlan* adapter_stale = volvoxai_ptq_plan_create();
    CHECK(adapter_stale != NULL);
    volvoxai_ptq_tensor_spec_t stale_tensor = VOLVOXAI_PTQ_TENSOR_SPEC_INIT;
    stale_tensor.tensor_name = "x";
    CHECK(volvoxai_ptq_plan_add_tensor(adapter_stale, &stale_tensor) == 0);
    CHECK(volvoxai_engine_adapter_activate("ptq-active") == 0);
    volvoxai_ptq_input_binding_t stale_binding = VOLVOXAI_PTQ_INPUT_BINDING_INIT;
    stale_binding.tensor_name = "x";
    stale_binding.data = sample0;
    stale_binding.nbytes = sizeof(sample0);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              adapter_stale, "adapter-stale", &stale_binding, 1) != 0);
    CHECK(volvoxai_engine_adapter_activate("") == 0);
    volvoxai_ptq_plan_destroy(adapter_stale);
    VolvoxAIPTQPlan* plan = volvoxai_ptq_plan_create();
    CHECK(plan != NULL);
    volvoxai_ptq_tensor_spec_t tensor_spec = VOLVOXAI_PTQ_TENSOR_SPEC_INIT;
    tensor_spec.tensor_name = "x";
    {
        volvoxai_ptq_tensor_spec_t oversized = tensor_spec;
        oversized.struct_size++;
        CHECK(volvoxai_ptq_plan_add_tensor(plan, &oversized) != 0);
    }
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor_spec) == 0);
    tensor_spec.tensor_name = "y";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor_spec) == 0);

    volvoxai_ptq_layer_spec_t layer_spec = VOLVOXAI_PTQ_LAYER_SPEC_INIT;
    layer_spec.node_index = 0;
    layer_spec.input_tensor_name = "x";
    layer_spec.output_tensor_name = "y";
    layer_spec.source_weight_name = "weight";
    layer_spec.packed_weight_name = "weight.i8";
    layer_spec.source_bias_name = "bias";
    layer_spec.packed_bias_name = "bias.i32";
    {
        volvoxai_ptq_layer_spec_t oversized = layer_spec;
        oversized.struct_size++;
        CHECK(volvoxai_ptq_plan_add_layer(plan, &oversized) != 0);
    }
    volvoxai_ptq_layer_spec_t wrong_source = layer_spec;
    wrong_source.source_weight_name = "other.weight";
    wrong_source.source_bias_name = "other.bias";
    CHECK(volvoxai_ptq_plan_add_layer(plan, &wrong_source) != 0);
    wrong_source = layer_spec;
    wrong_source.source_bias_name = "other.bias";
    CHECK(volvoxai_ptq_plan_add_layer(plan, &wrong_source) != 0);
    wrong_source = layer_spec;
    wrong_source.node_index = 1;
    CHECK(volvoxai_ptq_plan_add_layer(plan, &wrong_source) != 0);
    CHECK(volvoxai_ptq_plan_add_layer(plan, &layer_spec) == 0);

    volvoxai_ptq_input_binding_t binding = VOLVOXAI_PTQ_INPUT_BINDING_INIT;
    binding.tensor_name = "x";
    binding.data = sample0;
    binding.nbytes = sizeof(sample0);
    {
        volvoxai_ptq_input_binding_t oversized = binding;
        oversized.struct_size++;
        CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
                  plan, "oversized-binding", &oversized, 1) != 0);
        CHECK(volvoxai_ptq_plan_calibration_samples(plan) == 0);
    }
    volvoxai_ptq_input_binding_t invalid_binding = binding;
    invalid_binding.nbytes -= sizeof(float);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "receipt-invalid", &invalid_binding, 1) != 0);
    CHECK(volvoxai_ptq_plan_calibration_samples(plan) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        sample0, sizeof(sample0)) == 0);
    CHECK(volvoxai_engine_forward_row(0) == 0);
    CHECK(volvoxai_engine_execution_row() == 0);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "receipt-0", &binding, 1) == 0);
    CHECK(volvoxai_engine_execution_row() == 0);
    binding.data = sample1;
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "receipt-1", &binding, 1) == 0);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "receipt-1", &binding, 1) != 0);
    CHECK(volvoxai_ptq_plan_calibration_samples(plan) == 2);
    /* A successful sample freezes the observer/layer inventory. */
    tensor_spec.tensor_name = "unused";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor_spec) != 0);

    volvoxai_ptq_params_t input_params;
    volvoxai_ptq_params_t output_params;
    CHECK(volvoxai_ptq_plan_tensor_params(plan, "x", &input_params) == 0);
    CHECK(volvoxai_ptq_plan_tensor_params(plan, "y", &output_params) == 0);
    CHECK(closef(input_params.observed_min, -4.0f, 1.0e-6f));
    CHECK(closef(input_params.observed_max, 3.0f, 1.0e-6f));
    CHECK(closef(output_params.observed_min, -6.75f, 1.0e-6f));
    CHECK(closef(output_params.observed_max, 8.25f, 1.0e-6f));

    volvoxai_ptq_package_options_t package = VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT;
    package.template_graph_path = invalid_template_path;
    package.source_weights_path = source_weights_path;
    package.output_graph_path = output_graph_path;
    package.output_weights_path = output_weights_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    CHECK(fopen(output_graph_path, "rb") == NULL);
    CHECK(fopen(output_weights_path, "rb") == NULL);
    package.template_graph_path = retired_shape_system_template_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    CHECK(fopen(output_graph_path, "rb") == NULL);
    CHECK(fopen(output_weights_path, "rb") == NULL);
    package.template_graph_path = wrong_layout_template_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    package.template_graph_path = missing_dependency_template_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    package.template_graph_path = namespace_collision_template_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    package.template_graph_path = bias_descriptor_template_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    package.template_graph_path = template_graph_path;
    package.source_weights_path = legacy_weights_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    package.template_graph_path = mixed_template_path;
    package.source_weights_path = mixed_weights_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    package.source_weights_path = source_weights_path;
    package.template_graph_path = template_graph_path;
    {
        volvoxai_ptq_package_options_t oversized = package;
        oversized.struct_size++;
        CHECK(volvoxai_ptq_plan_write_package(plan, &oversized) != 0);
        CHECK(fopen(output_graph_path, "rb") == NULL);
        CHECK(fopen(output_weights_path, "rb") == NULL);
    }
#if defined(__unix__) || defined(__APPLE__) || defined(__ANDROID__)
    int start_pipe[2];
    CHECK(pipe(start_pipe) == 0);
    pid_t children[2] = {-1, -1};
    for (int child = 0; child < 2; child++) {
        children[child] = fork();
        CHECK(children[child] >= 0);
        if (children[child] == 0) {
            close(start_pipe[1]);
            char start = 0;
            if (read(start_pipe[0], &start, 1) != 1) _exit(2);
            close(start_pipe[0]);
            _exit(volvoxai_ptq_plan_write_package(plan, &package) == 0 ? 0 : 1);
        }
    }
    close(start_pipe[0]);
    CHECK(write(start_pipe[1], "xx", 2) == 2);
    close(start_pipe[1]);
    int successes = 0;
    int rejections = 0;
    for (int child = 0; child < 2; child++) {
        int status = 0;
        CHECK(waitpid(children[child], &status, 0) == children[child]);
        CHECK(WIFEXITED(status));
        if (WEXITSTATUS(status) == 0) successes++;
        else if (WEXITSTATUS(status) == 1) rejections++;
    }
    CHECK(successes == 1 && rejections == 1);
#else
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) == 0);
#endif
    CHECK(file_contains(output_graph_path,
                        "\"format\":\"volvox-graph/v1\""));
    CHECK(!file_contains(output_graph_path, "\"ptq_authoring\":"));
    CHECK(file_contains(output_graph_path,
                        "\"format\":\"volvox-affine-safetensors/v1\""));
    CHECK(file_contains(output_graph_path, "\"scale_tensor\":"));
    CHECK(file_contains(output_graph_path, "\"zero_point_tensor\":"));
    CHECK(!file_contains(output_graph_path, "\"weights_quantization\":"));
    CHECK(!file_contains(output_graph_path, "\"outputs_quantization\":"));
    CHECK(!file_contains(output_graph_path, "\"weight\":{}"));
    CHECK(!file_contains(output_graph_path, "\"bias\":{}"));
    /* Safe authoring does not replace an existing package implicitly. */
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) != 0);
    volvoxai_ptq_plan_destroy(plan);
    volvoxai_engine_shutdown();

    SafetensorsFile materialized;
    CHECK(safetensors_load(output_weights_path, &materialized) == 0);
    const SafetensorsTensor* packed_weight =
        safetensors_find_tensor(&materialized, "weight.i8");
    const SafetensorsTensor* packed_bias =
        safetensors_find_tensor(&materialized, "bias.i32");
    CHECK(packed_weight && packed_weight->dtype == SAFETENSORS_DTYPE_I8 &&
          packed_weight->nbytes == sizeof(weights) / sizeof(weights[0]));
    CHECK(packed_bias && packed_bias->dtype == SAFETENSORS_DTYPE_I32 &&
          packed_bias->nbytes == sizeof(bias));
    CHECK(safetensors_find_tensor(&materialized, "weight") == NULL);
    CHECK(safetensors_find_tensor(&materialized, "bias") == NULL);
    CHECK(safetensors_find_tensor(&materialized, "other.weight") != NULL);
    CHECK(safetensors_find_tensor(&materialized, "other.bias") != NULL);
    safetensors_free(&materialized);

    int8_t quantized_input[6] = {0};
    int8_t quantized_output[4] = {0};
    const int64_t quantized_input_shape[2] = {2, 3};
    CHECK(volvoxai_ptq_quantize_f32(sample0, 6, &input_params,
                                    quantized_input, NULL) == 0);
    CHECK(execute_public_i8_package(
              output_graph_path, output_weights_path, "x", 2u,
              quantized_input_shape, quantized_input,
              sizeof(quantized_input), "y", quantized_output,
              sizeof(quantized_output)) == 0);
    const float expected_output[4] = {-1.25f, -1.5f, 5.25f, -2.5f};
    for (int index = 0; index < 4; index++) {
        float dequantized =
            (quantized_output[index] - output_params.zero_point) * output_params.scale;
        CHECK(closef(dequantized, expected_output[index], 0.12f));
    }
    const char* stale_graph_path = "/tmp/volvox-ptq-stale-output.graph.json";
    const char* stale_weights_path = "/tmp/volvox-ptq-stale-output.safetensors";
    remove(stale_graph_path);
    remove(stale_weights_path);
    CHECK(volvoxai_engine_init(source_graph_path, source_weights_path) == 0);
    VolvoxAIPTQPlan* stale_plan = volvoxai_ptq_plan_create();
    CHECK(stale_plan != NULL);
    tensor_spec.tensor_name = "x";
    CHECK(volvoxai_ptq_plan_add_tensor(stale_plan, &tensor_spec) == 0);
    tensor_spec.tensor_name = "y";
    CHECK(volvoxai_ptq_plan_add_tensor(stale_plan, &tensor_spec) == 0);
    CHECK(volvoxai_ptq_plan_add_layer(stale_plan, &layer_spec) == 0);
    binding.data = sample0;
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              stale_plan, "before-reload", &binding, 1) == 0);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(source_graph_path, source_weights_path) == 0);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              stale_plan, "after-reload", &binding, 1) != 0);
    CHECK(volvoxai_ptq_plan_tensor_params(stale_plan, "x", &input_params) != 0);
    CHECK(volvoxai_ptq_plan_calibration_samples(stale_plan) == 0);
    package.template_graph_path = template_graph_path;
    package.output_graph_path = stale_graph_path;
    package.output_weights_path = stale_weights_path;
    CHECK(volvoxai_ptq_plan_write_package(stale_plan, &package) != 0);
    CHECK(fopen(stale_graph_path, "rb") == NULL);
    CHECK(fopen(stale_weights_path, "rb") == NULL);
    volvoxai_ptq_plan_destroy(stale_plan);
    volvoxai_engine_shutdown();

    remove(source_graph_path);
    remove(template_graph_path);
    remove(invalid_template_path);
    remove(retired_shape_system_template_path);
    remove(wrong_layout_template_path);
    remove(missing_dependency_template_path);
    remove(namespace_collision_template_path);
    remove(bias_descriptor_template_path);
    remove(mixed_template_path);
    remove(source_weights_path);
    remove(legacy_weights_path);
    remove(mixed_weights_path);
    remove(output_graph_path);
    remove(output_weights_path);
    remove(stale_graph_path);
    remove(stale_weights_path);
    return 0;
}

static int test_qconv2d_package(void) {
    const char* source_graph_path = "/tmp/volvox-ptq-qconv-source.graph.json";
    const char* template_graph_path = "/tmp/volvox-ptq-qconv-template.graph.json";
    const char* source_weights_path = "/tmp/volvox-ptq-qconv-source.safetensors";
    const char* output_graph_path = "/tmp/volvox-ptq-qconv-output.graph.json";
    const char* output_weights_path = "/tmp/volvox-ptq-qconv-output.safetensors";
    const char* source_config =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"cx\":{\"shape\":[1,3,3,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"cx\",\"weight\":\"conv.weight\","
        "\"bias\":\"conv.bias\"},\"outputs\":{\"out\":\"cy\"},"
        "\"outputs_shape\":{\"out\":[1,2,2,2]},\"params\":{"
        "\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"stride\":[1,1],\"padding\":[0,0],\"dilation\":[1,1],"
        "\"groups\":1}}],\"outputs\":[\"cy\"]}";
    const char* target_template =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{\"cx\":{\"shape\":[1,3,3,1],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qconv\",\"opType\":\"QConv2D\","
        "\"inputs\":{\"input\":\"cx\",\"weight\":\"conv.weight.i8\","
        "\"bias\":\"conv.bias.i32\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"cy\",\"dtype\":\"int8\",\"shape\":[1,2,2,2]}},"
        "\"params\":{"
        "\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"stride\":[1,1],\"padding\":[0,0],\"dilation\":[1,1],"
        "\"groups\":1}}],\"outputs\":[\"cy\"]}";
    const int weight_shape[4] = {2, 2, 2, 1};
    const int bias_shape[1] = {2};
    const float weight[8] = {1.0f, 0.0f, 0.0f, -1.0f,
                             0.5f, 1.0f, -0.5f, 2.0f};
    const float bias[2] = {0.5f, -0.25f};
    const float sample0[9] = {1.0f, 2.0f, 3.0f,
                              4.0f, 5.0f, 6.0f,
                              7.0f, 8.0f, 9.0f};
    const float sample1[9] = {-1.0f, -2.0f, -3.0f,
                              -4.0f, -5.0f, -6.0f,
                              -7.0f, -8.0f, -9.0f};
    remove(source_graph_path);
    remove(template_graph_path);
    remove(source_weights_path);
    remove(output_graph_path);
    remove(output_weights_path);
    CHECK(write_text(source_graph_path, source_config) == 0);
    CHECK(write_text(template_graph_path, target_template) == 0);
    SafetensorsFile source_file;
    CHECK(safetensors_init_empty(&source_file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&source_file, "conv.weight", SAFETENSORS_DTYPE_F32,
                                 weight_shape, 4, weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&source_file, "conv.bias", SAFETENSORS_DTYPE_F32,
                                 bias_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(source_weights_path, &source_file) == 0);
    safetensors_free(&source_file);

    CHECK(volvoxai_engine_init(source_graph_path, source_weights_path) == 0);
    VolvoxAIPTQPlan* plan = volvoxai_ptq_plan_create();
    CHECK(plan != NULL);
    volvoxai_ptq_tensor_spec_t tensor = VOLVOXAI_PTQ_TENSOR_SPEC_INIT;
    tensor.tensor_name = "cx";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor) == 0);
    tensor.tensor_name = "cy";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor) == 0);
    volvoxai_ptq_layer_spec_t layer = VOLVOXAI_PTQ_LAYER_SPEC_INIT;
    layer.kind = VX_PTQ_LAYER_QCONV2D;
    layer.node_index = 0;
    layer.input_tensor_name = "cx";
    layer.output_tensor_name = "cy";
    layer.source_weight_name = "conv.weight";
    layer.packed_weight_name = "conv.weight.i8";
    layer.source_bias_name = "conv.bias";
    layer.packed_bias_name = "conv.bias.i32";
    CHECK(volvoxai_ptq_plan_add_layer(plan, &layer) == 0);
    volvoxai_ptq_input_binding_t binding = VOLVOXAI_PTQ_INPUT_BINDING_INIT;
    binding.tensor_name = "cx";
    binding.data = sample0;
    binding.nbytes = sizeof(sample0);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "conv-0", &binding, 1) == 0);
    binding.data = sample1;
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "conv-1", &binding, 1) == 0);
    volvoxai_ptq_params_t input_params;
    volvoxai_ptq_params_t output_params;
    CHECK(volvoxai_ptq_plan_tensor_params(plan, "cx", &input_params) == 0);
    CHECK(volvoxai_ptq_plan_tensor_params(plan, "cy", &output_params) == 0);
    volvoxai_ptq_package_options_t package = VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT;
    package.template_graph_path = template_graph_path;
    package.source_weights_path = source_weights_path;
    package.output_graph_path = output_graph_path;
    package.output_weights_path = output_weights_path;
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) == 0);
    volvoxai_ptq_plan_destroy(plan);
    volvoxai_engine_shutdown();

    int8_t quantized_input[9] = {0};
    int8_t quantized_output[8] = {0};
    const int64_t quantized_input_shape[4] = {1, 3, 3, 1};
    CHECK(volvoxai_ptq_quantize_f32(sample0, 9, &input_params,
                                    quantized_input, NULL) == 0);
    CHECK(execute_public_i8_package(
              output_graph_path, output_weights_path, "cx", 4u,
              quantized_input_shape, quantized_input,
              sizeof(quantized_input), "cy", quantized_output,
              sizeof(quantized_output)) == 0);
    const float expected_output[8] = {
        -3.5f, 10.25f, -3.5f, 13.25f,
        -3.5f, 19.25f, -3.5f, 22.25f
    };
    for (int index = 0; index < 8; index++) {
        float dequantized =
            (quantized_output[index] - output_params.zero_point) * output_params.scale;
        CHECK(closef(dequantized, expected_output[index], 0.7f));
    }
    remove(source_graph_path);
    remove(template_graph_path);
    remove(source_weights_path);
    remove(output_graph_path);
    remove(output_weights_path);
    return 0;
}

static int test_rejects_noncanonical_source_layouts(void) {
    const char* weights_path = "/tmp/volvox-ptq-layout-source.safetensors";
    const char* linear_path = "/tmp/volvox-ptq-layout-linear.json";
    const char* conv_path = "/tmp/volvox-ptq-layout-conv.json";
    const char* linear_config =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"weight_layout\":\"din_dout\"}}],\"outputs\":[\"y\"]}";
    const char* conv_config =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"cx\":{\"shape\":[1,2,2,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"cx\",\"weight\":\"cw\",\"bias\":\"cb\"},"
        "\"outputs\":{\"out\":\"cy\"},"
        "\"outputs_shape\":{\"out\":[1,2,2,1]},\"params\":{"
        "\"data_layout\":\"NHWC\",\"weight_layout\":\"HWIO\","
        "\"stride\":[1,1],\"padding\":[0,0],\"dilation\":[1,1],"
        "\"groups\":1}}],\"outputs\":[\"cy\"]}";
    const int matrix_shape[2] = {2, 2};
    const int vector_shape[1] = {2};
    const int conv_shape[4] = {1, 1, 1, 1};
    const int conv_bias_shape[1] = {1};
    const float matrix[4] = {1, 2, 3, 4};
    const float vector[2] = {0, 0};
    const float conv_weight[1] = {1};
    const float conv_bias[1] = {0};
    remove(weights_path);
    remove(linear_path);
    remove(conv_path);
    CHECK(write_text(linear_path, linear_config) == 0);
    CHECK(write_text(conv_path, conv_config) == 0);
    SafetensorsFile weights;
    CHECK(safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&weights, "w", SAFETENSORS_DTYPE_F32,
                                 matrix_shape, 2, matrix, sizeof(matrix)) == 0);
    CHECK(safetensors_add_tensor(&weights, "b", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, vector, sizeof(vector)) == 0);
    CHECK(safetensors_add_tensor(&weights, "cw", SAFETENSORS_DTYPE_F32,
                                 conv_shape, 4, conv_weight,
                                 sizeof(conv_weight)) == 0);
    CHECK(safetensors_add_tensor(&weights, "cb", SAFETENSORS_DTYPE_F32,
                                 conv_bias_shape, 1, conv_bias,
                                 sizeof(conv_bias)) == 0);
    CHECK(safetensors_save(weights_path, &weights) == 0);
    safetensors_free(&weights);

    CHECK(volvoxai_engine_init(linear_path, weights_path) == 0);
    VolvoxAIPTQPlan* plan = volvoxai_ptq_plan_create();
    CHECK(plan != NULL);
    volvoxai_ptq_tensor_spec_t tensor = VOLVOXAI_PTQ_TENSOR_SPEC_INIT;
    tensor.tensor_name = "x";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor) == 0);
    tensor.tensor_name = "y";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor) == 0);
    volvoxai_ptq_layer_spec_t layer = VOLVOXAI_PTQ_LAYER_SPEC_INIT;
    layer.node_index = 0;
    layer.input_tensor_name = "x";
    layer.output_tensor_name = "y";
    layer.source_weight_name = "w";
    layer.packed_weight_name = "w.i8";
    layer.source_bias_name = "b";
    layer.packed_bias_name = "b.i32";
    CHECK(volvoxai_ptq_plan_add_layer(plan, &layer) != 0);
    volvoxai_ptq_plan_destroy(plan);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_init(conv_path, weights_path) == 0);
    plan = volvoxai_ptq_plan_create();
    CHECK(plan != NULL);
    tensor.tensor_name = "cx";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor) == 0);
    tensor.tensor_name = "cy";
    CHECK(volvoxai_ptq_plan_add_tensor(plan, &tensor) == 0);
    layer.kind = VX_PTQ_LAYER_QCONV2D;
    layer.input_tensor_name = "cx";
    layer.output_tensor_name = "cy";
    layer.source_weight_name = "cw";
    layer.packed_weight_name = "cw.i8";
    layer.source_bias_name = "cb";
    layer.packed_bias_name = "cb.i32";
    CHECK(volvoxai_ptq_plan_add_layer(plan, &layer) != 0);
    volvoxai_ptq_plan_destroy(plan);
    volvoxai_engine_shutdown();
    remove(weights_path);
    remove(linear_path);
    remove(conv_path);
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope state_scope;
    CHECK(state != NULL);
    CHECK(vx_engine_state_init(state) == 0);
    state_scope = vx_engine_state_scope_enter(state);
    CHECK(test_qlinear_package() == 0);
    CHECK(test_qconv2d_package() == 0);
    CHECK(test_rejects_noncanonical_source_layouts() == 0);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(state_scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("native PTQ package tests passed");
    return 0;
}
