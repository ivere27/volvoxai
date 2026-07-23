#include "engine_core.h"
#include "runtime_state.h"
#include "volvoxai_backend.h"
#include "training/training_core.h"
#include "cuda_engine.h"
#include "safetensors.h"
#include "training_kernels.h"

#include <math.h>
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(condition) do { \
    if (!(condition)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
        return 1; \
    } \
} while (0)

static int closef32(float left, float right) {
    return fabsf(left - right) <=
        1.0e-6f * (1.0f + fabsf(left) + fabsf(right));
}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length = text ? strlen(text) : 0u;
    int result = !file || !text ? -1 : 0;
    if (file && length && fwrite(text, 1u, length, file) != length) result = -1;
    if (file && fclose(file) != 0) result = -1;
    return result;
}

static int8_t requantize_reference(int64_t accumulator, float multiplier,
                                   int32_t zero_point) {
    double transformed = (double)accumulator * (double)multiplier +
        (double)zero_point;
    long rounded = lrint(transformed);
    if (rounded < -128) rounded = -128;
    if (rounded > 127) rounded = 127;
    return (int8_t)rounded;
}

static int test_device_authoring_and_w8a8(void) {
    const float source[6] = {1.0f, 2.0f, -1.0f,
                             -2.0f, 0.5f, 3.0f};
    const float bias[2] = {1.0f, -2.0f};
    const int32_t shape[2] = {2, 3};
    const float input_scale = 0.5f;
    int8_t expected_weight[6] = {0};
    float expected_scales[2] = {0};
    int32_t expected_bias[2] = {0};
    uint64_t expected_saturation = 0;
    CHECK(volvoxai_ptq_pack_weight_i8(
              source, shape, 2, 0, expected_weight, expected_scales, 2,
              &expected_saturation) == 0);
    CHECK(volvoxai_ptq_pack_bias_i32(
              bias, 2, input_scale, expected_scales, 2, expected_bias) == 0);

    int8_t packed[6] = {0};
    float scales[2] = {-1.0f, -1.0f};
    int32_t zero_points[2] = {17, 17};
    int32_t packed_bias[2] = {17, 17};
    uint64_t saturation = UINT64_MAX;
    uint32_t status = UINT32_MAX;
    CHECK(cuda_training_quantize_w8_f32(
              source, packed, scales, zero_points, 2u, 3u, 0u, 0u, 0u,
              bias, input_scale, packed_bias, &saturation, &status));
    CHECK(status == 0u && saturation == expected_saturation);
    CHECK(!memcmp(packed, expected_weight, sizeof(packed)));
    CHECK(!memcmp(packed_bias, expected_bias, sizeof(packed_bias)));
    CHECK(zero_points[0] == 0 && zero_points[1] == 0);
    CHECK(closef32(scales[0], expected_scales[0]));
    CHECK(closef32(scales[1], expected_scales[1]));

    /* The freshly authored buffers must feed the existing physical W8A8
     * kernel directly, without a CPU repack or metadata conversion. */
    const int8_t input[3] = {2, 3, 4};
    int8_t output[2] = {0};
    int8_t expected_output[2] = {0};
    for (int row = 0; row < 2; row++) {
        int64_t accumulator = packed_bias[row];
        for (int column = 0; column < 3; column++)
            accumulator += (int64_t)input[column] *
                (int64_t)(packed[row * 3 + column] - zero_points[row]);
        expected_output[row] = requantize_reference(
            accumulator, input_scale * scales[row], 0);
    }
    uint64_t w8a8_launches_before = cuda_test_launch_count();
    CHECK(cuda_graph_qlinear_i8u8(
              input, packed, scales, zero_points, packed_bias, output,
              1u, 3u, 2u, input_scale, 0, 1.0f, 0,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
    CHECK(cuda_test_launch_count() > w8a8_launches_before);
    CHECK(!memcmp(output, expected_output, sizeof(output)));
    cuda_graph_reset();

    /* IN_OUT masters and preserved scales retain the portable training ABI. */
    const float transposed_source[6] = {
        0.25f, 3.0f, -1.0f, -4.0f, 2.0f, 0.5f
    };
    float preserved_scales[2] = {0.5f, 0.25f};
    float expected_preserved_scales[2] = {0.5f, 0.25f};
    int32_t preserved_zero_points[2] = {0, 0};
    int8_t preserved[6] = {0};
    int8_t expected_preserved[6] = {0};
    uint64_t preserved_saturation = 0;
    uint64_t expected_preserved_saturation = 0;
    CHECK(volvoxai_training_quantize_weight_f32_to_i8(
              transposed_source, expected_preserved,
              expected_preserved_scales, 2u, 3u, 1u,
              VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE,
              &expected_preserved_saturation));
    CHECK(cuda_training_quantize_w8_f32(
              transposed_source, preserved, preserved_scales,
              preserved_zero_points, 2u, 3u, 1u, 0u, 1u,
              NULL, 0.0f, NULL, &preserved_saturation, &status));
    CHECK(!memcmp(preserved, expected_preserved, sizeof(preserved)));
    CHECK(!memcmp(preserved_scales, expected_preserved_scales,
                  sizeof(preserved_scales)));
    CHECK(preserved_saturation == expected_preserved_saturation);

    /* Asymmetric authoring covers the full signed-I8 domain. */
    const float asymmetric_source[6] = {-1.0f, 0.0f, 3.0f,
                                         0.0f, 0.0f, 0.0f};
    const int8_t asymmetric_expected[6] = {-128, -64, 127, 0, 0, 0};
    int8_t asymmetric[6] = {0};
    float asymmetric_scales[2] = {0};
    int32_t asymmetric_zero_points[2] = {0};
    CHECK(cuda_training_quantize_w8_f32(
              asymmetric_source, asymmetric, asymmetric_scales,
              asymmetric_zero_points, 2u, 3u, 0u, 1u, 0u,
              NULL, 0.0f, NULL, NULL, &status));
    CHECK(!memcmp(asymmetric, asymmetric_expected, sizeof(asymmetric)));
    CHECK(closef32(asymmetric_scales[0], 4.0f / 255.0f));
    CHECK(asymmetric_scales[1] == 1.0f);
    CHECK(asymmetric_zero_points[0] == -64 &&
          asymmetric_zero_points[1] == 0);

    /* Device rejection is transactional for all public destinations. */
    float invalid_source[6];
    memcpy(invalid_source, source, sizeof(invalid_source));
    invalid_source[4] = NAN;
    int8_t invalid_output[6] = {11, 12, 13, 14, 15, 16};
    const int8_t invalid_output_before[6] = {11, 12, 13, 14, 15, 16};
    float invalid_scales[2] = {7.0f, 8.0f};
    const float invalid_scales_before[2] = {7.0f, 8.0f};
    int32_t invalid_zero_points[2] = {9, 10};
    const int32_t invalid_zero_points_before[2] = {9, 10};
    int32_t invalid_bias_output[2] = {11, 12};
    const int32_t invalid_bias_before[2] = {11, 12};
    saturation = 123u;
    CHECK(!cuda_training_quantize_w8_f32(
              invalid_source, invalid_output, invalid_scales,
              invalid_zero_points, 2u, 3u, 0u, 0u, 0u,
              bias, input_scale, invalid_bias_output, &saturation, &status));
    CHECK((status & 2u) != 0u);
    CHECK(!memcmp(invalid_output, invalid_output_before,
                  sizeof(invalid_output)));
    CHECK(!memcmp(invalid_scales, invalid_scales_before,
                  sizeof(invalid_scales)));
    CHECK(!memcmp(invalid_zero_points, invalid_zero_points_before,
                  sizeof(invalid_zero_points)));
    CHECK(!memcmp(invalid_bias_output, invalid_bias_before,
                  sizeof(invalid_bias_output)));
    CHECK(saturation == 123u);
    return 0;
}

static int test_public_materialization_uses_cuda(void) {
    const char* graph_path = "/tmp/volvox-cuda-ptq-source.json";
    const char* weights_path = "/tmp/volvox-cuda-ptq-source.safetensors";
    const char* template_path = "/tmp/volvox-cuda-ptq-template.json";
    const char* output_graph_path = "/tmp/volvox-cuda-ptq-output.graph.json";
    const char* output_weights_path =
        "/tmp/volvox-cuda-ptq-output.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"weight_layout\":\"OUT_IN\"}}],\"outputs\":[\"y\"]}";
    const char* quantized_template =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"opType\":\"QLinear\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight.i8\","
        "\"bias\":\"bias.i32\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"weight_layout\":\"OUT_IN\"}}],\"outputs\":[\"y\"]}";
    const int shape[2] = {2, 3};
    const int bias_shape[1] = {2};
    const float source[6] = {1.0f, 2.0f, -1.0f,
                             -2.0f, 0.5f, 3.0f};
    const float bias[2] = {1.0f, -2.0f};
    const float sample[3] = {2.0f, 3.0f, 4.0f};
    float trained_source[6] = {0};
    float trained_bias[2] = {0};
    int8_t expected[6] = {0};
    float expected_scales[2] = {0};
    int8_t materialized[6] = {0};
    float scales[2] = {0};
    remove(graph_path);
    remove(weights_path);
    remove(template_path);
    remove(output_graph_path);
    remove(output_weights_path);
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(write_text(template_path, quantized_template) == 0);
    SafetensorsFile weights;
    CHECK(safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&weights, "weight", SAFETENSORS_DTYPE_F32,
                                 shape, 2, source, sizeof(source)) == 0);
    CHECK(safetensors_add_tensor(&weights, "bias", SAFETENSORS_DTYPE_F32,
                                 bias_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &weights) == 0);
    safetensors_free(&weights);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);

    /* Exercise the literal post-training path: update the loaded F32 master
     * with the strict public CUDA TrainStep, then feed that updated master to
     * public CUDA W8 materialization and package writing. */
    const int target = 1;
    const char* trainables[2] = {"weight", "bias"};
    volvoxai_cross_entropy_loss_t loss = {
        "cuda-ptq-train", "y", &target, 1, INT_MIN, -1, 1.0f, 1.0f,
    };
    volvoxai_cross_entropy_metric_t metric = {0};
    float training_loss = 0.0f;
    int microbatches = 0;
    int update_applied = 0;
    CHECK(volvoxai_engine_require_training_backend(
              VOLVOXAI_TRAINING_BACKEND_CUDA) == 0);
    CHECK(volvoxai_engine_set_input_f32("x", sample, 3) == 0);
    uint64_t training_launches_before = cuda_test_launch_count();
    CHECK(volvoxai_engine_train_step_multi(
              &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
              0.01f, 0.9f, 0.99f, 1.0e-8f, 0.01f, 0.0f, 1,
              1, 0, 0, &training_loss, &metric,
              &microbatches, &update_applied) == 0);
    CHECK(volvoxai_engine_last_training_backend() == VOLVOXAI_BACKEND_CUDA);
    CHECK(cuda_test_launch_count() > training_launches_before);
    CHECK(update_applied == 1 && microbatches == 1 && metric.examples == 1);
    CHECK(isfinite(training_loss) && training_loss > 0.0f);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "weight", trained_source, sizeof(trained_source)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "bias", trained_bias, sizeof(trained_bias)) == 0);
    CHECK(memcmp(trained_source, source, sizeof(source)) != 0);
    CHECK(memcmp(trained_bias, bias, sizeof(bias)) != 0);
    CHECK(volvoxai_ptq_pack_weight_i8(
              trained_source, shape, 2, 0, expected, expected_scales, 2,
              NULL) == 0);

    /* Package authoring deliberately requires the caller's source file and
     * the loaded model to be byte-identical. Publish the just-trained public
     * model mirror as that source file before asking PTQ to author W8. */
    CHECK(safetensors_init_empty(&weights,
                                 SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&weights, "weight", SAFETENSORS_DTYPE_F32,
                                 shape, 2, trained_source,
                                 sizeof(trained_source)) == 0);
    CHECK(safetensors_add_tensor(&weights, "bias", SAFETENSORS_DTYPE_F32,
                                 bias_shape, 1, trained_bias,
                                 sizeof(trained_bias)) == 0);
    CHECK(safetensors_save(weights_path, &weights) == 0);
    safetensors_free(&weights);

    uint64_t launches_before = cuda_test_launch_count();
    int scale_count = 0;
    int8_t zero_points[2] = {1, 1};
    CHECK(volvoxai_engine_ptq_materialize_weight_i8(
              "weight", "weight.i8", "weight.scale", "weight.zero_point",
              0, &scale_count) == 0);
    CHECK(cuda_test_launch_count() > launches_before);
    CHECK(scale_count == 2);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "weight.i8", materialized, sizeof(materialized)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "weight.scale", scales, sizeof(scales)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "weight.zero_point", zero_points, sizeof(zero_points)) == 0);
    CHECK(!memcmp(materialized, expected, sizeof(materialized)));
    CHECK(closef32(scales[0], expected_scales[0]));
    CHECK(closef32(scales[1], expected_scales[1]));
    CHECK(zero_points[0] == 0 && zero_points[1] == 0);
    CHECK(volvoxai_engine_remove_model_tensor("weight.zero_point") == 0);
    CHECK(volvoxai_engine_remove_model_tensor("weight.scale") == 0);
    CHECK(volvoxai_engine_remove_model_tensor("weight.i8") == 0);

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
    layer.source_weight_name = "weight";
    layer.packed_weight_name = "weight.i8";
    layer.source_bias_name = "bias";
    layer.packed_bias_name = "bias.i32";
    CHECK(volvoxai_ptq_plan_add_layer(plan, &layer) == 0);
    volvoxai_ptq_input_binding_t binding = VOLVOXAI_PTQ_INPUT_BINDING_INIT;
    binding.tensor_name = "x";
    binding.data = sample;
    binding.nbytes = sizeof(sample);
    CHECK(volvoxai_engine_ptq_plan_calibrate_sample(
              plan, "cuda-sample", &binding, 1) == 0);
    volvoxai_ptq_params_t input_params;
    CHECK(volvoxai_ptq_plan_tensor_params(plan, "x", &input_params) == 0);
    volvoxai_ptq_package_options_t package = VOLVOXAI_PTQ_PACKAGE_OPTIONS_INIT;
    package.template_graph_path = template_path;
    package.source_weights_path = weights_path;
    package.output_graph_path = output_graph_path;
    package.output_weights_path = output_weights_path;
    launches_before = cuda_test_launch_count();
    CHECK(volvoxai_ptq_plan_write_package(plan, &package) == 0);
    CHECK(cuda_test_launch_count() > launches_before);
    volvoxai_ptq_plan_destroy(plan);
    volvoxai_engine_shutdown();

    CHECK(safetensors_load(output_weights_path, &weights) == 0);
    const SafetensorsTensor* authored_weight =
        safetensors_find_tensor(&weights, "weight.i8");
    const SafetensorsTensor* authored_bias =
        safetensors_find_tensor(&weights, "bias.i32");
    int32_t expected_bias[2] = {0};
    CHECK(volvoxai_ptq_pack_bias_i32(
              trained_bias, 2, input_params.scale, expected_scales, 2,
              expected_bias) == 0);
    CHECK(authored_weight && authored_weight->dtype == SAFETENSORS_DTYPE_I8 &&
          authored_weight->nbytes == sizeof(expected) &&
          !memcmp(authored_weight->data, expected, sizeof(expected)));
    CHECK(authored_bias && authored_bias->dtype == SAFETENSORS_DTYPE_I32 &&
          authored_bias->nbytes == sizeof(expected_bias) &&
          !memcmp(authored_bias->data, expected_bias,
                  sizeof(expected_bias)));
    safetensors_free(&weights);
    remove(graph_path);
    remove(weights_path);
    remove(template_path);
    remove(output_graph_path);
    remove(output_weights_path);
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    if (volvoxai_engine_configure_backend("cuda") != 0) {
        if (cuda_init_failure_is_unavailable()) {
            vx_engine_state_scope_leave(scope);
            vx_engine_state_deinit(state);
            free(state);
            puts("native CUDA PTQ tests skipped: no CUDA device/driver");
            return 77;
        }
        fprintf(stderr, "CUDA PTQ initialization failed\n");
        return 1;
    }
    CHECK(cuda_training_quantize_w8_available());
    CHECK(test_device_authoring_and_w8a8() == 0);
    CHECK(test_public_materialization_uses_cuda() == 0);
    CHECK(volvoxai_engine_configure_backend("cpu") == 0);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("native CUDA PTQ tests passed");
    return 0;
}
