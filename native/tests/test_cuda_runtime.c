#include "cuda_engine.h"
#include "adapter_runtime_internal.h"
#include "engine_internal.h"
#include "engine_core.h"
#include "runtime_state.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        volvoxai_engine_shutdown(); \
        return 1; \
    } \
} while (0)

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length;
    int ok;
    if (!file) return -1;
    length = strlen(text);
    ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok ? 0 : -1;
}

static int close_array(const float* actual, const float* expected, int count) {
    for (int index = 0; index < count; index++) {
        if (fabsf(actual[index] - expected[index]) > 1.0e-6f) {
            fprintf(stderr, "mismatch[%d]: got %.9g expected %.9g\n",
                    index, actual[index], expected[index]);
            return 0;
        }
    }
    return 1;
}

static int cuda_runtime_expect_contract_failure(
        const VolvoxAIEngineOptions* options, const char* path,
        const char* graph, const char* parameter_name,
        const float* parameter, size_t parameter_bytes,
        const float* input, size_t input_bytes) {
    uint64_t launches;
    uint64_t uploads;
    uint64_t downloads;
    CHECK(options && path && graph && parameter_name && parameter && input);
    CHECK(write_text(path, graph) == 0);
    CHECK(volvoxai_engine_configure(options) == 0);
    CHECK(volvoxai_engine_init(path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "x", VOLVOXAI_DTYPE_F32, input, input_bytes) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              parameter_name, VOLVOXAI_DTYPE_F32,
              parameter, parameter_bytes) == 0);
    launches = cuda_test_launch_count();
    uploads = cuda_test_host_to_device_count();
    downloads = cuda_test_device_to_host_count();
    CHECK(volvoxai_engine_forward() != 0);
    CHECK(cuda_test_launch_count() == launches);
    CHECK(cuda_test_host_to_device_count() == uploads);
    CHECK(cuda_test_device_to_host_count() == downloads);
    volvoxai_engine_shutdown();
    return 0;
}

static int test_cuda_optional_forward_contracts(
        const VolvoxAIEngineOptions* cuda_options) {
    const char* path = "/tmp/volvoxai-cuda-runtime-optional-contracts.json";
    const char* correct_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,4],\"dtype\":\"float32\"},"
        "\"group_scale\":{\"shape\":[4],\"dtype\":\"float32\"},"
        "\"layer_weight\":{\"shape\":[4],\"dtype\":\"float32\"},"
        "\"slope\":{\"shape\":[4],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"GroupNorm\",\"inputs\":{\"input\":\"x\","
        "\"scale\":\"group_scale\"},\"outputs\":{\"out\":\"group\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,4]},"
        "\"params\":{\"num_groups\":2,\"eps\":0.0001}},"
        "{\"opType\":\"LayerNorm\",\"inputs\":{\"input\":\"group\","
        "\"weight\":\"layer_weight\"},\"outputs\":{\"out\":\"layer\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,4]},"
        "\"params\":{\"eps\":0.00001}},"
        "{\"opType\":\"PReLU\",\"inputs\":{\"input\":\"layer\","
        "\"slope\":\"slope\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,4]}}],"
        "\"outputs\":[\"y\"]}";
    const char* bad_prelu =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,4],\"dtype\":\"float32\"},"
        "\"slope\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"PReLU\",\"inputs\":{"
        "\"input\":\"x\",\"slope\":\"slope\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,4]}}],\"outputs\":[\"y\"]}";
    const char* bad_layernorm =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,4],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"LayerNorm\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,4]}}],\"outputs\":[\"y\"]}";
    const char* bad_groupnorm =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,4],"
        "\"dtype\":\"float32\"},"
        "\"scale\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"GroupNorm\",\"inputs\":{"
        "\"input\":\"x\",\"scale\":\"scale\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,4]},"
        "\"params\":{\"num_groups\":2}}],\"outputs\":[\"y\"]}";
    const float input[4] = {-1.0f, 0.5f, 2.0f, -0.25f};
    const float group_scale[4] = {0.8f, 1.1f, 0.9f, 1.2f};
    const float layer_weight[4] = {1.0f, 0.7f, 1.3f, 0.6f};
    const float slope[4] = {0.1f, 0.2f, 0.3f, 0.4f};
    const float bad_two[2] = {0.1f, 0.2f};
    const float bad_three[3] = {1.0f, 0.7f, 1.3f};
    float cpu_output[4] = {0};
    float cuda_output[4] = {0};
    uint64_t launches;
    uint64_t uploads;
    uint64_t downloads;
    VolvoxAIEngineOptions cpu_options = {
        .backend = VOLVOXAI_BACKEND_CPU,
        .debug = 0,
        .cpu_threads = 1,
    };

    CHECK(cuda_options != NULL);
    CHECK(write_text(path, correct_graph) == 0);
    CHECK(volvoxai_engine_configure(&cpu_options) == 0);
    CHECK(volvoxai_engine_init(path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "x", VOLVOXAI_DTYPE_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "group_scale", VOLVOXAI_DTYPE_F32,
              group_scale, sizeof(group_scale)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "layer_weight", VOLVOXAI_DTYPE_F32,
              layer_weight, sizeof(layer_weight)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "slope", VOLVOXAI_DTYPE_F32, slope, sizeof(slope)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "y", cpu_output, 4) == 0);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_configure(cuda_options) == 0);
    CHECK(volvoxai_engine_init(path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "x", VOLVOXAI_DTYPE_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "group_scale", VOLVOXAI_DTYPE_F32,
              group_scale, sizeof(group_scale)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "layer_weight", VOLVOXAI_DTYPE_F32,
              layer_weight, sizeof(layer_weight)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "slope", VOLVOXAI_DTYPE_F32, slope, sizeof(slope)) == 0);
    launches = cuda_test_launch_count();
    uploads = cuda_test_host_to_device_count();
    downloads = cuda_test_device_to_host_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launches == 3u);
    CHECK(cuda_test_host_to_device_count() - uploads == 4u);
    CHECK(cuda_test_device_to_host_count() == downloads);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "y", cuda_output, 4) == 0);
    for (int index = 0; index < 4; index++)
        CHECK(fabsf(cuda_output[index] - cpu_output[index]) <= 2.0e-5f);
    volvoxai_engine_shutdown();

    CHECK(cuda_runtime_expect_contract_failure(
        cuda_options, path, bad_prelu, "slope", bad_two, sizeof(bad_two),
        input, sizeof(input)) == 0);
    CHECK(cuda_runtime_expect_contract_failure(
        cuda_options, path, bad_layernorm, "weight", bad_three,
        sizeof(bad_three), input, sizeof(input)) == 0);
    CHECK(cuda_runtime_expect_contract_failure(
        cuda_options, path, bad_groupnorm, "scale", bad_three,
        sizeof(bad_three), input, sizeof(input)) == 0);
    remove(path);
    return 0;
}

static int test_cuda_split_declared_output_order(
        const VolvoxAIEngineOptions* cuda_options) {
    const char* path = "/tmp/volvoxai-cuda-split-declared-order.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"f\":{\"shape\":[12],\"dtype\":\"float32\"},"
        "\"i\":{\"shape\":[12],\"dtype\":\"int32\"}},\"nodes\":["
        "{\"opType\":\"Split\",\"inputs\":{\"input\":\"f\"},"
        "\"outputs\":{\"out0\":\"f0\",\"out1\":\"f1\","
        "\"out2\":\"f2\",\"out3\":\"f3\",\"out4\":\"f4\","
        "\"out5\":\"f5\",\"out6\":\"f6\",\"out7\":\"f7\","
        "\"out8\":\"f8\",\"out9\":\"f9\",\"out10\":\"f10\","
        "\"out11\":\"f11\"},"
        "\"outputs_shape\":{\"out0\":[1],\"out1\":[1],"
        "\"out2\":[1],\"out3\":[1],\"out4\":[1],\"out5\":[1],"
        "\"out6\":[1],\"out7\":[1],\"out8\":[1],\"out9\":[1],"
        "\"out10\":[1],\"out11\":[1]},\"params\":{\"axis\":0}},"
        "{\"opType\":\"Split\",\"inputs\":{\"input\":\"i\"},"
        "\"outputs\":{\"out0\":\"i0\",\"out1\":\"i1\","
        "\"out2\":\"i2\",\"out3\":\"i3\",\"out4\":\"i4\","
        "\"out5\":\"i5\",\"out6\":\"i6\",\"out7\":\"i7\","
        "\"out8\":\"i8\",\"out9\":\"i9\",\"out10\":\"i10\","
        "\"out11\":\"i11\"},"
        "\"outputs_shape\":{\"out0\":[1],\"out1\":[1],"
        "\"out2\":[1],\"out3\":[1],\"out4\":[1],\"out5\":[1],"
        "\"out6\":[1],\"out7\":[1],\"out8\":[1],\"out9\":[1],"
        "\"out10\":[1],\"out11\":[1]},"
        "\"outputs_dtype\":{\"out0\":\"int32\",\"out1\":\"int32\","
        "\"out2\":\"int32\",\"out3\":\"int32\","
        "\"out4\":\"int32\",\"out5\":\"int32\","
        "\"out6\":\"int32\",\"out7\":\"int32\","
        "\"out8\":\"int32\",\"out9\":\"int32\","
        "\"out10\":\"int32\",\"out11\":\"int32\"},"
        "\"params\":{\"axis\":0}}],"
        "\"outputs\":[\"f0\",\"f1\",\"f2\",\"f3\",\"f4\",\"f5\","
        "\"f6\",\"f7\",\"f8\",\"f9\",\"f10\",\"f11\","
        "\"i0\",\"i1\",\"i2\",\"i3\",\"i4\",\"i5\","
        "\"i6\",\"i7\",\"i8\",\"i9\",\"i10\",\"i11\"]}";
    const float float_input[12] = {
        10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21,
    };
    const int32_t int_input[12] = {
        100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111,
    };
    uint64_t launches;

    CHECK(cuda_options != NULL);
    CHECK(write_text(path, graph) == 0);
    CHECK(volvoxai_engine_configure(cuda_options) == 0);
    CHECK(volvoxai_engine_init(path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "f", VOLVOXAI_DTYPE_F32,
              float_input, sizeof(float_input)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "i", VOLVOXAI_DTYPE_I32,
              int_input, sizeof(int_input)) == 0);
    launches = cuda_test_launch_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launches == 24u);
    for (int index = 0; index < 12; index++) {
        char name[16];
        float float_output = 0.0f;
        int32_t int_output = 0;
        snprintf(name, sizeof(name), "f%d", index);
        CHECK(volvoxai_engine_copy_tensor_f32(
                  name, &float_output, 1) == 0);
        CHECK(float_output == float_input[index]);
        snprintf(name, sizeof(name), "i%d", index);
        CHECK(volvoxai_engine_copy_tensor_raw(
                  name, &int_output, sizeof(int_output)) == 0);
        CHECK(int_output == int_input[index]);
    }
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

static int test_cuda_required_onnx_graph_ops(
        const VolvoxAIEngineOptions* cuda_options) {
    const char* typed_path =
        "/tmp/volvoxai-cuda-runtime-required-typed.json";
    const char* batch_matmul_path =
        "/tmp/volvoxai-cuda-runtime-batch-matmul.json";
    const char* typed_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"a\":{\"shape\":[2,3],\"dtype\":\"int32\"},"
        "\"b\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Expand\",\"inputs\":{\"input\":\"b\"},"
        "\"outputs\":{\"out\":\"bcast\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Equal\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"equal\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"GreaterOrEqual\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"greater_equal\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Not\",\"inputs\":{\"input\":\"equal\"},"
        "\"outputs\":{\"out\":\"not_equal\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Where\",\"inputs\":{\"condition\":\"greater_equal\","
        "\"x\":\"a\",\"y\":\"bcast\"},"
        "\"outputs\":{\"out\":\"selected\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Clip\",\"inputs\":{\"input\":\"selected\"},"
        "\"outputs\":{\"out\":\"clipped\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"min\":2,\"max\":4}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"clipped\"},"
        "\"outputs\":{\"out\":\"casted\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"to\":\"float32\"}},"
        "{\"opType\":\"Slice\",\"inputs\":{\"input\":\"clipped\"},"
        "\"outputs\":{\"out\":\"sliced\"},"
        "\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"starts\":[1],\"ends\":[3],"
        "\"axes\":[1],\"steps\":[1]}},"
        "{\"opType\":\"Split\",\"inputs\":{\"input\":\"sliced\"},"
        "\"outputs\":{\"out0\":\"left\",\"out1\":\"right\"},"
        "\"outputs_shape\":{\"out0\":[2,1],\"out1\":[2,1]},"
        "\"outputs_dtype\":{\"out0\":\"int32\",\"out1\":\"int32\"},"
        "\"params\":{\"axis\":1}},"
        "{\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"right\",\"input1\":\"left\"},"
        "\"outputs\":{\"out\":\"concatenated\"},"
        "\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1}}],"
        "\"outputs\":[\"casted\",\"not_equal\",\"concatenated\"]}";
    const char* batch_matmul_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"a\":{\"shape\":[2,2,3],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[1,3,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"BatchMatMul\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[2,2,2]},"
        "\"outputs_dtype\":{\"out\":\"float32\"}}],"
        "\"outputs\":[\"y\"]}";
    const int32_t typed_a[6] = {1, 4, 3, 0, 5, 2};
    const int32_t typed_b[3] = {1, 3, 3};
    const float cast_expected[6] = {2, 4, 3, 2, 4, 3};
    const int32_t not_expected[6] = {0, 1, 0, 1, 1, 1};
    const int32_t concat_expected[4] = {3, 4, 3, 4};
    const float batch_a[12] = {
        1, 2, 3, 4, 5, 6,
        7, 8, 9, 10, 11, 12,
    };
    const float batch_b[6] = {1, 0, 0, 1, 1, 1};
    const float batch_expected[8] = {
        4, 5, 10, 11, 16, 17, 22, 23,
    };
    float cast_output[6] = {0};
    int32_t not_output[6] = {0};
    int32_t concat_output[4] = {0};
    float batch_output[8] = {0};
    uint64_t launches;
    CHECK(cuda_options != NULL);

    CHECK(write_text(typed_path, typed_graph) == 0);
    CHECK(volvoxai_engine_configure(cuda_options) == 0);
    CHECK(volvoxai_engine_init(typed_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "a", VOLVOXAI_DTYPE_I32,
              typed_a, sizeof(typed_a)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "b", VOLVOXAI_DTYPE_I32,
              typed_b, sizeof(typed_b)) == 0);
    launches = cuda_test_launch_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launches == 12u);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "casted", cast_output, 6) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "not_equal", not_output, sizeof(not_output)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "concatenated", concat_output, sizeof(concat_output)) == 0);
    CHECK(close_array(cast_output, cast_expected, 6));
    for (int index = 0; index < 6; index++)
        CHECK(not_output[index] == not_expected[index]);
    for (int index = 0; index < 4; index++)
        CHECK(concat_output[index] == concat_expected[index]);
    volvoxai_engine_shutdown();
    remove(typed_path);

    CHECK(write_text(batch_matmul_path, batch_matmul_graph) == 0);
    CHECK(volvoxai_engine_configure(cuda_options) == 0);
    CHECK(volvoxai_engine_init(batch_matmul_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "a", VOLVOXAI_DTYPE_F32,
              batch_a, sizeof(batch_a)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "b", VOLVOXAI_DTYPE_F32,
              batch_b, sizeof(batch_b)) == 0);
    launches = cuda_test_launch_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launches == 1u);
    CHECK(volvoxai_engine_copy_tensor_f32(
              "y", batch_output, 8) == 0);
    CHECK(close_array(batch_output, batch_expected, 8));
    volvoxai_engine_shutdown();
    remove(batch_matmul_path);
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
    const char* graph_path = "/tmp/volvoxai-cuda-runtime.json";
    const char* add3_path = "/tmp/volvoxai-cuda-runtime-add3.json";
    const char* add3_partial_path =
        "/tmp/volvoxai-cuda-runtime-add3-partial.json";
    const char* add3_public_path =
        "/tmp/volvoxai-cuda-runtime-add3-public.json";
    const char* conv_add_path = "/tmp/volvoxai-cuda-runtime-conv-add.json";
    const char* conv_add_public_path =
        "/tmp/volvoxai-cuda-runtime-conv-add-public.json";
    const char* conv_clip_add_path =
        "/tmp/volvoxai-cuda-runtime-conv-clip-add.json";
    const char* argmax_path = "/tmp/volvoxai-cuda-runtime-argmax.json";
    const char* out_in_path = "/tmp/volvoxai-cuda-runtime-out-in.json";
    const char* adapter_path = "/tmp/volvoxai-cuda-runtime-adapter.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,2,2],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[2,3],\"dtype\":\"float32\"},"
        "\"slope\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight\"},\"outputs\":{\"out\":\"linear\"},"
        "\"outputs_shape\":{\"out\":[1,2,3]}},"
        "{\"opType\":\"Reshape\",\"inputs\":{\"input\":\"linear\"},"
        "\"outputs\":{\"out\":\"reshaped\"},"
        "\"outputs_shape\":{\"out\":[1,2,3]}},"
        "{\"opType\":\"PReLU\",\"inputs\":{\"input\":\"reshaped\","
        "\"weight\":\"slope\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,2,3]}}],"
        "\"outputs\":[\"y\"]}";
    const char* add3_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"r0\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"r1\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Conv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight\"},\"outputs\":{\"out\":\"conv\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},\"params\":{"
        "\"weight_layout\":\"HWIO\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"groups\":1,\"relu\":0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"conv\","
        "\"b\":\"r0\"},\"outputs\":{\"out\":\"add0\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"relu\":0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"add0\","
        "\"b\":\"r1\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"relu\":2}}],\"outputs\":[\"y\"]}";
    const char* add3_partial_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"a\":{\"shape\":[1,2,2],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[1,2,2],\"dtype\":\"float32\"},"
        "\"c\":{\"shape\":[1,2,2],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"add0\"},"
        "\"outputs_shape\":{\"out\":[1,2,2]},"
        "\"params\":{\"relu\":0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"add0\","
        "\"b\":\"c\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,2,2]},"
        "\"params\":{\"relu\":0}}],\"outputs\":[\"y\"]}";
    const char* add3_public_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"a\":{\"shape\":[1],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[1],\"dtype\":\"float32\"},"
        "\"c\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"add0\"},"
        "\"outputs_shape\":{\"out\":[1]},\"params\":{\"relu\":0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"add0\",\"b\":\"c\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1]},\"params\":{\"relu\":0}}],"
        "\"outputs\":[\"add0\",\"y\"]}";
    const char* conv_add_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"residual\":{\"shape\":[1,1,1,1],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Conv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight\"},\"outputs\":{\"out\":\"conv\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},\"params\":{"
        "\"weight_layout\":\"HWIO\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"groups\":1,\"relu\":2}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"conv\","
        "\"b\":\"residual\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"relu\":0}}],\"outputs\":[\"y\"]}";
    const char* conv_clip_add_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"residual\":{\"shape\":[1,1,1,1],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Conv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight\"},\"outputs\":{\"out\":\"conv\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},\"params\":{"
        "\"weight_layout\":\"HWIO\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"groups\":1,\"relu\":0}},"
        "{\"opType\":\"Clip\",\"inputs\":{\"input\":\"conv\"},"
        "\"outputs\":{\"out\":\"clipped\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"min\":0.0,\"max\":6.0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"clipped\","
        "\"b\":\"residual\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"relu\":0}}],\"outputs\":[\"y\"]}";
    const char* conv_add_public_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"},"
        "\"residual\":{\"shape\":[1,1,1,1],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Conv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight\"},\"outputs\":{\"out\":\"conv\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},\"params\":{"
        "\"weight_layout\":\"HWIO\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"groups\":1,\"relu\":0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"conv\","
        "\"b\":\"residual\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"relu\":0}}],\"outputs\":[\"conv\",\"y\"]}";
    const char* argmax_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,2],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1,\"keepdims\":0,"
        "\"select_last_index\":0}},"
        "{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y_keep\"},"
        "\"outputs_shape\":{\"out\":[2,1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-2,\"keepdims\":1,"
        "\"select_last_index\":0}}],"
        "\"outputs\":[\"y\",\"y_keep\"]}";
    const char* out_in_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,2,3],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[2,3],\"dtype\":\"float32\"},"
        "\"bias\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,2,2]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}}],"
        "\"outputs\":[\"y\"]}";
    const char* adapter_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[2,1,2],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[2,3],\"dtype\":\"float32\"},"
        "\"bias\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[2,1,3]},"
        "\"params\":{\"weight_layout\":\"din_dout\"}}],"
        "\"outputs\":[\"y\"]}";
    const float x[4] = {1.0f, -2.0f, 3.0f, 4.0f};
    const float weight[6] = {1.0f, 2.0f, -1.0f,
                             0.5f, -1.0f, 3.0f};
    const float slope[1] = {0.25f};
    const float expected[6] = {0.0f, 4.0f, -1.75f, 5.0f, 2.0f, 9.0f};
    const float prefix_x[4] = {2.0f, 1.0f, -1.0f, 3.0f};
    const float replay_expected[6] = {2.5f, 3.0f, 1.0f,
                                      0.5f, -1.25f, 10.0f};
    const float prefix_expected[6] = {2.5f, 3.0f, 1.0f,
                                      0.5f, -1.25f, 10.0f};
    const float row_x[4] = {0.0f, 2.0f, 4.0f, -2.0f};
    const float row_expected[6] = {2.5f, 3.0f, 1.0f,
                                   3.0f, 10.0f, -2.5f};
    float output[6] = {0};
    uint64_t launch_count_before;
    uint64_t host_to_device_before;
    uint64_t device_to_host_before;
    uint64_t context_get_before;
    uint64_t context_set_before;
    uint64_t graph_capture_before;
    uint64_t graph_launch_before;
    uint64_t graph_replay_before;
    uint64_t graph_invalidation_before;
    uint64_t slot_exact_lookup_before;
    uint64_t slot_hash_probe_before;
    uint64_t slot_containing_scan_before;
    CudaGraphDynamicStateProbe dynamic_before = {0};
    CudaGraphDynamicStateProbe dynamic_after = {0};
    int graph_api;
    VolvoxAIEngineOptions options = {
        .backend = VOLVOXAI_BACKEND_CUDA,
        .debug = 0,
        .cpu_threads = 1,
    };

    if (volvoxai_engine_configure(&options) != 0) {
        if (cuda_init_failure_is_unavailable()) {
            puts("CUDA device unavailable; skipping CUDA runtime integration test");
            return 77;
        }
        fputs("CUDA device was found, but runtime selection failed\n", stderr);
        return 1;
    }
    CHECK(g_use_cuda == 1);
    graph_api = cuda_test_graph_api_available();
    CHECK(cuda_test_caller_context_is_clear());
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(g_nn == 3 && g_n[1].skip == 1);
    CHECK(t_find("linear") && t_find("reshaped") &&
          t_find("linear")->data == t_find("reshaped")->data);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                        weight, sizeof(weight)) == 0);
    CHECK(volvoxai_engine_set_input_raw("slope", VOLVOXAI_DTYPE_F32,
                                        slope, sizeof(slope)) == 0);
    launch_count_before = cuda_test_launch_count();
    host_to_device_before = cuda_test_host_to_device_count();
    device_to_host_before = cuda_test_device_to_host_count();
    context_get_before = cuda_test_context_get_current_count();
    context_set_before = cuda_test_context_set_current_count();
    graph_capture_before = cuda_test_graph_capture_count();
    graph_launch_before = cuda_test_graph_launch_count();
    graph_replay_before = cuda_test_graph_replay_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launch_count_before == 2u);
    CHECK(cuda_test_graph_capture_count() == graph_capture_before);
    CHECK(cuda_test_graph_launch_count() == graph_launch_before);
    CHECK(cuda_test_graph_replay_count() == graph_replay_before);
    /* The CUDA context is held once around the complete forward. Nested slot,
     * upload, and launch guards borrow that held context without Driver calls. */
    CHECK(cuda_test_context_get_current_count() - context_get_before == 1u);
    CHECK(cuda_test_context_set_current_count() - context_set_before == 2u);
    /* x, weight, and slope are uploaded exactly once.  The Linear output is
     * consumed by PReLU directly from its CUDA slot, with no intermediate
     * readback or re-upload. */
    CHECK(cuda_test_host_to_device_count() - host_to_device_before == 3u);
    CHECK(cuda_test_device_to_host_count() == device_to_host_before);
    CHECK(cuda_test_caller_context_is_clear());
    context_get_before = cuda_test_context_get_current_count();
    context_set_before = cuda_test_context_set_current_count();
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 6) == 0);
    /* Host synchronization occurs outside the forward and therefore retains
     * its own balanced context guard. */
    CHECK(cuda_test_context_get_current_count() - context_get_before == 1u);
    CHECK(cuda_test_context_set_current_count() - context_set_before == 2u);
    CHECK(cuda_test_device_to_host_count() - device_to_host_before == 1u);
    CHECK(close_array(output, expected, 6));

    /* The first successful static pass observes its read-before-write slots
     * and launch signature. The second pass pre-stages those inputs and, when
     * Driver Graph APIs are present, captures the same two launches. */
    launch_count_before = cuda_test_launch_count();
    host_to_device_before = cuda_test_host_to_device_count();
    device_to_host_before = cuda_test_device_to_host_count();
    graph_capture_before = cuda_test_graph_capture_count();
    graph_launch_before = cuda_test_graph_launch_count();
    graph_replay_before = cuda_test_graph_replay_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launch_count_before == 2u);
    CHECK(cuda_test_host_to_device_count() - host_to_device_before == 3u);
    CHECK(cuda_test_device_to_host_count() == device_to_host_before);
    CHECK(cuda_test_graph_capture_count() - graph_capture_before ==
          (uint64_t)(graph_api ? 1 : 0));
    CHECK(cuda_test_graph_launch_count() - graph_launch_before ==
          (uint64_t)(graph_api ? 1 : 0));
    CHECK(cuda_test_graph_replay_count() == graph_replay_before);
    int host_count = 0;
    const float* host_output = volvoxai_engine_tensor_row_f32("y", -1, &host_count);
    CHECK(host_output != NULL && host_count == 6);
    CHECK(cuda_test_device_to_host_count() - device_to_host_before == 1u);
    CHECK(close_array(host_output, expected, 6));

    /* Replay still traverses the complete node loop, including the skipped
     * host-pointer alias, but suppresses the two active Driver launches after
     * validating their signature. Changed inputs are staged before launch. */
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        prefix_x, sizeof(prefix_x)) == 0);
    launch_count_before = cuda_test_launch_count();
    host_to_device_before = cuda_test_host_to_device_count();
    device_to_host_before = cuda_test_device_to_host_count();
    graph_launch_before = cuda_test_graph_launch_count();
    graph_replay_before = cuda_test_graph_replay_count();
    slot_exact_lookup_before = cuda_test_slot_exact_lookup_count();
    slot_hash_probe_before = cuda_test_slot_hash_probe_count();
    slot_containing_scan_before = cuda_test_slot_containing_scan_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launch_count_before ==
          (uint64_t)(graph_api ? 0 : 2));
    CHECK(cuda_test_host_to_device_count() - host_to_device_before == 3u);
    CHECK(cuda_test_device_to_host_count() == device_to_host_before);
    CHECK(cuda_test_graph_launch_count() - graph_launch_before ==
          (uint64_t)(graph_api ? 1 : 0));
    CHECK(cuda_test_graph_replay_count() - graph_replay_before ==
          (uint64_t)(graph_api ? 1 : 0));
    CHECK(cuda_test_slot_exact_lookup_count() > slot_exact_lookup_before);
    CHECK(cuda_test_slot_hash_probe_count() > slot_hash_probe_before);
    CHECK(cuda_test_slot_containing_scan_count() ==
          slot_containing_scan_before);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 6) == 0);
    CHECK(cuda_test_device_to_host_count() - device_to_host_before == 1u);
    CHECK(close_array(output, replay_expected, 6));

    /* Prefix and explicit row entry points remain strict-CUDA operations while
     * touching only the requested resident rows. Unselected rows preserve the
     * result from the preceding pass. */
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        prefix_x, sizeof(prefix_x)) == 0);
    launch_count_before = cuda_test_launch_count();
    host_to_device_before = cuda_test_host_to_device_count();
    device_to_host_before = cuda_test_device_to_host_count();
    graph_launch_before = cuda_test_graph_launch_count();
    graph_replay_before = cuda_test_graph_replay_count();
    graph_invalidation_before = cuda_test_graph_invalidation_count();
    CHECK(volvoxai_engine_forward_prefix(1) == 0);
    CHECK(cuda_test_launch_count() - launch_count_before == 2u);
    /* A first prefix pass registers every owned base allocation. This test's
     * synthetic weight and slope are graph inputs, so all three inputs are
     * refreshed; packaged model weights are immutable non-owned storage. */
    CHECK(cuda_test_host_to_device_count() - host_to_device_before == 3u);
    CHECK(cuda_test_device_to_host_count() == device_to_host_before);
    CHECK(cuda_test_graph_launch_count() == graph_launch_before);
    CHECK(cuda_test_graph_replay_count() == graph_replay_before);
    CHECK(cuda_test_graph_invalidation_count() - graph_invalidation_before ==
          (uint64_t)(graph_api ? 1 : 0));
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 6) == 0);
    CHECK(cuda_test_device_to_host_count() - device_to_host_before == 1u);
    CHECK(close_array(output, prefix_expected, 6));

    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        row_x, sizeof(row_x)) == 0);
    launch_count_before = cuda_test_launch_count();
    host_to_device_before = cuda_test_host_to_device_count();
    device_to_host_before = cuda_test_device_to_host_count();
    graph_launch_before = cuda_test_graph_launch_count();
    graph_replay_before = cuda_test_graph_replay_count();
    CHECK(volvoxai_engine_forward_row(1) == 0);
    CHECK(cuda_test_launch_count() - launch_count_before == 2u);
    CHECK(cuda_test_host_to_device_count() - host_to_device_before == 1u);
    CHECK(cuda_test_device_to_host_count() == device_to_host_before);
    CHECK(cuda_test_graph_launch_count() == graph_launch_before);
    CHECK(cuda_test_graph_replay_count() == graph_replay_before);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 6) == 0);
    CHECK(cuda_test_device_to_host_count() - device_to_host_before == 1u);
    CHECK(close_array(output, row_expected, 6));

    /* Start a fresh ordinary-static session, build a ready replay plan, then
     * prove model shutdown destroys it before releasing CUDA slots/module. */
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_configure(&options) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                        weight, sizeof(weight)) == 0);
    CHECK(volvoxai_engine_set_input_raw("slope", VOLVOXAI_DTYPE_F32,
                                        slope, sizeof(slope)) == 0);
    graph_capture_before = cuda_test_graph_capture_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_graph_capture_count() - graph_capture_before ==
          (uint64_t)(graph_api ? 1 : 0));

    /* Exact semantic shape rebinding must invalidate a ready replay before
     * retiring transient slot identities. Rebinding the same exact key is a
     * no-op, and a rejected empty key cannot disturb the committed state. */
    CHECK(cuda_test_graph_dynamic_state("not-bound", &dynamic_before) == 0);
    graph_invalidation_before = cuda_test_graph_invalidation_count();
    CHECK(cuda_graph_bind_shape(
              "graph:x=f32[1,2,2];weight=f32[2,3];slope=f32[1]") == 0);
    CHECK(cuda_test_graph_dynamic_state(
              "graph:x=f32[1,2,2];weight=f32[2,3];slope=f32[1]",
              &dynamic_after) == 0);
    CHECK(dynamic_after.exact_signature_match &&
          dynamic_after.shape_generation > dynamic_before.shape_generation &&
          dynamic_after.capacity_generation >=
              dynamic_before.capacity_generation &&
          dynamic_after.replay_plan == 0 &&
          dynamic_after.replay_shape_generation == 0 &&
          dynamic_after.replay_capacity_generation == 0);
    CHECK(cuda_test_graph_invalidation_count() - graph_invalidation_before ==
          (uint64_t)(graph_api ? 1 : 0));
    dynamic_before = dynamic_after;
    graph_invalidation_before = cuda_test_graph_invalidation_count();
    CHECK(cuda_graph_bind_shape(
              "graph:x=f32[1,2,2];weight=f32[2,3];slope=f32[1]") == 0);
    CHECK(cuda_graph_bind_shape("") == -1);
    CHECK(cuda_test_graph_dynamic_state(
              "graph:x=f32[1,2,2];weight=f32[2,3];slope=f32[1]",
              &dynamic_after) == 0);
    CHECK(dynamic_after.shape_generation == dynamic_before.shape_generation &&
          dynamic_after.capacity_generation ==
              dynamic_before.capacity_generation &&
          cuda_test_graph_invalidation_count() == graph_invalidation_before);
    /* Rebuild an observed plan so shutdown still exercises plan destruction. */
    CHECK(volvoxai_engine_forward() == 0);
    graph_invalidation_before = cuda_test_graph_invalidation_count();
    volvoxai_engine_shutdown();
    CHECK(cuda_test_graph_invalidation_count() - graph_invalidation_before ==
          (uint64_t)(graph_api ? 1 : 0));

    /* Debug profiling is intentionally excluded because its per-node timing
     * and diagnostics require the ordinary launch/synchronization path. */
    options.debug = 1;
    CHECK(volvoxai_engine_configure(&options) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                        weight, sizeof(weight)) == 0);
    CHECK(volvoxai_engine_set_input_raw("slope", VOLVOXAI_DTYPE_F32,
                                        slope, sizeof(slope)) == 0);
    launch_count_before = cuda_test_launch_count();
    graph_capture_before = cuda_test_graph_capture_count();
    graph_launch_before = cuda_test_graph_launch_count();
    graph_replay_before = cuda_test_graph_replay_count();
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(cuda_test_launch_count() - launch_count_before == 4u);
    CHECK(cuda_test_graph_capture_count() == graph_capture_before);
    CHECK(cuda_test_graph_launch_count() == graph_launch_before);
    CHECK(cuda_test_graph_replay_count() == graph_replay_before);
    volvoxai_engine_shutdown();
    options.debug = 0;
    remove(graph_path);

    /* Tied token embeddings commonly expose an OUT_IN language-model head.
     * Explicit CUDA must execute that physical layout directly rather than
     * rejecting the dense node or silently transposing it on the CPU. */
    {
        const float out_in_input[6] = {
            1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f,
        };
        const float out_in_weight[6] = {
            1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f,
        };
        const float out_in_bias[2] = {0.5f, -0.5f};
        const float out_in_expected[4] = {22.5f, 27.5f, 11.0f, 11.5f};
        float out_in_output[4] = {0};
        CHECK(write_text(out_in_path, out_in_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(out_in_path, NULL) == 0);
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            out_in_input,
                                            sizeof(out_in_input)) == 0);
        CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                            out_in_weight,
                                            sizeof(out_in_weight)) == 0);
        CHECK(volvoxai_engine_set_input_raw("bias", VOLVOXAI_DTYPE_F32,
                                            out_in_bias,
                                            sizeof(out_in_bias)) == 0);
        launch_count_before = cuda_test_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 1u);
        CHECK(volvoxai_engine_copy_tensor_f32("y", out_in_output, 4) == 0);
        CHECK(close_array(out_in_output, out_in_expected, 4));
        volvoxai_engine_shutdown();
        remove(out_in_path);
    }

    /* Chained Add has priority over Conv+Add.  The Conv remains independent,
     * while Add0 owns the redirected Add1 output and executes one Add3+ReLU6
     * kernel.  This specifically guards against dropping r1 by subsequently
     * (and incorrectly) fusing Conv into the already-transformed Add0. */
    {
        const float add3_x = 2.0f;
        const float add3_replay_x = 1.0f;
        const float add3_weight = 3.0f;
        const float add3_r0 = -10.0f;
        const float add3_r1 = 12.0f;
        float add3_output = 0.0f;
        uint64_t add3_before;
        uint64_t conv_add_before;
        char saved_peer_source[128];
        int peer_source_index = -1;
        CHECK(write_text(add3_path, add3_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(add3_path, NULL) == 0);
        CHECK(g_nn == 3 && !g_n[0].skip && !g_n[1].skip && g_n[2].skip);
        CHECK(!strcmp(g_n[0].out, "conv") && !strcmp(g_n[1].out, "y"));
        CHECK(!strcmp(g_n[1].outs[0].name, "add0"));
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &add3_x, sizeof(add3_x)) == 0);
        CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                            &add3_weight,
                                            sizeof(add3_weight)) == 0);
        CHECK(volvoxai_engine_set_input_raw("r0", VOLVOXAI_DTYPE_F32,
                                            &add3_r0, sizeof(add3_r0)) == 0);
        CHECK(volvoxai_engine_set_input_raw("r1", VOLVOXAI_DTYPE_F32,
                                            &add3_r1, sizeof(add3_r1)) == 0);
        launch_count_before = cuda_test_launch_count();
        add3_before = cuda_test_add3_relu_launch_count();
        conv_add_before = cuda_test_conv2d_add_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 2u);
        CHECK(cuda_test_add3_relu_launch_count() - add3_before == 1u);
        CHECK(cuda_test_conv2d_add_launch_count() == conv_add_before);
        CHECK(volvoxai_engine_copy_tensor_f32("y", &add3_output, 1) == 0);
        CHECK(add3_output == 6.0f);

        graph_capture_before = cuda_test_graph_capture_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_graph_capture_count() - graph_capture_before ==
              (uint64_t)(graph_api ? 1 : 0));
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &add3_replay_x,
                                            sizeof(add3_replay_x)) == 0);
        launch_count_before = cuda_test_launch_count();
        graph_replay_before = cuda_test_graph_replay_count();
        add3_before = cuda_test_add3_relu_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before ==
              (uint64_t)(graph_api ? 0 : 2));
        CHECK(cuda_test_graph_replay_count() - graph_replay_before ==
              (uint64_t)(graph_api ? 1 : 0));
        CHECK(cuda_test_add3_relu_launch_count() - add3_before ==
              (uint64_t)(graph_api ? 0 : 1));
        CHECK(volvoxai_engine_copy_tensor_f32("y", &add3_output, 1) == 0);
        CHECK(add3_output == 5.0f);

        /* Corrupt the preserved pre-redirect output association.  The
         * transformed node must fail CUDA routing; it cannot fall through to
         * an ordinary binary Add because its peer is skipped. */
        for (int input_index = 0; input_index < g_n[2].nin; input_index++)
            if (!strcmp(g_n[2].ins[input_index].name, "add0"))
                peer_source_index = input_index;
        CHECK(peer_source_index >= 0);
        memcpy(saved_peer_source, g_n[2].ins[peer_source_index].name,
               sizeof(saved_peer_source));
        g_n[2].ins[peer_source_index].name[0] = 'z';
        add3_before = cuda_test_add3_relu_launch_count();
        CHECK(volvoxai_engine_forward() != 0);
        CHECK(cuda_test_add3_relu_launch_count() == add3_before);
        memcpy(g_n[2].ins[peer_source_index].name, saved_peer_source,
               sizeof(saved_peer_source));
        volvoxai_engine_shutdown();
        remove(add3_path);
    }

    /* Prefix and row execution pass interior tensor views to Add3.  Verify
     * the four-pointer offset path updates only the selected sequence rows. */
    {
        const float initial_a[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float initial_b[4] = {10.0f, 20.0f, 30.0f, 40.0f};
        const float initial_c[4] = {100.0f, 200.0f, 300.0f, 400.0f};
        const float initial_expected[4] = {111.0f, 222.0f, 333.0f, 444.0f};
        const float prefix_a[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
        const float prefix_b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float prefix_c[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        const float prefix_expected_add3[4] = {5.0f, 6.0f, 333.0f, 444.0f};
        const float row_a[4] = {10.0f, 20.0f, 30.0f, 40.0f};
        const float row_b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
        const float row_c[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
        const float row_expected_add3[4] = {5.0f, 6.0f, 30.0f, 40.0f};
        float partial_output[4] = {0};
        CHECK(write_text(add3_partial_path, add3_partial_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(add3_partial_path, NULL) == 0);
        CHECK(g_nn == 2 && !g_n[0].skip && g_n[1].skip);
        CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32,
                                            initial_a,
                                            sizeof(initial_a)) == 0);
        CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32,
                                            initial_b,
                                            sizeof(initial_b)) == 0);
        CHECK(volvoxai_engine_set_input_raw("c", VOLVOXAI_DTYPE_F32,
                                            initial_c,
                                            sizeof(initial_c)) == 0);
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("y", partial_output, 4) == 0);
        CHECK(close_array(partial_output, initial_expected, 4));

        CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32,
                                            prefix_a, sizeof(prefix_a)) == 0);
        CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32,
                                            prefix_b, sizeof(prefix_b)) == 0);
        CHECK(volvoxai_engine_set_input_raw("c", VOLVOXAI_DTYPE_F32,
                                            prefix_c, sizeof(prefix_c)) == 0);
        CHECK(volvoxai_engine_forward_prefix(1) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("y", partial_output, 4) == 0);
        CHECK(close_array(partial_output, prefix_expected_add3, 4));

        CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32,
                                            row_a, sizeof(row_a)) == 0);
        CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32,
                                            row_b, sizeof(row_b)) == 0);
        CHECK(volvoxai_engine_set_input_raw("c", VOLVOXAI_DTYPE_F32,
                                            row_c, sizeof(row_c)) == 0);
        CHECK(volvoxai_engine_forward_row(1) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("y", partial_output, 4) == 0);
        CHECK(close_array(partial_output, row_expected_add3, 4));
        volvoxai_engine_shutdown();
        remove(add3_partial_path);
    }

    /* A public intermediate is observable and therefore cannot be consumed
     * by a redirecting fusion, even when it has only one node consumer. */
    {
        const float public_a = 1.0f;
        const float public_b = 2.0f;
        const float public_c = 3.0f;
        float public_add0 = 0.0f;
        float public_y = 0.0f;
        uint64_t add3_before;
        CHECK(write_text(add3_public_path, add3_public_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(add3_public_path, NULL) == 0);
        CHECK(g_nn == 2 && !g_n[0].skip && !g_n[1].skip);
        CHECK(volvoxai_engine_set_input_raw("a", VOLVOXAI_DTYPE_F32,
                                            &public_a, sizeof(public_a)) == 0);
        CHECK(volvoxai_engine_set_input_raw("b", VOLVOXAI_DTYPE_F32,
                                            &public_b, sizeof(public_b)) == 0);
        CHECK(volvoxai_engine_set_input_raw("c", VOLVOXAI_DTYPE_F32,
                                            &public_c, sizeof(public_c)) == 0);
        launch_count_before = cuda_test_launch_count();
        add3_before = cuda_test_add3_relu_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 2u);
        CHECK(cuda_test_add3_relu_launch_count() == add3_before);
        CHECK(volvoxai_engine_copy_tensor_f32("add0", &public_add0, 1) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("y", &public_y, 1) == 0);
        CHECK(public_add0 == 3.0f && public_y == 6.0f);
        volvoxai_engine_shutdown();
        remove(add3_public_path);
    }

    /* Conv activation is semantically inside the Conv node, so fused Add is
     * applied after ReLU6: clamp(10, 0, 6) + -8 == -2. */
    {
        const float conv_add_x = 1.0f;
        const float conv_add_replay_x = -1.0f;
        const float conv_add_weight = 10.0f;
        const float conv_add_residual = -8.0f;
        float conv_add_output = 0.0f;
        uint64_t fused_before;
        uint64_t tiled_before;
        CHECK(write_text(conv_add_path, conv_add_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(conv_add_path, NULL) == 0);
        CHECK(g_nn == 2 && !g_n[0].skip && g_n[1].skip);
        CHECK(!strcmp(g_n[0].out, "y") &&
              !strcmp(g_n[0].outs[0].name, "conv"));
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &conv_add_x,
                                            sizeof(conv_add_x)) == 0);
        CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                            &conv_add_weight,
                                            sizeof(conv_add_weight)) == 0);
        CHECK(volvoxai_engine_set_input_raw("residual", VOLVOXAI_DTYPE_F32,
                                            &conv_add_residual,
                                            sizeof(conv_add_residual)) == 0);
        launch_count_before = cuda_test_launch_count();
        fused_before = cuda_test_conv2d_add_launch_count();
        tiled_before = cuda_test_conv2d_1x1_tiled_add_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 1u);
        CHECK(cuda_test_conv2d_add_launch_count() - fused_before == 1u);
        CHECK(cuda_test_conv2d_1x1_tiled_add_launch_count() - tiled_before ==
              1u);
        CHECK(volvoxai_engine_copy_tensor_f32("y", &conv_add_output, 1) == 0);
        CHECK(conv_add_output == -2.0f);

        graph_capture_before = cuda_test_graph_capture_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_graph_capture_count() - graph_capture_before ==
              (uint64_t)(graph_api ? 1 : 0));
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &conv_add_replay_x,
                                            sizeof(conv_add_replay_x)) == 0);
        launch_count_before = cuda_test_launch_count();
        graph_replay_before = cuda_test_graph_replay_count();
        fused_before = cuda_test_conv2d_add_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before ==
              (uint64_t)(graph_api ? 0 : 1));
        CHECK(cuda_test_graph_replay_count() - graph_replay_before ==
              (uint64_t)(graph_api ? 1 : 0));
        CHECK(cuda_test_conv2d_add_launch_count() - fused_before ==
              (uint64_t)(graph_api ? 0 : 1));
        CHECK(volvoxai_engine_copy_tensor_f32("y", &conv_add_output, 1) == 0);
        CHECK(conv_add_output == -8.0f);
        volvoxai_engine_shutdown();
        remove(conv_add_path);
    }

    /* ReLU6 graph folding redirects Conv to the Clip output before Conv+Add
     * is considered.  Fusion metadata preserves that immediate redirected
     * edge, rather than assuming the node's original output descriptor. */
    {
        const float clip_x = 1.0f;
        const float clip_weight = 10.0f;
        const float clip_residual = -8.0f;
        float clip_output = 0.0f;
        uint64_t fused_before;
        CHECK(write_text(conv_clip_add_path, conv_clip_add_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(conv_clip_add_path, NULL) == 0);
        CHECK(g_nn == 3 && !g_n[0].skip && g_n[1].skip && g_n[2].skip);
        CHECK(g_n[0].fuse_relu6 && !strcmp(g_n[0].out, "y"));
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &clip_x, sizeof(clip_x)) == 0);
        CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                            &clip_weight,
                                            sizeof(clip_weight)) == 0);
        CHECK(volvoxai_engine_set_input_raw("residual", VOLVOXAI_DTYPE_F32,
                                            &clip_residual,
                                            sizeof(clip_residual)) == 0);
        fused_before = cuda_test_conv2d_add_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_conv2d_add_launch_count() - fused_before == 1u);
        CHECK(volvoxai_engine_copy_tensor_f32("y", &clip_output, 1) == 0);
        CHECK(clip_output == -2.0f);
        volvoxai_engine_shutdown();
        remove(conv_clip_add_path);
    }

    {
        const float public_x = 1.0f;
        const float public_weight = 10.0f;
        const float public_residual = -8.0f;
        float public_conv = 0.0f;
        float public_y = 0.0f;
        uint64_t fused_before;
        CHECK(write_text(conv_add_public_path, conv_add_public_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(conv_add_public_path, NULL) == 0);
        CHECK(g_nn == 2 && !g_n[0].skip && !g_n[1].skip);
        CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                            &public_x, sizeof(public_x)) == 0);
        CHECK(volvoxai_engine_set_input_raw("weight", VOLVOXAI_DTYPE_F32,
                                            &public_weight,
                                            sizeof(public_weight)) == 0);
        CHECK(volvoxai_engine_set_input_raw("residual", VOLVOXAI_DTYPE_F32,
                                            &public_residual,
                                            sizeof(public_residual)) == 0);
        launch_count_before = cuda_test_launch_count();
        fused_before = cuda_test_conv2d_add_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 2u);
        CHECK(cuda_test_conv2d_add_launch_count() == fused_before);
        CHECK(volvoxai_engine_copy_tensor_f32("conv", &public_conv, 1) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32("y", &public_y, 1) == 0);
        CHECK(public_conv == 10.0f && public_y == 2.0f);
        volvoxai_engine_shutdown();
        remove(conv_add_public_path);
    }

    /* Runtime LoRA stays strict-CUDA, including one adapter route per batch
     * member. Bias is applied by the base Linear exactly once, while A/B and
     * both low-rank stages remain device resident. */
    {
        const float adapter_x[4] = {2.0f, 3.0f, 4.0f, -1.0f};
        const float adapter_weight[6] = {
            1.0f, 2.0f, 0.0f,
            -1.0f, 0.5f, 3.0f,
        };
        const float adapter_bias[3] = {0.25f, -0.5f, 1.0f};
        const float route0_a[2] = {1.0f, 0.0f};
        const float route0_b[3] = {1.0f, 2.0f, -1.0f};
        const float route1_a[2] = {0.0f, 1.0f};
        const float route1_b[3] = {-1.0f, 0.5f, 2.0f};
        const float adapter_expected[6] = {
            1.25f, 9.0f, 8.0f,
            5.75f, 6.75f, -3.0f,
        };
        const char* routes[2] = {"cuda-route-0", "cuda-route-1"};
        const float route_scales[2] = {2.0f, 0.25f};
        float adapter_output[6] = {0};
        VxAdapterTargetSpec targets[2];
        VxAdapterVersionSpec versions[2];
        memset(targets, 0, sizeof(targets));
        memset(versions, 0, sizeof(versions));
        targets[0].weight_name = "weight";
        targets[0].kind = VX_ADAPTER_LORA;
        targets[0].d_in = 2;
        targets[0].d_out = 3;
        targets[0].rank = 1;
        targets[0].alpha = 1.0f;
        targets[0].scale = 0.5f;
        targets[0].a = (VxAdapterTensorSpec){
            route0_a, sizeof(route0_a), VX_ADAPTER_DTYPE_F32, 2, 1,
        };
        targets[0].b = (VxAdapterTensorSpec){
            route0_b, sizeof(route0_b), VX_ADAPTER_DTYPE_F32, 1, 3,
        };
        targets[1] = targets[0];
        targets[1].scale = 2.0f;
        targets[1].a = (VxAdapterTensorSpec){
            route1_a, sizeof(route1_a), VX_ADAPTER_DTYPE_F32, 2, 1,
        };
        targets[1].b = (VxAdapterTensorSpec){
            route1_b, sizeof(route1_b), VX_ADAPTER_DTYPE_F32, 1, 3,
        };
        for (int route = 0; route < 2; route++) {
            versions[route] = (VxAdapterVersionSpec){
                routes[route], routes[route], &targets[route], 1, NULL,
            };
        }
        CHECK(write_text(adapter_path, adapter_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(adapter_path, NULL) == 0);
        CHECK(vx_adapter_stage(&versions[0]) == 0);
        CHECK(vx_adapter_stage(&versions[1]) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "x", VOLVOXAI_DTYPE_F32, adapter_x,
                  sizeof(adapter_x)) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "weight", VOLVOXAI_DTYPE_F32, adapter_weight,
                  sizeof(adapter_weight)) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "bias", VOLVOXAI_DTYPE_F32, adapter_bias,
                  sizeof(adapter_bias)) == 0);
        CHECK(volvoxai_engine_adapter_route_begin_many(
                  routes, route_scales, 2) == 0);
        launch_count_before = cuda_test_launch_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 5u);
        volvoxai_engine_adapter_route_end();
        CHECK(volvoxai_engine_copy_tensor_f32(
                  "y", adapter_output, 6) == 0);
        CHECK(close_array(adapter_output, adapter_expected, 6));
        volvoxai_engine_shutdown();
        remove(adapter_path);
    }

    CHECK(test_cuda_optional_forward_contracts(&options) == 0);
    CHECK(test_cuda_split_declared_output_order(&options) == 0);
    CHECK(test_cuda_required_onnx_graph_ops(&options) == 0);

    /* CUDA runs general-axis F32 ArgMax directly, validates both keepdims
     * layouts, normalizes negative axes, and preserves the first tie. */
    {
        const float argmax_input[12] = {
             1.0f, 9.0f,  5.0f, 9.0f,  5.0f, 3.0f,
            -1.0f, 2.0f, -3.0f, 7.0f, -1.0f, 7.0f,
        };
        const int32_t argmax_expected[4] = {1, 0, 0, 1};
        int32_t argmax_output[4] = {0};
        int32_t argmax_keep_output[4] = {0};
        CHECK(write_text(argmax_path, argmax_graph) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(argmax_path, NULL) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "x", VOLVOXAI_DTYPE_F32,
                  argmax_input, sizeof(argmax_input)) == 0);
        launch_count_before = cuda_test_launch_count();
        host_to_device_before = cuda_test_host_to_device_count();
        device_to_host_before = cuda_test_device_to_host_count();
        context_get_before = cuda_test_context_get_current_count();
        context_set_before = cuda_test_context_set_current_count();
        CHECK(volvoxai_engine_forward() == 0);
        CHECK(cuda_test_launch_count() - launch_count_before == 2u);
        CHECK(cuda_test_host_to_device_count() -
              host_to_device_before == 1u);
        CHECK(cuda_test_device_to_host_count() ==
              device_to_host_before);
        CHECK(cuda_test_context_get_current_count() -
              context_get_before == 1u);
        CHECK(cuda_test_context_set_current_count() -
              context_set_before == 2u);
        CHECK(cuda_test_caller_context_is_clear());
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "y", argmax_output, sizeof(argmax_output)) == 0);
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "y_keep", argmax_keep_output,
                  sizeof(argmax_keep_output)) == 0);
        CHECK(cuda_test_device_to_host_count() -
              device_to_host_before == 2u);
        CHECK(memcmp(
                  argmax_output, argmax_expected,
                  sizeof(argmax_output)) == 0);
        CHECK(memcmp(
                  argmax_keep_output, argmax_expected,
                  sizeof(argmax_keep_output)) == 0);
        volvoxai_engine_shutdown();
    }
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    remove(argmax_path);
    puts("CUDA strict runtime routing/residency test passed");
    return 0;
}
