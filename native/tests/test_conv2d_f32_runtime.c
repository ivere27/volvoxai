#include "engine_internal.h"
#include "conv_f32_opt.h"
#include "safetensors.h"
#include "engine_core.h"
#include "runtime_state.h"
#include "thread_pool.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

/* CPU-only test composition intentionally omits the shader store. */
void volvoxai_shader_store_shutdown(void) {}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok ? 0 : -1;
}

static int close_array(const float* actual, const float* expected,
                       size_t count) {
    for (size_t index = 0; index < count; index++) {
        float tolerance = 1.0e-6f * fmaxf(1.0f, fabsf(expected[index]));
        if (fabsf(actual[index] - expected[index]) > tolerance) {
            fprintf(stderr, "mismatch[%zu]: got %.9g expected %.9g\n",
                    index, actual[index], expected[index]);
            return 0;
        }
    }
    return 1;
}

static int run_conv_case(const char* graph_path, const char* weights_path,
                         const char* graph, const int* weight_shape,
                         const float* weight, size_t weight_count,
                         const float* bias, size_t bias_count,
                         const float* input, size_t input_count,
                         const float* expected, size_t output_count) {
    const int bias_shape[1] = {(int)bias_count};
    float output[8] = {0};
    SafetensorsFile file;

    CHECK(output_count <= sizeof(output) / sizeof(output[0]));
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_F32,
                                 weight_shape, 4, weight,
                                 weight_count * sizeof(float)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32,
                                 bias_shape, 1, bias,
                                 bias_count * sizeof(float)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, input,
                                        input_count * sizeof(float)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output,
                                          (long)output_count) == 0);
    CHECK(close_array(output, expected, output_count));
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_default_hwio_conv(void) {
    const char* graph_path = "/tmp/volvox-conv2d-f32-hwio.json";
    const char* weights_path = "/tmp/volvox-conv2d-f32-hwio.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1,1,2,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,2,3]},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const int weight_shape[4] = {1, 1, 2, 3};
    const float weight[6] = {1.0f, 2.0f, -1.0f, 0.5f, -1.0f, 3.0f};
    const float bias[3] = {0.25f, -0.5f, 1.0f};
    const float input[4] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float expected[6] = {2.25f, -0.5f, 6.0f, 0.75f, -5.5f, 11.0f};
    return run_conv_case(graph_path, weights_path, graph, weight_shape,
                         weight, 6, bias, 3, input, 4, expected, 6);
}

static int test_explicit_hwcm_depthwise_conv(void) {
    const char* graph_path = "/tmp/volvox-conv2d-f32-hwcm.json";
    const char* weights_path = "/tmp/volvox-conv2d-f32-hwcm.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1,1,2,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,2,4]},"
        "\"params\":{\"weight_layout\":\"HWCM\",\"groups\":2}}],"
        "\"outputs\":[\"y\"]}";
    const int weight_shape[4] = {1, 1, 2, 2};
    const float weight[4] = {2.0f, -1.0f, 0.5f, 3.0f};
    const float bias[4] = {0.1f, 0.2f, -0.3f, 1.0f};
    const float input[4] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float expected[8] = {2.1f, -0.8f, 0.7f, 7.0f,
                               -1.9f, 1.2f, 1.2f, 10.0f};
    return run_conv_case(graph_path, weights_path, graph, weight_shape,
                         weight, 4, bias, 4, input, 4, expected, 8);
}

static int test_maxpool_canonical_pads_and_batch(void) {
    const char* graph_path = "/tmp/volvox-maxpool-f32-pads.json";
    const char* weights_path = "/tmp/volvox-maxpool-f32-pads.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[2,2,3,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2,3,1]},"
        "\"params\":{\"kernel\":[2,2],\"stride\":[1,1],"
        "\"pads\":[1,0,0,1],\"dilation\":[1,1],\"ceil_mode\":false}}],"
        "\"outputs\":[\"y\"]}";
    const float input[12] = {
        1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f,
    };
    const float expected[12] = {
        2.0f, 3.0f, 3.0f, 5.0f, 6.0f, 6.0f,
        -1.0f, -2.0f, -3.0f, -1.0f, -2.0f, -3.0f,
    };
    float output[12] = {0};
    SafetensorsFile file;

    CHECK(write_text(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, input,
                                        sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 12) == 0);
    CHECK(close_array(output, expected, 12));
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_spatial_igemm_thread_parity(void) {
    enum {
        height = 20,
        width = 20,
        input_channels = 17,
        output_channels = 19,
        kernel = 3,
    };
    const int pads[4] = {1, 1, 1, 1};
    const size_t input_count =
        (size_t)height * width * input_channels;
    const size_t weight_count =
        (size_t)kernel * kernel * input_channels * output_channels;
    const size_t output_count =
        (size_t)height * width * output_channels;
    float* input = (float*)malloc(input_count * sizeof(float));
    float* weight = (float*)malloc(weight_count * sizeof(float));
    float* bias = (float*)malloc(output_channels * sizeof(float));
    float* serial = (float*)malloc(output_count * sizeof(float));
    float* parallel = (float*)malloc(output_count * sizeof(float));
    float* reference = (float*)malloc(output_count * sizeof(float));

    CHECK(input && weight && bias && serial && parallel && reference);
    for (size_t index = 0; index < input_count; index++)
        input[index] = (float)((int)((index * 17 + 3) % 29) - 14) / 32.0f;
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (float)((int)((index * 13 + 5) % 23) - 11) / 64.0f;
    for (int channel = 0; channel < output_channels; channel++)
        bias[channel] = (float)(channel - 9) / 128.0f;

    for (int oy = 0; oy < height; oy++) {
        for (int ox = 0; ox < width; ox++) {
            for (int oc = 0; oc < output_channels; oc++) {
                float sum = bias[oc];
                for (int ky = 0; ky < kernel; ky++) {
                    const int iy = oy + ky - pads[0];
                    if ((unsigned)iy >= height) continue;
                    for (int kx = 0; kx < kernel; kx++) {
                        const int ix = ox + kx - pads[1];
                        if ((unsigned)ix >= width) continue;
                        const float* source = input +
                            ((size_t)iy * width + ix) * input_channels;
                        const float* weights = weight +
                            ((size_t)ky * kernel + kx) * input_channels *
                                output_channels + oc;
                        for (int ic = 0; ic < input_channels; ic++)
                            sum += source[ic] * weights[ic * output_channels];
                    }
                }
                reference[((size_t)oy * width + ox) * output_channels + oc] =
                    sum;
            }
        }
    }

    vx_set_num_threads(1);
    CHECK(vx_kernels_thread_count() == 1);
    vx_conv2d_generic_f32(
        0, input, serial, weight, bias,
        1, height, width, input_channels,
        height, width, output_channels,
        kernel, kernel, input_channels, 1,
        1, 1, pads, 1, 1, 0);
    CHECK(close_array(serial, reference, output_count));

    vx_set_num_threads(6);
    CHECK(vx_kernels_thread_count() == 6);
    vx_conv2d_generic_f32(
        0, input, parallel, weight, bias,
        1, height, width, input_channels,
        height, width, output_channels,
        kernel, kernel, input_channels, 1,
        1, 1, pads, 1, 1, 0);
    CHECK(memcmp(serial, parallel, output_count * sizeof(float)) == 0);
    vx_set_num_threads(1);

    free(reference);
    free(parallel);
    free(serial);
    free(bias);
    free(weight);
    free(input);
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    VolvoxAIEngineOptions options = {
        .backend = VOLVOXAI_BACKEND_CPU,
        .debug = 0,
        .cpu_threads = 1,
    };
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    CHECK(volvoxai_engine_configure(&options) == 0);
    CHECK(test_default_hwio_conv() == 0);
    CHECK(test_explicit_hwcm_depthwise_conv() == 0);
    CHECK(test_maxpool_canonical_pads_and_batch() == 0);
    CHECK(test_spatial_igemm_thread_parity() == 0);
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("native F32 Conv2D and canonical-pads MaxPool2D tests passed");
    return 0;
}
