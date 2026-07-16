#include "safetensors.h"
#include "volvoxai.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

/* CPU-only test composition intentionally omits the device shader store. */
void volvoxai_shader_store_shutdown(void) {}

static int closef(float left, float right) {
    return fabsf(left - right) <= 2.0e-5f * fmaxf(1.0f, fabsf(right));
}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int result = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return result ? 0 : -1;
}

static int write_dummy_weights(const char* path) {
    const int shape[1] = {1};
    const float value = 0.0f;
    SafetensorsFile file;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0) return -1;
    if (safetensors_add_tensor(&file, "unused", SAFETENSORS_DTYPE_F32,
                               shape, 1, &value, sizeof(value)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    int result = safetensors_save(path, &file);
    safetensors_free(&file);
    return result;
}

static int expect_invalid_graph(const char* weights_path, const char* tag,
                                const char* config) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/volvox-invalid-graph-%s.json", tag);
    if (write_text(path, config) != 0) return -1;
    int result = volvoxai_engine_init(path, weights_path);
    volvoxai_engine_shutdown();
    remove(path);
    return result == -1 ? 0 : -1;
}

static int test_graph_contract_validation(const char* weights_path) {
    static const char* missing_op =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1]}}]}";
    static const char* unresolved_input =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Identity\",\"inputs\":{\"input\":\"missing\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]}}]}";
    static const char* forward_reference =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Identity\",\"inputs\":{\"input\":\"later\"},"
        "\"outputs\":{\"out\":\"first\"},\"outputs_shape\":{\"out\":[1]}},{"
        "\"op\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"later\"},\"outputs_shape\":{\"out\":[1]}}]}";
    static const char* duplicate_output =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Split\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"left\":\"same\",\"right\":\"same\"},"
        "\"outputs_shape\":{\"left\":[1],\"right\":[1]}}]}";
    static const char* missing_output_shapes =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"}}]}";
    static const char* missing_output_shape_entry =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Split\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"left\":\"left\",\"right\":\"right\"},"
        "\"outputs_shape\":{\"left\":[1]}}]}";
    static const char* too_many_inputs =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[{"
        "\"op\":\"Concat\",\"inputs\":{\"i0\":\"x\",\"i1\":\"x\","
        "\"i2\":\"x\",\"i3\":\"x\",\"i4\":\"x\",\"i5\":\"x\","
        "\"i6\":\"x\",\"i7\":\"x\",\"i8\":\"x\",\"i9\":\"x\","
        "\"i10\":\"x\",\"i11\":\"x\",\"i12\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[13]}}]}";
    static const char* undefined_graph_output =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[],"
        "\"outputs\":[\"not_declared\"]}";
    static const char* duplicate_graph_output =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[],"
        "\"outputs\":[\"x\",\"x\"]}";
    static const char* duplicate_legacy_graph_output =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[],"
        "\"outputs\":{\"first\":\"x\",\"second\":\"x\"}}";
    CHECK(expect_invalid_graph(weights_path, "missing-op", missing_op) == 0);
    CHECK(expect_invalid_graph(weights_path, "unresolved", unresolved_input) == 0);
    CHECK(expect_invalid_graph(weights_path, "forward-ref", forward_reference) == 0);
    CHECK(expect_invalid_graph(weights_path, "duplicate-output", duplicate_output) == 0);
    CHECK(expect_invalid_graph(weights_path, "missing-output-shapes", missing_output_shapes) == 0);
    CHECK(expect_invalid_graph(weights_path, "missing-output-shape-entry", missing_output_shape_entry) == 0);
    CHECK(expect_invalid_graph(weights_path, "too-many-inputs", too_many_inputs) == 0);
    CHECK(expect_invalid_graph(weights_path, "undefined-output", undefined_graph_output) == 0);
    CHECK(expect_invalid_graph(weights_path, "duplicate-graph-output", duplicate_graph_output) == 0);
    CHECK(expect_invalid_graph(weights_path, "duplicate-legacy-graph-output",
                               duplicate_legacy_graph_output) == 0);

    const char* oversized_path = "/tmp/volvox-invalid-graph-too-many-nodes.json";
    FILE* file = fopen(oversized_path, "wb");
    CHECK(file != NULL);
    CHECK(fputs("{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[", file) >= 0);
    for (int index = 0; index < 1025; index++) {
        CHECK(fprintf(file,
            "%s{\"op\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
            "\"outputs\":{\"out\":\"y%d\"},\"outputs_shape\":{\"out\":[1]}}",
            index ? "," : "", index) > 0);
    }
    CHECK(fputs("]}", file) >= 0 && fclose(file) == 0);
    CHECK(volvoxai_engine_init(oversized_path, weights_path) == -1);
    volvoxai_engine_shutdown();
    remove(oversized_path);
    return 0;
}

static int test_inputs_have_no_name_based_defaults(const char* weights_path) {
    const char* path = "/tmp/volvox-positions-are-caller-owned.json";
    const char* config =
        "{\"inputs\":{\"positions\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":[],\"outputs\":[\"positions\"]}";
    int32_t positions[3] = {-1, -1, -1};
    CHECK(write_text(path, config) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("positions", positions, sizeof(positions)) == 0);
    CHECK(positions[0] == 0 && positions[1] == 0 && positions[2] == 0);
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

static int test_argmax_f32_to_i32(const char* weights_path) {
    const char* path = "/tmp/volvox-argmax-f32-config.json";
    const char* last_axis_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"indices\"},"
        "\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1,\"keepdims\":0}}],"
        "\"outputs\":[\"indices\"]}";
    const char* kept_non_last_axis_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"indices\"},"
        "\"outputs_shape\":{\"out\":[1,1,4]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1,\"keepdims\":1}}],"
        "\"outputs\":[\"indices\"]}";
    const char* select_last_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"indices\"},"
        "\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1,\"keepdims\":0,\"select_last_index\":1}}],"
        "\"outputs\":[\"indices\"]}";
    const float input[8] = {1.0f, 7.0f, 7.0f, 2.0f,
                            -5.0f, -2.0f, -3.0f, -2.0f};
    const int32_t expected[2] = {1, 1};
    const int32_t expected_non_last[4] = {0, 0, 0, 0};
    int32_t output[2] = {-1, -1};
    int32_t non_last_output[4] = {-1, -1, -1, -1};
    CHECK(write_text(path, last_axis_config) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("indices", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();

    CHECK(write_text(path, kept_non_last_axis_config) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("indices", non_last_output,
                                           sizeof(non_last_output)) == 0);
    CHECK(memcmp(non_last_output, expected_non_last, sizeof(non_last_output)) == 0);
    volvoxai_engine_shutdown();

    CHECK(write_text(path, select_last_config) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

int main(void) {
    const char* config_path = "/tmp/volvox-sequence-runtime-config.json";
    const char* weights_path = "/tmp/volvox-sequence-runtime-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"},"
        "\"positions\":{\"shape\":[2],\"dtype\":\"int32\"},"
        "\"u\":{\"shape\":[1,2,1],\"dtype\":\"float32\"},"
        "\"delta\":{\"shape\":[1,2,1],\"dtype\":\"float32\"},"
        "\"A\":{\"shape\":[1,1],\"dtype\":\"float32\"},"
        "\"B\":{\"shape\":[2,1],\"dtype\":\"float32\"},"
        "\"C\":{\"shape\":[1],\"dtype\":\"float32\"},"
        "\"initial\":{\"shape\":[1,1,1],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Sin\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"sin_out\"},\"outputs_shape\":{\"out\":[1,2,4]}},"
        "{\"opType\":\"Cos\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"cos_out\"},\"outputs_shape\":{\"out\":[1,2,4]}},"
        "{\"opType\":\"Dropout\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"dropout_out\"},\"outputs_shape\":{\"out\":[1,2,4]},"
        "\"params\":{\"p\":0.5,\"seed\":12345}},"
        "{\"opType\":\"RotaryEmbedding\","
        "\"inputs\":{\"input\":\"x\",\"position_ids\":\"positions\"},"
        "\"outputs\":{\"out\":\"rope_out\"},\"outputs_shape\":{\"out\":[1,2,4]},"
        "\"params\":{\"rotary_dim\":4,\"theta\":10.0,\"interleaved\":false}},"
        "{\"opType\":\"SelectiveScan\","
        "\"inputs\":{\"input\":\"u\",\"delta\":\"delta\",\"A\":\"A\","
        "\"B\":\"B\",\"C\":\"C\",\"initial_state\":\"initial\"},"
        "\"outputs\":{\"out\":\"scan_out\",\"state\":\"scan_state\"},"
        "\"outputs_shape\":{\"out\":[1,2,1],\"state\":[1,1,1]},"
        "\"params\":{\"delta_softplus\":false}}],"
        "\"outputs\":[\"sin_out\",\"cos_out\",\"dropout_out\",\"rope_out\",\"scan_out\",\"scan_state\"]}";
    const float x[8] = {1, 2, 3, 4, -1, -2, -3, -4};
    const int32_t positions[2] = {0, 1};
    const float u[2] = {1, 2};
    const float delta[2] = {0.5f, 0.25f};
    const float a[1] = {-1};
    const float b[2] = {2, 3};
    const float c[1] = {0.5f};
    const float initial[1] = {0.25f};
    float sin_out[8], cos_out[8], dropout_out[8], rope_out[8], scan_out[2], scan_state[1];
    VolvoxAIEngineOptions options = {
        .backend = VOLVOXAI_BACKEND_CPU,
        .debug = 0,
        .cpu_threads = 2,
    };

    CHECK(write_text(config_path, config) == 0);
    CHECK(write_dummy_weights(weights_path) == 0);
    CHECK(volvoxai_engine_configure(&options) == 0);
    memset(&options, 0, sizeof(options));
    CHECK(volvoxai_engine_get_options(&options) == 0);
    CHECK(options.backend == VOLVOXAI_BACKEND_CPU && options.debug == 0 &&
          options.cpu_threads == 2);
    CHECK(strcmp(volvoxai_engine_backend_name(), "CPU") == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    CHECK(volvoxai_engine_graph_input_count() == 8);
    CHECK(strcmp(volvoxai_engine_graph_input_name(0), "x") == 0);
    CHECK(volvoxai_engine_graph_input_name(8) == NULL);
    CHECK(volvoxai_engine_graph_output_count() == 6);
    CHECK(strcmp(volvoxai_engine_graph_output_name(0), "sin_out") == 0);
    CHECK(strcmp(volvoxai_engine_graph_output_name(5), "scan_state") == 0);
    CHECK(volvoxai_engine_graph_output_name(6) == NULL);
    CHECK(volvoxai_engine_configure(&options) == -1);
    CHECK(volvoxai_engine_set_debug(1) == 0 && volvoxai_engine_debug() == 1);
    CHECK(volvoxai_engine_set_debug(0) == 0 && volvoxai_engine_debug() == 0);
    CHECK(volvoxai_engine_set_execution_row(1) == 0);
    CHECK(volvoxai_engine_execution_row() == 1);
    CHECK(volvoxai_engine_set_execution_row(2) == -1);
    CHECK(volvoxai_engine_set_execution_row(-1) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32, x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("positions", VOLVOXAI_DTYPE_I32,
                                        positions, sizeof(positions)) == 0);
    CHECK(volvoxai_engine_set_input_raw("u", VOLVOXAI_DTYPE_F32, u, sizeof(u)) == 0);
    CHECK(volvoxai_engine_set_input_raw("delta", VOLVOXAI_DTYPE_F32,
                                        delta, sizeof(delta)) == 0);
    CHECK(volvoxai_engine_set_input_raw("A", VOLVOXAI_DTYPE_F32, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("B", VOLVOXAI_DTYPE_F32, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_set_input_raw("C", VOLVOXAI_DTYPE_F32, c, sizeof(c)) == 0);
    CHECK(volvoxai_engine_set_input_raw("initial", VOLVOXAI_DTYPE_F32,
                                        initial, sizeof(initial)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("sin_out", sin_out, 8) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("cos_out", cos_out, 8) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("dropout_out", dropout_out, 8) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("rope_out", rope_out, 8) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("scan_out", scan_out, 2) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("scan_state", scan_state, 1) == 0);
    int row_count = 0;
    const float* sin_row = volvoxai_engine_tensor_row_f32("sin_out", 1, &row_count);
    CHECK(sin_row != NULL && row_count == 4);
    for (int index = 0; index < row_count; index++) CHECK(sin_row[index] == sin_out[4 + index]);
    CHECK(volvoxai_engine_tensor_row_f32(NULL, 0, &row_count) == NULL && row_count == 0);
    CHECK(volvoxai_engine_tensor_row_f32("sin_out", 2, &row_count) == NULL && row_count == 0);

    for (int index = 0; index < 8; index++) {
        CHECK(closef(sin_out[index], sinf(x[index])));
        CHECK(closef(cos_out[index], cosf(x[index])));
        CHECK(dropout_out[index] == x[index]);
    }
    for (int index = 0; index < 4; index++) CHECK(rope_out[index] == x[index]);
    const float angle0 = 1.0f;
    const float angle1 = 1.0f / sqrtf(10.0f);
    CHECK(closef(rope_out[4], x[4] * cosf(angle0) - x[6] * sinf(angle0)));
    CHECK(closef(rope_out[6], x[4] * sinf(angle0) + x[6] * cosf(angle0)));
    CHECK(closef(rope_out[5], x[5] * cosf(angle1) - x[7] * sinf(angle1)));
    CHECK(closef(rope_out[7], x[5] * sinf(angle1) + x[7] * cosf(angle1)));
    const float state0 = expf(-0.5f) * 0.25f + 0.5f * 2.0f;
    const float state1 = expf(-0.25f) * state0 + 0.25f * 3.0f * 2.0f;
    CHECK(closef(scan_out[0], state0 * 0.5f));
    CHECK(closef(scan_out[1], state1 * 0.5f));
    CHECK(closef(scan_state[0], state1));

    volvoxai_engine_shutdown();
    memset(&options, 0xff, sizeof(options));
    CHECK(volvoxai_engine_get_options(&options) == 0);
    CHECK(options.backend == VOLVOXAI_BACKEND_CPU && options.debug == 0 &&
          options.cpu_threads == 0);
    options.backend = VOLVOXAI_BACKEND_CPU;
    options.debug = 1;
    options.cpu_threads = 2;
    CHECK(volvoxai_engine_configure(&options) == 0);
    VolvoxAIEngineOptions rejected = options;
    rejected.backend = (VolvoxAIEngineBackend)99;
    CHECK(volvoxai_engine_configure(&rejected) == -1);
    memset(&rejected, 0, sizeof(rejected));
    CHECK(volvoxai_engine_get_options(&rejected) == 0);
    CHECK(rejected.backend == VOLVOXAI_BACKEND_CPU && rejected.debug == 1 &&
          rejected.cpu_threads == 2);
    rejected.backend = VOLVOXAI_BACKEND_VULKAN;
    rejected.debug = 0;
    rejected.cpu_threads = 1;
    CHECK(volvoxai_engine_configure(&rejected) == -1);
    memset(&rejected, 0, sizeof(rejected));
    CHECK(volvoxai_engine_get_options(&rejected) == 0);
    CHECK(rejected.backend == VOLVOXAI_BACKEND_CPU && rejected.debug == 1 &&
          rejected.cpu_threads == 2);
    CHECK(volvoxai_engine_init(NULL, NULL) == -1);
    CHECK(volvoxai_engine_init("", NULL) == -1);
    memset(&rejected, 0, sizeof(rejected));
    CHECK(volvoxai_engine_get_options(&rejected) == 0);
    CHECK(rejected.backend == VOLVOXAI_BACKEND_CPU && rejected.debug == 1 &&
          rejected.cpu_threads == 2);
    CHECK(test_graph_contract_validation(weights_path) == 0);
    CHECK(test_inputs_have_no_name_based_defaults(weights_path) == 0);
    CHECK(test_argmax_f32_to_i32(weights_path) == 0);
    remove(config_path);
    remove(weights_path);
    puts("native sequence runtime tests passed");
    return 0;
}
