#define _POSIX_C_SOURCE 200809L

#include "safetensors.h"
#include "engine_core.h"
#include "runtime_state.h"

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
                                const char* graph) {
    char path[256];
    snprintf(path, sizeof(path), "/tmp/volvox-invalid-graph-%s.json", tag);
    if (write_text(path, graph) != 0) return -1;
    int result = volvoxai_engine_init(path, weights_path);
    volvoxai_engine_shutdown();
    remove(path);
    return result == -1 ? 0 : -1;
}

static int test_graph_contract_validation(const char* weights_path) {
    static const char* missing_format =
        "{\"inputs\":{\"x\":{\"shape\":[1]}},\"nodes\":[],\"outputs\":[\"x\"]}";
    static const char* unsupported_format =
        "{\"format\":\"volvox-graph/v2\",\"inputs\":{\"x\":{\"shape\":[1]}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    static const char* wrong_case_format =
        "{\"format\":\"Volvox-Graph/v1\",\"inputs\":{\"x\":{\"shape\":[1]}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    static const char* wrong_type_format =
        "{\"format\":1,\"inputs\":{\"x\":{\"shape\":[1]}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    static const char* missing_op =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"y\"]}";
    static const char* retired_operator =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"RotaryEmbedding\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"y\"]}";
    static const char* unknown_operator =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"DefinitelyUnknown\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"y\"]}";
    static const char* unresolved_input =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Identity\",\"inputs\":{\"input\":\"missing\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"y\"]}";
    static const char* forward_reference =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Identity\",\"inputs\":{\"input\":\"later\"},"
        "\"outputs\":{\"out\":\"first\"},\"outputs_shape\":{\"out\":[1]}},{"
        "\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"later\"},\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"first\"]}";
    static const char* duplicate_output =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Split\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"left\":\"same\",\"right\":\"same\"},"
        "\"outputs_shape\":{\"left\":[1],\"right\":[1]}}],\"outputs\":[\"same\"]}";
    static const char* missing_output_shapes =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"}}],\"outputs\":[\"y\"]}";
    static const char* missing_output_shape_entry =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Split\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"left\":\"left\",\"right\":\"right\"},"
        "\"outputs_shape\":{\"left\":[1]}}],\"outputs\":[\"left\",\"right\"]}";
    static const char* too_many_inputs =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Concat\",\"inputs\":{\"i0\":\"x\",\"i1\":\"x\","
        "\"i2\":\"x\",\"i3\":\"x\",\"i4\":\"x\",\"i5\":\"x\","
        "\"i6\":\"x\",\"i7\":\"x\",\"i8\":\"x\",\"i9\":\"x\","
        "\"i10\":\"x\",\"i11\":\"x\",\"i12\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[13]}}],\"outputs\":[\"y\"]}";
    static const char* undefined_graph_output =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[],"
        "\"outputs\":[\"not_declared\"]}";
    static const char* duplicate_graph_output =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[],"
        "\"outputs\":[\"x\",\"x\"]}";
    static const char* object_graph_outputs =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[],"
        "\"outputs\":{\"first\":\"x\"}}";
    CHECK(expect_invalid_graph(weights_path, "missing-format", missing_format) == 0);
    CHECK(expect_invalid_graph(weights_path, "unsupported-format", unsupported_format) == 0);
    CHECK(expect_invalid_graph(weights_path, "wrong-case-format", wrong_case_format) == 0);
    CHECK(expect_invalid_graph(weights_path, "wrong-type-format", wrong_type_format) == 0);
    CHECK(expect_invalid_graph(weights_path, "missing-op", missing_op) == 0);
    CHECK(expect_invalid_graph(weights_path, "retired-operator", retired_operator) == 0);
    CHECK(expect_invalid_graph(weights_path, "unknown-operator", unknown_operator) == 0);
    CHECK(expect_invalid_graph(weights_path, "unresolved", unresolved_input) == 0);
    CHECK(expect_invalid_graph(weights_path, "forward-ref", forward_reference) == 0);
    CHECK(expect_invalid_graph(weights_path, "duplicate-output", duplicate_output) == 0);
    CHECK(expect_invalid_graph(weights_path, "missing-output-shapes", missing_output_shapes) == 0);
    CHECK(expect_invalid_graph(weights_path, "missing-output-shape-entry", missing_output_shape_entry) == 0);
    CHECK(expect_invalid_graph(weights_path, "too-many-inputs", too_many_inputs) == 0);
    CHECK(expect_invalid_graph(weights_path, "undefined-output", undefined_graph_output) == 0);
    CHECK(expect_invalid_graph(weights_path, "duplicate-graph-output", duplicate_graph_output) == 0);
    CHECK(expect_invalid_graph(weights_path, "object-graph-outputs",
                               object_graph_outputs) == 0);

    const char* oversized_path = "/tmp/volvox-invalid-graph-too-many-nodes.json";
    FILE* file = fopen(oversized_path, "wb");
    CHECK(file != NULL);
    CHECK(fputs("{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{"
                "\"shape\":[1],\"dtype\":\"float32\"}},\"nodes\":[",
                file) >= 0);
    for (int index = 0; index < 1025; index++) {
        CHECK(fprintf(file,
            "%s{\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
            "\"outputs\":{\"out\":\"y%d\"},\"outputs_shape\":{\"out\":[1]}}",
            index ? "," : "", index) > 0);
    }
    CHECK(fputs("],\"outputs\":[\"y1024\"]}", file) >= 0 &&
          fclose(file) == 0);
    CHECK(volvoxai_engine_init(oversized_path, weights_path) == -1);
    volvoxai_engine_shutdown();
    remove(oversized_path);
    return 0;
}

static int test_inputs_have_no_name_based_defaults(const char* weights_path) {
    const char* path = "/tmp/volvox-positions-are-caller-owned.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"positions\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":[],\"outputs\":[\"positions\"]}";
    int32_t positions[3] = {-1, -1, -1};
    CHECK(write_text(path, graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("positions", positions, sizeof(positions)) == 0);
    CHECK(positions[0] == 0 && positions[1] == 0 && positions[2] == 0);
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

static int test_argmax_f32_to_i32(const char* weights_path) {
    const char* path = "/tmp/volvox-argmax-f32-graph.json";
    const char* last_axis_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"indices\"},"
        "\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1,\"keepdims\":0}}],"
        "\"outputs\":[\"indices\"]}";
    const char* kept_non_last_axis_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"indices\"},"
        "\"outputs_shape\":{\"out\":[1,1,4]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1,\"keepdims\":1}}],"
        "\"outputs\":[\"indices\"]}";
    const char* select_last_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,4],\"dtype\":\"float32\"}},"
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
    CHECK(write_text(path, last_axis_graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("indices", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();

    CHECK(write_text(path, kept_non_last_axis_graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("indices", non_last_output,
                                           sizeof(non_last_output)) == 0);
    CHECK(memcmp(non_last_output, expected_non_last, sizeof(non_last_output)) == 0);
    volvoxai_engine_shutdown();

    CHECK(write_text(path, select_last_graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", VOLVOXAI_DTYPE_F32,
                                        input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

static int test_required_onnx_operator_graph(const char* weights_path) {
    const char* path = "/tmp/volvox-required-onnx-ops.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"ids\":{\"shape\":[2,3],\"dtype\":\"int32\"},"
        "\"zero\":{\"shape\":[1],\"dtype\":\"int32\"},"
        "\"threshold\":{\"shape\":[1],\"dtype\":\"int32\"},"
        "\"float_left\":{\"shape\":[2,3],\"dtype\":\"float32\"},"
        "\"float_right\":{\"shape\":[1,3],\"dtype\":\"float32\"},"
        "\"gather_indices\":{\"shape\":[2],\"dtype\":\"int32\"},"
        "\"mat_a\":{\"shape\":[2,2,3],\"dtype\":\"float32\"},"
        "\"mat_b\":{\"shape\":[1,3,2],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Equal\",\"inputs\":{\"a\":\"ids\",\"b\":\"zero\"},"
        "\"outputs\":{\"out\":\"equal\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"GreaterOrEqual\","
        "\"inputs\":{\"a\":\"ids\",\"b\":\"threshold\"},"
        "\"outputs\":{\"out\":\"greater_equal\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Not\",\"inputs\":{\"input\":\"equal\"},"
        "\"outputs\":{\"out\":\"not_equal\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Clip\",\"inputs\":{\"input\":\"ids\"},"
        "\"outputs\":{\"out\":\"clipped\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"min\":0,\"max\":3}},"
        "{\"opType\":\"Expand\",\"inputs\":{\"input\":\"zero\"},"
        "\"outputs\":{\"out\":\"zero_expanded\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Where\","
        "\"inputs\":{\"condition\":\"not_equal\",\"x\":\"clipped\","
        "\"y\":\"zero_expanded\"},\"outputs\":{\"out\":\"chosen\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Reshape\",\"inputs\":{\"input\":\"chosen\"},"
        "\"outputs\":{\"out\":\"reshaped\"},"
        "\"outputs_shape\":{\"out\":[3,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"}},"
        "{\"opType\":\"Transpose\",\"inputs\":{\"input\":\"reshaped\"},"
        "\"outputs\":{\"out\":\"transposed\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"perm\":[1,0]}},"
        "{\"opType\":\"Slice\",\"inputs\":{\"input\":\"transposed\"},"
        "\"outputs\":{\"out\":\"sliced\"},"
        "\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"starts\":[0],\"axes\":[1],\"steps\":[2]}},"
        "{\"opType\":\"Concat\",\"inputs\":{\"input0\":\"sliced\","
        "\"input1\":\"sliced\"},\"outputs\":{\"out\":\"concatenated\"},"
        "\"outputs_shape\":{\"out\":[2,4]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1}},"
        "{\"opType\":\"Split\",\"inputs\":{\"input\":\"concatenated\"},"
        "\"outputs\":{\"out0\":\"split0\",\"out1\":\"split1\"},"
        "\"outputs_shape\":{\"out0\":[2,2],\"out1\":[2,2]},"
        "\"outputs_dtype\":{\"out0\":\"int32\",\"out1\":\"int32\"},"
        "\"params\":{\"axis\":1}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"split0\"},"
        "\"outputs\":{\"out\":\"casted\"},"
        "\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"to\":\"float32\"}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"ids\"},"
        "\"outputs\":{\"out\":\"ids_i8\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"to\":\"int8\"}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"ids_i8\"},"
        "\"outputs\":{\"out\":\"ids_i8_i32\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"to\":\"int32\"}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"ids\"},"
        "\"outputs\":{\"out\":\"ids_u8\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},"
        "\"params\":{\"to\":\"uint8\"}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"ids_u8\"},"
        "\"outputs\":{\"out\":\"ids_u8_i32\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"to\":\"int32\"}},"
        "{\"opType\":\"Expand\",\"inputs\":{\"input\":\"float_right\"},"
        "\"outputs\":{\"out\":\"float_right_expanded\"},"
        "\"outputs_shape\":{\"out\":[2,3]}},"
        "{\"opType\":\"Cast\",\"inputs\":{\"input\":\"float_right_expanded\"},"
        "\"outputs\":{\"out\":\"float_to_i32\"},"
        "\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"to\":\"int32\"}},"
        "{\"opType\":\"Where\","
        "\"inputs\":{\"condition\":\"greater_equal\",\"x\":\"float_left\","
        "\"y\":\"float_right_expanded\"},"
        "\"outputs\":{\"out\":\"float_chosen\"},"
        "\"outputs_shape\":{\"out\":[2,3]}},"
        "{\"opType\":\"Sub\",\"inputs\":{\"a\":\"float_left\","
        "\"b\":\"float_right\"},\"outputs\":{\"out\":\"subtracted\"},"
        "\"outputs_shape\":{\"out\":[2,3]}},"
        "{\"opType\":\"Div\",\"inputs\":{\"a\":\"float_left\","
        "\"b\":\"float_right\"},\"outputs\":{\"out\":\"divided\"},"
        "\"outputs_shape\":{\"out\":[2,3]}},"
        "{\"opType\":\"Gather\",\"inputs\":{\"input\":\"float_left\","
        "\"indices\":\"gather_indices\"},"
        "\"outputs\":{\"out\":\"gathered\"},"
        "\"outputs_shape\":{\"out\":[2,2]},\"params\":{\"axis\":1}},"
        "{\"opType\":\"BatchMatMul\",\"inputs\":{\"a\":\"mat_a\",\"b\":\"mat_b\"},"
        "\"outputs\":{\"out\":\"batch_product\"},"
        "\"outputs_shape\":{\"out\":[2,2,2]}}],"
        "\"outputs\":[\"equal\",\"greater_equal\",\"not_equal\",\"clipped\","
        "\"chosen\",\"transposed\",\"sliced\",\"concatenated\",\"split0\","
        "\"split1\",\"casted\",\"ids_i8_i32\",\"ids_u8_i32\","
        "\"float_to_i32\",\"float_chosen\","
        "\"subtracted\",\"divided\","
        "\"gathered\",\"batch_product\"]}";
    const int32_t ids[6] = {-1, 0, 2, 3, 4, 5};
    const int32_t zero[1] = {0};
    const int32_t threshold[1] = {3};
    const int32_t gather_indices[2] = {-1, 0};
    const float float_left[6] = {6, 8, 10, 12, 14, 16};
    const float float_right[3] = {2, 4, 5};
    const float mat_a[12] = {1, 2, 3, 4, 5, 6,
                             7, 8, 9, 10, 11, 12};
    const float mat_b[6] = {1, 2, 3, 4, 5, 6};
    const int32_t expected_equal[6] = {0, 1, 0, 0, 0, 0};
    const int32_t expected_ge[6] = {0, 0, 0, 1, 1, 1};
    const int32_t expected_not[6] = {1, 0, 1, 1, 1, 1};
    const int32_t expected_clipped[6] = {0, 0, 2, 3, 3, 3};
    const int32_t expected_transposed[6] = {0, 2, 3, 0, 3, 3};
    const int32_t expected_sliced[4] = {0, 3, 0, 3};
    const int32_t expected_concat[8] = {0, 3, 0, 3, 0, 3, 0, 3};
    const int32_t expected_float_to_i32[6] = {2, 4, 5, 2, 4, 5};
    const int32_t expected_u8_roundtrip[6] = {255, 0, 2, 3, 4, 5};
    const float expected_cast[4] = {0, 3, 0, 3};
    const float expected_where[6] = {2, 4, 5, 12, 14, 16};
    const float expected_sub[6] = {4, 4, 5, 10, 10, 11};
    const float expected_div[6] = {3, 2, 2, 6, 3.5f, 3.2f};
    const float expected_gather[4] = {10, 6, 16, 12};
    const float expected_product[8] = {22, 28, 49, 64, 76, 100, 103, 136};
    int32_t i32_output[8] = {0};
    float f32_output[8] = {0};

    CHECK(write_text(path, graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "ids", VOLVOXAI_DTYPE_I32, ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "zero", VOLVOXAI_DTYPE_I32, zero, sizeof(zero)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "threshold", VOLVOXAI_DTYPE_I32, threshold, sizeof(threshold)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "float_left", VOLVOXAI_DTYPE_F32, float_left, sizeof(float_left)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "float_right", VOLVOXAI_DTYPE_F32, float_right, sizeof(float_right)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "gather_indices", VOLVOXAI_DTYPE_I32,
        gather_indices, sizeof(gather_indices)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "mat_a", VOLVOXAI_DTYPE_F32, mat_a, sizeof(mat_a)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "mat_b", VOLVOXAI_DTYPE_F32, mat_b, sizeof(mat_b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);

#define CHECK_I32_TENSOR(name, expected, count) do { \
    memset(i32_output, 0, sizeof(i32_output)); \
    CHECK(volvoxai_engine_copy_tensor_raw( \
        (name), i32_output, (count) * sizeof(int32_t)) == 0); \
    CHECK(memcmp(i32_output, (expected), (count) * sizeof(int32_t)) == 0); \
} while (0)
#define CHECK_F32_TENSOR(name, expected, count) do { \
    memset(f32_output, 0, sizeof(f32_output)); \
    CHECK(volvoxai_engine_copy_tensor_f32((name), f32_output, (count)) == 0); \
    for (int value_index = 0; value_index < (count); value_index++) \
        CHECK(closef(f32_output[value_index], (expected)[value_index])); \
} while (0)

    CHECK_I32_TENSOR("equal", expected_equal, 6);
    CHECK_I32_TENSOR("greater_equal", expected_ge, 6);
    CHECK_I32_TENSOR("not_equal", expected_not, 6);
    CHECK_I32_TENSOR("clipped", expected_clipped, 6);
    CHECK_I32_TENSOR("chosen", expected_clipped, 6);
    CHECK_I32_TENSOR("transposed", expected_transposed, 6);
    CHECK_I32_TENSOR("sliced", expected_sliced, 4);
    CHECK_I32_TENSOR("concatenated", expected_concat, 8);
    CHECK_I32_TENSOR("split0", expected_sliced, 4);
    CHECK_I32_TENSOR("split1", expected_sliced, 4);
    CHECK_F32_TENSOR("casted", expected_cast, 4);
    CHECK_I32_TENSOR("ids_i8_i32", ids, 6);
    CHECK_I32_TENSOR("ids_u8_i32", expected_u8_roundtrip, 6);
    CHECK_I32_TENSOR("float_to_i32", expected_float_to_i32, 6);
    CHECK_F32_TENSOR("float_chosen", expected_where, 6);
    CHECK_F32_TENSOR("subtracted", expected_sub, 6);
    CHECK_F32_TENSOR("divided", expected_div, 6);
    CHECK_F32_TENSOR("gathered", expected_gather, 4);
    CHECK_F32_TENSOR("batch_product", expected_product, 8);

#undef CHECK_F32_TENSOR
#undef CHECK_I32_TENSOR
    volvoxai_engine_shutdown();

    const char* invalid_where =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"condition\":{\"shape\":[1],\"dtype\":\"int32\"},"
        "\"x\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"y\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Where\","
        "\"inputs\":{\"condition\":\"condition\",\"x\":\"x\",\"y\":\"y\"},"
        "\"outputs\":{\"out\":\"output\"},"
        "\"outputs_shape\":{\"out\":[2]}}],\"outputs\":[\"output\"]}";
    const int32_t scalar_condition[1] = {1};
    const float pair[2] = {1, 2};
    CHECK(write_text(path, invalid_where) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "condition", VOLVOXAI_DTYPE_I32,
        scalar_condition, sizeof(scalar_condition)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "x", VOLVOXAI_DTYPE_F32, pair, sizeof(pair)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "y", VOLVOXAI_DTYPE_F32, pair, sizeof(pair)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

static int test_split_preserves_declared_output_order(
    const char* weights_path) {
    const char* path = "/tmp/volvox-split-declared-order-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[12],\"dtype\":\"int32\"}},"
        "\"nodes\":[{\"opType\":\"Split\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out0\":\"slice0\",\"out1\":\"slice1\","
        "\"out2\":\"slice2\",\"out3\":\"slice3\","
        "\"out4\":\"slice4\",\"out5\":\"slice5\","
        "\"out6\":\"slice6\",\"out7\":\"slice7\","
        "\"out8\":\"slice8\",\"out9\":\"slice9\","
        "\"out10\":\"slice10\",\"out11\":\"slice11\"},"
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
        "\"outputs\":[\"slice0\",\"slice1\",\"slice2\",\"slice3\","
        "\"slice4\",\"slice5\",\"slice6\",\"slice7\",\"slice8\","
        "\"slice9\",\"slice10\",\"slice11\"]}";
    const int32_t input[12] = {
        100, 101, 102, 103, 104, 105,
        106, 107, 108, 109, 110, 111,
    };

    CHECK(write_text(path, graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "x", VOLVOXAI_DTYPE_I32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int index = 0; index < 12; index++) {
        char name[32];
        int32_t output = -1;
        snprintf(name, sizeof(name), "slice%d", index);
        CHECK(volvoxai_engine_copy_tensor_raw(
            name, &output, sizeof(output)) == 0);
        CHECK(output == input[index]);
    }
    volvoxai_engine_shutdown();
    remove(path);
    return 0;
}

static int test_consumed_graph_output_survives_arena_reuse(
        const char* weights_path) {
    const char* path = "/tmp/volvox-consumed-output-arena-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1,4],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Mul\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":\"router_logits\"},"
        "\"outputs_shape\":{\"out\":[1,4]}},"
        "{\"opType\":\"ArgMax\",\"inputs\":{\"input\":\"router_logits\"},"
        "\"outputs\":{\"out\":\"selected\"},"
        "\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-1,\"keepdims\":0}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":\"later\"},"
        "\"outputs_shape\":{\"out\":[1,4]}},"
        "{\"opType\":\"ReLU\",\"inputs\":{\"input\":\"later\"},"
        "\"outputs\":{\"out\":\"final\"},"
        "\"outputs_shape\":{\"out\":[1,4]}}],"
        "\"outputs\":[\"router_logits\",\"selected\",\"final\"]}";
    const float input[4] = {1.0f, -2.0f, 3.0f, 0.5f};
    const float expected_router[4] = {1.0f, 4.0f, 9.0f, 0.25f};
    const float expected_final[4] = {2.0f, 0.0f, 6.0f, 1.0f};
    float router[4] = {0};
    float final[4] = {0};
    int32_t selected = -1;

    CHECK(setenv("VOLVOX_ARENA", "1", 1) == 0);
    CHECK(write_text(path, graph) == 0);
    CHECK(volvoxai_engine_init(path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw(
        "x", VOLVOXAI_DTYPE_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "router_logits", router, 4) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
        "selected", &selected, sizeof(selected)) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("final", final, 4) == 0);
    CHECK(memcmp(router, expected_router, sizeof(router)) == 0);
    CHECK(selected == 2);
    CHECK(memcmp(final, expected_final, sizeof(final)) == 0);
    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_ARENA") == 0);
    remove(path);
    return 0;
}

static int test_normalization_shape_contracts(const char* weights_path) {
    const char* path = "/tmp/volvox-normalization-shape-contract.json";
    const char* layernorm_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,4],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"LayerNorm\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,4]}}],\"outputs\":[\"y\"]}";
    const char* rmsnorm_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,4],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"RMSNorm\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"weight\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,4]}}],\"outputs\":[\"y\"]}";
    const float input[4] = {-1.0f, 0.5f, 2.0f, -0.25f};
    const float weight[3] = {1.0f, 0.7f, 1.3f};
    const char* graphs[2] = {layernorm_graph, rmsnorm_graph};
    VolvoxAIEngineOptions options = {
        .backend = VOLVOXAI_BACKEND_CPU,
        .debug = 0,
        .cpu_threads = 1,
    };

    for (int index = 0; index < 2; index++) {
        CHECK(write_text(path, graphs[index]) == 0);
        CHECK(volvoxai_engine_configure(&options) == 0);
        CHECK(volvoxai_engine_init(path, weights_path) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "x", VOLVOXAI_DTYPE_F32, input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "weight", VOLVOXAI_DTYPE_F32, weight, sizeof(weight)) == 0);
        CHECK(volvoxai_engine_forward() != 0);
        volvoxai_engine_shutdown();
    }
    remove(path);
    return 0;
}

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope state_scope;
    const char* graph_path = "/tmp/volvox-sequence-runtime-graph.json";
    const char* weights_path = "/tmp/volvox-sequence-runtime-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
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
        "{\"opType\":\"RoPE\","
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

    CHECK(state != NULL);
    CHECK(vx_engine_state_init(state) == 0);
    state_scope = vx_engine_state_scope_enter(state);
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(write_dummy_weights(weights_path) == 0);
    CHECK(volvoxai_engine_configure(&options) == 0);
    memset(&options, 0, sizeof(options));
    CHECK(volvoxai_engine_get_options(&options) == 0);
    CHECK(options.backend == VOLVOXAI_BACKEND_CPU && options.debug == 0 &&
          options.cpu_threads == 2);
    CHECK(strcmp(volvoxai_engine_backend_name(), "CPU") == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    CHECK(test_required_onnx_operator_graph(weights_path) == 0);
    CHECK(test_split_preserves_declared_output_order(weights_path) == 0);
    CHECK(test_consumed_graph_output_survives_arena_reuse(weights_path) == 0);
    CHECK(test_normalization_shape_contracts(weights_path) == 0);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(state_scope);
    vx_engine_state_deinit(state);
    free(state);
    remove(graph_path);
    remove(weights_path);
    puts("native sequence runtime tests passed");
    return 0;
}
