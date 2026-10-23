/* Weight-only package construction has no model or calibration dependency. */
#include "ptq_authoring.h"
#include "runtime_state.h"
#include "safetensors.h"
#include "cJSON.h"
#include "llm_ops.h"
#include "inference_kernels.h"
#include "vx_training_lifecycle.h"
#include "training_core.h"
#include "training_kernels.h"

#include <float.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(value) do { if (!(value)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #value); \
    return 0; } } while (0)

static const char* graph_source(int transpose, int shared, int source_output) {
    static char graph[4096];
    snprintf(graph, sizeof(graph),
        "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[1,130],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"fc\",\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"shape\":[1,2],\"dtype\":\"float32\"}},"
        "\"params\":{\"weight_layout\":\"%s\"}}%s],"
        "\"outputs\":[\"y\"%s],\"execution\":{\"stateless\":{},\"entrypoints\":[{"
        "\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],"
        "\"inputs\":[\"x\"],\"outputs\":[\"y\"%s]}]}}",
        transpose ? "din_dout" : "dout_din",
        shared ? ",{\"id\":\"keep\",\"opType\":\"Linear\","
            "\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
            "\"outputs\":{\"out\":{\"tensor\":\"z\",\"shape\":[1,2],\"dtype\":\"float32\"}},"
            "\"params\":{\"weight_layout\":\"dout_din\"}}" : "",
        source_output ? ",\"w\"" : "", source_output ? ",\"w\"" : "");
    return graph;
}

static int source_weights(int bits, int transpose, int bf16, int nonfinite, int metadata,
                           unsigned char** bytes, size_t* size) {
    float values[260] = {0};
    uint16_t narrow[260];
    float limit = bits == 4 ? 7.0f : 127.0f;
    const float prefix[] = {0.5f, 1.5f, 2.5f, -0.5f, -1.5f, -2.5f};
    for (int index = 0; index < 6; index++) values[transpose ? index * 2 : index] = prefix[index];
    values[transpose ? 14 : 7] = limit;
    values[transpose ? 256 : 128] = limit * 2;
    values[transpose ? 258 : 129] = -limit * 2;
    if (nonfinite) values[0] = NAN;
    for (int index = 0; index < 260; index++) {
        uint32_t word;
        memcpy(&word, &values[index], sizeof(word));
        narrow[index] = (uint16_t)(word >> 16);
    }
    const float bias[] = {3.0f, -2.0f};
    int shape[] = {transpose ? 130 : 2, transpose ? 2 : 130};
    int bias_shape[] = {2};
    SafetensorsFile file = {0};
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", bf16 ? SAFETENSORS_DTYPE_BF16 : SAFETENSORS_DTYPE_F32,
        shape, 2, bf16 ? (void*)narrow : (void*)values, bf16 ? sizeof(narrow) : sizeof(values)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32,
        bias_shape, 1, bias, sizeof(bias)) == 0);
    if (metadata) CHECK(safetensors_set_metadata_json(&file,
        "{\"author\":\"fixture\",\"format\":\"volvox.ptq.v1\","
        "\"profile_coverage\":\"stale\",\"logical_fingerprint\":\"stale\","
        "\"activation_sources\":\"stale\"}") == 0);
    CHECK(safetensors_serialize(&file, bytes, size) == 0);
    safetensors_free(&file);
    return 1;
}

static int test_values(VxPTQMode mode, int transpose, int bf16, int shared, int source_output) {
    int bits = (mode == VX_PTQ_MODE_W8A8 || mode == VX_PTQ_MODE_W8A16) ? 8 : 4;
    int lanes = 32 / bits;
    int offset = 1 << (bits - 1);
    int limit = offset - 1;
    unsigned char* source = NULL;
    size_t source_size = 0;
    CHECK(source_weights(bits, transpose, bf16, 0, 1, &source, &source_size));
    unsigned char* original = (unsigned char*)malloc(source_size);
    CHECK(original);
    memcpy(original, source, source_size);
    const char* graph = graph_source(transpose, shared, source_output);
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    const char* selected[] = {"fc"};
    config.mode = mode;
    config.selected_nodes = selected;
    config.selected_node_count = 1;
    VxPtqWeightBlob blob = {source, source_size};
    VxPtqAuthored info;
    char* packed_graph = NULL;
    unsigned char* packed_weights = NULL;
    size_t graph_size = 0, weights_size = 0;
    int packed = vx_pack_weights_graph(graph, strlen(graph), &blob, 1, &config,
        &packed_graph, &graph_size, &packed_weights, &weights_size, &info);
    if (packed) fprintf(stderr, "packing error: %s\n", info.message);
    CHECK(packed == 0);
    CHECK(info.observer_count == 0 && info.observers == NULL && info.layer_count == 1);
    CHECK(info.quantized_nodes == 1 && info.retained_float_nodes == (uint64_t)shared);
    CHECK(memcmp(source, original, source_size) == 0);
    CHECK(strstr(packed_graph, "PackedLinear") && !strstr(packed_graph, "quantization"));
    SafetensorsBorrowedBytes view = {0};
    CHECK(safetensors_parse_borrowed_bytes(packed_weights, weights_size, &view) == 0);
    const SafetensorsTensor* words = NULL;
    const SafetensorsTensor* scales = NULL;
    const SafetensorsTensor* bias = NULL;
    int retained = 0;
    for (int index = 0; index < view.tensor_count; index++) {
        const SafetensorsTensor* tensor = &view.tensors[index];
        if (!strcmp(tensor->name, info.layers[0].packed_weight_name)) words = tensor;
        if (!strcmp(tensor->name, info.layers[0].packed_scale_name)) scales = tensor;
        if (!strcmp(tensor->name, "b")) bias = tensor;
        if (!strcmp(tensor->name, "w")) retained++;
    }
    CHECK(retained == (shared || source_output));
    CHECK(words && scales && words->dtype == SAFETENSORS_DTYPE_I32);
    CHECK(words->shape[0] == 2 && words->shape[1] == (130 + lanes - 1) / lanes);
    CHECK(scales->shape[0] == 2 && scales->shape[1] == 2);
    const float expected_scales[] = {1, 2, 1, 1};
    CHECK(memcmp(scales->data, expected_scales, sizeof(expected_scales)) == 0);
    const int prefix[] = {0, 2, 2, 0, -2, -2};
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < words->shape[1] * lanes; column++) {
            uint32_t word;
            memcpy(&word, (const unsigned char*)words->data +
                ((size_t)row * words->shape[1] + column / lanes) * sizeof(word), sizeof(word));
            int actual = (int)((word >> ((column % lanes) * bits)) & ((1u << bits) - 1)) - offset;
            int expected = 0;
            if (!row && column < 6) expected = prefix[column];
            if (!row && (column == 7 || column == 128)) expected = limit;
            if (!row && column == 129) expected = -limit;
            CHECK(actual == expected);
        }
    }
    CHECK(!view.metadata_json || !strstr(view.metadata_json, "profile_coverage"));
    CHECK(view.metadata_json && strstr(view.metadata_json, "\"author\":\"fixture\""));
    CHECK(!strstr(view.metadata_json, "logical_fingerprint"));
    CHECK(!strstr(view.metadata_json, "activation_sources"));
    CHECK(!strstr(view.metadata_json, "volvox.ptq.v1"));
    /* Execute the authored packed representation independently of calibration.
     * The second row also proves dynamic A8 scales are per input row. */
    float activations[260], output[4] = {0};
    for (int index = 0; index < 260; index++)
        activations[index] = index < 130 ? 1.0f : -2.0f;
    VxLLMOpArgs args = {
        .kind = VX_OP_PACKED_LINEAR, .batch = 1, .sequence = 2,
        .key_dim = 130, .value_dim = 2, .groups = bits, .rate = 128,
        .normalize = (mode == VX_PTQ_MODE_W8A8 || mode == VX_PTQ_MODE_W4A8) ? 8 : 16,
        .x = {activations, words->data, scales->data, bias ? bias->data : NULL},
        .y = {output}, .x_elements = {260, words->nbytes / 4u, 4, 2},
        .y_elements = {4},
    };
    CHECK(bias && vx_llm_forward_f32(&args) == 0);
    CHECK(fabsf(output[0] - (float)(limit + 3)) < 0.0001f);
    CHECK(output[1] == -2.0f && output[3] == -2.0f);
    CHECK(fabsf(output[2] - (float)(-2 * limit + 3)) < 0.0001f);
    safetensors_borrowed_bytes_free(&view);
    vx_ptq_authored_free(&info);
    free(source); free(original); free(packed_graph); free(packed_weights);
    return 1;
}

static int test_refusals(void) {
    unsigned char* source = NULL;
    size_t source_size = 0;
    CHECK(source_weights(4, 0, 0, 1, 1, &source, &source_size));
    VxPtqWeightBlob blob = {source, source_size};
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    config.mode = VX_PTQ_MODE_W4A8;
    const char* graph = graph_source(0, 0, 0);
    char* result_graph = NULL;
    unsigned char* result_weights = NULL;
    size_t graph_size = 0, weights_size = 0;
    VxPtqAuthored info;
    CHECK(vx_pack_weights_graph(graph, strlen(graph), &blob, 1, &config,
        &result_graph, &graph_size, &result_weights, &weights_size, &info) != 0);
    CHECK(info.status == VX_STATUS_INVALID_GRAPH && !result_graph && !result_weights);
    free(source);
    CHECK(source_weights(4, 0, 0, 0, 1, &source, &source_size));
    blob.bytes = source;
    blob.byte_count = source_size;
    config.mode = (VxPTQMode)99;
    CHECK(vx_pack_weights_graph(graph, strlen(graph), &blob, 1, &config,
        &result_graph, &graph_size, &result_weights, &weights_size, &info) != 0);
    CHECK(info.status == VX_STATUS_INVALID_ARGUMENT);
    config.mode = VX_PTQ_MODE_W4A8;
    config.reduce_range = 1;
    CHECK(vx_pack_weights_graph(graph, strlen(graph), &blob, 1, &config,
        &result_graph, &graph_size, &result_weights, &weights_size, &info) != 0);
    CHECK(info.status == VX_STATUS_INVALID_ARGUMENT);
    config.reduce_range = 0;
    CHECK(vx_pack_weights_graph(graph, strlen(graph), &blob, 2, &config,
        &result_graph, &graph_size, &result_weights, &weights_size, &info) != 0);
    CHECK(info.status == VX_STATUS_INVALID_ARGUMENT);
    char malformed[8192];
    snprintf(malformed, sizeof(malformed), "%s trailing", graph);
    CHECK(vx_pack_weights_graph(malformed, strlen(malformed), &blob, 1, &config,
        &result_graph, &graph_size, &result_weights, &weights_size, &info) != 0);
    CHECK(info.status == VX_STATUS_INVALID_GRAPH);
    cJSON* changed = cJSON_Parse(graph);
    cJSON* shape = cJSON_GetObjectItemCaseSensitive(cJSON_GetObjectItemCaseSensitive(
        cJSON_GetObjectItemCaseSensitive(changed, "inputs"), "x"), "shape");
    CHECK(cJSON_ReplaceItemInArray(shape, 1, cJSON_CreateNumber(129)));
    char* changed_text = cJSON_PrintUnformatted(changed);
    CHECK(vx_pack_weights_graph(changed_text, strlen(changed_text), &blob, 1, &config,
        &result_graph, &graph_size, &result_weights, &weights_size, &info) != 0);
    CHECK(info.status == VX_STATUS_INVALID_GRAPH);
    cJSON_Delete(changed); free(changed_text); free(source);
    return 1;
}

static int test_plan_owner(int packed) {
    unsigned char* source = NULL;
    size_t source_size = 0;
    CHECK(source_weights(8, 0, 0, 0, 0, &source, &source_size));
    const char* graph = graph_source(0, 0, 0);
    VxPtqWeightBlob blob = {source, source_size};
    VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
    VxQuantizationPlanOptions options = VX_QUANTIZATION_PLAN_OPTIONS_INIT;
    options.source_graph = graph;
    options.source_graph_size = strlen(graph);
    options.weights = &blob;
    options.weight_count = 1;
    options.config = &config;
    options.packed_weights = packed;
    options.cpu_threads = 1;
    VxQuantizationPlan* plan = NULL;
    VxReport report = VX_REPORT_INIT;
    VxStatus created = vx_quantization_plan_create(&options, &plan, &report);
    if (created != VX_STATUS_OK) fprintf(stderr, "plan create: %s\n", report.message);
    CHECK(created == VX_STATUS_OK && plan);
    free(source);
    CHECK(vx_quantization_plan_requires_calibration(plan) == !packed);
    CHECK((vx_quantization_plan_static(plan) != NULL) == !packed);
    CHECK(vx_quantization_plan_input_count(plan) == (size_t)!packed);
    uint64_t quantized = 0, retained = 99;
    vx_quantization_plan_counts(plan, &quantized, &retained);
    CHECK(quantized == 1 && retained == 0);
    vx_quantization_plan_retain(plan);
    vx_quantization_plan_release(plan);
    VxPTQPackageOptions output = VX_PTQ_PACKAGE_OPTIONS_INIT;
    unsigned char* graph_bytes = NULL;
    unsigned char* weights_bytes = NULL;
    size_t graph_size = 0, weights_size = 0;
    output.graph_bytes = &graph_bytes;
    output.graph_size = &graph_size;
    output.weights_bytes = &weights_bytes;
    output.weights_size = &weights_size;
    if (!packed) {
        CHECK(vx_quantization_plan_write_package(plan, &output, &report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(!graph_bytes && !weights_bytes);
        float values[130];
        for (int index = 0; index < 130; index++) values[index] = 1;
        VxTensorBinding input = VX_TENSOR_BINDING_INIT;
        input.name = "x";
        input.rank = 2;
        input.shape[0] = 1;
        input.shape[1] = 130;
        input.data = values;
        input.byte_size = sizeof(values);
        VxPTQCalibrationBatch batch = VX_PTQ_CALIBRATION_BATCH_INIT;
        batch.profile_name = "default";
        batch.sample_name = "one";
        batch.sample_count = 1;
        batch.inputs = &input;
        batch.input_count = 1;
        VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
        CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan), &batch, &info, &report) == VX_STATUS_OK);
        CHECK(info.coverage_complete && info.calibration_samples == 1);
    }
    VxStatus exported = vx_quantization_plan_write_package(plan, &output, &report);
    if (exported != VX_STATUS_OK) fprintf(stderr, "plan export: %s\n", report.message);
    CHECK(exported == VX_STATUS_OK && graph_size && weights_size);
    unsigned char graph_first = graph_bytes[0], weights_first = weights_bytes[0];
    graph_bytes[0] ^= 0xff;
    weights_bytes[0] ^= 0xff;
    free(graph_bytes); free(weights_bytes);
    graph_bytes = weights_bytes = NULL;
    CHECK(vx_quantization_plan_write_package(plan, &output, &report) == VX_STATUS_OK);
    CHECK(graph_bytes[0] == graph_first && weights_bytes[0] == weights_first);
    vx_quantization_plan_release(plan);
    /* Exported snapshots survive the plan's release. */
    CHECK(graph_bytes[0] == '{' && weights_size > 8);
    free(graph_bytes); free(weights_bytes);
    return 1;
}

static const SafetensorsTensor* find_tensor(const SafetensorsBorrowedBytes* view,
                                            const char* name) {
    for (int index = 0; index < view->tensor_count; index++)
        if (!strcmp(view->tensors[index].name, name)) return &view->tensors[index];
    return NULL;
}

static const SafetensorsTensor* affine_scale(const SafetensorsBorrowedBytes* view,
                                             cJSON* root, const char* name) {
    cJSON* tensors = cJSON_GetObjectItemCaseSensitive(
        cJSON_GetObjectItemCaseSensitive(root, "quantization"), "tensors");
    cJSON* descriptor = cJSON_GetObjectItemCaseSensitive(tensors, name);
    const char* scale = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(
        descriptor, "scale_tensor"));
    return scale ? find_tensor(view, scale) : NULL;
}

/* Matches typed_ptq.py: reduced weights occupy [-64,64], zero rows keep scale
 * one, ties round to even, and I32 bias uses the changed accumulator scale. */
static int test_static_reduced_range(void) {
    const float values[260] = {64.0f, 1.5f, -2.5f};
    const float biases[] = {1.25f, -2.0f};
    const int shape[] = {2, 130}, bias_shape[] = {2};
    SafetensorsFile source = {0};
    unsigned char* source_bytes = NULL;
    size_t source_size = 0;
    CHECK(safetensors_init_empty(&source, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&source, "w", SAFETENSORS_DTYPE_F32,
        shape, 2, values, sizeof(values)) == 0);
    CHECK(safetensors_add_tensor(&source, "b", SAFETENSORS_DTYPE_F32,
        bias_shape, 1, biases, sizeof(biases)) == 0);
    CHECK(safetensors_serialize(&source, &source_bytes, &source_size) == 0);
    safetensors_free(&source);
    const char* graph = graph_source(0, 0, 0);
    int8_t actual[2][2] = {{0}};
    for (int reduced = 0; reduced < 2; reduced++) {
        VxPtqWeightBlob blob = {source_bytes, source_size};
        VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
        config.reduce_range = reduced;
        VxQuantizationPlanOptions options = VX_QUANTIZATION_PLAN_OPTIONS_INIT;
        options.source_graph = graph;
        options.source_graph_size = strlen(graph);
        options.weights = &blob;
        options.weight_count = 1;
        options.config = &config;
        options.cpu_threads = 1;
        VxQuantizationPlan* plan = NULL;
        VxReport report = VX_REPORT_INIT;
        CHECK(vx_quantization_plan_create(&options, &plan, &report) == VX_STATUS_OK);
        float calibration[130] = {0.1f};
        calibration[129] = 127;
        VxTensorBinding input = VX_TENSOR_BINDING_INIT;
        input.name = "x";
        input.rank = 2;
        input.shape[0] = 1;
        input.shape[1] = 130;
        input.data = calibration;
        input.byte_size = sizeof(calibration);
        VxPTQCalibrationBatch batch = VX_PTQ_CALIBRATION_BATCH_INIT;
        batch.profile_name = "default";
        batch.sample_name = "range";
        batch.sample_count = 1;
        batch.inputs = &input;
        batch.input_count = 1;
        VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
        CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan),
            &batch, &info, &report) == VX_STATUS_OK);
        unsigned char* graph_bytes = NULL;
        unsigned char* weights_bytes = NULL;
        size_t graph_size = 0, weights_size = 0;
        VxPTQPackageOptions output = VX_PTQ_PACKAGE_OPTIONS_INIT;
        output.graph_bytes = &graph_bytes;
        output.graph_size = &graph_size;
        output.weights_bytes = &weights_bytes;
        output.weights_size = &weights_size;
        CHECK(vx_quantization_plan_write_package(plan, &output, &report) == VX_STATUS_OK);
        vx_quantization_plan_release(plan);
        cJSON* root = cJSON_ParseWithLength((const char*)graph_bytes, graph_size);
        CHECK(root);
        cJSON* node = NULL;
        cJSON* candidate;
        cJSON_ArrayForEach(candidate, cJSON_GetObjectItemCaseSensitive(root, "nodes")) {
            const char* kind = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(candidate, "opType"));
            if (kind && !strcmp(kind, "QLinear")) node = candidate;
        }
        CHECK(node);
        cJSON* inputs = cJSON_GetObjectItemCaseSensitive(node, "inputs");
        const char* weight_name = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(inputs, "weight"));
        const char* bias_name = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(inputs, "bias"));
        const char* input_name = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(inputs, "input"));
        cJSON* out = cJSON_GetObjectItemCaseSensitive(
            cJSON_GetObjectItemCaseSensitive(node, "outputs"), "out");
        const char* output_name = cJSON_GetStringValue(cJSON_GetObjectItemCaseSensitive(out, "tensor"));
        CHECK(weight_name && bias_name && input_name && output_name);
        SafetensorsBorrowedBytes view = {0};
        CHECK(safetensors_parse_borrowed_bytes(weights_bytes, weights_size, &view) == 0);
        const SafetensorsTensor* weight = find_tensor(&view, weight_name);
        const SafetensorsTensor* bias = find_tensor(&view, bias_name);
        const SafetensorsTensor* scales = affine_scale(&view, root, weight_name);
        const SafetensorsTensor* input_scale = affine_scale(&view, root, input_name);
        const SafetensorsTensor* output_scale = affine_scale(&view, root, output_name);
        CHECK(weight && bias && scales && input_scale && output_scale);
        CHECK(weight->dtype == SAFETENSORS_DTYPE_I8 && bias->dtype == SAFETENSORS_DTYPE_I32);
        const int8_t* codes = weight->data;
        int32_t bias_codes[2];
        float weight_scales[2], input_scale_value, output_scale_value;
        /* SafeTensors byte views do not promise alignment for every tensor. */
        memcpy(bias_codes, bias->data, sizeof(bias_codes));
        memcpy(weight_scales, scales->data, sizeof(weight_scales));
        memcpy(&input_scale_value, input_scale->data, sizeof(input_scale_value));
        memcpy(&output_scale_value, output_scale->data, sizeof(output_scale_value));
        CHECK(codes[0] == (reduced ? 64 : 127));
        CHECK(codes[1] == (reduced ? 2 : 3));
        CHECK(codes[2] == (reduced ? -2 : -5));
        CHECK(weight_scales[0] == (reduced ? 1.0f : 64.0f / 127.0f));
        CHECK(weight_scales[1] == 1.0f && codes[130] == 0);
        CHECK(bias_codes[0] == (reduced ? 1 : 2) && bias_codes[1] == -2);
        const int8_t activation[130] = {0, 1};
        const int32_t zero_points[] = {0, 0};
        CHECK(qlinear_i8u8(activation, codes, bias_codes, weight_scales,
            zero_points, actual[reduced], 1, 130, 2,
            input_scale_value, 0, output_scale_value,
            0, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(actual[reduced][0] == (reduced ? 50 : 42) && actual[reduced][1] == -33);
        safetensors_borrowed_bytes_free(&view);
        cJSON_Delete(root);
        free(graph_bytes); free(weights_bytes);
    }
    CHECK(actual[0][0] != actual[1][0]);
    free(source_bytes);
    return 1;
}

static int test_weight_range_validation(void) {
    const float source[] = {-64, 64};
    const int32_t shape[] = {1, 2};
    const int32_t invalid_bounds[] = {INT32_MIN, -1, 0, 63, 65, 126, 128, INT32_MAX};
    for (size_t index = 0; index < sizeof(invalid_bounds) / sizeof(invalid_bounds[0]); index++) {
        int8_t output[] = {11, 22};
        float scale = 23;
        uint64_t saturation = 99;
        CHECK(volvoxai_ptq_pack_weight_i8_range(source, shape, 2, 0,
            output, &scale, 1, invalid_bounds[index], &saturation) != 0);
        CHECK(output[0] == 11 && output[1] == 22 && scale == 23 && saturation == 99);
    }
    int8_t original[2], explicit_range[2];
    float original_scale = 0, explicit_scale = 0;
    uint64_t original_saturation = 99, explicit_saturation = 99;
    CHECK(volvoxai_ptq_pack_weight_i8(source, shape, 2, 0, original,
        &original_scale, 1, &original_saturation) == 0);
    CHECK(volvoxai_ptq_pack_weight_i8_range(source, shape, 2, 0, explicit_range,
        &explicit_scale, 1, 127, &explicit_saturation) == 0);
    CHECK(memcmp(original, explicit_range, sizeof(original)) == 0 &&
        original_scale == explicit_scale && original_saturation == explicit_saturation);
    return 1;
}

static int test_grouped_fitting_contract(void) {
    float input[4] = {7.1f, 7.6f, -8.5f, 0.0f};
    float preserved[1] = {1.0f}, scales[1];
    uint32_t codes[1];
    uint64_t saturation = 99;
    CHECK(volvoxai_training_quantize_grouped_weight(input, 1, 4, 0, 4, 128,
        VX_WEIGHT_ALGORITHM_RTN, NULL, preserved, codes, scales, &saturation));
    CHECK(saturation == 1 && scales[0] == 1.0f);
    CHECK((codes[0] & 0xffffu) == 0x80ffu);
    for (uint32_t group = 0; group <= 512; group += 16) {
        if (group == 32 || group == 64 || group == 128 || group == 256) continue;
        CHECK(!volvoxai_training_quantize_grouped_weight(input, 1, 4, 0, 4, group,
            VX_WEIGHT_ALGORITHM_RTN, NULL, NULL, codes, scales, NULL));
    }
    CHECK(!volvoxai_training_quantize_grouped_weight(input, 1, 4, 0, 4, 128,
        VX_WEIGHT_ALGORITHM_RTN, NULL, NULL, (uint32_t*)input, scales, NULL));
    CHECK(!volvoxai_training_quantize_grouped_weight(input, 1, 4, 0, 4, 128,
        VX_WEIGHT_ALGORITHM_RTN, NULL, NULL, codes, input, NULL));
    CHECK(!volvoxai_training_quantize_grouped_weight(input, 1, 4, 0, 4, 128,
        VX_WEIGHT_ALGORITHM_RTN, NULL, preserved, codes, preserved, NULL));
    const float weights[2] = {0.1f, 0.2f};
    const double importance[2] = {1.0, 9.0};
    double unweighted[3] = {0}, weighted[3] = {0}, output_error[3] = {0};
    for (int algorithm = VX_WEIGHT_ALGORITHM_RTN; algorithm <= VX_WEIGHT_ALGORITHM_IMPORTANCE; algorithm++) {
        CHECK(volvoxai_training_quantize_grouped_weight(weights, 1, 2, 0, 4, 128,
            algorithm, importance, NULL, codes, scales, NULL));
        double output_delta = 0;
        for (int c = 0; c < 2; c++) {
            int code = (int)((codes[0] >> (4 * c)) & 15u) - 8;
            float reconstructed = (float)code * scales[0];
            double delta = (double)weights[c] - reconstructed;
            unweighted[algorithm] += delta * delta;
            weighted[algorithm] += importance[c] * delta * delta;
            output_delta += delta * (c == 0 ? 1.0 : 3.0);
        }
        output_error[algorithm] = output_delta * output_delta;
    }
    CHECK(unweighted[VX_WEIGHT_ALGORITHM_MSE_CLIP] <= unweighted[VX_WEIGHT_ALGORITHM_RTN]);
    CHECK(weighted[VX_WEIGHT_ALGORITHM_IMPORTANCE] <= weighted[VX_WEIGHT_ALGORITHM_MSE_CLIP]);
    /* Diagonal fitting is not an actual output-MSE guarantee: correlated
     * channels contribute cross terms absent from the sufficient statistic. */
    CHECK(output_error[VX_WEIGHT_ALGORITHM_IMPORTANCE] > output_error[VX_WEIGHT_ALGORITHM_MSE_CLIP]);
    return 1;
}

/* Decode individual bits rather than using the production lane arithmetic:
 * I3 columns 10/21 cross words, and columns 2/5 cross bytes. */
static unsigned test_dense_code(const uint8_t* bytes, size_t column, unsigned bits) {
    unsigned code = 0;
    for (unsigned bit = 0; bit < bits; bit++) {
        size_t address = column * bits + bit;
        code |= (unsigned)((bytes[address / 8] >> (address % 8)) & 1u) << bit;
    }
    return code;
}

static int test_lowbit_dense_storage_and_fitting(void) {
    const unsigned columns = 35, rows = 4;
    float source[4 * 35], scales[8], preserved[8];
    double importance[2 * 35];
    uint32_t words[4 * 5];
    uint8_t bytes[4 * 18];
    for (unsigned i = 0; i < 8; i++) preserved[i] = 1.0f;
    for (unsigned k = 0; k < columns; k++) {
        importance[k] = 1.0 + (k % 3) * 100.0;
        importance[columns + k] = 1.0 + (k % 5 == 0) * 10000.0;
    }
    for (unsigned bits = 2; bits <= 4; bits++) {
        unsigned zero = 1u << (bits - 1), mask = (1u << bits) - 1u;
        unsigned word_count = (columns * bits + 31u) / 32u;
        unsigned row_bytes = (columns * bits + 7u) / 8u;
        for (unsigned r = 0; r < rows; r++)
            for (unsigned k = 0; k < columns; k++)
                source[r * columns + k] = (float)((int)((k + 3 * r) & mask) - (int)zero);
        CHECK(volvoxai_training_quantize_grouped_weight(source, rows, columns, 0,
            bits, 32, VX_WEIGHT_ALGORITHM_RTN, NULL, preserved, words, scales, NULL));
        for (unsigned r = 0; r < rows; r++) {
            const uint8_t* row = (const uint8_t*)&words[r * word_count];
            for (unsigned k = 0; k < columns; k++)
                CHECK((int)test_dense_code(row, k, bits) - (int)zero == (int)source[r * columns + k]);
            if (bits == 3) CHECK((words[(r + 1) * word_count - 1] >> (columns * bits % 32)) == 0);
            else for (unsigned k = columns; k < word_count * 32u / bits; k++)
                CHECK(test_dense_code(row, k, bits) == zero);
        }
        /* A signed two-bit MSE grid must reach -2, not accidentally fit a
         * three-level quantizer through an I4-style narrow clipping grid. */
        if (bits == 2) {
            CHECK(volvoxai_training_quantize_grouped_weight(source, rows, columns, 0,
                bits, 32, VX_WEIGHT_ALGORITHM_MSE_CLIP, NULL, NULL, words, scales, NULL));
            CHECK(scales[0] == 1.0f && (words[0] & 3u) == 0);
        }
        for (unsigned r = 0; r < rows; r++)
            for (unsigned k = 0; k < columns; k++)
                source[r * columns + k] = (float)((int)((k * 17 + r * 3) % 29) - 14) * .071f;
        CHECK(!volvoxai_ptq_pack_conv_weight(source, rows, columns, bits,
            VX_WEIGHT_ALGORITHM_IMPORTANCE, importance, 2, bytes, scales));
        for (unsigned r = 0; r < rows; r++) {
            float maximum = 0.0f;
            for (unsigned k = 0; k < columns; k++)
                maximum = fmaxf(maximum, fabsf(source[r * columns + k]));
            float base = maximum / (float)(zero - 1), expected_scale = base;
            double minimum_loss = INFINITY;
            for (int percent = 100; percent >= (bits == 2 ? 1 : 50); percent--) {
                float candidate = base * ((float)percent / 100.0f);
                double loss = 0;
                for (unsigned k = 0; k < columns; k++) {
                    float code = nearbyintf(source[r * columns + k] / candidate);
                    code = fmaxf(-(float)zero, fminf((float)(zero - 1), code));
                    double delta = (double)source[r * columns + k] - (float)(code * candidate);
                    loss += delta * delta * importance[(r / 2) * columns + k];
                }
                if (loss < minimum_loss) { minimum_loss = loss; expected_scale = candidate; }
            }
            CHECK(scales[r] == expected_scale);
            for (unsigned k = 0; k < columns; k++) {
                int expected = (int)nearbyintf(source[r * columns + k] / expected_scale);
                if (expected < -(int)zero) expected = -(int)zero;
                if (expected >= (int)zero) expected = (int)zero - 1;
                CHECK((int)test_dense_code(bytes + r * row_bytes, k, bits) - (int)zero == expected);
            }
            if (bits == 3) CHECK((bytes[(r + 1) * row_bytes - 1] >> (columns * bits % 8)) == 0);
            else for (unsigned k = columns; k < row_bytes * 8u / bits; k++)
                CHECK(test_dense_code(bytes + r * row_bytes, k, bits) == zero);
        }
    }
    return 1;
}

static int test_extreme_finite_weight_fitting(void) {
    union { uint32_t bits; float value; } repeated = {UINT32_C(0x7f028f5b)};
    float source[128], scale;
    double importance[128];
    uint32_t codes[32];
    source[0] = -FLT_MAX;
    for (int k = 0; k < 128; ++k) {
        if (k) source[k] = repeated.value;
        importance[k] = 1.0 + k % 7;
    }
    const uint32_t formats[] = {2, 3, 4, 8};
    for (size_t format = 0; format < sizeof(formats) / sizeof(formats[0]); format++) {
        uint32_t bits = formats[format];
        double error[3] = {0}, weighted[3] = {0};
        for (int algorithm = VX_WEIGHT_ALGORITHM_RTN;
             algorithm <= VX_WEIGHT_ALGORITHM_IMPORTANCE; ++algorithm) {
            CHECK(volvoxai_training_quantize_grouped_weight(source, 1, 128, 0, bits, 128,
                algorithm, importance, NULL, codes, &scale, NULL));
            CHECK(isfinite(scale) && scale > 0.0f);
            for (uint32_t k = 0; k < 128; ++k) {
                int q = (int)test_dense_code((const uint8_t*)codes, k, bits) -
                    (int)(1u << (bits - 1u));
                float reconstructed = (float)q * scale;
                CHECK(isfinite(reconstructed));
                double delta = (double)source[k] - reconstructed;
                error[algorithm] += delta * delta;
                weighted[algorithm] += importance[k] * delta * delta;
            }
        }
        CHECK(error[VX_WEIGHT_ALGORITHM_MSE_CLIP] <= error[VX_WEIGHT_ALGORITHM_RTN]);
        CHECK(weighted[VX_WEIGHT_ALGORITHM_IMPORTANCE] <= weighted[VX_WEIGHT_ALGORITHM_RTN]);
    }
    /* The same endpoint correction applies to per-channel package authoring
     * and tensor conversion, including both signs and the reduced range. */
    const float endpoints[] = {-FLT_MAX, FLT_MAX};
    const int32_t shape[] = {1, 2};
    int8_t channel_codes[2];
    for (int bound = 64; bound <= 127; bound += 63) {
        CHECK(!volvoxai_ptq_pack_weight_i8_range(endpoints, shape, 2, 0,
            channel_codes, &scale, 1, bound, NULL));
        CHECK(channel_codes[0] == -bound && channel_codes[1] == bound);
        CHECK(isfinite((float)channel_codes[0] * scale));
        CHECK(isfinite((float)channel_codes[1] * scale));
    }
    CHECK(volvoxai_training_quantize_weight_f32_to_i8(endpoints, channel_codes,
        &scale, 1, 2, 0, VOLVOXAI_TRAINING_WEIGHT_SCALES_RECOMPUTE, NULL));
    CHECK(isfinite((float)channel_codes[0] * scale) &&
        isfinite((float)channel_codes[1] * scale));
    /* Supplied scales are a caller-owned numerical choice and stay exact. */
    const float preserved = FLT_MAX / 127.0f;
    scale = preserved;
    CHECK(volvoxai_training_quantize_weight_f32_to_i8(endpoints, channel_codes,
        &scale, 1, 2, 0, VOLVOXAI_TRAINING_WEIGHT_SCALES_PRESERVE, NULL));
    CHECK(scale == preserved);
    CHECK(volvoxai_training_quantize_grouped_weight(source, 1, 128, 0, 8, 128,
        VX_WEIGHT_ALGORITHM_RTN, NULL, &preserved, codes, &scale, NULL));
    CHECK(scale == preserved);
    return 1;
}

static int test_mixed_rules_and_shared_weights(void) {
    unsigned char* source = NULL;
    size_t source_size = 0;
    CHECK(source_weights(4, 0, 0, 0, 0, &source, &source_size));
    const char* graph = graph_source(0, 1, 0);
    VxPtqWeightBlob blob = {source, source_size};
    for (int variant = 0; variant < 11; variant++) {
        VxQuantizationRule rules[2] = {
            {.selector = VX_QUANTIZATION_NODE_ID, .match = "fc", .bits = 4, .group_size = 128,
             .algorithm = VX_WEIGHT_ALGORITHM_RTN},
            {.selector = VX_QUANTIZATION_NODE_GLOB, .match = "*", .bits = 4, .group_size = 128,
             .algorithm = VX_WEIGHT_ALGORITHM_RTN}
        };
        if (variant == 0) rules[1].bits = 8;
        if (variant == 2 || variant == 4) rules[1].keep_float = 1;
        if (variant == 3) rules[1].algorithm = VX_WEIGHT_ALGORITHM_MSE_CLIP;
        if (variant >= 5) rules[1].algorithm = VX_WEIGHT_ALGORITHM_IMPORTANCE;
        if (variant == 6 || variant == 8) rules[0].algorithm = VX_WEIGHT_ALGORITHM_IMPORTANCE;
        if (variant == 7) rules[1].algorithm = VX_WEIGHT_ALGORITHM_RTN;
        if (variant == 7 || variant == 8) {
            rules[0].group_size = 32;
            rules[1].group_size = variant == 7 ? 64 : 256;
        }
        if (variant >= 9) {
            rules[0].bits = variant == 9 ? 2u : 3u;
            rules[0].algorithm = VX_WEIGHT_ALGORITHM_IMPORTANCE;
        }
        VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
        config.mode = VX_PTQ_MODE_W4A8;
        config.weight_only = 1;
        config.rules = rules;
        config.rule_count = 2;
        if (variant == 4) config.weight_algorithm = VX_WEIGHT_ALGORITHM_IMPORTANCE;
        VxQuantizationPlanOptions options = VX_QUANTIZATION_PLAN_OPTIONS_INIT;
        options.source_graph = graph;
        options.source_graph_size = strlen(graph);
        options.weights = &blob; options.weight_count = 1;
        options.config = &config; options.packed_weights = 1;
        VxQuantizationPlan* plan = NULL;
        VxReport report = VX_REPORT_INIT;
        CHECK(vx_quantization_plan_create(&options, &plan, &report) == VX_STATUS_OK && plan);
        int calibrates = variant == 5 || variant == 6 || variant >= 8;
        CHECK(vx_quantization_plan_requires_calibration(plan) == calibrates);
        if (calibrates) {
            float values[130];
            for (int i = 0; i < 130; i++) values[i] = (float)(i + 1) / 130.0f;
            VxTensorBinding input = VX_TENSOR_BINDING_INIT;
            input.name = "x"; input.rank = 2; input.shape[0] = 1; input.shape[1] = 130;
            input.data = values; input.byte_size = sizeof(values);
            VxPTQCalibrationBatch batch = VX_PTQ_CALIBRATION_BATCH_INIT;
            batch.profile_name = "default"; batch.sample_name = "mixed"; batch.sample_count = 1;
            batch.inputs = &input; batch.input_count = 1;
            VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
            CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan), &batch, &info, &report) == VX_STATUS_OK);
            CHECK(info.tensor_count == 0 && info.coverage_complete);
        }
        size_t node_count;
        const VxQuantizationNode* nodes = vx_quantization_plan_nodes(plan, &node_count);
        CHECK(node_count == 2 && !nodes[0].keep_float && nodes[0].bits == rules[0].bits);
        CHECK(nodes[0].group_size == rules[0].group_size);
        CHECK(nodes[1].group_size == (rules[1].keep_float ? 128u : rules[1].group_size));
        CHECK(nodes[1].keep_float == (variant == 2 || variant == 4));
        CHECK(nodes[1].bits == (variant == 0 ? 8 : 4));
        VxPTQPackageOptions output = VX_PTQ_PACKAGE_OPTIONS_INIT;
        unsigned char *graph_bytes = NULL, *weight_bytes = NULL;
        size_t graph_size = 0, weight_size = 0;
        output.graph_bytes = &graph_bytes; output.graph_size = &graph_size;
        output.weights_bytes = &weight_bytes; output.weights_size = &weight_size;
        CHECK(vx_quantization_plan_write_package(plan, &output, &report) == VX_STATUS_OK);
        cJSON* root = cJSON_Parse((const char*)graph_bytes);
        cJSON* graph_nodes = cJSON_GetObjectItemCaseSensitive(root, "nodes");
        cJSON* first = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(graph_nodes, 0), "inputs");
        cJSON* second = cJSON_GetObjectItemCaseSensitive(cJSON_GetArrayItem(graph_nodes, 1), "inputs");
        const char* first_weight = cJSON_GetObjectItemCaseSensitive(first, "weight")->valuestring;
        const char* second_weight = cJSON_GetObjectItemCaseSensitive(second, "weight")->valuestring;
        CHECK((strcmp(first_weight, second_weight) == 0) == (variant == 1 || variant == 6));
        SafetensorsBorrowedBytes view = {0};
        CHECK(safetensors_parse_borrowed_bytes(weight_bytes, weight_size, &view) == 0);
        if (variant >= 7) for (int node = 0; node < 2; ++node) {
            cJSON* entry = cJSON_GetArrayItem(graph_nodes, node);
            cJSON* params = cJSON_GetObjectItemCaseSensitive(entry, "params");
            CHECK(cJSON_GetObjectItemCaseSensitive(params, "group_size")->valueint == (int)rules[node].group_size);
            cJSON* inputs = cJSON_GetObjectItemCaseSensitive(entry, "inputs");
            const char* scale_name = cJSON_GetObjectItemCaseSensitive(inputs, "scale")->valuestring;
            int found = 0;
            for (int tensor = 0; tensor < view.tensor_count; ++tensor)
                if (!strcmp(view.tensors[tensor].name, scale_name)) {
                    CHECK(view.tensors[tensor].shape[1] == (int)((130 + rules[node].group_size - 1) / rules[node].group_size));
                    found = 1;
                }
            CHECK(found);
        }
        int retained_source = 0;
        for (int i = 0; i < view.tensor_count; i++)
            if (!strcmp(view.tensors[i].name, "w")) retained_source = 1;
        CHECK(retained_source == (variant == 2 || variant == 4));
        safetensors_borrowed_bytes_free(&view);
        cJSON_Delete(root);
        vx_quantization_plan_release(plan);
        free(graph_bytes); free(weight_bytes);
    }
    free(source);
    return 1;
}

static int test_importance_and_mse_calibration(void) {
    unsigned char* source = NULL;
    size_t source_size = 0;
    CHECK(source_weights(4, 0, 0, 0, 0, &source, &source_size));
    const char* graph = graph_source(0, 0, 0);
    VxPtqWeightBlob blob = {source, source_size};
    const VxPTQMode modes[] = {VX_PTQ_MODE_W8A8, VX_PTQ_MODE_W4A8, VX_PTQ_MODE_W4A16, VX_PTQ_MODE_W8A16, VX_PTQ_MODE_W2A8, VX_PTQ_MODE_W3A8};
    for (size_t m = 0; m < sizeof(modes) / sizeof(modes[0]); m++) {
        VxPtqAuthoringConfig config = VX_PTQ_AUTHORING_CONFIG_INIT;
        config.mode = modes[m];
        config.weight_algorithm = VX_WEIGHT_ALGORITHM_IMPORTANCE;
        VxQuantizationPlanOptions options = VX_QUANTIZATION_PLAN_OPTIONS_INIT;
        options.source_graph = graph;
        options.source_graph_size = strlen(graph);
        options.weights = &blob;
        options.weight_count = 1;
        options.config = &config;
        options.packed_weights = 0;
        options.cpu_threads = 1;
        VxQuantizationPlan* plan = NULL;
        VxReport report = VX_REPORT_INIT;
        CHECK(vx_quantization_plan_create(&options, &plan, &report) == VX_STATUS_OK && plan);
        CHECK(vx_quantization_plan_requires_calibration(plan) == 1);
        float values[130];
        for (int index = 0; index < 130; index++) values[index] = (float)(index + 1) * 0.05f;
        VxTensorBinding input = VX_TENSOR_BINDING_INIT;
        input.name = "x";
        input.rank = 2;
        input.shape[0] = 1;
        input.shape[1] = 130;
        input.data = values;
        input.byte_size = sizeof(values);
        VxPTQCalibrationBatch batch = VX_PTQ_CALIBRATION_BATCH_INIT;
        batch.profile_name = "default";
        batch.sample_name = "imp_sample";
        batch.sample_count = 1;
        batch.inputs = &input;
        batch.input_count = 1;
        VxPTQPlanInfo info = VX_PTQ_PLAN_INFO_INIT;
        CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan), &batch, &info, &report) == VX_STATUS_OK);
        CHECK(info.coverage_complete && info.calibration_samples == 1);
        VxPTQPackageOptions output = VX_PTQ_PACKAGE_OPTIONS_INIT;
        unsigned char* graph_bytes = NULL;
        unsigned char* weights_bytes = NULL;
        size_t graph_size = 0, weights_size = 0;
        output.graph_bytes = &graph_bytes;
        output.graph_size = &graph_size;
        output.weights_bytes = &weights_bytes;
        output.weights_size = &weights_size;
        CHECK(vx_quantization_plan_write_package(plan, &output, &report) == VX_STATUS_OK);
        CHECK(graph_bytes && weights_bytes && strstr((const char*)graph_bytes, "PackedLinear"));
        unsigned char *graph_again = NULL, *weights_again = NULL;
        size_t graph_again_size = 0, weights_again_size = 0;
        VxPTQPackageOptions repeated = VX_PTQ_PACKAGE_OPTIONS_INIT;
        repeated.graph_bytes = &graph_again; repeated.graph_size = &graph_again_size;
        repeated.weights_bytes = &weights_again; repeated.weights_size = &weights_again_size;
        CHECK(vx_quantization_plan_write_package(plan, &repeated, &report) == VX_STATUS_OK);
        CHECK(graph_size == graph_again_size && weights_size == weights_again_size);
        CHECK(!memcmp(graph_bytes, graph_again, graph_size) && !memcmp(weights_bytes, weights_again, weights_size));
        free(graph_again); free(weights_again);
        CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan), &batch, &info, &report) == VX_STATUS_INVALID_ARGUMENT);
        batch.sample_name = "retry";
        float saved = values[0]; values[0] = NAN;
        CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan), &batch, &info, &report) != VX_STATUS_OK);
        CHECK(vx_quantization_plan_write_package(plan, &repeated, &report) == VX_STATUS_OK);
        CHECK(weights_size == weights_again_size && !memcmp(weights_bytes, weights_again, weights_size));
        free(graph_again); free(weights_again);
        values[0] = saved;
        CHECK(vx_ptq_plan_calibrate(vx_quantization_plan_static(plan), &batch, &info, &report) == VX_STATUS_OK);
        CHECK(info.calibration_batches == 2 && info.calibration_samples == 2);
        CHECK(vx_quantization_plan_write_package(plan, &repeated, &report) == VX_STATUS_OK);
        CHECK(weights_again_size > 0);
        free(graph_again); free(weights_again);
        vx_quantization_plan_release(plan);
        free(graph_bytes);
        free(weights_bytes);
    }
    free(source);
    return 1;
}

int main(void) {
    if (!test_grouped_fitting_contract() || !test_lowbit_dense_storage_and_fitting() ||
        !test_extreme_finite_weight_fitting() ||
        !test_mixed_rules_and_shared_weights() ||
        !test_importance_and_mse_calibration()) return 1;
    const VxPTQMode modes[] = {VX_PTQ_MODE_W8A8, VX_PTQ_MODE_W4A8, VX_PTQ_MODE_W4A16, VX_PTQ_MODE_W8A16};
    for (size_t mode = 0; mode < sizeof(modes) / sizeof(modes[0]); mode++)
        for (int transposed = 0; transposed < 2; transposed++)
            for (int bf16 = 0; bf16 < 2; bf16++)
                if (!test_values(modes[mode], transposed, bf16, 0, 0)) return 1;
    if (!test_values(VX_PTQ_MODE_W4A16, 0, 0, 1, 0) ||
        !test_values(VX_PTQ_MODE_W4A16, 0, 0, 0, 1) || !test_refusals() ||
        !test_plan_owner(1) || !test_plan_owner(0) ||
        !test_static_reduced_range() || !test_weight_range_validation()) return 1;
    puts("weight packing: all checks passed");
    return 0;
}
