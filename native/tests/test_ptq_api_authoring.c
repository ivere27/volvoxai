/* Unified quantization lifecycle through the generated public dispatch. */
#include "volvoxai_ffi.h"
#include "../cli/call_client.h"
#include "../src/generated/proto_methods.h"
#include "volvoxai_lite.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static VxCallClient client;
static int failures;
#define CHECK(value, message) do { if (!(value)) { \
    fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, message); failures++; \
} } while (0)

#define CALL_HELPER(name, Request, Response, request_codec, response_codec, method) \
static int name(const Request* request, Response* response) { \
    uint8_t* encoded = NULL; size_t encoded_size = 0; int32_t size = 0; \
    response_codec##_init(response); \
    if (request_codec##_encode(request, &encoded, &encoded_size) != SYNURANG_LITE_OK) { \
        CHECK(0, "request encodes"); return 0; \
    } \
    uint8_t* bytes = vx_call_bytes(&client, method, encoded, (int32_t)encoded_size, &size); \
    request->_allocator->deallocate(request->_allocator->context, encoded); \
    if (!bytes || size <= 0) { \
        int32_t error_size = 0; const uint8_t* error = vx_call_error(&client, &error_size); \
        fprintf(stderr, "dispatch: %.*s\n", error_size, error ? (const char*)error : ""); \
        CHECK(0, "RPC returns a response"); if (bytes) vx_call_free(&client, bytes); return 0; \
    } \
    int decoded = response_codec##_decode(response, bytes, (size_t)size) == SYNURANG_LITE_OK; \
    vx_call_free(&client, bytes); CHECK(decoded, "response decodes"); return decoded; \
}
CALL_HELPER(call_create, VolvoxaiV1CreateQuantizationPlanRequest, VolvoxaiV1QuantizationPlanHandle,
    volvoxai_v1_create_quantization_plan_request, volvoxai_v1_quantization_plan_handle,
    VX_RPC_VX_QUANTIZATION_SERVICE_CREATE_QUANTIZATION_PLAN)
CALL_HELPER(call_calibrate, VolvoxaiV1CalibrateQuantizationPlanRequest, VolvoxaiV1CalibrationProgress,
    volvoxai_v1_calibrate_quantization_plan_request, volvoxai_v1_calibration_progress,
    VX_RPC_VX_QUANTIZATION_SERVICE_CALIBRATE_QUANTIZATION_PLAN)
CALL_HELPER(call_inspect, VolvoxaiV1QuantizationPlanRef, VolvoxaiV1QuantizationPlanResponse,
    volvoxai_v1_quantization_plan_ref, volvoxai_v1_quantization_plan_response,
    VX_RPC_VX_QUANTIZATION_SERVICE_INSPECT_QUANTIZATION_PLAN)
CALL_HELPER(call_export, VolvoxaiV1ExportQuantizedModelRequest, VolvoxaiV1QuantizedModelInfo,
    volvoxai_v1_export_quantized_model_request, volvoxai_v1_quantized_model_info,
    VX_RPC_VX_QUANTIZATION_SERVICE_EXPORT_QUANTIZED_MODEL)
CALL_HELPER(call_release, VolvoxaiV1QuantizationPlanRef, VolvoxaiV1OperationReport,
    volvoxai_v1_quantization_plan_ref, volvoxai_v1_operation_report,
    VX_RPC_VX_QUANTIZATION_SERVICE_RELEASE_QUANTIZATION_PLAN)
#undef CALL_HELPER

static SynurangLiteBytes borrowed(const void* bytes, size_t size) {
    return (SynurangLiteBytes){.data = (uint8_t*)bytes, .len = size};
}
static SynurangLiteBytes text(const char* value) { return borrowed(value, strlen(value)); }
static int report_is(const VolvoxaiV1OperationReport* report, VolvoxaiV1NativeStatus status) {
    if (!report || report->field_status != status) {
        fprintf(stderr, "expected status %d, received %d: %.*s\n", (int)status,
            report ? (int)report->field_status : -999,
            report ? (int)report->field_message.len : 0,
            report && report->field_message.data ? (const char*)report->field_message.data : "");
        return 0;
    }
    return 1;
}
static int bytes_contain(const SynurangLiteBytes* bytes, const char* needle) {
    size_t length = strlen(needle);
    for (size_t i = 0; i + length <= bytes->len; i++)
        if (!memcmp(bytes->data + i, needle, length)) return 1;
    return 0;
}

static const char source_graph[] =
    "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
    "\"inputs\":{\"x\":{\"shape\":[1,4],\"dtype\":\"float32\"}},"
    "\"nodes\":[{\"id\":\"fc\",\"opType\":\"Linear\","
    "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
    "\"outputs\":{\"out\":{\"tensor\":\"y\",\"shape\":[1,2],\"dtype\":\"float32\"}},"
    "\"params\":{\"weight_layout\":\"dout_din\"}}],\"outputs\":[\"y\"],"
    "\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\","
    "\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}}";

static unsigned char* fixture_weights(size_t* size) {
    static const char header[] =
        "{\"w\":{\"dtype\":\"F32\",\"shape\":[2,4],\"data_offsets\":[0,32]},"
        "\"b\":{\"dtype\":\"F32\",\"shape\":[2],\"data_offsets\":[32,40]}}";
    const float values[] = {1,2,3,4,-1,-2,-3,-4,0.5f,-0.5f};
    size_t header_size = (sizeof(header) - 1u + 7u) & ~(size_t)7u;
    *size = 8u + header_size + sizeof(values);
    unsigned char* bytes = (unsigned char*)malloc(*size);
    if (!bytes) return NULL;
    for (unsigned i = 0; i < 8; i++) bytes[i] = (unsigned char)((uint64_t)header_size >> (8u * i));
    memset(bytes + 8, ' ', header_size);
    memcpy(bytes + 8, header, sizeof(header) - 1u);
    memcpy(bytes + 8 + header_size, values, sizeof(values));
    return bytes;
}

static void release_plan(int64_t id) {
    VolvoxaiV1QuantizationPlanRef ref;
    VolvoxaiV1OperationReport released;
    volvoxai_v1_quantization_plan_ref_init(&ref);
    ref.field_quantization_plan_id = id;
    if (call_release(&ref, &released)) {
        CHECK(report_is(&released, VOLVOXAI_V1_NATIVE_STATUS_OK), "release succeeds");
        volvoxai_v1_operation_report_free(&released);
    }
    VolvoxaiV1QuantizationPlanResponse inspected;
    if (call_inspect(&ref, &inspected)) {
        CHECK(report_is(inspected.field_report, VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED),
              "released plan is no longer inspectable");
        volvoxai_v1_quantization_plan_response_free(&inspected);
    }
}

static void check_calibration(int64_t id, int supported) {
    const float values[] = {1,-2,3,-4};
    int64_t shape[] = {1,4};
    VolvoxaiV1Tensor input;
    volvoxai_v1_tensor_init(&input);
    input.field_name = text("x");
    input.field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
    input.field_shape.data = shape;
    input.field_shape.len = input.field_shape.cap = 2;
    input.field_inline = borrowed(values, sizeof(values));
    /* Tensor bytes is a oneof. The generated discriminator is set below
     * using the schema field number, as in the existing native roundtrips. */
    input.which_payload = 5;
    VolvoxaiV1CalibrateQuantizationPlanRequest request;
    VolvoxaiV1CalibrationProgress response;
    volvoxai_v1_calibrate_quantization_plan_request_init(&request);
    request.field_quantization_plan_id = id;
    request.field_sample_name = text("sample");
    request.field_sample_count = 1;
    request.field_inputs.data = &input;
    request.field_inputs.len = request.field_inputs.cap = 1;
    if (call_calibrate(&request, &response)) {
        CHECK(report_is(response.field_report, supported ? VOLVOXAI_V1_NATIVE_STATUS_OK
            : VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT), "calibration policy is enforced");
        if (supported) CHECK(response.field_ready && response.field_samples == 1,
                             "static calibration commits coverage");
        volvoxai_v1_calibration_progress_free(&response);
    }
}

static void test_policy(SynurangLiteBytes shard, uint32_t bits, int packed, int floating) {
    VolvoxaiV1WeightQuantization weight;
    VolvoxaiV1StaticActivations activation;
    VolvoxaiV1WeightFormat format;
    VolvoxaiV1ModelBytes source_bytes;
    VolvoxaiV1ModelFiles source_files;
    VolvoxaiV1CalibrationOptions calibration_options;
    VolvoxaiV1QuantizationConfig config;
    VolvoxaiV1CreateQuantizationPlanRequest request;
    VolvoxaiV1QuantizationPlanHandle response;
    volvoxai_v1_weight_quantization_init(&weight);
    volvoxai_v1_static_activations_init(&activation);
    volvoxai_v1_weight_format_init(&format);
    volvoxai_v1_model_bytes_init(&source_bytes);
    volvoxai_v1_model_files_init(&source_files);
    volvoxai_v1_calibration_options_init(&calibration_options);
    weight.field_format = &format;
    volvoxai_v1_quantization_config_init(&config);
    volvoxai_v1_create_quantization_plan_request_init(&request);
    format.field_bits = bits;
    format.field_group_size = packed ? 128 : 0;
    (void)floating;
    activation.field_dtype = VOLVOXAI_V1_DATA_TYPE_I8;
    config.field_weights = &weight;
    config.field_static_activations = packed ? NULL : &activation;
    request.field_config = &config;
    request.which_source = 2; request.field_bytes = &source_bytes; source_bytes.field_graph = text(source_graph);
    source_bytes.field_weight_shards.data = &shard;
    source_bytes.field_weight_shards.len = source_bytes.field_weight_shards.cap = 1;
    request.field_calibration = &calibration_options; calibration_options.field_cpu_threads = 1;
    if (!call_create(&request, &response)) return;
    CHECK(report_is(response.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK), "plan creates without caller model/runtime");
    int64_t id = response.field_quantization_plan_id;
    CHECK(id && response.field_info, "plan handle and information are returned");
    if (response.field_info) {
        CHECK(response.field_info->field_requires_calibration == !packed, "calibration belongs to activation policy");
        CHECK(response.field_info->field_ready == packed, "group packing is immediately ready");
        CHECK((response.field_info->field_calibration != NULL) == !packed, "group packing has no calibration report");
        CHECK(response.field_info->field_calibration_inputs.len == (size_t)!packed, "only static calibration has inputs");
    }
    volvoxai_v1_quantization_plan_handle_free(&response);
    if (!id) return;
    VolvoxaiV1ExportQuantizedModelRequest export_request;
    VolvoxaiV1QuantizedModelInfo exported;
    volvoxai_v1_export_quantized_model_request_init(&export_request);
    export_request.field_quantization_plan_id = id;
    if (!packed && call_export(&export_request, &exported)) {
        CHECK(report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT),
              "static export requires calibration");
        volvoxai_v1_quantized_model_info_free(&exported);
    }
    check_calibration(id, !packed);
    VolvoxaiV1QuantizationPlanRef ref;
    VolvoxaiV1QuantizationPlanResponse inspected;
    volvoxai_v1_quantization_plan_ref_init(&ref);
    ref.field_quantization_plan_id = id;
    if (call_inspect(&ref, &inspected)) {
        CHECK(report_is(inspected.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK) && inspected.field_info->field_ready,
              "plan is ready after its required preparation");
        volvoxai_v1_quantization_plan_response_free(&inspected);
    }
    if (call_export(&export_request, &exported)) {
        CHECK(report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK), "package exports");
        CHECK(exported.field_bytes->field_graph.len && exported.field_bytes->field_weight_shards.data[0].len, "export returns package bytes");
        CHECK(bytes_contain(&exported.field_bytes->field_graph, packed ? "PackedLinear" : "QLinear"), "numerical operator matches policy");
        CHECK(bytes_contain(&exported.field_bytes->field_weight_shards.data[0], "profile_coverage") == !packed, "coverage metadata exists only for static calibration");
        if (packed) {
            char expected[32];
            snprintf(expected, sizeof(expected), "\"weight_bits\":%u", bits);
            CHECK(bytes_contain(&exported.field_bytes->field_graph, expected), "stored weight precision matches policy");
            CHECK(!bytes_contain(&exported.field_bytes->field_graph, "activation_bits"),
                  "weight artifact has no execution arithmetic");
        }
        release_plan(id); id = 0;
        CHECK(exported.field_bytes && bytes_contain(&exported.field_bytes->field_graph, "volvox-graph/v1"),
              "returned bytes remain owned after plan release");
        volvoxai_v1_quantized_model_info_free(&exported);
    }
    if (id) release_plan(id);
}

static void test_rule_resolution(SynurangLiteBytes shard) {
    VolvoxaiV1WeightFormat format;
    VolvoxaiV1WeightQuantization importance, rtn;
    VolvoxaiV1QuantizationRule rules[2];
    VolvoxaiV1QuantizationConfig config;
    VolvoxaiV1ModelBytes source;
    VolvoxaiV1CreateQuantizationPlanRequest request;
    VolvoxaiV1QuantizationPlanHandle response;
    volvoxai_v1_weight_format_init(&format);
    volvoxai_v1_weight_quantization_init(&importance);
    volvoxai_v1_weight_quantization_init(&rtn);
    volvoxai_v1_quantization_config_init(&config);
    volvoxai_v1_model_bytes_init(&source);
    volvoxai_v1_create_quantization_plan_request_init(&request);
    for (int i = 0; i < 2; i++) volvoxai_v1_quantization_rule_init(&rules[i]);
    format.field_bits = 4; format.field_group_size = 128;
    importance.field_format = &format; importance.field_algorithm = VOLVOXAI_V1_WEIGHT_ALGORITHM_IMPORTANCE;
    rtn.field_format = &format;
    rules[0].which_selector = 2; rules[0].field_operator_name = text("Linear");
    rules[0].which_action = 4; rules[0].field_weights = &rtn;
    rules[1].which_selector = 3; rules[1].field_node_glob = text("*");
    rules[1].which_action = 4; rules[1].field_weights = &importance;
    config.field_weights = &importance;
    config.field_rules.data = rules; config.field_rules.len = config.field_rules.cap = 2;
    source.field_graph = text(source_graph);
    source.field_weight_shards.data = &shard;
    source.field_weight_shards.len = source.field_weight_shards.cap = 1;
    request.which_source = 2; request.field_bytes = &source; request.field_config = &config;
    if (!call_create(&request, &response)) return;
    CHECK(report_is(response.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK), "typed rules resolve");
    if (response.field_info) {
        CHECK(!response.field_info->field_requires_calibration && response.field_info->field_ready,
              "shadowed importance rules do not require an engine");
        CHECK(response.field_info->field_resolved_nodes.len == 1 && response.field_info->field_quantized.len == 1,
              "resolved weight decisions and format counts are returned");
        if (response.field_info->field_resolved_nodes.len == 1) {
            const VolvoxaiV1QuantizationNodeInfo* decision = response.field_info->field_resolved_nodes.data;
            CHECK(decision->which_action == 2 && decision->field_weights &&
                  decision->field_weights->field_algorithm == VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN,
                  "first matching operator rule wins over later glob");
        }
    }
    int64_t id = response.field_quantization_plan_id;
    volvoxai_v1_quantization_plan_handle_free(&response);
    if (id) release_plan(id);
}

static void test_refusals(SynurangLiteBytes shard) {
    VolvoxaiV1WeightFormat format;
    VolvoxaiV1WeightQuantization weight;
    VolvoxaiV1StaticActivations activation;
    VolvoxaiV1QuantizationConfig config;
    VolvoxaiV1CreateQuantizationPlanRequest request;
    VolvoxaiV1ModelBytes source;
    VolvoxaiV1QuantizationRule rule;
    volvoxai_v1_weight_format_init(&format);
    volvoxai_v1_weight_quantization_init(&weight);
    volvoxai_v1_static_activations_init(&activation);
    volvoxai_v1_quantization_config_init(&config);
    volvoxai_v1_create_quantization_plan_request_init(&request);
    volvoxai_v1_model_bytes_init(&source);
    volvoxai_v1_quantization_rule_init(&rule);
    weight.field_format = &format;
    config.field_weights = &weight;
    request.field_config = &config;
    request.which_source = 2; request.field_bytes = &source;
    source.field_graph = text(source_graph);
    source.field_weight_shards.data = &shard;
    source.field_weight_shards.len = source.field_weight_shards.cap = 1;
    for (int variant = 0; variant < 10; variant++) {
        format.field_bits = 8; format.field_group_size = 128;
        activation.field_dtype = VOLVOXAI_V1_DATA_TYPE_I8;
        config.field_static_activations = NULL;
        config.field_rules.len = 0;
        source.field_weight_shards.len = 1;
        request.which_source = 2;
        weight.field_algorithm = VOLVOXAI_V1_WEIGHT_ALGORITHM_RTN;
        if (variant == 0) { format.field_bits = 4; format.field_group_size = 0;
            config.field_static_activations = &activation; }
        if (variant == 1) config.field_static_activations = &activation;
        if (variant == 2) format.field_group_size = 48;
        if (variant == 3) { format.field_group_size = 0;
            activation.field_dtype = VOLVOXAI_V1_DATA_TYPE_F16;
            config.field_static_activations = &activation; }
        if (variant == 4) request.which_source = 0;
        if (variant == 5) source.field_weight_shards.len = 0;
        if (variant == 6) { format.field_group_size = 0;
            config.field_static_activations = &activation;
            weight.field_algorithm = VOLVOXAI_V1_WEIGHT_ALGORITHM_IMPORTANCE; }
        if (variant >= 7) {
            config.field_rules.data = &rule; config.field_rules.len = config.field_rules.cap = 1;
            rule.which_selector = 1; rule.field_node_id = text("does-not-exist");
            rule.which_action = 4; rule.field_weights = &weight;
            if (variant == 8) { rule.which_selector = 2; rule.field_operator_name = text("NotAnOperator"); }
            if (variant == 9) { rule.field_node_id = text("fc"); rule.which_action = 0; }
        }
        VolvoxaiV1QuantizationPlanHandle response;
        if (call_create(&request, &response)) {
            CHECK(report_is(response.field_report, VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT), "unsupported policy/source is rejected");
            CHECK(!response.field_quantization_plan_id, "invalid requests publish no plan");
            volvoxai_v1_quantization_plan_handle_free(&response);
        }
    }
}

static int write_fixture(const char* path, const void* bytes, size_t size) {
    FILE* file = fopen(path, "wb");
    if (!file) return 0;
    int ok = fwrite(bytes, 1, size, file) == size;
    return fclose(file) == 0 && ok;
}

static void test_paths(SynurangLiteBytes shard, const char* graph_path, const char* output_path) {
    char weights_path[4096], output_weights[4096];
    snprintf(weights_path, sizeof(weights_path), "%s.safetensors", graph_path);
    const char* base = strrchr(output_path, '/');
    size_t parent = base ? (size_t)(base + 1 - output_path) : 0;
    snprintf(output_weights, sizeof(output_weights), "%.*sweights.safetensors", (int)parent, output_path);
    remove(output_path); remove(output_weights);
    CHECK(write_fixture(graph_path, source_graph, strlen(source_graph)), "path graph fixture writes");
    CHECK(write_fixture(weights_path, shard.data, shard.len), "path weight fixture writes");
    VolvoxaiV1WeightQuantization weight;
    VolvoxaiV1StaticActivations activation;
    VolvoxaiV1WeightFormat format;
    VolvoxaiV1ModelBytes source_bytes;
    VolvoxaiV1ModelFiles source_files;
    VolvoxaiV1CalibrationOptions calibration_options;
    VolvoxaiV1QuantizationConfig config;
    VolvoxaiV1CreateQuantizationPlanRequest request;
    VolvoxaiV1QuantizationPlanHandle response;
    volvoxai_v1_weight_quantization_init(&weight);
    volvoxai_v1_static_activations_init(&activation);
    volvoxai_v1_weight_format_init(&format);
    volvoxai_v1_model_bytes_init(&source_bytes);
    volvoxai_v1_model_files_init(&source_files);
    volvoxai_v1_calibration_options_init(&calibration_options);
    weight.field_format = &format;
    volvoxai_v1_quantization_config_init(&config);
    volvoxai_v1_create_quantization_plan_request_init(&request);
    format.field_bits = 4; format.field_group_size = 128;
    config.field_weights = &weight;
    request.field_config = &config;
    request.which_source = 1; request.field_files = &source_files; source_files.field_graph_path = text(graph_path);
    SynurangLiteBytes weight_path = text(weights_path);
    source_files.field_weight_paths.data = &weight_path;
    source_files.field_weight_paths.len = source_files.field_weight_paths.cap = 1;
    int64_t id = 0;
    if (call_create(&request, &response)) {
        CHECK(report_is(response.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK), "native paths create a plan");
        id = response.field_quantization_plan_id;
        volvoxai_v1_quantization_plan_handle_free(&response);
    }
    remove(graph_path); remove(weights_path);
    if (!id) return;
    VolvoxaiV1ExportQuantizedModelRequest export_request;
    VolvoxaiV1QuantizedModelInfo exported;
    volvoxai_v1_export_quantized_model_request_init(&export_request);
    export_request.field_quantization_plan_id = id;
    if (call_export(&export_request, &exported)) {
        CHECK(report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK) &&
              exported.field_bytes->field_graph.len && exported.field_bytes->field_weight_shards.data[0].len,
              "path inputs are snapshotted and may export bytes after source deletion");
        volvoxai_v1_quantized_model_info_free(&exported);
    }
    VolvoxaiV1ModelFiles output_files;
    volvoxai_v1_model_files_init(&output_files);
    export_request.field_output_files = &output_files;
    output_files.field_graph_path = text(output_path);
    if (call_export(&export_request, &exported)) {
        CHECK(report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT),
              "partial output paths are rejected");
        volvoxai_v1_quantized_model_info_free(&exported);
    }
    SynurangLiteBytes output_weight_path = text(output_weights);
    output_files.field_weight_paths.data = &output_weight_path;
    output_files.field_weight_paths.len = output_files.field_weight_paths.cap = 1;
    if (call_export(&export_request, &exported)) {
        CHECK(report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK),
              "native output paths export a package");
        CHECK(exported.which_artifact == 1 && !exported.field_bytes &&
              exported.field_files && exported.field_files->field_graph_path.len &&
              exported.field_files->field_weight_paths.len == 1,
              "file export returns paths and no duplicate payload");
        volvoxai_v1_quantized_model_info_free(&exported);
    }
    if (call_export(&export_request, &exported)) {
        CHECK(report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_IO_ERROR),
              "file export never overwrites existing destinations");
        volvoxai_v1_quantized_model_info_free(&exported);
    }
    release_plan(id);
    remove(output_path); remove(output_weights);
}

/* Native oracle for the WASM public-service serialization comparison. The
 * exported graph is a test artifact, so successive cases may replace it. */
static int export_fixture(const char* graph_path, const char* weights_path,
                          const char* output_path) {
    VolvoxaiV1WeightQuantization weight;
    VolvoxaiV1StaticActivations activation;
    VolvoxaiV1WeightFormat format;
    VolvoxaiV1ModelBytes source_bytes;
    VolvoxaiV1ModelFiles source_files;
    VolvoxaiV1CalibrationOptions calibration_options;
    VolvoxaiV1QuantizationConfig config;
    VolvoxaiV1CreateQuantizationPlanRequest request;
    VolvoxaiV1QuantizationPlanHandle response;
    volvoxai_v1_weight_quantization_init(&weight);
    volvoxai_v1_static_activations_init(&activation);
    volvoxai_v1_weight_format_init(&format);
    volvoxai_v1_model_bytes_init(&source_bytes);
    volvoxai_v1_model_files_init(&source_files);
    volvoxai_v1_calibration_options_init(&calibration_options);
    weight.field_format = &format;
    volvoxai_v1_quantization_config_init(&config);
    volvoxai_v1_create_quantization_plan_request_init(&request);
    format.field_bits = 8;
    activation.field_dtype = VOLVOXAI_V1_DATA_TYPE_I8;
    config.field_weights = &weight;
    config.field_static_activations = &activation;
    request.field_config = &config;
    request.field_calibration = &calibration_options; calibration_options.field_cpu_threads = 1;
    request.which_source = 1; request.field_files = &source_files; source_files.field_graph_path = text(graph_path);
    SynurangLiteBytes weight_path = text(weights_path);
    source_files.field_weight_paths.data = &weight_path;
    source_files.field_weight_paths.len = source_files.field_weight_paths.cap = 1;
    int ok = 0;
    int64_t id = 0;
    VolvoxaiV1Tensor* inputs = NULL;
    size_t input_count = 0;
    if (!call_create(&request, &response)) goto done;
    if (!report_is(response.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK) ||
        !response.field_quantization_plan_id || !response.field_info) goto done;
    id = response.field_quantization_plan_id;
    input_count = response.field_info->field_calibration_inputs.len;
    inputs = (VolvoxaiV1Tensor*)calloc(input_count, sizeof(*inputs));
    if (input_count && !inputs) goto done;
    for (size_t index = 0; index < input_count; index++) {
        const VolvoxaiV1TensorSpec* spec = &response.field_info->field_calibration_inputs.data[index];
        VolvoxaiV1Tensor* input = &inputs[index];
        volvoxai_v1_tensor_init(input);
        input->field_name = spec->field_name;
        input->field_dtype = spec->field_dtype;
        if (input->field_dtype != VOLVOXAI_V1_DATA_TYPE_F32) {
            fprintf(stderr, "export fixture requires F32 calibration inputs\n");
            goto done;
        }
        size_t rank = spec->field_dimensions.len;
        if (rank > SIZE_MAX / sizeof(int64_t)) goto done;
        input->field_shape.data = (int64_t*)calloc(rank, sizeof(int64_t));
        if (rank && !input->field_shape.data) goto done;
        input->field_shape.len = input->field_shape.cap = rank;
        size_t count = 1;
        for (size_t dimension = 0; dimension < rank; dimension++) {
            const VolvoxaiV1DimensionConstraint* extent = &spec->field_dimensions.data[dimension];
            if (extent->field_min <= 0 || extent->field_min != extent->field_max ||
                (uint64_t)extent->field_min > SIZE_MAX / count) {
                fprintf(stderr, "export fixture requires fixed positive input dimensions\n");
                goto done;
            }
            input->field_shape.data[dimension] = extent->field_min;
            count *= (size_t)extent->field_min;
        }
        if (count > SIZE_MAX / sizeof(float)) goto done;
        float* values = (float*)malloc(count * sizeof(*values));
        if (!values) goto done;
        for (size_t element = 0; element < count; element++)
            values[element] = (float)((int)((element % 17u) * 3u % 17u) - 8) / 8.0f;
        input->field_inline = borrowed(values, count * sizeof(*values));
        input->which_payload = 5;
    }
    VolvoxaiV1CalibrateQuantizationPlanRequest calibration;
    VolvoxaiV1CalibrationProgress calibrated;
    volvoxai_v1_calibrate_quantization_plan_request_init(&calibration);
    calibration.field_quantization_plan_id = id;
    calibration.field_sample_name = text("rendering");
    calibration.field_sample_count = 1;
    calibration.field_inputs.data = inputs;
    calibration.field_inputs.len = calibration.field_inputs.cap = input_count;
    int calibrated_ok = call_calibrate(&calibration, &calibrated) &&
        report_is(calibrated.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK);
    volvoxai_v1_calibration_progress_free(&calibrated);
    if (!calibrated_ok) goto done;
    VolvoxaiV1ExportQuantizedModelRequest export_request;
    VolvoxaiV1QuantizedModelInfo exported;
    volvoxai_v1_export_quantized_model_request_init(&export_request);
    export_request.field_quantization_plan_id = id;
    if (call_export(&export_request, &exported) &&
        report_is(exported.field_report, VOLVOXAI_V1_NATIVE_STATUS_OK)) {
        ok = exported.field_bytes->field_graph.len &&
            write_fixture(output_path, exported.field_bytes->field_graph.data, exported.field_bytes->field_graph.len);
        if (!ok) fprintf(stderr, "export fixture could not write %s\n", output_path);
    }
    volvoxai_v1_quantized_model_info_free(&exported);
done:
    if (inputs) for (size_t index = 0; index < input_count; index++) {
        free(inputs[index].field_shape.data);
        free(inputs[index].field_inline.data);
    }
    free(inputs);
    if (id) release_plan(id);
    volvoxai_v1_quantization_plan_handle_free(&response);
    return ok && !failures;
}

int main(int argc, char** argv) {
    if (!vx_call_client_open(&client)) return 1;
    if (argc == 5 && !strcmp(argv[1], "--export")) {
        int ok = export_fixture(argv[2], argv[3], argv[4]);
        vx_call_client_close(&client);
        return ok ? 0 : 1;
    }
    size_t weight_size = 0;
    unsigned char* weights = fixture_weights(&weight_size);
    if (!weights) { vx_call_client_close(&client); return 1; }
    SynurangLiteBytes shard = borrowed(weights, weight_size);
    test_policy(shard, 8, 0, 0);
    for (uint32_t bits = 4; bits <= 8; bits += 4)
        for (int floating = 0; floating < 2; floating++) test_policy(shard, bits, 1, floating);
    test_rule_resolution(shard);
    test_refusals(shard);
    if (argc == 3) test_paths(shard, argv[1], argv[2]);
    free(weights);
    vx_call_client_close(&client);
    if (failures) return 1;
    puts("quantization dispatch: all checks passed");
    return 0;
}
