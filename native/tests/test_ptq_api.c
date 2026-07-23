#include "volvoxai.h"
#include "volvoxai_full.h"
#include "safetensors.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

_Static_assert(VX_PTQ_MODE_W8A8 == 0,
               "protobuf PTQ mode must own the native value");
_Static_assert(VX_PTQ_SCHEME_SYMMETRIC == 0 &&
               VX_PTQ_SCHEME_ASYMMETRIC == 1,
               "protobuf PTQ scheme must own the native values");
_Static_assert(VX_PTQ_LAYER_QLINEAR == 0 && VX_PTQ_LAYER_QCONV2D == 1,
               "protobuf PTQ layer kind must own the native values");

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length = strlen(text);
    int status = 0;
    if (!file) return -1;
    if (length && fwrite(text, 1, length, file) != length) status = -1;
    if (fclose(file) != 0) status = -1;
    return status;
}

static int file_contains(const char* path, const char* needle) {
    char buffer[8192];
    FILE* file = fopen(path, "rb");
    size_t length;
    int complete;
    if (!file) return 0;
    length = fread(buffer, 1, sizeof(buffer) - 1u, file);
    complete = feof(file);
    if (fclose(file) != 0) complete = 0;
    buffer[length] = 0;
    return complete && strstr(buffer, needle) != NULL;
}

static int write_weights(const char* path) {
    const int matrix_shape[2] = {2, 3};
    const int bias_shape[1] = {2};
    const float weights[6] = {1.0f, 2.0f, -1.0f, -2.0f, 0.5f, 3.0f};
    const float bias[2] = {0.25f, -0.5f};
    SafetensorsFile file;
    int status;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(&file, "weight", SAFETENSORS_DTYPE_F32,
                               matrix_shape, 2, weights, sizeof(weights)) != 0 ||
        safetensors_add_tensor(&file, "bias", SAFETENSORS_DTYPE_F32,
                               bias_shape, 1, bias, sizeof(bias)) != 0) {
        safetensors_free(&file);
        return -1;
    }
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

int main(void) {
    const char* source_graph_path = "/tmp/volvox-ptq-api-source.graph.json";
    const char* template_graph_path = "/tmp/volvox-ptq-api-template.graph.json";
    const char* obsolete_graph_path = "/tmp/volvox-ptq-api-obsolete.graph.json";
    const char* obsolete_storage_graph_path =
        "/tmp/volvox-ptq-api-obsolete-storage.graph.json";
    const char* inline_input_graph_path =
        "/tmp/volvox-ptq-api-inline-input.graph.json";
    const char* inline_output_graph_path =
        "/tmp/volvox-ptq-api-inline-output.graph.json";
    const char* source_weights_path = "/tmp/volvox-ptq-api-source.safetensors";
    const char* output_directory = "/tmp/volvox-ptq-api-output";
    const char* output_graph_path = "/tmp/volvox-ptq-api-output/graph.json";
    const char* output_weights_path =
        "/tmp/volvox-ptq-api-output/model.safetensors";
    const char* source_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"params\":{\"weight_layout\":\"OUT_IN\"}}],\"outputs\":[\"y\"]}";
    const char* template_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.i8\","
        "\"bias\":\"bias.i32\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"params\":{\"weight_layout\":\"OUT_IN\"}}],"
        "\"outputs\":[\"y\"]}";
    const char* obsolete_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"weights_quantization\":{},"
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* obsolete_storage_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"weights_quantization_storage\":\"sidecar.json\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* inline_input_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"int8\","
        "\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":0}}},\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* inline_output_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\","
        "\"scale\":1.0,\"zero_point\":0}}}],\"outputs\":[\"y\"]}";
    const char* weight_paths[] = {source_weights_path};
    const float sample[6] = {1.0f, -1.0f, 0.5f, 0.0f, 2.0f, -1.0f};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource model_source = VX_MODEL_SOURCE_INIT;
    VxPTQObserverSpec observers[2] = {
        VX_PTQ_OBSERVER_SPEC_INIT,
        VX_PTQ_OBSERVER_SPEC_INIT,
    };
    VxPTQLayerSpec layer = VX_PTQ_LAYER_SPEC_INIT;
    VxPTQPlanOptions plan_options = VX_PTQ_PLAN_OPTIONS_INIT;
    VxPTQInput input = VX_PTQ_INPUT_INIT;
    VxPTQPackageOptions package = VX_PTQ_PACKAGE_OPTIONS_INIT;
    VxPTQPlanInfo plan_info = VX_PTQ_PLAN_INFO_INIT;
    VxPTQTensorParameters parameters = VX_PTQ_TENSOR_PARAMETERS_INIT;
    VxAdapterSource adapter = VX_ADAPTER_SOURCE_INIT;
    VxAdapterRevision adapter_revision = VX_ADAPTER_REVISION_INIT;
    VxTensorInfo input_info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxModel* rejected_model = NULL;
    VxPTQPlan* plan = NULL;
    VxPTQPlan* stale = NULL;
    VxStatus status;
    uint64_t samples = 0;

    remove(source_graph_path);
    remove(template_graph_path);
    remove(obsolete_graph_path);
    remove(obsolete_storage_graph_path);
    remove(inline_input_graph_path);
    remove(inline_output_graph_path);
    remove(source_weights_path);
    remove(output_graph_path);
    remove(output_weights_path);
    (void)rmdir(output_directory);
    CHECK(mkdir(output_directory, 0700) == 0);
    CHECK(write_text(source_graph_path, source_graph) == 0);
    CHECK(write_text(template_graph_path, template_graph) == 0);
    CHECK(write_text(obsolete_graph_path, obsolete_graph) == 0);
    CHECK(write_text(obsolete_storage_graph_path,
                     obsolete_storage_graph) == 0);
    CHECK(write_text(inline_input_graph_path, inline_input_graph) == 0);
    CHECK(write_text(inline_output_graph_path, inline_output_graph) == 0);
    CHECK(write_weights(source_weights_path) == 0);

    model_source.graph_path = source_graph_path;
    model_source.weight_paths = weight_paths;
    model_source.weight_path_count = 1;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    model_source.graph_path = obsolete_graph_path;
    CHECK(vx_runtime_load_model(
              runtime, &model_source, &rejected_model, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(rejected_model == NULL &&
          !strcmp(report.reason, "INVALID_GRAPH_CONTRACT"));
    model_source.graph_path = obsolete_storage_graph_path;
    CHECK(vx_runtime_load_model(
              runtime, &model_source, &rejected_model, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(rejected_model == NULL);
    model_source.graph_path = inline_input_graph_path;
    CHECK(vx_runtime_load_model(
              runtime, &model_source, &rejected_model, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(rejected_model == NULL);
    model_source.graph_path = inline_output_graph_path;
    CHECK(vx_runtime_load_model(
              runtime, &model_source, &rejected_model, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(rejected_model == NULL);
    model_source.graph_path = source_graph_path;
    CHECK(vx_runtime_load_model(runtime, &model_source, &model, &report) ==
          VX_STATUS_OK);

    observers[0].tensor_name = "x";
    observers[1].tensor_name = "y";
    layer.node_index = 0;
    layer.input_tensor_name = "x";
    layer.output_tensor_name = "y";
    layer.source_weight_name = "weight";
    layer.packed_weight_name = "weight.i8";
    layer.source_bias_name = "bias";
    layer.packed_bias_name = "bias.i32";
    plan_options.template_graph_path = template_graph_path;
    plan_options.observers = observers;
    plan_options.observer_count = 2;
    plan_options.layers = &layer;
    plan_options.layer_count = 1;
    status = vx_model_create_ptq_plan(model, &plan_options, &plan, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "create PTQ failed: status=%d reason=%s message=%s\n",
                status, report.reason, report.message);
    CHECK(status == VX_STATUS_OK);
    CHECK(plan && report.stage == VX_STAGE_PTQ_CREATE &&
          !strcmp(report.backend, "cpu") && report.route_attested);

    /* Creation owns an immutable 0600 snapshot; later caller mutation cannot
     * alter the graph that the package writer consumes. */
    CHECK(write_text(template_graph_path, "{\"format\":\"mutated\"}") == 0);
    CHECK(vx_ptq_plan_input_count(plan) == 1);
    CHECK(vx_ptq_plan_input_info(plan, 0, &input_info, &report) == VX_STATUS_OK);
    CHECK(input_info.name && !strcmp(input_info.name, "x") &&
          input_info.dtype == VX_DTYPE_F32 && input_info.byte_size == sizeof(sample));

    input.name = "x";
    input.data = sample;
    input.byte_size = sizeof(sample) - sizeof(float);
    CHECK(vx_ptq_plan_calibrate(plan, "bad-size", &input, 1, &samples,
                                &report) == VX_STATUS_INVALID_ARGUMENT);
    input.byte_size = sizeof(sample);
    CHECK(vx_ptq_plan_calibrate(plan, "sample-0", &input, 1, &samples,
                                &report) == VX_STATUS_OK);
    CHECK(samples == 1 && report.stage == VX_STAGE_PTQ_CALIBRATE);
    CHECK(vx_ptq_plan_calibrate(plan, "sample-0", &input, 1, &samples,
                                &report) == VX_STATUS_INVALID_ARGUMENT);
    CHECK(vx_ptq_plan_info(plan, &plan_info, &report) == VX_STATUS_OK);
    CHECK(plan_info.calibration_samples == 1 && plan_info.tensor_count == 2 &&
          plan_info.revision.weight_revision == 1);
    CHECK(vx_ptq_plan_tensor_parameters(plan, 0, &parameters, &report) ==
          VX_STATUS_OK);
    CHECK(!strcmp(parameters.tensor_name, "x") &&
          parameters.observed_values == 6 &&
          fabsf(parameters.observed_min - -1.0f) < 1.0e-6f &&
          fabsf(parameters.observed_max - 2.0f) < 1.0e-6f);

    package.output_graph_path = "/tmp/volvox-ptq-api-output/not-graph.json";
    package.output_weights_path = output_weights_path;
    CHECK(vx_ptq_plan_write_package(plan, &package, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    package.output_graph_path = output_graph_path;
    CHECK(vx_ptq_plan_write_package(plan, &package, &report) == VX_STATUS_OK);
    CHECK(report.stage == VX_STAGE_PTQ_WRITE &&
          file_contains(output_graph_path, "\"format\":\"volvox-graph/v1\"") &&
          file_contains(output_graph_path, "\"samples\":[\"sample-0\"]") &&
          file_contains(output_graph_path,
                        "\"format\":\"volvox-affine-safetensors/v1\"") &&
          file_contains(output_graph_path, "\"scale_tensor\":") &&
          file_contains(output_graph_path, "\"zero_point_tensor\":") &&
          !file_contains(output_graph_path, "\"weights_quantization\":") &&
          !file_contains(output_graph_path, "\"outputs_quantization\":"));
    {
        SafetensorsFile output;
        const SafetensorsTensor* packed_weight;
        const SafetensorsTensor* packed_bias;
        int scale_tensors = 0;
        int zero_point_tensors = 0;
        int scale_values = 0;
        int zero_point_values = 0;
        CHECK(safetensors_load(output_weights_path, &output) == 0);
        packed_weight = safetensors_find_tensor(&output, "weight.i8");
        packed_bias = safetensors_find_tensor(&output, "bias.i32");
        CHECK(packed_weight && packed_weight->dtype == SAFETENSORS_DTYPE_I8);
        CHECK(packed_bias && packed_bias->dtype == SAFETENSORS_DTYPE_I32);
        for (int index = 0; index < output.tensor_count; index++) {
            const SafetensorsTensor* tensor = &output.tensors[index];
            if (strncmp(tensor->name, "__quant__.", 10)) continue;
            if (strstr(tensor->name, ".scale")) {
                CHECK(tensor->dtype == SAFETENSORS_DTYPE_F32 &&
                      tensor->ndim == 1 && tensor->shape[0] > 0);
                scale_tensors++;
                scale_values += tensor->shape[0];
            } else if (strstr(tensor->name, ".zero_point")) {
                CHECK(tensor->dtype == SAFETENSORS_DTYPE_I8 &&
                      tensor->ndim == 1 && tensor->shape[0] > 0);
                zero_point_tensors++;
                zero_point_values += tensor->shape[0];
            } else {
                CHECK(0);
            }
        }
        CHECK(scale_tensors == 3 && zero_point_tensors == 3);
        CHECK(scale_values == 4 && zero_point_values == 4);
        safetensors_free(&output);
    }
    {
        const char* output_weight_paths[] = {output_weights_path};
        VxModelSource output_source = VX_MODEL_SOURCE_INIT;
        VxModel* output_model = NULL;
        output_source.graph_path = output_graph_path;
        output_source.weight_paths = output_weight_paths;
        output_source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(
                  runtime, &output_source, &output_model, &report) == VX_STATUS_OK);
        CHECK(output_model != NULL);
        vx_model_release(output_model);
    }
    CHECK(vx_ptq_plan_close(plan, &report) == VX_STATUS_OK);
    CHECK(vx_ptq_plan_close(plan, &report) == VX_STATUS_OK);
    plan_info = (VxPTQPlanInfo)VX_PTQ_PLAN_INFO_INIT;
    CHECK(vx_ptq_plan_info(plan, &plan_info, &report) ==
          VX_STATUS_HANDLE_DISPOSED);

    /* A plan is permanently stale as soon as Model lineage changes. */
    CHECK(write_text(template_graph_path, template_graph) == 0);
    CHECK(vx_model_create_ptq_plan(model, &plan_options, &stale, &report) ==
          VX_STATUS_OK);
    adapter.adapter_name = "ptq-stale";
    CHECK(vx_model_publish_adapter(model, &adapter, &adapter_revision, &report) ==
          VX_STATUS_OK);
    plan_info = (VxPTQPlanInfo)VX_PTQ_PLAN_INFO_INIT;
    CHECK(vx_ptq_plan_info(stale, &plan_info, &report) ==
          VX_STATUS_REVISION_CONFLICT);
    CHECK(!strcmp(report.reason, "REVISION_CONFLICT"));

    vx_ptq_plan_release(stale);
    vx_ptq_plan_release(plan);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    remove(source_graph_path);
    remove(template_graph_path);
    remove(obsolete_graph_path);
    remove(obsolete_storage_graph_path);
    remove(inline_input_graph_path);
    remove(inline_output_graph_path);
    remove(source_weights_path);
    remove(output_graph_path);
    remove(output_weights_path);
    CHECK(rmdir(output_directory) == 0);
    puts("native proto-owned PTQ lifecycle, snapshot, package, and staleness tests passed");
    return 0;
}
