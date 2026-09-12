/* Native module calls use the same generated protobuf messages as web clients.
 * The small command-line host drives Synurang's call lifecycle; operation
 * paths and payload codecs are generated from proto/volvoxai.proto.
 *
 * Build: make -C examples c_api_example
 * Run: ./examples/target/bin/c_api_client_raw models/tinystories_1m
 */
#include "../native/cli/call_client.h"
#include "../native/src/generated/proto_methods.h"
#include "volvoxai_lite.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void print_error(VxCallClient* client, const char* operation) {
    int32_t length = 0;
    const uint8_t* message = vx_call_error(client, &length);
    fprintf(stderr, "%s failed: %.*s\n", operation, (int)length,
            message ? (const char*)message : "(no detail)");
}

static int report_ok(const VolvoxaiV1OperationReport* report, const char* operation) {
    if (report && report->field_status ==
                      VOLVOXAI_V1_NATIVE_STATUS_OK) {
        return 1;
    }
    fprintf(stderr, "%s reported status %d: %.*s\n", operation,
            report ? (int)report->field_status : 0,
            report ? (int)report->field_message.len : 0,
            report && report->field_message.data
                ? (const char*)report->field_message.data : "");
    return 0;
}

int main(int argc, char** argv) {
    const char* model_dir = argc > 1 ? argv[1] : "models/tinystories_1m";
    char graph_path[1024];
    char weights_path[1024];
    VxCallClient client;
    uint8_t* payload = NULL;
    int32_t payload_len;
    int64_t runtime_id = 0;
    int64_t model_id = 0;
    int64_t compiled_id = 0;
    int64_t context_id = 0;
    int status = 1;

    if (!vx_call_client_open(&client)) return 1;
    snprintf(graph_path, sizeof(graph_path), "%s/graph.json", model_dir);
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

    {
        VolvoxaiV1CreateRuntimeRequest request;
        volvoxai_v1_create_runtime_request_init(&request);
        request.field_cpu_threads = 1;
        request.has_execution_mode = 1;
        request.field_execution_mode = VOLVOXAI_V1_EXECUTION_MODE_DIRECT;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME,
                        volvoxai_v1_create_runtime_request, &request, payload, payload_len);
        volvoxai_v1_create_runtime_request_free(&request);
    }
    if (!payload) {
        print_error(&client, "CreateRuntime");
        goto cleanup;
    }
    {
        VolvoxaiV1RuntimeHandle handle;
        volvoxai_v1_runtime_handle_init(&handle);
        if (volvoxai_v1_runtime_handle_decode(&handle, payload, (size_t)payload_len) ==
                SYNURANG_LITE_OK &&
            report_ok(handle.field_report, "CreateRuntime")) {
            runtime_id = handle.field_runtime_id;
        }
        volvoxai_v1_runtime_handle_free(&handle);
        vx_call_free(&client, payload);
        payload = NULL;
    }
    if (!runtime_id) goto cleanup;

    /* Every operation sends one generated protobuf request. */
    {
        VolvoxaiV1LoadModelRequest request;
        VolvoxaiV1ModelHandle handle;
        SynurangLiteBytes* weights;

        volvoxai_v1_load_model_request_init(&request);
        request.field_runtime_id = runtime_id;
        weights = volvoxai_v1_load_model_request_add_weight_paths(&request);
        if (!weights ||
            synurang_lite_bytes_assign(request._allocator, &request.field_graph_path,
                                       graph_path, strlen(graph_path)) != SYNURANG_LITE_OK ||
            synurang_lite_bytes_assign(request._allocator, weights, weights_path,
                                       strlen(weights_path)) != SYNURANG_LITE_OK) {
            volvoxai_v1_load_model_request_free(&request);
            goto cleanup;
        }
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL,
                        volvoxai_v1_load_model_request, &request, payload, payload_len);
        volvoxai_v1_load_model_request_free(&request);

        if (!payload) {
            print_error(&client, "LoadModel");
            goto cleanup;
        }
        volvoxai_v1_model_handle_init(&handle);
        if (volvoxai_v1_model_handle_decode(&handle, payload, (size_t)payload_len) ==
                SYNURANG_LITE_OK &&
            report_ok(handle.field_report, "LoadModel")) {
            model_id = handle.field_model_id;
        }
        volvoxai_v1_model_handle_free(&handle);
        vx_call_free(&client, payload);
        payload = NULL;
    }
    if (!model_id) goto cleanup;

    /* An empty policy takes the engine default, which prefers CPU. */
    {
        VolvoxaiV1CompileModelRequest request;
        volvoxai_v1_compile_model_request_init(&request);
        request.field_model_id = model_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL,
                        volvoxai_v1_compile_model_request, &request, payload, payload_len);
        volvoxai_v1_compile_model_request_free(&request);
    }
    if (!payload) {
        print_error(&client, "CompileModel");
        goto cleanup;
    }
    {
        VolvoxaiV1CompiledModelHandle handle;
        volvoxai_v1_compiled_model_handle_init(&handle);
        if (volvoxai_v1_compiled_model_handle_decode(&handle, payload,
                                                     (size_t)payload_len) ==
                SYNURANG_LITE_OK &&
            report_ok(handle.field_report, "CompileModel")) {
            compiled_id = handle.field_compiled_model_id;
            if (handle.field_report && handle.field_report->field_route) {
                const VolvoxaiV1RouteEvidence* route = handle.field_report->field_route;
                printf("compiled on %.*s: %u nodes, %u selected, attested=%d\n",
                       (int)route->field_provider.len,
                       route->field_provider.data
                           ? (const char*)route->field_provider.data : "",
                       route->field_active_nodes, route->field_selected_nodes,
                       route->field_attested);
            }
        }
        volvoxai_v1_compiled_model_handle_free(&handle);
        vx_call_free(&client, payload);
        payload = NULL;
    }
    if (!compiled_id) goto cleanup;

    {
        VolvoxaiV1CreateExecutionContextRequest request;
        volvoxai_v1_create_execution_context_request_init(&request);
        request.field_compiled_model_id = compiled_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_EXECUTION_CONTEXT,
                        volvoxai_v1_create_execution_context_request,
                        &request, payload, payload_len);
        volvoxai_v1_create_execution_context_request_free(&request);
    }
    if (!payload) {
        print_error(&client, "CreateExecutionContext");
        goto cleanup;
    }
    {
        VolvoxaiV1ExecutionContextHandle handle;
        size_t index;
        volvoxai_v1_execution_context_handle_init(&handle);
        if (volvoxai_v1_execution_context_handle_decode(&handle, payload,
                                                        (size_t)payload_len) ==
                SYNURANG_LITE_OK &&
            report_ok(handle.field_report, "CreateExecutionContext")) {
            context_id = handle.field_context_id;
            /* The declared inputs travel with the handle, so a caller learns
             * what to bind without a second round trip. */
            for (index = 0; index < handle.field_inputs.len; index++) {
                const VolvoxaiV1TensorSpec* spec = &handle.field_inputs.data[index];
                printf("input %.*s dtype=%d rank=%zu\n",
                       (int)spec->field_name.len,
                       spec->field_name.data ? (const char*)spec->field_name.data : "",
                       (int)spec->field_dtype, spec->field_dimensions.len);
            }
        }
        volvoxai_v1_execution_context_handle_free(&handle);
        vx_call_free(&client, payload);
        payload = NULL;
    }
    if (!context_id) goto cleanup;
    status = 0;

cleanup:
    vx_call_free(&client, payload);
    vx_call_client_close(&client);

    printf("%s\n", status == 0 ? "ok" : "failed");
    return status;
}
