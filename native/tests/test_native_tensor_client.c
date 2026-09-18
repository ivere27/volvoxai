/* A language-independent consumer: generated messages and native dispatch
 * only. No Python, DLPack adapter, CUDA SDK, or framework dependency. */
#define _POSIX_C_SOURCE 200809L
#include "../cli/call_client.h"
#include "../src/generated/proto_methods.h"
#include "../third_party/dlpack/dlpack.h"
#include <stdlib.h>
#include <time.h>

static int good(const VolvoxaiV1OperationReport* report) {
    if (report && report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK) return 1;
    fprintf(stderr, "native operation failed: %.*s\n", report ? (int)report->field_message.len : 0,
        report && report->field_message.data ? (const char*)report->field_message.data : "");
    return 0;
}
static double now_ms(void) {
    struct timespec value;
    clock_gettime(CLOCK_MONOTONIC, &value);
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}
static void count_dlpack_release(DLManagedTensor* tensor) {
    ++*(int*)tensor->manager_ctx;
}
#define CALL(method, request_type, request, response_type, response) do { \
    uint8_t* payload = NULL; int32_t size = 0; \
    VX_CALL_MESSAGE(&client, method, request_type, request, payload, size); \
    if (!payload) { fprintf(stderr, "native transport failed\n"); goto done; } \
    int decoded = response_type##_decode(response, payload, (size_t)size); \
    vx_call_free(&client, payload); \
    if (decoded != SYNURANG_LITE_OK || !good((response)->field_report)) goto done; \
} while (0)

int main(int argc, char** argv) {
    const char* backend = argc > 1 ? argv[1] : "cpu";
    int iterations = argc > 2 ? atoi(argv[2]) : 4;
    if (iterations < 1) return 2;
    VxCallClient client;
    if (!vx_call_client_open(&client)) return 1;
    int status = 1;
    VolvoxaiV1RuntimeHandle runtime;
    VolvoxaiV1ModelHandle model;
    VolvoxaiV1CompiledModelHandle compiled;
    VolvoxaiV1ExecutionContextHandle context;
    VolvoxaiV1TensorBatch first, second, feedback_result;
    VolvoxaiV1TensorBatch read;
    VolvoxaiV1TensorBatch imported;
    VolvoxaiV1DLPackExport exported;
    VolvoxaiV1BufferAccess access;
    VolvoxaiV1OperationReport completion_report;
    DLManagedTensor managed = {0};
    DLManagedTensor* consumer = NULL;
    int producer_releases = 0;
    volvoxai_v1_runtime_handle_init(&runtime);
    volvoxai_v1_model_handle_init(&model);
    volvoxai_v1_compiled_model_handle_init(&compiled);
    volvoxai_v1_execution_context_handle_init(&context);
    volvoxai_v1_tensor_batch_init(&first);
    volvoxai_v1_tensor_batch_init(&second);
    volvoxai_v1_tensor_batch_init(&feedback_result);
    volvoxai_v1_tensor_batch_init(&read);
    volvoxai_v1_tensor_batch_init(&imported);
    volvoxai_v1_dl_pack_export_init(&exported);
    volvoxai_v1_buffer_access_init(&access);
    volvoxai_v1_operation_report_init(&completion_report);
    VolvoxaiV1CreateRuntimeRequest create;
    volvoxai_v1_create_runtime_request_init(&create);
    create.field_cpu_threads = 2;
    create.has_execution_mode = 1;
    create.field_execution_mode = VOLVOXAI_V1_EXECUTION_MODE_DIRECT;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME, volvoxai_v1_create_runtime_request,
        &create, volvoxai_v1_runtime_handle, &runtime);

    const char graph[] = "{\"format\":\"volvox-graph/v1\",\"dimensions\":{},\"inputs\":{\"x\":{\"shape\":[2,4],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\",\"inputs\":{\"a\":\"x\",\"b\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"shape\":[2,4],\"dtype\":\"float32\"}},\"params\":{}}],\"outputs\":[\"y\"]}";
    VolvoxaiV1ModelPackage package;
    VolvoxaiV1LoadModelRequest load;
    volvoxai_v1_model_package_init(&package);
    volvoxai_v1_load_model_request_init(&load);
    package.field_graph_document.data = (uint8_t*)graph;
    package.field_graph_document.len = sizeof(graph) - 1;
    load.field_runtime_id = runtime.field_runtime_id;
    load.field_package = &package;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL, volvoxai_v1_load_model_request,
        &load, volvoxai_v1_model_handle, &model);
    VolvoxaiV1BackendPolicy policy;
    VolvoxaiV1CompileModelRequest compile;
    volvoxai_v1_backend_policy_init(&policy);
    volvoxai_v1_compile_model_request_init(&compile);
    SynurangLiteBytes selected = {.data = (uint8_t*)backend, .len = strlen(backend)};
    policy.field_backends.data = &selected;
    policy.field_backends.len = 1;
    policy.field_mode = VOLVOXAI_V1_BACKEND_POLICY_MODE_REQUIRE;
    policy.field_operator_fallback = VOLVOXAI_V1_OPERATOR_FALLBACK_FORBID;
    compile.field_model_id = model.field_model_id;
    compile.field_policy = &policy;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL, volvoxai_v1_compile_model_request,
        &compile, volvoxai_v1_compiled_model_handle, &compiled);
    VolvoxaiV1CreateExecutionContextRequest create_context;
    volvoxai_v1_create_execution_context_request_init(&create_context);
    create_context.field_compiled_model_id = compiled.field_compiled_model_id;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_CREATE_EXECUTION_CONTEXT, volvoxai_v1_create_execution_context_request,
        &create_context, volvoxai_v1_execution_context_handle, &context);

    float input[8] = {0,1,2,3,4,5,6,7};
    int64_t shape[2] = {2,4};
    VolvoxaiV1Tensor tensor;
    VolvoxaiV1ExecuteTensorsRequest execute;
    volvoxai_v1_tensor_init(&tensor);
    volvoxai_v1_execute_tensors_request_init(&execute);
    tensor.field_name.data = (uint8_t*)"x";
    tensor.field_name.len = 1;
    tensor.field_shape.data = shape;
    tensor.field_shape.len = 2;
    tensor.field_dtype = VOLVOXAI_V1_DATA_TYPE_F32;
    tensor.which_payload = 5;
    tensor.field_inline.data = (uint8_t*)input;
    tensor.field_inline.len = sizeof(input);
    execute.field_context_id = context.field_context_id;
    execute.field_inputs.data = &tensor;
    execute.field_inputs.len = 1;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_EXECUTE_TENSORS, volvoxai_v1_execute_tensors_request,
        &execute, volvoxai_v1_tensor_batch, &first);
    if (first.field_outputs.len != 1 || !first.field_outputs.data[0].field_buffer) goto done;
    VolvoxaiV1Tensor shared = first.field_outputs.data[0];
    shared.field_name = tensor.field_name;
    execute.field_inputs.data = &shared;
    double elapsed = 0;
    for (int index = -4; index < iterations; index++) {
        double started = now_ms();
        CALL(VX_RPC_VX_INFERENCE_SERVICE_EXECUTE_TENSORS, volvoxai_v1_execute_tensors_request,
            &execute, volvoxai_v1_tensor_batch, &second);
        if (index >= 0) elapsed += now_ms() - started;
        if (second.field_outputs.len != 1) goto done;
        if (index == iterations - 1) break;
        VolvoxaiV1BufferRefs release;
        volvoxai_v1_buffer_refs_init(&release);
        int64_t tensor_id = second.field_outputs.data[0].field_buffer->field_buffer_id;
        release.field_buffer_ids.data = &tensor_id;
        release.field_buffer_ids.len = release.field_buffer_ids.cap = 1;
        uint8_t* payload; int32_t size;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_BUFFER_SERVICE_RELEASE_BUFFERS,
            volvoxai_v1_buffer_refs, &release, payload, size);
        if (!payload) goto done;
        vx_call_free(&client, payload);
        volvoxai_v1_tensor_batch_free(&second);
        volvoxai_v1_tensor_batch_init(&second);
    }
    /* Feedback works through the same generated C dispatch, including an
     * intermediate execution that publishes no tensor handles. */
    VolvoxaiV1TensorFeedback feedback;
    VolvoxaiV1TensorOutputSelection selection;
    volvoxai_v1_tensor_feedback_init(&feedback);
    volvoxai_v1_tensor_output_selection_init(&selection);
    feedback.field_input_name = tensor.field_name;
    feedback.field_output_name.data = (uint8_t*)"y";
    feedback.field_output_name.len = 1;
    execute.field_inputs.len = 0;
    execute.field_feedback.data = &feedback;
    execute.field_feedback.len = 1;
    execute.field_outputs = &selection;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_EXECUTE_TENSORS, volvoxai_v1_execute_tensors_request,
        &execute, volvoxai_v1_tensor_batch, &feedback_result);
    if (feedback_result.field_outputs.len) goto done;
    volvoxai_v1_tensor_batch_free(&feedback_result);
    volvoxai_v1_tensor_batch_init(&feedback_result);
    execute.field_outputs = NULL;
    CALL(VX_RPC_VX_INFERENCE_SERVICE_EXECUTE_TENSORS, volvoxai_v1_execute_tensors_request,
        &execute, volvoxai_v1_tensor_batch, &feedback_result);
    if (feedback_result.field_outputs.len != 1) goto done;
    /* Retire the context before reading independently retained snapshots. */
    VolvoxaiV1ExecutionContextRef release_context;
    volvoxai_v1_execution_context_ref_init(&release_context);
    release_context.field_context_id = context.field_context_id;
    uint8_t* payload; int32_t size;
    VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_EXECUTION_CONTEXT,
        volvoxai_v1_execution_context_ref, &release_context, payload, size);
    if (!payload) goto done;
    vx_call_free(&client, payload);
    for (int index = 0; index < 3; index++) {
        VolvoxaiV1CopyTensorsRequest request;
        volvoxai_v1_copy_tensors_request_init(&request);
        request.field_sources.data = (index == 2 ? feedback_result : index ? second : first).field_outputs.data;
        request.field_sources.len = 1;
        request.field_inline_result = 1;
        CALL(VX_RPC_VX_BUFFER_SERVICE_COPY_TENSORS, volvoxai_v1_copy_tensors_request,
            &request, volvoxai_v1_tensor_batch, &read);
        if (read.field_outputs.len != 1 || read.field_outputs.data[0].field_inline.len != sizeof(input)) goto done;
        float values[8];
        memcpy(values, read.field_outputs.data[0].field_inline.data, sizeof(values));
        for (int element = 0; element < 8; element++)
            if (values[element] != input[element] * (index == 2 ? 16 : index ? 4 : 2)) goto done;
        volvoxai_v1_tensor_batch_free(&read);
        volvoxai_v1_tensor_batch_init(&read);
    }
    /* Scoped CPU/CUDA DLPack is the same C API that language wrappers use.
     * Ending permission permits native reads while the capsule stays alive. */
    if (!strcmp(backend, "cpu") || !strcmp(backend, "cuda")) {
        VolvoxaiV1BufferAccessRequest begin;
        volvoxai_v1_buffer_access_request_init(&begin);
        begin.field_view = first.field_outputs.data[0].field_buffer;
        begin.field_mode = VOLVOXAI_V1_BUFFER_ACCESS_MODE_WRITE;
        CALL(VX_RPC_VX_BUFFER_SERVICE_BEGIN_BUFFER_ACCESS, volvoxai_v1_buffer_access_request,
            &begin, volvoxai_v1_buffer_access, &access);
        VolvoxaiV1ExportDLPackRequest export_scoped;
        volvoxai_v1_export_dl_pack_request_init(&export_scoped);
        export_scoped.field_tensor = &first.field_outputs.data[0];
        export_scoped.field_access_id = access.field_access_id;
        CALL(VX_RPC_VX_BUFFER_SERVICE_EXPORT_DLPACK, volvoxai_v1_export_dl_pack_request,
            &export_scoped, volvoxai_v1_dl_pack_export, &exported);
        consumer = (DLManagedTensor*)(uintptr_t)exported.field_managed_tensor;
        if (!consumer || (uintptr_t)consumer->dl_tensor.data !=
                access.field_memory->field_resource->field_handle + access.field_memory->field_offset_bytes) goto done;
        VolvoxaiV1EndBufferAccessRequest end;
        VolvoxaiV1CudaStreamCompletion cuda;
        volvoxai_v1_end_buffer_access_request_init(&end);
        volvoxai_v1_cuda_stream_completion_init(&cuda);
        uint64_t stream = 1; /* Explicit legacy default; never host-thread-relative 0. */
        end.field_access_ids.data = &access.field_access_id;
        end.field_access_ids.len = 1;
        if (!strcmp(backend, "cuda")) {
            cuda.field_device_id = access.field_memory->field_resource->field_device_id;
            cuda.field_streams.data = &stream;
            cuda.field_streams.len = 1;
            end.field_cuda = &cuda;
        }
        VX_CALL_MESSAGE(&client, VX_RPC_VX_BUFFER_SERVICE_END_BUFFER_ACCESS,
            volvoxai_v1_end_buffer_access_request, &end, payload, size);
        if (!payload) goto done;
        int decoded = volvoxai_v1_operation_report_decode(&completion_report, payload, (size_t)size);
        vx_call_free(&client, payload);
        if (decoded != SYNURANG_LITE_OK || !good(&completion_report)) goto done;
        /* The consumer must no longer dereference its DLPack view. */
        VolvoxaiV1CopyTensorsRequest copy;
        volvoxai_v1_copy_tensors_request_init(&copy);
        copy.field_sources.data = first.field_outputs.data;
        copy.field_sources.len = 1;
        copy.field_inline_result = 1;
        CALL(VX_RPC_VX_BUFFER_SERVICE_COPY_TENSORS, volvoxai_v1_copy_tensors_request,
            &copy, volvoxai_v1_tensor_batch, &read);
        if (read.field_outputs.len != 1 || read.field_outputs.data[0].field_inline.len != sizeof(input)) goto done;
        float values[8];
        memcpy(values, read.field_outputs.data[0].field_inline.data, sizeof(values));
        for (int element = 0; element < 8; element++) if (values[element] != input[element] * 2) goto done;
        consumer->deleter(consumer);
        consumer = NULL;
        volvoxai_v1_dl_pack_export_free(&exported);
        volvoxai_v1_dl_pack_export_init(&exported);
        volvoxai_v1_tensor_batch_free(&read);
        volvoxai_v1_tensor_batch_init(&read);
    }
    /* Bulk retirement is also part of the generated C API. */
    {
        int64_t ids[3] = {first.field_outputs.data[0].field_buffer->field_buffer_id,
                         second.field_outputs.data[0].field_buffer->field_buffer_id,
                         feedback_result.field_outputs.data[0].field_buffer->field_buffer_id};
        VolvoxaiV1BufferRefs release;
        volvoxai_v1_buffer_refs_init(&release);
        release.field_buffer_ids.data = ids;
        release.field_buffer_ids.len = release.field_buffer_ids.cap = 3;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_BUFFER_SERVICE_RELEASE_BUFFERS,
            volvoxai_v1_buffer_refs, &release, payload, size);
        if (!payload) goto done;
        vx_call_free(&client, payload);
    }
    /* A standard C DLPack producer needs no Python bridge or framework. The
     * consumer lease outlives the imported public capability. */
    managed.dl_tensor = (DLTensor){.data = input, .device = {kDLCPU, 0},
        .ndim = 2, .dtype = {kDLFloat, 32, 1}, .shape = shape};
    managed.manager_ctx = &producer_releases;
    managed.deleter = count_dlpack_release;
    VolvoxaiV1ImportDLPackRequest import_request;
    volvoxai_v1_import_dl_pack_request_init(&import_request);
    import_request.field_managed_tensor = (uint64_t)(uintptr_t)&managed;
    CALL(VX_RPC_VX_BUFFER_SERVICE_IMPORT_DLPACK, volvoxai_v1_import_dl_pack_request,
        &import_request, volvoxai_v1_tensor_batch, &imported);
    if (imported.field_outputs.len != 1) goto done;
    VolvoxaiV1ExportDLPackRequest export_request;
    volvoxai_v1_export_dl_pack_request_init(&export_request);
    export_request.field_tensor = &imported.field_outputs.data[0];
    CALL(VX_RPC_VX_BUFFER_SERVICE_EXPORT_DLPACK, volvoxai_v1_export_dl_pack_request,
        &export_request, volvoxai_v1_dl_pack_export, &exported);
    consumer = (DLManagedTensor*)(uintptr_t)exported.field_managed_tensor;
    if (!consumer || consumer->dl_tensor.data != input) goto done;
    VolvoxaiV1BufferRefs release_import;
    volvoxai_v1_buffer_refs_init(&release_import);
    int64_t imported_id = imported.field_outputs.data[0].field_buffer->field_buffer_id;
    release_import.field_buffer_ids.data = &imported_id;
    release_import.field_buffer_ids.len = 1;
    VX_CALL_MESSAGE(&client, VX_RPC_VX_BUFFER_SERVICE_RELEASE_BUFFERS,
        volvoxai_v1_buffer_refs, &release_import, payload, size);
    if (!payload) goto done;
    vx_call_free(&client, payload);
    if (producer_releases || ((float*)consumer->dl_tensor.data)[7] != 7) goto done;
    consumer->deleter(consumer);
    consumer = NULL;
    if (producer_releases != 1) goto done;
    printf("{\"backend\":\"%s\",\"client\":\"native-c\",\"iterations\":%d,\"execute_mean_ms\":%.6f,\"correct\":true}\n",
        backend, iterations, elapsed / iterations);
    status = 0;
done:
    if (consumer) consumer->deleter(consumer);
    volvoxai_v1_buffer_access_free(&access);
    volvoxai_v1_operation_report_free(&completion_report);
    volvoxai_v1_dl_pack_export_free(&exported);
    volvoxai_v1_tensor_batch_free(&imported);
    volvoxai_v1_tensor_batch_free(&feedback_result);
    volvoxai_v1_tensor_batch_free(&read);
    volvoxai_v1_tensor_batch_free(&second);
    volvoxai_v1_tensor_batch_free(&first);
    volvoxai_v1_execution_context_handle_free(&context);
    volvoxai_v1_compiled_model_handle_free(&compiled);
    volvoxai_v1_model_handle_free(&model);
    volvoxai_v1_runtime_handle_free(&runtime);
    vx_call_client_close(&client);
    return status;
}
