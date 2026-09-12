/* Link and invoke generated operations plus the provider-composition SPI from
 * a raw libvolvoxai archive. No internal Runtime pointer is available here. */
#include <stdio.h>
#include <string.h>

#include "host_backend.h"
#include "volvoxai_ffi.h"
#include "../cli/call_client.h"
#include "../src/generated/proto_methods.h"
#include <assert.h>
#include <stdlib.h>

#include "volvoxai_lite.h"

static VxCallClient client;
static void close_client(void) { vx_call_client_close(&client); }

static int report_ok(const VolvoxaiV1OperationReport* report) {
    return report && report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK;
}

static int bytes_equal_text(const SynurangLiteBytes* value, const char* text) {
    const size_t length = strlen(text);
    return value && value->len == length && value->data &&
        memcmp(value->data, text, length) == 0;
}

static int positive_decimal(const SynurangLiteBytes* value) {
    size_t index;
    if (!value || !value->data || value->len == 0u ||
        value->data[0] < (uint8_t)'1' || value->data[0] > (uint8_t)'9') {
        return 0;
    }
    for (index = 1u; index < value->len; index++) {
        if (value->data[index] < (uint8_t)'0' ||
            value->data[index] > (uint8_t)'9') {
            return 0;
        }
    }
    return 1;
}

static uint8_t* create_runtime_with_domain_capture(int32_t* out_len) {
    VolvoxaiV1CreateRuntimeRequest request;
    VolvoxaiV1MemoryCaptureOptions capture;
    const SynurangLiteAllocator* allocator;
    uint8_t* encoded = NULL;
    uint8_t* response = NULL;
    size_t encoded_len = 0u;

    volvoxai_v1_create_runtime_request_init(&request);
    volvoxai_v1_memory_capture_options_init(&capture);
    allocator = request._allocator;
    if (synurang_lite_bytes_assign(
            capture._allocator, &capture.field_protocol,
            "volvoxai-memory-capture/v1",
            sizeof("volvoxai-memory-capture/v1") - 1u) != SYNURANG_LITE_OK) {
        goto done;
    }
    capture.field_include_domain_attestation = 1;
    request.field_memory_capture = &capture;
    if (volvoxai_v1_create_runtime_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
        goto done;
    }
    response = vx_call_bytes(&client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME,
        encoded, (int32_t)encoded_len, out_len);

done:
    request.field_memory_capture = NULL;
    if (encoded) allocator->deallocate(allocator->context, encoded);
    volvoxai_v1_memory_capture_options_free(&capture);
    volvoxai_v1_create_runtime_request_free(&request);
    return response;
}

static uint8_t* load_model(int64_t runtime_id,
                           const char* graph_path,
                           int32_t* out_len) {
    VolvoxaiV1LoadModelRequest request;
    const SynurangLiteAllocator* allocator;
    uint8_t* encoded = NULL;
    uint8_t* response = NULL;
    size_t encoded_len = 0u;

    volvoxai_v1_load_model_request_init(&request);
    allocator = request._allocator;
    request.field_runtime_id = runtime_id;
    if (synurang_lite_bytes_assign(
            allocator, &request.field_graph_path,
            graph_path, strlen(graph_path)) != SYNURANG_LITE_OK ||
        volvoxai_v1_load_model_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
        goto done;
    }
    response = vx_call_bytes(&client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL,
        encoded, (int32_t)encoded_len, out_len);

done:
    if (encoded) allocator->deallocate(allocator->context, encoded);
    volvoxai_v1_load_model_request_free(&request);
    return response;
}

static uint8_t* compile_with_example_provider(int64_t model_id,
                                              int32_t* out_len) {
    VolvoxaiV1BackendPolicy policy;
    const SynurangLiteAllocator* allocator;
    SynurangLiteBytes* backend;
    uint8_t* encoded = NULL;
    uint8_t* response = NULL;
    size_t encoded_len = 0u;

    volvoxai_v1_backend_policy_init(&policy);
    allocator = policy._allocator;
    policy.field_mode = VOLVOXAI_V1_BACKEND_POLICY_MODE_REQUIRE;
    policy.field_operator_fallback = VOLVOXAI_V1_OPERATOR_FALLBACK_FORBID;
    backend = volvoxai_v1_backend_policy_add_backends(&policy);
    if (!backend || synurang_lite_bytes_assign(
            allocator, backend, "example-host",
            sizeof("example-host") - 1u) != SYNURANG_LITE_OK ||
        volvoxai_v1_backend_policy_encode(
            &policy, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
        goto done;
    }
    {
        VolvoxaiV1CompileModelRequest call_request;
        volvoxai_v1_compile_model_request_init(&call_request);
        call_request.field_model_id = model_id;
        VolvoxaiV1BackendPolicy policy;
        volvoxai_v1_backend_policy_init(&policy);
        if ((encoded) && ((int32_t)encoded_len) > 0) {
            assert(volvoxai_v1_backend_policy_decode(&policy, encoded, (size_t)((int32_t)encoded_len)) == SYNURANG_LITE_OK);
            call_request.field_policy = &policy;
        }
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL,
            volvoxai_v1_compile_model_request, &call_request, response, *out_len);
        volvoxai_v1_backend_policy_free(&policy);
    }

done:
    if (encoded) allocator->deallocate(allocator->context, encoded);
    volvoxai_v1_backend_policy_free(&policy);
    return response;
}

static int example_attestation_ok(const VolvoxaiV1OperationReport* report) {
    const VolvoxaiV1MemoryEvidence* evidence;
    const VolvoxaiV1MemorySnapshot* snapshot;
    const VolvoxaiV1MemoryDomainAttestation* attestation;
    int found_maximum_tensor = 0;
    int found_provider_resident = 0;
    size_t index;

    if (!report_ok(report) ||
        !bytes_equal_text(&report->field_backend, "example-host") ||
        !(evidence = report->field_memory_evidence) ||
        evidence->field_snapshots.len != 1u ||
        !(snapshot = evidence->field_snapshots.data) ||
        snapshot->field_stage != VOLVOXAI_V1_OPERATION_STAGE_COMPILE ||
        snapshot->field_point != VOLVOXAI_V1_MEMORY_SNAPSHOT_POINT_AFTER ||
        !snapshot->field_subject ||
        snapshot->field_subject->field_kind !=
            VOLVOXAI_V1_MEMORY_OWNER_KIND_COMPILED_MODEL ||
        !positive_decimal(&snapshot->field_subject->field_owner_id) ||
        !(attestation = snapshot->field_domain_attestation) ||
        attestation->field_bounds.len != 2u) {
        fprintf(stderr, "FAIL external provider compile memory envelope\n");
        return 0;
    }

    for (index = 0u; index < attestation->field_bounds.len; index++) {
        const VolvoxaiV1MemoryBoundProof* bound =
            &attestation->field_bounds.data[index];
        if (bytes_equal_text(&bound->field_budget_domain_id,
                             "maximum-tensor")) {
            if (found_maximum_tensor ||
                bound->field_kind !=
                    VOLVOXAI_V1_MEMORY_BOUND_KIND_MAXIMUM_TENSOR ||
                !bound->field_maximum_bytes ||
                bound->field_maximum_bytes->field_bytes != sizeof(float) * 2u ||
                bound->field_limit_bytes) {
                fprintf(stderr, "FAIL external provider maximum-tensor bound\n");
                return 0;
            }
            found_maximum_tensor = 1;
        } else if (bytes_equal_text(&bound->field_budget_domain_id,
                                    "provider-resident")) {
            if (found_provider_resident ||
                bound->field_kind !=
                    VOLVOXAI_V1_MEMORY_BOUND_KIND_ORDINARY_RESIDENT ||
                !bound->field_maximum_bytes ||
                bound->field_maximum_bytes->field_bytes != sizeof(float) * 6u ||
                !bound->field_limit_bytes ||
                bound->field_limit_bytes->field_bytes != 1024u) {
                fprintf(stderr, "FAIL external provider resident/limit bound\n");
                return 0;
            }
            found_provider_resident = 1;
        } else {
            fprintf(stderr, "FAIL unexpected external provider memory bound\n");
            return 0;
        }
    }
    if (!found_maximum_tensor || !found_provider_resident) {
        fprintf(stderr, "FAIL incomplete external provider memory bounds\n");
        return 0;
    }
    return 1;
}

int main(int argc, char** argv) {
    if (!vx_call_client_open(&client)) return 1;
    atexit(close_client);
    uint8_t* response;
    int response_len = 0;
    int64_t runtime_id = 0;
    int64_t model_id = 0;
    int64_t compiled_model_id = 0;
    VxReport provider_report = VX_REPORT_INIT;

    if (argc != 2) {
        fprintf(stderr, "usage: %s <example.graph.json>\n", argv[0]);
        return 1;
    }

    if (volvoxai_example_host_backend_register(&provider_report) !=
            VX_STATUS_OK) {
        fprintf(stderr, "FAIL provider composition: %s\n",
                provider_report.message);
        return 1;
    }

    response = create_runtime_with_domain_capture(&response_len);
    if (!response) {
        fprintf(stderr, "FAIL generated CreateRuntime returned no payload\n");
        return 1;
    }

    {
        VolvoxaiV1RuntimeHandle handle;
        volvoxai_v1_runtime_handle_init(&handle);
        if (volvoxai_v1_runtime_handle_decode(
                &handle, (const uint8_t*)response, (size_t)response_len) !=
                SYNURANG_LITE_OK || !report_ok(handle.field_report) ||
            handle.field_runtime_id <= 0) {
            fprintf(stderr, "FAIL generated CreateRuntime response\n");
            volvoxai_v1_runtime_handle_free(&handle);
            vx_call_free(&client, (uint8_t*)response);
            return 1;
        }
        runtime_id = handle.field_runtime_id;
        volvoxai_v1_runtime_handle_free(&handle);
        vx_call_free(&client, (uint8_t*)response);
    }

    {
        VolvoxaiV1RuntimeRef call_request;
        volvoxai_v1_runtime_ref_init(&call_request);
        call_request.field_runtime_id = runtime_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_LIST_BACKENDS,
            volvoxai_v1_runtime_ref, &call_request, response, response_len);
    }
    if (!response) {
        fprintf(stderr, "FAIL generated ListBackends returned no payload\n");
        return 1;
    }
    {
        VolvoxaiV1BackendList list;
        int found = 0;
        size_t index;
        volvoxai_v1_backend_list_init(&list);
        if (volvoxai_v1_backend_list_decode(
                &list, (const uint8_t*)response, (size_t)response_len) !=
                SYNURANG_LITE_OK || !report_ok(list.field_report)) {
            fprintf(stderr, "FAIL generated ListBackends response\n");
            volvoxai_v1_backend_list_free(&list);
            vx_call_free(&client, (uint8_t*)response);
            return 1;
        }
        for (index = 0; index < list.field_backends.len; index++) {
            const SynurangLiteBytes* name = &list.field_backends.data[index];
            if (name->len == sizeof("example-host") - 1u && name->data &&
                !memcmp(name->data, "example-host", name->len)) {
                found = 1;
                break;
            }
        }
        volvoxai_v1_backend_list_free(&list);
        vx_call_free(&client, (uint8_t*)response);
        if (!found) {
            fprintf(stderr, "FAIL composed provider missing from ListBackends\n");
            return 1;
        }
    }

    response = load_model(runtime_id, argv[1], &response_len);
    if (!response) {
        fprintf(stderr, "FAIL generated LoadModel returned no payload\n");
        return 1;
    }
    {
        VolvoxaiV1ModelHandle handle;
        volvoxai_v1_model_handle_init(&handle);
        if (volvoxai_v1_model_handle_decode(
                &handle, (const uint8_t*)response, (size_t)response_len) !=
                SYNURANG_LITE_OK || !report_ok(handle.field_report) ||
            handle.field_model_id <= 0) {
            fprintf(stderr, "FAIL generated example LoadModel response\n");
            volvoxai_v1_model_handle_free(&handle);
            vx_call_free(&client, (uint8_t*)response);
            return 1;
        }
        model_id = handle.field_model_id;
        volvoxai_v1_model_handle_free(&handle);
        vx_call_free(&client, (uint8_t*)response);
    }

    response = compile_with_example_provider(model_id, &response_len);
    if (!response) {
        fprintf(stderr, "FAIL generated CompileModel returned no payload\n");
        return 1;
    }
    {
        VolvoxaiV1CompiledModelHandle handle;
        volvoxai_v1_compiled_model_handle_init(&handle);
        if (volvoxai_v1_compiled_model_handle_decode(
                &handle, (const uint8_t*)response, (size_t)response_len) !=
                SYNURANG_LITE_OK || handle.field_compiled_model_id <= 0 ||
            !example_attestation_ok(handle.field_report)) {
            fprintf(stderr,
                    "FAIL generated example CompileModel attestation response\n");
            volvoxai_v1_compiled_model_handle_free(&handle);
            vx_call_free(&client, (uint8_t*)response);
            return 1;
        }
        compiled_model_id = handle.field_compiled_model_id;
        volvoxai_v1_compiled_model_handle_free(&handle);
        vx_call_free(&client, (uint8_t*)response);
    }

    response = vx_call_bytes(&client, "/volvoxai.v1.VxPlatformService/GetPlatformInfo", NULL, 0,
        &response_len);
    if (!response || response_len <= 0) {
        fprintf(stderr, "FAIL static GetPlatformInfo returned %d bytes\n",
                response_len);
        if (response) vx_call_free(&client, response);
        return 1;
    }
    vx_call_free(&client, response);
    {
        VolvoxaiV1CompiledModelRef call_request;
        volvoxai_v1_compiled_model_ref_init(&call_request);
        call_request.field_compiled_model_id = compiled_model_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_COMPILED_MODEL,
            volvoxai_v1_compiled_model_ref, &call_request, response, response_len);
    }
    if (!response) {
        fprintf(stderr, "FAIL generated ReleaseCompiledModel returned no payload\n");
        return 1;
    }
    vx_call_free(&client, (uint8_t*)response);
    {
        VolvoxaiV1ModelRef call_request;
        volvoxai_v1_model_ref_init(&call_request);
        call_request.field_model_id = model_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_MODEL,
            volvoxai_v1_model_ref, &call_request, response, response_len);
    }
    if (!response) {
        fprintf(stderr, "FAIL generated ReleaseModel returned no payload\n");
        return 1;
    }
    vx_call_free(&client, (uint8_t*)response);
    {
        VolvoxaiV1RuntimeRef call_request;
        volvoxai_v1_runtime_ref_init(&call_request);
        call_request.field_runtime_id = runtime_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RUNTIME,
            volvoxai_v1_runtime_ref, &call_request, response, response_len);
    }
    if (!response) {
        fprintf(stderr, "FAIL generated ReleaseRuntime returned no payload\n");
        return 1;
    }
    vx_call_free(&client, (uint8_t*)response);
    puts("static generated API and provider composition passed");
    return 0;
}
