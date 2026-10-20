/* End-to-end check of the public protobuf API against a real model.
 *
 * Application-facing checks go through the generated C dispatch in
 * runtime/generated/c/volvoxai_ffi.c, exactly as an external C host
 * would. The engine's own vx_* lifecycle is never called directly, so a
 * regression in the handler tables, the converters or the schema shows up
 * here rather than only in a downstream host. One guarded engine-internal
 * check perturbs retained wire headers to enforce the public plan-identity
 * independence from private ABI/layout metadata.
 *
 * Usage: test_proto_api_roundtrip <model-dir>
 */
#include "volvoxai_ffi.h"
#include "../cli/call_client.h"
#include "../src/generated/proto_methods.h"
#include <assert.h>
#include <stdlib.h>

#include "volvoxai_lite.h"
#include "../src/runtime/graph_bind.h"
#include "../src/runtime/graph_domain.h"
#include "../src/runtime/graph_plan.h"
#include "../src/runtime/public_api_internal.h"
#include "../src/runtime/vx_lifecycle.h"

#include <stdint.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

static VxCallClient client;
static inline const uint8_t* call_error(int32_t* size) { return vx_call_error(&client, size); }
static void close_client(void) { vx_call_client_close(&client); }

static int failures;

#define CHECK(cond, what)                                                      \
    do {                                                                       \
        if (!(cond)) {                                                         \
            fprintf(stderr, "FAIL: %s\n", (what));                             \
            failures++;                                                        \
        }                                                                      \
    } while (0)

static const char* last_error(void) {
    int32_t length = 0;
    const uint8_t* message = call_error(&length);
    static char buffer[512];
    if (!message || length <= 0) return "(no error)";
    if ((size_t)length >= sizeof(buffer)) length = (int32_t)sizeof(buffer) - 1;
    memcpy(buffer, message, (size_t)length);
    buffer[length] = '\0';
    return buffer;
}

/* Decodes one response payload and hands the caller ownership of nothing:
 * the payload is freed here, the decoded message by the caller. */
#define DECODE(payload, len, msg, prefix)                                      \
    (prefix##_init(&(msg)),                                                    \
     (payload) && prefix##_decode(&(msg), (payload), (size_t)(len)) ==         \
                      SYNURANG_LITE_OK)

static uint64_t open_wait_observer(int64_t request_id) {
    while (client.api->has_work(client.instance)) client.api->poll(client.instance, 64);
    VolvoxaiV1RequestRef request;
    volvoxai_v1_request_ref_init(&request);
    request.field_request_id = request_id;
    uint8_t* data = NULL;
    size_t size = 0;
    CHECK(volvoxai_v1_request_ref_encode(&request, &data, &size) == SYNURANG_LITE_OK,
          "WaitRequest observer request encodes");
    SynurangCallOptions options = { sizeof(options), 0, 0, 0, UINT64_MAX };
    uint64_t call = client.api->open(client.instance,
        VX_RPC_VX_SCHEDULER_SERVICE_WAIT_REQUEST, &options);
    CHECK(call != 0, "WaitRequest observer opens");
    if (call) {
        CHECK(client.api->send(client.instance, call, data, (uint32_t)size) == SYNURANG_OK,
              "WaitRequest observer accepts its request");
        CHECK(client.api->half_close(client.instance, call) == SYNURANG_OK,
              "WaitRequest observer accepts half-close");
        /* Run open and message separately. Their exact callback budgets leave
         * the retained observer pending even if the tiny model finished. */
        client.api->poll(client.instance, 1);
        client.api->poll(client.instance, 1);
        SynurangReadResult response = {0};
        CHECK(client.api->receive(client.instance, call, &response) == SYNURANG_OK &&
              response.kind == SYNURANG_READ_PENDING,
              "WaitRequest suspends after registering its observer");
        vx_call_free(&client, response.data);
    }
    synurang_lite_release(request._allocator, data);
    volvoxai_v1_request_ref_free(&request);
    return call;
}

static void cancel_wait_observer(int64_t request_id, int32_t code) {
    uint64_t call = open_wait_observer(request_id);
    if (!call) return;
    SynurangReadResult response = {0};
    CHECK(client.api->cancel(client.instance, call, code) == SYNURANG_OK,
          "WaitRequest observer accepts transport cancellation");
    CHECK(client.api->receive(client.instance, call, &response) == SYNURANG_OK &&
          response.kind == SYNURANG_READ_FINISHED && response.code == code,
          "WaitRequest returns the caller's terminal cancellation status");
    vx_call_free(&client, response.data);
    client.api->release(client.instance, call);
}

static void wait_progresses_during_metadata_backlog(int64_t request_id, uint32_t budget) {
    uint64_t wait = open_wait_observer(request_id);
    if (!wait) return;
    uint64_t metadata[64] = {0};
    SynurangCallOptions options = { sizeof(options), 0, 0, 0, UINT64_MAX };
    for (size_t index = 0; index < 64; ++index) {
        metadata[index] = client.api->open(client.instance,
            VX_RPC_VX_PLATFORM_SERVICE_GET_PLATFORM_INFO, &options);
        CHECK(metadata[index] && client.api->send(client.instance, metadata[index], NULL, 0) == SYNURANG_OK &&
              client.api->half_close(client.instance, metadata[index]) == SYNURANG_OK,
              "metadata backlog opens while a wait is suspended");
    }
    SynurangReadResult response = {0};
    for (unsigned turn = 0; turn < 2; ++turn) {
        CHECK(client.api->poll(client.instance, budget) <= budget,
              "module respects caller's callback budget");
        CHECK(client.api->receive(client.instance, wait, &response) == SYNURANG_OK,
              "ready wait remains readable during metadata backlog");
        if (response.kind != SYNURANG_READ_PENDING) break;
    }
    CHECK(response.kind == SYNURANG_READ_MESSAGE,
          "ready wait progresses before metadata backlog is drained");
    vx_call_free(&client, response.data);
    CHECK(client.api->has_work(client.instance), "metadata backlog is still outstanding");
    if (response.kind == SYNURANG_READ_MESSAGE) {
        CHECK(client.api->receive(client.instance, wait, &response) == SYNURANG_OK &&
              response.kind == SYNURANG_READ_FINISHED && response.code == 0,
              "fairly scheduled wait finishes successfully");
        vx_call_free(&client, response.data);
    }
    client.api->release(client.instance, wait);
    for (size_t index = 0; index < 64; ++index)
        if (metadata[index]) client.api->release(client.instance, metadata[index]);
}

static void report_summary(const char* label,
                           const VolvoxaiV1OperationReport* report) {
    if (!report) {
        printf("  %-22s (no report)\n", label);
        return;
    }
    printf("  %-22s status=%d stage=%d backend=%.*s\n", label,
           (int)report->field_status, (int)report->field_stage,
           (int)report->field_backend.len,
           report->field_backend.data ? (const char*)report->field_backend.data : "");
    if (report->field_route) {
        printf("    route provider=%.*s attested=%d nodes=%u selected=%u fallback=%u\n",
               (int)report->field_route->field_provider.len,
               report->field_route->field_provider.data
                   ? (const char*)report->field_route->field_provider.data : "",
               report->field_route->field_attested,
               report->field_route->field_active_nodes,
               report->field_route->field_selected_nodes,
               report->field_route->field_fallback_nodes);
    }
}

typedef struct PlanningWeightFixture {
    const char* name;
    VolvoxaiV1DataType dtype;
    const int64_t* shape;
    size_t rank;
} PlanningWeightFixture;

static int bytes_equal(const SynurangLiteBytes* left,
                       const SynurangLiteBytes* right) {
    return left && right && left->len == right->len &&
        (!left->len || memcmp(left->data, right->data, left->len) == 0);
}

static int bytes_equal_text(const SynurangLiteBytes* value,
                            const char* text) {
    size_t length = text ? strlen(text) : 0u;
    return value && value->len == length &&
        (!length || memcmp(value->data, text, length) == 0);
}

static int create_standalone_plan(
        const char* document,
        const PlanningWeightFixture* weights,
        size_t weight_count,
        VolvoxaiV1GraphPlanHandle* output) {
    const SynurangLiteAllocator* allocator = synurang_lite_default_allocator();
    VolvoxaiV1CreateGraphPlanRequest request;
    VolvoxaiV1GraphPlanningSource* source = NULL;
    uint8_t* encoded = NULL;
    uint8_t* payload = NULL;
    size_t encoded_len = 0u;
    int32_t payload_len = 0;
    int ok = 0;

    volvoxai_v1_graph_plan_handle_init(output);
    volvoxai_v1_create_graph_plan_request_init(&request);
    source = (VolvoxaiV1GraphPlanningSource*)allocator->allocate(
        allocator->context, sizeof(*source));
    if (!source) goto done;
    volvoxai_v1_graph_planning_source_init_with_allocator(source, allocator);
    request.which_source = 2; /* CreateGraphPlanRequest.graph */
    request.field_graph = source;
    source->which_definition_source = 1;
    if (synurang_lite_bytes_assign(
            allocator, &source->field_graph_document,
            document, strlen(document)) != SYNURANG_LITE_OK)
        goto done;
    for (size_t index = 0u; index < weight_count; ++index) {
        VolvoxaiV1PlanningWeight* target =
            volvoxai_v1_graph_planning_source_add_weights(source);
        if (!target || synurang_lite_bytes_assign(
                allocator, &target->field_name, weights[index].name,
                strlen(weights[index].name)) != SYNURANG_LITE_OK)
            goto done;
        target->field_dtype = weights[index].dtype;
        for (size_t axis = 0u; axis < weights[index].rank; ++axis) {
            int64_t* extent = volvoxai_v1_planning_weight_add_shape(target);
            if (!extent) goto done;
            *extent = weights[index].shape[axis];
        }
    }
    if (volvoxai_v1_create_graph_plan_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK)
        goto done;
    payload = vx_call_bytes(&client, VX_RPC_VX_PLANNING_SERVICE_CREATE_GRAPH_PLAN,
        encoded, (int32_t)encoded_len, &payload_len);
    if (!payload || volvoxai_v1_graph_plan_handle_decode(
            output, payload, (size_t)payload_len) != SYNURANG_LITE_OK)
        goto done;
    ok = 1;
done:
    synurang_lite_release(allocator, encoded);
    if (payload) vx_call_free(&client, payload);
    volvoxai_v1_create_graph_plan_request_free(&request);
    return ok;
}

static int resolve_minimum_plan(
        int64_t graph_plan_id,
        const char* bank_name,
        const uint32_t* bank_slots,
        size_t bank_slot_count,
        VolvoxaiV1ResolvedGraphPlan* output) {
    const SynurangLiteAllocator* allocator = synurang_lite_default_allocator();
    VolvoxaiV1ResolveGraphPlanRequest request;
    VolvoxaiV1Empty minimum;
    uint8_t* encoded = NULL;
    uint8_t* payload = NULL;
    size_t encoded_len = 0u;
    int32_t payload_len = 0;
    int ok = 0;

    volvoxai_v1_resolved_graph_plan_init(output);
    volvoxai_v1_resolve_graph_plan_request_init(&request);
    volvoxai_v1_empty_init(&minimum);
    request.field_graph_plan_id = graph_plan_id;
    request.which_binding = 2; /* ResolveGraphPlanRequest.minimum */
    request.field_minimum = &minimum;
    if (bank_name) {
        VolvoxaiV1BankResidency* bank =
            volvoxai_v1_resolve_graph_plan_request_add_bank_residency(
                &request);
        if (!bank || synurang_lite_bytes_assign(
                allocator, &bank->field_bank, bank_name,
                strlen(bank_name)) != SYNURANG_LITE_OK)
            goto done;
        for (size_t index = 0u; index < bank_slot_count; ++index) {
            uint32_t* slot = volvoxai_v1_bank_residency_add_slots(bank);
            if (!slot) goto done;
            *slot = bank_slots[index];
        }
    }
    if (volvoxai_v1_resolve_graph_plan_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK)
        goto done;
    request.field_minimum = NULL;
    payload = vx_call_bytes(&client, VX_RPC_VX_PLANNING_SERVICE_RESOLVE_GRAPH_PLAN,
        encoded, (int32_t)encoded_len, &payload_len);
    if (!payload || volvoxai_v1_resolved_graph_plan_decode(
            output, payload, (size_t)payload_len) != SYNURANG_LITE_OK)
        goto done;
    ok = 1;
done:
    request.field_minimum = NULL;
    synurang_lite_release(allocator, encoded);
    if (payload) vx_call_free(&client, payload);
    volvoxai_v1_resolve_graph_plan_request_free(&request);
    return ok;
}

static void release_graph_plan_if_live(int64_t graph_plan_id) {
    int32_t payload_len = 0;
    uint8_t* payload;
    if (graph_plan_id <= 0) return;
    {
        VolvoxaiV1GraphPlanRef call_request;
        volvoxai_v1_graph_plan_ref_init(&call_request);
        call_request.field_graph_plan_id = graph_plan_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_PLANNING_SERVICE_RELEASE_GRAPH_PLAN,
            volvoxai_v1_graph_plan_ref, &call_request, payload, payload_len);
    }
    if (payload) vx_call_free(&client, payload);
}

static int inspect_safetensors_prefix(
        const uint8_t* prefix,
        size_t prefix_bytes,
        uint64_t file_bytes,
        VolvoxaiV1SafetensorsInfo* output) {
    const SynurangLiteAllocator* allocator = synurang_lite_default_allocator();
    VolvoxaiV1InspectSafetensorsRequest request;
    VolvoxaiV1SafetensorsInlineHeaderSource source;
    uint8_t* encoded = NULL;
    uint8_t* payload = NULL;
    size_t encoded_len = 0u;
    int32_t payload_len = 0;
    int ok = 0;

    volvoxai_v1_safetensors_info_init(output);
    volvoxai_v1_inspect_safetensors_request_init(&request);
    volvoxai_v1_safetensors_inline_header_source_init(&source);
    request.which_source = 2; /* InspectSafetensorsRequest.inline_header */
    request.field_inline_header = &source;
    source.field_file_size = file_bytes;
    if (synurang_lite_bytes_assign(
            allocator, &source.field_header_prefix,
            prefix, prefix_bytes) != SYNURANG_LITE_OK)
        goto done;
    if (volvoxai_v1_inspect_safetensors_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK)
        goto done;
    payload = vx_call_bytes(&client, VX_RPC_VX_PLANNING_SERVICE_INSPECT_SAFETENSORS,
        encoded, (int32_t)encoded_len, &payload_len);
    if (!payload || volvoxai_v1_safetensors_info_decode(
            output, payload, (size_t)payload_len) != SYNURANG_LITE_OK)
        goto done;
    ok = 1;
done:
    request.field_inline_header = NULL;
    synurang_lite_release(allocator, encoded);
    if (payload) vx_call_free(&client, payload);
    volvoxai_v1_safetensors_inline_header_source_free(&source);
    volvoxai_v1_inspect_safetensors_request_free(&request);
    return ok;
}

static int inspect_safetensors_view(
        const uint8_t* prefix,
        uint64_t prefix_bytes,
        uint64_t file_bytes,
        VolvoxaiV1SafetensorsInfo* output) {
    const SynurangLiteAllocator* allocator = synurang_lite_default_allocator();
    VolvoxaiV1InspectSafetensorsRequest request;
    VolvoxaiV1SafetensorsHeaderViewSource source;
    VolvoxaiV1BorrowedBuffer view;
    uint8_t* encoded = NULL;
    uint8_t* payload = NULL;
    size_t encoded_len = 0u;
    int32_t payload_len = 0;
    int ok = 0;

    if (!prefix || (uintptr_t)prefix > (uintptr_t)INT64_MAX ||
        prefix_bytes > (uint64_t)INT64_MAX)
        return 0;
    volvoxai_v1_safetensors_info_init(output);
    volvoxai_v1_inspect_safetensors_request_init(&request);
    volvoxai_v1_safetensors_header_view_source_init(&source);
    volvoxai_v1_borrowed_buffer_init(&view);
    request.which_source = 3; /* InspectSafetensorsRequest.header_view */
    request.field_header_view = &source;
    source.field_header_prefix = &view;
    source.field_file_size = file_bytes;
    VolvoxaiV1NativeResource resource;
    volvoxai_v1_native_resource_init(&resource);
    resource.field_kind = VOLVOXAI_V1_NATIVE_RESOURCE_KIND_HOST;
    resource.field_handle = (uint64_t)(uintptr_t)prefix;
    resource.field_size_bytes = prefix_bytes;
    view.field_resource = &resource;
    view.field_length_bytes = prefix_bytes;
    if (volvoxai_v1_inspect_safetensors_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK)
        goto done;
    request.field_header_view = NULL;
    payload = vx_call_bytes(&client, VX_RPC_VX_PLANNING_SERVICE_INSPECT_SAFETENSORS,
        encoded, (int32_t)encoded_len, &payload_len);
    if (!payload || volvoxai_v1_safetensors_info_decode(
            output, payload, (size_t)payload_len) != SYNURANG_LITE_OK)
        goto done;
    ok = 1;
done:
    request.field_header_view = NULL;
    source.field_header_prefix = NULL;
    synurang_lite_release(allocator, encoded);
    if (payload) vx_call_free(&client, payload);
    volvoxai_v1_safetensors_header_view_source_free(&source);
    volvoxai_v1_inspect_safetensors_request_free(&request);
    return ok;
}

static uint8_t* create_runtime(
    int32_t cpu_threads,
    VolvoxaiV1ExecutionMode mode,
    int32_t* payload_len) {
    VolvoxaiV1CreateRuntimeRequest request;
    const SynurangLiteAllocator* allocator;
    uint8_t* encoded = NULL;
    uint8_t* payload;
    size_t encoded_len = 0u;

    volvoxai_v1_create_runtime_request_init(&request);
    allocator = request._allocator;
    request.field_cpu_threads = cpu_threads;
    request.has_execution_mode = 1;
    request.field_execution_mode = mode;
    if (volvoxai_v1_create_runtime_request_encode(
            &request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
        volvoxai_v1_create_runtime_request_free(&request);
        return NULL;
    }
    volvoxai_v1_create_runtime_request_free(&request);
    payload = vx_call_bytes(&client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME,
        encoded, (int32_t)encoded_len, payload_len);
    allocator->deallocate(allocator->context, encoded);
    return payload;
}

static void test_plan_identity_ignores_private_headers(void) {
    static const char graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"prefix\",\"tokens\"],\"outputs\":[\"joined\"]}]},\"format\":\"volvox-graph/v1\",\"dimensions\":{"
        "\"B\":{\"min\":1,\"max\":1},"
        "\"Q\":{\"min\":1,\"max\":12},"
        "\"M\":{\"min\":7,\"max\":18}},"
        "\"inputs\":{"
        "\"prefix\":{\"shape\":[\"B\",6,16],\"dtype\":\"float32\"},"
        "\"tokens\":{\"shape\":[\"B\",\"Q\",16],"
        "\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"join\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"prefix\",\"input1\":\"tokens\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"joined\","
        "\"shape\":[\"B\",\"M\",16],\"dtype\":\"float32\"}},"
        "\"params\":{\"axis\":1}}],\"outputs\":[\"joined\"]}";
    VxStandaloneGraphPlanSource source;
    VxGraphPlan* plan = NULL;
    VxGraphPlanView view;
    VxGraphPlanIdentitySnapshot before;
    VxGraphPlanIdentitySnapshot after;
    VxReport report = VX_REPORT_INIT;
    VxGraphPlanRequestV1 saved_request;
    VxGraphPlanResponseV1 saved_response;
    VxGraphBindDefinitionV1 saved_definition;
    VxGraphDomainResponseV1 saved_domain;
    VxGraphPlanRequestV1* request;
    VxGraphPlanResponseV1* response;
    VxGraphBindDefinitionV1* definition;
    VxGraphDomainResponseV1* domain;
    VxStatus status;

    memset(&source, 0, sizeof(source));
    source.graph_document = (const uint8_t*)graph;
    source.graph_document_bytes = sizeof(graph) - 1u;
    status = vx_graph_plan_internal_create_standalone(
        &source, &plan, &report);
    CHECK(status == VX_STATUS_OK && plan,
          "internal identity fixture GraphPlan is created");
    if (status != VX_STATUS_OK || !plan) return;
    status = vx_graph_plan_internal_view(plan, &view);
    CHECK(status == VX_STATUS_OK,
          "internal identity fixture exposes its retained terminals");
    if (status != VX_STATUS_OK) goto done;
    status = vx_graph_plan_test_identity_snapshot(plan, &before);
    CHECK(status == VX_STATUS_OK &&
              strcmp(before.plan_identity, view.plan_identity) == 0 &&
              strcmp(before.shape_domain_proof_identity,
                     view.shape_domain_proof_identity) == 0 &&
              strcmp(before.independent_batch_proof_identity,
                     view.independent_batch_proof_identity) == 0,
          "semantic identity snapshot reproduces stored identities");
    if (status != VX_STATUS_OK) goto done;

    request = (VxGraphPlanRequestV1*)(uintptr_t)view.graph_plan_request_v1;
    response = (VxGraphPlanResponseV1*)(uintptr_t)view.graph_plan_response_v1;
    definition =
        (VxGraphBindDefinitionV1*)(uintptr_t)view.graph_bind_definition_v1;
    domain =
        (VxGraphDomainResponseV1*)(uintptr_t)view.graph_domain_response_v1;
    memcpy(&saved_request, request, sizeof(saved_request));
    memcpy(&saved_response, response, sizeof(saved_response));
    memcpy(&saved_definition, definition, sizeof(saved_definition));
    memcpy(&saved_domain, domain, sizeof(saved_domain));

    request->magic ^= UINT32_C(0xa5a5a5a5);
    request->abi_version += 17u;
    request->total_bytes ^= UINT32_C(0x01010101);
    request->flags ^= UINT32_C(0x80000000);
    request->reserved0 = UINT32_C(0x11223344);
    request->reserved1 = UINT32_C(0x55667788);
    response->magic ^= UINT32_C(0x5a5a5a5a);
    response->abi_version += 19u;
    response->written_bytes ^= UINT32_C(0x01010101);
    response->required_response_bytes ^= UINT32_C(0x02020202);
    response->required_scratch_bytes ^= UINT32_C(0x04040404);
    response->reserved0 = UINT32_C(0x99aabbcc);
    response->reserved1 = UINT32_C(0xddeeff00);
    definition->magic ^= UINT32_C(0x3c3c3c3c);
    definition->abi_version += 23u;
    definition->total_bytes ^= UINT32_C(0x08080808);
    definition->flags ^= UINT32_C(0x40000000);
    definition->reserved0 = UINT32_C(0x13579bdf);
    definition->reserved1 = UINT32_C(0x2468ace0);
    domain->magic ^= UINT32_C(0xc3c3c3c3);
    domain->abi_version += 29u;
    domain->written_bytes ^= UINT32_C(0x10101010);
    domain->required_response_bytes ^= UINT32_C(0x20202020);
    domain->required_scratch_bytes ^= UINT32_C(0x40404040);
    domain->reserved0 = UINT32_C(0x89abcdef);
    domain->reserved1 = UINT32_C(0x76543210);
    domain->reserved2 = UINT32_C(0xfedcba98);

    status = vx_graph_plan_test_identity_snapshot(plan, &after);
    CHECK(status == VX_STATUS_OK &&
              strcmp(after.plan_identity, before.plan_identity) == 0 &&
              strcmp(after.shape_domain_proof_identity,
                     before.shape_domain_proof_identity) == 0 &&
              strcmp(after.independent_batch_proof_identity,
                     before.independent_batch_proof_identity) == 0,
          "plan and proof identities ignore private wire header metadata");

    memcpy(request, &saved_request, sizeof(saved_request));
    memcpy(response, &saved_response, sizeof(saved_response));
    memcpy(definition, &saved_definition, sizeof(saved_definition));
    memcpy(domain, &saved_domain, sizeof(saved_domain));
done:
    vx_graph_plan_internal_release(plan);
}

static void test_model_and_plan_share_proof_identities(
        const char* graph_path,
        const char* weights_path) {
    const char* weight_paths[] = {weights_path};
    VxRuntimeOptions options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxGraphPlan* plan = NULL;
    VxGraphPlanView view;
    VxGraphPlanIdentitySnapshot model_identities;
    VxStatus status;

    options.cpu_threads = 1;
    options.execution_mode = VX_EXECUTION_MODE_DIRECT;
    source.graph_path = graph_path;
    source.weight_paths = weight_paths;
    source.weight_path_count = 1u;
    status = vx_runtime_create(&options, &runtime, &report);
    CHECK(status == VX_STATUS_OK && runtime,
          "identity-authority fixture Runtime is created");
    if (status != VX_STATUS_OK || !runtime) goto done;
    status = vx_runtime_load_model(runtime, &source, &model, &report);
    CHECK(status == VX_STATUS_OK && model,
          "identity-authority fixture Model is loaded");
    if (status != VX_STATUS_OK || !model) goto done;
    status = vx_graph_plan_internal_create_model(model, &plan, &report);
    CHECK(status == VX_STATUS_OK && plan,
          "identity-authority fixture model GraphPlan is created");
    if (status != VX_STATUS_OK || !plan) goto done;
    status = vx_graph_plan_internal_view(plan, &view);
    CHECK(status == VX_STATUS_OK,
          "identity-authority fixture GraphPlan view is available");
    if (status != VX_STATUS_OK) goto done;
    status = vx_model_test_proof_identity_snapshot(
        model, &model_identities);
    CHECK(status == VX_STATUS_OK,
          "execution model proof identities are inspectable in the test");
    if (status != VX_STATUS_OK) goto done;
    CHECK(view.shape_domain_proof_identity &&
              view.shape_domain_proof_identity[0] &&
              strcmp(model_identities.shape_domain_proof_identity,
                     view.shape_domain_proof_identity) == 0,
          "Model and GraphPlan share one canonical shape-proof identity");
    CHECK(view.independent_batch_validated &&
              view.independent_batch_proof_identity &&
              view.independent_batch_proof_identity[0] &&
              strcmp(model_identities.independent_batch_proof_identity,
                     view.independent_batch_proof_identity) == 0,
          "Model and GraphPlan share one canonical batch-proof identity");
done:
    vx_graph_plan_internal_release(plan);
    vx_model_release(model);
    vx_runtime_release(runtime);
}

static void test_standalone_planning_semantics(void) {
    static const char invalid_graph[] = "{}";
    static const char minimum_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]},\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"E\":{\"min\":3,\"max\":8,\"multiple_of\":4}},"
        "\"inputs\":{\"x\":{\"shape\":[\"E\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"copy\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"shape\":[\"E\"],\"dtype\":\"float32\"}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    static const char dropout_ratio_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]},\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{\"x\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"drop\",\"opType\":\"Dropout\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"params\":{\"ratio\":0.25}}],\"outputs\":[\"y\"]}";
    static const char dropout_p_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]}, \"format\": \"volvox-graph/v1\", \"dimensions\": {},"
        "\"inputs\":{\"x\":{\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"nodes\":[{\"id\":\"drop\",\"opType\":\"Dropout\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"params\":{\"p\":0.25}}],\"outputs\":[\"y\"] }";
    static const char affine_concat_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"prefix\",\"tokens\"],\"outputs\":[\"joined\"]}]},\"format\":\"volvox-graph/v1\",\"dimensions\":{"
        "\"B\":{\"min\":1,\"max\":1},"
        "\"Q\":{\"min\":1,\"max\":12},"
        "\"M\":{\"min\":7,\"max\":18}},"
        "\"inputs\":{"
        "\"prefix\":{\"shape\":[\"B\",6,16],\"dtype\":\"float32\"},"
        "\"tokens\":{\"shape\":[\"B\",\"Q\",16],"
        "\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"join\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"prefix\",\"input1\":\"tokens\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"joined\","
        "\"shape\":[\"B\",\"M\",16],\"dtype\":\"float32\"}},"
        "\"params\":{\"axis\":1}}],\"outputs\":[\"joined\"]}";
    static const char bank_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"x\"],\"outputs\":[\"y\"]}]},\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"E\":{\"min\":3,\"max\":10,\"multiple_of\":1}},"
        "\"inputs\":{\"x\":{\"shape\":[1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"experts\",\"opType\":\"MatMul\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"expert.weight\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"shape\":[1,\"E\"],"
        "\"dtype\":\"float32\"}},\"params\":{\"weight_layout\":"
        "\"dout_din\"}}],\"outputs\":[\"y\"],"
        "\"banks\":{\"expert.weight\":\"E\"}}";
    static const char rank_one_bank_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[],\"outputs\":[\"y\"]}]},\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"E\":{\"min\":1,\"max\":10,\"multiple_of\":1}},"
        "\"inputs\":{},\"nodes\":[{\"id\":\"copy\","
        "\"opType\":\"Identity\",\"inputs\":{\"input\":\"bank.weight\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\",\"shape\":[5],"
        "\"dtype\":\"float32\"}},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"banks\":{\"bank.weight\":\"E\"}}";
    static const char metadata_only_weight_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[],\"outputs\":[\"large.weight\"]}]},\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{},\"nodes\":[],\"outputs\":[\"large.weight\"]}";
    static const char wide_uint64_graph[] =
        "{\"execution\":{\"stateless\":{},\"entrypoints\":[{\"name\":\"evaluate\",\"effect\":1,\"allowed_ranges\":[1],\"inputs\":[\"wide\"],\"outputs\":[\"wide\"]}]},\"format\":\"volvox-graph/v1\",\"dimensions\":{},"
        "\"inputs\":{\"wide\":{\"shape\":[1000000000,1000000000],"
        "\"dtype\":\"float32\"}},\"nodes\":[],"
        "\"outputs\":[\"wide\"]}";
    static const int64_t bank_shape[] = {5, 2};
    static const int64_t smaller_bank_shape[] = {4, 2};
    static const int64_t rank_one_bank_shape[] = {5};
    static const int64_t out_of_domain_bank_shape[] = {11, 2};
    static const int64_t largest_i8_shape[] = {INT_MAX};
    static const int64_t oversized_f32_shape[] = {INT64_C(1073741824)};
    static const PlanningWeightFixture bank_weight = {
        "expert.weight", VOLVOXAI_V1_DATA_TYPE_F32, bank_shape, 2u
    };
    static const PlanningWeightFixture rank_one_bank_weight = {
        "bank.weight", VOLVOXAI_V1_DATA_TYPE_F32,
        rank_one_bank_shape, 1u
    };
    static const PlanningWeightFixture smaller_bank_weight = {
        "expert.weight", VOLVOXAI_V1_DATA_TYPE_F32,
        smaller_bank_shape, 2u
    };
    static const PlanningWeightFixture out_of_domain_bank_weight = {
        "expert.weight", VOLVOXAI_V1_DATA_TYPE_F32,
        out_of_domain_bank_shape, 2u
    };
    static const PlanningWeightFixture largest_i8_weight = {
        "large.weight", VOLVOXAI_V1_DATA_TYPE_I8, largest_i8_shape, 1u
    };
    static const PlanningWeightFixture oversized_f32_weight = {
        "large.weight", VOLVOXAI_V1_DATA_TYPE_F32, oversized_f32_shape, 1u
    };
    static const uint32_t partial_slots[] = {0u, 1u, 3u, 4u};
    static const char header_json[] =
        "{\"x\":{\"dtype\":\"F32\",\"shape\":[1],"
        "\"data_offsets\":[0,4]}}";
    static const char invalid_tensor_json[] =
        "{\"x\":{\"dtype\":\"NOT_A_DTYPE\",\"shape\":[1],"
        "\"data_offsets\":[0,4]}}";
    static const size_t header_prefix_limit = 67108864u;
    uint8_t header_prefix[8u + sizeof(header_json) - 1u];
    uint8_t invalid_prefix[sizeof(header_prefix)];
    uint8_t invalid_tensor_prefix[
        8u + sizeof(invalid_tensor_json) - 1u];
    const uint64_t header_bytes = sizeof(header_json) - 1u;
    const uint64_t file_bytes = sizeof(header_prefix) + 4u;
    VolvoxaiV1GraphPlanHandle invalid;
    VolvoxaiV1GraphPlanHandle minimum;
    VolvoxaiV1GraphPlanHandle ratio;
    VolvoxaiV1GraphPlanHandle p;
    VolvoxaiV1GraphPlanHandle affine;
    VolvoxaiV1GraphPlanHandle bank;
    VolvoxaiV1GraphPlanHandle smaller_bank;
    VolvoxaiV1GraphPlanHandle invalid_bank;
    VolvoxaiV1GraphPlanHandle boundary_weight;
    VolvoxaiV1GraphPlanHandle wide_uint64;
    VolvoxaiV1ResolvedGraphPlan resolved;
    VolvoxaiV1SafetensorsInfo safetensors;
    const VolvoxaiV1GraphTensor* input = NULL;
    const VolvoxaiV1GraphTensor* output = NULL;
    const VolvoxaiV1ResolvedGraphTensor* resolved_weight = NULL;

    CHECK(create_standalone_plan(
              invalid_graph, NULL, 0u, &invalid),
          "invalid standalone GraphPlan response decodes");
    CHECK(invalid.field_graph_plan_id == 0 && invalid.field_plan == NULL &&
              invalid.field_report &&
              invalid.field_report->field_status !=
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "invalid CreateGraphPlan is atomic and returns no partial plan");
    volvoxai_v1_graph_plan_handle_free(&invalid);

    CHECK(create_standalone_plan(
              metadata_only_weight_graph, &largest_i8_weight, 1u,
              &boundary_weight),
          "maximum portable I8 descriptor response decodes");
    CHECK(boundary_weight.field_graph_plan_id > 0 &&
              boundary_weight.field_plan && boundary_weight.field_report &&
              boundary_weight.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "metadata-only I8 descriptor at INT_MAX elements is accepted");
    release_graph_plan_if_live(boundary_weight.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&boundary_weight);

    CHECK(create_standalone_plan(
              wide_uint64_graph, NULL, 0u, &wide_uint64),
          "wide uint64 GraphPlan response decodes");
    CHECK(wide_uint64.field_graph_plan_id > 0 && wide_uint64.field_plan &&
              wide_uint64.field_report &&
              wide_uint64.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "checked uint64 products above the JavaScript safe integer create");
    if (wide_uint64.field_graph_plan_id > 0 && resolve_minimum_plan(
            wide_uint64.field_graph_plan_id, NULL, NULL, 0u, &resolved)) {
        CHECK(resolved.field_report && resolved.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK &&
                  resolved.field_tensors.len == 1u &&
                  resolved.field_tensors.data[0].field_element_count ==
                      UINT64_C(1000000000000000000) &&
                  resolved.field_tensors.data[0].field_byte_size ==
                      UINT64_C(4000000000000000000) &&
                  resolved.field_logical_activation_bytes ==
                      UINT64_C(4000000000000000000),
              "resolved tensor preserves exact uint64 element and byte counts");
        volvoxai_v1_resolved_graph_plan_free(&resolved);
    } else {
        CHECK(0, "wide uint64 GraphPlan resolves");
    }
    release_graph_plan_if_live(wide_uint64.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&wide_uint64);

    CHECK(create_standalone_plan(
              metadata_only_weight_graph, &oversized_f32_weight, 1u,
              &boundary_weight),
          "oversized F32 descriptor rejection response decodes");
    CHECK(boundary_weight.field_graph_plan_id == 0 &&
              boundary_weight.field_plan == NULL &&
              boundary_weight.field_report &&
              boundary_weight.field_report->field_status !=
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "packed weight spans above UINT32_MAX are rejected atomically");
    volvoxai_v1_graph_plan_handle_free(&boundary_weight);

    CHECK(create_standalone_plan(
              minimum_graph, NULL, 0u, &minimum),
          "minimum-rounding GraphPlan response decodes");
    CHECK(minimum.field_graph_plan_id > 0 && minimum.field_plan &&
              minimum.field_report && minimum.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "minimum-rounding GraphPlan is created");
    if (minimum.field_graph_plan_id > 0 &&
        resolve_minimum_plan(
            minimum.field_graph_plan_id, NULL, NULL, 0u, &resolved)) {
        CHECK(resolved.field_report && resolved.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
              "minimum GraphPlan binding succeeds");
        for (size_t index = 0u; index < resolved.field_tensors.len; ++index)
            if (bytes_equal_text(
                    &resolved.field_tensors.data[index].field_name, "x"))
                resolved_weight = &resolved.field_tensors.data[index];
        CHECK(resolved_weight && resolved_weight->field_shape.len == 1u &&
                  resolved_weight->field_shape.data[0] == 4,
              "minimum binding rounds 3 up to the smallest legal multiple 4");
        volvoxai_v1_resolved_graph_plan_free(&resolved);
    } else {
        CHECK(0, "minimum GraphPlan binding response decodes");
    }
    release_graph_plan_if_live(minimum.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&minimum);

    CHECK(create_standalone_plan(
              dropout_ratio_graph, NULL, 0u, &ratio),
          "Dropout ratio GraphPlan response decodes");
    CHECK(create_standalone_plan(
              dropout_p_graph, NULL, 0u, &p),
          "Dropout p GraphPlan response decodes");
    CHECK(ratio.field_plan && p.field_plan &&
              bytes_equal(&ratio.field_plan->field_plan_identity,
                          &p.field_plan->field_plan_identity),
          "registry-equivalent aliases share one semantic plan identity");
    CHECK(ratio.field_plan && p.field_plan &&
              !bytes_equal(&ratio.field_plan->field_graph_fingerprint,
                           &p.field_plan->field_graph_fingerprint),
          "different exact graph documents retain different fingerprints");
    if (ratio.field_plan) {
        for (size_t index = 0u; index < ratio.field_plan->field_tensors.len;
             ++index) {
            const VolvoxaiV1GraphTensor* tensor =
                &ratio.field_plan->field_tensors.data[index];
            if (bytes_equal_text(&tensor->field_name, "x")) input = tensor;
            if (bytes_equal_text(&tensor->field_name, "y")) output = tensor;
        }
        CHECK(input && output && output->field_storage_alias_root &&
                  bytes_equal_text(
                      &output->field_storage_alias_root->field_name, "x"),
              "Dropout output exposes its canonical logical alias root");
        CHECK(input && !input->field_public_output && output &&
                  output->field_public_output &&
                  output->field_public_output->field_output_index == 0u,
              "message presence preserves public output index zero");
        CHECK(input && output && input->field_canonical_last_use_step == 0 &&
                  output->field_canonical_last_use_step == 1,
              "logical alias does not widen either tensor's canonical liveness");
        CHECK(ratio.field_plan->field_nodes.len == 1u &&
                  ratio.field_plan->field_nodes.data[0].field_parameters.len ==
                      1u &&
                  bytes_equal_text(
                      &ratio.field_plan->field_nodes.data[0]
                           .field_parameters.data[0].field_canonical_name,
                      "dropout-probability"),
              "alias parameter projects its stable canonical group id");
    }
    release_graph_plan_if_live(ratio.field_graph_plan_id);
    release_graph_plan_if_live(p.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&ratio);
    volvoxai_v1_graph_plan_handle_free(&p);

    CHECK(create_standalone_plan(
              affine_concat_graph, NULL, 0u, &affine),
          "affine Concat GraphPlan response decodes");
    {
        const VolvoxaiV1ShapeDimensionRelation* affine_relation = NULL;
        if (affine.field_plan && affine.field_plan->field_shape_domain &&
            affine.field_plan->field_shape_domain->which_outcome == 1 &&
            affine.field_plan->field_shape_domain->field_supported) {
            const VolvoxaiV1ShapeDomainProof* proof =
                affine.field_plan->field_shape_domain->field_supported;
            for (size_t index = 0u; index < proof->field_relations.len;
                 ++index) {
                const VolvoxaiV1ShapeDimensionRelation* candidate =
                    &proof->field_relations.data[index];
                if (bytes_equal_text(&candidate->field_target, "M"))
                    affine_relation = candidate;
            }
        }
        CHECK(affine.field_graph_plan_id > 0 && affine.field_plan &&
                  affine.field_plan->field_plan_identity.len > 0u &&
                  affine_relation && affine_relation->which_relation == 4 &&
                  affine_relation->field_affine &&
                  bytes_equal_text(
                      &affine_relation->field_affine->field_source, "Q") &&
                  affine_relation->field_affine->field_offset == 6,
              "affine-only target is projected and participates in identity");
    }
    release_graph_plan_if_live(affine.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&affine);

    CHECK(create_standalone_plan(
              bank_graph, &bank_weight, 1u, &bank),
          "weight-bank GraphPlan response decodes");
        CHECK(bank.field_graph_plan_id > 0 && bank.field_plan &&
              bank.field_plan->field_weight_banks.len == 1u,
          "GraphPlan exposes the canonical bank declaration");
    if (bank.field_plan && bank.field_plan->field_dimensions.len == 1u &&
        bank.field_plan->field_tensors.len > 0u) {
        const VolvoxaiV1GraphTensor* declared_bank = NULL;
        for (size_t index = 0u;
             index < bank.field_plan->field_tensors.len; ++index)
            if (bytes_equal_text(
                    &bank.field_plan->field_tensors.data[index].field_name,
                    "expert.weight"))
                declared_bank = &bank.field_plan->field_tensors.data[index];
        CHECK(declared_bank && declared_bank->field_shape.len == 2u &&
                  declared_bank->field_shape.data[0].which_extent == 1 &&
                  declared_bank->field_shape.data[0].field_fixed_extent == 5 &&
                  bank.field_plan->field_dimensions.data[0].field_maximum == 10,
              "bank current slot count may be below append capacity");
    }
    if (bank.field_graph_plan_id > 0 &&
        resolve_minimum_plan(
            bank.field_graph_plan_id, "expert.weight", partial_slots,
            sizeof(partial_slots) / sizeof(partial_slots[0]), &resolved)) {
        CHECK(resolved.field_report && resolved.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
              "partial bank residency resolves");
        CHECK(resolved.field_weight_banks.len == 1u &&
                  resolved.field_weight_banks.data[0].which_residency == 4 &&
                  resolved.field_weight_banks.data[0].field_partial &&
                  resolved.field_weight_banks.data[0]
                          .field_partial->field_slots.len == 4u,
              "resolved bank uses the typed partial residency arm");
        resolved_weight = NULL;
        for (size_t index = 0u; index < resolved.field_tensors.len; ++index)
            if (bytes_equal_text(
                    &resolved.field_tensors.data[index].field_name,
                    "expert.weight"))
                resolved_weight = &resolved.field_tensors.data[index];
        CHECK(resolved_weight && resolved_weight->field_shape.len == 2u &&
                  resolved_weight->field_shape.data[0] == 4 &&
                  resolved_weight->field_byte_size == 32u &&
                  resolved.field_weight_bytes == 32u,
              "partial bank compacts axis zero and exact weight accounting");
        volvoxai_v1_resolved_graph_plan_free(&resolved);
    } else {
        CHECK(0, "partial bank resolve response decodes");
    }
    if (bank.field_graph_plan_id > 0 && resolve_minimum_plan(
            bank.field_graph_plan_id, NULL, NULL, 0u, &resolved)) {
        CHECK(resolved.field_report && resolved.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
              "fully resident bank resolves");
        CHECK(resolved.field_weight_banks.len == 1u &&
                  resolved.field_weight_banks.data[0].which_residency == 3 &&
                  resolved.field_weight_banks.data[0].field_fully_resident &&
                  resolved.field_weight_bytes == 40u,
              "omitted override projects the typed fully-resident arm");
        volvoxai_v1_resolved_graph_plan_free(&resolved);
    } else {
        CHECK(0, "fully resident bank resolve response decodes");
    }
    CHECK(create_standalone_plan(
              bank_graph, &smaller_bank_weight, 1u, &smaller_bank),
          "alternate current bank extent response decodes");
    CHECK(bank.field_plan && smaller_bank.field_plan &&
              !bytes_equal(&bank.field_plan->field_plan_identity,
                           &smaller_bank.field_plan->field_plan_identity),
          "current bank slot count participates in plan identity");
    release_graph_plan_if_live(smaller_bank.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&smaller_bank);
    release_graph_plan_if_live(bank.field_graph_plan_id);
    volvoxai_v1_graph_plan_handle_free(&bank);

    CHECK(create_standalone_plan(
              rank_one_bank_graph, &rank_one_bank_weight, 1u,
              &invalid_bank),
          "rank-one bank rejection response decodes");
    CHECK(invalid_bank.field_graph_plan_id == 0 &&
              invalid_bank.field_plan == NULL && invalid_bank.field_report &&
              invalid_bank.field_report->field_status !=
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "standalone bank tensors must have rank at least two");
    volvoxai_v1_graph_plan_handle_free(&invalid_bank);

    CHECK(create_standalone_plan(
              bank_graph, &out_of_domain_bank_weight, 1u,
              &invalid_bank),
          "out-of-domain bank rejection response decodes");
    CHECK(invalid_bank.field_graph_plan_id == 0 &&
              invalid_bank.field_plan == NULL && invalid_bank.field_report &&
              invalid_bank.field_report->field_status !=
                  VOLVOXAI_V1_NATIVE_STATUS_OK,
          "standalone bank slot count must be legal in its dimension");
    volvoxai_v1_graph_plan_handle_free(&invalid_bank);

    for (size_t index = 0u; index < 8u; ++index)
        header_prefix[index] = (uint8_t)(header_bytes >> (8u * index));
    memcpy(header_prefix + 8u, header_json, (size_t)header_bytes);
    CHECK(inspect_safetensors_prefix(
              header_prefix, sizeof(header_prefix), file_bytes,
              &safetensors),
          "valid SafeTensors inspection response decodes");
    CHECK(safetensors.field_report && safetensors.field_report->field_status ==
              VOLVOXAI_V1_NATIVE_STATUS_OK &&
              safetensors.field_diagnostic == NULL &&
              safetensors.field_header_json_bytes == header_bytes &&
              safetensors.field_data_region_offset == 8u + header_bytes &&
              safetensors.field_data_region_bytes == 4u &&
              safetensors.field_file_bytes == file_bytes &&
              safetensors.field_tensors.len == 1u &&
              safetensors.field_tensors.data[0].field_file_offset ==
                  8u + header_bytes &&
              safetensors.field_tensors.data[0].field_byte_size == 4u,
          "SafeTensors offsets and size equations are projected exactly");
    volvoxai_v1_safetensors_info_free(&safetensors);

    memcpy(invalid_prefix, header_prefix, sizeof(header_prefix));
    invalid_prefix[0]++;
    CHECK(inspect_safetensors_prefix(
              invalid_prefix, sizeof(invalid_prefix), file_bytes,
              &safetensors),
          "invalid SafeTensors inspection response decodes");
    CHECK(safetensors.field_report && safetensors.field_report->field_status !=
              VOLVOXAI_V1_NATIVE_STATUS_OK && safetensors.field_diagnostic &&
              safetensors.field_diagnostic->field_code ==
                  VOLVOXAI_V1_SAFETENSORS_DIAGNOSTIC_CODE_LENGTH_MISMATCH &&
              safetensors.field_diagnostic->field_entry == NULL,
          "file-wide parser refusal has a typed diagnostic without an entry");
    volvoxai_v1_safetensors_info_free(&safetensors);

    {
        const uint64_t invalid_tensor_bytes =
            sizeof(invalid_tensor_json) - 1u;
        for (size_t index = 0u; index < 8u; ++index)
            invalid_tensor_prefix[index] =
                (uint8_t)(invalid_tensor_bytes >> (8u * index));
        memcpy(invalid_tensor_prefix + 8u, invalid_tensor_json,
               (size_t)invalid_tensor_bytes);
        CHECK(inspect_safetensors_prefix(
                  invalid_tensor_prefix, sizeof(invalid_tensor_prefix),
                  sizeof(invalid_tensor_prefix) + 4u, &safetensors),
              "invalid tensor SafeTensors response decodes");
        CHECK(safetensors.field_report &&
                  safetensors.field_report->field_status !=
                      VOLVOXAI_V1_NATIVE_STATUS_OK &&
                  safetensors.field_diagnostic &&
                  safetensors.field_diagnostic->field_code ==
                      VOLVOXAI_V1_SAFETENSORS_DIAGNOSTIC_CODE_UNKNOWN_DTYPE &&
                  safetensors.field_diagnostic->field_entry &&
                  safetensors.field_diagnostic->field_entry->field_index == 0u,
              "tensor refusal preserves entry-zero presence");
        volvoxai_v1_safetensors_info_free(&safetensors);
    }

    {
        uint8_t* limit_prefix = (uint8_t*)malloc(header_prefix_limit);
        CHECK(limit_prefix != NULL,
              "SafeTensors header-prefix boundary fixture allocates");
        if (limit_prefix) {
            uint64_t json_length = (uint64_t)header_prefix_limit - 8u;
            memset(limit_prefix, ' ', header_prefix_limit);
            for (size_t index = 0u; index < 8u; ++index)
                limit_prefix[index] =
                    (uint8_t)(json_length >> (8u * index));
            limit_prefix[8] = '{';
            limit_prefix[9] = '}';
            CHECK(inspect_safetensors_view(
                      limit_prefix, header_prefix_limit,
                      header_prefix_limit, &safetensors),
                  "maximum SafeTensors header-prefix response decodes");
            CHECK(safetensors.field_report &&
                      safetensors.field_report->field_status ==
                          VOLVOXAI_V1_NATIVE_STATUS_OK &&
                      safetensors.field_header_json_bytes == json_length,
                  "SafeTensors common header-prefix maximum is inclusive");
            volvoxai_v1_safetensors_info_free(&safetensors);

            CHECK(inspect_safetensors_view(
                      limit_prefix, (uint64_t)header_prefix_limit + 1u,
                      (uint64_t)header_prefix_limit + 1u, &safetensors),
                  "oversized SafeTensors header-prefix response decodes");
            CHECK(safetensors.field_report &&
                      safetensors.field_report->field_status ==
                          VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT &&
                      safetensors.field_diagnostic == NULL,
                  "SafeTensors BufferView rejects a prefix over the maximum");
            volvoxai_v1_safetensors_info_free(&safetensors);
            free(limit_prefix);
        }
    }
}

static void test_text_service(void) {
    VolvoxaiV1CreateTokenizerRequest create;
    VolvoxaiV1TokenizerHandle handle;
    VolvoxaiV1EncodeTextRequest encode;
    VolvoxaiV1EncodedText encoded;
    VolvoxaiV1DecodeTokensRequest decode;
    VolvoxaiV1DecodedText decoded;
    uint8_t* request_bytes = NULL;
    size_t request_len = 0;
    uint8_t* response_bytes = NULL;
    int32_t response_len = 0;
    int64_t id = 0;
    const char vocabulary[] = "{\"a\":1,\"b\":2,\"ab\":3,\"\\u0000\":9}";
    const uint8_t text[] = {'a', 'b', 0, 'b'};
    volvoxai_v1_create_tokenizer_request_init(&create);
    volvoxai_v1_tokenizer_handle_init(&handle);
    volvoxai_v1_encode_text_request_init(&encode);
    volvoxai_v1_encoded_text_init(&encoded);
    volvoxai_v1_decode_tokens_request_init(&decode);
    volvoxai_v1_decoded_text_init(&decoded);
    create.which_vocabulary = 2;
    CHECK(synurang_lite_bytes_assign(create._allocator, &create.field_vocabulary_json,
              vocabulary, sizeof(vocabulary) - 1) == SYNURANG_LITE_OK, "tokenizer JSON allocation");
    CHECK(volvoxai_v1_create_tokenizer_request_encode(&create, &request_bytes, &request_len) ==
              SYNURANG_LITE_OK, "tokenizer create encode");
    response_bytes = vx_call_bytes(&client, VX_RPC_VX_TEXT_SERVICE_CREATE_TOKENIZER, request_bytes, (int32_t)request_len, &response_len);
    free(request_bytes); request_bytes = NULL;
    CHECK(response_bytes && volvoxai_v1_tokenizer_handle_decode(&handle, response_bytes, (size_t)response_len) ==
              SYNURANG_LITE_OK, "native text dispatch creates tokenizer");
    vx_call_free(&client, response_bytes); response_bytes = NULL;
    if (!handle.field_report || handle.field_report->field_status || !handle.field_tokenizer_id) {
        CHECK(0, "native tokenizer returns a successful handle"); goto done;
    }
    id = handle.field_tokenizer_id;
    memset(create.field_vocabulary_json.data, 0, create.field_vocabulary_json.len);
    encode.field_tokenizer_id = id;
    CHECK(synurang_lite_bytes_assign(encode._allocator, &encode.field_text, text, sizeof(text)) ==
              SYNURANG_LITE_OK, "native tokenizer text includes NUL");
    CHECK(volvoxai_v1_encode_text_request_encode(&encode, &request_bytes, &request_len) ==
              SYNURANG_LITE_OK, "tokenizer encode request");
    response_bytes = vx_call_bytes(&client, VX_RPC_VX_TEXT_SERVICE_ENCODE_TEXT, request_bytes, (int32_t)request_len, &response_len);
    free(request_bytes); request_bytes = NULL;
    CHECK(response_bytes && volvoxai_v1_encoded_text_decode(&encoded, response_bytes, (size_t)response_len) ==
              SYNURANG_LITE_OK, "native tokenizer encode response");
    vx_call_free(&client, response_bytes); response_bytes = NULL;
    CHECK(encoded.field_report && !encoded.field_report->field_status && encoded.field_tokens.len == 3 &&
              encoded.field_tokens.data[0] == 3 && encoded.field_tokens.data[1] == 9 &&
              encoded.field_tokens.data[2] == 2, "greedy UTF-8 uses retained vocabulary and NUL key");
    decode.field_tokenizer_id = id;
    if (encoded.field_tokens.len) {
        decode.field_tokens.data = decode._allocator->allocate(decode._allocator->context,
            encoded.field_tokens.len * sizeof(uint32_t));
        if (!decode.field_tokens.data) { CHECK(0, "decode allocation"); goto done; }
        decode.field_tokens.len = decode.field_tokens.cap = encoded.field_tokens.len;
        memcpy(decode.field_tokens.data, encoded.field_tokens.data, encoded.field_tokens.len * sizeof(uint32_t));
    }
    CHECK(volvoxai_v1_decode_tokens_request_encode(&decode, &request_bytes, &request_len) ==
              SYNURANG_LITE_OK, "tokenizer decode request");
    response_bytes = vx_call_bytes(&client, VX_RPC_VX_TEXT_SERVICE_DECODE_TOKENS, request_bytes, (int32_t)request_len, &response_len);
    free(request_bytes); request_bytes = NULL;
    CHECK(response_bytes && volvoxai_v1_decoded_text_decode(&decoded, response_bytes, (size_t)response_len) ==
              SYNURANG_LITE_OK, "native tokenizer decode response");
    vx_call_free(&client, response_bytes); response_bytes = NULL;
    CHECK(decoded.field_report && !decoded.field_report->field_status && decoded.field_text.len == sizeof(text) &&
              !memcmp(decoded.field_text.data, text, sizeof(text)), "native decode preserves embedded NUL");
    {
        VolvoxaiV1TokenizerRef call_request;
        volvoxai_v1_tokenizer_ref_init(&call_request);
        call_request.field_tokenizer_id = id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_TEXT_SERVICE_RELEASE_TOKENIZER,
            volvoxai_v1_tokenizer_ref, &call_request, response_bytes, response_len);
    }
    CHECK(response_bytes != NULL, "native tokenizer release");
    vx_call_free(&client, response_bytes); response_bytes = NULL;
    volvoxai_v1_encoded_text_free(&encoded); volvoxai_v1_encoded_text_init(&encoded);
    CHECK(volvoxai_v1_encode_text_request_encode(&encode, &request_bytes, &request_len) ==
              SYNURANG_LITE_OK, "retired tokenizer request");
    response_bytes = vx_call_bytes(&client, VX_RPC_VX_TEXT_SERVICE_ENCODE_TEXT, request_bytes, (int32_t)request_len, &response_len);
    CHECK(response_bytes && volvoxai_v1_encoded_text_decode(&encoded, response_bytes, (size_t)response_len) ==
              SYNURANG_LITE_OK, "retired tokenizer reports");
    CHECK(encoded.field_report && encoded.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED,
          "native tokenizer handle fails closed after release");
done:
    free(request_bytes); vx_call_free(&client, response_bytes);
    if (id) {
        VolvoxaiV1TokenizerRef request;
        volvoxai_v1_tokenizer_ref_init(&request);
        request.field_tokenizer_id = id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_TEXT_SERVICE_RELEASE_TOKENIZER,
            volvoxai_v1_tokenizer_ref, &request, response_bytes, response_len);
        vx_call_free(&client, response_bytes);
    }
    volvoxai_v1_create_tokenizer_request_free(&create);
    volvoxai_v1_tokenizer_handle_free(&handle);
    volvoxai_v1_encode_text_request_free(&encode);
    volvoxai_v1_encoded_text_free(&encoded);
    volvoxai_v1_decode_tokens_request_free(&decode);
    volvoxai_v1_decoded_text_free(&decoded);
}

static int assign_file_bytes(const SynurangLiteAllocator* allocator,
                             SynurangLiteBytes* target, const char* path) {
    FILE* file = fopen(path, "rb");
    long size;
    unsigned char* data;
    int ok;
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) || (size = ftell(file)) <= 0 || fseek(file, 0, SEEK_SET)) {
        fclose(file);
        return 0;
    }
    data = (unsigned char*)malloc((size_t)size);
    if (!data) { fclose(file); return 0; }
    ok = fread(data, 1, (size_t)size, file) == (size_t)size &&
         synurang_lite_bytes_assign(allocator, target, data, (size_t)size) == SYNURANG_LITE_OK;
    free(data);
    fclose(file);
    return ok;
}

static int64_t test_inline_load(int64_t runtime_id, const char* graph,
                                const char* weights, int path_weights) {
    VolvoxaiV1LoadModelRequest request;
    VolvoxaiV1ModelHandle handle;
    VolvoxaiV1ModelPackage* package;
    SynurangLiteBytes* shard;
    uint8_t* encoded = NULL;
    uint8_t* payload = NULL;
    size_t encoded_len = 0;
    int32_t payload_len = 0;
    int64_t model_id = 0;
    volvoxai_v1_load_model_request_init(&request);
    request.field_runtime_id = runtime_id;
    if (path_weights) {
        shard = volvoxai_v1_load_model_request_add_weight_paths(&request);
        if (!shard || !assign_file_bytes(request._allocator, &request.field_graph_document, graph) ||
            synurang_lite_bytes_assign(request._allocator, shard, weights, strlen(weights)) != SYNURANG_LITE_OK) {
            CHECK(0, "inline graph and original weight path assigned");
            goto done;
        }
    } else {
        package = request._allocator->allocate(request._allocator->context, sizeof(*package));
        CHECK(package != NULL, "ModelPackage storage allocated");
        if (!package) goto done;
        volvoxai_v1_model_package_init_with_allocator(package, request._allocator);
        request.field_package = package;
        shard = volvoxai_v1_model_package_add_weight_shards(package);
        if (!shard || !assign_file_bytes(request._allocator, &package->field_graph_document, graph) ||
            !assign_file_bytes(request._allocator, shard, weights)) {
            CHECK(0, "ModelPackage files read into inline bytes");
            goto done;
        }
    }
    for (int conflict = 1; conflict >= 0; conflict--) {
        CHECK(synurang_lite_bytes_assign(request._allocator, &request.field_graph_path,
            conflict ? "graph.json" : "", conflict ? 10 : 0) == SYNURANG_LITE_OK,
            "inline source conflict path assigned");
        if (volvoxai_v1_load_model_request_encode(&request, &encoded, &encoded_len) != SYNURANG_LITE_OK) {
            CHECK(0, "inline model request encodes");
            goto done;
        }
        payload = vx_call_bytes(&client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL, encoded, (int32_t)encoded_len, &payload_len);
        request._allocator->deallocate(request._allocator->context, encoded);
        encoded = NULL;
        if (!payload) { CHECK(0, "inline model dispatch returns payload"); goto done; }
        CHECK(DECODE(payload, payload_len, handle, volvoxai_v1_model_handle), "inline model response decodes");
        CHECK(handle.field_report && handle.field_report->field_status ==
              (conflict ? VOLVOXAI_V1_NATIVE_STATUS_INVALID_ARGUMENT : VOLVOXAI_V1_NATIVE_STATUS_OK),
              "inline model accepts its weights and rejects mixed graph sources");
        if (conflict) CHECK(handle.field_model_id == 0, "mixed sources publish no model");
        else model_id = handle.field_model_id;
        volvoxai_v1_model_handle_free(&handle);
        vx_call_free(&client, payload);
        payload = NULL;
    }
done:
    if (encoded) request._allocator->deallocate(request._allocator->context, encoded);
    if (payload) vx_call_free(&client, payload);
    /* The rest of the roundtrip compiles/executes after these bytes are gone. */
    volvoxai_v1_load_model_request_free(&request);
    return model_id;
}

static void test_api_description(void) {
    VolvoxaiV1ApiDescription description;
    uint8_t* payload;
    int32_t payload_len = 0;
    {
        VolvoxaiV1DescribeApiRequest call_request;
        volvoxai_v1_describe_api_request_init(&call_request);
        call_request.field_include_types = 1;
        call_request.field_service.data = (uint8_t*)((const uint8_t*)"VxInferenceService");
        call_request.field_service.len = (size_t)((int32_t)strlen("VxInferenceService"));
        call_request.field_method.data = (uint8_t*)((const uint8_t*)"LoadModel");
        call_request.field_method.len = (size_t)((int32_t)strlen("LoadModel"));
        VX_CALL_MESSAGE(&client, VX_RPC_VX_PLATFORM_SERVICE_DESCRIBE_API,
            volvoxai_v1_describe_api_request, &call_request, payload, payload_len);
    }
    CHECK(payload != NULL, "DescribeApi returns a payload");
    if (!payload) return;
    CHECK(DECODE(payload, payload_len, description, volvoxai_v1_api_description), "ApiDescription decodes");
    CHECK(description.field_report && description.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK &&
          description.field_methods.len == 1 && description.field_messages.len > 0 &&
          description.field_schema_sha256.len == 64, "DescribeApi returns filtered methods and dependent types");
    int found = 0;
    for (size_t i = 0; i < description.field_messages.len; i++) {
        const VolvoxaiV1ApiMessage* message = &description.field_messages.data[i];
        if (bytes_equal_text(&message->field_name, "volvoxai.v1.LoadModelRequest")) {
            found = 1;
            CHECK(message->field_rules.len == 1, "LoadModel graph-source rule is discoverable");
            if (message->field_rules.len == 1) {
                const VolvoxaiV1ApiRule* rule = &message->field_rules.data[0];
                CHECK(rule->field_kind == VOLVOXAI_V1_API_RULE_KIND_EXACTLY_ONE &&
                      rule->field_fields.len == 3, "LoadModel selects exactly one of three graph sources");
                if (rule->field_fields.len == 3) {
                    CHECK(bytes_equal_text(&rule->field_fields.data[0], "graph_path") &&
                          bytes_equal_text(&rule->field_fields.data[1], "graph_document") &&
                          bytes_equal_text(&rule->field_fields.data[2], "package"),
                          "LoadModel metadata exposes inline graph alongside path and package");
                }
            }
        }
    }
    CHECK(found, "LoadModelRequest metadata present");
    volvoxai_v1_api_description_free(&description);
    vx_call_free(&client, payload);
}

int main(int argc, char** argv) {
    if (!vx_call_client_open(&client)) return 1;
    atexit(close_client);
    const char* model_dir = argc > 1 ? argv[1] : "models/tinystories_1m";
    char graph_path[1024];
    char weights_path[1024];
    const SynurangLiteAllocator* allocator = synurang_lite_default_allocator();
    uint8_t* payload;
    int32_t payload_len;
    int64_t runtime_id = 0;
    int64_t model_id = 0;
    int64_t graph_plan_id = 0;
    int64_t compiled_id = 0;
    int64_t context_id = 0;

    snprintf(graph_path, sizeof(graph_path), "%s/graph.json", model_dir);
    snprintf(weights_path, sizeof(weights_path), "%s/model.safetensors", model_dir);

    test_api_description();
    test_text_service();
    test_plan_identity_ignores_private_headers();
    test_standalone_planning_semantics();
    test_model_and_plan_share_proof_identities(graph_path, weights_path);

    /* --- platform ------------------------------------------------------- */
    {
        VolvoxaiV1PlatformInfo info;
        {
            VolvoxaiV1Empty call_request;
            volvoxai_v1_empty_init(&call_request);
            VX_CALL_MESSAGE(&client, VX_RPC_VX_PLATFORM_SERVICE_GET_PLATFORM_INFO,
                volvoxai_v1_empty, &call_request, payload, payload_len);
        }
        CHECK(payload != NULL, "GetPlatformInfo returns a payload");
        if (payload) {
            CHECK(DECODE(payload, payload_len, info, volvoxai_v1_platform_info),
                  "PlatformInfo decodes");
            CHECK(info.field_api_version > 0u, "api_version reported");
            CHECK(info.field_max_tensor_rank > 0u, "max_tensor_rank reported");
            CHECK(info.field_compiled_backends.len > 0u, "compiled backends reported");
            CHECK(info.field_transport ==
                      VOLVOXAI_V1_TRANSPORT_PROFILE_IN_PROCESS,
                  "in-process transport advertised");
            printf("platform: api=%u backends=%zu build_profile=%d\n",
                   info.field_api_version, info.field_compiled_backends.len,
                   (int)info.field_build_profile);
            volvoxai_v1_platform_info_free(&info);
            vx_call_free(&client, payload);
        }
    }

    /* --- runtime -------------------------------------------------------- */
    {
        VolvoxaiV1RuntimeHandle handle;
        payload = create_runtime(1, VOLVOXAI_V1_EXECUTION_MODE_SCHEDULED, &payload_len);
        CHECK(payload != NULL, "CreateRuntime returns a payload");
        if (!payload) {
            fprintf(stderr, "  %s\n", last_error());
            return 1;
        }
        CHECK(DECODE(payload, payload_len, handle, volvoxai_v1_runtime_handle),
              "RuntimeHandle decodes");
        report_summary("CreateRuntime", handle.field_report);
        runtime_id = handle.field_runtime_id;
        CHECK(runtime_id > 0, "runtime id issued");
        volvoxai_v1_runtime_handle_free(&handle);
        vx_call_free(&client, payload);
    }

    /* PlatformInfo describes link-time capabilities. ListBackends is the
     * narrower per-Runtime view and must contain only initialized providers. */
    {
        VolvoxaiV1BackendList backends;
        {
            VolvoxaiV1RuntimeRef call_request;
            volvoxai_v1_runtime_ref_init(&call_request);
            call_request.field_runtime_id = runtime_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_LIST_BACKENDS,
                volvoxai_v1_runtime_ref, &call_request, payload, payload_len);
        }
        CHECK(payload != NULL, "ListBackends returns a payload");
        if (payload) {
            CHECK(DECODE(payload, payload_len, backends,
                         volvoxai_v1_backend_list),
                  "BackendList decodes");
            CHECK(backends.field_report &&
                      backends.field_report->field_status ==
                          VOLVOXAI_V1_NATIVE_STATUS_OK,
                  "ListBackends reports OK");
            CHECK(backends.field_backends.len == 1u,
                  "fresh runtime lists one initialized provider");
            if (backends.field_backends.len == 1u) {
                const SynurangLiteBytes* name = &backends.field_backends.data[0];
                CHECK(name->len == 3u &&
                          memcmp(name->data, "cpu", 3u) == 0,
                      "CPU is the initialized default provider");
            }
            volvoxai_v1_backend_list_free(&backends);
            vx_call_free(&client, payload);
        }
    }

    /* --- model ---------------------------------------------------------- */
    {
        VolvoxaiV1LoadModelRequest request;
        VolvoxaiV1ModelHandle handle;
        SynurangLiteBytes* weights;
        uint8_t* encoded = NULL;
        size_t encoded_len = 0u;

        volvoxai_v1_load_model_request_init(&request);
        request.field_runtime_id = runtime_id;
        CHECK(synurang_lite_bytes_assign(request._allocator, &request.field_graph_path,
                                         graph_path, strlen(graph_path)) ==
                  SYNURANG_LITE_OK,
              "graph path assigned");
        weights = (SynurangLiteBytes*)request._allocator->allocate(
            request._allocator->context, sizeof(*weights));
        CHECK(weights != NULL, "weight path storage allocated");
        memset(weights, 0, sizeof(*weights));
        CHECK(synurang_lite_bytes_assign(request._allocator, weights, weights_path,
                                         strlen(weights_path)) == SYNURANG_LITE_OK,
              "weight path assigned");
        request.field_weight_paths.data = weights;
        request.field_weight_paths.len = 1u;
        request.field_weight_paths.cap = 1u;

        CHECK(volvoxai_v1_load_model_request_encode(&request, &encoded, &encoded_len) ==
                  SYNURANG_LITE_OK,
              "LoadModelRequest encodes");
        payload = vx_call_bytes(&client, VX_RPC_VX_INFERENCE_SERVICE_LOAD_MODEL, encoded, (int32_t)encoded_len, &payload_len);
        allocator->deallocate(allocator->context, encoded);
        volvoxai_v1_load_model_request_free(&request);

        CHECK(payload != NULL, "LoadModel returns a payload");
        if (!payload) {
            fprintf(stderr, "  %s\n", last_error());
            return 1;
        }
        CHECK(DECODE(payload, payload_len, handle, volvoxai_v1_model_handle),
              "ModelHandle decodes");
        report_summary("LoadModel", handle.field_report);
        model_id = handle.field_model_id;
        CHECK(model_id > 0, "model id issued");
        volvoxai_v1_model_handle_free(&handle);
        vx_call_free(&client, payload);
        if (model_id <= 0) return 1;
    }

    /* Inspect before compilation, then cover both inline source forms. The
     * final model uses an inline graph and original path-backed weights. */
    {
        VolvoxaiV1ModelInfo info;
        {
            VolvoxaiV1ModelRef call_request;
            volvoxai_v1_model_ref_init(&call_request);
            call_request.field_model_id = model_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_GET_MODEL_INFO,
                volvoxai_v1_model_ref, &call_request, payload, payload_len);
        }
        CHECK(payload != NULL, "GetModelInfo returns a payload");
        if (payload) {
            CHECK(DECODE(payload, payload_len, info, volvoxai_v1_model_info), "ModelInfo decodes");
            CHECK(info.field_model_id == model_id && info.field_inputs.len > 0 && info.field_outputs.len > 0 &&
                  info.field_report && info.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK,
                  "ModelInfo exposes the logical I/O before compile");
            volvoxai_v1_model_info_free(&info);
            vx_call_free(&client, payload);
        }
        for (int path_weights = 0; path_weights <= 1; path_weights++) {
            int64_t bytes_model_id = test_inline_load(runtime_id, graph_path, weights_path, path_weights);
            CHECK(bytes_model_id > 0, "native inline graph publishes a model with inline or path weights");
            if (bytes_model_id <= 0) continue;
            {
                VolvoxaiV1ModelRef call_request;
                volvoxai_v1_model_ref_init(&call_request);
                call_request.field_model_id = model_id;
                VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_MODEL,
                    volvoxai_v1_model_ref, &call_request, payload, payload_len);
            }
            if (payload) vx_call_free(&client, payload);
            model_id = bytes_model_id;
        }
    }

    /* --- backend-neutral planning -------------------------------------- */
    {
        VolvoxaiV1CreateGraphPlanRequest request;
        VolvoxaiV1GraphPlanHandle handle;
        VolvoxaiV1GraphPlanInfo info;
        VolvoxaiV1ResolveGraphPlanRequest resolve;
        VolvoxaiV1ResolvedGraphPlan resolved;
        VolvoxaiV1OperationReport released;
        VolvoxaiV1Empty minimum;
        uint8_t* encoded = NULL;
        size_t encoded_len = 0u;

        volvoxai_v1_create_graph_plan_request_init(&request);
        request.which_source = 1; /* CreateGraphPlanRequest.model_id */
        request.field_model_id = model_id;
        CHECK(volvoxai_v1_create_graph_plan_request_encode(
                  &request, &encoded, &encoded_len) == SYNURANG_LITE_OK,
              "model GraphPlan request encodes");
        payload = vx_call_bytes(&client, VX_RPC_VX_PLANNING_SERVICE_CREATE_GRAPH_PLAN,
            encoded, (int32_t)encoded_len, &payload_len);
        allocator->deallocate(allocator->context, encoded);
        volvoxai_v1_create_graph_plan_request_free(&request);
        CHECK(payload != NULL, "CreateGraphPlan returns a payload");
        if (payload) {
            CHECK(DECODE(payload, payload_len, handle,
                         volvoxai_v1_graph_plan_handle),
                  "GraphPlanHandle decodes");
            graph_plan_id = handle.field_graph_plan_id;
            CHECK(handle.field_report &&
                      handle.field_report->field_status ==
                          VOLVOXAI_V1_NATIVE_STATUS_OK,
                  "CreateGraphPlan reports OK");
            CHECK(graph_plan_id > 0, "graph plan id issued");
            CHECK(handle.field_source_kind ==
                      VOLVOXAI_V1_GRAPH_PLAN_SOURCE_KIND_MODEL,
                  "model source kind retained");
            CHECK(handle.field_plan != NULL,
                  "CreateGraphPlan returns the complete plan");
            volvoxai_v1_graph_plan_handle_free(&handle);
            vx_call_free(&client, payload);
        }

        if (graph_plan_id > 0) {
            {
                VolvoxaiV1GraphPlanRef call_request;
                volvoxai_v1_graph_plan_ref_init(&call_request);
                call_request.field_graph_plan_id = graph_plan_id;
                VX_CALL_MESSAGE(&client, VX_RPC_VX_PLANNING_SERVICE_GET_GRAPH_PLAN,
                    volvoxai_v1_graph_plan_ref, &call_request, payload, payload_len);
            }
            CHECK(payload != NULL, "GetGraphPlan returns a payload");
            if (payload) {
                CHECK(DECODE(payload, payload_len, info,
                             volvoxai_v1_graph_plan_info),
                      "GraphPlanInfo decodes");
                CHECK(info.field_graph_plan_id == graph_plan_id &&
                          info.field_plan != NULL && info.field_report &&
                          info.field_report->field_status ==
                              VOLVOXAI_V1_NATIVE_STATUS_OK,
                      "GetGraphPlan preserves the live immutable plan");
                volvoxai_v1_graph_plan_info_free(&info);
                vx_call_free(&client, payload);
            }

            encoded = NULL;
            encoded_len = 0u;
            volvoxai_v1_resolve_graph_plan_request_init(&resolve);
            volvoxai_v1_empty_init(&minimum);
            resolve.field_graph_plan_id = graph_plan_id;
            resolve.which_binding = 2; /* ResolveGraphPlanRequest.minimum */
            resolve.field_minimum = &minimum;
            CHECK(volvoxai_v1_resolve_graph_plan_request_encode(
                      &resolve, &encoded, &encoded_len) == SYNURANG_LITE_OK,
                  "minimum GraphPlan binding encodes");
            resolve.field_minimum = NULL;
            volvoxai_v1_resolve_graph_plan_request_free(&resolve);
            payload = vx_call_bytes(&client, VX_RPC_VX_PLANNING_SERVICE_RESOLVE_GRAPH_PLAN,
                encoded, (int32_t)encoded_len, &payload_len);
            allocator->deallocate(allocator->context, encoded);
            CHECK(payload != NULL, "ResolveGraphPlan returns a payload");
            if (payload) {
                CHECK(DECODE(payload, payload_len, resolved,
                             volvoxai_v1_resolved_graph_plan),
                      "ResolvedGraphPlan decodes");
                CHECK(resolved.field_graph_plan_id == graph_plan_id &&
                          resolved.field_report &&
                          resolved.field_report->field_status ==
                              VOLVOXAI_V1_NATIVE_STATUS_OK,
                      "minimum GraphPlan binding resolves");
                CHECK(resolved.field_signature.len > 0u,
                      "resolved plan carries a canonical signature");
                volvoxai_v1_resolved_graph_plan_free(&resolved);
                vx_call_free(&client, payload);
            }

            {
                VolvoxaiV1GraphPlanRef call_request;
                volvoxai_v1_graph_plan_ref_init(&call_request);
                call_request.field_graph_plan_id = graph_plan_id;
                VX_CALL_MESSAGE(&client, VX_RPC_VX_PLANNING_SERVICE_RELEASE_GRAPH_PLAN,
                    volvoxai_v1_graph_plan_ref, &call_request, payload, payload_len);
            }
            CHECK(payload != NULL, "ReleaseGraphPlan returns a payload");
            if (payload) {
                CHECK(DECODE(payload, payload_len, released,
                             volvoxai_v1_operation_report),
                      "ReleaseGraphPlan report decodes");
                CHECK(released.field_status == VOLVOXAI_V1_NATIVE_STATUS_OK,
                      "ReleaseGraphPlan reports OK");
                volvoxai_v1_operation_report_free(&released);
                vx_call_free(&client, payload);
            }

            {
                VolvoxaiV1GraphPlanRef call_request;
                volvoxai_v1_graph_plan_ref_init(&call_request);
                call_request.field_graph_plan_id = graph_plan_id;
                VX_CALL_MESSAGE(&client, VX_RPC_VX_PLANNING_SERVICE_GET_GRAPH_PLAN,
                    volvoxai_v1_graph_plan_ref, &call_request, payload, payload_len);
            }
            CHECK(payload != NULL, "stale GetGraphPlan still reports");
            if (payload) {
                CHECK(DECODE(payload, payload_len, info,
                             volvoxai_v1_graph_plan_info),
                      "stale GraphPlanInfo decodes");
                CHECK(info.field_plan == NULL && info.field_report &&
                          info.field_report->field_status ==
                              VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED,
                      "released GraphPlan is a typed stale handle");
                CHECK(info.field_report && info.field_report->field_lineage &&
                          info.field_report->field_lineage->field_graph_plan_id ==
                              (uint64_t)graph_plan_id,
                      "stale GraphPlan report retains requested lineage");
                volvoxai_v1_graph_plan_info_free(&info);
                vx_call_free(&client, payload);
            }
        }
    }

    /* --- compile -------------------------------------------------------- */
    {
        /* Dot and underscore are part of the cross-host backend-name grammar.
         * This absent candidate must be treated as unavailable, not malformed. */
        static const char* const candidate_names[] = {
            "missing.provider_name", "cpu"
        };
        VolvoxaiV1BackendPolicy policy;
        VolvoxaiV1CompiledModelHandle handle;
        SynurangLiteBytes* candidates;
        uint8_t* encoded = NULL;
        size_t encoded_len = 0u;

        volvoxai_v1_backend_policy_init(&policy);
        candidates = (SynurangLiteBytes*)policy._allocator->allocate(
            policy._allocator->context, sizeof(*candidates) * 2u);
        CHECK(candidates != NULL, "compile candidate storage allocated");
        if (!candidates) return 1;
        memset(candidates, 0, sizeof(*candidates) * 2u);
        policy.field_backends.data = candidates;
        policy.field_backends.len = 2u;
        policy.field_backends.cap = 2u;
        for (size_t index = 0u; index < 2u; index++) {
            CHECK(synurang_lite_bytes_assign(
                      policy._allocator, &candidates[index],
                      candidate_names[index], strlen(candidate_names[index])) ==
                      SYNURANG_LITE_OK,
                  "compile candidate assigned");
        }
        CHECK(volvoxai_v1_backend_policy_encode(
                  &policy, &encoded, &encoded_len) == SYNURANG_LITE_OK,
              "BackendPolicy encodes");
        {
            VolvoxaiV1CompileModelRequest call_request;
            volvoxai_v1_compile_model_request_init(&call_request);
            call_request.field_model_id = model_id;
            VolvoxaiV1BackendPolicy policy;
            volvoxai_v1_backend_policy_init(&policy);
            if ((encoded) && ((int32_t)encoded_len) > 0) {
                CHECK(volvoxai_v1_backend_policy_decode(&policy, encoded, (size_t)((int32_t)encoded_len)) == SYNURANG_LITE_OK,
                      "compile policy decodes in every build profile");
                call_request.field_policy = &policy;
            }
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_COMPILE_MODEL,
                volvoxai_v1_compile_model_request, &call_request, payload, payload_len);
            volvoxai_v1_backend_policy_free(&policy);
        }
        allocator->deallocate(allocator->context, encoded);
        volvoxai_v1_backend_policy_free(&policy);
        CHECK(payload != NULL, "CompileModel returns a payload");
        if (!payload) {
            fprintf(stderr, "  %s\n", last_error());
            return 1;
        }
        CHECK(DECODE(payload, payload_len, handle, volvoxai_v1_compiled_model_handle),
              "CompiledModelHandle decodes");
        report_summary("CompileModel", handle.field_report);
        CHECK(handle.field_report && handle.field_report->field_compilation,
              "compile evidence is typed");
        if (handle.field_report && handle.field_report->field_compilation) {
            const VolvoxaiV1CompilationEvidence* evidence =
                handle.field_report->field_compilation;
            CHECK(evidence->field_candidates.len == 2u,
                  "every compile candidate is projected");
            if (evidence->field_candidates.len == 2u) {
                const VolvoxaiV1CompilationCandidate* first =
                    &evidence->field_candidates.data[0];
                const VolvoxaiV1CompilationCandidate* second =
                    &evidence->field_candidates.data[1];
                CHECK(first->field_outcome ==
                          VOLVOXAI_V1_CANDIDATE_OUTCOME_UNAVAILABLE,
                      "missing candidate is unavailable");
                CHECK(first->field_status ==
                          VOLVOXAI_V1_NATIVE_STATUS_BACKEND_UNAVAILABLE,
                      "missing candidate retains its status");
                CHECK(second->field_outcome ==
                          VOLVOXAI_V1_CANDIDATE_OUTCOME_SELECTED,
                      "CPU candidate is selected");
                CHECK(second->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK,
                      "selected candidate is successful");
            }
        }
        if (handle.field_memory_bounds) {
            const VolvoxaiV1MemoryDomainAttestation* attestation = handle.field_memory_bounds;
            uint64_t maximum_tensor = 0u, maximum_resident = 0u;
            size_t bound_index;
            CHECK(attestation != NULL,
                  "compile snapshot carries bounded-domain attestation");
            if (attestation) {
                CHECK(bytes_equal_text(&attestation->field_proof_protocol,
                                       "canonical-symbolic-domain-proof/v1"),
                      "compile attestation uses canonical proof protocol");
                CHECK(bytes_equal_text(&attestation->field_resource_protocol,
                                       "bounded-resource-maxima/v1"),
                      "compile attestation uses bounded-resource protocol");
                CHECK(attestation->field_bounds.len == 2u,
                      "compile attestation carries two independent coarse bounds");
                for (bound_index = 0u;
                     bound_index < attestation->field_bounds.len;
                     bound_index++) {
                    const VolvoxaiV1MemoryBoundProof* bound =
                        &attestation->field_bounds.data[bound_index];
                    CHECK(bound->field_maximum_bytes != NULL,
                          "every compile bound carries a present maximum");
                    if (bytes_equal_text(&bound->field_budget_domain_id,
                                         "maximum-tensor") &&
                        bound->field_maximum_bytes) {
                        CHECK(bound->field_kind ==
                                  VOLVOXAI_V1_MEMORY_BOUND_KIND_MAXIMUM_TENSOR,
                              "maximum-tensor bound kind is exact");
                        maximum_tensor = bound->field_maximum_bytes->field_bytes;
                    } else if (bytes_equal_text(
                                   &bound->field_budget_domain_id,
                                   "provider-resident") &&
                               bound->field_maximum_bytes) {
                        CHECK(bound->field_kind ==
                                  VOLVOXAI_V1_MEMORY_BOUND_KIND_ORDINARY_RESIDENT,
                              "provider-resident bound kind is exact");
                        maximum_resident = bound->field_maximum_bytes->field_bytes;
                        CHECK(bound->field_limit_bytes == NULL,
                              "built-in CPU publishes no artificial resource limit");
                    }
                }
                CHECK(maximum_tensor > 0u,
                      "maximum-tensor proof is nonzero for the fixture model");
                CHECK(maximum_resident >= maximum_tensor,
                      "resident proof covers the maximum tensor");
            }
        } else {
            CHECK(0, "compilation carries independent memory bounds");
        }
        compiled_id = handle.field_compiled_model_id;
        CHECK(compiled_id > 0, "compiled model id issued");
        volvoxai_v1_compiled_model_handle_free(&handle);
        vx_call_free(&client, payload);
        if (compiled_id <= 0) return 1;
    }

    /* A compiled child retains its Model and Runtime. Releasing the public
     * parent ids must return immediately and leave compiled work usable. */
    {
        VolvoxaiV1ModelRef call_request;
        volvoxai_v1_model_ref_init(&call_request);
        call_request.field_model_id = model_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_MODEL,
            volvoxai_v1_model_ref, &call_request, payload, payload_len);
    }
    CHECK(payload != NULL, "release retained model parent");
    if (payload) vx_call_free(&client, payload);
    {
        VolvoxaiV1RuntimeRef call_request;
        volvoxai_v1_runtime_ref_init(&call_request);
        call_request.field_runtime_id = runtime_id;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RUNTIME,
            volvoxai_v1_runtime_ref, &call_request, payload, payload_len);
    }
    CHECK(payload != NULL, "release retained runtime parent");
    if (payload) vx_call_free(&client, payload);

    /* --- context and its declared inputs -------------------------------- */
    {
        VolvoxaiV1ExecutionContextHandle handle;
        VolvoxaiV1CreateExecutionContextRequest request;
        uint8_t* encoded = NULL;
        size_t encoded_len = 0u;
        volvoxai_v1_create_execution_context_request_init(&request);
        request.field_compiled_model_id = compiled_id;
        request.field_graph = request._allocator->allocate(request._allocator->context, sizeof(*request.field_graph));
        CHECK(request.field_graph != NULL, "graph execution mode allocated");
        if (!request.field_graph) return 1;
        volvoxai_v1_graph_execution_init_with_allocator(request.field_graph, request._allocator);
        request.which_execution = 2;
        CHECK(volvoxai_v1_create_execution_context_request_encode(
                  &request, &encoded, &encoded_len) == SYNURANG_LITE_OK,
              "CreateExecutionContext request encodes");
        volvoxai_v1_create_execution_context_request_free(&request);
        payload = vx_call_bytes(&client, VX_RPC_VX_INFERENCE_SERVICE_CREATE_EXECUTION_CONTEXT,
            encoded, (int32_t)encoded_len, &payload_len);
        synurang_lite_default_allocator()->deallocate(
            synurang_lite_default_allocator()->context, encoded);
        CHECK(payload != NULL, "CreateExecutionContext returns a payload");
        if (!payload) {
            fprintf(stderr, "  %s\n", last_error());
            return 1;
        }
        CHECK(DECODE(payload, payload_len, handle,
                     volvoxai_v1_execution_context_handle),
              "ExecutionContextHandle decodes");
        report_summary("CreateContext", handle.field_report);
        context_id = handle.field_context_id;
        CHECK(context_id > 0, "context id issued");
        CHECK(handle.field_inputs.len > 0u, "declared inputs travel with the handle");
        if (handle.field_inputs.len) {
            const VolvoxaiV1TensorSpec* spec = &handle.field_inputs.data[0];
            printf("  input[0] name=%.*s dtype=%d rank=%zu\n",
                   (int)spec->field_name.len,
                   spec->field_name.data ? (const char*)spec->field_name.data : "",
                   (int)spec->field_dtype, spec->field_dimensions.len);
            CHECK(spec->field_name.len > 0u, "input spec carries a name");
            CHECK(spec->field_dimensions.len > 0u, "input spec carries dimensions");
            /* A zero DimensionKind would mean the axis never validated. */
            CHECK(spec->field_dimensions.data[0].field_kind !=
                      VOLVOXAI_V1_DIMENSION_KIND_UNSPECIFIED,
                  "dimension kind is explicit");
        }
        volvoxai_v1_execution_context_handle_free(&handle);
        vx_call_free(&client, payload);
        if (context_id <= 0) return 1;
    }

    /* Execute packets own inline bytes or buffer leases. No result handle is
     * needed, and releasing ancestors above must not break this context. */
    {
        VolvoxaiV1ExecuteRequest execute;
        VolvoxaiV1InputBindings bindings;
        VolvoxaiV1ExecutionScope scope;
        VolvoxaiV1Empty full;
        VolvoxaiV1OutputOptions outputs;
        VolvoxaiV1TensorOutputSelection selection;
        volvoxai_v1_execute_request_init(&execute);
        volvoxai_v1_input_bindings_init(&bindings);
        volvoxai_v1_execution_scope_init(&scope);
        volvoxai_v1_empty_init(&full);
        volvoxai_v1_output_options_init(&outputs);
        volvoxai_v1_tensor_output_selection_init(&selection);
        execute.field_context_id = context_id;
        execute.field_inputs = &bindings;
        execute.field_scope = &scope;
        execute.field_outputs = &outputs;
        scope.field_entrypoint = (SynurangLiteBytes){.data=(uint8_t*)"evaluate", .len=8};
        scope.which_range = 2; scope.field_full = &full;
        SynurangLiteBytes output_name = {.data=(uint8_t*)"logits", .len=6};
        selection.field_names.data = &output_name; selection.field_names.len = 1;
        outputs.field_selection = &selection;
        VolvoxaiV1ExecuteResponse response;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_EXECUTE,
            volvoxai_v1_execute_request, &execute, payload, payload_len);
        CHECK(payload != NULL, "Execute empty inputs returns typed refusal");
        if (payload) {
            CHECK(DECODE(payload, payload_len, response, volvoxai_v1_execute_response), "Execute refusal decodes");
            CHECK(response.field_report && response.field_report->field_input_issue &&
                response.field_report->field_input_issue->field_code == VOLVOXAI_V1_INPUT_VALIDATION_CODE_MISSING_INPUT,
                "missing inputs report expected tensor contract");
            CHECK(response.field_outputs.len == 0, "refusal transfers no outputs");
            volvoxai_v1_execute_response_free(&response); vx_call_free(&client, payload);
        }
        int32_t tokens[8] = {1,2,3,4,5,6,7,8}, positions[8] = {0,1,2,3,4,5,6,7};
        int32_t keep[8] = {1,1,1,1,1,1,1,1}, active_lengths[1] = {8};
        int64_t sequence_shape[2] = {1,8}, lengths_shape[1] = {1};
        const char* names[] = {"tokens", "positions", "keep", "active_lengths"};
        const int32_t* values[] = {tokens, positions, keep, active_lengths};
        VolvoxaiV1Tensor tensors[4];
        VolvoxaiV1BorrowedBuffer borrowed[4];
        VolvoxaiV1NativeResource resources[4];
        for (size_t i = 0; i < 4; i++) {
            volvoxai_v1_tensor_init(&tensors[i]);
            volvoxai_v1_borrowed_buffer_init(&borrowed[i]);
            volvoxai_v1_native_resource_init(&resources[i]);
            tensors[i].field_name = (SynurangLiteBytes){.data=(uint8_t*)names[i], .len=strlen(names[i])};
            tensors[i].field_dtype = VOLVOXAI_V1_DATA_TYPE_I32;
            tensors[i].field_shape.data = i == 3 ? lengths_shape : sequence_shape;
            tensors[i].field_shape.len = i == 3 ? 1 : 2;
            resources[i].field_kind = VOLVOXAI_V1_NATIVE_RESOURCE_KIND_HOST;
            resources[i].field_handle = (uint64_t)(uintptr_t)values[i];
            resources[i].field_size_bytes = i == 3 ? sizeof(active_lengths) : sizeof(tokens);
            borrowed[i].field_resource = &resources[i]; borrowed[i].field_length_bytes = resources[i].field_size_bytes;
            tensors[i].which_payload = 7; tensors[i].field_borrowed = &borrowed[i];
        }
        bindings.field_values.data = tensors; bindings.field_values.len = 4;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_EXECUTE,
            volvoxai_v1_execute_request, &execute, payload, payload_len);
        uint8_t* expected = NULL; size_t expected_bytes = 0;
        CHECK(payload != NULL, "Execute returns inline packet");
        if (payload) {
            CHECK(DECODE(payload, payload_len, response, volvoxai_v1_execute_response), "ExecuteResponse decodes");
            report_summary("Execute", response.field_report);
            CHECK(response.field_report && response.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK,
                "Execute reports OK");
            CHECK(response.field_execution_id && response.field_progress &&
                response.field_progress->field_state == VOLVOXAI_V1_EXECUTION_CONTEXT_STATE_READY,
                "packet carries stable call identity and READY progress");
            CHECK(response.field_outputs.len == 1 && response.field_outputs.data[0].which_payload == 5,
                "default storage is inline");
            if (response.field_outputs.len == 1) {
                expected_bytes = response.field_outputs.data[0].field_inline.len;
                expected = malloc(expected_bytes); CHECK(expected != NULL, "snapshot comparison allocated");
                if (expected) memcpy(expected, response.field_outputs.data[0].field_inline.data, expected_bytes);
            }
            volvoxai_v1_execute_response_free(&response); vx_call_free(&client, payload);
        }
        outputs.field_storage = VOLVOXAI_V1_OUTPUT_STORAGE_OWNED_BUFFER;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_EXECUTE,
            volvoxai_v1_execute_request, &execute, payload, payload_len);
        CHECK(payload != NULL, "owned-buffer Execute returns packet");
        if (payload) {
            CHECK(DECODE(payload, payload_len, response, volvoxai_v1_execute_response), "owned ExecuteResponse decodes");
            vx_call_free(&client, payload);
            CHECK(response.field_outputs.len == 1 && response.field_outputs.data[0].field_buffer,
                "owned storage returns independent buffer lease");
            if (response.field_outputs.len == 1 && response.field_outputs.data[0].field_buffer && expected) {
                uint8_t* destination = malloc(expected_bytes); CHECK(destination != NULL, "destination allocated");
                VolvoxaiV1CopyTensorsRequest copy;
                VolvoxaiV1BorrowedBuffer into;
                VolvoxaiV1NativeResource resource;
                volvoxai_v1_copy_tensors_request_init(&copy);
                volvoxai_v1_borrowed_buffer_init(&into); volvoxai_v1_native_resource_init(&resource);
                resource.field_kind = VOLVOXAI_V1_NATIVE_RESOURCE_KIND_HOST;
                resource.field_handle = (uint64_t)(uintptr_t)destination; resource.field_size_bytes = expected_bytes;
                into.field_resource = &resource; into.field_length_bytes = expected_bytes;
                copy.field_sources.data = response.field_outputs.data; copy.field_sources.len = 1;
                copy.field_into.data = &into; copy.field_into.len = 1;
                VX_CALL_MESSAGE(&client, VX_RPC_VX_BUFFER_SERVICE_COPY_TENSORS,
                    volvoxai_v1_copy_tensors_request, &copy, payload, payload_len);
                CHECK(payload != NULL, "CopyTensors accepts native destination");
                if (payload) {
                    VolvoxaiV1TensorBatch copied;
                    CHECK(DECODE(payload, payload_len, copied, volvoxai_v1_tensor_batch), "copied tensor packet decodes");
                    CHECK(copied.field_report && copied.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK &&
                        !memcmp(destination, expected, expected_bytes), "owned snapshot matches inline bytes");
                    volvoxai_v1_tensor_batch_free(&copied); vx_call_free(&client, payload);
                }
                free(destination);
                VolvoxaiV1BufferRefs release; volvoxai_v1_buffer_refs_init(&release);
                release.field_buffer_ids.data = &response.field_outputs.data[0].field_buffer->field_buffer_id;
                release.field_buffer_ids.len = 1;
                VX_CALL_MESSAGE(&client, VX_RPC_VX_BUFFER_SERVICE_RELEASE_BUFFERS,
                    volvoxai_v1_buffer_refs, &release, payload, payload_len);
                CHECK(payload != NULL, "release owned snapshot"); if (payload) vx_call_free(&client, payload);
            }
            volvoxai_v1_execute_response_free(&response);
        }
        free(expected);
        outputs.field_storage = VOLVOXAI_V1_OUTPUT_STORAGE_INLINE;
        /* Scheduler owns the identical Execute envelope and transfers it once. */
        for (size_t i = 0; i < 4; i++) {
            tensors[i].which_payload = 5;
            tensors[i].field_inline = (SynurangLiteBytes){.data=(uint8_t*)values[i], .len=i == 3 ? sizeof(active_lengths) : sizeof(tokens)};
        }
        VolvoxaiV1SubmitRequest submit; volvoxai_v1_submit_request_init(&submit);
        submit.which_execution = 1; submit.field_execute = &execute;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_SCHEDULER_SERVICE_SUBMIT,
            volvoxai_v1_submit_request, &submit, payload, payload_len);
        int64_t request_id = 0;
        CHECK(payload != NULL, "Submit returns handle");
        if (payload) {
            VolvoxaiV1RequestHandle handle;
            CHECK(DECODE(payload, payload_len, handle, volvoxai_v1_request_handle), "RequestHandle decodes");
            report_summary("Submit", handle.field_report);
            request_id = handle.field_request_id; CHECK(request_id > 0, "scheduler issues request identity");
            volvoxai_v1_request_handle_free(&handle); vx_call_free(&client, payload);
        }
        if (request_id) {
            cancel_wait_observer(request_id, 1); cancel_wait_observer(request_id, 4);
            VolvoxaiV1RequestRef ref; volvoxai_v1_request_ref_init(&ref); ref.field_request_id = request_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_SCHEDULER_SERVICE_WAIT_REQUEST,
                volvoxai_v1_request_ref, &ref, payload, payload_len);
            CHECK(payload != NULL, "WaitRequest returns terminal snapshot");
            if (payload) {
                VolvoxaiV1RequestInfo info;
                CHECK(DECODE(payload, payload_len, info, volvoxai_v1_request_info), "RequestInfo decodes");
                CHECK(info.field_state == VOLVOXAI_V1_REQUEST_STATE_SUCCEEDED, "scheduled execution succeeds");
                volvoxai_v1_request_info_free(&info); vx_call_free(&client, payload);
            }
            wait_progresses_during_metadata_backlog(request_id, 1);
            wait_progresses_during_metadata_backlog(request_id, 2);
            wait_progresses_during_metadata_backlog(request_id, 64);
            for (int take = 0; take < 2; take++) {
                VX_CALL_MESSAGE(&client, VX_RPC_VX_SCHEDULER_SERVICE_TAKE_REQUEST_RESULT,
                    volvoxai_v1_request_ref, &ref, payload, payload_len);
                CHECK(payload != NULL, "TakeRequestResult returns packet");
                if (payload) {
                    VolvoxaiV1ExecutionPacket packet;
                    CHECK(DECODE(payload, payload_len, packet, volvoxai_v1_execution_packet), "ExecutionPacket decodes");
                    CHECK(packet.field_report && packet.field_report->field_status == (take
                        ? VOLVOXAI_V1_NATIVE_STATUS_NOT_FOUND : VOLVOXAI_V1_NATIVE_STATUS_OK),
                        "scheduled packet handoff is one-shot");
                    CHECK(take ? packet.field_outputs.len == 0 : packet.field_outputs.len == 1,
                        "only first Take transfers outputs");
                    volvoxai_v1_execution_packet_free(&packet); vx_call_free(&client, payload);
                }
            }
            VX_CALL_MESSAGE(&client, VX_RPC_VX_SCHEDULER_SERVICE_RELEASE_REQUEST,
                volvoxai_v1_request_ref, &ref, payload, payload_len);
            CHECK(payload != NULL, "release scheduled request"); if (payload) vx_call_free(&client, payload);
        }
    }

    /* Unknown context references fail through the unified inspection API. */
    {
        VolvoxaiV1ExecutionContextRef request;
        volvoxai_v1_execution_context_ref_init(&request); request.field_context_id = 999999;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_GET_EXECUTION_CONTEXT_INFO,
            volvoxai_v1_execution_context_ref, &request, payload, payload_len);
        CHECK(payload != NULL, "unknown context returns typed report");
        if (payload) {
            VolvoxaiV1ExecutionContextInfo info;
            CHECK(DECODE(payload, payload_len, info, volvoxai_v1_execution_context_info), "context info decodes");
            CHECK(info.field_report && info.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED,
                "unknown context is refused");
            volvoxai_v1_execution_context_info_free(&info); vx_call_free(&client, payload);
        }
    }

    /* --- teardown in child-before-parent order -------------------------- */
    {
        {
            VolvoxaiV1ExecutionContextRef request;
            volvoxai_v1_execution_context_ref_init(&request);
            request.field_context_id = context_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_EXECUTION_CONTEXT,
                volvoxai_v1_execution_context_ref, &request, payload, payload_len);
            CHECK(payload != NULL, "RELEASE_EXECUTION_CONTEXT");
            vx_call_free(&client, payload);
        }
        {
            VolvoxaiV1CompiledModelRef request;
            volvoxai_v1_compiled_model_ref_init(&request);
            request.field_compiled_model_id = compiled_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_COMPILED_MODEL,
                volvoxai_v1_compiled_model_ref, &request, payload, payload_len);
            CHECK(payload != NULL, "RELEASE_COMPILED_MODEL");
            vx_call_free(&client, payload);
        }
        {
            VolvoxaiV1ModelRef request;
            volvoxai_v1_model_ref_init(&request);
            request.field_model_id = model_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_MODEL,
                volvoxai_v1_model_ref, &request, payload, payload_len);
            CHECK(payload != NULL, "RELEASE_MODEL");
            vx_call_free(&client, payload);
        }
        {
            VolvoxaiV1RuntimeRef request;
            volvoxai_v1_runtime_ref_init(&request);
            request.field_runtime_id = runtime_id;
            VX_CALL_MESSAGE(&client, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RUNTIME,
                volvoxai_v1_runtime_ref, &request, payload, payload_len);
            CHECK(payload != NULL, "RELEASE_RUNTIME");
            vx_call_free(&client, payload);
        }
    }

    /* --- explicit process observation ---------------------------------- */
    {
        VolvoxaiV1GetResourceSnapshotRequest request;
        VolvoxaiV1ResourceSnapshotResponse observation;
        VolvoxaiV1Empty module;
        volvoxai_v1_get_resource_snapshot_request_init(&request);
        volvoxai_v1_empty_init(&module);
        request.which_scope = 1;
        request.field_module = &module;
        VX_CALL_MESSAGE(&client, VX_RPC_VX_PROFILING_SERVICE_GET_RESOURCE_SNAPSHOT,
            volvoxai_v1_get_resource_snapshot_request, &request, payload, payload_len);
        CHECK(payload != NULL, "GetResourceSnapshot returns a payload");
        if (payload) {
            CHECK(DECODE(payload, payload_len, observation, volvoxai_v1_resource_snapshot_response), "snapshot decodes");
            CHECK(observation.field_snapshot != NULL, "explicit observation is present");
            if (observation.field_snapshot) {
                CHECK(observation.field_snapshot->field_memory->field_inventory == VOLVOXAI_V1_MEMORY_INVENTORY_KIND_PARTIAL, "partial inventory cannot imply a total");
                CHECK(observation.field_snapshot->field_process && observation.field_snapshot->field_cpu, "RSS and CPU are explicit observations");
                CHECK(observation.field_snapshot->field_observation_end_ns >= observation.field_snapshot->field_observation_start_ns, "monotonic observation interval");
            }
            volvoxai_v1_resource_snapshot_response_free(&observation);
            vx_call_free(&client, payload);
        }
    }

    if (failures == 0) {
        printf("all checks passed\n");
        return 0;
    }
    fprintf(stderr, "%d check(s) failed\n", failures);
    return 1;
}
