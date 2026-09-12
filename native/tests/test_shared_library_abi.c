/* Load each release profile without linking an engine. Its one module ABI
 * must own independent registries, report protocol errors, and drain calls
 * before the shared library is unloaded. */
#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../cli/call_client.h"
#include "../src/generated/proto_methods.h"

static int failures;
#define CHECK(condition, message) do { \
    if (!(condition)) { fprintf(stderr, "FAIL %s\n", message); ++failures; } \
} while (0)

static void* required(void* library, const char* name) {
    void* symbol = dlsym(library, name);
    if (!symbol) { fprintf(stderr, "FAIL missing export: %s\n", name); ++failures; }
    return symbol;
}

static void must_be_absent(void* library, const char* name) {
    if (dlsym(library, name)) {
        fprintf(stderr, "FAIL unexpected export: %s\n", name);
        ++failures;
    }
}

/* These requests inspect transport status rather than treating a protobuf
 * response as success. Unknown methods can reject Send before HalfClose. */
static int terminal_status(VxCallClient* client, const char* path,
                           const uint8_t* data, size_t size) {
    SynurangCallOptions options = { sizeof(options), 0, 0, 0, UINT64_MAX };
    uint64_t call = client->api->open(client->instance, path, &options);
    CHECK(call != 0, "protocol test call opens");
    if (!call) return -1;
    int status = client->api->send(client->instance, call, data, (uint32_t)size);
    if (status == SYNURANG_OK) (void)client->api->half_close(client->instance, call);
    int code = -1;
    for (;;) {
        SynurangReadResult result = {0};
        status = client->api->receive(client->instance, call, &result);
        if (status != SYNURANG_OK) {
            vx_call_free(client, result.data);
            break;
        }
        if (result.kind == SYNURANG_READ_FINISHED) {
            code = result.code;
            if (code) CHECK(result.data != NULL && result.size > 0,
                            "failed RPC includes structured core Error");
            vx_call_free(client, result.data);
            break;
        }
        vx_call_free(client, result.data);
        if (result.kind == SYNURANG_READ_PENDING) vx_call_turn(client);
    }
    client->api->release(client->instance, call);
    return code;
}

static size_t runtime_ref(uint8_t* target, int64_t id) {
    uint64_t value = (uint64_t)id;
    size_t size = 0;
    target[size++] = 8; /* RuntimeRef.runtime_id, field 1, varint. */
    do {
        target[size] = (uint8_t)(value & 127u);
        value >>= 7;
        if (value) target[size] |= 128u;
        ++size;
    } while (value);
    return size;
}

int main(int argc, char** argv) {
    const char* path = argc > 1 ? argv[1] : "native/libvolvoxai.so";
    const char* profile = argc > 2 ? argv[2] : "inference";
    int full = strcmp(profile, "full") == 0;
    if (!full && strcmp(profile, "inference") != 0) return 2;
    void* library = dlopen(path, RTLD_NOW | RTLD_LOCAL);
    if (!library) { fprintf(stderr, "FAIL dlopen %s: %s\n", path, dlerror()); return 1; }

    static const char* const absent[] = {
        "Synurang_Invoke_VxPlatformService", "Synurang_Invoke_VxInferenceService",
        "Synurang_Invoke_VxSchedulerService", "Synurang_Invoke_VxPlanningService",
        "Synurang_Invoke_VxTextService", "Synurang_Invoke_VxTrainingService",
        "Synurang_Invoke_VxQuantizationService", "Synurang_Stream_Send",
        "Synurang_Stream_TrySend", "Synurang_Stream_Recv", "Synurang_Stream_TryRecv",
        "Synurang_Stream_CloseSend", "Synurang_Stream_Close", "Synurang_Free",
        "synurang_runtime_create", "vx_platform_get_platform_info",
        "vx_inference_create_runtime_pb", "vx_training_create_trainer",
        "vx_runtime_create", "vx_runtime_register_provider", "vx_runtime_load_model",
        "vx_model_compile", "vx_execution_context_execute", "vx_arena_alloc",
        "vx_graph_plan_build", "vx_api_module_registry",
    };
    for (size_t index = 0; index < sizeof(absent) / sizeof(absent[0]); ++index)
        must_be_absent(library, absent[index]);
    (void)required(library, "vx_backend_register_provider");
    if (full) (void)required(library, "volvoxai_v1_create_trainer_request_init");
    else must_be_absent(library, "volvoxai_v1_create_trainer_request_init");

    SynurangGetApiFn get_api = (SynurangGetApiFn)required(library, "Synurang_GetApi");
    SynurangLiteStatus (*decode_runtime)(VolvoxaiV1RuntimeHandle*, const uint8_t*, size_t) =
        (SynurangLiteStatus (*)(VolvoxaiV1RuntimeHandle*, const uint8_t*, size_t))
            required(library, "volvoxai_v1_runtime_handle_decode");
    void (*free_runtime)(VolvoxaiV1RuntimeHandle*) =
        (void (*)(VolvoxaiV1RuntimeHandle*))required(library, "volvoxai_v1_runtime_handle_free");
    SynurangLiteStatus (*decode_backends)(VolvoxaiV1BackendList*, const uint8_t*, size_t) =
        (SynurangLiteStatus (*)(VolvoxaiV1BackendList*, const uint8_t*, size_t))
            required(library, "volvoxai_v1_backend_list_decode");
    void (*free_backends)(VolvoxaiV1BackendList*) =
        (void (*)(VolvoxaiV1BackendList*))required(library, "volvoxai_v1_backend_list_free");
    if (failures) { dlclose(library); return 1; }

    const SynurangApi* api = get_api();
    CHECK(api && api->abi_version == SYNURANG_CALL_ABI_VERSION &&
          api->struct_size == sizeof(*api), "exact module ABI table");
    if (failures) { dlclose(library); return 1; }
    VxCallClient first, second;
    if (!vx_call_client_open_api(&first, api)) { dlclose(library); return 1; }
    if (!vx_call_client_open_api(&second, api)) {
        vx_call_client_close(&first); dlclose(library); return 1;
    }
    CHECK(first.instance != second.instance, "module creates independent owners");

    int32_t size = 0;
    uint8_t* response = vx_call_bytes(&first, VX_RPC_VX_PLATFORM_SERVICE_GET_PLATFORM_INFO,
                                     NULL, 0, &size);
    CHECK(response && size > 0, "GetPlatformInfo responds through module ABI");
    vx_call_free(&first, response);

    response = vx_call_bytes(&first, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME, NULL, 0, &size);
    VolvoxaiV1RuntimeHandle runtime = { ._allocator = synurang_lite_default_allocator() };
    CHECK(response && decode_runtime(&runtime, response, (size_t)size) == SYNURANG_LITE_OK &&
          runtime.field_runtime_id > 0 && runtime.field_report &&
          runtime.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK,
          "first owner creates a Runtime");
    int64_t runtime_id = runtime.field_runtime_id;
    free_runtime(&runtime);
    vx_call_free(&first, response);
    if (runtime_id > 0) {
        uint8_t request[11];
        size_t request_size = runtime_ref(request, runtime_id);
        response = vx_call_bytes(&second, VX_RPC_VX_INFERENCE_SERVICE_LIST_BACKENDS,
                                 request, request_size, &size);
        VolvoxaiV1BackendList backends = { ._allocator = synurang_lite_default_allocator() };
        CHECK(response && decode_backends(&backends, response, (size_t)size) == SYNURANG_LITE_OK &&
              backends.field_report && backends.field_report->field_status ==
                  VOLVOXAI_V1_NATIVE_STATUS_HANDLE_DISPOSED,
              "second owner rejects first owner's Runtime ID");
        free_backends(&backends);
        vx_call_free(&second, response);
        response = vx_call_bytes(&second, VX_RPC_VX_INFERENCE_SERVICE_RELEASE_RUNTIME,
                                 request, request_size, &size);
        CHECK(response != NULL, "cross-owner release is harmless and idempotent");
        vx_call_free(&second, response);
        response = vx_call_bytes(&first, VX_RPC_VX_INFERENCE_SERVICE_LIST_BACKENDS,
                                 request, request_size, &size);
        CHECK(response && decode_backends(&backends, response, (size_t)size) == SYNURANG_LITE_OK &&
              backends.field_report && backends.field_report->field_status == VOLVOXAI_V1_NATIVE_STATUS_OK,
              "cross-owner release leaves original Runtime usable");
        free_backends(&backends);
        vx_call_free(&first, response);
    }

    CHECK(terminal_status(&first, "/volvoxai.v1.VxPlatformService/NoSuchMethod", NULL, 0) == 12,
          "unknown method finishes UNIMPLEMENTED");
    const uint8_t malformed[] = {0x80};
    CHECK(terminal_status(&first, VX_RPC_VX_INFERENCE_SERVICE_CREATE_RUNTIME,
                          malformed, sizeof(malformed)) == 3,
          "malformed protobuf finishes INVALID_ARGUMENT");
    CHECK(terminal_status(&first, "/volvoxai.v1.VxTrainingService/CreateTrainer", NULL, 0) ==
          (full ? 0 : 12), "profile-specific service registration");

    /* An open wait with no message cannot complete by itself. Module close
     * cancels it and drains its generated state without a timeout or polling
     * interval, then frees all public engine handles that remain unreleased. */
    SynurangCallOptions options = { sizeof(options), 0, 0, 0, UINT64_MAX };
    uint64_t wait = api->open(first.instance, VX_RPC_VX_SCHEDULER_SERVICE_WAIT_REQUEST, &options);
    CHECK(wait != 0, "open wait before owner shutdown");
    api->poll(first.instance, 64);
    vx_call_client_close(&first);
    response = vx_call_bytes(&second, VX_RPC_VX_PLATFORM_SERVICE_GET_PLATFORM_INFO, NULL, 0, &size);
    CHECK(response && size > 0, "closing one owner preserves the second owner");
    vx_call_free(&second, response);
    vx_call_client_close(&second);
    dlclose(library);
    if (!failures) printf("%s module ABI, owner isolation and shutdown passed\n", profile);
    return failures ? 1 : 0;
}
