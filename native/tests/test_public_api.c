#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "safetensors.h"

#include <errno.h>
#include <math.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

_Static_assert(VX_STATUS_OK == 0, "protobuf status contract");
_Static_assert(VX_STATUS_INVALID_ARGUMENT == -1, "protobuf status contract");
_Static_assert(VX_STATUS_OUT_OF_MEMORY == -2, "protobuf status contract");
_Static_assert(VX_STATUS_HANDLE_DISPOSED == -3, "protobuf status contract");
_Static_assert(VX_STATUS_IO_ERROR == -4, "protobuf status contract");
_Static_assert(VX_STATUS_INVALID_GRAPH == -5, "protobuf status contract");
_Static_assert(VX_STATUS_BACKEND_UNAVAILABLE == -6, "protobuf status contract");
_Static_assert(VX_STATUS_BACKEND_UNSUPPORTED == -7, "protobuf status contract");
_Static_assert(VX_STATUS_BACKEND_REQUIRED == -8, "protobuf status contract");
_Static_assert(VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN == -9,
               "protobuf status contract");
_Static_assert(VX_STATUS_EXECUTION_FAILED == -10, "protobuf status contract");
_Static_assert(VX_STATUS_RESULT_DISPOSED == -11, "protobuf status contract");
_Static_assert(VX_STATUS_DEVICE_LOST == -12, "protobuf status contract");
_Static_assert(VX_STATUS_ABI_UNSUPPORTED == -13, "protobuf status contract");
_Static_assert(VX_STATUS_NOT_FOUND == -14, "protobuf status contract");
_Static_assert(VX_STATUS_BUFFER_TOO_SMALL == -15, "protobuf status contract");
_Static_assert(VX_STATUS_INTERNAL == -16, "protobuf status contract");
_Static_assert(VX_STAGE_NONE == 0 && VX_STAGE_RUNTIME_CREATE == 1 &&
               VX_STAGE_MODEL_LOAD == 2 && VX_STAGE_COMPILE == 3 &&
               VX_STAGE_CONTEXT_CREATE == 4 && VX_STAGE_INPUT == 5 &&
               VX_STAGE_EXECUTE == 6 && VX_STAGE_READBACK == 7 &&
               VX_STAGE_CLOSE == 8 && VX_STAGE_DECODE == 9 &&
               VX_STAGE_ADAPTER == 10, "protobuf stage contract");
_Static_assert(VX_DTYPE_UNSPECIFIED == 0 && VX_DTYPE_BOOL == 1 &&
               VX_DTYPE_F4 == 2 && VX_DTYPE_F6_E2M3 == 3 &&
               VX_DTYPE_F6_E3M2 == 4 && VX_DTYPE_U8 == 5 &&
               VX_DTYPE_I8 == 6 && VX_DTYPE_F8_E5M2 == 7 &&
               VX_DTYPE_F8_E4M3 == 8 && VX_DTYPE_F8_E8M0 == 9 &&
               VX_DTYPE_F8_E4M3FNUZ == 10 &&
               VX_DTYPE_F8_E5M2FNUZ == 11 && VX_DTYPE_I16 == 12 &&
               VX_DTYPE_U16 == 13 && VX_DTYPE_F16 == 14 &&
               VX_DTYPE_BF16 == 15 && VX_DTYPE_I32 == 16 &&
               VX_DTYPE_U32 == 17 && VX_DTYPE_F32 == 18 &&
               VX_DTYPE_C64 == 19 && VX_DTYPE_F64 == 20 &&
               VX_DTYPE_I64 == 21 && VX_DTYPE_U64 == 22,
               "protobuf dtype contract");
_Static_assert(SAFETENSORS_DTYPE_UNKNOWN == VX_DTYPE_UNSPECIFIED &&
               SAFETENSORS_DTYPE_U8 == VX_DTYPE_U8 &&
               SAFETENSORS_DTYPE_I8 == VX_DTYPE_I8 &&
               SAFETENSORS_DTYPE_F16 == VX_DTYPE_F16 &&
               SAFETENSORS_DTYPE_I32 == VX_DTYPE_I32 &&
               SAFETENSORS_DTYPE_F32 == VX_DTYPE_F32 &&
               SAFETENSORS_DTYPE_U64 == VX_DTYPE_U64,
               "safetensors uses protobuf dtype contract");
_Static_assert(VX_BACKEND_PREFER == 0 && VX_BACKEND_REQUIRE == 1,
               "protobuf policy contract");
_Static_assert(VX_OPERATOR_FALLBACK_ALLOW == 0 &&
               VX_OPERATOR_FALLBACK_FORBID == 1,
               "protobuf fallback contract");
_Static_assert(VX_MEMORY_HOST == 0 && VX_MEMORY_DEVICE == 1,
               "protobuf memory contract");
_Static_assert(VX_DECODE_ROW_DISABLED == 0 && VX_DECODE_ROW_AUTO == 1 &&
               VX_DECODE_ROW_REQUIRED == 2, "protobuf decode contract");

typedef void (*VxPublicApiCpuExecuteHook)(void* user_data);
extern void vx_public_api_test_set_cpu_execute_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data);
extern VxStatus vx_public_api_test_evaluate_builtin_route(
    const char* selected_backend,
    const char* actual_route,
    VxOperatorFallback operator_fallback,
    VxReport* report);

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    size_t length = strlen(text);
    int ok = fwrite(text, 1, length, file) == length && fclose(file) == 0;
    return ok ? 0 : -1;
}

static int write_typed_weight(const char* path,
                              VxDataType dtype,
                              const void* value,
                              size_t byte_size) {
    const int shape[1] = {1};
    SafetensorsFile file;
    int status;
    if (safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(&file, "w", dtype, shape, 1, value,
                               byte_size) != 0) {
        safetensors_free(&file);
        return -1;
    }
    status = safetensors_save(path, &file);
    safetensors_free(&file);
    return status;
}

static int write_weight(const char* path, float value) {
    return write_typed_weight(path, SAFETENSORS_DTYPE_F32, &value,
                              sizeof(value));
}

static int read_text(const char* path, char* output, size_t capacity) {
    FILE* file;
    size_t count;
    int failed;
    if (!path || !output || capacity < 2u) return -1;
    file = fopen(path, "rb");
    if (!file) return -1;
    count = fread(output, 1, capacity - 1u, file);
    failed = ferror(file) || !feof(file);
    if (fclose(file) != 0) failed = 1;
    if (failed) return -1;
    output[count] = '\0';
    return 0;
}

static int closef(float a, float b) { return fabsf(a - b) < 1e-6f; }

static int create_cpu_context(VxRuntime* runtime,
                              const char* graph_path,
                              VxModel** out_model,
                              VxCompiledModel** out_compiled,
                              VxExecutionContext** out_context) {
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    source.graph_path = graph_path;
    if (vx_runtime_load_model(runtime, &source, out_model, &report) != VX_STATUS_OK)
        return -1;
    if (vx_model_compile(*out_model, &policy, out_compiled, &report) != VX_STATUS_OK)
        return -1;
    if (strcmp(report.backend, "cpu") != 0 || !report.runtime_id ||
        !report.model_id || !report.compiled_model_id || !report.graph_id ||
        !report.graph_revision || !report.weight_id || !report.weight_revision ||
        !report.adapter_id || !report.adapter_revision ||
        report.candidate_count != 1 || !report.route_attested ||
        report.operator_fallback_used || !report.route_evidence[0] ||
        report.compile_time_ms < 0.0) return -1;
    return vx_compiled_model_create_context(*out_compiled, &context_options,
                                             out_context, &report) == VX_STATUS_OK
        ? 0 : -1;
}

typedef struct ThreadCase {
    VxExecutionContext* context;
    float base;
    int ok;
} ThreadCase;

typedef struct CpuOverlapProbe {
    pthread_mutex_t mutex;
    pthread_cond_t condition;
    int entered;
    int timed_out;
} CpuOverlapProbe;

static void cpu_overlap_hook(void* opaque) {
    CpuOverlapProbe* probe = (CpuOverlapProbe*)opaque;
    struct timespec deadline;
    int wait_status = 0;
    pthread_mutex_lock(&probe->mutex);
    if (probe->entered < 2) probe->entered++;
    pthread_cond_broadcast(&probe->condition);
    timespec_get(&deadline, TIME_UTC);
    deadline.tv_sec += 3;
    while (probe->entered < 2 && !probe->timed_out && wait_status != ETIMEDOUT)
        wait_status = pthread_cond_timedwait(&probe->condition, &probe->mutex,
                                             &deadline);
    if (probe->entered < 2) {
        probe->timed_out = 1;
        pthread_cond_broadcast(&probe->condition);
    }
    pthread_mutex_unlock(&probe->mutex);
}

static void* run_thread_case(void* opaque) {
    ThreadCase* test = (ThreadCase*)opaque;
    test->ok = 1;
    for (int iteration = 0; iteration < 20; iteration++) {
        float a[2] = {test->base + iteration, test->base + iteration + 1.0f};
        float b[2] = {2.0f, -3.0f};
        float output[2] = {0};
        VxResult* result = NULL;
        if (vx_execution_context_set_input(test->context, "a", VX_DTYPE_F32,
                                           a, sizeof(a), NULL) != VX_STATUS_OK ||
            vx_execution_context_set_input(test->context, "b", VX_DTYPE_F32,
                                           b, sizeof(b), NULL) != VX_STATUS_OK ||
            vx_execution_context_execute(test->context, &result, NULL) != VX_STATUS_OK ||
            vx_result_read(result, "final", output, sizeof(output), NULL, NULL) !=
                VX_STATUS_OK ||
            !closef(output[0], a[0] + 2.0f * b[0]) ||
            !closef(output[1], a[1] + 2.0f * b[1])) {
            test->ok = 0;
            vx_result_release(result);
            return NULL;
        }
        vx_result_release(result);
    }
    return NULL;
}

typedef struct MockRuntime {
    int marker;
    char provider_name[VX_BACKEND_NAME_CAPACITY];
} MockRuntime;
typedef struct MockCompiled { int marker; } MockCompiled;
typedef struct MockContext { float input; } MockContext;

static int mock_runtime_destroyed;
static int mock_compiled_destroyed;
static int mock_context_destroyed;
static int mock_context_closed;
static int mock_adapter_selected;
static uint64_t mock_adapter_id;
static uint64_t mock_adapter_revision;
static int mock_adapter_package_was_null;
static char mock_adapter_version[96];
static char mock_adapter_package[96];

typedef enum MockOutputViolation {
    MOCK_OUTPUT_VALID = 0,
    MOCK_OUTPUT_MISSING,
    MOCK_OUTPUT_WRONG_NAME,
    MOCK_OUTPUT_WRONG_DTYPE,
    MOCK_OUTPUT_WRONG_RANK,
    MOCK_OUTPUT_WRONG_SHAPE,
    MOCK_OUTPUT_WRONG_BYTE_SIZE,
    MOCK_OUTPUT_DUPLICATE,
    MOCK_OUTPUT_F16,
    MOCK_OUTPUT_UNATTESTED_ROUTE
} MockOutputViolation;

static MockOutputViolation mock_output_violation;

static VxStatus mock_runtime_create(void* user_data,
                                    const VxRuntimeOptions* options,
                                    void** out,
                                    VxReport* report) {
    (void)options;
    (void)report;
    MockRuntime* runtime = (MockRuntime*)calloc(1, sizeof(*runtime));
    if (!runtime) return VX_STATUS_OUT_OF_MEMORY;
    runtime->marker = 11;
    snprintf(runtime->provider_name, sizeof(runtime->provider_name), "%s",
             user_data ? (const char*)user_data : "");
    *out = runtime;
    return VX_STATUS_OK;
}

static void mock_runtime_destroy(void* instance) {
    mock_runtime_destroyed++;
    free(instance);
}

static VxStatus mock_compile(void* runtime_instance,
                             const VxModelSource* source,
                             const VxBackendPolicy* policy,
                             void** out,
                             VxReport* report) {
    MockRuntime* runtime = (MockRuntime*)runtime_instance;
    int selected_present = 0;
    if (!runtime || !source || !source->graph_path || !policy ||
        !policy->backends || !policy->backend_count)
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < policy->backend_count; index++)
        if (!strcmp(policy->backends[index], runtime->provider_name))
            selected_present = 1;
    if (!selected_present) return VX_STATUS_INVALID_ARGUMENT;
    MockCompiled* compiled = (MockCompiled*)calloc(1, sizeof(*compiled));
    if (!compiled) return VX_STATUS_OUT_OF_MEMORY;
    compiled->marker = 22;
    *out = compiled;
    if (!strcmp(runtime->provider_name, "test-provider")) {
        report->route_attested = 1;
        report->operator_fallback_used = 0;
        snprintf(report->device, sizeof(report->device), "%s", "mock-device-0");
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=test-provider;nodes=2;route=test-provider-all");
        snprintf(report->fallback_evidence, sizeof(report->fallback_evidence),
                 "%s", "operator=none");
    }
    return VX_STATUS_OK;
}

static void mock_compiled_destroy(void* instance) {
    mock_compiled_destroyed++;
    free(instance);
}

static VxStatus mock_context_create(void* compiled_instance,
                                    const VxContextOptions* options,
                                    void** out,
                                    VxReport* report) {
    (void)options;
    (void)report;
    if (!compiled_instance) return VX_STATUS_INVALID_ARGUMENT;
    MockContext* context = (MockContext*)calloc(1, sizeof(*context));
    if (!context) return VX_STATUS_OUT_OF_MEMORY;
    *out = context;
    return VX_STATUS_OK;
}

static VxStatus mock_context_set_input(void* instance,
                                       const char* name,
                                       VxDataType dtype,
                                       const void* data,
                                       size_t byte_size,
                                       VxReport* report) {
    (void)report;
    if (!instance || !name || strcmp(name, "value") || dtype != VX_DTYPE_F32 ||
        !data || byte_size != sizeof(float)) return VX_STATUS_INVALID_ARGUMENT;
    memcpy(&((MockContext*)instance)->input, data, sizeof(float));
    return VX_STATUS_OK;
}

static VxStatus mock_context_execute(void* instance,
                                     const VxBackendOutputSink* sink,
                                     VxReport* report) {
    if (!instance || !sink || sink->struct_size < sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
    float output[2] = {((MockContext*)instance)->input * 2.0f, -1.0f};
    int64_t shape[1] = {1};
    VxStatus status;
    switch (mock_output_violation) {
        case MOCK_OUTPUT_MISSING:
            return VX_STATUS_OK;
        case MOCK_OUTPUT_WRONG_NAME:
            return sink->write(sink->user_data, "other", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output[0]));
        case MOCK_OUTPUT_WRONG_DTYPE:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_I32,
                               shape, 1, output, sizeof(output[0]));
        case MOCK_OUTPUT_WRONG_RANK:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               NULL, 0, output, sizeof(output[0]));
        case MOCK_OUTPUT_WRONG_SHAPE:
            shape[0] = 2;
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output));
        case MOCK_OUTPUT_WRONG_BYTE_SIZE:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output));
        case MOCK_OUTPUT_DUPLICATE:
            status = sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                                 shape, 1, output, sizeof(output[0]));
            return status == VX_STATUS_OK
                ? sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                              shape, 1, output, sizeof(output[0]))
                : status;
        case MOCK_OUTPUT_F16:
            return sink->write(sink->user_data, "mock-out", (VxDataType)4,
                               shape, 1, output, 2u);
        case MOCK_OUTPUT_UNATTESTED_ROUTE:
            report->route_attested = 0;
            report->route_evidence[0] = '\0';
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output[0]));
        case MOCK_OUTPUT_VALID:
        default:
            return sink->write(sink->user_data, "mock-out", VX_DTYPE_F32,
                               shape, 1, output, sizeof(output[0]));
    }
}

static VxStatus mock_context_select_adapter(void* instance,
                                            uint64_t adapter_id,
                                            uint64_t adapter_revision,
                                            const char* package_path,
                                            const char* version_name,
                                            VxReport* report) {
    (void)report;
    if (!instance || !adapter_id || !adapter_revision)
        return VX_STATUS_INVALID_ARGUMENT;
    mock_adapter_selected++;
    mock_adapter_id = adapter_id;
    mock_adapter_revision = adapter_revision;
    mock_adapter_package_was_null = package_path == NULL;
    mock_adapter_package[0] = '\0';
    if (package_path && read_text(package_path, mock_adapter_package,
                                  sizeof(mock_adapter_package)) != 0)
        return VX_STATUS_IO_ERROR;
    snprintf(mock_adapter_version, sizeof(mock_adapter_version), "%s",
             version_name ? version_name : "");
    return VX_STATUS_OK;
}

static VxStatus mock_context_close(void* instance, VxReport* report) {
    (void)report;
    if (!instance) return VX_STATUS_INVALID_ARGUMENT;
    mock_context_closed++;
    return VX_STATUS_OK;
}

static void mock_context_destroy(void* instance) {
    mock_context_destroyed++;
    free(instance);
}

static int test_provider(const char* readable_graph) {
    VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = "test-provider",
        .user_data = (void*)"test-provider",
        .runtime_create = mock_runtime_create,
        .runtime_destroy = mock_runtime_destroy,
        .compile = mock_compile,
        .compiled_destroy = mock_compiled_destroy,
        .context_create = mock_context_create,
        .context_set_input = mock_context_set_input,
        .context_execute = mock_context_execute,
        .context_select_adapter = mock_context_select_adapter,
        .context_close = mock_context_close,
        .context_destroy = mock_context_destroy,
    };
    VxBackendProvider invalid = provider;
    VxBackendProvider unattested = provider;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    VxAdapterSource adapter_source = VX_ADAPTER_SOURCE_INIT;
    VxAdapterRevision first_adapter_revision = VX_ADAPTER_REVISION_INIT;
    VxAdapterRevision second_adapter_revision = VX_ADAPTER_REVISION_INIT;
    float input = 7.5f;
    float output = 0.0f;
    const char* required_backend[1];
    const char* adapter_path = "/tmp/volvox-public-api-provider-adapter.bin";

    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    invalid.struct_size--;
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    invalid = provider;
    invalid.abi_version++;
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_ABI_UNSUPPORTED);
    CHECK(!strcmp(report.reason, "ABI_UNSUPPORTED"));
    invalid = provider;
    invalid.name = "Invalid_Name";
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    invalid = provider;
    invalid.name = "vulkan";
    CHECK(vx_runtime_register_provider(runtime, &invalid, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(vx_runtime_register_provider(runtime, &provider, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_register_provider(runtime, &provider, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    unattested.name = "unattested-provider";
    unattested.user_data = (void*)"unattested-provider";
    CHECK(vx_runtime_register_provider(runtime, &unattested, &report) ==
          VX_STATUS_OK);
    source.graph_path = readable_graph;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    required_backend[0] = "unattested-provider";
    policy.backends = required_backend;
    policy.backend_count = 1;
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN);
    CHECK(compiled == NULL &&
          !strcmp(report.reason, "OPERATOR_FALLBACK_FORBIDDEN"));
    policy.operator_fallback = VX_OPERATOR_FALLBACK_ALLOW;
    required_backend[0] = "test-provider";
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(!strcmp(report.backend, "test-provider"));
    CHECK(!strcmp(report.device, "mock-device-0") && report.route_attested &&
          !report.operator_fallback_used && report.route_evidence[0] &&
          report.operator_fallback == VX_OPERATOR_FALLBACK_ALLOW);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &context, &report) == VX_STATUS_OK);
    {
        VxAffineQuantization quantization = VX_AFFINE_QUANTIZATION_INIT;
        quantization.defined = 1;
        quantization.scale = 3.0f;
        quantization.zero_point = 17;
        CHECK(vx_execution_context_input_affine_quantization(
                  context, "value", &quantization, &report) ==
              VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(!quantization.defined && quantization.scale == 0.0f &&
              quantization.zero_point == 0 &&
              !strcmp(report.reason,
                      "INPUT_QUANTIZATION_INTROSPECTION_UNSUPPORTED"));
    }
    CHECK(write_text(adapter_path, "adapter-one") == 0);
    adapter_source.adapter_name = "provider-route";
    adapter_source.version_name = "provider-v1";
    adapter_source.package_path = adapter_path;
    CHECK(vx_model_publish_adapter(model, &adapter_source,
                                   &first_adapter_revision,
                                   &report) == VX_STATUS_OK);
    CHECK(write_text(adapter_path, "changed-one") == 0);
    CHECK(remove(adapter_path) == 0);
    CHECK(vx_execution_context_select_adapter(context, &first_adapter_revision,
                                               &report) == VX_STATUS_OK);
    CHECK(mock_adapter_selected == 1 &&
          mock_adapter_id == first_adapter_revision.adapter_id &&
          mock_adapter_revision == first_adapter_revision.adapter_revision &&
          !mock_adapter_package_was_null &&
          !strcmp(mock_adapter_package, "adapter-one") &&
          !strcmp(mock_adapter_version, "provider-v1"));
    CHECK(write_text(adapter_path, "adapter-two") == 0);
    adapter_source.version_name = "provider-v2";
    CHECK(vx_model_publish_adapter(model, &adapter_source,
                                   &second_adapter_revision,
                                   &report) == VX_STATUS_OK);
    CHECK(second_adapter_revision.adapter_id == first_adapter_revision.adapter_id &&
          second_adapter_revision.adapter_revision ==
              first_adapter_revision.adapter_revision + 1u);
    CHECK(write_text(adapter_path, "changed-two") == 0);
    CHECK(remove(adapter_path) == 0);
    CHECK(vx_execution_context_rebind_adapter(context, &report) == VX_STATUS_OK);
    CHECK(mock_adapter_selected == 2 &&
          !strcmp(mock_adapter_package, "adapter-two") &&
          !strcmp(mock_adapter_version, "provider-v2"));
    CHECK(vx_execution_context_select_adapter(context, &first_adapter_revision,
                                               &report) == VX_STATUS_OK);
    CHECK(mock_adapter_selected == 3 &&
          !strcmp(mock_adapter_package, "adapter-one") &&
          !strcmp(mock_adapter_version, "provider-v1"));
    CHECK(vx_execution_context_set_input(context, "value", (VxDataType)4,
                                         &input, 2u, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(vx_execution_context_set_input(context, "value", VX_DTYPE_F32,
                                         &input, sizeof(input), &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute(context, &result, &report) == VX_STATUS_OK);
    CHECK(report.context_id && report.execution_id && report.route_attested &&
          !report.operator_fallback_used && report.execution_time_ms >= 0.0);
    CHECK(vx_result_output_count(result) == 1);
    {
        VxResult* unsupported = NULL;
        CHECK(vx_execution_context_execute_prefix(
                  context, 1, &unsupported, &report) ==
              VX_STATUS_BACKEND_UNSUPPORTED);
        CHECK(unsupported == NULL &&
              !strcmp(report.reason, "BACKEND_UNSUPPORTED"));
    }
    {
        const MockOutputViolation violations[] = {
            MOCK_OUTPUT_MISSING,
            MOCK_OUTPUT_WRONG_NAME,
            MOCK_OUTPUT_WRONG_DTYPE,
            MOCK_OUTPUT_WRONG_RANK,
            MOCK_OUTPUT_WRONG_SHAPE,
            MOCK_OUTPUT_WRONG_BYTE_SIZE,
            MOCK_OUTPUT_DUPLICATE,
            MOCK_OUTPUT_F16,
            MOCK_OUTPUT_UNATTESTED_ROUTE,
        };
        for (size_t index = 0;
             index < sizeof(violations) / sizeof(violations[0]); index++) {
            VxResult* rejected = NULL;
            mock_output_violation = violations[index];
            CHECK(vx_execution_context_execute(context, &rejected, &report) ==
                  VX_STATUS_EXECUTION_FAILED);
            CHECK(rejected == NULL && report.status == VX_STATUS_EXECUTION_FAILED &&
                  !strcmp(report.reason, "EXECUTION_FAILED"));
        }
        mock_output_violation = MOCK_OUTPUT_VALID;
        {
            VxResult* recovered = NULL;
            CHECK(vx_execution_context_execute(context, &recovered, &report) ==
                  VX_STATUS_OK);
            vx_result_release(recovered);
        }
    }
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_register_provider(runtime, &provider, &report) ==
          VX_STATUS_HANDLE_DISPOSED);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    vx_runtime_release(runtime);
    CHECK(vx_result_read(result, "mock-out", &output, sizeof(output), NULL, &report) ==
          VX_STATUS_OK);
    CHECK(closef(output, 15.0f));
    vx_result_release(result);
    CHECK(mock_context_closed == 1);
    CHECK(mock_context_destroyed == 1);
    CHECK(mock_compiled_destroyed == 2);
    CHECK(mock_runtime_destroyed == 2);
    remove(adapter_path);
    return 0;
}

static int test_model_source_snapshots(void) {
    const char* graph_path = "/tmp/volvox-public-api-snapshot.graph.json";
    const char* weight_path =
        "/tmp/volvox-public-api-snapshot-weights.safetensors";
    const char* original_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"x\",\"b\":\"w\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"y\"]}";
    const char* replacement_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"different\"},"
        "\"outputs_shape\":{\"out\":[1]}}],"
        "\"outputs\":[\"different\"]}";
    const char* weights[] = {weight_path};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    float input = 3.0f;
    float output = 0.0f;

    CHECK(write_text(graph_path, original_graph) == 0);
    CHECK(write_weight(weight_path, 2.0f) == 0);
    source.graph_path = graph_path;
    source.weight_paths = weights;
    source.weight_path_count = 1;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);

    /* Compilation must consume the accepted graph and weight bytes, not these
     * caller-owned paths after load_model returns. */
    CHECK(write_text(graph_path, replacement_graph) == 0);
    CHECK(write_weight(weight_path, 99.0f) == 0);
    CHECK(remove(graph_path) == 0);
    CHECK(remove(weight_path) == 0);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_set_input(context, "x", VX_DTYPE_F32,
                                         &input, sizeof(input), &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_execute(context, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "y", &output, sizeof(output), NULL, &report) ==
          VX_STATUS_OK);
    CHECK(closef(output, 5.0f));

    vx_result_release(result);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    remove(graph_path);
    remove(weight_path);
    return 0;
}

static int test_declared_output_descriptors(void) {
    const char* direct_graph_path =
        "/tmp/volvox-public-api-direct-output.graph.json";
    const char* weight_graph_path =
        "/tmp/volvox-public-api-weight-output.graph.json";
    const char* f16_graph_path =
        "/tmp/volvox-public-api-f16-storage.graph.json";
    const char* weight_path =
        "/tmp/volvox-public-api-output-weight.safetensors";
    const char* f16_weight_path =
        "/tmp/volvox-public-api-f16-weight.safetensors";
    const char* direct_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"direct\":{\"shape\":[2],\"dtype\":\"int32\"}},"
        "\"nodes\":[],\"outputs\":[\"direct\"]}";
    const char* weight_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{},"
        "\"nodes\":[],\"outputs\":[\"w\"]}";
    const char* f16_compute_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"x\",\"b\":\"w\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1]}}],\"outputs\":[\"y\"]}";
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    uint16_t f16_two = UINT16_C(0x4000);
    float weight_value = 6.25f;

    CHECK(write_text(direct_graph_path, direct_graph) == 0);
    CHECK(write_text(weight_graph_path, weight_graph) == 0);
    CHECK(write_text(f16_graph_path, f16_compute_graph) == 0);
    CHECK(write_weight(weight_path, weight_value) == 0);
    CHECK(write_typed_weight(f16_weight_path, SAFETENSORS_DTYPE_F16,
                             &f16_two, sizeof(f16_two)) == 0);
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);

    {
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* model = NULL;
        VxCompiledModel* compiled = NULL;
        VxExecutionContext* context = NULL;
        VxResult* result = NULL;
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        const int32_t input[2] = {17, -9};
        int32_t output[2] = {0};
        source.graph_path = direct_graph_path;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_set_input(context, "direct", VX_DTYPE_I32,
                                             input, sizeof(input), &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context, &result, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_output_info(result, 0, &info, &report) == VX_STATUS_OK);
        CHECK(!strcmp(info.name, "direct") && info.dtype == VX_DTYPE_I32 &&
              info.rank == 1 && info.shape[0] == 2 &&
              info.byte_size == sizeof(input));
        CHECK(vx_result_read(result, "direct", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(input, output, sizeof(input)));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
    }

    {
        const char* weights[] = {weight_path};
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* model = NULL;
        VxCompiledModel* compiled = NULL;
        VxExecutionContext* context = NULL;
        VxResult* result = NULL;
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        float output = 0.0f;
        int32_t replacement = 123;
        source.graph_path = weight_graph_path;
        source.weight_paths = weights;
        source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(write_typed_weight(weight_path, SAFETENSORS_DTYPE_I32,
                                 &replacement, sizeof(replacement)) == 0);
        CHECK(remove(weight_path) == 0);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context, &result, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_output_info(result, 0, &info, &report) == VX_STATUS_OK);
        CHECK(!strcmp(info.name, "w") && info.dtype == VX_DTYPE_F32 &&
              info.rank == 1 && info.shape[0] == 1 &&
              info.byte_size == sizeof(output));
        CHECK(vx_result_read(result, "w", &output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output, weight_value));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
    }

    {
        const char* weights[] = {f16_weight_path};
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* model = NULL;
        VxCompiledModel* compiled = NULL;
        VxExecutionContext* context = NULL;
        VxResult* result = NULL;
        float input = 3.0f;
        float output = 0.0f;
        source.graph_path = f16_graph_path;
        source.weight_paths = weights;
        source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_set_input(context, "x", VX_DTYPE_F32,
                                             &input, sizeof(input), &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context, &result, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_read(result, "y", &output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output, 5.0f));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);

        source.graph_path = weight_graph_path;
        model = NULL;
        compiled = NULL;
        context = NULL;
        result = NULL;
        output = 0.0f;
        CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
              VX_STATUS_OK);
        CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                                &context, &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context, &result, &report) ==
              VX_STATUS_OK);
        {
            VxTensorInfo info = VX_TENSOR_INFO_INIT;
            CHECK(vx_result_output_info(result, 0, &info, &report) ==
                  VX_STATUS_OK);
            CHECK(info.dtype == VX_DTYPE_F32 && info.rank == 1 &&
                  info.shape[0] == 1 && info.byte_size == sizeof(float));
        }
        CHECK(vx_result_read(result, "w", &output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output, 2.0f));
        vx_result_release(result);
        CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
        vx_execution_context_release(context);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
    }

    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_runtime_release(runtime);
    remove(direct_graph_path);
    remove(weight_graph_path);
    remove(f16_graph_path);
    remove(weight_path);
    remove(f16_weight_path);
    return 0;
}

int main(void) {
    const char* graph_a = "/tmp/volvox-public-api-a.graph.json";
    const char* graph_b = "/tmp/volvox-public-api-b.graph.json";
    const char* provider_graph_path =
        "/tmp/volvox-public-api-provider.graph.json";
    const char* invalid_graph_path = "/tmp/volvox-public-api-invalid.graph.json";
    const char* add_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"a\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[2]}},"
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"sum\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"final\"},\"outputs_shape\":{\"out\":[2]}}],"
        "\"outputs\":[\"sum\",\"final\"]}";
    const char* identity_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[3]}}],\"outputs\":[\"y\"]}";
    const char* provider_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"value\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"value\"},"
        "\"outputs\":{\"out\":\"mock-out\"},"
        "\"outputs_shape\":{\"out\":[1]}}],"
        "\"outputs\":[\"mock-out\"]}";
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model_a = NULL;
    VxCompiledModel* compiled_a = NULL;
    VxExecutionContext* context_a = NULL;
    VxExecutionContext* context_b = NULL;
    VxModel* model_b = NULL;
    VxCompiledModel* compiled_b = NULL;
    VxExecutionContext* context_c = NULL;
    VxExecutionContext* decode_a = NULL;
    VxExecutionContext* decode_b = NULL;
    VxResult* retained = NULL;
    VxResult* stable_first = NULL;
    pthread_t first_thread;
    pthread_t second_thread;
    ThreadCase first = {0};
    ThreadCase second = {0};
    CpuOverlapProbe overlap = {0};

    CHECK(write_text(graph_a, add_graph) == 0);
    CHECK(write_text(graph_b, identity_graph) == 0);
    CHECK(write_text(provider_graph_path, provider_graph) == 0);
    CHECK(write_text(invalid_graph_path, "{\"inputs\":{},\"nodes\":[],\"outputs\":[]}") == 0);
    runtime_options.cpu_threads = 2;
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    {
        VxModelSource noncanonical = VX_MODEL_SOURCE_INIT;
        VxModel* rejected = NULL;
        noncanonical.graph_path = "/tmp/config.json";
        CHECK(vx_runtime_load_model(runtime, &noncanonical, &rejected, &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL && !strcmp(report.reason, "INVALID_GRAPH_PATH"));
    }
    {
        VxModelSource missing = VX_MODEL_SOURCE_INIT;
        VxModel* rejected = NULL;
        missing.graph_path = "/tmp/volvox-public-api-does-not-exist.graph.json";
        CHECK(vx_runtime_load_model(runtime, &missing, &rejected, &report) ==
              VX_STATUS_IO_ERROR);
        CHECK(report.status == VX_STATUS_IO_ERROR && report.stage == VX_STAGE_MODEL_LOAD);
    }
    {
        VxModelSource invalid_source = VX_MODEL_SOURCE_INIT;
        VxModel* invalid_model = NULL;
        const char* unreadable_weights[] = {
            "/tmp/volvox-public-api-does-not-exist.safetensors"
        };
        invalid_source.graph_path = invalid_graph_path;
        invalid_source.weight_paths = unreadable_weights;
        invalid_source.weight_path_count = 1;
        CHECK(vx_runtime_load_model(runtime, &invalid_source, &invalid_model, &report) ==
              VX_STATUS_INVALID_GRAPH);
        CHECK(invalid_model == NULL && report.stage == VX_STAGE_MODEL_LOAD &&
              !strcmp(report.reason, "INVALID_GRAPH_CONTRACT"));
    }
    {
        const char* invalid_contracts[] = {
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":{\"out\":\"y\"}}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":[]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":[\"y\",\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":[],"
                "\"outputs\":[\"\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{},\"nodes\":["
                "{\"op\":\"Identity\"}],\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"F32\"}},"
                "\"nodes\":[],\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1]}},\"nodes\":[],"
                "\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float16\"}},"
                "\"nodes\":[],\"outputs\":[\"x\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"Identity\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]},"
                "\"outputs_dtype\":{\"out\":\"float16\"}}],"
                "\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"Identity\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]},"
                "\"params\":{\"backend\":{\"options\":["
                "{\"zero_point\":0}]}}}],\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"RotaryEmbedding\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]}}],"
                "\"outputs\":[\"y\"]}",
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
                "\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
                "\"nodes\":[{\"opType\":\"DefinitelyUnknown\","
                "\"inputs\":{\"input\":\"x\"},"
                "\"outputs\":{\"out\":\"y\"},"
                "\"outputs_shape\":{\"out\":[1]}}],"
                "\"outputs\":[\"y\"]}",
        };
        const char* unreadable_weights[] = {
            "/tmp/volvox-public-api-does-not-exist.safetensors"
        };
        for (size_t index = 0;
             index < sizeof(invalid_contracts) / sizeof(invalid_contracts[0]);
             index++) {
            VxModelSource invalid_source = VX_MODEL_SOURCE_INIT;
            VxModel* invalid_model = NULL;
            CHECK(write_text(invalid_graph_path, invalid_contracts[index]) == 0);
            invalid_source.graph_path = invalid_graph_path;
            invalid_source.weight_paths = unreadable_weights;
            invalid_source.weight_path_count = 1;
            CHECK(vx_runtime_load_model(runtime, &invalid_source, &invalid_model,
                                        &report) == VX_STATUS_INVALID_GRAPH);
            CHECK(invalid_model == NULL && report.stage == VX_STAGE_MODEL_LOAD);
        }
    }
    {
        const char* raw_byte_graph =
            "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
            "\"x\":{\"shape\":[1],\"dtype\":\"int8\"}},"
            "\"nodes\":[],\"outputs\":[\"x\"]}";
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* raw_model = NULL;
        CHECK(write_text(invalid_graph_path, raw_byte_graph) == 0);
        source.graph_path = invalid_graph_path;
        CHECK(vx_runtime_load_model(runtime, &source, &raw_model, &report) ==
              VX_STATUS_OK);
        CHECK(raw_model != NULL);
        vx_model_release(raw_model);
    }
    CHECK(create_cpu_context(runtime, graph_a, &model_a, &compiled_a, &context_a) == 0);
    {
        VxAdapterSource adapter_source = VX_ADAPTER_SOURCE_INIT;
        VxAdapterRevision first_revision = VX_ADAPTER_REVISION_INIT;
        VxAdapterRevision second_revision = VX_ADAPTER_REVISION_INIT;
        VxAdapterRevision missing_revision = VX_ADAPTER_REVISION_INIT;
        VxRevisionInfo model_revision = VX_REVISION_INFO_INIT;
        VxReport pinned = VX_REPORT_INIT;
        float input[2] = {0};
        CHECK(vx_compiled_model_report(compiled_a, &pinned) == VX_STATUS_OK);
        uint64_t base_adapter_id = pinned.adapter_id;
        uint64_t base_adapter_revision = pinned.adapter_revision;
        adapter_source.adapter_name = "request-lora";
        CHECK(vx_model_publish_adapter(model_a, &adapter_source,
                                       &first_revision, &report) == VX_STATUS_OK);
        CHECK(first_revision.adapter_id && first_revision.adapter_revision == 1);
        CHECK(vx_model_revision_info(model_a, &model_revision, &report) ==
              VX_STATUS_OK);
        CHECK(model_revision.adapter_id == first_revision.adapter_id &&
              model_revision.adapter_revision == first_revision.adapter_revision);
        CHECK(vx_compiled_model_report(compiled_a, &pinned) == VX_STATUS_OK);
        CHECK(pinned.adapter_id == base_adapter_id &&
              pinned.adapter_revision == base_adapter_revision);
        CHECK(vx_execution_context_rebind_adapter(context_a, &report) ==
              VX_STATUS_OK);
        CHECK(report.adapter_id == first_revision.adapter_id &&
              report.adapter_revision == first_revision.adapter_revision);

        CHECK(vx_model_publish_adapter(model_a, &adapter_source,
                                       &second_revision, &report) == VX_STATUS_OK);
        CHECK(second_revision.adapter_id == first_revision.adapter_id &&
              second_revision.adapter_revision == first_revision.adapter_revision + 1u);
        CHECK(vx_execution_context_set_input(context_a, "a", VX_DTYPE_F32,
                                             input, sizeof(input), &report) ==
              VX_STATUS_OK);
        CHECK(report.adapter_revision == first_revision.adapter_revision);
        CHECK(vx_execution_context_select_adapter(context_a, &first_revision,
                                                   &report) == VX_STATUS_OK);
        CHECK(report.adapter_revision == first_revision.adapter_revision);
        CHECK(vx_execution_context_rebind_adapter(context_a, &report) ==
              VX_STATUS_OK);
        CHECK(report.adapter_revision == second_revision.adapter_revision);
        missing_revision.adapter_id = second_revision.adapter_id;
        missing_revision.adapter_revision = second_revision.adapter_revision + 100u;
        CHECK(vx_execution_context_select_adapter(context_a, &missing_revision,
                                                   &report) == VX_STATUS_NOT_FOUND);
        CHECK(report.adapter_revision == second_revision.adapter_revision);
    }
    {
        VxBackendPolicy preferred = VX_BACKEND_POLICY_INIT;
        VxCompiledModel* fallback = NULL;
        const char* ordered[] = { "missing-provider", "cpu" };
        preferred.backends = ordered;
        preferred.backend_count = 2;
        CHECK(vx_model_compile(model_a, &preferred, &fallback, &report) == VX_STATUS_OK);
        CHECK(fallback && !strcmp(report.backend, "cpu") &&
              report.tier_fallback_used && report.candidate_count == 2 &&
              strstr(report.candidate_outcomes, "missing-provider:unavailable") &&
              strstr(report.fallback_evidence, "->cpu"));
        vx_compiled_model_release(fallback);
    }
    {
        VxBackendPolicy unavailable = VX_BACKEND_POLICY_INIT;
        VxCompiledModel* rejected = NULL;
        VxReport compiled_report = VX_REPORT_INIT;
        const char* required[] = { "missing-provider" };
        CHECK(vx_compiled_model_report(compiled_a, &compiled_report) == VX_STATUS_OK);
        unavailable.mode = VX_BACKEND_REQUIRE;
        unavailable.backends = required;
        unavailable.backend_count = 1;
        CHECK(vx_model_compile(model_a, &unavailable, &rejected, &report) ==
              VX_STATUS_BACKEND_REQUIRED);
        CHECK(rejected == NULL && report.status == VX_STATUS_BACKEND_REQUIRED &&
              !strcmp(report.reason, "BACKEND_REQUIRED") &&
              report.candidate_count == 1 && report.model_id ==
              compiled_report.model_id);
    }
    {
        VxReport route = VX_REPORT_INIT;
        CHECK(vx_public_api_test_evaluate_builtin_route(
                  "cuda", "cpu", VX_OPERATOR_FALLBACK_ALLOW, &route) ==
              VX_STATUS_OK);
        CHECK(route.route_attested && route.operator_fallback_used &&
              strstr(route.route_evidence, "selected=0;fallback=1") &&
              strstr(route.fallback_evidence, "operator=used"));
        route = (VxReport)VX_REPORT_INIT;
        CHECK(vx_public_api_test_evaluate_builtin_route(
                  "cuda", "cpu", VX_OPERATOR_FALLBACK_FORBID, &route) ==
              VX_STATUS_OPERATOR_FALLBACK_FORBIDDEN);
        CHECK(route.operator_fallback_used &&
              !strcmp(route.reason, "OPERATOR_FALLBACK_FORBIDDEN") &&
              strstr(route.offending_node, "op=Identity"));
        route = (VxReport)VX_REPORT_INIT;
        CHECK(vx_public_api_test_evaluate_builtin_route(
                  "cuda", "cuda-linear", VX_OPERATOR_FALLBACK_FORBID,
                  &route) == VX_STATUS_OK);
        CHECK(route.route_attested && !route.operator_fallback_used);
    }
    {
        VxBackendPolicy builtin = VX_BACKEND_POLICY_INIT;
        VxCompiledModel* selected = NULL;
        const char* required[] = { "vulkan" };
        builtin.mode = VX_BACKEND_REQUIRE;
        builtin.backends = required;
        builtin.backend_count = 1;
        CHECK(vx_model_compile(model_a, &builtin, &selected, &report) ==
              VX_STATUS_BACKEND_UNAVAILABLE);
        CHECK(selected == NULL && !strcmp(report.backend, "vulkan") &&
              !strcmp(report.reason, "BACKEND_UNAVAILABLE"));
        {
            const char* preferred[] = { "vulkan", "cpu" };
            builtin.mode = VX_BACKEND_PREFER;
            builtin.backends = preferred;
            builtin.backend_count = 2;
            CHECK(vx_model_compile(model_a, &builtin, &selected, &report) ==
                  VX_STATUS_OK);
            CHECK(selected && !strcmp(report.backend, "cpu") &&
                  report.tier_fallback_used &&
                  strstr(report.candidate_outcomes, "vulkan:unavailable") &&
                  strstr(report.candidate_outcomes, "cpu:selected"));
            vx_compiled_model_release(selected);
        }
    }
    {
        VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
        CHECK(vx_compiled_model_create_context(compiled_a, &options,
                                                &context_b, &report) == VX_STATUS_OK);
    }
    CHECK(create_cpu_context(runtime, graph_b, &model_b, &compiled_b, &context_c) == 0);
    {
        VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
        options.decode_row_mode = VX_DECODE_ROW_AUTO;
        options.require_incremental = 1;
        CHECK(vx_compiled_model_create_context(compiled_b, &options,
                                                &decode_a, &report) == VX_STATUS_OK);
        CHECK(strstr(report.decode_state, "enabled=1") &&
              strstr(report.decode_state, "seeded=0"));
        CHECK(vx_compiled_model_create_context(compiled_b, &options,
                                                &decode_b, &report) == VX_STATUS_OK);
    }
    CHECK(vx_execution_context_input_count(context_a) == 2);
    CHECK(vx_execution_context_input_count(context_c) == 1);
    {
        VxReport report_a = VX_REPORT_INIT;
        VxReport report_b = VX_REPORT_INIT;
        VxTensorInfo info = VX_TENSOR_INFO_INIT;
        uint64_t context_a_id;
        CHECK(vx_compiled_model_report(compiled_a, &report_a) == VX_STATUS_OK);
        CHECK(vx_compiled_model_report(compiled_b, &report_b) == VX_STATUS_OK);
        CHECK(report_a.runtime_id == report_b.runtime_id &&
              report_a.model_id != report_b.model_id &&
              report_a.compiled_model_id != report_b.compiled_model_id &&
              report_a.graph_id != report_b.graph_id &&
              report_a.graph_revision != report_b.graph_revision &&
              report_a.weight_id != report_b.weight_id &&
              report_a.weight_revision == report_b.weight_revision);
        CHECK(vx_execution_context_input_info(context_a, 0, &info, &report) ==
              VX_STATUS_OK);
        {
            VxAffineQuantization quantization =
                VX_AFFINE_QUANTIZATION_INIT;
            CHECK(vx_execution_context_input_affine_quantization(
                      context_a, "a", &quantization, &report) == VX_STATUS_OK);
            CHECK(!quantization.defined && quantization.scale == 0.0f &&
                  quantization.zero_point == 0);
        }
        context_a_id = report.context_id;
        info = (VxTensorInfo)VX_TENSOR_INFO_INIT;
        CHECK(vx_execution_context_input_info(context_b, 0, &info, &report) ==
              VX_STATUS_OK);
        CHECK(context_a_id && report.context_id &&
              context_a_id != report.context_id);
    }

    {
        const float a[2] = {1.0f, 2.0f};
        const float b[2] = {3.0f, 4.0f};
        float sum[2] = {0};
        float too_small = 0.0f;
        size_t required = 0;
        uint64_t execution_adapter_revision;
        VxTensorInfo first_output = VX_TENSOR_INFO_INIT;
        VxTensorInfo second_output = VX_TENSOR_INFO_INIT;
        VxResult* later = NULL;
        CHECK(vx_execution_context_set_input(context_a, "a", VX_DTYPE_F32,
                                             a, sizeof(a), &report) == VX_STATUS_OK);
        CHECK(vx_execution_context_set_input(context_a, "b", VX_DTYPE_F32,
                                             b, sizeof(b), &report) == VX_STATUS_OK);
        CHECK(vx_execution_context_execute(context_a, &stable_first, &report) == VX_STATUS_OK);
        CHECK(report.context_id && report.execution_id && report.result_bytes ==
              2u * sizeof(sum) && report.route_attested &&
              !report.operator_fallback_used && report.route_evidence[0] &&
              report.execution_time_ms >= 0.0);
        execution_adapter_revision = report.adapter_revision;
        CHECK(vx_result_output_count(stable_first) == 2);
        CHECK(vx_result_output_info(stable_first, 0, &first_output, &report) == VX_STATUS_OK);
        CHECK(vx_result_output_info(stable_first, 1, &second_output, &report) == VX_STATUS_OK);
        CHECK(!strcmp(first_output.name, "sum") && !strcmp(second_output.name, "final"));
        CHECK(vx_result_read(stable_first, "sum", &too_small, sizeof(too_small),
                             &required, &report) == VX_STATUS_BUFFER_TOO_SMALL);
        CHECK(required == sizeof(sum));
        CHECK(vx_execution_context_execute(context_a, &later, &report) == VX_STATUS_OK);
        vx_result_release(later);
        {
            VxAdapterSource next_adapter = VX_ADAPTER_SOURCE_INIT;
            VxAdapterRevision next_revision = VX_ADAPTER_REVISION_INIT;
            next_adapter.adapter_name = "request-lora";
            CHECK(vx_model_publish_adapter(model_a, &next_adapter,
                                           &next_revision, &report) ==
                  VX_STATUS_OK);
            CHECK(vx_execution_context_rebind_adapter(context_a, &report) ==
                  VX_STATUS_OK);
            CHECK(next_revision.adapter_revision != execution_adapter_revision);
        }
        CHECK(vx_result_read(stable_first, "sum", sum, sizeof(sum), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(sum[0], 4.0f) && closef(sum[1], 6.0f));
        CHECK(report.adapter_revision == execution_adapter_revision);
    }

    first.context = context_a;
    first.base = 10.0f;
    second.context = context_b;
    second.base = 1000.0f;
    CHECK(pthread_mutex_init(&overlap.mutex, NULL) == 0);
    CHECK(pthread_cond_init(&overlap.condition, NULL) == 0);
    vx_public_api_test_set_cpu_execute_hook(cpu_overlap_hook, &overlap);
    CHECK(pthread_create(&first_thread, NULL, run_thread_case, &first) == 0);
    CHECK(pthread_create(&second_thread, NULL, run_thread_case, &second) == 0);
    CHECK(pthread_join(first_thread, NULL) == 0);
    CHECK(pthread_join(second_thread, NULL) == 0);
    vx_public_api_test_set_cpu_execute_hook(NULL, NULL);
    CHECK(first.ok && second.ok);
    CHECK(overlap.entered == 2 && !overlap.timed_out);
    pthread_cond_destroy(&overlap.condition);
    pthread_mutex_destroy(&overlap.mutex);

    {
        float input[3] = {3.0f, 4.0f, 5.0f};
        float output[3] = {0};
        VxResult* prefix = NULL;
        VxResult* rejected = NULL;
        CHECK(vx_execution_context_set_input(context_c, "x", VX_DTYPE_F32,
                                             input, sizeof(input), &report) == VX_STATUS_OK);
        CHECK(vx_execution_context_execute_prefix(context_c, 0, &rejected,
                                                  &report) ==
              VX_STATUS_INVALID_ARGUMENT);
        CHECK(rejected == NULL &&
              !strcmp(report.reason, "INVALID_PREFIX_ROW_COUNT"));
        CHECK(vx_execution_context_execute_prefix(context_c, 1, &prefix,
                                                  &report) == VX_STATUS_OK);
        CHECK(report.stage == VX_STAGE_EXECUTE &&
              vx_result_read(prefix, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(input, output, sizeof(input)));
        vx_result_release(prefix);
        memset(output, 0, sizeof(output));
        CHECK(vx_execution_context_execute(context_c, &retained, &report) == VX_STATUS_OK);
        CHECK(vx_result_output_count(retained) == 1);
        CHECK(vx_result_read(retained, "y", output, sizeof(output), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(!memcmp(input, output, sizeof(input)));
    }

    {
        float input_a[3] = {10.0f, 11.0f, 12.0f};
        float input_b[3] = {20.0f, 21.0f, 22.0f};
        float next_a[3] = {13.0f, 14.0f, 15.0f};
        float next_b[3] = {23.0f, 24.0f, 25.0f};
        float output[3] = {0};
        VxResult* seed_a = NULL;
        VxResult* seed_b = NULL;
        VxResult* step_a = NULL;
        VxResult* step_b = NULL;
        CHECK(vx_execution_context_set_input(decode_a, "x", VX_DTYPE_F32,
                                             input_a, sizeof(input_a), &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_set_input(decode_b, "x", VX_DTYPE_F32,
                                             input_b, sizeof(input_b), &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_decode_seed(decode_a, &seed_a, &report) ==
              VX_STATUS_OK);
        CHECK(report.stage == VX_STAGE_DECODE &&
              strstr(report.decode_state, "seeded=1") &&
              strstr(report.decode_state, "last=dependency"));
        CHECK(vx_execution_context_decode_seed(decode_b, &seed_b, &report) ==
              VX_STATUS_OK);
        CHECK(vx_result_read(seed_a, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(output, input_a, sizeof(output)));
        CHECK(vx_execution_context_set_input(decode_a, "x", VX_DTYPE_F32,
                                             next_a, sizeof(next_a), &report) ==
              VX_STATUS_OK);
        CHECK(vx_execution_context_set_input(decode_b, "x", VX_DTYPE_F32,
                                             next_b, sizeof(next_b), &report) ==
              VX_STATUS_OK);
        {
            const char* invalid_adapter_path =
                "/tmp/volvox-public-api-invalid-adapter.bin";
            VxAdapterSource invalid_adapter = VX_ADAPTER_SOURCE_INIT;
            VxAdapterRevision invalid_revision = VX_ADAPTER_REVISION_INIT;
            CHECK(write_text(invalid_adapter_path, "not-an-adapter") == 0);
            invalid_adapter.adapter_name = "invalid-decode-adapter";
            invalid_adapter.package_path = invalid_adapter_path;
            CHECK(vx_model_publish_adapter(model_b, &invalid_adapter,
                                           &invalid_revision, &report) ==
                  VX_STATUS_OK);
            CHECK(remove(invalid_adapter_path) == 0);
            CHECK(vx_execution_context_select_adapter(
                      decode_b, &invalid_revision, &report) ==
                  VX_STATUS_INVALID_ARGUMENT);
            /* A rejected adapter package must leave the already seeded decode
             * stream usable with its previously pinned revision. */
            CHECK(vx_execution_context_decode_step(decode_b, -1, &step_b,
                                                    &report) == VX_STATUS_OK);
            CHECK(vx_result_read(step_b, "y", output, sizeof(output), NULL,
                                 &report) == VX_STATUS_OK);
            CHECK(!memcmp(output, next_b, sizeof(output)));
        }
        CHECK(vx_execution_context_decode_step(decode_a, -1, &step_a,
                                                &report) == VX_STATUS_OK);
        CHECK(vx_result_read(step_a, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(!memcmp(output, next_a, sizeof(output)));
        CHECK(vx_execution_context_decode_reset(decode_a, &report) ==
              VX_STATUS_OK);
        {
            VxResult* rejected = NULL;
            CHECK(vx_execution_context_decode_step(decode_a, -1, &rejected,
                                                    &report) ==
                  VX_STATUS_EXECUTION_FAILED);
            CHECK(rejected == NULL && strstr(report.decode_state, "seeded=0"));
        }
        vx_result_release(step_b);
        vx_result_release(step_a);
        vx_result_release(seed_b);
        vx_result_release(seed_a);
    }

    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    {
        VxCompiledModel* rejected_compile = NULL;
        VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
        VxModelSource source = VX_MODEL_SOURCE_INIT;
        VxModel* rejected_model = NULL;
        float output[3] = {0};
        VxResult* final_step = NULL;
        source.graph_path = graph_b;
        CHECK(vx_model_compile(model_b, &policy, &rejected_compile, &report) ==
              VX_STATUS_HANDLE_DISPOSED);
        CHECK(rejected_compile == NULL);
        CHECK(vx_runtime_load_model(runtime, &source, &rejected_model, &report) ==
              VX_STATUS_HANDLE_DISPOSED);
        CHECK(rejected_model == NULL);
        CHECK(vx_execution_context_decode_step(decode_b, -1, &final_step,
                                                &report) == VX_STATUS_OK);
        CHECK(vx_result_read(final_step, "y", output, sizeof(output), NULL,
                             &report) == VX_STATUS_OK);
        CHECK(closef(output[0], 23.0f));
        vx_result_release(final_step);
    }
    vx_execution_context_release(decode_b);
    vx_execution_context_release(decode_a);
    CHECK(vx_execution_context_close(context_c, &report) == VX_STATUS_OK);
    vx_execution_context_release(context_c);
    vx_compiled_model_release(compiled_b);
    vx_model_release(model_b);
    vx_execution_context_release(context_b);
    vx_execution_context_release(context_a);
    vx_compiled_model_release(compiled_a);
    vx_model_release(model_a);
    vx_runtime_release(runtime);
    {
        float sum[2] = {0};
        CHECK(vx_result_read(stable_first, "sum", sum, sizeof(sum), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(sum[0], 4.0f) && closef(sum[1], 6.0f));
    }
    vx_result_release(stable_first);
    {
        float first_read[3] = {0};
        float second_read[3] = {0};
        size_t required = 0;
        CHECK(vx_result_read(retained, "y", NULL, 0, &required, &report) == VX_STATUS_OK);
        CHECK(required == sizeof(first_read));
        CHECK(vx_result_read(retained, "y", first_read, sizeof(first_read), NULL, &report) ==
              VX_STATUS_OK);
        first_read[0] = -999.0f;
        CHECK(vx_result_read(retained, "y", second_read, sizeof(second_read), NULL, &report) ==
              VX_STATUS_OK);
        CHECK(closef(second_read[0], 3.0f));
    }
    vx_result_release(retained);

    CHECK(test_model_source_snapshots() == 0);
    CHECK(test_declared_output_descriptors() == 0);
    CHECK(test_provider(provider_graph_path) == 0);
    vx_runtime_release(NULL);
    vx_model_release(NULL);
    vx_compiled_model_release(NULL);
    vx_execution_context_release(NULL);
    vx_result_release(NULL);
    remove(graph_a);
    remove(graph_b);
    remove(provider_graph_path);
    remove(invalid_graph_path);
    puts("native public handle and provider API tests passed");
    return 0;
}
