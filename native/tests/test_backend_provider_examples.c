#include "volvoxai.h"
#include "host_backend.h"
#include "safetensors.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

static int closef(float left, float right) {
    return fabsf(left - right) < 1.0e-6f;
}

static int bad_compiled_destroyed;

static VxStatus bad_runtime_create(void* user_data,
                                   const VxRuntimeOptions* options,
                                   void** out_runtime,
                                   VxReport* report) {
    (void)user_data;
    (void)options;
    (void)report;
    *out_runtime = malloc(1u);
    return *out_runtime ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
}

static void bad_runtime_destroy(void* runtime) { free(runtime); }

static VxStatus bad_compile(
    void* runtime,
    const VxBackendCompileInput* input,
    const VxBackendPolicy* policy,
    void** out_compiled,
    VxBackendShapeDomainAttestation* attestation,
    VxReport* report) {
    (void)policy;
    if (!runtime || !input || input->struct_size != sizeof(*input) ||
        !attestation || attestation->struct_size != sizeof(*attestation) ||
        !out_compiled)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_compiled = malloc(1u);
    if (!*out_compiled) return VX_STATUS_OUT_OF_MEMORY;
    attestation->struct_size++;
    attestation->graph_fingerprint = input->graph_fingerprint;
    attestation->shape_domain_proof_identity =
        input->shape_domain_proof_identity;
    attestation->maximum_tensor_bytes = 8u;
    attestation->maximum_resident_bytes = 24u;
    attestation->resource_limit_bytes = 1024u;
    attestation->has_resource_limit = 1;
    if (report && report->struct_size == sizeof(*report)) {
        report->route_attested = 1;
        snprintf(report->device, sizeof(report->device), "%s", "mock");
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=bad-attestation;route=all");
    }
    return VX_STATUS_OK;
}

static void bad_compiled_destroy(void* compiled) {
    bad_compiled_destroyed++;
    free(compiled);
}

static VxStatus bad_context_create(void* compiled,
                                   const VxContextOptions* options,
                                   void** out_context,
                                   VxReport* report) {
    (void)compiled;
    (void)options;
    (void)out_context;
    (void)report;
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static VxStatus bad_context_execute(void* context,
                                    const VxTensorBinding* inputs,
                                    size_t input_count,
                                    const VxBackendOutputSink* sink,
                                    VxReport* report) {
    (void)context;
    (void)inputs;
    (void)input_count;
    (void)sink;
    (void)report;
    return VX_STATUS_BACKEND_UNSUPPORTED;
}

static void bad_context_destroy(void* context) { (void)context; }

static int failed_runtime_destroyed;
static int failed_runtime_token;

static VxStatus reported_runtime_failure(void* user_data,
                                         const VxRuntimeOptions* options,
                                         void** out_runtime,
                                         VxReport* report) {
    (void)user_data;
    (void)options;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = &failed_runtime_token;
    if (report && report->struct_size == sizeof(*report)) {
        report->status = VX_STATUS_BACKEND_UNAVAILABLE;
        report->stage = VX_STAGE_EXECUTE;
        report->route_attested = 1;
        snprintf(report->device, sizeof(report->device), "%s",
                 "fixture-device");
        snprintf(report->reason, sizeof(report->reason), "%s",
                 "DRIVER_INIT_FAILED");
        snprintf(report->message, sizeof(report->message), "%s",
                 "fixture runtime creation failed");
    }
    return VX_STATUS_IO_ERROR;
}

static VxStatus silent_oom_runtime_failure(void* user_data,
                                           const VxRuntimeOptions* options,
                                           void** out_runtime,
                                           VxReport* report) {
    (void)user_data;
    (void)options;
    (void)report;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = NULL;
    return VX_STATUS_OUT_OF_MEMORY;
}

static VxStatus missing_runtime_create(void* user_data,
                                       const VxRuntimeOptions* options,
                                       void** out_runtime,
                                       VxReport* report) {
    (void)user_data;
    (void)options;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = NULL;
    if (report && report->struct_size == sizeof(*report)) {
        snprintf(report->reason, sizeof(report->reason), "%s", "OK");
        snprintf(report->message, sizeof(report->message), "%s",
                 "fixture reported success");
    }
    return VX_STATUS_OK;
}

static void failed_runtime_destroy(void* runtime) {
    (void)runtime;
    failed_runtime_destroyed++;
}

typedef VxStatus (*RuntimeCreateCallback)(void*, const VxRuntimeOptions*,
                                          void**, VxReport*);

extern const void* vx_stale_backend_provider_fixture(void);
extern size_t vx_stale_backend_provider_fixture_size(void);
extern int vx_stale_backend_provider_runtime_create_calls(void);

static VxStatus register_runtime_create_fixture(
    VxRuntime* runtime,
    const char* name,
    RuntimeCreateCallback runtime_create,
    VxReport* report) {
    const VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = name,
        .shape_domain = {
            .struct_size = sizeof(VxBackendShapeDomainCapability),
            .proof_protocol = VX_BACKEND_SHAPE_PROOF_PROTOCOL,
            .resource_protocol = VX_BACKEND_RESOURCE_PROTOCOL,
            .support = VX_BACKEND_SHAPE_DOMAIN_FULL,
        },
        .runtime_create = runtime_create,
        .runtime_destroy = failed_runtime_destroy,
        .compile = bad_compile,
        .compiled_destroy = bad_compiled_destroy,
        .context_create = bad_context_create,
        .context_execute = bad_context_execute,
        .context_destroy = bad_context_destroy,
        .exact_contract_marker = VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER,
        .exact_contract_extent = sizeof(VxBackendProvider),
    };
    return vx_runtime_register_provider(runtime, &provider, report);
}

static VxStatus bad_output_compile(
    void* runtime,
    const VxBackendCompileInput* input,
    const VxBackendPolicy* policy,
    void** out_compiled,
    VxBackendShapeDomainAttestation* attestation,
    VxReport* report) {
    (void)policy;
    if (!runtime || !input || input->struct_size != sizeof(*input) ||
        !attestation || attestation->struct_size != sizeof(*attestation) ||
        !out_compiled)
        return VX_STATUS_INVALID_ARGUMENT;
    *out_compiled = malloc(1u);
    if (!*out_compiled) return VX_STATUS_OUT_OF_MEMORY;
    attestation->graph_fingerprint = input->graph_fingerprint;
    attestation->shape_domain_proof_identity =
        input->shape_domain_proof_identity;
    attestation->maximum_tensor_bytes = 8u;
    attestation->maximum_resident_bytes = 24u;
    attestation->resource_limit_bytes = 0u;
    attestation->has_resource_limit = 0;
    if (report && report->struct_size == sizeof(*report)) {
        report->route_attested = 1;
        snprintf(report->device, sizeof(report->device), "%s", "mock-host");
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=bad-output;route=all");
    }
    return VX_STATUS_OK;
}

static VxStatus bad_output_context_create(
    void* compiled,
    const VxContextOptions* options,
    void** out_context,
    VxReport* report) {
    (void)options;
    (void)report;
    if (!compiled || !out_context) return VX_STATUS_INVALID_ARGUMENT;
    *out_context = malloc(1u);
    return *out_context ? VX_STATUS_OK : VX_STATUS_OUT_OF_MEMORY;
}

static VxStatus bad_output_execute(
    void* context,
    const VxTensorBinding* inputs,
    size_t input_count,
    const VxBackendOutputSink* sink,
    VxReport* report) {
    const int64_t wrong_shape[1] = {1};
    const float wrong_output[1] = {0.0f};
    (void)inputs;
    (void)input_count;
    if (!context || !inputs || input_count != 2u ||
        inputs[0].struct_size != sizeof(inputs[0]) ||
        inputs[1].struct_size != sizeof(inputs[1]) ||
        !sink || sink->struct_size != sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
    if (report && report->struct_size == sizeof(*report)) {
        report->route_attested = 1;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=bad-output;route=all");
    }
    return sink->write(sink->user_data, "sum", VX_DTYPE_F32,
                       wrong_shape, 1u, wrong_output,
                       sizeof(wrong_output));
}

static VxStatus register_bad_output(VxRuntime* runtime, VxReport* report) {
    const VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = "bad-output",
        .shape_domain = {
            .struct_size = sizeof(VxBackendShapeDomainCapability),
            .proof_protocol = VX_BACKEND_SHAPE_PROOF_PROTOCOL,
            .resource_protocol = VX_BACKEND_RESOURCE_PROTOCOL,
            .support = VX_BACKEND_SHAPE_DOMAIN_FULL,
        },
        .runtime_create = bad_runtime_create,
        .runtime_destroy = bad_runtime_destroy,
        .compile = bad_output_compile,
        .compiled_destroy = bad_runtime_destroy,
        .context_create = bad_output_context_create,
        .context_execute = bad_output_execute,
        .context_destroy = bad_runtime_destroy,
        .exact_contract_marker = VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER,
        .exact_contract_extent = sizeof(VxBackendProvider),
    };
    return vx_runtime_register_provider(runtime, &provider, report);
}

static VxStatus register_bad_attestation(VxRuntime* runtime,
                                         VxReport* report) {
    const VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = "bad-attestation",
        .shape_domain = {
            .struct_size = sizeof(VxBackendShapeDomainCapability),
            .proof_protocol = VX_BACKEND_SHAPE_PROOF_PROTOCOL,
            .resource_protocol = VX_BACKEND_RESOURCE_PROTOCOL,
            .support = VX_BACKEND_SHAPE_DOMAIN_FULL,
        },
        .runtime_create = bad_runtime_create,
        .runtime_destroy = bad_runtime_destroy,
        .compile = bad_compile,
        .compiled_destroy = bad_compiled_destroy,
        .context_create = bad_context_create,
        .context_execute = bad_context_execute,
        .context_destroy = bad_context_destroy,
        .exact_contract_marker = VX_BACKEND_PROVIDER_EXACT_CONTRACT_MARKER,
        .exact_contract_extent = sizeof(VxBackendProvider),
    };
    return vx_runtime_register_provider(runtime, &provider, report);
}

static int write_graph(const char* path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},"
        "\"inputs\":{"
        "\"a\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\",\"inputs\":{"
        "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"sum\",\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"params\":{}}],\"outputs\":[\"sum\"]}";
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    return fwrite(graph, 1, sizeof(graph) - 1u, file) == sizeof(graph) - 1u &&
        fclose(file) == 0 ? 0 : -1;
}

extern int vx_public_api_test_dynamic_shape_state(
    const VxExecutionContext* context,
    uint64_t* resource_generation,
    size_t* arena_capacity,
    size_t* arena_high_water,
    uint64_t* arena_grow_count);
extern int vx_public_api_test_copy_tensor(
    const VxExecutionContext* context,
    const char* name,
    void* bytes,
    size_t byte_size);

static int write_dynamic_graph(const char* path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"N\":{\"min\":2,\"max\":1024,"
        "\"multiple_of\":2}},\"inputs\":{"
        "\"a\":{\"shape\":[\"N\"],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[\"N\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\",\"inputs\":{"
        "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"sum\",\"dtype\":\"float32\","
        "\"shape\":[\"N\"]}},\"params\":{}}],\"outputs\":[\"sum\"]}";
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    return fwrite(graph, 1, sizeof(graph) - 1u, file) ==
               sizeof(graph) - 1u &&
        fclose(file) == 0 ? 0 : -1;
}

static int write_dynamic_dropout_graph(const char* path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"N\":{\"min\":2,\"max\":8,"
        "\"multiple_of\":2}},\"inputs\":{"
        "\"x\":{\"shape\":[\"N\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"dropout\",\"opType\":\"Dropout\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[\"N\"]}},\"params\":{}}],\"outputs\":[\"y\"]}";
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    return fwrite(graph, 1, sizeof(graph) - 1u, file) ==
               sizeof(graph) - 1u &&
        fclose(file) == 0 ? 0 : -1;
}

static int write_wrong_shape_assertion_graph(const char* path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"N\":{\"min\":2,\"max\":4,"
        "\"multiple_of\":2}},\"inputs\":{"
        "\"a\":{\"shape\":[\"N\"],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[\"N\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"bad-add\",\"opType\":\"Add\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"sum\","
        "\"dtype\":\"float32\",\"shape\":[2]}},\"params\":{}}],"
        "\"outputs\":[\"sum\"]}";
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    return fwrite(graph, 1, sizeof(graph) - 1u, file) ==
               sizeof(graph) - 1u &&
        fclose(file) == 0 ? 0 : -1;
}

static int write_static_qargmax_package(const char* graph_path,
                                        const char* weights_path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},\"inputs\":{\"x\":{"
        "\"shape\":[2,3,2],\"dtype\":\"int8\"}},\"nodes\":[{"
        "\"id\":\"qargmax\",\"opType\":\"QArgMax\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"int32\","
        "\"shape\":[2,2]}},\"params\":{\"axis\":-2}}],"
        "\"outputs\":[\"y\"],\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"x\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.7\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.13\"}}}}";
    const int shape[1] = {1};
    const float scale[1] = {0.5f};
    const int8_t zero[1] = {-17};
    SafetensorsFile weights = {0};
    FILE* file = fopen(graph_path, "wb");
    int result = -1;
    if (!file) return -1;
    if (fwrite(graph, 1, sizeof(graph) - 1u, file) !=
            sizeof(graph) - 1u) {
        (void)fclose(file);
        return -1;
    }
    if (fclose(file) != 0) return -1;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0) return -1;
    if (safetensors_add_tensor(
            &weights, "__fixture_affine.scale.7", SAFETENSORS_DTYPE_F32,
            shape, 1, scale, sizeof(scale)) == 0 &&
        safetensors_add_tensor(
            &weights, "__fixture_affine.zero.13", SAFETENSORS_DTYPE_I8,
            shape, 1, zero, sizeof(zero)) == 0 &&
        safetensors_save(weights_path, &weights) == 0) result = 0;
    safetensors_free(&weights);
    return result;
}

static int write_static_weight_identity_package(const char* graph_path,
                                                const char* weights_path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},\"inputs\":{},\"nodes\":[{"
        "\"id\":\"identity\",\"opType\":\"Identity\","
        "\"inputs\":{\"input\":\"w\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[2]}},\"params\":{}}],\"outputs\":[\"y\"]}";
    const int shape[1] = {2};
    const float values[2] = {7.0f, -3.0f};
    SafetensorsFile weights = {0};
    FILE* file = fopen(graph_path, "wb");
    int result = -1;
    if (!file) return -1;
    if (fwrite(graph, 1, sizeof(graph) - 1u, file) !=
            sizeof(graph) - 1u) {
        (void)fclose(file);
        return -1;
    }
    if (fclose(file) != 0) return -1;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0) return -1;
    if (safetensors_add_tensor(&weights, "w", SAFETENSORS_DTYPE_F32,
                               shape, 1, values, sizeof(values)) == 0 &&
        safetensors_save(weights_path, &weights) == 0) result = 0;
    safetensors_free(&weights);
    return result;
}

static int test_builtin_dynamic_shapes(void) {
    const char* graph_path = "/tmp/volvox-dynamic-cpu.graph.json";
    const char* dropout_graph_path =
        "/tmp/volvox-dynamic-dropout-cpu.graph.json";
    const char* wrong_assertion_path =
        "/tmp/volvox-dynamic-wrong-assertion-cpu.graph.json";
    const char* static_q_graph_path =
        "/tmp/volvox-static-deferred-qargmax.graph.json";
    const char* static_q_weights_path =
        "/tmp/volvox-static-deferred-qargmax.safetensors";
    const char* static_q_weight_paths[1] = {static_q_weights_path};
    const char* identity_graph_path =
        "/tmp/volvox-static-weight-identity.graph.json";
    const char* identity_weights_path =
        "/tmp/volvox-static-weight-identity.safetensors";
    const char* identity_weight_paths[1] = {identity_weights_path};
    const char* required[] = {"cpu"};
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
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    float small_a[2] = {1.0f, 2.0f};
    float small_b[2] = {3.0f, 4.0f};
    float small_output[2] = {0};
    float committed_input[2] = {0};
    float assertion_a[4] = {11.0f, 12.0f, 13.0f, 14.0f};
    float assertion_b[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float* large_a = (float*)malloc(1024u * sizeof(float));
    float* large_b = (float*)malloc(1024u * sizeof(float));
    float* large_output = (float*)malloc(1024u * sizeof(float));
    VxTensorBinding small[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {2},
         small_a, sizeof(small_a), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {2},
         small_b, sizeof(small_b), VX_MEMORY_HOST},
    };
    VxTensorBinding large[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {1024},
         large_a, 1024u * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {1024},
         large_b, 1024u * sizeof(float), VX_MEMORY_HOST},
    };
    VxTensorBinding invalid[2];
    VxTensorBinding wrong_assertion[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {4},
         assertion_a, sizeof(assertion_a), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {4},
         assertion_b, sizeof(assertion_b), VX_MEMORY_HOST},
    };
    uint64_t generation = 0;
    uint64_t failed_generation = 0;
    uint64_t grow_count = 0;
    size_t small_capacity = 0;
    size_t large_capacity = 0;
    size_t high_water = 0;
    CHECK(large_a && large_b && large_output);
    for (size_t index = 0; index < 1024u; index++) {
        large_a[index] = (float)index;
        large_b[index] = (float)(2048u - index);
    }
    CHECK(write_dynamic_graph(graph_path) == 0);
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    source.graph_path = graph_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = required;
    policy.backend_count = 1;
    {
        VxStatus status = vx_model_compile(
            model, &policy, &compiled, &report);
        if (status != VX_STATUS_OK)
            fprintf(stderr,
                    "dynamic CPU compile: status=%d reason=%s message=%s route=%s\n",
                    status, report.reason, report.message,
                    report.route_evidence);
        CHECK(status == VX_STATUS_OK);
    }
    CHECK(strstr(report.route_evidence, "domain_route=all") != NULL &&
          strstr(report.route_evidence, "dynamic=1") != NULL &&
          strstr(report.route_evidence, "canonical=1") != NULL &&
          strstr(report.route_evidence, "native_routes=1") != NULL &&
          strstr(report.route_evidence, "shape_proof=") != NULL);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                           &context, &report) == VX_STATUS_OK);

    {
        VxStatus status = vx_execution_context_execute(
            context, small, 2u, &result, &report);
        if (status != VX_STATUS_OK)
            fprintf(stderr, "dynamic small execute: status=%d reason=%s message=%s route=%s\n",
                    status, report.reason, report.message,
                    report.route_evidence);
        CHECK(status == VX_STATUS_OK);
    }
    CHECK(strstr(report.route_evidence, "shape_plan=cold") != NULL &&
          strstr(report.route_evidence, "shape_signature=") != NULL &&
          strstr(report.route_evidence, "arena_capacity=") != NULL);
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(info.rank == 1u && info.shape[0] == 2 &&
          info.byte_size == sizeof(small_output));
    CHECK(vx_result_read(result, "sum", small_output, sizeof(small_output),
                         NULL, &report) == VX_STATUS_OK);
    CHECK(closef(small_output[0], 4.0f) && closef(small_output[1], 6.0f));
    vx_result_release(result);
    result = NULL;
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &small_capacity, &high_water,
              &grow_count) == 0);
    CHECK(generation == 1u && small_capacity >= 3u * sizeof(small_a) &&
          high_water == small_capacity && grow_count == 1u);

    CHECK(vx_execution_context_execute(context, large, 2u, &result, &report) ==
          VX_STATUS_OK);
    CHECK(strstr(report.route_evidence, "shape_plan=cold") != NULL);
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(info.rank == 1u && info.shape[0] == 1024 &&
          info.byte_size == 1024u * sizeof(float));
    CHECK(vx_result_read(result, "sum", large_output,
                         1024u * sizeof(float), NULL, &report) == VX_STATUS_OK);
    for (size_t index = 0; index < 1024u; index++)
        CHECK(closef(large_output[index], 2048.0f));
    vx_result_release(result);
    result = NULL;
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &large_capacity, &high_water,
              &grow_count) == 0);
    CHECK(generation == 2u && large_capacity > small_capacity &&
          high_water == large_capacity && grow_count == 2u);

    memcpy(invalid, large, sizeof(invalid));
    invalid[1].name = "a";
    CHECK(vx_execution_context_execute(context, invalid, 2u, &result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    invalid[0] = small[0];
    invalid[1] = large[1];
    invalid[1].shape[0] = 4;
    invalid[1].byte_size = 4u * sizeof(float);
    CHECK(vx_execution_context_execute(context, invalid, 2u, &result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    invalid[0] = large[0];
    invalid[1] = large[1];
    invalid[0].shape[0] = invalid[1].shape[0] = 3;
    invalid[0].byte_size = invalid[1].byte_size = 3u * sizeof(float);
    CHECK(vx_execution_context_execute(context, invalid, 2u, &result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    invalid[0] = small[0];
    invalid[1] = small[1];
    invalid[0].shape[0] = invalid[1].shape[0] = 1;
    invalid[0].byte_size = invalid[1].byte_size = sizeof(float);
    CHECK(vx_execution_context_execute(context, invalid, 2u, &result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    invalid[0].shape[0] = invalid[1].shape[0] = 1026;
    invalid[0].byte_size = invalid[1].byte_size = 1026u * sizeof(float);
    CHECK(vx_execution_context_execute(context, invalid, 2u, &result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(result == NULL);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &failed_generation, NULL, NULL, NULL) == 0);
    CHECK(failed_generation == generation);

    CHECK(vx_execution_context_execute(context, small, 2u, &result, &report) ==
          VX_STATUS_OK);
    CHECK(strstr(report.route_evidence, "shape_plan=hit") != NULL);
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK &&
          info.shape[0] == 2);
    CHECK(vx_result_read(result, "sum", small_output, sizeof(small_output),
                         NULL, &report) == VX_STATUS_OK);
    CHECK(closef(small_output[0], 4.0f) && closef(small_output[1], 6.0f));
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &small_capacity, &high_water,
              &grow_count) == 0);
    CHECK(generation == 3u && small_capacity == large_capacity &&
          grow_count == 2u);

    vx_result_release(result);
    vx_execution_context_release(context);
    context = NULL;
    vx_compiled_model_release(compiled);
    vx_model_release(model);

    model = NULL;
    compiled = NULL;
    CHECK(write_dynamic_dropout_graph(dropout_graph_path) == 0);
    source.graph_path = dropout_graph_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(compiled != NULL &&
          strstr(report.route_evidence, "canonical=1") != NULL);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                           &context, &report) == VX_STATUS_OK);
    {
        float dropout_values[2] = {5.0f, -3.0f};
        float dropout_output[2] = {0};
        VxTensorBinding dropout_input = {
            sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {2},
            dropout_values, sizeof(dropout_values), VX_MEMORY_HOST
        };
        CHECK(vx_execution_context_execute(
                  context, &dropout_input, 1u, &result, &report) == VX_STATUS_OK);
        CHECK(vx_result_read(result, "y", dropout_output,
                             sizeof(dropout_output), NULL, &report) == VX_STATUS_OK);
        CHECK(closef(dropout_output[0], 5.0f) && closef(dropout_output[1], -3.0f));
        vx_result_release(result);
        result = NULL;
    }
    vx_execution_context_release(context);
    context = NULL;
    vx_compiled_model_release(compiled);
    compiled = NULL;
    vx_model_release(model);

    model = NULL;
    CHECK(write_static_qargmax_package(
              static_q_graph_path, static_q_weights_path) == 0);
    source.graph_path = static_q_graph_path;
    source.weight_paths = static_q_weight_paths;
    source.weight_path_count = 1u;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(compiled != NULL);
    vx_compiled_model_release(compiled);
    compiled = NULL;
    vx_model_release(model);

    model = NULL;
    CHECK(write_static_weight_identity_package(
              identity_graph_path, identity_weights_path) == 0);
    source.graph_path = identity_graph_path;
    source.weight_paths = identity_weight_paths;
    source.weight_path_count = 1u;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                           &context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute(
              context, NULL, 0u, &result, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "y", small_output, sizeof(small_output),
                         NULL, &report) == VX_STATUS_OK);
    CHECK(closef(small_output[0], 7.0f) && closef(small_output[1], -3.0f));
    vx_result_release(result);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);

    result = NULL;
    context = NULL;
    compiled = NULL;
    model = NULL;
    source.weight_paths = NULL;
    source.weight_path_count = 0u;
    CHECK(write_wrong_shape_assertion_graph(wrong_assertion_path) == 0);
    source.graph_path = wrong_assertion_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) ==
          VX_STATUS_OK);
    CHECK(vx_model_compile(model, &policy, &compiled, &report) ==
          VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                           &context, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_execute(context, small, 2u, &result, &report) ==
          VX_STATUS_OK);
    vx_result_release(result);
    result = NULL;
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, NULL, NULL, NULL) == 0 &&
          generation == 1u);
    CHECK(vx_public_api_test_copy_tensor(
              context, "a", committed_input, sizeof(committed_input)) == 0 &&
          !memcmp(committed_input, small_a, sizeof(committed_input)));
    CHECK(vx_execution_context_execute(
              context, wrong_assertion, 2u, &result, &report) ==
          VX_STATUS_INVALID_GRAPH);
    CHECK(result == NULL && !strcmp(report.reason, "INVALID_GRAPH"));
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &failed_generation, NULL, NULL, NULL) == 0 &&
          failed_generation == generation);
    memset(committed_input, 0, sizeof(committed_input));
    CHECK(vx_public_api_test_copy_tensor(
              context, "a", committed_input, sizeof(committed_input)) == 0 &&
          !memcmp(committed_input, small_a, sizeof(committed_input)));
    CHECK(vx_execution_context_execute(context, small, 2u, &result, &report) ==
          VX_STATUS_OK);
    CHECK(strstr(report.route_evidence, "shape_plan=hit") != NULL);
    vx_result_release(result);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    vx_runtime_release(runtime);
    free(large_output);
    free(large_b);
    free(large_a);
    remove(graph_path);
    remove(dropout_graph_path);
    remove(wrong_assertion_path);
    remove(static_q_graph_path);
    remove(static_q_weights_path);
    remove(identity_graph_path);
    remove(identity_weights_path);
    return 0;
}

int main(void) {
    const char* graph_path = "/tmp/volvox-example-provider.graph.json";
    const char* required[] = { "example-host" };
    const char* invalid_required[] = { "bad-attestation" };
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxCompiledModel* invalid_compiled = NULL;
    VxExecutionContext* first = NULL;
    VxExecutionContext* second = NULL;
    VxResult* first_result = NULL;
    VxResult* second_result = NULL;
    const float first_a[2] = {1.0f, 2.0f};
    const float first_b[2] = {3.0f, 4.0f};
    const float second_a[2] = {10.0f, 20.0f};
    const float second_b[2] = {-2.0f, 5.0f};
    const VxTensorBinding first_inputs[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {2},
         first_a, sizeof(first_a), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {2},
         first_b, sizeof(first_b), VX_MEMORY_HOST},
    };
    const VxTensorBinding second_inputs[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {2},
         second_a, sizeof(second_a), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {2},
         second_b, sizeof(second_b), VX_MEMORY_HOST},
    };
    VxTensorBinding invalid_inputs[2];
    VxTensorSpec input_spec = VX_TENSOR_SPEC_INIT;
    float output[2] = {0};

    CHECK(write_graph(graph_path) == 0);
    CHECK(volvoxai_example_host_backend_test_oversized_descriptors() == 0);
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    report = (VxReport)VX_REPORT_INIT;
    CHECK(vx_stale_backend_provider_fixture_size() ==
          offsetof(VxBackendProvider, exact_contract_marker));
    CHECK(vx_runtime_register_provider(
              runtime,
              (const VxBackendProvider*)vx_stale_backend_provider_fixture(),
              &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(report.status == VX_STATUS_INVALID_ARGUMENT &&
          !strcmp(report.reason, "INVALID_PROVIDER") &&
          strstr(report.message, "current exact contract") != NULL);
    CHECK(vx_stale_backend_provider_runtime_create_calls() == 0);
    {
        VxReport oversized_report = VX_REPORT_INIT;
        oversized_report.struct_size++;
        oversized_report.status = VX_STATUS_INTERNAL;
        snprintf(oversized_report.reason, sizeof(oversized_report.reason),
                 "%s", "UNCHANGED");
        CHECK(register_runtime_create_fixture(
                  runtime, "reported-failure", reported_runtime_failure,
                  &oversized_report) == VX_STATUS_INVALID_ARGUMENT);
        CHECK(oversized_report.status == VX_STATUS_INTERNAL);
        CHECK(!strcmp(oversized_report.reason, "UNCHANGED"));
    }
    report = (VxReport)VX_REPORT_INIT;
    CHECK(register_runtime_create_fixture(
              runtime, "reported-failure", reported_runtime_failure,
              &report) == VX_STATUS_IO_ERROR);
    CHECK(report.status == VX_STATUS_IO_ERROR &&
          report.stage == VX_STAGE_RUNTIME_CREATE && report.route_attested);
    CHECK(!strcmp(report.device, "fixture-device"));
    CHECK(!strcmp(report.reason, "DRIVER_INIT_FAILED"));
    CHECK(!strcmp(report.message, "fixture runtime creation failed"));
    CHECK(failed_runtime_destroyed == 0);
    report = (VxReport)VX_REPORT_INIT;
    CHECK(register_runtime_create_fixture(
              runtime, "silent-oom", silent_oom_runtime_failure,
              &report) == VX_STATUS_OUT_OF_MEMORY);
    CHECK(report.status == VX_STATUS_OUT_OF_MEMORY &&
          report.stage == VX_STAGE_RUNTIME_CREATE);
    CHECK(!strcmp(report.reason, "OUT_OF_MEMORY"));
    CHECK(!strcmp(report.message,
                  "backend provider runtime allocation failed"));
    CHECK(failed_runtime_destroyed == 0);
    report = (VxReport)VX_REPORT_INIT;
    CHECK(register_runtime_create_fixture(
              runtime, "missing-runtime", missing_runtime_create,
              &report) == VX_STATUS_BACKEND_UNAVAILABLE);
    CHECK(report.status == VX_STATUS_BACKEND_UNAVAILABLE &&
          report.stage == VX_STAGE_RUNTIME_CREATE);
    CHECK(!strcmp(report.reason, "BACKEND_UNAVAILABLE"));
    CHECK(!strcmp(report.message,
                  "backend provider runtime creation returned no runtime"));
    CHECK(failed_runtime_destroyed == 0);
    CHECK(volvoxai_example_host_backend_register(runtime, &report) == VX_STATUS_OK);
    CHECK(register_bad_attestation(runtime, &report) == VX_STATUS_OK);
    CHECK(register_bad_output(runtime, &report) == VX_STATUS_OK);
    source.graph_path = graph_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = invalid_required;
    policy.backend_count = 1;
    CHECK(vx_model_compile(model, &policy, &invalid_compiled, &report) ==
          VX_STATUS_ABI_UNSUPPORTED);
    CHECK(invalid_compiled == NULL && bad_compiled_destroyed == 1);
    policy.backends = required;
    policy.backend_count = 1;
    CHECK(vx_model_compile(model, &policy, &compiled, &report) == VX_STATUS_OK);
    CHECK(!strcmp(report.backend, "example-host") &&
          !strcmp(report.device, "host-cpu") && report.route_attested &&
          !report.operator_fallback_used);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &first, &report) == VX_STATUS_OK);
    CHECK(vx_compiled_model_create_context(compiled, &context_options,
                                            &second, &report) == VX_STATUS_OK);
    CHECK(vx_execution_context_input_count(first) == 2u);
    CHECK(vx_execution_context_input_spec(first, 0u, &input_spec, &report) ==
          VX_STATUS_OK);
    CHECK(!strcmp(input_spec.name, "a") && input_spec.rank == 1u &&
          input_spec.dimensions[0].kind == VX_DIMENSION_FIXED &&
          input_spec.dimensions[0].symbol == NULL &&
          input_spec.dimensions[0].min == 2 &&
          input_spec.dimensions[0].max == 2 &&
          input_spec.dimensions[0].multiple_of == 1);
    memcpy(invalid_inputs, first_inputs, sizeof(invalid_inputs));
    invalid_inputs[1].name = "a";
    CHECK(vx_execution_context_execute(first, invalid_inputs, 2u,
                                       &first_result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(first_result == NULL);
    CHECK(vx_execution_context_execute(first, first_inputs, 1u,
                                       &first_result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(first_result == NULL);
    memcpy(invalid_inputs, first_inputs, sizeof(invalid_inputs));
    invalid_inputs[0].location = VX_MEMORY_DEVICE;
    CHECK(vx_execution_context_execute(first, invalid_inputs, 2u,
                                       &first_result, &report) ==
          VX_STATUS_INVALID_ARGUMENT);
    CHECK(first_result == NULL);
    CHECK(vx_execution_context_execute(first, first_inputs, 2u,
                                       &first_result, &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_execute(second, second_inputs, 2u,
                                       &second_result, &report) ==
          VX_STATUS_OK);
    CHECK(vx_result_read(first_result, "sum", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(closef(output[0], 4.0f) && closef(output[1], 6.0f));
    CHECK(vx_result_read(second_result, "sum", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(closef(output[0], 8.0f) && closef(output[1], 25.0f));

    {
        const char* bad_output_required[] = {"bad-output"};
        VxCompiledModel* bad_output_compiled = NULL;
        VxExecutionContext* bad_output_context = NULL;
        VxResult* bad_output_result = NULL;
        policy.backends = bad_output_required;
        CHECK(vx_model_compile(model, &policy, &bad_output_compiled, &report) ==
              VX_STATUS_OK);
        CHECK(vx_compiled_model_create_context(
                  bad_output_compiled, &context_options,
                  &bad_output_context, &report) == VX_STATUS_OK);
        CHECK(vx_execution_context_execute(
                  bad_output_context, first_inputs, 2u,
                  &bad_output_result, &report) == VX_STATUS_EXECUTION_FAILED);
        CHECK(bad_output_result == NULL);
        vx_execution_context_release(bad_output_context);
        vx_compiled_model_release(bad_output_compiled);
    }

    CHECK(vx_runtime_close(runtime, &report) == VX_STATUS_OK);
    vx_execution_context_release(second);
    vx_execution_context_release(first);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    vx_runtime_release(runtime);
    CHECK(vx_result_read(first_result, "sum", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(closef(output[0], 4.0f) && closef(output[1], 6.0f));
    vx_result_release(second_result);
    vx_result_release(first_result);
    remove(graph_path);
    CHECK(test_builtin_dynamic_shapes() == 0);
    puts("runtime-scoped backend provider examples passed");
    return 0;
}
