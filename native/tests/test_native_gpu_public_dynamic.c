#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "batch_matmul_f32_plan.h"
#include "safetensors.h"
#include "../../examples/native_dynamic_batch_benchmark/evidence_tokens.h"

#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", \
                __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

extern int vx_public_api_test_dynamic_shape_state(
    const VxExecutionContext* context,
    uint64_t* resource_generation,
    size_t* arena_capacity,
    size_t* arena_high_water,
    uint64_t* arena_grow_count);
extern int vx_public_api_test_fill_engine_i32_tensor(
    VxExecutionContext* context, const char* name, int32_t value);
extern int vx_public_api_test_conv_cache_state(
    const VxExecutionContext* context, int node_index,
    int* transformed_weight, int* cpu_pack, int* cpu_indirection);
typedef void (*VxPublicApiCpuExecuteHook)(void* user_data);
extern void vx_public_api_test_set_cpu_execute_hook(
    VxPublicApiCpuExecuteHook hook, void* user_data);
extern int vx_public_api_test_runtime_coordinator_stats(
    const VxRuntime* runtime, size_t* active_requests,
    size_t* active_input_bytes, uint64_t* dispatches);
extern int vx_public_api_test_compiled_batch_contract(
    const VxCompiledModel* compiled, uint32_t* min_batch,
    uint32_t* max_batch, uint32_t* multiple_of, int32_t* batch_axis,
    int32_t* device_resident, uint64_t* device_epoch,
    const char** graph_fingerprint, const char** proof_identity);

typedef struct {
    const char* name;
    int enabled;
} NativeGpuCase;

static void count_physical_forward(void* user_data) {
    int* count = (int*)user_data;
    (*count)++;
}

static int report_token(const char* evidence, const char* key,
                        char* value, size_t value_capacity) {
    size_t key_length;
    const char* cursor;
    int found = 0;
    if (!evidence || !key || !key[0] || !value || value_capacity < 2u)
        return 0;
    key_length = strlen(key);
    cursor = evidence;
    while (*cursor) {
        const char* end = strchr(cursor, ';');
        size_t length = end ? (size_t)(end - cursor) : strlen(cursor);
        if (length > key_length + 1u &&
            !memcmp(cursor, key, key_length) && cursor[key_length] == '=') {
            size_t value_length = length - key_length - 1u;
            if (found || value_length >= value_capacity) return 0;
            memcpy(value, cursor + key_length + 1u, value_length);
            value[value_length] = '\0';
            found = 1;
        }
        if (!end) break;
        cursor = end + 1u;
    }
    return found;
}

static int report_u64_token(const VxReport* report, const char* key,
                            uint64_t* value) {
    char text[64];
    char* end = NULL;
    unsigned long long parsed;
    if (!report || !value ||
        !report_token(report->route_evidence, key, text, sizeof(text)))
        return 0;
    errno = 0;
    parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static int report_string_token(const VxReport* report, const char* key,
                               const char* expected) {
    char value[192];
    return report && expected &&
        report_token(report->route_evidence, key, value, sizeof(value)) &&
        !strcmp(value, expected);
}

static int report_vulkan_execution_evidence(const VxReport* report) {
    uint64_t arena = 0u;
    uint64_t staging = 0u;
    uint64_t coherent = 0u;
    uint64_t uploads = 0u;
    uint64_t downloads = 0u;
    return report &&
        report_string_token(report, "vk_mem", "device-local") &&
        report_u64_token(report, "vk_arena", &arena) && arena > 0u &&
        report_u64_token(report, "vk_stage", &staging) &&
        staging >= UINT64_C(32) * 1024u * 1024u &&
        report_u64_token(report, "vk_stage_coherent", &coherent) &&
        coherent <= 1u &&
        report_u64_token(report, "vk_up", &uploads) && uploads > 0u &&
        report_u64_token(report, "vk_down", &downloads) && downloads > 0u;
}

static int test_vulkan_duplicate_execution_evidence_rejected(void) {
    const char* duplicate =
        "vk_mem=device-local;vk_arena=1073741824;vk_stage=33554432;"
        "vk_stage_coherent=1;vk_up=;vk_down=1;vk_up=2";
    char value[64];
    CHECK(vx_native_batch_evidence_key_count(duplicate, "vk_up") == 2u);
    CHECK(!vx_native_batch_evidence_token(
        duplicate, "vk_up", value, sizeof(value)));
    return 0;
}

static int report_builtin_route_exact(const VxReport* report,
                                      const char* backend,
                                      uint64_t expected_nodes) {
    char provider[64];
    uint64_t nodes = 0u;
    uint64_t selected = 0u;
    uint64_t fallback = UINT64_MAX;
    uint64_t missing = UINT64_MAX;
    if (!report || !backend ||
        snprintf(provider, sizeof(provider), "builtin:%s", backend) <= 0)
        return 0;
    return report_string_token(report, "provider", provider) &&
        report_u64_token(report, "nodes", &nodes) &&
        nodes == expected_nodes &&
        report_u64_token(report, "selected", &selected) &&
        selected == expected_nodes &&
        report_u64_token(report, "fallback", &fallback) && fallback == 0u &&
        report_u64_token(report, "missing", &missing) && missing == 0u;
}

static int test_batch_matmul_plan_validation(void) {
    const int a_shape[3] = {2, 2, 3};
    const int b_shape[3] = {1, 3, 2};
    const int output_shape[3] = {2, 2, 2};
    const int bad_k_shape[3] = {1, 4, 2};
    const int bad_broadcast_shape[3] = {3, 3, 2};
    const int bad_broadcast_output[3] = {3, 2, 2};
    float a[12] = {0};
    float b[18] = {0};
    float output[12] = {0};
    VxBatchMatMulF32Plan plan;
    CHECK(vx_batch_matmul_f32_plan(
              a, a_shape, 3, b, b_shape, 3,
              output, output_shape, 3, &plan));
    CHECK(plan.batch_rank == 1u && plan.m == 2u && plan.k == 3u &&
          plan.n == 2u && plan.output_batches == 2u &&
          plan.metadata_words == 8u && plan.metadata[5] == 1u &&
          plan.metadata[6] == 6u && plan.metadata[7] == 0u);
    CHECK(!vx_batch_matmul_f32_plan(
              a, a_shape, 3, b, bad_k_shape, 3,
              output, output_shape, 3, &plan));
    CHECK(!vx_batch_matmul_f32_plan(
              a, a_shape, 3, b, bad_broadcast_shape, 3,
              output, bad_broadcast_output, 3, &plan));
    CHECK(!vx_batch_matmul_f32_plan(
              a, a_shape, 3, b, b_shape, 3,
              a, output_shape, 3, &plan));
    return 0;
}

static int write_text(const char* path, const char* text) {
    FILE* file = fopen(path, "wb");
    size_t length;
    size_t written;
    int close_status;
    if (!file) return -1;
    length = strlen(text);
    written = fwrite(text, 1u, length, file);
    close_status = fclose(file);
    return written == length && close_status == 0 ? 0 : -1;
}

static VxStatus compile_backend(VxRuntime* runtime,
                                const char* graph_path,
                                const char* weight_path,
                                const char* backend,
                                VxModel** model,
                                VxCompiledModel** compiled,
                                VxReport* report) {
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    const char* backends[1] = {backend};
    const char* weights[1] = {weight_path};
    VxStatus status;
    source.graph_path = graph_path;
    if (weight_path) {
        source.weight_paths = weights;
        source.weight_path_count = 1u;
    }
    status = vx_runtime_load_model(runtime, &source, model, report);
    if (status != VX_STATUS_OK) return status;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = backends;
    policy.backend_count = 1u;
    return vx_model_compile(*model, &policy, compiled, report);
}

static int execute_shape(VxExecutionContext* context,
                         const char* backend,
                         const float* values,
                         int64_t count,
                         int expect_cache_hit) {
    float output[4] = {0};
    VxTensorBinding inputs[1] = {
        {sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {count},
         values, (size_t)count * sizeof(float), VX_MEMORY_HOST},
    };
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status = vx_execution_context_execute(
        context, inputs, 1u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "%s execute shape=%" PRId64
                " failed: status=%s reason=%s message=%s route=%s\n",
                backend, count, vx_status_string(status), report.reason,
                report.message, report.route_evidence);
    CHECK(status == VX_STATUS_OK);
    CHECK(result && !strcmp(report.backend, backend) &&
          report.route_attested && !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none") &&
          strstr(report.route_evidence,
                 expect_cache_hit ? "shape_plan=hit" : "shape_plan=cold"));
    if (!strcmp(backend, "vulkan")) {
        CHECK(!strncmp(report.route_evidence, "vk_mem=device-local;",
                       strlen("vk_mem=device-local;")) &&
              report_vulkan_execution_evidence(&report));
    }
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "y") && info.dtype == VX_DTYPE_F32 &&
          info.rank == 1u && info.shape[0] == count &&
          info.byte_size == (size_t)count * sizeof(float));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    for (int64_t index = 0; index < count; index++)
        CHECK(output[index] > 0.0f && output[index] < values[index]);
    vx_result_release(result);
    return 0;
}

static int run_backend(VxRuntime* runtime,
                       const char* graph_path,
                       const char* backend) {
    const float small_a[1] = {3.0f};
    const float large[4] = {4.0f, 5.0f, 6.0f, 7.0f};
    const float small_b[1] = {8.0f};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    uint64_t generation = 0;
    uint64_t grow_count = 0;
    size_t capacity = 0;
    size_t high_water = 0;
    if (status == VX_STATUS_BACKEND_UNAVAILABLE) {
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 77;
    }
    if (status != VX_STATUS_OK) {
        fprintf(stderr, "%s compile failed: status=%s reason=%s message=%s "
                "route=%s\n", backend, vx_status_string(status),
                report.reason, report.message, report.route_evidence);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 1;
    }
    CHECK(compiled && !strcmp(report.backend, backend) &&
          report.route_attested && !report.operator_fallback_used &&
          strstr(report.route_evidence, "dynamic=1") &&
          strstr(report.route_evidence, "native_gpu_domain_spans=") &&
          strstr(report.route_evidence,
                 "native_gpu_storage_alignment=") &&
          strstr(report.route_evidence, "cuda_graph_") == NULL);
    if (!strcmp(backend, "vulkan"))
        CHECK(strstr(report.route_evidence,
                     "native_gpu_fixed_alloc=") != NULL);
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 &&
          generation == 0 && capacity > 0 && high_water == capacity &&
          grow_count == 1);
    CHECK(execute_shape(context, backend, small_a, 1, 0) == 0);
    CHECK(execute_shape(context, backend, large, 4, 0) == 0);
    CHECK(execute_shape(context, backend, small_b, 1, 1) == 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 &&
          generation == 3 && high_water == capacity && grow_count == 1);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int run_scheduled_batch_backend(VxRuntime* runtime,
                                       const char* graph_path,
                                       const char* backend) {
    const float first_value = 3.0f;
    const float second_value = 8.0f;
    VxTensorBinding first_binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {1},
        &first_value, sizeof(first_value), VX_MEMORY_HOST,
    };
    VxTensorBinding second_binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {1},
        &second_value, sizeof(second_value), VX_MEMORY_HOST,
    };
    VxRuntimeSubmitOptions submit = VX_RUNTIME_SUBMIT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxRequest* first = NULL;
    VxRequest* second = NULL;
    VxResult* first_result = NULL;
    VxResult* second_result = NULL;
    VxReport first_lane_report = VX_REPORT_INIT;
    VxReport second_lane_report = VX_REPORT_INIT;
    uint32_t min_batch = 0;
    uint32_t max_batch = 0;
    uint32_t multiple = 0;
    int32_t batch_axis = -1;
    int32_t device_resident = 0;
    uint64_t device_epoch = 0;
    uint64_t dispatches_before = 0;
    uint64_t dispatches_after = 0;
    uint64_t first_physical_execution = 0;
    uint64_t second_physical_execution = 0;
    uint64_t token = 0;
    const char* graph_fingerprint = NULL;
    const char* proof_identity = NULL;
    float first_output = 0.0f;
    float second_output = 0.0f;
    int physical_forwards = 0;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    if (status == VX_STATUS_BACKEND_UNAVAILABLE) {
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 77;
    }
    CHECK(status == VX_STATUS_OK && compiled &&
          vx_public_api_test_compiled_batch_contract(
              compiled, &min_batch, &max_batch, &multiple, &batch_axis,
              &device_resident, &device_epoch, &graph_fingerprint,
              &proof_identity) == 1 &&
          min_batch == 1u && max_batch == 4u && multiple == 1u &&
          batch_axis == 0 && device_resident == 1 && device_epoch == 1u &&
          graph_fingerprint && graph_fingerprint[0] && proof_identity &&
          strstr(proof_identity,
                 VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL));
    CHECK(vx_compiled_model_report(compiled, &report) == VX_STATUS_OK &&
          report_string_token(&report, "batchProtocol",
                              VX_BACKEND_BATCH_PROTOCOL) &&
          report_string_token(&report, "batchProof", proof_identity) &&
          report_u64_token(&report, "batchAxis", &token) && token == 0u &&
          report_u64_token(&report, "batchMin", &token) && token == 1u &&
          report_u64_token(&report, "batchMax", &token) && token == 4u &&
          report_u64_token(&report, "batchMultiple", &token) && token == 1u);
    CHECK(vx_public_api_test_runtime_coordinator_stats(
              runtime, NULL, NULL, &dispatches_before) >= 0);
    vx_public_api_test_set_cpu_execute_hook(
        count_physical_forward, &physical_forwards);
    CHECK(vx_runtime_submit(runtime, compiled, &first_binding, 1u, &submit,
                            &first, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_submit(runtime, compiled, &second_binding, 1u, &submit,
                            &second, &report) == VX_STATUS_OK);
    CHECK(vx_request_wait(first, VX_REQUEST_WAIT_INFINITE, &report) ==
          VX_STATUS_OK);
    CHECK(vx_request_wait(second, VX_REQUEST_WAIT_INFINITE, &report) ==
          VX_STATUS_OK);
    vx_public_api_test_set_cpu_execute_hook(NULL, NULL);
    CHECK(vx_request_result(first, &first_result, &first_lane_report) ==
              VX_STATUS_OK && first_result &&
          !strncmp(first_lane_report.route_evidence,
                   "trueBackendInvocations=1;batchSize=2;"
                   "physicalExecutionId=",
                   strlen("trueBackendInvocations=1;batchSize=2;"
                          "physicalExecutionId=")) &&
          strstr(first_lane_report.message, "trueBackendInvocations=1") &&
          report_u64_token(&first_lane_report, "trueBackendInvocations",
                           &token) && token == 1u &&
          report_u64_token(&first_lane_report, "batchSize", &token) &&
          token == 2u &&
          report_u64_token(&first_lane_report, "physicalExecutionId",
                           &first_physical_execution) &&
          first_physical_execution != 0u &&
          report_string_token(&first_lane_report, "batchProtocol",
                              VX_BACKEND_BATCH_PROTOCOL) &&
          report_string_token(&first_lane_report, "batchProof",
                              proof_identity) &&
          report_builtin_route_exact(&first_lane_report, backend, 1u));
    CHECK(vx_request_result(second, &second_result, &second_lane_report) ==
              VX_STATUS_OK && second_result &&
          strstr(second_lane_report.message, "trueBackendInvocations=1") &&
          report_u64_token(&second_lane_report, "trueBackendInvocations",
                           &token) && token == 1u &&
          report_u64_token(&second_lane_report, "batchSize", &token) &&
          token == 2u &&
          report_u64_token(&second_lane_report, "physicalExecutionId",
                           &second_physical_execution) &&
          second_physical_execution == first_physical_execution &&
          report_string_token(&second_lane_report, "batchProtocol",
                              VX_BACKEND_BATCH_PROTOCOL) &&
          report_string_token(&second_lane_report, "batchProof",
                              proof_identity) &&
          report_builtin_route_exact(&second_lane_report, backend, 1u));
    if (!strcmp(backend, "vulkan")) {
        const char* first_memory = strstr(
            first_lane_report.route_evidence, ";vk_mem=device-local;");
        const char* first_contract = strstr(
            first_lane_report.route_evidence, ";batchProtocol=");
        const char* first_provider = strstr(
            first_lane_report.route_evidence, ";provider=builtin:vulkan;");
        const char* second_memory = strstr(
            second_lane_report.route_evidence, ";vk_mem=device-local;");
        const char* second_contract = strstr(
            second_lane_report.route_evidence, ";batchProtocol=");
        const char* second_provider = strstr(
            second_lane_report.route_evidence, ";provider=builtin:vulkan;");
        if (!report_vulkan_execution_evidence(&first_lane_report) ||
            !report_vulkan_execution_evidence(&second_lane_report) ||
            !first_memory || !first_contract || !first_provider ||
            !second_memory || !second_contract || !second_provider)
            fprintf(stderr, "vulkan scheduled evidence: first=%s\nsecond=%s\n",
                    first_lane_report.route_evidence,
                    second_lane_report.route_evidence);
        CHECK(report_vulkan_execution_evidence(&first_lane_report) &&
              report_vulkan_execution_evidence(&second_lane_report) &&
              first_memory && first_contract && first_provider &&
              first_memory < first_contract && first_contract < first_provider &&
              second_memory && second_contract && second_provider &&
              second_memory < second_contract &&
              second_contract < second_provider);
    }
    CHECK(vx_result_read(first_result, "y", &first_output,
                         sizeof(first_output), NULL, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(second_result, "y", &second_output,
                         sizeof(second_output), NULL, &report) == VX_STATUS_OK);
    CHECK(fabsf(first_output - first_value /
                    (1.0f + expf(-first_value))) <= 2.0e-5f);
    CHECK(fabsf(second_output - second_value /
                    (1.0f + expf(-second_value))) <= 2.0e-5f);
    CHECK(physical_forwards == 1);
    CHECK(vx_public_api_test_runtime_coordinator_stats(
              runtime, NULL, NULL, &dispatches_after) == 1 &&
          dispatches_after == dispatches_before + 1u);
    vx_result_release(first_result);
    vx_result_release(second_result);
    vx_request_release(first);
    vx_request_release(second);
    first = NULL;
    second = NULL;
    first_result = NULL;
    second_result = NULL;
    vx_public_api_test_set_cpu_execute_hook(
        count_physical_forward, &physical_forwards);
    CHECK(vx_runtime_submit(runtime, compiled, &first_binding, 1u, &submit,
                            &first, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_submit(runtime, compiled, &second_binding, 1u, &submit,
                            &second, &report) == VX_STATUS_OK);
    CHECK(vx_request_wait(first, VX_REQUEST_WAIT_INFINITE, &report) ==
          VX_STATUS_OK);
    CHECK(vx_request_wait(second, VX_REQUEST_WAIT_INFINITE, &report) ==
          VX_STATUS_OK);
    vx_public_api_test_set_cpu_execute_hook(NULL, NULL);
    CHECK(vx_request_result(first, &first_result, &first_lane_report) ==
              VX_STATUS_OK && first_result &&
          report_u64_token(&first_lane_report, "batchSize", &token) &&
          token == 2u &&
          report_u64_token(&first_lane_report, "physicalExecutionId",
                           &second_physical_execution) &&
          second_physical_execution != 0u &&
          second_physical_execution != first_physical_execution);
    CHECK(vx_request_result(second, &second_result, &second_lane_report) ==
              VX_STATUS_OK && second_result &&
          report_u64_token(&second_lane_report, "physicalExecutionId",
                           &token) && token == second_physical_execution);
    CHECK(physical_forwards == 2);
    CHECK(vx_public_api_test_runtime_coordinator_stats(
              runtime, NULL, NULL, &dispatches_after) == 1 &&
          dispatches_after == dispatches_before + 2u);
    vx_result_release(first_result);
    vx_result_release(second_result);
    vx_request_release(first);
    vx_request_release(second);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int expect_independent_batch_contract_backend(
        VxRuntime* runtime, const char* graph_path,
        const char* weight_path, const char* backend) {
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    uint32_t min_batch = 0;
    uint32_t max_batch = 0;
    uint32_t multiple = 0;
    int32_t batch_axis = -1;
    const char* graph_fingerprint = NULL;
    const char* proof_identity = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, weight_path, backend,
        &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled &&
          vx_public_api_test_compiled_batch_contract(
              compiled, &min_batch, &max_batch, &multiple, &batch_axis,
              NULL, NULL, &graph_fingerprint, &proof_identity) == 1 &&
          min_batch == 1u && max_batch == 4u && multiple == 1u &&
          batch_axis == 0 && graph_fingerprint && proof_identity &&
          strstr(proof_identity,
                 VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL));
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int run_unproved_batch_fail_closed_backend(
        VxRuntime* runtime, const char* graph_path, const char* backend) {
    const float first_values[2] = {1.0f, 2.0f};
    const float second_values[2] = {10.0f, 20.0f};
    VxTensorBinding first_binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 2u, {1, 2},
        first_values, sizeof(first_values), VX_MEMORY_HOST,
    };
    VxTensorBinding second_binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 2u, {1, 2},
        second_values, sizeof(second_values), VX_MEMORY_HOST,
    };
    VxRuntimeSubmitOptions submit = VX_RUNTIME_SUBMIT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxRequest* first = NULL;
    VxRequest* second = NULL;
    VxResult* first_result = NULL;
    VxResult* second_result = NULL;
    uint32_t max_batch = 0;
    uint64_t dispatches_before = 0;
    uint64_t dispatches_after = 0;
    float first_output[2] = {0};
    float second_output[2] = {0};
    int physical_forwards = 0;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled &&
          vx_public_api_test_compiled_batch_contract(
              compiled, NULL, &max_batch, NULL, NULL, NULL, NULL,
              NULL, NULL) == 0 && max_batch == 1u);
    CHECK(vx_public_api_test_runtime_coordinator_stats(
              runtime, NULL, NULL, &dispatches_before) >= 0);
    vx_public_api_test_set_cpu_execute_hook(
        count_physical_forward, &physical_forwards);
    CHECK(vx_runtime_submit(runtime, compiled, &first_binding, 1u, &submit,
                            &first, &report) == VX_STATUS_OK);
    CHECK(vx_runtime_submit(runtime, compiled, &second_binding, 1u, &submit,
                            &second, &report) == VX_STATUS_OK);
    CHECK(vx_request_wait(first, VX_REQUEST_WAIT_INFINITE, &report) ==
          VX_STATUS_OK);
    CHECK(vx_request_wait(second, VX_REQUEST_WAIT_INFINITE, &report) ==
          VX_STATUS_OK);
    vx_public_api_test_set_cpu_execute_hook(NULL, NULL);
    CHECK(vx_request_result(first, &first_result, &report) == VX_STATUS_OK &&
          first_result);
    CHECK(vx_request_result(second, &second_result, &report) == VX_STATUS_OK &&
          second_result);
    CHECK(vx_result_read(first_result, "y", first_output,
                         sizeof(first_output), NULL, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(second_result, "y", second_output,
                         sizeof(second_output), NULL, &report) == VX_STATUS_OK);
    CHECK(!memcmp(first_output, first_values, sizeof(first_output)) &&
          !memcmp(second_output, second_values, sizeof(second_output)) &&
          physical_forwards == 2);
    CHECK(vx_public_api_test_runtime_coordinator_stats(
              runtime, NULL, NULL, &dispatches_after) == 1 &&
          dispatches_after == dispatches_before + 2u);
    vx_result_release(first_result);
    vx_result_release(second_result);
    vx_request_release(first);
    vx_request_release(second);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int write_quantize_weights(const char* path) {
    const int shape[1] = {1};
    const float scale = 0.5f;
    const int8_t zero_point = 0;
    SafetensorsFile weights;
    int status;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(
            &weights, "scale", SAFETENSORS_DTYPE_F32,
            shape, 1, &scale, sizeof(scale)) != 0 ||
        safetensors_add_tensor(
            &weights, "zero_point", SAFETENSORS_DTYPE_I8,
            shape, 1, &zero_point, sizeof(zero_point)) != 0) {
        safetensors_free(&weights);
        return -1;
    }
    status = safetensors_save(path, &weights);
    safetensors_free(&weights);
    return status;
}

static int write_linear_weights(const char* path) {
    const int input_major_shape[2] = {3, 2};
    const int output_major_shape[2] = {2, 3};
    const int bias_shape[1] = {2};
    const float input_major[6] = {
        1.0f, -2.0f,
        0.5f, 3.0f,
        -1.0f, 4.0f,
    };
    const float output_major[6] = {
        1.0f, 0.5f, -1.0f,
        -2.0f, 3.0f, 4.0f,
    };
    const float bias[2] = {0.25f, -0.5f};
    SafetensorsFile weights;
    int status;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(
            &weights, "weight.input_major", SAFETENSORS_DTYPE_F32,
            input_major_shape, 2, input_major, sizeof(input_major)) != 0 ||
        safetensors_add_tensor(
            &weights, "weight.output_major", SAFETENSORS_DTYPE_F32,
            output_major_shape, 2, output_major,
            sizeof(output_major)) != 0 ||
        safetensors_add_tensor(
            &weights, "bias", SAFETENSORS_DTYPE_F32,
            bias_shape, 1, bias, sizeof(bias)) != 0) {
        safetensors_free(&weights);
        return -1;
    }
    status = safetensors_save(path, &weights);
    safetensors_free(&weights);
    return status;
}

static int write_linear_f16_rejection_weights(const char* path) {
    const int weight_shape[2] = {3, 2};
    const int bias_shape[1] = {2};
    const uint16_t f16_weight[6] = {
        UINT16_C(0x3c00), UINT16_C(0xc000),
        UINT16_C(0x3800), UINT16_C(0x4200),
        UINT16_C(0xbc00), UINT16_C(0x4400),
    };
    const float f32_weight[6] = {
        1.0f, -2.0f,
        0.5f, 3.0f,
        -1.0f, 4.0f,
    };
    const uint16_t f16_bias[2] = {
        UINT16_C(0x3400), UINT16_C(0xb800),
    };
    SafetensorsFile weights;
    int status;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(
            &weights, "weight.f16", SAFETENSORS_DTYPE_F16,
            weight_shape, 2, f16_weight, sizeof(f16_weight)) != 0 ||
        safetensors_add_tensor(
            &weights, "weight.f32", SAFETENSORS_DTYPE_F32,
            weight_shape, 2, f32_weight, sizeof(f32_weight)) != 0 ||
        safetensors_add_tensor(
            &weights, "bias.f16", SAFETENSORS_DTYPE_F16,
            bias_shape, 1, f16_bias, sizeof(f16_bias)) != 0) {
        safetensors_free(&weights);
        return -1;
    }
    status = safetensors_save(path, &weights);
    safetensors_free(&weights);
    return status;
}

static int write_embedding_weights(const char* path) {
    const int shape[2] = {3, 2};
    const float table[6] = {
        10.0f, 11.0f,
        20.0f, 21.0f,
        30.0f, 31.0f,
    };
    SafetensorsFile weights;
    int status;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(
            &weights, "table", SAFETENSORS_DTYPE_F32,
            shape, 2, table, sizeof(table)) != 0) {
        safetensors_free(&weights);
        return -1;
    }
    status = safetensors_save(path, &weights);
    safetensors_free(&weights);
    return status;
}

static int write_conv_f16_weights(const char* path) {
    const int shape[4] = {1, 1, 1, 1};
    const uint16_t weight[1] = {UINT16_C(0x4400)}; /* 4.0 */
    SafetensorsFile weights;
    int status;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(
            &weights, "conv.f16", SAFETENSORS_DTYPE_F16,
            shape, 4, weight, sizeof(weight)) != 0) {
        safetensors_free(&weights);
        return -1;
    }
    status = safetensors_save(path, &weights);
    safetensors_free(&weights);
    return status;
}

static int execute_linear_shape(VxExecutionContext* context,
                                const char* backend,
                                const float* input,
                                int64_t rows,
                                int expect_cache_hit) {
    const float input_major[6] = {
        1.0f, -2.0f,
        0.5f, 3.0f,
        -1.0f, 4.0f,
    };
    const float bias[2] = {0.25f, -0.5f};
    float expected_without_bias[8] = {0};
    float with_bias[8] = {0};
    float without_bias[8] = {0};
    VxTensorBinding binding = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 2u, {rows, 3},
        input, (size_t)rows * 3u * sizeof(float), VX_MEMORY_HOST,
    };
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    for (int64_t row = 0; row < rows; row++) {
        for (int column = 0; column < 2; column++) {
            float sum = 0.0f;
            for (int k = 0; k < 3; k++) {
                sum += input[row * 3 + k] * input_major[k * 2 + column];
            }
            expected_without_bias[row * 2 + column] = sum;
        }
    }
    status = vx_execution_context_execute(
        context, &binding, 1u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "%s Linear execute rows=%" PRId64
                " failed: status=%s reason=%s message=%s route=%s\n",
                backend, rows, vx_status_string(status), report.reason,
                report.message, report.route_evidence);
    CHECK(status == VX_STATUS_OK && result && report.route_attested &&
          !report.operator_fallback_used &&
          strstr(report.route_evidence,
                 expect_cache_hit ? "shape_plan=hit" : "shape_plan=cold"));
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "with_bias") && info.dtype == VX_DTYPE_F32 &&
          info.rank == 2u && info.shape[0] == rows && info.shape[1] == 2 &&
          info.byte_size == (size_t)rows * 2u * sizeof(float));
    CHECK(vx_result_read(result, "with_bias", with_bias,
                         sizeof(with_bias), NULL, &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "without_bias", without_bias,
                         sizeof(without_bias), NULL, &report) == VX_STATUS_OK);
    for (int64_t index = 0; index < rows * 2; index++) {
        int column = (int)(index % 2);
        CHECK(fabsf(without_bias[index] - expected_without_bias[index]) <=
              1.0e-4f);
        CHECK(fabsf(with_bias[index] -
                    (expected_without_bias[index] + bias[column])) <=
              1.0e-4f);
    }
    vx_result_release(result);
    return 0;
}

static int run_linear_backend(VxRuntime* runtime,
                              const char* graph_path,
                              const char* weight_path,
                              const char* backend) {
    const float small_a[3] = {1.0f, 2.0f, -1.0f};
    const float large[12] = {
        1.0f, 0.0f, 2.0f,
        -1.0f, 3.0f, 0.5f,
        2.0f, -2.0f, 1.0f,
        0.25f, 0.5f, -0.75f,
    };
    const float small_b[3] = {-3.0f, 1.0f, 2.0f};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, weight_path, backend,
        &model, &compiled, &report);
    if (status == VX_STATUS_BACKEND_UNAVAILABLE) {
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 77;
    }
    if (status != VX_STATUS_OK) {
        fprintf(stderr, "%s Linear compile failed: status=%s reason=%s "
                "message=%s route=%s\n", backend,
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 1;
    }
    CHECK(report.route_attested && !report.operator_fallback_used &&
          strstr(report.route_evidence, "dynamic=1") &&
          strstr(report.route_evidence, "native_gpu_domain_spans="));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(execute_linear_shape(context, backend, small_a, 1, 0) == 0);
    CHECK(execute_linear_shape(context, backend, large, 4, 0) == 0);
    CHECK(execute_linear_shape(context, backend, small_b, 1, 1) == 0);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int reject_f16_linear_backend(VxRuntime* runtime,
                                     const char* graph_path,
                                     const char* weight_path,
                                     const char* backend,
                                     const char* rejected_role) {
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, weight_path, backend,
        &model, &compiled, &report);
    if (status != VX_STATUS_BACKEND_UNSUPPORTED)
        fprintf(stderr, "%s F16 Linear %s unexpectedly compiled: "
                "status=%s reason=%s message=%s route=%s\n",
                backend, rejected_role, vx_status_string(status),
                report.reason, report.message, report.route_evidence);
    CHECK(status == VX_STATUS_BACKEND_UNSUPPORTED && compiled == NULL &&
          !strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED") &&
          strstr(report.route_evidence,
                 "required=native-gpu-full-bounded-domain"));
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int run_odd_byte_backend(VxRuntime* runtime,
                                const char* graph_path,
                                const char* weight_path,
                                const char* backend) {
    const float values[5] = {0.0f, 1.0f, -1.0f, 2.0f, -2.0f};
    const int8_t expected[5] = {0, 2, -2, 4, -4};
    int8_t output[5] = {0};
    VxTensorBinding input = {
        sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {5},
        values, sizeof(values), VX_MEMORY_HOST,
    };
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* result = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, weight_path, backend,
        &model, &compiled, &report);
    if (status == VX_STATUS_BACKEND_UNAVAILABLE) {
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 77;
    }
    if (status != VX_STATUS_OK) {
        fprintf(stderr, "%s odd-byte compile failed: status=%s reason=%s "
                "message=%s route=%s\n", backend, vx_status_string(status),
                report.reason, report.message, report.route_evidence);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 1;
    }
    /* Maximum logical bytes are 20 + 5, but the byte output needs an 8-byte
     * packed span and the second lifetime starts at the next 64-byte host
     * arena boundary: the exact fixed host arena is therefore 72 bytes. */
    if (!strstr(report.route_evidence, "native_gpu_domain_bytes=72"))
        fprintf(stderr, "%s odd-byte resource evidence: %s\n",
                backend, report.route_evidence);
    CHECK(strstr(report.route_evidence, "native_gpu_domain_bytes=72") &&
          strstr(report.route_evidence,
                 "native_gpu_storage_alignment="));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    status = vx_execution_context_execute(
        context, &input, 1u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "%s odd-byte execute failed: status=%s reason=%s "
                "message=%s route=%s\n", backend, vx_status_string(status),
                report.reason, report.message, report.route_evidence);
    CHECK(status == VX_STATUS_OK && result &&
          !report.operator_fallback_used);
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(memcmp(output, expected, sizeof(expected)) == 0);
    vx_result_release(result);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_typed_batch_shape(VxExecutionContext* context,
                                     const char* backend,
                                     int64_t width,
                                     int expect_cache_hit) {
    static const int32_t mask_small[1] = {INT32_MIN};
    static const int32_t mask_large[3] = {
        INT32_MIN, INT32_C(2143289345), -1,
    };
    static const float a_small[2] = {2.0f, 3.0f};
    static const float b_small[4] = {4.0f, 5.0f, 6.0f, 7.0f};
    static const float expected_small[4] = {8.0f, 10.0f, 18.0f, 21.0f};
    static const float a_large[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    static const float b_large[12] = {
        1.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,
        2.0f, 1.0f, 1.0f, 2.0f, 3.0f, 1.0f,
    };
    static const float expected_large[4] = {4.0f, 5.0f, 31.0f, 20.0f};
    const int32_t* mask = width == 1 ? mask_small : mask_large;
    const float* a = width == 1 ? a_small : a_large;
    const float* b = width == 1 ? b_small : b_large;
    const float* expected = width == 1 ? expected_small : expected_large;
    int32_t expanded[6] = {0};
    float product[4] = {0};
    VxTensorBinding inputs[3] = {
        {sizeof(VxTensorBinding), "mask", VX_DTYPE_I32, 4u,
         {1, 1, 1, width}, mask, (size_t)width * sizeof(*mask),
         VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 4u,
         {1, 2, 1, width}, a, (size_t)(2 * width) * sizeof(*a),
         VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 4u,
         {1, 2, width, 2}, b, (size_t)(4 * width) * sizeof(*b),
         VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status = vx_execution_context_execute(
        context, inputs, 3u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "%s typed batch execute shape=%" PRId64
                " failed: status=%s reason=%s message=%s route=%s\n",
                backend, width, vx_status_string(status), report.reason,
                report.message, report.route_evidence);
    CHECK(status == VX_STATUS_OK && result &&
          !report.operator_fallback_used &&
          strstr(report.route_evidence,
                 expect_cache_hit ? "shape_plan=hit" : "shape_plan=cold"));
    CHECK(vx_result_read(result, "expanded", expanded,
                         (size_t)(2 * width) * sizeof(*expanded), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(vx_result_read(result, "product", product, sizeof(product), NULL,
                         &report) == VX_STATUS_OK);
    for (int64_t batch = 0; batch < 2; batch++)
        CHECK(memcmp(expanded + batch * width, mask,
                     (size_t)width * sizeof(*mask)) == 0);
    CHECK(memcmp(product, expected, sizeof(product)) == 0);
    vx_result_release(result);
    return 0;
}

static int run_typed_batch_backend(VxRuntime* runtime,
                                   const char* graph_path,
                                   const char* backend) {
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    if (status == VX_STATUS_BACKEND_UNAVAILABLE) {
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 77;
    }
    if (status != VX_STATUS_OK) {
        fprintf(stderr, "%s typed batch compile failed: status=%s reason=%s "
                "message=%s route=%s\n", backend, vx_status_string(status),
                report.reason, report.message, report.route_evidence);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 1;
    }
    CHECK(!report.operator_fallback_used &&
          strstr(report.route_evidence, "dynamic=1"));
    if (!strcmp(backend, "vulkan")) {
        const char* scratch = strstr(
            report.route_evidence, "native_gpu_fixed_scratch=");
        uint64_t required = 0u;
        uint64_t validation = 0u;
        uint64_t limit = 0u;
        int parsed = scratch ? sscanf(
            scratch,
            "native_gpu_fixed_scratch=%" SCNu64 "/%" SCNu64
            "/%" SCNu64,
            &required, &validation, &limit) : 0;
        if (parsed != 3 || !validation || required < validation ||
            required > limit)
            fprintf(stderr, "vulkan scratch evidence: %s\n",
                    report.route_evidence);
        CHECK(parsed == 3 && validation > 0u && required >= validation &&
              required <= limit);
    }
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(execute_typed_batch_shape(context, backend, 1, 0) == 0);
    CHECK(execute_typed_batch_shape(context, backend, 3, 0) == 0);
    CHECK(execute_typed_batch_shape(context, backend, 1, 1) == 0);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_gather_values(VxExecutionContext* context,
                                 const float* data,
                                 int64_t data_count,
                                 const int32_t indices[2],
                                 VxStatus expected_status,
                                 const float expected[2]) {
    unsigned char index_storage[1u + 2u * sizeof(int32_t)];
    float output[2] = {0.0f, 0.0f};
    VxTensorBinding inputs[2] = {
        {sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {data_count},
         data, (size_t)data_count * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "indices", VX_DTYPE_I32, 1u, {2},
         index_storage + 1u, 2u * sizeof(int32_t), VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    memcpy(index_storage + 1u, indices, 2u * sizeof(int32_t));
    status = vx_execution_context_execute(
        context, inputs, 2u, &result, &report);
    if (status != expected_status)
        fprintf(stderr, "Gather execute expected=%s actual=%s reason=%s "
                "message=%s route=%s\n",
                vx_status_string(expected_status), vx_status_string(status),
                report.reason, report.message, report.route_evidence);
    CHECK(status == expected_status);
    if (expected_status != VX_STATUS_OK) {
        CHECK(result == NULL && !strcmp(report.reason,
                                        "INVALID_INPUT_VALUES"));
        return 0;
    }
    CHECK(result && report.route_attested && !report.operator_fallback_used);
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(fabsf(output[0] - expected[0]) <= 1.0e-4f &&
          fabsf(output[1] - expected[1]) <= 1.0e-4f);
    vx_result_release(result);
    return 0;
}

static float silu_reference(float value) {
    return value / (1.0f + expf(-value));
}

static int run_gather_preflight_backend(VxRuntime* runtime,
                                        const char* graph_path,
                                        const char* backend) {
    const float small[2] = {1.0f, 2.0f};
    const float large[4] = {3.0f, 4.0f, 5.0f, 6.0f};
    const int32_t too_negative[2] = {-3, 0};
    const int32_t too_large[2] = {2, 0};
    const int32_t valid_negative[2] = {-1, 0};
    const int32_t valid_large[2] = {-4, 3};
    const float expected_small[2] = {
        silu_reference(2.0f), silu_reference(1.0f),
    };
    const float expected_large[2] = {
        silu_reference(3.0f), silu_reference(6.0f),
    };
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    uint64_t generation = 0u;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled && report.route_attested);
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, NULL, NULL, NULL) == 0 &&
          generation == 0u);

    /* Both failures happen before the preceding SiLU can dispatch and before
     * a shape plan/input binding can publish. The deliberately unaligned I32
     * binding also exercises the byte-safe preflight load. */
    CHECK(execute_gather_values(
              context, small, 2, too_negative,
              VX_STATUS_INVALID_ARGUMENT, NULL) == 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, NULL, NULL, NULL) == 0 &&
          generation == 0u);
    CHECK(execute_gather_values(
              context, small, 2, too_large,
              VX_STATUS_INVALID_ARGUMENT, NULL) == 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, NULL, NULL, NULL) == 0 &&
          generation == 0u);
    CHECK(execute_gather_values(
              context, small, 2, valid_negative,
              VX_STATUS_OK, expected_small) == 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, NULL, NULL, NULL) == 0 &&
          generation == 1u);
    CHECK(execute_gather_values(
              context, large, 4, valid_large,
              VX_STATUS_OK, expected_large) == 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, NULL, NULL, NULL) == 0 &&
          generation == 2u);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int reject_unsafe_gather_origin(VxRuntime* runtime,
                                       const char* graph_path,
                                       const char* backend) {
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    if (status != VX_STATUS_BACKEND_UNSUPPORTED || compiled != NULL ||
        strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED"))
        fprintf(stderr, "%s unsafe Gather origin compile status=%s "
                "reason=%s message=%s route=%s\n", backend,
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    CHECK(status == VX_STATUS_BACKEND_UNSUPPORTED && compiled == NULL &&
          !strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED"));
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_argmax_embedding(VxExecutionContext* context,
                                    const float logits[6],
                                    const float expected[4]) {
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VxTensorBinding input = {
        sizeof(VxTensorBinding), "logits", VX_DTYPE_F32, 2u, {2, 3},
        logits, 6u * sizeof(float), VX_MEMORY_HOST,
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status = vx_execution_context_execute(
        context, &input, 1u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "ArgMax->Embedding execute failed status=%s "
                "reason=%s message=%s route=%s\n",
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    CHECK(status == VX_STATUS_OK && result && report.route_attested &&
          !report.operator_fallback_used);
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    vx_result_release(result);
    return 0;
}

static int run_argmax_embedding_backend(VxRuntime* runtime,
                                        const char* graph_path,
                                        const char* weight_path,
                                        const char* backend) {
    const float first_logits[6] = {0, 1, 2, 4, 3, 2};
    const float second_logits[6] = {4, 5, 6, 0, 7, 1};
    const float first_expected[4] = {30, 31, 10, 11};
    const float second_expected[4] = {30, 31, 20, 21};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, weight_path, backend,
        &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled && report.route_attested);
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(execute_argmax_embedding(
              context, first_logits, first_expected) == 0);
    /* Poison only the CPU mirror. The second ArgMax authors current IDs on
     * device; bounded GPU Embedding must consume those IDs without scanning
     * this deliberately stale invalid host payload or falling back. */
    CHECK(vx_public_api_test_fill_engine_i32_tensor(
              context, "ids", 99) == 0);
    CHECK(execute_argmax_embedding(
              context, second_logits, second_expected) == 0);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_public_embedding_weight(
        VxExecutionContext* context, const int32_t* ids, int64_t token_count,
        const float table[6], const float* expected) {
    float output[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    VxTensorBinding inputs[2] = {
        {sizeof(VxTensorBinding), "ids", VX_DTYPE_I32, 1u, {token_count},
         ids, (size_t)token_count * sizeof(*ids), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "table", VX_DTYPE_F32, 2u, {3, 2},
         table, 6u * sizeof(*table), VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status = vx_execution_context_execute(
        context, inputs, 2u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "public Embedding weight execute failed status=%s "
                "reason=%s message=%s route=%s\n",
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    CHECK(status == VX_STATUS_OK && result && report.route_attested &&
          !report.operator_fallback_used);
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(memcmp(output, expected,
                 (size_t)token_count * 2u * sizeof(float)) == 0);
    vx_result_release(result);
    return 0;
}

static int run_public_embedding_weight_backend(
        VxRuntime* runtime, const char* graph_path, const char* backend) {
    const int32_t two_ids[2] = {2, 0};
    const int32_t one_id[1] = {1};
    const float first_table[6] = {10, 11, 20, 21, 30, 31};
    const float second_table[6] = {110, 111, 120, 121, 130, 131};
    const float third_table[6] = {210, 211, 220, 221, 230, 231};
    const float first_expected[4] = {30, 31, 10, 11};
    const float second_expected[4] = {130, 131, 110, 111};
    const float third_expected[2] = {220, 221};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled && report.route_attested &&
          strstr(report.route_evidence, "dynamic=1"));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(execute_public_embedding_weight(
              context, two_ids, 2, first_table, first_expected) == 0);
    /* Same shape, different table: the public span must stay mutable instead
     * of being promoted to an immutable backend weight after its first use. */
    CHECK(execute_public_embedding_weight(
              context, two_ids, 2, second_table, second_expected) == 0);
    CHECK(execute_public_embedding_weight(
              context, one_id, 1, third_table, third_expected) == 0);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static void reference_layernorm(const float* input, const float gamma[2],
                                const float beta[2], float* output,
                                int64_t rows) {
    const float epsilon = 1.0e-5f;
    for (int64_t row = 0; row < rows; row++) {
        const float* values = input + row * 2;
        float mean = (values[0] + values[1]) * 0.5f;
        float delta0 = values[0] - mean;
        float delta1 = values[1] - mean;
        float inverse = 1.0f / sqrtf(
            (delta0 * delta0 + delta1 * delta1) * 0.5f + epsilon);
        output[row * 2] = delta0 * inverse * gamma[0] + beta[0];
        output[row * 2 + 1] = delta1 * inverse * gamma[1] + beta[1];
    }
}

static int execute_public_layernorm_affines(
        VxExecutionContext* context, const float* input, int64_t rows,
        const float gamma[2], const float beta[2]) {
    float expected[4] = {0};
    float output[4] = {0};
    VxTensorBinding inputs[3] = {
        {sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 2u, {rows, 2},
         input, (size_t)rows * 2u * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "gamma", VX_DTYPE_F32, 1u, {2},
         gamma, 2u * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "beta", VX_DTYPE_F32, 1u, {2},
         beta, 2u * sizeof(float), VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    reference_layernorm(input, gamma, beta, expected, rows);
    status = vx_execution_context_execute(
        context, inputs, 3u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "public LayerNorm affine execute failed status=%s "
                "reason=%s message=%s route=%s\n",
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    CHECK(status == VX_STATUS_OK && result && report.route_attested &&
          !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none"));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    for (int64_t index = 0; index < rows * 2; index++)
        CHECK(fabsf(output[index] - expected[index]) <= 2.0e-4f);
    vx_result_release(result);
    return 0;
}

static int run_public_layernorm_affines_backend(
        VxRuntime* runtime, const char* graph_path, const char* backend) {
    const float minimum_input[2] = {1.0f, 3.0f};
    const float maximum_input[4] = {2.0f, 6.0f, -4.0f, 8.0f};
    const float first_gamma[2] = {1.0f, 1.0f};
    const float first_beta[2] = {0.0f, 0.0f};
    const float second_gamma[2] = {2.0f, 3.0f};
    const float second_beta[2] = {10.0f, 20.0f};
    const float third_gamma[2] = {-1.0f, 0.5f};
    const float third_beta[2] = {-7.0f, 9.0f};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    uint64_t generation = 0;
    uint64_t grow_count = 0;
    size_t capacity = 0;
    size_t initial_capacity = 0;
    size_t high_water = 0;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled && report.route_attested &&
          strstr(report.route_evidence, "dynamic=1"));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 && generation == 0 && capacity > 0 &&
          capacity == high_water && grow_count == 1);
    initial_capacity = capacity;
    CHECK(execute_public_layernorm_affines(
              context, minimum_input, 1,
              first_gamma, first_beta) == 0);
    CHECK(execute_public_layernorm_affines(
              context, maximum_input, 2,
              second_gamma, second_beta) == 0);
    /* Returning to the cached minimum shape with new affine payloads proves
     * the span was neither frozen nor promoted into the immutable cache. */
    CHECK(execute_public_layernorm_affines(
              context, minimum_input, 1,
              third_gamma, third_beta) == 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 && generation == 3 && capacity > 0 &&
          capacity == initial_capacity && capacity == high_water &&
          grow_count == 1);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_conv_weight(
        VxExecutionContext* context, const float* input, int64_t batch,
        const float* public_weight, float expected_scale) {
    float output[8] = {0};
    VxTensorBinding bindings[2] = {
        {sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 4u,
         {batch, 2, 2, 1}, input,
         (size_t)batch * 4u * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "weight", VX_DTYPE_F32, 4u,
         {1, 1, 1, 1}, public_weight,
         public_weight ? sizeof(float) : 0u, VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    size_t binding_count = public_weight ? 2u : 1u;
    VxStatus status = vx_execution_context_execute(
        context, bindings, binding_count, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "public Conv weight execute failed status=%s "
                "reason=%s message=%s route=%s\n",
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    CHECK(status == VX_STATUS_OK && result && report.route_attested &&
          !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none"));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    for (int64_t index = 0; index < batch * 4; index++)
        CHECK(fabsf(output[index] - input[index] * expected_scale) <= 1.0e-5f);
    vx_result_release(result);
    return 0;
}

static int run_public_conv_weight_backend(
        VxRuntime* runtime, const char* graph_path, const char* backend) {
    const float min_input[4] = {1, 2, 3, 4};
    const float max_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float weight_a[1] = {2.0f};
    const float weight_b[1] = {-3.0f};
    const float weight_c[1] = {5.0f};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    uint64_t generation = 0;
    uint64_t grow_count = 0;
    size_t capacity = 0;
    size_t initial_capacity = 0;
    size_t high_water = 0;
    int transformed = -1;
    int cpu_pack = -1;
    int cpu_indirection = -1;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled && report.route_attested &&
          strstr(report.route_evidence, "dynamic=1"));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 && generation == 0 && capacity > 0 &&
          capacity == high_water && grow_count == 1);
    initial_capacity = capacity;
    CHECK(execute_conv_weight(
              context, min_input, 1, weight_a, weight_a[0]) == 0);
    CHECK(execute_conv_weight(
              context, max_input, 2, weight_b, weight_b[0]) == 0);
    CHECK(execute_conv_weight(
              context, min_input, 1, weight_c, weight_c[0]) == 0);
    CHECK(vx_public_api_test_conv_cache_state(
              context, 0, &transformed, &cpu_pack,
              &cpu_indirection) == 0 && !transformed && !cpu_pack &&
          !cpu_indirection);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 && generation == 3 &&
          capacity == initial_capacity && high_water == capacity &&
          grow_count == 1);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int reject_transformed_public_conv_weight(
        VxRuntime* runtime, const char* graph_path, const char* backend) {
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    if (status != VX_STATUS_BACKEND_UNSUPPORTED)
        fprintf(stderr, "%s transformed public Conv compile status=%s "
                "reason=%s message=%s route=%s\n", backend,
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    CHECK(status == VX_STATUS_BACKEND_UNSUPPORTED && !compiled &&
          !strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED") &&
          strstr(report.route_evidence,
                 "conv-noncanonical-weight-layout-domain"));
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int run_immutable_widened_conv_backend(
        VxRuntime* runtime, const char* graph_path, const char* weight_path,
        const char* backend) {
    const float input[4] = {1, 2, 3, 4};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    int transformed = 0;
    int cpu_pack = -1;
    int cpu_indirection = -1;
    VxStatus status = compile_backend(
        runtime, graph_path, weight_path, backend,
        &model, &compiled, &report);
    CHECK(status == VX_STATUS_OK && compiled && report.route_attested);
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(execute_conv_weight(context, input, 1, NULL, 4.0f) == 0);
    CHECK(vx_public_api_test_conv_cache_state(
              context, 0, &transformed, &cpu_pack,
              &cpu_indirection) == 0 && transformed && !cpu_pack &&
          !cpu_indirection);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int execute_sigmoid_shape(VxExecutionContext* context,
                                 const char* backend,
                                 const float* values,
                                 int64_t count) {
    float output[4] = {0};
    VxTensorBinding inputs[1] = {
        {sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 1u, {count},
         values, (size_t)count * sizeof(float), VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status = vx_execution_context_execute(
        context, inputs, 1u, &result, &report);
    if (status != VX_STATUS_OK)
        fprintf(stderr, "%s sigmoid execute shape=%" PRId64
                " failed: status=%s reason=%s message=%s route=%s\n",
                backend, count, vx_status_string(status), report.reason,
                report.message, report.route_evidence);
    CHECK(status == VX_STATUS_OK);
    CHECK(result && !strcmp(report.backend, backend) &&
          report.route_attested && !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none"));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    for (int64_t index = 0; index < count; index++) {
        float expected = 1.0f / (1.0f + expf(-values[index]));
        CHECK(output[index] > 0.0f && output[index] < 1.0f);
        CHECK(fabsf(output[index] - expected) <= 1.0e-4f);
    }
    vx_result_release(result);
    return 0;
}

/* Sigmoid carries the same canonical activation-preserve shape contract as
 * SiLU and has a device kernel on every native GPU backend, but it was left
 * out of the bounded-domain proof and the exporter-qualified operator set, so
 * every graph containing one was refused outright on these backends. */
static int run_sigmoid_backend(VxRuntime* runtime,
                               const char* graph_path,
                               const char* backend) {
    const float values[4] = {-2.0f, 0.0f, 2.0f, 4.0f};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status = compile_backend(
        runtime, graph_path, NULL, backend, &model, &compiled, &report);
    if (status != VX_STATUS_OK) {
        fprintf(stderr, "%s sigmoid compile failed: status=%s reason=%s "
                "message=%s route=%s\n", backend, vx_status_string(status),
                report.reason, report.message, report.route_evidence);
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        return 1;
    }
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(execute_sigmoid_shape(context, backend, values, 4) == 0);
    CHECK(execute_sigmoid_shape(context, backend, values, 1) == 0);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

int main(void) {
    static const char* const graph_path =
        "/tmp/volvox-native-gpu-public-dynamic.graph.json";
    static const char* const sigmoid_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-sigmoid.graph.json";
    static const char* const odd_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-odd.graph.json";
    static const char* const odd_weight_path =
        "/tmp/volvox-native-gpu-public-dynamic-odd.safetensors";
    static const char* const typed_batch_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-typed-batch.graph.json";
    static const char* const independent_shape_graph_path =
        "/tmp/volvox-native-gpu-independent-shape.graph.json";
    static const char* const mixed_batch_graph_path =
        "/tmp/volvox-native-gpu-mixed-batch.graph.json";
    static const char* const linear_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-linear.graph.json";
    static const char* const linear_weight_path =
        "/tmp/volvox-native-gpu-public-dynamic-linear.safetensors";
    static const char* const f16_weight_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-linear-f16-weight.graph.json";
    static const char* const f16_bias_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-linear-f16-bias.graph.json";
    static const char* const f16_linear_weight_path =
        "/tmp/volvox-native-gpu-public-dynamic-linear-f16.safetensors";
    static const char* const gather_graph_path =
        "/tmp/volvox-native-gpu-public-dynamic-gather.graph.json";
    static const char* const unsafe_gather_graph_path =
        "/tmp/volvox-native-gpu-public-unsafe-gather.graph.json";
    static const char* const argmax_embedding_graph_path =
        "/tmp/volvox-native-gpu-public-argmax-embedding.graph.json";
    static const char* const embedding_weight_path =
        "/tmp/volvox-native-gpu-public-embedding.safetensors";
    static const char* const public_embedding_weight_graph_path =
        "/tmp/volvox-native-gpu-public-embedding-weight.graph.json";
    static const char* const public_layernorm_affines_graph_path =
        "/tmp/volvox-native-gpu-public-layernorm-affines.graph.json";
    static const char* const public_conv_weight_graph_path =
        "/tmp/volvox-native-gpu-public-conv-weight.graph.json";
    static const char* const transformed_public_conv_graph_path =
        "/tmp/volvox-native-gpu-public-conv-ohwi.graph.json";
    static const char* const immutable_widened_conv_graph_path =
        "/tmp/volvox-native-gpu-immutable-conv-f16.graph.json";
    static const char* const conv_f16_weight_path =
        "/tmp/volvox-native-gpu-immutable-conv-f16.safetensors";
    static const char* const graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"silu\",\"opType\":\"SiLU\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\"]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    static const char* const sigmoid_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"sigmoid\",\"opType\":\"Sigmoid\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\"]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    static const char* const odd_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":5}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"quantize\","
        "\"opType\":\"QuantizeLinear\","
        "\"inputs\":{\"input\":\"x\",\"scale\":\"scale\","
        "\"zero_point\":\"zero_point\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"int8\",\"shape\":[\"Q\"]}},"
        "\"params\":{}}],\"outputs\":[\"y\"],"
        "\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"scale\","
        "\"zero_point_tensor\":\"zero_point\"}}}}";
    static const char* const typed_batch_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"K\":{\"min\":1,\"max\":3}},"
        "\"inputs\":{"
        "\"mask\":{\"shape\":[1,1,1,\"K\"],\"dtype\":\"int32\"},"
        "\"a\":{\"shape\":[1,2,1,\"K\"],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[1,2,\"K\",2],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"id\":\"expand\",\"opType\":\"Expand\","
        "\"inputs\":{\"input\":\"mask\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"expanded\","
        "\"dtype\":\"int32\",\"shape\":[1,2,1,\"K\"]}},"
        "\"params\":{\"shape\":[1,2,1,\"K\"]}},"
        "{\"id\":\"batch\",\"opType\":\"BatchMatMul\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"product\","
        "\"dtype\":\"float32\",\"shape\":[1,2,1,2]}},"
        "\"params\":{}}],\"outputs\":[\"expanded\",\"product\"]}";
    static const char* const independent_shape_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\",2,3],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"id\":\"move\",\"opType\":\"Transpose\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"a\",\"dtype\":\"float32\","
        "\"shape\":[2,\"B\",3]}},\"params\":{\"perm\":[1,0,2]}},"
        "{\"id\":\"reshape\",\"opType\":\"Reshape\","
        "\"inputs\":{\"input\":\"a\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"b\",\"dtype\":\"float32\","
        "\"shape\":[2,\"B\",1,3]}},"
        "\"params\":{\"shape\":[2,\"B\",1,3]}},"
        "{\"id\":\"squeeze\",\"opType\":\"Squeeze\","
        "\"inputs\":{\"input\":\"b\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"c\",\"dtype\":\"float32\","
        "\"shape\":[2,\"B\",3]}},\"params\":{\"axes\":[2]}},"
        "{\"id\":\"restore\",\"opType\":\"Transpose\","
        "\"inputs\":{\"input\":\"c\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"d\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",2,3]}},\"params\":{\"perm\":[1,0,2]}},"
        "{\"id\":\"softmax\",\"opType\":\"Softmax\","
        "\"inputs\":{\"input\":\"d\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"e\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",2,3]}},\"params\":{\"axis\":2}},"
        "{\"id\":\"reduce\",\"opType\":\"ReduceSum\","
        "\"inputs\":{\"input\":\"e\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",2,1]}},"
        "\"params\":{\"axis\":2,\"keepdims\":true}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const mixed_batch_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\",2],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"id\":\"mix-lanes\",\"opType\":\"Reshape\","
        "\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"mixed\",\"dtype\":\"float32\","
        "\"shape\":[2,\"B\"]}},\"params\":{\"shape\":[2,\"B\"]}},"
        "{\"id\":\"cross-lane-sum\",\"opType\":\"ReduceSum\","
        "\"inputs\":{\"input\":\"mixed\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"summed\",\"dtype\":\"float32\","
        "\"shape\":[2]}},\"params\":{\"axis\":1,"
        "\"keepdims\":false}},"
        "{\"id\":\"restore-public-shape\",\"opType\":\"Expand\","
        "\"inputs\":{\"input\":\"summed\"},\"outputs\":{\"out\":{"
        "\"tensor\":\"y\",\"dtype\":\"float32\","
        "\"shape\":[\"B\",2]}},"
        "\"params\":{\"shape\":[\"B\",2]}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const linear_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\",3],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"id\":\"input-major\",\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight.input_major\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"with_bias\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\",2]}},"
        "\"params\":{\"weight_layout\":\"din_dout\"}},"
        "{\"id\":\"output-major\",\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\","
        "\"weight\":\"weight.output_major\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"without_bias\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\",2]}},"
        "\"params\":{\"weight_layout\":\"dout_din\"}}],"
        "\"outputs\":[\"with_bias\",\"without_bias\"]}";
    static const char* const f16_weight_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\",3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"f16-weight\",\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.f16\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\",2]}},"
        "\"params\":{\"weight_layout\":\"din_dout\"}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const f16_bias_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\",3],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"f16-bias\",\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight.f32\","
        "\"bias\":\"bias.f16\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\",2]}},"
        "\"params\":{\"weight_layout\":\"din_dout\"}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const gather_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":2,\"max\":4}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\"],\"dtype\":\"float32\"},"
        "\"indices\":{\"shape\":[2],\"dtype\":\"int32\"}},"
        "\"nodes\":["
        "{\"id\":\"before-gather\",\"opType\":\"SiLU\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"activated\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\"]}},\"params\":{}},"
        "{\"id\":\"gather\",\"opType\":\"Gather\","
        "\"inputs\":{\"input\":\"activated\","
        "\"indices\":\"indices\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"params\":{\"axis\":0}}],\"outputs\":[\"y\"]}";
    static const char* const unsafe_gather_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{},\"inputs\":{"
        "\"x\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"a\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"id\":\"unsafe-indices\",\"opType\":\"Equal\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"indices\","
        "\"dtype\":\"int32\",\"shape\":[2]}},\"params\":{}},"
        "{\"id\":\"gather\",\"opType\":\"Gather\","
        "\"inputs\":{\"input\":\"x\",\"indices\":\"indices\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[2]}},"
        "\"params\":{\"axis\":0}}],\"outputs\":[\"y\"]}";
    static const char* const argmax_embedding_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"S\":{\"min\":2,\"max\":3}},"
        "\"inputs\":{\"logits\":{\"shape\":[\"S\",3],"
        "\"dtype\":\"float32\"}},\"nodes\":["
        "{\"id\":\"argmax\",\"opType\":\"ArgMax\","
        "\"inputs\":{\"input\":\"logits\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"ids\","
        "\"dtype\":\"int32\",\"shape\":[\"S\"]}},"
        "\"params\":{\"axis\":1,\"keepdims\":false}},"
        "{\"id\":\"embedding\",\"opType\":\"Embedding\","
        "\"inputs\":{\"input\":\"ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"S\",2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    static const char* const public_embedding_weight_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"S\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{"
        "\"ids\":{\"shape\":[\"S\"],\"dtype\":\"int32\"},"
        "\"table\":{\"shape\":[3,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"embedding\",\"opType\":\"Embedding\","
        "\"inputs\":{\"input\":\"ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"S\",2]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    static const char* const public_layernorm_affines_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\",2],\"dtype\":\"float32\"},"
        "\"gamma\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"beta\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"norm\",\"opType\":\"LayerNorm\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"gamma\","
        "\"bias\":\"beta\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\",2]}},"
        "\"params\":{\"d_model\":2,\"eps\":0.00001}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const public_conv_weight_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"B\",2,2,1],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"conv\",\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",2,2,1]}},"
        "\"params\":{\"data_layout\":\"NHWC\","
        "\"weight_layout\":\"HWIO\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"dilation\":[1,1],\"groups\":1}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const transformed_public_conv_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"B\",2,2,1],\"dtype\":\"float32\"},"
        "\"weight\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"conv\",\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"weight\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",2,2,1]}},"
        "\"params\":{\"data_layout\":\"NHWC\","
        "\"weight_layout\":\"OHWI\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"dilation\":[1,1],\"groups\":1}}],"
        "\"outputs\":[\"y\"]}";
    static const char* const immutable_widened_conv_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":2}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"B\",2,2,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"conv\",\"opType\":\"Conv2D\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"conv.f16\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"B\",2,2,1]}},"
        "\"params\":{\"data_layout\":\"NHWC\","
        "\"weight_layout\":\"HWIO\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"dilation\":[1,1],\"groups\":1}}],"
        "\"outputs\":[\"y\"]}";
    const NativeGpuCase cases[] = {
        {"vulkan", VOLVOXAI_ENABLE_VULKAN},
        {"opengl", VOLVOXAI_ENABLE_OPENGL},
        {"metal", VOLVOXAI_ENABLE_METAL},
    };
    VxRuntimeOptions options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    int ran = 0;
    CHECK(test_vulkan_duplicate_execution_evidence_rejected() == 0);
    CHECK(test_batch_matmul_plan_validation() == 0);
    CHECK(write_text(graph_path, graph) == 0);
    CHECK(write_text(sigmoid_graph_path, sigmoid_graph) == 0);
    CHECK(write_text(odd_graph_path, odd_graph) == 0);
    CHECK(write_text(typed_batch_graph_path, typed_batch_graph) == 0);
    CHECK(write_text(independent_shape_graph_path,
                     independent_shape_graph) == 0);
    CHECK(write_text(mixed_batch_graph_path, mixed_batch_graph) == 0);
    CHECK(write_text(linear_graph_path, linear_graph) == 0);
    CHECK(write_text(f16_weight_graph_path, f16_weight_graph) == 0);
    CHECK(write_text(f16_bias_graph_path, f16_bias_graph) == 0);
    CHECK(write_text(gather_graph_path, gather_graph) == 0);
    CHECK(write_text(unsafe_gather_graph_path, unsafe_gather_graph) == 0);
    CHECK(write_text(
              argmax_embedding_graph_path, argmax_embedding_graph) == 0);
    CHECK(write_text(public_embedding_weight_graph_path,
                     public_embedding_weight_graph) == 0);
    CHECK(write_text(public_layernorm_affines_graph_path,
                     public_layernorm_affines_graph) == 0);
    CHECK(write_text(public_conv_weight_graph_path,
                     public_conv_weight_graph) == 0);
    CHECK(write_text(transformed_public_conv_graph_path,
                     transformed_public_conv_graph) == 0);
    CHECK(write_text(immutable_widened_conv_graph_path,
                     immutable_widened_conv_graph) == 0);
    CHECK(write_quantize_weights(odd_weight_path) == 0);
    CHECK(write_linear_weights(linear_weight_path) == 0);
    CHECK(write_linear_f16_rejection_weights(f16_linear_weight_path) == 0);
    CHECK(write_embedding_weights(embedding_weight_path) == 0);
    CHECK(write_conv_f16_weights(conv_f16_weight_path) == 0);
    options.max_batch_delay_milliseconds = 50u;
    CHECK(vx_runtime_create(&options, &runtime, &report) == VX_STATUS_OK);
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        int result;
        if (!cases[index].enabled) continue;
        result = run_backend(runtime, graph_path, cases[index].name);
        if (result == 77) {
            fprintf(stderr, "%s device unavailable; skipping\n",
                    cases[index].name);
            continue;
        }
        CHECK(result == 0);
        if (strcmp(cases[index].name, "metal"))
            CHECK(run_scheduled_batch_backend(
                      runtime, graph_path, cases[index].name) == 0);
        if (strcmp(cases[index].name, "metal")) {
            CHECK(expect_independent_batch_contract_backend(
                      runtime, independent_shape_graph_path, NULL,
                      cases[index].name) == 0);
            CHECK(run_unproved_batch_fail_closed_backend(
                      runtime, mixed_batch_graph_path,
                      cases[index].name) == 0);
        }
        CHECK(run_sigmoid_backend(
                  runtime, sigmoid_graph_path, cases[index].name) == 0);
        CHECK(run_odd_byte_backend(
                  runtime, odd_graph_path, odd_weight_path,
                  cases[index].name) == 0);
        CHECK(run_typed_batch_backend(
                  runtime, typed_batch_graph_path,
                  cases[index].name) == 0);
        CHECK(run_linear_backend(
                  runtime, linear_graph_path, linear_weight_path,
                  cases[index].name) == 0);
        CHECK(reject_f16_linear_backend(
                  runtime, f16_weight_graph_path, f16_linear_weight_path,
                  cases[index].name, "weight") == 0);
        CHECK(reject_f16_linear_backend(
                  runtime, f16_bias_graph_path, f16_linear_weight_path,
                  cases[index].name, "bias") == 0);
        CHECK(run_gather_preflight_backend(
                  runtime, gather_graph_path, cases[index].name) == 0);
        CHECK(reject_unsafe_gather_origin(
                  runtime, unsafe_gather_graph_path,
                  cases[index].name) == 0);
        CHECK(run_argmax_embedding_backend(
                  runtime, argmax_embedding_graph_path,
                  embedding_weight_path, cases[index].name) == 0);
        CHECK(run_public_embedding_weight_backend(
                  runtime, public_embedding_weight_graph_path,
                  cases[index].name) == 0);
        CHECK(run_public_layernorm_affines_backend(
                  runtime, public_layernorm_affines_graph_path,
                  cases[index].name) == 0);
        CHECK(run_public_conv_weight_backend(
                  runtime, public_conv_weight_graph_path,
                  cases[index].name) == 0);
        CHECK(reject_transformed_public_conv_weight(
                  runtime, transformed_public_conv_graph_path,
                  cases[index].name) == 0);
        CHECK(run_immutable_widened_conv_backend(
                  runtime, immutable_widened_conv_graph_path,
                  conv_f16_weight_path,
                  cases[index].name) == 0);
        ran++;
    }
    vx_runtime_release(runtime);
    CHECK(remove(graph_path) == 0);
    CHECK(remove(odd_graph_path) == 0);
    CHECK(remove(odd_weight_path) == 0);
    CHECK(remove(typed_batch_graph_path) == 0);
    CHECK(remove(independent_shape_graph_path) == 0);
    CHECK(remove(mixed_batch_graph_path) == 0);
    CHECK(remove(linear_graph_path) == 0);
    CHECK(remove(linear_weight_path) == 0);
    CHECK(remove(f16_weight_graph_path) == 0);
    CHECK(remove(f16_bias_graph_path) == 0);
    CHECK(remove(f16_linear_weight_path) == 0);
    CHECK(remove(gather_graph_path) == 0);
    CHECK(remove(unsafe_gather_graph_path) == 0);
    CHECK(remove(argmax_embedding_graph_path) == 0);
    CHECK(remove(embedding_weight_path) == 0);
    CHECK(remove(public_embedding_weight_graph_path) == 0);
    CHECK(remove(public_layernorm_affines_graph_path) == 0);
    CHECK(remove(public_conv_weight_graph_path) == 0);
    CHECK(remove(transformed_public_conv_graph_path) == 0);
    CHECK(remove(immutable_widened_conv_graph_path) == 0);
    CHECK(remove(conv_f16_weight_path) == 0);
    if (!ran) return 77;
    puts("native GPU public dynamic shape tests passed");
    return 0;
}
