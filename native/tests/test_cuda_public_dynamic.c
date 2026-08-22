#include "volvoxai.h"
#include "volvoxai_backend.h"
#include "safetensors.h"
#include "../../examples/native_dynamic_batch_benchmark/evidence_tokens.h"

#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        return 1; \
    } \
} while (0)

extern int vx_public_api_test_dynamic_shape_state(
    const VxExecutionContext* context,
    uint64_t* resource_generation,
    size_t* arena_capacity,
    size_t* arena_high_water,
    uint64_t* arena_grow_count);
extern int vx_public_api_test_cuda_dynamic_reservation(
    const VxExecutionContext* context,
    uint64_t* allocation_count,
    size_t* span_count,
    int* preload_complete,
    int* enforced,
    int* replay_plan);
extern int vx_public_api_test_cuda_dynamic_replay_state(
    const VxExecutionContext* context,
    int* graph_api_available,
    int* active_plan,
    size_t* plan_count,
    size_t* ready_plan_count,
    size_t* destroy_pending_count,
    uint64_t* capture_count,
    uint64_t* replay_count,
    uint64_t* invalidation_count,
    uint64_t* graph_exec_destroy_count,
    uint64_t* slot_epoch,
    uint64_t* capacity_generation,
    int* last_forward_replayed);
extern int vx_public_api_test_cuda_invalidate_replay_key(
    const VxExecutionContext* context, int key);
extern int vx_public_api_test_cuda_fail_next_graph_exec_destroy(
    const VxExecutionContext* context);
extern int vx_public_api_test_cuda_cleanup(
    const VxExecutionContext* context);
extern int vx_public_api_test_cuda_replay_resource_limits(
    const VxExecutionContext* context,
    uint32_t* plan_capacity,
    uint64_t* fixed_host_metadata_bytes);

typedef struct {
    int graph_api;
    int active_plan;
    size_t plans;
    size_t ready;
    size_t destroy_pending;
    uint64_t captures;
    uint64_t replays;
    uint64_t invalidations;
    uint64_t graph_exec_destroys;
    uint64_t slot_epoch;
    uint64_t capacity_generation;
    int last_replayed;
} PublicCudaReplayState;

enum {
    PUBLIC_CUDA_REPLAY_EMPTY = 0,
    PUBLIC_CUDA_REPLAY_OBSERVED = 1,
    PUBLIC_CUDA_REPLAY_READY = 2,
};

static int test_cuda_duplicate_execution_evidence_rejected(void) {
    const char* duplicate =
        "cuda_graph_replay=;provider=builtin:cuda;cuda_graph_replay=1";
    char value[8];
    CHECK(vx_native_batch_evidence_key_count(
              duplicate, "cuda_graph_replay") == 2u);
    CHECK(!vx_native_batch_evidence_token(
        duplicate, "cuda_graph_replay", value, sizeof(value)));
    return 0;
}

static int cuda_replay_state(const VxExecutionContext* context,
                             PublicCudaReplayState* state) {
    if (!state) return -1;
    memset(state, 0, sizeof(*state));
    return vx_public_api_test_cuda_dynamic_replay_state(
        context, &state->graph_api, &state->active_plan,
        &state->plans, &state->ready, &state->destroy_pending,
        &state->captures,
        &state->replays, &state->invalidations,
        &state->graph_exec_destroys, &state->slot_epoch,
        &state->capacity_generation, &state->last_replayed);
}
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

static void count_physical_forward(void* user_data) {
    int* count = (int*)user_data;
    (*count)++;
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

static int write_dynamic_weights(const char* path) {
    const int scalar_shape[1] = {1};
    const uint16_t offset_f16 = UINT16_C(0x4000); /* 2.0 */
    const float view_source = 1.25f;
    const float a_scale = 0.5f;
    const uint8_t a_zero = 173u;
    const float b_scale = 0.25f;
    const int8_t b_zero = -29;
    const float y_scale = 0.125f;
    const int8_t y_zero = 5;
    SafetensorsFile weights;
    int status;
    if (safetensors_init_empty(
            &weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
        return -1;
    if (safetensors_add_tensor(
            &weights, "offset", SAFETENSORS_DTYPE_F16,
            scalar_shape, 1, &offset_f16, sizeof(offset_f16)) != 0 ||
        safetensors_add_tensor(
            &weights, "view_source", SAFETENSORS_DTYPE_F32,
            scalar_shape, 1, &view_source, sizeof(view_source)) != 0 ||
        safetensors_add_tensor(
            &weights, "qbatch.a.scale", SAFETENSORS_DTYPE_F32,
            scalar_shape, 1, &a_scale, sizeof(a_scale)) != 0 ||
        safetensors_add_tensor(
            &weights, "qbatch.a.zero", SAFETENSORS_DTYPE_U8,
            scalar_shape, 1, &a_zero, sizeof(a_zero)) != 0 ||
        safetensors_add_tensor(
            &weights, "qbatch.b.scale", SAFETENSORS_DTYPE_F32,
            scalar_shape, 1, &b_scale, sizeof(b_scale)) != 0 ||
        safetensors_add_tensor(
            &weights, "qbatch.b.zero", SAFETENSORS_DTYPE_I8,
            scalar_shape, 1, &b_zero, sizeof(b_zero)) != 0 ||
        safetensors_add_tensor(
            &weights, "qbatch.y.scale", SAFETENSORS_DTYPE_F32,
            scalar_shape, 1, &y_scale, sizeof(y_scale)) != 0 ||
        safetensors_add_tensor(
            &weights, "qbatch.y.zero", SAFETENSORS_DTYPE_I8,
            scalar_shape, 1, &y_zero, sizeof(y_zero)) != 0) {
        safetensors_free(&weights);
        return -1;
    }
    status = safetensors_save(path, &weights);
    safetensors_free(&weights);
    return status;
}

static VxStatus compile_cuda(VxRuntime* runtime,
                             const char* graph_path,
                             const char* weight_path,
                             VxModel** model,
                             VxCompiledModel** compiled,
                             VxReport* report) {
    static const char* const backends[] = {"cuda"};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    const char* weights[1];
    VxStatus status;
    source.graph_path = graph_path;
    if (weight_path) {
        weights[0] = weight_path;
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

static int cuda_compile_attestation_complete(const VxReport* report) {
    int complete = report && !strcmp(report->backend, "cuda") &&
        report->route_attested && !report->operator_fallback_used &&
        strstr(report->route_evidence, "provider=builtin:cuda") &&
        strstr(report->route_evidence, "shape_proof=") &&
        strstr(report->route_evidence, "domain_route=all") &&
        strstr(report->route_evidence, "dynamic=1") &&
        strstr(report->route_evidence, "native_gpu_backend=cuda") &&
        strstr(report->route_evidence, "native_gpu_domain_spans=") &&
        strstr(report->route_evidence, "cuda_graph_cache=4");
    if (!complete && report)
        fprintf(stderr, "incomplete CUDA compile attestation: %s\n",
                report->route_evidence);
    return complete;
}

static int execute_concat(VxExecutionContext* context,
                          const float* dynamic_values,
                          int64_t dynamic_count,
                          int expect_cache_hit,
                          int expect_graph_replay) {
    const float fixed[2] = {10.0f, 11.0f};
    float output[8] = {0};
    VxTensorBinding inputs[2] = {
        {sizeof(VxTensorBinding), "fixed", VX_DTYPE_F32, 1u, {2},
         fixed, sizeof(fixed), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "dynamic", VX_DTYPE_F32, 1u,
         {dynamic_count}, dynamic_values,
         (size_t)dynamic_count * sizeof(float), VX_MEMORY_HOST},
    };
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    char replay_evidence[32];
    CHECK(vx_execution_context_execute(
              context, inputs, 2u, &result, &report) == VX_STATUS_OK);
    CHECK(result != NULL && !strcmp(report.backend, "cuda") &&
          report.route_attested && !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none") &&
          strstr(report.route_evidence,
                 expect_cache_hit ? "shape_plan=hit" : "shape_plan=cold"));
    CHECK(snprintf(replay_evidence, sizeof(replay_evidence),
                   "cuda_graph_replay=%d", expect_graph_replay) > 0 &&
          strstr(report.route_evidence, replay_evidence));
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "joined") && info.dtype == VX_DTYPE_F32 &&
          info.rank == 1u && info.shape[0] == dynamic_count + 3 &&
          info.byte_size == (size_t)(dynamic_count + 3) * sizeof(float));
    CHECK(vx_result_read(result, "joined", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(output[0] == fixed[0] && output[1] == fixed[1]);
    CHECK(output[2] == 1.25f);
    for (int64_t index = 0; index < dynamic_count; index++)
        CHECK(output[index + 3] == dynamic_values[index] * 0.5f);
    vx_result_release(result);
    return 0;
}

static int rejected_domain(VxRuntime* runtime,
                           const char* graph_path,
                           const char* graph,
                           const char* predicate_reason) {
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxReport report = VX_REPORT_INIT;
    VxStatus status;
    CHECK(write_text(graph_path, graph) == 0);
    status = compile_cuda(
        runtime, graph_path, NULL, &model, &compiled, &report);
    if (status != VX_STATUS_BACKEND_UNSUPPORTED) {
        fprintf(stderr,
                "unexpected CUDA domain rejection status=%s reason=%s "
                "message=%s route=%s\n",
                vx_status_string(status), report.reason, report.message,
                report.route_evidence);
    }
    CHECK(status == VX_STATUS_BACKEND_UNSUPPORTED);
    if (compiled != NULL ||
        strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED") ||
        !strstr(report.route_evidence,
                "required=native-gpu-full-bounded-domain") ||
        !strstr(report.route_evidence, predicate_reason))
        fprintf(stderr,
                "incomplete CUDA domain rejection reason=%s message=%s "
                "route=%s expected_predicate=%s\n",
                report.reason, report.message, report.route_evidence,
                predicate_reason);
    CHECK(compiled == NULL &&
          !strcmp(report.reason, "BOUNDED_DOMAIN_UNSUPPORTED") &&
          strstr(report.route_evidence,
                 "required=native-gpu-full-bounded-domain") &&
          strstr(report.route_evidence, predicate_reason));
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static void reference_layernorm(const float* input, const float gamma[2],
                                const float beta[2], float* output,
                                int64_t rows) {
    for (int64_t row = 0; row < rows; row++) {
        const float* values = input + row * 2;
        float mean = (values[0] + values[1]) * 0.5f;
        float d0 = values[0] - mean;
        float d1 = values[1] - mean;
        float inverse = 1.0f /
            sqrtf((d0 * d0 + d1 * d1) * 0.5f + 1.0e-5f);
        output[row * 2] = d0 * inverse * gamma[0] + beta[0];
        output[row * 2 + 1] = d1 * inverse * gamma[1] + beta[1];
    }
}

static int execute_layernorm(VxExecutionContext* context,
                             const float* input, int64_t rows,
                             const float gamma[2], const float beta[2]) {
    float expected[4] = {0};
    float output[4] = {0};
    VxTensorBinding bindings[3] = {
        {sizeof(VxTensorBinding), "x", VX_DTYPE_F32, 2u, {rows, 2},
         input, (size_t)rows * 2u * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "gamma", VX_DTYPE_F32, 1u, {2},
         gamma, 2u * sizeof(float), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "beta", VX_DTYPE_F32, 1u, {2},
         beta, 2u * sizeof(float), VX_MEMORY_HOST},
    };
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    reference_layernorm(input, gamma, beta, expected, rows);
    CHECK(vx_execution_context_execute(
              context, bindings, 3u, &result, &report) == VX_STATUS_OK);
    CHECK(result && report.route_attested && !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none"));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    for (int64_t index = 0; index < rows * 2; index++)
        CHECK(fabsf(output[index] - expected[index]) <= 2.0e-4f);
    vx_result_release(result);
    return 0;
}

static int run_public_layernorm_affines(
        VxRuntime* runtime, const char* graph_path) {
    const float min_input[2] = {1.0f, 3.0f};
    const float max_input[4] = {2.0f, 6.0f, -4.0f, 8.0f};
    const float gamma_a[2] = {1.0f, 1.0f};
    const float beta_a[2] = {0.0f, 0.0f};
    const float gamma_b[2] = {2.0f, 3.0f};
    const float beta_b[2] = {10.0f, 20.0f};
    const float gamma_c[2] = {-1.0f, 0.5f};
    const float beta_c[2] = {-7.0f, 9.0f};
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    uint64_t reserved_allocations = 0;
    uint64_t current_allocations = 0;
    uint64_t generation = 0;
    uint64_t grow_count = 0;
    size_t capacity = 0;
    size_t initial_capacity = 0;
    size_t high_water = 0;
    CHECK(compile_cuda(runtime, graph_path, NULL, &model, &compiled,
                       &report) == VX_STATUS_OK);
    CHECK(compiled && cuda_compile_attestation_complete(&report));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &reserved_allocations, NULL, NULL, NULL, NULL) == 0 &&
          reserved_allocations > 0);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &generation, &capacity, &high_water,
              &grow_count) == 0 && generation == 0 && capacity > 0 &&
          capacity == high_water && grow_count == 1);
    initial_capacity = capacity;
    CHECK(execute_layernorm(
              context, min_input, 1, gamma_a, beta_a) == 0);
    CHECK(execute_layernorm(
              context, max_input, 2, gamma_b, beta_b) == 0);
    CHECK(execute_layernorm(
              context, min_input, 1, gamma_c, beta_c) == 0);
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &current_allocations, NULL, NULL, NULL, NULL) == 0 &&
          current_allocations == reserved_allocations);
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

static int execute_qbatch(VxExecutionContext* context, int64_t reduction,
                          int expect_cache_hit) {
    enum { A_ZERO = 173, B_ZERO = -29, Y_ZERO = 5 };
    uint8_t a[14] = {0};
    int8_t b[21] = {0};
    int8_t expected[6] = {0};
    int8_t output[6] = {0};
    VxTensorBinding bindings[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_U8, 3u,
         {1, 2, reduction}, a, (size_t)(2 * reduction), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_I8, 3u,
         {1, reduction, 3}, b, (size_t)(3 * reduction), VX_MEMORY_HOST},
    };
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxResult* result = NULL;
    const char* tactic;
    uint64_t dp4a_groups = 0u;
    uint64_t tail_values = 0u;
    for (int64_t index = 0; index < 2 * reduction; index++) {
        int32_t centered = (int32_t)((index * 5 + 3) % 11) - 5;
        a[index] = (uint8_t)(A_ZERO + centered);
    }
    for (int64_t index = 0; index < 3 * reduction; index++) {
        int32_t centered = (int32_t)((index * 7 + 1) % 13) - 6;
        b[index] = (int8_t)(B_ZERO + centered);
    }
    for (int row = 0; row < 2; row++) {
        for (int column = 0; column < 3; column++) {
            int32_t accumulator = 0;
            int32_t quantized;
            for (int64_t inner = 0; inner < reduction; inner++) {
                int32_t a_value =
                    (int32_t)a[row * reduction + inner] - A_ZERO;
                int32_t b_value =
                    (int32_t)b[inner * 3 + column] - B_ZERO;
                accumulator += a_value * b_value;
            }
            quantized = accumulator + Y_ZERO;
            if (quantized < -128) quantized = -128;
            if (quantized > 127) quantized = 127;
            expected[row * 3 + column] = (int8_t)quantized;
        }
    }
    CHECK(vx_execution_context_execute(
              context, bindings, 2u, &result, &report) == VX_STATUS_OK);
    tactic = strstr(report.route_evidence, ";cuda_qbatch_dp4a=");
    CHECK(result && !strcmp(report.backend, "cuda") &&
          report.route_attested && !report.operator_fallback_used &&
          strstr(report.fallback_evidence, "operator=none") && tactic &&
          strstr(report.route_evidence,
                 expect_cache_hit ? "shape_plan=hit" : "shape_plan=cold"));
    CHECK(sscanf(tactic,
                 ";cuda_qbatch_dp4a=%" SCNu64
                 ";cuda_qbatch_tail=%" SCNu64,
                 &dp4a_groups, &tail_values) == 2 &&
          dp4a_groups > 0u && tail_values > 0u);
    CHECK(vx_result_output_info(result, 0u, &info, &report) == VX_STATUS_OK);
    CHECK(!strcmp(info.name, "y") && info.dtype == VX_DTYPE_I8 &&
          info.rank == 3u && info.shape[0] == 1 && info.shape[1] == 2 &&
          info.shape[2] == 3 && info.byte_size == sizeof(output));
    CHECK(vx_result_read(result, "y", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    vx_result_release(result);
    return 0;
}

static int run_public_qbatch(VxRuntime* runtime, const char* graph_path,
                             const char* weight_path) {
    VxContextOptions options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    uint64_t reserved_allocations = 0u;
    uint64_t current_allocations = 0u;
    CHECK(compile_cuda(runtime, graph_path, weight_path, &model, &compiled,
                       &report) == VX_STATUS_OK);
    CHECK(compiled && cuda_compile_attestation_complete(&report));
    CHECK(vx_compiled_model_create_context(
              compiled, &options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &reserved_allocations, NULL, NULL, NULL, NULL) == 0 &&
          reserved_allocations > 0u);
    CHECK(execute_qbatch(context, 5, 0) == 0);
    CHECK(execute_qbatch(context, 7, 0) == 0);
    CHECK(execute_qbatch(context, 5, 1) == 0);
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &current_allocations, NULL, NULL, NULL, NULL) == 0 &&
          current_allocations == reserved_allocations);
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

static int run_scheduled_cuda_batch(VxRuntime* runtime,
                                    const char* graph_path) {
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
    uint32_t min_batch = 0;
    uint32_t max_batch = 0;
    uint32_t multiple = 0;
    int32_t batch_axis = -1;
    int32_t device_resident = 0;
    uint64_t device_epoch = 0;
    uint64_t dispatches_before = 0;
    uint64_t dispatches_after = 0;
    const char* graph_fingerprint = NULL;
    const char* proof_identity = NULL;
    float first_output = 0.0f;
    float second_output = 0.0f;
    int physical_forwards = 0;
    CHECK(compile_cuda(runtime, graph_path, NULL, &model, &compiled,
                       &report) == VX_STATUS_OK);
    CHECK(compiled && cuda_compile_attestation_complete(&report) &&
          strstr(report.route_evidence,
                 "batchProtocol=" VX_BACKEND_BATCH_PROTOCOL) &&
          vx_public_api_test_compiled_batch_contract(
              compiled, &min_batch, &max_batch, &multiple, &batch_axis,
              &device_resident, &device_epoch, &graph_fingerprint,
              &proof_identity) == 1 &&
          min_batch == 1u && max_batch == 4u && multiple == 1u &&
          batch_axis == 0 && device_resident == 1 && device_epoch == 1u &&
          graph_fingerprint && graph_fingerprint[0] && proof_identity &&
          strstr(proof_identity,
                 VX_BACKEND_INDEPENDENT_BATCH_PROOF_PROTOCOL));
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
          first_result && report.route_attested &&
          !report.operator_fallback_used &&
          strstr(report.message, "trueBackendInvocations=1") &&
          strstr(report.route_evidence, "trueBackendInvocations=1") &&
          strstr(report.route_evidence, "cuda_graph_replay=0") &&
          strstr(report.route_evidence,
                 "batchProtocol=" VX_BACKEND_BATCH_PROTOCOL) &&
          strstr(report.route_evidence, "provider=builtin:cuda") &&
          strstr(report.route_evidence, "cuda_graph_cache=4"));
    CHECK(vx_request_result(second, &second_result, &report) == VX_STATUS_OK &&
          second_result && report.route_attested &&
          !report.operator_fallback_used &&
          strstr(report.message, "trueBackendInvocations=1") &&
          strstr(report.route_evidence, "trueBackendInvocations=1") &&
          strstr(report.route_evidence, "cuda_graph_replay=0") &&
          strstr(report.route_evidence,
                 "batchProtocol=" VX_BACKEND_BATCH_PROTOCOL) &&
          strstr(report.route_evidence, "provider=builtin:cuda") &&
          strstr(report.route_evidence, "cuda_graph_cache=4"));
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
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    return 0;
}

int main(void) {
    static const char* const graph_path =
        "/tmp/volvox-cuda-public-dynamic.graph.json";
    static const char* const weight_path =
        "/tmp/volvox-cuda-public-dynamic.safetensors";
    static const char* const reject_path =
        "/tmp/volvox-cuda-public-dynamic-reject.graph.json";
    static const char* const layernorm_path =
        "/tmp/volvox-cuda-public-dynamic-layernorm.graph.json";
    static const char* const qbatch_path =
        "/tmp/volvox-cuda-public-dynamic-qbatch.graph.json";
    static const char* const scheduled_batch_path =
        "/tmp/volvox-cuda-public-dynamic-scheduled-batch.graph.json";
    static const char* const concat_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{"
        "\"Q\":{\"min\":1,\"max\":5},"
        "\"M\":{\"min\":4,\"max\":8}},"
        "\"inputs\":{"
        "\"fixed\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"dynamic\":{\"shape\":[\"Q\"],\"dtype\":\"float32\"}},"
        "\"nodes\":["
        "{\"id\":\"view\",\"opType\":\"Reshape\","
        "\"inputs\":{\"input\":\"view_source\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"view\","
        "\"dtype\":\"float32\",\"shape\":[1]}},"
        "\"params\":{\"shape\":[1]}},"
        "{\"id\":\"scale\",\"opType\":\"Div\","
        "\"inputs\":{\"a\":\"dynamic\",\"b\":\"offset\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"scaled\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\"]}},"
        "\"params\":{}},"
        "{\"id\":\"join\",\"opType\":\"Concat\","
        "\"inputs\":{\"input0\":\"fixed\",\"input1\":\"view\","
        "\"input2\":\"scaled\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"joined\","
        "\"dtype\":\"float32\",\"shape\":[\"M\"]}},"
        "\"params\":{\"axis\":0}}],\"outputs\":[\"joined\"]}";
    static const char* const softmax_reject_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"Q\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{\"x\":{\"shape\":[\"Q\",2],"
        "\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"bad-softmax\","
        "\"opType\":\"Softmax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"Q\",2]}},"
        "\"params\":{\"axis\":0}}],\"outputs\":[\"y\"]}";
    static const char* const broadcast_reject_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{"
        "\"A\":{\"min\":1,\"max\":4},"
        "\"B\":{\"min\":1,\"max\":4},"
        "\"C\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{"
        "\"a\":{\"shape\":[\"A\"],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[\"B\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"bad-broadcast\",\"opType\":\"Mul\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"C\"]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    static const char* const slice_reject_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{"
        "\"Q\":{\"min\":2,\"max\":4},"
        "\"R\":{\"min\":1,\"max\":3}},"
        "\"inputs\":{"
        "\"x\":{\"shape\":[\"Q\"],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"unbound-slice\",\"opType\":\"Slice\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"R\"]}},"
        "\"params\":{\"starts\":[1],\"ends\":[2147483647],"
        "\"axes\":[0],\"steps\":[1]}}],\"outputs\":[\"y\"]}";
    static const char* const layernorm_graph =
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
    static const char* const qbatch_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"R\":{\"min\":5,\"max\":7}},"
        "\"inputs\":{"
        "\"a\":{\"shape\":[1,2,\"R\"],\"dtype\":\"uint8\"},"
        "\"b\":{\"shape\":[1,\"R\",3],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"id\":\"qbatch\","
        "\"opType\":\"QBatchMatMul\","
        "\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"int8\",\"shape\":[1,2,3]}},"
        "\"params\":{}}],\"outputs\":[\"y\"],"
        "\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"a\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"qbatch.a.scale\","
        "\"zero_point_tensor\":\"qbatch.a.zero\"},"
        "\"b\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"qbatch.b.scale\","
        "\"zero_point_tensor\":\"qbatch.b.zero\"},"
        "\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"qbatch.y.scale\","
        "\"zero_point_tensor\":\"qbatch.y.zero\"}}}}";
    static const char* const scheduled_batch_graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"dimensions\":{\"B\":{\"min\":1,\"max\":4}},"
        "\"inputs\":{\"x\":{\"shape\":[\"B\"],"
        "\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"id\":\"silu\",\"opType\":\"SiLU\","
        "\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":{\"tensor\":\"y\","
        "\"dtype\":\"float32\",\"shape\":[\"B\"]}},"
        "\"params\":{}}],\"outputs\":[\"y\"]}";
    const float batch_1_a[1] = {20.0f};
    const float batch_1_b[1] = {25.0f};
    const float batch_1_c[1] = {40.0f};
    const float batch_2[2] = {50.0f, 51.0f};
    const float batch_3[3] = {60.0f, 61.0f, 62.0f};
    const float batch_4_a[4] = {30.0f, 31.0f, 32.0f, 33.0f};
    const float batch_4_b[4] = {34.0f, 35.0f, 36.0f, 37.0f};
    const float batch_5[5] = {70.0f, 71.0f, 72.0f, 73.0f, 74.0f};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxStatus status;
    uint64_t reserved_allocations = 0;
    uint64_t current_allocations = 0;
    uint64_t resource_generation = 0;
    uint64_t arena_grow_count = 0;
    uint64_t replay_fixed_host_metadata = 0;
    size_t span_count = 0;
    size_t arena_capacity = 0;
    size_t arena_high_water = 0;
    int preload_complete = 0;
    int enforced = 0;
    int replay_plan = -1;
    uint32_t replay_plan_capacity = 0;
    PublicCudaReplayState replay_state = {0};
    PublicCudaReplayState replay_checkpoint = {0};
    int graph_api = 0;

    CHECK(test_cuda_duplicate_execution_evidence_rejected() == 0);
    CHECK(write_text(graph_path, concat_graph) == 0);
    CHECK(write_text(layernorm_path, layernorm_graph) == 0);
    CHECK(write_text(qbatch_path, qbatch_graph) == 0);
    CHECK(write_text(scheduled_batch_path, scheduled_batch_graph) == 0);
    CHECK(write_dynamic_weights(weight_path) == 0);
    runtime_options.max_batch_delay_milliseconds = 50u;
    CHECK(vx_runtime_create(
              &runtime_options, &runtime, &report) == VX_STATUS_OK);
    status = compile_cuda(
        runtime, graph_path, weight_path, &model, &compiled, &report);
    if (status == VX_STATUS_BACKEND_UNAVAILABLE) {
        vx_compiled_model_release(compiled);
        vx_model_release(model);
        vx_runtime_release(runtime);
        (void)remove(graph_path);
        (void)remove(weight_path);
        (void)remove(layernorm_path);
        (void)remove(qbatch_path);
        (void)remove(scheduled_batch_path);
        puts("CUDA device unavailable; skipping CUDA public dynamic test");
        return 77;
    }
    CHECK(status == VX_STATUS_OK && compiled != NULL &&
          cuda_compile_attestation_complete(&report));
    CHECK(vx_compiled_model_create_context(
              compiled, &context_options, &context, &report) == VX_STATUS_OK);
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &reserved_allocations, &span_count,
              &preload_complete, &enforced, &replay_plan) == 0 &&
          reserved_allocations > 0 && span_count > 0 && preload_complete &&
          enforced && replay_plan == 0);
    CHECK(vx_public_api_test_cuda_replay_resource_limits(
              context, &replay_plan_capacity,
              &replay_fixed_host_metadata) == 0 &&
          replay_plan_capacity == 4u && replay_fixed_host_metadata > 0u);
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, &resource_generation, &arena_capacity,
              &arena_high_water, &arena_grow_count) == 0 &&
          resource_generation == 0 && arena_capacity > 0 &&
          arena_high_water == arena_capacity && arena_grow_count == 1);
    CHECK(cuda_replay_state(context, &replay_state) == 0 &&
          replay_state.active_plan == PUBLIC_CUDA_REPLAY_EMPTY &&
          replay_state.plans == 0u && replay_state.ready == 0u &&
          replay_state.captures == 0u && replay_state.replays == 0u &&
          !replay_state.last_replayed);
    graph_api = replay_state.graph_api;

    /* A new exact signature is observed, captured, and only then replayed.
     * The third forward changes request data without changing any captured
     * pointer or scalar shape argument. */
    CHECK(execute_concat(context, batch_1_a, 1, 0, 0) == 0);
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &current_allocations, NULL, NULL, NULL,
              &replay_plan) == 0 &&
          current_allocations == reserved_allocations &&
          replay_plan == (graph_api ? PUBLIC_CUDA_REPLAY_OBSERVED
                                    : PUBLIC_CUDA_REPLAY_EMPTY));
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_OBSERVED &&
              replay_state.plans == 1u && replay_state.ready == 0u &&
              replay_state.captures == 0u && replay_state.replays == 0u &&
              !replay_state.last_replayed);
    }
    CHECK(vx_public_api_test_dynamic_shape_state(
              context, NULL, &arena_capacity, &arena_high_water,
              &arena_grow_count) == 0 &&
          arena_high_water == arena_capacity && arena_grow_count == 1);
    CHECK(execute_concat(context, batch_1_b, 1, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.plans == 1u && replay_state.ready == 1u &&
              replay_state.captures == 1u && replay_state.replays == 0u &&
              !replay_state.last_replayed);
    }
    CHECK(execute_concat(context, batch_1_c, 1, 1, graph_api ? 1 : 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.plans == 1u && replay_state.ready == 1u &&
              replay_state.captures == 1u && replay_state.replays == 1u &&
              replay_state.last_replayed);
    }

    /* The alternating direct-B1/scheduled-BN benchmark pattern must retain
     * both graph executables. Returning B1 after B4 is an immediate replay,
     * not a new observe/capture cycle. */
    CHECK(execute_concat(context, batch_4_a, 4, 0, 0) == 0);
    CHECK(execute_concat(context, batch_4_a, 4, 1, 0) == 0);
    CHECK(execute_concat(context, batch_4_b, 4, 1, graph_api ? 1 : 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.plans == 2u && replay_state.ready == 2u &&
              replay_state.captures == 2u && replay_state.replays == 2u &&
              replay_state.last_replayed);
    }
    CHECK(execute_concat(
              context, batch_1_a, 1, 1, graph_api ? 1 : 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.plans == 2u && replay_state.ready == 2u &&
              replay_state.captures == 2u && replay_state.replays == 3u &&
              replay_state.last_replayed);
    }

    /* Every global key mutation invalidates all retained shapes before the
     * next forward and starts a safe direct OBSERVE pass. */
    replay_checkpoint = replay_state;
    CHECK(vx_public_api_test_cuda_invalidate_replay_key(context, 2) == 0);
    CHECK(execute_concat(context, batch_1_b, 1, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.slot_epoch != replay_checkpoint.slot_epoch &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_OBSERVED &&
              replay_state.plans == 1u && replay_state.ready == 0u &&
              replay_state.captures == replay_checkpoint.captures &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations >=
                  replay_checkpoint.invalidations + 2u &&
              replay_state.graph_exec_destroys >=
                  replay_checkpoint.graph_exec_destroys + 2u &&
              !replay_state.last_replayed);
    }
    CHECK(execute_concat(context, batch_1_c, 1, 1, 0) == 0);
    CHECK(execute_concat(
              context, batch_1_a, 1, 1, graph_api ? 1 : 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api)
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.ready == 1u && replay_state.last_replayed);

    replay_checkpoint = replay_state;
    CHECK(vx_public_api_test_cuda_invalidate_replay_key(context, 1) == 0);
    CHECK(execute_concat(context, batch_1_b, 1, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_OBSERVED &&
              replay_state.plans == 1u && replay_state.ready == 0u &&
              replay_state.captures == replay_checkpoint.captures &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations >
                  replay_checkpoint.invalidations &&
              replay_state.graph_exec_destroys ==
                  replay_checkpoint.graph_exec_destroys + 1u &&
              !replay_state.last_replayed);
    }
    CHECK(execute_concat(context, batch_1_c, 1, 1, 0) == 0);
    CHECK(execute_concat(
              context, batch_1_a, 1, 1, graph_api ? 1 : 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api)
        CHECK(replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.ready == 1u && replay_state.last_replayed);

    replay_checkpoint = replay_state;
    CHECK(vx_public_api_test_cuda_invalidate_replay_key(context, 3) == 0);
    CHECK(execute_concat(context, batch_1_b, 1, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.capacity_generation !=
                  replay_checkpoint.capacity_generation &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_OBSERVED &&
              replay_state.plans == 1u && replay_state.ready == 0u &&
              replay_state.captures == replay_checkpoint.captures &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations >
                  replay_checkpoint.invalidations &&
              replay_state.graph_exec_destroys ==
                  replay_checkpoint.graph_exec_destroys + 1u &&
              !replay_state.last_replayed);
    }

    /* Rebuild B1 and fill the bounded four-entry cache with B1..B4. A failed
     * Driver destroy during B5 eviction must quarantine B1, execute B5
     * directly, and preserve ownership without requesting a fifth GraphExec.
     * The next B5 retries destruction before OBSERVE, then captures normally.
     * Returning to evicted B1 must observe rather than launch stale work. */
    CHECK(execute_concat(context, batch_1_c, 1, 1, 0) == 0);
    CHECK(execute_concat(
              context, batch_1_a, 1, 1, graph_api ? 1 : 0) == 0);
    CHECK(execute_concat(context, batch_2, 2, 0, 0) == 0);
    CHECK(execute_concat(context, batch_2, 2, 1, 0) == 0);
    CHECK(execute_concat(context, batch_3, 3, 0, 0) == 0);
    CHECK(execute_concat(context, batch_3, 3, 1, 0) == 0);
    CHECK(execute_concat(context, batch_4_a, 4, 1, 0) == 0);
    CHECK(execute_concat(context, batch_4_b, 4, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api)
        CHECK(replay_state.plans == 4u && replay_state.ready == 4u &&
              replay_state.destroy_pending == 0u &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              !replay_state.last_replayed);
    replay_checkpoint = replay_state;
    if (graph_api)
        CHECK(vx_public_api_test_cuda_fail_next_graph_exec_destroy(
                  context) == 0);
    CHECK(execute_concat(context, batch_5, 5, 0, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.plans == 3u && replay_state.ready == 3u &&
              replay_state.destroy_pending == 1u &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.captures == replay_checkpoint.captures &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations ==
                  replay_checkpoint.invalidations + 1u &&
              replay_state.graph_exec_destroys ==
                  replay_checkpoint.graph_exec_destroys &&
              !replay_state.last_replayed);
    }
    CHECK(execute_concat(context, batch_5, 5, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.plans == 4u && replay_state.ready == 3u &&
              replay_state.destroy_pending == 0u &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_OBSERVED &&
              replay_state.captures == replay_checkpoint.captures &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations ==
                  replay_checkpoint.invalidations + 1u &&
              replay_state.graph_exec_destroys ==
                  replay_checkpoint.graph_exec_destroys + 1u &&
              !replay_state.last_replayed);
    }
    CHECK(execute_concat(context, batch_5, 5, 1, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.plans == 4u && replay_state.ready == 4u &&
              replay_state.destroy_pending == 0u &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_READY &&
              replay_state.captures == replay_checkpoint.captures + 1u &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations ==
                  replay_checkpoint.invalidations + 1u &&
              replay_state.graph_exec_destroys ==
                  replay_checkpoint.graph_exec_destroys + 1u &&
              !replay_state.last_replayed);
    }
    replay_checkpoint = replay_state;
    CHECK(execute_concat(context, batch_1_b, 1, 0, 0) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.plans == 4u && replay_state.ready == 3u &&
              replay_state.destroy_pending == 0u &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_OBSERVED &&
              replay_state.captures == replay_checkpoint.captures &&
              replay_state.replays == replay_checkpoint.replays &&
              replay_state.invalidations ==
                  replay_checkpoint.invalidations + 1u &&
              replay_state.graph_exec_destroys ==
                  replay_checkpoint.graph_exec_destroys + 1u &&
              !replay_state.last_replayed);
    }
    CHECK(vx_public_api_test_cuda_dynamic_reservation(
              context, &current_allocations, NULL, NULL, NULL,
              &replay_plan) == 0 &&
          current_allocations == reserved_allocations &&
          replay_plan == (graph_api ? PUBLIC_CUDA_REPLAY_OBSERVED
                                    : PUBLIC_CUDA_REPLAY_EMPTY));
    replay_checkpoint = replay_state;
    if (graph_api)
        CHECK(vx_public_api_test_cuda_fail_next_graph_exec_destroy(
                  context) == 0);
    CHECK(vx_public_api_test_cuda_cleanup(context) == 0);
    CHECK(cuda_replay_state(context, &replay_state) == 0);
    if (graph_api) {
        CHECK(replay_state.plans == 0u && replay_state.ready == 0u &&
              replay_state.destroy_pending == 0u &&
              replay_state.active_plan == PUBLIC_CUDA_REPLAY_EMPTY &&
              replay_state.graph_exec_destroys ==
              replay_checkpoint.graph_exec_destroys + 3u &&
              !replay_state.last_replayed);
    }
    CHECK(vx_execution_context_close(context, &report) == VX_STATUS_OK);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);

    CHECK(run_public_layernorm_affines(runtime, layernorm_path) == 0);
    CHECK(run_public_qbatch(runtime, qbatch_path, weight_path) == 0);
    CHECK(run_scheduled_cuda_batch(runtime, scheduled_batch_path) == 0);

    CHECK(rejected_domain(runtime, reject_path, softmax_reject_graph,
                          "softmax-last-axis-domain") == 0);
    CHECK(rejected_domain(runtime, reject_path, broadcast_reject_graph,
                          "binary-broadcast-domain") == 0);
    CHECK(rejected_domain(runtime, reject_path, slice_reject_graph,
                          "slice-output-or-index-domain") == 0);
    vx_runtime_release(runtime);
    CHECK(remove(graph_path) == 0);
    CHECK(remove(weight_path) == 0);
    CHECK(remove(reject_path) == 0);
    CHECK(remove(layernorm_path) == 0);
    CHECK(remove(qbatch_path) == 0);
    CHECK(remove(scheduled_batch_path) == 0);
    puts("CUDA public dynamic shape tests passed");
    return 0;
}
