#include "volvoxai.h"
#include "host_backend.h"
#include "android_nnapi_backend.h"

#include <math.h>
#include <stdio.h>
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

static int write_graph(const char* path) {
    static const char graph[] =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{" 
        "\"a\":{\"shape\":[2],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Add\",\"inputs\":{" 
        "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"sum\"},"
        "\"outputs_shape\":{\"out\":[2]}}],\"outputs\":[\"sum\"]}";
    FILE* file = fopen(path, "wb");
    if (!file) return -1;
    return fwrite(graph, 1, sizeof(graph) - 1u, file) == sizeof(graph) - 1u &&
        fclose(file) == 0 ? 0 : -1;
}

int main(void) {
    const char* graph_path = "/tmp/volvox-example-provider.graph.json";
    const char* required[] = { "example-host" };
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* first = NULL;
    VxExecutionContext* second = NULL;
    VxResult* first_result = NULL;
    VxResult* second_result = NULL;
    const float first_a[2] = {1.0f, 2.0f};
    const float first_b[2] = {3.0f, 4.0f};
    const float second_a[2] = {10.0f, 20.0f};
    const float second_b[2] = {-2.0f, 5.0f};
    float output[2] = {0};

    CHECK(write_graph(graph_path) == 0);
    CHECK(vx_runtime_create(&runtime_options, &runtime, &report) == VX_STATUS_OK);
    CHECK(volvoxai_example_host_backend_register(runtime, &report) == VX_STATUS_OK);
#ifndef __ANDROID__
    CHECK(volvoxai_example_nnapi_backend_register(runtime, &report) ==
          VX_STATUS_BACKEND_UNAVAILABLE);
#endif
    source.graph_path = graph_path;
    CHECK(vx_runtime_load_model(runtime, &source, &model, &report) == VX_STATUS_OK);
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
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
    CHECK(vx_execution_context_set_input(first, "a", VX_DTYPE_F32,
                                         first_a, sizeof(first_a), &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_set_input(first, "b", VX_DTYPE_F32,
                                         first_b, sizeof(first_b), &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_set_input(second, "a", VX_DTYPE_F32,
                                         second_a, sizeof(second_a), &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_set_input(second, "b", VX_DTYPE_F32,
                                         second_b, sizeof(second_b), &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_execute(first, &first_result, &report) ==
          VX_STATUS_OK);
    CHECK(vx_execution_context_execute(second, &second_result, &report) ==
          VX_STATUS_OK);
    CHECK(vx_result_read(first_result, "sum", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(closef(output[0], 4.0f) && closef(output[1], 6.0f));
    CHECK(vx_result_read(second_result, "sum", output, sizeof(output), NULL,
                         &report) == VX_STATUS_OK);
    CHECK(closef(output[0], 8.0f) && closef(output[1], 25.0f));

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
    puts("runtime-scoped backend provider examples passed");
    return 0;
}
