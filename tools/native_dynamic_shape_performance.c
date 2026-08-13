#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>
#include <unistd.h>

#include "volvoxai.h"

typedef struct Route {
    VxModel* model;
    VxCompiledModel* compiled;
    VxExecutionContext* context;
    VxReport compile_report;
} Route;

typedef struct Sample {
    double wall_ms;
    double execution_ms;
    double shape_bind_ms;
    uint64_t logical_bytes;
    uint64_t arena_required;
    uint64_t arena_capacity;
    uint64_t arena_high_water;
    uint64_t arena_grows;
    uint64_t resource_generation;
    int plan_hit;
} Sample;

static int parse_positive(const char* text, uint64_t maximum, uint64_t* out) {
    char* end = NULL;
    unsigned long long value;
    if (!text || !text[0] || !out) return -1;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end || !value || value > maximum) return -1;
    *out = (uint64_t)value;
    return 0;
}

static int parse_nonnegative(const char* text, uint64_t maximum,
                             uint64_t* out) {
    char* end = NULL;
    unsigned long long value;
    if (!text || !text[0] || !out) return -1;
    errno = 0;
    value = strtoull(text, &end, 10);
    if (errno || !end || *end || value > maximum) return -1;
    *out = (uint64_t)value;
    return 0;
}

static double monotonic_ms(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec * 1000.0 + (double)value.tv_nsec / 1000000.0;
}

static uint64_t current_rss_bytes(void) {
#if defined(__linux__)
    FILE* stream = fopen("/proc/self/statm", "r");
    unsigned long long total_pages = 0;
    unsigned long long resident_pages = 0;
    long page_size = sysconf(_SC_PAGESIZE);
    if (!stream || page_size <= 0) {
        if (stream) fclose(stream);
        return 0;
    }
    if (fscanf(stream, "%llu %llu", &total_pages, &resident_pages) != 2) {
        fclose(stream);
        return 0;
    }
    fclose(stream);
    if (resident_pages > UINT64_MAX / (uint64_t)page_size) return 0;
    return (uint64_t)resident_pages * (uint64_t)page_size;
#else
    return 0;
#endif
}

static uint64_t process_high_water_bytes(void) {
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss < 0) return 0;
#if defined(__APPLE__)
    return (uint64_t)usage.ru_maxrss;
#else
    if ((uint64_t)usage.ru_maxrss > UINT64_MAX / UINT64_C(1024)) return 0;
    return (uint64_t)usage.ru_maxrss * UINT64_C(1024);
#endif
}

static uint64_t route_u64(const char* evidence, const char* name) {
    const char* value;
    char* end = NULL;
    unsigned long long parsed;
    if (!evidence || !name) return 0;
    value = strstr(evidence, name);
    if (!value) return 0;
    value += strlen(name);
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno || end == value) return 0;
    return (uint64_t)parsed;
}

static double route_double(const char* evidence, const char* name) {
    const char* value;
    char* end = NULL;
    double parsed;
    if (!evidence || !name) return 0.0;
    value = strstr(evidence, name);
    if (!value) return 0.0;
    value += strlen(name);
    errno = 0;
    parsed = strtod(value, &end);
    if (errno || end == value) return 0.0;
    return parsed;
}

static int report_failure(const char* operation, VxStatus status,
                          const VxReport* report) {
    fprintf(stderr, "%s failed: status=%s reason=%s message=%s route=%s\n",
            operation, vx_status_string(status),
            report && report->reason[0] ? report->reason : "unknown",
            report && report->message[0] ? report->message : "unknown",
            report && report->route_evidence[0]
                ? report->route_evidence : "none");
    return -1;
}

static int create_route(VxRuntime* runtime, const char* graph_path,
                        Route* route, VxReport* report) {
    const char* backends[] = {"cpu"};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxStatus status;
    if (!runtime || !graph_path || !route || !report) return -1;
    memset(route, 0, sizeof(*route));
    source.graph_path = graph_path;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = backends;
    policy.backend_count = 1;
    status = vx_runtime_load_model(runtime, &source, &route->model, report);
    if (status != VX_STATUS_OK)
        return report_failure("vx_runtime_load_model", status, report);
    status = vx_model_compile(route->model, &policy, &route->compiled, report);
    if (status != VX_STATUS_OK)
        return report_failure("vx_model_compile", status, report);
    route->compile_report = *report;
    status = vx_compiled_model_create_context(
        route->compiled, &context_options, &route->context, report);
    if (status != VX_STATUS_OK)
        return report_failure("vx_compiled_model_create_context", status, report);
    return 0;
}

static void release_route(Route* route) {
    if (!route) return;
    if (route->context) (void)vx_execution_context_close(route->context, NULL);
    vx_execution_context_release(route->context);
    vx_compiled_model_release(route->compiled);
    vx_model_release(route->model);
    memset(route, 0, sizeof(*route));
}

static int execute_width(Route* route, size_t width, const float* a,
                         const float* b, float* verification, int dynamic,
                         Sample* sample) {
    VxTensorBinding inputs[2] = {VX_TENSOR_BINDING_INIT, VX_TENSOR_BINDING_INIT};
    VxTensorInfo info = VX_TENSOR_INFO_INIT;
    VxReport report = VX_REPORT_INIT;
    VxReport execution_report = VX_REPORT_INIT;
    VxResult* result = NULL;
    VxStatus status;
    double started;
    double wall_ms;
    size_t byte_size;
    int return_code = -1;
    if (!route || !route->context || !a || !b || !sample ||
        width > SIZE_MAX / sizeof(float)) return -1;
    byte_size = width * sizeof(float);
    inputs[0].name = "a";
    inputs[0].dtype = VX_DTYPE_F32;
    inputs[0].rank = 1u;
    inputs[0].shape[0] = (int64_t)width;
    inputs[0].data = a;
    inputs[0].byte_size = byte_size;
    inputs[0].location = VX_MEMORY_HOST;
    inputs[1] = inputs[0];
    inputs[1].name = "b";
    inputs[1].data = b;
    started = monotonic_ms();
    status = vx_execution_context_execute(
        route->context, inputs, 2u, &result, &report);
    wall_ms = monotonic_ms() - started;
    if (status != VX_STATUS_OK) {
        (void)report_failure("vx_execution_context_execute", status, &report);
        goto done;
    }
    execution_report = report;
    status = vx_result_output_info(result, 0u, &info, &report);
    if (status != VX_STATUS_OK || info.rank != 1u ||
        info.shape[0] != (int64_t)width || info.byte_size != byte_size) {
        fprintf(stderr, "native dynamic result descriptor mismatch at %zu\n", width);
        goto done;
    }
    if (verification) {
        status = vx_result_read(
            result, "sum", verification, byte_size, NULL, &report);
        if (status != VX_STATUS_OK) {
            (void)report_failure("vx_result_read", status, &report);
            goto done;
        }
        for (size_t index = 0; index < width; index++) {
            if (verification[index] != a[index] + b[index]) {
                fprintf(stderr, "native dynamic parity mismatch at %zu/%zu\n",
                        index, width);
                goto done;
            }
        }
    }
    memset(sample, 0, sizeof(*sample));
    sample->wall_ms = wall_ms;
    sample->execution_ms = execution_report.execution_time_ms;
    if (dynamic) {
        if (!strstr(execution_report.route_evidence, "shape_signature=") ||
            !strstr(execution_report.route_evidence, "shape_plan=")) {
            fprintf(stderr, "native dynamic execution omitted shape telemetry\n");
            goto done;
        }
        sample->plan_hit =
            strstr(execution_report.route_evidence, "shape_plan=hit") != NULL;
        sample->shape_bind_ms = route_double(
            execution_report.route_evidence, "shape_bind_ms=");
        sample->logical_bytes = route_u64(
            execution_report.route_evidence, "logical_bytes=");
        sample->arena_required = route_u64(
            execution_report.route_evidence, "arena_required=");
        sample->arena_capacity = route_u64(
            execution_report.route_evidence, "arena_capacity=");
        sample->arena_high_water = route_u64(
            execution_report.route_evidence, "arena_high_water=");
        sample->arena_grows = route_u64(
            execution_report.route_evidence, "arena_grows=");
        sample->resource_generation =
            route_u64(execution_report.route_evidence, "resource_generation=");
    }
    return_code = 0;
done:
    vx_result_release(result);
    return return_code;
}

static void print_sample(const Sample* sample) {
    printf("{\"wallMs\":%.9f,\"executionMs\":%.9f,"
           "\"shapeBindMs\":%.9f,\"plan\":\"%s\","
           "\"logicalBytes\":%" PRIu64 ",\"arenaRequired\":%" PRIu64 ","
           "\"arenaCapacity\":%" PRIu64 ",\"arenaHighWater\":%" PRIu64 ","
           "\"arenaGrows\":%" PRIu64 ",\"resourceGeneration\":%" PRIu64 "}",
           sample->wall_ms, sample->execution_ms, sample->shape_bind_ms,
           sample->plan_hit ? "hit" : "cold", sample->logical_bytes,
           sample->arena_required, sample->arena_capacity,
           sample->arena_high_water, sample->arena_grows,
           sample->resource_generation);
}

static void print_latencies(const Sample* samples, size_t count) {
    putchar('[');
    for (size_t index = 0; index < count; index++) {
        if (index) putchar(',');
        printf("%.9f", samples[index].execution_ms);
    }
    putchar(']');
}

int main(int argc, char** argv) {
    uint64_t active_value;
    uint64_t maximum_value;
    uint64_t warmup_value;
    uint64_t samples_value;
    size_t active;
    size_t maximum;
    size_t warmup;
    size_t sample_count;
    size_t adversarial_widths[] = {2u, 4u, 8u, 16u, 32u, 64u, 2u};
    float* a = NULL;
    float* b = NULL;
    float* verification = NULL;
    Sample* dynamic_samples = NULL;
    Sample* padded_samples = NULL;
    Sample cold_active = {0};
    Sample cold_maximum = {0};
    Sample return_active = {0};
    Sample cold_padded = {0};
    Sample adversarial[7] = {{0}};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    Route dynamic_route = {0};
    Route padded_route = {0};
    uint64_t rss_before;
    uint64_t rss_after_compile;
    uint64_t rss_after_measurement;
    uint64_t high_water;
    VxStatus status;
    int return_code = 1;

    if (argc != 7 ||
        parse_positive(argv[3], UINT64_C(1048576), &active_value) != 0 ||
        parse_positive(argv[4], UINT64_C(1048576), &maximum_value) != 0 ||
        parse_nonnegative(argv[5], UINT64_C(1000), &warmup_value) != 0 ||
        parse_positive(argv[6], UINT64_C(1001), &samples_value) != 0 ||
        active_value < 2u || active_value > maximum_value || maximum_value < 64u ||
        (active_value & 1u) || (maximum_value & 1u)) {
        fprintf(stderr, "usage: %s <dynamic.graph.json> <padded.graph.json> "
                "<active-even> <maximum-even>=64+ <warmup> <samples>\n",
                argc > 0 ? argv[0] : "native_dynamic_shape_performance");
        return 2;
    }
    active = (size_t)active_value;
    maximum = (size_t)maximum_value;
    warmup = (size_t)warmup_value;
    sample_count = (size_t)samples_value;
    a = (float*)malloc(maximum * sizeof(float));
    b = (float*)malloc(maximum * sizeof(float));
    verification = (float*)malloc(maximum * sizeof(float));
    dynamic_samples = (Sample*)calloc(sample_count, sizeof(*dynamic_samples));
    padded_samples = (Sample*)calloc(sample_count, sizeof(*padded_samples));
    if (!a || !b || !verification || !dynamic_samples || !padded_samples) {
        fprintf(stderr, "native dynamic benchmark allocation failed\n");
        goto cleanup;
    }
    for (size_t index = 0; index < maximum; index++) {
        a[index] = (float)((int)(index % 251u) - 125) / 17.0f;
        b[index] = (float)((int)(index % 127u) - 63) / 19.0f;
    }
    runtime_options.cpu_threads = 1;
    rss_before = current_rss_bytes();
    status = vx_runtime_create(&runtime_options, &runtime, &report);
    if (status != VX_STATUS_OK) {
        (void)report_failure("vx_runtime_create", status, &report);
        goto cleanup;
    }
    if (create_route(runtime, argv[1], &dynamic_route, &report) != 0 ||
        create_route(runtime, argv[2], &padded_route, &report) != 0) goto cleanup;
    rss_after_compile = current_rss_bytes();

    if (execute_width(&dynamic_route, active, a, b, verification, 1,
                      &cold_active) != 0 || cold_active.plan_hit ||
        execute_width(&dynamic_route, maximum, a, b, verification, 1,
                      &cold_maximum) != 0 || cold_maximum.plan_hit ||
        execute_width(&dynamic_route, active, a, b, verification, 1,
                      &return_active) != 0 || !return_active.plan_hit ||
        execute_width(&padded_route, maximum, a, b, verification, 0,
                      &cold_padded) != 0) goto cleanup;

    for (size_t index = 0; index < warmup; index++) {
        Sample discard = {0};
        if (execute_width(&dynamic_route, active, a, b, NULL, 1, &discard) != 0 ||
            execute_width(&padded_route, maximum, a, b, NULL, 0, &discard) != 0)
            goto cleanup;
    }
    for (size_t index = 0; index < sample_count; index++) {
        if ((index & 1u) == 0u) {
            if (execute_width(&dynamic_route, active, a, b, NULL, 1,
                              &dynamic_samples[index]) != 0 ||
                execute_width(&padded_route, maximum, a, b, NULL, 0,
                              &padded_samples[index]) != 0) goto cleanup;
        } else {
            if (execute_width(&padded_route, maximum, a, b, NULL, 0,
                              &padded_samples[index]) != 0 ||
                execute_width(&dynamic_route, active, a, b, NULL, 1,
                              &dynamic_samples[index]) != 0) goto cleanup;
        }
        if (!dynamic_samples[index].plan_hit) {
            fprintf(stderr, "warm dynamic execution unexpectedly missed its plan\n");
            goto cleanup;
        }
    }
    for (size_t index = 0; index < 7u; index++) {
        if (adversarial_widths[index] > maximum ||
            execute_width(&dynamic_route, adversarial_widths[index], a, b,
                          NULL, 1, &adversarial[index]) != 0) goto cleanup;
    }
    if (adversarial[6].plan_hit) {
        fprintf(stderr, "native dynamic plan cache did not evict deterministically\n");
        goto cleanup;
    }
    rss_after_measurement = current_rss_bytes();
    high_water = process_high_water_bytes();

    printf("{\"schema\":\"volvoxai.native-dynamic-shape-worker/v1\","
           "\"nativeApiVersion\":%u,\"workload\":{"
           "\"operation\":\"Add\",\"dtype\":\"float32\","
           "\"active\":[%zu],\"paddedMaximum\":[%zu],"
           "\"warmupExecutions\":%zu,\"measuredExecutions\":%zu},"
           "\"compile\":{\"dynamicMs\":%.9f,\"paddedMs\":%.9f},"
           "\"cold\":{\"active\":",
           VX_NATIVE_API_VERSION, active, maximum, warmup, sample_count,
           dynamic_route.compile_report.compile_time_ms,
           padded_route.compile_report.compile_time_ms);
    print_sample(&cold_active);
    printf(",\"maximum\":");
    print_sample(&cold_maximum);
    printf(",\"returnActive\":");
    print_sample(&return_active);
    printf(",\"padded\":");
    print_sample(&cold_padded);
    printf("},\"warm\":{\"dynamicActiveMs\":");
    print_latencies(dynamic_samples, sample_count);
    printf(",\"paddedMaximumMs\":");
    print_latencies(padded_samples, sample_count);
    printf("},\"adversarial\":[");
    for (size_t index = 0; index < 7u; index++) {
        if (index) putchar(',');
        printf("{\"width\":%zu,\"sample\":", adversarial_widths[index]);
        print_sample(&adversarial[index]);
        putchar('}');
    }
    printf("],\"parity\":{\"dynamicActive\":true,"
           "\"dynamicMaximum\":true,\"paddedMaximum\":true},"
           "\"memory\":{\"metric\":\"process-rss\","
           "\"beforeRuntimeBytes\":%" PRIu64 ","
           "\"afterCompileBytes\":%" PRIu64 ","
           "\"afterMeasurementBytes\":%" PRIu64 ","
           "\"processHighWaterBytes\":%" PRIu64 "}}\n",
           rss_before, rss_after_compile, rss_after_measurement, high_water);
    return_code = 0;

cleanup:
    release_route(&padded_route);
    release_route(&dynamic_route);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    free(padded_samples);
    free(dynamic_samples);
    free(verification);
    free(b);
    free(a);
    return return_code;
}
