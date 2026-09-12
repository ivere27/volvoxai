#define _POSIX_C_SOURCE 200809L

#include <errno.h>
#include <inttypes.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>

#include "vx_lifecycle.h"

static int parse_positive(const char* value, uint64_t maximum,
                          uint64_t* result) {
    char* end = NULL;
    unsigned long long parsed;
    if (!value || !value[0] || !result) return -1;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno || !end || *end || !parsed || parsed > maximum) return -1;
    *result = (uint64_t)parsed;
    return 0;
}

static int parse_nonnegative(const char* value, uint64_t maximum,
                             uint64_t* result) {
    char* end = NULL;
    unsigned long long parsed;
    if (!value || !value[0] || !result) return -1;
    errno = 0;
    parsed = strtoull(value, &end, 10);
    if (errno || !end || *end || parsed > maximum) return -1;
    *result = (uint64_t)parsed;
    return 0;
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

static uint32_t byte_digest(const void* value, size_t size) {
    const uint8_t* bytes = (const uint8_t*)value;
    uint32_t digest = UINT32_C(0x811c9dc5);
    for (size_t index = 0; index < size; index++) {
        digest ^= bytes[index];
        digest *= UINT32_C(0x01000193);
    }
    return digest;
}

static int report_failure(const char* operation, VxStatus status,
                          const VxReport* report) {
    fprintf(stderr, "%s failed: status=%s code=%d message=%s\n",
            operation, vx_status_string(status),
            report ? report->code : VX_CODE_NONE,
            report && report->message[0] ? report->message : "unknown");
    return 1;
}

static void print_samples(const double* samples, size_t count) {
    putchar('[');
    for (size_t index = 0; index < count; index++) {
        if (index) putchar(',');
        printf("%.9f", samples[index]);
    }
    putchar(']');
}

int main(int argc, char** argv) {
    const char* graph_path;
    const char* selected_backends[] = {"cpu"};
    uint64_t width_value;
    uint64_t warmup_value;
    uint64_t samples_value;
    size_t width;
    size_t byte_size;
    size_t warmup;
    size_t sample_count;
    float* input = NULL;
    float* output = NULL;
    double* samples = NULL;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    VxReport compile_report = VX_REPORT_INIT;
    VxReport context_report = VX_REPORT_INIT;
    VxReport final_execution_report = VX_REPORT_INIT;
    VxRuntime* runtime = NULL;
    VxModel* model = NULL;
    VxCompiledModel* compiled = NULL;
    VxExecutionContext* context = NULL;
    VxResult* final_result = NULL;
    VxTensorSpec input_spec = VX_TENSOR_SPEC_INIT;
    VxTensorBinding input_binding = VX_TENSOR_BINDING_INIT;
    VxStatus status;
    uint64_t rss_baseline;
    uint64_t rss_after_load;
    uint64_t rss_after_compile;
    uint64_t rss_after_context;
    uint64_t rss_after_warmup;
    uint64_t rss_after_measurement;
    uint64_t rss_after_close;
    uint64_t high_water_baseline;
    uint64_t high_water_final;
    uint32_t expected_digest;
    uint32_t actual_digest;
    int parity;
    int return_code = 1;

    if (argc != 5 ||
        parse_positive(argv[2], UINT64_C(1048576), &width_value) != 0 ||
        parse_nonnegative(argv[3], UINT64_C(1000), &warmup_value) != 0 ||
        parse_positive(argv[4], UINT64_C(1001), &samples_value) != 0) {
        fprintf(stderr,
                "usage: %s <graph.json> <width> <warmup> <samples>\n",
                argc > 0 ? argv[0] : "native_runtime_baseline");
        return 2;
    }
    graph_path = argv[1];
    width = (size_t)width_value;
    warmup = (size_t)warmup_value;
    sample_count = (size_t)samples_value;
    if (width > SIZE_MAX / sizeof(float)) {
        fprintf(stderr, "native baseline width overflows size_t\n");
        return 2;
    }
    byte_size = width * sizeof(float);
    input = (float*)malloc(byte_size);
    output = (float*)malloc(byte_size);
    samples = (double*)calloc(sample_count, sizeof(*samples));
    if (!input || !output || !samples) {
        fprintf(stderr, "native baseline caller-buffer allocation failed\n");
        goto cleanup;
    }
    for (size_t index = 0; index < width; index++)
        input[index] = (float)(index % 257u) / 257.0f;
    memset(output, 0, byte_size);
    expected_digest = byte_digest(input, byte_size);

    /* Caller-owned input, output, and sample buffers are intentionally present
       in the baseline so process RSS deltas describe retained runtime state. */
    rss_baseline = current_rss_bytes();
    high_water_baseline = process_high_water_bytes();

    runtime_options.cpu_threads = 1;
    source.graph_path = graph_path;
    policy.mode = VX_BACKEND_REQUIRE;
    policy.operator_fallback = VX_OPERATOR_FALLBACK_FORBID;
    policy.backends = selected_backends;
    policy.backend_count = 1;

    status = vx_runtime_create(&runtime_options, &runtime, &report);
    if (status != VX_STATUS_OK) {
        return_code = report_failure("vx_runtime_create", status, &report);
        goto cleanup;
    }
    status = vx_runtime_load_model(runtime, &source, &model, &report);
    if (status != VX_STATUS_OK) {
        return_code = report_failure("vx_runtime_load_model", status, &report);
        goto cleanup;
    }
    rss_after_load = current_rss_bytes();

    status = vx_model_compile(model, &policy, &compiled, &report);
    if (status != VX_STATUS_OK) {
        return_code = report_failure("vx_model_compile", status, &report);
        goto cleanup;
    }
    compile_report = report;
    rss_after_compile = current_rss_bytes();

    status = vx_compiled_model_create_context(
        compiled, &context_options, &context, &report);
    if (status != VX_STATUS_OK) {
        return_code = report_failure(
            "vx_compiled_model_create_context", status, &report);
        goto cleanup;
    }
    context_report = report;
    rss_after_context = current_rss_bytes();

    status = vx_execution_context_input_spec(context, 0, &input_spec, &report);
    if (status != VX_STATUS_OK) {
        return_code = report_failure(
            "vx_execution_context_input_spec", status, &report);
        goto cleanup;
    }
    if (!input_spec.name || strcmp(input_spec.name, "x") ||
        input_spec.dtype != VX_DTYPE_F32 || input_spec.rank != 2 ||
        input_spec.dimensions[0].kind != VX_DIMENSION_FIXED ||
        input_spec.dimensions[0].min != 1 ||
        input_spec.dimensions[0].max != 1 ||
        input_spec.dimensions[1].kind != VX_DIMENSION_FIXED ||
        input_spec.dimensions[1].min != (int64_t)width ||
        input_spec.dimensions[1].max != (int64_t)width) {
        fprintf(stderr, "native baseline input descriptor is not [1,width] F32\n");
        goto cleanup;
    }
    input_binding.name = "x";
    input_binding.dtype = VX_DTYPE_F32;
    input_binding.rank = 2u;
    input_binding.shape[0] = 1;
    input_binding.shape[1] = (int64_t)width;
    input_binding.data = input;
    input_binding.byte_size = byte_size;
    input_binding.location = VX_MEMORY_HOST;

    for (size_t index = 0; index < warmup; index++) {
        VxResult* result = NULL;
        status = vx_execution_context_execute(
            context, &input_binding, 1u, &result, &report);
        if (status != VX_STATUS_OK) {
            return_code = report_failure("native warmup execution", status,
                                         &report);
            vx_result_release(result);
            goto cleanup;
        }
        vx_result_release(result);
    }
    rss_after_warmup = current_rss_bytes();

    for (size_t index = 0; index < sample_count; index++) {
        VxResult* result = NULL;
        status = vx_execution_context_execute(
            context, &input_binding, 1u, &result, &report);
        if (status != VX_STATUS_OK) {
            return_code = report_failure("native measured execution", status,
                                         &report);
            vx_result_release(result);
            goto cleanup;
        }
        if (!(report.execution_time_ms >= 0.0)) {
            fprintf(stderr, "native execution reported invalid latency\n");
            vx_result_release(result);
            goto cleanup;
        }
        samples[index] = report.execution_time_ms;
        if (index + 1 == sample_count) {
            final_result = result;
            final_execution_report = report;
        } else {
            vx_result_release(result);
        }
    }
    rss_after_measurement = current_rss_bytes();

    status = vx_result_read(final_result, "y", output, byte_size, NULL, &report);
    if (status != VX_STATUS_OK) {
        return_code = report_failure("vx_result_read", status, &report);
        goto cleanup;
    }
    actual_digest = byte_digest(output, byte_size);
    parity = !memcmp(input, output, byte_size) &&
        actual_digest == expected_digest;
    if (!parity) {
        fprintf(stderr, "native Identity output is not byte-identical to input\n");
        goto cleanup;
    }

    vx_result_release(final_result);
    final_result = NULL;
    (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    context = NULL;
    vx_compiled_model_release(compiled);
    compiled = NULL;
    vx_model_release(model);
    model = NULL;
    (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    runtime = NULL;
    rss_after_close = current_rss_bytes();
    high_water_final = process_high_water_bytes();

    printf("{\"schema\":\"volvoxai.native-runtime-baseline-worker/v1\",");
    printf("\"nativeApiVersion\":%u,", VX_NATIVE_API_VERSION);
    printf("\"workload\":{\"operation\":\"Identity\",\"dtype\":\"float32\","
           "\"shape\":[1,%zu],\"warmupExecutions\":%zu,"
           "\"measuredExecutions\":%zu,\"cpuThreads\":1},",
           width, warmup, sample_count);
    printf("\"latency\":{\"samplesMs\":");
    print_samples(samples, sample_count);
    printf("},\"compilation\":{\"compileTimeMs\":%.9f,"
           "\"reportedAllocationBytes\":%" PRIu64 "},",
           compile_report.compile_time_ms, compile_report.allocated_bytes);
    printf("\"allocation\":{\"contextReportedBytes\":%" PRIu64 ","
           "\"executionWithResultReportedBytes\":%" PRIu64 ","
           "\"resultSnapshotBytes\":%" PRIu64 ","
           "\"logicalInputBytes\":%zu,\"logicalOutputBytes\":%zu,"
           "\"logicalLiveTensorBytes\":%zu},",
           context_report.allocated_bytes,
           final_execution_report.allocated_bytes,
           final_execution_report.result_bytes,
           byte_size, byte_size, byte_size * 2u);
    printf("\"memory\":{\"metric\":\"process-rss\","
           "\"currentBytes\":{\"retainedBaseline\":%" PRIu64 ","
           "\"afterModelLoad\":%" PRIu64 ","
           "\"afterCompile\":%" PRIu64 ","
           "\"afterContextCreate\":%" PRIu64 ","
           "\"afterWarmup\":%" PRIu64 ","
           "\"afterMeasurement\":%" PRIu64 ","
           "\"afterAllOwnersClosed\":%" PRIu64 "},"
           "\"highWaterBaselineBytes\":%" PRIu64 ","
           "\"highWaterBytes\":%" PRIu64 ","
           "\"highWaterDeltaBytes\":%" PRIu64 "},",
           rss_baseline, rss_after_load, rss_after_compile,
           rss_after_context, rss_after_warmup, rss_after_measurement,
           rss_after_close, high_water_baseline, high_water_final,
           high_water_final >= high_water_baseline
               ? high_water_final - high_water_baseline : 0);
    printf("\"parity\":{\"byteExactIdentity\":true,"
           "\"expectedDigest\":%u,\"actualDigest\":%u}}\n",
           expected_digest, actual_digest);
    return_code = 0;

cleanup:
    vx_result_release(final_result);
    if (context) (void)vx_execution_context_close(context, NULL);
    vx_execution_context_release(context);
    vx_compiled_model_release(compiled);
    vx_model_release(model);
    if (runtime) (void)vx_runtime_close(runtime, NULL);
    vx_runtime_release(runtime);
    free(samples);
    free(output);
    free(input);
    return return_code;
}
