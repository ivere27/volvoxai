/* Minimal provider built from installed public headers only.
 *
 * To keep the example self-contained, its compiler accepts one exact graph:
 * rank-1 F32 inputs `a` and `b`, one Add node, and output `sum`, all with two
 * elements. A real provider would replace this small recognizer with its graph
 * compiler and retain an immutable compiled plan in ExampleHostCompiled. */

#include "host_backend.h"

#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define EXAMPLE_HOST_GRAPH \
    "{\"format\":\"volvox-graph/v1\",\"inputs\":{" \
    "\"a\":{\"shape\":[2],\"dtype\":\"float32\"}," \
    "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}}," \
    "\"nodes\":[{\"opType\":\"Add\",\"inputs\":{" \
    "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"sum\"}," \
    "\"outputs_shape\":{\"out\":[2]}}],\"outputs\":[\"sum\"]}"

typedef struct ExampleHostRuntime {
    atomic_uint_fast64_t executions;
} ExampleHostRuntime;

typedef struct ExampleHostCompiled {
    ExampleHostRuntime* runtime;
} ExampleHostCompiled;

typedef struct ExampleHostContext {
    ExampleHostCompiled* compiled;
    float a[2];
    float b[2];
    int has_a;
    int has_b;
    int closed;
} ExampleHostContext;

static void example_report(VxReport* report, VxStatus status, VxStage stage,
                           const char* reason, const char* message) {
    size_t struct_size;
    if (!report || report->struct_size < sizeof(*report)) return;
    struct_size = report->struct_size;
    memset(report, 0, sizeof(*report));
    report->struct_size = struct_size;
    report->status = status;
    report->stage = stage;
    snprintf(report->backend, sizeof(report->backend), "%s", "example-host");
    snprintf(report->device, sizeof(report->device), "%s", "host-cpu");
    snprintf(report->reason, sizeof(report->reason), "%s", reason);
    snprintf(report->message, sizeof(report->message), "%s", message);
}

static int example_graph_matches(const char* path) {
    FILE* file;
    char* bytes;
    long length;
    size_t read_count;
    int matches;
    if (!path) return 0;
    file = fopen(path, "rb");
    if (!file) return 0;
    if (fseek(file, 0, SEEK_END) != 0 || (length = ftell(file)) < 0 ||
        fseek(file, 0, SEEK_SET) != 0) {
        fclose(file);
        return 0;
    }
    bytes = (char*)malloc((size_t)length + 1u);
    if (!bytes) {
        fclose(file);
        return 0;
    }
    read_count = fread(bytes, 1, (size_t)length, file);
    matches = read_count == (size_t)length && fclose(file) == 0;
    bytes[read_count] = '\0';
    matches = matches && !strcmp(bytes, EXAMPLE_HOST_GRAPH);
    free(bytes);
    return matches;
}

static int policy_contains(const VxBackendPolicy* policy, const char* name) {
    if (!policy || !policy->backends) return 0;
    for (size_t index = 0; index < policy->backend_count; index++)
        if (policy->backends[index] && !strcmp(policy->backends[index], name))
            return 1;
    return 0;
}

static VxStatus example_runtime_create(void* user_data,
                                       const VxRuntimeOptions* options,
                                       void** out_runtime,
                                       VxReport* report) {
    ExampleHostRuntime* runtime;
    (void)user_data;
    (void)options;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = NULL;
    runtime = (ExampleHostRuntime*)calloc(1, sizeof(*runtime));
    if (!runtime) return VX_STATUS_OUT_OF_MEMORY;
    atomic_init(&runtime->executions, 0);
    *out_runtime = runtime;
    example_report(report, VX_STATUS_OK, VX_STAGE_RUNTIME_CREATE, "OK",
                   "example host provider runtime created");
    return VX_STATUS_OK;
}

static void example_runtime_destroy(void* runtime_instance) {
    free(runtime_instance);
}

static VxStatus example_compile(void* runtime_instance,
                                const VxModelSource* source,
                                const VxBackendPolicy* policy,
                                void** out_compiled,
                                VxReport* report) {
    ExampleHostCompiled* compiled;
    if (!runtime_instance || !source || !out_compiled ||
        !policy_contains(policy, "example-host"))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_compiled = NULL;
    if (!example_graph_matches(source->graph_path)) {
        example_report(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_COMPILE,
                       "BACKEND_UNSUPPORTED",
                       "example provider accepts only its documented Add graph");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    compiled = (ExampleHostCompiled*)calloc(1, sizeof(*compiled));
    if (!compiled) return VX_STATUS_OUT_OF_MEMORY;
    compiled->runtime = (ExampleHostRuntime*)runtime_instance;
    *out_compiled = compiled;
    example_report(report, VX_STATUS_OK, VX_STAGE_COMPILE, "OK",
                   "example Add graph compiled");
    if (report && report->struct_size >= sizeof(*report)) {
        report->route_attested = 1;
        report->operator_fallback_used = 0;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=example-host;nodes=1;route=example-host-all");
        snprintf(report->fallback_evidence,
                 sizeof(report->fallback_evidence), "%s", "operator=none");
    }
    return VX_STATUS_OK;
}

static void example_compiled_destroy(void* compiled_instance) {
    free(compiled_instance);
}

static VxStatus example_context_create(void* compiled_instance,
                                       const VxContextOptions* options,
                                       void** out_context,
                                       VxReport* report) {
    ExampleHostContext* context;
    (void)report;
    if (!compiled_instance || !options || !out_context ||
        options->decode_row_mode != VX_DECODE_ROW_DISABLED)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    *out_context = NULL;
    context = (ExampleHostContext*)calloc(1, sizeof(*context));
    if (!context) return VX_STATUS_OUT_OF_MEMORY;
    context->compiled = (ExampleHostCompiled*)compiled_instance;
    *out_context = context;
    return VX_STATUS_OK;
}

static VxStatus example_context_set_input(void* context_instance,
                                          const char* name,
                                          VxDataType dtype,
                                          const void* data,
                                          size_t byte_size,
                                          VxReport* report) {
    ExampleHostContext* context = (ExampleHostContext*)context_instance;
    (void)report;
    if (!context || context->closed || !name || dtype != VX_DTYPE_F32 ||
        !data || byte_size != sizeof(context->a))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!strcmp(name, "a")) {
        memcpy(context->a, data, sizeof(context->a));
        context->has_a = 1;
    } else if (!strcmp(name, "b")) {
        memcpy(context->b, data, sizeof(context->b));
        context->has_b = 1;
    } else {
        return VX_STATUS_NOT_FOUND;
    }
    return VX_STATUS_OK;
}

static VxStatus example_context_execute(void* context_instance,
                                        const VxBackendOutputSink* sink,
                                        VxReport* report) {
    ExampleHostContext* context = (ExampleHostContext*)context_instance;
    const int64_t shape[1] = {2};
    float sum[2];
    VxStatus status;
    if (!context || context->closed || !context->has_a || !context->has_b ||
        !sink || sink->struct_size < sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
    sum[0] = context->a[0] + context->b[0];
    sum[1] = context->a[1] + context->b[1];
    status = sink->write(sink->user_data, "sum", VX_DTYPE_F32, shape, 1,
                         sum, sizeof(sum));
    if (status != VX_STATUS_OK) return status;
    atomic_fetch_add_explicit(&context->compiled->runtime->executions, 1,
                              memory_order_relaxed);
    example_report(report, VX_STATUS_OK, VX_STAGE_EXECUTE, "OK",
                   "example Add executed");
    if (report && report->struct_size >= sizeof(*report)) {
        report->route_attested = 1;
        report->operator_fallback_used = 0;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=example-host;nodes=1;route=example-host-all");
    }
    return VX_STATUS_OK;
}

static VxStatus example_context_close(void* context_instance,
                                      VxReport* report) {
    ExampleHostContext* context = (ExampleHostContext*)context_instance;
    (void)report;
    if (!context) return VX_STATUS_INVALID_ARGUMENT;
    context->closed = 1;
    return VX_STATUS_OK;
}

static void example_context_destroy(void* context_instance) {
    free(context_instance);
}

VxStatus volvoxai_example_host_backend_register(VxRuntime* runtime,
                                                VxReport* report) {
    const VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = "example-host",
        .runtime_create = example_runtime_create,
        .runtime_destroy = example_runtime_destroy,
        .compile = example_compile,
        .compiled_destroy = example_compiled_destroy,
        .context_create = example_context_create,
        .context_set_input = example_context_set_input,
        .context_execute = example_context_execute,
        .context_select_adapter = NULL,
        .context_close = example_context_close,
        .context_destroy = example_context_destroy,
    };
    return vx_runtime_register_provider(runtime, &provider, report);
}
