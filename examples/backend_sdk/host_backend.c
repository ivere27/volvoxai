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
    "{\"format\":\"volvox-graph/v1\"," \
    "\"dimensions\":{}," \
    "\"inputs\":{" \
    "\"a\":{\"shape\":[2],\"dtype\":\"float32\"}," \
    "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}}," \
    "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\",\"inputs\":{" \
    "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":{" \
    "\"tensor\":\"sum\",\"dtype\":\"float32\",\"shape\":[2]}}," \
    "\"params\":{}}],\"outputs\":[\"sum\"]}"

typedef struct ExampleHostRuntime {
    atomic_uint_fast64_t executions;
} ExampleHostRuntime;

typedef struct ExampleHostCompiled {
    ExampleHostRuntime* runtime;
} ExampleHostCompiled;

typedef struct ExampleHostContext {
    ExampleHostCompiled* compiled;
    int closed;
} ExampleHostContext;

static void example_report(VxReport* report, VxStatus status, VxStage stage,
                           const char* reason, const char* message) {
    if (!report || report->struct_size != sizeof(*report)) return;
    memset(report, 0, sizeof(*report));
    report->struct_size = sizeof(*report);
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
    if (!policy || policy->struct_size != sizeof(*policy) ||
        !policy->backends) return 0;
    for (size_t index = 0; index < policy->backend_count; index++)
        if (policy->backends[index] && !strcmp(policy->backends[index], name))
            return 1;
    return 0;
}

static int example_tensor_spec_matches(const VxTensorSpec* spec,
                                       const char* name) {
    const VxDimensionConstraint* dimension;
    if (!spec || spec->struct_size != sizeof(*spec) || !spec->name ||
        strcmp(spec->name, name) || spec->dtype != VX_DTYPE_F32 ||
        spec->rank != 1u || spec->location != VX_MEMORY_HOST) return 0;
    dimension = &spec->dimensions[0];
    return dimension->struct_size == sizeof(*dimension) &&
        dimension->kind == VX_DIMENSION_FIXED && dimension->symbol == NULL &&
        dimension->min == 2 && dimension->max == 2 &&
        dimension->multiple_of == 1;
}

static VxStatus example_runtime_create(void* user_data,
                                       const VxRuntimeOptions* options,
                                       void** out_runtime,
                                       VxReport* report) {
    ExampleHostRuntime* runtime;
    (void)user_data;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = NULL;
    if (!options || options->struct_size != sizeof(*options))
        return VX_STATUS_INVALID_ARGUMENT;
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
                                const VxBackendCompileInput* input,
                                const VxBackendPolicy* policy,
                                void** out_compiled,
                                VxBackendShapeDomainAttestation* attestation,
                                VxReport* report) {
    ExampleHostCompiled* compiled;
    if (!runtime_instance || !input ||
        input->struct_size != sizeof(*input) || !input->source ||
        input->source->struct_size != sizeof(*input->source) ||
        !input->graph_fingerprint || !input->shape_domain_proof_identity ||
        !out_compiled || !attestation ||
        attestation->struct_size != sizeof(*attestation) ||
        !policy_contains(policy, "example-host"))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_compiled = NULL;
    if (input->input_count != 2u || input->output_count != 1u) {
        example_report(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_COMPILE,
                       "BACKEND_UNSUPPORTED",
                       "example provider accepts only its documented Add graph");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    if (!input->inputs || !input->outputs ||
        !example_tensor_spec_matches(&input->inputs[0], "a") ||
        !example_tensor_spec_matches(&input->inputs[1], "b") ||
        !example_tensor_spec_matches(&input->outputs[0], "sum"))
        return VX_STATUS_INVALID_ARGUMENT;
    if (!example_graph_matches(input->source->graph_path)) {
        example_report(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_COMPILE,
                       "BACKEND_UNSUPPORTED",
                       "example provider accepts only its documented Add graph");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    compiled = (ExampleHostCompiled*)calloc(1, sizeof(*compiled));
    if (!compiled) return VX_STATUS_OUT_OF_MEMORY;
    compiled->runtime = (ExampleHostRuntime*)runtime_instance;
    *out_compiled = compiled;
    attestation->graph_fingerprint = input->graph_fingerprint;
    attestation->shape_domain_proof_identity =
        input->shape_domain_proof_identity;
    attestation->maximum_tensor_bytes = sizeof(float) * 2u;
    attestation->maximum_resident_bytes = sizeof(float) * 6u;
    attestation->resource_limit_bytes = 1024u;
    attestation->has_resource_limit = 1;
    example_report(report, VX_STATUS_OK, VX_STAGE_COMPILE, "OK",
                   "example Add graph compiled");
    if (report && report->struct_size == sizeof(*report)) {
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
    if (!out_context) return VX_STATUS_INVALID_ARGUMENT;
    *out_context = NULL;
    if (!compiled_instance || !options ||
        options->struct_size != sizeof(*options))
        return VX_STATUS_INVALID_ARGUMENT;
    if (options->decode_row_mode != VX_DECODE_ROW_DISABLED)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    context = (ExampleHostContext*)calloc(1, sizeof(*context));
    if (!context) return VX_STATUS_OUT_OF_MEMORY;
    context->compiled = (ExampleHostCompiled*)compiled_instance;
    *out_context = context;
    return VX_STATUS_OK;
}

static VxStatus example_context_execute(void* context_instance,
                                        const VxTensorBinding* inputs,
                                        size_t input_count,
                                        const VxBackendOutputSink* sink,
                                        VxReport* report) {
    ExampleHostContext* context = (ExampleHostContext*)context_instance;
    const float* a = NULL;
    const float* b = NULL;
    const int64_t shape[1] = {2};
    float sum[2];
    VxStatus status;
    if (!context || context->closed || !inputs || input_count != 2u ||
        !sink || sink->struct_size != sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < input_count; index++) {
        const VxTensorBinding* binding = &inputs[index];
        if (binding->struct_size != sizeof(*binding) || !binding->name ||
            binding->dtype != VX_DTYPE_F32 || binding->rank != 1u ||
            binding->shape[0] != 2 || binding->location != VX_MEMORY_HOST ||
            !binding->data || binding->byte_size != sizeof(sum))
            return VX_STATUS_INVALID_ARGUMENT;
        if (!strcmp(binding->name, "a") && !a)
            a = (const float*)binding->data;
        else if (!strcmp(binding->name, "b") && !b)
            b = (const float*)binding->data;
        else
            return VX_STATUS_INVALID_ARGUMENT;
    }
    if (!a || !b) return VX_STATUS_INVALID_ARGUMENT;
    sum[0] = a[0] + b[0];
    sum[1] = a[1] + b[1];
    status = sink->write(sink->user_data, "sum", VX_DTYPE_F32, shape, 1,
                         sum, sizeof(sum));
    if (status != VX_STATUS_OK) return status;
    atomic_fetch_add_explicit(&context->compiled->runtime->executions, 1,
                              memory_order_relaxed);
    example_report(report, VX_STATUS_OK, VX_STAGE_EXECUTE, "OK",
                   "example Add executed");
    if (report && report->struct_size == sizeof(*report)) {
        report->route_attested = 1;
        report->operator_fallback_used = 0;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=example-host;nodes=1;route=example-host-all");
    }
    return VX_STATUS_OK;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
static VxStatus example_layout_test_write(void* user_data,
                                          const char* name,
                                          VxDataType dtype,
                                          const int64_t* shape,
                                          uint32_t rank,
                                          const void* data,
                                          size_t byte_size) {
    (void)user_data;
    (void)name;
    (void)dtype;
    (void)shape;
    (void)rank;
    (void)data;
    (void)byte_size;
    return VX_STATUS_OK;
}

int volvoxai_example_host_backend_test_oversized_descriptors(void) {
    const char* backend_names[1] = {"example-host"};
    float a[2] = {1.0f, 2.0f};
    float b[2] = {3.0f, 4.0f};
    ExampleHostRuntime runtime;
    ExampleHostCompiled compiled = {&runtime};
    ExampleHostContext context = {&compiled, 0};
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxTensorSpec inputs[2] = {VX_TENSOR_SPEC_INIT, VX_TENSOR_SPEC_INIT};
    VxTensorSpec outputs[1] = {VX_TENSOR_SPEC_INIT};
    VxBackendCompileInput input = VX_BACKEND_COMPILE_INPUT_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxBackendShapeDomainAttestation attestation =
        VX_BACKEND_SHAPE_DOMAIN_ATTESTATION_INIT;
    VxReport report = VX_REPORT_INIT;
    VxTensorBinding bindings[2] = {
        {sizeof(VxTensorBinding), "a", VX_DTYPE_F32, 1u, {2},
         a, sizeof(a), VX_MEMORY_HOST},
        {sizeof(VxTensorBinding), "b", VX_DTYPE_F32, 1u, {2},
         b, sizeof(b), VX_MEMORY_HOST},
    };
    VxBackendOutputSink sink = {
        sizeof(VxBackendOutputSink), NULL, example_layout_test_write,
    };
    void* runtime_instance = &runtime;
    void* compiled_instance = NULL;
    void* context_instance = &context;

    atomic_init(&runtime.executions, 0);
    source.graph_path = "unused";
    for (size_t index = 0; index < 2u; index++) {
        inputs[index].name = index ? "b" : "a";
        inputs[index].dtype = VX_DTYPE_F32;
        inputs[index].rank = 1u;
        inputs[index].location = VX_MEMORY_HOST;
        inputs[index].dimensions[0] =
            (VxDimensionConstraint)VX_DIMENSION_CONSTRAINT_INIT;
        inputs[index].dimensions[0].min = 2;
        inputs[index].dimensions[0].max = 2;
    }
    outputs[0] = inputs[0];
    outputs[0].name = "sum";
    input.source = &source;
    input.graph_fingerprint = "graph";
    input.shape_domain_proof_identity = "proof";
    input.inputs = inputs;
    input.input_count = 2u;
    input.outputs = outputs;
    input.output_count = 1u;
    policy.backends = backend_names;
    policy.backend_count = 1u;

    runtime_options.struct_size++;
    if (example_runtime_create(NULL, &runtime_options, &runtime_instance,
                               &report) != VX_STATUS_INVALID_ARGUMENT ||
        runtime_instance != NULL)
        return -1;
    runtime_options.struct_size = sizeof(runtime_options);
    policy.struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT)
        return -1;
    policy.struct_size = sizeof(policy);
    context_options.struct_size++;
    if (example_context_create(&compiled, &context_options, &context_instance,
                               &report) != VX_STATUS_INVALID_ARGUMENT ||
        context_instance != NULL)
        return -1;
    context_options.struct_size = sizeof(context_options);
    input.struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT)
        return -1;
    input.struct_size = sizeof(input);
    attestation.struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT)
        return -1;
    attestation.struct_size = sizeof(attestation);
    source.struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT)
        return -1;
    source.struct_size = sizeof(source);
    inputs[0].struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT)
        return -1;
    inputs[0].struct_size = sizeof(inputs[0]);
    inputs[0].dimensions[0].struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT)
        return -1;
    inputs[0].dimensions[0].struct_size =
        sizeof(inputs[0].dimensions[0]);
    sink.struct_size++;
    if (example_context_execute(&context, bindings, 2u, &sink, &report) !=
        VX_STATUS_INVALID_ARGUMENT) return -1;
    sink.struct_size = sizeof(sink);
    bindings[0].struct_size++;
    if (example_context_execute(&context, bindings, 2u, &sink, &report) !=
        VX_STATUS_INVALID_ARGUMENT) return -1;
    bindings[0].struct_size = sizeof(bindings[0]);
    report.struct_size++;
    report.status = VX_STATUS_INTERNAL;
    snprintf(report.reason, sizeof(report.reason), "%s", "UNCHANGED");
    if (example_context_execute(&context, bindings, 2u, &sink, &report) !=
        VX_STATUS_OK) return -1;
    if (report.status != VX_STATUS_INTERNAL ||
        strcmp(report.reason, "UNCHANGED")) return -1;
    return 0;
}
#endif

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
        .shape_domain = {
            .struct_size = sizeof(VxBackendShapeDomainCapability),
            .proof_protocol = VX_BACKEND_SHAPE_PROOF_PROTOCOL,
            .resource_protocol = VX_BACKEND_RESOURCE_PROTOCOL,
            .support = VX_BACKEND_SHAPE_DOMAIN_FULL,
        },
        .runtime_create = example_runtime_create,
        .runtime_destroy = example_runtime_destroy,
        .compile = example_compile,
        .compiled_destroy = example_compiled_destroy,
        .context_create = example_context_create,
        .context_execute = example_context_execute,
        .context_select_adapter = NULL,
        .context_close = example_context_close,
        .context_destroy = example_context_destroy,
    };
    return vx_runtime_register_provider(runtime, &provider, report);
}
