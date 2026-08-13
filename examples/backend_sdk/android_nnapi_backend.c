/* Conservative Android NNAPI provider built only against the public ABI.
 * It accepts the exact two-element F32 Add graph documented by the host
 * example and keeps all inputs and NNAPI execution objects context-local. */

#include "android_nnapi_backend.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ANDROID__
#include <android/NeuralNetworks.h>
#endif

#define EXAMPLE_NNAPI_GRAPH \
    "{\"format\":\"volvox-graph/v1\"," \
    "\"dimensions\":{}," \
    "\"inputs\":{" \
    "\"a\":{\"shape\":[2],\"dtype\":\"float32\"}," \
    "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}}," \
    "\"nodes\":[{\"id\":\"add\",\"opType\":\"Add\",\"inputs\":{" \
    "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":{" \
    "\"tensor\":\"sum\",\"dtype\":\"float32\",\"shape\":[2]}}," \
    "\"params\":{}}],\"outputs\":[\"sum\"]}"

typedef struct ExampleNnapiRuntime {
    int ready;
} ExampleNnapiRuntime;

typedef struct ExampleNnapiCompiled {
    ExampleNnapiRuntime* runtime;
} ExampleNnapiCompiled;

typedef struct ExampleNnapiContext {
    ExampleNnapiCompiled* compiled;
    int closed;
} ExampleNnapiContext;

static void example_report(VxReport* report, VxStatus status, VxStage stage,
                           const char* reason, const char* message) {
    if (!report || report->struct_size != sizeof(*report)) return;
    memset(report, 0, sizeof(*report));
    report->struct_size = sizeof(*report);
    report->status = status;
    report->stage = stage;
    snprintf(report->backend, sizeof(report->backend), "%s",
             "android-nnapi-add");
    snprintf(report->device, sizeof(report->device), "%s", "android-nnapi");
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
    matches = matches && !strcmp(bytes, EXAMPLE_NNAPI_GRAPH);
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
    ExampleNnapiRuntime* runtime;
    (void)user_data;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = NULL;
    if (!options || options->struct_size != sizeof(*options))
        return VX_STATUS_INVALID_ARGUMENT;
#ifdef __ANDROID__
    {
        uint32_t device_count = 0;
        if (ANeuralNetworks_getDeviceCount(&device_count) !=
                ANEURALNETWORKS_NO_ERROR || !device_count) {
            example_report(report, VX_STATUS_BACKEND_UNAVAILABLE,
                           VX_STAGE_RUNTIME_CREATE, "BACKEND_UNAVAILABLE",
                           "Android NNAPI device is unavailable");
            return VX_STATUS_BACKEND_UNAVAILABLE;
        }
    }
#else
    example_report(report, VX_STATUS_BACKEND_UNAVAILABLE,
                   VX_STAGE_RUNTIME_CREATE, "BACKEND_UNAVAILABLE",
                   "Android NNAPI is unavailable on this platform");
    return VX_STATUS_BACKEND_UNAVAILABLE;
#endif
    runtime = (ExampleNnapiRuntime*)calloc(1, sizeof(*runtime));
    if (!runtime) return VX_STATUS_OUT_OF_MEMORY;
    runtime->ready = 1;
    *out_runtime = runtime;
    example_report(report, VX_STATUS_OK, VX_STAGE_RUNTIME_CREATE, "OK",
                   "Android NNAPI provider runtime created");
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
    ExampleNnapiCompiled* compiled;
    ExampleNnapiRuntime* runtime = (ExampleNnapiRuntime*)runtime_instance;
    if (!runtime || !runtime->ready || !input ||
        input->struct_size != sizeof(*input) || !input->source ||
        input->source->struct_size != sizeof(*input->source) ||
        !input->graph_fingerprint || !input->shape_domain_proof_identity ||
        !out_compiled || !attestation ||
        attestation->struct_size != sizeof(*attestation) ||
        !policy_contains(policy, "android-nnapi-add"))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_compiled = NULL;
    if (input->input_count != 2u || input->output_count != 1u) {
        example_report(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_COMPILE,
                       "BACKEND_UNSUPPORTED",
                       "NNAPI example accepts only its documented Add graph");
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
                       "NNAPI example accepts only its documented Add graph");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    compiled = (ExampleNnapiCompiled*)calloc(1, sizeof(*compiled));
    if (!compiled) return VX_STATUS_OUT_OF_MEMORY;
    compiled->runtime = runtime;
    *out_compiled = compiled;
    attestation->graph_fingerprint = input->graph_fingerprint;
    attestation->shape_domain_proof_identity =
        input->shape_domain_proof_identity;
    attestation->maximum_tensor_bytes = sizeof(float) * 2u;
    attestation->maximum_resident_bytes = sizeof(float) * 6u;
    attestation->resource_limit_bytes = 1024u;
    attestation->has_resource_limit = 1;
    example_report(report, VX_STATUS_OK, VX_STAGE_COMPILE, "OK",
                   "NNAPI Add graph compiled");
    if (report && report->struct_size == sizeof(*report)) {
        report->route_attested = 1;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=android-nnapi-add;nodes=1;route=nnapi-all");
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
    ExampleNnapiContext* context;
    (void)report;
    if (!out_context) return VX_STATUS_INVALID_ARGUMENT;
    *out_context = NULL;
    if (!compiled_instance || !options ||
        options->struct_size != sizeof(*options))
        return VX_STATUS_INVALID_ARGUMENT;
    if (options->decode_row_mode != VX_DECODE_ROW_DISABLED)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    context = (ExampleNnapiContext*)calloc(1, sizeof(*context));
    if (!context) return VX_STATUS_OUT_OF_MEMORY;
    context->compiled = (ExampleNnapiCompiled*)compiled_instance;
    *out_context = context;
    return VX_STATUS_OK;
}

#ifdef __ANDROID__
static VxStatus example_nnapi_add(const float a[2], const float b[2],
                                  float output[2]) {
    const uint32_t dimensions[1] = {2};
    const uint32_t add_inputs[3] = {0, 1, 2};
    const uint32_t add_outputs[1] = {3};
    const uint32_t model_inputs[2] = {0, 1};
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksCompilation* compilation = NULL;
    ANeuralNetworksExecution* execution = NULL;
    int32_t activation = ANEURALNETWORKS_FUSED_NONE;
    VxStatus result = VX_STATUS_EXECUTION_FAILED;
    ANeuralNetworksOperandType tensor_type = {
        .type = ANEURALNETWORKS_TENSOR_FLOAT32,
        .dimensionCount = 1,
        .dimensions = dimensions,
        .scale = 0.0f,
        .zeroPoint = 0,
    };
    ANeuralNetworksOperandType activation_type = {
        .type = ANEURALNETWORKS_INT32,
        .dimensionCount = 0,
        .dimensions = NULL,
        .scale = 0.0f,
        .zeroPoint = 0,
    };
#define EXAMPLE_NNAPI_TRY(call) \
    do { if ((call) != ANEURALNETWORKS_NO_ERROR) goto done; } while (0)
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_create(&model));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &tensor_type));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &tensor_type));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &activation_type));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &tensor_type));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_setOperandValue(
        model, 2, &activation, sizeof(activation)));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperation(
        model, ANEURALNETWORKS_ADD, 3, add_inputs, 1, add_outputs));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_identifyInputsAndOutputs(
        model, 2, model_inputs, 1, add_outputs));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_finish(model));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksCompilation_create(model, &compilation));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksCompilation_finish(compilation));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_create(compilation, &execution));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_setInput(
        execution, 0, NULL, a, 2u * sizeof(float)));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_setInput(
        execution, 1, NULL, b, 2u * sizeof(float)));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_setOutput(
        execution, 0, NULL, output, 2u * sizeof(float)));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_compute(execution));
    result = VX_STATUS_OK;
done:
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
#undef EXAMPLE_NNAPI_TRY
    return result;
}
#endif

static VxStatus example_context_execute(void* context_instance,
                                        const VxTensorBinding* inputs,
                                        size_t input_count,
                                        const VxBackendOutputSink* sink,
                                        VxReport* report) {
    ExampleNnapiContext* context = (ExampleNnapiContext*)context_instance;
    const float* a = NULL;
    const float* b = NULL;
    const int64_t shape[1] = {2};
    float output[2];
    VxStatus status;
    if (!context || context->closed || !inputs || input_count != 2u ||
        !sink || sink->struct_size != sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
    for (size_t index = 0; index < input_count; index++) {
        const VxTensorBinding* binding = &inputs[index];
        if (binding->struct_size != sizeof(*binding) || !binding->name ||
            binding->dtype != VX_DTYPE_F32 || binding->rank != 1u ||
            binding->shape[0] != 2 || binding->location != VX_MEMORY_HOST ||
            !binding->data || binding->byte_size != sizeof(output))
            return VX_STATUS_INVALID_ARGUMENT;
        if (!strcmp(binding->name, "a") && !a)
            a = (const float*)binding->data;
        else if (!strcmp(binding->name, "b") && !b)
            b = (const float*)binding->data;
        else
            return VX_STATUS_INVALID_ARGUMENT;
    }
    if (!a || !b) return VX_STATUS_INVALID_ARGUMENT;
#ifdef __ANDROID__
    status = example_nnapi_add(a, b, output);
    if (status != VX_STATUS_OK) return status;
#else
    (void)output;
    return VX_STATUS_BACKEND_UNAVAILABLE;
#endif
    status = sink->write(sink->user_data, "sum", VX_DTYPE_F32, shape, 1,
                         output, sizeof(output));
    if (status != VX_STATUS_OK) return status;
    example_report(report, VX_STATUS_OK, VX_STAGE_EXECUTE, "OK",
                   "NNAPI Add executed");
    if (report && report->struct_size == sizeof(*report)) {
        report->route_attested = 1;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=android-nnapi-add;nodes=1;route=nnapi-all");
    }
    return VX_STATUS_OK;
}

#if defined(VOLVOXAI_PUBLIC_API_TESTING)
int volvoxai_example_nnapi_backend_test_oversized_descriptors(void) {
    const char* backend_names[1] = {"android-nnapi-add"};
    ExampleNnapiRuntime runtime = {1};
    ExampleNnapiCompiled compiled = {&runtime};
    VxRuntimeOptions runtime_options = VX_RUNTIME_OPTIONS_INIT;
    VxModelSource source = VX_MODEL_SOURCE_INIT;
    VxBackendCompileInput input = VX_BACKEND_COMPILE_INPUT_INIT;
    VxBackendPolicy policy = VX_BACKEND_POLICY_INIT;
    VxBackendShapeDomainAttestation attestation =
        VX_BACKEND_SHAPE_DOMAIN_ATTESTATION_INIT;
    VxContextOptions context_options = VX_CONTEXT_OPTIONS_INIT;
    VxReport report = VX_REPORT_INIT;
    void* runtime_instance = &runtime;
    void* compiled_instance = NULL;
    void* context_instance = &compiled;

    source.graph_path = "unused";
    input.source = &source;
    input.graph_fingerprint = "graph";
    input.shape_domain_proof_identity = "proof";
    policy.backends = backend_names;
    policy.backend_count = 1u;

    runtime_options.struct_size++;
    if (example_runtime_create(NULL, &runtime_options, &runtime_instance,
                               &report) != VX_STATUS_INVALID_ARGUMENT ||
        runtime_instance != NULL)
        return -1;
    policy.struct_size++;
    if (example_compile(&runtime, &input, &policy, &compiled_instance,
                        &attestation, &report) != VX_STATUS_INVALID_ARGUMENT ||
        compiled_instance != NULL)
        return -1;
    context_options.struct_size++;
    if (example_context_create(&compiled, &context_options, &context_instance,
                               &report) != VX_STATUS_INVALID_ARGUMENT ||
        context_instance != NULL)
        return -1;
    report.struct_size++;
    report.status = VX_STATUS_INTERNAL;
    snprintf(report.reason, sizeof(report.reason), "%s", "UNCHANGED");
    example_report(&report, VX_STATUS_OK, VX_STAGE_EXECUTE, "OK", "changed");
    if (report.status != VX_STATUS_INTERNAL ||
        strcmp(report.reason, "UNCHANGED")) return -1;
    return 0;
}
#endif

static VxStatus example_context_close(void* context_instance,
                                      VxReport* report) {
    ExampleNnapiContext* context = (ExampleNnapiContext*)context_instance;
    (void)report;
    if (!context) return VX_STATUS_INVALID_ARGUMENT;
    context->closed = 1;
    return VX_STATUS_OK;
}

static void example_context_destroy(void* context_instance) {
    free(context_instance);
}

VxStatus volvoxai_example_nnapi_backend_register(VxRuntime* runtime,
                                                 VxReport* report) {
    const VxBackendProvider provider = {
        .struct_size = sizeof(VxBackendProvider),
        .abi_version = VX_BACKEND_ABI_VERSION,
        .name = "android-nnapi-add",
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
