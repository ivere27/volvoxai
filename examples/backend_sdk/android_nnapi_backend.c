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
    "{\"format\":\"volvox-graph/v1\",\"inputs\":{" \
    "\"a\":{\"shape\":[2],\"dtype\":\"float32\"}," \
    "\"b\":{\"shape\":[2],\"dtype\":\"float32\"}}," \
    "\"nodes\":[{\"opType\":\"Add\",\"inputs\":{" \
    "\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"sum\"}," \
    "\"outputs_shape\":{\"out\":[2]}}],\"outputs\":[\"sum\"]}"

typedef struct ExampleNnapiRuntime {
    int ready;
} ExampleNnapiRuntime;

typedef struct ExampleNnapiCompiled {
    ExampleNnapiRuntime* runtime;
} ExampleNnapiCompiled;

typedef struct ExampleNnapiContext {
    ExampleNnapiCompiled* compiled;
    float a[2];
    float b[2];
    int has_a;
    int has_b;
    int closed;
} ExampleNnapiContext;

static void example_report(VxReport* report, VxStatus status, VxStage stage,
                           const char* reason, const char* message) {
    size_t struct_size;
    if (!report || report->struct_size < sizeof(*report)) return;
    struct_size = report->struct_size;
    memset(report, 0, sizeof(*report));
    report->struct_size = struct_size;
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
    ExampleNnapiRuntime* runtime;
    (void)user_data;
    (void)options;
    if (!out_runtime) return VX_STATUS_INVALID_ARGUMENT;
    *out_runtime = NULL;
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
                                const VxModelSource* source,
                                const VxBackendPolicy* policy,
                                void** out_compiled,
                                VxReport* report) {
    ExampleNnapiCompiled* compiled;
    ExampleNnapiRuntime* runtime = (ExampleNnapiRuntime*)runtime_instance;
    if (!runtime || !runtime->ready || !source || !out_compiled ||
        !policy_contains(policy, "android-nnapi-add"))
        return VX_STATUS_INVALID_ARGUMENT;
    *out_compiled = NULL;
    if (!example_graph_matches(source->graph_path)) {
        example_report(report, VX_STATUS_BACKEND_UNSUPPORTED, VX_STAGE_COMPILE,
                       "BACKEND_UNSUPPORTED",
                       "NNAPI example accepts only its documented Add graph");
        return VX_STATUS_BACKEND_UNSUPPORTED;
    }
    compiled = (ExampleNnapiCompiled*)calloc(1, sizeof(*compiled));
    if (!compiled) return VX_STATUS_OUT_OF_MEMORY;
    compiled->runtime = runtime;
    *out_compiled = compiled;
    example_report(report, VX_STATUS_OK, VX_STAGE_COMPILE, "OK",
                   "NNAPI Add graph compiled");
    if (report && report->struct_size >= sizeof(*report)) {
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
    if (!compiled_instance || !options || !out_context ||
        options->decode_row_mode != VX_DECODE_ROW_DISABLED)
        return VX_STATUS_BACKEND_UNSUPPORTED;
    *out_context = NULL;
    context = (ExampleNnapiContext*)calloc(1, sizeof(*context));
    if (!context) return VX_STATUS_OUT_OF_MEMORY;
    context->compiled = (ExampleNnapiCompiled*)compiled_instance;
    *out_context = context;
    return VX_STATUS_OK;
}

static VxStatus example_context_set_input(void* context_instance,
                                          const char* name,
                                          VxDataType dtype,
                                          const void* data,
                                          size_t byte_size,
                                          VxReport* report) {
    ExampleNnapiContext* context = (ExampleNnapiContext*)context_instance;
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
                                        const VxBackendOutputSink* sink,
                                        VxReport* report) {
    ExampleNnapiContext* context = (ExampleNnapiContext*)context_instance;
    const int64_t shape[1] = {2};
    float output[2];
    VxStatus status;
    if (!context || context->closed || !context->has_a || !context->has_b ||
        !sink || sink->struct_size < sizeof(*sink) || !sink->write)
        return VX_STATUS_INVALID_ARGUMENT;
#ifdef __ANDROID__
    status = example_nnapi_add(context->a, context->b, output);
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
    if (report && report->struct_size >= sizeof(*report)) {
        report->route_attested = 1;
        snprintf(report->route_evidence, sizeof(report->route_evidence), "%s",
                 "provider=android-nnapi-add;nodes=1;route=nnapi-all");
    }
    return VX_STATUS_OK;
}

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
