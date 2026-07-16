/*
 * Conservative Android NNAPI backend SDK example.
 *
 * The only VolvoxAI headers used here are public headers.  The backend handles
 * exact-shape F32 Add and declines broadcasting and every other operation so
 * the built-in CPU backend remains the correctness fallback.
 */

#include "android_nnapi_backend.h"

#include <stdint.h>
#include <string.h>

#ifdef __ANDROID__
#include <android/NeuralNetworks.h>
#endif

enum { EXAMPLE_NNAPI_MAX_RANK = 4 };

typedef struct {
    int ready;
} ExampleNnapiBackend;

static ExampleNnapiBackend g_example_nnapi;

static int tensor_is_exact_f32(const VxTensor* tensor) {
    return tensor && vx_tensor_dtype(tensor) == VX_DTYPE_F32 &&
           vx_tensor_element_size(tensor) == sizeof(float) &&
           vx_tensor_numel(tensor) > 0 &&
           (uint64_t)vx_tensor_numel(tensor) <= SIZE_MAX / sizeof(float) &&
           vx_tensor_shape(tensor);
}

int volvoxai_example_nnapi_add_supports(const VxNode* node) {
    const VxTensor* left;
    const VxTensor* right;
    const VxTensor* output;
    const int32_t* left_shape;
    const int32_t* right_shape;
    const int32_t* output_shape;
    int32_t rank;

    if (!node || !vx_node_op(node) || strcmp(vx_node_op(node), "Add") ||
        vx_node_attr_bool(node, "relu", 0) ||
        vx_node_input_count(node) != 2 || vx_node_output_count(node) != 1)
        return VX_DECLINED;
    left = vx_node_input_by_key(node, "a");
    right = vx_node_input_by_key(node, "b");
    output = vx_node_output_by_key(node, "out");
    if (!tensor_is_exact_f32(left) || !tensor_is_exact_f32(right) ||
        !tensor_is_exact_f32(output)) return VX_DECLINED;
    rank = vx_tensor_ndim(left);
    if (rank < 1 || rank > EXAMPLE_NNAPI_MAX_RANK ||
        vx_tensor_ndim(right) != rank || vx_tensor_ndim(output) != rank ||
        vx_tensor_numel(right) != vx_tensor_numel(left) ||
        vx_tensor_numel(output) != vx_tensor_numel(left)) return VX_DECLINED;
    left_shape = vx_tensor_shape(left);
    right_shape = vx_tensor_shape(right);
    output_shape = vx_tensor_shape(output);
    for (int32_t axis = 0; axis < rank; axis++) {
        if (left_shape[axis] <= 0 || right_shape[axis] != left_shape[axis] ||
            output_shape[axis] != left_shape[axis]) return VX_DECLINED;
    }
    return VX_HANDLED;
}

#ifdef __ANDROID__

static int example_init(void* user_data) {
    ExampleNnapiBackend* backend = (ExampleNnapiBackend*)user_data;
    uint32_t device_count = 0;
    int status = ANeuralNetworks_getDeviceCount(&device_count);
    backend->ready = 0;
    if (status != ANEURALNETWORKS_NO_ERROR) return VX_INIT_ERROR;
    if (!device_count) return VX_INIT_UNAVAILABLE;
    backend->ready = 1;
    return VX_INIT_READY;
}

static int example_supports(void* user_data, const VxNode* node) {
    const ExampleNnapiBackend* backend = (const ExampleNnapiBackend*)user_data;
    return backend->ready ? volvoxai_example_nnapi_add_supports(node) : VX_DECLINED;
}

static int example_run(void* user_data, const VxNode* node) {
    ExampleNnapiBackend* backend = (ExampleNnapiBackend*)user_data;
    const VxTensor* left;
    const VxTensor* right;
    VxTensor* output;
    const int32_t* shape;
    uint32_t dimensions[EXAMPLE_NNAPI_MAX_RANK];
    ANeuralNetworksModel* model = NULL;
    ANeuralNetworksCompilation* compilation = NULL;
    ANeuralNetworksExecution* execution = NULL;
    size_t bytes;
    int result = VX_ERROR;

    if (!backend->ready) return VX_ERROR;
    if (volvoxai_example_nnapi_add_supports(node) != VX_HANDLED) return VX_DECLINED;
    left = vx_node_input_by_key(node, "a");
    right = vx_node_input_by_key(node, "b");
    output = vx_node_output_by_key(node, "out");
    if (vx_tensor_sync_host(left) != 0 || vx_tensor_sync_host(right) != 0)
        return VX_ERROR;
    shape = vx_tensor_shape(left);
    for (int32_t axis = 0; axis < vx_tensor_ndim(left); axis++)
        dimensions[axis] = (uint32_t)shape[axis];
    bytes = (size_t)vx_tensor_numel(left) * sizeof(float);
    if (!vx_tensor_cdata(left) || !vx_tensor_cdata(right) || !vx_tensor_data(output))
        return VX_ERROR;

#define EXAMPLE_NNAPI_TRY(call) \
    do { if ((call) != ANEURALNETWORKS_NO_ERROR) goto done; } while (0)

    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_create(&model));
    ANeuralNetworksOperandType tensor_type = {
        .type = ANEURALNETWORKS_TENSOR_FLOAT32,
        .dimensionCount = (uint32_t)vx_tensor_ndim(left),
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
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &tensor_type)); /* a */
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &tensor_type)); /* b */
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &activation_type));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperand(model, &tensor_type)); /* out */
    int32_t activation = ANEURALNETWORKS_FUSED_NONE;
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_setOperandValue(
        model, 2, &activation, sizeof(activation)));
    const uint32_t add_inputs[] = {0, 1, 2};
    const uint32_t add_outputs[] = {3};
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_addOperation(
        model, ANEURALNETWORKS_ADD, 3, add_inputs, 1, add_outputs));
    const uint32_t model_inputs[] = {0, 1};
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_identifyInputsAndOutputs(
        model, 2, model_inputs, 1, add_outputs));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksModel_finish(model));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksCompilation_create(model, &compilation));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksCompilation_finish(compilation));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_create(compilation, &execution));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_setInput(
        execution, 0, NULL, vx_tensor_cdata(left), bytes));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_setInput(
        execution, 1, NULL, vx_tensor_cdata(right), bytes));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_setOutput(
        execution, 0, NULL, vx_tensor_data(output), bytes));
    EXAMPLE_NNAPI_TRY(ANeuralNetworksExecution_compute(execution));
    result = VX_HANDLED;

done:
    ANeuralNetworksExecution_free(execution);
    ANeuralNetworksCompilation_free(compilation);
    ANeuralNetworksModel_free(model);
#undef EXAMPLE_NNAPI_TRY
    return result;
}

#else

static int example_init(void* user_data) {
    ((ExampleNnapiBackend*)user_data)->ready = 0;
    return VX_INIT_UNAVAILABLE;
}

static int example_supports(void* user_data, const VxNode* node) {
    (void)user_data;
    (void)node;
    return VX_DECLINED;
}

static int example_run(void* user_data, const VxNode* node) {
    (void)user_data;
    (void)node;
    return VX_ERROR;
}

#endif

static void example_teardown(void* user_data) {
    ((ExampleNnapiBackend*)user_data)->ready = 0;
}

int volvoxai_example_nnapi_backend_register(void) {
    const VxBackendV1 backend = {
        .struct_size = sizeof(VxBackendV1),
        .abi_version = VX_BACKEND_ABI_V1,
        .name = "android-nnapi-add",
        .user_data = &g_example_nnapi,
        .init = example_init,
        .supports = example_supports,
        .run = example_run,
        .teardown = example_teardown,
    };
    return volvoxai_register_backend(&backend);
}
