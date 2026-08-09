#include "engine_core.h"
#include "runtime_state.h"
#include "volvoxai_backend.h"
#include "training/training_core.h"
#if VOLVOXAI_ENABLE_CUDA
#include "cuda_engine.h"
#endif

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do {                                                \
    if (!(expression)) {                                                     \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__,            \
                #expression);                                                \
        return -1;                                                           \
    }                                                                        \
} while (0)

static int near_f32(float actual, float expected) {
    return isfinite(actual) && fabsf(actual - expected) <= 1.0e-5f;
}

static int64_t shape_numel(const int32_t* shape, int32_t ndim) {
    int64_t numel = 1;
    for (int32_t axis = 0; axis < ndim; axis++) numel *= shape[axis];
    return numel;
}

static VolvoxAIAutogradTensor* make_f32(
        VolvoxAIAutogradContext* context, const float* values,
        const int32_t* shape, int32_t ndim, int requires_grad) {
    volvoxai_autograd_tensor_spec_t spec =
        VOLVOXAI_AUTOGRAD_TENSOR_SPEC_INIT;
    int64_t numel = shape_numel(shape, ndim);
    spec.ndim = ndim;
    for (int32_t axis = 0; axis < ndim; axis++) spec.shape[axis] = shape[axis];
    spec.data = values;
    spec.nbytes = (uint64_t)numel * sizeof(float);
    spec.requires_grad = requires_grad;
    return volvoxai_autograd_tensor_create(context, &spec);
}

static VolvoxAIAutogradTensor* make_i32(
        VolvoxAIAutogradContext* context, const int32_t* values,
        const int32_t* shape, int32_t ndim) {
    volvoxai_autograd_tensor_spec_t spec =
        VOLVOXAI_AUTOGRAD_TENSOR_SPEC_INIT;
    int64_t numel = shape_numel(shape, ndim);
    spec.dtype = VOLVOXAI_DTYPE_I32;
    spec.ndim = ndim;
    for (int32_t axis = 0; axis < ndim; axis++) spec.shape[axis] = shape[axis];
    spec.data = values;
    spec.nbytes = (uint64_t)numel * sizeof(int32_t);
    return volvoxai_autograd_tensor_create(context, &spec);
}

static VolvoxAIAutogradTensor* apply_op(
        VolvoxAIAutogradContext* context, const char* op,
        const volvoxai_autograd_input_t* inputs, int32_t input_count,
        const int32_t* shape, int32_t ndim, const char* params_json) {
    volvoxai_autograd_output_spec_t output_spec;
    volvoxai_autograd_op_t descriptor = VOLVOXAI_AUTOGRAD_OP_INIT;
    VolvoxAIAutogradTensor* output = NULL;
    memset(&output_spec, 0, sizeof(output_spec));
    output_spec.key = "output";
    output_spec.dtype = VOLVOXAI_DTYPE_F32;
    output_spec.ndim = ndim;
    for (int32_t axis = 0; axis < ndim; axis++)
        output_spec.shape[axis] = shape[axis];
    descriptor.op = op;
    descriptor.inputs = inputs;
    descriptor.input_count = input_count;
    descriptor.outputs = &output_spec;
    descriptor.output_count = 1;
    descriptor.params_json = params_json;
    if (volvoxai_autograd_apply(context, &descriptor, &output, 1) != 0)
        return NULL;
    return output;
}

static VolvoxAIAutogradTensor* apply_binary(
        VolvoxAIAutogradContext* context, const char* op,
        const VolvoxAIAutogradTensor* a, const VolvoxAIAutogradTensor* b,
        const int32_t* shape, int32_t ndim) {
    const volvoxai_autograd_input_t inputs[2] = {
        {"a", a}, {"b", b},
    };
    volvoxai_autograd_output_spec_t output_spec;
    volvoxai_autograd_op_t descriptor = VOLVOXAI_AUTOGRAD_OP_INIT;
    VolvoxAIAutogradTensor* output = NULL;
    memset(&output_spec, 0, sizeof(output_spec));
    output_spec.key = "output";
    output_spec.dtype = VOLVOXAI_DTYPE_F32;
    output_spec.ndim = ndim;
    for (int32_t axis = 0; axis < ndim; axis++)
        output_spec.shape[axis] = shape[axis];
    descriptor.op = op;
    descriptor.inputs = inputs;
    descriptor.input_count = 2;
    descriptor.outputs = &output_spec;
    descriptor.output_count = 1;
    if (volvoxai_autograd_apply(context, &descriptor, &output, 1) != 0)
        return NULL;
    return output;
}

static VolvoxAIAutogradContext* make_context(int32_t backend) {
    volvoxai_autograd_options_t options = VOLVOXAI_AUTOGRAD_OPTIONS_INIT;
    options.backend = backend;
    return volvoxai_autograd_context_create(&options);
}

static int copy_f32(const VolvoxAIAutogradTensor* tensor, float* output,
                    int64_t numel) {
    return volvoxai_autograd_tensor_copy_data(
        tensor, output, (uint64_t)numel * sizeof(float));
}

static int tensor_flags(const VolvoxAIAutogradTensor* tensor,
                        int32_t* requires_grad, int32_t* is_leaf) {
    int64_t numel = 0;
    int32_t shape[8] = {0};
    int32_t ndim = 0;
    int32_t dtype = -1;
    int32_t ignored_requires_grad = 0;
    int32_t ignored_is_leaf = 0;
    if (!requires_grad) requires_grad = &ignored_requires_grad;
    if (!is_leaf) is_leaf = &ignored_is_leaf;
    return volvoxai_autograd_tensor_info(
        tensor, &numel, shape, &ndim, &dtype, requires_grad, is_leaf);
}

static int test_fan_in_and_nonscalar_seed(void) {
    const int32_t shape[1] = {2};
    const float values[2] = {2.0f, -3.0f};
    const float seed[2] = {0.5f, -2.0f};
    float output_values[2] = {0.0f, 0.0f};
    float gradient[2] = {0.0f, 0.0f};
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, values, shape, 1, 1);
    CHECK(x != NULL);
    VolvoxAIAutogradTensor* square =
        apply_binary(context, "Mul", x, x, shape, 1);
    CHECK(square != NULL);
    VolvoxAIAutogradTensor* y =
        apply_binary(context, "Add", square, x, shape, 1);
    CHECK(y != NULL);
    CHECK(copy_f32(y, output_values, 2) == 0);
    CHECK(near_f32(output_values[0], 6.0f));
    CHECK(near_f32(output_values[1], 6.0f));

    /* A non-scalar output requires an exact, explicit seed.  Rejected seeds
       must not consume or damage the tape. */
    CHECK(volvoxai_autograd_backward(context, y, NULL, 0, 0) != 0);
    CHECK(volvoxai_autograd_backward(context, y, seed, 1, 0) != 0);
    CHECK(volvoxai_autograd_backward(context, y, seed, 2, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, gradient, 2) == 0);
    CHECK(near_f32(gradient[0], 2.5f));
    CHECK(near_f32(gradient[1], 10.0f));
    CHECK(isfinite(gradient[0]) && isfinite(gradient[1]));

    /* retain_graph=0 releases history, but leaves the result value readable. */
    CHECK(copy_f32(y, output_values, 2) == 0);
    CHECK(volvoxai_autograd_backward(context, y, seed, 2, 0) != 0);
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_accumulation_history_and_zero_grad(void) {
    const float value = 3.0f;
    float gradient = 0.0f;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, &value, NULL, 0, 1);
    CHECK(x != NULL);
    VolvoxAIAutogradTensor* y =
        apply_binary(context, "Mul", x, x, NULL, 0);
    CHECK(y != NULL);

    /* NULL plus zero elements is the implicit unit seed for a scalar.  Only
       leaf gradients accumulate; temporary graph gradients reset per call. */
    CHECK(volvoxai_autograd_backward(context, y, NULL, 0, 1) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 6.0f));
    CHECK(volvoxai_autograd_backward(context, y, NULL, 0, 1) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 12.0f));
    CHECK(volvoxai_autograd_backward(context, y, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 18.0f));
    CHECK(volvoxai_autograd_backward(context, y, NULL, 0, 0) != 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 18.0f));

    CHECK(volvoxai_autograd_zero_grad(context, x) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 0.0f));
    CHECK(volvoxai_autograd_zero_grad(context, NULL) == 0);
    CHECK(volvoxai_autograd_clear_graph(context) == 0);
    CHECK(tensor_flags(y, NULL, NULL) != 0);
    CHECK(tensor_flags(x, NULL, NULL) == 0);
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_dynamic_branch_and_clear_graph(void) {
    const float first_value = 3.0f;
    const float second_value = 4.0f;
    float result = 0.0f;
    float gradient = 0.0f;
    int take_add_branch = 0;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x =
        make_f32(context, &first_value, NULL, 0, 1);
    CHECK(x != NULL);
    VolvoxAIAutogradTensor* first = apply_binary(
        context, take_add_branch ? "Add" : "Mul", x, x, NULL, 0);
    CHECK(first != NULL);
    CHECK(volvoxai_autograd_backward(context, first, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 6.0f));

    CHECK(volvoxai_autograd_clear_graph(context) == 0);
    CHECK(copy_f32(first, &result, 1) != 0);
    CHECK(volvoxai_autograd_tensor_set_data(
              x, &second_value, sizeof(second_value)) == 0);
    take_add_branch = 1;
    VolvoxAIAutogradTensor* second = apply_binary(
        context, take_add_branch ? "Add" : "Mul", x, x, NULL, 0);
    CHECK(second != NULL);
    CHECK(copy_f32(second, &result, 1) == 0);
    CHECK(near_f32(result, 8.0f));
    CHECK(volvoxai_autograd_backward(context, second, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 8.0f));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_no_grad_and_detach(void) {
    const float value = 2.0f;
    float x_gradient = -1.0f;
    float detached_gradient = 0.0f;
    int32_t requires_grad = -1;
    int32_t is_leaf = -1;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, &value, NULL, 0, 1);
    CHECK(x != NULL);

    CHECK(volvoxai_autograd_no_grad_begin(context) == 0);
    VolvoxAIAutogradTensor* untracked =
        apply_binary(context, "Mul", x, x, NULL, 0);
    CHECK(untracked != NULL);
    CHECK(volvoxai_autograd_no_grad_end(context) == 0);
    CHECK(tensor_flags(untracked, &requires_grad, &is_leaf) == 0);
    CHECK(requires_grad == 0 && is_leaf == 1);
    CHECK(volvoxai_autograd_backward(
              context, untracked, NULL, 0, 0) != 0);
    /* An unmatched end is rejected without corrupting recording state. */
    CHECK(volvoxai_autograd_no_grad_end(context) != 0);

    VolvoxAIAutogradTensor* square =
        apply_binary(context, "Mul", x, x, NULL, 0);
    CHECK(square != NULL);
    CHECK(volvoxai_autograd_backward(context, square, NULL, 0, 1) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &x_gradient, 1) == 0);
    CHECK(near_f32(x_gradient, 4.0f));
    CHECK(volvoxai_autograd_zero_grad(context, x) == 0);

    VolvoxAIAutogradTensor* detached =
        volvoxai_autograd_tensor_detach(context, square);
    CHECK(detached != NULL);
    CHECK(tensor_flags(detached, &requires_grad, &is_leaf) == 0);
    CHECK(requires_grad == 0 && is_leaf == 1);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(detached, 1) == 0);
    VolvoxAIAutogradTensor* detached_square =
        apply_binary(context, "Mul", detached, detached, NULL, 0);
    CHECK(detached_square != NULL);
    CHECK(volvoxai_autograd_backward(
              context, detached_square, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(
              detached, &detached_gradient, 1) == 0);
    CHECK(near_f32(detached_gradient, 8.0f));
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &x_gradient, 1) == 0);
    CHECK(near_f32(x_gradient, 0.0f));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_unrecorded_operations_do_not_consume_tape(void) {
    const float value = 2.0f;
    float gradient = 0.0f;
    int32_t requires_grad = -1;
    int32_t is_leaf = -1;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, &value, NULL, 0, 1);
    CHECK(x != NULL);

    CHECK(volvoxai_autograd_no_grad_begin(context) == 0);
    /* MAXN is 1024.  More successful unrecorded operations than that must not
       consume tape capacity or retain otherwise releasable result handles. */
    for (int iteration = 0; iteration < 1100; iteration++) {
        VolvoxAIAutogradTensor* result =
            apply_binary(context, "Add", x, x, NULL, 0);
        CHECK(result != NULL);
        CHECK(tensor_flags(result, &requires_grad, &is_leaf) == 0);
        CHECK(requires_grad == 0 && is_leaf == 1);
        CHECK(volvoxai_autograd_tensor_release(result) == 0);
    }
    CHECK(volvoxai_autograd_no_grad_end(context) == 0);

    /* A leaf used only by unrecorded operations remains freely configurable,
       and ordinary recording still works without an intervening clear. */
    CHECK(volvoxai_autograd_tensor_set_requires_grad(x, 0) == 0);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(x, 1) == 0);
    VolvoxAIAutogradTensor* square =
        apply_binary(context, "Mul", x, x, NULL, 0);
    CHECK(square != NULL);
    CHECK(volvoxai_autograd_backward(context, square, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 4.0f));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_mutation_and_error_atomicity(void) {
    const float value = 2.0f;
    const float replacement = 5.0f;
    float copied = 0.0f;
    float gradient = 0.0f;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, &value, NULL, 0, 1);
    CHECK(x != NULL);
    VolvoxAIAutogradTensor* y =
        apply_binary(context, "Mul", x, x, NULL, 0);
    CHECK(y != NULL);

    /* A saved operand cannot change underneath a live backward graph. */
    CHECK(volvoxai_autograd_tensor_set_data(
              x, &replacement, sizeof(replacement)) != 0);
    CHECK(copy_f32(x, &copied, 1) == 0);
    CHECK(near_f32(copied, value));

    const volvoxai_autograd_input_t inputs[2] = {
        {"a", x}, {"b", x},
    };
    volvoxai_autograd_output_spec_t output_spec;
    volvoxai_autograd_op_t bad = VOLVOXAI_AUTOGRAD_OP_INIT;
    VolvoxAIAutogradTensor* escaped = NULL;
    memset(&output_spec, 0, sizeof(output_spec));
    output_spec.key = "output";
    output_spec.dtype = VOLVOXAI_DTYPE_F32;
    bad.op = "OperatorThatDoesNotExist";
    bad.inputs = inputs;
    bad.input_count = 2;
    bad.outputs = &output_spec;
    bad.output_count = 1;

    /* Parameter text must parse as a JSON object.  Both rejection paths keep
       output publication atomic and leave the existing tape usable. */
    bad.op = "Add";
    bad.params_json = "[1]";
    escaped = (VolvoxAIAutogradTensor*)(uintptr_t)1;
    CHECK(volvoxai_autograd_apply(context, &bad, &escaped, 1) != 0);
    CHECK(escaped == NULL);
    bad.params_json = "{";
    escaped = (VolvoxAIAutogradTensor*)(uintptr_t)1;
    CHECK(volvoxai_autograd_apply(context, &bad, &escaped, 1) != 0);
    CHECK(escaped == NULL);

    bad.op = "OperatorThatDoesNotExist";
    bad.params_json = NULL;
    CHECK(volvoxai_autograd_apply(context, &bad, &escaped, 1) != 0);
    CHECK(escaped == NULL);

    /* The rejected mutation and operation leave the earlier tape intact. */
    CHECK(volvoxai_autograd_backward(context, y, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gradient, 1) == 0);
    CHECK(near_f32(gradient, 4.0f));
    CHECK(volvoxai_autograd_clear_graph(context) == 0);
    CHECK(volvoxai_autograd_tensor_set_data(
              x, &replacement, sizeof(replacement)) == 0);
    CHECK(copy_f32(x, &copied, 1) == 0);
    CHECK(near_f32(copied, replacement));

    /* A failed apply on an empty tape does not poison the next operation. */
    escaped = NULL;
    CHECK(volvoxai_autograd_apply(context, &bad, &escaped, 1) != 0);
    CHECK(escaped == NULL);
    VolvoxAIAutogradTensor* valid =
        apply_binary(context, "Add", x, x, NULL, 0);
    CHECK(valid != NULL);
    CHECK(copy_f32(valid, &copied, 1) == 0);
    CHECK(near_f32(copied, 10.0f));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_context_lifecycle_exclusion(void) {
    VolvoxAIEngineOptions engine_options = {
        VOLVOXAI_BACKEND_CPU, 0, 0,
    };
    CHECK(volvoxai_engine_set_execution_row(7) == 0);
    CHECK(make_context(VOLVOXAI_BACKEND_CPU) == NULL);
    CHECK(volvoxai_engine_set_execution_row(-1) == 0);
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    CHECK(make_context(VOLVOXAI_BACKEND_CPU) == NULL);
    CHECK(volvoxai_engine_configure(&engine_options) != 0);
    CHECK(volvoxai_engine_configure_backend("cpu") != 0);
    CHECK(volvoxai_engine_adapter_route_begin("") != 0);
    CHECK(volvoxai_engine_adapter_activate("") != 0);
    CHECK(volvoxai_engine_set_execution_row(0) != 0);

    /* shutdown() has a void ABI; with a live tape it is deliberately a no-op
       rather than invalidating opaque context/backend state behind callers. */
    volvoxai_engine_shutdown();
    CHECK(volvoxai_autograd_context_backend(context) == VOLVOXAI_BACKEND_CPU);
    volvoxai_autograd_context_destroy(context);
    CHECK(volvoxai_engine_configure_backend("cpu") == 0);
    return 0;
}

static int test_linear_and_embedding(void) {
    const int32_t x_shape[2] = {2, 2};
    const int32_t w_shape[2] = {2, 3};
    const int32_t b_shape[1] = {3};
    const int32_t y_shape[2] = {2, 3};
    const float x_values[4] = {1, 2, 3, 4};
    const float w_values[6] = {1, 2, 3, 4, 5, 6};
    const float b_values[3] = {0.5f, -0.5f, 1.0f};
    const float seed[6] = {1, 1, 1, 1, 1, 1};
    float values[6] = {0};
    float gx[4] = {0}, gw[6] = {0}, gb[3] = {0};
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, x_values, x_shape, 2, 1);
    VolvoxAIAutogradTensor* w = make_f32(context, w_values, w_shape, 2, 1);
    VolvoxAIAutogradTensor* b = make_f32(context, b_values, b_shape, 1, 1);
    CHECK(x && w && b);
    const volvoxai_autograd_input_t linear_inputs[3] = {
        {"input", x}, {"weight", w}, {"bias", b},
    };
    VolvoxAIAutogradTensor* y = apply_op(
        context, "Linear", linear_inputs, 3, y_shape, 2,
        "{\"weight_layout\":\"din_dout\"}");
    CHECK(y != NULL);
    CHECK(copy_f32(y, values, 6) == 0);
    const float expected_y[6] = {9.5f, 11.5f, 16, 19.5f, 25.5f, 34};
    for (int i = 0; i < 6; i++) CHECK(near_f32(values[i], expected_y[i]));
    CHECK(volvoxai_autograd_backward(context, y, seed, 6, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, gx, 4) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(w, gw, 6) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(b, gb, 3) == 0);
    const float expected_gx[4] = {6, 15, 6, 15};
    const float expected_gw[6] = {4, 4, 4, 6, 6, 6};
    for (int i = 0; i < 4; i++) CHECK(near_f32(gx[i], expected_gx[i]));
    for (int i = 0; i < 6; i++) CHECK(near_f32(gw[i], expected_gw[i]));
    for (int i = 0; i < 3; i++) CHECK(near_f32(gb[i], 2.0f));
    CHECK(volvoxai_autograd_clear_graph(context) == 0);
    CHECK(volvoxai_autograd_tensor_release(y) == 0);
    volvoxai_autograd_context_destroy(context);

    const int32_t token_shape[1] = {3};
    const int32_t embedding_shape[2] = {3, 2};
    const int32_t output_shape[2] = {3, 2};
    const int32_t tokens_data[3] = {2, 0, 2};
    const float embedding_data[6] = {1, 2, 3, 4, 5, 6};
    const float embedding_seed[6] = {1, 2, 3, 4, 5, 6};
    const float expected_embedding_output[6] = {5, 6, 1, 2, 5, 6};
    const float expected_embedding_grad[6] = {3, 4, 0, 0, 6, 8};
    float embedding_output[6] = {0};
    float embedding_grad[6] = {0};
    context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* tokens =
        make_i32(context, tokens_data, token_shape, 1);
    VolvoxAIAutogradTensor* embedding =
        make_f32(context, embedding_data, embedding_shape, 2, 1);
    VolvoxAIAutogradTensor* embedding_bias =
        make_f32(context, embedding_data, &embedding_shape[1], 1, 1);
    CHECK(tokens && embedding && embedding_bias);
    const volvoxai_autograd_input_t invalid_embedding_inputs[3] = {
        {"input", tokens}, {"weight", embedding}, {"bias", embedding_bias},
    };
    CHECK(apply_op(context, "Embedding", invalid_embedding_inputs, 3,
                   output_shape, 2, NULL) == NULL);
    const volvoxai_autograd_input_t embedding_inputs[2] = {
        {"input", tokens}, {"weight", embedding},
    };
    y = apply_op(context, "Embedding", embedding_inputs, 2,
                 output_shape, 2, NULL);
    CHECK(y != NULL);
    CHECK(copy_f32(y, embedding_output, 6) == 0);
    for (int i = 0; i < 6; i++)
        CHECK(near_f32(embedding_output[i], expected_embedding_output[i]));
    CHECK(volvoxai_autograd_backward(
              context, y, embedding_seed, 6, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(
              embedding, embedding_grad, 6) == 0);
    for (int i = 0; i < 6; i++)
        CHECK(near_f32(embedding_grad[i], expected_embedding_grad[i]));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_transpose_default(void) {
    const int32_t input_shape[2] = {2, 3};
    const int32_t output_shape[2] = {3, 2};
    const float input_data[6] = {1, 2, 3, 4, 5, 6};
    const float seed[6] = {1, 2, 3, 4, 5, 6};
    const float expected_output[6] = {1, 4, 2, 5, 3, 6};
    const float expected_gradient[6] = {1, 3, 5, 2, 4, 6};
    float output_data[6] = {0};
    float gradient[6] = {0};
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* input = make_f32(
        context, input_data, input_shape, 2, 1);
    CHECK(input != NULL);
    const volvoxai_autograd_input_t inputs[1] = {{"input", input}};
    VolvoxAIAutogradTensor* output = apply_op(
        context, "Transpose", inputs, 1, output_shape, 2, NULL);
    CHECK(output != NULL);
    CHECK(copy_f32(output, output_data, 6) == 0);
    for (int i = 0; i < 6; i++)
        CHECK(near_f32(output_data[i], expected_output[i]));
    CHECK(volvoxai_autograd_backward(context, output, seed, 6, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(input, gradient, 6) == 0);
    for (int i = 0; i < 6; i++)
        CHECK(near_f32(gradient[i], expected_gradient[i]));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int run_maxpool_canonical_pads(int backend) {
    const int32_t input_shape[4] = {1, 2, 3, 1};
    const int32_t output_shape[4] = {1, 2, 2, 1};
    const int32_t wrong_output_shape[4] = {1, 2, 1, 1};
    const float input_data[6] = {1, 5, 2, 4, 3, 6};
    const float seed[4] = {1, 2, 3, 4};
    const float expected_output[4] = {5, 2, 5, 6};
    const float expected_gradient[6] = {0, 4, 2, 0, 0, 4};
    static const char* const invalid_pads[] = {
        "{\"kernel\":[2,2],\"stride\":[1,2],\"pads\":[-1,0,0,1]}",
        "{\"kernel\":[2,2],\"stride\":[1,2],\"pads\":[1,-1,0,1]}",
        "{\"kernel\":[2,2],\"stride\":[1,2],\"pads\":[1,0,-1,1]}",
        "{\"kernel\":[2,2],\"stride\":[1,2],\"pads\":[1,0,0,-1]}",
        "{\"kernel\":[2,2],\"stride\":[1,2],\"pads\":[1,0,1]}",
    };
    const char* params =
        "{\"kernel\":[2,2],\"stride\":[1,2],\"pads\":[1,0,0,1],"
        "\"dilation\":[1,1],\"ceil_mode\":false,\"data_layout\":\"NHWC\"}";
    float output_data[4] = {0};
    float gradient[6] = {0};
    VolvoxAIAutogradContext* context = make_context(backend);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* input = make_f32(
        context, input_data, input_shape, 4, 1);
    CHECK(input != NULL);
    const volvoxai_autograd_input_t inputs[1] = {{"input", input}};
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t launches_before = backend == VOLVOXAI_BACKEND_CUDA
        ? cuda_test_launch_count() : 0u;
#endif

    /* Candidate validation uses the shared GPU-plan geometry checker even on
       CPU. Every malformed four-sided pad and a bottom/right-dependent output
       mismatch must roll back without publishing a tensor or leaf gradient. */
    for (size_t index = 0;
         index < sizeof(invalid_pads) / sizeof(invalid_pads[0]); index++) {
        CHECK(apply_op(context, "MaxPool2D", inputs, 1, output_shape, 4,
                       invalid_pads[index]) == NULL);
        CHECK(volvoxai_autograd_tensor_copy_grad(input, gradient, 6) != 0);
#if defined(VOLVOXAI_CUDA_TESTING)
        if (backend == VOLVOXAI_BACKEND_CUDA)
            CHECK(cuda_test_launch_count() == launches_before);
#endif
    }
    CHECK(apply_op(context, "MaxPool2D", inputs, 1, wrong_output_shape, 4,
                   params) == NULL);
    CHECK(volvoxai_autograd_tensor_copy_grad(input, gradient, 6) != 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    if (backend == VOLVOXAI_BACKEND_CUDA)
        CHECK(cuda_test_launch_count() == launches_before);
#endif

    VolvoxAIAutogradTensor* output = apply_op(
        context, "MaxPool2D", inputs, 1, output_shape, 4, params);
    CHECK(output != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t launches_after_forward = backend == VOLVOXAI_BACKEND_CUDA
        ? cuda_test_launch_count() : 0u;
    if (backend == VOLVOXAI_BACKEND_CUDA)
        CHECK(launches_after_forward > launches_before);
#endif
    CHECK(copy_f32(output, output_data, 4) == 0);
    for (int index = 0; index < 4; index++)
        CHECK(near_f32(output_data[index], expected_output[index]));
    CHECK(volvoxai_autograd_backward(context, output, seed, 4, 0) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    if (backend == VOLVOXAI_BACKEND_CUDA)
        CHECK(cuda_test_launch_count() > launches_after_forward);
#endif
    CHECK(volvoxai_autograd_tensor_copy_grad(input, gradient, 6) == 0);
    for (int index = 0; index < 6; index++)
        CHECK(near_f32(gradient[index], expected_gradient[index]));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_maxpool_canonical_pads(void) {
    return run_maxpool_canonical_pads(VOLVOXAI_BACKEND_CPU);
}

static int test_nondifferentiable_input_roles(void) {
    const int32_t image_shape[4] = {1, 1, 1, 2};
    const int32_t channel_shape[1] = {2};
    const float image_data[2] = {1, 2};
    const float channel_data[2] = {1, 1};
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* image = make_f32(
        context, image_data, image_shape, 4, 0);
    VolvoxAIAutogradTensor* weight = make_f32(
        context, channel_data, channel_shape, 1, 0);
    VolvoxAIAutogradTensor* bias = make_f32(
        context, channel_data, channel_shape, 1, 0);
    VolvoxAIAutogradTensor* running_mean = make_f32(
        context, channel_data, channel_shape, 1, 1);
    VolvoxAIAutogradTensor* running_var = make_f32(
        context, channel_data, channel_shape, 1, 0);
    CHECK(image && weight && bias && running_mean && running_var);
    const volvoxai_autograd_input_t batchnorm_inputs[5] = {
        {"input", image}, {"weight", weight}, {"bias", bias},
        {"running_mean", running_mean}, {"running_var", running_var},
    };
    CHECK(apply_op(context, "BatchNorm2D", batchnorm_inputs, 5,
                   image_shape, 4, NULL) == NULL);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(running_mean, 0) == 0);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(running_var, 1) == 0);
    CHECK(apply_op(context, "BatchNorm2D", batchnorm_inputs, 5,
                   image_shape, 4, NULL) == NULL);

    const int32_t moe_input_shape[2] = {1, 2};
    const int32_t expert_shape[3] = {2, 2, 1};
    const int32_t route_shape[2] = {1, 1};
    const int32_t moe_output_shape[2] = {1, 1};
    const float expert_data[4] = {1, 2, 4, 5};
    const float route_index = 1;
    const float route_weight = 0.5f;
    VolvoxAIAutogradTensor* moe_input = make_f32(
        context, image_data, moe_input_shape, 2, 0);
    VolvoxAIAutogradTensor* expert_weight = make_f32(
        context, expert_data, expert_shape, 3, 0);
    VolvoxAIAutogradTensor* route_indices = make_f32(
        context, &route_index, route_shape, 2, 1);
    VolvoxAIAutogradTensor* route_weights = make_f32(
        context, &route_weight, route_shape, 2, 0);
    CHECK(moe_input && expert_weight && route_indices && route_weights);
    const volvoxai_autograd_input_t moe_inputs[4] = {
        {"input", moe_input}, {"expert_weight", expert_weight},
        {"route_indices", route_indices}, {"route_weights", route_weights},
    };
    CHECK(apply_op(context, "MoELinear", moe_inputs, 4,
                   moe_output_shape, 2, NULL) == NULL);

    const int32_t qkv_shape[2] = {1, 3};
    const int32_t attention_shape[2] = {1, 1};
    const float qkv_data[3] = {1, 2, 3};
    const float attention_data = 1;
    VolvoxAIAutogradTensor* qkv = make_f32(
        context, qkv_data, qkv_shape, 2, 0);
    VolvoxAIAutogradTensor* mask = make_f32(
        context, &attention_data, attention_shape, 2, 1);
    CHECK(qkv && mask);
    const volvoxai_autograd_input_t sdpa_inputs[2] = {
        {"qkv", qkv}, {"mask", mask},
    };
    CHECK(apply_op(context, "SDPA", sdpa_inputs, 2,
                   attention_shape, 2, "{\"heads\":1}") == NULL);

    VolvoxAIAutogradTensor* q = make_f32(
        context, &attention_data, attention_shape, 2, 0);
    VolvoxAIAutogradTensor* k = make_f32(
        context, &attention_data, attention_shape, 2, 0);
    VolvoxAIAutogradTensor* v = make_f32(
        context, &attention_data, attention_shape, 2, 0);
    CHECK(q && k && v);
    const volvoxai_autograd_input_t cross_inputs[4] = {
        {"q", q}, {"k", k}, {"v", v}, {"mask", mask},
    };
    CHECK(apply_op(context, "CrossSDPA", cross_inputs, 4,
                   attention_shape, 2, "{\"heads\":1}") == NULL);
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_moe_linear_i32_cpu(void) {
    const int32_t input_shape[2] = {1, 2};
    const int32_t expert_shape[3] = {2, 2, 1};
    const int32_t route_shape[2] = {1, 1};
    const int32_t output_shape[2] = {1, 1};
    const float input_data[2] = {2, 3};
    const float expert_data[4] = {1, 2, 4, 5};
    const int32_t route_index = 1;
    const float route_weight = 0.5f;
    float output_data = 0;
    float input_gradient[2] = {0};
    float expert_gradient[4] = {0};
    float route_gradient = 0;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* input = make_f32(
        context, input_data, input_shape, 2, 1);
    VolvoxAIAutogradTensor* expert_weight = make_f32(
        context, expert_data, expert_shape, 3, 1);
    VolvoxAIAutogradTensor* route_indices = make_i32(
        context, &route_index, route_shape, 2);
    VolvoxAIAutogradTensor* route_weights = make_f32(
        context, &route_weight, route_shape, 2, 1);
    CHECK(input && expert_weight && route_indices && route_weights);
    const volvoxai_autograd_input_t inputs[4] = {
        {"input", input}, {"expert_weight", expert_weight},
        {"route_indices", route_indices}, {"route_weights", route_weights},
    };
    VolvoxAIAutogradTensor* output = apply_op(
        context, "MoELinear", inputs, 4, output_shape, 2, NULL);
    CHECK(output != NULL);
    CHECK(copy_f32(output, &output_data, 1) == 0);
    CHECK(near_f32(output_data, 11.5f));
    CHECK(volvoxai_autograd_backward(context, output, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(input, input_gradient, 2) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(
              expert_weight, expert_gradient, 4) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(
              route_weights, &route_gradient, 1) == 0);
    CHECK(near_f32(input_gradient[0], 2.0f));
    CHECK(near_f32(input_gradient[1], 2.5f));
    CHECK(near_f32(expert_gradient[0], 0.0f));
    CHECK(near_f32(expert_gradient[1], 0.0f));
    CHECK(near_f32(expert_gradient[2], 1.0f));
    CHECK(near_f32(expert_gradient[3], 1.5f));
    CHECK(near_f32(route_gradient, 23.0f));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_schema_shape_preflight(void) {
    const int32_t shape[1] = {2};
    const int32_t short_shape[1] = {1};
    const int32_t long_shape[1] = {3};
    const float x_values[2] = {2, 3};
    const float weight_values[2] = {1, 1};
    float gradient[2] = {0};
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, x_values, shape, 1, 1);
    VolvoxAIAutogradTensor* weight =
        make_f32(context, weight_values, shape, 1, 1);
    CHECK(x && weight);
    VolvoxAIAutogradTensor* valid =
        apply_binary(context, "Mul", x, x, shape, 1);
    CHECK(valid != NULL);

    const volvoxai_autograd_input_t identity_input[1] = {{"input", x}};
    CHECK(apply_op(context, "Identity", identity_input, 1,
                   short_shape, 1, NULL) == NULL);
    CHECK(apply_op(context, "Identity", identity_input, 1,
                   long_shape, 1, NULL) == NULL);
    const volvoxai_autograd_input_t wrong_norm_inputs[2] = {
        {"x", x}, {"weight", weight},
    };
    CHECK(apply_op(context, "LayerNorm", wrong_norm_inputs, 2,
                   shape, 1, NULL) == NULL);
    CHECK(apply_op(context, "LayerNorm", identity_input, 1,
                   shape, 1, NULL) == NULL);
    const int32_t concat_shape[1] = {4};
    const volvoxai_autograd_input_t concat_inputs[2] = {
        {"left", x}, {"right", weight},
    };
    CHECK(apply_op(context, "Concat", concat_inputs, 2,
                   concat_shape, 1,
                   "{\"axis\":0,\"sigmoid\":1}") == NULL);

    /* Every rejected descriptor is preflighted before forward and leaves the
       pre-existing tape usable. */
    CHECK(volvoxai_autograd_backward(context, valid, NULL, 0, 0) != 0);
    const float seed[2] = {1, 1};
    CHECK(volvoxai_autograd_backward(context, valid, seed, 2, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, gradient, 2) == 0);
    CHECK(near_f32(gradient[0], 4) && near_f32(gradient[1], 6));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_disjoint_history_and_release(void) {
    const float x_value = 2.0f;
    const float z_value = 3.0f;
    float gx = 0, gz = 0;
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CPU);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, &x_value, NULL, 0, 1);
    VolvoxAIAutogradTensor* z = make_f32(context, &z_value, NULL, 0, 1);
    CHECK(x && z);
    VolvoxAIAutogradTensor* x2 = apply_binary(context, "Mul", x, x, NULL, 0);
    VolvoxAIAutogradTensor* z2 = apply_binary(context, "Mul", z, z, NULL, 0);
    CHECK(x2 && z2);
    CHECK(volvoxai_autograd_backward(context, x2, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_backward(context, x2, NULL, 0, 0) != 0);
    CHECK(volvoxai_autograd_backward(context, z2, NULL, 0, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(x, &gx, 1) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(z, &gz, 1) == 0);
    CHECK(near_f32(gx, 4) && near_f32(gz, 6));
    CHECK(volvoxai_autograd_tensor_set_requires_grad(x, 1) == 0);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(x, 0) != 0);
    CHECK(volvoxai_autograd_tensor_release(x2) != 0);
    CHECK(volvoxai_autograd_clear_graph(context) == 0);
    CHECK(volvoxai_autograd_tensor_release(x2) == 0);
    CHECK(volvoxai_autograd_tensor_release(z2) == 0);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(x, 0) == 0);
    CHECK(volvoxai_autograd_tensor_set_requires_grad(x, 1) == 0);

    /* A long-running loop can explicitly retire every invalidated result;
       unreleased stale handles are never recycled and therefore cannot ABA. */
    for (int iteration = 0; iteration < 128; iteration++) {
        VolvoxAIAutogradTensor* value =
            apply_binary(context, "Add", x, x, NULL, 0);
        CHECK(value != NULL);
        CHECK(volvoxai_autograd_tensor_release(value) != 0);
        CHECK(volvoxai_autograd_clear_graph(context) == 0);
        CHECK(volvoxai_autograd_tensor_release(value) == 0);
    }
    CHECK(volvoxai_autograd_tensor_release(x) == 0);
    CHECK(volvoxai_autograd_tensor_release(z) == 0);
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_moe_router_output_contract(void) {
    const int32_t input_shape[2] = {1, 2};
    const int32_t weight_shape[2] = {2, 2};
    const int32_t route_shape[2] = {1, 1};
    const float input_data[2] = {1, 2};
    const float weight_data[4] = {1, 0, 0, 1};
    const int indices_dtypes[2] = {
        VOLVOXAI_DTYPE_F32, VOLVOXAI_DTYPE_I32,
    };
    for (int dtype_case = 0; dtype_case < 2; dtype_case++) {
        int indices_dtype = indices_dtypes[dtype_case];
        VolvoxAIAutogradContext* context =
            make_context(VOLVOXAI_BACKEND_CPU);
        CHECK(context != NULL);
        VolvoxAIAutogradTensor* input =
            make_f32(context, input_data, input_shape, 2, 1);
        VolvoxAIAutogradTensor* weight =
            make_f32(context, weight_data, weight_shape, 2, 1);
        CHECK(input && weight);
        const volvoxai_autograd_input_t inputs[2] = {
            {"input", input}, {"weight", weight},
        };
        volvoxai_autograd_output_spec_t output_specs[2];
        memset(output_specs, 0, sizeof(output_specs));
        output_specs[0].key = "indices";
        output_specs[0].dtype = indices_dtype;
        output_specs[0].ndim = 2;
        memcpy(output_specs[0].shape, route_shape, sizeof(route_shape));
        output_specs[1].key = "weights";
        output_specs[1].dtype = VOLVOXAI_DTYPE_F32;
        output_specs[1].ndim = 2;
        memcpy(output_specs[1].shape, route_shape, sizeof(route_shape));
        volvoxai_autograd_op_t descriptor = VOLVOXAI_AUTOGRAD_OP_INIT;
        descriptor.op = "MoERouter";
        descriptor.inputs = inputs;
        descriptor.input_count = 2;
        descriptor.outputs = output_specs;
        descriptor.output_count = 2;
        descriptor.params_json =
            "{\"num_experts\":2,\"top_k\":1,\"normalize\":1}";
        VolvoxAIAutogradTensor* outputs[2] = {NULL, NULL};
        CHECK(volvoxai_autograd_apply(context, &descriptor, outputs, 2) == 0);
        CHECK(outputs[0] && outputs[1]);
        int32_t requires_grad = -1;
        CHECK(tensor_flags(outputs[0], &requires_grad, NULL) == 0);
        CHECK(requires_grad == 0);
        CHECK(tensor_flags(outputs[1], &requires_grad, NULL) == 0);
        CHECK(requires_grad == 1);
        if (indices_dtype == VOLVOXAI_DTYPE_I32) {
            int32_t index = -1;
            CHECK(volvoxai_autograd_tensor_copy_data(
                      outputs[0], &index, sizeof(index)) == 0);
            CHECK(index == 1);
        } else {
            float index = -1;
            CHECK(copy_f32(outputs[0], &index, 1) == 0);
            CHECK(near_f32(index, 1));
        }
        float route_weight = 0;
        CHECK(copy_f32(outputs[1], &route_weight, 1) == 0);
        CHECK(near_f32(route_weight, 1));
        CHECK(volvoxai_autograd_backward(
                  context, outputs[1], NULL, 0, 0) == 0);
        CHECK(volvoxai_autograd_clear_graph(context) == 0);
        CHECK(volvoxai_autograd_tensor_release(outputs[0]) == 0);
        CHECK(volvoxai_autograd_tensor_release(outputs[1]) == 0);
        volvoxai_autograd_context_destroy(context);
    }
    return 0;
}

#if VOLVOXAI_ENABLE_CUDA
static int test_cuda_eager_weight_slot_lifetime(void) {
    const int32_t x_shape[2] = {1, 2};
    const int32_t weight_shape[2] = {2, 2};
    const int32_t bias_shape[1] = {2};
    const int32_t y_shape[2] = {1, 2};
    const float x_data[2] = {1, 2};
    const float w0_data[4] = {1, 0, 0, 1};
    const float w1_data[4] = {0, 2, 3, 0};
    const float bias_data[2] = {0.5f, -0.5f};
    const float expected[2] = {7.5f, 3.5f};
    float copied[2] = {0};

    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CUDA);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* x = make_f32(context, x_data, x_shape, 2, 1);
    VolvoxAIAutogradTensor* w0 =
        make_f32(context, w0_data, weight_shape, 2, 1);
    VolvoxAIAutogradTensor* w1 =
        make_f32(context, w1_data, weight_shape, 2, 1);
    VolvoxAIAutogradTensor* bias =
        make_f32(context, bias_data, bias_shape, 1, 1);
    CHECK(x && w0 && w1 && bias);
    VolvoxAIAutogradTensor* weight =
        apply_binary(context, "Add", w0, w1, weight_shape, 2);
    CHECK(weight != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t resident_weights_before_linear =
        cuda_test_graph_resident_weight_slot_count();
#endif
    const volvoxai_autograd_input_t linear_inputs[3] = {
        {"input", x}, {"weight", weight}, {"bias", bias},
    };
    VolvoxAIAutogradTensor* y = apply_op(
        context, "Linear", linear_inputs, 3, y_shape, 2,
        "{\"weight_layout\":\"din_dout\"}");
    CHECK(y != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_graph_slot_count() > 0u);
    /* Linear classifies the computed Add output and bias as exactly two
       resident semantic-weight slots. */
    CHECK(cuda_test_graph_resident_weight_slot_count() ==
          resident_weights_before_linear + 2u);
#endif
    CHECK(volvoxai_autograd_clear_graph(context) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    /* The non-leaf weight was a persistent semantic-weight slot, but its host
       allocation is now invalid. Dynamic graph clearing must remove it. */
    CHECK(cuda_test_graph_slot_count() == 0u);
    CHECK(cuda_test_graph_resident_weight_slot_count() == 0u);
#endif
    CHECK(volvoxai_autograd_tensor_release(y) == 0);
    CHECK(volvoxai_autograd_tensor_release(weight) == 0);

    /* Recreate the same shapes immediately so allocator address reuse cannot
       discover a stale containing slot from the preceding eager graph. */
    weight = apply_binary(context, "Add", w0, w1, weight_shape, 2);
    CHECK(weight != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    resident_weights_before_linear =
        cuda_test_graph_resident_weight_slot_count();
#endif
    const volvoxai_autograd_input_t reused_inputs[3] = {
        {"input", x}, {"weight", weight}, {"bias", bias},
    };
    y = apply_op(context, "Linear", reused_inputs, 3, y_shape, 2,
                 "{\"weight_layout\":\"din_dout\"}");
    CHECK(y != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_graph_resident_weight_slot_count() ==
          resident_weights_before_linear + 2u);
#endif
    CHECK(copy_f32(y, copied, 2) == 0);
    CHECK(near_f32(copied[0], expected[0]));
    CHECK(near_f32(copied[1], expected[1]));
    volvoxai_autograd_context_destroy(context);

    /* An unrecorded operation retires its node immediately. Releasing one of
       its now-unreferenced live inputs must also drop every retained weight
       identity while preserving the detached output value. */
    context = make_context(VOLVOXAI_BACKEND_CUDA);
    CHECK(context != NULL);
    x = make_f32(context, x_data, x_shape, 2, 0);
    weight = make_f32(context, w0_data, weight_shape, 2, 0);
    bias = make_f32(context, bias_data, bias_shape, 1, 0);
    CHECK(x && weight && bias);
    const volvoxai_autograd_input_t unrecorded_inputs[3] = {
        {"input", x}, {"weight", weight}, {"bias", bias},
    };
#if defined(VOLVOXAI_CUDA_TESTING)
    resident_weights_before_linear =
        cuda_test_graph_resident_weight_slot_count();
#endif
    y = apply_op(context, "Linear", unrecorded_inputs, 3, y_shape, 2,
                 "{\"weight_layout\":\"din_dout\"}");
    CHECK(y != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_graph_slot_count() > 0u);
    CHECK(cuda_test_graph_resident_weight_slot_count() ==
          resident_weights_before_linear + 2u);
#endif
    CHECK(volvoxai_autograd_tensor_release(weight) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_graph_slot_count() == 0u);
    CHECK(cuda_test_graph_resident_weight_slot_count() == 0u);
#endif
    CHECK(copy_f32(y, copied, 2) == 0);
    CHECK(near_f32(copied[0], 1.5f));
    CHECK(near_f32(copied[1], 1.5f));
    volvoxai_autograd_context_destroy(context);
    return 0;
}

static int test_cuda_parity_optional(void) {
    const int32_t shape[1] = {2};
    const float values[2] = {2.0f, -3.0f};
    const float seed[2] = {0.5f, -2.0f};
    float output_values[2] = {0.0f, 0.0f};
    float gradient[2] = {0.0f, 0.0f};
    if (volvoxai_engine_configure_backend("cuda") != 0) {
        CHECK(cuda_init_failure_is_unavailable());
        puts("CUDA device unavailable; skipping dynamic-autograd CUDA parity");
        return 0;
    }
    CHECK(run_maxpool_canonical_pads(VOLVOXAI_BACKEND_CUDA) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_caller_context_is_clear());
#endif
    CHECK(test_cuda_eager_weight_slot_lifetime() == 0);
    VolvoxAIAutogradContext* context = make_context(VOLVOXAI_BACKEND_CUDA);
    CHECK(context != NULL);
    CHECK(volvoxai_autograd_context_backend(context) == VOLVOXAI_BACKEND_CUDA);
    VolvoxAIAutogradTensor* x = make_f32(context, values, shape, 1, 1);
    CHECK(x != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t launches_before = cuda_test_launch_count();
#endif
    VolvoxAIAutogradTensor* square =
        apply_binary(context, "Mul", x, x, shape, 1);
    CHECK(square != NULL);
    VolvoxAIAutogradTensor* y =
        apply_binary(context, "Add", square, x, shape, 1);
    CHECK(y != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t launches_after_forward = cuda_test_launch_count();
    CHECK(launches_after_forward > launches_before);
#endif
    CHECK(copy_f32(y, output_values, 2) == 0);
    CHECK(near_f32(output_values[0], 6.0f));
    CHECK(near_f32(output_values[1], 6.0f));
    CHECK(volvoxai_autograd_backward(context, y, seed, 2, 0) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    /* A concrete CUDA context is strict.  Launches in both eager forward and
       backward prove this case was not serviced by the CPU fallback. */
    CHECK(cuda_test_launch_count() > launches_after_forward);
#endif
    CHECK(volvoxai_autograd_tensor_copy_grad(x, gradient, 2) == 0);
    CHECK(near_f32(gradient[0], 2.5f));
    CHECK(near_f32(gradient[1], 10.0f));
    volvoxai_autograd_context_destroy(context);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_caller_context_is_clear());
#endif

    /* Omitted Transpose perm is reverse-axes on every native route.  Rejected
       fused-Concat and I32 CUDA MoE descriptors must not launch or damage the
       already-recorded tape. */
    const int32_t transpose_input_shape[2] = {2, 3};
    const int32_t transpose_output_shape[2] = {3, 2};
    const float transpose_input_data[6] = {1, 2, 3, 4, 5, 6};
    const float transpose_seed[6] = {1, 2, 3, 4, 5, 6};
    const float expected_transpose_output[6] = {1, 4, 2, 5, 3, 6};
    const float expected_transpose_gradient[6] = {1, 3, 5, 2, 4, 6};
    float transpose_output_data[6] = {0};
    float transpose_gradient[6] = {0};
    context = make_context(VOLVOXAI_BACKEND_CUDA);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* transpose_input = make_f32(
        context, transpose_input_data, transpose_input_shape, 2, 1);
    CHECK(transpose_input != NULL);
    const volvoxai_autograd_input_t transpose_inputs[1] = {
        {"input", transpose_input},
    };
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t transpose_launches_before = cuda_test_launch_count();
#endif
    VolvoxAIAutogradTensor* transpose_output = apply_op(
        context, "Transpose", transpose_inputs, 1,
        transpose_output_shape, 2, NULL);
    CHECK(transpose_output != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_launch_count() > transpose_launches_before);
#endif
    CHECK(copy_f32(transpose_output, transpose_output_data, 6) == 0);
    for (int i = 0; i < 6; i++)
        CHECK(near_f32(transpose_output_data[i], expected_transpose_output[i]));

    const int32_t concat_output_shape[2] = {4, 3};
    const volvoxai_autograd_input_t concat_inputs[2] = {
        {"left", transpose_input}, {"right", transpose_input},
    };
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t rejected_launches_before = cuda_test_launch_count();
#endif
    CHECK(apply_op(context, "Concat", concat_inputs, 2,
                   concat_output_shape, 2,
                   "{\"axis\":0,\"sigmoid\":1}") == NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_launch_count() == rejected_launches_before);
#endif

    const int32_t moe_input_shape[2] = {1, 2};
    const int32_t expert_shape[3] = {2, 2, 1};
    const int32_t route_shape[2] = {1, 1};
    const int32_t moe_output_shape[2] = {1, 1};
    const float moe_input_data[2] = {2, 3};
    const float expert_data[4] = {1, 2, 4, 5};
    const int32_t route_index = 1;
    const float route_weight = 0.5f;
    VolvoxAIAutogradTensor* moe_input = make_f32(
        context, moe_input_data, moe_input_shape, 2, 1);
    VolvoxAIAutogradTensor* expert_weight = make_f32(
        context, expert_data, expert_shape, 3, 1);
    VolvoxAIAutogradTensor* route_indices = make_i32(
        context, &route_index, route_shape, 2);
    VolvoxAIAutogradTensor* route_weights = make_f32(
        context, &route_weight, route_shape, 2, 1);
    CHECK(moe_input && expert_weight && route_indices && route_weights);
    const volvoxai_autograd_input_t moe_inputs[4] = {
        {"input", moe_input}, {"expert_weight", expert_weight},
        {"route_indices", route_indices}, {"route_weights", route_weights},
    };
#if defined(VOLVOXAI_CUDA_TESTING)
    rejected_launches_before = cuda_test_launch_count();
#endif
    CHECK(apply_op(context, "MoELinear", moe_inputs, 4,
                   moe_output_shape, 2, NULL) == NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_launch_count() == rejected_launches_before);
#endif
    CHECK(volvoxai_autograd_backward(
              context, transpose_output, transpose_seed, 6, 0) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(
              transpose_input, transpose_gradient, 6) == 0);
    for (int i = 0; i < 6; i++)
        CHECK(near_f32(transpose_gradient[i], expected_transpose_gradient[i]));
    volvoxai_autograd_context_destroy(context);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_caller_context_is_clear());
#endif

    /* Exercise keyed multi-input dispatch and parameter gradients, not only
       elementwise broadcast kernels. */
    const int32_t linear_x_shape[2] = {1, 2};
    const int32_t linear_w_shape[2] = {2, 2};
    const int32_t linear_b_shape[1] = {2};
    const int32_t linear_y_shape[2] = {1, 2};
    const float linear_x_data[2] = {1, 2};
    const float linear_w_data[4] = {1, 2, 3, 4};
    const float linear_b_data[2] = {0.5f, -0.5f};
    const float linear_seed[2] = {2, -1};
    const float expected_linear_y[2] = {7.5f, 9.5f};
    const float expected_linear_gx[2] = {0, 2};
    const float expected_linear_gw[4] = {2, -1, 4, -2};
    float linear_y_data[2] = {0}, linear_gx[2] = {0};
    float linear_gw[4] = {0}, linear_gb[2] = {0};
    context = make_context(VOLVOXAI_BACKEND_CUDA);
    CHECK(context != NULL);
    VolvoxAIAutogradTensor* linear_x = make_f32(
        context, linear_x_data, linear_x_shape, 2, 1);
    VolvoxAIAutogradTensor* linear_w = make_f32(
        context, linear_w_data, linear_w_shape, 2, 1);
    VolvoxAIAutogradTensor* linear_b = make_f32(
        context, linear_b_data, linear_b_shape, 1, 1);
    CHECK(linear_x && linear_w && linear_b);
    const volvoxai_autograd_input_t linear_inputs[3] = {
        {"input", linear_x}, {"weight", linear_w}, {"bias", linear_b},
    };
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t linear_launches_before = cuda_test_launch_count();
#endif
    VolvoxAIAutogradTensor* linear_y = apply_op(
        context, "Linear", linear_inputs, 3, linear_y_shape, 2,
        "{\"weight_layout\":\"din_dout\"}");
    CHECK(linear_y != NULL);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t linear_launches_after_forward = cuda_test_launch_count();
    CHECK(linear_launches_after_forward > linear_launches_before);
#endif
    CHECK(copy_f32(linear_y, linear_y_data, 2) == 0);
    CHECK(near_f32(linear_y_data[0], expected_linear_y[0]));
    CHECK(near_f32(linear_y_data[1], expected_linear_y[1]));
    CHECK(volvoxai_autograd_backward(
              context, linear_y, linear_seed, 2, 0) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_launch_count() > linear_launches_after_forward);
#endif
    CHECK(volvoxai_autograd_tensor_copy_grad(linear_x, linear_gx, 2) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(linear_w, linear_gw, 4) == 0);
    CHECK(volvoxai_autograd_tensor_copy_grad(linear_b, linear_gb, 2) == 0);
    for (int i = 0; i < 2; i++) {
        CHECK(near_f32(linear_gx[i], expected_linear_gx[i]));
        CHECK(near_f32(linear_gb[i], linear_seed[i]));
    }
    for (int i = 0; i < 4; i++)
        CHECK(near_f32(linear_gw[i], expected_linear_gw[i]));
    volvoxai_autograd_context_destroy(context);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_caller_context_is_clear());
#endif
    CHECK(volvoxai_engine_configure_backend("cpu") == 0);
    return 0;
}
#endif

int main(void) {
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    CHECK(volvoxai_autograd_abi_version() == VOLVOXAI_AUTOGRAD_ABI_VERSION);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_configure_backend("cpu") == 0);
    CHECK(test_fan_in_and_nonscalar_seed() == 0);
    CHECK(test_accumulation_history_and_zero_grad() == 0);
    CHECK(test_dynamic_branch_and_clear_graph() == 0);
    CHECK(test_no_grad_and_detach() == 0);
    CHECK(test_unrecorded_operations_do_not_consume_tape() == 0);
    CHECK(test_mutation_and_error_atomicity() == 0);
    CHECK(test_context_lifecycle_exclusion() == 0);
    CHECK(test_linear_and_embedding() == 0);
    CHECK(test_transpose_default() == 0);
    CHECK(test_maxpool_canonical_pads() == 0);
    CHECK(test_nondifferentiable_input_roles() == 0);
    CHECK(test_moe_linear_i32_cpu() == 0);
    CHECK(test_schema_shape_preflight() == 0);
    CHECK(test_disjoint_history_and_release() == 0);
    CHECK(test_moe_router_output_contract() == 0);
#if VOLVOXAI_ENABLE_CUDA
    CHECK(test_cuda_parity_optional() == 0);
#endif
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("native dynamic autograd tests passed");
    return 0;
}
