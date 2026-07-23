#ifndef VOLVOXAI_TEST_TRAINING_CUDA_MOE_INC
#define VOLVOXAI_TEST_TRAINING_CUDA_MOE_INC

#if VOLVOXAI_ENABLE_CUDA
typedef struct {
    float loss;
    float input[4];
    float router_weight[6];
    float router_bias[3];
    float expert_weight[12];
    float expert_bias[6];
} CudaMoeTrainStepResult;

static int cuda_moe_any_changed(const float* actual, const float* initial,
                                int count) {
    for (int index = 0; index < count; index++) {
        if (fabsf(actual[index] - initial[index]) > 1.0e-7f) return 1;
    }
    return 0;
}

static int cuda_moe_results_close(const float* cpu, const float* cuda,
                                  int count) {
    for (int index = 0; index < count; index++) {
        float tolerance = 3.0e-5f * (1.0f + fabsf(cpu[index]));
        if (!isfinite(cpu[index]) || !isfinite(cuda[index]) ||
            fabsf(cpu[index] - cuda[index]) > tolerance) return 0;
    }
    return 1;
}

static int run_cuda_moe_train_step_case(int use_cuda, int normalize,
                                        int with_bias,
                                        CudaMoeTrainStepResult* result) {
    const int input_shape[2] = {2, 2};
    const int router_weight_shape[2] = {2, 3};
    const int router_bias_shape[1] = {3};
    const int route_shape[2] = {2, 2};
    const int expert_weight_shape[3] = {3, 2, 2};
    const int expert_bias_shape[2] = {3, 2};
    const float input[4] = {1.0f, 0.2f, -0.3f, 0.8f};
    const float router_weight[6] = {
        0.8f, -0.2f, 0.3f,
        0.1f,  0.7f, -0.4f,
    };
    const float router_bias[3] = {0.05f, -0.1f, 0.2f};
    const float expert_weight[12] = {
         0.7f, -0.2f,  0.1f, 0.5f,
        -0.4f,  0.6f,  0.8f, 0.2f,
         0.3f,  0.9f, -0.5f, 0.4f,
    };
    const float expert_bias[6] = {
        0.1f, 0.0f, -0.1f, 0.2f, 0.05f, -0.05f,
    };
    const int targets[2] = {0, 1};
    const char* trainables_with_bias[5] = {
        "moe.input", "moe.router.weight", "moe.router.bias",
        "moe.expert.weight", "moe.expert.bias",
    };
    const char* trainables_without_bias[3] = {
        "moe.input", "moe.router.weight", "moe.expert.weight",
    };
    const char* router_params = normalize
        ? "{\"num_experts\":3,\"top_k\":2,\"normalize\":1,\"temperature\":0.5}"
        : "{\"num_experts\":3,\"top_k\":2,\"normalize\":0,\"temperature\":0.5}";

    CHECK(result && (normalize == 0 || normalize == 1) &&
          (with_bias == 0 || with_bias == 1));
    memset(result, 0, sizeof(*result));
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    g_use_cuda = use_cuda;
    if (use_cuda && cuda_init() != 0) {
        CHECK(finish_graph() == 0);
        return cuda_init_failure_is_unavailable() ? 1 : -1;
    }
    CHECK(volvoxai_engine_require_training_backend(
        use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU) == 0);
    CHECK(add_tensor("moe.input", input_shape, 2, input));
    CHECK(add_tensor("moe.router.weight", router_weight_shape, 2,
                     router_weight));
    if (with_bias) {
        CHECK(add_tensor("moe.router.bias", router_bias_shape, 1,
                         router_bias));
    }
    CHECK(add_tensor("moe.route.indices", route_shape, 2, NULL));
    CHECK(add_tensor("moe.route.weights", route_shape, 2, NULL));
    CHECK(add_tensor("moe.expert.weight", expert_weight_shape, 3,
                     expert_weight));
    if (with_bias) {
        CHECK(add_tensor("moe.expert.bias", expert_bias_shape, 2,
                         expert_bias));
    }
    CHECK(add_tensor("logits", input_shape, 2, NULL));

    Node* node = add_node("MoERouter", "moe.route.indices", router_params);
    CHECK(node && node_input(node, "input", "moe.input") == 0 &&
          node_input(node, "weight", "moe.router.weight") == 0);
    if (with_bias)
        CHECK(node_input(node, "bias", "moe.router.bias") == 0);
    strncpy(node->outs[0].key, "indices", sizeof(node->outs[0].key) - 1);
    strncpy(node->outs[1].key, "weights", sizeof(node->outs[1].key) - 1);
    strncpy(node->outs[1].name, "moe.route.weights",
            sizeof(node->outs[1].name) - 1);
    node->nout = 2;

    /* The discrete indices are routing metadata only.  MoELinear produces a
     * gradient for route.weights, the router's canonical differentiable
     * output, without making the secondary indices output gradient-live. */
    node = add_node("MoELinear", "logits", NULL);
    CHECK(node && node_input(node, "input", "moe.input") == 0 &&
          node_input(node, "expert_weight", "moe.expert.weight") == 0 &&
          node_input(node, "route_indices", "moe.route.indices") == 0 &&
          node_input(node, "route_weights", "moe.route.weights") == 0);
    if (with_bias)
        CHECK(node_input(node, "expert_bias", "moe.expert.bias") == 0);

    const char* const* trainables = with_bias
        ? trainables_with_bias : trainables_without_bias;
    int trainable_count = with_bias ? 5 : 3;
    CHECK(volvoxai_engine_train_step(
        "logits", targets, 2, INT_MIN, trainables, trainable_count,
        VOLVOXAI_TENSOR_UPDATE_SGD, 0.02f, 0.9f, 0.999f, 1.0e-8f,
        0.0f, 0.0f, 17, &result->loss, NULL, NULL) == 0);
    CHECK(volvoxai_engine_last_training_backend() ==
          (use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU));
    CHECK(isfinite(result->loss) && result->loss > 0.0f);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "moe.input", result->input, 4) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "moe.router.weight", result->router_weight, 6) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "moe.expert.weight", result->expert_weight, 12) == 0);
    if (with_bias) {
        CHECK(volvoxai_engine_copy_tensor_f32(
            "moe.router.bias", result->router_bias, 3) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32(
            "moe.expert.bias", result->expert_bias, 6) == 0);
    }
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CPU) == 0);
    return finish_graph();
}

static int test_cuda_moe_runtime_optional(void) {
    static const float initial_input[4] = {1.0f, 0.2f, -0.3f, 0.8f};
    static const float initial_router_weight[6] = {
        0.8f, -0.2f, 0.3f, 0.1f, 0.7f, -0.4f,
    };
    static const float initial_router_bias[3] = {0.05f, -0.1f, 0.2f};
    static const float initial_expert_weight[12] = {
         0.7f, -0.2f,  0.1f, 0.5f,
        -0.4f,  0.6f,  0.8f, 0.2f,
         0.3f,  0.9f, -0.5f, 0.4f,
    };
    static const float initial_expert_bias[6] = {
        0.1f, 0.0f, -0.1f, 0.2f, 0.05f, -0.05f,
    };
    CudaMoeTrainStepResult cpu;
    CudaMoeTrainStepResult cuda;

    /* Biases make this graph emit all four MoELinear and all four MoERouter
     * planner commands.  Requiring CUDA also rules out a CPU fallback. */
    CHECK(run_cuda_moe_train_step_case(0, 1, 1, &cpu) == 0);
    int cuda_result = run_cuda_moe_train_step_case(1, 1, 1, &cuda);
    if (cuda_result == 1) {
        puts("native CUDA MoE TrainStep integration skipped: no CUDA device");
        return 0;
    }
    CHECK(cuda_result == 0);
    CHECK(fabsf(cpu.loss - cuda.loss) < 1.0e-6f);
    CHECK(cuda_moe_results_close(cpu.input, cuda.input, 4));
    CHECK(cuda_moe_results_close(cpu.router_weight, cuda.router_weight, 6));
    CHECK(cuda_moe_results_close(cpu.router_bias, cuda.router_bias, 3));
    CHECK(cuda_moe_results_close(cpu.expert_weight, cuda.expert_weight, 12));
    CHECK(cuda_moe_results_close(cpu.expert_bias, cuda.expert_bias, 6));
    CHECK(cuda_moe_any_changed(cpu.input, initial_input, 4));
    CHECK(cuda_moe_any_changed(cpu.router_weight, initial_router_weight, 6));
    CHECK(cuda_moe_any_changed(cpu.router_bias, initial_router_bias, 3));
    CHECK(cuda_moe_any_changed(cpu.expert_weight, initial_expert_weight, 12));
    CHECK(cuda_moe_any_changed(cpu.expert_bias, initial_expert_bias, 6));

    /* Cover the full-softmax derivative and optional-bias bindings too. */
    CHECK(run_cuda_moe_train_step_case(0, 0, 0, &cpu) == 0);
    CHECK(run_cuda_moe_train_step_case(1, 0, 0, &cuda) == 0);
    CHECK(fabsf(cpu.loss - cuda.loss) < 1.0e-6f);
    CHECK(cuda_moe_results_close(cpu.input, cuda.input, 4));
    CHECK(cuda_moe_results_close(cpu.router_weight, cuda.router_weight, 6));
    CHECK(cuda_moe_results_close(cpu.expert_weight, cuda.expert_weight, 12));
    CHECK(cuda_moe_any_changed(cpu.input, initial_input, 4));
    CHECK(cuda_moe_any_changed(cpu.router_weight, initial_router_weight, 6));
    CHECK(cuda_moe_any_changed(cpu.expert_weight, initial_expert_weight, 12));
    return 0;
}
#endif

#endif
