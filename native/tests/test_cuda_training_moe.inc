#ifndef VOLVOXAI_TEST_CUDA_TRAINING_MOE_INC
#define VOLVOXAI_TEST_CUDA_TRAINING_MOE_INC

static int test_moe_linear_backward(void) {
    enum { ROWS = 2, D_IN = 2, D_OUT = 2, EXPERTS = 2, TOP_K = 2 };
    const float input[ROWS * D_IN] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float expert_weight[EXPERTS * D_IN * D_OUT] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 1.0f, 2.0f, -2.0f,
    };
    const float expert_bias[EXPERTS * D_OUT] = {
        0.5f, -0.5f, 1.0f, 2.0f,
    };
    const float route_indices[ROWS * TOP_K] = {0.0f, 1.0f, 1.0f, 0.0f};
    const float route_weights[ROWS * TOP_K] = {0.75f, 0.25f, 0.4f, 0.6f};
    const float grad_output[ROWS * D_OUT] = {1.0f, -2.0f, 0.5f, 1.0f};
    const float input_delta[ROWS * D_IN] = {-3.0f, -2.25f, 1.7f, 2.9f};
    const float weight_delta[EXPERTS * D_IN * D_OUT] = {
        0.45f, -2.1f, 2.4f, -1.2f,
        0.05f, -0.9f, 1.1f, 0.2f,
    };
    const float bias_delta[EXPERTS * D_OUT] = {
        1.05f, -0.9f, 0.45f, -0.1f,
    };
    const float route_delta[ROWS * TOP_K] = {-11.5f, 6.0f, -1.0f, 13.75f};
    float grad_input[ROWS * D_IN];
    float grad_expert_weight[EXPERTS * D_IN * D_OUT];
    float grad_expert_bias[EXPERTS * D_OUT];
    float grad_route_weights[ROWS * TOP_K];
    uint32_t params[8] = {
        ROWS, D_IN, D_OUT, EXPERTS, TOP_K, 1u, 0u, 0u,
    };
    void* hosts[11] = {
        (void*)input, (void*)expert_weight, (void*)expert_bias,
        (void*)route_indices, (void*)route_weights, (void*)grad_output,
        grad_input, grad_expert_weight, grad_expert_bias,
        grad_route_weights, params,
    };
    size_t bytes[11] = {
        sizeof(input), sizeof(expert_weight), sizeof(expert_bias),
        sizeof(route_indices), sizeof(route_weights), sizeof(grad_output),
        sizeof(grad_input), sizeof(grad_expert_weight),
        sizeof(grad_expert_bias), sizeof(grad_route_weights), sizeof(params),
    };
    const unsigned char access[11] = {
        1u, 1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[11] = {
        0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };

    cuda_graph_reset();
    for (uint32_t index = 0u; index < ROWS * D_IN; ++index)
        grad_input[index] = 0.125f;
    for (uint32_t index = 0u; index < EXPERTS * D_IN * D_OUT; ++index)
        grad_expert_weight[index] = -0.25f;
    for (uint32_t index = 0u; index < EXPERTS * D_OUT; ++index)
        grad_expert_bias[index] = 0.5f;
    for (uint32_t index = 0u; index < ROWS * TOP_K; ++index)
        grad_route_weights[index] = -0.75f;

    CHECK(cuda_training_supports("moeLinearBackward", "input_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_supports("moeLinearBackward", "weight_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_supports("moeLinearBackward", "bias_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_supports("moeLinearBackward", "route_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "input_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "bias_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "route_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_expert_weight,
                             sizeof(grad_expert_weight)) == 0);
    CHECK(cuda_training_sync(grad_expert_bias,
                             sizeof(grad_expert_bias)) == 0);
    CHECK(cuda_training_sync(grad_route_weights,
                             sizeof(grad_route_weights)) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < ROWS * D_IN; ++index)
        CHECK(close_f32(grad_input[index], 0.125f + input_delta[index]));
    for (uint32_t index = 0u; index < EXPERTS * D_IN * D_OUT; ++index)
        CHECK(close_f32(grad_expert_weight[index],
                        -0.25f + weight_delta[index]));
    for (uint32_t index = 0u; index < EXPERTS * D_OUT; ++index)
        CHECK(close_f32(grad_expert_bias[index],
                        0.5f + bias_delta[index]));
    for (uint32_t index = 0u; index < ROWS * TOP_K; ++index)
        CHECK(close_f32(grad_route_weights[index],
                        -0.75f + route_delta[index]));
    return 0;
}

static int test_moe_router_backward_normalized(void) {
    enum { ROWS = 2, D_MODEL = 2, EXPERTS = 3, TOP_K = 2 };
    const float input[ROWS * D_MODEL] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float router_weight[D_MODEL * EXPERTS] = {
        1.0f, 2.0f, -1.0f, 0.5f, -2.0f, 3.0f,
    };
    const float router_bias[EXPERTS] = {0.25f, -0.5f, 1.0f};
    const float route_indices[ROWS * TOP_K] = {2.0f, 0.0f, 1.0f, 2.0f};
    const float route_weights[ROWS * TOP_K] = {0.6f, 0.4f, 0.25f, 0.75f};
    const float grad_route_weights[ROWS * TOP_K] = {2.0f, -1.0f, 4.0f, 0.5f};
    const float expected_logits[ROWS * EXPERTS] = {
        -0.36f, 0.0f, 0.36f, 0.0f, 0.328125f, -0.328125f,
    };
    const float input_delta[ROWS * D_MODEL] = {
        -0.72f, 0.9f, 0.984375f, -1.640625f,
    };
    const float weight_delta[D_MODEL * EXPERTS] = {
        -0.36f, -0.328125f, 0.688125f,
        -0.72f, 0.984375f, -0.264375f,
    };
    const float bias_delta[EXPERTS] = {-0.36f, 0.328125f, 0.031875f};
    float grad_logits[ROWS * EXPERTS];
    float grad_input[ROWS * D_MODEL];
    float grad_router_weight[D_MODEL * EXPERTS];
    float grad_router_bias[EXPERTS];
    uint32_t params[8] = {
        ROWS, D_MODEL, EXPERTS, TOP_K, 1u, 1u, float_bits(2.0f), 0u,
    };
    void* hosts[11] = {
        (void*)input, (void*)router_weight, (void*)router_bias,
        (void*)route_indices, (void*)route_weights,
        (void*)grad_route_weights, grad_logits, grad_input,
        grad_router_weight, grad_router_bias, params,
    };
    size_t bytes[11] = {
        sizeof(input), sizeof(router_weight), sizeof(router_bias),
        sizeof(route_indices), sizeof(route_weights),
        sizeof(grad_route_weights), sizeof(grad_logits), sizeof(grad_input),
        sizeof(grad_router_weight), sizeof(grad_router_bias), sizeof(params),
    };
    const unsigned char access[11] = {
        1u, 1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[11] = {
        0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };

    cuda_graph_reset();
    for (uint32_t index = 0u; index < ROWS * EXPERTS; ++index)
        grad_logits[index] = 9.0f;
    for (uint32_t index = 0u; index < ROWS * D_MODEL; ++index)
        grad_input[index] = 0.125f;
    for (uint32_t index = 0u; index < D_MODEL * EXPERTS; ++index)
        grad_router_weight[index] = -0.25f;
    for (uint32_t index = 0u; index < EXPERTS; ++index)
        grad_router_bias[index] = 0.5f;

    CHECK(cuda_training_supports("moeRouterBackward", "logit_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_supports("moeRouterBackward", "input_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_supports("moeRouterBackward", "weight_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_supports("moeRouterBackward", "bias_main", bytes,
                                 11, 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "logit_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "input_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "bias_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_logits, sizeof(grad_logits)) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_router_weight,
                             sizeof(grad_router_weight)) == 0);
    CHECK(cuda_training_sync(grad_router_bias,
                             sizeof(grad_router_bias)) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < ROWS * EXPERTS; ++index)
        CHECK(close_f32(grad_logits[index], expected_logits[index]));
    for (uint32_t index = 0u; index < ROWS * D_MODEL; ++index)
        CHECK(close_f32(grad_input[index], 0.125f + input_delta[index]));
    for (uint32_t index = 0u; index < D_MODEL * EXPERTS; ++index)
        CHECK(close_f32(grad_router_weight[index],
                        -0.25f + weight_delta[index]));
    for (uint32_t index = 0u; index < EXPERTS; ++index)
        CHECK(close_f32(grad_router_bias[index],
                        0.5f + bias_delta[index]));
    return 0;
}

static int test_moe_router_backward_full_softmax(void) {
    const float input[1] = {0.0f};
    const float router_weight[2] = {2.0f, -3.0f};
    float dummy_bias[4] = {0.0f};
    const float route_indices[1] = {0.0f};
    const float route_weights[1] = {0.5f};
    const float grad_route_weights[1] = {2.0f};
    float grad_logits[2] = {9.0f, 9.0f};
    float grad_input[1] = {0.5f};
    float grad_router_weight[2] = {0.25f, 0.25f};
    float dummy_grad_bias[4] = {0.0f};
    uint32_t params[8] = {
        1u, 1u, 2u, 1u, 0u, 0u, float_bits(2.0f), 0u,
    };
    void* hosts[11] = {
        (void*)input, (void*)router_weight, dummy_bias,
        (void*)route_indices, (void*)route_weights,
        (void*)grad_route_weights, grad_logits, grad_input,
        grad_router_weight, dummy_grad_bias, params,
    };
    size_t bytes[11] = {
        sizeof(input), sizeof(router_weight), sizeof(dummy_bias),
        sizeof(route_indices), sizeof(route_weights),
        sizeof(grad_route_weights), sizeof(grad_logits), sizeof(grad_input),
        sizeof(grad_router_weight), sizeof(dummy_grad_bias), sizeof(params),
    };
    const unsigned char access[11] = {
        1u, 1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[11] = {
        0u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };

    cuda_graph_reset();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "logit_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "input_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("moeRouterBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 11,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_logits, sizeof(grad_logits)) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_router_weight,
                             sizeof(grad_router_weight)) == 0);
    cuda_training_end();

    CHECK(close_f32(grad_logits[0], 0.25f));
    CHECK(close_f32(grad_logits[1], -0.25f));
    CHECK(close_f32(grad_input[0], 1.75f));
    CHECK(close_f32(grad_router_weight[0], 0.25f));
    CHECK(close_f32(grad_router_weight[1], 0.25f));
    return 0;
}

static int test_moe_device_coherence_chain(void) {
    enum { ROWS = 2, WIDTH = 2, EXPERTS = 2, TOP_K = 2 };
    const float linear_input[ROWS * WIDTH] = {1.0f, 2.0f, -1.0f, 3.0f};
    const float expert_weight[EXPERTS * WIDTH * WIDTH] = {
        1.0f, 2.0f, 3.0f, 4.0f,
        -1.0f, 1.0f, 2.0f, -2.0f,
    };
    const float expert_bias[EXPERTS * WIDTH] = {
        0.5f, -0.5f, 1.0f, 2.0f,
    };
    const float route_indices[ROWS * TOP_K] = {0.0f, 1.0f, 1.0f, 0.0f};
    const float route_weights[ROWS * TOP_K] = {0.75f, 0.25f, 0.4f, 0.6f};
    const float grad_output[ROWS * WIDTH] = {1.0f, -2.0f, 0.5f, 1.0f};
    const float expected_route[ROWS * TOP_K] = {
        -11.5f, 6.0f, -1.0f, 13.75f,
    };
    const float expected_logits[ROWS * EXPERTS] = {
        -3.28125f, 3.28125f, 3.54f, -3.54f,
    };
    const float router_input[ROWS * WIDTH] = {0.0f};
    const float router_weight[WIDTH * EXPERTS] = {
        1.0f, 0.0f, 0.0f, 1.0f,
    };
    float dummy_router_bias[4] = {0.0f};
    float grad_route_weights[ROWS * TOP_K] = {0.0f};
    float grad_logits[ROWS * EXPERTS] = {7.0f, 7.0f, 7.0f, 7.0f};
    float router_grad_input[ROWS * WIDTH] = {0.25f, 0.25f, 0.25f, 0.25f};
    float unused_linear_grad_input[ROWS * WIDTH] = {0.0f};
    float unused_linear_grad_weight[EXPERTS * WIDTH * WIDTH] = {0.0f};
    float unused_linear_grad_bias[EXPERTS * WIDTH] = {0.0f};
    float unused_router_grad_weight[WIDTH * EXPERTS] = {0.0f};
    float dummy_router_grad_bias[4] = {0.0f};
    uint32_t linear_params[8] = {
        ROWS, WIDTH, WIDTH, EXPERTS, TOP_K, 1u, 0u, 0u,
    };
    uint32_t router_params[8] = {
        ROWS, WIDTH, EXPERTS, TOP_K, 1u, 0u, float_bits(1.0f), 0u,
    };
    void* linear_hosts[11] = {
        (void*)linear_input, (void*)expert_weight, (void*)expert_bias,
        (void*)route_indices, (void*)route_weights, (void*)grad_output,
        unused_linear_grad_input, unused_linear_grad_weight,
        unused_linear_grad_bias, grad_route_weights, linear_params,
    };
    size_t linear_bytes[11] = {
        sizeof(linear_input), sizeof(expert_weight), sizeof(expert_bias),
        sizeof(route_indices), sizeof(route_weights), sizeof(grad_output),
        sizeof(unused_linear_grad_input), sizeof(unused_linear_grad_weight),
        sizeof(unused_linear_grad_bias), sizeof(grad_route_weights),
        sizeof(linear_params),
    };
    void* router_hosts[11] = {
        (void*)router_input, (void*)router_weight, dummy_router_bias,
        (void*)route_indices, (void*)route_weights, grad_route_weights,
        grad_logits, router_grad_input, unused_router_grad_weight,
        dummy_router_grad_bias, router_params,
    };
    size_t router_bytes[11] = {
        sizeof(router_input), sizeof(router_weight), sizeof(dummy_router_bias),
        sizeof(route_indices), sizeof(route_weights),
        sizeof(grad_route_weights), sizeof(grad_logits),
        sizeof(router_grad_input), sizeof(unused_router_grad_weight),
        sizeof(dummy_router_grad_bias), sizeof(router_params),
    };
    const unsigned char access[11] = {
        1u, 1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[11] = {
        0u, 1u, 1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u,
    };

    cuda_graph_reset();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "route_main",
                                 linear_hosts, linear_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) == 0);
    /* Intentionally no sync: router binding 5 must reuse the device-written
     * route gradient instead of uploading the still-zero host buffer. */
    CHECK(cuda_training_dispatch("moeRouterBackward", "logit_main",
                                 router_hosts, router_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) == 0);
    /* Again no sync: input_main must consume device-written grad_logits. */
    CHECK(cuda_training_dispatch("moeRouterBackward", "input_main",
                                 router_hosts, router_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_route_weights,
                             sizeof(grad_route_weights)) == 0);
    CHECK(cuda_training_sync(grad_logits, sizeof(grad_logits)) == 0);
    CHECK(cuda_training_sync(router_grad_input,
                             sizeof(router_grad_input)) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < ROWS * TOP_K; ++index)
        CHECK(close_f32(grad_route_weights[index], expected_route[index]));
    for (uint32_t index = 0u; index < ROWS * EXPERTS; ++index) {
        CHECK(close_f32(grad_logits[index], expected_logits[index]));
        CHECK(close_f32(router_grad_input[index],
                        0.25f + expected_logits[index]));
    }
    return 0;
}

static int test_moe_backward_descriptors(void) {
    float linear_storage[10][4] = {{0.0f}};
    uint32_t linear_params[8] = {1u, 1u, 1u, 1u, 1u, 1u, 0u, 0u};
    void* linear_hosts[11];
    size_t linear_bytes[11] = {
        sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(float), sizeof(float), sizeof(float), sizeof(float),
        sizeof(float), sizeof(float), sizeof(linear_params),
    };
    unsigned char access[11] = {
        1u, 1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 3u, 1u,
    };
    unsigned char is_weight[11] = {0u};
    float linear_original[4][4];
    size_t malformed_bytes[11];

    for (int binding = 0; binding < 10; ++binding)
        linear_hosts[binding] = linear_storage[binding];
    linear_hosts[10] = linear_params;
    for (int destination = 0; destination < 4; ++destination) {
        for (int index = 0; index < 4; ++index)
            linear_storage[6 + destination][index] =
                (float)(destination * 4 + index + 1);
        memcpy(linear_original[destination], linear_storage[6 + destination],
               sizeof(linear_original[destination]));
    }

    CHECK(cuda_training_supports("moeLinearBackward", "input_main",
                                 linear_bytes, 11, 1u, 1u, 1u));
    CHECK(!cuda_training_supports("moeLinearBackward", "invalid",
                                  linear_bytes, 11, 1u, 1u, 1u));
    CHECK(!cuda_training_supports("moeLinearBackward", "input_main",
                                  linear_bytes, 10, 1u, 1u, 1u));
    CHECK(!cuda_training_supports("moeLinearBackward", "input_main",
                                  linear_bytes, 11, 1u, 2u, 1u));
    memcpy(malformed_bytes, linear_bytes, sizeof(linear_bytes));
    malformed_bytes[10] -= sizeof(uint32_t);
    CHECK(!cuda_training_supports("moeLinearBackward", "input_main",
                                  malformed_bytes, 11, 1u, 1u, 1u));
    memcpy(malformed_bytes, linear_bytes, sizeof(linear_bytes));
    malformed_bytes[4] += sizeof(float);
    CHECK(!cuda_training_supports("moeLinearBackward", "route_main",
                                  malformed_bytes, 11, 1u, 1u, 1u));

    cuda_graph_reset();
    access[6] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "input_main",
                                 linear_hosts, linear_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) != 0);
    cuda_training_end();
    access[6] = 3u;
    for (int destination = 0; destination < 4; ++destination)
        CHECK(memcmp(linear_storage[6 + destination],
                     linear_original[destination],
                     sizeof(linear_original[destination])) == 0);

    cuda_graph_reset();
    is_weight[10] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "input_main",
                                 linear_hosts, linear_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) != 0);
    cuda_training_end();
    is_weight[10] = 0u;

    cuda_graph_reset();
    linear_params[4] = 9u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "input_main",
                                 linear_hosts, linear_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) != 0);
    cuda_training_end();
    linear_params[4] = 1u;

    cuda_graph_reset();
    linear_params[7] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "weight_main",
                                 linear_hosts, linear_bytes, access, is_weight,
                                 11, 1u, 1u, 1u) != 0);
    cuda_training_end();
    linear_params[7] = 0u;

    cuda_graph_reset();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("moeLinearBackward", "input_main",
                                 linear_hosts, linear_bytes, access, is_weight,
                                 11, 2u, 1u, 1u) != 0);
    cuda_training_end();

    {
        float dummy_bias[4] = {0.0f};
        float dummy_grad_bias[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        const float original_dummy[4] = {5.0f, 6.0f, 7.0f, 8.0f};
        linear_params[5] = 0u;
        linear_hosts[2] = dummy_bias;
        linear_hosts[8] = dummy_grad_bias;
        linear_bytes[2] = sizeof(dummy_bias);
        linear_bytes[8] = sizeof(dummy_grad_bias);
        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeLinearBackward", "bias_main",
                                     linear_hosts, linear_bytes, access,
                                     is_weight, 11, 1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(dummy_grad_bias, original_dummy,
                     sizeof(dummy_grad_bias)) == 0);
        linear_params[5] = 1u;
    }

    for (int destination = 0; destination < 4; ++destination)
        CHECK(memcmp(linear_storage[6 + destination],
                     linear_original[destination],
                     sizeof(linear_original[destination])) == 0);

    {
        float input[1] = {0.0f};
        float weight[2] = {1.0f, -1.0f};
        float bias[3] = {0.0f, 0.0f, 17.0f};
        float indices[1] = {0.0f};
        float gates[1] = {0.5f};
        float grad_gates[1] = {1.0f};
        float grad_logits[2] = {0.5f, -1.0f};
        float grad_input[1] = {3.0f};
        float grad_weight[2] = {4.0f, 5.0f};
        float grad_bias[3] = {1.0f, 2.0f, 99.0f};
        const float original_input[1] = {3.0f};
        const float original_weight[2] = {4.0f, 5.0f};
        float stable_logits[2];
        float stable_bias[3];
        uint32_t params[8] = {
            1u, 1u, 2u, 1u, 1u, 1u, float_bits(1.0f), 0u,
        };
        void* hosts[11] = {
            input, weight, bias, indices, gates, grad_gates, grad_logits,
            grad_input, grad_weight, grad_bias, params,
        };
        size_t bytes[11] = {
            sizeof(input), sizeof(weight), sizeof(bias), sizeof(indices),
            sizeof(gates), sizeof(grad_gates), sizeof(grad_logits),
            sizeof(grad_input), sizeof(grad_weight), sizeof(grad_bias),
            sizeof(params),
        };

        CHECK(cuda_training_supports("moeRouterBackward", "bias_main",
                                     bytes, 11, 1u, 1u, 1u));
        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeRouterBackward", "bias_main",
                                     hosts, bytes, access, is_weight, 11,
                                     1u, 1u, 1u) == 0);
        CHECK(cuda_training_sync(grad_bias, sizeof(grad_bias)) == 0);
        cuda_training_end();
        CHECK(close_f32(grad_bias[0], 1.5f));
        CHECK(close_f32(grad_bias[1], 1.0f));
        CHECK(grad_bias[2] == 99.0f);
        memcpy(stable_logits, grad_logits, sizeof(stable_logits));
        memcpy(stable_bias, grad_bias, sizeof(stable_bias));

        params[4] = 2u;
        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeRouterBackward", "logit_main",
                                     hosts, bytes, access, is_weight, 11,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        params[4] = 1u;
        CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
        CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);

        params[6] = float_bits(NAN);
        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeRouterBackward", "input_main",
                                     hosts, bytes, access, is_weight, 11,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        params[6] = float_bits(1.0f);

        params[7] = 1u;
        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeRouterBackward", "weight_main",
                                     hosts, bytes, access, is_weight, 11,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        params[7] = 0u;

        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeRouterBackward", "input_main",
                                     hosts, bytes, access, is_weight, 11,
                                     2u, 1u, 1u) != 0);
        cuda_training_end();

        params[5] = 0u;
        bytes[2] = bytes[9] = 4u * sizeof(float);
        cuda_graph_reset();
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("moeRouterBackward", "bias_main",
                                     hosts, bytes, access, is_weight, 11,
                                     1u, 1u, 1u) != 0);
        cuda_training_end();
        CHECK(memcmp(grad_input, original_input, sizeof(grad_input)) == 0);
        CHECK(memcmp(grad_weight, original_weight, sizeof(grad_weight)) == 0);
        CHECK(memcmp(grad_logits, stable_logits, sizeof(grad_logits)) == 0);
        CHECK(memcmp(grad_bias, stable_bias, sizeof(grad_bias)) == 0);
    }
    return 0;
}

#endif
