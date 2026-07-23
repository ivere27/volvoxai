static int test_batchnorm2d_backward(void) {
    enum { LENGTH = 4, CHANNELS = 2 };
    const float input[LENGTH] = {3.0f, 2.0f, -1.0f, -4.0f};
    const float weight[CHANNELS] = {2.0f, -3.0f};
    const float running_mean[CHANNELS] = {1.0f, -1.0f};
    const float running_variance[CHANNELS] = {3.0f, 8.0f};
    const float grad_output[LENGTH] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float expected_input[LENGTH] = {1.1f, -1.8f, 3.3f, -3.6f};
    const float expected_weight[CHANNELS] = {-1.5f, -2.5f};
    const float expected_bias[CHANNELS] = {5.0f, 5.0f};
    float grad_input[LENGTH] = {0.1f, 0.2f, 0.3f, 0.4f};
    float grad_weight[CHANNELS] = {0.5f, -0.5f};
    float grad_bias[CHANNELS] = {1.0f, -1.0f};
    uint32_t params[4] = {LENGTH, CHANNELS, float_bits(1.0f), 0u};
    void* hosts[9] = {
        (void*)input, (void*)weight, (void*)running_mean,
        (void*)running_variance, (void*)grad_output, grad_input,
        grad_weight, grad_bias, params,
    };
    size_t bytes[9] = {
        sizeof(input), sizeof(weight), sizeof(running_mean),
        sizeof(running_variance), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(grad_bias), sizeof(params),
    };
    const unsigned char access[9] = {
        1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 1u,
    };
    const unsigned char is_weight[9] = {
        0u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 0u,
    };

    CHECK(cuda_training_supports("batchNorm2DBackward", "input_main",
                                 bytes, 9, 1u, 1u, 1u));
    CHECK(cuda_training_supports("batchNorm2DBackward", "param_main",
                                 bytes, 9, 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("batchNorm2DBackward", "input_main", hosts,
                                 bytes, access, is_weight, 9,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("batchNorm2DBackward", "param_main", hosts,
                                 bytes, access, is_weight, 9,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(cuda_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < LENGTH; ++index)
        CHECK(close_f32(grad_input[index], expected_input[index]));
    for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
        CHECK(close_f32(grad_weight[channel], expected_weight[channel]));
        CHECK(close_f32(grad_bias[channel], expected_bias[channel]));
    }
    return 0;
}

static int test_groupnorm_backward(void) {
    enum { LENGTH = 8, CHANNELS = 4 };
    const float input[LENGTH] = {
        1.0f, -1.0f, 1.0f, -1.0f,
        3.0f, -3.0f, 3.0f, -3.0f,
    };
    const float weight[CHANNELS] = {1.0f, 2.0f, -1.0f, 0.5f};
    const float grad_output[LENGTH] = {
        1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f,
    };
    const float expected_input[LENGTH] = {
        -0.12962964f, 0.12962964f, -0.19444445f, 0.19444445f,
        -0.05555555f, 0.05555555f, -0.08333334f, 0.08333334f,
    };
    const float expected_weight[CHANNELS] = {
        1.4333334f, -1.1333333f, 1.6333334f, -0.9333334f,
    };
    const float expected_bias[CHANNELS] = {1.9f, 1.8f, 1.7f, 1.6f};
    const float expected_weight_no_bias[CHANNELS] = {
        1.3333334f, -1.3333334f, 1.3333334f, -1.3333334f,
    };
    float grad_input[LENGTH] = {0};
    float grad_weight[CHANNELS] = {0.1f, 0.2f, 0.3f, 0.4f};
    float grad_bias[CHANNELS] = {-0.1f, -0.2f, -0.3f, -0.4f};
    uint32_t params[8] = {
        1u, 1u, 2u, CHANNELS, 2u, 1u, float_bits(4.0f), 0u,
    };
    void* hosts[7] = {
        (void*)input, (void*)weight, (void*)grad_output, grad_input,
        grad_weight, grad_bias, params,
    };
    size_t bytes[7] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(grad_input),
        sizeof(grad_weight), sizeof(grad_bias), sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};

    CHECK(cuda_training_supports("groupNormBackward", "input_main",
                                 bytes, 7, 2u, 1u, 1u));
    CHECK(cuda_training_supports("groupNormBackward", "param_main",
                                 bytes, 7, 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "input_main", hosts,
                                 bytes, access, is_weight, 7,
                                 2u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "param_main", hosts,
                                 bytes, access, is_weight, 7,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, sizeof(grad_weight)) == 0);
    CHECK(cuda_training_sync(grad_bias, sizeof(grad_bias)) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < LENGTH; ++index)
        CHECK(close_f32(grad_input[index], expected_input[index]));
    for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
        CHECK(close_f32(grad_weight[channel], expected_weight[channel]));
        CHECK(close_f32(grad_bias[channel], expected_bias[channel]));
    }

    float grad_input_no_bias[LENGTH] = {0};
    float grad_weight_no_bias[CHANNELS] = {0};
    uint32_t dummy_bias[4] = {
        0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u,
    };
    const uint32_t expected_dummy[4] = {
        0x11223344u, 0x55667788u, 0x99aabbccu, 0xddeeff00u,
    };
    params[5] = 0u;
    hosts[3] = grad_input_no_bias;
    hosts[4] = grad_weight_no_bias;
    hosts[5] = dummy_bias;
    bytes[3] = sizeof(grad_input_no_bias);
    bytes[4] = sizeof(grad_weight_no_bias);
    bytes[5] = sizeof(dummy_bias);

    CHECK(cuda_training_supports(
        "groupNormBackward", "param_cooperative", bytes, 7,
        CHANNELS, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "input_main", hosts,
                                 bytes, access, is_weight, 7,
                                 2u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "param_cooperative", hosts,
                                 bytes, access, is_weight, 7,
                                 CHANNELS, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input_no_bias,
                             sizeof(grad_input_no_bias)) == 0);
    CHECK(cuda_training_sync(grad_weight_no_bias,
                             sizeof(grad_weight_no_bias)) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < LENGTH; ++index)
        CHECK(close_f32(grad_input_no_bias[index], expected_input[index]));
    for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
        CHECK(close_f32(grad_weight_no_bias[channel],
                        expected_weight_no_bias[channel]));
    }
    CHECK(memcmp(dummy_bias, expected_dummy, sizeof(dummy_bias)) == 0);
    return 0;
}

static void groupnorm_backward_input_reference(
        const float* input, const float* weight, const float* grad_output,
        float* grad_input, const uint32_t params[8]) {
    uint32_t area = params[1] * params[2];
    uint32_t channels_per_group = params[3] / params[4];
    uint32_t count = area * channels_per_group;
    float epsilon;
    memcpy(&epsilon, &params[6], sizeof(epsilon));
    for (uint32_t batch = 0u; batch < params[0]; ++batch) {
        for (uint32_t group = 0u; group < params[4]; ++group) {
            uint32_t channel_start = group * channels_per_group;
            float sum = 0.0f;
            float square_sum = 0.0f;
            for (uint32_t spatial = 0u; spatial < area; ++spatial) {
                uint32_t base = (batch * area + spatial) * params[3] +
                    channel_start;
                for (uint32_t local_channel = 0u;
                     local_channel < channels_per_group; ++local_channel) {
                    float value = input[base + local_channel];
                    sum += value;
                    square_sum += value * value;
                }
            }
            float element_count = (float)count;
            float mean = sum / element_count;
            float variance = square_sum / element_count - mean * mean;
            if (variance < 0.0f) variance = 0.0f;
            float inverse_standard_deviation =
                1.0f / sqrtf(variance + epsilon);
            float sum_scaled_gradient = 0.0f;
            float sum_scaled_gradient_xhat = 0.0f;
            for (uint32_t spatial = 0u; spatial < area; ++spatial) {
                uint32_t base = (batch * area + spatial) * params[3] +
                    channel_start;
                for (uint32_t local_channel = 0u;
                     local_channel < channels_per_group; ++local_channel) {
                    uint32_t channel = channel_start + local_channel;
                    uint32_t index = base + local_channel;
                    float xhat = (input[index] - mean) *
                        inverse_standard_deviation;
                    float scaled_gradient = grad_output[index] *
                        weight[channel];
                    sum_scaled_gradient += scaled_gradient;
                    sum_scaled_gradient_xhat += scaled_gradient * xhat;
                }
            }
            for (uint32_t spatial = 0u; spatial < area; ++spatial) {
                uint32_t base = (batch * area + spatial) * params[3] +
                    channel_start;
                for (uint32_t local_channel = 0u;
                     local_channel < channels_per_group; ++local_channel) {
                    uint32_t channel = channel_start + local_channel;
                    uint32_t index = base + local_channel;
                    float xhat = (input[index] - mean) *
                        inverse_standard_deviation;
                    float scaled_gradient = grad_output[index] *
                        weight[channel];
                    float dx = inverse_standard_deviation / element_count *
                        (element_count * scaled_gradient -
                         sum_scaled_gradient -
                         xhat * sum_scaled_gradient_xhat);
                    grad_input[index] += dx;
                }
            }
        }
    }
}

static void groupnorm_backward_param_reference(
        const float* input, const float* grad_output, float* grad_weight,
        float* grad_bias, const uint32_t params[8]) {
    uint32_t area = params[1] * params[2];
    uint32_t channels_per_group = params[3] / params[4];
    uint32_t count = area * channels_per_group;
    float epsilon;
    memcpy(&epsilon, &params[6], sizeof(epsilon));
    for (uint32_t batch = 0u; batch < params[0]; ++batch) {
        for (uint32_t group = 0u; group < params[4]; ++group) {
            uint32_t channel_start = group * channels_per_group;
            float sum = 0.0f;
            float square_sum = 0.0f;
            for (uint32_t spatial = 0u; spatial < area; ++spatial) {
                uint32_t base = (batch * area + spatial) * params[3] +
                    channel_start;
                for (uint32_t local_channel = 0u;
                     local_channel < channels_per_group; ++local_channel) {
                    float value = input[base + local_channel];
                    sum += value;
                    square_sum += value * value;
                }
            }
            float element_count = (float)count;
            float mean = sum / element_count;
            float variance = square_sum / element_count - mean * mean;
            if (variance < 0.0f) variance = 0.0f;
            float inverse_standard_deviation =
                1.0f / sqrtf(variance + epsilon);
            for (uint32_t local_channel = 0u;
                 local_channel < channels_per_group; ++local_channel) {
                uint32_t channel = channel_start + local_channel;
                float weight_sum = 0.0f;
                float bias_sum = 0.0f;
                for (uint32_t spatial = 0u; spatial < area; ++spatial) {
                    uint32_t index = (batch * area + spatial) * params[3] +
                        channel;
                    weight_sum += grad_output[index] *
                        (input[index] - mean) * inverse_standard_deviation;
                    bias_sum += grad_output[index];
                }
                grad_weight[channel] += weight_sum;
                if (params[5]) grad_bias[channel] += bias_sum;
            }
        }
    }
}

static int test_groupnorm_backward_input_cooperative_tail(void) {
    enum {
        BATCH = 2,
        HEIGHT = 1,
        WIDTH = 17,
        CHANNELS = 10,
        GROUPS = 2,
        LENGTH = BATCH * HEIGHT * WIDTH * CHANNELS,
    };
    float input[LENGTH];
    float weight[CHANNELS];
    float grad_output[LENGTH];
    float initial[LENGTH];
    float expected[LENGTH];
    float first[LENGTH];
    float second[LENGTH];
    float initial_weight[CHANNELS];
    float initial_bias[CHANNELS];
    float expected_weight[CHANNELS];
    float expected_bias[CHANNELS];
    float first_weight[CHANNELS];
    float first_bias[CHANNELS];
    float second_weight[CHANNELS];
    float second_bias[CHANNELS];
    uint32_t params[8] = {
        BATCH, HEIGHT, WIDTH, CHANNELS, GROUPS, 1u,
        float_bits(1.0e-3f), 0u,
    };
    void* hosts[7] = {
        input, weight, grad_output, first, first_weight, first_bias, params,
    };
    size_t bytes[7] = {
        sizeof(input), sizeof(weight), sizeof(grad_output), sizeof(first),
        sizeof(first_weight), sizeof(first_bias), sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};

    for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
        weight[channel] = 0.4f + (float)channel * 0.075f;
        initial_weight[channel] =
            ((float)(int)(channel % 5u) - 2.0f) * 0.017f;
        initial_bias[channel] =
            ((float)(int)(channel % 7u) - 3.0f) * 0.013f;
    }
    for (uint32_t index = 0u; index < LENGTH; ++index) {
        input[index] =
            ((float)(int)((index * 13u) % 37u) - 18.0f) * 0.041f;
        grad_output[index] =
            ((float)(int)((index * 19u) % 31u) - 15.0f) * 0.027f;
        initial[index] = ((float)(int)(index % 11u) - 5.0f) * 0.003f;
    }
    memcpy(expected, initial, sizeof(expected));
    groupnorm_backward_input_reference(
        input, weight, grad_output, expected, params);
    memcpy(expected_weight, initial_weight, sizeof(expected_weight));
    memcpy(expected_bias, initial_bias, sizeof(expected_bias));
    groupnorm_backward_param_reference(
        input, grad_output, expected_weight, expected_bias, params);

    CHECK(cuda_training_supports("groupNormBackward", "input_main",
                                 bytes, 7, BATCH * GROUPS, 1u, 1u));
    CHECK(!cuda_training_supports(
        "groupNormBackward", "input_main", bytes, 7,
        (uint32_t)INT32_MAX + 1u, 1u, 1u));
    memcpy(first, initial, sizeof(first));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "input_main",
                                 hosts, bytes, access, is_weight, 7,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(first, initial, sizeof(first)) == 0);
    for (uint32_t run = 0u; run < 2u; ++run) {
        float* output = run == 0u ? first : second;
        memcpy(output, initial, sizeof(initial));
        hosts[3] = output;
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch("groupNormBackward", "input_main",
                                     hosts, bytes, access, is_weight, 7,
                                     BATCH * GROUPS, 1u, 1u) == 0);
        CHECK(cuda_training_sync(output, sizeof(first)) == 0);
        cuda_training_end();
        if (run == 0u) cuda_graph_reset();
    }
    CHECK(memcmp(first, second, sizeof(first)) == 0);
    for (uint32_t index = 0u; index < LENGTH; ++index) {
        float scale = fmaxf(1.0f, fmaxf(fabsf(first[index]),
                                        fabsf(expected[index])));
        CHECK(fabsf(first[index] - expected[index]) <= 5.0e-5f * scale);
    }

    CHECK(cuda_training_supports(
        "groupNormBackward", "param_cooperative", bytes, 7,
        CHANNELS, 1u, 1u));
    CHECK(!cuda_training_supports(
        "groupNormBackward", "param_cooperative", bytes, 7,
        CHANNELS - 1u, 1u, 1u));
    CHECK(!cuda_training_supports(
        "groupNormBackward", "param_cooperative", bytes, 7,
        (uint32_t)INT32_MAX + 1u, 1u, 1u));
    memcpy(first_weight, initial_weight, sizeof(first_weight));
    memcpy(first_bias, initial_bias, sizeof(first_bias));
    hosts[4] = first_weight;
    hosts[5] = first_bias;
    cuda_graph_reset();
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch(
        "groupNormBackward", "param_cooperative", hosts, bytes, access,
        is_weight, 7, CHANNELS - 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(first_weight, initial_weight, sizeof(first_weight)) == 0);
    CHECK(memcmp(first_bias, initial_bias, sizeof(first_bias)) == 0);
    for (uint32_t run = 0u; run < 2u; ++run) {
        float* output_weight = run == 0u ? first_weight : second_weight;
        float* output_bias = run == 0u ? first_bias : second_bias;
        memcpy(output_weight, initial_weight, sizeof(initial_weight));
        memcpy(output_bias, initial_bias, sizeof(initial_bias));
        hosts[4] = output_weight;
        hosts[5] = output_bias;
        CHECK(cuda_training_begin() == 0);
        CHECK(cuda_training_dispatch(
            "groupNormBackward", "param_cooperative", hosts, bytes, access,
            is_weight, 7, CHANNELS, 1u, 1u) == 0);
        CHECK(cuda_training_sync(output_weight, sizeof(first_weight)) == 0);
        CHECK(cuda_training_sync(output_bias, sizeof(first_bias)) == 0);
        cuda_training_end();
        if (run == 0u) cuda_graph_reset();
    }
    CHECK(memcmp(first_weight, second_weight, sizeof(first_weight)) == 0);
    CHECK(memcmp(first_bias, second_bias, sizeof(first_bias)) == 0);
    for (uint32_t channel = 0u; channel < CHANNELS; ++channel) {
        float weight_scale = fmaxf(
            1.0f, fmaxf(fabsf(first_weight[channel]),
                         fabsf(expected_weight[channel])));
        float bias_scale = fmaxf(
            1.0f, fmaxf(fabsf(first_bias[channel]),
                         fabsf(expected_bias[channel])));
        CHECK(fabsf(first_weight[channel] - expected_weight[channel]) <=
              5.0e-5f * weight_scale);
        CHECK(fabsf(first_bias[channel] - expected_bias[channel]) <=
              5.0e-5f * bias_scale);
    }
    return 0;
}

static int test_prelu_backward_mode(uint32_t weight_length) {
    enum { LENGTH = 6, CHANNELS = 3 };
    const float input[LENGTH] = {-2.0f, 0.0f, 3.0f, -1.0f, 2.0f, -4.0f};
    const float grad_output[LENGTH] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
    const float expected_channel_input[LENGTH] = {
        0.6f, 0.9f, 3.5f, 0.9f, 5.5f, 2.3f,
    };
    const float expected_scalar_input[LENGTH] = {
        0.75f, 1.0f, 3.5f, 1.5f, 5.5f, 2.0f,
    };
    const float channel_weight[CHANNELS] = {0.1f, 0.2f, 0.3f};
    const float scalar_weight[1] = {0.25f};
    const float* weight = weight_length == 1u ? scalar_weight : channel_weight;
    const float* expected_input = weight_length == 1u
        ? expected_scalar_input : expected_channel_input;
    float grad_input[LENGTH] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    float grad_weight[CHANNELS] = {1.0f, 2.0f, 3.0f};
    const float expected_channel_weight[CHANNELS] = {-5.0f, 2.0f, -21.0f};
    const float expected_scalar_weight[1] = {-29.0f};
    uint32_t params[4] = {LENGTH, CHANNELS, weight_length, 0u};
    void* hosts[6] = {
        (void*)input, (void*)weight, (void*)grad_output, grad_input,
        grad_weight, params,
    };
    size_t bytes[6] = {
        sizeof(input), (size_t)weight_length * sizeof(float),
        sizeof(grad_output), sizeof(grad_input),
        (size_t)weight_length * sizeof(float), sizeof(params),
    };
    const unsigned char access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char is_weight[6] = {0u, 1u, 0u, 0u, 0u, 0u};

    CHECK(weight_length == 1u || weight_length == CHANNELS);
    CHECK(cuda_training_supports("preluBackward", "input_main", bytes, 6,
                                 1u, 1u, 1u));
    CHECK(cuda_training_supports("preluBackward", "weight_main", bytes, 6,
                                 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("preluBackward", "input_main", hosts,
                                 bytes, access, is_weight, 6,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_dispatch("preluBackward", "weight_main", hosts,
                                 bytes, access, is_weight, 6,
                                 1u, 1u, 1u) == 0);
    CHECK(cuda_training_sync(grad_input, sizeof(grad_input)) == 0);
    CHECK(cuda_training_sync(grad_weight, bytes[4]) == 0);
    cuda_training_end();

    for (uint32_t index = 0u; index < LENGTH; ++index)
        CHECK(close_f32(grad_input[index], expected_input[index]));
    for (uint32_t index = 0u; index < weight_length; ++index) {
        float expected = weight_length == 1u
            ? expected_scalar_weight[index] : expected_channel_weight[index];
        CHECK(close_f32(grad_weight[index], expected));
    }
    return 0;
}

static int test_prelu_backward(void) {
    CHECK(test_prelu_backward_mode(1u) == 0);
    CHECK(test_prelu_backward_mode(3u) == 0);
    return 0;
}

static int test_norm_extra_backward_descriptors(void) {
    float batch_input[4] = {0};
    float batch_weight[2] = {1.0f, 1.0f};
    float batch_mean[2] = {0};
    float batch_variance[2] = {1.0f, 1.0f};
    float batch_grad_output[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float batch_grad_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float batch_grad_weight[2] = {-1.0f, -2.0f};
    float batch_grad_bias[2] = {0.25f, -0.5f};
    const float original_batch_input[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    const float original_batch_weight[2] = {-1.0f, -2.0f};
    const float original_batch_bias[2] = {0.25f, -0.5f};
    uint32_t batch_params[4] = {4u, 2u, float_bits(1.0f), 0u};
    void* batch_hosts[9] = {
        batch_input, batch_weight, batch_mean, batch_variance,
        batch_grad_output, batch_grad_input, batch_grad_weight,
        batch_grad_bias, batch_params,
    };
    size_t batch_bytes[9] = {
        sizeof(batch_input), sizeof(batch_weight), sizeof(batch_mean),
        sizeof(batch_variance), sizeof(batch_grad_output),
        sizeof(batch_grad_input), sizeof(batch_grad_weight),
        sizeof(batch_grad_bias), sizeof(batch_params),
    };
    const unsigned char batch_access[9] = {
        1u, 1u, 1u, 1u, 1u, 3u, 3u, 3u, 1u,
    };
    const unsigned char batch_bad_access[9] = {
        1u, 1u, 1u, 1u, 1u, 1u, 3u, 3u, 1u,
    };
    const unsigned char batch_is_weight[9] = {0};
    size_t malformed_batch_bytes[9];

    memcpy(malformed_batch_bytes, batch_bytes, sizeof(batch_bytes));
    malformed_batch_bytes[8] -= sizeof(uint32_t);
    CHECK(!cuda_training_supports("batchNorm2DBackward", "input_main",
                                  malformed_batch_bytes, 9,
                                  1u, 1u, 1u));
    CHECK(!cuda_training_supports("batchNorm2DBackward", "bad", batch_bytes,
                                  9, 1u, 1u, 1u));
    CHECK(!cuda_training_supports("batchNorm2DBackward", "input_main",
                                  batch_bytes, 8, 1u, 1u, 1u));
    CHECK(!cuda_training_supports("batchNorm2DBackward", "input_main",
                                  batch_bytes, 9, 1u, 2u, 1u));

    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("batchNorm2DBackward", "input_main",
                                 batch_hosts, batch_bytes, batch_bad_access,
                                 batch_is_weight, 9, 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(batch_grad_input, original_batch_input,
                 sizeof(batch_grad_input)) == 0);
    CHECK(memcmp(batch_grad_weight, original_batch_weight,
                 sizeof(batch_grad_weight)) == 0);
    CHECK(memcmp(batch_grad_bias, original_batch_bias,
                 sizeof(batch_grad_bias)) == 0);

    batch_params[0] = 3u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("batchNorm2DBackward", "param_main",
                                 batch_hosts, batch_bytes, batch_access,
                                 batch_is_weight, 9, 1u, 1u, 1u) != 0);
    cuda_training_end();
    batch_params[0] = 4u;
    CHECK(memcmp(batch_grad_input, original_batch_input,
                 sizeof(batch_grad_input)) == 0);
    CHECK(memcmp(batch_grad_weight, original_batch_weight,
                 sizeof(batch_grad_weight)) == 0);
    CHECK(memcmp(batch_grad_bias, original_batch_bias,
                 sizeof(batch_grad_bias)) == 0);

    float group_input[8] = {0};
    float group_weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
    float group_grad_output[8] = {0};
    float group_grad_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float group_grad_weight[4] = {-1, -2, -3, -4};
    uint32_t group_dummy[4] = {11u, 22u, 33u, 44u};
    const float original_group_input[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    const float original_group_weight[4] = {-1, -2, -3, -4};
    const uint32_t original_group_dummy[4] = {11u, 22u, 33u, 44u};
    uint32_t group_params[8] = {
        1u, 1u, 2u, 4u, 2u, 0u, float_bits(1.0f), 0u,
    };
    void* group_hosts[7] = {
        group_input, group_weight, group_grad_output, group_grad_input,
        group_grad_weight, group_dummy, group_params,
    };
    size_t group_bytes[7] = {
        sizeof(group_input), sizeof(group_weight), sizeof(group_grad_output),
        sizeof(group_grad_input), sizeof(group_grad_weight),
        sizeof(group_dummy), sizeof(group_params),
    };
    size_t malformed_group_bytes[7];
    const unsigned char group_access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char group_is_weight[7] = {0};

    memcpy(malformed_group_bytes, group_bytes, sizeof(group_bytes));
    malformed_group_bytes[5] -= sizeof(uint32_t);
    CHECK(cuda_training_supports("groupNormBackward", "param_main",
                                 malformed_group_bytes, 7, 1u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "param_main",
                                 group_hosts, malformed_group_bytes,
                                 group_access, group_is_weight, 7,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(group_grad_input, original_group_input,
                 sizeof(group_grad_input)) == 0);
    CHECK(memcmp(group_grad_weight, original_group_weight,
                 sizeof(group_grad_weight)) == 0);
    CHECK(memcmp(group_dummy, original_group_dummy,
                 sizeof(group_dummy)) == 0);

    group_params[6] = float_bits(NAN);
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("groupNormBackward", "input_main",
                                 group_hosts, group_bytes, group_access,
                                 group_is_weight, 7, 2u, 1u, 1u) != 0);
    cuda_training_end();
    group_params[6] = float_bits(1.0f);
    CHECK(memcmp(group_grad_input, original_group_input,
                 sizeof(group_grad_input)) == 0);
    CHECK(memcmp(group_grad_weight, original_group_weight,
                 sizeof(group_grad_weight)) == 0);
    CHECK(memcmp(group_dummy, original_group_dummy,
                 sizeof(group_dummy)) == 0);

    float prelu_input[6] = {0};
    float prelu_weight[3] = {0.1f, 0.2f, 0.3f};
    float prelu_grad_output[6] = {1, 2, 3, 4, 5, 6};
    float prelu_grad_input[6] = {1, 2, 3, 4, 5, 6};
    float prelu_grad_weight[3] = {-1, -2, -3};
    const float original_prelu_input[6] = {1, 2, 3, 4, 5, 6};
    const float original_prelu_weight[3] = {-1, -2, -3};
    uint32_t prelu_params[4] = {6u, 3u, 3u, 0u};
    void* prelu_hosts[6] = {
        prelu_input, prelu_weight, prelu_grad_output, prelu_grad_input,
        prelu_grad_weight, prelu_params,
    };
    size_t prelu_bytes[6] = {
        sizeof(prelu_input), sizeof(prelu_weight), sizeof(prelu_grad_output),
        sizeof(prelu_grad_input), sizeof(prelu_grad_weight),
        sizeof(prelu_params),
    };
    size_t malformed_prelu_bytes[6];
    const unsigned char prelu_access[6] = {1u, 1u, 1u, 3u, 3u, 1u};
    const unsigned char prelu_is_weight[6] = {0};
    unsigned char prelu_bad_is_weight[6] = {0};

    memcpy(malformed_prelu_bytes, prelu_bytes, sizeof(prelu_bytes));
    malformed_prelu_bytes[4] -= sizeof(float);
    CHECK(!cuda_training_supports("preluBackward", "weight_main",
                                  malformed_prelu_bytes, 6,
                                  1u, 1u, 1u));
    prelu_params[2] = 2u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("preluBackward", "weight_main",
                                 prelu_hosts, prelu_bytes, prelu_access,
                                 prelu_is_weight, 6, 1u, 1u, 1u) != 0);
    cuda_training_end();
    prelu_params[2] = 3u;
    CHECK(memcmp(prelu_grad_input, original_prelu_input,
                 sizeof(prelu_grad_input)) == 0);
    CHECK(memcmp(prelu_grad_weight, original_prelu_weight,
                 sizeof(prelu_grad_weight)) == 0);

    prelu_bad_is_weight[5] = 1u;
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("preluBackward", "input_main",
                                 prelu_hosts, prelu_bytes, prelu_access,
                                 prelu_bad_is_weight, 6,
                                 1u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(prelu_grad_input, original_prelu_input,
                 sizeof(prelu_grad_input)) == 0);
    CHECK(memcmp(prelu_grad_weight, original_prelu_weight,
                 sizeof(prelu_grad_weight)) == 0);

    CHECK(cuda_training_supports("preluBackward", "input_main", prelu_bytes,
                                 6, 2u, 1u, 1u));
    CHECK(cuda_training_begin() == 0);
    CHECK(cuda_training_dispatch("preluBackward", "input_main",
                                 prelu_hosts, prelu_bytes, prelu_access,
                                 prelu_is_weight, 6, 2u, 1u, 1u) != 0);
    cuda_training_end();
    CHECK(memcmp(prelu_grad_input, original_prelu_input,
                 sizeof(prelu_grad_input)) == 0);
    CHECK(memcmp(prelu_grad_weight, original_prelu_weight,
                 sizeof(prelu_grad_weight)) == 0);
    return 0;
}
