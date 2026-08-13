#ifndef VOLVOXAI_TEST_TRAINING_CUDA_ACCUM_OPTIMIZER_INC
#define VOLVOXAI_TEST_TRAINING_CUDA_ACCUM_OPTIMIZER_INC

/* Textually included by test_training_backward.c after its graph helpers.
 * The TrainStep calls below are public API calls.  The private optimizer-buffer
 * accessor is used only to observe AdamW state after the public update. */

#if VOLVOXAI_ENABLE_CUDA

typedef struct {
    float weight[12];
    float bias[4];
    float weight_m[12];
    float weight_v[12];
    float bias_m[4];
    float bias_v[4];
    float losses[4];
    int correct[4];
    int examples[4];
} CudaAccumOptimizerResult;

typedef struct {
    float weight[12];
    float bias[4];
    float weight_m[12];
    float weight_v[12];
    float bias_m[4];
    float bias_v[4];
} CudaOptimizerHandoffResult;

static int cuda_accum_optimizer_close(float actual, float expected,
                                      float tolerance) {
    float scale = 1.0f + fmaxf(fabsf(actual), fabsf(expected));
    return isfinite(actual) && isfinite(expected) &&
        fabsf(actual - expected) <= tolerance * scale;
}

static int cuda_accum_optimizer_array_close(const float* left,
                                            const float* right, int count,
                                            float tolerance) {
    if (!left || !right || count <= 0) return 0;
    for (int index = 0; index < count; index++)
        if (!cuda_accum_optimizer_close(
                left[index], right[index], tolerance)) return 0;
    return 1;
}

static int cuda_accum_optimizer_any_changed(const float* before,
                                            const float* after, int count) {
    if (!before || !after || count <= 0) return 0;
    for (int index = 0; index < count; index++)
        if (fabsf(before[index] - after[index]) > 1.0e-8f) return 1;
    return 0;
}

static int run_cuda_accum_optimizer_case(
        int use_cuda, CudaAccumOptimizerResult* result) {
    const int input_shape[2] = {2, 3};
    const int weight_shape[2] = {4, 3};
    const int channel_shape[1] = {4};
    const int logits_shape[2] = {2, 4};
    const float input[6] = {
         0.8f, -0.3f, 1.1f,
        -0.5f,  0.7f, 0.2f,
    };
    const float initial_weight[12] = {
         0.3f, -0.2f,  0.5f,
        -0.4f,  0.6f,  0.1f,
         0.2f,  0.4f, -0.3f,
        -0.1f, -0.5f,  0.7f,
    };
    const float initial_bias[4] = {0.05f, -0.1f, 0.2f, -0.05f};
    const float offset[4] = {0.15f, 0.05f, -0.2f, 0.1f};
    const int targets[4][2] = {
        {0, 3}, {2, 1}, {1, 2}, {3, 0},
    };
    const int ignored_targets[2] = {-100, -100};
    const char* trainables[2] = {
        "cuda.accum.weight", "cuda.accum.bias",
    };
    const float learning_rate = 0.02f;
    const float beta1 = 0.85f;
    const float beta2 = 0.95f;
    const float epsilon = 1.0e-6f;
    const float weight_decay = 0.03f;
    const float max_grad_norm = 0.05f;
    const size_t trainable_bytes =
        sizeof(initial_weight) + sizeof(initial_bias);
    float weight_after_step2[12];
    float bias_after_step2[4];
    float saved_weight_m[12], saved_weight_v[12];
    float saved_bias_m[4], saved_bias_v[4];
    float *weight_m = NULL, *weight_v = NULL;
    float *bias_m = NULL, *bias_v = NULL;
    volvoxai_cross_entropy_loss_t loss = {
        "main", "logits", targets[0], 2, INT_MIN, -1, 1.0f, 4.0f,
    };
    volvoxai_cross_entropy_metric_t metric = {0};
    int microbatches = 0;
    int applied = 0;

    CHECK(result != NULL);
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
    CHECK(add_tensor("cuda.accum.input", input_shape, 2, input));
    CHECK(add_tensor("cuda.accum.weight", weight_shape, 2, initial_weight));
    CHECK(add_tensor("cuda.accum.bias", channel_shape, 1, initial_bias));
    CHECK(add_tensor("cuda.accum.offset", channel_shape, 1, offset));
    CHECK(add_tensor("cuda.accum.linear", logits_shape, 2, NULL));
    CHECK(add_tensor("cuda.accum.relu", logits_shape, 2, NULL));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));
    Node* node = add_node("Linear", "cuda.accum.linear",
                          "{\"weight_layout\":\"dout_din\"}");
    CHECK(node && node_input(node, "input", "cuda.accum.input") == 0 &&
          node_input(node, "weight", "cuda.accum.weight") == 0 &&
          node_input(node, "bias", "cuda.accum.bias") == 0);
    node = add_node("ReLU", "cuda.accum.relu", NULL);
    CHECK(node && node_input(node, "input", "cuda.accum.linear") == 0);
    node = add_node("Add", "logits", NULL);
    CHECK(node && node_input(node, "a", "cuda.accum.relu") == 0 &&
          node_input(node, "b", "cuda.accum.offset") == 0);

    for (int optimizer_step = 1; optimizer_step <= 2; optimizer_step++) {
        for (int microbatch = 0; microbatch < 2; microbatch++) {
            int observation = (optimizer_step - 1) * 2 + microbatch;
#if defined(VOLVOXAI_CUDA_TESTING)
            uint64_t d2h_before = use_cuda
                ? cuda_test_device_to_host_bytes() : 0u;
            uint64_t weight_h2d_count_before = use_cuda
                ? cuda_test_weight_host_to_device_count() : 0u;
            uint64_t weight_h2d_bytes_before = use_cuda
                ? cuda_test_weight_host_to_device_bytes() : 0u;
            uint64_t weight_promotions_before = use_cuda
                ? cuda_test_weight_promotion_count() : 0u;
            uint64_t weight_promotion_bytes_before = use_cuda
                ? cuda_test_weight_promotion_bytes() : 0u;
#endif
            loss.targets = targets[observation];
            memset(&metric, 0, sizeof(metric));
            CHECK(volvoxai_engine_train_step_multi(
                &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
                learning_rate, beta1, beta2, epsilon, weight_decay,
                max_grad_norm, optimizer_step, 2, 0, 0,
                &result->losses[observation], &metric,
                &microbatches, &applied) == 0);
            CHECK(volvoxai_engine_last_training_backend() ==
                  (use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU));
            CHECK(microbatches == microbatch + 1);
            CHECK(applied == (microbatch == 1));
            CHECK(metric.examples == (microbatch + 1) * 2);
            CHECK(isfinite(result->losses[observation]) &&
                  result->losses[observation] > 0.0f);
            result->correct[observation] = metric.correct;
            result->examples[observation] = metric.examples;
#if defined(VOLVOXAI_CUDA_TESTING)
            if (use_cuda) {
                uint64_t d2h = cuda_test_device_to_host_bytes() - d2h_before;
                /* One active microbatch returns four loss scalars plus the
                 * finite-status word. Applying the window returns only status
                 * and clip scale; weights, logits, and full gradients remain
                 * device-resident. */
                uint64_t expected = 5u * sizeof(uint32_t);
                if (microbatch == 1)
                    expected += 2u * sizeof(uint32_t);
                CHECK(d2h == expected);
                /* Linear weight+bias are known weight roles at cold upload.
                 * This synthetic graph also stores its input and Add offset
                 * in the model weight file, so both generic operands are
                 * promoted after that forward. All four survive both cache
                 * transitions. Device AdamW leaves the trainables device-
                 * authoritative, so later micros upload no model weight. */
                CHECK(cuda_test_weight_host_to_device_count() -
                      weight_h2d_count_before ==
                      (optimizer_step == 1 && microbatch == 0 ? 2u : 0u));
                CHECK(cuda_test_weight_host_to_device_bytes() -
                      weight_h2d_bytes_before ==
                      (optimizer_step == 1 && microbatch == 0
                           ? trainable_bytes : 0u));
                CHECK(cuda_test_weight_promotion_count() -
                      weight_promotions_before ==
                      (optimizer_step == 1 && microbatch == 0 ? 2u : 0u));
                CHECK(cuda_test_weight_promotion_bytes() -
                      weight_promotion_bytes_before ==
                      (optimizer_step == 1 && microbatch == 0
                           ? sizeof(input) + sizeof(offset) : 0u));
            }
#endif
            if (microbatch == 0) {
                if (optimizer_step == 1) {
                    CHECK(!memcmp(t_find("cuda.accum.weight")->data,
                                  initial_weight, sizeof(initial_weight)));
                    CHECK(!memcmp(t_find("cuda.accum.bias")->data,
                                  initial_bias, sizeof(initial_bias)));
                }
            }
        }
    }
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t weights_d2h_before = use_cuda
        ? cuda_test_device_to_host_bytes() : 0u;
#endif
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.accum.weight", weight_after_step2, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.accum.bias", bias_after_step2, 4) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    if (use_cuda) {
        CHECK(cuda_test_device_to_host_bytes() - weights_d2h_before ==
              trainable_bytes);
        float repeated_weight[12], repeated_bias[4];
        CHECK(volvoxai_engine_copy_tensor_f32(
            "cuda.accum.weight", repeated_weight, 12) == 0);
        CHECK(volvoxai_engine_copy_tensor_f32(
            "cuda.accum.bias", repeated_bias, 4) == 0);
        CHECK(cuda_test_device_to_host_bytes() - weights_d2h_before ==
              trainable_bytes);
    }
#endif
    CHECK(cuda_accum_optimizer_any_changed(
        initial_weight, weight_after_step2, 12));
    CHECK(cuda_accum_optimizer_any_changed(
        initial_bias, bias_after_step2, 4));

    CHECK(volvoxai_engine_optimizer_state_buffers_f32(
        "cuda.accum.weight", 12, &weight_m, &weight_v) == 0);
    CHECK(volvoxai_engine_optimizer_state_buffers_f32(
        "cuda.accum.bias", 4, &bias_m, &bias_v) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t moments_d2h_before = use_cuda
        ? cuda_test_device_to_host_bytes() : 0u;
#endif
    if (use_cuda) {
        CHECK(cuda_training_optimizer_materialize(
            weight_m, weight_v, sizeof(saved_weight_m)) == 0);
        CHECK(cuda_training_optimizer_materialize(
            bias_m, bias_v, sizeof(saved_bias_m)) == 0);
    }
#if defined(VOLVOXAI_CUDA_TESTING)
    if (use_cuda)
        CHECK(cuda_test_device_to_host_bytes() - moments_d2h_before ==
              2u * trainable_bytes);
#endif
    memcpy(saved_weight_m, weight_m, sizeof(saved_weight_m));
    memcpy(saved_weight_v, weight_v, sizeof(saved_weight_v));
    memcpy(saved_bias_m, bias_m, sizeof(saved_bias_m));
    memcpy(saved_bias_v, bias_v, sizeof(saved_bias_v));

    /* A complete all-ignore window must close without applying AdamW weight
     * decay, mutating moments, or consuming any device-to-host payload. */
    loss.targets = ignored_targets;
    loss.ignore_index = -100;
    for (int microbatch = 0; microbatch < 2; microbatch++) {
        float ignored_loss = -1.0f;
#if defined(VOLVOXAI_CUDA_TESTING)
        uint64_t d2h_before = use_cuda
            ? cuda_test_device_to_host_bytes() : 0u;
#endif
        memset(&metric, 0, sizeof(metric));
        CHECK(volvoxai_engine_train_step_multi(
            &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
            learning_rate, beta1, beta2, epsilon, weight_decay,
            max_grad_norm, 3, 2, 0, 0, &ignored_loss, &metric,
            &microbatches, &applied) == 0);
        CHECK(microbatches == microbatch + 1 && !applied);
        CHECK(ignored_loss == 0.0f && metric.loss == 0.0f &&
              metric.correct == 0 && metric.examples == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
        if (use_cuda)
            CHECK(cuda_test_device_to_host_bytes() == d2h_before);
#endif
    }
    CHECK(!memcmp(weight_after_step2, t_find("cuda.accum.weight")->data,
                  sizeof(weight_after_step2)));
    CHECK(!memcmp(bias_after_step2, t_find("cuda.accum.bias")->data,
                  sizeof(bias_after_step2)));
    if (use_cuda) {
        CHECK(cuda_training_optimizer_materialize(
            weight_m, weight_v, sizeof(saved_weight_m)) == 0);
        CHECK(cuda_training_optimizer_materialize(
            bias_m, bias_v, sizeof(saved_bias_m)) == 0);
    }
    CHECK(!memcmp(saved_weight_m, weight_m, sizeof(saved_weight_m)));
    CHECK(!memcmp(saved_weight_v, weight_v, sizeof(saved_weight_v)));
    CHECK(!memcmp(saved_bias_m, bias_m, sizeof(saved_bias_m)));
    CHECK(!memcmp(saved_bias_v, bias_v, sizeof(saved_bias_v)));

    memcpy(result->weight, weight_after_step2, sizeof(result->weight));
    memcpy(result->bias, bias_after_step2, sizeof(result->bias));
    memcpy(result->weight_m, saved_weight_m, sizeof(result->weight_m));
    memcpy(result->weight_v, saved_weight_v, sizeof(result->weight_v));
    memcpy(result->bias_m, saved_bias_m, sizeof(result->bias_m));
    memcpy(result->bias_v, saved_bias_v, sizeof(result->bias_v));
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CPU) == 0);
    return finish_graph();
}

static int test_cuda_accum_optimizer_runtime_optional(void) {
    CudaAccumOptimizerResult cpu;
    CudaAccumOptimizerResult cuda;
    CHECK(run_cuda_accum_optimizer_case(0, &cpu) == 0);
    int cuda_result = run_cuda_accum_optimizer_case(1, &cuda);
    if (cuda_result == 1) {
        puts("native CUDA accumulation/AdamW integration skipped: no CUDA device");
        return 0;
    }
    CHECK(cuda_result == 0);
    CHECK(cuda_accum_optimizer_array_close(cpu.losses, cuda.losses, 4,
                                           2.0e-3f));
    for (int index = 0; index < 4; index++) {
        CHECK(cpu.correct[index] == cuda.correct[index]);
        CHECK(cpu.examples[index] == cuda.examples[index]);
    }
    CHECK(cuda_accum_optimizer_array_close(cpu.weight, cuda.weight, 12,
                                           2.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.bias, cuda.bias, 4,
                                           2.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.weight_m, cuda.weight_m, 12,
                                           3.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.weight_v, cuda.weight_v, 12,
                                           3.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.bias_m, cuda.bias_m, 4,
                                           3.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.bias_v, cuda.bias_v, 4,
                                           3.0e-4f));
    return 0;
}

static int run_cuda_optimizer_handoff_checkpoint_case(
        int use_cuda, CudaOptimizerHandoffResult* handoff) {
    const int input_shape[2] = {2, 3};
    const int weight_shape[2] = {4, 3};
    const int channel_shape[1] = {4};
    const int logits_shape[2] = {2, 4};
    const float input[6] = {
         0.8f, -0.3f, 1.1f,
        -0.5f,  0.7f, 0.2f,
    };
    const float initial_weight[12] = {
         0.3f, -0.2f,  0.5f,
        -0.4f,  0.6f,  0.1f,
         0.2f,  0.4f, -0.3f,
        -0.1f, -0.5f,  0.7f,
    };
    const float initial_bias[4] = {0.05f, -0.1f, 0.2f, -0.05f};
    const int targets[5][2] = {
        {0, 3}, {2, 1}, {1, 2}, {3, 0}, {0, 2},
    };
    const char* trainables[2] = {
        "cuda.checkpoint.weight", "cuda.checkpoint.bias",
    };
    const char* checkpoint_path =
        "/tmp/volvoxai-cuda-optimizer-checkpoint.safetensors";
    const float learning_rate = 0.02f;
    const float beta1 = 0.85f;
    const float beta2 = 0.95f;
    const float epsilon = 1.0e-6f;
    const float weight_decay = 0.03f;
    const size_t trainable_bytes =
        sizeof(initial_weight) + sizeof(initial_bias);
    float checkpoint_weight[12], checkpoint_bias[4];
    float checkpoint_weight_m[12], checkpoint_weight_v[12];
    float checkpoint_bias_m[4], checkpoint_bias_v[4];
    float uninterrupted_weight[12], uninterrupted_bias[4];
    float uninterrupted_weight_m[12], uninterrupted_weight_v[12];
    float uninterrupted_bias_m[4], uninterrupted_bias_v[4];
    float *weight_m = NULL, *weight_v = NULL;
    float *bias_m = NULL, *bias_v = NULL;
    volvoxai_cross_entropy_loss_t loss = {
        "main", "logits", targets[0], 2, INT_MIN, -1, 1.0f, 2.0f,
    };
    volvoxai_cross_entropy_metric_t metric = {0};
    int microbatches = 0;
    int applied = 0;

    CHECK(handoff != NULL);
    memset(handoff, 0, sizeof(*handoff));
    remove(checkpoint_path);
    begin_graph();
    g_use_vulkan = g_use_nnapi = g_use_opengl = g_use_metal = 0;
    g_use_cuda = use_cuda;
    if (use_cuda && cuda_init() != 0) {
        CHECK(finish_graph() == 0);
        return cuda_init_failure_is_unavailable() ? 1 : -1;
    }
    CHECK(volvoxai_engine_require_training_backend(
        use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU) == 0);
    CHECK(add_tensor("cuda.checkpoint.input", input_shape, 2, input));
    CHECK(add_tensor("cuda.checkpoint.weight", weight_shape, 2,
                     initial_weight));
    CHECK(add_tensor("cuda.checkpoint.bias", channel_shape, 1,
                     initial_bias));
    CHECK(add_tensor("logits", logits_shape, 2, NULL));
    Node* node = add_node("Linear", "logits",
                          "{\"weight_layout\":\"dout_din\"}");
    CHECK(node && node_input(node, "input", "cuda.checkpoint.input") == 0 &&
          node_input(node, "weight", "cuda.checkpoint.weight") == 0 &&
          node_input(node, "bias", "cuda.checkpoint.bias") == 0);

    /* Leave the CUDA moments device-dirty after two public AdamW steps. */
    for (int step = 1; step <= 2; step++) {
        loss.targets = targets[step - 1];
        memset(&metric, 0, sizeof(metric));
        CHECK(volvoxai_engine_train_step_multi(
            &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
            learning_rate, beta1, beta2, epsilon, weight_decay, 0.0f,
            step, 1, 0, 0, NULL, &metric, &microbatches, &applied) == 0);
        CHECK(microbatches == 1 && applied && metric.examples == 2);
        CHECK(volvoxai_engine_last_training_backend() ==
              (use_cuda ? VOLVOXAI_BACKEND_CUDA : VOLVOXAI_BACKEND_CPU));
    }

    /* Select the CPU path without finishing the graph.  In the CUDA case the
     * public CPU AdamW update must first materialize both device moment
     * buffers for both trainables, then make host moments authoritative. */
    g_use_cuda = 0;
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CPU) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t handoff_d2h_before = use_cuda
        ? cuda_test_device_to_host_bytes() : 0u;
#endif
    loss.targets = targets[2];
    memset(&metric, 0, sizeof(metric));
    CHECK(volvoxai_engine_train_step_multi(
        &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
        learning_rate, beta1, beta2, epsilon, weight_decay, 0.0f,
        3, 1, 0, 0, NULL, &metric, &microbatches, &applied) == 0);
    CHECK(microbatches == 1 && applied && metric.examples == 2);
    CHECK(volvoxai_engine_last_training_backend() == VOLVOXAI_BACKEND_CPU);
#if defined(VOLVOXAI_CUDA_TESTING)
    if (use_cuda)
        CHECK(cuda_test_device_to_host_bytes() - handoff_d2h_before ==
              3u * trainable_bytes);
#endif
    CHECK(volvoxai_engine_optimizer_state_buffers_f32(
        "cuda.checkpoint.weight", 12, &weight_m, &weight_v) == 0);
    CHECK(volvoxai_engine_optimizer_state_buffers_f32(
        "cuda.checkpoint.bias", 4, &bias_m, &bias_v) == 0);
    memcpy(handoff->weight, t_find("cuda.checkpoint.weight")->data,
           sizeof(handoff->weight));
    memcpy(handoff->bias, t_find("cuda.checkpoint.bias")->data,
           sizeof(handoff->bias));
    memcpy(handoff->weight_m, weight_m, sizeof(handoff->weight_m));
    memcpy(handoff->weight_v, weight_v, sizeof(handoff->weight_v));
    memcpy(handoff->bias_m, bias_m, sizeof(handoff->bias_m));
    memcpy(handoff->bias_v, bias_v, sizeof(handoff->bias_v));

    if (!use_cuda) return finish_graph();

    /* Make the next CUDA state device-dirty, then exercise independent public
     * checkpoint boundaries. Optimizer save materializes only Adam moments;
     * model copies materialize only the requested weights. */
    g_use_cuda = 1;
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CUDA) == 0);
    loss.targets = targets[3];
    memset(&metric, 0, sizeof(metric));
    CHECK(volvoxai_engine_train_step_multi(
        &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
        learning_rate, beta1, beta2, epsilon, weight_decay, 0.0f,
        4, 1, 0, 0, NULL, &metric, &microbatches, &applied) == 0);
    CHECK(microbatches == 1 && applied && metric.examples == 2);
    CHECK(volvoxai_engine_last_training_backend() == VOLVOXAI_BACKEND_CUDA);
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t save_d2h_before = cuda_test_device_to_host_bytes();
#endif
    CHECK(volvoxai_engine_save_optimizer_state(checkpoint_path, 4) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_device_to_host_bytes() - save_d2h_before ==
          2u * trainable_bytes);
#endif
#if defined(VOLVOXAI_CUDA_TESTING)
    uint64_t weight_save_d2h_before = cuda_test_device_to_host_bytes();
#endif
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.checkpoint.weight", checkpoint_weight, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.checkpoint.bias", checkpoint_bias, 4) == 0);
#if defined(VOLVOXAI_CUDA_TESTING)
    CHECK(cuda_test_device_to_host_bytes() - weight_save_d2h_before ==
          trainable_bytes);
#endif
    memcpy(checkpoint_weight_m, weight_m, sizeof(checkpoint_weight_m));
    memcpy(checkpoint_weight_v, weight_v, sizeof(checkpoint_weight_v));
    memcpy(checkpoint_bias_m, bias_m, sizeof(checkpoint_bias_m));
    memcpy(checkpoint_bias_v, bias_v, sizeof(checkpoint_bias_v));

    /* Record an uninterrupted step 5, then restore both public weights and
     * public optimizer checkpoint and replay exactly that step. */
    loss.targets = targets[4];
    memset(&metric, 0, sizeof(metric));
    CHECK(volvoxai_engine_train_step_multi(
        &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
        learning_rate, beta1, beta2, epsilon, weight_decay, 0.0f,
        5, 1, 0, 0, NULL, &metric, &microbatches, &applied) == 0);
    CHECK(microbatches == 1 && applied && metric.examples == 2);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.checkpoint.weight", uninterrupted_weight, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.checkpoint.bias", uninterrupted_bias, 4) == 0);
    CHECK(cuda_training_optimizer_materialize(
        weight_m, weight_v, sizeof(uninterrupted_weight_m)) == 0);
    CHECK(cuda_training_optimizer_materialize(
        bias_m, bias_v, sizeof(uninterrupted_bias_m)) == 0);
    memcpy(uninterrupted_weight_m, weight_m, sizeof(uninterrupted_weight_m));
    memcpy(uninterrupted_weight_v, weight_v, sizeof(uninterrupted_weight_v));
    memcpy(uninterrupted_bias_m, bias_m, sizeof(uninterrupted_bias_m));
    memcpy(uninterrupted_bias_v, bias_v, sizeof(uninterrupted_bias_v));

    CHECK(volvoxai_engine_set_tensor_f32(
        "cuda.checkpoint.weight", checkpoint_weight, 12) == 0);
    CHECK(volvoxai_engine_set_tensor_f32(
        "cuda.checkpoint.bias", checkpoint_bias, 4) == 0);
    long loaded_step = -1;
    CHECK(volvoxai_engine_load_optimizer_state(
        checkpoint_path, &loaded_step) == 0);
    CHECK(loaded_step == 4);
    CHECK(volvoxai_engine_optimizer_state_buffers_f32(
        "cuda.checkpoint.weight", 12, &weight_m, &weight_v) == 0);
    CHECK(volvoxai_engine_optimizer_state_buffers_f32(
        "cuda.checkpoint.bias", 4, &bias_m, &bias_v) == 0);
    CHECK(!memcmp(checkpoint_weight_m, weight_m, sizeof(checkpoint_weight_m)));
    CHECK(!memcmp(checkpoint_weight_v, weight_v, sizeof(checkpoint_weight_v)));
    CHECK(!memcmp(checkpoint_bias_m, bias_m, sizeof(checkpoint_bias_m)));
    CHECK(!memcmp(checkpoint_bias_v, bias_v, sizeof(checkpoint_bias_v)));

    loss.targets = targets[4];
    memset(&metric, 0, sizeof(metric));
    CHECK(volvoxai_engine_train_step_multi(
        &loss, 1, trainables, 2, VOLVOXAI_TENSOR_UPDATE_ADAMW,
        learning_rate, beta1, beta2, epsilon, weight_decay, 0.0f,
        5, 1, 0, 0, NULL, &metric, &microbatches, &applied) == 0);
    CHECK(microbatches == 1 && applied && metric.examples == 2);
    float replayed_weight[12], replayed_bias[4];
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.checkpoint.weight", replayed_weight, 12) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32(
        "cuda.checkpoint.bias", replayed_bias, 4) == 0);
    CHECK(cuda_accum_optimizer_array_close(
        replayed_weight, uninterrupted_weight, 12, 2.0e-6f));
    CHECK(cuda_accum_optimizer_array_close(
        replayed_bias, uninterrupted_bias, 4, 2.0e-6f));
    CHECK(cuda_training_optimizer_materialize(
        weight_m, weight_v, sizeof(uninterrupted_weight_m)) == 0);
    CHECK(cuda_training_optimizer_materialize(
        bias_m, bias_v, sizeof(uninterrupted_bias_m)) == 0);
    CHECK(cuda_accum_optimizer_array_close(
        weight_m, uninterrupted_weight_m, 12, 2.0e-6f));
    CHECK(cuda_accum_optimizer_array_close(
        weight_v, uninterrupted_weight_v, 12, 2.0e-6f));
    CHECK(cuda_accum_optimizer_array_close(
        bias_m, uninterrupted_bias_m, 4, 2.0e-6f));
    CHECK(cuda_accum_optimizer_array_close(
        bias_v, uninterrupted_bias_v, 4, 2.0e-6f));
    CHECK(volvoxai_engine_require_training_backend(VOLVOXAI_BACKEND_CPU) == 0);
    CHECK(finish_graph() == 0);
    CHECK(remove(checkpoint_path) == 0);
    return 0;
}

static int test_cuda_optimizer_handoff_checkpoint_runtime_optional(void) {
    CudaOptimizerHandoffResult cpu;
    CudaOptimizerHandoffResult cuda;
    CHECK(run_cuda_optimizer_handoff_checkpoint_case(0, &cpu) == 0);
    int cuda_result = run_cuda_optimizer_handoff_checkpoint_case(1, &cuda);
    if (cuda_result == 1) {
        puts("native CUDA optimizer checkpoint integration skipped: no CUDA device");
        return 0;
    }
    CHECK(cuda_result == 0);
    CHECK(cuda_accum_optimizer_array_close(cpu.weight, cuda.weight, 12,
                                           3.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.bias, cuda.bias, 4,
                                           3.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.weight_m, cuda.weight_m, 12,
                                           4.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.weight_v, cuda.weight_v, 12,
                                           4.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.bias_m, cuda.bias_m, 4,
                                           4.0e-4f));
    CHECK(cuda_accum_optimizer_array_close(cpu.bias_v, cuda.bias_v, 4,
                                           4.0e-4f));
    return 0;
}

#endif

#endif
