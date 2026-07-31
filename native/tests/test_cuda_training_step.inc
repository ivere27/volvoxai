/* Focused numeric coverage for the CUDA device-resident training window.
 * Textually included by test_cuda_training.c after CHECK and close_f32. */

#ifndef VOLVOXAI_TEST_CUDA_TRAINING_STEP_INC
#define VOLVOXAI_TEST_CUDA_TRAINING_STEP_INC

static int cuda_training_step_close(float actual, float expected,
                                    float tolerance) {
    float scale = fmaxf(1.0f, fmaxf(fabsf(actual), fabsf(expected)));
    return fabsf(actual - expected) <= tolerance * scale;
}

static void cuda_training_step_adamw_reference(
        float* weight, float* first_moment, float* second_moment,
        const float* gradient, int elements, float learning_rate,
        float beta1, float beta2, float epsilon, float weight_decay,
        long step) {
    float bias_correction1 = 1.0f - powf(beta1, (float)step);
    float bias_correction2 = 1.0f - powf(beta2, (float)step);
    if (bias_correction1 <= 0.0f) bias_correction1 = 1.0f;
    if (bias_correction2 <= 0.0f) bias_correction2 = 1.0f;
    for (int index = 0; index < elements; index++) {
        float value = gradient[index];
        first_moment[index] = beta1 * first_moment[index] +
            (1.0f - beta1) * value;
        second_moment[index] = beta2 * second_moment[index] +
            (1.0f - beta2) * value * value;
        float corrected_m = first_moment[index] / bias_correction1;
        float corrected_v = second_moment[index] / bias_correction2;
        if (weight_decay != 0.0f)
            weight[index] -= learning_rate * weight_decay * weight[index];
        weight[index] -= learning_rate * corrected_m /
            (sqrtf(corrected_v) + epsilon);
    }
}

int test_cuda_training_step_window(void) {
    enum { CE_ROWS = 4, CE_CLASSES = 3, CE_ELEMENTS = 12 };
    float ce_logits[CE_ELEMENTS] = {
        1.25f, -0.5f, 0.25f,
        0.2f, -0.3f, 1.1f,
        -0.4f, 0.8f, 0.1f,
        0.6f, 0.5f, -0.2f,
    };
    float ce_gradient[CE_ELEMENTS];
    const int ce_targets[2] = {2, -7};
    const int zero_weight_target[1] = {0};
    float sgd_weight_a[2] = {10.0f, -2.0f};
    float sgd_weight_b[1] = {1.0f};
    float sgd_gradient_a[2] = {1.0f, 1.0f};
    float sgd_gradient_b[1] = {5.0f};
    void* sgd_weights[2] = {sgd_weight_a, sgd_weight_b};
    void* sgd_gradients[2] = {sgd_gradient_a, sgd_gradient_b};
    const size_t sgd_bytes[2] = {
        sizeof(sgd_weight_a), sizeof(sgd_weight_b),
    };
    CudaTrainingWindow* window = cuda_training_window_create(
        sgd_weights, sgd_bytes, 2);
    CudaTrainingLossResult losses[2];
    uint32_t status = 0;
    CHECK(window != NULL);
    CHECK(cuda_training_begin() == 0);

    /* The gradient seed starts device-side at zero. The first loss uses the
     * per-batch mapping: two targets each own a two-row span, with -1 choosing
     * the final row. The second target is ignored. Repeating the same loss
     * exercises additive metric and gradient behavior. */
    void* ce_gradient_hosts[1] = {ce_gradient};
    const size_t ce_gradient_bytes[1] = {sizeof(ce_gradient)};
    CHECK(cuda_training_step_zero_gradients(
        ce_gradient_hosts, ce_gradient_bytes, 1) == 0);
    CHECK(cuda_training_window_reset_losses(window, 2) == 0);
    CHECK(cuda_training_window_seed_cross_entropy(
        window, 0, ce_logits, sizeof(ce_logits), ce_targets, 2,
        ce_gradient, CE_ROWS, CE_CLASSES, 2, -1, -7, 0.5f, 2.0f) == 0);
    CHECK(cuda_training_window_seed_cross_entropy(
        window, 0, ce_logits, sizeof(ce_logits), ce_targets, 2,
        ce_gradient, CE_ROWS, CE_CLASSES, 2, -1, -7, 0.5f, 2.0f) == 0);

    /* A zero-weight loss still reports its active example but must not alter
     * loss, correctness, or the already seeded gradient. row_index=0 covers
     * the single-target absolute-row mapping. */
    CHECK(cuda_training_window_seed_cross_entropy(
        window, 1, ce_logits, sizeof(ce_logits), zero_weight_target, 1,
        ce_gradient, CE_ROWS, CE_CLASSES, 0, 0, -7, 0.0f, 0.0f) == 0);
    CHECK(cuda_training_window_read_losses(window, losses, 2) == 0);
    CHECK(losses[0].status == 0u);
    CHECK(losses[0].correct == 2u);
    CHECK(losses[0].examples == 2u);
    CHECK(losses[1].status == 0u);
    CHECK(losses[1].loss == 0.0f);
    CHECK(losses[1].correct == 0u);
    CHECK(losses[1].examples == 1u);
    CHECK(cuda_training_sync(ce_gradient, sizeof(ce_gradient)) == 0);

    float maximum = ce_logits[CE_CLASSES];
    for (int column = 1; column < CE_CLASSES; column++) {
        float value = ce_logits[CE_CLASSES + column];
        if (value > maximum) maximum = value;
    }
    float exponential_sum = 0.0f;
    for (int column = 0; column < CE_CLASSES; column++)
        exponential_sum += expf(ce_logits[CE_CLASSES + column] - maximum);
    float expected_loss = 0.5f *
        (maximum + logf(exponential_sum) - ce_logits[CE_CLASSES + 2]);
    CHECK(cuda_training_step_close(losses[0].loss, expected_loss, 2.0e-3f));
    for (int row = 0; row < CE_ROWS; row++) {
        for (int column = 0; column < CE_CLASSES; column++) {
            float expected = 0.0f;
            if (row == 1) {
                float probability =
                    expf(ce_logits[CE_CLASSES + column] - maximum) /
                    exponential_sum;
                expected = 0.5f *
                    (probability - (column == 2 ? 1.0f : 0.0f));
            }
            CHECK(cuda_training_step_close(
                ce_gradient[row * CE_CLASSES + column], expected, 2.0e-3f));
        }
    }

    /* Commit one finite microbatch, reject a NaN transactionally, then commit
     * another finite microbatch. The retained accumulators become [3,4] and
     * [12], whose single global norm is 13 and clip scale is exactly 0.5. */
    cuda_graph_mark_host(sgd_gradient_a, sizeof(sgd_gradient_a), 0);
    cuda_graph_mark_host(sgd_gradient_b, sizeof(sgd_gradient_b), 0);
    CHECK(cuda_training_window_commit(
        window, sgd_gradients, sgd_bytes, 2, &status) == 0);
    CHECK(status == 0u);

    sgd_gradient_a[0] = NAN;
    sgd_gradient_a[1] = 100.0f;
    sgd_gradient_b[0] = 100.0f;
    cuda_graph_mark_host(sgd_gradient_a, sizeof(sgd_gradient_a), 0);
    cuda_graph_mark_host(sgd_gradient_b, sizeof(sgd_gradient_b), 0);
    CHECK(cuda_training_window_commit(
        window, sgd_gradients, sgd_bytes, 2, &status) == 1);
    CHECK(status != 0u);

    sgd_gradient_a[0] = 2.0f;
    sgd_gradient_a[1] = 3.0f;
    sgd_gradient_b[0] = 7.0f;
    cuda_graph_mark_host(sgd_gradient_a, sizeof(sgd_gradient_a), 0);
    cuda_graph_mark_host(sgd_gradient_b, sizeof(sgd_gradient_b), 0);
    CHECK(cuda_training_window_commit(
        window, sgd_gradients, sgd_bytes, 2, &status) == 0);
    CHECK(status == 0u);
    uint64_t sgd_apply_sync_before =
        cuda_test_training_step_stream_sync_count();
    uint64_t sgd_apply_d2h_before = cuda_test_device_to_host_bytes();
    CHECK(cuda_training_window_apply(
        window, sgd_weights, sgd_bytes, NULL, NULL, 2, 2,
        0.1f, 0.9f, 0.999f, 1.0e-8f, 0.2f, 6.5f, 1,
        &status) == 0);
    CHECK(status == 0u);
    CHECK(cuda_test_training_step_stream_sync_count() -
          sgd_apply_sync_before == 2u);
    /* Apply returns only finite status and clip scale. Weights remain
     * device-authoritative until the explicit public-mirror boundary below. */
    CHECK(cuda_test_device_to_host_bytes() - sgd_apply_d2h_before ==
          2u * sizeof(uint32_t));
    uint64_t sgd_weight_d2h_before = cuda_test_device_to_host_bytes();
    CHECK(cuda_graph_sync_host(
        sgd_weight_a, sizeof(sgd_weight_a), 1));
    CHECK(cuda_graph_sync_host(
        sgd_weight_b, sizeof(sgd_weight_b), 1));
    CHECK(cuda_test_device_to_host_bytes() - sgd_weight_d2h_before ==
          sizeof(sgd_weight_a) + sizeof(sgd_weight_b));
    CHECK(cuda_graph_sync_host(
        sgd_weight_a, sizeof(sgd_weight_a), 1));
    CHECK(cuda_graph_sync_host(
        sgd_weight_b, sizeof(sgd_weight_b), 1));
    CHECK(cuda_test_device_to_host_bytes() - sgd_weight_d2h_before ==
          sizeof(sgd_weight_a) + sizeof(sgd_weight_b));
    CHECK(cuda_training_step_close(sgd_weight_a[0], 9.65f, 2.0e-6f));
    CHECK(cuda_training_step_close(sgd_weight_a[1], -2.16f, 2.0e-6f));
    CHECK(cuda_training_step_close(sgd_weight_b[0], 0.38f, 2.0e-6f));
    cuda_training_end();
    cuda_training_window_destroy(window);

    /* AdamW moments stay device-resident across fresh gradient windows. An
     * explicit checkpoint-style materialization validates the public mirror. */
    float adam_weight[2] = {1.0f, -2.0f};
    float adam_first[2] = {0.0f, 0.0f};
    float adam_second[2] = {0.0f, 0.0f};
    float expected_weight[2] = {1.0f, -2.0f};
    float expected_first[2] = {0.0f, 0.0f};
    float expected_second[2] = {0.0f, 0.0f};
    const float adam_updates[2][2] = {
        {0.5f, -1.0f},
        {1.5f, 0.25f},
    };
    void* adam_weights[1] = {adam_weight};
    void* adam_gradients[1];
    const size_t adam_bytes[1] = {sizeof(adam_weight)};
    float* adam_first_hosts[1] = {adam_first};
    float* adam_second_hosts[1] = {adam_second};
    const float adam_learning_rate = 0.01f;
    const float adam_beta1 = 0.9f;
    const float adam_beta2 = 0.99f;
    const float adam_epsilon = 1.0e-8f;
    const float adam_weight_decay = 0.1f;

    for (int optimizer_step = 1; optimizer_step <= 2; optimizer_step++) {
        float adam_gradient[2] = {
            adam_updates[optimizer_step - 1][0],
            adam_updates[optimizer_step - 1][1],
        };
        adam_gradients[0] = adam_gradient;
        window = cuda_training_window_create(adam_weights, adam_bytes, 1);
        CHECK(window != NULL);
        CHECK(cuda_training_begin() == 0);
        cuda_graph_mark_host(adam_gradient, sizeof(adam_gradient), 0);
        CHECK(cuda_training_window_commit(
            window, adam_gradients, adam_bytes, 1, &status) == 0);
        CHECK(status == 0u);
        uint64_t weight_uploads_before_apply =
            cuda_test_weight_host_to_device_count();
        uint64_t apply_d2h_before = cuda_test_device_to_host_bytes();
        CHECK(cuda_training_window_apply(
            window, adam_weights, adam_bytes, adam_first_hosts,
            adam_second_hosts, 1, 3, adam_learning_rate, adam_beta1,
            adam_beta2, adam_epsilon, adam_weight_decay, 0.0f,
            optimizer_step, &status) == 0);
        CHECK(status == 0u);
        CHECK(cuda_test_weight_host_to_device_count() -
              weight_uploads_before_apply ==
              (optimizer_step == 1 ? 1u : 0u));
        CHECK(cuda_test_device_to_host_bytes() - apply_d2h_before ==
              2u * sizeof(uint32_t));
        CHECK(cuda_graph_sync_host(
            adam_weight, sizeof(adam_weight), 1));
        CHECK(cuda_test_device_to_host_bytes() - apply_d2h_before ==
              2u * sizeof(uint32_t) + sizeof(adam_weight));
        CHECK(cuda_graph_sync_host(
            adam_weight, sizeof(adam_weight), 1));
        CHECK(cuda_test_device_to_host_bytes() - apply_d2h_before ==
              2u * sizeof(uint32_t) + sizeof(adam_weight));
        CHECK(cuda_training_optimizer_materialize(
            adam_first, adam_second, sizeof(adam_first)) == 0);
        cuda_training_step_adamw_reference(
            expected_weight, expected_first, expected_second,
            adam_updates[optimizer_step - 1], 2, adam_learning_rate,
            adam_beta1, adam_beta2, adam_epsilon, adam_weight_decay,
            optimizer_step);
        for (int index = 0; index < 2; index++) {
            CHECK(cuda_training_step_close(
                adam_weight[index], expected_weight[index], 3.0e-6f));
            CHECK(cuda_training_step_close(
                adam_first[index], expected_first[index], 3.0e-6f));
            CHECK(cuda_training_step_close(
                adam_second[index], expected_second[index], 3.0e-6f));
        }
        cuda_training_end();
        cuda_training_window_destroy(window);
    }

    /* A retained clean weight feeds the next graph without another H2D copy.
     * Explicit host invalidation then uploads exactly the changed input and
     * weight, proving residency does not weaken host/device coherence. */
    {
        float residency_input[2] = {2.0f, -1.0f};
        float residency_output[1] = {0.0f};
        uint64_t uploads = cuda_test_host_to_device_count();
        cuda_graph_begin_forward();
        CHECK(cuda_graph_matmul_f32(
            residency_input, adam_weight, NULL, residency_output,
            1, 2, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(
            residency_output, sizeof(residency_output), 0));
        CHECK(cuda_test_host_to_device_count() - uploads == 1u);

        residency_input[0] = -3.0f;
        adam_weight[0] = 4.0f;
        cuda_graph_mark_host(
            residency_input, sizeof(residency_input), 0);
        cuda_graph_mark_host(adam_weight, sizeof(adam_weight), 1);
        uploads = cuda_test_host_to_device_count();
        cuda_graph_begin_forward();
        CHECK(cuda_graph_matmul_f32(
            residency_input, adam_weight, NULL, residency_output,
            1, 2, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(
            residency_output, sizeof(residency_output), 0));
        CHECK(cuda_test_host_to_device_count() - uploads == 2u);
        CHECK(cuda_training_step_close(
            residency_output[0],
            residency_input[0] * adam_weight[0] +
                residency_input[1] * adam_weight[1],
            2.0e-6f));
    }
    cuda_training_optimizer_forget(adam_first, adam_second);
    CHECK(cuda_test_caller_context_is_clear());
    return 0;
}

#endif
