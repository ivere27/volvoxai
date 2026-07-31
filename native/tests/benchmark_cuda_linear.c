#define _POSIX_C_SOURCE 200809L

#include "cuda_engine.h"
#include "cuda_test_engine_scope.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {
    CUDA_LINEAR_SAMPLES = 9,
    CUDA_LINEAR_BATCH_ITERATIONS = 16,
    CUDA_LINEAR_WARMUP_ITERATIONS = 3,
    CUDA_LINEAR_VALIDATION_SAMPLES = 17,
};

#if VOLVOXAI_CUDA_FAST_FP32
#define CUDA_LINEAR_FP32_CONTRACT "fast_fma"
#else
#define CUDA_LINEAR_FP32_CONTRACT "strict_no_fma"
#endif

typedef struct {
    const char* label;
    uint32_t rows;
    uint32_t d_in;
    uint32_t d_out;
} CudaLinearCase;

typedef struct {
    float* input;
    float* weight;
    float* bias;
    float* output;
    float* grad_output;
    float* grad_input;
    float* grad_weight;
    float* grad_bias;
} CudaLinearBuffers;

static double elapsed_us(const struct timespec* start,
                         const struct timespec* end) {
    return (double)(end->tv_sec - start->tv_sec) * 1000000.0 +
        (double)(end->tv_nsec - start->tv_nsec) / 1000.0;
}

static double median(double values[CUDA_LINEAR_SAMPLES]) {
    for (int index = 1; index < CUDA_LINEAR_SAMPLES; index++) {
        double value = values[index];
        int destination = index;
        while (destination > 0 && values[destination - 1] > value) {
            values[destination] = values[destination - 1];
            destination--;
        }
        values[destination] = value;
    }
    return values[CUDA_LINEAR_SAMPLES / 2];
}

static float initialized_value(size_t index, uint32_t salt, float scale) {
    uint32_t bits = (uint32_t)index * 1664525u + salt * 1013904223u;
    int32_t centered = (int32_t)(bits >> 24u) - 128;
    return (float)centered * scale;
}

static float strict_multiply_add(float accumulator, float a, float b) {
    volatile float product = a * b;
    volatile float result = accumulator + product;
    return result;
}

static int close_enough(float actual, float expected, float* max_error) {
    float error = fabsf(actual - expected);
    float tolerance = 1.0e-3f + fabsf(expected) * 1.0e-4f;
    if (error > *max_error) *max_error = error;
    return isfinite(actual) && error <= tolerance;
}

static int allocate_buffers(const CudaLinearCase* test,
                            CudaLinearBuffers* buffers) {
    size_t input_count = (size_t)test->rows * test->d_in;
    size_t weight_count = (size_t)test->d_in * test->d_out;
    size_t output_count = (size_t)test->rows * test->d_out;
    buffers->input = (float*)malloc(input_count * sizeof(float));
    buffers->weight = (float*)malloc(weight_count * sizeof(float));
    buffers->bias = (float*)malloc((size_t)test->d_out * sizeof(float));
    buffers->output = (float*)calloc(output_count, sizeof(float));
    buffers->grad_output = (float*)malloc(output_count * sizeof(float));
    buffers->grad_input = (float*)calloc(input_count, sizeof(float));
    buffers->grad_weight = (float*)calloc(weight_count, sizeof(float));
    buffers->grad_bias = (float*)calloc((size_t)test->d_out, sizeof(float));
    if (!buffers->input || !buffers->weight || !buffers->bias ||
        !buffers->output || !buffers->grad_output || !buffers->grad_input ||
        !buffers->grad_weight || !buffers->grad_bias) return 0;

    for (size_t index = 0; index < input_count; index++)
        buffers->input[index] = initialized_value(index, 3u, 1.0f / 1024.0f);
    for (size_t index = 0; index < weight_count; index++)
        buffers->weight[index] = initialized_value(index, 7u, 1.0f / 2048.0f);
    for (uint32_t index = 0; index < test->d_out; index++)
        buffers->bias[index] = initialized_value(index, 11u, 1.0f / 512.0f);
    for (size_t index = 0; index < output_count; index++)
        buffers->grad_output[index] =
            initialized_value(index, 13u, 1.0f / 4096.0f);
    return 1;
}

static void free_buffers(CudaLinearBuffers* buffers) {
    free(buffers->grad_bias);
    free(buffers->grad_weight);
    free(buffers->grad_input);
    free(buffers->grad_output);
    free(buffers->output);
    free(buffers->bias);
    free(buffers->weight);
    free(buffers->input);
}

static int launch_forward_batch(const CudaLinearCase* test,
                                const CudaLinearBuffers* buffers,
                                int iterations) {
    int ok = 1;
    cuda_graph_begin_forward();
    for (int iteration = 0; iteration < iterations; iteration++) {
        if (!cuda_graph_matmul_layout_f32(
                buffers->input, buffers->weight, buffers->bias,
                buffers->output, (int)test->rows, (int)test->d_in,
                (int)test->d_out, 0)) {
            ok = 0;
            break;
        }
    }
    if (cuda_graph_end_forward() != 0) ok = 0;
    return ok;
}

static int validate_forward(const CudaLinearCase* test,
                            const CudaLinearBuffers* buffers,
                            float* max_error) {
    size_t output_count = (size_t)test->rows * test->d_out;
    if (!cuda_graph_sync_host(buffers->output,
                              output_count * sizeof(float), 0)) return 0;
    *max_error = 0.0f;
    for (uint32_t sample = 0; sample < CUDA_LINEAR_VALIDATION_SAMPLES;
         sample++) {
        size_t output_index =
            ((size_t)sample * 104729u + test->d_in + test->d_out) %
            output_count;
        uint32_t row = (uint32_t)(output_index / test->d_out);
        uint32_t column = (uint32_t)(output_index % test->d_out);
        float expected = buffers->bias[column];
        for (uint32_t inner = 0; inner < test->d_in; inner++) {
            expected = strict_multiply_add(expected,
                buffers->input[(size_t)row * test->d_in + inner],
                buffers->weight[(size_t)inner * test->d_out + column]);
        }
        if (!close_enough(buffers->output[output_index], expected, max_error))
            return 0;
    }
    return 1;
}

static int benchmark_forward(const CudaLinearCase* test,
                             const CudaLinearBuffers* buffers) {
    double samples[CUDA_LINEAR_SAMPLES];
    float max_error;
    uint64_t h2d_before;
    uint64_t launches_before;
    uint64_t launches_after;

    cuda_graph_reset();
    if (!launch_forward_batch(test, buffers, CUDA_LINEAR_WARMUP_ITERATIONS) ||
        !validate_forward(test, buffers, &max_error)) {
        fprintf(stderr, "CUDA Linear forward validation failed: %s\n",
                test->label);
        return 0;
    }
    h2d_before = cuda_test_host_to_device_count();
    launches_before = cuda_test_launch_count();
    for (int sample = 0; sample < CUDA_LINEAR_SAMPLES; sample++) {
        struct timespec start;
        struct timespec end;
        if (clock_gettime(CLOCK_MONOTONIC, &start) != 0 ||
            !launch_forward_batch(test, buffers,
                                  CUDA_LINEAR_BATCH_ITERATIONS) ||
            clock_gettime(CLOCK_MONOTONIC, &end) != 0) {
            fprintf(stderr, "CUDA Linear forward timing failed: %s\n",
                    test->label);
            return 0;
        }
        samples[sample] = elapsed_us(&start, &end) /
            (double)CUDA_LINEAR_BATCH_ITERATIONS;
    }
    launches_after = cuda_test_launch_count();
    if (cuda_test_host_to_device_count() != h2d_before ||
        launches_after - launches_before !=
            (uint64_t)CUDA_LINEAR_SAMPLES * CUDA_LINEAR_BATCH_ITERATIONS) {
        fprintf(stderr, "CUDA Linear forward residency check failed: %s\n",
                test->label);
        return 0;
    }
    {
        double median_iteration_us = median(samples);
        double flops = 2.0 * (double)test->rows * test->d_in * test->d_out;
        printf("cuda_linear direction=forward case=%s m=%u k=%u n=%u "
               "launches_per_iteration=1 median_us_iteration=%.3f "
               "median_us_per_launch=%.3f gflops=%.3f sync=stream "
               "validation_max_abs=%.9g\n",
               test->label, test->rows, test->d_in, test->d_out,
               median_iteration_us, median_iteration_us,
               flops / (median_iteration_us * 1000.0),
               (double)max_error);
    }
    return 1;
}

static int dispatch_backward(const CudaLinearCase* test,
                             void* const hosts[7], const size_t bytes[7],
                             const unsigned char access[7],
                             const unsigned char is_weight[7]) {
    uint32_t input_groups_x = (test->d_in + 15u) / 16u;
    uint32_t input_groups_y = (test->rows + 15u) / 16u;
    uint32_t weight_groups_x = (test->d_out + 15u) / 16u;
    uint32_t weight_groups_y = (test->d_in + 15u) / 16u;
    uint32_t bias_groups_x = (test->d_out + 63u) / 64u;
    return cuda_training_dispatch("matMulBackward", "input_main", hosts,
               bytes, access, is_weight, 7, input_groups_x, input_groups_y,
               1u) == 0 &&
        cuda_training_dispatch("matMulBackward", "weight_main", hosts,
               bytes, access, is_weight, 7, weight_groups_x, weight_groups_y,
               1u) == 0 &&
        cuda_training_dispatch("matMulBackward", "bias_main", hosts,
               bytes, access, is_weight, 7, bias_groups_x, 1u, 1u) == 0;
}

static int validate_backward(const CudaLinearCase* test,
                             const CudaLinearBuffers* buffers,
                             float* max_error) {
    size_t input_count = (size_t)test->rows * test->d_in;
    size_t weight_count = (size_t)test->d_in * test->d_out;
    size_t bias_count = test->d_out;
    if (cuda_training_sync(buffers->grad_input,
                           input_count * sizeof(float)) != 0 ||
        cuda_training_sync(buffers->grad_weight,
                           weight_count * sizeof(float)) != 0 ||
        cuda_training_sync(buffers->grad_bias,
                           bias_count * sizeof(float)) != 0) return 0;
    *max_error = 0.0f;
    for (uint32_t sample = 0; sample < CUDA_LINEAR_VALIDATION_SAMPLES;
         sample++) {
        size_t index = ((size_t)sample * 65537u + test->d_out) % input_count;
        uint32_t row = (uint32_t)(index / test->d_in);
        uint32_t inner = (uint32_t)(index % test->d_in);
        float contribution = 0.0f;
        float expected = 0.0f;
        for (uint32_t column = 0; column < test->d_out; column++) {
            contribution = strict_multiply_add(contribution,
                buffers->grad_output[(size_t)row * test->d_out + column],
                buffers->weight[(size_t)inner * test->d_out + column]);
        }
        for (int warmup = 0; warmup < CUDA_LINEAR_WARMUP_ITERATIONS; warmup++)
            expected += contribution;
        if (!close_enough(buffers->grad_input[index], expected, max_error))
            return 0;
    }
    for (uint32_t sample = 0; sample < CUDA_LINEAR_VALIDATION_SAMPLES;
         sample++) {
        size_t index = ((size_t)sample * 65539u + test->rows) % weight_count;
        uint32_t inner = (uint32_t)(index / test->d_out);
        uint32_t column = (uint32_t)(index % test->d_out);
        float contribution = 0.0f;
        float expected = 0.0f;
        for (uint32_t row = 0; row < test->rows; row++) {
            contribution = strict_multiply_add(contribution,
                buffers->input[(size_t)row * test->d_in + inner],
                buffers->grad_output[(size_t)row * test->d_out + column]);
        }
        for (int warmup = 0; warmup < CUDA_LINEAR_WARMUP_ITERATIONS; warmup++)
            expected += contribution;
        if (!close_enough(buffers->grad_weight[index], expected, max_error))
            return 0;
    }
    for (uint32_t sample = 0; sample < CUDA_LINEAR_VALIDATION_SAMPLES;
         sample++) {
        uint32_t column =
            (sample * 257u + test->d_in) % test->d_out;
        float contribution = 0.0f;
        float expected = 0.0f;
        for (uint32_t row = 0; row < test->rows; row++)
            contribution +=
                buffers->grad_output[(size_t)row * test->d_out + column];
        for (int warmup = 0; warmup < CUDA_LINEAR_WARMUP_ITERATIONS; warmup++)
            expected += contribution;
        if (!close_enough(buffers->grad_bias[column], expected, max_error))
            return 0;
    }
    return 1;
}

static int benchmark_backward(const CudaLinearCase* test,
                              const CudaLinearBuffers* buffers) {
    size_t input_count = (size_t)test->rows * test->d_in;
    size_t weight_count = (size_t)test->d_in * test->d_out;
    size_t output_count = (size_t)test->rows * test->d_out;
    uint32_t params[8] = {
        test->rows, test->d_in, test->d_out, 1u, 1u, 1u, 0u, 0u,
    };
    void* hosts[7] = {
        buffers->input, buffers->weight, buffers->grad_output,
        buffers->grad_input, buffers->grad_weight, buffers->grad_bias, params,
    };
    size_t bytes[7] = {
        input_count * sizeof(float), weight_count * sizeof(float),
        output_count * sizeof(float), input_count * sizeof(float),
        weight_count * sizeof(float), (size_t)test->d_out * sizeof(float),
        sizeof(params),
    };
    const unsigned char access[7] = {1u, 1u, 1u, 3u, 3u, 3u, 1u};
    const unsigned char is_weight[7] = {0u, 1u, 0u, 0u, 0u, 0u, 0u};
    uint32_t input_groups_x = (test->d_in + 15u) / 16u;
    uint32_t input_groups_y = (test->rows + 15u) / 16u;
    uint32_t weight_groups_x = (test->d_out + 15u) / 16u;
    uint32_t weight_groups_y = (test->d_in + 15u) / 16u;
    uint32_t bias_groups_x = (test->d_out + 63u) / 64u;
    double samples[CUDA_LINEAR_SAMPLES];
    float max_error;
    uint64_t h2d_before;
    uint64_t launches_before;
    uint64_t launches_after;
    int training_active = 0;
    int ok = 0;

    cuda_graph_reset();
    if (!cuda_training_supports("matMulBackward", "input_main", bytes, 7,
                                input_groups_x, input_groups_y, 1u) ||
        !cuda_training_supports("matMulBackward", "weight_main", bytes, 7,
                                weight_groups_x, weight_groups_y, 1u) ||
        !cuda_training_supports("matMulBackward", "bias_main", bytes, 7,
                                bias_groups_x, 1u, 1u) ||
        cuda_training_begin() != 0) {
        fprintf(stderr, "CUDA Linear backward setup failed: %s\n",
                test->label);
        goto done;
    }
    training_active = 1;
    for (int warmup = 0; warmup < CUDA_LINEAR_WARMUP_ITERATIONS; warmup++) {
        if (!dispatch_backward(test, hosts, bytes, access, is_weight)) {
            fprintf(stderr, "CUDA Linear backward warmup failed: %s\n",
                    test->label);
            goto done;
        }
    }
    if (!validate_backward(test, buffers, &max_error)) {
        fprintf(stderr, "CUDA Linear backward validation failed: %s\n",
                test->label);
        goto done;
    }
    h2d_before = cuda_test_host_to_device_count();
    launches_before = cuda_test_launch_count();
    for (int sample = 0; sample < CUDA_LINEAR_SAMPLES; sample++) {
        struct timespec start;
        struct timespec end;
        if (clock_gettime(CLOCK_MONOTONIC, &start) != 0) goto done;
        for (int iteration = 0; iteration < CUDA_LINEAR_BATCH_ITERATIONS;
             iteration++) {
            if (!dispatch_backward(test, hosts, bytes, access, is_weight)) {
                fprintf(stderr, "CUDA Linear backward timing failed: %s\n",
                        test->label);
                goto done;
            }
        }
        /* A partial bias readback synchronizes the stream without evicting or
         * materializing the large resident dX/dW slots. */
        if (cuda_training_sync(buffers->grad_bias, sizeof(float)) != 0 ||
            clock_gettime(CLOCK_MONOTONIC, &end) != 0) goto done;
        samples[sample] = elapsed_us(&start, &end) /
            (double)CUDA_LINEAR_BATCH_ITERATIONS;
    }
    launches_after = cuda_test_launch_count();
    if (cuda_test_host_to_device_count() != h2d_before ||
        launches_after - launches_before != 3u *
            (uint64_t)CUDA_LINEAR_SAMPLES * CUDA_LINEAR_BATCH_ITERATIONS) {
        fprintf(stderr, "CUDA Linear backward residency check failed: %s\n",
                test->label);
        goto done;
    }
    {
        double median_iteration_us = median(samples);
        double flops = 4.0 * (double)test->rows * test->d_in * test->d_out +
            (double)test->rows * test->d_out;
        printf("cuda_linear direction=backward case=%s m=%u k=%u n=%u "
               "launches_per_iteration=3 median_us_iteration=%.3f "
               "median_us_per_launch=%.3f gflops=%.3f "
               "sync=stream_plus_4b_d2h "
               "validation_max_abs=%.9g\n",
               test->label, test->rows, test->d_in, test->d_out,
               median_iteration_us, median_iteration_us / 3.0,
               flops / (median_iteration_us * 1000.0),
               (double)max_error);
    }
    ok = 1;
done:
    if (training_active) cuda_training_end();
    return ok;
}

static int benchmark_case(const CudaLinearCase* test) {
    CudaLinearBuffers buffers = {0};
    int ok = 0;
    if (!allocate_buffers(test, &buffers)) {
        fprintf(stderr, "CUDA Linear allocation failed: %s\n", test->label);
        goto done;
    }
    if (!benchmark_forward(test, &buffers) ||
        !benchmark_backward(test, &buffers)) goto done;
    ok = 1;
done:
    cuda_graph_reset();
    free_buffers(&buffers);
    return ok;
}

int main(void) {
    CudaTestEngineScope engine_scope = {0};
    static const CudaLinearCase cases[] = {
        {"encoder_attention", 402u, 320u, 320u},
        {"encoder_ffn_up", 402u, 320u, 1280u},
        {"encoder_ffn_down", 402u, 1280u, 320u},
        /* max_out_len=192 includes BOS, so the training decoder has 191
         * physical rows. These six base dense calls use IN_OUT weights. */
        {"decoder_attention", 191u, 320u, 320u},
        {"decoder_ffn_up", 191u, 320u, 1280u},
        {"decoder_ffn_down", 191u, 1280u, 320u},
    };
    if (cuda_test_engine_scope_begin(&engine_scope) != 0) {
        fputs("CUDA benchmark engine-state initialization failed\n", stderr);
        return 1;
    }
    if (cuda_init() != 0) {
        if (cuda_init_failure_is_unavailable()) {
            puts("CUDA device unavailable; skipping CUDA Linear benchmark");
            cuda_test_engine_scope_end(&engine_scope);
            return 77;
        }
        fputs("CUDA device was found, but full-profile CUDA initialization failed\n",
              stderr);
        cuda_test_engine_scope_end(&engine_scope);
        return 1;
    }
    if (!cuda_training_available()) {
        fputs("CUDA full-profile training scheduler is unavailable\n", stderr);
        cuda_cleanup();
        cuda_test_engine_scope_end(&engine_scope);
        return 1;
    }
    printf("cuda_linear metadata samples=%d batch_iterations=%d warmups=%d "
           "timing=clock_monotonic residency=warm fp32_contract=%s\n",
           CUDA_LINEAR_SAMPLES, CUDA_LINEAR_BATCH_ITERATIONS,
           CUDA_LINEAR_WARMUP_ITERATIONS, CUDA_LINEAR_FP32_CONTRACT);
    for (size_t index = 0; index < sizeof(cases) / sizeof(cases[0]); index++) {
        if (!benchmark_case(&cases[index])) {
            cuda_cleanup();
            cuda_test_engine_scope_end(&engine_scope);
            return 1;
        }
    }
    cuda_cleanup();
    cuda_test_engine_scope_end(&engine_scope);
    return 0;
}
