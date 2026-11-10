/* Compile with production cpu_features.c, thread_pool.c and sequence_ops.c.
 * Kept standalone so the identical check can run on native ARM devices. */
#include "fast_math.inc"
#include "elementwise_f32.inc"
#include "activations_hard.inc"
#include "activations.inc"
#include "sequence_ops.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum { CAPACITY = 65537 };
static float input[CAPACITY], expected[CAPACITY], actual[CAPACITY], shuffled[CAPACITY];
static unsigned checks;

static void run(int op, const float* x, float* y, int n) {
    switch (op) {
    case 0: sin_f32(x, y, (uint32_t)n); break;
    case 1: cos_f32(x, y, (uint32_t)n); break;
    case 2: tanh_f32(x, y, n); break;
    case 3: sigmoid_f32(x, y, n); break;
    case 4: silu_f32(x, y, n); break;
    default: gelu_f32(x, y, n); break;
    }
}

static int same(float a, float b) {
    checks++;
    if (isnan(a) && isnan(b)) return 1;
    return memcmp(&a, &b, sizeof(a)) == 0;
}

static int oracle(int op, float x, float actual_value) {
    double reference;
    double tolerance;
    if (op == 0 || op == 1) {
        reference = op == 0 ? sin((double)x) : cos((double)x);
        tolerance = isfinite(x) && fabsf(x) <= 32.0f ? 2e-6 : 1.5e-7;
    } else if (op == 2) {
        reference = tanh((double)x);
        tolerance = 8e-7 * fabs(reference);
    } else if (op == 5) {
        reference = 0.5 * x * (1.0 + erf((double)x * 0.7071067811865475244));
        tolerance = 5e-7 + 2e-6 * fabs(reference);
    } else {
        /* Sigmoid/SiLU retain the public kernel's bounded-exp contract. */
        float scalar = 1.0f / (1.0f + accurate_expf(-x));
        reference = op == 3 ? scalar : x * scalar;
        tolerance = 2e-6 * fabs(reference);
    }
    checks++;
    if (isnan(reference)) return isnan(actual_value);
    if (isinf(reference)) return actual_value == reference;
    return isfinite(actual_value) && fabs(actual_value - reference) <= tolerance;
}

int main(void) {
    const int lengths[] = {1, 3, 4, 7, 8, 9, 15, 16, 17, 31, 32, 33,
                           2049, 4097, 8191, 8192, 8193,
                           16383, 16384, 16385, CAPACITY};
    const float edges[] = {0.0f, -0.0f, INFINITY, -INFINITY, NAN, 1e10f,
                            -1e10f, 32.0f, -32.0f, 1e-40f, -1e-40f, 1e-8f};
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(1);
    if (!pool) return 2;
    VxKernelThreadPoolScope scope = vx_kernel_thread_pool_scope_enter(pool);
    uint32_t state = 7171;
    for (int i = 0; i < CAPACITY; i++) {
        state = state * 1664525u + 1013904223u;
        input[i] = ((int)(state >> 8) - 8388608) * 0x1p-18f;
        if (i % 97 < (int)(sizeof(edges) / sizeof(edges[0])))
            input[i] = edges[i % 97];
    }
    int valid = 1;
    for (int op = 0; op < 6 && valid; op++) {
        /* One-element calls force the tail, independent of array position. */
        for (int i = 0; i < CAPACITY; i++) {
            run(op, input + i, expected + i, 1);
            if (!oracle(op, input[i], expected[i])) {
                fprintf(stderr, "oracle op=%d i=%d x=%a actual=%a\n",
                        op, i, input[i], expected[i]);
                valid = 0;
                break;
            }
        }
        for (int threads = 1; threads <= 4 && valid; threads += 3) {
            vx_set_num_threads(threads);
            for (unsigned test = 0; test < sizeof(lengths) / sizeof(lengths[0]); test++) {
                int n = lengths[test];
                run(op, input, actual, n);
                for (int i = 0; i < n; i++)
                    if (!same(actual[i], expected[i])) {
                        fprintf(stderr, "partition op=%d threads=%d n=%d i=%d actual=%a expected=%a\n",
                                op, threads, n, i, actual[i], expected[i]);
                        valid = 0;
                        break;
                    }
                /* Reverse changes lane, grouping and scalar-tail membership. */
                for (int i = 0; i < n; i++) shuffled[i] = input[n - 1 - i];
                run(op, shuffled, shuffled, n);
                for (int i = 0; i < n; i++)
                    if (!same(shuffled[i], expected[n - 1 - i])) {
                        fprintf(stderr, "permutation op=%d threads=%d n=%d i=%d\n", op, threads, n, i);
                        valid = 0;
                        break;
                    }
            }
            /* Irregular explicit splits model graph fusion/transpose spans. */
            for (int i = 0; i < CAPACITY;) {
                int n = 1 + i % 37;
                if (n > CAPACITY - i) n = CAPACITY - i;
                run(op, input + i, actual + i, n);
                i += n;
            }
            for (int i = 0; i < CAPACITY; i++)
                if (!same(actual[i], expected[i])) {
                    fprintf(stderr, "explicit split op=%d i=%d\n", op, i);
                    valid = 0;
                    break;
                }
        }
    }
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    if (valid) printf("unary partition/oracle passed: %u checks, threads=1,4\n", checks);
    return valid ? 0 : 1;
}
