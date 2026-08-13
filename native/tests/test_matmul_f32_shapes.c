/*
 * matmul_f32 and matmul_f32_out_in against the definition.
 *
 * These are the unpacked dense kernels the Linear node falls back to when its
 * weight is not an immutable model tensor, and the ones BatchMatMul always
 * uses.  The vector path computes VX_MATMUL_NR columns at a time and hands the
 * remainder to a scalar tail, and that tail used to receive the remaining
 * column count as both its column count and its row stride.  For any d_out
 * above the tile width that is not a multiple of it, every tail element then
 * read and wrote the wrong row: at d_out=17 the error was 1.4 on values of
 * order 1, and at 257 it was 21.7.  Nothing caught it because the model driving
 * kernel work at the time had feature widths of 320, 960 and 1280, all exact
 * multiples, and because the bit-exactness checks around it compared one engine
 * build against another rather than against the definition.
 *
 * So this test is deliberately written against a triple loop stated here, and
 * it sweeps widths rather than sampling the ones some model happens to use.
 */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

void matmul_f32(const float* input, const float* weight, const float* bias,
                float* output, int seq_len, int d_in, int d_out);
void matmul_f32_out_in(const float* input, const float* weight,
                       const float* bias, float* output,
                       int rows, int d_in, int d_out);

static uint32_t g_state = 0x2545f491u;

static float next_value(void) {
    g_state = g_state * 1664525u + 1013904223u;
    return (float)((int32_t)(g_state >> 9) % 401 - 200) / 200.0f;
}

/* C = A*B + bias, with B in [K,N] order.  Double accumulation so the reference
 * is not the thing under test. */
static void reference_k_major(const float* a, const float* b, const float* bias,
                              float* c, int m, int k, int n) {
    for (int row = 0; row < m; row++)
        for (int column = 0; column < n; column++) {
            double sum = bias ? bias[column] : 0.0;
            for (int inner = 0; inner < k; inner++)
                sum += (double)a[(size_t)row * k + inner] *
                       b[(size_t)inner * n + column];
            c[(size_t)row * n + column] = (float)sum;
        }
}

static int check(int m, int k, int n, int use_bias) {
    float* a = (float*)malloc((size_t)m * k * sizeof(float));
    float* b_kn = (float*)malloc((size_t)k * n * sizeof(float));
    float* b_nk = (float*)malloc((size_t)k * n * sizeof(float));
    float* bias = use_bias ? (float*)malloc((size_t)n * sizeof(float)) : NULL;
    float* expected = (float*)malloc((size_t)m * n * sizeof(float));
    float* actual = (float*)malloc((size_t)m * n * sizeof(float));
    int failed = 0;
    if (!a || !b_kn || !b_nk || !expected || !actual || (use_bias && !bias)) {
        fprintf(stderr, "allocation failed\n");
        free(a); free(b_kn); free(b_nk); free(bias); free(expected); free(actual);
        return 1;
    }
    for (size_t index = 0; index < (size_t)m * k; index++) a[index] = next_value();
    for (size_t index = 0; index < (size_t)k * n; index++) b_kn[index] = next_value();
    if (bias) for (int index = 0; index < n; index++) bias[index] = next_value();
    for (int inner = 0; inner < k; inner++)
        for (int column = 0; column < n; column++)
            b_nk[(size_t)column * k + inner] = b_kn[(size_t)inner * n + column];
    reference_k_major(a, b_kn, bias, expected, m, k, n);

    for (int variant = 0; variant < 2; variant++) {
        const char* label = variant ? "matmul_f32_out_in" : "matmul_f32";
        if (variant)
            matmul_f32_out_in(a, b_nk, bias, actual, m, k, n);
        else
            matmul_f32(a, b_kn, bias, actual, m, k, n);
        for (size_t index = 0; index < (size_t)m * n; index++) {
            const double want = expected[index];
            const double got = actual[index];
            const double scale = fabs(want) > 1.0 ? fabs(want) : 1.0;
            if (fabs(got - want) / scale > 1e-4) {
                fprintf(stderr,
                        "%s m=%d k=%d n=%d bias=%d: [%zu] got %.9g want %.9g\n",
                        label, m, k, n, use_bias, index, got, want);
                failed = 1;
                break;
            }
        }
    }
    free(a); free(b_kn); free(b_nk); free(bias); free(expected); free(actual);
    return failed;
}

int main(void) {
    int failures = 0;
    /* Sweep the output width across and past the tile boundary.  Widths 17
     * through 48 cover the first tail, one exact tile, and a tail after two. */
    for (int n = 1; n <= 48; n++) {
        failures += check(3, 5, n, 0);
        failures += check(3, 5, n, 1);
    }
    /* Ragged in every dimension at a size that spans several tiles. */
    failures += check(101, 333, 257, 1);
    failures += check(1, 512, 129, 1);
    failures += check(7, 1, 33, 0);
    /* The widths a model with power-of-two features actually uses, which is
     * what used to be the whole of the coverage. */
    failures += check(4, 320, 320, 1);
    failures += check(2, 320, 1280, 0);
    if (failures) {
        fprintf(stderr, "matmul_f32 shape sweep failed (%d cases)\n", failures);
        return 1;
    }
    printf("matmul_f32 shape tests passed\n");
    return 0;
}
