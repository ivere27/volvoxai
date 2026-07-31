#include "cuda_engine.h"
#include "cuda_test_engine_scope.h"

#include <float.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(expression) do { \
    if (!(expression)) { \
        fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #expression); \
        cuda_cleanup(); \
        return 1; \
    } \
} while (0)

static int near(float actual, float expected, float tolerance) {
    return fabsf(actual - expected) <= tolerance;
}

static float f32_from_bits(uint32_t bits) {
    union { uint32_t u; float f; } value = {bits};
    return value.f;
}

static uint32_t f32_bits(float value) {
    union { uint32_t u; float f; } bits = {0};
    bits.f = value;
    return bits.u;
}

static int32_t reference_round_ties_even(float value) {
    int32_t lower = (int32_t)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

static uint8_t reference_quantize(float transformed,
                                  int32_t output_zero_point,
                                  uint32_t output_dtype) {
    int32_t minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    int32_t maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    int32_t quantized;
    if (transformed != transformed) quantized = output_zero_point;
    else if (transformed <= (float)minimum) quantized = minimum;
    else if (transformed >= (float)maximum) quantized = maximum;
    else quantized = reference_round_ties_even(transformed);
    return (uint8_t)quantized;
}

static int32_t reference_raw_value(uint8_t raw, uint32_t dtype) {
    return dtype == VX_DTYPE_I8 && raw >= 128u
        ? (int32_t)raw - 256 : (int32_t)raw;
}

static float reference_qgelu_erf(float value) {
    const float sign = value >= 0.0f ? 1.0f : -1.0f;
    const float magnitude = fabsf(value);
    const float denominator = 1.0f + 0.3275911f * magnitude;
    const float t = 1.0f / denominator;
    float polynomial = 1.061405429f * t;
    polynomial = polynomial - 1.453152027f;
    polynomial = polynomial * t;
    polynomial = polynomial + 1.421413741f;
    polynomial = polynomial * t;
    polynomial = polynomial - 0.284496736f;
    polynomial = polynomial * t;
    polynomial = polynomial + 0.254829592f;
    polynomial = polynomial * t;
    {
        const float exponential = expf(-(magnitude * magnitude));
        return sign * (1.0f - polynomial * exponential);
    }
}

static uint8_t reference_qactivation(uint8_t raw, float input_scale,
                                     int32_t input_zero_point,
                                     float output_scale,
                                     int32_t output_zero_point,
                                     uint32_t input_dtype,
                                     uint32_t output_dtype, int gelu) {
    const float input_value =
        (float)(reference_raw_value(raw, input_dtype) - input_zero_point) *
        input_scale;
    float activated;
    if (!gelu) {
        activated = input_value / (1.0f + expf(-input_value));
    } else {
        const float erf_input = input_value * 0.7071067811865476f;
        const float cdf = 0.5f * (1.0f + reference_qgelu_erf(erf_input));
        activated = input_value * cdf;
    }
    {
        const float scaled = activated / output_scale;
        const float transformed = scaled + (float)output_zero_point;
        return reference_quantize(transformed, output_zero_point,
                                  output_dtype);
    }
}

static void reference_layernorm(const float* input, float* output,
                                int rows, int width, float epsilon) {
    for (int row = 0; row < rows; row++) {
        const float* source = input + row * width;
        float* destination = output + row * width;
        float mean = 0.0f;
        float variance = 0.0f;
        for (int column = 0; column < width; column++) mean += source[column];
        mean /= (float)width;
        for (int column = 0; column < width; column++) {
            float delta = source[column] - mean;
            variance += delta * delta;
        }
        float inverse = 1.0f / sqrtf(variance / (float)width + epsilon);
        for (int column = 0; column < width; column++)
            destination[column] = (source[column] - mean) * inverse;
    }
}

static int8_t reference_requantize_i8(int8_t input, float multiplier,
                                      int32_t input_zero_point,
                                      int32_t output_zero_point) {
    float centered = (float)((int32_t)input - input_zero_point);
    float scaled = centered * multiplier;
    float transformed = scaled + (float)output_zero_point;
    return (int8_t)reference_quantize(
        transformed, output_zero_point, VX_DTYPE_I8);
}

enum {
    DEPTHWISE_TACTIC_GENERIC = 0,
    DEPTHWISE_TACTIC_3X3_C1 = 1,
    DEPTHWISE_TACTIC_3X3_C4 = 2,
    DEPTHWISE_TACTIC_5X5_C1 = 3,
    DEPTHWISE_TACTIC_5X5_C4 = 4,
};

static float reference_mac_strict(float sum, float input, float weight) {
    volatile float product = input * weight;
    volatile float result = sum + product;
    return result;
}

static void reference_depthwise_conv2d(
        const float* input, float* output, const float* weight,
        const float* bias, int batch, int input_height, int input_width,
        int input_channels, int output_channels, int kernel_height,
        int kernel_width, int output_height, int output_width,
        int stride_y, int stride_x, int padding_top, int padding_left,
        int relu, int dilation_y, int dilation_x) {
    int multiplier = output_channels / input_channels;
    for (int n = 0; n < batch; n++) {
        for (int oy = 0; oy < output_height; oy++) {
            for (int ox = 0; ox < output_width; ox++) {
                for (int oc = 0; oc < output_channels; oc++) {
                    int input_channel = oc / multiplier;
                    int depth_multiplier = oc - input_channel * multiplier;
                    float sum = bias ? bias[oc] : 0.0f;
                    for (int ky = 0; ky < kernel_height; ky++) {
                        int iy = oy * stride_y + ky * dilation_y -
                            padding_top;
                        if (iy < 0 || iy >= input_height) continue;
                        for (int kx = 0; kx < kernel_width; kx++) {
                            int ix = ox * stride_x + kx * dilation_x -
                                padding_left;
                            int input_index;
                            int weight_index;
                            if (ix < 0 || ix >= input_width) continue;
                            input_index = ((n * input_height + iy) *
                                input_width + ix) * input_channels +
                                input_channel;
                            weight_index = (((ky * kernel_width + kx) *
                                input_channels + input_channel) * multiplier) +
                                depth_multiplier;
                            sum = reference_mac_strict(sum,
                                input[input_index], weight[weight_index]);
                        }
                    }
                    if (relu != 0) {
                        sum = sum > 0.0f ? sum : 0.0f;
                        if (relu >= 2) sum = sum < 6.0f ? sum : 6.0f;
                    }
                    output[((n * output_height + oy) * output_width + ox) *
                        output_channels + oc] = sum;
                }
            }
        }
    }
}

static int check_depthwise_case(
        const char* name, const float* input, const float* weight,
        const float* bias, int batch, int input_height, int input_width,
        int input_channels, int output_channels, int kernel_height,
        int kernel_width, int output_height, int output_width,
        int stride_y, int stride_x, int padding_top, int padding_left,
        int relu, int dilation_y, int dilation_x, int expected_tactic) {
    size_t output_count = (size_t)batch * (size_t)output_height *
        (size_t)output_width * (size_t)output_channels;
    float* output = (float*)calloc(output_count, sizeof(float));
    float* expected = (float*)malloc(output_count * sizeof(float));
    uint64_t before[4];
    uint64_t after[4];
    int ok = 0;
    if (!output || !expected) goto done;
    reference_depthwise_conv2d(input, expected, weight, bias, batch,
        input_height, input_width, input_channels, output_channels,
        kernel_height, kernel_width, output_height, output_width,
        stride_y, stride_x, padding_top, padding_left, relu,
        dilation_y, dilation_x);
    before[0] = cuda_test_depthwise_conv2d_3x3_c1_launch_count();
    before[1] = cuda_test_depthwise_conv2d_3x3_c4_launch_count();
    before[2] = cuda_test_depthwise_conv2d_5x5_c1_launch_count();
    before[3] = cuda_test_depthwise_conv2d_5x5_c4_launch_count();
    cuda_graph_reset();
    cuda_graph_begin_forward();
    if (!cuda_graph_conv2d_f32(input, output, weight, bias, batch,
          input_height, input_width, input_channels, output_channels,
          kernel_height, kernel_width, output_height, output_width,
          stride_y, stride_x, padding_top, padding_left, input_channels,
          relu, dilation_y, dilation_x) || cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output, output_count * sizeof(float), 0)) {
        fprintf(stderr, "CUDA depthwise case '%s' did not execute\n", name);
        goto done;
    }
    after[0] = cuda_test_depthwise_conv2d_3x3_c1_launch_count();
    after[1] = cuda_test_depthwise_conv2d_3x3_c4_launch_count();
    after[2] = cuda_test_depthwise_conv2d_5x5_c1_launch_count();
    after[3] = cuda_test_depthwise_conv2d_5x5_c4_launch_count();
    for (int tactic = 1; tactic <= 4; tactic++) {
        uint64_t want = expected_tactic == tactic ? 1u : 0u;
        if (after[tactic - 1] - before[tactic - 1] != want) {
            fprintf(stderr,
                    "CUDA depthwise case '%s' selected tactic %d, expected %d\n",
                    name, tactic, expected_tactic);
            goto done;
        }
    }
    for (size_t index = 0; index < output_count; index++) {
        if (f32_bits(output[index]) != f32_bits(expected[index])) {
            fprintf(stderr,
                    "CUDA depthwise case '%s' mismatch[%zu]: actual=%g "
                    "(0x%08x) expected=%g (0x%08x)\n",
                    name, index, output[index], f32_bits(output[index]),
                    expected[index], f32_bits(expected[index]));
            goto done;
        }
    }
    ok = 1;
done:
    free(output);
    free(expected);
    return ok;
}

enum {
    CONV1X1_TACTIC_BASELINE = 0,
    CONV1X1_TACTIC_BM32_BN32_BK16 = 1,
    CONV1X1_TACTIC_BM16_BN64_BK16 = 2,
};

static int check_conv1x1_tactic(int rows, int input_channels,
                                int output_channels, int relu,
                                int with_bias, int expected_tactic) {
    size_t input_count = (size_t)rows * (size_t)input_channels;
    size_t weight_count = (size_t)input_channels * (size_t)output_channels;
    size_t output_count = (size_t)rows * (size_t)output_channels;
    float* input = (float*)malloc(input_count * sizeof(float));
    float* weight = (float*)malloc(weight_count * sizeof(float));
    float* bias = (float*)malloc((size_t)output_channels * sizeof(float));
    float* output = (float*)calloc(output_count, sizeof(float));
    uint64_t tiled_before;
    uint64_t bm32_before;
    uint64_t bm16_before;
    int ok = 0;
    if (!input || !weight || !bias || !output) goto done;
    for (size_t index = 0; index < input_count; index++)
        input[index] = (float)((int)(index % 7u) - 3) * 0.125f;
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (float)((int)(index % 5u) - 2) * 0.0625f;
    for (int column = 0; column < output_channels; column++)
        bias[column] = (float)(column % 5 - 2) * 0.25f;

    cuda_graph_reset();
    tiled_before = cuda_test_conv2d_1x1_tiled_launch_count();
    bm32_before = cuda_test_conv2d_1x1_bm32_bn32_bk16_launch_count();
    bm16_before = cuda_test_conv2d_1x1_bm16_bn64_bk16_launch_count();
    cuda_graph_begin_forward();
    if (!cuda_graph_conv2d_f32(input, output, weight,
          with_bias ? bias : NULL, 1, 1, rows, input_channels,
          output_channels, 1, 1, 1, rows, 1, 1, 0, 0, 1, relu, 1, 1))
        goto done;
    if (cuda_test_conv2d_1x1_tiled_launch_count() - tiled_before != 1u ||
        cuda_test_conv2d_1x1_bm32_bn32_bk16_launch_count() - bm32_before !=
            (uint64_t)(expected_tactic == CONV1X1_TACTIC_BM32_BN32_BK16) ||
        cuda_test_conv2d_1x1_bm16_bn64_bk16_launch_count() - bm16_before !=
            (uint64_t)(expected_tactic == CONV1X1_TACTIC_BM16_BN64_BK16)) {
        fprintf(stderr, "unexpected CUDA 1x1 tactic for M=%d K=%d N=%d\n",
                rows, input_channels, output_channels);
        goto done;
    }
    if (cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output, output_count * sizeof(float), 0))
        goto done;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < output_channels; column++) {
            float expected = with_bias ? bias[column] : 0.0f;
            for (int k = 0; k < input_channels; k++)
                expected += input[row * input_channels + k] *
                    weight[k * output_channels + column];
            if (relu != 0 && expected < 0.0f) expected = 0.0f;
            if (relu >= 2 && expected > 6.0f) expected = 6.0f;
            if (!near(output[row * output_channels + column], expected,
                      5.0e-5f)) {
                fprintf(stderr,
                        "CUDA 1x1 mismatch M=%d K=%d N=%d row=%d col=%d: "
                        "actual=%g expected=%g\n",
                        rows, input_channels, output_channels, row, column,
                        output[row * output_channels + column], expected);
                goto done;
            }
        }
    }
    ok = 1;
done:
    free(input);
    free(weight);
    free(bias);
    free(output);
    return ok;
}

static float reference_add_strict(float left, float right) {
    volatile float result = left + right;
    return result;
}

static int check_conv1x1_add_tactic(int rows, int input_channels,
                                    int output_channels, int relu,
                                    int with_bias, int expected_tactic) {
    size_t input_count = (size_t)rows * (size_t)input_channels;
    size_t weight_count = (size_t)input_channels * (size_t)output_channels;
    size_t output_count = (size_t)rows * (size_t)output_channels;
    float* input = (float*)malloc(input_count * sizeof(float));
    float* weight = (float*)malloc(weight_count * sizeof(float));
    float* bias = (float*)malloc((size_t)output_channels * sizeof(float));
    float* residual = (float*)malloc(output_count * sizeof(float));
    float* output = (float*)calloc(output_count, sizeof(float));
    uint64_t add_before;
    uint64_t tiled_before;
    uint64_t bm32_before;
    uint64_t bm16_before;
    int ok = 0;
    if (!input || !weight || !bias || !residual || !output) goto done;
    for (size_t index = 0; index < input_count; index++)
        input[index] = (float)((int)(index % 7u) - 3) * 0.125f;
    for (size_t index = 0; index < weight_count; index++)
        weight[index] = (float)((int)(index % 5u) - 2) * 0.0625f;
    for (int column = 0; column < output_channels; column++)
        bias[column] = (float)(column % 5 - 2) * 0.25f;
    for (size_t index = 0; index < output_count; index++)
        residual[index] = (float)((int)(index % 11u) - 5) * 0.03125f;
    if (!with_bias) goto done;
    /* 4097 * 4097 is exactly 16785409, halfway between representable F32
     * values at this magnitude. Strict mul.rn then add.rn cancels the rounded
     * product to zero; an allowed fast-FMA build retains the exact +1. */
    for (int k = 0; k < input_channels; k++) {
        input[k] = 0.0f;
        weight[k * output_channels] = 0.0f;
    }
    input[0] = 4097.0f;
    weight[0] = 4097.0f;
    bias[0] = -16785408.0f;
    residual[0] = 0.25f;

    cuda_graph_reset();
    add_before = cuda_test_conv2d_add_launch_count();
    tiled_before = cuda_test_conv2d_1x1_tiled_add_launch_count();
    bm32_before =
        cuda_test_conv2d_1x1_bm32_bn32_bk16_add_launch_count();
    bm16_before =
        cuda_test_conv2d_1x1_bm16_bn64_bk16_add_launch_count();
    cuda_graph_begin_forward();
    if (!cuda_graph_conv2d_add_f32(input, output, weight,
          with_bias ? bias : NULL, residual, 1, 1, rows, input_channels,
          output_channels, 1, 1, 1, rows, 1, 1, 0, 0, 1, relu, 1, 1))
        goto done;
    if (cuda_test_conv2d_add_launch_count() - add_before != 1u ||
        cuda_test_conv2d_1x1_tiled_add_launch_count() - tiled_before != 1u ||
        cuda_test_conv2d_1x1_bm32_bn32_bk16_add_launch_count() -
            bm32_before !=
            (uint64_t)(expected_tactic ==
                       CONV1X1_TACTIC_BM32_BN32_BK16) ||
        cuda_test_conv2d_1x1_bm16_bn64_bk16_add_launch_count() -
            bm16_before !=
            (uint64_t)(expected_tactic ==
                       CONV1X1_TACTIC_BM16_BN64_BK16)) {
        fprintf(stderr,
                "unexpected fused CUDA 1x1 tactic for M=%d K=%d N=%d\n",
                rows, input_channels, output_channels);
        goto done;
    }
    if (cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output, output_count * sizeof(float), 0))
        goto done;
    for (int row = 0; row < rows; row++) {
        for (int column = 0; column < output_channels; column++) {
            float expected = with_bias ? bias[column] : 0.0f;
            size_t index = (size_t)row * (size_t)output_channels +
                (size_t)column;
            for (int k = 0; k < input_channels; k++)
                expected = reference_mac_strict(expected,
                    input[row * input_channels + k],
                    weight[k * output_channels + column]);
            if (relu != 0 && expected < 0.0f) expected = 0.0f;
            if (relu >= 2 && expected > 6.0f) expected = 6.0f;
            expected = reference_add_strict(expected, residual[index]);
            if (index == 0) {
#if VOLVOXAI_CUDA_FAST_FP32
                expected = 1.25f;
#else
                expected = 0.25f;
#endif
                if (f32_bits(output[index]) != f32_bits(expected)) {
                    fprintf(stderr,
                            "fused CUDA 1x1 contraction sentinel M=%d K=%d "
                            "N=%d: actual=%g (0x%08x) expected=%g "
                            "(0x%08x)\n",
                            rows, input_channels, output_channels,
                            output[index], f32_bits(output[index]), expected,
                            f32_bits(expected));
                    goto done;
                }
                continue;
            }
            if (!near(output[index], expected, 5.0e-5f)) {
                fprintf(stderr,
                        "fused CUDA 1x1 mismatch M=%d K=%d N=%d row=%d "
                        "col=%d: actual=%g expected=%g\n",
                        rows, input_channels, output_channels, row, column,
                        output[index], expected);
                goto done;
            }
        }
    }
    ok = 1;
done:
    free(input);
    free(weight);
    free(bias);
    free(residual);
    free(output);
    return ok;
}

static int test_matmul_tiled_tail(int out_in) {
    enum { ROWS = 17, D_IN = 19, D_OUT = 23 };
    float input[ROWS * D_IN];
    float weight[D_IN * D_OUT];
    float bias[D_OUT];
    float output[ROWS * D_OUT];
    float expected[ROWS * D_OUT];
    int ok = 0;
    cuda_graph_reset();
    for (int index = 0; index < ROWS * D_IN; index++)
        input[index] = ((float)(index % 29) - 14.0f) * 0.03125f;
    for (int column = 0; column < D_OUT; column++) {
        bias[column] = ((float)(column % 7) - 3.0f) * 0.0625f;
        for (int k = 0; k < D_IN; k++) {
            float value = ((float)((k * 11 + column * 7) % 31) - 15.0f) *
                0.015625f;
            weight[out_in ? column * D_IN + k : k * D_OUT + column] = value;
        }
    }
    for (int row = 0; row < ROWS; row++) {
        for (int column = 0; column < D_OUT; column++) {
            float sum = bias[column];
            for (int k = 0; k < D_IN; k++)
                sum = reference_mac_strict(sum, input[row * D_IN + k],
                    weight[out_in ? column * D_IN + k
                                  : k * D_OUT + column]);
            expected[row * D_OUT + column] = sum;
        }
    }
    if (out_in) {
        cuda_graph_begin_forward();
        if (!cuda_graph_matmul_layout_f32(input, weight, bias, output,
                ROWS, D_IN, D_OUT, 1) || cuda_graph_end_forward() != 0 ||
            !cuda_graph_sync_host(output, sizeof(output), 0)) goto done;
    } else if (!cuda_matmul(
            input, weight, bias, output, ROWS, D_IN, D_OUT)) {
        goto done;
    }
    for (int index = 0; index < ROWS * D_OUT; index++) {
        if (!near(output[index], expected[index], 2.0e-5f)) {
            fprintf(stderr,
                    "CUDA tiled matmul tail mismatch layout=%s index=%d "
                    "actual=%g expected=%g\n",
                    out_in ? "OUT_IN" : "IN_OUT", index,
                    output[index], expected[index]);
            goto done;
        }
    }
    ok = 1;
done:
    cuda_graph_reset();
    return ok;
}

static int test_transient_release_failure_safety(int fail_sync) {
    float input[2] = {1.0f, 2.0f};
    const float weight[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    float output[2] = {0.0f, 0.0f};
    uint64_t uploads;
    int phase = 0;
    int ok = 0;

    cuda_graph_reset();
    cuda_graph_begin_forward();
    if (!cuda_graph_matmul_f32(input, weight, NULL, output, 1, 2, 2) ||
        cuda_graph_end_forward() != 0 ||
        !cuda_graph_sync_host(output, sizeof(output), 0) ||
        output[0] != 7.0f || output[1] != 10.0f)
        goto done;

    phase = 1;
    if (fail_sync) cuda_test_fail_next_transient_release_sync();
    else cuda_test_fail_next_transient_release_context();
    cuda_graph_release_transients();
    if (cuda_test_quarantined_transient_slot_count() != 2u) goto done;

    /* Reuse the exact host addresses with different contents. Quarantine must
     * force a fresh input slot/upload while the true weight remains resident;
     * matching either stale transient slot would produce the old result. */
    input[0] = -2.0f;
    input[1] = 4.0f;
    output[0] = output[1] = 0.0f;
    uploads = cuda_test_host_to_device_count();
    phase = 2;
    cuda_graph_begin_forward();
    if (!cuda_graph_matmul_f32(input, weight, NULL, output, 1, 2, 2) ||
        cuda_graph_end_forward() != 0 ||
        cuda_test_host_to_device_count() - uploads != 1u ||
        !cuda_graph_sync_host(output, sizeof(output), 0) ||
        output[0] != 10.0f || output[1] != 12.0f)
        goto done;

    cuda_graph_release_transients();
    phase = 3;
    if (cuda_test_quarantined_transient_slot_count() != 0u) goto done;
    ok = 1;
done:
    if (!ok) {
        fprintf(stderr,
                "CUDA transient release failure-safety mismatch stage=%s "
                "phase=%d output={%g,%g} orphaned=%llu\n",
                fail_sync ? "sync" : "context", phase,
                output[0], output[1],
                (unsigned long long)
                    cuda_test_quarantined_transient_slot_count());
    }
    cuda_graph_reset();
    return ok;
}

int main(void) {
    CudaTestEngineScope engine_scope = {0};
    if (cuda_test_engine_scope_begin(&engine_scope) != 0) {
        fputs("CUDA test engine-state initialization failed\n", stderr);
        return 1;
    }
    if (cuda_init() != 0) {
        if (!cuda_init_failure_is_unavailable()) {
            fputs("CUDA device was found, but backend initialization failed\n", stderr);
            cuda_test_engine_scope_end(&engine_scope);
            return 1;
        }
        puts("CUDA device unavailable; skipping CUDA kernel correctness test");
        cuda_test_engine_scope_end(&engine_scope);
        return 77;
    }
    CHECK(cuda_test_caller_context_is_clear());
    CHECK(test_matmul_tiled_tail(0));
    CHECK(test_matmul_tiled_tail(1));
    CHECK(test_transient_release_failure_safety(0));
    CHECK(test_transient_release_failure_safety(1));

    {
        const float input = 1.0f;
        float output = 0.0f;
        uint64_t get_before = cuda_test_context_get_current_count();
        uint64_t set_before = cuda_test_context_set_current_count();
        cuda_graph_begin_forward();
        cuda_graph_begin_forward();
        CHECK(!cuda_graph_copy_f32(&input, &output, 1));
        CHECK(cuda_graph_end_forward() != 0);
        CHECK(cuda_test_context_get_current_count() - get_before == 1u);
        CHECK(cuda_test_context_set_current_count() - set_before == 2u);
        CHECK(cuda_test_caller_context_is_clear());
        get_before = cuda_test_context_get_current_count();
        set_before = cuda_test_context_set_current_count();
        CHECK(cuda_graph_end_forward() != 0);
        CHECK(cuda_test_context_get_current_count() == get_before);
        CHECK(cuda_test_context_set_current_count() == set_before);
    }

    {
        const float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f};
        const float weight[6] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
        const float bias[2] = {0.5f, -0.5f};
        const float expected[4] = {22.5f, 27.5f, 11.0f, 11.5f};
        float output[4] = {0};
        CHECK(cuda_matmul(input, weight, bias, output, 2, 3, 2));
        for (int i = 0; i < 4; i++) CHECK(near(output[i], expected[i], 1.0e-5f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[6] = {1.0f, 2.0f, 3.0f, -1.0f, 0.5f, 2.0f};
        const float weight_out_in[6] = {1.0f, 3.0f, 5.0f, 2.0f, 4.0f, 6.0f};
        const float bias[2] = {0.5f, -0.5f};
        const float expected[4] = {22.5f, 27.5f, 11.0f, 11.5f};
        float output[4] = {0};
        CHECK(cuda_graph_matmul_layout_f32(input, weight_out_in, bias, output,
                                           2, 3, 2, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int i = 0; i < 4; i++) CHECK(near(output[i], expected[i], 1.0e-5f));
    }

    cuda_graph_reset();
    {
        float storage[4] = {99.0f, 99.0f, 1.0f, 2.0f};
        const float weight[2] = {1.0f, 1.0f};
        float output = 0.0f;
        CHECK(cuda_matmul(storage + 2, weight, NULL, &output, 1, 2, 1));
        CHECK(output == 3.0f);
        storage[2] = 4.0f;
        storage[3] = 5.0f;
        CHECK(cuda_matmul(storage + 2, weight, NULL, &output, 1, 2, 1));
        CHECK(output == 9.0f);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int a_shape[3] = {2, 2, 3};
        const int b_shape[3] = {1, 3, 2};
        const int output_shape[3] = {2, 2, 2};
        const int invalid_output_shape[3] = {1, 2, 2};
        const float a[12] = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
        };
        const float b[6] = {
            1, 0,
            0, 1,
            1, 1,
        };
        const float expected[8] = {
            4, 5, 10, 11,
            16, 17, 22, 23,
        };
        float output[8] = {0};
        CHECK(!cuda_graph_batch_matmul_f32(
              a, a_shape, 3, b, b_shape, 3, output,
              invalid_output_shape, 3));
        CHECK(cuda_graph_batch_matmul_f32(
              a, a_shape, 3, b, b_shape, 3, output, output_shape, 3));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 8; index++)
            CHECK(output[index] == expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int a_shape[4] = {2, 1, 2, 2};
        const int b_shape[3] = {3, 2, 2};
        const int output_shape[4] = {2, 3, 2, 2};
        const int invalid_output_shape[4] = {2, 2, 2, 2};
        const uint8_t a[8] = {
            129, 130, 131, 132,
            127, 128, 130, 126,
        };
        const int8_t b[12] = {
            0, -1, -1, 0,
            1, 0, -2, 1,
            -1, -2, 2, 0,
        };
        const int8_t expected[24] = {
            4, 4, 4, 5,
            3, 6, 4, 8,
            6, 4, 9, 4,
            2, 3, 4, 2,
            2, 2, 6, 2,
            3, 4, 0, 1,
        };
        int8_t output[24] = {0};
        CHECK(!cuda_graph_qbatch_matmul_i8u8(
              a, a_shape, 4, 0.5f, 128, VX_DTYPE_U8,
              b, b_shape, 3, 0.25f, -1, VX_DTYPE_I8,
              output, invalid_output_shape, 4,
              0.25f, 3, VX_DTYPE_I8));
        CHECK(cuda_graph_qbatch_matmul_i8u8(
              a, a_shape, 4, 0.5f, 128, VX_DTYPE_U8,
              b, b_shape, 3, 0.25f, -1, VX_DTYPE_I8,
              output, output_shape, 4,
              0.25f, 3, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(memcmp(output, expected, sizeof(output)) == 0);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[12] = {
             1.0f, 9.0f,  5.0f, 9.0f,  5.0f, 3.0f,
            -1.0f, 2.0f, -3.0f, 7.0f, -1.0f, 7.0f,
        };
        const float rank_one_input[4] = {2.0f, 8.0f, 8.0f, 1.0f};
        const int32_t expected[4] = {1, 0, 0, 1};
        int32_t output[4] = {0};
        int32_t duplicate_output[4] = {0};
        int32_t rank_one_output = -1;
        union {
            float input[12];
            int32_t output[12];
        } alias = {{0}};
        CHECK(!cuda_graph_argmax_f32(
              input, output, 0u, 3u, 2u));
        CHECK(!cuda_graph_argmax_f32(
              input, output, 2u, 0u, 2u));
        CHECK(!cuda_graph_argmax_f32(
              input, output, 2u, 3u, 0u));
        CHECK(!cuda_graph_argmax_f32(
              input, output, 1u, UINT32_MAX, 1u));
        CHECK(!cuda_graph_argmax_f32(
              alias.input, alias.output, 2u, 3u, 2u));
        CHECK(cuda_graph_argmax_f32(
              input, output, 2u, 3u, 2u));
        CHECK(cuda_graph_argmax_f32(
              input, duplicate_output, 2u, 3u, 2u));
        CHECK(cuda_graph_argmax_f32(
              rank_one_input, &rank_one_output, 1u, 4u, 1u));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(cuda_graph_sync_host(
              duplicate_output, sizeof(duplicate_output), 0));
        CHECK(cuda_graph_sync_host(
              &rank_one_output, sizeof(rank_one_output), 0));
        CHECK(memcmp(output, expected, sizeof(output)) == 0);
        CHECK(memcmp(
              duplicate_output, expected, sizeof(duplicate_output)) == 0);
        CHECK(rank_one_output == 1);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int32_t a[6] = {1, 2, 3, 4, 5, 6};
        const int32_t b[3] = {1, 3, 3};
        const int32_t equal_expected[6] = {1, 0, 1, 0, 0, 0};
        const int32_t greater_equal_expected[6] = {1, 0, 1, 1, 1, 1};
        const uint32_t output_strides[2] = {3, 1};
        const uint32_t a_strides[2] = {3, 1};
        const uint32_t b_strides[2] = {0, 1};
        int32_t equal[6] = {0};
        int32_t greater_equal[6] = {0};
        CHECK(cuda_graph_compare_i32(
              a, 6, b, 3, equal, 6, output_strides,
              a_strides, b_strides, 2, 0));
        CHECK(cuda_graph_compare_i32(
              a, 6, b, 3, greater_equal, 6, output_strides,
              a_strides, b_strides, 2, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(equal, sizeof(equal), 0));
        CHECK(cuda_graph_sync_host(
              greater_equal, sizeof(greater_equal), 0));
        for (int index = 0; index < 6; index++) {
            CHECK(equal[index] == equal_expected[index]);
            CHECK(greater_equal[index] == greater_equal_expected[index]);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int32_t logical_input[4] = {0, 1, -3, 0};
        const int32_t logical_expected[4] = {1, 0, 0, 1};
        const int32_t clip_input[5] = {INT_MIN, -2, 0, 5, INT_MAX};
        const int32_t clip_expected[5] = {-1, -1, 0, 4, 4};
        const int32_t condition[4] = {0, 1, -1, 0};
        const int32_t where_a[4] = {10, 20, 30, 40};
        const int32_t where_b[4] = {1, 2, 3, 4};
        const int32_t where_expected[4] = {1, 20, 30, 4};
        const int32_t cast_i32_input[4] = {
            INT_MIN, -3, 7, INT_MAX,
        };
        const float cast_f32_input[6] = {
            1.9f, -2.9f, 2147483648.0f, 4294967296.0f, NAN, INFINITY,
        };
        const int32_t cast_i32_expected[6] = {
            1, -2, INT_MIN, 0, 0, 0,
        };
        const int8_t cast_i8_input[4] = {-128, -1, 0, 127};
        const float cast_i8_expected[4] = {-128, -1, 0, 127};
        const uint8_t cast_u8_input[4] = {0, 127, 128, 255};
        const int32_t cast_u8_expected[4] = {0, 127, 128, 255};
        const int32_t cast_i8_source[4] = {-129, -1, 128, 257};
        const int8_t cast_i8_wrapped_expected[4] = {127, -1, -128, 1};
        const float cast_u8_source[3] = {-1.9f, 258.9f, NAN};
        const uint8_t cast_u8_wrapped_expected[3] = {255, 2, 0};
        int32_t logical_output[4] = {0};
        int32_t clipped[5] = {0};
        int32_t selected[4] = {0};
        float cast_f32_output[4] = {0};
        int32_t cast_i32_output[6] = {0};
        float cast_i8_output[4] = {0};
        int32_t cast_u8_output[4] = {0};
        int8_t cast_i8_wrapped[4] = {0};
        uint8_t cast_u8_wrapped[3] = {0};
        CHECK(cuda_graph_not_i32(
              logical_input, logical_output, 4));
        CHECK(cuda_graph_clip_i32(
              clip_input, clipped, 5, -1, 4));
        CHECK(cuda_graph_where_32(
              condition, where_a, where_b, selected, 4));
        CHECK(cuda_graph_cast_i32_f32(
              cast_i32_input, cast_f32_output, 4, 1));
        CHECK(cuda_graph_cast_i32_f32(
              cast_f32_input, cast_i32_output, 6, 0));
        CHECK(cuda_graph_cast_typed(
              cast_i8_input, VX_DTYPE_I8,
              cast_i8_output, VX_DTYPE_F32, 4));
        CHECK(cuda_graph_cast_typed(
              cast_u8_input, VX_DTYPE_U8,
              cast_u8_output, VX_DTYPE_I32, 4));
        CHECK(cuda_graph_cast_typed(
              cast_i8_source, VX_DTYPE_I32,
              cast_i8_wrapped, VX_DTYPE_I8, 4));
        CHECK(cuda_graph_cast_typed(
              cast_u8_source, VX_DTYPE_F32,
              cast_u8_wrapped, VX_DTYPE_U8, 3));
        /* Canonical F4 has value 2, which was the removed private I8 code. */
        CHECK(!cuda_graph_cast_typed(
              cast_i8_input, VX_DTYPE_F4,
              cast_i8_output, VX_DTYPE_F32, 4));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(
              logical_output, sizeof(logical_output), 0));
        CHECK(cuda_graph_sync_host(clipped, sizeof(clipped), 0));
        CHECK(cuda_graph_sync_host(selected, sizeof(selected), 0));
        CHECK(cuda_graph_sync_host(
              cast_f32_output, sizeof(cast_f32_output), 0));
        CHECK(cuda_graph_sync_host(
              cast_i32_output, sizeof(cast_i32_output), 0));
        CHECK(cuda_graph_sync_host(
              cast_i8_output, sizeof(cast_i8_output), 0));
        CHECK(cuda_graph_sync_host(
              cast_u8_output, sizeof(cast_u8_output), 0));
        CHECK(cuda_graph_sync_host(
              cast_i8_wrapped, sizeof(cast_i8_wrapped), 0));
        CHECK(cuda_graph_sync_host(
              cast_u8_wrapped, sizeof(cast_u8_wrapped), 0));
        for (int index = 0; index < 4; index++) {
            CHECK(logical_output[index] == logical_expected[index]);
            CHECK(selected[index] == where_expected[index]);
            CHECK(cast_f32_output[index] == (float)cast_i32_input[index]);
            CHECK(cast_i8_output[index] == cast_i8_expected[index]);
            CHECK(cast_u8_output[index] == cast_u8_expected[index]);
            CHECK(cast_i8_wrapped[index] ==
                  cast_i8_wrapped_expected[index]);
        }
        for (int index = 0; index < 5; index++)
            CHECK(clipped[index] == clip_expected[index]);
        for (int index = 0; index < 6; index++)
            CHECK(cast_i32_output[index] == cast_i32_expected[index]);
        for (int index = 0; index < 3; index++)
            CHECK(cast_u8_wrapped[index] ==
                  cast_u8_wrapped_expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float a[6] = {-2.0f, 1.0f, 4.0f, 3.0f, -5.0f, 2.0f};
        const float b[3] = {1.0f, 2.0f, -3.0f};
        const uint32_t output_strides[2] = {3, 1};
        const uint32_t a_strides[2] = {3, 1};
        const uint32_t b_strides[2] = {0, 1};
        const float expected[6] = {0.0f, 3.0f, 1.0f, 4.0f, 0.0f, 0.0f};
        float sum[6] = {0};
        float output[6] = {0};
        CHECK(cuda_graph_binary_f32(a, 6, b, 3, sum, 6, output_strides,
                                    a_strides, b_strides, 2, 3));
        CHECK(cuda_graph_relu_f32(sum, output, 6));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int i = 0; i < 6; i++) CHECK(near(output[i], expected[i], 1.0e-5f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float a[4] = {-8.0f, -2.0f, 1.0f, 5.0f};
        const float b[4] = {1.0f, 3.0f, 4.0f, 4.0f};
        const float expected[4] = {-7.0f, 1.0f, 5.0f, 9.0f};
        const float relu_expected[4] = {0.0f, 1.0f, 5.0f, 9.0f};
        const float relu6_expected[4] = {0.0f, 1.0f, 5.0f, 6.0f};
        float output[4] = {0};
        float relu[4] = {0};
        float relu6[4] = {0};
        CHECK(cuda_graph_add_relu_f32(a, b, output, 4, 0));
        CHECK(cuda_graph_add_relu_f32(a, b, relu, 4, 1));
        CHECK(cuda_graph_add_relu_f32(a, b, relu6, 4, 2));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(cuda_graph_sync_host(relu, sizeof(relu), 0));
        CHECK(cuda_graph_sync_host(relu6, sizeof(relu6), 0));
        for (int index = 0; index < 4; index++) {
            CHECK(output[index] == expected[index]);
            CHECK(relu[index] == relu_expected[index]);
            CHECK(relu6[index] == relu6_expected[index]);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float a[4] = {16777216.0f, -5.0f, 10.0f, 1.0f};
        const float b[4] = {-16777216.0f, 1.0f, 0.0f, 2.0f};
        const float c[4] = {1.0f, 2.0f, -8.0f, 4.0f};
        const float expected[4] = {1.0f, -2.0f, 2.0f, 7.0f};
        const float relu6_expected[4] = {1.0f, 0.0f, 2.0f, 6.0f};
        float output[4] = {0};
        float relu6[4] = {0};
        float overlap[5] = {0};
        uint64_t add3_before = cuda_test_add3_relu_launch_count();
        CHECK(!cuda_graph_add3_relu_f32(NULL, b, c, output, 4, 0));
        CHECK(!cuda_graph_add3_relu_f32(a, b, c, output, 0, 0));
        CHECK(!cuda_graph_add3_relu_f32(a, b, c, output, 4, 3));
        CHECK(!cuda_graph_add3_relu_f32(overlap, b, c, overlap + 1, 4, 0));
        CHECK(cuda_graph_add3_relu_f32(a, b, c, output, 4, 0));
        CHECK(cuda_graph_add3_relu_f32(a, b, c, relu6, 4, 2));
        CHECK(cuda_test_add3_relu_launch_count() - add3_before == 2u);
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(cuda_graph_sync_host(relu6, sizeof(relu6), 0));
        for (int index = 0; index < 4; index++) {
            CHECK(f32_bits(output[index]) == f32_bits(expected[index]));
            CHECK(f32_bits(relu6[index]) == f32_bits(relu6_expected[index]));
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float a[4] = {8.0f, -6.0f, 9.0f, 4.0f};
        const float b[4] = {2.0f, 3.0f, -3.0f, 0.5f};
        const float multiply_expected[4] = {16.0f, -18.0f, -27.0f, 2.0f};
        const float subtract_expected[4] = {6.0f, -9.0f, 12.0f, 3.5f};
        const float divide_expected[4] = {4.0f, -2.0f, -3.0f, 8.0f};
        const uint32_t strides[1] = {1};
        float multiplied[4] = {0};
        float subtracted[4] = {0};
        float divided[4] = {0};
        CHECK(cuda_graph_binary_f32(a, 4, b, 4, multiplied, 4,
                                    strides, strides, strides, 1, 0));
        CHECK(cuda_graph_binary_f32(a, 4, b, 4, subtracted, 4,
                                    strides, strides, strides, 1, 1));
        CHECK(cuda_graph_binary_f32(a, 4, b, 4, divided, 4,
                                    strides, strides, strides, 1, 2));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(multiplied, sizeof(multiplied), 0));
        CHECK(cuda_graph_sync_host(subtracted, sizeof(subtracted), 0));
        CHECK(cuda_graph_sync_host(divided, sizeof(divided), 0));
        for (int index = 0; index < 4; index++) {
            CHECK(multiplied[index] == multiply_expected[index]);
            CHECK(subtracted[index] == subtract_expected[index]);
            CHECK(divided[index] == divide_expected[index]);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[5] = {-4.0f, -1.0f, 0.0f, 2.0f, 6.0f};
        const float hardswish_expected[5] = {
            0.0f, -1.0f / 3.0f, 0.0f, 5.0f / 3.0f, 6.0f,
        };
        const float clip_expected[5] = {-1.0f, -1.0f, 0.0f, 2.0f, 2.0f};
        float sigmoid[5] = {0};
        float silu[5] = {0};
        float hyperbolic_tangent[5] = {0};
        float hardswish[5] = {0};
        float clipped[5] = {0};
        CHECK(cuda_graph_sigmoid_f32(input, sigmoid, 5));
        CHECK(cuda_graph_silu_f32(input, silu, 5));
        CHECK(cuda_graph_tanh_f32(input, hyperbolic_tangent, 5));
        CHECK(cuda_graph_hardswish_f32(input, hardswish, 5));
        CHECK(cuda_graph_clip_f32(input, clipped, 5, -1.0f, 2.0f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(sigmoid, sizeof(sigmoid), 0));
        CHECK(cuda_graph_sync_host(silu, sizeof(silu), 0));
        CHECK(cuda_graph_sync_host(hyperbolic_tangent,
                                   sizeof(hyperbolic_tangent), 0));
        CHECK(cuda_graph_sync_host(hardswish, sizeof(hardswish), 0));
        CHECK(cuda_graph_sync_host(clipped, sizeof(clipped), 0));
        for (int index = 0; index < 5; index++) {
            const float sigmoid_expected =
                1.0f / (1.0f + expf(-input[index]));
            CHECK(near(sigmoid[index], sigmoid_expected, 2.0e-3f));
            CHECK(near(silu[index], input[index] * sigmoid_expected,
                       2.0e-3f));
            CHECK(near(hyperbolic_tangent[index], tanhf(input[index]),
                       2.0e-3f));
            CHECK(near(hardswish[index], hardswish_expected[index],
                       1.0e-5f));
            CHECK(clipped[index] == clip_expected[index]);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[4] = {3.0f, 4.0f, 0.0f, -2.0f};
        const float weight[2] = {2.0f, 0.5f};
        const float expected[4] = {
            6.0f / sqrtf(13.5f), 2.0f / sqrtf(13.5f),
            0.0f, -1.0f / sqrtf(3.0f),
        };
        float output[4] = {0};
        CHECK(cuda_graph_rmsnorm_f32(input, weight, output, 2, 2, 1.0f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 4; index++)
            CHECK(near(output[index], expected[index], 5.0e-4f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        /* NHWC input with two channel groups.  Each group has variance 5;
         * epsilon 4 makes the normalization divisor exactly 3. */
        const float input[8] = {
            1.0f, 3.0f, 2.0f, 4.0f,
            5.0f, 7.0f, 6.0f, 8.0f,
        };
        const float weight[4] = {1.0f, 2.0f, 0.5f, -1.0f};
        const float bias[4] = {0.0f, 1.0f, -1.0f, 2.0f};
        const float expected[8] = {
            -1.0f, 1.0f / 3.0f, -1.5f, 7.0f / 3.0f,
             1.0f / 3.0f, 3.0f, -5.0f / 6.0f, 1.0f,
        };
        float output[8] = {0};
        CHECK(cuda_graph_groupnorm_f32(input, weight, bias, output,
                                       1, 1, 2, 4, 2, 4.0f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 8; index++)
            CHECK(near(output[index], expected[index], 5.0e-4f));
    }

    cuda_graph_reset();
    {
        enum {
            GROUP_BATCH = 2,
            GROUP_HEIGHT = 1,
            GROUP_WIDTH = 17,
            GROUP_CHANNELS = 10,
            GROUP_GROUPS = 2,
            GROUP_ELEMENTS = GROUP_BATCH * GROUP_HEIGHT * GROUP_WIDTH *
                GROUP_CHANNELS,
        };
        float input[GROUP_ELEMENTS];
        float weight[GROUP_CHANNELS];
        float expected[GROUP_ELEMENTS];
        float first[GROUP_ELEMENTS];
        float second[GROUP_ELEMENTS];
        float* outputs[2] = {first, second};
        const int area = GROUP_HEIGHT * GROUP_WIDTH;
        const int channels_per_group = GROUP_CHANNELS / GROUP_GROUPS;

        for (int index = 0; index < GROUP_ELEMENTS; ++index)
            input[index] =
                ((float)((index * 17 + 5) % 43) - 21.0f) * 0.03125f;
        for (int channel = 0; channel < GROUP_CHANNELS; ++channel)
            weight[channel] =
                ((float)((channel * 7 + 3) % 13) - 6.0f) * 0.125f;
        for (int batch_index = 0; batch_index < GROUP_BATCH; ++batch_index) {
            for (int group = 0; group < GROUP_GROUPS; ++group) {
                const int channel_start = group * channels_per_group;
                const int count = area * channels_per_group;
                float sum = 0.0f;
                float square_sum = 0.0f;
                for (int spatial = 0; spatial < area; ++spatial) {
                    const int base = (batch_index * area + spatial) *
                        GROUP_CHANNELS + channel_start;
                    for (int local = 0; local < channels_per_group; ++local) {
                        const float value = input[base + local];
                        sum += value;
                        square_sum += value * value;
                    }
                }
                const float mean = sum / (float)count;
                const float variance = fmaxf(
                    0.0f, square_sum / (float)count - mean * mean);
                const float inverse = 1.0f / sqrtf(variance + 1.0e-4f);
                for (int spatial = 0; spatial < area; ++spatial) {
                    const int base = (batch_index * area + spatial) *
                        GROUP_CHANNELS + channel_start;
                    for (int local = 0; local < channels_per_group; ++local) {
                        const int channel = channel_start + local;
                        expected[base + local] =
                            (input[base + local] - mean) * inverse *
                            weight[channel];
                    }
                }
            }
        }
        for (int run = 0; run < 2; ++run) {
            if (run) cuda_graph_reset();
            memset(outputs[run], 0, sizeof(first));
            cuda_graph_begin_forward();
            CHECK(cuda_graph_groupnorm_f32(
                input, weight, NULL, outputs[run], GROUP_BATCH, GROUP_HEIGHT,
                GROUP_WIDTH, GROUP_CHANNELS, GROUP_GROUPS, 1.0e-4f));
            CHECK(cuda_graph_end_forward() == 0);
            CHECK(cuda_graph_sync_host(
                outputs[run], sizeof(first), 0));
        }
        for (int index = 0; index < GROUP_ELEMENTS; ++index) {
            CHECK(f32_bits(first[index]) == f32_bits(second[index]));
            CHECK(near(first[index], expected[index], 5.0e-4f));
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        float one = 1.0f;
        float output = -7.0f;
        CHECK(!cuda_graph_groupnorm_f32(
            &one, &one, NULL, &output, INT_MAX, 1, 1, 2, 2, 1.0e-4f));
        CHECK(output == -7.0f);
        CHECK(cuda_graph_end_forward() == 0);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[4] = {0.0f, 0.0f, 0.0f, logf(3.0f)};
        const float expected[4] = {
            -logf(2.0f), -logf(2.0f), -logf(4.0f), logf(0.75f),
        };
        float output[4] = {0};
        CHECK(cuda_graph_logsoftmax_f32(input, output, 2, 2));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 4; index++)
            CHECK(near(output[index], expected[index], 2.0e-3f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[6] = {1.0f, 2.0f, 3.0f, -3.0f, 5.0f, 1.0f};
        const float sum_expected[2] = {6.0f, 3.0f};
        const float mean_expected[2] = {2.0f, 1.0f};
        float sum[2] = {0};
        float mean[2] = {0};
        CHECK(cuda_graph_reduce_f32(input, sum, 2, 3, 1.0f));
        CHECK(cuda_graph_reduce_f32(input, mean, 2, 3, 1.0f / 3.0f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(sum, sizeof(sum), 0));
        CHECK(cuda_graph_sync_host(mean, sizeof(mean), 0));
        for (int index = 0; index < 2; index++) {
            CHECK(sum[index] == sum_expected[index]);
            CHECK(mean[index] == mean_expected[index]);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[4] = {1.0f, 6.0f, 5.0f, 2.0f};
        const float weight[2] = {2.0f, -1.0f};
        const float bias[2] = {1.0f, 3.0f};
        const float mean[2] = {3.0f, 4.0f};
        const float variance[2] = {3.0f, 0.0f};
        const float expected[4] = {-1.0f, 1.0f, 3.0f, 5.0f};
        float output[4] = {0};
        CHECK(cuda_graph_batchnorm2d_f32(input, weight, bias, mean,
                                         variance, output, 1, 1, 2, 2,
                                         1.0f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 4; index++)
            CHECK(near(output[index], expected[index], 5.0e-4f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[8] = {
            1.0f, 10.0f, 3.0f, 20.0f,
            5.0f, 30.0f, 7.0f, 40.0f,
        };
        const float expected[2] = {4.0f, 25.0f};
        float output[2] = {0};
        CHECK(cuda_graph_global_average_pool_f32(input, output,
                                                 1, 2, 2, 2));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 2; index++)
            CHECK(output[index] == expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int32_t tokens[3] = {2, 0, 1};
        const float weight[6] = {1.0f, 2.0f, 3.0f,
                                 4.0f, 5.0f, 6.0f};
        const float expected[6] = {5.0f, 6.0f, 1.0f,
                                   2.0f, 3.0f, 4.0f};
        float output[6] = {0};
        CHECK(cuda_graph_embedding_f32(tokens, weight, output, 3, 2, 3));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 6; index++)
            CHECK(output[index] == expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[4] = {-2.0f, 0.0f, 3.5f, 7.0f};
        float copied[4] = {0};
        float aliased[4] = {0};
        CHECK(cuda_graph_copy_f32(input, copied, 4));
        CHECK(cuda_graph_alias_f32(copied, aliased, 4));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(aliased, sizeof(aliased), 0));
        for (int index = 0; index < 4; index++)
            CHECK(aliased[index] == input[index]);
    }

    /* An exact host pointer keeps one stable hash entry while its device
     * allocation grows. Exact synchronization must bypass the containing-view
     * scan, and reset must clear stale entries before the same host is reused. */
    cuda_graph_reset();
    {
        const float small[2] = {1.0f, 2.0f};
        const float grown[4] = {3.0f, 4.0f, 5.0f, 6.0f};
        const float after_reset[4] = {-1.0f, -2.0f, -3.0f, -4.0f};
        float storage[4] = {0};
        uint64_t lookup_before;
        uint64_t probe_before;
        uint64_t scan_before;

        cuda_graph_begin_forward();
        CHECK(cuda_graph_copy_f32(small, storage, 2));
        CHECK(cuda_graph_end_forward() == 0);
        cuda_graph_begin_forward();
        CHECK(cuda_graph_copy_f32(grown, storage, 4));
        CHECK(cuda_graph_end_forward() == 0);
        lookup_before = cuda_test_slot_exact_lookup_count();
        probe_before = cuda_test_slot_hash_probe_count();
        scan_before = cuda_test_slot_containing_scan_count();
        CHECK(cuda_graph_sync_host(storage, sizeof(storage), 0));
        CHECK(cuda_test_slot_exact_lookup_count() - lookup_before == 1u);
        CHECK(cuda_test_slot_hash_probe_count() > probe_before);
        CHECK(cuda_test_slot_containing_scan_count() == scan_before);
        for (int index = 0; index < 4; index++)
            CHECK(storage[index] == grown[index]);

        cuda_graph_reset();
        cuda_graph_begin_forward();
        CHECK(cuda_graph_copy_f32(after_reset, storage, 4));
        CHECK(cuda_graph_end_forward() == 0);
        lookup_before = cuda_test_slot_exact_lookup_count();
        probe_before = cuda_test_slot_hash_probe_count();
        scan_before = cuda_test_slot_containing_scan_count();
        CHECK(cuda_graph_sync_host(storage, sizeof(storage), 0));
        CHECK(cuda_test_slot_exact_lookup_count() - lookup_before == 1u);
        CHECK(cuda_test_slot_hash_probe_count() > probe_before);
        CHECK(cuda_test_slot_containing_scan_count() == scan_before);
        for (int index = 0; index < 4; index++)
            CHECK(storage[index] == after_reset[index]);
    }

    /* A view without its own exact slot chooses the smallest containing
     * allocation. An exact child that is too short must still fall through to
     * the full scan so a larger parent can satisfy the requested range. */
    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float child_values[2] = {101.0f, 102.0f};
        const float parent_values[8] = {
            10.0f, 11.0f, 12.0f, 13.0f,
            14.0f, 15.0f, 16.0f, 17.0f,
        };
        float storage[8] = {0};
        float smallest = 0.0f;
        float parent_view[4] = {0};
        uint64_t lookup_before;
        uint64_t probe_before;
        uint64_t scan_before;

        CHECK(cuda_graph_copy_f32(child_values, storage + 2, 2));
        CHECK(cuda_graph_copy_f32(parent_values, storage, 8));
        /* Pre-register both destinations so their later output lookups are
         * exact and do not obscure the input-view scan assertions. */
        CHECK(cuda_graph_copy_f32(parent_values, &smallest, 1));
        CHECK(cuda_graph_copy_f32(parent_values, parent_view, 4));

        lookup_before = cuda_test_slot_exact_lookup_count();
        probe_before = cuda_test_slot_hash_probe_count();
        scan_before = cuda_test_slot_containing_scan_count();
        CHECK(cuda_graph_copy_f32(storage + 3, &smallest, 1));
        CHECK(cuda_test_slot_exact_lookup_count() - lookup_before == 2u);
        CHECK(cuda_test_slot_hash_probe_count() > probe_before);
        CHECK(cuda_test_slot_containing_scan_count() > scan_before);

        lookup_before = cuda_test_slot_exact_lookup_count();
        probe_before = cuda_test_slot_hash_probe_count();
        scan_before = cuda_test_slot_containing_scan_count();
        CHECK(cuda_graph_copy_f32(storage + 2, parent_view, 4));
        CHECK(cuda_test_slot_exact_lookup_count() - lookup_before == 2u);
        CHECK(cuda_test_slot_hash_probe_count() > probe_before);
        CHECK(cuda_test_slot_containing_scan_count() > scan_before);

        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(&smallest, sizeof(smallest), 0));
        CHECK(cuda_graph_sync_host(parent_view, sizeof(parent_view), 0));
        CHECK(smallest == child_values[1]);
        for (int index = 0; index < 4; index++)
            CHECK(parent_view[index] == parent_values[index + 2]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float first[2] = {0.0f, logf(3.0f)};
        const float second[1] = {-logf(3.0f)};
        const float* inputs[2] = {first, second};
        const long sizes[2] = {2, 1};
        const float expected[3] = {0.5f, 0.75f, 0.25f};
        float output[3] = {0};
        CHECK(cuda_graph_concat_sigmoid_flat_f32(inputs, sizes, 2, output));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 3; index++)
            CHECK(near(output[index], expected[index], 2.0e-3f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[8] = {1.0f, 2.0f, 4.0f, 8.0f,
                                -2.0f, -1.0f, 1.0f, 2.0f};
        const float weight[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const float bias[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        float expected[8];
        float output[8] = {0};
        reference_layernorm(input, expected, 2, 4, 1.0e-5f);
        CHECK(cuda_graph_layernorm_f32(input, weight, bias, output,
                                        2, 4, 1.0e-5f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int i = 0; i < 8; i++) CHECK(near(output[i], expected[i], 5.0e-4f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[3] = {f32_from_bits(0x3fcd72a9u), 0.0f, -0.0f};
        float hard_sigmoid[3] = {0};
        float leaky_relu[3] = {0};
        float canonical = input[0] + 3.0f;
        canonical = canonical < 0.0f ? 0.0f : canonical;
        canonical = canonical > 6.0f ? 6.0f : canonical;
        canonical /= 6.0f;
        CHECK(cuda_graph_hardsigmoid_f32(input, hard_sigmoid, 3));
        CHECK(cuda_graph_leaky_relu_f32(input, leaky_relu, 3, -0.25f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(hard_sigmoid, sizeof(hard_sigmoid), 0));
        CHECK(cuda_graph_sync_host(leaky_relu, sizeof(leaky_relu), 0));
        CHECK(f32_bits(hard_sigmoid[0]) == f32_bits(canonical));
        CHECK(f32_bits(leaky_relu[1]) == 0x80000000u);
        CHECK(f32_bits(leaky_relu[2]) == 0x00000000u);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[6] = {1.0f, 2.0f, 3.0f, 3.0f, 1.0f, -2.0f};
        float output[6] = {0};
        CHECK(cuda_graph_softmax_f32(input, output, 2, 3));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int row = 0; row < 2; row++) {
            float sum = output[row * 3] + output[row * 3 + 1] + output[row * 3 + 2];
            CHECK(near(sum, 1.0f, 2.0e-3f));
            CHECK(output[row * 3] > 0.0f && output[row * 3 + 1] > 0.0f &&
                  output[row * 3 + 2] > 0.0f);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[4] = {0.0f, -INFINITY, -100.0f, -1.0f};
        float output[4] = {0};
        CHECK(cuda_graph_softmax_f32(input, output, 1, 4));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(output[0] > output[3]);
        CHECK(output[1] == 0.0f);
        CHECK(output[2] == 0.0f);
        CHECK(near(output[0] + output[3], 1.0f, 2.0e-3f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[7] = {-3.0f, -1.0f, -0.25f, 0.0f,
                                0.5f, 2.0f, 4.0f};
        float output[7] = {0};
        CHECK(cuda_graph_gelu_f32(input, output, 7, 0));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 7; index++) {
            float cdf = 0.5f * (1.0f + reference_qgelu_erf(
                input[index] * 0.7071067811865476f));
            CHECK(near(output[index], input[index] * cdf, 3.0e-4f));
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        /* NHWC/HWIO regular convolution and NHWC/HWIM depthwise
         * convolution.  The three regular outputs exercise no activation,
         * fused ReLU, and fused ReLU6 from the same device inputs. */
        const float input[8] = {1.0f, 2.0f, 3.0f, 4.0f,
                                -1.0f, 5.0f, 10.0f, -2.0f};
        const float regular_weight[4] = {1.0f, -1.0f, 2.0f, 0.5f};
        const float regular_bias[2] = {0.0f, 0.0f};
        const float regular_expected[8] = {
            5.0f, 0.0f, 11.0f, -1.0f, 9.0f, 3.5f, 6.0f, -11.0f,
        };
        const float relu_expected[8] = {
            5.0f, 0.0f, 11.0f, 0.0f, 9.0f, 3.5f, 6.0f, 0.0f,
        };
        const float relu6_expected[8] = {
            5.0f, 0.0f, 6.0f, 0.0f, 6.0f, 3.5f, 6.0f, 0.0f,
        };
        const float depthwise_weight[4] = {2.0f, -1.0f, 0.5f, 3.0f};
        const float depthwise_bias[4] = {1.0f, 0.0f, 0.0f, -1.0f};
        const float depthwise_expected[16] = {
            3.0f, -1.0f, 1.0f, 5.0f,
            7.0f, -3.0f, 2.0f, 11.0f,
            -1.0f, 1.0f, 2.5f, 14.0f,
            21.0f, -10.0f, -1.0f, -7.0f,
        };
        const float fallback_weight[16] = {
            1.0f, -1.0f, 0.5f, 2.0f,
            -0.5f, 0.25f, 1.5f, -2.0f,
            2.0f, 0.5f, -1.0f, 1.0f,
            0.25f, -0.75f, 2.5f, 0.125f,
        };
        const float fallback_bias[2] = {0.25f, -0.5f};
        float regular[8] = {0};
        float relu[8] = {0};
        float relu6[8] = {0};
        float depthwise[16] = {0};
        float fallback[2] = {0};
        float fallback_expected[2] = {fallback_bias[0], fallback_bias[1]};
        uint64_t tiled_before =
            cuda_test_conv2d_1x1_tiled_launch_count();
        for (int spatial = 0; spatial < 4; spatial++) {
            for (int channel = 0; channel < 2; channel++) {
                for (int output_channel = 0; output_channel < 2;
                     output_channel++) {
                    fallback_expected[output_channel] +=
                        input[spatial * 2 + channel] *
                        fallback_weight[(spatial * 2 + channel) * 2 +
                                        output_channel];
                }
            }
        }
        CHECK(cuda_graph_conv2d_f32(input, regular, regular_weight,
              regular_bias, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0,
              1, 0, 1, 1));
        CHECK(cuda_graph_conv2d_f32(input, relu, regular_weight,
              regular_bias, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0,
              1, 1, 1, 1));
        CHECK(cuda_graph_conv2d_f32(input, relu6, regular_weight,
              regular_bias, 1, 2, 2, 2, 2, 1, 1, 2, 2, 1, 1, 0, 0,
              1, 2, 1, 1));
        CHECK(cuda_graph_conv2d_f32(input, depthwise, depthwise_weight,
              depthwise_bias, 1, 2, 2, 2, 4, 1, 1, 2, 2, 1, 1, 0, 0,
              2, 0, 1, 1));
        CHECK(cuda_graph_conv2d_f32(input, fallback, fallback_weight,
              fallback_bias, 1, 2, 2, 2, 2, 2, 2, 1, 1, 1, 1, 0, 0,
              1, 0, 1, 1));
        CHECK(cuda_test_conv2d_1x1_tiled_launch_count() - tiled_before == 3u);
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(regular, sizeof(regular), 0));
        CHECK(cuda_graph_sync_host(relu, sizeof(relu), 0));
        CHECK(cuda_graph_sync_host(relu6, sizeof(relu6), 0));
        CHECK(cuda_graph_sync_host(depthwise, sizeof(depthwise), 0));
        CHECK(cuda_graph_sync_host(fallback, sizeof(fallback), 0));
        for (int index = 0; index < 8; index++) {
            CHECK(near(regular[index], regular_expected[index], 1.0e-5f));
            CHECK(near(relu[index], relu_expected[index], 1.0e-5f));
            CHECK(near(relu6[index], relu6_expected[index], 1.0e-5f));
        }
        for (int index = 0; index < 16; index++)
            CHECK(near(depthwise[index], depthwise_expected[index], 1.0e-5f));
        for (int index = 0; index < 2; index++)
            CHECK(near(fallback[index], fallback_expected[index], 1.0e-5f));
    }

    {
        float input[1200];
        float weight[600];
        float bias[32];

        /* C4 3x3, asymmetric left padding, and non-warp spatial/channel-vector
         * tails.  Channels 0 and 1 are order sentinels: changing bias-first or
         * ascending kx accumulation changes the exact zero result. */
        for (int index = 0; index < 3 * 7 * 12; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 3 * 3 * 12; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        for (int channel = 0; channel < 12; channel++)
            bias[channel] = (float)(channel % 5 - 2) * 0.5f;
        for (int spatial = 0; spatial < 3 * 7; spatial++) {
            input[spatial * 12] = 0.0f;
            input[spatial * 12 + 1] = 0.0f;
        }
        for (int tap = 0; tap < 9; tap++) {
            weight[tap * 12] = 0.0f;
            weight[tap * 12 + 1] = 0.0f;
        }
        bias[0] = 1.0f;
        input[0 * 12] = 1.0f;
        input[1 * 12] = 1.0f;
        weight[(1 * 3 + 0) * 12] = 16777216.0f;
        weight[(1 * 3 + 1) * 12] = -16777216.0f;
        bias[1] = 0.0f;
        input[0 * 12 + 1] = 1.0f;
        input[1 * 12 + 1] = 1.0f;
        input[2 * 12 + 1] = 1.0f;
        weight[(1 * 3 + 0) * 12 + 1] = 16777216.0f;
        weight[(1 * 3 + 1) * 12 + 1] = 1.0f;
        weight[(1 * 3 + 2) * 12 + 1] = -16777216.0f;
        CHECK(check_depthwise_case("3x3-c4-order", input, weight, bias,
              1, 3, 7, 12, 12, 3, 3, 3, 5, 1, 1, 1, 0, 0, 1, 1,
              DEPTHWISE_TACTIC_3X3_C4));

        /* The audited selector keeps this aligned C8 stride-2 shape on C1.
         * OH/OW encode asymmetric bottom/right SAME padding. */
        for (int index = 0; index < 4 * 6 * 8; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 3 * 3 * 8; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        CHECK(check_depthwise_case("3x3-c1-stride2", input, weight, NULL,
              1, 4, 6, 8, 8, 3, 3, 2, 3, 2, 2, 0, 0, 1, 1, 1,
              DEPTHWISE_TACTIC_3X3_C1));

        /* Every tap outside the only output's valid input window is poisoned.
         * A zero-padding multiply would turn the finite ReLU6 outputs into
         * NaNs, whereas the required OOB-tap skip leaves them exact. */
        for (int spatial = 0; spatial < 3 * 2; spatial++) {
            for (int channel = 0; channel < 12; channel++) {
                int lane = channel % 3;
                input[spatial * 12 + channel] =
                    lane == 0 ? 8.0f : (lane == 1 ? -2.0f : 2.0f);
            }
        }
        for (int index = 0; index < 5 * 5 * 12; index++) {
            if (index % 3 == 0) weight[index] = f32_from_bits(0x7fc12345u);
            else if (index % 3 == 1) weight[index] = INFINITY;
            else weight[index] = -INFINITY;
        }
        for (int ky = 1; ky <= 3; ky++) {
            for (int kx = 2; kx <= 3; kx++) {
                for (int channel = 0; channel < 12; channel++)
                    weight[(ky * 5 + kx) * 12 + channel] = 0.0f;
            }
        }
        for (int channel = 0; channel < 12; channel++) {
            weight[(1 * 5 + 2) * 12 + channel] = 1.0f;
            bias[channel] = channel % 3 == 2 ? 1.0f : 0.0f;
        }
        CHECK(check_depthwise_case("5x5-c4-padded-poison", input, weight,
              bias, 1, 3, 2, 12, 12, 5, 5, 1, 1, 1, 1, 1, 2, 2, 1, 1,
              DEPTHWISE_TACTIC_5X5_C4));

        /* K5 stride-2 stays vectorized.  Top/left one with the supplied
         * output shape leaves the EfficientDet-style two-cell bottom/right
         * tail, which the kernel handles by skipping those taps. */
        for (int index = 0; index < 8 * 10 * 8; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 5 * 5 * 8; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        CHECK(check_depthwise_case("5x5-c4-stride2", input, weight, NULL,
              1, 8, 10, 8, 8, 5, 5, 4, 5, 2, 2, 1, 1, 1, 1, 1,
              DEPTHWISE_TACTIC_5X5_C4));

        /* C%4 selects the unrolled C1 5x5 kernel; odd channels and spatial
         * dimensions cover both task and launch tails. */
        for (int index = 0; index < 9 * 10 * 7; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 5 * 5 * 7; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        CHECK(check_depthwise_case("5x5-c1-channel-tail", input, weight,
              NULL, 1, 9, 10, 7, 7, 5, 5, 5, 5, 2, 2, 2, 1, 0, 1, 1,
              DEPTHWISE_TACTIC_5X5_C1));

        /* Multiplier, dilation, unsupported kernel, and out-of-profile stride
         * remain on the general Conv2D kernel and must not move any depthwise
         * specialization counter. */
        for (int index = 0; index < 4 * 5 * 3; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 3 * 3 * 3 * 2; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        for (int channel = 0; channel < 6; channel++)
            bias[channel] = (float)(channel % 5 - 2) * 0.5f;
        CHECK(check_depthwise_case("generic-multiplier2", input, weight,
              bias, 1, 4, 5, 3, 6, 3, 3, 4, 3, 1, 1, 1, 0, 0, 1, 1,
              DEPTHWISE_TACTIC_GENERIC));

        for (int index = 0; index < 7 * 8 * 8; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 3 * 3 * 8; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        CHECK(check_depthwise_case("generic-dilation2", input, weight, NULL,
              1, 7, 8, 8, 8, 3, 3, 7, 6, 1, 1, 2, 1, 0, 2, 2,
              DEPTHWISE_TACTIC_GENERIC));

        for (int index = 0; index < 2 * 3 * 4; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int channel = 0; channel < 4; channel++) {
            weight[channel] = (float)(channel - 2) * 0.25f;
            bias[channel] = (float)(channel - 1) * 0.5f;
        }
        CHECK(check_depthwise_case("generic-kernel1", input, weight, bias,
              1, 2, 3, 4, 4, 1, 1, 2, 3, 1, 1, 0, 0, 0, 1, 1,
              DEPTHWISE_TACTIC_GENERIC));

        for (int index = 0; index < 7 * 8 * 4; index++)
            input[index] = (float)((index * 17) % 15 - 7) * 0.25f;
        for (int index = 0; index < 3 * 3 * 4; index++)
            weight[index] = (float)((index * 13) % 9 - 4) * 0.125f;
        CHECK(check_depthwise_case("generic-stride3", input, weight, NULL,
              1, 7, 8, 4, 4, 3, 3, 2, 2, 3, 3, 0, 0, 0, 1, 1,
              DEPTHWISE_TACTIC_GENERIC));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        enum { ROWS = 17, INPUT_CHANNELS = 19, OUTPUT_CHANNELS = 23 };
        float input[ROWS * INPUT_CHANNELS];
        float weight[INPUT_CHANNELS * OUTPUT_CHANNELS];
        float bias[OUTPUT_CHANNELS];
        float output_bias[ROWS * OUTPUT_CHANNELS] = {0};
        float output_relu6[ROWS * OUTPUT_CHANNELS] = {0};
        float expected_bias[ROWS * OUTPUT_CHANNELS];
        float expected_relu6[ROWS * OUTPUT_CHANNELS];
        uint64_t tiled_before =
            cuda_test_conv2d_1x1_tiled_launch_count();
        for (int index = 0; index < ROWS * INPUT_CHANNELS; index++)
            input[index] = (float)((index * 17) % 29 - 14) * 0.03125f;
        for (int index = 0; index < INPUT_CHANNELS * OUTPUT_CHANNELS;
             index++)
            weight[index] = (float)((index * 13) % 31 - 15) * 0.015625f;
        for (int column = 0; column < OUTPUT_CHANNELS; column++)
            bias[column] = (float)(column % 7 - 3) * 0.125f;
        for (int row = 0; row < ROWS; row++) {
            for (int column = 0; column < OUTPUT_CHANNELS; column++) {
                float with_bias = bias[column];
                float without_bias = 0.0f;
                for (int k = 0; k < INPUT_CHANNELS; k++) {
                    float product = input[row * INPUT_CHANNELS + k] *
                        weight[k * OUTPUT_CHANNELS + column];
                    with_bias += product;
                    without_bias += product;
                }
                expected_bias[row * OUTPUT_CHANNELS + column] = with_bias;
                if (without_bias < 0.0f) without_bias = 0.0f;
                if (without_bias > 6.0f) without_bias = 6.0f;
                expected_relu6[row * OUTPUT_CHANNELS + column] = without_bias;
            }
        }
        CHECK(cuda_graph_conv2d_f32(input, output_bias, weight, bias,
              1, 1, ROWS, INPUT_CHANNELS, OUTPUT_CHANNELS, 1, 1,
              1, ROWS, 1, 1, 0, 0, 1, 0, 1, 1));
        CHECK(cuda_graph_conv2d_f32(input, output_relu6, weight, NULL,
              1, 1, ROWS, INPUT_CHANNELS, OUTPUT_CHANNELS, 1, 1,
              1, ROWS, 1, 1, 0, 0, 1, 2, 1, 1));
        CHECK(cuda_test_conv2d_1x1_tiled_launch_count() - tiled_before == 2u);
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output_bias, sizeof(output_bias), 0));
        CHECK(cuda_graph_sync_host(output_relu6, sizeof(output_relu6), 0));
        for (int index = 0; index < ROWS * OUTPUT_CHANNELS; index++) {
            CHECK(near(output_bias[index], expected_bias[index], 5.0e-5f));
            CHECK(near(output_relu6[index], expected_relu6[index], 5.0e-5f));
        }
    }

    /* Exact selector boundaries plus row/column/K tails.  These cases also
     * cover bias, no bias, ReLU, ReLU6, and the preserved 16x16 fallback. */
    CHECK(check_conv1x1_tactic(17, 16, 48, 2, 0,
          CONV1X1_TACTIC_BM32_BN32_BK16));
    CHECK(check_conv1x1_tactic(8, 19, 49, 1, 1,
          CONV1X1_TACTIC_BM16_BN64_BK16));
    CHECK(check_conv1x1_tactic(9, 15, 65, 0, 0,
          CONV1X1_TACTIC_BASELINE));
    CHECK(check_conv1x1_add_tactic(17, 16, 48, 2, 1,
          CONV1X1_TACTIC_BM32_BN32_BK16));
    CHECK(check_conv1x1_add_tactic(8, 19, 49, 1, 1,
          CONV1X1_TACTIC_BM16_BN64_BK16));
    CHECK(check_conv1x1_add_tactic(9, 15, 65, 0, 1,
          CONV1X1_TACTIC_BASELINE));

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[1] = {1.0f};
        const float weight[1] = {10.0f};
        const float residual[1] = {-8.0f};
        float output[1] = {0.0f};
        float overlap_input[3] = {1.0f, 2.0f, 3.0f};
        float overlap_weight[3] = {1.0f, 0.0f, 0.0f};
        float overlap_bias[3] = {0.0f, 0.0f, 0.0f};
        float overlap_residual[3] = {4.0f, 5.0f, 6.0f};
        CHECK(!cuda_graph_conv2d_add_f32(overlap_input,
              overlap_residual + 1, overlap_weight, NULL, overlap_residual,
              1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 1));
        CHECK(!cuda_graph_conv2d_add_f32(overlap_input,
              overlap_input + 1, overlap_weight, NULL, overlap_residual,
              1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 1));
        CHECK(!cuda_graph_conv2d_add_f32(overlap_input,
              overlap_weight, overlap_weight, NULL, overlap_residual,
              1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 1));
        CHECK(!cuda_graph_conv2d_add_f32(overlap_input,
              overlap_bias, overlap_weight, overlap_bias, overlap_residual,
              1, 1, 2, 1, 1, 1, 1, 1, 2, 1, 1, 0, 0, 1, 0, 1, 1));
        CHECK(cuda_graph_conv2d_add_f32(input, output, weight, NULL,
              residual, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 1, 2,
              1, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        /* The Conv ReLU6 epilogue runs before the residual add: 6 + -8. */
        CHECK(f32_bits(output[0]) == f32_bits(-2.0f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float pool_input[6] = {1.0f, 5.0f, 2.0f,
                                     -1.0f, 3.0f, 4.0f};
        const float resize_input[4] = {1.0f, 3.0f, 5.0f, 7.0f};
        const float nearest_expected[16] = {
            1.0f, 1.0f, 3.0f, 3.0f,
            1.0f, 1.0f, 3.0f, 3.0f,
            5.0f, 5.0f, 7.0f, 7.0f,
            5.0f, 5.0f, 7.0f, 7.0f,
        };
        const float bilinear_expected[16] = {
            1.0f, 1.5f, 2.5f, 3.0f,
            2.0f, 2.5f, 3.5f, 4.0f,
            4.0f, 4.5f, 5.5f, 6.0f,
            5.0f, 5.5f, 6.5f, 7.0f,
        };
        const float concat_a[2] = {1.0f, 10.0f};
        const float concat_b[4] = {2.0f, 3.0f, 20.0f, 30.0f};
        const float* concat_inputs[2] = {concat_a, concat_b};
        const long concat_sizes[2] = {2, 4};
        const int concat_axes[2] = {1, 2};
        const float concat_expected[6] = {
            1.0f, 2.0f, 3.0f, 10.0f, 20.0f, 30.0f,
        };
        float maximum[2] = {0};
        float average[2] = {0};
        float nearest[16] = {0};
        float bilinear[16] = {0};
        float concatenated[6] = {0};
        CHECK(cuda_graph_maxpool2d_f32(pool_input, maximum,
              1, 2, 3, 1, 1, 2, 2, 2, 1, 1, 0, 0));
        CHECK(cuda_graph_average_pool2d_f32(pool_input, average,
              1, 2, 3, 1, 1, 2, 2, 2, 1, 1, 0, 0));
        CHECK(cuda_graph_resize_f32(resize_input, nearest,
              1, 2, 2, 1, 4, 4, 0));
        CHECK(cuda_graph_resize_f32(resize_input, bilinear,
              1, 2, 2, 1, 4, 4, 1));
        CHECK(cuda_graph_concat_f32(concat_inputs, concat_sizes, concat_axes,
              2, concatenated, 3, 1, 0));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(maximum, sizeof(maximum), 0));
        CHECK(cuda_graph_sync_host(average, sizeof(average), 0));
        CHECK(cuda_graph_sync_host(nearest, sizeof(nearest), 0));
        CHECK(cuda_graph_sync_host(bilinear, sizeof(bilinear), 0));
        CHECK(cuda_graph_sync_host(concatenated, sizeof(concatenated), 0));
        CHECK(maximum[0] == 5.0f && maximum[1] == 5.0f);
        CHECK(average[0] == 2.0f && average[1] == 3.5f);
        for (int index = 0; index < 16; index++) {
            CHECK(near(nearest[index], nearest_expected[index], 1.0e-5f));
            CHECK(near(bilinear[index], bilinear_expected[index], 1.0e-5f));
        }
        for (int index = 0; index < 6; index++)
            CHECK(concatenated[index] == concat_expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float qkv[12] = {
            0.0f, 0.0f, 1.0f, 0.0f, 2.0f, 4.0f,
            0.0f, 0.0f, 0.0f, 1.0f, 6.0f, 8.0f,
        };
        const int32_t keep_first[2] = {1, 0};
        const int32_t keep_none[2] = {0, 0};
        float unmasked[4] = {0};
        float masked[4] = {0};
        float all_masked[4] = {1, 1, 1, 1};
        float ranged[4] = {0};
        CHECK(cuda_graph_sdpa_f32(qkv, NULL, 0, unmasked,
              2, 2, 1, 2, 1, 1.0f, 0, 0));
        CHECK(cuda_graph_sdpa_f32(qkv, keep_first, 2, masked,
              2, 2, 1, 2, 1, 1.0f, 0, 1));
        CHECK(cuda_graph_sdpa_f32(qkv, keep_none, 2, all_masked,
              2, 2, 1, 2, 1, 1.0f, 0, 1));
        CHECK(cuda_graph_sdpa_f32(qkv, NULL, 0, ranged,
              2, 2, 1, 2, 1, 1.0f, 0, 0));
        CHECK(cuda_graph_sdpa_range_f32(qkv, keep_first, 2, ranged,
              2, 2, 1, 2, 1, 1.0f, 0, 1, 1, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(unmasked, sizeof(unmasked), 0));
        CHECK(cuda_graph_sync_host(masked, sizeof(masked), 0));
        CHECK(cuda_graph_sync_host(all_masked, sizeof(all_masked), 0));
        CHECK(cuda_graph_sync_host(ranged, sizeof(ranged), 0));
        for (int token = 0; token < 2; token++) {
            CHECK(near(unmasked[token * 2], 4.0f, 2.0e-3f));
            CHECK(near(unmasked[token * 2 + 1], 6.0f, 2.0e-3f));
            CHECK(masked[token * 2] == 2.0f);
            CHECK(masked[token * 2 + 1] == 4.0f);
            CHECK(all_masked[token * 2] == 0.0f);
            CHECK(all_masked[token * 2 + 1] == 0.0f);
        }
        CHECK(near(ranged[0], 4.0f, 2.0e-3f));
        CHECK(near(ranged[1], 6.0f, 2.0e-3f));
        CHECK(ranged[2] == 2.0f);
        CHECK(ranged[3] == 4.0f);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t input[4] = {-2, 1, 3, 4};
        const int8_t weight[8] = {1, 2, -1, 1, 2, 0, 1, -2};
        const float weight_scales[2] = {0.5f, 0.5f};
        const float tiny_scales[2] = {0x1p-149f, 0x1p-149f};
        const int32_t weight_zero_points[2] = {0, 0};
        const int32_t bias[2] = {0, 0};
        int8_t output[2] = {0};
        CHECK(!cuda_graph_qlinear_i8u8(input, weight, tiny_scales,
            weight_zero_points, bias, output, 1, 4, 2,
            0x1p-149f, 0, 1.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_qlinear_i8u8(input, weight, weight_scales,
            weight_zero_points, bias, output, 1, 4, 2,
            0.5f, 0, 0.25f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(output[0] == 1);
        CHECK(output[1] == -9);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t input[1] = {0};
        const int8_t weight[1] = {0};
        const float weight_scales[1] = {0.9499295353889465f};
        const int32_t weight_zero_points[1] = {0};
        const int32_t bias[1] = {-497898};
        int8_t output[1] = {0};
        /* This lands exactly on -45.5 only with the canonical
         * (input_scale * weight_scale) / output_scale schedule. */
        CHECK(cuda_graph_qlinear_i8u8(input, weight, weight_scales,
            weight_zero_points, bias, output, 1, 1, 1,
            1.991038334381301e-5f, 0, 0.20696647465229034f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(output[0] == -46);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t input[1] = {0};
        const int8_t weight[1] = {0};
        const float weight_scales[1] = {f32_from_bits(0x3bb4b15eu)};
        const int32_t weight_zero_points[1] = {0};
        const int32_t bias[1] = {-9611382};
        int8_t output[1] = {0};
        CHECK(cuda_graph_qlinear_i8u8(input, weight, weight_scales,
            weight_zero_points, bias, output, 1, 1, 1,
            f32_from_bits(0x38bcdf9eu), 0, f32_from_bits(0x3d30ef8eu),
            48, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(output[0] == -62);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t a[1] = {-9};
        const int8_t b[1] = {120};
        int8_t output[1] = {0};
        CHECK(cuda_graph_qadd_i8u8(a, 1, b, 1, output, 1,
            f32_from_bits(0x3dc70f9au), 0, f32_from_bits(0x3705bf15u), 0,
            f32_from_bits(0x3be7d981u), 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, 0));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(output[0] == -123);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int32_t tokens[2] = {1, 0};
        const uint8_t weight[6] = {126, 128, 130, 10, 12, 14};
        const float weight_scales[2] = {0.5f, 0.25f};
        const int32_t weight_zero_points[2] = {128, 12};
        const uint8_t expected[6] = {98, 100, 102, 96, 100, 104};
        uint8_t output[6] = {0};
        CHECK(cuda_graph_qembedding_i8u8(tokens, weight, weight_scales,
            weight_zero_points, output, 2, 2, 3, 0.25f, 100,
            VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 6; index++)
            CHECK(output[index] == expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t a[3] = {-10, 10, 100};
        const uint8_t b[3] = {100, 100, 100};
        const uint8_t expected_relu[3] = {128, 138, 228};
        const uint8_t expected_relu6[3] = {128, 134, 134};
        uint8_t relu[3] = {0};
        uint8_t relu6[3] = {0};
        CHECK(cuda_graph_qadd_i8u8(a, 3, b, 3, relu, 3,
            1.0f, 0, 1.0f, 100, 1.0f, 128,
            VX_DTYPE_I8, VX_DTYPE_U8, VX_DTYPE_U8, 1));
        CHECK(cuda_graph_qadd_i8u8(a, 3, b, 3, relu6, 3,
            1.0f, 0, 1.0f, 100, 1.0f, 128,
            VX_DTYPE_I8, VX_DTYPE_U8, VX_DTYPE_U8, 2));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(relu, sizeof(relu), 0));
        CHECK(cuda_graph_sync_host(relu6, sizeof(relu6), 0));
        for (int index = 0; index < 3; index++) {
            CHECK(relu[index] == expected_relu[index]);
            CHECK(relu6[index] == expected_relu6[index]);
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        /* Grouped U8/I8 convolution with asymmetric zero points, per-output
         * scales, optional I32 bias, and an output whose length is not word
         * aligned. */
        const uint8_t input[4] = {129, 126, 131, 132};
        const int8_t weight[2] = {2, -3};
        const float weight_scales[2] = {0.5f, 0.25f};
        const int32_t weight_zero_points[2] = {0, 0};
        const int32_t bias[2] = {0, 4};
        const uint8_t expected[4] = {101, 102, 103, 98};
        uint8_t output[4] = {0};
        CHECK(cuda_graph_qconv2d_i8u8(input, weight, weight_scales,
              weight_zero_points, bias, output,
              1u, 1u, 2u, 2u, 1u, 2u, 2u, 1u, 1u, 1u,
              1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 2u, 0u,
              0.25f, 128, 0.25f, 100,
              VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int index = 0; index < 4; index++)
            CHECK(output[index] == expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t group_input[3] = {-5, 0, 4};
        const uint8_t layer_input[6] = {123, 128, 132, 134, 126, 129};
        const float gamma[3] = {1.0f, 0.5f, -0.75f};
        const float beta[3] = {0.25f, -0.5f, 0.75f};
        const uint8_t expected_group[3] = {120, 124, 127};
        const int8_t expected_layer[6] = {-11, -7, -4, 10, -11, 4};
        uint8_t group_output[3] = {0};
        int8_t layer_output[6] = {0};
        CHECK(cuda_graph_qgroupnorm_i8u8(group_input, gamma, beta,
              group_output, 1u, 1u, 1u, 3u, 1u, 0.5f, -1,
              0.125f, 128, 1.0e-5f, VX_DTYPE_I8, VX_DTYPE_U8));
        CHECK(cuda_graph_qlayernorm_i8u8(layer_input, gamma, beta,
              layer_output, 2u, 3u, 0.5f, 127, 0.125f, -3,
              1.0e-5f, VX_DTYPE_U8, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(group_output, sizeof(group_output), 0));
        CHECK(cuda_graph_sync_host(layer_output, sizeof(layer_output), 0));
        for (int index = 0; index < 3; index++)
            CHECK(group_output[index] == expected_group[index]);
        for (int index = 0; index < 6; index++)
            CHECK(layer_output[index] == expected_layer[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t query[8] = {0, -1, -2, 1, -2, 1, 0, -1};
        const int8_t key[8] = {0, 0, -1, -1, -1, 0, 0, -2};
        const int8_t value[8] = {3, -3, 0, -1, -5, 1, 2, -2};
        const uint8_t unsigned_query[4] = {129, 126, 131, 128};
        const uint8_t unsigned_key[8] = {121, 120, 119, 122,
                                         119, 123, 120, 118};
        const uint8_t unsigned_value[8] = {134, 126, 132, 130,
                                           128, 132, 136, 128};
        const int32_t keep_none[2] = {0, 0};
        const uint8_t expected[8] = {128, 128, 130, 128,
                                     128, 128, 130, 127};
        uint8_t output[8] = {0};
        uint8_t all_masked[4] = {0};
        uint8_t ranged[8] = {0};
        CHECK(cuda_graph_qsdpa_i8u8(query, key, value, NULL, output,
              1u, 2u, 2u, 4u, 1u, 0.25f, -1, 0.25f, -1,
              0.25f, -1, 0.25f, 128, 0.5f,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_U8,
              0u, 0u));
        CHECK(cuda_graph_qsdpa_i8u8(unsigned_query, unsigned_key,
              unsigned_value, keep_none, all_masked,
              1u, 1u, 2u, 4u, 1u, 0.25f, 128, 0.5f, 120,
              0.25f, 130, 0.25f, 127, 0.5f,
              VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
              0u, 1u));
        CHECK(cuda_graph_qsdpa_i8u8(query, key, value, NULL, ranged,
              1u, 2u, 2u, 4u, 1u, 0.25f, -1, 0.25f, -1,
              0.25f, -1, 0.25f, 128, 0.5f,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_U8,
              0u, 0u));
        CHECK(cuda_graph_qsdpa_range_i8u8(query, key, value, keep_none,
              ranged, 1u, 2u, 2u, 4u, 1u, 0.25f, -1, 0.25f, -1,
              0.25f, -1, 0.25f, 128, 0.5f,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_U8,
              0u, 1u, 1u, 1u));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        CHECK(cuda_graph_sync_host(all_masked, sizeof(all_masked), 0));
        CHECK(cuda_graph_sync_host(ranged, sizeof(ranged), 0));
        for (int index = 0; index < 8; index++)
            CHECK(output[index] == expected[index]);
        for (int index = 0; index < 4; index++)
            CHECK(all_masked[index] == 127u);
        for (int index = 0; index < 4; index++)
            CHECK(ranged[index] == expected[index]);
        for (int index = 4; index < 8; index++)
            CHECK(ranged[index] == 128u);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t argmax_i8[12] = {
            -5, 4, -1, 4, -1, 3,
            -128, 0, -127, 1, -126, 1,
        };
        const uint8_t argmax_u8[6] = {130, 3, 129, 255, 130, 254};
        const int32_t expected_i8[4] = {1, 0, 2, 1};
        const int32_t expected_u8[2] = {0, 1};
        const int8_t mean_i8[24] = {
            -3, -1, 1, 1, 1, -5, -3, -1, 99, 98, 97, 96,
            40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51,
        };
        const int32_t mean_mask_i8[6] = {1, 1, 0, 0, 0, 0};
        const int8_t expected_mean_i8[8] = {6, 5, 6, 6, 5, 5, 5, 5};
        const uint8_t mean_u8[6] = {130, 134, 20, 20, 131, 129};
        const int32_t mean_mask_u8[3] = {1, 0, 1};
        const uint8_t expected_mean_u8[2] = {132, 134};
        int32_t output_i8[4] = {0};
        int32_t output_u8[2] = {0};
        int8_t mean_output_i8[8] = {0};
        uint8_t mean_output_u8[2] = {0};
        CHECK(cuda_graph_qargmax_i8u8(argmax_i8, output_i8,
              2u, 3u, 2u, VX_DTYPE_I8));
        CHECK(cuda_graph_qargmax_i8u8(argmax_u8, output_u8,
              1u, 3u, 2u, VX_DTYPE_U8));
        CHECK(cuda_graph_qmaskedmean_i8u8(mean_i8, mean_mask_i8,
              mean_output_i8, 2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
              VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_qmaskedmean_i8u8(mean_u8, mean_mask_u8,
              mean_output_u8, 1u, 3u, 2u, 0.25f, 128, 0.25f, 130,
              VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output_i8, sizeof(output_i8), 0));
        CHECK(cuda_graph_sync_host(output_u8, sizeof(output_u8), 0));
        CHECK(cuda_graph_sync_host(mean_output_i8, sizeof(mean_output_i8), 0));
        CHECK(cuda_graph_sync_host(mean_output_u8, sizeof(mean_output_u8), 0));
        for (int index = 0; index < 4; index++)
            CHECK(output_i8[index] == expected_i8[index]);
        for (int index = 0; index < 2; index++) {
            CHECK(output_u8[index] == expected_u8[index]);
            CHECK(mean_output_u8[index] == expected_mean_u8[index]);
        }
        for (int index = 0; index < 8; index++)
            CHECK(mean_output_i8[index] == expected_mean_i8[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t a[2] = {1, 4};
        const int8_t b[4] = {2, 3, 5, 6};
        const void* inputs[2] = {a, b};
        const uint32_t input_elements[2] = {2u, 4u};
        const uint32_t input_axes[2] = {1u, 2u};
        const float scales[2] = {1.0f, 1.0f};
        const int32_t zero_points[2] = {0, 0};
        const uint32_t dtypes[2] = {VX_DTYPE_I8, VX_DTYPE_I8};
        const int8_t concat_expected[6] = {1, 2, 3, 4, 5, 6};
        const int8_t resize_expected[12] = {
            4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6,
        };
        const uint32_t transpose_shape[4] = {1u, 2u, 2u, 3u};
        const uint32_t transpose_permutation[4] = {0u, 2u, 3u, 1u};
        const uint32_t transpose_duplicate[4] = {0u, 2u, 2u, 1u};
        const uint8_t transpose_u8_input[12] = {
            117, 118, 119, 120, 121, 122,
            123, 124, 125, 126, 127, 128,
        };
        const uint8_t transpose_u8_expected[12] = {
            117, 123, 118, 124, 119, 125,
            120, 126, 121, 127, 122, 128,
        };
        const int8_t transpose_i8_input[12] = {
            -6, -5, -4, -3, -2, -1,
            0, 1, 2, 3, 4, 5,
        };
        const int8_t transpose_i8_expected[12] = {
            -6, 0, -5, 1, -4, 2,
            -3, 3, -2, 4, -1, 5,
        };
        const int8_t negative = -5;
        const uint8_t unsigned_pool_input[2] = {250, 5};
        int8_t concatenated[6] = {0};
        int8_t pooled[3] = {0};
        int8_t resized[12] = {0};
        int8_t padded_pool[4] = {0};
        uint8_t unsigned_pool = 0;
        uint8_t transpose_u8_output[12] = {0};
        int8_t transpose_i8_output[12] = {0};
        CHECK(cuda_graph_concat_i8u8(inputs, input_elements, input_axes,
              scales, zero_points, dtypes, 2u, concatenated,
              6u, 3u, 1u, 1.0f, 0, VX_DTYPE_I8));
        CHECK(cuda_graph_maxpool2d_i8u8(concatenated, pooled,
              1u, 2u, 1u, 3u, 1u, 1u, 2u, 1u, 1u, 1u,
              0u, 0u, 0u, 0u, 1.0f, 0, 1.0f, 0,
              VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_resize_nearest_i8u8(pooled, resized,
              1u, 1u, 1u, 3u, 2u, 2u,
              1.0f, 0, 1.0f, 0, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_transpose_i8u8(
              transpose_u8_input, transpose_u8_output, transpose_shape,
              transpose_permutation, 4u, 12u, 0.125f, 123,
              0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(cuda_graph_transpose_i8u8(
              transpose_i8_input, transpose_i8_output, transpose_shape,
              transpose_permutation, 4u, 12u, 0.25f, -3,
              0.25f, -3, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_transpose_i8u8(
              transpose_u8_input, transpose_u8_output, transpose_shape,
              transpose_duplicate, 4u, 12u, 0.125f, 123,
              0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(!cuda_graph_transpose_i8u8(
              transpose_u8_input, transpose_u8_output, transpose_shape,
              transpose_permutation, 4u, 12u, 0.125f, 123,
              0.25f, 123, VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(!cuda_graph_transpose_i8u8(
              transpose_u8_output, transpose_u8_output, transpose_shape,
              transpose_permutation, 4u, 12u, 0.125f, 123,
              0.125f, 123, VX_DTYPE_U8, VX_DTYPE_U8));
        /* Padding must not introduce signed-domain zero, and U8 max pooling
         * must order 250 above 5 rather than interpreting it as -6. */
        CHECK(cuda_graph_maxpool2d_i8u8(&negative, padded_pool,
              1u, 1u, 1u, 1u, 2u, 2u, 2u, 2u, 1u, 1u,
              1u, 1u, 1u, 1u, 1.0f, 0, 1.0f, 0,
              VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_maxpool2d_i8u8(unsigned_pool_input, &unsigned_pool,
              1u, 1u, 2u, 1u, 1u, 1u, 1u, 2u, 1u, 1u,
              0u, 0u, 0u, 0u, 1.0f, 128, 1.0f, 128,
              VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(concatenated, sizeof(concatenated), 0));
        CHECK(cuda_graph_sync_host(pooled, sizeof(pooled), 0));
        CHECK(cuda_graph_sync_host(resized, sizeof(resized), 0));
        CHECK(cuda_graph_sync_host(padded_pool, sizeof(padded_pool), 0));
        CHECK(cuda_graph_sync_host(&unsigned_pool, sizeof(unsigned_pool), 0));
        CHECK(cuda_graph_sync_host(
              transpose_u8_output, sizeof(transpose_u8_output), 0));
        CHECK(cuda_graph_sync_host(
              transpose_i8_output, sizeof(transpose_i8_output), 0));
        for (int index = 0; index < 6; index++)
            CHECK(concatenated[index] == concat_expected[index]);
        CHECK(pooled[0] == 4 && pooled[1] == 5 && pooled[2] == 6);
        for (int index = 0; index < 12; index++)
            CHECK(resized[index] == resize_expected[index]);
        for (int index = 0; index < 4; index++)
            CHECK(padded_pool[index] == -5);
        CHECK(unsigned_pool == 250u);
        CHECK(memcmp(transpose_u8_output, transpose_u8_expected,
                     sizeof(transpose_u8_output)) == 0);
        CHECK(memcmp(transpose_i8_output, transpose_i8_expected,
                     sizeof(transpose_i8_output)) == 0);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[9] = {NAN, -INFINITY, INFINITY, -12.75f, -0.5f,
                                0.5f, 1.5f, 2.5f, 12.75f};
        const int8_t expected[9] = {0, -128, 127, -13, 0, 0, 2, 2, 13};
        const uint8_t packed[3] = {0, 127, 255};
        const float expected_dequantized[3] = {-31.75f, 0.0f, 32.0f};
        int8_t quantized[9] = {0};
        float dequantized[3] = {0};
        CHECK(cuda_graph_quantize_typed_f32_i8u8(
              input, 9, quantized, 1.0f, 0, VX_DTYPE_I8));
        CHECK(cuda_graph_dequantize_typed_i8u8_f32(
              packed, 3, 0.25f, 127, VX_DTYPE_U8, dequantized));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(quantized, sizeof(quantized), 0));
        CHECK(cuda_graph_sync_host(dequantized, sizeof(dequantized), 0));
        for (int index = 0; index < 9; index++)
            CHECK(quantized[index] == expected[index]);
        for (int index = 0; index < 3; index++)
            CHECK(dequantized[index] == expected_dequantized[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        uint8_t input[256];
        uint8_t silu_output[4][256] = {{0}};
        uint8_t gelu_output[4][256] = {{0}};
        const float input_scale = 0.03125f;
        const float output_scale = 0.017f;
        const uint32_t byte_dtypes[2] = {VX_DTYPE_I8, VX_DTYPE_U8};
        for (uint32_t raw = 0; raw < 256; raw++) input[raw] = (uint8_t)raw;
        for (uint32_t combination = 0; combination < 4; combination++) {
            const uint32_t input_dtype = byte_dtypes[combination / 2u];
            const uint32_t output_dtype = byte_dtypes[combination % 2u];
            const int32_t input_zero_point =
                input_dtype == VX_DTYPE_I8 ? -3 : 129;
            const int32_t output_zero_point =
                output_dtype == VX_DTYPE_I8 ? 4 : 117;
            CHECK(cuda_graph_qsilu_i8u8(input, silu_output[combination], 256,
                input_scale, input_zero_point, output_scale, output_zero_point,
                input_dtype, output_dtype));
            CHECK(cuda_graph_qgelu_i8u8(input, gelu_output[combination], 256,
                input_scale, input_zero_point, output_scale, output_zero_point,
                input_dtype, output_dtype));
        }
        CHECK(cuda_graph_end_forward() == 0);
        for (uint32_t combination = 0; combination < 4; combination++) {
            const uint32_t input_dtype = byte_dtypes[combination / 2u];
            const uint32_t output_dtype = byte_dtypes[combination % 2u];
            const int32_t input_zero_point =
                input_dtype == VX_DTYPE_I8 ? -3 : 129;
            const int32_t output_zero_point =
                output_dtype == VX_DTYPE_I8 ? 4 : 117;
            CHECK(cuda_graph_sync_host(silu_output[combination], 256, 0));
            CHECK(cuda_graph_sync_host(gelu_output[combination], 256, 0));
            for (uint32_t raw = 0; raw < 256; raw++) {
                CHECK(silu_output[combination][raw] == reference_qactivation(
                    (uint8_t)raw, input_scale, input_zero_point, output_scale,
                    output_zero_point, input_dtype, output_dtype, 0));
                CHECK(gelu_output[combination][raw] == reference_qactivation(
                    (uint8_t)raw, input_scale, input_zero_point, output_scale,
                    output_zero_point, input_dtype, output_dtype, 1));
            }
        }
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t silu_input = 48;
        const int8_t overflow_input = -2;
        int8_t silu_output = 0;
        int8_t overflow_output = 0;
        CHECK(cuda_graph_qsilu_i8u8(&silu_input, &silu_output, 1,
            f32_from_bits(0x3d30b979u), 0, f32_from_bits(0x3f068539u),
            0, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_qsilu_i8u8(&overflow_input, &overflow_output, 1,
            FLT_MAX, 0, 0.5f, 7, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(&silu_output, sizeof(silu_output), 0));
        CHECK(cuda_graph_sync_host(&overflow_output,
                                   sizeof(overflow_output), 0));
        CHECK(silu_output == 3);
        CHECK(overflow_output == 7);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int8_t input[7] = {-128, -3, -1, 0, 1, 3, 127};
        int8_t requantized[7] = {0};
        const uint8_t packed[5] = {0x00u, 0x7fu, 0x80u, 0xfeu, 0xffu};
        uint8_t copied[5] = {0};
        const float input_scale = 0.3f;
        const float output_scale = 0.2f;
        const float multiplier = input_scale / output_scale;
        const int8_t boundary_input = -128;
        int8_t boundary_output = -1;
        CHECK(cuda_graph_requantize_linear_i8u8(input, 7, requantized, 7,
              input_scale, -2, output_scale, 5,
              VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_requantize_linear_i8u8(&boundary_input, 1,
              &boundary_output, 1, f32_from_bits(0x3efdfdfeu), 127,
              1.0f, 126, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_requantize_linear_i8u8(input, 7, requantized, 7,
              FLT_MAX, -2, FLT_MIN, 5, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(cuda_graph_copy_i8u8(packed, 5, copied, 5,
              FLT_MIN, 128, FLT_MIN, 128,
              VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(!cuda_graph_copy_i8u8(packed, 5, copied, 5,
              1.0f, 128, 0.5f, 128,
              VX_DTYPE_U8, VX_DTYPE_U8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(requantized, sizeof(requantized), 0));
        CHECK(cuda_graph_sync_host(copied, sizeof(copied), 0));
        CHECK(cuda_graph_sync_host(&boundary_output,
                                   sizeof(boundary_output), 0));
        for (int i = 0; i < 7; i++)
            CHECK(requantized[i] == reference_requantize_i8(
                  input[i], multiplier, -2, 5));
        for (int i = 0; i < 5; i++) CHECK(copied[i] == packed[i]);
        CHECK(boundary_output == 0);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        enum { large_transfer_elements = 320 * 320 * 3 };
        uint8_t* input = (uint8_t*)malloc(large_transfer_elements);
        uint8_t* output = (uint8_t*)malloc(large_transfer_elements);
        CHECK(input != NULL && output != NULL);
        for (size_t index = 0; index < large_transfer_elements; index++) {
            /* Avoid input zero so every expected I8 output differs from the
             * -128 value produced by an unwritten/zero device-input tail. */
            input[index] = (uint8_t)(index % 255u + 1u);
            output[index] = 0u;
        }
        CHECK(cuda_graph_requantize_linear_i8u8(
              input, large_transfer_elements, output, large_transfer_elements,
              0.0078125f, 127, 0.0078125f, -1,
              VX_DTYPE_U8, VX_DTYPE_I8));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, large_transfer_elements, 0));
        for (size_t index = 0; index < large_transfer_elements; index++)
            CHECK(output[index] == (uint8_t)((int)input[index] - 128));
        cuda_graph_reset();
        free(output);
        free(input);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float transpose_input[6] = {1, 2, 3, 4, 5, 6};
        const float transpose_expected[6] = {1, 4, 2, 5, 3, 6};
        const int transpose_shape[2] = {2, 3};
        const int transpose_permutation[2] = {1, 0};
        const int duplicate_permutation[2] = {0, 0};
        const float condition[4] = {0.0f, 1.0f, -1.0f, NAN};
        const float where_a[4] = {10, 20, 30, 40};
        const float where_b[4] = {1, 2, 3, 4};
        const float where_expected[4] = {1, 20, 30, 40};
        const float expand_input[2] = {1, 2};
        const float expand_expected[6] = {1, 1, 1, 2, 2, 2};
        const int expand_input_shape[2] = {2, 1};
        const int expand_output_shape[2] = {2, 3};
        const int invalid_expand_shape[2] = {3, 3};
        const float gather_input[12] = {
            1, 2, 3, 4, 5, 6,
            7, 8, 9, 10, 11, 12,
        };
        const int32_t gather_indices[4] = {2, 0, -1, 3};
        const float gather_expected[16] = {
            5, 6, 1, 2, 5, 6, -1, -1,
            11, 12, 7, 8, 11, 12, -1, -1,
        };
        const float pad_input[2] = {1, 2};
        const float pad_expected[12] = {
            -1, -1, -1, -1,
            -1,  1,  2, -1,
            -1, -1, -1, -1,
        };
        const int pad_input_shape[4] = {1, 1, 2, 1};
        const int pad_output_shape[4] = {1, 3, 4, 1};
        const float slice_input[6] = {1, 2, 3, 4, 5, 6};
        const float slice_expected[2] = {4, 6};
        const int slice_input_shape[4] = {1, 2, 3, 1};
        const int slice_output_shape[4] = {1, 1, 2, 1};
        const int slice_starts[4] = {0, 1, 0, 0};
        const int slice_steps[4] = {1, 1, 2, 1};
        const float split_input[6] = {1, 2, 3, 4, 5, 6};
        const float split_expected[4] = {2, 3, 5, 6};
        float transposed[6] = {0};
        float selected[4] = {0};
        float expanded[6] = {0};
        float gathered[16] = {0};
        float padded[12] = {0};
        float sliced[2] = {0};
        float split[4] = {0};
        CHECK(!cuda_graph_transpose_f32(transpose_input, transposed,
              transpose_shape, duplicate_permutation, 2));
        CHECK(!cuda_graph_transpose_f32(transpose_input,
              (float*)transpose_input, transpose_shape,
              transpose_permutation, 2));
        CHECK(cuda_graph_transpose_f32(transpose_input, transposed,
              transpose_shape, transpose_permutation, 2));
        CHECK(!cuda_graph_where_f32(condition, where_a, where_b,
              (float*)where_a, 4));
        CHECK(cuda_graph_where_f32(condition, where_a, where_b, selected, 4));
        CHECK(!cuda_graph_expand_f32(expand_input, expanded,
              expand_input_shape, 2, invalid_expand_shape, 2));
        CHECK(cuda_graph_expand_f32(expand_input, expanded,
              expand_input_shape, 2, expand_output_shape, 2));
        CHECK(cuda_graph_gather_i32_f32(gather_input, gather_indices,
              gathered, 2, 3, 2, 4, 16));
        CHECK(cuda_graph_pad4d_f32(pad_input, padded, pad_input_shape, 4,
              pad_output_shape, 4, 1, 1, -1.0f));
        CHECK(cuda_graph_slice4d_f32(slice_input, sliced, slice_input_shape, 4,
              slice_output_shape, 4, slice_starts, slice_steps));
        CHECK(!cuda_graph_split_f32(split_input, split, 6, 4, 1, 2, 3, 2));
        CHECK(cuda_graph_split_f32(split_input, split, 6, 4, 1, 2, 3, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(transposed, sizeof(transposed), 0));
        CHECK(cuda_graph_sync_host(selected, sizeof(selected), 0));
        CHECK(cuda_graph_sync_host(expanded, sizeof(expanded), 0));
        CHECK(cuda_graph_sync_host(gathered, sizeof(gathered), 0));
        CHECK(cuda_graph_sync_host(padded, sizeof(padded), 0));
        CHECK(cuda_graph_sync_host(sliced, sizeof(sliced), 0));
        CHECK(cuda_graph_sync_host(split, sizeof(split), 0));
        for (int i = 0; i < 6; i++) {
            CHECK(transposed[i] == transpose_expected[i]);
            CHECK(expanded[i] == expand_expected[i]);
        }
        for (int i = 0; i < 4; i++) CHECK(selected[i] == where_expected[i]);
        for (int i = 0; i < 16; i++) CHECK(gathered[i] == gather_expected[i]);
        for (int i = 0; i < 12; i++) CHECK(padded[i] == pad_expected[i]);
        for (int i = 0; i < 2; i++) CHECK(sliced[i] == slice_expected[i]);
        for (int i = 0; i < 4; i++) CHECK(split[i] == split_expected[i]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const int32_t copy_input[3] = {INT_MIN, 7, INT_MAX};
        const int32_t transpose_input[6] = {1, 2, 3, 4, 5, 6};
        const int32_t transpose_expected[6] = {1, 4, 2, 5, 3, 6};
        const int transpose_shape[2] = {2, 3};
        const int transpose_permutation[2] = {1, 0};
        const int32_t expand_input[2] = {7, 9};
        const int32_t expand_expected[6] = {7, 7, 7, 9, 9, 9};
        const int expand_input_shape[2] = {2, 1};
        const int expand_output_shape[2] = {2, 3};
        const int32_t scalar_expand_input = 42;
        const int scalar_expand_output_shape[3] = {2, 1, 3};
        const int32_t slice_input[6] = {1, 2, 3, 4, 5, 6};
        const int32_t slice_expected[2] = {4, 6};
        const int slice_input_shape[5] = {1, 1, 2, 3, 1};
        const int slice_output_shape[5] = {1, 1, 1, 2, 1};
        const int slice_starts[5] = {0, 0, 1, 0, 0};
        const int slice_steps[5] = {1, 1, 1, 2, 1};
        const int32_t split_input[6] = {1, 2, 3, 4, 5, 6};
        const int32_t split_expected[4] = {2, 3, 5, 6};
        const int32_t concat_a[2] = {1, 10};
        const int32_t concat_b[4] = {2, 3, 20, 30};
        const void* concat_inputs[2] = {concat_a, concat_b};
        const long concat_sizes[2] = {2, 4};
        const int concat_axes[2] = {1, 2};
        const int32_t concat_expected[6] = {1, 2, 3, 10, 20, 30};
        int32_t copied[3] = {0};
        int32_t transposed[6] = {0};
        int32_t expanded[6] = {0};
        int32_t scalar_expanded[6] = {0};
        int32_t sliced[2] = {0};
        int32_t split[4] = {0};
        int32_t concatenated[6] = {0};
        CHECK(cuda_graph_copy_32(copy_input, copied, 3));
        CHECK(cuda_graph_transpose_32(
              transpose_input, transposed, transpose_shape,
              transpose_permutation, 2));
        CHECK(cuda_graph_expand_32(
              expand_input, expanded, expand_input_shape, 2,
              expand_output_shape, 2));
        CHECK(cuda_graph_expand_32(
              &scalar_expand_input, scalar_expanded, NULL, 0,
              scalar_expand_output_shape, 3));
        CHECK(cuda_graph_slice_32(
              slice_input, sliced, slice_input_shape, 5,
              slice_output_shape, 5, slice_starts, slice_steps));
        CHECK(cuda_graph_split_32(
              split_input, split, 6, 4, 1, 2, 3, 1));
        CHECK(cuda_graph_concat_32(
              concat_inputs, concat_sizes, concat_axes, 2,
              concatenated, 3, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(copied, sizeof(copied), 0));
        CHECK(cuda_graph_sync_host(transposed, sizeof(transposed), 0));
        CHECK(cuda_graph_sync_host(expanded, sizeof(expanded), 0));
        CHECK(cuda_graph_sync_host(
              scalar_expanded, sizeof(scalar_expanded), 0));
        CHECK(cuda_graph_sync_host(sliced, sizeof(sliced), 0));
        CHECK(cuda_graph_sync_host(split, sizeof(split), 0));
        CHECK(cuda_graph_sync_host(
              concatenated, sizeof(concatenated), 0));
        for (int index = 0; index < 3; index++)
            CHECK(copied[index] == copy_input[index]);
        for (int index = 0; index < 6; index++) {
            CHECK(transposed[index] == transpose_expected[index]);
            CHECK(expanded[index] == expand_expected[index]);
            CHECK(scalar_expanded[index] == scalar_expand_input);
            CHECK(concatenated[index] == concat_expected[index]);
        }
        for (int index = 0; index < 2; index++)
            CHECK(sliced[index] == slice_expected[index]);
        for (int index = 0; index < 4; index++)
            CHECK(split[index] == split_expected[index]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float interp_input[4] = {0, 4, 10, 14};
        const float interp_expected[8] = {0, 1, 3, 4, 10, 11, 13, 14};
        const float conv1d_input[4] = {1, 2, 3, 4};
        const float conv1d_weight[3] = {1, 0, -1};
        const float conv1d_bias[1] = {-1};
        const float conv1d_expected[4] = {-3, -3, -3, 2};
        const float conv1d_relu_expected[4] = {0, 0, 0, 2};
        const float transpose_input[4] = {1, 2, 3, 4};
        const float transpose_weight[4] = {1, 1, 1, 1};
        const float transpose_bias[1] = {0.5f};
        const float transpose_expected[9] = {
            1.5f, 3.5f, 2.5f,
            4.5f, 10.5f, 6.5f,
            3.5f, 7.5f, 4.5f,
        };
        float interpolated[8] = {0};
        float convolved[4] = {0};
        float convolved_relu[4] = {0};
        float transposed[9] = {0};
        CHECK(cuda_graph_interp1d_f32(interp_input, interpolated, 2, 2, 4));
        CHECK(!cuda_graph_interp1d_f32(interp_input,
              (float*)interp_input, 2, 2, 2));
        CHECK(!cuda_graph_conv1d_f32(conv1d_input, conv1d_weight,
              conv1d_bias, convolved, 1, 1, 4, 1, 3, 3, 1, 1, 0));
        CHECK(cuda_graph_conv1d_f32(conv1d_input, conv1d_weight,
              conv1d_bias, convolved, 1, 1, 4, 1, 4, 3, 1, 1, 0));
        CHECK(cuda_graph_conv1d_f32(conv1d_input, conv1d_weight,
              conv1d_bias, convolved_relu, 1, 1, 4, 1, 4, 3, 1, 1, 1));
        CHECK(!cuda_graph_conv_transpose2d_f32(transpose_input,
              transpose_weight, transpose_bias, transposed,
              1, 2, 2, 1, 2, 3, 1, 2, 2, 1, 1, 0, 0));
        CHECK(cuda_graph_conv_transpose2d_f32(transpose_input,
              transpose_weight, transpose_bias, transposed,
              1, 2, 2, 1, 3, 3, 1, 2, 2, 1, 1, 0, 0));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(interpolated, sizeof(interpolated), 0));
        CHECK(cuda_graph_sync_host(convolved, sizeof(convolved), 0));
        CHECK(cuda_graph_sync_host(convolved_relu, sizeof(convolved_relu), 0));
        CHECK(cuda_graph_sync_host(transposed, sizeof(transposed), 0));
        for (int i = 0; i < 8; i++)
            CHECK(near(interpolated[i], interp_expected[i], 1.0e-5f));
        for (int i = 0; i < 4; i++) {
            CHECK(convolved[i] == conv1d_expected[i]);
            CHECK(convolved_relu[i] == conv1d_relu_expected[i]);
        }
        for (int i = 0; i < 9; i++)
            CHECK(transposed[i] == transpose_expected[i]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float quantize_input[10] = {
            NAN, -200, -100, -1, 0, 1, 1.5f, 2.5f, 100, 200,
        };
        const int8_t quantize_expected[10] = {
            0, -128, -100, -1, 0, 1, 2, 2, 100, 127,
        };
        const float dequantize_input[4] = {-2, 0, 2, 5};
        const float dequantize_scale[1] = {0.25f};
        const float dequantize_zero[1] = {1.0f};
        const float dequantize_expected[4] = {-0.75f, -0.25f, 0.25f, 1.0f};
        int8_t quantized[10] = {0};
        float dequantized[4] = {0};
        CHECK(!cuda_graph_quantize_linear_i8(quantize_input, quantized, 10,
              1.0f, 0, 0.0f, 0));
        CHECK(cuda_graph_quantize_linear_i8(quantize_input, quantized, 10,
              1.0f, 0, 1.0f, 0));
        CHECK(cuda_graph_dequantize_linear_f32(dequantize_input,
              dequantize_scale, dequantize_zero, dequantized, 4, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(quantized, sizeof(quantized), 0));
        CHECK(cuda_graph_sync_host(dequantized, sizeof(dequantized), 0));
        for (int i = 0; i < 10; i++)
            CHECK(quantized[i] == quantize_expected[i]);
        for (int i = 0; i < 4; i++)
            CHECK(dequantized[i] == dequantize_expected[i]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float query[2] = {0, 0};
        const float key[4] = {1, 0, 0, 1};
        const float value[4] = {2, 4, 6, 8};
        const int32_t keep_second[2] = {0, 1};
        const int32_t keep_none[2] = {0, 0};
        float unmasked[2] = {0};
        float masked[2] = {0};
        float all_masked[2] = {1, 1};
        CHECK(!cuda_graph_cross_sdpa_f32(query, key, value, keep_second, 1,
              masked, 1, 2, 2, 1, 2, 1, 1.0f, 0, 1));
        CHECK(cuda_graph_cross_sdpa_f32(query, key, value, NULL, 0,
              unmasked, 1, 2, 2, 1, 2, 1, 1.0f, 0, 0));
        CHECK(cuda_graph_cross_sdpa_f32(query, key, value, keep_second, 2,
              masked, 1, 2, 2, 1, 2, 1, 1.0f, 0, 1));
        CHECK(cuda_graph_cross_sdpa_f32(query, key, value, keep_none, 2,
              all_masked, 1, 2, 2, 1, 2, 1, 1.0f, 0, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(unmasked, sizeof(unmasked), 0));
        CHECK(cuda_graph_sync_host(masked, sizeof(masked), 0));
        CHECK(cuda_graph_sync_host(all_masked, sizeof(all_masked), 0));
        CHECK(near(unmasked[0], 4.0f, 2.0e-3f));
        CHECK(near(unmasked[1], 6.0f, 2.0e-3f));
        CHECK(masked[0] == 6.0f && masked[1] == 8.0f);
        CHECK(all_masked[0] == 0.0f && all_masked[1] == 0.0f);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float query[1] = {5};
        const float key_value[2] = {2, 4};
        const float weight[3] = {2, 0, 3};
        const float scale[3] = {0.5f, 1.0f, 2.0f};
        const float bias[3] = {1.0f, 0.0f, -1.0f};
        float plain[1] = {0};
        float affine[1] = {0};
        CHECK(!cuda_graph_cross_attention_f32(query, key_value, weight,
              NULL, bias, affine, 1, 2, 1, 1, 1, 1, 1, 1));
        CHECK(cuda_graph_cross_attention_f32(query, key_value, weight,
              NULL, NULL, plain, 1, 2, 1, 1, 1, 1, 0, 0));
        CHECK(cuda_graph_cross_attention_f32(query, key_value, weight,
              scale, bias, affine, 1, 2, 1, 1, 1, 1, 1, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(plain, sizeof(plain), 0));
        CHECK(cuda_graph_sync_host(affine, sizeof(affine), 0));
        CHECK(near(plain[0], 9.0f, 2.0e-3f));
        CHECK(near(affine[0], 17.0f, 2.0e-3f));
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float input[4] = {1, 3, 5, 7};
        const float profile_x_expected[4] = {5, 7, 3, 5};
        const float profile_y_expected[4] = {3, 7, 2, 6};
        const float mean_expected[2] = {3, 5};
        const float exponential = expf(-4.0f);
        const float soft_expected =
            (0.25f * exponential + 0.75f) / (exponential + 1.0f);
        float softargmax[2] = {0};
        float profile_x[4] = {0};
        float profile_y[4] = {0};
        float mean_height[2] = {0};
        CHECK(cuda_graph_spatial_softargmax_y_f32(input, softargmax,
              1, 2, 2, 1));
        CHECK(cuda_graph_profile_x_f32(input, profile_x, 1, 2, 2, 1));
        CHECK(cuda_graph_profile_y_f32(input, profile_y, 1, 2, 2, 1));
        CHECK(cuda_graph_mean_height_f32(input, mean_height, 1, 2, 2, 1));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(softargmax, sizeof(softargmax), 0));
        CHECK(cuda_graph_sync_host(profile_x, sizeof(profile_x), 0));
        CHECK(cuda_graph_sync_host(profile_y, sizeof(profile_y), 0));
        CHECK(cuda_graph_sync_host(mean_height, sizeof(mean_height), 0));
        CHECK(near(softargmax[0], soft_expected, 2.0e-3f));
        CHECK(near(softargmax[1], soft_expected, 2.0e-3f));
        for (int i = 0; i < 4; i++) {
            CHECK(profile_x[i] == profile_x_expected[i]);
            CHECK(profile_y[i] == profile_y_expected[i]);
        }
        for (int i = 0; i < 2; i++)
            CHECK(mean_height[i] == mean_expected[i]);
    }

    cuda_graph_reset();
    cuda_graph_begin_forward();
    {
        const float boxes[12] = {
            0, 0, 2, 2,
            0, 0, 2, 2,
            3, 3, 4, 4,
        };
        const float scores[3] = {0.9f, 0.8f, 0.7f};
        const float expected[12] = {
             0,  0,  0,
             0,  0,  2,
            -1, -1, -1,
            -1, -1, -1,
        };
        float output[12] = {0};
        CHECK(!cuda_graph_nms_f32(boxes, scores, output,
              1, 3, 1, 3, 4, 1.0f, 0.1f));
        CHECK(cuda_graph_nms_f32(boxes, scores, output,
              1, 3, 1, 3, 4, 0.5f, 0.1f));
        CHECK(cuda_graph_end_forward() == 0);
        CHECK(cuda_graph_sync_host(output, sizeof(output), 0));
        for (int i = 0; i < 12; i++) CHECK(output[i] == expected[i]);
    }

    cuda_graph_reset();
    {
        uint8_t alias[8] = {0};
        uint8_t other[8] = {0};
        union { float values[4]; uint8_t bytes[16]; } typed = {{0}};
        const float one_scale = 1.0f;
        const int32_t zero = 0;
        const int32_t token = 0;
        float distinct_output[4] = {0};
        uint8_t distinct_byte = 0;
        CHECK(!cuda_graph_qsilu_i8u8(alias, alias, 8,
                                     1.0f, 0, 1.0f, 0,
                                     VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_qgelu_i8u8(alias, alias, 8,
                                     1.0f, 0, 1.0f, 0,
                                     VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_qadd_i8u8(alias, 8, other, 8, alias, 8,
                                    1.0f, 0, 1.0f, 0, 1.0f, 0,
                                    VX_DTYPE_I8, VX_DTYPE_I8,
                                    VX_DTYPE_I8, 0));
        CHECK(!cuda_graph_quantize_typed_f32_i8u8(
              typed.values, 2, typed.bytes + 1, 1.0f, 0, VX_DTYPE_I8));
        CHECK(!cuda_graph_dequantize_typed_i8u8_f32(
              typed.bytes, 2, 1.0f, 0, VX_DTYPE_I8, typed.values));
        CHECK(!cuda_graph_requantize_linear_i8u8(
              alias, 4, alias + 1, 4, 1.0f, 0, 1.0f, 0,
              VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_copy_i8u8(
              alias, 4, alias + 1, 4, 1.0f, 0, 1.0f, 0,
              VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_qlinear_i8u8(alias, alias + 1, &one_scale,
              &zero, &zero, &distinct_byte, UINT32_MAX, 2, 1,
              1.0f, 0, 1.0f, 0,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!cuda_graph_layernorm_f32(typed.values, &one_scale,
              &one_scale, distinct_output, INT_MAX, 3, 1.0e-5f));
        CHECK(!cuda_graph_embedding_f32(&token, &one_scale,
              distinct_output, INT_MAX, 3, 1));
    }

    CHECK(cuda_test_caller_context_is_clear());
    cuda_cleanup();
    cuda_test_engine_scope_end(&engine_scope);
    puts("CUDA kernel correctness tests passed");
    return 0;
}
