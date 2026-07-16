#include "engine_internal.h"
#include "cpu_features.h"
#include "inference_kernels.h"
#include "qconv_w8a8_arm.h"
#include "qlinear_w8a8_arm.h"
#include "quant_cpu_opt.h"
#include "safetensors.h"
#include "thread_pool.h"
#include "volvoxai_backend.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#else
/* GPU parity cases use initialization failure as their ordinary skip path. */
static int vk_init(void) { return -1; }
static void vk_cleanup(void) {}
#endif
#include "volvoxai.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return -1; } } while (0)

static int closef(float left, float right) {
    return fabsf(left - right) <= 1.0e-6f * (1.0f + fabsf(left) + fabsf(right));
}

static int set_companion_scale_metadata(SafetensorsFile* file,
                                        const char* storage_format,
                                        const char* manifest_json) {
    cJSON* metadata;
    char* metadata_json;
    int result;
    if (!file || !storage_format || !manifest_json) return -1;
    metadata = cJSON_CreateObject();
    if (!metadata) return -1;
    cJSON_AddStringToObject(metadata, "weights_quantization_storage", storage_format);
    cJSON_AddStringToObject(metadata, "weights_quantization", manifest_json);
    metadata_json = cJSON_PrintUnformatted(metadata);
    cJSON_Delete(metadata);
    if (!metadata_json) return -1;
    result = safetensors_set_metadata_json(file, metadata_json);
    free(metadata_json);
    return result;
}

static int write_raw_safetensors(const char* path, const char* header,
                                 const void* data, size_t data_size) {
    unsigned char length_bytes[8];
    size_t header_size;
    uint64_t encoded_size;
    FILE* file;
    int result = 0;
    if (!path || !header || (!data && data_size)) return -1;
    header_size = strlen(header);
    encoded_size = (uint64_t)header_size;
    for (int index = 0; index < 8; index++) {
        length_bytes[index] = (unsigned char)(encoded_size & 0xffu);
        encoded_size >>= 8;
    }
    file = fopen(path, "wb");
    if (!file) return -1;
    if (fwrite(length_bytes, 1, sizeof(length_bytes), file) != sizeof(length_bytes) ||
        fwrite(header, 1, header_size, file) != header_size ||
        (data_size && fwrite(data, 1, data_size, file) != data_size)) result = -1;
    if (fclose(file) != 0) result = -1;
    return result;
}

static int test_safetensors_rejects_recursive_duplicate_header_keys(void) {
    const char* path = "/tmp/volvox-duplicate-safetensors-header.safetensors";
    const char* duplicate_tensor =
        "{\"w\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]},"
        "\"w\":{\"dtype\":\"I8\",\"shape\":[1],\"data_offsets\":[0,1]}}";
    const char* duplicate_metadata =
        "{\"__metadata__\":{\"weights_quantization\":\"{}\","
        "\"weights_quantization\":\"{}\"}}";
    const char* non_string_metadata =
        "{\"__metadata__\":{\"weights_quantization\":1}}";
    const int8_t byte = 1;
    SafetensorsFile file;
    memset(&file, 0, sizeof(file));
    CHECK(write_raw_safetensors(path, duplicate_tensor, &byte, sizeof(byte)) == 0);
    CHECK(safetensors_load(path, &file) != 0);
    safetensors_free(&file);
    CHECK(write_raw_safetensors(path, duplicate_metadata, NULL, 0) == 0);
    CHECK(safetensors_load(path, &file) != 0);
    safetensors_free(&file);
    CHECK(write_raw_safetensors(path, non_string_metadata, NULL, 0) == 0);
    CHECK(safetensors_load(path, &file) != 0);
    safetensors_free(&file);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_set_metadata_json(
              &file, "{\"weights_quantization\":1}") != 0);
    CHECK(safetensors_set_metadata_json(
              &file, "{\"weights_quantization\":\"{}\"}") == 0);
    safetensors_free(&file);
    remove(path);
    return 0;
}

static int test_native_cpu_feature_contract(void) {
    const int avx512_vnni = vx_cpu_has_avx512_vnni();
    CHECK(avx512_vnni == 0 || avx512_vnni == 1);
    CHECK(avx512_vnni == vx_cpu_has_avx512_vnni());
    /* The AVX-512 kernel gate deliberately requires AVX2 in addition to the
     * AVX-512F/BW/VL/VNNI and OS ZMM-state checks. */
    if (avx512_vnni) CHECK(vx_cpu_has_avx2());
    return 0;
}

static int test_groupnorm_optional_bias_kernel(void) {
    const float input[4] = {1.0f, 3.0f, 2.0f, 6.0f};
    const float weight[4] = {1.0f, 2.0f, 0.5f, -1.0f};
    float output[4] = {0};
    const double epsilon = 1.0e-5;
    const float first_inverse = (float)(1.0 / sqrt(1.0 + epsilon));
    const float second_inverse = (float)(1.0 / sqrt(4.0 + epsilon));
    CHECK(groupnorm_f32(input, weight, NULL, output, 1, 1, 1, 4, 2,
                        epsilon));
    CHECK(closef(output[0], -first_inverse));
    CHECK(closef(output[1], 2.0f * first_inverse));
    CHECK(closef(output[2], -second_inverse));
    CHECK(closef(output[3], -2.0f * second_inverse));
    return 0;
}

static int8_t quantize_i8(float value, float scale, int zp) {
    int q = (int)lrintf(value / scale + (float)zp);
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static int32_t qembedding_round_ties_even(float value) {
    int32_t lower = (int32_t)floorf(value);
    float fraction = value - (float)lower;
    if (fraction < 0.5f) return lower;
    if (fraction > 0.5f) return lower + 1;
    return lower % 2 == 0 ? lower : lower + 1;
}

static void qembedding_reference(const int32_t* ids, const void* weight,
                                 const float* scales, const int32_t* zero_points,
                                 void* output, uint32_t tokens, uint32_t hidden,
                                 float output_scale, int32_t output_zero_point,
                                 uint32_t weight_dtype, uint32_t output_dtype) {
    const int32_t minimum = output_dtype == 2u ? -128 : 0;
    const int32_t maximum = output_dtype == 2u ? 127 : 255;
    for (uint32_t token = 0; token < tokens; token++) {
        uint32_t row = (uint32_t)ids[token];
        for (uint32_t column = 0; column < hidden; column++) {
            size_t index = (size_t)row * hidden + column;
            int32_t raw = weight_dtype == 2u ? ((const int8_t*)weight)[index] :
                                                ((const uint8_t*)weight)[index];
            float dequantized = (float)(raw - zero_points[row]) * scales[row];
            float transformed = dequantized / output_scale + (float)output_zero_point;
            int32_t quantized;
            if (transformed != transformed) quantized = output_zero_point;
            else if (transformed <= (float)minimum) quantized = minimum;
            else if (transformed >= (float)maximum) quantized = maximum;
            else quantized = qembedding_round_ties_even(transformed);
            index = (size_t)token * hidden + column;
            if (output_dtype == 2u) ((int8_t*)output)[index] = (int8_t)quantized;
            else ((uint8_t*)output)[index] = (uint8_t)quantized;
        }
    }
}

static int test_qembedding_portable_kernel(void) {
    enum { tokens = 3, vocab = 3, hidden = 3 };
    const int32_t ids[tokens] = {1, 0, 2};
    const int32_t invalid_ids[tokens] = {1, 3, 0};
    const float scales[vocab] = {0.5f, 0.25f, 1.0f};
    const int32_t i8_zero_points[vocab] = {0, 1, -1};
    const int32_t u8_zero_points[vocab] = {128, 129, 127};
    const int8_t i8_weight[vocab * hidden] = {
        0, 1, -1,
        2, -2, 4,
        99, -101, 4,
    };
    const uint8_t u8_weight[vocab * hidden] = {
        128, 129, 127,
        130, 126, 132,
        227, 27, 132,
    };
    for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
        const void* weight = weight_kind ? (const void*)u8_weight : (const void*)i8_weight;
        const int32_t* zero_points = weight_kind ? u8_zero_points : i8_zero_points;
        const uint32_t weight_dtype = weight_kind ? 3u : 2u;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[tokens * hidden];
            int8_t expected_i8[tokens * hidden];
            uint8_t actual_u8[tokens * hidden];
            uint8_t expected_u8[tokens * hidden];
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            void* expected = output_kind ? (void*)expected_u8 : (void*)expected_i8;
            uint32_t output_dtype = output_kind ? 3u : 2u;
            int32_t output_zero_point = output_kind ? 128 : -3;
            memset(actual, 0x5a, tokens * hidden);
            memset(expected, 0, tokens * hidden);
            qembedding_reference(ids, weight, scales, zero_points, expected,
                                 tokens, hidden, 0.25f, output_zero_point,
                                 weight_dtype, output_dtype);
            CHECK(qembedding_i8u8(ids, weight, scales, zero_points, actual,
                                  tokens, vocab, hidden, 0.25f, output_zero_point,
                                  weight_dtype, output_dtype) == 1);
            CHECK(memcmp(actual, expected, tokens * hidden) == 0);
            memset(actual, 0x5a, tokens * hidden);
            CHECK(qembedding_i8u8(invalid_ids, weight, scales, zero_points, actual,
                                  tokens, vocab, hidden, 0.25f, output_zero_point,
                                  weight_dtype, output_dtype) == 0);
            for (int index = 0; index < tokens * hidden; index++) {
                CHECK(((const uint8_t*)actual)[index] == 0x5a);
            }
        }
    }
    return 0;
}

static void qsilu_reference(const void* input, void* output, uint32_t elements,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype) {
    const int32_t minimum = output_dtype == 2u ? -128 : 0;
    const int32_t maximum = output_dtype == 2u ? 127 : 255;
    for (uint32_t index = 0; index < elements; index++) {
        const int32_t raw = input_dtype == 2u ? ((const int8_t*)input)[index] :
                                                ((const uint8_t*)input)[index];
        const float x = (float)(raw - input_zero_point) * input_scale;
        const float silu = x / (1.0f + expf(-x));
        const float transformed = silu / output_scale + (float)output_zero_point;
        int32_t quantized;
        if (transformed != transformed) quantized = output_zero_point;
        else if (transformed <= (float)minimum) quantized = minimum;
        else if (transformed >= (float)maximum) quantized = maximum;
        else quantized = qembedding_round_ties_even(transformed);
        if (output_dtype == 2u) ((int8_t*)output)[index] = (int8_t)quantized;
        else ((uint8_t*)output)[index] = (uint8_t)quantized;
    }
}

static float qgelu_erf_reference(float value) {
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
    return sign * (1.0f - polynomial * expf(-(magnitude * magnitude)));
}

static void qgelu_reference(const void* input, void* output, uint32_t elements,
                            float input_scale, int32_t input_zero_point,
                            float output_scale, int32_t output_zero_point,
                            uint32_t input_dtype, uint32_t output_dtype) {
    const int32_t minimum = output_dtype == 2u ? -128 : 0;
    const int32_t maximum = output_dtype == 2u ? 127 : 255;
    for (uint32_t index = 0; index < elements; index++) {
        const int32_t raw = input_dtype == 2u ? ((const int8_t*)input)[index] :
                                                ((const uint8_t*)input)[index];
        const float value = (float)(raw - input_zero_point) * input_scale;
        const float erf_input = value * 0.7071067811865476f;
        const float cdf = 0.5f * (1.0f + qgelu_erf_reference(erf_input));
        const float transformed = (value * cdf) / output_scale +
            (float)output_zero_point;
        int32_t quantized;
        if (transformed != transformed) quantized = output_zero_point;
        else if (transformed <= (float)minimum) quantized = minimum;
        else if (transformed >= (float)maximum) quantized = maximum;
        else quantized = qembedding_round_ties_even(transformed);
        if (output_dtype == 2u) ((int8_t*)output)[index] = (int8_t)quantized;
        else ((uint8_t*)output)[index] = (uint8_t)quantized;
    }
}

/* Test every I8/U8 activation pairing and a five-byte tail.  The reference
 * deliberately uses the direct division spelling mandated by the shared
 * JS/WGSL/C contract so reciprocal-multiply substitutions cannot hide here. */
static int test_qsilu_portable_kernel(void) {
    const int8_t input_i8[5] = {-8, -2, 0, 2, 8};
    const uint8_t input_u8[5] = {120, 126, 128, 130, 136};
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zero_point = input_kind ? 128 : 0;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[5] = {0};
            int8_t expected_i8[5] = {0};
            uint8_t actual_u8[5] = {0};
            uint8_t expected_u8[5] = {0};
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            void* expected = output_kind ? (void*)expected_u8 : (void*)expected_i8;
            const uint32_t output_dtype = output_kind ? 3u : 2u;
            const int32_t output_zero_point = output_kind ? 128 : -3;
            qsilu_reference(input, expected, 5u, 0.5f, input_zero_point,
                            0.25f, output_zero_point, input_dtype, output_dtype);
            CHECK(qsilu_i8u8(input, actual, 5u, 0.5f, input_zero_point,
                              0.25f, output_zero_point,
                              input_dtype, output_dtype) == 1);
            CHECK(memcmp(actual, expected, sizeof(actual_i8)) == 0);
            memset(actual, 0x5a, sizeof(actual_i8));
            CHECK(qsilu_i8u8(input, actual, 5u, 0.0f, input_zero_point,
                              0.25f, output_zero_point,
                              input_dtype, output_dtype) == 0);
            for (int index = 0; index < 5; index++) {
                CHECK(((const uint8_t*)actual)[index] == 0x5a);
            }
        }
    }
    {
        int8_t alias[5] = {-8, -2, 0, 2, 8};
        const int8_t before[5] = {-8, -2, 0, 2, 8};
        CHECK(qsilu_i8u8(alias, alias, 5u, 0.5f, 0,
                          0.25f, -3, 2u, 2u) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
    }
    return 0;
}

/* These expected bytes are fixed from the A-S 7.1.26 evaluation order used
 * by gELU.wgsl, not native erff.  The five-byte vector covers a packed-word
 * tail and all four I8/U8 activation boundary combinations. */
static int test_qgelu_portable_kernel(void) {
    const int8_t input_i8[5] = {-8, -2, 0, 2, 8};
    const uint8_t input_u8[5] = {120, 126, 128, 130, 136};
    const int8_t expected_i8[5] = {-3, -4, -3, 4, 29};
    const uint8_t expected_u8[5] = {128, 127, 128, 135, 160};
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zero_point = input_kind ? 128 : 0;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[5] = {0};
            uint8_t actual_u8[5] = {0};
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            const uint32_t output_dtype = output_kind ? 3u : 2u;
            const int32_t output_zero_point = output_kind ? 128 : -3;
            const void* expected = output_kind ? (const void*)expected_u8 :
                                                 (const void*)expected_i8;
            CHECK(qgelu_i8u8(input, actual, 5u, 0.5f, input_zero_point,
                              0.125f, output_zero_point,
                              input_dtype, output_dtype) == 1);
            CHECK(memcmp(actual, expected, sizeof(actual_i8)) == 0);
            memset(actual, 0x5a, sizeof(actual_i8));
            CHECK(qgelu_i8u8(input, actual, 5u, 0.5f, input_zero_point,
                              0.0f, output_zero_point,
                              input_dtype, output_dtype) == 0);
            for (int index = 0; index < 5; index++) {
                CHECK(((const uint8_t*)actual)[index] == 0x5a);
            }
        }
    }
    {
        int8_t alias[5] = {-8, -2, 0, 2, 8};
        const int8_t before[5] = {-8, -2, 0, 2, 8};
        CHECK(qgelu_i8u8(alias, alias, 5u, 0.5f, 0,
                          0.125f, -3, 2u, 2u) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
    }
    return 0;
}

/* Exercise all 256 physical input bytes through the cold scalar route, the
 * table-building route, and a warm-cache route.  The 768-byte vector crosses
 * the kernel's build threshold and repeats every byte three times. */
static int test_quantized_activation_lut_exactness(void) {
    enum { byte_values = 256, large_elements = 768 };
    uint8_t input[byte_values];
    uint8_t large_input[large_elements];
    uint8_t expected[byte_values];
    uint8_t actual[byte_values];
    uint8_t large_actual[large_elements];
    for (uint32_t index = 0; index < byte_values; index++) input[index] = (uint8_t)index;
    for (uint32_t index = 0; index < large_elements; index++)
        large_input[index] = (uint8_t)index;
    for (uint32_t op = 0; op < 2u; op++) {
        for (uint32_t input_kind = 0; input_kind < 2u; input_kind++) {
            const uint32_t input_dtype = input_kind ? 3u : 2u;
            const int32_t input_zero_point = input_kind ? 121 : -7;
            const float input_scale = 0.03125f + (float)input_kind * 0.00390625f +
                (float)op * 0.0009765625f;
            for (uint32_t output_kind = 0; output_kind < 2u; output_kind++) {
                const uint32_t output_dtype = output_kind ? 3u : 2u;
                const int32_t output_zero_point = output_kind ? 139 : -13;
                const float output_scale = 0.01953125f +
                    (float)output_kind * 0.0029296875f +
                    (float)op * 0.00048828125f;
                if (op == 0u) {
                    qsilu_reference(input, expected, byte_values,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype);
                    CHECK(qsilu_i8u8(input, actual, byte_values,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype) == 1);
                } else {
                    qgelu_reference(input, expected, byte_values,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype);
                    CHECK(qgelu_i8u8(input, actual, byte_values,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype) == 1);
                }
                CHECK(memcmp(actual, expected, byte_values) == 0);
                memset(large_actual, 0x5a, sizeof(large_actual));
                if (op == 0u) {
                    CHECK(qsilu_i8u8(large_input, large_actual, large_elements,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype) == 1);
                } else {
                    CHECK(qgelu_i8u8(large_input, large_actual, large_elements,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype) == 1);
                }
                for (uint32_t index = 0; index < large_elements; index++)
                    CHECK(large_actual[index] == expected[(uint8_t)index]);
                memset(actual, 0x5a, sizeof(actual));
                if (op == 0u) {
                    CHECK(qsilu_i8u8(input, actual, byte_values,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype) == 1);
                } else {
                    CHECK(qgelu_i8u8(input, actual, byte_values,
                        input_scale, input_zero_point, output_scale,
                        output_zero_point, input_dtype, output_dtype) == 1);
                }
                CHECK(memcmp(actual, expected, byte_values) == 0);
            }
        }
    }
    return 0;
}

/* C=3 forces a partial packed word in GPU backends. The reference bytes are
 * intentionally away from half-LSB boundaries so the same direct vector is
 * suitable for CPU, Vulkan, OpenGL, and Metal conformance tests. */
static int test_qgroupnorm_portable_kernel(void) {
    const int8_t input_i8[3] = {-5, 0, 4};
    const uint8_t input_u8[3] = {123, 128, 132};
    const float gamma[3] = {1.0f, 0.5f, -0.75f};
    const float beta[3] = {0.25f, -0.5f, 0.75f};
    const int8_t expected_i8[3] = {-11, -7, -4};
    const uint8_t expected_u8[3] = {120, 124, 127};
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        uint32_t input_dtype = input_kind ? 3u : 2u;
        int32_t input_zero_point = input_kind ? 127 : -1;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[3] = {0};
            uint8_t actual_u8[3] = {0};
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            const void* expected = output_kind ? (const void*)expected_u8 :
                                                 (const void*)expected_i8;
            uint32_t output_dtype = output_kind ? 3u : 2u;
            int32_t output_zero_point = output_kind ? 128 : -3;
            CHECK(qgroupnorm_i8u8(input, gamma, beta, actual, 1u, 1u, 1u, 3u,
                                  1u, 0.5f, input_zero_point, 0.125f,
                                  output_zero_point, 1.0e-5f, input_dtype,
                                  output_dtype) == 1);
            CHECK(memcmp(actual, expected, sizeof(actual_i8)) == 0);
            memset(actual, 0x5a, sizeof(actual_i8));
            CHECK(qgroupnorm_i8u8(input, gamma, beta, actual, 1u, 1u, 1u, 3u,
                                  0u, 0.5f, input_zero_point, 0.125f,
                                  output_zero_point, 1.0e-5f, input_dtype,
                                  output_dtype) == 0);
            for (int index = 0; index < 3; index++)
                CHECK(((const uint8_t*)actual)[index] == 0x5a);
        }
    }
    {
        int8_t alias[3] = {-5, 0, 4};
        const int8_t before[3] = {-5, 0, 4};
        float nonfinite_gamma[3] = {1.0f, NAN, 1.0f};
        CHECK(qgroupnorm_i8u8(alias, gamma, beta, alias, 1u, 1u, 1u, 3u,
                              1u, 0.5f, -1, 0.125f, -3, 1.0e-5f, 2u, 2u) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
        CHECK(qgroupnorm_i8u8(input_i8, nonfinite_gamma, beta, alias, 1u, 1u,
                              1u, 3u, 1u, 0.5f, -1, 0.125f, -3, 1.0e-5f,
                              2u, 2u) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
    }
    {
        enum { high_elements = 16 * 16 * 4 };
        uint8_t input[high_elements];
        uint8_t output[high_elements];
        const float gamma[4] = {1.0f, 1.0f, 1.0f, 1.0f};
        const float beta[4] = {0.0f, 0.0f, 0.0f, 0.0f};
        for (int index = 0; index < high_elements; index++) input[index] =
            (uint8_t)(index & 1 ? 241 : 240);
        CHECK(qgroupnorm_i8u8(input, gamma, beta, output, 1u, 16u, 16u, 4u,
                              1u, 0.5f, 17, 0.25f, 128, 1.0e-5f, 3u, 3u) == 1);
        for (int index = 0; index < high_elements; index++)
            CHECK(output[index] == (uint8_t)(index & 1 ? 132 : 124));
    }
    return 0;
}

/* Two D=3 rows exercise a packed tail and prove that each final-axis row gets
 * independent byte-domain statistics. The values leave margin around GPU
 * rounding thresholds, so the direct backend versions reuse this vector. */
static int test_qlayernorm_portable_kernel(void) {
    const int8_t input_i8[6] = {-5, 0, 4, 6, -2, 1};
    const uint8_t input_u8[6] = {123, 128, 132, 134, 126, 129};
    const float gamma[3] = {1.0f, 0.5f, -0.75f};
    const float beta[3] = {0.25f, -0.5f, 0.75f};
    const int8_t expected_i8[6] = {-11, -7, -4, 10, -11, 4};
    const uint8_t expected_u8[6] = {120, 124, 127, 141, 120, 135};
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        uint32_t input_dtype = input_kind ? 3u : 2u;
        int32_t input_zero_point = input_kind ? 127 : -1;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[6];
            uint8_t actual_u8[6];
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            const void* expected = output_kind ? (const void*)expected_u8 :
                                                 (const void*)expected_i8;
            uint32_t output_dtype = output_kind ? 3u : 2u;
            int32_t output_zero_point = output_kind ? 128 : -3;
            CHECK(qlayernorm_i8u8(input, gamma, beta, actual, 2u, 3u,
                                  0.5f, input_zero_point, 0.125f,
                                  output_zero_point, 1.0e-5f, input_dtype,
                                  output_dtype) == 1);
            CHECK(memcmp(actual, expected, sizeof(actual_i8)) == 0);
            memset(actual, 0x5a, sizeof(actual_i8));
            CHECK(qlayernorm_i8u8(input, gamma, beta, actual, 2u, 0u,
                                  0.5f, input_zero_point, 0.125f,
                                  output_zero_point, 1.0e-5f, input_dtype,
                                  output_dtype) == 0);
            for (int index = 0; index < 6; index++)
                CHECK(((const uint8_t*)actual)[index] == 0x5a);
        }
    }
    {
        int8_t alias[6] = {-5, 0, 4, 6, -2, 1};
        const int8_t before[6] = {-5, 0, 4, 6, -2, 1};
        float nonfinite_beta[3] = {0.0f, NAN, 0.0f};
        CHECK(qlayernorm_i8u8(alias, gamma, beta, alias, 2u, 3u,
                              0.5f, -1, 0.125f, -3, 1.0e-5f, 2u, 2u) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
        CHECK(qlayernorm_i8u8(input_i8, gamma, nonfinite_beta, alias, 2u, 3u,
                              0.5f, -1, 0.125f, -3, 1.0e-5f, 2u, 2u) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
    }
    {
        const int8_t ties_input[3] = {-3, -2, -1};
        const float zero_gamma[3] = {0.0f, 0.0f, 0.0f};
        const float ties_beta[3] = {0.5f, 1.5f, -1.5f};
        int8_t ties_output[3] = {0};
        const int8_t ties_expected[3] = {0, 2, -2};
        CHECK(qlayernorm_i8u8(ties_input, zero_gamma, ties_beta, ties_output,
                              1u, 3u, 1.0f, 0, 1.0f, 0, 1.0e-5f, 2u, 2u) == 1);
        CHECK(memcmp(ties_output, ties_expected, sizeof(ties_expected)) == 0);
    }
    {
        enum { high_d_model = 1024, high_elements = 2 * high_d_model };
        uint8_t input[high_elements];
        uint8_t output[high_elements] = {0};
        float gamma[high_d_model];
        float beta[high_d_model];
        for (int index = 0; index < high_elements; index++)
            input[index] = (uint8_t)(index & 1 ? 241 : 240);
        for (int channel = 0; channel < high_d_model; channel++) {
            gamma[channel] = 1.0f;
            beta[channel] = 0.0f;
        }
        CHECK(qlayernorm_i8u8(input, gamma, beta, output, 2u, high_d_model,
                              0.5f, 17, 0.25f, 128, 1.0e-5f, 3u, 3u) == 1);
        for (int index = 0; index < high_elements; index++)
            CHECK(output[index] == (uint8_t)(index & 1 ? 132 : 124));
    }
    return 0;
}

/* QSDPA keeps Q/K/V in independent physical byte domains.  The first fixture
 * is deliberately invariant under each I8/U8 storage choice, so all sixteen
 * input/output combinations exercise the same raw-dot and requantization
 * result. */
static int test_qsdpa_portable_kernel(void) {
    const int8_t q_i8[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    const int8_t k_i8[8] = {0, 0, -1, -1, -1, 0, 0, -2};
    const int8_t v_i8[8] = {3, -3, 0, -1, -5, 1, 2, -2};
    const uint8_t q_u8[8] = {128, 127, 126, 129, 126, 129, 128, 127};
    const uint8_t k_u8[8] = {128, 128, 127, 127, 127, 128, 128, 126};
    const uint8_t v_u8[8] = {131, 125, 128, 127, 123, 129, 130, 126};
    const int8_t expected_i8[8] = {0, 0, 2, 0, 0, 0, 2, -1};
    const int8_t expected_causal_i8[8] = {4, -2, 1, 0, 0, 0, 2, -1};
    const uint8_t expected_u8[8] = {128, 128, 130, 128, 128, 128, 130, 127};
    const uint8_t expected_causal_u8[8] = {132, 126, 129, 128, 128, 128, 130, 127};
    for (int q_kind = 0; q_kind < 2; q_kind++) {
        const void* q = q_kind ? (const void*)q_u8 : (const void*)q_i8;
        uint32_t q_dtype = q_kind ? 3u : 2u;
        int32_t q_zp = q_kind ? 127 : -1;
        for (int k_kind = 0; k_kind < 2; k_kind++) {
            const void* k = k_kind ? (const void*)k_u8 : (const void*)k_i8;
            uint32_t k_dtype = k_kind ? 3u : 2u;
            int32_t k_zp = k_kind ? 127 : -1;
            for (int v_kind = 0; v_kind < 2; v_kind++) {
                const void* v = v_kind ? (const void*)v_u8 : (const void*)v_i8;
                uint32_t v_dtype = v_kind ? 3u : 2u;
                int32_t v_zp = v_kind ? 127 : -1;
                for (int output_kind = 0; output_kind < 2; output_kind++) {
                    int8_t actual_i8[8] = {0};
                    uint8_t actual_u8[8] = {0};
                    void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
                    const void* expected = output_kind ? (const void*)expected_u8 :
                                                        (const void*)expected_i8;
                    const void* expected_causal = output_kind ?
                        (const void*)expected_causal_u8 : (const void*)expected_causal_i8;
                    uint32_t output_dtype = output_kind ? 3u : 2u;
                    int32_t output_zp = output_kind ? 128 : 0;
                    CHECK(qsdpa_i8u8(q, k, v, NULL, actual, 1u, 2u, 2u, 4u, 1u,
                                      0.25f, q_zp, 0.25f, k_zp, 0.25f, v_zp,
                                      0.25f, output_zp, 0.5f, q_dtype, k_dtype,
                                      v_dtype, output_dtype, 0u, 0u) == 1);
                    CHECK(memcmp(actual, expected, sizeof(actual_i8)) == 0);
                    CHECK(qsdpa_i8u8(q, k, v, NULL, actual, 1u, 2u, 2u, 4u, 1u,
                                      0.25f, q_zp, 0.25f, k_zp, 0.25f, v_zp,
                                      0.25f, output_zp, 0.5f, q_dtype, k_dtype,
                                      v_dtype, output_dtype, 1u, 0u) == 1);
                    CHECK(memcmp(actual, expected_causal, sizeof(actual_i8)) == 0);
                }
            }
        }
    }
    {
        const uint8_t q[4] = {129, 126, 131, 128};
        const uint8_t k[8] = {121, 120, 119, 122, 119, 123, 120, 118};
        const uint8_t v[8] = {134, 126, 132, 130, 128, 132, 136, 128};
        const int32_t keep_second[2] = {0, 1};
        const int32_t mask_none[2] = {0, 0};
        const uint8_t expected[4] = {128, 126, 131, 126};
        const uint8_t expected_second[4] = {125, 129, 133, 125};
        const uint8_t expected_zero[4] = {127, 127, 127, 127};
        uint8_t output[4] = {0};
        CHECK(qsdpa_i8u8(q, k, v, NULL, output, 1u, 1u, 2u, 4u, 1u,
                          0.25f, 128, 0.5f, 120, 0.25f, 130, 0.25f, 127,
                          0.5f, 3u, 3u, 3u, 3u, 0u, 0u) == 1);
        CHECK(memcmp(output, expected, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, keep_second, output, 1u, 1u, 2u, 4u, 1u,
                          0.25f, 128, 0.5f, 120, 0.25f, 130, 0.25f, 127,
                          0.5f, 3u, 3u, 3u, 3u, 0u, 1u) == 1);
        CHECK(memcmp(output, expected_second, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, mask_none, output, 1u, 1u, 2u, 4u, 1u,
                          0.25f, 128, 0.5f, 120, 0.25f, 130, 0.25f, 127,
                          0.5f, 3u, 3u, 3u, 3u, 0u, 1u) == 1);
        CHECK(memcmp(output, expected_zero, sizeof(output)) == 0);
    }
    {
        int8_t q[16];
        int8_t k[16];
        int8_t v[16];
        int8_t output[16] = {0};
        const int32_t batch_key[4] = {1, 1, 0, 0};
        const int32_t query_key[4] = {1, 0, 0, 1};
        const int32_t batch_query_key[8] = {1, 0, 0, 1, 0, 0, 1, 1};
        const int8_t expected_batch_key[16] = {
            0, 0, 2, 0, 0, 0, 2, -1, 0, 0, 0, 0, 0, 0, 0, 0,
        };
        const int8_t expected_query_key[16] = {
            4, -2, 1, 0, -4, 2, 3, -1, 4, -2, 1, 0, -4, 2, 3, -1,
        };
        const int8_t expected_batch_query_key[16] = {
            4, -2, 1, 0, -4, 2, 3, -1, 0, 0, 0, 0, 0, 0, 2, -1,
        };
        for (int batch = 0; batch < 2; batch++) {
            memcpy(q + batch * 8, q_i8, 8);
            memcpy(k + batch * 8, k_i8, 8);
            memcpy(v + batch * 8, v_i8, 8);
        }
        CHECK(qsdpa_i8u8(q, k, v, batch_key, output, 2u, 2u, 2u, 4u, 1u,
                          0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, 2u, 2u, 2u, 2u, 0u, 2u) == 1);
        CHECK(memcmp(output, expected_batch_key, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, query_key, output, 2u, 2u, 2u, 4u, 1u,
                          0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, 2u, 2u, 2u, 2u, 0u, 3u) == 1);
        CHECK(memcmp(output, expected_query_key, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, batch_query_key, output, 2u, 2u, 2u, 4u, 1u,
                          0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, 2u, 2u, 2u, 2u, 0u, 4u) == 1);
        CHECK(memcmp(output, expected_batch_query_key, sizeof(output)) == 0);
    }
    {
        const int8_t q[4] = {0, 0, 0, 0};
        const int8_t k[4] = {0, 0, 0, 0};
        const int8_t v[4] = {1, 3, -3, 5};
        const int8_t expected[4] = {0, 2, -2, 2};
        int8_t output[4] = {0};
        CHECK(qsdpa_i8u8(q, k, v, NULL, output, 1u, 1u, 1u, 4u, 1u,
                          1.0f, 0, 1.0f, 0, 0.5f, 0, 1.0f, 0,
                          1.0f, 2u, 2u, 2u, 2u, 0u, 0u) == 1);
        CHECK(memcmp(output, expected, sizeof(output)) == 0);
    }
    {
        enum { head_dim = 64 };
        int8_t q[head_dim] = {0};
        int8_t k[head_dim] = {0};
        int8_t v[head_dim];
        int8_t output[head_dim] = {0};
        for (int channel = 0; channel < head_dim; channel++)
            v[channel] = (int8_t)((channel % 15) - 7);
        CHECK(qsdpa_i8u8(q, k, v, NULL, output, 1u, 1u, 1u,
                          head_dim, 1u, 0.25f, 0, 0.25f, 0, 0.25f, 0,
                          0.25f, 0, 1.0f, 2u, 2u, 2u, 2u, 0u, 0u) == 1);
        CHECK(memcmp(output, v, sizeof(output)) == 0);
    }
    {
        int8_t overlap[16] = {0, -1, -2, 1, -2, 1, 0, -1};
        int8_t before[16];
        int32_t mask_overlap[3] = {1, 1, 0};
        int32_t mask_before[3];
        int8_t sentinel[8];
        memcpy(before, overlap, sizeof(before));
        memcpy(mask_before, mask_overlap, sizeof(mask_overlap));
        memset(sentinel, 0x5a, sizeof(sentinel));
        CHECK(qsdpa_i8u8(overlap, k_i8, v_i8, NULL, overlap + 1, 1u, 2u, 2u,
                          4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                          0.25f, 0, 0.5f, 2u, 2u, 2u, 2u, 0u, 0u) == 0);
        CHECK(memcmp(overlap, before, sizeof(overlap)) == 0);
        CHECK(qsdpa_i8u8(q_i8, k_i8, v_i8, mask_overlap,
                          (unsigned char*)mask_overlap + sizeof(int32_t), 1u, 2u,
                          2u, 4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                          0.25f, 0, 0.5f, 2u, 2u, 2u, 2u, 0u, 1u) == 0);
        CHECK(memcmp(mask_overlap, mask_before, sizeof(mask_overlap)) == 0);
        CHECK(qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, sentinel, 1u, 2u, 2u, 2u,
                          1u, 0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, 2u, 2u, 2u, 2u, 0u, 0u) == 0);
        CHECK(qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, sentinel, 1u, 2u, 2u, 4u,
                          1u, 1.0e30f, -1, 1.0e30f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, 2u, 2u, 2u, 2u, 0u, 0u) == 0);
        for (int index = 0; index < 8; index++) CHECK((unsigned char)sentinel[index] == 0x5a);
    }
    return 0;
}

/* QArgMax reduces raw typed bytes, not dequantized F32 values. The [2,3,2]
 * fixture exercises a non-last axis, signed negative ordering, and first-tie
 * semantics in one compact descriptor. */
static int test_qargmax_portable_kernel(void) {
    const int8_t input_i8[12] = {
        -5, 4, -1, 4, -1, 3,
        -128, 0, -127, 1, -126, 1,
    };
    const int32_t expected_i8[4] = {1, 0, 2, 1};
    const uint8_t input_u8[6] = {130, 3, 129, 255, 130, 254};
    const int32_t expected_u8[2] = {0, 1};
    const int8_t last_axis_i8[8] = {-3, -2, -2, -4, -128, -127, -127, -126};
    const int32_t expected_last_axis[2] = {1, 3};
    int32_t output_i8[4] = {0};
    int32_t output_u8[2] = {0};
    int32_t output_last_axis[2] = {0};
    union {
        int32_t aligned[4];
        uint8_t bytes[16];
    } alias;
    uint8_t alias_before[sizeof(alias.bytes)];
    int32_t sentinel[4] = {17, 23, 31, 47};
    const int32_t sentinel_before[4] = {17, 23, 31, 47};

    CHECK(qargmax_i8u8(input_i8, output_i8, 2u, 3u, 2u, 2u) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    /* U8 with a nonzero asymmetric zero point still compares raw U8 ordering. */
    CHECK(qargmax_i8u8(input_u8, output_u8, 1u, 3u, 2u, 3u) == 1);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(qargmax_i8u8(last_axis_i8, output_last_axis, 2u, 4u, 1u, 2u) == 1);
    CHECK(memcmp(output_last_axis, expected_last_axis, sizeof(output_last_axis)) == 0);

    memset(alias.bytes, 0x5a, sizeof(alias.bytes));
    memcpy(alias.bytes, input_i8, 6u);
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    CHECK(qargmax_i8u8(alias.bytes, alias.aligned, 1u, 3u, 2u, 2u) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, 0u, 3u, 2u, 2u) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, 2u, 3u, 2u, 1u) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, UINT32_MAX, 2u, 1u, 2u) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    return 0;
}

/* QMaskedMean averages only nonzero
 * I32 mask entries in centered raw-byte space, then requantizes once.  This
 * fixture covers signed and unsigned domains, an exact ties-to-even result,
 * and the all-masked real-zero convention. */
static int test_qmaskedmean_portable_kernel(void) {
    const int8_t input_i8[24] = {
        -3, -1,  1,  1,
         1, -5, -3, -1,
        99, 98, 97, 96,
        40, 41, 42, 43,
        44, 45, 46, 47,
        48, 49, 50, 51,
    };
    const int32_t mask_i8[6] = {1, 1, 0, 0, 0, 0};
    const int8_t expected_i8[8] = {6, 5, 6, 6, 5, 5, 5, 5};
    int8_t output_i8[8] = {0};
    const uint8_t input_u8[6] = {130, 134, 20, 20, 131, 129};
    const int32_t mask_u8[3] = {1, 0, 1};
    const uint8_t expected_u8[2] = {132, 134};
    uint8_t output_u8[2] = {0};
    int8_t staged_input[127];
    int32_t staged_mask[127];
    int8_t staged_output = 0;
    union {
        int8_t input[24];
        uint8_t bytes[24];
    } alias;
    union {
        int32_t mask[6];
        uint8_t bytes[24];
    } mask_alias;
    uint8_t alias_before[sizeof(alias.bytes)];
    uint8_t mask_before[sizeof(mask_alias.bytes)];
    int8_t sentinel[8] = {17, 23, 31, 47, 53, 59, 61, 67};
    const int8_t sentinel_before[8] = {17, 23, 31, 47, 53, 59, 61, 67};

    CHECK(qmaskedmean_i8u8(input_i8, mask_i8, output_i8, 2u, 3u, 4u,
                            0.25f, -3, 0.5f, 5, 2u, 2u) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(qmaskedmean_i8u8(input_u8, mask_u8, output_u8, 1u, 3u, 2u,
                            0.25f, 128, 0.25f, 130, 3u, 3u) == 1);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    /* This descriptor lands on an adversarial half step: an FMA would round
     * to -63, while the canonical F32 product then add schedule ties-even to
     * -62.  It protects the portable C/WASM parity boundary. */
    for (uint32_t index = 0; index < 127u; index++) {
        staged_input[index] = -128;
        staged_mask[index] = 1;
    }
    CHECK(qmaskedmean_i8u8(staged_input, staged_mask, &staged_output,
                            1u, 127u, 1u, 0.001f, 127, 0.01f, -37,
                            2u, 2u) == 1);
    CHECK(staged_output == -62);

    memcpy(alias.input, input_i8, sizeof(input_i8));
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    CHECK(qmaskedmean_i8u8(alias.input, mask_i8, alias.bytes, 2u, 3u, 4u,
                            0.25f, -3, 0.5f, 5, 2u, 2u) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias_before)) == 0);
    memcpy(mask_alias.mask, mask_i8, sizeof(mask_i8));
    memcpy(mask_before, mask_alias.bytes, sizeof(mask_before));
    CHECK(qmaskedmean_i8u8(input_i8, mask_alias.mask, mask_alias.bytes,
                            2u, 3u, 4u, 0.25f, -3, 0.5f, 5, 2u, 2u) == 0);
    CHECK(memcmp(mask_alias.bytes, mask_before, sizeof(mask_before)) == 0);
    CHECK(qmaskedmean_i8u8(input_i8, mask_i8, sentinel, 2u, 0u, 4u,
                            0.25f, -3, 0.5f, 5, 2u, 2u) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qmaskedmean_i8u8(input_i8, mask_i8, sentinel, 1u,
                            (uint32_t)INT32_MAX / 255u + 1u, 1u,
                            0.25f, -128, 0.5f, 5, 2u, 2u) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    return 0;
}

static int test_w8a8_linear_vector_tail(void) {
    enum { rows = 2, d_in = 17, d_out = 2 };
    int8_t input[rows * d_in];
    int8_t weight[d_out * d_in];
    const float scale[d_out] = {0.03125f, 0.046875f};
    const int32_t zero_point[d_out] = {-6, 9};
    const int32_t bias[d_out] = {37, -53};
    int8_t actual[rows * d_out] = {0};
    for (int row = 0; row < rows; row++) {
        for (int k = 0; k < d_in; k++) input[row * d_in + k] = (int8_t)(((row * 29 + k * 17) % 201) - 100);
    }
    for (int oc = 0; oc < d_out; oc++) {
        for (int k = 0; k < d_in; k++) weight[oc * d_in + k] = (int8_t)(((oc * 43 + k * 13) % 201) - 100);
    }
    /* d_in=17 exercises the AVX2 16-byte dot product plus scalar tail. */
    CHECK(vx_qlinear_i8((const signed char*)input, (signed char*)actual,
                        weight, T_I8, scale, d_out, zero_point, T_I32, d_out, bias,
                        rows, d_in, d_out, 1, 0.0625f, -7, 0.125f, 4) == 1);
    for (int row = 0; row < rows; row++) {
        for (int oc = 0; oc < d_out; oc++) {
            int64_t accumulator = bias[oc];
            for (int k = 0; k < d_in; k++) {
                accumulator += ((int)input[row * d_in + k] + 7) *
                    ((int)weight[oc * d_in + k] - zero_point[oc]);
            }
            int8_t expected = quantize_i8(
                (float)((double)accumulator * 0.0625 * scale[oc]), 0.125f, 4);
            CHECK(actual[row * d_out + oc] == expected);
        }
    }
    const int8_t zero_input[1] = {0};
    const int8_t zero_weight[2] = {0, 0};
    const float unit_scale[2] = {1.0f, 1.0f};
    const int32_t unit_zero_point[2] = {0, 0};
    const int32_t extreme_bias[2] = {INT32_MAX, INT32_MIN};
    int8_t saturated[2] = {0};
    CHECK(vx_qlinear_i8((const signed char*)zero_input, (signed char*)saturated,
                        zero_weight, T_I8, unit_scale, 2, unit_zero_point, T_I32, 2,
                        extreme_bias, 1, 1, 2, 1, 1.0f, 0, 1.0e-30f, 0) == 1);
    CHECK(saturated[0] == 127 && saturated[1] == -128);
    return 0;
}

/* The native dispatcher must agree byte-for-byte with the portable canonical
 * ABI for each physical I8/U8 combination. Five rows exercise one four-row
 * seed tile plus its row tail. d_in=67 exercises an AVX-512 VNNI 64-byte dot
 * plus scalar tail, two AVX-VNNI blocks plus tail, or the AVX2 widening path
 * plus its tail according to the runtime feature gate. */
static int test_physical_qlinear_native_dispatch(void) {
    enum { rows = 5, d_in = 67, d_out = 3 };
    int8_t input_i8[rows * d_in];
    uint8_t input_u8[rows * d_in];
    int8_t weight_i8[d_out * d_in];
    uint8_t weight_u8[d_out * d_in];
    const float weight_scales[d_out] = {0.03125f, 0.046875f, 0.0625f};
    const int32_t weight_zp_i8[d_out] = {-21, 12, 0};
    const int32_t weight_zp_u8[d_out] = {121, 132, 5};
    const int32_t bias[d_out] = {37, -53, 211};
    for (int row = 0; row < rows; row++) {
        for (int k = 0; k < d_in; k++) {
            input_i8[row * d_in + k] = (int8_t)(((row * 31 + k * 17) % 201) - 100);
            input_u8[row * d_in + k] = (uint8_t)((row * 43 + k * 11) % 241);
        }
    }
    for (int oc = 0; oc < d_out; oc++) {
        for (int k = 0; k < d_in; k++) {
            weight_i8[oc * d_in + k] = (int8_t)(((oc * 47 + k * 13) % 201) - 100);
            weight_u8[oc * d_in + k] = (uint8_t)((oc * 37 + k * 19) % 251);
        }
    }
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zp = input_kind ? 119 : -13;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ? 3u : 2u;
            const int32_t* weight_zp = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (int output_kind = 0; output_kind < 2; output_kind++) {
                int8_t reference_i8[rows * d_out] = {0};
                int8_t actual_i8[rows * d_out] = {0};
                int8_t arm_i8[rows * d_out] = {0};
                uint8_t reference_u8[rows * d_out] = {0};
                uint8_t actual_u8[rows * d_out] = {0};
                uint8_t arm_u8[rows * d_out] = {0};
                void* reference = output_kind ? (void*)reference_u8 : (void*)reference_i8;
                void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
                void* arm_actual = output_kind ? (void*)arm_u8 : (void*)arm_i8;
                const uint32_t output_dtype = output_kind ? 3u : 2u;
                const int32_t output_zp = output_kind ? 127 : -5;
                CHECK(qlinear_i8u8(input, weight, bias, weight_scales, weight_zp,
                                    reference, rows, d_in, d_out,
                                    0.0625f, input_zp, 0.125f, output_zp,
                                    input_dtype, weight_dtype, output_dtype) == 1);
                CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales, weight_zp,
                                              actual, rows, d_in, d_out,
                                              0.0625f, input_zp, 0.125f, output_zp,
                                              input_dtype, weight_dtype, output_dtype) == 1);
                CHECK(memcmp(reference, actual, sizeof(reference_i8)) == 0);
                {
                    int arm_handled = vx_qlinear_i8u8_arm_try(
                        input, weight, bias, weight_scales, weight_zp, arm_actual,
                        rows, d_in, d_out, 0.0625f, input_zp, 0.125f, output_zp,
                        input_dtype, weight_dtype, output_dtype);
                    CHECK(arm_handled == 0 || arm_handled == 1);
                    if (arm_handled) CHECK(memcmp(reference, arm_actual, sizeof(reference_i8)) == 0);
                }
            }
        }
    }
    {
        int8_t portable_alias[rows * d_in];
        int8_t native_alias[rows * d_in];
        memcpy(portable_alias, input_i8, sizeof(portable_alias));
        memcpy(native_alias, input_i8, sizeof(native_alias));
        /* Output occupies the beginning of the input allocation. Tiled rows
         * would overwrite bytes that a later output column still reads, so
         * the native path must retain the canonical row-ordered fallback. */
        CHECK(qlinear_i8u8(portable_alias, weight_i8, bias, weight_scales,
                            weight_zp_i8, portable_alias, rows, d_in, d_out,
                            0.0625f, -13, 0.125f, -5, 2u, 2u, 2u) == 1);
        CHECK(vx_qlinear_i8u8_native(native_alias, weight_i8, bias,
                                      weight_scales, weight_zp_i8, native_alias,
                                      rows, d_in, d_out, 0.0625f, -13,
                                      0.125f, -5, 2u, 2u, 2u) == 1);
        CHECK(memcmp(portable_alias, native_alias, sizeof(portable_alias)) == 0);
    }
    {
        const int8_t overflow_input[8] = {127, 127, 127, 127, 127, 127, 127, 127};
        const int8_t overflow_weight[8] = {127, 127, 127, 127, 127, 127, 127, 127};
        const int32_t overflow_bias[1] = {INT32_MAX};
        const float unit_scale[1] = {1.0f};
        const int32_t minus_128[1] = {-128};
        int8_t reference = 41;
        int8_t actual = 41;
        int8_t arm_actual = 41;
        /* This is deliberately outside the vector proof envelope. The
         * portable ABI rejects it at the first overflowing product, and the
         * dispatcher must preserve both the failure and untouched output. */
        CHECK(qlinear_i8u8(overflow_input, overflow_weight, overflow_bias,
                            unit_scale, minus_128, &reference, 1, 8, 1,
                            1.0f, -128, 1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(vx_qlinear_i8u8_native(overflow_input, overflow_weight, overflow_bias,
                                      unit_scale, minus_128, &actual, 1, 8, 1,
                                      1.0f, -128, 1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(vx_qlinear_i8u8_arm_try(overflow_input, overflow_weight, overflow_bias,
                                       unit_scale, minus_128, &arm_actual, 1, 8, 1,
                                       1.0f, -128, 1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(reference == 41 && actual == 41 && arm_actual == 41);
    }
    return 0;
}

/* The TinyReceipt decoder seed presents full 192-row dense calls.  Force the
 * native row scheduler through one- and four-thread configurations and keep
 * the incremental M=1 result in the same parity fixture. */
static int test_physical_qlinear_native_threaded_dispatch(void) {
    enum {
        rows = 192, d_in = 320, d_out = 320,
        input_elements = rows * d_in,
        weight_elements = d_out * d_in,
        output_elements = rows * d_out,
    };
    static int8_t input[input_elements];
    static int8_t weight[weight_elements];
    static int32_t bias[d_out];
    static float weight_scales[d_out];
    static int32_t weight_zero_points[d_out];
    static int8_t reference[output_elements];
    static int8_t single_thread[output_elements];
    static int8_t threaded[output_elements];
    int8_t row_single[d_out];
    int8_t row_thread_config[d_out];
    int expected_parallel = 0;
    for (int index = 0; index < input_elements; index++)
        input[index] = (int8_t)(((index * 29 + 17) % 221) - 110);
    for (int index = 0; index < weight_elements; index++)
        weight[index] = (int8_t)(((index * 37 + 11) % 211) - 105);
    for (int column = 0; column < d_out; column++) {
        bias[column] = column * 41 - 503;
        weight_scales[column] = 1.0f / (float)(48 + column % 7 * 8);
        weight_zero_points[column] = column % 13 - 6;
    }
    CHECK(qlinear_i8u8(input, weight, bias, weight_scales,
                       weight_zero_points, reference, rows, d_in, d_out,
                       1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                       2u, 2u, 2u) == 1);
    vx_set_num_threads(1);
    CHECK(!vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, single_thread,
        rows, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        2u, 2u, 2u));
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, single_thread,
                                 rows, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3, 2u, 2u, 2u) == 1);
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, row_single,
                                 1u, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3, 2u, 2u, 2u) == 1);
    vx_set_num_threads(4);
#if defined(__i386__) || defined(__x86_64__)
    expected_parallel = vx_kernels_thread_count() > 1 &&
        (vx_cpu_has_avx2() || vx_cpu_has_avx_vnni() ||
         vx_cpu_has_avx512_vnni());
#endif
    CHECK(vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, threaded,
        rows, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        2u, 2u, 2u) == expected_parallel);
    /* These are the runtime's packed-policy cases: M=1 is incremental, and
     * M=2 is below the native pool threshold despite having multiple rows. */
    CHECK(!vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, threaded,
        1u, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        2u, 2u, 2u));
    CHECK(!vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, threaded,
        2u, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        2u, 2u, 2u));
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, threaded,
                                 rows, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3, 2u, 2u, 2u) == 1);
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, row_thread_config,
                                 1u, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3, 2u, 2u, 2u) == 1);
    vx_set_num_threads(0);
    CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
    CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
    CHECK(memcmp(reference, row_single, sizeof(row_single)) == 0);
    CHECK(memcmp(reference, row_thread_config, sizeof(row_thread_config)) == 0);
    return 0;
}

/* VPDPBUSD is U8 x I8. Verify the remapped-domain compensation identity with
 * extreme zero points and a 64-byte vector block plus tail before relying on
 * a VNNI-capable CPU to execute the specialization. */
static int64_t qlinear_vnni_remapped_accumulator(const void* input, const void* weight,
                                                   const int32_t* bias,
                                                   const int32_t* weight_zero_points,
                                                   uint32_t d_in, uint32_t input_dtype,
                                                   uint32_t weight_dtype,
                                                   int32_t input_zero_point,
                                                   uint32_t column) {
    const int32_t input_zero_unsigned = input_zero_point + (input_dtype == 2u ? 128 : 0);
    const int32_t weight_zero_signed = weight_zero_points[column] -
        (weight_dtype == 3u ? 128 : 0);
    int64_t input_sum = 0;
    int64_t weight_sum = 0;
    int64_t dot = 0;
    for (uint32_t dimension = 0; dimension < d_in; dimension++) {
        const int32_t input_raw = input_dtype == 2u
            ? (int32_t)((const int8_t*)input)[dimension]
            : (int32_t)((const uint8_t*)input)[dimension];
        const int32_t weight_raw = weight_dtype == 2u
            ? (int32_t)((const int8_t*)weight)[(size_t)column * d_in + dimension]
            : (int32_t)((const uint8_t*)weight)[(size_t)column * d_in + dimension];
        const int32_t input_unsigned = input_dtype == 2u ? input_raw + 128 : input_raw;
        const int32_t weight_signed = weight_dtype == 3u ? weight_raw - 128 : weight_raw;
        input_sum += input_unsigned;
        weight_sum += weight_signed;
        dot += (int64_t)input_unsigned * weight_signed;
    }
    return (int64_t)bias[column] + dot - (int64_t)weight_zero_signed * input_sum -
        (int64_t)input_zero_unsigned * weight_sum +
        (int64_t)d_in * input_zero_unsigned * weight_zero_signed;
}

static int test_physical_qlinear_vnni_remapping(void) {
    enum { d_in = 67, d_out = 2 };
    int8_t input_i8[d_in];
    uint8_t input_u8[d_in];
    int8_t weight_i8[d_out * d_in];
    uint8_t weight_u8[d_out * d_in];
    const int32_t bias[d_out] = {17, -29};
    const int32_t weight_zp_i8[d_out] = {-128, 127};
    const int32_t weight_zp_u8[d_out] = {0, 255};
    for (int dimension = 0; dimension < d_in; dimension++) {
        input_i8[dimension] = (int8_t)((dimension & 1) ? 127 : -128);
        input_u8[dimension] = (uint8_t)((dimension & 1) ? 255 : 0);
    }
    for (int column = 0; column < d_out; column++) {
        for (int dimension = 0; dimension < d_in; dimension++) {
            weight_i8[column * d_in + dimension] = (int8_t)((dimension + column) & 1 ? 127 : -128);
            weight_u8[column * d_in + dimension] = (uint8_t)((dimension + column) & 1 ? 255 : 0);
        }
    }
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zero_point = input_kind ? 255 : -128;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ? 3u : 2u;
            const int32_t* weight_zero_points = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (uint32_t column = 0; column < d_out; column++) {
                int64_t conventional = bias[column];
                for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                    const int32_t input_raw = input_dtype == 2u
                        ? (int32_t)((const int8_t*)input)[dimension]
                        : (int32_t)((const uint8_t*)input)[dimension];
                    const int32_t weight_raw = weight_dtype == 2u
                        ? (int32_t)((const int8_t*)weight)[(size_t)column * d_in + dimension]
                        : (int32_t)((const uint8_t*)weight)[(size_t)column * d_in + dimension];
                    conventional += (int64_t)(input_raw - input_zero_point) *
                        (int64_t)(weight_raw - weight_zero_points[column]);
                }
                CHECK(qlinear_vnni_remapped_accumulator(input, weight, bias,
                    weight_zero_points, d_in, input_dtype, weight_dtype,
                    input_zero_point, column) == conventional);
            }
        }
    }
    return 0;
}

/* Exercise the native QConv dispatcher with VNNI-width channel blocks,
 * groups, asymmetric padding, dilation, stride, ReLU6, and scalar tails.
 * Every byte-domain combination is compared directly with the portable ABI. */
static int test_physical_qconv_native_dispatch(void) {
    enum {
        batch = 1, input_height = 4, input_width = 4, input_per_group = 67,
        groups = 2, input_channels = input_per_group * groups,
        output_height = 3, output_width = 5, output_channels = 4,
        kernel_height = 2, kernel_width = 2,
        input_elements = batch * input_height * input_width * input_channels,
        weight_elements = output_channels * kernel_height * kernel_width * input_per_group,
        output_elements = batch * output_height * output_width * output_channels,
    };
    int8_t input_i8[input_elements];
    uint8_t input_u8[input_elements];
    int8_t weight_i8[weight_elements];
    uint8_t weight_u8[weight_elements];
    const float weight_scales[output_channels] = {0.03125f, 0.046875f, 0.0625f, 0.078125f};
    const int32_t weight_zp_i8[output_channels] = {-21, 12, 0, -7};
    const int32_t weight_zp_u8[output_channels] = {121, 132, 5, 201};
    const int32_t bias[output_channels] = {37, -53, 211, -317};
    for (int index = 0; index < input_elements; index++) {
        input_i8[index] = (int8_t)(((index * 17 + 29) % 201) - 100);
        input_u8[index] = (uint8_t)((index * 23 + 41) % 241);
    }
    for (int index = 0; index < weight_elements; index++) {
        weight_i8[index] = (int8_t)(((index * 13 + 47) % 201) - 100);
        weight_u8[index] = (uint8_t)((index * 19 + 31) % 251);
    }
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zp = input_kind ? 119 : -13;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ? 3u : 2u;
            const int32_t* weight_zp = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (int output_kind = 0; output_kind < 2; output_kind++) {
                const uint32_t output_dtype = output_kind ? 3u : 2u;
                const int32_t output_zp = output_kind ? 127 : -5;
                for (int bias_kind = 0; bias_kind < 2; bias_kind++) {
                    const int32_t* selected_bias = bias_kind ? NULL : bias;
                    int8_t reference_i8[output_elements] = {0};
                    int8_t actual_i8[output_elements] = {0};
                    uint8_t reference_u8[output_elements] = {0};
                    uint8_t actual_u8[output_elements] = {0};
                    void* reference = output_kind ? (void*)reference_u8 : (void*)reference_i8;
                    void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
                    CHECK(qconv2d_i8u8(input, weight, selected_bias, weight_scales, weight_zp,
                                        reference, batch, input_height, input_width, input_channels,
                                        output_height, output_width, output_channels,
                                        kernel_height, kernel_width, input_per_group,
                                        2, 1, 2, 1, 1, 1, 2, 1, groups, 2,
                                        0.0625f, input_zp, 0.125f, output_zp,
                                        input_dtype, weight_dtype, output_dtype) == 1);
                    CHECK(vx_qconv2d_i8u8_native(input, weight, selected_bias, weight_scales,
                                                  weight_zp, actual, batch, input_height,
                                                  input_width, input_channels, output_height,
                                                  output_width, output_channels, kernel_height,
                                                  kernel_width, input_per_group,
                                                  2, 1, 2, 1, 1, 1, 2, 1, groups, 2,
                                                  0.0625f, input_zp, 0.125f, output_zp,
                                                  input_dtype, weight_dtype, output_dtype) == 1);
                    CHECK(memcmp(reference, actual, sizeof(reference_i8)) == 0);
                }
            }
        }
    }
    {
        const int8_t overflow_input[16] = {
            127, 127, 127, 127, 127, 127, 127, 127,
            127, 127, 127, 127, 127, 127, 127, 127,
        };
        const int8_t overflow_weight[16] = {
            127, 127, 127, 127, 127, 127, 127, 127,
            127, 127, 127, 127, 127, 127, 127, 127,
        };
        const int32_t overflow_bias[1] = {INT32_MAX};
        const float unit_scale[1] = {1.0f};
        const int32_t minus_128[1] = {-128};
        int8_t reference = 41;
        int8_t actual = 41;
        CHECK(qconv2d_i8u8(overflow_input, overflow_weight, overflow_bias,
                            unit_scale, minus_128, &reference,
                            1, 1, 1, 16, 1, 1, 1, 1, 1, 16,
                            1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                            1.0f, -128, 1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(vx_qconv2d_i8u8_native(overflow_input, overflow_weight, overflow_bias,
                                      unit_scale, minus_128, &actual,
                                      1, 1, 1, 16, 1, 1, 1, 1, 1, 16,
                                      1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                                      1.0f, -128, 1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(reference == 41 && actual == 41);
    }
    return 0;
}

/* A large, grouped, multi-batch convolution crosses the x86 QConv threading
 * threshold. Compare both the direct one-thread path and the tiled pool path
 * byte-for-byte with the canonical portable implementation. */
static int test_physical_qconv_native_threaded_dispatch(void) {
    enum {
        batch = 2, input_height = 11, input_width = 13,
        input_per_group = 64, groups = 2,
        input_channels = input_per_group * groups,
        output_height = 11, output_width = 13, output_channels = 32,
        kernel_height = 3, kernel_width = 3,
        input_elements = batch * input_height * input_width * input_channels,
        weight_elements = output_channels * kernel_height * kernel_width * input_per_group,
        output_elements = batch * output_height * output_width * output_channels,
    };
    int8_t input[input_elements];
    int8_t weight[weight_elements];
    int32_t bias[output_channels];
    float weight_scales[output_channels];
    int32_t weight_zero_points[output_channels];
    int8_t reference[output_elements];
    int8_t single_thread[output_elements];
    int8_t threaded[output_elements];
    for (int index = 0; index < input_elements; index++)
        input[index] = (int8_t)(((index * 29 + 17) % 221) - 110);
    for (int index = 0; index < weight_elements; index++)
        weight[index] = (int8_t)(((index * 37 + 11) % 211) - 105);
    for (int channel = 0; channel < output_channels; channel++) {
        bias[channel] = channel * 41 - 503;
        weight_scales[channel] = 1.0f / (float)(48 + channel % 7 * 8);
        weight_zero_points[channel] = channel % 13 - 6;
    }
    CHECK(qconv2d_i8u8(input, weight, bias, weight_scales,
                       weight_zero_points, reference, batch, input_height,
                       input_width, input_channels, output_height, output_width,
                       output_channels, kernel_height, kernel_width,
                       input_per_group, 1, 1, 1, 1, 1, 1, 1, 1, groups, 2,
                       1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                       2u, 2u, 2u) == 1);
    vx_set_num_threads(1);
    CHECK(vx_qconv2d_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, single_thread, batch,
                                 input_height, input_width, input_channels,
                                 output_height, output_width, output_channels,
                                 kernel_height, kernel_width, input_per_group,
                                 1, 1, 1, 1, 1, 1, 1, 1, groups, 2,
                                 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                                 2u, 2u, 2u) == 1);
    vx_set_num_threads(4);
    CHECK(vx_qconv2d_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, threaded, batch,
                                 input_height, input_width, input_channels,
                                 output_height, output_width, output_channels,
                                 kernel_height, kernel_width, input_per_group,
                                 1, 1, 1, 1, 1, 1, 1, 1, groups, 2,
                                 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                                 2u, 2u, 2u) == 1);
    vx_set_num_threads(0);
    CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
    CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
    return 0;
}

/* TinyReceipt starts with a grayscale 3x3/stride-2 convolution.  Its one input
 * channel does not fill one AVX2 lane block, so the native path amortizes the
 * shared input/tap traversal across eight output channels and parallelizes
 * spatial locations.  Cover that small-C path on both sides of the pool. */
static int test_physical_qconv_native_grayscale_stem_dispatch(void) {
    enum {
        batch = 1, input_height = 129, input_width = 131, input_channels = 1,
        output_height = 65, output_width = 66, output_channels = 48,
        kernel_height = 3, kernel_width = 3,
        input_elements = batch * input_height * input_width * input_channels,
        weight_elements = output_channels * kernel_height * kernel_width *
                          input_channels,
        output_elements = batch * output_height * output_width * output_channels,
    };
    int8_t input_i8[input_elements];
    uint8_t input_u8[input_elements];
    int8_t weight_i8[weight_elements];
    uint8_t weight_u8[weight_elements];
    int32_t bias[output_channels];
    float weight_scales[output_channels];
    int32_t weight_zp_i8[output_channels];
    int32_t weight_zp_u8[output_channels];
    uint8_t reference[output_elements];
    uint8_t single_thread[output_elements];
    uint8_t threaded[output_elements];
    for (int index = 0; index < input_elements; index++) {
        input_i8[index] = (int8_t)(((index * 31 + 7) % 255) - 127);
        input_u8[index] = (uint8_t)((index * 37 + 13) % 251);
    }
    for (int index = 0; index < weight_elements; index++) {
        weight_i8[index] = (int8_t)(((index * 43 + 19) % 251) - 125);
        weight_u8[index] = (uint8_t)((index * 47 + 23) % 253);
    }
    for (int channel = 0; channel < output_channels; channel++) {
        bias[channel] = channel * 101 - 1703;
        weight_scales[channel] = 1.0f / (float)(64 + channel % 5 * 8);
        weight_zp_i8[channel] = channel % 11 - 5;
        weight_zp_u8[channel] = 121 + channel % 11;
    }
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void *input = input_kind ? (const void *)input_u8 :
                                         (const void *)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zero_point = input_kind ? 127 : -3;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void *weight = weight_kind ? (const void *)weight_u8 :
                                               (const void *)weight_i8;
            const uint32_t weight_dtype = weight_kind ? 3u : 2u;
            const int32_t *weight_zero_points = weight_kind ? weight_zp_u8 :
                                                              weight_zp_i8;
            for (int output_kind = 0; output_kind < 2; output_kind++) {
                const uint32_t output_dtype = output_kind ? 3u : 2u;
                const int32_t output_zero_point = output_kind ? 117 : -2;
                CHECK(qconv2d_i8u8(input, weight, bias, weight_scales,
                                   weight_zero_points, reference, batch,
                                   input_height, input_width, input_channels,
                                   output_height, output_width, output_channels,
                                   kernel_height, kernel_width, input_channels,
                                   2, 2, 1, 1, 1, 1, 1, 1, 1, 2,
                                   1.0f / 32.0f, input_zero_point,
                                   1.0f / 16.0f, output_zero_point,
                                   input_dtype, weight_dtype, output_dtype) == 1);
                vx_set_num_threads(1);
                CHECK(vx_qconv2d_i8u8_native(input, weight, bias,
                                             weight_scales, weight_zero_points,
                                             single_thread, batch, input_height,
                                             input_width, input_channels,
                                             output_height, output_width,
                                             output_channels, kernel_height,
                                             kernel_width, input_channels,
                                             2, 2, 1, 1, 1, 1, 1, 1, 1, 2,
                                             1.0f / 32.0f, input_zero_point,
                                             1.0f / 16.0f, output_zero_point,
                                             input_dtype, weight_dtype,
                                             output_dtype) == 1);
                vx_set_num_threads(4);
                CHECK(vx_qconv2d_i8u8_native(input, weight, bias,
                                             weight_scales, weight_zero_points,
                                             threaded, batch, input_height,
                                             input_width, input_channels,
                                             output_height, output_width,
                                             output_channels, kernel_height,
                                             kernel_width, input_channels,
                                             2, 2, 1, 1, 1, 1, 1, 1, 1, 2,
                                             1.0f / 32.0f, input_zero_point,
                                             1.0f / 16.0f, output_zero_point,
                                             input_dtype, weight_dtype,
                                             output_dtype) == 1);
                CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
                CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
            }
        }
    }
    vx_set_num_threads(0);
    return 0;
}

/* Verify VPDPBUSD compensation per output location. The asymmetric pad means
 * border outputs include different kernel masks, so K/sum(A)/sum(B) must count
 * only valid taps rather than the complete OHWI kernel. */
static int test_physical_qconv_vnni_remapping(void) {
    enum {
        input_height = 2, input_width = 3, input_per_group = 67,
        output_height = 2, output_width = 4, output_channels = 2,
        kernel_height = 2, kernel_width = 2,
        input_elements = input_height * input_width * input_per_group,
        weight_elements = output_channels * kernel_height * kernel_width * input_per_group,
    };
    int8_t input_i8[input_elements];
    uint8_t input_u8[input_elements];
    int8_t weight_i8[weight_elements];
    uint8_t weight_u8[weight_elements];
    const int32_t bias[output_channels] = {17, -29};
    const int32_t weight_zp_i8[output_channels] = {-128, 127};
    const int32_t weight_zp_u8[output_channels] = {0, 255};
    for (int index = 0; index < input_elements; index++) {
        input_i8[index] = (int8_t)((index & 1) ? 127 : -128);
        input_u8[index] = (uint8_t)((index & 1) ? 255 : 0);
    }
    for (int index = 0; index < weight_elements; index++) {
        weight_i8[index] = (int8_t)((index & 1) ? 127 : -128);
        weight_u8[index] = (uint8_t)((index & 1) ? 255 : 0);
    }
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void* input = input_kind ? (const void*)input_u8 : (const void*)input_i8;
        const uint32_t input_dtype = input_kind ? 3u : 2u;
        const int32_t input_zero_point = input_kind ? 255 : -128;
        const int32_t input_zero_unsigned = input_zero_point + (input_dtype == 2u ? 128 : 0);
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ? 3u : 2u;
            const int32_t* weight_zero_points = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (uint32_t output_y = 0; output_y < output_height; output_y++) {
                for (uint32_t output_x = 0; output_x < output_width; output_x++) {
                    for (uint32_t output_channel = 0; output_channel < output_channels;
                            output_channel++) {
                        const int32_t weight_zero_signed = weight_zero_points[output_channel] -
                            (weight_dtype == 3u ? 128 : 0);
                        int64_t conventional = bias[output_channel];
                        int64_t dot = 0, input_sum = 0, weight_sum = 0;
                        uint32_t valid_terms = 0;
                        for (uint32_t kernel_y = 0; kernel_y < kernel_height; kernel_y++) {
                            const int64_t input_y = (int64_t)output_y + kernel_y - 1;
                            if (input_y < 0 || input_y >= input_height) continue;
                            for (uint32_t kernel_x = 0; kernel_x < kernel_width; kernel_x++) {
                                const int64_t input_x = (int64_t)output_x + kernel_x - 1;
                                if (input_x < 0 || input_x >= input_width) continue;
                                for (uint32_t channel = 0; channel < input_per_group; channel++) {
                                    const size_t input_index = ((size_t)input_y * input_width +
                                        (uint32_t)input_x) * input_per_group + channel;
                                    const size_t weight_index = (((size_t)output_channel * kernel_height +
                                        kernel_y) * kernel_width + kernel_x) * input_per_group + channel;
                                    const int32_t input_raw = input_dtype == 2u
                                        ? (int32_t)((const int8_t*)input)[input_index]
                                        : (int32_t)((const uint8_t*)input)[input_index];
                                    const int32_t weight_raw = weight_dtype == 2u
                                        ? (int32_t)((const int8_t*)weight)[weight_index]
                                        : (int32_t)((const uint8_t*)weight)[weight_index];
                                    const int32_t input_unsigned = input_dtype == 2u
                                        ? input_raw + 128 : input_raw;
                                    const int32_t weight_signed = weight_dtype == 3u
                                        ? weight_raw - 128 : weight_raw;
                                    conventional += (int64_t)(input_raw - input_zero_point) *
                                        (int64_t)(weight_raw - weight_zero_points[output_channel]);
                                    dot += (int64_t)input_unsigned * weight_signed;
                                    input_sum += input_unsigned;
                                    weight_sum += weight_signed;
                                    valid_terms++;
                                }
                            }
                        }
                        CHECK((int64_t)bias[output_channel] + dot -
                            (int64_t)weight_zero_signed * input_sum -
                            (int64_t)input_zero_unsigned * weight_sum +
                            (int64_t)valid_terms * input_zero_unsigned * weight_zero_signed ==
                            conventional);
                    }
                }
            }
        }
    }
    return 0;
}

/* On an ARM NEON build this covers grouped NHWC/OHWI convolution with a
 * contiguous 16-channel SDOT-eligible block plus a scalar tail, asymmetric
 * pads, dilation, stride, mixed U8/I8 operands, and ReLU6. Other hosts verify
 * that the isolated ARM candidate cleanly declines without touching the result. */
static int test_physical_qconv_arm_try(void) {
    enum {
        batch = 1, input_height = 4, input_width = 5, input_channels = 38,
        output_height = 4, output_width = 3, output_channels = 4,
        kernel_height = 2, kernel_width = 2, input_per_group = 19, groups = 2,
    };
    uint8_t input[batch * input_height * input_width * input_channels];
    int8_t weight[output_channels * kernel_height * kernel_width * input_per_group];
    const float weight_scales[output_channels] = {0.03125f, 0.046875f, 0.0625f, 0.078125f};
    const int32_t weight_zero_points[output_channels] = {-17, 9, -5, 21};
    const int32_t bias[output_channels] = {37, -53, 211, -97};
    uint8_t reference[batch * output_height * output_width * output_channels] = {0};
    uint8_t arm_actual[batch * output_height * output_width * output_channels] = {0};
    for (size_t index = 0; index < sizeof(input); index++) input[index] = (uint8_t)((index * 31u + 17u) % 251u);
    for (size_t index = 0; index < sizeof(weight); index++) weight[index] =
        (int8_t)(((index * 29u + 11u) % 201u) - 100);
    CHECK(qconv2d_i8u8(input, weight, bias, weight_scales, weight_zero_points,
                        reference, batch, input_height, input_width, input_channels,
                        output_height, output_width, output_channels, kernel_height,
                        kernel_width, input_per_group, 1u, 2u, 2u, 1u,
                        1u, 1u, 1u, 0u, groups, 2u, 0.0625f, 127,
                        0.125f, 121, 3u, 2u, 3u) == 1);
    {
        int arm_handled = vx_qconv2d_i8u8_arm_try(
            input, weight, bias, weight_scales, weight_zero_points, arm_actual,
            batch, input_height, input_width, input_channels, output_height,
            output_width, output_channels, kernel_height, kernel_width,
            input_per_group, 1u, 2u, 2u, 1u, 1u, 1u, 1u, 0u, groups, 2u,
            0.0625f, 127, 0.125f, 121, 3u, 2u, 3u);
        CHECK(arm_handled == 0 || arm_handled == 1);
        if (arm_handled) CHECK(memcmp(reference, arm_actual, sizeof(reference)) == 0);
    }
    {
        const int8_t overflow_input[8] = {127, 127, 127, 127, 127, 127, 127, 127};
        const int8_t overflow_weight[8] = {127, 127, 127, 127, 127, 127, 127, 127};
        const int32_t overflow_bias[1] = {INT32_MAX};
        const float unit_scale[1] = {1.0f};
        const int32_t minus_128[1] = {-128};
        int8_t reference_overflow = 51;
        int8_t arm_overflow = 51;
        CHECK(qconv2d_i8u8(overflow_input, overflow_weight, overflow_bias,
                            unit_scale, minus_128, &reference_overflow,
                            1u, 1u, 1u, 8u, 1u, 1u, 1u, 1u, 1u, 8u,
                            1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u, 1u, 0u,
                            1.0f, -128, 1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(vx_qconv2d_i8u8_arm_try(overflow_input, overflow_weight,
                                       overflow_bias, unit_scale, minus_128,
                                       &arm_overflow, 1u, 1u, 1u, 8u, 1u,
                                       1u, 1u, 1u, 1u, 8u, 1u, 1u, 1u, 1u,
                                       0u, 0u, 0u, 0u, 1u, 0u, 1.0f, -128,
                                       1.0f, 0, 2u, 2u, 2u) == 0);
        CHECK(reference_overflow == 51 && arm_overflow == 51);
    }
    return 0;
}

static int test_physical_qlinear_i8_to_u8(void) {
    const char* config_path = "/tmp/volvox-physical-qlinear-i8-u8-config.json";
    const char* saved_config_path = "/tmp/volvox-physical-qlinear-i8-u8-saved.json";
    const char* weights_path = "/tmp/volvox-physical-qlinear-i8-u8-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}}},"
        "\"weights_quantization\":{\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[0.5,0.5,0.5],\"zero_points\":[0,0,0]}},\"nodes\":[{"
        "\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":128}},\"params\":{}}]}";
    const int weight_shape[2] = {3, 1};
    const int bias_shape[1] = {3};
    const int8_t weight[3] = {1, 3, 1};
    const int32_t bias[3] = {0, 0, 1000};
    const int8_t input[1] = {1};
    uint8_t output[3] = {0, 0, 0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    /* Place I32 bias after three I8 bytes so the runtime must not assume the
       safetensors payload has native I32 alignment. */
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_I8 && y->dtype == T_U8 &&
          x->elem_size == 1 && y->elem_size == 1 &&
          x->quantization.valid && y->quantization.valid &&
          closef(x->quantization.scale, 1.0f) && x->quantization.zero_point == 0 &&
          closef(y->quantization.scale, 1.0f) && y->quantization.zero_point == 128);
    CHECK(g_qt[x - g_t].data == NULL && !g_qt[x - g_t].has_params &&
          g_qt[y - g_t].data == NULL && !g_qt[y - g_t].has_params);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* 1 * {1,3} * 0.5 + 128 = {128.5,129.5}; ties round to even.
       The I32 bias on the last channel also verifies U8 saturation. */
    CHECK(output[0] == 128 && output[1] == 130 && output[2] == 255);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_save_config(saved_config_path) == 0);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(saved_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(output[0] == 128 && output[1] == 130 && output[2] == 255);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(saved_config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qembedding_i8_to_u8(void) {
    const char* config_path = "/tmp/volvox-physical-qembedding-i8-u8-config.json";
    const char* saved_config_path = "/tmp/volvox-physical-qembedding-i8-u8-saved.json";
    const char* weights_path = "/tmp/volvox-physical-qembedding-i8-u8-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"ids\":{\"shape\":[2,2],\"dtype\":\"int32\"}},"
        "\"weights_quantization\":{\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[0.5,0.25,1.0],\"zero_points\":[0,1,-1]}},\"nodes\":[{"
        "\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},\"params\":{}}]}";
    const int table_shape[2] = {3, 3};
    const int8_t table[9] = {
        0, 1, -1,
        2, -2, 4,
        99, -101, 4,
    };
    const int32_t ids_first[4] = {1, 0, 2, 1};
    const int32_t ids_second[4] = {2, 0, 0, 2};
    const int32_t ids_invalid[4] = {1, 3, 0, 2};
    const uint8_t expected_first[12] = {
        129, 125, 131, 128, 130, 126,
        255, 0, 148, 129, 125, 131,
    };
    const uint8_t expected_second[12] = {
        255, 0, 148, 128, 130, 126,
        128, 130, 126, 255, 0, 148,
    };
    uint8_t output[12] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 table, sizeof(table)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* ids = t_find("ids");
    T* y = t_find("y");
    CHECK(ids && y && ids->is_graph_input && ids->dtype == T_I32 && y->dtype == T_U8 &&
          ids->elem_size == sizeof(int32_t) && y->elem_size == 1 &&
          y->quantization.valid && closef(y->quantization.scale, 0.25f) &&
          y->quantization.zero_point == 128 && g_qembedding_meta[0].valid &&
          g_qembedding_meta[0].vocab == 3u && g_qembedding_meta[0].hidden == 3u &&
          g_qembedding_meta[0].weight_dtype == T_I8 &&
          g_qembedding_meta[0].output_dtype == T_U8 &&
          g_qt[ids - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, ids_first, sizeof(ids_first)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_first, sizeof(output)) == 0);
    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, ids_second, sizeof(ids_second)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_second, sizeof(output)) == 0);
    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, ids_invalid, sizeof(ids_invalid)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_second, sizeof(output)) == 0);
    CHECK(volvoxai_engine_save_config(saved_config_path) == 0);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_init(saved_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, ids_first, sizeof(ids_first)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_first, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(saved_config_path);
    remove(weights_path);
    return 0;
}

/* The first graph invocation creates the I32 ID device slot. Changing IDs
 * before the second invocation must upload the new host bytes rather than
 * reuse that stale slot; this is the regression for T_I32 host-dirty tracking. */
static int test_physical_qembedding_vulkan_changed_ids(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qembedding-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qembedding-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"ids\":{\"shape\":[4],\"dtype\":\"int32\"}},"
        "\"weights_quantization\":{\"table\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[0.5,0.25,1.0],\"zero_points\":[0,1,-1]}},\"nodes\":[{"
        "\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[4,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},\"params\":{}}]}";
    const int table_shape[2] = {3, 3};
    const int8_t table[9] = {0, 1, -1, 2, -2, 4, 99, -101, 4};
    const int32_t first_ids[4] = {1, 0, 2, 1};
    const int32_t second_ids[4] = {2, 0, 0, 2};
    const uint8_t expected_first[12] = {
        129, 125, 131, 128, 130, 126,
        255, 0, 148, 129, 125, 131,
    };
    const uint8_t expected_second[12] = {
        255, 0, 148, 128, 130, 126,
        128, 130, 126, 255, 0, 148,
    };
    uint8_t output[12] = {0};
    SafetensorsFile file;
    FILE* config_file;
    if (vk_init() != 0) {
        puts("quantized Vulkan QEmbedding ID regression skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 table, sizeof(table)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* y = t_find("y");
    CHECK(y && y->dtype == T_U8);
    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, first_ids, sizeof(first_ids)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    /* GPU output is intentionally not materialized until the public copy. */
    CHECK(((const uint8_t*)y->data)[0] == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_first, sizeof(output)) == 0);

    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, second_ids, sizeof(second_ids)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(((const uint8_t*)y->data)[0] == expected_first[0]);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_second, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qmatmul_u8_to_i8(void) {
    const char* config_path = "/tmp/volvox-physical-qmatmul-u8-i8-config.json";
    const char* weights_path = "/tmp/volvox-physical-qmatmul-u8-i8-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[2,2],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}}},"
        "\"weights_quantization\":{\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[0.5,0.25],\"zero_points\":[128,128]}},\"nodes\":[{"
        "\"opType\":\"QMatMul\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-3}},\"params\":{}}]}";
    const int weight_shape[2] = {2, 2};
    const int bias_shape[1] = {2};
    const uint8_t weight[4] = {130, 126, 127, 129};
    const int32_t bias[2] = {1, -2};
    const uint8_t input[4] = {132, 124, 128, 128};
    int8_t output[4] = {0, 0, 0, 0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_U8, weight_shape, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && y->quantization.valid &&
          x->quantization.zero_point == 128 && y->quantization.zero_point == -3);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(output[0] == 14 && output[1] == -8 && output[2] == -2 && output[3] == -4);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qconv2d_grouped_relu6(void) {
    const char* config_path = "/tmp/volvox-physical-qconv-grouped-config.json";
    const char* saved_config_path = "/tmp/volvox-physical-qconv-grouped-saved.json";
    const char* weights_path = "/tmp/volvox-physical-qconv-grouped-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[2,3,3,2],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}}},"
        "\"weights_quantization\":{\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[0.25,0.5],\"zero_points\":[128,127]}},\"nodes\":[{"
        "\"opType\":\"QConv2D\",\"inputs\":{\"x\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3,2,2]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":123}},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\",\"groups\":2,"
        "\"stride\":[1,2],\"dilation\":[2,1],\"pads\":[1,1,1,0],\"relu\":2}}]}";
    const int weight_shape[4] = {2, 2, 2, 1};
    const uint8_t weight[8] = {130, 126, 132, 124, 129, 125, 128, 126};
    const uint8_t input[36] = {
        128, 130, 132, 126, 124, 134, 136, 122, 120, 138, 140, 118, 128, 128, 130, 126, 126, 132,
        134, 122, 122, 136, 138, 120, 128, 130, 132, 126, 124, 134, 136, 122, 120, 138, 140, 118,
    };
    const uint8_t expected[24] = {
        123, 126, 123, 133, 123, 123, 131, 123, 123, 129, 123, 143,
        123, 123, 131, 123, 123, 132, 123, 147, 123, 123, 127, 123,
    };
    uint8_t output[24] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_U8, weight_shape, 4,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_U8 && y->dtype == T_U8 && x->elem_size == 1 && y->elem_size == 1 &&
          x->quantization.valid && y->quantization.valid && x->quantization.zero_point == 128 &&
          y->quantization.zero_point == 123);
    CHECK(g_qconv_meta[0].valid && g_qconv_meta[0].bias == NULL && g_qconv_meta[0].groups == 2 &&
          g_qconv_meta[0].stride_y == 1 && g_qconv_meta[0].stride_x == 2 &&
          g_qconv_meta[0].dilation_y == 2 && g_qconv_meta[0].dilation_x == 1 &&
          g_qconv_meta[0].padding_top == 1 && g_qconv_meta[0].padding_left == 1 &&
          g_qconv_meta[0].padding_bottom == 1 && g_qconv_meta[0].padding_right == 0 &&
          g_qconv_meta[0].relu == 2);
    CHECK(g_qt[x - g_t].data == NULL && !g_qt[x - g_t].has_params &&
          g_qt[y - g_t].data == NULL && !g_qt[y - g_t].has_params);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* Nonzero input ZP makes the asymmetric padding observable; the final 147
       is the ReLU6 output-domain cap at scale=.25 and ZP=123. */
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_save_config(saved_config_path) == 0);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(saved_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(saved_config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qconv2d_i8_bias_relu(void) {
    const char* config_path = "/tmp/volvox-physical-qconv-i8-config.json";
    const char* weights_path = "/tmp/volvox-physical-qconv-i8-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,2,1],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}}},"
        "\"weights_quantization\":{\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[0.5,0.5,0.5],\"zero_points\":[0,0,0]}},\"nodes\":[{"
        "\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,2,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":10}},\"params\":{\"relu\":1}}]}";
    const int weight_shape[4] = {3, 1, 1, 1};
    const int bias_shape[1] = {3};
    const int8_t weight[3] = {1, 1, 1};
    const int32_t bias[3] = {0, 1000, -1000};
    const int8_t input[2] = {-1, 3};
    const uint8_t expected[6] = {10, 255, 10, 12, 255, 10};
    uint8_t output[6] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    /* The three-byte I8 payload makes the following I32 tensor unaligned in
       the saved data region; QConv metadata must own an aligned copy. */
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8, weight_shape, 4,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    T* b = t_find("b");
    CHECK(x && y && b && x->dtype == T_I8 && y->dtype == T_U8 && b->dtype == T_I32 &&
          x->quantization.valid && y->quantization.valid && g_qconv_meta[0].valid &&
          g_qconv_meta[0].bias != NULL && g_qconv_meta[0].relu == 1);
    CHECK(((uintptr_t)b->data % sizeof(int32_t)) != 0);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* Channel 0 covers nearest-even .5 ties, channel 1 saturates, and channel
       2 proves ReLU clamps to the nonzero output zero point. */
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qadd_and_requantize(void) {
    const char* config_path = "/tmp/volvox-physical-qadd-config.json";
    const char* weights_path = "/tmp/volvox-physical-qadd-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"a\":{\"shape\":[4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}},"
        "\"b\":{\"shape\":[4],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},"
        "\"zero\":{\"shape\":[4],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},"
        "\"nodes\":["
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":3}},\"params\":{}},"
        "{\"opType\":\"RequantizeLinear\",\"inputs\":{\"input\":\"sum\"},"
        "\"outputs\":{\"out\":\"rq\"},\"outputs_shape\":{\"out\":[4]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"rq\",\"b\":\"zero\"},"
        "\"outputs\":{\"out\":\"relu\"},\"outputs_shape\":{\"out\":[4]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"relu\":1}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"rq\",\"b\":\"zero\"},"
        "\"outputs\":{\"out\":\"cap\"},\"outputs_shape\":{\"out\":[4]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"relu\":2}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t a[4] = {1, 3, -1, 127};
    const uint8_t b[4] = {128, 128, 124, 255};
    const uint8_t zero[4] = {120, 120, 120, 120};
    const int8_t expected_sum[4] = {4, 6, -2, 127};
    const uint8_t expected_rq[4] = {120, 122, 118, 182};
    const uint8_t expected_relu[4] = {120, 122, 120, 182};
    const uint8_t expected_cap[4] = {120, 122, 120, 132};
    int8_t sum[4] = {0};
    uint8_t rq[4] = {0};
    uint8_t relu[4] = {0};
    uint8_t cap[4] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* ta = t_find("a");
    T* tb = t_find("b");
    T* tz = t_find("zero");
    T* tsum = t_find("sum");
    T* trq = t_find("rq");
    T* trelu = t_find("relu");
    T* tcap = t_find("cap");
    CHECK(ta && tb && tz && tsum && trq && trelu && tcap &&
          ta->dtype == T_I8 && tb->dtype == T_U8 && tsum->dtype == T_I8 &&
          trq->dtype == T_U8 && trelu->dtype == T_U8 && tcap->dtype == T_U8 &&
          tsum->quantization.valid && trq->quantization.valid &&
          tsum->quantization.zero_point == 3 && trq->quantization.zero_point == 120);
    CHECK(g_qt[ta - g_t].data == NULL && g_qt[tb - g_t].data == NULL &&
          g_qt[tsum - g_t].data == NULL && g_qt[trq - g_t].data == NULL &&
          g_qt[trelu - g_t].data == NULL && g_qt[tcap - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("a", T_I8, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", T_U8, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_set_input_raw("zero", T_U8, zero, sizeof(zero)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("sum", sum, sizeof(sum)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("rq", rq, sizeof(rq)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("relu", relu, sizeof(relu)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("cap", cap, sizeof(cap)) == 0);
    /* Requantization covers nearest-even ties: 120.5, 121.5 and 117.5.
       The two QAdd forms then exercise nonzero-domain ReLU and ReLU6. */
    CHECK(memcmp(sum, expected_sum, sizeof(sum)) == 0);
    CHECK(memcmp(rq, expected_rq, sizeof(rq)) == 0);
    CHECK(memcmp(relu, expected_relu, sizeof(relu)) == 0);
    CHECK(memcmp(cap, expected_cap, sizeof(cap)) == 0);
    CHECK(g_qt[tsum - g_t].data == NULL && g_qt[trq - g_t].data == NULL &&
          g_qt[trelu - g_t].data == NULL && g_qt[tcap - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* Static byte constants are model weights, not graph inputs.  Their
 * per-tensor mapping still has to reach QAdd; this mirrors quantized position
 * and type embeddings in a fixed-shape W8A8 encoder-decoder package. */
static int test_physical_qadd_static_weight_quantization(void) {
    const char* config_path = "/tmp/volvox-physical-qadd-static-config.json";
    const char* weights_path = "/tmp/volvox-physical-qadd-static-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}}},"
        "\"weights_quantization\":{\"constant\":{\"scheme\":\"per_tensor\","
        "\"scale\":0.25,\"zero_point\":0}},\"nodes\":["
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"x\",\"b\":\"constant\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}},\"params\":{}}]}";
    const int shape[1] = {4};
    const int8_t constant[4] = {1, 2, -3, 4};
    const int8_t input[4] = {1, 2, 3, -4};
    const int8_t expected[4] = {2, 4, 0, 0};
    int8_t output[4] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "constant", SAFETENSORS_DTYPE_I8, shape, 1,
                                 constant, sizeof(constant)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* constant_tensor = t_find("constant");
    CHECK(constant_tensor && constant_tensor->dtype == T_I8 && constant_tensor->quantization.valid &&
          closef(constant_tensor->quantization.scale, 0.25f) &&
          constant_tensor->quantization.zero_point == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* The engine must keep both QSiLU boundaries physical bytes.  A pair of
 * differently-quantized activations catches accidental F32/QTensor fallback
 * between nodes as well as the descriptor handoff into the portable kernel. */
static int test_physical_qsilu_chain(void) {
    const char* config_path = "/tmp/volvox-physical-qsilu-config.json";
    const char* invalid_config_path = "/tmp/volvox-physical-qsilu-invalid-input-config.json";
    const char* weights_path = "/tmp/volvox-physical-qsilu-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":["
        "{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},\"params\":{}},"
        "{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"hidden\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},\"params\":{}}]}";
    const char* invalid_input_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":[{\"opType\":\"QSiLU\",\"inputs\":{\"unsupported\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},\"params\":{}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[5] = {-8, -2, 0, 2, 8};
    const uint8_t expected_hidden[5] = {128, 127, 128, 131, 144};
    const int8_t expected_output[5] = {-4, -5, -4, 0, 27};
    uint8_t hidden[5] = {0};
    int8_t output[5] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    CHECK(x && h && y && x->dtype == T_I8 && h->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          h->quantization.scale == 0.25f && h->quantization.zero_point == 128 &&
          y->quantization.scale == 0.125f && y->quantization.zero_point == -4);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[h - g_t].data == NULL &&
          g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    CHECK(g_qt[h - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);

    config_file = fopen(invalid_config_path, "wb");
    CHECK(config_file != NULL && fwrite(invalid_input_config, 1, strlen(invalid_input_config),
                                        config_file) == strlen(invalid_input_config));
    CHECK(fclose(config_file) == 0);
    CHECK(volvoxai_engine_init(invalid_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    /* `unsupported` is deliberately not one of input/x/data. The strict
       physical route must reject it rather than treating the sole edge as a
       generic F32 SiLU or silently reading a different tensor. */
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(invalid_config_path);
    remove(weights_path);
    return 0;
}

/* QGELU's graph spelling may omit params or state the one permitted
 * `approximate: "none"` value. Both routes retain raw activation bytes; tanh
 * plus unknown/duplicate attributes and invalid input names must fail before
 * the graph can enter an F32 GELU. */
static int test_physical_qgelu_chain(void) {
    const char* config_path = "/tmp/volvox-physical-qgelu-config.json";
    const char* omitted_params_config_path = "/tmp/volvox-physical-qgelu-omitted-params-config.json";
    const char* tanh_config_path = "/tmp/volvox-physical-qgelu-tanh-config.json";
    const char* unknown_config_path = "/tmp/volvox-physical-qgelu-unknown-config.json";
    const char* duplicate_config_path = "/tmp/volvox-physical-qgelu-duplicate-config.json";
    const char* invalid_input_config_path = "/tmp/volvox-physical-qgelu-invalid-input-config.json";
    const char* weights_path = "/tmp/volvox-physical-qgelu-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":["
        "{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":128}},"
        "\"params\":{\"approximate\":\"none\"}},"
        "{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"hidden\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},\"params\":{}}]}";
    const char* omitted_params_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":[{\"opType\":\"QGELU\",\"inputs\":{\"data\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-3}}}]}";
    const char* invalid_tanh_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":[{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"approximate\":\"tanh\"}}]}";
    const char* invalid_unknown_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":[{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"approximate\":\"none\",\"unknown\":1}}]}";
    const char* invalid_duplicate_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":[{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"approximate\":\"none\",\"approximate\":\"none\"}}]}";
    const char* invalid_input_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":[{\"opType\":\"QGELU\",\"inputs\":{\"unsupported\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"approximate\":\"none\"}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[5] = {-8, -2, 0, 2, 8};
    const uint8_t expected_hidden[5] = {128, 127, 128, 135, 160};
    const int8_t expected_output[5] = {-4, -4, -4, 2, 28};
    const int8_t expected_omitted_output[5] = {-3, -4, -3, 4, 29};
    uint8_t hidden[5] = {0};
    int8_t output[5] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    CHECK(x && h && y && x->dtype == T_I8 && h->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          h->quantization.scale == 0.125f && h->quantization.zero_point == 128 &&
          y->quantization.scale == 0.125f && y->quantization.zero_point == -4);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[h - g_t].data == NULL &&
          g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    CHECK(g_qt[h - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);

    /* A missing params member is the graph spelling paired with the `{}`
       serialized form above. `data` is the third permitted activation alias. */
    config_file = fopen(omitted_params_config_path, "wb");
    CHECK(config_file != NULL && fwrite(omitted_params_config, 1,
                                        strlen(omitted_params_config), config_file) ==
                                        strlen(omitted_params_config));
    CHECK(fclose(config_file) == 0);
    memset(output, 0, sizeof(output));
    CHECK(volvoxai_engine_init(omitted_params_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_omitted_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(omitted_params_config_path);

    config_file = fopen(tanh_config_path, "wb");
    CHECK(config_file != NULL && fwrite(invalid_tanh_config, 1, strlen(invalid_tanh_config),
                                        config_file) == strlen(invalid_tanh_config));
    CHECK(fclose(config_file) == 0);
    CHECK(volvoxai_engine_init(tanh_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(tanh_config_path);

    config_file = fopen(unknown_config_path, "wb");
    CHECK(config_file != NULL && fwrite(invalid_unknown_config, 1,
                                        strlen(invalid_unknown_config), config_file) ==
                                        strlen(invalid_unknown_config));
    CHECK(fclose(config_file) == 0);
    CHECK(volvoxai_engine_init(unknown_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(unknown_config_path);

    config_file = fopen(duplicate_config_path, "wb");
    CHECK(config_file != NULL && fwrite(invalid_duplicate_config, 1,
                                        strlen(invalid_duplicate_config), config_file) ==
                                        strlen(invalid_duplicate_config));
    CHECK(fclose(config_file) == 0);
    CHECK(volvoxai_engine_init(duplicate_config_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(duplicate_config_path);

    config_file = fopen(invalid_input_config_path, "wb");
    CHECK(config_file != NULL && fwrite(invalid_input_config, 1,
                                        strlen(invalid_input_config), config_file) ==
                                        strlen(invalid_input_config));
    CHECK(fclose(config_file) == 0);
    CHECK(volvoxai_engine_init(invalid_input_config_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(invalid_input_config_path);
    remove(weights_path);
    return 0;
}

/* QGroupNorm keeps an I8/U8 NHWC activation island while loading only its
 * [C] gamma/beta tensors as F32.  The one-byte prefix deliberately leaves the
 * first F32 affine payload potentially unaligned in SafeTensors storage. */
static int test_physical_qgroupnorm_chain(void) {
    const char* config_path = "/tmp/volvox-physical-qgroupnorm-config.json";
    const char* unknown_config_path = "/tmp/volvox-physical-qgroupnorm-unknown-config.json";
    const char* duplicate_config_path = "/tmp/volvox-physical-qgroupnorm-duplicate-config.json";
    const char* alias_config_path = "/tmp/volvox-physical-qgroupnorm-alias-config.json";
    const char* layout_config_path = "/tmp/volvox-physical-qgroupnorm-layout-config.json";
    const char* malformed_affine_config_path = "/tmp/volvox-physical-qgroupnorm-affine-config.json";
    const char* missing_groups_config_path = "/tmp/volvox-physical-qgroupnorm-missing-groups-config.json";
    const char* weights_path = "/tmp/volvox-physical-qgroupnorm-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},"
        "\"nodes\":["
        "{\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":128}},"
        "\"params\":{\"num_groups\":1,\"eps\":null,\"data_layout\":null}},"
        "{\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1}}]}";
    const char* unknown_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1,\"groups\":1}}]}";
    const char* duplicate_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1,\"num_groups\":1}}]}";
    const char* alias_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QGroupNorm\",\"inputs\":{\"data\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1}}]}";
    const char* layout_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1,\"data_layout\":\"NCHW\"}}]}";
    const char* malformed_affine_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"bad\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1}}]}";
    const char* missing_groups_config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"eps\":0.00001}}]}";
    const int one[1] = {1};
    const int three[1] = {3};
    const int two[1] = {2};
    const uint8_t prefix[1] = {0};
    const int8_t input[3] = {-5, 0, 4};
    const float gamma1[3] = {1.0f, 0.5f, -0.75f};
    const float beta1[3] = {0.25f, -0.5f, 0.75f};
    const float gamma2[3] = {0.0f, 0.0f, 0.0f};
    const float beta2[3] = {0.25f, -0.5f, 0.75f};
    const float malformed_gamma[2] = {1.0f, 1.0f};
    const uint8_t expected_hidden[3] = {120, 124, 127};
    const int8_t expected_output[3] = {-2, -8, 2};
    uint8_t hidden[3] = {0};
    int8_t output[3] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "prefix", SAFETENSORS_DTYPE_U8, one, 1,
                                 prefix, sizeof(prefix)) == 0);
    CHECK(safetensors_add_tensor(&file, "g1", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma1, sizeof(gamma1)) == 0);
    CHECK(safetensors_add_tensor(&file, "b1", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta1, sizeof(beta1)) == 0);
    CHECK(safetensors_add_tensor(&file, "g2", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma2, sizeof(gamma2)) == 0);
    CHECK(safetensors_add_tensor(&file, "b2", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta2, sizeof(beta2)) == 0);
    CHECK(safetensors_add_tensor(&file, "bad", SAFETENSORS_DTYPE_F32, two, 1,
                                 malformed_gamma, sizeof(malformed_gamma)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    T* g1 = t_find("g1");
    T* b1 = t_find("b1");
    CHECK(x && h && y && g1 && b1 && x->dtype == T_I8 && h->dtype == T_U8 &&
          y->dtype == T_I8 && g1->dtype == T_F32 && b1->dtype == T_F32 &&
          g1->ndim == 1 && g1->shape[0] == 3 && b1->ndim == 1 && b1->shape[0] == 3 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          h->quantization.zero_point == 128 && y->quantization.zero_point == -4 &&
          g_qt[x - g_t].data == NULL && g_qt[h - g_t].data == NULL &&
          g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    CHECK(g_qt[h - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);

    const char* invalid_paths[] = {
        unknown_config_path, duplicate_config_path, alias_config_path, layout_config_path,
        malformed_affine_config_path, missing_groups_config_path,
    };
    const char* invalid_configs[] = {
        unknown_config, duplicate_config, alias_config, layout_config,
        malformed_affine_config, missing_groups_config,
    };
    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        config_file = fopen(invalid_paths[index], "wb");
        CHECK(config_file != NULL && fwrite(invalid_configs[index], 1,
                                             strlen(invalid_configs[index]), config_file) ==
                                         strlen(invalid_configs[index]));
        CHECK(fclose(config_file) == 0);
        int init_result = volvoxai_engine_init(invalid_paths[index], weights_path);
        if (!strcmp(invalid_paths[index], duplicate_config_path)) {
            CHECK(init_result != 0);
            volvoxai_engine_shutdown();
            remove(invalid_paths[index]);
            continue;
        }
        CHECK(init_result == 0);
        T* rejected = t_find("y");
        CHECK(rejected && rejected->data && rejected->dtype == T_I8);
        memset(rejected->data, 0x5a, 3u);
        CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_forward() != 0);
        for (int byte = 0; byte < 3; byte++)
            CHECK(((const uint8_t*)rejected->data)[byte] == 0x5a);
        volvoxai_engine_shutdown();
        remove(invalid_paths[index]);
    }
    remove(weights_path);
    return 0;
}

/* QLayerNorm accepts arbitrary-rank byte activations and owns the final axis
 * only. This rank-2 chain proves that F32 affine parameters do not force an
 * F32 activation boundary, including when SafeTensors starts them unaligned. */
static int test_physical_qlayernorm_chain(void) {
    const char* config_path = "/tmp/volvox-physical-qlayernorm-config.json";
    const char* unknown_config_path = "/tmp/volvox-physical-qlayernorm-unknown-config.json";
    const char* duplicate_config_path = "/tmp/volvox-physical-qlayernorm-duplicate-config.json";
    const char* dmodel_config_path = "/tmp/volvox-physical-qlayernorm-dmodel-config.json";
    const char* alias_config_path = "/tmp/volvox-physical-qlayernorm-alias-config.json";
    const char* malformed_affine_config_path = "/tmp/volvox-physical-qlayernorm-affine-config.json";
    const char* weights_path = "/tmp/volvox-physical-qlayernorm-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},"
        "\"nodes\":["
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":128}},"
        "\"params\":{\"eps\":null,\"d_model\":null}},"
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"d_model\":3}}]}";
    const char* unknown_config =
        "{\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"unknown\":1}}]}";
    const char* duplicate_config =
        "{\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"eps\":0.00001,\"eps\":0.00001}}]}";
    const char* dmodel_config =
        "{\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"d_model\":2}}]}";
    const char* alias_config =
        "{\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QLayerNorm\",\"inputs\":{\"data\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{}}]}";
    const char* malformed_affine_config =
        "{\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},\"nodes\":[{"
        "\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"bad\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{}}]}";
    const int one[1] = {1};
    const int three[1] = {3};
    const int two[1] = {2};
    const uint8_t prefix[1] = {0};
    const int8_t input[6] = {-5, 0, 4, 6, -2, 1};
    const float gamma1[3] = {1.0f, 0.5f, -0.75f};
    const float beta1[3] = {0.25f, -0.5f, 0.75f};
    const float gamma2[3] = {0.0f, 0.0f, 0.0f};
    const float beta2[3] = {0.25f, -0.5f, 0.75f};
    const float malformed_gamma[2] = {1.0f, 1.0f};
    const uint8_t expected_hidden[6] = {120, 124, 127, 141, 120, 135};
    const int8_t expected_output[6] = {-2, -8, 2, -2, -8, 2};
    uint8_t hidden[6] = {0};
    int8_t output[6] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "prefix", SAFETENSORS_DTYPE_U8, one, 1,
                                 prefix, sizeof(prefix)) == 0);
    CHECK(safetensors_add_tensor(&file, "g1", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma1, sizeof(gamma1)) == 0);
    CHECK(safetensors_add_tensor(&file, "b1", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta1, sizeof(beta1)) == 0);
    CHECK(safetensors_add_tensor(&file, "g2", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma2, sizeof(gamma2)) == 0);
    CHECK(safetensors_add_tensor(&file, "b2", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta2, sizeof(beta2)) == 0);
    CHECK(safetensors_add_tensor(&file, "bad", SAFETENSORS_DTYPE_F32, two, 1,
                                 malformed_gamma, sizeof(malformed_gamma)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    CHECK(x && h && y && x->ndim == 2 && x->shape[0] == 2 && x->shape[1] == 3 &&
          x->dtype == T_I8 && h->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          g_qt[x - g_t].data == NULL && g_qt[h - g_t].data == NULL &&
          g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    CHECK(g_qt[h - g_t].data == NULL && g_qt[y - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);

    const char* invalid_paths[] = {
        unknown_config_path, duplicate_config_path, dmodel_config_path,
        alias_config_path, malformed_affine_config_path,
    };
    const char* invalid_configs[] = {
        unknown_config, duplicate_config, dmodel_config, alias_config,
        malformed_affine_config,
    };
    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        config_file = fopen(invalid_paths[index], "wb");
        CHECK(config_file != NULL && fwrite(invalid_configs[index], 1,
                                             strlen(invalid_configs[index]), config_file) ==
                                         strlen(invalid_configs[index]));
        CHECK(fclose(config_file) == 0);
        int init_result = volvoxai_engine_init(invalid_paths[index], weights_path);
        if (!strcmp(invalid_paths[index], duplicate_config_path)) {
            CHECK(init_result != 0);
            volvoxai_engine_shutdown();
            remove(invalid_paths[index]);
            continue;
        }
        CHECK(init_result == 0);
        T* rejected = t_find("y");
        CHECK(rejected && rejected->data && rejected->dtype == T_I8);
        memset(rejected->data, 0x5a, 6u);
        CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_forward() != 0);
        for (int byte = 0; byte < 6; byte++)
            CHECK(((const uint8_t*)rejected->data)[byte] == 0x5a);
        volvoxai_engine_shutdown();
        remove(invalid_paths[index]);
    }
    remove(weights_path);
    return 0;
}

/* QSDPA owns independently quantized Q/K/V byte tensors.  This physical
 * graph deliberately makes B == Q and supplies rank-2 [B,K] mask storage so
 * the dispatcher must apply the documented batch-key precedence, rather than
 * silently treating it as [Q,K]. */
static int test_physical_qsdpa_chain(void) {
    const char* config_path = "/tmp/volvox-physical-qsdpa-config.json";
    const char* unknown_config_path = "/tmp/volvox-physical-qsdpa-unknown-config.json";
    const char* missing_causal_config_path =
        "/tmp/volvox-physical-qsdpa-missing-causal-config.json";
    const char* weights_path = "/tmp/volvox-physical-qsdpa-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"q\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"k\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"v\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"mask\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}},"
        "\"params\":{\"heads\":1,\"causal\":false,\"scale\":0.5}}]}";
    const char* unknown_config =
        "{\"inputs\":{"
        "\"q\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"k\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"v\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"mask\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}},"
        "\"params\":{\"heads\":1,\"causal\":false,\"unknown\":1}}]}";
    const char* missing_causal_config =
        "{\"inputs\":{"
        "\"q\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"k\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"v\":{\"shape\":[2,2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"mask\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}},"
        "\"params\":{\"heads\":1,\"scale\":0.5}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t q_one[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    const int8_t k_one[8] = {0, 0, -1, -1, -1, 0, 0, -2};
    const int8_t v_one[8] = {3, -3, 0, -1, -5, 1, 2, -2};
    const int32_t mask[4] = {1, 1, 0, 0};
    const int8_t expected[16] = {
        0, 0, 2, 0, 0, 0, 2, -1,
        0, 0, 0, 0, 0, 0, 0, 0,
    };
    int8_t q[16];
    int8_t k[16];
    int8_t v[16];
    int8_t output[16] = {0};
    SafetensorsFile file;
    FILE* config_file;
    const char* invalid_paths[] = {
        unknown_config_path,
        missing_causal_config_path,
    };
    const char* invalid_configs[] = {
        unknown_config,
        missing_causal_config,
    };

    memcpy(q, q_one, sizeof(q_one));
    memcpy(q + sizeof(q_one), q_one, sizeof(q_one));
    memcpy(k, k_one, sizeof(k_one));
    memcpy(k + sizeof(k_one), k_one, sizeof(k_one));
    memcpy(v, v_one, sizeof(v_one));
    memcpy(v + sizeof(v_one), v_one, sizeof(v_one));
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* q_tensor = t_find("q");
    T* k_tensor = t_find("k");
    T* v_tensor = t_find("v");
    T* mask_tensor = t_find("mask");
    T* y = t_find("y");
    CHECK(q_tensor && k_tensor && v_tensor && mask_tensor && y &&
          q_tensor->dtype == T_I8 && k_tensor->dtype == T_I8 &&
          v_tensor->dtype == T_I8 && mask_tensor->dtype == T_I32 &&
          y->dtype == T_I8 && q_tensor->quantization.valid &&
          k_tensor->quantization.valid && v_tensor->quantization.valid &&
          y->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw("q", T_I8, q, sizeof(q)) == 0);
    CHECK(volvoxai_engine_set_input_raw("k", T_I8, k, sizeof(k)) == 0);
    CHECK(volvoxai_engine_set_input_raw("v", T_I8, v, sizeof(v)) == 0);
    CHECK(volvoxai_engine_set_input_raw("mask", T_I32, mask, sizeof(mask)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(config_path);

    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        config_file = fopen(invalid_paths[index], "wb");
        CHECK(config_file != NULL && fwrite(invalid_configs[index], 1,
                                             strlen(invalid_configs[index]), config_file) ==
                                         strlen(invalid_configs[index]));
        CHECK(fclose(config_file) == 0);
        CHECK(volvoxai_engine_init(invalid_paths[index], weights_path) == 0);
        y = t_find("y");
        CHECK(y && y->data && y->dtype == T_I8);
        memset(y->data, 0x5a, sizeof(output));
        CHECK(volvoxai_engine_set_input_raw("q", T_I8, q, sizeof(q)) == 0);
        CHECK(volvoxai_engine_set_input_raw("k", T_I8, k, sizeof(k)) == 0);
        CHECK(volvoxai_engine_set_input_raw("v", T_I8, v, sizeof(v)) == 0);
        CHECK(volvoxai_engine_set_input_raw("mask", T_I32, mask, sizeof(mask)) == 0);
        CHECK(volvoxai_engine_forward() != 0);
        for (int byte = 0; byte < (int)sizeof(output); byte++)
            CHECK(((const uint8_t*)y->data)[byte] == 0x5a);
        volvoxai_engine_shutdown();
        remove(invalid_paths[index]);
    }
    remove(weights_path);
    return 0;
}

/* Exercise the native dispatcher rather than the backend entry point alone:
 * the second QGELU must consume `hidden` from Vulkan packed storage, with no
 * host F32 materialization between the two byte-domain activations. */
/* QArgMax's runtime route keeps typed activation bytes intact and normalizes
 * the required negative axis before forming [outer, axis, inner]. */
static int test_physical_qargmax(void) {
    const char* config_path = "/tmp/volvox-physical-qargmax-config.json";
    const char* unknown_config_path = "/tmp/volvox-physical-qargmax-unknown-config.json";
    const char* missing_axis_config_path =
        "/tmp/volvox-physical-qargmax-missing-axis-config.json";
    const char* alias_config_path = "/tmp/volvox-physical-qargmax-alias-config.json";
    const char* weights_path = "/tmp/volvox-physical-qargmax-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-17}}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-2}}]}";
    const char* unknown_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-17}}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":-2,\"unknown\":1}}]}";
    const char* missing_axis_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-17}}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},\"params\":{}}]}";
    const char* alias_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-17}}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"data\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[12] = {
        -5, 4, -1, 4, -1, 3,
        -128, 0, -127, 1, -126, 1,
    };
    const int32_t expected[4] = {1, 0, 2, 1};
    int32_t output[4] = {0};
    const char* invalid_paths[] = {
        unknown_config_path,
        missing_axis_config_path,
        alias_config_path,
    };
    const char* invalid_configs[] = {
        unknown_config,
        missing_axis_config,
        alias_config,
    };
    SafetensorsFile file;
    FILE* config_file;

    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_I8 && y->dtype == T_I32 &&
          x->quantization.valid && x->quantization.zero_point == -17 &&
          x->ndim == 3 && y->ndim == 2 && y->shape[0] == 2 && y->shape[1] == 2);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(config_path);

    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        config_file = fopen(invalid_paths[index], "wb");
        CHECK(config_file != NULL && fwrite(invalid_configs[index], 1,
                                             strlen(invalid_configs[index]), config_file) ==
                                         strlen(invalid_configs[index]));
        CHECK(fclose(config_file) == 0);
        CHECK(volvoxai_engine_init(invalid_paths[index], weights_path) == 0);
        y = t_find("y");
        CHECK(y && y->data && y->dtype == T_I32);
        memset(y->data, 0x5a, sizeof(output));
        CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_forward() != 0);
        for (int byte = 0; byte < (int)sizeof(output); byte++)
            CHECK(((const uint8_t*)y->data)[byte] == 0x5a);
        volvoxai_engine_shutdown();
        remove(invalid_paths[index]);
    }
    remove(weights_path);
    return 0;
}

/* The native graph route must retain a physical byte activation on both sides
 * of the router reduction.  Unknown parameters and legacy input aliases are
 * deliberately rejected before the output buffer is touched. */
static int test_physical_qmaskedmean(void) {
    const char* config_path = "/tmp/volvox-physical-qmaskedmean-config.json";
    const char* unknown_config_path =
        "/tmp/volvox-physical-qmaskedmean-unknown-config.json";
    const char* alias_config_path =
        "/tmp/volvox-physical-qmaskedmean-alias-config.json";
    const char* weights_path = "/tmp/volvox-physical-qmaskedmean-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-3}},"
        "\"mask\":{\"shape\":[2,3],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QMaskedMean\",\"inputs\":{\"input\":\"x\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":5}},\"params\":{}}]}";
    const char* unknown_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-3}},"
        "\"mask\":{\"shape\":[2,3],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QMaskedMean\",\"inputs\":{\"input\":\"x\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":5}},"
        "\"params\":{\"axis\":1}}]}";
    const char* alias_config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-3}},"
        "\"mask\":{\"shape\":[2,3],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QMaskedMean\",\"inputs\":{\"data\":\"x\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":5}},\"params\":{}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[24] = {
        -3, -1,  1,  1,
         1, -5, -3, -1,
        99, 98, 97, 96,
        40, 41, 42, 43,
        44, 45, 46, 47,
        48, 49, 50, 51,
    };
    const int32_t mask[6] = {1, 1, 0, 0, 0, 0};
    const int8_t expected[8] = {6, 5, 6, 6, 5, 5, 5, 5};
    int8_t output[8] = {0};
    const char* invalid_paths[] = {unknown_config_path, alias_config_path};
    const char* invalid_configs[] = {unknown_config, alias_config};
    SafetensorsFile file;
    FILE* config_file;

    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* mask_tensor = t_find("mask");
    T* y = t_find("y");
    CHECK(x && mask_tensor && y && x->dtype == T_I8 && mask_tensor->dtype == T_I32 &&
          y->dtype == T_I8 && x->quantization.valid && !mask_tensor->quantization.valid &&
          y->quantization.valid && x->ndim == 3 && mask_tensor->ndim == 2 && y->ndim == 2);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_set_input_raw("mask", T_I32, mask, sizeof(mask)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(config_path);

    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        config_file = fopen(invalid_paths[index], "wb");
        CHECK(config_file != NULL && fwrite(invalid_configs[index], 1,
                                             strlen(invalid_configs[index]), config_file) ==
                                         strlen(invalid_configs[index]));
        CHECK(fclose(config_file) == 0);
        CHECK(volvoxai_engine_init(invalid_paths[index], weights_path) == 0);
        y = t_find("y");
        CHECK(y && y->data && y->dtype == T_I8);
        memset(y->data, 0x5a, sizeof(output));
        CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
        CHECK(volvoxai_engine_set_input_raw("mask", T_I32, mask, sizeof(mask)) == 0);
        CHECK(volvoxai_engine_forward() != 0);
        for (int byte = 0; byte < (int)sizeof(output); byte++)
            CHECK(((const uint8_t*)y->data)[byte] == 0x5a);
        volvoxai_engine_shutdown();
        remove(invalid_paths[index]);
    }
    remove(weights_path);
    return 0;
}

static int test_physical_qgelu_vulkan_chain(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qgelu-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qgelu-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[5],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},"
        "\"nodes\":["
        "{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":128}},"
        "\"params\":{\"approximate\":\"none\"}},"
        "{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"hidden\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},\"params\":{}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[5] = {-8, -2, 0, 2, 8};
    const int8_t expected_output[5] = {-4, -4, -4, 2, 28};
    int8_t output[5] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QGELU chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* hidden = t_find("hidden");
    T* y = t_find("y");
    CHECK(hidden && y && hidden->dtype == T_U8 && y->dtype == T_I8 &&
          hidden->quantization.valid && y->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int index = 0; index < 5; index++) {
        CHECK(((const uint8_t*)hidden->data)[index] == 0);
        CHECK(((const int8_t*)y->data)[index] == 0);
    }
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* The dispatcher must retain both QGroupNorm activation boundaries in Vulkan
 * packed-byte storage.  Gamma/beta are immutable F32 parameter buffers; they
 * are not a reason to materialize either activation on the host. */
static int test_physical_qgroupnorm_vulkan_chain(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qgroupnorm-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qgroupnorm-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},"
        "\"nodes\":["
        "{\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":128}},"
        "\"params\":{\"num_groups\":1,\"eps\":0.00001,\"data_layout\":\"NHWC\"}},"
        "{\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{\"num_groups\":1}}]}";
    const int one[1] = {1};
    const int three[1] = {3};
    const uint8_t prefix[1] = {0};
    const int8_t input[3] = {-5, 0, 4};
    const float gamma1[3] = {1.0f, 0.5f, -0.75f};
    const float beta1[3] = {0.25f, -0.5f, 0.75f};
    const float gamma2[3] = {0.0f, 0.0f, 0.0f};
    const float beta2[3] = {0.25f, -0.5f, 0.75f};
    const int8_t expected_output[3] = {-2, -8, 2};
    int8_t output[3] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QGroupNorm chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "prefix", SAFETENSORS_DTYPE_U8, one, 1,
                                 prefix, sizeof(prefix)) == 0);
    CHECK(safetensors_add_tensor(&file, "g1", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma1, sizeof(gamma1)) == 0);
    CHECK(safetensors_add_tensor(&file, "b1", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta1, sizeof(beta1)) == 0);
    CHECK(safetensors_add_tensor(&file, "g2", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma2, sizeof(gamma2)) == 0);
    CHECK(safetensors_add_tensor(&file, "b2", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta2, sizeof(beta2)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* hidden = t_find("hidden");
    T* y = t_find("y");
    CHECK(hidden && y && hidden->dtype == T_U8 && y->dtype == T_I8 &&
          hidden->quantization.valid && y->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    /* Before an explicit raw copy, both activation host allocations are stale. */
    for (int index = 0; index < 3; index++) {
        CHECK(((const uint8_t*)hidden->data)[index] == 0);
        CHECK(((const int8_t*)y->data)[index] == 0);
    }
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* The second QLayerNorm consumes the first node's packed U8 rows directly on
 * Vulkan. The host buffers must remain stale until the explicit raw copy. */
static int test_physical_qlayernorm_vulkan_chain(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qlayernorm-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qlayernorm-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[2,3],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-1}}},"
        "\"nodes\":["
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":128}},"
        "\"params\":{\"eps\":0.00001,\"d_model\":3}},"
        "{\"opType\":\"QLayerNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.125,\"zero_point\":-4}},"
        "\"params\":{}}]}";
    const int one[1] = {1};
    const int three[1] = {3};
    const uint8_t prefix[1] = {0};
    const int8_t input[6] = {-5, 0, 4, 6, -2, 1};
    const float gamma1[3] = {1.0f, 0.5f, -0.75f};
    const float beta1[3] = {0.25f, -0.5f, 0.75f};
    const float gamma2[3] = {0.0f, 0.0f, 0.0f};
    const float beta2[3] = {0.25f, -0.5f, 0.75f};
    const int8_t expected_output[6] = {-2, -8, 2, -2, -8, 2};
    int8_t output[6] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QLayerNorm chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "prefix", SAFETENSORS_DTYPE_U8, one, 1,
                                 prefix, sizeof(prefix)) == 0);
    CHECK(safetensors_add_tensor(&file, "g1", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma1, sizeof(gamma1)) == 0);
    CHECK(safetensors_add_tensor(&file, "b1", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta1, sizeof(beta1)) == 0);
    CHECK(safetensors_add_tensor(&file, "g2", SAFETENSORS_DTYPE_F32, three, 1,
                                 gamma2, sizeof(gamma2)) == 0);
    CHECK(safetensors_add_tensor(&file, "b2", SAFETENSORS_DTYPE_F32, three, 1,
                                 beta2, sizeof(beta2)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* hidden = t_find("hidden");
    T* y = t_find("y");
    CHECK(hidden && y && hidden->dtype == T_U8 && y->dtype == T_I8 &&
          hidden->quantization.valid && y->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int index = 0; index < 6; index++) {
        CHECK(((const uint8_t*)hidden->data)[index] == 0);
        CHECK(((const int8_t*)y->data)[index] == 0);
    }
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* The second QSDPA consumes the first node's packed U8 output as Q directly
 * from Vulkan storage.  There is no graph-visible F32 score tensor and the
 * ordinary no-mask path must bind its valid static I32 dummy. */
static int test_physical_qsdpa_vulkan_chain(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qsdpa-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qsdpa-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"q\":{\"shape\":[2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"k\":{\"shape\":[2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}},"
        "\"v\":{\"shape\":[2,4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-1}}},\"nodes\":["
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\"},"
        "\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[2,4]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},"
        "\"params\":{\"heads\":1,\"causal\":false}},"
        "{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"hidden\",\"k\":\"k\",\"v\":\"v\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}},"
        "\"params\":{\"heads\":1,\"causal\":false}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t q[8] = {0, -1, -2, 1, -2, 1, 0, -1};
    const int8_t k[8] = {0, 0, -1, -1, -1, 0, 0, -2};
    const int8_t v[8] = {3, -3, 0, -1, -5, 1, 2, -2};
    uint8_t expected_hidden[8] = {0};
    int8_t expected_output[8] = {0};
    uint8_t hidden[8] = {0};
    int8_t output[8] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QSDPA chain skipped: no Vulkan compute device");
        return 0;
    }
    CHECK(qsdpa_i8u8(q, k, v, NULL, expected_hidden, 1u, 2u, 2u, 4u, 1u,
                      0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 128,
                      0.5f, 2u, 2u, 2u, 3u, 0u, 0u) == 1);
    CHECK(qsdpa_i8u8(expected_hidden, k, v, NULL, expected_output,
                      1u, 2u, 2u, 4u, 1u, 0.25f, 128, 0.25f, -1,
                      0.25f, -1, 0.25f, 0, 0.5f, 3u, 2u, 2u, 2u,
                      0u, 0u) == 1);
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* hidden_tensor = t_find("hidden");
    T* y = t_find("y");
    CHECK(hidden_tensor && y && hidden_tensor->dtype == T_U8 && y->dtype == T_I8 &&
          hidden_tensor->quantization.valid && y->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw("q", T_I8, q, sizeof(q)) == 0);
    CHECK(volvoxai_engine_set_input_raw("k", T_I8, k, sizeof(k)) == 0);
    CHECK(volvoxai_engine_set_input_raw("v", T_I8, v, sizeof(v)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    for (int index = 0; index < 8; index++) {
        CHECK(((const uint8_t*)hidden_tensor->data)[index] == 0);
        CHECK(((const int8_t*)y->data)[index] == 0);
    }
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* This exercises the engine route rather than the backend APIs in isolation.
 * Before the explicit public copy, the three raw outputs must still be host
 * stale, proving that QConv2D -> QAdd -> RequantizeLinear remained a packed
 * Vulkan device chain instead of falling through to the CPU implementation. */
/* The native Vulkan route reads raw U8 storage and leaves the I32 result on
 * device until the public raw copy. A nonzero asymmetric zero point cannot
 * affect the selected raw-byte maximum. */
static int test_physical_qargmax_vulkan(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qargmax-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qargmax-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[1,3,2],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":129}}},"
        "\"nodes\":[{\"opType\":\"QArgMax\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int32\"},"
        "\"params\":{\"axis\":1}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const uint8_t input[6] = {130, 3, 129, 255, 130, 254};
    const int32_t expected[2] = {0, 1};
    int32_t output[2] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QArgMax skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* y = t_find("y");
    CHECK(y && y->dtype == T_I32 && y->ndim == 2 && y->shape[0] == 1 &&
          y->shape[1] == 2);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(((const int32_t*)y->data)[0] == 0 && ((const int32_t*)y->data)[1] == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

/* Exercise the strict runtime GPU route, not merely the backend API.  The
 * output remains stale on the host until the public raw copy, demonstrating
 * that no F32/host activation boundary was inserted around QMaskedMean. */
static int test_physical_qmaskedmean_vulkan(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qmaskedmean-config.json";
    const char* weights_path =
        "/tmp/volvox-physical-vulkan-qmaskedmean-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[1,3,2],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":128}},"
        "\"mask\":{\"shape\":[1,3],\"dtype\":\"int32\"}},\"nodes\":[{"
        "\"opType\":\"QMaskedMean\",\"inputs\":{\"input\":\"x\",\"mask\":\"mask\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":130}},\"params\":{}}]}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const uint8_t input[6] = {130, 134, 20, 20, 131, 129};
    const int32_t mask[3] = {1, 0, 1};
    const uint8_t expected[2] = {132, 134};
    uint8_t output[2] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QMaskedMean skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* y = t_find("y");
    CHECK(y && y->dtype == T_U8 && y->ndim == 2 && y->shape[0] == 1 &&
          y->shape[1] == 2 && y->quantization.valid);
    memset(y->data, 0, sizeof(output));
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_set_input_raw("mask", T_I32, mask, sizeof(mask)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(((const uint8_t*)y->data)[0] == 0 && ((const uint8_t*)y->data)[1] == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qconv_qadd_requantize_vulkan_chain(void) {
    const char* config_path = "/tmp/volvox-physical-vulkan-qconv-qadd-config.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qconv-qadd-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":128}},"
        "\"add\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":128}}},"
        "\"weights_quantization\":{\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scales\":[1.0],\"zero_points\":[128]}},\"nodes\":["
        "{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":\"conv\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":128}},\"params\":{"
        "\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\",\"groups\":1,"
        "\"stride\":[1,1],\"dilation\":[1,1],\"pads\":[0,0,0,0]}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"conv\",\"b\":\"add\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":128}},\"params\":{}},"
        "{\"opType\":\"RequantizeLinear\",\"inputs\":{\"input\":\"sum\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}},\"params\":{}}]}";
    const int weight_shape[4] = {1, 1, 1, 1};
    const uint8_t weight[1] = {130};
    const uint8_t x[1] = {131};
    const uint8_t add[1] = {129};
    int8_t output[1] = {0};
    SafetensorsFile file;
    FILE* config_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan runtime chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_U8, weight_shape, 4,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* conv = t_find("conv");
    T* sum = t_find("sum");
    T* y = t_find("y");
    CHECK(conv && sum && y && conv->dtype == T_U8 && sum->dtype == T_U8 && y->dtype == T_I8);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, x, sizeof(x)) == 0);
    CHECK(volvoxai_engine_set_input_raw("add", T_U8, add, sizeof(add)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(((const uint8_t*)conv->data)[0] == 0);
    CHECK(((const uint8_t*)sum->data)[0] == 0);
    CHECK(((const int8_t*)y->data)[0] == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(output[0] == 14);
    volvoxai_engine_shutdown();
    g_use_vulkan = 0;
    vk_cleanup();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_dequantize_sigmoid_quantize(void) {
    const char* config_path = "/tmp/volvox-physical-boundary-config.json";
    const char* weights_path = "/tmp/volvox-physical-boundary-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[4],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":-2}}},\"nodes\":["
        "{\"opType\":\"DequantizeLinear\",\"inputs\":{\"input\":\"x\",\"scale\":\"sdq\","
        "\"zero_point\":\"zdq\"},\"outputs\":{\"out\":\"f\"},"
        "\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},"
        "{\"opType\":\"Sigmoid\",\"inputs\":{\"input\":\"f\"},\"outputs\":{\"out\":\"s\"},"
        "\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},"
        "{\"opType\":\"QuantizeLinear\",\"inputs\":{\"input\":\"s\",\"scale\":\"sq\","
        "\"zero_point\":\"zq\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"uint8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.25,"
        "\"zero_point\":128}},\"params\":{}}]}";
    const int one[1] = {1};
    const int8_t zdq[1] = {-2};
    const float sdq[1] = {0.5f};
    const uint8_t zq[1] = {128};
    const float sq[1] = {0.25f};
    const int8_t input[4] = {-2, 0, 2, 126};
    const float expected_f[4] = {0.0f, 1.0f, 2.0f, 64.0f};
    const uint8_t expected_y[4] = {130, 131, 132, 132};
    float f[4] = {0};
    uint8_t y[4] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    /* F32 scalars deliberately follow byte tensors. SafeTensors guarantees no
       natural alignment, so the runtime must pass aligned local scalar copies
       into the portable ABI. */
    CHECK(safetensors_add_tensor(&file, "zdq", SAFETENSORS_DTYPE_I8, one, 1,
                                 zdq, sizeof(zdq)) == 0);
    CHECK(safetensors_add_tensor(&file, "sdq", SAFETENSORS_DTYPE_F32, one, 1,
                                 sdq, sizeof(sdq)) == 0);
    CHECK(safetensors_add_tensor(&file, "zq", SAFETENSORS_DTYPE_U8, one, 1,
                                 zq, sizeof(zq)) == 0);
    CHECK(safetensors_add_tensor(&file, "sq", SAFETENSORS_DTYPE_F32, one, 1,
                                 sq, sizeof(sq)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* tf = t_find("f");
    T* ty = t_find("y");
    T* scale_dq = t_find("sdq");
    T* scale_q = t_find("sq");
    CHECK(x && tf && ty && scale_dq && scale_q && x->dtype == T_I8 &&
          tf->dtype == T_F32 && ty->dtype == T_U8 && x->quantization.valid &&
          ty->quantization.valid && g_qt[x - g_t].data == NULL && g_qt[ty - g_t].data == NULL);
    CHECK(((uintptr_t)scale_dq->data % sizeof(float)) != 0 &&
          ((uintptr_t)scale_q->data % sizeof(float)) != 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("f", f, 4) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", y, sizeof(y)) == 0);
    for (int i = 0; i < 4; i++) CHECK(closef(f[i], expected_f[i]));
    CHECK(memcmp(y, expected_y, sizeof(y)) == 0);
    CHECK(g_qt[x - g_t].data == NULL && g_qt[ty - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_quantize_nan_and_saturation(void) {
    const char* config_path = "/tmp/volvox-physical-quantize-special-config.json";
    const char* weights_path = "/tmp/volvox-physical-quantize-special-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"f\":{\"shape\":[5],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"QuantizeLinear\",\"inputs\":{\"input\":\"f\",\"scale\":\"scale\","
        "\"zero_point\":\"zero_point\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"out\":\"uint8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":128}},\"params\":{}}]}";
    const int one[1] = {1};
    const uint8_t zero_point[1] = {128};
    const float scale[1] = {1.0f};
    const float input[5] = {NAN, -INFINITY, INFINITY, 0.5f, 1.5f};
    const uint8_t expected[5] = {128, 0, 255, 128, 130};
    uint8_t output[5] = {0};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "zero_point", SAFETENSORS_DTYPE_U8, one, 1,
                                 zero_point, sizeof(zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "scale", SAFETENSORS_DTYPE_F32, one, 1,
                                 scale, sizeof(scale)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* f = t_find("f");
    T* y = t_find("y");
    T* scale_tensor = t_find("scale");
    CHECK(f && y && scale_tensor && f->dtype == T_F32 && y->dtype == T_U8 &&
          y->quantization.valid && y->quantization.zero_point == 128 &&
          ((uintptr_t)scale_tensor->data % sizeof(float)) != 0 &&
          g_qt[y - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("f", T_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* NaN maps to output zero point; infinities saturate; .5 ties round even. */
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(g_qt[y - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int write_quantized_test_config(const char* path, const char* config) {
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL && fwrite(config, 1, strlen(config), file) == strlen(config));
    CHECK(fclose(file) == 0);
    return 0;
}

static int test_set_input_f32_quantizes_physical_inputs(void) {
    const char* config_path = "/tmp/volvox-set-input-f32-config.json";
    const char* config =
        "{\"inputs\":{"
        "\"f\":{\"shape\":[6],\"dtype\":\"float32\"},"
        "\"u\":{\"shape\":[6],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":100}},"
        "\"i\":{\"shape\":[6],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-4}},"
        "\"ids\":{\"shape\":[6],\"dtype\":\"int32\"}},"
        "\"nodes\":[],\"outputs\":[\"f\"]}";
    const float f_values[6] = {-2.0f, -0.5f, 0.0f, 0.25f, 1.0f, 3.0f};
    const float u_values[6] = {-100.0f, 0.0f, 0.25f, 0.75f, 100.0f, NAN};
    const float i_values[6] = {-100.0f, 0.0f, 0.125f, 0.375f, 100.0f, NAN};
    const uint8_t expected_u[6] = {0, 100, 100, 102, 255, 100};
    const int8_t expected_i[6] = {-128, -4, -4, -2, 127, -4};
    float copied_f[6] = {0};
    uint8_t copied_u[6] = {0};
    int8_t copied_i[6] = {0};

    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(volvoxai_engine_init(config_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_f32("f", f_values, 6) == 0);
    CHECK(volvoxai_engine_set_input_f32("u", u_values, 6) == 0);
    CHECK(volvoxai_engine_set_input_f32("i", i_values, 6) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("f", copied_f, sizeof(copied_f)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("u", copied_u, sizeof(copied_u)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("i", copied_i, sizeof(copied_i)) == 0);
    CHECK(memcmp(copied_f, f_values, sizeof(copied_f)) == 0);
    CHECK(memcmp(copied_u, expected_u, sizeof(copied_u)) == 0);
    CHECK(memcmp(copied_i, expected_i, sizeof(copied_i)) == 0);
    CHECK(volvoxai_engine_set_input_f32("ids", f_values, 6) != 0);
    CHECK(volvoxai_engine_set_input_f32("missing", f_values, 6) != 0);
    CHECK(volvoxai_engine_set_input_f32("u", f_values, 5) != 0);
    CHECK(volvoxai_engine_set_input_f32("u", NULL, 6) != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    return 0;
}

static int test_concat_sigmoid_fusion_backend_scope(void) {
    const char* config_path = "/tmp/volvox-concat-sigmoid-fusion-scope.json";
    const char* config =
        "{\"inputs\":{\"a\":{\"shape\":[1,1]},\"b\":{\"shape\":[1,1]}},"
        "\"nodes\":[{\"op\":\"Concat\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"joined\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"axis\":1}},{\"op\":\"Sigmoid\","
        "\"inputs\":{\"input\":\"joined\"},\"outputs\":{\"out\":\"result\"},"
        "\"outputs_shape\":{\"out\":[1,2]},\"params\":{}}],"
        "\"outputs\":[\"result\"]}";
    const float zero = 0.0f;
    float output[2] = {0.0f, 0.0f};

    CHECK(write_quantized_test_config(config_path, config) == 0);

    g_use_vulkan = 0;
    g_use_opengl = 0;
    g_use_metal = 0;
    g_use_nnapi = 0;
    CHECK(volvoxai_engine_init(config_path, NULL) == 0);
    CHECK(g_nn == 2 && g_concat_sigmoid_fuse[0] == 1 && g_n[1].skip);
    CHECK(volvoxai_engine_set_input_raw("a", T_F32, &zero, sizeof(zero)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", T_F32, &zero, sizeof(zero)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("result", output, 2) == 0);
    CHECK(fabsf(output[0] - 0.5f) < 0.02f &&
          fabsf(output[1] - 0.5f) < 0.02f);
    volvoxai_engine_shutdown();

    /* Graph optimization only needs the selected-backend flag.  This verifies
     * Metal classification and fusion eligibility without initializing or
     * executing a physical Metal device. */
    g_use_metal = 1;
    CHECK(volvoxai_engine_init(config_path, NULL) == 0);
    CHECK(g_nn == 2 && g_concat_sigmoid_fuse[0] == -1 && !g_n[1].skip);
    g_use_metal = 0;
    volvoxai_engine_shutdown();

    remove(config_path);
    return 0;
}

static int save_quantized_test_weights(const char* path) {
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(path, &file) == 0);
    safetensors_free(&file);
    return 0;
}

static int run_physical_shape_chain(void) {
    const uint8_t a[4] = {121, 125, 123, 124};
    const uint8_t b[4] = {120, 120, 120, 120};
    const uint8_t tail[2] = {122, 124};
    const uint8_t input2[1] = {123};
    const uint8_t input10[1] = {124};
    const uint8_t expected[10] = {125, 125, 125, 125, 125, 125, 123, 124, 122, 124};
    const char* const byte_tensors[] = {
        "a", "b", "sum", "pool", "resize_nearest", "resize", "reshape",
        "flatten", "squeeze", "unsqueeze", "identity", "tail", "input2", "input10", "out"
    };
    uint8_t output[10] = {0};
    CHECK(volvoxai_engine_set_input_raw("a", T_U8, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw("b", T_U8, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_set_input_raw("tail", T_U8, tail, sizeof(tail)) == 0);
    CHECK(volvoxai_engine_set_input_raw("input2", T_U8, input2, sizeof(input2)) == 0);
    CHECK(volvoxai_engine_set_input_raw("input10", T_U8, input10, sizeof(input10)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("out", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    for (size_t index = 0; index < sizeof(byte_tensors) / sizeof(byte_tensors[0]); index++) {
        T* tensor = t_find(byte_tensors[index]);
        CHECK(tensor && tensor->dtype == T_U8 && tensor->quantization.valid &&
              closef(tensor->quantization.scale, 0.5f) && tensor->quantization.zero_point == 120 &&
              g_qt[tensor - g_t].data == NULL && !g_qt[tensor - g_t].has_params);
    }
    return 0;
}

static int test_physical_shape_island_chain(void) {
    const char* config_path = "/tmp/volvox-physical-shape-chain-config.json";
    const char* saved_config_path = "/tmp/volvox-physical-shape-chain-saved.json";
    const char* weights_path = "/tmp/volvox-physical-shape-chain-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"a\":{\"shape\":[1,2,2,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},"
        "\"b\":{\"shape\":[1,2,2,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},"
        "\"tail\":{\"shape\":[1,2],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},"
        "\"input2\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},"
        "\"input10\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},"
        "\"nodes\":["
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"sum\"},\"outputs\":{\"out\":\"pool\"},\"outputs_shape\":{\"out\":[1,1,1,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"kernel\":[2,2],\"stride\":[2,2],\"padding\":[0,0],\"dilation\":[1,1],\"ceil_mode\":false}},"
        "{\"opType\":\"ResizeNearest2D\",\"inputs\":{\"input\":\"pool\"},\"outputs\":{\"out\":\"resize_nearest\"},\"outputs_shape\":{\"out\":[1,2,3,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"coordinate_transformation_mode\":\"asymmetric\",\"nearest_mode\":\"floor\"}},"
        "{\"opType\":\"Resize\",\"inputs\":{\"input\":\"resize_nearest\"},\"outputs\":{\"out\":\"resize\"},\"outputs_shape\":{\"out\":[1,3,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"mode\":\"nearest\",\"coordinate_transformation_mode\":\"asymmetric\",\"nearest_mode\":\"floor\"}},"
        "{\"opType\":\"Reshape\",\"inputs\":{\"input\":\"resize\"},\"outputs\":{\"out\":\"reshape\"},\"outputs_shape\":{\"out\":[1,1,3,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"Flatten\",\"inputs\":{\"input\":\"reshape\"},\"outputs\":{\"out\":\"flatten\"},\"outputs_shape\":{\"out\":[1,6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"Squeeze\",\"inputs\":{\"input\":\"flatten\"},\"outputs\":{\"out\":\"squeeze\"},\"outputs_shape\":{\"out\":[6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"Unsqueeze\",\"inputs\":{\"input\":\"squeeze\"},\"outputs\":{\"out\":\"unsqueeze\"},\"outputs_shape\":{\"out\":[1,6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"Identity\",\"inputs\":{\"input\":\"unsqueeze\"},\"outputs\":{\"out\":\"identity\"},\"outputs_shape\":{\"out\":[1,6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{}},"
        "{\"opType\":\"Concat\",\"inputs\":{\"input10\":\"input10\",\"z\":\"tail\",\"input2\":\"input2\",\"input\":\"identity\"},\"outputs\":{\"out\":\"out\"},\"outputs_shape\":{\"out\":[1,10]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"axis\":-1}}]}";
    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(save_quantized_test_weights(weights_path) == 0);

    /* Disable aliases first so the portable byte copy kernel executes for every
       reshape-like edge. */
    CHECK(setenv("VOLVOX_DISABLE_OPERATOR_FUSION", "1", 1) == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    CHECK(g_nn == 10 && !g_n[4].skip && !g_n[5].skip && !g_n[6].skip &&
          !g_n[7].skip && !g_n[8].skip);
    CHECK(run_physical_shape_chain() == 0);
    CHECK(volvoxai_engine_save_config(saved_config_path) == 0);
    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_DISABLE_OPERATOR_FUSION") == 0);

    /* The normal optimizer may alias identical physical domains, and must
       still preserve the raw bytes through the same final concat. */
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    CHECK(g_nn == 10 && g_n[4].skip && g_n[5].skip && g_n[6].skip &&
          g_n[7].skip && g_n[8].skip);
    CHECK(run_physical_shape_chain() == 0);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_init(saved_config_path, weights_path) == 0);
    CHECK(run_physical_shape_chain() == 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(saved_config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_shape_maxpool_asymmetric_pads(void) {
    const char* config_path = "/tmp/volvox-physical-shape-pool-config.json";
    const char* weights_path = "/tmp/volvox-physical-shape-pool-weights.safetensors";
    const char* config =
        "{\"inputs\":{"
        "\"u\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":4}},"
        "\"i\":{\"shape\":[1,1,1,1],\"dtype\":\"int8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-5}}},"
        "\"nodes\":["
        "{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"u\"},\"outputs\":{\"out\":\"uout\"},\"outputs_shape\":{\"out\":[1,3,3,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":4}},\"params\":{\"kernel\":[2,2],\"stride\":[1,1],\"pads\":[2,1,1,2],\"dilation\":[1,1],\"ceil_mode\":false}},"
        "{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"i\"},\"outputs\":{\"out\":\"iout\"},\"outputs_shape\":{\"out\":[1,3,3,1]},\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":-5}},\"params\":{\"kernel\":[2,2],\"stride\":[1,1],\"pads\":[2,1,1,2],\"dilation\":[1,1],\"ceil_mode\":false}}]}";
    const uint8_t u[1] = {4};
    const int8_t i[1] = {-5};
    const uint8_t expected_u[9] = {0, 0, 0, 4, 4, 0, 4, 4, 0};
    const int8_t expected_i[9] = {-128, -128, -128, -5, -5, -128, -5, -5, -128};
    uint8_t actual_u[9] = {0};
    int8_t actual_i[9] = {0};
    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(save_quantized_test_weights(weights_path) == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* tu = t_find("u");
    T* ti = t_find("i");
    T* tuout = t_find("uout");
    T* tiout = t_find("iout");
    CHECK(tu && ti && tuout && tiout && tu->dtype == T_U8 && ti->dtype == T_I8 &&
          tuout->dtype == T_U8 && tiout->dtype == T_I8 &&
          g_qt[tu - g_t].data == NULL && g_qt[ti - g_t].data == NULL &&
          g_qt[tuout - g_t].data == NULL && g_qt[tiout - g_t].data == NULL);
    CHECK(volvoxai_engine_set_input_raw("u", T_U8, u, sizeof(u)) == 0);
    CHECK(volvoxai_engine_set_input_raw("i", T_I8, i, sizeof(i)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("uout", actual_u, sizeof(actual_u)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("iout", actual_i, sizeof(actual_i)) == 0);
    /* [top,left,bottom,right] is deliberately asymmetric. Padded-only windows
       must use the raw dtype minimum: 0 for U8 and -128 for I8. */
    CHECK(memcmp(actual_u, expected_u, sizeof(actual_u)) == 0);
    CHECK(memcmp(actual_i, expected_i, sizeof(actual_i)) == 0);
    CHECK(g_qt[tuout - g_t].data == NULL && g_qt[tiout - g_t].data == NULL);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int expect_physical_shape_rejected(const char* tag, const char* config,
                                          int require_unaliased_first_node) {
    char config_path[160];
    char weights_path[160];
    CHECK(snprintf(config_path, sizeof(config_path), "/tmp/volvox-physical-shape-%s-config.json", tag) > 0);
    CHECK(snprintf(weights_path, sizeof(weights_path), "/tmp/volvox-physical-shape-%s-weights.safetensors", tag) > 0);
    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(save_quantized_test_weights(weights_path) == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    if (require_unaliased_first_node) CHECK(g_nn == 1 && !g_n[0].skip);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_physical_shape_rejections(void) {
    const char* copy_descriptor_mismatch =
        "{\"inputs\":{\"x\":{\"shape\":[2],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"Reshape\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":120}},\"params\":{}}]}";
    const char* concat_descriptor_mismatch =
        "{\"inputs\":{\"a\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"b\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"b\":\"b\",\"a\":\"a\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"axis\":1}}]}";
    const char* concat_dtype_mismatch =
        "{\"inputs\":{\"a\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"b\":{\"shape\":[1,1],\"dtype\":\"int8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":0}}},\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"axis\":1}}]}";
    const char* bilinear_resize =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"Resize\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"mode\":\"linear\"}}]}";
    const char* unsupported_resize_transform =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"Resize\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"mode\":\"nearest\",\"coordinate_transformation_mode\":\"half_pixel\"}}]}";
    const char* pool_nonunit_dilation =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"kernel\":[1,1],\"dilation\":[2,1]}}]}";
    const char* pool_ceil_mode =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"kernel\":[1,1],\"ceil_mode\":true}}]}";
    const char* concat_bad_axis =
        "{\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"axis\":2}}]}";
    const char* concat_sigmoid =
        "{\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"uint8\",\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}}},\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":120}},\"params\":{\"axis\":1,\"sigmoid\":1}}]}";
    CHECK(expect_physical_shape_rejected("copy-descriptor", copy_descriptor_mismatch, 1) == 0);
    CHECK(expect_physical_shape_rejected("concat-descriptor", concat_descriptor_mismatch, 0) == 0);
    CHECK(expect_physical_shape_rejected("concat-dtype", concat_dtype_mismatch, 0) == 0);
    CHECK(expect_physical_shape_rejected("resize-bilinear", bilinear_resize, 0) == 0);
    CHECK(expect_physical_shape_rejected("resize-transform", unsupported_resize_transform, 0) == 0);
    CHECK(expect_physical_shape_rejected("pool-dilation", pool_nonunit_dilation, 0) == 0);
    CHECK(expect_physical_shape_rejected("pool-ceil", pool_ceil_mode, 0) == 0);
    CHECK(expect_physical_shape_rejected("concat-axis", concat_bad_axis, 0) == 0);
    CHECK(expect_physical_shape_rejected("concat-sigmoid", concat_sigmoid, 0) == 0);
    return 0;
}

static int test_generic_add_rejects_implicit_quantization(void) {
    const char* config_path = "/tmp/volvox-implicit-quantized-add-config.json";
    const char* weights_path = "/tmp/volvox-implicit-quantized-add-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"a\":{\"shape\":[1,1],\"dtype\":\"float32\"},"
        "\"b\":{\"shape\":[1,1],\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,1]},\"params\":{}},"
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"sum\",\"weight\":\"w\","
        "\"weight_scale\":\"ws\",\"weight_zero_point\":\"wz\",\"bias\":\"bias\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1]},"
        "\"params\":{\"weight_layout\":\"OUT_IN\",\"input_scale\":0.25,"
        "\"input_zero_point\":0,\"output_scale\":0.5,\"output_zero_point\":0}}]}";
    const int matrix_shape[2] = {1, 1};
    const int vector_shape[1] = {1};
    const int8_t weight[1] = {1};
    const float weight_scale[1] = {0.5f};
    const int32_t weight_zero_point[1] = {0};
    const int32_t bias[1] = {0};
    SafetensorsFile file;

    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8,
                                 matrix_shape, 2, weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "ws", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, weight_scale,
                                 sizeof(weight_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "wz", SAFETENSORS_DTYPE_I32,
                                 vector_shape, 1, weight_zero_point,
                                 sizeof(weight_zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "bias", SAFETENSORS_DTYPE_I32,
                                 vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_generic_add_rejects_physical_bytes(void) {
    const char* config_path = "/tmp/volvox-physical-generic-add-config.json";
    const char* config =
        "{\"inputs\":{\"a\":{\"shape\":[1],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":128}},"
        "\"b\":{\"shape\":[1],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":120}}},\"nodes\":[{"
        "\"opType\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"outputs_quantization\":{\"out\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":128}},\"params\":{}}]}";

    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(volvoxai_engine_init(config_path, NULL) != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    return 0;
}

static int test_w8a8_linear_chain(void) {
    const char* config_path = "/tmp/volvox-qlinear-config.json";
    const char* weights_path = "/tmp/volvox-qlinear-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[1,3],\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Linear\",\"inputs\":{\"x\":\"x\",\"weight\":\"w0\","
        "\"weight_scale\":\"s0\",\"weight_zero_point\":\"z0\",\"bias\":\"b0\"},"
        "\"outputs\":{\"out\":\"h\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"weight_layout\":\"OUT_IN\",\"input_scale\":0.25,"
        "\"input_zero_point\":-1,\"output_scale\":0.25,\"output_zero_point\":2}},"
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"h\",\"weight\":\"w1\","
        "\"weight_scale\":\"s1\",\"weight_zero_point\":\"z1\",\"bias\":\"b1\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"weight_layout\":\"IN_OUT\",\"input_scale\":0.25,"
        "\"input_zero_point\":2,\"output_scale\":0.125,\"output_zero_point\":-3}},"
        "{\"opType\":\"QuantizeLinear\",\"inputs\":{\"input\":\"y\"},"
        "\"outputs\":{\"out\":\"r\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"input_scale\":0.125,\"input_zero_point\":-3,"
        "\"output_scale\":0.125,\"output_zero_point\":-3}}]}";
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);

    const int matrix03[2] = {2, 3};
    const int matrix22[2] = {2, 2};
    const int vector2[1] = {2};
    const int vector1[1] = {1};
    const int8_t w0[6] = {2, -3, 4, -5, 1, 3};
    const float s0[2] = {0.5f, 0.25f};
    const int32_t z0[2] = {1, -2};
    const int32_t b0[2] = {1, -4};
    /* IN_OUT: [d_in,d_out]. The second channel deliberately saturates. */
    const uint8_t w1[4] = {4, 7, 2, 9};
    const float s1[1] = {0.25f};
    const uint8_t z1[1] = {3};
    const int32_t b1[2] = {1, 500};

    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w0", SAFETENSORS_DTYPE_I8, matrix03, 2, w0, sizeof(w0)) == 0);
    CHECK(safetensors_add_tensor(&file, "s0", SAFETENSORS_DTYPE_F32, vector2, 1, s0, sizeof(s0)) == 0);
    CHECK(safetensors_add_tensor(&file, "z0", SAFETENSORS_DTYPE_I32, vector2, 1, z0, sizeof(z0)) == 0);
    CHECK(safetensors_add_tensor(&file, "b0", SAFETENSORS_DTYPE_I32, vector2, 1, b0, sizeof(b0)) == 0);
    CHECK(safetensors_add_tensor(&file, "w1", SAFETENSORS_DTYPE_U8, matrix22, 2, w1, sizeof(w1)) == 0);
    CHECK(safetensors_add_tensor(&file, "s1", SAFETENSORS_DTYPE_F32, vector1, 1, s1, sizeof(s1)) == 0);
    CHECK(safetensors_add_tensor(&file, "z1", SAFETENSORS_DTYPE_U8, vector1, 1, z1, sizeof(z1)) == 0);
    CHECK(safetensors_add_tensor(&file, "b1", SAFETENSORS_DTYPE_I32, vector2, 1, b1, sizeof(b1)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    long input_numel = 0;
    float* input = volvoxai_engine_input_ptr("x", &input_numel);
    CHECK(input && input_numel == 3);
    input[0] = 0.2f;
    input[1] = -0.6f;
    input[2] = 0.9f;
    CHECK(volvoxai_engine_forward() == 0);

    T* hidden = t_find("h");
    T* output = t_find("y");
    T* requantized = t_find("r");
    CHECK(hidden && output && requantized);
    QTensor* hidden_q = &g_qt[hidden - g_t];
    QTensor* output_q = &g_qt[output - g_t];
    QTensor* requantized_q = &g_qt[requantized - g_t];
    CHECK(hidden_q->valid && hidden_q->has_params && closef(hidden_q->scale, 0.25f) && hidden_q->zp == 2);
    CHECK(output_q->valid && output_q->has_params && closef(output_q->scale, 0.125f) && output_q->zp == -3);
    CHECK(hidden_q->data[0] == 13 && hidden_q->data[1] == 4);
    CHECK(output_q->data[0] == 2 && output_q->data[1] == 127);
    CHECK(requantized_q->valid && requantized_q->data[0] == 2 && requantized_q->data[1] == 127);

    float raw_hidden_values[2] = {0};
    float raw_output_values[2] = {0};
    CHECK(volvoxai_engine_copy_tensor_raw("h", raw_hidden_values,
                                          sizeof(raw_hidden_values)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", raw_output_values,
                                          sizeof(raw_output_values)) == 0);
    CHECK(closef(raw_hidden_values[0], (13 - 2) * 0.25f));
    CHECK(closef(raw_hidden_values[1], (4 - 2) * 0.25f));
    CHECK(closef(raw_output_values[0], (2 + 3) * 0.125f));
    CHECK(closef(raw_output_values[1], (127 + 3) * 0.125f));

    float hidden_values[2] = {0};
    float output_values[2] = {0};
    CHECK(volvoxai_engine_copy_tensor_f32("h", hidden_values, 2) == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output_values, 2) == 0);
    CHECK(closef(hidden_values[0], (13 - 2) * 0.25f));
    CHECK(closef(hidden_values[1], (4 - 2) * 0.25f));
    CHECK(closef(output_values[0], (2 + 3) * 0.125f));
    CHECK(closef(output_values[1], (127 + 3) * 0.125f));

    /* Repeated execution must quantize from the current F32 input, not stale
       backing storage materialized by the public read above. */
    input[0] = -0.2f;
    input[1] = 0.4f;
    input[2] = 0.1f;
    int8_t xq[3];
    for (int i = 0; i < 3; i++) xq[i] = quantize_i8(input[i], 0.25f, -1);
    CHECK(xq[0] == -2 && xq[1] == 1 && xq[2] == -1);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(hidden_q->valid && output_q->valid && requantized_q->valid);
    CHECK(hidden_q->data[0] == -2 && hidden_q->data[1] == 3);
    CHECK(output_q->data[0] == -5 && output_q->data[1] == 127);
    CHECK(requantized_q->data[0] == -5 && requantized_q->data[1] == 127);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output_values, 2) == 0);
    CHECK(closef(output_values[0], -0.25f));
    CHECK(closef(output_values[1], 16.25f));

    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_implicit_qconv2d_activation_rejected(void) {
    const char* config_path = "/tmp/volvox-qlinear-qconv-config.json";
    const char* weights_path = "/tmp/volvox-qlinear-qconv-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"float32\"}},\"nodes\":["
        "{\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"lw\","
        "\"weight_scale\":\"ls\",\"weight_zero_point\":\"lz\",\"bias\":\"lb\"},"
        "\"outputs\":{\"out\":\"h\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"weight_layout\":\"OUT_IN\",\"input_scale\":0.5,\"input_zero_point\":0,"
        "\"output_scale\":0.25,\"output_zero_point\":1}},"
        "{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"h\",\"weight\":\"cw\","
        "\"weight_scale\":\"cs\",\"weight_zero_point\":\"cz\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"params\":{\"weight_layout\":\"OIHW\",\"stride\":[1,1],\"padding\":[0,0],"
        "\"dilation\":[1,1],\"groups\":1,\"input_scale\":0.5,\"input_zero_point\":-2}}]}";
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);

    const int one[1] = {1};
    const int one_by_one[2] = {1, 1};
    const int conv_shape[4] = {1, 1, 1, 1};
    const int padding_shape[1] = {3};
    const int8_t linear_weight[1] = {3};
    const float linear_scale[1] = {0.5f};
    const int32_t linear_zero_point[1] = {0};
    const int32_t linear_bias[1] = {0};
    const int8_t padding[3] = {0, 0, 0};
    const int8_t conv_weight[1] = {2};
    const float conv_scale[1] = {0.5f};
    const int32_t conv_zero_point[1] = {0};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    /* Keep F32/I32 payloads aligned; the separate W8A8 test covers deliberately
       unaligned metadata through the new safe-load paths. */
    CHECK(safetensors_add_tensor(&file, "ls", SAFETENSORS_DTYPE_F32, one, 1,
                                 linear_scale, sizeof(linear_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "lz", SAFETENSORS_DTYPE_I32, one, 1,
                                 linear_zero_point, sizeof(linear_zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "lb", SAFETENSORS_DTYPE_I32, one, 1,
                                 linear_bias, sizeof(linear_bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "lw", SAFETENSORS_DTYPE_I8, one_by_one, 2,
                                 linear_weight, sizeof(linear_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "padding", SAFETENSORS_DTYPE_I8, padding_shape, 1,
                                 padding, sizeof(padding)) == 0);
    CHECK(safetensors_add_tensor(&file, "cs", SAFETENSORS_DTYPE_F32, one, 1,
                                 conv_scale, sizeof(conv_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "cz", SAFETENSORS_DTYPE_I32, one, 1,
                                 conv_zero_point, sizeof(conv_zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "cw", SAFETENSORS_DTYPE_I8, conv_shape, 4,
                                 conv_weight, sizeof(conv_weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_qconv2d_weight_only_f32_activation(void) {
    const char* config_path = "/tmp/volvox-qconv-weight-only-config.json";
    const char* weights_path = "/tmp/volvox-qconv-weight-only-weights.safetensors";
    const char* config =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"w\",\"weight_scale\":\"ws\","
        "\"weight_zero_point\":\"wz\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"dilation\":[1,1],\"groups\":1,"
        "\"input_scale\":0.5,\"input_zero_point\":0,\"weight_only\":true}}],"
        "\"outputs\":[\"y\"]}";
    const int vector_shape[1] = {2};
    const int weight_shape[4] = {2, 1, 1, 2};
    const float weight_scale[2] = {0.25f, 0.5f};
    const int32_t weight_zero_point[2] = {1, -1};
    const float bias[2] = {0.1f, -0.2f};
    const int8_t weight[4] = {2, -1, -3, 4};
    const float input[2] = {0.26f, -0.4f};
    const float expected_weight_only[2] = {0.365f, -1.46f};
    float output[2] = {0.0f, 0.0f};
    SafetensorsFile file;
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL &&
          fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "ws", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, weight_scale,
                                 sizeof(weight_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "wz", SAFETENSORS_DTYPE_I32,
                                 vector_shape, 1, weight_zero_point,
                                 sizeof(weight_zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8,
                                 weight_shape, 4, weight, sizeof(weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_F32 && y->dtype == T_F32);
    /* The W8A32 contract must not predeclare or populate a QTensor activation
     * sidecar merely because input_scale remains in a compatible blueprint. */
    CHECK(!g_qt[x - g_t].has_params && !g_qt[x - g_t].valid);
    CHECK(volvoxai_engine_set_input_raw("x", T_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 2) == 0);
    CHECK(closef(output[0], expected_weight_only[0]));
    CHECK(closef(output[1], expected_weight_only[1]));

    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int expect_qconv2d_weight_only_init_rejection(
    const char* config_path, const char* weights_path, const char* config
) {
    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    return 0;
}

static int test_qconv2d_weight_only_rejects_malformed_descriptors(void) {
    const char* config_path = "/tmp/volvox-qconv-weight-only-invalid-config.json";
    const char* weights_path = "/tmp/volvox-qconv-weight-only-invalid-weights.safetensors";
    const char* config_bad_scale =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"w\",\"weight_scale\":\"ws_bad\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"groups\":1,\"weight_only\":true}}]}";
    const char* config_bad_bias =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"w\",\"weight_scale\":\"ws\",\"bias\":\"b_short\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"groups\":1,\"weight_only\":true}}]}";
    const char* config_bad_geometry =
        "{\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"w\",\"weight_scale\":\"ws\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"groups\":1,\"weight_only\":true}}]}";
    const int vector_shape[1] = {2};
    const int bad_scale_shape[1] = {3};
    const int short_shape[1] = {1};
    const int weight_shape[4] = {2, 1, 1, 2};
    const float scale[2] = {0.25f, 0.5f};
    const float bad_scale[3] = {0.25f, 0.5f, 1.0f};
    const float bias[2] = {0.1f, -0.2f};
    const float short_bias[1] = {0.1f};
    const int8_t weight[4] = {2, -1, -3, 4};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "ws", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, scale, sizeof(scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "ws_bad", SAFETENSORS_DTYPE_F32,
                                 bad_scale_shape, 1, bad_scale, sizeof(bad_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32,
                                 vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "b_short", SAFETENSORS_DTYPE_F32,
                                 short_shape, 1, short_bias, sizeof(short_bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8,
                                 weight_shape, 4, weight, sizeof(weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(expect_qconv2d_weight_only_init_rejection(
              config_path, weights_path, config_bad_scale) == 0);
    CHECK(expect_qconv2d_weight_only_init_rejection(
              config_path, weights_path, config_bad_bias) == 0);
    CHECK(expect_qconv2d_weight_only_init_rejection(
              config_path, weights_path, config_bad_geometry) == 0);
    remove(weights_path);
    return 0;
}

static int test_binary_weight_companion_scales_runtime(void) {
    const char* config_path = "/tmp/volvox-binary-companion-scales-config.json";
    const char* weights_path = "/tmp/volvox-binary-companion-scales-weights.safetensors";
    const char* config =
        "{\"weights_quantization_storage\":{\"format\":"
        "\"volvoxai-f32-companion-scales-v1\"},"
        "\"inputs\":{"
        "\"xl\":{\"shape\":[1,2],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"ids\":{\"shape\":[1],\"dtype\":\"int32\"},"
        "\"xc\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.5,\"zero_point\":128}}},"
        "\"nodes\":[{\"opType\":\"QLinear\",\"inputs\":{\"input\":\"xl\","
        "\"weight\":\"lw\",\"bias\":\"lb\"},\"outputs\":{\"out\":\"yl\"},"
        "\"outputs_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\","
        "\"scale\":0.25,\"zero_point\":0}},\"params\":{}},{"
        "\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\","
        "\"weight\":\"etable\"},\"outputs\":{\"out\":\"ye\"},"
        "\"outputs_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\","
        "\"scale\":0.25,\"zero_point\":0}},\"params\":{}},{"
        "\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"xc\",\"weight\":\"cw\","
        "\"bias\":\"cb\"},\"outputs\":{\"out\":\"yc\"},"
        "\"outputs_shape\":{\"out\":[1,1,1,2]},\"outputs_dtype\":{\"out\":\"uint8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\","
        "\"scale\":0.25,\"zero_point\":128}},\"params\":{\"data_layout\":\"NHWC\","
        "\"weight_layout\":\"OHWI\",\"groups\":1}}]}";
    const int one[1] = {1};
    const int two[1] = {2};
    const int matrix[2] = {2, 2};
    const int conv_shape[4] = {2, 1, 1, 1};
    const uint8_t prefix[1] = {7};
    const float linear_scales[2] = {0.5f, 0.25f};
    const float embedding_scales[2] = {0.25f, 0.75f};
    const float conv_scales[2] = {0.125f, 0.5f};
    const float constant_scale[1] = {0.125f};
    const int8_t linear_weight[4] = {2, 1, -1, 3};
    const int32_t linear_bias[2] = {0, 0};
    const int8_t embedding_weight[4] = {2, -4, 1, 3};
    const int8_t conv_weight[2] = {4, -2};
    const int32_t conv_bias[2] = {0, 0};
    const int8_t constant[1] = {3};
    const int8_t linear_input[2] = {2, -4};
    const int32_t ids[1] = {1};
    const uint8_t conv_input[1] = {130};
    int8_t linear_output[2] = {0};
    int8_t embedding_output[2] = {0};
    uint8_t conv_output[2] = {0};
    SafetensorsFile file;

    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "prefix", SAFETENSORS_DTYPE_U8, one, 1,
                                 prefix, sizeof(prefix)) == 0);
    CHECK(safetensors_add_tensor(&file, "lw_scale", SAFETENSORS_DTYPE_F32, two, 1,
                                 linear_scales, sizeof(linear_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "lw", SAFETENSORS_DTYPE_I8, matrix, 2,
                                 linear_weight, sizeof(linear_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "lb", SAFETENSORS_DTYPE_I32, two, 1,
                                 linear_bias, sizeof(linear_bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "etable_scale", SAFETENSORS_DTYPE_F32,
                                 two, 1, embedding_scales,
                                 sizeof(embedding_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "etable", SAFETENSORS_DTYPE_I8, matrix, 2,
                                 embedding_weight, sizeof(embedding_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "cw_scale", SAFETENSORS_DTYPE_F32, two, 1,
                                 conv_scales, sizeof(conv_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "cw", SAFETENSORS_DTYPE_I8, conv_shape, 4,
                                 conv_weight, sizeof(conv_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "cb", SAFETENSORS_DTYPE_I32, two, 1,
                                 conv_bias, sizeof(conv_bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "constant", SAFETENSORS_DTYPE_I8, one, 1,
                                 constant, sizeof(constant)) == 0);
    CHECK(safetensors_add_tensor(&file, "constant_scale", SAFETENSORS_DTYPE_F32,
                                 one, 1, constant_scale,
                                 sizeof(constant_scale)) == 0);
    CHECK(set_companion_scale_metadata(
              &file, "volvoxai-f32-companion-scales-v1",
              "{\"constant\":{\"scheme\":\"per_tensor\"},"
              "\"cw\":{\"scheme\":\"per_axis\",\"axis\":0},"
              "\"etable\":{\"scheme\":\"per_axis\",\"axis\":0},"
              "\"lw\":{\"scheme\":\"per_axis\",\"axis\":0}}") == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    {
        long numel = 0;
        int shape[8] = {0};
        int ndim = 0;
        int dtype = -1;
        size_t elem_size = 0;
        float copied[2] = {0.0f, 0.0f};
        char* inspection;
        CHECK(t_find("lw_scale") != NULL);
        CHECK(volvoxai_engine_is_model_weight("lw_scale") == 0);
        CHECK(volvoxai_engine_tensor_weight_file_index("lw_scale") == -1);
        CHECK(volvoxai_engine_tensor_info(
                  "lw_scale", &numel, shape, &ndim) != 0);
        CHECK(volvoxai_engine_tensor_info_ex(
                  "lw_scale", &numel, shape, &ndim, &dtype, &elem_size) != 0);
        CHECK(volvoxai_engine_copy_tensor_f32("lw_scale", copied, 2) != 0);
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "lw_scale", copied, sizeof(copied)) != 0);
        CHECK(volvoxai_engine_tensor_row_f32("lw_scale", -1, &ndim) == NULL);
        CHECK(volvoxai_engine_set_tensor_f32("lw_scale", linear_scales, 2) != 0);
        CHECK(volvoxai_engine_set_tensor_raw(
                  "lw_scale", T_F32, linear_scales,
                  sizeof(linear_scales)) != 0);
        CHECK(volvoxai_engine_remove_model_tensor("lw_scale") != 0);
        CHECK(t_find("lw_scale") != NULL);
        inspection = volvoxai_engine_inspect_model_json(0, 1, 0, 0);
        CHECK(inspection != NULL);
        CHECK(strstr(inspection, "\"name\":\"lw_scale\"") == NULL);
        free(inspection);
    }
    CHECK(g_qlinear_meta[0].valid && g_qembedding_meta[1].valid &&
          g_qconv_meta[2].valid);
    CHECK(memcmp(g_qlinear_meta[0].weight_scales, linear_scales,
                 2 * sizeof(float)) == 0);
    CHECK(memcmp(g_qembedding_meta[1].weight_scales, embedding_scales,
                 2 * sizeof(float)) == 0);
    CHECK(memcmp(g_qconv_meta[2].weight_scales, conv_scales,
                 2 * sizeof(float)) == 0);
    CHECK(g_qlinear_meta[0].weight_zero_points[0] == 0 &&
          g_qlinear_meta[0].weight_zero_points[1] == 0 &&
          g_qembedding_meta[1].weight_zero_points[0] == 0 &&
          g_qembedding_meta[1].weight_zero_points[1] == 0 &&
          g_qconv_meta[2].weight_zero_points[0] == 0 &&
          g_qconv_meta[2].weight_zero_points[1] == 0);
    CHECK(t_find("constant") && t_find("constant")->quantization.valid &&
          closef(t_find("constant")->quantization.scale, 0.125f) &&
          t_find("constant")->quantization.zero_point == 0);
    CHECK(volvoxai_engine_set_input_raw("xl", T_I8, linear_input,
                                        sizeof(linear_input)) == 0);
    CHECK(volvoxai_engine_set_input_raw("ids", T_I32, ids, sizeof(ids)) == 0);
    CHECK(volvoxai_engine_set_input_raw("xc", T_U8, conv_input,
                                        sizeof(conv_input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("yl", linear_output,
                                          sizeof(linear_output)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("ye", embedding_output,
                                          sizeof(embedding_output)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("yc", conv_output,
                                          sizeof(conv_output)) == 0);
    CHECK(linear_output[0] == 0 && linear_output[1] == -14);
    CHECK(embedding_output[0] == 3 && embedding_output[1] == 9);
    CHECK(conv_output[0] == 130 && conv_output[1] == 124);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int write_binary_companion_rejection_weights(const char* path,
                                                    SafetensorsDType scale_dtype,
                                                    const int* scale_shape,
                                                    int scale_ndim,
                                                    const void* scale_data,
                                                    size_t scale_bytes,
                                                    const char* storage_format,
                                                    const char* manifest_json) {
    const int weight_shape[2] = {2, 1};
    const int two[1] = {2};
    const int8_t weight[2] = {1, 2};
    const int32_t bias[2] = {0, 0};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, two, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "w_scale", scale_dtype, scale_shape,
                                 scale_ndim, scale_data, scale_bytes) == 0);
    if (storage_format || manifest_json) {
        CHECK(storage_format && manifest_json);
        CHECK(set_companion_scale_metadata(&file, storage_format, manifest_json) == 0);
    }
    CHECK(safetensors_save(path, &file) == 0);
    safetensors_free(&file);
    return 0;
}

static int expect_binary_companion_scales_rejected(const char* config_path,
                                                   const char* weights_path,
                                                   const char* config) {
    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(volvoxai_engine_init(config_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(config_path);
    return 0;
}

static int test_binary_weight_companion_scale_rejections(void) {
    const char* config_path = "/tmp/volvox-binary-companion-reject-config.json";
    const char* valid_path = "/tmp/volvox-binary-companion-reject-valid.safetensors";
    const char* dtype_path = "/tmp/volvox-binary-companion-reject-dtype.safetensors";
    const char* rank_path = "/tmp/volvox-binary-companion-reject-rank.safetensors";
    const char* count_path = "/tmp/volvox-binary-companion-reject-count.safetensors";
    const char* nan_path = "/tmp/volvox-binary-companion-reject-nan.safetensors";
    const char* no_metadata_path =
        "/tmp/volvox-binary-companion-reject-no-metadata.safetensors";
    const char* bad_marker_path =
        "/tmp/volvox-binary-companion-reject-marker.safetensors";
    const char* bad_json_path =
        "/tmp/volvox-binary-companion-reject-json.safetensors";
    const char* duplicate_path =
        "/tmp/volvox-binary-companion-reject-duplicate.safetensors";
    const char* weight_shard_path =
        "/tmp/volvox-binary-companion-weight-shard.safetensors";
    const char* unquantized_shard_path =
        "/tmp/volvox-binary-companion-unquantized-shard.safetensors";
    const char* valid_manifest =
        "{\"w\":{\"scheme\":\"per_axis\",\"axis\":0}}";
    const char* graph_prefix =
        "\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\","
        "\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":0}}},\"nodes\":[{\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"outputs_quantization\":{"
        "\"out\":{\"scheme\":\"per_tensor\",\"scale\":1.0,\"zero_point\":0}},"
        "\"params\":{}}]}";
    const int vector2[1] = {2};
    const int vector1[1] = {1};
    const int matrix12[2] = {1, 2};
    const float valid_scales[2] = {0.5f, 0.25f};
    const float nan_scales[2] = {NAN, 0.25f};
    const int32_t integer_scales[2] = {1, 1};
    char config[2048];

    CHECK(write_binary_companion_rejection_weights(
              valid_path, SAFETENSORS_DTYPE_F32, vector2, 1,
              valid_scales, sizeof(valid_scales),
              "volvoxai-f32-companion-scales-v1", valid_manifest) == 0);
    CHECK(write_binary_companion_rejection_weights(
              dtype_path, SAFETENSORS_DTYPE_I32, vector2, 1,
              integer_scales, sizeof(integer_scales),
              "volvoxai-f32-companion-scales-v1", valid_manifest) == 0);
    CHECK(write_binary_companion_rejection_weights(
              rank_path, SAFETENSORS_DTYPE_F32, matrix12, 2,
              valid_scales, sizeof(valid_scales),
              "volvoxai-f32-companion-scales-v1", valid_manifest) == 0);
    CHECK(write_binary_companion_rejection_weights(
              count_path, SAFETENSORS_DTYPE_F32, vector1, 1,
              valid_scales, sizeof(float),
              "volvoxai-f32-companion-scales-v1", valid_manifest) == 0);
    CHECK(write_binary_companion_rejection_weights(
              nan_path, SAFETENSORS_DTYPE_F32, vector2, 1,
              nan_scales, sizeof(nan_scales),
              "volvoxai-f32-companion-scales-v1", valid_manifest) == 0);
    CHECK(write_binary_companion_rejection_weights(
              no_metadata_path, SAFETENSORS_DTYPE_F32, vector2, 1,
              valid_scales, sizeof(valid_scales), NULL, NULL) == 0);
    CHECK(write_binary_companion_rejection_weights(
              bad_marker_path, SAFETENSORS_DTYPE_F32, vector2, 1,
              valid_scales, sizeof(valid_scales), "unknown", valid_manifest) == 0);
    CHECK(write_binary_companion_rejection_weights(
              bad_json_path, SAFETENSORS_DTYPE_F32, vector2, 1,
              valid_scales, sizeof(valid_scales),
              "volvoxai-f32-companion-scales-v1", "{") == 0);
    {
        SafetensorsFile duplicate;
        CHECK(safetensors_init_empty(&duplicate, SAFETENSORS_OPEN_READ_WRITE) == 0);
        CHECK(safetensors_add_tensor(&duplicate, "w_scale", SAFETENSORS_DTYPE_F32,
                                     vector2, 1, valid_scales,
                                     sizeof(valid_scales)) == 0);
        CHECK(safetensors_save(duplicate_path, &duplicate) == 0);
        safetensors_free(&duplicate);
    }

#define EXPECT_COMPANION_FILE_REJECTED(path) do { \
    CHECK(snprintf(config, sizeof(config), \
                   "{\"weights_quantization_storage\":{\"format\":" \
                   "\"volvoxai-f32-companion-scales-v1\"},%s", \
                   graph_prefix) > 0); \
    CHECK(expect_binary_companion_scales_rejected(config_path, path, config) == 0); \
} while (0)

    EXPECT_COMPANION_FILE_REJECTED(dtype_path);
    EXPECT_COMPANION_FILE_REJECTED(rank_path);
    EXPECT_COMPANION_FILE_REJECTED(count_path);
    EXPECT_COMPANION_FILE_REJECTED(nan_path);
    EXPECT_COMPANION_FILE_REJECTED(no_metadata_path);
    EXPECT_COMPANION_FILE_REJECTED(bad_marker_path);
    EXPECT_COMPANION_FILE_REJECTED(bad_json_path);

    for (int malformed = 0; malformed < 7; malformed++) {
        const char* manifests[] = {
            "{\"w\":{\"scheme\":\"per_axis\",\"axis\":0,\"scales\":[0.5,0.25]}}",
            "{\"w\":{\"scheme\":\"per_axis\",\"axis\":1}}",
            "{\"w\":{\"scheme\":\"per_tensor\",\"scale\":0.5}}",
            "{\"w\":{\"scheme\":\"unknown\"}}",
            "{\"missing\":{\"scheme\":\"per_axis\",\"axis\":0}}",
            "{\"w\":{\"scheme\":\"per_axis\",\"axis\":0},"
                "\"w\":{\"scheme\":\"per_axis\",\"axis\":0}}",
            "{\"w\":{\"scheme\":\"per_axis\","
                "\"scheme\":\"per_axis\",\"axis\":0}}",
        };
        CHECK(write_binary_companion_rejection_weights(
                  valid_path, SAFETENSORS_DTYPE_F32, vector2, 1,
                  valid_scales, sizeof(valid_scales),
                  "volvoxai-f32-companion-scales-v1", manifests[malformed]) == 0);
        EXPECT_COMPANION_FILE_REJECTED(valid_path);
    }
    CHECK(write_binary_companion_rejection_weights(
              valid_path, SAFETENSORS_DTYPE_F32, vector2, 1,
              valid_scales, sizeof(valid_scales),
              "volvoxai-f32-companion-scales-v1", valid_manifest) == 0);

    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\",\"extra\":1},%s",
                   graph_prefix) > 0);
    CHECK(expect_binary_companion_scales_rejected(
              config_path, valid_path, config) == 0);
    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":\"unknown\"},%s",
                   graph_prefix) > 0);
    CHECK(expect_binary_companion_scales_rejected(
              config_path, valid_path, config) == 0);
    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\"},"
                   "\"WEIGHTS_QUANTIZATION\":{\"w\":{\"scheme\":"
                   "\"per_axis\",\"axis\":0}},%s", graph_prefix) > 0);
    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(volvoxai_engine_init(config_path, valid_path) == 0);
    CHECK(g_qlinear_meta[0].valid);
    volvoxai_engine_shutdown();
    remove(config_path);
    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\"},"
                   "\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\"},%s",
                   graph_prefix) > 0);
    CHECK(expect_binary_companion_scales_rejected(
              config_path, valid_path, config) == 0);
    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\"},"
                   "\"weights_quantization\":{\"w\":{\"scheme\":\"per_axis\","
                   "\"axis\":0}},%s", graph_prefix) > 0);
    CHECK(expect_binary_companion_scales_rejected(
              config_path, valid_path, config) == 0);
    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\"},"
                   "\"nodes\":[{\"op\":\"Identity\","
                   "\"inputs\":{\"input\":\"w_scale\"},"
                   "\"outputs\":{\"out\":\"leaked\"},"
                   "\"outputs_shape\":{\"out\":[2]}}]}") > 0);
    CHECK(expect_binary_companion_scales_rejected(
              config_path, valid_path, config) == 0);
    CHECK(snprintf(config, sizeof(config),
                   "{\"weights_quantization_storage\":{\"format\":"
                   "\"volvoxai-f32-companion-scales-v1\"},%s",
                   graph_prefix) > 0);
    CHECK(write_quantized_test_config(config_path, config) == 0);
    {
        const int weight_shape[2] = {2, 1};
        const int bias_shape[1] = {2};
        const int8_t weight[2] = {1, 2};
        const int32_t bias[2] = {0, 0};
        SafetensorsFile weight_shard;
        SafetensorsFile unquantized_shard;
        const char* paths[2] = {weight_shard_path, unquantized_shard_path};
        CHECK(safetensors_init_empty(
                  &weight_shard, SAFETENSORS_OPEN_READ_WRITE) == 0);
        CHECK(safetensors_add_tensor(
                  &weight_shard, "w", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                  weight, sizeof(weight)) == 0);
        CHECK(safetensors_add_tensor(
                  &weight_shard, "w_scale", SAFETENSORS_DTYPE_F32, vector2, 1,
                  valid_scales, sizeof(valid_scales)) == 0);
        CHECK(set_companion_scale_metadata(
                  &weight_shard, "volvoxai-f32-companion-scales-v1",
                  valid_manifest) == 0);
        CHECK(safetensors_save(weight_shard_path, &weight_shard) == 0);
        safetensors_free(&weight_shard);
        CHECK(safetensors_init_empty(
                  &unquantized_shard, SAFETENSORS_OPEN_READ_WRITE) == 0);
        CHECK(safetensors_add_tensor(
                  &unquantized_shard, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                  bias, sizeof(bias)) == 0);
        CHECK(safetensors_save(unquantized_shard_path, &unquantized_shard) == 0);
        safetensors_free(&unquantized_shard);
        CHECK(volvoxai_engine_init_with_weight_files(config_path, paths, 2) == 0);
        CHECK(g_qlinear_meta[0].valid);
        volvoxai_engine_shutdown();

        /* A descriptor, base, and companion must remain in one shard. */
        CHECK(safetensors_init_empty(
                  &weight_shard, SAFETENSORS_OPEN_READ_WRITE) == 0);
        CHECK(safetensors_add_tensor(
                  &weight_shard, "w", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                  weight, sizeof(weight)) == 0);
        CHECK(set_companion_scale_metadata(
                  &weight_shard, "volvoxai-f32-companion-scales-v1",
                  valid_manifest) == 0);
        CHECK(safetensors_save(weight_shard_path, &weight_shard) == 0);
        safetensors_free(&weight_shard);
        CHECK(safetensors_init_empty(
                  &unquantized_shard, SAFETENSORS_OPEN_READ_WRITE) == 0);
        CHECK(safetensors_add_tensor(
                  &unquantized_shard, "w_scale", SAFETENSORS_DTYPE_F32,
                  vector2, 1, valid_scales, sizeof(valid_scales)) == 0);
        CHECK(safetensors_add_tensor(
                  &unquantized_shard, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                  bias, sizeof(bias)) == 0);
        CHECK(safetensors_save(unquantized_shard_path, &unquantized_shard) == 0);
        safetensors_free(&unquantized_shard);
        CHECK(volvoxai_engine_init_with_weight_files(config_path, paths, 2) != 0);
        volvoxai_engine_shutdown();
    }
    {
        const char* paths[2] = {valid_path, duplicate_path};
        CHECK(volvoxai_engine_init_with_weight_files(config_path, paths, 2) != 0);
    }
    volvoxai_engine_shutdown();
    remove(config_path);
#undef EXPECT_COMPANION_FILE_REJECTED
    remove(valid_path);
    remove(dtype_path);
    remove(rank_path);
    remove(count_path);
    remove(nan_path);
    remove(no_metadata_path);
    remove(bad_marker_path);
    remove(bad_json_path);
    remove(duplicate_path);
    remove(weight_shard_path);
    remove(unquantized_shard_path);
    return 0;
}

/* Tensor names are case-sensitive. Keep a valid but incompatible `W`
 * descriptor before `w` so cJSON's default case-insensitive lookup would
 * select the wrong quantization record and fail QLinear initialization. */
static int test_case_distinct_weight_quantization_descriptors(void) {
    const char* config_path =
        "/tmp/volvox-case-distinct-quantization-config.json";
    const char* weights_path =
        "/tmp/volvox-case-distinct-quantization-weights.safetensors";
    const char* config =
        "{\"weights_quantization_storage\":{\"format\":"
        "\"volvoxai-f32-companion-scales-v1\"},"
        "\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\","
        "\"quantization\":{\"scheme\":\"per_tensor\",\"scale\":1.0,"
        "\"zero_point\":0}}},\"nodes\":[{\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\","
        "\"scale\":1.0,\"zero_point\":0}},\"params\":{}}]}";
    const char* manifest =
        "{\"W\":{\"scheme\":\"per_tensor\"},"
        "\"w\":{\"scheme\":\"per_axis\",\"axis\":0}}";
    const int weight_shape[2] = {2, 1};
    const int vector2[1] = {2};
    const int vector1[1] = {1};
    const int8_t lower_weight[2] = {1, 2};
    const int8_t upper_weight[2] = {3, 4};
    const int32_t bias[2] = {0, 0};
    const float lower_scales[2] = {0.5f, 0.25f};
    const float upper_scale[1] = {0.75f};
    int32_t axis = -1;
    int32_t count = 0;
    SafetensorsFile file;

    CHECK(write_quantized_test_config(config_path, config) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "W", SAFETENSORS_DTYPE_I8,
                                 weight_shape, 2, upper_weight,
                                 sizeof(upper_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "W_scale", SAFETENSORS_DTYPE_F32,
                                 vector1, 1, upper_scale,
                                 sizeof(upper_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8,
                                 weight_shape, 2, lower_weight,
                                 sizeof(lower_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "w_scale", SAFETENSORS_DTYPE_F32,
                                 vector2, 1, lower_scales,
                                 sizeof(lower_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32,
                                 vector2, 1, bias, sizeof(bias)) == 0);
    CHECK(set_companion_scale_metadata(
              &file, "volvoxai-f32-companion-scales-v1", manifest) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    CHECK(g_qlinear_meta[0].valid);
    CHECK(vx_tensor_quant_axis((const VxTensor*)t_find("w"), &axis, &count));
    CHECK(axis == 0 && count == 2);
    CHECK(vx_tensor_quantization_kind((const VxTensor*)t_find("W")) ==
          VX_QUANT_PER_TENSOR);
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

static int test_w8a32_linear_regression(void) {
    const char* config_path = "/tmp/volvox-w8a32-config.json";
    const char* weights_path = "/tmp/volvox-w8a32-weights.safetensors";
    const char* config =
        /* Historic native blueprints used this label with F32 backing and no
           descriptor. The physical-byte parser must retain that behavior. */
        "{\"inputs\":{\"x\":{\"shape\":[1,3],\"dtype\":\"int8\"}},\"nodes\":[{"
        "\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\","
        "\"weight_scale\":\"s\",\"weight_zero_point\":\"z\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"weight_layout\":\"OUT_IN\"}}]}";
    FILE* config_file = fopen(config_path, "wb");
    CHECK(config_file != NULL && fwrite(config, 1, strlen(config), config_file) == strlen(config));
    CHECK(fclose(config_file) == 0);

    const int matrix_shape[2] = {2, 3};
    const int vector_shape[1] = {2};
    const int8_t weight[6] = {2, -1, 3, -2, 4, 1};
    const float scale[2] = {0.5f, 0.25f};
    const int32_t zero_point[2] = {1, -1};
    const float bias[2] = {0.25f, -0.5f};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8, matrix_shape, 2, weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "s", SAFETENSORS_DTYPE_F32, vector_shape, 1, scale, sizeof(scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "z", SAFETENSORS_DTYPE_I32, vector_shape, 1,
                                 zero_point, sizeof(zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32, vector_shape, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(config_path, weights_path) == 0);
    CHECK(t_find("x") && t_find("x")->dtype == T_F32 &&
          !t_find("x")->quantization.valid);
    long input_numel = 0;
    float* input = volvoxai_engine_input_ptr("x", &input_numel);
    CHECK(input && input_numel == 3);
    input[0] = 1.0f;
    input[1] = 2.0f;
    input[2] = -1.0f;
    CHECK(volvoxai_engine_forward() == 0);
    float output[2] = {0};
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 2) == 0);
    CHECK(closef(output[0], -2.25f));
    CHECK(closef(output[1], 1.25f));
    volvoxai_engine_shutdown();
    remove(config_path);
    remove(weights_path);
    return 0;
}

typedef struct {
    int* visits;
} ThreadPoolVisitContext;

static void visit_parallel_range(void* opaque, int begin, int end) {
    ThreadPoolVisitContext* context = (ThreadPoolVisitContext*)opaque;
    for (int index = begin; index < end; index++) context->visits[index]++;
}

static int test_kernel_thread_pool_restart(void) {
    int visits[97] = {0};
    ThreadPoolVisitContext context = {visits};
    vx_set_num_threads(3);
#if !defined(__wasm__) && !defined(_WIN32)
    CHECK(vx_kernels_thread_count() == 3);
#else
    CHECK(vx_kernels_thread_count() == 1);
#endif
    vx_kernels_parallel_for(97, 4, visit_parallel_range, &context);
    for (int index = 0; index < 97; index++) CHECK(visits[index] == 1);
    vx_kernels_shutdown();
    memset(visits, 0, sizeof(visits));
    vx_kernels_parallel_for(97, 3, visit_parallel_range, &context);
    for (int index = 0; index < 97; index++) CHECK(visits[index] == 1);
    vx_kernels_shutdown();
    return 0;
}

int main(void) {
    CHECK(test_safetensors_rejects_recursive_duplicate_header_keys() == 0);
    CHECK(test_native_cpu_feature_contract() == 0);
    CHECK(test_groupnorm_optional_bias_kernel() == 0);
    CHECK(test_qembedding_portable_kernel() == 0);
    CHECK(test_qsilu_portable_kernel() == 0);
    CHECK(test_qgelu_portable_kernel() == 0);
    CHECK(test_quantized_activation_lut_exactness() == 0);
    CHECK(test_qgroupnorm_portable_kernel() == 0);
    CHECK(test_qlayernorm_portable_kernel() == 0);
    CHECK(test_qsdpa_portable_kernel() == 0);
    CHECK(test_qargmax_portable_kernel() == 0);
    CHECK(test_qmaskedmean_portable_kernel() == 0);
    CHECK(test_w8a8_linear_vector_tail() == 0);
    CHECK(test_physical_qlinear_native_dispatch() == 0);
    CHECK(test_physical_qlinear_native_threaded_dispatch() == 0);
    CHECK(test_physical_qlinear_vnni_remapping() == 0);
    CHECK(test_physical_qconv_native_dispatch() == 0);
    CHECK(test_physical_qconv_native_threaded_dispatch() == 0);
    CHECK(test_physical_qconv_native_grayscale_stem_dispatch() == 0);
    CHECK(test_physical_qconv_vnni_remapping() == 0);
    CHECK(test_physical_qconv_arm_try() == 0);
    CHECK(test_physical_qlinear_i8_to_u8() == 0);
    CHECK(test_physical_qembedding_i8_to_u8() == 0);
    CHECK(test_physical_qembedding_vulkan_changed_ids() == 0);
    CHECK(test_physical_qmatmul_u8_to_i8() == 0);
    CHECK(test_physical_qconv2d_grouped_relu6() == 0);
    CHECK(test_physical_qconv2d_i8_bias_relu() == 0);
    CHECK(test_physical_qadd_and_requantize() == 0);
    CHECK(test_physical_qadd_static_weight_quantization() == 0);
    CHECK(test_physical_qsilu_chain() == 0);
    CHECK(test_physical_qgelu_chain() == 0);
    CHECK(test_physical_qgroupnorm_chain() == 0);
    CHECK(test_physical_qlayernorm_chain() == 0);
    CHECK(test_physical_qsdpa_chain() == 0);
    CHECK(test_physical_qargmax() == 0);
    CHECK(test_physical_qmaskedmean() == 0);
    CHECK(test_physical_qgelu_vulkan_chain() == 0);
    CHECK(test_physical_qgroupnorm_vulkan_chain() == 0);
    CHECK(test_physical_qlayernorm_vulkan_chain() == 0);
    CHECK(test_physical_qsdpa_vulkan_chain() == 0);
    CHECK(test_physical_qargmax_vulkan() == 0);
    CHECK(test_physical_qmaskedmean_vulkan() == 0);
    CHECK(test_physical_qconv_qadd_requantize_vulkan_chain() == 0);
    CHECK(test_physical_dequantize_sigmoid_quantize() == 0);
    CHECK(test_physical_quantize_nan_and_saturation() == 0);
    CHECK(test_set_input_f32_quantizes_physical_inputs() == 0);
    CHECK(test_concat_sigmoid_fusion_backend_scope() == 0);
    CHECK(test_physical_shape_island_chain() == 0);
    CHECK(test_physical_shape_maxpool_asymmetric_pads() == 0);
    CHECK(test_physical_shape_rejections() == 0);
    CHECK(test_generic_add_rejects_implicit_quantization() == 0);
    CHECK(test_generic_add_rejects_physical_bytes() == 0);
    CHECK(test_w8a8_linear_chain() == 0);
    CHECK(test_implicit_qconv2d_activation_rejected() == 0);
    CHECK(test_qconv2d_weight_only_f32_activation() == 0);
    CHECK(test_qconv2d_weight_only_rejects_malformed_descriptors() == 0);
    CHECK(test_binary_weight_companion_scales_runtime() == 0);
    CHECK(test_binary_weight_companion_scale_rejections() == 0);
    CHECK(test_case_distinct_weight_quantization_descriptors() == 0);
    CHECK(test_w8a32_linear_regression() == 0);
    CHECK(test_kernel_thread_pool_restart() == 0);
    puts("quantized runtime tests passed");
    return 0;
}
