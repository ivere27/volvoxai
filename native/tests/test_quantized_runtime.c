#include "engine_internal.h"
#include "cpu_features.h"
#include "inference_kernels.h"
#include "qconv_w8a8_arm.h"
#include "qlinear_w8a8_arm.h"
#include "quant_cpu_opt.h"
#include "safetensors.h"
#include "thread_pool.h"
#include "tensor_f32_opt.h"
#include "volvoxai_backend.h"
#include "runtime_state.h"
#if VOLVOXAI_ENABLE_VULKAN
#include "vulkan_engine.h"
#else
/* GPU parity cases use initialization failure as their ordinary skip path. */
static int vk_init(void) { return -1; }
static void vk_cleanup(void) {}
#endif
#include "engine_core.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do { if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #x); return -1; } } while (0)

static int closef(float left, float right) {
    return fabsf(left - right) <= 1.0e-6f * (1.0f + fabsf(left) + fabsf(right));
}

/* Canonical physical-quantization fixtures share immutable parameter tensors.
 * Positive graphs name these tensors from the central
 * graph.quantization.tensors table; affine values live only in safetensors. */
static int fixture_add_current_affine_catalog(SafetensorsFile* file) {
    if (!file) return -1;
    static const float scale_0[] = {1.0f};
    static const float scale_1[] = {0.5f, 0.5f, 0.5f};
    static const float scale_2[] = {0.25f};
    static const float scale_3[] = {0.5f, 0.25f, 1.0f};
    static const float scale_4[] = {0.125f};
    static const float scale_5[] = {0.5f, 0.25f};
    static const float scale_6[] = {0.25f, 0.5f};
    static const float scale_7[] = {0.5f};
    static const int8_t zero_0[] = {0};
    static const uint8_t zero_1[] = {128};
    static const int8_t zero_2[] = {0, 0, 0};
    static const int8_t zero_3[] = {0, 1, -1};
    static const int8_t zero_4[] = {-3};
    static const uint8_t zero_5[] = {128, 128};
    static const uint8_t zero_6[] = {123};
    static const uint8_t zero_7[] = {128, 127};
    static const uint8_t zero_8[] = {10};
    static const uint8_t zero_9[] = {120};
    static const int8_t zero_10[] = {3};
    static const int8_t zero_11[] = {-1};
    static const int8_t zero_12[] = {-4};
    static const int8_t zero_13[] = {-17};
    static const int8_t zero_14[] = {5};
    static const uint8_t zero_15[] = {129};
    static const uint8_t zero_16[] = {130};
    static const int8_t zero_17[] = {-2};
    static const uint8_t zero_18[] = {100};
    static const uint8_t zero_19[] = {4};
    static const int8_t zero_20[] = {-5};

    if (!safetensors_find_tensor(file, "__fixture_affine.scale.0") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.0",
            VX_DTYPE_F32, (const int[]){1}, 1, scale_0,
            sizeof(scale_0)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.1") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.1",
            VX_DTYPE_F32, (const int[]){3}, 1, scale_1,
            sizeof(scale_1)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.2") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.2",
            VX_DTYPE_F32, (const int[]){1}, 1, scale_2,
            sizeof(scale_2)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.3") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.3",
            VX_DTYPE_F32, (const int[]){3}, 1, scale_3,
            sizeof(scale_3)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.4") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.4",
            VX_DTYPE_F32, (const int[]){1}, 1, scale_4,
            sizeof(scale_4)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.5") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.5",
            VX_DTYPE_F32, (const int[]){2}, 1, scale_5,
            sizeof(scale_5)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.6") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.6",
            VX_DTYPE_F32, (const int[]){2}, 1, scale_6,
            sizeof(scale_6)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.scale.7") &&
        safetensors_add_tensor(file, "__fixture_affine.scale.7",
            VX_DTYPE_F32, (const int[]){1}, 1, scale_7,
            sizeof(scale_7)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.0") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.0",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_0,
            sizeof(zero_0)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.1") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.1",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_1,
            sizeof(zero_1)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.2") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.2",
            VX_DTYPE_I8, (const int[]){3}, 1, zero_2,
            sizeof(zero_2)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.3") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.3",
            VX_DTYPE_I8, (const int[]){3}, 1, zero_3,
            sizeof(zero_3)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.4") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.4",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_4,
            sizeof(zero_4)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.5") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.5",
            VX_DTYPE_U8, (const int[]){2}, 1, zero_5,
            sizeof(zero_5)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.6") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.6",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_6,
            sizeof(zero_6)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.7") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.7",
            VX_DTYPE_U8, (const int[]){2}, 1, zero_7,
            sizeof(zero_7)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.8") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.8",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_8,
            sizeof(zero_8)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.9") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.9",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_9,
            sizeof(zero_9)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.10") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.10",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_10,
            sizeof(zero_10)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.11") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.11",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_11,
            sizeof(zero_11)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.12") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.12",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_12,
            sizeof(zero_12)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.13") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.13",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_13,
            sizeof(zero_13)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.14") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.14",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_14,
            sizeof(zero_14)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.15") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.15",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_15,
            sizeof(zero_15)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.16") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.16",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_16,
            sizeof(zero_16)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.17") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.17",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_17,
            sizeof(zero_17)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.18") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.18",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_18,
            sizeof(zero_18)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.19") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.19",
            VX_DTYPE_U8, (const int[]){1}, 1, zero_19,
            sizeof(zero_19)) != 0) return -1;
    if (!safetensors_find_tensor(file, "__fixture_affine.zero.20") &&
        safetensors_add_tensor(file, "__fixture_affine.zero.20",
            VX_DTYPE_I8, (const int[]){1}, 1, zero_20,
            sizeof(zero_20)) != 0) return -1;
    return 0;
}

static int fixture_engine_init(const char* graph_path, const char* weights_path) {
    char generated_weights_path[512] = {0};
    const char* prepared_weights_path = weights_path;
    SafetensorsFile weights;
    SafetensorsLoadOptions options = {SAFETENSORS_OPEN_READ_WRITE};
    int generated = 0;
    int result;
    memset(&weights, 0, sizeof(weights));
    if (!graph_path) return -1;
    if (!prepared_weights_path) {
        int length = snprintf(generated_weights_path,
                              sizeof(generated_weights_path),
                              "%s.current-affine.safetensors", graph_path);
        if (length <= 0 || (size_t)length >= sizeof(generated_weights_path) ||
            safetensors_init_empty(&weights, SAFETENSORS_OPEN_READ_WRITE) != 0)
            return -1;
        prepared_weights_path = generated_weights_path;
        generated = 1;
    } else if (safetensors_load_with_options(
                   prepared_weights_path, &options, &weights) != 0) {
        return -1;
    }
    if (fixture_add_current_affine_catalog(&weights) != 0 ||
        safetensors_save(prepared_weights_path, &weights) != 0) {
        safetensors_free(&weights);
        if (generated) remove(generated_weights_path);
        return -1;
    }
    safetensors_free(&weights);
    result = (volvoxai_engine_init)(graph_path, prepared_weights_path);
    if (generated) remove(generated_weights_path);
    return result;
}

#define volvoxai_engine_init(graph_path, weights_path) \
    fixture_engine_init((graph_path), (weights_path))

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
        "{\"__metadata__\":{\"fixture_tag\":\"first\","
        "\"fixture_tag\":\"second\"}}";
    const char* non_string_metadata =
        "{\"__metadata__\":{\"fixture_tag\":1}}";
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
              &file, "{\"fixture_tag\":1}") != 0);
    CHECK(safetensors_set_metadata_json(
              &file, "{\"fixture_tag\":\"value\"}") == 0);
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

static int test_groupnorm_narrow_group_kernel(void) {
    enum { SPATIAL = 5, CHANNELS = 12, GROUPS = 4, CPG = 3 };
    float input[SPATIAL * CHANNELS];
    float weight[CHANNELS];
    float bias[CHANNELS];
    float expected[SPATIAL * CHANNELS];
    float actual[SPATIAL * CHANNELS];
    const double epsilon = 1.0e-5;
    for (int index = 0; index < SPATIAL * CHANNELS; index++)
        input[index] = (float)((index * 17 % 29) - 14) * 0.125f;
    for (int channel = 0; channel < CHANNELS; channel++) {
        weight[channel] = 0.5f + (float)(channel % 5) * 0.125f;
        bias[channel] = (float)(channel - 6) * 0.03125f;
    }
    for (int group = 0; group < GROUPS; group++) {
        double sum = 0.0;
        double square = 0.0;
        const int first = group * CPG;
        for (int point = 0; point < SPATIAL; point++)
            for (int local = 0; local < CPG; local++)
                sum += (double)input[point * CHANNELS + first + local];
        {
            const double mean = sum / (double)(SPATIAL * CPG);
            for (int point = 0; point < SPATIAL; point++) {
                for (int local = 0; local < CPG; local++) {
                    const double centered =
                        (double)input[point * CHANNELS + first + local] - mean;
                    square += centered * centered;
                }
            }
            {
                const double inverse = 1.0 /
                    sqrt(square / (double)(SPATIAL * CPG) + epsilon);
                for (int point = 0; point < SPATIAL; point++) {
                    for (int local = 0; local < CPG; local++) {
                        const int channel = first + local;
                        const int index = point * CHANNELS + channel;
                        expected[index] = (float)(((double)input[index] - mean) *
                            inverse * weight[channel] + bias[channel]);
                    }
                }
            }
        }
    }
    CHECK(groupnorm_f32(input, weight, bias, actual, 1, 1, SPATIAL,
                        CHANNELS, GROUPS, epsilon));
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    return 0;
}

static int8_t quantize_i8(float value, float scale, int zp) {
    int q = (int)lrintf(value / scale + (float)zp);
    if (q < -128) q = -128;
    if (q > 127) q = 127;
    return (int8_t)q;
}

static int test_qbatch_matmul_portable_kernel(void) {
    const uint8_t a[4] = {129, 130, 131, 132};
    const int8_t b[4] = {0, -1, -1, 0};
    const int8_t expected[4] = {4, 4, 4, 5};
    int8_t output[4] = {0};
    CHECK(qbatch_matmul_i8u8(
              a, b, output, 2, 2, 2,
              0.5f, 128, 0.25f, -1, 0.25f, 3,
              VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    /* A centered magnitude of 255 on both operands exceeds the defined I32
       accumulator at K=33026 and must fail before reading matrix storage. */
    CHECK(!qbatch_matmul_i8u8(
              a, b, output, 1, 33026, 1,
              1.0f, 0, 1.0f, -128, 1.0f, 0,
              VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8));
    return 0;
}

static int test_qbatch_matmul_native_dispatch(void) {
    enum { M = 113, K = 37, N = 257 };
    uint8_t *a = (uint8_t *)malloc((size_t)M * K);
    uint8_t *b = (uint8_t *)malloc((size_t)K * N);
    uint8_t *expected = (uint8_t *)malloc((size_t)M * N);
    uint8_t *actual = (uint8_t *)malloc((size_t)M * N);
    CHECK(a && b && expected && actual);
    for (size_t index = 0; index < (size_t)M * K; index++)
        a[index] = (uint8_t)(index * 197u + (index >> 2u) * 29u + 255u);
    for (size_t index = 0; index < (size_t)K * N; index++)
        b[index] = (uint8_t)(index * 131u + (index >> 3u) * 53u + 128u);
    for (uint32_t a_unsigned = 0; a_unsigned < 2u; a_unsigned++) {
        for (uint32_t b_unsigned = 0; b_unsigned < 2u; b_unsigned++) {
            for (uint32_t output_unsigned = 0; output_unsigned < 2u;
                 output_unsigned++) {
                const uint32_t a_dtype =
                    a_unsigned ? VX_DTYPE_U8 : VX_DTYPE_I8;
                const uint32_t b_dtype =
                    b_unsigned ? VX_DTYPE_U8 : VX_DTYPE_I8;
                const uint32_t output_dtype =
                    output_unsigned ? VX_DTYPE_U8 : VX_DTYPE_I8;
                const int32_t a_zero_point = a_unsigned ? 141 : -23;
                const int32_t b_zero_point = b_unsigned ? 109 : 17;
                const int32_t output_zero_point = output_unsigned ? 129 : -7;
                CHECK(qbatch_matmul_i8u8(
                    a, b, expected, M, K, N,
                    0.013f, a_zero_point, 0.021f, b_zero_point,
                    0.031f, output_zero_point,
                    a_dtype, b_dtype, output_dtype));
                for (int threads = 1; threads <= 4; threads += 3) {
                    memset(actual, 0x5a, (size_t)M * N);
                    vx_set_num_threads(threads);
                    CHECK(vx_qbatch_matmul_i8u8_native(
                        a, b, actual, M, K, N,
                        0.013f, a_zero_point, 0.021f, b_zero_point,
                        0.031f, output_zero_point,
                        a_dtype, b_dtype, output_dtype));
                    CHECK(memcmp(actual, expected, (size_t)M * N) == 0);
                }
            }
        }
    }
    vx_set_num_threads(0);
    free(actual);
    free(expected);
    free(b);
    free(a);
    return 0;
}

static int test_transpose_i8u8_portable_kernel(void) {
    const uint32_t input_shape[4] = {1, 2, 2, 3};
    const uint32_t nchw_to_nhwc[4] = {0, 2, 3, 1};
    const uint32_t duplicate_axis[4] = {0, 2, 2, 1};
    const uint8_t input[12] = {
        0, 1, 2, 3, 4, 5,
        6, 7, 8, 9, 10, 11,
    };
    const uint8_t expected[12] = {
        0, 6, 1, 7, 2, 8,
        3, 9, 4, 10, 5, 11,
    };
    uint8_t output[12] = {0};
    CHECK(transpose_nd_i8u8(
              input, output, input_shape, nchw_to_nhwc, 4, 12,
              VX_DTYPE_U8));
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(!transpose_nd_i8u8(
              input, output, input_shape, duplicate_axis, 4, 12,
              VX_DTYPE_U8));
    CHECK(!transpose_nd_i8u8(
              input, output, input_shape, nchw_to_nhwc, 4, 11,
              VX_DTYPE_U8));
    CHECK(!transpose_nd_i8u8(
              input, output, input_shape, nchw_to_nhwc, 4, 12,
              VX_DTYPE_I32));
    CHECK(!transpose_nd_i8u8(
              output, output, input_shape, nchw_to_nhwc, 4, 12,
              VX_DTYPE_U8));
    return 0;
}

static int test_transpose_i8u8_native_kernel(void) {
    enum {
        image_batch = 2,
        image_height = 17,
        image_width = 19,
        image_channels = 13,
        image_elements = image_batch * image_height * image_width * image_channels,
        matrix_batch = 2,
        matrix_rows = 37,
        matrix_columns = 45,
        matrix_elements = matrix_batch * matrix_rows * matrix_columns,
    };
    const uint32_t nhwc_shape[4] = {
        image_batch, image_height, image_width, image_channels,
    };
    const uint32_t nchw_shape[4] = {
        image_batch, image_channels, image_height, image_width,
    };
    const uint32_t nhwc_to_nchw[4] = {0, 3, 1, 2};
    const uint32_t nchw_to_nhwc[4] = {0, 2, 3, 1};
    const uint32_t matrix_shape[3] = {
        matrix_batch, matrix_rows, matrix_columns,
    };
    const uint32_t matrix_permutation[3] = {0, 2, 1};
    uint8_t image_input[image_elements];
    uint8_t image_reference[image_elements];
    uint8_t image_actual[image_elements];
    uint8_t image_roundtrip[image_elements];
    uint8_t matrix_input[matrix_elements];
    uint8_t matrix_reference[matrix_elements];
    uint8_t matrix_actual[matrix_elements];
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    int forward_result;
    int reverse_result;
    int matrix_result;
    for (int index = 0; index < image_elements; index++)
        image_input[index] = (uint8_t)((index * 37 + index / 11) & 0xff);
    for (int index = 0; index < matrix_elements; index++)
        matrix_input[index] = (uint8_t)((index * 29 + index / 7) & 0xff);
    CHECK(transpose_nd_i8u8(image_input, image_reference, nhwc_shape,
                            nhwc_to_nchw, 4u, image_elements,
                            VX_DTYPE_U8) == 1);
    CHECK(transpose_nd_i8u8(matrix_input, matrix_reference, matrix_shape,
                            matrix_permutation, 3u, matrix_elements,
                            VX_DTYPE_U8) == 1);
    pool = vx_kernel_thread_pool_create(4);
    CHECK(pool != NULL);
    scope = vx_kernel_thread_pool_scope_enter(pool);
    forward_result = vx_transpose_nd_i8u8_native_validated(
        image_input, image_actual, nhwc_shape, nhwc_to_nchw, 4u,
        image_elements, VX_DTYPE_U8);
    reverse_result = vx_transpose_nd_i8u8_native_validated(
        image_actual, image_roundtrip, nchw_shape, nchw_to_nhwc, 4u,
        image_elements, VX_DTYPE_U8);
    matrix_result = vx_transpose_nd_i8u8_native_validated(
        matrix_input, matrix_actual, matrix_shape, matrix_permutation, 3u,
        matrix_elements, VX_DTYPE_U8);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    CHECK(forward_result == 1);
    CHECK(reverse_result == 1);
    CHECK(matrix_result == 1);
    CHECK(memcmp(image_actual, image_reference, image_elements) == 0);
    CHECK(memcmp(image_roundtrip, image_input, image_elements) == 0);
    CHECK(memcmp(matrix_actual, matrix_reference, matrix_elements) == 0);
    return 0;
}

static int test_transpose_f32_native_kernel(void) {
    enum {
        batch = 2,
        height = 17,
        width = 19,
        channels = 13,
        elements = batch * height * width * channels,
    };
    const uint32_t nhwc_shape[4] = {batch, height, width, channels};
    const uint32_t nchw_shape[4] = {batch, channels, height, width};
    const uint32_t nhwc_to_nchw[4] = {0, 3, 1, 2};
    const uint32_t nchw_to_nhwc[4] = {0, 2, 3, 1};
    float input[elements];
    float reference[elements];
    float actual[elements];
    float roundtrip[elements];
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    int forward_result;
    int reverse_result;
    for (int index = 0; index < elements; index++)
        input[index] = (float)((index * 37 + index / 11) % 1021) * 0.03125f;
    CHECK(transpose_nd_f32(input, reference, nhwc_shape, nhwc_to_nchw, 4u,
                           elements) == 1);
    pool = vx_kernel_thread_pool_create(4);
    CHECK(pool != NULL);
    scope = vx_kernel_thread_pool_scope_enter(pool);
    forward_result = vx_transpose_nd_f32_native_validated(
        input, actual, nhwc_shape, nhwc_to_nchw, 4u, elements);
    reverse_result = vx_transpose_nd_f32_native_validated(
        actual, roundtrip, nchw_shape, nchw_to_nhwc, 4u, elements);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    CHECK(forward_result == 1);
    CHECK(reverse_result == 1);
    CHECK(memcmp(actual, reference, sizeof(actual)) == 0);
    CHECK(memcmp(roundtrip, input, sizeof(roundtrip)) == 0);
    return 0;
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
    const int32_t minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t token = 0; token < tokens; token++) {
        uint32_t row = (uint32_t)ids[token];
        for (uint32_t column = 0; column < hidden; column++) {
            size_t index = (size_t)row * hidden + column;
            int32_t raw = weight_dtype == VX_DTYPE_I8 ?
                                               ((const int8_t*)weight)[index] :
                                               ((const uint8_t*)weight)[index];
            float dequantized = (float)(raw - zero_points[row]) * scales[row];
            float transformed = dequantized / output_scale + (float)output_zero_point;
            int32_t quantized;
            if (transformed != transformed) quantized = output_zero_point;
            else if (transformed <= (float)minimum) quantized = minimum;
            else if (transformed >= (float)maximum) quantized = maximum;
            else quantized = qembedding_round_ties_even(transformed);
            index = (size_t)token * hidden + column;
            if (output_dtype == VX_DTYPE_I8)
                ((int8_t*)output)[index] = (int8_t)quantized;
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
        const uint32_t weight_dtype = weight_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[tokens * hidden];
            int8_t expected_i8[tokens * hidden];
            uint8_t actual_u8[tokens * hidden];
            uint8_t expected_u8[tokens * hidden];
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            void* expected = output_kind ? (void*)expected_u8 : (void*)expected_i8;
            uint32_t output_dtype = output_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
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
    const int32_t minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t index = 0; index < elements; index++) {
        const int32_t raw = input_dtype == VX_DTYPE_I8 ?
                                              ((const int8_t*)input)[index] :
                                              ((const uint8_t*)input)[index];
        const float x = (float)(raw - input_zero_point) * input_scale;
        const float silu = x / (1.0f + expf(-x));
        const float transformed = silu / output_scale + (float)output_zero_point;
        int32_t quantized;
        if (transformed != transformed) quantized = output_zero_point;
        else if (transformed <= (float)minimum) quantized = minimum;
        else if (transformed >= (float)maximum) quantized = maximum;
        else quantized = qembedding_round_ties_even(transformed);
        if (output_dtype == VX_DTYPE_I8)
            ((int8_t*)output)[index] = (int8_t)quantized;
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
    const int32_t minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    for (uint32_t index = 0; index < elements; index++) {
        const int32_t raw = input_dtype == VX_DTYPE_I8 ?
                                              ((const int8_t*)input)[index] :
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
        if (output_dtype == VX_DTYPE_I8)
            ((int8_t*)output)[index] = (int8_t)quantized;
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 128 : 0;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[5] = {0};
            int8_t expected_i8[5] = {0};
            uint8_t actual_u8[5] = {0};
            uint8_t expected_u8[5] = {0};
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            void* expected = output_kind ? (void*)expected_u8 : (void*)expected_i8;
            const uint32_t output_dtype = output_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
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
                          0.25f, -3, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 128 : 0;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[5] = {0};
            uint8_t actual_u8[5] = {0};
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            const uint32_t output_dtype = output_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
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
                          0.125f, -3, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
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
            const uint32_t input_dtype = input_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t input_zero_point = input_kind ? 121 : -7;
            const float input_scale = 0.03125f + (float)input_kind * 0.00390625f +
                (float)op * 0.0009765625f;
            for (uint32_t output_kind = 0; output_kind < 2u; output_kind++) {
                const uint32_t output_dtype = output_kind ?
                    VX_DTYPE_U8 : VX_DTYPE_I8;
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
        uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        int32_t input_zero_point = input_kind ? 127 : -1;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[3] = {0};
            uint8_t actual_u8[3] = {0};
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            const void* expected = output_kind ? (const void*)expected_u8 :
                                                 (const void*)expected_i8;
            uint32_t output_dtype = output_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
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
                              1u, 0.5f, -1, 0.125f, -3, 1.0e-5f,
                              VX_DTYPE_I8, VX_DTYPE_I8) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
        CHECK(qgroupnorm_i8u8(input_i8, nonfinite_gamma, beta, alias, 1u, 1u,
                              1u, 3u, 1u, 0.5f, -1, 0.125f, -3, 1.0e-5f,
                              VX_DTYPE_I8, VX_DTYPE_I8) == 0);
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
                              1u, 0.5f, 17, 0.25f, 128, 1.0e-5f,
                              VX_DTYPE_U8, VX_DTYPE_U8) == 1);
        for (int index = 0; index < high_elements; index++)
            CHECK(output[index] == (uint8_t)(index & 1 ? 132 : 124));
    }
    return 0;
}

static int test_qgroupnorm_native_parallel_kernel(void) {
    enum {
        batch = 2,
        height = 17,
        width = 13,
        channels = 96,
        groups = 32,
        elements = batch * height * width * channels,
    };
    uint8_t input[elements];
    uint8_t reference[elements];
    uint8_t actual[elements];
    float gamma[channels];
    float beta[channels];
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    int result = 1;
    for (int index = 0; index < elements; index++)
        input[index] = (uint8_t)((index * 17 + index / 13 + 71) & 0xff);
    for (int channel = 0; channel < channels; channel++) {
        gamma[channel] = (float)((channel % 7) - 3) * 0.125f;
        beta[channel] = (float)((channel % 11) - 5) * 0.0625f;
    }
    pool = vx_kernel_thread_pool_create(4);
    CHECK(pool != NULL);
    for (uint32_t input_kind = 0; input_kind < 2u; input_kind++) {
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 123 : -5;
        for (uint32_t output_kind = 0; output_kind < 2u; output_kind++) {
            const uint32_t output_dtype = output_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t output_zero_point = output_kind ? 137 : -7;
            CHECK(qgroupnorm_i8u8(
                input, gamma, beta, reference, batch, height, width, channels,
                groups, 0.037f, input_zero_point, 0.041f,
                output_zero_point, 1.0e-5f, input_dtype,
                output_dtype) == 1);
            /* No bound pool exercises the same SIMD group-quad kernel through
             * its single-threaded route.  Keep that path byte-exact separately
             * from the parallel scheduling check below. */
            result = vx_qgroupnorm_i8u8_native_validated(
                input, gamma, beta, actual, batch, height, width, channels,
                groups, 0.037f, input_zero_point, 0.041f,
                output_zero_point, 1.0e-5f, input_dtype, output_dtype);
            CHECK(result == 1);
            CHECK(memcmp(actual, reference, elements) == 0);
            scope = vx_kernel_thread_pool_scope_enter(pool);
            result = vx_qgroupnorm_i8u8_native_validated(
                input, gamma, beta, actual, batch, height, width, channels,
                groups, 0.037f, input_zero_point, 0.041f,
                output_zero_point, 1.0e-5f, input_dtype, output_dtype);
            vx_kernel_thread_pool_scope_leave(scope);
            CHECK(result == 1);
            CHECK(memcmp(actual, reference, elements) == 0);
        }
    }
    vx_kernel_thread_pool_destroy(pool);
    return 0;
}

static int test_qsilu_native_parallel_kernel(void) {
    enum { elements = 768 * 1024 };
    uint8_t* input = (uint8_t*)malloc(elements);
    int8_t* reference = (int8_t*)malloc(elements);
    int8_t* actual = (int8_t*)malloc(elements);
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    int result;
    CHECK(input != NULL && reference != NULL && actual != NULL);
    for (int index = 0; index < elements; index++)
        input[index] = (uint8_t)((index * 29 + index / 31 + 17) & 0xff);
    CHECK(qsilu_i8u8(input, reference, elements, 0.037f, 121,
                     0.029f, -9, VX_DTYPE_U8, VX_DTYPE_I8) == 1);
    pool = vx_kernel_thread_pool_create(4);
    CHECK(pool != NULL);
    scope = vx_kernel_thread_pool_scope_enter(pool);
    result = vx_qsilu_i8u8_native_validated(
        input, actual, elements, 0.037f, 121, 0.029f, -9,
        VX_DTYPE_U8, VX_DTYPE_I8);
    vx_kernel_thread_pool_scope_leave(scope);
    vx_kernel_thread_pool_destroy(pool);
    CHECK(result == 1);
    CHECK(memcmp(actual, reference, elements) == 0);
    free(actual);
    free(reference);
    free(input);
    return 0;
}

static int test_qlayernorm_native_parallel_kernel(void) {
    enum { rows = 401, d_model = 323, elements = rows * d_model };
    uint8_t* input = (uint8_t*)malloc(elements);
    uint8_t* reference = (uint8_t*)malloc(elements);
    uint8_t* actual = (uint8_t*)malloc(elements);
    float gamma[d_model];
    float beta[d_model];
    VxKernelThreadPool* pool;
    VxKernelThreadPoolScope scope;
    int result = 1;
    CHECK(input != NULL && reference != NULL && actual != NULL);
    for (int index = 0; index < elements; index++)
        input[index] = (uint8_t)((index * 19 + index / 37 + 53) & 0xff);
    for (int channel = 0; channel < d_model; channel++) {
        gamma[channel] = (float)((channel % 17) - 8) * 0.0625f;
        beta[channel] = (float)((channel % 13) - 6) * 0.03125f;
    }
    pool = vx_kernel_thread_pool_create(4);
    CHECK(pool != NULL);
    for (uint32_t input_kind = 0; input_kind < 2u; input_kind++) {
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 119 : -9;
        for (uint32_t output_kind = 0; output_kind < 2u; output_kind++) {
            const uint32_t output_dtype = output_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t output_zero_point = output_kind ? 137 : -11;
            CHECK(qlayernorm_i8u8(
                input, gamma, beta, reference, rows, d_model, 0.03125f,
                input_zero_point, 0.0234375f, output_zero_point, 1.0e-5f,
                input_dtype, output_dtype) == 1);
            result = vx_qlayernorm_i8u8_native_validated(
                input, gamma, beta, actual, rows, d_model, 0.03125f,
                input_zero_point, 0.0234375f, output_zero_point, 1.0e-5f,
                input_dtype, output_dtype);
            CHECK(result == 1);
            CHECK(memcmp(actual, reference, elements) == 0);
            scope = vx_kernel_thread_pool_scope_enter(pool);
            result = vx_qlayernorm_i8u8_native_validated(
                input, gamma, beta, actual, rows, d_model, 0.03125f,
                input_zero_point, 0.0234375f, output_zero_point, 1.0e-5f,
                input_dtype, output_dtype);
            vx_kernel_thread_pool_scope_leave(scope);
            CHECK(result == 1);
            CHECK(memcmp(actual, reference, elements) == 0);
        }
    }
    vx_kernel_thread_pool_destroy(pool);
    free(actual);
    free(reference);
    free(input);
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
        uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        int32_t input_zero_point = input_kind ? 127 : -1;
        for (int output_kind = 0; output_kind < 2; output_kind++) {
            int8_t actual_i8[6];
            uint8_t actual_u8[6];
            void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
            const void* expected = output_kind ? (const void*)expected_u8 :
                                                 (const void*)expected_i8;
            uint32_t output_dtype = output_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
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
                              0.5f, -1, 0.125f, -3, 1.0e-5f,
                              VX_DTYPE_I8, VX_DTYPE_I8) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
        CHECK(qlayernorm_i8u8(input_i8, gamma, nonfinite_beta, alias, 2u, 3u,
                              0.5f, -1, 0.125f, -3, 1.0e-5f,
                              VX_DTYPE_I8, VX_DTYPE_I8) == 0);
        CHECK(memcmp(alias, before, sizeof(alias)) == 0);
    }
    {
        const int8_t ties_input[3] = {-3, -2, -1};
        const float zero_gamma[3] = {0.0f, 0.0f, 0.0f};
        const float ties_beta[3] = {0.5f, 1.5f, -1.5f};
        int8_t ties_output[3] = {0};
        const int8_t ties_expected[3] = {0, 2, -2};
        CHECK(qlayernorm_i8u8(ties_input, zero_gamma, ties_beta, ties_output,
                              1u, 3u, 1.0f, 0, 1.0f, 0, 1.0e-5f,
                              VX_DTYPE_I8, VX_DTYPE_I8) == 1);
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
                              0.5f, 17, 0.25f, 128, 1.0e-5f,
                              VX_DTYPE_U8, VX_DTYPE_U8) == 1);
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
        uint32_t q_dtype = q_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        int32_t q_zp = q_kind ? 127 : -1;
        for (int k_kind = 0; k_kind < 2; k_kind++) {
            const void* k = k_kind ? (const void*)k_u8 : (const void*)k_i8;
            uint32_t k_dtype = k_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
            int32_t k_zp = k_kind ? 127 : -1;
            for (int v_kind = 0; v_kind < 2; v_kind++) {
                const void* v = v_kind ? (const void*)v_u8 : (const void*)v_i8;
                uint32_t v_dtype = v_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
                int32_t v_zp = v_kind ? 127 : -1;
                for (int output_kind = 0; output_kind < 2; output_kind++) {
                    int8_t actual_i8[8] = {0};
                    uint8_t actual_u8[8] = {0};
                    void* actual = output_kind ? (void*)actual_u8 : (void*)actual_i8;
                    const void* expected = output_kind ? (const void*)expected_u8 :
                                                        (const void*)expected_i8;
                    const void* expected_causal = output_kind ?
                        (const void*)expected_causal_u8 : (const void*)expected_causal_i8;
                    uint32_t output_dtype = output_kind ?
                        VX_DTYPE_U8 : VX_DTYPE_I8;
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
                          0.5f, VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
                          VX_DTYPE_U8, 0u, 0u) == 1);
        CHECK(memcmp(output, expected, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, keep_second, output, 1u, 1u, 2u, 4u, 1u,
                          0.25f, 128, 0.5f, 120, 0.25f, 130, 0.25f, 127,
                          0.5f, VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
                          VX_DTYPE_U8, 0u, 1u) == 1);
        CHECK(memcmp(output, expected_second, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, mask_none, output, 1u, 1u, 2u, 4u, 1u,
                          0.25f, 128, 0.5f, 120, 0.25f, 130, 0.25f, 127,
                          0.5f, VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,
                          VX_DTYPE_U8, 0u, 1u) == 1);
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
                          0.5f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, 0u, 2u) == 1);
        CHECK(memcmp(output, expected_batch_key, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, query_key, output, 2u, 2u, 2u, 4u, 1u,
                          0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, 0u, 3u) == 1);
        CHECK(memcmp(output, expected_query_key, sizeof(output)) == 0);
        CHECK(qsdpa_i8u8(q, k, v, batch_query_key, output, 2u, 2u, 2u, 4u, 1u,
                          0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, 0u, 4u) == 1);
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
                          1.0f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, 0u, 0u) == 1);
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
                          0.25f, 0, 1.0f, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, VX_DTYPE_I8, 0u, 0u) == 1);
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
                          0.25f, 0, 0.5f, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, VX_DTYPE_I8, 0u, 0u) == 0);
        CHECK(memcmp(overlap, before, sizeof(overlap)) == 0);
        CHECK(qsdpa_i8u8(q_i8, k_i8, v_i8, mask_overlap,
                          (unsigned char*)mask_overlap + sizeof(int32_t), 1u, 2u,
                          2u, 4u, 1u, 0.25f, -1, 0.25f, -1, 0.25f, -1,
                          0.25f, 0, 0.5f, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, VX_DTYPE_I8, 0u, 1u) == 0);
        CHECK(memcmp(mask_overlap, mask_before, sizeof(mask_overlap)) == 0);
        CHECK(qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, sentinel, 1u, 2u, 2u, 2u,
                          1u, 0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, 0u, 0u) == 0);
        CHECK(qsdpa_i8u8(q_i8, k_i8, v_i8, NULL, sentinel, 1u, 2u, 2u, 4u,
                          1u, 1.0e30f, -1, 1.0e30f, -1, 0.25f, -1, 0.25f, 0,
                          0.5f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                          VX_DTYPE_I8, 0u, 0u) == 0);
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

    CHECK(qargmax_i8u8(input_i8, output_i8, 2u, 3u, 2u,
                       VX_DTYPE_I8) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    /* U8 with a nonzero asymmetric zero point still compares raw U8 ordering. */
    CHECK(qargmax_i8u8(input_u8, output_u8, 1u, 3u, 2u,
                       VX_DTYPE_U8) == 1);
    CHECK(memcmp(output_u8, expected_u8, sizeof(output_u8)) == 0);
    CHECK(qargmax_i8u8(last_axis_i8, output_last_axis, 2u, 4u, 1u,
                       VX_DTYPE_I8) == 1);
    CHECK(memcmp(output_last_axis, expected_last_axis, sizeof(output_last_axis)) == 0);

    memset(alias.bytes, 0x5a, sizeof(alias.bytes));
    memcpy(alias.bytes, input_i8, 6u);
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    CHECK(qargmax_i8u8(alias.bytes, alias.aligned, 1u, 3u, 2u,
                       VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias.bytes)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, 0u, 3u, 2u,
                       VX_DTYPE_I8) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, 2u, 3u, 2u,
                       VX_DTYPE_BOOL) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, 2u, 3u, 2u,
                       VX_DTYPE_F4) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qargmax_i8u8(input_i8, sentinel, UINT32_MAX, 2u, 1u,
                       VX_DTYPE_I8) == 0);
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
                            0.25f, -3, 0.5f, 5,
                            VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(memcmp(output_i8, expected_i8, sizeof(output_i8)) == 0);
    CHECK(qmaskedmean_i8u8(input_u8, mask_u8, output_u8, 1u, 3u, 2u,
                            0.25f, 128, 0.25f, 130,
                            VX_DTYPE_U8, VX_DTYPE_U8) == 1);
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
                            VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(staged_output == -62);

    memcpy(alias.input, input_i8, sizeof(input_i8));
    memcpy(alias_before, alias.bytes, sizeof(alias_before));
    CHECK(qmaskedmean_i8u8(alias.input, mask_i8, alias.bytes, 2u, 3u, 4u,
                            0.25f, -3, 0.5f, 5,
                            VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(alias.bytes, alias_before, sizeof(alias_before)) == 0);
    memcpy(mask_alias.mask, mask_i8, sizeof(mask_i8));
    memcpy(mask_before, mask_alias.bytes, sizeof(mask_before));
    CHECK(qmaskedmean_i8u8(input_i8, mask_alias.mask, mask_alias.bytes,
                            2u, 3u, 4u, 0.25f, -3, 0.5f, 5,
                            VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(mask_alias.bytes, mask_before, sizeof(mask_before)) == 0);
    CHECK(qmaskedmean_i8u8(input_i8, mask_i8, sentinel, 2u, 0u, 4u,
                            0.25f, -3, 0.5f, 5,
                            VX_DTYPE_I8, VX_DTYPE_I8) == 0);
    CHECK(memcmp(sentinel, sentinel_before, sizeof(sentinel)) == 0);
    CHECK(qmaskedmean_i8u8(input_i8, mask_i8, sentinel, 1u,
                            (uint32_t)INT32_MAX / 255u + 1u, 1u,
                            0.25f, -128, 0.5f, 5,
                            VX_DTYPE_I8, VX_DTYPE_I8) == 0);
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
 * plus scalar tail, two AVX-VNNI blocks plus tail, or the exact AVX2
 * PMADDUBSW activation split plus its tail according to the runtime gate. */
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zp = input_kind ? 119 : -13;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
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
                const uint32_t output_dtype = output_kind ?
                    VX_DTYPE_U8 : VX_DTYPE_I8;
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
        enum { saturation_rows = 4, saturation_k = 32, saturation_n = 2 };
        int8_t saturation_input[saturation_rows * saturation_k];
        int8_t saturation_weight[saturation_n * saturation_k];
        const int32_t saturation_bias[saturation_n] = {0, 0};
        const float saturation_scales[saturation_n] = {0.001f, 0.001f};
        const int32_t saturation_zero_points[saturation_n] = {0, 0};
        int8_t saturation_reference[saturation_rows * saturation_n] = {0};
        int8_t saturation_actual[saturation_rows * saturation_n] = {0};
        /* Adjacent raw products are +64770 and -65280. A direct
         * PMADDUBSW would saturate both I16 lanes; the split kernel must
         * nevertheless reproduce the portable centered I32 dot exactly. */
        memset(saturation_input, 127, sizeof(saturation_input));
        memset(saturation_weight, 127, saturation_k);
        memset(saturation_weight + saturation_k, -128, saturation_k);
        CHECK(qlinear_i8u8(
                  saturation_input, saturation_weight, saturation_bias,
                  saturation_scales, saturation_zero_points,
                  saturation_reference, saturation_rows, saturation_k,
                  saturation_n, 0.001f, -128, 0.1f, 0,
                  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
        CHECK(vx_qlinear_i8u8_native(
                  saturation_input, saturation_weight, saturation_bias,
                  saturation_scales, saturation_zero_points,
                  saturation_actual, saturation_rows, saturation_k,
                  saturation_n, 0.001f, -128, 0.1f, 0,
                  VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
        CHECK(memcmp(saturation_reference, saturation_actual,
                     sizeof(saturation_actual)) == 0);
        CHECK(saturation_actual[0] == 10 && saturation_actual[1] == -10);
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
                            0.0625f, -13, 0.125f, -5,
                            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
        CHECK(vx_qlinear_i8u8_native(native_alias, weight_i8, bias,
                                      weight_scales, weight_zp_i8, native_alias,
                                      rows, d_in, d_out, 0.0625f, -13,
                                      0.125f, -5, VX_DTYPE_I8,
                                      VX_DTYPE_I8, VX_DTYPE_I8) == 1);
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
                            1.0f, -128, 1.0f, 0,
                            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
        CHECK(vx_qlinear_i8u8_native(overflow_input, overflow_weight, overflow_bias,
                                      unit_scale, minus_128, &actual, 1, 8, 1,
                                      1.0f, -128, 1.0f, 0,
                                      VX_DTYPE_I8, VX_DTYPE_I8,
                                      VX_DTYPE_I8) == 0);
        CHECK(vx_qlinear_i8u8_arm_try(overflow_input, overflow_weight, overflow_bias,
                                       unit_scale, minus_128, &arm_actual, 1, 8, 1,
                                       1.0f, -128, 1.0f, 0,
                                       VX_DTYPE_I8, VX_DTYPE_I8,
                                       VX_DTYPE_I8) == 0);
        CHECK(reference == 41 && actual == 41 && arm_actual == 41);
    }
    {
        int8_t input[32] = {0};
        int8_t weight[32] = {0};
        const int32_t bias[1] = {0};
        const float tiny_scale[1] = {0x1p-149f};
        const int32_t zero_point[1] = {0};
        int8_t portable_output = 41;
        int8_t native_output = 41;
        int8_t arm_output = 41;
        CHECK(!qlinear_i8u8(input, weight, bias, tiny_scale, zero_point,
            &portable_output, 1, 32, 1, 0x1p-149f, 0, 1.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!vx_qlinear_i8u8_native(input, weight, bias, tiny_scale,
            zero_point, &native_output, 1, 32, 1, 0x1p-149f, 0, 1.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(!vx_qlinear_i8u8_arm_try(input, weight, bias, tiny_scale,
            zero_point, &arm_output, 1, 32, 1, 0x1p-149f, 0, 1.0f, 0,
            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
        CHECK(portable_output == 41 && native_output == 41 && arm_output == 41);
    }
    return 0;
}

/* A bounded decoder seed presents full 192-row dense calls. Force the
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
    int8_t row_thread_graph[d_out];
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
                       VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    vx_set_num_threads(1);
    CHECK(!vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, single_thread,
        rows, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, single_thread,
                                 rows, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, row_single,
                                 1u, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    vx_set_num_threads(4);
#if defined(__i386__) || defined(__x86_64__)
    expected_parallel = vx_kernels_thread_count() > 1 &&
        (vx_cpu_has_avx2() || vx_cpu_has_avx_vnni() ||
         vx_cpu_has_avx512_vnni());
#endif
    CHECK(vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, threaded,
        rows, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == expected_parallel);
    /* These are the runtime's packed-policy cases: M=1 is incremental, and
     * M=2 is below the native pool threshold despite having multiple rows. */
    CHECK(!vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, threaded,
        1u, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(!vx_qlinear_i8u8_native_will_parallelize(
        input, weight, bias, weight_scales, weight_zero_points, threaded,
        2u, d_in, d_out, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
        VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8));
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, threaded,
                                 rows, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    CHECK(vx_qlinear_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, row_thread_graph,
                                 1u, d_in, d_out, 1.0f / 32.0f, -7,
                                 1.0f / 16.0f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    vx_set_num_threads(0);
    CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
    CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
    CHECK(memcmp(reference, row_single, sizeof(row_single)) == 0);
    CHECK(memcmp(reference, row_thread_graph, sizeof(row_thread_graph)) == 0);
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
    const int32_t input_zero_unsigned = input_zero_point +
        (input_dtype == VX_DTYPE_I8 ? 128 : 0);
    const int32_t weight_zero_signed = weight_zero_points[column] -
        (weight_dtype == VX_DTYPE_U8 ? 128 : 0);
    int64_t input_sum = 0;
    int64_t weight_sum = 0;
    int64_t dot = 0;
    for (uint32_t dimension = 0; dimension < d_in; dimension++) {
        const int32_t input_raw = input_dtype == VX_DTYPE_I8
            ? (int32_t)((const int8_t*)input)[dimension]
            : (int32_t)((const uint8_t*)input)[dimension];
        const int32_t weight_raw = weight_dtype == VX_DTYPE_I8
            ? (int32_t)((const int8_t*)weight)[(size_t)column * d_in + dimension]
            : (int32_t)((const uint8_t*)weight)[(size_t)column * d_in + dimension];
        const int32_t input_unsigned = input_dtype == VX_DTYPE_I8 ?
            input_raw + 128 : input_raw;
        const int32_t weight_signed = weight_dtype == VX_DTYPE_U8 ?
            weight_raw - 128 : weight_raw;
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 255 : -128;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t* weight_zero_points = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (uint32_t column = 0; column < d_out; column++) {
                int64_t conventional = bias[column];
                for (uint32_t dimension = 0; dimension < d_in; dimension++) {
                    const int32_t input_raw = input_dtype == VX_DTYPE_I8
                        ? (int32_t)((const int8_t*)input)[dimension]
                        : (int32_t)((const uint8_t*)input)[dimension];
                    const int32_t weight_raw = weight_dtype == VX_DTYPE_I8
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

/* The canonical affine requantization rounds the multiply to F32 before it
 * adds the output zero point.  On an AVX-512-VNNI host, contracting these two
 * operations to FMA changes this exact fixture from the ties-to-even value 6
 * to 7.  Keep the widest native QLinear route byte-identical to the portable
 * ABI while retaining a useful check on hosts that select a narrower route. */
static int test_physical_qlinear_requantize_no_fma(void) {
    enum { d_in = 320 };
    int8_t input[d_in];
    int8_t weight[d_in];
    const int32_t bias[1] = {1672};
    const float weight_scale[1] = {1.0f / 88.0f};
    const int32_t weight_zero_point[1] = {0};
    int8_t portable_output = -1;
    int8_t native_output = -1;
    memset(input, -7, sizeof(input));
    memset(weight, 0, sizeof(weight));
    CHECK(qlinear_i8u8(input, weight, bias, weight_scale,
                       weight_zero_point, &portable_output, 1u, d_in, 1u,
                       1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                       VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vx_qlinear_i8u8_native(
              input, weight, bias, weight_scale, weight_zero_point,
              &native_output, 1u, d_in, 1u, 1.0f / 32.0f, -7,
              1.0f / 16.0f, -3, VX_DTYPE_I8, VX_DTYPE_I8,
              VX_DTYPE_I8) == 1);
    CHECK(portable_output == 6);
    CHECK(native_output == portable_output);
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zp = input_kind ? 119 : -13;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t* weight_zp = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (int output_kind = 0; output_kind < 2; output_kind++) {
                const uint32_t output_dtype = output_kind ?
                    VX_DTYPE_U8 : VX_DTYPE_I8;
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
                            1.0f, -128, 1.0f, 0,
                            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
        CHECK(vx_qconv2d_i8u8_native(overflow_input, overflow_weight, overflow_bias,
                                      unit_scale, minus_128, &actual,
                                      1, 1, 1, 16, 1, 1, 1, 1, 1, 16,
                                      1, 1, 1, 1, 0, 0, 0, 0, 1, 0,
                                      1.0f, -128, 1.0f, 0,
                                      VX_DTYPE_I8, VX_DTYPE_I8,
                                      VX_DTYPE_I8) == 0);
        CHECK(reference == 41 && actual == 41);
    }
    return 0;
}

/* Mirror the QLinear tie fixture through the direct 1x1 QConv route.  A host
 * selecting the AVX-512-VNNI target must retain the same separate F32
 * multiply/add boundaries as every portable physical W8A8 kernel. */
static int test_physical_qconv_requantize_no_fma(void) {
    enum { channels = 320 };
    int8_t input[channels];
    int8_t weight[channels];
    const int32_t bias[1] = {1672};
    const float weight_scale[1] = {1.0f / 88.0f};
    const int32_t weight_zero_point[1] = {0};
    int8_t portable_output = -1;
    int8_t native_output = -1;
    memset(input, -7, sizeof(input));
    memset(weight, 0, sizeof(weight));
    CHECK(qconv2d_i8u8(
              input, weight, bias, weight_scale, weight_zero_point,
              &portable_output, 1u, 1u, 1u, channels, 1u, 1u, 1u,
              1u, 1u, channels, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u,
              1u, 0u, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(vx_qconv2d_i8u8_native(
              input, weight, bias, weight_scale, weight_zero_point,
              &native_output, 1u, 1u, 1u, channels, 1u, 1u, 1u,
              1u, 1u, channels, 1u, 1u, 1u, 1u, 0u, 0u, 0u, 0u,
              1u, 0u, 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
              VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    CHECK(portable_output == 6);
    CHECK(native_output == portable_output);
    return 0;
}

/* The AVX2-only dense 3x3 route expands NHWC patches and reuses the exact
 * QLinear VPMADDUBSW kernel.  Cover asymmetric padding and every physical
 * I8/U8 input, weight, and output combination on both sides of the pool. */
static int test_physical_qconv_im2col_qlinear_dispatch(void) {
    enum {
        batch = 1, input_height = 5, input_width = 6, input_channels = 64,
        output_height = 5, output_width = 6, output_channels = 64,
        kernel_height = 3, kernel_width = 3,
        input_elements = batch * input_height * input_width * input_channels,
        weight_elements = output_channels * kernel_height * kernel_width *
                          input_channels,
        output_elements = batch * output_height * output_width *
                          output_channels,
    };
    int8_t input_i8[input_elements];
    uint8_t input_u8[input_elements];
    int8_t weight_i8[weight_elements];
    uint8_t weight_u8[weight_elements];
    int32_t bias[output_channels];
    float weight_scales[output_channels];
    int32_t weight_zp_i8[output_channels];
    int32_t weight_zp_u8[output_channels];
    unsigned char reference[output_elements];
    unsigned char single_thread[output_elements];
    unsigned char threaded[output_elements];
    for (int index = 0; index < input_elements; index++) {
        input_i8[index] = (int8_t)(((index * 29 + 17) % 231) - 115);
        input_u8[index] = (uint8_t)((index * 31 + 19) % 251);
    }
    for (int index = 0; index < weight_elements; index++) {
        weight_i8[index] = (int8_t)(((index * 37 + 23) % 227) - 113);
        weight_u8[index] = (uint8_t)((index * 41 + 29) % 253);
    }
    for (int channel = 0; channel < output_channels; channel++) {
        bias[channel] = channel * 43 - 1301;
        weight_scales[channel] = 1.0f / (float)(48 + channel % 7 * 8);
        weight_zp_i8[channel] = channel % 13 - 6;
        weight_zp_u8[channel] = 119 + channel % 13;
    }
    for (int input_kind = 0; input_kind < 2; input_kind++) {
        const void *input = input_kind ? (const void *)input_u8 :
                                         (const void *)input_i8;
        const uint32_t input_dtype = input_kind ? VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 123 : -7;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void *weight = weight_kind ? (const void *)weight_u8 :
                                               (const void *)weight_i8;
            const uint32_t weight_dtype = weight_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t *weight_zero_points = weight_kind ?
                weight_zp_u8 : weight_zp_i8;
            for (int output_kind = 0; output_kind < 2; output_kind++) {
                const uint32_t output_dtype = output_kind ?
                    VX_DTYPE_U8 : VX_DTYPE_I8;
                const int32_t output_zero_point = output_kind ? 129 : -3;
                CHECK(qconv2d_i8u8(input, weight, bias, weight_scales,
                    weight_zero_points, reference, batch, input_height,
                    input_width, input_channels, output_height, output_width,
                    output_channels, kernel_height, kernel_width,
                    input_channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    1.0f / 32.0f, input_zero_point, 1.0f / 16.0f,
                    output_zero_point, input_dtype, weight_dtype,
                    output_dtype) == 1);
                vx_set_num_threads(1);
                CHECK(vx_qconv2d_i8u8_native(input, weight, bias,
                    weight_scales, weight_zero_points, single_thread, batch,
                    input_height, input_width, input_channels, output_height,
                    output_width, output_channels, kernel_height, kernel_width,
                    input_channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    1.0f / 32.0f, input_zero_point, 1.0f / 16.0f,
                    output_zero_point, input_dtype, weight_dtype,
                    output_dtype) == 1);
                vx_set_num_threads(4);
                CHECK(vx_qconv2d_i8u8_native(input, weight, bias,
                    weight_scales, weight_zero_points, threaded, batch,
                    input_height, input_width, input_channels, output_height,
                    output_width, output_channels, kernel_height, kernel_width,
                    input_channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                    1.0f / 32.0f, input_zero_point, 1.0f / 16.0f,
                    output_zero_point, input_dtype, weight_dtype,
                    output_dtype) == 1);
                CHECK(memcmp(reference, single_thread,
                             sizeof(reference)) == 0);
                CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
            }
        }
    }
    {
        const uint32_t packed_d_in =
            kernel_height * kernel_width * input_channels;
        const uint32_t packed_bytes = vx_packed_q8_weight_size(
            packed_d_in, output_channels);
        void* packed = malloc(packed_bytes);
        memset(weight_zp_i8, 0, sizeof(weight_zp_i8));
        CHECK(packed != NULL && packed_bytes != 0u);
        CHECK(vx_pack_q8_weight(packed, packed_bytes, weight_i8, packed_d_in,
                                output_channels, VX_DTYPE_I8, 1u) == 1);
        CHECK(qconv2d_i8u8(input_u8, weight_i8, bias, weight_scales,
                           weight_zp_i8, reference, batch, input_height,
                           input_width, input_channels, output_height,
                           output_width, output_channels, kernel_height,
                           kernel_width, input_channels, 1, 1, 1, 1, 1, 1,
                           1, 1, 1, 0, 1.0f / 32.0f, 123,
                           1.0f / 16.0f, -3, VX_DTYPE_U8, VX_DTYPE_I8,
                           VX_DTYPE_I8) == 1);
        /* The prepacked one-thread route is where N32 VNNI consumes ordinary
         * U8 activations.  It must not inherit the signed-domain im2col policy
         * used by the multi-threaded N16 AVX2 route. */
        vx_set_num_threads(1);
        CHECK(vx_qconv2d_i8u8_native_prepacked(
                  input_u8, weight_i8, bias, weight_scales, weight_zp_i8,
                  single_thread, batch, input_height, input_width,
                  input_channels, output_height, output_width,
                  output_channels, kernel_height, kernel_width,
                  input_channels, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0,
                  1.0f / 32.0f, 123, 1.0f / 16.0f, -3,
                  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8, packed, NULL) == 1);
        CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
        vx_set_num_threads(4);
        CHECK(vx_qconv2d_i8u8_native_prepacked(
                  input_u8, weight_i8, bias, weight_scales, weight_zp_i8,
                  threaded, batch, input_height, input_width, input_channels,
                  output_height, output_width, output_channels, kernel_height,
                  kernel_width, input_channels, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                  0, 1.0f / 32.0f, 123, 1.0f / 16.0f, -3, VX_DTYPE_U8,
                  VX_DTYPE_I8, VX_DTYPE_I8, packed, NULL) == 1);
        CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);

        /* Dilation disables the contiguous 3x3-row writer and exercises the
         * generic im2col worker specializations.  The unpacked call selects
         * the plain worker; the symmetric prepack selects the remap worker on
         * AVX2.  Both one-thread and pooled writers must remain byte-exact. */
        CHECK(qconv2d_i8u8(input_u8, weight_i8, bias, weight_scales,
                           weight_zp_i8, reference, batch, input_height,
                           input_width, input_channels, output_height,
                           output_width, output_channels, kernel_height,
                           kernel_width, input_channels, 1, 1, 2, 2, 2, 2,
                           2, 2, 1, 0, 1.0f / 32.0f, 123,
                           1.0f / 16.0f, -3, VX_DTYPE_U8, VX_DTYPE_I8,
                           VX_DTYPE_I8) == 1);
        vx_set_num_threads(1);
        CHECK(vx_qconv2d_i8u8_native(
                  input_u8, weight_i8, bias, weight_scales, weight_zp_i8,
                  single_thread, batch, input_height, input_width,
                  input_channels, output_height, output_width,
                  output_channels, kernel_height, kernel_width,
                  input_channels, 1, 1, 2, 2, 2, 2, 2, 2, 1, 0,
                  1.0f / 32.0f, 123, 1.0f / 16.0f, -3,
                  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
        CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
        CHECK(vx_qconv2d_i8u8_native_prepacked(
                  input_u8, weight_i8, bias, weight_scales, weight_zp_i8,
                  single_thread, batch, input_height, input_width,
                  input_channels, output_height, output_width,
                  output_channels, kernel_height, kernel_width,
                  input_channels, 1, 1, 2, 2, 2, 2, 2, 2, 1, 0,
                  1.0f / 32.0f, 123, 1.0f / 16.0f, -3,
                  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8, packed, NULL) == 1);
        CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
        vx_set_num_threads(4);
        CHECK(vx_qconv2d_i8u8_native_prepacked(
                  input_u8, weight_i8, bias, weight_scales, weight_zp_i8,
                  threaded, batch, input_height, input_width, input_channels,
                  output_height, output_width, output_channels, kernel_height,
                  kernel_width, input_channels, 1, 1, 2, 2, 2, 2, 2, 2, 1,
                  0, 1.0f / 32.0f, 123, 1.0f / 16.0f, -3,
                  VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8, packed, NULL) == 1);
        CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
        free(packed);
    }
    vx_set_num_threads(0);
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
                       VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 1);
    vx_set_num_threads(1);
    CHECK(vx_qconv2d_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, single_thread, batch,
                                 input_height, input_width, input_channels,
                                 output_height, output_width, output_channels,
                                 kernel_height, kernel_width, input_per_group,
                                 1, 1, 1, 1, 1, 1, 1, 1, groups, 2,
                                 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    vx_set_num_threads(4);
    CHECK(vx_qconv2d_i8u8_native(input, weight, bias, weight_scales,
                                 weight_zero_points, threaded, batch,
                                 input_height, input_width, input_channels,
                                 output_height, output_width, output_channels,
                                 kernel_height, kernel_width, input_per_group,
                                 1, 1, 1, 1, 1, 1, 1, 1, groups, 2,
                                 1.0f / 32.0f, -7, 1.0f / 16.0f, -3,
                                 VX_DTYPE_I8, VX_DTYPE_I8,
                                 VX_DTYPE_I8) == 1);
    vx_set_num_threads(0);
    CHECK(memcmp(reference, single_thread, sizeof(reference)) == 0);
    CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
    return 0;
}

/* A grayscale encoder starts with a 3x3/stride-2 convolution. Its one input
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 127 : -3;
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void *weight = weight_kind ? (const void *)weight_u8 :
                                               (const void *)weight_i8;
            const uint32_t weight_dtype = weight_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t *weight_zero_points = weight_kind ? weight_zp_u8 :
                                                              weight_zp_i8;
            for (int output_kind = 0; output_kind < 2; output_kind++) {
                const uint32_t output_dtype = output_kind ?
                    VX_DTYPE_U8 : VX_DTYPE_I8;
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

                /* Same call served from the load-time transpose the runtime
                 * caches in QConv2D node metadata. It must be byte-identical
                 * to the per-call transpose it replaces. */
                {
                    const size_t packed_bytes =
                        vx_w8a8_qconv_small_c_pack_size(kernel_height,
                            kernel_width, input_channels, output_channels, 1u);
                    if (packed_bytes) {
                        void *packed = malloc(packed_bytes);
                        CHECK(packed != NULL);
                        CHECK(vx_w8a8_qconv_pack_small_c(packed, packed_bytes,
                                  weight, kernel_height, kernel_width,
                                  input_channels, output_channels, 1u) == 1);
                        memset(threaded, 0, sizeof(threaded));
                        CHECK(vx_qconv2d_i8u8_native_prepacked(input, weight,
                                  bias, weight_scales, weight_zero_points,
                                  threaded, batch, input_height, input_width,
                                  input_channels, output_height, output_width,
                                  output_channels, kernel_height, kernel_width,
                                  input_channels, 2, 2, 1, 1, 1, 1, 1, 1, 1, 2,
                                  1.0f / 32.0f, input_zero_point,
                                  1.0f / 16.0f, output_zero_point,
                                  input_dtype, weight_dtype, output_dtype,
                                  NULL, packed) == 1);
                        CHECK(memcmp(reference, threaded, sizeof(reference)) == 0);
                        free(packed);
                    }
                }
            }
        }
    }
    /* The size query is the eligibility gate the runtime allocates against:
     * wide inputs never reach the narrow-input kernel. */
    CHECK(vx_w8a8_qconv_small_c_pack_size(3u, 3u, 64u, 48u, 1u) == 0u);
    CHECK(vx_w8a8_qconv_small_c_pack_size(3u, 3u, 1u, 50u, 1u) == 0u);
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
        const uint32_t input_dtype = input_kind ?
            VX_DTYPE_U8 : VX_DTYPE_I8;
        const int32_t input_zero_point = input_kind ? 255 : -128;
        const int32_t input_zero_unsigned = input_zero_point +
            (input_dtype == VX_DTYPE_I8 ? 128 : 0);
        for (int weight_kind = 0; weight_kind < 2; weight_kind++) {
            const void* weight = weight_kind ? (const void*)weight_u8 : (const void*)weight_i8;
            const uint32_t weight_dtype = weight_kind ?
                VX_DTYPE_U8 : VX_DTYPE_I8;
            const int32_t* weight_zero_points = weight_kind ? weight_zp_u8 : weight_zp_i8;
            for (uint32_t output_y = 0; output_y < output_height; output_y++) {
                for (uint32_t output_x = 0; output_x < output_width; output_x++) {
                    for (uint32_t output_channel = 0; output_channel < output_channels;
                            output_channel++) {
                        const int32_t weight_zero_signed = weight_zero_points[output_channel] -
                            (weight_dtype == VX_DTYPE_U8 ? 128 : 0);
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
                                    const int32_t input_raw =
                                        input_dtype == VX_DTYPE_I8
                                        ? (int32_t)((const int8_t*)input)[input_index]
                                        : (int32_t)((const uint8_t*)input)[input_index];
                                    const int32_t weight_raw =
                                        weight_dtype == VX_DTYPE_I8
                                        ? (int32_t)((const int8_t*)weight)[weight_index]
                                        : (int32_t)((const uint8_t*)weight)[weight_index];
                                    const int32_t input_unsigned =
                                        input_dtype == VX_DTYPE_I8
                                        ? input_raw + 128 : input_raw;
                                    const int32_t weight_signed =
                                        weight_dtype == VX_DTYPE_U8
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
                        0.125f, 121, VX_DTYPE_U8, VX_DTYPE_I8,
                        VX_DTYPE_U8) == 1);
    {
        int arm_handled = vx_qconv2d_i8u8_arm_try(
            input, weight, bias, weight_scales, weight_zero_points, arm_actual,
            batch, input_height, input_width, input_channels, output_height,
            output_width, output_channels, kernel_height, kernel_width,
            input_per_group, 1u, 2u, 2u, 1u, 1u, 1u, 1u, 0u, groups, 2u,
            0.0625f, 127, 0.125f, 121, VX_DTYPE_U8, VX_DTYPE_I8,
            VX_DTYPE_U8);
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
                            1.0f, -128, 1.0f, 0,
                            VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8) == 0);
        CHECK(vx_qconv2d_i8u8_arm_try(overflow_input, overflow_weight,
                                       overflow_bias, unit_scale, minus_128,
                                       &arm_overflow, 1u, 1u, 1u, 8u, 1u,
                                       1u, 1u, 1u, 1u, 8u, 1u, 1u, 1u, 1u,
                                       0u, 0u, 0u, 0u, 1u, 0u, 1.0f, -128,
                                       1.0f, 0, VX_DTYPE_I8, VX_DTYPE_I8,
                                       VX_DTYPE_I8) == 0);
        CHECK(reference_overflow == 51 && arm_overflow == 51);
    }
    return 0;
}

static int test_physical_qlinear_i8_to_u8(void) {
    const char* graph_path = "/tmp/volvox-physical-qlinear-i8-u8-graph.json";
    const char* saved_graph_path = "/tmp/volvox-physical-qlinear-i8-u8-saved.json";
    const char* weights_path = "/tmp/volvox-physical-qlinear-i8-u8-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "inear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"ou"
        "t\":[1,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\""
        ":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_"
        "affine.scale.0\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale"
        "_tensor\":\"__fixture_affine.scale.0\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"w\":{\"scheme\""
        ":\"per_axis\",\"axis\":0,\"scale_tensor\":\"__fixture_affine.scale.1\",\"zero_point_tensor\":\"__fixture_af"
        "fine.zero.2\"}}}}";
    const int weight_shape[2] = {3, 1};
    const int bias_shape[1] = {3};
    const int8_t weight[3] = {1, 3, 1};
    const int32_t bias[3] = {0, 0, 1000};
    const int8_t input[1] = {1};
    uint8_t output[3] = {0, 0, 0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    /* Place I32 bias after three I8 bytes so the runtime must not assume the
       safetensors payload has native I32 alignment. */
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8, weight_shape, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_I8 && y->dtype == T_U8 &&
          x->elem_size == 1 && y->elem_size == 1 &&
          x->quantization.valid && y->quantization.valid &&
          closef(x->quantization.scale, 1.0f) && x->quantization.zero_point == 0 &&
          closef(y->quantization.scale, 1.0f) && y->quantization.zero_point == 128);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* 1 * {1,3} * 0.5 + 128 = {128.5,129.5}; ties round to even.
       The I32 bias on the last channel also verifies U8 saturation. */
    CHECK(output[0] == 128 && output[1] == 130 && output[2] == 255);
    CHECK(volvoxai_engine_save_graph(saved_graph_path) == 0);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(saved_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(output[0] == 128 && output[1] == 130 && output[2] == 255);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(saved_graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qembedding_i8_to_u8(void) {
    const char* graph_path = "/tmp/volvox-physical-qembedding-i8-u8-graph.json";
    const char* saved_graph_path = "/tmp/volvox-physical-qembedding-i8-u8-saved.json";
    const char* weights_path = "/tmp/volvox-physical-qembedding-i8-u8-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"raw_ids\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"nodes\":[{\"opType\":"
        "\"Clip\",\"inputs\":{\"input\":\"raw_ids\"},\"outputs\":{\"out\":\"ids\"},\"outputs_shape\":{\"out\":[2,2]},\"outputs_dtype\":"
        "{\"out\":\"int32\"},\"params\":{\"min\":0,\"max\":2}},{\"opType\":\"QEmbedding\",\"inputs\":{\"input\":\"ids\",\"weight\":\"table\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"form"
        "at\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtu"
        "re_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"table\":{\"scheme\":\"per_axis\",\""
        "axis\":0,\"scale_tensor\":\"__fixture_affine.scale.3\",\"zero_point_tensor\":\"__fixture_affine.zero.3\"}"
        "}}}";
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
    const uint8_t expected_clipped[12] = {
        129, 125, 131, 255, 0, 148,
        128, 130, 126, 255, 0, 148,
    };
    uint8_t output[12] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 table, sizeof(table)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* raw_ids = t_find("raw_ids");
    T* ids = t_find("ids");
    T* y = t_find("y");
    CHECK(raw_ids && ids && y && raw_ids->is_graph_input && !ids->is_graph_input &&
          ids->dtype == T_I32 && y->dtype == T_U8 &&
          ids->elem_size == sizeof(int32_t) && y->elem_size == 1 &&
          y->quantization.valid && closef(y->quantization.scale, 0.25f) &&
          y->quantization.zero_point == 128 && g_qembedding_meta[1].valid &&
          g_qembedding_meta[1].vocab == 3u && g_qembedding_meta[1].hidden == 3u &&
          g_qembedding_meta[1].weight_dtype == T_I8 &&
          g_qembedding_meta[1].output_dtype == T_U8);
    CHECK(volvoxai_engine_set_input_raw("raw_ids", T_I32, ids_first, sizeof(ids_first)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_first, sizeof(output)) == 0);
    CHECK(volvoxai_engine_set_input_raw("raw_ids", T_I32, ids_second, sizeof(ids_second)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_second, sizeof(output)) == 0);
    CHECK(volvoxai_engine_set_input_raw("raw_ids", T_I32, ids_invalid, sizeof(ids_invalid)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_clipped, sizeof(output)) == 0);
    CHECK(volvoxai_engine_save_graph(saved_graph_path) == 0);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_init(saved_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("raw_ids", T_I32, ids_first, sizeof(ids_first)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_first, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(saved_graph_path);
    remove(weights_path);
    return 0;
}

/* The first graph invocation creates the I32 ID device slot. Changing IDs
 * before the second invocation must upload the new host bytes rather than
 * reuse that stale slot; this is the regression for T_I32 host-dirty tracking. */
static int test_physical_qembedding_vulkan_changed_ids(void) {
    const char* graph_path = "/tmp/volvox-physical-vulkan-qembedding-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qembedding-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"ids\":{\"shape\":[4],\"dtype\":\"int32\"}},\"nodes\":[{\"opType\":\"Q"
        "Embedding\",\"inputs\":{\"input\":\"ids\",\"weight\":\"table\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out"
        "\":[4,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":"
        "\"volvox-affine-safetensors/v1\",\"tensors\":{\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_a"
        "ffine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"table\":{\"scheme\":\"per_axis\",\"axis"
        "\":0,\"scale_tensor\":\"__fixture_affine.scale.3\",\"zero_point_tensor\":\"__fixture_affine.zero.3\"}}}}";
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
    FILE* graph_file;
    if (vk_init() != 0) {
        puts("quantized Vulkan QEmbedding ID regression skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "table", SAFETENSORS_DTYPE_I8, table_shape, 2,
                                 table, sizeof(table)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qmatmul_u8_to_i8(void) {
    const char* graph_path = "/tmp/volvox-physical-qmatmul-u8-i8-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qmatmul-u8-i8-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,2],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":\"Q"
        "MatMul\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"o"
        "ut\":[2,2]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\""
        ":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_"
        "affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale"
        "_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.4\"},\"w\":{\"scheme\""
        ":\"per_axis\",\"axis\":0,\"scale_tensor\":\"__fixture_affine.scale.5\",\"zero_point_tensor\":\"__fixture_af"
        "fine.zero.5\"}}}}";
    const int weight_shape[2] = {2, 2};
    const int bias_shape[1] = {2};
    const uint8_t weight[4] = {130, 126, 127, 129};
    const int32_t bias[2] = {1, -2};
    const uint8_t input[4] = {132, 124, 128, 128};
    int8_t output[4] = {0, 0, 0, 0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_U8, weight_shape, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && y->quantization.valid &&
          x->quantization.zero_point == 128 && y->quantization.zero_point == -3);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(output[0] == 14 && output[1] == -8 && output[2] == -2 && output[3] == -4);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qconv2d_grouped_relu6(void) {
    const char* graph_path = "/tmp/volvox-physical-qconv-grouped-graph.json";
    const char* saved_graph_path = "/tmp/volvox-physical-qconv-grouped-saved.json";
    const char* weights_path = "/tmp/volvox-physical-qconv-grouped-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,3,2],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"QConv2D\",\"inputs\":{\"x\":\"x\",\"weight\":\"w\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,3,2,"
        "2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\",\"group"
        "s\":2,\"stride\":[1,2],\"dilation\":[2,1],\"pads\":[1,1,1,0],\"relu\":2}}],\"outputs\":[\"y\"],\"quantization\""
        ":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\""
        "__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"y\":{\"scheme\":\"per_tens"
        "or\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"},\"w\""
        ":{\"scheme\":\"per_axis\",\"axis\":0,\"scale_tensor\":\"__fixture_affine.scale.6\",\"zero_point_tensor\":\"__"
        "fixture_affine.zero.7\"}}}}";
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
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_U8, weight_shape, 4,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* Nonzero input ZP makes the asymmetric padding observable; the final 147
       is the ReLU6 output-domain cap at scale=.25 and ZP=123. */
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    CHECK(volvoxai_engine_save_graph(saved_graph_path) == 0);
    volvoxai_engine_shutdown();
    CHECK(volvoxai_engine_init(saved_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(saved_graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qconv2d_i8_bias_relu(void) {
    const char* graph_path = "/tmp/volvox-physical-qconv-i8-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qconv-i8-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,2,1],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QConv2D\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":"
        "{\"out\":[1,1,2,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"relu\":1}}],\"outputs\":[\"y\"],\"quantiz"
        "ation\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_ten"
        "sor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"pe"
        "r_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.8"
        "\"},\"w\":{\"scheme\":\"per_axis\",\"axis\":0,\"scale_tensor\":\"__fixture_affine.scale.1\",\"zero_point_tenso"
        "r\":\"__fixture_affine.zero.2\"}}}}";
    const int weight_shape[4] = {3, 1, 1, 1};
    const int bias_shape[1] = {3};
    const int8_t weight[3] = {1, 1, 1};
    const int32_t bias[3] = {0, 1000, -1000};
    const int8_t input[2] = {-1, 3};
    const uint8_t expected[6] = {10, 255, 10, 12, 255, 10};
    uint8_t output[6] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    /* The three-byte I8 payload makes the following I32 tensor unaligned in
       the saved data region; QConv metadata must own an aligned copy. */
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8, weight_shape, 4,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_I32, bias_shape, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    T* b = t_find("b");
    CHECK(x && y && b && x->dtype == T_I8 && y->dtype == T_U8 && b->dtype == T_I32 &&
          x->quantization.valid && y->quantization.valid && g_qconv_meta[0].valid &&
          g_qconv_meta[0].bias != NULL && g_qconv_meta[0].relu == 1);
    CHECK(((uintptr_t)b->data % sizeof(int32_t)) != 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* Channel 0 covers nearest-even .5 ties, channel 1 saturates, and channel
       2 proves ReLU clamps to the nonzero output zero point. */
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qadd_and_requantize(void) {
    const char* graph_path = "/tmp/volvox-physical-qadd-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qadd-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[4],\"dtype\":\"int8\"},\"b\":{\"shape\":[4],\"dtype\":"
        "\"uint8\"},\"zero\":{\"shape\":[4],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"a\",\"b\":\""
        "b\"},\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":"
        "{}},{\"opType\":\"RequantizeLinear\",\"inputs\":{\"input\":\"sum\"},\"outputs\":{\"out\":\"rq\"},\"outputs_shape\""
        ":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}},{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"rq\",\"b"
        "\":\"zero\"},\"outputs\":{\"out\":\"relu\"},\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"uint8\"},\""
        "params\":{\"relu\":1}},{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"rq\",\"b\":\"zero\"},\"outputs\":{\"out\":\"cap\"},\"out"
        "puts_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"relu\":2}}],\"outputs\":[\"relu\","
        "\"cap\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_ten"
        "sor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"b"
        "\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixtur"
        "e_affine.zero.1\"},\"zero\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_"
        "point_tensor\":\"__fixture_affine.zero.9\"},\"sum\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_"
        "affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.10\"},\"rq\":{\"scheme\":\"per_tensor\",\"sca"
        "le_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"relu\":{\"sc"
        "heme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affi"
        "ne.zero.9\"},\"cap\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_t"
        "ensor\":\"__fixture_affine.zero.9\"}}}}";
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
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qbatch_matmul_broadcast(void) {
    const char* graph_path = "/tmp/volvox-physical-qbatch-graph.json";
    const char* invalid_graph_path =
        "/tmp/volvox-physical-qbatch-invalid-graph.json";
    const char* weights_path =
        "/tmp/volvox-physical-qbatch-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[2,1,2,2],\"dtype\":\"uint8\"},\"b\":{\"shape\":[3,2,"
        "2],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QBatchMatMul\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out"
        "\":\"y\"},\"outputs_shape\":{\"out\":[2,3,2,2]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":"
        "[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_tens"
        "or\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"b\""
        ":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture"
        "_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_poi"
        "nt_tensor\":\"__fixture_affine.zero.10\"}}}}";
    const char* invalid_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[2,2],\"dtype\":\"uint8\"},\"b\":{\"shape\":[2,2],\"dt"
        "ype\":\"int8\"}},\"nodes\":[{\"opType\":\"QBatchMatMul\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"y\"}"
        ",\"outputs_shape\":{\"out\":[2,2]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"transA\":true}}],\"output"
        "s\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_t"
        "ensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},"
        "\"b\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixt"
        "ure_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_"
        "point_tensor\":\"__fixture_affine.zero.10\"}}}}";
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
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    int8_t output[24] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL &&
          fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    graph_file = fopen(invalid_graph_path, "wb");
    CHECK(graph_file != NULL &&
          fwrite(invalid_graph, 1, strlen(invalid_graph), graph_file) ==
              strlen(invalid_graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(
              &file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(
              &file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
              dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* ta = t_find("a");
    T* tb = t_find("b");
    T* ty = t_find("y");
    CHECK(ta && tb && ty && ta->dtype == T_U8 && tb->dtype == T_I8 &&
          ty->dtype == T_I8 && ta->quantization.valid &&
          tb->quantization.valid && ty->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw(
              "a", T_U8, a, sizeof(a)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "b", T_I8, b, sizeof(b)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();

    if (vk_init() == 0) {
        memset(output, 0, sizeof(output));
        g_use_vulkan = 1;
        CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
        ty = t_find("y");
        CHECK(ty && ty->dtype == T_I8);
        CHECK(volvoxai_engine_set_input_raw(
                  "a", T_U8, a, sizeof(a)) == 0);
        CHECK(volvoxai_engine_set_input_raw(
                  "b", T_I8, b, sizeof(b)) == 0);
        CHECK(volvoxai_engine_forward() == 0);
        /* The physical route keeps its output device-authoritative until the
           public copy boundary. */
        CHECK(((const int8_t*)ty->data)[0] == 0);
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "y", output, sizeof(output)) == 0);
        CHECK(memcmp(output, expected, sizeof(output)) == 0);
        volvoxai_engine_shutdown();
        g_use_vulkan = 0;
        vk_cleanup();
    } else {
        puts("quantized Vulkan QBatchMatMul skipped: no Vulkan compute device");
    }

    CHECK(volvoxai_engine_init(invalid_graph_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(invalid_graph_path);
    remove(weights_path);
    return 0;
}

/* Static byte constants are model weights, not graph inputs.  Their
 * per-tensor mapping still has to reach QAdd; this mirrors quantized position
 * and type embeddings in a fixed-shape W8A8 encoder-decoder package. */
static int test_physical_qadd_static_weight_quantization(void) {
    const char* graph_path = "/tmp/volvox-physical-qadd-static-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qadd-static-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[4],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QAdd"
        "\",\"inputs\":{\"a\":\"x\",\"b\":\"constant\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[4]},\"outputs_d"
        "type\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safet"
        "ensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero"
        "_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_a"
        "ffine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"constant\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"}}}}";
    const int shape[1] = {4};
    const int8_t constant[4] = {1, 2, -3, 4};
    const int8_t input[4] = {1, 2, 3, -4};
    const int8_t expected[4] = {2, 4, 0, 0};
    int8_t output[4] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "constant", SAFETENSORS_DTYPE_I8, shape, 1,
                                 constant, sizeof(constant)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* constant_tensor = t_find("constant");
    CHECK(constant_tensor && constant_tensor->dtype == T_I8 && constant_tensor->quantization.valid &&
          closef(constant_tensor->quantization.scale, 0.25f) &&
          constant_tensor->quantization.zero_point == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

/* The engine must keep both QSiLU boundaries physical bytes. A pair of
 * differently-quantized activations verifies the descriptor handoff into the
 * portable kernel. */
static int test_physical_qsilu_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-qsilu-graph.json";
    const char* invalid_graph_path = "/tmp/volvox-physical-qsilu-invalid-input-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qsilu-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QSiL"
        "U\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\""
        ":{\"out\":\"uint8\"},\"params\":{}},{\"opType\":\"QSiLU\",\"inputs\":{\"input\":\"hidden\"},\"outputs\":{\"out\":\"y\""
        "},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quan"
        "tization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_"
        "tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"hidden\":{\"sch"
        "eme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affin"
        "e.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tens"
        "or\":\"__fixture_affine.zero.12\"}}}}";
    const char* invalid_input_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QSiL"
        "U\",\"inputs\":{\"unsupported\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype"
        "\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetens"
        "ors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_po"
        "int_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affi"
        "ne.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"}}}}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[5] = {-8, -2, 0, 2, 8};
    const uint8_t expected_hidden[5] = {128, 127, 128, 131, 144};
    const int8_t expected_output[5] = {-4, -5, -4, 0, 27};
    uint8_t hidden[5] = {0};
    int8_t output[5] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    CHECK(x && h && y && x->dtype == T_I8 && h->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          h->quantization.scale == 0.25f && h->quantization.zero_point == 128 &&
          y->quantization.scale == 0.125f && y->quantization.zero_point == -4);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);

    graph_file = fopen(invalid_graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(invalid_input_graph, 1, strlen(invalid_input_graph),
                                        graph_file) == strlen(invalid_input_graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(volvoxai_engine_init(invalid_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    /* `unsupported` is deliberately not one of input/x/data. The strict
       physical route must reject it rather than treating the sole edge as a
       generic F32 SiLU or silently reading a different tensor. */
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(invalid_graph_path);
    remove(weights_path);
    return 0;
}

/* QGELU's graph spelling may omit params or state the one permitted
 * `approximate: "none"` value. Both routes retain raw activation bytes; tanh
 * plus unknown/duplicate attributes and invalid input names must fail before
 * the graph can enter an F32 GELU. */
static int test_physical_qgelu_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-qgelu-graph.json";
    const char* omitted_params_graph_path = "/tmp/volvox-physical-qgelu-omitted-params-graph.json";
    const char* tanh_graph_path = "/tmp/volvox-physical-qgelu-tanh-graph.json";
    const char* unknown_graph_path = "/tmp/volvox-physical-qgelu-unknown-graph.json";
    const char* duplicate_graph_path = "/tmp/volvox-physical-qgelu-duplicate-graph.json";
    const char* invalid_input_graph_path = "/tmp/volvox-physical-qgelu-invalid-input-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qgelu-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\""
        ":{\"out\":\"uint8\"},\"params\":{\"approximate\":\"none\"}},{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"hidden\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\""
        "outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":"
        "\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zer"
        "o.0\"},\"hidden\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tens"
        "or\":\"__fixture_affine.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale"
        ".4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* omitted_params_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"data\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"out"
        "\":\"int8\"}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\""
        "x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixtu"
        "re_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_po"
        "int_tensor\":\"__fixture_affine.zero.4\"}}}}";
    const char* invalid_tanh_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"ou"
        "t\":\"int8\"},\"params\":{\"approximate\":\"tanh\"}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-af"
        "fine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scal"
        "e.7\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"_"
        "_fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* invalid_unknown_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"ou"
        "t\":\"int8\"},\"params\":{\"approximate\":\"none\",\"unknown\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format"
        "\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture"
        "_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scal"
        "e_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* invalid_duplicate_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"ou"
        "t\":\"int8\"},\"params\":{\"approximate\":\"none\",\"approximate\":\"none\"}}],\"outputs\":[\"y\"],\"quantization\""
        ":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\""
        "__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tens"
        "or\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* invalid_input_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"unsupported\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype"
        "\":{\"out\":\"int8\"},\"params\":{\"approximate\":\"none\"}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"vol"
        "vox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affin"
        "e.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tens"
        "or\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[5] = {-8, -2, 0, 2, 8};
    const uint8_t expected_hidden[5] = {128, 127, 128, 135, 160};
    const int8_t expected_output[5] = {-4, -4, -4, 2, 28};
    const int8_t expected_omitted_output[5] = {-3, -4, -3, 4, 29};
    uint8_t hidden[5] = {0};
    int8_t output[5] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    CHECK(x && h && y && x->dtype == T_I8 && h->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          h->quantization.scale == 0.125f && h->quantization.zero_point == 128 &&
          y->quantization.scale == 0.125f && y->quantization.zero_point == -4);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);

    /* A missing params member is the graph spelling paired with the `{}`
       serialized form above. `data` is the third permitted activation alias. */
    graph_file = fopen(omitted_params_graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(omitted_params_graph, 1,
                                        strlen(omitted_params_graph), graph_file) ==
                                        strlen(omitted_params_graph));
    CHECK(fclose(graph_file) == 0);
    memset(output, 0, sizeof(output));
    CHECK(volvoxai_engine_init(omitted_params_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(output, expected_omitted_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(omitted_params_graph_path);

    graph_file = fopen(tanh_graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(invalid_tanh_graph, 1, strlen(invalid_tanh_graph),
                                        graph_file) == strlen(invalid_tanh_graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(volvoxai_engine_init(tanh_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(tanh_graph_path);

    graph_file = fopen(unknown_graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(invalid_unknown_graph, 1,
                                        strlen(invalid_unknown_graph), graph_file) ==
                                        strlen(invalid_unknown_graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(volvoxai_engine_init(unknown_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(unknown_graph_path);

    graph_file = fopen(duplicate_graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(invalid_duplicate_graph, 1,
                                        strlen(invalid_duplicate_graph), graph_file) ==
                                        strlen(invalid_duplicate_graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(volvoxai_engine_init(duplicate_graph_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(duplicate_graph_path);

    graph_file = fopen(invalid_input_graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(invalid_input_graph, 1,
                                        strlen(invalid_input_graph), graph_file) ==
                                        strlen(invalid_input_graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(volvoxai_engine_init(invalid_input_graph_path, weights_path) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(invalid_input_graph_path);
    remove(weights_path);
    return 0;
}

/* QGroupNorm keeps an I8/U8 NHWC activation island while loading only its
 * [C] gamma/beta tensors as F32.  The one-byte prefix deliberately leaves the
 * first F32 affine payload potentially unaligned in SafeTensors storage. */
static int test_physical_qgroupnorm_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-qgroupnorm-graph.json";
    const char* unknown_graph_path = "/tmp/volvox-physical-qgroupnorm-unknown-graph.json";
    const char* duplicate_graph_path = "/tmp/volvox-physical-qgroupnorm-duplicate-graph.json";
    const char* alias_graph_path = "/tmp/volvox-physical-qgroupnorm-alias-graph.json";
    const char* layout_graph_path = "/tmp/volvox-physical-qgroupnorm-layout-graph.json";
    const char* malformed_affine_graph_path = "/tmp/volvox-physical-qgroupnorm-affine-graph.json";
    const char* missing_groups_graph_path = "/tmp/volvox-physical-qgroupnorm-missing-groups-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qgroupnorm-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"hidden\"},\"outpu"
        "ts_shape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"num_groups\":1,\"eps\":null,"
        "\"data_layout\":null}},{\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\""
        "},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params"
        "\":{\"num_groups\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"te"
        "nsors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\""
        ":\"__fixture_affine.zero.11\"},\"hidden\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.sc"
        "ale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":"
        "\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* unknown_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_sh"
        "ape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"num_groups\":1,\"groups\":1}}],\"ou"
        "tputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"p"
        "er_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero."
        "11\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"_"
        "_fixture_affine.zero.12\"}}}}";
    const char* duplicate_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_sh"
        "ape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"num_groups\":1,\"num_groups\":1}}]"
        ",\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme"
        "\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.z"
        "ero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor"
        "\":\"__fixture_affine.zero.12\"}}}}";
    const char* alias_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"data\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_sha"
        "pe\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"num_groups\":1}}],\"outputs\":[\"y\"]"
        ",\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\""
        "scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"s"
        "cheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_aff"
        "ine.zero.12\"}}}}";
    const char* layout_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_sh"
        "ape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"num_groups\":1,\"data_layout\":\"NC"
        "HW\"}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\""
        "scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_af"
        "fine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_"
        "tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* malformed_affine_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"bad\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_s"
        "hape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"num_groups\":1}}],\"outputs\":[\"y"
        "\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\""
        ",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{"
        "\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_a"
        "ffine.zero.12\"}}}}";
    const char* missing_groups_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_sh"
        "ape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"eps\":1e-05}}],\"outputs\":[\"y\"],\""
        "quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"sch"
        "eme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affin"
        "e.zero.12\"}}}}";
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
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    T* g1 = t_find("g1");
    T* b1 = t_find("b1");
    CHECK(x && h && y && g1 && b1 && x->dtype == T_I8 && h->dtype == T_U8 &&
          y->dtype == T_I8 && g1->dtype == T_F32 && b1->dtype == T_F32 &&
          g1->ndim == 1 && g1->shape[0] == 3 && b1->ndim == 1 && b1->shape[0] == 3 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid &&
          h->quantization.zero_point == 128 && y->quantization.zero_point == -4);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);

    const char* invalid_paths[] = {
        unknown_graph_path, duplicate_graph_path, alias_graph_path, layout_graph_path,
        malformed_affine_graph_path, missing_groups_graph_path,
    };
    const char* invalid_graphs[] = {
        unknown_graph, duplicate_graph, alias_graph, layout_graph,
        malformed_affine_graph, missing_groups_graph,
    };
    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        graph_file = fopen(invalid_paths[index], "wb");
        CHECK(graph_file != NULL && fwrite(invalid_graphs[index], 1,
                                             strlen(invalid_graphs[index]), graph_file) ==
                                         strlen(invalid_graphs[index]));
        CHECK(fclose(graph_file) == 0);
        int init_result = volvoxai_engine_init(invalid_paths[index], weights_path);
        if (!strcmp(invalid_paths[index], duplicate_graph_path)) {
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
    const char* graph_path = "/tmp/volvox-physical-qlayernorm-graph.json";
    const char* unknown_graph_path = "/tmp/volvox-physical-qlayernorm-unknown-graph.json";
    const char* duplicate_graph_path = "/tmp/volvox-physical-qlayernorm-duplicate-graph.json";
    const char* dmodel_graph_path = "/tmp/volvox-physical-qlayernorm-dmodel-graph.json";
    const char* alias_graph_path = "/tmp/volvox-physical-qlayernorm-alias-graph.json";
    const char* malformed_affine_graph_path = "/tmp/volvox-physical-qlayernorm-affine-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qlayernorm-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"hidden\"},\"outputs_s"
        "hape\":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"eps\":null,\"d_model\":null}},{\"opTy"
        "pe\":\"QLayerNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\"},\"outputs\":{\"out\":\"y\"},\"ou"
        "tputs_shape\":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"d_model\":3}}],\"outputs\":[\"y"
        "\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\""
        ",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"hidd"
        "en\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixt"
        "ure_affine.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_p"
        "oint_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* unknown_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\""
        ":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"unknown\":1}}],\"outputs\":[\"y\"],\"quantiza"
        "tion\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tens"
        "or\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"pe"
        "r_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.1"
        "2\"}}}}";
    const char* duplicate_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\""
        ":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"eps\":1e-05,\"eps\":1e-05}}],\"outputs\":[\"y"
        "\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\""
        ",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{"
        "\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_a"
        "ffine.zero.12\"}}}}";
    const char* dmodel_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\""
        ":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"d_model\":2}}],\"outputs\":[\"y\"],\"quantiza"
        "tion\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tens"
        "or\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"pe"
        "r_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.1"
        "2\"}}}}";
    const char* alias_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"data\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":"
        "{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"form"
        "at\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtu"
        "re_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"s"
        "cale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const char* malformed_affine_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"bad\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape"
        "\":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"fo"
        "rmat\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fix"
        "ture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
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
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* h = t_find("hidden");
    T* y = t_find("y");
    CHECK(x && h && y && x->ndim == 2 && x->shape[0] == 2 && x->shape[1] == 3 &&
          x->dtype == T_I8 && h->dtype == T_U8 && y->dtype == T_I8 &&
          x->quantization.valid && h->quantization.valid && y->quantization.valid);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("hidden", hidden, sizeof(hidden)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    CHECK(memcmp(hidden, expected_hidden, sizeof(hidden)) == 0);
    CHECK(memcmp(output, expected_output, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);

    const char* invalid_paths[] = {
        unknown_graph_path, duplicate_graph_path, dmodel_graph_path,
        alias_graph_path, malformed_affine_graph_path,
    };
    const char* invalid_graphs[] = {
        unknown_graph, duplicate_graph, dmodel_graph, alias_graph,
        malformed_affine_graph,
    };
    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        graph_file = fopen(invalid_paths[index], "wb");
        CHECK(graph_file != NULL && fwrite(invalid_graphs[index], 1,
                                             strlen(invalid_graphs[index]), graph_file) ==
                                         strlen(invalid_graphs[index]));
        CHECK(fclose(graph_file) == 0);
        int init_result = volvoxai_engine_init(invalid_paths[index], weights_path);
        if (!strcmp(invalid_paths[index], duplicate_graph_path)) {
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
    const char* graph_path = "/tmp/volvox-physical-qsdpa-graph.json";
    const char* unknown_graph_path = "/tmp/volvox-physical-qsdpa-unknown-graph.json";
    const char* missing_causal_graph_path =
        "/tmp/volvox-physical-qsdpa-missing-causal-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qsdpa-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"q\":{\"shape\":[2,2,4],\"dtype\":\"int8\"},\"k\":{\"shape\":[2,2,4],"
        "\"dtype\":\"int8\"},\"v\":{\"shape\":[2,2,4],\"dtype\":\"int8\"},\"mask\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"no"
        "des\":[{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"mask\":\"mask\"},\"outputs\":{\"out\":\"y\"},\""
        "outputs_shape\":{\"out\":[2,2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"heads\":1,\"causal\":false"
        ",\"scale\":0.5}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors"
        "\":{\"q\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__f"
        "ixture_affine.zero.11\"},\"k\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"ze"
        "ro_point_tensor\":\"__fixture_affine.zero.11\"},\"v\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtur"
        "e_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"}}}}";
    const char* unknown_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"q\":{\"shape\":[2,2,4],\"dtype\":\"int8\"},\"k\":{\"shape\":[2,2,4],"
        "\"dtype\":\"int8\"},\"v\":{\"shape\":[2,2,4],\"dtype\":\"int8\"},\"mask\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"no"
        "des\":[{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"mask\":\"mask\"},\"outputs\":{\"out\":\"y\"},\""
        "outputs_shape\":{\"out\":[2,2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"heads\":1,\"causal\":false"
        ",\"unknown\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors"
        "\":{\"q\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__f"
        "ixture_affine.zero.11\"},\"k\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"ze"
        "ro_point_tensor\":\"__fixture_affine.zero.11\"},\"v\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtur"
        "e_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"}}}}";
    const char* missing_causal_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"q\":{\"shape\":[2,2,4],\"dtype\":\"int8\"},\"k\":{\"shape\":[2,2,4],"
        "\"dtype\":\"int8\"},\"v\":{\"shape\":[2,2,4],\"dtype\":\"int8\"},\"mask\":{\"shape\":[2,2],\"dtype\":\"int32\"}},\"no"
        "des\":[{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\",\"mask\":\"mask\"},\"outputs\":{\"out\":\"y\"},\""
        "outputs_shape\":{\"out\":[2,2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"heads\":1,\"scale\":0.5}}]"
        ",\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"q\":{\"scheme"
        "\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.z"
        "ero.11\"},\"k\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor"
        "\":\"__fixture_affine.zero.11\"},\"v\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale."
        "2\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__"
        "fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.0\"}}}}";
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
    FILE* graph_file;
    const char* invalid_paths[] = {
        unknown_graph_path,
        missing_causal_graph_path,
    };
    const char* invalid_graphs[] = {
        unknown_graph,
        missing_causal_graph,
    };

    memcpy(q, q_one, sizeof(q_one));
    memcpy(q + sizeof(q_one), q_one, sizeof(q_one));
    memcpy(k, k_one, sizeof(k_one));
    memcpy(k + sizeof(k_one), k_one, sizeof(k_one));
    memcpy(v, v_one, sizeof(v_one));
    memcpy(v + sizeof(v_one), v_one, sizeof(v_one));
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);

    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        graph_file = fopen(invalid_paths[index], "wb");
        CHECK(graph_file != NULL && fwrite(invalid_graphs[index], 1,
                                             strlen(invalid_graphs[index]), graph_file) ==
                                         strlen(invalid_graphs[index]));
        CHECK(fclose(graph_file) == 0);
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
    const char* graph_path = "/tmp/volvox-physical-qargmax-graph.json";
    const char* unknown_graph_path = "/tmp/volvox-physical-qargmax-unknown-graph.json";
    const char* missing_axis_graph_path =
        "/tmp/volvox-physical-qargmax-missing-axis-graph.json";
    const char* alias_graph_path = "/tmp/volvox-physical-qargmax-alias-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qargmax-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\""
        "QArgMax\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},\"outputs_dty"
        "pe\":{\"out\":\"int32\"},\"params\":{\"axis\":-2}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affi"
        "ne-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale."
        "7\",\"zero_point_tensor\":\"__fixture_affine.zero.13\"}}}}";
    const char* unknown_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\""
        "QArgMax\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},\"outputs_dty"
        "pe\":{\"out\":\"int32\"},\"params\":{\"axis\":-2,\"unknown\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":"
        "\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_a"
        "ffine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.13\"}}}}";
    const char* missing_axis_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\""
        "QArgMax\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},\"outputs_dty"
        "pe\":{\"out\":\"int32\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safete"
        "nsors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_"
        "point_tensor\":\"__fixture_affine.zero.13\"}}}}";
    const char* alias_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,2],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\""
        "QArgMax\",\"inputs\":{\"data\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,2]},\"outputs_dtyp"
        "e\":{\"out\":\"int32\"},\"params\":{\"axis\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine"
        "-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\""
        ",\"zero_point_tensor\":\"__fixture_affine.zero.13\"}}}}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[12] = {
        -5, 4, -1, 4, -1, 3,
        -128, 0, -127, 1, -126, 1,
    };
    const int32_t expected[4] = {1, 0, 2, 1};
    int32_t output[4] = {0};
    const char* invalid_paths[] = {
        unknown_graph_path,
        missing_axis_graph_path,
        alias_graph_path,
    };
    const char* invalid_graphs[] = {
        unknown_graph,
        missing_axis_graph,
        alias_graph,
    };
    SafetensorsFile file;
    FILE* graph_file;

    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);

    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        graph_file = fopen(invalid_paths[index], "wb");
        CHECK(graph_file != NULL && fwrite(invalid_graphs[index], 1,
                                             strlen(invalid_graphs[index]), graph_file) ==
                                         strlen(invalid_graphs[index]));
        CHECK(fclose(graph_file) == 0);
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
 * of the router reduction. Unknown parameters and unsupported input aliases are
 * deliberately rejected before the output buffer is touched. */
static int test_physical_qmaskedmean(void) {
    const char* graph_path = "/tmp/volvox-physical-qmaskedmean-graph.json";
    const char* unknown_graph_path =
        "/tmp/volvox-physical-qmaskedmean-unknown-graph.json";
    const char* alias_graph_path =
        "/tmp/volvox-physical-qmaskedmean-alias-graph.json";
    const char* weights_path = "/tmp/volvox-physical-qmaskedmean-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,4],\"dtype\":\"int8\"},\"mask\":{\"shape\":[2,3]"
        ",\"dtype\":\"int32\"}},\"nodes\":[{\"opType\":\"QMaskedMean\",\"inputs\":{\"input\":\"x\",\"mask\":\"mask\"},\"output"
        "s\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outpu"
        "ts\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_"
        "tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.4\"}"
        ",\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fix"
        "ture_affine.zero.14\"}}}}";
    const char* unknown_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,4],\"dtype\":\"int8\"},\"mask\":{\"shape\":[2,3]"
        ",\"dtype\":\"int32\"}},\"nodes\":[{\"opType\":\"QMaskedMean\",\"inputs\":{\"input\":\"x\",\"mask\":\"mask\"},\"output"
        "s\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"axis\":1}}"
        "],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"schem"
        "e\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine."
        "zero.4\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor"
        "\":\"__fixture_affine.zero.14\"}}}}";
    const char* alias_graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3,4],\"dtype\":\"int8\"},\"mask\":{\"shape\":[2,3]"
        ",\"dtype\":\"int32\"}},\"nodes\":[{\"opType\":\"QMaskedMean\",\"inputs\":{\"data\":\"x\",\"mask\":\"mask\"},\"outputs"
        "\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"output"
        "s\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_t"
        "ensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.4\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixt"
        "ure_affine.zero.14\"}}}}";
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
    const char* invalid_paths[] = {unknown_graph_path, alias_graph_path};
    const char* invalid_graphs[] = {unknown_graph, alias_graph};
    SafetensorsFile file;
    FILE* graph_file;

    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);

    for (size_t index = 0; index < sizeof(invalid_paths) / sizeof(invalid_paths[0]); index++) {
        graph_file = fopen(invalid_paths[index], "wb");
        CHECK(graph_file != NULL && fwrite(invalid_graphs[index], 1,
                                             strlen(invalid_graphs[index]), graph_file) ==
                                         strlen(invalid_graphs[index]));
        CHECK(fclose(graph_file) == 0);
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
    const char* graph_path = "/tmp/volvox-physical-vulkan-qgelu-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qgelu-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[5],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QGEL"
        "U\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\""
        ":{\"out\":\"uint8\"},\"params\":{\"approximate\":\"none\"}},{\"opType\":\"QGELU\",\"inputs\":{\"input\":\"hidden\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\""
        "outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":"
        "\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zer"
        "o.0\"},\"hidden\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tens"
        "or\":\"__fixture_affine.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale"
        ".4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const int8_t input[5] = {-8, -2, 0, 2, 8};
    const int8_t expected_output[5] = {-4, -4, -4, 2, 28};
    int8_t output[5] = {0};
    SafetensorsFile file;
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QGELU chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

/* The dispatcher must retain both QGroupNorm activation boundaries in Vulkan
 * packed-byte storage.  Gamma/beta are immutable F32 parameter buffers; they
 * are not a reason to materialize either activation on the host. */
static int test_physical_qgroupnorm_vulkan_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-vulkan-qgroupnorm-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qgroupnorm-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\""
        ":\"QGroupNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"hidden\"},\"outpu"
        "ts_shape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"num_groups\":1,\"eps\":1e-05"
        ",\"data_layout\":\"NHWC\"}},{\"opType\":\"QGroupNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\""
        "b2\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"par"
        "ams\":{\"num_groups\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tens"
        "or\":\"__fixture_affine.zero.11\"},\"hidden\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine"
        ".scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tenso"
        "r\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.12\"}}}}";
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
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QGroupNorm chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

/* The second QLayerNorm consumes the first node's packed U8 rows directly on
 * Vulkan. The host buffers must remain stale until the explicit raw copy. */
static int test_physical_qlayernorm_vulkan_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-vulkan-qlayernorm-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qlayernorm-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QL"
        "ayerNorm\",\"inputs\":{\"input\":\"x\",\"weight\":\"g1\",\"bias\":\"b1\"},\"outputs\":{\"out\":\"hidden\"},\"outputs_s"
        "hape\":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"eps\":1e-05,\"d_model\":3}},{\"opType"
        "\":\"QLayerNorm\",\"inputs\":{\"input\":\"hidden\",\"weight\":\"g2\",\"bias\":\"b2\"},\"outputs\":{\"out\":\"y\"},\"outp"
        "uts_shape\":{\"out\":[2,3]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantizat"
        "ion\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tenso"
        "r\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"hidden\":{\"scheme\""
        ":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.ze"
        "ro.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":"
        "\"__fixture_affine.zero.12\"}}}}";
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
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QLayerNorm chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

/* The second QSDPA consumes the first node's packed U8 output as Q directly
 * from Vulkan storage.  There is no graph-visible F32 score tensor and the
 * ordinary no-mask path must bind its valid static I32 dummy. */
static int test_physical_qsdpa_vulkan_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-vulkan-qsdpa-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qsdpa-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"q\":{\"shape\":[2,4],\"dtype\":\"int8\"},\"k\":{\"shape\":[2,4],\"dty"
        "pe\":\"int8\"},\"v\":{\"shape\":[2,4],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"q\",\"k\""
        ":\"k\",\"v\":\"v\"},\"outputs\":{\"out\":\"hidden\"},\"outputs_shape\":{\"out\":[2,4]},\"outputs_dtype\":{\"out\":\"u"
        "int8\"},\"params\":{\"heads\":1,\"causal\":false}},{\"opType\":\"QSDPA\",\"inputs\":{\"q\":\"hidden\",\"k\":\"k\",\"v\""
        ":\"v\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2,4]},\"outputs_dtype\":{\"out\":\"int8\"},\"params"
        "\":{\"heads\":1,\"causal\":false}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetenso"
        "rs/v1\",\"tensors\":{\"q\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_poi"
        "nt_tensor\":\"__fixture_affine.zero.11\"},\"k\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affi"
        "ne.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"v\":{\"scheme\":\"per_tensor\",\"scale_te"
        "nsor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.11\"},\"hidden\":{\"sche"
        "me\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine"
        ".zero.1\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tenso"
        "r\":\"__fixture_affine.zero.0\"}}}}";
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
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QSDPA chain skipped: no Vulkan compute device");
        return 0;
    }
    CHECK(qsdpa_i8u8(q, k, v, NULL, expected_hidden, 1u, 2u, 2u, 4u, 1u,
                      0.25f, -1, 0.25f, -1, 0.25f, -1, 0.25f, 128,
                      0.5f, VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,
                      VX_DTYPE_U8, 0u, 0u) == 1);
    CHECK(qsdpa_i8u8(expected_hidden, k, v, NULL, expected_output,
                      1u, 2u, 2u, 4u, 1u, 0.25f, 128, 0.25f, -1,
                      0.25f, -1, 0.25f, 0, 0.5f, VX_DTYPE_U8,
                      VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8, 0u, 0u) == 1);
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
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
    const char* graph_path = "/tmp/volvox-physical-vulkan-qargmax-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qargmax-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,3,2],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":"
        "\"QArgMax\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},\"outputs_dt"
        "ype\":{\"out\":\"int32\"},\"params\":{\"axis\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affi"
        "ne-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale."
        "7\",\"zero_point_tensor\":\"__fixture_affine.zero.15\"}}}}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const uint8_t input[6] = {130, 3, 129, 255, 130, 254};
    const int32_t expected[2] = {0, 1};
    int32_t output[2] = {0};
    SafetensorsFile file;
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QArgMax skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

/* Exercise the strict runtime GPU route, not merely the backend API.  The
 * output remains stale on the host until the public raw copy, demonstrating
 * that no F32/host activation boundary was inserted around QMaskedMean. */
static int test_physical_qmaskedmean_vulkan(void) {
    const char* graph_path = "/tmp/volvox-physical-vulkan-qmaskedmean-graph.json";
    const char* weights_path =
        "/tmp/volvox-physical-vulkan-qmaskedmean-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,3,2],\"dtype\":\"uint8\"},\"mask\":{\"shape\":[1,3"
        "],\"dtype\":\"int32\"}},\"nodes\":[{\"opType\":\"QMaskedMean\",\"inputs\":{\"input\":\"x\",\"mask\":\"mask\"},\"outpu"
        "ts\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],\"out"
        "puts\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"pe"
        "r_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.1"
        "\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__f"
        "ixture_affine.zero.16\"}}}}";
    const int one[1] = {1};
    const uint8_t dummy[1] = {0};
    const uint8_t input[6] = {130, 134, 20, 20, 131, 129};
    const int32_t mask[3] = {1, 0, 1};
    const uint8_t expected[2] = {132, 134};
    uint8_t output[2] = {0};
    SafetensorsFile file;
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan QMaskedMean skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "dummy", SAFETENSORS_DTYPE_U8, one, 1,
                                 dummy, sizeof(dummy)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_qconv_qadd_requantize_vulkan_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-vulkan-qconv-qadd-graph.json";
    const char* weights_path = "/tmp/volvox-physical-vulkan-qconv-qadd-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"},"
        "\"add\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"}},"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit_scale\","
        "\"zero_point_tensor\":\"u8_zero\"},"
        "\"add\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit_scale\","
        "\"zero_point_tensor\":\"u8_zero\"},"
        "\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"unit_scale\",\"zero_point_tensor\":\"u8_zero\"},"
        "\"conv\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit_scale\","
        "\"zero_point_tensor\":\"u8_zero\"},"
        "\"sum\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"unit_scale\","
        "\"zero_point_tensor\":\"u8_zero\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"half_scale\","
        "\"zero_point_tensor\":\"i8_zero\"}}},\"nodes\":["
        "{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\"},"
        "\"outputs\":{\"out\":\"conv\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{"
        "\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\",\"groups\":1,"
        "\"stride\":[1,1],\"dilation\":[1,1],\"pads\":[0,0,0,0]}},"
        "{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"conv\",\"b\":\"add\"},"
        "\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}},"
        "{\"opType\":\"RequantizeLinear\",\"inputs\":{\"input\":\"sum\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],\"outputs\":[\"y\"]}";
    const int weight_shape[4] = {1, 1, 1, 1};
    const int parameter_shape[1] = {1};
    const uint8_t weight[1] = {130};
    const float unit_scale[1] = {1.0f};
    const uint8_t u8_zero[1] = {128};
    const float half_scale[1] = {0.5f};
    const int8_t i8_zero[1] = {0};
    const uint8_t x[1] = {131};
    const uint8_t add[1] = {129};
    int8_t output[1] = {0};
    SafetensorsFile file;
    FILE* graph_file;

    if (vk_init() != 0) {
        puts("quantized Vulkan runtime chain skipped: no Vulkan compute device");
        return 0;
    }
    g_use_vulkan = 1;
    graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_U8, weight_shape, 4,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "unit_scale", SAFETENSORS_DTYPE_F32,
                                 parameter_shape, 1, unit_scale,
                                 sizeof(unit_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "u8_zero", SAFETENSORS_DTYPE_U8,
                                 parameter_shape, 1, u8_zero,
                                 sizeof(u8_zero)) == 0);
    CHECK(safetensors_add_tensor(&file, "half_scale", SAFETENSORS_DTYPE_F32,
                                 parameter_shape, 1, half_scale,
                                 sizeof(half_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "i8_zero", SAFETENSORS_DTYPE_I8,
                                 parameter_shape, 1, i8_zero,
                                 sizeof(i8_zero)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_dequantize_sigmoid_quantize(void) {
    const char* graph_path = "/tmp/volvox-physical-boundary-graph.json";
    const char* weights_path = "/tmp/volvox-physical-boundary-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[4],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"Dequ"
        "antizeLinear\",\"inputs\":{\"input\":\"x\",\"scale\":\"sdq\",\"zero_point\":\"zdq\"},\"outputs\":{\"out\":\"f\"},\"out"
        "puts_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"float32\"},\"params\":{}},{\"opType\":\"Sigmoid\",\"inpu"
        "ts\":{\"input\":\"f\"},\"outputs\":{\"out\":\"s\"},\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"floa"
        "t32\"},\"params\":{}},{\"opType\":\"QuantizeLinear\",\"inputs\":{\"input\":\"s\",\"scale\":\"sq\",\"zero_point\":\"z"
        "q\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[4]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{"
        "}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"sch"
        "eme\":\"per_tensor\",\"scale_tensor\":\"sdq\",\"zero_point_tensor\":\"zdq\"},\"y\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"sq\",\"zero_point_tensor\":\"zq\"}}}}";
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
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* tf = t_find("f");
    T* ty = t_find("y");
    T* scale_dq = t_find("sdq");
    T* scale_q = t_find("sq");
    CHECK(x && tf && ty && scale_dq && scale_q && x->dtype == T_I8 &&
          tf->dtype == T_F32 && ty->dtype == T_U8 && x->quantization.valid &&
          ty->quantization.valid);
    CHECK(((uintptr_t)scale_dq->data % sizeof(float)) != 0 &&
          ((uintptr_t)scale_q->data % sizeof(float)) != 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("f", f, 4) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", y, sizeof(y)) == 0);
    for (int i = 0; i < 4; i++) CHECK(closef(f[i], expected_f[i]));
    CHECK(memcmp(y, expected_y, sizeof(y)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_quantize_nan_and_saturation(void) {
    const char* graph_path = "/tmp/volvox-physical-quantize-special-graph.json";
    const char* weights_path = "/tmp/volvox-physical-quantize-special-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"f\":{\"shape\":[5],\"dtype\":\"float32\"}},\"nodes\":[{\"opType\":\"Q"
        "uantizeLinear\",\"inputs\":{\"input\":\"f\",\"scale\":\"scale\",\"zero_point\":\"zero_point\"},\"outputs\":{\"out\""
        ":\"y\"},\"outputs_shape\":{\"out\":[5]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"y\":{\"scheme\":\"per_tensor\",\"s"
        "cale_tensor\":\"scale\",\"zero_point_tensor\":\"zero_point\"}}}}";
    const int one[1] = {1};
    const uint8_t zero_point[1] = {128};
    const float scale[1] = {1.0f};
    const float input[5] = {NAN, -INFINITY, INFINITY, 0.5f, 1.5f};
    const uint8_t expected[5] = {128, 0, 255, 128, 130};
    uint8_t output[5] = {0};
    SafetensorsFile file;
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "zero_point", SAFETENSORS_DTYPE_U8, one, 1,
                                 zero_point, sizeof(zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "scale", SAFETENSORS_DTYPE_F32, one, 1,
                                 scale, sizeof(scale)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* f = t_find("f");
    T* y = t_find("y");
    T* scale_tensor = t_find("scale");
    CHECK(f && y && scale_tensor && f->dtype == T_F32 && y->dtype == T_U8 &&
          y->quantization.valid && y->quantization.zero_point == 128 &&
          ((uintptr_t)scale_tensor->data % sizeof(float)) != 0);
    CHECK(volvoxai_engine_set_input_raw("f", T_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", output, sizeof(output)) == 0);
    /* NaN maps to output zero point; infinities saturate; .5 ties round even. */
    CHECK(memcmp(output, expected, sizeof(output)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int write_quantized_test_graph(const char* path, const char* graph) {
    FILE* file = fopen(path, "wb");
    CHECK(file != NULL && fwrite(graph, 1, strlen(graph), file) == strlen(graph));
    CHECK(fclose(file) == 0);
    return 0;
}

static int test_physical_cached_bias_requires_immutable_bias(void) {
    static const char* const paths[4] = {
        "/tmp/volvox-qlinear-public-bias-rejected.json",
        "/tmp/volvox-qlinear-produced-bias-rejected.json",
        "/tmp/volvox-qconv-public-bias-rejected.json",
        "/tmp/volvox-qconv-produced-bias-rejected.json",
    };
    static const char* const graphs[4] = {
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1],\"dtype\":\"int8\"},"
        "\"b\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":[{\"opType\":\"QLinear\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"lw\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,3]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],"
        "\"outputs\":[\"y\"],\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":"
        "\"__fixture_affine.scale.0\",\"zero_point_tensor\":"
        "\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.0\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.1\"},"
        "\"lw\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"__fixture_affine.scale.1\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.2\"}}}}",

        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1],\"dtype\":\"int8\"},"
        "\"bias_source\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":[{\"opType\":\"Reshape\",\"inputs\":{"
        "\"input\":\"bias_source\"},\"outputs\":{\"out\":\"b\"},"
        "\"outputs_shape\":{\"out\":[3]},\"outputs_dtype\":{"
        "\"out\":\"int32\"},\"params\":{\"shape\":[3]}},{"
        "\"opType\":\"QLinear\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"lw\",\"bias\":\"b\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,3]},\"outputs_dtype\":{"
        "\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{\"x\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.0\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.0\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":"
        "\"__fixture_affine.scale.0\",\"zero_point_tensor\":"
        "\"__fixture_affine.zero.1\"},\"lw\":{\"scheme\":\"per_axis\","
        "\"axis\":0,\"scale_tensor\":\"__fixture_affine.scale.1\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.2\"}}}}",

        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,2,1],\"dtype\":\"int8\"},"
        "\"b\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"cw\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{"
        "\"out\":[1,1,2,3]},\"outputs_dtype\":{\"out\":\"uint8\"},"
        "\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":"
        "\"__fixture_affine.scale.2\",\"zero_point_tensor\":"
        "\"__fixture_affine.zero.0\"},\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.2\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.8\"},"
        "\"cw\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"__fixture_affine.scale.1\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.2\"}}}}",

        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1,1,2,1],\"dtype\":\"int8\"},"
        "\"bias_source\":{\"shape\":[3],\"dtype\":\"int32\"}},"
        "\"nodes\":[{\"opType\":\"Reshape\",\"inputs\":{"
        "\"input\":\"bias_source\"},\"outputs\":{\"out\":\"b\"},"
        "\"outputs_shape\":{\"out\":[3]},\"outputs_dtype\":{"
        "\"out\":\"int32\"},\"params\":{\"shape\":[3]}},{"
        "\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"cw\",\"bias\":\"b\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,2,3]},\"outputs_dtype\":{"
        "\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{\"x\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.2\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.0\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":"
        "\"__fixture_affine.scale.2\",\"zero_point_tensor\":"
        "\"__fixture_affine.zero.8\"},\"cw\":{\"scheme\":\"per_axis\","
        "\"axis\":0,\"scale_tensor\":\"__fixture_affine.scale.1\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.2\"}}}}",
    };
    const char* weights_path =
        "/tmp/volvox-physical-cached-bias-rejected.safetensors";
    const int linear_shape[2] = {3, 1};
    const int conv_shape[4] = {3, 1, 1, 1};
    const int8_t weights[3] = {1, 1, 1};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(
              &file, "lw", SAFETENSORS_DTYPE_I8, linear_shape, 2,
              weights, sizeof(weights)) == 0);
    CHECK(safetensors_add_tensor(
              &file, "cw", SAFETENSORS_DTYPE_I8, conv_shape, 4,
              weights, sizeof(weights)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);
    for (size_t index = 0; index < 4u; index++) {
        CHECK(write_quantized_test_graph(paths[index], graphs[index]) == 0);
        CHECK(volvoxai_engine_init(paths[index], weights_path) != 0);
        volvoxai_engine_shutdown();
        CHECK(remove(paths[index]) == 0);
    }
    CHECK(remove(weights_path) == 0);
    return 0;
}

static int test_physical_qbatch_matmul_prefix_scope(void) {
    const char* graph_path =
        "/tmp/volvox-physical-qbatch-prefix-scope-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"memory_a\":{\"shape\":[1,5,2],\"dtype\":\"uint8\"},"
        "\"memory_b\":{\"shape\":[1,2,2],\"dtype\":\"uint8\"},"
        "\"query_a\":{\"shape\":[1,3,2],\"dtype\":\"uint8\"},"
        "\"query_b\":{\"shape\":[1,2,2],\"dtype\":\"uint8\"}},"
        "\"nodes\":["
        "{\"opType\":\"QBatchMatMul\",\"inputs\":{\"a\":\"memory_a\","
        "\"b\":\"memory_b\"},\"outputs\":{\"out\":\"memory_y\"},"
        "\"outputs_shape\":{\"out\":[1,5,2]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}},"
        "{\"opType\":\"QBatchMatMul\",\"inputs\":{\"a\":\"query_a\","
        "\"b\":\"query_b\"},\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,3,2]},"
        "\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],"
        "\"outputs\":[\"memory_y\",\"y\"],"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"memory_a\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"},"
        "\"memory_b\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"},"
        "\"query_a\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"},"
        "\"query_b\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"},"
        "\"memory_y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"},"
        "\"y\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    const uint8_t memory_a[10] = {
        124, 122, 126, 121, 120, 127, 125, 119, 128, 123,
    };
    const uint8_t memory_b[4] = {124, 121, 126, 122};
    const uint8_t query_a[6] = {124, 122, 126, 121, 120, 127};
    const uint8_t query_b[4] = {122, 125, 124, 120};
    uint8_t expected_memory[10] = {0};
    uint8_t expected_query[6] = {0};
    uint8_t actual_memory[10] = {0};
    uint8_t actual_query[6] = {0};
    const uint8_t sentinel = 0x5a;

    CHECK(qbatch_matmul_i8u8(
              memory_a, memory_b, expected_memory, 5, 2, 2,
              0.125f, 123, 0.125f, 123, 0.125f, 123,
              VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(qbatch_matmul_i8u8(
              query_a, query_b, expected_query, 3, 2, 2,
              0.125f, 123, 0.125f, 123, 0.125f, 123,
              VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8) == 1);
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "memory_a", T_U8, memory_a, sizeof(memory_a)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "memory_b", T_U8, memory_b, sizeof(memory_b)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query_a", T_U8, query_a, sizeof(query_a)) == 0);
    CHECK(volvoxai_engine_set_input_raw(
              "query_b", T_U8, query_b, sizeof(query_b)) == 0);
    CHECK(t_find("memory_y") && t_find("y"));
    memset(t_find("memory_y")->data, sentinel, sizeof(actual_memory));
    memset(t_find("y")->data, sentinel, sizeof(actual_query));
    CHECK(volvoxai_engine_forward_prefix(2) == 0);
    CHECK(g_prefix_rows == 0 && g_prefix_row_capacity == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "memory_y", actual_memory, sizeof(actual_memory)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "y", actual_query, sizeof(actual_query)) == 0);
    CHECK(memcmp(actual_memory, expected_memory, sizeof(actual_memory)) == 0);
    CHECK(memcmp(actual_query, expected_query, 4) == 0);
    CHECK(actual_query[4] == sentinel && actual_query[5] == sentinel);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int expect_current_graph_init(const char* tag, const char* graph,
                                     int expected_result) {
    char graph_path[192];
    CHECK(tag && graph &&
          snprintf(graph_path, sizeof(graph_path),
                   "/tmp/volvox-current-v1-%s.graph.json", tag) > 0);
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    /* Parentheses bypass the positive-fixture catalog supplier. The production
     * reader sees this exact rejection document without test-side migration. */
    CHECK((volvoxai_engine_init)(graph_path, NULL) == expected_result);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int test_retired_quantization_layouts_rejected(void) {
    const char* inline_input =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"x\":{\"shape\":[1],\"dtype\":\"int8\",\"quantization\":{"
        "\"scheme\":\"per_tensor\",\"scale\":0.25,\"zero_point\":0}}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* inline_output =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]},"
        "\"outputs_quantization\":{\"out\":{\"scheme\":\"per_tensor\","
        "\"scale\":0.25,\"zero_point\":0}},\"params\":{}}],"
        "\"outputs\":[\"y\"]}";
    const char* root_weight_table =
        "{\"format\":\"volvox-graph/v1\","
        "\"weights_quantization\":{\"w\":{\"scheme\":\"per_tensor\","
        "\"scale\":0.25,\"zero_point\":0}},"
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* root_storage =
        "{\"format\":\"volvox-graph/v1\","
        "\"weights_quantization_storage\":\"safetensors_metadata\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* numeric_central_descriptor =
        "{\"format\":\"volvox-graph/v1\",\"quantization\":{"
        "\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale\":0.25,"
        "\"zero_point\":0}}},\"inputs\":{"
        "\"x\":{\"shape\":[1],\"dtype\":\"int8\"}},"
        "\"nodes\":[],\"outputs\":[\"x\"]}";
    const char* numeric_node_params =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]},"
        "\"params\":{\"output_scale\":0.25,\"output_zero_point\":0}}],"
        "\"outputs\":[\"y\"]}";
    const char* nested_numeric_node_params =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Identity\",\"inputs\":{\"input\":\"x\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1]},"
        "\"params\":{\"backend\":{\"options\":[{\"zero_point\":0}]}}}],"
        "\"outputs\":[\"y\"]}";
    const char* attention_operator_scale =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{"
        "\"q\":{\"shape\":[1,1,2],\"dtype\":\"float32\"},"
        "\"k\":{\"shape\":[1,1,2],\"dtype\":\"float32\"},"
        "\"v\":{\"shape\":[1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"CrossSDPA\","
        "\"inputs\":{\"q\":\"q\",\"k\":\"k\",\"v\":\"v\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1,2]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"heads\":1,\"causal\":false,\"scale\":0.5}}],"
        "\"outputs\":[\"y\"]}";

    CHECK(expect_current_graph_init("inline-input", inline_input, -1) == 0);
    CHECK(expect_current_graph_init("inline-output", inline_output, -1) == 0);
    CHECK(expect_current_graph_init("root-weight-table", root_weight_table, -1) == 0);
    CHECK(expect_current_graph_init("root-storage", root_storage, -1) == 0);
    CHECK(expect_current_graph_init(
              "numeric-central", numeric_central_descriptor, -1) == 0);
    CHECK(expect_current_graph_init("numeric-params", numeric_node_params, -1) == 0);
    CHECK(expect_current_graph_init(
              "nested-numeric-params", nested_numeric_node_params, -1) == 0);
    CHECK(expect_current_graph_init(
              "attention-scale", attention_operator_scale, 0) == 0);
    return 0;
}

static int test_set_input_f32_quantizes_physical_inputs(void) {
    const char* graph_path = "/tmp/volvox-set-input-f32-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"f\":{\"shape\":[6],\"dtype\":\"float32\"},\"u\":{\"shape\":[6],\"dtyp"
        "e\":\"uint8\"},\"i\":{\"shape\":[6],\"dtype\":\"int8\"},\"ids\":{\"shape\":[6],\"dtype\":\"int32\"}},\"nodes\":[],\"ou"
        "tputs\":[\"f\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"u\":{\"scheme\":\"p"
        "er_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero."
        "18\"},\"i\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"_"
        "_fixture_affine.zero.12\"}}}}";
    const float f_values[6] = {-2.0f, -0.5f, 0.0f, 0.25f, 1.0f, 3.0f};
    const float u_values[6] = {-100.0f, 0.0f, 0.25f, 0.75f, 100.0f, NAN};
    const float i_values[6] = {-100.0f, 0.0f, 0.125f, 0.375f, 100.0f, NAN};
    const uint8_t expected_u[6] = {0, 100, 100, 102, 255, 100};
    const int8_t expected_i[6] = {-128, -4, -4, -2, 127, -4};
    float copied_f[6] = {0};
    uint8_t copied_u[6] = {0};
    int8_t copied_i[6] = {0};

    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
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
    remove(graph_path);
    return 0;
}

static int test_concat_sigmoid_fusion_backend_scope(void) {
    const char* graph_path = "/tmp/volvox-concat-sigmoid-fusion-scope.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[1,1],\"dtype\":\"float32\"},\"b\":{\"shape\":[1,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},"
        "\"outputs\":{\"out\":\"joined\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"axis\":1}},{\"opType\":\"Sigmoid\","
        "\"inputs\":{\"input\":\"joined\"},\"outputs\":{\"out\":\"result\"},"
        "\"outputs_shape\":{\"out\":[1,2]},\"params\":{}}],"
        "\"outputs\":[\"result\"]}";
    const float zero = 0.0f;
    float output[2] = {0.0f, 0.0f};

    CHECK(write_quantized_test_graph(graph_path, graph) == 0);

    g_use_vulkan = 0;
    g_use_opengl = 0;
    g_use_metal = 0;
    g_use_nnapi = 0;
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
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
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(g_nn == 2 && g_concat_sigmoid_fuse[0] == -1 && !g_n[1].skip);
    g_use_metal = 0;
    volvoxai_engine_shutdown();

    remove(graph_path);
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
              closef(tensor->quantization.scale, 0.5f) &&
              tensor->quantization.zero_point == 120);
    }
    return 0;
}

static int test_physical_shape_island_chain(void) {
    const char* graph_path = "/tmp/volvox-physical-shape-chain-graph.json";
    const char* saved_graph_path = "/tmp/volvox-physical-shape-chain-saved.json";
    const char* weights_path = "/tmp/volvox-physical-shape-chain-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[1,2,2,1],\"dtype\":\"uint8\"},\"b\":{\"shape\":[1,2,"
        "2,1],\"dtype\":\"uint8\"},\"tail\":{\"shape\":[1,2],\"dtype\":\"uint8\"},\"input2\":{\"shape\":[1,1],\"dtype\":\"ui"
        "nt8\"},\"input10\":{\"shape\":[1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":\"QAdd\",\"inputs\":{\"a\":\"a\",\"b\""
        ":\"b\"},\"outputs\":{\"out\":\"sum\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},"
        "\"params\":{}},{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"sum\"},\"outputs\":{\"out\":\"pool\"},\"outputs_sh"
        "ape\":{\"out\":[1,1,1,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"kernel\":[2,2],\"stride\":[2,2],\""
        "padding\":[0,0],\"dilation\":[1,1],\"ceil_mode\":false}},{\"opType\":\"ResizeNearest2D\",\"inputs\":{\"input"
        "\":\"pool\"},\"outputs\":{\"out\":\"resize_nearest\"},\"outputs_shape\":{\"out\":[1,2,3,1]},\"outputs_dtype\":{"
        "\"out\":\"uint8\"},\"params\":{\"coordinate_transformation_mode\":\"asymmetric\",\"nearest_mode\":\"floor\"}},"
        "{\"opType\":\"Resize\",\"inputs\":{\"input\":\"resize_nearest\"},\"outputs\":{\"out\":\"resize\"},\"outputs_shape"
        "\":{\"out\":[1,3,2,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"mode\":\"nearest\",\"coordinate_trans"
        "formation_mode\":\"asymmetric\",\"nearest_mode\":\"floor\"}},{\"opType\":\"Reshape\",\"inputs\":{\"input\":\"res"
        "ize\"},\"outputs\":{\"out\":\"reshape\"},\"outputs_shape\":{\"out\":[1,1,3,2,1]},\"outputs_dtype\":{\"out\":\"ui"
        "nt8\"},\"params\":{}},{\"opType\":\"Flatten\",\"inputs\":{\"input\":\"reshape\"},\"outputs\":{\"out\":\"flatten\"},"
        "\"outputs_shape\":{\"out\":[1,6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}},{\"opType\":\"Squeeze\",\""
        "inputs\":{\"input\":\"flatten\"},\"outputs\":{\"out\":\"squeeze\"},\"outputs_shape\":{\"out\":[6]},\"outputs_dty"
        "pe\":{\"out\":\"uint8\"},\"params\":{}},{\"opType\":\"Unsqueeze\",\"inputs\":{\"input\":\"squeeze\"},\"outputs\":{\""
        "out\":\"unsqueeze\"},\"outputs_shape\":{\"out\":[1,6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}},{\"o"
        "pType\":\"Identity\",\"inputs\":{\"input\":\"unsqueeze\"},\"outputs\":{\"out\":\"identity\"},\"outputs_shape\":{\""
        "out\":[1,6]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}},{\"opType\":\"Concat\",\"inputs\":{\"input10\":"
        "\"input10\",\"z\":\"tail\",\"input2\":\"input2\",\"input\":\"identity\"},\"outputs\":{\"out\":\"out\"},\"outputs_shap"
        "e\":{\"out\":[1,10]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"axis\":-1}}],\"outputs\":[\"out\"],\"quan"
        "tization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_tensor\",\"scale_"
        "tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"b\":{\"scheme\":"
        "\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zer"
        "o.9\"},\"tail\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor"
        "\":\"__fixture_affine.zero.9\"},\"input2\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.sc"
        "ale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"input10\":{\"scheme\":\"per_tensor\",\"scale_te"
        "nsor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"sum\":{\"scheme\":"
        "\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zer"
        "o.9\"},\"pool\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor"
        "\":\"__fixture_affine.zero.9\"},\"resize_nearest\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_a"
        "ffine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"resize\":{\"scheme\":\"per_tensor\",\"s"
        "cale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"reshape\""
        ":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture"
        "_affine.zero.9\"},\"flatten\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zer"
        "o_point_tensor\":\"__fixture_affine.zero.9\"},\"squeeze\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fi"
        "xture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"unsqueeze\":{\"scheme\":\"per_"
        "tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"}"
        ",\"identity\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\""
        ":\"__fixture_affine.zero.9\"},\"out\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale."
        "7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"}}}}";
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(save_quantized_test_weights(weights_path) == 0);

    /* Disable aliases first so the portable byte copy kernel executes for every
       reshape-like edge. */
    CHECK(setenv("VOLVOX_DISABLE_OPERATOR_FUSION", "1", 1) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(g_nn == 10 && !g_n[4].skip && !g_n[5].skip && !g_n[6].skip &&
          !g_n[7].skip && !g_n[8].skip);
    CHECK(run_physical_shape_chain() == 0);
    CHECK(volvoxai_engine_save_graph(saved_graph_path) == 0);
    volvoxai_engine_shutdown();
    CHECK(unsetenv("VOLVOX_DISABLE_OPERATOR_FUSION") == 0);

    /* The normal optimizer may alias identical physical domains, and must
       still preserve the raw bytes through the same final concat. */
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(g_nn == 10 && g_n[4].skip && g_n[5].skip && g_n[6].skip &&
          g_n[7].skip && g_n[8].skip);
    CHECK(run_physical_shape_chain() == 0);
    volvoxai_engine_shutdown();

    CHECK(volvoxai_engine_init(saved_graph_path, weights_path) == 0);
    CHECK(run_physical_shape_chain() == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(saved_graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_byte_transpose(void) {
    const char* graph_path =
        "/tmp/volvox-physical-byte-transpose-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"u\":{\"shape\":[1,2,2,3],\"dtype\":\"uint8\"},\"i\":{\"shape\":[1,2,"
        "2,3],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"Transpose\",\"inputs\":{\"input\":\"u\"},\"outputs\":{\"out\":\"uo"
        "ut\"},\"outputs_shape\":{\"out\":[1,2,3,2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"perm\":[0,2,3,1"
        "]}},{\"opType\":\"Transpose\",\"inputs\":{\"input\":\"i\"},\"outputs\":{\"out\":\"iout\"},\"outputs_shape\":{\"out\""
        ":[1,2,3,2]},\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{\"perm\":[0,2,3,1]}}],\"outputs\":[\"uout\",\"iout"
        "\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"u\":{\"scheme\":\"per_tensor\""
        ",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"},\"i\":{\""
        "scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_af"
        "fine.zero.4\"},\"uout\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_poin"
        "t_tensor\":\"__fixture_affine.zero.6\"},\"iout\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_aff"
        "ine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.4\"}}}}";
    const uint8_t u[12] = {
        117, 118, 119, 120, 121, 122,
        123, 124, 125, 126, 127, 128,
    };
    const int8_t i[12] = {
        -6, -5, -4, -3, -2, -1,
        0, 1, 2, 3, 4, 5,
    };
    const uint8_t expected_u[12] = {
        117, 123, 118, 124, 119, 125,
        120, 126, 121, 127, 122, 128,
    };
    const int8_t expected_i[12] = {
        -6, 0, -5, 1, -4, 2,
        -3, 3, -2, 4, -1, 5,
    };
    uint8_t actual_u[12] = {0};
    int8_t actual_i[12] = {0};
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("u", T_U8, u, sizeof(u)) == 0);
    CHECK(volvoxai_engine_set_input_raw("i", T_I8, i, sizeof(i)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "uout", actual_u, sizeof(actual_u)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw(
              "iout", actual_i, sizeof(actual_i)) == 0);
    CHECK(memcmp(actual_u, expected_u, sizeof(actual_u)) == 0);
    CHECK(memcmp(actual_i, expected_i, sizeof(actual_i)) == 0);
    CHECK(t_find("uout")->dtype == T_U8 &&
          t_find("uout")->quantization.valid &&
          closef(t_find("uout")->quantization.scale, 0.125f) &&
          t_find("uout")->quantization.zero_point == 123);
    CHECK(t_find("iout")->dtype == T_I8 &&
          t_find("iout")->quantization.valid &&
          closef(t_find("iout")->quantization.scale, 0.25f) &&
          t_find("iout")->quantization.zero_point == -3);
    volvoxai_engine_shutdown();

    if (vk_init() == 0) {
        g_use_vulkan = 1;
        memset(actual_u, 0, sizeof(actual_u));
        memset(actual_i, 0, sizeof(actual_i));
        CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
        T* device_u = t_find("uout");
        T* device_i = t_find("iout");
        CHECK(device_u && device_i);
        memset(device_u->data, 0x5a, sizeof(actual_u));
        memset(device_i->data, 0x5a, sizeof(actual_i));
        CHECK(volvoxai_engine_set_input_raw("u", T_U8, u, sizeof(u)) == 0);
        CHECK(volvoxai_engine_set_input_raw("i", T_I8, i, sizeof(i)) == 0);
        CHECK(volvoxai_engine_forward() == 0);
        /* Strict Vulkan execution keeps the destination resident until the
           explicit result synchronization below. */
        for (size_t index = 0; index < sizeof(actual_u); index++) {
            CHECK(((const uint8_t*)device_u->data)[index] == 0x5a);
            CHECK(((const uint8_t*)device_i->data)[index] == 0x5a);
        }
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "uout", actual_u, sizeof(actual_u)) == 0);
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "iout", actual_i, sizeof(actual_i)) == 0);
        CHECK(memcmp(actual_u, expected_u, sizeof(actual_u)) == 0);
        CHECK(memcmp(actual_i, expected_i, sizeof(actual_i)) == 0);
        volvoxai_engine_shutdown();
        g_use_vulkan = 0;
        vk_cleanup();
    } else {
        puts("quantized Vulkan byte Transpose skipped: no Vulkan compute device");
    }
    remove(graph_path);
    return 0;
}

static int expect_physical_transpose_model_rejected(
        const char* tag, const char* graph) {
    char graph_path[160];
    CHECK(snprintf(graph_path, sizeof(graph_path),
                   "/tmp/volvox-physical-transpose-%s.json", tag) > 0);
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int test_physical_byte_transpose_model_rejections(void) {
    const char* duplicate_permutation =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,2,3],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Transpose\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,3,2]},\"ou"
        "tputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"perm\":[0,2,2,1]}}],\"outputs\":[\"y\"],\"quantization\":{\"form"
        "at\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtu"
        "re_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    const char* incompatible_shape =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,2,3],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Transpose\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,3,2,2]},\"ou"
        "tputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"perm\":[0,2,3,1]}}],\"outputs\":[\"y\"],\"quantization\":{\"form"
        "at\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtu"
        "re_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    const char* descriptor_change =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,2,3],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Transpose\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,3,2]},\"ou"
        "tputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"perm\":[0,2,3,1]}}],\"outputs\":[\"y\"],\"quantization\":{\"form"
        "at\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixtu"
        "re_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    const char* unsupported_parameter =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,2,2,3],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Transpose\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,3,2]},\"ou"
        "tputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"perm\":[0,2,3,1],\"axis\":1}}],\"outputs\":[\"y\"],\"quantizatio"
        "n\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\""
        ":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_te"
        "nsor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}"
        "}";
    CHECK(expect_physical_transpose_model_rejected(
              "duplicate-perm", duplicate_permutation) == 0);
    CHECK(expect_physical_transpose_model_rejected(
              "shape", incompatible_shape) == 0);
    CHECK(expect_physical_transpose_model_rejected(
              "descriptor", descriptor_change) == 0);
    CHECK(expect_physical_transpose_model_rejected(
              "parameter", unsupported_parameter) == 0);
    return 0;
}

static int test_physical_byte_expand(void) {
    const char* graph_path = "/tmp/volvox-physical-byte-expand-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,4],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\""
        ":\"Expand\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_d"
        "type\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safeten"
        "sors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_t"
        "ensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    const uint8_t input[4] = {120, 121, 122, 123};
    const uint8_t expected[12] = {
        120, 121, 122, 123,
        120, 121, 122, 123,
        120, 121, 122, 123,
    };
    const uint8_t expected_row[12] = {
        0x5a, 0x5a, 0x5a, 0x5a,
        120, 121, 122, 123,
        0x5a, 0x5a, 0x5a, 0x5a,
    };
    const uint8_t expected_prefix[12] = {
        120, 121, 122, 123,
        120, 121, 122, 123,
        0x5a, 0x5a, 0x5a, 0x5a,
    };
    const uint32_t input_shape[3] = {1, 1, 4};
    const uint32_t invalid_output_shape[3] = {1, 3, 5};
    uint8_t actual[12] = {0};
    CHECK(expand_nd_i8u8(
              input, actual, input_shape, invalid_output_shape,
              3, 3, 15) == 0);
    {
        const uint8_t blocked_input[6] = {1, 2, 3, 11, 12, 13};
        const uint8_t blocked_expected[24] = {
            1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3,
            11, 12, 13, 11, 12, 13, 11, 12, 13, 11, 12, 13,
        };
        const uint32_t blocked_input_shape[3] = {2, 1, 3};
        const uint32_t blocked_output_shape[3] = {2, 4, 3};
        uint8_t blocked_actual[24] = {0};
        CHECK(expand_nd_i8u8(
                  blocked_input, blocked_actual,
                  blocked_input_shape, blocked_output_shape,
                  3, 3, 24) == 1);
        CHECK(memcmp(blocked_actual, blocked_expected,
                     sizeof(blocked_actual)) == 0);
    }
    {
        const uint8_t right_aligned_input[2] = {7, 11};
        const uint32_t right_aligned_input_shape[2] = {2, 1};
        const uint32_t right_aligned_output_shape[4] = {3, 4, 2, 5};
        uint8_t right_aligned_actual[120] = {0};
        CHECK(expand_nd_i8u8(
                  right_aligned_input, right_aligned_actual,
                  right_aligned_input_shape, right_aligned_output_shape,
                  2, 4, 120) == 1);
        for (uint32_t outer = 0; outer < 3; outer++) {
            for (uint32_t row = 0; row < 4; row++) {
                for (uint32_t channel = 0; channel < 2; channel++) {
                    for (uint32_t lane = 0; lane < 5; lane++) {
                        const uint32_t index =
                            ((outer * 4u + row) * 2u + channel) * 5u + lane;
                        CHECK(right_aligned_actual[index] ==
                              right_aligned_input[channel]);
                    }
                }
            }
        }
    }
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) == 0);
    CHECK(volvoxai_engine_set_input_raw("x", T_U8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("y", actual, sizeof(actual)) == 0);
    CHECK(memcmp(actual, expected, sizeof(actual)) == 0);
    CHECK(t_find("y") && t_find("y")->dtype == T_U8 &&
          t_find("y")->quantization.valid &&
          closef(t_find("y")->quantization.scale, 0.125f) &&
          t_find("y")->quantization.zero_point == 123);

    /* The incremental contract accepts only an invariant [1,1,D] source.
       Selecting row one must neither offset that source nor overwrite cached
       rows zero/two. */
    {
        int saved_cache_valid = g_cache_valid;
        g_cache_valid = 0;
        CHECK(vx_runtime_node_incremental_row_compatible(&g_n[0], 0, 1) == 1);
        g_cache_valid = saved_cache_valid;
    }
    memset(t_find("y")->data, 0x5a, sizeof(actual));
    CHECK(volvoxai_engine_forward_row(1) == 0);
    memset(actual, 0, sizeof(actual));
    CHECK(volvoxai_engine_copy_tensor_raw("y", actual, sizeof(actual)) == 0);
    CHECK(memcmp(actual, expected_row, sizeof(actual)) == 0);
    memset(t_find("y")->data, 0x5a, sizeof(actual));
    CHECK(volvoxai_engine_forward_prefix(2) == 0);
    memset(actual, 0, sizeof(actual));
    CHECK(volvoxai_engine_copy_tensor_raw("y", actual, sizeof(actual)) == 0);
    CHECK(memcmp(actual, expected_prefix, sizeof(actual)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int expect_physical_expand_model_rejected(
        const char* tag, const char* graph) {
    char graph_path[160];
    CHECK(snprintf(graph_path, sizeof(graph_path),
                   "/tmp/volvox-physical-expand-%s.json", tag) > 0);
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int test_physical_byte_expand_model_rejections(void) {
    const char* incompatible_shape =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,4],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\""
        ":\"Expand\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,3,5]},\"outputs_d"
        "type\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safeten"
        "sors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_t"
        "ensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    const char* descriptor_change =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,4],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\""
        ":\"Expand\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,3,4]},\"outputs_d"
        "type\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safeten"
        "sors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.4\",\"zero_point_t"
        "ensor\":\"__fixture_affine.zero.6\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.6\"}}}}";
    CHECK(expect_physical_expand_model_rejected(
              "shape", incompatible_shape) == 0);
    CHECK(expect_physical_expand_model_rejected(
              "descriptor", descriptor_change) == 0);
    return 0;
}

static int test_physical_shape_maxpool_asymmetric_pads(void) {
    const char* graph_path = "/tmp/volvox-physical-shape-pool-graph.json";
    const char* weights_path = "/tmp/volvox-physical-shape-pool-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"u\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"},\"i\":{\"shape\":[1,1,"
        "1,1],\"dtype\":\"int8\"}},\"nodes\":[{\"opType\":\"MaxPool2D\",\"inputs\":{\"input\":\"u\"},\"outputs\":{\"out\":\"uo"
        "ut\"},\"outputs_shape\":{\"out\":[1,3,3,1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"kernel\":[2,2],"
        "\"stride\":[1,1],\"pads\":[2,1,1,2],\"dilation\":[1,1],\"ceil_mode\":false}},{\"opType\":\"MaxPool2D\",\"inpu"
        "ts\":{\"input\":\"i\"},\"outputs\":{\"out\":\"iout\"},\"outputs_shape\":{\"out\":[1,3,3,1]},\"outputs_dtype\":{\"o"
        "ut\":\"int8\"},\"params\":{\"kernel\":[2,2],\"stride\":[1,1],\"pads\":[2,1,1,2],\"dilation\":[1,1],\"ceil_mode"
        "\":false}}],\"outputs\":[\"uout\",\"iout\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"te"
        "nsors\":{\"u\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\""
        ":\"__fixture_affine.zero.19\"},\"i\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2"
        "\",\"zero_point_tensor\":\"__fixture_affine.zero.20\"},\"uout\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\""
        "__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.19\"},\"iout\":{\"scheme\":\"per_"
        "tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.20\""
        "}}}}";
    const uint8_t u[1] = {4};
    const int8_t i[1] = {-5};
    const uint8_t expected_u[9] = {0, 0, 0, 4, 4, 0, 4, 4, 0};
    const int8_t expected_i[9] = {-128, -128, -128, -5, -5, -128, -5, -5, -128};
    uint8_t actual_u[9] = {0};
    int8_t actual_i[9] = {0};
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(save_quantized_test_weights(weights_path) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* tu = t_find("u");
    T* ti = t_find("i");
    T* tuout = t_find("uout");
    T* tiout = t_find("iout");
    CHECK(tu && ti && tuout && tiout && tu->dtype == T_U8 && ti->dtype == T_I8 &&
          tuout->dtype == T_U8 && tiout->dtype == T_I8);
    CHECK(volvoxai_engine_set_input_raw("u", T_U8, u, sizeof(u)) == 0);
    CHECK(volvoxai_engine_set_input_raw("i", T_I8, i, sizeof(i)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("uout", actual_u, sizeof(actual_u)) == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("iout", actual_i, sizeof(actual_i)) == 0);
    /* [top,left,bottom,right] is deliberately asymmetric. Padded-only windows
       must use the raw dtype minimum: 0 for U8 and -128 for I8. */
    CHECK(memcmp(actual_u, expected_u, sizeof(actual_u)) == 0);
    CHECK(memcmp(actual_i, expected_i, sizeof(actual_i)) == 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int expect_physical_shape_rejected(const char* tag, const char* graph,
                                          int require_unaliased_first_node) {
    char graph_path[160];
    char weights_path[160];
    CHECK(snprintf(graph_path, sizeof(graph_path), "/tmp/volvox-physical-shape-%s-graph.json", tag) > 0);
    CHECK(snprintf(weights_path, sizeof(weights_path), "/tmp/volvox-physical-shape-%s-weights.safetensors", tag) > 0);
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(save_quantized_test_weights(weights_path) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    if (require_unaliased_first_node) CHECK(g_nn == 1 && !g_n[0].skip);
    CHECK(volvoxai_engine_forward() != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int test_physical_shape_rejections(void) {
    const char* copy_descriptor_mismatch =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[2],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":\"Res"
        "hape\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[2]},\"outputs_dtype\":{"
        "\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors"
        "/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point"
        "_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine."
        "scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"}}}}";
    const char* concat_descriptor_mismatch =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[1,1],\"dtype\":\"uint8\"},\"b\":{\"shape\":[1,1],\"dt"
        "ype\":\"uint8\"}},\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"b\":\"b\",\"a\":\"a\"},\"outputs\":{\"out\":\"y\"},\"out"
        "puts_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"axis\":1}}],\"outputs\":[\"y\"],"
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_tensor\",\"s"
        "cale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"b\":{\"sch"
        "eme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affin"
        "e.zero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tens"
        "or\":\"__fixture_affine.zero.9\"}}}}";
    const char* concat_dtype_mismatch =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[1,1],\"dtype\":\"uint8\"},\"b\":{\"shape\":[1,1],\"dt"
        "ype\":\"int8\"}},\"nodes\":[{\"opType\":\"Concat\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"y\"},\"outp"
        "uts_shape\":{\"out\":[1,2]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"axis\":1}}],\"outputs\":[\"y\"],\""
        "quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_tensor\",\"sc"
        "ale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"b\":{\"sche"
        "me\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine"
        ".zero.0\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tenso"
        "r\":\"__fixture_affine.zero.9\"}}}}";
    const char* bilinear_resize =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Resize\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outpu"
        "ts_dtype\":{\"out\":\"uint8\"},\"params\":{\"mode\":\"linear\"}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":"
        "\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_a"
        "ffine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_"
        "tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"}}}}";
    const char* unsupported_resize_transform =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Resize\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outpu"
        "ts_dtype\":{\"out\":\"uint8\"},\"params\":{\"mode\":\"nearest\",\"coordinate_transformation_mode\":\"half_pixe"
        "l\"}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"s"
        "cheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_aff"
        "ine.zero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_te"
        "nsor\":\"__fixture_affine.zero.9\"}}}}";
    const char* unsupported_short_resize_transform =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"Resize\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,2,1]},\"outpu"
        "ts_dtype\":{\"out\":\"uint8\"},\"params\":{\"mode\":\"nearest\",\"coordinate_transform_mode\":\"asymmetric\"}}]"
        ",\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme"
        "\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.z"
        "ero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\""
        ":\"__fixture_affine.zero.9\"}}}}";
    const char* pool_nonunit_dilation =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"MaxPool2D\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},\"ou"
        "tputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"kernel\":[1,1],\"dilation\":[2,1]}}],\"outputs\":[\"y\"],\"quant"
        "ization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_t"
        "ensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\"scheme\":\""
        "per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero"
        ".9\"}}}}";
    const char* pool_ceil_mode =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType"
        "\":\"MaxPool2D\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,1]},\"ou"
        "tputs_dtype\":{\"out\":\"uint8\"},\"params\":{\"kernel\":[1,1],\"ceil_mode\":true}}],\"outputs\":[\"y\"],\"quant"
        "ization\":{\"format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_t"
        "ensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\"scheme\":\""
        "per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero"
        ".9\"}}}}";
    const char* concat_bad_axis =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":\"C"
        "oncat\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1]},\"outputs_dtype"
        "\":{\"out\":\"uint8\"},\"params\":{\"axis\":2}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"volvox-affine-"
        "safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\","
        "\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixt"
        "ure_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"}}}}";
    const char* concat_sigmoid =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"uint8\"}},\"nodes\":[{\"opType\":\"C"
        "oncat\",\"inputs\":{\"input\":\"x\"},\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1]},\"outputs_dtype"
        "\":{\"out\":\"uint8\"},\"params\":{\"axis\":1,\"sigmoid\":1}}],\"outputs\":[\"y\"],\"quantization\":{\"format\":\"vo"
        "lvox-affine-safetensors/v1\",\"tensors\":{\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affi"
        "ne.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\"scheme\":\"per_tensor\",\"scale_ten"
        "sor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"}}}}";
    CHECK(expect_physical_shape_rejected("copy-descriptor", copy_descriptor_mismatch, 1) == 0);
    CHECK(expect_physical_shape_rejected("concat-descriptor", concat_descriptor_mismatch, 0) == 0);
    CHECK(expect_physical_shape_rejected("concat-dtype", concat_dtype_mismatch, 0) == 0);
    CHECK(expect_physical_shape_rejected("resize-bilinear", bilinear_resize, 0) == 0);
    CHECK(expect_physical_shape_rejected("resize-transform", unsupported_resize_transform, 0) == 0);
    CHECK(expect_physical_shape_rejected("resize-short-transform", unsupported_short_resize_transform, 0) == 0);
    CHECK(expect_physical_shape_rejected("pool-dilation", pool_nonunit_dilation, 0) == 0);
    CHECK(expect_physical_shape_rejected("pool-ceil", pool_ceil_mode, 0) == 0);
    CHECK(expect_physical_shape_rejected("concat-axis", concat_bad_axis, 0) == 0);
    CHECK(expect_physical_shape_rejected("concat-sigmoid", concat_sigmoid, 0) == 0);
    return 0;
}

static int test_generic_add_rejects_physical_bytes(void) {
    const char* graph_path = "/tmp/volvox-physical-generic-add-graph.json";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"a\":{\"shape\":[1],\"dtype\":\"uint8\"},\"b\":{\"shape\":[1],\"dtype\""
        ":\"uint8\"}},\"nodes\":[{\"opType\":\"Add\",\"inputs\":{\"a\":\"a\",\"b\":\"b\"},\"outputs\":{\"out\":\"y\"},\"outputs_sh"
        "ape\":{\"out\":[1]},\"outputs_dtype\":{\"out\":\"uint8\"},\"params\":{}}],\"outputs\":[\"y\"],\"quantization\":{\""
        "format\":\"volvox-affine-safetensors/v1\",\"tensors\":{\"a\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"__f"
        "ixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_affine.zero.1\"},\"b\":{\"scheme\":\"per_tensor\""
        ",\"scale_tensor\":\"__fixture_affine.scale.2\",\"zero_point_tensor\":\"__fixture_affine.zero.9\"},\"y\":{\""
        "scheme\":\"per_tensor\",\"scale_tensor\":\"__fixture_affine.scale.7\",\"zero_point_tensor\":\"__fixture_af"
        "fine.zero.1\"}}}}";

    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, NULL) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int test_numeric_affine_params_rejected(void) {
    const char* graph_path = "/tmp/volvox-inline-affine-params-graph.json";
    const char* weights_path = "/tmp/volvox-inline-affine-params-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"Linear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\","
        "\"weight_scale\":\"ws\",\"weight_zero_point\":\"wz\","
        "\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},"
        "\"outputs_shape\":{\"out\":[1,1]},"
        "\"outputs_dtype\":{\"out\":\"float32\"},"
        "\"params\":{\"weight_layout\":\"dout_din\","
        "\"input_scale\":0.25,\"input_zero_point\":0,"
        "\"output_scale\":0.5,\"output_zero_point\":0}}],"
        "\"outputs\":[\"y\"]}";
    const int one[1] = {1};
    const int matrix[2] = {1, 1};
    const int8_t weight[1] = {1};
    const float scale[1] = {0.25f};
    const int8_t zero_point[1] = {0};
    const float bias[1] = {0.0f};
    SafetensorsFile file;

    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", SAFETENSORS_DTYPE_I8,
                                 matrix, 2, weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "ws", SAFETENSORS_DTYPE_F32,
                                 one, 1, scale, sizeof(scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "wz", SAFETENSORS_DTYPE_I8,
                                 one, 1, zero_point, sizeof(zero_point)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", SAFETENSORS_DTYPE_F32,
                                 one, 1, bias, sizeof(bias)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}
static int test_qconv2d_weight_only_f32_activation(void) {
    const char* graph_path = "/tmp/volvox-qconv-weight-only-graph.json";
    const char* weights_path = "/tmp/volvox-qconv-weight-only-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{"
        "\"input\":\"x\",\"weight\":\"w\",\"weight_scale\":\"ws\","
        "\"weight_zero_point\":\"wz\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\",\"stride\":[1,1],"
        "\"padding\":[0,0],\"dilation\":[1,1],\"groups\":1,"
        "\"weight_only\":true}}],"
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
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL &&
          fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);
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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    T* x = t_find("x");
    T* y = t_find("y");
    CHECK(x && y && x->dtype == T_F32 && y->dtype == T_F32);
    CHECK(volvoxai_engine_set_input_raw("x", T_F32, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_f32("y", output, 2) == 0);
    CHECK(closef(output[0], expected_weight_only[0]));
    CHECK(closef(output[1], expected_weight_only[1]));

    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int expect_qconv2d_weight_only_init_rejection(
    const char* graph_path, const char* weights_path, const char* graph
) {
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int test_qconv2d_weight_only_rejects_malformed_descriptors(void) {
    const char* graph_path = "/tmp/volvox-qconv-weight-only-invalid-graph.json";
    const char* weights_path = "/tmp/volvox-qconv-weight-only-invalid-weights.safetensors";
    const char* graph_bad_scale =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"w\",\"weight_scale\":\"ws_bad\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"groups\":1,\"weight_only\":true}}],\"outputs\":[\"y\"]}";
    const char* graph_bad_bias =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"w\",\"weight_scale\":\"ws\",\"bias\":\"b_short\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,1,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"groups\":1,\"weight_only\":true}}],\"outputs\":[\"y\"]}";
    const char* graph_bad_geometry =
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,1,1,2],\"dtype\":\"float32\"}},"
        "\"nodes\":[{\"opType\":\"QConv2D\",\"inputs\":{\"input\":\"x\","
        "\"weight\":\"w\",\"weight_scale\":\"ws\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2,1,2]},"
        "\"params\":{\"data_layout\":\"NHWC\",\"weight_layout\":\"OHWI\","
        "\"groups\":1,\"weight_only\":true}}],\"outputs\":[\"y\"]}";
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
              graph_path, weights_path, graph_bad_scale) == 0);
    CHECK(expect_qconv2d_weight_only_init_rejection(
              graph_path, weights_path, graph_bad_bias) == 0);
    CHECK(expect_qconv2d_weight_only_init_rejection(
              graph_path, weights_path, graph_bad_geometry) == 0);
    remove(weights_path);
    return 0;
}

static int test_safetensors_quantization_refs_runtime_and_immutability(void) {
    const char* graph_path = "/tmp/volvox-quantization-refs-graph.json";
    const char* weights_path = "/tmp/volvox-quantization-refs-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"xl\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"xl_scale\","
        "\"zero_point_tensor\":\"xl_zero\"},"
        "\"lw\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"lw_scale\",\"zero_point_tensor\":\"lw_zero\"},"
        "\"yl\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"yl_scale\","
        "\"zero_point_tensor\":\"yl_zero\"},"
        "\"constant\":{\"scheme\":\"per_tensor\","
        "\"scale_tensor\":\"constant_scale\","
        "\"zero_point_tensor\":\"constant_zero\"}}},"
        "\"inputs\":{\"xl\":{\"shape\":[1,2],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"xl\",\"weight\":\"lw\",\"bias\":\"lb\"},"
        "\"outputs\":{\"out\":\"yl\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"yl\"]}";
    const int one[1] = {1};
    const int two[1] = {2};
    const int matrix[2] = {2, 2};
    const float input_scale[1] = {1.0f};
    const float weight_scales[2] = {0.5f, 0.25f};
    const float output_scale[1] = {0.25f};
    const float constant_scale[1] = {0.125f};
    const int8_t zero1[1] = {0};
    const int8_t zero2[2] = {0, 0};
    const int8_t weight[4] = {2, 1, -1, 3};
    const int32_t bias[2] = {0, 0};
    const int8_t constant[1] = {3};
    const int8_t input[2] = {2, -4};
    int8_t output[2] = {0};
    SafetensorsFile file;

    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "lw", VX_DTYPE_I8, matrix, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "lb", VX_DTYPE_I32, two, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "constant", VX_DTYPE_I8, one, 1,
                                 constant, sizeof(constant)) == 0);
    CHECK(safetensors_add_tensor(&file, "xl_scale", VX_DTYPE_F32, one, 1,
                                 input_scale, sizeof(input_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "xl_zero", VX_DTYPE_I8, one, 1,
                                 zero1, sizeof(zero1)) == 0);
    CHECK(safetensors_add_tensor(&file, "lw_scale", VX_DTYPE_F32, two, 1,
                                 weight_scales, sizeof(weight_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "lw_zero", VX_DTYPE_I8, two, 1,
                                 zero2, sizeof(zero2)) == 0);
    CHECK(safetensors_add_tensor(&file, "yl_scale", VX_DTYPE_F32, one, 1,
                                 output_scale, sizeof(output_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "yl_zero", VX_DTYPE_I8, one, 1,
                                 zero1, sizeof(zero1)) == 0);
    CHECK(safetensors_add_tensor(&file, "constant_scale", VX_DTYPE_F32, one, 1,
                                 constant_scale, sizeof(constant_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "constant_zero", VX_DTYPE_I8, one, 1,
                                 zero1, sizeof(zero1)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(t_find("xl") && t_find("xl")->quantization.valid &&
          closef(t_find("xl")->quantization.scale, input_scale[0]));
    CHECK(t_find("yl") && t_find("yl")->quantization.valid &&
          closef(t_find("yl")->quantization.scale, output_scale[0]));
    CHECK(t_find("constant") && t_find("constant")->quantization.valid &&
          closef(t_find("constant")->quantization.scale, constant_scale[0]));
    CHECK(g_qlinear_meta[0].valid &&
          memcmp(g_qlinear_meta[0].weight_scales, weight_scales,
                 sizeof(weight_scales)) == 0 &&
          g_qlinear_meta[0].weight_zero_points[0] == 0 &&
          g_qlinear_meta[0].weight_zero_points[1] == 0);
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
        CHECK(volvoxai_engine_tensor_info("lw_scale", &numel, shape, &ndim) != 0);
        CHECK(volvoxai_engine_tensor_info_ex(
                  "lw_scale", &numel, shape, &ndim, &dtype, &elem_size) != 0);
        CHECK(volvoxai_engine_copy_tensor_f32("lw_scale", copied, 2) != 0);
        CHECK(volvoxai_engine_copy_tensor_raw(
                  "lw_scale", copied, sizeof(copied)) != 0);
        CHECK(volvoxai_engine_set_tensor_f32("lw_scale", weight_scales, 2) != 0);
        CHECK(volvoxai_engine_set_tensor_raw(
                  "lw_scale", T_F32, weight_scales, sizeof(weight_scales)) != 0);
        CHECK(volvoxai_engine_remove_model_tensor("lw_scale") != 0);
        CHECK(t_find("lw_scale") != NULL);
        inspection = volvoxai_engine_inspect_model_json(0, 1, 0, 0);
        CHECK(inspection != NULL);
        CHECK(strstr(inspection, "\"name\":\"lw_scale\"") == NULL);
        free(inspection);
    }
    CHECK(volvoxai_engine_set_input_raw("xl", T_I8, input, sizeof(input)) == 0);
    CHECK(volvoxai_engine_forward() == 0);
    CHECK(volvoxai_engine_copy_tensor_raw("yl", output, sizeof(output)) == 0);
    CHECK(output[0] == 0 && output[1] == -14);
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}

static int write_quantization_ref_rejection_weights(const char* path) {
    const int one[1] = {1};
    const int two[1] = {2};
    const int matrix[2] = {2, 1};
    const int rank_two[2] = {1, 1};
    const int8_t weight[2] = {1, 2};
    const int32_t bias[2] = {0, 0};
    const float one_scale[1] = {1.0f};
    const float two_scales[2] = {0.5f, 0.25f};
    const float nan_scale[1] = {NAN};
    const int32_t wrong_scale[1] = {1};
    const int8_t zero_i8[2] = {0, 0};
    const uint8_t zero_u8[1] = {0};
    SafetensorsFile file;
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "w", VX_DTYPE_I8, matrix, 2,
                                 weight, sizeof(weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", VX_DTYPE_I32, two, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "scale1", VX_DTYPE_F32, one, 1,
                                 one_scale, sizeof(one_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "other_scale", VX_DTYPE_F32, one, 1,
                                 one_scale, sizeof(one_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "scales2", VX_DTYPE_F32, two, 1,
                                 two_scales, sizeof(two_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "scale_rank2", VX_DTYPE_F32, rank_two, 2,
                                 one_scale, sizeof(one_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "scale_i32", VX_DTYPE_I32, one, 1,
                                 wrong_scale, sizeof(wrong_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "scale_nan", VX_DTYPE_F32, one, 1,
                                 nan_scale, sizeof(nan_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "zero_i8_1", VX_DTYPE_I8, one, 1,
                                 zero_i8, 1) == 0);
    CHECK(safetensors_add_tensor(&file, "other_zero", VX_DTYPE_I8, one, 1,
                                 zero_i8, 1) == 0);
    CHECK(safetensors_add_tensor(&file, "zero_i8_2", VX_DTYPE_I8, two, 1,
                                 zero_i8, sizeof(zero_i8)) == 0);
    CHECK(safetensors_add_tensor(&file, "zero_u8", VX_DTYPE_U8, one, 1,
                                 zero_u8, sizeof(zero_u8)) == 0);
    CHECK(safetensors_save(path, &file) == 0);
    safetensors_free(&file);
    return 0;
}

static int expect_quantization_refs_rejected(const char* graph_path,
                                             const char* weights_path,
                                             const char* graph) {
    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(volvoxai_engine_init(graph_path, weights_path) != 0);
    volvoxai_engine_shutdown();
    remove(graph_path);
    return 0;
}

static int test_safetensors_quantization_ref_rejections(void) {
    const char* graph_path = "/tmp/volvox-quantization-ref-reject-graph.json";
    const char* weights_path = "/tmp/volvox-quantization-ref-reject-weights.safetensors";
    const char* prefix =
        "{\"format\":\"volvox-graph/v1\","
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{";
    const char* suffix =
        "\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"scales2\",\"zero_point_tensor\":\"zero_i8_2\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale1\","
        "\"zero_point_tensor\":\"zero_i8_1\"}}},"
        "\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"y\"]}";
    const char* invalid_x_descriptors[] = {
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"missing\","
            "\"zero_point_tensor\":\"zero_i8_1\"},",
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale_i32\","
            "\"zero_point_tensor\":\"zero_i8_1\"},",
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale_rank2\","
            "\"zero_point_tensor\":\"zero_i8_1\"},",
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale_nan\","
            "\"zero_point_tensor\":\"zero_i8_1\"},",
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale1\","
            "\"zero_point_tensor\":\"zero_u8\"},",
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale1\","
            "\"zero_point_tensor\":\"zero_i8_1\",\"extra\":1},",
    };
    char graph[4096];

    CHECK(write_quantization_ref_rejection_weights(weights_path) == 0);
    for (size_t index = 0;
         index < sizeof(invalid_x_descriptors) / sizeof(invalid_x_descriptors[0]);
         index++) {
        CHECK(snprintf(graph, sizeof(graph), "%s%s%s",
                       prefix, invalid_x_descriptors[index], suffix) > 0);
        CHECK(expect_quantization_refs_rejected(
                  graph_path, weights_path, graph) == 0);
    }
    CHECK(snprintf(
              graph, sizeof(graph),
              "%s\"x\":{\"scheme\":\"per_tensor\","
              "\"scale_tensor\":\"scale1\",\"zero_point_tensor\":\"zero_i8_1\"},"
              "\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
              "\"scale_tensor\":\"scale1\",\"zero_point_tensor\":\"zero_i8_1\"},"
              "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"scale1\","
              "\"zero_point_tensor\":\"zero_i8_1\"}}},"
              "\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\"}},"
              "\"nodes\":[{\"opType\":\"QLinear\","
              "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
              "\"outputs\":{\"out\":\"y\"},"
              "\"outputs_shape\":{\"out\":[1,2]},"
              "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
              "\"outputs\":[\"y\"]}", prefix) > 0);
    CHECK(expect_quantization_refs_rejected(
              graph_path, weights_path, graph) == 0);

    CHECK(snprintf(
              graph, sizeof(graph),
              "{\"format\":\"volvox-graph/v1\","
              "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
              "\"tensors\":{\"q\":{\"scheme\":\"per_tensor\","
              "\"scale_tensor\":\"other_scale\","
              "\"zero_point_tensor\":\"other_zero\"}}},"
              "\"inputs\":{\"x\":{\"shape\":[1],\"dtype\":\"float32\"}},"
              "\"nodes\":[{\"opType\":\"QuantizeLinear\","
              "\"inputs\":{\"input\":\"x\",\"scale\":\"scale1\","
              "\"zero_point\":\"zero_i8_1\"},\"outputs\":{\"out\":\"q\"},"
              "\"outputs_shape\":{\"out\":[1]},"
              "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
              "\"outputs\":[\"q\"]}") > 0);
    CHECK(expect_quantization_refs_rejected(
              graph_path, weights_path, graph) == 0);

    remove(weights_path);
    return 0;
}

static int test_case_distinct_quantization_refs(void) {
    const char* graph_path = "/tmp/volvox-case-distinct-quantization-graph.json";
    const char* weights_path = "/tmp/volvox-case-distinct-quantization-weights.safetensors";
    const char* graph =
        "{\"format\":\"volvox-graph/v1\","
        "\"quantization\":{\"format\":\"volvox-affine-safetensors/v1\","
        "\"tensors\":{"
        "\"x\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"x_scale\","
        "\"zero_point_tensor\":\"x_zero\"},"
        "\"W\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"W_scale\","
        "\"zero_point_tensor\":\"W_zero\"},"
        "\"w\":{\"scheme\":\"per_axis\",\"axis\":0,"
        "\"scale_tensor\":\"w_scale\",\"zero_point_tensor\":\"w_zero\"},"
        "\"y\":{\"scheme\":\"per_tensor\",\"scale_tensor\":\"y_scale\","
        "\"zero_point_tensor\":\"y_zero\"}}},"
        "\"inputs\":{\"x\":{\"shape\":[1,1],\"dtype\":\"int8\"}},"
        "\"nodes\":[{\"opType\":\"QLinear\","
        "\"inputs\":{\"input\":\"x\",\"weight\":\"w\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"outputs_dtype\":{\"out\":\"int8\"},\"params\":{}}],"
        "\"outputs\":[\"y\"]}";
    const int one[1] = {1};
    const int two[1] = {2};
    const int weight_shape[2] = {2, 1};
    const int8_t lower_weight[2] = {1, 2};
    const int8_t upper_weight[2] = {3, 4};
    const int32_t bias[2] = {0, 0};
    const float one_scale[1] = {1.0f};
    const float lower_scales[2] = {0.5f, 0.25f};
    const float upper_scale[1] = {0.75f};
    const int8_t zero1[1] = {0};
    const int8_t zero2[2] = {0, 0};
    SafetensorsFile file;

    CHECK(write_quantized_test_graph(graph_path, graph) == 0);
    CHECK(safetensors_init_empty(&file, SAFETENSORS_OPEN_READ_WRITE) == 0);
    CHECK(safetensors_add_tensor(&file, "W", VX_DTYPE_I8, weight_shape, 2,
                                 upper_weight, sizeof(upper_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "W_scale", VX_DTYPE_F32, one, 1,
                                 upper_scale, sizeof(upper_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "W_zero", VX_DTYPE_I8, one, 1,
                                 zero1, sizeof(zero1)) == 0);
    CHECK(safetensors_add_tensor(&file, "w", VX_DTYPE_I8, weight_shape, 2,
                                 lower_weight, sizeof(lower_weight)) == 0);
    CHECK(safetensors_add_tensor(&file, "w_scale", VX_DTYPE_F32, two, 1,
                                 lower_scales, sizeof(lower_scales)) == 0);
    CHECK(safetensors_add_tensor(&file, "w_zero", VX_DTYPE_I8, two, 1,
                                 zero2, sizeof(zero2)) == 0);
    CHECK(safetensors_add_tensor(&file, "b", VX_DTYPE_I32, two, 1,
                                 bias, sizeof(bias)) == 0);
    CHECK(safetensors_add_tensor(&file, "x_scale", VX_DTYPE_F32, one, 1,
                                 one_scale, sizeof(one_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "x_zero", VX_DTYPE_I8, one, 1,
                                 zero1, sizeof(zero1)) == 0);
    CHECK(safetensors_add_tensor(&file, "y_scale", VX_DTYPE_F32, one, 1,
                                 one_scale, sizeof(one_scale)) == 0);
    CHECK(safetensors_add_tensor(&file, "y_zero", VX_DTYPE_I8, one, 1,
                                 zero1, sizeof(zero1)) == 0);
    CHECK(safetensors_save(weights_path, &file) == 0);
    safetensors_free(&file);

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
    CHECK(g_qlinear_meta[0].valid && g_qlinear_meta[0].weight_scales != NULL);
    CHECK(closef(g_qlinear_meta[0].weight_scales[0], lower_scales[0]));
    CHECK(closef(g_qlinear_meta[0].weight_scales[1], lower_scales[1]));
    CHECK(t_find("W") && t_find("W")->quantization.valid);
    CHECK(closef(t_find("W")->quantization.scale, upper_scale[0]));
    volvoxai_engine_shutdown();
    remove(graph_path);
    remove(weights_path);
    return 0;
}
static int test_w8a32_linear_regression(void) {
    const char* graph_path = "/tmp/volvox-w8a32-graph.json";
    const char* weights_path = "/tmp/volvox-w8a32-weights.safetensors";
    const char* graph =
        /* Weight-only Linear keeps F32 activations while consuming quantized
           weights and explicit scale/zero-point parameter tensors. */
        "{\"format\":\"volvox-graph/v1\",\"inputs\":{\"x\":{\"shape\":[1,3],\"dtype\":\"float32\"}},\"nodes\":[{"
        "\"opType\":\"Linear\",\"inputs\":{\"input\":\"x\",\"weight\":\"w\","
        "\"weight_scale\":\"s\",\"weight_zero_point\":\"z\",\"bias\":\"b\"},"
        "\"outputs\":{\"out\":\"y\"},\"outputs_shape\":{\"out\":[1,2]},"
        "\"params\":{\"weight_layout\":\"dout_din\"}}],\"outputs\":[\"y\"]}";
    FILE* graph_file = fopen(graph_path, "wb");
    CHECK(graph_file != NULL && fwrite(graph, 1, strlen(graph), graph_file) == strlen(graph));
    CHECK(fclose(graph_file) == 0);

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

    CHECK(volvoxai_engine_init(graph_path, weights_path) == 0);
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
    remove(graph_path);
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
    VxEngineState* state = (VxEngineState*)calloc(1, sizeof(*state));
    VxEngineStateScope scope;
    if (!state || vx_engine_state_init(state) != 0) {
        free(state);
        return 1;
    }
    scope = vx_engine_state_scope_enter(state);
    CHECK(test_safetensors_rejects_recursive_duplicate_header_keys() == 0);
    CHECK(test_native_cpu_feature_contract() == 0);
    CHECK(test_groupnorm_optional_bias_kernel() == 0);
    CHECK(test_groupnorm_narrow_group_kernel() == 0);
    CHECK(test_qbatch_matmul_portable_kernel() == 0);
    CHECK(test_qbatch_matmul_native_dispatch() == 0);
    CHECK(test_transpose_i8u8_portable_kernel() == 0);
    CHECK(test_transpose_i8u8_native_kernel() == 0);
    CHECK(test_transpose_f32_native_kernel() == 0);
    CHECK(test_qembedding_portable_kernel() == 0);
    CHECK(test_qsilu_portable_kernel() == 0);
    CHECK(test_qgelu_portable_kernel() == 0);
    CHECK(test_quantized_activation_lut_exactness() == 0);
    CHECK(test_qsilu_native_parallel_kernel() == 0);
    CHECK(test_qgroupnorm_portable_kernel() == 0);
    CHECK(test_qgroupnorm_native_parallel_kernel() == 0);
    CHECK(test_qlayernorm_portable_kernel() == 0);
    CHECK(test_qlayernorm_native_parallel_kernel() == 0);
    CHECK(test_qsdpa_portable_kernel() == 0);
    CHECK(test_qargmax_portable_kernel() == 0);
    CHECK(test_qmaskedmean_portable_kernel() == 0);
    CHECK(test_w8a8_linear_vector_tail() == 0);
    CHECK(test_physical_qlinear_native_dispatch() == 0);
    CHECK(test_physical_qlinear_native_threaded_dispatch() == 0);
    CHECK(test_physical_qlinear_vnni_remapping() == 0);
    CHECK(test_physical_qlinear_requantize_no_fma() == 0);
    CHECK(test_physical_qconv_native_dispatch() == 0);
    CHECK(test_physical_qconv_requantize_no_fma() == 0);
    CHECK(test_physical_qconv_im2col_qlinear_dispatch() == 0);
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
    CHECK(test_physical_cached_bias_requires_immutable_bias() == 0);
    CHECK(test_physical_qadd_and_requantize() == 0);
    CHECK(test_physical_qbatch_matmul_broadcast() == 0);
    CHECK(test_physical_qbatch_matmul_prefix_scope() == 0);
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
    CHECK(test_retired_quantization_layouts_rejected() == 0);
    CHECK(test_set_input_f32_quantizes_physical_inputs() == 0);
    CHECK(test_concat_sigmoid_fusion_backend_scope() == 0);
    CHECK(test_physical_shape_island_chain() == 0);
    CHECK(test_physical_byte_transpose() == 0);
    CHECK(test_physical_byte_transpose_model_rejections() == 0);
    CHECK(test_physical_byte_expand() == 0);
    CHECK(test_physical_byte_expand_model_rejections() == 0);
    CHECK(test_physical_shape_maxpool_asymmetric_pads() == 0);
    CHECK(test_physical_shape_rejections() == 0);
    CHECK(test_generic_add_rejects_physical_bytes() == 0);
    CHECK(test_numeric_affine_params_rejected() == 0);
    CHECK(test_qconv2d_weight_only_f32_activation() == 0);
    CHECK(test_qconv2d_weight_only_rejects_malformed_descriptors() == 0);
    CHECK(test_safetensors_quantization_refs_runtime_and_immutability() == 0);
    CHECK(test_safetensors_quantization_ref_rejections() == 0);
    CHECK(test_case_distinct_quantization_refs() == 0);
    CHECK(test_w8a32_linear_regression() == 0);
    CHECK(test_kernel_thread_pool_restart() == 0);
    volvoxai_engine_shutdown();
    vx_engine_state_scope_leave(scope);
    vx_engine_state_deinit(state);
    free(state);
    puts("quantized runtime tests passed");
    return 0;
}
