/* Packed OHWI I2/I3/I4 x static affine A8 convolution. Bounded panels reuse the
 * exact I8 GEMM dispatcher. SIMD128 uses five-row multiples and up to 256
 * channels within the original 64-channel workspace bound. K above 4096 uses
 * centered-I16 tiles; neither route expands the complete model. */
#include "inference_kernels.h"
#include "packed_quant_gemm.h"
#define VX_QCONV4_PANEL_K 4096u
#define VX_QCONV4_PANEL_N 64u
#include "w8a8_affine.h"
#include <stddef.h>
#include <stdint.h>
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#elif defined(__SSE2__)
#include <emmintrin.h>
#endif

#define VX_QCONV4_M 64u
#define VX_QCONV4_N 16u
#define VX_QCONV4_K 64u

size_t qconv2d_packed_a8_workspace_bytes(void) {
    /* The shared pack sizing owns target-specific layouts and alignment. */
    return (size_t)VX_QCONV4_PANEL_K * (VX_QCONV4_PANEL_N + VX_QCONV4_M) +
           vx_packed_q8_weight_size(VX_QCONV4_PANEL_K, VX_QCONV4_PANEL_N) +
           VX_QCONV4_M * VX_QCONV4_PANEL_N + sizeof(int32_t) * VX_QCONV4_PANEL_N;
}

static int vx_qconv4_mul(size_t *value, size_t factor) {
    if (factor && *value > SIZE_MAX / factor)
        return 0;
    *value *= factor;
    return 1;
}

static int vx_qconv4_positive_finite(float value) {
    union {
        float f;
        uint32_t u;
    } bits = {value};
    return value > 0.0f && ((bits.u >> 23u) & 255u) != 255u;
}

/* Four output channels share each activation pair; four rows reuse each
 * interleaved weight vector. I32 lanes retain complete partial sums. */
static void vx_qconv4_tile(const int16_t *patch, const int16_t *weight, int32_t *acc, size_t rows,
                           uint32_t columns, size_t terms) {
    for (size_t r = 0; r < rows; r += 4u) {
        for (uint32_t c = 0; c < columns; c += 4u) {
            const int16_t *w = weight + (size_t)c * VX_QCONV4_K;
#if defined(__wasm_simd128__) || defined(__SSE2__)
#if defined(__wasm_simd128__)
            v128_t sums[4];
            for (unsigned i = 0; i < 4u; i++)
                sums[i] = wasm_v128_load(acc + (r + i) * VX_QCONV4_N + c);
#else
            __m128i sums[4];
            for (unsigned i = 0; i < 4u; i++)
                sums[i] = _mm_loadu_si128((const __m128i *)(acc + (r + i) * VX_QCONV4_N + c));
#endif
            for (size_t k = 0; k < terms; k += 2u) {
#if defined(__wasm_simd128__)
                const v128_t weights = wasm_v128_load(w + k * 4u);
#else
                const __m128i weights = _mm_loadu_si128((const __m128i *)(w + k * 4u));
#endif
                for (unsigned i = 0; i < 4u; i++) {
                    int32_t pair;
                    __builtin_memcpy(&pair, patch + (r + i) * VX_QCONV4_K + k, sizeof(pair));
#if defined(__wasm_simd128__)
                    sums[i] = wasm_i32x4_add(sums[i],
                                             wasm_i32x4_dot_i16x8(wasm_i32x4_splat(pair), weights));
#else
                    sums[i] = _mm_add_epi32(sums[i], _mm_madd_epi16(_mm_set1_epi32(pair), weights));
#endif
                }
            }
            for (unsigned i = 0; i < 4u; i++) {
#if defined(__wasm_simd128__)
                wasm_v128_store(acc + (r + i) * VX_QCONV4_N + c, sums[i]);
#else
                _mm_storeu_si128((__m128i *)(acc + (r + i) * VX_QCONV4_N + c), sums[i]);
#endif
            }
#else
            for (unsigned i = 0; i < 4u; i++)
                for (unsigned j = 0; j < 4u; j++) {
                    int32_t sum = acc[(r + i) * VX_QCONV4_N + c + j];
                    for (size_t k = 0; k < terms; k += 2u)
                        sum += (int32_t)patch[(r + i) * VX_QCONV4_K + k] * w[k * 4u + j * 2u] +
                               (int32_t)patch[(r + i) * VX_QCONV4_K + k + 1u] *
                                   w[k * 4u + j * 2u + 1u];
                    acc[(r + i) * VX_QCONV4_N + c + j] = sum;
                }
#endif
        }
    }
}

static inline int32_t vx_qconv_packed_value(const uint8_t *row, size_t k, uint32_t bits) {
    /* Quotient/remainder avoids overflowing k*bits for large rejected shapes. */
    size_t byte=(k/8u)*bits+((k%8u)*bits)/8u;
    uint32_t shift=(uint32_t)((k%8u)*bits)%8u;
    uint32_t value=(uint32_t)row[byte]>>shift;
    if(shift+bits>8u)value|=(uint32_t)row[byte+1u]<<(8u-shift);
    return (int32_t)(value&((1u<<bits)-1u))-(1<<(bits-1u));
}

int qconv2d_packed_a8(uint32_t weight_bits, const void *input, const uint8_t *weight, const int32_t *bias,
                 const float *weight_scales, void *output, uint32_t batch, uint32_t input_height,
                 uint32_t input_width, uint32_t input_channels, uint32_t output_height,
                 uint32_t output_width, uint32_t output_channels, uint32_t kernel_height,
                 uint32_t kernel_width, uint32_t input_per_group, uint32_t stride_y,
                 uint32_t stride_x, uint32_t dilation_y, uint32_t dilation_x, uint32_t padding_top,
                 uint32_t padding_left, uint32_t padding_bottom, uint32_t padding_right,
                 uint32_t groups, uint32_t relu, float input_scale, int32_t input_zero_point,
                 float output_scale, int32_t output_zero_point, uint32_t input_dtype,
                 uint32_t output_dtype, void *workspace, size_t workspace_capacity) {
    size_t input_elements = batch, output_elements = batch, terms = kernel_height;
    size_t packed_bytes, row_bytes, rows;
    uint64_t padded_h, padded_w, effective_h, effective_w, bound;
    uint32_t channels_per_group;
    int32_t output_minimum, output_maximum, relu6_upper = 0, maximum_input;
    int16_t *patch;
    int16_t *unpacked;
    int32_t *acc;
    if ((weight_bits != 2u && weight_bits != 3u && weight_bits != 4u) ||
        !input || !weight || !weight_scales || !output || !workspace ||
        ((uintptr_t)workspace % _Alignof(int32_t)) ||
        workspace_capacity < qconv2d_packed_a8_workspace_bytes() || !batch || !input_height ||
        !input_width || !input_channels || !output_height || !output_width || !output_channels ||
        !kernel_height || !kernel_width || !input_per_group || !stride_y || !stride_x ||
        !dilation_y || !dilation_x || !groups || relu > 2u || input_channels % groups ||
        output_channels % groups || (uint64_t)input_per_group * groups != input_channels ||
        !vx_w8a8_byte_dtype(input_dtype) || !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype) ||
        !vx_qconv4_positive_finite(input_scale) || !vx_qconv4_positive_finite(output_scale))
        return 0;
    if (!vx_qconv4_mul(&input_elements, input_height) ||
        !vx_qconv4_mul(&input_elements, input_width) ||
        !vx_qconv4_mul(&input_elements, input_channels) ||
        !vx_qconv4_mul(&output_elements, output_height) ||
        !vx_qconv4_mul(&output_elements, output_width) ||
        !vx_qconv4_mul(&output_elements, output_channels) || !vx_qconv4_mul(&terms, kernel_width) ||
        !vx_qconv4_mul(&terms, input_per_group))
        return 0;
    const uint32_t offset = 1u << (weight_bits - 1u);
    row_bytes = (terms / 8u) * weight_bits + ((terms % 8u) * weight_bits + 7u) / 8u;
    packed_bytes = row_bytes;
    if (!vx_qconv4_mul(&packed_bytes, output_channels))
        return 0;
    padded_h = (uint64_t)input_height + padding_top + padding_bottom;
    padded_w = (uint64_t)input_width + padding_left + padding_right;
    effective_h = (uint64_t)(kernel_height - 1u) * dilation_y + 1u;
    effective_w = (uint64_t)(kernel_width - 1u) * dilation_x + 1u;
    if (padded_h < effective_h || padded_w < effective_w ||
        (padded_h - effective_h) / stride_y + 1u != output_height ||
        (padded_w - effective_w) / stride_x + 1u != output_width)
        return 0;
    maximum_input = input_dtype == VX_DTYPE_I8
                        ? (input_zero_point < 0 ? 127 - input_zero_point : 128 + input_zero_point)
                        : (input_zero_point < 128 ? 255 - input_zero_point : input_zero_point);
    if (terms > INT32_MAX / ((uint32_t)maximum_input * offset))
        return 0;
    bound = (uint64_t)terms * (uint32_t)maximum_input * offset;
    for (uint32_t oc = 0; oc < output_channels; oc++) {
        const float product = input_scale * weight_scales[oc];
        const float multiplier = product / output_scale;
        uint64_t magnitude =
            bias ? (bias[oc] < 0 ? (uint64_t) - (int64_t)bias[oc] : (uint64_t)bias[oc]) : 0u;
        if (!vx_qconv4_positive_finite(weight_scales[oc]) || !vx_qconv4_positive_finite(product) ||
            !vx_qconv4_positive_finite(multiplier) || magnitude > INT32_MAX - bound ||
            (((terms % 8u) * weight_bits % 8u) &&
             (weight[(size_t)oc * row_bytes + row_bytes - 1u] >> ((terms % 8u) * weight_bits % 8u)) !=
             ((weight_bits == 3u ? 0u : weight_bits == 2u ? 0xaau : 0x88u) >> ((terms % 8u) * weight_bits % 8u))))
            return 0;
    }
    output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    if (relu == 2u)
        relu6_upper = vx_w8a8_requantize(6.0f / output_scale + (float)output_zero_point,
                                         output_minimum, output_maximum, 0);
    if (terms <= VX_QCONV4_PANEL_K) {
        uint32_t columns = VX_QCONV4_PANEL_N;
        uint32_t panel_rows = VX_QCONV4_M;
#if defined(__wasm_simd128__)
        /* The shared SIMD128 GEMM consumes five rows at a time. Avoid a
         * four-row remainder in every 64-row call, and spend unused K-panel
         * space on more columns so they share the same im2col rows. Keep the
         * original workspace bound and at least half its row capacity. */
        panel_rows -= panel_rows % 5u;
        for (uint32_t candidate = 128u;
             candidate <= 256u && candidate / 2u < output_channels / groups;
             candidate *= 2u) {
            const uint32_t candidate_pack =
                vx_packed_q8_weight_size((uint32_t)terms, candidate);
            const size_t fixed_bytes = terms * candidate + candidate_pack +
                                       sizeof(int32_t) * candidate + 3u;
            const size_t capacity = qconv2d_packed_a8_workspace_bytes();
            if (!candidate_pack || fixed_bytes >= capacity)
                break;
            size_t available_rows = (capacity - fixed_bytes) / (terms + candidate);
            if (available_rows > VX_QCONV4_M)
                available_rows = VX_QCONV4_M;
            available_rows -= available_rows % 5u;
            if (available_rows < VX_QCONV4_M / 2u)
                break;
            columns = candidate;
            panel_rows = (uint32_t)available_rows;
        }
#endif
        const uint32_t packed_capacity = vx_packed_q8_weight_size((uint32_t)terms, columns);
        int8_t *decoded = (int8_t *)workspace;
        uint8_t *packed = (uint8_t *)decoded + terms * columns;
        uint8_t *patch_bytes = packed + packed_capacity;
        /* An odd K and a five-row multiple can leave the byte patch unaligned. */
        const size_t patch_capacity = (terms * panel_rows + 3u) & ~(size_t)3u;
        uint8_t *temporary_output = patch_bytes + patch_capacity;
        int32_t *zeros = (int32_t *)(temporary_output + panel_rows * columns);
        for (uint32_t c = 0; c < columns; c++)
            zeros[c] = 0;
        const size_t all_rows = output_elements / output_channels;
        const uint32_t group_outputs = output_channels / groups;
        for (uint32_t group = 0; group < groups; group++) {
            for (uint32_t c0 = 0; c0 < group_outputs;) {
                uint32_t nc = group_outputs - c0;
                if (nc > columns)
                    nc = columns;
                const uint32_t first = group * group_outputs + c0;
                for (uint32_t c = 0; c < nc; c++) {
                    const uint8_t *source = weight + (size_t)(first + c) * row_bytes;
                    int8_t *destination = decoded + (size_t)c * terms;
                    size_t k = 0;
                    if (weight_bits == 4u) {
                        for (; terms - k >= 2u; k += 2u) {
                            uint8_t code = source[k / 2u];
                            destination[k] = (int8_t)((int32_t)(code & 15u) - 8);
                            destination[k + 1u] = (int8_t)((int32_t)(code >> 4u) - 8);
                        }
                    } else if (weight_bits == 2u) {
                        for (; terms - k >= 4u; k += 4u) {
                            uint8_t code = source[k / 4u];
                            destination[k] = (int8_t)((int32_t)(code & 3u) - 2);
                            destination[k + 1u] = (int8_t)((int32_t)((code >> 2u) & 3u) - 2);
                            destination[k + 2u] = (int8_t)((int32_t)((code >> 4u) & 3u) - 2);
                            destination[k + 3u] = (int8_t)((int32_t)(code >> 6u) - 2);
                        }
                    }
                    for (; k < terms; k++)
                        destination[k] = (int8_t)vx_qconv_packed_value(source, k, weight_bits);
                }
                if (!vx_pack_q8_weight(packed, packed_capacity, decoded, (uint32_t)terms, nc,
                                       VX_DTYPE_I8, 1u))
                    return 0;
                for (size_t r0 = 0; r0 < all_rows;) {
                    uint32_t nr =
                        (uint32_t)(all_rows - r0 < panel_rows ? all_rows - r0 : panel_rows);
                    uint32_t ox = (uint32_t)(r0 % output_width),
                             oy = (uint32_t)((r0 / output_width) % output_height);
                    size_t b = r0 / output_width / output_height;
                    for (uint32_t r = 0; r < nr; r++) {
                        for (uint32_t ky = 0; ky < kernel_height; ky++)
                            for (uint32_t kx = 0; kx < kernel_width; kx++) {
                                uint64_t py = (uint64_t)oy * stride_y + (uint64_t)ky * dilation_y;
                                uint64_t px = (uint64_t)ox * stride_x + (uint64_t)kx * dilation_x;
                                uint8_t *destination =
                                    patch_bytes + (size_t)r * terms +
                                    ((size_t)ky * kernel_width + kx) * input_per_group;
                                if (py >= padding_top && py - padding_top < input_height &&
                                    px >= padding_left && px - padding_left < input_width) {
                                    size_t index =
                                        ((b * input_height + (size_t)(py - padding_top)) *
                                             input_width +
                                         (size_t)(px - padding_left)) *
                                            input_channels +
                                        group * input_per_group;
                                    const uint8_t *source = (const uint8_t *)input + index;
                                    for (uint32_t ci = 0; ci < input_per_group; ci++)
                                        destination[ci] = source[ci];
                                } else
                                    for (uint32_t ci = 0; ci < input_per_group; ci++)
                                        destination[ci] = (uint8_t)input_zero_point;
                            }
                        if (++ox == output_width) {
                            ox = 0;
                            if (++oy == output_height) {
                                oy = 0;
                                b++;
                            }
                        }
                    }
                    if (!vx_qlinear_i8u8_packed(patch_bytes, packed, bias ? bias + first : zeros,
                                                weight_scales + first, zeros, temporary_output, nr,
                                                (uint32_t)terms, nc, input_scale, input_zero_point,
                                                output_scale, output_zero_point, input_dtype,
                                                VX_DTYPE_I8, output_dtype))
                        return 0;
                    for (uint32_t r = 0; r < nr; r++)
                        for (uint32_t c = 0; c < nc; c++) {
                            int32_t q = vx_w8a8_byte_value(temporary_output, output_dtype,
                                                           (size_t)r * nc + c);
                            if (relu && q < output_zero_point)
                                q = output_zero_point;
                            if (relu == 2u && q > relu6_upper)
                                q = relu6_upper;
                            vx_w8a8_store_byte(output, output_dtype,
                                               (r0 + r) * output_channels + first + c, q);
                        }
                    r0 += nr;
                }
                c0 += nc;
            }
        }
        return 1;
    }
    /* Derive typed slices only after proving the supplied extent and alignment. */
    patch = (int16_t *)workspace;
    unpacked = patch + VX_QCONV4_M * VX_QCONV4_K;
    acc = (int32_t *)(unpacked + VX_QCONV4_N * VX_QCONV4_K);
    rows = output_elements / output_channels;
    channels_per_group = output_channels / groups;
    for (uint32_t group = 0; group < groups; group++) {
        for (size_t r0 = 0; r0 < rows; r0 += rows - r0 < VX_QCONV4_M ? rows - r0 : VX_QCONV4_M) {
            size_t nr = rows - r0 < VX_QCONV4_M ? rows - r0 : VX_QCONV4_M;
            for (uint32_t c0 = 0; c0 < channels_per_group;
                 c0 +=
                 channels_per_group - c0 < VX_QCONV4_N ? channels_per_group - c0 : VX_QCONV4_N) {
                uint32_t nc =
                    channels_per_group - c0 < VX_QCONV4_N ? channels_per_group - c0 : VX_QCONV4_N;
                uint32_t first_channel = group * channels_per_group + c0;
                const size_t tile_rows = (nr + 3u) & ~(size_t)3u;
                const uint32_t tile_columns = (nc + 3u) & ~3u;
                for (size_t r = 0; r < tile_rows; r++)
                    for (uint32_t c = 0; c < tile_columns; c++)
                        acc[r * VX_QCONV4_N + c] = c < nc && bias ? bias[first_channel + c] : 0;
                for (size_t k0 = 0; k0 < terms; k0 += VX_QCONV4_K) {
                    size_t nk = terms - k0 < VX_QCONV4_K ? terms - k0 : VX_QCONV4_K;
                    size_t tile_terms = (nk + 1u) & ~(size_t)1u;
                    /* I2/I4 pairs share one byte; dense I3 can cross bytes.
                     * Interleave four channels for four I32 SIMD lanes. */
                    for (uint32_t c = 0; c < tile_columns; c += 4u) {
                        for (size_t k = 0; k < tile_terms; k += 2u) {
                            for (uint32_t lane = 0; lane < 4u; lane++) {
                                size_t at = (size_t)c * VX_QCONV4_K + k * 4u + lane * 2u;
                                const uint8_t *source = c + lane < nc
                                    ? weight + (size_t)(first_channel + c + lane) * row_bytes : weight;
                                if (weight_bits != 3u) {
                                    /* The admitted I32 dot bound also proves this bit offset fits. */
                                    size_t bit = (k0 + k) * weight_bits;
                                    uint32_t code = c + lane < nc ? source[bit / 8u] >> (bit % 8u)
                                        : (weight_bits == 2u ? 0xaau : 0x88u);
                                    uint32_t mask = (1u << weight_bits) - 1u;
                                    unpacked[at] = (int16_t)((int32_t)(code & mask) - (int32_t)offset);
                                    unpacked[at + 1u] = k + 1u < nk
                                        ? (int16_t)((int32_t)((code >> weight_bits) & mask) - (int32_t)offset) : 0;
                                } else {
                                    unpacked[at] = c + lane < nc
                                        ? (int16_t)vx_qconv_packed_value(source, k0 + k, weight_bits) : 0;
                                    unpacked[at + 1u] = c + lane < nc && k + 1u < nk
                                        ? (int16_t)vx_qconv_packed_value(source, k0 + k + 1u, weight_bits) : 0;
                                }
                            }
                        }
                    }
                    /* NHWC channel segments are contiguous. Resolve spatial
                     * division once per segment, never once per input value. */
                    for (size_t k = 0; k < nk;) {
                        size_t at = k0 + k, position = at / input_per_group;
                        uint32_t ci = (uint32_t)(at % input_per_group);
                        size_t count = input_per_group - ci;
                        if (count > nk - k)
                            count = nk - k;
                        uint64_t ky = position / kernel_width, kx = position % kernel_width;
                        uint32_t ox = (uint32_t)(r0 % output_width);
                        uint32_t oy = (uint32_t)((r0 / output_width) % output_height);
                        size_t b = r0 / output_width / output_height;
                        for (size_t r = 0; r < nr; r++) {
                            uint64_t py = (uint64_t)oy * stride_y + ky * dilation_y;
                            uint64_t px = (uint64_t)ox * stride_x + kx * dilation_x;
                            int16_t *destination = patch + r * VX_QCONV4_K + k;
                            if (py >= padding_top && py - padding_top < input_height &&
                                px >= padding_left && px - padding_left < input_width) {
                                size_t index =
                                    ((b * input_height + (size_t)(py - padding_top)) * input_width +
                                     (size_t)(px - padding_left)) *
                                        input_channels +
                                    group * input_per_group + ci;
                                if (input_dtype == VX_DTYPE_I8) {
                                    const int8_t *source = (const int8_t *)input + index;
                                    for (size_t j = 0; j < count; j++)
                                        destination[j] =
                                            (int16_t)((int32_t)source[j] - input_zero_point);
                                } else {
                                    const uint8_t *source = (const uint8_t *)input + index;
                                    for (size_t j = 0; j < count; j++)
                                        destination[j] =
                                            (int16_t)((int32_t)source[j] - input_zero_point);
                                }
                            } else
                                for (size_t j = 0; j < count; j++)
                                    destination[j] = 0;
                            if (++ox == output_width) {
                                ox = 0;
                                if (++oy == output_height) {
                                    oy = 0;
                                    b++;
                                }
                            }
                        }
                        k += count;
                    }
                    for (size_t r = 0; r < nr; r++)
                        if (tile_terms != nk)
                            patch[r * VX_QCONV4_K + nk] = 0;
                    for (size_t r = nr; r < tile_rows; r++)
                        for (size_t k = 0; k < tile_terms; k++)
                            patch[r * VX_QCONV4_K + k] = 0;
                    vx_qconv4_tile(patch, unpacked, acc, tile_rows, tile_columns, tile_terms);
                }
                for (uint32_t c = 0; c < nc; c++) {
                    const float product_scale = input_scale * weight_scales[first_channel + c];
                    const float multiplier = product_scale / output_scale;
                    for (size_t r = 0; r < nr; r++) {
                        const float scaled = (float)acc[r * VX_QCONV4_N + c] * multiplier;
                        const float transformed = scaled + (float)output_zero_point;
                        int32_t q = vx_w8a8_requantize(transformed, output_minimum, output_maximum,
                                                       output_zero_point);
                        if (transformed == transformed && relu) {
                            if (q < output_zero_point)
                                q = output_zero_point;
                            if (relu == 2u && q > relu6_upper)
                                q = relu6_upper;
                        }
                        vx_w8a8_store_byte(output, output_dtype,
                                           (r0 + r) * output_channels + first_channel + c, q);
                    }
                }
            }
        }
    }
    return 1;
}
