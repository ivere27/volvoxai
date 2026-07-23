/*
 * One VNNI QLinear body, instantiated per vector width.
 *
 * AVX-VNNI and AVX-512-VNNI run the identical algorithm: remap both operands
 * into the U8 x I8 domain VPDPBUSD accepts, accumulate the dot in I32 lanes,
 * and carry the asymmetric zero points as separate operand sums so the
 * regrouped vector reduction stays exact.  Only the vector width, the lane
 * counts, the instruction spelling and the operand-sum helper differ.
 *
 * Keeping one body means a fix to the remapping or the compensation applies to
 * both widths at once.  When those were two hand-maintained copies, a
 * correction had to be made twice, and this session found two defects that
 * arose exactly that way elsewhere in the kernels.
 *
 * Contract the includer must define, then #undef:
 *
 *   VX_MK_TARGET         target attribute selecting the ISA
 *   VX_MK_SUFFIX         symbol infix, e.g. avxvnni
 *   VX_MK_VEC            vector register type
 *   VX_MK_BYTES          bytes per vector, as a `u` literal
 *   VX_MK_L32            I32 lanes per vector
 *   VX_MK_L64            I64 lanes per vector
 *   VX_MK_SUM_REMAPPED   operand-sum helper at this width
 *   VX_MK_ZERO/SET1_I8/XOR/ADD_I64/SAD_U8/DPBUSD/LOADU/STOREU
 *
 * This file is included more than once on purpose and therefore has no include
 * guard.
 */

#ifndef VX_MK_CAT
#define VX_MK_CAT2(a, b) a##b
#define VX_MK_CAT(a, b) VX_MK_CAT2(a, b)
#endif
#define VX_MK_FN(base) VX_MK_CAT(VX_MK_CAT(vx_w8a8_qlinear_, VX_MK_SUFFIX), base)

/* The AVX-512 VNNI specialization uses the same canonical U8 x I8 remapping
 * and compensation identity as the AVX-VNNI implementation, but consumes a
 * full 64-byte K block with ZMM VPDPBUSD. The eligibility proof bounds the raw
 * dot lanes as well as the centered result; dimensions after the last complete
 * ZMM block remain an ordinary scalar tail. */
static VX_MK_TARGET void VX_MK_FN(_range)(
        const VxW8A8QLinearCall *call, uint32_t begin, uint32_t end) {
    const void *input = call->input;
    const void *weight = call->weight;
    const int32_t *bias = call->bias;
    const float *weight_scales = call->weight_scales;
    const int32_t *weight_zero_points = call->weight_zero_points;
    void *output = call->output;
    const uint32_t d_in = call->d_in;
    const uint32_t d_out = call->d_out;
    const float input_scale = call->input_scale;
    const int32_t input_zero_point = call->input_zero_point;
    const float output_scale = call->output_scale;
    const int32_t output_zero_point = call->output_zero_point;
    const uint32_t input_dtype = call->input_dtype;
    const uint32_t weight_dtype = call->weight_dtype;
    const uint32_t output_dtype = call->output_dtype;
    const unsigned char *input_bytes = (const unsigned char *)input;
    const unsigned char *weight_bytes = (const unsigned char *)weight;
    const int input_xor_sign = input_dtype == VX_W8A8_I8;
    const int weight_xor_sign = weight_dtype == VX_W8A8_U8;
    const int32_t input_zero_unsigned = input_zero_point + (input_xor_sign ? 128 : 0);
    const VX_MK_VEC sign_bit = VX_MK_SET1_I8((char)0x80);
    const VX_MK_VEC zero = VX_MK_ZERO();
    const int32_t output_minimum = output_dtype == VX_W8A8_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_W8A8_I8 ? 127 : 255;
    for (uint32_t row = begin; row < end; row++) {
        const size_t input_offset = (size_t)row * d_in;
        const size_t output_offset = (size_t)row * d_out;
        const int64_t input_sum = VX_MK_SUM_REMAPPED(
            input_bytes + input_offset, d_in, input_xor_sign);
        for (uint32_t column = 0; column < d_out; column++) {
            const size_t weight_offset = (size_t)column * d_in;
            const int32_t weight_zero_signed = weight_zero_points[column] -
                (weight_dtype == VX_W8A8_U8 ? 128 : 0);
            VX_MK_VEC dot_lanes = VX_MK_ZERO();
            VX_MK_VEC weight_sum_lanes = VX_MK_ZERO();
            int32_t dot_lane_values[VX_MK_L32];
            uint64_t weight_sum_lane_values[VX_MK_L64];
            int64_t dot = 0;
            int64_t weight_sum;
            uint32_t dimension = 0;
            for (; dimension + VX_MK_BYTES <= d_in; dimension += VX_MK_BYTES) {
                VX_MK_VEC input_values = VX_MK_LOADU((input_bytes + input_offset + dimension));
                VX_MK_VEC weight_values = VX_MK_LOADU((weight_bytes + weight_offset + dimension));
                VX_MK_VEC input_unsigned = input_xor_sign
                    ? VX_MK_XOR(input_values, sign_bit) : input_values;
                VX_MK_VEC weight_signed = weight_xor_sign
                    ? VX_MK_XOR(weight_values, sign_bit) : weight_values;
                VX_MK_VEC weight_unsigned = VX_MK_XOR(weight_signed, sign_bit);
                dot_lanes = VX_MK_DPBUSD(dot_lanes, input_unsigned,
                                                 weight_signed);
                weight_sum_lanes = VX_MK_ADD_I64(weight_sum_lanes,
                    VX_MK_SAD_U8(weight_unsigned, zero));
            }
            VX_MK_STOREU(dot_lane_values, dot_lanes);
            for (uint32_t lane = 0; lane < VX_MK_L32; lane++) dot += dot_lane_values[lane];
            VX_MK_STOREU(weight_sum_lane_values, weight_sum_lanes);
            weight_sum = -(int64_t)128 * (int64_t)dimension;
            for (uint32_t lane = 0; lane < VX_MK_L64; lane++)
                weight_sum += (int64_t)weight_sum_lane_values[lane];
            for (; dimension < d_in; dimension++) {
                const int64_t input_unsigned = input_xor_sign
                    ? (int64_t)(input_bytes[input_offset + dimension] ^ 0x80u)
                    : (int64_t)input_bytes[input_offset + dimension];
                const int64_t weight_signed = weight_xor_sign
                    ? (int64_t)(int8_t)(weight_bytes[weight_offset + dimension] ^ 0x80u)
                    : (int64_t)(int8_t)weight_bytes[weight_offset + dimension];
                dot += input_unsigned * weight_signed;
                weight_sum += weight_signed;
            }
            {
                const int64_t accumulator = (int64_t)bias[column] + dot -
                    (int64_t)weight_zero_signed * input_sum -
                    (int64_t)input_zero_unsigned * weight_sum +
                    (int64_t)d_in * input_zero_unsigned * weight_zero_signed;
                const float product_scale = input_scale * weight_scales[column];
                const float multiplier = product_scale / output_scale;
                const float scaled = (float)accumulator * multiplier;
                const float transformed = scaled + (float)output_zero_point;
                const int32_t quantized = vx_w8a8_requantize(transformed,
                    output_minimum, output_maximum, output_zero_point);
                vx_w8a8_store_byte(output, output_dtype,
                    output_offset + column, quantized);
            }
        }
    }
}
/* ZMM counterpart of the four-row seed tile.  Keeping four accumulator
 * vectors live remains well within the AVX-512 register file and turns every
 * 64-byte weight load into four exact VPDPBUSD updates. */
static VX_MK_TARGET void VX_MK_FN(_tiled_range)(
        const VxW8A8QLinearCall *call, uint32_t begin, uint32_t end) {
    const unsigned char *input_bytes = (const unsigned char *)call->input;
    const unsigned char *weight_bytes = (const unsigned char *)call->weight;
    const uint32_t d_in = call->d_in;
    const uint32_t d_out = call->d_out;
    const int input_xor_sign = call->input_dtype == VX_W8A8_I8;
    const int weight_xor_sign = call->weight_dtype == VX_W8A8_U8;
    const int32_t input_zero_unsigned = call->input_zero_point +
        (input_xor_sign ? 128 : 0);
    const VX_MK_VEC sign_bit = VX_MK_SET1_I8((char)0x80);
    const VX_MK_VEC zero = VX_MK_ZERO();
    const int32_t output_minimum = call->output_dtype == VX_W8A8_I8 ? -128 : 0;
    const int32_t output_maximum = call->output_dtype == VX_W8A8_I8 ? 127 : 255;
    uint32_t row = begin;
    if (end - begin < 4u || !vx_w8a8_qlinear_parallel_alias_safe(call)) {
        VX_MK_FN(_range)(call, begin, end);
        return;
    }
    for (; row + 4u <= end; row += 4u) {
        const size_t input_offsets[4] = {
            (size_t)row * d_in,
            (size_t)(row + 1u) * d_in,
            (size_t)(row + 2u) * d_in,
            (size_t)(row + 3u) * d_in,
        };
        const size_t output_offsets[4] = {
            (size_t)row * d_out,
            (size_t)(row + 1u) * d_out,
            (size_t)(row + 2u) * d_out,
            (size_t)(row + 3u) * d_out,
        };
        const int64_t input_sums[4] = {
            VX_MK_SUM_REMAPPED(input_bytes + input_offsets[0],
                d_in, input_xor_sign),
            VX_MK_SUM_REMAPPED(input_bytes + input_offsets[1],
                d_in, input_xor_sign),
            VX_MK_SUM_REMAPPED(input_bytes + input_offsets[2],
                d_in, input_xor_sign),
            VX_MK_SUM_REMAPPED(input_bytes + input_offsets[3],
                d_in, input_xor_sign),
        };
        for (uint32_t column = 0; column < d_out; column++) {
            const size_t weight_offset = (size_t)column * d_in;
            const int32_t weight_zero_signed = call->weight_zero_points[column] -
                (call->weight_dtype == VX_W8A8_U8 ? 128 : 0);
            VX_MK_VEC dots0 = VX_MK_ZERO();
            VX_MK_VEC dots1 = VX_MK_ZERO();
            VX_MK_VEC dots2 = VX_MK_ZERO();
            VX_MK_VEC dots3 = VX_MK_ZERO();
            VX_MK_VEC weight_sum_lanes = VX_MK_ZERO();
            int32_t dot_lanes0[VX_MK_L32], dot_lanes1[VX_MK_L32], dot_lanes2[VX_MK_L32], dot_lanes3[VX_MK_L32];
            uint64_t weight_sum_lane_values[VX_MK_L64];
            int64_t dots[4] = {0, 0, 0, 0};
            int64_t weight_sum;
            uint32_t dimension = 0;
            for (; dimension + VX_MK_BYTES <= d_in; dimension += VX_MK_BYTES) {
                const VX_MK_VEC weight_values = VX_MK_LOADU(
                    (weight_bytes + weight_offset + dimension));
                const VX_MK_VEC weight_signed = weight_xor_sign
                    ? VX_MK_XOR(weight_values, sign_bit) : weight_values;
                const VX_MK_VEC weight_unsigned =
                    VX_MK_XOR(weight_signed, sign_bit);
                weight_sum_lanes = VX_MK_ADD_I64(weight_sum_lanes,
                    VX_MK_SAD_U8(weight_unsigned, zero));
#define VX_MK_ACCUMULATE_ROW(ROW, DOTS) do { \
                const VX_MK_VEC input_values = VX_MK_LOADU( \
                    (input_bytes + input_offsets[(ROW)] + dimension)); \
                const VX_MK_VEC input_unsigned = input_xor_sign \
                    ? VX_MK_XOR(input_values, sign_bit) : input_values; \
                (DOTS) = VX_MK_DPBUSD((DOTS), input_unsigned, \
                                              weight_signed); \
            } while (0)
                VX_MK_ACCUMULATE_ROW(0, dots0);
                VX_MK_ACCUMULATE_ROW(1, dots1);
                VX_MK_ACCUMULATE_ROW(2, dots2);
                VX_MK_ACCUMULATE_ROW(3, dots3);
#undef VX_MK_ACCUMULATE_ROW
            }
            VX_MK_STOREU(dot_lanes0, dots0);
            VX_MK_STOREU(dot_lanes1, dots1);
            VX_MK_STOREU(dot_lanes2, dots2);
            VX_MK_STOREU(dot_lanes3, dots3);
            for (uint32_t lane = 0; lane < VX_MK_L32; lane++) {
                dots[0] += dot_lanes0[lane];
                dots[1] += dot_lanes1[lane];
                dots[2] += dot_lanes2[lane];
                dots[3] += dot_lanes3[lane];
            }
            VX_MK_STOREU(weight_sum_lane_values, weight_sum_lanes);
            weight_sum = -(int64_t)128 * (int64_t)dimension;
            for (uint32_t lane = 0; lane < VX_MK_L64; lane++)
                weight_sum += (int64_t)weight_sum_lane_values[lane];
            for (; dimension < d_in; dimension++) {
                const int64_t weight_signed = weight_xor_sign
                    ? (int64_t)(int8_t)
                        (weight_bytes[weight_offset + dimension] ^ 0x80u)
                    : (int64_t)(int8_t)weight_bytes[weight_offset + dimension];
                weight_sum += weight_signed;
                for (uint32_t tile_row = 0; tile_row < 4u; tile_row++) {
                    const int64_t input_unsigned = input_xor_sign
                        ? (int64_t)(input_bytes[input_offsets[tile_row] + dimension] ^
                            0x80u)
                        : (int64_t)input_bytes[input_offsets[tile_row] + dimension];
                    dots[tile_row] += input_unsigned * weight_signed;
                }
            }
            {
                const float product_scale =
                    call->input_scale * call->weight_scales[column];
                const float multiplier = product_scale / call->output_scale;
                for (uint32_t tile_row = 0; tile_row < 4u; tile_row++) {
                    const int64_t accumulator = (int64_t)call->bias[column] +
                        dots[tile_row] -
                        (int64_t)weight_zero_signed * input_sums[tile_row] -
                        (int64_t)input_zero_unsigned * weight_sum +
                        (int64_t)d_in * input_zero_unsigned * weight_zero_signed;
                    const float scaled = (float)accumulator * multiplier;
                    const float transformed =
                        scaled + (float)call->output_zero_point;
                    const int32_t quantized = vx_w8a8_requantize(
                        transformed, output_minimum, output_maximum,
                        call->output_zero_point);
                    vx_w8a8_store_byte(call->output, call->output_dtype,
                        output_offsets[tile_row] + column, quantized);
                }
            }
        }
    }
    if (row < end) VX_MK_FN(_range)(call, row, end);
}
#undef VX_MK_FN
