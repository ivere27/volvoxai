/*
 * VolvoxAI CUDA Forward Dual GEMV Kernels
 *
 * Parallel row decode for pairs of linear projections sharing the same input
 * (e.g. Gate + Up projection in LLaMA / TinyLlama MLP).
 */

#ifndef VOLVOXAI_CUDA_FORWARD_GEMV_PAIR_KERNELS_INC
#define VOLVOXAI_CUDA_FORWARD_GEMV_PAIR_KERNELS_INC

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE int vx_cuda_gemv_shuffle_down_i32(
        int value, vx_u32 delta) {
    int shuffled;
    asm("shfl.sync.down.b32 %0, %1, %2, 0x1f, 0xffffffff;"
        : "=r"(shuffled) : "r"(value), "r"(delta));
    return shuffled;
}

/* Dense GEMV pair: computes two outputs from one input row */
extern "C" VX_CUDA_GLOBAL void vx_cuda_gemv_pair_out_in_f32(
        const float* input, const float* weight0, const float* weight1,
        float* output0, float* output1,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight0, weight1, output0, output1, rows, d_in, d_out);
    if (rows != 1u) return;
    vx_u32 lane = threadIdx.x & 31u;
    vx_u32 warp = threadIdx.x >> 5u;
    vx_u32 column = blockIdx.x * 4u + warp;
    float sum0 = 0.0f, sum1 = 0.0f;
    if (column < d_out)
#if defined(__CUDA_ARCH__)
#pragma unroll 4
#endif
        for (vx_u32 k = lane; k < d_in; k += 32u) {
            float value = input[k];
            sum0 += value * weight0[column * d_in + k];
            sum1 += value * weight1[column * d_in + k];
        }
    for (vx_u32 delta = 16u; delta != 0u; delta >>= 1u) {
        sum0 += vx_cuda_gemv_shuffle_down(sum0, delta);
        sum1 += vx_cuda_gemv_shuffle_down(sum1, delta);
    }
    if (rows == 1u && lane == 0u && column < d_out) {
        output0[column] = sum0 + 0.0f;
        output1[column] = sum1 + 0.0f;
    }
}

#ifndef TEST_CUDA_GEMV_SOURCE
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE vx_u32 vx_packed_cuda_w4_tail_word(
        const vx_u32* w_row, vx_u32 input_width, vx_u32 k) {
    if (k >= input_width) return 0u;
    vx_u32 word = w_row[k >> 3u];
    return (k & 4u) ? vx_packed_cuda_unpack_w4_high(word) : vx_packed_cuda_unpack_w4_low(word);
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE vx_u32 vx_cuda_stage_rmsnorm_float(
        const float* input, const float* norm_weight, float* norm_output,
        float inverse, vx_u32 input_width, vx_u32 groups, vx_u32 activation_bits) {
    VX_CUDA_SHARED vx_u32 s_invalid[8];
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 invalid = 0u;
    if (activation_bits == 16u) {
        for (vx_u32 word = threadIdx.x; word < groups * 64u; word += 256u) {
            vx_u32 k = word * 2u;
            float x0 = 0.0f, x1 = 0.0f;
            if (k < input_width) {
                x0 = vx_mul_rn(vx_mul_rn(input[k], inverse), norm_weight[k]);
                if (norm_output && !blockIdx.x) norm_output[k] = x0;
                invalid |= !(x0 <= 3.402823466e38f && x0 >= -3.402823466e38f);
            }
            if (k + 1u < input_width) {
                x1 = vx_mul_rn(vx_mul_rn(input[k + 1u], inverse), norm_weight[k + 1u]);
                if (norm_output && !blockIdx.x) norm_output[k + 1u] = x1;
                invalid |= !(x1 <= 3.402823466e38f && x1 >= -3.402823466e38f);
            }
            vx_packed_tile_shared[word] = (vx_u32)vx_packed_cuda_half_bits(x0) |
                ((vx_u32)vx_packed_cuda_half_bits(x1) << 16u);
        }
    } else {
        float* s_float = (float*)vx_packed_tile_shared;
        for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
            float x = 0.0f;
            if (k < input_width) {
                x = vx_mul_rn(vx_mul_rn(input[k], inverse), norm_weight[k]);
                if (norm_output && !blockIdx.x) norm_output[k] = x;
                invalid |= !(x <= 3.402823466e38f && x >= -3.402823466e38f);
            }
            s_float[k] = x;
        }
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u) {
        vx_u32 other;
        asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(invalid), "r"(delta));
        invalid |= other;
    }
    if (!lane) s_invalid[warp] = invalid;
    vx_sync_threads();
    return s_invalid[0] | s_invalid[1] | s_invalid[2] | s_invalid[3] |
           s_invalid[4] | s_invalid[5] | s_invalid[6] | s_invalid[7];
}

template <bool EightBit, bool HalfRound>
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_gemv_pair_float(
        const vx_u32* w0_row, const float* s0_row,
        const vx_u32* w1_row, const float* s1_row,
        vx_u32 input_width, vx_u32 groups, vx_u32 sub_lane, bool valid_col,
        float* out_sum0, float* out_sum1) {
    float sum0 = 0.0f, sum1 = 0.0f;
    vx_u32 full_groups = input_width >> 7u;
    vx_u32 g = 0u;
    if (valid_col) {
        for (; g < full_groups; ++g) {
            VxCudaFloat4 a0, a1, a2, a3;
            vx_cuda_load_act16_float<HalfRound>(g, sub_lane, &a0, &a1, &a2, &a3);
            sum0 += vx_packed_cuda_dot16_float<EightBit, HalfRound>(a0, a1, a2, a3, w0_row, g, sub_lane, s0_row[g]);
            sum1 += vx_packed_cuda_dot16_float<EightBit, HalfRound>(a0, a1, a2, a3, w1_row, g, sub_lane, s1_row[g]);
        }
        for (; g < groups; ++g) {
            sum0 += vx_packed_cuda_tail_float<EightBit, HalfRound>(w0_row, s0_row[g], input_width, g, sub_lane);
            sum1 += vx_packed_cuda_tail_float<EightBit, HalfRound>(w1_row, s1_row[g], input_width, g, sub_lane);
        }
    }
    for (vx_u32 delta = 4u; delta; delta >>= 1u) {
        sum0 += vx_cuda_gemv_shuffle_down(sum0, delta);
        sum1 += vx_cuda_gemv_shuffle_down(sum1, delta);
    }
    *out_sum0 = sum0;
    *out_sum1 = sum1;
}

static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE void vx_cuda_gemv_pair_float_dispatch(
        const vx_u32* w0_row, const float* s0_row,
        const vx_u32* w1_row, const float* s1_row,
        vx_u32 input_width, vx_u32 groups, vx_u32 sub_lane, bool valid_col,
        vx_u32 weight_bits, vx_u32 activation_bits,
        float* out_sum0, float* out_sum1) {
    if (weight_bits == 8u) {
        if (activation_bits == 16u) {
            vx_cuda_gemv_pair_float<true, true>(w0_row, s0_row, w1_row, s1_row, input_width, groups, sub_lane, valid_col, out_sum0, out_sum1);
        } else {
            vx_cuda_gemv_pair_float<true, false>(w0_row, s0_row, w1_row, s1_row, input_width, groups, sub_lane, valid_col, out_sum0, out_sum1);
        }
    } else {
        if (activation_bits == 16u) {
            vx_cuda_gemv_pair_float<false, true>(w0_row, s0_row, w1_row, s1_row, input_width, groups, sub_lane, valid_col, out_sum0, out_sum1);
        } else {
            vx_cuda_gemv_pair_float<false, false>(w0_row, s0_row, w1_row, s1_row, input_width, groups, sub_lane, valid_col, out_sum0, out_sum1);
        }
    }
}

/* Adjacent same-input PackedLinear nodes share the single input quantization
 * and row load in shared memory. Each CTA of 256 threads processes 32 output
 * columns for both projections, overlapping 4 groups with 128-bit loads. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_linear_pair_decode_a8(
        const float* input,
        const vx_u32* packed_weights0, const float* weight_scales0, const float* bias0, float* output0,
        const vx_u32* packed_weights1, const float* weight_scales1, const float* bias1, float* output1,
        vx_u32 input_width, vx_u32 output_width, vx_u32 weight_bits, vx_u32 activation_bits,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, packed_weights0, weight_scales0, bias0, output0,
        packed_weights1, weight_scales1, bias1, output1,
        input_width, output_width, weight_bits, activation_bits);
    (void)weight_bits;

    float* maxima = (float*)vx_packed_tile_shared;
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 sub_lane = lane & 7u;
    vx_u64 column_tiles = ((vx_u64)output_width + 31u) / 32u;
    vx_u64 column = ((vx_u64)blockIdx.x % column_tiles) * 32u + warp * 4u + (lane >> 3u);

    if (activation_bits == 16u || activation_bits == 32u) {
        vx_u32 groups = (input_width + 127u) >> 7u;
        bool valid_col = column < (vx_u64)output_width;
        vx_u32 invalid = vx_cuda_stage_input_float(input, input_width, groups, activation_bits);
        if (invalid) {
            if (!sub_lane && valid_col) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output0[column] = nan.value;
                output1[column] = nan.value;
            }
            return;
        }
        vx_u64 stride = (weight_bits == 8u) ? ((input_width + 3u) >> 2u) : ((input_width + 7u) >> 3u);
        const vx_u32* w0_row = valid_col ? packed_weights0 + column * stride : 0;
        const vx_u32* w1_row = valid_col ? packed_weights1 + column * stride : 0;
        const float* s0_row = valid_col ? weight_scales0 + column * groups : 0;
        const float* s1_row = valid_col ? weight_scales1 + column * groups : 0;
        float sum0 = 0.0f, sum1 = 0.0f;
        vx_cuda_gemv_pair_float_dispatch(w0_row, s0_row, w1_row, s1_row,
            input_width, groups, sub_lane, valid_col, weight_bits, activation_bits,
            &sum0, &sum1);
        if (!sub_lane && valid_col) {
            if (bias0) sum0 += bias0[column];
            if (bias1) sum1 += bias1[column];
            output0[column] = sum0;
            output1[column] = sum1;
        }
        return;
    }

    float maximum = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float val = input[k];
        maximum = !(val <= 3.402823466e38f && val >= -3.402823466e38f) ?
            __builtin_inff() : vx_max(maximum, vx_abs(val));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
    if (!lane) maxima[warp] = maximum;
    vx_sync_threads();
    if (!warp) {
        maximum = lane < 8u ? maxima[lane] : 0.0f;
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
        if (!lane) maxima[0] = maximum;
    }
    vx_sync_threads();
    maximum = maxima[0];
    vx_sync_threads();

    float activation_scale = maximum > 0.0f ?
        vx_max(maximum / 127.0f, 1.1754943508222875e-38f) : 1.0f;
    if (activation_scale == __builtin_inff()) {
        if (!sub_lane && column < (vx_u64)output_width) {
            union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
            output0[column] = nan.value;
            output1[column] = nan.value;
        }
        return;
    }

    vx_u32 groups = (input_width + 127u) >> 7u;
    for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
        vx_u32 value = (vx_u32)(k < input_width ? vx_packed_cuda_quantize(input[k], activation_scale) : 0) & 255u;
        for (vx_u32 shift = 1u; shift < 4u; shift <<= 1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value), "r"(shift));
            value |= other << (shift * 8u);
        }
        if (!(lane & 3u)) vx_packed_tile_shared[k >> 2u] = value;
    }
    vx_sync_threads();

    float sum0 = 0.0f, sum1 = 0.0f;
    vx_u32 full_groups = input_width >> 7u;
    vx_u64 stride = (input_width + 7u) >> 3u;
    const vx_u32* w0_row = column < (vx_u64)output_width ? packed_weights0 + column * stride : 0;
    const vx_u32* w1_row = column < (vx_u64)output_width ? packed_weights1 + column * stride : 0;
    const float* s0_row = column < (vx_u64)output_width ? weight_scales0 + column * groups : 0;
    const float* s1_row = column < (vx_u64)output_width ? weight_scales1 + column * groups : 0;

    vx_u32 g = 0u;
    for (; g + 3u < full_groups; g += 4u) {
        int dot0_0 = 0, dot0_1 = 0, dot0_2 = 0, dot0_3 = 0;
        int dot1_0 = 0, dot1_1 = 0, dot1_2 = 0, dot1_3 = 0;
        float scale0_0 = 0.0f, scale0_1 = 0.0f, scale0_2 = 0.0f, scale0_3 = 0.0f;
        float scale1_0 = 0.0f, scale1_1 = 0.0f, scale1_2 = 0.0f, scale1_3 = 0.0f;

        if (column < (vx_u64)output_width) {
            VxCudaUint4 act0 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 0u) * 32u))[sub_lane];
            VxCudaUint4 act1 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 1u) * 32u))[sub_lane];
            VxCudaUint4 act2 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 2u) * 32u))[sub_lane];
            VxCudaUint4 act3 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 3u) * 32u))[sub_lane];

            const vx_u32* wp0 = w0_row + g * 16u + sub_lane * 2u;
            vx_u32 w0_0 = wp0[0], w0_1 = wp0[1];
            vx_u32 w0_2 = wp0[16], w0_3 = wp0[17];
            vx_u32 w0_4 = wp0[32], w0_5 = wp0[33];
            vx_u32 w0_6 = wp0[48], w0_7 = wp0[49];

            const vx_u32* wp1 = w1_row + g * 16u + sub_lane * 2u;
            vx_u32 w1_0 = wp1[0], w1_1 = wp1[1];
            vx_u32 w1_2 = wp1[16], w1_3 = wp1[17];
            vx_u32 w1_4 = wp1[32], w1_5 = wp1[33];
            vx_u32 w1_6 = wp1[48], w1_7 = wp1[49];

            if (!sub_lane) {
                scale0_0 = activation_scale * s0_row[g + 0u];
                scale0_1 = activation_scale * s0_row[g + 1u];
                scale0_2 = activation_scale * s0_row[g + 2u];
                scale0_3 = activation_scale * s0_row[g + 3u];
                scale1_0 = activation_scale * s1_row[g + 0u];
                scale1_1 = activation_scale * s1_row[g + 1u];
                scale1_2 = activation_scale * s1_row[g + 2u];
                scale1_3 = activation_scale * s1_row[g + 3u];
            }

            dot0_0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w0_0))
                   + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w0_0))
                   + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w0_1))
                   + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w0_1));
            dot0_1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w0_2))
                   + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w0_2))
                   + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w0_3))
                   + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w0_3));
            dot0_2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w0_4))
                   + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w0_4))
                   + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w0_5))
                   + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w0_5));
            dot0_3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w0_6))
                   + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w0_6))
                   + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w0_7))
                   + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w0_7));

            dot1_0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w1_0))
                   + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w1_0))
                   + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w1_1))
                   + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w1_1));
            dot1_1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w1_2))
                   + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w1_2))
                   + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w1_3))
                   + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w1_3));
            dot1_2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w1_4))
                   + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w1_4))
                   + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w1_5))
                   + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w1_5));
            dot1_3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w1_6))
                   + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w1_6))
                   + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w1_7))
                   + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w1_7));
        }

        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0_0 += vx_cuda_gemv_shuffle_down_i32(dot0_0, delta);
            dot0_1 += vx_cuda_gemv_shuffle_down_i32(dot0_1, delta);
            dot0_2 += vx_cuda_gemv_shuffle_down_i32(dot0_2, delta);
            dot0_3 += vx_cuda_gemv_shuffle_down_i32(dot0_3, delta);
            dot1_0 += vx_cuda_gemv_shuffle_down_i32(dot1_0, delta);
            dot1_1 += vx_cuda_gemv_shuffle_down_i32(dot1_1, delta);
            dot1_2 += vx_cuda_gemv_shuffle_down_i32(dot1_2, delta);
            dot1_3 += vx_cuda_gemv_shuffle_down_i32(dot1_3, delta);
        }

        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0_0 * scale0_0;
            sum0 += (float)dot0_1 * scale0_1;
            sum0 += (float)dot0_2 * scale0_2;
            sum0 += (float)dot0_3 * scale0_3;
            sum1 += (float)dot1_0 * scale1_0;
            sum1 += (float)dot1_1 * scale1_1;
            sum1 += (float)dot1_2 * scale1_2;
            sum1 += (float)dot1_3 * scale1_3;
        }
    }

    for (; g < full_groups; g++) {
        int dot0 = 0, dot1 = 0;
        if (column < (vx_u64)output_width) {
            VxCudaUint4 act = ((const VxCudaUint4*)(vx_packed_tile_shared + g * 32u))[sub_lane];
            const vx_u32* wp0 = w0_row + g * 16u + sub_lane * 2u;
            const vx_u32* wp1 = w1_row + g * 16u + sub_lane * 2u;
            dot0 = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp0[0]))
                 + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp0[0]))
                 + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp0[1]))
                 + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp0[1]));
            dot1 = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp1[0]))
                 + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp1[0]))
                 + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp1[1]))
                 + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp1[1]));
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
        }
        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0 * (activation_scale * s0_row[g]);
            sum1 += (float)dot1 * (activation_scale * s1_row[g]);
        }
    }

    for (; g < groups; g++) {
        int dot0 = 0, dot1 = 0;
        if (column < (vx_u64)output_width) {
            for (vx_u32 chunk = 0u; chunk < 4u; chunk++) {
                vx_u32 word = sub_lane * 4u + chunk;
                vx_u32 activation = vx_packed_tile_shared[g * 32u + word];
                vx_u32 k = g * 128u + word * 4u;
                dot0 += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w0_row, input_width, k));
                dot1 += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w1_row, input_width, k));
            }
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
        }
        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0 * (activation_scale * s0_row[g]);
            sum1 += (float)dot1 * (activation_scale * s1_row[g]);
        }
    }

    if (!sub_lane && column < (vx_u64)output_width) {
        if (bias0) sum0 += bias0[column];
        if (bias1) sum1 += bias1[column];
        output0[column] = sum0;
        output1[column] = sum1;
    }
}

/* Adjacent same-input PackedLinear nodes with fused SwiGLU: computes Gate and Up
 * projections and immediately evaluates silu(gate) * up in registers, writing all
 * four logical outputs without extra kernel launches or memory round trips. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_linear_pair_silu_mul_decode_a8(
        const float* input,
        const vx_u32* packed_weights0, const float* weight_scales0, const float* bias0, float* output0,
        const vx_u32* packed_weights1, const float* weight_scales1, const float* bias1, float* output1,
        float* output_silu, float* output_product,
        vx_u32 input_width, vx_u32 output_width, vx_u32 weight_bits, vx_u32 activation_bits,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input,
        packed_weights0, weight_scales0, bias0, output0,
        packed_weights1, weight_scales1, bias1, output1,
        output_silu, output_product,
        input_width, output_width, weight_bits, activation_bits);
    (void)weight_bits;

    float* maxima = (float*)vx_packed_tile_shared;
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 sub_lane = lane & 7u;
    vx_u64 column_tiles = ((vx_u64)output_width + 31u) / 32u;
    vx_u64 column = ((vx_u64)blockIdx.x % column_tiles) * 32u + warp * 4u + (lane >> 3u);

    if (activation_bits == 16u || activation_bits == 32u) {
        vx_u32 groups = (input_width + 127u) >> 7u;
        bool valid_col = column < (vx_u64)output_width;
        vx_u32 invalid = vx_cuda_stage_input_float(input, input_width, groups, activation_bits);
        if (invalid) {
            if (!sub_lane && valid_col) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                if (output0) output0[column] = nan.value;
                if (output1) output1[column] = nan.value;
                if (output_silu) output_silu[column] = nan.value;
                if (output_product) output_product[column] = nan.value;
            }
            return;
        }
        vx_u64 stride = (weight_bits == 8u) ? ((input_width + 3u) >> 2u) : ((input_width + 7u) >> 3u);
        const vx_u32* w0_row = valid_col ? packed_weights0 + column * stride : 0;
        const vx_u32* w1_row = valid_col ? packed_weights1 + column * stride : 0;
        const float* s0_row = valid_col ? weight_scales0 + column * groups : 0;
        const float* s1_row = valid_col ? weight_scales1 + column * groups : 0;
        float sum0 = 0.0f, sum1 = 0.0f;
        vx_cuda_gemv_pair_float_dispatch(w0_row, s0_row, w1_row, s1_row,
            input_width, groups, sub_lane, valid_col, weight_bits, activation_bits,
            &sum0, &sum1);
        if (!sub_lane && valid_col) {
            float val0 = sum0 + (bias0 ? bias0[column] : 0.0f);
            float val1 = sum1 + (bias1 ? bias1[column] : 0.0f);
            if (output0) output0[column] = val0;
            if (output1) output1[column] = val1;
            float silu_val = val0 * vx_sigmoid(val0);
            if (output_silu) output_silu[column] = silu_val;
            if (output_product) output_product[column] = silu_val * val1;
        }
        return;
    }

    float maximum = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float val = input[k];
        maximum = !(val <= 3.402823466e38f && val >= -3.402823466e38f) ?
            __builtin_inff() : vx_max(maximum, vx_abs(val));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
    if (!lane) maxima[warp] = maximum;
    vx_sync_threads();
    if (!warp) {
        maximum = lane < 8u ? maxima[lane] : 0.0f;
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
        if (!lane) maxima[0] = maximum;
    }
    vx_sync_threads();
    maximum = maxima[0];
    vx_sync_threads();

    float activation_scale = maximum > 0.0f ?
        vx_max(maximum / 127.0f, 1.1754943508222875e-38f) : 1.0f;
    if (activation_scale == __builtin_inff()) {
        if (!sub_lane && column < (vx_u64)output_width) {
            union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
            if (output0) output0[column] = nan.value;
            if (output1) output1[column] = nan.value;
            if (output_silu) output_silu[column] = nan.value;
            if (output_product) output_product[column] = nan.value;
        }
        return;
    }

    vx_u32 groups = (input_width + 127u) >> 7u;
    for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
        vx_u32 value = (vx_u32)(k < input_width ? vx_packed_cuda_quantize(input[k], activation_scale) : 0) & 255u;
        for (vx_u32 shift = 1u; shift < 4u; shift <<= 1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value), "r"(shift));
            value |= other << (shift * 8u);
        }
        if (!(lane & 3u)) vx_packed_tile_shared[k >> 2u] = value;
    }
    vx_sync_threads();

    float sum0 = 0.0f, sum1 = 0.0f;
    vx_u32 full_groups = input_width >> 7u;
    vx_u64 stride = (input_width + 7u) >> 3u;
    const vx_u32* w0_row = column < (vx_u64)output_width ? packed_weights0 + column * stride : 0;
    const vx_u32* w1_row = column < (vx_u64)output_width ? packed_weights1 + column * stride : 0;
    const float* s0_row = column < (vx_u64)output_width ? weight_scales0 + column * groups : 0;
    const float* s1_row = column < (vx_u64)output_width ? weight_scales1 + column * groups : 0;

    vx_u32 g = 0u;
    for (; g + 3u < full_groups; g += 4u) {
        int dot0_0 = 0, dot0_1 = 0, dot0_2 = 0, dot0_3 = 0;
        int dot1_0 = 0, dot1_1 = 0, dot1_2 = 0, dot1_3 = 0;
        float scale0_0 = 0.0f, scale0_1 = 0.0f, scale0_2 = 0.0f, scale0_3 = 0.0f;
        float scale1_0 = 0.0f, scale1_1 = 0.0f, scale1_2 = 0.0f, scale1_3 = 0.0f;

        if (column < (vx_u64)output_width) {
            VxCudaUint4 act0 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 0u) * 32u))[sub_lane];
            VxCudaUint4 act1 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 1u) * 32u))[sub_lane];
            VxCudaUint4 act2 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 2u) * 32u))[sub_lane];
            VxCudaUint4 act3 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 3u) * 32u))[sub_lane];

            const vx_u32* wp0 = w0_row + g * 16u + sub_lane * 2u;
            vx_u32 p0_0 = wp0[0], p0_1 = wp0[1];
            vx_u32 p0_2 = wp0[16], p0_3 = wp0[17];
            vx_u32 p0_4 = wp0[32], p0_5 = wp0[33];
            vx_u32 p0_6 = wp0[48], p0_7 = wp0[49];

            const vx_u32* wp1 = w1_row + g * 16u + sub_lane * 2u;
            vx_u32 p1_0 = wp1[0], p1_1 = wp1[1];
            vx_u32 p1_2 = wp1[16], p1_3 = wp1[17];
            vx_u32 p1_4 = wp1[32], p1_5 = wp1[33];
            vx_u32 p1_6 = wp1[48], p1_7 = wp1[49];

            if (!sub_lane) {
                scale0_0 = activation_scale * s0_row[g + 0u];
                scale0_1 = activation_scale * s0_row[g + 1u];
                scale0_2 = activation_scale * s0_row[g + 2u];
                scale0_3 = activation_scale * s0_row[g + 3u];
                scale1_0 = activation_scale * s1_row[g + 0u];
                scale1_1 = activation_scale * s1_row[g + 1u];
                scale1_2 = activation_scale * s1_row[g + 2u];
                scale1_3 = activation_scale * s1_row[g + 3u];
            }

            dot0_0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(p0_0))
                   + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(p0_0))
                   + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(p0_1))
                   + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(p0_1));
            dot1_0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(p1_0))
                   + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(p1_0))
                   + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(p1_1))
                   + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(p1_1));

            dot0_1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(p0_2))
                   + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(p0_2))
                   + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(p0_3))
                   + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(p0_3));
            dot1_1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(p1_2))
                   + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(p1_2))
                   + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(p1_3))
                   + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(p1_3));

            dot0_2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(p0_4))
                   + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(p0_4))
                   + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(p0_5))
                   + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(p0_5));
            dot1_2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(p1_4))
                   + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(p1_4))
                   + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(p1_5))
                   + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(p1_5));

            dot0_3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(p0_6))
                   + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(p0_6))
                   + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(p0_7))
                   + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(p0_7));
            dot1_3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(p1_6))
                   + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(p1_6))
                   + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(p1_7))
                   + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(p1_7));
        }

        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0_0 += vx_cuda_gemv_shuffle_down_i32(dot0_0, delta);
            dot1_0 += vx_cuda_gemv_shuffle_down_i32(dot1_0, delta);
            dot0_1 += vx_cuda_gemv_shuffle_down_i32(dot0_1, delta);
            dot1_1 += vx_cuda_gemv_shuffle_down_i32(dot1_1, delta);
            dot0_2 += vx_cuda_gemv_shuffle_down_i32(dot0_2, delta);
            dot1_2 += vx_cuda_gemv_shuffle_down_i32(dot1_2, delta);
            dot0_3 += vx_cuda_gemv_shuffle_down_i32(dot0_3, delta);
            dot1_3 += vx_cuda_gemv_shuffle_down_i32(dot1_3, delta);
        }

        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0_0 * scale0_0;
            sum0 += (float)dot0_1 * scale0_1;
            sum0 += (float)dot0_2 * scale0_2;
            sum0 += (float)dot0_3 * scale0_3;
            sum1 += (float)dot1_0 * scale1_0;
            sum1 += (float)dot1_1 * scale1_1;
            sum1 += (float)dot1_2 * scale1_2;
            sum1 += (float)dot1_3 * scale1_3;
        }
    }

    for (; g < full_groups; g++) {
        int dot0 = 0, dot1 = 0;
        if (column < (vx_u64)output_width) {
            VxCudaUint4 act = ((const VxCudaUint4*)(vx_packed_tile_shared + g * 32u))[sub_lane];
            const vx_u32* wp0 = w0_row + g * 16u + sub_lane * 2u;
            const vx_u32* wp1 = w1_row + g * 16u + sub_lane * 2u;
            dot0 = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp0[0]))
                 + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp0[0]))
                 + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp0[1]))
                 + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp0[1]));
            dot1 = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp1[0]))
                 + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp1[0]))
                 + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp1[1]))
                 + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp1[1]));
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
        }
        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0 * (activation_scale * s0_row[g]);
            sum1 += (float)dot1 * (activation_scale * s1_row[g]);
        }
    }

    for (; g < groups; g++) {
        int dot0 = 0, dot1 = 0;
        if (column < (vx_u64)output_width) {
            for (vx_u32 chunk = 0u; chunk < 4u; chunk++) {
                vx_u32 word = sub_lane * 4u + chunk;
                vx_u32 activation = vx_packed_tile_shared[g * 32u + word];
                vx_u32 k = g * 128u + word * 4u;
                dot0 += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w0_row, input_width, k));
                dot1 += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w1_row, input_width, k));
            }
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
        }
        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0 * (activation_scale * s0_row[g]);
            sum1 += (float)dot1 * (activation_scale * s1_row[g]);
        }
    }

    if (!sub_lane && column < (vx_u64)output_width) {
        float val0 = sum0 + (bias0 ? bias0[column] : 0.0f);
        float val1 = sum1 + (bias1 ? bias1[column] : 0.0f);
        if (output0) output0[column] = val0;
        if (output1) output1[column] = val1;
        float silu_val = val0 * vx_sigmoid(val0);
        if (output_silu) output_silu[column] = silu_val;
        if (output_product) output_product[column] = silu_val * val1;
    }
}

/* Fused PackedLinear decode with residual addition: computes dense projection
 * and accumulates with residual in a single pass without extra round trips. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_linear_add_decode_a8(
        const float* input,
        const vx_u32* packed_weights, const float* weight_scales, const float* bias, float* output,
        const float* residual, float* final_output,
        vx_u32 input_width, vx_u32 output_width, vx_u32 weight_bits, vx_u32 activation_bits,
        vx_u32 residual_first,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, packed_weights, weight_scales, bias, output,
        residual, final_output,
        input_width, output_width, weight_bits, activation_bits, residual_first);
    (void)weight_bits;

    float* maxima = (float*)vx_packed_tile_shared;
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 sub_lane = lane & 7u;
    vx_u64 column_tiles = ((vx_u64)output_width + 31u) / 32u;
    vx_u64 column = ((vx_u64)blockIdx.x % column_tiles) * 32u + warp * 4u + (lane >> 3u);

    if (activation_bits == 16u || activation_bits == 32u) {
        vx_u32 groups = (input_width + 127u) >> 7u;
        bool valid_col = column < (vx_u64)output_width;
        vx_u32 invalid = vx_cuda_stage_input_float(input, input_width, groups, activation_bits);
        if (invalid) {
            if (!sub_lane && valid_col) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output[column] = nan.value;
                if (final_output) final_output[column] = nan.value;
            }
            return;
        }
        vx_u64 stride = (weight_bits == 8u) ? ((input_width + 3u) >> 2u) : ((input_width + 7u) >> 3u);
        const vx_u32* w_row = valid_col ? packed_weights + column * stride : 0;
        const float* s_row = valid_col ? weight_scales + column * groups : 0;
        float sum = vx_cuda_gemv_single_float_dispatch(
            w_row, s_row, input_width, groups, sub_lane, valid_col, weight_bits, activation_bits);
        if (!sub_lane && valid_col) {
            float val = sum + (bias ? bias[column] : 0.0f);
            output[column] = val;
            if (residual && final_output)
                final_output[column] = residual_first ? vx_add_rn(residual[column], val)
                                                      : vx_add_rn(val, residual[column]);
        }
        return;
    }

    float maximum = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float val = input[k];
        maximum = !(val <= 3.402823466e38f && val >= -3.402823466e38f) ?
            __builtin_inff() : vx_max(maximum, vx_abs(val));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
    if (!lane) maxima[warp] = maximum;
    vx_sync_threads();
    if (!warp) {
        maximum = lane < 8u ? maxima[lane] : 0.0f;
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
        if (!lane) maxima[0] = maximum;
    }
    vx_sync_threads();
    maximum = maxima[0];
    vx_sync_threads();

    float activation_scale = maximum > 0.0f ?
        vx_max(maximum / 127.0f, 1.1754943508222875e-38f) : 1.0f;
    if (activation_scale == __builtin_inff()) {
        if (!sub_lane && column < (vx_u64)output_width) {
            union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
            output[column] = nan.value;
            if (final_output) final_output[column] = nan.value;
        }
        return;
    }

    vx_u32 groups = (input_width + 127u) >> 7u;
    for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
        vx_u32 value = (vx_u32)(k < input_width ? vx_packed_cuda_quantize(input[k], activation_scale) : 0) & 255u;
        for (vx_u32 shift = 1u; shift < 4u; shift <<= 1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value), "r"(shift));
            value |= other << (shift * 8u);
        }
        if (!(lane & 3u)) vx_packed_tile_shared[k >> 2u] = value;
    }
    vx_sync_threads();

    float sum = 0.0f;
    vx_u32 full_groups = input_width >> 7u;
    vx_u64 stride = (input_width + 7u) >> 3u;
    const vx_u32* w_row = column < (vx_u64)output_width ? packed_weights + column * stride : 0;
    const float* s_row = column < (vx_u64)output_width ? weight_scales + column * groups : 0;

    vx_u32 g = 0u;
    for (; g + 3u < full_groups; g += 4u) {
        int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
        float scale0 = 0.0f, scale1 = 0.0f, scale2 = 0.0f, scale3 = 0.0f;

        if (column < (vx_u64)output_width) {
            VxCudaUint4 act0 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 0u) * 32u))[sub_lane];
            VxCudaUint4 act1 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 1u) * 32u))[sub_lane];
            VxCudaUint4 act2 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 2u) * 32u))[sub_lane];
            VxCudaUint4 act3 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 3u) * 32u))[sub_lane];

            const vx_u32* wp = w_row + g * 16u + sub_lane * 2u;
            vx_u32 w0 = wp[0], w1 = wp[1];
            vx_u32 w2 = wp[16], w3 = wp[17];
            vx_u32 w4 = wp[32], w5 = wp[33];
            vx_u32 w6 = wp[48], w7 = wp[49];

            if (!sub_lane) {
                scale0 = activation_scale * s_row[g + 0u];
                scale1 = activation_scale * s_row[g + 1u];
                scale2 = activation_scale * s_row[g + 2u];
                scale3 = activation_scale * s_row[g + 3u];
            }

            dot0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w0))
                 + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w0))
                 + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w1))
                 + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w1));
            dot1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w2))
                 + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w2))
                 + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w3))
                 + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w3));
            dot2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w4))
                 + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w4))
                 + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w5))
                 + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w5));
            dot3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w6))
                 + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w6))
                 + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w7))
                 + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w7));
        }

        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
            dot2 += vx_cuda_gemv_shuffle_down_i32(dot2, delta);
            dot3 += vx_cuda_gemv_shuffle_down_i32(dot3, delta);
        }

        if (!sub_lane && column < (vx_u64)output_width) {
            sum += (float)dot0 * scale0;
            sum += (float)dot1 * scale1;
            sum += (float)dot2 * scale2;
            sum += (float)dot3 * scale3;
        }
    }

    for (; g < full_groups; g++) {
        int dot = 0;
        if (column < (vx_u64)output_width) {
            VxCudaUint4 act = ((const VxCudaUint4*)(vx_packed_tile_shared + g * 32u))[sub_lane];
            const vx_u32* wp = w_row + g * 16u + sub_lane * 2u;
            dot = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp[0]))
                + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp[0]))
                + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp[1]))
                + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp[1]));
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            dot += vx_cuda_gemv_shuffle_down_i32(dot, delta);
        if (!sub_lane && column < (vx_u64)output_width)
            sum += (float)dot * (activation_scale * s_row[g]);
    }

    for (; g < groups; g++) {
        int dot = 0;
        if (column < (vx_u64)output_width) {
            for (vx_u32 chunk = 0u; chunk < 4u; chunk++) {
                vx_u32 word = sub_lane * 4u + chunk;
                vx_u32 activation = vx_packed_tile_shared[g * 32u + word];
                vx_u32 k = g * 128u + word * 4u;
                dot += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w_row, input_width, k));
            }
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            dot += vx_cuda_gemv_shuffle_down_i32(dot, delta);
        if (!sub_lane && column < (vx_u64)output_width)
            sum += (float)dot * (activation_scale * s_row[g]);
    }

    if (!sub_lane && column < (vx_u64)output_width) {
        float val = sum + (bias ? bias[column] : 0.0f);
        output[column] = val;
        if (residual && final_output)
            final_output[column] = residual_first ? vx_add_rn(residual[column], val)
                                                  : vx_add_rn(val, residual[column]);
    }
}

/* Three adjacent same-input PackedLinear nodes (e.g. Q, K, V) share the single
 * input activation quantization in shared memory. CTAs are partitioned across
 * the three output matrices (tiles0 for Q, tiles1 for K, tiles2 for V). */
extern "C" VX_CUDA_GLOBAL void vx_cuda_packed_linear_triple_decode_a8(
        const float* input,
        const vx_u32* packed_weights0, const float* weight_scales0, const float* bias0, float* output0, vx_u32 width0,
        const vx_u32* packed_weights1, const float* weight_scales1, const float* bias1, float* output1, vx_u32 width1,
        const vx_u32* packed_weights2, const float* weight_scales2, const float* bias2, float* output2, vx_u32 width2,
        vx_u32 input_width, vx_u32 weight_bits, vx_u32 activation_bits,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input,
        packed_weights0, weight_scales0, bias0, output0, width0,
        packed_weights1, weight_scales1, bias1, output1, width1,
        packed_weights2, weight_scales2, bias2, output2, width2,
        input_width, weight_bits, activation_bits);
    (void)weight_bits;

    float* maxima = (float*)vx_packed_tile_shared;
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 sub_lane = lane & 7u;

    if (activation_bits == 16u || activation_bits == 32u) {
        vx_u32 groups = (input_width + 127u) >> 7u;
        vx_u32 tiles0 = (width0 + 31u) / 32u;
        vx_u32 tiles1 = (width1 + 31u) / 32u;
        const vx_u32* packed_weights;
        const float* weight_scales;
        const float* bias_vec;
        float* output_vec;
        vx_u32 cur_out_width;
        vx_u64 cur_block_col;
        if (blockIdx.x < tiles0) {
            packed_weights = packed_weights0;
            weight_scales = weight_scales0;
            bias_vec = bias0;
            output_vec = output0;
            cur_out_width = width0;
            cur_block_col = (vx_u64)blockIdx.x * 32u;
        } else if (blockIdx.x < tiles0 + tiles1) {
            packed_weights = packed_weights1;
            weight_scales = weight_scales1;
            bias_vec = bias1;
            output_vec = output1;
            cur_out_width = width1;
            cur_block_col = (vx_u64)(blockIdx.x - tiles0) * 32u;
        } else {
            packed_weights = packed_weights2;
            weight_scales = weight_scales2;
            bias_vec = bias2;
            output_vec = output2;
            cur_out_width = width2;
            cur_block_col = (vx_u64)(blockIdx.x - (tiles0 + tiles1)) * 32u;
        }
        vx_u64 column = cur_block_col + warp * 4u + (lane >> 3u);
        bool valid_col = column < (vx_u64)cur_out_width;
        vx_u32 invalid = vx_cuda_stage_input_float(input, input_width, groups, activation_bits);
        if (invalid) {
            if (!sub_lane && valid_col) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output_vec[column] = nan.value;
            }
            return;
        }
        vx_u64 stride = (weight_bits == 8u) ? ((input_width + 3u) >> 2u) : ((input_width + 7u) >> 3u);
        const vx_u32* w_row = valid_col ? packed_weights + column * stride : 0;
        const float* s_row = valid_col ? weight_scales + column * groups : 0;
        float sum = vx_cuda_gemv_single_float_dispatch(
            w_row, s_row, input_width, groups, sub_lane, valid_col, weight_bits, activation_bits);
        if (!sub_lane && valid_col) {
            output_vec[column] = sum + (bias_vec ? bias_vec[column] : 0.0f);
        }
        return;
    }

    float maximum = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float val = input[k];
        maximum = !(val <= 3.402823466e38f && val >= -3.402823466e38f) ?
            __builtin_inff() : vx_max(maximum, vx_abs(val));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
    if (!lane) maxima[warp] = maximum;
    vx_sync_threads();
    if (!warp) {
        maximum = lane < 8u ? maxima[lane] : 0.0f;
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
        if (!lane) maxima[0] = maximum;
    }
    vx_sync_threads();
    maximum = maxima[0];
    vx_sync_threads();

    float activation_scale = maximum > 0.0f ?
        vx_max(maximum / 127.0f, 1.1754943508222875e-38f) : 1.0f;
    if (activation_scale == __builtin_inff()) {
        vx_u32 tiles0 = (width0 + 31u) / 32u;
        vx_u32 tiles1 = (width1 + 31u) / 32u;
        if (blockIdx.x < tiles0) {
            vx_u64 col = (vx_u64)blockIdx.x * 32u + warp * 4u + (lane >> 3u);
            if (!sub_lane && col < (vx_u64)width0) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output0[col] = nan.value;
            }
        } else if (blockIdx.x < tiles0 + tiles1) {
            vx_u64 col = (vx_u64)(blockIdx.x - tiles0) * 32u + warp * 4u + (lane >> 3u);
            if (!sub_lane && col < (vx_u64)width1) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output1[col] = nan.value;
            }
        } else {
            vx_u64 col = (vx_u64)(blockIdx.x - (tiles0 + tiles1)) * 32u + warp * 4u + (lane >> 3u);
            if (!sub_lane && col < (vx_u64)width2) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output2[col] = nan.value;
            }
        }
        return;
    }

    vx_u32 groups = (input_width + 127u) >> 7u;
    for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
        vx_u32 value = (vx_u32)(k < input_width ? vx_packed_cuda_quantize(input[k], activation_scale) : 0) & 255u;
        for (vx_u32 shift = 1u; shift < 4u; shift <<= 1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value), "r"(shift));
            value |= other << (shift * 8u);
        }
        if (!(lane & 3u)) vx_packed_tile_shared[k >> 2u] = value;
    }
    vx_sync_threads();

    vx_u32 tiles0 = (width0 + 31u) / 32u;
    vx_u32 tiles1 = (width1 + 31u) / 32u;
    const vx_u32* packed_weights;
    const float* weight_scales;
    const float* bias_vec;
    float* output_vec;
    vx_u32 cur_out_width;
    vx_u64 cur_block_col;

    if (blockIdx.x < tiles0) {
        packed_weights = packed_weights0;
        weight_scales = weight_scales0;
        bias_vec = bias0;
        output_vec = output0;
        cur_out_width = width0;
        cur_block_col = (vx_u64)blockIdx.x * 32u;
    } else if (blockIdx.x < tiles0 + tiles1) {
        packed_weights = packed_weights1;
        weight_scales = weight_scales1;
        bias_vec = bias1;
        output_vec = output1;
        cur_out_width = width1;
        cur_block_col = (vx_u64)(blockIdx.x - tiles0) * 32u;
    } else {
        packed_weights = packed_weights2;
        weight_scales = weight_scales2;
        bias_vec = bias2;
        output_vec = output2;
        cur_out_width = width2;
        cur_block_col = (vx_u64)(blockIdx.x - (tiles0 + tiles1)) * 32u;
    }

    vx_u64 column = cur_block_col + warp * 4u + (lane >> 3u);
    float sum = 0.0f;
    vx_u32 full_groups = input_width >> 7u;
    vx_u64 stride = (input_width + 7u) >> 3u;
    const vx_u32* w_row = column < (vx_u64)cur_out_width ? packed_weights + column * stride : 0;
    const float* s_row = column < (vx_u64)cur_out_width ? weight_scales + column * groups : 0;

    vx_u32 g = 0u;
    for (; g + 3u < full_groups; g += 4u) {
        int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;
        float scale0 = 0.0f, scale1 = 0.0f, scale2 = 0.0f, scale3 = 0.0f;

        if (column < (vx_u64)cur_out_width) {
            VxCudaUint4 act0 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 0u) * 32u))[sub_lane];
            VxCudaUint4 act1 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 1u) * 32u))[sub_lane];
            VxCudaUint4 act2 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 2u) * 32u))[sub_lane];
            VxCudaUint4 act3 = ((const VxCudaUint4*)(vx_packed_tile_shared + (g + 3u) * 32u))[sub_lane];

            const vx_u32* wp = w_row + g * 16u + sub_lane * 2u;
            vx_u32 w0 = wp[0], w1 = wp[1];
            vx_u32 w2 = wp[16], w3 = wp[17];
            vx_u32 w4 = wp[32], w5 = wp[33];
            vx_u32 w6 = wp[48], w7 = wp[49];

            if (!sub_lane) {
                scale0 = activation_scale * s_row[g + 0u];
                scale1 = activation_scale * s_row[g + 1u];
                scale2 = activation_scale * s_row[g + 2u];
                scale3 = activation_scale * s_row[g + 3u];
            }

            dot0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w0))
                 + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w0))
                 + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w1))
                 + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w1));
            dot1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w2))
                 + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w2))
                 + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w3))
                 + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w3));
            dot2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w4))
                 + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w4))
                 + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w5))
                 + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w5));
            dot3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w6))
                 + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w6))
                 + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w7))
                 + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w7));
        }

        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
            dot2 += vx_cuda_gemv_shuffle_down_i32(dot2, delta);
            dot3 += vx_cuda_gemv_shuffle_down_i32(dot3, delta);
        }

        if (!sub_lane && column < (vx_u64)cur_out_width) {
            sum += (float)dot0 * scale0;
            sum += (float)dot1 * scale1;
            sum += (float)dot2 * scale2;
            sum += (float)dot3 * scale3;
        }
    }

    for (; g < full_groups; g++) {
        int dot = 0;
        if (column < (vx_u64)cur_out_width) {
            VxCudaUint4 act = ((const VxCudaUint4*)(vx_packed_tile_shared + g * 32u))[sub_lane];
            const vx_u32* wp = w_row + g * 16u + sub_lane * 2u;
            dot = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp[0]))
                + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp[0]))
                + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp[1]))
                + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp[1]));
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            dot += vx_cuda_gemv_shuffle_down_i32(dot, delta);
        if (!sub_lane && column < (vx_u64)cur_out_width)
            sum += (float)dot * (activation_scale * s_row[g]);
    }

    for (; g < groups; g++) {
        int dot = 0;
        if (column < (vx_u64)cur_out_width) {
            for (vx_u32 chunk = 0u; chunk < 4u; chunk++) {
                vx_u32 word = sub_lane * 4u + chunk;
                vx_u32 activation = vx_packed_tile_shared[g * 32u + word];
                vx_u32 k = g * 128u + word * 4u;
                dot += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w_row, input_width, k));
            }
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            dot += vx_cuda_gemv_shuffle_down_i32(dot, delta);
        if (!sub_lane && column < (vx_u64)cur_out_width)
            sum += (float)dot * (activation_scale * s_row[g]);
    }

    if (!sub_lane && column < (vx_u64)cur_out_width) {
        float val = sum + (bias_vec ? bias_vec[column] : 0.0f);
        output_vec[column] = val;
    }
}

/* RMSNorm fused with Three adjacent same-input PackedLinear nodes (Norm + Q, K, V).
 * Evaluates RMSNorm, persists norm_output via block 0, dynamically quantizes
 * normalized activations into shared memory, and executes Triple GEMV. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_rmsnorm_packed_linear_triple_decode_a8(
        const float* input, const float* norm_weight, float* norm_output, float epsilon,
        const vx_u32* packed_weights0, const float* weight_scales0, const float* bias0, float* output0, vx_u32 width0,
        const vx_u32* packed_weights1, const float* weight_scales1, const float* bias1, float* output1, vx_u32 width1,
        const vx_u32* packed_weights2, const float* weight_scales2, const float* bias2, float* output2, vx_u32 width2,
        vx_u32 input_width, vx_u32 weight_bits, vx_u32 activation_bits,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, norm_weight, norm_output, epsilon,
        packed_weights0, weight_scales0, bias0, output0, width0,
        packed_weights1, weight_scales1, bias1, output1, width1,
        packed_weights2, weight_scales2, bias2, output2, width2,
        input_width, weight_bits, activation_bits);
    (void)weight_bits;

    float* maxima = (float*)vx_packed_tile_shared;
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 sub_lane = lane & 7u;

    float sum_squares = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float val = input[k];
        sum_squares = vx_add_rn(sum_squares, vx_mul_rn(val, val));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        sum_squares = vx_add_rn(sum_squares, vx_cuda_gemv_shuffle_down(sum_squares, delta));
    if (!lane) maxima[warp] = sum_squares;
    vx_sync_threads();
    if (!threadIdx.x) {
        float total = maxima[0];
        for (vx_u32 w = 1u; w < 8u; ++w)
            total = vx_add_rn(total, maxima[w]);
        maxima[0] = 1.0f / vx_sqrt(vx_add_rn(total / (float)input_width, epsilon));
    }
    vx_sync_threads();
    float inverse = maxima[0];
    vx_sync_threads();

    if (activation_bits == 16u || activation_bits == 32u) {
        vx_u32 groups = (input_width + 127u) >> 7u;
        vx_u32 tiles0 = (width0 + 31u) / 32u;
        vx_u32 tiles1 = (width1 + 31u) / 32u;
        const vx_u32* w_mat;
        const float* s_mat;
        const float* bias_vec;
        float* output_vec;
        vx_u32 cur_out_width, cur_tile;
        if (blockIdx.x < tiles0) {
            w_mat = packed_weights0;
            s_mat = weight_scales0;
            bias_vec = bias0;
            output_vec = output0;
            cur_out_width = width0;
            cur_tile = blockIdx.x;
        } else if (blockIdx.x < tiles0 + tiles1) {
            w_mat = packed_weights1;
            s_mat = weight_scales1;
            bias_vec = bias1;
            output_vec = output1;
            cur_out_width = width1;
            cur_tile = blockIdx.x - tiles0;
        } else {
            w_mat = packed_weights2;
            s_mat = weight_scales2;
            bias_vec = bias2;
            output_vec = output2;
            cur_out_width = width2;
            cur_tile = blockIdx.x - (tiles0 + tiles1);
        }
        vx_u64 column = (vx_u64)cur_tile * 32u + warp * 4u + (lane >> 3u);
        bool valid_col = column < (vx_u64)cur_out_width;
        vx_u32 invalid = vx_cuda_stage_rmsnorm_float(input, norm_weight, norm_output, inverse, input_width, groups, activation_bits);
        if (invalid) {
            if (!sub_lane && valid_col) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output_vec[column] = nan.value;
            }
            return;
        }
        vx_u64 stride = (weight_bits == 8u) ? ((input_width + 3u) >> 2u) : ((input_width + 7u) >> 3u);
        const vx_u32* w_row = valid_col ? w_mat + column * stride : 0;
        const float* s_row = valid_col ? s_mat + column * (vx_u64)groups : 0;
        float sum = vx_cuda_gemv_single_float_dispatch(
            w_row, s_row, input_width, groups, sub_lane, valid_col, weight_bits, activation_bits);
        if (!sub_lane && valid_col) {
            output_vec[column] = sum + (bias_vec ? bias_vec[column] : 0.0f);
        }
        return;
    }

    float maximum = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float normed = vx_mul_rn(vx_mul_rn(input[k], inverse), norm_weight[k]);
        if (norm_output && !blockIdx.x) norm_output[k] = normed;
        maximum = !(normed <= 3.402823466e38f && normed >= -3.402823466e38f) ?
            __builtin_inff() : vx_max(maximum, vx_abs(normed));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
    if (!lane) maxima[warp] = maximum;
    vx_sync_threads();
    if (!warp) {
        maximum = lane < 8u ? maxima[lane] : 0.0f;
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
        if (!lane) maxima[0] = maximum;
    }
    vx_sync_threads();
    maximum = maxima[0];
    vx_sync_threads();

    float activation_scale = maximum > 0.0f ?
        vx_max(maximum / 127.0f, 1.1754943508222875e-38f) : 1.0f;
    if (activation_scale == __builtin_inff()) {
        vx_u32 tiles0 = (width0 + 31u) / 32u;
        vx_u32 tiles1 = (width1 + 31u) / 32u;
        if (blockIdx.x < tiles0) {
            vx_u64 col = (vx_u64)blockIdx.x * 32u + warp * 4u + (lane >> 3u);
            if (!sub_lane && col < (vx_u64)width0) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output0[col] = nan.value;
            }
        } else if (blockIdx.x < tiles0 + tiles1) {
            vx_u64 col = (vx_u64)(blockIdx.x - tiles0) * 32u + warp * 4u + (lane >> 3u);
            if (!sub_lane && col < (vx_u64)width1) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output1[col] = nan.value;
            }
        } else {
            vx_u64 col = (vx_u64)(blockIdx.x - (tiles0 + tiles1)) * 32u + warp * 4u + (lane >> 3u);
            if (!sub_lane && col < (vx_u64)width2) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                output2[col] = nan.value;
            }
        }
        return;
    }

    vx_u32 groups = (input_width + 127u) >> 7u;
    for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
        float normed = k < input_width ? vx_mul_rn(vx_mul_rn(input[k], inverse), norm_weight[k]) : 0.0f;
        vx_u32 value = (vx_u32)(k < input_width ? vx_packed_cuda_quantize(normed, activation_scale) : 0) & 255u;
        for (vx_u32 shift = 1u; shift < 4u; shift <<= 1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value), "r"(shift));
            value |= other << (shift * 8u);
        }
        if (!(lane & 3u)) vx_packed_tile_shared[k >> 2u] = value;
    }
    vx_sync_threads();

    vx_u32 tiles0 = (width0 + 31u) / 32u;
    vx_u32 tiles1 = (width1 + 31u) / 32u;
    const vx_u32* w_mat;
    const float* s_mat;
    const float* bias_vec;
    float* output_vec;
    vx_u32 cur_out_width, cur_tile;

    if (blockIdx.x < tiles0) {
        w_mat = packed_weights0;
        s_mat = weight_scales0;
        bias_vec = bias0;
        output_vec = output0;
        cur_out_width = width0;
        cur_tile = blockIdx.x;
    } else if (blockIdx.x < tiles0 + tiles1) {
        w_mat = packed_weights1;
        s_mat = weight_scales1;
        bias_vec = bias1;
        output_vec = output1;
        cur_out_width = width1;
        cur_tile = blockIdx.x - tiles0;
    } else {
        w_mat = packed_weights2;
        s_mat = weight_scales2;
        bias_vec = bias2;
        output_vec = output2;
        cur_out_width = width2;
        cur_tile = blockIdx.x - (tiles0 + tiles1);
    }

    vx_u64 column = (vx_u64)cur_tile * 32u + warp * 4u + (lane >> 3u);
    vx_u32 full_groups = input_width >> 7u;
    vx_u64 stride = (input_width + 7u) >> 3u;
    const vx_u32* w_row = column < (vx_u64)cur_out_width ? w_mat + column * stride : 0;
    const float* s_row = column < (vx_u64)cur_out_width ? s_mat + column * (vx_u64)groups : 0;
    float sum = 0.0f;

    vx_u32 g = 0;
    for (; g + 3u < full_groups; g += 4u) {
        float scale0 = 0.0f, scale1 = 0.0f, scale2 = 0.0f, scale3 = 0.0f;
        int dot0 = 0, dot1 = 0, dot2 = 0, dot3 = 0;

        if (column < (vx_u64)cur_out_width) {
            const VxCudaUint4* ap = (const VxCudaUint4*)(vx_packed_tile_shared + g * 32u);
            VxCudaUint4 act0 = ap[sub_lane];
            VxCudaUint4 act1 = ap[8u + sub_lane];
            VxCudaUint4 act2 = ap[16u + sub_lane];
            VxCudaUint4 act3 = ap[24u + sub_lane];

            const vx_u32* wp = w_row + g * 16u + sub_lane * 2u;
            vx_u32 w0 = wp[0], w1 = wp[1];
            vx_u32 w2 = wp[16], w3 = wp[17];
            vx_u32 w4 = wp[32], w5 = wp[33];
            vx_u32 w6 = wp[48], w7 = wp[49];

            if (!sub_lane) {
                scale0 = activation_scale * s_row[g];
                scale1 = activation_scale * s_row[g + 1u];
                scale2 = activation_scale * s_row[g + 2u];
                scale3 = activation_scale * s_row[g + 3u];
            }

            dot0 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w0))
                 + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w0))
                 + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w1))
                 + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w1));
            dot1 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w2))
                 + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w2))
                 + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w3))
                 + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w3));
            dot2 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w4))
                 + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w4))
                 + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w5))
                 + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w5));
            dot3 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w6))
                 + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w6))
                 + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w7))
                 + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w7));
        }

        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
            dot2 += vx_cuda_gemv_shuffle_down_i32(dot2, delta);
            dot3 += vx_cuda_gemv_shuffle_down_i32(dot3, delta);
        }

        if (!sub_lane && column < (vx_u64)cur_out_width) {
            sum += (float)dot0 * scale0;
            sum += (float)dot1 * scale1;
            sum += (float)dot2 * scale2;
            sum += (float)dot3 * scale3;
        }
    }

    for (; g < full_groups; g++) {
        int dot = 0;
        if (column < (vx_u64)cur_out_width) {
            VxCudaUint4 act = ((const VxCudaUint4*)(vx_packed_tile_shared + g * 32u))[sub_lane];
            const vx_u32* wp = w_row + g * 16u + sub_lane * 2u;
            dot = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp[0]))
                + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp[0]))
                + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp[1]))
                + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp[1]));
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            dot += vx_cuda_gemv_shuffle_down_i32(dot, delta);
        if (!sub_lane && column < (vx_u64)cur_out_width)
            sum += (float)dot * (activation_scale * s_row[g]);
    }

    for (; g < groups; g++) {
        int dot = 0;
        if (column < (vx_u64)cur_out_width) {
            for (vx_u32 chunk = 0u; chunk < 4u; chunk++) {
                vx_u32 word = sub_lane * 4u + chunk;
                vx_u32 activation = vx_packed_tile_shared[g * 32u + word];
                vx_u32 k = g * 128u + word * 4u;
                dot += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w_row, input_width, k));
            }
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            dot += vx_cuda_gemv_shuffle_down_i32(dot, delta);
        if (!sub_lane && column < (vx_u64)cur_out_width)
            sum += (float)dot * (activation_scale * s_row[g]);
    }

    if (!sub_lane && column < (vx_u64)cur_out_width) {
        float val = sum + (bias_vec ? bias_vec[column] : 0.0f);
        output_vec[column] = val;
    }
}

/* RMSNorm fused with Quad SwiGLU: computes RMSNorm, persists norm_output via block 0,
 * dynamically quantizes normalized activations into shared memory, computes Gate & Up
 * GEMV projections and immediately evaluates silu(gate) * up in registers. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_rmsnorm_packed_linear_pair_silu_mul_decode_a8(
        const float* input, const float* norm_weight, float* norm_output, float epsilon,
        const vx_u32* packed_weights0, const float* weight_scales0, const float* bias0, float* output0,
        const vx_u32* packed_weights1, const float* weight_scales1, const float* bias1, float* output1,
        float* output_silu, float* output_product,
        vx_u32 input_width, vx_u32 output_width,
        vx_u32 weight_bits, vx_u32 activation_bits,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, norm_weight, norm_output, epsilon,
        packed_weights0, weight_scales0, bias0, output0,
        packed_weights1, weight_scales1, bias1, output1,
        output_silu, output_product,
        input_width, output_width,
        weight_bits, activation_bits);
    (void)weight_bits;

    float* maxima = (float*)vx_packed_tile_shared;
    vx_u32 lane = threadIdx.x & 31u, warp = threadIdx.x >> 5u;
    vx_u32 sub_lane = lane & 7u;

    float sum_squares = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float val = input[k];
        sum_squares = vx_add_rn(sum_squares, vx_mul_rn(val, val));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        sum_squares = vx_add_rn(sum_squares, vx_cuda_gemv_shuffle_down(sum_squares, delta));
    if (!lane) maxima[warp] = sum_squares;
    vx_sync_threads();
    if (!threadIdx.x) {
        float total = maxima[0];
        for (vx_u32 w = 1u; w < 8u; ++w)
            total = vx_add_rn(total, maxima[w]);
        maxima[0] = 1.0f / vx_sqrt(vx_add_rn(total / (float)input_width, epsilon));
    }
    vx_sync_threads();
    float inverse = maxima[0];
    vx_sync_threads();

    if (activation_bits == 16u || activation_bits == 32u) {
        vx_u32 groups = (input_width + 127u) >> 7u;
        vx_u64 column = (vx_u64)blockIdx.x * 32u + warp * 4u + (lane >> 3u);
        bool valid_col = column < (vx_u64)output_width;
        vx_u32 invalid = vx_cuda_stage_rmsnorm_float(input, norm_weight, norm_output, inverse, input_width, groups, activation_bits);
        if (invalid) {
            if (!sub_lane && valid_col) {
                union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
                if (output0) output0[column] = nan.value;
                if (output1) output1[column] = nan.value;
                if (output_silu) output_silu[column] = nan.value;
                if (output_product) output_product[column] = nan.value;
            }
            return;
        }
        vx_u64 stride = (weight_bits == 8u) ? ((input_width + 3u) >> 2u) : ((input_width + 7u) >> 3u);
        const vx_u32* w0_row = valid_col ? packed_weights0 + column * stride : 0;
        const float* s0_row = valid_col ? weight_scales0 + column * (vx_u64)groups : 0;
        const vx_u32* w1_row = valid_col ? packed_weights1 + column * stride : 0;
        const float* s1_row = valid_col ? weight_scales1 + column * (vx_u64)groups : 0;
        float sum0 = 0.0f, sum1 = 0.0f;
        vx_cuda_gemv_pair_float_dispatch(w0_row, s0_row, w1_row, s1_row,
            input_width, groups, sub_lane, valid_col, weight_bits, activation_bits,
            &sum0, &sum1);
        if (!sub_lane && valid_col) {
            float val0 = sum0 + (bias0 ? bias0[column] : 0.0f);
            float val1 = sum1 + (bias1 ? bias1[column] : 0.0f);
            if (output0) output0[column] = val0;
            if (output1) output1[column] = val1;
            float silu_val = val0 * vx_sigmoid(val0);
            if (output_silu) output_silu[column] = silu_val;
            if (output_product) output_product[column] = silu_val * val1;
        }
        return;
    }

    float maximum = 0.0f;
    for (vx_u32 k = threadIdx.x; k < input_width; k += 256u) {
        float normed = vx_mul_rn(vx_mul_rn(input[k], inverse), norm_weight[k]);
        if (norm_output && !blockIdx.x) norm_output[k] = normed;
        maximum = !(normed <= 3.402823466e38f && normed >= -3.402823466e38f) ?
            __builtin_inff() : vx_max(maximum, vx_abs(normed));
    }
    for (vx_u32 delta = 16u; delta; delta >>= 1u)
        maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
    if (!lane) maxima[warp] = maximum;
    vx_sync_threads();
    if (!warp) {
        maximum = lane < 8u ? maxima[lane] : 0.0f;
        for (vx_u32 delta = 4u; delta; delta >>= 1u)
            maximum = vx_max(maximum, vx_cuda_gemv_shuffle_down(maximum, delta));
        if (!lane) maxima[0] = maximum;
    }
    vx_sync_threads();
    maximum = maxima[0];
    vx_sync_threads();

    float activation_scale = maximum > 0.0f ?
        vx_max(maximum / 127.0f, 1.1754943508222875e-38f) : 1.0f;
    if (activation_scale == __builtin_inff()) {
        vx_u64 col = (vx_u64)blockIdx.x * 32u + warp * 4u + (lane >> 3u);
        if (!sub_lane && col < (vx_u64)output_width) {
            union { vx_u32 bits; float value; } nan; nan.bits = 0x7fc00000u;
            if (output0) output0[col] = nan.value;
            if (output1) output1[col] = nan.value;
            if (output_silu) output_silu[col] = nan.value;
            if (output_product) output_product[col] = nan.value;
        }
        return;
    }

    vx_u32 groups = (input_width + 127u) >> 7u;
    for (vx_u32 k = threadIdx.x; k < groups * 128u; k += 256u) {
        float normed = k < input_width ? vx_mul_rn(vx_mul_rn(input[k], inverse), norm_weight[k]) : 0.0f;
        vx_u32 value = (vx_u32)(k < input_width ? vx_packed_cuda_quantize(normed, activation_scale) : 0) & 255u;
        for (vx_u32 shift = 1u; shift < 4u; shift <<= 1u) {
            vx_u32 other;
            asm("shfl.sync.down.b32 %0, %1, %2, 31, -1;" : "=r"(other) : "r"(value), "r"(shift));
            value |= other << (shift * 8u);
        }
        if (!(lane & 3u)) vx_packed_tile_shared[k >> 2u] = value;
    }
    vx_sync_threads();

    vx_u64 column = (vx_u64)blockIdx.x * 32u + warp * 4u + (lane >> 3u);
    vx_u32 full_groups = input_width >> 7u;
    vx_u64 stride = (input_width + 7u) >> 3u;
    const vx_u32* w0_row = column < (vx_u64)output_width ? packed_weights0 + column * stride : 0;
    const float* s0_row = column < (vx_u64)output_width ? weight_scales0 + column * (vx_u64)groups : 0;
    const vx_u32* w1_row = column < (vx_u64)output_width ? packed_weights1 + column * stride : 0;
    const float* s1_row = column < (vx_u64)output_width ? weight_scales1 + column * (vx_u64)groups : 0;

    float sum0 = 0.0f, sum1 = 0.0f;
    vx_u32 g = 0;
    for (; g + 3u < full_groups; g += 4u) {
        float scale00 = 0.0f, scale01 = 0.0f, scale02 = 0.0f, scale03 = 0.0f;
        float scale10 = 0.0f, scale11 = 0.0f, scale12 = 0.0f, scale13 = 0.0f;

        int dot00 = 0, dot01 = 0, dot02 = 0, dot03 = 0;
        int dot10 = 0, dot11 = 0, dot12 = 0, dot13 = 0;

        if (column < (vx_u64)output_width) {
            const VxCudaUint4* ap = (const VxCudaUint4*)(vx_packed_tile_shared + g * 32u);
            VxCudaUint4 act0 = ap[sub_lane];
            VxCudaUint4 act1 = ap[8u + sub_lane];
            VxCudaUint4 act2 = ap[16u + sub_lane];
            VxCudaUint4 act3 = ap[24u + sub_lane];

            const vx_u32* wp0 = w0_row + g * 16u + sub_lane * 2u;
            vx_u32 w00 = wp0[0], w01 = wp0[1];
            vx_u32 w02 = wp0[16], w03 = wp0[17];
            vx_u32 w04 = wp0[32], w05 = wp0[33];
            vx_u32 w06 = wp0[48], w07 = wp0[49];

            const vx_u32* wp1 = w1_row + g * 16u + sub_lane * 2u;
            vx_u32 w10 = wp1[0], w11 = wp1[1];
            vx_u32 w12 = wp1[16], w13 = wp1[17];
            vx_u32 w14 = wp1[32], w15 = wp1[33];
            vx_u32 w16 = wp1[48], w17 = wp1[49];

            if (!sub_lane) {
                scale00 = activation_scale * s0_row[g];
                scale01 = activation_scale * s0_row[g + 1u];
                scale02 = activation_scale * s0_row[g + 2u];
                scale03 = activation_scale * s0_row[g + 3u];
                scale10 = activation_scale * s1_row[g];
                scale11 = activation_scale * s1_row[g + 1u];
                scale12 = activation_scale * s1_row[g + 2u];
                scale13 = activation_scale * s1_row[g + 3u];
            }

            dot00 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w00))
                  + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w00))
                  + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w01))
                  + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w01));
            dot01 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w02))
                  + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w02))
                  + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w03))
                  + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w03));
            dot02 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w04))
                  + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w04))
                  + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w05))
                  + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w05));
            dot03 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w06))
                  + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w06))
                  + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w07))
                  + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w07));

            dot10 = vx_packed_cuda_dot4(act0.x, vx_packed_cuda_unpack_w4_low(w10))
                  + vx_packed_cuda_dot4(act0.y, vx_packed_cuda_unpack_w4_high(w10))
                  + vx_packed_cuda_dot4(act0.z, vx_packed_cuda_unpack_w4_low(w11))
                  + vx_packed_cuda_dot4(act0.w, vx_packed_cuda_unpack_w4_high(w11));
            dot11 = vx_packed_cuda_dot4(act1.x, vx_packed_cuda_unpack_w4_low(w12))
                  + vx_packed_cuda_dot4(act1.y, vx_packed_cuda_unpack_w4_high(w12))
                  + vx_packed_cuda_dot4(act1.z, vx_packed_cuda_unpack_w4_low(w13))
                  + vx_packed_cuda_dot4(act1.w, vx_packed_cuda_unpack_w4_high(w13));
            dot12 = vx_packed_cuda_dot4(act2.x, vx_packed_cuda_unpack_w4_low(w14))
                  + vx_packed_cuda_dot4(act2.y, vx_packed_cuda_unpack_w4_high(w14))
                  + vx_packed_cuda_dot4(act2.z, vx_packed_cuda_unpack_w4_low(w15))
                  + vx_packed_cuda_dot4(act2.w, vx_packed_cuda_unpack_w4_high(w15));
            dot13 = vx_packed_cuda_dot4(act3.x, vx_packed_cuda_unpack_w4_low(w16))
                  + vx_packed_cuda_dot4(act3.y, vx_packed_cuda_unpack_w4_high(w16))
                  + vx_packed_cuda_dot4(act3.z, vx_packed_cuda_unpack_w4_low(w17))
                  + vx_packed_cuda_dot4(act3.w, vx_packed_cuda_unpack_w4_high(w17));
        }

        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot00 += vx_cuda_gemv_shuffle_down_i32(dot00, delta);
            dot01 += vx_cuda_gemv_shuffle_down_i32(dot01, delta);
            dot02 += vx_cuda_gemv_shuffle_down_i32(dot02, delta);
            dot03 += vx_cuda_gemv_shuffle_down_i32(dot03, delta);
            dot10 += vx_cuda_gemv_shuffle_down_i32(dot10, delta);
            dot11 += vx_cuda_gemv_shuffle_down_i32(dot11, delta);
            dot12 += vx_cuda_gemv_shuffle_down_i32(dot12, delta);
            dot13 += vx_cuda_gemv_shuffle_down_i32(dot13, delta);
        }

        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot00 * scale00;
            sum0 += (float)dot01 * scale01;
            sum0 += (float)dot02 * scale02;
            sum0 += (float)dot03 * scale03;
            sum1 += (float)dot10 * scale10;
            sum1 += (float)dot11 * scale11;
            sum1 += (float)dot12 * scale12;
            sum1 += (float)dot13 * scale13;
        }
    }

    for (; g < full_groups; g++) {
        int dot0 = 0, dot1 = 0;
        if (column < (vx_u64)output_width) {
            VxCudaUint4 act = ((const VxCudaUint4*)(vx_packed_tile_shared + g * 32u))[sub_lane];
            const vx_u32* wp0 = w0_row + g * 16u + sub_lane * 2u;
            const vx_u32* wp1 = w1_row + g * 16u + sub_lane * 2u;
            dot0 = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp0[0]))
                 + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp0[0]))
                 + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp0[1]))
                 + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp0[1]));
            dot1 = vx_packed_cuda_dot4(act.x, vx_packed_cuda_unpack_w4_low(wp1[0]))
                 + vx_packed_cuda_dot4(act.y, vx_packed_cuda_unpack_w4_high(wp1[0]))
                 + vx_packed_cuda_dot4(act.z, vx_packed_cuda_unpack_w4_low(wp1[1]))
                 + vx_packed_cuda_dot4(act.w, vx_packed_cuda_unpack_w4_high(wp1[1]));
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
        }
        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0 * (activation_scale * s0_row[g]);
            sum1 += (float)dot1 * (activation_scale * s1_row[g]);
        }
    }

    for (; g < groups; g++) {
        int dot0 = 0, dot1 = 0;
        if (column < (vx_u64)output_width) {
            for (vx_u32 chunk = 0u; chunk < 4u; chunk++) {
                vx_u32 word = sub_lane * 4u + chunk;
                vx_u32 activation = vx_packed_tile_shared[g * 32u + word];
                vx_u32 k = g * 128u + word * 4u;
                dot0 += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w0_row, input_width, k));
                dot1 += vx_packed_cuda_dot4(activation, vx_packed_cuda_w4_tail_word(w1_row, input_width, k));
            }
        }
        for (vx_u32 delta = 4u; delta; delta >>= 1u) {
            dot0 += vx_cuda_gemv_shuffle_down_i32(dot0, delta);
            dot1 += vx_cuda_gemv_shuffle_down_i32(dot1, delta);
        }
        if (!sub_lane && column < (vx_u64)output_width) {
            sum0 += (float)dot0 * (activation_scale * s0_row[g]);
            sum1 += (float)dot1 * (activation_scale * s1_row[g]);
        }
    }

    if (!sub_lane && column < (vx_u64)output_width) {
        float val0 = sum0 + (bias0 ? bias0[column] : 0.0f);
        float val1 = sum1 + (bias1 ? bias1[column] : 0.0f);
        if (output0) output0[column] = val0;
        if (output1) output1[column] = val1;
        float silu_val = val0 * vx_sigmoid(val0);
        if (output_silu) output_silu[column] = silu_val;
        if (output_product) output_product[column] = silu_val * val1;
    }
}
#endif /* !TEST_CUDA_GEMV_SOURCE */

#endif /* VOLVOXAI_CUDA_FORWARD_GEMV_PAIR_KERNELS_INC */
