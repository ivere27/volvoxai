/*
 * N32 pair kernel template: two activation rows against thirty-two output
 * channels, four reduction elements at a time.
 *
 * The reduction spelling is the only thing that changes between instruction
 * sets, so it is the only thing this file leaves to the includer.  Everything
 * else -- the packed layout walk, the affine compensation, the requantizing
 * epilogue -- is shared, which is what keeps the VNNI tiers from drifting away
 * from the AVX2 one they have to agree with bit for bit.
 *
 * The includer defines:
 *   VX_PN32_TARGET             function target attribute
 *   VX_PN32_NAME               function name
 *   VX_PN32_ALLOW_SIGNED_ABS   1 when the magnitude/sign spelling is reachable;
 *                              0 lets the compiler drop it entirely
 *   VX_PN32_DOT_SIGNED_ABS(..) accumulate via magnitudes and transferred sign
 *   VX_PN32_DOT_PLAIN(..)      accumulate the operands directly
 * with vx_inputs, weights0..3 and ones16 in scope at the expansion point.
 */

static VX_PN32_TARGET __attribute__((always_inline)) inline void
VX_PN32_NAME(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end,
        const uint32_t input_xor_mask, const int use_signed_abs) {
    const VxPackedQ8Header* header = call->header;
    const uint8_t* pair_data =
        (const uint8_t*)header + header->pair_data_offset;
    const int32_t output_minimum =
        call->output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum =
        call->output_dtype == VX_DTYPE_I8 ? 127 : 255;
    const __m256 output_minimum_f32 = _mm256_set1_ps((float)output_minimum);
    const __m256 output_maximum_f32 = _mm256_set1_ps((float)output_maximum);
    const __m256 output_zero_f32 =
        _mm256_set1_ps((float)call->output_zero_point);
    const __m256i ones16 = _mm256_set1_epi16(1);
    /* The signed-absolute spelling reduces the remapped operand, so its
     * compensation is written against the signed domain; the plain spelling
     * reduces the unsigned operand and compensates against that instead. */
    const int32_t input_sum_correction = use_signed_abs
        ? (call->input_dtype == VX_DTYPE_U8
            ? 128 - call->input_zero_point : -call->input_zero_point)
        : -(call->input_zero_point +
            (call->input_dtype == VX_DTYPE_I8 ? 128 : 0));
    const __m256 input_scales = _mm256_set1_ps(call->input_scale);
    const __m256 output_scales = _mm256_set1_ps(call->output_scale);
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const uint32_t full_k_blocks = header->d_in / 4u;
    for (uint32_t group = group_begin; group < group_end; group++) {
        const uint32_t first_block = group * 4u;
        const uint32_t base = first_block * 8u;
        const __m256i correction =
            _mm256_set1_epi32(input_sum_correction);
        const __m256i initial0 = _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i*)(const void*)
                (call->bias + base)),
            _mm256_mullo_epi32(correction, _mm256_loadu_si256(
                (const __m256i*)(const void*)(raw_sums + base))));
        const __m256i initial1 = _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i*)(const void*)
                (call->bias + base + 8u)),
            _mm256_mullo_epi32(correction, _mm256_loadu_si256(
                (const __m256i*)(const void*)(raw_sums + base + 8u))));
        const __m256i initial2 = _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i*)(const void*)
                (call->bias + base + 16u)),
            _mm256_mullo_epi32(correction, _mm256_loadu_si256(
                (const __m256i*)(const void*)(raw_sums + base + 16u))));
        const __m256i initial3 = _mm256_add_epi32(
            _mm256_loadu_si256((const __m256i*)(const void*)
                (call->bias + base + 24u)),
            _mm256_mullo_epi32(correction, _mm256_loadu_si256(
                (const __m256i*)(const void*)(raw_sums + base + 24u))));
        const __m256 multiplier0 = _mm256_div_ps(_mm256_mul_ps(input_scales,
            _mm256_loadu_ps(call->weight_scales + base)), output_scales);
        const __m256 multiplier1 = _mm256_div_ps(_mm256_mul_ps(input_scales,
            _mm256_loadu_ps(call->weight_scales + base + 8u)), output_scales);
        const __m256 multiplier2 = _mm256_div_ps(_mm256_mul_ps(input_scales,
            _mm256_loadu_ps(call->weight_scales + base + 16u)), output_scales);
        const __m256 multiplier3 = _mm256_div_ps(_mm256_mul_ps(input_scales,
            _mm256_loadu_ps(call->weight_scales + base + 24u)), output_scales);
        const uint8_t* weights0_base = pair_data +
            (size_t)first_block * header->pair_k_blocks * 32u;
        const uint8_t* weights1_base = weights0_base +
            (size_t)header->pair_k_blocks * 32u;
        const uint8_t* weights2_base = weights1_base +
            (size_t)header->pair_k_blocks * 32u;
        const uint8_t* weights3_base = weights2_base +
            (size_t)header->pair_k_blocks * 32u;
        uint32_t row = 0;

#define VX_QGEMM_SIGNED_N32_ACCUMULATE(INPUT_PTR, COUNT, A0, A1, A2, A3) do { \
            uint32_t vx_input_pack = 0u; \
            __builtin_memcpy(&vx_input_pack, (INPUT_PTR), (COUNT)); \
            vx_input_pack ^= input_xor_mask; \
            const __m256i vx_inputs = \
                _mm256_set1_epi32((int32_t)vx_input_pack); \
            if (VX_PN32_ALLOW_SIGNED_ABS && use_signed_abs) { \
                VX_PN32_DOT_SIGNED_ABS(A0, A1, A2, A3); \
            } else { \
                VX_PN32_DOT_PLAIN(A0, A1, A2, A3); \
            } \
        } while (0)

        for (; row + 2u <= call->rows; row += 2u) {
            const uint8_t* input0 = (const uint8_t*)call->input +
                (size_t)row * header->d_in;
            const uint8_t* input1 = input0 + header->d_in;
            __m256i accum00 = initial0, accum01 = initial1;
            __m256i accum02 = initial2, accum03 = initial3;
            __m256i accum10 = initial0, accum11 = initial1;
            __m256i accum12 = initial2, accum13 = initial3;
            uint32_t k_block = 0;
            for (; k_block < full_k_blocks; k_block++) {
                const uint32_t dimension = k_block * 4u;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)k_block * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)k_block * 32u));
                const __m256i weights2 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights2_base + (size_t)k_block * 32u));
                const __m256i weights3 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights3_base + (size_t)k_block * 32u));
                VX_QGEMM_SIGNED_N32_ACCUMULATE(input0 + dimension, 4u,
                    accum00, accum01, accum02, accum03);
                VX_QGEMM_SIGNED_N32_ACCUMULATE(input1 + dimension, 4u,
                    accum10, accum11, accum12, accum13);
            }
            if (header->d_in & 3u) {
                const uint32_t dimension = full_k_blocks * 4u;
                const uint32_t tail = header->d_in - dimension;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)full_k_blocks * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)full_k_blocks * 32u));
                const __m256i weights2 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights2_base + (size_t)full_k_blocks * 32u));
                const __m256i weights3 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights3_base + (size_t)full_k_blocks * 32u));
                VX_QGEMM_SIGNED_N32_ACCUMULATE(input0 + dimension, tail,
                    accum00, accum01, accum02, accum03);
                VX_QGEMM_SIGNED_N32_ACCUMULATE(input1 + dimension, tail,
                    accum10, accum11, accum12, accum13);
            }
            vx_qgemm_w8a8_avx2_store8(call, row, base, accum00, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 8u, accum01,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 16u, accum02,
                multiplier2, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 24u, accum03,
                multiplier3, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base, accum10,
                multiplier0, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base + 8u, accum11,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base + 16u, accum12,
                multiplier2, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base + 24u, accum13,
                multiplier3, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
        }
        for (; row < call->rows; row++) {
            const uint8_t* input0 = (const uint8_t*)call->input +
                (size_t)row * header->d_in;
            __m256i accum0 = initial0, accum1 = initial1;
            __m256i accum2 = initial2, accum3 = initial3;
            uint32_t k_block = 0;
            for (; k_block < full_k_blocks; k_block++) {
                const uint32_t dimension = k_block * 4u;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)k_block * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)k_block * 32u));
                const __m256i weights2 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights2_base + (size_t)k_block * 32u));
                const __m256i weights3 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights3_base + (size_t)k_block * 32u));
                VX_QGEMM_SIGNED_N32_ACCUMULATE(input0 + dimension, 4u,
                    accum0, accum1, accum2, accum3);
            }
            if (header->d_in & 3u) {
                const uint32_t dimension = full_k_blocks * 4u;
                const uint32_t tail = header->d_in - dimension;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)full_k_blocks * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)full_k_blocks * 32u));
                const __m256i weights2 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights2_base + (size_t)full_k_blocks * 32u));
                const __m256i weights3 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights3_base + (size_t)full_k_blocks * 32u));
                VX_QGEMM_SIGNED_N32_ACCUMULATE(input0 + dimension, tail,
                    accum0, accum1, accum2, accum3);
            }
            vx_qgemm_w8a8_avx2_store8(call, row, base, accum0, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 8u, accum1,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 16u, accum2,
                multiplier2, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 24u, accum3,
                multiplier3, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
        }
#undef VX_QGEMM_SIGNED_N32_ACCUMULATE
    }
}
