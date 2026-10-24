/* Range reduction uses the same F32 angle as native CPU RoPE. Positive powers
 * are canonical host-prepared operator metadata, not activation fallback. */
static VX_CUDA_DEVICE void vx_rope_sincos_f32(float angle, float* sine,
                                            float* cosine) {
    /* theta >= 1 and nonnegative I32 positions bound angle to 2^31.
     * Split tau avoids losing its low part in the double range reduction.
     * PTX sin/cos approximations are only used in their documented [-pi,pi]
     * range, never on an arbitrary un-reduced rotary position. */
    double wide = (double)angle;
    vx_u32 turns = (vx_u32)(wide * .15915494309189533577 + .5);
    double reduced = (wide - (double)turns * 6.2831853071795862320) -
        (double)turns * 2.4492935982947064143e-16;
    float argument = (float)reduced;
    asm("sin.approx.f32 %0, %1;" : "=f"(*sine) : "f"(argument));
    asm("cos.approx.f32 %0, %1;" : "=f"(*cosine) : "f"(argument));
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_rope_heads_f32(
        const float* input, const vx_i32* positions, float* output,
        vx_u32 elements, vx_u32 sequence, vx_u32 width, vx_u32 heads,
        vx_u32 rotary_width, const float* powers, vx_u32 position_offset,
        vx_u32 interleaved, vx_u32 position_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, positions, output, elements, sequence, width, heads, rotary_width, powers, position_offset, interleaved, position_mode);
    vx_u32 gid = vx_global_x();
    if (gid >= elements) return;
    vx_u32 head_width = width / heads;
    vx_u32 dimension = gid % head_width;
    if (dimension >= rotary_width) {
        output[gid] = input[gid];
        return;
    }
    vx_u32 row = gid / width;
    vx_u32 sequence_index = row % sequence;
    vx_i32 position = position_mode == 1u ? positions[sequence_index] :
        position_mode == 2u ? positions[row] :
        (vx_i32)(position_offset + sequence_index);
    vx_u32 half = rotary_width / 2u;
    /* One thread owns both rotated dimensions, sharing angle reduction and
     * input loads. Dimensions outside the rotary span keep their own copy. */
    if (interleaved ? (dimension & 1u) : dimension >= half) return;
    vx_u32 pair = interleaved ? dimension / 2u : dimension % half;
    vx_u32 left = interleaved ? pair * 2u : pair;
    vx_u32 right = interleaved ? left + 1u : left + half;
    vx_u32 base = gid - dimension;
    float angle = (float)position / powers[pair];
    float sine, cosine;
    vx_rope_sincos_f32(angle, &sine, &cosine);
    float a = input[base + left], b = input[base + right];
    output[base + left] = a * cosine - b * sine;
    output[base + right] = a * sine + b * cosine;
}

/* Pair RoPE kernel for adjacent Q and K heads: rotates both projections in a single
 * launch without warp divergence, sharing position reduction and frequency powers. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_rope_pair_heads_f32(
        const float* input0, const vx_i32* positions, float* output0,
        const float* input1, float* output1,
        vx_u32 elements0, vx_u32 elements1,
        vx_u32 width0, vx_u32 heads0,
        vx_u32 width1, vx_u32 heads1,
        vx_u32 rotary_width, const float* powers, vx_u32 position_offset,
        vx_u32 interleaved, vx_u32 position_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input0, positions, output0,
        input1, output1,
        elements0, elements1,
        width0, heads0,
        width1, heads1,
        rotary_width, powers, position_offset,
        interleaved, position_mode);

    vx_u32 half = rotary_width / 2u;
    vx_u32 pairs0 = heads0 * half;
    vx_u32 pairs1 = heads1 * half;
    vx_u32 total_pairs = pairs0 + pairs1;
    vx_u32 gid = vx_global_x();
    if (gid >= total_pairs) return;

    vx_i32 position = position_mode == 1u ? positions[0] :
        position_mode == 2u ? positions[0] : (vx_i32)position_offset;

    const float* in;
    float* out;
    vx_u32 pair_in_head, head, head_width = rotary_width;
    vx_u32 left, right;

    if (gid < pairs0) {
        in = input0;
        out = output0;
        head = gid / half;
        pair_in_head = gid % half;
    } else {
        in = input1;
        out = output1;
        vx_u32 k_gid = gid - pairs0;
        head = k_gid / half;
        pair_in_head = k_gid % half;
    }

    left = head * head_width + (interleaved ? pair_in_head * 2u : pair_in_head);
    right = interleaved ? left + 1u : left + half;

    float angle = (float)position / powers[pair_in_head];
    float sine, cosine;
    vx_rope_sincos_f32(angle, &sine, &cosine);
    float a = in[left], b = in[right];
    out[left] = a * cosine - b * sine;
    out[right] = a * sine + b * cosine;
}
