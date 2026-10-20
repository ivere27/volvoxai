/* Private D=128 head RMS -> contiguous reshape -> RoPE numerical entry.
 * The caller must retain all three output owners and prove disjoint input and
 * output extents, immutable powers, nonnegative positions and checked products.
 * Launch exactly 128 threads per head. Reduction order is the ordinary
 * cooperative RMS128 order; flat reshape stores the same rounded F32 values. */
extern "C" VX_CUDA_GLOBAL void vx_cuda_head_rms_reshape_rope_f32(
        const float* input, const float* weight, float* normalized,
        float* flat, const vx_i32* positions, float* output,
        vx_u32 rows, vx_u32 sequence, vx_u32 heads, vx_u32 rotary_width,
        const float* powers, float epsilon, vx_u32 position_offset,
        vx_u32 interleaved, vx_u32 position_mode,
        const unsigned char* argument_storage) {
    if (argument_storage) vx_cuda_restore_arguments(argument_storage,
        input, weight, normalized, flat, positions, output, rows, sequence,
        heads, rotary_width, powers, epsilon, position_offset, interleaved,
        position_mode);
    VX_CUDA_SHARED float warp_sums[4];
    vx_u32 head = blockIdx.x;
    if (!rows || !sequence || !heads || heads > 0xffffffffu / 128u ||
        rows > 0xffffffffu / (heads * 128u) || !rotary_width || rotary_width > 128u ||
        (rotary_width & 1u) || blockDim.x != 128u || head >= rows * heads) return;
    vx_u32 lane = threadIdx.x;
    vx_u32 base = head * 128u;
    float sum_squares = 0.0f;
    float value = input[base + lane];
    sum_squares = vx_add_rn(sum_squares, vx_mul_rn(value, value));
    for (vx_u32 delta = 16u; delta; delta /= 2u)
        sum_squares = vx_add_rn(
            sum_squares, vx_cuda_norm_shuffle_down(sum_squares, delta));
    if ((lane & 31u) == 0u) warp_sums[lane / 32u] = sum_squares;
    vx_sync_threads();
    if (lane == 0u) {
        float total = warp_sums[0];
        for (vx_u32 warp = 1u; warp < 4u; ++warp)
            total = vx_add_rn(total, warp_sums[warp]);
        warp_sums[0] = 1.0f / vx_sqrt(total / 128.0f + epsilon);
    }
    vx_sync_threads();
    float inverse = warp_sums[0];
    value = vx_mul_rn(vx_mul_rn(input[base + lane], inverse), weight[lane]);
    normalized[base + lane] = value;
    flat[base + lane] = value;
    if (lane >= rotary_width) {
        output[base + lane] = value;
        return;
    }
    vx_u32 row = head / heads;
    vx_u32 sequence_index = row % sequence;
    vx_i32 position = position_mode == 1u ? positions[sequence_index] :
        position_mode == 2u ? positions[row] :
        (vx_i32)(position_offset + sequence_index);
    vx_u32 half = rotary_width / 2u;
    vx_u32 pair = interleaved ? lane / 2u : lane % half;
    vx_u32 left = interleaved ? pair * 2u : pair;
    vx_u32 right = interleaved ? left + 1u : left + half;
    float angle = (float)position / powers[pair];
    float sine, cosine;
    vx_rope_sincos_f32(angle, &sine, &cosine);
    float a = vx_mul_rn(vx_mul_rn(input[base + left], inverse), weight[left]);
    float b = vx_mul_rn(vx_mul_rn(input[base + right], inverse), weight[right]);
    output[base + lane] = lane == left ? a * cosine - b * sine :
        a * sine + b * cosine;
}
