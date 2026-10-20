#ifndef VOLVOXAI_CUDA_PACKED_LAUNCH_CONFIG_H
#define VOLVOXAI_CUDA_PACKED_LAUNCH_CONFIG_H

/* Private host/device tile policy. Built-in integer types keep this header
 * usable by the freestanding CUDA compiler and the native C launch planner. */
#if defined(VX_CUDA_DEVICE)
#define VX_CUDA_PACKED_INLINE static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE
#else
#define VX_CUDA_PACKED_INLINE static inline
#endif

VX_CUDA_PACKED_INLINE unsigned int vx_cuda_packed_linear_tile_columns(
        unsigned long long rows, unsigned int output_width,
        unsigned int activation_bits, unsigned int sm_count) {
    if (!rows || !output_width || (activation_bits != 8u && activation_bits != 16u))
        return 0u;
    unsigned long long row_tiles = rows / 16u + (rows % 16u != 0u);
    unsigned int columns64 = output_width / 64u + (output_width % 64u != 0u);
    if (activation_bits == 8u) {
        unsigned int columns128 = output_width / 128u + (output_width % 128u != 0u);
        unsigned int minimum128 = sm_count > 128u ? sm_count : 128u;
        unsigned int minimum64 = sm_count > 64u ? sm_count : 64u;
        /* Division expresses row_tiles * column_tiles >= minimum without
         * overflowing for large declared shapes. Unknown SM capacity keeps
         * the legacy tile thresholds for direct kernel callers. */
        if (row_tiles >= minimum128 / columns128 + (minimum128 % columns128 != 0u))
            return 128u;
        if (row_tiles >= minimum64 / columns64 + (minimum64 % columns64 != 0u))
            return 64u;
    } else {
        unsigned int minimum64 = sm_count > 246u ? sm_count : 246u;
        if (row_tiles >= 2u &&
            row_tiles >= minimum64 / columns64 + (minimum64 % columns64 != 0u))
            return 64u;
    }
    return 32u;
}

/* Preparation adds a launch. Matrix tiles amortize it across rows; only wide
 * A8 column projections amortize it during decode. A16 columns stay direct. */
VX_CUDA_PACKED_INLINE int vx_cuda_packed_linear_prepares_input(
        unsigned int activation_bits, unsigned int output_width,
        unsigned int window) {
    if (output_width < 128u) return 0;
    if (activation_bits == 8u)
        return window == 2u ||
            (output_width >= 8192u && (window == 1u || window == 4u));
    return activation_bits == 16u && window == 3u;
}

/* Narrow chunked projections lose parallelism; long A16 reductions also pay
 * more for repeated launches than they save in input conversion. */
VX_CUDA_PACKED_INLINE int vx_cuda_packed_linear_prepares_chunks(
        unsigned int activation_bits, unsigned int input_width,
        unsigned int output_width) {
    return output_width >= 1024u &&
        (activation_bits == 8u || (activation_bits == 16u && input_width <= 2048u));
}

/* Qualified prepared matrix geometry. Select once for the complete operation:
 * the bounded workspace can split a long input into chunks smaller than this
 * threshold. Kernels mask final row fragments without changing sum order.
 * Narrow/decode/unknown geometry retains the established tile policy. */
VX_CUDA_PACKED_INLINE unsigned int vx_cuda_packed_linear_prepared_tile_rows(
        unsigned long long rows, unsigned int input_width,
        unsigned int output_width, unsigned int activation_bits,
        unsigned int weight_bits, unsigned int compiled_arch) {
    if (activation_bits == 8u && weight_bits == 4u && compiled_arch >= 80u) {
        /* Narrow KV projections need enough row tiles to fill the device.
         * Keep shorter prompts and lengths beyond the measured interval on
         * their established route. */
        if (rows >= 1024u && rows <= 2048u && input_width == 2048u &&
            output_width == 256u) return 64u;
        if (rows >= 16u && rows <= 32u && input_width == 2048u &&
            (output_width == 2048u || output_width == 5632u)) return 16u;
        if (rows >= 512u &&
            ((input_width == 2048u && (output_width == 2048u || output_width == 5632u)) ||
             (input_width == 5632u && output_width == 2048u))) return 64u;
    }
    if (activation_bits == 16u && (weight_bits == 4u || weight_bits == 8u) &&
        compiled_arch >= 75u && rows >= 512u &&
        ((input_width == 2048u && (output_width == 2048u || output_width == 5632u)) ||
         (input_width == 5632u && output_width == 2048u))) return 64u;
    return 0u;
}

/* Narrow A16 projections benefit from preparing the complete prefix while
 * retaining the ordinary 16-row compute tile. A wider row tile reduces
 * parallel work too far for this output width. */
VX_CUDA_PACKED_INLINE int vx_cuda_packed_linear_prepared_narrow_workspace(
        unsigned long long rows, unsigned int input_width,
        unsigned int output_width, unsigned int activation_bits,
        unsigned int weight_bits, unsigned int compiled_arch) {
    return compiled_arch >= 80u && rows >= 512u && rows <= 2048u &&
        input_width == 2048u && output_width == 256u &&
        activation_bits == 16u && (weight_bits == 4u || weight_bits == 8u);
}

/* Eight A16 column warps reduce register residency on measured long shapes.
 * The 512-row gate/up projection and SM75 retain four warps. Select from the
 * complete operation before workspace chunking, just like its row tile. */
VX_CUDA_PACKED_INLINE unsigned int vx_cuda_packed_linear_prepared_threads(
        unsigned long long rows, unsigned int input_width,
        unsigned int output_width, unsigned int activation_bits,
        unsigned int weight_bits, unsigned int compiled_arch) {
    if (activation_bits == 8u) return 256u;
    if (compiled_arch >= 80u && (output_width == 2048u || rows >= 1024u) &&
        vx_cuda_packed_linear_prepared_tile_rows(rows, input_width, output_width,
            activation_bits, weight_bits, compiled_arch) == 64u) return 256u;
    return 128u;
}

#undef VX_CUDA_PACKED_INLINE
#endif
