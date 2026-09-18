/*
 * Cache-friendly packed dense kernels for quantized weights.
 *
 * B is packed once as NR-wide output panels: [N/NR, K, NR].  The inner loop
 * therefore loads one activation and reuses it across NR output channels.
 * MR rows share the same hot B panel.  With the conservative 32 KiB L1
 * fallback used for tile design, MR=4, NR=8, and KC=960 consume about 7.5 KiB
 * of packed Q8 weights plus either 3.75 KiB of byte activations (W8A8) or
 * 15 KiB of F32 activations (W8A32).  Both remain below a conservative 75%
 * L1 budget after their accumulators.  KC is an iteration
 * boundary, not serialized metadata; all constants remain internal.
 */
#include "packed_quant_gemm.h"
#include "w8a8_affine.h"
#include "kernel_platform.h"
#include "gemm_f32.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __wasm__
#include <wasm_simd128.h>
#define VX_QGEMM_WASM_SIMD 1
#define VX_QGEMM_WASM_PAIR_PACK 1
#else
#define VX_QGEMM_WASM_SIMD 0
#define VX_QGEMM_WASM_PAIR_PACK 0
#endif

#if defined(VOLVOXAI_ARM_I8MM_OBJECT) && defined(__aarch64__)
#define VX_QGEMM_ARM_I8MM 1
#else
#define VX_QGEMM_ARM_I8MM 0
#endif

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include "cpu_features.h"
#include "thread_pool.h"
#include <immintrin.h>
#define VX_QGEMM_X86_AVX2 1
#define VX_QGEMM_TARGET_AVX2 __attribute__((target("avx2")))
/* Two ways to reach a 256-bit VPDPBUSD: the VEX encoding that Alder Lake and
 * Zen 4 expose as AVX-VNNI, and the EVEX encoding available on the AVX-512-VNNI
 * parts that predate it, such as Ice Lake and Tiger Lake.  Both accumulate four
 * byte products straight into I32, so neither can saturate and neither needs
 * the magnitude/sign decomposition the AVX2 spelling pays for. */
#define VX_QGEMM_TARGET_AVXVNNI __attribute__((target("avx2,avxvnni")))
#define VX_QGEMM_TARGET_AVX512VNNI \
    __attribute__((target("avx512f,avx512bw,avx512vl,avx512vnni")))
#else
#define VX_QGEMM_X86_AVX2 0
#define VX_QGEMM_TARGET_AVX2
#endif

#ifndef WASM_EXPORT
#ifdef __wasm__
#define WASM_EXPORT(name) __attribute__((export_name(name)))
#else
#define WASM_EXPORT(name)
#endif
#endif

enum {
    VX_QGEMM_MAGIC = VX_PACKED_Q8_MAGIC,
    VX_QGEMM_NR = VX_PACKED_Q8_NR,
    VX_QGEMM_MR = 4u,
    /* Five rows amortize each SIMD128 weight shuffle while limiting the
     * paired output accumulator set to ten vectors. */
    VX_QGEMM_WASM_MR = 5u,
    /* Baseline K block, and the value the derivation below has to reproduce on
     * the 32 KiB L1 this file used to assume unconditionally. */
    VX_QGEMM_KC_BASELINE = 960u,
    VX_QGEMM_KC_MIN = 64u,
    VX_QGEMM_KC_ALIGNMENT = 64u,
    VX_QGEMM_PAIR_SIGNED_I8 = 1u,
    VX_QGEMM_PAIR_NO_NEG128 = 2u,
    /* Every weight satisfies |w| <= VX_QGEMM_PAIR_SAFE_ABS_WEIGHT.  VPMADDUBSW
     * sums two U8xI8 products into one I16 lane, so the exact lane range is
     * 2 * 255 * max|w|.  At that bound it is +-32640, inside I16, and the raw
     * unsigned activation can be multiplied directly: no centering, no
     * magnitude/sign decomposition, and no data-dependent high-bit test.
     * Above the bound the same sequence saturates and silently returns a wrong
     * dot product, so the flag is only set when packing proves the range. */
    VX_QGEMM_PAIR_NO_SATURATE = 4u,
};

enum { VX_QGEMM_PAIR_SAFE_ABS_WEIGHT = 64 };

/*
 * The K block, derived from the same detected L1 the F32 path uses.
 *
 * It was the literal 960 with no cache query at all, while gemm_f32.c next
 * door detected L1, L2 and L3 — so on a machine with a 48 KiB L1 the float
 * kernels adapted and the quantized ones did not.  The budget here is the one
 * the microkernel guide already describes: three quarters of L1, holding one
 * byte-wide B panel of KC x NR, the widest A form the kernel accepts (four F32
 * activation rows), and the I32 accumulators.
 *
 * On a 32 KiB L1 that is (4*4 + 8*1) * KC + 4*8*4 <= 24576, so KC <= 1018,
 * which floors to 960 at 64-element alignment: exactly the constant this
 * replaces. vx_qgemm_kc_is_baseline() keeps the derivation pinned to the
 * measured value it came from rather than quietly moving.
 */
static uint32_t vx_qgemm_kc_for_l1(uint32_t l1_bytes) {
    const uint32_t budget = l1_bytes - l1_bytes / 4u;
    const uint32_t accumulators = VX_QGEMM_MR * VX_QGEMM_NR * 4u;
    const uint32_t per_k = VX_QGEMM_MR * 4u + VX_QGEMM_NR * 1u;
    uint32_t available = budget > accumulators ? budget - accumulators : 0u;
    uint32_t kc = per_k ? available / per_k : 0u;
    kc -= kc % VX_QGEMM_KC_ALIGNMENT;
    return kc < VX_QGEMM_KC_MIN ? VX_QGEMM_KC_MIN : kc;
}

uint32_t vx_qgemm_kc(void) {
#if defined(__wasm__)
    /* Browsers report no cache topology, so this is the deterministic policy
     * the F32 path uses for the same reason. */
    return vx_qgemm_kc_for_l1(32u * 1024u);
#else
    return vx_qgemm_kc_for_l1(vx_gemm_f32_tile_config().l1_bytes);
#endif
}

int vx_qgemm_kc_is_baseline(void) {
    return vx_qgemm_kc_for_l1(32u * 1024u) == VX_QGEMM_KC_BASELINE;
}

static uint32_t vx_qgemm_align16(uint32_t value) {
    return (value + 15u) & ~15u;
}

static float vx_qgemm_f32_le(const void* data, size_t index) {
    const uint8_t* bytes = (const uint8_t*)data + index * 4u;
    union { uint32_t u; float f; } value = {
        (uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u)
    };
    return value.f;
}

static int32_t vx_qgemm_i32_le(const void* data, size_t index) {
    const uint8_t* bytes = (const uint8_t*)data + index * 4u;
    return (int32_t)((uint32_t)bytes[0] | ((uint32_t)bytes[1] << 8u) |
        ((uint32_t)bytes[2] << 16u) | ((uint32_t)bytes[3] << 24u));
}

static double vx_qgemm_typed_value(const void* data, uint32_t dtype, size_t index) {
    if (dtype == VX_DTYPE_F32) return vx_qgemm_f32_le(data, index);
    if (dtype == VX_DTYPE_I32) return vx_qgemm_i32_le(data, index);
    return vx_w8a8_byte_value(data, dtype, index);
}

static int vx_qgemm_typed_dtype(uint32_t dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

static int vx_qgemm_positive_product_multiplier(
        float input_scale, float weight_scale, float output_scale) {
    volatile float product_scale = input_scale * weight_scale;
    volatile float multiplier = product_scale / output_scale;
    return vx_w8a8_finite_f32(multiplier) && multiplier > 0.0f;
}

/* Header, per-N8 sums, and canonical KxN8 bytes only.  W8A32 and scalar
 * kernels consume this representation directly and must not pay for the
 * widened SIMD-only payload. */
static uint32_t vx_qgemm_canonical_weight_size(
        uint32_t d_in, uint32_t d_out) {
    uint64_t n_blocks;
    uint64_t sums_bytes;
    uint64_t data_bytes;
    uint64_t data_offset;
    uint64_t total;
    if (!d_in || !d_out || d_in > (uint32_t)(INT32_MAX / 255)) return 0;
    n_blocks = ((uint64_t)d_out - 1u) / VX_QGEMM_NR + 1u;
    sums_bytes = n_blocks * VX_QGEMM_NR * sizeof(int32_t);
    data_bytes = n_blocks * d_in * VX_QGEMM_NR;
    data_offset = ((uint64_t)sizeof(VxPackedQ8Header) + sums_bytes + 15u) &
        ~(uint64_t)15u;
    total = (data_offset + data_bytes + 15u) & ~(uint64_t)15u;
    return total <= UINT32_MAX ? (uint32_t)total : 0u;
}

#if VX_QGEMM_WASM_PAIR_PACK
WASM_EXPORT("packed_q8_weight_canonical_size")
uint32_t vx_packed_q8_weight_canonical_size(
        uint32_t d_in, uint32_t d_out) {
    return vx_qgemm_canonical_weight_size(d_in, d_out);
}
#endif

WASM_EXPORT("packed_q8_weight_size")
uint32_t vx_packed_q8_weight_size(uint32_t d_in, uint32_t d_out) {
    uint64_t n_blocks;
    uint64_t sums_bytes;
    uint64_t data_bytes;
    uint64_t data_offset;
    uint64_t pair_n_blocks = 0;
    uint64_t pair_k_blocks = 0;
    uint64_t pair_bytes = 0;
    uint64_t pair_data_offset;
    uint64_t total;
    if (!d_in || !d_out || d_in > (uint32_t)(INT32_MAX / 255)) return 0;
    n_blocks = ((uint64_t)d_out - 1u) / VX_QGEMM_NR + 1u;
    sums_bytes = n_blocks * VX_QGEMM_NR * sizeof(int32_t);
    data_bytes = n_blocks * d_in * VX_QGEMM_NR;
    data_offset = ((uint64_t)sizeof(VxPackedQ8Header) + sums_bytes + 15u) &
        ~(uint64_t)15u;
#if VX_QGEMM_X86_AVX2
    /* The native exact MxN kernel consumes K4 groups for eight output
     * channels, matching VPMADDUBSW/VPMADDWD's reduction layout. Keep the
     * canonical Kx8 bytes as well: W8A32, WASM and scalar tails share that
     * portable in-memory pack. Packs are derived CompiledModel state and are
     * never serialized into release artifacts. */
    pair_n_blocks = ((uint64_t)d_out + 7u) / 8u;
    pair_k_blocks = ((uint64_t)d_in + 3u) / 4u;
    pair_bytes = pair_n_blocks * pair_k_blocks * 32u;
#elif VX_QGEMM_ARM_I8MM
    /* SMMLA consumes two K8 rows and two K8 columns from each pair of
     * int8x16 operands.  Four column pairs therefore form one K8xN8 panel. */
    pair_n_blocks = n_blocks;
    pair_k_blocks = ((uint64_t)d_in + 7u) / 8u;
    pair_bytes = pair_n_blocks * pair_k_blocks * 64u;
#elif VX_QGEMM_WASM_PAIR_PACK
    /* SIMD128 consumes adjacent K values for one N8 panel.  Keep the
     * canonical Kx8 bytes for scalar/W8A32 execution and add a target-derived
     * I16 pack laid out as
     *   [w(k0,n0), w(k1,n0), ..., w(k0,n7), w(k1,n7)].
     * Two aligned v128 loads now produce the dot-product vectors directly,
     * without sign extension or rebuilding the interleave in every row panel.
     * The measured compute saving outweighs the doubled derived payload for
     * encoder-sized QLinear/QConv shapes.  This private pack is
     * context-derived and never serialized. */
    pair_n_blocks = n_blocks;
    pair_k_blocks = ((uint64_t)d_in + 1u) / 2u;
    pair_bytes = pair_n_blocks * pair_k_blocks * 32u;
#endif
    pair_data_offset = (data_offset + data_bytes + 15u) & ~(uint64_t)15u;
    total = pair_data_offset + pair_bytes;
    return total <= UINT32_MAX ? (uint32_t)total : 0u;
}

static int vx_pack_q8_weight_impl(
        void* packed, uint32_t packed_bytes, const void* weight,
        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype,
        uint32_t out_in, int canonical_only) {
    VxPackedQ8Header* header = (VxPackedQ8Header*)packed;
    uint32_t needed = canonical_only
        ? vx_qgemm_canonical_weight_size(d_in, d_out)
        : vx_packed_q8_weight_size(d_in, d_out);
    uint32_t n_blocks = (d_out - 1u) / VX_QGEMM_NR + 1u;
    uint32_t sums_offset = vx_qgemm_align16((uint32_t)sizeof(*header));
    uint32_t data_offset = vx_qgemm_align16(sums_offset +
        n_blocks * VX_QGEMM_NR * (uint32_t)sizeof(int32_t));
    uint32_t pair_n_blocks = 0;
    uint32_t pair_k_blocks = 0;
    uint32_t pair_data_offset = vx_qgemm_align16(data_offset +
        n_blocks * d_in * VX_QGEMM_NR);
    int32_t* sums;
    uint8_t* data;
    uint8_t* pair_data;
    uint32_t pair_flags = !canonical_only && weight_dtype == VX_DTYPE_I8
        ? VX_QGEMM_PAIR_SIGNED_I8 | VX_QGEMM_PAIR_NO_NEG128 |
          VX_QGEMM_PAIR_NO_SATURATE : 0u;
    if (!packed || !weight || !needed || packed_bytes < needed ||
        !vx_w8a8_byte_dtype(weight_dtype) || out_in > 1u) return 0;
    header->magic = VX_QGEMM_MAGIC;
    header->bytes = needed;
    header->d_in = d_in;
    header->d_out = d_out;
    header->n_blocks = n_blocks;
    header->weight_dtype = weight_dtype;
    header->sums_offset = sums_offset;
    header->data_offset = data_offset;
    if (!canonical_only) {
#if VX_QGEMM_X86_AVX2
        pair_n_blocks = (d_out + 7u) / 8u;
        pair_k_blocks = (d_in + 3u) / 4u;
#elif VX_QGEMM_ARM_I8MM
        pair_n_blocks = n_blocks;
        pair_k_blocks = (d_in + 7u) / 8u;
#elif VX_QGEMM_WASM_PAIR_PACK
        pair_n_blocks = n_blocks;
        pair_k_blocks = (d_in + 1u) / 2u;
#else
        pair_flags = 0u;
#endif
    }
    header->pair_n_blocks = pair_n_blocks;
    header->pair_k_blocks = pair_k_blocks;
    header->pair_data_offset = pair_data_offset;
    header->pair_flags = 0u;
    sums = (int32_t*)((uint8_t*)packed + sums_offset);
    data = (uint8_t*)packed + data_offset;
    for (uint32_t column = 0; column < n_blocks * VX_QGEMM_NR; column++) {
        int64_t sum = 0;
        uint32_t block = column / VX_QGEMM_NR;
        uint32_t lane = column % VX_QGEMM_NR;
        for (uint32_t dimension = 0; dimension < d_in; dimension++) {
            uint8_t raw = 0;
            if (column < d_out) {
                size_t source = out_in ? (size_t)column * d_in + dimension :
                    (size_t)dimension * d_out + column;
                raw = ((const uint8_t*)weight)[source];
                sum += vx_w8a8_byte_value(weight, weight_dtype, source);
                if (weight_dtype == VX_DTYPE_I8) {
                    const int value = (int)(int8_t)raw;
                    if (raw == 0x80u) pair_flags &= ~VX_QGEMM_PAIR_NO_NEG128;
                    if (value > VX_QGEMM_PAIR_SAFE_ABS_WEIGHT ||
                        value < -VX_QGEMM_PAIR_SAFE_ABS_WEIGHT)
                        pair_flags &= ~VX_QGEMM_PAIR_NO_SATURATE;
                }
            }
            data[((size_t)block * d_in + dimension) * VX_QGEMM_NR + lane] = raw;
        }
        sums[column] = (int32_t)sum;
    }
    if (canonical_only) return 1;
    pair_data = (uint8_t*)packed + pair_data_offset;
#if VX_QGEMM_WASM_PAIR_PACK
    for (uint32_t block = 0; block < pair_n_blocks; block++) {
        for (uint32_t k_block = 0; k_block < pair_k_blocks; k_block++) {
            for (uint32_t lane = 0; lane < VX_QGEMM_NR; lane++) {
                const uint32_t column = block * VX_QGEMM_NR + lane;
                for (uint32_t k_lane = 0; k_lane < 2u; k_lane++) {
                    const uint32_t dimension = k_block * 2u + k_lane;
                    uint8_t raw = 0;
                    if (column < d_out && dimension < d_in) {
                        size_t source = out_in
                            ? (size_t)column * d_in + dimension
                            : (size_t)dimension * d_out + column;
                        raw = ((const uint8_t*)weight)[source];
                    }
                    ((int16_t*)pair_data)[
                        ((size_t)block * pair_k_blocks + k_block) *
                            16u + lane * 2u + k_lane] =
                        weight_dtype == VX_DTYPE_I8
                            ? (int16_t)(int8_t)raw : (int16_t)raw;
                }
            }
        }
    }
#elif VX_QGEMM_ARM_I8MM
    for (uint32_t block = 0; block < pair_n_blocks; block++) {
        for (uint32_t k_block = 0; k_block < pair_k_blocks; k_block++) {
            uint8_t* panel = pair_data +
                ((size_t)block * pair_k_blocks + k_block) * 64u;
            for (uint32_t pair = 0; pair < 4u; pair++) {
                for (uint32_t column_lane = 0; column_lane < 2u;
                     column_lane++) {
                    const uint32_t column =
                        block * VX_QGEMM_NR + pair * 2u + column_lane;
                    for (uint32_t k_lane = 0; k_lane < 8u; k_lane++) {
                        const uint32_t dimension = k_block * 8u + k_lane;
                        uint8_t raw = 0;
                        if (column < d_out && dimension < d_in) {
                            const size_t source = out_in
                                ? (size_t)column * d_in + dimension
                                : (size_t)dimension * d_out + column;
                            raw = ((const uint8_t*)weight)[source];
                        }
                        panel[pair * 16u + column_lane * 8u + k_lane] = raw;
                    }
                }
            }
        }
    }
#else
    for (uint32_t block = 0; block < pair_n_blocks; block++) {
        for (uint32_t k_block = 0; k_block < pair_k_blocks; k_block++) {
            for (uint32_t lane = 0; lane < 8u; lane++) {
                const uint32_t column = block * 8u + lane;
                for (uint32_t k_lane = 0; k_lane < 4u; k_lane++) {
                    const uint32_t dimension = k_block * 4u + k_lane;
                    uint8_t raw = 0;
                    if (column < d_out && dimension < d_in) {
                        size_t source = out_in
                            ? (size_t)column * d_in + dimension
                            : (size_t)dimension * d_out + column;
                        raw = ((const uint8_t*)weight)[source];
                    }
                    pair_data[(((size_t)block * pair_k_blocks + k_block) *
                        8u + lane) * 4u + k_lane] = raw;
                }
            }
        }
    }
#endif
    header->pair_flags = pair_flags;
    return 1;
}

WASM_EXPORT("pack_q8_weight")
int vx_pack_q8_weight(void* packed, uint32_t packed_bytes, const void* weight,
        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype,
        uint32_t out_in) {
    return vx_pack_q8_weight_impl(packed, packed_bytes, weight, d_in, d_out,
        weight_dtype, out_in, 0);
}

#if VX_QGEMM_WASM_PAIR_PACK
WASM_EXPORT("pack_q8_weight_canonical")
int vx_pack_q8_weight_canonical(
        void* packed, uint32_t packed_bytes, const void* weight,
        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype,
        uint32_t out_in) {
    return vx_pack_q8_weight_impl(packed, packed_bytes, weight, d_in, d_out,
        weight_dtype, out_in, 1);
}
#endif

static const VxPackedQ8Header* vx_qgemm_validate(const void* packed,
        uint32_t d_in, uint32_t d_out, uint32_t weight_dtype) {
    const VxPackedQ8Header* header = (const VxPackedQ8Header*)packed;
    uint32_t expected;
    int canonical_only = 0;
    if (!header) return NULL;
#if VX_QGEMM_WASM_PAIR_PACK
    canonical_only = header->pair_n_blocks == 0u &&
        header->pair_k_blocks == 0u && header->pair_flags == 0u;
    expected = canonical_only
        ? vx_qgemm_canonical_weight_size(d_in, d_out)
        : vx_packed_q8_weight_size(d_in, d_out);
#else
    expected = vx_packed_q8_weight_size(d_in, d_out);
#endif
    if (!expected) return NULL;
    uint32_t n_blocks = (d_out - 1u) / VX_QGEMM_NR + 1u;
    uint32_t expected_sums = vx_qgemm_align16((uint32_t)sizeof(*header));
    uint32_t expected_data = vx_qgemm_align16(expected_sums +
        n_blocks * VX_QGEMM_NR * (uint32_t)sizeof(int32_t));
    uint32_t expected_pair_data = vx_qgemm_align16(expected_data +
        n_blocks * d_in * VX_QGEMM_NR);
    uint32_t expected_pair_n = 0;
    uint32_t expected_pair_k = 0;
#if VX_QGEMM_X86_AVX2
    expected_pair_n = (d_out + 7u) / 8u;
    expected_pair_k = (d_in + 3u) / 4u;
#elif VX_QGEMM_ARM_I8MM
    expected_pair_n = n_blocks;
    expected_pair_k = (d_in + 7u) / 8u;
#elif VX_QGEMM_WASM_PAIR_PACK
    if (!canonical_only) {
        expected_pair_n = n_blocks;
        expected_pair_k = (d_in + 1u) / 2u;
    }
#endif
    if (header->magic != VX_QGEMM_MAGIC ||
        header->bytes != expected || header->d_in != d_in ||
        header->d_out != d_out || header->weight_dtype != weight_dtype ||
        header->n_blocks != n_blocks || header->sums_offset != expected_sums ||
        header->data_offset != expected_data ||
        header->pair_n_blocks != expected_pair_n ||
        header->pair_k_blocks != expected_pair_k ||
        header->pair_data_offset != expected_pair_data ||
        (header->pair_flags & ~(VX_QGEMM_PAIR_SIGNED_I8 |
                                VX_QGEMM_PAIR_NO_NEG128 |
                                VX_QGEMM_PAIR_NO_SATURATE)) != 0u ||
        ((header->pair_flags & VX_QGEMM_PAIR_NO_NEG128) &&
         !(header->pair_flags & VX_QGEMM_PAIR_SIGNED_I8)) ||
        ((header->pair_flags & VX_QGEMM_PAIR_NO_SATURATE) &&
         !(header->pair_flags & VX_QGEMM_PAIR_NO_NEG128)) ||
        (header->pair_flags && weight_dtype != VX_DTYPE_I8) ||
#if VX_QGEMM_WASM_PAIR_PACK
        (!canonical_only && weight_dtype == VX_DTYPE_I8 &&
         !(header->pair_flags & VX_QGEMM_PAIR_SIGNED_I8)) ||
#endif
        (uint64_t)expected_pair_data +
            (uint64_t)expected_pair_n * expected_pair_k *
                (VX_QGEMM_ARM_I8MM ? 64u : 32u) != expected)
        return NULL;
    return header;
}

static int vx_qgemm_w8a32_m1(const float* input,
        const VxPackedQ8Header* header, const float* scale,
        const void* zero_point, const float* bias, float* output,
        uint32_t scale_elements, uint32_t zero_point_dtype,
        uint32_t zero_point_elements) {
    const uint32_t qgemm_kc = vx_qgemm_kc();
    const uint8_t* packed = (const uint8_t*)header + header->data_offset;
    for (uint32_t block = 0; block < header->n_blocks; block++) {
        double accum[VX_QGEMM_NR] = {0};
        double zero[VX_QGEMM_NR] = {0};
        uint32_t base = block * VX_QGEMM_NR;
        uint32_t lanes = header->d_out - base;
        if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
        for (uint32_t lane = 0; lane < lanes; lane++) {
            uint32_t column = base + lane;
            zero[lane] = zero_point ? vx_qgemm_typed_value(zero_point,
                zero_point_dtype, zero_point_elements == 1u ? 0u : column) : 0.0;
        }
        for (uint32_t kc = 0; kc < header->d_in; kc += qgemm_kc) {
            uint32_t kend = kc + qgemm_kc;
            if (kend > header->d_in) kend = header->d_in;
            for (uint32_t k = kc; k < kend; k++) {
                const uint8_t* weights = packed +
                    ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                double x = input[k];
                for (uint32_t lane = 0; lane < lanes; lane++) {
                    int32_t weight_value = header->weight_dtype == VX_DTYPE_I8
                        ? (int32_t)(int8_t)weights[lane] : (int32_t)weights[lane];
                    accum[lane] += x * ((double)weight_value - zero[lane]);
                }
            }
        }
        for (uint32_t lane = 0; lane < lanes; lane++) {
            uint32_t column = base + lane;
            double value = accum[lane] * vx_qgemm_f32_le(scale,
                scale_elements == 1u ? 0u : column);
            if (bias) value += vx_qgemm_f32_le(bias, column);
            output[column] = (float)value;
        }
    }
    return 1;
}

#if VX_QGEMM_WASM_SIMD

/* The F32 accumulator is intentionally limited to an audited bounded W8A32
 * numerical envelope. Other valid descriptors retain the scalar/double ABI. */
static int vx_qgemm_w8a32_m1_wasm_simd_eligible(const float* input,
        const VxPackedQ8Header* header, const float* scale,
        uint32_t scale_elements) {
    if (header->weight_dtype != VX_DTYPE_I8 || header->d_in > 1280u)
        return 0;
    for (uint32_t column = 0; column < header->d_out; column++) {
        if (vx_qgemm_f32_le(scale,
                scale_elements == 1u ? 0u : column) > 0.025f) return 0;
    }
    for (uint32_t k = 0; k < header->d_in; k++) {
        if (!vx_w8a8_finite_f32(input[k]) || input[k] < -35.0f ||
            input[k] > 35.0f) return 0;
    }
    return 1;
}

static int vx_qgemm_w8a32_m1_wasm_simd(const float* input,
        const VxPackedQ8Header* header, const float* scale,
        const float* bias, float* output, uint32_t scale_elements) {
    const uint32_t qgemm_kc = vx_qgemm_kc();
    const uint8_t* packed = (const uint8_t*)header + header->data_offset;
    uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
    for (uint32_t block = 0; block < full_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        v128_t accum_lo = wasm_f32x4_splat(0.0f);
        v128_t accum_hi = wasm_f32x4_splat(0.0f);
        v128_t correction_lo = wasm_f32x4_splat(0.0f);
        v128_t correction_hi = wasm_f32x4_splat(0.0f);
        for (uint32_t kc = 0; kc < header->d_in; kc += qgemm_kc) {
            uint32_t kend = kc + qgemm_kc;
            if (kend > header->d_in) kend = header->d_in;
            for (uint32_t k = kc; k < kend; k++) {
                const uint8_t* weights = packed +
                    ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                v128_t bytes = wasm_v128_load64_zero(weights);
                v128_t w16 = wasm_i16x8_extend_low_i8x16(bytes);
                v128_t w_lo = wasm_f32x4_convert_i32x4(
                    wasm_i32x4_extend_low_i16x8(w16));
                v128_t w_hi = wasm_f32x4_convert_i32x4(
                    wasm_i32x4_extend_high_i16x8(w16));
                v128_t x = wasm_f32x4_splat(input[k]);
                v128_t product_lo = wasm_f32x4_mul(x, w_lo);
                v128_t product_hi = wasm_f32x4_mul(x, w_hi);
                v128_t corrected_lo = wasm_f32x4_sub(product_lo, correction_lo);
                v128_t corrected_hi = wasm_f32x4_sub(product_hi, correction_hi);
                v128_t next_lo = wasm_f32x4_add(accum_lo, corrected_lo);
                v128_t next_hi = wasm_f32x4_add(accum_hi, corrected_hi);
                correction_lo = wasm_f32x4_sub(
                    wasm_f32x4_sub(next_lo, accum_lo), corrected_lo);
                correction_hi = wasm_f32x4_sub(
                    wasm_f32x4_sub(next_hi, accum_hi), corrected_hi);
                accum_lo = next_lo;
                accum_hi = next_hi;
            }
        }
        {
            v128_t scale_lo = scale_elements == 1u
                ? wasm_f32x4_splat(vx_qgemm_f32_le(scale, 0u))
                : wasm_v128_load(scale + base);
            v128_t scale_hi = scale_elements == 1u
                ? scale_lo : wasm_v128_load(scale + base + 4u);
            accum_lo = wasm_f32x4_mul(accum_lo, scale_lo);
            accum_hi = wasm_f32x4_mul(accum_hi, scale_hi);
            if (bias) {
                accum_lo = wasm_f32x4_add(accum_lo, wasm_v128_load(bias + base));
                accum_hi = wasm_f32x4_add(accum_hi,
                    wasm_v128_load(bias + base + 4u));
            }
            wasm_v128_store(output + base, accum_lo);
            wasm_v128_store(output + base + 4u, accum_hi);
        }
    }
    /* Match the fast path's compensated F32 arithmetic for an odd-N tail. */
    if (full_blocks * VX_QGEMM_NR < header->d_out) {
        uint32_t block = full_blocks;
        uint32_t base = block * VX_QGEMM_NR;
        for (uint32_t lane = 0; base + lane < header->d_out; lane++) {
            uint32_t column = base + lane;
            float accum = 0.0f;
            float correction = 0.0f;
            for (uint32_t k = 0; k < header->d_in; k++) {
                const uint8_t* weights = packed +
                    ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                int32_t weight_value = (int32_t)(int8_t)weights[lane];
                float corrected = input[k] * (float)weight_value - correction;
                float next = accum + corrected;
                correction = (next - accum) - corrected;
                accum = next;
            }
            accum *= vx_qgemm_f32_le(scale,
                scale_elements == 1u ? 0u : column);
            if (bias) accum += vx_qgemm_f32_le(bias, column);
            output[column] = (float)accum;
        }
    }
    return 1;
}
#endif

WASM_EXPORT("matmul_quantized_f32_packed")
int vx_matmul_quantized_f32_packed(const float* input, const void* packed_weight,
        const float* scale, const void* zero_point, const float* bias,
        float* output, uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t weight_dtype, uint32_t scale_elements,
        uint32_t zero_point_dtype, uint32_t zero_point_elements) {
    const uint32_t qgemm_kc = vx_qgemm_kc();
    const VxPackedQ8Header* header = vx_qgemm_validate(
        packed_weight, d_in, d_out, weight_dtype);
    size_t input_elements = rows;
    size_t output_elements = rows;
#if VX_QGEMM_WASM_SIMD
    int symmetric_weight = 1;
#endif
    if (!input || !scale || !output || !rows || !header ||
        !vx_w8a8_mul_size(&input_elements, d_in) ||
        !vx_w8a8_mul_size(&output_elements, d_out) ||
        (scale_elements != 1u && scale_elements != d_out) ||
        (zero_point && !zero_point_elements) ||
        (zero_point_elements && (!zero_point ||
            !vx_qgemm_typed_dtype(zero_point_dtype) ||
            (zero_point_elements != 1u && zero_point_elements != d_out)))) return 0;
    for (uint32_t column = 0; column < d_out; column++) {
        float s = vx_qgemm_f32_le(scale, scale_elements == 1u ? 0u : column);
        double z = zero_point ? vx_qgemm_typed_value(zero_point, zero_point_dtype,
            zero_point_elements == 1u ? 0u : column) : 0.0;
#if VX_QGEMM_WASM_SIMD
        if (z != 0.0) symmetric_weight = 0;
#endif
        if (!vx_w8a8_finite_f32(s) || s <= 0.0f ||
            (weight_dtype == VX_DTYPE_I8 && (z < -128.0 || z > 127.0)) ||
            (weight_dtype == VX_DTYPE_U8 && (z < 0.0 || z > 255.0))) return 0;
    }
    if (rows == 1u) {
#if VX_QGEMM_WASM_SIMD
        if (d_out >= VX_QGEMM_NR && symmetric_weight &&
            vx_qgemm_w8a32_m1_wasm_simd_eligible(input, header, scale,
                scale_elements)) {
            return vx_qgemm_w8a32_m1_wasm_simd(input, header, scale, bias,
                output, scale_elements);
        }
#endif
        return vx_qgemm_w8a32_m1(input, header, scale, zero_point,
            bias, output, scale_elements, zero_point_dtype,
            zero_point_elements);
    }
    {
        const uint8_t* packed = (const uint8_t*)header + header->data_offset;
        for (uint32_t block = 0; block < header->n_blocks; block++) {
            uint32_t base = block * VX_QGEMM_NR;
            uint32_t lanes = d_out - base;
            double zero[VX_QGEMM_NR] = {0};
            if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
            for (uint32_t lane = 0; lane < lanes; lane++) {
                uint32_t column = base + lane;
                zero[lane] = zero_point ? vx_qgemm_typed_value(zero_point,
                    zero_point_dtype, zero_point_elements == 1u ? 0u : column) : 0.0;
            }
            for (uint32_t row_base = 0; row_base < rows; row_base += VX_QGEMM_MR) {
                uint32_t mr = rows - row_base;
                double accum[VX_QGEMM_MR][VX_QGEMM_NR] = {{0}};
                if (mr > VX_QGEMM_MR) mr = VX_QGEMM_MR;
                for (uint32_t kc = 0; kc < d_in; kc += qgemm_kc) {
                    uint32_t kend = kc + qgemm_kc;
                    if (kend > d_in) kend = d_in;
                    for (uint32_t k = kc; k < kend; k++) {
                        const uint8_t* weights = packed +
                            ((size_t)block * d_in + k) * VX_QGEMM_NR;
                        for (uint32_t row = 0; row < mr; row++) {
                            double x = input[(size_t)(row_base + row) * d_in + k];
                            for (uint32_t lane = 0; lane < lanes; lane++) {
                                int32_t w = weight_dtype == VX_DTYPE_I8
                                    ? (int32_t)(int8_t)weights[lane] : weights[lane];
                                accum[row][lane] += x * ((double)w - zero[lane]);
                            }
                        }
                    }
                }
                for (uint32_t row = 0; row < mr; row++) {
                    for (uint32_t lane = 0; lane < lanes; lane++) {
                        uint32_t column = base + lane;
                        double value = accum[row][lane] * vx_qgemm_f32_le(scale,
                            scale_elements == 1u ? 0u : column);
                        if (bias) value += vx_qgemm_f32_le(bias, column);
                        output[(size_t)(row_base + row) * d_out + column] = (float)value;
                    }
                }
            }
        }
    }
    return 1;
}

static void vx_qgemm_store(void* output, uint32_t dtype, size_t index, int32_t value) {
    if (dtype == VX_DTYPE_I8) ((int8_t*)output)[index] = (int8_t)value;
    else ((uint8_t*)output)[index] = (uint8_t)value;
}

int vx_packed_q8_preferred_for_native_w8a8(uint32_t rows, uint32_t d_in,
        uint32_t d_out, uint32_t weight_dtype, int weight_zero_all_zero) {
#if VX_QGEMM_X86_AVX2
    if (!rows || d_out < 16u || weight_dtype != VX_DTYPE_I8 ||
        !weight_zero_all_zero || !vx_kernel_platform()->has_avx2) return 0;
    return rows > 1u || (d_in < 32u && d_out >= 32u);
#elif VX_QGEMM_ARM_I8MM
    /* Pair packing is immutable model state.  At least two rows are required
     * to fill both row halves of SMMLA; decoder GEMV keeps the SDOT path. */
    return rows >= 2u && d_out % VX_QGEMM_NR == 0u &&
        weight_dtype == VX_DTYPE_I8 && weight_zero_all_zero &&
        vx_kernel_platform()->has_arm_i8mm;
#elif VX_QGEMM_WASM_SIMD
    /* The C owner already packs immutable weights for these nodes.  Use the
     * portable SIMD128 implementation for GEMV as well as multi-row GEMM;
     * its own admission keeps asymmetric zero points and tails exact. */
    (void)weight_zero_all_zero;
    return rows > 0u && d_in > 0u && d_out >= VX_QGEMM_NR &&
        (weight_dtype == VX_DTYPE_I8 || weight_dtype == VX_DTYPE_U8);
#else
    /* Keep the existing runtime-gated NEON/SDOT dispatcher on ARM until a
     * benchmark-backed packed-N microkernel is available there. */
    (void)rows;
    (void)d_in;
    (void)d_out;
    (void)weight_dtype;
    (void)weight_zero_all_zero;
    return 0;
#endif
}

static int vx_qgemm_w8a8_simd_eligible(const VxPackedQ8Header* header,
        const int32_t* bias, int32_t input_zero_point, uint32_t input_dtype) {
    uint64_t raw_limit = input_dtype == VX_DTYPE_I8 ? 128u : 255u;
    uint64_t correction_limit = input_zero_point < 0
        ? (uint64_t)(-(int64_t)input_zero_point) : (uint64_t)input_zero_point;
    uint64_t product_bound = (uint64_t)header->d_in * 255u *
        (raw_limit + correction_limit);
    if (product_bound > (uint64_t)INT32_MAX) return 0;
    for (uint32_t column = 0; column < header->d_out; column++) {
        uint64_t bias_magnitude = bias[column] < 0
            ? (uint64_t)(-(int64_t)bias[column]) : (uint64_t)bias[column];
        if (bias_magnitude > (uint64_t)INT32_MAX - product_bound) return 0;
    }
    return 1;
}

#if VX_QGEMM_WASM_SIMD

static int vx_qgemm_w8a8_wasm_simd_eligible(
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, float input_scale,
        int32_t input_zero_point, float output_scale, uint32_t input_dtype) {
    if (!vx_qgemm_w8a8_simd_eligible(
            header, bias, input_zero_point, input_dtype)) return 0;
    /* The vector requantizer relies on finite arithmetic.  The portable path
     * retains the specified NaN fallback for otherwise valid extreme scales. */
    for (uint32_t column = 0; column < header->d_out; column++) {
        float multiplier = input_scale * weight_scales[column] / output_scale;
        if (!vx_w8a8_finite_f32(multiplier)) return 0;
    }
    return 1;
}

static v128_t vx_qgemm_w8a8_wasm_requantize(v128_t accum,
        v128_t multiplier, int32_t output_zero_point,
        int32_t output_minimum, int32_t output_maximum) {
    v128_t transformed = wasm_f32x4_add(wasm_f32x4_mul(
        wasm_f32x4_convert_i32x4(accum), multiplier),
        wasm_f32x4_splat((float)output_zero_point));
    v128_t rounded = wasm_i32x4_trunc_sat_f32x4(
        wasm_f32x4_nearest(transformed));
    return wasm_i32x4_min(wasm_i32x4_max(rounded,
        wasm_i32x4_splat(output_minimum)), wasm_i32x4_splat(output_maximum));
}

/* The materialized W8A8 model uses symmetric I8 tensors throughout its dense
 * islands.  Keep that common case separate so the K loop does not reload and
 * subtract eight zero weight zero-points for every two input elements. */
/* Activation panel for the hoisted pair-pack, in i16-pair u32 layout.
 *
 * `wasm_i32x4_dot_i16x8` wants two activations side by side as i16 lanes, which
 * the byte input does not provide. Rebuilding that pair inside the innermost
 * loop costs six scalar ops per sixteen MACs, and because the d_out block loop
 * sits outside, every activation pair is rebuilt d_out/8 times -- 160 times for
 * a 1280-wide FFN. Packing each panel once and swapping the block loop inside
 * row_base leaves a single v128.load32_splat there instead.
 *
 * The WASM build is single-threaded (vx_kernels_parallel_for runs its range
 * inline), so one module-scope panel is safe. Panels wider than this fall back
 * to the original loop rather than growing the buffer. */
enum { VX_QGEMM_WASM_PANEL_MAX_PAIRS = 4096u };
static uint32_t vx_qgemm_wasm_panel[VX_QGEMM_WASM_MR *
                                    VX_QGEMM_WASM_PANEL_MAX_PAIRS];

static int vx_qgemm_w8a8_wasm_symmetric_i8(const int8_t* input,
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, int8_t* output, uint32_t rows,
        float input_scale, float output_scale) {
    const uint8_t* pair_weights =
        (const uint8_t*)header + header->pair_data_offset;
    const uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
    const uint32_t panel_pairs = header->pair_k_blocks;
    if (panel_pairs <= VX_QGEMM_WASM_PANEL_MAX_PAIRS) {
        for (uint32_t row_base = 0; row_base < rows;
             row_base += VX_QGEMM_WASM_MR) {
            uint32_t mr = rows - row_base;
            if (mr > VX_QGEMM_WASM_MR) mr = VX_QGEMM_WASM_MR;
            /* One scalar pass over this panel's activations, reused by every
             * output block below. Rows are adjacent within a pair so the inner
             * loop can splat straight out of the panel. */
            for (uint32_t row = 0; row < mr; row++) {
                const int8_t* source = input +
                    (size_t)(row_base + row) * header->d_in;
                for (uint32_t pair = 0; pair < panel_pairs; pair++) {
                    const int32_t x0 = source[pair * 2u];
                    const int32_t x1 = pair * 2u + 1u < header->d_in
                        ? source[pair * 2u + 1u] : 0;
                    vx_qgemm_wasm_panel[(size_t)pair * VX_QGEMM_WASM_MR + row] =
                        (uint16_t)(int16_t)x0 |
                        ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
                }
            }
            for (uint32_t block = 0; block < full_blocks; block++) {
                const uint32_t base = block * VX_QGEMM_NR;
                const v128_t bias_lo = wasm_v128_load(bias + base);
                const v128_t bias_hi = wasm_v128_load(bias + base + 4u);
                const v128_t multiplier_lo = wasm_f32x4_div(wasm_f32x4_mul(
                    wasm_f32x4_splat(input_scale),
                    wasm_v128_load(weight_scales + base)),
                    wasm_f32x4_splat(output_scale));
                const v128_t multiplier_hi = wasm_f32x4_div(wasm_f32x4_mul(
                    wasm_f32x4_splat(input_scale),
                    wasm_v128_load(weight_scales + base + 4u)),
                    wasm_f32x4_splat(output_scale));
                v128_t accum_lo[VX_QGEMM_WASM_MR];
                v128_t accum_hi[VX_QGEMM_WASM_MR];
                for (uint32_t row = 0; row < mr; row++) {
                    accum_lo[row] = bias_lo;
                    accum_hi[row] = bias_hi;
                }
                for (uint32_t pair = 0; pair < panel_pairs; pair++) {
                    const uint8_t* packed = pair_weights +
                        ((size_t)block * panel_pairs + pair) * 32u;
                    const v128_t w_lo = wasm_v128_load(packed);
                    const v128_t w_hi = wasm_v128_load(packed + 16u);
                    const uint32_t* pairs = vx_qgemm_wasm_panel +
                        (size_t)pair * VX_QGEMM_WASM_MR;
                    for (uint32_t row = 0; row < mr; row++) {
                        const v128_t x = wasm_v128_load32_splat(pairs + row);
                        accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                            wasm_i32x4_dot_i16x8(x, w_lo));
                        accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                            wasm_i32x4_dot_i16x8(x, w_hi));
                    }
                }
                for (uint32_t row = 0; row < mr; row++) {
                    const v128_t quantized_lo = vx_qgemm_w8a8_wasm_requantize(
                        accum_lo[row], multiplier_lo, 0, -128, 127);
                    const v128_t quantized_hi = vx_qgemm_w8a8_wasm_requantize(
                        accum_hi[row], multiplier_hi, 0, -128, 127);
                    const v128_t quantized16 = wasm_i16x8_narrow_i32x4(
                        quantized_lo, quantized_hi);
                    const v128_t quantized8 = wasm_i8x16_narrow_i16x8(
                        quantized16, quantized16);
                    wasm_v128_store64_lane(output +
                        (size_t)(row_base + row) * header->d_out + base,
                        quantized8, 0);
                }
            }
        }
        return 1;
    }
    for (uint32_t block = 0; block < full_blocks; block++) {
        const uint32_t base = block * VX_QGEMM_NR;
        const v128_t bias_lo = wasm_v128_load(bias + base);
        const v128_t bias_hi = wasm_v128_load(bias + base + 4u);
        const v128_t multiplier_lo = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base)),
            wasm_f32x4_splat(output_scale));
        const v128_t multiplier_hi = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base + 4u)),
            wasm_f32x4_splat(output_scale));
        for (uint32_t row_base = 0; row_base < rows;
             row_base += VX_QGEMM_WASM_MR) {
            uint32_t mr = rows - row_base;
            v128_t accum_lo[VX_QGEMM_WASM_MR];
            v128_t accum_hi[VX_QGEMM_WASM_MR];
            if (mr > VX_QGEMM_WASM_MR) mr = VX_QGEMM_WASM_MR;
            for (uint32_t row = 0; row < mr; row++) {
                accum_lo[row] = bias_lo;
                accum_hi[row] = bias_hi;
            }
            for (uint32_t pair_index = 0;
                 pair_index < panel_pairs; pair_index++) {
                const uint8_t* packed = pair_weights +
                    ((size_t)block * panel_pairs + pair_index) * 32u;
                const v128_t w_lo = wasm_v128_load(packed);
                const v128_t w_hi = wasm_v128_load(packed + 16u);
                for (uint32_t row = 0; row < mr; row++) {
                    const size_t index = (size_t)(row_base + row) *
                        header->d_in + pair_index * 2u;
                    const int32_t x0 = input[index];
                    const int32_t x1 = pair_index * 2u + 1u < header->d_in
                        ? input[index + 1u] : 0;
                    const uint32_t pair = (uint16_t)(int16_t)x0 |
                        ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
                    const v128_t x = wasm_i32x4_splat((int32_t)pair);
                    accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                        wasm_i32x4_dot_i16x8(x, w_lo));
                    accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                        wasm_i32x4_dot_i16x8(x, w_hi));
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                const v128_t quantized_lo = vx_qgemm_w8a8_wasm_requantize(
                    accum_lo[row], multiplier_lo, 0, -128, 127);
                const v128_t quantized_hi = vx_qgemm_w8a8_wasm_requantize(
                    accum_hi[row], multiplier_hi, 0, -128, 127);
                const v128_t quantized16 = wasm_i16x8_narrow_i32x4(
                    quantized_lo, quantized_hi);
                const v128_t quantized8 = wasm_i8x16_narrow_i16x8(
                    quantized16, quantized16);
                wasm_v128_store64_lane(output +
                    (size_t)(row_base + row) * header->d_out + base,
                    quantized8, 0);
            }
        }
    }
    return 1;
}

/* QConv2D keeps symmetric I8 weights but ordinarily has affine U8
 * activations and outputs.  The generic SIMD spelling rebuilds each raw
 * activation pair once per N8 block and also subtracts a vector of zero
 * weight zero-points inside the K loop.  For a proved-zero I8 weight zero
 * point, pack each MR panel once just like the fully symmetric dense path and
 * fold the activation zero point into the initial accumulator:
 *
 *   sum((x-xz)w) = sum(xw) - xz*sum(w).
 *
 * Raw U8 values fit exactly in I16, so wasm_i32x4_dot_i16x8 retains the
 * canonical I32 accumulation and requantization boundaries. */
static int vx_qgemm_w8a8_wasm_symmetric_i8_affine_panel(
        const void* input, const VxPackedQ8Header* header,
        const int32_t* bias, const float* weight_scales, void* output,
        uint32_t rows, float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t output_dtype) {
    const uint32_t panel_pairs = header->pair_k_blocks;
    const uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
    const uint8_t* pair_weights =
        (const uint8_t*)header + header->pair_data_offset;
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const int32_t output_minimum =
        output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum =
        output_dtype == VX_DTYPE_I8 ? 127 : 255;
    if (panel_pairs > VX_QGEMM_WASM_PANEL_MAX_PAIRS ||
        header->d_out % VX_QGEMM_NR) return 0;
    for (uint32_t row_base = 0; row_base < rows;
         row_base += VX_QGEMM_WASM_MR) {
        uint32_t mr = rows - row_base;
        if (mr > VX_QGEMM_WASM_MR) mr = VX_QGEMM_WASM_MR;
        for (uint32_t row = 0; row < mr; row++) {
            const size_t source_base =
                (size_t)(row_base + row) * header->d_in;
            for (uint32_t pair = 0; pair < panel_pairs; pair++) {
                const size_t index = source_base + pair * 2u;
                const int32_t x0 = input_dtype == VX_DTYPE_I8
                    ? ((const int8_t*)input)[index]
                    : ((const uint8_t*)input)[index];
                const int32_t x1 = pair * 2u + 1u < header->d_in
                    ? (input_dtype == VX_DTYPE_I8
                        ? ((const int8_t*)input)[index + 1u]
                        : ((const uint8_t*)input)[index + 1u])
                    : 0;
                vx_qgemm_wasm_panel[(size_t)pair * VX_QGEMM_WASM_MR + row] =
                    (uint16_t)(int16_t)x0 |
                    ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
            }
        }
        for (uint32_t block = 0; block < full_blocks; block++) {
            const uint32_t base = block * VX_QGEMM_NR;
            const v128_t negative_input_zero =
                wasm_i32x4_splat(-input_zero_point);
            const v128_t initial_lo = wasm_i32x4_add(
                wasm_v128_load(bias + base), wasm_i32x4_mul(
                    negative_input_zero, wasm_v128_load(raw_sums + base)));
            const v128_t initial_hi = wasm_i32x4_add(
                wasm_v128_load(bias + base + 4u), wasm_i32x4_mul(
                    negative_input_zero,
                    wasm_v128_load(raw_sums + base + 4u)));
            const v128_t multiplier_lo = wasm_f32x4_div(wasm_f32x4_mul(
                wasm_f32x4_splat(input_scale),
                wasm_v128_load(weight_scales + base)),
                wasm_f32x4_splat(output_scale));
            const v128_t multiplier_hi = wasm_f32x4_div(wasm_f32x4_mul(
                wasm_f32x4_splat(input_scale),
                wasm_v128_load(weight_scales + base + 4u)),
                wasm_f32x4_splat(output_scale));
            v128_t accum_lo[VX_QGEMM_WASM_MR];
            v128_t accum_hi[VX_QGEMM_WASM_MR];
            for (uint32_t row = 0; row < mr; row++) {
                accum_lo[row] = initial_lo;
                accum_hi[row] = initial_hi;
            }
            for (uint32_t pair = 0; pair < panel_pairs; pair++) {
                const uint8_t* packed = pair_weights +
                    ((size_t)block * panel_pairs + pair) * 32u;
                const v128_t w_lo = wasm_v128_load(packed);
                const v128_t w_hi = wasm_v128_load(packed + 16u);
                const uint32_t* pairs = vx_qgemm_wasm_panel +
                    (size_t)pair * VX_QGEMM_WASM_MR;
                for (uint32_t row = 0; row < mr; row++) {
                    const v128_t x =
                        wasm_v128_load32_splat(pairs + row);
                    accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                        wasm_i32x4_dot_i16x8(x, w_lo));
                    accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                        wasm_i32x4_dot_i16x8(x, w_hi));
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                const v128_t quantized_lo = vx_qgemm_w8a8_wasm_requantize(
                    accum_lo[row], multiplier_lo, output_zero_point,
                    output_minimum, output_maximum);
                const v128_t quantized_hi = vx_qgemm_w8a8_wasm_requantize(
                    accum_hi[row], multiplier_hi, output_zero_point,
                    output_minimum, output_maximum);
                const v128_t quantized16 = wasm_i16x8_narrow_i32x4(
                    quantized_lo, quantized_hi);
                const v128_t quantized8 = output_dtype == VX_DTYPE_I8
                    ? wasm_i8x16_narrow_i16x8(quantized16, quantized16)
                    : wasm_u8x16_narrow_i16x8(quantized16, quantized16);
                wasm_v128_store64_lane((uint8_t*)output +
                    (size_t)(row_base + row) * header->d_out + base,
                    quantized8, 0);
            }
        }
    }
    return 1;
}

static int vx_qgemm_w8a8_wasm_simd(const void* input,
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, const int32_t* weight_zero_points,
        void* output, uint32_t rows, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    const uint8_t* weights_base = (const uint8_t*)header + header->data_offset;
    const uint8_t* pair_weights =
        (const uint8_t*)header + header->pair_data_offset;
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const int32_t output_minimum =
        output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum =
        output_dtype == VX_DTYPE_I8 ? 127 : 255;
    uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
    for (uint32_t block = 0; block < full_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        v128_t zp_lo = wasm_v128_load(weight_zero_points + base);
        v128_t zp_hi = wasm_v128_load(weight_zero_points + base + 4u);
        v128_t k = wasm_i32x4_splat((int32_t)header->d_in);
        v128_t centered_sum_lo = wasm_i32x4_sub(wasm_v128_load(raw_sums + base),
            wasm_i32x4_mul(k, zp_lo));
        v128_t centered_sum_hi = wasm_i32x4_sub(wasm_v128_load(raw_sums + base + 4u),
            wasm_i32x4_mul(k, zp_hi));
        v128_t negative_input_zero = wasm_i32x4_splat(-input_zero_point);
        v128_t correction_lo = wasm_i32x4_mul(negative_input_zero, centered_sum_lo);
        v128_t correction_hi = wasm_i32x4_mul(negative_input_zero, centered_sum_hi);
        v128_t zp16 = wasm_i16x8_narrow_i32x4(zp_lo, zp_hi);
        v128_t zp_pair_lo = wasm_i16x8_shuffle(
            zp16, zp16, 0, 0, 1, 1, 2, 2, 3, 3);
        v128_t zp_pair_hi = wasm_i16x8_shuffle(
            zp16, zp16, 4, 4, 5, 5, 6, 6, 7, 7);
        v128_t multiplier_lo = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base)),
            wasm_f32x4_splat(output_scale));
        v128_t multiplier_hi = wasm_f32x4_div(wasm_f32x4_mul(
            wasm_f32x4_splat(input_scale),
            wasm_v128_load(weight_scales + base + 4u)),
            wasm_f32x4_splat(output_scale));
        for (uint32_t row_base = 0; row_base < rows;
             row_base += VX_QGEMM_WASM_MR) {
            uint32_t mr = rows - row_base;
            v128_t accum_lo[VX_QGEMM_WASM_MR];
            v128_t accum_hi[VX_QGEMM_WASM_MR];
            if (mr > VX_QGEMM_WASM_MR) mr = VX_QGEMM_WASM_MR;
            for (uint32_t row = 0; row < mr; row++) {
                accum_lo[row] = wasm_i32x4_add(wasm_v128_load(bias + base), correction_lo);
                accum_hi[row] = wasm_i32x4_add(wasm_v128_load(bias + base + 4u), correction_hi);
            }
            for (uint32_t pair_index = 0;
                 pair_index < header->pair_k_blocks; pair_index++) {
                const uint8_t* packed = pair_weights +
                    ((size_t)block * header->pair_k_blocks + pair_index) *
                    32u;
                v128_t w_lo = wasm_i16x8_sub(
                    wasm_v128_load(packed), zp_pair_lo);
                v128_t w_hi = wasm_i16x8_sub(
                    wasm_v128_load(packed + 16u), zp_pair_hi);
                for (uint32_t row = 0; row < mr; row++) {
                    size_t index = (size_t)(row_base + row) *
                        header->d_in + pair_index * 2u;
                    int32_t x0 = input_dtype == VX_DTYPE_I8
                        ? ((const int8_t*)input)[index]
                        : ((const uint8_t*)input)[index];
                    int32_t x1 = pair_index * 2u + 1u < header->d_in
                        ? (input_dtype == VX_DTYPE_I8
                            ? ((const int8_t*)input)[index + 1u]
                            : ((const uint8_t*)input)[index + 1u])
                        : 0;
                    uint32_t pair = (uint16_t)(int16_t)x0 |
                        ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
                    v128_t x = wasm_i32x4_splat((int32_t)pair);
                    accum_lo[row] = wasm_i32x4_add(accum_lo[row],
                        wasm_i32x4_dot_i16x8(x, w_lo));
                    accum_hi[row] = wasm_i32x4_add(accum_hi[row],
                        wasm_i32x4_dot_i16x8(x, w_hi));
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                v128_t quantized_lo = vx_qgemm_w8a8_wasm_requantize(
                    accum_lo[row], multiplier_lo, output_zero_point,
                    output_minimum, output_maximum);
                v128_t quantized_hi = vx_qgemm_w8a8_wasm_requantize(
                    accum_hi[row], multiplier_hi, output_zero_point,
                    output_minimum, output_maximum);
                v128_t quantized16 = wasm_i16x8_narrow_i32x4(
                    quantized_lo, quantized_hi);
                v128_t quantized8 = output_dtype == VX_DTYPE_I8
                    ? wasm_i8x16_narrow_i16x8(quantized16, quantized16)
                    : wasm_u8x16_narrow_i16x8(quantized16, quantized16);
                wasm_v128_store64_lane((uint8_t*)output +
                    (size_t)(row_base + row) * header->d_out + base,
                    quantized8, 0);
            }
        }
    }
    if (full_blocks * VX_QGEMM_NR < header->d_out) {
        uint32_t base = full_blocks * VX_QGEMM_NR;
        uint32_t block = full_blocks;
        for (uint32_t row = 0; row < rows; row++) {
            for (uint32_t column = base; column < header->d_out; column++) {
                uint32_t lane = column - base;
                int64_t accumulator = bias[column];
                for (uint32_t dimension = 0; dimension < header->d_in; dimension++) {
                    size_t input_index = (size_t)row * header->d_in + dimension;
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + dimension) * VX_QGEMM_NR;
                    int32_t x = (input_dtype == VX_DTYPE_I8
                        ? ((const int8_t*)input)[input_index]
                        : ((const uint8_t*)input)[input_index]) - input_zero_point;
                    int32_t w = (header->weight_dtype == VX_DTYPE_I8
                        ? (int32_t)(int8_t)packed[lane] : packed[lane]) -
                        weight_zero_points[column];
                    accumulator += (int64_t)x * w;
                }
                {
                    float multiplier = input_scale * weight_scales[column] / output_scale;
                    int32_t quantized = vx_w8a8_requantize(
                        (float)accumulator * multiplier + (float)output_zero_point,
                        output_minimum, output_maximum, output_zero_point);
                    vx_qgemm_store(output, output_dtype,
                        (size_t)row * header->d_out + column, quantized);
                }
            }
        }
    }
    return 1;
}
#endif

#if VX_QGEMM_X86_AVX2
enum { VX_QGEMM_AVX2_PARALLEL_PRODUCTS = 1024u * 1024u };

typedef struct {
    const void* input;
    const VxPackedQ8Header* header;
    const int32_t* bias;
    const float* weight_scales;
    const int32_t* weight_zero_points;
    void* output;
    uint32_t rows;
    float input_scale;
    int32_t input_zero_point;
    float output_scale;
    int32_t output_zero_point;
    uint32_t input_dtype;
    uint32_t output_dtype;
    int weight_zero_all_zero;
} VxQGemmW8A8Avx2Call;

/* B is already persistent [N/8,K,8].  Pair adjacent K rows, widen their
 * sixteen bytes to I16, and interleave them as
 *   [w(k0,n0),w(k1,n0),...,w(k0,n7),w(k1,n7)].
 * One VPMADDWD then produces eight exact I32 output-channel products.  This
 * avoids VPMADDUBSW's I16 saturation without the two dot products needed by
 * the raw U8xI8 kernel.  Raw x/w sums move arbitrary affine zero points out
 * of the hot loop:
 *   sum((x-xz)(w-wz)) = sum(xw) - wz*sum(x)
 *                        - xz*(sum(w)-K*wz).
 */
static VX_QGEMM_TARGET_AVX2 void vx_qgemm_w8a8_avx2_range(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t block_begin, uint32_t block_end) {
    const uint32_t qgemm_kc = vx_qgemm_kc();
    const VxPackedQ8Header* header = call->header;
    const uint8_t* weights_base = (const uint8_t*)header + header->data_offset;
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const int32_t output_minimum =
        call->output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum =
        call->output_dtype == VX_DTYPE_I8 ? 127 : 255;
    const __m256 output_minimum_f32 = _mm256_set1_ps((float)output_minimum);
    const __m256 output_maximum_f32 = _mm256_set1_ps((float)output_maximum);
    const __m256 output_zero_f32 = _mm256_set1_ps((float)call->output_zero_point);
    const __m256 input_scale_f32 = _mm256_set1_ps(call->input_scale);
    const __m256 output_scale_f32 = _mm256_set1_ps(call->output_scale);
    const __m128i zero128 = _mm_setzero_si128();
    for (uint32_t block = block_begin; block < block_end; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        uint32_t lanes = header->d_out - base;
        if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
        if (lanes < VX_QGEMM_NR) break;
        __m256i zero_points = _mm256_loadu_si256(
            (const __m256i*)(const void*)(call->weight_zero_points + base));
        __m256i centered_sums = _mm256_sub_epi32(
            _mm256_loadu_si256((const __m256i*)(const void*)(raw_sums + base)),
            _mm256_mullo_epi32(_mm256_set1_epi32((int32_t)header->d_in), zero_points));
        __m256i correction = _mm256_mullo_epi32(
            _mm256_set1_epi32(-call->input_zero_point), centered_sums);
        __m256i initial = _mm256_add_epi32(_mm256_loadu_si256(
            (const __m256i*)(const void*)(call->bias + base)), correction);
        __m256 multiplier = _mm256_div_ps(_mm256_mul_ps(input_scale_f32,
            _mm256_loadu_ps(call->weight_scales + base)), output_scale_f32);
        for (uint32_t row_base = 0; row_base < call->rows;
             row_base += VX_QGEMM_MR) {
            uint32_t mr = call->rows - row_base;
            __m256i accum[VX_QGEMM_MR];
            int32_t input_sums[VX_QGEMM_MR] = {0};
            if (mr > VX_QGEMM_MR) mr = VX_QGEMM_MR;
            for (uint32_t row = 0; row < mr; row++) accum[row] = initial;
            for (uint32_t kc = 0; kc < header->d_in; kc += qgemm_kc) {
                uint32_t kend = kc + qgemm_kc;
                uint32_t k = kc;
                if (kend > header->d_in) kend = header->d_in;
                for (; k + 1u < kend; k += 2u) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                    __m128i bytes = _mm_loadu_si128(
                        (const __m128i*)(const void*)packed);
                    __m128i first = header->weight_dtype == VX_DTYPE_I8
                        ? _mm_cvtepi8_epi16(bytes)
                        : _mm_cvtepu8_epi16(bytes);
                    __m128i second_bytes = _mm_srli_si128(bytes, 8);
                    __m128i second = header->weight_dtype == VX_DTYPE_I8
                        ? _mm_cvtepi8_epi16(second_bytes)
                        : _mm_cvtepu8_epi16(second_bytes);
                    __m256i weight_pairs = _mm256_castsi128_si256(
                        _mm_unpacklo_epi16(first, second));
                    weight_pairs = _mm256_inserti128_si256(weight_pairs,
                        _mm_unpackhi_epi16(first, second), 1);
                    for (uint32_t row = 0; row < mr; row++) {
                        size_t input_index = (size_t)(row_base + row) *
                            header->d_in + k;
                        int32_t x0 = call->input_dtype == VX_DTYPE_I8
                            ? ((const int8_t*)call->input)[input_index]
                            : ((const uint8_t*)call->input)[input_index];
                        int32_t x1 = call->input_dtype == VX_DTYPE_I8
                            ? ((const int8_t*)call->input)[input_index + 1u]
                            : ((const uint8_t*)call->input)[input_index + 1u];
                        uint32_t pair = (uint16_t)(int16_t)x0 |
                            ((uint32_t)(uint16_t)(int16_t)x1 << 16u);
                        accum[row] = _mm256_add_epi32(accum[row],
                            _mm256_madd_epi16(_mm256_set1_epi32((int32_t)pair),
                                             weight_pairs));
                        input_sums[row] += x0 + x1;
                    }
                }
                if (k < kend) {
                    const uint8_t* packed = weights_base +
                        ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                    __m128i bytes = _mm_loadl_epi64(
                        (const __m128i*)(const void*)packed);
                    __m128i w16 = header->weight_dtype == VX_DTYPE_I8
                        ? _mm_cvtepi8_epi16(bytes)
                        : _mm_cvtepu8_epi16(bytes);
                    __m256i weight_pairs = _mm256_castsi128_si256(
                        _mm_unpacklo_epi16(w16, zero128));
                    weight_pairs = _mm256_inserti128_si256(weight_pairs,
                        _mm_unpackhi_epi16(w16, zero128), 1);
                    for (uint32_t row = 0; row < mr; row++) {
                        size_t input_index = (size_t)(row_base + row) *
                            header->d_in + k;
                        int32_t x = call->input_dtype == VX_DTYPE_I8
                            ? ((const int8_t*)call->input)[input_index]
                            : ((const uint8_t*)call->input)[input_index];
                        accum[row] = _mm256_add_epi32(accum[row],
                            _mm256_madd_epi16(_mm256_set1_epi32(
                                (int32_t)(uint16_t)(int16_t)x), weight_pairs));
                        input_sums[row] += x;
                    }
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                int32_t values[VX_QGEMM_NR];
                __m256i weight_zero_correction = _mm256_mullo_epi32(
                    _mm256_set1_epi32(-input_sums[row]), zero_points);
                __m256 transformed = _mm256_add_ps(_mm256_mul_ps(
                    _mm256_cvtepi32_ps(_mm256_add_epi32(
                        accum[row], weight_zero_correction)), multiplier),
                    output_zero_f32);
                transformed = _mm256_min_ps(_mm256_max_ps(
                    transformed, output_minimum_f32), output_maximum_f32);
                transformed = _mm256_round_ps(transformed,
                    _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
                _mm256_storeu_si256((__m256i*)(void*)values,
                    _mm256_cvttps_epi32(transformed));
                for (uint32_t lane = 0; lane < VX_QGEMM_NR; lane++) {
                    uint32_t column = base + lane;
                    vx_qgemm_store(call->output, call->output_dtype,
                        (size_t)(row_base + row) * header->d_out + column,
                        values[lane]);
                }
            }
        }
    }
}

/* MLAS-style K4/N16 microkernel over the producer's common symmetric I8
 * weights. The persistent representation remains K4/N8 so its second panel
 * can be omitted for odd-N tails and the portable pack remains available.
 * If the pack proves that no weight is -128, signed input bytes use
 * abs(input)*sign(weight,input): each pair is bounded by 2*128*127 and cannot
 * saturate VPMADDUBSW. General I8 weights retain the unsigned split into
 * min(a,127) plus its at-most-128 remainder. VPMADDWD combines each K4 group
 * into eight exact I32 output-channel accumulators. Affine input compensation
 * is folded into the initial accumulator from persistent weight sums. */
static VX_QGEMM_TARGET_AVX2 __attribute__((always_inline)) inline void
vx_qgemm_w8a8_avx2_store8(const VxQGemmW8A8Avx2Call* call,
        uint32_t row, uint32_t base, __m256i accumulator,
        __m256 multiplier, __m256 output_zero_f32,
        __m256 output_minimum_f32, __m256 output_maximum_f32) {
    __m256 transformed = _mm256_add_ps(_mm256_mul_ps(
        _mm256_cvtepi32_ps(accumulator), multiplier), output_zero_f32);
    transformed = _mm256_min_ps(_mm256_max_ps(
        transformed, output_minimum_f32), output_maximum_f32);
    transformed = _mm256_round_ps(transformed,
        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
    {
        const __m256i values = _mm256_cvttps_epi32(transformed);
        const __m128i values16 = _mm_packs_epi32(
            _mm256_castsi256_si128(values),
            _mm256_extracti128_si256(values, 1));
        const __m128i values8 = call->output_dtype == VX_DTYPE_I8
            ? _mm_packs_epi16(values16, values16)
            : _mm_packus_epi16(values16, values16);
        _mm_storel_epi64((__m128i*)(void*)((uint8_t*)call->output +
            (size_t)row * call->header->d_out + base), values8);
    }
}

/* Non-saturating variant, used only when packing proved every |w| <= 64.
 * It is a separate function rather than a mode inside the general kernel on
 * purpose: folding both forms into one body left the common path carrying the
 * unused form's code and measurably slowed it down.  Keeping them apart means
 * the general kernel's machine code is unchanged by this addition.
 *
 * The activation is fed to VPMADDUBSW unsigned and uncentered, so this uses the
 * plain unsigned affine: an I8 activation is shifted into U8 by the XOR and the
 * accumulator correction absorbs both that shift and the zero point. */
static VX_QGEMM_TARGET_AVX2 void vx_qgemm_w8a8_avx2_pair_range_nosat(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end) {
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
    const uint32_t input_xor_mask =
        call->input_dtype == VX_DTYPE_I8 ? 0x80808080u : 0u;
    const int32_t input_sum_correction = -(call->input_zero_point +
        (call->input_dtype == VX_DTYPE_I8 ? 128 : 0));
    /* Canonical QLinear requantization is
     *   f32(f32(input_scale * weight_scale) / output_scale),
     * the form in native/src/backends/qlinear_multiplier.h that every other
     * backend and the portable reference use.  Folding input_scale/output_scale
     * into one constant first is algebraically equal but rounds differently and
     * left this path off by one LSB on a small fraction of outputs. */
    const __m256 input_scales = _mm256_set1_ps(call->input_scale);
    const __m256 output_scales = _mm256_set1_ps(call->output_scale);
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    const uint32_t full_k_blocks = header->d_in / 4u;
    for (uint32_t group = group_begin; group < group_end; group++) {
        const uint32_t first_block = group * 2u;
        const uint32_t base = first_block * 8u;
        const __m256i initial0 = _mm256_add_epi32(
            _mm256_loadu_si256(
                (const __m256i*)(const void*)(call->bias + base)),
            _mm256_mullo_epi32(_mm256_set1_epi32(input_sum_correction),
                _mm256_loadu_si256(
                    (const __m256i*)(const void*)(raw_sums + base))));
        const __m256i initial1 = _mm256_add_epi32(
            _mm256_loadu_si256(
                (const __m256i*)(const void*)(call->bias + base + 8u)),
            _mm256_mullo_epi32(_mm256_set1_epi32(input_sum_correction),
                _mm256_loadu_si256(
                    (const __m256i*)(const void*)(raw_sums + base + 8u))));
        const __m256 multiplier0 = _mm256_div_ps(
            _mm256_mul_ps(input_scales,
                _mm256_loadu_ps(call->weight_scales + base)), output_scales);
        const __m256 multiplier1 = _mm256_div_ps(
            _mm256_mul_ps(input_scales,
                _mm256_loadu_ps(call->weight_scales + base + 8u)), output_scales);
        const uint8_t* weights0_base = pair_data +
            (size_t)first_block * header->pair_k_blocks * 32u;
        const uint8_t* weights1_base = pair_data +
            (size_t)(first_block + 1u) * header->pair_k_blocks * 32u;
        uint32_t row = 0;

#define VX_QGEMM_NOSAT_ACCUMULATE(INPUT_PTR, COUNT, ACC0, ACC1) do { \
            uint32_t vx_input_pack = 0u; \
            __builtin_memcpy(&vx_input_pack, (INPUT_PTR), (COUNT)); \
            vx_input_pack ^= input_xor_mask; \
            const __m256i vx_inputs = \
                _mm256_set1_epi32((int32_t)vx_input_pack); \
            (ACC0) = _mm256_add_epi32((ACC0), _mm256_madd_epi16( \
                _mm256_maddubs_epi16(vx_inputs, weights0), ones16)); \
            (ACC1) = _mm256_add_epi32((ACC1), _mm256_madd_epi16( \
                _mm256_maddubs_epi16(vx_inputs, weights1), ones16)); \
        } while (0)

        for (; row + 4u <= call->rows; row += 4u) {
            const uint8_t* input0 = (const uint8_t*)call->input +
                (size_t)row * header->d_in;
            const uint8_t* input1 = input0 + header->d_in;
            const uint8_t* input2 = input1 + header->d_in;
            const uint8_t* input3 = input2 + header->d_in;
            __m256i accum00 = initial0, accum01 = initial1;
            __m256i accum10 = initial0, accum11 = initial1;
            __m256i accum20 = initial0, accum21 = initial1;
            __m256i accum30 = initial0, accum31 = initial1;
            uint32_t k_block = 0;
            for (; k_block < full_k_blocks; k_block++) {
                const uint32_t dimension = k_block * 4u;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)k_block * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)k_block * 32u));
                VX_QGEMM_NOSAT_ACCUMULATE(input0 + dimension, 4u, accum00, accum01);
                VX_QGEMM_NOSAT_ACCUMULATE(input1 + dimension, 4u, accum10, accum11);
                VX_QGEMM_NOSAT_ACCUMULATE(input2 + dimension, 4u, accum20, accum21);
                VX_QGEMM_NOSAT_ACCUMULATE(input3 + dimension, 4u, accum30, accum31);
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
                VX_QGEMM_NOSAT_ACCUMULATE(input0 + dimension, tail, accum00, accum01);
                VX_QGEMM_NOSAT_ACCUMULATE(input1 + dimension, tail, accum10, accum11);
                VX_QGEMM_NOSAT_ACCUMULATE(input2 + dimension, tail, accum20, accum21);
                VX_QGEMM_NOSAT_ACCUMULATE(input3 + dimension, tail, accum30, accum31);
            }
            vx_qgemm_w8a8_avx2_store8(call, row, base, accum00, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 8u, accum01, multiplier1,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base, accum10, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base + 8u, accum11, multiplier1,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 2u, base, accum20, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 2u, base + 8u, accum21, multiplier1,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 3u, base, accum30, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 3u, base + 8u, accum31, multiplier1,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
        }
        for (; row < call->rows; row++) {
            const uint8_t* input0 = (const uint8_t*)call->input +
                (size_t)row * header->d_in;
            __m256i accum0 = initial0, accum1 = initial1;
            uint32_t k_block = 0;
            for (; k_block < full_k_blocks; k_block++) {
                const uint32_t dimension = k_block * 4u;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)k_block * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)k_block * 32u));
                VX_QGEMM_NOSAT_ACCUMULATE(input0 + dimension, 4u, accum0, accum1);
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
                VX_QGEMM_NOSAT_ACCUMULATE(input0 + dimension, tail, accum0, accum1);
            }
            vx_qgemm_w8a8_avx2_store8(call, row, base, accum0, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 8u, accum1, multiplier1,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
        }
#undef VX_QGEMM_NOSAT_ACCUMULATE
    }
}

/* AVX2 targets have enough vector registers for four N8 panels and
 * two activation rows at once.  The ordinary N16/M4 loop is balanced for the
 * general affine path, but the common signed-absolute path then rebuilds each
 * activation's magnitude once per N16 group.  N32/M2 keeps the same eight
 * independent I32 accumulators while sharing that work across twice as many
 * output channels.  Restrict this spelling to the single-threaded route: N16
 * exposes twice as many independent channel tasks to the pool. */
/* --- N32 pair kernel instantiations -------------------------------------- */
#define VX_PN32_TARGET            VX_QGEMM_TARGET_AVX2
#define VX_PN32_NAME              vx_qgemm_w8a8_avx2_pair_signed_n32_body
#define VX_PN32_ALLOW_SIGNED_ABS  1
#define VX_PN32_DOT_SIGNED_ABS(A0, A1, A2, A3) do { \
        const __m256i vx_magnitudes = _mm256_abs_epi8(vx_inputs); \
        (A0) = _mm256_add_epi32((A0), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_magnitudes, \
                _mm256_sign_epi8(weights0, vx_inputs)), ones16)); \
        (A1) = _mm256_add_epi32((A1), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_magnitudes, \
                _mm256_sign_epi8(weights1, vx_inputs)), ones16)); \
        (A2) = _mm256_add_epi32((A2), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_magnitudes, \
                _mm256_sign_epi8(weights2, vx_inputs)), ones16)); \
        (A3) = _mm256_add_epi32((A3), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_magnitudes, \
                _mm256_sign_epi8(weights3, vx_inputs)), ones16)); \
    } while (0)
#define VX_PN32_DOT_PLAIN(A0, A1, A2, A3) do { \
        (A0) = _mm256_add_epi32((A0), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_inputs, weights0), ones16)); \
        (A1) = _mm256_add_epi32((A1), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_inputs, weights1), ones16)); \
        (A2) = _mm256_add_epi32((A2), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_inputs, weights2), ones16)); \
        (A3) = _mm256_add_epi32((A3), _mm256_madd_epi16( \
            _mm256_maddubs_epi16(vx_inputs, weights3), ones16)); \
    } while (0)
#include "qgemm_pair_n32_kernel.inc"
#undef VX_PN32_TARGET
#undef VX_PN32_NAME
#undef VX_PN32_ALLOW_SIGNED_ABS
#undef VX_PN32_DOT_SIGNED_ABS
#undef VX_PN32_DOT_PLAIN

/* VPDPBUSD cannot saturate, so these tiers never need the signed-absolute
 * spelling and the compiler drops it. */
#define VX_PN32_TARGET            VX_QGEMM_TARGET_AVXVNNI
#define VX_PN32_NAME              vx_qgemm_w8a8_avxvnni_pair_signed_n32_body
#define VX_PN32_ALLOW_SIGNED_ABS  0
#define VX_PN32_DOT_SIGNED_ABS(A0, A1, A2, A3) do { (void)ones16; } while (0)
#define VX_PN32_DOT_PLAIN(A0, A1, A2, A3) do { \
        (A0) = _mm256_dpbusd_avx_epi32((A0), vx_inputs, weights0); \
        (A1) = _mm256_dpbusd_avx_epi32((A1), vx_inputs, weights1); \
        (A2) = _mm256_dpbusd_avx_epi32((A2), vx_inputs, weights2); \
        (A3) = _mm256_dpbusd_avx_epi32((A3), vx_inputs, weights3); \
    } while (0)
#include "qgemm_pair_n32_kernel.inc"
#undef VX_PN32_TARGET
#undef VX_PN32_NAME
#undef VX_PN32_ALLOW_SIGNED_ABS
#undef VX_PN32_DOT_SIGNED_ABS
#undef VX_PN32_DOT_PLAIN

#define VX_PN32_TARGET            VX_QGEMM_TARGET_AVX512VNNI
#define VX_PN32_NAME              vx_qgemm_w8a8_avx512vnni_pair_signed_n32_body
#define VX_PN32_ALLOW_SIGNED_ABS  0
#define VX_PN32_DOT_SIGNED_ABS(A0, A1, A2, A3) do { (void)ones16; } while (0)
#define VX_PN32_DOT_PLAIN(A0, A1, A2, A3) do { \
        (A0) = _mm256_dpbusd_epi32((A0), vx_inputs, weights0); \
        (A1) = _mm256_dpbusd_epi32((A1), vx_inputs, weights1); \
        (A2) = _mm256_dpbusd_epi32((A2), vx_inputs, weights2); \
        (A3) = _mm256_dpbusd_epi32((A3), vx_inputs, weights3); \
    } while (0)
#include "qgemm_pair_n32_kernel.inc"
#undef VX_PN32_TARGET
#undef VX_PN32_NAME
#undef VX_PN32_ALLOW_SIGNED_ABS
#undef VX_PN32_DOT_SIGNED_ABS
#undef VX_PN32_DOT_PLAIN

/* The activation remap is one general-purpose XOR per broadcast operand, and it
 * sits on the dependency chain feeding every PMADDUBSW.  Passing the mask as a
 * literal lets an already-remapped tensor compile to a plain broadcast from
 * memory: the scalar load, the GPR xor and the GPR-to-XMM transfer all fold
 * away.  The two instantiations are otherwise identical, so the only cost is
 * code size. */
/* One dispatch macro so the four (spelling x activation domain) combinations
 * stay literal at every tier; passing either as a runtime value stops the
 * remap and the dead spelling from folding away. */
#define VX_PN32_DISPATCH(BODY) do { \
        if (use_signed_abs) { \
            if (call->input_dtype == VX_DTYPE_U8) \
                BODY(call, group_begin, group_end, 0x80808080u, 1); \
            else \
                BODY(call, group_begin, group_end, 0u, 1); \
        } else { \
            if (call->input_dtype == VX_DTYPE_I8) \
                BODY(call, group_begin, group_end, 0x80808080u, 0); \
            else \
                BODY(call, group_begin, group_end, 0u, 0); \
        } \
    } while (0)

static VX_QGEMM_TARGET_AVX2 void vx_qgemm_w8a8_avx2_pair_signed_n32_avx2(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end, int use_signed_abs) {
    VX_PN32_DISPATCH(vx_qgemm_w8a8_avx2_pair_signed_n32_body);
}

static VX_QGEMM_TARGET_AVXVNNI void vx_qgemm_w8a8_avx2_pair_signed_n32_avxvnni(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end, int use_signed_abs) {
    (void)use_signed_abs;
    use_signed_abs = 0;
    VX_PN32_DISPATCH(vx_qgemm_w8a8_avxvnni_pair_signed_n32_body);
}

static VX_QGEMM_TARGET_AVX512VNNI void
vx_qgemm_w8a8_avx2_pair_signed_n32_avx512vnni(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end, int use_signed_abs) {
    (void)use_signed_abs;
    use_signed_abs = 0;
    VX_PN32_DISPATCH(vx_qgemm_w8a8_avx512vnni_pair_signed_n32_body);
}

static int vx_qgemm_w8a8_n32_uses_vnni(void) {
    const VxKernelPlatform* platform = vx_kernel_platform();
    return platform->has_avx_vnni || platform->has_avx512_vnni;
}

/* VPDPBUSD accumulates into I32, so a VNNI tier is both faster and free of the
 * saturation question: it needs neither the proved weight bound nor the
 * magnitude/sign fallback. Select it through the resolved platform so the
 * VOLVOXAI_CPU_ISA ceiling remains authoritative for deployments.
 * The VEX form is preferred where present because it avoids the frequency
 * behaviour of EVEX-encoded 512-bit state on some parts; the EVEX form covers
 * Ice Lake and Tiger Lake, which have AVX-512-VNNI but no AVX-VNNI. */
static VX_QGEMM_TARGET_AVX2 void vx_qgemm_w8a8_avx2_pair_signed_n32_range(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end, int use_signed_abs) {
    const VxKernelPlatform* platform = vx_kernel_platform();
    if (platform->has_avx_vnni) {
        vx_qgemm_w8a8_avx2_pair_signed_n32_avxvnni(
            call, group_begin, group_end, use_signed_abs);
        return;
    }
    if (platform->has_avx512_vnni) {
        vx_qgemm_w8a8_avx2_pair_signed_n32_avx512vnni(
            call, group_begin, group_end, use_signed_abs);
        return;
    }
    vx_qgemm_w8a8_avx2_pair_signed_n32_avx2(
        call, group_begin, group_end, use_signed_abs);
}

static VX_QGEMM_TARGET_AVX2 void vx_qgemm_w8a8_avx2_pair_range(
        const VxQGemmW8A8Avx2Call* call,
        uint32_t group_begin, uint32_t group_end) {
    const VxPackedQ8Header* header = call->header;
    const uint8_t* pair_data =
        (const uint8_t*)header + header->pair_data_offset;
    const int32_t output_minimum =
        call->output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum =
        call->output_dtype == VX_DTYPE_I8 ? 127 : 255;
    const __m256 output_minimum_f32 = _mm256_set1_ps((float)output_minimum);
    const __m256 output_maximum_f32 = _mm256_set1_ps((float)output_maximum);
    const __m256 output_zero_f32 = _mm256_set1_ps((float)call->output_zero_point);
    const __m256i split_point = _mm256_set1_epi8(127);
    const __m256i ones16 = _mm256_set1_epi16(1);
    const int use_signed_abs =
        (header->pair_flags & VX_QGEMM_PAIR_NO_NEG128) != 0u;
    const uint32_t input_xor_mask = use_signed_abs
        ? (call->input_dtype == VX_DTYPE_U8 ? 0x80808080u : 0u)
        : (call->input_dtype == VX_DTYPE_I8 ? 0x80808080u : 0u);
    const int32_t input_sum_correction = use_signed_abs
        ? (call->input_dtype == VX_DTYPE_U8
            ? 128 - call->input_zero_point : -call->input_zero_point)
        : -(call->input_zero_point +
            (call->input_dtype == VX_DTYPE_I8 ? 128 : 0));
    /* Canonical QLinear requantization is
     *   f32(f32(input_scale * weight_scale) / output_scale),
     * the form in native/src/backends/qlinear_multiplier.h that every other
     * backend and the portable reference use.  Folding input_scale/output_scale
     * into one constant first is algebraically equal but rounds differently and
     * left this path off by one LSB on a small fraction of outputs. */
    const __m256 input_scales = _mm256_set1_ps(call->input_scale);
    const __m256 output_scales = _mm256_set1_ps(call->output_scale);
    const int32_t* raw_sums = (const int32_t*)((const uint8_t*)header +
        header->sums_offset);
    for (uint32_t group = group_begin; group < group_end; group++) {
        const uint32_t first_block = group * 2u;
        const uint32_t base = first_block * 8u;
        const __m256i initial0 = _mm256_add_epi32(
            _mm256_loadu_si256(
                (const __m256i*)(const void*)(call->bias + base)),
            _mm256_mullo_epi32(_mm256_set1_epi32(input_sum_correction),
                _mm256_loadu_si256(
                    (const __m256i*)(const void*)(raw_sums + base))));
        const __m256i initial1 = _mm256_add_epi32(
            _mm256_loadu_si256(
                (const __m256i*)(const void*)(call->bias + base + 8u)),
            _mm256_mullo_epi32(_mm256_set1_epi32(input_sum_correction),
                _mm256_loadu_si256(
                    (const __m256i*)(const void*)(raw_sums + base + 8u))));
        const __m256 multiplier0 = _mm256_div_ps(
            _mm256_mul_ps(input_scales,
                _mm256_loadu_ps(call->weight_scales + base)), output_scales);
        const __m256 multiplier1 = _mm256_div_ps(
            _mm256_mul_ps(input_scales,
                _mm256_loadu_ps(call->weight_scales + base + 8u)), output_scales);
        const uint8_t* weights0_base = pair_data +
            (size_t)first_block * header->pair_k_blocks * 32u;
        const uint8_t* weights1_base = pair_data +
            (size_t)(first_block + 1u) * header->pair_k_blocks * 32u;
        uint32_t row = 0;

#define VX_QGEMM_PAIR_ACCUMULATE(INPUT_PTR, COUNT, ACC0, ACC1) do { \
            uint32_t vx_input_pack = 0u; \
            __builtin_memcpy(&vx_input_pack, (INPUT_PTR), (COUNT)); \
            vx_input_pack ^= input_xor_mask; \
            const __m256i vx_inputs = \
                _mm256_set1_epi32((int32_t)vx_input_pack); \
            if (use_signed_abs) { \
                const __m256i vx_input_magnitudes = \
                    _mm256_abs_epi8(vx_inputs); \
                (ACC0) = _mm256_add_epi32((ACC0), _mm256_madd_epi16( \
                    _mm256_maddubs_epi16(vx_input_magnitudes, \
                        _mm256_sign_epi8(weights0, vx_inputs)), ones16)); \
                (ACC1) = _mm256_add_epi32((ACC1), _mm256_madd_epi16( \
                    _mm256_maddubs_epi16(vx_input_magnitudes, \
                        _mm256_sign_epi8(weights1, vx_inputs)), ones16)); \
            } else if ((vx_input_pack & 0x80808080u) == 0u) { \
                (ACC0) = _mm256_add_epi32((ACC0), _mm256_madd_epi16( \
                    _mm256_maddubs_epi16(vx_inputs, weights0), ones16)); \
                (ACC1) = _mm256_add_epi32((ACC1), _mm256_madd_epi16( \
                    _mm256_maddubs_epi16(vx_inputs, weights1), ones16)); \
            } else { \
                const __m256i vx_input_low = \
                    _mm256_min_epu8(vx_inputs, split_point); \
                const __m256i vx_input_high = \
                    _mm256_sub_epi8(vx_inputs, vx_input_low); \
                (ACC0) = _mm256_add_epi32((ACC0), _mm256_add_epi32( \
                    _mm256_madd_epi16(_mm256_maddubs_epi16( \
                        vx_input_low, weights0), ones16), \
                    _mm256_madd_epi16(_mm256_maddubs_epi16( \
                        vx_input_high, weights0), ones16))); \
                (ACC1) = _mm256_add_epi32((ACC1), _mm256_add_epi32( \
                    _mm256_madd_epi16(_mm256_maddubs_epi16( \
                        vx_input_low, weights1), ones16), \
                    _mm256_madd_epi16(_mm256_maddubs_epi16( \
                        vx_input_high, weights1), ones16))); \
            } \
        } while (0)

        for (; row + 4u <= call->rows; row += 4u) {
            const uint8_t* input0 = (const uint8_t*)call->input +
                (size_t)row * header->d_in;
            const uint8_t* input1 = input0 + header->d_in;
            const uint8_t* input2 = input1 + header->d_in;
            const uint8_t* input3 = input2 + header->d_in;
            __m256i accum00 = initial0, accum01 = initial1;
            __m256i accum10 = initial0, accum11 = initial1;
            __m256i accum20 = initial0, accum21 = initial1;
            __m256i accum30 = initial0, accum31 = initial1;
            const uint32_t full_k_blocks = header->d_in / 4u;
            uint32_t k_block = 0;
            for (; k_block < full_k_blocks; k_block++) {
                const uint32_t dimension = k_block * 4u;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)k_block * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)k_block * 32u));
                VX_QGEMM_PAIR_ACCUMULATE(input0 + dimension, 4u,
                                         accum00, accum01);
                VX_QGEMM_PAIR_ACCUMULATE(input1 + dimension, 4u,
                                         accum10, accum11);
                VX_QGEMM_PAIR_ACCUMULATE(input2 + dimension, 4u,
                                         accum20, accum21);
                VX_QGEMM_PAIR_ACCUMULATE(input3 + dimension, 4u,
                                         accum30, accum31);
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
                VX_QGEMM_PAIR_ACCUMULATE(input0 + dimension, tail,
                                         accum00, accum01);
                VX_QGEMM_PAIR_ACCUMULATE(input1 + dimension, tail,
                                         accum10, accum11);
                VX_QGEMM_PAIR_ACCUMULATE(input2 + dimension, tail,
                                         accum20, accum21);
                VX_QGEMM_PAIR_ACCUMULATE(input3 + dimension, tail,
                                         accum30, accum31);
            }
            vx_qgemm_w8a8_avx2_store8(call, row, base, accum00, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 8u, accum01,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base, accum10,
                multiplier0, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 1u, base + 8u, accum11,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 2u, base, accum20,
                multiplier0, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 2u, base + 8u, accum21,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 3u, base, accum30,
                multiplier0, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row + 3u, base + 8u, accum31,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
        }
        for (; row < call->rows; row++) {
            const uint8_t* input0 = (const uint8_t*)call->input +
                (size_t)row * header->d_in;
            __m256i accum0 = initial0, accum1 = initial1;
            const uint32_t full_k_blocks = header->d_in / 4u;
            uint32_t k_block = 0;
            for (; k_block < full_k_blocks; k_block++) {
                const uint32_t dimension = k_block * 4u;
                const __m256i weights0 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights0_base + (size_t)k_block * 32u));
                const __m256i weights1 = _mm256_loadu_si256(
                    (const __m256i*)(const void*)
                    (weights1_base + (size_t)k_block * 32u));
                VX_QGEMM_PAIR_ACCUMULATE(input0 + dimension, 4u,
                                         accum0, accum1);
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
                VX_QGEMM_PAIR_ACCUMULATE(input0 + dimension, tail,
                                         accum0, accum1);
            }
            vx_qgemm_w8a8_avx2_store8(call, row, base, accum0, multiplier0,
                output_zero_f32, output_minimum_f32, output_maximum_f32);
            vx_qgemm_w8a8_avx2_store8(call, row, base + 8u, accum1,
                multiplier1, output_zero_f32, output_minimum_f32,
                output_maximum_f32);
        }
#undef VX_QGEMM_PAIR_ACCUMULATE
    }
}

static void vx_qgemm_w8a8_avx2_worker(void* opaque, int begin, int end) {
    vx_qgemm_w8a8_avx2_range((const VxQGemmW8A8Avx2Call*)opaque,
                              (uint32_t)begin, (uint32_t)end);
}

static void vx_qgemm_w8a8_avx2_pair_worker(
        void* opaque, int begin, int end) {
    vx_qgemm_w8a8_avx2_pair_range(
        (const VxQGemmW8A8Avx2Call*)opaque,
        (uint32_t)begin, (uint32_t)end);
}

static void vx_qgemm_w8a8_avx2_pair_nosat_worker(
        void* opaque, int begin, int end) {
    vx_qgemm_w8a8_avx2_pair_range_nosat(
        (const VxQGemmW8A8Avx2Call*)opaque,
        (uint32_t)begin, (uint32_t)end);
}

static int vx_qgemm_w8a8_avx2(
        const void* input, const VxPackedQ8Header* header,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output, uint32_t rows,
        float input_scale, int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype) {
    const uint8_t* weights_base =
        (const uint8_t*)header + header->data_offset;
    const int32_t output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    const int32_t output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
    int weight_zero_all_zero = 1;
    for (uint32_t column = 0; column < header->d_out; column++) {
        if (weight_zero_points[column] != 0) {
            weight_zero_all_zero = 0;
            break;
        }
    }
    const VxQGemmW8A8Avx2Call call = {
        input, header, bias, weight_scales, weight_zero_points, output, rows,
        input_scale, input_zero_point, output_scale, output_zero_point,
        input_dtype, output_dtype, weight_zero_all_zero,
    };
    const int threads = vx_kernels_thread_count();
    const uint64_t products = (uint64_t)rows * header->d_in * header->d_out;
    uint32_t tail_base;
    if ((header->pair_flags & VX_QGEMM_PAIR_SIGNED_I8) &&
        header->pair_n_blocks && header->pair_k_blocks &&
        weight_zero_all_zero && header->d_out >= 16u) {
        const uint32_t full_blocks = header->d_out / 8u;
        const uint32_t full_groups = full_blocks / 2u;
        /* Choose the accumulation form once, outside every loop. */
        const int no_saturate =
            (header->pair_flags & VX_QGEMM_PAIR_NO_SATURATE) != 0u;
        const int signed_absolute =
            (header->pair_flags & VX_QGEMM_PAIR_NO_NEG128) != 0u;
        const int n32_vnni = vx_qgemm_w8a8_n32_uses_vnni();
        if (threads == 1 && (n32_vnni || signed_absolute) &&
            full_blocks >= 4u) {
            const uint32_t n32_groups = full_blocks / 4u;
            const uint32_t n16_begin = n32_groups * 2u;
            vx_qgemm_w8a8_avx2_pair_signed_n32_range(
                &call, 0u, n32_groups, signed_absolute && !no_saturate);
            if (n16_begin < full_groups) {
                if (no_saturate)
                    vx_qgemm_w8a8_avx2_pair_range_nosat(
                        &call, n16_begin, full_groups);
                else
                    vx_qgemm_w8a8_avx2_pair_range(
                        &call, n16_begin, full_groups);
            }
        } else if (threads > 1 && full_groups > 1u &&
            products >= VX_QGEMM_AVX2_PARALLEL_PRODUCTS) {
            uint32_t grain =
                (full_groups + (uint32_t)threads * 4u - 1u) /
                ((uint32_t)threads * 4u);
            if (!grain) grain = 1u;
            vx_kernels_parallel_for((int)full_groups, (int)grain,
                                    no_saturate
                                        ? vx_qgemm_w8a8_avx2_pair_nosat_worker
                                        : vx_qgemm_w8a8_avx2_pair_worker,
                                    (void*)&call);
        } else if (no_saturate) {
            vx_qgemm_w8a8_avx2_pair_range_nosat(&call, 0u, full_groups);
        } else {
            vx_qgemm_w8a8_avx2_pair_range(&call, 0u, full_groups);
        }
        tail_base = full_groups * 16u;
    } else {
        const uint32_t full_blocks = header->d_out / VX_QGEMM_NR;
        if (threads > 1 && full_blocks > 1u &&
            products >= VX_QGEMM_AVX2_PARALLEL_PRODUCTS) {
            uint32_t grain =
                (full_blocks + (uint32_t)threads * 4u - 1u) /
                ((uint32_t)threads * 4u);
            grain = ((grain + 7u) / 8u) * 8u;
            if (!grain) grain = 8u;
            vx_kernels_parallel_for((int)full_blocks, (int)grain,
                                    vx_qgemm_w8a8_avx2_worker,
                                    (void*)&call);
        } else {
            vx_qgemm_w8a8_avx2_range(&call, 0u, full_blocks);
        }
        tail_base = full_blocks * VX_QGEMM_NR;
    }
    /* Odd N tails keep the exact scalar K order. */
    {
        if (tail_base < header->d_out) {
            for (uint32_t row = 0; row < rows; row++) {
                for (uint32_t column = tail_base; column < header->d_out; column++) {
                    uint32_t block = column / VX_QGEMM_NR;
                    uint32_t lane = column % VX_QGEMM_NR;
                    int64_t accumulator = bias[column];
                    for (uint32_t k = 0; k < header->d_in; k++) {
                        size_t input_index = (size_t)row * header->d_in + k;
                        const uint8_t* packed = weights_base +
                            ((size_t)block * header->d_in + k) * VX_QGEMM_NR;
                        int32_t x = (input_dtype == VX_DTYPE_I8
                            ? ((const int8_t*)input)[input_index]
                            : ((const uint8_t*)input)[input_index]) - input_zero_point;
                        int32_t w = (header->weight_dtype == VX_DTYPE_I8
                            ? (int32_t)(int8_t)packed[lane] : packed[lane]) -
                            weight_zero_points[column];
                        accumulator += (int64_t)x * w;
                    }
                    {
                        float multiplier = input_scale * weight_scales[column] / output_scale;
                        int32_t quantized = vx_w8a8_requantize(
                            (float)accumulator * multiplier + (float)output_zero_point,
                            output_minimum, output_maximum, output_zero_point);
                        vx_qgemm_store(output, output_dtype,
                            (size_t)row * header->d_out + column, quantized);
                    }
                }
            }
        }
    }
    return 1;
}
#endif

/* Which activation domain the proved spelling wants, for callers that are
 * already writing the activation buffer and can hand it over in that domain for
 * free -- today the QConv im2col.  Returning 1 means the signed-absolute
 * spelling will run and wants bytes remapped to the signed domain; 0 means the
 * plain spelling will run and wants them left unsigned.  Keep this beside the
 * dispatcher below: the two encode the same decision. */
#if VX_QGEMM_X86_AVX2
int vx_packed_q8_prefers_signed_activations(const void* packed_weight,
        const int32_t* weight_zero_points, uint32_t output_channels) {
    const VxPackedQ8Header* header = (const VxPackedQ8Header*)packed_weight;
    const VxKernelPlatform* platform = vx_kernel_platform();
    if (!header || !weight_zero_points || !platform->has_avx2 ||
        !(header->pair_flags & VX_QGEMM_PAIR_SIGNED_I8) ||
        !header->pair_n_blocks || !header->pair_k_blocks ||
        header->d_out < 16u || header->d_out != output_channels) return 0;
    for (uint32_t column = 0; column < output_channels; column++)
        if (weight_zero_points[column] != 0) return 0;
    if (header->pair_flags & VX_QGEMM_PAIR_NO_SATURATE) return 0;
    if (!(header->pair_flags & VX_QGEMM_PAIR_NO_NEG128)) return 0;
    /* A single-thread N32 VNNI body consumes ordinary U8 bytes.  Pre-remapping
     * them would make that body XOR every K4 pack back again.  Multi-threaded
     * execution intentionally uses the N16 AVX2 spelling, and sub-N32 output
     * widths never reach the VNNI body, so those retain the signed-domain
     * im2col optimization. */
    if (vx_kernels_thread_count() == 1 && header->d_out / 8u >= 4u &&
        vx_qgemm_w8a8_n32_uses_vnni()) return 0;
    return 1;
}
#endif

WASM_EXPORT("qlinear_i8u8_packed")
int vx_qlinear_i8u8_packed(const void* input, const void* packed_weight,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype) {
    const uint32_t qgemm_kc = vx_qgemm_kc();
    const VxPackedQ8Header* header = vx_qgemm_validate(
        packed_weight, d_in, d_out, weight_dtype);
    size_t input_elements = rows;
    size_t output_elements = rows;
    const uint8_t* packed;
    int32_t output_minimum = output_dtype == VX_DTYPE_I8 ? -128 : 0;
    int32_t output_maximum = output_dtype == VX_DTYPE_I8 ? 127 : 255;
#if VX_QGEMM_WASM_SIMD
    int symmetric_i8 = input_dtype == VX_DTYPE_I8 &&
        weight_dtype == VX_DTYPE_I8 && output_dtype == VX_DTYPE_I8 &&
        input_zero_point == 0 && output_zero_point == 0 &&
        d_out % VX_QGEMM_NR == 0u;
    int symmetric_i8_weight = weight_dtype == VX_DTYPE_I8 &&
        d_out % VX_QGEMM_NR == 0u;
#endif
    if (!input || !bias || !weight_scales || !weight_zero_points || !output ||
        !rows || !header || !vx_w8a8_byte_dtype(input_dtype) ||
        !vx_w8a8_mul_size(&input_elements, d_in) ||
        !vx_w8a8_mul_size(&output_elements, d_out) ||
        !vx_w8a8_byte_dtype(output_dtype) ||
        !vx_w8a8_finite_f32(input_scale) || input_scale <= 0.0f ||
        !vx_w8a8_finite_f32(output_scale) || output_scale <= 0.0f ||
        !vx_w8a8_zero_point_valid(input_zero_point, input_dtype) ||
        !vx_w8a8_zero_point_valid(output_zero_point, output_dtype)) return 0;
    for (uint32_t column = 0; column < d_out; column++) {
        if (!vx_w8a8_finite_f32(weight_scales[column]) ||
            weight_scales[column] <= 0.0f ||
            !vx_qgemm_positive_product_multiplier(
                input_scale, weight_scales[column], output_scale) ||
            !vx_w8a8_zero_point_valid(weight_zero_points[column], weight_dtype)) return 0;
#if VX_QGEMM_WASM_SIMD
        if (weight_zero_points[column] != 0) {
            symmetric_i8 = 0;
            symmetric_i8_weight = 0;
        }
#endif
    }
    packed = (const uint8_t*)header + header->data_offset;
#if VX_QGEMM_WASM_SIMD
    if (header->pair_n_blocks && header->pair_k_blocks &&
        d_out >= VX_QGEMM_NR &&
        vx_qgemm_w8a8_wasm_simd_eligible(header, bias, weight_scales,
            input_scale, input_zero_point, output_scale, input_dtype)) {
        if (symmetric_i8) {
            return vx_qgemm_w8a8_wasm_symmetric_i8(
                (const int8_t*)input, header, bias, weight_scales,
                (int8_t*)output, rows, input_scale, output_scale);
        }
        if (symmetric_i8_weight &&
            vx_qgemm_w8a8_wasm_symmetric_i8_affine_panel(
                input, header, bias, weight_scales, output, rows,
                input_scale, input_zero_point, output_scale,
                output_zero_point, input_dtype, output_dtype)) return 1;
        return vx_qgemm_w8a8_wasm_simd(input, header, bias, weight_scales,
            weight_zero_points, output, rows, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, output_dtype);
    }
#endif
#if VX_QGEMM_X86_AVX2
    if (d_out >= VX_QGEMM_NR && vx_kernel_platform()->has_avx2 &&
        vx_qgemm_w8a8_simd_eligible(header, bias, input_zero_point, input_dtype)) {
        return vx_qgemm_w8a8_avx2(input, header, bias, weight_scales,
            weight_zero_points, output, rows, input_scale, input_zero_point,
            output_scale, output_zero_point, input_dtype, output_dtype);
    }
#endif
#if VX_QGEMM_ARM_I8MM
    if (d_out >= VX_QGEMM_NR && vx_kernel_platform()->has_arm_i8mm &&
        vx_qgemm_w8a8_simd_eligible(
            header, bias, input_zero_point, input_dtype) &&
        vx_qlinear_i8u8_arm_i8mm_packed_try(input, header, bias,
            weight_scales, weight_zero_points, output, rows, input_scale,
            input_zero_point, output_scale, output_zero_point, input_dtype,
            output_dtype)) return 1;
#endif
    for (uint32_t block = 0; block < header->n_blocks; block++) {
        uint32_t base = block * VX_QGEMM_NR;
        uint32_t lanes = d_out - base;
        if (lanes > VX_QGEMM_NR) lanes = VX_QGEMM_NR;
        for (uint32_t row_base = 0; row_base < rows;
             row_base += rows == 1u ? 1u : VX_QGEMM_MR) {
            uint32_t mr = rows - row_base;
            int64_t accum[VX_QGEMM_MR][VX_QGEMM_NR] = {{0}};
            if (mr > VX_QGEMM_MR) mr = VX_QGEMM_MR;
            for (uint32_t row = 0; row < mr; row++)
                for (uint32_t lane = 0; lane < lanes; lane++)
                    accum[row][lane] = bias[base + lane];
            for (uint32_t kc = 0; kc < d_in; kc += qgemm_kc) {
                uint32_t kend = kc + qgemm_kc;
                if (kend > d_in) kend = d_in;
                for (uint32_t k = kc; k < kend; k++) {
                    const uint8_t* weights = packed +
                        ((size_t)block * d_in + k) * VX_QGEMM_NR;
                    for (uint32_t row = 0; row < mr; row++) {
                        int32_t x = vx_w8a8_byte_value(input, input_dtype,
                            (size_t)(row_base + row) * d_in + k) - input_zero_point;
                        for (uint32_t lane = 0; lane < lanes; lane++) {
                            int32_t w = (weight_dtype == VX_DTYPE_I8
                                ? (int32_t)(int8_t)weights[lane] : weights[lane]) -
                                weight_zero_points[base + lane];
                            accum[row][lane] += (int64_t)x * w;
                            if (accum[row][lane] < INT32_MIN ||
                                accum[row][lane] > INT32_MAX) return 0;
                        }
                    }
                }
            }
            for (uint32_t row = 0; row < mr; row++) {
                for (uint32_t lane = 0; lane < lanes; lane++) {
                    uint32_t column = base + lane;
                    float multiplier = input_scale * weight_scales[column] / output_scale;
                    float transformed = (float)accum[row][lane] * multiplier +
                        (float)output_zero_point;
                    int32_t quantized = vx_w8a8_requantize(transformed,
                        output_minimum, output_maximum, output_zero_point);
                    vx_qgemm_store(output, output_dtype,
                        (size_t)(row_base + row) * d_out + column, quantized);
                }
            }
        }
    }
    return 1;
}
