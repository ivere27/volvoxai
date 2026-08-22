#ifndef VOLVOXAI_PACKED_QUANT_GEMM_H
#define VOLVOXAI_PACKED_QUANT_GEMM_H

#include <stddef.h>
#include <stdint.h>
#include "../../include/volvoxai_enums.h"

/*
 * Private in-memory V8Q2 contract shared by the baseline parent and the
 * optional Relaxed-SIMD child.  Packs are derived runtime state and are never
 * serialized or exposed through the public C ABI.
 */
enum {
    VX_PACKED_Q8_MAGIC = 0x32513856u, /* "V8Q2" in little endian. */
    VX_PACKED_Q8_NR = 8u,
};

typedef struct {
    uint32_t magic;
    uint32_t bytes;
    uint32_t d_in;
    uint32_t d_out;
    uint32_t n_blocks;
    uint32_t weight_dtype;
    uint32_t sums_offset;
    uint32_t data_offset;
    uint32_t pair_n_blocks;
    uint32_t pair_k_blocks;
    uint32_t pair_data_offset;
    uint32_t pair_flags;
} VxPackedQ8Header;

#if defined(__cplusplus)
static_assert(sizeof(VxPackedQ8Header) == 48u,
              "V8Q2 packed header size must remain stable");
static_assert(offsetof(VxPackedQ8Header, sums_offset) == 24u,
              "V8Q2 sums offset field must remain stable");
static_assert(offsetof(VxPackedQ8Header, pair_n_blocks) == 32u,
              "V8Q2 extension offset must remain stable");
#else
_Static_assert(sizeof(VxPackedQ8Header) == 48u,
               "V8Q2 packed header size must remain stable");
_Static_assert(offsetof(VxPackedQ8Header, sums_offset) == 24u,
               "V8Q2 sums offset field must remain stable");
_Static_assert(offsetof(VxPackedQ8Header, pair_n_blocks) == 32u,
               "V8Q2 extension offset must remain stable");
#endif

#ifdef __cplusplus
extern "C" {
#endif

/* Internal packed-Q8 format used by native CPU and WASM dense kernels.
 * The format is deliberately not a public ABI: tile sizes may change after
 * benchmark evidence without changing model files or the public C surface. */
uint32_t vx_packed_q8_weight_size(uint32_t d_in, uint32_t d_out);
int vx_pack_q8_weight(void* packed, uint32_t packed_bytes, const void* weight,
                      uint32_t d_in, uint32_t d_out, uint32_t weight_dtype,
                      uint32_t out_in);

int vx_matmul_quantized_f32_packed(const float* input, const void* packed_weight,
        const float* scale, const void* zero_point, const float* bias,
        float* output, uint32_t rows, uint32_t d_in, uint32_t d_out,
        uint32_t weight_dtype, uint32_t scale_elements,
        uint32_t zero_point_dtype, uint32_t zero_point_elements);

int vx_packed_q8_prefers_signed_activations(const void* packed_weight,
        const int32_t* weight_zero_points, uint32_t output_channels);

int vx_qlinear_i8u8_packed(const void* input, const void* packed_weight,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype);

/* Separately compiled Arm I8MM consumer for the target-derived pair payload.
 * The baseline dispatcher calls it only after HWCAP2_I8MM admission and full
 * packed-header/descriptor validation. */
#if defined(VOLVOXAI_ARM_I8MM_OBJECT) && defined(__aarch64__)
int vx_qlinear_i8u8_arm_i8mm_packed_try(const void* input,
        const VxPackedQ8Header* header, const int32_t* bias,
        const float* weight_scales, const int32_t* weight_zero_points,
        void* output, uint32_t rows, float input_scale,
        int32_t input_zero_point, float output_scale,
        int32_t output_zero_point, uint32_t input_dtype,
        uint32_t output_dtype);
#endif

/* Native physical-QLinear policy. The packed pair layout is available only for
 * symmetric I8 weights. AVX2 uses its exact K4/N16 kernels generally; a
 * single-threaded, wide-enough call uses N32 with either the proved
 * saturation-free plain dot or the signed-absolute spelling for weights that
 * exclude -128. VNNI N32 accepts the complete signed byte range through
 * VPDPBUSD. Short-K M=1 adapter
 * projections also benefit, while ordinary M=1 decode keeps the raw GEMV
 * dispatcher. Other ISAs retain their runtime-gated raw kernels. */
int vx_packed_q8_preferred_for_native_w8a8(uint32_t rows, uint32_t d_in,
        uint32_t d_out, uint32_t weight_dtype, int weight_zero_all_zero);

#ifdef __cplusplus
}
#endif


/* The K block the packed quantized kernels step, derived from the detected L1
 * rather than fixed, and the check that the derivation still reproduces the
 * value it replaced on a 32 KiB L1. */
uint32_t vx_qgemm_kc(void);
int vx_qgemm_kc_is_baseline(void);

#endif
