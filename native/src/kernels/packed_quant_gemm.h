#ifndef VOLVOXAI_PACKED_QUANT_GEMM_H
#define VOLVOXAI_PACKED_QUANT_GEMM_H

#include <stdint.h>

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

int vx_qlinear_i8u8_packed(const void* input, const void* packed_weight,
        const int32_t* bias, const float* weight_scales,
        const int32_t* weight_zero_points, void* output,
        uint32_t rows, uint32_t d_in, uint32_t d_out,
        float input_scale, int32_t input_zero_point,
        float output_scale, int32_t output_zero_point,
        uint32_t input_dtype, uint32_t weight_dtype, uint32_t output_dtype);

/* Native physical-QLinear policy. The raw ISA dispatcher wins the M=1 decode
 * shape; the packed AVX2 B-panel kernel wins once multiple rows share B.
 * Other ISAs retain their existing runtime-gated raw microkernels. */
int vx_packed_q8_preferred_for_native_w8a8(uint32_t rows);

#ifdef __cplusplus
}
#endif

#endif
