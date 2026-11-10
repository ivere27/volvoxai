/* Cast, N-D Slice/Gather, MatMul dispatch and broadcasting Add/Mul. */
#include "../../include/volvoxai_enums.h"
#include "kernel_platform.h"
#include <stddef.h>
#include <stdint.h>

#if defined(__wasm__) && defined(__wasm_simd128__) && \
    !defined(VOLVOXAI_DISABLE_MATMUL_WASM_SIMD)
#include <wasm_simd128.h>
#define VX_MATMUL_WASM_SIMD 1
#else
#define VX_MATMUL_WASM_SIMD 0
#endif

static int32_t vx_cast_integer_read(const void *input, int dtype, int index) {
    if (dtype == VX_DTYPE_I32) return ((const int32_t *)input)[index];
    if (dtype == VX_DTYPE_I8) return (int32_t)((const int8_t *)input)[index];
    return (int32_t)((const uint8_t *)input)[index];
}

static int vx_cast_dtype_supported(int dtype) {
    return dtype == VX_DTYPE_F32 || dtype == VX_DTYPE_I32 ||
        dtype == VX_DTYPE_I8 || dtype == VX_DTYPE_U8;
}

/* Match ECMAScript TypedArray ToInt32 exactly for an F32 source: non-finite
   values become zero; finite values truncate toward zero and wrap modulo
   2^32. Doing this from bits avoids undefined C float-to-int conversions. */
static uint32_t vx_cast_f32_to_u32_mod(float value) {
    union { float f; uint32_t u; } bits = { value };
    uint32_t exponent = (bits.u >> 23u) & 0xffu;
    if (exponent == 0xffu || exponent < 127u) return 0;
    uint32_t significand = (bits.u & 0x7fffffu) | 0x800000u;
    int32_t shift = (int32_t)exponent - 127 - 23;
    uint32_t magnitude;
    if (shift >= 32) magnitude = 0;
    else if (shift >= 0) magnitude = significand << (uint32_t)shift;
    else magnitude = significand >> (uint32_t)(-shift);
    return (bits.u & 0x80000000u) ? 0u - magnitude : magnitude;
}

static int32_t vx_cast_i32_from_bits(uint32_t bits) {
    if (bits <= 0x7fffffffu) return (int32_t)bits;
    return (int32_t)(bits - 0x80000000u) - 2147483647 - 1;
}

static int32_t vx_cast_f32_to_i32(float value) {
    return vx_cast_i32_from_bits(vx_cast_f32_to_u32_mod(value));
}

static int8_t vx_cast_i8_from_u8(uint8_t value) {
    if (value <= 127u) return (int8_t)value;
    return (int8_t)(value - 128u) - 128;
}

static void vx_cast_integer_store(void *output, int dtype, int index,
        int32_t integer) {
    if (dtype == VX_DTYPE_I32) ((int32_t *)output)[index] = integer;
    else if (dtype == VX_DTYPE_I8) {
        ((int8_t *)output)[index] = vx_cast_i8_from_u8((uint8_t)integer);
    }
    else ((uint8_t *)output)[index] = (uint8_t)integer;
}

/* Dtypes use canonical VxDataType protobuf values. Integer writes retain
 * TypedArray wrapping. */
int cast_typed(const void *input, int input_dtype, void *output, int output_dtype, int n) {
    if (!input || !output || n < 0 ||
        !vx_cast_dtype_supported(input_dtype) ||
        !vx_cast_dtype_supported(output_dtype)) return 0;
    for (int index = 0; index < n; index++) {
        if (input_dtype == VX_DTYPE_F32) {
            float value = ((const float *)input)[index];
            if (output_dtype == VX_DTYPE_F32)
                ((float *)output)[index] = value;
            else vx_cast_integer_store(output, output_dtype, index, vx_cast_f32_to_i32(value));
        } else {
            int32_t value = vx_cast_integer_read(input, input_dtype, index);
            if (output_dtype == VX_DTYPE_F32)
                ((float *)output)[index] = (float)value;
            else vx_cast_integer_store(output, output_dtype, index, value);
        }
    }
    return 1;
}

void cast_i32_to_f32(const int* input, float* output, int n) {
    for (int i = 0; i < n; i++) output[i] = (float)input[i];
}
void cast_f32_to_i32(const float* input, int* output, int n) {
    for (int i = 0; i < n; i++) output[i] = (int)input[i];
}

void slice_4d_f32(const float* input, float* output, 
                  int in_n, int in_c, int in_h, int in_w,
                  int out_n, int out_c, int out_h, int out_w,
                  int start_n, int start_c, int start_h, int start_w,
                  int step_n, int step_c, int step_h, int step_w) {
    for (int on = 0; on < out_n; on++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int in_n_idx = start_n + on * step_n;
                    int in_c_idx = start_c + oc * step_c;
                    int in_h_idx = start_h + oh * step_h;
                    int in_w_idx = start_w + ow * step_w;
                    output[on * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow] = 
                        input[in_n_idx * (in_c * in_h * in_w) + in_c_idx * (in_h * in_w) + in_h_idx * in_w + in_w_idx];
                }
            }
        }
    }
}

void gather_4d_f32(const float* input, const float* indices, float* output,
                   int in_n, int in_c, int in_h, int in_w,
                   int num_indices, int axis) {
    // A simplified gather that replaces 'axis' with a 1D list of indices.
    // E.g., if axis == 1 (channel), output will have shape [in_n, num_indices, in_h, in_w]
    int out_n = axis == 0 ? num_indices : in_n;
    int out_c = axis == 1 ? num_indices : in_c;
    int out_h = axis == 2 ? num_indices : in_h;
    int out_w = axis == 3 ? num_indices : in_w;
    
    for (int on = 0; on < out_n; on++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int in_n_idx = axis == 0 ? (int)indices[on] : on;
                    int in_c_idx = axis == 1 ? (int)indices[oc] : oc;
                    int in_h_idx = axis == 2 ? (int)indices[oh] : oh;
                    int in_w_idx = axis == 3 ? (int)indices[ow] : ow;
                    
                    if (in_n_idx < 0) in_n_idx += in_n;
                    if (in_c_idx < 0) in_c_idx += in_c;
                    if (in_h_idx < 0) in_h_idx += in_h;
                    if (in_w_idx < 0) in_w_idx += in_w;
                    
                    output[on * (out_c * out_h * out_w) + oc * (out_h * out_w) + oh * out_w + ow] = 
                        input[in_n_idx * (in_c * in_h * in_w) + in_c_idx * (in_h * in_w) + in_h_idx * in_w + in_w_idx];
                }
            }
        }
    }
}

/*
 * Dense F32 matrix multiply for the paths that cannot use a packed weight.
 *
 * The Linear/Gemm node prefers vx_gemm_f32_run_packed, but that needs an
 * immutable model weight it can pack once and cache.  BatchMatMul's B operand is
 * a graph value, and Linear falls back here when its weight is not a model
 * tensor, so these two kernels stay on the hot path.  Both were scalar triple
 * loops, and neither auto-vectorizes even under -mavx2: the reduction is an FP
 * accumulation the compiler may not reorder without -ffast-math, which this
 * build deliberately does not set.
 *
 * No packing happens here — it would cost O(K*N) writes against a single
 * O(M*K*N) call, which does not amortize for the small M these paths see.
 * Instead each kernel is written against the layout it is given, so every load
 * is contiguous in the dimension that matters:
 *
 *   matmul_f32        B is [K,N]: contiguous along N, so broadcast one A value
 *                     and FMA whole B rows into column accumulators.
 *   matmul_f32_out_in B is [N,K]: contiguous along K, so it is a dot product,
 *                     reduced with several independent accumulators.
 *
 * Both use four-row by sixteen-column blocking to hold eight ymm accumulators.
 * Four was measured too few: an FMA has ~4 cycles of latency against two per
 * cycle of throughput, so fewer than eight independent chains cannot keep the
 * unit busy regardless of how the loads are arranged.
 */
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_MATMUL_X86_AVX2 1
#define VX_MATMUL_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define VX_MATMUL_X86_AVX2 0
#define VX_MATMUL_TARGET_AVX2
#endif

#if !defined(__wasm__) && \
    (defined(__aarch64__) || defined(__arm64__)) && \
    !defined(VOLVOXAI_DISABLE_MATMUL_ARM_NEON)
#include <arm_neon.h>
#define VX_MATMUL_ARM_NEON 1
#else
#define VX_MATMUL_ARM_NEON 0
#endif

enum { VX_MATMUL_MR = 4, VX_MATMUL_NR = 16 };

static void matmul_f32_k_major_scalar(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      int seq_len, int d_in, int columns,
                                      int stride);

#if VX_MATMUL_WASM_SIMD

enum { VX_MATMUL_WASM_MR = 4, VX_MATMUL_WASM_NR = 8 };


/* B is [K,N], so adjacent output columns are independent reduction lanes.
 * Four rows reuse each pair of contiguous B vectors while eight accumulators
 * keep the separate multiply/add chains in registers.  Every lane visits K in
 * the same order as matmul_f32_k_major_scalar; baseline SIMD128 has no FMA, so
 * the graph-visible F32 result remains bit-identical to the scalar definition.
 * Ragged columns retain the generic scalar tail. */
static void matmul_f32_k_major_wasm(
        const float* input, const float* weight, const float* bias,
        float* output, int seq_len, int d_in, int d_out) {
    int row = 0;
    for (; row < seq_len; row += VX_MATMUL_WASM_MR) {
        const int rows = seq_len - row < VX_MATMUL_WASM_MR
            ? seq_len - row : VX_MATMUL_WASM_MR;
        int column = 0;
        for (; column + VX_MATMUL_WASM_NR <= d_out;
             column += VX_MATMUL_WASM_NR) {
            v128_t low[VX_MATMUL_WASM_MR];
            v128_t high[VX_MATMUL_WASM_MR];
            for (int r = 0; r < rows; r++) {
                low[r] = bias ? wasm_v128_load(bias + column)
                              : wasm_f32x4_splat(0.0f);
                high[r] = bias ? wasm_v128_load(bias + column + 4)
                               : wasm_f32x4_splat(0.0f);
            }
            for (int k = 0; k < d_in; k++) {
                const float* weight_row =
                    weight + (size_t)k * d_out + column;
                const v128_t weight_low = wasm_v128_load(weight_row);
                const v128_t weight_high = wasm_v128_load(weight_row + 4);
                for (int r = 0; r < rows; r++) {
                    const v128_t value = wasm_f32x4_splat(
                        input[(size_t)(row + r) * d_in + k]);
                    low[r] = wasm_f32x4_add(
                        low[r], wasm_f32x4_mul(value, weight_low));
                    high[r] = wasm_f32x4_add(
                        high[r], wasm_f32x4_mul(value, weight_high));
                }
            }
            for (int r = 0; r < rows; r++) {
                float* output_row =
                    output + (size_t)(row + r) * d_out + column;
                wasm_v128_store(output_row, low[r]);
                wasm_v128_store(output_row + 4, high[r]);
            }
        }
        if (column < d_out)
            matmul_f32_k_major_scalar(
                input + (size_t)row * d_in, weight + column,
                bias ? bias + column : NULL,
                output + (size_t)row * d_out + column,
                rows, d_in, d_out - column, d_out);
    }
}

#endif /* VX_MATMUL_WASM_SIMD */

/* `columns` is how many outputs to compute; `stride` is the row pitch of both
 * weight and output.  They differ whenever this runs as the column tail of the
 * vector kernel, which computes a slice of each row but must still step whole
 * rows.  Collapsing the two into one parameter silently read and wrote the
 * wrong rows for every d_out above VX_MATMUL_NR that is not a multiple of it. */
static void matmul_f32_k_major_scalar(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      int seq_len, int d_in, int columns,
                                      int stride) {
    for (int i = 0; i < seq_len; i++) {
        for (int j = 0; j < columns; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < d_in; k++) {
                sum += input[(size_t)i * d_in + k] * weight[(size_t)k * stride + j];
            }
            output[(size_t)i * stride + j] = sum;
        }
    }
}

static void matmul_f32_out_in_scalar(const float* input, const float* weight,
                                     const float* bias, float* output,
                                     int rows, int d_in, int d_out) {
    for (int row = 0; row < rows; row++) {
        for (int j = 0; j < d_out; j++) {
            float sum = bias ? bias[j] : 0.0f;
            for (int k = 0; k < d_in; k++)
                sum += input[(size_t)row * d_in + k] * weight[(size_t)j * d_in + k];
            output[(size_t)row * d_out + j] = sum;
        }
    }
}

#if VX_MATMUL_ARM_NEON

enum {
    /* Eighteen accumulators plus three weight vectors and staged scalar lanes
     * fit AArch64's caller-saved vector registers without a K-loop spill. Six
     * rows reuse each B load, while twelve columns divide the encoder's
     * 218-wide matrices with only a two-column scalar tail. */
    VX_MATMUL_ARM_MR = 6,
    VX_MATMUL_ARM_NR = 12,
    VX_MATMUL_ARM_VECTOR_NR = 4,
};

/* One-row path for decode and for the ragged rows below MR.  B remains in its
 * native [K,N] layout: every K step loads three adjacent vectors and updates
 * twelve independent output columns.  Each lane visits K in scalar order. */
static void matmul_f32_k_major_neon_m1(
        const float* input, const float* weight, const float* bias,
        float* output, int d_in, int d_out) {
    int column = 0;
    for (; column + VX_MATMUL_ARM_NR <= d_out;
         column += VX_MATMUL_ARM_NR) {
        float32x4_t acc0 = bias
            ? vld1q_f32(bias + column) : vdupq_n_f32(0.0f);
        float32x4_t acc1 = bias
            ? vld1q_f32(bias + column + 4) : vdupq_n_f32(0.0f);
        float32x4_t acc2 = bias
            ? vld1q_f32(bias + column + 8) : vdupq_n_f32(0.0f);
        for (int k = 0; k < d_in; k++) {
            const float* weight_row =
                weight + (size_t)k * d_out + column;
            const float value = input[k];
            acc0 = vfmaq_n_f32(acc0, vld1q_f32(weight_row), value);
            acc1 = vfmaq_n_f32(acc1, vld1q_f32(weight_row + 4), value);
            acc2 = vfmaq_n_f32(acc2, vld1q_f32(weight_row + 8), value);
        }
        vst1q_f32(output + column, acc0);
        vst1q_f32(output + column + 4, acc1);
        vst1q_f32(output + column + 8, acc2);
    }
    for (; column + VX_MATMUL_ARM_VECTOR_NR <= d_out;
         column += VX_MATMUL_ARM_VECTOR_NR) {
        float32x4_t acc = bias
            ? vld1q_f32(bias + column) : vdupq_n_f32(0.0f);
        for (int k = 0; k < d_in; k++) {
            const float* weight_row =
                weight + (size_t)k * d_out + column;
            acc = vfmaq_n_f32(acc, vld1q_f32(weight_row), input[k]);
        }
        vst1q_f32(output + column, acc);
    }
    if (column < d_out)
        matmul_f32_k_major_scalar(
            input, weight + column, bias ? bias + column : NULL,
            output + column, 1, d_in, d_out - column, d_out);
}

/* Direct no-pack [M,K] x [K,N] microkernel.  Named accumulators make register
 * residency explicit to the compiler; indexing a variable-sized accumulator
 * array here causes Clang to materialize it on the stack on AArch64. */
static void matmul_f32_k_major_neon(
        const float* input, const float* weight, const float* bias,
        float* output, int seq_len, int d_in, int d_out) {
    int row = 0;
    for (; row + VX_MATMUL_ARM_MR <= seq_len;
         row += VX_MATMUL_ARM_MR) {
        const float* a0 = input + (size_t)(row + 0) * d_in;
        const float* a1 = input + (size_t)(row + 1) * d_in;
        const float* a2 = input + (size_t)(row + 2) * d_in;
        const float* a3 = input + (size_t)(row + 3) * d_in;
        const float* a4 = input + (size_t)(row + 4) * d_in;
        const float* a5 = input + (size_t)(row + 5) * d_in;
        int column = 0;
        for (; column + VX_MATMUL_ARM_NR <= d_out;
             column += VX_MATMUL_ARM_NR) {
            const float32x4_t initial0 = bias
                ? vld1q_f32(bias + column) : vdupq_n_f32(0.0f);
            const float32x4_t initial1 = bias
                ? vld1q_f32(bias + column + 4) : vdupq_n_f32(0.0f);
            const float32x4_t initial2 = bias
                ? vld1q_f32(bias + column + 8) : vdupq_n_f32(0.0f);
            float32x4_t c00 = initial0, c01 = initial1, c02 = initial2;
            float32x4_t c10 = initial0, c11 = initial1, c12 = initial2;
            float32x4_t c20 = initial0, c21 = initial1, c22 = initial2;
            float32x4_t c30 = initial0, c31 = initial1, c32 = initial2;
            float32x4_t c40 = initial0, c41 = initial1, c42 = initial2;
            float32x4_t c50 = initial0, c51 = initial1, c52 = initial2;
            for (int k = 0; k < d_in; k++) {
                const float* weight_row =
                    weight + (size_t)k * d_out + column;
                const float32x4_t w0 = vld1q_f32(weight_row);
                const float32x4_t w1 = vld1q_f32(weight_row + 4);
                const float32x4_t w2 = vld1q_f32(weight_row + 8);
                c00 = vfmaq_n_f32(c00, w0, a0[k]);
                c01 = vfmaq_n_f32(c01, w1, a0[k]);
                c02 = vfmaq_n_f32(c02, w2, a0[k]);
                c10 = vfmaq_n_f32(c10, w0, a1[k]);
                c11 = vfmaq_n_f32(c11, w1, a1[k]);
                c12 = vfmaq_n_f32(c12, w2, a1[k]);
                c20 = vfmaq_n_f32(c20, w0, a2[k]);
                c21 = vfmaq_n_f32(c21, w1, a2[k]);
                c22 = vfmaq_n_f32(c22, w2, a2[k]);
                c30 = vfmaq_n_f32(c30, w0, a3[k]);
                c31 = vfmaq_n_f32(c31, w1, a3[k]);
                c32 = vfmaq_n_f32(c32, w2, a3[k]);
                c40 = vfmaq_n_f32(c40, w0, a4[k]);
                c41 = vfmaq_n_f32(c41, w1, a4[k]);
                c42 = vfmaq_n_f32(c42, w2, a4[k]);
                c50 = vfmaq_n_f32(c50, w0, a5[k]);
                c51 = vfmaq_n_f32(c51, w1, a5[k]);
                c52 = vfmaq_n_f32(c52, w2, a5[k]);
            }
            float* c0 = output + (size_t)(row + 0) * d_out + column;
            float* c1 = output + (size_t)(row + 1) * d_out + column;
            float* c2 = output + (size_t)(row + 2) * d_out + column;
            float* c3 = output + (size_t)(row + 3) * d_out + column;
            float* c4 = output + (size_t)(row + 4) * d_out + column;
            float* c5 = output + (size_t)(row + 5) * d_out + column;
            vst1q_f32(c0, c00); vst1q_f32(c0 + 4, c01); vst1q_f32(c0 + 8, c02);
            vst1q_f32(c1, c10); vst1q_f32(c1 + 4, c11); vst1q_f32(c1 + 8, c12);
            vst1q_f32(c2, c20); vst1q_f32(c2 + 4, c21); vst1q_f32(c2 + 8, c22);
            vst1q_f32(c3, c30); vst1q_f32(c3 + 4, c31); vst1q_f32(c3 + 8, c32);
            vst1q_f32(c4, c40); vst1q_f32(c4 + 4, c41); vst1q_f32(c4 + 8, c42);
            vst1q_f32(c5, c50); vst1q_f32(c5 + 4, c51); vst1q_f32(c5 + 8, c52);
        }
        for (; column + VX_MATMUL_ARM_VECTOR_NR <= d_out;
             column += VX_MATMUL_ARM_VECTOR_NR) {
            const float32x4_t initial = bias
                ? vld1q_f32(bias + column) : vdupq_n_f32(0.0f);
            float32x4_t c0 = initial, c1 = initial, c2 = initial;
            float32x4_t c3 = initial, c4 = initial, c5 = initial;
            for (int k = 0; k < d_in; k++) {
                const float32x4_t w = vld1q_f32(
                    weight + (size_t)k * d_out + column);
                c0 = vfmaq_n_f32(c0, w, a0[k]);
                c1 = vfmaq_n_f32(c1, w, a1[k]);
                c2 = vfmaq_n_f32(c2, w, a2[k]);
                c3 = vfmaq_n_f32(c3, w, a3[k]);
                c4 = vfmaq_n_f32(c4, w, a4[k]);
                c5 = vfmaq_n_f32(c5, w, a5[k]);
            }
            vst1q_f32(output + (size_t)(row + 0) * d_out + column, c0);
            vst1q_f32(output + (size_t)(row + 1) * d_out + column, c1);
            vst1q_f32(output + (size_t)(row + 2) * d_out + column, c2);
            vst1q_f32(output + (size_t)(row + 3) * d_out + column, c3);
            vst1q_f32(output + (size_t)(row + 4) * d_out + column, c4);
            vst1q_f32(output + (size_t)(row + 5) * d_out + column, c5);
        }
        if (column < d_out)
            matmul_f32_k_major_scalar(
                input + (size_t)row * d_in, weight + column,
                bias ? bias + column : NULL,
                output + (size_t)row * d_out + column,
                VX_MATMUL_ARM_MR, d_in, d_out - column, d_out);
    }
    for (; row < seq_len; row++)
        matmul_f32_k_major_neon_m1(
            input + (size_t)row * d_in, weight, bias,
            output + (size_t)row * d_out, d_in, d_out);
}

/* OUT_IN weights are contiguous along K.  Four independent accumulators hide
 * FMA latency without packing or changing the caller-visible layout. */
static inline float matmul_f32_dot_neon(
        const float* input, const float* weight, int count) {
    float32x4_t acc0 = vdupq_n_f32(0.0f);
    float32x4_t acc1 = vdupq_n_f32(0.0f);
    float32x4_t acc2 = vdupq_n_f32(0.0f);
    float32x4_t acc3 = vdupq_n_f32(0.0f);
    int k = 0;
    for (; k + 16 <= count; k += 16) {
        acc0 = vfmaq_f32(acc0, vld1q_f32(input + k),
                         vld1q_f32(weight + k));
        acc1 = vfmaq_f32(acc1, vld1q_f32(input + k + 4),
                         vld1q_f32(weight + k + 4));
        acc2 = vfmaq_f32(acc2, vld1q_f32(input + k + 8),
                         vld1q_f32(weight + k + 8));
        acc3 = vfmaq_f32(acc3, vld1q_f32(input + k + 12),
                         vld1q_f32(weight + k + 12));
    }
    for (; k + 4 <= count; k += 4)
        acc0 = vfmaq_f32(acc0, vld1q_f32(input + k),
                         vld1q_f32(weight + k));
    acc0 = vaddq_f32(acc0, acc2);
    acc1 = vaddq_f32(acc1, acc3);
    float sum = vaddvq_f32(vaddq_f32(acc0, acc1));
    for (; k < count; k++) sum += input[k] * weight[k];
    return sum;
}

static void matmul_f32_out_in_neon(
        const float* input, const float* weight, const float* bias,
        float* output, int rows, int d_in, int d_out) {
    for (int row = 0; row < rows; row++) {
        const float* input_row = input + (size_t)row * d_in;
        float* output_row = output + (size_t)row * d_out;
        for (int column = 0; column < d_out; column++)
            output_row[column] = (bias ? bias[column] : 0.0f) +
                matmul_f32_dot_neon(
                    input_row, weight + (size_t)column * d_in, d_in);
    }
}

#endif /* VX_MATMUL_ARM_NEON */

#if VX_MATMUL_X86_AVX2

/* B is [K,N].  One A element broadcasts across sixteen B columns, so the inner
 * step is two contiguous B loads and eight FMAs against register-resident
 * accumulators; B is walked with a d_out stride the prefetcher handles as two
 * sequential streams. */
static VX_MATMUL_TARGET_AVX2 void matmul_f32_k_major_avx2(
        const float* input, const float* weight, const float* bias,
        float* output, int seq_len, int d_in, int d_out) {
    int row = 0;
    for (; row < seq_len; row += VX_MATMUL_MR) {
        const int rows = seq_len - row < VX_MATMUL_MR ? seq_len - row : VX_MATMUL_MR;
        int column = 0;
        for (; column + VX_MATMUL_NR <= d_out; column += VX_MATMUL_NR) {
            __m256 low[VX_MATMUL_MR], high[VX_MATMUL_MR];
            int r;
            for (r = 0; r < rows; r++) {
                low[r] = bias ? _mm256_loadu_ps(bias + column) : _mm256_setzero_ps();
                high[r] = bias ? _mm256_loadu_ps(bias + column + 8)
                               : _mm256_setzero_ps();
            }
            for (int k = 0; k < d_in; k++) {
                const float* weight_row = weight + (size_t)k * d_out + column;
                const __m256 weight_low = _mm256_loadu_ps(weight_row);
                const __m256 weight_high = _mm256_loadu_ps(weight_row + 8);
                for (r = 0; r < rows; r++) {
                    const __m256 value =
                        _mm256_set1_ps(input[(size_t)(row + r) * d_in + k]);
                    low[r] = _mm256_fmadd_ps(value, weight_low, low[r]);
                    high[r] = _mm256_fmadd_ps(value, weight_high, high[r]);
                }
            }
            for (r = 0; r < rows; r++) {
                float* out_row = output + (size_t)(row + r) * d_out + column;
                _mm256_storeu_ps(out_row, low[r]);
                _mm256_storeu_ps(out_row + 8, high[r]);
            }
        }
        if (column < d_out)
            matmul_f32_k_major_scalar(input + (size_t)row * d_in,
                                      weight + column, bias ? bias + column : NULL,
                                      output + (size_t)row * d_out + column,
                                      rows, d_in, d_out - column, d_out);
    }
}

/* B is [N,K], so each output is a dot product over contiguous memory.  Four
 * accumulators per column pair keep the reduction chains independent; the A row
 * is reused across the four columns of a block. */
static inline VX_MATMUL_TARGET_AVX2 float matmul_f32_dot_avx2(
        const float* a, const float* b, int n) {
    __m256 acc0 = _mm256_setzero_ps();
    __m256 acc1 = _mm256_setzero_ps();
    __m128 folded;
    float sum;
    int k = 0;
    for (; k + 16 <= n; k += 16) {
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),
                               _mm256_loadu_ps(b + k), acc0);
        acc1 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k + 8),
                               _mm256_loadu_ps(b + k + 8), acc1);
    }
    for (; k + 8 <= n; k += 8)
        acc0 = _mm256_fmadd_ps(_mm256_loadu_ps(a + k),
                               _mm256_loadu_ps(b + k), acc0);
    acc0 = _mm256_add_ps(acc0, acc1);
    folded = _mm_add_ps(_mm256_castps256_ps128(acc0),
                        _mm256_extractf128_ps(acc0, 1));
    folded = _mm_add_ps(folded, _mm_movehl_ps(folded, folded));
    folded = _mm_add_ss(folded, _mm_shuffle_ps(folded, folded, 1));
    sum = _mm_cvtss_f32(folded);
    for (; k < n; k++) sum += a[k] * b[k];
    return sum;
}

typedef struct {
    const float* input;
    const float* weight;
    const float* bias;
    float* output;
    int d_in;
    int d_out;
} VxMatMulOutInTask;

static void vx_matmul_out_in_row_worker(void* context, int begin, int end) {
    const VxMatMulOutInTask* task = (const VxMatMulOutInTask*)context;
    for (int row = begin; row < end; row++) {
        const float* input_row = task->input + (size_t)row * task->d_in;
        float* output_row = task->output + (size_t)row * task->d_out;
        for (int j = 0; j < task->d_out; j++)
            output_row[j] = (task->bias ? task->bias[j] : 0.0f) +
                matmul_f32_dot_avx2(input_row, task->weight + (size_t)j * task->d_in, task->d_in);
    }
}

static void vx_matmul_out_in_col_worker(void* context, int begin, int end) {
    const VxMatMulOutInTask* task = (const VxMatMulOutInTask*)context;
    const float* input_row = task->input;
    float* output_row = task->output;
    for (int j = begin; j < end; j++)
        output_row[j] = (task->bias ? task->bias[j] : 0.0f) +
            matmul_f32_dot_avx2(input_row, task->weight + (size_t)j * task->d_in, task->d_in);
}

static VX_MATMUL_TARGET_AVX2 void matmul_f32_out_in_avx2(
        const float* input, const float* weight, const float* bias,
        float* output, int rows, int d_in, int d_out) {
    int threads = vx_kernels_thread_count();
    if (threads > 1 && rows >= 4 && (size_t)rows * d_out >= 4096) {
        int grain = 1 + (rows - 1) / (threads * 2);
        if (grain < 1) grain = 1;
        VxMatMulOutInTask task = {input, weight, bias, output, d_in, d_out};
        vx_kernels_parallel_for(rows, grain, vx_matmul_out_in_row_worker, &task);
    } else if (threads > 1 && rows == 1 && d_out >= 512 && (size_t)d_in * d_out >= 131072) {
        int grain = 1 + (d_out - 1) / (threads * 2);
        if (grain < 64) grain = 64;
        VxMatMulOutInTask task = {input, weight, bias, output, d_in, d_out};
        vx_kernels_parallel_for(d_out, grain, vx_matmul_out_in_col_worker, &task);
    } else {
        for (int row = 0; row < rows; row++) {
            const float* input_row = input + (size_t)row * d_in;
            float* output_row = output + (size_t)row * d_out;
            for (int j = 0; j < d_out; j++)
                output_row[j] = (bias ? bias[j] : 0.0f) +
                    matmul_f32_dot_avx2(input_row, weight + (size_t)j * d_in, d_in);
        }
    }
}

#endif /* VX_MATMUL_X86_AVX2 */

/* Resolve this translation unit's ISA cache on the dispatching thread before
 * callers fan independent MatMul jobs out to the shared pool. The cache in
 * kernel_platform.h is intentionally TU-local, so resolving a copy from the
 * runtime translation unit would not make the worker calls below race-free. */
void matmul_f32_prepare_dispatch(void) {
#if VX_MATMUL_X86_AVX2
    (void)vx_kernel_platform();
#endif
}

void matmul_f32(const float* input, const float* weight, const float* bias, float* output, int seq_len, int d_in, int d_out) {
#if VX_MATMUL_X86_AVX2
    if (vx_kernel_platform()->has_avx2 && __builtin_cpu_supports("fma")) {
        matmul_f32_k_major_avx2(input, weight, bias, output, seq_len, d_in, d_out);
        return;
    }
#endif
#if VX_MATMUL_ARM_NEON
    if (vx_kernel_platform()->has_neon &&
        d_out >= VX_MATMUL_ARM_VECTOR_NR) {
        matmul_f32_k_major_neon(
            input, weight, bias, output, seq_len, d_in, d_out);
        return;
    }
#endif
#if VX_MATMUL_WASM_SIMD
    if (d_out >= VX_MATMUL_WASM_NR) {
        matmul_f32_k_major_wasm(input, weight, bias, output,
                               seq_len, d_in, d_out);
        return;
    }
#endif
    matmul_f32_k_major_scalar(input, weight, bias, output, seq_len, d_in, d_out,
                              d_out);
}

/* Shares matmul_f32's dispatch so the Linear node's OUT_IN fallback is not the
 * one shape left running scalar. */
void matmul_f32_out_in(const float* input, const float* weight, const float* bias,
                       float* output, int rows, int d_in, int d_out) {
#if VX_MATMUL_X86_AVX2
    if (vx_kernel_platform()->has_avx2 && __builtin_cpu_supports("fma")) {
        matmul_f32_out_in_avx2(input, weight, bias, output, rows, d_in, d_out);
        return;
    }
#endif
#if VX_MATMUL_ARM_NEON
    if (vx_kernel_platform()->has_neon &&
        d_in >= VX_MATMUL_ARM_VECTOR_NR) {
        matmul_f32_out_in_neon(
            input, weight, bias, output, rows, d_in, d_out);
        return;
    }
#endif
    matmul_f32_out_in_scalar(input, weight, bias, output, rows, d_in, d_out);
}


#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_BROADCAST_TARGET_AVX2 __attribute__((target("avx2")))

static VX_BROADCAST_TARGET_AVX2 int vx_add_broadcast_f32_avx2(
        const float* a, float b_val, float* out, int spatial) {
    int i = 0;
    const __m256 vb = _mm256_set1_ps(b_val);
    for (; i + 8 <= spatial; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_add_ps(_mm256_loadu_ps(a + i), vb));
    }
    return i;
}

static VX_BROADCAST_TARGET_AVX2 int vx_mul_f32_avx2(
        const float* a, const float* b, float* out, int n) {
    int i = 0;
    if (a == b) {
        for (; i + 32 <= n; i += 32) {
            __m256 a0 = _mm256_loadu_ps(a + i);
            __m256 a1 = _mm256_loadu_ps(a + i + 8);
            __m256 a2 = _mm256_loadu_ps(a + i + 16);
            __m256 a3 = _mm256_loadu_ps(a + i + 24);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(a0, a0));
            _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(a1, a1));
            _mm256_storeu_ps(out + i + 16, _mm256_mul_ps(a2, a2));
            _mm256_storeu_ps(out + i + 24, _mm256_mul_ps(a3, a3));
        }
        for (; i + 8 <= n; i += 8) {
            __m256 a0 = _mm256_loadu_ps(a + i);
            _mm256_storeu_ps(out + i, _mm256_mul_ps(a0, a0));
        }
        return i;
    }
    for (; i + 32 <= n; i += 32) {
        __m256 a0 = _mm256_loadu_ps(a + i);
        __m256 a1 = _mm256_loadu_ps(a + i + 8);
        __m256 a2 = _mm256_loadu_ps(a + i + 16);
        __m256 a3 = _mm256_loadu_ps(a + i + 24);
        __m256 b0 = _mm256_loadu_ps(b + i);
        __m256 b1 = _mm256_loadu_ps(b + i + 8);
        __m256 b2 = _mm256_loadu_ps(b + i + 16);
        __m256 b3 = _mm256_loadu_ps(b + i + 24);
        _mm256_storeu_ps(out + i, _mm256_mul_ps(a0, b0));
        _mm256_storeu_ps(out + i + 8, _mm256_mul_ps(a1, b1));
        _mm256_storeu_ps(out + i + 16, _mm256_mul_ps(a2, b2));
        _mm256_storeu_ps(out + i + 24, _mm256_mul_ps(a3, b3));
    }
    for (; i + 8 <= n; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), _mm256_loadu_ps(b + i)));
    }
    return i;
}

static VX_BROADCAST_TARGET_AVX2 int vx_mul_broadcast_f32_avx2(
        const float* a, float b_val, float* out, int spatial) {
    int i = 0;
    const __m256 vb = _mm256_set1_ps(b_val);
    for (; i + 8 <= spatial; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_mul_ps(_mm256_loadu_ps(a + i), vb));
    }
    return i;
}
#else
#define VX_BROADCAST_TARGET_AVX2
#endif

void add_broadcast_f32(const float* a, const float* b, float* out, int c, int h, int w) {
    int spatial = h * w;
    const int use_avx2 =
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        vx_kernel_platform()->has_avx2 && spatial >= 8;
#else
        0;
#endif
    for (int ic = 0; ic < c; ic++) {
        float b_val = b[ic];
        int offset = ic * spatial;
        int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        if (use_avx2) {
            i = vx_add_broadcast_f32_avx2(a + offset, b_val, out + offset, spatial);
        }
#endif
#if VX_MATMUL_WASM_SIMD
        v128_t vb = wasm_f32x4_splat(b_val);
        for (; i + 16 <= spatial; i += 16) {
            v128_t a0 = wasm_v128_load(a + offset + i + 0);
            v128_t a1 = wasm_v128_load(a + offset + i + 4);
            v128_t a2 = wasm_v128_load(a + offset + i + 8);
            v128_t a3 = wasm_v128_load(a + offset + i + 12);
            wasm_v128_store(out + offset + i + 0, wasm_f32x4_add(a0, vb));
            wasm_v128_store(out + offset + i + 4, wasm_f32x4_add(a1, vb));
            wasm_v128_store(out + offset + i + 8, wasm_f32x4_add(a2, vb));
            wasm_v128_store(out + offset + i + 12, wasm_f32x4_add(a3, vb));
        }
        for (; i + 4 <= spatial; i += 4) {
            wasm_v128_store(out + offset + i, wasm_f32x4_add(wasm_v128_load(a + offset + i), vb));
        }
#endif
        for (; i < spatial; i++) {
            out[offset + i] = a[offset + i] + b_val;
        }
    }
}

void mul_f32(const float* a, const float* b, float* out, int n) {
    int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
    if (vx_kernel_platform()->has_avx2 && n >= 8) {
        i = vx_mul_f32_avx2(a, b, out, n);
    }
#endif
#if VX_MATMUL_WASM_SIMD
    for (; i + 16 <= n; i += 16) {
        v128_t a0 = wasm_v128_load(a + i + 0);
        v128_t a1 = wasm_v128_load(a + i + 4);
        v128_t a2 = wasm_v128_load(a + i + 8);
        v128_t a3 = wasm_v128_load(a + i + 12);
        v128_t b0 = wasm_v128_load(b + i + 0);
        v128_t b1 = wasm_v128_load(b + i + 4);
        v128_t b2 = wasm_v128_load(b + i + 8);
        v128_t b3 = wasm_v128_load(b + i + 12);
        wasm_v128_store(out + i + 0, wasm_f32x4_mul(a0, b0));
        wasm_v128_store(out + i + 4, wasm_f32x4_mul(a1, b1));
        wasm_v128_store(out + i + 8, wasm_f32x4_mul(a2, b2));
        wasm_v128_store(out + i + 12, wasm_f32x4_mul(a3, b3));
    }
    for (; i + 4 <= n; i += 4) {
        wasm_v128_store(out + i, wasm_f32x4_mul(wasm_v128_load(a + i), wasm_v128_load(b + i)));
    }
#endif
    for (; i < n; i++) out[i] = a[i] * b[i];
}

void mul_broadcast_f32(const float* a, const float* b, float* out, int c, int h, int w) {
    int spatial = h * w;
    const int use_avx2 =
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        vx_kernel_platform()->has_avx2 && spatial >= 8;
#else
        0;
#endif
    for (int ic = 0; ic < c; ic++) {
        float b_val = b[ic];
        int offset = ic * spatial;
        int i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        if (use_avx2) {
            i = vx_mul_broadcast_f32_avx2(a + offset, b_val, out + offset, spatial);
        }
#endif
#if VX_MATMUL_WASM_SIMD
        v128_t vb = wasm_f32x4_splat(b_val);
        for (; i + 16 <= spatial; i += 16) {
            v128_t a0 = wasm_v128_load(a + offset + i + 0);
            v128_t a1 = wasm_v128_load(a + offset + i + 4);
            v128_t a2 = wasm_v128_load(a + offset + i + 8);
            v128_t a3 = wasm_v128_load(a + offset + i + 12);
            wasm_v128_store(out + offset + i + 0, wasm_f32x4_mul(a0, vb));
            wasm_v128_store(out + offset + i + 4, wasm_f32x4_mul(a1, vb));
            wasm_v128_store(out + offset + i + 8, wasm_f32x4_mul(a2, vb));
            wasm_v128_store(out + offset + i + 12, wasm_f32x4_mul(a3, vb));
        }
        for (; i + 4 <= spatial; i += 4) {
            wasm_v128_store(out + offset + i, wasm_f32x4_mul(wasm_v128_load(a + offset + i), vb));
        }
#endif
        for (; i < spatial; i++) {
            out[offset + i] = a[offset + i] * b_val;
        }
    }
}

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
static VX_BROADCAST_TARGET_AVX2 int vx_sub_broadcast_f32_avx2(
        const float* a, float b_val, float* out, int spatial) {
    int i = 0;
    const __m256 vb = _mm256_set1_ps(b_val);
    for (; i + 8 <= spatial; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_sub_ps(_mm256_loadu_ps(a + i), vb));
    }
    return i;
}

static VX_BROADCAST_TARGET_AVX2 int vx_rsub_broadcast_f32_avx2(
        float a_val, const float* b, float* out, int spatial) {
    int i = 0;
    const __m256 va = _mm256_set1_ps(a_val);
    for (; i + 8 <= spatial; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_sub_ps(va, _mm256_loadu_ps(b + i)));
    }
    return i;
}

static VX_BROADCAST_TARGET_AVX2 int vx_div_broadcast_f32_avx2(
        const float* a, float b_val, float* out, int spatial) {
    int i = 0;
    const __m256 vb = _mm256_set1_ps(b_val);
    for (; i + 8 <= spatial; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(a + i), vb));
    }
    return i;
}

static VX_BROADCAST_TARGET_AVX2 int vx_rdiv_broadcast_f32_avx2(
        float a_val, const float* b, float* out, int spatial) {
    int i = 0;
    const __m256 va = _mm256_set1_ps(a_val);
    for (; i + 8 <= spatial; i += 8) {
        _mm256_storeu_ps(out + i, _mm256_div_ps(va, _mm256_loadu_ps(b + i)));
    }
    return i;
}
#endif

void channel_broadcast_op_f32(const float* cont, const float* bc, float* out,
                              int blk_begin, int blk_end, long spatial,
                              int op, int is_lhs, int is_single, int relu) {
    const int use_avx2 =
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        vx_kernel_platform()->has_avx2 && spatial >= 8;
#else
        0;
#endif

    for (int blk = blk_begin; blk < blk_end; blk++) {
        float b_val = is_single ? bc[0] : bc[blk];
        long offset = (long)blk * spatial;
        const float* src = cont + offset;
        float* dst = out + offset;
        long i = 0;
#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && (defined(__clang__) || defined(__GNUC__))
        if (use_avx2) {
            if (op == 0) {
                i = vx_add_broadcast_f32_avx2(src, b_val, dst, (int)spatial);
            } else if (op == 1) {
                i = vx_mul_broadcast_f32_avx2(src, b_val, dst, (int)spatial);
            } else if (op == 2) {
                if (!is_lhs) i = vx_sub_broadcast_f32_avx2(src, b_val, dst, (int)spatial);
                else i = vx_rsub_broadcast_f32_avx2(b_val, src, dst, (int)spatial);
            } else if (op == 3) {
                if (!is_lhs) i = vx_div_broadcast_f32_avx2(src, b_val, dst, (int)spatial);
                else i = vx_rdiv_broadcast_f32_avx2(b_val, src, dst, (int)spatial);
            }
        }
#endif
        for (; i < spatial; i++) {
            float val = src[i];
            float res;
            if (op == 0) res = val + b_val;
            else if (op == 1) res = val * b_val;
            else if (op == 2) res = !is_lhs ? (val - b_val) : (b_val - val);
            else res = !is_lhs ? (val / b_val) : (b_val / val);
            dst[i] = res;
        }
        if (relu) {
            for (long j = 0; j < spatial; j++) {
                if (dst[j] < 0.0f) dst[j] = 0.0f;
                else if (relu >= 2 && dst[j] > 6.0f) dst[j] = 6.0f;
            }
        }
    }
}

void row_broadcast_op_f32(const float* cont, const float* bc, float* out,
                          int row_begin, int row_end, long chunk_size,
                          int op, int is_lhs, int relu) {
    for (int r = row_begin; r < row_end; r++) {
        long offset = (long)r * chunk_size;
        const float* src = cont + offset;
        float* dst = out + offset;
        const float* a_ptr = is_lhs ? bc : src;
        const float* b_ptr = is_lhs ? src : bc;
        if (op == 0) {
            add_f32(a_ptr, b_ptr, dst, (int)chunk_size);
        } else if (op == 1) {
            mul_f32(a_ptr, b_ptr, dst, (int)chunk_size);
        } else if (op == 2) {
            sub_f32(a_ptr, b_ptr, dst, (int)chunk_size);
        } else {
            div_f32(a_ptr, b_ptr, dst, (int)chunk_size);
        }
        if (relu) {
            for (long j = 0; j < chunk_size; j++) {
                if (dst[j] < 0.0f) dst[j] = 0.0f;
                else if (relu >= 2 && dst[j] > 6.0f) dst[j] = 6.0f;
            }
        }
    }
}
