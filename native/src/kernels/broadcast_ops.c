// --- The Final N-Dimensional "No Stubs" Primitives (Batch 5) ---
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

enum { VX_MATMUL_MR = 4, VX_MATMUL_NR = 16 };

static void matmul_f32_k_major_scalar(const float* input, const float* weight,
                                      const float* bias, float* output,
                                      int seq_len, int d_in, int columns,
                                      int stride);

#if VX_MATMUL_WASM_SIMD

enum { VX_MATMUL_WASM_MR = 4, VX_MATMUL_WASM_NR = 8 };

#if defined(VOLVOXAI_MATMUL_WASM_SIMD_TESTING)
static uint32_t vx_matmul_wasm_simd_calls;

WASM_EXPORT("matmul_wasm_simd_calls")
uint32_t vx_matmul_wasm_simd_call_count(void) {
    return vx_matmul_wasm_simd_calls;
}

WASM_EXPORT("reset_matmul_wasm_simd_calls")
void vx_reset_matmul_wasm_simd_call_count(void) {
    vx_matmul_wasm_simd_calls = 0;
}
#endif

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
#if defined(VOLVOXAI_MATMUL_WASM_SIMD_TESTING)
    vx_matmul_wasm_simd_calls++;
#endif
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

static VX_MATMUL_TARGET_AVX2 void matmul_f32_out_in_avx2(
        const float* input, const float* weight, const float* bias,
        float* output, int rows, int d_in, int d_out) {
    for (int row = 0; row < rows; row++) {
        const float* input_row = input + (size_t)row * d_in;
        float* output_row = output + (size_t)row * d_out;
        for (int j = 0; j < d_out; j++)
            output_row[j] = (bias ? bias[j] : 0.0f) +
                matmul_f32_dot_avx2(input_row, weight + (size_t)j * d_in, d_in);
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

WASM_EXPORT("matmul_f32")
void matmul_f32(const float* input, const float* weight, const float* bias, float* output, int seq_len, int d_in, int d_out) {
#if VX_MATMUL_X86_AVX2
    if (vx_kernel_platform()->has_avx2) {
        matmul_f32_k_major_avx2(input, weight, bias, output, seq_len, d_in, d_out);
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
    if (vx_kernel_platform()->has_avx2) {
        matmul_f32_out_in_avx2(input, weight, bias, output, rows, d_in, d_out);
        return;
    }
#endif
    matmul_f32_out_in_scalar(input, weight, bias, output, rows, d_in, d_out);
}


WASM_EXPORT("add_broadcast_f32")
void add_broadcast_f32(const float* a, const float* b, float* out, int c, int h, int w) {
    int spatial = h * w;
    for (int ic = 0; ic < c; ic++) {
        float b_val = b[ic];
        int offset = ic * spatial;
        for (int i = 0; i < spatial; i++) {
            out[offset + i] = a[offset + i] + b_val;
        }
    }
}

WASM_EXPORT("mul_f32")
void mul_f32(const float* a, const float* b, float* out, int n) {
    for (int i = 0; i < n; i++) out[i] = a[i] * b[i];
}

WASM_EXPORT("mul_broadcast_f32")
void mul_broadcast_f32(const float* a, const float* b, float* out, int c, int h, int w) {
    int spatial = h * w;
    for (int ic = 0; ic < c; ic++) {
        float b_val = b[ic];
        int offset = ic * spatial;
        for (int i = 0; i < spatial; i++) {
            out[offset + i] = a[offset + i] * b_val;
        }
    }
}
