#include "tensor_f32_isa.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
#if defined(__aarch64__)
#include <arm_neon.h>
#define VX_TENSOR_F32_ARM_NEON 1
#else
#define VX_TENSOR_F32_ARM_NEON 0
#endif
#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define VX_TENSOR_F32_WASM_SIMD 1
#else
#define VX_TENSOR_F32_WASM_SIMD 0
#endif
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_TENSOR_F32_X86_AVX2 1
#define VX_TENSOR_F32_TARGET_AVX2 __attribute__((target("avx2")))
#else
#define VX_TENSOR_F32_X86_AVX2 0
#define VX_TENSOR_F32_TARGET_AVX2
#endif

extern int transpose_nd_f32(const float* input, float* output,
                            const uint32_t* input_shape,
                            const uint32_t* permutation, uint32_t rank,
                            uint32_t elements);

enum { VX_TRANSPOSE_F32_TILE = 8 };
enum { VX_TRANSPOSE_F32_PARALLEL_ELEMENTS = 131072 };

typedef struct {
    const float* source;
    float* destination;
    long batch;
    long rows;
    long columns;
    long row_tiles;
    int use_avx2;
    int use_neon;
    int use_wasm;
} VxTransposeF32Context;

#if VX_TENSOR_F32_ARM_NEON
/* Store one transposed 4x4 block.  Loads and stores stay bit preserving: no
 * floating-point arithmetic is performed, so NaN payloads and signed zero are
 * copied exactly like the portable Transpose implementation. */
static inline void vx_transpose_f32_4x4_neon(
        const float* source, long source_stride,
        float* destination, long destination_stride) {
    const float32x4_t r0 = vld1q_f32(source + 0 * source_stride);
    const float32x4_t r1 = vld1q_f32(source + 1 * source_stride);
    const float32x4_t r2 = vld1q_f32(source + 2 * source_stride);
    const float32x4_t r3 = vld1q_f32(source + 3 * source_stride);
    const float32x4x2_t t01 = vtrnq_f32(r0, r1);
    const float32x4x2_t t23 = vtrnq_f32(r2, r3);
    const float32x4_t c0 = vcombine_f32(
        vget_low_f32(t01.val[0]), vget_low_f32(t23.val[0]));
    const float32x4_t c1 = vcombine_f32(
        vget_low_f32(t01.val[1]), vget_low_f32(t23.val[1]));
    const float32x4_t c2 = vcombine_f32(
        vget_high_f32(t01.val[0]), vget_high_f32(t23.val[0]));
    const float32x4_t c3 = vcombine_f32(
        vget_high_f32(t01.val[1]), vget_high_f32(t23.val[1]));
    vst1q_f32(destination + 0 * destination_stride, c0);
    vst1q_f32(destination + 1 * destination_stride, c1);
    vst1q_f32(destination + 2 * destination_stride, c2);
    vst1q_f32(destination + 3 * destination_stride, c3);
}

static void vx_transpose_f32_tile4_neon(
        const VxTransposeF32Context* context, const float* source,
        float* destination, long row_begin) {
    long column = 0;
    for (; column + 4 <= context->columns; column += 4) {
        vx_transpose_f32_4x4_neon(
            source + row_begin * context->columns + column,
            context->columns,
            destination + column * context->rows + row_begin,
            context->rows);
    }
    for (; column < context->columns; column++) {
        for (long local = 0; local < 4; local++) {
            destination[column * context->rows + row_begin + local] =
                source[(row_begin + local) * context->columns + column];
        }
    }
}

/* Eight source rows are the scheduler's native tile.  Handle two 4x4 row
 * halves per column block while those cache lines are hot. */
static void vx_transpose_f32_tile8_neon(
        const VxTransposeF32Context* context, const float* source,
        float* destination, long row_begin) {
    long column = 0;
    for (; column + 4 <= context->columns; column += 4) {
        const float* source_block =
            source + row_begin * context->columns + column;
        float* destination_block =
            destination + column * context->rows + row_begin;
        vx_transpose_f32_4x4_neon(
            source_block, context->columns,
            destination_block, context->rows);
        vx_transpose_f32_4x4_neon(
            source_block + 4 * context->columns, context->columns,
            destination_block + 4, context->rows);
    }
    for (; column < context->columns; column++) {
        for (long local = 0; local < VX_TRANSPOSE_F32_TILE; local++) {
            destination[column * context->rows + row_begin + local] =
                source[(row_begin + local) * context->columns + column];
        }
    }
}
#endif

#if VX_TENSOR_F32_WASM_SIMD
static inline void vx_transpose_f32_4x4_wasm(
        const float* source, long source_stride,
        float* destination, long destination_stride) {
    const v128_t r0 = wasm_v128_load(source + 0 * source_stride);
    const v128_t r1 = wasm_v128_load(source + 1 * source_stride);
    const v128_t r2 = wasm_v128_load(source + 2 * source_stride);
    const v128_t r3 = wasm_v128_load(source + 3 * source_stride);

    const v128_t t01_lo = wasm_i32x4_shuffle(r0, r1, 0, 4, 1, 5);
    const v128_t t01_hi = wasm_i32x4_shuffle(r0, r1, 2, 6, 3, 7);
    const v128_t t23_lo = wasm_i32x4_shuffle(r2, r3, 0, 4, 1, 5);
    const v128_t t23_hi = wasm_i32x4_shuffle(r2, r3, 2, 6, 3, 7);

    const v128_t c0 = wasm_i32x4_shuffle(t01_lo, t23_lo, 0, 1, 4, 5);
    const v128_t c1 = wasm_i32x4_shuffle(t01_lo, t23_lo, 2, 3, 6, 7);
    const v128_t c2 = wasm_i32x4_shuffle(t01_hi, t23_hi, 0, 1, 4, 5);
    const v128_t c3 = wasm_i32x4_shuffle(t01_hi, t23_hi, 2, 3, 6, 7);

    wasm_v128_store(destination + 0 * destination_stride, c0);
    wasm_v128_store(destination + 1 * destination_stride, c1);
    wasm_v128_store(destination + 2 * destination_stride, c2);
    wasm_v128_store(destination + 3 * destination_stride, c3);
}

static void vx_transpose_f32_tile4_wasm(
        const VxTransposeF32Context* context, const float* source,
        float* destination, long row_begin) {
    long column = 0;
    for (; column + 4 <= context->columns; column += 4) {
        vx_transpose_f32_4x4_wasm(
            source + row_begin * context->columns + column,
            context->columns,
            destination + column * context->rows + row_begin,
            context->rows);
    }
    for (; column < context->columns; column++) {
        for (long local = 0; local < 4; local++) {
            destination[column * context->rows + row_begin + local] =
                source[(row_begin + local) * context->columns + column];
        }
    }
}

static void vx_transpose_f32_tile8_wasm(
        const VxTransposeF32Context* context, const float* source,
        float* destination, long row_begin) {
    long column = 0;
    for (; column + 4 <= context->columns; column += 4) {
        const float* source_block =
            source + row_begin * context->columns + column;
        float* destination_block =
            destination + column * context->rows + row_begin;
        vx_transpose_f32_4x4_wasm(
            source_block, context->columns,
            destination_block, context->rows);
        vx_transpose_f32_4x4_wasm(
            source_block + 4 * context->columns, context->columns,
            destination_block + 4, context->rows);
    }
    for (; column < context->columns; column++) {
        for (long local = 0; local < VX_TRANSPOSE_F32_TILE; local++) {
            destination[column * context->rows + row_begin + local] =
                source[(row_begin + local) * context->columns + column];
        }
    }
}
#endif

#if VX_TENSOR_F32_X86_AVX2
static VX_TENSOR_F32_TARGET_AVX2 void vx_transpose_f32_tile8_avx2(
        const VxTransposeF32Context* context, const float* source,
        float* destination, long row_begin) {
    long column = 0;
    for (; column + 8 <= context->columns; column += 8) {
        __m256 r0 = _mm256_loadu_ps(source +
            (row_begin + 0) * context->columns + column);
        __m256 r1 = _mm256_loadu_ps(source +
            (row_begin + 1) * context->columns + column);
        __m256 r2 = _mm256_loadu_ps(source +
            (row_begin + 2) * context->columns + column);
        __m256 r3 = _mm256_loadu_ps(source +
            (row_begin + 3) * context->columns + column);
        __m256 r4 = _mm256_loadu_ps(source +
            (row_begin + 4) * context->columns + column);
        __m256 r5 = _mm256_loadu_ps(source +
            (row_begin + 5) * context->columns + column);
        __m256 r6 = _mm256_loadu_ps(source +
            (row_begin + 6) * context->columns + column);
        __m256 r7 = _mm256_loadu_ps(source +
            (row_begin + 7) * context->columns + column);
        __m256 t0 = _mm256_unpacklo_ps(r0, r1);
        __m256 t1 = _mm256_unpackhi_ps(r0, r1);
        __m256 t2 = _mm256_unpacklo_ps(r2, r3);
        __m256 t3 = _mm256_unpackhi_ps(r2, r3);
        __m256 t4 = _mm256_unpacklo_ps(r4, r5);
        __m256 t5 = _mm256_unpackhi_ps(r4, r5);
        __m256 t6 = _mm256_unpacklo_ps(r6, r7);
        __m256 t7 = _mm256_unpackhi_ps(r6, r7);
        __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
        __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xEE);
        __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
        __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xEE);
        __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
        __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xEE);
        __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
        __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xEE);
        _mm256_storeu_ps(destination +
            (column + 0) * context->rows + row_begin,
            _mm256_permute2f128_ps(s0, s4, 0x20));
        _mm256_storeu_ps(destination +
            (column + 1) * context->rows + row_begin,
            _mm256_permute2f128_ps(s1, s5, 0x20));
        _mm256_storeu_ps(destination +
            (column + 2) * context->rows + row_begin,
            _mm256_permute2f128_ps(s2, s6, 0x20));
        _mm256_storeu_ps(destination +
            (column + 3) * context->rows + row_begin,
            _mm256_permute2f128_ps(s3, s7, 0x20));
        _mm256_storeu_ps(destination +
            (column + 4) * context->rows + row_begin,
            _mm256_permute2f128_ps(s0, s4, 0x31));
        _mm256_storeu_ps(destination +
            (column + 5) * context->rows + row_begin,
            _mm256_permute2f128_ps(s1, s5, 0x31));
        _mm256_storeu_ps(destination +
            (column + 6) * context->rows + row_begin,
            _mm256_permute2f128_ps(s2, s6, 0x31));
        _mm256_storeu_ps(destination +
            (column + 7) * context->rows + row_begin,
            _mm256_permute2f128_ps(s3, s7, 0x31));
    }
    for (; column < context->columns; column++) {
        for (long local = 0; local < VX_TRANSPOSE_F32_TILE; local++) {
            destination[column * context->rows + row_begin + local] =
                source[(row_begin + local) * context->columns + column];
        }
    }
}

static VX_TENSOR_F32_TARGET_AVX2 int vx_maxpool_f32_update_avx2(
        float* destination, const float* source, int channels) {
    int channel = 0;
    for (; channel + 8 <= channels; channel += 8) {
        __m256 current = _mm256_loadu_ps(destination + channel);
        __m256 value = _mm256_loadu_ps(source + channel);
        _mm256_storeu_ps(destination + channel,
                         _mm256_max_ps(current, value));
    }
    return channel;
}
#endif

static void vx_transpose_f32_worker(void* opaque, int begin, int end) {
    VxTransposeF32Context* context = (VxTransposeF32Context*)opaque;
    for (int task = begin; task < end; task++) {
        const long batch_index = task / context->row_tiles;
        const long tile = task % context->row_tiles;
        long row_begin = tile * VX_TRANSPOSE_F32_TILE;
        long row_end = row_begin + VX_TRANSPOSE_F32_TILE;
        const float* source = context->source +
            batch_index * context->rows * context->columns;
        float* destination = context->destination +
            batch_index * context->columns * context->rows;
        if (row_end > context->rows) row_end = context->rows;
#if VX_TENSOR_F32_X86_AVX2
        if (context->use_avx2 &&
            row_end - row_begin == VX_TRANSPOSE_F32_TILE) {
            vx_transpose_f32_tile8_avx2(context, source, destination,
                                        row_begin);
            continue;
        }
#endif
#if VX_TENSOR_F32_ARM_NEON
        if (context->use_neon) {
            if (row_end - row_begin == VX_TRANSPOSE_F32_TILE) {
                vx_transpose_f32_tile8_neon(context, source, destination,
                                            row_begin);
                continue;
            }
            if (row_end - row_begin >= 4) {
                vx_transpose_f32_tile4_neon(context, source, destination,
                                            row_begin);
                row_begin += 4;
            }
        }
#endif
#if VX_TENSOR_F32_WASM_SIMD
        if (context->use_wasm) {
            if (row_end - row_begin == VX_TRANSPOSE_F32_TILE) {
                vx_transpose_f32_tile8_wasm(context, source, destination,
                                            row_begin);
                continue;
            }
            if (row_end - row_begin >= 4) {
                vx_transpose_f32_tile4_wasm(context, source, destination,
                                            row_begin);
                row_begin += 4;
            }
        }
#endif
        for (long row = row_begin; row < row_end; row++) {
            for (long column = 0; column < context->columns; column++) {
                destination[column * context->rows + row] =
                    source[row * context->columns + column];
            }
        }
    }
}

typedef struct VxTranspose2DBlockedContext {
    const float* source;
    float* destination;
    long rows;
    long columns;
    long tile_m;
    long tile_n;
    long col_tiles;
    int use_avx2;
    int use_neon;
    int use_wasm;
} VxTranspose2DBlockedContext;

static void vx_transpose_2d_blocked_worker(void* opaque, int begin, int end) {
    const VxTranspose2DBlockedContext* ctx = (const VxTranspose2DBlockedContext*)opaque;
    const long rows = ctx->rows;
    const long cols = ctx->columns;
    const long col_tiles = ctx->col_tiles;
    const long tile_m = ctx->tile_m;
    const long tile_n = ctx->tile_n;
    const long row_tiles = (rows + tile_m - 1) / tile_m;
    const long tiles_per_batch = row_tiles * col_tiles;

    for (int task = begin; task < end; task++) {
        long b = task / tiles_per_batch;
        long tile = task % tiles_per_batch;
        long r_tile, c_tile;
        if (cols >= rows) {
            c_tile = tile / row_tiles;
            r_tile = tile % row_tiles;
        } else {
            r_tile = tile / col_tiles;
            c_tile = tile % col_tiles;
        }

        long r0 = r_tile * tile_m;
        long r_max = r0 + tile_m < rows ? r0 + tile_m : rows;
        long c0 = c_tile * tile_n;
        long c_max = c0 + tile_n < cols ? c0 + tile_n : cols;

        const float* src = ctx->source + b * rows * cols;
        float* dst = ctx->destination + b * cols * rows;

        if (r_max - r0 == 16 && c_max - c0 == 16) {
#if VX_TENSOR_F32_ARM_NEON
            if (ctx->use_neon) {
                for (int i = 0; i < 4; i++) {
                    for (int j = 0; j < 4; j++) {
                        vx_transpose_f32_4x4_neon(
                            src + (r0 + i * 4) * cols + (c0 + j * 4), cols,
                            dst + (c0 + j * 4) * rows + (r0 + i * 4), rows);
                    }
                }
                continue;
            }
#elif VX_TENSOR_F32_WASM_SIMD
            if (ctx->use_wasm) {
                float local[16][16];
                for (int r = 0; r < 16; r++) {
                    const float* s = src + (r0 + r) * cols + c0;
                    for (int c = 0; c < 16; c++) {
                        local[c][r] = s[c];
                    }
                }
                for (int c = 0; c < 16; c++) {
                    float* d = dst + (c0 + c) * rows + r0;
                    wasm_v128_store(d + 0, wasm_v128_load(&local[c][0]));
                    wasm_v128_store(d + 4, wasm_v128_load(&local[c][4]));
                    wasm_v128_store(d + 8, wasm_v128_load(&local[c][8]));
                    wasm_v128_store(d + 12, wasm_v128_load(&local[c][12]));
                }
                continue;
            }
#endif
            float local[16][16];
            for (int r = 0; r < 16; r++) {
                const float* s = src + (r0 + r) * cols + c0;
                for (int c = 0; c < 16; c++) {
                    local[c][r] = s[c];
                }
            }
            for (int c = 0; c < 16; c++) {
                float* d = dst + (c0 + c) * rows + r0;
                memcpy(d, local[c], 16 * sizeof(float));
            }
            continue;
        }
#if VX_TENSOR_F32_WASM_SIMD
        if (ctx->use_wasm) {
            long r = r0;
            for (; r + 4 <= r_max; r += 4) {
                long c = c0;
                for (; c + 4 <= c_max; c += 4) {
                    vx_transpose_f32_4x4_wasm(
                        src + r * cols + c, cols,
                        dst + c * rows + r, rows);
                }
                for (; c < c_max; c++) {
                    for (long lr = 0; lr < 4; lr++) {
                        dst[c * rows + (r + lr)] = src[(r + lr) * cols + c];
                    }
                }
            }
            for (; r < r_max; r++) {
                for (long c = c0; c < c_max; c++) {
                    dst[c * rows + r] = src[r * cols + c];
                }
            }
            continue;
        }
#endif
        for (long r = r0; r < r_max; r++) {
            for (long c = c0; c < c_max; c++) {
                dst[c * rows + r] = src[r * cols + c];
            }
        }
    }
}

static int vx_transpose_batched2d_f32(const float* source,
                                      float* destination, long batch,
                                      long rows, long columns) {
    if (rows >= 16 && columns >= 16) {
        const long tile_m = 16;
        const long tile_n = 16;
        const long row_tiles_2d = (rows + tile_m - 1) / tile_m;
        const long col_tiles_2d = (columns + tile_n - 1) / tile_n;
        const uint64_t tasks_2d = (uint64_t)batch * (uint64_t)row_tiles_2d * (uint64_t)col_tiles_2d;
        if (tasks_2d <= INT_MAX) {
            VxTranspose2DBlockedContext ctx_2d = {
                source, destination, rows, columns, tile_m, tile_n, col_tiles_2d,
                VX_TENSOR_F32_X86_AVX2 && vx_kernel_platform()->has_avx2,
                VX_TENSOR_F32_ARM_NEON && vx_kernel_platform()->has_neon,
                VX_TENSOR_F32_WASM_SIMD,
            };
            int threads = vx_kernels_thread_count();
            if ((uint64_t)batch * (uint64_t)rows * (uint64_t)columns <
                VX_TRANSPOSE_F32_PARALLEL_ELEMENTS || threads <= 1) {
                vx_transpose_2d_blocked_worker(&ctx_2d, 0, (int)tasks_2d);
                return 1;
            }
            int grain = 1 + ((int)tasks_2d - 1) / (threads * 4);
            if (grain < 1) grain = 1;
            vx_kernels_parallel_for((int)tasks_2d, grain,
                                    vx_transpose_2d_blocked_worker, &ctx_2d);
            return 1;
        }
    }
    const long row_tiles =
        (rows + VX_TRANSPOSE_F32_TILE - 1) / VX_TRANSPOSE_F32_TILE;
    const uint64_t tasks = (uint64_t)batch * (uint64_t)row_tiles;
    VxTransposeF32Context context = {
        source, destination, batch, rows, columns, row_tiles,
        VX_TENSOR_F32_X86_AVX2 && vx_kernel_platform()->has_avx2,
        VX_TENSOR_F32_ARM_NEON && vx_kernel_platform()->has_neon,
        VX_TENSOR_F32_WASM_SIMD,
    };
    int grain;
    int threads = vx_kernels_thread_count();
    if (tasks > INT_MAX) return 0;
    if ((uint64_t)batch * (uint64_t)rows * (uint64_t)columns <
        VX_TRANSPOSE_F32_PARALLEL_ELEMENTS || threads <= 1) {
        vx_transpose_f32_worker(&context, 0, (int)tasks);
        return 1;
    }
    grain = 1 + ((int)tasks - 1) / (threads * 4);
    if (grain < 1) grain = 1;
    vx_kernels_parallel_for((int)tasks, grain,
                            vx_transpose_f32_worker, &context);
    return 1;
}

void vx_transpose2d_f32(const float* src, float* dst, long rows, long cols) {
    if (!src || !dst || rows <= 0 || cols <= 0) return;
    (void)vx_transpose_batched2d_f32(src, dst, 1, rows, cols);
}

int vx_transpose_nd_f32_native_validated(const float* input, float* output,
        const uint32_t* input_shape, const uint32_t* permutation,
        uint32_t rank, uint32_t elements) {
    uint64_t product = 1;
    uint32_t seen = 0;
    if (!input || !output || !input_shape || !permutation || !rank || rank > 8)
        return transpose_nd_f32(input, output, input_shape, permutation,
                                rank, elements);
    for (uint32_t axis = 0; axis < rank; axis++) {
        const uint32_t source_axis = permutation[axis];
        if (!input_shape[axis] || source_axis >= rank ||
            (seen & (1u << source_axis)) ||
            product > UINT64_MAX / input_shape[axis]) {
            return transpose_nd_f32(input, output, input_shape, permutation,
                                    rank, elements);
        }
        seen |= 1u << source_axis;
        product *= input_shape[axis];
    }
    if (product != elements) {
        return transpose_nd_f32(input, output, input_shape, permutation,
                                rank, elements);
    }
    for (uint32_t axis = 0; axis < rank; axis++) {
        if (permutation[axis] != axis) goto non_identity;
    }
    if (input != output) memcpy(output, input, (size_t)elements * sizeof(float));
    return 1;

non_identity:
    if (input == output) {
        return transpose_nd_f32(input, output, input_shape, permutation,
                                rank, elements);
    }
    if (rank == 4 && permutation[0] == 0 && permutation[1] == 3 &&
        permutation[2] == 1 && permutation[3] == 2) {
        const long rows = (long)input_shape[1] * input_shape[2];
        if (vx_transpose_batched2d_f32(input, output, input_shape[0], rows,
                                       input_shape[3])) return 1;
    }
    if (rank == 4 && permutation[0] == 0 && permutation[1] == 2 &&
        permutation[2] == 3 && permutation[3] == 1) {
        const long columns = (long)input_shape[2] * input_shape[3];
        if (vx_transpose_batched2d_f32(input, output, input_shape[0],
                                       input_shape[1], columns)) return 1;
    }
    if (rank == 4 && permutation[0] == 0 && permutation[1] == 1 &&
        permutation[2] == 3 && permutation[3] == 2) {
        if (vx_transpose_batched2d_f32(input, output,
                                       (long)input_shape[0] * input_shape[1],
                                       input_shape[2], input_shape[3])) return 1;
    }
    if (rank == 3 && permutation[0] == 0 && permutation[1] == 2 &&
        permutation[2] == 1) {
        if (vx_transpose_batched2d_f32(input, output, input_shape[0],
                                       input_shape[1], input_shape[2])) return 1;
    }
    return transpose_nd_f32(input, output, input_shape, permutation,
                            rank, elements);
}

void vx_maxpool2d_f32(const float* input, float* output,
                           int n, int h, int w, int c,
                           int oh, int ow,
                           int ky, int kx, int sy, int sx,
                           int py, int px) {
    const int use_avx2 = VX_TENSOR_F32_X86_AVX2 && vx_kernel_platform()->has_avx2;
    for (int b = 0; b < n; b++) {
        const float* src_b = input + (long)b * h * w * c;
        float* dst_b = output + (long)b * oh * ow * c;
        for (int oy = 0; oy < oh; oy++) {
            int iy0 = oy * sy - py;
            for (int ox = 0; ox < ow; ox++) {
                int ix0 = ox * sx - px;
                float* dst = dst_b + ((long)oy * ow + ox) * c;
                for (int ch = 0; ch < c; ch++) dst[ch] = -3.4028234663852886e38f;
                for (int dy = 0; dy < ky; dy++) {
                    int iy = iy0 + dy;
                    if ((unsigned)iy >= (unsigned)h) continue;
                    for (int dx = 0; dx < kx; dx++) {
                        int ix = ix0 + dx;
                        if ((unsigned)ix >= (unsigned)w) continue;
                        const float* src = src_b + ((long)iy * w + ix) * c;
                        int ch = 0;
#if VX_TENSOR_F32_X86_AVX2
                        if (use_avx2)
                            ch = vx_maxpool_f32_update_avx2(dst, src, c);
#else
                        (void)use_avx2;
#endif
                        for (; ch < c; ch++) {
                            if (src[ch] > dst[ch]) dst[ch] = src[ch];
                        }
                    }
                }
            }
        }
    }
}
