#include "tensor_f32_opt.h"
#include "cpu_features.h"
#include "kernel_platform.h"
#include "thread_pool.h"
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>
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
enum { VX_TRANSPOSE_F32_PARALLEL_ELEMENTS = 256 * 1024 };

typedef struct {
    const float* source;
    float* destination;
    long batch;
    long rows;
    long columns;
    long row_tiles;
    int use_avx2;
} VxTransposeF32Context;

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
        const long row_begin = tile * VX_TRANSPOSE_F32_TILE;
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
        for (long row = row_begin; row < row_end; row++) {
            for (long column = 0; column < context->columns; column++) {
                destination[column * context->rows + row] =
                    source[row * context->columns + column];
            }
        }
    }
}

static int vx_transpose_batched2d_f32(const float* source,
                                      float* destination, long batch,
                                      long rows, long columns) {
    const long row_tiles =
        (rows + VX_TRANSPOSE_F32_TILE - 1) / VX_TRANSPOSE_F32_TILE;
    const uint64_t tasks = (uint64_t)batch * (uint64_t)row_tiles;
    VxTransposeF32Context context = {
        source, destination, batch, rows, columns, row_tiles,
        VX_TENSOR_F32_X86_AVX2 && vx_kernel_platform()->has_avx2,
    };
    int grain;
    int threads = vx_kernels_thread_count();
    if (tasks > INT_MAX) return 0;
    if ((uint64_t)batch * (uint64_t)rows * (uint64_t)columns <
        VX_TRANSPOSE_F32_PARALLEL_ELEMENTS || threads <= 1) {
        vx_transpose_f32_worker(&context, 0, (int)tasks);
        return 1;
    }
    grain = ((int)tasks + threads * 4 - 1) / (threads * 4);
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
