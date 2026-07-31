#include "gemm_f32.h"
#include "kernel_platform.h"
#include "thread_pool.h"

#include <limits.h>
#include <stddef.h>
#include <stdint.h>

#ifndef WASM_EXPORT
#define WASM_EXPORT(name)
#endif

#if defined(__wasm_simd128__)
#include <wasm_simd128.h>
#define VX_GEMM_F32_WASM_SIMD 1
#else
#define VX_GEMM_F32_WASM_SIMD 0
#endif

#if !defined(__wasm__) && \
    (defined(__aarch64__) || defined(__arm64__) || \
     defined(__ARM_NEON) || defined(__ARM_NEON__))
#include <arm_neon.h>
#define VX_GEMM_F32_ARM_NEON 1
#if defined(__aarch64__) || defined(__arm64__)
#define VX_GEMM_F32_NEON_MLAQ_N(accumulator, weight, value) \
    vfmaq_n_f32((accumulator), (weight), (value))
#else
#define VX_GEMM_F32_NEON_MLAQ_N(accumulator, weight, value) \
    vmlaq_n_f32((accumulator), (weight), (value))
#endif
#else
#define VX_GEMM_F32_ARM_NEON 0
#endif

#if !defined(__wasm__)
#include <pthread.h>
#include <stdlib.h>
#include <unistd.h>
#if defined(__APPLE__)
#include <sys/sysctl.h>
#include <sys/types.h>
#endif
#endif

#if !defined(__wasm__) && (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <immintrin.h>
#define VX_GEMM_F32_X86_AVX2 1
#define VX_GEMM_F32_TARGET_AVX2 __attribute__((target("avx2,fma")))
#else
#define VX_GEMM_F32_X86_AVX2 0
#define VX_GEMM_F32_TARGET_AVX2
#endif

enum {
    /* MR is the number of C rows one microkernel call accumulates, so with
     * NR=8 it is also the number of live ymm accumulators.  It was 4, which
     * cannot saturate the FMA unit: an FMA retires two per cycle with about
     * four cycles of latency, so fewer than eight independent chains leaves the
     * pipeline waiting on its own results no matter how the loads are arranged.
     * Measured with benchmark_kernel_unit on a Ryzen 5 5600U at one thread, the
     * packed kernel sat at 8.7 GMAC/s while the unpacked MR4xNR16 kernel in
     * broadcast_ops.c reached 26.2 on the same 402x320x320 shape.  Eight rows
     * against NR=8 needs 8 accumulators plus a weight and a broadcast, which
     * still fits the 16 architectural ymm registers.
     *
     * NR stays 8 because it is the packed panel width and the WASM, NEON and
     * scalar microkernels are all written against it; MR is private to the
     * blocking loop, so widening it changes no layout. */
    VX_GEMM_F32_MR = 8,
    VX_GEMM_F32_NR = 8,
    VX_GEMM_F32_FALLBACK_L1_BYTES = 32 * 1024,
    VX_GEMM_F32_MAX_KC = 512,
    VX_GEMM_F32_MIN_KC = 64,
    VX_GEMM_F32_K_ALIGNMENT = 16,
    VX_GEMM_F32_PARALLEL_FLOPS = 256 * 1024,
};

#if !defined(__wasm__)
static uint32_t vx_gemm_f32_detect_l1_bytes(void) {
    uint32_t detected = VX_GEMM_F32_FALLBACK_L1_BYTES;
#if defined(__APPLE__)
    uint64_t value = 0;
    size_t value_size = sizeof(value);
    if (sysctlbyname("hw.l1dcachesize", &value, &value_size, NULL, 0) == 0 &&
        value_size == sizeof(value) && value >= 8u * 1024u &&
        value <= 1024u * 1024u) {
        detected = (uint32_t)value;
    }
#else
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    {
        long value = sysconf(_SC_LEVEL1_DCACHE_SIZE);
        if (value >= 8l * 1024l && value <= 1024l * 1024l)
            detected = (uint32_t)value;
    }
#endif
#endif
    return detected;
}
#endif

static VxGemmF32TileConfig vx_gemm_f32_make_tile_config(uint32_t l1_bytes) {
    VxGemmF32TileConfig config;
    uint32_t per_k_bytes;
    uint32_t accumulator_bytes;
    uint32_t available;
    uint32_t kc;
    config.mr = VX_GEMM_F32_MR;
    config.nr = VX_GEMM_F32_NR;
    config.l1_bytes = l1_bytes;
    config.cache_budget_bytes = config.l1_bytes - config.l1_bytes / 4u;
    per_k_bytes = (config.mr + config.nr) * (uint32_t)sizeof(float);
    accumulator_bytes = config.mr * config.nr * (uint32_t)sizeof(float);
    available = config.cache_budget_bytes > accumulator_bytes
        ? config.cache_budget_bytes - accumulator_bytes : 0u;
    kc = per_k_bytes ? available / per_k_bytes : 0u;
    kc -= kc % VX_GEMM_F32_K_ALIGNMENT;
    if (kc > VX_GEMM_F32_MAX_KC) kc = VX_GEMM_F32_MAX_KC;
    if (kc < VX_GEMM_F32_MIN_KC) kc = VX_GEMM_F32_MIN_KC;
    config.kc = kc;
    config.working_set_bytes = per_k_bytes * kc + accumulator_bytes;
    return config;
}

#if !defined(__wasm__)
static pthread_once_t g_vx_gemm_f32_runtime_once = PTHREAD_ONCE_INIT;
static VxGemmF32TileConfig g_vx_gemm_f32_tile_config;
#if VX_GEMM_F32_X86_AVX2
static int g_vx_gemm_f32_has_avx2_fma;
#endif

static void vx_gemm_f32_init_runtime_config(void) {
    g_vx_gemm_f32_tile_config =
        vx_gemm_f32_make_tile_config(vx_gemm_f32_detect_l1_bytes());
#if VX_GEMM_F32_X86_AVX2
    /* Ask the resolved platform rather than the CPU directly.  Querying
     * __builtin_cpu_supports here meant this kernel was the one place that
     * ignored VOLVOXAI_CPU_ISA, so clamping to baseline still ran the AVX2
     * microkernel and benchmark_kernel_unit reported both tiers as identical —
     * which hid the microkernel's throughput from tier comparison entirely.
     * FMA stays a separate question because the platform record does not
     * distinguish it. */
    __builtin_cpu_init();
    g_vx_gemm_f32_has_avx2_fma =
        vx_kernel_platform()->has_avx2 && __builtin_cpu_supports("fma");
#endif
}
#endif

VxGemmF32TileConfig vx_gemm_f32_tile_config(void) {
#if defined(__wasm__)
    /* Browsers expose no reliable cache-topology query. Keep this deterministic
       and conservative; the compiler folds the constant policy. */
    return vx_gemm_f32_make_tile_config(VX_GEMM_F32_FALLBACK_L1_BYTES);
#else
    pthread_once(&g_vx_gemm_f32_runtime_once, vx_gemm_f32_init_runtime_config);
    return g_vx_gemm_f32_tile_config;
#endif
}

WASM_EXPORT("gemm_f32_tile_mr")
uint32_t vx_gemm_f32_tile_mr(void) { return vx_gemm_f32_tile_config().mr; }

WASM_EXPORT("gemm_f32_tile_nr")
uint32_t vx_gemm_f32_tile_nr(void) { return vx_gemm_f32_tile_config().nr; }

WASM_EXPORT("gemm_f32_tile_kc")
uint32_t vx_gemm_f32_tile_kc(void) { return vx_gemm_f32_tile_config().kc; }

WASM_EXPORT("gemm_f32_cache_budget_bytes")
uint32_t vx_gemm_f32_cache_budget_bytes(void) {
    return vx_gemm_f32_tile_config().cache_budget_bytes;
}

WASM_EXPORT("gemm_f32_working_set_bytes")
uint32_t vx_gemm_f32_working_set_bytes(void) {
    return vx_gemm_f32_tile_config().working_set_bytes;
}

WASM_EXPORT("gemm_f32_packed_elements")
uint32_t vx_gemm_f32_packed_elements(uint32_t k, uint32_t n) {
    uint64_t panels;
    uint64_t elements;
    if (!k || !n) return 0;
    panels = ((uint64_t)n + VX_GEMM_F32_NR - 1u) / VX_GEMM_F32_NR;
    elements = panels * (uint64_t)k * VX_GEMM_F32_NR;
    return elements <= UINT32_MAX ? (uint32_t)elements : 0u;
}

WASM_EXPORT("gemm_f32_pack_b")
int vx_gemm_f32_pack_b(const float* weight, float* packed,
                       uint32_t k, uint32_t n, int out_in) {
    uint32_t panels;
    if (!weight || !packed || (out_in != 0 && out_in != 1) ||
        !vx_gemm_f32_packed_elements(k, n)) return 0;
    panels = (n + VX_GEMM_F32_NR - 1u) / VX_GEMM_F32_NR;
    for (uint32_t panel = 0; panel < panels; panel++) {
        uint32_t column = panel * VX_GEMM_F32_NR;
        float* panel_data = packed + (size_t)panel * k * VX_GEMM_F32_NR;
        for (uint32_t inner = 0; inner < k; inner++) {
            float* dst = panel_data + (size_t)inner * VX_GEMM_F32_NR;
            for (uint32_t lane = 0; lane < VX_GEMM_F32_NR; lane++) {
                uint32_t output = column + lane;
                dst[lane] = output < n
                    ? weight[out_in ? (size_t)output * k + inner
                                    : (size_t)inner * n + output]
                    : 0.0f;
            }
        }
    }
    return 1;
}

static void vx_gemm_f32_m1_scalar(const float* a, const float* packed_b,
                                  const float* bias, float* c,
                                  uint32_t k, uint32_t column, uint32_t nr,
                                  uint32_t kc, int accumulate) {
    float sums[VX_GEMM_F32_NR];
    for (uint32_t lane = 0; lane < nr; lane++)
        sums[lane] = (bias ? bias[column + lane] : 0.0f) +
                     (accumulate ? c[column + lane] : 0.0f);
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            const float value = a[inner];
            const float* weights = packed_b + (size_t)inner * VX_GEMM_F32_NR;
            for (uint32_t lane = 0; lane < nr; lane++)
                sums[lane] += value * weights[lane];
        }
    }
    for (uint32_t lane = 0; lane < nr; lane++) c[column + lane] = sums[lane];
}

static void vx_gemm_f32_mrxnr_scalar(const float* a, const float* packed_b,
                                     const float* bias, float* c,
                                     uint32_t k, uint32_t n,
                                     uint32_t row, uint32_t column,
                                     uint32_t mr, uint32_t nr, uint32_t kc,
                                     int accumulate) {
    float sums[VX_GEMM_F32_MR][VX_GEMM_F32_NR];
    for (uint32_t r = 0; r < mr; r++) {
        for (uint32_t lane = 0; lane < nr; lane++) {
            size_t output_index = (size_t)(row + r) * n + column + lane;
            sums[r][lane] = (bias ? bias[column + lane] : 0.0f) +
                            (accumulate ? c[output_index] : 0.0f);
        }
    }
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            const float* weights = packed_b + (size_t)inner * VX_GEMM_F32_NR;
            for (uint32_t r = 0; r < mr; r++) {
                float value = a[(size_t)(row + r) * k + inner];
                for (uint32_t lane = 0; lane < nr; lane++)
                    sums[r][lane] += value * weights[lane];
            }
        }
    }
    for (uint32_t r = 0; r < mr; r++) {
        for (uint32_t lane = 0; lane < nr; lane++)
            c[(size_t)(row + r) * n + column + lane] = sums[r][lane];
    }
}

#if VX_GEMM_F32_WASM_SIMD
static void vx_gemm_f32_m1_wasm(const float* a, const float* packed_b,
                                const float* bias, float* c,
                                uint32_t k, uint32_t column, uint32_t kc,
                                int accumulate) {
    v128_t low = bias ? wasm_v128_load(bias + column) : wasm_f32x4_splat(0.0f);
    v128_t high = bias ? wasm_v128_load(bias + column + 4u) : wasm_f32x4_splat(0.0f);
    if (accumulate) {
        low = wasm_f32x4_add(low, wasm_v128_load(c + column));
        high = wasm_f32x4_add(high, wasm_v128_load(c + column + 4u));
    }
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            const float* weights = packed_b + (size_t)inner * VX_GEMM_F32_NR;
            v128_t value = wasm_f32x4_splat(a[inner]);
            low = wasm_f32x4_add(low, wasm_f32x4_mul(value, wasm_v128_load(weights)));
            high = wasm_f32x4_add(high, wasm_f32x4_mul(value, wasm_v128_load(weights + 4u)));
        }
    }
    wasm_v128_store(c + column, low);
    wasm_v128_store(c + column + 4u, high);
}

static void vx_gemm_f32_mrxnr_wasm(const float* a, const float* packed_b,
                                   const float* bias, float* c,
                                   uint32_t k, uint32_t n,
                                   uint32_t row, uint32_t column,
                                   uint32_t mr, uint32_t kc, int accumulate) {
    v128_t low[VX_GEMM_F32_MR];
    v128_t high[VX_GEMM_F32_MR];
    for (uint32_t r = 0; r < mr; r++) {
        low[r] = bias ? wasm_v128_load(bias + column) : wasm_f32x4_splat(0.0f);
        high[r] = bias ? wasm_v128_load(bias + column + 4u) : wasm_f32x4_splat(0.0f);
        if (accumulate) {
            const float* output = c + (size_t)(row + r) * n + column;
            low[r] = wasm_f32x4_add(low[r], wasm_v128_load(output));
            high[r] = wasm_f32x4_add(high[r], wasm_v128_load(output + 4u));
        }
    }
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            const float* weights = packed_b + (size_t)inner * VX_GEMM_F32_NR;
            v128_t weight_low = wasm_v128_load(weights);
            v128_t weight_high = wasm_v128_load(weights + 4u);
            for (uint32_t r = 0; r < mr; r++) {
                v128_t value = wasm_f32x4_splat(a[(size_t)(row + r) * k + inner]);
                low[r] = wasm_f32x4_add(low[r], wasm_f32x4_mul(value, weight_low));
                high[r] = wasm_f32x4_add(high[r], wasm_f32x4_mul(value, weight_high));
            }
        }
    }
    for (uint32_t r = 0; r < mr; r++) {
        float* output = c + (size_t)(row + r) * n + column;
        wasm_v128_store(output, low[r]);
        wasm_v128_store(output + 4u, high[r]);
    }
}
#endif

#if VX_GEMM_F32_ARM_NEON
static void vx_gemm_f32_m1_neon(const float* a, const float* packed_b,
                                const float* bias, float* c,
                                uint32_t k, uint32_t column, uint32_t kc,
                                int accumulate) {
    float32x4_t low = bias ? vld1q_f32(bias + column) : vdupq_n_f32(0.0f);
    float32x4_t high = bias
        ? vld1q_f32(bias + column + 4u) : vdupq_n_f32(0.0f);
    if (accumulate) {
        low = vaddq_f32(low, vld1q_f32(c + column));
        high = vaddq_f32(high, vld1q_f32(c + column + 4u));
    }
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            const float* weights = packed_b +
                (size_t)inner * VX_GEMM_F32_NR;
            float value = a[inner];
            low = VX_GEMM_F32_NEON_MLAQ_N(
                low, vld1q_f32(weights), value);
            high = VX_GEMM_F32_NEON_MLAQ_N(
                high, vld1q_f32(weights + 4u), value);
        }
    }
    vst1q_f32(c + column, low);
    vst1q_f32(c + column + 4u, high);
}

static void vx_gemm_f32_mrxnr_neon(const float* a, const float* packed_b,
                                   const float* bias, float* c,
                                   uint32_t k, uint32_t n,
                                   uint32_t row, uint32_t column,
                                   uint32_t mr, uint32_t kc,
                                   int accumulate) {
    float32x4_t low[VX_GEMM_F32_MR];
    float32x4_t high[VX_GEMM_F32_MR];
    for (uint32_t r = 0; r < mr; r++) {
        low[r] = bias ? vld1q_f32(bias + column) : vdupq_n_f32(0.0f);
        high[r] = bias
            ? vld1q_f32(bias + column + 4u) : vdupq_n_f32(0.0f);
        if (accumulate) {
            const float* output = c + (size_t)(row + r) * n + column;
            low[r] = vaddq_f32(low[r], vld1q_f32(output));
            high[r] = vaddq_f32(high[r], vld1q_f32(output + 4u));
        }
    }
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            const float* weights = packed_b +
                (size_t)inner * VX_GEMM_F32_NR;
            float32x4_t weight_low = vld1q_f32(weights);
            float32x4_t weight_high = vld1q_f32(weights + 4u);
            for (uint32_t r = 0; r < mr; r++) {
                float value = a[(size_t)(row + r) * k + inner];
                low[r] = VX_GEMM_F32_NEON_MLAQ_N(
                    low[r], weight_low, value);
                high[r] = VX_GEMM_F32_NEON_MLAQ_N(
                    high[r], weight_high, value);
            }
        }
    }
    for (uint32_t r = 0; r < mr; r++) {
        float* output = c + (size_t)(row + r) * n + column;
        vst1q_f32(output, low[r]);
        vst1q_f32(output + 4u, high[r]);
    }
}
#endif

#if VX_GEMM_F32_X86_AVX2
static int vx_gemm_f32_has_avx2_fma(void) {
    return g_vx_gemm_f32_has_avx2_fma;
}

static VX_GEMM_F32_TARGET_AVX2 void vx_gemm_f32_m1_avx2(
        const float* a, const float* packed_b, const float* bias, float* c,
        uint32_t k, uint32_t column, uint32_t kc, int accumulate) {
    __m256 sum = bias ? _mm256_loadu_ps(bias + column) : _mm256_setzero_ps();
    if (accumulate) sum = _mm256_add_ps(sum, _mm256_loadu_ps(c + column));
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            __m256 value = _mm256_set1_ps(a[inner]);
            __m256 weight = _mm256_loadu_ps(packed_b + (size_t)inner * VX_GEMM_F32_NR);
            sum = _mm256_fmadd_ps(value, weight, sum);
        }
    }
    _mm256_storeu_ps(c + column, sum);
}

static VX_GEMM_F32_TARGET_AVX2 void vx_gemm_f32_mrxnr_avx2(
        const float* a, const float* packed_b, const float* bias, float* c,
        uint32_t k, uint32_t n, uint32_t row, uint32_t column,
        uint32_t mr, uint32_t kc, int accumulate) {
    __m256 sums[VX_GEMM_F32_MR];
    for (uint32_t r = 0; r < mr; r++) {
        sums[r] = bias ? _mm256_loadu_ps(bias + column) : _mm256_setzero_ps();
        if (accumulate)
            sums[r] = _mm256_add_ps(sums[r],
                _mm256_loadu_ps(c + (size_t)(row + r) * n + column));
    }
    for (uint32_t inner_base = 0; inner_base < k; inner_base += kc) {
        uint32_t inner_end = inner_base + kc < k ? inner_base + kc : k;
        for (uint32_t inner = inner_base; inner < inner_end; inner++) {
            __m256 weight = _mm256_loadu_ps(packed_b + (size_t)inner * VX_GEMM_F32_NR);
            for (uint32_t r = 0; r < mr; r++) {
                __m256 value = _mm256_set1_ps(a[(size_t)(row + r) * k + inner]);
                sums[r] = _mm256_fmadd_ps(value, weight, sums[r]);
            }
        }
    }
    for (uint32_t r = 0; r < mr; r++)
        _mm256_storeu_ps(c + (size_t)(row + r) * n + column, sums[r]);
}
#endif

enum {
    VX_GEMM_F32_KERNEL_SCALAR = 0,
    VX_GEMM_F32_KERNEL_WASM = 1,
    VX_GEMM_F32_KERNEL_AVX2 = 2,
    VX_GEMM_F32_KERNEL_NEON = 3,
};

typedef struct {
    const float* a;
    const float* packed_b;
    const float* bias;
    float* c;
    uint32_t m;
    uint32_t k;
    uint32_t n;
    uint32_t row_blocks;
    uint32_t column_blocks;
    uint32_t kc;
    int accumulate;
    int kernel;
} VxGemmF32Context;

static void vx_gemm_f32_worker(void* opaque, int begin, int end) {
    VxGemmF32Context* context = (VxGemmF32Context*)opaque;
    for (int task = begin; task < end; task++) {
        uint32_t column_block = (uint32_t)task / context->row_blocks;
        uint32_t row_block = (uint32_t)task % context->row_blocks;
        uint32_t row = row_block * VX_GEMM_F32_MR;
        uint32_t column = column_block * VX_GEMM_F32_NR;
        uint32_t mr = context->m - row < VX_GEMM_F32_MR
            ? context->m - row : VX_GEMM_F32_MR;
        uint32_t nr = context->n - column < VX_GEMM_F32_NR
            ? context->n - column : VX_GEMM_F32_NR;
        const float* panel = context->packed_b +
            (size_t)column_block * context->k * VX_GEMM_F32_NR;
        if (nr == VX_GEMM_F32_NR && mr == 1u) {
#if VX_GEMM_F32_WASM_SIMD
            if (context->kernel == VX_GEMM_F32_KERNEL_WASM) {
                vx_gemm_f32_m1_wasm(context->a + (size_t)row * context->k,
                    panel, context->bias, context->c + (size_t)row * context->n,
                    context->k, column, context->kc, context->accumulate);
                continue;
            }
#endif
#if VX_GEMM_F32_ARM_NEON
            if (context->kernel == VX_GEMM_F32_KERNEL_NEON) {
                vx_gemm_f32_m1_neon(context->a +
                    (size_t)row * context->k, panel, context->bias,
                    context->c + (size_t)row * context->n, context->k,
                    column, context->kc, context->accumulate);
                continue;
            }
#endif
#if VX_GEMM_F32_X86_AVX2
            if (context->kernel == VX_GEMM_F32_KERNEL_AVX2) {
                vx_gemm_f32_m1_avx2(context->a + (size_t)row * context->k,
                    panel, context->bias, context->c + (size_t)row * context->n,
                    context->k, column, context->kc, context->accumulate);
                continue;
            }
#endif
            vx_gemm_f32_m1_scalar(context->a + (size_t)row * context->k,
                panel, context->bias, context->c + (size_t)row * context->n,
                context->k, column, nr, context->kc, context->accumulate);
        } else if (nr == VX_GEMM_F32_NR) {
#if VX_GEMM_F32_WASM_SIMD
            if (context->kernel == VX_GEMM_F32_KERNEL_WASM) {
                vx_gemm_f32_mrxnr_wasm(context->a, panel, context->bias,
                    context->c, context->k, context->n, row, column, mr,
                    context->kc, context->accumulate);
                continue;
            }
#endif
#if VX_GEMM_F32_ARM_NEON
            if (context->kernel == VX_GEMM_F32_KERNEL_NEON) {
                vx_gemm_f32_mrxnr_neon(context->a, panel, context->bias,
                    context->c, context->k, context->n, row, column, mr,
                    context->kc, context->accumulate);
                continue;
            }
#endif
#if VX_GEMM_F32_X86_AVX2
            if (context->kernel == VX_GEMM_F32_KERNEL_AVX2) {
                vx_gemm_f32_mrxnr_avx2(context->a, panel, context->bias,
                    context->c, context->k, context->n, row, column, mr,
                    context->kc, context->accumulate);
                continue;
            }
#endif
            vx_gemm_f32_mrxnr_scalar(context->a, panel, context->bias,
                context->c, context->k, context->n, row, column, mr, nr,
                context->kc, context->accumulate);
        } else if (mr == 1u) {
            vx_gemm_f32_m1_scalar(context->a + (size_t)row * context->k,
                panel, context->bias, context->c + (size_t)row * context->n,
                context->k, column, nr, context->kc, context->accumulate);
        } else {
            vx_gemm_f32_mrxnr_scalar(context->a, panel, context->bias,
                context->c, context->k, context->n, row, column, mr, nr,
                context->kc, context->accumulate);
        }
    }
}

static int vx_gemm_f32_run_packed_impl(const float* a, const float* packed_b,
                                       const float* bias, float* c,
                                       uint32_t m, uint32_t k, uint32_t n,
                                       int accumulate) {
    VxGemmF32TileConfig tile;
    VxGemmF32Context context;
    uint64_t tasks;
    uint64_t rows_by_inner;
    int parallel_work;
    if (!a || !packed_b || !c || !m || !k || !n ||
        !vx_gemm_f32_packed_elements(k, n)) return 0;
    tile = vx_gemm_f32_tile_config();
    context.a = a;
    context.packed_b = packed_b;
    context.bias = bias;
    context.c = c;
    context.m = m;
    context.k = k;
    context.n = n;
    context.row_blocks = (m + VX_GEMM_F32_MR - 1u) / VX_GEMM_F32_MR;
    context.column_blocks = (n + VX_GEMM_F32_NR - 1u) / VX_GEMM_F32_NR;
    context.kc = tile.kc;
    context.accumulate = accumulate;
    context.kernel = VX_GEMM_F32_KERNEL_SCALAR;
#if VX_GEMM_F32_WASM_SIMD
    context.kernel = VX_GEMM_F32_KERNEL_WASM;
#elif VX_GEMM_F32_ARM_NEON
    context.kernel = VX_GEMM_F32_KERNEL_NEON;
#elif VX_GEMM_F32_X86_AVX2
    if (vx_gemm_f32_has_avx2_fma()) context.kernel = VX_GEMM_F32_KERNEL_AVX2;
#endif
    tasks = (uint64_t)context.row_blocks * context.column_blocks;
    if (!tasks || tasks > INT_MAX) return 0;
    /* Avoid a three-operand overflow check here: wasm32 lowers the resulting
     * 128-bit multiply through compiler-rt's unavailable __multi3 helper. */
    rows_by_inner = (uint64_t)m * k;
    parallel_work = rows_by_inner >= VX_GEMM_F32_PARALLEL_FLOPS ||
        rows_by_inner >= ((uint64_t)VX_GEMM_F32_PARALLEL_FLOPS + n - 1u) / n;
    if (parallel_work && tasks > 1u)
        vx_kernels_parallel_for((int)tasks, 1, vx_gemm_f32_worker, &context);
    else
        vx_gemm_f32_worker(&context, 0, (int)tasks);
    return 1;
}

WASM_EXPORT("gemm_f32_packed")
int vx_gemm_f32_run_packed(const float* a, const float* packed_b,
                           const float* bias, float* c,
                           uint32_t m, uint32_t k, uint32_t n) {
    return vx_gemm_f32_run_packed_impl(a, packed_b, bias, c, m, k, n, 0);
}

int vx_gemm_f32_run_packed_add(const float* a, const float* packed_b,
                               const float* bias, float* c,
                               uint32_t m, uint32_t k, uint32_t n) {
    return vx_gemm_f32_run_packed_impl(a, packed_b, bias, c, m, k, n, 1);
}

#ifndef __wasm__
const float* vx_gemm_f32_pack_cache(VxGemmF32Cache* cache, int node_index,
                                    const float* weight,
                                    uint32_t k, uint32_t n, int out_in) {
    VxGemmF32CacheEntry* entry;
    uint32_t elements;
    float* packed;
    if (!cache || node_index < 0 ||
        node_index >= VX_GEMM_F32_CACHE_CAPACITY || !weight ||
        (out_in != 0 && out_in != 1)) return NULL;
    elements = vx_gemm_f32_packed_elements(k, n);
    if (!elements) return NULL;
#if SIZE_MAX <= UINT32_MAX
    if (elements > SIZE_MAX / sizeof(float)) return NULL;
#endif
    entry = &cache->entries[node_index];
    if (entry->packed && entry->source == weight && entry->k == k &&
        entry->n == n && entry->out_in == out_in) return entry->packed;
    free(entry->packed);
    entry->packed = NULL;
    packed = (float*)malloc((size_t)elements * sizeof(float));
    if (!packed || !vx_gemm_f32_pack_b(weight, packed, k, n, out_in)) {
        free(packed);
        entry->source = NULL;
        entry->k = entry->n = 0;
        entry->out_in = 0;
        return NULL;
    }
    entry->source = weight;
    entry->packed = packed;
    entry->k = k;
    entry->n = n;
    entry->out_in = out_in;
    return packed;
}

void vx_gemm_f32_cache_free_all(VxGemmF32Cache* cache) {
    if (!cache) return;
    for (int index = 0; index < VX_GEMM_F32_CACHE_CAPACITY; index++) {
        free(cache->entries[index].packed);
        cache->entries[index].source = NULL;
        cache->entries[index].packed = NULL;
        cache->entries[index].k = 0;
        cache->entries[index].n = 0;
        cache->entries[index].out_in = 0;
    }
}
#else
const float* vx_gemm_f32_pack_cache(VxGemmF32Cache* cache, int node_index,
                                    const float* weight,
                                    uint32_t k, uint32_t n, int out_in) {
    (void)cache;
    (void)node_index;
    (void)weight;
    (void)k;
    (void)n;
    (void)out_in;
    return NULL;
}

void vx_gemm_f32_cache_free_all(VxGemmF32Cache* cache) { (void)cache; }
#endif
