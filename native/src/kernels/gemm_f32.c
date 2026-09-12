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
/* Multiply by one lane of an already-loaded vector, so packed A needs no
 * broadcast.  Only aarch64 has the laneq form over a full 128-bit vector. */
#define VX_GEMM_F32_NEON_LANE(accumulator, weight, values, lane) \
    vfmaq_laneq_f32((accumulator), (weight), (values), (lane))
#else
#define VX_GEMM_F32_NEON_MLAQ_N(accumulator, weight, value) \
    vmlaq_n_f32((accumulator), (weight), (value))
#define VX_GEMM_F32_NEON_LANE(accumulator, weight, values, lane) \
    vmlaq_n_f32((accumulator), (weight), vgetq_lane_f32((values), (lane)))
#endif
#else
#define VX_GEMM_F32_ARM_NEON 0
#endif

#if !defined(__wasm__)
#include "../runtime/vx_thread.h"
#include <stdlib.h>
#include <string.h>
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
#define VX_GEMM_F32_X86_AVX512 1
#define VX_GEMM_F32_TARGET_AVX512 \
    __attribute__((target("avx512f,avx512vl,avx512bw,avx512dq")))
#else
#define VX_GEMM_F32_X86_AVX2 0
#define VX_GEMM_F32_TARGET_AVX2
#define VX_GEMM_F32_X86_AVX512 0
#define VX_GEMM_F32_TARGET_AVX512
#endif

enum {
    /* Baseline tile, used by the WASM, NEON and scalar microkernels, which are
     * all written against an eight-wide panel.  MR=8 against NR=8 gives eight
     * independent FMA chains, which is the minimum that covers two FMA pipes at
     * about four cycles of latency.
     *
     * These are the *baseline* values, not the only ones.  The panel width is
     * resolved per ISA by vx_gemm_f32_panel_width; an ISA with a wider
     * microkernel packs wider panels.  Everything that touches the packed
     * layout — packed_elements, pack_b, and every microkernel — reads that one
     * resolved value, so the layout can never disagree with the kernel walking
     * it. */
    VX_GEMM_F32_MR = 8,
    VX_GEMM_F32_NR = 8,
    /* AVX2 tile.  Twelve accumulators plus two B vectors plus one broadcast is
     * fifteen of the sixteen architectural ymm registers, so it does not spill;
     * six rows by two vectors is also the widest blocking that does not.  The
     * spatial convolution kernel settled on the same six-by-sixteen shape for
     * the same reason.  Against NR=8 the ratio is worse than it looks: eight
     * FMAs need one weight load and eight broadcasts, while 6x16 gets twelve
     * FMAs from two loads and six broadcasts. */
    VX_GEMM_F32_MR_AVX2 = 6,
    VX_GEMM_F32_NR_AVX2 = 16,
    VX_GEMM_F32_FALLBACK_L1_BYTES = 32 * 1024,
    VX_GEMM_F32_FALLBACK_L2_BYTES = 512 * 1024,
    VX_GEMM_F32_FALLBACK_L3_BYTES = 8 * 1024 * 1024,
    VX_GEMM_F32_MAX_KC = 512,
    VX_GEMM_F32_MIN_KC = 64,
    VX_GEMM_F32_K_ALIGNMENT = 16,
    VX_GEMM_F32_PARALLEL_FLOPS = 256 * 1024,
    /* AVX-512 tile.  Thirty-two zmm instead of sixteen ymm is the reason to
     * have a separate kernel here at all: on a client part such as Tiger Lake
     * the two 256-bit FMA units are fused into one 512-bit unit, so the peak
     * rate is unchanged and only the register file and the instruction count
     * improve.  Eight rows by two 512-bit lanes is sixteen accumulators plus
     * two B and one broadcast, nineteen of thirty-two, and gets sixteen
     * multiply-adds from ten memory operations against the AVX2 tile's twelve
     * from eight. */
    VX_GEMM_F32_MR_AVX512 = 8,
    VX_GEMM_F32_NR_AVX512 = 32,
    /* NEON tile.  Sixteen accumulator q-registers plus two B and two A is
     * twenty of the thirty-two aarch64 has.  The panel stays eight wide, which
     * is what the existing NEON and scalar kernels already assume. */
    VX_GEMM_F32_MR_NEON = 8,
    VX_GEMM_F32_NR_NEON = 8,
    /* WASM tile.  No architectural register count is exposed to count against,
     * so this was measured rather than reasoned: under node on this host,
     * 402x320x1280 runs at 18.3 GMAC/s at four rows and 17.6 at six, and
     * 402x320x320 at 19.5 against 11.9.  Six is where an engine starts
     * spilling twelve v128 accumulators plus operands. */
    VX_GEMM_F32_MR_WASM = 4,
    VX_GEMM_F32_NR_WASM = 8,
    /* Largest tile any ISA here selects.  Fixes the stack tile in the edge
     * microkernel so it needs no allocation. */
    VX_GEMM_F32_NR_MAX = 32,
    VX_GEMM_F32_MR_MAX = 8,
    /* WASM has no allocator, so its A block lives in a fixed buffer and the
     * plan clamps mc to fit.  64 KiB holds a 44-row block at the WASM kc,
     * which is enough for the blocking to pay while staying a small fraction
     * of a browser heap. */
    VX_GEMM_F32_WASM_SCRATCH_FLOATS = 16384,
};

#if !defined(__wasm__)
static uint32_t vx_gemm_f32_detect_cache_bytes(int level, uint32_t fallback,
                                               uint32_t low, uint32_t high) {
    uint32_t detected = fallback;
#if defined(__APPLE__)
    const char* name = level == 1 ? "hw.l1dcachesize"
        : (level == 2 ? "hw.l2cachesize" : "hw.l3cachesize");
    uint64_t value = 0;
    size_t value_size = sizeof(value);
    if (sysctlbyname(name, &value, &value_size, NULL, 0) == 0 &&
        value_size == sizeof(value) && value >= low && value <= high) {
        detected = (uint32_t)value;
    }
#else
    long value = 0;
    switch (level) {
#if defined(_SC_LEVEL1_DCACHE_SIZE)
    case 1: value = sysconf(_SC_LEVEL1_DCACHE_SIZE); break;
#endif
#if defined(_SC_LEVEL2_CACHE_SIZE)
    case 2: value = sysconf(_SC_LEVEL2_CACHE_SIZE); break;
#endif
#if defined(_SC_LEVEL3_CACHE_SIZE)
    case 3: value = sysconf(_SC_LEVEL3_CACHE_SIZE); break;
#endif
    default: break;
    }
    if (value >= (long)low && value <= (long)high) detected = (uint32_t)value;
#endif
    return detected;
}

static uint32_t vx_gemm_f32_detect_l1_bytes(void) {
    return vx_gemm_f32_detect_cache_bytes(1, VX_GEMM_F32_FALLBACK_L1_BYTES,
                                          8u * 1024u, 1024u * 1024u);
}

static uint32_t vx_gemm_f32_detect_l2_bytes(void) {
    return vx_gemm_f32_detect_cache_bytes(2, VX_GEMM_F32_FALLBACK_L2_BYTES,
                                          64u * 1024u, 64u * 1024u * 1024u);
}

static uint32_t vx_gemm_f32_detect_l3_bytes(void) {
    return vx_gemm_f32_detect_cache_bytes(3, VX_GEMM_F32_FALLBACK_L3_BYTES,
                                          256u * 1024u, 512u * 1024u * 1024u);
}
#endif

static VxGemmF32TileConfig vx_gemm_f32_make_tile_config(uint32_t l1_bytes,
                                                        uint32_t mr,
                                                        uint32_t nr) {
    VxGemmF32TileConfig config;
    uint32_t per_k_bytes;
    uint32_t accumulator_bytes;
    uint32_t available;
    uint32_t kc;
    config.mr = mr;
    config.nr = nr;
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
static VxOnce g_vx_gemm_f32_runtime_once = VX_ONCE_INITIALIZER;
static VxGemmF32TileConfig g_vx_gemm_f32_tile_config;
static uint32_t g_vx_gemm_f32_l2_bytes;
static uint32_t g_vx_gemm_f32_l3_bytes;
#if VX_GEMM_F32_X86_AVX2
static int g_vx_gemm_f32_has_avx2_fma;
static int g_vx_gemm_f32_has_avx512f;
#endif

static void vx_gemm_f32_init_runtime_config(void) {
    uint32_t mr = VX_GEMM_F32_MR;
    uint32_t nr = VX_GEMM_F32_NR;
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
    /*
     * Default off, and this one is a microarchitecture policy rather than a
     * shape policy, which is why it is a switch and not part of the plan.
     *
     * Measured on an i3-1115G4 (Tiger Lake), clang, one thread, both tiers
     * built from this file and interleaved: AVX-512 is 0.95-0.97x on the
     * encoder and im2col shapes, 1.03-1.07x on large prefill, 1.91x on a
     * cache-resident M=1 stream, and a wash on a DRAM-bound one.  Client parts
     * fuse their two 256-bit FMA units into one 512-bit unit, so the peak rate
     * does not change and the wider panel costs K blocking: kc falls from 400
     * to 224 because one B micro-panel is twice the bytes.
     *
     * That reasoning predicted a loss and the rebuilt kernel does not show one.
     * Re-measured on the same i3-1115G4, clang, one thread, both tiers from
     * this file: 1.04x at M=1 K=1024 N=1024, 1.09x at M=1 K=319 N=333, 1.48x at
     * M=8 and 1.25x at M=32.  The one loss is 0.92x at M=4, which is a tile
     * occupancy effect and is handled in the plan rather than by refusing the
     * tier: MR is 8 here against AVX2's 6, so four rows fill half a tile
     * instead of two thirds.
     *
     * So this detects and defaults on, like every other runtime-dispatch engine
     * -- an opt-in tier is a tier nobody runs.  VOLVOX_F32_AVX512=0 turns it
     * off for a host that regresses; the switch is a cap, not the only way in.
     * Selecting it preserves byte-identical output across the supported
     * baseline, AVX2 and AVX-512 paths, so the tier is not part of the result.
     */
    {
        const char* value = getenv("VOLVOX_F32_AVX512");
        const int disabled = value && value[0] && strcmp(value, "0") == 0;
        g_vx_gemm_f32_has_avx512f =
            g_vx_gemm_f32_has_avx2_fma && vx_kernel_platform()->has_avx512f &&
            !disabled;
    }
    if (g_vx_gemm_f32_has_avx512f) {
        mr = VX_GEMM_F32_MR_AVX512;
        nr = VX_GEMM_F32_NR_AVX512;
    } else if (g_vx_gemm_f32_has_avx2_fma) {
        mr = VX_GEMM_F32_MR_AVX2;
        nr = VX_GEMM_F32_NR_AVX2;
    }
#elif VX_GEMM_F32_ARM_NEON
    if (vx_kernel_platform()->has_neon) {
        mr = VX_GEMM_F32_MR_NEON;
        nr = VX_GEMM_F32_NR_NEON;
    }
#endif
    g_vx_gemm_f32_tile_config =
        vx_gemm_f32_make_tile_config(vx_gemm_f32_detect_l1_bytes(), mr, nr);
    g_vx_gemm_f32_l2_bytes = vx_gemm_f32_detect_l2_bytes();
    g_vx_gemm_f32_l3_bytes = vx_gemm_f32_detect_l3_bytes();
}
#endif

VxGemmF32TileConfig vx_gemm_f32_tile_config(void) {
#if defined(__wasm__)
    /* Browsers expose no reliable cache-topology query. Keep this deterministic
       and conservative; the compiler folds the constant policy. */
    return vx_gemm_f32_make_tile_config(
        VX_GEMM_F32_FALLBACK_L1_BYTES,
#if VX_GEMM_F32_WASM_SIMD
        VX_GEMM_F32_MR_WASM, VX_GEMM_F32_NR_WASM);
#else
        VX_GEMM_F32_MR, VX_GEMM_F32_NR);
#endif
#else
    vx_once(&g_vx_gemm_f32_runtime_once, vx_gemm_f32_init_runtime_config);
    return g_vx_gemm_f32_tile_config;
#endif
}

/* The one resolved panel width.  packed_elements, pack_b and every microkernel
 * read this, so the packed layout and the kernel walking it cannot disagree. */
static uint32_t vx_gemm_f32_panel_width(void) {
    return vx_gemm_f32_tile_config().nr;
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
    const uint32_t nr = vx_gemm_f32_panel_width();
    uint64_t panels;
    uint64_t elements;
    if (!k || !n) return 0;
    panels = ((uint64_t)n + nr - 1u) / nr;
    elements = panels * (uint64_t)k * nr;
    return elements <= UINT32_MAX ? (uint32_t)elements : 0u;
}

WASM_EXPORT("gemm_f32_pack_b")
int vx_gemm_f32_pack_b(const float* weight, float* packed,
                       uint32_t k, uint32_t n, int out_in) {
    const uint32_t nr = vx_gemm_f32_panel_width();
    uint32_t panels;
    if (!weight || !packed || (out_in != 0 && out_in != 1) ||
        !vx_gemm_f32_packed_elements(k, n)) return 0;
    panels = (n + nr - 1u) / nr;
    for (uint32_t panel = 0; panel < panels; panel++) {
        uint32_t column = panel * nr;
        float* panel_data = packed + (size_t)panel * k * nr;
        for (uint32_t inner = 0; inner < k; inner++) {
            float* dst = panel_data + (size_t)inner * nr;
            for (uint32_t lane = 0; lane < nr; lane++) {
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

#if VX_GEMM_F32_X86_AVX512

/* Eight rows by two 512-bit lanes.  The row loop is written out rather than
 * unrolled by hand so the tile above is the only place MR is stated; a
 * constant trip count of eight unrolls anyway. */
static VX_GEMM_F32_TARGET_AVX512 void vx_gemm_f32_micro_avx512(
        const float* ap, const float* bp, float* c, uint32_t ldc, uint32_t kc,
        uint32_t rows, const float* bias, int first, int accumulate) {
    enum { MR = VX_GEMM_F32_MR_AVX512, NR = VX_GEMM_F32_NR_AVX512 };
    _Static_assert(NR == 32, "the AVX-512 microkernel writes two 512-bit lanes");
    __m512 acc[MR][2];
    uint32_t r, inner;
    for (r = 0; r < MR; r++) {
        if (first) {
            acc[r][0] = bias ? _mm512_loadu_ps(bias) : _mm512_setzero_ps();
            acc[r][1] = bias ? _mm512_loadu_ps(bias + 16) : _mm512_setzero_ps();
        }
        if ((!first || accumulate) && r < rows) {
            const __m512 c0 = _mm512_loadu_ps(c + (size_t)r * ldc);
            const __m512 c1 = _mm512_loadu_ps(c + (size_t)r * ldc + 16);
            acc[r][0] = first ? _mm512_add_ps(acc[r][0], c0) : c0;
            acc[r][1] = first ? _mm512_add_ps(acc[r][1], c1) : c1;
        } else if (!first) {
            acc[r][0] = _mm512_setzero_ps();
            acc[r][1] = _mm512_setzero_ps();
        }
    }
    for (inner = 0; inner < kc; inner++) {
        const float* bv = bp + (size_t)inner * NR;
        const float* av = ap + (size_t)inner * MR;
        const __m512 b0 = _mm512_loadu_ps(bv);
        const __m512 b1 = _mm512_loadu_ps(bv + 16);
        for (r = 0; r < MR; r++) {
            const __m512 value = _mm512_set1_ps(av[r]);
            acc[r][0] = _mm512_fmadd_ps(value, b0, acc[r][0]);
            acc[r][1] = _mm512_fmadd_ps(value, b1, acc[r][1]);
        }
    }
    for (r = 0; r < rows; r++) {
        _mm512_storeu_ps(c + (size_t)r * ldc, acc[r][0]);
        _mm512_storeu_ps(c + (size_t)r * ldc + 16, acc[r][1]);
    }
}

/* The stream regime writes a partial panel through a mask instead of spilling
 * to a scalar tail, which is the one place AVX-512 simplifies rather than
 * widens. */
static VX_GEMM_F32_TARGET_AVX512 void vx_gemm_f32_stream_avx512(
        const float* a, const float* bp, const float* bias, float* c,
        uint32_t k, uint32_t cols, int accumulate) {
    const __mmask16 low_mask = cols >= 16u
        ? (__mmask16)0xffffu : (__mmask16)((1u << cols) - 1u);
    const __mmask16 high_mask = cols <= 16u
        ? (__mmask16)0u : (__mmask16)((1u << (cols - 16u)) - 1u);
    __m512 low = bias ? _mm512_maskz_loadu_ps(low_mask, bias)
                      : _mm512_setzero_ps();
    __m512 high = bias ? _mm512_maskz_loadu_ps(high_mask, bias + 16)
                       : _mm512_setzero_ps();
    uint32_t inner;
    if (accumulate) {
        low = _mm512_add_ps(low, _mm512_maskz_loadu_ps(low_mask, c));
        high = _mm512_add_ps(high, _mm512_maskz_loadu_ps(high_mask, c + 16));
    }
    for (inner = 0; inner < k; inner++) {
        const __m512 value = _mm512_set1_ps(a[inner]);
        const float* bv = bp + (size_t)inner * VX_GEMM_F32_NR_AVX512;
        low = _mm512_fmadd_ps(value, _mm512_loadu_ps(bv), low);
        high = _mm512_fmadd_ps(value, _mm512_loadu_ps(bv + 16), high);
    }
    _mm512_mask_storeu_ps(c, low_mask, low);
    _mm512_mask_storeu_ps(c + 16, high_mask, high);
}

#endif /* VX_GEMM_F32_X86_AVX512 */

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

static int vx_gemm_f32_has_avx512f(void) {
    return g_vx_gemm_f32_has_avx512f;
}

/* The eight-wide AVX2 microkernels that used to live here are gone.  AVX2
 * resolves to a sixteen-wide panel, so they described a layout this build never
 * produces; the replacements are the packed-A microkernels below. */
#endif

enum {
    VX_GEMM_F32_MICRO_NONE = 0,
    VX_GEMM_F32_MICRO_AVX2 = 1,
    VX_GEMM_F32_MICRO_NEON = 2,
    VX_GEMM_F32_MICRO_WASM = 3,
    VX_GEMM_F32_MICRO_AVX512 = 4,
};

/* Which packed-A microkernel this build and CPU resolve to, or NONE.  The
 * order matches the compile-time exclusivity the file already assumes: a WASM
 * build is never also NEON or AVX2. */
static int vx_gemm_f32_micro_kind(void) {
#if VX_GEMM_F32_WASM_SIMD
    return VX_GEMM_F32_MICRO_WASM;
#elif VX_GEMM_F32_ARM_NEON
    return vx_kernel_platform()->has_neon
        ? VX_GEMM_F32_MICRO_NEON : VX_GEMM_F32_MICRO_NONE;
#elif VX_GEMM_F32_X86_AVX2
    if (vx_gemm_f32_has_avx512f()) return VX_GEMM_F32_MICRO_AVX512;
    return vx_gemm_f32_has_avx2_fma()
        ? VX_GEMM_F32_MICRO_AVX2 : VX_GEMM_F32_MICRO_NONE;
#else
    return VX_GEMM_F32_MICRO_NONE;
#endif
}

/* --- the plan: everything the shape decides ----------------------------- */

VxGemmF32Plan vx_gemm_f32_plan(uint32_t m, uint32_t k, uint32_t n) {
    const VxGemmF32TileConfig tile = vx_gemm_f32_tile_config();
    VxGemmF32Plan plan;
    uint32_t l2 = VX_GEMM_F32_FALLBACK_L2_BYTES;
    uint32_t l3 = VX_GEMM_F32_FALLBACK_L3_BYTES;
    uint32_t kc, mc, nc;
#if !defined(__wasm__)
    l2 = g_vx_gemm_f32_l2_bytes;
    l3 = g_vx_gemm_f32_l3_bytes;
#endif
    plan.mr = tile.mr;
    plan.nr = tile.nr;
    plan.micro = vx_gemm_f32_micro_kind();
    plan.regime = m <= 1u ? VX_GEMM_F32_REGIME_STREAM : VX_GEMM_F32_REGIME_BLOCK;
    /* The one measured AVX-512 regression is 0.92x at m=4, where MR=8 fills half
     * a row tile against AVX2's MR=6 filling two thirds. Dropping to the AVX2
     * microkernel for short row blocks is the obvious repair and is *not*
     * available: the packed B layout is resolved once, globally, by
     * vx_gemm_f32_panel_width, and every microkernel is instantiated for one
     * (MR, NR) pair. Narrowing plan.nr here would leave an AVX2 kernel walking
     * a 32-wide panel as though it were 16 -- the exact disagreement the tile
     * comment above says can never happen.
     *
     * Making it available means packing B per plan rather than per process,
     * which is a larger change than an eight percent regression on one shape
     * justifies. Recorded rather than silently left as a gap. */
    plan.blocked = plan.micro != VX_GEMM_F32_MICRO_NONE &&
        plan.regime == VX_GEMM_F32_REGIME_BLOCK;

    /* kc: one B micro-panel (kc x nr) takes about half of L1, leaving the other
     * half for the A micro-panel, the accumulators and everything else the core
     * is touching. */
    kc = tile.l1_bytes / 2u / (plan.nr * (uint32_t)sizeof(float));
    kc -= kc % 8u;
    if (kc == 0u) kc = 8u;
    if (kc > k) kc = k;

    /* mc: the packed A block (mc x kc) should sit in L2, where it is reused
     * once per column panel. */
    mc = l2 / 2u / (kc * (uint32_t)sizeof(float));
#if defined(__wasm__)
    /* No allocator here, so the A block has a hard ceiling rather than a cache
     * target.  Clamping the plan is the whole adaptation: the driver does not
     * know which platform it is on. */
    {
        const uint32_t ceiling = VX_GEMM_F32_WASM_SCRATCH_FLOATS / kc;
        if (mc > ceiling) mc = ceiling;
    }
#endif
    mc -= mc % plan.mr;
    if (mc == 0u) mc = plan.mr;
    if (mc > m) mc = m;

    /* nc: bounds how much of C the K loop revisits.  B is packed once for the
     * whole K, so unlike a general GEMM this does not also gate repacking. */
    nc = l3 / 2u / (kc * (uint32_t)sizeof(float));
    nc -= nc % plan.nr;
    if (nc == 0u) nc = plan.nr;
    if (nc > n) nc = n;

    plan.kc = kc;
    plan.mc = mc;
    plan.nc = nc;
    return plan;
}

/* ------------------------------------------------------------------------
 * The blocked path.
 *
 * Everything here except the microkernels is ISA-independent: the blocking
 * loops, the A packing, the parallel decomposition and the edge handling are
 * integer bookkeeping over the plan.  A microkernel sees only two contiguous
 * packed panels and a fixed tile, so adding an ISA is writing one function and
 * naming its tile, not redesigning the driver.
 *
 * Each ISA keeps the arithmetic it already used — AVX2 fused multiply-add,
 * NEON fused lane multiply-add, WASM separate multiply and add, since plain
 * simd128 has no FMA.  So a result is bit-identical to the same build's
 * unblocked kernel, which is the property blocking must not break, and is
 * merely close across ISAs, which was already true.
 * ------------------------------------------------------------------------ */

/* A[M,K] block -> [panel][kc][mr]: the mr values one microkernel step
 * broadcasts land on a single cache line instead of mr strides of K.  That
 * scatter, not the panel width, is why the previous packed kernel lost to the
 * unpacked one — it packed B and left A strided. */
static inline void vx_gemm_f32_pack_a(const float* a, float* packed, uint32_t k,
                               uint32_t i0, uint32_t rows, uint32_t p0,
                               uint32_t kc, uint32_t mr) {
    const uint32_t panels = (rows + mr - 1u) / mr;
    for (uint32_t panel = 0; panel < panels; panel++) {
        const uint32_t row = panel * mr;
        float* out = packed + (size_t)panel * kc * mr;
        for (uint32_t inner = 0; inner < kc; inner++) {
            float* dst = out + (size_t)inner * mr;
            for (uint32_t lane = 0; lane < mr; lane++) {
                const uint32_t r = row + lane;
                dst[lane] = r < rows
                    ? a[(size_t)(i0 + r) * k + p0 + inner] : 0.0f;
            }
        }
    }
}

_Static_assert(VX_GEMM_F32_MR_AVX512 <= VX_GEMM_F32_MR_MAX &&
               VX_GEMM_F32_NR_AVX512 <= VX_GEMM_F32_NR_MAX &&
               VX_GEMM_F32_MR_AVX2 <= VX_GEMM_F32_MR_MAX &&
               VX_GEMM_F32_MR_NEON <= VX_GEMM_F32_MR_MAX &&
               VX_GEMM_F32_MR_WASM <= VX_GEMM_F32_MR_MAX &&
               VX_GEMM_F32_MR <= VX_GEMM_F32_MR_MAX &&
               VX_GEMM_F32_NR_AVX2 <= VX_GEMM_F32_NR_MAX &&
               VX_GEMM_F32_NR_NEON <= VX_GEMM_F32_NR_MAX &&
               VX_GEMM_F32_NR_WASM <= VX_GEMM_F32_NR_MAX,
               "the shared edge microkernel holds the widest tile");

/* Ragged right or bottom edge, for every ISA.  Scalar, but it visits k in the
 * same order as the vector tile, so a shape with edges is still bit-identical
 * to one without. */
static void vx_gemm_f32_micro_edge(const float* ap, const float* bp, float* c,
                                   uint32_t ldc, uint32_t kc, uint32_t rows,
                                   uint32_t cols, uint32_t mr, uint32_t nr,
                                   const float* bias, int first,
                                   int accumulate) {
    float acc[VX_GEMM_F32_MR_MAX][VX_GEMM_F32_NR_MAX];
    uint32_t r, j, inner;
    for (r = 0; r < rows; r++) {
        for (j = 0; j < cols; j++) {
            const float previous = c[(size_t)r * ldc + j];
            acc[r][j] = first
                ? (bias ? bias[j] : 0.0f) + (accumulate ? previous : 0.0f)
                : previous;
        }
    }
    for (inner = 0; inner < kc; inner++) {
        for (r = 0; r < rows; r++) {
            const float value = ap[(size_t)inner * mr + r];
            for (j = 0; j < cols; j++)
                acc[r][j] += value * bp[(size_t)inner * nr + j];
        }
    }
    for (r = 0; r < rows; r++)
        for (j = 0; j < cols; j++) c[(size_t)r * ldc + j] = acc[r][j];
}

/* Shared prologue for a full tile: bias on the first K block, the running C
 * otherwise, and nothing loaded for rows past the tile's occupancy — the last
 * tile of the last block sits at the end of the allocation. */
#define VX_GEMM_F32_TILE_INIT(LOAD_BIAS, LOAD_C, ADD, ZERO)                   \
    do {                                                                      \
        uint32_t _r;                                                          \
        for (_r = 0; _r < MR; _r++) {                                         \
            int _live = _r < rows;                                            \
            if (first) { LOAD_BIAS(_r); }                                     \
            if ((!first || accumulate) && _live) { LOAD_C(_r); ADD(_r); }     \
            else if (!first) { ZERO(_r); }                                    \
        }                                                                     \
    } while (0)

#if VX_GEMM_F32_X86_AVX2

/* Six rows by sixteen columns: twelve accumulators, two B vectors and one
 * broadcast is fifteen ymm, so nothing spills.  Both operands are contiguous,
 * so each load advances one cache line and the prefetchers see two clean
 * streams. */
static VX_GEMM_F32_TARGET_AVX2 void vx_gemm_f32_micro_avx2(
        const float* ap, const float* bp, float* c, uint32_t ldc, uint32_t kc,
        uint32_t rows, const float* bias, int first, int accumulate) {
    enum { MR = VX_GEMM_F32_MR_AVX2, NR = VX_GEMM_F32_NR_AVX2 };
    /* Unrolled to exactly MR rows below. */
    _Static_assert(MR == 6, "AVX2 microkernel is unrolled to six rows");
    __m256 acc[MR][2];
    uint32_t r, inner;
    for (r = 0; r < MR; r++) {
        if (first) {
            acc[r][0] = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
            acc[r][1] = bias ? _mm256_loadu_ps(bias + 8) : _mm256_setzero_ps();
        }
        if ((!first || accumulate) && r < rows) {
            const __m256 c0 = _mm256_loadu_ps(c + (size_t)r * ldc);
            const __m256 c1 = _mm256_loadu_ps(c + (size_t)r * ldc + 8);
            acc[r][0] = first ? _mm256_add_ps(acc[r][0], c0) : c0;
            acc[r][1] = first ? _mm256_add_ps(acc[r][1], c1) : c1;
        } else if (!first) {
            acc[r][0] = _mm256_setzero_ps();
            acc[r][1] = _mm256_setzero_ps();
        }
    }
    for (inner = 0; inner < kc; inner++) {
        const float* bv = bp + (size_t)inner * NR;
        const float* av = ap + (size_t)inner * MR;
        const __m256 b0 = _mm256_loadu_ps(bv);
        const __m256 b1 = _mm256_loadu_ps(bv + 8);
        __m256 value;
        value = _mm256_broadcast_ss(av + 0);
        acc[0][0] = _mm256_fmadd_ps(value, b0, acc[0][0]);
        acc[0][1] = _mm256_fmadd_ps(value, b1, acc[0][1]);
        value = _mm256_broadcast_ss(av + 1);
        acc[1][0] = _mm256_fmadd_ps(value, b0, acc[1][0]);
        acc[1][1] = _mm256_fmadd_ps(value, b1, acc[1][1]);
        value = _mm256_broadcast_ss(av + 2);
        acc[2][0] = _mm256_fmadd_ps(value, b0, acc[2][0]);
        acc[2][1] = _mm256_fmadd_ps(value, b1, acc[2][1]);
        value = _mm256_broadcast_ss(av + 3);
        acc[3][0] = _mm256_fmadd_ps(value, b0, acc[3][0]);
        acc[3][1] = _mm256_fmadd_ps(value, b1, acc[3][1]);
        value = _mm256_broadcast_ss(av + 4);
        acc[4][0] = _mm256_fmadd_ps(value, b0, acc[4][0]);
        acc[4][1] = _mm256_fmadd_ps(value, b1, acc[4][1]);
        value = _mm256_broadcast_ss(av + 5);
        acc[5][0] = _mm256_fmadd_ps(value, b0, acc[5][0]);
        acc[5][1] = _mm256_fmadd_ps(value, b1, acc[5][1]);
    }
    for (r = 0; r < rows; r++) {
        _mm256_storeu_ps(c + (size_t)r * ldc, acc[r][0]);
        _mm256_storeu_ps(c + (size_t)r * ldc + 8, acc[r][1]);
    }
}

/* STREAM regime: one row against the whole weight.  There is no reuse to block
 * for, so this is a single pass that must simply not waste loads. */
static VX_GEMM_F32_TARGET_AVX2 void vx_gemm_f32_stream_avx2(
        const float* a, const float* bp, const float* bias, float* c,
        uint32_t k, uint32_t cols, int accumulate) {
    __m256 low = bias ? _mm256_loadu_ps(bias) : _mm256_setzero_ps();
    __m256 high = bias ? _mm256_loadu_ps(bias + 8) : _mm256_setzero_ps();
    uint32_t inner, j;
    float lane[VX_GEMM_F32_NR_AVX2];
    if (accumulate && cols == VX_GEMM_F32_NR_AVX2) {
        low = _mm256_add_ps(low, _mm256_loadu_ps(c));
        high = _mm256_add_ps(high, _mm256_loadu_ps(c + 8));
    }
    for (inner = 0; inner < k; inner++) {
        const __m256 value = _mm256_broadcast_ss(a + inner);
        const float* bv = bp + (size_t)inner * VX_GEMM_F32_NR_AVX2;
        low = _mm256_fmadd_ps(value, _mm256_loadu_ps(bv), low);
        high = _mm256_fmadd_ps(value, _mm256_loadu_ps(bv + 8), high);
    }
    if (cols == VX_GEMM_F32_NR_AVX2) {
        _mm256_storeu_ps(c, low);
        _mm256_storeu_ps(c + 8, high);
        return;
    }
    _mm256_storeu_ps(lane, low);
    _mm256_storeu_ps(lane + 8, high);
    for (j = 0; j < cols; j++)
        c[j] = accumulate ? c[j] + lane[j] : lane[j];
}

#endif /* VX_GEMM_F32_X86_AVX2 */

#if VX_GEMM_F32_ARM_NEON

/* Eight rows by eight columns: sixteen accumulator q-registers, two B and two
 * A, so twenty of the thirty-two.  vfmaq_laneq_f32 multiplies by a lane of a
 * loaded vector, so packed A needs no broadcast at all — one k step is four
 * loads against sixteen fused multiply-adds. */
static void vx_gemm_f32_micro_neon(
        const float* ap, const float* bp, float* c, uint32_t ldc, uint32_t kc,
        uint32_t rows, const float* bias, int first, int accumulate) {
    enum { MR = VX_GEMM_F32_MR_NEON, NR = VX_GEMM_F32_NR_NEON };
    /* Unrolled to exactly MR rows below. */
    _Static_assert(MR == 8, "NEON microkernel is unrolled to eight rows");
    _Static_assert(NR == 8, "NEON microkernel writes two 128-bit lanes");
    float32x4_t acc[MR][2];
    uint32_t r, inner;
    for (r = 0; r < MR; r++) {
        if (first) {
            acc[r][0] = bias ? vld1q_f32(bias) : vdupq_n_f32(0.0f);
            acc[r][1] = bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.0f);
        }
        if ((!first || accumulate) && r < rows) {
            const float32x4_t c0 = vld1q_f32(c + (size_t)r * ldc);
            const float32x4_t c1 = vld1q_f32(c + (size_t)r * ldc + 4);
            acc[r][0] = first ? vaddq_f32(acc[r][0], c0) : c0;
            acc[r][1] = first ? vaddq_f32(acc[r][1], c1) : c1;
        } else if (!first) {
            acc[r][0] = vdupq_n_f32(0.0f);
            acc[r][1] = vdupq_n_f32(0.0f);
        }
    }
    for (inner = 0; inner < kc; inner++) {
        const float* bv = bp + (size_t)inner * NR;
        const float* av = ap + (size_t)inner * MR;
        const float32x4_t b0 = vld1q_f32(bv);
        const float32x4_t b1 = vld1q_f32(bv + 4);
        const float32x4_t a0 = vld1q_f32(av);
        const float32x4_t a1 = vld1q_f32(av + 4);
        acc[0][0] = VX_GEMM_F32_NEON_LANE(acc[0][0], b0, a0, 0);
        acc[0][1] = VX_GEMM_F32_NEON_LANE(acc[0][1], b1, a0, 0);
        acc[1][0] = VX_GEMM_F32_NEON_LANE(acc[1][0], b0, a0, 1);
        acc[1][1] = VX_GEMM_F32_NEON_LANE(acc[1][1], b1, a0, 1);
        acc[2][0] = VX_GEMM_F32_NEON_LANE(acc[2][0], b0, a0, 2);
        acc[2][1] = VX_GEMM_F32_NEON_LANE(acc[2][1], b1, a0, 2);
        acc[3][0] = VX_GEMM_F32_NEON_LANE(acc[3][0], b0, a0, 3);
        acc[3][1] = VX_GEMM_F32_NEON_LANE(acc[3][1], b1, a0, 3);
        acc[4][0] = VX_GEMM_F32_NEON_LANE(acc[4][0], b0, a1, 0);
        acc[4][1] = VX_GEMM_F32_NEON_LANE(acc[4][1], b1, a1, 0);
        acc[5][0] = VX_GEMM_F32_NEON_LANE(acc[5][0], b0, a1, 1);
        acc[5][1] = VX_GEMM_F32_NEON_LANE(acc[5][1], b1, a1, 1);
        acc[6][0] = VX_GEMM_F32_NEON_LANE(acc[6][0], b0, a1, 2);
        acc[6][1] = VX_GEMM_F32_NEON_LANE(acc[6][1], b1, a1, 2);
        acc[7][0] = VX_GEMM_F32_NEON_LANE(acc[7][0], b0, a1, 3);
        acc[7][1] = VX_GEMM_F32_NEON_LANE(acc[7][1], b1, a1, 3);
    }
    for (r = 0; r < rows; r++) {
        vst1q_f32(c + (size_t)r * ldc, acc[r][0]);
        vst1q_f32(c + (size_t)r * ldc + 4, acc[r][1]);
    }
}

static void vx_gemm_f32_stream_neon(
        const float* a, const float* bp, const float* bias, float* c,
        uint32_t k, uint32_t cols, int accumulate) {
    float32x4_t low = bias ? vld1q_f32(bias) : vdupq_n_f32(0.0f);
    float32x4_t high = bias ? vld1q_f32(bias + 4) : vdupq_n_f32(0.0f);
    uint32_t inner, j;
    float lane[8];
    if (accumulate && cols == 8u) {
        low = vaddq_f32(low, vld1q_f32(c));
        high = vaddq_f32(high, vld1q_f32(c + 4));
    }
    for (inner = 0; inner < k; inner++) {
        const float* bv = bp + (size_t)inner * 8u;
        const float value = a[inner];
        low = VX_GEMM_F32_NEON_MLAQ_N(low, vld1q_f32(bv), value);
        high = VX_GEMM_F32_NEON_MLAQ_N(high, vld1q_f32(bv + 4), value);
    }
    if (cols == 8u) {
        vst1q_f32(c, low);
        vst1q_f32(c + 4, high);
        return;
    }
    vst1q_f32(lane, low);
    vst1q_f32(lane + 4, high);
    for (j = 0; j < cols; j++)
        c[j] = accumulate ? c[j] + lane[j] : lane[j];
}

#endif /* VX_GEMM_F32_ARM_NEON */

#if VX_GEMM_F32_WASM_SIMD

/* Four rows by eight columns: eight accumulators, two B vectors, one splat.
 * WASM exposes no register file to count, so this stays near the conservative
 * end; the multiply and add are kept separate because plain simd128 has no
 * fused multiply-add, and using the relaxed one would change results relative
 * to the same build's unblocked kernel. */
static void vx_gemm_f32_micro_wasm(
        const float* ap, const float* bp, float* c, uint32_t ldc, uint32_t kc,
        uint32_t rows, const float* bias, int first, int accumulate) {
    enum { MR = VX_GEMM_F32_MR_WASM, NR = VX_GEMM_F32_NR_WASM };
    /* The row loop is written over MR, so only the panel width is fixed. */
    _Static_assert(NR == 8, "WASM microkernel writes two 128-bit lanes");
    v128_t acc[MR][2];
    uint32_t r, inner;
    for (r = 0; r < MR; r++) {
        if (first) {
            acc[r][0] = bias ? wasm_v128_load(bias) : wasm_f32x4_splat(0.0f);
            acc[r][1] = bias ? wasm_v128_load(bias + 4) : wasm_f32x4_splat(0.0f);
        }
        if ((!first || accumulate) && r < rows) {
            const v128_t c0 = wasm_v128_load(c + (size_t)r * ldc);
            const v128_t c1 = wasm_v128_load(c + (size_t)r * ldc + 4);
            acc[r][0] = first ? wasm_f32x4_add(acc[r][0], c0) : c0;
            acc[r][1] = first ? wasm_f32x4_add(acc[r][1], c1) : c1;
        } else if (!first) {
            acc[r][0] = wasm_f32x4_splat(0.0f);
            acc[r][1] = wasm_f32x4_splat(0.0f);
        }
    }
    for (inner = 0; inner < kc; inner++) {
        const float* bv = bp + (size_t)inner * NR;
        const float* av = ap + (size_t)inner * MR;
        const v128_t b0 = wasm_v128_load(bv);
        const v128_t b1 = wasm_v128_load(bv + 4);
        for (r = 0; r < MR; r++) {
            const v128_t value = wasm_f32x4_splat(av[r]);
            acc[r][0] = wasm_f32x4_add(acc[r][0], wasm_f32x4_mul(value, b0));
            acc[r][1] = wasm_f32x4_add(acc[r][1], wasm_f32x4_mul(value, b1));
        }
    }
    for (r = 0; r < rows; r++) {
        wasm_v128_store(c + (size_t)r * ldc, acc[r][0]);
        wasm_v128_store(c + (size_t)r * ldc + 4, acc[r][1]);
    }
}

static void vx_gemm_f32_stream_wasm(
        const float* a, const float* bp, const float* bias, float* c,
        uint32_t k, uint32_t cols, int accumulate) {
    v128_t low = bias ? wasm_v128_load(bias) : wasm_f32x4_splat(0.0f);
    v128_t high = bias ? wasm_v128_load(bias + 4) : wasm_f32x4_splat(0.0f);
    uint32_t inner, j;
    float lane[8];
    if (accumulate && cols == 8u) {
        low = wasm_f32x4_add(low, wasm_v128_load(c));
        high = wasm_f32x4_add(high, wasm_v128_load(c + 4));
    }
    for (inner = 0; inner < k; inner++) {
        const float* bv = bp + (size_t)inner * 8u;
        const v128_t value = wasm_f32x4_splat(a[inner]);
        low = wasm_f32x4_add(low, wasm_f32x4_mul(value, wasm_v128_load(bv)));
        high = wasm_f32x4_add(high,
                              wasm_f32x4_mul(value, wasm_v128_load(bv + 4)));
    }
    if (cols == 8u) {
        wasm_v128_store(c, low);
        wasm_v128_store(c + 4, high);
        return;
    }
    wasm_v128_store(lane, low);
    wasm_v128_store(lane + 4, high);
    for (j = 0; j < cols; j++)
        c[j] = accumulate ? c[j] + lane[j] : lane[j];
}

/* WASM has no allocator in this build — vx_gemm_f32_pack_cache is stubbed out
 * for the same reason — so the A block lives in a static buffer and the plan
 * clamps mc to fit it.  One buffer is safe because this build has no threads:
 * vx_kernels_parallel_for runs serially without a bound pool. */
static float g_vx_gemm_f32_wasm_scratch[VX_GEMM_F32_WASM_SCRATCH_FLOATS];

#endif /* VX_GEMM_F32_WASM_SIMD */

static void vx_gemm_f32_stream(int kind, const float* a, const float* bp,
                               const float* bias, float* c, uint32_t k,
                               uint32_t cols, int accumulate) {
    switch (kind) {
#if VX_GEMM_F32_X86_AVX2
    case VX_GEMM_F32_MICRO_AVX2:
        vx_gemm_f32_stream_avx2(a, bp, bias, c, k, cols, accumulate);
        return;
#endif
#if VX_GEMM_F32_X86_AVX512
    case VX_GEMM_F32_MICRO_AVX512:
        vx_gemm_f32_stream_avx512(a, bp, bias, c, k, cols, accumulate);
        return;
#endif
#if VX_GEMM_F32_ARM_NEON
    case VX_GEMM_F32_MICRO_NEON:
        vx_gemm_f32_stream_neon(a, bp, bias, c, k, cols, accumulate);
        return;
#endif
#if VX_GEMM_F32_WASM_SIMD
    case VX_GEMM_F32_MICRO_WASM:
        vx_gemm_f32_stream_wasm(a, bp, bias, c, k, cols, accumulate);
        return;
#endif
    default:
        break;
    }
}

typedef struct {
    const float* ap;
    const float* packed_b;
    const float* bias;
    float* c;
    uint32_t k;
    uint32_t n;
    uint32_t ic;
    uint32_t mc;
    uint32_t pc;
    uint32_t kc;
    uint32_t jc;
    uint32_t nc;
    uint32_t mr;
    uint32_t nr;
    int first;
    int accumulate;
} VxGemmF32BlockedContext;

/*
 * One task is one column panel across the whole A block, so every C tile has
 * exactly one writer and the packed A block is shared read-only.
 *
 * The loop is written once and instantiated per ISA rather than dispatching a
 * microkernel per tile.  Both were measured: a shared loop reading the tile
 * from the context and calling through a switch cost 14% on x86
 * (402x320x320 fell from 56.6 to 48.5 GMAC/s), because the row stride stops
 * being a constant and the microkernel stops inlining.  Generality belongs in
 * the source, not in the object code.
 */
#define VX_GEMM_F32_DEFINE_WORKER(SUFFIX, TILE_MR, TILE_NR, MICRO)            \
static void vx_gemm_f32_pack_a_##SUFFIX(                                      \
        const float* a, float* packed, uint32_t k, uint32_t i0,               \
        uint32_t rows, uint32_t p0, uint32_t kc) {                            \
    vx_gemm_f32_pack_a(a, packed, k, i0, rows, p0, kc, (TILE_MR));            \
}                                                                             \
                                                                              \
static void vx_gemm_f32_blocked_worker_##SUFFIX(                              \
        void* opaque, int begin, int end) {                                   \
    const VxGemmF32BlockedContext* ctx =                                      \
        (const VxGemmF32BlockedContext*)opaque;                               \
    int task;                                                                 \
    for (task = begin; task < end; task++) {                                  \
        const uint32_t jr = (uint32_t)task * (TILE_NR);                       \
        const uint32_t cols =                                                 \
            ctx->nc - jr < (TILE_NR) ? ctx->nc - jr : (TILE_NR);              \
        const uint32_t panel = (ctx->jc + jr) / (TILE_NR);                    \
        const float* bp = ctx->packed_b +                                     \
            (size_t)panel * ctx->k * (TILE_NR) + (size_t)ctx->pc * (TILE_NR); \
        const float* bias = ctx->bias ? ctx->bias + ctx->jc + jr : NULL;      \
        uint32_t ir;                                                          \
        for (ir = 0; ir < ctx->mc; ir += (TILE_MR)) {                          \
            const uint32_t rows =                                             \
                ctx->mc - ir < (TILE_MR) ? ctx->mc - ir : (TILE_MR);          \
            const float* ap =                                                 \
                ctx->ap + (size_t)(ir / (TILE_MR)) * ctx->kc * (TILE_MR);     \
            float* cp = ctx->c + (size_t)(ctx->ic + ir) * ctx->n +            \
                ctx->jc + jr;                                                 \
            if (cols == (TILE_NR))                                            \
                MICRO(ap, bp, cp, ctx->n, ctx->kc, rows, bias, ctx->first,    \
                      ctx->accumulate);                                       \
            else                                                              \
                vx_gemm_f32_micro_edge(ap, bp, cp, ctx->n, ctx->kc, rows,     \
                                       cols, (TILE_MR), (TILE_NR), bias,      \
                                       ctx->first, ctx->accumulate);          \
        }                                                                     \
    }                                                                         \
}

#if VX_GEMM_F32_X86_AVX2
VX_GEMM_F32_DEFINE_WORKER(avx2, VX_GEMM_F32_MR_AVX2, VX_GEMM_F32_NR_AVX2,
                          vx_gemm_f32_micro_avx2)
#endif
#if VX_GEMM_F32_X86_AVX512
VX_GEMM_F32_DEFINE_WORKER(avx512, VX_GEMM_F32_MR_AVX512, VX_GEMM_F32_NR_AVX512,
                          vx_gemm_f32_micro_avx512)
#endif
#if VX_GEMM_F32_ARM_NEON
VX_GEMM_F32_DEFINE_WORKER(neon, VX_GEMM_F32_MR_NEON, VX_GEMM_F32_NR_NEON,
                          vx_gemm_f32_micro_neon)
#endif
#if VX_GEMM_F32_WASM_SIMD
VX_GEMM_F32_DEFINE_WORKER(wasm, VX_GEMM_F32_MR_WASM, VX_GEMM_F32_NR_WASM,
                          vx_gemm_f32_micro_wasm)
#endif

/* The two functions an ISA contributes, resolved together so the tile can
 * never differ between the pack and the kernel walking what it packed. */
typedef struct {
    VxKernelParallelFn worker;
    void (*pack_a)(const float*, float*, uint32_t, uint32_t, uint32_t,
                   uint32_t, uint32_t);
} VxGemmF32Kernels;

static VxGemmF32Kernels vx_gemm_f32_kernels_for(int kind) {
    VxGemmF32Kernels kernels;
    kernels.worker = NULL;
    kernels.pack_a = NULL;
    switch (kind) {
#if VX_GEMM_F32_X86_AVX2
    case VX_GEMM_F32_MICRO_AVX2:
        kernels.worker = vx_gemm_f32_blocked_worker_avx2;
        kernels.pack_a = vx_gemm_f32_pack_a_avx2;
        break;
#endif
#if VX_GEMM_F32_X86_AVX512
    case VX_GEMM_F32_MICRO_AVX512:
        kernels.worker = vx_gemm_f32_blocked_worker_avx512;
        kernels.pack_a = vx_gemm_f32_pack_a_avx512;
        break;
#endif
#if VX_GEMM_F32_ARM_NEON
    case VX_GEMM_F32_MICRO_NEON:
        kernels.worker = vx_gemm_f32_blocked_worker_neon;
        kernels.pack_a = vx_gemm_f32_pack_a_neon;
        break;
#endif
#if VX_GEMM_F32_WASM_SIMD
    case VX_GEMM_F32_MICRO_WASM:
        kernels.worker = vx_gemm_f32_blocked_worker_wasm;
        kernels.pack_a = vx_gemm_f32_pack_a_wasm;
        break;
#endif
    default: break;
    }
    return kernels;
}

static float* vx_gemm_f32_scratch_acquire(uint64_t elements) {
#if defined(__wasm__)
#if VX_GEMM_F32_WASM_SIMD
    return elements <= VX_GEMM_F32_WASM_SCRATCH_FLOATS
        ? g_vx_gemm_f32_wasm_scratch : NULL;
#else
    (void)elements;
    return NULL;
#endif
#else
    if (elements > SIZE_MAX / sizeof(float)) return NULL;
    /* One allocation per call, bounded by the L2 budget the plan imposed.  A
     * failure returns null so the caller takes the unblocked path rather than
     * producing a partial result. */
    return (float*)malloc((size_t)elements * sizeof(float));
#endif
}

static void vx_gemm_f32_scratch_release(float* scratch) {
#if defined(__wasm__)
    (void)scratch;
#else
    free(scratch);
#endif
}

static int vx_gemm_f32_run_blocked(const float* a, const float* packed_b,
                                   const float* bias, float* c,
                                   uint32_t m, uint32_t k, uint32_t n,
                                   const VxGemmF32Plan* plan, int accumulate) {
    const uint32_t mr = plan->mr;
    const uint32_t nr = plan->nr;
    const uint64_t scratch_elements =
        (uint64_t)((plan->mc + mr - 1u) / mr) * plan->kc * mr;
    const VxGemmF32Kernels kernels = vx_gemm_f32_kernels_for(plan->micro);
    float* scratch;
    uint32_t jc;
    if (!scratch_elements || !kernels.worker) return 0;
    scratch = vx_gemm_f32_scratch_acquire(scratch_elements);
    if (!scratch) return 0;
    for (jc = 0; jc < n; jc += plan->nc) {
        const uint32_t nc = n - jc < plan->nc ? n - jc : plan->nc;
        const int panels = (int)((nc + nr - 1u) / nr);
        uint32_t pc;
        for (pc = 0; pc < k; pc += plan->kc) {
            const uint32_t kc = k - pc < plan->kc ? k - pc : plan->kc;
            uint32_t ic;
            for (ic = 0; ic < m; ic += plan->mc) {
                const uint32_t mc = m - ic < plan->mc ? m - ic : plan->mc;
                VxGemmF32BlockedContext ctx;
                kernels.pack_a(a, scratch, k, ic, mc, pc, kc);
                ctx.ap = scratch;
                ctx.packed_b = packed_b;
                ctx.bias = bias;
                ctx.c = c;
                ctx.k = k;
                ctx.n = n;
                ctx.ic = ic;
                ctx.mc = mc;
                ctx.pc = pc;
                ctx.kc = kc;
                ctx.jc = jc;
                ctx.nc = nc;
                ctx.mr = mr;
                ctx.nr = nr;
                ctx.first = pc == 0;
                ctx.accumulate = accumulate;
                if (panels > 1)
                    vx_kernels_parallel_for(panels, 1, kernels.worker,
                                            &ctx);
                else
                    kernels.worker(&ctx, 0, panels);
            }
        }
    }
    vx_gemm_f32_scratch_release(scratch);
    return 1;
}

static int vx_gemm_f32_run_stream(const float* a, const float* packed_b,
                                  const float* bias, float* c,
                                  uint32_t k, uint32_t n,
                                  const VxGemmF32Plan* plan, int accumulate) {
    const uint32_t nr = plan->nr;
    uint32_t column;
    for (column = 0; column < n; column += nr) {
        const uint32_t cols = n - column < nr ? n - column : nr;
        vx_gemm_f32_stream(plan->micro, a,
            packed_b + (size_t)(column / nr) * k * nr,
            bias ? bias + column : NULL, c + column, k, cols, accumulate);
    }
    return 1;
}

/* Kernels for the unblocked worker, which serves the ISAs that do not yet have
 * a packed-A microkernel.  AVX2 is not among them: it never reaches here. */
enum {
    VX_GEMM_F32_KERNEL_SCALAR = 0,
    VX_GEMM_F32_KERNEL_WASM = 1,
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
    /* The worker below walks VX_GEMM_F32_NR-wide panels at the baseline tile.
     * Whenever an ISA resolves a different tile that layout is not merely
     * slower, it is wrong, so this dispatch is exhaustive rather than a
     * preference: any ISA with a packed-A microkernel leaves here through one
     * of the two regimes and never reaches the worker. */
    {
        const VxGemmF32Plan plan = vx_gemm_f32_plan(m, k, n);
        if (plan.micro != VX_GEMM_F32_MICRO_NONE) {
            if (plan.regime == VX_GEMM_F32_REGIME_STREAM)
                return vx_gemm_f32_run_stream(a, packed_b, bias, c, k, n,
                                              &plan, accumulate);
            if (vx_gemm_f32_run_blocked(a, packed_b, bias, c, m, k, n, &plan,
                                        accumulate))
                return 1;
            /* Only a scratch failure lands here, and only off WASM. */
            return 0;
        }
    }
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
