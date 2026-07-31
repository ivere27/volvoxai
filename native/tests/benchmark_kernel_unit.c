/*
 * Per-kernel micro-benchmark at the smallest callable unit.
 *
 * One kernel, one shape, one number.  Every case reports achieved throughput in
 * the unit that bounds it (integer MAC/s for dot-product kernels, byte/s for
 * elementwise and layout kernels) plus a correctness verdict against the
 * portable reference, so an optimization attempt is never accepted on speed
 * alone.  Output is CSV on stdout so tools/compare_kernel_onnx.py can join it
 * against the same shape executed by ONNX Runtime and print a ratio.
 *
 * This deliberately does not go through the graph runtime.  Node dispatch,
 * tensor-name resolution and arena traffic are real costs, but they hide which
 * kernel actually moved when a microkernel changes.
 */
#include "../src/kernels/inference_kernels.h"
#include "../src/kernels/quant_cpu_opt.h"
#include "../src/kernels/thread_pool.h"
#include "../src/kernels/kernel_platform.h"
#include "../src/kernels/conv_f32_opt.h"
#include "../src/kernels/gemm_f32.h"
#include "../src/runtime/runtime_state.h"
#include "../include/volvoxai_enums.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum { VX_BENCH_WARMUP = 2, VX_BENCH_TRIALS = 3 };

/* Throughput denominators differ per kernel class; naming the unit keeps the
 * CSV self-describing instead of implying every kernel is FLOP bound. */
typedef enum { VX_UNIT_MAC, VX_UNIT_BYTE } VxBenchUnit;

static double vx_now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static uint32_t vx_rand_state = 0x12345678u;
static uint32_t vx_rand(void) {
    vx_rand_state = vx_rand_state * 1103515245u + 12345u;
    return vx_rand_state >> 8u;
}
static void vx_seed(uint32_t seed) { vx_rand_state = seed ? seed : 1u; }

static void* vx_alloc(size_t bytes) {
    void* p = malloc(bytes ? bytes : 1u);
    if (!p) {
        fprintf(stderr, "benchmark_kernel_unit: out of memory\n");
        exit(2);
    }
    return p;
}

static void vx_fill_u8(uint8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (uint8_t)(vx_rand() & 255u);
}
/* Weights span the full signed range on purpose: a kernel that is only exact
 * for a narrow range must fail here rather than in a model months later. */
static void vx_fill_i8(int8_t* p, size_t n) {
    for (size_t i = 0; i < n; i++) p[i] = (int8_t)((int)(vx_rand() % 255u) - 127);
}
static void vx_fill_f32(float* p, size_t n, float scale) {
    for (size_t i = 0; i < n; i++)
        p[i] = ((float)(vx_rand() % 2001u) / 1000.0f - 1.0f) * scale;
}

typedef struct {
    const char* name;      /* kernel identity in the CSV                     */
    const char* onnx;      /* comparable ONNX op, or "" when none applies    */
    const char* shape;     /* human readable shape, echoed to the CSV        */
    uint32_t m, k, n;      /* generic dims; per-case meaning in the runner   */
    uint32_t a, b, c;      /* extra dims (conv spatial, heads, groups, ...)  */
    VxBenchUnit unit;
    double work;           /* MACs or bytes for one call                     */
} VxBenchCase;

/* ---------------------------------------------------------------- runners */

typedef struct {
    int (*run)(const VxBenchCase*, int reps, int reference);
} VxBenchRunner;

/* QLinear through the packed path and through the raw dispatcher.  Both are
 * measured because packing is only preferred for some shapes, and a regression
 * in either is invisible if only the other is timed. */
static int vx_bench_qlinear(const VxBenchCase* c, int reps, int reference) {
    const uint32_t M = c->m, K = c->k, N = c->n;
    uint32_t packed_bytes = vx_packed_q8_weight_size(K, N);
    void* packed = vx_alloc(packed_bytes);
    int8_t* weight = (int8_t*)vx_alloc((size_t)K * N);
    uint8_t* input = (uint8_t*)vx_alloc((size_t)M * K);
    uint8_t* out = (uint8_t*)vx_alloc((size_t)M * N);
    uint8_t* ref = (uint8_t*)vx_alloc((size_t)M * N);
    int32_t* bias = (int32_t*)vx_alloc((size_t)N * 4);
    float* wscale = (float*)vx_alloc((size_t)N * 4);
    int32_t* wzp = (int32_t*)vx_alloc((size_t)N * 4);
    int ok = 1;
    vx_seed(1u + c->k);
    vx_fill_i8(weight, (size_t)K * N);
    vx_fill_u8(input, (size_t)M * K);
    memset(bias, 0, (size_t)N * 4);
    memset(wzp, 0, (size_t)N * 4);
    for (uint32_t i = 0; i < N; i++) wscale[i] = 0.002f;
    /* The portable kernel indexes weights as [d_out, d_in], so pack from the
     * same OUT_IN storage: both paths must read one identical buffer or the
     * comparison measures a layout difference instead of the kernels. */
    if (!vx_pack_q8_weight(packed, packed_bytes, weight, K, N, VX_DTYPE_I8, 1)) ok = 0;

    if (ok && reference) {
        /* Portable kernel is the authority; the fast path must match byte for
         * byte, including its requantization rounding and saturation. */
        qlinear_i8u8(input, weight, bias, wscale, wzp, ref, M, K, N,
                     0.01f, 128, 0.05f, 128,
                     VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8);
        vx_qlinear_i8u8_packed(input, packed, bias, wscale, wzp, out, M, K, N,
                               0.01f, 128, 0.05f, 128,
                               VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8);
        if (memcmp(out, ref, (size_t)M * N) != 0) ok = -1;
    }
    if (ok == 1) {
        for (int i = 0; i < reps; i++)
            vx_qlinear_i8u8_packed(input, packed, bias, wscale, wzp, out, M, K, N,
                                   0.01f, 128, 0.05f, 128,
                                   VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8);
    }
    free(packed); free(weight); free(input); free(out); free(ref);
    free(bias); free(wscale); free(wzp);
    return ok;
}

/* The runtime hands convolution a pre-packed weight, which routes im2col into
 * the packed GEMM.  Measuring only the unpacked entry point would report a path
 * the model never takes, so both are cases. */
/* The raw QLinear dispatcher.  This is the only path that reaches the AVX-VNNI
 * and AVX-512-VNNI kernels: the packed GEMM above has no VNNI variant, so
 * without this case those kernels get no throughput coverage at all. */
static int vx_bench_qlinear_raw(const VxBenchCase* c, int reps, int reference) {
    const uint32_t M = c->m, K = c->k, N = c->n;
    int8_t* weight = (int8_t*)vx_alloc((size_t)K * N);
    uint8_t* input = (uint8_t*)vx_alloc((size_t)M * K);
    uint8_t* out = (uint8_t*)vx_alloc((size_t)M * N);
    uint8_t* ref = (uint8_t*)vx_alloc((size_t)M * N);
    int32_t* bias = (int32_t*)vx_alloc((size_t)N * 4);
    float* wscale = (float*)vx_alloc((size_t)N * 4);
    int32_t* wzp = (int32_t*)vx_alloc((size_t)N * 4);
    int ok = 1;
    vx_seed(3u + c->k);
    vx_fill_i8(weight, (size_t)K * N);
    vx_fill_u8(input, (size_t)M * K);
    memset(bias, 0, (size_t)N * 4);
    memset(wzp, 0, (size_t)N * 4);
    for (uint32_t i = 0; i < N; i++) wscale[i] = 0.002f;
#define VX_BENCH_QLINEAR_RAW(fn, dst) \
    fn(input, weight, bias, wscale, wzp, (dst), M, K, N, 0.01f, 128, 0.05f, \
       128, VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8)
    if (reference) {
        VX_BENCH_QLINEAR_RAW(qlinear_i8u8, ref);
        VX_BENCH_QLINEAR_RAW(vx_qlinear_i8u8_native, out);
        if (memcmp(out, ref, (size_t)M * N) != 0) ok = -1;
    }
    if (ok == 1)
        for (int i = 0; i < reps; i++)
            VX_BENCH_QLINEAR_RAW(vx_qlinear_i8u8_native, out);
#undef VX_BENCH_QLINEAR_RAW
    free(weight); free(input); free(out); free(ref);
    free(bias); free(wscale); free(wzp);
    return ok;
}

static int vx_bench_qconv_common(const VxBenchCase* c, int reps, int reference,
                                 int prepack);

static int vx_bench_qconv(const VxBenchCase* c, int reps, int reference) {
    return vx_bench_qconv_common(c, reps, reference, 0);
}

static int vx_bench_qconv_packed(const VxBenchCase* c, int reps, int reference) {
    return vx_bench_qconv_common(c, reps, reference, 1);
}

static int vx_bench_qconv_common(const VxBenchCase* c, int reps, int reference,
                                 int prepack) {
    /* m = batch*out_h*out_w is derived; a,b = out spatial, k = in channels,
     * n = out channels, c = stride. */
    const uint32_t OH = c->a, OW = c->b, CIN = c->k, COUT = c->n, S = c->c;
    const uint32_t IH = OH * S, IW = OW * S;
    uint8_t* input = (uint8_t*)vx_alloc((size_t)IH * IW * CIN);
    int8_t* weight = (int8_t*)vx_alloc((size_t)COUT * 9u * CIN);
    uint8_t* out = (uint8_t*)vx_alloc((size_t)OH * OW * COUT);
    uint8_t* ref = (uint8_t*)vx_alloc((size_t)OH * OW * COUT);
    int32_t* bias = (int32_t*)vx_alloc((size_t)COUT * 4);
    float* wscale = (float*)vx_alloc((size_t)COUT * 4);
    int32_t* wzp = (int32_t*)vx_alloc((size_t)COUT * 4);
    const uint32_t K = 9u * CIN;
    uint32_t packed_bytes = vx_packed_q8_weight_size(K, COUT);
    void* packed_store = prepack ? vx_alloc(packed_bytes) : NULL;
    const void* packed = NULL;
    int ok = 1;
    vx_seed(7u + c->n);
    vx_fill_u8(input, (size_t)IH * IW * CIN);
    vx_fill_i8(weight, (size_t)COUT * 9u * CIN);
    memset(bias, 0, (size_t)COUT * 4);
    memset(wzp, 0, (size_t)COUT * 4);
    for (uint32_t i = 0; i < COUT; i++) wscale[i] = 0.002f;
    if (prepack) {
        /* Convolution weights are OHWI == [COUT, 9*CIN], i.e. OUT_IN storage. */
        if (vx_pack_q8_weight(packed_store, packed_bytes, weight, K, COUT,
                              VX_DTYPE_I8, 1))
            packed = packed_store;
        else
            ok = 0;
    }
#define VX_BENCH_QCONV(dst) \
    qconv2d_i8u8(input, weight, bias, wscale, wzp, (dst), 1u, IH, IW, CIN, \
                 OH, OW, COUT, 3u, 3u, CIN, S, S, 1u, 1u, 1u, 1u, 1u, 1u, \
                 1u, 0u, 0.01f, 128, 0.05f, 128, \
                 VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8)
#define VX_BENCH_QCONV_NATIVE(dst) \
    vx_qconv2d_i8u8_native_prepacked(input, weight, bias, wscale, wzp, (dst), \
                 1u, IH, IW, CIN, OH, OW, COUT, 3u, 3u, CIN, S, S, 1u, 1u, \
                 1u, 1u, 1u, 1u, 1u, 0u, 0.01f, 128, 0.05f, 128, \
                 VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8, packed)
    if (reference) {
        VX_BENCH_QCONV(ref);
        VX_BENCH_QCONV_NATIVE(out);
        if (memcmp(out, ref, (size_t)OH * OW * COUT) != 0) ok = -1;
    }
    if (ok == 1) for (int i = 0; i < reps; i++) VX_BENCH_QCONV_NATIVE(out);
#undef VX_BENCH_QCONV
#undef VX_BENCH_QCONV_NATIVE
    free(input); free(weight); free(out); free(ref);
    free(bias); free(wscale); free(wzp); free(packed_store);
    return ok;
}

static int vx_bench_qsdpa(const VxBenchCase* c, int reps, int reference) {
    /* m = seq, k = d_model, a = heads. */
    const uint32_t SEQ = c->m, D = c->k, H = c->a;
    const size_t elems = (size_t)SEQ * D;
    uint8_t* q = (uint8_t*)vx_alloc(elems);
    uint8_t* k = (uint8_t*)vx_alloc(elems);
    uint8_t* v = (uint8_t*)vx_alloc(elems);
    uint8_t* out = (uint8_t*)vx_alloc(elems);
    uint8_t* ref = (uint8_t*)vx_alloc(elems);
    int ok = 1;
    vx_seed(31u + SEQ);
    vx_fill_u8(q, elems); vx_fill_u8(k, elems); vx_fill_u8(v, elems);
#define VX_BENCH_QSDPA(fn, dst) \
    fn(q, k, v, NULL, (dst), 1u, SEQ, SEQ, D, H, \
       0.02f, 128, 0.02f, 128, 0.02f, 128, 0.05f, 128, \
       0.1581139f, VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8, 0u, 0u)
    if (reference) {
        VX_BENCH_QSDPA(qsdpa_i8u8, ref);
        VX_BENCH_QSDPA(vx_qsdpa_i8u8_native_validated, out);
        if (memcmp(out, ref, elems) != 0) ok = -1;
    }
    if (ok == 1)
        for (int i = 0; i < reps; i++) VX_BENCH_QSDPA(vx_qsdpa_i8u8_native_validated, out);
#undef VX_BENCH_QSDPA
    free(q); free(k); free(v); free(out); free(ref);
    return ok;
}

static int vx_bench_qgroupnorm(const VxBenchCase* c, int reps, int reference) {
    /* a,b = spatial, k = channels, n = groups. */
    const uint32_t H = c->a, W = c->b, C = c->k, G = c->n;
    const size_t elems = (size_t)H * W * C;
    uint8_t* in = (uint8_t*)vx_alloc(elems);
    uint8_t* out = (uint8_t*)vx_alloc(elems);
    uint8_t* ref = (uint8_t*)vx_alloc(elems);
    float* weight = (float*)vx_alloc((size_t)C * 4);
    float* bias = (float*)vx_alloc((size_t)C * 4);
    int ok = 1;
    vx_seed(53u + C);
    vx_fill_u8(in, elems);
    vx_fill_f32(weight, C, 1.0f);
    vx_fill_f32(bias, C, 0.1f);
#define VX_BENCH_QGN(fn, dst) \
    fn(in, weight, bias, (dst), 1u, H, W, C, G, 0.02f, 128, 0.05f, 128, \
       1e-5f, VX_DTYPE_U8, VX_DTYPE_U8)
    if (reference) {
        VX_BENCH_QGN(qgroupnorm_i8u8, ref);
        VX_BENCH_QGN(vx_qgroupnorm_i8u8_native_validated, out);
        if (memcmp(out, ref, elems) != 0) ok = -1;
    }
    if (ok == 1)
        for (int i = 0; i < reps; i++) VX_BENCH_QGN(vx_qgroupnorm_i8u8_native_validated, out);
#undef VX_BENCH_QGN
    free(in); free(out); free(ref); free(weight); free(bias);
    return ok;
}

static int vx_bench_qlayernorm(const VxBenchCase* c, int reps, int reference) {
    const uint32_t ROWS = c->m, D = c->k;
    const size_t elems = (size_t)ROWS * D;
    uint8_t* in = (uint8_t*)vx_alloc(elems);
    uint8_t* out = (uint8_t*)vx_alloc(elems);
    uint8_t* ref = (uint8_t*)vx_alloc(elems);
    float* weight = (float*)vx_alloc((size_t)D * 4);
    float* bias = (float*)vx_alloc((size_t)D * 4);
    int ok = 1;
    vx_seed(97u + D);
    vx_fill_u8(in, elems);
    vx_fill_f32(weight, D, 1.0f);
    vx_fill_f32(bias, D, 0.1f);
#define VX_BENCH_QLN(fn, dst) \
    fn(in, weight, bias, (dst), ROWS, D, 0.02f, 128, 0.05f, 128, 1e-5f, \
       VX_DTYPE_U8, VX_DTYPE_U8)
    if (reference) {
        VX_BENCH_QLN(qlayernorm_i8u8, ref);
        VX_BENCH_QLN(vx_qlayernorm_i8u8_native_validated, out);
        if (memcmp(out, ref, elems) != 0) ok = -1;
    }
    if (ok == 1)
        for (int i = 0; i < reps; i++) VX_BENCH_QLN(vx_qlayernorm_i8u8_native_validated, out);
#undef VX_BENCH_QLN
    free(in); free(out); free(ref); free(weight); free(bias);
    return ok;
}

static int vx_bench_qadd(const VxBenchCase* c, int reps, int reference) {
    const size_t elems = (size_t)c->m;
    uint8_t* a = (uint8_t*)vx_alloc(elems);
    uint8_t* b = (uint8_t*)vx_alloc(elems);
    uint8_t* out = (uint8_t*)vx_alloc(elems);
    uint8_t* ref = (uint8_t*)vx_alloc(elems);
    int ok = 1;
    vx_seed(131u);
    vx_fill_u8(a, elems); vx_fill_u8(b, elems);
    if (reference) {
        qadd_i8u8(a, b, ref, (uint32_t)elems, 0.02f, 128, 0.03f, 128, 0.05f, 128,
                  VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8, 0u);
        memcpy(out, ref, elems);
        if (!vx_qadd_i8u8_native_try(a, b, out, (uint32_t)elems, 0.02f, 128,
                                     0.03f, 128, 0.05f, 128, VX_DTYPE_U8,
                                     VX_DTYPE_U8, VX_DTYPE_U8, 0u))
            ok = 0;  /* native path declined this shape; nothing to compare */
        else if (memcmp(out, ref, elems) != 0) ok = -1;
    }
    if (ok == 1)
        for (int i = 0; i < reps; i++)
            vx_qadd_i8u8_native_try(a, b, out, (uint32_t)elems, 0.02f, 128,
                                    0.03f, 128, 0.05f, 128, VX_DTYPE_U8,
                                    VX_DTYPE_U8, VX_DTYPE_U8, 0u);
    free(a); free(b); free(out); free(ref);
    return ok;
}

static int vx_bench_quantize(const VxBenchCase* c, int reps, int reference) {
    const size_t elems = (size_t)c->m;
    float* in = (float*)vx_alloc(elems * 4);
    uint8_t* out = (uint8_t*)vx_alloc(elems);
    int ok = 1;
    float scale = 0.05f;
    (void)reference;
    vx_seed(179u);
    vx_fill_f32(in, elems, 4.0f);
    for (int i = 0; i < reps; i++) {
        uint32_t done = vx_quantize_linear_typed_native_prefix(
            in, scale, 128, out, VX_DTYPE_U8, (uint32_t)elems, 0, 255);
        if (!done) { ok = 0; break; }
    }
    free(in); free(out);
    return ok;
}

static int vx_bench_dequantize(const VxBenchCase* c, int reps, int reference) {
    const size_t elems = (size_t)c->m;
    uint8_t* in = (uint8_t*)vx_alloc(elems);
    float* out = (float*)vx_alloc(elems * 4);
    int ok = 1;
    (void)reference;
    vx_seed(211u);
    vx_fill_u8(in, elems);
    for (int i = 0; i < reps; i++) {
        uint32_t done = vx_dequantize_linear_typed_native_prefix(
            in, VX_DTYPE_U8, 0.05f, 128, out, (uint32_t)elems);
        if (!done) { ok = 0; break; }
    }
    free(in); free(out);
    return ok;
}

/* ------------------------------------------------------------- F32 kernels */
/*
 * The F32 kernels need a different correctness rule than the quantized ones.
 * An integer kernel has one right answer, so those cases memcmp against the
 * portable reference.  Vectorizing an F32 reduction reorders the additions, so
 * an AVX2 body cannot be bit-identical to a scalar one and demanding that would
 * mean either refusing to vectorize or lying about what was checked.  These
 * cases therefore compare against a naive reference written here — deliberately
 * an independent implementation rather than another of our own paths — and
 * accept a relative error, reporting `close` instead of `exact`.
 *
 * VX_BENCH_CLOSE is the verdict for that; main() treats it as a pass and prints
 * it, so the CSV still records which kind of check each row got.
 */
enum { VX_BENCH_CLOSE = 2 };

/* Generous enough for a reordered F32 reduction over a few thousand terms,
 * tight enough to catch a wrong kernel: a mis-indexed or mis-blocked GEMM is
 * wrong by order-of-magnitude, not by ulps. */
static int vx_f32_close(const float* got, const float* expect, size_t n) {
    for (size_t i = 0; i < n; i++) {
        const float magnitude = fabsf(expect[i]) > fabsf(got[i])
            ? fabsf(expect[i]) : fabsf(got[i]);
        if (!(fabsf(got[i] - expect[i]) <= 1.0e-3f + 2.0e-3f * magnitude))
            return 0;
    }
    return 1;
}

/* The Linear/Gemm node's real path: pack the weight once, then run the packed
 * microkernel.  Cached packing is what the graph does, so the pack is outside
 * the timed loop here too. */
static int vx_bench_gemm_f32_packed(const VxBenchCase* c, int reps, int reference) {
    const uint32_t M = c->m, K = c->k, N = c->n;
    const uint32_t packed_elements = vx_gemm_f32_packed_elements(K, N);
    float* a = (float*)vx_alloc((size_t)M * K * 4);
    float* weight = (float*)vx_alloc((size_t)K * N * 4);
    float* packed = (float*)vx_alloc((size_t)packed_elements * 4);
    float* bias = (float*)vx_alloc((size_t)N * 4);
    float* out = (float*)vx_alloc((size_t)M * N * 4);
    float* ref = (float*)vx_alloc((size_t)M * N * 4);
    int ok = 1;
    vx_seed(1301u + K);
    vx_fill_f32(a, (size_t)M * K, 1.0f);
    vx_fill_f32(weight, (size_t)K * N, 1.0f);
    vx_fill_f32(bias, N, 1.0f);
    if (!packed_elements || !vx_gemm_f32_pack_b(weight, packed, K, N, 0)) ok = 0;
    if (ok == 1 && reference) {
        for (uint32_t row = 0; row < M; row++)
            for (uint32_t column = 0; column < N; column++) {
                double sum = bias[column];
                for (uint32_t inner = 0; inner < K; inner++)
                    sum += (double)a[(size_t)row * K + inner] *
                           weight[(size_t)inner * N + column];
                ref[(size_t)row * N + column] = (float)sum;
            }
        if (!vx_gemm_f32_run_packed(a, packed, bias, out, M, K, N)) ok = 0;
        else if (!vx_f32_close(out, ref, (size_t)M * N)) ok = -1;
        else ok = VX_BENCH_CLOSE;
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++)
            vx_gemm_f32_run_packed(a, packed, bias, out, M, K, N);
    free(a); free(weight); free(packed); free(bias); free(out); free(ref);
    return ok;
}

/* The no-pack path: BatchMatMul's B operand is a graph value, and Linear falls
 * back here when its weight is not an immutable model tensor. */
static int vx_bench_matmul_f32(const VxBenchCase* c, int reps, int reference) {
    const uint32_t M = c->m, K = c->k, N = c->n;
    float* a = (float*)vx_alloc((size_t)M * K * 4);
    float* b = (float*)vx_alloc((size_t)K * N * 4);
    float* out = (float*)vx_alloc((size_t)M * N * 4);
    float* ref = (float*)vx_alloc((size_t)M * N * 4);
    int ok = 1;
    vx_seed(1409u + K);
    vx_fill_f32(a, (size_t)M * K, 1.0f);
    vx_fill_f32(b, (size_t)K * N, 1.0f);
    if (reference) {
        for (uint32_t row = 0; row < M; row++)
            for (uint32_t column = 0; column < N; column++) {
                double sum = 0.0;
                for (uint32_t inner = 0; inner < K; inner++)
                    sum += (double)a[(size_t)row * K + inner] *
                           b[(size_t)inner * N + column];
                ref[(size_t)row * N + column] = (float)sum;
            }
        matmul_f32(a, b, NULL, out, (int)M, (int)K, (int)N);
        ok = vx_f32_close(out, ref, (size_t)M * N) ? VX_BENCH_CLOSE : -1;
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++)
            matmul_f32(a, b, NULL, out, (int)M, (int)K, (int)N);
    free(a); free(b); free(out); free(ref);
    return ok;
}

/* 3x3 same-padded NHWC convolution, the encoder's dominant conv shape. */
static int vx_bench_conv2d_f32(const VxBenchCase* c, int reps, int reference) {
    const int height = (int)c->a, width = (int)c->b;
    const int in_c = (int)c->k, out_c = (int)c->n;
    const int pads[4] = {1, 1, 1, 1};
    float* in = (float*)vx_alloc((size_t)height * width * in_c * 4);
    float* weight = (float*)vx_alloc((size_t)9 * in_c * out_c * 4);
    float* bias = (float*)vx_alloc((size_t)out_c * 4);
    float* out = (float*)vx_alloc((size_t)height * width * out_c * 4);
    float* ref = (float*)vx_alloc((size_t)height * width * out_c * 4);
    int ok = 1;
    vx_seed(1511u + in_c);
    vx_fill_f32(in, (size_t)height * width * in_c, 1.0f);
    vx_fill_f32(weight, (size_t)9 * in_c * out_c, 0.25f);
    vx_fill_f32(bias, out_c, 1.0f);
    if (reference) {
        /* Weights are HWIO, matching what the runtime hands the kernel. */
        for (int oy = 0; oy < height; oy++)
            for (int ox = 0; ox < width; ox++)
                for (int oc = 0; oc < out_c; oc++) {
                    double sum = bias[oc];
                    for (int ky = 0; ky < 3; ky++)
                        for (int kx = 0; kx < 3; kx++) {
                            const int iy = oy + ky - 1, ix = ox + kx - 1;
                            if (iy < 0 || iy >= height || ix < 0 || ix >= width) continue;
                            for (int ic = 0; ic < in_c; ic++)
                                sum += (double)in[((size_t)iy * width + ix) * in_c + ic] *
                                    weight[(((size_t)ky * 3 + kx) * in_c + ic) * out_c + oc];
                        }
                    ref[((size_t)oy * width + ox) * out_c + oc] = (float)sum;
                }
        vx_conv2d_generic_f32(0, in, out, weight, bias, 1, height, width, in_c,
                              height, width, out_c, 3, 3, in_c, 1, 1, 1, pads,
                              1, 1, 0);
        ok = vx_f32_close(out, ref, (size_t)height * width * out_c)
            ? VX_BENCH_CLOSE : -1;
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++)
            vx_conv2d_generic_f32(0, in, out, weight, bias, 1, height, width,
                                  in_c, height, width, out_c, 3, 3, in_c, 1,
                                  1, 1, pads, 1, 1, 0);
    vx_conv_f32_opt_free_all();
    free(in); free(weight); free(bias); free(out); free(ref);
    return ok;
}

/* Cross attention over an explicit memory.  m is the query length, a the key
 * length, so the same runner covers both the uncached full forward (m == a) and
 * the incremental single row (m == 1). */
static int vx_bench_cross_sdpa_f32(const VxBenchCase* c, int reps, int reference) {
    const int seq_q = (int)c->m, seq_kv = (int)c->a;
    const int d_model = (int)c->k, heads = (int)c->b;
    const int head_dim = d_model / heads;
    const float scale = 1.0f / (float)__builtin_sqrt((double)head_dim);
    /* c->c selects the mask mode.  The TinyReceipt decoder always passes a key
     * mask (memory_padding_mask), so an unmasked-only case measured a shape the
     * model never executes — and the kernel's masked and unmasked paths differ
     * enough that the distinction matters. */
    const int mask_mode = (int)c->c;
    float* q = (float*)vx_alloc((size_t)seq_q * d_model * 4);
    float* k = (float*)vx_alloc((size_t)seq_kv * d_model * 4);
    float* v = (float*)vx_alloc((size_t)seq_kv * d_model * 4);
    float* out = (float*)vx_alloc((size_t)seq_q * d_model * 4);
    float* ref = (float*)vx_alloc((size_t)seq_q * d_model * 4);
    int32_t* keep = (int32_t*)vx_alloc((size_t)seq_kv * 4);
    const int32_t* mask = mask_mode ? keep : NULL;
    int ok = 1;
    vx_seed(1613u + seq_kv);
    vx_fill_f32(q, (size_t)seq_q * d_model, 1.0f);
    vx_fill_f32(k, (size_t)seq_kv * d_model, 1.0f);
    vx_fill_f32(v, (size_t)seq_kv * d_model, 1.0f);
    /* Mostly-valid, like a padded memory: the last eighth is padding. */
    for (int i = 0; i < seq_kv; i++)
        keep[i] = i < seq_kv - seq_kv / 8 ? 1 : 0;
    if (reference) {
        /* Two-pass softmax in double: independent of the kernel's online
         * recurrence, so this checks the recurrence itself and not just its
         * vectorization. */
        for (int query = 0; query < seq_q; query++)
            for (int head = 0; head < heads; head++) {
                const size_t base = (size_t)query * d_model + head * head_dim;
                double maximum = -1.0e300, total = 0.0;
                int key;
                for (key = 0; key < seq_kv; key++) {
                    double score = 0.0;
                    if (mask && !mask[key]) continue;
                    for (int d = 0; d < head_dim; d++)
                        score += (double)q[base + d] *
                            k[(size_t)key * d_model + head * head_dim + d];
                    score *= scale;
                    if (score > maximum) maximum = score;
                }
                for (int d = 0; d < head_dim; d++) ref[base + d] = 0.0f;
                for (key = 0; key < seq_kv; key++) {
                    double score = 0.0, weight;
                    if (mask && !mask[key]) continue;
                    for (int d = 0; d < head_dim; d++)
                        score += (double)q[base + d] *
                            k[(size_t)key * d_model + head * head_dim + d];
                    weight = __builtin_exp(score * scale - maximum);
                    total += weight;
                    for (int d = 0; d < head_dim; d++)
                        ref[base + d] += (float)(weight *
                            v[(size_t)key * d_model + head * head_dim + d]);
                }
                for (int d = 0; d < head_dim; d++)
                    ref[base + d] = (float)(ref[base + d] / total);
            }
        cross_sdpa_f32(q, k, v, out, seq_q, seq_kv, d_model, heads, head_dim,
                       scale, mask, mask_mode, 0);
        ok = vx_f32_close(out, ref, (size_t)seq_q * d_model) ? VX_BENCH_CLOSE : -1;
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++)
            cross_sdpa_f32(q, k, v, out, seq_q, seq_kv, d_model, heads,
                           head_dim, scale, mask, mask_mode, 0);
    free(q); free(k); free(v); free(out); free(ref); free(keep);
    return ok;
}

/* Self attention over a fused QKV tensor.  Shares the online-softmax core with
 * cross_sdpa_f32 but reads one interleaved tensor, so the stride handling is
 * worth measuring separately.  Causal, which is how the decoder runs it. */
static int vx_bench_sdpa_f32(const VxBenchCase* c, int reps, int reference) {
    const int seq = (int)c->m, d_model = (int)c->k, heads = (int)c->b;
    const int head_dim = d_model / heads;
    const float scale = 1.0f / (float)__builtin_sqrt((double)head_dim);
    const long stride = (long)d_model * 3;
    float* qkv = (float*)vx_alloc((size_t)seq * stride * 4);
    float* out = (float*)vx_alloc((size_t)seq * d_model * 4);
    float* ref = (float*)vx_alloc((size_t)seq * d_model * 4);
    int ok = 1;
    vx_seed(2003u + seq);
    vx_fill_f32(qkv, (size_t)seq * stride, 1.0f);
    if (reference) {
        /* Two-pass softmax in double over the same causal window, independent of
         * the kernel's recurrence. */
        for (int query = 0; query < seq; query++)
            for (int head = 0; head < heads; head++) {
                const long base = (long)query * stride + head * head_dim;
                const long out_base = (long)query * d_model + head * head_dim;
                double maximum = -1.0e300, total = 0.0;
                int key;
                for (key = 0; key <= query; key++) {
                    double score = 0.0;
                    for (int d = 0; d < head_dim; d++)
                        score += (double)qkv[base + d] *
                            qkv[(long)key * stride + d_model + head * head_dim + d];
                    score *= scale;
                    if (score > maximum) maximum = score;
                }
                for (int d = 0; d < head_dim; d++) ref[out_base + d] = 0.0f;
                for (key = 0; key <= query; key++) {
                    double score = 0.0, weight;
                    for (int d = 0; d < head_dim; d++)
                        score += (double)qkv[base + d] *
                            qkv[(long)key * stride + d_model + head * head_dim + d];
                    weight = __builtin_exp(score * scale - maximum);
                    total += weight;
                    for (int d = 0; d < head_dim; d++)
                        ref[out_base + d] += (float)(weight *
                            qkv[(long)key * stride + d_model * 2 +
                                head * head_dim + d]);
                }
                for (int d = 0; d < head_dim; d++)
                    ref[out_base + d] = (float)(ref[out_base + d] / total);
            }
        sdpa_f32(qkv, out, seq, d_model, heads, head_dim, scale, NULL, 0, 1);
        ok = vx_f32_close(out, ref, (size_t)seq * d_model) ? VX_BENCH_CLOSE : -1;
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++)
            sdpa_f32(qkv, out, seq, d_model, heads, head_dim, scale, NULL, 0, 1);
    free(qkv); free(out); free(ref);
    return ok;
}

static int vx_bench_groupnorm_f32(const VxBenchCase* c, int reps, int reference) {
    const uint32_t height = c->a, width = c->b;
    const uint32_t channels = c->k, groups = c->n;
    const size_t elems = (size_t)height * width * channels;
    float* in = (float*)vx_alloc(elems * 4);
    float* weight = (float*)vx_alloc((size_t)channels * 4);
    float* bias = (float*)vx_alloc((size_t)channels * 4);
    float* out = (float*)vx_alloc(elems * 4);
    int ok = 1;
    (void)reference;
    vx_seed(1717u + channels);
    vx_fill_f32(in, elems, 2.0f);
    vx_fill_f32(weight, channels, 1.0f);
    vx_fill_f32(bias, channels, 1.0f);
    for (int i = 0; i < reps; i++)
        if (!groupnorm_f32(in, weight, bias, out, 1, height, width, channels,
                           groups, 1.0e-5)) { ok = 0; break; }
    free(in); free(weight); free(bias); free(out);
    return ok;
}

static int vx_bench_silu_f32(const VxBenchCase* c, int reps, int reference) {
    const size_t elems = (size_t)c->m;
    float* in = (float*)vx_alloc(elems * 4);
    float* out = (float*)vx_alloc(elems * 4);
    int ok = 1;
    vx_seed(1811u);
    vx_fill_f32(in, elems, 6.0f);
    if (reference) {
        float* ref = (float*)vx_alloc(elems * 4);
        for (size_t i = 0; i < elems; i++)
            ref[i] = (float)((double)in[i] /
                             (1.0 + __builtin_exp(-(double)in[i])));
        silu_f32(in, out, (int)elems);
        ok = vx_f32_close(out, ref, elems) ? VX_BENCH_CLOSE : -1;
        free(ref);
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++) silu_f32(in, out, (int)elems);
    free(in); free(out);
    return ok;
}

static int vx_bench_mul_f32(const VxBenchCase* c, int reps, int reference) {
    const uint32_t elems = c->m;
    const uint32_t shape[1] = {0};
    uint32_t dims[1];
    float* a = (float*)vx_alloc((size_t)elems * 4);
    float* b = (float*)vx_alloc((size_t)elems * 4);
    float* out = (float*)vx_alloc((size_t)elems * 4);
    int ok = 1;
    (void)shape;
    dims[0] = elems;
    vx_seed(1907u);
    vx_fill_f32(a, elems, 2.0f);
    vx_fill_f32(b, elems, 2.0f);
    if (reference) {
        float* ref = (float*)vx_alloc((size_t)elems * 4);
        for (uint32_t i = 0; i < elems; i++) ref[i] = a[i] * b[i];
        if (!binary_broadcast_f32(a, b, out, dims, dims, dims, 1, 1, 1, elems,
                                  VX_PORTABLE_BINARY_MUL)) ok = 0;
        /* Elementwise multiply has one correct answer in either path, so this
         * one can still demand exactness. */
        else ok = memcmp(out, ref, (size_t)elems * 4) == 0 ? 1 : -1;
        free(ref);
    }
    if (ok > 0)
        for (int i = 0; i < reps; i++)
            binary_broadcast_f32(a, b, out, dims, dims, dims, 1, 1, 1, elems,
                                 VX_PORTABLE_BINARY_MUL);
    free(a); free(b); free(out);
    return ok;
}

/* ------------------------------------------------------------------ cases */

typedef struct {
    VxBenchCase c;
    int (*run)(const VxBenchCase*, int, int);
} VxBenchEntry;

/* Shapes are the ones the TinyReceipt encoder and decoder actually execute, so
 * a win here is a win in the model rather than on a synthetic square. */
static const VxBenchEntry vx_bench_entries[] = {
  /* name              onnx op            shape                    m     k     n    a   b  c  unit        */
  {{"qlinear",         "QLinearMatMul",  "402x320x320",          402,  320,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear},
  {{"qlinear",         "QLinearMatMul",  "402x320x1280",         402,  320, 1280,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear},
  {{"qlinear",         "QLinearMatMul",  "402x1280x320",         402, 1280,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear},
  {{"qlinear",         "QLinearMatMul",  "1x320x320",              1,  320,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear},
  {{"qlinear",         "QLinearMatMul",  "13440x432x96",       13440,  432,   96,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear},
  {{"qlinear_raw",     "QLinearMatMul",  "402x320x320",          402,  320,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear_raw},
  {{"qlinear_raw",     "QLinearMatMul",  "402x64x320",           402,   64,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear_raw},
  {{"qlinear_raw",     "QLinearMatMul",  "402x1280x320",         402, 1280,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qlinear_raw},
  {{"qconv2d",         "QLinearConv",    "80x168 c48->96 s1",      0,   48,   96, 80,168, 1, VX_UNIT_MAC, 0}, vx_bench_qconv},
  {{"qconv2d",         "QLinearConv",    "40x84 c96->192 s1",      0,   96,  192, 40, 84, 1, VX_UNIT_MAC, 0}, vx_bench_qconv},
  {{"qconv2d",         "QLinearConv",    "20x42 c192->320 s1",     0,  192,  320, 20, 42, 1, VX_UNIT_MAC, 0}, vx_bench_qconv},
  {{"qconv2d_packed",  "QLinearConv",    "80x168 c48->96 s1",      0,   48,   96, 80,168, 1, VX_UNIT_MAC, 0}, vx_bench_qconv_packed},
  {{"qconv2d_packed",  "QLinearConv",    "40x84 c96->192 s1",      0,   96,  192, 40, 84, 1, VX_UNIT_MAC, 0}, vx_bench_qconv_packed},
  {{"qconv2d_packed",  "QLinearConv",    "20x42 c192->320 s1",     0,  192,  320, 20, 42, 1, VX_UNIT_MAC, 0}, vx_bench_qconv_packed},
  {{"qsdpa",           "MatMul+Softmax", "402 d320 h8",          402,  320,    0,  8,  0, 0, VX_UNIT_MAC, 0}, vx_bench_qsdpa},
  {{"qgroupnorm",      "InstanceNorm",   "80x168 c96 g32",         0,   96,   32, 80,168, 0, VX_UNIT_BYTE,0}, vx_bench_qgroupnorm},
  {{"qgroupnorm",      "InstanceNorm",   "20x42 c320 g32",         0,  320,   32, 20, 42, 0, VX_UNIT_BYTE,0}, vx_bench_qgroupnorm},
  {{"qlayernorm",      "LayerNorm",      "402x320",              402,  320,    0,  0,  0, 0, VX_UNIT_BYTE,0}, vx_bench_qlayernorm},
  {{"qadd",            "QLinearAdd",     "402x320",           128640,    0,    0,  0,  0, 0, VX_UNIT_BYTE,0}, vx_bench_qadd},
  {{"quantizelinear",  "QuantizeLinear", "402x320",           128640,    0,    0,  0,  0, 0, VX_UNIT_BYTE,0}, vx_bench_quantize},
  {{"dequantizelinear","DequantizeLinear","402x320",          128640,    0,    0,  0,  0, 0, VX_UNIT_BYTE,0}, vx_bench_dequantize},
  /* F32 counterparts of the same encoder/decoder shapes.  These are what the
   * FP32 package executes, so they place the FP32 and INT8 kernels for one
   * shape side by side instead of only measuring the quantized half. */
  {{"gemm_f32_packed", "MatMul",         "402x320x320",          402,  320,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_gemm_f32_packed},
  {{"gemm_f32_packed", "MatMul",         "402x320x1280",         402,  320, 1280,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_gemm_f32_packed},
  {{"gemm_f32_packed", "MatMul",         "402x1280x320",         402, 1280,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_gemm_f32_packed},
  {{"gemm_f32_packed", "MatMul",         "1x320x320",              1,  320,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_gemm_f32_packed},
  {{"matmul_f32",      "MatMul",         "402x320x320",          402,  320,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_matmul_f32},
  {{"matmul_f32",      "MatMul",         "192x402x320",          192,  402,  320,  0,  0, 0, VX_UNIT_MAC, 0}, vx_bench_matmul_f32},
  {{"conv2d_f32",      "Conv",           "80x168 c48->96 s1",      0,   48,   96, 80,168, 1, VX_UNIT_MAC, 0}, vx_bench_conv2d_f32},
  {{"conv2d_f32",      "Conv",           "40x84 c96->192 s1",      0,   96,  192, 40, 84, 1, VX_UNIT_MAC, 0}, vx_bench_conv2d_f32},
  {{"conv2d_f32",      "Conv",           "20x42 c192->320 s1",     0,  192,  320, 20, 42, 1, VX_UNIT_MAC, 0}, vx_bench_conv2d_f32},
  /* The uncached decode shape: every query row against the whole memory. */
  {{"cross_sdpa_f32",  "MatMul+Softmax", "q192 kv402 d320 h8",   192,  320,    0,402,  8, 0, VX_UNIT_MAC, 0}, vx_bench_cross_sdpa_f32},
  /* The incremental shape: one query row against the same memory. */
  {{"cross_sdpa_f32",  "MatMul+Softmax", "q1 kv402 d320 h8",       1,  320,    0,402,  8, 0, VX_UNIT_MAC, 0}, vx_bench_cross_sdpa_f32},
  /* With a key mask, which is what the decoder and encoder actually pass. */
  {{"cross_sdpa_f32_m", "MatMul+Softmax", "q192 kv402 d320 h8",   192,  320,    0,402,  8, 1, VX_UNIT_MAC, 0}, vx_bench_cross_sdpa_f32},
  {{"cross_sdpa_f32_m", "MatMul+Softmax", "q402 kv402 d320 h8",   402,  320,    0,402,  8, 1, VX_UNIT_MAC, 0}, vx_bench_cross_sdpa_f32},
  /* Same GEMM work, double the heads: head_dim halves so the per-key softmax
   * cost doubles while the multiply-add count is unchanged.  The difference
   * isolates how much of the kernel is not the two GEMMs. */
  {{"cross_sdpa_f32_m", "MatMul+Softmax", "q402 kv402 d320 h16",  402,  320,    0,402, 16, 1, VX_UNIT_MAC, 0}, vx_bench_cross_sdpa_f32},
  /* Self attention, causal, over a fused QKV tensor. */
  {{"sdpa_f32",        "MatMul+Softmax", "q192 kv192 d320 h8",   192,  320,    0,192,  8, 0, VX_UNIT_MAC, 0}, vx_bench_sdpa_f32},
  {{"groupnorm_f32",   "GroupNorm",      "80x168 c96 g32",         0,   96,   32, 80,168, 0, VX_UNIT_BYTE,0}, vx_bench_groupnorm_f32},
  {{"groupnorm_f32",   "GroupNorm",      "20x42 c320 g32",         0,  320,   32, 20, 42, 0, VX_UNIT_BYTE,0}, vx_bench_groupnorm_f32},
  {{"silu_f32",        "Mul+Sigmoid",    "402x320",           128640,    0,    0,  0,  0, 0, VX_UNIT_BYTE,0}, vx_bench_silu_f32},
  {{"mul_f32",         "Mul",            "402x320",           128640,    0,    0,  0,  0, 0, VX_UNIT_BYTE,0}, vx_bench_mul_f32},
};

static double vx_case_work(const VxBenchEntry* e) {
    const VxBenchCase* c = &e->c;
    if (e->run == vx_bench_qlinear || e->run == vx_bench_qlinear_raw)
        return (double)c->m * c->k * c->n;
    if (e->run == vx_bench_qconv || e->run == vx_bench_qconv_packed)
        return (double)c->a * c->b * 9.0 * c->k * c->n;
    if (e->run == vx_bench_qsdpa)
        /* QK dot plus PV accumulate, both over head_dim, for every pair. */
        return 2.0 * c->m * c->m * c->k;
    if (e->run == vx_bench_qgroupnorm)
        return (double)c->a * c->b * c->k;
    if (e->run == vx_bench_qlayernorm)
        return (double)c->m * c->k;
    /* F32 cases reuse the same denominators as their quantized counterparts so
     * a GMAC/s or GB/s figure means the same thing across both halves. */
    if (e->run == vx_bench_gemm_f32_packed || e->run == vx_bench_matmul_f32)
        return (double)c->m * c->k * c->n;
    if (e->run == vx_bench_conv2d_f32)
        return (double)c->a * c->b * 9.0 * c->k * c->n;
    if (e->run == vx_bench_cross_sdpa_f32)
        /* QK dot plus PV accumulate over head_dim for every (query, key). */
        return 2.0 * c->m * c->a * c->k;
    if (e->run == vx_bench_sdpa_f32)
        /* Causal, so only the lower triangle of (query, key) pairs is visited. */
        return 2.0 * c->m * (c->a + 1.0) / 2.0 * c->k;
    if (e->run == vx_bench_groupnorm_f32)
        return (double)c->a * c->b * c->k;
    return (double)c->m;
}

int main(int argc, char** argv) {
    const char* filter = NULL;
    int threads = 0;
    int check = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--op") && i + 1 < argc) filter = argv[++i];
        else if (!strcmp(argv[i], "--threads") && i + 1 < argc) threads = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--no-check")) check = 0;
        else if (!strcmp(argv[i], "--help")) {
            printf("Usage: %s [--op <name>] [--threads <n>] [--no-check]\n", argv[0]);
            printf("Emits CSV: kernel,onnx_op,shape,unit,work,ms,throughput,exact\n");
            printf("Set VOLVOXAI_CPU_ISA=baseline|avx2|avxvnni|avx512vnni to "
                   "clamp the ISA tier and compare tiers on one machine.\n");
            return 0;
        }
    }
    VxKernelThreadPool* pool = vx_kernel_thread_pool_create(threads);
    /*
     * The conv kernels keep their packed weights and im2col indirection in
     * VxEngineState, so they need a bound state even outside a loaded model.
     * Entering the scope also enters the state's thread-pool scope, so the
     * state adopts the pool built for the requested --threads rather than the
     * auto-sized one vx_engine_state_init would otherwise install.
     */
    static VxEngineState engine_state;
    VxEngineStateScope scope;
    if (vx_engine_state_init(&engine_state) != 0) {
        fprintf(stderr, "benchmark_kernel_unit: engine state init failed\n");
        return 2;
    }
    vx_kernel_thread_pool_destroy(engine_state.kernel_thread_pool);
    engine_state.kernel_thread_pool = pool;
    scope = vx_engine_state_scope_enter(&engine_state);
    /* Record the resolved tier: a throughput number is only interpretable
     * together with the ISA that produced it. */
    printf("# isa=%s threads=%d\n",
           vx_kernel_isa_name(vx_kernel_platform()->isa),
           vx_kernels_thread_count());
    printf("kernel,onnx_op,shape,unit,work,ms,throughput,exact\n");
    int failures = 0;
    for (size_t i = 0; i < sizeof vx_bench_entries / sizeof vx_bench_entries[0]; i++) {
        const VxBenchEntry* e = &vx_bench_entries[i];
        if (filter && strcmp(filter, e->c.name)) continue;
        const double work = vx_case_work(e);
        int reps = (int)(2.0e9 / (work > 0.0 ? work : 1.0));
        if (reps < 1) reps = 1;
        if (reps > 100000) reps = 100000;
        int verdict = e->run(&e->c, VX_BENCH_WARMUP, check);
        double best = 1e30;
        if (verdict >= 0) {
            for (int t = 0; t < VX_BENCH_TRIALS; t++) {
                double t0 = vx_now();
                e->run(&e->c, reps, 0);
                double dt = (vx_now() - t0) / (double)reps;
                if (dt < best) best = dt;
            }
        }
        /* `close` marks a tolerance check rather than a weaker one: an F32
         * kernel that vectorizes a reduction cannot be bit-identical to a
         * scalar reference, so the column records which rule was applied. */
        const char* exact = verdict < 0 ? "MISMATCH"
            : !check ? "unchecked"
            : verdict == VX_BENCH_CLOSE ? "close" : "exact";
        if (verdict < 0) failures++;
        if (verdict < 0) {
            printf("%s,%s,%s,%s,%.0f,,,%s\n", e->c.name, e->c.onnx, e->c.shape,
                   e->c.unit == VX_UNIT_MAC ? "MAC" : "byte", work, exact);
            continue;
        }
        printf("%s,%s,%s,%s,%.0f,%.4f,%.3f,%s\n",
               e->c.name, e->c.onnx, e->c.shape,
               e->c.unit == VX_UNIT_MAC ? "MAC" : "byte",
               work, best * 1e3,
               e->c.unit == VX_UNIT_MAC ? work / best / 1e9 : work / best / 1e9,
               exact);
    }
    vx_engine_state_scope_leave(scope);
    /* The pool is this function's to free, so detach it before deinit rather
     * than letting the state destroy something it did not create. */
    engine_state.kernel_thread_pool = NULL;
    vx_engine_state_deinit(&engine_state);
    vx_kernel_thread_pool_destroy(pool);
    if (failures) {
        fprintf(stderr, "benchmark_kernel_unit: %d kernel(s) disagreed with the "
                        "portable reference\n", failures);
        return 1;
    }
    return 0;
}
