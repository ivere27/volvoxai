/*
 * Byte-exact parity of every ISA tier the host can run, for the raw QLinear and
 * QConv dispatchers.
 *
 * test_packed_quant_gemm owns the packed N32 AVX2/VNNI tier walk.  This test
 * drives the distinct raw QLinear and direct QConv dispatchers across shapes
 * that straddle their thresholds and compares them against the portable
 * reference byte for byte.
 *
 * VOLVOXAI_CPU_ISA lets one binary walk every tier the CPU supports, so the same
 * test covers baseline, AVX2, AVX-VNNI and AVX-512-VNNI on whatever host it runs
 * on.  That matters because the VNNI kernels cannot execute at all on a
 * pre-VNNI CPU: on such a host this test still passes, but it silently only
 * proves the tiers that exist.  The printed tier list is therefore part of the
 * result, not decoration.
 */
#include "../src/kernels/inference_kernels.h"
#include "../src/kernels/quant_cpu_opt.h"
#include "../src/kernels/thread_pool.h"
#include "../src/kernels/kernel_platform.h"
#include "../include/volvoxai_enums.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;

static uint32_t g_rand = 0x9e3779b9u;
static uint32_t nextr(void) { g_rand = g_rand * 1103515245u + 12345u; return g_rand >> 8; }
static void seed(uint32_t s) { g_rand = s ? s : 1u; }

static void* xalloc(size_t n) {
    void* p = malloc(n ? n : 1u);
    if (!p) { fprintf(stderr, "oom\n"); exit(2); }
    return p;
}

typedef struct { uint32_t rows, d_in, d_out; } Shape;

/*
 * Shapes chosen around the dispatch thresholds rather than for realism:
 * d_in below/at/above the AVX-VNNI (32) and AVX-512-VNNI (64) minimums, and row
 * counts that are and are not multiples of the 4-row tile so both the tiled body
 * and its scalar-tail fallback execute.
 */
static const Shape SHAPES[] = {
    {  1,  16,  16 }, {  1,  32,  32 }, {  1,  64,  64 }, {  1, 128,  64 },
    {  3,  31,  16 }, {  4,  32,  32 }, {  5,  64,  32 }, {  7,  65,  48 },
    {  8, 128,  64 }, { 13,  64,  80 }, { 16, 320, 320 }, { 17,  63,  33 },
    { 32,  96, 128 }, { 33, 129,  17 },
};

/* Every canonical byte dtype combination, including asymmetric zero points,
 * because the VNNI kernels remap operands per dtype and a sign error there is
 * invisible in the symmetric case. */
typedef struct {
    uint32_t input_dtype, weight_dtype, output_dtype;
    int32_t input_zp, weight_zp, output_zp;
} DtypeCase;

static const DtypeCase DTYPES[] = {
    { VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8, 128,   0, 128 },
    { VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8,  17,   0,  91 },
    { VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8, 200,   0, -12 },
    { VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_U8, -33,   0, 128 },
    { VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,   0,   0,   0 },
    { VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8, 128, 128, 128 },
    { VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8,  40, 200,  60 },
    { VX_DTYPE_I8, VX_DTYPE_U8, VX_DTYPE_U8, -70, 130, 128 },
};

static void fill_bytes(void* p, size_t n, uint32_t dtype) {
    uint8_t* b = (uint8_t*)p;
    for (size_t i = 0; i < n; i++) {
        /* Full range for both dtypes, so a kernel that is only exact over a
         * narrow band fails here. */
        b[i] = (uint8_t)(nextr() & 255u);
        if (dtype == VX_DTYPE_I8 && b[i] == 0x80u) b[i] = 0x81u;  /* avoid -128 */
    }
}

static int qlinear_parity(const char* tier) {
    int mismatches = 0;
    for (size_t s = 0; s < sizeof SHAPES / sizeof SHAPES[0]; s++) {
        for (size_t d = 0; d < sizeof DTYPES / sizeof DTYPES[0]; d++) {
            const Shape sh = SHAPES[s];
            const DtypeCase dc = DTYPES[d];
            const size_t in_n = (size_t)sh.rows * sh.d_in;
            const size_t w_n = (size_t)sh.d_out * sh.d_in;
            const size_t out_n = (size_t)sh.rows * sh.d_out;
            void* input = xalloc(in_n);
            void* weight = xalloc(w_n);
            void* got = xalloc(out_n);
            void* want = xalloc(out_n);
            int32_t* bias = (int32_t*)xalloc(sh.d_out * sizeof(int32_t));
            float* wscale = (float*)xalloc(sh.d_out * sizeof(float));
            int32_t* wzp = (int32_t*)xalloc(sh.d_out * sizeof(int32_t));

            seed((uint32_t)(1000u + s * 97u + d));
            fill_bytes(input, in_n, dc.input_dtype);
            fill_bytes(weight, w_n, dc.weight_dtype);
            for (uint32_t c = 0; c < sh.d_out; c++) {
                bias[c] = (int32_t)(nextr() % 4001u) - 2000;
                wscale[c] = 0.0005f + (float)(nextr() % 100u) * 0.0001f;
                wzp[c] = dc.weight_zp;
            }
            memset(got, 0xAA, out_n);
            memset(want, 0x55, out_n);

            const int ref = qlinear_i8u8(input, weight, bias, wscale, wzp, want,
                sh.rows, sh.d_in, sh.d_out, 0.011f, dc.input_zp, 0.037f,
                dc.output_zp, dc.input_dtype, dc.weight_dtype, dc.output_dtype);
            const int fast = vx_qlinear_i8u8_native(input, weight, bias, wscale,
                wzp, got, sh.rows, sh.d_in, sh.d_out, 0.011f, dc.input_zp,
                0.037f, dc.output_zp, dc.input_dtype, dc.weight_dtype,
                dc.output_dtype);
            if (ref != fast) {
                printf("  [%s] %ux%ux%u dtypes(%u,%u,%u): status %d vs %d\n",
                       tier, sh.rows, sh.d_in, sh.d_out, dc.input_dtype,
                       dc.weight_dtype, dc.output_dtype, ref, fast);
                mismatches++;
            } else if (ref == 1 && memcmp(got, want, out_n) != 0) {
                size_t first = 0;
                while (first < out_n &&
                       ((uint8_t*)got)[first] == ((uint8_t*)want)[first]) first++;
                printf("  [%s] %ux%ux%u dtypes(%u,%u,%u): first diff at %zu "
                       "got %u want %u\n", tier, sh.rows, sh.d_in, sh.d_out,
                       dc.input_dtype, dc.weight_dtype, dc.output_dtype, first,
                       ((uint8_t*)got)[first], ((uint8_t*)want)[first]);
                mismatches++;
            }
            free(input); free(weight); free(got); free(want);
            free(bias); free(wscale); free(wzp);
        }
    }
    return mismatches;
}

static int qconv_parity(const char* tier) {
    /* 3x3 stride-1 NHWC/OHWI, the shape family the encoder actually runs, with
     * channel counts around the AVX-512-VNNI input_per_group minimum. */
    static const uint32_t CH[][2] = { {8,16}, {16,32}, {48,96}, {64,64}, {96,32} };
    static const uint32_t HW[][2] = { {5,7}, {8,8}, {10,12} };
    int mismatches = 0;
    for (size_t c = 0; c < sizeof CH / sizeof CH[0]; c++) {
        for (size_t s = 0; s < sizeof HW / sizeof HW[0]; s++) {
            const uint32_t cin = CH[c][0], cout = CH[c][1];
            const uint32_t oh = HW[s][0], ow = HW[s][1];
            const size_t in_n = (size_t)oh * ow * cin;
            const size_t w_n = (size_t)cout * 9u * cin;
            const size_t out_n = (size_t)oh * ow * cout;
            void* input = xalloc(in_n);
            void* weight = xalloc(w_n);
            void* got = xalloc(out_n);
            void* want = xalloc(out_n);
            int32_t* bias = (int32_t*)xalloc(cout * sizeof(int32_t));
            float* wscale = (float*)xalloc(cout * sizeof(float));
            int32_t* wzp = (int32_t*)xalloc(cout * sizeof(int32_t));
            seed((uint32_t)(7000u + c * 31u + s));
            fill_bytes(input, in_n, VX_DTYPE_U8);
            fill_bytes(weight, w_n, VX_DTYPE_I8);
            for (uint32_t k = 0; k < cout; k++) {
                bias[k] = (int32_t)(nextr() % 2001u) - 1000;
                wscale[k] = 0.0008f + (float)(nextr() % 50u) * 0.0001f;
                wzp[k] = 0;
            }
            memset(got, 0xAA, out_n); memset(want, 0x55, out_n);
#define ARGS(dst) input, weight, bias, wscale, wzp, (dst), 1u, oh, ow, cin, \
    oh, ow, cout, 3u, 3u, cin, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 1u, 0u, \
    0.011f, 128, 0.037f, 128, VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8
            const int ref = qconv2d_i8u8(ARGS(want));
            const int fast = vx_qconv2d_i8u8_native(ARGS(got));
#undef ARGS
            if (ref != fast || (ref == 1 && memcmp(got, want, out_n) != 0)) {
                printf("  [%s] conv %ux%u c%u->%u: status %d vs %d%s\n", tier,
                       oh, ow, cin, cout, ref, fast,
                       (ref == fast) ? " bytes differ" : "");
                mismatches++;
            }
            free(input); free(weight); free(got); free(want);
            free(bias); free(wscale); free(wzp);
        }
    }
    return mismatches;
}

int main(int argc, char** argv) {
    static const char* TIERS[] = {
        "baseline", "avx2", "avxvnni", "avx512vnni",
    };
    const VxKernelPlatform* host = vx_kernel_platform();
    printf("host isa=%s (avx2=%d avxvnni=%d avx512vnni=%d neon=%d dotprod=%d)\n",
           vx_kernel_isa_name(host->isa), host->has_avx2, host->has_avx_vnni,
           host->has_avx512_vnni, host->has_neon, host->has_arm_dotprod);

    /* The platform cache is per process, so each tier runs as a child.  When
     * VOLVOXAI_CPU_ISA is already set, honour it and test that tier only. */
    const char* pinned = getenv("VOLVOXAI_CPU_ISA");
    if (pinned && pinned[0]) {
        VxKernelPlatform p;
        vx_kernel_platform_resolve(&p);
        printf("tier %-11s -> resolved %s\n", pinned, vx_kernel_isa_name(p.isa));
        g_failures += qlinear_parity(pinned);
        g_failures += qconv_parity(pinned);
        if (g_failures) {
            printf("qlinear/qconv ISA parity FAILED (%d) on tier %s\n",
                   g_failures, pinned);
            return 1;
        }
        printf("qlinear/qconv ISA parity passed on tier %s (resolved %s)\n",
               pinned, vx_kernel_isa_name(p.isa));
        return 0;
    }

    /* No pin: re-exec once per tier so every tier gets a fresh resolution. */
    if (argc < 1 || !argv[0] || !argv[0][0]) {
        fprintf(stderr, "cannot resolve test executable path\n");
        return 2;
    }
    int failures = 0;
    for (size_t i = 0; i < sizeof TIERS / sizeof TIERS[0]; i++) {
        char command[512];
        snprintf(command, sizeof command,
                 "VOLVOXAI_CPU_ISA=%s \"%s\"", TIERS[i], argv[0]);
        const int rc = system(command);
        if (rc != 0) failures++;
    }
    if (failures) {
        printf("qlinear/qconv ISA parity FAILED on %d tier(s)\n", failures);
        return 1;
    }
    printf("qlinear/qconv ISA parity passed on all %zu tiers\n",
           sizeof TIERS / sizeof TIERS[0]);
    return 0;
}
