/*
 * Byte-exact parity of every ISA tier the host can run, for QBatchMatMul.
 *
 * test_qlinear_isa_parity covers the QLinear and QConv dispatchers; QBatchMatMul
 * had no equivalent, and it is the operator static-INT8 graphs use for score
 * and other dynamic matrix products.  Its allocation-free AVX2 edge kernel uses
 * _mm256_maddubs_epi16 and must split the unsigned operand to avoid its
 * saturating I16 intermediate.  The multi-row kernel instead packs centered
 * I16 pairs and uses _mm256_madd_epi16; both paths must remain identical to
 * the portable I32 accumulation and affine requantization contract.
 *
 * Shapes therefore straddle the reduction lengths where saturation becomes
 * reachable, and operand values include the full byte range rather than a small
 * centered band, because a mid-range-only test cannot reach the saturation
 * bound at all.
 *
 * VOLVOXAI_CPU_ISA lets one binary walk every tier the host supports; the
 * printed tier list is part of the result, not decoration.
 */
#include "../src/kernels/inference_kernels.h"
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

typedef struct { uint32_t m, k, n; } Shape;

/* k values bracket the vector body/tail boundary and reach reductions long
 * enough for a saturating int16 intermediate to matter. */
static const Shape SHAPES[] = {
    {  1,  16,  16 }, {  1,  32,  32 }, {  1,  40,  40 }, {  4,  32,  64 },
    {  7,  31,  17 }, {  8,  64,  64 }, { 12,  40, 251 }, { 16,  40,  40 },
    { 17,  65,  33 }, { 32, 128, 128 }, { 41, 210,  40 }, { 64, 320, 320 },
    /* The centered ABI is still I32-safe here, while the old raw U8xI8
     * compensation bound is not.  This proves the centered-I16 packed route
     * does not accidentally inherit the edge kernel's narrower predicate. */
    {  4, 70000, 16 },
};

typedef struct {
    uint32_t a_dtype, b_dtype, output_dtype;
    int32_t a_zero_point, b_zero_point, output_zero_point;
} DtypeCase;

static const DtypeCase DTYPES[] = {
    { VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_U8, 128,   0, 128 },
    { VX_DTYPE_U8, VX_DTYPE_I8, VX_DTYPE_I8, 137,  -5,   0 },
    { VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_I8,   0,   0,   0 },
    { VX_DTYPE_I8, VX_DTYPE_I8, VX_DTYPE_U8,  -7,   3, 128 },
    { VX_DTYPE_U8, VX_DTYPE_U8, VX_DTYPE_U8, 128, 128, 128 },
};

static size_t element_size(uint32_t dtype) { (void)dtype; return 1u; }

/* Fill across the whole byte range: clustering near the zero point hides
 * saturation, which is exactly the failure this test exists to catch. */
static void fill_bytes(void* buffer, size_t count, uint32_t dtype) {
    for (size_t index = 0; index < count; index++) {
        const uint32_t value = nextr() & 0xffu;
        if (dtype == VX_DTYPE_I8) ((int8_t*)buffer)[index] = (int8_t)(value - 128u);
        else ((uint8_t*)buffer)[index] = (uint8_t)value;
    }
}

static int compare_case(const Shape shape, const DtypeCase dtypes,
                        const char* tier) {
    const size_t a_count = (size_t)shape.m * shape.k;
    const size_t b_count = (size_t)shape.k * shape.n;
    const size_t out_count = (size_t)shape.m * shape.n;
    void* a = xalloc(a_count * element_size(dtypes.a_dtype));
    void* b = xalloc(b_count * element_size(dtypes.b_dtype));
    void* native = xalloc(out_count);
    void* portable = xalloc(out_count);
    const size_t workspace_bytes =
        vx_qbatch_matmul_i8u8_native_workspace_bytes(
            shape.m, shape.k, shape.n);
    void* workspace = workspace_bytes ? xalloc(workspace_bytes) : NULL;
    int mismatch = 0;

    seed((uint32_t)(shape.m * 7919u + shape.k * 104729u + shape.n));
    fill_bytes(a, a_count, dtypes.a_dtype);
    fill_bytes(b, b_count, dtypes.b_dtype);
    memset(native, 0, out_count);
    memset(portable, 0, out_count);

    /* Output scale is deliberately coarse so requantization does not mask a
     * wrong accumulator behind rounding. */
    const float a_scale = 1.0f / 64.0f;
    const float b_scale = 1.0f / 32.0f;
    const float output_scale = 1.0f / 4.0f;

    const int native_status = vx_qbatch_matmul_i8u8_native_with_workspace(
        a, b, native, shape.m, shape.k, shape.n,
        a_scale, dtypes.a_zero_point, b_scale, dtypes.b_zero_point,
        output_scale, dtypes.output_zero_point,
        dtypes.a_dtype, dtypes.b_dtype, dtypes.output_dtype,
        workspace, workspace_bytes);
    const int portable_status = qbatch_matmul_i8u8(
        a, b, portable, shape.m, shape.k, shape.n,
        a_scale, dtypes.a_zero_point, b_scale, dtypes.b_zero_point,
        output_scale, dtypes.output_zero_point,
        dtypes.a_dtype, dtypes.b_dtype, dtypes.output_dtype);

    if (native_status != portable_status) {
        printf("  status mismatch tier=%s m=%u k=%u n=%u native=%d portable=%d\n",
               tier, shape.m, shape.k, shape.n, native_status, portable_status);
        mismatch = 1;
    } else if (native_status != 0) {
        for (size_t index = 0; index < out_count; index++) {
            const int native_value = dtypes.output_dtype == VX_DTYPE_I8
                ? (int)((int8_t*)native)[index] : (int)((uint8_t*)native)[index];
            const int portable_value = dtypes.output_dtype == VX_DTYPE_I8
                ? (int)((int8_t*)portable)[index] : (int)((uint8_t*)portable)[index];
            if (native_value != portable_value) {
                if (!mismatch) {
                    printf("  MISMATCH tier=%s m=%u k=%u n=%u dtypes=%u/%u/%u "
                           "zp=%d/%d/%d at %zu: native=%d portable=%d\n",
                           tier, shape.m, shape.k, shape.n,
                           dtypes.a_dtype, dtypes.b_dtype, dtypes.output_dtype,
                           dtypes.a_zero_point, dtypes.b_zero_point,
                           dtypes.output_zero_point,
                           index, native_value, portable_value);
                }
                mismatch++;
            }
        }
        if (mismatch)
            printf("    %d of %zu elements differ\n", mismatch, out_count);
    }

    free(a); free(b); free(native); free(portable); free(workspace);
    return mismatch ? 1 : 0;
}

static int parity(const char* tier) {
    int failures = 0;
    for (size_t s = 0; s < sizeof(SHAPES) / sizeof(SHAPES[0]); s++)
        for (size_t d = 0; d < sizeof(DTYPES) / sizeof(DTYPES[0]); d++)
            failures += compare_case(SHAPES[s], DTYPES[d], tier);
    return failures;
}

typedef struct {
    const char* name;
    VxKernelIsa isa;
} IsaTier;

static const IsaTier TIERS[] = {
    { "baseline", VX_KERNEL_ISA_BASELINE },
    { "avx2", VX_KERNEL_ISA_AVX2 },
    { "avxvnni", VX_KERNEL_ISA_AVX_VNNI },
    { "avx512vnni", VX_KERNEL_ISA_AVX512_VNNI },
    { "neon", VX_KERNEL_ISA_NEON },
    { "neondotprod", VX_KERNEL_ISA_NEON_DOTPROD },
};

static const IsaTier* find_tier(const char* name) {
    if (!name) return NULL;
    for (size_t index = 0; index < sizeof(TIERS) / sizeof(TIERS[0]); index++)
        if (!strcmp(name, TIERS[index].name)) return &TIERS[index];
    return NULL;
}

static int host_supports_tier(const VxKernelPlatform* host,
                              VxKernelIsa isa) {
    switch (isa) {
        case VX_KERNEL_ISA_AVX2: return host->has_avx2;
        case VX_KERNEL_ISA_AVX_VNNI: return host->has_avx_vnni;
        case VX_KERNEL_ISA_AVX512_VNNI: return host->has_avx512_vnni;
        case VX_KERNEL_ISA_NEON: return host->has_neon;
        case VX_KERNEL_ISA_NEON_DOTPROD: return host->has_arm_dotprod;
        default: return isa == VX_KERNEL_ISA_BASELINE;
    }
}

int main(int argc, char** argv) {
    VxKernelPlatform platform;
    vx_kernel_platform_resolve(&platform);

    const char* pinned = getenv("VOLVOXAI_CPU_ISA");
    if (pinned && pinned[0]) {
        const IsaTier* requested = find_tier(pinned);
        printf("tier %-11s -> resolved %s\n", pinned, vx_kernel_isa_name(platform.isa));
        if (!requested || platform.isa != requested->isa) {
            printf("qbatch_matmul ISA parity FAILED: requested %s resolved %s\n",
                   pinned, vx_kernel_isa_name(platform.isa));
            return 1;
        }
        g_failures += parity(pinned);
        if (g_failures) {
            printf("qbatch_matmul ISA parity FAILED (%d) on tier %s\n",
                   g_failures, pinned);
            return 1;
        }
        printf("qbatch_matmul ISA parity passed on tier %s (resolved %s)\n",
               pinned, vx_kernel_isa_name(platform.isa));
        return 0;
    }

    printf("host isa=%s\n", vx_kernel_isa_name(platform.isa));
    fflush(stdout);
    if (argc < 1 || !argv[0] || !argv[0][0]) {
        fprintf(stderr, "cannot resolve test executable path\n");
        return 2;
    }
    size_t tested = 0;
    size_t skipped = 0;
    for (size_t index = 0; index < sizeof(TIERS) / sizeof(TIERS[0]); index++) {
        char command[512];
        if (!host_supports_tier(&platform, TIERS[index].isa)) {
            printf("tier %-11s -> skipped (unsupported by host)\n",
                   TIERS[index].name);
            skipped++;
            continue;
        }
        snprintf(command, sizeof(command),
                 "VOLVOXAI_CPU_ISA=%s \"%s\"", TIERS[index].name, argv[0]);
        tested++;
        if (system(command) != 0) g_failures++;
    }
    if (g_failures) {
        printf("qbatch_matmul ISA parity FAILED on %d tier(s)\n", g_failures);
        return 1;
    }
    printf("qbatch_matmul ISA parity passed on %zu supported tier(s); "
           "%zu skipped\n", tested, skipped);
    return 0;
}
