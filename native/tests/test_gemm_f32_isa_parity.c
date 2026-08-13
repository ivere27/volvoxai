/*
 * How far apart the F32 GEMM's ISA tiers actually are.
 *
 * The quantized dispatchers have byte-exact parity tests because their tiers
 * are required to agree exactly. This one cannot assert that: gemm_f32.c
 * states the contract as bit-identical to the same build's unblocked kernel
 * and "merely close across ISAs", because each tier keeps its own arithmetic
 * (AVX2 and AVX-512 fuse the multiply-add, the baseline does not) and resolves
 * its own tile, which changes kc and therefore how the K reduction is grouped.
 *
 * "Merely close" had never been measured, which matters as soon as anything
 * chooses a tier at run time: if two tiers disagree, then the tier is part of
 * the numerical result, and picking it from a timing measurement would make
 * that result depend on machine load. This test turns the qualitative claim
 * into a number per tier, against a double-precision reference and against the
 * baseline tier, and fails if either exceeds a bound stated here.
 *
 * VOLVOXAI_CPU_ISA lets one binary walk every tier the host supports; the
 * printed tier list is part of the result, not decoration.
 */
#include "../src/kernels/gemm_f32.h"
#include "../src/kernels/kernel_platform.h"

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

/* A tier may differ from the double reference by rounding that accumulates
 * over K; it may not differ by a factor. Both bounds are relative and are the
 * measured envelope with headroom, not aspirations. */
#define VX_GEMM_PARITY_REFERENCE_BOUND 2.5e-5
#define VX_GEMM_PARITY_TIER_BOUND      2.0e-5

static int g_failures;

static uint32_t g_rand = 0x9e3779b9u;
static uint32_t nextr(void) { g_rand = g_rand * 1103515245u + 12345u; return g_rand >> 8; }
static void seed(uint32_t s) { g_rand = s ? s : 1u; }
static float nextf(void) { return (float)((int32_t)(nextr() % 2001u) - 1000) / 1000.0f; }

/* FNV-1a over the raw output bytes. Printing this per shape is what lets two
 * tiers be compared for bit-identity rather than for similar error: equal error
 * against a double reference is consistent with different results. */
static uint64_t digest(const float* values, size_t count) {
    const unsigned char* bytes = (const unsigned char*)values;
    uint64_t hash = 1469598103934665603ull;
    for (size_t i = 0; i < count * sizeof(float); i++) {
        hash ^= bytes[i];
        hash *= 1099511628211ull;
    }
    return hash;
}

static void* xalloc(size_t n) {
    void* p = malloc(n ? n : 1u);
    if (!p) { fprintf(stderr, "oom\n"); exit(2); }
    return p;
}

typedef struct { uint32_t m, k, n; const char* label; } Shape;

/* One decode row, one encoder-sized block, one prefill, and shapes whose
 * extents are not multiples of any tile so the edge path is exercised. */
static const Shape SHAPES[] = {
    {   1,  320,  320, "decode stream" },
    {   1, 1280,  320, "decode wide-k" },
    {  41,  320,  960, "prompt prefill" },
    { 402,  320,  320, "encoder block" },
    { 251,  960,  320, "memory block" },
    {   7,  113,   53, "edges everywhere" },
    {  64,  256, 1024, "large block" },
};

typedef struct {
    const char* label;
    const char* clamp;
    const char* avx512;
    uint32_t mr;
    uint32_t nr;
    int needs_avx2;
    int needs_avx512;
} Tier;

/* F32 AVX-512 is selected by an AVX512F capability plus its separate policy
 * switch; the integer platform ladder names that ceiling avx512vnni.  Keep the
 * requested clamp and the expected GEMM micro-tile together so a typo cannot
 * silently exercise the host's highest tier under the wrong label. */
static const Tier TIERS[] = {
    { "baseline", "baseline",    "0", 8u,  8u, 0, 0 },
    { "avx2",     "avx2",        "0", 6u, 16u, 1, 0 },
    { "avx512",   "avx512vnni",  "1", 8u, 32u, 1, 1 },
};

/* A child writes each deterministic output in SHAPES order.  The parent reads
 * those raw F32 values back and enforces the tier-to-baseline bound; a digest
 * alone is useful evidence but cannot measure a numerical gap. */
static int compare_case(Shape shape, const char* tier, FILE* raw_output) {
    const uint32_t m = shape.m, k = shape.k, n = shape.n;
    const uint32_t packed = vx_gemm_f32_packed_elements(k, n);
    float* a = xalloc((size_t)m * k * sizeof(float));
    float* b = xalloc((size_t)k * n * sizeof(float));
    float* bias = xalloc((size_t)n * sizeof(float));
    float* pack = xalloc((size_t)packed * sizeof(float));
    float* c = xalloc((size_t)m * n * sizeof(float));
    double worst_reference = 0.0;
    int failures = 0;

    seed(shape.m * 7919u + shape.k * 104729u + shape.n + 1u);
    for (size_t i = 0; i < (size_t)m * k; i++) a[i] = nextf();
    for (size_t i = 0; i < (size_t)k * n; i++) b[i] = nextf();
    for (uint32_t i = 0; i < n; i++) bias[i] = nextf();

    if (!vx_gemm_f32_pack_b(b, pack, k, n, 0) ||
        !vx_gemm_f32_run_packed(a, pack, bias, c, m, k, n)) {
        printf("  refused tier=%s %s m=%u k=%u n=%u\n", tier, shape.label, m, k, n);
        failures++;
    } else {
        for (uint32_t row = 0; row < m; row++)
            for (uint32_t col = 0; col < n; col++) {
                /* Double accumulation in the source order, which is what the
                 * tiers are approximating; not a different algorithm. */
                double truth = bias[col];
                double scale;
                double delta;
                for (uint32_t d = 0; d < k; d++)
                    truth += (double)a[(size_t)row * k + d] *
                             (double)b[(size_t)d * n + col];
                scale = fabs(truth) > 1.0 ? fabs(truth) : 1.0;
                if (!isfinite(truth) ||
                    !isfinite((double)c[(size_t)row * n + col])) {
                    worst_reference = INFINITY;
                    continue;
                }
                delta = fabs((double)c[(size_t)row * n + col] - truth) / scale;
                if (delta > worst_reference) worst_reference = delta;
            }
        printf("  %-18s m=%-4u k=%-5u n=%-5u vs double %.3e digest %016llx",
               shape.label, m, k, n, worst_reference,
               (unsigned long long)digest(c, (size_t)m * n));
        printf("\n");
        if (worst_reference > VX_GEMM_PARITY_REFERENCE_BOUND) {
            printf("  EXCEEDS reference bound tier=%s %s\n", tier, shape.label);
            failures++;
        }
        if (raw_output && fwrite(c, sizeof(float), (size_t)m * n,
                                 raw_output) != (size_t)m * n) {
            printf("  failed to write raw output tier=%s %s\n",
                   tier, shape.label);
            failures++;
        }
    }

    free(a); free(b); free(bias); free(pack); free(c);
    return failures;
}

static int parity(const char* tier, FILE* raw_output) {
    VxGemmF32Plan plan;
    int failures = 0;
    for (size_t s = 0; s < sizeof(SHAPES) / sizeof(SHAPES[0]); s++) {
        plan = vx_gemm_f32_plan(SHAPES[s].m, SHAPES[s].k, SHAPES[s].n);
        printf("  plan mr=%u nr=%u mc=%u nc=%u kc=%u regime=%s blocked=%d\n",
               plan.mr, plan.nr, plan.mc, plan.nc, plan.kc,
               plan.regime == VX_GEMM_F32_REGIME_STREAM ? "stream" : "block",
               plan.blocked);
        failures += compare_case(SHAPES[s], tier, raw_output);
    }
    return failures;
}

static const Tier* find_tier(const char* label) {
    if (!label) return NULL;
    for (size_t index = 0; index < sizeof(TIERS) / sizeof(TIERS[0]); index++)
        if (!strcmp(label, TIERS[index].label)) return &TIERS[index];
    return NULL;
}

static int host_has_fma(void) {
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
    __builtin_cpu_init();
    return __builtin_cpu_supports("fma") != 0;
#else
    return 0;
#endif
}

static int tier_supported(const Tier* tier, const VxKernelPlatform* platform) {
    if (!tier || !platform) return 0;
    if (tier->needs_avx512)
        return platform->has_avx512f && host_has_fma();
    if (tier->needs_avx2)
        return platform->has_avx2 && host_has_fma();
    return 1;
}

static int child_route_matches(const Tier* tier,
                               const VxKernelPlatform* platform,
                               const char* pinned) {
    const char* avx512 = getenv("VOLVOX_F32_AVX512");
    const VxGemmF32Plan plan = vx_gemm_f32_plan(
        SHAPES[0].m, SHAPES[0].k, SHAPES[0].n);
    int platform_matches;
    if (!tier || !platform || !pinned || strcmp(pinned, tier->clamp) ||
        !avx512 || strcmp(avx512, tier->avx512)) return 0;
    if (!strcmp(tier->label, "baseline"))
        platform_matches = platform->isa == VX_KERNEL_ISA_BASELINE &&
            plan.micro == 0;
    else if (!strcmp(tier->label, "avx2"))
        platform_matches = platform->isa == VX_KERNEL_ISA_AVX2 &&
            platform->has_avx2 && !platform->has_avx512f && plan.micro != 0;
    else
        platform_matches = platform->has_avx512f && plan.micro != 0;
    if (!platform_matches || plan.mr != tier->mr || plan.nr != tier->nr) {
        fprintf(stderr,
            "requested tier %s resolved integer tier %s with GEMM "
            "micro=%d mr=%u nr=%u\n",
            tier->label, vx_kernel_isa_name(platform->isa), plan.micro,
            plan.mr, plan.nr);
        return 0;
    }
    printf("requested tier %-8s -> resolved integer %-11s GEMM mr=%u nr=%u\n",
           tier->label, vx_kernel_isa_name(platform->isa), plan.mr, plan.nr);
    return 1;
}

static size_t result_elements(void) {
    size_t elements = 0;
    for (size_t index = 0; index < sizeof(SHAPES) / sizeof(SHAPES[0]); index++)
        elements += (size_t)SHAPES[index].m * SHAPES[index].n;
    return elements;
}

static int read_results(const char* path, float* output, size_t elements) {
    FILE* file = fopen(path, "rb");
    int ok;
    if (!file) return 0;
    ok = fread(output, sizeof(float), elements, file) == elements &&
        fgetc(file) == EOF && !ferror(file);
    if (fclose(file) != 0) ok = 0;
    return ok;
}

static int compare_with_baseline(const char* tier, const float* baseline,
                                 const float* candidate) {
    size_t offset = 0;
    int failures = 0;
    for (size_t shape = 0; shape < sizeof(SHAPES) / sizeof(SHAPES[0]); shape++) {
        const size_t count = (size_t)SHAPES[shape].m * SHAPES[shape].n;
        double worst = 0.0;
        for (size_t index = 0; index < count; index++) {
            const double base = baseline[offset + index];
            const double value = candidate[offset + index];
            const double denominator = fabs(base) > 1.0 ? fabs(base) : 1.0;
            const double gap = fabs(value - base) / denominator;
            if (!isfinite(base) || !isfinite(value) || !isfinite(gap)) {
                worst = INFINITY;
                break;
            }
            if (gap > worst) worst = gap;
        }
        printf("  %-18s tier=%-8s vs baseline %.3e exact=%s\n",
               SHAPES[shape].label, tier, worst,
               memcmp(baseline + offset, candidate + offset,
                      count * sizeof(float)) == 0 ? "yes" : "no");
        if (worst > VX_GEMM_PARITY_TIER_BOUND) {
            printf("  EXCEEDS tier bound tier=%s %s\n",
                   tier, SHAPES[shape].label);
            failures++;
        }
        offset += count;
    }
    return failures;
}

int main(int argc, char** argv) {
    VxKernelPlatform platform;
    const char* pinned;
    const char* requested;
    vx_kernel_platform_resolve(&platform);
    pinned = getenv("VOLVOXAI_CPU_ISA");
    requested = getenv("VOLVOX_GEMM_PARITY_TIER");

    if (pinned && pinned[0]) {
        const Tier* tier = requested && requested[0]
            ? find_tier(requested) : NULL;
        const char* output_path = getenv("VOLVOX_GEMM_PARITY_OUTPUT");
        FILE* raw_output = NULL;
        if (requested && requested[0] &&
            (!tier || !output_path || !output_path[0] ||
             !child_route_matches(tier, &platform, pinned))) return 2;
        if (output_path && output_path[0]) {
            raw_output = fopen(output_path, "wb");
            if (!raw_output) return 2;
        }
        printf("tier %-9s -> resolved %s\n", pinned,
               vx_kernel_isa_name(platform.isa));
        g_failures += parity(tier ? tier->label : pinned, raw_output);
        if (raw_output && fclose(raw_output) != 0) g_failures++;
        if (g_failures) {
            printf("gemm_f32 ISA parity FAILED (%d) on tier %s\n",
                   g_failures, tier ? tier->label : pinned);
            return 1;
        }
        printf("gemm_f32 ISA parity passed on tier %s (resolved %s)\n",
               tier ? tier->label : pinned, vx_kernel_isa_name(platform.isa));
        return 0;
    }

    printf("host isa=%s\n", vx_kernel_isa_name(platform.isa));
    if (argc < 1 || !argv[0] || !argv[0][0]) {
        fprintf(stderr, "cannot resolve test executable path\n");
        return 2;
    }
    fflush(stdout);
    {
        char result_path[] = "/tmp/volvox-gemm-parity-XXXXXX";
        const size_t elements = result_elements();
        float* baseline = (float*)xalloc(elements * sizeof(float));
        float* candidate = (float*)xalloc(elements * sizeof(float));
        size_t tested = 0;
        size_t skipped = 0;
        int baseline_valid = 0;
        int descriptor = mkstemp(result_path);
        if (descriptor < 0 || close(descriptor) != 0) {
            fprintf(stderr, "cannot create GEMM parity result file\n");
            if (descriptor >= 0) (void)remove(result_path);
            free(candidate);
            free(baseline);
            return 2;
        }
        for (size_t index = 0;
             index < sizeof(TIERS) / sizeof(TIERS[0]); index++) {
            char command[1024];
            int command_size;
            if (!tier_supported(&TIERS[index], &platform)) {
                printf("tier %-8s SKIP: host lacks the required ISA\n",
                       TIERS[index].label);
                skipped++;
                continue;
            }
            if (remove(result_path) != 0 && index != 0u) {
                fprintf(stderr, "cannot clear GEMM parity result file\n");
                g_failures++;
                continue;
            }
            command_size = snprintf(command, sizeof(command),
                "VOLVOXAI_CPU_ISA=%s VOLVOX_F32_AVX512=%s "
                "VOLVOX_GEMM_PARITY_TIER=%s "
                "VOLVOX_GEMM_PARITY_OUTPUT=\"%s\" \"%s\"",
                TIERS[index].clamp, TIERS[index].avx512,
                TIERS[index].label, result_path, argv[0]);
            if (command_size < 0 || (size_t)command_size >= sizeof(command) ||
                system(command) != 0 ||
                !read_results(result_path,
                    index == 0u ? baseline : candidate, elements)) {
                fprintf(stderr, "tier %s child/result failed\n",
                        TIERS[index].label);
                g_failures++;
                continue;
            }
            tested++;
            if (index == 0u)
                baseline_valid = 1;
            else if (baseline_valid)
                g_failures += compare_with_baseline(
                    TIERS[index].label, baseline, candidate);
        }
        (void)remove(result_path);
        free(candidate);
        free(baseline);
        printf("GEMM tier coverage: tested=%zu skipped=%zu\n", tested, skipped);
    }
    if (g_failures) {
        printf("gemm_f32 ISA parity FAILED on %d tier(s)\n", g_failures);
        return 1;
    }
    printf("gemm_f32 ISA parity passed on every supported tier\n");
    return 0;
}
