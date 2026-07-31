/*
 * The resolved kernel platform is what every dispatcher now reads instead of
 * probing CPUID inline, so its two safety properties need to be pinned:
 *
 *   - it never reports a capability the CPU does not have, and
 *   - VOLVOXAI_CPU_ISA only ever removes capability.
 *
 * The second is what makes the clamp safe to expose: a typo or a request for a
 * wider ISA than the host has must never select an instruction that would fault.
 */
/* setenv/unsetenv are POSIX, and this target compiles with -std=c11. */
#define _POSIX_C_SOURCE 200809L

#include "../src/kernels/kernel_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int g_failures;
#define CHECK(cond) do { \
    if (!(cond)) { \
        printf("CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, #cond); \
        g_failures++; \
    } \
} while (0)

/* Resolve with an explicit clamp, bypassing the process-wide cache so one test
 * binary can exercise every tier. */
static VxKernelPlatform resolve_with(const char* isa) {
    VxKernelPlatform platform;
    memset(&platform, 0, sizeof platform);
    if (isa) setenv("VOLVOXAI_CPU_ISA", isa, 1);
    else unsetenv("VOLVOXAI_CPU_ISA");
    vx_kernel_platform_resolve(&platform);
    return platform;
}

static void test_never_invents_capability(void) {
    /* The unclamped resolution is the ceiling: it may drop a capability the CPU
     * reports, but it must never add one. */
    const VxKernelPlatform full = resolve_with(NULL);
    CHECK(!full.has_avx2 || vx_cpu_has_avx2());
    CHECK(!full.has_avx_vnni || vx_cpu_has_avx_vnni());
    CHECK(!full.has_avx512_vnni || vx_cpu_has_avx512_vnni());
    CHECK(!full.has_neon || vx_cpu_has_neon());
    CHECK(!full.has_arm_dotprod || vx_cpu_has_arm_dotprod());

    /* Tier implications: a wider tier presupposes the narrower one, otherwise a
     * dispatcher could reach a VNNI kernel on a CPU without AVX2 state. */
    CHECK(!full.has_avx_vnni || full.has_avx2);
    CHECK(!full.has_avx512_vnni || full.has_avx2);
    CHECK(!full.has_arm_dotprod || full.has_neon);

    /* The reported tier must agree with the predicates. */
    if (full.has_avx512_vnni)      CHECK(full.isa == VX_KERNEL_ISA_AVX512_VNNI);
    else if (full.has_avx_vnni)    CHECK(full.isa == VX_KERNEL_ISA_AVX_VNNI);
    else if (full.has_avx2)        CHECK(full.isa == VX_KERNEL_ISA_AVX2);
    else if (full.has_arm_dotprod) CHECK(full.isa == VX_KERNEL_ISA_NEON_DOTPROD);
    else if (full.has_neon)        CHECK(full.isa == VX_KERNEL_ISA_NEON);
    else                           CHECK(full.isa == VX_KERNEL_ISA_BASELINE);
}

static void test_clamp_only_removes(void) {
    const VxKernelPlatform full = resolve_with(NULL);
    static const char* tiers[] = {
        "baseline", "neon", "neondotprod", "avx2", "avxvnni", "avx512vnni",
    };
    for (size_t i = 0; i < sizeof tiers / sizeof tiers[0]; i++) {
        const VxKernelPlatform clamped = resolve_with(tiers[i]);
        /* No clamp may turn a capability on. */
        CHECK(!clamped.has_avx2 || full.has_avx2);
        CHECK(!clamped.has_avx_vnni || full.has_avx_vnni);
        CHECK(!clamped.has_avx512_vnni || full.has_avx512_vnni);
        CHECK(!clamped.has_neon || full.has_neon);
        CHECK(!clamped.has_arm_dotprod || full.has_arm_dotprod);
        /* Implications must survive clamping too. */
        CHECK(!clamped.has_avx_vnni || clamped.has_avx2);
        CHECK(!clamped.has_avx512_vnni || clamped.has_avx2);
        CHECK(!clamped.has_arm_dotprod || clamped.has_neon);
    }
}

static void test_baseline_clamp_disables_everything(void) {
    const VxKernelPlatform baseline = resolve_with("baseline");
    CHECK(baseline.isa == VX_KERNEL_ISA_BASELINE);
    CHECK(!baseline.has_avx2);
    CHECK(!baseline.has_avx_vnni);
    CHECK(!baseline.has_avx512_vnni);
    CHECK(!baseline.has_neon);
    CHECK(!baseline.has_arm_dotprod);
}

static void test_avx2_clamp_keeps_avx2_drops_vnni(void) {
    const VxKernelPlatform full = resolve_with(NULL);
    const VxKernelPlatform avx2 = resolve_with("avx2");
    CHECK(!avx2.has_avx_vnni);
    CHECK(!avx2.has_avx512_vnni);
    /* AVX2 itself must survive when the host has it. */
    CHECK(avx2.has_avx2 == full.has_avx2);
    if (avx2.has_avx2) CHECK(avx2.isa == VX_KERNEL_ISA_AVX2);
}

static void test_unknown_value_is_not_a_clamp(void) {
    /* A typo must not silently disable every vector kernel; that would look
     * like an unexplained slowdown rather than a configuration error. */
    const VxKernelPlatform full = resolve_with(NULL);
    const VxKernelPlatform typo = resolve_with("avx-2");
    const VxKernelPlatform empty = resolve_with("");
    CHECK(typo.isa == full.isa);
    CHECK(typo.has_avx2 == full.has_avx2);
    CHECK(typo.has_avx512_vnni == full.has_avx512_vnni);
    CHECK(empty.isa == full.isa);
    CHECK(empty.has_avx2 == full.has_avx2);
}

static void test_thresholds_and_names(void) {
    const VxKernelPlatform p = resolve_with(NULL);
    /* A zero threshold would silently make a tier always eligible. */
    CHECK(p.qlinear_avx512_vnni_min_d_in > 0);
    CHECK(p.qlinear_avx_vnni_min_d_in > 0);
    CHECK(p.qlinear_maddubs_min_d_in > 0);
    CHECK(p.qconv_avx512_vnni_min_input_per_group > 0);
    /* Wider tiers need at least as long a reduction to pay for themselves. */
    CHECK(p.qlinear_avx512_vnni_min_d_in >= p.qlinear_avx_vnni_min_d_in);

    CHECK(!strcmp(vx_kernel_isa_name(VX_KERNEL_ISA_BASELINE), "baseline"));
    CHECK(!strcmp(vx_kernel_isa_name(VX_KERNEL_ISA_AVX2), "avx2"));
    CHECK(!strcmp(vx_kernel_isa_name(VX_KERNEL_ISA_AVX_VNNI), "avxvnni"));
    CHECK(!strcmp(vx_kernel_isa_name(VX_KERNEL_ISA_AVX512_VNNI), "avx512vnni"));
    CHECK(!strcmp(vx_kernel_isa_name(VX_KERNEL_ISA_NEON), "neon"));
    CHECK(!strcmp(vx_kernel_isa_name(VX_KERNEL_ISA_NEON_DOTPROD), "neondotprod"));
    /* Every name must round trip back to the same tier through the parser. */
    static const VxKernelIsa tiers[] = {
        VX_KERNEL_ISA_BASELINE, VX_KERNEL_ISA_NEON, VX_KERNEL_ISA_NEON_DOTPROD,
        VX_KERNEL_ISA_AVX2, VX_KERNEL_ISA_AVX_VNNI, VX_KERNEL_ISA_AVX512_VNNI,
    };
    for (size_t i = 0; i < sizeof tiers / sizeof tiers[0]; i++) {
        setenv("VOLVOXAI_CPU_ISA", vx_kernel_isa_name(tiers[i]), 1);
        CHECK(vx_kernel_isa_clamp_request() == tiers[i]);
    }
    unsetenv("VOLVOXAI_CPU_ISA");
}

static void test_cached_accessor_is_stable(void) {
    const VxKernelPlatform* first = vx_kernel_platform();
    const VxKernelPlatform* second = vx_kernel_platform();
    CHECK(first == second);
    CHECK(first->isa == second->isa);
}

int main(void) {
    test_never_invents_capability();
    test_clamp_only_removes();
    test_baseline_clamp_disables_everything();
    test_avx2_clamp_keeps_avx2_drops_vnni();
    test_unknown_value_is_not_a_clamp();
    test_thresholds_and_names();
    test_cached_accessor_is_stable();
    unsetenv("VOLVOXAI_CPU_ISA");
    if (g_failures) {
        printf("kernel platform tests FAILED (%d)\n", g_failures);
        return 1;
    }
    printf("kernel platform tests passed (host isa=%s)\n",
           vx_kernel_isa_name(vx_kernel_platform()->isa));
    return 0;
}
