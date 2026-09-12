#if !defined(_WIN32)
#define _POSIX_C_SOURCE 200809L
#endif

#include "../src/kernels/kernel_platform.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;

#define CHECK(condition) do {                                                \
    if (!(condition)) {                                                      \
        fprintf(stderr, "CHECK failed at %s:%d: %s\n", __FILE__, __LINE__, \
                #condition);                                                 \
        failures++;                                                          \
    }                                                                        \
} while (0)

static void set_clamp(const char* value) {
#if defined(_WIN32)
    _putenv_s("VOLVOXAI_CPU_ISA", value ? value : "");
#else
    if (value) setenv("VOLVOXAI_CPU_ISA", value, 1);
    else unsetenv("VOLVOXAI_CPU_ISA");
#endif
}

static VxKernelPlatform resolve_host(const char* clamp) {
    VxKernelPlatform platform;
    set_clamp(clamp);
    memset(&platform, 0, sizeof(platform));
    vx_kernel_platform_resolve(&platform);
    return platform;
}

static VxKernelPlatform resolve_fake(VxKernelArchitecture architecture,
                                     VxKernelIsa ceiling) {
    const VxKernelCpuCapabilities capabilities = {
        .has_avx2 = 1,
        .has_avx_vnni = 1,
        .has_avx512f = 1,
        .has_avx512_vnni = 1,
        .has_neon = 1,
        .has_arm_dotprod = 1,
        .has_arm_i8mm = 1,
        .has_arm_sve = 1,
        .has_arm_sve2 = 1,
        .has_arm_sve_i8mm = 1,
        .arm_sve_vl_bytes = 32u,
    };
    VxKernelPlatform platform;
    vx_kernel_platform_resolve_capabilities(
        &platform, architecture, ceiling, &capabilities);
    return platform;
}

static void check_subset(const VxKernelPlatform* lower,
                         const VxKernelPlatform* upper) {
    CHECK(!lower->has_avx2 || upper->has_avx2);
    CHECK(!lower->has_avx_vnni || upper->has_avx_vnni);
    CHECK(!lower->has_avx512f || upper->has_avx512f);
    CHECK(!lower->has_avx512_vnni || upper->has_avx512_vnni);
    CHECK(!lower->has_neon || upper->has_neon);
    CHECK(!lower->has_arm_dotprod || upper->has_arm_dotprod);
    CHECK(!lower->has_arm_i8mm || upper->has_arm_i8mm);
    CHECK(!lower->has_arm_sve || upper->has_arm_sve);
    CHECK(!lower->has_arm_sve2 || upper->has_arm_sve2);
    CHECK(!lower->has_arm_sve_i8mm || upper->has_arm_sve_i8mm);
}

static void test_host_clamps_only_remove(void) {
    static const char* clamps[] = {
        "baseline", "neon", "neondotprod", "neoni8mm", "sve2",
        "avx2", "avxvnni", "avx512vnni",
    };
    const VxKernelPlatform automatic = resolve_host(NULL);
    CHECK(automatic.configuration_valid);
    CHECK(!automatic.has_avx2 || vx_cpu_has_avx2());
    CHECK(!automatic.has_avx_vnni || vx_cpu_has_avx_vnni());
    CHECK(!automatic.has_avx512f || vx_cpu_has_avx512f());
    CHECK(!automatic.has_avx512_vnni || vx_cpu_has_avx512_vnni());
    CHECK(!automatic.has_neon || vx_cpu_has_neon());
    CHECK(!automatic.has_arm_dotprod || vx_cpu_has_arm_dotprod());
    CHECK(!automatic.has_arm_i8mm || vx_cpu_has_arm_i8mm());
    CHECK(!automatic.has_arm_sve || vx_cpu_has_arm_sve());
    CHECK(!automatic.has_arm_sve2 || vx_cpu_has_arm_sve2());

    for (size_t index = 0u;
         index < sizeof(clamps) / sizeof(clamps[0]); index++) {
        const VxKernelPlatform clamped = resolve_host(clamps[index]);
        check_subset(&clamped, &automatic);
    }

    {
        const VxKernelPlatform baseline = resolve_host("baseline");
        CHECK(baseline.configuration_valid);
        CHECK(baseline.isa == VX_KERNEL_ISA_BASELINE);
        CHECK(!baseline.has_avx2 && !baseline.has_avx_vnni &&
              !baseline.has_avx512f && !baseline.has_avx512_vnni);
        CHECK(!baseline.has_neon && !baseline.has_arm_dotprod &&
              !baseline.has_arm_i8mm && !baseline.has_arm_sve &&
              !baseline.has_arm_sve2 && !baseline.has_arm_sve_i8mm);
        CHECK(baseline.arm_sve_vl_bytes == 0u);
    }
}

static void test_architecture_ladders(void) {
    VxKernelPlatform platform =
        resolve_fake(VX_KERNEL_ARCH_X86, VX_KERNEL_ISA_AUTO);
    CHECK(platform.configuration_valid);
    CHECK(platform.isa == VX_KERNEL_ISA_AVX512_VNNI);
    CHECK(platform.has_avx2 && platform.has_avx_vnni &&
          platform.has_avx512f && platform.has_avx512_vnni);
    CHECK(!platform.has_neon && !platform.has_arm_sve);

    platform = resolve_fake(VX_KERNEL_ARCH_X86, VX_KERNEL_ISA_AVX2);
    CHECK(platform.configuration_valid && platform.has_avx2);
    CHECK(platform.isa == VX_KERNEL_ISA_AVX2);
    CHECK(!platform.has_avx_vnni && !platform.has_avx512f &&
          !platform.has_avx512_vnni);

    platform = resolve_fake(VX_KERNEL_ARCH_X86, VX_KERNEL_ISA_AVX_VNNI);
    CHECK(platform.configuration_valid && platform.has_avx2 &&
          platform.has_avx_vnni);
    CHECK(platform.isa == VX_KERNEL_ISA_AVX_VNNI);
    CHECK(!platform.has_avx512f && !platform.has_avx512_vnni);

    platform = resolve_fake(VX_KERNEL_ARCH_ARM, VX_KERNEL_ISA_AUTO);
    CHECK(platform.configuration_valid);
    CHECK(platform.isa == VX_KERNEL_ISA_SVE2);
    CHECK(platform.has_neon && platform.has_arm_dotprod &&
          platform.has_arm_i8mm && platform.has_arm_sve &&
          platform.has_arm_sve2 && platform.has_arm_sve_i8mm);
    CHECK(platform.arm_sve_vl_bytes == 32u);
    CHECK(!platform.has_avx2 && !platform.has_avx512_vnni);

    platform = resolve_fake(VX_KERNEL_ARCH_ARM, VX_KERNEL_ISA_NEON_DOTPROD);
    CHECK(platform.configuration_valid && platform.has_neon &&
          platform.has_arm_dotprod);
    CHECK(platform.isa == VX_KERNEL_ISA_NEON_DOTPROD);
    CHECK(!platform.has_arm_i8mm && !platform.has_arm_sve);

    platform = resolve_fake(VX_KERNEL_ARCH_ARM, VX_KERNEL_ISA_NEON_I8MM);
    CHECK(platform.configuration_valid && platform.has_neon &&
          platform.has_arm_dotprod && platform.has_arm_i8mm);
    CHECK(platform.isa == VX_KERNEL_ISA_NEON_I8MM);
    CHECK(!platform.has_arm_sve && !platform.has_arm_sve2);
}

static void test_invalid_requests_fail_closed(void) {
    const VxKernelCpuCapabilities none = {0};
    VxKernelPlatform platform;

    vx_kernel_platform_resolve_capabilities(
        &platform, VX_KERNEL_ARCH_X86, VX_KERNEL_ISA_AVX2, &none);
    CHECK(!platform.configuration_valid);
    CHECK(platform.isa == VX_KERNEL_ISA_BASELINE && !platform.has_avx2);

    platform = resolve_fake(VX_KERNEL_ARCH_X86, VX_KERNEL_ISA_SVE2);
    CHECK(!platform.configuration_valid);
    CHECK(platform.isa == VX_KERNEL_ISA_BASELINE);

    platform = resolve_fake(VX_KERNEL_ARCH_ARM, VX_KERNEL_ISA_AVX512_VNNI);
    CHECK(!platform.configuration_valid);
    CHECK(platform.isa == VX_KERNEL_ISA_BASELINE);

    platform = resolve_host("not-an-isa");
    CHECK(platform.requested_isa == VX_KERNEL_ISA_INVALID);
    CHECK(!platform.configuration_valid);
    CHECK(platform.isa == VX_KERNEL_ISA_BASELINE);
}

static void test_names_thresholds_and_cache(void) {
    static const VxKernelIsa tiers[] = {
        VX_KERNEL_ISA_AUTO, VX_KERNEL_ISA_BASELINE,
        VX_KERNEL_ISA_NEON, VX_KERNEL_ISA_NEON_DOTPROD,
        VX_KERNEL_ISA_NEON_I8MM, VX_KERNEL_ISA_SVE2,
        VX_KERNEL_ISA_AVX2, VX_KERNEL_ISA_AVX_VNNI,
        VX_KERNEL_ISA_AVX512_VNNI,
    };
    for (size_t index = 0u; index < sizeof(tiers) / sizeof(tiers[0]); index++) {
        set_clamp(vx_kernel_isa_name(tiers[index]));
        CHECK(vx_kernel_isa_clamp_request() == tiers[index]);
    }
    set_clamp(NULL);

    {
        const VxKernelPlatform automatic = resolve_host(NULL);
        CHECK(automatic.qlinear_avx512_vnni_min_d_in >=
              automatic.qlinear_avx_vnni_min_d_in);
        CHECK(automatic.qlinear_avx_vnni_min_d_in > 0u);
        CHECK(automatic.qlinear_maddubs_min_d_in > 0u);
    }
    {
        const VxKernelPlatform* first = vx_kernel_platform();
        const VxKernelPlatform* second = vx_kernel_platform();
        CHECK(first == second);
        CHECK(first->configuration_valid);
    }
}

int main(void) {
    test_host_clamps_only_remove();
    test_architecture_ladders();
    test_invalid_requests_fail_closed();
    test_names_thresholds_and_cache();
    set_clamp(NULL);
    if (failures) {
        fprintf(stderr, "kernel platform contract failed (%d checks)\n",
                failures);
        return 1;
    }
    puts("kernel platform contract passed");
    return 0;
}
