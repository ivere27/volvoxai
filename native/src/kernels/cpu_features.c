#include "cpu_features.h"

#include <stdint.h>

#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#include <cpuid.h>
#define VX_CPU_X86_RUNTIME_QUERY 1
#else
#define VX_CPU_X86_RUNTIME_QUERY 0
#endif

#if defined(__aarch64__) && (defined(__linux__) || defined(__ANDROID__))
#include <sys/auxv.h>
#ifndef AT_HWCAP
#define AT_HWCAP 16
#endif
#ifndef HWCAP_ASIMD
#define HWCAP_ASIMD (1UL << 1)
#endif
#ifndef HWCAP_ASIMDDP
#define HWCAP_ASIMDDP (1UL << 20)
#endif
#define VX_CPU_ARM_RUNTIME_QUERY 1
#else
#define VX_CPU_ARM_RUNTIME_QUERY 0
#endif

#if VX_CPU_X86_RUNTIME_QUERY
/* Keep XGETBV in the baseline translation unit without requiring a global
 * -mxsave flag.  The caller executes it only after CPUID reports OSXSAVE. */
static uint64_t vx_cpu_xgetbv0(void) {
    unsigned int eax, edx;
    __asm__ volatile("xgetbv" : "=a"(eax), "=d"(edx) : "c"(0));
    return (uint64_t)eax | ((uint64_t)edx << 32u);
}
#endif

/* CPUID is serializing and, under a hypervisor, traps to the VMM.  Kernels ask
 * these questions on every dispatch, so each answer is decided once and then
 * read from memory.  The probes are pure functions of the CPU, so a benign race
 * between first callers computes the same value twice and stores it twice. */
#define VX_CPU_FEATURE_UNKNOWN (-1)

int vx_cpu_has_avx2(void) {
#if VX_CPU_X86_RUNTIME_QUERY
    /* __builtin_cpu_supports already reads a resolved global, so this only
     * needs the one-time __builtin_cpu_init. */
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx2");
#else
    return 0;
#endif
}

static int vx_cpu_probe_avx_vnni(void) {
#if VX_CPU_X86_RUNTIME_QUERY
    unsigned int eax, ebx, ecx, edx;
    if (!vx_cpu_has_avx2() || (unsigned int)__get_cpuid_max(0, 0) < 7u) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    if (eax < 1u) return 0;
    __cpuid_count(7, 1, eax, ebx, ecx, edx);
    return (eax & (1u << 4u)) != 0;
#else
    return 0;
#endif
}

int vx_cpu_has_avx_vnni(void) {
    static int cached = VX_CPU_FEATURE_UNKNOWN;
    int value = cached;
    if (value == VX_CPU_FEATURE_UNKNOWN) {
        value = vx_cpu_probe_avx_vnni();
        cached = value;
    }
    return value;
}

/* AVX-512F plus the VL/BW/DQ pieces a float kernel needs, without VNNI.
 * VNNI is an integer dot product, so gating an F32 kernel on it would exclude
 * every AVX-512 machine that predates it — Skylake-X has AVX-512F and no VNNI
 * — while implying a capability the kernel never uses. */
static int vx_cpu_probe_avx512f(void) {
#if VX_CPU_X86_RUNTIME_QUERY
    unsigned int eax, ebx, ecx, edx;
    const unsigned int required_ebx = (1u << 5u) |  /* AVX2 */
        (1u << 16u) | /* AVX-512F */
        (1u << 17u) | /* AVX-512DQ */
        (1u << 30u) | /* AVX-512BW */
        (1u << 31u);  /* AVX-512VL */
    const uint64_t required_xcr0 = (1ull << 1u) | (1ull << 2u) |
        (1ull << 5u) | (1ull << 6u) | (1ull << 7u);
    if ((unsigned int)__get_cpuid_max(0, 0) < 7u) return 0;
    __cpuid(1, eax, ebx, ecx, edx);
    if ((ecx & (1u << 28u)) == 0 || (ecx & (1u << 27u)) == 0 ||
        (ecx & (1u << 26u)) == 0 ||
        (vx_cpu_xgetbv0() & required_xcr0) != required_xcr0) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & required_ebx) == required_ebx;
#else
    return 0;
#endif
}

int vx_cpu_has_avx512f(void) {
    static int cached = VX_CPU_FEATURE_UNKNOWN;
    int value = cached;
    if (value == VX_CPU_FEATURE_UNKNOWN) {
        value = vx_cpu_probe_avx512f();
        cached = value;
    }
    return value;
}

static int vx_cpu_probe_avx512_vnni(void) {
#if VX_CPU_X86_RUNTIME_QUERY
    unsigned int eax, ebx, ecx, edx;
    const unsigned int required_ebx = (1u << 5u) |  /* AVX2 */
        (1u << 16u) | /* AVX-512F */
        (1u << 30u) | /* AVX-512BW */
        (1u << 31u);  /* AVX-512VL */
    const uint64_t required_xcr0 = (1ull << 1u) | (1ull << 2u) |
        (1ull << 5u) | (1ull << 6u) | (1ull << 7u);
    if ((unsigned int)__get_cpuid_max(0, 0) < 7u) return 0;
    __cpuid(1, eax, ebx, ecx, edx);
    /* AVX, XSAVE, and OSXSAVE are prerequisites for safely querying/restoring
     * extended vector state. XMM/YMM, opmask, ZMM-hi256 and hi16-ZMM must all
     * be enabled by the OS before any target-attributed function is called. */
    if ((ecx & (1u << 28u)) == 0 || (ecx & (1u << 27u)) == 0 ||
        (ecx & (1u << 26u)) == 0 ||
        (vx_cpu_xgetbv0() & required_xcr0) != required_xcr0) return 0;
    __cpuid_count(7, 0, eax, ebx, ecx, edx);
    return (ebx & required_ebx) == required_ebx &&
        (ecx & (1u << 11u)) != 0; /* AVX-512 VNNI */
#else
    return 0;
#endif
}

int vx_cpu_has_avx512_vnni(void) {
    static int cached = VX_CPU_FEATURE_UNKNOWN;
    int value = cached;
    if (value == VX_CPU_FEATURE_UNKNOWN) {
        value = vx_cpu_probe_avx512_vnni();
        cached = value;
    }
    return value;
}

static int vx_cpu_probe_neon(void) {
#if VX_CPU_ARM_RUNTIME_QUERY
    return (getauxval(AT_HWCAP) & HWCAP_ASIMD) != 0;
#elif defined(__ARM_NEON) || defined(__ARM_NEON__)
    return 1;
#else
    return 0;
#endif
}

int vx_cpu_has_neon(void) {
    static int cached = VX_CPU_FEATURE_UNKNOWN;
    int value = cached;
    if (value == VX_CPU_FEATURE_UNKNOWN) {
        value = vx_cpu_probe_neon();
        cached = value;
    }
    return value;
}

static int vx_cpu_probe_arm_dotprod(void) {
#if VX_CPU_ARM_RUNTIME_QUERY
    return (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0;
#elif defined(__ARM_FEATURE_DOTPROD)
    return 1;
#else
    return 0;
#endif
}

int vx_cpu_has_arm_dotprod(void) {
    static int cached = VX_CPU_FEATURE_UNKNOWN;
    int value = cached;
    if (value == VX_CPU_FEATURE_UNKNOWN) {
        value = vx_cpu_probe_arm_dotprod();
        cached = value;
    }
    return value;
}
