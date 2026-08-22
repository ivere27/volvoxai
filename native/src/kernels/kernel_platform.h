#ifndef VOLVOXAI_KERNEL_PLATFORM_H
#define VOLVOXAI_KERNEL_PLATFORM_H

/*
 * One resolved description of the CPU the kernels will run on.
 *
 * Kernels used to ask `vx_cpu_has_*()` inline at every dispatch, with the ISA
 * selection thresholds written as literals beside each call.  That had three
 * costs.  The policy for a given operation was spread across the file that
 * happened to implement it, so there was no place to read "what runs where".
 * Adding an ISA meant editing every dispatcher.  And on x86 the AVX-VNNI and
 * AVX-512-VNNI probes execute CPUID, which is serializing and traps to the VMM
 * under a hypervisor, so the questions were not free even though the answers
 * cannot change while the process runs.
 *
 * This header resolves the answers once and names the thresholds, so a
 * dispatcher reads a field instead of probing, and a new ISA is a tier plus its
 * thresholds rather than an edit in every kernel.
 *
 * It is header-only on purpose: the kernels are linked into roughly thirty
 * different test, benchmark and release targets, and a new translation unit
 * would have to be added to each one.  The cache is function-local, so each
 * translation unit resolves at most once.  CPU capabilities and the clamp are
 * process-stable; the cached SVE length is only a diagnostic snapshot because
 * Linux permits each thread to change its vector length.  A future SVE kernel
 * must query vx_cpu_arm_sve_vl_bytes() in the dispatching thread.
 *
 * `VOLVOXAI_CPU_ISA` clamps the tier downward.  The x86 ladder is `baseline`,
 * `avx2`, `avxvnni`, `avx512vnni`; the Arm ladder is `baseline`, `neon`,
 * `neondotprod`, `neoni8mm`, `sve2`.  The ladders are deliberately resolved by
 * architecture rather than compared as one numeric enum: requesting an Arm
 * name on x86 (or an x86 name on Arm) selects baseline instead of accidentally
 * admitting a similarly numbered capability.
 */

#include "cpu_features.h"

#include <stdint.h>
#if defined(__wasm__)
/* The freestanding wasm32 inference build has no <stdlib.h> or <string.h>, and
 * no environment to read.  Every vx_cpu_has_* is already 0 there, so the clamp
 * has nothing to remove and the tier is always baseline. */
#else
#include <stdlib.h>
#include <string.h>
#endif

typedef enum {
    VX_KERNEL_ISA_INVALID = -2,
    VX_KERNEL_ISA_AUTO = -1,
    VX_KERNEL_ISA_BASELINE = 0,
    VX_KERNEL_ISA_NEON = 1,
    VX_KERNEL_ISA_NEON_DOTPROD = 2,
    VX_KERNEL_ISA_NEON_I8MM = 3,
    VX_KERNEL_ISA_SVE2 = 4,
    VX_KERNEL_ISA_AVX2 = 5,
    VX_KERNEL_ISA_AVX_VNNI = 6,
    VX_KERNEL_ISA_AVX512_VNNI = 7
} VxKernelIsa;

typedef enum {
    VX_KERNEL_ARCH_OTHER = 0,
    VX_KERNEL_ARCH_ARM = 1,
    VX_KERNEL_ARCH_X86 = 2
} VxKernelArchitecture;

/* Raw capability input is named so the resolver can be tested with an Arm CPU
 * description on an x86 build (and vice versa).  Runtime callers use
 * vx_kernel_platform_resolve(), which fills this from cpu_features.c. */
typedef struct {
    int has_avx2;
    int has_avx_vnni;
    int has_avx512f;
    int has_avx512_vnni;
    int has_neon;
    int has_arm_dotprod;
    int has_arm_i8mm;
    int has_arm_sve;
    int has_arm_sve2;
    int has_arm_sve_i8mm;
    uint32_t arm_sve_vl_bytes;
} VxKernelCpuCapabilities;

typedef struct {
    /* The explicit operator request and whether this host can honor it.  AUTO
     * and BASELINE are always valid; an unknown, cross-architecture, or
     * unavailable tier is a configuration error rather than a soft clamp. */
    VxKernelIsa requested_isa;
    int configuration_valid;

    /* Highest tier this process will use, after any clamp. */
    VxKernelIsa isa;

    /* Precomputed tier predicates.  A dispatcher tests these instead of
     * probing, and they already reflect the clamp. */
    int has_avx2;
    int has_avx_vnni;
    /* Vector width, not a tier.  The ladder above orders the integer dot
     * product kernels, where AVX-512 VNNI is the top rung; an F32 kernel needs
     * AVX-512F and gets nothing from VNNI, and the two are independent in
     * hardware.  So this is a separate predicate and the reported isa name
     * still describes the integer tier. */
    int has_avx512f;
    int has_avx512_vnni;
    int has_neon;
    int has_arm_dotprod;
    int has_arm_i8mm;
    int has_arm_sve;
    int has_arm_sve2;
    int has_arm_sve_i8mm;
    /* Snapshot for the thread which resolved this structure.  SVE kernels
     * whose callers may alter vector length must query
     * vx_cpu_arm_sve_vl_bytes() again at dispatch. */
    uint32_t arm_sve_vl_bytes;

    /*
     * ISA selection thresholds, named rather than inlined as literals.
     *
     * The VNNI kernels amortize their wider accumulator setup over the K loop,
     * so they only win once the reduction is long enough; below the threshold
     * the narrower kernel is faster despite the weaker instruction.  These are
     * the values the dispatchers previously hard-coded; they are gathered here
     * so a per-microarchitecture retune is one edit and can be measured with
     * benchmark_kernel_unit.
     */
    uint32_t qlinear_avx512_vnni_min_d_in;
    uint32_t qlinear_avx_vnni_min_d_in;
    uint32_t qlinear_maddubs_min_d_in;
    uint32_t qconv_avx512_vnni_min_input_per_group;

    /*
     * Largest scalar remainder the AVX-512 QLinear body may be given, as a
     * divisor of the reduction length: a remainder above d_in/N sends the tier
     * to the narrower kernel instead.
     *
     * The 512-bit body consumes 64 bytes per iteration and hands whatever is
     * left to a scalar loop, while the AVX2 body steps 32 bytes.  A reduction
     * that is a multiple of 32 but not of 64 therefore runs tail-free on AVX2
     * and with a 32-element scalar tail on AVX-512.  Measured on Tiger Lake
     * (i3-1115G4, M=402 N=320): at d_in=96 that costs 2.12 versus 5.98 GMAC/s,
     * so the wider tier is 2.8x slower, and at d_in=160 it is 3.48 versus 9.07.
     * With a tail-free reduction the tier wins as expected, from 1.06x at
     * d_in=192 up to 1.77x at d_in=1280.
     */
    uint32_t qlinear_avx512_vnni_tail_budget;
} VxKernelPlatform;

/* AVX-512 QLinear admission: long enough reduction, and a remainder small
 * enough that the scalar tail does not eat the tier's advantage. */
static inline int vx_kernel_qlinear_prefers_avx512_vnni(
        const VxKernelPlatform* platform, uint32_t d_in) {
    if (!platform->has_avx512_vnni) return 0;
    if (d_in < platform->qlinear_avx512_vnni_min_d_in) return 0;
    {
        const uint32_t tail = d_in % 64u;
        return tail == 0u ||
               tail * platform->qlinear_avx512_vnni_tail_budget <= d_in;
    }
}

static inline const char* vx_kernel_isa_name(VxKernelIsa isa) {
    switch (isa) {
        case VX_KERNEL_ISA_AVX512_VNNI: return "avx512vnni";
        case VX_KERNEL_ISA_AVX_VNNI:    return "avxvnni";
        case VX_KERNEL_ISA_AVX2:        return "avx2";
        case VX_KERNEL_ISA_SVE2:        return "sve2";
        case VX_KERNEL_ISA_NEON_I8MM:   return "neoni8mm";
        case VX_KERNEL_ISA_NEON_DOTPROD:return "neondotprod";
        case VX_KERNEL_ISA_NEON:        return "neon";
        case VX_KERNEL_ISA_AUTO:        return "auto";
        case VX_KERNEL_ISA_INVALID:     return "invalid";
        default:                        return "baseline";
    }
}

/* Parse the clamp request. An unset/empty value means AUTO. Unknown values are
 * explicit configuration failures; silently treating a typo as AUTO would run
 * a different ISA than the operator requested. */
static inline VxKernelIsa vx_kernel_isa_clamp_request(void) {
#if defined(__wasm__)
    return VX_KERNEL_ISA_AUTO;  /* no environment and nothing to clamp */
#else
    const char* requested = getenv("VOLVOXAI_CPU_ISA");
    if (!requested || !requested[0] || !strcmp(requested, "auto"))
        return VX_KERNEL_ISA_AUTO;
    if (!strcmp(requested, "baseline"))    return VX_KERNEL_ISA_BASELINE;
    if (!strcmp(requested, "neon"))        return VX_KERNEL_ISA_NEON;
    if (!strcmp(requested, "neondotprod")) return VX_KERNEL_ISA_NEON_DOTPROD;
    if (!strcmp(requested, "neoni8mm"))    return VX_KERNEL_ISA_NEON_I8MM;
    if (!strcmp(requested, "sve2"))        return VX_KERNEL_ISA_SVE2;
    if (!strcmp(requested, "avx2"))        return VX_KERNEL_ISA_AVX2;
    if (!strcmp(requested, "avxvnni"))     return VX_KERNEL_ISA_AVX_VNNI;
    if (!strcmp(requested, "avx512vnni"))  return VX_KERNEL_ISA_AVX512_VNNI;
    return VX_KERNEL_ISA_INVALID;
#endif
}

static inline VxKernelArchitecture vx_kernel_architecture(void) {
#if defined(__i386__) || defined(__x86_64__)
    return VX_KERNEL_ARCH_X86;
#elif defined(__aarch64__) || defined(__arm__)
    return VX_KERNEL_ARCH_ARM;
#else
    return VX_KERNEL_ARCH_OTHER;
#endif
}

static inline void vx_kernel_platform_resolve_capabilities(
        VxKernelPlatform* platform, VxKernelArchitecture architecture,
        VxKernelIsa ceiling, const VxKernelCpuCapabilities* raw) {
    int allow_avx2 = 0, allow_avx_vnni = 0, allow_avx512 = 0;
    int allow_neon = 0, allow_dotprod = 0, allow_i8mm = 0;
    int allow_sve = 0, allow_sve2 = 0, allow_sve_i8mm = 0;

    platform->requested_isa = ceiling;
    platform->configuration_valid = 0;
    platform->isa = VX_KERNEL_ISA_BASELINE;
    platform->has_avx2 = 0;
    platform->has_avx_vnni = 0;
    platform->has_avx512f = 0;
    platform->has_avx512_vnni = 0;
    platform->has_neon = 0;
    platform->has_arm_dotprod = 0;
    platform->has_arm_i8mm = 0;
    platform->has_arm_sve = 0;
    platform->has_arm_sve2 = 0;
    platform->has_arm_sve_i8mm = 0;
    platform->arm_sve_vl_bytes = 0u;
    if (ceiling == VX_KERNEL_ISA_AUTO ||
        ceiling == VX_KERNEL_ISA_BASELINE) {
        platform->configuration_valid = 1;
    } else if (architecture == VX_KERNEL_ARCH_X86) {
        platform->configuration_valid =
            (ceiling == VX_KERNEL_ISA_AVX2 && raw->has_avx2) ||
            (ceiling == VX_KERNEL_ISA_AVX_VNNI && raw->has_avx2 &&
             raw->has_avx_vnni) ||
            (ceiling == VX_KERNEL_ISA_AVX512_VNNI && raw->has_avx2 &&
             raw->has_avx512_vnni);
    } else if (architecture == VX_KERNEL_ARCH_ARM) {
        platform->configuration_valid =
            (ceiling == VX_KERNEL_ISA_NEON && raw->has_neon) ||
            (ceiling == VX_KERNEL_ISA_NEON_DOTPROD && raw->has_neon &&
             raw->has_arm_dotprod) ||
            (ceiling == VX_KERNEL_ISA_NEON_I8MM && raw->has_neon &&
             raw->has_arm_i8mm) ||
            (ceiling == VX_KERNEL_ISA_SVE2 && raw->has_arm_sve &&
             raw->has_arm_sve2);
    }
    if (architecture == VX_KERNEL_ARCH_X86) {
        switch (ceiling) {
            case VX_KERNEL_ISA_AUTO:
            case VX_KERNEL_ISA_AVX512_VNNI:
                allow_avx512 = 1;
                /* fall through */
            case VX_KERNEL_ISA_AVX_VNNI:
                allow_avx_vnni = 1;
                /* fall through */
            case VX_KERNEL_ISA_AVX2:
                allow_avx2 = 1;
                break;
            default:
                break;
        }
    } else if (architecture == VX_KERNEL_ARCH_ARM) {
        switch (ceiling) {
            case VX_KERNEL_ISA_AUTO:
            case VX_KERNEL_ISA_SVE2:
                allow_sve = 1;
                allow_sve2 = 1;
                allow_sve_i8mm = 1;
                /* fall through */
            case VX_KERNEL_ISA_NEON_I8MM:
                allow_i8mm = 1;
                /* fall through */
            case VX_KERNEL_ISA_NEON_DOTPROD:
                allow_dotprod = 1;
                /* fall through */
            case VX_KERNEL_ISA_NEON:
                allow_neon = 1;
                break;
            default:
                break;
        }
    }

    platform->has_avx2 = allow_avx2 && raw->has_avx2;
    platform->has_avx_vnni = allow_avx_vnni && platform->has_avx2 &&
        raw->has_avx_vnni;
    platform->has_avx512f = allow_avx512 && platform->has_avx2 &&
        raw->has_avx512f;
    platform->has_avx512_vnni = allow_avx512 && platform->has_avx2 &&
        raw->has_avx512_vnni;
    platform->has_neon = allow_neon && raw->has_neon;
    platform->has_arm_dotprod = allow_dotprod && platform->has_neon &&
        raw->has_arm_dotprod;
    platform->has_arm_i8mm = allow_i8mm && platform->has_neon &&
        raw->has_arm_i8mm;
    platform->has_arm_sve = allow_sve && raw->has_arm_sve;
    platform->has_arm_sve2 = allow_sve2 && platform->has_arm_sve &&
        raw->has_arm_sve2;
    platform->has_arm_sve_i8mm = allow_sve_i8mm && platform->has_arm_sve &&
        raw->has_arm_sve_i8mm;
    platform->arm_sve_vl_bytes = platform->has_arm_sve
        ? raw->arm_sve_vl_bytes : 0u;

    if (platform->has_avx512_vnni)      platform->isa = VX_KERNEL_ISA_AVX512_VNNI;
    else if (platform->has_avx_vnni)    platform->isa = VX_KERNEL_ISA_AVX_VNNI;
    else if (platform->has_avx2)        platform->isa = VX_KERNEL_ISA_AVX2;
    else if (platform->has_arm_sve2)    platform->isa = VX_KERNEL_ISA_SVE2;
    else if (platform->has_arm_i8mm)    platform->isa = VX_KERNEL_ISA_NEON_I8MM;
    else if (platform->has_arm_dotprod) platform->isa = VX_KERNEL_ISA_NEON_DOTPROD;
    else if (platform->has_neon)        platform->isa = VX_KERNEL_ISA_NEON;
    else                                platform->isa = VX_KERNEL_ISA_BASELINE;

    /* Measured crossover on Tiger Lake, not the instruction's minimum: the tier
     * is only ahead from a tail-free d_in of 192 upward. */
    platform->qlinear_avx512_vnni_min_d_in = 192u;
    platform->qlinear_avx_vnni_min_d_in = 32u;
    platform->qlinear_maddubs_min_d_in = 32u;
    platform->qconv_avx512_vnni_min_input_per_group = 64u;
    platform->qlinear_avx512_vnni_tail_budget = 8u;
}

static inline void vx_kernel_platform_resolve(VxKernelPlatform* platform) {
    const VxKernelCpuCapabilities raw = {
        .has_avx2 = vx_cpu_has_avx2(),
        .has_avx_vnni = vx_cpu_has_avx_vnni(),
        .has_avx512f = vx_cpu_has_avx512f(),
        .has_avx512_vnni = vx_cpu_has_avx512_vnni(),
        .has_neon = vx_cpu_has_neon(),
        .has_arm_dotprod = vx_cpu_has_arm_dotprod(),
        .has_arm_i8mm = vx_cpu_has_arm_i8mm(),
        .has_arm_sve = vx_cpu_has_arm_sve(),
        .has_arm_sve2 = vx_cpu_has_arm_sve2(),
        .has_arm_sve_i8mm = vx_cpu_has_arm_sve_i8mm(),
        .arm_sve_vl_bytes = vx_cpu_arm_sve_vl_bytes(),
    };
    vx_kernel_platform_resolve_capabilities(platform,
        vx_kernel_architecture(), vx_kernel_isa_clamp_request(), &raw);
}

static inline const VxKernelPlatform* vx_kernel_platform(void) {
    static VxKernelPlatform platform;
    static int resolved;
    if (!resolved) {
        vx_kernel_platform_resolve(&platform);
        resolved = 1;
    }
    return &platform;
}

#endif
