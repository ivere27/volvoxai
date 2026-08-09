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
 * translation unit resolves at most once and the result is identical because
 * it is a pure function of the CPU and one environment variable.
 *
 * `VOLVOXAI_CPU_ISA` clamps the tier downward: `baseline`, `avx2`, `avxvnni`,
 * `avx512vnni`, `neon`, `neondotprod`.  Clamping only ever removes capability,
 * so it cannot ask for an instruction the CPU lacks.  This is what makes it
 * possible to exercise and benchmark the AVX2 path on a VNNI machine, which
 * previously required a rebuild.
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
    VX_KERNEL_ISA_BASELINE = 0,
    VX_KERNEL_ISA_NEON = 1,
    VX_KERNEL_ISA_NEON_DOTPROD = 2,
    VX_KERNEL_ISA_AVX2 = 3,
    VX_KERNEL_ISA_AVX_VNNI = 4,
    VX_KERNEL_ISA_AVX512_VNNI = 5
} VxKernelIsa;

typedef struct {
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
        case VX_KERNEL_ISA_NEON_DOTPROD:return "neondotprod";
        case VX_KERNEL_ISA_NEON:        return "neon";
        default:                        return "baseline";
    }
}

/* Parse the clamp request.  An unset or unrecognized value means "no clamp",
 * which keeps a typo from silently disabling every vector kernel. */
static inline VxKernelIsa vx_kernel_isa_clamp_request(void) {
#if defined(__wasm__)
    return VX_KERNEL_ISA_AVX512_VNNI;  /* no clamp; nothing to clamp */
#else
    const char* requested = getenv("VOLVOXAI_CPU_ISA");
    if (!requested || !requested[0]) return VX_KERNEL_ISA_AVX512_VNNI;
    if (!strcmp(requested, "baseline"))    return VX_KERNEL_ISA_BASELINE;
    if (!strcmp(requested, "neon"))        return VX_KERNEL_ISA_NEON;
    if (!strcmp(requested, "neondotprod")) return VX_KERNEL_ISA_NEON_DOTPROD;
    if (!strcmp(requested, "avx2"))        return VX_KERNEL_ISA_AVX2;
    if (!strcmp(requested, "avxvnni"))     return VX_KERNEL_ISA_AVX_VNNI;
    if (!strcmp(requested, "avx512vnni"))  return VX_KERNEL_ISA_AVX512_VNNI;
    return VX_KERNEL_ISA_AVX512_VNNI;
#endif
}

static inline void vx_kernel_platform_resolve(VxKernelPlatform* platform) {
    const VxKernelIsa ceiling = vx_kernel_isa_clamp_request();
    /* x86 and ARM tiers are disjoint, so one ordered enum can carry both: a
     * clamp naming the other architecture's tier simply leaves this one at
     * baseline, which is the conservative outcome. */
    const int allow_avx2 = ceiling >= VX_KERNEL_ISA_AVX2;
    const int allow_avx_vnni = ceiling >= VX_KERNEL_ISA_AVX_VNNI;
    const int allow_avx512_vnni = ceiling >= VX_KERNEL_ISA_AVX512_VNNI;
    const int allow_neon = ceiling == VX_KERNEL_ISA_NEON ||
        ceiling >= VX_KERNEL_ISA_NEON_DOTPROD;
    const int allow_dotprod = ceiling >= VX_KERNEL_ISA_NEON_DOTPROD;

    platform->has_avx2 = allow_avx2 && vx_cpu_has_avx2();
    platform->has_avx_vnni = allow_avx_vnni && platform->has_avx2 &&
        vx_cpu_has_avx_vnni();
    platform->has_avx512f = allow_avx512_vnni && platform->has_avx2 &&
        vx_cpu_has_avx512f();
    platform->has_avx512_vnni = allow_avx512_vnni && platform->has_avx2 &&
        vx_cpu_has_avx512_vnni();
    platform->has_neon = allow_neon && vx_cpu_has_neon();
    platform->has_arm_dotprod = allow_dotprod && platform->has_neon &&
        vx_cpu_has_arm_dotprod();

    if (platform->has_avx512_vnni)      platform->isa = VX_KERNEL_ISA_AVX512_VNNI;
    else if (platform->has_avx_vnni)    platform->isa = VX_KERNEL_ISA_AVX_VNNI;
    else if (platform->has_avx2)        platform->isa = VX_KERNEL_ISA_AVX2;
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
