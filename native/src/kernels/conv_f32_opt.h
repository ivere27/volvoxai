#ifndef VOLVOXAI_CONV_F32_OPT_H
#define VOLVOXAI_CONV_F32_OPT_H

/*
 * FP32 convolution kernels, selected by ISA at runtime.
 *
 * conv_f32_opt.c contains hand-written AVX2 bodies behind `#if
 * defined(__AVX2__)`.  That is a compile-time question, so a binary configured
 * at VOLVOXAI_CPU_TARGET=baseline — the default — had the AVX2 code removed by
 * the preprocessor and ran the scalar fallback on machines that support AVX2.
 * Measured on a Ryzen 5 5600U at one thread, that cost 23.5x on a representative
 * thirteen-node image encoder, which is 88% of its FP32 time.
 *
 * The fix keeps the bodies untouched and compiles the file twice: once at the
 * build's baseline ISA, once with -mavx2 -mfma.  Each copy's exported symbols
 * get an ISA suffix, and conv_f32_dispatch.c owns the unsuffixed names and
 * forwards to whichever copy the CPU supports.  This mirrors the existing
 * volvox_arm_dotprod object library, which already builds Armv8.2 SDOT bodies
 * as a separately-attributed translation unit.
 *
 * Duplicating the translation unit rather than annotating each body with
 * __attribute__((target("avx2,fma"))) is deliberate: the file's vector code
 * lives in `static inline` helpers and in blocks nested inside functions that
 * also hold the scalar fallback, so the attribute form would mean restructuring
 * tuned intrinsics, and a target-attributed function cannot be inlined into an
 * unattributed caller.  Compiling the whole unit at the higher ISA keeps
 * inlining intact and leaves the kernel source as the single authority.
 *
 * Callers include this header and call the plain names; they do not choose.
 * Both copies write the same packed layouts into the same engine-state caches,
 * and vx_kernel_platform() resolves once per process, so a process only ever
 * populates and reads one copy's results.
 */

/* VX_CONV_F32_ISA_SUFFIX is set by the build for each copy.  Unset means this
 * is the dispatcher (or a consumer), which wants the unsuffixed names. */
#if defined(VX_CONV_F32_ISA_SUFFIX)
#define VX_CONV_F32_JOIN_(a, b) a##b
#define VX_CONV_F32_JOIN(a, b) VX_CONV_F32_JOIN_(a, b)
#define VX_CONV_F32_SYM(name) VX_CONV_F32_JOIN(name, VX_CONV_F32_ISA_SUFFIX)

/* Renaming through the preprocessor keeps conv_f32_opt.c free of any knowledge
 * that it is built more than once. */
#define vx_pwf32_pack_cache VX_CONV_F32_SYM(vx_pwf32_pack_cache)
#define vx_conv2d_pointwise_f32 VX_CONV_F32_SYM(vx_conv2d_pointwise_f32)
#define vx_conv2d_dw3x3s1_f32 VX_CONV_F32_SYM(vx_conv2d_dw3x3s1_f32)
#define vx_conv2d_dw5x5s1_f32 VX_CONV_F32_SYM(vx_conv2d_dw5x5s1_f32)
#define vx_conv2d_depthwise_f32 VX_CONV_F32_SYM(vx_conv2d_depthwise_f32)
#define vx_conv2d_generic_f32 VX_CONV_F32_SYM(vx_conv2d_generic_f32)
#define vx_conv2d_depthwise_pointwise_f32 \
    VX_CONV_F32_SYM(vx_conv2d_depthwise_pointwise_f32)
#define vx_f32_igemm_indirection_cache \
    VX_CONV_F32_SYM(vx_f32_igemm_indirection_cache)
#define vx_f32_igemm_pack_cache VX_CONV_F32_SYM(vx_f32_igemm_pack_cache)
#define vx_conv2d_spatial_igemm_f32 VX_CONV_F32_SYM(vx_conv2d_spatial_igemm_f32)
#define vx_conv_f32_opt_free_all VX_CONV_F32_SYM(vx_conv_f32_opt_free_all)
#endif

/* Declare the names this translation unit uses — suffixed inside either copy,
 * plain everywhere else. */
#define VX_CONV_F32_ENTRY(ret, name, params, args) ret name params;
#define VX_CONV_F32_ENTRY_VOID(ret, name, params, args) ret name params;
#include "conv_f32_opt_entrypoints.h"

#endif
