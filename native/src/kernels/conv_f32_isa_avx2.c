/*
 * The AVX2 copy of the FP32 convolution kernels.
 *
 * The build gives this one file -mavx2 -mfma (see the
 * set_source_files_properties call in CMakeLists.txt), which is what makes the
 * `#if defined(__AVX2__)` bodies inside conv_f32_isa.c live here even when the
 * rest of the binary is built for a baseline CPU.  conv_f32_dispatch.c calls
 * into it only after vx_kernel_platform() confirms the CPU has AVX2, so adding
 * the flag to this translation unit cannot raise the binary's CPU requirement.
 *
 * Nothing here may be called unconditionally, and nothing else in the build may
 * be given these flags.
 */
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))

#define VX_CONV_F32_ISA_SUFFIX _avx2
#include "conv_f32_isa.c"

#else
/* Non-x86 builds have no second tier to offer; conv_f32_dispatch.c drops the
 * AVX2 branch on these targets.  C requires a non-empty translation unit. */
typedef int vx_conv_f32_isa_avx2_unused;
#endif
