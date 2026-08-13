/*
 * The baseline-ISA copy of the FP32 convolution kernels.
 *
 * Compiled with whatever CPU flags the build's VOLVOXAI_CPU_TARGET selects, so
 * on the default `baseline` target this is the portable scalar path and remains
 * what runs on a CPU without AVX2.  conv_f32_isa_avx2.c is the same source at a
 * higher ISA; conv_f32_dispatch.c chooses between them at runtime.
 *
 * Both copies are ordinary members of KERNEL_SRCS rather than a shared OBJECT
 * library: conv_f32_isa.c reaches into VxEngineState, whose layout depends on
 * VOLVOXAI_ENABLE_TRAINING, and each target sets that macro itself.  Compiling
 * per target keeps every copy's struct offsets in agreement with its consumer.
 */
#define VX_CONV_F32_ISA_SUFFIX _baseline
#include "conv_f32_isa.c"
