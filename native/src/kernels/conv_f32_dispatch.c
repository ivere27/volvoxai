/*
 * Runtime ISA selection for the FP32 convolution kernels.
 *
 * Owns the unsuffixed conv entry points and forwards each to the highest-ISA
 * copy of conv_f32_opt.c the CPU actually supports.  See conv_f32_opt.h for why
 * the kernels are built as one translation unit per ISA.
 *
 * The choice is a single load from the resolved platform record, so it costs a
 * branch per conv node — against convolutions measured in hundreds of
 * microseconds each.  Honouring VOLVOXAI_CPU_ISA comes for free through
 * vx_kernel_platform(), so `VOLVOXAI_CPU_ISA=baseline` now reproduces the old
 * scalar behaviour from the same binary, which is what makes the two paths
 * comparable in benchmark_kernel_unit without a rebuild.
 */
#include "conv_f32_opt.h"
#include "kernel_platform.h"

/* Both copies exist in every x86 binary; on other architectures only the
 * baseline copy is built and the AVX2 branch is never reachable. */
#if (defined(__i386__) || defined(__x86_64__)) && \
    (defined(__clang__) || defined(__GNUC__))
#define VX_CONV_F32_HAVE_AVX2_COPY 1
#else
#define VX_CONV_F32_HAVE_AVX2_COPY 0
#endif

#define VX_CONV_F32_ENTRY(ret, name, params, args) ret name##_baseline params;
#define VX_CONV_F32_ENTRY_VOID(ret, name, params, args) ret name##_baseline params;
#include "conv_f32_opt_entrypoints.h"

#if VX_CONV_F32_HAVE_AVX2_COPY
#define VX_CONV_F32_ENTRY(ret, name, params, args) ret name##_avx2 params;
#define VX_CONV_F32_ENTRY_VOID(ret, name, params, args) ret name##_avx2 params;
#include "conv_f32_opt_entrypoints.h"
#endif

#if VX_CONV_F32_HAVE_AVX2_COPY
#define VX_CONV_F32_ENTRY(ret, name, params, args)             \
    ret name params {                                          \
        return vx_kernel_platform()->has_avx2 ? name##_avx2 args \
                                              : name##_baseline args; \
    }
#define VX_CONV_F32_ENTRY_VOID(ret, name, params, args)        \
    ret name params {                                          \
        if (vx_kernel_platform()->has_avx2) name##_avx2 args;  \
        else name##_baseline args;                             \
    }
#else
#define VX_CONV_F32_ENTRY(ret, name, params, args) \
    ret name params { return name##_baseline args; }
#define VX_CONV_F32_ENTRY_VOID(ret, name, params, args) \
    ret name params { name##_baseline args; }
#endif
#include "conv_f32_opt_entrypoints.h"
