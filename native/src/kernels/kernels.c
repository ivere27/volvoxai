/* thread_pool.c is amalgamated below and queries CPU affinity through the GNU
 * cpu_set_t interface.  Its own guard is too late in this translation unit,
 * because the includes above it have already pulled in features.h. */
#if !defined(__wasm__) && !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

#include "core.c"
#include "fast_math.c"
#include "gemm_f32.c"
#include "matmul.c"
#include "layernorm.c"
#include "activations.c"
#include "sdpa.c"
#include "cross_sdpa.c"
#include "cross_attention.c"
#include "embedding.c"
#ifndef __wasm__
#include "lora_linear.c"
#endif
#include "thread_pool.c"
#include "vision_ops.c"
#include "edge_primitives.c"
#include "math_nlp.c"
#include "cv_nlp.c"
#include "misc_ops.c"
#include "shape_math.c"
#include "broadcast_ops.c"
#include "fusion_ops.c"
#include "portable_inference_kernels.c"
#if defined(__wasm_simd128__)
#include "qbatch_matmul_wasm_simd.c"
#endif
#include "packed_quant_gemm.c"
#include "sequence_ops.c"
