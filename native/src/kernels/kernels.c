/* thread_pool.c is amalgamated below and queries CPU affinity through the GNU
 * cpu_set_t interface.  Its own guard is too late in this translation unit,
 * because the includes above it have already pulled in features.h. */
#if !defined(__wasm__) && !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif

/* The include order is the definition order of one translation unit: a
 * fragment may only use what the fragments above it already defined. */
#include "wasm_heap_arena.c"
#include "fast_math.inc"
#include "gemm_f32.c"
#include "matmul.inc"
#include "layernorm.inc"
#include "activations.inc"
#include "sdpa.inc"
#include "cross_sdpa.inc"
#include "cross_attention.inc"
#include "embedding.inc"
#ifndef __wasm__
#include "lora_linear.inc"
#endif
#include "thread_pool.c"
#include "vision_ops.inc"
#include "activations_hard.inc"
#include "tensor_basic_ops.inc"
#include "elementwise_f32.inc"
#include "softmax_rmsnorm.inc"
#include "prelu_logsoftmax.inc"
#include "reduction_argmax.inc"
#include "non_max_suppression.inc"
#include "pool_gather_1d.inc"
#include "select_where_clip.inc"
#include "quantize_linear_ops.inc"
#include "conv_transpose2d.inc"
#include "shape_math.inc"
#include "broadcast_ops.inc"
#include "fusion_ops.inc"
#include "portable_inference_kernels.inc"
#if defined(__wasm_simd128__)
#include "qbatch_matmul_wasm_simd.inc"
#endif
#include "packed_quant_gemm.c"
#include "sequence_ops.c"
