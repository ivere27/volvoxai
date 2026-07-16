#ifndef VOLVOXAI_GEMM_F32_H
#define VOLVOXAI_GEMM_F32_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Dense F32 GEMM uses K-major, NR-interleaved B panels.  The tile policy is
 * deliberately an internal kernel contract, not a public engine ABI: it may
 * be tuned without changing model files or installed headers.
 *
 * The default assumes a 32 KiB L1D and budgets 75% of it for one A/B tile,
 * leaving headroom for stack data and cache conflicts. Native builds detect
 * the cache once through the host OS (sysconf on Linux/Android, sysctl on
 * macOS); WASM and failed/unsupported queries use the conservative 32 KiB
 * fallback. Cache policy is entirely automatic and has no user tuning knob.
 */
typedef struct {
    uint32_t mr;
    uint32_t nr;
    uint32_t kc;
    uint32_t l1_bytes;
    uint32_t cache_budget_bytes;
    uint32_t working_set_bytes;
} VxGemmF32TileConfig;

VxGemmF32TileConfig vx_gemm_f32_tile_config(void);

/* Return the padded number of F32 elements needed for a packed KxN matrix, or
 * zero if the dimensions are invalid or cannot be represented by this ABI. */
uint32_t vx_gemm_f32_packed_elements(uint32_t k, uint32_t n);

/* Pack either [K,N] (out_in=0) or [N,K] (out_in=1) source storage into the
 * common K-major panel representation. */
int vx_gemm_f32_pack_b(const float* weight, float* packed,
                       uint32_t k, uint32_t n, int out_in);

/* C = A * packed(B) + bias.  The add variant accumulates into the existing C
 * after applying the optional bias and is used by gradient paths. */
int vx_gemm_f32_run_packed(const float* a, const float* packed_b,
                           const float* bias, float* c,
                           uint32_t m, uint32_t k, uint32_t n);
int vx_gemm_f32_run_packed_add(const float* a, const float* packed_b,
                               const float* bias, float* c,
                               uint32_t m, uint32_t k, uint32_t n);

#ifndef __wasm__
/* Native model-runtime cache.  Entries are keyed by graph-node index and must
 * be cleared when an in-place weight update makes source contents mutable. */
const float* vx_gemm_f32_pack_cache(int node_index, const float* weight,
                                    uint32_t k, uint32_t n, int out_in);
void vx_gemm_f32_cache_free_all(void);
#endif

#ifdef __cplusplus
}
#endif

#endif
