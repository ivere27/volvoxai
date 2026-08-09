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
 *
 * MR and NR are resolved from the ISA, not from the shape.  That distinction
 * is the point: once both operands are packed the microkernel sees only
 * contiguous fixed-width panels, so one microkernel per ISA covers every
 * shape and the number of kernels to write is the number of ISAs rather than
 * their product with the shapes a particular model happens to use.
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

/*
 * Shape-driven execution plan.  Everything the shape decides lives here and is
 * backend-independent integer arithmetic; everything the ISA decides lives in
 * the microkernel.  This is the narrow tactic identity the dynamic-shape ADR
 * separates from the shape signature, computed for one GEMM.
 *
 * Two regimes, not a spectrum.  STREAM is m == 1 decode: no reuse exists, the
 * cost is one pass over B, and blocking cannot improve a memory-bound loop.
 * BLOCK is everything else, where cache blocking and A packing decide whether
 * the kernel reaches peak or streams weights from DRAM once per row block.
 */
enum { VX_GEMM_F32_REGIME_STREAM = 0, VX_GEMM_F32_REGIME_BLOCK = 1 };

typedef struct {
    uint32_t mr;      /* microkernel row tile, resolved by ISA            */
    uint32_t nr;      /* microkernel column tile and packed panel width   */
    uint32_t mc;      /* A row block; the packed A block targets L2       */
    uint32_t nc;      /* B column block; bounds how often C is revisited  */
    uint32_t kc;      /* K block; one B micro-panel targets half of L1    */
    int regime;
    int micro;        /* resolved microkernel; zero when the ISA has none   */
    int blocked;      /* the blocked driver owns this shape                 */
} VxGemmF32Plan;

VxGemmF32Plan vx_gemm_f32_plan(uint32_t m, uint32_t k, uint32_t n);

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

enum { VX_GEMM_F32_CACHE_CAPACITY = 1024 };

typedef struct VxGemmF32CacheEntry {
    const float* source;
    float* packed;
    uint32_t k;
    uint32_t n;
    int out_in;
} VxGemmF32CacheEntry;

typedef struct VxGemmF32Cache {
    VxGemmF32CacheEntry entries[VX_GEMM_F32_CACHE_CAPACITY];
} VxGemmF32Cache;

/* Owner-scoped native model cache. Entries are keyed by graph-node index and
 * cleared when in-place weight updates make source contents mutable. */
const float* vx_gemm_f32_pack_cache(VxGemmF32Cache* cache, int node_index,
                                    const float* weight,
                                    uint32_t k, uint32_t n, int out_in);
void vx_gemm_f32_cache_free_all(VxGemmF32Cache* cache);

#ifdef __cplusplus
}
#endif

#endif
