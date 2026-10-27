#ifndef VOLVOXAI_CUDA_PACKED_LAUNCH_PLAN_H
#define VOLVOXAI_CUDA_PACKED_LAUNCH_PLAN_H
#include "dense_launch_plan.h"

/* Function handles are owned by the existing forward registry. This projection
 * adds launch facts, not a second kernel loader or a public tuning interface. */
typedef enum {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol, storage)
#define VOLVOXAI_CUDA_PACKED_VARIANT(id, handle, route, bm, bn, bk, threads, shared, static_shared, arch, arithmetic, group, prepared, fast, cache) VX_CUDA_PACKED_VARIANT_##id,
#include "host/cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_PACKED_VARIANT
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
    VX_CUDA_PACKED_VARIANT_COUNT
} VxCudaPackedVariantId;
typedef struct {
    const char* id;
    VxCudaPackedExecution execution;
    uint32_t tile_rows,tile_k,threads,shared_bytes,static_shared_bytes,minimum_arch,arithmetic_mask;
    uint32_t group_size,prepared_only,fast_only,cache_bytes;
} VxCudaPackedVariant;
static const VxCudaPackedVariant vx_cuda_packed_variants[] = {
#define VOLVOXAI_CUDA_FORWARD_FUNCTION(requirement, handle, symbol, storage)
#define VOLVOXAI_CUDA_PACKED_VARIANT(id, handle, route, bm, bn, bk, threads, shared, static_shared, arch, arithmetic, group, prepared, fast, cache) \
    {#id, {route,bn}, bm,bk,threads,shared,static_shared,arch,arithmetic,group,prepared,fast,cache},
#include "host/cuda_forward_function_registry_host.inc"
#undef VOLVOXAI_CUDA_PACKED_VARIANT
#undef VOLVOXAI_CUDA_FORWARD_FUNCTION
};
typedef struct {
    VxCudaPackedVariantId variant;
    VxCudaPackedExecution execution;
    uint32_t tile_rows,tile_k,grid,threads,shared_bytes,total_shared_bytes;
    uint64_t workspace_stride,workspace_limit;
    uint32_t chunk_capacity;
} VxCudaPackedLaunchPlan;

/* Build a requested variant or refuse it. A row-tile number is never used to
 * infer a function: the selected variant supplies every launch dimension. */
static inline int vx_cuda_packed_plan(VxCudaPackedVariantId id,
        uint64_t rows,uint64_t work,uint32_t k,uint32_t n,uint32_t group,
        uint32_t activation_bits,uint32_t weight_bits,uint32_t arch,
        int fast,int prepared,uint32_t max_shared,VxCudaPackedLaunchPlan* out) {
    if(!out || (unsigned)id>=VX_CUDA_PACKED_VARIANT_COUNT || !rows || !work || !k || !n ||
       (group!=32u && group!=64u && group!=128u && group!=256u) ||
       (weight_bits!=4u && weight_bits!=8u))return 0;
    const VxCudaPackedVariant* v=&vx_cuda_packed_variants[id];
    unsigned arithmetic=activation_bits==8u?1u:activation_bits==16u?2u:activation_bits==32u?4u:0u;
    if(!(v->arithmetic_mask&arithmetic) || arch<v->minimum_arch ||
       (v->group_size && v->group_size!=group) || (v->prepared_only&&!prepared) ||
       (v->fast_only&&!fast))return 0;
    uint64_t grid;
    if(v->execution.route==VX_CUDA_PACKED_PREPARE)grid=rows;
    else if(!v->tile_rows)grid=work/2u+(work%2u!=0u);
    else {
        if(work/n!=rows || work%n)return 0;
        uint64_t m=rows/v->tile_rows+(rows%v->tile_rows!=0u);
        uint64_t columns=n/v->execution.tile_columns+(n%v->execution.tile_columns!=0u);
        if(m>UINT64_MAX/columns)return 0;
        grid=m*columns;
    }
    uint32_t shared=v->shared_bytes;
    /* Only the A8 column kernel can replace its row cache with prepared
     * input. Float column kernels still stage x[0], even if x[4] is supplied. */
    if(v->cache_bytes && (!prepared || activation_bits!=8u)) {
        uint32_t cache=vx_cuda_packed_linear_row_cache_bytes(k);
        if(!cache)return 0;
        shared=cache*v->cache_bytes;
    }
    /* Static storage belongs to the registered function, including shared
     * arenas reserved by other routes in the same compiled entrypoint. */
    if(!grid || grid>UINT32_MAX || shared>UINT32_MAX-v->static_shared_bytes ||
       shared+v->static_shared_bytes>max_shared)return 0;
    *out=(VxCudaPackedLaunchPlan){id,v->execution,v->tile_rows,v->tile_k,
        (uint32_t)grid,v->threads,shared,shared+v->static_shared_bytes,0u,0u,0u};
    return 1;
}

static inline VxCudaPackedVariantId vx_cuda_packed_matrix_variant(unsigned bits,unsigned columns) {
    if(bits==8u) {
        if(columns==32u)return VX_CUDA_PACKED_VARIANT_A8_M16_N32;
        if(columns==64u)return VX_CUDA_PACKED_VARIANT_A8_M16_N64;
        if(columns==128u)return VX_CUDA_PACKED_VARIANT_A8_M16_N128;
    } else if(bits==16u) {
        if(columns==32u)return VX_CUDA_PACKED_VARIANT_A16_M16_N32;
        if(columns==64u)return VX_CUDA_PACKED_VARIANT_A16_M16_N64;
    }
    return VX_CUDA_PACKED_VARIANT_COUNT;
}

static inline int vx_cuda_packed_select(uint64_t rows,uint64_t work,uint32_t k,uint32_t n,
        uint32_t group,uint32_t bits,uint32_t weights,uint32_t arch,int fast,
        int complete,int row_aligned,uint32_t sms,uint32_t max_shared,VxCudaPackedLaunchPlan* out) {
    VxCudaPackedVariantId id=group==128u?VX_CUDA_PACKED_VARIANT_SCALAR:VX_CUDA_PACKED_VARIANT_GENERIC;
    if(!vx_cuda_packed_plan(id,rows,work,k,n,group,bits,weights,arch,fast,0,max_shared,out))return 0;
    if(group!=128u)return 1;
    VxCudaPackedLaunchPlan candidate;
    if(row_aligned && n>=32u) {
        id=bits==8u?VX_CUDA_PACKED_VARIANT_A8_STREAM:VX_CUDA_PACKED_VARIANT_COUNT;
        if(vx_cuda_packed_plan(id,rows,work,k,n,group,bits,weights,arch,fast,0,max_shared,&candidate))*out=candidate;
        id=bits==8u?VX_CUDA_PACKED_VARIANT_A8_CACHE:bits==16u?
            VX_CUDA_PACKED_VARIANT_A16_CACHE:VX_CUDA_PACKED_VARIANT_F32_CACHE;
        if(vx_cuda_packed_plan(id,rows,work,k,n,group,bits,weights,arch,fast,0,max_shared,&candidate))*out=candidate;
    }
    /* M16 still computes padded rows: SM coverage alone is not a win for a
     * two-row long reduction, or for four-row F32. Keep the cached route
     * until enough rows share the tile. F16 partial groups and narrow long
     * reductions need the higher row floor too. Short reductions and wide
     * A8 projections have separately qualified two-row cases. */
    unsigned minimum_rows=bits==32u || (bits==16u &&
        (k%128u || (k>2048u && n<4096u)))?8u:4u;
    int small_matrix=(k<=128u && (bits!=8u || n>=4096u)) ||
        (bits==8u && k<=2048u && n>=8192u &&
            vx_cuda_packed_linear_tile_columns(16u,n,bits,sms)>=64u);
    if(complete && rows>=2u && (rows>=minimum_rows || small_matrix)) {
        uint64_t matrix_rows=rows<16u?16u:rows;
        if(bits==32u)id=rows>=16u?(fast&&arch>=80u&&rows>=128u?
            VX_CUDA_PACKED_VARIANT_FAST_M16_N128:VX_CUDA_PACKED_VARIANT_F32_M16_N32):
            (sms && n/32u+(n%32u!=0u)>=sms?VX_CUDA_PACKED_VARIANT_F32_M16_N32:VX_CUDA_PACKED_VARIANT_COUNT);
        else if(vx_cuda_packed_linear_matrix_tiles(matrix_rows,n,bits,sms,arch))
            id=vx_cuda_packed_matrix_variant(bits,vx_cuda_packed_linear_tile_columns(matrix_rows,n,bits,sms));
        else id=VX_CUDA_PACKED_VARIANT_COUNT;
        if(vx_cuda_packed_plan(id,rows,work,k,n,group,bits,weights,arch,fast,0,max_shared,&candidate))*out=candidate;
    }
    return 1;
}

/* Select qualified prepared function IDs, not arbitrary row counts. Adding M32
 * to a threshold helper cannot silently select M16 with an M32 grid. */
static inline VxCudaPackedVariantId vx_cuda_packed_prepared_variant(
        uint64_t rows,uint32_t k,uint32_t n,uint32_t bits,uint32_t weights,uint32_t arch) {
    unsigned bm=vx_cuda_packed_linear_prepared_tile_rows(rows,k,n,bits,weights,arch);
    if(bits==8u && bm==16u)return VX_CUDA_PACKED_VARIANT_A8_PREPARED_M16;
    if(bits==8u && bm==64u)return VX_CUDA_PACKED_VARIANT_A8_PREPARED_M64;
    if(bits==16u && bm==64u)return vx_cuda_packed_linear_prepared_threads(rows,k,n,bits,weights,arch)==256u?
        VX_CUDA_PACKED_VARIANT_A16_PREPARED_M64_W8:VX_CUDA_PACKED_VARIANT_A16_PREPARED_M64;
    return VX_CUDA_PACKED_VARIANT_COUNT;
}

static inline int vx_cuda_packed_workspace(VxCudaPackedLaunchPlan* plan,uint64_t rows,
        uint32_t k,uint32_t n,uint32_t bits,uint32_t weights,uint32_t arch,int matrix) {
    if(!plan || (unsigned)plan->variant>=VX_CUDA_PACKED_VARIANT_COUNT || !rows ||
       (matrix&&!plan->tile_rows) || (bits!=8u && bits!=16u))return 0;
    const VxCudaPackedVariant* v=&vx_cuda_packed_variants[plan->variant];
    int narrow=matrix&&vx_cuda_packed_linear_prepared_narrow_workspace(rows,k,n,bits,weights,arch);
    uint64_t limit=(v->prepared_only&&rows>=512u)||narrow?
        VX_CUDA_PACKED_WORKSPACE_BYTES:VX_CUDA_MATMUL_WORKSPACE_BYTES;
    /* K128 here is a G128 preparation layout, not an execution tile or a
     * dynamically inferred quantization group. Only fixed-G128 variants use it. */
    if(v->group_size!=128u)return 0;
    uint64_t stride=((uint64_t)k+127u)/128u*128u*(bits==8u?1u:2u);
    if(stride>limit-4u)return 0;
    uint32_t capacity=(uint32_t)(limit/(stride+4u));
    if(rows>capacity) {
        if(limit==VX_CUDA_MATMUL_WORKSPACE_BYTES&&!vx_cuda_packed_linear_prepares_chunks(bits,k,n))return 0;
        if(matrix)capacity-=capacity%plan->tile_rows;
    }
    if(!capacity)return 0;
    plan->workspace_stride=stride;plan->workspace_limit=limit;plan->chunk_capacity=capacity;
    return 1;
}
#endif
