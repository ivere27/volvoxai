/* Standalone numerical fixture; not a public runtime or release export.
 * Compile natively as a shared library, or use the pinned WASM fixture recipe.
 * Timed work is only repeated PackedLinear calls with preallocated scratch. */
#include "llm_ops.h"
#include "packed_linear_ops.inc"

#include <stdlib.h>

/* Per-case storage removes model-specific shape limits. This private fixture
 * bounds total tensor/scratch payload before allocation; allocation is never
 * included in timed kernel calls. Keep eight guard floats around scratch. */
enum {TILE_BENCH_MEMORY_LIMIT=128u*1024u*1024u, SCRATCH_GUARD=4};
static float *input,*scales,*bias,*output,*scratch;
static uint32_t* weights;
static uint64_t scratch_elements;
static VxLLMOpArgs args;

typedef struct {
    uint64_t input,scales,bias,output,weights,scratch;
    uint32_t bytes;
} TileBenchStorage;

static int tile_bench_storage(int rows,int k,int n,int bits,int group,TileBenchStorage* storage) {
    if(rows<1||k<1||n<1||(bits!=2&&bits!=3&&bits!=4&&bits!=8)||
       !vx_packed_group_size_supported(group))return 0;
    uint64_t words=((uint64_t)k*(uint32_t)bits+31u)/32u;
    uint64_t groups=((uint64_t)k+(uint32_t)group-1u)/(uint32_t)group;
    uint64_t work=vx_packed_linear_scratch_elements((uint64_t)k,(uint64_t)rows,
                                                   (uint32_t)bits,(uint64_t)n);
    if(work>TILE_BENCH_MEMORY_LIMIT/sizeof(float)-2u*SCRATCH_GUARD)return 0;
    *storage=(TileBenchStorage){.input=(uint64_t)rows*(uint64_t)k,
        .scales=(uint64_t)n*groups,.bias=(uint64_t)n,
        .output=(uint64_t)rows*(uint64_t)n,.weights=(uint64_t)n*words,
        .scratch=work+2u*SCRATCH_GUARD};
    uint64_t counts[]={storage->input,storage->scales,storage->bias,
                       storage->output,storage->weights,storage->scratch};
    uint64_t total=0;
    for(unsigned i=0;i<sizeof(counts)/sizeof(counts[0]);i++) {
        if(counts[i]>(TILE_BENCH_MEMORY_LIMIT-total)/sizeof(float))return 0;
        total+=counts[i]*sizeof(float);
    }
    storage->bytes=(uint32_t)total;
    return 1;
}

uint32_t tile_bench_memory_limit(void) {return TILE_BENCH_MEMORY_LIMIT;}
uint32_t tile_bench_required_bytes(int rows,int k,int n,int bits,int group) {
    TileBenchStorage storage;
    return tile_bench_storage(rows,k,n,bits,group,&storage)?storage.bytes:0u;
}
void tile_bench_release(void) {
    free(input);free(scales);free(bias);free(output);free(weights);free(scratch);
    input=scales=bias=output=scratch=NULL;weights=NULL;scratch_elements=0;
    args=(VxLLMOpArgs){0};
}
int tile_bench_init(int rows,int k,int n,int bits,int group) {
    tile_bench_release();
    TileBenchStorage storage;
    if(!tile_bench_storage(rows,k,n,bits,group,&storage))return -1;
    input=malloc((size_t)storage.input*sizeof(float));
    scales=malloc((size_t)storage.scales*sizeof(float));
    bias=malloc((size_t)storage.bias*sizeof(float));
    output=malloc((size_t)storage.output*sizeof(float));
    weights=malloc((size_t)storage.weights*sizeof(uint32_t));
    scratch=malloc((size_t)storage.scratch*sizeof(float));
    if(!input||!scales||!bias||!output||!weights||!scratch) {
        tile_bench_release();return -2;
    }
    scratch_elements=storage.scratch-2u*SCRATCH_GUARD;
    for(int r=0;r<rows;r++)for(int j=0;j<k;j++)
        input[(uint64_t)r*k+j]=(float)((int32_t)(((uint64_t)j*17u+(uint64_t)r*13u)%255u)-127)*
                                   (float)(1+r%7)*.007f;
    for(uint64_t i=0;i<storage.scales;i++)scales[i]=(float)(1+i%9)*.001f;
    for(int i=0;i<n;i++)bias[i]=(float)(i%11-5)*.013f;
    for(uint64_t i=0;i<storage.weights;i++)weights[i]=(uint32_t)i*1664525u+1013904223u;
    for(uint64_t i=0;i<storage.scratch;i++)scratch[i]=9876.0f;
    args=(VxLLMOpArgs){.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=rows,.key_dim=k,.value_dim=n,
        .groups=bits,.rate=group,.normalize=8,.x={input,(const float*)weights,scales,bias},.y={output}};
    return 0;
}
int tile_bench_run(int repeats) {
    if(repeats<1||repeats>10000||!args.sequence)return -1;
    for(int i=0;i<repeats;i++) {
        int status=vx_packed_linear_forward(&args,scratch+SCRATCH_GUARD);if(status)return status;
    }
    return 0;
}
uintptr_t tile_bench_output_ptr(void) {return (uintptr_t)output;}
uint32_t tile_bench_output_bytes(void) {return (uint32_t)args.sequence*(uint32_t)args.value_dim*4u;}
uint32_t tile_bench_hash(void) {
    uint32_t value=2166136261u;
    for(uint32_t i=0;i<tile_bench_output_bytes();i++)value=(value^((const uint8_t*)output)[i])*16777619u;
    return value;
}
uint32_t tile_bench_scratch_bytes(void) {
    return (uint32_t)(scratch_elements*sizeof(float));
}
int tile_bench_check_scratch(void) {
    if(!scratch)return -1;
    for(unsigned i=0;i<SCRATCH_GUARD;i++)
        if(scratch[i]!=9876.0f||scratch[SCRATCH_GUARD+scratch_elements+i]!=9876.0f)return -1;
    return 0;
}
int tile_bench_rows(void) {return VX_PACKED_LINEAR_TILE_ROWS;}
int tile_bench_two(void) {return VX_PACKED_LINEAR_TILE2;}
int tile_bench_all_bits(void) {return VX_PACKED_LINEAR_TILE_ALL_BITS;}
