#include "../src/backends/llm_gpu_plan.h"
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <string.h>

typedef struct {
    unsigned calls, up, down, state_copy;
    const void* state;
    uint32_t query_offset, first_unit;
    const void* output;
    float epsilon;
    uint32_t packed_capacity;
} Capture;
static int inspect(void* user,const VxLlmGpuDispatch* d) {
    Capture* c=user;c->calls++;
    assert(d->buffer_count<=8);
    assert(vx_llm_gpu_shader_bindings(d->shader)==d->buffer_count+1);
    assert(d->groups[0]&&d->groups[1]==1&&d->groups[2]==1);
    for(unsigned i=0;i<d->buffer_count;i++)assert(d->buffers[i].bytes&&d->buffers[i].bytes<=UINT32_MAX);
    if(d->shader==VX_LLM_GPU_MOE_UP) {c->up++;assert(d->buffers[5].workspace==1&&d->buffers[5].output);}
    if(d->shader==VX_LLM_GPU_MOE_DOWN) {c->down++;assert(d->buffers[4].host==c->output);}
    if(d->shader==VX_LLM_GPU_MOE_ROUTER) {float v;memcpy(&v,d->uniform+22,4);assert(v==c->epsilon);assert(d->groups[0]==3);}
    if(d->shader==VX_LLM_GPU_DELTA_NET) {
        if(c->state) {assert(d->buffers[4].host==c->state);assert(d->buffers[6].workspace==2&&d->buffers[6].output);}
        else {assert(d->buffers[4].workspace==3&&!d->buffers[4].output);assert(d->buffers[6].workspace==3&&d->buffers[6].output);}
    }
    if(d->shader==VX_LLM_GPU_COPY) {c->state_copy++;assert(d->buffers[0].workspace==2&&!d->buffers[0].output);assert(d->buffers[1].host==c->state&&d->buffers[1].output);assert(d->uniform[0]==160);}
    if(d->shader==VX_LLM_GPU_SPARSE_ATTENTION || d->shader==VX_LLM_GPU_QSA_SCORES || d->shader==VX_LLM_GPU_QSA_MASK)
        assert(d->uniform[28]==c->query_offset);
    if(d->shader==VX_LLM_GPU_KV_COMPRESS || d->shader==VX_LLM_GPU_NGRAM) {
        assert(d->uniform[23]==c->first_unit);assert(d->uniform[22]==1);
    }
    if(d->shader==VX_LLM_GPU_PACKED_LINEAR) {
        assert(d->uniform[8]==4 || d->uniform[8]==8);
        assert(d->buffer_count==5 && d->buffers[1].bytes==7u*(d->uniform[8]==8?33u:17u)*4u);
        assert(d->buffers[2].bytes==7u*2u*4u && d->buffers[4].host==c->output);
        assert(d->uniform[23]==c->first_unit && d->uniform[9]==128);
        assert(d->uniform[17]==8 || d->uniform[17]==16);
        if(!d->uniform[21])assert(d->buffers[3].workspace==3 && d->buffers[3].bytes==4u);
        if(c->packed_capacity) {
            assert(d->buffers[0].bytes==c->packed_capacity*129u*4u);
            assert(d->buffers[4].bytes==c->packed_capacity*7u*4u);
            assert(d->uniform[0]==1u && d->uniform[1]==3u && d->groups[0]==1u);
        }
    }
    return 1;
}
int main(void) {
    float storage[16]={0};uint64_t scratch[2];Capture c={0};
    VxLlmGpuGatedArgs gated={storage,storage+1,storage+2,storage+3,storage+4,storage+5,storage+6,
        3,8,7168,8,8,8,0,6.0f};c.output=gated.output;
    assert(vx_llm_gpu_gated_plan(&gated,scratch,inspect,&c));
    assert(scratch[0]==3u*7168u*sizeof(float)&&scratch[1]==0);
    assert(c.calls==16&&c.up==8&&c.down==8);
    gated.hidden=UINT32_MAX;c.calls=0;
    assert(!vx_llm_gpu_gated_plan(&gated,scratch,inspect,&c)&&c.calls==0);

    VxLlmGpuRouterArgs router={0};router.input=storage;router.weight=storage+1;
    router.indices=storage+2;router.routes=storage+3;router.rows=3;router.d_in=8;router.experts=8;router.top_k=2;
    router.groups=2;router.selected_groups=1;router.group_top_k=2;router.scoring=1;router.normalize=1;
    router.temperature=0.7f;router.scale=2.5f;router.epsilon=0.123f;c.epsilon=router.epsilon;
    assert(vx_llm_gpu_router_plan(&router,inspect,&c));
    router.experts=1025;c.calls=0;assert(!vx_llm_gpu_router_plan(&router,inspect,&c)&&c.calls==0);

    VxLLMOpArgs delta={0};delta.kind=VX_OP_GATED_DELTA_NET;delta.batch=1;delta.sequence=3;
    delta.heads=2;delta.kv_heads=1;delta.key_dim=4;delta.value_dim=5;delta.eps=1e-6f;delta.normalize=1;
    const uint64_t extents[7]={12,12,30,6,6,2,2};
    for(unsigned i=0;i<7;i++){delta.x[i]=storage+i;delta.x_elements[i]=extents[i];}
    delta.y[0]=storage+8;delta.y_elements[0]=30;
    assert(vx_llm_gpu_plan(&delta,scratch,inspect,&c));assert(scratch[0]==96&&scratch[1]==0);
    delta.key_dim=513;c.calls=0;assert(!vx_llm_gpu_plan(&delta,scratch,inspect,&c)&&c.calls==0);

    delta.key_dim=4;delta.sequence=1;delta.query_offset=4;
    delta.x[7]=delta.y[1]=storage+9;delta.x_elements[7]=delta.y_elements[1]=40;
    c=(Capture){.state=delta.y[1]};
    assert(vx_llm_gpu_plan(&delta,scratch,inspect,&c));
    assert(scratch[0]==32&&scratch[1]==160&&c.calls==3&&c.state_copy==1);
    delta.y_elements[1]=39;c.calls=0;
    assert(!vx_llm_gpu_plan(&delta,scratch,inspect,&c)&&c.calls==0);

    VxLLMOpArgs compress={0};compress.kind=VX_OP_KV_COMPRESS;compress.batch=1;compress.sequence=8;
    compress.key_dim=4;compress.groups=2;compress.rate=2;compress.first_unit=2;compress.unit_count=1;
    for(unsigned i=0;i<4;i++) {compress.x[i]=storage+i;compress.x_elements[i]=64;}
    compress.y[0]=storage+8;compress.y_elements[0]=32;c=(Capture){.first_unit=2};
    assert(vx_llm_gpu_plan(&compress,scratch,inspect,&c)&&c.calls==1);
    compress.first_unit=8;c.calls=0;assert(!vx_llm_gpu_plan(&compress,scratch,inspect,&c)&&c.calls==0);
    compress.first_unit=1;compress.unit_count=0;assert(!vx_llm_gpu_plan(&compress,scratch,NULL,NULL));

    VxLLMOpArgs sparse={0};sparse.kind=VX_OP_SPARSE_ATTENTION;sparse.batch=sparse.sequence=1;
    sparse.keys=8;sparse.heads=2;sparse.kv_heads=1;sparse.key_dim=sparse.value_dim=4;
    sparse.rate=2;sparse.query_offset=6;
    for(unsigned i=0;i<3;i++) {sparse.x[i]=storage+i;sparse.x_elements[i]=32;}
    sparse.y[0]=storage+8;sparse.y_elements[0]=8;c=(Capture){.query_offset=6};
    assert(vx_llm_gpu_plan(&sparse,scratch,inspect,&c));
    sparse.query_offset=-1;c.calls=0;assert(!vx_llm_gpu_plan(&sparse,scratch,inspect,&c)&&c.calls==0);
    sparse.query_offset=INT32_MAX;sparse.sequence=2;assert(!vx_llm_gpu_plan(&sparse,scratch,NULL,NULL));

    sparse.sequence=1;sparse.query_offset=0;sparse.x[3]=storage+3;sparse.x_elements[3]=32;
    sparse.x[5]=storage+5;sparse.x_elements[5]=8;c=(Capture){0};
    assert(vx_llm_gpu_plan(&sparse,scratch,inspect,&c)&&c.calls==2&&scratch[0]==9u*4u);
    VxLLMOpArgs indexer={0};indexer.kind=VX_OP_SPARSE_INDEXER;indexer.batch=2;indexer.sequence=indexer.keys=7;
    indexer.heads=2;indexer.key_dim=4;indexer.rate=2;indexer.top_k=3;indexer.window=2;indexer.groups=1;
    for(unsigned i=0;i<3;i++){indexer.x[i]=storage+i;indexer.x_elements[i]=112;}
    indexer.y[0]=storage+8;indexer.y[1]=storage+9;indexer.y_elements[0]=indexer.y_elements[1]=98;c=(Capture){0};
    assert(vx_llm_gpu_plan(&indexer,scratch,inspect,&c)&&c.calls==2&&scratch[0]==98u*4u);
    indexer.x[3]=storage+3;indexer.x_elements[3]=98;c.calls=0;
    assert(!vx_llm_gpu_plan(&indexer,scratch,inspect,&c)&&c.calls==0);
    indexer.y[1]=NULL;indexer.window=indexer.groups=0;
    assert(vx_llm_gpu_plan(&indexer,scratch,inspect,&c)&&c.calls==2);
    indexer.query_offset=1;c.calls=0;assert(!vx_llm_gpu_plan(&indexer,scratch,inspect,&c)&&c.calls==0);
    indexer.query_offset=0;indexer.y[1]=storage+9;indexer.window=indexer.groups=-1;indexer.x[3]=NULL;
    assert(!vx_llm_gpu_plan(&indexer,scratch,inspect,&c)&&c.calls==0);
    indexer.window=2;indexer.groups=1;
    indexer.sequence=indexer.keys=INT32_MAX;c.calls=0;
    assert(!vx_llm_gpu_plan(&indexer,scratch,inspect,&c)&&c.calls==0);

    VxLLMOpArgs packed={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=3,
        .key_dim=129,.value_dim=7,.rate=128,.normalize=16,
        .x={storage,storage+1,storage+2},.y={storage+8},
        .x_elements={387,119,14},.y_elements={21}};
    c=(Capture){.output=packed.y[0]};
    assert(vx_llm_gpu_plan(&packed,scratch,inspect,&c)&&c.calls==1&&!scratch[0]&&!scratch[1]);
    /* Logical prefix work must retain the full cached slot ownership span. */
    packed.x_elements[0]=256u*129u;packed.y_elements[0]=256u*7u;c.packed_capacity=256;
    assert(vx_llm_gpu_plan(&packed,scratch,inspect,&c)&&c.calls==2);
    packed.x_elements[0]=386;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.x_elements[0]=256u*129u;packed.y_elements[0]=20;
    assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.x_elements[0]=387;packed.y_elements[0]=21;c.packed_capacity=0;c.calls=1;
    packed.normalize=8;packed.x[3]=storage+3;packed.x_elements[3]=7;
    packed.first_unit=7;packed.unit_count=7;c.first_unit=7;
    assert(vx_llm_gpu_plan(&packed,scratch,inspect,&c)&&c.calls==2);
    packed.unit_count=15;c.calls=0;assert(!vx_llm_gpu_plan(&packed,scratch,inspect,&c)&&!c.calls);
    packed.unit_count=0;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.first_unit=0;packed.x_elements[1]--;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.x_elements[1]++;packed.x_elements[2]--;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.x_elements[2]++;packed.rate=64;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.rate=128;packed.normalize=4;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.normalize=16;packed.groups=8;packed.x_elements[1]=7u*33u;
    c=(Capture){.output=packed.y[0]};
    assert(vx_llm_gpu_plan(&packed,scratch,inspect,&c)&&c.calls==1&&!scratch[0]&&!scratch[1]);
    packed.groups=4;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.groups=3;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));
    packed.groups=-1;assert(!vx_llm_gpu_plan(&packed,scratch,NULL,NULL));

    int32_t integers[16]={0};
    VxLlmGpuNgramArgs ngram={.tokens=integers,.multipliers=integers+1,.sizes=integers+2,
        .offsets=integers+3,.output=integers+4,.batch=1,.sequence=8,.ngram=2,.heads_per_ngram=1,
        .eos=1,.pad=1,.vocabulary=32,.first_unit=6,.unit_count=1};c=(Capture){.first_unit=6};
    assert(vx_llm_gpu_ngram_plan(&ngram,inspect,&c)&&c.calls==1);
    ngram.unit_count=3;c.calls=0;assert(!vx_llm_gpu_ngram_plan(&ngram,inspect,&c)&&c.calls==0);

    size_t raw[2]={128,320},slots[2];
    assert(vx_llm_gpu_workspace_slots(464,raw,slots)&&slots[0]==128&&slots[1]==320);
    assert(!vx_llm_gpu_workspace_slots(448,raw,slots));
    raw[0]=raw[1]=0;assert(vx_llm_gpu_workspace_slots(144,raw,slots)&&slots[0]==128&&slots[1]==0);
    raw[0]=SIZE_MAX;assert(!vx_llm_gpu_workspace_slots(16,raw,slots));
    c=(Capture){0};

    VxLlmGpuRopeArgs rope={storage,NULL,storage+8,1,3,8,2,4,0,0,0,10000.0f};
    assert(vx_llm_gpu_rope_plan(&rope,inspect,&c));
    rope.sequence=UINT32_MAX;c.calls=0;assert(!vx_llm_gpu_rope_plan(&rope,inspect,&c)&&c.calls==0);
    puts("LLM GPU plans: bounded workspaces, packed controls and atomic refusals passed");
    return 0;
}
