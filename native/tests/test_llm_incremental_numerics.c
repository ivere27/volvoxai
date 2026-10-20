#include "llm_ops.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x) do {if(!(x)){fprintf(stderr,"%s:%d: %s\n",__FILE__,__LINE__,#x);exit(1);}} while(0)
static void close_values(const float* a,const float* b,size_t n) {
    for(size_t i=0;i<n;i++) CHECK(fabsf(a[i]-b[i])<=2e-6f+2e-6f*fabsf(b[i]));
}

static void recurrent_state(void) {
    enum {S=7,H=2,DK=3,DV=2};
    float q[S*DK],k[S*DK],v[S*H*DV],gate[S*H],beta[S*H],alog[H]={-.7f,-.4f},bias[H]={.1f,-.2f};
    float retained[S*H*DV],full[S*H*DV],state[H*DK*DV],full_state[H*DK*DV];
    for(size_t i=0;i<S*DK;i++){q[i]=sinf((float)i*.13f);k[i]=cosf((float)i*.17f);}
    for(size_t i=0;i<S*H*DV;i++)v[i]=sinf((float)i*.19f);
    for(size_t i=0;i<S*H;i++){gate[i]=(float)i*.07f-.2f;beta[i]=(float)i*.03f;}
    VxLLMOpArgs a={.kind=VX_OP_GATED_DELTA_NET,.batch=1,.sequence=3,.heads=H,.kv_heads=1,.key_dim=DK,.value_dim=DV,.normalize=1,.eps=1e-6f,.scale=.5f,
        .x={q,k,v,gate,beta,alog,bias,NULL},.y={retained,state,NULL}};
    for(size_t i=0;i<S*H*DV;i++)retained[i]=1234.0f;
    CHECK(vx_llm_forward_f32(&a)==0);
    float first[3*H*DV];memcpy(first,retained,sizeof(first));
    for(int row=3;row<5;row++) {
        VxLLMOpArgs step=a;step.sequence=1;step.x[0]=q+row*DK;step.x[1]=k+row*DK;step.x[2]=v+row*H*DV;
        step.x[3]=gate+row*H;step.x[4]=beta+row*H;step.x[7]=state;step.y[0]=retained+row*H*DV;step.y[1]=state;
        CHECK(vx_llm_forward_f32(&step)==0);
        a.sequence=row+1;a.y[0]=full;a.y[1]=full_state;
        CHECK(vx_llm_forward_f32(&a)==0);
        close_values(retained,full,(size_t)(row+1)*H*DV);close_values(state,full_state,H*DK*DV);
    }
    CHECK(!memcmp(first,retained,sizeof(first)));
    for(int i=5*H*DV;i<S*H*DV;i++)CHECK(retained[i]==1234.0f);
    /* A new initialization ignores the preceding context's retained state. */
    a.sequence=2;a.y[0]=retained;a.y[1]=state;CHECK(vx_llm_forward_f32(&a)==0);
    a.y[0]=full;a.y[1]=full_state;CHECK(vx_llm_forward_f32(&a)==0);
    close_values(state,full_state,H*DK*DV);
}

static void compression_block(void) {
    enum {S=8,D=4,RATE=2,GROUPS=2};
    float input[S*D*GROUPS],gates[S*D*GROUPS],bias[RATE*D*GROUPS],weight[D],full[S*D],subset[S*D];
    for(size_t i=0;i<S*D*GROUPS;i++){input[i]=sinf((float)i*.07f);gates[i]=cosf((float)i*.11f);}
    for(size_t i=0;i<RATE*D*GROUPS;i++)bias[i]=(float)i*.01f;
    for(int i=0;i<D;i++)weight[i]=1.0f+(float)i*.1f;
    VxLLMOpArgs a={.kind=VX_OP_KV_COMPRESS,.batch=1,.sequence=S,.key_dim=D,.groups=GROUPS,.rate=RATE,.eps=1e-6f,.x={input,gates,bias,weight},.y={full}};
    CHECK(vx_llm_forward_f32(&a)==0);
    for(int i=0;i<S*D;i++)subset[i]=4321.0f;
    a.first_unit=2;a.unit_count=1;a.y[0]=subset;CHECK(vx_llm_forward_f32(&a)==0);
    close_values(subset+2*D,full+2*D,D);
    for(int row=0;row<S;row++)if(row!=2)for(int d=0;d<D;d++)CHECK(subset[row*D+d]==4321.0f);
    a.first_unit=S;CHECK(vx_llm_forward_f32(&a)!=0);
}

static void attention_cursor(void) {
    enum {S=6,H=2,D=3,DV=2};
    float q[S*H*D],k[S*D],v[S*DV],full[S*H*DV],row[H*DV];int32_t mask[S]={1,1,1,1,1,1};
    for(size_t i=0;i<S*H*D;i++)q[i]=sinf((float)i*.05f);
    for(size_t i=0;i<S*D;i++)k[i]=cosf((float)i*.08f);
    for(size_t i=0;i<S*DV;i++)v[i]=(float)i*.07f;
    VxLLMOpArgs a={.kind=VX_OP_SPARSE_ATTENTION,.batch=1,.sequence=S,.keys=S,.heads=H,.kv_heads=1,.key_dim=D,.value_dim=DV,.causal=1,.window=2,.scale=.5f,.rate=1,
        .x={q,k,v},.y={full},.mask=mask,.mask_mode=1};
    CHECK(vx_llm_forward_f32(&a)==0);
    a.sequence=1;a.query_offset=3;a.x[0]=q+3*H*D;a.y[0]=row;CHECK(vx_llm_forward_f32(&a)==0);
    close_values(row,full+3*H*DV,H*DV);
    /* Future KV poison must never affect the retained query. */
    for(int i=4*D;i<S*D;i++)k[i]=9999.0f;
    for(int i=4*DV;i<S*DV;i++)v[i]=-9999.0f;
    CHECK(vx_llm_forward_f32(&a)==0);close_values(row,full+3*H*DV,H*DV);
}
int main(void) {recurrent_state();compression_block();attention_cursor();puts("incremental modern numerical projections: PASS");return 0;}
