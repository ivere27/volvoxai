/* Tiny, independent scalar oracles exercise routed nonlinear experts without
 * model weights or a device. Compile in both profile closures. */
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#define WASM_EXPORT(name)
#include "../src/kernels/portable_inference_kernels.inc"
static int failures;
#define CHECK(x) do { if (!(x)) {fprintf(stderr,"line %d: %s\n",__LINE__,#x);++failures;} } while(0)
static void near(float actual,double expected) { CHECK(fabs(actual-expected)<2e-5*(1+fabs(expected))); }
static double sigmoid(double x) {return 1/(1+exp(-x));}

static void router(void) {
    float x[]={1}, w[]={0,1,2,3}, ids[2], weights[2], correction[]={10,0,0,0};
    CHECK(vx_moe_router_extended_f32(x,w,NULL,correction,NULL,NULL,ids,weights,1,1,4,2,1,1,1,2,1,2,2,0,0));
    CHECK(ids[0]==0 && ids[1]==1); /* correction selects group0; scores unchanged */
    near(weights[0],2*0.5/(0.5+sigmoid(1)));near(weights[1],2*sigmoid(1)/(0.5+sigmoid(1)));
    CHECK(vx_moe_router_extended_f32(x,w,NULL,NULL,NULL,NULL,ids,weights,1,1,4,2,1,0,0,1,1,1,1,0,0));
    CHECK(ids[0]==3 && ids[1]==2);near(weights[0],exp(3)/(1+exp(1)+exp(2)+exp(3)));
    CHECK(vx_moe_router_extended_f32(x,w,NULL,NULL,NULL,NULL,ids,weights,1,1,4,2,1,1,0,1,1,1,1,0,0));
    near(weights[0],exp(3)/(exp(2)+exp(3)));
    int32_t tokens[]={1}, table[]={3,2,0,2};
    CHECK(vx_moe_router_extended_f32(x,w,NULL,NULL,tokens,table,ids,weights,1,1,4,2,1,1,2,1,1,1,1.5,1e-20,2));
    CHECK(ids[0]==0 && ids[1]==2);
    double a=sqrt(log(2)), b=sqrt(log1p(exp(2)));
    near(weights[0],1.5*a/(a+b));near(weights[1],1.5*b/(a+b));
    tokens[0]=2;CHECK(!vx_moe_router_extended_f32(x,w,NULL,NULL,tokens,table,ids,weights,1,1,4,2,1,1,2,1,1,1,1,0,2));
    tokens[0]=0;table[0]=4;CHECK(!vx_moe_router_extended_f32(x,w,NULL,NULL,tokens,table,ids,weights,1,1,4,2,1,1,2,1,1,1,1,0,2));
    float saturated[]={80,90,100,110};
    CHECK(vx_moe_router_extended_f32(x,saturated,NULL,NULL,NULL,NULL,ids,weights,1,1,4,2,1,0,1,1,1,1,1,0,0));
    CHECK(ids[0]==3 && ids[1]==2); /* select logits before sigmoid saturates */
    float tied[]={0,0,0,0};
    CHECK(vx_moe_router_extended_f32(x,tied,NULL,NULL,NULL,NULL,ids,weights,1,1,4,2,1,1,0,1,1,1,1,0,0));
    CHECK(ids[0]==0 && ids[1]==1);
    CHECK(!vx_moe_router_extended_f32(x,w,NULL,NULL,NULL,NULL,ids,weights,1,1,4,2,1,1,0,3,1,1,1,0,0));
}
static void gated(void) {
    float x[]={1,-2,.5f,3}, gate[]={1,2,-1,.5f, 2,-3,1,2}, up[]={2,1,3,-2, -1,2,4,3};
    float down[]={1,2,-2,1, 3,-1,2,4}, ids[]={1,0,0,1}, weights[]={.25f,.75f,.6f,.4f}, result[4];
    for (int scale_input=0;scale_input<2;++scale_input) for(int clip=0;clip<2;++clip) {
        double expected[4]={0};
        for(int r=0;r<2;++r) for(int k=0;k<2;++k) {
            int e=(int)ids[r*2+k];double route=weights[r*2+k], hidden[2];
            for(int h=0;h<2;++h) {
                double g=0,u=0;
                for(int d=0;d<2;++d) {double value=x[r*2+d]*(scale_input?route:1);g+=value*gate[(e*2+d)*2+h];u+=value*up[(e*2+d)*2+h];}
                if(clip){g=fmin(g,1.2);u=fmax(-1.2,fmin(u,1.2));}
                hidden[h]=g*sigmoid(g)*u;
            }
            for(int o=0;o<2;++o) for(int h=0;h<2;++h) expected[r*2+o]+=(scale_input?1:route)*hidden[h]*down[(e*2+h)*2+o];
        }
        CHECK(vx_moe_gated_mlp_f32(x,gate,up,down,ids,weights,result,2,2,2,2,2,2,scale_input,clip?1.2f:0));
        for(int i=0;i<4;++i) near(result[i],expected[i]);
    }
    ids[0]=.5f;CHECK(!vx_moe_gated_mlp_f32(x,gate,up,down,ids,weights,result,2,2,2,2,2,2,0,0));
    ids[0]=2;CHECK(!vx_moe_gated_mlp_f32(x,gate,up,down,ids,weights,result,2,2,2,2,2,2,0,0));
    ids[0]=1;weights[0]=NAN;CHECK(!vx_moe_gated_mlp_f32(x,gate,up,down,ids,weights,result,2,2,2,2,2,2,0,0));
}
int main(void) {router();gated();if(!failures)puts("modern MoE scalar oracles PASS");return failures?1:0;}
