/* Development-only bounded microbenchmark of the exact packed-I8 blocks.
 * Compile with -O3 -ffp-contract=off and native/src/kernels on the include path.
 * -U__SSE2__ selects the previous scalar block for the matched x86 control.
 * Timed blocks are independent synthetic probes, not additive model self-times.
 */
#define _POSIX_C_SOURCE 200809L
#include "llm_ops.h"
#include "packed_linear_ops.inc"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

enum {ROWS=8,K=320,N=1280,GROUP=32,GROUPS=10};
static volatile float sink;
static void* volatile escaped;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static int compare(const void* a,const void* b){double x=*(const double*)a,y=*(const double*)b;return(x>y)-(x<y);}
static float prepare(const float* input,float* scratch){
    float maximum=0.f;
    for(unsigned j=0;j<K;j++){
        if(!vx_packed_finite(input[j]))return __builtin_nanf("");
        float absolute=input[j]<0.f?-input[j]:input[j];if(absolute>maximum)maximum=absolute;
    }
    float scale=maximum==0.f?1.f:maximum/127.f;if(scale<0x1p-126f)scale=0x1p-126f;
    for(unsigned j=0;j<K;j++)((int8_t*)scratch)[j]=(int8_t)vx_packed_round_i8(input[j]/scale);
    return scale;
}
int main(void){
    float *x=malloc(ROWS*K*sizeof(float)),*scales=malloc(N*GROUPS*sizeof(float)),*out=malloc(ROWS*N*sizeof(float));
    float scratch[K];uint32_t* weights=malloc(N*K*sizeof(uint32_t)/4);
    if(!x||!scales||!out||!weights)return 1;
    for(int i=0;i<ROWS*K;i++)x[i]=(float)(i%255-127)*.007f;
    for(int i=0;i<N*GROUPS;i++)scales[i]=(float)(1+i%9)*.001f;
    for(int i=0;i<N*K/4;i++)weights[i]=(uint32_t)i*UINT32_C(1664525)+UINT32_C(1013904223);
    double times[4][9];
    puts("{\"shape\":[8,320,1280],\"group_size\":32,\"boundary\":\"independent synthetic blocks, not additive model self-times\",\"bits\":{");
    for(int bits=4;bits<=8;bits+=4){
        VxLLMOpArgs args={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=ROWS,.key_dim=K,.value_dim=N,
            .groups=bits,.rate=GROUP,.normalize=8,.x={x,(const float*)weights,scales,NULL},.y={out}};
        vx_packed_linear_forward(&args,scratch);
        for(int trial=0;trial<9;trial++){
            double start=now();
            for(int repeat=0;repeat<1000;repeat++)for(int row=0;row<ROWS;row++)sink=prepare(x+row*K,scratch);
            times[0][trial]=(now()-start)/1000;
            float activation_scale=prepare(x,scratch);
            start=now();
            for(int repeat=0;repeat<10;repeat++)for(int row=0;row<ROWS;row++)for(int n=0;n<N;n++){
                float sum=0.f;
                for(int g=0;g<GROUPS;g++)sum+=(float)vx_packed_dot_i8((int8_t*)scratch+g*GROUP,
                    weights+n*(K/(32/bits))+g*(GROUP/(32/bits)),GROUP,bits)*(activation_scale*scales[n*GROUPS+g]);
                out[row*N+n]=sum;
            }
            sink=out[0];times[1][trial]=(now()-start)/10;
            start=now();for(int repeat=0;repeat<10;repeat++)vx_packed_linear_forward(&args,scratch);
            sink=out[0];times[2][trial]=(now()-start)/10;
            start=now();for(int repeat=0;repeat<100000;repeat++){void* p=malloc(K*sizeof(float));escaped=p;free(p);}
            times[3][trial]=(now()-start)/100000;
        }
        for(int part=0;part<4;part++)qsort(times[part],9,sizeof(double),compare);
        printf("%s\"%d\":{\"activation_prepare_us\":%.3f,\"integer_dot_and_group_scale_us\":%.3f,\"full_forward_us\":%.3f,\"scratch_malloc_free_us\":%.6f}",
            bits==4?"":",\n",bits,times[0][4]/1000,times[1][4]/1000,times[2][4]/1000,times[3][4]/1000);
    }
    puts("\n}}");free(x);free(scales);free(out);free(weights);return 0;
}
