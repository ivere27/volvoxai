/* Standalone native/WASM projection microbenchmark, never a product entry.
 * Compile with -O3 -ffp-contract=off. PACKED_KERNEL_SOURCE can name an archived
 * implementation for an identical-input before/after comparison. No whole
 * decoded weights, framework dispatch, model I/O or UI work is timed here. */
#define _POSIX_C_SOURCE 200809L
#include "llm_ops.h"
#ifndef PACKED_KERNEL_SOURCE
#define PACKED_KERNEL_SOURCE "packed_linear_ops.inc"
#endif
#include PACKED_KERNEL_SOURCE
#ifndef __wasm__
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#endif

enum {MAX_ROWS=402,MAX_K=1283,MAX_N=1280};
static float x[MAX_ROWS*MAX_K],scales[MAX_N*((MAX_K+31)/32)],output[MAX_ROWS*MAX_N],scratch[MAX_K];
static uint32_t weights[MAX_N*((MAX_K+3)/4)];
static VxLLMOpArgs args;

int packed_bench_init(int rows,int k,int n,int bits) {
    if(rows<1||rows>MAX_ROWS||k<1||k>MAX_K||n<1||n>MAX_N||(bits!=4&&bits!=8))return -1;
    int words=(k+32/bits-1)/(32/bits),groups=(k+31)/32;
    for(int i=0;i<rows*k;i++)x[i]=(float)(i%255-127)*.007f;
    for(int i=0;i<n*groups;i++)scales[i]=(float)(1+i%9)*.001f;
    for(int i=0;i<n*words;i++)weights[i]=(uint32_t)i*1664525u+1013904223u;
    args=(VxLLMOpArgs){.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=rows,.key_dim=k,.value_dim=n,
        .groups=bits,.rate=32,.normalize=8,.x={x,(const float*)weights,scales,NULL},.y={output}};
    return 0;
}
int packed_bench_run(int repeats) {
    if(repeats<1||repeats>10000||!args.sequence)return -1;
    for(int i=0;i<repeats;i++){int status=vx_packed_linear_forward(&args,scratch);if(status)return status;}
    return 0;
}
uint32_t packed_bench_hash(void) {
    uint32_t hash=2166136261u;
    for(int i=0;i<args.sequence*args.value_dim*(int)sizeof(float);i++)hash=(hash^((unsigned char*)output)[i])*16777619u;
    return hash;
}

#ifndef __wasm__
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return t.tv_sec*1e9+t.tv_nsec;}
static int compare(const void* a,const void* b){double x0=*(const double*)a,y0=*(const double*)b;return(x0>y0)-(x0<y0);}
int main(void) {
    const int shape[][3]={{1,320,1280},{4,320,1280},{8,320,1280},{402,320,960},{402,320,1280},{402,1280,320},{9,257,13}};
    puts("{\"boundary\":\"standalone projection; group32; 2 warmups; median of 5 batches\",\"samples\":[");
    int first=1;
    for(unsigned s=0;s<sizeof(shape)/sizeof(shape[0]);s++)for(int bits=4;bits<=8;bits+=4) {
        int rows=shape[s][0],k=shape[s][1],n=shape[s][2],repeats=rows>100?2:100;double times[5];
        if(packed_bench_init(rows,k,n,bits)||packed_bench_run(2))return 1;
        for(int trial=0;trial<5;trial++){double start=now();if(packed_bench_run(repeats))return 1;times[trial]=(now()-start)/repeats;}
        qsort(times,5,sizeof(double),compare);
        printf("%s{\"rows\":%d,\"k\":%d,\"n\":%d,\"bits\":%d,\"median_ms\":%.6f,\"hash\":%u}",
            first?"":",\n",rows,k,n,bits,times[2]/1e6,packed_bench_hash());first=0;
    }
    puts("\n]}");return 0;
}
#endif
