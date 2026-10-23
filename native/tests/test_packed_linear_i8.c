#define _POSIX_C_SOURCE 200809L
#include "llm_ops.h"
#include "packed_linear_ops.inc"
#include <string.h>
#include <stdlib.h>
#ifndef __wasm__
#include <stdio.h>
#include <time.h>
#endif

#ifdef __wasm__
#define CHECK(x) do { if (!(x)) return __LINE__; } while (0)
#else
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"line %d: %s\n",__LINE__,#x); return __LINE__; } } while (0)
#endif
static uint32_t rng=1;
static uint32_t random_word(void) { rng=rng*1664525u+1013904223u;return rng; }

static uint32_t oracle_code(const uint32_t* row,uint32_t k,uint32_t bits) {
    uint64_t bit=(uint64_t)k*bits;uint32_t shift=(uint32_t)(bit%32u);
    uint64_t pair=row[bit/32u];
    if(shift+bits>32u)pair|=(uint64_t)row[bit/32u+1u]<<32u;
    return (uint32_t)(pair>>shift)&((1u<<bits)-1u);
}
static void oracle_store(uint32_t* row,uint32_t k,uint32_t bits,uint32_t code) {
    uint64_t bit=(uint64_t)k*bits;uint32_t shift=(uint32_t)(bit%32u);
    row[bit/32u]|=code<<shift;
    if(shift+bits>32u)row[bit/32u+1u]|=code>>(32u-shift);
}

int test_packed_linear_i8(void) {
    int8_t activation[258];uint32_t weights[66];
    for(uint32_t bits=2;bits<=8;bits=bits==4?8:bits+1)for(int pattern=0;pattern<64;pattern++) {
        for(int i=0;i<258;i++)activation[i]=pattern==0?-127:pattern==1?127:(int8_t)(random_word()>>24);
        for(int i=0;i<66;i++)weights[i]=pattern==0?0:pattern==1?UINT32_MAX:random_word();
        for(uint32_t count=0;count<=256;count++) {
            int32_t expected=0;
            for(uint32_t k=0;k<count;k++)expected+=(int32_t)activation[k+1]*
                ((int32_t)oracle_code(weights+1,k,bits)-(1<<(bits-1)));
            CHECK(vx_packed_dot_i8(activation+1,weights+1,count,bits)==expected);
        }
    }
    /* Independent scalar oracle, with full groups, partial groups, all code
     * extrema, per-row activation scales, bias and partial work. */
    enum {K=257,N=7,R=3};
    float x[K*R],scales[N*9],bias[N],actual[N*R],expected[N*R],scratch[K];
    uint32_t packed[N*65];
    for(uint32_t bits=2;bits<=8;bits=bits==4?8:bits+1)for(int group=32;group<=256;group*=2) {
        int words=(K*(int)bits+31)/32,groups=(K+group-1)/group;
        memset(packed,0,sizeof(packed));
        for(int row=0;row<R;row++)for(int k=0;k<K;k++)x[row*K+k]=row==2?0.f:
            (float)(k==0?127:(int)(random_word()%255)-127)*(row==0?.125f:.5f);
        for(int out=0;out<N;out++) {
            bias[out]=(float)(out-3)*.125f;
            for(int g=0;g<groups;g++)scales[out*groups+g]=(float)(g+1)*.0625f;
            for(int k=0;k<K;k++)oracle_store(packed+out*words,k,bits,random_word()&((1u<<bits)-1u));
        }
        for(int row=0;row<R;row++)for(int out=0;out<N;out++) {
            float sum=0.f,as=row==0?.125f:row==1?.5f:1.f;
            for(int g=0;g<groups;g++) {
                int dot=0;
                for(int k=g*group;k<K&&k<(g+1)*group;k++) {
                    int a=(int)(x[row*K+k]/as);
                    int w=(int)oracle_code(packed+out*words,k,bits)-(1<<(bits-1));
                    dot+=a*w;
                }
                sum+=(float)dot*(as*scales[out*groups+g]);
            }
            expected[row*N+out]=sum+bias[out];
        }
        VxLLMOpArgs args={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=R,.key_dim=K,.value_dim=N,
            .groups=(int)bits,.rate=group,.normalize=8,.x={x,(const float*)packed,scales,bias},.y={actual}};
        CHECK(!vx_packed_linear_forward(&args,scratch));
        for(int i=0;i<N*R;i++)CHECK(actual[i]==expected[i]);
        for(int i=0;i<N*R;i++)actual[i]=12345.f;
        args.first_unit=3;args.unit_count=14;
        CHECK(!vx_packed_linear_forward(&args,scratch));
        for(int i=0;i<N*R;i++)CHECK(actual[i]==(i>=3&&i<17?expected[i]:12345.f));
    }
    return 0;
}

/* An independent signed rounding oracle uses magnitude, with explicit
 * ties-to-even. It does not call the kernel's input conversion helpers. */
static int oracle_i8(float value) {
    int negative=value<0.0f;float magnitude=negative?-value:value;
    if(magnitude>=127.0f)return negative?-127:127;
    int whole=(int)magnitude;float fraction=magnitude-(float)whole;
    if(fraction>0.5f||(fraction==0.5f&&(whole%2)))whole++;
    return negative?-whole:whole;
}
static int finite_value(float value) {
    union {float f;uint32_t u;} v={value};return (v.u&0x7f800000u)!=0x7f800000u;
}
static uint32_t float_bits(float value) {union {float f;uint32_t u;} v={value};return v.u;}

int test_packed_linear_i8_tiles(void) {
    enum {MAX_K=1283,MAX_N=19,R=11};
    static float x[R*MAX_K],scales[MAX_N*((MAX_K+31)/32)],bias[MAX_N];
    static float actual[R*MAX_N],expected[R*MAX_N],scratch[MAX_K+8];
    static uint32_t packed[MAX_N*((MAX_K+3)/4)+2];
    const int widths[]={1,2,3,7,8,9,10,11,15,16,17,21,22,31,32,33,63,64,65,255,256,257,320,1280,1283};
    const int columns[]={1,3,4,5,13,19};
    CHECK(oracle_i8(.5f)==0&&oracle_i8(1.5f)==2&&oracle_i8(2.5f)==2);
    CHECK(oracle_i8(-.5f)==0&&oracle_i8(-1.5f)==-2&&oracle_i8(-2.5f)==-2);
    for(unsigned ki=0;ki<sizeof(widths)/sizeof(widths[0]);ki++)
    for(unsigned ni=0;ni<sizeof(columns)/sizeof(columns[0]);ni++)
    for(int bits=2;bits<=8;bits=bits==4?8:bits+1)for(int group=32;group<=256;group*=2)
    for(int mode=0;mode<3;mode++) {
        int k=widths[ki],n=columns[ni];
        int words=(k*bits+31)/32,groups=(k+group-1)/group;
        uint32_t* w=packed+1; /* 4-byte alignment, not forced SIMD alignment. */
        for(int r=0;r<R;r++)for(int j=0;j<k;j++) {
            float scale=r%2?.0073f:.125f;
            x[r*k+j]=(float)((int)(random_word()%255)-127)*scale;
            if(j==0)x[r*k+j]=127.0f*scale;
            if(r==1)x[r*k+j]=0.0f;
            if(r==2)x[r*k+j]=(float)(j%9-4)*0x1p-148f;
            if(r==3)x[r*k+j]=(float)(j%255-127)*0x1p+120f;
            if(r==4)x[r*k+j]=j==0?127.0f:(float)(j%19)-9.5f;
        }
        if(mode==1){x[2*k]=__builtin_nanf("");x[7*k]=-__builtin_inff();x[8*k]=__builtin_inff();}
        if(mode==2)for(int j=0;j<R*k;j++)x[j]=j%2?-0.0f:0.0f;
        for(int i=0;i<n*words;i++)w[i]=random_word();
        for(int out=0;out<n;out++) {
            bias[out]=(float)(out-9)*.03125f;
            for(int g=0;g<groups;g++)scales[out*groups+g]=(float)(1+random_word()%101)*.00013f;
        }
        for(int r=0;r<R;r++) {
            float maximum=0.0f;int valid=1;
            for(int j=0;j<k;j++) {
                float v=x[r*k+j];if(!finite_value(v)){valid=0;break;}
                float magnitude=v<0.0f?-v:v;if(magnitude>maximum)maximum=magnitude;
            }
            float as=maximum==0.0f?1.0f:maximum/127.0f;
            if(as<0x1p-126f)as=0x1p-126f;
            for(int out=0;out<n;out++) {
                float sum=0.0f;
                if(valid)for(int g=0;g<groups;g++) {
                    int32_t dot=0;
                    for(int j=g*group;j<k&&j<(g+1)*group;j++) {
                        int a=oracle_i8(x[r*k+j]/as);
                        int q=(int)oracle_code(w+out*words,j,bits)-(1<<(bits-1));
                        dot+=a*q;
                    }
                    sum+=(float)dot*(as*scales[out*groups+g]);
                }
                expected[r*n+out]=valid?sum+(mode==2?0.0f:bias[out]):__builtin_nanf("");
            }
        }
        VxLLMOpArgs args={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=R,.key_dim=k,.value_dim=n,
            .groups=bits,.rate=group,.normalize=8,.x={x,(const float*)w,scales,mode==2?NULL:bias},.y={actual}};
        const uint64_t firsts[]={0,1,(uint64_t)n-1,(uint64_t)n+2,(uint64_t)n};
        const uint64_t counts[]={0,(uint64_t)R*n-2,(uint64_t)4*n+2,(uint64_t)3*n-2,(uint64_t)8*n};
        for(int part=0;part<5;part++) {
            args.first_unit=firsts[part];args.unit_count=counts[part];
            uint64_t end=args.first_unit+(args.unit_count?args.unit_count:(uint64_t)R*n);
            for(int i=0;i<R*MAX_N;i++)actual[i]=12345.0f;
            for(int i=0;i<MAX_K+8;i++)scratch[i]=9876.0f;
            CHECK(!vx_packed_linear_forward(&args,scratch+4));
            for(int i=0;i<R*MAX_N;i++) {
                if((uint64_t)i>=args.first_unit&&(uint64_t)i<end) {
                    if(!finite_value(expected[i])&&(float_bits(expected[i])&0x007fffffu))
                        CHECK(!finite_value(actual[i])&&(float_bits(actual[i])&0x007fffffu));
                    else CHECK(float_bits(actual[i])==float_bits(expected[i]));
                } else CHECK(actual[i]==12345.0f);
            }
            for(int i=0;i<4;i++)CHECK(scratch[i]==9876.0f);
            for(int i=k+4;i<MAX_K+8;i++)CHECK(scratch[i]==9876.0f);
        }
    }
    return 0;
}

int test_packed_linear_exact_bounds(void) {
    for(uint32_t bits=2;bits<=8;bits=bits==4?8:bits+1)for(uint32_t count=0;count<=256;count++) {
        uint32_t words=(count*bits+31u)/32u;
        int8_t* input=(int8_t*)malloc(count?count:1u);
        uint32_t* packed=(uint32_t*)malloc(words?words*4u:4u);
        CHECK(input&&packed);
        for(uint32_t i=0;i<count;i++)input[i]=(int8_t)(random_word()>>24);
        for(uint32_t i=0;i<words;i++)packed[i]=random_word();
        int32_t expected=0;
        for(uint32_t i=0;i<count;i++)expected+=(int32_t)input[i]*
            ((int32_t)oracle_code(packed,i,bits)-(1<<(bits-1u)));
        CHECK(vx_packed_dot_i8(input,packed,count,bits)==expected);
        free(input);free(packed);
    }
    return 0;
}

int test_packed_linear_lowbit_arithmetic(void) {
    enum {K=65,N=7,R=9};
    float x[K*R],scales[N*3],bias[N],actual[N*R],reference[N*R],scratch[K];
    uint32_t low[N*9],carrier[N*9];
    for(int bits=2;bits<=3;bits++) {
        int words=(K*bits+31)/32;
        memset(low,0,sizeof(low));memset(carrier,0,sizeof(carrier));
        for(int r=0;r<R;r++)for(int j=0;j<K;j++)
            x[r*K+j]=(float)((int)(random_word()%1000)-500)*.0123f;
        x[4*K]=__builtin_nanf("");x[6*K]=__builtin_inff();
        for(int n=0;n<N;n++) {
            bias[n]=(float)(n-3)*.00017f;
            for(int g=0;g<3;g++)scales[n*3+g]=(float)(g+n+1)*.00237f;
            for(int k=0;k<K;k++) {
                int q=(int)(random_word()&((1u<<bits)-1u))-(1<<(bits-1));
                oracle_store(low+n*words,k,bits,(uint32_t)(q+(1<<(bits-1))));
                oracle_store(carrier+n*9,k,4,(uint32_t)(q+8));
            }
        }
        for(int arithmetic=8;arithmetic<=32;arithmetic*=2)for(int part=0;part<2;part++) {
            for(int i=0;i<N*R;i++)actual[i]=reference[i]=12345.f;
            VxLLMOpArgs a={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=R,.key_dim=K,.value_dim=N,
                .groups=4,.rate=32,.normalize=arithmetic,.x={x,(const float*)carrier,scales,bias},.y={reference},
                .first_unit=part?1u:0u,.unit_count=part?N*R-2u:0u};
            CHECK(!vx_packed_linear_forward(&a,scratch));
            a.groups=bits;a.x[1]=(const float*)low;a.y[0]=actual;
            CHECK(!vx_packed_linear_forward(&a,scratch));
            for(int i=0;i<N*R;i++) {
                if(reference[i]!=reference[i])CHECK(actual[i]!=actual[i]);
                else CHECK(float_bits(actual[i])==float_bits(reference[i]));
            }
        }
    }
    return 0;
}

#ifndef __wasm__
int main(void) {
    int result=test_packed_linear_i8();if(result)return result;
    result=test_packed_linear_i8_tiles();if(result)return result;
    result=test_packed_linear_exact_bounds();if(result)return result;
    result=test_packed_linear_lowbit_arithmetic();if(result)return result;
    puts("packed I2/I3/I4/I8 integer oracle, tails, tiles and arithmetic controls passed");return 0;
}
#endif
