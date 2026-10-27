#if !defined(__wasm__) && !defined(_WIN32) && !defined(_GNU_SOURCE)
#define _GNU_SOURCE
#endif
#define _POSIX_C_SOURCE 200809L
#ifndef __wasm__
#include "thread_pool.c"
static int vx_packed_force_sse2_test;
static uint64_t vx_packed_parallel_macs_test=16384u;
#define VX_PACKED_FORCE_SSE2_TEST vx_packed_force_sse2_test
#define VX_PACKED_LINEAR_PARALLEL_MACS vx_packed_parallel_macs_test
#endif
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
#define TEST_SCRATCH(k) ((k)*(VX_PACKED_LINEAR_TILE_ROWS+4)/2+512)
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

int test_packed_linear_i3_unpack(void) {
    /* One-hot activations isolate every decoded lane. Walking bits and
     * arbitrary surrounding bits cover all four eight-code word offsets,
     * including codes spanning two I32 words. */
    for(uint32_t pattern=0;pattern<128u;pattern++) {
        uint32_t words[3];
        for(uint32_t i=0;i<3u;i++)words[i]=pattern<96u?
            (pattern/32u==i?1u<<(pattern%32u):0u):random_word();
        for(uint32_t block=0;block<4u;block++) {
#if defined(__wasm_simd128__) || defined(__SSE2__)
            int16_t decoded[8];
#if defined(__wasm_simd128__)
            wasm_v128_store(decoded,vx_packed_i3_i16x8(words+block*24u/32u,(block*24u)%32u));
#else
            _mm_storeu_si128((__m128i*)decoded,vx_packed_i3_i16x8(words+block*24u/32u,(block*24u)%32u));
#endif
            for(uint32_t lane=0;lane<8u;lane++)
                CHECK(decoded[lane]==(int32_t)oracle_code(words,block*8u+lane,3u)-4);
#endif
            for(uint32_t lane=0;lane<8u;lane++) {
                int8_t activation[32]={0};activation[block*8u+lane]=-128;
                CHECK(vx_packed_dot_i8(activation,words,32u,3u)==
                    -128*((int32_t)oracle_code(words,block*8u+lane,3u)-4));
            }
        }
    }
    return 0;
}

static int test_packed_linear_i8_preparation(void) {
#if VX_PACKED_HAS_SIMD_QUANT
    /* Compare every prepared integer, including adjacent floats around ties,
     * independently of the dot-product oracle. Sentinels catch narrow/wide
     * stores past short rows, and odd wide rows must zero their padded lane. */
    static const float units[]={0x1p-149f,0x1p-126f,0x1p-120f,0.125f,
        1.0f,0.003f,0x1p+100f,0x1p+120f};
    static const uint32_t lengths[]={1,2,3,4,5,7,8,31,32,33,257};
    float input[257];unsigned char actual[520];
    for(uint32_t size=0;size<sizeof(lengths)/sizeof(lengths[0]);size++) {
        uint32_t k=lengths[size];
        for(uint32_t u=0;u<sizeof(units)/sizeof(units[0]);u++)
        for(uint32_t pattern=0;pattern<255u;pattern++) {
            for(uint32_t j=0;j<k;j++) {
                union {float f;uint32_t u;} value;
                value.f=((float)((int)((j+pattern)%255u)-127)+0.5f)*units[u];
                if(j%3u==1u&&value.f!=0.0f)value.u--;
                if(j%3u==2u)value.u++;
                input[j]=value.f;
            }
            input[k-1u]=127.0f*units[u];
            float maximum=0.0f;
            for(uint32_t j=0;j<k;j++) {
                float absolute=input[j]<0.0f?-input[j]:input[j];
                if(absolute>maximum)maximum=absolute;
            }
            float expected_scale=maximum==0.0f?1.0f:maximum/127.0f;
            if(expected_scale<0x1p-126f)expected_scale=0x1p-126f;
            for(int wide=0;wide<2;wide++) {
                memset(actual,0x5a,sizeof(actual));float scale;
                CHECK(vx_packed_prepare_i8_wasm(input,k,actual+1,wide,&scale));
                CHECK(scale==expected_scale);
                for(uint32_t j=0;j<k;j++) {
                    int32_t expected=vx_packed_round_i8(input[j]/expected_scale);
                    if(wide) {
                        int16_t value;memcpy(&value,actual+1+j*2u,sizeof(value));
                        CHECK(value==expected);
                    } else CHECK((int8_t)actual[j+1u]==expected);
                }
                uint32_t bytes=wide?((k+1u)&~1u)*2u:k;
                if(wide&&(k&1u))CHECK(actual[1u+k*2u]==0&&actual[2u+k*2u]==0);
                CHECK(actual[0]==0x5a);
                for(uint32_t j=bytes+1u;j<sizeof(actual);j++)CHECK(actual[j]==0x5a);
            }
        }
        for(uint32_t j=0;j<k;j++)input[j]=j&1u?-0.0f:0.0f;
        for(int wide=0;wide<2;wide++) {
            float scale;CHECK(vx_packed_prepare_i8_wasm(input,k,actual,wide,&scale));
            CHECK(scale==1.0f);
            for(uint32_t j=0;j<k*(wide?2u:1u);j++)CHECK(actual[j]==0);
        }
        const uint32_t exceptional[]={0x7f800000u,0xff800000u,0x7fc00001u,0x7f800001u};
        for(uint32_t lane=0;lane<k;lane++)for(uint32_t e=0;e<4u;e++) {
            memcpy(input+lane,exceptional+e,sizeof(float));
            for(int wide=0;wide<2;wide++) {
                float scale;memset(actual,0x5a,sizeof(actual));
                CHECK(!vx_packed_prepare_i8_wasm(input,k,actual+1,wide,&scale));
                uint32_t bytes=wide?((k+1u)&~1u)*2u:k;
                CHECK(actual[0]==0x5a);
                for(uint32_t j=1;j<=bytes;j++)CHECK(actual[j]==0);
                for(uint32_t j=bytes+1u;j<sizeof(actual);j++)CHECK(actual[j]==0x5a);
            }
            input[lane]=0.0f;
        }
    }
#endif
    return 0;
}

int test_packed_linear_i8(void) {
    int preparation=test_packed_linear_i8_preparation();if(preparation)return preparation;
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
    float x[K*R],scales[N*9],bias[N],actual[N*R],expected[N*R],scratch[TEST_SCRATCH(K)];
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
    static float actual[R*MAX_N],expected[R*MAX_N],scratch[TEST_SCRATCH(MAX_K)+8];
    static uint32_t packed[MAX_N*((MAX_K+3)/4)+2];
    const int widths[]={1,2,3,7,8,9,10,11,15,16,17,21,22,23,24,25,31,32,33,
        39,40,41,47,48,49,55,56,57,63,64,65,255,256,257,320,1280,1283};
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
            for(int i=0;i<TEST_SCRATCH(MAX_K)+8;i++)scratch[i]=9876.0f;
            CHECK(!vx_packed_linear_forward(&args,scratch+4));
            for(int i=0;i<R*MAX_N;i++) {
                if((uint64_t)i>=args.first_unit&&(uint64_t)i<end) {
                    if(!finite_value(expected[i])&&(float_bits(expected[i])&0x007fffffu))
                        CHECK(!finite_value(actual[i])&&(float_bits(actual[i])&0x007fffffu));
                    else CHECK(float_bits(actual[i])==float_bits(expected[i]));
                } else CHECK(actual[i]==12345.0f);
            }
            for(int i=0;i<4;i++)CHECK(scratch[i]==9876.0f);
            for(uint64_t i=vx_packed_linear_scratch_elements((uint64_t)k,R,(uint32_t)bits,(uint32_t)n)+4u;
                    i<TEST_SCRATCH(MAX_K)+8u;i++)CHECK(scratch[i]==9876.0f);
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
    /* Exact allocation spans make ASan check the larger-row panel and its
     * two-row remainder, including the last output column and odd K tails. */
    const uint32_t widths[]={31u,32u,33u,63u,65u,127u,129u,255u,257u};
    for(uint32_t bits=2;bits<=8;bits=bits==4?8:bits+1)
    for(unsigned ki=0;ki<sizeof(widths)/sizeof(widths[0]);ki++)
    for(uint32_t n=4;n<=5;n++)for(uint32_t small=0;small<2u;small++) {
        uint32_t k=widths[ki],rows=small?2u:VX_PACKED_LINEAR_TILE_ROWS+2u;
        uint32_t words=(k*bits+31u)/32u,groups=(k+31u)/32u;
        float* input=(float*)calloc((size_t)rows*k,sizeof(float));
        float* output=(float*)malloc((size_t)rows*n*sizeof(float));
        float* scales=(float*)malloc((size_t)n*groups*sizeof(float));
        uint32_t* packed=(uint32_t*)malloc((size_t)n*words*sizeof(uint32_t));
        float* scratch=(float*)malloc((size_t)vx_packed_linear_scratch_elements(k,rows,bits,n)*sizeof(float));
        CHECK(input&&output&&scales&&packed&&scratch);
        for(uint32_t i=0;i<n*words;i++)packed[i]=random_word();
        for(uint32_t i=0;i<n*groups;i++)scales[i]=1.0f;
        VxLLMOpArgs args={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=(int)rows,.key_dim=(int)k,.value_dim=(int)n,
            .groups=(int)bits,.rate=32,.normalize=8,.x={input,(const float*)packed,scales,NULL},.y={output}};
        CHECK(!vx_packed_linear_forward(&args,scratch));
        for(uint32_t i=0;i<rows*n;i++)CHECK(output[i]==0.0f);
        free(input);free(output);free(scales);free(packed);free(scratch);
    }
    return 0;
}

int test_packed_linear_lowbit_arithmetic(void) {
    enum {K=65,N=7,R=9};
    float x[K*R],scales[N*3],bias[N],actual[N*R],reference[N*R],scratch[TEST_SCRATCH(K)];
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

int test_packed_linear_row_tiles(void) {
    enum {R=402,MAX_K=1283,MAX_N=20};
    static float x[R*MAX_K],scales[MAX_N*((MAX_K+31)/32)],bias[MAX_N];
    static float actual[R*MAX_N],expected[R*MAX_N],scratch[TEST_SCRATCH(MAX_K)+8];
    static uint32_t packed[MAX_N*((MAX_K+3)/4)];
    const int row_counts[]={1,2,3,4,6,7,8,10,16,17,31,32,33,63,64,65,66,128,130,402};
    /* Small K exercises every storage/group pair at every row boundary.
     * Realistic W3/W4 widths cover the two-row microtile, complete group32
     * dots and a partial final group against the independent integer oracle. */
    for(int fixture=0;fixture<20;fixture++) {
        int bits=fixture<16?(int[]){2,3,4,8}[fixture/4]:fixture<18?4:3;
        int group=fixture<16?32<<(fixture%4):32;
        int k=fixture<16?65:fixture%2==0?320:1283,n=fixture<16?8:20;
        int words=(k*bits+31)/32,groups=(k+group-1)/group;
        for(int r=0;r<R;r++)for(int j=0;j<k;j++) {
            float scale=(float)(1+r%4)*.03125f;
            x[r*k+j]=(float)(j==0?127:(int)(random_word()%255)-127)*scale;
            if(r%17==0)x[r*k+j]=0.0f;
        }
        x[130*k]=__builtin_nanf("");x[257*k]=-__builtin_inff();
        for(int i=0;i<n*words;i++)packed[i]=random_word();
        for(int out=0;out<n;out++) {
            bias[out]=(float)(out-7)*.03125f;
            for(int g=0;g<groups;g++)scales[out*groups+g]=(float)(1+random_word()%101)*.00013f;
        }
        for(int r=0;r<R;r++) {
            float maximum=0.0f;int valid=1;
            for(int j=0;j<k;j++) {
                float value=x[r*k+j];if(!finite_value(value)){valid=0;break;}
                float absolute=value<0.0f?-value:value;if(absolute>maximum)maximum=absolute;
            }
            float as=maximum==0.0f?1.0f:maximum/127.0f;
            for(int out=0;out<n;out++) {
                float sum=0.0f;
                if(valid)for(int g=0;g<groups;g++) {
                    int32_t dot=0;
                    for(int j=g*group;j<k&&j<(g+1)*group;j++)dot+=
                        oracle_i8(x[r*k+j]/as)*((int32_t)oracle_code(packed+out*words,(uint32_t)j,(uint32_t)bits)-(1<<(bits-1)));
                    sum+=(float)dot*(as*scales[out*groups+g]);
                }
                expected[r*n+out]=valid?sum+bias[out]:__builtin_nanf("");
            }
        }
        for(unsigned ri=0;ri<sizeof(row_counts)/sizeof(row_counts[0]);ri++)for(int partial=0;partial<2;partial++) {
            int rows=row_counts[ri];uint64_t total=(uint64_t)rows*n,first=partial?3u:0u,units=partial?total-6u:total;
            if(partial&&total<=6u)continue;
            VxLLMOpArgs args={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=rows,.key_dim=k,.value_dim=n,
                .groups=bits,.rate=group,.normalize=8,.x={x,(const float*)packed,scales,bias},.y={actual},
                .first_unit=first,.unit_count=units};
            for(int i=0;i<R*MAX_N;i++)actual[i]=12345.0f;
            for(int i=0;i<TEST_SCRATCH(MAX_K)+8;i++)scratch[i]=9876.0f;
            CHECK(!vx_packed_linear_forward(&args,scratch+4));
            for(int i=0;i<R*MAX_N;i++) {
                if((uint64_t)i>=first&&(uint64_t)i<first+units) {
                    if(expected[i]!=expected[i])CHECK(actual[i]!=actual[i]);
                    else CHECK(float_bits(actual[i])==float_bits(expected[i]));
                } else CHECK(actual[i]==12345.0f);
            }
            for(int i=0;i<4;i++)CHECK(scratch[i]==9876.0f);
            for(uint64_t i=vx_packed_linear_scratch_elements((uint64_t)k,(uint64_t)rows,(uint32_t)bits,(uint32_t)n)+4u;
                    i<TEST_SCRATCH(MAX_K)+8u;i++)CHECK(scratch[i]==9876.0f);
        }
    }
    CHECK(vx_packed_linear_scratch_elements(UINT64_MAX,R,4u,4u)==UINT64_MAX);
    return 0;
}

#ifndef __wasm__
static int test_packed_linear_parallel_pool(void) {
    /* Verify that N-axis panel/column partitioning across worker threads is
     * bit-identical to serial execution for single-row decode, 2-row and 4-row
     * decode batching, and multi-tile prefill across all bit-widths/modes. */
    enum {R=66,K=129,N=35};
    static float x[R*K],scales[N*5],bias[N],serial[R*N],parallel[R*N],scratch[TEST_SCRATCH(K)+8];
    static uint32_t packed[N*33];
    const int row_counts[]={1,2,3,4,6,8,17,66};
    const int col_counts[]={4,7,16,35};
    VxKernelThreadPool* pool=vx_kernel_thread_pool_create(4);
    CHECK(pool!=NULL);
    vx_packed_parallel_macs_test=1u;
    for(int sse2_only=0;sse2_only<2;sse2_only++) {
        vx_packed_force_sse2_test=sse2_only;
        for(uint32_t bits=2;bits<=8;bits=bits==4?8:bits+1)
        for(int group=32;group<=128;group*=4) {
            int words=(K*(int)bits+31)/32,groups=(K+group-1)/group;
            memset(packed,0,sizeof(packed));
            for(int r=0;r<R;r++)for(int j=0;j<K;j++)
                x[r*K+j]=(float)((int)(random_word()%255)-127)*(float)(1+r%5)*.03125f;
            x[5*K]=__builtin_nanf("");x[12*K]=__builtin_inff();
            for(int out=0;out<N;out++) {
                bias[out]=(float)(out-11)*.015625f;
                for(int g=0;g<groups;g++)scales[out*groups+g]=(float)(1+random_word()%63)*.00037f;
                for(int j=0;j<K;j++)oracle_store(packed+out*words,(uint32_t)j,bits,random_word()&((1u<<bits)-1u));
            }
            for(unsigned ri=0;ri<sizeof(row_counts)/sizeof(row_counts[0]);ri++)
            for(unsigned ni=0;ni<sizeof(col_counts)/sizeof(col_counts[0]);ni++)
            for(int arithmetic=8;arithmetic<=32;arithmetic*=2) {
                int rows=row_counts[ri],n=col_counts[ni];
                if(arithmetic!=8 && (rows>4 || n>7))continue;
                VxLLMOpArgs args={.kind=VX_OP_PACKED_LINEAR,.batch=1,.sequence=rows,.key_dim=K,.value_dim=n,
                    .groups=(int)bits,.rate=group,.normalize=arithmetic,
                    .x={x,(const float*)packed,scales,bias},.y={serial}};
                for(int i=0;i<rows*n;i++)serial[i]=parallel[i]=12345.0f;
                CHECK(!vx_packed_linear_forward(&args,scratch));
                args.y[0]=parallel;
                for(int i=0;i<TEST_SCRATCH(K)+8;i++)scratch[i]=9876.0f;
                VxKernelThreadPoolScope scope=vx_kernel_thread_pool_scope_enter(pool);
                CHECK(vx_kernels_thread_count()==4);
                int status=vx_packed_linear_forward(&args,scratch+4);
                vx_kernel_thread_pool_scope_leave(scope);
                CHECK(!status);
                for(int i=0;i<rows*n;i++) {
                    if(serial[i]!=serial[i])CHECK(parallel[i]!=parallel[i]);
                    else CHECK(float_bits(parallel[i])==float_bits(serial[i]));
                }
                for(int i=0;i<4;i++)CHECK(scratch[i]==9876.0f);
                for(uint64_t i=vx_packed_linear_scratch_elements(K,(uint64_t)rows,bits,(uint32_t)n)+4u;
                        i<TEST_SCRATCH(K)+8u;i++)CHECK(scratch[i]==9876.0f);
            }
        }
    }
    vx_packed_parallel_macs_test=16384u;
    vx_packed_force_sse2_test=0;
    vx_kernel_thread_pool_destroy(pool);
    return 0;
}

static int run_all_tests(void) {
    int result=test_packed_linear_i3_unpack();if(result)return result;
    result=test_packed_linear_i8();if(result)return result;
    result=test_packed_linear_i8_tiles();if(result)return result;
    result=test_packed_linear_exact_bounds();if(result)return result;
    result=test_packed_linear_lowbit_arithmetic();if(result)return result;
    result=test_packed_linear_row_tiles();if(result)return result;
    return 0;
}

int main(void) {
    for(int sse2_only=0;sse2_only<2;sse2_only++) {
        vx_packed_force_sse2_test=sse2_only;
        int result=run_all_tests();
        if(result)return result;
    }
    vx_packed_force_sse2_test=0;
    int result=test_packed_linear_parallel_pool();
    if(result)return result;
    puts("packed I2/I3/I4/I8 integer oracle, tails, tiles and arithmetic controls passed");return 0;
}
#endif
