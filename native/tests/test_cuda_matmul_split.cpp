/* This cooperative host emulator requires GCC x86/SSE NaN operand semantics.
 * Other hosts still run the separate physical CUDA numerical gates. */
#if defined(__GNUC__) && !defined(__clang__) && defined(__SSE__) && \
    (defined(__i386__) || defined(__x86_64__)) && defined(__unix__)
/* Cooperative host execution of the literal original compact body and the
 * new private include. This is an addressing/numerical proof, not GPU code. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <ucontext.h>
#include <vector>
using vx_u32=std::uint32_t;
using vx_u64=std::uint64_t;
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_SHARED static
#define VX_CUDA_FORCEINLINE inline
struct Dimension { vx_u32 x=0,y=0,z=0; };
static Dimension threadIdx,blockIdx,blockDim;
static constexpr unsigned lanes=128,stack_bytes=16384;
static std::array<ucontext_t,lanes> contexts;
static std::array<std::array<unsigned char,stack_bytes>,lanes> stacks;
static std::array<bool,lanes> finished;
static ucontext_t scheduler;
static unsigned current_lane;
static std::function<void()> kernel;
static void vx_sync_threads() {
    if(swapcontext(&contexts[current_lane],&scheduler)) std::abort();
}
static float vx_mul_rn(float a,float b) {
    asm volatile("mulss %1, %0" : "+x"(a) : "x"(b)); return a;
}
static float vx_add_rn(float a,float b) {
    asm volatile("addss %1, %0" : "+x"(a) : "x"(b)); return a;
}
static void enter_lane() { kernel(); finished[current_lane]=true; }
static void launch(Dimension grid_index,Dimension block,const std::function<void()>& body) {
    blockIdx=grid_index;blockDim=block;kernel=body;finished.fill(false);
    unsigned threads=block.x*block.y;
    for(unsigned lane=0;lane<threads;++lane) {
        if(getcontext(&contexts[lane])) std::abort();
        contexts[lane].uc_stack.ss_sp=stacks[lane].data();
        contexts[lane].uc_stack.ss_size=stack_bytes;
        contexts[lane].uc_link=&scheduler;
        makecontext(&contexts[lane],enter_lane,0);
    }
    for(;;) {
        unsigned count=0;
        for(current_lane=0;current_lane<threads;++current_lane) {
            threadIdx={current_lane%block.x,current_lane/block.x,0};
            if(swapcontext(&scheduler,&contexts[current_lane])) std::abort();
            count+=finished[current_lane];
        }
        if(count==threads) break;
        if(count) { std::fprintf(stderr,"divergent barrier\n"); std::abort(); }
    }
}
#include TEST_CUDA_SPLIT_SOURCE
#include "test_cuda_argument_record.h"
static float bits_float(vx_u32 bits) { float v;std::memcpy(&v,&bits,4);return v; }
static vx_u32 float_bits(float v) { vx_u32 b;std::memcpy(&b,&v,4);return b; }
static vx_u32 random_state=98743;
static float sample() {
    random_state=random_state*1664525u+1013904223u;
    return (float(random_state>>8)/16777216.f-.5f)*.3f;
}
struct Case { unsigned m,k,n,kind; bool bias; };
static bool run(Case c,bool out_in,unsigned split_k_slices,unsigned test,unsigned& payload_differences) {
    constexpr unsigned guard=8;
    constexpr float sentinel=-98765.5f;
    random_state=98743u+c.m*997u+c.k*17u+c.n+c.kind*313u+unsigned(out_in);
    std::vector<float> input(c.m*c.k),weight(c.k*c.n),bias(c.n);
    for(auto& v:input) v=sample();for(auto& v:weight) v=sample();for(auto& v:bias) v=sample();
    if(c.kind==1) { // signed zeros, including bias and empty-K identity
        for(auto& v:input) v=-0.f;
        for(unsigned i=0;i<weight.size();++i) weight[i]=i%2?-1.f:1.f;
        for(unsigned i=0;i<bias.size();++i) bias[i]=i%2?-0.f:0.f;
    } else if(c.kind==2) { // NaN/Inf/0 multiplication; one explicit NaN payload
        std::array<float,8> values={bits_float(0x7fc00111u),INFINITY,-INFINITY,0.f,-0.f,1.f,-1.f,2.f};
        for(unsigned i=0;i<input.size();++i) input[i]=values[i%8];
        for(unsigned i=0;i<weight.size();++i) weight[i]=i%3==0?0.f:i%3==1?1.f:-1.f;
        for(unsigned i=0;i<bias.size();++i) bias[i]=i%2?-0.f:0.f;
    } else if(c.kind==3) { // exactly representable cancellation across partitions
        for(unsigned i=0;i<input.size();++i) input[i]=((i%c.k)&1u)?-16.f:16.f;
        for(auto& v:weight) v=0.0625f;
        for(auto& v:bias) v=0.f;
    } else if(c.kind==4) { // subnormal inputs and products without FTZ policy change
        for(unsigned i=0;i<input.size();++i) input[i]=bits_float((i%3)+1);
        for(unsigned i=0;i<weight.size();++i) weight[i]=i%2?0.5f:-0.5f;
        for(auto& v:bias) v=-0.f;
    } else if(c.kind==5 || c.kind==6) { // payload order across a stage boundary
        for(auto& v:input) v=1.f;
        for(unsigned k=0;k<c.k;++k) for(unsigned col=0;col<c.n;++col)
            weight[out_in?col*c.k+k:k*c.n+col]=k==15u?bits_float(c.kind==5?0x7fc00111u:0x7fc00222u):
                k==16u?bits_float(c.kind==5?0x7fc00222u:0x7fc00111u):1.f;
    } else if(c.kind==7) {
        for(auto& v:input) v=1.f;for(auto& v:weight) v=-0.f;for(auto& v:bias) v=-0.f;
    } else if(c.kind==8) {
        for(auto& v:input) v=1.f;
        for(unsigned k=0;k<c.k;++k) for(unsigned col=0;col<c.n;++col)
            weight[out_in?col*c.k+k:k*c.n+col]=k==15u?INFINITY:k==16u?-INFINITY:1.f;
    }
    auto saved_input=input,saved_weight=weight,saved_bias=bias;
    std::vector<float> old(c.m*c.n+guard*2,sentinel);
    for(unsigned y=0;y<(c.m+15)/16+1;++y) for(unsigned x=0;x<(c.n+15)/16+1;++x)
        launch({x,y,0},{8,16,1},[&] {
            if(out_in) vx_cuda_matmul_compact_body<true>(input.data(),weight.data(),
                c.bias?bias.data():nullptr,old.data()+guard,c.m,c.k,c.n);
            else vx_cuda_matmul_compact_body<false>(input.data(),weight.data(),
                c.bias?bias.data():nullptr,old.data()+guard,c.m,c.k,c.n);
        });

    bool pass=true;unsigned changed=0,nf_payload=0,zero_diff=0,category_diff=0;
    double maximum=0,square_error=0,square_reference=0;
    std::vector<float> first,first_partials;
    for(unsigned repetition=0;repetition<3;++repetition) {
        // Stored route rebinds both operand allocations and both writable owners.
        auto other_input=input,other_weight=weight,other_bias=bias;
        auto* in=repetition==1?other_input.data():input.data();
        auto* w=repetition==1?other_weight.data():weight.data();
        auto* b=c.bias?(repetition==1?other_bias.data():bias.data()):nullptr;
        std::vector<float> actual(c.m*c.n+2*guard,sentinel);
        std::vector<float> partials(split_k_slices*c.m*c.n+2*guard,sentinel);
        auto partial_record=cuda_argument_record(in,w,partials.data()+guard,c.m,c.k,c.n,split_k_slices);
        auto reduce_record=cuda_argument_record(in,w,b,static_cast<const float*>(partials.data()+guard),
            actual.data()+guard,c.m,c.k,c.n,vx_u32(out_in),split_k_slices);
        for(unsigned z=0;z<split_k_slices+1;++z) for(unsigned y=0;y<(c.m+31)/32+1;++y)
            for(unsigned x=0;x<(c.n+31)/32+1;++x)
                launch({x,y,z},{8,16,1},[&] {
                    auto body=out_in?vx_cuda_matmul_splitk_out_in_f32:vx_cuda_matmul_splitk_f32;
                    if(repetition) body(nullptr,nullptr,nullptr,0,0,0,0,partial_record.bytes.data());
                    else body(in,w,partials.data()+guard,c.m,c.k,c.n,split_k_slices,nullptr);
                });
        for(unsigned x=0;x<(c.m*c.n+127)/128+1;++x)
            launch({x,0,0},{128,1,1},[&] {
                if(repetition) vx_cuda_matmul_splitk_reduce_f32(nullptr,nullptr,nullptr,nullptr,
                    nullptr,0,0,0,999,0,reduce_record.bytes.data());
                else vx_cuda_matmul_splitk_reduce_f32(in,w,b,partials.data()+guard,
                    actual.data()+guard,c.m,c.k,c.n,out_in,split_k_slices,nullptr);
            });


        for(unsigned i=0;i<guard;++i) {
            pass=pass&&actual[i]==sentinel&&actual[guard+c.m*c.n+i]==sentinel;
            pass=pass&&partials[i]==sentinel&&partials[guard+split_k_slices*c.m*c.n+i]==sentinel;
        }
        if(repetition) {
            pass=pass&&std::memcmp(first.data(),actual.data(),actual.size()*4)==0;
            pass=pass&&std::memcmp(first_partials.data(),partials.data(),partials.size()*4)==0;
        } else {
            first=actual;first_partials=partials;
            for(unsigned r=0;r<c.m;++r) for(unsigned col=0;col<c.n;++col) {
                auto index=r*c.n+col;float a=actual[guard+index],o=old[guard+index];
                bool same=float_bits(a)==float_bits(o);
                if(!same) ++changed;
                if(std::isnan(o)) {
                    if(!std::isnan(a)) ++category_diff;
                    else if(!same) ++nf_payload;
                } else if(std::isinf(o)) {
                    if(!std::isinf(a)||std::signbit(o)!=std::signbit(a)) ++category_diff;
                } else {
                    if(!std::isfinite(a)) { ++category_diff;continue; }
                    if((c.kind==1 || c.kind==7) && !same) ++zero_diff;
                    double oracle=c.bias?bias[col]:0.;
                    for(unsigned k=0;k<c.k;++k)
                        oracle+=double(input[r*c.k+k])*weight[out_in?col*c.k+k:k*c.n+col];
                    double error=double(a)-oracle;
                    maximum=std::max(maximum,std::fabs(error));
                    square_error+=error*error;square_reference+=oracle*oracle;
                    pass=pass&&std::fabs(error)<=2e-5+std::fabs(oracle)*1e-5;
                    pass=pass&&std::fabs(double(o)-oracle)<=2e-5+std::fabs(oracle)*1e-5;
                }
            }
        }
        pass=pass&&(input.empty()||std::memcmp(other_input.data(),saved_input.data(),input.size()*4)==0);
        pass=pass&&(weight.empty()||std::memcmp(other_weight.data(),saved_weight.data(),weight.size()*4)==0);
    }
    double nrms=std::sqrt(square_error/(square_reference?square_reference:1.));
    pass=pass&&nrms<=1e-5&&category_diff==0&&zero_diff==0;
    pass=pass&&(input.empty()||std::memcmp(input.data(),saved_input.data(),input.size()*4)==0);
    pass=pass&&(weight.empty()||std::memcmp(weight.data(),saved_weight.data(),weight.size()*4)==0);
    pass=pass&&std::memcmp(bias.data(),saved_bias.data(),bias.size()*4)==0;
    payload_differences+=nf_payload;
    std::printf("{\"case\":%u,\"rows\":%u,\"K\":%u,\"N\":%u,\"kind\":%u,\"layout\":%u,\"split_k_slices\":%u,\"pass\":%s,\"changed_f32\":%u,\"NF_payload_differences\":%u,\"category_differences\":%u,\"zero_differences\":%u,\"max_abs\":%.17g,\"NRMS\":%.17g}\n",
        test,c.m,c.k,c.n,c.kind,out_in,split_k_slices,pass?"true":"false",changed,nf_payload,category_diff,zero_diff,maximum,nrms);
    return pass;
}
static bool invalid_launches_leave_owners() {
    std::array<float,32> owner;owner.fill(-98765.5f);
    const auto before=owner;
    for(vx_u32 split_k_slices:{0u,1u,3u,5u,32u,UINT32_MAX}) for(bool stored:{false,true}) {
        auto partial_record=cuda_argument_record(static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),owner.data(),1u,1u,1u,split_k_slices);
        auto reduce_record=cuda_argument_record(static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),
            static_cast<const float*>(nullptr),owner.data(),1u,1u,1u,0u,split_k_slices);
        for(bool out_in:{false,true}) launch({0,0,0},{8,16,1},[&] {
            auto body=out_in?vx_cuda_matmul_splitk_out_in_f32:vx_cuda_matmul_splitk_f32;
            body(nullptr,nullptr,owner.data(),1,1,1,stored?2u:split_k_slices,
                stored?partial_record.bytes.data():nullptr);
        });
        launch({0,0,0},{128,1,1},[&] {
            vx_cuda_matmul_splitk_reduce_f32(nullptr,nullptr,nullptr,nullptr,
                owner.data(),1,1,1,0,stored?2u:split_k_slices,
                stored?reduce_record.bytes.data():nullptr);
        });
        if(std::memcmp(owner.data(),before.data(),sizeof(owner))) return false;
    }
    return true;
}
int main() {
    const std::array<Case,21> cases={{{2,512,17,0,true},{31,576,33,0,false},
        {17,513,31,0,true},{33,129,17,0,false},{1,0,9,1,true},
        {1,7,9,1,true},{3,129,17,2,true},{2,512,17,3,false},
        {2,513,17,4,true},{1,1,9,0,true},
        {17,1024,1024,0,true},{17,1025,1025,0,false},{32,1024,1024,0,false},
        {17,1025,33,0,true},{31,3072,33,0,false},{3,4097,17,0,true},
        {2,3072,17,1,true},{2,3072,17,7,true},
        {2,3072,17,5,false},{2,3072,17,6,true},{2,3072,17,8,false}}};
    bool pass=invalid_launches_leave_owners();unsigned test=0,payload=0;
    for(auto c:cases) for(bool layout:{false,true}) for(unsigned split_k_slices:{2u,4u,8u,16u})
        pass=run(c,layout,split_k_slices,test++,payload)&&pass;
    std::printf("SUMMARY cases=%u pass=%s payload_differences=%u\n",test,pass?"true":"false",payload);
    return pass?(payload?2:0):1;
}

#else
int main() { return 77; }
#endif
