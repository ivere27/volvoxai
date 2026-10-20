/* Execute extracted production Linear bodies with cooperative host lanes.
 * This checks addressing, reductions and barriers, not GPU execution/PTX. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <ucontext.h>
#include <vector>
using vx_u32 = std::uint32_t;
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_SHARED static
#define VX_CUDA_FORCEINLINE inline
struct Dimension { vx_u32 x=0,y=0,z=0; };
static Dimension threadIdx,blockIdx,blockDim{128,1,1};
static constexpr unsigned maximum_lanes=256,stack_bytes=16384;
static std::array<ucontext_t,maximum_lanes> contexts;
static std::array<std::array<unsigned char,stack_bytes>,maximum_lanes> stacks;
static std::array<bool,maximum_lanes> finished;
static std::array<float,maximum_lanes> shuffle_values;
static ucontext_t scheduler;
static unsigned current_lane;
static std::function<void()> kernel;
static void vx_sync_threads() {
    if(swapcontext(&contexts[current_lane],&scheduler)) std::abort();
}
static float host_shuffle(float value,vx_u32 delta) {
    shuffle_values[current_lane]=value;
    vx_sync_threads();
    unsigned lane=current_lane&31u;
    float result=shuffle_values[lane+delta<32u?current_lane+delta:current_lane];
    vx_sync_threads();
    return result;
}
static float vx_add_rn(float a,float b) { return a+b; }
static void enter_lane() { kernel(); finished[current_lane]=true; }
static void launch(unsigned bx,unsigned by,unsigned threads,
        const std::function<void()>& body) {
    blockIdx={bx,by,0}; kernel=body; finished.fill(false);
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
            threadIdx={current_lane%blockDim.x,current_lane/blockDim.x,0};
            if(swapcontext(&scheduler,&contexts[current_lane])) std::abort();
            count+=finished[current_lane];
        }
        if(count==threads) break;
        if(count) { std::fprintf(stderr,"divergent barrier\n"); std::abort(); }
    }
}
#include TEST_CUDA_GEMV_SOURCE
#include "test_cuda_argument_record.h"
static vx_u32 state=98743;
static float sample() {
    state=state*1664525u+1013904223u;
    return (float(state>>8)/16777216.f-.5f)*.3f;
}
static bool run(vx_u32 rows,vx_u32 k,vx_u32 n,bool out_in,bool bias,bool specialized,
        bool edges=false) {
    std::vector<float> input(rows*k),weight(k*n),offset(n);
    for(auto& v:input) v=sample();
    for(auto& v:weight) v=sample();
    for(auto& v:offset) v=sample();
    if(edges) {
        const std::array<float,9> values{{-0.f,0.f,-0.f,INFINITY,1.f,-1.f,NAN,1.f,2.f}};
        for(unsigned i=0;i<input.size();++i) input[i]=values[i%values.size()];
        for(unsigned i=0;i<weight.size();++i) weight[i]=i%3==0?0.f:i%3==1?1.f:-1.f;
        for(unsigned i=0;i<offset.size();++i) offset[i]=i%2?-0.f:0.f;
    }
    constexpr unsigned guard=8;
    constexpr float sentinel=-98765.5f;
    std::vector<float> actual(rows*n+2*guard,sentinel),first;
    std::vector<double> oracle(rows*n);
    std::vector<float> strict_oracle(rows*n);
    for(vx_u32 r=0;r<rows;++r) for(vx_u32 c=0;c<n;++c) {
        double sum=bias?offset[c]:0.;
        float strict_sum=bias?offset[c]:0.f;
        for(vx_u32 j=0;j<k;++j) {
            sum+=double(input[r*k+j])*weight[out_in?c*k+j:j*n+c];
            strict_sum+=input[r*k+j]*weight[out_in?c*k+j:j*n+c];
        }
        oracle[r*n+c]=sum;
        strict_oracle[r*n+c]=strict_sum;
    }
    bool pass=true; double max_abs=0.,square_error=0.,square_reference=0.;
    for(unsigned repetition=0;repetition<2;++repetition) {
        std::fill(actual.begin(),actual.end(),sentinel);
        blockDim=specialized?Dimension{128,1,1}:Dimension{8,16,1};
        bool register_tile=rows>32u&&n>=1024u;
        unsigned tile=specialized?(out_in?4:8):(register_tile?64:16);
        unsigned row_tile=specialized?1:(register_tile?32:16);
        for(unsigned by=0;by<(rows+row_tile-1)/row_tile;++by)
            for(unsigned bx=0;bx<(n+tile-1)/tile+1;++bx)
                launch(bx,by,128,[&] {
                    auto body=specialized?vx_cuda_gemv_f32
                        :register_tile?(out_in?vx_cuda_matmul_register_out_in_f32:vx_cuda_matmul_register_f32)
                        :(out_in?vx_cuda_matmul_out_in_f32:vx_cuda_matmul_f32);
                    if(specialized&&out_in) {
                        auto record=cuda_argument_record(input.data(),weight.data(),
                            bias?offset.data():nullptr,actual.data()+guard,rows,k,n,
                            static_cast<const float*>(nullptr),static_cast<float*>(nullptr),vx_u32{0});
                        if(repetition) vx_cuda_gemv_out_in_f32(nullptr,nullptr,nullptr,
                            nullptr,0,0,0,nullptr,nullptr,0,record.bytes.data());
                        else vx_cuda_gemv_out_in_f32(input.data(),weight.data(),
                            bias?offset.data():nullptr,actual.data()+guard,rows,k,n,
                            nullptr,nullptr,0,nullptr);
                        return;
                    }
                    auto record=cuda_argument_record(input.data(),weight.data(),
                        bias?offset.data():nullptr,actual.data()+guard,rows,k,n);
                    if(repetition) body(nullptr,nullptr,nullptr,nullptr,0,0,0,record.bytes.data());
                    else body(input.data(),weight.data(),bias?offset.data():nullptr,
                        actual.data()+guard,rows,k,n,nullptr);
                });
        for(unsigned i=0;i<guard;++i)
            pass=pass&&actual[i]==sentinel&&actual[guard+rows*n+i]==sentinel;
        for(unsigned i=0;i<rows*n;++i) {
            if(!std::isfinite(oracle[i])) {
                pass=pass&&(std::isnan(oracle[i])?std::isnan(actual[guard+i]):
                    double(actual[guard+i])==oracle[i]);
                if(!specialized && register_tile)
                    pass=pass&&(std::isnan(strict_oracle[i])?std::isnan(actual[guard+i]):
                        std::memcmp(&actual[guard+i],&strict_oracle[i],sizeof(float))==0);
                continue;
            }
            double error=std::abs(double(actual[guard+i])-oracle[i]);
            max_abs=std::max(max_abs,error);
            if(!repetition) { square_error+=error*error; square_reference+=oracle[i]*oracle[i]; }
            pass=pass&&std::isfinite(actual[guard+i])&&error<=2e-5+1e-5*std::abs(oracle[i]);
            if(!specialized && register_tile)
                pass=pass&&std::memcmp(&actual[guard+i],&strict_oracle[i],sizeof(float))==0;
        }
        if(!repetition) first=actual;
        else pass=pass&&std::memcmp(first.data(),actual.data(),actual.size()*4)==0;
    }
    double normalized=std::sqrt(square_error/std::max(square_reference,1e-24));
    pass=pass&&normalized<=1e-5;
    std::printf("%s %s rows=%u k=%u n=%u bias=%u edges=%u max_abs=%.9g nrms=%.9g %s\n",
        specialized?"GEMV":rows>32u&&n>=1024u?"register":"compact",
        out_in?"OUT_IN":"IN_OUT",rows,k,n,bias,edges,max_abs,normalized,pass?"PASS":"FAIL");
    return pass;
}
/* Both retained owners must equal separate GEMV and rounded residual Add.
 * The stored route receives invalid direct arguments to exercise all11 slots. */
static bool run_dual(vx_u32 k,vx_u32 n,bool residual_first,bool edges) {
    constexpr unsigned guard=8;
    constexpr float sentinel=-98765.5f;
    std::vector<float> input(k),weight(k*n),residual(n);
    for(auto& v:input) v=sample();
    for(auto& v:weight) v=sample();
    for(auto& v:residual) v=sample();
    if(edges) {
        const std::array<float,9> values{{-0.f,0.f,-0.f,INFINITY,1.f,-1.f,NAN,1.f,2.f}};
        for(unsigned i=0;i<k;++i) input[i]=values[i%values.size()];
        for(unsigned i=0;i<weight.size();++i) weight[i]=i%3==0?0.f:i%3==1?1.f:-1.f;
        for(unsigned i=0;i<n;++i) residual[i]=values[(i+2)%values.size()];
    }
    std::vector<float> separate(n+2*guard,sentinel);
    blockDim={128,1,1};
    for(unsigned bx=0;bx<(n+3)/4+1;++bx)
        launch(bx,0,128,[&] { vx_cuda_gemv_out_in_f32(input.data(),weight.data(),
            nullptr,separate.data()+guard,1,k,n,nullptr,nullptr,0,nullptr); });
    std::vector<float> dense(n+2*guard,sentinel),final(n+2*guard,sentinel),first_dense,first_final;
    bool pass=true;
    double squared_error[2]={0.,0.},squared_reference[2]={0.,0.};
    for(unsigned stored=0;stored<2;++stored) {
        std::fill(dense.begin(),dense.end(),sentinel);
        std::fill(final.begin(),final.end(),sentinel);
        auto record=cuda_argument_record(input.data(),weight.data(),
            static_cast<const float*>(nullptr),dense.data()+guard,vx_u32{1},k,n,
            residual.data(),final.data()+guard,vx_u32(residual_first));
        for(unsigned bx=0;bx<(n+3)/4+1;++bx)
            launch(bx,0,128,[&] {
                if(stored) vx_cuda_gemv_out_in_f32(nullptr,nullptr,nullptr,nullptr,
                    0,0,0,nullptr,nullptr,0,record.bytes.data());
                else vx_cuda_gemv_out_in_f32(input.data(),weight.data(),nullptr,
                    dense.data()+guard,1,k,n,residual.data(),final.data()+guard,
                    residual_first,nullptr);
            });
        for(unsigned i=0;i<guard;++i)
            pass=pass&&dense[i]==sentinel&&dense[n+guard+i]==sentinel&&
                final[i]==sentinel&&final[n+guard+i]==sentinel;
        for(unsigned c=0;c<n;++c) {
            float expected=residual_first?residual[c]+separate[c+guard]:separate[c+guard]+residual[c];
            pass=pass&&((std::isnan(separate[c+guard])&&std::isnan(dense[c+guard]))||
                std::memcmp(&separate[c+guard],&dense[c+guard],sizeof(float))==0);
            pass=pass&&((std::isnan(expected)&&std::isnan(final[c+guard]))||
                std::memcmp(&expected,&final[c+guard],sizeof(float))==0);
            double sum=0.;
            for(unsigned j=0;j<k;++j) sum+=double(input[j])*weight[c*k+j];
            const double references[2]={sum,sum+double(residual[c])};
            const float observed[2]={dense[c+guard],final[c+guard]};
            for(unsigned owner=0;owner<2;++owner) if(std::isfinite(references[owner])) {
                double error=std::abs(double(observed[owner])-references[owner]);
                pass=pass&&std::isfinite(observed[owner])&&error<=2e-5+1e-5*std::abs(references[owner]);
                if(!stored) { squared_error[owner]+=error*error;
                    squared_reference[owner]+=references[owner]*references[owner]; }
            }
        }
        if(!stored) { first_dense=dense;first_final=final; }
        else pass=pass&&std::memcmp(first_dense.data(),dense.data(),dense.size()*4)==0&&
            std::memcmp(first_final.data(),final.data(),final.size()*4)==0;
    }
    for(unsigned owner=0;owner<2;++owner)
        pass=pass&&std::sqrt(squared_error[owner]/std::max(squared_reference[owner],1e-24))<=1e-5;
    std::printf("dual OUT_IN k=%u n=%u residual_first=%u edges=%u %s\n",
        k,n,residual_first,edges,pass?"PASS":"FAIL");
    return pass;
}

/* Adjacent projections keep two dense owners. The compiler's host NaN
 * payload propagation is not a GPU oracle; category is checked against the
 * two original bodies, while direct/stored candidate owners are byte-exact. */
static bool run_projection_pair(vx_u32 k,vx_u32 n,bool swap_weights,bool edges) {
    constexpr unsigned guard=8;
    constexpr float sentinel=-98765.5f;
    std::vector<float> input(k),weight0(k*n),weight1(k*n);
    for(auto& v:input) v=sample();
    for(auto& v:weight0) v=sample();
    for(auto& v:weight1) v=sample();
    if(edges) {
        const std::array<float,8> values{{-0.f,0.f,1.f,-1.f,INFINITY,-INFINITY,NAN,NAN}};
        for(unsigned i=0;i<k;++i) input[i]=values[i%values.size()];
        for(unsigned i=0;i<weight0.size();++i) {
            weight0[i]=i%5==0?NAN:i%3==0?0.f:i%3==1?1.f:-1.f;
            weight1[i]=i%7==0?NAN:i%3==0?-0.f:i%3==1?-1.f:1.f;
        }
    }
    if(swap_weights) weight0.swap(weight1);
    std::array<std::vector<float>,2> separate{{std::vector<float>(n+2*guard,sentinel),std::vector<float>(n+2*guard,sentinel)}};
    blockDim={128,1,1};
    for(unsigned owner=0;owner<2;++owner)
        for(unsigned bx=0;bx<(n+3)/4+1;++bx)
            launch(bx,0,128,[&] { vx_cuda_gemv_out_in_f32(input.data(),
                owner?weight1.data():weight0.data(),nullptr,separate[owner].data()+guard,
                1,k,n,nullptr,nullptr,0,nullptr); });
    bool pass=true;
    std::array<std::vector<float>,2> first;
    for(unsigned stored=0;stored<2;++stored) {
        std::array<std::vector<float>,2> outputs{{std::vector<float>(n+2*guard,sentinel),std::vector<float>(n+2*guard,sentinel)}};
        auto record=cuda_argument_record(input.data(),weight0.data(),weight1.data(),
            outputs[0].data()+guard,outputs[1].data()+guard,vx_u32{1},k,n);
        for(unsigned bx=0;bx<(n+3)/4+1;++bx)
            launch(bx,0,128,[&] {
                if(stored) vx_cuda_gemv_pair_out_in_f32(nullptr,nullptr,nullptr,
                    nullptr,nullptr,0,0,0,record.bytes.data());
                else vx_cuda_gemv_pair_out_in_f32(input.data(),weight0.data(),weight1.data(),
                    outputs[0].data()+guard,outputs[1].data()+guard,1,k,n,nullptr);
            });
        for(unsigned owner=0;owner<2;++owner) {
            double se=0.,sr=0.;
            for(unsigned i=0;i<guard;++i)
                pass=pass&&outputs[owner][i]==sentinel&&outputs[owner][n+guard+i]==sentinel;
            for(unsigned c=0;c<n;++c) {
                const float actual=outputs[owner][guard+c],old=separate[owner][guard+c];
                pass=pass&&((std::isnan(actual)&&std::isnan(old))||std::memcmp(&actual,&old,4)==0);
                double ref=0.;const auto& weights=owner?weight1:weight0;
                for(unsigned j=0;j<k;++j) ref+=double(input[j])*weights[c*k+j];
                ref+=0.;
                if(std::isfinite(ref)) {
                    double error=std::abs(double(actual)-ref);
                    pass=pass&&std::isfinite(actual)&&error<=2e-5+1e-5*std::abs(ref);
                    se+=error*error;sr+=ref*ref;
                } else pass=pass&&(std::isnan(ref)?std::isnan(actual):double(actual)==ref);
            }
            pass=pass&&std::sqrt(se/std::max(sr,1e-24))<=1e-5;
            if(stored) pass=pass&&std::memcmp(first[owner].data(),outputs[owner].data(),outputs[owner].size()*4)==0;
        }
        if(!stored) first=outputs;
        for(vx_u32 rows:{0u,2u}) {
            auto inactive=cuda_argument_record(static_cast<const float*>(nullptr),
                static_cast<const float*>(nullptr),static_cast<const float*>(nullptr),
                outputs[0].data()+guard,outputs[1].data()+guard,rows,k,n);
            for(unsigned bx=0;bx<(n+3)/4+1;++bx)
                launch(bx,0,128,[&] { vx_cuda_gemv_pair_out_in_f32(nullptr,nullptr,nullptr,
                    stored?nullptr:outputs[0].data()+guard,stored?nullptr:outputs[1].data()+guard,
                    stored?0:rows,stored?0:k,stored?0:n,stored?inactive.bytes.data():nullptr); });
            for(unsigned owner=0;owner<2;++owner)
                pass=pass&&std::memcmp(first[owner].data(),outputs[owner].data(),outputs[owner].size()*4)==0;
        }
    }
    std::printf("paired OUT_IN k=%u n=%u swap=%u edges=%u %s\n",k,n,swap_weights,edges,pass?"PASS":"FAIL");
    return pass;
}

/* Additional independent F64 and exceptional tests for the reassociated route.
 * Original 196 cases are executed separately with their original F64 budgets. */
static bool run_fixed_partition_edge(vx_u32 k, bool layout, bool bias, unsigned pattern) {
    const vx_u32 rows=3,n=17; const unsigned guard=8;
    const float sentinel=-98765.5f, maximum=3.402823466e38f;
    std::vector<float> input(rows*k),weight(k*n,1.f),offset(n,bias?-0.f:0.f);
    for(vx_u32 r=0;r<rows;++r) for(vx_u32 j=0;j<k;++j) {
        float v=0.f;
        if(pattern==0) v=j%2?-0.f:0.f;
        if(pattern==1) v=float(int(j%7)-3)*.125f;
        if(pattern==2) v=j%3==0?INFINITY:j%3==1?-INFINITY:NAN;
        if(pattern==3) v=j==0?1e10f:-1e10f;
        if(pattern==4) v=j<2?maximum:-maximum;
        if(pattern==5) v=j%2?-maximum:maximum;
        input[r*k+j]=v;
    }
    if(pattern==3) std::fill(offset.begin(),offset.end(),1.f);
    if(pattern==4) std::fill(offset.begin(),offset.end(),-maximum);
    std::vector<double> oracle(rows*n);std::vector<float> scalar(rows*n);
    for(vx_u32 r=0;r<rows;++r) for(vx_u32 c=0;c<n;++c) {
        double sum=bias?offset[c]:0.;float old=bias?offset[c]:0.f;
        for(vx_u32 j=0;j<k;++j) {
            double term=double(input[r*k+j])*weight[layout?c*k+j:j*n+c];
            sum+=term;old+=input[r*k+j]*weight[layout?c*k+j:j*n+c];
        }
        oracle[r*n+c]=sum;scalar[r*n+c]=old;
    }
    bool pass=true;double se=0.,sr=0.,max_abs=0.,old_max_abs=0.;
    unsigned finite_bits=0,nan_payload_bits=0;std::vector<float> first;
    for(unsigned stored=0;stored<2;++stored) {
        std::vector<float> actual(rows*n+2*guard,sentinel);
        auto record=cuda_argument_record(input.data(),weight.data(),
            bias?offset.data():nullptr,actual.data()+guard,rows,k,n);
        blockDim={8,16,1};
        for(unsigned by=0;by<(rows+15)/16;++by)
            for(unsigned bx=0;bx<(n+15)/16+1;++bx)
                launch(bx,by,128,[&] {
                    auto body=layout?vx_cuda_matmul_out_in_f32:vx_cuda_matmul_f32;
                    if(stored) body(nullptr,nullptr,nullptr,nullptr,0,0,0,record.bytes.data());
                    else body(input.data(),weight.data(),bias?offset.data():nullptr,
                        actual.data()+guard,rows,k,n,nullptr);
                });
        for(unsigned i=0;i<guard;++i)
            pass=pass&&actual[i]==sentinel&&actual[guard+rows*n+i]==sentinel;
        for(unsigned i=0;i<rows*n;++i) {
            float value=actual[guard+i],old=scalar[i];double reference=oracle[i];
            if(std::isfinite(reference)) {
                double error=std::abs(double(value)-reference);
                pass=pass&&std::isfinite(value)&&error<=2e-5+1e-5*std::abs(reference);
                max_abs=std::max(max_abs,error);
                old_max_abs=std::max(old_max_abs,std::abs(double(old)-reference));
                if(!stored){se+=error*error;sr+=reference*reference;}
                if(std::memcmp(&value,&old,4)) ++finite_bits;
            } else pass=pass&&(std::isnan(reference)?std::isnan(value):double(value)==reference);
            // A zero or nonfinite candidate must use the original ascending-K repair.
            if(value==0.f || !std::isfinite(value)) {
                if(std::isnan(value)) {
                    pass=pass&&std::isnan(old);
                    if(std::memcmp(&value,&old,4)) ++nan_payload_bits;
                } else pass=pass&&std::memcmp(&value,&old,4)==0;
            }
            if(pattern==3) pass=pass&&old==0.f&&value==1.f&&reference==1.;
            if(pattern==4||pattern==5) pass=pass&&value==0.f&&std::memcmp(&value,&old,4)==0;
        }
        if(stored) pass=pass&&std::memcmp(first.data(),actual.data(),actual.size()*4)==0;
        else first=actual;
    }
    double nrms=std::sqrt(se/std::max(sr,1e-24));pass=pass&&nrms<=1e-5;
    std::printf("F-edge k=%u layout=%u bias=%u pattern=%u max_abs=%.9g nrms=%.9g old_max_abs=%.9g finite_bits=%u nan_payload_bits=%u %s\n",
        k,layout,bias,pattern,max_abs,nrms,old_max_abs,finite_bits,nan_payload_bits,pass?"PASS":"FAIL");
    return pass;
}
static bool run_fixed_partition_edges() {
    bool pass=true;
    for(vx_u32 k:{0u,1u,3u,17u,129u,131u})
        for(bool layout:{false,true}) for(bool bias:{false,true})
            for(unsigned pattern:{0u,1u,2u})
                pass=run_fixed_partition_edge(k,layout,bias,pattern)&&pass;
    for(bool layout:{false,true}) {
        pass=run_fixed_partition_edge(2,layout,true,3)&&pass;
        pass=run_fixed_partition_edge(3,layout,true,4)&&pass;
        pass=run_fixed_partition_edge(4,layout,false,5)&&pass;
    }
    return pass;
}

/* Device views may be 16-byte aligned or offset by one F32 value. Reusing the
 * same argument record with distinct live owners must bind the current views. */
static bool run_alignment_rebind(vx_u32 k,vx_u32 n,bool out_in) {
    constexpr vx_u32 rows=3,guard=8;
    constexpr float sentinel=-98765.5f;
    std::array<std::vector<float>,4> inputs,weights;
    std::vector<float> bias(n),actual(rows*n+2*guard,sentinel),first;
    for(auto& value:bias) value=sample();
    auto view=[](std::vector<float>& owner,unsigned offset) {
        auto address=reinterpret_cast<std::uintptr_t>(owner.data()+guard);
        return reinterpret_cast<float*>((address+15u)&~std::uintptr_t{15u})+offset;
    };
    auto record=cuda_argument_record(static_cast<const float*>(nullptr),
        static_cast<const float*>(nullptr),static_cast<const float*>(bias.data()),
        actual.data()+guard,rows,k,n);
    bool pass=true;double max_abs=0.,max_nrms=0.;
    for(unsigned binding=0;binding<4;++binding) {
        inputs[binding].assign(rows*k+2*guard+4,sentinel);
        weights[binding].assign(k*n+2*guard+4,sentinel);
        const unsigned input_offset=binding&1u,weight_offset=binding>>1u;
        float* input=view(inputs[binding],input_offset);
        float* weight=view(weights[binding],weight_offset);
        pass=pass&&(reinterpret_cast<std::uintptr_t>(input)&15u)==4u*input_offset;
        pass=pass&&(reinterpret_cast<std::uintptr_t>(weight)&15u)==4u*weight_offset;
        for(vx_u32 i=0;i<rows*k;++i) input[i]=sample();
        for(vx_u32 i=0;i<k*n;++i) weight[i]=sample();
        const auto saved_input=inputs[binding],saved_weight=weights[binding];
        std::vector<double> oracle(rows*n);
        for(vx_u32 r=0;r<rows;++r) for(vx_u32 c=0;c<n;++c) {
            double sum=bias[c];
            for(vx_u32 j=0;j<k;++j)
                sum+=double(input[r*k+j])*weight[out_in?c*k+j:j*n+c];
            oracle[r*n+c]=sum;
        }
        record=cuda_argument_record(static_cast<const float*>(input),
            static_cast<const float*>(weight),static_cast<const float*>(bias.data()),
            actual.data()+guard,rows,k,n);
        for(unsigned stored=0;stored<2;++stored) {
            std::fill(actual.begin(),actual.end(),sentinel);
            blockDim={8,16,1};
            for(unsigned bx=0;bx<(n+15u)/16u+1u;++bx)
                launch(bx,0,128,[&] {
                    auto body=out_in?vx_cuda_matmul_out_in_f32:vx_cuda_matmul_f32;
                    if(stored) body(nullptr,nullptr,nullptr,nullptr,0,0,0,record.bytes.data());
                    else body(input,weight,bias.data(),actual.data()+guard,rows,k,n,nullptr);
                });
            for(vx_u32 i=0;i<guard;++i)
                pass=pass&&actual[i]==sentinel&&actual[guard+rows*n+i]==sentinel;
            double se=0.,sr=0.;
            for(vx_u32 i=0;i<rows*n;++i) {
                double error=std::abs(double(actual[guard+i])-oracle[i]);
                pass=pass&&std::isfinite(actual[guard+i])&&error<=2e-5+1e-5*std::abs(oracle[i]);
                max_abs=std::max(max_abs,error);se+=error*error;sr+=oracle[i]*oracle[i];
            }
            double nrms=std::sqrt(se/std::max(sr,1e-24));
            pass=pass&&nrms<=1e-5;max_nrms=std::max(max_nrms,nrms);
            pass=pass&&std::memcmp(inputs[binding].data(),saved_input.data(),saved_input.size()*4)==0;
            pass=pass&&std::memcmp(weights[binding].data(),saved_weight.data(),saved_weight.size()*4)==0;
            if(stored) pass=pass&&std::memcmp(first.data(),actual.data(),actual.size()*4)==0;
            else first=actual;
        }
    }
    std::printf("alignment-rebind %s rows=%u k=%u n=%u bindings=4 routes=2 max_abs=%.9g nrms=%.9g %s\n",
        out_in?"OUT_IN":"IN_OUT",rows,k,n,max_abs,max_nrms,pass?"PASS":"FAIL");
    return pass;
}

/* A four-way K partition crosses its first and fourth K16 stage around
   K=64 and K=256. Exercise both layouts, bias, row tails and exceptional data
   through the same F64, guarded-owner and stored-argument checks. */
static bool run_prefetch_stage_boundaries() {
    bool pass=true;
    for(vx_u32 k:{63u,64u,65u,255u,256u,257u})
        for(vx_u32 rows:{3u,17u}) for(bool layout:{false,true}) for(bool bias:{false,true})
            pass=run(rows,k,17u,layout,bias,false)&&pass;
    for(vx_u32 k:{65u,257u})
        for(bool layout:{false,true}) for(bool bias:{false,true}) for(unsigned pattern:{0u,2u})
            pass=run_fixed_partition_edge(k,layout,bias,pattern)&&pass;
    return pass;
}

int main() {
    bool pass=true;
    for(auto shape:std::array<std::array<vx_u32,2>,10>{{
            {1,1},{15,3},{16,7},{17,8},{31,9},{32,17},{33,33},
            {257,19},{576,17},{1024,9}}})
        for(bool layout:{false,true}) for(bool bias:{false,true})
            pass=run(1,shape[0],shape[1],layout,bias,true)&&pass;
    for(auto shape:std::array<std::array<vx_u32,3>,21>{{
            {1,1,1},{15,15,7},{16,16,8},{17,17,9},{31,33,17},
            {32,65,31},{33,33,33},{33,17,63},{63,33,64},{65,17,65},
            {33,17,1024},{65,33,1025},{256,3,1057},
            {39,5,1027},{65,17,1057},
            {31,17,1537},{128,63,65},
            {16,128,64},{17,129,65},{31,257,129},{32,128,64}}})
        for(bool layout:{false,true}) for(bool bias:{false,true})
            pass=run(shape[0],shape[1],shape[2],layout,bias,false)&&pass;
    for(auto shape:std::array<std::array<vx_u32,3>,5>{{
            {3,3,17},{17,17,9},{33,3,1025},{17,129,65},{39,5,1027}}})
        for(bool layout:{false,true}) for(bool bias:{false,true})
            pass=run(shape[0],shape[1],shape[2],layout,bias,false,true)&&pass;
    for(auto shape:std::array<std::array<vx_u32,2>,8>{{
            {1,1},{17,3},{33,7},{257,19},{576,17},
            {1536,576},{2048,1024},{3072,1024}}})
        for(bool order:{false,true}) for(bool edges:{false,true})
            pass=run_dual(shape[0],shape[1],order,edges)&&pass;
    for(auto shape:std::array<std::array<vx_u32,2>,8>{{
            {1,1},{17,3},{33,7},{257,19},{576,1536},
            {1024,3072},{2048,1024},{3072,1024}}})
        for(bool swap:{false,true}) for(bool edges:{false,true})
            pass=run_projection_pair(shape[0],shape[1],swap,edges)&&pass;
    pass=run_fixed_partition_edges()&&pass;
    for(auto shape:std::array<std::array<vx_u32,2>,2>{{{128,20},{129,19}}})
        for(bool layout:{false,true})
            pass=run_alignment_rebind(shape[0],shape[1],layout)&&pass;
    pass=run_prefetch_stage_boundaries()&&pass;
    return pass?0:1;
}
