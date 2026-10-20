/* Production dual-output body: F64 oracle and exact two-node F32 arithmetic.
 * This host check does not qualify GPU/PTX execution. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>
using vx_u32=std::uint32_t;
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_FORCEINLINE inline
static vx_u32 host_index;
static vx_u32 vx_global_x() {return host_index;}
static float vx_sigmoid(float value) {
    float x=-value;
    float e=x<-87.f?0.f:std::exp2(std::min(88.f,x)*1.4426950408889634f);
    return 1.f/(1.f+e);
}
#include TEST_CUDA_SILU_MUL_SOURCE
#include "test_cuda_argument_record.h"
static bool run(vx_u32 count,bool edge) {
    constexpr unsigned guard=8;
    constexpr float marker=-98765.5f;
    std::vector<float> input(count),other(count),activation(count+2*guard,marker),output=activation;
    for(vx_u32 i=0;i<count;++i) {
        input[i]=(float(int((i*2654435761u)>>16)-32768)/327.68f);
        other[i]=float(int((i*1277737u)>>16)-32768)/4096.f;
    }
    if(edge) {
        const float values[]={-0.f,0.f,-std::numeric_limits<float>::infinity(),
            std::numeric_limits<float>::infinity(),std::numeric_limits<float>::quiet_NaN(),-87.f,88.f};
        for(vx_u32 i=0;i<count;++i) input[i]=values[i%7];
    }
    auto saved_input=input,saved_other=other;
    std::vector<float> first_activation,first_output;
    for(unsigned stored=0;stored<2;++stored) {
        auto record=cuda_argument_record(input.data(),other.data(),activation.data()+guard,
            output.data()+guard,count);
        for(host_index=0;host_index<count+17;++host_index) {
            if(stored) vx_cuda_silu_mul_dual_f32(nullptr,nullptr,nullptr,nullptr,0,record.bytes.data());
            else vx_cuda_silu_mul_dual_f32(input.data(),other.data(),activation.data()+guard,
                output.data()+guard,count,nullptr);
        }
        if(!stored) {first_activation=activation;first_output=output;}
        else if(std::memcmp(first_activation.data(),activation.data(),activation.size()*4) ||
                std::memcmp(first_output.data(),output.data(),output.size()*4)) return false;
        if(!stored) {
            std::fill(activation.begin(),activation.end(),marker);
            std::fill(output.begin(),output.end(),marker);
        }
    }
    bool pass=std::memcmp(input.data(),saved_input.data(),count*4)==0&&
        std::memcmp(other.data(),saved_other.data(),count*4)==0;
    double max_abs=0.,errors=0.,references=0.;
    for(vx_u32 i=0;i<count;++i) {
        float expected=input[i]*vx_sigmoid(input[i]);
        float product=expected*other[i];
        if(std::isnan(expected)) pass=pass&&std::isnan(activation[guard+i]);
        else pass=pass&&std::memcmp(&expected,&activation[guard+i],4)==0;
        if(std::isnan(product)) pass=pass&&std::isnan(output[guard+i]);
        else pass=pass&&std::memcmp(&product,&output[guard+i],4)==0;
        if(std::isfinite(input[i])) {
            double activated=double(input[i])/(1.+std::exp(-double(input[i])));
            double wanted[]={activated,activated*other[i]};
            double actual[]={activation[guard+i],output[guard+i]};
            for(int j=0;j<2;++j) {
                double error=std::abs(actual[j]-wanted[j]);
                max_abs=std::max(max_abs,error);errors+=error*error;references+=wanted[j]*wanted[j];
                pass=pass&&std::isfinite(actual[j])&&error<=2e-5+1e-5*std::abs(wanted[j]);
            }
        }
    }
    for(unsigned i=0;i<guard;++i)
        pass=pass&&activation[i]==marker&&activation[guard+count+i]==marker&&
            output[i]==marker&&output[guard+count+i]==marker;
    double nrms=std::sqrt(errors/std::max(references,1e-24));
    pass=pass&&nrms<=1e-5;
    std::printf("SiLU/Mul count=%u edge=%u max_abs=%.9g nrms=%.9g %s\n",
        count,edge,max_abs,nrms,pass?"PASS":"FAIL");
    return pass;
}
int main() {
    bool pass=true;
    for(vx_u32 count:{1u,3u,31u,32u,33u,255u,256u,257u,1536u,3072u,128u*3072u})
        for(bool edge:{false,true}) pass=run(count,edge)&&pass;
    return pass?0:1;
}
