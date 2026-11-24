/* Execute the authored CUDA bodies cooperatively on a host. Physical GPU
 * qualification remains separate; this covers storage decoding/addressing. */
#if defined(__unix__)
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <ucontext.h>
#include <vector>
#include "../include/volvoxai_enums.h"
using vx_u32=std::uint32_t;
using vx_i32=std::int32_t;
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_SHARED static
#define VX_CUDA_FORCEINLINE inline
struct Dimension {vx_u32 x=0,y=0,z=0;};
static Dimension threadIdx,blockIdx,blockDim;
static std::array<ucontext_t,256> contexts;
static std::array<std::array<unsigned char,16384>,256> stacks;
static std::array<bool,256> finished;
static ucontext_t scheduler;
static unsigned lane;
static std::function<void()> body;
static void vx_sync_threads(){assert(!swapcontext(&contexts[lane],&scheduler));}
static vx_u32 vx_global_x(){return blockIdx.x*blockDim.x+threadIdx.x;}
static float vx_min(float a,float b){return a<b?a:b;}
static float vx_max(float a,float b){return a>b?a:b;}
static float vx_sigmoid(float x){return 1.f/(1.f+std::exp(-x));}
static void enter(){body();finished[lane]=true;}
static void launch(Dimension grid,Dimension block,const std::function<void()>& function) {
    blockIdx=grid;blockDim=block;body=function;finished.fill(false);
    unsigned count=block.x*block.y;assert(count&&count<=256);
    for(unsigned i=0;i<count;i++) {
        assert(!getcontext(&contexts[i]));contexts[i].uc_stack.ss_sp=stacks[i].data();
        contexts[i].uc_stack.ss_size=stacks[i].size();contexts[i].uc_link=&scheduler;
        makecontext(&contexts[i],enter,0);
    }
    for(;;) {
        unsigned done=0;
        for(lane=0;lane<count;lane++) {
            threadIdx={lane%block.x,lane/block.x,0};
            assert(!swapcontext(&scheduler,&contexts[lane]));done+=finished[lane];
        }
        if(done==count)break;
        assert(!done);
    }
}
#include "../src/backends/cuda/kernels/cuda_forward_weight_storage_kernels.inc"

static float from_bits(vx_u32 bits){float value;std::memcpy(&value,&bits,4);return value;}
static vx_u32 bits_of(float value){vx_u32 bits;std::memcpy(&bits,&value,4);return bits;}
static float half_oracle(std::uint16_t raw) {
    unsigned exponent=(raw>>10)&31,mantissa=raw&1023;
    if(exponent==31)return from_bits((vx_u32(raw&32768)<<16)|0x7f800000u|(mantissa<<13));
    float value=std::ldexp(float(mantissa+(exponent?1024:0)),exponent?int(exponent)-25:-24);
    return std::copysign(value,raw&32768?-1.f:1.f);
}
static void near(float actual,float expected) {
    assert(std::isfinite(actual)&&std::isfinite(expected));
    if(std::fabs(actual-expected)>2e-5f*(1+std::fabs(expected))) {
        std::fprintf(stderr,"actual=%g expected=%g\n",actual,expected);std::abort();
    }
}
struct Storage {
    std::vector<float> values;
    std::vector<std::uint16_t> compact;
    vx_u32 dtype;
    Storage(unsigned count,vx_u32 type,unsigned seed):values(count),compact(count),dtype(type) {
        for(unsigned i=0;i<count;i++) {
            values[i]=float(int((i*7+seed)%23)-11)*.125f;
            vx_u32 bits=bits_of(values[i]);
            compact[i]=type==VX_DTYPE_BF16?std::uint16_t(bits>>16):
                std::uint16_t(((bits>>16)&0x8000u)|((bits&0x7fffffffu)?
                    ((((bits>>23)&255u)-112u)<<10)|((bits>>13)&1023u):0));
        }
    }
    const void* data()const{return dtype==VX_DTYPE_F32?(const void*)values.data():(const void*)compact.data();}
};
static void decoding() {
    for(vx_u32 raw=0;raw<65536;raw++) {
        std::uint16_t value=std::uint16_t(raw);
        assert(bits_of(vx_weight_storage_load(&value,VX_DTYPE_F16,0))==bits_of(half_oracle(value)));
        assert(bits_of(vx_weight_storage_load(&value,VX_DTYPE_BF16,0))==raw<<16);
    }
}
static void dense(vx_u32 dtype) {
    constexpr unsigned rows=17,k=19,n=21;
    Storage input(rows*k,VX_DTYPE_F32,5),weight(k*n,dtype,7),bias(n,VX_DTYPE_F32,3);
    std::vector<float> out(rows*n);
    for(unsigned layout=0;layout<2;layout++)for(unsigned has_bias=0;has_bias<2;has_bias++) {
        for(unsigned y=0;y<(rows+15)/16;y++)for(unsigned x=0;x<(n+15)/16;x++)
            launch({x,y,0},{16,16,1},[&]{vx_cuda_matmul_storage_f32(input.values.data(),weight.data(),
                bias.values.data(),out.data(),rows,k,n,layout,has_bias,dtype);});
        for(unsigned r=0;r<rows;r++)for(unsigned o=0;o<n;o++) {
            float sum=0;for(unsigned d=0;d<k;d++)sum+=input.values[r*k+d]*weight.values[layout?o*k+d:d*n+o];
            near(out[r*n+o],sum+(has_bias?bias.values[o]:0));
        }
    }
}
static void lookup(vx_u32 dtype) {
    Storage weight(35,dtype,2);std::array<vx_i32,3> tokens={6,0,3};std::array<float,15> out;
    launch({0,0,0},{32,1,1},[&]{vx_cuda_embedding_storage_f32(tokens.data(),weight.data(),out.data(),3,5,dtype);});
    for(unsigned i=0;i<out.size();i++)near(out[i],weight.values[tokens[i/5]*5+i%5]);
    Storage bank(2*3*5,dtype,3);std::array<vx_i32,2> indices={-1,0};std::array<float,20> gathered;
    launch({0,0,0},{32,1,1},[&]{vx_cuda_gather_storage_f32(bank.data(),indices.data(),gathered.data(),2,3,5,2,dtype);});
    for(unsigned outer=0;outer<2;outer++)for(unsigned i=0;i<2;i++)for(unsigned inner=0;inner<5;inner++)
        near(gathered[(outer*2+i)*5+inner],bank.values[(outer*3+(i?0:2))*5+inner]);
}
static void linear(vx_u32 dtype) {
    Storage input(2*5,VX_DTYPE_F32,7),weight(3*5*7,dtype,9),bias(3*7,VX_DTYPE_F32,1);
    float ids[]={3,1,2,0},routes[]={.25f,.75f,.4f,.6f};vx_u32 slots[]={2,UINT32_MAX,0,1};float output[14];
    launch({0,0,0},{32,1,1},[&]{vx_cuda_moe_linear_storage_f32(input.values.data(),weight.data(),
        bias.values.data(),ids,routes,output,2,5,7,3,2,1,slots,4,dtype);});
    for(unsigned r=0;r<2;r++)for(unsigned o=0;o<7;o++) {
        float sum=0;
        for(unsigned i=0;i<2;i++) {
            unsigned e=slots[unsigned(ids[r*2+i])];if(e==UINT32_MAX)continue;
            float value=bias.values[e*7+o];
            for(unsigned d=0;d<5;d++)value+=input.values[r*5+d]*weight.values[(e*5+d)*7+o];
            sum+=routes[r*2+i]*value;
        }
        near(output[r*7+o],sum);
    }
}
static void gated(vx_u32 gt,vx_u32 ut,vx_u32 dt) {
    constexpr unsigned rows=2,experts=3,din=3,hidden=5,dout=7;
    Storage x(rows*din,VX_DTYPE_F32,2),g(experts*din*hidden,gt,3),u(experts*din*hidden,ut,5),
        d(experts*hidden*dout,dt,7);
    float ids[]={2,0,1,2},routes[]={.25f,.75f,.6f,.4f},output[rows*dout];
    for(unsigned scale=0;scale<2;scale++)for(float limit:{0.f,.7f}) {
        for(unsigned row=0;row<rows;row++)launch({row,0,0},{128,1,1},[&]{
            vx_cuda_moe_gated_mlp_storage_f32(x.values.data(),g.data(),u.data(),d.data(),ids,routes,
                output,rows,din,hidden,dout,experts,2,scale,limit,gt,ut,dt);});
        for(unsigned r=0;r<rows;r++)for(unsigned o=0;o<dout;o++) {
            float sum=0;
            for(unsigned k=0;k<2;k++) {
                unsigned e=unsigned(ids[r*2+k]);float route=routes[r*2+k],value=0;
                for(unsigned h=0;h<hidden;h++) {
                    float gate=0,up=0;
                    for(unsigned i=0;i<din;i++) {
                        float v=x.values[r*din+i]*(scale?route:1.f);
                        gate+=v*g.values[(e*din+i)*hidden+h];up+=v*u.values[(e*din+i)*hidden+h];
                    }
                    if(limit>0){gate=std::fmin(gate,limit);up=std::fmax(-limit,std::fmin(up,limit));}
                    value+=gate/(1+std::exp(-gate))*up*d.values[(e*hidden+h)*dout+o];
                }
                sum+=(scale?1.f:route)*value;
            }
            near(output[r*dout+o],sum);
        }
    }
}
int main() {
    decoding();
    for(vx_u32 type:{VX_DTYPE_F32,VX_DTYPE_F16,VX_DTYPE_BF16}){dense(type);lookup(type);linear(type);gated(type,type,type);}
    gated(VX_DTYPE_F16,VX_DTYPE_BF16,VX_DTYPE_F32);
    std::puts("CUDA authored F32/F16/BF16 storage kernels: decode, dense, lookup and MoE numerical PASS");
}
#else
int main(){return 77;}
#endif
