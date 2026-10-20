/* Execute production CUDA reduction bodies with cooperative host lanes.
 * This checks F32 arithmetic, indexing and barriers, not PTX/GPU execution. */
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <ucontext.h>
#include <vector>
using vx_u32 = std::uint32_t;
using vx_i32 = std::int32_t;
#define VX_CUDA_GLOBAL
#define VX_CUDA_DEVICE
#define VX_CUDA_SHARED static
#define VX_CUDA_FORCEINLINE inline
struct Dimension { vx_u32 x=0; };
static Dimension threadIdx,blockIdx,blockDim{256};
static constexpr unsigned maximum_lanes=1024,stack_bytes=16384,guard=8;
static constexpr float sentinel=-98765.5f;
static std::array<ucontext_t,maximum_lanes> contexts;
static std::array<std::array<unsigned char,stack_bytes>,maximum_lanes> stacks;
static std::array<bool,maximum_lanes> finished;
static ucontext_t scheduler;
static unsigned current_lane;
static std::function<void()> kernel;
static void vx_sync_threads() {
    if(swapcontext(&contexts[current_lane],&scheduler)) std::abort();
}
template<class T> static T host_shuffle(T value,vx_u32 delta) {
    static std::array<T,maximum_lanes> values;
    values[current_lane]=value;
    vx_sync_threads();
    unsigned lane=current_lane&31u;
    T result=values[lane+delta<32u?current_lane+delta:current_lane];
    vx_sync_threads();
    return result;
}
static float vx_mul_rn(float a,float b) { return a*b; }
static float vx_add_rn(float a,float b) { return a+b; }
static float vx_sqrt(float value) { return std::sqrt(value); }
static vx_u32 vx_global_x() { return blockIdx.x*blockDim.x+threadIdx.x; }
static void enter_lane() { kernel(); finished[current_lane]=true; }
static unsigned threads_for(vx_u32 width,bool argmax=false) {
    unsigned threads=32;
    const unsigned limit=argmax&&width>=16384u?1024u:256u;
    while(threads<width&&threads<limit) threads*=2;
    return threads;
}
static void launch(unsigned row,unsigned threads,const std::function<void()>& body) {
    blockIdx={row};blockDim={threads};kernel=body;finished.fill(false);
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
            threadIdx={current_lane};
            if(swapcontext(&scheduler,&contexts[current_lane])) std::abort();
            count+=finished[current_lane];
        }
        if(count==threads) break;
        if(count) { std::fprintf(stderr,"divergent reduction barrier\n");std::abort(); }
    }
}
#include TEST_CUDA_REDUCTION_SOURCE
#include "test_cuda_argument_record.h"
#ifdef TEST_CUDA_HEAD_SOURCE
#include TEST_CUDA_HEAD_SOURCE
#endif
static vx_u32 random_state=123987;
static float sample() {
    random_state=random_state*1664525u+1013904223u;
    return (float(random_state>>8)/16777216.f-.5f)*4.f;
}
static bool run_norm(vx_u32 rows,vx_u32 width,int kind,bool inplace) {
    std::vector<float> input(rows*width+2*guard,sentinel),weight(width),actual;
    const float epsilon=kind==2?1e-12f:1e-6f;
    for(vx_u32 i=0;i<rows*width;++i) {
        float value=sample();
        if(kind==1) value=std::ldexp(value,int(i%41)-20);
        if(kind==2) value=i&1u?-0.f:0.f;
        input[guard+i]=value;
    }
    for(auto& value:weight) value=sample();
    if(kind==3) input[guard+width/2]=std::numeric_limits<float>::quiet_NaN();
    if(kind==4) input[guard+width/2]=std::numeric_limits<float>::infinity();
    std::vector<double> oracle(rows*width);
    for(vx_u32 row=0;row<rows;++row) {
        double squares=0.;
        for(vx_u32 col=0;col<width;++col) {
            double value=input[guard+row*width+col];squares+=value*value;
        }
        double inverse=1./std::sqrt(squares/double(width)+epsilon);
        for(vx_u32 col=0;col<width;++col)
            oracle[row*width+col]=double(input[guard+row*width+col])*inverse*weight[col];
    }
    std::vector<float> first;
    bool pass=true;double max_abs=0.,error_square=0.,reference_square=0.;
    for(unsigned repetition=0;repetition<2;++repetition) {
        actual=inplace?input:std::vector<float>(input.size(),sentinel);
        for(vx_u32 row=0;row<=rows;++row)
            launch(row,threads_for(width),[&] {
                auto record=cuda_argument_record(inplace?actual.data()+guard:input.data()+guard,
                    weight.data(),actual.data()+guard,rows,width,epsilon);
                if(repetition) vx_cuda_rmsnorm_cooperative_f32(nullptr,nullptr,nullptr,
                    0,0,0,record.bytes.data());
                else vx_cuda_rmsnorm_cooperative_f32(inplace?actual.data()+guard:input.data()+guard,
                    weight.data(),actual.data()+guard,rows,width,epsilon,nullptr);
            });
        for(unsigned i=0;i<guard;++i)
            pass=pass&&actual[i]==sentinel&&actual[guard+rows*width+i]==sentinel;
        for(vx_u32 i=0;i<rows*width;++i) {
            float value=actual[guard+i];double expected=oracle[i];
            if(std::isnan(expected)) pass=pass&&std::isnan(value);
            else if(std::isinf(expected)) pass=pass&&value==expected;
            else {
                double error=std::abs(double(value)-expected);
                max_abs=std::max(max_abs,error);
                pass=pass&&std::isfinite(value)&&error<=2e-5+1e-5*std::abs(expected);
                if(expected==0.) pass=pass&&std::signbit(value)==std::signbit(expected);
                if(!repetition) {error_square+=error*error;reference_square+=expected*expected;}
            }
        }
        if(!repetition) first=actual;
        else pass=pass&&std::memcmp(first.data(),actual.data(),actual.size()*sizeof(float))==0;
    }
    double nrms=std::sqrt(error_square/std::max(reference_square,1e-24));
    pass=pass&&nrms<=1e-5;
    std::printf("RMS rows=%u width=%u kind=%d inplace=%u max_abs=%.9g nrms=%.9g %s\n",
        rows,width,kind,inplace,max_abs,nrms,pass?"PASS":"FAIL");
    return pass;
}
static bool run_argmax(vx_u32 outer,vx_u32 axis,vx_u32 inner,int kind) {
    const vx_u32 outputs=outer*inner;
    std::vector<float> input(outer*axis*inner+2*guard,sentinel);
    for(vx_u32 i=0;i<outer*axis*inner;++i) input[guard+i]=sample();
    for(vx_u32 row=0;row<outputs;++row) {
        vx_u32 base=guard+(row/inner)*axis*inner+row%inner;
        if(kind==1) { // Equal maxima in different warps and loop iterations.
            input[base]=9.f;input[base+(axis-1)*inner]=9.f;
            if(axis>33) input[base+33*inner]=9.f;
        } else if(kind==2) input[base]=std::numeric_limits<float>::quiet_NaN();
        else if(kind==3&&axis>1) input[base+(axis-1)*inner]=std::numeric_limits<float>::quiet_NaN();
        else if(kind>=4&&kind<=7)
            for(vx_u32 j=0;j<axis;++j)
                input[base+j*inner]=kind==4?-std::numeric_limits<float>::infinity():
                    kind==5?std::numeric_limits<float>::infinity():
                    kind==6?(j&1u?-0.f:0.f):std::numeric_limits<float>::quiet_NaN();
        else if(kind==8) input[base+(axis-1)*inner]=10.f;
        else if(kind==9) {
            input[base+std::min(33u,axis-1)*inner]=9.f;
            input[base+(axis-1)*inner]=9.f;
        } else if(kind==10) {
            input[base+std::min(257u,axis-1)*inner]=9.f;
            input[base+std::min(1025u,axis-1)*inner]=9.f;
        }
    }
    std::vector<vx_i32> oracle(outputs),first;
    for(vx_u32 row=0;row<outputs;++row) {
        vx_u32 base=guard+(row/inner)*axis*inner+row%inner;
        float best=input[base];vx_i32 index=0;
        for(vx_u32 j=1;j<axis;++j) if(input[base+j*inner]>best) {
            best=input[base+j*inner];index=vx_i32(j);
        }
        oracle[row]=index;
    }
    bool pass=true;
    for(unsigned repetition=0;repetition<2;++repetition) {
        std::vector<vx_i32> actual(outputs+2*guard,-1234567);
        for(vx_u32 row=0;row<=outputs;++row)
            launch(row,threads_for(axis,true),[&] {
                auto record=cuda_argument_record(input.data()+guard,actual.data()+guard,outer,axis,inner);
                if(repetition) vx_cuda_argmax_cooperative_f32_i32(nullptr,nullptr,0,0,0,
                    record.bytes.data());
                else vx_cuda_argmax_cooperative_f32_i32(input.data()+guard,actual.data()+guard,
                    outer,axis,inner,nullptr);
            });
        for(vx_u32 row=0;row<outputs;++row) pass=pass&&actual[guard+row]==oracle[row];
        for(unsigned i=0;i<guard;++i)
            pass=pass&&actual[i]==-1234567&&actual[guard+outputs+i]==-1234567;
        if(!repetition) first=actual;
        else pass=pass&&first==actual;
    }
    std::printf("ArgMax outer=%u axis=%u inner=%u kind=%d %s\n",
        outer,axis,inner,kind,pass?"PASS":"FAIL");
    return pass;
}
#ifdef TEST_CUDA_HEAD_SOURCE
static vx_u32 head_rng=237413;
static float head_sample(){head_rng=head_rng*1664525u+1013904223u;return(float(head_rng>>8)/16777216.f-.5f)*2.f;}
static float head_bits(vx_u32 u){float f;std::memcpy(&f,&u,4);return f;}
static void head_fail(const char* what,unsigned index,vx_u32 batch,vx_u32 seq,vx_u32 heads,vx_u32 rotary,unsigned mode,unsigned kind,bool stored){std::fprintf(stderr,"FAIL %s index%u B%u S%u H%u R%u mode%u kind%u stored%u\n",what,index,batch,seq,heads,rotary,mode,kind,stored);std::abort();}
static std::uint64_t head_total_hash=1469598103934665603ull;static unsigned head_cases,head_routes;
static void head_check(vx_u32 batch,vx_u32 seq,vx_u32 heads,vx_u32 rotary,unsigned mode,unsigned kind,bool interleaved,bool large=false,bool alias=false) {
 const vx_u32 rows=batch*seq,count=rows*heads*128u,width=heads*128u; const float eps=kind==2?1e-12f:1e-6f;vx_u32 offset=31;
 std::vector<float> input(count),weight(128),powers(rotary/2);std::vector<vx_i32> positions(rows);
 for(unsigned i=0;i<count;i++){float x=head_sample();if(kind==1)x=std::ldexp(x,int(i%41)-20);if(kind==2)x=i&1u?-0.f:0.f;if(kind==6)x=i%4==0?std::numeric_limits<float>::denorm_min():i%4==1?-std::numeric_limits<float>::denorm_min():0.f;input[i]=x;}
 for(unsigned i=0;i<128;i++)weight[i]=head_sample();
 for(unsigned i=0;i<rows;i++)positions[i]=large?INT32_MAX-vx_i32(i*1024):vx_i32(31+i);
 if(kind==3)for(unsigned i=0;i<rows*heads;i++)input[i*128+3]=head_bits(0x7fc12345u);
 if(kind==4)for(unsigned i=0;i<rows*heads;i++)input[i*128+127]=INFINITY;
 if(kind==5){weight[4]=head_bits(0x7fc54321u);weight[5]=-0.f;weight[6]=INFINITY;}
 for(unsigned i=0;i<rotary/2;i++)powers[i]=std::pow(1e6f,(2.f*float(i))/float(rotary));
 std::vector<float> expected[3];for(auto& v:expected)v.assign(count+2*guard,sentinel);
 for(unsigned h=0;h<=rows*heads;h++)launch(h,128,[&]{vx_cuda_rmsnorm_cooperative_f32(input.data(),weight.data(),expected[0].data()+guard,rows*heads,128,eps,nullptr);});
 for(unsigned block=0;block<=(count+127u)/128u;block++)launch(block,128,[&]{vx_cuda_copy_f32(expected[0].data()+guard,expected[1].data()+guard,count,nullptr);});
 for(unsigned block=0;block<=(count+127u)/128u;block++)launch(block,128,[&]{vx_cuda_rope_heads_f32(expected[1].data()+guard,mode?positions.data():nullptr,expected[2].data()+guard,count,seq,width,heads,rotary,powers.data(),offset,interleaved,mode,nullptr);});

 std::vector<float> old_stored[3];for(auto& v:old_stored)v.assign(count+2*guard,sentinel);
 for(unsigned h=0;h<=rows*heads;h++)launch(h,128,[&]{auto image=cuda_argument_record(input.data(),weight.data(),old_stored[0].data()+guard,rows*heads,128u,eps);vx_cuda_rmsnorm_cooperative_f32(nullptr,nullptr,nullptr,0,0,0,image.bytes.data());});
 for(unsigned block=0;block<=(count+127u)/128u;block++)launch(block,128,[&]{auto image=cuda_argument_record(old_stored[0].data()+guard,old_stored[1].data()+guard,count);vx_cuda_copy_f32(nullptr,nullptr,0,image.bytes.data());});
 for(unsigned block=0;block<=(count+127u)/128u;block++)launch(block,128,[&]{auto image=cuda_argument_record(old_stored[1].data()+guard,mode?positions.data():nullptr,old_stored[2].data()+guard,count,seq,width,heads,rotary,powers.data(),offset,vx_u32(interleaved),vx_u32(mode));vx_cuda_rope_heads_f32(nullptr,nullptr,nullptr,0,0,0,0,0,nullptr,0,0,999u,image.bytes.data());});
 for(unsigned out=0;out<3;out++)if(std::memcmp(old_stored[out].data(),expected[out].data(),expected[out].size()*4))head_fail("old direct/stored owner",out,batch,seq,heads,rotary,mode,kind,true);
 std::vector<double> oracle[3];for(auto& v:oracle)v.resize(count);
 for(unsigned h=0;h<rows*heads;h++){double ss=0;for(unsigned d=0;d<128;d++)ss+=double(input[h*128+d])*input[h*128+d];double inverse=1./std::sqrt(ss/128.+double(eps));for(unsigned d=0;d<128;d++)oracle[0][h*128+d]=double(input[h*128+d])*inverse*weight[d];}
 oracle[1]=oracle[0];
 for(unsigned h=0;h<rows*heads;h++)for(unsigned d=0;d<128;d++){
  if(d>=rotary){oracle[2][h*128+d]=oracle[0][h*128+d];continue;}
  unsigned row=h/heads,si=row%seq;vx_i32 pos=mode==1?positions[si]:mode==2?positions[row]:vx_i32(offset+si);unsigned half=rotary/2,pair=interleaved?d/2:d%half,left=interleaved?pair*2:pair,right=interleaved?left+1:left+half;
  // Canonical scalar controls/powers retain their authentic F32 angle. F64
  // evaluates normalization and rotation values independently at that angle.
  float angle=float(pos)/powers[pair];double sine=std::sin(double(angle)),cosine=std::cos(double(angle)),a=oracle[0][h*128+left],b=oracle[0][h*128+right];oracle[2][h*128+d]=d==left?a*cosine-b*sine:a*sine+b*cosine;
 }
 double maximum=0;
 for(bool stored:{false,true}){
  std::vector<float> actual[3];for(auto& v:actual)v.assign(count+2*guard,sentinel);
  // An elided contiguous Reshape retains a second logical descriptor for
  // exactly the same owner. Each head/lane writes the same index/value twice.
  float* flat=alias?actual[0].data()+guard:actual[1].data()+guard;
  for(unsigned h=0;h<=rows*heads;h++)launch(h,128,[&]{
   auto image=cuda_argument_record(input.data(),weight.data(),actual[0].data()+guard,flat,mode?positions.data():nullptr,actual[2].data()+guard,rows,seq,heads,rotary,powers.data(),eps,offset,vx_u32(interleaved),vx_u32(mode));
   if(stored)vx_cuda_head_rms_reshape_rope_f32(nullptr,nullptr,nullptr,nullptr,nullptr,nullptr,0,0,0,0,nullptr,0,0,0,999u,image.bytes.data());
   else vx_cuda_head_rms_reshape_rope_f32(input.data(),weight.data(),actual[0].data()+guard,flat,mode?positions.data():nullptr,actual[2].data()+guard,rows,seq,heads,rotary,powers.data(),eps,offset,interleaved,mode,nullptr);
  });
  if(alias)for(float value:actual[1])if(value!=sentinel)std::abort();
  for(unsigned out=0;out<3;out++){
   const auto& owner=actual[alias&&out==1?0:out];
   double es=0,rs=0;
   for(unsigned i=0;i<owner.size();i++){
    if(std::memcmp(&owner[i],&expected[out][i],4)){
     vx_u32 a,e;std::memcpy(&a,&owner[i],4);std::memcpy(&e,&expected[out][i],4);std::fprintf(stderr,"BITS owner%u actual%08x expected%08x\n",out,a,e);head_fail("full-owner-bitidentity",i,batch,seq,heads,rotary,mode,kind,stored);}
    if(i>=guard&&i<count+guard){double e=oracle[out][i-guard];float a=owner[i];if(std::isfinite(e)){double er=std::abs(double(a)-e);if(!std::isfinite(a)||er>2e-5+1e-5*std::abs(e))head_fail("F64",i,batch,seq,heads,rotary,mode,kind,stored);maximum=std::max(maximum,er);es+=er*er;rs+=e*e;}else if(std::isnan(e)){if(!std::isnan(a))head_fail("F64nan",i,batch,seq,heads,rotary,mode,kind,stored);}else if(a!=e)head_fail("F64inf",i,batch,seq,heads,rotary,mode,kind,stored);}
    unsigned char b[4];std::memcpy(b,&owner[i],4);for(unsigned char v:b)head_total_hash=(head_total_hash^v)*1099511628211ull;
   }
   if(std::sqrt(es/std::max(rs,1e-24))>1e-5)head_fail("F64nrms",out,batch,seq,heads,rotary,mode,kind,stored);
  }
  head_routes++;
 }
 head_cases++;std::printf("PASS B%u S%u H%u R%u mode%u kind%u inter%u large%u %s/direct+stored F64max%.9g\n",batch,seq,heads,rotary,mode,kind,interleaved,large,alias?"exact-alias":"threeowners",maximum);
}
static void head_no_effect(){
 float out[3]={sentinel,sentinel,sentinel};
 for(unsigned geometry=0;geometry<7;geometry++)for(bool stored:{false,true}){
  vx_u32 rows=geometry==0?0u:geometry==5?UINT32_MAX:1u,seq=geometry==1?0u:1u,heads=geometry==2?0u:geometry==6?UINT32_MAX:1u,rot=geometry==3?0u:geometry==4?127u:128u;
  launch(0,128,[&]{auto image=cuda_argument_record((const float*)nullptr,(const float*)nullptr,&out[0],&out[1],(const vx_i32*)nullptr,&out[2],rows,seq,heads,rot,(const float*)nullptr,1e-6f,0u,0u,0u);vx_cuda_head_rms_reshape_rope_f32(nullptr,nullptr,&out[0],&out[1],nullptr,&out[2],stored?0u:rows,stored?0u:seq,stored?0u:heads,stored?0u:rot,nullptr,1e-6f,0,0,0,stored?image.bytes.data():nullptr);});
  if(out[0]!=sentinel||out[1]!=sentinel||out[2]!=sentinel)std::abort();
 }
 launch(0,64,[&]{vx_cuda_head_rms_reshape_rope_f32(nullptr,nullptr,&out[0],&out[1],nullptr,&out[2],1,1,1,128,nullptr,1e-6f,0,0,0,nullptr);});
 puts("PASS uniform inactive/invalid geometry and block no-read/no-write 15 routes");
}
#endif
int main() {
#ifdef TEST_CUDA_HEAD_SOURCE

 for(unsigned h:{8u,16u})for(unsigned mode:{0u,1u,2u})for(bool inter:{false,true})head_check(1,1,h,128,mode,0,inter);
 for(unsigned mode:{0u,1u,2u})for(bool inter:{false,true})head_check(2,3,3,128,mode,0,inter);
 for(unsigned rotary:{2u,64u,126u,128u})for(unsigned mode:{0u,1u,2u})for(bool inter:{false,true})head_check(2,3,1,rotary,mode,0,inter);
 for(unsigned h:{8u,16u})for(unsigned kind=1;kind<=6;kind++)head_check(1,1,h,128,2,kind,false);
 head_check(1,3,1,128,2,0,false,true);head_no_effect();
 std::printf("RESULT cases=%u routes=%u bits=%llu\n",head_cases,head_routes,(unsigned long long)head_total_hash);
 // Append aliases after the original corpus, preserving its inputs and hash.
 head_cases=head_routes=0;head_total_hash=1469598103934665603ull;
 for(unsigned h:{8u,16u})for(unsigned kind=0;kind<=6;kind++)
  head_check(1,kind?1u:8u,h,128,2,kind,false,false,true);
 std::printf("ALIAS cases=%u routes=%u bits=%llu\n",head_cases,head_routes,(unsigned long long)head_total_hash);return 0;

#else
    bool pass=true;
    for(vx_u32 width:{1u,2u,15u,31u,32u,33u,64u,65u,127u,128u,129u,255u,256u,
            257u,576u,1024u,1025u,4097u})
        for(bool inplace:{false,true}) pass=run_norm(3,width,0,inplace)&&pass;
    for(int kind=1;kind<=4;++kind) for(bool inplace:{false,true})
        pass=run_norm(2,576,kind,inplace)&&pass;
    for(vx_u32 axis:{1u,2u,15u,31u,32u,33u,64u,65u,127u,128u,129u,255u,256u,
            257u,576u,1024u,16383u,16384u,16385u,49152u,49153u,151936u,151937u})
        for(vx_u32 inner:{1u,3u}) pass=run_argmax(2,axis,inner,0)&&pass;
    for(int kind=1;kind<=10;++kind) for(vx_u32 inner:{1u,3u})
        pass=run_argmax(2,577,inner,kind)&&pass;
    for(vx_u32 axis:{49152u,151936u})
        for(int kind=1;kind<=10;++kind) for(vx_u32 inner:{1u,3u})
            pass=run_argmax(2,axis,inner,kind)&&pass;
    return pass?0:1;
#endif
}
