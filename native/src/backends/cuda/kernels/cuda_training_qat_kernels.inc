/* Weight-only QAT uses one thread per fitted group. The serial K/error loops
 * preserve the CPU fitter's F32 reconstruction and ascending F64 error sum.
 * Groups run independently; no master or optimizer buffer is written here.
 * Included after cuda_quantize_w8_kernels.inc for strict IEEE helpers. */
static VX_CUDA_DEVICE float vx_qat_f32(unsigned bits) {
    union { unsigned bits; float value; } v; v.bits=bits; return v.value;
}
static VX_CUDA_DEVICE int vx_qat_code(float value, float scale, int low, int high) {
    float ratio=vx_cuda_w8_div_rn(value,scale);
    if (ratio <= (float)low) return low;
    if (ratio >= (float)high) return high;
    int result;
    asm("cvt.rni.s32.f32 %0, %1;" : "=r"(result) : "f"(ratio));
    return result;
}
static VX_CUDA_DEVICE double vx_qat_error_add(double total, float value, float restored) {
    double delta, square, result;
    asm("sub.rn.f64 %0, %1, %2;" : "=d"(delta) : "d"((double)value), "d"((double)restored));
    asm("mul.rn.f64 %0, %1, %1;" : "=d"(square) : "d"(delta));
    asm("add.rn.f64 %0, %1, %2;" : "=d"(result) : "d"(total), "d"(square));
    return result;
}
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_qat_fake_f32(
        const float* master, float* fake, unsigned char* pass, unsigned* status,
        unsigned rows, unsigned columns, unsigned group_size,
        unsigned bits, unsigned transpose, unsigned convolution, unsigned algorithm) {
    unsigned groups=convolution ? 1u : columns/group_size+(columns%group_size!=0u);
    unsigned group=vx_training_global_x();
    if (group >= rows*groups) return;
    unsigned row=group/groups, begin=convolution ? 0u : (group%groups)*group_size;
    unsigned length=columns-begin;
    if (!convolution && length>group_size) length=group_size;
    int zero=1<<(bits-1u), low=-zero, high=zero-1;
    float maximum=0.0f;
    for(unsigned j=0;j<length;j++) {
        unsigned k=begin+j, index=transpose ? k*rows+row : row*columns+k;
        float v=master[index];
        if(!vx_cuda_w8_isfinite(v)) {vx_cuda_w8_atomic_or(status,2u);return;}
        float magnitude=v<0.0f ? -v : v;
        if(magnitude>maximum)maximum=magnitude;
    }
    float base;
    if(convolution) {
        double divided;
        asm("div.rn.f64 %0, %1, %2;" : "=d"(divided) : "d"((double)maximum), "d"((double)high));
        base=maximum==0.0f ? 1.0f : (float)divided;
        if(base==0.0f)base=vx_qat_f32(0x00800000u);
        float limit=vx_cuda_w8_div_rn(vx_qat_f32(0x7f7fffffu),(float)zero);
        if(base>limit)base=limit;
    } else {
        base=maximum==0.0f ? 1.0f : vx_cuda_w8_div_rn(maximum,(float)high);
        if(base==0.0f)base=vx_qat_f32(1u);
        if(!vx_cuda_w8_isfinite(vx_cuda_w8_mul_rn((float)high,base)))
            base=vx_qat_f32(vx_cuda_w8_float_bits(base)-1u);
    }
    float scale=base;
    if(convolution || (algorithm!=0u && maximum>0.0f)) {
        double best_error;
        asm("mov.b64 %0, 0x7ff0000000000000;" : "=d"(best_error));
        unsigned steps=algorithm==0u ? 0u : bits==2u ? 99u :
            (convolution || bits==3u) ? 50u : 25u;
        for(unsigned step=0;step<=steps;step++) {
            float factor;
            if(convolution || bits<=3u)factor=vx_cuda_w8_div_rn((float)(100u-step),100.0f);
            else {
                float reduction=vx_cuda_w8_mul_rn((float)step,0.01f);
                asm("sub.rn.f32 %0, %1, %2;" : "=f"(factor) : "f"(1.0f), "f"(reduction));
            }
            float candidate=vx_cuda_w8_mul_rn(base,factor);
            if(candidate==0.0f)candidate=vx_qat_f32(convolution ? 0x00800000u : 1u);
            double error=0.0;
            for(unsigned j=0;j<length;j++) {
                unsigned k=begin+j,index=transpose ? k*rows+row : row*columns+k;
                float v=master[index],restored=vx_cuda_w8_mul_rn((float)vx_qat_code(v,candidate,low,high),candidate);
                if(!vx_cuda_w8_isfinite(restored)) {asm("mov.b64 %0, 0x7ff0000000000000;" : "=d"(error));break;}
                error=vx_qat_error_add(error,v,restored);
            }
            if(error<best_error) {best_error=error;scale=candidate;}
        }
    }
    for(unsigned j=0;j<length;j++) {
        unsigned k=begin+j,index=transpose ? k*rows+row : row*columns+k;
        float ratio=vx_cuda_w8_div_rn(master[index],scale);
        float restored=vx_cuda_w8_mul_rn((float)vx_qat_code(master[index],scale,low,high),scale);
        if(!vx_cuda_w8_isfinite(restored))vx_cuda_w8_atomic_or(status,4u);
        fake[index]=restored;
        pass[index]=(unsigned char)(ratio>=(float)low && ratio<=(float)high);
    }
}
extern "C" VX_CUDA_GLOBAL void vx_cuda_training_qat_ste_f32(
        float* gradient, const unsigned char* pass, unsigned elements) {
    unsigned index=vx_training_global_x();
    if(index<elements && !pass[index] && vx_cuda_w8_isfinite(gradient[index]))gradient[index]=0.0f;
}
