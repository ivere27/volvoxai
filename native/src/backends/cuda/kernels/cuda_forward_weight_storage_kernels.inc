/* Immutable compact storage, with F32 activation/accumulation. Bit conversion
 * preserves F16 subnormals and BF16 payloads without half arithmetic or FTZ. */
static VX_CUDA_DEVICE VX_CUDA_FORCEINLINE float vx_weight_storage_load(
        const void* pointer, vx_u32 dtype, vx_u32 index) {
    if (dtype == VX_DTYPE_F32) return ((const float*)pointer)[index];
    vx_u32 raw = ((const unsigned short*)pointer)[index], bits;
    if (dtype == VX_DTYPE_BF16) bits = raw << 16;
    else {
        vx_u32 sign = (raw & 0x8000u) << 16;
        vx_u32 exponent = (raw >> 10) & 31u, mantissa = raw & 1023u;
        if (!exponent) {
            if (!mantissa) bits = sign;
            else {
                int power = -14;
                while (!(mantissa & 1024u)) { mantissa <<= 1; --power; }
                bits = sign | ((vx_u32)(power + 127) << 23) | ((mantissa & 1023u) << 13);
            }
        } else bits = sign | ((exponent == 31u ? 255u : exponent + 112u) << 23) | (mantissa << 13);
    }
    union { vx_u32 bits; float value; } converted;
    converted.bits = bits;
    return converted.value;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_matmul_storage_f32(
        const float* input, const void* weight, const float* bias, float* output,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 out_in,
        vx_u32 has_bias, vx_u32 dtype) {
    VX_CUDA_SHARED float a[16][16], b[16][16];
    vx_u32 row = blockIdx.y * 16u + threadIdx.y;
    vx_u32 column = blockIdx.x * 16u + threadIdx.x;
    float sum = 0;
    for (vx_u32 base = 0; base < d_in; base += 16u) {
        vx_u32 ak = base + threadIdx.x, bk = base + threadIdx.y;
        a[threadIdx.y][threadIdx.x] = row < rows && ak < d_in ? input[row*d_in+ak] : 0;
        b[threadIdx.y][threadIdx.x] = column < d_out && bk < d_in
            ? vx_weight_storage_load(weight,dtype,out_in ? column*d_in+bk : bk*d_out+column) : 0;
        vx_sync_threads();
        vx_u32 remaining = d_in-base < 16u ? d_in-base : 16u;
        for (vx_u32 k=0;k<remaining;++k) sum += a[threadIdx.y][k]*b[k][threadIdx.x];
        vx_sync_threads();
    }
    if (row < rows && column < d_out) output[row*d_out+column] = sum + (has_bias ? bias[column] : 0);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_embedding_storage_f32(
        const vx_i32* tokens, const void* weight, float* output,
        vx_u32 count, vx_u32 width, vx_u32 dtype) {
    vx_u32 index=vx_global_x();
    if(index<count*width) output[index]=vx_weight_storage_load(weight,dtype,
        (vx_u32)tokens[index/width]*width+index%width);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_gather_storage_f32(
        const void* input, const vx_i32* indices, float* output,
        vx_u32 outer, vx_u32 axis, vx_u32 inner, vx_u32 count, vx_u32 dtype) {
    vx_u32 index=vx_global_x();
    if(index>=outer*count*inner)return;
    vx_u32 selected=(index/inner)%count;
    vx_i32 coordinate=indices[selected];
    if(coordinate<0)coordinate+=(vx_i32)axis;
    output[index]=vx_weight_storage_load(input,dtype,
        ((index/(count*inner))*axis+(vx_u32)coordinate)*inner+index%inner);
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_linear_storage_f32(
        const float* input, const void* weight, const float* bias,
        const float* indices, const float* routes, float* output,
        vx_u32 rows, vx_u32 d_in, vx_u32 d_out, vx_u32 experts, vx_u32 top_k,
        vx_u32 has_bias, const vx_u32* slot_rows, vx_u32 slot_domain, vx_u32 dtype) {
    vx_u32 index=vx_global_x();
    if(index>=rows*d_out)return;
    vx_u32 row=index/d_out,column=index%d_out;
    float sum=0;
    for(vx_u32 k=0;k<top_k;++k) {
        vx_u32 route=row*top_k+k,expert=(vx_u32)indices[route];
        if(slot_domain){if(expert>=slot_domain)continue;expert=slot_rows[expert];}
        if(expert>=experts)continue;
        float value=has_bias ? bias[expert*d_out+column] : 0;
        for(vx_u32 d=0;d<d_in;++d)value+=input[row*d_in+d]*
            vx_weight_storage_load(weight,dtype,(expert*d_in+d)*d_out+column);
        sum+=routes[route]*value;
    }
    output[index]=sum;
}

extern "C" VX_CUDA_GLOBAL void vx_cuda_moe_gated_mlp_storage_f32(
        const float* input, const void* gate_weight, const void* up_weight,
        const void* down_weight, const float* indices, const float* routes,
        float* output, vx_u32 rows, vx_u32 d_in, vx_u32 hidden, vx_u32 d_out,
        vx_u32 experts, vx_u32 top_k, vx_u32 input_scale, float limit,
        vx_u32 gate_dtype, vx_u32 up_dtype, vx_u32 down_dtype) {
    VX_CUDA_SHARED float intermediate[8192];
    vx_u32 row=blockIdx.x;
    if(row>=rows)return;
    for(vx_u32 o=threadIdx.x;o<d_out;o+=blockDim.x)output[row*d_out+o]=0;
    for(vx_u32 k=0;k<top_k;++k) {
        vx_u32 expert=(vx_u32)indices[row*top_k+k];
        float route=routes[row*top_k+k];
        if(expert>=experts)return;
        for(vx_u32 h=threadIdx.x;h<hidden;h+=blockDim.x) {
            float gate=0,up=0;
            for(vx_u32 d=0;d<d_in;++d) {
                float x=input[row*d_in+d]*(input_scale?route:1.0f);
                vx_u32 offset=(expert*d_in+d)*hidden+h;
                gate+=x*vx_weight_storage_load(gate_weight,gate_dtype,offset);
                up+=x*vx_weight_storage_load(up_weight,up_dtype,offset);
            }
            if(limit>0){gate=vx_min(gate,limit);up=vx_max(-limit,vx_min(up,limit));}
            intermediate[h]=gate*vx_sigmoid(gate)*up;
        }
        vx_sync_threads();
        for(vx_u32 o=threadIdx.x;o<d_out;o+=blockDim.x) {
            float value=0;
            for(vx_u32 h=0;h<hidden;++h)value+=intermediate[h]*
                vx_weight_storage_load(down_weight,down_dtype,(expert*hidden+h)*d_out+o);
            output[row*d_out+o]+=(input_scale?1.0f:route)*value;
        }
        vx_sync_threads();
    }
}
