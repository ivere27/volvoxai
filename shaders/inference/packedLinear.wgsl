// Packed signed INT4/INT8 weights with A16 or dynamic per-row A8 operands.
// ABI: native/src/backends/llm_shader_abi.md. No optional shader-f16 feature.
@group(0) @binding(0) var<storage, read> input:array<f32>;
@group(0) @binding(1) var<storage, read> weight:array<u32>;
@group(0) @binding(2) var<storage, read> scales:array<f32>;
@group(0) @binding(3) var<storage, read> bias:array<f32>;
@group(0) @binding(4) var<storage, read_write> output:array<f32>;
struct Params {
    batch:u32, sequence:u32, keys:u32, heads:u32, kv_heads:u32, key_dim:u32, value_dim:u32, streams:u32,
    groups:u32, rate:u32, window:u32, top_k:u32, iterations:u32, index_heads:u32, index_dim:u32, rotary_dim:u32,
    rotary_offset:u32, normalize:u32, causal:u32, interleaved:u32, mask_mode:u32, flags:u32, work:u32, first:u32,
    eps:f32, hc_eps:f32, scale:f32, pad1:f32, pad2:u32, pad3:u32, pad4:u32, pad5:u32,
}
@group(0) @binding(5) var<uniform> p:Params;

// Round to binary16, ties to even, then widen exactly. Integer rounding also
// preserves half subnormals on devices whose native half arithmetic flushes them.
fn half_round(value:f32)->f32 {
    let bits=bitcast<u32>(value);let sign=bits&0x80000000u;let magnitude=bits&0x7fffffffu;
    if(magnitude>=0x7f800000u) {return value;}
    if(magnitude>=0x477ff000u) {return bitcast<f32>(sign|0x7f800000u);}
    if(magnitude<=0x33000000u) {return bitcast<f32>(sign);}
    if(magnitude<0x38800000u) {
        let mantissa=(magnitude&0x007fffffu)|0x00800000u;
        let shift=126u-(magnitude>>23u);let base=mantissa>>shift;
        let remainder=mantissa&((1u<<shift)-1u);let halfway=1u<<(shift-1u);
        let rounded=base+select(0u,1u,remainder>halfway || (remainder==halfway && (base&1u)!=0u));
        return bitcast<f32>(sign|bitcast<u32>(f32(rounded)*0.000000059604644775390625));
    }
    return bitcast<f32>(sign|((magnitude+0xfffu+((magnitude>>13u)&1u))&0xffffe000u));
}
// WGSL division may differ by several ULPs at a quantization midpoint. Divide
// the 24-bit significands with integer arithmetic so A8 ties match IEEE F32
// division followed by ties-to-even integer rounding on every backend.
// A subnormal quotient is returned as zero: its consumers either clamp the
// activation scale to FLT_MIN, or round a quotient below 0.5 to integer zero.
fn positive_divide(value:f32,divisor:f32)->f32 {
    let bits=bitcast<u32>(value)&0x7fffffffu;
    if(bits==0u) {return 0.0;}
    let denominator=bitcast<u32>(divisor);
    var exponent=i32(bits>>23u)-127;
    var numerator=(bits&0x007fffffu)|0x00800000u;
    if((bits>>23u)==0u) {
        numerator=bits;exponent=-126;
        loop {if(numerator>=0x00800000u) {break;} numerator<<=1u;exponent-=1;}
    }
    exponent-=i32(denominator>>23u)-127;
    let mantissa=(denominator&0x007fffffu)|0x00800000u;
    if(numerator<mantissa) {numerator<<=1u;exponent-=1;}
    var remainder=numerator-mantissa;var quotient=1u;
    for(var digit=0u;digit<23u;digit++) {
        remainder<<=1u;quotient<<=1u;
        if(remainder>=mantissa) {remainder-=mantissa;quotient|=1u;}
    }
    if(2u*remainder>mantissa || (2u*remainder==mantissa && (quotient&1u)!=0u)) {quotient+=1u;}
    if(quotient==0x01000000u) {quotient>>=1u;exponent+=1;}
    if(exponent< -126) {return 0.0;}
    if(exponent>127) {return bitcast<f32>(0x7f800000u);}
    return bitcast<f32>((u32(exponent+127)<<23u)|(quotient&0x007fffffu));
}
fn quantize(value:f32,scale:f32)->i32 {
    let magnitude=min(positive_divide(value,scale),127.0);
    let base=u32(floor(magnitude));let fraction=magnitude-f32(base);
    let rounded=i32(base+select(0u,1u,fraction>0.5 || (fraction==0.5 && (base&1u)!=0u)));
    return select(rounded,-rounded,(bitcast<u32>(value)&0x80000000u)!=0u);
}
fn unpack(column:u32,k:u32)->i32 {
    let shift=select(3u,2u,p.groups==8u);let per_word=1u<<shift;let bits=32u>>shift;
    let word=weight[column*((p.key_dim+per_word-1u)>>shift)+(k>>shift)];
    return i32((word>>((k&(per_word-1u))*bits))&((1u<<bits)-1u))-i32(1u<<(bits-1u));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>) {
    if(gid.x>=p.work) {return;}
    let index=p.first+gid.x;let row=index/p.value_dim;let column=index%p.value_dim;
    let start=row*p.key_dim;let groups=(p.key_dim+127u)/128u;
    var maximum_bits=0u;
    for(var k=0u;k<p.key_dim;k++) {
        let value=input[start+k];
        if((bitcast<u32>(value)&0x7f800000u)==0x7f800000u) {output[index]=bitcast<f32>(0x7fc00000u);return;}
        maximum_bits=max(maximum_bits,bitcast<u32>(value)&0x7fffffffu);
    }
    var result=0.0;
    if(p.normalize==8u) {
        var activation_scale=1.0;
        if(maximum_bits!=0u) {activation_scale=max(positive_divide(bitcast<f32>(maximum_bits),127.0),bitcast<f32>(0x00800000u));}
        for(var group=0u;group<groups;group++) {
            var dot=0i;let end=min((group+1u)*128u,p.key_dim);
            for(var k=group*128u;k<end;k++) {dot+=quantize(input[start+k],activation_scale)*unpack(column,k);}
            result+=f32(dot)*(activation_scale*scales[column*groups+group]);
        }
    } else {
        for(var k=0u;k<p.key_dim;k++) {
            let decoded=half_round(f32(unpack(column,k))*scales[column*groups+k/128u]);
            result+=half_round(input[start+k])*decoded;
        }
    }
    if((p.flags&1u)!=0u) {result+=bias[column];}
    output[index]=result;
}
