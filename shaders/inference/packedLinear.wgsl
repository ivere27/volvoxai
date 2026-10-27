// Packed signed INT4/INT8 weights with F32, F16 or dynamic per-row I8 operands.
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

// Nonfinite values must be runtime expressions: WGSL constant evaluation
// rejects NaN/Inf even when their exact IEEE bit pattern is intentional.
fn nonfinite(bits:u32)->f32 {return bitcast<f32>(bits);}

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
    if(exponent>127) {return nonfinite(0x7f800000u);}
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

// A 64-column tile shares only one bounded storage group, never an expanded
// weight matrix or an input-width-sized activation cache. Aligned output ranges
// cannot straddle rows when N is a multiple of 64; other ranges keep the direct
// path below. The dispatch and its exact partial-write range stay unchanged.
var<workgroup> row_maximum:array<u32,64>;
var<workgroup> row_scale:f32;
var<workgroup> activation_group:array<i32,256>;

fn shared_i8(tile:u32,lane:u32) {
    let base=p.first+tile*64u;let row=base/p.value_dim;
    let index=base+lane;let column=index%p.value_dim;
    let selected=tile*64u+lane<p.work;let start=row*p.key_dim;
    let groups=p.key_dim/p.rate+select(0u,1u,p.key_dim%p.rate!=0u);
    var maximum_bits=0u;
    for(var k=lane;k<p.key_dim;k+=64u) {
        maximum_bits=max(maximum_bits,bitcast<u32>(input[start+k])&0x7fffffffu);
    }
    row_maximum[lane]=maximum_bits;
    workgroupBarrier();
    for(var distance=32u;distance>0u;distance>>=1u) {
        if(lane<distance) {row_maximum[lane]=max(row_maximum[lane],row_maximum[lane+distance]);}
        workgroupBarrier();
    }
    let finite=row_maximum[0]<0x7f800000u;
    if(lane==0u) {
        row_scale=1.0;
        if(finite && row_maximum[0]!=0u) {
            row_scale=max(positive_divide(bitcast<f32>(row_maximum[0]),127.0),bitcast<f32>(0x00800000u));
        }
    }
    workgroupBarrier();
    let activation_scale=row_scale;
    var result=0.0;
    for(var group=0u;group<groups;group++) {
        let first=group*p.rate;let count=min(p.rate,p.key_dim-first);
        for(var offset=lane;offset<count;offset+=64u) {
            var value=0i;
            if(finite) {value=quantize(input[start+first+offset],activation_scale);}
            activation_group[offset]=value;
        }
        workgroupBarrier();
        if(selected) {
            var dot=0i;
            for(var offset=0u;offset<count;offset++) {dot+=activation_group[offset]*unpack(column,first+offset);}
            result+=f32(dot)*(activation_scale*scales[column*groups+group]);
        }
        // Every lane, including a masked final column or nonfinite row, reaches
        // this barrier before the next group overwrites the shared operands.
        workgroupBarrier();
    }
    if(selected) {
        if((p.flags&1u)!=0u) {result+=bias[column];}
        output[index]=select(nonfinite(0x7fc00000u),result,finite);
    }
}

// Multi-row 4x64 tile for sufficiently large aligned projections.
// Up to 4 rows' quantized I8 activations are byte-packed into each element of
// activation_group so each lane unpacks its column's weight word once and
// accumulates 4 row dots in registers without increasing workgroup memory.
fn shared_i8_mr(tile:u32,lane:u32) {
    let cols_tiles=p.value_dim/64u;
    let total_rows=p.work/p.value_dim;
    let row_tiles=(total_rows+3u)/4u;
    if(tile>=row_tiles*cols_tiles) {return;}
    let row_tile=tile/cols_tiles;
    let col_tile=tile%cols_tiles;
    let first_row=p.first/p.value_dim+row_tile*4u;
    let end_row=p.first/p.value_dim+total_rows;
    let column=col_tile*64u+lane;
    let groups=p.key_dim/p.rate+select(0u,1u,p.key_dim%p.rate!=0u);
    let sub_row=lane>>4u;
    let sub_lane=lane&15u;
    let my_row=first_row+sub_row;
    var maximum_bits=0u;
    if(my_row<end_row) {
        let start=my_row*p.key_dim;
        for(var k=sub_lane;k<p.key_dim;k+=16u) {
            maximum_bits=max(maximum_bits,bitcast<u32>(input[start+k])&0x7fffffffu);
        }
    }
    row_maximum[lane]=maximum_bits;
    workgroupBarrier();
    for(var distance=8u;distance>0u;distance>>=1u) {
        if(sub_lane<distance) {
            row_maximum[lane]=max(row_maximum[lane],row_maximum[lane+distance]);
        }
        workgroupBarrier();
    }
    if(sub_lane==0u) {
        let max_val=row_maximum[lane];
        let finite=max_val<0x7f800000u;
        var sc=1.0;
        if(finite && max_val!=0u) {
            sc=max(positive_divide(bitcast<f32>(max_val),127.0),bitcast<f32>(0x00800000u));
        }
        row_maximum[lane+1u]=bitcast<u32>(sc);
    }
    workgroupBarrier();
    let finite0=row_maximum[0u]<0x7f800000u;
    let finite1=row_maximum[16u]<0x7f800000u;
    let finite2=row_maximum[32u]<0x7f800000u;
    let finite3=row_maximum[48u]<0x7f800000u;
    let scale0=bitcast<f32>(row_maximum[1u]);
    let scale1=bitcast<f32>(row_maximum[17u]);
    let scale2=bitcast<f32>(row_maximum[33u]);
    let scale3=bitcast<f32>(row_maximum[49u]);
    var result0=0.0;var result1=0.0;var result2=0.0;var result3=0.0;
    for(var group=0u;group<groups;group++) {
        let first=group*p.rate;let count=min(p.rate,p.key_dim-first);
        for(var offset=lane;offset<count;offset+=64u) {
            let k=first+offset;
            var q0=128u;var q1=128u;var q2=128u;var q3=128u;
            if(first_row<end_row && finite0) {
                q0=u32(quantize(input[first_row*p.key_dim+k],scale0)+128);
            }
            if(first_row+1u<end_row && finite1) {
                q1=u32(quantize(input[(first_row+1u)*p.key_dim+k],scale1)+128);
            }
            if(first_row+2u<end_row && finite2) {
                q2=u32(quantize(input[(first_row+2u)*p.key_dim+k],scale2)+128);
            }
            if(first_row+3u<end_row && finite3) {
                q3=u32(quantize(input[(first_row+3u)*p.key_dim+k],scale3)+128);
            }
            activation_group[offset]=bitcast<i32>((q0&255u)|((q1&255u)<<8u)|((q2&255u)<<16u)|((q3&255u)<<24u));
        }
        workgroupBarrier();
        var dot0=0i;var dot1=0i;var dot2=0i;var dot3=0i;
        for(var offset=0u;offset<count;offset++) {
            let packed=bitcast<u32>(activation_group[offset]);
            let w=unpack(column,first+offset);
            dot0+=(i32(packed&255u)-128)*w;
            dot1+=(i32((packed>>8u)&255u)-128)*w;
            dot2+=(i32((packed>>16u)&255u)-128)*w;
            dot3+=(i32((packed>>24u)&255u)-128)*w;
        }
        let w_scale=scales[column*groups+group];
        result0+=f32(dot0)*(scale0*w_scale);
        result1+=f32(dot1)*(scale1*w_scale);
        result2+=f32(dot2)*(scale2*w_scale);
        result3+=f32(dot3)*(scale3*w_scale);
        workgroupBarrier();
    }
    var b=0.0;
    if((p.flags&1u)!=0u) {b=bias[column];}
    let nan=nonfinite(0x7fc00000u);
    if(first_row<end_row) {
        output[first_row*p.value_dim+column]=select(nan,result0+b,finite0);
    }
    if(first_row+1u<end_row) {
        output[(first_row+1u)*p.value_dim+column]=select(nan,result1+b,finite1);
    }
    if(first_row+2u<end_row) {
        output[(first_row+2u)*p.value_dim+column]=select(nan,result2+b,finite2);
    }
    if(first_row+3u<end_row) {
        output[(first_row+3u)*p.value_dim+column]=select(nan,result3+b,finite3);
    }
}

fn direct(index:u32) {
    let row=index/p.value_dim;let column=index%p.value_dim;
    let start=row*p.key_dim;let groups=p.key_dim/p.rate+select(0u,1u,p.key_dim%p.rate!=0u);
    var maximum_bits=0u;
    for(var k=0u;k<p.key_dim;k++) {
        let value=input[start+k];
        if((bitcast<u32>(value)&0x7f800000u)==0x7f800000u) {output[index]=nonfinite(0x7fc00000u);return;}
        maximum_bits=max(maximum_bits,bitcast<u32>(value)&0x7fffffffu);
    }
    var result=0.0;
    if(p.normalize==8u) {
        var activation_scale=1.0;
        if(maximum_bits!=0u) {activation_scale=max(positive_divide(bitcast<f32>(maximum_bits),127.0),bitcast<f32>(0x00800000u));}
        for(var group=0u;group<groups;group++) {
            // Integer dots span the complete storage group before scaling.
            var dot=0i;let end=min((group+1u)*p.rate,p.key_dim);
            for(var k=group*p.rate;k<end;k++) {dot+=quantize(input[start+k],activation_scale)*unpack(column,k);}
            result+=f32(dot)*(activation_scale*scales[column*groups+group]);
        }
    } else if(p.normalize==32u) {
        for(var k=0u;k<p.key_dim;k++) {
            let decoded=f32(unpack(column,k))*scales[column*groups+k/p.rate];
            result+=input[start+k]*decoded;
        }
    } else {
        for(var k=0u;k<p.key_dim;k++) {
            let decoded=half_round(f32(unpack(column,k))*scales[column*groups+k/p.rate]);
            result+=half_round(input[start+k])*decoded;
        }
    }
    if((p.flags&1u)!=0u) {result+=bias[column];}
    output[index]=result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid:vec3<u32>,
        @builtin(workgroup_id) tile:vec3<u32>,@builtin(local_invocation_id) lane:vec3<u32>) {
    if(p.normalize==8u && p.value_dim%64u==0u && p.first%64u==0u) {
        // Four-row tiles trade parallelism for weight reuse. Keep small decode
        // batches on shared_i8. At least 64 rows and 384 active tiles retain
        // the measured RTX 3090 prefill gain. Check cheap bounds before the
        // whole-row divisions; work/64 also avoids overflow in rows*columns.
        if(p.work>=98304u && p.value_dim<=p.work/64u &&
                p.first%p.value_dim==0u && p.work%p.value_dim==0u) {
            shared_i8_mr(tile.x,lane.x);
        } else {
            shared_i8(tile.x,lane.x);
        }
    } else if(gid.x<p.work) {direct(p.first+gid.x);}
}
