// @volvoxai-native-spv-only
requires packed_4x8_integer_dot_product;

// Cooperative DP4a W8A8 dense kernel: eight rows by 32 output channels share
// a 16-byte K tile. Each invocation owns one packed four-channel output word.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> weight_words : array<u32>;
@group(0) @binding(2) var<storage, read> weight_scales : array<f32>;
@group(0) @binding(3) var<storage, read> weight_zero_points : array<i32>;
@group(0) @binding(4) var<storage, read> bias_values : array<i32>;
@group(0) @binding(5) var<storage, read_write> output_words : array<u32>;
struct Params {
  rows:u32, d_in:u32, d_out:u32, input_type:u32,
  weight_type:u32, output_type:u32, _pad0:u32, _pad1:u32,
  input_zero_point:i32, output_zero_point:i32, _pad2:i32, _pad3:i32,
  input_scale:f32, output_scale:f32, _pad4:f32, _pad5:f32,
}
@group(0) @binding(6) var<uniform> params : Params;
var<workgroup> input_tile : array<u32, 32>;
var<workgroup> weight_tile : array<u32, 128>;

fn raw_input_byte(index:u32)->u32 {
  return (input_words[index/4u] >> ((index%4u)*8u)) & 255u;
}
fn raw_weight_byte(index:u32)->u32 {
  return (weight_words[index/4u] >> ((index%4u)*8u)) & 255u;
}
fn repeated_byte(value:i32)->u32 {
  let b=bitcast<u32>(value)&255u;
  return b|(b<<8u)|(b<<16u)|(b<<24u);
}
fn input_word_padded(base:u32,count:u32)->u32 {
  if(count>=4u){
    let wi=base/4u; let offset=base%4u;
    if(offset==0u){return input_words[wi];}
    let shift=offset*8u;
    return (input_words[wi]>>shift)|(input_words[wi+1u]<<(32u-shift));
  }
  var result=repeated_byte(params.input_zero_point);
  for(var lane=0u;lane<count;lane=lane+1u){
    let shift=lane*8u;
    result=(result&~(255u<<shift))|(raw_input_byte(base+lane)<<shift);
  }
  return result;
}
fn weight_word_padded(base:u32,count:u32,zero_point:i32)->u32 {
  if(count>=4u){
    let wi=base/4u; let offset=base%4u;
    if(offset==0u){return weight_words[wi];}
    let shift=offset*8u;
    return (weight_words[wi]>>shift)|(weight_words[wi+1u]<<(32u-shift));
  }
  var result=repeated_byte(zero_point);
  for(var lane=0u;lane<count;lane=lane+1u){
    let shift=lane*8u;
    result=(result&~(255u<<shift))|(raw_weight_byte(base+lane)<<shift);
  }
  return result;
}
fn signed_word(word:u32,dtype:u32)->u32 {
  if(dtype==3u){return word^0x80808080u;} return word;
}
fn centered_zero_point(zero_point:i32,dtype:u32)->i32 {
  if(dtype==3u){return zero_point-128;} return zero_point;
}
fn corrected_dot(input_raw:u32,weight_raw:u32,weight_zp:i32)->i32 {
  let x=signed_word(input_raw,params.input_type);
  let w=signed_word(weight_raw,params.weight_type);
  let xz=centered_zero_point(params.input_zero_point,params.input_type);
  let wz=centered_zero_point(weight_zp,params.weight_type);
  var value=dot4I8Packed(x,w);
  if(wz!=0){value=value-wz*dot4I8Packed(x,0x01010101u);}
  if(xz!=0){value=value-xz*dot4I8Packed(w,0x01010101u);}
  return value+4*xz*wz;
}
fn round_even(value:f32)->i32 {
  let lower=floor(value); let fraction=value-lower;
  if(fraction<0.5){return i32(lower);}
  if(fraction>0.5){return i32(lower+1.0);}
  let low=i32(lower); if((low&1)==0){return low;} return low+1;
}
fn output_byte(value:i32)->u32{return bitcast<u32>(value)&255u;}
fn requantize(accumulator:i32,channel:u32)->u32 {
  let multiplier=(params.input_scale*weight_scales[channel])/params.output_scale;
  let transformed=f32(accumulator)*multiplier+f32(params.output_zero_point);
  let minimum=select(0,-128,params.output_type==2u);
  let maximum=select(255,127,params.output_type==2u);
  if(transformed<=f32(minimum)){return output_byte(minimum);}
  if(transformed>=f32(maximum)){return output_byte(maximum);}
  if(transformed!=transformed){return output_byte(params.output_zero_point);}
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(8,8,1)
fn main(@builtin(workgroup_id) group_id:vec3<u32>,
        @builtin(local_invocation_id) local_id:vec3<u32>){
  let row=group_id.y*8u+local_id.y;
  let word_in_row=group_id.x*8u+local_id.x;
  let words_per_row=params.d_out/4u;
  let valid=row<params.rows&&word_in_row<words_per_row;
  let channel_base=group_id.x*32u;
  let output_channel=channel_base+local_id.x*4u;
  var a0=0; var a1=0; var a2=0; var a3=0;
  var z0=0; var z1=0; var z2=0; var z3=0;
  if(valid){
    a0=bias_values[output_channel]; a1=bias_values[output_channel+1u];
    a2=bias_values[output_channel+2u]; a3=bias_values[output_channel+3u];
    z0=weight_zero_points[output_channel]; z1=weight_zero_points[output_channel+1u];
    z2=weight_zero_points[output_channel+2u]; z3=weight_zero_points[output_channel+3u];
  }
  let local_linear=local_id.y*8u+local_id.x;
  for(var k_base=0u;k_base<params.d_in;k_base=k_base+16u){
    if(local_id.x<4u){
      let k=k_base+local_id.x*4u; var count=0u;
      if(row<params.rows&&k<params.d_in){count=min(4u,params.d_in-k);}
      input_tile[local_id.y*4u+local_id.x]=input_word_padded(row*params.d_in+k,count);
    }
    for(var slot=local_linear;slot<128u;slot=slot+64u){
      let tile_channel=slot/4u; let k_word=slot%4u;
      let channel=channel_base+tile_channel; let k=k_base+k_word*4u;
      var count=0u; var zero=0;
      if(channel<params.d_out){
        zero=weight_zero_points[channel];
        if(k<params.d_in){count=min(4u,params.d_in-k);}
      }
      weight_tile[k_word*32u+tile_channel]=weight_word_padded(
        channel*params.d_in+k,count,zero);
    }
    workgroupBarrier();
    for(var k_word=0u;k_word<4u;k_word=k_word+1u){
      let x=input_tile[local_id.y*4u+k_word];
      let w=k_word*32u+local_id.x*4u;
      a0=a0+corrected_dot(x,weight_tile[w],z0);
      a1=a1+corrected_dot(x,weight_tile[w+1u],z1);
      a2=a2+corrected_dot(x,weight_tile[w+2u],z2);
      a3=a3+corrected_dot(x,weight_tile[w+3u],z3);
    }
    workgroupBarrier();
  }
  if(valid){
    var packed=requantize(a0,output_channel);
    packed=packed|(requantize(a1,output_channel+1u)<<8u);
    packed=packed|(requantize(a2,output_channel+2u)<<16u);
    packed=packed|(requantize(a3,output_channel+3u)<<24u);
    output_words[row*words_per_row+word_in_row]=packed;
  }
}
