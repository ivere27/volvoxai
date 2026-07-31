// @volvoxai-browser-only
// Typed F32/I32 Clip. Raw words preserve the selected output dtype exactly.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  dimensions : vec4<u32>, // elements, canonical dtype (18=F32, 16=I32), pad, pad
  f32_bounds : vec4<f32>, // minimum, maximum, pad, pad
  i32_bounds : vec4<i32>, // minimum, maximum, pad, pad
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.dimensions.x) { return; }
  if (params.dimensions.y == 18u) {
    let value = clamp(bitcast<f32>(input_words[index]),
      params.f32_bounds.x, params.f32_bounds.y);
    output_words[index] = bitcast<u32>(value);
  } else {
    let value = clamp(bitcast<i32>(input_words[index]),
      params.i32_bounds.x, params.i32_bounds.y);
    output_words[index] = bitcast<u32>(value);
  }
}
