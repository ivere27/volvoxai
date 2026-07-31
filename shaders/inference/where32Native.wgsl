// Native exact-shape Where. The I32 condition follows ONNX truth semantics;
// selected F32/I32 values are copied as raw 32-bit words.
@group(0) @binding(0) var<storage, read> condition : array<i32>;
@group(0) @binding(1) var<storage, read> a_words : array<u32>;
@group(0) @binding(2) var<storage, read> b_words : array<u32>;
@group(0) @binding(3) var<storage, read_write> output_words : array<u32>;

struct Params {
  elements : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.elements) { return; }
  output_words[index] =
    select(b_words[index], a_words[index], condition[index] != 0);
}
