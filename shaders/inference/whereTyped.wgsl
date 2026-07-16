// Browser-only typed Where/Mask. The native-facing where.wgsl keeps its
// historical F32 ABI; this variant reads raw 32-bit condition words so both
// F32 and I32 condition tensors retain their declared semantics.
@group(0) @binding(0) var<storage, read> condition_words : array<u32>;
@group(0) @binding(1) var<storage, read> a : array<f32>;
@group(0) @binding(2) var<storage, read> b : array<f32>;
@group(0) @binding(3) var<storage, read_write> output : array<f32>;

struct Params {
  // elements, condition_dtype (0=F32, 1=I32), pad, pad
  dimensions : vec4<u32>,
}
@group(0) @binding(4) var<uniform> params : Params;

fn condition_is_nonzero(index : u32) -> bool {
  if (params.dimensions.y == 0u) {
    // Match JavaScript's `f32 !== 0`: both signed zero encodings are false;
    // every other F32 encoding, including NaN, is true.
    return (condition_words[index] & 0x7fffffffu) != 0u;
  }
  // Do not reinterpret I32 data as F32: 0x80000000 (Int32.MIN_VALUE) is a
  // true integer condition even though that bit pattern is negative F32 zero.
  return bitcast<i32>(condition_words[index]) != 0;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let index = global_id.x;
  if (index >= params.dimensions.x) { return; }
  output[index] = select(b[index], a[index], condition_is_nonzero(index));
}
