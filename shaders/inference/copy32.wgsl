// @volvoxai-browser-only
// Bit-preserving copy for ordinary 32-bit F32/I32 shape-only tensors.
@group(0) @binding(0) var<storage, read> input : array<u32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;

struct Params {
  elements : u32,
  dispatch_stride : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x + gid.y * params.dispatch_stride;
  if (index >= params.elements) { return; }
  output[index] = input[index];
}
