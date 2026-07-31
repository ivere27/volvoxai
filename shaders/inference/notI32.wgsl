// @volvoxai-browser-only
@group(0) @binding(0) var<storage, read> input : array<i32>;
@group(0) @binding(1) var<storage, read_write> output : array<i32>;

struct Params {
  elements : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.elements) { return; }
  output[index] = select(0, 1, input[index] == 0);
}
