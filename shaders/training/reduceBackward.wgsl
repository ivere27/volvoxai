@group(0) @binding(0) var<storage, read> grad_output : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;

struct Params {
  length : u32,
  row_width : u32,
  scale : f32,
  _pad : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.length) { return; }
  grad_input[index] = grad_input[index] +
    grad_output[index / params.row_width] * params.scale;
}
