// Exact-shape Where/Mask backward. The condition is bound as raw u32 words so
// the same authoritative shader supports F32 and I32 condition tensors.
@group(0) @binding(0) var<storage, read> condition : array<u32>;
@group(0) @binding(1) var<storage, read> grad_out : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_x : array<f32>;
@group(0) @binding(3) var<storage, read_write> grad_y : array<f32>;

struct Params {
  size : u32,
  condition_is_f32 : u32,
  _pad0 : u32,
  _pad1 : u32,
}

@group(0) @binding(4) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.size) { return; }

  var select_x = condition[index] != 0u;
  if (params.condition_is_f32 != 0u) {
    select_x = bitcast<f32>(condition[index]) != 0.0;
  }
  if (select_x) {
    grad_x[index] = grad_x[index] + grad_out[index];
  } else {
    grad_y[index] = grad_y[index] + grad_out[index];
  }
}
