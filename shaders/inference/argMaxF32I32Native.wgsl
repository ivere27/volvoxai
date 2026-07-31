// Native F32 ArgMax with an ordinary I32 output. Strict comparison preserves
// the first index when maxima are tied.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<i32>;

struct Params {
  outer : u32,
  axis_size : u32,
  inner : u32,
  _pad0 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_index = gid.x;
  let output_elements = params.outer * params.inner;
  if (output_index >= output_elements) { return; }

  let outer_index = output_index / params.inner;
  let inner_index = output_index % params.inner;
  let base =
    outer_index * params.axis_size * params.inner + inner_index;
  var best = input[base];
  var best_index = 0u;
  for (var axis_index = 1u;
       axis_index < params.axis_size;
       axis_index = axis_index + 1u) {
    let value = input[base + axis_index * params.inner];
    if (value > best) {
      best = value;
      best_index = axis_index;
    }
  }
  output[output_index] = i32(best_index);
}
