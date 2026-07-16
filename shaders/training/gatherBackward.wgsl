// Canonical I32 Gather backward.  Each invocation owns one data-gradient
// element and reduces all matching output-gradient slices, so repeated indices
// scatter-add deterministically without requiring floating-point atomics.
@group(0) @binding(0) var<storage, read> indices : array<i32>;
@group(0) @binding(1) var<storage, read> grad_output : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input : array<f32>;

struct Params {
  outer : u32,
  axis_size : u32,
  inner : u32,
  indices_elements : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let input_index = gid.x;
  let input_elements = params.outer * params.axis_size * params.inner;
  if (input_index >= input_elements) { return; }

  let outer_index = input_index / (params.axis_size * params.inner);
  let within_outer = input_index % (params.axis_size * params.inner);
  let axis_index = within_outer / params.inner;
  let inner_index = within_outer % params.inner;

  var sum = 0.0;
  for (var index_position = 0u; index_position < params.indices_elements;
       index_position = index_position + 1u) {
    // Gather does not normalize negative indices. Invalid values cannot match
    // a valid data-axis coordinate, so they make no contribution here.
    if (indices[index_position] == i32(axis_index)) {
      let output_index = (outer_index * params.indices_elements + index_position) *
        params.inner + inner_index;
      sum = sum + grad_output[output_index];
    }
  }
  grad_input[input_index] = grad_input[input_index] + sum;
}
