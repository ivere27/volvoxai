// Positive-step Slice backward for the canonical rank-1..8 WebGPU contract.
// The uniform layout deliberately matches inference/sliceNd.wgsl:
// rank, output_elements, output_shape[8], starts[8], steps[8], input_strides[8].
@group(0) @binding(0) var<storage, read> grad_out : array<f32>;
@group(0) @binding(1) var<storage, read_write> grad_input : array<f32>;

struct Params {
  // rank, output_elements, pad, pad
  dimensions : vec4<u32>,
  output_shape0 : vec4<u32>,
  output_shape1 : vec4<u32>,
  starts0 : vec4<u32>,
  starts1 : vec4<u32>,
  steps0 : vec4<u32>,
  steps1 : vec4<u32>,
  input_strides0 : vec4<u32>,
  input_strides1 : vec4<u32>,
}
@group(0) @binding(2) var<uniform> params : Params;

fn output_dim(index : u32) -> u32 {
  if (index < 4u) { return params.output_shape0[index]; }
  return params.output_shape1[index - 4u];
}

fn start_at(index : u32) -> u32 {
  if (index < 4u) { return params.starts0[index]; }
  return params.starts1[index - 4u];
}

fn step_at(index : u32) -> u32 {
  if (index < 4u) { return params.steps0[index]; }
  return params.steps1[index - 4u];
}

fn input_stride(index : u32) -> u32 {
  if (index < 4u) { return params.input_strides0[index]; }
  return params.input_strides1[index - 4u];
}

fn input_index(output_index : u32) -> u32 {
  var remaining = output_index;
  var source = 0u;
  var axis = params.dimensions.x;
  loop {
    if (axis == 0u) { break; }
    axis = axis - 1u;
    let coordinate = remaining % output_dim(axis);
    remaining = remaining / output_dim(axis);
    source = source + (start_at(axis) + coordinate * step_at(axis)) * input_stride(axis);
  }
  return source;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.dimensions.y) { return; }
  // Positive nonzero steps make this mapping injective, so no atomic add is needed.
  let source = input_index(index);
  grad_input[source] = grad_input[source] + grad_out[index];
}
