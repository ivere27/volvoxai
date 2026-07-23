// @volvoxai-browser-only
// Browser-only canonical positive-step Slice. The native-facing slice.wgsl has
// a rank-4 contract; this path supports rank 1..8 metadata.
@group(0) @binding(0) var<storage, read> input : array<u32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_index = gid.x;
  let rank = params.dimensions.x;
  if (output_index >= params.dimensions.y) { return; }

  var remaining = output_index;
  var input_index = 0u;
  var dimension = rank;
  loop {
    if (dimension == 0u) { break; }
    dimension = dimension - 1u;
    let coordinate = remaining % output_dim(dimension);
    remaining = remaining / output_dim(dimension);
    input_index = input_index +
      (start_at(dimension) + coordinate * step_at(dimension)) * input_stride(dimension);
  }
  output[output_index] = input[input_index];
}
