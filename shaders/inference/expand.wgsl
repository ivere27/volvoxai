@group(0) @binding(0) var<storage, read> input: array<f32>;
@group(0) @binding(1) var<storage, read_write> output: array<f32>;

struct Params {
  dimensions: vec4<u32>, // input_rank, output_rank, unused, output_elements
  input_shape0: vec4<u32>,
  input_shape1: vec4<u32>,
  output_shape0: vec4<u32>,
  output_shape1: vec4<u32>,
}
@group(0) @binding(2) var<uniform> params: Params;

fn input_dim(index: u32) -> u32 {
  if (index < 4u) { return params.input_shape0[index]; }
  return params.input_shape1[index - 4u];
}

fn output_dim(index: u32) -> u32 {
  if (index < 4u) { return params.output_shape0[index]; }
  return params.output_shape1[index - 4u];
}

fn input_stride(index: u32) -> u32 {
  var stride = 1u;
  var dimension = params.dimensions.x;
  loop {
    if (dimension == index + 1u) { break; }
    dimension = dimension - 1u;
    stride = stride * input_dim(dimension);
  }
  return stride;
}

fn input_index(output_index: u32) -> u32 {
  var remaining = output_index;
  var result = 0u;
  var dimension = params.dimensions.y;
  loop {
    if (dimension == 0u) { break; }
    dimension = dimension - 1u;
    let coordinate = remaining % output_dim(dimension);
    remaining = remaining / output_dim(dimension);
    let offset = params.dimensions.y - params.dimensions.x;
    if (dimension >= offset && input_dim(dimension - offset) != 1u) {
      result = result + coordinate * input_stride(dimension - offset);
    }
  }
  return result;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let index = gid.x;
  if (index >= params.dimensions.w) { return; }
  output[index] = input[input_index(index)];
}
