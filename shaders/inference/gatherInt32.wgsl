// General ONNX-style Gather for F32 data and I32 indices.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

struct Params {
  // data_rank, indices_rank, axis, output_elements
  dimensions : vec4<u32>,
  data_shape0 : vec4<u32>,
  data_shape1 : vec4<u32>,
  output_shape0 : vec4<u32>,
  output_shape1 : vec4<u32>,
}
@group(0) @binding(3) var<uniform> params : Params;

fn data_dim(index : u32) -> u32 {
  if (index < 4u) { return params.data_shape0[index]; }
  return params.data_shape1[index - 4u];
}

fn output_dim(index : u32) -> u32 {
  if (index < 4u) { return params.output_shape0[index]; }
  return params.output_shape1[index - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_index = gid.x;
  if (output_index >= params.dimensions.w) { return; }

  let data_rank = params.dimensions.x;
  let indices_rank = params.dimensions.y;
  let axis = params.dimensions.z;
  let output_rank = data_rank + indices_rank - 1u;

  var output_coords : array<u32, 8>;
  var remaining = output_index;
  var dimension = output_rank;
  loop {
    if (dimension == 0u) { break; }
    dimension = dimension - 1u;
    let size = output_dim(dimension);
    output_coords[dimension] = remaining % size;
    remaining = remaining / size;
  }

  var index_offset = 0u;
  for (var index_dimension = 0u; index_dimension < indices_rank;
       index_dimension = index_dimension + 1u) {
    let output_dimension = axis + index_dimension;
    index_offset = index_offset * output_dim(output_dimension) + output_coords[output_dimension];
  }
  var gathered = indices[index_offset];
  if (gathered < 0) {
    gathered = gathered + i32(data_dim(axis));
  }
  if (gathered < 0 || gathered >= i32(data_dim(axis))) {
    // Match the portable Gather sentinel without forming an out-of-bounds
    // address when one ONNX negative-index normalization is still invalid.
    output[output_index] = -1.0;
    return;
  }

  var input_index = 0u;
  for (var input_dimension = 0u; input_dimension < data_rank;
       input_dimension = input_dimension + 1u) {
    var coordinate = 0u;
    if (input_dimension < axis) {
      coordinate = output_coords[input_dimension];
    } else if (input_dimension == axis) {
      coordinate = u32(gathered);
    } else {
      coordinate = output_coords[input_dimension - 1u + indices_rank];
    }
    input_index = input_index * data_dim(input_dimension) + coordinate;
  }
  output[output_index] = input[input_index];
}
