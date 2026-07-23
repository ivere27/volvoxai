// Canonical I32 GatherElements backward.  Each invocation owns one data
// gradient element and scans only the index-axis positions that can map to it.
// This is a deterministic scatter-add reduction and avoids float atomics.
@group(0) @binding(0) var<storage, read> indices : array<i32>;
@group(0) @binding(1) var<storage, read> grad_output : array<f32>;
@group(0) @binding(2) var<storage, read_write> grad_input : array<f32>;

struct Params {
  // rank, axis, input_elements, output_elements
  dimensions : vec4<u32>,
  input_shape0 : vec4<u32>,
  input_shape1 : vec4<u32>,
  index_shape0 : vec4<u32>,
  index_shape1 : vec4<u32>,
}
@group(0) @binding(3) var<uniform> params : Params;

fn input_dim(index : u32) -> u32 {
  if (index < 4u) { return params.input_shape0[index]; }
  return params.input_shape1[index - 4u];
}

fn index_dim(index : u32) -> u32 {
  if (index < 4u) { return params.index_shape0[index]; }
  return params.index_shape1[index - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let input_index = gid.x;
  let rank = params.dimensions.x;
  let axis = params.dimensions.y;
  if (input_index >= params.dimensions.z) { return; }

  var input_coords : array<u32, 8>;
  var remaining = input_index;
  var dimension = rank;
  loop {
    if (dimension == 0u) { break; }
    dimension = dimension - 1u;
    let size = input_dim(dimension);
    input_coords[dimension] = remaining % size;
    remaining = remaining / size;
  }

  // At dimensions other than the gather axis, an output coordinate can only
  // address the identical input coordinate. Inputs beyond an index dimension
  // therefore receive no contribution.
  for (var d = 0u; d < rank; d = d + 1u) {
    if (d != axis && input_coords[d] >= index_dim(d)) { return; }
  }

  let axis_coordinate = input_coords[axis];
  var sum = 0.0;
  for (var index_axis_coordinate = 0u; index_axis_coordinate < index_dim(axis);
       index_axis_coordinate = index_axis_coordinate + 1u) {
    var output_index = 0u;
    for (var d = 0u; d < rank; d = d + 1u) {
      let coordinate = select(input_coords[d], index_axis_coordinate, d == axis);
      output_index = output_index * index_dim(d) + coordinate;
    }
    if (output_index >= params.dimensions.w) { continue; }
    var selected = indices[output_index];
    // ONNX gather-family indices normalize once by the selected axis size.
    if (selected < 0) { selected = selected + i32(input_dim(axis)); }
    if (selected == i32(axis_coordinate)) {
      sum = sum + grad_output[output_index];
    }
  }
  grad_input[input_index] = grad_input[input_index] + sum;
}
