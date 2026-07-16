// ONNX GatherElements for F32 data and I32 indices. Each output coordinate
// matches the index tensor, except the selected axis is replaced by its index.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;

struct Params {
  // rank, axis, output_elements, unused
  dimensions : vec4<u32>,
  data_shape0 : vec4<u32>,
  data_shape1 : vec4<u32>,
  index_shape0 : vec4<u32>,
  index_shape1 : vec4<u32>,
}
@group(0) @binding(3) var<uniform> params : Params;

fn data_dim(index : u32) -> u32 {
  if (index < 4u) { return params.data_shape0[index]; }
  return params.data_shape1[index - 4u];
}

fn index_dim(index : u32) -> u32 {
  if (index < 4u) { return params.index_shape0[index]; }
  return params.index_shape1[index - 4u];
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_index = gid.x;
  if (output_index >= params.dimensions.z) { return; }

  let rank = params.dimensions.x;
  let axis = params.dimensions.y;
  var coords : array<u32, 8>;
  var remaining = output_index;
  var dimension = rank;
  loop {
    if (dimension == 0u) { break; }
    dimension = dimension - 1u;
    let size = index_dim(dimension);
    coords[dimension] = remaining % size;
    remaining = remaining / size;
  }

  var gathered = indices[output_index];
  if (gathered < 0) { gathered = gathered + i32(data_dim(axis)); }
  if (gathered < 0 || gathered >= i32(data_dim(axis))) {
    output[output_index] = -1.0;
    return;
  }

  var input_index = 0u;
  for (var input_dimension = 0u; input_dimension < rank;
       input_dimension = input_dimension + 1u) {
    let coordinate = select(coords[input_dimension], u32(gathered), input_dimension == axis);
    input_index = input_index * data_dim(input_dimension) + coordinate;
  }
  output[output_index] = input[input_index];
}
