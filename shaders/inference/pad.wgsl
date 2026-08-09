@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output : array<f32>;

struct Params {
  input_shape : vec4<u32>,
  output_shape : vec4<u32>,
  pads_before : vec4<u32>,
  output_elements : u32,
  value : f32,
  _pad : vec2<u32>,
}
@group(0) @binding(2) var<uniform> params : Params;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let output_index = global_id.x;
  if (output_index >= params.output_elements) { return; }

  var coordinates : vec4<u32>;
  var remaining = output_index;
  coordinates.w = remaining % params.output_shape.w;
  remaining = remaining / params.output_shape.w;
  coordinates.z = remaining % params.output_shape.z;
  remaining = remaining / params.output_shape.z;
  coordinates.y = remaining % params.output_shape.y;
  coordinates.x = remaining / params.output_shape.y;

  let after_start = coordinates >= params.pads_before;
  let before_end = coordinates < params.pads_before + params.input_shape;
  if (!all(after_start) || !all(before_end)) {
    output[output_index] = params.value;
    return;
  }

  let source = coordinates - params.pads_before;
  let input_index = ((source.x * params.input_shape.y + source.y) *
    params.input_shape.z + source.z) * params.input_shape.w + source.w;
  output[output_index] = input[input_index];
}
