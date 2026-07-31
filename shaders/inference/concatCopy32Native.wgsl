// Native raw 32-bit concat copy. Each invocation owns one complete destination
// element, so F32 bit patterns and I32 values are preserved exactly.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  size : u32,
  axis_offset : u32,
  input_axis : u32,
  output_axis : u32,
  inner : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn output_index(input_index : u32) -> u32 {
  let inner_index = input_index % params.inner;
  let outer_axis = input_index / params.inner;
  let axis_index = outer_axis % params.input_axis;
  let outer_index = outer_axis / params.input_axis;
  return ((outer_index * params.output_axis +
           params.axis_offset + axis_index) *
          params.inner) + inner_index;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let input_index = gid.x;
  if (input_index >= params.size) { return; }
  output_words[output_index(input_index)] = input_words[input_index];
}
