// @volvoxai-browser-only
// Bit-preserving F32/I32 concat copy without fused arithmetic.
@group(0) @binding(0) var<storage, read> input : array<u32>;
@group(0) @binding(1) var<storage, read_write> output : array<u32>;

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

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let input_index = gid.x;
  if (input_index >= params.size) { return; }
  let inner_index = input_index % params.inner;
  let outer_axis = input_index / params.inner;
  let axis_index = outer_axis % params.input_axis;
  let outer_index = outer_axis / params.input_axis;
  let output_index =
    ((outer_index * params.output_axis + params.axis_offset + axis_index) *
      params.inner) + inner_index;
  output[output_index] = input[input_index];
}
