@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read> indices : array<i32>;
@group(0) @binding(2) var<storage, read_write> output : array<f32>;
// General Gather flattened as [outer, axis, inner]. The indices tensor may
// have any supported shape; its flattened elements replace the selected axis.
struct Params {
  outer : u32,
  axis_size : u32,
  inner : u32,
  indices_elements : u32,
  output_elements : u32,
}
@group(0) @binding(3) var<uniform> p : Params;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.output_elements) { return; }
  let inner_index = idx % p.inner;
  let flattened = idx / p.inner;
  let index_position = flattened % p.indices_elements;
  let outer_index = flattened / p.indices_elements;
  var selected = indices[index_position];
  if (selected < 0) {
    selected = selected + i32(p.axis_size);
  }
  // Keep the native GPU contract fail-closed for truly invalid indices: emit
  // the established sentinel instead of forming an out-of-bounds address.
  if (selected < 0 || selected >= i32(p.axis_size)) {
    output[idx] = -1.0;
    return;
  }
  let input_index = ((outer_index * p.axis_size + u32(selected)) * p.inner) + inner_index;
  output[idx] = input[input_index];
}
