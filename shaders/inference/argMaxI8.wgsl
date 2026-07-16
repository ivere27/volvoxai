// General axis ArgMax for F32 inputs and packed I8/U8 index outputs. One
// invocation writes a complete four-byte word, matching the raw TypedArray
// storage used by the graph allocator for both signed and unsigned byte types.
@group(0) @binding(0) var<storage, read> input : array<f32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  outer : u32,
  axis_size : u32,
  inner : u32,
  output_elements : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn argmax_index(output_index : u32) -> u32 {
  let outer_index = output_index / params.inner;
  let inner_index = output_index % params.inner;
  let base = outer_index * params.axis_size * params.inner + inner_index;
  var best = input[base];
  var best_index = 0u;
  for (var axis_index = 1u; axis_index < params.axis_size; axis_index = axis_index + 1u) {
    let value = input[base + axis_index * params.inner];
    if (value > best) {
      best = value;
      best_index = axis_index;
    }
  }
  return best_index;
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let word_index = gid.x;
  let first_output = word_index * 4u;
  if (first_output >= params.output_elements) { return; }

  var packed = 0u;
  for (var lane = 0u; lane < 4u; lane = lane + 1u) {
    let output_index = first_output + lane;
    if (output_index >= params.output_elements) { break; }
    let byte_value = argmax_index(output_index) & 0xffu;
    packed = packed | (byte_value << (lane * 8u));
  }
  output_words[word_index] = packed;
}
