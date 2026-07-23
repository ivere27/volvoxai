// Canonical byte-domain QArgMax.  Positive per-tensor affine quantization
// preserves raw I8/U8 ordering, so this shader reads bytes directly and
// writes only I32 winner indices.  Strict comparison retains the first index
// when a row has tied maxima.
//
// Fixed 16-byte ABI shared with native GPU backends:
//   u32 outer, axis_size, inner, input_dtype
// input_dtype uses canonical protobuf values I8=6 or U8=5. One 64-lane
// invocation owns one output index.
struct Params {
  outer : u32,
  axis_size : u32,
  inner : u32,
  input_dtype : u32,
}

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output : array<i32>;
@group(0) @binding(2) var<uniform> params : Params;

fn input_value(index : u32) -> i32 {
  let word = input_words[index / 4u];
  let byte = (word >> ((index % 4u) * 8u)) & 255u;
  if (params.input_dtype == 6u && byte >= 128u) {
    return i32(byte) - 256;
  }
  return i32(byte);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_id : vec3<u32>) {
  let output_index = global_id.x;
  let output_elements = params.outer * params.inner;
  if (output_index >= output_elements) { return; }

  let outer_index = output_index / params.inner;
  let inner_index = output_index % params.inner;
  let base = outer_index * params.axis_size * params.inner + inner_index;
  var best = input_value(base);
  var best_index = 0u;
  for (var axis_index = 1u; axis_index < params.axis_size; axis_index = axis_index + 1u) {
    let value = input_value(base + axis_index * params.inner);
    if (value > best) {
      best = value;
      best_index = axis_index;
    }
  }
  output[output_index] = i32(best_index);
}
