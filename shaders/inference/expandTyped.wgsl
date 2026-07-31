// @volvoxai-browser-only
// Descriptor-preserving I8/U8 right-aligned broadcast. Physical byte tensors
// are exposed as packed u32 words; each invocation owns one complete output
// word so byte lanes never race.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  dimensions : vec4<u32>, // input_rank, output_rank, unused, output_elements
  input_shape0 : vec4<u32>,
  input_shape1 : vec4<u32>,
  output_shape0 : vec4<u32>,
  output_shape1 : vec4<u32>,
}
@group(0) @binding(2) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn input_dim(index : u32) -> u32 {
  if (index < 4u) { return params.input_shape0[index]; }
  return params.input_shape1[index - 4u];
}

fn output_dim(index : u32) -> u32 {
  if (index < 4u) { return params.output_shape0[index]; }
  return params.output_shape1[index - 4u];
}

fn input_stride(index : u32) -> u32 {
  var stride = 1u;
  var dimension = params.dimensions.x;
  loop {
    if (dimension == index + 1u) { break; }
    dimension = dimension - 1u;
    stride = stride * input_dim(dimension);
  }
  return stride;
}

fn input_index(output_index : u32) -> u32 {
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

fn expanded_byte(output_index : u32) -> u32 {
  if (output_index >= params.dimensions.w) { return 0u; }
  let source_index = input_index(output_index);
  return word_byte(input_words[source_index / 4u], source_index);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  if (first_element >= params.dimensions.w) { return; }
  var packed = expanded_byte(first_element);
  packed = packed | (expanded_byte(first_element + 1u) << 8u);
  packed = packed | (expanded_byte(first_element + 2u) << 16u);
  packed = packed | (expanded_byte(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
