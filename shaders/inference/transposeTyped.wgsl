// Descriptor-preserving I8/U8 transpose. Physical byte tensors are exposed
// to WGSL as packed u32 words; one invocation owns one complete destination
// word so byte lanes never race.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;
@group(0) @binding(2) var<storage, read> md : array<u32>;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn transposed_byte(output_index : u32) -> u32 {
  let elements = md[0];
  if (output_index >= elements) { return 0u; }
  let rank = md[1];
  var remainder = output_index;
  var input_index = 0u;
  for (var dimension = 0u; dimension < rank; dimension = dimension + 1u) {
    let output_stride = md[2u + dimension];
    let coordinate = remainder / output_stride;
    remainder = remainder - coordinate * output_stride;
    input_index = input_index + coordinate * md[2u + rank + dimension];
  }
  return word_byte(input_words[input_index / 4u], input_index);
}

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups) grid : vec3<u32>,
) {
  let output_word = gid.x + gid.y * (grid.x * 64u);
  let first_element = output_word * 4u;
  if (first_element >= md[0]) { return; }
  var packed = transposed_byte(first_element);
  packed = packed | (transposed_byte(first_element + 1u) << 8u);
  packed = packed | (transposed_byte(first_element + 2u) << 16u);
  packed = packed | (transposed_byte(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
