// Canonical typed QuantizeLinear for the W8A8 graph contract. Input and scale
// are F32; output storage is packed I8/U8 bytes. One invocation owns a full
// u32 destination word, avoiding byte-lane write races on WebGPU and every
// native GPU backend that consumes this source.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> scale_words : array<u32>;
@group(0) @binding(2) var<storage, read> zero_point_words : array<u32>;
@group(0) @binding(3) var<storage, read_write> output_words : array<u32>;

struct Params {
  elements : u32,
  output_type : u32,
  zero_point_type : u32,
  has_zero_point : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

// Dtype IDs are canonical protobuf values: I8=6, U8=5. QuantizeLinear accepts
// only typed byte outputs, but retaining the ID makes the ABI self-describing.
fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn zero_point_value() -> i32 {
  if (params.has_zero_point == 0u) { return 0; }
  let byte = word_byte(zero_point_words[0], 0u);
  if (params.zero_point_type == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn round_even(value : f32) -> i32 {
  let lower = floor(value);
  let fraction = value - lower;
  if (fraction < 0.5) { return i32(lower); }
  if (fraction > 0.5) { return i32(lower + 1.0); }
  let lower_i = i32(lower);
  if ((lower_i & 1) == 0) { return lower_i; }
  return lower_i + 1;
}

fn output_byte(value : i32) -> u32 {
  return bitcast<u32>(value) & 255u;
}

fn quantize_one(index : u32) -> u32 {
  if (index >= params.elements) { return 0u; }
  let zero = zero_point_value();
  let input = bitcast<f32>(input_words[index]);
  if (input != input) { return output_byte(zero); }
  let scale = bitcast<f32>(scale_words[0]);
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  // Invalid dynamic scales are rejected by the CPU/WASM validators. Keep the
  // shader defined as well, so an accidental runtime mutation cannot produce
  // an undefined integer conversion.
  if (!(scale > 0.0) || scale != scale) { return output_byte(zero); }
  let transformed = input / scale + f32(zero);
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  if (transformed != transformed) { return output_byte(zero); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(
  @builtin(global_invocation_id) gid : vec3<u32>,
  @builtin(num_workgroups) grid : vec3<u32>,
) {
  let output_word = gid.x + gid.y * (grid.x * 64u);
  let first_element = output_word * 4u;
  if (first_element >= params.elements) { return; }
  var packed = quantize_one(first_element);
  packed = packed | (quantize_one(first_element + 1u) << 8u);
  packed = packed | (quantize_one(first_element + 2u) << 16u);
  packed = packed | (quantize_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
