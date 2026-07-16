// Canonical metadata-only W8A8 requantization. Both sides already own their
// immutable quantization descriptors; this kernel changes the packed byte
// representation without introducing mutable scale tensors or an F32 buffer.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  elements : u32,
  input_type : u32,
  output_type : u32,
  _pad0 : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad1 : i32,
  _pad2 : i32,
  multiplier : f32,
  _pad3 : f32,
  _pad4 : f32,
  _pad5 : f32,
}
@group(0) @binding(2) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn input_value(index : u32) -> i32 {
  let value = word_byte(input_words[index / 4u], index);
  if (params.input_type == 2u) { return signed_byte(value); }
  return i32(value);
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

fn requantize_one(index : u32) -> u32 {
  if (index >= params.elements) { return 0u; }
  let minimum = select(0, -128, params.output_type == 2u);
  let maximum = select(255, 127, params.output_type == 2u);
  let centered = input_value(index) - params.input_zero_point;
  let transformed = f32(f32(centered) * params.multiplier) + f32(params.output_zero_point);
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  if (first_element >= params.elements) { return; }
  var packed = requantize_one(first_element);
  packed = packed | (requantize_one(first_element + 1u) << 8u);
  packed = packed | (requantize_one(first_element + 2u) << 16u);
  packed = packed | (requantize_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
