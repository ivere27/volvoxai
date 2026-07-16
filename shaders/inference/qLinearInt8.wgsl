// Canonical W8A8 dense kernel. Activations, weights, and outputs are packed
// bytes in u32 storage words; every invocation owns a complete output word.
// Accumulation is I32 and bias is already in the input_scale*weight_scale
// domain. This generic path deliberately avoids optional integer-dot features.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> weight_words : array<u32>;
@group(0) @binding(2) var<storage, read> weight_scales : array<f32>;
@group(0) @binding(3) var<storage, read> weight_zero_points : array<i32>;
@group(0) @binding(4) var<storage, read> bias_values : array<i32>;
@group(0) @binding(5) var<storage, read_write> output_words : array<u32>;

struct Params {
  rows : u32,
  d_in : u32,
  d_out : u32,
  input_type : u32,
  weight_type : u32,
  output_type : u32,
  _pad0 : u32,
  _pad1 : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad2 : i32,
  _pad3 : i32,
  input_scale : f32,
  output_scale : f32,
  _pad4 : f32,
  _pad5 : f32,
}
@group(0) @binding(6) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn typed_byte(byte : u32, dtype : u32) -> i32 {
  if (dtype == 2u) { return signed_byte(byte); }
  return i32(byte);
}

fn input_value(index : u32) -> i32 {
  return typed_byte(word_byte(input_words[index / 4u], index), params.input_type);
}

fn weight_value(index : u32) -> i32 {
  return typed_byte(word_byte(weight_words[index / 4u], index), params.weight_type);
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

fn qlinear_one(index : u32) -> u32 {
  let elements = params.rows * params.d_out;
  if (index >= elements) { return 0u; }
  let row = index / params.d_out;
  let output_channel = index % params.d_out;
  var accumulator = bias_values[output_channel];
  let input_base = row * params.d_in;
  let weight_base = output_channel * params.d_in;
  for (var input_channel = 0u; input_channel < params.d_in; input_channel = input_channel + 1u) {
    accumulator = accumulator +
      (input_value(input_base + input_channel) - params.input_zero_point) *
      (weight_value(weight_base + input_channel) - weight_zero_points[output_channel]);
  }
  let multiplier = (params.input_scale * weight_scales[output_channel]) / params.output_scale;
  let transformed = f32(accumulator) * multiplier + f32(params.output_zero_point);
  let minimum = select(0, -128, params.output_type == 2u);
  let maximum = select(255, 127, params.output_type == 2u);
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  let elements = params.rows * params.d_out;
  if (first_element >= elements) { return; }
  var packed = qlinear_one(first_element);
  packed = packed | (qlinear_one(first_element + 1u) << 8u);
  packed = packed | (qlinear_one(first_element + 2u) << 16u);
  packed = packed | (qlinear_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
