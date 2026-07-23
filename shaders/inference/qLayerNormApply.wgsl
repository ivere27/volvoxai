// QLayerNorm apply pass. Statistics are already stable and row-local; each
// invocation owns one packed u32 output word, so an odd logical byte tail is
// zero-filled rather than racing another invocation.
//
// Params is the fixed 48-byte ABI shared with qLayerNormStats.
struct Params {
  rows : u32,
  d_model : u32,
  input_type : u32,
  output_type : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad0 : i32,
  _pad1 : i32,
  input_scale : f32,
  output_scale : f32,
  epsilon : f32,
  _pad2 : f32,
}

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> gamma : array<f32>;
@group(0) @binding(2) var<storage, read> beta : array<f32>;
@group(0) @binding(3) var<storage, read> row_stats : array<f32>;
@group(0) @binding(4) var<storage, read_write> output_words : array<u32>;
@group(0) @binding(5) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn input_value(index : u32) -> i32 {
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 6u) { return signed_byte(byte); }
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

fn qlayernorm_one(index : u32) -> u32 {
  let elements = params.rows * params.d_model;
  if (index >= elements) { return 0u; }
  let row = index / params.d_model;
  let channel = index % params.d_model;
  let stats_offset = row * 2u;
  let raw = f32(input_value(index) - params.input_zero_point);
  let normalized = (raw - row_stats[stats_offset]) * params.input_scale *
    row_stats[stats_offset + 1u];
  let value = normalized * gamma[channel] + beta[channel];
  let transformed = value / params.output_scale + f32(params.output_zero_point);
  let minimum : i32 = select(0, -128, params.output_type == 6u);
  let maximum : i32 = select(255, 127, params.output_type == 6u);
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
  let output_word = global_invocation_id.x;
  let first_element = output_word * 4u;
  let elements = params.rows * params.d_model;
  if (first_element >= elements) { return; }
  var packed = qlayernorm_one(first_element);
  packed = packed | (qlayernorm_one(first_element + 1u) << 8u);
  packed = packed | (qlayernorm_one(first_element + 2u) << 16u);
  packed = packed | (qlayernorm_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
