// Exact-shape W8A8 QAdd. Activations are packed as four I8/U8 values per u32;
// one invocation writes one complete destination word so byte lanes never race.
@group(0) @binding(0) var<storage, read> a_words : array<u32>;
@group(0) @binding(1) var<storage, read> b_words : array<u32>;
@group(0) @binding(2) var<storage, read_write> output_words : array<u32>;

struct Params {
  elements : u32,
  a_type : u32,
  b_type : u32,
  output_type : u32,
  a_zero_point : i32,
  b_zero_point : i32,
  output_zero_point : i32,
  _pad0 : i32,
  a_scale : f32,
  b_scale : f32,
  output_scale : f32,
  relu : u32,
}
@group(0) @binding(3) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn typed_byte_value(word : u32, dtype : u32) -> i32 {
  let byte = word & 255u;
  if (dtype == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn a_value(index : u32) -> i32 {
  return typed_byte_value(word_byte(a_words[index / 4u], index), params.a_type);
}

fn b_value(index : u32) -> i32 {
  return typed_byte_value(word_byte(b_words[index / 4u], index), params.b_type);
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

fn qadd_one(index : u32) -> u32 {
  if (index >= params.elements) { return 0u; }
  let av = f32(a_value(index) - params.a_zero_point) * params.a_scale;
  let bv = f32(b_value(index) - params.b_zero_point) * params.b_scale;
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  let transformed = (av + bv) / params.output_scale + f32(params.output_zero_point);
  var quantized : i32;
  if (transformed != transformed) { quantized = params.output_zero_point; }
  else if (transformed <= f32(minimum)) { quantized = minimum; }
  else if (transformed >= f32(maximum)) { quantized = maximum; }
  else { quantized = round_even(transformed); }
  if (params.relu != 0u) {
    quantized = max(quantized, params.output_zero_point);
    if (params.relu >= 2u) {
      let upper_transformed = 6.0 / params.output_scale + f32(params.output_zero_point);
      var upper : i32;
      if (upper_transformed <= f32(minimum)) { upper = minimum; }
      else if (upper_transformed >= f32(maximum)) { upper = maximum; }
      else { upper = round_even(upper_transformed); }
      quantized = min(quantized, upper);
    }
  }
  return output_byte(quantized);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  if (first_element >= params.elements) { return; }
  var packed = qadd_one(first_element);
  packed = packed | (qadd_one(first_element + 1u) << 8u);
  packed = packed | (qadd_one(first_element + 2u) << 16u);
  packed = packed | (qadd_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
