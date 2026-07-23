// @volvoxai-native-spv-only
requires packed_4x8_integer_dot_product;

// Packed-dot W8A8 dense kernel for single-row and small matrices.  Raw bytes
// stay packed in storage; U8 operands are xor-centered into I8 before DP4a.
// The correction terms preserve arbitrary canonical I8/U8 zero points.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> weight_words : array<u32>;
@group(0) @binding(2) var<storage, read> requant_multipliers : array<f32>;
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
  if (dtype == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn input_value(index : u32) -> i32 {
  return typed_byte(word_byte(input_words[index / 4u], index), params.input_type);
}

fn weight_value(index : u32) -> i32 {
  return typed_byte(word_byte(weight_words[index / 4u], index), params.weight_type);
}

fn input_word(base : u32) -> u32 {
  let word_index = base / 4u;
  let byte_offset = base % 4u;
  if (byte_offset == 0u) { return input_words[word_index]; }
  let shift = byte_offset * 8u;
  return (input_words[word_index] >> shift) |
    (input_words[word_index + 1u] << (32u - shift));
}

fn weight_word(base : u32) -> u32 {
  let word_index = base / 4u;
  let byte_offset = base % 4u;
  if (byte_offset == 0u) { return weight_words[word_index]; }
  let shift = byte_offset * 8u;
  return (weight_words[word_index] >> shift) |
    (weight_words[word_index + 1u] << (32u - shift));
}

fn signed_word(word : u32, dtype : u32) -> u32 {
  if (dtype == 5u) { return word ^ 0x80808080u; }
  return word;
}

fn centered_zero_point(zero_point : i32, dtype : u32) -> i32 {
  if (dtype == 5u) { return zero_point - 128; }
  return zero_point;
}

fn corrected_dot(input_word_raw : u32, weight_word_raw : u32,
                 weight_zero_point : i32) -> i32 {
  let input_word_signed = signed_word(input_word_raw, params.input_type);
  let weight_word_signed = signed_word(weight_word_raw, params.weight_type);
  let input_zero = centered_zero_point(params.input_zero_point, params.input_type);
  let weight_zero = centered_zero_point(weight_zero_point, params.weight_type);
  var value = dot4I8Packed(input_word_signed, weight_word_signed);
  if (weight_zero != 0) {
    value = value - weight_zero * dot4I8Packed(input_word_signed, 0x01010101u);
  }
  if (input_zero != 0) {
    value = value - input_zero * dot4I8Packed(weight_word_signed, 0x01010101u);
  }
  return value + 4 * input_zero * weight_zero;
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

fn highest_bit_u32(value : u32) -> u32 {
  var remaining = value;
  var result = 0u;
  while (remaining > 1u) {
    remaining = remaining >> 1u;
    result = result + 1u;
  }
  return result;
}

fn requantize_scaled_bits(bits : u32, zero_point : i32,
                          minimum : i32, maximum : i32) -> i32 {
  let magnitude = bits & 0x7fffffffu;
  let negative = (bits & 0x80000000u) != 0u;
  if (magnitude > 0x7f800000u) {
    return clamp(zero_point, minimum, maximum);
  }
  if (magnitude == 0x7f800000u) {
    return select(maximum, minimum, negative);
  }

  let exponent_bits = magnitude >> 23u;
  if (exponent_bits == 0u) {
    return clamp(zero_point, minimum, maximum);
  }
  let exponent = i32(exponent_bits) - 127;
  if (exponent < -2) {
    return clamp(zero_point, minimum, maximum);
  }
  if (exponent >= 9) {
    return select(maximum, minimum, negative);
  }

  let significand = (magnitude & 0x007fffffu) | 0x00800000u;
  let shift = u32(23 - exponent);
  let zero_negative = zero_point < 0;
  let zero_magnitude = u32(abs(zero_point));
  let zero_low = zero_magnitude << shift;
  let zero_high = zero_magnitude >> (32u - shift);

  var sum_low = 0u;
  var sum_high = 0u;
  var sum_negative = false;
  if (zero_point == 0) {
    sum_low = significand;
    sum_negative = negative;
  } else if (zero_negative == negative) {
    sum_low = zero_low + significand;
    sum_high = zero_high + select(0u, 1u, sum_low < zero_low);
    sum_negative = negative;
  } else {
    let zero_is_larger =
      zero_high != 0u || (zero_high == 0u && zero_low >= significand);
    if (zero_is_larger) {
      sum_low = zero_low - significand;
      sum_high = zero_high - select(0u, 1u, zero_low < significand);
      sum_negative = zero_negative;
    } else {
      sum_low = significand - zero_low;
      sum_negative = negative;
    }
  }
  if (sum_low == 0u && sum_high == 0u) {
    return 0;
  }

  let top_bit = select(
    highest_bit_u32(sum_low),
    32u + highest_bit_u32(sum_high),
    sum_high != 0u);
  if (top_bit > 23u) {
    let discard_count = top_bit - 23u;
    var rounded_significand =
      (sum_low >> discard_count) |
      (sum_high << (32u - discard_count));
    let discarded = sum_low & ((1u << discard_count) - 1u);
    let addition_halfway = 1u << (discard_count - 1u);
    if (discarded > addition_halfway ||
        (discarded == addition_halfway &&
         (rounded_significand & 1u) != 0u)) {
      rounded_significand = rounded_significand + 1u;
    }
    sum_low = rounded_significand << discard_count;
    sum_high = rounded_significand >> (32u - discard_count);
  }

  let integer_magnitude =
    (sum_low >> shift) | (sum_high << (32u - shift));
  let remainder = sum_low & ((1u << shift) - 1u);
  let halfway = 1u << (shift - 1u);
  var rounded = select(
    i32(integer_magnitude), -i32(integer_magnitude), sum_negative);
  let step = select(1, -1, sum_negative);
  if (remainder > halfway ||
      (remainder == halfway && (rounded & 1) != 0)) {
    rounded = rounded + step;
  }
  return clamp(rounded, minimum, maximum);
}

fn qlinear_one(index : u32) -> u32 {
  let elements = params.rows * params.d_out;
  if (index >= elements) { return 0u; }
  let row = index / params.d_out;
  let output_channel = index % params.d_out;
  let input_base = row * params.d_in;
  let weight_base = output_channel * params.d_in;
  let weight_zero_point = weight_zero_points[output_channel];
  var accumulator = bias_values[output_channel];
  var input_channel = 0u;
  for (; input_channel + 4u <= params.d_in; input_channel = input_channel + 4u) {
    accumulator = accumulator + corrected_dot(
      input_word(input_base + input_channel),
      weight_word(weight_base + input_channel),
      weight_zero_point);
  }
  for (; input_channel < params.d_in; input_channel = input_channel + 1u) {
    accumulator = accumulator +
      (input_value(input_base + input_channel) - params.input_zero_point) *
      (weight_value(weight_base + input_channel) - weight_zero_point);
  }
  let multiplier = requant_multipliers[output_channel];
  let scaled_bits = bitcast<u32>(f32(accumulator) * multiplier);
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  return output_byte(requantize_scaled_bits(
    scaled_bits, params.output_zero_point, minimum, maximum));
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
