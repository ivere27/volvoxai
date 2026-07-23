// Typed DequantizeLinear over F32/I32/I8/U8 input and zero-point storage.
// The portable contract uses scalar F32 scale and F32 output; raw words retain
// integer input precision until the subtraction is complete.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> scale_words : array<u32>;
@group(0) @binding(2) var<storage, read> zero_point_words : array<u32>;
@group(0) @binding(3) var<storage, read_write> output_words : array<u32>;

struct Params {
  size : u32,
  input_type : u32,
  scale_type : u32,
  zero_point_type : u32,
  output_type : u32,
  has_zero_point : u32,
  _pad0 : u32,
  _pad1 : u32,
}
@group(0) @binding(4) var<uniform> params : Params;

struct SignedMagnitude {
  negative : u32,
  magnitude : u32,
  valid : u32,
}

struct DifferenceResult {
  value : f32,
  valid : u32,
}

// Dtype IDs are canonical protobuf values: F32=18, I32=16, I8=6, U8=5.
fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn input_value(index : u32) -> f32 {
  if (params.input_type == 18u) { return bitcast<f32>(input_words[index]); }
  if (params.input_type == 16u) { return f32(bitcast<i32>(input_words[index])); }
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 6u) { return f32(signed_byte(byte)); }
  return f32(byte);
}

fn input_integer_value(index : u32) -> i32 {
  if (params.input_type == 16u) { return bitcast<i32>(input_words[index]); }
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn scale_value() -> f32 {
  if (params.scale_type == 18u) { return bitcast<f32>(scale_words[0]); }
  if (params.scale_type == 16u) { return f32(bitcast<i32>(scale_words[0])); }
  let byte = word_byte(scale_words[0], 0u);
  if (params.scale_type == 6u) { return f32(signed_byte(byte)); }
  return f32(byte);
}

fn zero_point_value() -> f32 {
  if (params.zero_point_type == 18u) { return bitcast<f32>(zero_point_words[0]); }
  if (params.zero_point_type == 16u) { return f32(bitcast<i32>(zero_point_words[0])); }
  let byte = word_byte(zero_point_words[0], 0u);
  if (params.zero_point_type == 6u) { return f32(signed_byte(byte)); }
  return f32(byte);
}

fn zero_point_integer_value() -> i32 {
  if (params.zero_point_type == 16u) { return bitcast<i32>(zero_point_words[0]); }
  let byte = word_byte(zero_point_words[0], 0u);
  if (params.zero_point_type == 6u) { return signed_byte(byte); }
  return i32(byte);
}

fn signed_magnitude_i32(value : i32) -> SignedMagnitude {
  let bits = bitcast<u32>(value);
  if (value < 0) { return SignedMagnitude(1u, 0u - bits, 1u); }
  return SignedMagnitude(0u, bits, 1u);
}

// Decode an integral F32 whose magnitude fits in u32. This covers the whole
// cancellation range against an I32 (including +2^31 through <2^32); larger
// values cannot retain a small I32 delta in an F32 result anyway.
fn f32_integral_signed_magnitude(bits : u32) -> SignedMagnitude {
  let exponent = (bits >> 23u) & 255u;
  let sign = (bits & 0x80000000u) != 0u;
  if (exponent == 255u) { return SignedMagnitude(0u, 0u, 0u); }
  if (exponent == 0u) {
    if ((bits & 0x7fffffffu) == 0u) { return SignedMagnitude(0u, 0u, 1u); }
    return SignedMagnitude(0u, 0u, 0u);
  }
  if (exponent < 127u) { return SignedMagnitude(0u, 0u, 0u); }

  let unbiased_exponent = exponent - 127u;
  if (unbiased_exponent >= 32u) { return SignedMagnitude(0u, 0u, 0u); }
  let significand = (bits & 0x007fffffu) | 0x00800000u;
  var magnitude = 0u;
  if (unbiased_exponent < 23u) {
    let discarded_mask = (1u << (23u - unbiased_exponent)) - 1u;
    if ((significand & discarded_mask) != 0u) { return SignedMagnitude(0u, 0u, 0u); }
    magnitude = significand >> (23u - unbiased_exponent);
  } else {
    magnitude = significand << (unbiased_exponent - 23u);
  }
  return SignedMagnitude(select(0u, 1u, sign && magnitude != 0u), magnitude, 1u);
}

fn input_signed_magnitude(index : u32) -> SignedMagnitude {
  if (params.input_type == 18u) { return f32_integral_signed_magnitude(input_words[index]); }
  return signed_magnitude_i32(input_integer_value(index));
}

fn zero_point_signed_magnitude() -> SignedMagnitude {
  if (params.zero_point_type == 18u) { return f32_integral_signed_magnitude(zero_point_words[0]); }
  return signed_magnitude_i32(zero_point_integer_value());
}

// JS and the C reference subtract typed values before F32 output conversion.
// This keeps high I32 deltas such as 16,777,217 - 16,777,216 (whether the zero
// point is I32 or F32) intact until the one final F32 conversion.
fn signed_magnitude_difference(left : SignedMagnitude, right : SignedMagnitude) -> DifferenceResult {
  if (left.negative == right.negative) {
    if (left.magnitude >= right.magnitude) {
      let magnitude = left.magnitude - right.magnitude;
      if (left.negative != 0u) { return DifferenceResult(-f32(magnitude), 1u); }
      return DifferenceResult(f32(magnitude), 1u);
    }
    let magnitude = right.magnitude - left.magnitude;
    if (right.negative != 0u) { return DifferenceResult(f32(magnitude), 1u); }
    return DifferenceResult(-f32(magnitude), 1u);
  }

  // The only unrepresentable u32 magnitude is an opposite-sign sum above
  // 2^32-1. Fall back to ordinary F32 arithmetic for that non-cancelling case.
  if (left.magnitude > 0xffffffffu - right.magnitude) {
    return DifferenceResult(0.0, 0u);
  }
  let magnitude = left.magnitude + right.magnitude;
  if (left.negative != 0u) { return DifferenceResult(-f32(magnitude), 1u); }
  return DifferenceResult(f32(magnitude), 1u);
}

fn input_minus_zero_point(index : u32) -> f32 {
  if (params.has_zero_point != 0u) {
    let left = input_signed_magnitude(index);
    let right = zero_point_signed_magnitude();
    if (left.valid != 0u && right.valid != 0u) {
      let difference = signed_magnitude_difference(left, right);
      if (difference.valid != 0u) { return difference.value; }
    }
  }
  var zero_point = 0.0;
  if (params.has_zero_point != 0u) { zero_point = zero_point_value(); }
  return input_value(index) - zero_point;
}

fn dequantize_one(index : u32) -> u32 {
  let value = input_minus_zero_point(index) * scale_value();
  // The portable DequantizeLinear contract has an F32 output. GraphExecutor
  // rejects non-F32 declarations before this shader is compiled.
  return bitcast<u32>(value);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  if (params.output_type == 18u || params.output_type == 16u) {
    if (output_word >= params.size) { return; }
    output_words[output_word] = dequantize_one(output_word);
    return;
  }

  let first_element = output_word * 4u;
  if (first_element >= params.size) { return; }
  var packed = 0u;
  for (var lane = 0u; lane < 4u; lane = lane + 1u) {
    let element = first_element + lane;
    if (element >= params.size) { break; }
    packed = packed | ((dequantize_one(element) & 255u) << (lane * 8u));
  }
  output_words[output_word] = packed;
}
