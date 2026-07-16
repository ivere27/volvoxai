// DequantizeLinear backward keeps the forward shader's portable raw-word
// decoding for scale gradients. Quantized inputs intentionally receive no
// gradient; input_main is dispatched only for the F32-input case.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> zero_point_words : array<u32>;
@group(0) @binding(2) var<storage, read> scale : array<f32>;
@group(0) @binding(3) var<storage, read> grad_output : array<f32>;
@group(0) @binding(4) var<storage, read_write> grad_input : array<f32>;
@group(0) @binding(5) var<storage, read_write> grad_scale : array<f32>;

struct Params {
  size : u32,
  input_type : u32,
  zero_point_type : u32,
  has_zero_point : u32,
}
@group(0) @binding(6) var<uniform> params : Params;

struct SignedMagnitude {
  negative : u32,
  magnitude : u32,
  valid : u32,
}

struct DifferenceResult {
  value : f32,
  valid : u32,
}

// Runtime dtype IDs are F32=0, I32=1, I8=2, U8=3.
fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn input_value(index : u32) -> f32 {
  if (params.input_type == 0u) { return bitcast<f32>(input_words[index]); }
  if (params.input_type == 1u) { return f32(bitcast<i32>(input_words[index])); }
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 2u) { return f32(signed_byte(byte)); }
  return f32(byte);
}

fn input_integer_value(index : u32) -> i32 {
  if (params.input_type == 1u) { return bitcast<i32>(input_words[index]); }
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 2u) { return signed_byte(byte); }
  return i32(byte);
}

fn zero_point_value() -> f32 {
  if (params.zero_point_type == 0u) { return bitcast<f32>(zero_point_words[0]); }
  if (params.zero_point_type == 1u) { return f32(bitcast<i32>(zero_point_words[0])); }
  let byte = word_byte(zero_point_words[0], 0u);
  if (params.zero_point_type == 2u) { return f32(signed_byte(byte)); }
  return f32(byte);
}

fn zero_point_integer_value() -> i32 {
  if (params.zero_point_type == 1u) { return bitcast<i32>(zero_point_words[0]); }
  let byte = word_byte(zero_point_words[0], 0u);
  if (params.zero_point_type == 2u) { return signed_byte(byte); }
  return i32(byte);
}

fn signed_magnitude_i32(value : i32) -> SignedMagnitude {
  let bits = bitcast<u32>(value);
  if (value < 0) { return SignedMagnitude(1u, 0u - bits, 1u); }
  return SignedMagnitude(0u, bits, 1u);
}

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
  if (params.input_type == 0u) { return f32_integral_signed_magnitude(input_words[index]); }
  return signed_magnitude_i32(input_integer_value(index));
}

fn zero_point_signed_magnitude() -> SignedMagnitude {
  if (params.zero_point_type == 0u) { return f32_integral_signed_magnitude(zero_point_words[0]); }
  return signed_magnitude_i32(zero_point_integer_value());
}

// Match the forward/CPU contract: raw typed values subtract before F32
// conversion, including a high integral F32 zero point against an I32 input.
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

@compute @workgroup_size(64)
fn input_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let index = gid.x;
  if (index >= params.size || params.input_type != 0u) { return; }
  grad_input[index] = grad_input[index] + grad_output[index] * scale[0];
}

@compute @workgroup_size(1)
fn scale_main(@builtin(global_invocation_id) gid : vec3<u32>) {
  if (gid.x != 0u) { return; }
  var value = 0.0;
  for (var index = 0u; index < params.size; index = index + 1u) {
    value = value + grad_output[index] * input_minus_zero_point(index);
  }
  grad_scale[0] = grad_scale[0] + value;
}
