// Typed runtime Cast over the graph's four portable storage dtypes.  Every
// tensor buffer is viewed as raw u32 words so F32/I32 elements and packed
// I8/U8 bytes can share one authoritative shader without aliasing assumptions.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read_write> output_words : array<u32>;

struct Params {
  size : u32,
  input_type : u32,
  output_type : u32,
  _pad : u32,
}
@group(0) @binding(2) var<uniform> params : Params;

// Runtime dtype IDs are F32=0, I32=1, I8=2, U8=3.
fn input_byte(index : u32) -> u32 {
  let word = input_words[index / 4u];
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn signed_byte(value : u32) -> i32 {
  if (value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

// ECMAScript integer typed-array assignment truncates toward zero, then keeps
// the low 32 bits (with NaN and infinities becoming zero).  Performing that
// modulo arithmetic in f32 loses the low bits of large finite values, so
// decode the source IEEE-754 word instead.  The significand is only 24 bits;
// shifts of 32 or more contribute no low bits.
fn f32_integer_bits(bits : u32) -> u32 {
  let exponent = (bits >> 23u) & 255u;
  if (exponent == 255u || exponent < 127u) { return 0u; }

  let unbiased_exponent = exponent - 127u;
  let significand = (bits & 0x007fffffu) | 0x00800000u;
  var magnitude = 0u;
  if (unbiased_exponent <= 23u) {
    magnitude = significand >> (23u - unbiased_exponent);
  } else if (unbiased_exponent < 55u) {
    magnitude = significand << (unbiased_exponent - 23u);
  }

  if ((bits & 0x80000000u) != 0u) { return 0u - magnitude; }
  return magnitude;
}

fn cast_f32(bits : u32) -> u32 {
  if (params.output_type == 0u) { return bits; }
  return f32_integer_bits(bits);
}

fn cast_i32(bits : u32) -> u32 {
  if (params.output_type == 0u) { return bitcast<u32>(f32(bitcast<i32>(bits))); }
  return bits;
}

fn cast_i8(byte : u32) -> u32 {
  if (params.output_type == 0u) { return bitcast<u32>(f32(signed_byte(byte))); }
  if (params.output_type == 1u) { return bitcast<u32>(signed_byte(byte)); }
  return byte;
}

fn cast_u8(byte : u32) -> u32 {
  if (params.output_type == 0u) { return bitcast<u32>(f32(byte)); }
  if (params.output_type == 1u) { return byte; }
  return byte;
}

fn cast_one(index : u32) -> u32 {
  if (params.input_type == 0u) { return cast_f32(input_words[index]); }
  if (params.input_type == 1u) { return cast_i32(input_words[index]); }
  let byte = input_byte(index);
  if (params.input_type == 2u) { return cast_i8(byte); }
  return cast_u8(byte);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  if (params.output_type < 2u) {
    if (output_word >= params.size) { return; }
    output_words[output_word] = cast_one(output_word);
    return;
  }

  let first_element = output_word * 4u;
  if (first_element >= params.size) { return; }
  var packed = 0u;
  for (var lane = 0u; lane < 4u; lane = lane + 1u) {
    let element = first_element + lane;
    if (element >= params.size) { break; }
    packed = packed | ((cast_one(element) & 255u) << (lane * 8u));
  }
  output_words[output_word] = packed;
}
