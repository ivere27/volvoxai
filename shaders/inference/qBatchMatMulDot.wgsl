// @volvoxai-native-spv-only
requires packed_4x8_integer_dot_product;

// Packed-dot physical-byte ONNX MatMul with right-aligned broadcast batch
// axes.  The common four-adjacent-column path shares A and B storage loads;
// arbitrary K is handled by DP4a chunks followed by a scalar tail.
@group(0) @binding(0) var<storage, read> a_words : array<u32>;
@group(0) @binding(1) var<storage, read> b_words : array<u32>;
@group(0) @binding(2) var<storage, read_write> output_words : array<u32>;
@group(0) @binding(3) var<storage, read> metadata : array<u32>;

struct Params {
  batch_rank : u32,
  m : u32,
  k : u32,
  n : u32,
  output_elements : u32,
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
  _pad1 : f32,
}
@group(0) @binding(4) var<uniform> params : Params;

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

fn a_raw_byte(index : u32) -> u32 {
  return word_byte(a_words[index / 4u], index);
}

fn b_raw_byte(index : u32) -> u32 {
  return word_byte(b_words[index / 4u], index);
}

fn a_value(index : u32) -> i32 {
  return typed_byte(a_raw_byte(index), params.a_type);
}

fn b_value(index : u32) -> i32 {
  return typed_byte(b_raw_byte(index), params.b_type);
}

fn a_word(base : u32) -> u32 {
  let word_index = base / 4u;
  let byte_offset = base % 4u;
  if (byte_offset == 0u) { return a_words[word_index]; }
  let shift = byte_offset * 8u;
  return (a_words[word_index] >> shift) |
    (a_words[word_index + 1u] << (32u - shift));
}

// Callers only use this helper when four bytes remain in the logical B row.
fn b_row_word(base : u32) -> u32 {
  let word_index = base / 4u;
  let byte_offset = base % 4u;
  if (byte_offset == 0u) { return b_words[word_index]; }
  let shift = byte_offset * 8u;
  return (b_words[word_index] >> shift) |
    (b_words[word_index + 1u] << (32u - shift));
}

fn b_column_word(base : u32, column : u32, inner : u32) -> u32 {
  let offset0 = base + inner * params.n + column;
  let offset1 = offset0 + params.n;
  let offset2 = offset1 + params.n;
  let offset3 = offset2 + params.n;
  return b_raw_byte(offset0) |
    (b_raw_byte(offset1) << 8u) |
    (b_raw_byte(offset2) << 16u) |
    (b_raw_byte(offset3) << 24u);
}

fn signed_word(word : u32, dtype : u32) -> u32 {
  if (dtype == 5u) { return word ^ 0x80808080u; }
  return word;
}

fn centered_zero_point(zero_point : i32, dtype : u32) -> i32 {
  if (dtype == 5u) { return zero_point - 128; }
  return zero_point;
}

fn corrected_dot_signed(a_word_signed : u32, b_word_raw : u32,
                        a_sum : i32, a_zero : i32, b_zero : i32) -> i32 {
  let b_word_signed = signed_word(b_word_raw, params.b_type);
  var value = dot4I8Packed(a_word_signed, b_word_signed);
  if (b_zero != 0) {
    value = value - b_zero * a_sum;
  }
  if (a_zero != 0) {
    value = value - a_zero * dot4I8Packed(b_word_signed, 0x01010101u);
  }
  return value + 4 * a_zero * b_zero;
}

fn corrected_dot(a_word_raw : u32, b_word_raw : u32) -> i32 {
  let a_word_signed = signed_word(a_word_raw, params.a_type);
  let a_zero = centered_zero_point(params.a_zero_point, params.a_type);
  let b_zero = centered_zero_point(params.b_zero_point, params.b_type);
  var a_sum = 0;
  if (b_zero != 0) {
    a_sum = dot4I8Packed(a_word_signed, 0x01010101u);
  }
  return corrected_dot_signed(
    a_word_signed, b_word_raw, a_sum, a_zero, b_zero);
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

fn requantize(accumulator : i32) -> u32 {
  let multiplier = (params.a_scale * params.b_scale) / params.output_scale;
  let transformed = f32(accumulator) * multiplier +
    f32(params.output_zero_point);
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  if (transformed != transformed) {
    return output_byte(params.output_zero_point);
  }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

fn matmul_one(index : u32) -> u32 {
  if (index >= params.output_elements) { return 0u; }
  let matrix_elements = params.m * params.n;
  let batch = index / matrix_elements;
  let matrix_index = index % matrix_elements;
  let row = matrix_index / params.n;
  let column = matrix_index % params.n;
  var remaining = batch;
  var a_base = 0u;
  var b_base = 0u;
  for (var axis = 0u; axis < params.batch_rank; axis = axis + 1u) {
    let output_stride = metadata[axis];
    let coordinate = remaining / output_stride;
    remaining = remaining % output_stride;
    a_base = a_base + coordinate * metadata[params.batch_rank + axis];
    b_base = b_base + coordinate * metadata[2u * params.batch_rank + axis];
  }
  let a_row = a_base + row * params.k;
  var accumulator = 0;
  var inner = 0u;
  for (; inner + 4u <= params.k; inner = inner + 4u) {
    accumulator = accumulator + corrected_dot(
      a_word(a_row + inner), b_column_word(b_base, column, inner));
  }
  for (; inner < params.k; inner = inner + 1u) {
    accumulator = accumulator +
      (a_value(a_row + inner) - params.a_zero_point) *
      (b_value(b_base + inner * params.n + column) - params.b_zero_point);
  }
  return requantize(accumulator);
}

fn matmul_word(first : u32) -> u32 {
  let matrix_elements = params.m * params.n;
  let matrix_index = first % matrix_elements;
  let column = matrix_index % params.n;
  if (first + 3u >= params.output_elements || column + 3u >= params.n) {
    var packed = matmul_one(first);
    packed = packed | (matmul_one(first + 1u) << 8u);
    packed = packed | (matmul_one(first + 2u) << 16u);
    packed = packed | (matmul_one(first + 3u) << 24u);
    return packed;
  }

  let batch = first / matrix_elements;
  let row = matrix_index / params.n;
  var remaining = batch;
  var a_base = 0u;
  var b_base = 0u;
  for (var axis = 0u; axis < params.batch_rank; axis = axis + 1u) {
    let output_stride = metadata[axis];
    let coordinate = remaining / output_stride;
    remaining = remaining % output_stride;
    a_base = a_base + coordinate * metadata[params.batch_rank + axis];
    b_base = b_base + coordinate * metadata[2u * params.batch_rank + axis];
  }

  let a_row = a_base + row * params.k;
  var accumulator0 = 0;
  var accumulator1 = 0;
  var accumulator2 = 0;
  var accumulator3 = 0;
  let a_zero = centered_zero_point(params.a_zero_point, params.a_type);
  let b_zero = centered_zero_point(params.b_zero_point, params.b_type);
  var inner = 0u;
  for (; inner + 4u <= params.k; inner = inner + 4u) {
    let a_packed = a_word(a_row + inner);
    let a_signed = signed_word(a_packed, params.a_type);
    var a_sum = 0;
    if (b_zero != 0) {
      a_sum = dot4I8Packed(a_signed, 0x01010101u);
    }
    let b_row0 = b_row_word(b_base + inner * params.n + column);
    let b_row1 = b_row_word(b_base + (inner + 1u) * params.n + column);
    let b_row2 = b_row_word(b_base + (inner + 2u) * params.n + column);
    let b_row3 = b_row_word(b_base + (inner + 3u) * params.n + column);
    let b_column0 = word_byte(b_row0, 0u) |
      (word_byte(b_row1, 0u) << 8u) |
      (word_byte(b_row2, 0u) << 16u) |
      (word_byte(b_row3, 0u) << 24u);
    let b_column1 = word_byte(b_row0, 1u) |
      (word_byte(b_row1, 1u) << 8u) |
      (word_byte(b_row2, 1u) << 16u) |
      (word_byte(b_row3, 1u) << 24u);
    let b_column2 = word_byte(b_row0, 2u) |
      (word_byte(b_row1, 2u) << 8u) |
      (word_byte(b_row2, 2u) << 16u) |
      (word_byte(b_row3, 2u) << 24u);
    let b_column3 = word_byte(b_row0, 3u) |
      (word_byte(b_row1, 3u) << 8u) |
      (word_byte(b_row2, 3u) << 16u) |
      (word_byte(b_row3, 3u) << 24u);
    accumulator0 = accumulator0 + corrected_dot_signed(
      a_signed, b_column0, a_sum, a_zero, b_zero);
    accumulator1 = accumulator1 + corrected_dot_signed(
      a_signed, b_column1, a_sum, a_zero, b_zero);
    accumulator2 = accumulator2 + corrected_dot_signed(
      a_signed, b_column2, a_sum, a_zero, b_zero);
    accumulator3 = accumulator3 + corrected_dot_signed(
      a_signed, b_column3, a_sum, a_zero, b_zero);
  }
  for (; inner < params.k; inner = inner + 1u) {
    let a_element = a_value(a_row + inner) - params.a_zero_point;
    let b_offset = b_base + inner * params.n + column;
    accumulator0 = accumulator0 + a_element *
      (b_value(b_offset) - params.b_zero_point);
    accumulator1 = accumulator1 + a_element *
      (b_value(b_offset + 1u) - params.b_zero_point);
    accumulator2 = accumulator2 + a_element *
      (b_value(b_offset + 2u) - params.b_zero_point);
    accumulator3 = accumulator3 + a_element *
      (b_value(b_offset + 3u) - params.b_zero_point);
  }

  return requantize(accumulator0) |
    (requantize(accumulator1) << 8u) |
    (requantize(accumulator2) << 16u) |
    (requantize(accumulator3) << 24u);
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let first = gid.x * 4u;
  if (first >= params.output_elements) { return; }
  output_words[gid.x] = matmul_word(first);
}
