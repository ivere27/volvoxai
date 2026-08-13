// Physical-byte ONNX MatMul with right-aligned broadcast batch axes.
// Every invocation owns one packed u32 output word, avoiding byte-write races.
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

fn signed_or_unsigned(byte : u32, dtype : u32) -> i32 {
  if (dtype == 6u && byte >= 128u) { return i32(byte) - 256; }
  return i32(byte);
}

fn a_value(index : u32) -> i32 {
  return signed_or_unsigned(
    word_byte(a_words[index / 4u], index), params.a_type);
}

fn b_value(index : u32) -> i32 {
  return signed_or_unsigned(
    word_byte(b_words[index / 4u], index), params.b_type);
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
  var accumulator = 0;
  for (var inner = 0u; inner < params.k; inner = inner + 1u) {
    let a_element = a_value(a_base + row * params.k + inner);
    let b_element = b_value(b_base + inner * params.n + column);
    accumulator = accumulator +
      (a_element - params.a_zero_point) *
      (b_element - params.b_zero_point);
  }
  return requantize(accumulator);
}

// The portable path is also the OpenGL fallback, where no signed integer dot
// product extension is guaranteed.  One storage invocation still owns one
// packed output word, but the common four-adjacent-column case shares the
// broadcast indexing and A loads across all four results.  This preserves the
// arbitrary-N contract: packed words that straddle a matrix row use the scalar
// path below.
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

  var accumulator0 = 0;
  var accumulator1 = 0;
  var accumulator2 = 0;
  var accumulator3 = 0;
  for (var inner = 0u; inner < params.k; inner = inner + 1u) {
    let a_element = a_value(a_base + row * params.k + inner) -
      params.a_zero_point;
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
