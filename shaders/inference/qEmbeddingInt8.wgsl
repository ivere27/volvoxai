// Canonical W8A8 embedding gather. Token IDs remain conventional I32 values;
// the [vocab,hidden] table is byte-packed and has a scale/zero point per row.
// The result is immediately requantized into a byte-packed per-tensor output.
// GraphExecutor preflights every graph-input ID before dispatch, so the invalid
// ID fallback here is defensive and cannot write a partial logical output.
@group(0) @binding(0) var<storage, read> token_ids : array<i32>;
@group(0) @binding(1) var<storage, read> weight_words : array<u32>;
@group(0) @binding(2) var<storage, read> weight_scales : array<f32>;
@group(0) @binding(3) var<storage, read> weight_zero_points : array<i32>;
@group(0) @binding(4) var<storage, read_write> output_words : array<u32>;

// `token_offset` is where this dispatch starts reading `token_ids`, and it is
// what lets a decode row run here at all. Every other activation binding can be
// a window into its tensor, because a row's byte offset is a multiple of the
// device's storage alignment whenever the row stride is. A token row is one i32
// -- four bytes -- so no row past the first ever satisfies a 256-byte alignment,
// and the ids window could never resolve. Binding the ids whole and naming the
// first token as a scalar moves that address out of the descriptor, where the
// alignment rule does not reach.
//
// It occupies the padding word the 32-byte uniform already carried, so the ABI
// shared with Vulkan, OpenGL, Metal and CUDA keeps its size and every existing
// writer, which left that word zero, keeps meaning "start at token 0".
struct Params {
  tokens : u32,
  vocab : u32,
  hidden : u32,
  weight_type : u32,
  output_type : u32,
  output_zero_point : i32,
  output_scale : f32,
  token_offset : u32,
}
@group(0) @binding(5) var<uniform> params : Params;

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

fn qembedding_one(index : u32) -> u32 {
  let elements = params.tokens * params.hidden;
  if (index >= elements) { return 0u; }
  let token_index = index / params.hidden;
  let hidden_index = index % params.hidden;
  let signed_id = token_ids[params.token_offset + token_index];
  if (signed_id < 0 || u32(signed_id) >= params.vocab) {
    return output_byte(params.output_zero_point);
  }
  let row = u32(signed_id);
  let centered = f32(weight_value(row * params.hidden + hidden_index) - weight_zero_points[row]);
  let dequantized = centered * weight_scales[row];
  let scaled = dequantized / params.output_scale;
  let transformed = scaled + f32(params.output_zero_point);
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_element = output_word * 4u;
  let elements = params.tokens * params.hidden;
  if (first_element >= elements) { return; }
  var packed = qembedding_one(first_element);
  packed = packed | (qembedding_one(first_element + 1u) << 8u);
  packed = packed | (qembedding_one(first_element + 2u) << 16u);
  packed = packed | (qembedding_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
