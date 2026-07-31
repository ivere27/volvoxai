// Canonical byte-domain QMaskedMean for hard routing and sequence pooling. The
// I32 mask keeps a token when nonzero. One invocation owns one packed output word
// (up to four [B,D] lanes), so byte-output lanes never race.
//
// Fixed 48-byte ABI shared with native GPU backends:
//   u32 batch, sequence, width, input_type,
//       output_type, pad0, pad1, pad2;
//   i32 input_zero_point, output_zero_point;
//   f32 input_scale, output_scale.
// input_type/output_type use canonical protobuf values I8=6 and U8=5.

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> keep_mask : array<i32>;
@group(0) @binding(2) var<storage, read_write> output_words : array<u32>;

struct Params {
  batch : u32,
  sequence : u32,
  width : u32,
  input_type : u32,
  output_type : u32,
  _pad0 : u32,
  _pad1 : u32,
  _pad2 : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  input_scale : f32,
  output_scale : f32,
}
@group(0) @binding(3) var<uniform> params : Params;

fn word_byte(word : u32, index : u32) -> u32 {
  return (word >> ((index % 4u) * 8u)) & 255u;
}

fn input_value(index : u32) -> i32 {
  let byte = word_byte(input_words[index / 4u], index);
  if (params.input_type == 6u && byte >= 128u) { return i32(byte) - 256; }
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

fn qmaskedmean_one(output_index : u32) -> u32 {
  let output_elements = params.batch * params.width;
  if (output_index >= output_elements) { return 0u; }
  let batch_index = output_index / params.width;
  let column = output_index % params.width;
  let mask_base = batch_index * params.sequence;
  let input_base = mask_base * params.width;
  var keep_count : i32 = 0;
  var centered_sum : i32 = 0;
  for (var token = 0u; token < params.sequence; token = token + 1u) {
    if (keep_mask[mask_base + token] != 0) {
      keep_count = keep_count + 1;
      centered_sum = centered_sum + input_value(input_base + token * params.width + column) -
        params.input_zero_point;
    }
  }
  if (keep_count == 0) { return output_byte(params.output_zero_point); }
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  let mean = f32(centered_sum) / f32(keep_count);
  let multiplier = params.input_scale / params.output_scale;
  // The bitcast round-trip makes the product's F32 bits observable before the
  // addition, preventing a backend from contracting this into an FMA. It
  // matches the portable C/WASM and JS reference rounding schedule.
  let scaled = bitcast<f32>(bitcast<u32>(mean * multiplier));
  let transformed = scaled + f32(params.output_zero_point);
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let output_word = gid.x;
  let first_output = output_word * 4u;
  let output_elements = params.batch * params.width;
  if (first_output >= output_elements) { return; }
  var packed = qmaskedmean_one(first_output);
  packed = packed | (qmaskedmean_one(first_output + 1u) << 8u);
  packed = packed | (qmaskedmean_one(first_output + 2u) << 16u);
  packed = packed | (qmaskedmean_one(first_output + 3u) << 24u);
  output_words[output_word] = packed;
}
