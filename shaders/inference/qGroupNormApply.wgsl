// QGroupNorm apply pass. The paired statistics pass owns reduction; this pass
// owns complete packed destination words, eliminating byte-lane races even
// when a channel group boundary is not divisible by four.
//
// Params is the fixed 64-byte cross-backend ABI shared with qGroupNormStats.
struct Params {
  batch : u32,
  height : u32,
  width : u32,
  channels : u32,
  groups : u32,
  input_type : u32,
  output_type : u32,
  _pad0 : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad1 : i32,
  _pad2 : i32,
  input_scale : f32,
  output_scale : f32,
  epsilon : f32,
  _pad3 : f32,
}

@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> gamma : array<f32>;
@group(0) @binding(2) var<storage, read> beta : array<f32>;
// Two F32 values per flat [batch, group]: mean_raw and inverse standard
// deviation. It is reusable scratch allocated once for the graph execution.
@group(0) @binding(3) var<storage, read> group_stats : array<f32>;
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

fn qgroupnorm_one(index : u32) -> u32 {
  let elements = params.batch * params.height * params.width * params.channels;
  if (index >= elements) { return 0u; }
  let area = params.height * params.width;
  let sample_stride = area * params.channels;
  let batch_index = index / sample_stride;
  let sample_index = index % sample_stride;
  let channel = sample_index % params.channels;
  let channels_per_group = params.channels / params.groups;
  let group = channel / channels_per_group;
  let stats_offset = (batch_index * params.groups + group) * 2u;
  let raw = f32(input_value(index) - params.input_zero_point);
  let normalized = (raw - group_stats[stats_offset]) * params.input_scale *
    group_stats[stats_offset + 1u];
  let value = normalized * gamma[channel] + beta[channel];
  let transformed = value / params.output_scale + f32(params.output_zero_point);
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
  if (transformed != transformed) { return output_byte(params.output_zero_point); }
  if (transformed <= f32(minimum)) { return output_byte(minimum); }
  if (transformed >= f32(maximum)) { return output_byte(maximum); }
  return output_byte(round_even(transformed));
}

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) global_invocation_id : vec3<u32>) {
  let output_word = global_invocation_id.x;
  let first_element = output_word * 4u;
  let elements = params.batch * params.height * params.width * params.channels;
  if (first_element >= elements) { return; }
  var packed = qgroupnorm_one(first_element);
  packed = packed | (qgroupnorm_one(first_element + 1u) << 8u);
  packed = packed | (qgroupnorm_one(first_element + 2u) << 16u);
  packed = packed | (qgroupnorm_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
