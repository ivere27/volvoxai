// Canonical generic W8A8 Conv2D: NHWC activations, OHWI weights, I32
// accumulation/bias, and packed I8/U8 output. One invocation owns one output
// word; this baseline remains portable to WebGPU and generated native shaders.
@group(0) @binding(0) var<storage, read> input_words : array<u32>;
@group(0) @binding(1) var<storage, read> weight_words : array<u32>;
@group(0) @binding(2) var<storage, read> weight_scales : array<f32>;
@group(0) @binding(3) var<storage, read> weight_zero_points : array<i32>;
@group(0) @binding(4) var<storage, read> bias_values : array<i32>;
@group(0) @binding(5) var<storage, read_write> output_words : array<u32>;

struct Params {
  batch : u32,
  input_height : u32,
  input_width : u32,
  input_channels : u32,
  output_height : u32,
  output_width : u32,
  output_channels : u32,
  kernel_height : u32,
  kernel_width : u32,
  stride_y : u32,
  stride_x : u32,
  dilation_y : u32,
  dilation_x : u32,
  pad_top : u32,
  pad_left : u32,
  groups : u32,
  input_type : u32,
  weight_type : u32,
  output_type : u32,
  relu : u32,
  input_zero_point : i32,
  output_zero_point : i32,
  _pad0 : i32,
  _pad1 : i32,
  input_scale : f32,
  output_scale : f32,
  _pad2 : f32,
  _pad3 : f32,
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
  if (dtype == 2u) { return signed_byte(byte); }
  return i32(byte);
}

fn input_value(index : u32) -> i32 {
  return typed_byte(word_byte(input_words[index / 4u], index), params.input_type);
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

fn qconv_one(index : u32) -> u32 {
  let elements = params.batch * params.output_height * params.output_width * params.output_channels;
  if (index >= elements) { return 0u; }
  let image_elements = params.output_height * params.output_width * params.output_channels;
  let batch_index = index / image_elements;
  let image_offset = index % image_elements;
  let output_y = image_offset / (params.output_width * params.output_channels);
  let output_x = (image_offset / params.output_channels) % params.output_width;
  let output_channel = image_offset % params.output_channels;
  let group_output_channels = params.output_channels / params.groups;
  let input_per_group = params.input_channels / params.groups;
  let group = output_channel / group_output_channels;
  var accumulator = bias_values[output_channel];
  for (var kernel_y = 0u; kernel_y < params.kernel_height; kernel_y = kernel_y + 1u) {
    let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) - i32(params.pad_top);
    if (input_y < 0 || input_y >= i32(params.input_height)) { continue; }
    for (var kernel_x = 0u; kernel_x < params.kernel_width; kernel_x = kernel_x + 1u) {
      let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) - i32(params.pad_left);
      if (input_x < 0 || input_x >= i32(params.input_width)) { continue; }
      for (var local_channel = 0u; local_channel < input_per_group; local_channel = local_channel + 1u) {
        let input_channel = group * input_per_group + local_channel;
        let input_index = (((batch_index * params.input_height + u32(input_y)) * params.input_width +
          u32(input_x)) * params.input_channels) + input_channel;
        let weight_index = (((output_channel * params.kernel_height + kernel_y) * params.kernel_width +
          kernel_x) * input_per_group) + local_channel;
        accumulator = accumulator +
          (input_value(input_index) - params.input_zero_point) *
          (weight_value(weight_index) - weight_zero_points[output_channel]);
      }
    }
  }
  let multiplier = (params.input_scale * weight_scales[output_channel]) / params.output_scale;
  let transformed = f32(accumulator) * multiplier + f32(params.output_zero_point);
  let minimum = select(0, -128, params.output_type == 2u);
  let maximum = select(255, 127, params.output_type == 2u);
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
  let elements = params.batch * params.output_height * params.output_width * params.output_channels;
  if (first_element >= elements) { return; }
  var packed = qconv_one(first_element);
  packed = packed | (qconv_one(first_element + 1u) << 8u);
  packed = packed | (qconv_one(first_element + 2u) << 16u);
  packed = packed | (qconv_one(first_element + 3u) << 24u);
  output_words[output_word] = packed;
}
