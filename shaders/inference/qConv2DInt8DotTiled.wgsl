// @volvoxai-native-spv-only
requires packed_4x8_integer_dot_product;

// Cooperative W8A8 Conv2D DP4a kernel for groups=1. An 8x4 workgroup computes
// four NHW positions by 32 output channels and shares a 32-byte reduction tile.
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

var<workgroup> input_tile : array<u32, 32>;
var<workgroup> weight_tile : array<u32, 256>;

fn raw_input_byte(index : u32) -> u32 {
  return (input_words[index / 4u] >> ((index % 4u) * 8u)) & 255u;
}
fn raw_weight_byte(index : u32) -> u32 {
  return (weight_words[index / 4u] >> ((index % 4u) * 8u)) & 255u;
}
fn repeated_byte(value : i32) -> u32 {
  let byte = bitcast<u32>(value) & 255u;
  return byte | (byte << 8u) | (byte << 16u) | (byte << 24u);
}
fn conv_input_byte(spatial : u32, reduction : u32) -> u32 {
  let spatial_count = params.batch * params.output_height * params.output_width;
  let reduction_size = params.kernel_height * params.kernel_width * params.input_channels;
  if (spatial >= spatial_count || reduction >= reduction_size) {
    return bitcast<u32>(params.input_zero_point) & 255u;
  }
  let image_elements = params.output_height * params.output_width;
  let batch_index = spatial / image_elements;
  let image_offset = spatial % image_elements;
  let output_y = image_offset / params.output_width;
  let output_x = image_offset % params.output_width;
  let kernel_pixel = reduction / params.input_channels;
  let input_channel = reduction % params.input_channels;
  let kernel_y = kernel_pixel / params.kernel_width;
  let kernel_x = kernel_pixel % params.kernel_width;
  let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) - i32(params.pad_top);
  let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) - i32(params.pad_left);
  if (input_y < 0 || input_y >= i32(params.input_height) ||
      input_x < 0 || input_x >= i32(params.input_width)) {
    return bitcast<u32>(params.input_zero_point) & 255u;
  }
  let index = (((batch_index * params.input_height + u32(input_y)) * params.input_width +
    u32(input_x)) * params.input_channels) + input_channel;
  return raw_input_byte(index);
}
fn conv_input_word(spatial : u32, reduction : u32) -> u32 {
  let spatial_count = params.batch * params.output_height * params.output_width;
  let reduction_size = params.kernel_height * params.kernel_width * params.input_channels;
  if (spatial >= spatial_count || reduction >= reduction_size) {
    return repeated_byte(params.input_zero_point);
  }
  let image_elements = params.output_height * params.output_width;
  let batch_index = spatial / image_elements;
  let image_offset = spatial % image_elements;
  let output_y = image_offset / params.output_width;
  let output_x = image_offset % params.output_width;
  let kernel_pixel = reduction / params.input_channels;
  let input_channel = reduction % params.input_channels;
  let kernel_y = kernel_pixel / params.kernel_width;
  let kernel_x = kernel_pixel % params.kernel_width;
  let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) - i32(params.pad_top);
  let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) - i32(params.pad_left);
  if (input_channel + 4u <= params.input_channels &&
      input_y >= 0 && input_y < i32(params.input_height) &&
      input_x >= 0 && input_x < i32(params.input_width)) {
    let index = (((batch_index * params.input_height + u32(input_y)) * params.input_width +
      u32(input_x)) * params.input_channels) + input_channel;
    let word_index = index / 4u;
    let byte_offset = index % 4u;
    if (byte_offset == 0u) { return input_words[word_index]; }
    let shift = byte_offset * 8u;
    return (input_words[word_index] >> shift) |
      (input_words[word_index + 1u] << (32u - shift));
  }
  var result = 0u;
  for (var lane = 0u; lane < 4u; lane = lane + 1u) {
    result = result | (conv_input_byte(spatial, reduction + lane) << (lane * 8u));
  }
  return result;
}
fn weight_word_padded(base : u32, count : u32, zero_point : i32) -> u32 {
  if (count >= 4u) {
    let word_index = base / 4u;
    let byte_offset = base % 4u;
    if (byte_offset == 0u) { return weight_words[word_index]; }
    let shift = byte_offset * 8u;
    return (weight_words[word_index] >> shift) |
      (weight_words[word_index + 1u] << (32u - shift));
  }
  var result = repeated_byte(zero_point);
  for (var lane = 0u; lane < count; lane = lane + 1u) {
    let shift = lane * 8u;
    result = (result & ~(255u << shift)) | (raw_weight_byte(base + lane) << shift);
  }
  return result;
}
fn signed_word(word : u32, dtype : u32) -> u32 {
  if (dtype == 3u) { return word ^ 0x80808080u; }
  return word;
}
fn centered_zero_point(zero_point : i32, dtype : u32) -> i32 {
  if (dtype == 3u) { return zero_point - 128; }
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
fn output_byte(value : i32) -> u32 { return bitcast<u32>(value) & 255u; }
fn requantize(accumulator : i32, channel : u32) -> u32 {
  let multiplier = (params.input_scale * weight_scales[channel]) / params.output_scale;
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

@compute @workgroup_size(8, 4, 1)
fn main(@builtin(workgroup_id) group_id : vec3<u32>,
        @builtin(local_invocation_id) local_id : vec3<u32>) {
  let spatial = group_id.y * 4u + local_id.y;
  let spatial_count = params.batch * params.output_height * params.output_width;
  let words_per_spatial = params.output_channels / 4u;
  let word_in_spatial = group_id.x * 8u + local_id.x;
  let output_channel = group_id.x * 32u + local_id.x * 4u;
  let valid_word = spatial < spatial_count && word_in_spatial < words_per_spatial;
  let reduction_size = params.kernel_height * params.kernel_width * params.input_channels;
  var accumulator0 = 0;
  var accumulator1 = 0;
  var accumulator2 = 0;
  var accumulator3 = 0;
  var weight_zero0 = 0;
  var weight_zero1 = 0;
  var weight_zero2 = 0;
  var weight_zero3 = 0;
  if (valid_word) {
    accumulator0 = bias_values[output_channel];
    accumulator1 = bias_values[output_channel + 1u];
    accumulator2 = bias_values[output_channel + 2u];
    accumulator3 = bias_values[output_channel + 3u];
    weight_zero0 = weight_zero_points[output_channel];
    weight_zero1 = weight_zero_points[output_channel + 1u];
    weight_zero2 = weight_zero_points[output_channel + 2u];
    weight_zero3 = weight_zero_points[output_channel + 3u];
  }
  let local_linear = local_id.y * 8u + local_id.x;
  let channel_base = group_id.x * 32u;
  for (var reduction_base = 0u; reduction_base < reduction_size; reduction_base = reduction_base + 32u) {
    input_tile[local_id.y * 8u + local_id.x] =
      conv_input_word(spatial, reduction_base + local_id.x * 4u);
    for (var slot = local_linear; slot < 256u; slot = slot + 32u) {
      let tile_channel = slot / 8u;
      let reduction_word = slot % 8u;
      let channel = channel_base + tile_channel;
      let reduction = reduction_base + reduction_word * 4u;
      var count = 0u;
      var weight_zero = 0;
      if (channel < params.output_channels) {
        weight_zero = weight_zero_points[channel];
        if (reduction < reduction_size) { count = min(4u, reduction_size - reduction); }
      }
      weight_tile[reduction_word * 32u + tile_channel] = weight_word_padded(
        channel * reduction_size + reduction, count, weight_zero);
    }
    workgroupBarrier();
    for (var reduction_word = 0u; reduction_word < 8u; reduction_word = reduction_word + 1u) {
      let input_word_raw = input_tile[local_id.y * 8u + reduction_word];
      let weight_base = reduction_word * 32u + local_id.x * 4u;
      accumulator0 = accumulator0 + corrected_dot(input_word_raw, weight_tile[weight_base], weight_zero0);
      accumulator1 = accumulator1 + corrected_dot(input_word_raw, weight_tile[weight_base + 1u], weight_zero1);
      accumulator2 = accumulator2 + corrected_dot(input_word_raw, weight_tile[weight_base + 2u], weight_zero2);
      accumulator3 = accumulator3 + corrected_dot(input_word_raw, weight_tile[weight_base + 3u], weight_zero3);
    }
    workgroupBarrier();
  }
  if (valid_word) {
    var packed = requantize(accumulator0, output_channel);
    packed = packed | (requantize(accumulator1, output_channel + 1u) << 8u);
    packed = packed | (requantize(accumulator2, output_channel + 2u) << 16u);
    packed = packed | (requantize(accumulator3, output_channel + 3u) << 24u);
    output_words[spatial * words_per_spatial + word_in_spatial] = packed;
  }
}
