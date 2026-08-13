// Portable cooperative W8A8 Conv2D kernel for groups=1. An 8x4 workgroup
// computes four NHW positions by 32 output channels. Each input reduction word
// is shared by eight adjacent output words, avoiding the baseline kernel's
// repeated input unpack for every output channel.
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
  let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) -
    i32(params.pad_top);
  let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) -
    i32(params.pad_left);
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
  let input_y = i32(output_y * params.stride_y + kernel_y * params.dilation_y) -
    i32(params.pad_top);
  let input_x = i32(output_x * params.stride_x + kernel_x * params.dilation_x) -
    i32(params.pad_left);
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
  var result = repeated_byte(params.input_zero_point);
  for (var lane = 0u; lane < 4u; lane = lane + 1u) {
    if (reduction + lane < reduction_size) {
      let shift = lane * 8u;
      result = (result & ~(255u << shift)) |
        (conv_input_byte(spatial, reduction + lane) << shift);
    }
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

fn typed_byte(word : u32, lane : u32, dtype : u32) -> i32 {
  let value = (word >> (lane * 8u)) & 255u;
  if (dtype == 6u && value >= 128u) { return i32(value) - 256; }
  return i32(value);
}

fn centered_dot4(input_word : u32,
                 weight_word0 : u32, weight_word1 : u32,
                 weight_word2 : u32, weight_word3 : u32,
                 weight_zero_point : vec4<i32>) -> vec4<i32> {
  var result = vec4<i32>(0);
  for (var lane = 0u; lane < 4u; lane = lane + 1u) {
    let input_value = typed_byte(input_word, lane, params.input_type) -
      params.input_zero_point;
    let weight_value = vec4<i32>(
      typed_byte(weight_word0, lane, params.weight_type),
      typed_byte(weight_word1, lane, params.weight_type),
      typed_byte(weight_word2, lane, params.weight_type),
      typed_byte(weight_word3, lane, params.weight_type)) - weight_zero_point;
    result = result + vec4<i32>(input_value) * weight_value;
  }
  return result;
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

fn requantize(accumulator : i32, channel : u32) -> u32 {
  let multiplier = (params.input_scale * weight_scales[channel]) / params.output_scale;
  let transformed = f32(accumulator) * multiplier + f32(params.output_zero_point);
  let minimum = select(0, -128, params.output_type == 6u);
  let maximum = select(255, 127, params.output_type == 6u);
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
  let output_channel = word_in_spatial * 4u;
  let valid_word = spatial < spatial_count && word_in_spatial < words_per_spatial;
  let reduction_size = params.kernel_height * params.kernel_width * params.input_channels;
  var accumulator = vec4<i32>(0);
  var weight_zero_point = vec4<i32>(0);
  if (valid_word) {
    accumulator = vec4<i32>(
      bias_values[output_channel],
      bias_values[output_channel + 1u],
      bias_values[output_channel + 2u],
      bias_values[output_channel + 3u]);
    weight_zero_point = vec4<i32>(
      weight_zero_points[output_channel],
      weight_zero_points[output_channel + 1u],
      weight_zero_points[output_channel + 2u],
      weight_zero_points[output_channel + 3u]);
  }
  for (var reduction_base = 0u; reduction_base < reduction_size;
       reduction_base = reduction_base + 32u) {
    input_tile[local_id.y * 8u + local_id.x] =
      conv_input_word(spatial, reduction_base + local_id.x * 4u);
    workgroupBarrier();
    if (valid_word) {
      for (var reduction_word = 0u; reduction_word < 8u;
           reduction_word = reduction_word + 1u) {
        let reduction = reduction_base + reduction_word * 4u;
        if (reduction >= reduction_size) { continue; }
        let count = min(4u, reduction_size - reduction);
        let input_word = input_tile[local_id.y * 8u + reduction_word];
        let weight_base0 = output_channel * reduction_size + reduction;
        let weight_base1 = weight_base0 + reduction_size;
        let weight_base2 = weight_base1 + reduction_size;
        let weight_base3 = weight_base2 + reduction_size;
        accumulator = accumulator + centered_dot4(
          input_word,
          weight_word_padded(weight_base0, count, weight_zero_point.x),
          weight_word_padded(weight_base1, count, weight_zero_point.y),
          weight_word_padded(weight_base2, count, weight_zero_point.z),
          weight_word_padded(weight_base3, count, weight_zero_point.w),
          weight_zero_point);
      }
    }
    workgroupBarrier();
  }
  if (valid_word) {
    var packed = requantize(accumulator.x, output_channel);
    packed = packed | (requantize(accumulator.y, output_channel + 1u) << 8u);
    packed = packed | (requantize(accumulator.z, output_channel + 2u) << 16u);
    packed = packed | (requantize(accumulator.w, output_channel + 3u) << 24u);
    output_words[spatial * words_per_spatial + word_in_spatial] = packed;
  }
}
